"""V126: complete-well sequence encoder for constrained tail coefficients.

The network never predicts absolute TVT. It consumes legal full-well sequences
and predicts the six V123 coefficient targets (datum 1, affine 2, piecewise 3).
Fold1+2 train, 24 Fold3 wells select epoch/consumption, and 130 Fold3 wells
confirm once. Every correction basis is exactly zero through 1000 ft.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "final_consensus_sequence_coefficient_v126"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
BASE_PATH = OOF_ROOT / "rawdata_lattice_consensus_center_fold2_replay_v115" / "fold123_consensus_center_oof.parquet"
TARGET_PATH = OOF_ROOT / "final_consensus_residual_oracle_v123" / "oracle_bywell.csv"
TRAIN_FOLDS = (1, 2)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
NODES = 192
SEED = 20260801
EPOCHS = 240
PATIENCE = 35
BATCH_SIZE = 32
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 2e-3
FREEZE_FT = 1000.0
RAMP_FT = 250.0
DOSES = (0.10, 0.25, 0.50, 1.00)
CAPS_FT = (2.0, 4.0, 8.0)
TRUTH = "tvt_true"
BASE = "consensus_center"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
TARGET_COLUMNS = (
    "tail_datum_ft",
    "tail_affine_intercept_ft",
    "tail_affine_slope_ft_per_1000ft",
    "tail_piecewise_intercept_ft",
    "tail_piecewise_hinge1250_ft",
    "tail_piecewise_hinge2250_ft",
)
PROFILE_SLICES = {
    "tail_datum": slice(0, 1),
    "tail_affine": slice(1, 3),
    "tail_piecewise": slice(3, 6),
}


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v110():
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("final_consensus_v126_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _fill(values, coordinate):
    values = np.asarray(values, np.float64)
    coordinate = np.asarray(coordinate, np.float64)
    finite = np.isfinite(values) & np.isfinite(coordinate)
    if finite.sum() < 2:
        raise RuntimeError(f"Only {int(finite.sum())} finite sequence values")
    output = values.copy()
    output[~finite] = np.interp(coordinate[~finite], coordinate[finite], values[finite])
    return output


def _robust(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = float(np.std(values))
    return (values - centre) / max(scale, 1e-3)


def _interp(progress, values, grid):
    return np.interp(grid, progress, np.asarray(values, np.float64))


def _load_record_relaxed(V, well):
    original = V["ROW_STRIDE"]
    try:
        V["ROW_STRIDE"] = 1
        return V["_load_record"](well)
    finally:
        V["ROW_STRIDE"] = original


def _sequence_features(local, record):
    eval_rows = record["eval_rows"]
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{local['well'].iloc[0]}: row mismatch")
    truth_gap = float(np.max(np.abs(
        local[TRUTH].to_numpy(np.float64) - record["truth"][eval_rows].astype(np.float64)
    )))
    if truth_gap > 0.01:
        raise RuntimeError(f"{local['well'].iloc[0]}: truth gap {truth_gap}")
    full_md = record["md"].astype(np.float64)
    md = full_md[eval_rows]
    distance = md - float(record["anchor_md"])
    progress = distance / max(float(distance[-1]), 1.0)
    grid = np.linspace(0.0, 1.0, NODES)
    gr = _fill(record["gr"], full_md)[eval_rows]
    z = _fill(record["z"], full_md)[eval_rows]
    base = local[BASE].to_numpy(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    parent = local[PARENT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)

    type_tvt = np.asarray(record["type_tvt"], np.float64)
    type_gr = np.asarray(record["type_gr"], np.float64)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_frame = pd.DataFrame({"tvt": type_tvt[finite], "gr": type_gr[finite]})
    type_frame = type_frame.groupby("tvt", as_index=False, sort=True)["gr"].mean()
    if len(type_frame) < 16:
        raise RuntimeError(f"{local['well'].iloc[0]}: only {len(type_frame)} typewell samples")
    type_tvt = type_frame["tvt"].to_numpy(np.float64)
    type_gr = type_frame["gr"].to_numpy(np.float64)
    mismatch = [gr - np.interp(base + offset, type_tvt, type_gr) for offset in (-24, -12, 0, 12, 24)]
    spacing = max(float(np.median(np.diff(md))), 1e-3)
    channels = [
        grid,
        np.log1p(grid * max(float(distance[-1]), 1.0)) / math.log1p(max(float(distance[-1]), 1.0)),
        _interp(progress, base - base[0], grid) / 32.0,
        _interp(progress, exact - base, grid) / 8.0,
        _interp(progress, parent - base, grid) / 16.0,
        _interp(progress, coarse - base, grid) / 16.0,
        _interp(progress, base - exact, grid) / 8.0,
        _interp(progress, coarse - parent, grid) / 16.0,
        _interp(progress, _robust(gr), grid),
        _interp(progress, np.gradient(_robust(gr), spacing), grid),
        _interp(progress, (z - z[0]) / 100.0, grid),
        _interp(progress, np.gradient(z, md), grid),
        _interp(progress, np.gradient(base, md), grid),
        _interp(progress, np.gradient(np.gradient(base, md), md) * 1000.0, grid),
    ]
    channels.extend(_interp(progress, values / max(float(record["noise_scale"]), 5.0), grid) for values in mismatch)

    known = np.flatnonzero(np.isfinite(record["tvt_input"]) & np.isfinite(record["gr"]))
    prefix = known[record["md"][known] >= float(record["anchor_md"]) - 500.0]
    if len(prefix) < 2:
        raise RuntimeError(f"{local['well'].iloc[0]}: visible prefix is unavailable")
    prefix_gr = record["gr"][prefix].astype(np.float64)
    metadata = np.asarray([
        float(distance[-1]) / 5000.0,
        spacing / 2.0,
        float(record["noise_scale"]) / 20.0,
        float(record["anchor_surface"]) / 15000.0,
        float(record["tvt_input"][record["anchor_index"]]) / 15000.0,
        float(record["z"][record["anchor_index"]]) / 15000.0,
        float(np.nanmean(prefix_gr)) / 100.0,
        float(np.nanstd(prefix_gr)) / 50.0,
        float(base[0]) / 15000.0,
        float(base[-1] - base[0]) / 100.0,
        float(np.std(base - exact)) / 10.0,
        float(np.std(coarse - parent)) / 20.0,
    ], np.float64)
    sequence = np.asarray(channels, np.float64)
    if not np.isfinite(sequence).all() or not np.isfinite(metadata).all():
        raise RuntimeError(f"{local['well'].iloc[0]}: non-finite sequence features")
    return sequence.astype(np.float32), metadata.astype(np.float32)


def _basis(profile, distance):
    distance = np.asarray(distance, np.float64)
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    x = (distance - np.mean(distance)) / 1000.0
    if profile == "tail_datum":
        return taper[:, None]
    if profile == "tail_affine":
        return np.column_stack([taper, taper * x])
    if profile == "tail_piecewise":
        hinge1 = taper * np.clip((distance - 1250.0) / 1000.0, 0.0, None)
        hinge2 = taper * np.clip((distance - 2250.0) / 1000.0, 0.0, None)
        return np.column_stack([taper, hinge1, hinge2])
    raise KeyError(profile)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, width),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class SequenceCoefficientEncoder(nn.Module):
    def __init__(self, channels, metadata_features, outputs=6, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8)))
        self.attention = nn.Conv1d(width, 1, 1)
        self.metadata = nn.Sequential(
            nn.Linear(metadata_features, 24), nn.LayerNorm(24), nn.SiLU(), nn.Dropout(0.10)
        )
        self.head = nn.Sequential(
            nn.Linear(width * 3 + 24, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(96, outputs),
        )

    def forward(self, sequence, metadata):
        encoded = self.blocks(self.stem(sequence))
        attention = torch.softmax(self.attention(encoded), dim=-1)
        attended = torch.sum(attention * encoded, dim=-1)
        mean = torch.mean(encoded, dim=-1)
        maximum = torch.amax(encoded, dim=-1)
        meta = self.metadata(metadata)
        return self.head(torch.cat([attended, mean, maximum, meta], dim=1))


def _predict(model, sequence, metadata, device):
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(sequence).to(device),
            torch.from_numpy(metadata).to(device),
        ).cpu().numpy()
    return output


for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
_seed_everything(SEED)
R = _bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

frame = pd.read_parquet(BASE_PATH)
frame["well"] = frame["well"].astype(str)
required_base = {"id", "well", "row_idx", "fold", TRUTH, "md_since", BASE, "exact_v76"}
missing_base = sorted(required_base.difference(frame.columns))
if missing_base:
    raise RuntimeError(f"Base OOF missing columns: {missing_base}")
if frame["id"].duplicated().any():
    raise RuntimeError("Base OOF contains duplicate ids")
missing_lineage = [column for column in (PARENT, COARSE) if column not in frame]
if missing_lineage:
    lineage = pd.read_parquet(
        R.V70_PATH, columns=["id", "fold", TRUTH, *missing_lineage],
        filters=[("fold", "in", [1, 2, 3])],
    )
    frame = frame.merge(lineage, on="id", how="left", validate="one_to_one", suffixes=("", "_lineage"))
    if frame[missing_lineage].isna().any().any():
        raise RuntimeError("Incomplete V70 lineage")
    truth_gap = float(np.max(np.abs(frame[TRUTH] - frame[f"{TRUTH}_lineage"])))
    fold_gap = int(np.max(np.abs(frame["fold"] - frame["fold_lineage"])))
    if truth_gap > 0.01 or fold_gap != 0:
        raise RuntimeError(f"Lineage mismatch truth={truth_gap} fold={fold_gap}")
    frame = frame.drop(columns=[f"{TRUTH}_lineage", "fold_lineage"])
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)

targets = pd.read_csv(TARGET_PATH, dtype={"well": str})
targets["well"] = targets["well"].astype(str)
targets = targets[["well", "fold", *TARGET_COLUMNS]].copy()
if targets["well"].duplicated().any():
    raise RuntimeError("V123 coefficient targets contain duplicate wells")
target_map = targets.set_index("well")

well_order = sorted(frame["well"].unique())
missing_targets = sorted(set(well_order).difference(target_map.index))
if missing_targets:
    raise RuntimeError(f"V123 targets missing {len(missing_targets)} wells: {missing_targets[:10]}")
sequence_rows, metadata_rows, target_rows, fold_rows = [], [], [], []
for number, well in enumerate(well_order, 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable")
    local_fold = int(local["fold"].iloc[0])
    target_fold = int(target_map.loc[well, "fold"])
    if local_fold != target_fold:
        raise RuntimeError(f"{well}: base/target fold mismatch {local_fold}/{target_fold}")
    sequence, metadata = _sequence_features(local, _load_record_relaxed(V, well))
    sequence_rows.append(sequence)
    metadata_rows.append(metadata)
    target_rows.append(target_map.loc[well, list(TARGET_COLUMNS)].to_numpy(np.float32))
    fold_rows.append(local_fold)
    if number % 100 == 0 or number == len(well_order):
        print(f"V126 sequences {number}/{len(well_order)}", flush=True)

sequence = np.stack(sequence_rows).astype(np.float32)
metadata = np.stack(metadata_rows).astype(np.float32)
target = np.stack(target_rows).astype(np.float32)
folds = np.asarray(fold_rows, np.int8)
train_index = np.flatnonzero(np.isin(folds, TRAIN_FOLDS))
fold3_index = np.flatnonzero(folds == EVAL_FOLD)
fold3_wells = [well_order[index] for index in fold3_index]
discovery_index = fold3_index[:DISCOVERY_WELLS]
confirmation_index = fold3_index[DISCOVERY_WELLS:]
discovery_wells = set(well_order[index] for index in discovery_index)
confirmation_wells = set(well_order[index] for index in confirmation_index)
if len(train_index) != 310 or len(discovery_index) != 24 or len(confirmation_index) < 129:
    raise RuntimeError(
        f"Coverage train/discovery/confirmation={len(train_index)}/{len(discovery_index)}/{len(confirmation_index)}"
    )

sequence_mean = np.mean(sequence[train_index], axis=(0, 2), keepdims=True)
sequence_std = np.std(sequence[train_index], axis=(0, 2), keepdims=True)
sequence_std = np.maximum(sequence_std, 1e-3)
sequence = ((sequence - sequence_mean) / sequence_std).astype(np.float32)
metadata_mean = np.mean(metadata[train_index], axis=0, keepdims=True)
metadata_std = np.maximum(np.std(metadata[train_index], axis=0, keepdims=True), 1e-3)
metadata = ((metadata - metadata_mean) / metadata_std).astype(np.float32)
target_mean = np.mean(target[train_index], axis=0, keepdims=True)
target_std = np.maximum(np.std(target[train_index], axis=0, keepdims=True), 0.25)
target_normalized = np.clip((target - target_mean) / target_std, -6.0, 6.0).astype(np.float32)
target_low = np.quantile(target[train_index], 0.01, axis=0)
target_high = np.quantile(target[train_index], 0.99, axis=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SequenceCoefficientEncoder(sequence.shape[1], metadata.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.SmoothL1Loss(beta=0.75)
dataset = TensorDataset(
    torch.from_numpy(sequence[train_index]),
    torch.from_numpy(metadata[train_index]),
    torch.from_numpy(target_normalized[train_index]),
)
generator = torch.Generator().manual_seed(SEED)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

best_loss = float("inf")
best_epoch = 0
best_state = None
stale = 0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for batch_sequence, batch_metadata, batch_target in loader:
        batch_sequence = batch_sequence.to(device)
        batch_metadata = batch_metadata.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_sequence, batch_metadata)
        loss = criterion(prediction, batch_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))
    validation_prediction = _predict(
        model, sequence[discovery_index], metadata[discovery_index], device
    )
    validation_loss = float(np.mean(np.square(
        validation_prediction - target_normalized[discovery_index]
    )))
    history.append({
        "epoch": epoch,
        "train_loss": float(np.mean(train_losses)),
        "discovery_coefficient_mse": validation_loss,
    })
    if validation_loss < best_loss - 1e-5:
        best_loss = validation_loss
        best_epoch = epoch
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stale = 0
    else:
        stale += 1
    if epoch == 1 or epoch % 10 == 0:
        print("V126 EPOCH", history[-1], flush=True)
    if stale >= PATIENCE:
        break

if best_state is None:
    raise RuntimeError("V126 did not select a checkpoint")
model.load_state_dict(best_state)
torch.save(
    {
        "state_dict": best_state,
        "sequence_mean": sequence_mean,
        "sequence_std": sequence_std,
        "metadata_mean": metadata_mean,
        "metadata_std": metadata_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_low": target_low,
        "target_high": target_high,
        "target_columns": TARGET_COLUMNS,
        "nodes": NODES,
    },
    MODEL_DIR / "best_sequence_coefficient.pt",
)
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

predicted_normalized = _predict(model, sequence, metadata, device)
predicted_target = predicted_normalized * target_std + target_mean
predicted_target = np.clip(predicted_target, target_low, target_high)
coefficient_map = dict(zip(well_order, predicted_target))
recipes = [
    (profile, dose, cap)
    for profile in PROFILE_SLICES for dose in DOSES for cap in CAPS_FT
]


def _evaluate(selected_wells, recipe, endpoint, coefficient_override=None):
    profile, dose, cap = recipe
    source = coefficient_map if coefficient_override is None else coefficient_override
    coefficient_slice = PROFILE_SLICES[profile]
    local_parts, prediction_parts = [], []
    for well in sorted(selected_wells):
        local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        distance = local["md_since"].to_numpy(np.float64)
        coefficient = np.asarray(source[well], np.float64)[coefficient_slice]
        move = _basis(profile, distance) @ coefficient
        move = np.clip(float(dose) * move, -float(cap), float(cap))
        prediction_parts.append(local[BASE].to_numpy(np.float64) + move)
        local_parts.append(local)
    matched = pd.concat(local_parts, ignore_index=True)
    prediction = np.concatenate(prediction_parts)
    return R._metric(matched, prediction, endpoint), matched, prediction


discovery_frame = frame[frame["well"].isin(discovery_wells)].copy()
discovery_base = R._metric(discovery_frame, discovery_frame[BASE].to_numpy(np.float64), BASE)
scan_rows = []
for recipe in recipes:
    result, _, _ = _evaluate(discovery_wells, recipe, "sequence_coefficient")
    profile, dose, cap = recipe
    result.update({
        "profile": profile, "dose": dose, "cap_ft": cap,
        "gain": float(discovery_base["rmse"] - result["rmse"]),
        "far_gain": float(discovery_base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
        "near_degradation": float(result["near_0_500_rmse"] - discovery_base["near_0_500_rmse"]),
        "p95_gain": float(discovery_base["p95_well_rmse"] - result["p95_well_rmse"]),
    })
    scan_rows.append(result)
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.05) & (scan["far_gain"] >= 0.0)
    & (scan["near_degradation"].abs() <= 1e-5) & (scan["p95_gain"] >= -0.10)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (
    str(selected_row["profile"]), float(selected_row["dose"]), float(selected_row["cap_ft"])
)

summary_rows = [dict(discovery_base, stage="fold3_discovery")]
discovery_candidate, _, _ = _evaluate(discovery_wells, selected, "sequence_coefficient")
summary_rows.append(dict(discovery_candidate, stage="fold3_discovery"))
confirmation_values = None
checks = None
if len(eligible):
    confirmation_frame = frame[frame["well"].isin(confirmation_wells)].copy()
    confirmation_base = R._metric(
        confirmation_frame, confirmation_frame[BASE].to_numpy(np.float64), BASE
    )
    candidate, matched, prediction = _evaluate(
        confirmation_wells, selected, "sequence_coefficient"
    )
    ordered = sorted(confirmation_wells)
    rolled_values = np.roll(
        np.vstack([coefficient_map[well] for well in ordered]), max(1, len(ordered) // 7), axis=0
    )
    rolled_map = dict(coefficient_map)
    for well, value in zip(ordered, rolled_values):
        rolled_map[well] = value
    rolled, _, rolled_prediction = _evaluate(
        confirmation_wells, selected, "rolled_sequence_control", rolled_map
    )
    summary_rows.extend([
        dict(confirmation_base, stage="fold3_confirmation"),
        dict(candidate, stage="fold3_confirmation"),
        dict(rolled, stage="fold3_confirmation"),
    ])
    confirmation_values = {
        "gain": float(confirmation_base["rmse"] - candidate["rmse"]),
        "far_gain": float(confirmation_base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        "near_degradation": float(candidate["near_0_500_rmse"] - confirmation_base["near_0_500_rmse"]),
        "p95_gain": float(confirmation_base["p95_well_rmse"] - candidate["p95_well_rmse"]),
        "aligned_increment_vs_rolled": float(rolled["rmse"] - candidate["rmse"]),
    }
    checks = {
        "gain_ge_0p05": confirmation_values["gain"] >= 0.05,
        "far_gain_nonnegative": confirmation_values["far_gain"] >= 0.0,
        "near_frozen_exact": abs(confirmation_values["near_degradation"]) <= 1e-5,
        "p95_not_worse_by_gt_0p10": confirmation_values["p95_gain"] >= -0.10,
        "aligned_beats_rolled_ge_0p02": confirmation_values["aligned_increment_vs_rolled"] >= 0.02,
    }
    output = matched[["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
    output["sequence_coefficient"] = prediction.astype(np.float32)
    output["rolled_sequence_control"] = rolled_prediction.astype(np.float32)
    output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
coefficient_panel = pd.DataFrame(predicted_target, columns=[f"pred_{name}" for name in TARGET_COLUMNS])
coefficient_panel.insert(0, "well", well_order)
coefficient_panel.insert(1, "fold", folds)
for number, name in enumerate(TARGET_COLUMNS):
    coefficient_panel[f"true_{name}"] = target[:, number]
coefficient_panel.to_csv(OUT_DIR / "coefficient_predictions.csv", index=False)
strict_pass = bool(len(eligible) and checks and all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_sequence_training_Fold3_discovery_confirmation",
    "sample_unit": "one complete physical well",
    "architecture": "dilated_1D_residual_encoder_attention_pooling_constrained_coefficient_heads",
    "legal_inputs_only": True,
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "sequence_nodes": NODES,
    "sequence_channels": int(sequence.shape[1]),
    "metadata_features": int(metadata.shape[1]),
    "train_wells": len(train_index),
    "selection_wells": len(discovery_index),
    "confirmation_wells": len(confirmation_index),
    "best_epoch": best_epoch,
    "best_discovery_coefficient_mse": best_loss,
    "search_count": len(recipes),
    "selected_recipe": {
        "profile": selected[0], "dose": selected[1], "cap_ft": selected[2],
        "passed_discovery_constraints": bool(len(eligible)),
    },
    "discovery": {
        "gain": float(selected_row["gain"]), "far_gain": float(selected_row["far_gain"]),
        "near_degradation": float(selected_row["near_degradation"]),
        "p95_gain": float(selected_row["p95_gain"]),
    },
    "confirmation": confirmation_values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> cross-fit sequence coefficient heads and build deployment model"
        if strict_pass else
        "FAIL -> constrained residual is not inferable from current full-well evidence; stop postprocessing"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV126 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV126 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV126 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
