"""V128: high-upside anchored sequence-to-sequence residual model.

Unlike V120's row model and V126's six coefficient heads, this model predicts
an entire smooth residual trajectory on a fixed progress grid. Fold1+2 train;
the first 24 Fold3 wells select epoch and a guarded consumption recipe; the
remaining Fold3 wells are touched once. Truth is used only for fitting and
metrics. Deployment inputs are the current legal endpoints and raw well logs.
"""

from __future__ import annotations

import json
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
EXPERIMENT = "final_consensus_residual_tcn_v128"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V126_SOURCE = ROOT / "final_consensus_sequence_coefficient_v126_cell.py"
TRAIN_FOLDS = (1, 2)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
NODES = 192
SEED = 20260801
EPOCHS = 160
PATIENCE = 25
BATCH_SIZE = 24
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 3e-3
TARGET_SCALE_FT = 8.0
FREEZE_VALUES = (0.0, 500.0, 1000.0)
RAMP_FT = 250.0
DOSES = (0.25, 0.50, 0.75, 1.00)
CAPS_FT = (2.0, 4.0, 8.0)


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


def _bootstrap_v126():
    if not V126_SOURCE.is_file():
        raise FileNotFoundError(V126_SOURCE)
    source = V126_SOURCE.read_text(encoding="utf-8")
    marker = "for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):"
    if marker not in source:
        raise RuntimeError(f"V126 definition marker missing from {V126_SOURCE}")
    module = types.ModuleType("final_consensus_v128_v126_definitions")
    module.__file__ = str(V126_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V126_SOURCE), "exec"), module.__dict__)
    return module


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _smooth(values):
    values = np.asarray(values, np.float64)
    kernel = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1], np.float64)
    kernel /= kernel.sum()
    padded = np.pad(values, (4, 4), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _taper(distance, freeze_ft):
    return Q._smoothstep((np.asarray(distance, np.float64) - float(freeze_ft)) / RAMP_FT)


Q = _bootstrap_v126()
_seed_everything(SEED)
R = Q._bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

if not Q.BASE_PATH.is_file():
    raise FileNotFoundError(Q.BASE_PATH)
frame = pd.read_parquet(Q.BASE_PATH)
frame["well"] = frame["well"].astype(str)
required_base = {
    "id", "well", "row_idx", "fold", Q.TRUTH, "md_since", Q.BASE, "exact_v76"
}
missing_base = sorted(required_base.difference(frame.columns))
if missing_base:
    raise RuntimeError(f"Base OOF missing columns: {missing_base}")
if frame["id"].duplicated().any():
    raise RuntimeError("Base OOF contains duplicate ids")

missing_lineage = [column for column in (Q.PARENT, Q.COARSE) if column not in frame]
if missing_lineage:
    lineage = pd.read_parquet(
        R.V70_PATH,
        columns=["id", "fold", Q.TRUTH, *missing_lineage],
        filters=[("fold", "in", [1, 2, 3])],
    )
    frame = frame.merge(
        lineage, on="id", how="left", validate="one_to_one", suffixes=("", "_lineage")
    )
    if frame[missing_lineage].isna().any().any():
        raise RuntimeError("Incomplete V70 endpoint lineage")
    truth_gap = float(np.max(np.abs(frame[Q.TRUTH] - frame[f"{Q.TRUTH}_lineage"])))
    fold_gap = int(np.max(np.abs(frame["fold"] - frame["fold_lineage"])))
    if truth_gap > 0.01 or fold_gap != 0:
        raise RuntimeError(f"Lineage mismatch truth={truth_gap} fold={fold_gap}")
    frame = frame.drop(columns=[f"{Q.TRUTH}_lineage", "fold_lineage"])
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
well_positions = {
    str(well): np.asarray(positions, np.int64)
    for well, positions in frame.groupby("well", sort=False).indices.items()
}


def _well_frame(well):
    return frame.iloc[well_positions[str(well)]]


well_order = sorted(well_positions)
sequence_rows = []
metadata_rows = []
target_rows = []
distance_rows = []
fold_rows = []
oracle_parts = []
oracle_predictions = []
grid = np.linspace(0.0, 1.0, NODES)
for number, well in enumerate(well_order, 1):
    local = _well_frame(well)
    record = Q._load_record_relaxed(V, well)
    sequence, metadata = Q._sequence_features(local, record)
    distance = local["md_since"].to_numpy(np.float64)
    progress = distance / max(float(distance[-1]), 1.0)
    residual = local[Q.TRUTH].to_numpy(np.float64) - local[Q.BASE].to_numpy(np.float64)
    target_grid = _smooth(np.interp(grid, progress, residual))
    oracle_grid = np.clip(target_grid, -24.0, 24.0)
    oracle_row = np.interp(progress, grid, oracle_grid)
    sequence_rows.append(sequence)
    metadata_rows.append(metadata)
    target_rows.append(target_grid.astype(np.float32))
    distance_rows.append(float(distance[-1]))
    fold_rows.append(int(local["fold"].iloc[0]))
    oracle_parts.append(local)
    oracle_predictions.append(local[Q.BASE].to_numpy(np.float64) + oracle_row)
    if number % 100 == 0 or number == len(well_order):
        print(f"V128 sequences {number}/{len(well_order)}", flush=True)

sequence_raw = np.stack(sequence_rows).astype(np.float32)
metadata_raw = np.stack(metadata_rows).astype(np.float32)
target_raw = np.stack(target_rows).astype(np.float32)
distance_max = np.asarray(distance_rows, np.float32)
fold_by_index = np.asarray(fold_rows, np.int8)
train_index = np.flatnonzero(np.isin(fold_by_index, TRAIN_FOLDS))
fold3_index = np.flatnonzero(fold_by_index == EVAL_FOLD)
discovery_index = fold3_index[:DISCOVERY_WELLS]
confirmation_index = fold3_index[DISCOVERY_WELLS:]
if len(train_index) != 310 or len(discovery_index) != 24 or len(confirmation_index) < 129:
    raise RuntimeError(
        f"Coverage train/discovery/confirmation={len(train_index)}/{len(discovery_index)}/{len(confirmation_index)}"
    )
discovery_wells = set(well_order[index] for index in discovery_index)
confirmation_wells = set(well_order[index] for index in confirmation_index)

oracle_frame = pd.concat(oracle_parts, ignore_index=True)
oracle_prediction = np.concatenate(oracle_predictions)
oracle_metric = R._metric(oracle_frame, oracle_prediction, "smooth_grid_oracle")
base_metric_all = R._metric(
    oracle_frame, oracle_frame[Q.BASE].to_numpy(np.float64), Q.BASE
)
oracle_gain = float(base_metric_all["rmse"] - oracle_metric["rmse"])
print("V128 REPRESENTATION ORACLE", {
    "base_rmse": base_metric_all["rmse"],
    "oracle_rmse": oracle_metric["rmse"],
    "oracle_gain": oracle_gain,
}, flush=True)
if oracle_gain < 0.50:
    raise RuntimeError(f"V128 representation oracle gain is only {oracle_gain:.6f} ft")

sequence_mean = np.mean(sequence_raw[train_index], axis=(0, 2), keepdims=True)
sequence_std = np.maximum(
    np.std(sequence_raw[train_index], axis=(0, 2), keepdims=True), 1e-3
)
metadata_mean = np.mean(metadata_raw[train_index], axis=0, keepdims=True)
metadata_std = np.maximum(np.std(metadata_raw[train_index], axis=0, keepdims=True), 1e-3)
sequence = ((sequence_raw - sequence_mean) / sequence_std).astype(np.float32)
metadata = ((metadata_raw - metadata_mean) / metadata_std).astype(np.float32)
target_normalized = np.clip(target_raw / TARGET_SCALE_FT, -6.0, 6.0).astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class ResidualTCN(nn.Module):
    def __init__(self, channels, metadata_features, width=64):
        super().__init__()
        self.metadata = nn.Sequential(
            nn.Linear(metadata_features, 32), nn.LayerNorm(32), nn.SiLU()
        )
        self.stem = nn.Sequential(
            nn.Conv1d(channels + 32, width, 5, padding=2),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(
            ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 1)
        ))
        self.head = nn.Sequential(
            nn.Conv1d(width, 32, 3, padding=1), nn.SiLU(), nn.Conv1d(32, 1, 1)
        )

    def forward(self, sequence_value, metadata_value):
        meta = self.metadata(metadata_value).unsqueeze(-1).expand(-1, -1, sequence_value.shape[-1])
        encoded = self.blocks(self.stem(torch.cat([sequence_value, meta], dim=1)))
        return self.head(encoded).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualTCN(sequence.shape[1], metadata.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
dataset = TensorDataset(
    torch.from_numpy(sequence[train_index]),
    torch.from_numpy(metadata[train_index]),
    torch.from_numpy(target_normalized[train_index]),
    torch.from_numpy(distance_max[train_index]),
)
generator = torch.Generator().manual_seed(SEED)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
grid_tensor = torch.from_numpy(grid.astype(np.float32)).to(device)


def _predict(indices):
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(sequence[indices]).to(device),
            torch.from_numpy(metadata[indices]).to(device),
        ).cpu().numpy()
    return output * TARGET_SCALE_FT


def _loss(prediction, target, maximum_distance):
    distance = grid_tensor[None, :] * maximum_distance[:, None]
    taper = torch.clamp(distance / 250.0, 0.0, 1.0)
    taper = taper * taper * (3.0 - 2.0 * taper)
    prediction = prediction * taper
    point = nn.functional.smooth_l1_loss(prediction, target, beta=0.75)
    first = torch.mean(torch.abs(prediction[:, 1:] - prediction[:, :-1]))
    second = torch.mean(torch.abs(prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]))
    return point + 0.015 * first + 0.05 * second


best_loss = float("inf")
best_epoch = 0
best_state = None
stale = 0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for batch_sequence, batch_metadata, batch_target, batch_distance in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_sequence.to(device), batch_metadata.to(device))
        loss = _loss(
            prediction,
            batch_target.to(device),
            batch_distance.to(device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        discovery_prediction = model(
            torch.from_numpy(sequence[discovery_index]).to(device),
            torch.from_numpy(metadata[discovery_index]).to(device),
        )
        validation_loss = float(_loss(
            discovery_prediction,
            torch.from_numpy(target_normalized[discovery_index]).to(device),
            torch.from_numpy(distance_max[discovery_index]).to(device),
        ).cpu())
    row = {
        "epoch": epoch,
        "train_loss": float(np.mean(losses)),
        "discovery_loss": validation_loss,
    }
    history.append(row)
    if validation_loss < best_loss - 1e-5:
        best_loss = validation_loss
        best_epoch = epoch
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stale = 0
    else:
        stale += 1
    if epoch == 1 or epoch % 10 == 0:
        print("V128 EPOCH", row, flush=True)
    if stale >= PATIENCE:
        break
if best_state is None:
    raise RuntimeError("V128 failed to select a checkpoint")
model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
torch.save({
    "state_dict": best_state,
    "sequence_mean": sequence_mean,
    "sequence_std": sequence_std,
    "metadata_mean": metadata_mean,
    "metadata_std": metadata_std,
    "target_scale_ft": TARGET_SCALE_FT,
    "nodes": NODES,
    "best_epoch": best_epoch,
}, MODEL_DIR / "best_residual_tcn.pt")

predicted_grid = _predict(np.arange(len(well_order)))
predicted_grid = np.clip(predicted_grid, -24.0, 24.0)
grid_map = dict(zip(well_order, predicted_grid))


def _evaluate(wells, recipe, endpoint, grid_override=None):
    freeze_ft, dose, cap_ft = recipe
    source = grid_map if grid_override is None else grid_override
    local_parts = []
    prediction_parts = []
    move_parts = []
    for well in sorted(wells):
        local = _well_frame(well)
        distance = local["md_since"].to_numpy(np.float64)
        progress = distance / max(float(distance[-1]), 1.0)
        move = np.interp(progress, grid, np.asarray(source[well], np.float64))
        move *= _taper(distance, freeze_ft)
        move = np.clip(float(dose) * move, -float(cap_ft), float(cap_ft))
        local_parts.append(local)
        move_parts.append(move)
        prediction_parts.append(local[Q.BASE].to_numpy(np.float64) + move)
    matched = pd.concat(local_parts, ignore_index=True)
    prediction = np.concatenate(prediction_parts)
    move = np.concatenate(move_parts)
    return R._metric(matched, prediction, endpoint), matched, prediction, move


recipes = [
    (freeze_ft, dose, cap_ft)
    for freeze_ft in FREEZE_VALUES for dose in DOSES for cap_ft in CAPS_FT
]
discovery_frame = frame[frame["well"].isin(discovery_wells)].copy()
discovery_base = R._metric(
    discovery_frame, discovery_frame[Q.BASE].to_numpy(np.float64), Q.BASE
)
scan_rows = []
for recipe in recipes:
    result, _, _, _ = _evaluate(discovery_wells, recipe, "residual_tcn")
    row = dict(result)
    row.update({
        "freeze_ft": recipe[0],
        "dose": recipe[1],
        "cap_ft": recipe[2],
        "gain": float(discovery_base["rmse"] - result["rmse"]),
        "far_gain": float(discovery_base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
        "near_degradation": float(result["near_0_500_rmse"] - discovery_base["near_0_500_rmse"]),
        "p95_gain": float(discovery_base["p95_well_rmse"] - result["p95_well_rmse"]),
    })
    scan_rows.append(row)
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.30)
    & (scan["far_gain"] >= 0.20)
    & (scan["near_degradation"] <= 0.05)
    & (scan["p95_gain"] >= -0.20)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (
    float(selected_row["freeze_ft"]),
    float(selected_row["dose"]),
    float(selected_row["cap_ft"]),
)

summary_rows = [dict(discovery_base, stage="fold3_discovery")]
discovery_candidate, _, _, _ = _evaluate(discovery_wells, selected, "residual_tcn")
summary_rows.append(dict(discovery_candidate, stage="fold3_discovery"))
confirmation_values = None
checks = None
if len(eligible):
    confirmation_frame = frame[frame["well"].isin(confirmation_wells)].copy()
    confirmation_base = R._metric(
        confirmation_frame, confirmation_frame[Q.BASE].to_numpy(np.float64), Q.BASE
    )
    candidate, matched, prediction, move = _evaluate(
        confirmation_wells, selected, "residual_tcn"
    )
    ordered = sorted(confirmation_wells)
    rolled_values = np.roll(
        np.vstack([grid_map[well] for well in ordered]), max(1, len(ordered) // 7), axis=0
    )
    rolled_map = dict(grid_map)
    for well, value in zip(ordered, rolled_values):
        rolled_map[well] = value
    rolled, _, rolled_prediction, _ = _evaluate(
        confirmation_wells, selected, "rolled_residual_control", rolled_map
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
        "gain_ge_0p20": confirmation_values["gain"] >= 0.20,
        "far_gain_ge_0p10": confirmation_values["far_gain"] >= 0.10,
        "near_degradation_le_0p05": confirmation_values["near_degradation"] <= 0.05,
        "p95_not_worse_by_gt_0p20": confirmation_values["p95_gain"] >= -0.20,
        "aligned_beats_rolled_ge_0p05": confirmation_values["aligned_increment_vs_rolled"] >= 0.05,
    }
    output = matched[["id", "well", "row_idx", "fold", Q.TRUTH, "md_since", Q.BASE]].copy()
    output["residual_tcn"] = prediction.astype(np.float32)
    output["rolled_residual_control"] = rolled_prediction.astype(np.float32)
    output["residual_tcn_move"] = move.astype(np.float32)
    output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
strict_pass = bool(len(eligible) and checks and all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_full_trajectory_training_Fold3_discovery_confirmation",
    "sample_unit": "one complete physical well",
    "architecture": "anchored_multiscale_dilated_TCN_sequence_to_sequence_residual",
    "legal_inputs_only": True,
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "representation_oracle": {
        "base_rmse": base_metric_all["rmse"],
        "oracle_rmse": oracle_metric["rmse"],
        "oracle_gain": oracle_gain,
    },
    "train_wells": len(train_index),
    "selection_wells": len(discovery_index),
    "confirmation_wells": len(confirmation_index),
    "best_epoch": best_epoch,
    "selected_recipe": {
        "freeze_ft": selected[0], "ramp_ft": RAMP_FT,
        "dose": selected[1], "cap_ft": selected[2],
        "passed_discovery_constraints": bool(len(eligible)),
    },
    "discovery": {
        "gain": float(selected_row["gain"]),
        "far_gain": float(selected_row["far_gain"]),
        "near_degradation": float(selected_row["near_degradation"]),
        "p95_gain": float(selected_row["p95_gain"]),
    },
    "confirmation": confirmation_values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> cross-fit the residual TCN and measure complete 464-well CV"
        if strict_pass else
        "FAIL -> full residual shape is not observable; replace the trajectory generator/data representation"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV128 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV128 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV128 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
