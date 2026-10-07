"""V135: dense GR cost-volume correction anchored on the exact V127 endpoint.

V36 decoded an absolute trajectory from raw evidence, while V128 exposed only
five sparse mismatch channels to a direct residual regressor.  This experiment
builds a dense local candidate axis around V127 and uses a complete-well 2D
encoder to predict only the residual from that strong endpoint.  The first
1000 ft are frozen exactly at consumption time.
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "anchored_v127_costvolume_residual_v135"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V126_SOURCE = ROOT / "final_consensus_sequence_coefficient_v126_cell.py"
V127_PATH = OOF_ROOT / "final_consensus_sequence_coefficient_crossfit_v127" / "crossfit_oof.parquet"
TRAIN_FOLDS = (1, 2)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
BASE = "nested_sequence_coefficient"
TRUTH = "tvt_true"
NODES = 192
HALF_WIDTH_FT = 48.0
STATE_STEP_FT = 3.0
OFFSETS = np.arange(-HALF_WIDTH_FT, HALF_WIDTH_FT + 0.1, STATE_STEP_FT).astype(np.float32)
SEED = 20260802
EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 2e-3
FREEZE_FT = 1000.0
RAMP_FT = 250.0
DOSES = (0.25, 0.50, 0.75, 1.00)
CAPS_FT = (4.0, 8.0, 16.0, 24.0)


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
    module = types.ModuleType("anchored_v135_v126_definitions")
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


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _rolling(values, width):
    values = np.asarray(values, np.float64)
    kernel = np.ones(int(width), np.float64) / float(width)
    pad = int(width) // 2
    return np.apply_along_axis(
        lambda x: np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid"),
        0,
        values,
    )


def _rolling_z(values, width=15):
    mean = _rolling(values, width)
    square = _rolling(np.square(values), width)
    return (values - mean) / np.sqrt(np.maximum(square - np.square(mean), 1.0))


Q = _bootstrap_v126()
_seed_everything(SEED)
R = Q._bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

if not V127_PATH.is_file():
    raise FileNotFoundError(V127_PATH)
frame = pd.read_parquet(V127_PATH)
frame["well"] = frame["well"].astype(str)
required = {"id", "well", "row_idx", "fold", TRUTH, "md_since", BASE}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V127 OOF missing columns: {missing}")
if frame["id"].duplicated().any():
    raise RuntimeError("V127 OOF contains duplicate ids")
frame = frame[frame["fold"].isin((1, 2, 3))].copy()

# V127 deliberately persisted a compact endpoint table.  Reconstruct the
# endpoint lineage required by V126's legal sequence feature builder.
lineage_columns = ("exact_v76", Q.BASE, Q.PARENT, Q.COARSE)
missing_lineage = [column for column in lineage_columns if column not in frame]
if "exact_v76" in missing_lineage or Q.BASE in missing_lineage:
    base_columns = ["id", "fold", Q.TRUTH]
    base_columns.extend(
        column for column in ("exact_v76", Q.BASE)
        if column in missing_lineage
    )
    base_lineage = pd.read_parquet(
        Q.BASE_PATH,
        columns=base_columns,
        filters=[("fold", "in", [1, 2, 3])],
    )
    frame = frame.merge(
        base_lineage, on="id", how="left", validate="one_to_one",
        suffixes=("", "_base_lineage"),
    )
    truth_gap = float(np.max(np.abs(
        frame[TRUTH].to_numpy(np.float64)
        - frame[f"{Q.TRUTH}_base_lineage"].to_numpy(np.float64)
    )))
    fold_gap = int(np.max(np.abs(
        frame["fold"].to_numpy(np.int16)
        - frame["fold_base_lineage"].to_numpy(np.int16)
    )))
    if truth_gap > 0.01 or fold_gap != 0:
        raise RuntimeError(f"V127/base lineage mismatch truth={truth_gap} fold={fold_gap}")
    frame = frame.drop(columns=[f"{Q.TRUTH}_base_lineage", "fold_base_lineage"])

remaining_lineage = [column for column in (Q.PARENT, Q.COARSE) if column not in frame]
if remaining_lineage:
    endpoint_lineage = pd.read_parquet(
        R.V70_PATH,
        columns=["id", "fold", Q.TRUTH, *remaining_lineage],
        filters=[("fold", "in", [1, 2, 3])],
    )
    frame = frame.merge(
        endpoint_lineage, on="id", how="left", validate="one_to_one",
        suffixes=("", "_endpoint_lineage"),
    )
    truth_gap = float(np.max(np.abs(
        frame[TRUTH].to_numpy(np.float64)
        - frame[f"{Q.TRUTH}_endpoint_lineage"].to_numpy(np.float64)
    )))
    fold_gap = int(np.max(np.abs(
        frame["fold"].to_numpy(np.int16)
        - frame["fold_endpoint_lineage"].to_numpy(np.int16)
    )))
    if truth_gap > 0.01 or fold_gap != 0:
        raise RuntimeError(f"V127/endpoint lineage mismatch truth={truth_gap} fold={fold_gap}")
    frame = frame.drop(columns=[f"{Q.TRUTH}_endpoint_lineage", "fold_endpoint_lineage"])

missing_lineage = sorted(set(lineage_columns).difference(frame.columns))
if missing_lineage or frame[list(lineage_columns)].isna().any().any():
    raise RuntimeError(f"Incomplete V135 endpoint lineage: {missing_lineage}")
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
well_positions = {
    str(well): np.asarray(position, np.int64)
    for well, position in frame.groupby("well", sort=False).indices.items()
}


def _well_frame(well):
    return frame.iloc[well_positions[str(well)]]


grid = np.linspace(0.0, 1.0, NODES).astype(np.float64)
well_order = sorted(well_positions)
feature_rows = []
target_rows = []
distance_rows = []
fold_rows = []
coverage_rows = []

for number, well in enumerate(well_order, 1):
    local = _well_frame(well)
    record = Q._load_record_relaxed(V, well)
    eval_rows = record["eval_rows"]
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: V127/raw row mismatch {len(local)}/{len(eval_rows)}")
    truth_gap = float(np.max(np.abs(
        local[TRUTH].to_numpy(np.float64) - record["truth"][eval_rows].astype(np.float64)
    )))
    if truth_gap > 0.01:
        raise RuntimeError(f"{well}: truth mismatch {truth_gap}")

    full_md = record["md"].astype(np.float64)
    md = full_md[eval_rows]
    distance = local["md_since"].to_numpy(np.float64)
    progress = distance / max(float(distance[-1]), 1.0)
    base = local[BASE].to_numpy(np.float64)
    truth = local[TRUTH].to_numpy(np.float64)
    base_grid = np.interp(grid, progress, base)
    target = np.interp(grid, progress, truth - base)
    gr = Q._fill(record["gr"], full_md)[eval_rows]
    gr_grid = np.interp(grid, progress, gr)

    type_tvt = np.asarray(record["type_tvt"], np.float64)
    type_gr = np.asarray(record["type_gr"], np.float64)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_frame = pd.DataFrame({"tvt": type_tvt[finite], "gr": type_gr[finite]})
    type_frame = type_frame.groupby("tvt", as_index=False, sort=True)["gr"].mean()
    if len(type_frame) < 16:
        raise RuntimeError(f"{well}: insufficient typewell samples")
    type_tvt = type_frame["tvt"].to_numpy(np.float64)
    type_gr = type_frame["gr"].to_numpy(np.float64)
    surfaces = base_grid[:, None] + OFFSETS[None, :]
    candidate_gr = np.interp(surfaces, type_tvt, type_gr)
    noise = max(float(record["noise_scale"]), 5.0)
    difference = (gr_grid[:, None] - candidate_gr) / noise
    smooth5 = _rolling(difference, 5)
    smooth15 = _rolling(difference, 15)
    gr_gradient = np.gradient(gr_grid)[:, None] / noise
    candidate_gradient = np.gradient(candidate_gr, axis=0) / noise
    local_shape = _rolling_z(gr_grid[:, None], 15) - _rolling_z(candidate_gr, 15)

    sequence, _ = Q._sequence_features(local, record)
    if sequence.shape[1] != NODES:
        raise RuntimeError(f"{well}: sequence nodes {sequence.shape}")
    broadcast = np.broadcast_to(
        sequence.T[:, None, :], (NODES, len(OFFSETS), sequence.shape[0])
    )
    state_features = np.stack([
        difference,
        np.abs(difference),
        smooth5,
        np.abs(smooth5),
        smooth15,
        np.abs(smooth15),
        gr_gradient - candidate_gradient,
        local_shape,
        np.abs(local_shape),
        np.broadcast_to(OFFSETS[None, :] / HALF_WIDTH_FT, difference.shape),
        np.broadcast_to(grid[:, None], difference.shape),
    ], axis=2)
    features = np.concatenate([broadcast, state_features], axis=2)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise RuntimeError(f"{well}: non-finite V135 tensors")
    feature_rows.append(features.astype(np.float16))
    target_rows.append(target.astype(np.float32))
    distance_rows.append(float(distance[-1]))
    fold_rows.append(int(local["fold"].iloc[0]))
    coverage_rows.append(float(np.mean(np.abs(target) <= HALF_WIDTH_FT)))
    if number % 50 == 0 or number == len(well_order):
        print(f"V135 cost volumes {number}/{len(well_order)}", flush=True)

features_raw = np.stack(feature_rows).astype(np.float32)
targets = np.stack(target_rows).astype(np.float32)
maximum_distance = np.asarray(distance_rows, np.float32)
fold_by_index = np.asarray(fold_rows, np.int8)
train_index = np.flatnonzero(np.isin(fold_by_index, TRAIN_FOLDS))
fold3_index = np.flatnonzero(fold_by_index == EVAL_FOLD)
discovery_index = fold3_index[:DISCOVERY_WELLS]
confirmation_index = fold3_index[DISCOVERY_WELLS:]
if len(train_index) != 310 or len(discovery_index) != 24 or len(confirmation_index) < 129:
    raise RuntimeError(
        f"Coverage train/discovery/confirmation={len(train_index)}/{len(discovery_index)}/{len(confirmation_index)}"
    )

feature_mean = np.mean(features_raw[train_index], axis=(0, 1, 2), keepdims=True)
feature_scale = np.maximum(
    np.std(features_raw[train_index], axis=(0, 1, 2), keepdims=True), 1e-3
)
features = ((features_raw - feature_mean) / feature_scale).astype(np.float16)


class CostBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 5), padding=(dilation, 2), dilation=(dilation, 1)),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Dropout2d(0.08),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class AnchoredCostEncoder(nn.Module):
    def __init__(self, input_features, width=32):
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(input_features, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.blocks = nn.Sequential(*(
            CostBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 1)
        ))
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, value):
        encoded = self.candidate(value.float()).permute(0, 3, 1, 2)
        return self.head(self.blocks(encoded)).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnchoredCostEncoder(features.shape[-1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
dataset = TensorDataset(
    torch.from_numpy(features[train_index]),
    torch.from_numpy(targets[train_index]),
    torch.from_numpy(maximum_distance[train_index]),
)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
offset_tensor = torch.from_numpy(OFFSETS).to(device)
grid_tensor = torch.from_numpy(grid.astype(np.float32)).to(device)


def _loss(logits, target, distance_max):
    distance = grid_tensor[None, :] * distance_max[:, None]
    tail = torch.clamp((distance - FREEZE_FT) / RAMP_FT, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    covered = torch.abs(target) <= HALF_WIDTH_FT
    weight = tail * covered.float()
    soft = torch.exp(-0.5 * ((offset_tensor[None, None, :] - target[:, :, None]) / 4.0) ** 2)
    soft = soft / torch.clamp(soft.sum(dim=2, keepdim=True), min=1e-8)
    ce_node = -(soft * F.log_softmax(logits, dim=2)).sum(dim=2)
    ce = torch.sum(weight * ce_node) / torch.clamp(weight.sum(), min=1.0)
    probability = F.softmax(logits, dim=2)
    expected = torch.sum(probability * offset_tensor[None, None, :], dim=2)
    regression = torch.sum(weight * F.smooth_l1_loss(expected / 8.0, target / 8.0, reduction="none"))
    regression = regression / torch.clamp(weight.sum(), min=1.0)
    pair_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    change_error = (expected[:, 1:] - expected[:, :-1]) - (target[:, 1:] - target[:, :-1])
    change = torch.sum(pair_weight * F.smooth_l1_loss(change_error / 4.0, torch.zeros_like(change_error), reduction="none"))
    change = change / torch.clamp(pair_weight.sum(), min=1.0)
    return ce + 0.35 * regression + 0.08 * change


@torch.no_grad()
def _predict(indices):
    model.eval()
    output = []
    for start in range(0, len(indices), BATCH_SIZE):
        index = indices[start:start + BATCH_SIZE]
        logits = model(torch.from_numpy(features[index]).to(device))
        probability = F.softmax(logits, dim=2)
        output.append(torch.sum(probability * offset_tensor[None, None, :], dim=2).cpu().numpy())
    return np.concatenate(output)


best_loss = float("inf")
best_epoch = 0
best_state = None
stale = 0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for batch_features, batch_target, batch_distance in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features.to(device))
        loss = _loss(logits, batch_target.to(device), batch_distance.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        validation_logits = model(torch.from_numpy(features[discovery_index]).to(device))
        validation_loss = float(_loss(
            validation_logits,
            torch.from_numpy(targets[discovery_index]).to(device),
            torch.from_numpy(maximum_distance[discovery_index]).to(device),
        ).cpu())
    row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "discovery_loss": validation_loss}
    history.append(row)
    if validation_loss < best_loss - 1e-5:
        best_loss = validation_loss
        best_epoch = epoch
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stale = 0
    else:
        stale += 1
    if epoch == 1 or epoch % 5 == 0:
        print("V135 EPOCH", row, flush=True)
    if stale >= PATIENCE:
        break
if best_state is None:
    raise RuntimeError("V135 failed to select a checkpoint")
model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

predicted_grids = _predict(np.arange(len(well_order)))
grid_map = dict(zip(well_order, predicted_grids))
fold_map = dict(zip(well_order, fold_by_index))


def _evaluate(wells, recipe, endpoint, source=None):
    dose, cap_ft = recipe
    source = grid_map if source is None else source
    parts = []
    predictions = []
    moves = []
    for well in sorted(wells):
        local = _well_frame(well)
        distance = local["md_since"].to_numpy(np.float64)
        progress = distance / max(float(distance[-1]), 1.0)
        move = np.interp(progress, grid, np.asarray(source[well], np.float64))
        move *= _smoothstep((distance - FREEZE_FT) / RAMP_FT)
        move = np.clip(float(dose) * move, -float(cap_ft), float(cap_ft))
        parts.append(local)
        moves.append(move)
        predictions.append(local[BASE].to_numpy(np.float64) + move)
    matched = pd.concat(parts, ignore_index=True)
    prediction = np.concatenate(predictions)
    move = np.concatenate(moves)
    return R._metric(matched, prediction, endpoint), matched, prediction, move


discovery_wells = [well_order[index] for index in discovery_index]
confirmation_wells = [well_order[index] for index in confirmation_index]
discovery_frame = frame[frame["well"].isin(discovery_wells)].copy()
discovery_base = R._metric(discovery_frame, discovery_frame[BASE].to_numpy(np.float64), BASE)
recipes = [(dose, cap_ft) for dose in DOSES for cap_ft in CAPS_FT]
scan_rows = []
for recipe in recipes:
    metric, _, _, _ = _evaluate(discovery_wells, recipe, "anchored_costvolume")
    scan_rows.append({
        **metric, "dose": recipe[0], "cap_ft": recipe[1],
        "gain": discovery_base["rmse"] - metric["rmse"],
        "far_gain": discovery_base["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "near_degradation": metric["near_0_500_rmse"] - discovery_base["near_0_500_rmse"],
        "p95_gain": discovery_base["p95_well_rmse"] - metric["p95_well_rmse"],
    })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.20) & (scan["far_gain"] >= 0.15)
    & (scan["near_degradation"] <= 1e-4) & (scan["p95_gain"] >= -0.20)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (float(selected_row["dose"]), float(selected_row["cap_ft"]))

summary_rows = [dict(discovery_base, stage="fold3_discovery")]
discovery_candidate, _, _, _ = _evaluate(discovery_wells, selected, "anchored_costvolume")
summary_rows.append(dict(discovery_candidate, stage="fold3_discovery"))
confirmation_values = None
checks = None
if len(eligible):
    confirmation_frame = frame[frame["well"].isin(confirmation_wells)].copy()
    confirmation_base = R._metric(
        confirmation_frame, confirmation_frame[BASE].to_numpy(np.float64), BASE
    )
    candidate, matched, prediction, move = _evaluate(
        confirmation_wells, selected, "anchored_costvolume"
    )
    rolled_values = np.roll(
        np.vstack([grid_map[well] for well in confirmation_wells]),
        max(1, len(confirmation_wells) // 7), axis=0,
    )
    rolled_map = dict(grid_map)
    rolled_map.update(dict(zip(confirmation_wells, rolled_values)))
    rolled, _, rolled_prediction, _ = _evaluate(
        confirmation_wells, selected, "rolled_move_control", rolled_map
    )
    summary_rows.extend([
        dict(confirmation_base, stage="fold3_confirmation"),
        dict(candidate, stage="fold3_confirmation"),
        dict(rolled, stage="fold3_confirmation"),
    ])
    confirmation_values = {
        "gain": confirmation_base["rmse"] - candidate["rmse"],
        "far_gain": confirmation_base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"],
        "near_degradation": candidate["near_0_500_rmse"] - confirmation_base["near_0_500_rmse"],
        "p95_gain": confirmation_base["p95_well_rmse"] - candidate["p95_well_rmse"],
        "aligned_increment_vs_rolled": rolled["rmse"] - candidate["rmse"],
    }
    checks = {
        "gain_ge_0p15": confirmation_values["gain"] >= 0.15,
        "far_gain_ge_0p10": confirmation_values["far_gain"] >= 0.10,
        "near_frozen_exact": abs(confirmation_values["near_degradation"]) <= 1e-4,
        "p95_not_worse_by_gt_0p20": confirmation_values["p95_gain"] >= -0.20,
        "aligned_beats_rolled_ge_0p05": confirmation_values["aligned_increment_vs_rolled"] >= 0.05,
    }
    output = matched[["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
    output["anchored_costvolume"] = prediction.astype(np.float32)
    output["rolled_move_control"] = rolled_prediction.astype(np.float32)
    output["move"] = move.astype(np.float32)
    output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
strict_pass = bool(len(eligible) and checks and all(checks.values()))
torch.save({
    "state_dict": best_state,
    "feature_mean": feature_mean,
    "feature_scale": feature_scale,
    "offsets": OFFSETS,
    "nodes": NODES,
    "best_epoch": best_epoch,
    "recipe": selected,
}, MODEL_DIR / "best_anchored_costvolume.pt")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_dense_costvolume_training_Fold3_discovery_confirmation",
    "base": "exact V127 nested sequence coefficient",
    "sample_unit": "one complete physical well",
    "architecture": "V127-anchored dense GR cost volume plus dilated 2D encoder",
    "candidate_axis": {
        "half_width_ft": HALF_WIDTH_FT, "step_ft": STATE_STEP_FT,
        "states": len(OFFSETS), "nodes": NODES,
    },
    "coverage": {
        "train_wells": len(train_index), "discovery_wells": len(discovery_index),
        "confirmation_wells": len(confirmation_index),
        "mean_target_support": float(np.mean(coverage_rows)),
    },
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "best_epoch": best_epoch,
    "selected_recipe": {
        "freeze_ft": FREEZE_FT, "ramp_ft": RAMP_FT,
        "dose": selected[0], "cap_ft": selected[1],
        "passed_discovery_constraints": bool(len(eligible)),
    },
    "discovery": {
        "gain": selected_row["gain"], "far_gain": selected_row["far_gain"],
        "near_degradation": selected_row["near_degradation"],
        "p95_gain": selected_row["p95_gain"],
    },
    "confirmation": confirmation_values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> three-fold cross-fit the anchored cost-volume correction"
        if strict_pass else
        "FAIL -> close local cost-volume correction and replace the absolute trajectory generator"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV135 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV135 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV135 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
