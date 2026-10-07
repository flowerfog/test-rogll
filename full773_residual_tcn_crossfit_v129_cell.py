"""V129: five-fold cross-fit full-trajectory residual TCN on all 773 wells.

The exact honest V76 baseline is reconstructed from the completed V96 coarse
OOF and the original V76 dose model. Each held-out fold is predicted by a TCN
trained on the other four folds. Consumption recipes are likewise selected
only from the other four cross-fit folds.
"""

from __future__ import annotations

import json
import random
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "full773_residual_tcn_crossfit_v129"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V126_SOURCE = ROOT / "final_consensus_sequence_coefficient_v126_cell.py"
V96_SOURCE = ROOT / "rawdata_lattice_v76_full773_training_v96_cell.py"
V96_COARSE = OOF_ROOT / "rawdata_lattice_v76_full773_training_v96" / "fivefold_coarse_oof.parquet"
FOLDS = (1, 2, 3, 4, 5)
NODES = 192
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 3e-3
TARGET_SCALE_FT = 8.0
RAMP_FT = 250.0
SEED = 20260801
RECIPES = tuple(
    (freeze_ft, dose, cap_ft)
    for freeze_ft in (500.0, 1000.0)
    for dose in (0.10, 0.25, 0.50, 0.75, 1.00)
    for cap_ft in (2.0, 4.0, 8.0)
)


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


def _bootstrap(path, marker, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Definition marker missing from {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
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
    kernel = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1], np.float64)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, (4, 4), mode="edge"), kernel, mode="valid")


Q = _bootstrap(
    V126_SOURCE,
    "for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):",
    "v129_v126_definitions",
)
M = _bootstrap(
    V96_SOURCE,
    "for path in (V4_SOURCE, V66_PATH, V70_PATH, V76_BUNDLE, V76_OOF_WEIGHTS):",
    "v129_v96_definitions",
)
R = Q._bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

for required in (V96_COARSE, M.V76_BUNDLE, M.V76_OOF_WEIGHTS):
    if not Path(required).is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(V96_COARSE).sort_values(
    ["fold", "well", "row_idx"], kind="stable"
).reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame["id"].duplicated().any():
    raise RuntimeError(
        f"Invalid V96 coarse OOF rows={len(frame)} wells={frame['well'].nunique()}"
    )

# Reconstruct the exact V76 baseline used in the complete five-fold comparison.
well_features = pd.DataFrame([
    M._feature_row(local) for _, local in frame.groupby(["fold", "well"], sort=False)
]).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
x, feature_names = M._augment(well_features)
bundle = joblib.load(M.V76_BUNDLE)
weights = np.clip(
    bundle["hist_fraction"] * bundle["hist_model"].predict(x)
    + bundle["ridge_fraction"] * bundle["ridge_model"].predict(x),
    M.MIN_WEIGHT,
    M.MAX_WEIGHT,
)
old_oof = pd.read_parquet(
    M.V76_OOF_WEIGHTS, columns=["fold", "well", "ensemble_weight"]
)
old_map = dict(zip(old_oof["well"].astype(str), old_oof["ensemble_weight"].astype(float)))
honest = well_features["fold"].isin(M.PRIOR_FOLDS).to_numpy()
weights[honest] = (
    well_features.loc[honest, "well"].astype(str).map(old_map).to_numpy(np.float64)
)
if not np.isfinite(weights).all():
    raise RuntimeError("Incomplete V76 baseline weights")
weight_map = dict(zip(well_features["well"].astype(str), weights))
row_weight = frame["well"].map(weight_map).to_numpy(np.float64)
parent = frame[M.PARENT].to_numpy(np.float64)
coarse = frame[M.COARSE].to_numpy(np.float64)
frame[Q.BASE] = (parent + row_weight * (coarse - parent)).astype(np.float32)
frame["exact_v76"] = frame[Q.BASE]
if not np.isfinite(frame[Q.BASE]).all():
    raise RuntimeError("Non-finite exact V76 reconstruction")
base_parity = M._metric(frame, frame[Q.BASE].to_numpy(np.float64), "v76_reconstructed")
if abs(float(base_parity["rmse"]) - 7.9510830730808735) > 2e-4:
    raise RuntimeError(f"V76 reconstruction parity failed: {base_parity['rmse']}")
print("V129 V76 PARITY", base_parity, flush=True)

well_positions = {
    str(well): np.asarray(positions, np.int64)
    for well, positions in frame.groupby("well", sort=False).indices.items()
}


def _well_frame(well):
    return frame.iloc[well_positions[str(well)]]


well_order = sorted(well_positions)
grid = np.linspace(0.0, 1.0, NODES)
sequence_rows = []
metadata_rows = []
target_rows = []
distance_rows = []
fold_rows = []
for number, well in enumerate(well_order, 1):
    local = _well_frame(well)
    record = Q._load_record_relaxed(V, well)
    sequence, metadata = Q._sequence_features(local, record)
    distance = local["md_since"].to_numpy(np.float64)
    progress = distance / max(float(distance[-1]), 1.0)
    residual = local[M.TRUTH].to_numpy(np.float64) - local[Q.BASE].to_numpy(np.float64)
    sequence_rows.append(sequence)
    metadata_rows.append(metadata)
    target_rows.append(_smooth(np.interp(grid, progress, residual)).astype(np.float32))
    distance_rows.append(float(distance[-1]))
    fold_rows.append(int(local["fold"].iloc[0]))
    if number % 100 == 0 or number == len(well_order):
        print(f"V129 sequences {number}/{len(well_order)}", flush=True)

sequence_raw = np.stack(sequence_rows).astype(np.float32)
metadata_raw = np.stack(metadata_rows).astype(np.float32)
target_raw = np.stack(target_rows).astype(np.float32)
distance_max = np.asarray(distance_rows, np.float32)
fold_by_index = np.asarray(fold_rows, np.int8)
if len(well_order) != 773 or set(fold_by_index.tolist()) != set(FOLDS):
    raise RuntimeError(f"Unexpected V129 coverage wells={len(well_order)} folds={set(fold_by_index)}")


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
        value = torch.cat([sequence_value, meta], dim=1)
        return self.head(self.blocks(self.stem(value))).squeeze(1)


def _fit_predict(heldout_fold):
    train_index = np.flatnonzero(fold_by_index != heldout_fold)
    valid_index = np.flatnonzero(fold_by_index == heldout_fold)
    sequence_mean = np.mean(sequence_raw[train_index], axis=(0, 2), keepdims=True)
    sequence_std = np.maximum(
        np.std(sequence_raw[train_index], axis=(0, 2), keepdims=True), 1e-3
    )
    metadata_mean = np.mean(metadata_raw[train_index], axis=0, keepdims=True)
    metadata_std = np.maximum(np.std(metadata_raw[train_index], axis=0, keepdims=True), 1e-3)
    sequence = ((sequence_raw - sequence_mean) / sequence_std).astype(np.float32)
    metadata = ((metadata_raw - metadata_mean) / metadata_std).astype(np.float32)
    target = np.clip(target_raw / TARGET_SCALE_FT, -6.0, 6.0).astype(np.float32)

    _seed_everything(SEED + heldout_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualTCN(sequence.shape[1], metadata.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    dataset = TensorDataset(
        torch.from_numpy(sequence[train_index]),
        torch.from_numpy(metadata[train_index]),
        torch.from_numpy(target[train_index]),
        torch.from_numpy(distance_max[train_index]),
    )
    generator = torch.Generator().manual_seed(SEED + heldout_fold)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    grid_tensor = torch.from_numpy(grid.astype(np.float32)).to(device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch_sequence, batch_metadata, batch_target, batch_distance in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_sequence.to(device), batch_metadata.to(device))
            distance = grid_tensor[None, :] * batch_distance.to(device)[:, None]
            taper = torch.clamp(distance / 250.0, 0.0, 1.0)
            taper = taper * taper * (3.0 - 2.0 * taper)
            prediction = prediction * taper
            point = nn.functional.smooth_l1_loss(
                prediction, batch_target.to(device), beta=0.75
            )
            first = torch.mean(torch.abs(prediction[:, 1:] - prediction[:, :-1]))
            second = torch.mean(torch.abs(
                prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
            ))
            loss = point + 0.015 * first + 0.05 * second
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"heldout_fold": heldout_fold, "epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print("V129 EPOCH", row, flush=True)
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(sequence[valid_index]).to(device),
            torch.from_numpy(metadata[valid_index]).to(device),
        ).cpu().numpy()
    prediction = np.clip(prediction * TARGET_SCALE_FT, -24.0, 24.0).astype(np.float32)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "sequence_mean": sequence_mean,
        "sequence_std": sequence_std,
        "metadata_mean": metadata_mean,
        "metadata_std": metadata_std,
        "target_scale_ft": TARGET_SCALE_FT,
        "heldout_fold": heldout_fold,
        "epochs": EPOCHS,
        "feature_names": feature_names,
    }, MODEL_DIR / f"heldout_fold{heldout_fold}.pt")
    report = {
        "heldout_fold": heldout_fold,
        "train_wells": len(train_index),
        "valid_wells": len(valid_index),
        "target_grid_rmse": float(np.sqrt(np.mean(np.square(
            prediction - target_raw[valid_index]
        )))),
    }
    print("V129 MODEL", report, flush=True)
    return valid_index, prediction, history, report


crossfit_grid = np.full_like(target_raw, np.nan, dtype=np.float32)
history_rows = []
model_reports = []
for heldout_fold in FOLDS:
    valid_index, prediction, history, report = _fit_predict(heldout_fold)
    crossfit_grid[valid_index] = prediction
    history_rows.extend(history)
    model_reports.append(report)
if not np.isfinite(crossfit_grid).all():
    raise RuntimeError("Incomplete V129 cross-fit trajectories")
pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False)
grid_map = dict(zip(well_order, crossfit_grid))


def _taper(distance, freeze_ft):
    return Q._smoothstep((np.asarray(distance, np.float64) - freeze_ft) / RAMP_FT)


def _evaluate(folds, recipe, endpoint, grid_override=None):
    fold_set = set(int(value) for value in folds)
    source = grid_map if grid_override is None else grid_override
    freeze_ft, dose, cap_ft = recipe
    local_parts = []
    prediction_parts = []
    for well, fold in zip(well_order, fold_by_index):
        if int(fold) not in fold_set:
            continue
        local = _well_frame(well)
        distance = local["md_since"].to_numpy(np.float64)
        progress = distance / max(float(distance[-1]), 1.0)
        move = np.interp(progress, grid, np.asarray(source[well], np.float64))
        move *= _taper(distance, freeze_ft)
        move = np.clip(float(dose) * move, -float(cap_ft), float(cap_ft))
        local_parts.append(local)
        prediction_parts.append(local[Q.BASE].to_numpy(np.float64) + move)
    matched = pd.concat(local_parts, ignore_index=True)
    prediction = np.concatenate(prediction_parts)
    return R._metric(matched, prediction, endpoint), matched, prediction


def _base_metric(folds):
    local = frame[frame["fold"].isin(list(folds))]
    return R._metric(local, local[Q.BASE].to_numpy(np.float64), Q.BASE)


scan_rows = []
selected_by_fold = {}
base_by_fold = {fold: _base_metric((fold,)) for fold in FOLDS}
far_rows_by_fold = {
    fold: int(np.sum(
        (frame["fold"].to_numpy(np.int8) == fold)
        & (frame["md_since"].to_numpy(np.float64) > 2000.0)
    ))
    for fold in FOLDS
}
metric_cache = {}
for fold in FOLDS:
    for recipe in RECIPES:
        metric_cache[(fold, recipe)] = _evaluate(
            (fold,), recipe, "residual_tcn"
        )[0]

for heldout_fold in FOLDS:
    selection_folds = tuple(fold for fold in FOLDS if fold != heldout_fold)
    selection_rows = sum(base_by_fold[fold]["rows"] for fold in selection_folds)
    selection_base_sse = sum(base_by_fold[fold]["exact_sse"] for fold in selection_folds)
    selection_base_rmse = float(np.sqrt(selection_base_sse / selection_rows))
    selection_far_rows = sum(far_rows_by_fold[fold] for fold in selection_folds)
    selection_base_far_sse = sum(
        far_rows_by_fold[fold] * base_by_fold[fold]["far_gt2000_rmse"] ** 2
        for fold in selection_folds
    )
    selection_base_far = float(np.sqrt(selection_base_far_sse / selection_far_rows))
    candidates = []
    for recipe in RECIPES:
        fold_results = [metric_cache[(fold, recipe)] for fold in selection_folds]
        candidate_sse = sum(result["exact_sse"] for result in fold_results)
        candidate_rmse = float(np.sqrt(candidate_sse / selection_rows))
        candidate_far_sse = sum(
            far_rows_by_fold[fold] * metric_cache[(fold, recipe)]["far_gt2000_rmse"] ** 2
            for fold in selection_folds
        )
        candidate_far = float(np.sqrt(candidate_far_sse / selection_far_rows))
        fold_gains = [
            base_by_fold[fold]["rmse"] - metric_cache[(fold, recipe)]["rmse"]
            for fold in selection_folds
        ]
        fold_p95_gains = [
            base_by_fold[fold]["p95_well_rmse"] - metric_cache[(fold, recipe)]["p95_well_rmse"]
            for fold in selection_folds
        ]
        row = {
            "endpoint": "residual_tcn",
            "rows": selection_rows,
            "wells": sum(base_by_fold[fold]["wells"] for fold in selection_folds),
            "exact_sse": candidate_sse,
            "rmse": candidate_rmse,
            "far_gt2000_rmse": candidate_far,
            "heldout_fold": heldout_fold,
            "selection_folds": "".join(str(value) for value in selection_folds),
            "freeze_ft": recipe[0], "dose": recipe[1], "cap_ft": recipe[2],
            "gain": float(selection_base_rmse - candidate_rmse),
            "far_gain": float(selection_base_far - candidate_far),
            "min_p95_gain": float(min(fold_p95_gains)),
            "positive_selection_folds": int(sum(value > 0.0 for value in fold_gains)),
            "min_fold_gain": float(min(fold_gains)),
        }
        candidates.append(row)
        scan_rows.append(row)
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["rmse", "min_p95_gain"], ascending=[True, False], kind="stable"
    )
    eligible = candidate_frame[
        (candidate_frame["gain"] >= 0.10)
        & (candidate_frame["far_gain"] >= 0.05)
        & (candidate_frame["min_p95_gain"] >= -0.20)
        & (candidate_frame["positive_selection_folds"] >= 3)
    ]
    selected_row = (eligible if len(eligible) else candidate_frame).iloc[0]
    selected_by_fold[heldout_fold] = {
        "recipe": (
            float(selected_row["freeze_ft"]),
            float(selected_row["dose"]),
            float(selected_row["cap_ft"]),
        ),
        "selection_pass": bool(len(eligible)),
        "selection_gain": float(selected_row["gain"]),
        "selection_far_gain": float(selected_row["far_gain"]),
        "selection_min_p95_gain": float(selected_row["min_p95_gain"]),
        "positive_selection_folds": int(selected_row["positive_selection_folds"]),
    }
pd.DataFrame(scan_rows).to_csv(OUT_DIR / "nested_recipe_scan.csv", index=False)

summary_rows = []
matched_parts = []
prediction_parts = []
rolled_parts = []
fold_values = {}
for heldout_fold in FOLDS:
    recipe = selected_by_fold[heldout_fold]["recipe"]
    base = _base_metric((heldout_fold,))
    candidate, matched, prediction = _evaluate(
        (heldout_fold,), recipe, "nested_residual_tcn"
    )
    wells = [well for well, fold in zip(well_order, fold_by_index) if fold == heldout_fold]
    rolled_values = np.roll(
        np.vstack([grid_map[well] for well in wells]), max(1, len(wells) // 7), axis=0
    )
    rolled_map = dict(grid_map)
    for well, value in zip(wells, rolled_values):
        rolled_map[well] = value
    rolled, _, rolled_prediction = _evaluate(
        (heldout_fold,), recipe, "rolled_residual_control", rolled_map
    )
    summary_rows.extend([
        dict(base, fold=heldout_fold),
        dict(candidate, fold=heldout_fold),
        dict(rolled, fold=heldout_fold),
    ])
    matched_parts.append(matched)
    prediction_parts.append(prediction)
    rolled_parts.append(rolled_prediction)
    fold_values[str(heldout_fold)] = {
        "recipe": recipe,
        "selection_pass": selected_by_fold[heldout_fold]["selection_pass"],
        "gain": float(base["rmse"] - candidate["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
        "aligned_increment_vs_rolled": float(rolled["rmse"] - candidate["rmse"]),
    }

matched = pd.concat(matched_parts, ignore_index=True)
prediction = np.concatenate(prediction_parts)
rolled_prediction = np.concatenate(rolled_parts)
pooled_base = R._metric(matched, matched[Q.BASE].to_numpy(np.float64), Q.BASE)
pooled_candidate = R._metric(matched, prediction, "nested_residual_tcn")
pooled_rolled = R._metric(matched, rolled_prediction, "rolled_residual_control")
summary_rows.extend([
    dict(pooled_base, fold=0),
    dict(pooled_candidate, fold=0),
    dict(pooled_rolled, fold=0),
])
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

output = matched[["id", "well", "row_idx", "fold", M.TRUTH, "md_since", Q.BASE]].copy()
output["nested_residual_tcn"] = prediction.astype(np.float32)
output["rolled_residual_control"] = rolled_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "crossfit_oof.parquet", index=False, compression="zstd")

pooled = {
    "base_rmse": float(pooled_base["rmse"]),
    "rmse": float(pooled_candidate["rmse"]),
    "gain": float(pooled_base["rmse"] - pooled_candidate["rmse"]),
    "far_gain": float(pooled_base["far_gt2000_rmse"] - pooled_candidate["far_gt2000_rmse"]),
    "p95_gain": float(pooled_base["p95_well_rmse"] - pooled_candidate["p95_well_rmse"]),
    "aligned_increment_vs_rolled": float(pooled_rolled["rmse"] - pooled_candidate["rmse"]),
}
checks = {
    "pooled_gain_ge_0p25": pooled["gain"] >= 0.25,
    "positive_folds_ge_4of5": sum(value["gain"] > 0.0 for value in fold_values.values()) >= 4,
    "far_gain_ge_0p20": pooled["far_gain"] >= 0.20,
    "p95_not_worse_by_gt_0p20": pooled["p95_gain"] >= -0.20,
    "aligned_beats_rolled_ge_0p05": pooled["aligned_increment_vs_rolled"] >= 0.05,
    "all_nested_selections_pass": all(value["selection_pass"] for value in fold_values.values()),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_773well_fivefold_residual_trajectory_crossfit",
    "sample_unit": "one complete physical well",
    "base": "exact honest V76 reconstructed from V96 coarse OOF",
    "architecture": "anchored_multiscale_dilated_TCN_sequence_to_sequence_residual",
    "fixed_epochs": EPOCHS,
    "coverage": {"rows": len(matched), "wells": int(matched["well"].nunique())},
    "model_reports": model_reports,
    "nested_recipes": selected_by_fold,
    "fold_values": fold_values,
    "pooled": pooled,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train one 773-well deployment TCN and test after V76"
        if strict_pass else
        "FAIL -> residual trajectory is not learnable at useful scale; replace the raw trajectory generator"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV129 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV129 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
