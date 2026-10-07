"""V133: cross-fit V127 tail-affine head against final pooled TVT SSE."""

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
EXPERIMENT = "direct_sse_tail_affine_crossfit_v133"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V127_SOURCE = ROOT / "final_consensus_sequence_coefficient_crossfit_v127_cell.py"
V127_OOF = OOF_ROOT / "final_consensus_sequence_coefficient_crossfit_v127" / "crossfit_oof.parquet"
FOLDS = (1, 2, 3)
EPOCHS = 8
SEED = 20260801
BATCH_SIZE = 32
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 2e-3
CAP_FT = 4.0
GRID_NODES = 192
COEFFICIENT_SLICE = slice(1, 3)


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


def _bootstrap_v127():
    if not V127_SOURCE.is_file():
        raise FileNotFoundError(V127_SOURCE)
    source = V127_SOURCE.read_text(encoding="utf-8")
    marker = "crossfit_prediction = np.full_like(target_raw, np.nan, dtype=np.float32)"
    if marker not in source:
        raise RuntimeError(f"V127 marker missing from {V127_SOURCE}")
    module = types.ModuleType("v133_v127_definitions")
    module.__file__ = str(V127_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V127_SOURCE), "exec"), module.__dict__)
    return module


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


N = _bootstrap_v127()
Q, R = N.Q, N.R
frame = N.frame
well_order = list(N.well_order)
fold_by_index = N.fold_by_index.astype(np.int8)
sequence_raw = N.sequence_raw.astype(np.float32)
metadata_raw = N.metadata_raw.astype(np.float32)
target_raw = N.target_raw[:, COEFFICIENT_SLICE].astype(np.float32)
if not V127_OOF.is_file():
    raise FileNotFoundError(V127_OOF)

grid = np.linspace(0.0, 1.0, GRID_NODES)
base_grid_rows, truth_grid_rows, basis_rows, row_count_rows = [], [], [], []
for number, well in enumerate(well_order, 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable")
    distance = local["md_since"].to_numpy(np.float64)
    progress = distance / max(float(distance[-1]), 1.0)
    grid_distance = grid * max(float(distance[-1]), 1.0)
    base_grid_rows.append(np.interp(grid, progress, local[Q.BASE].to_numpy(np.float64)))
    truth_grid_rows.append(np.interp(grid, progress, local[Q.TRUTH].to_numpy(np.float64)))
    basis_rows.append(Q._basis("tail_affine", grid_distance))
    row_count_rows.append(len(local))
    if number % 100 == 0 or number == len(well_order):
        print(f"V133 grids {number}/{len(well_order)}", flush=True)

base_grid = np.stack(base_grid_rows).astype(np.float32)
truth_grid = np.stack(truth_grid_rows).astype(np.float32)
basis_grid = np.stack(basis_rows).astype(np.float32)
row_count = np.asarray(row_count_rows, np.float32)


def _fit_predict(heldout_fold):
    train_index = np.flatnonzero(fold_by_index != heldout_fold)
    valid_index = np.flatnonzero(fold_by_index == heldout_fold)
    sequence_mean = np.mean(sequence_raw[train_index], axis=(0, 2), keepdims=True)
    sequence_std = np.maximum(np.std(sequence_raw[train_index], axis=(0, 2), keepdims=True), 1e-3)
    metadata_mean = np.mean(metadata_raw[train_index], axis=0, keepdims=True)
    metadata_std = np.maximum(np.std(metadata_raw[train_index], axis=0, keepdims=True), 1e-3)
    target_mean = np.mean(target_raw[train_index], axis=0, keepdims=True)
    target_std = np.maximum(np.std(target_raw[train_index], axis=0, keepdims=True), 0.25)
    target_low = np.quantile(target_raw[train_index], 0.005, axis=0)
    target_high = np.quantile(target_raw[train_index], 0.995, axis=0)
    sequence = ((sequence_raw - sequence_mean) / sequence_std).astype(np.float32)
    metadata = ((metadata_raw - metadata_mean) / metadata_std).astype(np.float32)
    target_normalized = ((target_raw - target_mean) / target_std).astype(np.float32)

    _seed(SEED + heldout_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Q.SequenceCoefficientEncoder(
        sequence.shape[1], metadata.shape[1], outputs=2
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    dataset = TensorDataset(
        torch.from_numpy(sequence[train_index]),
        torch.from_numpy(metadata[train_index]),
        torch.from_numpy(target_normalized[train_index]),
        torch.from_numpy(base_grid[train_index]),
        torch.from_numpy(truth_grid[train_index]),
        torch.from_numpy(basis_grid[train_index]),
        torch.from_numpy(row_count[train_index]),
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED + heldout_fold),
    )
    mean_tensor = torch.from_numpy(target_mean.astype(np.float32)).to(device)
    std_tensor = torch.from_numpy(target_std.astype(np.float32)).to(device)
    coefficient_loss = nn.SmoothL1Loss(beta=0.75)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses, endpoint_losses, coefficient_losses = [], [], []
        for batch in loader:
            seq, meta, target_n, base, truth, basis, rows = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            predicted_n = model(seq, meta)
            coefficient = predicted_n * std_tensor + mean_tensor
            move = torch.sum(basis * coefficient[:, None, :], dim=2)
            prediction = base + torch.clamp(move, -CAP_FT, CAP_FT)
            well_mse = torch.mean(torch.square(prediction - truth), dim=1)
            endpoint_loss = torch.sum(rows * well_mse) / torch.sum(rows)
            auxiliary = coefficient_loss(predicted_n, target_n)
            loss = endpoint_loss / 25.0 + 0.25 * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            endpoint_losses.append(float(endpoint_loss.detach().cpu()))
            coefficient_losses.append(float(auxiliary.detach().cpu()))
        report = {
            "heldout_fold": heldout_fold, "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_endpoint_mse": float(np.mean(endpoint_losses)),
            "train_coefficient_loss": float(np.mean(coefficient_losses)),
        }
        history.append(report)
        print("V133 EPOCH", report, flush=True)

    model.eval()
    with torch.no_grad():
        predicted_n = model(
            torch.from_numpy(sequence[valid_index]).to(device),
            torch.from_numpy(metadata[valid_index]).to(device),
        ).cpu().numpy()
    coefficient = predicted_n * target_std + target_mean
    coefficient = np.clip(coefficient, target_low, target_high).astype(np.float32)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "sequence_mean": sequence_mean, "sequence_std": sequence_std,
        "metadata_mean": metadata_mean, "metadata_std": metadata_std,
        "target_mean": target_mean, "target_std": target_std,
        "target_low": target_low, "target_high": target_high,
        "heldout_fold": heldout_fold, "epochs": EPOCHS,
    }, MODEL_DIR / f"heldout_fold{heldout_fold}.pt")
    return valid_index, coefficient, history


crossfit_coefficient = np.full((len(well_order), 2), np.nan, np.float32)
history_rows = []
for heldout_fold in FOLDS:
    valid_index, coefficient, history = _fit_predict(heldout_fold)
    crossfit_coefficient[valid_index] = coefficient
    history_rows.extend(history)
if not np.isfinite(crossfit_coefficient).all():
    raise RuntimeError("Incomplete V133 coefficients")
pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False)

coefficient_map = dict(zip(well_order, crossfit_coefficient))
parts, prediction_parts, rolled_parts = [], [], []
for fold in FOLDS:
    fold_wells = sorted(well for well, value in zip(well_order, fold_by_index) if value == fold)
    values = np.vstack([coefficient_map[well] for well in fold_wells])
    rolled = np.roll(values, max(1, len(values) // 7), axis=0)
    for number, well in enumerate(fold_wells):
        local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        basis = Q._basis("tail_affine", local["md_since"].to_numpy(np.float64))
        base = local[Q.BASE].to_numpy(np.float64)
        prediction_parts.append(base + np.clip(basis @ values[number], -CAP_FT, CAP_FT))
        rolled_parts.append(base + np.clip(basis @ rolled[number], -CAP_FT, CAP_FT))
        parts.append(local)
matched = pd.concat(parts, ignore_index=True)
prediction = np.concatenate(prediction_parts)
rolled_prediction = np.concatenate(rolled_parts)

old = pd.read_parquet(V127_OOF)
old = old[["id", "nested_sequence_coefficient"]].copy()
matched = matched.merge(old, on="id", how="left", validate="one_to_one")
if matched["nested_sequence_coefficient"].isna().any():
    raise RuntimeError("Incomplete V127 baseline alignment")

summary_rows = []
for fold in (*FOLDS, 0):
    mask = np.ones(len(matched), bool) if fold == 0 else matched["fold"].to_numpy() == fold
    local = matched.loc[mask]
    old_prediction = matched.loc[mask, "nested_sequence_coefficient"].to_numpy(np.float64)
    summary_rows.extend([
        {"fold": fold, **R._metric(local, old_prediction, "v127")},
        {"fold": fold, **R._metric(local, prediction[mask], "direct_sse_tail_affine")},
        {"fold": fold, **R._metric(local, rolled_prediction[mask], "rolled_coefficient_control")},
    ])
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
overall = summary[summary["fold"].eq(0)].set_index("endpoint")
base = overall.loc["v127"]
candidate = overall.loc["direct_sse_tail_affine"]
control = overall.loc["rolled_coefficient_control"]
fold_gains = {
    str(fold): float(
        summary[(summary["fold"].eq(fold)) & summary["endpoint"].eq("v127")]["rmse"].iloc[0]
        - summary[(summary["fold"].eq(fold)) & summary["endpoint"].eq("direct_sse_tail_affine")]["rmse"].iloc[0]
    ) for fold in FOLDS
}
values = {
    "gain": float(base["rmse"] - candidate["rmse"]),
    "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
    "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
    "aligned_increment_vs_rolled": float(control["rmse"] - candidate["rmse"]),
    "fold_gains": fold_gains,
}
checks = {
    "gain_ge_0p05": values["gain"] >= 0.05,
    "positive_folds_3of3": all(value > 0.0 for value in fold_gains.values()),
    "far_gain_nonnegative": values["far_gain"] >= 0.0,
    "p95_not_worse_by_gt_0p10": values["p95_gain"] >= -0.10,
    "aligned_beats_rolled_ge_0p02": values["aligned_increment_vs_rolled"] >= 0.02,
}
strict_pass = bool(all(checks.values()))
output = matched[["id", "well", "row_idx", "fold", Q.TRUTH, "md_since", Q.BASE, "nested_sequence_coefficient"]].copy()
output["direct_sse_tail_affine"] = prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "crossfit_oof.parquet", index=False, compression="zstd")
coefficient_frame = pd.DataFrame(crossfit_coefficient, columns=["intercept_ft", "slope_ft_per_1000ft"])
coefficient_frame.insert(0, "well", well_order)
coefficient_frame.insert(1, "fold", fold_by_index)
coefficient_frame.to_csv(OUT_DIR / "crossfit_coefficients.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold123_direct_final_SSE_tail_affine_crossfit",
    "base": "exact V127 nested sequence coefficient",
    "changed_variable": "coefficient SmoothL1 objective to row-weighted final TVT SSE plus auxiliary coefficient loss",
    "model_family": "unchanged V127 sequence encoder with two-output tail-affine head",
    "epochs": EPOCHS,
    "coverage": {"rows": len(matched), "wells": int(matched["well"].nunique())},
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train the direct-SSE head on all 464 wells and build deployment"
        if strict_pass else
        "FAIL -> legal sequence evidence cannot predict catastrophic tail coefficients; replace the trajectory model"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV133 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV133 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
