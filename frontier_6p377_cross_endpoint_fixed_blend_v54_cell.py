"""Nested fixed convex blend of the independent V51 endpoints.

V52/V53 showed that endpoint correctness is not reliably observable per well.
This final endpoint-bank test asks the simpler covariance question: does one
global, low-dose convex blend improve every held fold?  Each outer-fold weight
is selected using only the other two folds.  No learned selector is involved.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_cross_endpoint_fixed_blend_v54"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V52_SOURCE = ROOT / "frontier_6p377_cross_endpoint_wholewell_selector_v52_cell.py"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = (1, 2, 3)
MAX_CANDIDATE_DOSE = 0.40
WEIGHT_STEP = 0.025
MINIMUM_INNER_FOLD_GAIN = 0.005
started = time.time()


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


if not V52_SOURCE.is_file():
    raise FileNotFoundError(V52_SOURCE)
source = V52_SOURCE.read_text(encoding="utf-8")
marker = "\ndef _fit_model(local, seed):"
if marker not in source:
    raise RuntimeError("V52 definition marker not found")
V = {"__name__": "v52_fixed_blend_builder", "__file__": str(V52_SOURCE)}
exec(compile(source.split(marker, 1)[0], str(V52_SOURCE), "exec"), V)

frame = V["frame"]
candidate_names = list(V["candidate_names"])
candidate_matrix = np.asarray(V["candidate_matrix"], np.float64)
base = np.asarray(V["base"], np.float64)
truth = np.asarray(V["truth"], np.float64)
distance = np.asarray(V["distance"], np.float64)
fold_values = np.asarray(V["fold_values"], np.int8)
frozen = np.asarray(V["frozen"], bool)
_metrics = V["_metrics"]

if candidate_matrix.shape[1] != 4:
    raise RuntimeError(f"Expected four endpoint states, got {candidate_names}")

errors = candidate_matrix - truth[:, None]
errors[frozen] = (base[frozen] - truth[frozen])[:, None]


def _gram(mask):
    local = errors[np.asarray(mask, bool)]
    return (local.T @ local) / float(len(local))


def _rmse_from_gram(weights, gram):
    weights = np.asarray(weights, np.float64)
    return float(np.sqrt(max(float(weights @ gram @ weights), 0.0)))


levels = np.arange(0.0, MAX_CANDIDATE_DOSE + WEIGHT_STEP / 2.0, WEIGHT_STEP)
weight_grid = []
for psg, common, track8 in product(levels, repeat=3):
    candidate_dose = float(psg + common + track8)
    if candidate_dose <= MAX_CANDIDATE_DOSE + 1e-12:
        weight_grid.append(np.array([1.0 - candidate_dose, psg, common, track8], np.float64))
weight_grid = np.asarray(weight_grid)
print(
    "V54 CONFIG",
    {"candidates": candidate_names, "weight_vectors": len(weight_grid), "step": WEIGHT_STEP},
    flush=True,
)

fold_grams = {fold: _gram(fold_values == fold) for fold in FOLDS}
fold_far_grams = {
    fold: _gram((fold_values == fold) & (distance > 2000.0)) for fold in FOLDS
}
fold_base_rmse = {fold: _rmse_from_gram([1, 0, 0, 0], fold_grams[fold]) for fold in FOLDS}

oof_prediction = base.copy()
fold_reports = []
scan_rows = []
for held_fold in FOLDS:
    train_folds = [fold for fold in FOLDS if fold != held_fold]
    train_mask = np.isin(fold_values, train_folds)
    train_gram = _gram(train_mask)
    candidates = []
    for weights in weight_grid:
        train_rmse = _rmse_from_gram(weights, train_gram)
        fold_gains = [
            fold_base_rmse[fold] - _rmse_from_gram(weights, fold_grams[fold])
            for fold in train_folds
        ]
        row = {
            "held_fold": held_fold,
            "weight_base": float(weights[0]),
            "weight_psg": float(weights[1]),
            "weight_track8_common": float(weights[2]),
            "weight_track8": float(weights[3]),
            "inner_rmse": train_rmse,
            "inner_gain": _rmse_from_gram([1, 0, 0, 0], train_gram) - train_rmse,
            "minimum_inner_fold_gain": float(min(fold_gains)),
        }
        candidates.append(row)
    scan = pd.DataFrame(candidates)
    safe = scan[scan["minimum_inner_fold_gain"] >= MINIMUM_INNER_FOLD_GAIN]
    selected = (safe if len(safe) else scan).sort_values(
        ["inner_rmse", "minimum_inner_fold_gain"],
        ascending=[True, False], kind="stable",
    ).iloc[0]
    selected_weights = selected[
        ["weight_base", "weight_psg", "weight_track8_common", "weight_track8"]
    ].to_numpy(np.float64)
    held_mask = fold_values == held_fold
    prediction = candidate_matrix @ selected_weights
    prediction[frozen] = base[frozen]
    oof_prediction[held_mask] = prediction[held_mask]

    base_metrics = _metrics(base, "base", held_mask)
    candidate_metrics = _metrics(prediction, "fixed_blend", held_mask)
    report = {
        "fold": held_fold,
        **{key: float(selected[key]) for key in (
            "weight_base", "weight_psg", "weight_track8_common", "weight_track8",
            "inner_rmse", "inner_gain", "minimum_inner_fold_gain",
        )},
        "base_rmse": base_metrics["rmse"],
        "rmse": candidate_metrics["rmse"],
        "gain": base_metrics["rmse"] - candidate_metrics["rmse"],
        "base_far_rmse": base_metrics["far_gt2000_rmse"],
        "far_rmse": candidate_metrics["far_gt2000_rmse"],
        "far_gain": base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"],
        "p95_change": candidate_metrics["p95_well_rmse"] - base_metrics["p95_well_rmse"],
    }
    fold_reports.append(report)
    scan_rows.append(scan)
    print("V54 FOLD", report, flush=True)

oof_prediction[frozen] = base[frozen]
base_metrics = _metrics(base, "base")
blend_metrics = _metrics(oof_prediction, "nested_fixed_blend")
fold_table = pd.DataFrame(fold_reports)
checks = {
    "overall_gain_ge_0p05": bool(base_metrics["rmse"] - blend_metrics["rmse"] >= 0.05),
    "far_gain_ge_0p075": bool(base_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"] >= 0.075),
    "minimum_fold_gain_ge_0": bool(fold_table["gain"].min() >= 0.0),
    "minimum_fold_far_gain_ge_0": bool(fold_table["far_gain"].min() >= 0.0),
    "maximum_p95_change_le_0p20": bool(fold_table["p95_change"].max() <= 0.20),
    "near_rows_bitwise_unchanged": bool(np.array_equal(oof_prediction[frozen], base[frozen])),
}
strict_pass = bool(all(checks.values()))

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["base_6p377"] = base.astype(np.float32)
output["nested_fixed_blend"] = oof_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fixed_blend_oof.parquet", index=False, compression="zstd")
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
pd.concat(scan_rows, ignore_index=True).to_csv(OUT_DIR / "inner_scan.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_global_convex_endpoint_blend",
    "sample_unit": "fixed endpoint weights selected on two folds and applied to the third",
    "source": str(V52_SOURCE),
    "candidates": candidate_names,
    "weight_step": WEIGHT_STEP,
    "maximum_candidate_dose": MAX_CANDIDATE_DOSE,
    "base": base_metrics,
    "blend": blend_metrics,
    "gain": base_metrics["rmse"] - blend_metrics["rmse"],
    "far_gain": base_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"],
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> refit one robust fixed weight vector on folds 1/2/3 and package a conservative Kaggle candidate"
        if strict_pass else
        "FAIL -> close the existing cross-endpoint bank; generate a genuinely new endpoint before further blending"
    ),
}
_write_json(OUT_DIR / "result.json", verdict)

print("\nV54 FOLDS", flush=True)
print(fold_table.to_string(index=False), flush=True)
print("\nV54 FINAL VERDICT", flush=True)
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
