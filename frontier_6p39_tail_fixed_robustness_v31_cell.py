"""Retrospective robustness scan for the frozen V28 CP8 tail endpoint.

All three available folds have already been observed. This script therefore
does not claim nested confirmation. It answers the narrower deployment
question: is there a conservative fixed dose that improves every observed
fold, improves every far tail, and keeps per-well p95 risk controlled?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
SOURCE_DIR = ROOT / "oof_exports" / "frontier_6p39_tail_endpoint_oracle_v28"
SOURCE_PATH = SOURCE_DIR / "tail_endpoint_oof.parquet"
OUT_DIR = ROOT / "oof_exports" / "frontier_6p39_tail_fixed_robustness_v31"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = np.round(np.arange(0.0, 0.401, 0.025), 3)
FOLDS = (3, 2, 1)
P95_TOLERANCE_FT = 0.10
MIN_EACH_FOLD_GAIN = 0.02
MIN_EACH_FAR_GAIN = 0.02

if not SOURCE_PATH.is_file():
    raise FileNotFoundError(SOURCE_PATH)

frame = pd.read_parquet(SOURCE_PATH).sort_values(
    ["well", "row_idx"], kind="stable"
).reset_index(drop=True)
required = {
    "well", "fold", "tvt_true", "md_since", "neighbor_v27",
    "tail_cp8_endpoint",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V28 output missing columns: {missing}")

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["neighbor_v27"].to_numpy(np.float64)
endpoint = frame["tail_cp8_endpoint"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold_values = frame["fold"].to_numpy(np.int8)
well_values = frame["well"].astype(str).to_numpy()
direction = endpoint - base


def metrics(mask: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction[mask] - truth[mask]
    local_wells = well_values[mask]
    per_well = (
        pd.DataFrame({"well": local_wells, "error2": error * error})
        .groupby("well", sort=False)["error2"].mean().pow(0.5)
    )
    local_distance = distance[mask]
    far = local_distance > 2000.0
    near = local_distance <= 1000.0
    return {
        "rows": int(mask.sum()),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


base_metrics = {}
for fold in FOLDS:
    mask = fold_values == fold
    base_metrics[fold] = metrics(mask, base)

scan_rows = []
for weight in WEIGHTS:
    prediction = base + float(weight) * direction
    fold_rows = []
    for fold in FOLDS:
        mask = fold_values == fold
        result = metrics(mask, prediction)
        reference = base_metrics[fold]
        fold_rows.append({
            "weight": float(weight),
            "fold": int(fold),
            "rows": result["rows"],
            "wells": result["wells"],
            "rmse": result["rmse"],
            "gain": reference["rmse"] - result["rmse"],
            "far_rmse": result["far_gt2000_rmse"],
            "far_gain": reference["far_gt2000_rmse"] - result["far_gt2000_rmse"],
            "p95": result["p95_well_rmse"],
            "p95_change": result["p95_well_rmse"] - reference["p95_well_rmse"],
            "near_change": result["near_le1000_rmse"] - reference["near_le1000_rmse"],
        })
    gains = np.array([row["gain"] for row in fold_rows])
    far_gains = np.array([row["far_gain"] for row in fold_rows])
    p95_changes = np.array([row["p95_change"] for row in fold_rows])
    near_changes = np.array([row["near_change"] for row in fold_rows])
    robust = bool(
        gains.min() >= MIN_EACH_FOLD_GAIN
        and far_gains.min() >= MIN_EACH_FAR_GAIN
        and p95_changes.max() <= P95_TOLERANCE_FT
        and np.max(np.abs(near_changes)) <= 1e-8
    )
    for row in fold_rows:
        row.update({
            "mean_gain": float(gains.mean()),
            "minimum_gain": float(gains.min()),
            "mean_far_gain": float(far_gains.mean()),
            "minimum_far_gain": float(far_gains.min()),
            "maximum_p95_change": float(p95_changes.max()),
            "robust": robust,
        })
        scan_rows.append(row)

scan = pd.DataFrame(scan_rows)
summary = (
    scan.groupby("weight", as_index=False)
    .agg(
        mean_gain=("gain", "mean"),
        minimum_gain=("gain", "min"),
        mean_far_gain=("far_gain", "mean"),
        minimum_far_gain=("far_gain", "min"),
        maximum_p95_change=("p95_change", "max"),
        maximum_abs_near_change=("near_change", lambda x: float(np.max(np.abs(x)))),
        robust=("robust", "all"),
    )
)

eligible = summary[summary["robust"]].copy()
if len(eligible):
    # Maximin first; mean gain only breaks ties. This intentionally favors a
    # stable low dose over the best average retrospective score.
    selected_row = eligible.sort_values(
        ["minimum_gain", "mean_gain", "weight"],
        ascending=[False, False, True],
    ).iloc[0]
else:
    selected_row = summary.sort_values(
        ["maximum_p95_change", "minimum_gain"], ascending=[True, False]
    ).iloc[0]

selected_weight = float(selected_row["weight"])
selected_prediction = base + selected_weight * direction
selected_metrics = {
    str(fold): metrics(fold_values == fold, selected_prediction) for fold in FOLDS
}

summary.to_csv(OUT_DIR / "dose_summary.csv", index=False)
scan.to_csv(OUT_DIR / "dose_by_fold.csv", index=False)
frame.loc[:, ["well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    neighbor_v27=base,
    tail_cp8_endpoint=endpoint,
    selected_tail_robust=selected_prediction,
).to_parquet(OUT_DIR / "tail_robust_oof.parquet", index=False)

print("TAIL FIXED ROBUSTNESS SCAN")
print(summary.to_string(index=False), flush=True)
print("\nSELECTED BY FOLD")
print(
    scan[scan["weight"].eq(selected_weight)].to_string(index=False),
    flush=True,
)

strict_pass = bool(selected_row["robust"])
verdict = {
    "experiment": "frontier_6p39_tail_fixed_robustness_v31",
    "stage": "retrospective_threefold_fixed_tail_dose_robustness",
    "source": str(SOURCE_PATH),
    "validation_status": (
        "retrospective_only_all_available_folds_already_observed"
    ),
    "selected_weight": selected_weight,
    "selection_rule": "maximize minimum fold gain among p95-safe fixed doses",
    "thresholds": {
        "minimum_each_fold_gain": MIN_EACH_FOLD_GAIN,
        "minimum_each_far_gain": MIN_EACH_FAR_GAIN,
        "maximum_p95_change": P95_TOLERANCE_FT,
        "near_le1000_change_abs_le": 1e-8,
    },
    "selected_summary": {
        key: (bool(value) if key == "robust" else float(value))
        for key, value in selected_row.to_dict().items()
    },
    "selected_metrics": selected_metrics,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> package as a conservative LB-only candidate; no CV claim remains"
        if strict_pass else
        "FAIL -> even a conservative fixed dose is not tail-safe; close endpoint"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
