"""V135b: read-only dose attribution for the V135 confirmation move.

This is not a deployable selection step.  Fold3 confirmation truth is used to
measure whether V135 failed because its move was over-dosed or because its
direction was wrong.  No model is trained and no new fold is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE_DIR = ROOT / "oof_exports" / "anchored_v127_costvolume_residual_v135"
SOURCE = SOURCE_DIR / "fold3_confirmation_oof.parquet"
OUT_DIR = ROOT / "oof_exports" / "anchored_v127_costvolume_dose_audit_v135b"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE = "nested_sequence_coefficient"
TRUTH = "tvt_true"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - frame[TRUTH].to_numpy(np.float64)
    distance = frame["md_since"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
    well_rmse = np.sqrt(bywell.groupby("well", sort=False)["error2"].mean())
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE)
required = {"id", "well", "row_idx", TRUTH, "md_since", BASE, "move"}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V135 confirmation OOF missing columns: {missing}")
if frame["id"].duplicated().any() or frame[list(required)].isna().any().any():
    raise RuntimeError("Invalid V135 confirmation OOF")

base = frame[BASE].to_numpy(np.float64)
truth = frame[TRUTH].to_numpy(np.float64)
move = frame["move"].to_numpy(np.float64)
error = base - truth
denominator = float(np.dot(move, move))
if denominator <= 0.0:
    raise RuntimeError("V135 move has zero energy")
analytic_alpha = float(-np.dot(error, move) / denominator)

alphas = np.unique(np.concatenate([
    np.arange(-1.0, 1.0001, 0.025),
    np.asarray([0.0, 0.25, 0.50, 0.75, 1.0, analytic_alpha]),
]))
rows = []
base_metric = _metric(frame, base, "v127")
for alpha in alphas:
    prediction = base + float(alpha) * move
    metric = _metric(frame, prediction, "scaled_v135_move")
    metric.update({
        "alpha": float(alpha),
        "gain": float(base_metric["rmse"] - metric["rmse"]),
        "far_gain": float(base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"]),
        "p95_gain": float(base_metric["p95_well_rmse"] - metric["p95_well_rmse"]),
    })
    rows.append(metric)
scan = pd.DataFrame(rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "dose_scan.csv", index=False)
best = scan.iloc[0]

well_rows = []
for well, local in frame.groupby("well", sort=True):
    local_error = local[BASE].to_numpy(np.float64) - local[TRUTH].to_numpy(np.float64)
    local_move = local["move"].to_numpy(np.float64)
    local_denominator = float(np.dot(local_move, local_move))
    alpha = (
        float(-np.dot(local_error, local_move) / local_denominator)
        if local_denominator > 0.0 else 0.0
    )
    well_rows.append({
        "well": str(well),
        "oracle_alpha": alpha,
        "move_rms": float(np.sqrt(np.mean(np.square(local_move)))),
        "signed_alignment": float(-np.dot(local_error, local_move)),
    })
well_frame = pd.DataFrame(well_rows)
well_frame.to_csv(OUT_DIR / "bywell_alpha.csv", index=False)

positive_fraction = float(np.mean(well_frame["oracle_alpha"] > 0.0))
interior_fraction = float(np.mean(well_frame["oracle_alpha"].between(0.0, 1.0)))
verdict = {
    "experiment": "anchored_v127_costvolume_dose_audit_v135b",
    "stage": "read_only_Fold3_confirmation_move_dose_attribution",
    "deployable": False,
    "selection_or_training_count": 0,
    "base_rmse": base_metric["rmse"],
    "analytic_alpha": analytic_alpha,
    "best_grid_alpha": float(best["alpha"]),
    "oracle_global_gain": float(best["gain"]),
    "oracle_global_far_gain": float(best["far_gain"]),
    "oracle_global_p95_gain": float(best["p95_gain"]),
    "positive_well_alpha_fraction": positive_fraction,
    "interior_well_alpha_fraction": interior_fraction,
    "interpretation": (
        "dose_failure_with_positive_direction"
        if analytic_alpha > 0.0 and best["gain"] >= 0.10
        else "direction_not_stably_consumable"
    ),
    "next": (
        "Positive ceiling only: cross-fit calibration before any deployment claim"
        if analytic_alpha > 0.0 and best["gain"] >= 0.10
        else "Close V135 and replace the absolute trajectory generator"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("V135b DOSE TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV135b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
