"""V143b: read-only projection audit for the V143 absolute trajectory.

Truth is used only to measure oracle projection ceilings. Nothing produced by
this script is deployable. The purpose is to decide whether V143 contains a
small complementary direction worth cross-fitting, or should be closed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = (
    ROOT / "oof_exports" / "pseudo_ps_pretrain_actual_finetune_v143"
    / "fold3_confirmation_oof.parquet"
)
OUT_DIR = ROOT / "oof_exports" / "pseudo_ps_finetune_projection_audit_v143b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRUTH = "tvt_true"
BASE = "nested_sequence_coefficient"
CANDIDATE = "direct_absolute_trajectory"


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
    truth = frame[TRUTH].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    well_sse = pd.Series(np.square(error), index=frame["well"]).groupby(level=0).mean()
    result = {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_sse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_sse), 0.95)),
    }
    bands = {
        "near_0_500_rmse": (distance >= 0.0) & (distance < 500.0),
        "mid_500_1000_rmse": (distance >= 500.0) & (distance < 1000.0),
        "mid_1000_2000_rmse": (distance >= 1000.0) & (distance < 2000.0),
        "far_gt2000_rmse": distance >= 2000.0,
    }
    for name, mask in bands.items():
        result[name] = float(np.sqrt(np.mean(np.square(error[mask])))) if mask.any() else np.nan
    return result


def _smooth_mask(distance, start, ramp):
    if start <= 0.0:
        return np.ones(len(distance), np.float64)
    value = np.clip((distance - start) / max(ramp, 1.0), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _analytic_alpha(error, move, lower=-1.0, upper=1.0):
    curvature = float(np.dot(move, move))
    if curvature <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(error, move) / curvature, lower, upper))


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE)
frame["well"] = frame["well"].astype(str)
required = {"id", "well", "row_idx", TRUTH, "md_since", BASE, CANDIDATE}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V143 OOF missing columns: {missing}")
if frame["id"].duplicated().any() or frame[list(required)].isna().any().any():
    raise RuntimeError("Invalid V143 confirmation OOF")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

truth = frame[TRUTH].to_numpy(np.float64)
base = frame[BASE].to_numpy(np.float64)
direct = frame[CANDIDATE].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
base_error = base - truth
raw_move = direct - base
base_metric = _metric(frame, base, BASE)

profiles = [
    ("full", 0.0, 1.0),
    ("freeze500", 500.0, 250.0),
    ("freeze1000", 1000.0, 250.0),
    ("freeze2000", 2000.0, 250.0),
]
rows = [dict(base_metric, profile="base", alpha=0.0, gain=0.0)]
profile_reports = []
for profile, start, ramp in profiles:
    move = raw_move * _smooth_mask(distance, start, ramp)
    alpha_signed = _analytic_alpha(base_error, move, -1.0, 1.0)
    alpha_positive = _analytic_alpha(base_error, move, 0.0, 1.0)
    for label, alpha in (("signed", alpha_signed), ("positive", alpha_positive)):
        metric = _metric(frame, base + alpha * move, f"{profile}_{label}")
        metric.update(
            profile=profile,
            alpha=alpha,
            gain=base_metric["rmse"] - metric["rmse"],
            far_gain=base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            p95_gain=base_metric["p95_well_rmse"] - metric["p95_well_rmse"],
        )
        rows.append(metric)
    profile_reports.append({
        "profile": profile,
        "signed_alpha": alpha_signed,
        "positive_alpha": alpha_positive,
        "move_rms": float(np.sqrt(np.mean(np.square(move)))),
        "signed_error_move_correlation": float(np.corrcoef(base_error, move)[0, 1]),
    })

# Per-well oracle is an investment ceiling, never a candidate prediction.
perwell_prediction = base.copy()
perwell_rows = []
for well, indices in frame.groupby("well", sort=True).indices.items():
    index = np.asarray(indices, np.int64)
    alpha = _analytic_alpha(base_error[index], raw_move[index], -1.0, 1.0)
    perwell_prediction[index] += alpha * raw_move[index]
    perwell_rows.append({"well": well, "oracle_alpha": alpha})
perwell_metric = _metric(frame, perwell_prediction, "perwell_signed_oracle")
perwell_gain = base_metric["rmse"] - perwell_metric["rmse"]
rows.append(dict(perwell_metric, profile="perwell", alpha=np.nan, gain=perwell_gain))

summary = pd.DataFrame(rows).sort_values("rmse", kind="stable")
summary.to_csv(OUT_DIR / "projection_summary.csv", index=False)
pd.DataFrame(profile_reports).to_csv(OUT_DIR / "profile_diagnostics.csv", index=False)
perwell = pd.DataFrame(perwell_rows)
perwell.to_csv(OUT_DIR / "perwell_oracle_alpha.csv", index=False)

global_rows = summary[summary["profile"].isin([item[0] for item in profiles])]
best = global_rows.iloc[0].to_dict()
verdict = {
    "experiment": "pseudo_ps_finetune_projection_audit_v143b",
    "stage": "read_only_Fold3_confirmation_projection_ceiling",
    "deployable": False,
    "selection_or_training_count": 0,
    "truth_usage": "analytic global and per-well oracle doses only",
    "base_rmse": base_metric["rmse"],
    "best_global_profile": best.get("profile"),
    "best_global_endpoint": best.get("endpoint"),
    "best_global_alpha": best.get("alpha"),
    "best_global_gain": base_metric["rmse"] - float(best["rmse"]),
    "best_global_far_gain": base_metric["far_gt2000_rmse"] - float(best["far_gt2000_rmse"]),
    "best_global_p95_gain": base_metric["p95_well_rmse"] - float(best["p95_well_rmse"]),
    "perwell_oracle_gain": perwell_gain,
    "perwell_positive_alpha_fraction": float((perwell["oracle_alpha"] > 0.0).mean()),
    "perwell_negative_alpha_fraction": float((perwell["oracle_alpha"] < 0.0).mean()),
    "raw_error_correlation": float(np.corrcoef(base_error, direct - truth)[0, 1]),
    "checks": {
        "best_global_gain_ge_0p05": base_metric["rmse"] - float(best["rmse"]) >= 0.05,
        "best_global_far_gain_positive": base_metric["far_gt2000_rmse"] - float(best["far_gt2000_rmse"]) > 0.0,
        "perwell_oracle_gain_ge_0p20": perwell_gain >= 0.20,
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
verdict["interpretation"] = (
    "retain_direction_for_crossfit_calibration"
    if verdict["checks"]["best_global_gain_ge_0p05"]
    else "close_V143_direction"
)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(_clean(verdict), handle, indent=2, allow_nan=False)

print("V143b PROJECTION SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV143b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
