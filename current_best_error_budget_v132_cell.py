"""V132: error budget and oracle decomposition of the exact V127 endpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "current_best_error_budget_v132"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = OOF_ROOT / "final_consensus_sequence_coefficient_crossfit_v127" / "crossfit_oof.parquet"
TRUTH = "tvt_true"
PREDICTION = "nested_sequence_coefficient"
TARGET_RMSE = (7.0, 6.5, 6.0, 5.7)
FREEZE_FT = 1000.0
RAMP_FT = 250.0


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


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _project(target, design):
    target = np.asarray(target, np.float64)
    design = np.asarray(design, np.float64)
    if design.ndim == 1:
        design = design[:, None]
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    keep = np.isfinite(scale) & (scale > 1e-10)
    correction = np.zeros_like(target)
    coefficient = np.zeros(design.shape[1], np.float64)
    if keep.any():
        normalized = design[:, keep] / scale[keep]
        fitted, *_ = np.linalg.lstsq(normalized, target, rcond=1e-10)
        correction = normalized @ fitted
        coefficient[keep] = fitted / scale[keep]
    return correction, coefficient


def _metric(frame, prediction, endpoint):
    truth = frame[TRUTH].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    error2 = np.square(error)
    bywell = pd.Series(error2).groupby(frame["well"].to_numpy()).mean().pow(0.5)
    result = {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    masks = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in masks.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error2[mask]))) if mask.any() else np.nan
    return result


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
required = {"id", "well", "row_idx", "fold", TRUTH, "md_since", PREDICTION}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V127 OOF missing columns: {missing}")
if frame["id"].duplicated().any() or frame["well"].nunique() != 464:
    raise RuntimeError(f"Invalid V127 coverage rows={len(frame)} wells={frame['well'].nunique()}")

base_prediction = frame[PREDICTION].to_numpy(np.float64)
truth = frame[TRUTH].to_numpy(np.float64)
base_error2 = np.square(base_prediction - truth)
oracle_predictions = {
    "oracle_datum": np.empty(len(frame), np.float64),
    "oracle_affine": np.empty(len(frame), np.float64),
    "oracle_tail_datum": np.empty(len(frame), np.float64),
    "oracle_tail_affine": np.empty(len(frame), np.float64),
    "oracle_tail_piecewise": np.empty(len(frame), np.float64),
    "oracle_tail_cp8": np.empty(len(frame), np.float64),
}
well_rows = []
for number, (well, positions) in enumerate(frame.groupby("well", sort=True).indices.items(), 1):
    positions = np.asarray(positions, np.int64)
    local = frame.iloc[positions]
    prediction = base_prediction[positions]
    target = truth[positions] - prediction
    distance = local["md_since"].to_numpy(np.float64)
    centered = (distance - np.mean(distance)) / 1000.0
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    hinge1 = taper * np.clip((distance - 1250.0) / 1000.0, 0.0, None)
    hinge2 = taper * np.clip((distance - 2250.0) / 1000.0, 0.0, None)
    cp_centers = np.linspace(FREEZE_FT, max(float(distance.max()), FREEZE_FT + 1.0), 8)
    cp_width = max(float(cp_centers[1] - cp_centers[0]), 1.0)
    cp_design = []
    for center in cp_centers:
        cp_design.append(taper * np.maximum(1.0 - np.abs(distance - center) / cp_width, 0.0))
    designs = {
        "oracle_datum": np.ones(len(local)),
        "oracle_affine": np.column_stack([np.ones(len(local)), centered]),
        "oracle_tail_datum": taper,
        "oracle_tail_affine": np.column_stack([taper, taper * centered]),
        "oracle_tail_piecewise": np.column_stack([taper, hinge1, hinge2]),
        "oracle_tail_cp8": np.column_stack(cp_design),
    }
    base_sse = float(np.sum(np.square(target)))
    report = {
        "well": well, "fold": int(local["fold"].iloc[0]), "rows": len(local),
        "base_sse": base_sse, "base_rmse": float(np.sqrt(base_sse / len(local))),
    }
    for endpoint, design in designs.items():
        correction, coefficient = _project(target, design)
        oracle_predictions[endpoint][positions] = prediction + correction
        remaining_sse = float(np.sum(np.square(target - correction)))
        report[f"{endpoint}_sse"] = remaining_sse
        report[f"{endpoint}_explained_fraction"] = (base_sse - remaining_sse) / max(base_sse, 1e-12)
        report[f"{endpoint}_coefficient_rms"] = float(np.sqrt(np.mean(np.square(coefficient))))
    if np.max(np.abs(oracle_predictions["oracle_tail_cp8"][positions][distance <= FREEZE_FT] - prediction[distance <= FREEZE_FT])) > 1e-8:
        raise RuntimeError(f"{well}: tail oracle changed frozen prefix")
    well_rows.append(report)
    if number % 100 == 0 or number == 464:
        print(f"V132 oracle {number}/464", flush=True)

summary_rows = []
for fold in (1, 2, 3, 0):
    mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].to_numpy() == fold
    local = frame.loc[mask]
    summary_rows.append({"fold": fold, **_metric(local, base_prediction[mask], PREDICTION)})
    for endpoint, prediction in oracle_predictions.items():
        summary_rows.append({"fold": fold, **_metric(local, prediction[mask], endpoint)})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "oracle_summary.csv", index=False)
well_frame = pd.DataFrame(well_rows).sort_values("base_sse", ascending=False, kind="stable")
well_frame["sse_fraction"] = well_frame["base_sse"] / well_frame["base_sse"].sum()
well_frame["cumulative_sse_fraction"] = well_frame["sse_fraction"].cumsum()
well_frame.to_csv(OUT_DIR / "well_error_budget.csv", index=False)

total_rows = len(frame)
total_sse = float(base_error2.sum())
sorted_sse = well_frame["base_sse"].to_numpy(np.float64)
cumulative = np.cumsum(sorted_sse)
target_rows = []
for target_rmse in TARGET_RMSE:
    required_removal = total_sse - float(target_rmse) ** 2 * total_rows
    wells_needed = int(np.searchsorted(cumulative, max(required_removal, 0.0), side="left") + 1) if required_removal > 0 else 0
    wells_needed = min(wells_needed, len(well_frame))
    removed = float(cumulative[wells_needed - 1]) if wells_needed else 0.0
    achieved = float(np.sqrt(max(total_sse - removed, 0.0) / total_rows))
    target_rows.append({
        "target_rmse": target_rmse,
        "perfectly_fixed_wells_required": wells_needed,
        "well_fraction": wells_needed / len(well_frame),
        "sse_fraction_removed": removed / total_sse,
        "achieved_rmse_after_whole_well_oracle": achieved,
    })
target_budget = pd.DataFrame(target_rows)
target_budget.to_csv(OUT_DIR / "target_budget.csv", index=False)

distance = frame["md_since"].to_numpy(np.float64)
strata = {
    "near_0_500": distance <= 500.0,
    "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
    "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
    "far_gt2000": distance > 2000.0,
}
strata_rows = []
for name, mask in strata.items():
    removed = float(base_error2[mask].sum())
    strata_rows.append({
        "perfectly_fixed_stratum": name,
        "rows": int(mask.sum()),
        "row_fraction": float(mask.mean()),
        "sse_fraction": removed / total_sse,
        "remaining_rmse": float(np.sqrt(max(total_sse - removed, 0.0) / total_rows)),
    })
strata_budget = pd.DataFrame(strata_rows)
strata_budget.to_csv(OUT_DIR / "strata_budget.csv", index=False)

overall = summary[summary["fold"].eq(0)].set_index("endpoint")
base_rmse = float(overall.loc[PREDICTION, "rmse"])
oracle_values = {
    endpoint: {
        "rmse": float(overall.loc[endpoint, "rmse"]),
        "gain": base_rmse - float(overall.loc[endpoint, "rmse"]),
    }
    for endpoint in oracle_predictions
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "exact_V127_error_budget_and_oracle_decomposition",
    "deployable": False,
    "truth_usage": "oracle projections and metrics only",
    "coverage": {"rows": len(frame), "wells": int(frame["well"].nunique())},
    "base_rmse": base_rmse,
    "oracle": oracle_values,
    "targets": target_rows,
    "distance_strata_perfect_fix": strata_rows,
    "decision": {
        "tail_cp8_can_reach_6": oracle_values["oracle_tail_cp8"]["rmse"] <= 6.0,
        "tail_piecewise_can_reach_6": oracle_values["oracle_tail_piecewise"]["rmse"] <= 6.0,
        "full_affine_can_reach_6": oracle_values["oracle_affine"]["rmse"] <= 6.0,
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "Choose the next model family from the measured error budget; do not deploy oracle outputs.",
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV132 ORACLE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV132 TARGET BUDGET")
print(target_budget.to_string(index=False), flush=True)
print("\nV132 STRATA BUDGET")
print(strata_budget.to_string(index=False), flush=True)
print("\nV132 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
