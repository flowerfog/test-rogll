"""V123: residual-shape investment gate on the frozen top-33 consensus.

This is a truth-only oracle diagnostic, not a deployable correction. It asks
how much of the remaining error is complete-well datum/drift versus nonlinear
trajectory error, and whether corrections that freeze the first 1000 ft have
enough ceiling to justify another postprocessing model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "final_consensus_residual_oracle_v123"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_PATH = (
    OOF_ROOT
    / "rawdata_lattice_consensus_center_fold2_replay_v115"
    / "fold123_consensus_center_oof.parquet"
)
TRUTH = "tvt_true"
BASE = "consensus_center"
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


def _fit_projection(target, design):
    target = np.asarray(target, np.float64)
    design = np.asarray(design, np.float64)
    if design.ndim == 1:
        design = design[:, None]
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    keep = np.isfinite(scale) & (scale > 1e-10)
    if not keep.any():
        return np.zeros_like(target), np.zeros(design.shape[1], np.float64)
    normalized = design[:, keep] / scale[keep]
    coefficient_scaled, *_ = np.linalg.lstsq(normalized, target, rcond=1e-10)
    correction = normalized @ coefficient_scaled
    coefficient = np.zeros(design.shape[1], np.float64)
    coefficient[keep] = coefficient_scaled / scale[keep]
    return correction, coefficient


def _metric(frame, prediction, endpoint, fold):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    error = prediction - truth
    error2 = np.square(error)
    bywell = pd.Series(error2).groupby(frame["well"].to_numpy()).mean().pow(0.5)
    result = {
        "fold": int(fold),
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
    for label, mask in masks.items():
        result[f"{label}_rmse"] = (
            float(np.sqrt(error2[mask].mean())) if mask.any() else float("nan")
        )
    return result


if not BASE_PATH.is_file():
    raise FileNotFoundError(BASE_PATH)
frame = pd.read_parquet(BASE_PATH)
required = {"id", "well", "row_idx", "fold", TRUTH, "md_since", BASE}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Base OOF missing columns: {missing}")
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
if frame["id"].duplicated().any() or frame["well"].nunique() != 464:
    raise RuntimeError(f"Invalid base coverage rows={len(frame)} wells={frame['well'].nunique()}")
if not np.isfinite(frame[[TRUTH, "md_since", BASE]].to_numpy(np.float64)).all():
    raise RuntimeError("Non-finite base values")

prediction_parts = {
    "oracle_datum": [],
    "oracle_affine": [],
    "oracle_tail_datum": [],
    "oracle_tail_affine": [],
    "oracle_tail_piecewise": [],
}
base_parts = []
well_rows = []

for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    local = local.sort_values("row_idx", kind="stable").copy()
    base = local[BASE].to_numpy(np.float64)
    target = local[TRUTH].to_numpy(np.float64) - base
    distance = local["md_since"].to_numpy(np.float64)
    x = (distance - float(np.mean(distance))) / 1000.0
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)

    datum_move, datum_coef = _fit_projection(target, np.ones(len(local)))
    affine_move, affine_coef = _fit_projection(target, np.column_stack([np.ones(len(local)), x]))
    tail_datum_move, tail_datum_coef = _fit_projection(target, taper)
    tail_affine_move, tail_affine_coef = _fit_projection(
        target, np.column_stack([taper, taper * x])
    )
    hinge1 = taper * np.clip((distance - 1250.0) / 1000.0, 0.0, None)
    hinge2 = taper * np.clip((distance - 2250.0) / 1000.0, 0.0, None)
    tail_piecewise_move, tail_piecewise_coef = _fit_projection(
        target, np.column_stack([taper, hinge1, hinge2])
    )

    moves = {
        "oracle_datum": datum_move,
        "oracle_affine": affine_move,
        "oracle_tail_datum": tail_datum_move,
        "oracle_tail_affine": tail_affine_move,
        "oracle_tail_piecewise": tail_piecewise_move,
    }
    base_sse = float(np.sum(np.square(base - local[TRUTH].to_numpy(np.float64))))
    report = {
        "well": well,
        "fold": int(local["fold"].iloc[0]),
        "rows": len(local),
        "base_sse": base_sse,
        "datum_ft": float(datum_coef[0]),
        "affine_intercept_ft": float(affine_coef[0]),
        "affine_slope_ft_per_1000ft": float(affine_coef[1]),
        "tail_datum_ft": float(tail_datum_coef[0]),
        "tail_affine_intercept_ft": float(tail_affine_coef[0]),
        "tail_affine_slope_ft_per_1000ft": float(tail_affine_coef[1]),
        "tail_piecewise_intercept_ft": float(tail_piecewise_coef[0]),
        "tail_piecewise_hinge1250_ft": float(tail_piecewise_coef[1]),
        "tail_piecewise_hinge2250_ft": float(tail_piecewise_coef[2]),
    }
    for endpoint, move in moves.items():
        candidate = base + move
        candidate_sse = float(np.sum(np.square(candidate - local[TRUTH].to_numpy(np.float64))))
        report[f"{endpoint}_sse"] = candidate_sse
        report[f"{endpoint}_explained_fraction"] = (
            (base_sse - candidate_sse) / max(base_sse, 1e-12)
        )
        prediction_parts[endpoint].append(candidate)
    if np.max(np.abs(tail_piecewise_move[distance <= FREEZE_FT])) > 1e-10:
        raise RuntimeError(f"{well}: tail oracle changed frozen prefix")
    base_parts.append(local)
    well_rows.append(report)
    if number % 100 == 0 or number == frame["well"].nunique():
        print(f"V123 decompose {number}/{frame['well'].nunique()}", flush=True)

matched = pd.concat(base_parts, ignore_index=True)
all_predictions = {BASE: matched[BASE].to_numpy(np.float64)}
all_predictions.update({name: np.concatenate(parts) for name, parts in prediction_parts.items()})
summary_rows = []
for fold in (1, 2, 3, 0):
    mask = np.ones(len(matched), bool) if fold == 0 else matched["fold"].to_numpy() == fold
    local = matched.loc[mask].reset_index(drop=True)
    for endpoint, prediction in all_predictions.items():
        summary_rows.append(_metric(local, prediction[mask], endpoint, fold))
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "oracle_summary.csv", index=False)
well_frame = pd.DataFrame(well_rows)
well_frame.to_csv(OUT_DIR / "oracle_bywell.csv", index=False)

overall = summary[summary["fold"].eq(0)].set_index("endpoint")
base_rmse = float(overall.loc[BASE, "rmse"])
gains = {
    endpoint: float(base_rmse - overall.loc[endpoint, "rmse"])
    for endpoint in prediction_parts
}
tail_piecewise_rmse = float(overall.loc["oracle_tail_piecewise", "rmse"])
verdict = {
    "experiment": EXPERIMENT,
    "stage": "current_top33_residual_shape_oracle_investment_gate",
    "truth_usage": "oracle projection and metrics only",
    "deployable": False,
    "coverage": {"rows": len(matched), "wells": int(matched["well"].nunique())},
    "base_rmse": base_rmse,
    "oracle_rmse": {
        endpoint: float(overall.loc[endpoint, "rmse"])
        for endpoint in prediction_parts
    },
    "oracle_gain": gains,
    "tail_piecewise_oracle_rmse": tail_piecewise_rmse,
    "checks": {
        "tail_datum_gain_ge_0p20": gains["oracle_tail_datum"] >= 0.20,
        "tail_affine_increment_ge_0p20": (
            float(overall.loc["oracle_tail_datum", "rmse"])
            - float(overall.loc["oracle_tail_affine", "rmse"])
        ) >= 0.20,
        "tail_piecewise_rmse_le_6": tail_piecewise_rmse <= 6.0,
        "full_affine_rmse_le_6": float(overall.loc["oracle_affine", "rmse"]) <= 6.0,
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "If tail affine/piecewise has large ceiling, test legal well-level coefficient prediction; "
        "otherwise stop postprocessing and replace trajectory generation."
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV123 ORACLE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV123 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
