"""Cross-fitted far-tail distance-bias calibration for the exact 6.377 path.

The first 2000 ft is bitwise frozen.  Beyond it, this experiment subtracts a
smooth population signed-error curve learned exclusively from other folds.
The correction dose and smoothing scale are selected by cross-predicting the
two available training folds before refitting their shared bias curve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


EXPERIMENT = "frontier_6p377_far_bias_calibration_v50"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
SOURCE = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = (1, 2, 3)
FAR_START_FT = 2000.0
TRANSITION_FT = 256.0
BIN_FT = 128.0
SMOOTH_SIGMAS_BINS = (1.0, 2.0, 4.0, 8.0)
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_BIN_ROWS = 500
MAX_CORRECTION_FT = 8.0
P95_TOLERANCE_FT = 0.10


def _smoothstep(distance: np.ndarray) -> np.ndarray:
    t = np.clip((distance - FAR_START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _metrics(local: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    per_well = (
        pd.DataFrame({"well": local["well"].astype(str), "error2": error**2})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    near = distance <= FAR_START_FT
    far = distance > FAR_START_FT
    return {
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "near_le2000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
        "mean_far_bias": float(np.mean(error[far])),
    }


def _fit_curve(local: pd.DataFrame, sigma_bins: float) -> tuple[np.ndarray, np.ndarray]:
    distance = local["md_since"].to_numpy(np.float64)
    base = local["current_dose_0p20"].to_numpy(np.float64)
    truth = local["tvt_true"].to_numpy(np.float64)
    far = distance > FAR_START_FT
    x = distance[far]
    residual = base[far] - truth[far]
    if len(x) < MIN_BIN_ROWS:
        raise RuntimeError("Too few far rows to fit bias curve")
    maximum = float(np.ceil(x.max() / BIN_FT) * BIN_FT)
    edges = np.arange(FAR_START_FT, maximum + BIN_FT + 1e-6, BIN_FT)
    bins = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    count = np.bincount(bins, minlength=len(edges) - 1).astype(np.float64)
    total = np.bincount(bins, weights=residual, minlength=len(edges) - 1)
    means = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    valid = np.isfinite(means) & (count >= MIN_BIN_ROWS)
    if valid.sum() < 3:
        raise RuntimeError("Too few populated distance bins")
    means = np.interp(centres, centres[valid], means[valid])
    weighted_mean = gaussian_filter1d(means * count, sigma_bins, mode="nearest")
    weighted_count = gaussian_filter1d(count, sigma_bins, mode="nearest")
    curve = np.divide(
        weighted_mean,
        weighted_count,
        out=np.zeros_like(weighted_mean),
        where=weighted_count > 0,
    )
    return centres, np.clip(curve, -MAX_CORRECTION_FT, MAX_CORRECTION_FT)


def _predict(local: pd.DataFrame, curve, dose: float) -> np.ndarray:
    distance = local["md_since"].to_numpy(np.float64)
    base = local["current_dose_0p20"].to_numpy(np.float64)
    centres, values = curve
    correction = np.interp(distance, centres, values, left=values[0], right=values[-1])
    output = base - float(dose) * _smoothstep(distance) * correction
    output[distance <= FAR_START_FT] = base[distance <= FAR_START_FT]
    return output


started = time.time()
if not SOURCE.is_file():
    raise FileNotFoundError(f"Run V33 first: {SOURCE}")
frame = pd.read_parquet(SOURCE)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since", "current_dose_0p20"
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V33 source missing columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame[["tvt_true", "md_since", "current_dose_0p20"]].isna().any().any():
    raise RuntimeError("Non-finite exact-lineage values")
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids")

baseline = frame["current_dose_0p20"].to_numpy(np.float64)
oof_prediction = baseline.copy()
fold_reports = []
curve_rows = []

for held_fold in FOLDS:
    train_folds = [fold for fold in FOLDS if fold != held_fold]
    inner_reports = []
    for sigma in SMOOTH_SIGMAS_BINS:
        for dose in DOSES:
            exact_sse = 0.0
            rows = 0
            p95_changes = []
            for inner_held in train_folds:
                fit = frame[frame["fold"].astype(int) == train_folds[1 - train_folds.index(inner_held)]]
                valid = frame[frame["fold"].astype(int) == inner_held]
                curve = _fit_curve(fit, sigma)
                pred = _predict(valid, curve, dose)
                metric = _metrics(valid, pred)
                base_metric = _metrics(valid, valid["current_dose_0p20"].to_numpy(np.float64))
                exact_sse += metric["exact_sse"]
                rows += metric["rows"]
                p95_changes.append(metric["p95_well_rmse"] - base_metric["p95_well_rmse"])
            inner_reports.append({
                "sigma_bins": float(sigma),
                "dose": float(dose),
                "rmse": float(np.sqrt(exact_sse / rows)),
                "maximum_p95_change": float(max(p95_changes)),
            })
    inner = pd.DataFrame(inner_reports)
    safe = inner[inner["maximum_p95_change"] <= P95_TOLERANCE_FT]
    if safe.empty:
        selected = inner[(inner["dose"] == 0.0)].iloc[0]
    else:
        selected = safe.sort_values(["rmse", "dose", "sigma_bins"]).iloc[0]

    train = frame[frame["fold"].astype(int) != held_fold]
    held_mask = frame["fold"].astype(int) == held_fold
    held = frame[held_mask]
    curve = _fit_curve(train, float(selected["sigma_bins"]))
    held_prediction = _predict(held, curve, float(selected["dose"]))
    positions = np.flatnonzero(held_mask.to_numpy())
    oof_prediction[positions] = held_prediction
    base_metric = _metrics(held, held["current_dose_0p20"].to_numpy(np.float64))
    metric = _metrics(held, held_prediction)
    fold_reports.append({
        "fold": int(held_fold),
        "selected_sigma_bins": float(selected["sigma_bins"]),
        "selected_dose": float(selected["dose"]),
        "inner_rmse": float(selected["rmse"]),
        "base_rmse": base_metric["rmse"],
        "rmse": metric["rmse"],
        "gain": base_metric["rmse"] - metric["rmse"],
        "base_far_rmse": base_metric["far_gt2000_rmse"],
        "far_rmse": metric["far_gt2000_rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "base_far_bias": base_metric["mean_far_bias"],
        "far_bias": metric["mean_far_bias"],
        "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
    })
    for distance, value in zip(*curve):
        curve_rows.append({
            "held_fold": int(held_fold),
            "distance_ft": float(distance),
            "signed_bias_ft": float(value),
        })

near = frame["md_since"].to_numpy(np.float64) <= FAR_START_FT
if not np.array_equal(oof_prediction[near], baseline[near]):
    raise RuntimeError("Rows through 2000 ft changed")

base_metrics = _metrics(frame, baseline)
candidate_metrics = _metrics(frame, oof_prediction)
fold_table = pd.DataFrame(fold_reports)
gain = base_metrics["rmse"] - candidate_metrics["rmse"]
far_gain = base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"]
checks = {
    "overall_gain_ge_0p03": bool(gain >= 0.03),
    "far_gain_ge_0p05": bool(far_gain >= 0.05),
    "minimum_fold_gain_ge_0": bool(fold_table["gain"].min() >= 0.0),
    "minimum_fold_far_gain_ge_0": bool(fold_table["far_gain"].min() >= 0.0),
    "maximum_p95_change_le_0p10": bool(fold_table["p95_change"].max() <= 0.10),
    "near_rows_bitwise_unchanged": True,
}
strict_pass = bool(all(checks.values()))

fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
pd.DataFrame(curve_rows).to_csv(OUT_DIR / "bias_curves.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    base_6p377=baseline.astype(np.float32),
    far_bias_calibrated=oof_prediction.astype(np.float32),
).to_parquet(OUT_DIR / "far_bias_calibrated_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_population_far_distance_bias_calibration",
    "parent": str(SOURCE),
    "validation_status": "retrospective_threefold_nested; no untouched-fold claim",
    "frozen_through_ft": FAR_START_FT,
    "base": base_metrics,
    "candidate": candidate_metrics,
    "gain": float(gain),
    "far_gain": float(far_gain),
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> fit one all-OOF bias curve and package a conservative submission"
        if strict_pass
        else "FAIL -> signed tail bias is not fold-stable; close output-only postprocessing"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("V50 FAR-BIAS FOLDS")
print(fold_table.to_string(index=False), flush=True)
print("\nV50 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
