"""V92: pooled-SSE-consistent calibration of V76 adaptive coarse doses.

For one well, with move = coarse - parent, the excess SSE from predicting
dose w instead of the oracle dose w* is proportional to

    sum(move ** 2) * (w - w*) ** 2.

V76 regressed oracle doses with equal/sqrt(row) weighting.  This cell keeps
the V76 endpoint and features frozen and only calibrates its predicted dose
using the exact per-well SSE curvature.  Fold3 selects one of three fixed,
low-complexity calibrators; Fold2 confirms it and Fold1 is the final check.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_v76_curvature_calibration_v92"
OUT_DIR = OOF_ROOT / EXPERIMENT
DEPLOYMENT_DIR = OUT_DIR / "deployment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V74_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_oracle_v74" / "perwell_oracle_features.parquet"
V76_WEIGHT_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"
V76_BUNDLE_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "deployment" / "adaptive_coarse_dose_bundle_v76.joblib"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.60
FOLDS = (1, 2, 3)
CALIBRATORS = ("curvature_affine", "curvature_isotonic", "curvature_mean")
STRATA = (
    ("near_0_500", -np.inf, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
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


def _clip(values):
    return np.clip(np.asarray(values, np.float64), MIN_WEIGHT, MAX_WEIGHT)


def _fit_affine(x, y, curvature):
    design = np.column_stack([np.ones(len(x), np.float64), x])
    weight = np.maximum(np.asarray(curvature, np.float64), 1e-12)
    weight /= np.mean(weight)
    root = np.sqrt(weight)
    coefficients = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    return coefficients


def _apply_affine(coefficients, x):
    return _clip(float(coefficients[0]) + float(coefficients[1]) * np.asarray(x, np.float64))


def _fit_calibrators(train):
    train = train[
        np.isfinite(train["curvature"].to_numpy(np.float64))
        & train["curvature"].gt(1e-12)
    ].copy()
    if len(train) < 50:
        raise RuntimeError(f"Only {len(train)} positive-curvature training wells")
    x = train["v76_weight"].to_numpy(np.float64)
    y = train["oracle_weight"].to_numpy(np.float64)
    curvature = train["curvature"].to_numpy(np.float64)
    affine = _fit_affine(x, y, curvature)
    isotonic = IsotonicRegression(
        y_min=MIN_WEIGHT, y_max=MAX_WEIGHT, increasing=True, out_of_bounds="clip"
    )
    isotonic.fit(x, y, sample_weight=curvature / np.mean(curvature))
    return affine, isotonic


def _calibrated_predictions(affine, isotonic, x):
    affine_prediction = _apply_affine(affine, x)
    isotonic_prediction = _clip(isotonic.predict(np.asarray(x, np.float64)))
    return {
        "curvature_affine": affine_prediction,
        "curvature_isotonic": isotonic_prediction,
        "curvature_mean": _clip(0.5 * affine_prediction + 0.5 * isotonic_prediction),
    }


def _prediction(local, weight_map):
    weights = local["well"].astype(str).map(weight_map)
    if weights.isna().any():
        raise RuntimeError("Missing calibrated dose")
    parent = local[PARENT].to_numpy(np.float64)
    move = local[COARSE].to_numpy(np.float64) - parent
    return parent + weights.to_numpy(np.float64) * move


def _metric(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local[TRUTH].to_numpy(np.float64)
    error2 = error * error
    bywell = pd.DataFrame({"well": local["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = local["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    for name, lower, upper in STRATA:
        mask = (distance > lower) & (distance <= upper)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _delta(base, candidate):
    return {
        "gain": base["rmse"] - candidate["rmse"],
        "far_gain": base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"],
        "near_degradation": candidate["near_0_500_rmse"] - base["near_0_500_rmse"],
        "p95_gain": base["p95_well_rmse"] - candidate["p95_well_rmse"],
    }


for required in (V70_PATH, V74_PATH, V76_WEIGHT_PATH, V76_BUNDLE_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
features = pd.read_parquet(V74_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
weights = pd.read_parquet(V76_WEIGHT_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
if not features[["fold", "well"]].equals(weights[["fold", "well"]]):
    raise RuntimeError("V74/V76 well alignment mismatch")
if frame.groupby("fold")["well"].nunique().astype(int).to_dict() != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError("Unexpected V70 fold coverage")

well_table = features[["fold", "well", "rows", "oracle_weight", "move_rms"]].copy()
well_table["v76_weight"] = weights["ensemble_weight"].to_numpy(np.float64)
well_table["curvature"] = (
    well_table["rows"].to_numpy(np.float64)
    * np.square(well_table["move_rms"].to_numpy(np.float64))
)
if (well_table["curvature"] < 0.0).any() or not np.isfinite(well_table["curvature"]).all():
    raise RuntimeError("Invalid per-well SSE curvature")
zero_curvature_wells = int(well_table["curvature"].le(1e-12).sum())
print(
    "V92 CURVATURE",
    {
        "wells": len(well_table),
        "positive": len(well_table) - zero_curvature_wells,
        "zero_unidentifiable": zero_curvature_wells,
        "median": float(well_table["curvature"].median()),
        "maximum": float(well_table["curvature"].max()),
    },
    flush=True,
)

oof_predictions = {name: np.full(len(well_table), np.nan, np.float64) for name in CALIBRATORS}
fit_reports = []
for heldout_fold in FOLDS:
    train_mask = well_table["fold"].ne(heldout_fold).to_numpy()
    valid_mask = ~train_mask
    affine, isotonic = _fit_calibrators(well_table.loc[train_mask])
    predictions = _calibrated_predictions(
        affine, isotonic, well_table.loc[valid_mask, "v76_weight"].to_numpy(np.float64)
    )
    for name, values in predictions.items():
        oof_predictions[name][valid_mask] = values
    fit_reports.append({
        "heldout_fold": heldout_fold,
        "train_wells": int(train_mask.sum()),
        "valid_wells": int(valid_mask.sum()),
        "affine_intercept": float(affine[0]),
        "affine_slope": float(affine[1]),
        "v76_mean": float(well_table.loc[valid_mask, "v76_weight"].mean()),
        **{f"{name}_mean": float(values.mean()) for name, values in predictions.items()},
    })

if any(not np.isfinite(values).all() for values in oof_predictions.values()):
    raise RuntimeError("Incomplete calibrated OOF doses")

weight_export = well_table[["fold", "well", "oracle_weight", "curvature", "v76_weight"]].copy()
for name, values in oof_predictions.items():
    weight_export[name] = values
weight_export.to_parquet(OUT_DIR / "calibrated_weights_oof.parquet", index=False)
pd.DataFrame(fit_reports).to_csv(OUT_DIR / "fit_reports.csv", index=False)

summary_rows = []


def _evaluate_fold(fold, candidate_names):
    local = frame[frame["fold"].eq(fold)].copy()
    fold_weights = weight_export[weight_export["fold"].eq(fold)].copy()
    baseline_map = dict(zip(fold_weights["well"].astype(str), fold_weights["v76_weight"]))
    baseline = _metric(local, _prediction(local, baseline_map), "v76")
    output = {"v76": baseline}
    rows = [{"fold": fold, **baseline}]
    for name in candidate_names:
        weight_map = dict(zip(fold_weights["well"].astype(str), fold_weights[name]))
        candidate = _metric(local, _prediction(local, weight_map), name)
        output[name] = candidate
        rows.append({"fold": fold, **candidate, **_delta(baseline, candidate)})
    return output, rows

# Select the calibration family on Fold3 only, then read Fold2 and Fold1 once.
metrics3, rows3 = _evaluate_fold(3, CALIBRATORS)
summary_rows.extend(rows3)
fold3_base = metrics3["v76"]
fold3_candidates = []
for name in CALIBRATORS:
    candidate = metrics3[name]
    delta = _delta(fold3_base, candidate)
    fold3_candidates.append({"calibrator": name, **delta})
fold3_scan = pd.DataFrame(fold3_candidates).sort_values("gain", ascending=False, kind="stable")
fold3_scan.to_csv(OUT_DIR / "fold3_calibrator_scan.csv", index=False)
eligible = fold3_scan[
    fold3_scan["gain"].ge(0.02)
    & fold3_scan["far_gain"].ge(0.0)
    & fold3_scan["near_degradation"].le(0.05)
    & fold3_scan["p95_gain"].ge(-0.10)
]

selected = None if eligible.empty else str(eligible.iloc[0]["calibrator"])
fold2_delta = None
fold1_delta = None
fold1_touched = False
strict_pass = False
if selected is not None:
    metrics2, rows2 = _evaluate_fold(2, (selected,))
    summary_rows.extend(rows2)
    fold2_delta = _delta(metrics2["v76"], metrics2[selected])
    fold2_pass = bool(
        fold2_delta["gain"] >= 0.02
        and fold2_delta["far_gain"] >= 0.0
        and fold2_delta["near_degradation"] <= 0.05
        and fold2_delta["p95_gain"] >= -0.10
    )
    if fold2_pass:
        metrics1, rows1 = _evaluate_fold(1, (selected,))
        summary_rows.extend(rows1)
        fold1_touched = True
        fold1_delta = _delta(metrics1["v76"], metrics1[selected])
        strict_pass = bool(
            fold1_delta["gain"] > 0.0
            and fold1_delta["far_gain"] >= 0.0
            and fold1_delta["near_degradation"] <= 0.05
            and fold1_delta["p95_gain"] >= -0.10
        )

bundle_path = None
if strict_pass:
    full_affine, full_isotonic = _fit_calibrators(well_table)
    bundle = {
        "experiment": EXPERIMENT,
        "selected_calibrator": selected,
        "affine_coefficients": full_affine,
        "isotonic_model": full_isotonic,
        "min_weight": MIN_WEIGHT,
        "max_weight": MAX_WEIGHT,
        "input": "raw V76 adaptive dose",
        "curvature_formula": "rows * move_rms**2",
        "training_wells": len(well_table),
    }
    bundle_path = DEPLOYMENT_DIR / "v76_curvature_calibration_v92.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_calibrator_selection_Fold2_confirmation_Fold1_final",
    "base": "V69 All773 coarse plus V76 adaptive dose",
    "objective": "exact pooled-SSE curvature-weighted calibration",
    "zero_curvature_unidentifiable_wells": zero_curvature_wells,
    "selection_search_count": len(CALIBRATORS),
    "selected_calibrator": selected,
    "fold3_scan": fold3_candidates,
    "fold2_confirmation": fold2_delta,
    "fold1_final": fold1_delta,
    "fold1_touched": fold1_touched,
    "strict_pass": strict_pass,
    "bundle": bundle_path,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> append the calibration bundle immediately after V76 dose prediction"
        if strict_pass else "FAIL -> retain uncalibrated V76; close dose-calibration branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV92 FIT REPORTS")
print(pd.DataFrame(fit_reports).to_string(index=False), flush=True)
print("\nV92 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV92 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
