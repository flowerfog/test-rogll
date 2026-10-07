"""Fixed deployment analogue for the V75 adaptive coarse dose.

V75's nested selector chose a conservative HistGradientBoosting model in two
outer folds and Ridge(alpha=100) in the third.  This cell predefines their
equal-weight ensemble, cross-fits it once, and only then fits the deployment
bundle on all available honest dose labels.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_coarse_adaptive_dose_bundle_v76"
OUT_DIR = OOF_ROOT / EXPERIMENT
DEPLOYMENT_DIR = OUT_DIR / "deployment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V74_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_oracle_v74" / "perwell_oracle_features.parquet"
V75_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_crossfit_v75" / "adaptive_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
FIXED_WEIGHT = 0.30
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.60
HIST_FRACTION = 0.50
RIDGE_FRACTION = 0.50
OUTER_FOLDS = (1, 2, 3)
BASE_FEATURES = (
    "maximum_distance_ft", "move_mean", "move_std", "move_rms",
    "move_abs_mean", "move_abs_p90", "move_abs_max", "move_slope_per_ft",
    "endpoint_correlation", "parent_slope_per_ft", "coarse_slope_per_ft",
    "parent_roughness", "coarse_roughness", "near_0_500_fraction",
    "near_0_500_move_rms", "near_0_500_move_mean", "mid_500_1000_fraction",
    "mid_500_1000_move_rms", "mid_500_1000_move_mean",
    "mid_1000_2000_fraction", "mid_1000_2000_move_rms",
    "mid_1000_2000_move_mean", "far_gt2000_fraction",
    "far_gt2000_move_rms", "far_gt2000_move_mean",
)
LOG_FEATURES = (
    "move_std", "move_rms", "move_abs_mean", "move_abs_p90", "move_abs_max",
    "parent_roughness", "coarse_roughness", "near_0_500_move_rms",
    "mid_500_1000_move_rms", "mid_1000_2000_move_rms", "far_gt2000_move_rms",
)
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


def _augment(features):
    output = features.loc[:, BASE_FEATURES].astype(np.float64).copy()
    for column in LOG_FEATURES:
        output[f"log1p_{column}"] = np.log1p(np.maximum(output[column], 0.0))
    output["roughness_ratio"] = output["coarse_roughness"] / np.maximum(
        output["parent_roughness"], 1e-4
    )
    output["slope_difference"] = output["coarse_slope_per_ft"] - output["parent_slope_per_ft"]
    values = output.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite dose features")
    return values, list(output.columns)


def _hist_model():
    return HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=45,
        l2_regularization=12.0,
        random_state=20260730,
    )


def _ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


def _clip(values):
    return np.clip(np.asarray(values, np.float64), MIN_WEIGHT, MAX_WEIGHT)


def _prediction(frame, weights_by_well):
    weights = frame["well"].map(weights_by_well)
    if weights.isna().any():
        raise RuntimeError("Missing dose for an evaluation well")
    parent = frame[PARENT].to_numpy(np.float64)
    return parent + weights.to_numpy(np.float64) * (
        frame[COARSE].to_numpy(np.float64) - parent
    )


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    error = prediction - frame[TRUTH].to_numpy(np.float64)
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
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


for required in (V70_PATH, V74_PATH, V75_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
features = pd.read_parquet(V74_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
v75_weights = pd.read_parquet(V75_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
if not features[["fold", "well"]].equals(v75_weights[["fold", "well"]]):
    raise RuntimeError("V74/V75 well alignment mismatch")
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError(f"Unexpected fold coverage: {fold_wells}")

x, feature_names = _augment(features)
y = features["oracle_weight"].to_numpy(np.float64)
sample_weight = np.sqrt(features["rows"].to_numpy(np.float64))
hist_oof = np.full(len(features), np.nan, np.float64)
ridge_oof = np.full(len(features), np.nan, np.float64)

for fold in OUTER_FOLDS:
    valid = features["fold"].to_numpy() == fold
    train = ~valid
    hist = _hist_model()
    ridge = _ridge_model()
    hist.fit(x[train], y[train], sample_weight=sample_weight[train])
    ridge.fit(x[train], y[train])
    hist_oof[valid] = _clip(hist.predict(x[valid]))
    ridge_oof[valid] = _clip(ridge.predict(x[valid]))
    print(
        "V76 FOLD",
        {
            "fold": fold,
            "train_wells": int(train.sum()),
            "valid_wells": int(valid.sum()),
            "hist_mean": float(hist_oof[valid].mean()),
            "ridge_mean": float(ridge_oof[valid].mean()),
        },
        flush=True,
    )

if not np.isfinite(hist_oof).all() or not np.isfinite(ridge_oof).all():
    raise RuntimeError("Incomplete fixed-family OOF doses")
ensemble_oof = _clip(HIST_FRACTION * hist_oof + RIDGE_FRACTION * ridge_oof)

weight_table = features[["fold", "well", "oracle_weight"]].copy()
weight_table["hist_weight"] = hist_oof
weight_table["ridge_weight"] = ridge_oof
weight_table["ensemble_weight"] = ensemble_oof
weight_table["v75_nested_weight"] = v75_weights["adaptive_weight"].to_numpy(np.float64)
weight_table.to_parquet(OUT_DIR / "fixed_ensemble_weights_oof.parquet", index=False, compression="zstd")

weight_maps = {
    "fixed_w0p30": {well: FIXED_WEIGHT for well in features["well"]},
    "hist_crossfit": dict(zip(features["well"], hist_oof)),
    "ridge_crossfit": dict(zip(features["well"], ridge_oof)),
    "fixed_ensemble": dict(zip(features["well"], ensemble_oof)),
    "v75_nested_selector": dict(zip(features["well"], v75_weights["adaptive_weight"])),
}
predictions = {name: _prediction(frame, mapping) for name, mapping in weight_maps.items()}

summary_rows = []
metrics = {}
for fold in (0, *OUTER_FOLDS):
    mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].to_numpy() == fold
    local = frame.loc[mask]
    metrics[str(fold)] = {}
    for endpoint, prediction in predictions.items():
        result = _metric(local, prediction[mask], endpoint)
        metrics[str(fold)][endpoint] = result
        summary_rows.append({"fold": fold, **result})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
overall = metrics["0"]
gain = overall["fixed_w0p30"]["rmse"] - overall["fixed_ensemble"]["rmse"]
far_gain = overall["fixed_w0p30"]["far_gt2000_rmse"] - overall["fixed_ensemble"]["far_gt2000_rmse"]
loss_vs_nested = overall["fixed_ensemble"]["rmse"] - overall["v75_nested_selector"]["rmse"]
fold_gains = {
    str(fold): metrics[str(fold)]["fixed_w0p30"]["rmse"] - metrics[str(fold)]["fixed_ensemble"]["rmse"]
    for fold in OUTER_FOLDS
}

checks = {
    "overall_gain_ge_0p05": gain >= 0.05,
    "positive_folds_3of3": all(value > 0.0 for value in fold_gains.values()),
    "far_gain_positive": far_gain > 0.0,
    "loss_vs_nested_le_0p03": loss_vs_nested <= 0.03,
}
strict_pass = all(checks.values())

bundle_path = DEPLOYMENT_DIR / "adaptive_coarse_dose_bundle_v76.joblib"
if strict_pass:
    final_hist = _hist_model()
    final_ridge = _ridge_model()
    final_hist.fit(x, y, sample_weight=sample_weight)
    final_ridge.fit(x, y)
    bundle = {
        "experiment": EXPERIMENT,
        "hist_model": final_hist,
        "ridge_model": final_ridge,
        "hist_fraction": HIST_FRACTION,
        "ridge_fraction": RIDGE_FRACTION,
        "fixed_weight": FIXED_WEIGHT,
        "min_weight": MIN_WEIGHT,
        "max_weight": MAX_WEIGHT,
        "base_features": list(BASE_FEATURES),
        "log_features": list(LOG_FEATURES),
        "feature_names": feature_names,
        "training_wells": int(len(features)),
        "training_folds": list(OUTER_FOLDS),
    }
    joblib.dump(bundle, bundle_path, compress=3)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fixed_Hist_Ridge_deployment_analogue_crossfit_and_bundle",
    "selection_or_tuning_count": 0,
    "recipe": {
        "hist": "HGBR_leaves7_leaf45_l2_12",
        "ridge": "StandardScaler_Ridge_alpha100",
        "hist_fraction": HIST_FRACTION,
        "ridge_fraction": RIDGE_FRACTION,
        "weight_bounds": [MIN_WEIGHT, MAX_WEIGHT],
    },
    "coverage": {"rows": len(frame), "wells": frame["well"].nunique(), "fold_wells": fold_wells},
    "metrics": metrics,
    "gain": gain,
    "far_gain": far_gain,
    "loss_vs_v75_nested": loss_vs_nested,
    "fold_gains": fold_gains,
    "checks": checks,
    "strict_pass": strict_pass,
    "bundle": str(bundle_path) if strict_pass else None,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> integrate bundle after All773 coarse inference and before submission write"
        if strict_pass
        else "FAIL -> deployment analogue is unstable; retain fixed coarse weight 0.30"
    ),
}

print("\nV76 MATCHED SUMMARY")
print(summary.to_string(index=False))
print("\nV76 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
_write_json(OUT_DIR / "verdict.json", verdict)
print(f"saved: {OUT_DIR}")
