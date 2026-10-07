"""V93: refit the V76 dose model with pooled-SSE-consistent well weights.

V92 showed that calibrating the already-predicted V76 dose does not transfer.
This experiment instead refits the frozen V76 Hist/Ridge family from the
original label-free endpoint features.  For a well, the exact excess SSE of a
dose error is curvature * (predicted_dose - oracle_dose)^2, where curvature is
sum((coarse - parent)^2).  Fold3 selects one of three predefined curvature
weight profiles, Fold2 confirms it, and Fold1 is read only unless Fold2 passes.
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
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_v76_curvature_refit_v93"
OUT_DIR = OOF_ROOT / EXPERIMENT
DEPLOYMENT_DIR = OUT_DIR / "deployment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V74_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_oracle_v74" / "perwell_oracle_features.parquet"
V76_WEIGHT_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.60
FOLDS = (1, 2, 3)
PROFILES = ("sqrt_curvature", "clip90_curvature", "exact_curvature")
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


def _clip(values):
    return np.clip(np.asarray(values, np.float64), MIN_WEIGHT, MAX_WEIGHT)


def _profile_weight(curvature, profile):
    curvature = np.asarray(curvature, np.float64)
    positive = curvature[curvature > 1e-12]
    if len(positive) < 50:
        raise RuntimeError(f"Only {len(positive)} positive-curvature training wells")
    if profile == "sqrt_curvature":
        weight = np.sqrt(curvature)
        threshold = None
    elif profile == "clip90_curvature":
        threshold = float(np.quantile(positive, 0.90))
        weight = np.minimum(curvature, threshold)
    elif profile == "exact_curvature":
        weight = curvature.copy()
        threshold = None
    else:
        raise KeyError(profile)
    positive_weight = weight[weight > 0.0]
    weight /= float(np.mean(positive_weight))
    return weight, threshold


def _fit_family(x, y, curvature, profile):
    sample_weight, threshold = _profile_weight(curvature, profile)
    positive = sample_weight > 0.0
    hist = _hist_model()
    hist.fit(x[positive], y[positive], sample_weight=sample_weight[positive])
    scaler = StandardScaler()
    scaler.fit(x[positive], sample_weight=sample_weight[positive])
    x_scaled = scaler.transform(x[positive])
    ridge = Ridge(alpha=100.0)
    ridge.fit(x_scaled, y[positive], sample_weight=sample_weight[positive])
    return {"hist": hist, "scaler": scaler, "ridge": ridge, "threshold": threshold}


def _apply_family(model, x):
    hist = model["hist"].predict(x)
    ridge = model["ridge"].predict(model["scaler"].transform(x))
    return _clip(0.5 * hist + 0.5 * ridge)


def _prediction(local, weight_map):
    weights = local["well"].astype(str).map(weight_map)
    if weights.isna().any():
        raise RuntimeError("Missing adaptive dose")
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


for required in (V70_PATH, V74_PATH, V76_WEIGHT_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
features = pd.read_parquet(V74_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
v76 = pd.read_parquet(V76_WEIGHT_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
if not features[["fold", "well"]].equals(v76[["fold", "well"]]):
    raise RuntimeError("V74/V76 well alignment mismatch")
if frame.groupby("fold")["well"].nunique().astype(int).to_dict() != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError("Unexpected V70 fold coverage")

x, feature_names = _augment(features)
y = features["oracle_weight"].to_numpy(np.float64)
curvature = features["rows"].to_numpy(np.float64) * np.square(
    features["move_rms"].to_numpy(np.float64)
)
if (curvature < 0.0).any() or not np.isfinite(curvature).all():
    raise RuntimeError("Invalid per-well curvature")

predicted = {profile: np.full(len(features), np.nan, np.float64) for profile in PROFILES}
fit_reports = []
for heldout_fold in FOLDS:
    valid = features["fold"].to_numpy() == heldout_fold
    train = ~valid
    for profile in PROFILES:
        model = _fit_family(x[train], y[train], curvature[train], profile)
        predicted[profile][valid] = _apply_family(model, x[valid])
        fit_reports.append({
            "heldout_fold": heldout_fold,
            "profile": profile,
            "train_wells": int(train.sum()),
            "valid_wells": int(valid.sum()),
            "clip_threshold": model["threshold"],
            "mean_dose": float(predicted[profile][valid].mean()),
        })

if any(not np.isfinite(values).all() for values in predicted.values()):
    raise RuntimeError("Incomplete V93 OOF doses")

weights = features[["fold", "well", "oracle_weight"]].copy()
weights["curvature"] = curvature
weights["v76"] = v76["ensemble_weight"].to_numpy(np.float64)
for profile, values in predicted.items():
    weights[profile] = values
weights.to_parquet(OUT_DIR / "curvature_refit_weights_oof.parquet", index=False, compression="zstd")
pd.DataFrame(fit_reports).to_csv(OUT_DIR / "fit_reports.csv", index=False)


def _evaluate(fold, endpoints):
    local = frame[frame["fold"].eq(fold)].copy()
    local_weights = weights[weights["fold"].eq(fold)]
    result = {}
    rows = []
    base_map = dict(zip(local_weights["well"].astype(str), local_weights["v76"]))
    base = _metric(local, _prediction(local, base_map), "v76")
    result["v76"] = base
    rows.append({"fold": fold, **base})
    for endpoint in endpoints:
        dose_map = dict(zip(local_weights["well"].astype(str), local_weights[endpoint]))
        metric = _metric(local, _prediction(local, dose_map), endpoint)
        result[endpoint] = metric
        rows.append({"fold": fold, **metric, **_delta(base, metric)})
    return result, rows


summary_rows = []
metrics3, rows3 = _evaluate(3, PROFILES)
summary_rows.extend(rows3)
scan = []
for profile in PROFILES:
    scan.append({"profile": profile, **_delta(metrics3["v76"], metrics3[profile])})
scan_frame = pd.DataFrame(scan).sort_values("gain", ascending=False, kind="stable")
scan_frame.to_csv(OUT_DIR / "fold3_profile_scan.csv", index=False)
eligible = scan_frame[
    scan_frame["gain"].ge(0.02)
    & scan_frame["far_gain"].ge(0.0)
    & scan_frame["near_degradation"].le(0.05)
    & scan_frame["p95_gain"].ge(-0.10)
]

selected = None if eligible.empty else str(eligible.iloc[0]["profile"])
fold2_delta = None
fold1_delta = None
fold1_touched = False
strict_pass = False
if selected is not None:
    metrics2, rows2 = _evaluate(2, (selected,))
    summary_rows.extend(rows2)
    fold2_delta = _delta(metrics2["v76"], metrics2[selected])
    fold2_pass = bool(
        fold2_delta["gain"] >= 0.02
        and fold2_delta["far_gain"] >= 0.0
        and fold2_delta["near_degradation"] <= 0.05
        and fold2_delta["p95_gain"] >= -0.10
    )
    if fold2_pass:
        metrics1, rows1 = _evaluate(1, (selected,))
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
    final_model = _fit_family(x, y, curvature, selected)
    bundle = {
        "experiment": EXPERIMENT,
        "profile": selected,
        "model": final_model,
        "base_features": list(BASE_FEATURES),
        "log_features": list(LOG_FEATURES),
        "feature_names": feature_names,
        "min_weight": MIN_WEIGHT,
        "max_weight": MAX_WEIGHT,
        "training_wells": int(len(features)),
        "application_order": "V69 coarse -> V93 adaptive dose",
    }
    bundle_path = DEPLOYMENT_DIR / "adaptive_coarse_curvature_refit_v93.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_weight_profile_selection_Fold2_confirmation_Fold1_final",
    "base": "V69 All773 coarse plus V76 adaptive dose",
    "change_from_v76": "same features and model family; pooled-SSE curvature training weights only",
    "profiles": list(PROFILES),
    "selection_search_count": len(PROFILES),
    "selected_profile": selected,
    "fold3_scan": scan,
    "fold2_confirmation": fold2_delta,
    "fold1_final": fold1_delta,
    "fold1_touched": fold1_touched,
    "strict_pass": strict_pass,
    "bundle": bundle_path,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> replace the V76 bundle after V69 with the V93 bundle"
        if strict_pass else "FAIL -> retain V76; exact-objective refit does not transfer"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV93 FIT REPORTS")
print(pd.DataFrame(fit_reports).to_string(index=False), flush=True)
print("\nV93 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV93 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
