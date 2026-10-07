"""Nested cross-fit adaptive dosing for the All773 coarse endpoint.

V74 established a large per-well dose oracle and measurable label-free
associations.  This cell tests whether that oracle is reachable without
using the held-out fold for model or shrinkage selection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_coarse_adaptive_dose_crossfit_v75"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = (
    OOF_ROOT
    / "rawdata_lattice_coarse_all773_crossfit_v70"
    / "all773_proxy_crossfit_oof.parquet"
)
V74_PATH = (
    OOF_ROOT
    / "rawdata_lattice_coarse_adaptive_dose_oracle_v74"
    / "perwell_oracle_features.parquet"
)

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
FIXED_WEIGHT = 0.30
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.60
OUTER_FOLDS = (1, 2, 3)
SHRINKAGES = (0.25, 0.50, 0.75, 1.00)
FEATURES = (
    "maximum_distance_ft",
    "move_mean",
    "move_std",
    "move_rms",
    "move_abs_mean",
    "move_abs_p90",
    "move_abs_max",
    "move_slope_per_ft",
    "endpoint_correlation",
    "parent_slope_per_ft",
    "coarse_slope_per_ft",
    "parent_roughness",
    "coarse_roughness",
    "near_0_500_fraction",
    "near_0_500_move_rms",
    "near_0_500_move_mean",
    "mid_500_1000_fraction",
    "mid_500_1000_move_rms",
    "mid_500_1000_move_mean",
    "mid_1000_2000_fraction",
    "mid_1000_2000_move_rms",
    "mid_1000_2000_move_mean",
    "far_gt2000_fraction",
    "far_gt2000_move_rms",
    "far_gt2000_move_mean",
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


def _model(config):
    family = config["family"]
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=config["alpha"]))
    if family == "extra":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=20260730,
            n_jobs=-1,
        )
    if family == "hist":
        return HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.035,
            max_leaf_nodes=config["max_leaf_nodes"],
            min_samples_leaf=config["min_samples_leaf"],
            l2_regularization=config["l2_regularization"],
            random_state=20260730,
        )
    raise KeyError(family)


CONFIGS = (
    {"id": "ridge_a1", "family": "ridge", "alpha": 1.0},
    {"id": "ridge_a10", "family": "ridge", "alpha": 10.0},
    {"id": "ridge_a100", "family": "ridge", "alpha": 100.0},
    {
        "id": "extra_d2_l20_f07",
        "family": "extra",
        "max_depth": 2,
        "min_samples_leaf": 20,
        "max_features": 0.7,
    },
    {
        "id": "extra_d3_l20_f07",
        "family": "extra",
        "max_depth": 3,
        "min_samples_leaf": 20,
        "max_features": 0.7,
    },
    {
        "id": "extra_d4_l30_f10",
        "family": "extra",
        "max_depth": 4,
        "min_samples_leaf": 30,
        "max_features": 1.0,
    },
    {
        "id": "hist_n7_l30_r4",
        "family": "hist",
        "max_leaf_nodes": 7,
        "min_samples_leaf": 30,
        "l2_regularization": 4.0,
    },
    {
        "id": "hist_n7_l45_r12",
        "family": "hist",
        "max_leaf_nodes": 7,
        "min_samples_leaf": 45,
        "l2_regularization": 12.0,
    },
)


def _augment(features):
    output = features.loc[:, FEATURES].astype(np.float64).copy()
    for column in (
        "move_std",
        "move_rms",
        "move_abs_mean",
        "move_abs_p90",
        "move_abs_max",
        "parent_roughness",
        "coarse_roughness",
        "near_0_500_move_rms",
        "mid_500_1000_move_rms",
        "mid_1000_2000_move_rms",
        "far_gt2000_move_rms",
    ):
        output[f"log1p_{column}"] = np.log1p(np.maximum(output[column], 0.0))
    output["roughness_ratio"] = output["coarse_roughness"] / np.maximum(
        output["parent_roughness"], 1e-4
    )
    output["slope_difference"] = (
        output["coarse_slope_per_ft"] - output["parent_slope_per_ft"]
    )
    values = output.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite adaptive-dose features")
    return values, list(output.columns)


def _weights_to_rows(frame, well_weights):
    weights = frame["well"].map(well_weights)
    if weights.isna().any():
        missing = frame.loc[weights.isna(), "well"].drop_duplicates().head().tolist()
        raise RuntimeError(f"Missing predicted weights for {missing}")
    return weights.to_numpy(np.float64)


def _prediction(frame, well_weights):
    weights = _weights_to_rows(frame, well_weights)
    parent = frame[PARENT].to_numpy(np.float64)
    move = frame[COARSE].to_numpy(np.float64) - parent
    return parent + weights * move


def _sse(frame, prediction):
    error = np.asarray(prediction, np.float64) - frame[TRUTH].to_numpy(np.float64)
    return float(np.dot(error, error))


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
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


for required in (V70_PATH, V74_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
features = pd.read_parquet(V74_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)

if frame["id"].duplicated().any() or features.duplicated(["fold", "well"]).any():
    raise RuntimeError("Duplicate row or well keys")
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError(f"Unexpected V70 coverage: {fold_wells}")
if features.groupby("fold")["well"].nunique().astype(int).to_dict() != fold_wells:
    raise RuntimeError("V74 feature coverage differs from V70")

x_all, feature_names = _augment(features)
y_all = features["oracle_weight"].to_numpy(np.float64)
sample_weight = np.sqrt(features["rows"].to_numpy(np.float64))
crossfit_weights = np.full(len(features), np.nan, np.float64)
outer_reports = []
inner_reports = []

for outer_fold in OUTER_FOLDS:
    outer_mask = features["fold"].to_numpy() == outer_fold
    train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
    candidates = []
    for config in CONFIGS:
        for shrinkage in SHRINKAGES:
            total_sse = 0.0
            total_rows = 0
            for inner_valid_fold in train_folds:
                inner_train_fold = next(f for f in train_folds if f != inner_valid_fold)
                train_mask = features["fold"].to_numpy() == inner_train_fold
                valid_mask = features["fold"].to_numpy() == inner_valid_fold
                model = _model(config)
                model.fit(
                    x_all[train_mask],
                    y_all[train_mask],
                    **({"sample_weight": sample_weight[train_mask]} if config["family"] != "ridge" else {}),
                )
                raw = np.clip(model.predict(x_all[valid_mask]), MIN_WEIGHT, MAX_WEIGHT)
                predicted = np.clip(
                    FIXED_WEIGHT + shrinkage * (raw - FIXED_WEIGHT), MIN_WEIGHT, MAX_WEIGHT
                )
                local_features = features.loc[valid_mask, ["well"]]
                weight_map = dict(zip(local_features["well"], predicted))
                local_frame = frame[frame["fold"].eq(inner_valid_fold)]
                total_sse += _sse(local_frame, _prediction(local_frame, weight_map))
                total_rows += len(local_frame)
            rmse = float(np.sqrt(total_sse / total_rows))
            report = {
                "outer_fold": outer_fold,
                "config": config["id"],
                "family": config["family"],
                "shrinkage": shrinkage,
                "inner_rmse": rmse,
            }
            inner_reports.append(report)
            candidates.append((rmse, config["id"], shrinkage, config))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, selected_shrinkage, selected_config = candidates[0]

    train_mask = ~outer_mask
    model = _model(selected_config)
    model.fit(
        x_all[train_mask],
        y_all[train_mask],
        **({"sample_weight": sample_weight[train_mask]} if selected_config["family"] != "ridge" else {}),
    )
    raw = np.clip(model.predict(x_all[outer_mask]), MIN_WEIGHT, MAX_WEIGHT)
    predicted = np.clip(
        FIXED_WEIGHT + selected_shrinkage * (raw - FIXED_WEIGHT), MIN_WEIGHT, MAX_WEIGHT
    )
    crossfit_weights[outer_mask] = predicted
    outer_reports.append(
        {
            "outer_fold": outer_fold,
            "train_folds": list(train_folds),
            "valid_wells": int(outer_mask.sum()),
            "selected_config": selected_config["id"],
            "selected_family": selected_config["family"],
            "selected_shrinkage": selected_shrinkage,
            "mean_weight": float(predicted.mean()),
            "std_weight": float(predicted.std()),
            "min_weight": float(predicted.min()),
            "max_weight": float(predicted.max()),
        }
    )
    print("V75 OUTER", outer_reports[-1], flush=True)

if not np.isfinite(crossfit_weights).all():
    raise RuntimeError("Cross-fit weights are incomplete")

weight_frame = features[["fold", "well", "oracle_weight"]].copy()
weight_frame["adaptive_weight"] = crossfit_weights
weight_frame["fixed_weight"] = FIXED_WEIGHT
weight_frame.to_parquet(OUT_DIR / "adaptive_weights_oof.parquet", index=False, compression="zstd")
pd.DataFrame(inner_reports).to_csv(OUT_DIR / "inner_selection.csv", index=False)
pd.DataFrame(outer_reports).to_csv(OUT_DIR / "outer_models.csv", index=False)

adaptive_map = dict(zip(weight_frame["well"], weight_frame["adaptive_weight"]))
fixed_map = {well: FIXED_WEIGHT for well in weight_frame["well"]}
adaptive_prediction = _prediction(frame, adaptive_map)
fixed_prediction = _prediction(frame, fixed_map)

rng = np.random.default_rng(20260730)
shuffled_weights = crossfit_weights.copy()
for fold in OUTER_FOLDS:
    mask = features["fold"].to_numpy() == fold
    shuffled_weights[mask] = rng.permutation(shuffled_weights[mask])
shuffled_map = dict(zip(features["well"], shuffled_weights))
shuffled_prediction = _prediction(frame, shuffled_map)

summary_rows = []
metrics = {}
for fold in (0, *OUTER_FOLDS):
    mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].to_numpy() == fold
    local = frame.loc[mask]
    metrics[str(fold)] = {}
    for endpoint, prediction in (
        ("fixed_w0p30", fixed_prediction[mask]),
        ("adaptive_crossfit", adaptive_prediction[mask]),
        ("shuffled_weight", shuffled_prediction[mask]),
    ):
        result = _metric(local, prediction, endpoint)
        metrics[str(fold)][endpoint] = result
        summary_rows.append({"fold": fold, **result})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
overall = metrics["0"]
gain = float(overall["fixed_w0p30"]["rmse"] - overall["adaptive_crossfit"]["rmse"])
far_gain = float(
    overall["fixed_w0p30"]["far_gt2000_rmse"]
    - overall["adaptive_crossfit"]["far_gt2000_rmse"]
)
shuffle_increment = float(
    overall["shuffled_weight"]["rmse"] - overall["adaptive_crossfit"]["rmse"]
)
fold_gains = {
    str(fold): float(
        metrics[str(fold)]["fixed_w0p30"]["rmse"]
        - metrics[str(fold)]["adaptive_crossfit"]["rmse"]
    )
    for fold in OUTER_FOLDS
}
checks = {
    "overall_gain_ge_0p03": gain >= 0.03,
    "positive_folds_3of3": all(value > 0.0 for value in fold_gains.values()),
    "far_gain_nonnegative": far_gain >= 0.0,
    "adaptive_beats_shuffled_ge_0p02": shuffle_increment >= 0.02,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_inner_selection_outer_crossfit_adaptive_All773_coarse_dose",
    "truth_usage": "outer-fold truth is metrics only; model and shrinkage use other folds",
    "source": {"v70": str(V70_PATH), "v74": str(V74_PATH)},
    "coverage": {"rows": len(frame), "wells": frame["well"].nunique(), "fold_wells": fold_wells},
    "fixed_weight": FIXED_WEIGHT,
    "weight_bounds": [MIN_WEIGHT, MAX_WEIGHT],
    "features": feature_names,
    "outer_models": outer_reports,
    "metrics": metrics,
    "overall_gain": gain,
    "far_gain": far_gain,
    "shuffle_increment": shuffle_increment,
    "fold_gains": fold_gains,
    "checks": checks,
    "strict_pass": all(checks.values()),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze model family/shrinkage and build all-773 deployment bundle"
        if all(checks.values())
        else "FAIL -> oracle is not reachable by stable label-free dose prediction; retain fixed 0.30"
    ),
}

print("\nV75 MATCHED SUMMARY")
print(summary.to_string(index=False))
print("\nV75 OUTER MODELS")
print(pd.DataFrame(outer_reports).to_string(index=False))
print("\nV75 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
_write_json(OUT_DIR / "verdict.json", verdict)
print(f"saved: {OUT_DIR}")
