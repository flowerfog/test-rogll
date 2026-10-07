"""Cross-fit legal confidence dose for the frozen V28 CP8 tail endpoint.

Fold 3 is the only training/selection population. Five grouped inner folds
select a small regressor and shrinkage around the robust fixed weight 0.35.
The fitted rule is then applied once to Fold 2 and Fold 1. Features use only
the two deployable prediction paths and measured distance; truth is used only
to construct Fold-3 training targets and evaluation metrics.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE_DIR = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28"
SOURCE_PATH = SOURCE_DIR / "tail_endpoint_oof.parquet"
SOURCE_VERDICT = SOURCE_DIR / "verdict.json"
OUT_DIR = OOF_ROOT / "frontier_6p39_tail_confidence_dose_v30"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLD = 3
CONFIRMATION_FOLDS = (2, 1)
FIXED_WEIGHT = 0.35
MAX_WEIGHT = 1.0
SHRINKAGES = (0.0, 0.25, 0.50, 0.75, 1.0)
SEED = 20260727

for required in (SOURCE_PATH, SOURCE_VERDICT):
    if not required.is_file():
        raise FileNotFoundError(required)

source_verdict = json.loads(SOURCE_VERDICT.read_text(encoding="utf-8"))
if source_verdict.get("fold3_selected_source") != "cp8_endpoint":
    raise RuntimeError("V28 did not freeze cp8_endpoint")

frame = pd.read_parquet(SOURCE_PATH)
required_columns = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "parent", "neighbor_v27", "tail_cp8_endpoint",
}
missing = sorted(required_columns.difference(frame.columns))
if missing:
    raise RuntimeError(f"V28 output missing {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)

truth = frame["tvt_true"].to_numpy(np.float64)
neighbor = frame["neighbor_v27"].to_numpy(np.float64)
endpoint = frame["tail_cp8_endpoint"].to_numpy(np.float64)
parent = frame["parent"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
row_fold = frame["fold"].to_numpy(np.int8)
direction = endpoint - neighbor


def _metrics(mask, prediction):
    local_error = prediction[mask] - truth[mask]
    local_wells = frame.loc[mask, "well"].to_numpy()
    bywell = (
        pd.DataFrame({"well": local_wells, "error2": local_error * local_error})
        .groupby("well", sort=False)["error2"].mean().pow(0.5)
    )
    result = {
        "rows": int(mask.sum()),
        "wells": int(pd.Series(local_wells).nunique()),
        "exact_sse": float(np.dot(local_error, local_error)),
        "rmse": float(np.sqrt(np.mean(local_error * local_error))),
        "mae": float(np.mean(np.abs(local_error))),
        "median_well_rmse": float(bywell.median()),
        "p75_well_rmse": float(bywell.quantile(0.75)),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    local_distance = distance[mask]
    for name, band in {
        "near_0_500_rmse": local_distance <= 500.0,
        "mid_500_1000_rmse": (local_distance > 500.0) & (local_distance <= 1000.0),
        "mid_1000_2000_rmse": (local_distance > 1000.0) & (local_distance <= 2000.0),
        "far_gt2000_rmse": local_distance > 2000.0,
    }.items():
        result[name] = (
            float(np.sqrt(np.mean(local_error[band] ** 2))) if band.any() else None
        )
    return result


def _summaries(values, prefix):
    values = np.asarray(values, np.float64)
    absolute = np.abs(values)
    if len(values) == 0:
        return {f"{prefix}_{name}": 0.0 for name in (
            "mean", "std", "rms", "abs_mean", "abs_p50", "abs_p90",
            "abs_p99", "endpoint", "positive_fraction", "roughness",
        )}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_abs_mean": float(np.mean(absolute)),
        f"{prefix}_abs_p50": float(np.quantile(absolute, 0.50)),
        f"{prefix}_abs_p90": float(np.quantile(absolute, 0.90)),
        f"{prefix}_abs_p99": float(np.quantile(absolute, 0.99)),
        f"{prefix}_endpoint": float(values[-1]),
        f"{prefix}_positive_fraction": float(np.mean(values > 0.0)),
        f"{prefix}_roughness": (
            float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
        ),
    }


feature_rows = []
row_groups = {}
for well, indices in frame.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    row_groups[str(well)] = indices
    local_distance = distance[indices]
    local_direction = direction[indices]
    local_parent_move = endpoint[indices] - parent[indices]
    tail = local_distance > 1000.0
    far = local_distance > 2000.0
    error_neighbor = neighbor[indices] - truth[indices]
    denominator = float(np.dot(local_direction, local_direction))
    oracle_weight = (
        float(np.clip(-np.dot(error_neighbor, local_direction) / denominator, 0.0, 1.0))
        if denominator > 1e-12 else 0.0
    )
    row = {
        "well": str(well),
        "fold": int(row_fold[indices[0]]),
        "rows": int(len(indices)),
        "max_md_since": float(np.max(local_distance)),
        "tail_fraction": float(np.mean(tail)),
        "far_fraction": float(np.mean(far)),
        "oracle_weight": oracle_weight,
    }
    row.update(_summaries(local_direction[tail], "gap_tail"))
    row.update(_summaries(local_direction[far], "gap_far"))
    row.update(_summaries(local_parent_move[tail], "endpoint_move_tail"))
    if tail.sum() >= 3:
        x = local_distance[tail]
        x = (x - x.mean()) / max(x.std(), 1.0)
        row["gap_tail_slope"] = float(np.dot(x, local_direction[tail]) / np.dot(x, x))
        a = local_direction[tail]
        b = local_parent_move[tail]
        row["gap_parent_move_corr"] = (
            float(np.corrcoef(a, b)[0, 1])
            if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0
        )
    else:
        row["gap_tail_slope"] = 0.0
        row["gap_parent_move_corr"] = 0.0
    feature_rows.append(row)

wells = pd.DataFrame(feature_rows).sort_values("well").reset_index(drop=True)
feature_names = [
    column for column in wells.columns
    if column not in {"well", "fold", "rows", "oracle_weight"}
]
X = wells[feature_names].replace([np.inf, -np.inf], np.nan)
y = wells["oracle_weight"].to_numpy(np.float64)
well_fold = wells["fold"].to_numpy(np.int8)
well_position = dict(zip(wells["well"], np.arange(len(wells))))
row_well_position = frame["well"].map(well_position).to_numpy(np.int64)


def _prediction(weights):
    return neighbor + np.clip(weights[row_well_position], 0.0, MAX_WEIGHT) * direction


models = {
    "ridge_1": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
    "ridge_10": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
    "ridge_100": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=100.0)),
    "extra_leaf12": make_pipeline(SimpleImputer(), ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=12, max_features=0.75,
        random_state=SEED, n_jobs=-1,
    )),
    "extra_leaf24": make_pipeline(SimpleImputer(), ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=24, max_features=0.75,
        random_state=SEED + 1, n_jobs=-1,
    )),
    "hist_leaf20": make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.04, max_leaf_nodes=7,
        min_samples_leaf=20, l2_regularization=10.0, random_state=SEED,
    )),
}

train_positions = np.flatnonzero(well_fold == TRAIN_FOLD)
if len(train_positions) < 150:
    raise RuntimeError(f"Only {len(train_positions)} Fold3 training wells")
kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
scan_rows = []
oof_store = {}
train_row_mask = row_fold == TRAIN_FOLD
neighbor_train_metric = _metrics(train_row_mask, neighbor)
fixed_weights = np.full(len(wells), FIXED_WEIGHT, np.float64)
fixed_train_metric = _metrics(train_row_mask, _prediction(fixed_weights))

for model_name, template in models.items():
    raw_oof = np.full(len(wells), np.nan, np.float64)
    for local_train, local_valid in kfold.split(train_positions):
        fit_positions = train_positions[local_train]
        valid_positions = train_positions[local_valid]
        fitted = clone(template).fit(X.iloc[fit_positions], y[fit_positions])
        raw_oof[valid_positions] = fitted.predict(X.iloc[valid_positions])
    if not np.isfinite(raw_oof[train_positions]).all():
        raise RuntimeError(f"Incomplete Fold3 OOF for {model_name}")
    oof_store[model_name] = raw_oof
    for shrinkage in SHRINKAGES:
        weights = fixed_weights.copy()
        weights[train_positions] = np.clip(
            FIXED_WEIGHT
            + float(shrinkage) * (raw_oof[train_positions] - FIXED_WEIGHT),
            0.0,
            MAX_WEIGHT,
        )
        metric = _metrics(train_row_mask, _prediction(weights))
        p95_change = metric["p95_well_rmse"] - neighbor_train_metric["p95_well_rmse"]
        scan_rows.append({
            "model": model_name,
            "shrinkage": float(shrinkage),
            "mean_weight": float(np.mean(weights[train_positions])),
            "gain_vs_neighbor": neighbor_train_metric["rmse"] - metric["rmse"],
            "increment_vs_fixed": fixed_train_metric["rmse"] - metric["rmse"],
            "p95_change_vs_neighbor": p95_change,
            "risk_pass": bool(p95_change <= 0.10),
            "selection_score": float(metric["rmse"] + 0.5 * max(p95_change - 0.10, 0.0)),
            **metric,
        })

scan = pd.DataFrame(scan_rows).sort_values(
    ["risk_pass", "selection_score", "p95_well_rmse"],
    ascending=[False, True, True], kind="stable",
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_inner_oof_scan.csv", index=False)
selected = scan.iloc[0]
selected_model = str(selected["model"])
selected_shrinkage = float(selected["shrinkage"])

fitted = clone(models[selected_model]).fit(X.iloc[train_positions], y[train_positions])
raw_all = fitted.predict(X)
adaptive_weights = np.clip(
    FIXED_WEIGHT + selected_shrinkage * (raw_all - FIXED_WEIGHT),
    0.0, MAX_WEIGHT,
)
adaptive_prediction = _prediction(adaptive_weights)
fixed_prediction = _prediction(fixed_weights)

rng = np.random.default_rng(SEED + 77)
shuffle_weights = adaptive_weights.copy()
for fold_value in (3, 2, 1):
    positions = np.flatnonzero(well_fold == fold_value)
    shuffle_weights[positions] = adaptive_weights[rng.permutation(positions)]
shuffle_prediction = _prediction(shuffle_weights)

details = {}
rows = []
for fold_value, role in ((3, "train_refit_diagnostic"), (2, "confirmation"), (1, "final")):
    mask = row_fold == fold_value
    endpoints = {
        "neighbor_v27": neighbor,
        "fixed_0p35": fixed_prediction,
        "adaptive": adaptive_prediction,
        "weight_shuffle": shuffle_prediction,
    }
    local = {name: _metrics(mask, prediction) for name, prediction in endpoints.items()}
    details[f"fold{fold_value}_{role}"] = local
    for name, metric in local.items():
        rows.append({"fold": fold_value, "role": role, "endpoint": name, **metric})

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
wells["adaptive_weight"] = adaptive_weights
wells["shuffled_weight"] = shuffle_weights
wells.to_csv(OUT_DIR / "well_weights.csv", index=False)

checks = {}
for fold_value, role in ((2, "confirmation"), (1, "final")):
    local = details[f"fold{fold_value}_{role}"]
    prefix = f"fold{fold_value}"
    checks[f"{prefix}_adaptive_gain_positive"] = (
        local["neighbor_v27"]["rmse"] - local["adaptive"]["rmse"] > 0.0
    )
    checks[f"{prefix}_adaptive_not_worse_than_fixed_by_0p02"] = (
        local["adaptive"]["rmse"] - local["fixed_0p35"]["rmse"] <= 0.02
    )
    checks[f"{prefix}_p95_change_le_0p10"] = (
        local["adaptive"]["p95_well_rmse"]
        - local["neighbor_v27"]["p95_well_rmse"] <= 0.10
    )
    checks[f"{prefix}_aligned_weights_beat_shuffle"] = (
        local["adaptive"]["rmse"] < local["weight_shuffle"]["rmse"]
    )

strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "frontier_6p39_tail_confidence_dose_v30",
    "stage": "Fold3_inner_crossfit_train_Fold2_confirmation_Fold1_final",
    "source": str(SOURCE_PATH),
    "features": feature_names,
    "selected_model": selected_model,
    "selected_shrinkage": selected_shrinkage,
    "fixed_weight": FIXED_WEIGHT,
    "fold3_inner_selection": selected.to_dict(),
    "metrics": details,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / 1024**3,
    "next": (
        "PASS -> package the adaptive tail weight with fixed-0.35 fallback"
        if strict_pass else
        "FAIL -> confidence is not transferable; fixed-0.35 remains an LB candidate but fails the strict p95 gate"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=lambda value: float(value)
               if isinstance(value, (np.floating, np.integer)) else value),
    encoding="utf-8",
)
print("\nTAIL CONFIDENCE INNER SCAN")
print(scan.head(20).to_string(index=False), flush=True)
print("\nTAIL CONFIDENCE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=str), flush=True)
print("saved:", OUT_DIR, flush=True)
