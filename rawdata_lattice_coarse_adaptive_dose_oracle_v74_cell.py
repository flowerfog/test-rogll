"""Oracle and observability gateway for adaptive All773 coarse dosing.

The source is the honest V70 Fold1/2/3 cross-fit endpoint on the complete
6.377 parent.  Truth is used only to build oracle targets and metrics.  Every
candidate feature is derived from the two deployable prediction endpoints and
distance from the prediction anchor.
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
EXPERIMENT = "rawdata_lattice_coarse_adaptive_dose_oracle_v74"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = (
    OOF_ROOT
    / "rawdata_lattice_coarse_all773_crossfit_v70"
    / "all773_proxy_crossfit_oof.parquet"
)
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
FIXED_WEIGHT = 0.30
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.60
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


def _optimal_weight(parent, coarse, truth, fallback=FIXED_WEIGHT):
    move = np.asarray(coarse, np.float64) - np.asarray(parent, np.float64)
    target = np.asarray(truth, np.float64) - np.asarray(parent, np.float64)
    denominator = float(np.dot(move, move))
    if denominator <= 1e-12:
        return float(fallback)
    return float(np.clip(np.dot(move, target) / denominator, MIN_WEIGHT, MAX_WEIGHT))


def _slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0.0 else 0.0


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
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


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(
    SOURCE,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("V70 source contains duplicate ids")
if frame[[TRUTH, "md_since", PARENT, COARSE]].isna().any().any():
    raise RuntimeError("V70 source contains missing values")
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError(f"Unexpected fold coverage: {fold_wells}")

fixed_prediction = (
    frame[PARENT].to_numpy(np.float64)
    + FIXED_WEIGHT
    * (frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64))
)
perwell_prediction = np.empty(len(frame), np.float64)
stratified_prediction = np.empty(len(frame), np.float64)
feature_rows = []

for number, ((fold, well), indices) in enumerate(
    frame.groupby(["fold", "well"], sort=False).indices.items(), start=1
):
    indices = np.asarray(indices, np.int64)
    local = frame.iloc[indices]
    parent = local[PARENT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    truth = local[TRUTH].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    move = coarse - parent
    weight = _optimal_weight(parent, coarse, truth)
    perwell_prediction[indices] = parent + weight * move

    stratum_weights = {}
    local_stratified = parent.copy()
    for name, lower, upper in STRATA:
        mask = (distance > lower) & (distance <= upper)
        local_weight = (
            _optimal_weight(parent[mask], coarse[mask], truth[mask], fallback=weight)
            if mask.any()
            else weight
        )
        stratum_weights[name] = local_weight
        local_stratified[mask] = parent[mask] + local_weight * move[mask]
    stratified_prediction[indices] = local_stratified

    fixed = parent + FIXED_WEIGHT * move
    oracle = parent + weight * move
    fixed_sse = float(np.sum((fixed - truth) ** 2))
    oracle_sse = float(np.sum((oracle - truth) ** 2))
    row = {
        "fold": int(fold),
        "well": str(well),
        "rows": int(len(local)),
        "maximum_distance_ft": float(distance.max()),
        "oracle_weight": weight,
        "oracle_sse_gain_vs_fixed": fixed_sse - oracle_sse,
        "oracle_rmse_gain_vs_fixed": float(
            np.sqrt(fixed_sse / len(local)) - np.sqrt(oracle_sse / len(local))
        ),
        "move_mean": float(move.mean()),
        "move_std": float(move.std()),
        "move_rms": float(np.sqrt(np.mean(move * move))),
        "move_abs_mean": float(np.abs(move).mean()),
        "move_abs_p90": float(np.quantile(np.abs(move), 0.90)),
        "move_abs_max": float(np.max(np.abs(move))),
        "move_slope_per_ft": _slope(distance, move),
        "endpoint_correlation": float(np.corrcoef(parent, coarse)[0, 1])
        if np.std(parent) > 0.0 and np.std(coarse) > 0.0
        else 0.0,
        "parent_slope_per_ft": _slope(distance, parent),
        "coarse_slope_per_ft": _slope(distance, coarse),
        "parent_roughness": float(np.sqrt(np.mean(np.diff(parent) ** 2)))
        if len(parent) > 1
        else 0.0,
        "coarse_roughness": float(np.sqrt(np.mean(np.diff(coarse) ** 2)))
        if len(coarse) > 1
        else 0.0,
    }
    for name, lower, upper in STRATA:
        mask = (distance > lower) & (distance <= upper)
        row[f"{name}_fraction"] = float(mask.mean())
        row[f"{name}_move_rms"] = (
            float(np.sqrt(np.mean(move[mask] ** 2))) if mask.any() else 0.0
        )
        row[f"{name}_move_mean"] = float(move[mask].mean()) if mask.any() else 0.0
        row[f"{name}_oracle_weight"] = float(stratum_weights[name])
    feature_rows.append(row)
    if number % 100 == 0 or number == len(fold_wells) + sum(fold_wells.values()) - len(fold_wells):
        print(f"V74 oracle wells {number}/{sum(fold_wells.values())}", flush=True)

features = pd.DataFrame(feature_rows)
features.to_parquet(OUT_DIR / "perwell_oracle_features.parquet", index=False, compression="zstd")

target_columns = ("oracle_weight", "oracle_sse_gain_vs_fixed", "oracle_rmse_gain_vs_fixed")
excluded = {"fold", "well", "rows", *target_columns}
feature_columns = [
    column
    for column in features.columns
    if column not in excluded and not column.endswith("_oracle_weight")
]
association_rows = []
for feature in feature_columns:
    for target in target_columns:
        correlation = features[[feature, target]].corr(method="spearman").iloc[0, 1]
        association_rows.append(
            {
                "feature": feature,
                "target": target,
                "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
                "abs_spearman": float(abs(correlation)) if np.isfinite(correlation) else 0.0,
            }
        )
associations = pd.DataFrame(association_rows).sort_values(
    ["target", "abs_spearman"], ascending=[True, False], kind="stable"
)
associations.to_csv(OUT_DIR / "feature_associations.csv", index=False)

metric_rows = []
metrics = {}
for fold in (0, 1, 2, 3):
    mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].to_numpy() == fold
    local = frame.loc[mask]
    metrics[str(fold)] = {}
    for endpoint, prediction in {
        "parent": frame.loc[mask, PARENT].to_numpy(np.float64),
        "fixed_w0p30": fixed_prediction[mask],
        "perwell_oracle": perwell_prediction[mask],
        "perwell_distance_oracle": stratified_prediction[mask],
    }.items():
        result = _metric(local, prediction, endpoint)
        metrics[str(fold)][endpoint] = result
        metric_rows.append({"fold": fold, **result})
metrics_frame = pd.DataFrame(metric_rows)
metrics_frame.to_csv(OUT_DIR / "oracle_summary.csv", index=False)

overall = metrics["0"]
oracle_gain = float(overall["fixed_w0p30"]["rmse"] - overall["perwell_oracle"]["rmse"])
distance_oracle_gain = float(
    overall["fixed_w0p30"]["rmse"] - overall["perwell_distance_oracle"]["rmse"]
)
deployable_associations = associations[associations["target"].isin(target_columns)]
best_association = float(deployable_associations["abs_spearman"].max())
checks = {
    "perwell_oracle_gain_ge_0p30": bool(oracle_gain >= 0.30),
    "distance_oracle_gain_ge_0p40": bool(distance_oracle_gain >= 0.40),
    "best_deployable_abs_spearman_ge_0p15": bool(best_association >= 0.15),
    "perwell_oracle_improves_each_fold": bool(
        all(
            metrics[str(fold)]["perwell_oracle"]["rmse"]
            < metrics[str(fold)]["fixed_w0p30"]["rmse"]
            for fold in (1, 2, 3)
        )
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "All773_adaptive_dose_oracle_and_label_free_observability_gateway",
    "truth_usage": "oracle targets and metrics only",
    "deployable_features": "endpoint disagreement and anchor-distance summaries only",
    "coverage": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "fold_wells": fold_wells,
    },
    "fixed_weight": FIXED_WEIGHT,
    "weight_bounds": [MIN_WEIGHT, MAX_WEIGHT],
    "metrics": metrics,
    "oracle_gain_vs_fixed": oracle_gain,
    "distance_oracle_gain_vs_fixed": distance_oracle_gain,
    "best_deployable_abs_spearman": best_association,
    "top_associations": associations.groupby("target", sort=False).head(10).to_dict("records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train V75 fixed-complexity cross-fit adaptive dose model"
        if strict_pass
        else "FAIL -> adaptive dose is not observably reachable; retain fixed All773 weight 0.30"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV74 ORACLE SUMMARY")
print(metrics_frame.to_string(index=False), flush=True)
print("\nV74 TOP FEATURE ASSOCIATIONS")
print(associations.groupby("target", sort=False).head(10).to_string(index=False), flush=True)
print("\nV74 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
