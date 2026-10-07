"""Nested cross-fit audit of smooth well-by-distance coarse dosing.

V76 predicts one All773-coarse blend dose per physical well.  V78 asks the
next narrower question: can label-free endpoint disagreement predict how that
dose should change with distance from the anchor?  Four smooth distance bases
are used, so the resulting prediction remains continuous at 500/1000/2000 ft.

The held-out fold is never used to select model family or shrinkage.  Truth is
used only to construct outer-training dose labels and to score held-out rows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_coarse_bandwise_adaptive_crossfit_v78"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V74_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_oracle_v74" / "perwell_oracle_features.parquet"
V76_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
OUTER_FOLDS = (1, 2, 3)
FIXED_WEIGHT = 0.30
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.60
TRANSITION_HALF_WIDTH_FT = 64.0
BOUNDARIES_FT = (500.0, 1000.0, 2000.0)
SHRINKAGES = (0.0, 0.25, 0.50, 0.75, 1.0)
BAND_NAMES = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
STRATA = (
    ("near_0_500", -np.inf, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)
GLOBAL_FEATURES = (
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
LOCAL_FEATURES = (
    "band_fraction", "band_distance_mean", "band_distance_std",
    "band_move_mean", "band_move_std", "band_move_rms", "band_move_abs_mean",
    "band_move_abs_p90", "band_move_abs_max", "band_move_slope_per_ft",
    "band_parent_slope_per_ft", "band_coarse_slope_per_ft",
    "band_parent_roughness", "band_coarse_roughness", "band_endpoint_correlation",
    "previous_move_rms", "next_move_rms", "previous_move_mean", "next_move_mean",
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


def _slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0.0 else 0.0


def _roughness(values):
    values = np.asarray(values, np.float64)
    return float(np.sqrt(np.mean(np.diff(values) ** 2))) if len(values) > 1 else 0.0


def _correlation(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    if len(left) < 2 or left.std() <= 0.0 or right.std() <= 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _distance_basis(distance):
    """Return four nonnegative bases summing exactly to one per row."""
    distance = np.asarray(distance, np.float64)
    basis = np.zeros((len(distance), 4), np.float64)
    hard = np.searchsorted(np.asarray(BOUNDARIES_FT), distance, side="left")
    basis[np.arange(len(distance)), hard] = 1.0
    half = TRANSITION_HALF_WIDTH_FT
    for boundary_index, boundary in enumerate(BOUNDARIES_FT):
        mask = (distance > boundary - half) & (distance < boundary + half)
        if not mask.any():
            continue
        blend = _smoothstep((distance[mask] - (boundary - half)) / (2.0 * half))
        basis[mask] = 0.0
        basis[mask, boundary_index] = 1.0 - blend
        basis[mask, boundary_index + 1] = blend
    if not np.allclose(basis.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("Distance bases do not sum to one")
    return basis


def _oracle_band_weights(parent, coarse, truth, basis, fallback):
    move = np.asarray(coarse, np.float64) - np.asarray(parent, np.float64)
    target = np.asarray(truth, np.float64) - np.asarray(parent, np.float64)
    design = basis * move[:, None]
    scale = max(float(np.trace(design.T @ design)) / 4.0, 1.0)
    ridge = 1e-6 * scale
    lhs = design.T @ design + ridge * np.eye(4)
    rhs = design.T @ target + ridge * np.asarray(fallback, np.float64)
    weights = np.linalg.solve(lhs, rhs)
    return np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)


def _row_prediction(local, band_weights):
    parent = local[PARENT].to_numpy(np.float64)
    move = local[COARSE].to_numpy(np.float64) - parent
    basis = _distance_basis(local["md_since"].to_numpy(np.float64))
    row_weight = basis @ np.asarray(band_weights, np.float64)
    return parent + row_weight * move


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    error = prediction - frame[TRUTH].to_numpy(np.float64)
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
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _hist_model():
    return HistGradientBoostingRegressor(
        max_iter=160, learning_rate=0.035, max_leaf_nodes=7,
        min_samples_leaf=45, l2_regularization=12.0, random_state=20260730,
    )


def _ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


def _fit_predict(x_train, y_train, sample_weight, x_valid):
    hist = _hist_model()
    ridge = _ridge_model()
    hist.fit(x_train, y_train, sample_weight=sample_weight)
    ridge.fit(x_train, y_train)
    prediction = 0.5 * hist.predict(x_valid) + 0.5 * ridge.predict(x_valid)
    return np.clip(prediction, MIN_WEIGHT, MAX_WEIGHT)


def _expand_predictions(frame, table, weight_column):
    output = np.empty(len(frame), np.float64)
    weight_map = {
        str(well): local.sort_values("band_index")[weight_column].to_numpy(np.float64)
        for well, local in table.groupby("well", sort=False)
    }
    for well, indices in frame.groupby("well", sort=False).indices.items():
        if str(well) not in weight_map:
            raise RuntimeError(f"Missing band weights for {well}")
        output[np.asarray(indices, np.int64)] = _row_prediction(
            frame.iloc[np.asarray(indices, np.int64)], weight_map[str(well)]
        )
    return output


for required in (V70_PATH, V74_PATH, V76_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
global_features = pd.read_parquet(V74_PATH).sort_values(["fold", "well"], kind="stable")
v76 = pd.read_parquet(V76_PATH).sort_values(["fold", "well"], kind="stable")
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate V70 ids")
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError(f"Unexpected honest-fold coverage: {fold_wells}")
if not global_features[["fold", "well"]].reset_index(drop=True).equals(
    v76[["fold", "well"]].reset_index(drop=True)
):
    raise RuntimeError("V74/V76 well alignment mismatch")

well_global = global_features.merge(
    v76[["fold", "well", "ensemble_weight"]],
    on=["fold", "well"], how="inner", validate="one_to_one",
).set_index(["fold", "well"])

rows = []
oracle_prediction = np.empty(len(frame), np.float64)
for number, ((fold, well), indices) in enumerate(frame.groupby(["fold", "well"], sort=False).indices.items(), 1):
    indices = np.asarray(indices, np.int64)
    local = frame.iloc[indices]
    distance = local["md_since"].to_numpy(np.float64)
    parent = local[PARENT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    truth = local[TRUTH].to_numpy(np.float64)
    move = coarse - parent
    baseline_weight = float(well_global.loc[(fold, well), "ensemble_weight"])
    basis = _distance_basis(distance)
    oracle_weights = _oracle_band_weights(
        parent, coarse, truth, basis, np.repeat(baseline_weight, 4)
    )
    oracle_prediction[indices] = parent + (basis @ oracle_weights) * move

    local_stats = []
    for band_index in range(4):
        contribution = basis[:, band_index]
        mask = contribution > 1e-9
        d = distance[mask]
        p = parent[mask]
        c = coarse[mask]
        m = move[mask]
        if not mask.any():
            stats = {name: 0.0 for name in LOCAL_FEATURES[:15]}
        else:
            stats = {
                "band_fraction": float(mask.mean()),
                "band_distance_mean": float(d.mean()),
                "band_distance_std": float(d.std()),
                "band_move_mean": float(m.mean()),
                "band_move_std": float(m.std()),
                "band_move_rms": float(np.sqrt(np.mean(m * m))),
                "band_move_abs_mean": float(np.abs(m).mean()),
                "band_move_abs_p90": float(np.quantile(np.abs(m), 0.90)),
                "band_move_abs_max": float(np.abs(m).max()),
                "band_move_slope_per_ft": _slope(d, m),
                "band_parent_slope_per_ft": _slope(d, p),
                "band_coarse_slope_per_ft": _slope(d, c),
                "band_parent_roughness": _roughness(p),
                "band_coarse_roughness": _roughness(c),
                "band_endpoint_correlation": _correlation(p, c),
            }
        local_stats.append(stats)

    global_row = well_global.loc[(fold, well)]
    for band_index, band_name in enumerate(BAND_NAMES):
        stats = local_stats[band_index]
        row = {
            "fold": int(fold), "well": str(well), "band_index": band_index,
            "band_name": band_name, "rows": int(np.sum(basis[:, band_index] > 1e-9)),
            "baseline_weight": baseline_weight,
            "oracle_weight": float(oracle_weights[band_index]),
            **{name: float(global_row[name]) for name in GLOBAL_FEATURES},
            **stats,
            "previous_move_rms": float(local_stats[max(0, band_index - 1)]["band_move_rms"]),
            "next_move_rms": float(local_stats[min(3, band_index + 1)]["band_move_rms"]),
            "previous_move_mean": float(local_stats[max(0, band_index - 1)]["band_move_mean"]),
            "next_move_mean": float(local_stats[min(3, band_index + 1)]["band_move_mean"]),
        }
        for index in range(4):
            row[f"band_is_{index}"] = float(index == band_index)
        rows.append(row)
    if number % 100 == 0 or number == sum(fold_wells.values()):
        print(f"V78 features {number}/{sum(fold_wells.values())}", flush=True)

features = pd.DataFrame(rows).sort_values(["fold", "well", "band_index"], kind="stable").reset_index(drop=True)
features.to_parquet(OUT_DIR / "band_oracle_features.parquet", index=False, compression="zstd")
model_features = [
    "baseline_weight", *GLOBAL_FEATURES, *LOCAL_FEATURES,
    "band_is_0", "band_is_1", "band_is_2", "band_is_3",
]
x = features[model_features].to_numpy(np.float64)
y = features["oracle_weight"].to_numpy(np.float64)
sample_weight = np.sqrt(np.maximum(features["rows"].to_numpy(np.float64), 1.0))
if not np.isfinite(x).all() or not np.isfinite(y).all():
    raise RuntimeError("Non-finite V78 training matrix")

crossfit = np.full(len(features), np.nan, np.float64)
fold_reports = []
for outer_fold in OUTER_FOLDS:
    outer_valid = features["fold"].to_numpy() == outer_fold
    train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
    shrinkage_scores = []
    for shrinkage in SHRINKAGES:
        total_sse = 0.0
        total_rows = 0
        for inner_valid_fold in train_folds:
            inner_train_fold = next(fold for fold in train_folds if fold != inner_valid_fold)
            train_mask = features["fold"].to_numpy() == inner_train_fold
            valid_mask = features["fold"].to_numpy() == inner_valid_fold
            raw = _fit_predict(x[train_mask], y[train_mask], sample_weight[train_mask], x[valid_mask])
            baseline = features.loc[valid_mask, "baseline_weight"].to_numpy(np.float64)
            predicted = np.clip(baseline + shrinkage * (raw - baseline), MIN_WEIGHT, MAX_WEIGHT)
            table = features.loc[valid_mask, ["well", "band_index"]].copy()
            table["candidate"] = predicted
            local_frame = frame[frame["fold"].eq(inner_valid_fold)]
            local_prediction = _expand_predictions(local_frame, table, "candidate")
            error = local_prediction - local_frame[TRUTH].to_numpy(np.float64)
            total_sse += float(np.dot(error, error))
            total_rows += len(error)
        shrinkage_scores.append({"shrinkage": shrinkage, "sse": total_sse, "rows": total_rows})
    selected = min(shrinkage_scores, key=lambda item: (item["sse"], item["shrinkage"]))
    train_mask = ~outer_valid
    raw = _fit_predict(x[train_mask], y[train_mask], sample_weight[train_mask], x[outer_valid])
    baseline = features.loc[outer_valid, "baseline_weight"].to_numpy(np.float64)
    predicted = np.clip(
        baseline + selected["shrinkage"] * (raw - baseline), MIN_WEIGHT, MAX_WEIGHT
    )
    crossfit[outer_valid] = predicted
    fold_reports.append({
        "heldout_fold": outer_fold,
        "train_folds": list(train_folds),
        "selected_shrinkage": selected["shrinkage"],
        "inner_rmse": float(np.sqrt(selected["sse"] / selected["rows"])),
        "mean_weight": float(predicted.mean()),
        "std_weight": float(predicted.std()),
    })
    print("V78 FOLD", fold_reports[-1], flush=True)

if not np.isfinite(crossfit).all():
    raise RuntimeError("Incomplete V78 cross-fit weights")
features["adaptive_band_weight"] = crossfit

rng = np.random.default_rng(20260730)
well_shuffled = crossfit.copy()
band_shuffled = crossfit.copy()
for fold in OUTER_FOLDS:
    fold_well_list = features.loc[features["fold"].eq(fold), "well"].drop_duplicates().tolist()
    permutation = rng.permutation(fold_well_list)
    source = {
        well: features.loc[
            features["fold"].eq(fold) & features["well"].eq(well), "adaptive_band_weight"
        ].to_numpy(np.float64)
        for well in fold_well_list
    }
    for target_well, source_well in zip(fold_well_list, permutation):
        target_mask = features["fold"].eq(fold) & features["well"].eq(target_well)
        well_shuffled[target_mask.to_numpy()] = source[source_well]
        band_shuffled[target_mask.to_numpy()] = np.roll(source[target_well], 1)
features["well_shuffled_weight"] = well_shuffled
features["band_shuffled_weight"] = band_shuffled
features.to_parquet(OUT_DIR / "bandwise_weights_oof.parquet", index=False, compression="zstd")

v76_table = features[["well", "band_index", "baseline_weight"]].copy()
predictions = {
    "fixed_w0p30": frame[PARENT].to_numpy(np.float64) + FIXED_WEIGHT * (
        frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
    ),
    "v76_well_adaptive": _expand_predictions(frame, v76_table, "baseline_weight"),
    "v78_band_adaptive": _expand_predictions(frame, features, "adaptive_band_weight"),
    "well_shuffled_control": _expand_predictions(frame, features, "well_shuffled_weight"),
    "band_shuffled_control": _expand_predictions(frame, features, "band_shuffled_weight"),
    "smooth_band_oracle": oracle_prediction,
}

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
gain = overall["v76_well_adaptive"]["rmse"] - overall["v78_band_adaptive"]["rmse"]
far_gain = overall["v76_well_adaptive"]["far_gt2000_rmse"] - overall["v78_band_adaptive"]["far_gt2000_rmse"]
near_degradation = overall["v78_band_adaptive"]["near_0_500_rmse"] - overall["v76_well_adaptive"]["near_0_500_rmse"]
fold_gains = {
    str(fold): metrics[str(fold)]["v76_well_adaptive"]["rmse"] - metrics[str(fold)]["v78_band_adaptive"]["rmse"]
    for fold in OUTER_FOLDS
}
well_shuffle_loss = overall["well_shuffled_control"]["rmse"] - overall["v78_band_adaptive"]["rmse"]
band_shuffle_loss = overall["band_shuffled_control"]["rmse"] - overall["v78_band_adaptive"]["rmse"]
checks = {
    "gain_vs_v76_ge_0p05": gain >= 0.05,
    "positive_folds_3of3": all(value > 0.0 for value in fold_gains.values()),
    "far_gain_ge_0p10": far_gain >= 0.10,
    "near_degradation_le_0p02": near_degradation <= 0.02,
    "well_shuffle_loses_ge_0p03": well_shuffle_loss >= 0.03,
    "band_shuffle_loses_ge_0p03": band_shuffle_loss >= 0.03,
}
strict_pass = all(checks.values())

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_smooth_well_by_distance_adaptive_dose_gateway",
    "sample_unit": "one complete physical well",
    "truth_usage": "outer-training smooth band dose labels and heldout metrics only",
    "protocol": {
        "honest_folds": list(OUTER_FOLDS),
        "fold_wells": fold_wells,
        "fold45_excluded_because_used_for_structural_prior": True,
        "transition_half_width_ft": TRANSITION_HALF_WIDTH_FT,
        "boundaries_ft": list(BOUNDARIES_FT),
    },
    "model": "fixed equal HistGradientBoosting(leaves7) and Ridge(alpha100)",
    "fold_reports": fold_reports,
    "metrics": metrics,
    "gain_vs_v76": gain,
    "far_gain_vs_v76": far_gain,
    "near_degradation_vs_v76": near_degradation,
    "fold_gains": fold_gains,
    "well_shuffle_loss": well_shuffle_loss,
    "band_shuffle_loss": band_shuffle_loss,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze one fixed bandwise deployment model and build Kaggle adapter"
        if strict_pass else
        "FAIL -> distance-conditioned dosing is not stable; retain V76 well-only adaptive dose"
    ),
}

print("\nV78 MATCHED SUMMARY")
print(summary.to_string(index=False))
print("\nV78 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
_write_json(OUT_DIR / "verdict.json", verdict)
print(f"saved: {OUT_DIR}")
