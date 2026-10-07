"""Low-frequency residual head on the exact V76 honest-fold endpoint.

This gateway predicts smooth residual control nodes rather than another coarse
blend dose.  Targets are constructed only on outer-training wells.  Held-out
truth is used for metrics and matched controls only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_lowfreq_residual_crossfit_v80b"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V74_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_oracle_v74" / "perwell_oracle_features.parquet"
V76_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
OUTER_FOLDS = (1, 2, 3)
NODE_SPACING_FT = 256.0
TARGET_HALF_WIDTH_FT = 128.0
TARGET_SIGMA_FT = 96.0
FEATURE_HALF_WIDTHS_FT = (128.0, 256.0, 512.0)
ANCHOR_RAMP_FT = 256.0
TARGET_CLIP_FT = 12.0
CORRECTION_CAP_FT = 2.0
SHRINKAGES = (0.0, 0.25, 0.50, 0.75, 1.0)
RANDOM_SEED = 20260730

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


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


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
    if len(left) < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


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


def _v76_prediction(frame, weights):
    row_weight = frame["well"].map(weights)
    if row_weight.isna().any():
        missing = frame.loc[row_weight.isna(), "well"].drop_duplicates().head().tolist()
        raise RuntimeError(f"Missing V76 weights for {missing}")
    parent = frame[PARENT].to_numpy(np.float64)
    move = frame[COARSE].to_numpy(np.float64) - parent
    return parent + row_weight.to_numpy(np.float64) * move


def _local_stats(distance, parent, coarse, base, centre, half_width):
    mask = np.abs(distance - centre) <= half_width
    if mask.sum() < 4:
        nearest = np.argsort(np.abs(distance - centre))[: min(8, len(distance))]
        mask = np.zeros(len(distance), bool)
        mask[nearest] = True
    d = distance[mask]
    p = parent[mask]
    c = coarse[mask]
    b = base[mask]
    move = c - p
    anchor_p = float(parent[0])
    anchor_c = float(coarse[0])
    anchor_b = float(base[0])
    prefix = f"w{int(half_width)}"
    return {
        f"{prefix}_rows": float(mask.sum()),
        f"{prefix}_move_mean": float(move.mean()),
        f"{prefix}_move_std": float(move.std()),
        f"{prefix}_move_rms": float(np.sqrt(np.mean(move * move))),
        f"{prefix}_move_abs_p90": float(np.quantile(np.abs(move), 0.90)),
        f"{prefix}_move_slope": _slope(d, move),
        f"{prefix}_parent_rel_mean": float((p - anchor_p).mean()),
        f"{prefix}_coarse_rel_mean": float((c - anchor_c).mean()),
        f"{prefix}_base_rel_mean": float((b - anchor_b).mean()),
        f"{prefix}_parent_slope": _slope(d, p),
        f"{prefix}_coarse_slope": _slope(d, c),
        f"{prefix}_base_slope": _slope(d, b),
        f"{prefix}_parent_roughness": _roughness(p),
        f"{prefix}_coarse_roughness": _roughness(c),
        f"{prefix}_base_roughness": _roughness(b),
        f"{prefix}_parent_coarse_corr": _correlation(p, c),
    }


def _build_nodes(frame, global_features, base_prediction):
    global_lookup = global_features.set_index(["fold", "well"])
    rows = []
    oracle_correction = np.empty(len(frame), np.float64)
    for number, ((fold, well), indices) in enumerate(frame.groupby(["fold", "well"], sort=False).indices.items(), 1):
        indices = np.asarray(indices, np.int64)
        local = frame.iloc[indices]
        distance = local["md_since"].to_numpy(np.float64)
        parent = local[PARENT].to_numpy(np.float64)
        coarse = local[COARSE].to_numpy(np.float64)
        base = base_prediction[indices]
        residual = local[TRUTH].to_numpy(np.float64) - base
        maximum = float(distance.max())
        nodes = np.arange(0.0, maximum + NODE_SPACING_FT, NODE_SPACING_FT, dtype=np.float64)
        if nodes[-1] < maximum:
            nodes = np.append(nodes, maximum)
        else:
            nodes[-1] = maximum
        nodes = np.unique(nodes)
        target_values = []
        global_row = global_lookup.loc[(fold, well)]
        base_weight = float(global_row["ensemble_weight"])
        for node_index, centre in enumerate(nodes):
            target_mask = np.abs(distance - centre) <= TARGET_HALF_WIDTH_FT
            if target_mask.sum() < 4:
                nearest = np.argsort(np.abs(distance - centre))[: min(8, len(distance))]
                target_mask = np.zeros(len(distance), bool)
                target_mask[nearest] = True
            target_distance = distance[target_mask]
            target_weight = np.exp(-0.5 * ((target_distance - centre) / TARGET_SIGMA_FT) ** 2)
            target = float(np.average(residual[target_mask], weights=target_weight))
            target = float(np.clip(target, -TARGET_CLIP_FT, TARGET_CLIP_FT))
            target_values.append(target)
            row = {
                "fold": int(fold), "well": str(well), "node_index": int(node_index),
                "node_distance_ft": float(centre), "maximum_distance_ft_node": maximum,
                "progress": float(centre / max(maximum, 1.0)),
                "log1p_distance": float(np.log1p(max(centre, 0.0))),
                "distance_sqrt": float(np.sqrt(max(centre, 0.0))),
                "distance_gt500": float(centre > 500.0),
                "distance_gt1000": float(centre > 1000.0),
                "distance_gt2000": float(centre > 2000.0),
                "base_weight": base_weight,
                "target_residual": target,
                "target_rows": int(target_mask.sum()),
            }
            for column in GLOBAL_FEATURES:
                row[f"global_{column}"] = float(global_row[column])
            for half_width in FEATURE_HALF_WIDTHS_FT:
                row.update(_local_stats(distance, parent, coarse, base, centre, half_width))
            rows.append(row)
        oracle_nodes = np.clip(np.asarray(target_values, np.float64), -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
        oracle_nodes[0] = 0.0
        correction = np.interp(distance, nodes, oracle_nodes)
        correction *= _smoothstep(distance / ANCHOR_RAMP_FT)
        oracle_correction[indices] = correction
        if number % 100 == 0 or number == frame["well"].nunique():
            print(f"V80b nodes {number}/{frame['well'].nunique()}", flush=True)
    output = pd.DataFrame(rows).sort_values(["fold", "well", "node_index"], kind="stable").reset_index(drop=True)
    return output, oracle_correction


class _RidgeModel:
    def __init__(self, alpha):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, x, y, sample_weight):
        transformed = self.scaler.fit_transform(x)
        self.model.fit(transformed, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))


def _make_model(config):
    if config["family"] == "ridge":
        return _RidgeModel(config["alpha"])
    if config["family"] == "hist":
        return HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.035,
            max_leaf_nodes=config["leaves"], min_samples_leaf=config["leaf"],
            l2_regularization=config["l2"], random_state=RANDOM_SEED,
        )
    if config["family"] == "extra":
        return ExtraTreesRegressor(
            n_estimators=300, max_depth=config["depth"],
            min_samples_leaf=config["leaf"], max_features=0.75,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    raise KeyError(config["family"])


CONFIGS = (
    {"id": "ridge_a30", "family": "ridge", "alpha": 30.0},
    {"id": "ridge_a100", "family": "ridge", "alpha": 100.0},
    {"id": "hist_l7_n30_r8", "family": "hist", "leaves": 7, "leaf": 30, "l2": 8.0},
    {"id": "hist_l11_n45_r16", "family": "hist", "leaves": 11, "leaf": 45, "l2": 16.0},
    {"id": "extra_d4_n20", "family": "extra", "depth": 4, "leaf": 20},
)


def _fit(config, x, y, sample_weight):
    model = _make_model(config)
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _smooth_nodes(values):
    values = np.asarray(values, np.float64).copy()
    for _ in range(2):
        if len(values) > 2:
            values[1:-1] = 0.25 * values[:-2] + 0.50 * values[1:-1] + 0.25 * values[2:]
    values[0] = 0.0
    return np.clip(values, -CORRECTION_CAP_FT, CORRECTION_CAP_FT)


def _expand(frame, nodes, prediction_column):
    output = np.empty(len(frame), np.float64)
    node_map = {
        str(well): local.sort_values("node_index", kind="stable")
        for well, local in nodes.groupby("well", sort=False)
    }
    for well, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        local = frame.iloc[indices]
        table = node_map[str(well)]
        node_distance = table["node_distance_ft"].to_numpy(np.float64)
        node_value = _smooth_nodes(table[prediction_column].to_numpy(np.float64))
        distance = local["md_since"].to_numpy(np.float64)
        correction = np.interp(distance, node_distance, node_value)
        correction *= _smoothstep(distance / ANCHOR_RAMP_FT)
        output[indices] = correction
    return output


for required in (V70_PATH, V74_PATH, V76_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
global_features = pd.read_parquet(V74_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
v76_weights = pd.read_parquet(V76_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)

fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155, 3: 154} or frame["id"].duplicated().any():
    raise RuntimeError(f"Unexpected V70 coverage: {fold_wells}")
if not global_features[["fold", "well"]].equals(v76_weights[["fold", "well"]]):
    raise RuntimeError("V74/V76 well alignment mismatch")

global_features = global_features.merge(
    v76_weights[["fold", "well", "ensemble_weight"]],
    on=["fold", "well"], how="inner", validate="one_to_one",
)
weight_map = dict(zip(global_features["well"], global_features["ensemble_weight"]))
base_prediction = _v76_prediction(frame, weight_map)
nodes, oracle_correction = _build_nodes(frame, global_features, base_prediction)
nodes.to_parquet(OUT_DIR / "node_training_table.parquet", index=False, compression="zstd")

key_columns = {"fold", "well", "node_index", "target_residual", "target_rows"}
model_features = [column for column in nodes.columns if column not in key_columns]
x = nodes[model_features].to_numpy(np.float64)
y = nodes["target_residual"].to_numpy(np.float64)
sample_weight = np.sqrt(np.maximum(nodes["target_rows"].to_numpy(np.float64), 1.0))
if not np.isfinite(x).all() or not np.isfinite(y).all():
    raise RuntimeError("Non-finite V80b training matrix")

crossfit_nodes = np.full(len(nodes), np.nan, np.float64)
outer_reports = []
inner_reports = []
fold_values = nodes["fold"].to_numpy(np.int8)

for outer_fold in OUTER_FOLDS:
    outer_valid = fold_values == outer_fold
    train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
    candidates = []
    for config in CONFIGS:
        split_predictions = []
        for inner_valid_fold in train_folds:
            train_mask = fold_values == next(fold for fold in train_folds if fold != inner_valid_fold)
            valid_mask = fold_values == inner_valid_fold
            model = _fit(config, x[train_mask], y[train_mask], sample_weight[train_mask])
            raw = np.clip(model.predict(x[valid_mask]), -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
            table = nodes.loc[valid_mask, ["fold", "well", "node_index", "node_distance_ft"]].copy()
            local_frame = frame[frame["fold"].eq(inner_valid_fold)]
            local_base = base_prediction[frame["fold"].to_numpy() == inner_valid_fold]
            split_predictions.append((inner_valid_fold, raw, table, local_frame, local_base))
        for shrinkage in SHRINKAGES:
            total_sse = 0.0
            total_rows = 0
            for inner_valid_fold, raw, table_base, local_frame, local_base in split_predictions:
                table = table_base.copy()
                table["candidate"] = shrinkage * raw
                correction = _expand(local_frame, table, "candidate")
                error = local_base + correction - local_frame[TRUTH].to_numpy(np.float64)
                total_sse += float(np.dot(error, error))
                total_rows += len(error)
            rmse = float(np.sqrt(total_sse / total_rows))
            report = {"outer_fold": outer_fold, "config": config["id"], "shrinkage": shrinkage, "inner_rmse": rmse}
            inner_reports.append(report)
            candidates.append((rmse, config["id"], shrinkage, config))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, selected_shrinkage, selected_config = candidates[0]
    model = _fit(selected_config, x[~outer_valid], y[~outer_valid], sample_weight[~outer_valid])
    raw = np.clip(model.predict(x[outer_valid]), -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
    crossfit_nodes[outer_valid] = selected_shrinkage * raw
    outer_reports.append({
        "heldout_fold": outer_fold, "train_folds": list(train_folds),
        "selected_config": selected_config["id"], "selected_shrinkage": selected_shrinkage,
        "inner_rmse": float(candidates[0][0]), "valid_nodes": int(outer_valid.sum()),
        "mean_abs_node_correction": float(np.mean(np.abs(crossfit_nodes[outer_valid]))),
    })
    print("V80b FOLD", outer_reports[-1], flush=True)

if not np.isfinite(crossfit_nodes).all():
    raise RuntimeError("Incomplete V80b cross-fit node predictions")

nodes["crossfit_residual"] = crossfit_nodes
nodes.to_parquet(OUT_DIR / "node_predictions_oof.parquet", index=False, compression="zstd")
pd.DataFrame(inner_reports).to_csv(OUT_DIR / "inner_selection.csv", index=False)
pd.DataFrame(outer_reports).to_csv(OUT_DIR / "outer_models.csv", index=False)

correction = _expand(frame, nodes, "crossfit_residual")
candidate_prediction = base_prediction + correction
opposite_prediction = base_prediction - correction
oracle_prediction = base_prediction + oracle_correction

rng = np.random.default_rng(RANDOM_SEED)
shuffled_correction = np.empty(len(frame), np.float64)
for fold in OUTER_FOLDS:
    fold_mask = frame["fold"].to_numpy() == fold
    fold_well_list = frame.loc[fold_mask, "well"].drop_duplicates().tolist()
    source_wells = rng.permutation(fold_well_list)
    source_curves = {}
    for well in fold_well_list:
        indices = np.asarray(frame.index[frame["well"].eq(well)], np.int64)
        local_distance = frame.loc[indices, "md_since"].to_numpy(np.float64)
        progress = local_distance / max(float(local_distance.max()), 1.0)
        source_curves[well] = (progress, correction[indices])
    for target_well, source_well in zip(fold_well_list, source_wells):
        indices = np.asarray(frame.index[frame["well"].eq(target_well)], np.int64)
        local_distance = frame.loc[indices, "md_since"].to_numpy(np.float64)
        progress = local_distance / max(float(local_distance.max()), 1.0)
        source_progress, source_value = source_curves[str(source_well)]
        shuffled_correction[indices] = np.interp(progress, source_progress, source_value)
shuffled_prediction = base_prediction + shuffled_correction

predictions = {
    "v76_well_adaptive": base_prediction,
    "lowfreq_residual_crossfit": candidate_prediction,
    "opposite_residual_control": opposite_prediction,
    "well_shuffled_residual_control": shuffled_prediction,
    "cap2_lowfreq_oracle": oracle_prediction,
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
row_output = frame[["id", "well", "row_idx", "fold", TRUTH, "md_since"]].copy()
row_output["v76_well_adaptive"] = base_prediction.astype(np.float32)
row_output["lowfreq_residual_crossfit"] = candidate_prediction.astype(np.float32)
row_output["lowfreq_correction"] = correction.astype(np.float32)
row_output.to_parquet(OUT_DIR / "lowfreq_residual_oof.parquet", index=False, compression="zstd")

overall = metrics["0"]
gain = overall["v76_well_adaptive"]["rmse"] - overall["lowfreq_residual_crossfit"]["rmse"]
far_gain = overall["v76_well_adaptive"]["far_gt2000_rmse"] - overall["lowfreq_residual_crossfit"]["far_gt2000_rmse"]
near_degradation = overall["lowfreq_residual_crossfit"]["near_0_500_rmse"] - overall["v76_well_adaptive"]["near_0_500_rmse"]
p95_degradation = overall["lowfreq_residual_crossfit"]["p95_well_rmse"] - overall["v76_well_adaptive"]["p95_well_rmse"]
shuffle_loss = overall["well_shuffled_residual_control"]["rmse"] - overall["lowfreq_residual_crossfit"]["rmse"]
opposite_loss = overall["opposite_residual_control"]["rmse"] - overall["lowfreq_residual_crossfit"]["rmse"]
fold_gains = {
    str(fold): metrics[str(fold)]["v76_well_adaptive"]["rmse"] - metrics[str(fold)]["lowfreq_residual_crossfit"]["rmse"]
    for fold in OUTER_FOLDS
}
checks = {
    "gain_vs_v76_ge_0p03": gain >= 0.03,
    "positive_folds_3of3": all(value > 0.0 for value in fold_gains.values()),
    "far_gain_nonnegative": far_gain >= 0.0,
    "near_degradation_le_0p02": near_degradation <= 0.02,
    "p95_degradation_le_0p05": p95_degradation <= 0.05,
    "aligned_beats_shuffle_ge_0p02": shuffle_loss >= 0.02,
    "aligned_beats_opposite_ge_0p03": opposite_loss >= 0.03,
}
strict_pass = all(checks.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_low_frequency_residual_head_on_exact_V76_endpoint",
    "sample_unit": "one complete physical well",
    "truth_usage": "outer-training node targets and heldout metrics only",
    "protocol": {
        "honest_folds": list(OUTER_FOLDS), "fold_wells": fold_wells,
        "fold45_excluded_because_used_for_structural_prior": True,
        "node_spacing_ft": NODE_SPACING_FT, "target_half_width_ft": TARGET_HALF_WIDTH_FT,
        "feature_half_widths_ft": list(FEATURE_HALF_WIDTHS_FT),
        "anchor_ramp_ft": ANCHOR_RAMP_FT, "correction_cap_ft": CORRECTION_CAP_FT,
    },
    "sources": {"v70": str(V70_PATH), "v74": str(V74_PATH), "v76": str(V76_PATH)},
    "features": model_features,
    "outer_models": outer_reports,
    "metrics": metrics,
    "gain_vs_v76": gain, "far_gain_vs_v76": far_gain,
    "near_degradation_vs_v76": near_degradation,
    "p95_degradation_vs_v76": p95_degradation,
    "shuffle_loss": shuffle_loss, "opposite_loss": opposite_loss,
    "fold_gains": fold_gains, "checks": checks, "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the residual recipe and build an all-773 deployment model"
        if strict_pass else
        "FAIL -> low-frequency residual is not predictably recoverable; retain V76 unchanged"
    ),
}

print("\nV80b MATCHED SUMMARY")
print(summary.to_string(index=False))
print("\nV80b OUTER MODELS")
print(pd.DataFrame(outer_reports).to_string(index=False))
print("\nV80b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
_write_json(OUT_DIR / "verdict.json", verdict)
print(f"saved: {OUT_DIR}")
