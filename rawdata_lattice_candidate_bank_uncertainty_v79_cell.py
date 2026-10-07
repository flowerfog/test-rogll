"""V79: candidate-bank uncertainty as a label-free adaptive-dose signal.

Reconstruct the exact V70 candidate bank with its leave-one-fold-out ranker,
compress top-k score/path dispersion into well-by-distance-band features, and
test whether those features predict the V78 smooth-band oracle dose beyond the
frozen V76 well-only dose.  Candidate trajectories are never persisted.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import types
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_candidate_bank_uncertainty_v79"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "bank_feature_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V66_PATH = OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet"
V67_DIR = OOF_ROOT / "rawdata_lattice_coarse_canonical_gateway_v67"
V67_PRIOR = V67_DIR / "canonical_fold45_prior.json"
V67_CACHE = V67_DIR / "train_samples"
V68_CACHE = OOF_ROOT / "rawdata_lattice_coarse_canonical_final_bundle_v68" / "sampled_training_rows"
V70_DIR = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70"
V70_PATH = V70_DIR / "all773_proxy_crossfit_oof.parquet"
V78_DIR = OOF_ROOT / "rawdata_lattice_coarse_bandwise_adaptive_crossfit_v78"
V78_FEATURES = V78_DIR / "band_oracle_features.parquet"
V78_WEIGHTS = V78_DIR / "bandwise_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
OUTER_FOLDS = (1, 2, 3)
TOP_K = 16
BOUNDARIES_FT = (500.0, 1000.0, 2000.0)
TRANSITION_HALF_WIDTH_FT = 125.0
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.75
SHRINKAGES = (0.0, 0.25, 0.50, 0.75, 1.0)

GLOBAL_FEATURES = (
    "maximum_distance_ft", "move_mean", "move_std", "move_rms",
    "move_abs_mean", "move_abs_p90", "move_abs_max", "move_slope_per_ft",
    "endpoint_correlation", "parent_slope_per_ft", "coarse_slope_per_ft",
    "parent_roughness", "coarse_roughness", "near_0_500_fraction",
    "mid_500_1000_fraction", "mid_1000_2000_fraction", "far_gt2000_fraction",
    "near_0_500_move_rms", "near_0_500_move_mean",
    "mid_500_1000_move_rms", "mid_500_1000_move_mean",
    "mid_1000_2000_move_rms", "mid_1000_2000_move_mean",
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
BANK_GLOBAL_FEATURES = (
    "bank_log_candidate_count", "bank_top1_top2_margin", "bank_top1_top8_margin",
    "bank_topk_score_std", "bank_score_entropy", "bank_effective_fraction",
    "bank_unique_source_masks", "bank_source_entropy", "bank_top1_source_count",
    "bank_top8_same_source_fraction", "bank_available",
)
BANK_LOCAL_FEATURES = (
    "bank_path_std_mean", "bank_path_std_rms", "bank_path_std_p90",
    "bank_weighted_std_mean", "bank_weighted_std_rms", "bank_weighted_std_p90",
    "bank_top1_median_rms", "bank_top1_weighted_mean_rms",
    "bank_endpoint_std", "bank_slope_std", "bank_rows_fraction",
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


def _bootstrap_v4():
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 bootstrap marker missing from {V4_SOURCE}")
    module = types.ModuleType("raw_lattice_v4_v79_definitions")
    module.__file__ = str(V4_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V4_SOURCE), "exec"), module.__dict__)
    return module


def _sample_path(well):
    return next((path for path in (V68_CACHE / f"{well}.npz", V67_CACHE / f"{well}.npz") if path.is_file()), None)


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _distance_basis(distance):
    distance = np.asarray(distance, np.float64)
    basis = np.zeros((len(distance), 4), np.float64)
    hard = np.searchsorted(np.asarray(BOUNDARIES_FT), distance, side="left")
    basis[np.arange(len(distance)), hard] = 1.0
    for index, boundary in enumerate(BOUNDARIES_FT):
        mask = (distance > boundary - TRANSITION_HALF_WIDTH_FT) & (distance < boundary + TRANSITION_HALF_WIDTH_FT)
        if mask.any():
            blend = _smoothstep((distance[mask] - (boundary - TRANSITION_HALF_WIDTH_FT)) / (2.0 * TRANSITION_HALF_WIDTH_FT))
            basis[mask] = 0.0
            basis[mask, index] = 1.0 - blend
            basis[mask, index + 1] = blend
    return basis


def _hard_band(distance):
    return np.searchsorted(np.asarray(BOUNDARIES_FT), np.asarray(distance, np.float64), side="left")


def _entropy(probability):
    probability = np.asarray(probability, np.float64)
    probability = probability[probability > 0.0]
    if len(probability) <= 1:
        return 0.0
    return float(-np.sum(probability * np.log(probability)) / np.log(len(probability)))


def _score_probability(scores):
    scores = np.asarray(scores, np.float64)
    median = float(np.median(scores))
    scale = max(float(np.median(np.abs(scores - median))) * 1.4826, float(np.std(scores)) * 0.25, 1e-3)
    logits = np.clip((scores - np.max(scores)) / scale, -40.0, 0.0)
    probability = np.exp(logits)
    return probability / probability.sum()


def _slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 2:
        return 0.0
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered, y - y.mean()) / denominator) if denominator > 0.0 else 0.0


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
        "endpoint": endpoint, "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()), "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()), "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    masks = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in masks.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _hist_model():
    return HistGradientBoostingRegressor(
        max_iter=160, learning_rate=0.035, max_leaf_nodes=7,
        min_samples_leaf=45, l2_regularization=12.0, random_state=20260730,
    )


def _fit_predict(x_train, y_train, sample_weight, x_valid):
    hist = _hist_model()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    hist.fit(x_train, y_train, sample_weight=sample_weight)
    ridge.fit(x_train, y_train)
    return np.clip(0.5 * hist.predict(x_valid) + 0.5 * ridge.predict(x_valid), MIN_WEIGHT, MAX_WEIGHT)


def _row_prediction(local, weights):
    parent = local[PARENT].to_numpy(np.float64)
    move = local[COARSE].to_numpy(np.float64) - parent
    return parent + (_distance_basis(local["md_since"].to_numpy(np.float64)) @ np.asarray(weights, np.float64)) * move


def _expand_predictions(frame, table, weight_column):
    output = np.empty(len(frame), np.float64)
    mapping = {
        str(well): local.sort_values("band_index")[weight_column].to_numpy(np.float64)
        for well, local in table.groupby("well", sort=False)
    }
    for well, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        output[indices] = _row_prediction(frame.iloc[indices], mapping[str(well)])
    return output


for required in (V4_SOURCE, V66_PATH, V67_PRIOR, V70_PATH, V78_FEATURES, V78_WEIGHTS):
    if not required.is_file():
        raise FileNotFoundError(required)
for fold in OUTER_FOLDS:
    model_path = V70_DIR / "models" / f"coarse_ranker_leave_fold{fold}_out.txt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing exact V70 ranker: {model_path}")

Q = _bootstrap_v4()
V = Q.V
B = Q.B
prior_payload = json.loads(V67_PRIOR.read_text(encoding="utf-8"))
prior = {key: np.asarray(value, np.float64) if key == "rate_grid" else value for key, value in prior_payload.items()}

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
features = pd.read_parquet(V78_FEATURES).sort_values(["fold", "well", "band_index"], kind="stable").reset_index(drop=True)
v78_weights = pd.read_parquet(V78_WEIGHTS).sort_values(["fold", "well", "band_index"], kind="stable").reset_index(drop=True)
if frame.groupby("fold")["well"].nunique().astype(int).to_dict() != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError("Unexpected V70 honest-fold coverage")
if not features[["fold", "well", "band_index"]].equals(v78_weights[["fold", "well", "band_index"]]):
    raise RuntimeError("V78 feature/weight alignment mismatch")
features["v78_adaptive_band_weight"] = v78_weights["adaptive_band_weight"].to_numpy(np.float64)

sample = next((_sample_path(well) for well in frame["well"].drop_duplicates() if _sample_path(well) is not None), None)
if sample is None:
    raise RuntimeError("Cannot recover V70 feature schema")
with np.load(sample, allow_pickle=False) as data:
    feature_names = json.loads(str(data["feature_names_json"].item()))


def _bank_features(well, fold, model):
    cache_path = CACHE_DIR / f"fold{fold}_{well}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        # V79 initially required the rebuilt argmax to equal the persisted V70
        # path. Near-tied candidates can exchange rank on replay, so retry only
        # those unavailable wells and identify the path V70 actually selected.
        if cached.get("status") == "built":
            return cached
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable")
    report = {"well": str(well), "fold": int(fold), "status": "built", "reason": "", "rows": []}
    try:
        record = V["_load_record"](well)
        paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            evidence, names = Q._candidate_evidence(record, prior, paths, segments, arm_rank, arm_score_z, source_mask)
        if list(names) != list(feature_names):
            raise RuntimeError("feature schema mismatch")
        scores = np.asarray(model.predict(evidence), np.float64)
        order = np.argsort(scores)[::-1]
        search = order[: min(max(64, TOP_K), len(order))]
        eval_rows = record["eval_rows"]
        search_trajectories = np.stack([
            B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
            for index in search
        ]).astype(np.float64)
        persisted = local[COARSE].to_numpy(np.float64)
        replay_gap = np.max(np.abs(search_trajectories - persisted[None, :]), axis=1)
        selected_position = int(np.argmin(replay_gap))
        if float(replay_gap[selected_position]) > 0.01:
            raise RuntimeError(
                f"persisted V70 path absent from top64: best_gap={replay_gap[selected_position]}"
            )
        selected_index = int(search[selected_position])
        top = np.asarray(
            [selected_index]
            + [int(index) for index in order if int(index) != selected_index][: TOP_K - 1],
            dtype=np.int64,
        )
        top_scores = scores[top]
        probability = _score_probability(top_scores)
        trajectory_by_index = {
            int(index): search_trajectories[position]
            for position, index in enumerate(search)
        }
        trajectories = np.stack([
            trajectory_by_index[int(index)]
            if int(index) in trajectory_by_index else
            B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
            for index in top
        ]).astype(np.float64)
        distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
        truth = record["truth"][eval_rows].astype(np.float64)
        if len(local) != len(eval_rows):
            raise RuntimeError(f"row count {len(local)}/{len(eval_rows)}")
        truth_gap = float(np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)))
        distance_gap = float(np.max(np.abs(local["md_since"].to_numpy(np.float64) - distance)))
        parity = float(np.max(np.abs(local[COARSE].to_numpy(np.float64) - trajectories[0])))
        if truth_gap > 0.01 or distance_gap > 1.01 or parity > 0.01:
            raise RuntimeError(f"alignment truth={truth_gap} distance={distance_gap} top1={parity}")
        masks = source_mask[top].astype(np.int64)
        unique_masks, mask_counts = np.unique(masks, return_counts=True)
        mask_probability = mask_counts / mask_counts.sum()
        same_source_fraction = float(np.mean(masks[: min(8, len(masks))] == masks[0]))
        score8 = top_scores[: min(8, len(top_scores))]
        global_stats = {
            "bank_log_candidate_count": float(np.log1p(len(paths))),
            "bank_top1_top2_margin": float(top_scores[0] - top_scores[min(1, len(top_scores) - 1)]),
            "bank_top1_top8_margin": float(top_scores[0] - np.mean(score8)),
            "bank_topk_score_std": float(np.std(top_scores)),
            "bank_score_entropy": _entropy(probability),
            "bank_effective_fraction": float(1.0 / np.sum(probability**2) / len(probability)),
            "bank_unique_source_masks": float(len(unique_masks)),
            "bank_source_entropy": _entropy(mask_probability),
            "bank_top1_source_count": float(int(masks[0]).bit_count()),
            "bank_top8_same_source_fraction": same_source_fraction,
            "bank_available": 1.0,
        }
        hard_band = _hard_band(distance)
        for band_index in range(4):
            selected = hard_band == band_index
            if not selected.any():
                local_stats = {name: 0.0 for name in BANK_LOCAL_FEATURES}
            else:
                values = trajectories[:, selected]
                row_std = np.std(values, axis=0)
                weighted_mean = np.sum(probability[:, None] * values, axis=0)
                weighted_std = np.sqrt(np.sum(probability[:, None] * (values - weighted_mean) ** 2, axis=0))
                median_path = np.median(values, axis=0)
                slopes = np.asarray([_slope(distance[selected], candidate) for candidate in values], np.float64)
                local_stats = {
                    "bank_path_std_mean": float(np.mean(row_std)),
                    "bank_path_std_rms": float(np.sqrt(np.mean(row_std**2))),
                    "bank_path_std_p90": float(np.quantile(row_std, 0.90)),
                    "bank_weighted_std_mean": float(np.mean(weighted_std)),
                    "bank_weighted_std_rms": float(np.sqrt(np.mean(weighted_std**2))),
                    "bank_weighted_std_p90": float(np.quantile(weighted_std, 0.90)),
                    "bank_top1_median_rms": float(np.sqrt(np.mean((values[0] - median_path) ** 2))),
                    "bank_top1_weighted_mean_rms": float(np.sqrt(np.mean((values[0] - weighted_mean) ** 2))),
                    "bank_endpoint_std": float(np.std(values[:, -1])),
                    "bank_slope_std": float(np.std(slopes)),
                    "bank_rows_fraction": float(np.mean(selected)),
                }
            report["rows"].append({"well": str(well), "fold": int(fold), "band_index": band_index, **global_stats, **local_stats})
    except Exception as exc:
        report["status"] = "unavailable"
        report["reason"] = repr(exc)
        neutral = {name: 0.0 for name in (*BANK_GLOBAL_FEATURES, *BANK_LOCAL_FEATURES)}
        for band_index in range(4):
            report["rows"].append({"well": str(well), "fold": int(fold), "band_index": band_index, **neutral})
    _write_json(cache_path, report)
    return report


bank_rows = []
bank_reports = []
well_rows = frame[["fold", "well"]].drop_duplicates().sort_values(["fold", "well"]).reset_index(drop=True)
models = {fold: lgb.Booster(model_file=str(V70_DIR / "models" / f"coarse_ranker_leave_fold{fold}_out.txt")) for fold in OUTER_FOLDS}
for number, row in enumerate(well_rows.itertuples(index=False), 1):
    report = _bank_features(str(row.well), int(row.fold), models[int(row.fold)])
    bank_reports.append({key: value for key, value in report.items() if key != "rows"})
    bank_rows.extend(report["rows"])
    if number % 10 == 0 or number == len(well_rows):
        failures = sum(item["status"] != "built" for item in bank_reports)
        print(f"V79 candidate-bank {number}/{len(well_rows)} unavailable={failures} elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True)

bank = pd.DataFrame(bank_rows).sort_values(["fold", "well", "band_index"], kind="stable").reset_index(drop=True)
if not bank[["fold", "well", "band_index"]].equals(features[["fold", "well", "band_index"]]):
    raise RuntimeError("Candidate-bank/V78 feature alignment mismatch")
pd.DataFrame(bank_reports).to_csv(OUT_DIR / "bank_build_report.csv", index=False)
bank.to_parquet(OUT_DIR / "candidate_bank_features.parquet", index=False, compression="zstd")
features = features.merge(bank, on=["fold", "well", "band_index"], how="left", validate="one_to_one")

BASE_MODEL_FEATURES = ["baseline_weight", *GLOBAL_FEATURES, *LOCAL_FEATURES, "band_is_0", "band_is_1", "band_is_2", "band_is_3"]
ALIGNED_MODEL_FEATURES = [*BASE_MODEL_FEATURES, *BANK_GLOBAL_FEATURES, *BANK_LOCAL_FEATURES]
target = features["oracle_weight"].to_numpy(np.float64)
sample_weight = np.sqrt(np.maximum(features["rows"].to_numpy(np.float64), 1.0))


def _crossfit(feature_frame, model_features, label):
    x = feature_frame[model_features].to_numpy(np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite {label} matrix")
    output = np.full(len(feature_frame), np.nan, np.float64)
    reports = []
    fold_values = feature_frame["fold"].to_numpy()
    for outer_fold in OUTER_FOLDS:
        outer_valid = fold_values == outer_fold
        train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
        trials = []
        for shrinkage in SHRINKAGES:
            sse = 0.0
            rows = 0
            for inner_valid_fold in train_folds:
                inner_train_fold = next(fold for fold in train_folds if fold != inner_valid_fold)
                train_mask = fold_values == inner_train_fold
                valid_mask = fold_values == inner_valid_fold
                raw = _fit_predict(x[train_mask], target[train_mask], sample_weight[train_mask], x[valid_mask])
                baseline = feature_frame.loc[valid_mask, "baseline_weight"].to_numpy(np.float64)
                predicted = np.clip(baseline + shrinkage * (raw - baseline), MIN_WEIGHT, MAX_WEIGHT)
                table = feature_frame.loc[valid_mask, ["well", "band_index"]].copy()
                table["candidate"] = predicted
                local_frame = frame[frame["fold"].eq(inner_valid_fold)]
                error = _expand_predictions(local_frame, table, "candidate") - local_frame[TRUTH].to_numpy(np.float64)
                sse += float(np.dot(error, error))
                rows += len(error)
            trials.append({"shrinkage": shrinkage, "sse": sse, "rows": rows})
        selected = min(trials, key=lambda item: (item["sse"], item["shrinkage"]))
        train_mask = ~outer_valid
        raw = _fit_predict(x[train_mask], target[train_mask], sample_weight[train_mask], x[outer_valid])
        baseline = feature_frame.loc[outer_valid, "baseline_weight"].to_numpy(np.float64)
        output[outer_valid] = np.clip(baseline + selected["shrinkage"] * (raw - baseline), MIN_WEIGHT, MAX_WEIGHT)
        reports.append({
            "label": label, "heldout_fold": outer_fold, "train_folds": list(train_folds),
            "selected_shrinkage": selected["shrinkage"],
            "inner_rmse": float(np.sqrt(selected["sse"] / selected["rows"])),
            "mean_weight": float(np.mean(output[outer_valid])),
        })
        print("V79 FOLD", reports[-1], flush=True)
    if not np.isfinite(output).all():
        raise RuntimeError(f"Incomplete {label} crossfit")
    return output, reports


base_replay, base_reports = _crossfit(features, BASE_MODEL_FEATURES, "v78_feature_replay")
aligned, aligned_reports = _crossfit(features, ALIGNED_MODEL_FEATURES, "bank_aligned")

shuffled = features.copy()
rng = np.random.default_rng(20260730)
for fold in OUTER_FOLDS:
    wells = shuffled.loc[shuffled["fold"].eq(fold), "well"].drop_duplicates().tolist()
    permutation = rng.permutation(wells)
    source = {
        well: shuffled.loc[shuffled["fold"].eq(fold) & shuffled["well"].eq(well), [*BANK_GLOBAL_FEATURES, *BANK_LOCAL_FEATURES]].sort_index().to_numpy(np.float64)
        for well in wells
    }
    for target_well, source_well in zip(wells, permutation):
        mask = shuffled["fold"].eq(fold) & shuffled["well"].eq(target_well)
        target_indices = shuffled.index[mask]
        values = source[source_well]
        if len(target_indices) != len(values):
            raise RuntimeError("Bank shuffle band count mismatch")
        shuffled.loc[target_indices, [*BANK_GLOBAL_FEATURES, *BANK_LOCAL_FEATURES]] = values
bank_shuffled, shuffled_reports = _crossfit(shuffled, ALIGNED_MODEL_FEATURES, "bank_shuffled")

features["v78_replay_weight"] = base_replay
features["v79_bank_weight"] = aligned
features["bank_shuffled_weight"] = bank_shuffled
features.to_parquet(OUT_DIR / "bank_adaptive_weights_oof.parquet", index=False, compression="zstd")

predictions = {
    "v76_well_adaptive": _expand_predictions(frame, features, "baseline_weight"),
    "v78_existing": _expand_predictions(frame, features, "v78_adaptive_band_weight"),
    "v78_feature_replay": _expand_predictions(frame, features, "v78_replay_weight"),
    "v79_bank_adaptive": _expand_predictions(frame, features, "v79_bank_weight"),
    "bank_shuffled_control": _expand_predictions(frame, features, "bank_shuffled_weight"),
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
gain_vs_v76 = overall["v76_well_adaptive"]["rmse"] - overall["v79_bank_adaptive"]["rmse"]
gain_vs_v78 = overall["v78_existing"]["rmse"] - overall["v79_bank_adaptive"]["rmse"]
far_gain = overall["v76_well_adaptive"]["far_gt2000_rmse"] - overall["v79_bank_adaptive"]["far_gt2000_rmse"]
near_degradation = overall["v79_bank_adaptive"]["near_0_500_rmse"] - overall["v76_well_adaptive"]["near_0_500_rmse"]
shuffle_loss = overall["bank_shuffled_control"]["rmse"] - overall["v79_bank_adaptive"]["rmse"]
fold_gains = {str(fold): metrics[str(fold)]["v76_well_adaptive"]["rmse"] - metrics[str(fold)]["v79_bank_adaptive"]["rmse"] for fold in OUTER_FOLDS}
v78_replay_gap = float(np.max(np.abs(predictions["v78_existing"] - predictions["v78_feature_replay"])))
unavailable = sum(report["status"] != "built" for report in bank_reports)
checks = {
    "candidate_bank_coverage_ge_0p99": unavailable <= 4,
    "v78_replay_row_gap_le_1e_6": v78_replay_gap <= 1e-6,
    "gain_vs_v76_ge_0p05": gain_vs_v76 >= 0.05,
    "gain_vs_v78_ge_0p03": gain_vs_v78 >= 0.03,
    "positive_folds_3of3": all(value > 0.0 for value in fold_gains.values()),
    "far_gain_ge_0p10": far_gain >= 0.10,
    "near_degradation_le_0p02": near_degradation <= 0.02,
    "bank_shuffle_loses_ge_0p03": shuffle_loss >= 0.03,
}
strict_pass = all(checks.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "exact_V70_topk_candidate_bank_uncertainty_nested_band_dose_gateway",
    "sample_unit": "one complete physical well",
    "candidate_bank": {
        "source": str(V70_PATH), "top_k": TOP_K, "wells": int(len(well_rows)),
        "unavailable_wells": unavailable, "persisted_trajectories": False,
        "label_free_features": [*BANK_GLOBAL_FEATURES, *BANK_LOCAL_FEATURES],
    },
    "protocol": {
        "honest_folds": list(OUTER_FOLDS),
        "fold45_excluded_because_used_for_structural_prior": True,
        "nested_shrinkages": list(SHRINKAGES),
        "truth_usage": "outer-training smooth band oracle labels and heldout metrics only",
    },
    "fold_reports": {"base_replay": base_reports, "aligned": aligned_reports, "bank_shuffled": shuffled_reports},
    "metrics": metrics,
    "gain_vs_v76": gain_vs_v76, "gain_vs_v78": gain_vs_v78,
    "far_gain_vs_v76": far_gain, "near_degradation_vs_v76": near_degradation,
    "bank_shuffle_loss": shuffle_loss, "fold_gains": fold_gains,
    "v78_replay_maximum_row_gap": v78_replay_gap,
    "checks": checks, "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze bank-aware dose model and build all-773 deployment adapter"
        if strict_pass else
        "FAIL -> candidate-bank uncertainty does not predict consumable dose; retain V76"
    ),
}

print("\nV79 MATCHED SUMMARY")
print(summary.to_string(index=False))
print("\nV79 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
_write_json(OUT_DIR / "verdict.json", verdict)
print("saved:", OUT_DIR)
