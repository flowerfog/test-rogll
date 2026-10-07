"""Nested dSSE-weighted selector for the frozen shape-basket handoff.

The candidate factory has already established a large whole-well oracle.  This
cell tests whether deployment-visible evidence can safely recover that ceiling.

Key safeguards:
* Candidate paths and labels are whole-well objects; no row-wise selection.
* Pairwise training weights are the exact absolute whole-well delta-SSE.
* Every outer fold uses inner OOF predictions to choose selector type, route
  confidence, and movement dose.
* The hidden-safe fallback is frozen Neighbor, not a learned shape.
* The catastrophically dominated carry-S path is pruned before learning.  The
  resulting B7 and optional CP12-augmented B8 are evaluated independently.
"""

from pathlib import Path
from itertools import combinations
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
DATA_ROOT_CANDIDATES = [
    ROOT / "rogii-wellbore-geology-prediction",
    ROOT / "wellbore-geology-prediction",
]
DATA_ROOT = next((path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()), None)
SOURCE_DIR = OOF_ROOT / "shape_basket_ceiling_audit_v0"
SOURCE_PATH = SOURCE_DIR / "shape_basket_oof.npz"
NEIGHBOR_META_PATH = OOF_ROOT / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / "dsse_shape_selector_nested_v0"

RANDOM_STATE = 20260717
MAX_SAMPLE_ROWS = 640
SMOOTH_SCALES_FT = (64, 128, 256, 512, 1024)
DERIVATIVE_SCALES_FT = (64, 256, 512)
SEGMENTS = 4

PAIR_ESTIMATORS = 650
LIST_ESTIMATORS = 650
DOSES = (0.25, 0.50, 0.75, 1.00)
PAIR_THRESHOLDS = (0.50, 0.60, 0.70, 0.80)
LIST_THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 4.0)
INNER_MIN_GAIN = 0.02

ARMS = {
    "pruned_b7": [
        "geom_q15", "geom_q50", "geom_q85", "full_q50", "full_l1",
        "neighbor_frozen", "strong",
    ],
    "pruned_b8_cp12": [
        "geom_q15", "geom_q50", "geom_q85", "full_q50", "full_l1",
        "neighbor_frozen", "strong", "cp12",
    ],
}
SAFE_CANDIDATE = "neighbor_frozen"


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(np.sum(sse) / np.sum(rows)))


def _finite_stats(values, prefix):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.10)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
    }


def _fit_affine(horizontal_gr, typewell_gr):
    x = np.asarray(horizontal_gr, np.float64)
    y = np.asarray(typewell_gr, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        return 1.0, 0.0
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, float(np.median(y) - np.median(x))
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - centre)) + 1e-6)
        next_keep = np.abs(residual - centre) <= 3.5 * scale
        if next_keep.sum() < 24:
            break
        keep = next_keep
    return float(np.clip(slope, -5.0, 5.0)), float(intercept)


def _metrics(reference, observed, valid, prefix):
    reference = np.asarray(reference, np.float64)
    observed = np.asarray(observed, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(reference) & np.isfinite(observed)
    count = int(valid.sum())
    if count < 4:
        return {
            f"{prefix}_coverage": float(valid.mean()),
            f"{prefix}_bias": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_ncc": np.nan,
        }
    x = reference[valid]
    y = observed[valid]
    difference = x - y
    denominator = np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2))
    ncc = np.sum((x - x.mean()) * (y - y.mean())) / max(denominator, 1e-12)
    return {
        f"{prefix}_coverage": float(valid.mean()),
        f"{prefix}_bias": float(np.mean(difference)),
        f"{prefix}_mae": float(np.mean(np.abs(difference))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(difference**2))),
        f"{prefix}_p90": float(np.quantile(np.abs(difference), 0.90)),
        f"{prefix}_ncc": float(ncc),
    }


def _smooth(values, valid, size):
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    weight = np.asarray(valid, np.float64)
    count = uniform_filter1d(weight, size=size, mode="constant", cval=0.0) * size
    total = uniform_filter1d(
        np.where(valid, values, 0.0), size=size, mode="constant", cval=0.0
    ) * size
    output = total / np.maximum(count, 1.0)
    output_valid = count >= max(4, int(np.ceil(0.5 * size)))
    return output, output_valid


def _path_features(candidate_tvt, strong, neighbor, cp12, z, progress, observed, observed_valid,
                   tw_tvt, tw_gr, md_step, anchor_s):
    candidate_tvt = np.asarray(candidate_tvt, np.float64)
    valid = (
        observed_valid
        & np.isfinite(candidate_tvt)
        & (candidate_tvt >= tw_tvt[0])
        & (candidate_tvt <= tw_tvt[-1])
    )
    reference = np.interp(candidate_tvt, tw_tvt, tw_gr)
    features = {}
    features.update(_metrics(reference, observed, valid, "raw"))

    smoothed = {}
    for scale_ft in SMOOTH_SCALES_FT:
        size = max(3, int(round(scale_ft / max(md_step, 1e-3))))
        reference_smooth, reference_valid = _smooth(reference, valid, size)
        observed_smooth, observed_smooth_valid = _smooth(observed, valid, size)
        smooth_valid = reference_valid & observed_smooth_valid
        features.update(
            _metrics(reference_smooth, observed_smooth, smooth_valid, f"s{scale_ft}")
        )
        smoothed[scale_ft] = (reference_smooth, observed_smooth, smooth_valid)

    reference_128, observed_128, valid_128 = smoothed[128]
    segment_ncc = []
    segment_mae = []
    for segment in range(SEGMENTS):
        left = segment / SEGMENTS
        right = (segment + 1) / SEGMENTS
        mask = (progress >= left) & (
            progress <= right if segment == SEGMENTS - 1 else progress < right
        )
        result = _metrics(
            reference_128[mask], observed_128[mask], valid_128[mask], "segment"
        )
        segment_ncc.append(result["segment_ncc"])
        segment_mae.append(result["segment_mae"])
    features.update(_finite_stats(segment_ncc, "segment_ncc"))
    features.update(_finite_stats(segment_mae, "segment_mae"))

    for scale_ft in DERIVATIVE_SCALES_FT:
        ref, obs, smooth_valid = smoothed[scale_ft]
        derivative_valid = smooth_valid[1:] & smooth_valid[:-1]
        features.update(
            _metrics(np.diff(ref), np.diff(obs), derivative_valid, f"d{scale_ft}")
        )

    correction = candidate_tvt - strong
    candidate_s_delta = candidate_tvt + z - anchor_s
    features.update(_finite_stats(correction, "correction"))
    features["correction_end"] = float(correction[-1])
    features["correction_max_abs"] = float(np.max(np.abs(correction)))
    features["correction_tv"] = float(np.mean(np.abs(np.diff(correction))))
    features["correction_curvature"] = float(
        np.mean(np.abs(np.diff(correction, n=2))) if len(correction) >= 3 else 0.0
    )
    features.update(_finite_stats(candidate_s_delta, "s_delta"))
    features["s_delta_end"] = float(candidate_s_delta[-1])
    features["s_delta_tv"] = float(np.mean(np.abs(np.diff(candidate_s_delta))))
    features["s_delta_curvature"] = float(
        np.mean(np.abs(np.diff(candidate_s_delta, n=2))) if len(candidate_s_delta) >= 3 else 0.0
    )
    design = np.column_stack([progress, np.ones(len(progress))])
    features["s_delta_linear_slope"] = float(
        np.linalg.lstsq(design, candidate_s_delta, rcond=None)[0][0]
    )
    for label, reference_path in (
        ("strong", strong), ("neighbor", neighbor), ("cp12", cp12)
    ):
        difference = candidate_tvt - reference_path
        features[f"distance_{label}_mae"] = float(np.mean(np.abs(difference)))
        features[f"distance_{label}_rmse"] = float(np.sqrt(np.mean(difference**2)))
        features[f"distance_{label}_end"] = float(difference[-1])
    features["typewell_lower_margin_p10"] = float(
        np.quantile(candidate_tvt - tw_tvt[0], 0.10)
    )
    features["typewell_upper_margin_p10"] = float(
        np.quantile(tw_tvt[-1] - candidate_tvt, 0.10)
    )
    return features


def _pair_model(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        objective="binary",
        n_estimators=PAIR_ESTIMATORS,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=1.0,
        reg_lambda=25.0,
        max_bin=127,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )


def _list_model(seed):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=LIST_ESTIMATORS,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=1.0,
        reg_lambda=25.0,
        max_bin=127,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )


def _fit_predict(train_indices, valid_indices, candidate_features, context_features,
                 candidate_sse, well_rows, seed):
    """Return pairwise scores/probabilities and listwise expected SSE gains."""
    candidate_count = candidate_features.shape[1]
    pair_x = []
    pair_y = []
    pair_weight = []
    for well_index in train_indices:
        context = context_features[well_index]
        for left, right in combinations(range(candidate_count), 2):
            delta_sse = candidate_sse[well_index, right] - candidate_sse[well_index, left]
            if abs(delta_sse) <= 1e-8:
                continue
            difference = candidate_features[well_index, left] - candidate_features[well_index, right]
            pair_x.append(np.r_[difference, context])
            pair_y.append(int(delta_sse > 0))
            pair_weight.append(abs(delta_sse))
            pair_x.append(np.r_[-difference, context])
            pair_y.append(int(delta_sse < 0))
            pair_weight.append(abs(delta_sse))
    pair_x = np.asarray(pair_x, np.float32)
    pair_y = np.asarray(pair_y, np.int8)
    pair_weight = np.asarray(pair_weight, np.float64)
    pair_weight /= max(pair_weight.mean(), 1e-12)

    pair_model = _pair_model(seed)
    pair_model.fit(pair_x, pair_y, sample_weight=pair_weight)

    pair_probability = np.full(
        (len(valid_indices), candidate_count, candidate_count), 0.5, np.float32
    )
    for local_well, well_index in enumerate(valid_indices):
        context = context_features[well_index]
        for left, right in combinations(range(candidate_count), 2):
            difference = candidate_features[well_index, left] - candidate_features[well_index, right]
            probability = float(
                pair_model.predict_proba(np.asarray([np.r_[difference, context]], np.float32))[0, 1]
            )
            pair_probability[local_well, left, right] = probability
            pair_probability[local_well, right, left] = 1.0 - probability
    pair_score = pair_probability.mean(axis=2)

    train_candidate = candidate_features[train_indices]
    train_context = np.repeat(
        context_features[train_indices, None, :], candidate_count, axis=1
    )
    list_x = np.concatenate([train_candidate, train_context], axis=2).reshape(
        -1, train_candidate.shape[2] + train_context.shape[2]
    )
    safe_local = CURRENT_SAFE_LOCAL
    safe_sse = candidate_sse[train_indices, safe_local]
    gain_mse = (
        safe_sse[:, None] - candidate_sse[train_indices]
    ) / well_rows[train_indices, None]
    list_weight = np.repeat(well_rows[train_indices], candidate_count).astype(np.float64)
    list_weight /= list_weight.mean()
    list_model = _list_model(seed + 1)
    list_model.fit(list_x, gain_mse.ravel(), sample_weight=list_weight)

    valid_candidate = candidate_features[valid_indices]
    valid_context = np.repeat(
        context_features[valid_indices, None, :], candidate_count, axis=1
    )
    valid_x = np.concatenate([valid_candidate, valid_context], axis=2).reshape(
        -1, valid_candidate.shape[2] + valid_context.shape[2]
    )
    list_gain = list_model.predict(valid_x).reshape(len(valid_indices), candidate_count)
    return pair_score, pair_probability, list_gain


def _selector_outputs(pair_score, pair_probability, list_gain, safe_local):
    pair_selected = np.argmax(pair_score, axis=1)
    list_selected = np.argmax(list_gain, axis=1)
    pair_z = (pair_score - pair_score.mean(axis=1, keepdims=True)) / np.maximum(
        pair_score.std(axis=1, keepdims=True), 1e-6
    )
    list_z = (list_gain - list_gain.mean(axis=1, keepdims=True)) / np.maximum(
        list_gain.std(axis=1, keepdims=True), 1e-6
    )
    ensemble_selected = np.argmax(pair_z + list_z, axis=1)

    output = {}
    for name, selected in (
        ("pair", pair_selected),
        ("list", list_selected),
        ("ensemble", ensemble_selected),
    ):
        rows = np.arange(len(selected))
        pair_confidence = pair_probability[rows, selected, safe_local]
        pair_confidence[selected == safe_local] = 1.0
        list_advantage = list_gain[rows, selected] - list_gain[:, safe_local]
        output[name] = {
            "selected": selected,
            "pair_confidence": pair_confidence,
            "list_advantage": list_advantage,
        }
    return output


def _configs():
    rows = [{"selector": "safe", "threshold": np.nan, "dose": 0.0}]
    for selector in ("pair", "ensemble"):
        for threshold in PAIR_THRESHOLDS:
            for dose in DOSES:
                rows.append({"selector": selector, "threshold": threshold, "dose": dose})
    for threshold in LIST_THRESHOLDS:
        for dose in DOSES:
            rows.append({"selector": "list", "threshold": threshold, "dose": dose})
    return rows


def _apply_config(outputs, config, indices, arm_global_indices, blend_sse, safe_sse):
    if config["selector"] == "safe":
        return safe_sse[indices].copy(), np.full(len(indices), CURRENT_SAFE_LOCAL), np.zeros(len(indices), bool)
    result = outputs[config["selector"]]
    selected = result["selected"]
    if config["selector"] == "list":
        routed = result["list_advantage"] >= config["threshold"]
    else:
        routed = result["pair_confidence"] >= config["threshold"]
    routed &= selected != CURRENT_SAFE_LOCAL
    global_selected = arm_global_indices[selected]
    dose_index = DOSES.index(config["dose"])
    sse = safe_sse[indices].copy()
    sse[routed] = blend_sse[indices[routed], global_selected[routed], dose_index]
    return sse, selected, routed


def _select_inner_config(inner_indices, inner_outputs, arm_global_indices, blend_sse,
                         safe_sse, well_rows, well_folds):
    rows = []
    for config in _configs():
        sse, _, routed = _apply_config(
            inner_outputs, config, inner_indices, arm_global_indices, blend_sse, safe_sse
        )
        safe_rmse = _rmse_from_sse(safe_sse[inner_indices], well_rows[inner_indices])
        rmse = _rmse_from_sse(sse, well_rows[inner_indices])
        fold_gains = []
        for fold in sorted(np.unique(well_folds[inner_indices])):
            mask = well_folds[inner_indices] == fold
            fold_gains.append(
                _rmse_from_sse(safe_sse[inner_indices][mask], well_rows[inner_indices][mask])
                - _rmse_from_sse(sse[mask], well_rows[inner_indices][mask])
            )
        rows.append(
            {
                **config,
                "safe_rmse": safe_rmse,
                "rmse": rmse,
                "gain": safe_rmse - rmse,
                "mean_fold_gain": float(np.mean(fold_gains)),
                "min_fold_gain": float(np.min(fold_gains)),
                "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
                "routed_fraction": float(np.mean(routed)),
            }
        )
    report = pd.DataFrame(rows)
    non_safe = report[
        (report["selector"] != "safe")
        & (report["gain"] >= INNER_MIN_GAIN)
        & (report["min_fold_gain"] >= 0.0)
        & (report["positive_folds"] == len(np.unique(well_folds[inner_indices])))
    ].copy()
    if len(non_safe):
        selected_row = non_safe.sort_values(
            ["rmse", "dose", "threshold"], ascending=[True, True, False]
        ).iloc[0]
    else:
        selected_row = report.loc[report["selector"] == "safe"].iloc[0]
    selected = {
        "selector": str(selected_row["selector"]),
        "threshold": float(selected_row["threshold"]),
        "dose": float(selected_row["dose"]),
    }
    return selected, report


for path in (SOURCE_PATH, NEIGHBOR_META_PATH):
    if not path.exists():
        raise FileNotFoundError(path)
if DATA_ROOT is None:
    raise FileNotFoundError("competition data root not found")
if POOL_PATH is None:
    raise FileNotFoundError("candidate pool not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)

started = time.time()
print(
    {
        "experiment": "dsse_shape_selector_nested_v0",
        "source": str(SOURCE_PATH),
        "data_root": str(DATA_ROOT),
        "pool": str(POOL_PATH),
        "arms": ARMS,
        "safe": SAFE_CANDIDATE,
    },
    flush=True,
)


# -----------------------------------------------------------------------------
# Load candidate handoff and aligned truth.
# -----------------------------------------------------------------------------
with np.load(SOURCE_PATH, allow_pickle=False) as handoff:
    predictions = handoff["predictions"].astype(np.float32)
    candidate_names = handoff["candidate_names"].astype(str).tolist()
    row_folds = handoff["row_folds"].astype(np.int8)
    well_names = handoff["well_names"].astype(str).tolist()
    well_starts = handoff["well_starts"].astype(np.int64)
    well_ends = handoff["well_ends"].astype(np.int64)
candidate_lookup = {name: index for index, name in enumerate(candidate_names)}

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [column for column in ("id", "well", "row_idx", "tvt_true") if column in header]
pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": "string", "well": "string"})
if "id" in pool:
    pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
truth = pool["tvt_true"].to_numpy(np.float64)
pool_row_idx = pool["row_idx"].to_numpy(np.int32)
if len(pool) != len(predictions):
    raise RuntimeError("candidate handoff/pool row mismatch")

neighbor_meta = pd.read_csv(NEIGHBOR_META_PATH, dtype={"well": str}).set_index("well")


# -----------------------------------------------------------------------------
# Build deployment-visible whole-path GR and shape evidence for only 8/9 paths.
# -----------------------------------------------------------------------------
feature_rows = []
context_rows = []
well_rows = well_ends - well_starts
well_folds = np.zeros(len(well_names), np.int8)
well_candidate_sse = np.zeros((len(well_names), len(candidate_names)), np.float64)

for well_index, well in enumerate(tqdm(well_names, desc="shape selector evidence")):
    start, end = int(well_starts[well_index]), int(well_ends[well_index])
    rows = pool_row_idx[start:end]
    well_folds[well_index] = int(row_folds[start])
    well_truth = truth[start:end]
    well_predictions = predictions[start:end].astype(np.float64)
    well_candidate_sse[well_index] = np.sum((well_predictions - well_truth[:, None]) ** 2, axis=0)

    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=lambda column: column in {"MD", "X", "Y", "Z", "GR", "TVT_input"},
    )
    typewell = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"]
    )
    for column in horizontal.columns:
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    typewell = typewell.apply(pd.to_numeric, errors="coerce").dropna(subset=["TVT"])
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    typewell["GR"] = typewell["GR"].interpolate(limit_direction="both").fillna(0.0)

    md = horizontal["MD"].to_numpy(np.float64)
    z_all = horizontal["Z"].to_numpy(np.float64)
    gr_all = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    tw_tvt = typewell["TVT"].to_numpy(np.float64)
    tw_gr_raw = typewell["GR"].to_numpy(np.float64)
    anchor = int(rows.min() - 1)
    if anchor < 0 or not np.isfinite(tvt_input[anchor]):
        raise RuntimeError(f"invalid anchor for {well}")

    if len(rows) <= MAX_SAMPLE_ROWS:
        local_sample = np.arange(len(rows), dtype=int)
    else:
        local_sample = np.unique(
            np.round(np.linspace(0, len(rows) - 1, MAX_SAMPLE_ROWS)).astype(int)
        )
    sample_rows = rows[local_sample]
    sample_md = md[sample_rows]
    z = z_all[sample_rows]
    progress = (sample_md - md[anchor]) / max(sample_md[-1] - md[anchor], 1.0)
    md_step = float(np.median(np.diff(sample_md))) if len(sample_md) > 1 else 1.0

    known = np.isfinite(tvt_input) & np.isfinite(gr_all)
    known_valid = known & (tvt_input >= tw_tvt[0]) & (tvt_input <= tw_tvt[-1])
    known_tw = np.interp(tvt_input[known_valid], tw_tvt, tw_gr_raw)
    slope, intercept = _fit_affine(gr_all[known_valid], known_tw)
    calibrated = slope * gr_all + intercept
    location_values = known_tw if len(known_tw) >= 16 else tw_gr_raw[np.isfinite(tw_gr_raw)]
    location = float(np.median(location_values))
    scale = float(1.4826 * np.median(np.abs(location_values - location)) + 1e-3)
    observed = np.clip((calibrated[sample_rows] - location) / scale, -8.0, 8.0)
    observed_valid = np.isfinite(observed)
    tw_gr = np.clip((tw_gr_raw - location) / scale, -8.0, 8.0)
    anchor_s = float(tvt_input[anchor] + z_all[anchor])

    sampled_predictions = well_predictions[local_sample]
    strong = sampled_predictions[:, candidate_lookup["strong"]]
    neighbor = sampled_predictions[:, candidate_lookup["neighbor_frozen"]]
    cp12 = sampled_predictions[:, candidate_lookup["cp12"]]
    disagreement = np.std(sampled_predictions, axis=1)

    context = {
        "well": well,
        "fold": int(well_folds[well_index]),
        "ctx_rows": len(rows),
        "ctx_known_frac": float(np.mean(np.isfinite(tvt_input))),
        "ctx_eval_span_kft": float((md[rows[-1]] - md[anchor]) / 1000.0),
        "ctx_gr_missing_all": float(np.mean(~np.isfinite(gr_all))),
        "ctx_gr_missing_eval": float(np.mean(~np.isfinite(gr_all[rows]))),
        "ctx_anchor_tvt_10k": float(tvt_input[anchor] / 10000.0),
        "ctx_anchor_s_10k": float(anchor_s / 10000.0),
        "ctx_z_change_kft": float((z_all[rows[-1]] - z_all[anchor]) / 1000.0),
        "ctx_calibration_slope": slope,
        "ctx_calibration_intercept_scaled": intercept / scale,
        "ctx_typewell_range_kft": float((tw_tvt[-1] - tw_tvt[0]) / 1000.0),
        "ctx_disagreement_mean": float(np.mean(disagreement)),
        "ctx_disagreement_p95": float(np.quantile(disagreement, 0.95)),
        "ctx_disagreement_max": float(np.max(disagreement)),
    }
    context.update({f"ctx_{key}": value for key, value in _finite_stats(gr_all[rows], "eval_gr").items()})
    context.update({f"ctx_{key}": value for key, value in _finite_stats(gr_all[: anchor + 1], "prefix_gr").items()})
    if well in neighbor_meta.index:
        for column in ("median_distance", "p90_distance", "anchor_distance_full", "direction_dot"):
            context[f"ctx_neighbor_{column}"] = float(neighbor_meta.loc[well, column])
    context_rows.append(context)

    for candidate_i, candidate_name in enumerate(candidate_names):
        features = _path_features(
            sampled_predictions[:, candidate_i], strong, neighbor, cp12, z, progress,
            observed, observed_valid, tw_tvt, tw_gr, md_step, anchor_s,
        )
        features.update(
            {
                "well": well,
                "fold": int(well_folds[well_index]),
                "candidate": candidate_name,
                "candidate_index": candidate_i,
                "rows": len(rows),
                "candidate_sse": well_candidate_sse[well_index, candidate_i],
                "candidate_mse": well_candidate_sse[well_index, candidate_i] / len(rows),
            }
        )
        feature_rows.append(features)

candidate_frame = pd.DataFrame(feature_rows)
context_frame = pd.DataFrame(context_rows).set_index("well").loc[well_names].reset_index()

for name in candidate_names:
    candidate_frame[f"identity_{name}"] = (candidate_frame["candidate"] == name).astype(np.int8)
rank_low = [
    "raw_rmse", "raw_mae", "s128_rmse", "s256_rmse", "s512_rmse",
    "segment_mae_mean", "distance_neighbor_rmse", "distance_cp12_rmse",
]
rank_high = [
    "raw_ncc", "s128_ncc", "s256_ncc", "s512_ncc", "s1024_ncc",
    "segment_ncc_mean", "segment_ncc_p10",
]
for column in rank_low:
    if column in candidate_frame:
        candidate_frame[f"rank_{column}"] = candidate_frame.groupby("well")[column].rank(
            pct=True, ascending=True
        )
for column in rank_high:
    if column in candidate_frame:
        candidate_frame[f"rank_{column}"] = candidate_frame.groupby("well")[column].rank(
            pct=True, ascending=False
        )

candidate_frame.to_pickle(OUT_DIR / "candidate_evidence.pkl")
context_frame.to_csv(OUT_DIR / "well_context.csv", index=False)

metadata_columns = {
    "well", "fold", "candidate", "candidate_index", "rows", "candidate_sse", "candidate_mse"
}
candidate_feature_columns = [
    column for column in candidate_frame.columns
    if column not in metadata_columns and pd.api.types.is_numeric_dtype(candidate_frame[column])
]
context_feature_columns = [
    column for column in context_frame.columns
    if column.startswith("ctx_") and pd.api.types.is_numeric_dtype(context_frame[column])
]

candidate_frame = candidate_frame.sort_values(["well", "candidate_index"]).reset_index(drop=True)
expected_rows = len(well_names) * len(candidate_names)
if len(candidate_frame) != expected_rows:
    raise RuntimeError(f"candidate evidence row mismatch: {len(candidate_frame)} != {expected_rows}")
candidate_feature_cube = candidate_frame[candidate_feature_columns].to_numpy(np.float32).reshape(
    len(well_names), len(candidate_names), -1
)
context_features = context_frame[context_feature_columns].to_numpy(np.float32)

print(
    "evidence:",
    {
        "wells": len(well_names),
        "candidates": len(candidate_names),
        "candidate_features": len(candidate_feature_columns),
        "context_features": len(context_feature_columns),
    },
    flush=True,
)


# -----------------------------------------------------------------------------
# Precompute exact SSE for each candidate dose around frozen Neighbor.
# -----------------------------------------------------------------------------
safe_global = candidate_lookup[SAFE_CANDIDATE]
safe_sse = well_candidate_sse[:, safe_global]
blend_sse = np.zeros(
    (len(well_names), len(candidate_names), len(DOSES)), np.float64
)
for well_index, (start, end) in enumerate(zip(well_starts, well_ends)):
    start, end = int(start), int(end)
    y = truth[start:end]
    safe_path = predictions[start:end, safe_global].astype(np.float64)
    for candidate_i in range(len(candidate_names)):
        candidate_path = predictions[start:end, candidate_i].astype(np.float64)
        for dose_i, dose in enumerate(DOSES):
            path = safe_path + dose * (candidate_path - safe_path)
            blend_sse[well_index, candidate_i, dose_i] = np.sum((path - y) ** 2)


# -----------------------------------------------------------------------------
# Strict nested five-fold evaluation for B8 and B9.
# -----------------------------------------------------------------------------
all_indices = np.arange(len(well_names))
outer_rows = []
inner_reports = []
selection_records = []
arm_oof_sse = {}
arm_oof_config = {}

for arm_number, (arm_name, arm_names) in enumerate(ARMS.items()):
    print("\nARM", arm_name, arm_names, flush=True)
    arm_global = np.asarray([candidate_lookup[name] for name in arm_names], int)
    arm_features = candidate_feature_cube[:, arm_global]
    arm_sse = well_candidate_sse[:, arm_global]
    CURRENT_SAFE_LOCAL = arm_names.index(SAFE_CANDIDATE)
    oof_sse = np.full(len(well_names), np.nan, np.float64)
    oof_configs = {}

    for outer_fold in sorted(np.unique(well_folds)):
        outer_train = all_indices[well_folds != outer_fold]
        outer_valid = all_indices[well_folds == outer_fold]
        print(
            f"{arm_name} outer={outer_fold} train={len(outer_train)} valid={len(outer_valid)}",
            flush=True,
        )

        inner_pair_score = np.full((len(outer_train), len(arm_names)), np.nan, np.float32)
        inner_pair_probability = np.full(
            (len(outer_train), len(arm_names), len(arm_names)), np.nan, np.float32
        )
        inner_list_gain = np.full((len(outer_train), len(arm_names)), np.nan, np.float32)
        train_position = {well_index: position for position, well_index in enumerate(outer_train)}

        for inner_fold in sorted(np.unique(well_folds[outer_train])):
            inner_valid = outer_train[well_folds[outer_train] == inner_fold]
            inner_train = outer_train[well_folds[outer_train] != inner_fold]
            pair_score, pair_probability, list_gain = _fit_predict(
                inner_train,
                inner_valid,
                arm_features,
                context_features,
                arm_sse,
                well_rows,
                RANDOM_STATE + 10000 * arm_number + 100 * int(outer_fold) + int(inner_fold),
            )
            positions = np.asarray([train_position[index] for index in inner_valid], int)
            inner_pair_score[positions] = pair_score
            inner_pair_probability[positions] = pair_probability
            inner_list_gain[positions] = list_gain

        if not (
            np.isfinite(inner_pair_score).all()
            and np.isfinite(inner_pair_probability).all()
            and np.isfinite(inner_list_gain).all()
        ):
            raise RuntimeError(f"incomplete inner OOF selector predictions for {arm_name} fold {outer_fold}")
        inner_outputs = _selector_outputs(
            inner_pair_score, inner_pair_probability, inner_list_gain, CURRENT_SAFE_LOCAL
        )
        selected_config, inner_report = _select_inner_config(
            outer_train,
            inner_outputs,
            arm_global,
            blend_sse,
            safe_sse,
            well_rows,
            well_folds,
        )
        inner_report.insert(0, "arm", arm_name)
        inner_report.insert(1, "outer_fold", int(outer_fold))
        inner_reports.append(inner_report)

        pair_score, pair_probability, list_gain = _fit_predict(
            outer_train,
            outer_valid,
            arm_features,
            context_features,
            arm_sse,
            well_rows,
            RANDOM_STATE + 10000 * arm_number + 100 * int(outer_fold) + 99,
        )
        outer_outputs = _selector_outputs(
            pair_score, pair_probability, list_gain, CURRENT_SAFE_LOCAL
        )
        selected_sse, selected_local, routed = _apply_config(
            outer_outputs,
            selected_config,
            outer_valid,
            arm_global,
            blend_sse,
            safe_sse,
        )
        oof_sse[outer_valid] = selected_sse
        oof_configs[int(outer_fold)] = selected_config

        safe_rmse = _rmse_from_sse(safe_sse[outer_valid], well_rows[outer_valid])
        selector_rmse = _rmse_from_sse(selected_sse, well_rows[outer_valid])
        outer_rows.append(
            {
                "arm": arm_name,
                "fold": int(outer_fold),
                **selected_config,
                "wells": len(outer_valid),
                "rows": int(well_rows[outer_valid].sum()),
                "safe_rmse": safe_rmse,
                "selector_rmse": selector_rmse,
                "gain_vs_safe": safe_rmse - selector_rmse,
                "routed_wells": int(routed.sum()),
                "routed_fraction": float(routed.mean()),
            }
        )
        for local_position, well_index in enumerate(outer_valid):
            local_candidate = int(selected_local[local_position])
            selection_records.append(
                {
                    "arm": arm_name,
                    "well": well_names[well_index],
                    "fold": int(outer_fold),
                    "selector": selected_config["selector"],
                    "threshold": selected_config["threshold"],
                    "dose": selected_config["dose"] if routed[local_position] else 0.0,
                    "routed": bool(routed[local_position]),
                    "selected_candidate": arm_names[local_candidate],
                }
            )
        print("outer result", outer_rows[-1], flush=True)

    if not np.isfinite(oof_sse).all():
        raise RuntimeError(f"incomplete outer OOF SSE for {arm_name}")
    arm_oof_sse[arm_name] = oof_sse
    arm_oof_config[arm_name] = oof_configs

outer_report = pd.DataFrame(outer_rows)
inner_report = pd.concat(inner_reports, ignore_index=True)
selection_report = pd.DataFrame(selection_records)
outer_report.to_csv(OUT_DIR / "outer_folds.csv", index=False)
inner_report.to_csv(OUT_DIR / "inner_config_scan.csv", index=False)
selection_report.to_csv(OUT_DIR / "well_selections.csv", index=False)

top40_indices = np.argsort(safe_sse)[-40:]
summary_rows = []
for arm_name, oof_sse in arm_oof_sse.items():
    fold_gain = outer_report.loc[outer_report["arm"] == arm_name, "gain_vs_safe"]
    arm_indices = np.asarray([candidate_lookup[name] for name in ARMS[arm_name]], int)
    arm_oracle_sse = np.min(well_candidate_sse[:, arm_indices], axis=1)
    summary_rows.append(
        {
            "arm": arm_name,
            "safe_rmse": _rmse_from_sse(safe_sse, well_rows),
            "selector_rmse": _rmse_from_sse(oof_sse, well_rows),
            "gain_vs_safe": _rmse_from_sse(safe_sse, well_rows) - _rmse_from_sse(oof_sse, well_rows),
            "whole_well_oracle_rmse": _rmse_from_sse(arm_oracle_sse, well_rows),
            "oracle_gain_vs_safe": (
                _rmse_from_sse(safe_sse, well_rows)
                - _rmse_from_sse(arm_oracle_sse, well_rows)
            ),
            "mean_fold_gain": float(fold_gain.mean()),
            "min_fold_gain": float(fold_gain.min()),
            "positive_folds": int((fold_gain > 0).sum()),
            "top40_safe_rmse": _rmse_from_sse(safe_sse[top40_indices], well_rows[top40_indices]),
            "top40_selector_rmse": _rmse_from_sse(oof_sse[top40_indices], well_rows[top40_indices]),
            "top40_gain": (
                _rmse_from_sse(safe_sse[top40_indices], well_rows[top40_indices])
                - _rmse_from_sse(oof_sse[top40_indices], well_rows[top40_indices])
            ),
            "routed_fraction": float(
                selection_report.loc[selection_report["arm"] == arm_name, "routed"].mean()
            ),
        }
    )
summary = pd.DataFrame(summary_rows).sort_values("selector_rmse")
summary.to_csv(OUT_DIR / "summary.csv", index=False)

best = summary.iloc[0]
best_arm = str(best["arm"])
best_selections = selection_report[selection_report["arm"] == best_arm].set_index("well").loc[well_names]
selected_prediction = np.empty(len(pool), np.float32)
for well_index, well in enumerate(well_names):
    start, end = int(well_starts[well_index]), int(well_ends[well_index])
    record = best_selections.loc[well]
    safe_path = predictions[start:end, safe_global].astype(np.float64)
    if bool(record["routed"]):
        selected_i = candidate_lookup[str(record["selected_candidate"])]
        candidate_path = predictions[start:end, selected_i].astype(np.float64)
        path = safe_path + float(record["dose"]) * (candidate_path - safe_path)
    else:
        path = safe_path
    selected_prediction[start:end] = path.astype(np.float32)

np.savez_compressed(
    OUT_DIR / "selector_oof.npz",
    prediction=selected_prediction,
    well_names=np.asarray(well_names, str),
    well_starts=well_starts,
    well_ends=well_ends,
    best_arm=np.asarray([best_arm], str),
)

strict_pass = bool(
    best["gain_vs_safe"] >= 0.10
    and best["min_fold_gain"] >= 0.0
    and best["positive_folds"] == 5
    and best["top40_gain"] > 0.0
)
verdict = {
    "experiment": "dsse_shape_selector_nested_v0",
    "source": str(SOURCE_PATH),
    "pool": str(POOL_PATH),
    "wells": len(well_names),
    "candidate_features": len(candidate_feature_columns),
    "context_features": len(context_feature_columns),
    "pair_weight": "exact abs whole-well delta-SSE, mean-normalized without clipping",
    "best_arm": best_arm,
    "safe_rmse": float(best["safe_rmse"]),
    "selector_rmse": float(best["selector_rmse"]),
    "gain_vs_safe": float(best["gain_vs_safe"]),
    "min_fold_gain": float(best["min_fold_gain"]),
    "positive_folds": int(best["positive_folds"]),
    "top40_gain": float(best["top40_gain"]),
    "routed_fraction": float(best["routed_fraction"]),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> freeze selector recipe, rebuild full models, then test pre-Gold insertion"
        if strict_pass
        else "FAIL -> stop learned whole-well selection; retain basket as oracle diagnostic"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nOUTER FOLDS")
print(outer_report.to_string(index=False))
print("\nSUMMARY")
print(summary.to_string(index=False))
print("\nSELECTION COUNTS")
print(
    selection_report.groupby(["arm", "selected_candidate", "routed"]).size()
    .rename("wells").reset_index().to_string(index=False)
)
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
