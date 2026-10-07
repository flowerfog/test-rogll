"""Zero-training audit of GR alignment around the fold-clean strong proposal.

This is a matched follow-up to completewell_alignment_feature_sufficiency_audit_v0.
The v0 audit searched a 640-ft absolute structure corridor and found almost no
usable absolute registration.  Here the same legal GR/typewell evidence ranks
residual states around the honest OOF ``strong`` path.  A prior-only arm makes
the test explicit: GR must improve truth rank beyond merely staying near strong.

Validation truth is used only for final ranking metrics.  No Gold, CP12,
Neighbor, validation-well fitting, or hidden/LB feedback is used.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


EXPERIMENT = "strong_centered_alignment_sufficiency_audit_v1"
ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / EXPERIMENT

AUDIT_FOLD = 1
STATE_HALF_WIDTH_FT = 128.0
STATE_STEP_FT = 1.0
STATE_OFFSETS = np.arange(
    -STATE_HALF_WIDTH_FT,
    STATE_HALF_WIDTH_FT + 0.5 * STATE_STEP_FT,
    STATE_STEP_FT,
    dtype=np.float32,
)
WINDOWS_FT = (32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 1600.0)
MAX_STATIONS_PER_WELL = 256
MIN_WINDOW_COVERAGE = 0.55
PRIOR_SCALE_FT = 12.0
PRIOR_WEIGHTS = (0.25, 0.5, 1.0, 2.0)
MAX_WORKERS = min(3, max(1, os.cpu_count() or 1))

OUT_DIR.mkdir(parents=True, exist_ok=True)
if POOL_PATH is None:
    raise FileNotFoundError("candidate pool not found")
if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
if not (DATA_ROOT / "train").exists():
    raise FileNotFoundError(DATA_ROOT / "train")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _deduplicate_axis(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).dropna()
    frame = frame.groupby("axis", as_index=False)["value"].mean().sort_values("axis")
    return frame["axis"].to_numpy(np.float64), frame["value"].to_numpy(np.float64)


def _robust_location_scale(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    scale = max(1.4826 * mad, 0.25 * float(np.std(values)), 1e-3)
    return centre, scale


def _robust_affine(source, target):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    valid = np.isfinite(source) & np.isfinite(target)
    x = source[valid]
    y = target[valid]
    if len(x) < 32:
        shift = float(np.median(y) - np.median(x)) if len(x) else 0.0
        return 1.0, shift, int(len(x)), np.nan
    keep = np.ones(len(x), bool)
    gain, intercept = 1.0, 0.0
    for _ in range(5):
        design = np.c_[x[keep], np.ones(int(keep.sum()))]
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (gain * x + intercept)
        centre, scale = _robust_location_scale(residual[keep])
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if next_keep.sum() < 32 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    gain = float(np.clip(gain, 0.15, 6.0))
    residual = y[keep] - (gain * x[keep] + intercept)
    rmse = float(np.sqrt(np.mean(residual * residual))) if len(residual) else np.nan
    return gain, float(intercept), int(keep.sum()), rmse


def _fill_1d(axis, values):
    axis = np.asarray(axis, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.isfinite(axis) & np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros(len(values), np.float64)
    order = np.argsort(axis[valid])
    return np.interp(axis, axis[valid][order], values[valid][order])


def _sample_rows(indices, maximum):
    indices = np.asarray(indices, np.int64)
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum)
    return np.unique(indices[np.round(positions).astype(np.int64)])


def _distance_subset(distance):
    distance = np.asarray(distance, np.float64)
    result = np.full(len(distance), "far_gt2000", dtype=object)
    result[distance < 2000.0] = "mid_1000_2000"
    result[distance < 1000.0] = "mid_500_1000"
    result[distance < 500.0] = "near_0_500"
    return result


def _cumulative(matrix):
    matrix = np.asarray(matrix, np.float32)
    return np.concatenate(
        [np.zeros((1, matrix.shape[1]), np.float32), np.cumsum(matrix, axis=0)],
        axis=0,
    )


def _window_scores(observed, expected, valid, sample_index, window_rows):
    observed = np.asarray(observed, np.float32)
    expected = np.asarray(expected, np.float32)
    valid = np.asarray(valid, bool)
    observed_column = observed[:, None]
    valid_float = valid.astype(np.float32)
    expected_zero = np.where(valid, expected, 0.0).astype(np.float32)
    observed_zero = np.where(valid, observed_column, 0.0).astype(np.float32)

    c_valid = _cumulative(valid_float)
    c_observed = _cumulative(observed_zero)
    c_expected = _cumulative(expected_zero)
    c_observed2 = _cumulative(observed_zero * observed_zero)
    c_expected2 = _cumulative(expected_zero * expected_zero)
    c_cross = _cumulative(observed_zero * expected_zero)

    result = {}
    n_rows = len(observed)
    for window_ft, rows in window_rows.items():
        half = max(int(rows) // 2, 1)
        low = np.maximum(sample_index - half, 0)
        high = np.minimum(sample_index + half + 1, n_rows)
        count = c_valid[high] - c_valid[low]
        sum_o = c_observed[high] - c_observed[low]
        sum_e = c_expected[high] - c_expected[low]
        sum_o2 = c_observed2[high] - c_observed2[low]
        sum_e2 = c_expected2[high] - c_expected2[low]
        sum_cross = c_cross[high] - c_cross[low]
        denominator = np.maximum(count, 1.0)
        covariance = sum_cross - sum_o * sum_e / denominator
        variance_o = np.maximum(sum_o2 - sum_o * sum_o / denominator, 0.0)
        variance_e = np.maximum(sum_e2 - sum_e * sum_e / denominator, 0.0)
        ncc = covariance / np.sqrt(variance_o * variance_e + 1e-6)
        rmse = np.sqrt(
            np.maximum((sum_o2 + sum_e2 - 2.0 * sum_cross) / denominator, 0.0)
        )
        nominal = np.maximum((high - low)[:, None], 1)
        coverage = count / nominal
        usable = (coverage >= MIN_WINDOW_COVERAGE) & (count >= 8)
        result[window_ft] = (
            np.where(usable, ncc, np.nan).astype(np.float32),
            np.where(usable, rmse, np.nan).astype(np.float32),
            coverage.astype(np.float32),
        )
    return result


def _row_standardise(score):
    score = np.asarray(score, np.float32)
    centre = np.nanmedian(score, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(score - centre), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-3)
    return np.clip((score - centre) / scale, -8.0, 8.0).astype(np.float32)


def _finite_mean(arrays):
    stack = np.stack(arrays, axis=0).astype(np.float32)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    return np.where(
        count > 0,
        np.where(valid, stack, 0.0).sum(axis=0) / np.maximum(count, 1),
        np.nan,
    ).astype(np.float32)


def _global_ncc(observed, expected, valid):
    observed = np.asarray(observed, np.float64)[:, None]
    expected = np.asarray(expected, np.float64)
    valid = np.asarray(valid, bool)
    count = valid.sum(axis=0).astype(np.float64)
    denominator = np.maximum(count, 1.0)
    observed_zero = np.where(valid, observed, 0.0)
    expected_zero = np.where(valid, expected, 0.0)
    sum_o = observed_zero.sum(axis=0)
    sum_e = expected_zero.sum(axis=0)
    covariance = (observed_zero * expected_zero).sum(axis=0) - sum_o * sum_e / denominator
    variance_o = (observed_zero * observed_zero).sum(axis=0) - sum_o * sum_o / denominator
    variance_e = (expected_zero * expected_zero).sum(axis=0) - sum_e * sum_e / denominator
    score = covariance / np.sqrt(
        np.maximum(variance_o, 0.0) * np.maximum(variance_e, 0.0) + 1e-6
    )
    score[count < max(32, 0.25 * len(observed))] = np.nan
    return score.astype(np.float32)


def _rank_rows(score, truth_residual):
    score = np.asarray(score, np.float64)
    truth_residual = np.asarray(truth_residual, np.float64)
    n_rows = len(truth_residual)
    prediction = np.full(n_rows, np.nan, np.float64)
    rank = np.full(n_rows, np.nan, np.float64)
    top5 = np.zeros(n_rows, bool)
    top8 = np.zeros(n_rows, bool)
    for row in range(n_rows):
        values = score[row]
        valid = np.isfinite(values)
        if valid.sum() < 8:
            continue
        valid_indices = np.flatnonzero(valid)
        order = valid_indices[np.argsort(values[valid_indices])[::-1]]
        prediction[row] = float(STATE_OFFSETS[order[0]])
        nearest = int(np.argmin(np.abs(STATE_OFFSETS - truth_residual[row])))
        if not valid[nearest]:
            continue
        rank[row] = float(1 + np.sum(values[valid] > values[nearest]))
        top5[row] = bool(
            np.any(np.abs(STATE_OFFSETS[order[:5]] - truth_residual[row]) <= 5.0)
        )
        top8[row] = bool(
            np.any(np.abs(STATE_OFFSETS[order[:8]] - truth_residual[row]) <= 5.0)
        )
    return prediction, rank, top5, top8


fold_frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
well_column = next(
    (column for column in ("well", "well_id", "WELL") if column in fold_frame), None
)
fold_column = next(
    (column for column in ("fold", "outer_fold", "Fold") if column in fold_frame), None
)
if well_column is None or fold_column is None:
    raise RuntimeError(f"invalid fold metadata: {fold_frame.columns.tolist()}")
fold_frame[fold_column] = pd.to_numeric(fold_frame[fold_column], errors="coerce")
audit_wells = sorted(
    fold_frame.loc[fold_frame[fold_column].eq(AUDIT_FOLD), well_column]
    .astype(str)
    .unique()
    .tolist()
)
if len(audit_wells) < 100:
    raise RuntimeError(f"unexpected Fold-{AUDIT_FOLD} wells: {len(audit_wells)}")
audit_set = set(audit_wells)

pool_header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
required = {"id", "strong", "tvt_true"}
if not required.issubset(pool_header):
    raise RuntimeError(f"pool misses {sorted(required - set(pool_header))}")
pool_usecols = [
    column
    for column in ("id", "well", "row_idx", "strong", "tvt_true")
    if column in pool_header
]
pool_parts = []
for chunk in pd.read_csv(
    POOL_PATH,
    usecols=pool_usecols,
    dtype={"id": str, "well": str},
    chunksize=500_000,
):
    if "well" not in chunk:
        chunk["well"] = chunk["id"].astype(str).str.rsplit("_", n=1).str[0]
    chunk["well"] = chunk["well"].astype(str)
    selected = chunk[chunk["well"].isin(audit_set)].copy()
    if len(selected):
        pool_parts.append(selected)
if not pool_parts:
    raise RuntimeError("no Fold-1 pool rows loaded")
pool = pd.concat(pool_parts, ignore_index=True)
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].astype(str).str.rsplit("_", n=1).str[-1]
pool["row_idx"] = pd.to_numeric(pool["row_idx"], errors="raise").astype(np.int32)
pool["strong"] = pd.to_numeric(pool["strong"], errors="coerce").astype(np.float32)
pool["tvt_true"] = pd.to_numeric(pool["tvt_true"], errors="coerce").astype(np.float32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool_groups = {well: frame.copy() for well, frame in pool.groupby("well", sort=True)}
if len(pool_groups) != len(audit_wells):
    missing = sorted(audit_set.difference(pool_groups))
    raise RuntimeError(f"pool has {len(pool_groups)} wells; missing {missing[:10]}")


def _audit_well(well):
    frame = pool_groups[well]
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    row_index = frame["row_idx"].to_numpy(np.int64)
    if row_index.max() >= len(horizontal):
        raise RuntimeError("row index exceeds horizontal file")

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.isfinite(visible)
    if known.sum() < 64 or len(row_index) < 64:
        raise RuntimeError(f"insufficient known/eval rows {known.sum()}/{len(row_index)}")
    anchor_index = int(np.flatnonzero(known)[-1])
    anchor_md = float(md[anchor_index])

    strong_eval = frame["strong"].to_numpy(np.float64)
    truth_eval = frame["tvt_true"].to_numpy(np.float64)
    if not np.isfinite(strong_eval).all() or not np.isfinite(truth_eval).all():
        raise RuntimeError("non-finite strong/truth")
    base_tvt = np.full(len(horizontal), np.nan, np.float64)
    base_tvt[known] = visible[known]
    base_tvt[row_index] = strong_eval
    base_tvt = _fill_1d(md, base_tvt)

    tw_tvt, tw_gr = _deduplicate_axis(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    tw_at_known = np.interp(visible[known], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    gain, intercept, calibration_pairs, calibration_rmse = _robust_affine(
        tw_at_known, gr[known]
    )
    centre, scale = _robust_location_scale(gr[known])
    observed = (_fill_1d(md, gr) - centre) / scale
    calibrated_tw = (gain * tw_gr + intercept - centre) / scale

    candidate_tvt = base_tvt[:, None] + STATE_OFFSETS[None, :]
    expected = np.interp(
        candidate_tvt.reshape(-1),
        tw_tvt,
        calibrated_tw,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape).astype(np.float32)
    valid = np.isfinite(expected) & np.isfinite(gr)[:, None]

    sample_positions = _sample_rows(np.arange(len(row_index)), MAX_STATIONS_PER_WELL)
    sample_full_index = row_index[sample_positions]
    truth_residual = truth_eval[sample_positions] - strong_eval[sample_positions]
    axis_covered = np.abs(truth_residual) <= STATE_HALF_WIDTH_FT
    distance = md[sample_full_index] - anchor_md
    subsets = _distance_subset(distance)
    median_step = max(float(abs(np.nanmedian(np.diff(md[np.isfinite(md)])))), 0.25)
    window_rows = {
        float(window): max(9, int(round(float(window) / median_step)))
        for window in WINDOWS_FT
    }
    scores = _window_scores(observed, expected, valid, sample_full_index, window_rows)

    emissions = {}
    for window in WINDOWS_FT:
        ncc, rmse, _ = scores[window]
        emissions[f"ncc_{int(window)}"] = _row_standardise(ncc)
        emissions[f"neg_rmse_{int(window)}"] = _row_standardise(-rmse)
    emissions["ncc_all"] = _finite_mean(
        [emissions[f"ncc_{int(window)}"] for window in WINDOWS_FT]
    )
    emissions["neg_rmse_all"] = _finite_mean(
        [emissions[f"neg_rmse_{int(window)}"] for window in WINDOWS_FT]
    )
    emissions["value_combo"] = _finite_mean(
        [emissions["ncc_all"], emissions["neg_rmse_all"]]
    )
    global_profile = _global_ncc(observed[row_index], expected[row_index], valid[row_index])
    global_score = _row_standardise(global_profile[None, :])
    emissions["global_ncc"] = np.broadcast_to(
        global_score, (len(sample_positions), len(STATE_OFFSETS))
    ).copy()
    emissions["local_global_combo"] = _finite_mean(
        [emissions["value_combo"], emissions["global_ncc"]]
    )

    prior = -0.5 * (STATE_OFFSETS / PRIOR_SCALE_FT) ** 2
    prior = np.broadcast_to(prior[None, :], (len(sample_positions), len(STATE_OFFSETS)))
    variants = {"prior_only": prior.astype(np.float32)}
    for name, emission in emissions.items():
        for weight in PRIOR_WEIGHTS:
            variants[f"{name}__prior{weight:g}"] = (
                emission + float(weight) * prior
            ).astype(np.float32)
    reference = variants["neg_rmse_128__prior0.5"]
    shift = 31 + (int(str(well)[:4], 16) % 61)
    variants["control_shifted_neg_rmse128__prior0.5"] = (
        np.roll(emissions["neg_rmse_128"], shift, axis=1) + 0.5 * prior
    ).astype(np.float32)

    payloads = {}
    for name, score in variants.items():
        prediction, rank, top5, top8 = _rank_rows(score, truth_residual)
        payloads[name] = {
            "prediction": prediction,
            "rank": rank,
            "top5": top5,
            "top8": top8,
            "truth_residual": truth_residual,
            "axis_covered": axis_covered,
            "distance_subset": subsets,
        }
    return {
        "well": well,
        "eval_rows": int(len(row_index)),
        "sample_rows": int(len(sample_positions)),
        "axis_coverage": float(np.mean(axis_covered)),
        "base_sse": float(np.sum((strong_eval - truth_eval) ** 2)),
        "base_rows": int(len(frame)),
        "calibration_pairs": calibration_pairs,
        "calibration_rmse": calibration_rmse,
        "variants": payloads,
    }


print(
    {
        "experiment": EXPERIMENT,
        "audit_fold": AUDIT_FOLD,
        "wells": len(audit_wells),
        "states": len(STATE_OFFSETS),
        "state_axis_ft": [float(STATE_OFFSETS[0]), float(STATE_OFFSETS[-1])],
        "windows_ft": WINDOWS_FT,
        "workers": MAX_WORKERS,
    },
    flush=True,
)

started = time.time()
results = []
failures = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(_audit_well, well): well for well in audit_wells}
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="strong-centred alignment"
    ):
        well = futures[future]
        try:
            results.append(future.result())
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})

pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
if len(results) < int(0.95 * len(audit_wells)):
    raise RuntimeError(f"only {len(results)} / {len(audit_wells)} wells succeeded")

guard_frame = pd.DataFrame(
    [
        {
            "well": result["well"],
            "eval_rows": result["eval_rows"],
            "sample_rows": result["sample_rows"],
            "axis_coverage": result["axis_coverage"],
            "calibration_pairs": result["calibration_pairs"],
            "calibration_rmse": result["calibration_rmse"],
        }
        for result in results
    ]
).sort_values("well")
guard_frame.to_csv(OUT_DIR / "well_guards.csv", index=False)

subsets = ("all", "near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
metric_rows = []
variant_names = sorted(results[0]["variants"])
for variant in variant_names:
    for subset in subsets:
        pooled_error = []
        pooled_rank = []
        pooled_top5 = []
        pooled_top8 = []
        well_rmse = []
        requested_rows = 0
        for result in results:
            payload = result["variants"][variant]
            eligible = payload["axis_covered"].copy()
            if subset != "all":
                eligible &= payload["distance_subset"] == subset
            requested_rows += int(eligible.sum())
            mask = eligible & np.isfinite(payload["prediction"])
            if not mask.any():
                continue
            error = payload["prediction"][mask] - payload["truth_residual"][mask]
            pooled_error.append(error)
            pooled_rank.append(payload["rank"][mask & np.isfinite(payload["rank"])])
            pooled_top5.append(payload["top5"][mask])
            pooled_top8.append(payload["top8"][mask])
            well_rmse.append(float(np.sqrt(np.mean(error * error))))
        if not pooled_error:
            continue
        error = np.concatenate(pooled_error)
        ranks = np.concatenate(pooled_rank) if pooled_rank else np.array([])
        top5 = np.concatenate(pooled_top5)
        top8 = np.concatenate(pooled_top8)
        metric_rows.append(
            {
                "variant": variant,
                "subset": subset,
                "rows": int(len(error)),
                "requested_rows": int(requested_rows),
                "row_coverage": float(len(error) / max(requested_rows, 1)),
                "wells": int(len(well_rmse)),
                "argmax_rmse": float(np.sqrt(np.mean(error * error))),
                "argmax_mae": float(np.mean(np.abs(error))),
                "top1_within5": float(np.mean(np.abs(error) <= 5.0)),
                "top5_contains_truth": float(np.mean(top5)),
                "top8_contains_truth": float(np.mean(top8)),
                "median_truth_rank": float(np.median(ranks)) if len(ranks) else np.nan,
                "p90_truth_rank": float(np.quantile(ranks, 0.90)) if len(ranks) else np.nan,
                "median_well_rmse": float(np.median(well_rmse)),
                "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            }
        )

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUT_DIR / "feature_metrics.csv", index=False)
overall = metrics[metrics["subset"].eq("all")].copy()
overall = overall.sort_values(
    ["top8_contains_truth", "median_truth_rank", "argmax_rmse"],
    ascending=[False, True, True],
)
prior = overall[overall["variant"].eq("prior_only")].iloc[0]
evidence = overall[
    ~overall["variant"].isin(
        ("prior_only", "control_shifted_neg_rmse128__prior0.5")
    )
    & (overall["row_coverage"] >= 0.98)
]
if evidence.empty:
    raise RuntimeError("no evidence variant reaches 98% row coverage")
best = evidence.iloc[0]
best_deep = metrics[
    metrics["variant"].eq(best["variant"])
    & metrics["subset"].eq("far_gt2000")
].iloc[0]
prior_deep = metrics[
    metrics["variant"].eq("prior_only")
    & metrics["subset"].eq("far_gt2000")
].iloc[0]
control = overall[
    overall["variant"].eq("control_shifted_neg_rmse128__prior0.5")
].iloc[0]

base_rows = sum(result["base_rows"] for result in results)
base_sse = sum(result["base_sse"] for result in results)
base_rmse = float(np.sqrt(base_sse / base_rows))
top8_gain = float(best["top8_contains_truth"] - prior["top8_contains_truth"])
deep_top8_gain = float(
    best_deep["top8_contains_truth"] - prior_deep["top8_contains_truth"]
)
gates = {
    "best_top8_ge": 0.85,
    "top8_gain_vs_prior_ge": 0.08,
    "deep_top8_ge": 0.75,
    "deep_top8_gain_vs_prior_ge": 0.08,
    "median_rank_le": 3.0,
    "row_coverage_ge": 0.98,
    "axis_coverage_ge": 0.995,
}
checks = {
    "best_top8": bool(best["top8_contains_truth"] >= gates["best_top8_ge"]),
    "top8_gain": bool(top8_gain >= gates["top8_gain_vs_prior_ge"]),
    "deep_top8": bool(best_deep["top8_contains_truth"] >= gates["deep_top8_ge"]),
    "deep_top8_gain": bool(deep_top8_gain >= gates["deep_top8_gain_vs_prior_ge"]),
    "median_rank": bool(best["median_truth_rank"] <= gates["median_rank_le"]),
    "row_coverage": bool(best["row_coverage"] >= gates["row_coverage_ge"]),
    "axis_coverage": bool(
        guard_frame["axis_coverage"].mean() >= gates["axis_coverage_ge"]
    ),
}
strict_pass = bool(all(checks.values()))

print("\nTOP STRONG-CENTRED FEATURES")
print(overall.head(25).to_string(index=False))
print("\nBEST VS PRIOR")
print(
    metrics[
        metrics["variant"].isin([best["variant"], "prior_only"])
    ].to_string(index=False)
)

verdict = _jsonable(
    {
        "experiment": EXPERIMENT,
        "stage": "zero_training_strong_centered_alignment_gateway",
        "audit_fold": AUDIT_FOLD,
        "sample_unit": "complete held-out physical well; sampled stations for rank audit",
        "centre": "fold-clean strong OOF",
        "uses_neighbor": False,
        "uses_cp12_gold": False,
        "wells_requested": len(audit_wells),
        "wells_succeeded": len(results),
        "base_rmse": base_rmse,
        "states": len(STATE_OFFSETS),
        "state_half_width_ft": STATE_HALF_WIDTH_FT,
        "state_step_ft": STATE_STEP_FT,
        "mean_axis_coverage": float(guard_frame["axis_coverage"].mean()),
        "min_axis_coverage": float(guard_frame["axis_coverage"].min()),
        "median_calibration_rmse": float(guard_frame["calibration_rmse"].median()),
        "prior_only": prior.to_dict(),
        "best_evidence": best.to_dict(),
        "best_evidence_deep": best_deep.to_dict(),
        "top8_gain_vs_prior": top8_gain,
        "deep_top8_gain_vs_prior": deep_top8_gain,
        "shift_control": control.to_dict(),
        "gates": gates,
        "checks": checks,
        "strict_pass": strict_pass,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "next": (
            "PASS -> train a pathwise decoder on this strong-centred cost volume"
            if strict_pass
            else "FAIL -> proposal prior is not the missing ingredient; stop GR-emission decoder work"
        ),
    }
)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
