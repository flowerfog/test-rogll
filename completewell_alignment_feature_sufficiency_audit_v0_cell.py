"""Zero-training audit of complete-well GR alignment input sufficiency.

The audit asks whether legal, explicitly constructed alignment channels rank
the true TVT state well enough before another neural decoder is trained.

One complete held-out well is the sample unit. For every sampled evaluation
station, states represent structure offsets around the visible heel anchor:

    candidate_tvt(md, state) = anchor_tvt + anchor_z + state - z(md)

All emissions use only full horizontal GR, the same well's typewell, visible
TVT_input prefix, and trajectory. Validation TVT is used only to calculate the
true state and ranking metrics.
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


EXPERIMENT = "completewell_alignment_feature_sufficiency_audit_v0"
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
FOLD_SOURCE = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT

AUDIT_FOLD = 1
STATE_HALF_WIDTH_FT = 320.0
STATE_STEP_FT = 2.0
STATE_OFFSETS = np.arange(
    -STATE_HALF_WIDTH_FT,
    STATE_HALF_WIDTH_FT + 0.5 * STATE_STEP_FT,
    STATE_STEP_FT,
    dtype=np.float32,
)
WINDOWS_FT = (32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 1600.0)
DERIVATIVE_WINDOWS_FT = (128.0, 256.0, 512.0)
HEEL_WINDOWS_FT = (128.0, 256.0, 512.0)
MAX_STATIONS_PER_WELL = 256
MIN_WINDOW_COVERAGE = 0.55
MAX_WORKERS = min(3, max(1, os.cpu_count() or 1))

OUT_DIR.mkdir(parents=True, exist_ok=True)
if not DATA_ROOT.exists():
    raise FileNotFoundError(DATA_ROOT)
if not FOLD_SOURCE.exists():
    raise FileNotFoundError(FOLD_SOURCE)


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(float(sse) / max(int(rows), 1)))


def _robust_location_scale(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-3)
    return centre, scale


def _robust_affine(source, target):
    """Fit target ~= gain * source + intercept using clipped residual IRLS."""

    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    valid = np.isfinite(source) & np.isfinite(target)
    x = source[valid]
    y = target[valid]
    if len(x) < 32:
        shift = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, shift, int(len(x)), np.nan
    keep = np.ones(len(x), bool)
    gain, intercept = 1.0, 0.0
    for _ in range(5):
        design = np.c_[x[keep], np.ones(int(keep.sum()))]
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (gain * x + intercept)
        centre, scale = _robust_location_scale(residual[keep])
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if int(next_keep.sum()) < 32 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    gain = float(np.clip(gain, 0.15, 6.0))
    residual = y[keep] - (gain * x[keep] + intercept)
    rmse = float(np.sqrt(np.mean(residual * residual))) if len(residual) else np.nan
    return gain, float(intercept), int(keep.sum()), rmse


def _deduplicate_axis(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).dropna()
    frame = frame.groupby("axis", as_index=False)["value"].mean().sort_values("axis")
    return frame["axis"].to_numpy(np.float64), frame["value"].to_numpy(np.float64)


def _fill_1d(axis, values):
    axis = np.asarray(axis, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.isfinite(axis) & np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros(len(values), np.float64)
    order = np.argsort(axis[valid])
    return np.interp(axis, axis[valid][order], values[valid][order])


def _gradient_on_axis(values, axis):
    """Differentiate on a strictly increasing surrogate for noisy MD axes."""

    values = np.asarray(values, np.float32)
    axis = np.asarray(axis, np.float64)
    finite_axis = np.isfinite(axis)
    if finite_axis.sum() < 2:
        return np.zeros_like(values, dtype=np.float32)
    safe_axis = axis.copy()
    safe_axis[~finite_axis] = np.interp(
        np.flatnonzero(~finite_axis),
        np.flatnonzero(finite_axis),
        safe_axis[finite_axis],
    )
    spacing = np.diff(safe_axis)
    positive = spacing[np.isfinite(spacing) & (spacing > 1e-6)]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    for index in range(1, len(safe_axis)):
        if safe_axis[index] <= safe_axis[index - 1]:
            safe_axis[index] = safe_axis[index - 1] + fallback
    return np.gradient(values, safe_axis, axis=0).astype(np.float32)


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


def _sample_eval_rows(eval_index, maximum):
    eval_index = np.asarray(eval_index, np.int64)
    if len(eval_index) <= maximum:
        return eval_index
    positions = np.linspace(0, len(eval_index) - 1, maximum)
    return np.unique(eval_index[np.round(positions).astype(np.int64)])


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


def _window_bounds(sample_index, rows_per_window, n_rows):
    half = max(int(rows_per_window) // 2, 1)
    low = np.maximum(np.asarray(sample_index, np.int64) - half, 0)
    high = np.minimum(np.asarray(sample_index, np.int64) + half + 1, n_rows)
    return low, high


def _window_scores(observed, expected, valid, sample_index, window_rows):
    """Return local NCC, RMSE and valid fraction for every sampled row/state."""

    observed = np.asarray(observed, np.float32)
    expected = np.asarray(expected, np.float32)
    valid = np.asarray(valid, bool)
    observed_column = observed[:, None]
    valid_float = valid.astype(np.float32)
    expected_zero = np.where(valid, expected, 0.0).astype(np.float32)
    observed_zero = np.where(valid, observed_column, 0.0).astype(np.float32)

    cumulative_valid = _cumulative(valid_float)
    cumulative_observed = _cumulative(observed_zero)
    cumulative_expected = _cumulative(expected_zero)
    cumulative_observed2 = _cumulative(observed_zero * observed_zero)
    cumulative_expected2 = _cumulative(expected_zero * expected_zero)
    cumulative_cross = _cumulative(observed_zero * expected_zero)

    output = {}
    for window_ft, rows in window_rows.items():
        low, high = _window_bounds(sample_index, rows, len(observed))
        count = cumulative_valid[high] - cumulative_valid[low]
        sum_observed = cumulative_observed[high] - cumulative_observed[low]
        sum_expected = cumulative_expected[high] - cumulative_expected[low]
        sum_observed2 = cumulative_observed2[high] - cumulative_observed2[low]
        sum_expected2 = cumulative_expected2[high] - cumulative_expected2[low]
        sum_cross = cumulative_cross[high] - cumulative_cross[low]

        denominator = np.maximum(count, 1.0)
        covariance = sum_cross - sum_observed * sum_expected / denominator
        variance_observed = np.maximum(
            sum_observed2 - sum_observed * sum_observed / denominator, 0.0
        )
        variance_expected = np.maximum(
            sum_expected2 - sum_expected * sum_expected / denominator, 0.0
        )
        ncc = covariance / np.sqrt(variance_observed * variance_expected + 1e-6)
        squared_error = (
            sum_observed2 + sum_expected2 - 2.0 * sum_cross
        ) / denominator
        rmse = np.sqrt(np.maximum(squared_error, 0.0))
        nominal = np.maximum((high - low)[:, None], 1)
        coverage = count / nominal
        usable = (coverage >= MIN_WINDOW_COVERAGE) & (count >= 8)
        ncc = np.where(usable, ncc, np.nan).astype(np.float32)
        rmse = np.where(usable, rmse, np.nan).astype(np.float32)
        output[float(window_ft)] = (ncc, rmse, coverage.astype(np.float32))
    return output


def _finite_mean(arrays):
    stack = np.stack(arrays, axis=0).astype(np.float32)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def _global_ncc(observed, expected, valid):
    observed = np.asarray(observed, np.float64)[:, None]
    expected = np.asarray(expected, np.float64)
    valid = np.asarray(valid, bool)
    count = valid.sum(axis=0).astype(np.float64)
    denominator = np.maximum(count, 1.0)
    observed_zero = np.where(valid, observed, 0.0)
    expected_zero = np.where(valid, expected, 0.0)
    sum_observed = observed_zero.sum(axis=0)
    sum_expected = expected_zero.sum(axis=0)
    covariance = (observed_zero * expected_zero).sum(axis=0) - (
        sum_observed * sum_expected / denominator
    )
    variance_observed = (observed_zero * observed_zero).sum(axis=0) - (
        sum_observed * sum_observed / denominator
    )
    variance_expected = (expected_zero * expected_zero).sum(axis=0) - (
        sum_expected * sum_expected / denominator
    )
    score = covariance / np.sqrt(
        np.maximum(variance_observed, 0.0) * np.maximum(variance_expected, 0.0)
        + 1e-6
    )
    score[count < max(32, 0.25 * len(observed))] = np.nan
    return score.astype(np.float32)


def _rank_rows(score, true_offset):
    score = np.asarray(score, np.float64)
    true_offset = np.asarray(true_offset, np.float64)
    n_rows = len(true_offset)
    predicted = np.full(n_rows, np.nan, np.float64)
    rank = np.full(n_rows, np.nan, np.float64)
    top5 = np.zeros(n_rows, bool)
    top8 = np.zeros(n_rows, bool)
    score_coverage = np.zeros(n_rows, np.float64)
    for row in range(n_rows):
        values = score[row]
        valid = np.isfinite(values)
        score_coverage[row] = float(valid.mean())
        if valid.sum() < 8:
            continue
        valid_indices = np.flatnonzero(valid)
        order = valid_indices[np.argsort(values[valid_indices])[::-1]]
        predicted[row] = float(STATE_OFFSETS[order[0]])
        nearest = int(np.argmin(np.abs(STATE_OFFSETS - true_offset[row])))
        if not valid[nearest]:
            continue
        rank[row] = float(1 + np.sum(values[valid] > values[nearest]))
        top5[row] = bool(
            np.any(np.abs(STATE_OFFSETS[order[:5]] - true_offset[row]) <= 5.0)
        )
        top8[row] = bool(
            np.any(np.abs(STATE_OFFSETS[order[:8]] - true_offset[row]) <= 5.0)
        )
    return {
        "predicted_offset": predicted,
        "rank": rank,
        "top5": top5,
        "top8": top8,
        "score_coverage": score_coverage,
    }


def _well_audit(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.isfinite(visible)
    evaluation = ~known & np.isfinite(truth)
    if known.sum() < 64 or evaluation.sum() < 64:
        raise RuntimeError(f"insufficient known/eval rows {known.sum()}/{evaluation.sum()}")
    anchor_index = int(np.flatnonzero(known)[-1])
    anchor_md = float(md[anchor_index])
    anchor_structure = float(visible[anchor_index] + z[anchor_index])

    tw_tvt, tw_gr = _deduplicate_axis(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    if len(tw_tvt) < 32:
        raise RuntimeError("typewell has insufficient finite rows")

    tw_at_known = np.interp(visible[known], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    gain, intercept, calibration_pairs, calibration_rmse = _robust_affine(
        tw_at_known, gr[known]
    )
    calibrated_tw_gr = gain * tw_gr + intercept
    centre, scale = _robust_location_scale(gr[known])
    horizontal_normalised = (_fill_1d(md, gr) - centre) / scale
    calibrated_tw_normalised = (calibrated_tw_gr - centre) / scale

    candidate_tvt = (
        anchor_structure
        + STATE_OFFSETS[None, :].astype(np.float64)
        - z[:, None]
    )
    flat_candidate = candidate_tvt.reshape(-1)
    expected_type = np.interp(
        flat_candidate,
        tw_tvt,
        calibrated_tw_normalised,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape).astype(np.float32)
    horizontal_valid = np.isfinite(gr)
    type_valid = np.isfinite(expected_type) & horizontal_valid[:, None]

    known_tvt, known_gr = _deduplicate_axis(visible[known], gr[known])
    known_gr_normalised = (known_gr - centre) / scale
    expected_heel = np.interp(
        flat_candidate,
        known_tvt,
        known_gr_normalised,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape).astype(np.float32)
    heel_valid = np.isfinite(expected_heel) & horizontal_valid[:, None]

    eval_index = np.flatnonzero(evaluation)
    sample_index = _sample_eval_rows(eval_index, MAX_STATIONS_PER_WELL)
    true_offset = truth[sample_index] + z[sample_index] - anchor_structure
    axis_covered = np.abs(true_offset) <= STATE_HALF_WIDTH_FT
    distance = md[sample_index] - anchor_md
    distance_subset = _distance_subset(distance)

    median_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
    median_step = max(abs(median_step), 0.25)
    window_rows = {
        float(window): max(9, int(round(float(window) / median_step)))
        for window in WINDOWS_FT
    }
    value_scores = _window_scores(
        horizontal_normalised,
        expected_type,
        type_valid,
        sample_index,
        window_rows,
    )

    horizontal_derivative = _gradient_on_axis(horizontal_normalised, md)
    expected_derivative = _gradient_on_axis(expected_type, md)
    derivative_valid = np.isfinite(expected_derivative) & horizontal_valid[:, None]
    derivative_rows = {
        window: window_rows[window] for window in DERIVATIVE_WINDOWS_FT
    }
    derivative_scores = _window_scores(
        horizontal_derivative,
        expected_derivative,
        derivative_valid,
        sample_index,
        derivative_rows,
    )

    heel_rows = {window: window_rows[window] for window in HEEL_WINDOWS_FT}
    heel_scores = _window_scores(
        horizontal_normalised,
        expected_heel,
        heel_valid,
        sample_index,
        heel_rows,
    )

    pointwise_type = -np.abs(
        horizontal_normalised[sample_index, None] - expected_type[sample_index]
    )
    pointwise_type[~type_valid[sample_index]] = np.nan
    pointwise_heel = -np.abs(
        horizontal_normalised[sample_index, None] - expected_heel[sample_index]
    )
    pointwise_heel[~heel_valid[sample_index]] = np.nan

    score_variants = {
        "pointwise_type": pointwise_type.astype(np.float32),
        "pointwise_heel": pointwise_heel.astype(np.float32),
    }
    for window in WINDOWS_FT:
        ncc, rmse, _ = value_scores[window]
        score_variants[f"type_ncc_{int(window)}"] = ncc
        score_variants[f"type_neg_rmse_{int(window)}"] = -rmse
    for window in HEEL_WINDOWS_FT:
        score_variants[f"heel_ncc_{int(window)}"] = heel_scores[window][0]
    for window in DERIVATIVE_WINDOWS_FT:
        score_variants[f"derivative_ncc_{int(window)}"] = derivative_scores[window][0]

    short_ncc = _finite_mean([value_scores[w][0] for w in WINDOWS_FT[:4]])
    long_ncc = _finite_mean([value_scores[w][0] for w in WINDOWS_FT[3:]])
    all_ncc = _finite_mean([value_scores[w][0] for w in WINDOWS_FT])
    derivative_ncc = _finite_mean(
        [derivative_scores[w][0] for w in DERIVATIVE_WINDOWS_FT]
    )
    heel_ncc = _finite_mean([heel_scores[w][0] for w in HEEL_WINDOWS_FT])
    rmse_128 = value_scores[128.0][1]
    type_combo = (
        0.55 * all_ncc + 0.25 * derivative_ncc - 0.20 * np.clip(rmse_128, 0.0, 5.0)
    ).astype(np.float32)
    type_heel_combo = _finite_mean([type_combo, heel_ncc])
    global_profile = _global_ncc(
        horizontal_normalised[evaluation],
        expected_type[evaluation],
        type_valid[evaluation],
    )
    global_volume = np.broadcast_to(global_profile[None, :], type_combo.shape)
    global_combo = _finite_mean([type_combo, 0.35 * global_volume])

    score_variants.update(
        {
            "type_ncc_short": short_ncc,
            "type_ncc_long": long_ncc,
            "type_ncc_all": all_ncc,
            "type_derivative_combo": _finite_mean([all_ncc, derivative_ncc]),
            "type_value_derivative_combo": type_combo,
            "type_heel_combo": type_heel_combo,
            "type_global_combo": global_combo,
        }
    )

    # Matched negative control: preserve each score profile but destroy its
    # absolute TVT registration with a deterministic circular state shift.
    control_shift = 73 + (int(str(well)[:4], 16) % 97)
    score_variants["control_shifted_type_combo"] = np.roll(
        type_combo, control_shift, axis=1
    )

    result = {
        "well": str(well),
        "eval_rows": int(len(eval_index)),
        "sample_rows": int(len(sample_index)),
        "axis_coverage": float(np.mean(axis_covered)),
        "calibration_gain": gain,
        "calibration_intercept": intercept,
        "calibration_pairs": calibration_pairs,
        "calibration_rmse": calibration_rmse,
        "median_md_step": median_step,
        "variants": {},
    }
    for name, score in score_variants.items():
        ranked = _rank_rows(score, true_offset)
        result["variants"][name] = {
            **ranked,
            "true_offset": true_offset.copy(),
            "distance": distance.copy(),
            "distance_subset": distance_subset.copy(),
            "axis_covered": axis_covered.copy(),
        }
    return result


fold_frame = pd.read_csv(FOLD_SOURCE, dtype={"well": str})
well_column = next(
    (column for column in ("well", "well_id", "WELL") if column in fold_frame), None
)
fold_column = next(
    (column for column in ("fold", "outer_fold", "Fold") if column in fold_frame), None
)
if well_column is None or fold_column is None:
    raise RuntimeError(f"Invalid fold metadata: {fold_frame.columns.tolist()}")
fold_frame[fold_column] = pd.to_numeric(fold_frame[fold_column], errors="coerce")
audit_wells = sorted(
    fold_frame.loc[fold_frame[fold_column].eq(AUDIT_FOLD), well_column]
    .astype(str)
    .unique()
    .tolist()
)
if len(audit_wells) < 100:
    raise RuntimeError(f"Unexpected Fold-{AUDIT_FOLD} well count: {len(audit_wells)}")

print(
    {
        "experiment": EXPERIMENT,
        "audit_fold": AUDIT_FOLD,
        "wells": len(audit_wells),
        "states": len(STATE_OFFSETS),
        "state_axis_ft": [float(STATE_OFFSETS[0]), float(STATE_OFFSETS[-1])],
        "windows_ft": WINDOWS_FT,
        "max_stations_per_well": MAX_STATIONS_PER_WELL,
        "workers": MAX_WORKERS,
    },
    flush=True,
)

started = time.time()
results = []
failures = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_well = {executor.submit(_well_audit, well): well for well in audit_wells}
    for future in tqdm(
        as_completed(future_to_well),
        total=len(future_to_well),
        desc="complete-well alignment features",
    ):
        well = future_to_well[future]
        try:
            results.append(future.result())
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})

pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
if len(results) < int(0.95 * len(audit_wells)):
    raise RuntimeError(f"Only {len(results)} / {len(audit_wells)} wells succeeded")

guard_rows = [
    {
        "well": result["well"],
        "eval_rows": result["eval_rows"],
        "sample_rows": result["sample_rows"],
        "axis_coverage": result["axis_coverage"],
        "calibration_gain": result["calibration_gain"],
        "calibration_intercept": result["calibration_intercept"],
        "calibration_pairs": result["calibration_pairs"],
        "calibration_rmse": result["calibration_rmse"],
        "median_md_step": result["median_md_step"],
    }
    for result in results
]
guard_frame = pd.DataFrame(guard_rows).sort_values("well")
guard_frame.to_csv(OUT_DIR / "well_guards.csv", index=False)

variant_names = sorted(results[0]["variants"])
metric_rows = []
per_well_rows = []
subsets = ("all", "near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
for variant in variant_names:
    for subset in subsets:
        pooled_error = []
        pooled_rank = []
        pooled_top5 = []
        pooled_top8 = []
        pooled_coverage = []
        well_rmse = []
        well_top8 = []
        requested_rows = 0
        for result in results:
            payload = result["variants"][variant]
            eligible = payload["axis_covered"].copy()
            if subset != "all":
                eligible &= payload["distance_subset"] == subset
            requested_rows += int(eligible.sum())
            mask = eligible & np.isfinite(payload["predicted_offset"])
            if not mask.any():
                continue
            error = payload["predicted_offset"][mask] - payload["true_offset"][mask]
            valid_rank = np.isfinite(payload["rank"][mask])
            pooled_error.append(error)
            pooled_rank.append(payload["rank"][mask][valid_rank])
            pooled_top5.append(payload["top5"][mask])
            pooled_top8.append(payload["top8"][mask])
            pooled_coverage.append(payload["score_coverage"][mask])
            well_rmse.append(float(np.sqrt(np.mean(error * error))))
            well_top8.append(float(np.mean(payload["top8"][mask])))
            if subset == "all":
                per_well_rows.append(
                    {
                        "variant": variant,
                        "well": result["well"],
                        "rows": int(mask.sum()),
                        "rmse": well_rmse[-1],
                        "top8_within5": well_top8[-1],
                        "median_rank": (
                            float(np.median(payload["rank"][mask][valid_rank]))
                            if valid_rank.any()
                            else np.nan
                        ),
                    }
                )
        if not pooled_error:
            continue
        errors = np.concatenate(pooled_error)
        ranks = np.concatenate(pooled_rank) if pooled_rank else np.array([])
        top5 = np.concatenate(pooled_top5)
        top8 = np.concatenate(pooled_top8)
        coverage = np.concatenate(pooled_coverage)
        metric_rows.append(
            {
                "variant": variant,
                "subset": subset,
                "rows": int(len(errors)),
                "requested_rows": int(requested_rows),
                "row_coverage": float(len(errors) / max(requested_rows, 1)),
                "wells": int(len(well_rmse)),
                "argmax_rmse": float(np.sqrt(np.mean(errors * errors))),
                "argmax_mae": float(np.mean(np.abs(errors))),
                "top1_within5": float(np.mean(np.abs(errors) <= 5.0)),
                "top5_contains_truth": float(np.mean(top5)),
                "top8_contains_truth": float(np.mean(top8)),
                "median_truth_rank": float(np.median(ranks)) if len(ranks) else np.nan,
                "p90_truth_rank": float(np.quantile(ranks, 0.90)) if len(ranks) else np.nan,
                "mean_score_coverage": float(np.mean(coverage)),
                "median_well_rmse": float(np.median(well_rmse)),
                "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
                "median_well_top8": float(np.median(well_top8)),
            }
        )

metrics = pd.DataFrame(metric_rows)
per_well = pd.DataFrame(per_well_rows)
metrics.to_csv(OUT_DIR / "feature_metrics.csv", index=False)
per_well.to_csv(OUT_DIR / "feature_per_well.csv", index=False)

overall = metrics[metrics["subset"].eq("all")].copy()
overall = overall.sort_values(
    ["top8_contains_truth", "median_truth_rank", "argmax_rmse"],
    ascending=[False, True, True],
)
deployable = overall[~overall["variant"].str.startswith("control_")].copy()
deployable = deployable[deployable["row_coverage"] >= 0.98].copy()
if deployable.empty:
    raise RuntimeError("No deployable feature has at least 98% eligible-row coverage")
best = deployable.iloc[0]
deep = metrics[
    metrics["subset"].eq("far_gt2000") & metrics["variant"].eq(best["variant"])
]
deep_record = deep.iloc[0] if len(deep) else None
control = overall[overall["variant"].eq("control_shifted_type_combo")]
control_record = control.iloc[0] if len(control) else None

gates = {
    "overall_top8_ge": 0.90,
    "deep_top8_ge": 0.85,
    "median_rank_le": 3.0,
    "p90_rank_le": 12.0,
    "overall_top1_ge": 0.55,
    "row_coverage_ge": 0.98,
    "axis_coverage_ge": 0.995,
    "shift_control_top8_drop_ge": 0.30,
}
checks = {
    "overall_top8": bool(best["top8_contains_truth"] >= gates["overall_top8_ge"]),
    "deep_top8": bool(
        deep_record is not None
        and deep_record["top8_contains_truth"] >= gates["deep_top8_ge"]
    ),
    "median_rank": bool(best["median_truth_rank"] <= gates["median_rank_le"]),
    "p90_rank": bool(best["p90_truth_rank"] <= gates["p90_rank_le"]),
    "overall_top1": bool(best["top1_within5"] >= gates["overall_top1_ge"]),
    "row_coverage": bool(best["row_coverage"] >= gates["row_coverage_ge"]),
    "axis_coverage": bool(guard_frame["axis_coverage"].mean() >= gates["axis_coverage_ge"]),
    "shift_control": bool(
        control_record is not None
        and best["top8_contains_truth"] - control_record["top8_contains_truth"]
        >= gates["shift_control_top8_drop_ge"]
    ),
}
strict_pass = bool(all(checks.values()))

print("\nTOP OVERALL FEATURES")
print(overall.head(20).to_string(index=False))
print("\nBEST FEATURE DISTANCE STRATA")
print(metrics[metrics["variant"].eq(best["variant"])].to_string(index=False))

verdict = {
    "experiment": EXPERIMENT,
    "stage": "zero_training_completewell_alignment_input_gateway",
    "sample_unit": "complete held-out physical well; sampled stations only for ranking",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "audit_fold": AUDIT_FOLD,
    "wells_requested": len(audit_wells),
    "wells_succeeded": len(results),
    "states": len(STATE_OFFSETS),
    "state_half_width_ft": STATE_HALF_WIDTH_FT,
    "state_step_ft": STATE_STEP_FT,
    "windows_ft": WINDOWS_FT,
    "guards": {
        "mean_axis_coverage": float(guard_frame["axis_coverage"].mean()),
        "min_axis_coverage": float(guard_frame["axis_coverage"].min()),
        "median_calibration_pairs": float(guard_frame["calibration_pairs"].median()),
        "median_calibration_rmse": float(guard_frame["calibration_rmse"].median()),
        "failed_wells": len(failures),
    },
    "best_feature": best.to_dict(),
    "best_feature_deep": deep_record.to_dict() if deep_record is not None else None,
    "shifted_state_control": (
        control_record.to_dict() if control_record is not None else None
    ),
    "gates": gates,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> freeze these channels; train on folds 1/3/4/5 and validate on untouched Fold2"
        if strict_pass
        else "FAIL -> raw fields are present but this explicit emission is insufficient; upgrade candidate-conditioned alignment before another decoder"
    ),
}
verdict = _jsonable(verdict)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
