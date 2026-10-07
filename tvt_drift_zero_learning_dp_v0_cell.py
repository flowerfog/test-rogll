"""Zero-learning fixed-TVT-drift global alignment audit.

This is the strict follow-up to the RankSDF axis audit.  It removes the neural
network and decodes a compact TVT state directly from target-free GR emissions.
All calibration and optional slope centring use only each well's visible
prefix.  Truth is read only after every path for that well has been decoded.
"""

from pathlib import Path
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")


DATA_ROOT_OVERRIDE = os.environ.get("ROGII_DATA_ROOT")
DATA_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path(r"E:\rogii-wellbore-geology-prediction"),
)
OOF_ROOT = Path(os.environ.get("ROGII_OOF_ROOT", "/root/autodl-tmp/oof_exports"))
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
FOLD_SOURCE_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
OUT_DIR = OOF_ROOT / "tvt_drift_zero_learning_dp_v0"
WELL_CACHE = OUT_DIR / "wells"

PRIMARY_HALF_WIDTH_FT = 96
EDGE_RERUN_HALF_WIDTH_FT = 128
STATE_STEP_FT = 1.0
ROW_STRIDE = 8
N_JOBS = 4
BATCH_WELLS = 50
MAX_WELLS = int(os.environ.get("ROGII_MAX_WELLS", "0"))

# Horizontal smoothing is in MD rows; typewell smoothing is in TVT feet.
HORIZONTAL_SIGMA_ROWS = (0.0, 4.0, 12.0, 32.0)
TYPEWELL_SIGMA_FT = (0.0, 1.5, 4.0, 10.0)
SCALE_WEIGHTS = (0.40, 0.30, 0.20, 0.10)

DP_CONFIGS = {
    "flat_loose": {
        "slope_source": "flat",
        "transition": 0.03,
        "jump_cost": 0.005,
        "max_slope": 0.25,
        "centre_sigma_ft": 64.0,
    },
    "flat_mid": {
        "slope_source": "flat",
        "transition": 0.15,
        "jump_cost": 0.02,
        "max_slope": 0.16,
        "centre_sigma_ft": 48.0,
    },
    "flat_strict": {
        "slope_source": "flat",
        "transition": 0.60,
        "jump_cost": 0.06,
        "max_slope": 0.10,
        "centre_sigma_ft": 32.0,
    },
    "prefix_mid": {
        "slope_source": "prefix",
        "transition": 0.15,
        "jump_cost": 0.02,
        "max_slope": 0.16,
        "centre_sigma_ft": 48.0,
    },
}
BLEND_WEIGHTS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)

for directory in (OUT_DIR, WELL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)


def _find_data_root():
    candidates = []
    if DATA_ROOT_OVERRIDE:
        candidates.append(Path(DATA_ROOT_OVERRIDE))
    candidates.extend(DATA_ROOT_CANDIDATES)
    for root in candidates:
        train = root / "train"
        if train.exists() and next(train.glob("*__horizontal_well.csv"), None):
            return root
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the competition train directory. Searched:\n" + searched
    )


DATA_ROOT = _find_data_root()
TRAIN_DIR = DATA_ROOT / "train"
HORIZONTAL_FILES = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
if MAX_WELLS > 0:
    HORIZONTAL_FILES = HORIZONTAL_FILES[:MAX_WELLS]


def _rmse(truth, prediction):
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _mad(values, floor=1e-6):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = np.median(values)
    return float(max(1.4826 * np.median(np.abs(values - centre)), floor))


def _fill(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    if valid.all():
        return values.copy()
    if valid.sum() < 2:
        fallback = float(np.nanmedian(values)) if valid.any() else 0.0
        return np.full(len(values), fallback, dtype=float)
    positions = np.arange(len(values), dtype=float)
    return np.interp(positions, positions[valid], values[valid])


def _smooth(values, sigma):
    values = np.asarray(values, float)
    if sigma <= 0:
        return values.copy()
    return gaussian_filter1d(values, sigma=float(sigma), mode="nearest")


def _robust_affine(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 30 or np.std(x) < 1e-6:
        return 0.0, 1.0
    keep = np.ones(len(x), dtype=bool)
    intercept, slope = 0.0, 1.0
    for _ in range(6):
        design = np.column_stack((np.ones(keep.sum()), x[keep]))
        intercept, slope = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (intercept + slope * x)
        scale = _mad(residual[keep], 1e-3)
        next_keep = np.abs(residual - np.median(residual[keep])) <= 3.5 * scale
        if next_keep.sum() < 30 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return float(intercept), float(np.clip(slope, 0.20, 4.0))


def _prepare_typewell(frame):
    frame = frame[["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame[np.isfinite(frame["TVT"]) & np.isfinite(frame["GR"])]
    frame = frame.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    tvt = frame["TVT"].to_numpy(float)
    gr = frame["GR"].to_numpy(float)
    if len(tvt) < 32 or np.any(np.diff(tvt) <= 0):
        raise ValueError("invalid typewell TVT/GR curve")
    return tvt, gr


def _prefix_slope(md, tvt_input, anchor):
    start_md = md[anchor] - 600.0
    use = (
        (np.arange(len(md)) <= anchor)
        & (md >= start_md)
        & np.isfinite(md)
        & np.isfinite(tvt_input)
    )
    if use.sum() < 30:
        return 0.0
    x = md[use] - md[anchor]
    y = tvt_input[use] - tvt_input[anchor]
    keep = np.ones(len(x), dtype=bool)
    slope = 0.0
    for _ in range(5):
        slope = float(np.polyfit(x[keep], y[keep], 1)[0])
        residual = y - slope * x
        scale = _mad(residual[keep], 1e-4)
        next_keep = np.abs(residual - np.median(residual[keep])) <= 3.5 * scale
        if next_keep.sum() < 20 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return float(np.clip(slope, -0.08, 0.08))


def _build_emission(
    gr_raw,
    tvt_input,
    sample_positions,
    candidate_tvt,
    tw_tvt,
    tw_gr,
):
    prefix = np.isfinite(gr_raw) & np.isfinite(tvt_input)
    prefix_expected = np.interp(
        tvt_input[prefix], tw_tvt, tw_gr, left=np.nan, right=np.nan
    )
    intercept, slope = _robust_affine(gr_raw[prefix], prefix_expected)
    calibrated = intercept + slope * gr_raw
    filled = _fill(calibrated)
    raw_valid = np.isfinite(gr_raw).astype(float)

    prefix_residual = calibrated[prefix] - prefix_expected
    residual_scale = max(_mad(prefix_residual, 1.0), 5.0)
    tw_step = max(float(np.median(np.diff(tw_tvt))), 1e-3)

    emission = np.zeros((len(sample_positions), len(candidate_tvt)), np.float32)
    confidence_total = np.zeros(len(sample_positions), np.float32)
    for h_sigma, tw_sigma_ft, weight in zip(
        HORIZONTAL_SIGMA_ROWS,
        TYPEWELL_SIGMA_FT,
        SCALE_WEIGHTS,
    ):
        observed = _smooth(filled, h_sigma)[sample_positions]
        confidence = (
            raw_valid[sample_positions]
            if h_sigma <= 0
            else np.clip(_smooth(raw_valid, h_sigma)[sample_positions], 0.0, 1.0)
        )
        expected_curve = _smooth(tw_gr, tw_sigma_ft / tw_step)
        expected = np.interp(
            candidate_tvt,
            tw_tvt,
            expected_curve,
            left=expected_curve[0],
            right=expected_curve[-1],
        )
        standardised = (observed[:, None] - expected[None, :]) / residual_scale
        # Smooth pseudo-Huber-like truncation prevents one GR spike dominating a path.
        robust_cost = np.minimum(standardised * standardised, 16.0)
        emission += (float(weight) * confidence[:, None] * robust_cost).astype(np.float32)
        confidence_total += (float(weight) * confidence).astype(np.float32)

    outside = (candidate_tvt < tw_tvt[0]) | (candidate_tvt > tw_tvt[-1])
    emission[:, outside] += 6.0
    emission -= np.min(emission, axis=1, keepdims=True)
    sorted_cost = np.partition(emission, kth=1, axis=1)[:, :2]
    sharpness = sorted_cost[:, 1] - sorted_cost[:, 0]
    return (
        emission,
        confidence_total,
        sharpness.astype(np.float32),
        intercept,
        slope,
        residual_scale,
    )


@njit(cache=True, nogil=True)
def _viterbi(
    emission,
    state_values,
    md_steps,
    expected_slope,
    transition_penalty,
    jump_cost,
    max_slope,
    centre_sigma_ft,
    anchor_state,
):
    n_rows, n_states = emission.shape
    large = np.float32(1e30)
    previous = np.full(n_states, large, dtype=np.float32)
    previous[anchor_state] = np.float32(0.0)
    backpointer = np.empty((n_rows, n_states), dtype=np.int16)

    for row in range(n_rows):
        current = np.full(n_states, large, dtype=np.float32)
        md_delta = max(float(md_steps[row]), 1.0)
        expected_change = float(expected_slope) * md_delta
        max_jump_ft = float(max_slope) * md_delta + 1.5 * STATE_STEP_FT
        max_jump_states = max(1, int(np.ceil(max_jump_ft / STATE_STEP_FT)))
        expected_shift = int(np.round(expected_change / STATE_STEP_FT))
        for state in range(n_states):
            expected_old = state - expected_shift
            start = max(0, expected_old - max_jump_states)
            stop = min(n_states, expected_old + max_jump_states + 1)
            best_cost = large
            best_old = state
            for old_state in range(start, stop):
                change = state_values[state] - state_values[old_state]
                residual_change = change - expected_change
                cost = (
                    previous[old_state]
                    + transition_penalty * residual_change * residual_change / md_delta
                )
                if abs(residual_change) > 0.50 * STATE_STEP_FT:
                    cost += jump_cost
                if cost < best_cost:
                    best_cost = cost
                    best_old = old_state
            centre_cost = 0.5 * (state_values[state] / centre_sigma_ft) ** 2
            current[state] = best_cost + emission[row, state] + centre_cost
            backpointer[row, state] = best_old
        previous = current

    end_state = int(np.argmin(previous))
    states = np.empty(n_rows, dtype=np.int16)
    state = end_state
    for row in range(n_rows - 1, -1, -1):
        states[row] = state
        state = int(backpointer[row, state])
    return states, float(previous[end_state])


def _decode_at_width(
    width,
    gr_raw,
    tvt_input,
    md,
    sample_positions,
    anchor,
    tw_tvt,
    tw_gr,
    prefix_slope,
):
    offsets = np.arange(-width, width + STATE_STEP_FT, STATE_STEP_FT, dtype=np.float32)
    anchor_tvt = float(tvt_input[anchor])
    candidate_tvt = anchor_tvt + offsets
    anchor_state = int(np.argmin(np.abs(offsets)))
    emission_data = _build_emission(
        gr_raw,
        tvt_input,
        sample_positions,
        candidate_tvt,
        tw_tvt,
        tw_gr,
    )
    emission, confidence, sharpness = emission_data[:3]
    previous_positions = np.concatenate(([anchor], sample_positions[:-1]))
    md_steps = (md[sample_positions] - md[previous_positions]).astype(np.float32)
    output = {}
    path_stats = {}
    for name, config in DP_CONFIGS.items():
        expected_slope = prefix_slope if config["slope_source"] == "prefix" else 0.0
        states, path_cost = _viterbi(
            emission,
            offsets,
            md_steps,
            np.float32(expected_slope),
            np.float32(config["transition"]),
            np.float32(config["jump_cost"]),
            np.float32(config["max_slope"]),
            np.float32(config["centre_sigma_ft"]),
            anchor_state,
        )
        sampled_offsets = offsets[states].astype(float)
        output[name] = sampled_offsets
        path_stats[name] = {
            "path_cost_per_sample": path_cost / max(len(sample_positions), 1),
            "edge_fraction": float(np.mean(np.abs(sampled_offsets) >= width - 2.0)),
            "mean_abs_offset": float(np.mean(np.abs(sampled_offsets))),
            "max_abs_offset": float(np.max(np.abs(sampled_offsets))),
        }
    metadata = {
        "confidence_mean": float(np.mean(confidence)),
        "sharpness_mean": float(np.mean(sharpness)),
        "calibration_intercept": float(emission_data[3]),
        "calibration_slope": float(emission_data[4]),
        "calibration_residual_scale": float(emission_data[5]),
    }
    return output, path_stats, metadata


def _decode_well(horizontal_path):
    well = horizontal_path.name.split("__", 1)[0]
    cache_path = WELL_CACHE / f"{well}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path), None
    try:
        typewell_path = TRAIN_DIR / f"{well}__typewell.csv"
        horizontal = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
        tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        evaluation = np.flatnonzero(~np.isfinite(tvt_input))
        if len(known) < 30 or len(evaluation) < 32:
            raise ValueError("insufficient known/evaluation rows")
        anchor = int(known[-1])
        if int(evaluation[0]) != anchor + 1:
            raise ValueError("TVT_input mask is not a contiguous prefix")
        sample_positions = evaluation[::ROW_STRIDE]
        if sample_positions[-1] != evaluation[-1]:
            sample_positions = np.append(sample_positions, evaluation[-1])

        tw_tvt, tw_gr = _prepare_typewell(typewell)
        prefix_slope = _prefix_slope(md, tvt_input, anchor)
        decoded, path_stats, metadata = _decode_at_width(
            PRIMARY_HALF_WIDTH_FT,
            gr_raw,
            tvt_input,
            md,
            sample_positions,
            anchor,
            tw_tvt,
            tw_gr,
            prefix_slope,
        )
        rerun = any(stats["edge_fraction"] > 0.01 for stats in path_stats.values())
        width = PRIMARY_HALF_WIDTH_FT
        if rerun:
            width = EDGE_RERUN_HALF_WIDTH_FT
            decoded, path_stats, metadata = _decode_at_width(
                width,
                gr_raw,
                tvt_input,
                md,
                sample_positions,
                anchor,
                tw_tvt,
                tw_gr,
                prefix_slope,
            )

        anchor_tvt = float(tvt_input[anchor])
        interpolation_x = np.concatenate(([md[anchor]], md[sample_positions]))
        frame = pd.DataFrame(
            {
                "well": well,
                "row_idx": evaluation.astype(np.int32),
                "tvt_true": tvt[evaluation].astype(np.float32),
                "carry": np.float32(anchor_tvt),
            }
        )
        for name, sampled_offsets in decoded.items():
            offsets = np.interp(
                md[evaluation],
                interpolation_x,
                np.concatenate(([0.0], sampled_offsets)),
            )
            frame[name] = (anchor_tvt + offsets).astype(np.float32)
        frame.attrs["diagnostic"] = {
            "well": well,
            "half_width_ft": width,
            "edge_rerun": rerun,
            "prefix_slope": prefix_slope,
            **metadata,
            **{
                f"{name}_{key}": value
                for name, stats in path_stats.items()
                for key, value in stats.items()
            },
        }
        frame.to_pickle(cache_path)
        return frame, None
    except Exception as exc:
        return None, {"well": well, "path": str(horizontal_path), "reason": repr(exc)}


def _load_fold_map():
    for path in FOLD_SOURCE_CANDIDATES:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"well": str})
        if {"well", "fold"}.issubset(frame.columns):
            return frame.set_index("well")["fold"].astype(int).to_dict(), path
    return {}, None


def _load_external_pool(wells):
    path = next((candidate for candidate in POOL_CANDIDATES if candidate.exists()), None)
    if path is None:
        return None, None
    header = pd.read_csv(path, nrows=0).columns.tolist()
    prediction_columns = [column for column in ("strong", "strong_bias", "cp12") if column in header]
    identity = ["well", "row_idx"] if {"well", "row_idx"}.issubset(header) else ["id"]
    pool = pd.read_csv(path, usecols=identity + prediction_columns, dtype={"well": str, "id": str})
    if "well" not in pool:
        split = pool["id"].str.rsplit("_", n=1, expand=True)
        pool["well"] = split[0].astype(str)
        pool["row_idx"] = split[1].astype(np.int32)
        pool = pool.drop(columns="id")
    pool["well"] = pool["well"].astype(str)
    pool["row_idx"] = pool["row_idx"].astype(np.int32)
    pool = pool[pool["well"].isin(wells)].copy()
    return pool, path


print(
    {
        "experiment": "tvt_drift_zero_learning_dp_v0",
        "data_root": str(DATA_ROOT),
        "wells": len(HORIZONTAL_FILES),
        "primary_half_width_ft": PRIMARY_HALF_WIDTH_FT,
        "state_step_ft": STATE_STEP_FT,
        "row_stride": ROW_STRIDE,
        "configs": DP_CONFIGS,
    },
    flush=True,
)

# Compile the Numba kernel before parallel threads enter it.
_ = _viterbi(
    np.zeros((2, 3), np.float32),
    np.arange(3, dtype=np.float32),
    np.ones(2, np.float32),
    np.float32(0.0),
    np.float32(0.1),
    np.float32(0.01),
    np.float32(0.2),
    np.float32(48.0),
    1,
)

started = time.time()
decoded_parts = []
failures = []
for start in range(0, len(HORIZONTAL_FILES), BATCH_WELLS):
    paths = HORIZONTAL_FILES[start : start + BATCH_WELLS]
    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_decode_well)(path) for path in paths
    )
    for frame, failure in results:
        if frame is not None:
            decoded_parts.append(frame)
        if failure is not None:
            failures.append(failure)
    done = min(start + len(paths), len(HORIZONTAL_FILES))
    print(
        f"decode {done}/{len(HORIZONTAL_FILES)} ok={len(decoded_parts)} "
        f"failures={len(failures)} elapsed={(time.time() - started) / 60.0:.1f} min",
        flush=True,
    )

if not decoded_parts:
    raise RuntimeError(f"No wells decoded. First failures: {failures[:10]}")

diagnostics = pd.DataFrame([frame.attrs["diagnostic"] for frame in decoded_parts])
decoded = pd.concat(decoded_parts, ignore_index=True)
fold_map, fold_source = _load_fold_map()
decoded["fold"] = decoded["well"].map(fold_map)
pool, pool_path = _load_external_pool(set(decoded["well"]))
if pool is not None:
    decoded = decoded.merge(pool, on=["well", "row_idx"], how="left", validate="one_to_one")

models = ["carry"] + list(DP_CONFIGS)
truth = decoded["tvt_true"].to_numpy(float)
well_indices = {
    str(well): np.asarray(list(indices), dtype=np.int64)
    for well, indices in decoded.groupby("well", sort=False).groups.items()
}
fold_indices = (
    {
        int(fold): np.asarray(list(indices), dtype=np.int64)
        for fold, indices in decoded.groupby("fold", sort=True).groups.items()
    }
    if decoded["fold"].notna().all()
    else {}
)

score_rows = []
fold_rows = []
bywell_rows = []
for model_name in models:
    prediction = decoded[model_name].to_numpy(float)
    perwell_gain = []
    carry_all = decoded["carry"].to_numpy(float)
    for well, indices in well_indices.items():
        y = truth[indices]
        carry = carry_all[indices]
        candidate = prediction[indices]
        gain = _rmse(y, carry) - _rmse(y, candidate)
        perwell_gain.append(gain)
        bywell_rows.append(
            {
                "model": model_name,
                "well": well,
                "carry_rmse": _rmse(y, carry),
                "rmse": _rmse(y, candidate),
                "gain_vs_carry": gain,
            }
        )
    score_rows.append(
        {
            "model": model_name,
            "rows": len(decoded),
            "wells": decoded["well"].nunique(),
            "carry_rmse": _rmse(truth, decoded["carry"]),
            "rmse": _rmse(truth, prediction),
            "gain_vs_carry": _rmse(truth, decoded["carry"]) - _rmse(truth, prediction),
            "mean_well_gain": float(np.mean(perwell_gain)),
            "median_well_gain": float(np.median(perwell_gain)),
            "positive_well_fraction": float(np.mean(np.asarray(perwell_gain) > 0)),
        }
    )
    if fold_indices:
        for fold, indices in fold_indices.items():
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "rows": len(indices),
                    "carry_rmse": _rmse(truth[indices], carry_all[indices]),
                    "rmse": _rmse(truth[indices], prediction[indices]),
                    "gain_vs_carry": _rmse(truth[indices], carry_all[indices])
                    - _rmse(truth[indices], prediction[indices]),
                }
            )

scores = pd.DataFrame(score_rows).sort_values("rmse").reset_index(drop=True)
fold_scores = pd.DataFrame(fold_rows)
bywell = pd.DataFrame(bywell_rows)

blend_rows = []
blend_fold_rows = []
external_bases = [column for column in ("strong", "strong_bias", "cp12") if column in decoded]
for base_name in external_bases:
    base = decoded[base_name].to_numpy(float)
    if not np.isfinite(base).all():
        continue
    base_rmse = _rmse(truth, base)
    for model_name in DP_CONFIGS:
        route = decoded[model_name].to_numpy(float)
        move = route - base
        error = base - truth
        analytic_weight = float(-np.mean(error * move) / max(np.mean(move * move), 1e-12))
        for weight in BLEND_WEIGHTS:
            prediction = base + float(weight) * move
            well_gains = []
            for indices in well_indices.values():
                well_gains.append(
                    _rmse(truth[indices], base[indices])
                    - _rmse(truth[indices], prediction[indices])
                )
            row = {
                "base": base_name,
                "model": model_name,
                "weight_route": weight,
                "base_rmse": base_rmse,
                "rmse": _rmse(truth, prediction),
                "gain_vs_base": base_rmse - _rmse(truth, prediction),
                "analytic_weight": analytic_weight,
                "mean_well_gain": float(np.mean(well_gains)),
                "median_well_gain": float(np.median(well_gains)),
                "positive_well_fraction": float(np.mean(np.asarray(well_gains) > 0)),
            }
            fold_gains = []
            if fold_indices:
                for fold, indices in fold_indices.items():
                    fold_base = _rmse(truth[indices], base[indices])
                    fold_prediction = _rmse(truth[indices], prediction[indices])
                    gain = fold_base - fold_prediction
                    fold_gains.append(gain)
                    blend_fold_rows.append(
                        {
                            "base": base_name,
                            "model": model_name,
                            "weight_route": weight,
                            "fold": int(fold),
                            "gain_vs_base": gain,
                            "base_rmse": fold_base,
                            "rmse": fold_prediction,
                        }
                    )
            row["min_fold_gain"] = float(np.min(fold_gains)) if fold_gains else np.nan
            row["positive_folds"] = int(np.sum(np.asarray(fold_gains) > 0)) if fold_gains else 0
            blend_rows.append(row)

blend_scores = pd.DataFrame(blend_rows)
blend_fold_scores = pd.DataFrame(blend_fold_rows)
if len(blend_scores):
    blend_scores = blend_scores.sort_values("rmse").reset_index(drop=True)

scores.to_csv(OUT_DIR / "standalone_scores.csv", index=False)
fold_scores.to_csv(OUT_DIR / "standalone_folds.csv", index=False)
bywell.to_csv(OUT_DIR / "standalone_bywell.csv", index=False)
diagnostics.to_csv(OUT_DIR / "diagnostics.csv", index=False)
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
blend_scores.to_csv(OUT_DIR / "blend_scores.csv", index=False)
blend_fold_scores.to_csv(OUT_DIR / "blend_folds.csv", index=False)

# The per-well cache is sufficient to reconstruct full paths. Avoid another
# large monolithic OOF artifact while disk space is tight.
best_standalone = scores.iloc[0].to_dict()
best_noncarry = scores[scores["model"] != "carry"].iloc[0].to_dict()
best_blend = blend_scores.iloc[0].to_dict() if len(blend_scores) else None
standalone_family_signal = bool(best_noncarry["rmse"] <= 12.0)
blend_signal = bool(
    best_blend is not None
    and best_blend["gain_vs_base"] >= 0.10
    and best_blend["positive_well_fraction"] >= 0.55
    and (np.isnan(best_blend["min_fold_gain"]) or best_blend["min_fold_gain"] >= 0.0)
)
strict_pass = bool(standalone_family_signal and blend_signal)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


verdict = _jsonable(
    {
        "experiment": "tvt_drift_zero_learning_dp_v0",
        "data_root": str(DATA_ROOT),
        "pool": str(pool_path) if pool_path else None,
        "fold_source": str(fold_source) if fold_source else None,
        "decoded_wells": decoded["well"].nunique(),
        "failed_wells": len(failures),
        "edge_rerun_wells": int(diagnostics["edge_rerun"].sum()),
        "carry_rmse": _rmse(truth, decoded["carry"]),
        "best_standalone": best_standalone,
        "best_noncarry": best_noncarry,
        "best_blend": best_blend,
        "standalone_family_signal": standalone_family_signal,
        "blend_signal": blend_signal,
        "strict_pass": strict_pass,
        "next": (
            "PASS -> add an explicit dip dimension / patch NCC before any NN"
            if strict_pass
            else "FAIL -> compact TVT axis does not rescue zero-learning GR alignment"
        ),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nSTANDALONE")
print(scores.to_string(index=False))
if len(fold_scores):
    print("\nSTANDALONE FOLDS")
    print(fold_scores.to_string(index=False))
if len(blend_scores):
    print("\nBEST BLENDS")
    print(blend_scores.head(30).to_string(index=False))
print("\nDIAGNOSTICS")
diagnostic_columns = [
    "half_width_ft",
    "edge_rerun",
    "prefix_slope",
    "confidence_mean",
    "sharpness_mean",
    "calibration_slope",
    "calibration_residual_scale",
]
print(diagnostics[diagnostic_columns].describe(include="all").to_string())
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
