"""Single-reference GR/TVT full-grid + global DP kill-test.

This is standalone. It uses only horizontal-well GR/Z/TVT_input, the assigned
typewell TVT/GR curve, and the official anchor. TVT truth enters only after all
paths have been decoded, through the honest multi-cut v0 OOF file.

The decoder uses the assigned typewell's complete TVT range. It starts exactly
at the visible TVT anchor. Each transition is centred on the geometry increment
``-delta_Z`` but may accumulate a bounded residual at every sampled step. This
allows structural drift to grow beyond a fixed correction band while preserving
anchor-reset physics. Three fixed prior strengths are evaluated; no model is
trained in this cell.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BASE_DIR = OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
BASE_OOF = BASE_DIR / "multicut_v0_oof.pkl"

CACHE_VERSION = "v1_fullgrid_step2_stride4_multiscale"
CACHE_DIR = OOF_ROOT / f"global_dp_costvolume_{CACHE_VERSION}"
WELL_CACHE = CACHE_DIR / "wells"

STATE_STEP_FT = 2.0
ROW_STRIDE = 4
MAX_RESIDUAL_JUMP_STATES = 4
HORIZONTAL_SMOOTH_ROWS = (0.0, 8.0, 24.0)
TYPEWELL_SMOOTH_FT = (0.0, 8.0, 24.0)
SCALE_WEIGHTS = (0.50, 0.30, 0.20)

DP_CONFIGS = {
    "dp_loose": {"transition": 0.05},
    "dp_mid": {"transition": 0.20},
    "dp_strict": {"transition": 0.80},
}
BLEND_WEIGHTS = (0.10, 0.20, 0.30, 0.40)

for directory in (CACHE_DIR, WELL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)

if not BASE_OOF.exists():
    raise FileNotFoundError(f"Missing multi-cut OOF: {BASE_OOF}")

print(
    {
        "experiment": "global_dp_costvolume_killtest",
        "base_oof": str(BASE_OOF),
        "state_space": "full assigned-typewell TVT range",
        "state_step_ft": STATE_STEP_FT,
        "row_stride": ROW_STRIDE,
        "dp_configs": DP_CONFIGS,
    }
)


def _rmse(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _mad(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)) + 1e-6)


def _fill_1d(values):
    series = pd.Series(np.asarray(values, float))
    return series.interpolate(limit_direction="both").fillna(0.0).to_numpy(float)


def _prepare_typewell(typewell):
    tw = typewell[["TVT", "GR"]].copy()
    tw = tw[np.isfinite(tw["TVT"])].sort_values("TVT")
    tw = tw.groupby("TVT", as_index=False)["GR"].mean()
    tvt = tw["TVT"].to_numpy(float)
    gr = _fill_1d(tw["GR"].to_numpy(float))
    if len(tvt) < 32 or np.ptp(tvt) < 20:
        raise RuntimeError("typewell is too short")
    return tvt, gr


def _calibrate_horizontal_gr(horizontal_gr, known_tvt, tw_tvt, tw_gr):
    horizontal_gr = np.asarray(horizontal_gr, float)
    known_tvt = np.asarray(known_tvt, float)
    valid = np.isfinite(horizontal_gr) & np.isfinite(known_tvt)
    if int(valid.sum()) < 30:
        return _fill_1d(horizontal_gr), 1.0, 0.0

    x = horizontal_gr[valid]
    y = np.interp(known_tvt[valid], tw_tvt, tw_gr)
    x_q25, x_q50, x_q75 = np.quantile(x, [0.25, 0.50, 0.75])
    y_q25, y_q50, y_q75 = np.quantile(y, [0.25, 0.50, 0.75])
    slope = float(np.clip((y_q75 - y_q25) / max(x_q75 - x_q25, 1e-3), 0.25, 4.0))
    intercept = float(y_q50 - slope * x_q50)
    calibrated = slope * _fill_1d(horizontal_gr) + intercept
    return calibrated, slope, intercept


def _smooth(values, sigma):
    values = np.asarray(values, float)
    if sigma <= 0:
        return values.copy()
    return gaussian_filter1d(values, sigma=float(sigma), mode="nearest")


def _build_cost_volume(hw, tw, sample_positions, candidate_tvt):
    tw_tvt, tw_gr = _prepare_typewell(tw)
    gr_raw = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    calibrated_gr, calibration_slope, calibration_intercept = _calibrate_horizontal_gr(
        gr_raw, tvt_input, tw_tvt, tw_gr
    )

    raw_valid = np.isfinite(gr_raw).astype(float)
    tw_step = float(np.median(np.diff(tw_tvt)))
    tw_step = max(abs(tw_step), 1e-3)
    gr_scale = max(_mad(tw_gr), 5.0)

    emission = np.zeros(candidate_tvt.shape, dtype=np.float32)
    confidence_total = np.zeros(len(sample_positions), dtype=np.float32)

    for h_sigma, tw_sigma_ft, scale_weight in zip(
        HORIZONTAL_SMOOTH_ROWS, TYPEWELL_SMOOTH_FT, SCALE_WEIGHTS
    ):
        h_smoothed = _smooth(calibrated_gr, h_sigma)
        tw_smoothed = _smooth(tw_gr, tw_sigma_ft / tw_step)
        if h_sigma <= 0:
            confidence = raw_valid[sample_positions]
        else:
            confidence = np.clip(
                _smooth(raw_valid, h_sigma)[sample_positions], 0.0, 1.0
            )
        expected = np.interp(
            candidate_tvt.reshape(-1), tw_tvt, tw_smoothed
        ).reshape(candidate_tvt.shape)
        difference = (h_smoothed[sample_positions, None] - expected) / gr_scale
        robust_square = np.minimum(difference * difference, 9.0)
        emission += (
            float(scale_weight) * confidence[:, None] * robust_square
        ).astype(np.float32)
        confidence_total += (float(scale_weight) * confidence).astype(np.float32)

    outside = (candidate_tvt < tw_tvt[0]) | (candidate_tvt > tw_tvt[-1])
    emission += outside.astype(np.float32) * 4.0
    return emission, confidence_total, calibration_slope, calibration_intercept


@njit
def _viterbi_decode(
    emission,
    state_values,
    expected_changes,
    transition_penalty,
    max_residual_jump,
    anchor_state,
):
    n_rows, n_states = emission.shape
    large = np.float32(1e30)
    previous = np.full(n_states, large, dtype=np.float32)
    previous[anchor_state] = np.float32(0.0)
    backpointer = np.empty((n_rows, n_states), dtype=np.int16)

    for row in range(n_rows):
        current = np.full(n_states, large, dtype=np.float32)
        expected_change = expected_changes[row]
        expected_state_shift = int(np.round(expected_change / STATE_STEP_FT))
        for state in range(n_states):
            expected_old_state = state - expected_state_shift
            start = max(0, expected_old_state - max_residual_jump)
            stop = min(n_states, expected_old_state + max_residual_jump + 1)
            best_cost = large
            best_previous = state
            for old_state in range(start, stop):
                actual_change = state_values[state] - state_values[old_state]
                residual_change = actual_change - expected_change
                cost = (
                    previous[old_state]
                    + transition_penalty * residual_change * residual_change
                )
                if cost < best_cost:
                    best_cost = cost
                    best_previous = old_state
            current[state] = best_cost + emission[row, state]
            backpointer[row, state] = best_previous
        previous = current

    end_state = int(np.argmin(previous))
    states = np.empty(n_rows, dtype=np.int16)
    state = end_state
    for row in range(n_rows - 1, -1, -1):
        states[row] = state
        state = int(backpointer[row, state])
    return states, float(previous[end_state])


def _decode_well(wid):
    cache_path = WELL_CACHE / f"{wid}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path), None

    horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
    try:
        hw = pd.read_csv(horizontal_path)
        tw = pd.read_csv(typewell_path)
        tvt_input = hw["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        evaluation = np.flatnonzero(~np.isfinite(tvt_input))
        if len(known) < 30 or len(evaluation) < 64:
            raise RuntimeError("insufficient known/eval rows")
        anchor_position = int(known[-1])
        if int(evaluation[0]) != anchor_position + 1:
            raise RuntimeError("TVT_input is not a contiguous prefix")

        sampled = evaluation[::ROW_STRIDE]
        if sampled[-1] != evaluation[-1]:
            sampled = np.append(sampled, evaluation[-1])

        z = hw["Z"].to_numpy(float)
        anchor_tvt = float(tvt_input[anchor_position])
        anchor_z = float(z[anchor_position])
        tw_tvt, _ = _prepare_typewell(tw)
        lower_step = int(np.floor((float(tw_tvt[0]) - anchor_tvt) / STATE_STEP_FT))
        upper_step = int(np.ceil((float(tw_tvt[-1]) - anchor_tvt) / STATE_STEP_FT))
        state_values = (
            anchor_tvt
            + np.arange(lower_step, upper_step + 1, dtype=np.float32) * STATE_STEP_FT
        ).astype(np.float32)
        anchor_state = int(-lower_step)
        if not (0 <= anchor_state < len(state_values)):
            raise RuntimeError("anchor is outside typewell TVT range")
        candidates = np.broadcast_to(
            state_values[None, :], (len(sampled), len(state_values))
        )
        emission, confidence, cal_slope, cal_intercept = _build_cost_volume(
            hw, tw, sampled, candidates
        )

        previous_positions = np.concatenate(([anchor_position], sampled[:-1]))
        expected_changes = (-(z[sampled] - z[previous_positions])).astype(np.float32)
        geometry_center = anchor_tvt - (z[evaluation] - anchor_z)

        frame = pd.DataFrame(
            {
                "id": [f"{wid}_{position}" for position in evaluation],
                "well": wid,
                "row_idx": evaluation.astype(int),
                "geometry_center": geometry_center,
                "typewell_tvt_min": float(tw_tvt[0]),
                "typewell_tvt_max": float(tw_tvt[-1]),
            }
        )

        x_anchor = np.concatenate(([anchor_position], sampled)).astype(float)
        for name, config in DP_CONFIGS.items():
            states, path_cost = _viterbi_decode(
                emission,
                state_values,
                expected_changes,
                np.float32(config["transition"]),
                MAX_RESIDUAL_JUMP_STATES,
                anchor_state,
            )
            path_sampled = state_values[states].astype(float)
            path = np.interp(
                evaluation.astype(float),
                x_anchor,
                np.concatenate(([anchor_tvt], path_sampled)),
            )
            window = min(61, len(path))
            if window % 2 == 0:
                window -= 1
            if window >= 5:
                path = savgol_filter(path, window, 3)
            frame[name] = path
            frame[f"{name}_mean_abs_correction"] = float(
                np.mean(np.abs(path - geometry_center))
            )
            frame[f"{name}_path_cost_per_sample"] = path_cost / max(1, len(sampled))

        frame["mean_gr_confidence"] = float(np.mean(confidence))
        frame["calibration_slope"] = cal_slope
        frame["calibration_intercept"] = cal_intercept
        frame.to_pickle(cache_path)
        return frame, None
    except Exception as exc:
        return None, {"well": wid, "reason": repr(exc)}


# ---------------------------------------------------------------------------
# Decode every well represented by the honest multi-cut OOF.
# ---------------------------------------------------------------------------
base = pd.read_pickle(BASE_OOF)
required_base = {
    "id",
    "well",
    "fold",
    "tvt_true",
    "pred_single_cut",
    "pred_multi_cut",
}
missing = sorted(required_base - set(base.columns))
if missing:
    raise RuntimeError(f"Missing base OOF columns: {missing}")
base["well"] = base["well"].astype(str)
base["id"] = base["id"].astype(str)
wells = sorted(base["well"].unique())

started = time.time()
decoded_parts = []
failures = []
for number, wid in enumerate(tqdm(wells, desc="global DP wells"), 1):
    frame, failure = _decode_well(wid)
    if frame is not None:
        decoded_parts.append(frame)
    if failure is not None:
        failures.append(failure)
    if number % 50 == 0:
        print(
            f"DP {number}/{len(wells)} ok={len(decoded_parts)} "
            f"elapsed={(time.time()-started)/60:.1f} min",
            flush=True,
        )

if not decoded_parts:
    raise RuntimeError("No DP paths decoded")

decoded = pd.concat(decoded_parts, ignore_index=True)
merged = base.merge(
    decoded,
    on=["id", "well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
print("merged rows/wells:", merged.shape, merged["well"].nunique())
if len(merged) < 0.98 * len(base):
    warnings.warn(f"Only matched {len(merged)} / {len(base)} OOF rows")


# ---------------------------------------------------------------------------
# Absolute, fold, distance/top-error, cosine, and fixed blend diagnostics.
# ---------------------------------------------------------------------------
y = merged["tvt_true"].to_numpy(float)
base_prediction = merged["pred_multi_cut"].to_numpy(float)
base_rmse = _rmse(y, base_prediction)
base_error = base_prediction - y

summary_rows = []
fold_rows = []
prediction_columns = list(DP_CONFIGS)

for model_name in prediction_columns:
    prediction = merged[model_name].to_numpy(float)
    model_rmse = _rmse(y, prediction)
    model_error = prediction - y
    cosine = float(
        np.dot(base_error, model_error)
        / max(np.linalg.norm(base_error) * np.linalg.norm(model_error), 1e-12)
    )
    summary_rows.append(
        {
            "model": model_name,
            "blend_weight": 1.0,
            "base_rmse": base_rmse,
            "rmse": model_rmse,
            "gain": base_rmse - model_rmse,
            "error_cosine": cosine,
            "standalone_ratio": model_rmse / base_rmse,
        }
    )
    for blend_weight in BLEND_WEIGHTS:
        blend = (1.0 - blend_weight) * base_prediction + blend_weight * prediction
        summary_rows.append(
            {
                "model": model_name,
                "blend_weight": blend_weight,
                "base_rmse": base_rmse,
                "rmse": _rmse(y, blend),
                "gain": base_rmse - _rmse(y, blend),
                "error_cosine": cosine,
                "standalone_ratio": model_rmse / base_rmse,
            }
        )

for row in summary_rows:
    model_name = row["model"]
    blend_weight = float(row["blend_weight"])
    raw_prediction = merged[model_name].to_numpy(float)
    prediction = (
        raw_prediction
        if blend_weight == 1.0
        else (1.0 - blend_weight) * base_prediction + blend_weight * raw_prediction
    )
    gains = []
    for fold, index in merged.groupby("fold", sort=True).groups.items():
        position = merged.index.get_indexer(index)
        fold_base = _rmse(y[position], base_prediction[position])
        fold_score = _rmse(y[position], prediction[position])
        gain = fold_base - fold_score
        gains.append(gain)
        fold_rows.append(
            {
                "model": model_name,
                "blend_weight": blend_weight,
                "fold": int(fold),
                "rows": len(position),
                "base_rmse": fold_base,
                "rmse": fold_score,
                "gain": gain,
            }
        )
    row["mean_fold_gain"] = float(np.mean(gains))
    row["min_fold_gain"] = float(np.min(gains))
    row["positive_folds"] = int(np.sum(np.asarray(gains) > 0))

summary = pd.DataFrame(summary_rows)
fold_summary = pd.DataFrame(fold_rows)

# Top40 is frozen by base SSE, not by DP performance.
bywell_base = []
for wid, group in merged.groupby("well", sort=False):
    truth = group["tvt_true"].to_numpy(float)
    pred = group["pred_multi_cut"].to_numpy(float)
    bywell_base.append(
        {
            "well": wid,
            "rows": len(group),
            "base_sse": float(np.sum((pred - truth) ** 2)),
        }
    )
bywell_base = pd.DataFrame(bywell_base)
top40_wells = set(bywell_base.nlargest(40, "base_sse")["well"].astype(str))
top40_mask = merged["well"].isin(top40_wells).to_numpy()
top40_base = _rmse(y[top40_mask], base_prediction[top40_mask])

top40_gains = []
for row in summary.itertuples(index=False):
    raw = merged[row.model].to_numpy(float)
    pred = raw if row.blend_weight == 1.0 else (
        (1.0 - row.blend_weight) * base_prediction + row.blend_weight * raw
    )
    top40_score = _rmse(y[top40_mask], pred[top40_mask])
    top40_gains.append(top40_base - top40_score)
summary["top40_gain"] = top40_gains

blend_rows = summary[summary["blend_weight"] < 1.0].copy()
blend_rows["gate"] = (
    (blend_rows["standalone_ratio"] <= 1.25)
    & (blend_rows["error_cosine"] <= 0.90)
    & (blend_rows["gain"] >= 0.05)
    & (blend_rows["positive_folds"] >= 4)
    & (blend_rows["min_fold_gain"] > -0.10)
    & (blend_rows["top40_gain"] >= 0.0)
)
summary = summary.merge(
    blend_rows[["model", "blend_weight", "gate"]],
    on=["model", "blend_weight"],
    how="left",
)
summary["gate"] = summary["gate"].fillna(False)
summary = summary.sort_values(["gate", "gain"], ascending=[False, False])

# State-space coverage is diagnostic only and is computed after decoding.
state_coverage = float(
    np.mean(
        (merged["tvt_true"].to_numpy(float) >= merged["typewell_tvt_min"].to_numpy(float))
        & (merged["tvt_true"].to_numpy(float) <= merged["typewell_tvt_max"].to_numpy(float))
    )
)

strict_pass = bool(summary["gate"].any() and state_coverage >= 0.99)

print("\nSUMMARY")
display(summary)
print("\nFOLDS")
display(fold_summary)
print(
    "\nKILL-TEST",
    {
        "base_rmse": base_rmse,
        "state_coverage": state_coverage,
        "decoded_wells": merged["well"].nunique(),
        "failed_wells": len(failures),
        "strict_pass": strict_pass,
        "verdict": "PASS -> train TCN decoder" if strict_pass else "FAIL / STOP DP-TCN",
    },
)

summary.to_csv(CACHE_DIR / "global_dp_summary.csv", index=False)
fold_summary.to_csv(CACHE_DIR / "global_dp_folds.csv", index=False)
merged.to_pickle(CACHE_DIR / "global_dp_oof.pkl")
pd.DataFrame(failures).to_csv(CACHE_DIR / "global_dp_failures.csv", index=False)
pd.DataFrame(
    [
        {
            "base_rmse": base_rmse,
            "state_coverage": state_coverage,
            "strict_pass": strict_pass,
            "decoded_wells": merged["well"].nunique(),
            "failed_wells": len(failures),
        }
    ]
).to_csv(CACHE_DIR / "global_dp_verdict.csv", index=False)
print("saved:", CACHE_DIR)
