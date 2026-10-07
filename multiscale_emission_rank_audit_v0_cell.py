"""Full-data trajectory-dynamics and multiscale emission-rank audit.

This is the pre-DP gate. Around the strong CP12 path, it asks whether fixed,
target-free pointwise and forward-window GR emissions rank the true TVT state
inside a +/-96 ft band. No path decoder is trained or tuned here.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "multiscale_emission_rank_audit_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OFFSETS = np.arange(-96.0, 96.1, 4.0, dtype=np.float64)
WINDOWS_FT = (64.0, 128.0, 256.0, 512.0, 800.0)
WINDOW_WEIGHTS = np.asarray((0.10, 0.15, 0.20, 0.25, 0.30), np.float64)
ROW_STRIDE = 16
DEEP_MD_SINCE = 1500.0
N_JOBS = 4

if not POOL_PATH.exists():
    raise FileNotFoundError(POOL_PATH)


def rmse_from_arrays(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def robust_affine(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 20 or np.std(x) < 1e-6:
        return 0.0, 1.0
    keep = np.ones(len(x), dtype=bool)
    intercept, slope = 0.0, 1.0
    for _ in range(5):
        design = np.column_stack([np.ones(keep.sum()), x[keep]])
        intercept, slope = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (intercept + slope * x)
        median = np.median(residual[keep])
        mad = 1.4826 * np.median(np.abs(residual[keep] - median))
        if not np.isfinite(mad) or mad < 1e-6:
            break
        keep = np.abs(residual - median) <= 3.5 * mad
        if keep.sum() < 20:
            break
    return float(intercept), float(np.clip(slope, 0.20, 4.0))


def forward_ncc_multiscale(md, observed, expected, windows_ft):
    md = np.asarray(md, np.float64)
    observed = np.asarray(observed, np.float64)
    expected = np.asarray(expected, np.float64)
    n_rows = len(md)
    valid = np.isfinite(observed) & np.isfinite(expected)
    x = np.where(valid, observed, 0.0)
    y = np.where(valid, expected, 0.0)
    arrays = (
        valid.astype(np.float64),
        x,
        y,
        x * x,
        y * y,
        x * y,
    )
    cumulative = [np.concatenate([[0.0], np.cumsum(values)]) for values in arrays]
    start = np.arange(n_rows, dtype=np.int64)
    output = {}
    for window_ft in windows_ft:
        end = np.searchsorted(md, md + float(window_ft), side="right")
        end = np.maximum(end, start + 1)
        sums = [values[end] - values[start] for values in cumulative]
        count, sx, sy, sxx, syy, sxy = sums
        safe_count = np.maximum(count, 1.0)
        covariance = sxy - sx * sy / safe_count
        var_x = np.maximum(sxx - sx * sx / safe_count, 0.0)
        var_y = np.maximum(syy - sy * sy / safe_count, 0.0)
        denominator = np.sqrt(var_x * var_y)
        enough = count >= np.maximum(12.0, 0.30 * (end - start))
        good = enough & (denominator > 1e-8)
        correlation = np.zeros(n_rows, np.float64)
        correlation[good] = covariance[good] / denominator[good]
        output[window_ft] = (np.clip(correlation, -1.0, 1.0), good)
    return output


def score_rank_metrics(scores, valid, truth_offset, base, truth):
    scores = np.asarray(scores, np.float64)
    valid = np.asarray(valid, bool)
    truth_offset = np.asarray(truth_offset, np.float64)
    target_index = np.argmin(np.abs(OFFSETS[:, None] - truth_offset[None, :]), axis=0)
    columns = np.arange(scores.shape[1])
    target_score = scores[target_index, columns]
    rank = 1 + np.sum(scores > target_score[None, :] + 1e-12, axis=0)
    best_index = np.argmax(scores, axis=0)
    selected = base + OFFSETS[best_index]
    return rank, selected, valid


print("loading CP12 pool", flush=True)
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = ["cp12"]
if "well" in header and "row_idx" in header:
    usecols += ["well", "row_idx"]
    pool = pd.read_csv(POOL_PATH, usecols=usecols)
    pool["well"] = pool["well"].astype(str)
    pool["row_idx"] = pool["row_idx"].astype(np.int32)
else:
    usecols += ["id"]
    pool = pd.read_csv(POOL_PATH, usecols=usecols)
    pool["id"] = pool["id"].astype(str)
    split = pool["id"].str.rsplit("_", n=1, expand=True)
    pool["well"] = split[0].astype(str)
    pool["row_idx"] = split[1].astype(np.int32)

cp12_by_well = {}
for wid, group in pool.groupby("well", sort=False):
    order = np.argsort(group["row_idx"].to_numpy())
    cp12_by_well[str(wid)] = (
        group["row_idx"].to_numpy(np.int32)[order],
        group["cp12"].to_numpy(np.float64)[order],
    )
del pool

train_paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
wells = [path.name.replace("__horizontal_well.csv", "") for path in train_paths]
wells = [wid for wid in wells if wid in cp12_by_well]
print("eligible wells:", len(wells))


def aligned_cp12(wid, eval_pos):
    positions, values = cp12_by_well[wid]
    locations = np.searchsorted(positions, eval_pos)
    if (
        np.any(locations >= len(positions))
        or not np.array_equal(positions[locations], eval_pos)
    ):
        raise RuntimeError(f"CP12 row mismatch for {wid}")
    return values[locations]


# Fix top40 from full official-eval CP12 SSE before any emission diagnostics.
well_sse_rows = []
for wid in tqdm(wells, desc="CP12 top40 audit"):
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
        usecols=["TVT", "TVT_input"],
    )
    eval_pos = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    truth = horizontal["TVT"].to_numpy(np.float64)[eval_pos]
    base = aligned_cp12(wid, eval_pos)
    mask = np.isfinite(truth) & np.isfinite(base)
    well_sse_rows.append(
        {
            "well": wid,
            "rows": int(mask.sum()),
            "cp12_sse": float(np.sum((truth[mask] - base[mask]) ** 2)),
        }
    )
well_sse = pd.DataFrame(well_sse_rows).sort_values("cp12_sse", ascending=False)
top40 = set(well_sse.head(40)["well"].astype(str))


def process_well(wid):
    try:
        horizontal = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        typewell = pd.read_csv(DATA_ROOT / "train" / f"{wid}__typewell.csv").sort_values("TVT")
        tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
        true_tvt = horizontal["TVT"].to_numpy(np.float64)
        md_all = horizontal["MD"].to_numpy(np.float64)
        gr_all = horizontal["GR"].to_numpy(np.float64)
        known_pos = np.flatnonzero(np.isfinite(tvt_input))
        eval_pos = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(true_tvt))
        if len(known_pos) < 64 or len(eval_pos) < 128:
            raise RuntimeError("too few known/eval rows")
        anchor = int(known_pos[-1])

        base = aligned_cp12(wid, eval_pos)
        truth = true_tvt[eval_pos]
        md = md_all[eval_pos]
        observed = gr_all[eval_pos]
        md_since = md - md_all[anchor]

        tw_tvt = typewell["TVT"].to_numpy(np.float64)
        tw_gr_series = typewell["GR"].interpolate(limit_direction="both")
        tw_gr = tw_gr_series.fillna(float(tw_gr_series.mean())).to_numpy(np.float64)
        if len(tw_gr) >= 7:
            tw_gr = gaussian_filter1d(tw_gr, sigma=1.0, mode="nearest")

        prefix_valid = np.isfinite(tvt_input[: anchor + 1]) & np.isfinite(gr_all[: anchor + 1])
        type_prefix = np.interp(
            tvt_input[: anchor + 1][prefix_valid], tw_tvt, tw_gr
        )
        intercept, slope = robust_affine(type_prefix, gr_all[: anchor + 1][prefix_valid])
        tw_gr_calibrated = intercept + slope * tw_gr
        prefix_expected = intercept + slope * type_prefix
        prefix_residual = gr_all[: anchor + 1][prefix_valid] - prefix_expected
        residual_scale = float(
            max(1.4826 * np.median(np.abs(prefix_residual - np.median(prefix_residual))), 8.0)
        )

        # Truth dynamics and prefix-trend deviation.
        tail_pos = known_pos[-min(300, len(known_pos)):]
        x_tail = md_all[tail_pos] - md_all[anchor]
        y_tail = tvt_input[tail_pos] - tvt_input[anchor]
        trend_slope = float(np.polyfit(x_tail, y_tail, 1)[0]) if len(tail_pos) >= 3 else 0.0
        trend = tvt_input[anchor] + trend_slope * md_since
        trend_residual = truth - trend
        delta_md = np.diff(md)
        delta_tvt = np.diff(truth)
        valid_step = np.isfinite(delta_md) & np.isfinite(delta_tvt) & (delta_md > 0)
        dip = delta_tvt[valid_step] / delta_md[valid_step]
        dip_change_100 = np.array([], np.float64)
        if len(dip) >= 2:
            mid_md = 0.5 * (md[1:] + md[:-1])[valid_step]
            change_md = np.diff(mid_md)
            ok = np.isfinite(change_md) & (change_md > 0)
            dip_change_100 = np.diff(dip)[ok] / change_md[ok] * 100.0

        sample_local = np.arange(0, len(eval_pos), ROW_STRIDE, dtype=np.int64)
        if sample_local[-1] != len(eval_pos) - 1:
            sample_local = np.append(sample_local, len(eval_pos) - 1)
        sample_base = base[sample_local]
        sample_truth = truth[sample_local]
        sample_truth_offset = sample_truth - sample_base

        point_scores = np.empty((len(OFFSETS), len(eval_pos)), np.float64)
        point_valid = np.isfinite(observed)
        window_scores = {
            window: np.empty((len(OFFSETS), len(eval_pos)), np.float64)
            for window in WINDOWS_FT
        }
        window_valid = {
            window: np.zeros((len(OFFSETS), len(eval_pos)), bool)
            for window in WINDOWS_FT
        }
        observed_diff = np.diff(observed, prepend=np.nan)

        for offset_index, offset in enumerate(OFFSETS):
            expected = np.interp(
                base + offset,
                tw_tvt,
                tw_gr_calibrated,
                left=tw_gr_calibrated[0],
                right=tw_gr_calibrated[-1],
            )
            standardized = (observed - expected) / residual_scale
            point_scores[offset_index] = -np.minimum(standardized * standardized, 25.0)
            expected_diff = np.diff(expected, prepend=np.nan)
            intensity_by_window = forward_ncc_multiscale(
                md, observed, expected, WINDOWS_FT
            )
            derivative_by_window = forward_ncc_multiscale(
                md, observed_diff, expected_diff, WINDOWS_FT
            )
            for window in WINDOWS_FT:
                intensity, valid_intensity = intensity_by_window[window]
                derivative, valid_derivative = derivative_by_window[window]
                use_derivative = valid_intensity & valid_derivative
                score = intensity.copy()
                score[use_derivative] = (
                    0.80 * intensity[use_derivative]
                    + 0.20 * derivative[use_derivative]
                )
                window_scores[window][offset_index] = score
                window_valid[window][offset_index] = valid_intensity

        output = pd.DataFrame(
            {
                "well": wid,
                "row_idx": eval_pos[sample_local],
                "md_since": md_since[sample_local],
                "top40": wid in top40,
                "truth": sample_truth,
                "cp12": sample_base,
                "truth_offset": sample_truth_offset,
                "band": np.abs(sample_truth_offset) <= np.max(np.abs(OFFSETS)),
                "trend_residual": trend_residual[sample_local],
            }
        )

        rank, selected, valid = score_rank_metrics(
            point_scores[:, sample_local],
            point_valid[sample_local],
            sample_truth_offset,
            sample_base,
            sample_truth,
        )
        output["point_valid"] = valid
        output["point_rank"] = rank
        output["point_selected"] = selected

        combined_numerator = np.zeros((len(OFFSETS), len(sample_local)), np.float64)
        combined_denominator = np.zeros((len(OFFSETS), len(sample_local)), np.float64)
        for weight, window in zip(WINDOW_WEIGHTS, WINDOWS_FT):
            scores = window_scores[window][:, sample_local]
            valid_matrix = window_valid[window][:, sample_local]
            combined_numerator += weight * scores * valid_matrix
            combined_denominator += weight * valid_matrix
            # Validity is effectively state-independent here, but keep the
            # exact matrix and require the target state to be valid.
            target_index = np.argmin(
                np.abs(OFFSETS[:, None] - sample_truth_offset[None, :]), axis=0
            )
            local_valid = valid_matrix[target_index, np.arange(len(sample_local))]
            rank, selected, _ = score_rank_metrics(
                scores, local_valid, sample_truth_offset, sample_base, sample_truth
            )
            label = f"w{int(window)}"
            output[f"{label}_valid"] = local_valid
            output[f"{label}_rank"] = rank
            output[f"{label}_selected"] = selected

        combined = combined_numerator / np.maximum(combined_denominator, 1e-12)
        target_index = np.argmin(
            np.abs(OFFSETS[:, None] - sample_truth_offset[None, :]), axis=0
        )
        combined_valid = (
            combined_denominator[target_index, np.arange(len(sample_local))] > 0
        )
        rank, selected, _ = score_rank_metrics(
            combined, combined_valid, sample_truth_offset, sample_base, sample_truth
        )
        output["combined_valid"] = combined_valid
        output["combined_rank"] = rank
        output["combined_selected"] = selected

        dynamics = {
            "well": wid,
            "eval_rows": len(eval_pos),
            "trend_slope": trend_slope,
            "trend_abs_p50": float(np.median(np.abs(trend_residual))),
            "trend_abs_p95": float(np.quantile(np.abs(trend_residual), 0.95)),
            "trend_abs_max": float(np.max(np.abs(trend_residual))),
            "dip_p01": float(np.quantile(dip, 0.01)) if len(dip) else np.nan,
            "dip_p50": float(np.quantile(dip, 0.50)) if len(dip) else np.nan,
            "dip_p99": float(np.quantile(dip, 0.99)) if len(dip) else np.nan,
            "dip_change100_abs_p95": float(np.quantile(np.abs(dip_change_100), 0.95)) if len(dip_change_100) else np.nan,
            "cp12_rmse": rmse_from_arrays(truth, base),
        }
        return {"ok": True, "rows": output, "dynamics": dynamics}
    except Exception as exc:
        return {"ok": False, "well": wid, "reason": repr(exc)}


started = time.time()
results = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(process_well)(wid)
    for wid in tqdm(wells, desc="multiscale emission audit")
)
failures = pd.DataFrame([result for result in results if not result["ok"]])
if len(failures):
    display(failures.head(30))
    failures.to_csv(OUT_DIR / "failures.csv", index=False)
    raise RuntimeError(f"emission audit failed for {len(failures)} wells")

rows = pd.concat([result["rows"] for result in results], ignore_index=True)
dynamics = pd.DataFrame([result["dynamics"] for result in results])
print(
    "audit rows:",
    {"rows": len(rows), "wells": rows["well"].nunique(), "elapsed_min": (time.time() - started) / 60.0},
)


def summarize_method(frame, subset, method):
    valid = frame[f"{method}_valid"].to_numpy(bool)
    band = frame["band"].to_numpy(bool)
    eligible = valid & band
    rank = frame[f"{method}_rank"].to_numpy(float)
    truth = frame["truth"].to_numpy(float)
    selected = frame[f"{method}_selected"].to_numpy(float)
    cp12 = frame["cp12"].to_numpy(float)
    result = {
        "subset": subset,
        "method": method,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "band_coverage": float(np.mean(band)),
        "valid_fraction": float(np.mean(valid)),
        "effective_top5_all": float(np.mean(eligible & (rank <= 5))),
        "top5_given_band_valid": float(np.mean(rank[eligible] <= 5)) if eligible.any() else 0.0,
        "top1_given_band_valid": float(np.mean(rank[eligible] == 1)) if eligible.any() else 0.0,
        "median_rank_given_band": float(np.median(rank[eligible])) if eligible.any() else np.nan,
        "cp12_rmse": rmse_from_arrays(truth, cp12),
        "selected_rmse_valid": rmse_from_arrays(truth[valid], selected[valid]) if valid.any() else np.nan,
    }
    result["selected_gain_vs_cp12_valid"] = (
        rmse_from_arrays(truth[valid], cp12[valid]) - result["selected_rmse_valid"]
        if valid.any()
        else np.nan
    )
    return result


methods = ["point"] + [f"w{int(window)}" for window in WINDOWS_FT] + ["combined"]
summary_rows = []
for subset, frame in (
    ("all", rows),
    ("deep_gt1500", rows[rows["md_since"] > DEEP_MD_SINCE]),
    ("top40", rows[rows["top40"]]),
    ("deep_top40", rows[(rows["top40"]) & (rows["md_since"] > DEEP_MD_SINCE)]),
):
    for method in methods:
        summary_rows.append(summarize_method(frame, subset, method))
summary = pd.DataFrame(summary_rows)

dynamics_summary = pd.DataFrame(
    [
        {
            "metric": column,
            "count": int(dynamics[column].notna().sum()),
            "mean": float(dynamics[column].mean()),
            "p50": float(dynamics[column].median()),
            "p90": float(dynamics[column].quantile(0.90)),
            "p95": float(dynamics[column].quantile(0.95)),
            "p99": float(dynamics[column].quantile(0.99)),
            "max": float(dynamics[column].max()),
        }
        for column in (
            "trend_abs_p50",
            "trend_abs_p95",
            "trend_abs_max",
            "dip_p01",
            "dip_p50",
            "dip_p99",
            "dip_change100_abs_p95",
            "cp12_rmse",
        )
    ]
)

deep_combined = summary[
    (summary["subset"] == "deep_gt1500") & (summary["method"] == "combined")
].iloc[0]
deep_point = summary[
    (summary["subset"] == "deep_gt1500") & (summary["method"] == "point")
].iloc[0]
top40_combined = summary[
    (summary["subset"] == "deep_top40") & (summary["method"] == "combined")
].iloc[0]
verdict = {
    "band_ft": 96,
    "deep_band_coverage": float(deep_combined["band_coverage"]),
    "deep_combined_effective_top5": float(deep_combined["effective_top5_all"]),
    "deep_combined_top5_given_band_valid": float(deep_combined["top5_given_band_valid"]),
    "deep_point_effective_top5": float(deep_point["effective_top5_all"]),
    "deep_top5_improvement_vs_point": float(
        deep_combined["effective_top5_all"] - deep_point["effective_top5_all"]
    ),
    "deep_top40_combined_effective_top5": float(top40_combined["effective_top5_all"]),
}
verdict["strict_pass"] = bool(
    verdict["deep_band_coverage"] >= 0.90
    and verdict["deep_combined_effective_top5"] >= 0.50
    and verdict["deep_top5_improvement_vs_point"] >= 0.15
    and verdict["deep_top40_combined_effective_top5"] >= 0.25
)
verdict["next"] = (
    "build coarse-to-fine second-order DP on a fixed 150-well sample"
    if verdict["strict_pass"]
    else "stop second-order DP engine; multiscale emission cannot rank truth reliably"
)

print("\nDYNAMICS SUMMARY")
display(dynamics_summary)
print("\nEMISSION SUMMARY")
display(summary)
print("\nPRE-DP VERDICT")
print(verdict)

dynamics.to_csv(OUT_DIR / "dynamics_bywell.csv", index=False)
dynamics_summary.to_csv(OUT_DIR / "dynamics_summary.csv", index=False)
summary.to_csv(OUT_DIR / "emission_summary.csv", index=False)
rows.to_pickle(OUT_DIR / "emission_sample_rows.pkl")
well_sse.to_csv(OUT_DIR / "cp12_well_sse.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
