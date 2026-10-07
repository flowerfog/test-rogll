"""Audit the missing same-well prefix emission proposed for RankSDF v1.

For each well, the visible prefix supplies pairs (TVT_input, horizontal GR).
Those pairs define a prefix-only TVT-to-GR reference curve. Around the exact
CP12 OOF path, this cell ranks candidate offsets using:

1. the existing typewell GR emission;
2. the same-well prefix reference emission; and
3. an equal-weight robust combination.

No network is trained. The purpose is to kill the input channel cheaply before
rebuilding the RankSDF panel cache.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "ranksdf_self_reference_emission_audit_v0"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]

OFFSETS = np.arange(-64.0, 64.0 + 0.1, 2.0, dtype=np.float32)
WINDOWS_FT = (64.0, 256.0, 512.0)
PREFIX_BIN_FT = 0.5
ROW_SAMPLE_STRIDE = 4
DEEP_FT = 1500.0
TOP_WELLS = 40

PASS_SELF_VALID_FRACTION = 0.50
PASS_TOP5_INCREMENT = 0.03
PASS_TOP40_INCREMENT = 0.01


def _rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    finite = np.isfinite(y) & np.isfinite(prediction)
    if not finite.any():
        return np.nan
    return float(np.sqrt(np.mean((prediction[finite] - y[finite]) ** 2)))


def _fill(values):
    return (
        pd.Series(np.asarray(values, np.float64))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _normalise(values, valid=None):
    values = np.asarray(values, np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, bool) & np.isfinite(values)
    output = np.zeros_like(values, np.float32)
    sample = values[valid]
    if len(sample) < 8:
        return output
    median = float(np.median(sample))
    scale = float(1.4826 * np.median(np.abs(sample - median)))
    if scale < 1e-3:
        scale = float(np.std(sample) + 1e-3)
    output[valid] = np.clip((values[valid] - median) / scale, -6.0, 6.0)
    return output


def _discover(wells, suffix):
    wells = set(map(str, wells))
    output = {}
    for root in (DATA_ROOT / "train", DATA_ROOT):
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            if not path.name.endswith(suffix):
                continue
            well = path.name[: -len(suffix)]
            if well in wells and "test" not in {part.lower() for part in path.parts}:
                output.setdefault(well, path)
    missing = sorted(wells - set(output))
    if missing:
        raise FileNotFoundError(f"missing {suffix}: {missing[:20]}")
    return output


def _prefix_reference(tvt_input, gr_raw):
    valid = np.isfinite(tvt_input) & np.isfinite(gr_raw)
    if valid.sum() < 32:
        return None
    tvt = np.asarray(tvt_input[valid], np.float64)
    gr = np.asarray(gr_raw[valid], np.float64)
    bins = np.round(tvt / PREFIX_BIN_FT) * PREFIX_BIN_FT
    frame = pd.DataFrame({"tvt": bins, "gr": gr})
    curve = frame.groupby("tvt", as_index=False)["gr"].median().sort_values("tvt")
    if len(curve) < 24 or float(curve["tvt"].max() - curve["tvt"].min()) < 16.0:
        return None
    return curve["tvt"].to_numpy(float), curve["gr"].to_numpy(float)


def _masked_window_cost(reference_grid, current_gr, valid, window_rows):
    squared = (reference_grid - current_gr[None, :]) ** 2
    valid_float = valid.astype(np.float32)
    numerator = uniform_filter1d(
        squared * valid_float, size=window_rows, axis=1, mode="nearest"
    )
    denominator = uniform_filter1d(
        valid_float, size=window_rows, axis=1, mode="nearest"
    )
    cost = numerator / np.maximum(denominator, 1e-3)
    cost[denominator < 0.50] = np.inf
    return cost.astype(np.float32)


def _robust_cost(cost):
    finite = np.isfinite(cost)
    safe = np.where(finite, cost, np.nan)
    median = np.nanmedian(safe, axis=0)
    q25 = np.nanpercentile(safe, 25, axis=0)
    q75 = np.nanpercentile(safe, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    output = (cost - median[None, :]) / scale[None, :]
    output[~finite] = np.inf
    return output.astype(np.float32)


def _score_rows(cost, truth_state, eligible):
    columns = np.arange(cost.shape[1])
    truth_cost = cost[truth_state, columns]
    valid = eligible & np.isfinite(truth_cost) & np.isfinite(cost).any(axis=0)
    rank = np.full(cost.shape[1], np.nan, np.float32)
    if valid.any():
        rank[valid] = 1.0 + np.sum(
            cost[:, valid] < truth_cost[valid][None, :], axis=0
        ).astype(np.float32)
    selected = np.argmin(cost, axis=0)
    selected[~np.isfinite(cost).any(axis=0)] = len(OFFSETS) // 2
    return rank, selected, valid


started = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
required = {"id", "tvt_true", "cp12"}
if not required.issubset(header):
    raise RuntimeError(f"pool missing {sorted(required - set(header))}")
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "cp12") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
wells = sorted(pool["well"].unique())
horizontal_files = _discover(wells, "__horizontal_well.csv")
typewell_files = _discover(wells, "__typewell.csv")

pool["se_cp12"] = (pool["cp12"].to_numpy(float) - pool["tvt_true"].to_numpy(float)) ** 2
well_sse = pool.groupby("well")["se_cp12"].sum().sort_values(ascending=False)
top40 = set(well_sse.head(TOP_WELLS).index.astype(str))

detail_parts = []
well_rows = []
for well, frame in tqdm(pool.groupby("well", sort=False), total=len(wells), desc="self-reference emission"):
    horizontal = pd.read_csv(horizontal_files[well])
    needed = {"MD", "Z", "GR", "TVT_input"}
    if not needed.issubset(horizontal.columns):
        raise RuntimeError(f"{well} horizontal missing {sorted(needed - set(horizontal.columns))}")
    typewell = pd.read_csv(typewell_files[well], usecols=["TVT", "GR"])
    typewell = typewell[np.isfinite(typewell["TVT"])].sort_values("TVT")
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean()

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    if len(known) < 32:
        continue
    anchor_md = float(md[known[-1]])

    horizontal_gr = _normalise(_fill(gr_raw), np.isfinite(gr_raw))
    prefix = _prefix_reference(
        tvt_input,
        np.where(np.isfinite(gr_raw), horizontal_gr, np.nan),
    )
    if prefix is None:
        continue
    prefix_tvt, prefix_gr = prefix

    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr_raw = typewell["GR"].to_numpy(float)
    tw_gr = _normalise(_fill(tw_gr_raw), np.isfinite(tw_gr_raw))

    frame = frame.sort_values("row_idx")
    row_idx = frame["row_idx"].to_numpy(int)
    row_idx = np.clip(row_idx, 0, len(horizontal) - 1)
    row_take = np.arange(0, len(frame), ROW_SAMPLE_STRIDE, dtype=int)
    eval_idx = row_idx[row_take]
    truth = frame["tvt_true"].to_numpy(float)[row_take]
    cp12 = frame["cp12"].to_numpy(float)[row_take]
    current = horizontal_gr[eval_idx]
    current_valid = np.isfinite(gr_raw[eval_idx])
    candidate_tvt = cp12[None, :] + OFFSETS[:, None]

    self_valid = (
        (candidate_tvt >= prefix_tvt[0])
        & (candidate_tvt <= prefix_tvt[-1])
        & current_valid[None, :]
    )
    self_grid = np.interp(candidate_tvt.ravel(), prefix_tvt, prefix_gr).reshape(candidate_tvt.shape).astype(np.float32)
    self_grid[~self_valid] = 0.0

    tw_valid = (
        (candidate_tvt >= tw_tvt[0])
        & (candidate_tvt <= tw_tvt[-1])
        & current_valid[None, :]
    )
    tw_grid = np.interp(candidate_tvt.ravel(), tw_tvt, tw_gr).reshape(candidate_tvt.shape).astype(np.float32)
    tw_grid[~tw_valid] = 0.0

    truth_offset = truth - cp12
    truth_state = np.clip(
        np.rint((truth_offset - OFFSETS[0]) / (OFFSETS[1] - OFFSETS[0])).astype(int),
        0,
        len(OFFSETS) - 1,
    )
    band = np.abs(truth_offset) <= float(np.max(np.abs(OFFSETS)))
    median_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
    median_step = max(abs(median_step), 0.25)
    deep = (md[eval_idx] - anchor_md) > DEEP_FT

    per_well = {"well": well, "rows": len(row_take), "top40": well in top40}
    for window_ft in WINDOWS_FT:
        window_rows = max(3, int(round(window_ft / median_step / ROW_SAMPLE_STRIDE)))
        if window_rows % 2 == 0:
            window_rows += 1
        self_cost = _masked_window_cost(self_grid, current, self_valid, window_rows)
        tw_cost = _masked_window_cost(tw_grid, current, tw_valid, window_rows)
        combined_cost = _robust_cost(self_cost) + _robust_cost(tw_cost)
        combined_cost[~(self_valid & tw_valid)] = np.inf

        for method, cost in (("typewell", tw_cost), ("self", self_cost), ("combined", combined_cost)):
            rank, selected, valid = _score_rows(cost, truth_state, band)
            key = f"{method}_w{int(window_ft)}"
            prediction = cp12 + OFFSETS[selected]
            detail_parts.append(
                pd.DataFrame(
                    {
                        "well": well,
                        "method": key,
                        "top40": well in top40,
                        "deep": deep,
                        "band": band,
                        "valid": valid,
                        "rank": rank,
                        "truth": truth,
                        "cp12": cp12,
                        "prediction": prediction,
                    }
                )
            )
            per_well[f"{key}_valid"] = float(np.mean(valid))
            per_well[f"{key}_top5"] = float(np.mean(valid & (rank <= 5)))
    well_rows.append(per_well)

detail = pd.concat(detail_parts, ignore_index=True)
well_report = pd.DataFrame(well_rows)

summary_rows = []
for subset_name, subset_mask in (
    ("all", np.ones(len(detail), bool)),
    ("deep", detail["deep"].to_numpy(bool)),
    ("top40", detail["top40"].to_numpy(bool)),
    ("deep_top40", detail["deep"].to_numpy(bool) & detail["top40"].to_numpy(bool)),
):
    for method, group in detail[subset_mask].groupby("method", sort=False):
        valid = group["valid"].to_numpy(bool)
        rank = group["rank"].to_numpy(float)
        summary_rows.append(
            {
                "subset": subset_name,
                "method": method,
                "rows": len(group),
                "wells": group["well"].nunique(),
                "band_coverage": float(group["band"].mean()),
                "valid_fraction": float(valid.mean()),
                "effective_top1": float(np.mean(valid & (rank <= 1))),
                "effective_top5": float(np.mean(valid & (rank <= 5))),
                "top5_given_valid": float(np.mean(rank[valid] <= 5)) if valid.any() else 0.0,
                "median_rank_given_valid": float(np.nanmedian(rank[valid])) if valid.any() else np.nan,
                "cp12_rmse": _rmse(group["truth"], group["cp12"]),
                "selected_rmse_valid": _rmse(
                    group.loc[valid, "truth"], group.loc[valid, "prediction"]
                ),
            }
        )
summary = pd.DataFrame(summary_rows)

def _metric(subset, method, column):
    row = summary[(summary["subset"] == subset) & (summary["method"] == method)]
    if len(row) != 1:
        raise RuntimeError(f"missing summary row: {subset} {method}")
    return float(row.iloc[0][column])


candidate_rows = []
for window_ft in WINDOWS_FT:
    tw = f"typewell_w{int(window_ft)}"
    self_name = f"self_w{int(window_ft)}"
    combined = f"combined_w{int(window_ft)}"
    candidate_rows.append(
        {
            "window_ft": window_ft,
            "self_valid_fraction": _metric("all", self_name, "valid_fraction"),
            "typewell_top5": _metric("all", tw, "effective_top5"),
            "self_top5": _metric("all", self_name, "effective_top5"),
            "combined_top5": _metric("all", combined, "effective_top5"),
            "combined_increment": _metric("all", combined, "effective_top5") - _metric("all", tw, "effective_top5"),
            "top40_typewell_top5": _metric("top40", tw, "effective_top5"),
            "top40_combined_top5": _metric("top40", combined, "effective_top5"),
            "top40_increment": _metric("top40", combined, "effective_top5") - _metric("top40", tw, "effective_top5"),
        }
    )
candidate_report = pd.DataFrame(candidate_rows).sort_values("combined_increment", ascending=False)
best = candidate_report.iloc[0].to_dict()
strict_pass = bool(
    best["self_valid_fraction"] >= PASS_SELF_VALID_FRACTION
    and best["combined_increment"] >= PASS_TOP5_INCREMENT
    and best["top40_increment"] >= PASS_TOP40_INCREMENT
)
verdict = {
    "experiment": "ranksdf_self_reference_emission_audit_v0",
    "pool": str(pool_path),
    "wells": int(detail["well"].nunique()),
    "sampled_rows": int(len(detail) // (len(WINDOWS_FT) * 3)),
    "offset_band_ft": float(np.max(np.abs(OFFSETS))),
    "best_window": best,
    "gates": {
        "self_valid_fraction_min": PASS_SELF_VALID_FRACTION,
        "combined_top5_increment_min": PASS_TOP5_INCREMENT,
        "top40_combined_top5_increment_min": PASS_TOP40_INCREMENT,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> rebuild RankSDF panel with same-well emission and run frozen folds 1/2"
        if strict_pass
        else "FAIL -> stop same-well RankSDF input; do not rebuild panel cache"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

summary.to_csv(OUT_DIR / "emission_summary.csv", index=False)
candidate_report.to_csv(OUT_DIR / "window_comparison.csv", index=False)
well_report.to_csv(OUT_DIR / "well_summary.csv", index=False)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("EMISSION SUMMARY")
print(summary.to_string(index=False))
print("\nWINDOW COMPARISON")
print(candidate_report.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
