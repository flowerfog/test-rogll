"""Audit legal per-column centres for a heel-conditioned CorrNet v4.

The current CorrNet searches around a constant anchor TVT and discards the
known-prefix trajectory.  This cell compares deployment-legal dynamic centres:

* constant anchor TVT and constant anchor S;
* robust tail extrapolation in S and TVT space;
* surviving official PF/likelihood-PF paths when their feature cache exists;
* existing OOF paths only as diagnostic upper references.

No target is used to fit a centre.  Truth is used only for the OOF audit.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() else DATA_ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "corrnet_center_trajectory_audit_v0"

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
PF_FEATURE_DIRS = (
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "features",
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "features",
)

TAIL_ROWS = (80, 160, 320, 640)
S_SLOPE_CAPS = (0.03, 0.05, 0.08)
TVT_SLOPE_CAPS = (0.02, 0.04)
STATE_WIDTHS = (32.0, 48.0, 64.0, 96.0, 128.0)
PF_COLUMNS = (
    "pf_ancc",
    "likpf_scale_3",
    "likpf_scale_5",
    "likpf_scale_8",
    "likpf_scale_12",
    "likpf_mean",
)
POOL_REFERENCE_COLUMNS = (
    "strong",
    "strong_bias",
    "cp12",
    "sp45",
    "fleongg",
)


def _rmse(truth, prediction, valid=None):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    if valid is None:
        valid = np.isfinite(truth) & np.isfinite(prediction)
    return float(np.sqrt(np.mean((prediction[valid] - truth[valid]) ** 2)))


def _anchored_robust_slope(md, values, anchor, tail_rows, cap):
    left = max(0, anchor - int(tail_rows) + 1)
    rows = np.arange(left, anchor + 1)
    x = np.asarray(md[rows] - md[anchor], np.float64)
    y = np.asarray(values[rows] - values[anchor], np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-6)
    x, y = x[valid], y[valid]
    if len(x) < 24:
        return 0.0
    keep = np.ones(len(x), bool)
    slope = 0.0
    for _ in range(5):
        denominator = float(np.dot(x[keep], x[keep]))
        if denominator <= 1e-8:
            break
        slope = float(np.dot(x[keep], y[keep]) / denominator)
        residual = y - slope * x
        centre = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - centre)) + 1e-6)
        next_keep = np.abs(residual - centre) <= 3.5 * scale
        if next_keep.sum() < 20 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return float(np.clip(slope, -cap, cap))


def _feature_path(well):
    names = (
        f"{well}__official_full.pkl",
        f"{well}__official__full.pkl",
    )
    for directory in PF_FEATURE_DIRS:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None


def _frame_row_index(frame):
    if "_row_idx" in frame.columns:
        return pd.to_numeric(frame["_row_idx"], errors="coerce").to_numpy(float)
    if "row_idx" in frame.columns:
        return pd.to_numeric(frame["row_idx"], errors="coerce").to_numpy(float)
    if "id" in frame.columns:
        return pd.to_numeric(
            frame["id"].astype(str).str.rsplit("_", n=1).str[-1], errors="coerce"
        ).to_numpy(float)
    raise RuntimeError("PF feature frame has no row identity")


if POOL_PATH is None:
    raise FileNotFoundError("candidate pool not found")
if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)

started = time.time()
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", *POOL_REFERENCE_COLUMNS)
    if column in header
]
pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": "string", "well": "string"})
if "id" in pool.columns:
    pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)

fold_frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
fold_frame = fold_frame.groupby("well", as_index=False)["fold"].first()
fold_map = dict(zip(fold_frame["well"].astype(str), fold_frame["fold"].astype(int)))

row_wells = pool["well"].to_numpy(str)
boundaries = np.r_[0, np.flatnonzero(row_wells[1:] != row_wells[:-1]) + 1, len(pool)]
wells = [str(row_wells[start]) for start in boundaries[:-1]]
well_starts = boundaries[:-1].astype(np.int64)
well_ends = boundaries[1:].astype(np.int64)
missing_fold_wells = sorted(set(wells) - set(fold_map))
if missing_fold_wells:
    raise RuntimeError(
        f"Fold metadata misses {len(missing_fold_wells)} wells; "
        f"first 20: {missing_fold_wells[:20]}"
    )
row_folds = pool["well"].map(fold_map).to_numpy(np.int8)
truth = pool["tvt_true"].to_numpy(np.float64)

raw_names = ["anchor_tvt", "anchor_s"]
for tail in TAIL_ROWS:
    for cap in S_SLOPE_CAPS:
        raw_names.append(f"s_tail{tail}_cap{int(round(cap * 100)):02d}")
for tail in (160, 320, 640):
    for cap in TVT_SLOPE_CAPS:
        raw_names.append(f"tvt_tail{tail}_cap{int(round(cap * 100)):02d}")
pf_names = [f"pf_{column}" for column in PF_COLUMNS]
pool_reference_names = [
    f"reference_{column}" for column in POOL_REFERENCE_COLUMNS if column in pool.columns
]
candidate_names = [*raw_names, *pf_names, *pool_reference_names]
candidate_lookup = {name: index for index, name in enumerate(candidate_names)}
predictions = np.full((len(pool), len(candidate_names)), np.nan, np.float32)

# Existing row-aligned OOF paths are diagnostic references only.
for column in POOL_REFERENCE_COLUMNS:
    if column in pool.columns:
        predictions[:, candidate_lookup[f"reference_{column}"]] = pool[column].to_numpy(np.float32)

pf_wells_loaded = 0
failures = []
well_rows_report = []
for well_i, well in enumerate(tqdm(wells, desc="centre trajectory audit")):
    start, end = int(well_starts[well_i]), int(well_ends[well_i])
    eval_rows = pool.loc[start:end - 1, "row_idx"].to_numpy(np.int32)
    try:
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=["MD", "Z", "TVT_input", "TVT"],
        )
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
        z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
        horizontal_truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
        known = np.flatnonzero(np.isfinite(tvt_input))
        if not len(known):
            raise RuntimeError("no known TVT prefix")
        anchor = int(known[-1])
        local_truth = horizontal_truth[eval_rows]
        truth_difference = np.abs(local_truth - truth[start:end])
        if not np.isfinite(truth_difference).all() or float(truth_difference.max()) > 1e-3:
            raise RuntimeError(
                "pool/horizontal truth mismatch: "
                f"max_abs={float(np.nanmax(truth_difference)):.6g}"
            )
        eval_md = md[eval_rows]
        eval_z = z[eval_rows]
        anchor_tvt = float(tvt_input[anchor])
        anchor_s = float(anchor_tvt + z[anchor])
        prefix_s = tvt_input + z

        predictions[start:end, candidate_lookup["anchor_tvt"]] = anchor_tvt
        predictions[start:end, candidate_lookup["anchor_s"]] = (
            anchor_s - eval_z
        ).astype(np.float32)

        for tail in TAIL_ROWS:
            for cap in S_SLOPE_CAPS:
                name = f"s_tail{tail}_cap{int(round(cap * 100)):02d}"
                slope = _anchored_robust_slope(md, prefix_s, anchor, tail, cap)
                centre_s = anchor_s + slope * (eval_md - md[anchor])
                predictions[start:end, candidate_lookup[name]] = (
                    centre_s - eval_z
                ).astype(np.float32)
        for tail in (160, 320, 640):
            for cap in TVT_SLOPE_CAPS:
                name = f"tvt_tail{tail}_cap{int(round(cap * 100)):02d}"
                slope = _anchored_robust_slope(md, tvt_input, anchor, tail, cap)
                centre_tvt = anchor_tvt + slope * (eval_md - md[anchor])
                predictions[start:end, candidate_lookup[name]] = centre_tvt.astype(np.float32)

        pf_path = _feature_path(well)
        if pf_path is not None:
            frame = pd.read_pickle(pf_path)
            frame_rows = _frame_row_index(frame)
            valid_rows = np.isfinite(frame_rows)
            indexed = frame.loc[valid_rows].copy()
            indexed["_audit_row_idx"] = frame_rows[valid_rows].astype(np.int32)
            indexed = indexed.drop_duplicates("_audit_row_idx", keep="last").set_index("_audit_row_idx")
            loaded_any = False
            for column in PF_COLUMNS:
                if column not in indexed.columns:
                    continue
                values = pd.to_numeric(
                    indexed[column].reindex(eval_rows), errors="coerce"
                ).to_numpy(np.float32)
                predictions[start:end, candidate_lookup[f"pf_{column}"]] = values
                loaded_any |= bool(np.isfinite(values).mean() >= 0.99)
            pf_wells_loaded += int(loaded_any)

        well_rows_report.append(
            {
                "well": well,
                "fold": int(fold_map[well]),
                "rows": end - start,
                "known_frac": (anchor + 1) / len(md),
                "eval_span_ft": float(eval_md[-1] - md[anchor]),
            }
        )
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})

failure_frame = pd.DataFrame(failures)
failure_frame.to_csv(OUT_DIR / "failures.csv", index=False)
if failures:
    print(failure_frame.head(20).to_string(index=False))
    raise RuntimeError(f"failed to audit {len(failures)} wells")

well_metadata = pd.DataFrame(well_rows_report)
well_metadata.to_csv(OUT_DIR / "well_metadata.csv", index=False)

score_rows = []
fold_rows = []
per_well_rows = []
for candidate_i, name in enumerate(candidate_names):
    prediction = predictions[:, candidate_i].astype(np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    absolute = np.abs(prediction - truth)
    candidate_well_coverage = {width: [] for width in STATE_WIDTHS}
    candidate_well_sse = []
    candidate_well_rows = []
    for well_i, well in enumerate(wells):
        start, end = int(well_starts[well_i]), int(well_ends[well_i])
        local_valid = valid[start:end]
        if not local_valid.any():
            local_sse = np.nan
            local_rmse = np.nan
        else:
            local_error = prediction[start:end][local_valid] - truth[start:end][local_valid]
            local_sse = float(np.sum(local_error**2))
            local_rmse = float(np.sqrt(np.mean(local_error**2)))
        candidate_well_sse.append(local_sse)
        candidate_well_rows.append(int(local_valid.sum()))
        row = {
            "candidate": name,
            "well": well,
            "fold": int(fold_map[well]),
            "rows": int(local_valid.sum()),
            "rmse": local_rmse,
        }
        for width in STATE_WIDTHS:
            coverage = float(np.mean(absolute[start:end] <= width))
            candidate_well_coverage[width].append(coverage)
            row[f"coverage_{int(width)}"] = coverage
        per_well_rows.append(row)

    valid_fraction = float(valid.mean())
    result = {
        "candidate": name,
        "family": (
            "raw_prefix" if name in raw_names else
            "pf_singlewell" if name in pf_names else
            "oof_reference"
        ),
        "valid_fraction": valid_fraction,
        "rows": int(valid.sum()),
        "rmse": _rmse(truth, prediction, valid) if valid.any() else np.nan,
        "median_well_rmse": float(
            np.nanmedian(np.sqrt(np.asarray(candidate_well_sse) / np.maximum(candidate_well_rows, 1)))
        ),
    }
    for width in STATE_WIDTHS:
        result[f"coverage_{int(width)}"] = float(np.mean(valid & (absolute <= width)))
        result[f"state_p10_coverage_{int(width)}"] = float(
            np.quantile(candidate_well_coverage[width], 0.10)
        )
    score_rows.append(result)

    for fold in sorted(set(fold_map.values())):
        mask = (row_folds == fold) & valid
        fold_rows.append(
            {
                "candidate": name,
                "fold": fold,
                "rows": int(mask.sum()),
                "valid_fraction": float(np.mean(valid[row_folds == fold])),
                "rmse": _rmse(truth, prediction, mask) if mask.any() else np.nan,
                **{
                    f"coverage_{int(width)}": float(
                        np.mean(mask & (absolute <= width)) / max(np.mean(row_folds == fold), 1e-12)
                    )
                    for width in STATE_WIDTHS
                },
            }
        )

score_report = pd.DataFrame(score_rows).sort_values(
    ["valid_fraction", "rmse"], ascending=[False, True]
)
fold_report = pd.DataFrame(fold_rows)
per_well_report = pd.DataFrame(per_well_rows)
score_report.to_csv(OUT_DIR / "center_scores.csv", index=False)
fold_report.to_csv(OUT_DIR / "center_folds.csv", index=False)
per_well_report.to_csv(OUT_DIR / "center_bywell.csv", index=False)

eligible_raw = score_report[
    (score_report["family"] == "raw_prefix")
    & (score_report["valid_fraction"] >= 0.999)
].copy()
eligible_singlewell = score_report[
    score_report["family"].isin(["raw_prefix", "pf_singlewell"])
    & (score_report["valid_fraction"] >= 0.995)
].copy()
if not len(eligible_raw):
    raise RuntimeError("no complete raw-prefix centre")

best_raw = eligible_raw.sort_values(
    ["coverage_64", "rmse"], ascending=[False, True]
).iloc[0]
best_singlewell = (
    eligible_singlewell.sort_values(
        ["coverage_64", "rmse"], ascending=[False, True]
    ).iloc[0]
    if len(eligible_singlewell)
    else best_raw
)
best_raw_name = str(best_raw["candidate"])
best_singlewell_name = str(best_singlewell["candidate"])

np.savez_compressed(
    OUT_DIR / "recommended_centres_oof.npz",
    raw_prefix_center=predictions[:, candidate_lookup[best_raw_name]],
    raw_prefix_name=np.asarray([best_raw_name], str),
    singlewell_center=predictions[:, candidate_lookup[best_singlewell_name]],
    singlewell_name=np.asarray([best_singlewell_name], str),
    row_folds=row_folds,
    well_names=np.asarray(wells, str),
    well_starts=well_starts,
    well_ends=well_ends,
)

raw_pass = bool(
    float(best_raw["coverage_64"]) >= 0.99
    and float(best_raw["state_p10_coverage_64"]) >= 0.95
)
singlewell_pass = bool(
    float(best_singlewell["coverage_64"]) >= 0.99
    and float(best_singlewell["state_p10_coverage_64"]) >= 0.95
)
verdict = {
    "experiment": "corrnet_center_trajectory_audit_v0",
    "pool": str(POOL_PATH),
    "fold_source": str(FOLD_PATH),
    "rows": len(pool),
    "wells": len(wells),
    "pf_wells_loaded": pf_wells_loaded,
    "best_raw_prefix_center": best_raw_name,
    "best_raw_prefix_rmse": float(best_raw["rmse"]),
    "best_raw_prefix_coverage64": float(best_raw["coverage_64"]),
    "best_raw_prefix_state_p10_coverage64": float(best_raw["state_p10_coverage_64"]),
    "best_singlewell_center": best_singlewell_name,
    "best_singlewell_rmse": float(best_singlewell["rmse"]),
    "best_singlewell_coverage64": float(best_singlewell["coverage_64"]),
    "best_singlewell_state_p10_coverage64": float(
        best_singlewell["state_p10_coverage_64"]
    ),
    "raw_prefix_pass_at_64": raw_pass,
    "singlewell_pass_at_64": singlewell_pass,
    "next": (
        "build heel-conditioned CorrNet v4 with +/-64 ft dynamic state axis"
        if raw_pass or singlewell_pass
        else "use +/-96 or +/-128 ft and first add the known-prefix condition channels"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nCENTER SCORES")
print(score_report.to_string(index=False))
print("\nBEST FOLD SCORES")
print(
    fold_report[
        fold_report["candidate"].isin([best_raw_name, best_singlewell_name])
    ].to_string(index=False)
)
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
