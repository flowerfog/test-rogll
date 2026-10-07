"""RankSDF-CP v1 phase 1: lock and audit the PDF panel specification.

This cell does not train a model or build a large tensor cache. It audits the
exact candidate coordinate system and writes the frozen v1 manifest consumed
by later cells.

The fold-safe strong OOF curve is used only as the candidate-grid centre. It
is not exposed as a prior-SDF input channel in stage A.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "ranksdf_cp_v1_exact_panel_audit"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]

STATE_HALF_WIDTH_FT = 96.0
STATE_STEP_FT = 1.0
N_STATES = int(round(2 * STATE_HALF_WIDTH_FT / STATE_STEP_FT)) + 1
TARGET_MD_STEP_FT = 16.0
MAX_COLUMNS = 1024
TOP_WELLS = 40
DEEP_FT = 1500.0
FINGERPRINT_POINTS = 256

PASS_COVERAGE = 0.995
PASS_TOP40_COVERAGE = 0.995
PASS_KNOWN_ANCHOR_ERROR_FT = 0.05
PASS_TRUE_INSIDE_TYPEWELL = 0.995

CHANNELS_STAGE_A = [
    "typewell_gr",
    "horizontal_gr",
    "gr_difference",
    "gr_abs_difference",
    "gr_difference_smooth4ft",
    "gr_difference_smooth8ft",
    "gr_difference_smooth16ft",
    "gr_difference_smooth32ft",
    "typewell_valid",
    "horizontal_gr_valid",
    "known_path_sdf",
    "known_mask",
    "state_offset_coordinate",
    "md_relative",
    "dz_dmd",
    "d2z_dmd2",
    "curvature_xy",
    "azimuth_sin",
    "azimuth_cos",
    "self_reference_gr",
    "self_reference_difference",
    "self_reference_abs_difference",
    "self_reference_valid",
    "self_reference_coverage",
]


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


def _normalised_fingerprint(tvt, gr, points=FINGERPRINT_POINTS):
    tvt = np.asarray(tvt, np.float64)
    gr = np.asarray(gr, np.float64)
    valid = np.isfinite(tvt) & np.isfinite(gr)
    tvt, gr = tvt[valid], gr[valid]
    if len(tvt) < 32 or np.ptp(tvt) < 10.0:
        return None
    order = np.argsort(tvt)
    tvt, gr = tvt[order], gr[order]
    unique, inverse = np.unique(tvt, return_inverse=True)
    sums = np.bincount(inverse, weights=gr)
    counts = np.bincount(inverse)
    gr = sums / np.maximum(counts, 1)
    relative = (unique - unique[0]) / max(float(unique[-1] - unique[0]), 1e-6)
    sample = np.interp(np.linspace(0.0, 1.0, points), relative, gr)
    median = float(np.median(sample))
    scale = float(1.4826 * np.median(np.abs(sample - median)))
    if scale < 1e-3:
        scale = float(np.std(sample) + 1e-3)
    return np.clip((sample - median) / scale, -6.0, 6.0).astype(np.float32)


def _state_valid_fraction(center_tvt, lower, upper):
    raw_lower = np.ceil((lower - center_tvt + STATE_HALF_WIDTH_FT) / STATE_STEP_FT)
    raw_upper = np.floor((upper - center_tvt + STATE_HALF_WIDTH_FT) / STATE_STEP_FT)
    overlaps = (raw_upper >= 0) & (raw_lower <= N_STATES - 1)
    lower_index = np.clip(raw_lower, 0, N_STATES - 1)
    upper_index = np.clip(raw_upper, 0, N_STATES - 1)
    count = np.where(overlaps, np.maximum(upper_index - lower_index + 1, 0), 0)
    return count / float(N_STATES)


started = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
required = {"id", "tvt_true", "strong"}
if not required.issubset(header):
    raise RuntimeError(f"pool missing {sorted(required - set(header))}")
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong") if column in header]
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

pool["strong_se"] = (pool["strong"].to_numpy(float) - pool["tvt_true"].to_numpy(float)) ** 2
well_sse = pool.groupby("well")["strong_se"].sum().sort_values(ascending=False)
top40 = set(well_sse.head(TOP_WELLS).index.astype(str))

well_rows = []
fingerprint_wells = []
fingerprints = []
for well, frame in tqdm(pool.groupby("well", sort=False), total=len(wells), desc="exact RankSDF panel audit"):
    horizontal = pd.read_csv(horizontal_files[well])
    required_hw = {"MD", "Z", "TVT", "TVT_input"}
    if not required_hw.issubset(horizontal.columns):
        raise RuntimeError(f"{well} horizontal missing {sorted(required_hw - set(horizontal.columns))}")
    typewell = pd.read_csv(typewell_files[well], usecols=["TVT", "GR"])

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md) & np.isfinite(z))
    if len(known) < 16:
        raise RuntimeError(f"{well} has insufficient known heel")
    anchor = int(known[-1])
    anchor_md = float(md[anchor])

    frame = frame.sort_values("row_idx")
    row_index = np.clip(frame["row_idx"].to_numpy(int), 0, len(horizontal) - 1)
    truth_tvt = frame["tvt_true"].to_numpy(float)
    strong_tvt = frame["strong"].to_numpy(float)
    center_error = truth_tvt - strong_tvt
    coverage = np.abs(center_error) <= STATE_HALF_WIDTH_FT
    deep = (md[row_index] - anchor_md) > DEEP_FT

    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(float)
    finite_tw = np.isfinite(tw_tvt)
    if finite_tw.sum() < 16:
        raise RuntimeError(f"{well} typewell has insufficient TVT")
    tw_lower = float(np.min(tw_tvt[finite_tw]))
    tw_upper = float(np.max(tw_tvt[finite_tw]))
    true_inside_tw = (truth_tvt >= tw_lower) & (truth_tvt <= tw_upper)
    candidate_valid_fraction = _state_valid_fraction(strong_tvt, tw_lower, tw_upper)

    known_anchor_error = float(abs(tvt_input[anchor] - tvt[anchor]))
    full_span = float(np.nanmax(md) - np.nanmin(md))
    columns_at_16 = int(np.ceil(full_span / TARGET_MD_STEP_FT)) + 1
    adaptive_step = max(TARGET_MD_STEP_FT, full_span / max(MAX_COLUMNS - 1, 1))
    effective_columns = int(np.ceil(full_span / adaptive_step)) + 1

    prefix_tvt = tvt_input[known]
    prefix_range = float(np.nanmax(prefix_tvt) - np.nanmin(prefix_tvt))
    eval_in_prefix_range = (
        (truth_tvt >= float(np.nanmin(prefix_tvt)))
        & (truth_tvt <= float(np.nanmax(prefix_tvt)))
    )

    well_rows.append(
        {
            "well": well,
            "rows": len(frame),
            "deep_rows": int(deep.sum()),
            "top40": well in top40,
            "known_rows": len(known),
            "known_frac": float(len(known) / len(horizontal)),
            "anchor_error_ft": known_anchor_error,
            "horizontal_span_ft": full_span,
            "columns_at_16ft": columns_at_16,
            "adaptive_md_step_ft": adaptive_step,
            "effective_columns": effective_columns,
            "coverage": float(np.mean(coverage)),
            "deep_coverage": float(np.mean(coverage[deep])) if deep.any() else np.nan,
            "strong_rmse": float(np.sqrt(np.mean(center_error ** 2))),
            "true_inside_typewell": float(np.mean(true_inside_tw)),
            "candidate_typewell_valid_fraction": float(np.mean(candidate_valid_fraction)),
            "prefix_tvt_range_ft": prefix_range,
            "truth_in_prefix_tvt_range": float(np.mean(eval_in_prefix_range)),
        }
    )

    fingerprint = _normalised_fingerprint(typewell["TVT"], typewell["GR"])
    if fingerprint is not None:
        fingerprint_wells.append(well)
        fingerprints.append(fingerprint)

well_report = pd.DataFrame(well_rows)

# Near-duplicate typewell audit for the family-grouped CV requested by the PDF.
fingerprint_matrix = np.asarray(fingerprints, np.float32)
correlation = np.corrcoef(fingerprint_matrix)
np.fill_diagonal(correlation, -np.inf)
nearest_index = np.argmax(correlation, axis=1)
nearest_corr = correlation[np.arange(len(correlation)), nearest_index]
fingerprint_report = pd.DataFrame(
    {
        "well": fingerprint_wells,
        "nearest_well": np.asarray(fingerprint_wells)[nearest_index],
        "nearest_correlation": nearest_corr,
    }
).sort_values("nearest_correlation", ascending=False)


def _pooled(mask, column, weights="rows"):
    frame = well_report.loc[mask]
    if frame.empty:
        return np.nan
    return float(np.average(frame[column], weights=frame[weights]))


overall_coverage = _pooled(np.ones(len(well_report), bool), "coverage")
top40_coverage = _pooled(well_report["top40"].to_numpy(bool), "coverage")
deep_coverage = _pooled(
    well_report["deep_coverage"].notna().to_numpy(bool),
    "deep_coverage",
    weights="deep_rows",
)
true_inside_typewell = _pooled(np.ones(len(well_report), bool), "true_inside_typewell")
candidate_typewell_valid = _pooled(np.ones(len(well_report), bool), "candidate_typewell_valid_fraction")
max_anchor_error = float(well_report["anchor_error_ft"].max())

tensor_rows = []
for dtype_bytes, dtype_name in ((2, "float16"), (4, "float32")):
    bytes_per_well = len(CHANNELS_STAGE_A) * N_STATES * MAX_COLUMNS * dtype_bytes
    tensor_rows.append(
        {
            "dtype": dtype_name,
            "channels": len(CHANNELS_STAGE_A),
            "states": N_STATES,
            "columns": MAX_COLUMNS,
            "mb_per_well": bytes_per_well / 2**20,
            "gb_773_wells": bytes_per_well * len(wells) / 2**30,
        }
    )
tensor_budget = pd.DataFrame(tensor_rows)

strict_pass = bool(
    overall_coverage >= PASS_COVERAGE
    and top40_coverage >= PASS_TOP40_COVERAGE
    and max_anchor_error <= PASS_KNOWN_ANCHOR_ERROR_FT
    and true_inside_typewell >= PASS_TRUE_INSIDE_TYPEWELL
    and int(well_report["effective_columns"].max()) <= MAX_COLUMNS
)

manifest = {
    "version": "ranksdf_cp_v1_pdf_exact",
    "coordinate": "S=TVT+Z; final TVT=S-Z",
    "candidate_center": "fold-safe strong OOF for indexing only",
    "candidate_center_exposed_as_input": False,
    "prior_sdf_stage_a": False,
    "state_half_width_ft": STATE_HALF_WIDTH_FT,
    "state_step_ft": STATE_STEP_FT,
    "states": N_STATES,
    "target_md_step_ft": TARGET_MD_STEP_FT,
    "max_columns": MAX_COLUMNS,
    "channels_stage_a": CHANNELS_STAGE_A,
    "heads": ["signed_distance", "ranking_energy"],
    "loss": {
        "sdf": 1.0,
        "rank": 0.5,
        "softargmin": 0.2,
        "piecewise_s": 0.05,
        "soft_label_sigma_ft": [2.0, 4.0],
    },
    "real_masking": {"masked_heel_fraction": [0.60, 0.90]},
    "backbone": {"type": "four_level_unet", "base_channels": [24, 32], "normalisation": "GroupNorm"},
    "decode": "temperature-calibrated probabilistic DP/beam posterior mean",
}
verdict = {
    "experiment": "ranksdf_cp_v1_exact_panel_audit",
    "pool": str(pool_path),
    "wells": len(wells),
    "rows": int(well_report["rows"].sum()),
    "overall_coverage": overall_coverage,
    "deep_coverage": deep_coverage,
    "top40_coverage": top40_coverage,
    "true_inside_typewell": true_inside_typewell,
    "candidate_typewell_valid_fraction": candidate_typewell_valid,
    "max_anchor_error_ft": max_anchor_error,
    "max_columns": int(well_report["effective_columns"].max()),
    "p95_adaptive_md_step_ft": float(well_report["adaptive_md_step_ft"].quantile(0.95)),
    "typewell_near_duplicate_fraction_corr_ge_0p98": float(np.mean(nearest_corr >= 0.98)),
    "typewell_near_duplicate_fraction_corr_ge_0p995": float(np.mean(nearest_corr >= 0.995)),
    "gates": {
        "coverage_min": PASS_COVERAGE,
        "top40_coverage_min": PASS_TOP40_COVERAGE,
        "known_anchor_error_max_ft": PASS_KNOWN_ANCHOR_ERROR_FT,
        "true_inside_typewell_min": PASS_TRUE_INSIDE_TYPEWELL,
        "max_columns": MAX_COLUMNS,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build exact two-head overfit harness"
        if strict_pass
        else "FAIL -> repair candidate centre/grid before training"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

well_report.to_csv(OUT_DIR / "well_panel_audit.csv", index=False)
fingerprint_report.to_csv(OUT_DIR / "typewell_nearest_fingerprint.csv", index=False)
tensor_budget.to_csv(OUT_DIR / "tensor_budget.csv", index=False)
with (OUT_DIR / "panel_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("PANEL DISTRIBUTION")
print(
    well_report[
        [
            "known_frac", "horizontal_span_ft", "columns_at_16ft",
            "adaptive_md_step_ft", "effective_columns", "coverage",
            "deep_coverage", "candidate_typewell_valid_fraction",
            "truth_in_prefix_tvt_range",
        ]
    ].describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]).to_string()
)
print("\nTYPEWELL FINGERPRINT")
print(fingerprint_report["nearest_correlation"].describe(percentiles=[0.50, 0.90, 0.95, 0.99]).to_string())
print("\nTENSOR BUDGET")
print(tensor_budget.to_string(index=False))
print("\nMANIFEST")
print(json.dumps(manifest, indent=2))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
