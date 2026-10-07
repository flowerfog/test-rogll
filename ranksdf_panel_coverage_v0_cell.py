"""RankSDF Step 0: audit the two-dimensional candidate panel.

The audit answers four questions before any U-Net training:

1. How wide must the shared S = TVT + Z candidate axis be?
2. Can prefix-only centres cover truth, or is a fold-safe base required?
3. How much of the panel lies inside the assigned typewell TVT range?
4. What tensor shape and cache size follow from each width/step choice?

No model is trained and no parameter is selected against an OOF RMSE.  The
primary preregistered gate is >=99.8% row coverage with a prefix-only centre.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "ranksdf_panel_coverage_v0"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]

WIDTHS_FT = (64.0, 96.0, 120.0, 160.0, 192.0, 256.0, 320.0)
STATE_STEPS_FT = (1.0, 1.5, 2.0, 3.0, 4.0)
PANEL_COLUMNS = (512, 768, 1024)
ASSUMED_CHANNELS = (8, 12, 16)
PRIMARY_STATE_STEP_FT = 2.0
PASS_COVERAGE = 0.998
MAX_PRIMARY_ROWS = 321

PREFIX_CENTRES = ("prefix_last500", "prefix_all", "anchor_constant")
BASE_CENTRES = ("strong", "cp12")


def _robust_slope(md, structure, anchor_md, anchor_structure):
    md = np.asarray(md, np.float64)
    structure = np.asarray(structure, np.float64)
    x = md - float(anchor_md)
    y = structure - float(anchor_structure)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 8 or np.ptp(x) < 20.0:
        return 0.0
    denominator = float(np.sum(x * x))
    slope = float(np.sum(x * y) / max(denominator, 1e-12))
    for _ in range(5):
        residual = y - slope * x
        scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6)
        weight = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        slope = float(np.sum(weight * x * y) / max(np.sum(weight * x * x), 1e-12))
    return float(np.clip(slope, -0.5, 0.5))


def _candidate_valid_fraction(center_s, z, tw_min, tw_max, width, step):
    """Discrete fraction of candidate cells that lie inside typewell bounds."""
    offsets = np.arange(-width, width + 0.25 * step, step, dtype=np.float64)
    lower_offset = tw_min + z - center_s
    upper_offset = tw_max + z - center_s
    first = np.ceil((lower_offset + width) / step).astype(np.int64)
    last = np.floor((upper_offset + width) / step).astype(np.int64)
    intersects = (upper_offset >= -width) & (lower_offset <= width)
    first = np.clip(first, 0, len(offsets) - 1)
    last = np.clip(last, 0, len(offsets) - 1)
    valid_count = np.where(intersects, np.maximum(last - first + 1, 0), 0)
    return float(np.sum(valid_count) / (len(center_s) * len(offsets)))


def _quantisation_rmse(residual_s, width, step):
    residual_s = np.asarray(residual_s, np.float64)
    covered = np.abs(residual_s) <= width + 1e-9
    if not covered.any():
        return np.nan
    clipped = np.clip(residual_s[covered], -width, width)
    nearest = -width + np.round((clipped + width) / step) * step
    return float(np.sqrt(np.mean((nearest - residual_s[covered]) ** 2)))


def _discover_files(wells, suffix):
    wells = set(map(str, wells))
    selected = {}
    for root in (DATA_ROOT / "train", DATA_ROOT):
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            name = path.name
            if name.endswith(suffix):
                well = name[: -len(suffix)]
            else:
                continue
            if well in wells and "test" not in {part.lower() for part in path.parts}:
                selected.setdefault(well, path)
    missing = sorted(wells - set(selected))
    if missing:
        raise FileNotFoundError(
            f"missing {suffix} for {len(missing)} wells; examples={missing[:20]}"
        )
    return selected


started = time.time()
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError(f"candidate pool not found under {OOF_ROOT}")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
prediction_columns = [column for column in BASE_CENTRES if column in header]
usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", *prediction_columns)
    if column in header
]
print("loading:", pool_path)
pool = pd.read_csv(
    pool_path,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
wells = sorted(pool["well"].unique())
horizontal_files = _discover_files(wells, "__horizontal_well.csv")
typewell_files = _discover_files(wells, "__typewell.csv")

# Freeze top40 from strong when available.  It is diagnostic only.
if "strong" in pool:
    pool["_base_se"] = (
        pool["strong"].to_numpy(float) - pool["tvt_true"].to_numpy(float)
    ) ** 2
    top40_wells = set(
        pool.groupby("well")["_base_se"].sum().nlargest(40).index.astype(str)
    )
else:
    top40_wells = set()

centres = {
    name: np.full(len(pool), np.nan, np.float32)
    for name in (*PREFIX_CENTRES, *prediction_columns)
}
z_all = np.full(len(pool), np.nan, np.float32)
md_since_all = np.full(len(pool), np.nan, np.float32)
tw_min_all = np.full(len(pool), np.nan, np.float32)
tw_max_all = np.full(len(pool), np.nan, np.float32)
per_well_rows = []

for count, (well, frame) in enumerate(pool.groupby("well", sort=False), 1):
    horizontal = pd.read_csv(horizontal_files[well])
    typewell = pd.read_csv(typewell_files[well], usecols=["TVT", "GR"])
    row_index = frame["row_idx"].to_numpy(int)
    if row_index.max() >= len(horizontal):
        raise RuntimeError(f"row index outside horizontal file for {well}")
    required = {"MD", "Z", "TVT_input"}
    if not required.issubset(horizontal.columns):
        raise RuntimeError(f"{well} horizontal missing {sorted(required - set(horizontal.columns))}")

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(z) & np.isfinite(md))
    if len(known) < 16:
        raise RuntimeError(f"insufficient known prefix for {well}")
    anchor = int(known[-1])
    anchor_md = float(md[anchor])
    anchor_s = float(tvt_input[anchor] + z[anchor])
    known_s = tvt_input[known] + z[known]
    slope_all = _robust_slope(md[known], known_s, anchor_md, anchor_s)
    recent = known[md[known] >= anchor_md - 500.0]
    if len(recent) < 16:
        recent = known[-min(len(known), 64):]
    recent_s = tvt_input[recent] + z[recent]
    slope_last500 = _robust_slope(md[recent], recent_s, anchor_md, anchor_s)

    position = frame.index.to_numpy(int)
    eval_md = md[row_index]
    eval_z = z[row_index]
    centres["anchor_constant"][position] = anchor_s
    centres["prefix_all"][position] = anchor_s + slope_all * (eval_md - anchor_md)
    centres["prefix_last500"][position] = anchor_s + slope_last500 * (eval_md - anchor_md)
    for column in prediction_columns:
        centres[column][position] = (
            frame[column].to_numpy(float) + eval_z
        ).astype(np.float32)
    z_all[position] = eval_z.astype(np.float32)
    md_since_all[position] = np.maximum(eval_md - anchor_md, 0.0).astype(np.float32)

    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(float)
    tw_tvt = tw_tvt[np.isfinite(tw_tvt)]
    if len(tw_tvt) < 32:
        raise RuntimeError(f"insufficient typewell TVT coverage for {well}")
    tw_min = float(np.min(tw_tvt))
    tw_max = float(np.max(tw_tvt))
    tw_min_all[position] = tw_min
    tw_max_all[position] = tw_max
    truth_s = frame["tvt_true"].to_numpy(float) + eval_z
    per_well_rows.append(
        {
            "well": well,
            "rows": len(frame),
            "anchor_s": anchor_s,
            "slope_all": slope_all,
            "slope_last500": slope_last500,
            "truth_s_span": float(np.ptp(truth_s)),
            "truth_tvt_inside_typewell": float(
                np.mean(
                    (frame["tvt_true"].to_numpy(float) >= tw_min)
                    & (frame["tvt_true"].to_numpy(float) <= tw_max)
                )
            ),
            "top40": well in top40_wells,
        }
    )
    if count % 100 == 0:
        print(f"panel metadata {count}/{len(wells)}", flush=True)

truth_s_all = pool["tvt_true"].to_numpy(float) + z_all.astype(float)
if not np.isfinite(truth_s_all).all() or not np.isfinite(z_all).all():
    raise RuntimeError("non-finite truth S or Z")

per_well = pd.DataFrame(per_well_rows)
all_summary_rows = []
residual_rows = []

subsets = {
    "all": np.ones(len(pool), dtype=bool),
    "deep_gt1500": md_since_all > 1500.0,
}
if top40_wells:
    top40_mask = pool["well"].isin(top40_wells).to_numpy()
    subsets["top40"] = top40_mask
    subsets["deep_top40"] = top40_mask & (md_since_all > 1500.0)

for centre_name, center_s in centres.items():
    center_s = center_s.astype(float)
    if not np.isfinite(center_s).all():
        raise RuntimeError(f"non-finite centre {centre_name}")
    residual = truth_s_all - center_s

    # Per-well path diagnostics on the native row order.
    for well, index in pool.groupby("well", sort=False).indices.items():
        values = residual[np.asarray(index, int)]
        first = np.diff(values)
        second = np.diff(values, n=2)
        residual_rows.append(
            {
                "well": well,
                "centre": centre_name,
                "rows": len(values),
                "residual_abs_p50": float(np.quantile(np.abs(values), 0.50)),
                "residual_abs_p95": float(np.quantile(np.abs(values), 0.95)),
                "residual_abs_max": float(np.max(np.abs(values))),
                "delta_abs_p99": float(np.quantile(np.abs(first), 0.99)) if len(first) else 0.0,
                "second_delta_abs_p99": float(np.quantile(np.abs(second), 0.99)) if len(second) else 0.0,
                "top40": well in top40_wells,
            }
        )

    for subset_name, subset in subsets.items():
        subset_residual = residual[subset]
        subset_center = center_s[subset]
        subset_z = z_all[subset].astype(float)
        subset_tw_min = tw_min_all[subset].astype(float)
        subset_tw_max = tw_max_all[subset].astype(float)
        for width in WIDTHS_FT:
            coverage = float(np.mean(np.abs(subset_residual) <= width + 1e-9))
            typewell_valid = _candidate_valid_fraction(
                subset_center,
                subset_z,
                subset_tw_min,
                subset_tw_max,
                width,
                PRIMARY_STATE_STEP_FT,
            )
            all_summary_rows.append(
                {
                    "centre": centre_name,
                    "subset": subset_name,
                    "width_ft": width,
                    "state_step_ft": PRIMARY_STATE_STEP_FT,
                    "candidate_rows": int(np.floor(2 * width / PRIMARY_STATE_STEP_FT)) + 1,
                    "rows": int(subset.sum()),
                    "coverage": coverage,
                    "outside_rows": int(np.sum(np.abs(subset_residual) > width + 1e-9)),
                    "residual_abs_p50": float(np.quantile(np.abs(subset_residual), 0.50)),
                    "residual_abs_p95": float(np.quantile(np.abs(subset_residual), 0.95)),
                    "residual_abs_p99": float(np.quantile(np.abs(subset_residual), 0.99)),
                    "residual_abs_max": float(np.max(np.abs(subset_residual))),
                    "candidate_typewell_valid_fraction": typewell_valid,
                    "quantisation_rmse": _quantisation_rmse(
                        subset_residual, width, PRIMARY_STATE_STEP_FT
                    ),
                }
            )

summary = pd.DataFrame(all_summary_rows)
residual_by_well = pd.DataFrame(residual_rows)

# Tensor/storage table is independent of data values.
tensor_rows = []
for width in WIDTHS_FT:
    for step in STATE_STEPS_FT:
        states = int(np.floor(2 * width / step)) + 1
        for columns in PANEL_COLUMNS:
            for channels in ASSUMED_CHANNELS:
                values = states * columns * channels
                tensor_rows.append(
                    {
                        "width_ft": width,
                        "state_step_ft": step,
                        "candidate_rows": states,
                        "md_columns": columns,
                        "channels": channels,
                        "float16_mb_per_well": values * 2 / 2**20,
                        "float16_gb_773_wells": values * 2 * len(wells) / 2**30,
                        "float32_gb_773_wells": values * 4 * len(wells) / 2**30,
                    }
                )
tensor_report = pd.DataFrame(tensor_rows)

# Select the narrowest prefix-only panel that satisfies the all-row gate and
# remains within the declared first-pass row budget.
primary_candidates = summary[
    (summary["subset"] == "all")
    & (summary["centre"].isin(PREFIX_CENTRES))
    & (summary["coverage"] >= PASS_COVERAGE)
    & (summary["candidate_rows"] <= MAX_PRIMARY_ROWS)
].copy()
primary_candidates["centre_priority"] = primary_candidates["centre"].map(
    {name: priority for priority, name in enumerate(PREFIX_CENTRES)}
)
primary_candidates = primary_candidates.sort_values(
    ["candidate_rows", "centre_priority", "candidate_typewell_valid_fraction"],
    ascending=[True, True, False],
)

selected = None
if not primary_candidates.empty:
    row = primary_candidates.iloc[0]
    selected = {
        "centre": str(row["centre"]),
        "width_ft": float(row["width_ft"]),
        "state_step_ft": PRIMARY_STATE_STEP_FT,
        "candidate_rows": int(row["candidate_rows"]),
        "coverage": float(row["coverage"]),
        "candidate_typewell_valid_fraction": float(
            row["candidate_typewell_valid_fraction"]
        ),
        "md_columns_v0": 768,
        "reason": "narrowest prefix-only panel passing coverage gate",
    }

truth_inside_typewell = float(
    np.mean(
        (pool["tvt_true"].to_numpy(float) >= tw_min_all)
        & (pool["tvt_true"].to_numpy(float) <= tw_max_all)
    )
)
verdict = {
    "experiment": "ranksdf_panel_coverage_v0",
    "pool": str(pool_path),
    "rows": len(pool),
    "wells": len(wells),
    "truth_tvt_inside_typewell": truth_inside_typewell,
    "coverage_gate": PASS_COVERAGE,
    "max_primary_candidate_rows": MAX_PRIMARY_ROWS,
    "selected_panel": selected,
    "strict_pass": selected is not None and truth_inside_typewell >= 0.999,
    "next": (
        "build 8-16 well Rank-UNet overfit harness"
        if selected is not None and truth_inside_typewell >= 0.999
        else "stop and redesign candidate centre/range before training"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_DIR / "panel_coverage_summary.csv", index=False)
residual_by_well.to_csv(OUT_DIR / "panel_residual_by_well.csv", index=False)
per_well.to_csv(OUT_DIR / "panel_well_metadata.csv", index=False)
tensor_report.to_csv(OUT_DIR / "panel_tensor_budget.csv", index=False)
with (OUT_DIR / "selected_panel.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

display_columns = [
    "centre",
    "subset",
    "width_ft",
    "candidate_rows",
    "coverage",
    "outside_rows",
    "residual_abs_p95",
    "residual_abs_p99",
    "candidate_typewell_valid_fraction",
    "quantisation_rmse",
]
print("\nALL-ROW COVERAGE")
print(
    summary[summary["subset"] == "all"][display_columns]
    .sort_values(["centre", "width_ft"])
    .to_string(index=False)
)
if "deep_top40" in subsets:
    print("\nDEEP TOP40 COVERAGE")
    print(
        summary[summary["subset"] == "deep_top40"][display_columns]
        .sort_values(["centre", "width_ft"])
        .to_string(index=False)
    )
print("\nTENSOR BUDGET (selected neighborhood)")
if selected is not None:
    near = tensor_report[
        (tensor_report["width_ft"] == selected["width_ft"])
        & (tensor_report["state_step_ft"] == selected["state_step_ft"])
    ]
else:
    near = tensor_report[
        (tensor_report["width_ft"] == 192.0)
        & (tensor_report["state_step_ft"] == PRIMARY_STATE_STEP_FT)
    ]
print(near.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
