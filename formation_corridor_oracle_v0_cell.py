"""Perfect-formation-top corridor oracle for honest OOF predictions.

This is a Step-0 ceiling test, not a deployable model. Formation tops come from
the training typewell Geology labels, while corridor membership is determined
only by the visible anchor TVT. Wells whose eval truth crosses the anchor
corridor remain in the score; truth is used only for evaluation diagnostics.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / "formation_corridor_oracle_v0"

MAJOR_FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
BASE_CANDIDATES = ("strong_bias", "cp12", "strong")
PRIMARY_ZONE = "EGFDL-BUDA"

# u expands the true corridor outwards and approximates top-location error.
UNCERTAINTY_FT = (0.0, 4.0, 8.0, 12.0, 16.0)
SHRINKS = (0.5, 1.0)
MOVE_CAPS_FT = (6.0, 12.0, 24.0, np.inf)
SIDES = ("both", "low_only", "high_only")

N_FOLDS = 5
RANDOM_STATE = 20260715
ORACLE_GAIN_GATE = 0.30
TOP40_GAIN_GATE = 0.30

if POOL_PATH is None:
    raise FileNotFoundError(f"candidate pool not found under {OOF_ROOT}")
if not (DATA_ROOT / "train").exists():
    raise FileNotFoundError(DATA_ROOT / "train")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(max(float(sse), 0.0) / max(int(rows), 1)))


def _formation_tops(typewell):
    geology = typewell["Geology"].astype("string").str.strip()
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce")
    tops = {}
    for formation in MAJOR_FORMATIONS:
        values = tvt[geology.eq(formation)].dropna()
        if len(values):
            tops[formation] = float(values.min())
    return tops


def _anchor_corridor(anchor_tvt, tops):
    ordered = sorted((value, name) for name, value in tops.items())
    if len(ordered) < 2 or not np.isfinite(anchor_tvt):
        return None
    values = np.asarray([value for value, _ in ordered], np.float64)
    index = int(np.searchsorted(values, anchor_tvt, side="right") - 1)
    if index < 0 or index >= len(ordered) - 1:
        return None
    low, low_name = ordered[index]
    high, high_name = ordered[index + 1]
    return {
        "zone": f"{low_name}-{high_name}",
        "low_name": low_name,
        "high_name": high_name,
        "low": float(low),
        "high": float(high),
        "width": float(high - low),
    }


print(
    {
        "experiment": "formation_corridor_oracle_v0",
        "pool": str(POOL_PATH),
        "primary_zone": PRIMARY_ZONE,
        "uncertainty_ft": UNCERTAINTY_FT,
        "shrinks": SHRINKS,
        "move_caps_ft": [None if not np.isfinite(x) else x for x in MOVE_CAPS_FT],
        "sides": SIDES,
    }
)


# -----------------------------------------------------------------------------
# Load the OOF pool
# -----------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
base_cols = [column for column in BASE_CANDIDATES if column in header]
required = {"id", "tvt_true"}
missing = sorted(required - set(header))
if missing:
    raise RuntimeError(f"pool missing columns: {missing}; columns={header}")
if not base_cols:
    raise RuntimeError(f"no supported base columns found; columns={header}")

usecols = [column for column in ("id", "well", "row_idx", "tvt_true", *base_cols) if column in header]
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)

truth = pool["tvt_true"].to_numpy(np.float64)
if not np.isfinite(truth).all():
    raise RuntimeError("non-finite truth in OOF pool")
print("pool:", pool.shape, "wells:", pool["well"].nunique(), "bases:", base_cols)


# -----------------------------------------------------------------------------
# Build a perfect-top corridor table without using eval truth for assignment
# -----------------------------------------------------------------------------
metadata_rows = []
failures = []
started = time.time()

for wid in tqdm(sorted(pool["well"].unique()), desc="formation corridors"):
    try:
        horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
        typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=["TVT", "TVT_input"])
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "Geology"])

        visible = horizontal["TVT_input"].notna().to_numpy()
        if not visible.any() or visible.all():
            raise ValueError("invalid TVT_input mask")
        anchor_tvt = float(horizontal.loc[visible, "TVT_input"].iloc[-1])
        corridor = _anchor_corridor(anchor_tvt, _formation_tops(typewell))

        local_truth = horizontal.loc[~visible, "TVT"].to_numpy(np.float64)
        eval_min = float(np.nanmin(local_truth))
        eval_max = float(np.nanmax(local_truth))
        row = {
            "well": str(wid),
            "anchor_tvt": anchor_tvt,
            "eval_min_tvt": eval_min,
            "eval_max_tvt": eval_max,
            "eval_rows_file": int((~visible).sum()),
            "typewell_min_tvt": float(pd.to_numeric(typewell["TVT"], errors="coerce").min()),
            "typewell_max_tvt": float(pd.to_numeric(typewell["TVT"], errors="coerce").max()),
        }
        if corridor is None:
            row.update(
                {
                    "zone": "unbounded_or_missing",
                    "low_name": "",
                    "high_name": "",
                    "corridor_low": np.nan,
                    "corridor_high": np.nan,
                    "corridor_width": np.nan,
                    "crosses_corridor": True,
                    "lower_truth_margin": np.nan,
                    "upper_truth_margin": np.nan,
                }
            )
        else:
            crosses = eval_min < corridor["low"] or eval_max >= corridor["high"]
            row.update(
                {
                    "zone": corridor["zone"],
                    "low_name": corridor["low_name"],
                    "high_name": corridor["high_name"],
                    "corridor_low": corridor["low"],
                    "corridor_high": corridor["high"],
                    "corridor_width": corridor["width"],
                    "crosses_corridor": bool(crosses),
                    "lower_truth_margin": eval_min - corridor["low"],
                    "upper_truth_margin": corridor["high"] - eval_max,
                }
            )
        metadata_rows.append(row)
    except Exception as exc:
        failures.append({"well": str(wid), "reason": repr(exc)})

well_meta = pd.DataFrame(metadata_rows)
failure_report = pd.DataFrame(failures)
well_meta.to_csv(OUT_DIR / "well_corridor_audit.csv", index=False)
failure_report.to_csv(OUT_DIR / "failures.csv", index=False)
if len(well_meta) != pool["well"].nunique():
    raise RuntimeError(f"corridor metadata incomplete: {len(well_meta)} / {pool['well'].nunique()}")

zone_report = (
    well_meta.groupby("zone", as_index=False)
    .agg(
        wells=("well", "size"),
        crossing_wells=("crosses_corridor", "sum"),
        crossing_rate=("crosses_corridor", "mean"),
        median_width=("corridor_width", "median"),
        median_lower_margin=("lower_truth_margin", "median"),
        median_upper_margin=("upper_truth_margin", "median"),
    )
    .sort_values("wells", ascending=False)
)
zone_report.to_csv(OUT_DIR / "zone_summary.csv", index=False)
print("\nZONE SUMMARY")
display(zone_report)
print(
    "corridor metadata:",
    {
        "wells": len(well_meta),
        "failures": len(failure_report),
        "zero_cross_rate": float((~well_meta["crosses_corridor"]).mean()),
        "primary_wells": int(well_meta["zone"].eq(PRIMARY_ZONE).sum()),
        "elapsed_sec": time.time() - started,
    },
)


# -----------------------------------------------------------------------------
# Attach well-level corridor metadata and stable five-fold labels
# -----------------------------------------------------------------------------
meta_index = well_meta.set_index("well")
low = pool["well"].map(meta_index["corridor_low"]).to_numpy(np.float64)
high = pool["well"].map(meta_index["corridor_high"]).to_numpy(np.float64)
zone = pool["well"].map(meta_index["zone"]).astype(str).to_numpy()
crossing_well = pool["well"].map(meta_index["crosses_corridor"]).fillna(True).to_numpy(bool)

fold_source_candidates = [
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "well_metadata.csv",
]
fold_map = None
fold_source = None
for path in fold_source_candidates:
    if not path.exists():
        continue
    candidate = pd.read_csv(path)
    if {"well", "fold"}.issubset(candidate.columns):
        candidate["well"] = candidate["well"].astype(str)
        fold_map = candidate.groupby("well")["fold"].first().astype(int).to_dict()
        if set(pool["well"].unique()).issubset(fold_map):
            fold_source = str(path)
            break
        fold_map = None

if fold_map is None:
    counts = pool.groupby("well", as_index=False).size().rename(columns={"size": "rows"})
    counts["stratum"] = pd.qcut(
        counts["rows"].rank(method="first"), q=10, labels=False
    ).astype(int)
    counts["fold"] = -1
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, valid_indices) in enumerate(splitter.split(counts, counts["stratum"]), 1):
        counts.loc[valid_indices, "fold"] = fold
    fold_map = dict(zip(counts["well"].astype(str), counts["fold"].astype(int)))
    fold_source = "deterministic row-count StratifiedKFold"

fold_values = pool["well"].map(fold_map)
if fold_values.isna().any():
    raise RuntimeError(f"missing folds for {int(fold_values.isna().sum())} rows")
fold_values = fold_values.to_numpy(np.int8)
well_codes, well_names = pd.factorize(pool["well"], sort=True)
print("fold source:", fold_source)


# -----------------------------------------------------------------------------
# Scan a deliberately small family of deployable clipping rules
# -----------------------------------------------------------------------------
scope_mask = (zone == PRIMARY_ZONE) & np.isfinite(low) & np.isfinite(high)
scope_indices = np.flatnonzero(scope_mask)
if len(scope_indices) == 0:
    raise RuntimeError(f"no rows in primary zone {PRIMARY_ZONE}")

scan_rows = []
fold_rows = []
for base_col in base_cols:
    base = pool[base_col].to_numpy(np.float64)
    if not np.isfinite(base).all():
        print(f"skip {base_col}: non-finite predictions", flush=True)
        continue

    error = base - truth
    squared_error = error * error
    base_sse = float(squared_error.sum())
    base_rmse = _rmse_from_sse(base_sse, len(pool))

    well_sse = np.bincount(well_codes, weights=squared_error, minlength=len(well_names))
    top40_codes = np.argsort(well_sse)[-40:]
    top40_mask = np.isin(well_codes, top40_codes)
    top40_rows = int(top40_mask.sum())
    top40_base_sse = float(squared_error[top40_mask].sum())
    top40_base_rmse = _rmse_from_sse(top40_base_sse, top40_rows)

    cross_mask = crossing_well
    noncross_mask = ~crossing_well
    cross_rows = int(cross_mask.sum())
    noncross_rows = int(noncross_mask.sum())
    cross_base_sse = float(squared_error[cross_mask].sum())
    noncross_base_sse = float(squared_error[noncross_mask].sum())

    fold_base = {}
    for fold in range(1, N_FOLDS + 1):
        mask = fold_values == fold
        fold_base[fold] = {
            "rows": int(mask.sum()),
            "sse": float(squared_error[mask].sum()),
        }

    for uncertainty in UNCERTAINTY_FT:
        corridor_low = low[scope_indices] - float(uncertainty)
        corridor_high = high[scope_indices] + float(uncertainty)
        local_base = base[scope_indices]

        for sides in SIDES:
            target = local_base.copy()
            if sides in ("both", "low_only"):
                target = np.maximum(target, corridor_low)
            if sides in ("both", "high_only"):
                target = np.minimum(target, corridor_high)
            raw_move = target - local_base

            for shrink in SHRINKS:
                for cap in MOVE_CAPS_FT:
                    if np.isfinite(cap):
                        move = float(shrink) * np.clip(raw_move, -float(cap), float(cap))
                        cap_label = float(cap)
                    else:
                        move = float(shrink) * raw_move
                        cap_label = "full"

                    modified_local = np.abs(move) > 1e-9
                    modified_indices = scope_indices[modified_local]
                    modified_move = move[modified_local]
                    if len(modified_indices):
                        old_error = error[modified_indices]
                        delta_sse = 2.0 * old_error * modified_move + modified_move * modified_move
                    else:
                        delta_sse = np.empty(0, np.float64)

                    total_sse = base_sse + float(delta_sse.sum())
                    score = _rmse_from_sse(total_sse, len(pool))

                    fold_gains = []
                    for fold in range(1, N_FOLDS + 1):
                        local_fold = fold_values[modified_indices] == fold
                        fold_delta = float(delta_sse[local_fold].sum())
                        candidate_sse = fold_base[fold]["sse"] + fold_delta
                        candidate_rmse = _rmse_from_sse(candidate_sse, fold_base[fold]["rows"])
                        candidate_base_rmse = _rmse_from_sse(
                            fold_base[fold]["sse"], fold_base[fold]["rows"]
                        )
                        gain = candidate_base_rmse - candidate_rmse
                        fold_gains.append(gain)
                        fold_rows.append(
                            {
                                "base_col": base_col,
                                "zone_scope": PRIMARY_ZONE,
                                "uncertainty_ft": uncertainty,
                                "sides": sides,
                                "shrink": shrink,
                                "move_cap_ft": cap_label,
                                "fold": fold,
                                "rows": fold_base[fold]["rows"],
                                "base_rmse": candidate_base_rmse,
                                "rmse": candidate_rmse,
                                "gain": gain,
                            }
                        )

                    top_local = top40_mask[modified_indices]
                    top_candidate_sse = top40_base_sse + float(delta_sse[top_local].sum())
                    top40_rmse = _rmse_from_sse(top_candidate_sse, top40_rows)

                    crossing_local = crossing_well[modified_indices]
                    cross_delta = float(delta_sse[crossing_local].sum())
                    noncross_delta = float(delta_sse[~crossing_local].sum())
                    cross_gain = _rmse_from_sse(cross_base_sse, cross_rows) - _rmse_from_sse(
                        cross_base_sse + cross_delta, cross_rows
                    )
                    noncross_gain = _rmse_from_sse(noncross_base_sse, noncross_rows) - _rmse_from_sse(
                        noncross_base_sse + noncross_delta, noncross_rows
                    )

                    abs_move = np.abs(modified_move)
                    scan_rows.append(
                        {
                            "base_col": base_col,
                            "zone_scope": PRIMARY_ZONE,
                            "uncertainty_ft": uncertainty,
                            "sides": sides,
                            "shrink": shrink,
                            "move_cap_ft": cap_label,
                            "base_rmse": base_rmse,
                            "rmse": score,
                            "gain": base_rmse - score,
                            "mean_fold_gain": float(np.mean(fold_gains)),
                            "min_fold_gain": float(np.min(fold_gains)),
                            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
                            "top40_base_rmse": top40_base_rmse,
                            "top40_rmse": top40_rmse,
                            "top40_gain": top40_base_rmse - top40_rmse,
                            "modified_wells": int(np.unique(well_codes[modified_indices]).size),
                            "modified_rows": int(len(modified_indices)),
                            "modified_row_fraction": float(len(modified_indices) / len(pool)),
                            "mean_abs_move_modified": float(abs_move.mean()) if len(abs_move) else 0.0,
                            "p95_abs_move_modified": float(np.quantile(abs_move, 0.95)) if len(abs_move) else 0.0,
                            "max_abs_move": float(abs_move.max()) if len(abs_move) else 0.0,
                            "crossing_well_gain": cross_gain,
                            "noncrossing_well_gain": noncross_gain,
                        }
                    )

scan = pd.DataFrame(scan_rows)
fold_report = pd.DataFrame(fold_rows)
if len(scan) == 0:
    raise RuntimeError("empty corridor scan")

scan["strict_gate"] = (
    (scan["gain"] >= ORACLE_GAIN_GATE)
    & (scan["positive_folds"] == N_FOLDS)
    & (scan["min_fold_gain"] > 0.0)
    & (scan["top40_gain"] >= TOP40_GAIN_GATE)
)
scan["robust_score"] = scan["gain"] + np.minimum(scan["min_fold_gain"], 0.0)
scan = scan.sort_values(["strict_gate", "robust_score", "gain"], ascending=False).reset_index(drop=True)
fold_report = fold_report.merge(
    scan[
        [
            "base_col",
            "zone_scope",
            "uncertainty_ft",
            "sides",
            "shrink",
            "move_cap_ft",
            "strict_gate",
        ]
    ],
    on=["base_col", "zone_scope", "uncertainty_ft", "sides", "shrink", "move_cap_ft"],
    how="left",
)

scan.to_csv(OUT_DIR / "corridor_oracle_scan.csv", index=False)
fold_report.to_csv(OUT_DIR / "corridor_oracle_folds.csv", index=False)

best_by_base = (
    scan.sort_values(["strict_gate", "robust_score", "gain"], ascending=False)
    .groupby("base_col", as_index=False)
    .first()
    .sort_values("gain", ascending=False)
)
best_by_base.to_csv(OUT_DIR / "best_by_base.csv", index=False)

print("\nBEST BY BASE")
display(best_by_base)
print("\nTOP 30 CONFIGS")
display(scan.head(30))

passing = scan[scan["strict_gate"]].copy()
verdict = {
    "pool": str(POOL_PATH),
    "rows": int(len(pool)),
    "wells": int(pool["well"].nunique()),
    "primary_zone": PRIMARY_ZONE,
    "primary_zone_wells": int(well_meta["zone"].eq(PRIMARY_ZONE).sum()),
    "primary_zone_crossing_rate": float(
        well_meta.loc[well_meta["zone"].eq(PRIMARY_ZONE), "crosses_corridor"].mean()
    ),
    "overall_zero_cross_rate": float((~well_meta["crosses_corridor"]).mean()),
    "passing_configs": int(len(passing)),
    "best_gain": float(scan.iloc[0]["gain"]),
    "best_top40_gain": float(scan.iloc[0]["top40_gain"]),
    "strict_pass": bool(len(passing) > 0),
    "next": (
        "PASS -> train leave-typewell-out EGFDL/BUDA top locator"
        if len(passing) > 0
        else "FAIL -> stop formation-corridor route"
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, ensure_ascii=True)

print("\nVERDICT", verdict)
print("saved:", OUT_DIR)
