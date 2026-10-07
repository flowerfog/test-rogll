"""Nested OOF audit for a distance-ramped PF branch hedge.

The source 6.594 recipe is kept unchanged at well level:
  alpha=0.60, cap=2 ft, minor mass >=0.25, separation in [4, 40] ft.

Only its row dose changes. The pre-registered primary leaves the first 1000 ft
untouched, ramps linearly from 1000 to 2000 ft, and uses the full source shift
beyond 2000 ft. This follows directly from the v1 distance table: the constant
shift helped only beyond 2000 ft and hurt all earlier strata.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
STATISTICS_PATH = (
    OOF_ROOT / "pf_bimodal_midpoint_hedge_oof_v0" / "bimodal_well_statistics.csv"
)
GEOMETRY_PATH = (
    OOF_ROOT
    / "neighbor_fullcurve_projection_direction_v1"
    / "primary_candidate_oof.npz"
)
NEIGHBOR_SOURCE = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
OUT_DIR = OOF_ROOT / "pf_bimodal_far_ramp_mainline_audit_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_OOF_PATH = os.environ.get("ROGII_GOLD_OOF_PATH", "").strip()

NEIGHBOR_WEIGHT = 0.25
NEIGHBOR_RAW_CAP_FT = 24.0
MIN_ABS_DIRECTION_DOT = 0.70

BRANCH_ALPHA = 0.60
BRANCH_CAP_FT = 2.0
BRANCH_MINOR_MASS = 0.25
BRANCH_SEPARATION_LOW_FT = 4.0
BRANCH_SEPARATION_HIGH_FT = 40.0

PRIMARY_ID = "ramp_s1000_w1000_p1"
CONSERVATIVE_ID = "ramp_s1500_w1000_p1"
STARTS_FT = (500.0, 1000.0, 1500.0, 2000.0)
WIDTHS_FT = (250.0, 500.0, 1000.0, 1500.0)
POWERS = (1.0, 2.0)

BOOTSTRAP_ITERATIONS = 4000
SHUFFLE_ITERATIONS = 2000
SEED = 20260722


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _rmse(sse, rows):
    return float(math.sqrt(float(sse) / max(int(rows), 1)))


def _load_pool():
    header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
    requested = [
        "id",
        "well",
        "row_idx",
        "fold",
        "tvt_true",
        "cp12",
        "md_since",
    ]
    frame = pd.read_csv(
        POOL_PATH,
        usecols=[column for column in requested if column in header],
        dtype={"id": "string", "well": "string"},
        low_memory=False,
    )
    required = {"id", "tvt_true", "cp12"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"pool missing {sorted(required - set(frame.columns))}")
    if "well" not in frame:
        frame["well"] = frame["id"].str.rsplit("_", n=1).str[0]
    frame["well"] = frame["well"].astype(str)
    if "row_idx" not in frame:
        frame["row_idx"] = frame["id"].str.rsplit("_", n=1).str[-1].astype(int)
    if "fold" not in frame:
        metadata = pd.read_csv(FOLD_PATH, dtype={"well": str})
        fold_map = metadata.groupby("well")["fold"].first().astype(int).to_dict()
        frame["fold"] = frame["well"].map(fold_map)
    for column in ("tvt_true", "cp12", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"non-finite {column}")
    if "md_since" in frame:
        frame["md_since"] = pd.to_numeric(frame["md_since"], errors="coerce")
    frame["fold"] = frame["fold"].astype(int)
    return frame.reset_index(drop=True)


def _ensure_md_since(pool):
    if "md_since" in pool:
        existing = pd.to_numeric(pool["md_since"], errors="coerce").to_numpy(float)
        if np.isfinite(existing).all():
            return existing

    output = np.full(len(pool), np.nan, np.float64)
    grouped = pool.groupby("well", sort=False).indices
    for number, (well, indices) in enumerate(grouped.items(), 1):
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        horizontal = pd.read_csv(path, usecols=["MD", "TVT_input"])
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        known = horizontal["TVT_input"].notna().to_numpy(bool)
        if not known.any():
            raise RuntimeError(f"{well}: no visible TVT_input rows")
        anchor = int(np.flatnonzero(known)[-1])
        row_indices = pool.loc[indices, "row_idx"].to_numpy(int)
        if row_indices.min() < 0 or row_indices.max() >= len(horizontal):
            raise RuntimeError(f"{well}: pool row index outside horizontal file")
        output[indices] = md[row_indices] - md[anchor]
        if number % 100 == 0 or number == len(grouped):
            print(f"reconstruct md_since {number}/{len(grouped)}", flush=True)

    if not np.isfinite(output).all():
        raise RuntimeError("failed to reconstruct complete md_since")
    return output


def _load_gold(pool):
    if not GOLD_OOF_PATH:
        return None
    path = Path(GOLD_OOF_PATH)
    if not path.exists():
        raise FileNotFoundError(path)
    endpoint = pd.read_csv(path, dtype={"id": "string"})
    if not {"id", "tvt"}.issubset(endpoint.columns):
        raise RuntimeError("Gold OOF must contain id,tvt")
    endpoint = endpoint[["id", "tvt"]].drop_duplicates("id", keep="last")
    joined = pool[["id"]].merge(endpoint, on="id", how="left", validate="one_to_one")
    values = pd.to_numeric(joined["tvt"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Gold OOF does not cover the complete pool")
    return values


def _eligible_neighbor_rows(pool):
    path = NEIGHBOR_SOURCE / "selected_neighbors.csv"
    selected = pd.read_csv(path, dtype={"well": str}).set_index("well")
    direction = "abs_direction_dot" if "abs_direction_dot" in selected else "direction_dot"
    required = {"median_distance", "p90_distance", "anchor_distance_full", direction}
    if not required.issubset(selected.columns):
        raise RuntimeError(f"neighbor metadata missing {sorted(required - set(selected.columns))}")
    eligible = (
        (selected["median_distance"] <= 600.0)
        & (selected["p90_distance"] <= 1200.0)
        & (selected["anchor_distance_full"] <= 1200.0)
        & (selected[direction].abs() >= MIN_ABS_DIRECTION_DOT)
    )
    return pool["well"].map(eligible.to_dict()).fillna(False).to_numpy(bool)


def _neighbor_prediction(upstream, candidate, eligible):
    move = np.zeros(len(upstream), np.float64)
    valid = eligible & np.isfinite(candidate)
    move[valid] = NEIGHBOR_WEIGHT * np.clip(
        candidate[valid] - upstream[valid],
        -NEIGHBOR_RAW_CAP_FT,
        NEIGHBOR_RAW_CAP_FT,
    )
    return upstream + move


def _dose(md_since, config):
    if config["id"] == "constant_source":
        return np.ones(len(md_since), np.float64)
    width = max(float(config["width_ft"]), 1e-6)
    value = np.clip((md_since - float(config["start_ft"])) / width, 0.0, 1.0)
    return value ** float(config["power"])


def _report(pool, truth, prediction, base, md_since, well_codes, wells, well_folds):
    error = prediction - truth
    base_error = base - truth
    rows_by_well = np.bincount(well_codes, minlength=len(wells)).astype(np.int64)
    sse_by_well = np.bincount(
        well_codes, weights=error * error, minlength=len(wells)
    )
    base_sse_by_well = np.bincount(
        well_codes, weights=base_error * base_error, minlength=len(wells)
    )
    report = {
        "rows": int(len(error)),
        "wells": int(len(wells)),
        "exact_sse": float(sse_by_well.sum()),
        "rmse": _rmse(sse_by_well.sum(), rows_by_well.sum()),
        "base_rmse": _rmse(base_sse_by_well.sum(), rows_by_well.sum()),
    }
    report["gain"] = report["base_rmse"] - report["rmse"]
    fold_gains = []
    for fold in sorted(np.unique(well_folds)):
        mask = well_folds == fold
        fold_rows = int(rows_by_well[mask].sum())
        fold_base = _rmse(base_sse_by_well[mask].sum(), fold_rows)
        fold_score = _rmse(sse_by_well[mask].sum(), fold_rows)
        gain = fold_base - fold_score
        report[f"fold{int(fold)}_gain"] = float(gain)
        fold_gains.append(gain)
    report["min_fold_gain"] = float(np.min(fold_gains))
    report["positive_folds"] = int(np.sum(np.asarray(fold_gains) > 0.0))
    strata = {
        "near_0_500": md_since <= 500.0,
        "mid_500_1000": (md_since > 500.0) & (md_since <= 1000.0),
        "mid_1000_2000": (md_since > 1000.0) & (md_since <= 2000.0),
        "far_gt2000": md_since > 2000.0,
    }
    for name, mask in strata.items():
        base_score = float(np.sqrt(np.mean(base_error[mask] ** 2)))
        score = float(np.sqrt(np.mean(error[mask] ** 2)))
        report[f"{name}_rmse"] = score
        report[f"{name}_base_rmse"] = base_score
        report[f"{name}_gain"] = base_score - score
    return report, sse_by_well, base_sse_by_well, rows_by_well


for path in (POOL_PATH, FOLD_PATH, STATISTICS_PATH, GEOMETRY_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

started = time.time()
pool = _load_pool()
truth = pool["tvt_true"].to_numpy(np.float64)
md_since = _ensure_md_since(pool)
pool["md_since"] = md_since
cp12 = pool["cp12"].to_numpy(np.float64)
gold = _load_gold(pool)
if gold is None:
    upstream = cp12
    upstream_name = "cp12_surrogate_no_gold"
    exact_mainline = False
else:
    upstream = gold
    upstream_name = "cp12_gold_oof"
    exact_mainline = True

with np.load(GEOMETRY_PATH, allow_pickle=False) as payload:
    candidate = payload["candidate"].astype(np.float64)
if len(candidate) != len(pool):
    raise RuntimeError("geometry candidate row mismatch")
eligible = _eligible_neighbor_rows(pool)
base = _neighbor_prediction(upstream, candidate, eligible)

wells = sorted(pool["well"].unique())
well_to_code = {well: index for index, well in enumerate(wells)}
well_codes = pool["well"].map(well_to_code).to_numpy(np.int32)
fold_map = pool.groupby("well")["fold"].first().astype(int).to_dict()
well_folds = np.asarray([fold_map[well] for well in wells], dtype=np.int16)

statistics = pd.read_csv(STATISTICS_PATH, dtype={"well": str})
statistics = statistics.drop_duplicates("well", keep="last").set_index("well")
required_stats = {"minor_mass", "separation_ft", "midpoint_delta_ft"}
if not required_stats.issubset(statistics.columns):
    raise RuntimeError(f"statistics missing {sorted(required_stats - set(statistics.columns))}")
statistics = statistics.reindex(wells)
if statistics[list(required_stats)].isna().any().any():
    raise RuntimeError("branch statistics do not cover all wells")

qualified = (
    (statistics["minor_mass"].to_numpy(float) >= BRANCH_MINOR_MASS)
    & (statistics["separation_ft"].to_numpy(float) >= BRANCH_SEPARATION_LOW_FT)
    & (statistics["separation_ft"].to_numpy(float) <= BRANCH_SEPARATION_HIGH_FT)
)
well_shift = np.zeros(len(wells), np.float64)
proposed = BRANCH_ALPHA * statistics["midpoint_delta_ft"].to_numpy(float)
well_shift[qualified] = np.clip(
    proposed[qualified], -BRANCH_CAP_FT, BRANCH_CAP_FT
)
well_shift[np.abs(well_shift) < 0.01] = 0.0
row_shift = well_shift[well_codes]

configs = [
    {"id": "constant_source", "start_ft": 0.0, "width_ft": 0.0, "power": 0.0}
]
for start_ft in STARTS_FT:
    for width_ft in WIDTHS_FT:
        for power in POWERS:
            configs.append(
                {
                    "id": f"ramp_s{int(start_ft)}_w{int(width_ft)}_p{int(power)}",
                    "start_ft": start_ft,
                    "width_ft": width_ft,
                    "power": power,
                }
            )

scan_rows = []
config_sse = {}
base_sse_by_well = None
rows_by_well = None
for config in configs:
    dose = _dose(md_since, config)
    prediction = base + row_shift * dose
    report, sse, base_sse, row_counts = _report(
        pool, truth, prediction, base, md_since, well_codes, wells, well_folds
    )
    scan_rows.append({**config, **report})
    config_sse[config["id"]] = sse
    base_sse_by_well = base_sse
    rows_by_well = row_counts

scan = pd.DataFrame(scan_rows).sort_values(
    ["gain", "start_ft", "width_ft"], ascending=[False, False, False]
)
scan.to_csv(OUT_DIR / "ramp_scan.csv", index=False)
config_by_id = {config["id"]: config for config in configs}
report_by_id = scan.set_index("id").to_dict(orient="index")

if PRIMARY_ID not in report_by_id or CONSERVATIVE_ID not in report_by_id:
    raise RuntimeError("pre-registered ramp config missing")
primary_config = config_by_id[PRIMARY_ID]
primary_report = {"id": PRIMARY_ID, **report_by_id[PRIMARY_ID]}
constant_report = {"id": "constant_source", **report_by_id["constant_source"]}
conservative_report = {"id": CONSERVATIVE_ID, **report_by_id[CONSERVATIVE_ID]}

# Nested selection includes a no-op. The validation fold never selects its own
# row-dose shape.
folds = sorted(np.unique(well_folds).tolist())
nested_rows = []
nested_sse_by_well = base_sse_by_well.copy()
for held_fold in folds:
    train_mask = well_folds != held_fold
    valid_mask = ~train_mask
    train_rows = int(rows_by_well[train_mask].sum())
    train_base = _rmse(base_sse_by_well[train_mask].sum(), train_rows)
    choices = [
        {
            "id": "no_op",
            "train_gain": 0.0,
            "start_ft": np.inf,
            "width_ft": np.inf,
            "power": 1.0,
        }
    ]
    for config in configs:
        score = _rmse(config_sse[config["id"]][train_mask].sum(), train_rows)
        choices.append({**config, "train_gain": train_base - score})
    selected = max(
        choices,
        key=lambda row: (
            row["train_gain"],
            row["start_ft"],
            row["width_ft"],
            row["power"],
        ),
    )
    if selected["id"] != "no_op":
        nested_sse_by_well[valid_mask] = config_sse[selected["id"]][valid_mask]
    valid_rows = int(rows_by_well[valid_mask].sum())
    valid_base = _rmse(base_sse_by_well[valid_mask].sum(), valid_rows)
    valid_score = _rmse(nested_sse_by_well[valid_mask].sum(), valid_rows)
    nested_rows.append(
        {
            "held_fold": int(held_fold),
            **selected,
            "valid_base_rmse": valid_base,
            "valid_rmse": valid_score,
            "valid_gain": valid_base - valid_score,
        }
    )
nested = pd.DataFrame(nested_rows)
nested.to_csv(OUT_DIR / "nested_ramp_folds.csv", index=False)
nested_rows_total = int(rows_by_well.sum())
nested_base_rmse = _rmse(base_sse_by_well.sum(), nested_rows_total)
nested_rmse = _rmse(nested_sse_by_well.sum(), nested_rows_total)
nested_report = {
    "rows": nested_rows_total,
    "base_rmse": nested_base_rmse,
    "rmse": nested_rmse,
    "gain": nested_base_rmse - nested_rmse,
    "positive_folds": int((nested["valid_gain"] > 0.0).sum()),
    "min_fold_gain": float(nested["valid_gain"].min()),
}

# Well bootstrap for the pre-registered primary.
rng = np.random.default_rng(SEED)
primary_sse_by_well = config_sse[PRIMARY_ID]
bootstrap_gain = np.empty(BOOTSTRAP_ITERATIONS, np.float64)
for iteration in range(BOOTSTRAP_ITERATIONS):
    selected = rng.integers(0, len(wells), size=len(wells))
    sampled_rows = int(rows_by_well[selected].sum())
    bootstrap_gain[iteration] = _rmse(
        base_sse_by_well[selected].sum(), sampled_rows
    ) - _rmse(primary_sse_by_well[selected].sum(), sampled_rows)

# Matched control: preserve the exact branch-shift distribution but assign it
# to different wells. Quadratic SSE coefficients avoid 2000 full-row passes.
primary_dose = _dose(md_since, primary_config)
base_error = base - truth
linear_by_well = np.bincount(
    well_codes,
    weights=base_error * primary_dose,
    minlength=len(wells),
)
quadratic_by_well = np.bincount(
    well_codes,
    weights=primary_dose * primary_dose,
    minlength=len(wells),
)
base_total_sse = float(base_sse_by_well.sum())
total_rows = int(rows_by_well.sum())
base_rmse = _rmse(base_total_sse, total_rows)
shuffle_gain = np.empty(SHUFFLE_ITERATIONS, np.float64)
for iteration in range(SHUFFLE_ITERATIONS):
    shuffled_shift = rng.permutation(well_shift)
    shuffled_sse = (
        base_total_sse
        + 2.0 * float(np.sum(shuffled_shift * linear_by_well))
        + float(np.sum(shuffled_shift * shuffled_shift * quadratic_by_well))
    )
    shuffle_gain[iteration] = base_rmse - _rmse(shuffled_sse, total_rows)

primary_prediction = base + row_shift * primary_dose
primary_error = primary_prediction - truth
primary_sse_by_well_check = np.bincount(
    well_codes, weights=primary_error * primary_error, minlength=len(wells)
)
if not np.allclose(primary_sse_by_well_check, primary_sse_by_well, atol=1e-5):
    raise RuntimeError("primary SSE reconstruction mismatch")

bywell = pd.DataFrame(
    {
        "well": wells,
        "fold": well_folds,
        "rows": rows_by_well,
        "base_sse": base_sse_by_well,
        "primary_sse": primary_sse_by_well,
        "branch_shift_ft": well_shift,
    }
)
bywell["base_rmse"] = np.sqrt(bywell["base_sse"] / bywell["rows"])
bywell["primary_rmse"] = np.sqrt(bywell["primary_sse"] / bywell["rows"])
bywell["gain"] = bywell["base_rmse"] - bywell["primary_rmse"]
bywell.sort_values("base_rmse", ascending=False).to_csv(
    OUT_DIR / "primary_bywell.csv", index=False
)

gates = {
    "primary_gain_ge": 0.005,
    "primary_min_fold_gain_ge": -0.01,
    "primary_positive_folds_ge": 4,
    "near_degradation_le": 0.001,
    "mid_500_1000_degradation_le": 0.001,
    "far_gain_gt": 0.0,
    "bootstrap_p05_gain_gt": 0.0,
}
checks = {
    "primary_gain": bool(primary_report["gain"] >= gates["primary_gain_ge"]),
    "primary_min_fold_gain": bool(
        primary_report["min_fold_gain"] >= gates["primary_min_fold_gain_ge"]
    ),
    "primary_positive_folds": bool(
        primary_report["positive_folds"] >= gates["primary_positive_folds_ge"]
    ),
    "near_guard": bool(
        -primary_report["near_0_500_gain"] <= gates["near_degradation_le"]
    ),
    "mid_500_1000_guard": bool(
        -primary_report["mid_500_1000_gain"]
        <= gates["mid_500_1000_degradation_le"]
    ),
    "far_gain": bool(primary_report["far_gt2000_gain"] > gates["far_gain_gt"]),
    "bootstrap": bool(
        np.quantile(bootstrap_gain, 0.05) > gates["bootstrap_p05_gain_gt"]
    ),
    "matched_shuffle": bool(
        primary_report["gain"] > np.quantile(shuffle_gain, 0.95)
    ),
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "pf_bimodal_far_ramp_mainline_audit_v2",
    "stage": "foldclean_distance_ramped_source_branch_audit",
    "upstream_endpoint": upstream_name,
    "exact_cp12_gold_neighbor_chain": exact_mainline,
    "gold_oof_path": GOLD_OOF_PATH or None,
    "rows": len(pool),
    "wells": len(wells),
    "qualified_wells": int(np.sum(np.abs(well_shift) > 0.0)),
    "source_branch_recipe": {
        "alpha": BRANCH_ALPHA,
        "cap_ft": BRANCH_CAP_FT,
        "minor_mass": BRANCH_MINOR_MASS,
        "separation_low_ft": BRANCH_SEPARATION_LOW_FT,
        "separation_high_ft": BRANCH_SEPARATION_HIGH_FT,
    },
    "constant_source": constant_report,
    "pre_registered_primary": primary_report,
    "pre_registered_conservative": conservative_report,
    "best_same_data_diagnostic": scan.iloc[0].to_dict(),
    "nested": {"result": nested_report, "folds": nested_rows},
    "bootstrap": {
        "iterations": BOOTSTRAP_ITERATIONS,
        "p01_gain": float(np.quantile(bootstrap_gain, 0.01)),
        "p05_gain": float(np.quantile(bootstrap_gain, 0.05)),
        "p50_gain": float(np.quantile(bootstrap_gain, 0.50)),
        "p95_gain": float(np.quantile(bootstrap_gain, 0.95)),
        "positive_probability": float(np.mean(bootstrap_gain > 0.0)),
    },
    "matched_shuffle": {
        "iterations": SHUFFLE_ITERATIONS,
        "p50_gain": float(np.quantile(shuffle_gain, 0.50)),
        "p95_gain": float(np.quantile(shuffle_gain, 0.95)),
        "primary_gain_gt_shuffle_p95": bool(
            primary_report["gain"] > np.quantile(shuffle_gain, 0.95)
        ),
    },
    "gates": gates,
    "checks": checks,
    "strict_pass": strict_pass,
    "interpretation": (
        "Exact CP12+Gold+Neighbor OOF"
        if exact_mainline
        else "CP12+Neighbor surrogate; Gold OOF is absent"
    ),
    "next": (
        "PASS -> use the pre-registered ramp in one isolated hidden-LB A/B"
        if strict_pass
        else "FAIL -> close PF branch hedging on the 6.781 mainline"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

scan.to_csv(OUT_DIR / "ramp_scan.csv", index=False)
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nTOP RAMP SCAN")
print(scan.head(30).to_string(index=False))
print("\nNESTED RAMP FOLDS")
print(nested.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT_DIR)
