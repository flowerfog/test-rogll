"""Prefix-verified full-curve Neighbor adaptation for the complete 6.39 tail.

The original Neighbor candidate is preserved.  This audit changes only how it
is consumed by the newest deployable 6.39 parent:

* the old geometry-only recipe is evaluated unchanged;
* a new gate checks how well the transferred reference reconstructs the
  target's legally visible prefix;
* Fold 3 selects the gate/dose, Fold 2 confirms it, and Fold 1 is final.

No hidden prediction is read and no target suffix truth participates in
reference selection, prefix scoring, or candidate construction.
"""

from pathlib import Path
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
FULL_PATH = (
    OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18"
    / "full_oof_cp_endpoints.parquet"
)
NEIGHBOR_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
NEIGHBOR_BANK = NEIGHBOR_DIR / "neighbor_transfer_oof.npz"
NEIGHBOR_META = NEIGHBOR_DIR / "selected_neighbors.csv"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
OUT_DIR = OOF_ROOT / "frontier_6p39_prefix_verified_neighbor_v24"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_COLUMN = "full_parent_matched_vp_replay"
NEIGHBOR_METHOD = "structure_progress"
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
PREFIX_WINDOWS_FT = (256.0, 512.0, 1024.0)
PREFIX_RMSE_LIMITS = (2.0, 4.0, 6.0, 8.0, 12.0, np.inf)
PREFIX_COVERAGE_LIMITS = (0.50, 0.80)
DISAGREEMENT_LIMITS = (6.0, 12.0, 24.0, np.inf)
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50)
CAPS_FT = (6.0, 12.0, 24.0, np.inf)
GEOMETRY_GATES = {
    "coherent150": dict(median=150.0, p90=300.0, anchor=300.0),
    "coherent300": dict(median=300.0, p90=600.0, anchor=600.0),
    "coherent600": dict(median=600.0, p90=1200.0, anchor=1200.0),
}
MIN_ABS_DIRECTION_DOT = 0.70
OLD_RECIPE = {
    "geometry_gate": "coherent600",
    "prefix_window_ft": 512.0,
    "prefix_rmse_limit": np.inf,
    "prefix_coverage_limit": 0.0,
    "disagreement_limit": np.inf,
    "weight": 0.35,
    "cap_ft": np.inf,
}
SEED = 20260725
started = time.time()


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    well_rmse = np.sqrt(
        pd.Series(error * error).groupby(frame["well"], sort=False).mean().to_numpy()
    )
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, mask in {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _cumulative_xy(xy):
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    step[~np.isfinite(step)] = 0.0
    return np.r_[0.0, np.cumsum(step)]


def _unique_axis(axis, values):
    keep = np.r_[np.diff(axis) > 1e-8, True]
    return axis[keep], values[keep]


def _first_file(paths, label):
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {label}: {[str(path) for path in paths]}")
    return path


for required in (FULL_PATH, NEIGHBOR_BANK, NEIGHBOR_META):
    if not required.is_file():
        raise FileNotFoundError(required)
pool_path = _first_file(POOL_CANDIDATES, "candidate pool aligned to Neighbor bank")

frame = pd.read_parquet(
    FULL_PATH,
    columns=["id", "well", "row_idx", "fold", "tvt_true", "md_since", PARENT_COLUMN],
)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in full 6.39 parent")

pool_header = pd.read_csv(pool_path, nrows=0).columns.tolist()
pool_columns = [column for column in ("id", "well", "row_idx") if column in pool_header]
if "id" not in pool_columns:
    raise RuntimeError("Candidate pool has no id")
pool = pd.read_csv(pool_path, usecols=pool_columns, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

with np.load(NEIGHBOR_BANK, allow_pickle=False) as handle:
    if NEIGHBOR_METHOD not in handle.files:
        raise RuntimeError(f"Neighbor bank lacks {NEIGHBOR_METHOD}")
    bank_candidate = handle[NEIGHBOR_METHOD].astype(np.float64)
if len(bank_candidate) != len(pool):
    raise RuntimeError("Neighbor bank row count differs from candidate pool")

pool_position = pd.Series(np.arange(len(pool), dtype=np.int64), index=pool["id"])
position = frame["id"].map(pool_position)
if position.isna().any():
    raise RuntimeError(f"Neighbor bank misses {int(position.isna().sum())} full-parent rows")
neighbor = bank_candidate[position.to_numpy(np.int64)]

meta = pd.read_csv(NEIGHBOR_META, dtype={"well": str, "neighbor": str})
needed_meta = {
    "well", "neighbor", "median_distance", "p90_distance",
    "anchor_distance_full", "abs_direction_dot",
}
missing_meta = sorted(needed_meta.difference(meta.columns))
if missing_meta:
    raise RuntimeError(f"Neighbor metadata missing {missing_meta}")
meta = meta.drop_duplicates("well").set_index("well")
missing_meta_wells = sorted(set(frame["well"].unique()).difference(meta.index))
if missing_meta_wells:
    raise RuntimeError(
        f"Neighbor metadata misses {len(missing_meta_wells)} OOF wells: "
        f"{missing_meta_wells[:10]}"
    )

parent = frame[PARENT_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}


def _load_curve(well):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    local = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
    values = local.apply(pd.to_numeric, errors="coerce")
    xy = values[["X", "Y"]].to_numpy(np.float64)
    z = values["Z"].to_numpy(np.float64)
    tvt = values["TVT"].to_numpy(np.float64)
    tvt_input = values["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known) or known[-1] >= len(local) - 1:
        raise RuntimeError(f"{well}: invalid visible-prefix mask")
    if not np.isfinite(xy).all() or not np.isfinite(z).all() or not np.isfinite(tvt).all():
        raise RuntimeError(f"{well}: non-finite geometry or truth")
    anchor = int(known[-1])
    progress = _cumulative_xy(xy)
    direction = xy[-1] - xy[anchor]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0:
        raise RuntimeError(f"{well}: degenerate direction")
    direction /= norm
    unique_progress, unique_structure = _unique_axis(progress, tvt + z)
    return {
        "xy": xy,
        "z": z,
        "tvt": tvt,
        "tvt_input": tvt_input,
        "anchor": anchor,
        "progress": progress,
        "direction": direction,
        "structure": tvt + z,
        "unique_progress": unique_progress,
        "unique_structure": unique_structure,
        "lateral_tree": cKDTree(xy[anchor:], leafsize=64),
    }


curve_cache = {}


def _curve(well):
    if well not in curve_cache:
        curve_cache[well] = _load_curve(well)
    return curve_cache[well]


def _prefix_diagnostics(target_well, reference_well):
    target = _curve(target_well)
    reference = _curve(reference_well)
    anchor = target["anchor"]
    _, ref_local = reference["lateral_tree"].query(target["xy"][anchor], k=1)
    ref_anchor = reference["anchor"] + int(ref_local)
    sign = 1.0 if np.dot(target["direction"], reference["direction"]) >= 0.0 else -1.0
    relative_progress = target["progress"] - target["progress"][anchor]
    query = reference["progress"][ref_anchor] + sign * relative_progress
    valid = (
        (query >= reference["unique_progress"][0])
        & (query <= reference["unique_progress"][-1])
    )
    mapped_structure = np.full(len(query), np.nan, np.float64)
    mapped_structure[valid] = np.interp(
        query[valid], reference["unique_progress"], reference["unique_structure"]
    )
    ref_anchor_structure = float(reference["structure"][ref_anchor])
    target_anchor_tvt = float(target["tvt_input"][anchor])
    shift = target_anchor_tvt - (ref_anchor_structure - float(target["z"][anchor]))
    prediction = mapped_structure - target["z"] + shift

    rows = {"well": target_well, "neighbor": reference_well, "direction_sign": sign}
    prefix_distance = target["progress"][anchor] - target["progress"]
    known = np.isfinite(target["tvt_input"]) & (np.arange(len(query)) < anchor)
    for window in PREFIX_WINDOWS_FT:
        selected = known & (prefix_distance <= window) & (prefix_distance >= 8.0)
        valid_selected = selected & np.isfinite(prediction)
        coverage = float(valid_selected.sum() / max(1, selected.sum()))
        if valid_selected.sum() >= 16:
            error = prediction[valid_selected] - target["tvt_input"][valid_selected]
            rmse = float(np.sqrt(np.mean(error * error)))
            mae = float(np.mean(np.abs(error)))
        else:
            rmse = np.inf
            mae = np.inf
        rows[f"prefix_rmse_{int(window)}"] = rmse
        rows[f"prefix_mae_{int(window)}"] = mae
        rows[f"prefix_coverage_{int(window)}"] = coverage
        rows[f"prefix_rows_{int(window)}"] = int(valid_selected.sum())
    return rows


prefix_rows = []
prefix_failures = []
well_list = sorted(groups)
for number, well in enumerate(well_list, 1):
    try:
        reference_well = str(meta.loc[well, "neighbor"])
        prefix_rows.append(_prefix_diagnostics(well, reference_well))
    except Exception as exc:
        prefix_failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == len(well_list):
        print(
            f"PREFIX-NEIGHBOR diagnostics {number}/{len(well_list)} "
            f"failures={len(prefix_failures)}",
            flush=True,
        )
prefix = pd.DataFrame(prefix_rows).set_index("well")
pd.DataFrame(prefix_rows).to_csv(OUT_DIR / "prefix_diagnostics.csv", index=False)
pd.DataFrame(prefix_failures).to_csv(OUT_DIR / "prefix_failures.csv", index=False)
if len(prefix_failures) > 3:
    raise RuntimeError(f"Too many prefix diagnostic failures: {prefix_failures[:5]}")

# Build one row per complete well.  Suffix labels are used only to evaluate a
# frozen recipe; all eligibility columns are label-free.
well_table_rows = []
cap_stats = {cap: [] for cap in CAPS_FT}
for well, indices in groups.items():
    local_parent = parent[indices]
    local_truth = truth[indices]
    local_error = local_parent - local_truth
    local_candidate = neighbor[indices]
    raw_delta = np.where(np.isfinite(local_candidate), local_candidate - local_parent, 0.0)
    valid_fraction = float(np.isfinite(local_candidate).mean())
    local_distance = distance[indices]
    metadata = meta.loc[well]
    prefix_metadata = prefix.loc[well] if well in prefix.index else None
    base_row = {
        "well": well,
        "fold": int(fold[indices[0]]),
        "rows": int(len(indices)),
        "parent_sse": float(np.dot(local_error, local_error)),
        "neighbor_valid_fraction": valid_fraction,
        "mean_abs_disagreement": float(np.mean(np.abs(raw_delta))),
        "p90_abs_disagreement": float(np.quantile(np.abs(raw_delta), 0.90)),
        "median_distance": float(metadata["median_distance"]),
        "p90_distance": float(metadata["p90_distance"]),
        "anchor_distance_full": float(metadata["anchor_distance_full"]),
        "abs_direction_dot": float(metadata["abs_direction_dot"]),
    }
    for window in PREFIX_WINDOWS_FT:
        key = int(window)
        base_row[f"prefix_rmse_{key}"] = (
            float(prefix_metadata[f"prefix_rmse_{key}"]) if prefix_metadata is not None else np.inf
        )
        base_row[f"prefix_coverage_{key}"] = (
            float(prefix_metadata[f"prefix_coverage_{key}"]) if prefix_metadata is not None else 0.0
        )
    well_table_rows.append(base_row)
    for cap in CAPS_FT:
        delta = raw_delta.copy()
        if np.isfinite(cap):
            delta = np.clip(delta, -cap, cap)
        stat = {
            "well": well,
            "cross": float(np.dot(local_error, delta)),
            "delta2": float(np.dot(delta, delta)),
        }
        for name, mask in {
            "near": local_distance <= 500.0,
            "far": local_distance > 2000.0,
        }.items():
            stat[f"{name}_rows"] = int(mask.sum())
            stat[f"{name}_parent_sse"] = float(np.dot(local_error[mask], local_error[mask]))
            stat[f"{name}_cross"] = float(np.dot(local_error[mask], delta[mask]))
            stat[f"{name}_delta2"] = float(np.dot(delta[mask], delta[mask]))
        cap_stats[cap].append(stat)

well_table = pd.DataFrame(well_table_rows).set_index("well")
cap_tables = {
    cap: pd.DataFrame(rows).set_index("well").reindex(well_table.index)
    for cap, rows in cap_stats.items()
}
well_table.reset_index().to_csv(OUT_DIR / "well_features.csv", index=False)


def _geometry_mask(table, gate_name):
    limits = GEOMETRY_GATES[gate_name]
    return (
        (table["median_distance"].to_numpy() <= limits["median"])
        & (table["p90_distance"].to_numpy() <= limits["p90"])
        & (table["anchor_distance_full"].to_numpy() <= limits["anchor"])
        & (table["abs_direction_dot"].to_numpy() >= MIN_ABS_DIRECTION_DOT)
    )


def _eligible_wells(recipe, table=well_table):
    window = int(float(recipe["prefix_window_ft"]))
    return (
        _geometry_mask(table, str(recipe["geometry_gate"]))
        & (table[f"prefix_rmse_{window}"].to_numpy() <= float(recipe["prefix_rmse_limit"]))
        & (table[f"prefix_coverage_{window}"].to_numpy() >= float(recipe["prefix_coverage_limit"]))
        & (table["mean_abs_disagreement"].to_numpy() <= float(recipe["disagreement_limit"]))
        & (table["neighbor_valid_fraction"].to_numpy() >= 0.80)
    )


def _scan_score(recipe, held_fold):
    table = well_table
    fold_mask = table["fold"].to_numpy() == int(held_fold)
    eligible = _eligible_wells(recipe, table) & fold_mask
    stats = cap_tables[float(recipe["cap_ft"])]
    weight = float(recipe["weight"])
    base_sse = float(table.loc[fold_mask, "parent_sse"].sum())
    rows = int(table.loc[fold_mask, "rows"].sum())
    cross = float(stats.loc[eligible, "cross"].sum())
    delta2 = float(stats.loc[eligible, "delta2"].sum())
    sse = base_sse + 2.0 * weight * cross + weight * weight * delta2

    output = {
        "rows": rows,
        "wells": int(fold_mask.sum()),
        "selected_wells": int(eligible.sum()),
        "rmse": float(np.sqrt(max(sse, 0.0) / rows)),
        "parent_rmse": float(np.sqrt(base_sse / rows)),
    }
    for name in ("near", "far"):
        band_rows = int(stats.loc[fold_mask, f"{name}_rows"].sum())
        band_base = float(stats.loc[fold_mask, f"{name}_parent_sse"].sum())
        band_cross = float(stats.loc[eligible, f"{name}_cross"].sum())
        band_delta2 = float(stats.loc[eligible, f"{name}_delta2"].sum())
        band_sse = band_base + 2.0 * weight * band_cross + weight * weight * band_delta2
        output[f"{name}_rmse"] = float(np.sqrt(max(band_sse, 0.0) / band_rows))
        output[f"{name}_parent_rmse"] = float(np.sqrt(band_base / band_rows))
    output["gain"] = output["parent_rmse"] - output["rmse"]
    output["near_degradation"] = output["near_rmse"] - output["near_parent_rmse"]
    output["far_degradation"] = output["far_rmse"] - output["far_parent_rmse"]
    output["risk_pass"] = bool(
        output["selected_wells"] >= 5
        and output["near_degradation"] <= 0.05
        and output["far_degradation"] <= 0.05
    )
    output["selection_score"] = (
        output["rmse"]
        + 0.50 * max(output["near_degradation"], 0.0)
        + 0.25 * max(output["far_degradation"], 0.0)
    )
    return output


scan_rows = []
for values in itertools.product(
    GEOMETRY_GATES,
    PREFIX_WINDOWS_FT,
    PREFIX_RMSE_LIMITS,
    PREFIX_COVERAGE_LIMITS,
    DISAGREEMENT_LIMITS,
    WEIGHTS,
    CAPS_FT,
):
    recipe = dict(zip(
        (
            "geometry_gate", "prefix_window_ft", "prefix_rmse_limit",
            "prefix_coverage_limit", "disagreement_limit", "weight", "cap_ft",
        ),
        values,
    ))
    scan_rows.append({**recipe, **_scan_score(recipe, SELECTION_FOLD)})
scan = pd.DataFrame(scan_rows).sort_values(
    ["risk_pass", "selection_score", "rmse", "selected_wells"],
    ascending=[False, True, True, False],
    kind="stable",
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_prefix_neighbor_scan.csv", index=False)
selected = {
    key: scan.iloc[0][key].item() if hasattr(scan.iloc[0][key], "item") else scan.iloc[0][key]
    for key in (
        "geometry_gate", "prefix_window_ft", "prefix_rmse_limit",
        "prefix_coverage_limit", "disagreement_limit", "weight", "cap_ft",
    )
}


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction(recipe, eligibility_override=None):
    output = parent.copy()
    well_eligible = _eligible_wells(recipe) if eligibility_override is None else eligibility_override
    eligible_map = dict(zip(well_table.index, well_eligible))
    row_eligible = (
        frame["well"].map(eligible_map).fillna(False).to_numpy(dtype=bool, copy=True)
    )
    row_eligible &= np.isfinite(neighbor)
    move = neighbor[row_eligible] - parent[row_eligible]
    cap = float(recipe["cap_ft"])
    if np.isfinite(cap):
        move = np.clip(move, -cap, cap)
    output[row_eligible] += float(recipe["weight"]) * move
    return output, row_eligible


old_prediction, old_rows = _prediction(OLD_RECIPE)
new_prediction, new_rows = _prediction(selected)

summary_rows = []
for held_fold, role in (
    (SELECTION_FOLD, "selection"),
    (CONFIRMATION_FOLD, "confirmation"),
    (FINAL_FOLD, "final"),
):
    mask = fold == held_fold
    local = frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in {
        "parent": parent[mask],
        "old_geometry_neighbor": old_prediction[mask],
        "prefix_verified_neighbor": new_prediction[mask],
    }.items():
        summary_rows.append({"fold": held_fold, "role": role, **_metric(local, prediction, endpoint)})

primary_mask = np.isin(fold, (FINAL_FOLD, CONFIRMATION_FOLD))
primary_frame = frame.loc[primary_mask].reset_index(drop=True)
for endpoint, prediction in {
    "parent": parent[primary_mask],
    "old_geometry_neighbor": old_prediction[primary_mask],
    "prefix_verified_neighbor": new_prediction[primary_mask],
}.items():
    summary_rows.append({
        "fold": 0, "role": "fold1_plus_fold2_primary",
        **_metric(primary_frame, prediction, endpoint),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

# Shuffle only the prefix-quality measurements within each fold. Geometry,
# candidate curves, and current-parent disagreement remain fixed.
rng = np.random.default_rng(SEED)
shuffled_table = well_table.copy()
window = int(float(selected["prefix_window_ft"]))
for _, indices in shuffled_table.groupby("fold", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    for column in (f"prefix_rmse_{window}", f"prefix_coverage_{window}"):
        values = shuffled_table.iloc[indices][column].to_numpy()
        shuffled_table.iloc[indices, shuffled_table.columns.get_loc(column)] = rng.permutation(values)
shuffled_eligibility = _eligible_wells(selected, shuffled_table)
shuffled_prediction, _ = _prediction(selected, shuffled_eligibility)

control_rows = []
for held_fold, role in ((CONFIRMATION_FOLD, "confirmation"), (FINAL_FOLD, "final")):
    mask = fold == held_fold
    local = frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in {
        "parent": parent[mask],
        "prefix_verified_neighbor": new_prediction[mask],
        "shuffled_prefix_gate": shuffled_prediction[mask],
    }.items():
        control_rows.append({"fold": held_fold, "role": role, **_metric(local, prediction, endpoint)})
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)


def _value(fold_value, endpoint, column):
    row = summary[(summary["fold"].eq(fold_value)) & (summary["endpoint"].eq(endpoint))]
    return float(row.iloc[0][column])


old_gain = {
    held_fold: _value(held_fold, "parent", "rmse")
    - _value(held_fold, "old_geometry_neighbor", "rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD, 0)
}
new_gain = {
    held_fold: _value(held_fold, "parent", "rmse")
    - _value(held_fold, "prefix_verified_neighbor", "rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD, 0)
}
increment_over_old = {
    held_fold: _value(held_fold, "old_geometry_neighbor", "rmse")
    - _value(held_fold, "prefix_verified_neighbor", "rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD, 0)
}
new_far_change = {
    held_fold: _value(held_fold, "prefix_verified_neighbor", "far_gt2000_rmse")
    - _value(held_fold, "parent", "far_gt2000_rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
new_p95_change = {
    held_fold: _value(held_fold, "prefix_verified_neighbor", "p95_well_rmse")
    - _value(held_fold, "parent", "p95_well_rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}

old_rebase_gates = {
    "old_fold2_gain_ge_0p05": bool(old_gain[CONFIRMATION_FOLD] >= 0.05),
    "old_fold1_gain_ge_0p05": bool(old_gain[FINAL_FOLD] >= 0.05),
    "old_fold12_gain_ge_0p05": bool(old_gain[0] >= 0.05),
}
prefix_adaptation_gates = {
    "new_fold2_gain_ge_0p05": bool(new_gain[CONFIRMATION_FOLD] >= 0.05),
    "new_fold1_gain_ge_0p05": bool(new_gain[FINAL_FOLD] >= 0.05),
    "new_fold12_gain_ge_0p05": bool(new_gain[0] >= 0.05),
    "new_beats_old_fold2_ge_0p02": bool(increment_over_old[CONFIRMATION_FOLD] >= 0.02),
    "new_beats_old_fold1_ge_0p02": bool(increment_over_old[FINAL_FOLD] >= 0.02),
    "fold2_far_degradation_le_0p02": bool(new_far_change[CONFIRMATION_FOLD] <= 0.02),
    "fold1_far_degradation_le_0p02": bool(new_far_change[FINAL_FOLD] <= 0.02),
    "fold2_p95_degradation_le_0p10": bool(new_p95_change[CONFIRMATION_FOLD] <= 0.10),
    "fold1_p95_degradation_le_0p10": bool(new_p95_change[FINAL_FOLD] <= 0.10),
    "prefix_gate_is_active": bool(np.isfinite(float(selected["prefix_rmse_limit"]))),
}
old_rebase_pass = bool(all(old_rebase_gates.values()))
prefix_adaptation_pass = bool(all(prefix_adaptation_gates.values()))

row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["parent"] = parent.astype(np.float32)
row_output["old_geometry_neighbor"] = old_prediction.astype(np.float32)
row_output["prefix_verified_neighbor"] = new_prediction.astype(np.float32)
row_output["old_active"] = old_rows
row_output["prefix_active"] = new_rows
row_output.to_parquet(OUT_DIR / "neighbor_endpoints_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "frontier_6p39_prefix_verified_neighbor_v24",
    "stage": "Fold3_prefix_gate_selection_Fold2_confirmation_Fold1_final",
    "parent": PARENT_COLUMN,
    "neighbor_method": NEIGHBOR_METHOD,
    "hidden_predictions_observed": False,
    "target_suffix_truth_used_for_candidate_or_gate": False,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "Fold1_tuning_or_search_count": 0,
        "Fold2_tuning_or_search_count": 0,
    },
    "old_recipe": {
        key: _json_scalar(value) for key, value in OLD_RECIPE.items()
    },
    "selected_recipe": {
        key: _json_scalar(value) for key, value in selected.items()
    },
    "selection_search_count": int(len(scan)),
    "prefix_diagnostic_failures": prefix_failures,
    "old_gain": {str(key): float(value) for key, value in old_gain.items()},
    "new_gain": {str(key): float(value) for key, value in new_gain.items()},
    "increment_over_old": {str(key): float(value) for key, value in increment_over_old.items()},
    "new_far_change": {str(key): float(value) for key, value in new_far_change.items()},
    "new_p95_change": {str(key): float(value) for key, value in new_p95_change.items()},
    "old_rebase_gates": old_rebase_gates,
    "prefix_adaptation_gates": prefix_adaptation_gates,
    "old_rebase_pass": old_rebase_pass,
    "prefix_adaptation_pass": prefix_adaptation_pass,
    "strict_pass": prefix_adaptation_pass,
    "test_well_optimizer_or_search_count": 0,
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS prefix adaptation -> integrate the frozen prefix-verified recipe after the full parent"
        if prefix_adaptation_pass
        else (
            "PREFIX FAIL but old rebase PASS -> use the old geometry recipe on the full parent"
            if old_rebase_pass
            else "FAIL -> Neighbor does not survive the complete 6.39 parent"
        )
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

display_columns = [
    "fold", "role", "endpoint", "rows", "wells", "rmse", "mae",
    "median_well_rmse", "p95_well_rmse", "near_0_500_rmse",
    "mid_500_1000_rmse", "mid_1000_2000_rmse", "far_gt2000_rmse",
]
print("PREFIX-NEIGHBOR FOLD3 TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nPREFIX-NEIGHBOR SUMMARY")
print(summary[display_columns].to_string(index=False), flush=True)
print("\nMATCHED CONTROLS")
print(controls[display_columns].to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
