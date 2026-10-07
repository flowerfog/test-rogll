"""Tail-only full-curve Neighbor consumption on the complete 6.39 parent.

This is the matched adaptation of tail_only_reanchored_neighbor_gateway_v0:

* the first 1000 ft is exactly frozen;
* the old 1000/250/no-reanchor/0.20 recipe is replayed unchanged;
* Fold 3 independently selects a recipe for the new 6.39 parent;
* Fold 2 confirms it and Fold 1 is the final untouched check.

The Neighbor path is an absolute full-curve candidate.  Its delta is rebased
against the current parent before the distal mask is applied.  No hidden
prediction is inspected and no Fold 1/2 target is used for selection.
"""

from pathlib import Path
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
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
OUT_DIR = OOF_ROOT / "frontier_6p39_tail_only_neighbor_v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_COLUMN = "full_parent_matched_vp_replay"
NEIGHBOR_METHOD = "structure_progress"
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
STARTS_FT = (1000.0, 1250.0, 1500.0, 1750.0, 2000.0)
TRANSITION_WIDTHS_FT = (250.0, 500.0)
REANCHOR_MODES = ("none", "value")
CAPS_FT = (12.0, 24.0, 40.0, np.inf)
DOSES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
GEOMETRY_GATES = {
    "all": None,
    "coherent600": dict(median=600.0, p90=1200.0, anchor=1200.0, dot=0.70),
}
OLD_RECIPE = {
    "start_ft": 1000.0,
    "transition_width_ft": 250.0,
    "reanchor": "none",
    "cap_ft": 24.0,
    "dose": 0.20,
    "geometry_gate": "all",
}
started = time.time()


def _smoothstep(distance, start, width):
    t = np.clip((distance - float(start)) / float(width), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    well_rmse = np.sqrt(
        pd.Series(error * error).groupby(frame["well"], sort=False).mean().to_numpy()
    )
    output = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    bands = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in bands.items():
        output[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return output


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

if len(frame) == len(pool) and np.array_equal(
    frame["id"].to_numpy(), pool["id"].to_numpy()
):
    neighbor = bank_candidate.copy()
else:
    pool_position = pd.Series(np.arange(len(pool), dtype=np.int64), index=pool["id"])
    position = frame["id"].map(pool_position)
    if position.isna().any():
        raise RuntimeError(f"Neighbor bank misses {int(position.isna().sum())} rows")
    neighbor = bank_candidate[position.to_numpy(np.int64)]
del pool, bank_candidate

meta = pd.read_csv(NEIGHBOR_META, dtype={"well": str, "neighbor": str})
needed_meta = {
    "well", "median_distance", "p90_distance", "anchor_distance_full",
    "abs_direction_dot",
}
missing_meta = sorted(needed_meta.difference(meta.columns))
if missing_meta:
    raise RuntimeError(f"Neighbor metadata missing {missing_meta}")
meta = meta.drop_duplicates("well").set_index("well")
missing_wells = sorted(set(frame["well"].unique()).difference(meta.index))
if missing_wells:
    raise RuntimeError(f"Neighbor metadata misses {len(missing_wells)} wells: {missing_wells[:10]}")

# Fold 4/5 remain available in the source but are not consumed after the
# Fold3 recipe selection.  Restricting the working table saves memory.
working_mask = frame["fold"].isin((1, 2, 3)).to_numpy()
frame = frame.loc[working_mask].reset_index(drop=True)
neighbor = neighbor[working_mask]

if len(neighbor) != len(frame):
    raise RuntimeError("Internal Neighbor alignment failed after Fold1/2/3 restriction")

parent = frame[PARENT_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
raw_delta = np.where(np.isfinite(neighbor), neighbor - parent, 0.0)
coverage = np.isfinite(neighbor)
groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}


def _geometry_eligible(well, gate_name):
    if gate_name == "all":
        return True
    limits = GEOMETRY_GATES[gate_name]
    row = meta.loc[well]
    return bool(
        float(row["median_distance"]) <= limits["median"]
        and float(row["p90_distance"]) <= limits["p90"]
        and float(row["anchor_distance_full"]) <= limits["anchor"]
        and float(row["abs_direction_dot"]) >= limits["dot"]
    )


print(
    "TAIL6P39 CONFIG",
    {
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "parent": PARENT_COLUMN,
        "neighbor": NEIGHBOR_METHOD,
        "starts_ft": STARTS_FT,
    },
    flush=True,
)

# Build per-well sufficient statistics for every non-dose configuration.
stat_rows = []
for number, (well, indices) in enumerate(groups.items(), 1):
    local_error = parent[indices] - truth[indices]
    local_distance = distance[indices]
    local_raw = raw_delta[indices]
    local_coverage = coverage[indices]
    held_fold = int(fold[indices[0]])
    base = {
        "well": well,
        "fold": held_fold,
        "rows": int(len(indices)),
        "base_sse": float(np.dot(local_error, local_error)),
    }
    for start_ft, width_ft, reanchor, cap_ft in itertools.product(
        STARTS_FT, TRANSITION_WIDTHS_FT, REANCHOR_MODES, CAPS_FT
    ):
        if reanchor == "value" and local_coverage.any():
            usable_distance = local_distance[local_coverage]
            usable_raw = local_raw[local_coverage]
            anchor = float(np.interp(start_ft, usable_distance, usable_raw))
        else:
            anchor = 0.0
        anchored = np.where(local_coverage, local_raw - anchor, 0.0)
        clipped = anchored if not np.isfinite(cap_ft) else np.clip(anchored, -cap_ft, cap_ft)
        correction = _smoothstep(local_distance, start_ft, width_ft) * clipped
        row = {
            **base,
            "start_ft": float(start_ft),
            "transition_width_ft": float(width_ft),
            "reanchor": reanchor,
            "cap_ft": float(cap_ft),
            "cross": float(np.dot(local_error, correction)),
            "delta2": float(np.dot(correction, correction)),
            "coverage": float(local_coverage.mean()),
        }
        for band_name, mask in {
            "near": local_distance <= 1000.0,
            "mid": (local_distance > 1000.0) & (local_distance <= 2000.0),
            "far": local_distance > 2000.0,
        }.items():
            row[f"{band_name}_rows"] = int(mask.sum())
            row[f"{band_name}_base_sse"] = float(np.dot(local_error[mask], local_error[mask]))
            row[f"{band_name}_cross"] = float(np.dot(local_error[mask], correction[mask]))
            row[f"{band_name}_delta2"] = float(np.dot(correction[mask], correction[mask]))
        stat_rows.append(row)
    if number % 50 == 0 or number == len(groups):
        print(f"TAIL6P39 stats {number}/{len(groups)}", flush=True)
stats = pd.DataFrame(stat_rows)


def _evaluate(recipe, held_folds):
    table = stats[
        stats["fold"].isin(tuple(int(value) for value in held_folds))
        & stats["start_ft"].eq(float(recipe["start_ft"]))
        & stats["transition_width_ft"].eq(float(recipe["transition_width_ft"]))
        & stats["reanchor"].eq(str(recipe["reanchor"]))
        & stats["cap_ft"].eq(float(recipe["cap_ft"]))
    ].copy()
    geometry = np.asarray([
        _geometry_eligible(well, str(recipe["geometry_gate"]))
        for well in table["well"]
    ], bool)
    selected = geometry & (table["coverage"].to_numpy() >= 0.80)
    dose = float(recipe["dose"])
    base_sse = table["base_sse"].to_numpy(np.float64)
    well_sse = base_sse.copy()
    cross = table["cross"].to_numpy(np.float64)
    delta2 = table["delta2"].to_numpy(np.float64)
    well_sse[selected] += 2.0 * dose * cross[selected] + dose * dose * delta2[selected]
    rows_by_well = table["rows"].to_numpy(np.float64)
    well_rmse = np.sqrt(np.maximum(well_sse, 0.0) / rows_by_well)
    parent_well_rmse = np.sqrt(base_sse / rows_by_well)
    total_rows = int(rows_by_well.sum())
    output = {
        "rows": total_rows,
        "wells": int(len(table)),
        "selected_wells": int(selected.sum()),
        "rmse": float(np.sqrt(float(well_sse.sum()) / total_rows)),
        "parent_rmse": float(np.sqrt(float(base_sse.sum()) / total_rows)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "parent_p95_well_rmse": float(np.quantile(parent_well_rmse, 0.95)),
    }
    for band_name in ("near", "mid", "far"):
        band_rows = table[f"{band_name}_rows"].to_numpy(np.float64)
        band_base = table[f"{band_name}_base_sse"].to_numpy(np.float64)
        band_sse = band_base.copy()
        band_cross = table[f"{band_name}_cross"].to_numpy(np.float64)
        band_delta2 = table[f"{band_name}_delta2"].to_numpy(np.float64)
        band_sse[selected] += (
            2.0 * dose * band_cross[selected] + dose * dose * band_delta2[selected]
        )
        output[f"{band_name}_rmse"] = float(
            np.sqrt(float(band_sse.sum()) / float(band_rows.sum()))
        )
        output[f"{band_name}_parent_rmse"] = float(
            np.sqrt(float(band_base.sum()) / float(band_rows.sum()))
        )
    output["gain"] = output["parent_rmse"] - output["rmse"]
    output["near_degradation"] = output["near_rmse"] - output["near_parent_rmse"]
    output["mid_degradation"] = output["mid_rmse"] - output["mid_parent_rmse"]
    output["far_gain"] = output["far_parent_rmse"] - output["far_rmse"]
    output["p95_degradation"] = output["p95_well_rmse"] - output["parent_p95_well_rmse"]
    output["risk_pass"] = bool(
        output["selected_wells"] >= 20
        and abs(output["near_degradation"]) <= 1e-12
        and output["mid_degradation"] <= 0.02
        and output["far_gain"] >= 0.05
        and output["p95_degradation"] <= 0.10
    )
    output["selection_score"] = (
        output["rmse"]
        + 0.50 * max(output["mid_degradation"], 0.0)
        + 0.50 * max(output["p95_degradation"], 0.0)
        + 0.25 * max(-output["far_gain"], 0.0)
    )
    return output


scan_rows = []
for start_ft, width_ft, reanchor, cap_ft, dose, gate in itertools.product(
    STARTS_FT, TRANSITION_WIDTHS_FT, REANCHOR_MODES, CAPS_FT, DOSES, GEOMETRY_GATES
):
    recipe = {
        "start_ft": float(start_ft),
        "transition_width_ft": float(width_ft),
        "reanchor": reanchor,
        "cap_ft": float(cap_ft),
        "dose": float(dose),
        "geometry_gate": gate,
    }
    scan_rows.append({**recipe, **_evaluate(recipe, (SELECTION_FOLD,))})
scan = pd.DataFrame(scan_rows).sort_values(
    ["risk_pass", "selection_score", "rmse", "p95_well_rmse"],
    ascending=[False, True, True, True],
    kind="stable",
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_tail_scan.csv", index=False)
selected = {
    key: scan.iloc[0][key].item() if hasattr(scan.iloc[0][key], "item") else scan.iloc[0][key]
    for key in (
        "start_ft", "transition_width_ft", "reanchor", "cap_ft", "dose",
        "geometry_gate",
    )
}


def _prediction(recipe, shuffled=False):
    output = parent.copy()
    active = np.zeros(len(frame), bool)
    for well, indices in groups.items():
        if not _geometry_eligible(well, str(recipe["geometry_gate"])):
            continue
        local_distance = distance[indices]
        local_raw = raw_delta[indices].copy()
        local_coverage = coverage[indices]
        if local_coverage.mean() < 0.80:
            continue
        if shuffled:
            local_raw = np.roll(local_raw, max(1, len(indices) // 3))
            local_coverage = np.roll(local_coverage, max(1, len(indices) // 3))
        if str(recipe["reanchor"]) == "value" and local_coverage.any():
            anchor = float(np.interp(
                float(recipe["start_ft"]),
                local_distance[local_coverage],
                local_raw[local_coverage],
            ))
        else:
            anchor = 0.0
        anchored = np.where(local_coverage, local_raw - anchor, 0.0)
        cap_ft = float(recipe["cap_ft"])
        if np.isfinite(cap_ft):
            anchored = np.clip(anchored, -cap_ft, cap_ft)
        move = (
            _smoothstep(
                local_distance,
                float(recipe["start_ft"]),
                float(recipe["transition_width_ft"]),
            )
            * float(recipe["dose"])
            * anchored
        )
        output[indices] += move
        active[indices] = local_coverage & (local_distance > float(recipe["start_ft"]))
    return output, active


old_prediction, old_active = _prediction(OLD_RECIPE)
new_prediction, new_active = _prediction(selected)
shuffle_prediction, _ = _prediction(selected, shuffled=True)

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
        "old_frozen_tail": old_prediction[mask],
        "selected_tail": new_prediction[mask],
    }.items():
        summary_rows.append({"fold": held_fold, "role": role, **_metric(local, prediction, endpoint)})
primary_mask = np.isin(fold, (CONFIRMATION_FOLD, FINAL_FOLD))
primary_frame = frame.loc[primary_mask].reset_index(drop=True)
for endpoint, prediction in {
    "parent": parent[primary_mask],
    "old_frozen_tail": old_prediction[primary_mask],
    "selected_tail": new_prediction[primary_mask],
}.items():
    summary_rows.append({
        "fold": 0,
        "role": "fold1_plus_fold2_primary",
        **_metric(primary_frame, prediction, endpoint),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

control_rows = []
for held_fold, role in ((CONFIRMATION_FOLD, "confirmation"), (FINAL_FOLD, "final")):
    mask = fold == held_fold
    local = frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in {
        "parent": parent[mask],
        "selected_tail": new_prediction[mask],
        "shuffled_neighbor_tail": shuffle_prediction[mask],
        "opposite_tail": (2.0 * parent - new_prediction)[mask],
    }.items():
        control_rows.append({"fold": held_fold, "role": role, **_metric(local, prediction, endpoint)})
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)


def _value(table, held_fold, endpoint, column):
    row = table[table["fold"].eq(held_fold) & table["endpoint"].eq(endpoint)]
    return float(row.iloc[0][column])


def _gains(endpoint):
    return {
        held_fold: _value(summary, held_fold, "parent", "rmse")
        - _value(summary, held_fold, endpoint, "rmse")
        for held_fold in (SELECTION_FOLD, CONFIRMATION_FOLD, FINAL_FOLD, 0)
    }


old_gain = _gains("old_frozen_tail")
new_gain = _gains("selected_tail")
far_gain = {
    held_fold: _value(summary, held_fold, "parent", "far_gt2000_rmse")
    - _value(summary, held_fold, "selected_tail", "far_gt2000_rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
mid_change = {
    held_fold: _value(summary, held_fold, "selected_tail", "mid_1000_2000_rmse")
    - _value(summary, held_fold, "parent", "mid_1000_2000_rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
p95_change = {
    held_fold: _value(summary, held_fold, "selected_tail", "p95_well_rmse")
    - _value(summary, held_fold, "parent", "p95_well_rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
maximum_frozen_change = float(
    np.max(np.abs(new_prediction[distance <= 1000.0] - parent[distance <= 1000.0]))
)
shuffle_gain = {
    held_fold: _value(controls, held_fold, "parent", "rmse")
    - _value(controls, held_fold, "shuffled_neighbor_tail", "rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}

old_survival_gates = {
    "old_fold2_gain_positive": bool(old_gain[CONFIRMATION_FOLD] > 0.0),
    "old_fold1_gain_positive": bool(old_gain[FINAL_FOLD] > 0.0),
    "old_fold12_gain_ge_0p05": bool(old_gain[0] >= 0.05),
}
selected_gates = {
    "first_1000ft_exactly_frozen": bool(maximum_frozen_change <= 1e-12),
    "fold2_gain_ge_0p05": bool(new_gain[CONFIRMATION_FOLD] >= 0.05),
    "fold1_gain_ge_0p05": bool(new_gain[FINAL_FOLD] >= 0.05),
    "fold12_gain_ge_0p08": bool(new_gain[0] >= 0.08),
    "fold2_far_gain_ge_0p05": bool(far_gain[CONFIRMATION_FOLD] >= 0.05),
    "fold1_far_gain_ge_0p05": bool(far_gain[FINAL_FOLD] >= 0.05),
    "fold2_mid_degradation_le_0p02": bool(mid_change[CONFIRMATION_FOLD] <= 0.02),
    "fold1_mid_degradation_le_0p02": bool(mid_change[FINAL_FOLD] <= 0.02),
    "fold2_p95_degradation_le_0p10": bool(p95_change[CONFIRMATION_FOLD] <= 0.10),
    "fold1_p95_degradation_le_0p10": bool(p95_change[FINAL_FOLD] <= 0.10),
    "shuffle_retains_le_50pct_fold2": bool(
        shuffle_gain[CONFIRMATION_FOLD]
        <= 0.50 * max(new_gain[CONFIRMATION_FOLD], 1e-9)
    ),
    "shuffle_retains_le_50pct_fold1": bool(
        shuffle_gain[FINAL_FOLD]
        <= 0.50 * max(new_gain[FINAL_FOLD], 1e-9)
    ),
}

row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["parent"] = parent.astype(np.float32)
row_output["neighbor_absolute"] = neighbor.astype(np.float32)
row_output["old_frozen_tail"] = old_prediction.astype(np.float32)
row_output["selected_tail"] = new_prediction.astype(np.float32)
row_output["old_active"] = old_active
row_output["selected_active"] = new_active
row_output.to_parquet(OUT_DIR / "tail_neighbor_oof.parquet", index=False, compression="zstd")


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


verdict = {
    "experiment": "frontier_6p39_tail_only_neighbor_v26",
    "stage": "Fold3_tail_selection_Fold2_confirmation_Fold1_final",
    "parent": PARENT_COLUMN,
    "neighbor_method": NEIGHBOR_METHOD,
    "hidden_predictions_observed": False,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "Fold1_tuning_or_search_count": 0,
        "Fold2_tuning_or_search_count": 0,
    },
    "old_recipe": {key: _json_scalar(value) for key, value in OLD_RECIPE.items()},
    "selected_recipe": {key: _json_scalar(value) for key, value in selected.items()},
    "selection_search_count": int(len(scan)),
    "old_gain": {str(key): float(value) for key, value in old_gain.items()},
    "new_gain": {str(key): float(value) for key, value in new_gain.items()},
    "far_gain": {str(key): float(value) for key, value in far_gain.items()},
    "mid_change": {str(key): float(value) for key, value in mid_change.items()},
    "p95_change": {str(key): float(value) for key, value in p95_change.items()},
    "shuffle_gain": {str(key): float(value) for key, value in shuffle_gain.items()},
    "maximum_frozen_change": maximum_frozen_change,
    "old_survival_gates": old_survival_gates,
    "selected_gates": selected_gates,
    "old_recipe_survives": bool(all(old_survival_gates.values())),
    "strict_pass": bool(all(selected_gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> integrate the frozen selected tail recipe after the complete 6.39 parent"
        if all(selected_gates.values())
        else (
            "OLD RECIPE SURVIVES -> consider a direct frozen old-recipe A/B"
            if all(old_survival_gates.values())
            else "FAIL -> tail-only Neighbor does not transfer to the complete 6.39 parent"
        )
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

display_columns = [
    "fold", "role", "endpoint", "rows", "wells", "rmse", "mae",
    "median_well_rmse", "p95_well_rmse", "near_0_500_rmse",
    "mid_500_1000_rmse", "mid_1000_2000_rmse", "far_gt2000_rmse",
]
print("TAIL6P39 FOLD3 TOP 30")
print(scan.head(30).to_string(index=False), flush=True)
print("\nTAIL6P39 SUMMARY")
print(summary[display_columns].to_string(index=False), flush=True)
print("\nTAIL6P39 MATCHED CONTROLS")
print(controls[display_columns].to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
