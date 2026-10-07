"""Re-test Neighbor after the fold-clean CP8-noCP12 endpoint.

Protocol
--------
* Fold 3 selects one Neighbor method/gate/dose/cap recipe.
* Fold 2 confirms the frozen recipe.
* Fold 1 is the final untouched confirmation; it performs no search.
* The Neighbor bank is converted back to an absolute path before it is
  rebased on CP8-noCP12.  No historical CP12 prediction is consumed.

Run after ``frontier_6p39_cp8_nocp12_matched_v9_cell.py`` on AutoDL.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
CP_PATH = OOF_ROOT / "frontier_6p39_cp8_nocp12_matched_v9" / "cp8_nocp12_oof.parquet"
OLD_NEIGHBOR_PATH = (
    OOF_ROOT / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet"
)
NEIGHBOR_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
NEIGHBOR_BANK = NEIGHBOR_DIR / "neighbor_transfer_oof.npz"
NEIGHBOR_META = NEIGHBOR_DIR / "selected_neighbors.csv"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
OUT_DIR = OOF_ROOT / "frontier_6p39_cp8_nocp12_neighbor_v10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
OLD_NEIGHBOR_WEIGHT = 0.35
MIN_DIRECTION_DOT = 0.70
METHODS = ("structure_xy", "tvt_xy", "structure_progress", "tvt_progress")
GATES = {
    "coherent150": dict(min=np.inf, median=150.0, p90=300.0, anchor=300.0),
    "coherent300": dict(min=np.inf, median=300.0, p90=600.0, anchor=600.0),
    "coherent600": dict(min=np.inf, median=600.0, p90=1200.0, anchor=1200.0),
    "touch150": dict(min=150.0, median=600.0, p90=1200.0, anchor=1200.0),
}
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0)
CAPS = (3.0, 6.0, 9.0, 12.0, 18.0, 24.0, np.inf)


def first_file(paths, label):
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {label}; inspected={[str(p) for p in paths]}")
    return path


for required in (CP_PATH, NEIGHBOR_BANK, NEIGHBOR_META):
    if not required.is_file():
        raise FileNotFoundError(required)
pool_path = first_file(POOL_CANDIDATES, "candidate pool aligned to Neighbor bank")

cp = pd.read_parquet(CP_PATH)
required_cp = {
    "id", "well", "row_idx", "fold", "tvt_true", "frontier_6p39_core", "cp8_nocp12"
}
missing = sorted(required_cp.difference(cp.columns))
if missing:
    raise RuntimeError(f"CP8-noCP12 OOF is missing {missing}")
cp = cp.sort_values(["well", "row_idx"]).reset_index(drop=True)
cp["id"] = cp["id"].astype(str)
cp["well"] = cp["well"].astype(str)
if cp["id"].duplicated().any():
    raise RuntimeError("Duplicate CP8 OOF ids")

pool_header = pd.read_csv(pool_path, nrows=0).columns.tolist()
pool_columns = [column for column in ("id", "well", "row_idx") if column in pool_header]
if "id" not in pool_columns:
    raise RuntimeError(f"Candidate pool has no id: {pool_header}")
pool = pd.read_csv(pool_path, usecols=pool_columns, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
if pool["id"].duplicated().any():
    raise RuntimeError("Duplicate candidate-pool ids")

with np.load(NEIGHBOR_BANK, allow_pickle=False) as handle:
    bank = {key: handle[key].astype(np.float64) for key in handle.files}
for method in METHODS:
    if method not in bank or len(bank[method]) != len(pool):
        raise RuntimeError(f"Malformed Neighbor bank method: {method}")

pool_position = pd.Series(np.arange(len(pool), dtype=np.int64), index=pool["id"])
position = cp["id"].map(pool_position)
if position.isna().any():
    raise RuntimeError(f"Neighbor bank misses {int(position.isna().sum())} CP8 rows")
position = position.to_numpy(np.int64)
for method in METHODS:
    cp[method] = bank[method][position]

# Reuse the exact historical distance convention when available.
if OLD_NEIGHBOR_PATH.is_file():
    old = pd.read_parquet(OLD_NEIGHBOR_PATH, columns=["id", "md_since"])
    old["id"] = old["id"].astype(str)
    cp = cp.merge(old, on="id", how="left", validate="one_to_one")
else:
    cp["md_since"] = cp.groupby("well", sort=False).cumcount().to_numpy(float) + 1.0
if cp["md_since"].isna().any():
    raise RuntimeError(f"Missing md_since for {int(cp['md_since'].isna().sum())} rows")

meta = pd.read_csv(NEIGHBOR_META, dtype={"well": str})
needed_meta = {
    "well", "min_distance", "median_distance", "p90_distance",
    "anchor_distance_full", "abs_direction_dot",
}
missing_meta = sorted(needed_meta.difference(meta.columns))
if missing_meta:
    raise RuntimeError(f"Neighbor metadata missing {missing_meta}")
meta = meta.drop_duplicates("well").set_index("well")
for column in needed_meta - {"well"}:
    cp[column] = cp["well"].map(meta[column]).astype(float)

gate_masks = {}
for name, limits in GATES.items():
    gate_masks[name] = (
        (cp["min_distance"].to_numpy() <= limits["min"])
        & (cp["median_distance"].to_numpy() <= limits["median"])
        & (cp["p90_distance"].to_numpy() <= limits["p90"])
        & (cp["anchor_distance_full"].to_numpy() <= limits["anchor"])
        & (cp["abs_direction_dot"].to_numpy() >= MIN_DIRECTION_DOT)
    )

truth = cp["tvt_true"].to_numpy(np.float64)
parent = cp["frontier_6p39_core"].to_numpy(np.float64)
cp8 = cp["cp8_nocp12"].to_numpy(np.float64)
fold = cp["fold"].to_numpy(np.int8)
distance = cp["md_since"].to_numpy(np.float64)


def metrics(mask, prediction):
    error = np.asarray(prediction, np.float64)[mask] - truth[mask]
    local_wells = cp.loc[mask, "well"].to_numpy()
    bywell = (
        pd.DataFrame({"well": local_wells, "error2": error * error})
        .groupby("well")["error2"].mean().pow(0.5)
    )
    result = {
        "rows": int(mask.sum()),
        "wells": int(pd.Series(local_wells).nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p75_well_rmse": float(bywell.quantile(0.75)),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    local_distance = distance[mask]
    for name, local_mask in {
        "near_0_500_rmse": local_distance <= 500.0,
        "mid_500_1000_rmse": (local_distance > 500.0) & (local_distance <= 1000.0),
        "mid_1000_2000_rmse": (local_distance > 1000.0) & (local_distance <= 2000.0),
        "far_gt2000_rmse": local_distance > 2000.0,
    }.items():
        result[name] = float(np.sqrt(np.mean(error[local_mask] ** 2))) if local_mask.any() else None
    return result


def scan_metrics(mask, prediction):
    """Cheap selection metrics; full per-well metrics are computed after freezing."""
    error = np.asarray(prediction, np.float64)[mask] - truth[mask]
    local_distance = distance[mask]
    near = local_distance <= 500.0
    far = local_distance > 2000.0
    return {
        "rows": int(mask.sum()),
        "wells": int(cp.loc[mask, "well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "near_0_500_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def predict(recipe):
    output = cp8.copy()
    weight = float(recipe["weight"])
    if weight == 0.0:
        return output
    absolute = cp[str(recipe["method"])].to_numpy(np.float64)
    eligible = gate_masks[str(recipe["gate"])] & np.isfinite(absolute)
    move = absolute[eligible] - output[eligible]
    cap = float(recipe["cap_ft"])
    if np.isfinite(cap):
        move = np.clip(move, -cap, cap)
    output[eligible] += weight * move
    return output


selection_mask = fold == SELECTION_FOLD
confirmation_mask = fold == CONFIRMATION_FOLD
final_mask = fold == FINAL_FOLD
if not selection_mask.any() or not confirmation_mask.any() or not final_mask.any():
    raise RuntimeError("Selection/confirmation/final fold is empty")
cp8_selection = scan_metrics(selection_mask, cp8)

scan_rows = []
for method in METHODS:
    for gate in GATES:
        for weight in WEIGHTS:
            for cap in CAPS:
                recipe = {"method": method, "gate": gate, "weight": weight, "cap_ft": cap}
                candidate = predict(recipe)
                result = scan_metrics(selection_mask, candidate)
                scan_rows.append({
                    **recipe,
                    **result,
                    "gain_vs_cp8": cp8_selection["rmse"] - result["rmse"],
                    "near_degradation": result["near_0_500_rmse"] - cp8_selection["near_0_500_rmse"],
                    "far_gain": cp8_selection["far_gt2000_rmse"] - result["far_gt2000_rmse"],
                })

scan = pd.DataFrame(scan_rows)
scan["selection_score"] = (
    scan["rmse"]
    + 0.50 * np.maximum(scan["near_degradation"], 0.0)
    - 0.05 * np.maximum(scan["far_gain"], 0.0)
)
scan = scan.sort_values(
    ["selection_score", "rmse", "weight", "cap_ft", "method", "gate"],
    ascending=[True, True, True, True, True, True],
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_neighbor_scan.csv", index=False)
selected = scan.iloc[0]
recipe = {
    "method": str(selected["method"]),
    "gate": str(selected["gate"]),
    "weight": float(selected["weight"]),
    "cap_ft": float(selected["cap_ft"]),
}
candidate = predict(recipe)

subsets = {
    "selection_fold3": selection_mask,
    "confirmation_fold2": confirmation_mask,
    "final_fold1": final_mask,
    "all_folds_diagnostic": np.ones(len(cp), dtype=bool),
}
endpoints = {"parent_6p39": parent, "cp8_nocp12": cp8, "cp8_neighbor": candidate}
summary_rows = []
details = {}
for subset, mask in subsets.items():
    details[subset] = {}
    for endpoint, prediction in endpoints.items():
        report = metrics(mask, prediction)
        details[subset][endpoint] = report
        summary_rows.append({"subset": subset, "endpoint": endpoint, **report})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

def gain(subset, field="rmse"):
    return float(details[subset]["cp8_nocp12"][field] - details[subset]["cp8_neighbor"][field])

fold2_gain = gain("confirmation_fold2")
fold1_gain = gain("final_fold1")
fold2_far_gain = gain("confirmation_fold2", "far_gt2000_rmse")
fold1_far_gain = gain("final_fold1", "far_gt2000_rmse")
fold2_near_degradation = -gain("confirmation_fold2", "near_0_500_rmse")
fold1_near_degradation = -gain("final_fold1", "near_0_500_rmse")
fold2_p95_gain = gain("confirmation_fold2", "p95_well_rmse")
fold1_p95_gain = gain("final_fold1", "p95_well_rmse")

# Matched controls use the same selected move, with sign reversed or assigned
# to a random well.  They are diagnostics only and never tune the recipe.
selected_move = candidate - cp8
reverse = cp8 - selected_move
rng = np.random.default_rng(20260724)
unique_wells = cp["well"].drop_duplicates().to_numpy()
per_well_moves = {
    well: selected_move[np.flatnonzero(cp["well"].to_numpy() == well)] for well in unique_wells
}
shuffled = cp8.copy()
source_wells = rng.permutation(unique_wells)
for target_well, source_well in zip(unique_wells, source_wells):
    target_rows = np.flatnonzero(cp["well"].to_numpy() == target_well)
    source_curve = per_well_moves[source_well]
    if len(source_curve) == 1:
        transferred = np.full(len(target_rows), source_curve[0])
    else:
        transferred = np.interp(
            np.linspace(0.0, 1.0, len(target_rows)),
            np.linspace(0.0, 1.0, len(source_curve)),
            source_curve,
        )
    shuffled[target_rows] += transferred

control_rows = []
for subset, mask in subsets.items():
    for endpoint, prediction in {"cp8_nocp12": cp8, "reverse_move": reverse, "shuffled_well_move": shuffled}.items():
        control_rows.append({"subset": subset, "endpoint": endpoint, **metrics(mask, prediction)})
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)

gates = {
    "selected_neighbor_weight_gt_0": bool(recipe["weight"] > 0.0),
    "fold2_gain_ge_0p05": bool(fold2_gain >= 0.05),
    "fold1_gain_ge_0p05": bool(fold1_gain >= 0.05),
    "fold2_far_nonworse": bool(fold2_far_gain >= -0.05),
    "fold1_far_nonworse": bool(fold1_far_gain >= -0.05),
    "fold2_near_degradation_le_0p05": bool(fold2_near_degradation <= 0.05),
    "fold1_near_degradation_le_0p05": bool(fold1_near_degradation <= 0.05),
    "fold2_p95_nonworse": bool(fold2_p95_gain >= -0.05),
    "fold1_p95_nonworse": bool(fold1_p95_gain >= -0.05),
}
strict_pass = bool(all(gates.values()))

output = cp[["id", "well", "row_idx", "fold", "tvt_true", "frontier_6p39_core"]].copy()
output["cp8_nocp12"] = cp8.astype(np.float32)
output["cp8_neighbor"] = candidate.astype(np.float32)
output["neighbor_move"] = selected_move.astype(np.float32)
output["md_since"] = distance.astype(np.float32)
output.to_parquet(OUT_DIR / "cp8_nocp12_neighbor_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "frontier_6p39_cp8_nocp12_neighbor_v10",
    "stage": "Fold3_selection_to_Fold2_and_untouched_Fold1_confirmation",
    "parent": "frontier_6p39_cp8_nocp12_matched_v9",
    "uses_historical_cp12": False,
    "selected_recipe": recipe,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "Fold1_search_count": 0,
    },
    "metrics": details,
    "gain_vs_cp8": {
        "fold2": fold2_gain,
        "fold1": fold1_gain,
        "fold2_far": fold2_far_gain,
        "fold1_far": fold1_far_gain,
        "fold2_p95": fold2_p95_gain,
        "fold1_p95": fold1_p95_gain,
    },
    "gates": gates,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024 ** 3),
    "next": (
        "PASS -> carry CP8+Neighbor into the matched legal postprocess replay"
        if strict_pass
        else "FAIL -> retain CP8-noCP12 alone for the matched legal postprocess replay"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("FOLD3 NEIGHBOR TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nMATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nMATCHED CONTROLS")
print(controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
