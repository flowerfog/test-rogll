"""Rebase the original full-curve neighbor transfer on the 6.47 parent OOF.

AutoDL order:
1. Finish ``mainline_6p47_foldclean_fle_gs13_v0_cell.py`` for all five folds.
2. Keep the parent-independent artifacts from ``neighbor_fullcurve_transfer_v0``.
3. Run this file as one cell.

Fold 3 selects the neighbor recipe, Fold 2 confirms it once, and Fold 1 is
untouched.  The 6.39 Q0522 submission edit is deliberately not replayed: it is
a hard-coded edit of one hidden well, not a fold-clean transformation.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "frontier_6p47_parent_neighbor_rebase_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLE_OOF_CANDIDATES = [
    OOF_ROOT / "mainline_6p47_foldclean_fle_gs13_v0" / "mainline_6p47_fle_gs13_oof.pkl",
    OOF_ROOT / "mainline_6p47_foldclean_fle_gs13_v0" / "mainline_6p47_fle_gs13_oof.csv",
]
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
FOLD_CANDIDATES = [
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ROOT / "well_metadata.csv",
]
NEIGHBOR_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
NEIGHBOR_BANK = NEIGHBOR_DIR / "neighbor_transfer_oof.npz"
NEIGHBOR_META = NEIGHBOR_DIR / "selected_neighbors.csv"

SP45_WEIGHT = 0.60
FLE_WEIGHT = 0.40
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
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


def first_file(candidates, label):
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {label}; inspected={[str(p) for p in candidates]}")
    return path


def metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    result = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }
    per_well = (
        pd.DataFrame({"well": frame["well"].to_numpy(), "error2": error**2})
        .groupby("well")["error2"].mean().pow(0.5)
    )
    result.update({
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
    })
    md = frame["md_since"].to_numpy(np.float64)
    for name, mask in {
        "near_0_500_rmse": md <= 500,
        "mid_500_1000_rmse": (md > 500) & (md <= 1000),
        "mid_1000_2000_rmse": (md > 1000) & (md <= 2000),
        "far_gt2000_rmse": md > 2000,
    }.items():
        result[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None
    return result


started = time.time()
fle_path = first_file(FLE_OOF_CANDIDATES, "completed 6.47 FLE OOF")
pool_path = first_file(POOL_CANDIDATES, "candidate pool")
fold_path = first_file(FOLD_CANDIDATES, "frozen fold metadata")
for path in (NEIGHBOR_BANK, NEIGHBOR_META):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run neighbor_fullcurve_transfer_v0 once; its absolute "
            "transfer curves are parent-independent and do not need regeneration later."
        )

fle = pd.read_pickle(fle_path) if fle_path.suffix == ".pkl" else pd.read_csv(fle_path)
required_fle = {"id", "fle_gs13", "tvt_true", "md_since"}
if not required_fle.issubset(fle.columns):
    raise RuntimeError(f"FLE OOF missing {sorted(required_fle - set(fle.columns))}")
fle = fle[["id", "fle_gs13", "tvt_true", "md_since"]].copy()
fle["id"] = fle["id"].astype(str)
if fle["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in FLE OOF")

pool_header = pd.read_csv(pool_path, nrows=0).columns.tolist()
if not {"id", "sp45", "tvt_true"}.issubset(pool_header):
    raise RuntimeError(f"Pool lacks id/sp45/tvt_true: {pool_header}")
pool_cols = [c for c in ("id", "well", "row_idx", "tvt_true", "sp45") if c in pool_header]
pool = pd.read_csv(pool_path, usecols=pool_cols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
if pool["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in candidate pool")

# The transfer bank explicitly records this sorted pool order.  Validate its
# length before using any array; id alignment for the parent remains a merge.
with np.load(NEIGHBOR_BANK, allow_pickle=False) as bank_file:
    bank = {key: bank_file[key].astype(np.float64) for key in bank_file.files}
if any(len(bank[key]) != len(pool) for key in METHODS):
    raise RuntimeError("Neighbor bank row count does not match its source candidate pool")

frame = pool.merge(fle, on="id", how="inner", suffixes=("_pool", "_fle"), validate="one_to_one")
if len(frame) != len(fle):
    missing = fle.loc[~fle["id"].isin(frame["id"]), "id"].head(20).tolist()
    raise RuntimeError(f"FLE/pool alignment incomplete: {len(frame)}/{len(fle)}; missing={missing}")
truth_gap = np.abs(frame["tvt_true_pool"].to_numpy(float) - frame["tvt_true_fle"].to_numpy(float))
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
    raise RuntimeError(f"Truth mismatch after alignment: max={float(np.nanmax(truth_gap))}")

# Merge changes row order, so map bank source rows into the aligned frame by id.
source_pos = pd.Series(np.arange(len(pool), dtype=np.int64), index=pool["id"])
bank_pos = frame["id"].map(source_pos).to_numpy(np.int64)
for key in METHODS:
    frame[key] = bank[key][bank_pos]

frame["tvt_true"] = frame["tvt_true_pool"].astype(np.float64)
frame["md_since"] = frame["md_since"].astype(np.float64)
frame["parent_6p47"] = (
    SP45_WEIGHT * frame["sp45"].to_numpy(np.float64)
    + FLE_WEIGHT * frame["fle_gs13"].to_numpy(np.float64)
)

folds = pd.read_csv(fold_path, usecols=["well", "fold"])
folds["well"] = folds["well"].astype(str)
if folds["well"].duplicated().any():
    raise RuntimeError("Duplicate wells in fold metadata")
fold_map = dict(zip(folds["well"], folds["fold"].astype(int)))
frame["fold"] = frame["well"].map(fold_map)
if frame["fold"].isna().any():
    raise RuntimeError("Fold metadata does not cover all aligned wells")
frame["fold"] = frame["fold"].astype(np.int8)

meta = pd.read_csv(NEIGHBOR_META)
meta["well"] = meta["well"].astype(str)
needed_meta = {
    "well", "min_distance", "median_distance", "p90_distance",
    "anchor_distance_full", "abs_direction_dot",
}
if not needed_meta.issubset(meta.columns):
    raise RuntimeError(f"Neighbor metadata missing {sorted(needed_meta - set(meta.columns))}")
meta = meta.drop_duplicates("well").set_index("well")
for column in needed_meta - {"well"}:
    frame[column] = frame["well"].map(meta[column]).astype(float)

gate_masks = {}
for name, limits in GATES.items():
    gate_masks[name] = (
        (frame["min_distance"].to_numpy() <= limits["min"])
        & (frame["median_distance"].to_numpy() <= limits["median"])
        & (frame["p90_distance"].to_numpy() <= limits["p90"])
        & (frame["anchor_distance_full"].to_numpy() <= limits["anchor"])
        & (frame["abs_direction_dot"].to_numpy() >= MIN_DIRECTION_DOT)
    )

truth = frame["tvt_true"].to_numpy(np.float64)
parent = frame["parent_6p47"].to_numpy(np.float64)


def apply_recipe(recipe):
    prediction = parent.copy()
    if float(recipe["weight"]) == 0.0:
        return prediction
    candidate = frame[str(recipe["method"])].to_numpy(np.float64)
    eligible = gate_masks[str(recipe["gate"])] & np.isfinite(candidate)
    cap = float(recipe["cap_ft"])
    delta = candidate[eligible] - prediction[eligible]
    if np.isfinite(cap):
        delta = np.clip(delta, -cap, cap)
    prediction[eligible] += float(recipe["weight"]) * delta
    return prediction


selection_mask = frame["fold"].to_numpy() == SELECTION_FOLD
confirmation_mask = frame["fold"].to_numpy() == CONFIRMATION_FOLD
untouched_mask = frame["fold"].to_numpy() == UNTOUCHED_FOLD
if not selection_mask.any() or not confirmation_mask.any():
    raise RuntimeError("Selection/confirmation folds are empty")

selection_truth = truth[selection_mask]
selection_parent = parent[selection_mask]
selection_md = frame.loc[selection_mask, "md_since"].to_numpy(np.float64)
selection_base_error = selection_parent - selection_truth
selection_base_rmse = float(np.sqrt(np.mean(selection_base_error**2)))
selection_near = selection_md <= 500
selection_far = selection_md > 2000
selection_base_near = float(np.sqrt(np.mean(selection_base_error[selection_near] ** 2)))
selection_base_far = float(np.sqrt(np.mean(selection_base_error[selection_far] ** 2)))

scan_rows = []
for method in METHODS:
    candidate = frame[method].to_numpy(np.float64)
    for gate in GATES:
        eligible = gate_masks[gate] & np.isfinite(candidate)
        for weight in WEIGHTS:
            for cap in CAPS:
                recipe = dict(method=method, gate=gate, weight=weight, cap_ft=cap)
                pred = apply_recipe(recipe)
                local_error = pred[selection_mask] - selection_truth
                local_rmse = float(np.sqrt(np.mean(local_error**2)))
                local_near = float(np.sqrt(np.mean(local_error[selection_near] ** 2)))
                local_far = float(np.sqrt(np.mean(local_error[selection_far] ** 2)))
                scan_rows.append({
                    "method": method,
                    "gate": gate,
                    "weight": float(weight),
                    "cap_ft": "full" if not np.isfinite(cap) else float(cap),
                    "cap_order": 1e9 if not np.isfinite(cap) else float(cap),
                    "eligible_wells": int(frame.loc[eligible & selection_mask, "well"].nunique()),
                    "eligible_rows": int((eligible & selection_mask).sum()),
                    "rmse": local_rmse,
                    "gain": float(selection_base_rmse - local_rmse),
                    "near_0_500_rmse": local_near,
                    "far_gt2000_rmse": local_far,
                    "near_degradation": float(local_near - selection_base_near),
                    "far_gain": float(selection_base_far - local_far),
                })

scan = pd.DataFrame(scan_rows)
# Tie-break toward lower dose and tighter cap.  A zero-dose arm guarantees that
# selection cannot force a harmful correction.
scan["selection_score"] = (
    scan["rmse"]
    + 0.25 * np.maximum(scan["near_degradation"], 0.0)
    - 0.05 * np.maximum(scan["far_gain"], 0.0)
)
scan = scan.sort_values(
    ["selection_score", "rmse", "weight", "cap_order"],
    ascending=[True, True, True, True],
).reset_index(drop=True)
selected = scan.iloc[0].to_dict()
selected_recipe = {
    "method": selected["method"],
    "gate": selected["gate"],
    "weight": float(selected["weight"]),
    "cap_ft": np.inf if selected["cap_ft"] == "full" else float(selected["cap_ft"]),
}
prediction = apply_recipe(selected_recipe)

selection_base = metrics(frame.loc[selection_mask], parent[selection_mask])
selection_result = metrics(frame.loc[selection_mask], prediction[selection_mask])
confirmation_base = metrics(frame.loc[confirmation_mask], parent[confirmation_mask])
confirmation_result = metrics(frame.loc[confirmation_mask], prediction[confirmation_mask])
selection_gain = selection_base["rmse"] - selection_result["rmse"]
confirmation_gain = confirmation_base["rmse"] - confirmation_result["rmse"]
confirmation_far_gain = (
    confirmation_base["far_gt2000_rmse"] - confirmation_result["far_gt2000_rmse"]
)

scan.to_csv(OUT_DIR / "fold3_neighbor_recipe_scan.csv", index=False)
export = frame[["id", "well", "row_idx", "fold", "md_since", "tvt_true", "sp45", "fle_gs13", "parent_6p47"]].copy()
export["neighbor_rebased"] = prediction
export["neighbor_move"] = prediction - parent
export.to_pickle(OUT_DIR / "frontier_parent_neighbor_oof.pkl")

strict_pass = bool(
    float(selected_recipe["weight"]) > 0.0
    and selection_gain >= 0.05
    and confirmation_gain >= 0.05
    and confirmation_far_gain >= 0.0
    and confirmation_result["near_0_500_rmse"]
        <= confirmation_base["near_0_500_rmse"] + 0.05
)
verdict = {
    "experiment": "frontier_6p47_parent_neighbor_rebase_v0",
    "stage": "Fold3_selection_to_Fold2_confirmation",
    "parent": {
        "formula": "0.60 * foldclean_sp45 + 0.40 * foldclean_fle_gs13",
        "fle_oof": str(fle_path),
        "pool": str(pool_path),
        "fold_source": str(fold_path),
    },
    "q0522": {
        "included": False,
        "reason": "hard-coded +0.522 ft on hidden well 00e12e8b after LB feedback; no fold-clean operator exists",
    },
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": bool(untouched_mask.any() and False),
    },
    "selected_recipe": {
        **selected_recipe,
        "cap_ft": None if not np.isfinite(selected_recipe["cap_ft"]) else selected_recipe["cap_ft"],
    },
    "selection_base": selection_base,
    "selection": selection_result,
    "selection_gain": float(selection_gain),
    "confirmation_base": confirmation_base,
    "confirmation": confirmation_result,
    "confirmation_gain": float(confirmation_gain),
    "confirmation_far_gain": float(confirmation_far_gain),
    "gates": {
        "selection_gain_ge_0p05": bool(selection_gain >= 0.05),
        "confirmation_gain_ge_0p05": bool(confirmation_gain >= 0.05),
        "confirmation_far_gain_ge_0": bool(confirmation_far_gain >= 0.0),
        "confirmation_near_degradation_le_0p05": bool(
            confirmation_result["near_0_500_rmse"]
            <= confirmation_base["near_0_500_rmse"] + 0.05
        ),
    },
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": (
        "PASS -> keep frozen Neighbor dose and run CP rebase independently"
        if strict_pass
        else "FAIL -> Neighbor does not transfer to the 6.47 parent; keep dose zero and run CP rebase"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("NEIGHBOR FOLD3 TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
