"""Nested dose/order audit for the confirmed CP8 and Neighbor components."""

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
CP_PATH = OOF_ROOT / "frontier_6p39_cp8_parent_audit_v2" / "cp8_rebased_oof.pkl"
NEIGHBOR_PATH = (
    OOF_ROOT / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet"
)
OUT_DIR = OOF_ROOT / "frontier_6p39_cp_neighbor_sequence_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
FROZEN_NEIGHBOR_WEIGHT = 0.35
DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
ORDERS = ("neighbor_then_cp", "cp_then_neighbor")

for required in (CP_PATH, NEIGHBOR_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

cp = pd.read_pickle(CP_PATH)
neighbor = pd.read_parquet(NEIGHBOR_PATH)
required_cp = {
    "id", "well", "row_idx", "fold", "tvt_true", "frontier_6p39_core",
    "cp8_rebased", "capped_move",
}
required_neighbor = {
    "id", "truth", "published_6p39_core_parent",
    "published_6p39_core_neighbor", "published_6p39_core_neighbor_move",
}
missing_cp = sorted(required_cp.difference(cp.columns))
missing_neighbor = sorted(required_neighbor.difference(neighbor.columns))
if missing_cp or missing_neighbor:
    raise RuntimeError(f"Missing columns: cp={missing_cp}, neighbor={missing_neighbor}")

cp = cp.sort_values(["well", "row_idx"]).reset_index(drop=True)
neighbor = neighbor[[
    "id", "truth", "md_since", "published_6p39_core_parent",
    "published_6p39_core_neighbor", "published_6p39_core_neighbor_move",
]].copy()
cp["id"] = cp["id"].astype(str)
neighbor["id"] = neighbor["id"].astype(str)
frame = cp.merge(neighbor, on="id", how="inner", validate="one_to_one")
if len(frame) != len(cp) or len(frame) != len(neighbor):
    raise RuntimeError(f"Incomplete CP/Neighbor alignment: {len(frame)}/{len(cp)}/{len(neighbor)}")

truth_gap = np.abs(frame["tvt_true"].to_numpy(float) - frame["truth"].to_numpy(float))
base_gap = np.abs(
    frame["frontier_6p39_core"].to_numpy(float)
    - frame["published_6p39_core_parent"].to_numpy(float)
)
if float(np.nanmax(truth_gap)) > 0.01 or float(np.nanmax(base_gap)) > 0.01:
    raise RuntimeError(
        f"Endpoint mismatch: truth_max={float(np.nanmax(truth_gap))}, "
        f"base_max={float(np.nanmax(base_gap))}"
    )

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["frontier_6p39_core"].to_numpy(np.float64)
cp_move = frame["capped_move"].to_numpy(np.float64)
neighbor_move = frame["published_6p39_core_neighbor_move"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
distance = frame["md_since"].to_numpy(np.float64)
eligible = np.abs(neighbor_move) > 1e-10

# Recover the parent-independent absolute transfer curve where Neighbor acted.
neighbor_absolute = base.copy()
neighbor_absolute[eligible] = (
    base[eligible] + neighbor_move[eligible] / FROZEN_NEIGHBOR_WEIGHT
)


def compose(order: str, cp_dose: float, neighbor_dose: float) -> np.ndarray:
    cp_dose = float(cp_dose)
    neighbor_dose = float(neighbor_dose)
    if order == "neighbor_then_cp":
        return base + neighbor_dose * neighbor_move + cp_dose * cp_move
    if order == "cp_then_neighbor":
        output = base + cp_dose * cp_move
        output[eligible] += (
            neighbor_dose
            * FROZEN_NEIGHBOR_WEIGHT
            * (neighbor_absolute[eligible] - output[eligible])
        )
        return output
    raise ValueError(order)


def metrics(mask: np.ndarray, prediction: np.ndarray) -> dict:
    local_error = prediction[mask] - truth[mask]
    local_well = frame.loc[mask, "well"].to_numpy()
    bywell = (
        pd.DataFrame({"well": local_well, "error2": local_error * local_error})
        .groupby("well")["error2"]
        .mean()
        .pow(0.5)
    )
    result = {
        "rows": int(mask.sum()),
        "wells": int(pd.Series(local_well).nunique()),
        "exact_sse": float(np.dot(local_error, local_error)),
        "rmse": float(np.sqrt(np.mean(local_error * local_error))),
        "mae": float(np.mean(np.abs(local_error))),
        "median_well_rmse": float(bywell.median()),
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
        result[name] = (
            float(np.sqrt(np.mean(local_error[local_mask] ** 2)))
            if local_mask.any()
            else None
        )
    return result


selection_mask = fold == SELECTION_FOLD
confirmation_mask = fold == CONFIRMATION_FOLD
final_mask = fold == FINAL_FOLD
base_selection = metrics(selection_mask, base)

scan_rows = []
for order in ORDERS:
    for cp_dose in DOSES:
        for neighbor_dose in DOSES:
            prediction = compose(order, cp_dose, neighbor_dose)
            result = metrics(selection_mask, prediction)
            scan_rows.append({
                "order": order,
                "cp_dose": cp_dose,
                "neighbor_dose": neighbor_dose,
                **result,
                "gain_vs_base": base_selection["rmse"] - result["rmse"],
                "near_degradation": result["near_0_500_rmse"] - base_selection["near_0_500_rmse"],
                "far_gain": base_selection["far_gt2000_rmse"] - result["far_gt2000_rmse"],
            })

scan = pd.DataFrame(scan_rows)
scan["selection_score"] = (
    scan["rmse"]
    + 0.25 * np.maximum(scan["near_degradation"], 0.0)
    - 0.05 * np.maximum(scan["far_gain"], 0.0)
)
scan = scan.sort_values(
    ["selection_score", "rmse", "cp_dose", "neighbor_dose", "order"],
    ascending=[True, True, True, True, True],
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_dose_order_scan.csv", index=False)
selected = scan.iloc[0]
recipe = {
    "order": str(selected["order"]),
    "cp_dose": float(selected["cp_dose"]),
    "neighbor_dose": float(selected["neighbor_dose"]),
}
candidate = compose(**recipe)

cp_only = base + cp_move
neighbor_only = base + neighbor_move
rows = []
details = {}
for label, mask in {
    "selection_fold3": selection_mask,
    "confirmation_fold2": confirmation_mask,
    "final_fold1": final_mask,
    "all_folds_diagnostic": np.ones(len(frame), dtype=bool),
}.items():
    local = {
        "base": metrics(mask, base),
        "cp_only": metrics(mask, cp_only),
        "neighbor_only": metrics(mask, neighbor_only),
        "selected_combination": metrics(mask, candidate),
    }
    details[label] = local
    for endpoint, result in local.items():
        rows.append({"subset": label, "endpoint": endpoint, **result})

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "sequence_summary.csv", index=False)
best_standalone_fold2 = min(
    details["confirmation_fold2"]["cp_only"]["rmse"],
    details["confirmation_fold2"]["neighbor_only"]["rmse"],
)
best_standalone_fold1 = min(
    details["final_fold1"]["cp_only"]["rmse"],
    details["final_fold1"]["neighbor_only"]["rmse"],
)
fold2_gain = best_standalone_fold2 - details["confirmation_fold2"]["selected_combination"]["rmse"]
fold1_gain = best_standalone_fold1 - details["final_fold1"]["selected_combination"]["rmse"]
fold2_far_best = min(
    details["confirmation_fold2"]["cp_only"]["far_gt2000_rmse"],
    details["confirmation_fold2"]["neighbor_only"]["far_gt2000_rmse"],
)
fold1_far_best = min(
    details["final_fold1"]["cp_only"]["far_gt2000_rmse"],
    details["final_fold1"]["neighbor_only"]["far_gt2000_rmse"],
)
fold2_far_gain = fold2_far_best - details["confirmation_fold2"]["selected_combination"]["far_gt2000_rmse"]
fold1_far_gain = fold1_far_best - details["final_fold1"]["selected_combination"]["far_gt2000_rmse"]

strict_pass = bool(
    recipe["cp_dose"] > 0.0
    and recipe["neighbor_dose"] > 0.0
    and fold2_gain >= 0.05
    and fold1_gain >= 0.05
    and fold2_far_gain >= -0.05
    and fold1_far_gain >= -0.05
)
verdict = {
    "experiment": "frontier_6p39_cp_neighbor_sequence_v3",
    "stage": "Fold3_order_dose_selection_to_Fold2_and_Fold1_confirmation",
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "first_confirmation_fold": CONFIRMATION_FOLD,
        "second_confirmation_fold": FINAL_FOLD,
        "searched_orders": list(ORDERS),
        "searched_doses": list(DOSES),
    },
    "selected_recipe": recipe,
    "metrics": details,
    "gain_over_best_standalone": {
        "fold2": float(fold2_gain),
        "fold1": float(fold1_gain),
        "fold2_far": float(fold2_far_gain),
        "fold1_far": float(fold1_far_gain),
    },
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": (
        "PASS -> freeze sequence and reproduce the deployment inference chain"
        if strict_pass
        else "NO ADDITIVE GAIN -> deploy the stronger standalone component only"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

output = frame[["id", "well", "row_idx", "fold", "tvt_true"]].copy()
output["base"] = base
output["cp_only"] = cp_only
output["neighbor_only"] = neighbor_only
output["cp_neighbor"] = candidate
output.to_parquet(OUT_DIR / "cp_neighbor_oof.parquet", index=False, compression="zstd")

print("FOLD3 ORDER/DOSE TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nSEQUENCE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
