"""Nested dose/cap calibration for the confirmed CP residual endpoint.

Uses the already cross-fitted CP raw move. Fold 3 selects one dose/cap pair;
Fold 2 and Fold 1 confirm it without search. No model is retrained.
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
SOURCE = OOF_ROOT / "frontier_6p39_cp8_parent_audit_v2" / "cp8_rebased_oof.pkl"
OUT_DIR = OOF_ROOT / "frontier_6p39_cp_dose_cap_nested_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLDS = (2, 1)
DOSES = (0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50)
CAPS_FT = (6.0, 9.0, 12.0, 15.0, 18.0, 24.0, np.inf)
CURRENT_RECIPE = {"dose": 1.0, "cap_ft": 12.0}

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

frame = pd.read_pickle(SOURCE)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "frontier_6p39_core", "raw_move"
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"CP OOF is missing {missing}")
frame = frame.sort_values(["well", "row_idx"]).reset_index(drop=True)

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["frontier_6p39_core"].to_numpy(np.float64)
raw_move = frame["raw_move"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
distance = frame.groupby("well", sort=False).cumcount().to_numpy(np.float64) + 1.0


def prediction(dose: float, cap_ft: float) -> np.ndarray:
    move = float(dose) * raw_move
    if np.isfinite(cap_ft):
        move = np.clip(move, -float(cap_ft), float(cap_ft))
    return base + move


def metrics(mask: np.ndarray, pred: np.ndarray) -> dict:
    error = pred[mask] - truth[mask]
    wells = frame.loc[mask, "well"].to_numpy()
    bywell = (
        pd.DataFrame({"well": wells, "error2": error * error})
        .groupby("well")["error2"]
        .mean()
        .pow(0.5)
    )
    result = {
        "rows": int(mask.sum()),
        "wells": int(pd.Series(wells).nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
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
            float(np.sqrt(np.mean(error[local_mask] ** 2))) if local_mask.any() else None
        )
    return result


selection_mask = fold == SELECTION_FOLD
base_selection = metrics(selection_mask, base)
current = prediction(**CURRENT_RECIPE)
current_selection = metrics(selection_mask, current)

scan_rows = []
for dose in DOSES:
    for cap_ft in CAPS_FT:
        pred = prediction(dose, cap_ft)
        result = metrics(selection_mask, pred)
        scan_rows.append({
            "dose": float(dose),
            "cap_ft": float(cap_ft),
            **result,
            "gain_vs_base": base_selection["rmse"] - result["rmse"],
            "gain_vs_current": current_selection["rmse"] - result["rmse"],
            "near_degradation_vs_current": (
                result["near_0_500_rmse"] - current_selection["near_0_500_rmse"]
            ),
            "far_gain_vs_current": (
                current_selection["far_gt2000_rmse"] - result["far_gt2000_rmse"]
            ),
        })

scan = pd.DataFrame(scan_rows)
scan["selection_score"] = (
    scan["rmse"]
    + 0.25 * np.maximum(scan["near_degradation_vs_current"], 0.0)
    - 0.05 * np.maximum(scan["far_gain_vs_current"], 0.0)
)
scan = scan.sort_values(
    ["selection_score", "rmse", "dose", "cap_ft"],
    ascending=[True, True, True, True],
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_dose_cap_scan.csv", index=False)

selected = scan.iloc[0]
selected_recipe = {
    "dose": float(selected["dose"]),
    "cap_ft": float(selected["cap_ft"]),
}
candidate = prediction(**selected_recipe)

rows = []
details = {}
confirmation_gains = {}
for fold_value, label in ((3, "selection_fold3"), (2, "confirmation_fold2"), (1, "final_fold1")):
    mask = fold == fold_value
    local = {
        "base": metrics(mask, base),
        "current_cp": metrics(mask, current),
        "selected_cp": metrics(mask, candidate),
    }
    details[label] = local
    confirmation_gains[label] = {
        "rmse": local["current_cp"]["rmse"] - local["selected_cp"]["rmse"],
        "far": (
            local["current_cp"]["far_gt2000_rmse"]
            - local["selected_cp"]["far_gt2000_rmse"]
        ),
        "near": (
            local["current_cp"]["near_0_500_rmse"]
            - local["selected_cp"]["near_0_500_rmse"]
        ),
    }
    for endpoint, result in local.items():
        rows.append({"subset": label, "endpoint": endpoint, **result})

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)

changed = selected_recipe != CURRENT_RECIPE
confirmation_pass = bool(
    changed
    and confirmation_gains["confirmation_fold2"]["rmse"] >= 0.02
    and confirmation_gains["final_fold1"]["rmse"] >= 0.02
    and confirmation_gains["confirmation_fold2"]["far"] >= -0.05
    and confirmation_gains["final_fold1"]["far"] >= -0.05
)
frozen_recipe = selected_recipe if confirmation_pass else CURRENT_RECIPE
frozen_prediction = candidate if confirmation_pass else current

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "frontier_6p39_core"]].copy()
output["raw_move"] = raw_move.astype(np.float32)
output["cp_frozen"] = frozen_prediction.astype(np.float32)
output["cp_frozen_move"] = (frozen_prediction - base).astype(np.float32)
output.to_parquet(OUT_DIR / "cp_frozen_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "frontier_6p39_cp_dose_cap_nested_v5",
    "stage": "Fold3_CP_dose_cap_selection_to_Fold2_and_Fold1_confirmation",
    "current_recipe": CURRENT_RECIPE,
    "selected_recipe": selected_recipe,
    "frozen_recipe": frozen_recipe,
    "metrics": details,
    "selected_gain_vs_current": confirmation_gains,
    "strict_pass": confirmation_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024 ** 3),
    "next": (
        "PASS -> freeze selected CP recipe and rebuild CP->Neighbor increments"
        if confirmation_pass
        else "NO STABLE CALIBRATION GAIN -> retain CP dose=1.0/cap=12ft"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("FOLD3 CP DOSE/CAP TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nCP CONFIRMATION SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
