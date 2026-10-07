"""Final untouched-Fold1 confirmation of the frozen 6.39 Neighbor recipe."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
SOURCE_DIR = ROOT / "oof_exports" / "frontier_6p39_neighbor_parent_audit_v2"
SOURCE = SOURCE_DIR / "neighbor_parent_oof.parquet"
SOURCE_VERDICT = SOURCE_DIR / "verdict.json"
OUT_DIR = ROOT / "oof_exports" / "frontier_6p39_neighbor_fold1_final_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT = "published_6p39_core_parent"
CANDIDATE = "published_6p39_core_neighbor"
FINAL_FOLD = 1

for required in (SOURCE, SOURCE_VERDICT):
    if not required.is_file():
        raise FileNotFoundError(required)

upstream = json.loads(SOURCE_VERDICT.read_text(encoding="utf-8"))
frozen = upstream["parents"]["published_6p39_core"]
expected = {
    "method": "structure_progress",
    "gate": "coherent600",
    "weight": 0.35,
    "cap_ft": None,
}
if frozen["selected_recipe"] != expected:
    raise RuntimeError(
        f"Upstream recipe changed; expected={expected}, found={frozen['selected_recipe']}"
    )
if not frozen["strict_pass"]:
    raise RuntimeError("Upstream Fold2 confirmation did not pass")

frame = pd.read_parquet(SOURCE)
required_columns = {"id", "well", "row_idx", "fold", "truth", "md_since", PARENT, CANDIDATE}
missing = sorted(required_columns.difference(frame.columns))
if missing:
    raise RuntimeError(f"Neighbor OOF missing {missing}")
frame = frame.sort_values(["well", "row_idx"]).reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate OOF ids")


def metrics(local: pd.DataFrame, column: str) -> dict:
    truth = local["truth"].to_numpy(np.float64)
    prediction = local[column].to_numpy(np.float64)
    error = prediction - truth
    bywell = (
        pd.DataFrame({"well": local["well"].to_numpy(), "error2": error * error})
        .groupby("well")["error2"]
        .mean()
        .pow(0.5)
    )
    result = {
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p75_well_rmse": float(bywell.quantile(0.75)),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    distance = local["md_since"].to_numpy(np.float64)
    for name, mask in {
        "near_0_500_rmse": distance <= 500.0,
        "mid_500_1000_rmse": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000_rmse": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000_rmse": distance > 2000.0,
    }.items():
        result[name] = (
            float(np.sqrt(np.mean(error[mask] * error[mask]))) if mask.any() else None
        )
    return result


fold1 = frame.loc[frame["fold"].eq(FINAL_FOLD)].copy()
if fold1["well"].nunique() != 155:
    raise RuntimeError(f"Expected 155 Fold1 wells, found {fold1['well'].nunique()}")

parent = metrics(fold1, PARENT)
candidate = metrics(fold1, CANDIDATE)
gain = parent["rmse"] - candidate["rmse"]
far_gain = parent["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]
near_degradation = candidate["near_0_500_rmse"] - parent["near_0_500_rmse"]
p95_gain = parent["p95_well_rmse"] - candidate["p95_well_rmse"]

fold12 = frame.loc[frame["fold"].isin([1, 2])].copy()
combined_parent = metrics(fold12, PARENT)
combined_candidate = metrics(fold12, CANDIDATE)

strict_pass = bool(
    gain >= 0.05
    and far_gain >= 0.0
    and near_degradation <= 0.05
    and p95_gain >= 0.0
)
verdict = {
    "experiment": "frontier_6p39_neighbor_fold1_final_v3",
    "stage": "untouched_Fold1_final_confirmation_of_frozen_neighbor_recipe",
    "source": str(SOURCE),
    "frozen_recipe": expected,
    "protocol": {
        "recipe_selection_fold": 3,
        "first_confirmation_fold": 2,
        "final_confirmation_fold": 1,
        "Fold1_tuning_or_search_count": 0,
    },
    "fold1_parent": parent,
    "fold1_candidate": candidate,
    "fold1_gain": float(gain),
    "fold1_far_gain": float(far_gain),
    "fold1_near_degradation": float(near_degradation),
    "fold1_p95_gain": float(p95_gain),
    "fold1_and_fold2_parent": combined_parent,
    "fold1_and_fold2_candidate": combined_candidate,
    "fold1_and_fold2_gain": float(combined_parent["rmse"] - combined_candidate["rmse"]),
    "gates": {
        "fold1_gain_ge_0p05": bool(gain >= 0.05),
        "fold1_far_gain_ge_0": bool(far_gain >= 0.0),
        "fold1_near_degradation_le_0p05": bool(near_degradation <= 0.05),
        "fold1_p95_nonworse": bool(p95_gain >= 0.0),
    },
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": (
        "PASS -> retain Neighbor on the published 6.39 parent and train the fixed CP residual head"
        if strict_pass
        else "FAIL -> Neighbor was Fold2-specific; keep it out of the reconstructed 6.39 core"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

summary = pd.DataFrame([
    {"endpoint": "fold1_parent", **parent},
    {"endpoint": "fold1_neighbor", **candidate},
    {"endpoint": "fold12_parent", **combined_parent},
    {"endpoint": "fold12_neighbor", **combined_candidate},
])
summary.to_csv(OUT_DIR / "summary.csv", index=False)

print("FOLD1 FINAL SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
