"""V124b: read-only confirmation of V124's safest near-gate recipe."""

from __future__ import annotations

import json
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "final_consensus_coefficient_confirmation_v124b"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V124_SOURCE = ROOT / "final_consensus_coefficient_regressor_v124_cell.py"
FIXED_RECIPE = ("hist", "tail_affine", 0.50, 4.0)
BASE = "consensus_center"
TRUTH = "tvt_true"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


if not V124_SOURCE.is_file():
    raise FileNotFoundError(V124_SOURCE)
N = runpy.run_path(str(V124_SOURCE), run_name="final_consensus_v124b_replay")
confirmation_wells = set(N["confirmation_wells"])
if len(confirmation_wells) < 129:
    raise RuntimeError(f"Only {len(confirmation_wells)} confirmation wells")

frame = N["frame"]
R = N["R"]
confirmation_frame = frame[frame["well"].isin(confirmation_wells)].copy()
base_prediction = confirmation_frame[BASE].to_numpy(np.float64)
base = R._metric(confirmation_frame, base_prediction, BASE)
candidate, matched, prediction = N["_evaluate"](
    confirmation_wells, FIXED_RECIPE, "coefficient_regressor"
)

# Matched negative control: preserve the coefficient distribution but assign
# each complete well another well's predicted coefficient vector.
family, profile, _, _ = FIXED_RECIPE
coefficient_predictions = N["coefficient_predictions"]
key = (family, profile)
original_map = coefficient_predictions[key]
ordered_wells = sorted(confirmation_wells)
rolled_values = np.roll(
    np.vstack([original_map[well] for well in ordered_wells]),
    shift=max(1, len(ordered_wells) // 7),
    axis=0,
)
rolled_map = dict(original_map)
for well, value in zip(ordered_wells, rolled_values):
    rolled_map[well] = value
coefficient_predictions[key] = rolled_map
try:
    rolled, _, rolled_prediction = N["_evaluate"](
        confirmation_wells, FIXED_RECIPE, "rolled_coefficient_control"
    )
finally:
    coefficient_predictions[key] = original_map

summary = pd.DataFrame([base, candidate, rolled])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
values = {
    "gain": float(base["rmse"] - candidate["rmse"]),
    "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
    "near_degradation": float(candidate["near_0_500_rmse"] - base["near_0_500_rmse"]),
    "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
    "aligned_increment_vs_rolled": float(rolled["rmse"] - candidate["rmse"]),
}
checks = {
    "gain_ge_0p03": values["gain"] >= 0.03,
    "far_gain_nonnegative": values["far_gain"] >= 0.0,
    "near_frozen_exact": abs(values["near_degradation"]) <= 1e-5,
    "p95_not_worse_by_gt_0p10": values["p95_gain"] >= -0.10,
    "aligned_beats_rolled_ge_0p02": values["aligned_increment_vs_rolled"] >= 0.02,
}
strict_pass = bool(all(checks.values()))

output = matched[["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
output["coefficient_regressor"] = prediction.astype(np.float32)
output["rolled_coefficient_control"] = rolled_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "read_only_confirmation_of_V124_safest_near_gate_recipe",
    "selection_or_tuning_count": 0,
    "frozen_recipe": {
        "family": FIXED_RECIPE[0],
        "profile": FIXED_RECIPE[1],
        "dose": FIXED_RECIPE[2],
        "cap_ft": FIXED_RECIPE[3],
    },
    "protocol": {
        "train_folds": [1, 2],
        "selection_fold": 3,
        "selection_wells_excluded": 24,
        "confirmation_wells": len(confirmation_wells),
        "fold2_used_for_training": True,
        "fold1_used_for_training": True,
    },
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build full cross-fit coefficient OOF and deployment bundle"
        if strict_pass else
        "FAIL -> oracle coefficients are not legally observable with current well evidence"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV124b MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV124b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
