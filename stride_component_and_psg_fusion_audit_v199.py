"""V199: read-only STRIDE component audit plus fixed PSG blend stability.

This script does not train or produce a submission. It establishes whether
V155's implemented calibration/persistence components helped, records the
missing drilled-trend ablation, quantifies repaired-STRIDE blend ceilings, and
checks whether the V198 PSG pair has a conservative fixed weight that is stable
across the canonical five folds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / "stride_component_and_psg_fusion_audit_v199"
OUT.mkdir(parents=True, exist_ok=True)

COMPONENT_PATH = OOF / "stride_independent_engine_v155" / "component_by_fold.csv"
BASE_PATH = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
PSG_PATH = OOF / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet"
BASE_COLUMN = "metadata_analog_w0p05"
PSG_COLUMN = "psg_tvt"
EXPECTED_ROWS = 3_783_989
WEIGHTS = np.round(np.arange(0.0, 0.20001, 0.0125), 4)


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


def _rmse(error):
    return float(np.sqrt(np.mean(np.square(error))))


for required in (COMPONENT_PATH, BASE_PATH, PSG_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

# Aggregate fold SSE rather than averaging fold RMSE.
component = pd.read_csv(COMPONENT_PATH)
required_component = {"fold", "mode", "rows", "exact_sse", "rmse"}
missing = sorted(required_component.difference(component.columns))
if missing:
    raise RuntimeError(f"V155 component table missing columns: {missing}")
component["fold"] = pd.to_numeric(component["fold"], errors="raise").astype(int)
if set(component["fold"]) != {1, 2, 3, 4, 5}:
    raise RuntimeError(f"Incomplete V155 folds: {sorted(component['fold'].unique())}")

mode_summary = component.groupby("mode", as_index=False).agg(
    rows=("rows", "sum"), exact_sse=("exact_sse", "sum")
)
mode_summary["rmse"] = np.sqrt(mode_summary["exact_sse"] / mode_summary["rows"])
full_rmse = float(mode_summary.loc[mode_summary["mode"].eq("full"), "rmse"].iloc[0])
mode_summary["degradation_vs_full"] = mode_summary["rmse"] - full_rmse

full_by_fold = component.loc[component["mode"].eq("full"), ["fold", "rmse"]].rename(
    columns={"rmse": "full_rmse"}
)
component_delta = component.merge(full_by_fold, on="fold", validate="many_to_one")
component_delta["degradation_vs_full"] = component_delta["rmse"] - component_delta["full_rmse"]

component_effects = {}
for mode in ("no_calibration", "no_persistence", "gaussian"):
    row = mode_summary.loc[mode_summary["mode"].eq(mode)]
    component_effects[mode] = (
        float(row["degradation_vs_full"].iloc[0]) if len(row) else None
    )

# Correct full-row V168/PSG alignment.
base = pd.read_parquet(
    BASE_PATH, columns=["well", "row_idx", "fold", "tvt_true", BASE_COLUMN]
)
base["well"] = base["well"].astype(str)
base["row_idx"] = base["row_idx"].astype(np.int64)
base = base.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
psg = pd.read_parquet(PSG_PATH, columns=["well", "row_idx", PSG_COLUMN])
psg["well"] = psg["well"].astype(str)
psg["row_idx"] = psg["row_idx"].astype(np.int64)
psg = psg.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base) != EXPECTED_ROWS or len(psg) != EXPECTED_ROWS:
    raise RuntimeError(f"Incomplete V168/PSG rows: {len(base)}/{len(psg)}")
if not np.array_equal(base["well"].to_numpy(), psg["well"].to_numpy()):
    raise RuntimeError("V168/PSG well alignment failed")
if not np.array_equal(base["row_idx"].to_numpy(), psg["row_idx"].to_numpy()):
    raise RuntimeError("V168/PSG row alignment failed")

truth = base["tvt_true"].to_numpy(np.float64)
base_prediction = base[BASE_COLUMN].to_numpy(np.float64)
psg_prediction = psg[PSG_COLUMN].to_numpy(np.float64)
folds = base["fold"].to_numpy(np.int8)
base_error = base_prediction - truth
base_rmse = _rmse(base_error)

scan_rows = []
for weight in WEIGHTS:
    prediction = base_prediction + float(weight) * (psg_prediction - base_prediction)
    error = prediction - truth
    row = {"weight": float(weight), "rmse": _rmse(error), "gain": base_rmse - _rmse(error)}
    gains = []
    for fold in range(1, 6):
        mask = folds == fold
        gain = _rmse(base_error[mask]) - _rmse(error[mask])
        row[f"fold{fold}_gain"] = gain
        gains.append(gain)
    row["positive_folds"] = int(np.sum(np.asarray(gains) > 0.0))
    row["minimum_fold_gain"] = float(np.min(gains))
    scan_rows.append(row)
fixed_scan = pd.DataFrame(scan_rows).sort_values("rmse", ignore_index=True)
stable = fixed_scan[
    (fixed_scan["positive_folds"] >= 4)
    & (fixed_scan["minimum_fold_gain"] >= -0.03)
]
best_stable = stable.iloc[0].to_dict() if len(stable) else None

# Per-fold analytic alpha is diagnostic and shows whether weight direction
# reverses; it is not used for held-fold prediction.
fold_alpha = []
for fold in range(1, 6):
    mask = folds == fold
    error = base_error[mask]
    delta = psg_prediction[mask] - base_prediction[mask]
    alpha = float(np.clip(-np.dot(error, delta) / max(np.dot(delta, delta), 1e-20), 0.0, 1.0))
    blended = error + alpha * delta
    fold_alpha.append({
        "fold": fold,
        "analytic_alpha": alpha,
        "base_rmse": _rmse(error),
        "oracle_blend_rmse": _rmse(blended),
        "oracle_gain": _rmse(error) - _rmse(blended),
    })
fold_alpha = pd.DataFrame(fold_alpha)

# Theoretical gain if STRIDE accuracy is repaired while preserving V198's
# uncentred error cosine. This is an arithmetic ceiling, not a prediction.
stride_cosine = 0.37767368724540484
base_mse = base_rmse ** 2
repair_rows = []
for stride_rmse in (18.093856140372427, 14.0, 12.0, 11.0, 10.0, 9.0):
    stride_mse = stride_rmse ** 2
    cross = stride_cosine * np.sqrt(base_mse * stride_mse)
    denominator = base_mse + stride_mse - 2.0 * cross
    alpha = (base_mse - cross) / denominator
    minimum_mse = (base_mse * stride_mse - cross * cross) / denominator
    repair_rows.append({
        "hypothetical_stride_rmse": stride_rmse,
        "preserved_uncentred_cosine": stride_cosine,
        "unclipped_alpha": float(alpha),
        "theoretical_blend_rmse": float(np.sqrt(max(minimum_mse, 0.0))),
        "theoretical_gain": float(base_rmse - np.sqrt(max(minimum_mse, 0.0))),
    })
repair_ceiling = pd.DataFrame(repair_rows)

checks = {
    "v155_full_mode_present": bool((mode_summary["mode"] == "full").any()),
    "calibration_ablation_present": component_effects["no_calibration"] is not None,
    "persistence_ablation_present": component_effects["no_persistence"] is not None,
    "drilled_trend_ablation_present": False,
    "calibration_improves_ge_0p20": bool(
        component_effects["no_calibration"] is not None
        and component_effects["no_calibration"] >= 0.20
    ),
    "persistence_improves_ge_0p30": bool(
        component_effects["no_persistence"] is not None
        and component_effects["no_persistence"] >= 0.30
    ),
    "stable_psg_weight_exists": best_stable is not None,
    "stable_psg_gain_ge_0p05": bool(best_stable is not None and best_stable["gain"] >= 0.05),
}

verdict = {
    "experiment": "stride_component_and_psg_fusion_audit_v199",
    "stage": "read_only_V155_component_recovery_and_PSG_fixed_weight_stability",
    "v155_component_effects": component_effects,
    "drilled_trend_ablation": {
        "present_in_v155_modes": False,
        "reason": "V155 modes were full/gaussian/no_calibration/no_persistence only",
    },
    "best_stable_psg_weight": best_stable,
    "checks": checks,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "Repair STRIDE only after its component deltas are compared with the required parity anchors; "
        "deploy PSG blend only if a stable fixed weight gains at least 0.05."
    ),
}

mode_summary.to_csv(OUT / "stride_mode_summary.csv", index=False)
component_delta.to_csv(OUT / "stride_mode_by_fold.csv", index=False)
fixed_scan.to_csv(OUT / "psg_fixed_weight_scan.csv", index=False)
fold_alpha.to_csv(OUT / "psg_fold_analytic_alpha.csv", index=False)
repair_ceiling.to_csv(OUT / "stride_repair_blend_ceiling.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV199 STRIDE MODES")
print(mode_summary.to_string(index=False), flush=True)
print("\nV199 STRIDE MODE BY FOLD")
print(component_delta.to_string(index=False), flush=True)
print("\nV199 PSG FIXED WEIGHT TOP 20")
print(fixed_scan.head(20).to_string(index=False), flush=True)
print("\nV199 PSG FOLD ALPHA")
print(fold_alpha.to_string(index=False), flush=True)
print("\nV199 STRIDE REPAIR CEILING")
print(repair_ceiling.to_string(index=False), flush=True)
print("\nV199 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
