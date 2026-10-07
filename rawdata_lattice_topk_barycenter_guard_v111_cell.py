"""V111: deterministic tail-risk guard for the V110 barycenter endpoint.

V110 improved pooled and far RMSE but worsened the Fold3 p95 well.  This
development scan applies only global, label-free safeguards to V110's move:
prefix freezing, a smooth ramp, scalar shrinkage, and an absolute row cap.
The complete Fold3 is development data.  Any selected recipe still requires a
new Fold2 reconstruction and confirmation before deployment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_topk_barycenter_guard_v111"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = OOF_ROOT / "rawdata_lattice_topk_path_barycenter_v110" / "fold3_topk_barycenter_oof.parquet"
TRUTH = "tvt_true"
BASE = "exact_v76"
CANDIDATE = "topk_barycenter"
FREEZE_VALUES = (0.0, 500.0, 1000.0)
RAMP_FT = 250.0
CAP_VALUES = (0.5, 1.0, 1.5, 2.0, 3.0, 6.0)
SHRINKAGES = (0.50, 0.75, 1.00)


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


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    error = prediction - frame[TRUTH].to_numpy(np.float64)
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid endpoint {endpoint}")
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    output = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    masks = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in masks.items():
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
required = {"id", "well", "row_idx", "fold", TRUTH, "md_since", BASE, CANDIDATE}
missing = sorted(required.difference(frame.columns))
if missing or frame.empty or frame["id"].duplicated().any():
    raise RuntimeError(f"Invalid V110 source, missing={missing}, rows={len(frame)}")
if frame["well"].nunique() != 154 or set(frame["fold"].astype(int).unique()) != {3}:
    raise RuntimeError("V111 expects the complete Fold3 V110 endpoint")

base_prediction = frame[BASE].to_numpy(np.float64)
raw_move = frame[CANDIDATE].to_numpy(np.float64) - base_prediction
distance = frame["md_since"].to_numpy(np.float64)
base = _metric(frame, base_prediction, BASE)
raw = _metric(frame, base_prediction + raw_move, "v110_raw")

scan_rows = []
predictions = {}
for freeze_ft in FREEZE_VALUES:
    taper = np.ones(len(frame), np.float64) if freeze_ft <= 0.0 else _smoothstep(
        (distance - freeze_ft) / RAMP_FT
    )
    for cap_ft in CAP_VALUES:
        capped = np.clip(raw_move, -cap_ft, cap_ft)
        for shrinkage in SHRINKAGES:
            prediction = base_prediction + taper * shrinkage * capped
            recipe = (freeze_ft, cap_ft, shrinkage)
            predictions[recipe] = prediction
            result = _metric(frame, prediction, "guarded_barycenter")
            result.update({
                "freeze_ft": freeze_ft,
                "ramp_ft": RAMP_FT,
                "cap_ft": cap_ft,
                "shrinkage": shrinkage,
                "gain": float(base["rmse"] - result["rmse"]),
                "far_gain": float(base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
                "near_degradation": float(result["near_0_500_rmse"] - base["near_0_500_rmse"]),
                "p95_gain": float(base["p95_well_rmse"] - result["p95_well_rmse"]),
                "mean_abs_move": float(np.mean(np.abs(prediction - base_prediction))),
                "p95_abs_move": float(np.quantile(np.abs(prediction - base_prediction), 0.95)),
            })
            scan_rows.append(result)

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "guard_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.04)
    & (scan["far_gain"] >= 0.05)
    & (scan["near_degradation"] <= 0.01)
    & (scan["p95_gain"] >= 0.0)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (
    float(selected_row["freeze_ft"]),
    float(selected_row["cap_ft"]),
    float(selected_row["shrinkage"]),
)
selected_prediction = predictions[selected]
selected_metric = _metric(frame, selected_prediction, "guarded_barycenter")
values = {
    "raw_v110_gain": float(base["rmse"] - raw["rmse"]),
    "raw_v110_p95_gain": float(base["p95_well_rmse"] - raw["p95_well_rmse"]),
    "selected_gain": float(base["rmse"] - selected_metric["rmse"]),
    "selected_far_gain": float(base["far_gt2000_rmse"] - selected_metric["far_gt2000_rmse"]),
    "selected_near_degradation": float(selected_metric["near_0_500_rmse"] - base["near_0_500_rmse"]),
    "selected_p95_gain": float(base["p95_well_rmse"] - selected_metric["p95_well_rmse"]),
}
checks = {
    "eligible_recipe_exists": bool(len(eligible)),
    "selected_gain_ge_0p04": values["selected_gain"] >= 0.04,
    "selected_far_gain_ge_0p05": values["selected_far_gain"] >= 0.05,
    "selected_near_degradation_le_0p01": values["selected_near_degradation"] <= 0.01,
    "selected_p95_nonworse": values["selected_p95_gain"] >= 0.0,
}
strict_pass = bool(all(checks.values()))

output = frame.copy()
output["guarded_barycenter"] = selected_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_guarded_barycenter_oof.parquet", index=False, compression="zstd")
summary = pd.DataFrame([base, raw, selected_metric])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_Fold3_development_scan_before_independent_Fold2_confirmation",
    "selection_search_count": len(scan),
    "base": "exact V76",
    "source_recipe": {"topk": 3, "aggregator": "rank_weighted", "alpha": 0.25},
    "guard_family": {
        "freeze_ft": list(FREEZE_VALUES),
        "ramp_ft": RAMP_FT,
        "cap_ft": list(CAP_VALUES),
        "shrinkage": list(SHRINKAGES),
    },
    "selected_recipe": {
        "freeze_ft": selected[0],
        "ramp_ft": RAMP_FT,
        "cap_ft": selected[1],
        "shrinkage": selected[2],
    },
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "fold2_touched": False,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild frozen V110/V111 recipe on Fold2 for one independent confirmation"
        if strict_pass else
        "FAIL -> barycenter tail risk cannot be bounded without losing its gain; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV111 GUARD TOP 25")
print(scan.head(25).to_string(index=False), flush=True)
print("\nV111 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV111 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
