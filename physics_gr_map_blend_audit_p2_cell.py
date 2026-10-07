"""Cheap nested audit of the raw P1 GR-map endpoint against exact 6.377 OOF.

P1's replay gate selected the wrong endpoint for most wells.  This audit does
not rerun the beam.  It tests the saved ungated GR path directly.  Fold 3
selects one fixed blend mode/weight; Fold 2 is evaluated once.  Fold 1 remains
untouched.  A within-well reversed-delta control checks that any gain depends
on row-aligned trajectory evidence rather than generic shrinkage.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "physics_gr_map_blend_audit_p2"
ROOT = Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT if os.name != "nt" else Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHYSICS_PATH = OOF_ROOT / "physics_gr_null_replay_gateway_p1" / "physics_gr_null_replay_oof.parquet"
BASE_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
BASE_COLUMN = "current_dose_0p20"
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
WEIGHTS = np.arange(0.0, 0.4001, 0.025)
MODES = ("global", "far_ramp512")
started = time.time()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    work = pd.DataFrame({"well": frame["well"].to_numpy(), "e2": error * error})
    bywell = np.sqrt(work.groupby("well")["e2"].mean().to_numpy(np.float64))
    distance = frame["md_since"].to_numpy(np.float64)
    near = distance <= 2000.0
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        "near_le2000_rmse": float(np.sqrt(np.mean(error[near] ** 2))) if near.any() else None,
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else None,
    }


def mode_gate(frame, mode):
    if mode == "global":
        return np.ones(len(frame), np.float64)
    if mode == "far_ramp512":
        return np.clip((frame["md_since"].to_numpy(np.float64) - 2000.0) / 512.0, 0.0, 1.0)
    raise KeyError(mode)


def predict(frame, mode, weight, candidate_column="gr_map"):
    base = frame[BASE_COLUMN].to_numpy(np.float64)
    candidate = frame[candidate_column].to_numpy(np.float64)
    return base + float(weight) * mode_gate(frame, mode) * (candidate - base)


def centred_correlation(frame, left_error, right_error):
    work = pd.DataFrame({"well": frame["well"], "left": left_error, "right": right_error})
    work["left"] -= work.groupby("well")["left"].transform("mean")
    work["right"] -= work.groupby("well")["right"].transform("mean")
    return float(np.corrcoef(work["left"], work["right"])[0, 1])


if not PHYSICS_PATH.is_file():
    raise FileNotFoundError(f"Run P1 first: {PHYSICS_PATH}")
if not BASE_PATH.is_file():
    raise FileNotFoundError(BASE_PATH)

physics = pd.read_parquet(
    PHYSICS_PATH,
    columns=["id", "well", "fold", "tvt_true", "md_since", "gr_map", "prediction"],
).rename(columns={"fold": "physics_fold", "tvt_true": "physics_truth", "md_since": "physics_md_since"})
base = pd.read_parquet(
    BASE_PATH,
    columns=["id", "well", "fold", "tvt_true", "md_since", BASE_COLUMN],
)
frame = base.merge(physics, on=["id", "well"], how="inner", validate="one_to_one")
if not frame["fold"].eq(frame["physics_fold"]).all():
    raise RuntimeError("P1 and exact 6.377 fold lineage disagree")
truth_gap = np.abs(frame["tvt_true"].to_numpy(np.float64) - frame["physics_truth"].to_numpy(np.float64))
distance_gap = np.abs(frame["md_since"].to_numpy(np.float64) - frame["physics_md_since"].to_numpy(np.float64))
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
    raise RuntimeError(f"Truth alignment failed: max gap={np.nanmax(truth_gap)}")
if not np.isfinite(distance_gap).all() or float(distance_gap.max()) > 0.01:
    raise RuntimeError(f"Distance alignment failed: max gap={np.nanmax(distance_gap)}")
if not np.isfinite(frame[["tvt_true", BASE_COLUMN, "gr_map"]].to_numpy(np.float64)).all():
    raise RuntimeError("Non-finite aligned endpoint values")

selection = frame.loc[frame["fold"].eq(SELECTION_FOLD)].copy()
application = frame.loc[frame["fold"].eq(APPLICATION_FOLD)].copy()
if selection["well"].nunique() < 125 or application["well"].nunique() < 150:
    raise RuntimeError(
        f"Insufficient fold coverage: selection={selection['well'].nunique()}, application={application['well'].nunique()}"
    )

scan_rows = []
selection_base_metrics = metrics(selection, selection[BASE_COLUMN])
for mode in MODES:
    for weight in WEIGHTS:
        result = metrics(selection, predict(selection, mode, weight))
        scan_rows.append(
            {
                "mode": mode,
                "weight": float(weight),
                "gain": selection_base_metrics["rmse"] - result["rmse"],
                "p95_change": result["p95_well_rmse"] - selection_base_metrics["p95_well_rmse"],
                **result,
            }
        )
scan = pd.DataFrame(scan_rows)
safe = scan.loc[scan["p95_change"] <= 0.20].copy()
if safe.empty:
    safe = scan.loc[scan["weight"].eq(0.0)].copy()
selected = safe.sort_values(["rmse", "weight", "mode"], kind="stable").iloc[0]
selected_mode = str(selected["mode"])
selected_weight = float(selected["weight"])

# Preserve each well's delta distribution while destroying MD alignment.
application["gr_map_reversed"] = application["gr_map"].to_numpy(np.float64)
for _, positions in application.groupby("well", sort=False).indices.items():
    positions = np.asarray(positions, np.int64)
    base_values = application.iloc[positions][BASE_COLUMN].to_numpy(np.float64)
    delta = application.iloc[positions]["gr_map"].to_numpy(np.float64) - base_values
    application.iloc[positions, application.columns.get_loc("gr_map_reversed")] = base_values + delta[::-1]

selection_base = metrics(selection, selection[BASE_COLUMN])
selection_candidate = metrics(selection, selection["gr_map"])
selection_blend = metrics(selection, predict(selection, selected_mode, selected_weight))
application_base = metrics(application, application[BASE_COLUMN])
application_candidate = metrics(application, application["gr_map"])
application_prediction = predict(application, selected_mode, selected_weight)
application_blend = metrics(application, application_prediction)
application_reverse = metrics(
    application,
    predict(application, selected_mode, selected_weight, candidate_column="gr_map_reversed"),
)

truth = frame["tvt_true"].to_numpy(np.float64)
base_error = frame[BASE_COLUMN].to_numpy(np.float64) - truth
candidate_error = frame["gr_map"].to_numpy(np.float64) - truth
raw_rho = float(np.corrcoef(base_error, candidate_error)[0, 1])
centred_rho = centred_correlation(frame, base_error, candidate_error)
distance = frame["md_since"].to_numpy(np.float64)
bins = {
    "0_500": distance <= 500.0,
    "500_1000": (distance > 500.0) & (distance <= 1000.0),
    "1000_2000": (distance > 1000.0) & (distance <= 2000.0),
    "2000_inf": distance > 2000.0,
}
distance_rho = {
    name: float(np.corrcoef(base_error[mask], candidate_error[mask])[0, 1])
    for name, mask in bins.items()
    if mask.sum() > 2
}

application_gain = application_base["rmse"] - application_blend["rmse"]
application_far_gain = application_base["far_gt2000_rmse"] - application_blend["far_gt2000_rmse"]
reverse_loss = application_reverse["rmse"] - application_blend["rmse"]
checks = {
    "selected_weight_ge_0p025": selected_weight >= 0.025,
    "application_gain_ge_0p05": application_gain >= 0.05,
    "application_far_gain_ge_0p075": application_far_gain >= 0.075,
    "application_p95_change_le_0p20": application_blend["p95_well_rmse"] <= application_base["p95_well_rmse"] + 0.20,
    "reversed_alignment_loses_ge_0p03": reverse_loss >= 0.03,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold3_fixed_weight_selection_fold2_raw_gr_map_confirmation",
    "sample_unit": "complete physical well",
    "sources": {"physics": PHYSICS_PATH, "base": BASE_PATH},
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "selection_wells": int(selection["well"].nunique()),
        "application_fold": APPLICATION_FOLD,
        "application_wells": int(application["well"].nunique()),
        "fold1_touched": False,
    },
    "selected": {"mode": selected_mode, "weight": selected_weight},
    "correlations": {
        "raw_signed_error_rho": raw_rho,
        "within_well_centred_error_rho": centred_rho,
        "distance_bin_error_rho": distance_rho,
    },
    "selection": {"base": selection_base, "candidate": selection_candidate, "blend": selection_blend},
    "application": {
        "base": application_base,
        "candidate": application_candidate,
        "blend": application_blend,
        "reversed_delta_control": application_reverse,
        "gain": application_gain,
        "far_gain": application_far_gain,
        "reversed_alignment_loss": reverse_loss,
    },
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["next"] = (
    "PASS -> retain one frozen raw-GR endpoint dose for Kaggle packaging"
    if verdict["strict_pass"]
    else "FAIL -> raw GR beam is independent but not blend-useful; close the physical endpoint branch"
)

scan.to_csv(OUT_DIR / "fold3_blend_scan.csv", index=False)
output = application[["id", "well", "fold", "tvt_true", "md_since", BASE_COLUMN, "gr_map"]].copy()
output["blend"] = application_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold2_application_oof.parquet", index=False, compression="zstd")
(OUT_DIR / "result.json").write_text(json.dumps(clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("P2 SELECTED", {"mode": selected_mode, "weight": selected_weight}, flush=True)
print("\nP2 FOLD3 SCAN")
print(scan.sort_values("rmse").head(20).to_string(index=False), flush=True)
print("\nP2 FINAL VERDICT")
print(json.dumps(clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
