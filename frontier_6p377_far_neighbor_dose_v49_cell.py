"""Nested far-only Neighbor dose calibration on the exact V33 lineage.

V33 applies one 0.20 Neighbor dose after 1000 ft.  This audit preserves that
path bitwise through 2000 ft and asks whether only the far tail should use a
different dose.  For every held fold, dose and transition width are selected
using the other two folds.  No labels from the held fold enter selection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_far_neighbor_dose_v49"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
SOURCE = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = (1, 2, 3)
BASE_DOSE = 0.20
FAR_START_FT = 2000.0
DOSES = np.round(np.arange(0.0, 0.401, 0.025), 3)
TRANSITIONS_FT = (128.0, 256.0, 512.0)
CAP_FT = 24.0
P95_TOLERANCE_FT = 0.10


def _smoothstep(distance: np.ndarray, width: float) -> np.ndarray:
    t = np.clip((distance - FAR_START_FT) / width, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _metrics(local: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    per_well = (
        pd.DataFrame({"well": local["well"].astype(str), "error2": error**2})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    near = distance <= FAR_START_FT
    far = distance > FAR_START_FT
    return {
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "near_le2000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _candidate(local: pd.DataFrame, dose: float, transition_ft: float) -> np.ndarray:
    base = local["current_dose_0p20"].to_numpy(np.float64)
    cp8 = local["pre_neighbor_cp8"].to_numpy(np.float64)
    neighbor = local["neighbor_absolute"].to_numpy(np.float64)
    active = local["neighbor_active"].fillna(False).to_numpy(bool)
    distance = local["md_since"].to_numpy(np.float64)
    move = np.where(active, np.clip(neighbor - cp8, -CAP_FT, CAP_FT), 0.0)
    far_weight = _smoothstep(distance, transition_ft)
    output = base + far_weight * (float(dose) - BASE_DOSE) * move
    output[distance <= FAR_START_FT] = base[distance <= FAR_START_FT]
    return output


def _combined_metrics(local: pd.DataFrame, config: dict) -> dict:
    return _metrics(local, _candidate(local, config["dose"], config["transition_ft"]))


started = time.time()
if not SOURCE.is_file():
    raise FileNotFoundError(f"Run V33 first: {SOURCE}")

frame = pd.read_parquet(SOURCE)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "pre_neighbor_cp8", "neighbor_absolute", "current_dose_0p20",
    "neighbor_active",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V33 source missing columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
numeric = [
    "tvt_true", "md_since", "pre_neighbor_cp8", "neighbor_absolute",
    "current_dose_0p20",
]
if frame[numeric].isna().any().any():
    raise RuntimeError("Non-finite values in exact V33 lineage")
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in V33 lineage")
if sorted(frame["fold"].astype(int).unique().tolist()) != list(FOLDS):
    raise RuntimeError(f"Unexpected folds: {sorted(frame['fold'].unique().tolist())}")

configs = [
    {"dose": float(dose), "transition_ft": float(width)}
    for width in TRANSITIONS_FT
    for dose in DOSES
]
baseline_config = {"dose": BASE_DOSE, "transition_ft": TRANSITIONS_FT[0]}
baseline = frame["current_dose_0p20"].to_numpy(np.float64)
near = frame["md_since"].to_numpy(np.float64) <= FAR_START_FT

fold_reports = []
oof_prediction = baseline.copy()
for held_fold in FOLDS:
    train = frame[frame["fold"].astype(int) != held_fold].copy()
    held = frame[frame["fold"].astype(int) == held_fold].copy()
    train_base = _metrics(train, train["current_dose_0p20"].to_numpy(np.float64))
    scan = []
    for config in configs:
        metric = _combined_metrics(train, config)
        scan.append({
            **config,
            **metric,
            "gain": train_base["rmse"] - metric["rmse"],
            "far_gain": train_base["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - train_base["p95_well_rmse"],
        })
    scan_frame = pd.DataFrame(scan)
    safe = scan_frame[scan_frame["p95_change"] <= P95_TOLERANCE_FT].copy()
    if safe.empty:
        selected = pd.Series({**baseline_config, "gain": 0.0, "far_gain": 0.0})
    else:
        selected = safe.sort_values(
            ["exact_sse", "far_gt2000_rmse", "transition_ft", "dose"],
            ascending=[True, True, False, True],
        ).iloc[0]
    held_prediction = _candidate(
        held, float(selected["dose"]), float(selected["transition_ft"])
    )
    held_base = _metrics(held, held["current_dose_0p20"].to_numpy(np.float64))
    held_metric = _metrics(held, held_prediction)
    positions = frame.index[frame["fold"].astype(int) == held_fold].to_numpy()
    oof_prediction[positions] = held_prediction
    fold_reports.append({
        "fold": int(held_fold),
        "selected_dose": float(selected["dose"]),
        "selected_transition_ft": float(selected["transition_ft"]),
        "inner_gain": float(selected["gain"]),
        "inner_far_gain": float(selected["far_gain"]),
        "base_rmse": held_base["rmse"],
        "rmse": held_metric["rmse"],
        "gain": held_base["rmse"] - held_metric["rmse"],
        "base_far_rmse": held_base["far_gt2000_rmse"],
        "far_rmse": held_metric["far_gt2000_rmse"],
        "far_gain": held_base["far_gt2000_rmse"] - held_metric["far_gt2000_rmse"],
        "p95_change": held_metric["p95_well_rmse"] - held_base["p95_well_rmse"],
    })

if not np.array_equal(oof_prediction[near], baseline[near]):
    raise RuntimeError("Rows through 2000 ft changed")

base_metrics = _metrics(frame, baseline)
oof_metrics = _metrics(frame, oof_prediction)
fold_table = pd.DataFrame(fold_reports)
gain = base_metrics["rmse"] - oof_metrics["rmse"]
far_gain = base_metrics["far_gt2000_rmse"] - oof_metrics["far_gt2000_rmse"]
checks = {
    "overall_gain_ge_0p03": bool(gain >= 0.03),
    "far_gain_ge_0p05": bool(far_gain >= 0.05),
    "minimum_fold_gain_ge_0": bool(fold_table["gain"].min() >= 0.0),
    "minimum_fold_far_gain_ge_0": bool(fold_table["far_gain"].min() >= 0.0),
    "maximum_p95_change_le_0p10": bool(fold_table["p95_change"].max() <= 0.10),
    "near_rows_bitwise_unchanged": True,
}
strict_pass = bool(all(checks.values()))

fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    base_6p377=baseline.astype(np.float32),
    far_neighbor_dose=oof_prediction.astype(np.float32),
).to_parquet(OUT_DIR / "far_neighbor_dose_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_far_only_neighbor_increment_dose",
    "parent": str(SOURCE),
    "validation_status": "retrospective_threefold_nested; no untouched-fold claim",
    "frozen_through_ft": FAR_START_FT,
    "base_dose": BASE_DOSE,
    "candidate_doses": DOSES.tolist(),
    "transition_widths_ft": list(TRANSITIONS_FT),
    "base": base_metrics,
    "candidate": oof_metrics,
    "gain": float(gain),
    "far_gain": float(far_gain),
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> translate the fold-stable far-dose schedule into Kaggle inference"
        if strict_pass
        else "FAIL -> retain the exact 6.377 Neighbor dose and close far-dose calibration"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("V49 FAR-NEIGHBOR FOLDS")
print(fold_table.to_string(index=False), flush=True)
print("\nV49 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
