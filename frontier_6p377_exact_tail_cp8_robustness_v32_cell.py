"""Retrospective tail-only CP8 audit on the exact 6.377 notebook lineage.

The baseline is ``old_frozen_tail`` from V27, which reproduces the notebook's
final 0.20 Neighbor dose (1000-ft freeze, 250-ft smoothstep, 24-ft raw cap).
The alternative endpoint is ``full_cp8_ordered`` from the notebook-matched
``full_oof_cp_endpoints.parquet``. No endpoint or preprocessing choice is
selected here; only a conservative fixed blend weight is scanned.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V27_PATH = OOF_ROOT / "frontier_6p39_tail_lowpass_profile_v27" / "tail_lowpass_oof.parquet"
CP_PATH = OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18" / "full_oof_cp_endpoints.parquet"
OUT_DIR = OOF_ROOT / "frontier_6p377_exact_tail_cp8_robustness_v32"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "old_frozen_tail"
CP_COLUMN = "full_cp8_ordered"
ENDPOINT_DOSE = 0.40
START_FT = 1000.0
TRANSITION_FT = 250.0
LOWPASS_WINDOW_FT = 256.0
CAP_FT = 24.0
WEIGHTS = np.round(np.arange(0.0, 0.401, 0.025), 3)
FOLDS = (3, 2, 1)
P95_TOLERANCE_FT = 0.10
MIN_EACH_FOLD_GAIN = 0.02
MIN_EACH_FAR_GAIN = 0.02


def smoothstep(distance: np.ndarray) -> np.ndarray:
    t = np.clip((distance - START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def centred_lowpass(md: np.ndarray, values: np.ndarray) -> np.ndarray:
    half = 0.5 * LOWPASS_WINDOW_FT
    left = np.searchsorted(md, md - half, side="left")
    right = np.searchsorted(md, md + half, side="right")
    csum = np.r_[0.0, np.cumsum(values)]
    return (csum[right] - csum[left]) / np.maximum(right - left, 1)


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    per_well = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
        .groupby("well", sort=False)["error2"].mean().pow(0.5)
    )
    near = distance <= 1000.0
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


for path in (V27_PATH, CP_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)

base_frame = pd.read_parquet(
    V27_PATH,
    columns=["id", "well", "row_idx", "fold", "tvt_true", "md_since", "parent", BASE_COLUMN],
)
cp_frame = pd.read_parquet(CP_PATH, columns=["id", CP_COLUMN])
base_frame["id"] = base_frame["id"].astype(str)
cp_frame["id"] = cp_frame["id"].astype(str)
frame = base_frame.merge(cp_frame, on="id", how="left", validate="one_to_one")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

required = ["tvt_true", "md_since", "parent", BASE_COLUMN, CP_COLUMN]
if frame[required].isna().any().any():
    raise RuntimeError("Non-finite or missing values after exact-lineage alignment")

truth = frame["tvt_true"].to_numpy(np.float64)
parent = frame["parent"].to_numpy(np.float64)
base = frame[BASE_COLUMN].to_numpy(np.float64)
cp8 = frame[CP_COLUMN].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold_values = frame["fold"].to_numpy(np.int8)

filtered_delta = np.zeros(len(frame), np.float64)
for _, indices in frame.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    local_md = distance[indices]
    if np.any(np.diff(local_md) < 0.0):
        raise RuntimeError("Rows are not MD ordered within a well")
    filtered_delta[indices] = centred_lowpass(local_md, cp8[indices] - parent[indices])

endpoint = parent + ENDPOINT_DOSE * smoothstep(distance) * np.clip(filtered_delta, -CAP_FT, CAP_FT)
direction = endpoint - base

# The exact notebook promise: no candidate may move the first 1000 ft.
if not np.array_equal(direction[distance <= START_FT], np.zeros(np.sum(distance <= START_FT))):
    raise RuntimeError("Exact prefix freeze violated")

print(
    "V32 EXACT-LINEAGE CONFIG",
    {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "folds": sorted(np.unique(fold_values).tolist()),
        "base": BASE_COLUMN,
        "cp_endpoint": CP_COLUMN,
        "endpoint_dose": ENDPOINT_DOSE,
        "lowpass_ft": LOWPASS_WINDOW_FT,
        "cap_ft": CAP_FT,
    },
    flush=True,
)

base_metrics = {
    fold: metrics(frame.loc[fold_values == fold].reset_index(drop=True), base[fold_values == fold])
    for fold in FOLDS
}
scan_rows = []
for weight in WEIGHTS:
    prediction = base + float(weight) * direction
    local_rows = []
    for fold in FOLDS:
        mask = fold_values == fold
        result = metrics(frame.loc[mask].reset_index(drop=True), prediction[mask])
        reference = base_metrics[fold]
        row = {
            "weight": float(weight),
            "fold": int(fold),
            **result,
            "gain": reference["rmse"] - result["rmse"],
            "far_gain": reference["far_gt2000_rmse"] - result["far_gt2000_rmse"],
            "p95_change": result["p95_well_rmse"] - reference["p95_well_rmse"],
            "near_change": result["near_le1000_rmse"] - reference["near_le1000_rmse"],
        }
        local_rows.append(row)
    gains = np.array([row["gain"] for row in local_rows])
    far_gains = np.array([row["far_gain"] for row in local_rows])
    p95_changes = np.array([row["p95_change"] for row in local_rows])
    near_changes = np.array([row["near_change"] for row in local_rows])
    robust = bool(
        gains.min() >= MIN_EACH_FOLD_GAIN
        and far_gains.min() >= MIN_EACH_FAR_GAIN
        and p95_changes.max() <= P95_TOLERANCE_FT
        and np.max(np.abs(near_changes)) <= 1e-8
    )
    for row in local_rows:
        row["robust"] = robust
        scan_rows.append(row)

scan = pd.DataFrame(scan_rows)
summary = (
    scan.groupby("weight", as_index=False)
    .agg(
        mean_gain=("gain", "mean"),
        minimum_gain=("gain", "min"),
        mean_far_gain=("far_gain", "mean"),
        minimum_far_gain=("far_gain", "min"),
        maximum_p95_change=("p95_change", "max"),
        maximum_abs_near_change=("near_change", lambda x: float(np.max(np.abs(x)))),
        robust=("robust", "all"),
    )
)
eligible = summary[summary["robust"]]
selected = (
    eligible.sort_values(["minimum_gain", "mean_gain", "weight"], ascending=[False, False, True]).iloc[0]
    if len(eligible)
    else summary.sort_values(["maximum_p95_change", "minimum_gain"], ascending=[True, False]).iloc[0]
)
selected_weight = float(selected["weight"])
selected_prediction = base + selected_weight * direction

scan.to_csv(OUT_DIR / "dose_by_fold.csv", index=False)
summary.to_csv(OUT_DIR / "dose_summary.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    exact_6p377_base=base.astype(np.float32),
    cp8_tail_endpoint=endpoint.astype(np.float32),
    selected_exact_tail=(selected_prediction.astype(np.float32)),
).to_parquet(OUT_DIR / "exact_tail_oof.parquet", index=False, compression="zstd")

selected_by_fold = scan[scan["weight"].eq(selected_weight)]
strict_pass = bool(selected["robust"])
verdict = {
    "experiment": "frontier_6p377_exact_tail_cp8_robustness_v32",
    "stage": "retrospective_exact_6p377_lineage_fixed_tail_dose",
    "sources": {"v27": str(V27_PATH), "notebook_oof": str(CP_PATH)},
    "base_column": BASE_COLUMN,
    "cp_column": CP_COLUMN,
    "selected_weight": selected_weight,
    "selected_summary": {
        key: (bool(value) if key == "robust" else float(value))
        for key, value in selected.to_dict().items()
    },
    "selected_by_fold": selected_by_fold.to_dict(orient="records"),
    "strict_pass": strict_pass,
    "validation_status": "retrospective_only_folds_1_2_3_already_observed",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> translate this fixed delta into one final Kaggle cell on top of the 6.377 notebook"
        if strict_pass else
        "FAIL -> V31 gain came from the research-selected V27 baseline, not the exact 6.377 endpoint"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nV32 EXACT-LINEAGE DOSE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV32 SELECTED BY FOLD")
print(selected_by_fold.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
