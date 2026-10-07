"""Audit the stability of the All773 coarse dose on the complete 6.377 OOF.

Fold 3 selects one dose.  Folds 2 and 1 are read-only confirmations.  The cell
also reports each fold's unconstrained least-squares optimum for diagnosis.
No model is trained and no endpoint is regenerated.
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
EXPERIMENT = "rawdata_lattice_coarse_all773_dose_stability_v73"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = (
    OOF_ROOT
    / "rawdata_lattice_coarse_all773_crossfit_v70"
    / "all773_proxy_crossfit_oof.parquet"
)
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
SELECTION_FOLD = 3
CONFIRMATION_FOLDS = (2, 1)
REFERENCE_WEIGHT = 0.30
WEIGHTS = np.round(np.arange(0.10, 0.4001, 0.025), 3)


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


def _prediction(frame, weight):
    parent = frame[PARENT].to_numpy(np.float64)
    coarse = frame[COARSE].to_numpy(np.float64)
    return parent + float(weight) * (coarse - parent)


def _continuous_optimum(frame):
    parent = frame[PARENT].to_numpy(np.float64)
    move = frame[COARSE].to_numpy(np.float64) - parent
    target = frame[TRUTH].to_numpy(np.float64) - parent
    denominator = float(np.dot(move, move))
    return float(np.dot(move, target) / denominator) if denominator > 0.0 else 0.0


def _metric(frame, prediction, endpoint, weight):
    truth = frame[TRUTH].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "weight": float(weight),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(
    SOURCE,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
)
if frame["id"].duplicated().any():
    raise RuntimeError("V70 source contains duplicate ids")
if frame[[TRUTH, "md_since", PARENT, COARSE]].isna().any().any():
    raise RuntimeError("V70 source contains missing values")
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError(f"Unexpected fold coverage: {fold_wells}")

scan_rows = []
continuous = {}
for fold in (1, 2, 3):
    local = frame[frame["fold"].eq(fold)]
    continuous[str(fold)] = _continuous_optimum(local)
    for weight in WEIGHTS:
        scan_rows.append(
            {
                "fold": fold,
                **_metric(local, _prediction(local, weight), "blend", weight),
            }
        )
scan = pd.DataFrame(scan_rows)
selection_scan = scan[scan["fold"].eq(SELECTION_FOLD)].sort_values(
    ["exact_sse", "weight"], kind="stable"
)
selected_weight = float(selection_scan.iloc[0]["weight"])
scan.to_csv(OUT_DIR / "fold_weight_scan.csv", index=False)

summary_rows = []
metrics = {}
for fold in (3, 2, 1):
    local = frame[frame["fold"].eq(fold)]
    predictions = {
        "parent": (local[PARENT].to_numpy(np.float64), 0.0),
        "reference_w0p30": (_prediction(local, REFERENCE_WEIGHT), REFERENCE_WEIGHT),
        "selected_weight": (_prediction(local, selected_weight), selected_weight),
    }
    metrics[str(fold)] = {}
    for endpoint, (prediction, weight) in predictions.items():
        result = _metric(local, prediction, endpoint, weight)
        metrics[str(fold)][endpoint] = result
        summary_rows.append({"fold": fold, **result})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

gains = {}
for fold in (3, 2, 1):
    local = metrics[str(fold)]
    gains[str(fold)] = {
        "selected_vs_reference": float(
            local["reference_w0p30"]["rmse"] - local["selected_weight"]["rmse"]
        ),
        "selected_far_vs_reference": float(
            local["reference_w0p30"]["far_gt2000_rmse"]
            - local["selected_weight"]["far_gt2000_rmse"]
        ),
        "selected_p95_vs_reference": float(
            local["reference_w0p30"]["p95_well_rmse"]
            - local["selected_weight"]["p95_well_rmse"]
        ),
    }

checks = {
    "selected_weight_between_0p25_and_0p35": bool(0.25 <= selected_weight <= 0.35),
    "fold2_not_worse_by_gt_0p02": bool(gains["2"]["selected_vs_reference"] >= -0.02),
    "fold1_not_worse_by_gt_0p02": bool(gains["1"]["selected_vs_reference"] >= -0.02),
    "fold2_far_not_worse": bool(gains["2"]["selected_far_vs_reference"] >= 0.0),
    "fold1_far_not_worse": bool(gains["1"]["selected_far_vs_reference"] >= 0.0),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_dose_selection_Fold2_and_Fold1_confirmation",
    "read_only": True,
    "parent": "complete_6p377_CP8_plus_tail_only_Neighbor",
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_folds": list(CONFIRMATION_FOLDS),
        "weight_grid": WEIGHTS.tolist(),
        "reference_weight": REFERENCE_WEIGHT,
    },
    "coverage": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "fold_wells": fold_wells,
    },
    "continuous_optimum_by_fold": continuous,
    "selected_weight": selected_weight,
    "metrics": metrics,
    "gains": gains,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> use the Fold3-selected stable All773 dose"
        if strict_pass
        else "FAIL -> retain the preregistered All773 dose 0.30"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV73 PER-FOLD OPTIMA")
print(json.dumps(_clean(continuous), indent=2), flush=True)
print("\nV73 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV73 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
