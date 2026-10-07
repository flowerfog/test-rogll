"""Test the V67 coarse endpoint on the reproducible pre-hedge 6.361 proxy.

The public 6.361 notebook and the 6.377 notebook become identical after the
visible-prefix/contact layers and remain identical through PF branch hedge and
Q0522, until the latter appends tail-only Neighbor.  Q0522 is hidden-ID-specific
and has no honest OOF analogue.  The closest reproducible parent is therefore
V18 ``full_parent_matched_vp_replay`` immediately before PF hedge/Q0522.

This read-only AutoDL cell selects one coarse blend weight on canonical Fold 2
and confirms it once on untouched Fold 1.  It also reports the CP8-before-
Neighbor parent as a secondary lineage diagnostic.  No model is retrained and
Fold 1 never selects a parameter.
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
EXPERIMENT = "rawdata_lattice_coarse_prehedge_parent_audit_v71"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V18_PATH = (
    OOF_ROOT
    / "frontier_6p39_full_oof_cp_increment_v18"
    / "full_oof_cp_endpoints.parquet"
)
V67_PATH = (
    OOF_ROOT
    / "rawdata_lattice_coarse_canonical_gateway_v67"
    / "canonical_fold12_coarse_oof.parquet"
)
V66_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
TRUTH_COLUMN = "tvt_true"
COARSE_COLUMN = "coarse_path"
PRIMARY_PARENT = "full_parent_matched_vp_replay"
SECONDARY_PARENT = "full_cp8_ordered"
SELECTION_FOLD = 2
CONFIRMATION_FOLD = 1
WEIGHT_GRID = np.round(np.arange(0.0, 0.5001, 0.05), 2)


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


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH_COLUMN].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid endpoint {endpoint}")
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    for name, mask in {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _blend(frame, parent_column, weight):
    parent = frame[parent_column].to_numpy(np.float64)
    coarse = frame[COARSE_COLUMN].to_numpy(np.float64)
    return parent + float(weight) * (coarse - parent)


def _signed_error_correlation(frame, parent_column):
    truth = frame[TRUTH_COLUMN].to_numpy(np.float64)
    parent_error = frame[parent_column].to_numpy(np.float64) - truth
    coarse_error = frame[COARSE_COLUMN].to_numpy(np.float64) - truth
    return float(np.corrcoef(parent_error, coarse_error)[0, 1])


for required in (V18_PATH, V67_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v18 = pd.read_parquet(
    V18_PATH,
    columns=[
        "id", "well", "row_idx", "fold", TRUTH_COLUMN, "md_since",
        PRIMARY_PARENT, SECONDARY_PARENT,
    ],
)
v67 = pd.read_parquet(
    V67_PATH,
    columns=["id", "fold", TRUTH_COLUMN, COARSE_COLUMN],
).rename(columns={TRUTH_COLUMN: "coarse_truth"})
for label, source in (("V18", v18), ("V67", v67)):
    if source["id"].duplicated().any():
        raise RuntimeError(f"{label} contains duplicate ids")

frame = v18[v18["fold"].isin((SELECTION_FOLD, CONFIRMATION_FOLD))].merge(
    v67, on=["id", "fold"], how="inner", validate="one_to_one"
)
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
truth_gap = float(
    np.max(np.abs(frame[TRUTH_COLUMN].to_numpy(np.float64) - frame["coarse_truth"].to_numpy(np.float64)))
)
if truth_gap > 0.01:
    raise RuntimeError(f"V18/V67 truth mismatch: {truth_gap}")
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155}:
    raise RuntimeError(f"Incomplete matched Fold1/2 coverage: {fold_wells}")

scan_rows = []
results = {}
summary_rows = []
for parent_column in (PRIMARY_PARENT, SECONDARY_PARENT):
    selection = frame[frame["fold"].eq(SELECTION_FOLD)].copy()
    confirmation = frame[frame["fold"].eq(CONFIRMATION_FOLD)].copy()
    for weight in WEIGHT_GRID:
        metric = _metric(selection, _blend(selection, parent_column, weight), f"blend_w{weight:.2f}")
        scan_rows.append({"parent": parent_column, "weight": float(weight), **metric})
    parent_scan = pd.DataFrame([row for row in scan_rows if row["parent"] == parent_column])
    selected_weight = float(
        parent_scan.sort_values(["exact_sse", "weight"], kind="stable").iloc[0]["weight"]
    )
    parent_metrics = {}
    for fold, local in ((2, selection), (1, confirmation)):
        predictions = {
            "parent": local[parent_column].to_numpy(np.float64),
            "coarse": local[COARSE_COLUMN].to_numpy(np.float64),
            "selected_blend": _blend(local, parent_column, selected_weight),
            "fixed_w0p30": _blend(local, parent_column, 0.30),
        }
        parent_metrics[str(fold)] = {}
        for endpoint, prediction in predictions.items():
            metric = _metric(local, prediction, endpoint)
            parent_metrics[str(fold)][endpoint] = metric
            summary_rows.append({"parent": parent_column, "fold": fold, **metric})
    results[parent_column] = {
        "selected_weight_on_fold2": selected_weight,
        "signed_error_correlation_fold2": _signed_error_correlation(selection, parent_column),
        "signed_error_correlation_fold1": _signed_error_correlation(confirmation, parent_column),
        "metrics": parent_metrics,
        "fold2_gain": (
            parent_metrics["2"]["parent"]["rmse"]
            - parent_metrics["2"]["selected_blend"]["rmse"]
        ),
        "fold1_gain": (
            parent_metrics["1"]["parent"]["rmse"]
            - parent_metrics["1"]["selected_blend"]["rmse"]
        ),
        "fold1_far_gain": (
            parent_metrics["1"]["parent"]["far_gt2000_rmse"]
            - parent_metrics["1"]["selected_blend"]["far_gt2000_rmse"]
        ),
        "fold1_p95_gain": (
            parent_metrics["1"]["parent"]["p95_well_rmse"]
            - parent_metrics["1"]["selected_blend"]["p95_well_rmse"]
        ),
    }

pd.DataFrame(scan_rows).to_csv(OUT_DIR / "fold2_weight_scan.csv", index=False)
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

primary = results[PRIMARY_PARENT]
checks = {
    "primary_selected_weight_positive": bool(primary["selected_weight_on_fold2"] > 0.0),
    "primary_fold2_gain_ge_0p05": bool(primary["fold2_gain"] >= 0.05),
    "primary_fold1_gain_ge_0p05": bool(primary["fold1_gain"] >= 0.05),
    "primary_fold1_far_gain_nonnegative": bool(primary["fold1_far_gain"] >= 0.0),
    "primary_fold1_p95_gain_nonnegative": bool(primary["fold1_p95_gain"] >= 0.0),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_weight_selection_Fold1_confirmation_on_prehedge_6p361_proxy",
    "read_only": True,
    "placement": "after visible-prefix/model-package and before PF branch hedge/Q0522",
    "primary_parent": PRIMARY_PARENT,
    "secondary_parent": SECONDARY_PARENT,
    "scope_limit": (
        "Q0522 and hidden contact overrides have no honest OOF analogue; this tests the "
        "last reproducible parent immediately before those hidden/test-only layers"
    ),
    "alignment": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "fold_wells": fold_wells,
        "maximum_truth_gap": truth_gap,
    },
    "weight_grid": WEIGHT_GRID.tolist(),
    "results": results,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build a 6.361 notebook variant inserting coarse blend before PF hedge/Q0522"
        if strict_pass
        else "FAIL -> coarse fusion is lineage-specific; retain the 6.361 submission unchanged"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV71 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV71 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)

