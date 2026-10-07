"""Zero-training re-audit of differentiated endpoints against an exact core OOF.

Run after ``frontier_6p39_existing_models_oof_v0_cell.py`` and the V4/V5
endpoint cells. This cell intentionally refuses proxy references.
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
OUT_DIR = OOF_ROOT / "exact_core_endpoint_reaudit_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CANDIDATES = [
    (ROOT / "mainline_6p47_core_oof.parquet", "mainline_6p47"),
    (OOF_ROOT / "frontier_6p39_existing_models_oof_v0"
     / "frontier_6p39_core_oof.parquet", "frontier_6p39_core"),
]
ENDPOINTS = [
    ("surface_gbdt_v4",
     OOF_ROOT / "decorrelated_surface_gbdt_frontier_v4"
     / "decorrelated_segment_surface_oof.parquet", "selected_candidate"),
    ("surface_bigru_v5",
     OOF_ROOT / "fullwell_surface_bigru_frontier_v5"
     / "fullwell_surface_bigru_oof.parquet", "surface_bigru"),
]
MIN_COVERAGE = 0.995
WEIGHTS = np.round(np.arange(0.0, 0.501, 0.01), 2)
DISTANCE_BANDS = [0.0, 500.0, 1000.0, 2000.0, np.inf]


def _first_reference():
    for path, column in REFERENCE_CANDIDATES:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if column in frame:
            return path, column, frame
    raise FileNotFoundError(
        "No exact model-core OOF found. Run "
        "frontier_6p39_existing_models_oof_v0_cell.py first. "
        "Proxy references are deliberately forbidden."
    )


def _normalise(frame):
    frame = frame.copy()
    if "id" not in frame:
        raise RuntimeError("OOF artifact has no id column")
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise RuntimeError("OOF artifact contains duplicate ids")
    if "well" not in frame:
        frame["well"] = frame["id"].str.rsplit("_", n=1).str[0]
    frame["well"] = frame["well"].astype(str)
    truth = next((c for c in ("truth", "tvt_true", "target") if c in frame), None)
    if truth is None:
        raise RuntimeError("OOF artifact has no truth column")
    if truth != "truth":
        frame = frame.rename(columns={truth: "truth"})
    return frame


def _rmse(truth, prediction):
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(error * error)))


def _metrics(frame, column):
    error = frame[column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    result = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
    }
    well_rmse = pd.Series(error * error).groupby(frame["well"].to_numpy()).mean().pow(0.5)
    result.update({
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    })
    if "md_since" in frame:
        md = frame["md_since"].to_numpy(np.float64)
        names = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
        for name, low, high in zip(names, DISTANCE_BANDS[:-1], DISTANCE_BANDS[1:]):
            mask = (md >= low) & (md < high)
            result[name + "_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
    return result


def _correlations(frame, candidate):
    truth = frame["truth"].to_numpy(np.float64)
    base_error = frame["reference"].to_numpy(np.float64) - truth
    cand_error = frame[candidate].to_numpy(np.float64) - truth
    output = {
        "raw_signed_error_rho": float(np.corrcoef(base_error, cand_error)[0, 1]),
        "raw_error_cosine": float(np.dot(base_error, cand_error)
                                  / np.sqrt(np.dot(base_error, base_error)
                                            * np.dot(cand_error, cand_error))),
    }
    if "md_since" not in frame:
        return output
    md = frame["md_since"].to_numpy(np.float64)
    band = np.digitize(md, DISTANCE_BANDS[1:-1], right=False)
    base_dt = base_error.copy()
    cand_dt = cand_error.copy()
    names = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
    for index, name in enumerate(names):
        mask = band == index
        base_dt[mask] -= np.mean(base_dt[mask])
        cand_dt[mask] -= np.mean(cand_dt[mask])
        output[name + "_error_rho"] = float(
            np.corrcoef(base_error[mask], cand_error[mask])[0, 1]
        )
    output["distance_band_detrended_error_rho"] = float(
        np.corrcoef(base_dt, cand_dt)[0, 1]
    )
    return output


reference_path, reference_column, reference_raw = _first_reference()
reference_raw = _normalise(reference_raw)
required = ["id", "well", "truth", reference_column]
for optional in ("row_idx", "fold", "md_since"):
    if optional in reference_raw:
        required.append(optional)
reference = reference_raw[required].rename(columns={reference_column: "reference"})
if "fold" not in reference:
    raise RuntimeError("Exact reference must contain fold for nested blend selection")

endpoint_reports = []
fold_reports = []
prediction_exports = reference.copy()

for endpoint_name, endpoint_path, endpoint_column in ENDPOINTS:
    if not endpoint_path.is_file():
        print(f"SKIP {endpoint_name}: missing {endpoint_path}", flush=True)
        continue
    endpoint = _normalise(pd.read_parquet(endpoint_path))
    keep = ["id", endpoint_column]
    if "md_since" not in reference and "md_since" in endpoint:
        keep.append("md_since")
    joined = reference.merge(endpoint[keep], on="id", how="inner", validate="one_to_one")
    coverage = len(joined) / len(reference)
    if coverage < MIN_COVERAGE:
        raise RuntimeError(f"{endpoint_name} coverage {coverage:.6%} < {MIN_COVERAGE:.3%}")
    truth_gap_source = reference.set_index("id").loc[joined["id"], "truth"].to_numpy()
    if np.max(np.abs(joined["truth"].to_numpy() - truth_gap_source)) > 1e-3:
        raise RuntimeError(f"{endpoint_name}: truth alignment mismatch")

    nested = np.full(len(joined), np.nan, np.float64)
    for held_fold in sorted(joined["fold"].unique()):
        select = joined["fold"].ne(held_fold).to_numpy()
        held = ~select
        y_select = joined.loc[select, "truth"].to_numpy(np.float64)
        b_select = joined.loc[select, "reference"].to_numpy(np.float64)
        c_select = joined.loc[select, endpoint_column].to_numpy(np.float64)
        scan_rmse = np.array([
            _rmse(y_select, b_select + weight * (c_select - b_select))
            for weight in WEIGHTS
        ])
        weight = float(WEIGHTS[int(np.argmin(scan_rmse))])
        base = joined.loc[held, "reference"].to_numpy(np.float64)
        candidate = joined.loc[held, endpoint_column].to_numpy(np.float64)
        nested[held] = base + weight * (candidate - base)
        held_truth = joined.loc[held, "truth"].to_numpy(np.float64)
        base_rmse = _rmse(held_truth, base)
        blend_rmse = _rmse(held_truth, nested[held])
        fold_reports.append({
            "endpoint": endpoint_name,
            "fold": int(held_fold),
            "weight_candidate": weight,
            "reference_rmse": base_rmse,
            "candidate_rmse": _rmse(held_truth, candidate),
            "blend_rmse": blend_rmse,
            "gain": base_rmse - blend_rmse,
        })

    joined["nested_blend"] = nested
    base_metrics = _metrics(joined, "reference")
    candidate_metrics = _metrics(joined, endpoint_column)
    blend_metrics = _metrics(joined, "nested_blend")
    correlations = _correlations(joined, endpoint_column)
    endpoint_fold_report = [row for row in fold_reports if row["endpoint"] == endpoint_name]
    endpoint_reports.append({
        "endpoint": endpoint_name,
        "path": str(endpoint_path),
        "coverage": float(coverage),
        "reference_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "correlations": correlations,
        "nested_blend_metrics": blend_metrics,
        "nested_blend_gain": float(base_metrics["rmse"] - blend_metrics["rmse"]),
        "positive_folds": int(sum(row["gain"] > 0 for row in endpoint_fold_report)),
        "minimum_fold_gain": float(min(row["gain"] for row in endpoint_fold_report)),
        "standalone_pass": bool(candidate_metrics["rmse"] <= 10.5),
        "fusion_pass": bool(
            base_metrics["rmse"] - blend_metrics["rmse"] >= 0.15
            and sum(row["gain"] > 0 for row in endpoint_fold_report) >= 4
            and min(row["weight_candidate"] for row in endpoint_fold_report) >= 0.05
        ),
    })
    prediction_exports = prediction_exports.merge(
        joined[["id", endpoint_column, "nested_blend"]].rename(columns={
            endpoint_column: endpoint_name,
            "nested_blend": endpoint_name + "_nested_blend",
        }), on="id", how="left", validate="one_to_one"
    )

if not endpoint_reports:
    raise RuntimeError("No V4/V5 endpoint artifacts were found")

verdict = {
    "experiment": "exact_core_endpoint_reaudit_v0",
    "stage": "zero_training_exact_core_differentiated_endpoint_reaudit",
    "optimizer_steps": 0,
    "reference": {"path": str(reference_path), "column": reference_column},
    "endpoints": endpoint_reports,
    "any_standalone_pass": bool(any(item["standalone_pass"] for item in endpoint_reports)),
    "any_fusion_pass": bool(any(item["fusion_pass"] for item in endpoint_reports)),
    "next": (
        "PASS -> freeze the passing endpoint and build deployment refit"
        if any(item["fusion_pass"] for item in endpoint_reports)
        else "FAIL -> close the 426-feature surface representation; change evidence inputs"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024 ** 3),
}

pd.DataFrame(fold_reports).to_csv(OUT_DIR / "fold_report.csv", index=False)
prediction_exports.to_parquet(
    OUT_DIR / "exact_core_endpoint_predictions.parquet", index=False, compression="zstd"
)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nEXACT-CORE ENDPOINT FOLDS")
print(pd.DataFrame(fold_reports).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
