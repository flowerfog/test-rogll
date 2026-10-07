"""V211b: read-only downstream survival audit of the V211 PF-core delta.

This does not claim to replay nonlinear CP8/Neighbor processing.  It aligns
the frozen x2-x1 paired delta to complete downstream OOF endpoints and asks
whether that direction remains useful after the historical downstream chain.
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
OUT = OOF / "mainline_pf_particle_delta_survival_v211b"
OUT.mkdir(parents=True, exist_ok=True)
PAIRED = OOF / "mainline_pf_particle_count_full773_v211" / "paired_predictions.parquet"
EXPECTED_ROWS = 3_783_989
FIXED_DOSES = (0.0, 0.35, 0.70, 1.0)

ENDPOINTS = (
    (
        "v66_incumbent",
        OOF / "frontier_6p377_full_fivefold_lineage_v66"
        / "incumbent_6p377_full_fivefold_oof.parquet",
        "incumbent_6p377_replay",
    ),
    (
        "v144_sequence",
        OOF / "final_consensus_sequence_coefficient_full5_v144"
        / "crossfit_oof.parquet",
        "nested_sequence_coefficient",
    ),
    (
        "v168_metadata_analog",
        OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet",
        "metadata_analog_w0p05",
    ),
    (
        "v169_nested_consensus",
        OOF / "historical_analog_ranked_consensus_v169" / "fullrow_oof.parquet",
        "nested_consensus",
    ),
)


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["truth"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    e2 = np.square(error)
    distance = frame["md_since"].to_numpy(np.float64)
    by_well = pd.DataFrame({"well": frame["well"].astype(str), "e2": e2}).groupby("well")["e2"].mean().pow(0.5)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "rmse": float(np.sqrt(e2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(by_well.median()),
        "p95_well_rmse": float(by_well.quantile(0.95)),
        "far_gt2000_rmse": float(np.sqrt(e2[far].mean())) if far.any() else float("nan"),
    }


if not PAIRED.is_file():
    raise FileNotFoundError(PAIRED)
long = pd.read_parquet(PAIRED, columns=["id", "well", "md_since", "truth", "prediction", "variant"])
long["id"] = long["id"].astype(str)
long["well"] = long["well"].astype(str)
keys = ["id", "well", "md_since", "truth"]
wide = long.pivot(index=keys, columns="variant", values="prediction").reset_index()
if len(wide) != EXPECTED_ROWS or not {"particles_x1", "particles_x2"}.issubset(wide.columns):
    raise RuntimeError(f"Incomplete V211 paired matrix: rows={len(wide)}, columns={wide.columns.tolist()}")
wide["delta"] = wide["particles_x2"].to_numpy(np.float64) - wide["particles_x1"].to_numpy(np.float64)

summary_rows = []
fold_rows = []
endpoint_reports = []
for endpoint_name, path, prediction_column in ENDPOINTS:
    if not path.is_file():
        endpoint_reports.append({"endpoint": endpoint_name, "status": "missing", "path": str(path)})
        continue
    header = pd.read_parquet(path).columns.tolist()
    if prediction_column not in header:
        endpoint_reports.append(
            {"endpoint": endpoint_name, "status": "missing_column", "path": str(path), "column": prediction_column}
        )
        continue
    truth_column = next((column for column in ("tvt_true", "truth", "TVT") if column in header), None)
    required = ["id", "well", prediction_column]
    for optional in ("fold", "md_since", truth_column):
        if optional is not None and optional in header and optional not in required:
            required.append(optional)
    parent = pd.read_parquet(path, columns=required)
    parent["id"] = parent["id"].astype(str)
    parent["well"] = parent["well"].astype(str)
    aligned = wide.merge(parent, on=["id", "well"], how="inner", validate="one_to_one", suffixes=("", "_parent"))
    if len(aligned) != EXPECTED_ROWS:
        raise RuntimeError(f"{endpoint_name}: only {len(aligned)} aligned rows")
    if truth_column is not None:
        gap = np.max(np.abs(aligned["truth"].to_numpy(np.float64) - aligned[truth_column].to_numpy(np.float64)))
        if gap > 0.002:
            raise RuntimeError(f"{endpoint_name}: truth gap {gap}")
    if "fold" not in aligned:
        raise RuntimeError(f"{endpoint_name}: fold column is required")

    base = aligned[prediction_column].to_numpy(np.float64)
    truth = aligned["truth"].to_numpy(np.float64)
    delta = aligned["delta"].to_numpy(np.float64)
    denominator = float(np.dot(delta, delta))
    analytic = float(np.clip(-np.dot(base - truth, delta) / denominator, -1.0, 2.0)) if denominator > 0 else 0.0
    doses = tuple(dict.fromkeys((*FIXED_DOSES, analytic)))
    base_metric = _metric(aligned, base, endpoint_name)
    for dose in doses:
        label = f"dose_{dose:.6f}" if dose == analytic else f"dose_{dose:.2f}"
        candidate_metric = _metric(aligned, base + dose * delta, label)
        summary_rows.append(
            {
                "parent": endpoint_name,
                "dose": float(dose),
                "analytic_oracle": int(dose == analytic),
                **candidate_metric,
                "gain": base_metric["rmse"] - candidate_metric["rmse"],
                "far_gain": base_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"],
                "p95_gain": base_metric["p95_well_rmse"] - candidate_metric["p95_well_rmse"],
            }
        )
        if dose in (0.70, 1.0):
            for fold, local in aligned.groupby("fold", sort=True):
                index = local.index.to_numpy()
                # Merge preserves a RangeIndex, so positions and indices coincide.
                local_base = base[index]
                local_candidate = local_base + dose * delta[index]
                local_base_metric = _metric(local, local_base, "base")
                local_candidate_metric = _metric(local, local_candidate, "candidate")
                fold_rows.append(
                    {
                        "parent": endpoint_name,
                        "dose": float(dose),
                        "fold": int(fold),
                        "wells": int(local["well"].nunique()),
                        "base_rmse": local_base_metric["rmse"],
                        "rmse": local_candidate_metric["rmse"],
                        "gain": local_base_metric["rmse"] - local_candidate_metric["rmse"],
                        "far_gain": local_base_metric["far_gt2000_rmse"] - local_candidate_metric["far_gt2000_rmse"],
                        "p95_gain": local_base_metric["p95_well_rmse"] - local_candidate_metric["p95_well_rmse"],
                    }
                )
    endpoint_reports.append(
        {
            "endpoint": endpoint_name,
            "status": "complete",
            "path": str(path),
            "prediction_column": prediction_column,
            "base_rmse": base_metric["rmse"],
            "analytic_alpha": analytic,
            "delta_error_correlation": float(np.corrcoef(delta, base - truth)[0, 1]),
        }
    )

summary = pd.DataFrame(summary_rows)
folds = pd.DataFrame(fold_rows)
if summary.empty:
    raise RuntimeError(f"No complete downstream endpoint was found: {endpoint_reports}")
fixed = summary[summary["analytic_oracle"].eq(0)].copy()
fixed070 = fixed[np.isclose(fixed["dose"], 0.70)]
checks = {
    "complete_endpoint_exists": any(report["status"] == "complete" for report in endpoint_reports),
    "dose0p70_positive_for_all_complete_parents": bool((fixed070["gain"] > 0.0).all()),
    "dose0p70_positive_folds_ge_4_for_each_parent": bool(
        folds[np.isclose(folds["dose"], 0.70)].groupby("parent")["gain"].apply(lambda values: int((values > 0).sum()) >= 4).all()
    ),
    "all_analytic_alphas_positive": all(
        report.get("analytic_alpha", 0.0) > 0.0 for report in endpoint_reports if report["status"] == "complete"
    ),
}
verdict = {
    "experiment": "mainline_pf_particle_delta_survival_v211b",
    "stage": "read_only_complete773_downstream_direction_survival_audit",
    "deployable": 0,
    "selection_or_training_count": 0,
    "delta": "V211 particles_x2 - particles_x1",
    "fixed_doses": list(FIXED_DOSES),
    "dose_0p70_interpretation": "linear attenuation expected after a fixed 0.30 independent-coarse blend",
    "endpoints": endpoint_reports,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> exact CP8/Neighbor replay; FAIL -> V211 core gain is consumed downstream",
}
summary.to_csv(OUT / "dose_scan.csv", index=False)
folds.to_csv(OUT / "folds.csv", index=False)
(OUT / "result.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("V211b DOSE SCAN")
print(summary[["parent", "dose", "analytic_oracle", "rmse", "gain", "far_gain", "p95_gain"]].to_string(index=False), flush=True)
print("\nV211b FOLDS")
print(folds.to_string(index=False), flush=True)
print("\nV211b FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT, flush=True)
