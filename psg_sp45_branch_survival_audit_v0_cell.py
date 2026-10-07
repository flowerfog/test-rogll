"""Audit whether the cross-fitted PSG-SP45 branch survives stronger OOF stages.

Prerequisites:
  1. psg_sp45_crossfit_audit_v0_cell.py
  2. neighbor_fullcurve_projection_direction_v1_cell.py for the frozen
     full_progress_geom neighbor endpoint (optional but required for the final
     deployment gate).

No historical column is called the 6.781 OOF. The closest reproducible endpoint
is reported explicitly as CP12 plus the frozen full_progress_geom neighbor move.
Gold/LB-specific calibration and the stochastic test PF realization are not
claimed by this audit.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


EXPERIMENT = "psg_sp45_branch_survival_audit_v0"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
BRANCH_PATH = OOF_ROOT / "psg_sp45_crossfit_audit_v0" / "psg_sp45_crossfit_oof.parquet"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
NEIGHBOR_ROOT = OOF_ROOT / "neighbor_fullcurve_projection_direction_v1"
NEIGHBOR_CANDIDATE_PATH = NEIGHBOR_ROOT / "primary_candidate_oof.npz"
NEIGHBOR_META_PATH = OOF_ROOT / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_WEIGHT = 0.25
RAW_MOVE_CAP = 24.0
MIN_ABS_DIRECTION_DOT = 0.70
MIN_WEIGHT = 0.02
MIN_GAIN_FT = 0.05
MIN_FOLD_GAIN_FT = -0.10


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _rho(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def _optimal_weight(base_error, candidate_error):
    direction = np.asarray(candidate_error, np.float64) - np.asarray(base_error, np.float64)
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0, 0.0
    raw = float(-np.dot(base_error, direction) / denominator)
    return raw, float(np.clip(raw, 0.0, 1.0))


def _metrics(truth, prediction, wells):
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    codes, names = pd.factorize(np.asarray(wells).astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error**2, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    return {
        "rows": int(len(error)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }


def _audit_pair(frame, base_name, candidate_name):
    truth = frame["truth"].to_numpy(np.float64)
    base = frame[base_name].to_numpy(np.float64)
    candidate = frame[candidate_name].to_numpy(np.float64)
    wells = frame["well"].to_numpy()
    folds = frame["fold"].to_numpy(np.int8)
    base_error = base - truth
    candidate_error = candidate - truth
    output = np.full(len(frame), np.nan, np.float64)
    weights = np.full(len(frame), np.nan, np.float32)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        raw_weight, weight = _optimal_weight(base_error[train], candidate_error[train])
        prediction = base[valid] + weight * (candidate[valid] - base[valid])
        output[valid] = prediction
        weights[valid] = weight
        base_metrics = _metrics(truth[valid], base[valid], wells[valid])
        blend_metrics = _metrics(truth[valid], prediction, wells[valid])
        fold_rows.append(
            {
                "base": base_name,
                "candidate": candidate_name,
                "held_fold": int(fold),
                "weight": weight,
                "unconstrained_weight": raw_weight,
                "base_rmse": base_metrics["rmse"],
                "blend_rmse": blend_metrics["rmse"],
                "gain": base_metrics["rmse"] - blend_metrics["rmse"],
                "held_error_rho": _rho(base_error[valid], candidate_error[valid]),
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError(f"Incomplete crossfit output for {base_name}")
    base_metrics = _metrics(truth, base, wells)
    candidate_metrics = _metrics(truth, candidate, wells)
    blend_metrics = _metrics(truth, output, wells)
    fold_report = pd.DataFrame(fold_rows)
    return {
        "base": base_name,
        "candidate": candidate_name,
        "base_rmse": base_metrics["rmse"],
        "candidate_rmse": candidate_metrics["rmse"],
        "crossfit_rmse": blend_metrics["rmse"],
        "crossfit_gain": base_metrics["rmse"] - blend_metrics["rmse"],
        "raw_error_rho": _rho(base_error, candidate_error),
        "minimum_weight": float(fold_report["weight"].min()),
        "maximum_weight": float(fold_report["weight"].max()),
        "minimum_fold_gain": float(fold_report["gain"].min()),
        "positive_folds": int((fold_report["gain"] > 0.0).sum()),
    }, fold_report, output, weights


started = time.time()
for required in (BRANCH_PATH, POOL_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

branch = pd.read_parquet(BRANCH_PATH)
required_branch = {"id", "well", "row_idx", "fold", "true_tvt", "crossfit_blend"}
if not required_branch.issubset(branch.columns):
    raise RuntimeError(f"Branch artifact missing {sorted(required_branch - set(branch.columns))}")
# The branch sidecar also carries diagnostic copies of SP45/PSG. Drop them
# before merging so every historical endpoint has one unambiguous pool source.
branch = branch[["id", "well", "row_idx", "fold", "true_tvt", "crossfit_blend"]].copy()
branch["id"] = branch["id"].astype(str).str.lower()
if branch["id"].duplicated().any():
    raise RuntimeError("Branch artifact has duplicate ids")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
base_columns = [name for name in ("sp45", "fleongg", "strong", "strong_bias", "cp12") if name in header]
if "id" not in header or "tvt_true" not in header or "cp12" not in base_columns:
    raise RuntimeError(f"Pool lacks id/tvt_true/cp12: {header}")
usecols = [name for name in ("id", "well", "row_idx", "tvt_true", *base_columns) if name in header]
pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": str, "well": str})
pool["id"] = pool["id"].astype(str).str.lower()
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str).str.lower()
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)

neighbor_available = NEIGHBOR_CANDIDATE_PATH.exists() and NEIGHBOR_META_PATH.exists()
neighbor_note = "not available"
if neighbor_available:
    payload = np.load(NEIGHBOR_CANDIDATE_PATH, allow_pickle=False)
    candidate = payload["candidate"].astype(np.float64)
    if len(candidate) != len(pool):
        raise RuntimeError(f"Neighbor candidate rows {len(candidate)} != pool rows {len(pool)}")
    meta = pd.read_csv(NEIGHBOR_META_PATH)
    meta["well"] = meta["well"].astype(str).str.lower()
    meta = meta.set_index("well")
    eligible_well = (
        (meta["median_distance"] <= 600.0)
        & (meta["p90_distance"] <= 1200.0)
        & (meta["anchor_distance_full"] <= 1200.0)
        & (meta["abs_direction_dot"] >= MIN_ABS_DIRECTION_DOT)
    )
    eligible = pool["well"].map(eligible_well.to_dict()).fillna(False).to_numpy(bool)
    cp12 = pool["cp12"].to_numpy(np.float64)
    move = np.zeros(len(pool), np.float64)
    valid = eligible & np.isfinite(candidate)
    move[valid] = SAFE_WEIGHT * np.clip(candidate[valid] - cp12[valid], -RAW_MOVE_CAP, RAW_MOVE_CAP)
    pool["cp12_neighbor_full_progress_geom"] = cp12 + move
    base_columns.append("cp12_neighbor_full_progress_geom")
    neighbor_note = (
        "Reconstructed fold-clean CP12 + frozen full_progress_geom move; "
        "does not include a claimed exact Gold/LB stochastic realization"
    )

merge_columns = ["id", "tvt_true", *base_columns]
data = branch.merge(pool[merge_columns], on="id", how="left", validate="one_to_one", indicator=True)
coverage = float(data["_merge"].eq("both").mean())
if coverage < 0.995:
    raise RuntimeError(f"Branch/pool coverage {coverage:.6%}")
data = data[data["_merge"].eq("both")].drop(columns=["_merge"]).reset_index(drop=True)
data = data.rename(columns={"true_tvt": "branch_truth", "tvt_true": "truth"})
truth_gap = float(np.max(np.abs(data["branch_truth"] - data["truth"])))
if truth_gap > 0.02:
    raise RuntimeError(f"Truth mismatch {truth_gap:.6f} ft")
data["psg_sp45_branch"] = data["crossfit_blend"].astype(np.float64)

summary_rows = []
all_folds = []
predictions = {}
weights = {}
for base_name in base_columns:
    required = data[["truth", base_name, "psg_sp45_branch"]].to_numpy(np.float64)
    if not np.isfinite(required).all():
        continue
    summary, fold_report, prediction, weight = _audit_pair(
        data, base_name, "psg_sp45_branch"
    )
    summary_rows.append(summary)
    all_folds.append(fold_report)
    predictions[base_name] = prediction
    weights[base_name] = weight

summary = pd.DataFrame(summary_rows).sort_values("base_rmse", ignore_index=True)
fold_report = pd.concat(all_folds, ignore_index=True)
summary.to_csv(OUT_DIR / "branch_survival_summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "branch_survival_folds.csv", index=False)

primary_name = "cp12_neighbor_full_progress_geom"
if neighbor_available and primary_name in set(summary["base"]):
    primary = summary[summary["base"].eq(primary_name)].iloc[0]
    checks = {
        "each_weight_ge_0p02": float(primary["minimum_weight"]) >= MIN_WEIGHT,
        "crossfit_gain_ge_0p05": float(primary["crossfit_gain"]) >= MIN_GAIN_FT,
        "minimum_fold_gain_ge_minus0p10": float(primary["minimum_fold_gain"]) >= MIN_FOLD_GAIN_FT,
        "all_folds_positive": int(primary["positive_folds"]) == 5,
    }
    strict_pass = bool(all(checks.values()))
    export = data[["id", "well", "row_idx", "fold", "truth", primary_name, "psg_sp45_branch"]].copy()
    export["meta_weight"] = weights[primary_name]
    export["crossfit_blend"] = predictions[primary_name].astype(np.float32)
    export.to_parquet(OUT_DIR / "neighbor_branch_crossfit_oof.parquet", index=False)
else:
    primary = None
    checks = {"neighbor_artifact_available": False}
    strict_pass = False

verdict = {
    "experiment": EXPERIMENT,
    "stage": "psg_sp45_branch_survival_across_stronger_oof_endpoints",
    "inputs": {
        "branch": str(BRANCH_PATH),
        "pool": str(POOL_PATH),
        "neighbor_candidate": str(NEIGHBOR_CANDIDATE_PATH),
        "neighbor_meta": str(NEIGHBOR_META_PATH),
    },
    "alignment": {"coverage": coverage, "rows": len(data), "wells": data["well"].nunique(), "truth_gap": truth_gap},
    "neighbor_available": neighbor_available,
    "neighbor_interpretation": neighbor_note,
    "primary": None if primary is None else primary.to_dict(),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> branch survives the frozen neighbor OOF; only then consider an exact deployment-lineage rebuild"
        if strict_pass
        else (
            "MISSING -> run neighbor_fullcurve_projection_direction_v1_cell.py, then rerun this audit"
            if not neighbor_available
            else "FAIL -> PSG-SP45 complementarity is consumed by the stronger mainline; do not submit a final blend"
        )
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nPSG-SP45 BRANCH SURVIVAL")
print(summary.to_string(index=False), flush=True)
print("\nSURVIVAL FOLDS")
print(fold_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
