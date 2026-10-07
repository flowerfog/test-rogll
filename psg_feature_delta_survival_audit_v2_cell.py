"""Zero-training nested survival audit for the PSG feature correction.

The absolute PSG-feature model is weak. This audit isolates its incremental
correction relative to its identical baseline and tests whether that correction
survives when transplanted onto strong/CP12/neighbor OOF paths.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


EXPERIMENT = "psg_feature_delta_survival_audit_v2"
ROOT = Path("/root/autodl-tmp/oof_exports")
PSG_MODEL_PATH = (
    ROOT / "psg_foldclean_fullstack_lgb_gateway_v1" / "psg_feature_lgb_oof.parquet"
)
PSG_RAW_PATH = ROOT / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet"
POOL_CANDIDATES = [
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ROOT / "candidate_pool_full_merged_oof.csv",
]
NEIGHBOR_DIR = ROOT / "neighbor_fullcurve_transfer_v0"
OUT_DIR = ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOSES = np.array([0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0])
NEIGHBOR_WEIGHT = 0.25
NEIGHBOR_CAP_FT = 24.0
MIN_ABS_DIRECTION_DOT = 0.70
started = time.time()


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


def _rmse(truth, prediction):
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(error**2)))


def _detrend_error(error, distance):
    error = np.asarray(error, np.float64)
    distance = np.asarray(distance, np.float64)
    knots = np.array([0.0, 500.0, 1000.0, 2000.0, np.inf])
    output = error.copy()
    for lower, upper in zip(knots[:-1], knots[1:]):
        selected = (distance >= lower) & (distance < upper)
        if selected.any():
            output[selected] -= float(np.mean(error[selected]))
    return output


def _nested_dose(base, correction, truth, folds):
    prediction = np.empty(len(truth), np.float64)
    selected_rows = []
    for held_fold in sorted(np.unique(folds)):
        train = folds != held_fold
        valid = folds == held_fold
        scan = []
        for dose in DOSES:
            scan.append((float(dose), _rmse(truth[train], base[train] + dose * correction[train])))
        dose, selection_rmse = min(scan, key=lambda item: (item[1], item[0]))
        prediction[valid] = base[valid] + dose * correction[valid]
        selected_rows.append(
            {
                "held_fold": int(held_fold),
                "selected_dose": float(dose),
                "selection_rmse": float(selection_rmse),
                "base_rmse": _rmse(truth[valid], base[valid]),
                "candidate_rmse": _rmse(truth[valid], prediction[valid]),
            }
        )
        selected_rows[-1]["gain"] = (
            selected_rows[-1]["base_rmse"] - selected_rows[-1]["candidate_rmse"]
        )
    return prediction, pd.DataFrame(selected_rows)


for required in (PSG_MODEL_PATH, PSG_RAW_PATH):
    if not required.exists():
        raise FileNotFoundError(required)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError(f"candidate pool not found: {POOL_CANDIDATES}")

psg = pd.read_parquet(PSG_MODEL_PATH)
required_psg = {"id", "well", "row_idx", "fold", "tvt_true", "baseline", "psg_features"}
missing = sorted(required_psg.difference(psg.columns))
if missing:
    raise RuntimeError(f"PSG model OOF missing columns: {missing}")
psg = psg[list(required_psg)].copy()
psg["id"] = psg["id"].astype(str).str.lower()
psg["well"] = psg["well"].astype(str).str.lower()
psg["row_idx"] = psg["row_idx"].astype(np.int32)
if psg.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("PSG model OOF contains duplicate keys")

raw = pd.read_parquet(PSG_RAW_PATH, columns=["well", "row_idx", "md_since"])
raw["well"] = raw["well"].astype(str).str.lower()
raw["row_idx"] = raw["row_idx"].astype(np.int32)
psg = psg.merge(raw, on=["well", "row_idx"], how="left", validate="one_to_one")
if not np.isfinite(psg["md_since"]).all():
    raise RuntimeError("PSG raw/model OOF distance alignment failed")

pool_header = pd.read_csv(pool_path, nrows=0).columns.tolist()
base_columns = [
    column for column in ("strong_bias", "cp12", "strong", "fleongg", "sp45")
    if column in pool_header
]
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", *base_columns) if column in pool_header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
if "well" not in pool:
    pool["well"] = pool["id"].astype(str).str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["id"] = pool["id"].astype(str).str.lower()
pool["well"] = pool["well"].astype(str).str.lower()
pool["row_idx"] = pool["row_idx"].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
if pool.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("candidate pool contains duplicate keys")

# Reconstruct the frozen v0 neighbor move in the exact candidate-pool order.
sidecar_path = NEIGHBOR_DIR / "neighbor_transfer_oof.npz"
meta_path = NEIGHBOR_DIR / "selected_neighbors.csv"
if sidecar_path.exists() and meta_path.exists() and "strong" in pool:
    sidecar = np.load(sidecar_path, allow_pickle=False)
    neighbor_candidate = sidecar["structure_progress"].astype(np.float64)
    if len(neighbor_candidate) != len(pool):
        raise RuntimeError("neighbor sidecar/candidate-pool row mismatch")
    neighbor_meta = pd.read_csv(meta_path)
    neighbor_meta["well"] = neighbor_meta["well"].astype(str).str.lower()
    neighbor_meta = neighbor_meta.drop_duplicates("well").set_index("well")
    median_distance = pool["well"].map(neighbor_meta["median_distance"]).to_numpy(float)
    p90_distance = pool["well"].map(neighbor_meta["p90_distance"]).to_numpy(float)
    anchor_distance = pool["well"].map(neighbor_meta["anchor_distance_full"]).to_numpy(float)
    direction = pool["well"].map(neighbor_meta["abs_direction_dot"]).to_numpy(float)
    eligible = (
        (median_distance <= 600.0)
        & (p90_distance <= 1200.0)
        & (anchor_distance <= 1200.0)
        & (direction >= MIN_ABS_DIRECTION_DOT)
        & np.isfinite(neighbor_candidate)
    )
    strong = pool["strong"].to_numpy(np.float64)
    move = np.zeros(len(pool), np.float64)
    move[eligible] = NEIGHBOR_WEIGHT * np.clip(
        neighbor_candidate[eligible] - strong[eligible], -NEIGHBOR_CAP_FT, NEIGHBOR_CAP_FT
    )
    for source in ("strong", "strong_bias", "cp12"):
        if source in pool:
            pool[f"{source}_neighbor"] = pool[source].to_numpy(np.float64) + move
            base_columns.append(f"{source}_neighbor")

pool_columns = ["well", "row_idx", *base_columns]
data = psg.merge(pool[pool_columns], on=["well", "row_idx"], how="inner", validate="one_to_one")
coverage = len(data) / len(psg)
if coverage < 0.995:
    raise RuntimeError(f"PSG/candidate pool alignment only {coverage:.6%}")
numeric = ["tvt_true", "baseline", "psg_features", "md_since", *base_columns]
if not np.isfinite(data[numeric].to_numpy(np.float64)).all():
    raise RuntimeError("aligned audit data contains non-finite values")

truth = data["tvt_true"].to_numpy(np.float64)
folds = data["fold"].to_numpy(np.int8)
wells = data["well"].to_numpy()
distance = data["md_since"].to_numpy(np.float64)
feature_baseline = data["baseline"].to_numpy(np.float64)
feature_candidate = data["psg_features"].to_numpy(np.float64)
feature_correction = feature_candidate - feature_baseline

summary_rows = []
fold_parts = []
prediction_parts = {"well": data["well"], "row_idx": data["row_idx"], "fold": folds, "truth": truth}
for base_name in base_columns:
    base = data[base_name].to_numpy(np.float64)
    candidate, fold_report = _nested_dose(base, feature_correction, truth, folds)
    base_metrics = _metrics(truth, base, wells)
    candidate_metrics = _metrics(truth, candidate, wells)
    raw_rho = float(np.corrcoef(base - truth, feature_candidate - truth)[0, 1])
    detrended_rho = float(
        np.corrcoef(
            _detrend_error(base - truth, distance),
            _detrend_error(feature_candidate - truth, distance),
        )[0, 1]
    )
    summary_rows.append(
        {
            "base": base_name,
            "rows": len(data),
            "wells": len(np.unique(wells)),
            "base_rmse": base_metrics["rmse"],
            "candidate_rmse": candidate_metrics["rmse"],
            "gain": base_metrics["rmse"] - candidate_metrics["rmse"],
            "raw_error_rho_vs_psg_model": raw_rho,
            "detrended_error_rho_vs_psg_model": detrended_rho,
            "median_selected_dose": float(fold_report["selected_dose"].median()),
            "minimum_fold_gain": float(fold_report["gain"].min()),
            "positive_folds": int((fold_report["gain"] > 0.0).sum()),
        }
    )
    fold_report.insert(0, "base", base_name)
    fold_parts.append(fold_report)
    prediction_parts[f"{base_name}_psg_delta"] = candidate

summary = pd.DataFrame(summary_rows).sort_values("gain", ascending=False, ignore_index=True)
fold_report = pd.concat(fold_parts, ignore_index=True)
predictions = pd.DataFrame(prediction_parts)
best = summary.iloc[0].to_dict()
checks = {
    "best_nested_gain_ge_0p05": float(best["gain"]) >= 0.05,
    "best_positive_folds_ge_4": int(best["positive_folds"]) >= 4,
    "best_minimum_fold_gain_ge_minus0p05": float(best["minimum_fold_gain"]) >= -0.05,
    "best_median_dose_gt_zero": float(best["median_selected_dose"]) > 0.0,
}
strict_pass = bool(all(checks.values()))

summary.to_csv(OUT_DIR / "summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
predictions.to_parquet(OUT_DIR / "crossfit_predictions.parquet", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "zero_training_nested_psg_feature_delta_survival",
    "sample_unit": "complete physical well; row metric pooled",
    "psg_model_source": str(PSG_MODEL_PATH),
    "candidate_pool_source": str(pool_path),
    "aligned_rows": len(data),
    "aligned_wells": int(data["well"].nunique()),
    "alignment_coverage": coverage,
    "dose_grid": DOSES.tolist(),
    "best": best,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> carry PSG features into the exact full production refit"
        if strict_pass
        else "FAIL -> PSG helps its own weak model but its correction is consumed by strong downstream paths"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nPSG FEATURE DELTA SURVIVAL")
print(summary.to_string(index=False), flush=True)
print("\nNESTED FOLD DOSES")
print(fold_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
