"""M342-V1: recover V74 OOF predictions and test complementarity with K2.

V74 did not export v74_oof.parquet because its original attribution gate
failed, despite positive five-fold gains versus V66. This wrapper executes V74
in an isolated namespace (reusing its feature cache), then immediately tests a
small fixed blend with the stronger frozen K2 endpoint. The station-shuffled
V74 arm remains an attribution control.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
V74_SOURCE = ROOT / "longcontext_v66_residual_gbdt_v74_cell.py"
P1_SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_longcontext_k2_compatibility_v1"
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_EVALUATION_FOLD = 1
WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
MAX_P95_CHANGE = 0.20


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _execute_v74():
    if not V74_SOURCE.is_file():
        raise FileNotFoundError(V74_SOURCE)
    name = "m342_v1_v74_runtime"
    module = types.ModuleType(name)
    module.__file__ = str(V74_SOURCE)
    sys.modules[name] = module
    try:
        source = V74_SOURCE.read_text(encoding="utf-8")
        exec(compile(source, str(V74_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    required = {"frame", "endpoint_predictions"}
    missing = sorted(required.difference(module.__dict__))
    if missing:
        raise RuntimeError("V74 runtime outputs missing: " + repr(missing))
    return module.__dict__


def _load_p1_definitions():
    source = P1_SOURCE.read_text(encoding="utf-8")
    marker = 'base, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError("P1 execution marker not found")
    name = "m342_v1_p1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(P1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(P1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


V74 = _execute_v74()
P = _load_p1_definitions()
B = P["B"]
base, _, _, coverage = P["O"]["_prepare_inventory"]()
k2 = P["_endpoint_table"]()[["id", "k2_parent"]]

v74_frame = V74["frame"][["id", "well", "fold", "tvt_true", "md_since"]].copy()
v74_frame["id"] = v74_frame["id"].astype(str)
predictions = V74["endpoint_predictions"]
for key in ("aligned_longcontext", "station_shuffled_longcontext", "geometry_only"):
    if key not in predictions:
        raise RuntimeError(f"V74 endpoint missing: {key}")
    v74_frame[key] = np.asarray(predictions[key], np.float64)
frame = v74_frame.merge(k2, on="id", how="inner", validate="one_to_one")
if len(frame) != len(base) or frame["well"].nunique() != 773:
    raise RuntimeError(f"V1 alignment incomplete: rows={len(frame)}, wells={frame['well'].nunique()}")
if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
    raise RuntimeError("Non-finite V1 values")

folds = frame["fold"].to_numpy(np.int8)
parent = frame["k2_parent"].to_numpy(np.float64)
aligned = frame["aligned_longcontext"].to_numpy(np.float64)
shuffled = frame["station_shuffled_longcontext"].to_numpy(np.float64)
geometry = frame["geometry_only"].to_numpy(np.float64)


def _metric(mask, prediction, endpoint):
    local = frame.loc[mask].reset_index(drop=True)
    return B["_metrics"](local, np.asarray(prediction, np.float64)[mask], endpoint)


def _fold_result(fold, candidate, control):
    mask = folds == fold
    base_metric = _metric(mask, parent, "k2_parent")
    metric = _metric(mask, candidate, "aligned_longcontext_blend")
    control_metric = _metric(mask, control, "station_shuffle_blend")
    return {
        "fold": fold,
        "gain": base_metric["rmse"] - metric["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
        "aligned_increment_vs_shuffle": control_metric["rmse"] - metric["rmse"],
    }


rows = []
for weight in WEIGHTS:
    candidate = parent + float(weight) * (aligned - parent)
    control = parent + float(weight) * (shuffled - parent)
    reports = [_fold_result(fold, candidate, control) for fold in DISCOVERY_FOLDS]
    pooled = np.isin(folds, DISCOVERY_FOLDS)
    base_metric = _metric(pooled, parent, "k2_parent")
    metric = _metric(pooled, candidate, "aligned_longcontext_blend")
    control_metric = _metric(pooled, control, "station_shuffle_blend")
    rows.append({
        "weight": float(weight),
        "pooled_rmse": metric["rmse"],
        "pooled_gain": base_metric["rmse"] - metric["rmse"],
        "minimum_fold_gain": min(item["gain"] for item in reports),
        "minimum_fold_far_gain": min(item["far_gain"] for item in reports),
        "maximum_fold_p95_change": max(item["p95_change"] for item in reports),
        "aligned_increment_vs_shuffle": control_metric["rmse"] - metric["rmse"],
        "fold4_gain": reports[0]["gain"],
        "fold5_gain": reports[1]["gain"],
    })

scan = pd.DataFrame(rows).sort_values("pooled_rmse", ignore_index=True)
safe = scan.loc[
    scan["minimum_fold_gain"].ge(0.005)
    & scan["minimum_fold_far_gain"].ge(0.0)
    & scan["maximum_fold_p95_change"].le(MAX_P95_CHANGE)
    & scan["aligned_increment_vs_shuffle"].ge(0.005)
].copy()
selected = (safe.iloc[0] if len(safe) else scan.iloc[0]).to_dict()
weight = float(selected["weight"])
candidate = parent + weight * (aligned - parent)
control = parent + weight * (shuffled - parent)
confirmation = _fold_result(CONFIRMATION_FOLD, candidate, control)
mechanism_pass = bool(
    len(safe) > 0
    and confirmation["gain"] >= 0.01
    and confirmation["far_gain"] >= 0.0
    and confirmation["p95_change"] <= MAX_P95_CHANGE
    and confirmation["aligned_increment_vs_shuffle"] >= 0.005
)
application = _fold_result(APPLICATION_FOLD, candidate, control) if mechanism_pass else None

error = frame["tvt_true"].to_numpy(np.float64)
error_rho = float(np.corrcoef(parent - error, aligned - error)[0, 1])
geometry_rho = float(np.corrcoef(parent - error, geometry - error)[0, 1])
checks = {
    "discovery_has_safe_weight": len(safe) > 0,
    "fold3_gain_ge_0p01": confirmation["gain"] >= 0.01,
    "fold3_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
    "fold3_aligned_increment_ge_0p005": confirmation["aligned_increment_vs_shuffle"] >= 0.005,
    "fold3_p95_change_le_0p20": confirmation["p95_change"] <= MAX_P95_CHANGE,
}
if application is not None:
    checks.update({
        "fold2_gain_ge_0p01": application["gain"] >= 0.01,
        "fold2_far_gain_ge_0": application["far_gain"] >= 0.0,
        "fold2_aligned_increment_ge_0p005": application["aligned_increment_vs_shuffle"] >= 0.005,
        "fold2_p95_change_le_0p20": application["p95_change"] <= MAX_P95_CHANGE,
    })
strict_pass = bool(application is not None and all(checks.values()))

verdict = {
    "experiment": "main342_longcontext_k2_compatibility_v1",
    "stage": "v74_recovery_fold45_blend_selection_fold3_fold2_confirmation",
    "selection_or_tuning_count": 1,
    "sources": {"v74": V74_SOURCE, "k2": P1_SOURCE},
    "protocol": {
        "discovery_folds": DISCOVERY_FOLDS,
        "confirmation_fold": CONFIRMATION_FOLD,
        "application_fold": APPLICATION_FOLD if mechanism_pass else None,
        "untouched_evaluation_fold": UNTOUCHED_EVALUATION_FOLD,
    },
    "coverage": coverage,
    "error_correlations": {"k2_vs_v74": error_rho, "k2_vs_geometry_control": geometry_rho},
    "selected": selected,
    "confirmation": confirmation,
    "application": application,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build a deployment-equivalent long-context endpoint before any Kaggle submission"
        if strict_pass else
        "FAIL -> V74 adds no attributable residual information after K2; close long-context reuse"
    ),
}

scan.to_csv(OUT / "blend_scan.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-V1 BLEND SCAN")
print(scan.to_string(index=False), flush=True)
print("\nM342-V1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
