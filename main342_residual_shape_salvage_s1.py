"""M342-S1: salvage test for Q1 residual shape after removing datum/trend.

Q1 improved Fold2 MAE while worsening pooled RMSE and p95. S1 tests whether
the learned local correction shape transfers after removing unobservable
per-well datum or long linear trend. It reuses the frozen Q1 models and does
not train another model.
"""

from pathlib import Path
import json
import sys
import time
import types

import joblib
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
Q1_SOURCE = ROOT / "main342_multicut_invariant_residual_q1.py"
Q1_DIR = ROOT / "oof_exports" / "main342_multicut_invariant_residual_q1"
OUT = ROOT / "oof_exports" / "main342_residual_shape_salvage_s1"
OUT.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
DOSES = (0.0, 0.25, 0.50, 0.75, 1.00)
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


def _load_q1_definitions():
    if not Q1_SOURCE.is_file():
        raise FileNotFoundError(Q1_SOURCE)
    source = Q1_SOURCE.read_text(encoding="utf-8")
    marker = 'base, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError("Q1 execution marker not found")
    name = "m342_s1_q1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(Q1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(Q1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


Q = _load_q1_definitions()
P = Q["P"]
B = Q["B"]
O = Q["O"]
base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()
features = list(dict.fromkeys(base_features + list(P["ENGINEERED"])))
official_manifest = manifest[manifest["tag"].eq("official")].copy()
official_manifest["cut_bin"] = 0


def _held_frame(fold):
    wells = sorted(base.loc[base["fold"].eq(fold), "well"].unique())
    frame = B["_load_full"](wells, official_manifest, base)
    frame = frame.merge(endpoints, on="id", how="inner", validate="one_to_one")
    if frame["well"].nunique() != len(wells):
        raise RuntimeError(f"Fold {fold} alignment incomplete")
    return frame.reset_index(drop=True)


def _profiles(frame, correction):
    correction = np.asarray(correction, np.float64)
    output = {
        "original": correction.copy(),
        "mean_removed": np.empty_like(correction),
        "median_removed": np.empty_like(correction),
        "slope_removed_mean_preserved": np.empty_like(correction),
        "datum_and_slope_removed": np.empty_like(correction),
        "prefix_reanchored": np.empty_like(correction),
    }
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        local = correction[positions]
        n = len(local)
        x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        design = np.column_stack([np.ones(n), x])
        coefficient, *_ = np.linalg.lstsq(design, local, rcond=None)
        trend = design @ coefficient
        edge = max(1, min(64, n // 8))
        prefix_level = float(np.median(local[:edge]))
        output["mean_removed"][positions] = local - float(np.mean(local))
        output["median_removed"][positions] = local - float(np.median(local))
        output["slope_removed_mean_preserved"][positions] = local - coefficient[1] * x
        output["datum_and_slope_removed"][positions] = local - trend
        output["prefix_reanchored"][positions] = local - prefix_level
    return output


def _metric(frame, prediction, endpoint):
    return B["_metrics"](frame, np.asarray(prediction, np.float64), endpoint)


selection_model_path = Q1_DIR / "models" / "selection_multicut_residual.joblib"
confirmation_model_path = Q1_DIR / "models" / "confirmation_multicut_residual.joblib"
for path in (selection_model_path, confirmation_model_path):
    if not path.is_file():
        raise FileNotFoundError(path)

selection_frame = _held_frame(SELECTION_FOLD)
selection_model = joblib.load(selection_model_path)
selection_raw = P["_correction"](selection_frame, selection_model, features)
selection_profiles = _profiles(selection_frame, selection_raw)
selection_parent = selection_frame["k2_parent"].to_numpy(np.float64)
selection_base = _metric(selection_frame, selection_parent, "k2_parent")

scan_rows = []
for profile, correction in selection_profiles.items():
    rolled = np.empty_like(correction)
    for positions in selection_frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        rolled[positions] = np.roll(correction[positions], max(1, len(positions) // 4))
    for dose in DOSES:
        prediction = selection_parent + float(dose) * correction
        roll_prediction = selection_parent + float(dose) * rolled
        metric = _metric(selection_frame, prediction, profile)
        roll_metric = _metric(selection_frame, roll_prediction, "depth_roll_control")
        scan_rows.append({
            "profile": profile,
            "dose": float(dose),
            "rmse": metric["rmse"],
            "gain": selection_base["rmse"] - metric["rmse"],
            "far_gain": selection_base["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - selection_base["p95_well_rmse"],
            "roll_loss": roll_metric["rmse"] - metric["rmse"],
        })

scan = pd.DataFrame(scan_rows).sort_values("rmse", ignore_index=True)
safe = scan.loc[
    scan["dose"].gt(0.0)
    & scan["gain"].ge(0.03)
    & scan["far_gain"].ge(0.0)
    & scan["p95_change"].le(MAX_P95_CHANGE)
    & scan["roll_loss"].ge(0.015)
].copy()
selected = (safe.iloc[0] if len(safe) else scan.iloc[0]).to_dict()
mechanism_pass = bool(len(safe) > 0)

confirmation = None
checks = {"selection_has_safe_shape_profile": mechanism_pass}
if mechanism_pass:
    confirmation_frame = _held_frame(CONFIRMATION_FOLD)
    confirmation_model = joblib.load(confirmation_model_path)
    confirmation_raw = P["_correction"](confirmation_frame, confirmation_model, features)
    correction = _profiles(confirmation_frame, confirmation_raw)[str(selected["profile"])]
    dose = float(selected["dose"])
    parent = confirmation_frame["k2_parent"].to_numpy(np.float64)
    candidate = parent + dose * correction
    reverse = parent - dose * correction
    base_metric = _metric(confirmation_frame, parent, "k2_parent")
    candidate_metric = _metric(confirmation_frame, candidate, "shape_residual")
    reverse_metric = _metric(confirmation_frame, reverse, "negative_direction_control")
    confirmation = {
        "base": base_metric,
        "candidate": candidate_metric,
        "reverse": reverse_metric,
        "gain": base_metric["rmse"] - candidate_metric["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"],
        "reverse_loss": reverse_metric["rmse"] - candidate_metric["rmse"],
        "p95_change": candidate_metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
    }
    checks.update({
        "confirmation_gain_ge_0p02": confirmation["gain"] >= 0.02,
        "confirmation_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
        "reverse_loses_ge_0p03": confirmation["reverse_loss"] >= 0.03,
        "confirmation_p95_change_le_0p20": confirmation["p95_change"] <= MAX_P95_CHANGE,
    })

strict_pass = bool(confirmation is not None and all(checks.values()))
verdict = {
    "experiment": "main342_residual_shape_salvage_s1",
    "stage": "frozen_q1_correction_shape_attribution_fold3_fold2",
    "selection_or_tuning_count": 1,
    "training_count": 0,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD if mechanism_pass else None,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "coverage": coverage,
    "profiles": list(selection_profiles),
    "selected": selected,
    "confirmation": confirmation,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild a full-data multi-cut residual bundle with the frozen shape transform"
        if strict_pass else
        "FAIL -> Q1 local shape also does not transfer; close supervised K2-residual modeling"
    ),
}

scan.to_csv(OUT / "selection_scan.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-S1 SELECTION SCAN")
print(scan.head(20).to_string(index=False), flush=True)
print("\nM342-S1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
