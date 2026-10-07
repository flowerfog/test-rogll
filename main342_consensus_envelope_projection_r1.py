"""M342-R1: truth-free geometric projection of the frozen K2 correction.

K2 already estimates a correction from the endpoint bank. R1 does not learn
another residual direction. It only projects that correction into geometric
support supplied by the independent coarse, PSG and Neighbor deltas.

Folds 4/5 select one fixed projection and dose. Fold 3 is confirmation. Fold 2
is evaluated only after Fold 3 passes; Fold 1 remains untouched.
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
P1_SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_consensus_envelope_projection_r1"
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DOSES = (0.25, 0.50, 0.75, 1.00)
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


def _load_p1_definitions():
    if not P1_SOURCE.is_file():
        raise FileNotFoundError(P1_SOURCE)
    source = P1_SOURCE.read_text(encoding="utf-8")
    marker = 'base, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError("P1 execution marker not found")
    name = "m342_r1_p1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(P1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(P1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


P = _load_p1_definitions()
O = P["O"]
B = P["B"]
base, _, _, coverage = O["_prepare_inventory"]()
endpoint = P["_endpoint_table"]()
if endpoint["id"].duplicated().any() or base["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in R1 inputs")
frame = base.merge(endpoint, on="id", how="inner", validate="one_to_one")
if len(frame) != len(base) or frame["well"].nunique() != 773:
    raise RuntimeError(f"R1 alignment incomplete: rows={len(frame)}, wells={frame['well'].nunique()}")

parent = frame["k2_parent"].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
current = frame["k2_move"].to_numpy(np.float64)
deltas = frame[["coarse_delta", "psg_delta", "neighbor_delta"]].to_numpy(np.float64).T
if not np.isfinite(np.vstack([parent, truth, current, deltas])).all():
    raise RuntimeError("Non-finite R1 input")


def _projected_corrections():
    lower = np.minimum(0.0, np.min(deltas, axis=0))
    upper = np.maximum(0.0, np.max(deltas, axis=0))
    hull = np.clip(current, lower, upper)
    magnitude = np.abs(deltas)
    median_cap = np.sign(current) * np.minimum(np.abs(current), np.median(magnitude, axis=0))
    q75_cap = np.sign(current) * np.minimum(np.abs(current), np.quantile(magnitude, 0.75, axis=0))
    same_sign = np.mean((deltas * current[None, :]) > 0.0, axis=0)
    majority = (same_sign >= (2.0 / 3.0)).astype(np.float64)
    unanimous = (same_sign >= 1.0).astype(np.float64)
    support_soft = 0.25 + 0.75 * same_sign
    return {
        "hull_clip": hull,
        "median_magnitude_cap": median_cap,
        "q75_magnitude_cap": q75_cap,
        "majority_support": current * majority,
        "unanimous_support": current * unanimous,
        "soft_support": current * support_soft,
        "hull_soft_support": hull * support_soft,
    }


projected = _projected_corrections()
fold_array = frame["fold"].to_numpy(np.int8)


def _metrics(mask, prediction, label):
    local = frame.loc[mask].reset_index(drop=True)
    return B["_metrics"](local, np.asarray(prediction, np.float64)[mask], label)


base_metrics = {
    fold: _metrics(fold_array == fold, parent, "k2_parent")
    for fold in range(1, 6)
}
rows = []
for formula, correction in projected.items():
    increment = correction - current
    for dose in DOSES:
        prediction = parent + float(dose) * increment
        fold_values = []
        for fold in DISCOVERY_FOLDS:
            metric = _metrics(fold_array == fold, prediction, formula)
            baseline = base_metrics[fold]
            fold_values.append({
                "fold": fold,
                "gain": baseline["rmse"] - metric["rmse"],
                "far_gain": baseline["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                "p95_change": metric["p95_well_rmse"] - baseline["p95_well_rmse"],
            })
        pooled_mask = np.isin(fold_array, DISCOVERY_FOLDS)
        pooled_metric = _metrics(pooled_mask, prediction, formula)
        pooled_base = _metrics(pooled_mask, parent, "k2_parent")
        rows.append({
            "formula": formula,
            "dose": float(dose),
            "pooled_rmse": pooled_metric["rmse"],
            "pooled_gain": pooled_base["rmse"] - pooled_metric["rmse"],
            "minimum_fold_gain": min(item["gain"] for item in fold_values),
            "minimum_fold_far_gain": min(item["far_gain"] for item in fold_values),
            "maximum_fold_p95_change": max(item["p95_change"] for item in fold_values),
            "fold4_gain": fold_values[0]["gain"],
            "fold5_gain": fold_values[1]["gain"],
        })

scan = pd.DataFrame(rows).sort_values(
    ["pooled_rmse", "minimum_fold_gain"], ascending=[True, False], ignore_index=True
)
safe = scan.loc[
    scan["minimum_fold_gain"].ge(0.005)
    & scan["minimum_fold_far_gain"].ge(0.0)
    & scan["maximum_fold_p95_change"].le(MAX_P95_CHANGE)
].copy()
selected = (safe.iloc[0] if len(safe) else scan.iloc[0]).to_dict()
formula = str(selected["formula"])
dose = float(selected["dose"])
prediction = parent + dose * (projected[formula] - current)

confirmation_mask = fold_array == CONFIRMATION_FOLD
confirmation_base = base_metrics[CONFIRMATION_FOLD]
confirmation_candidate = _metrics(confirmation_mask, prediction, formula)
confirmation = {
    "fold": CONFIRMATION_FOLD,
    "base": confirmation_base,
    "candidate": confirmation_candidate,
    "gain": confirmation_base["rmse"] - confirmation_candidate["rmse"],
    "far_gain": confirmation_base["far_gt2000_rmse"] - confirmation_candidate["far_gt2000_rmse"],
    "p95_change": confirmation_candidate["p95_well_rmse"] - confirmation_base["p95_well_rmse"],
}
mechanism_pass = bool(
    len(safe) > 0
    and confirmation["gain"] >= 0.01
    and confirmation["far_gain"] >= 0.0
    and confirmation["p95_change"] <= MAX_P95_CHANGE
)

application = None
if mechanism_pass:
    application_mask = fold_array == APPLICATION_FOLD
    application_base = base_metrics[APPLICATION_FOLD]
    application_candidate = _metrics(application_mask, prediction, formula)
    application = {
        "fold": APPLICATION_FOLD,
        "base": application_base,
        "candidate": application_candidate,
        "gain": application_base["rmse"] - application_candidate["rmse"],
        "far_gain": application_base["far_gt2000_rmse"] - application_candidate["far_gt2000_rmse"],
        "p95_change": application_candidate["p95_well_rmse"] - application_base["p95_well_rmse"],
    }

checks = {
    "discovery_has_safe_projection": len(safe) > 0,
    "fold3_gain_ge_0p01": confirmation["gain"] >= 0.01,
    "fold3_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
    "fold3_p95_change_le_0p20": confirmation["p95_change"] <= MAX_P95_CHANGE,
}
if application is not None:
    checks.update({
        "fold2_gain_ge_0p01": application["gain"] >= 0.01,
        "fold2_far_gain_ge_0": application["far_gain"] >= 0.0,
        "fold2_p95_change_le_0p20": application["p95_change"] <= MAX_P95_CHANGE,
    })
strict_pass = bool(application is not None and all(checks.values()))

verdict = {
    "experiment": "main342_consensus_envelope_projection_r1",
    "stage": "fold45_geometric_projection_selection_fold3_fold2_confirmation",
    "selection_or_tuning_count": 1,
    "legal_inputs_only": True,
    "mechanism": "truth-free projection of the frozen K2 correction into independent endpoint support",
    "protocol": {
        "discovery_folds": DISCOVERY_FOLDS,
        "confirmation_fold": CONFIRMATION_FOLD,
        "application_fold": APPLICATION_FOLD if mechanism_pass else None,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "coverage": coverage,
    "formulas": list(projected),
    "selected": selected,
    "confirmation": confirmation,
    "application": application,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze projection and run one zero-retuning Fold1 audit"
        if strict_pass else
        "FAIL -> K2 correction already uses the transferable endpoint geometry; close R1"
    ),
}

scan.to_csv(OUT / "discovery_scan.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-R1 DISCOVERY")
print(scan.head(20).to_string(index=False), flush=True)
print("\nM342-R1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
