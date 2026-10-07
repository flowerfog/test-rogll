"""M342-T1: fold-jackknife stability aggregation of the frozen K2 stack.

No row-level residual model is trained. For each held fold, the G3 coefficient
fit is repeated after dropping one additional donor fold. Agreement among these
legal jackknife predictions provides a model-uncertainty signal. Folds 4/5
select one fixed aggregation and dose, Fold 3 confirms it, and Fold 2 is read
only after that confirmation passes. Fold 1 is never evaluated or selected on.
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
OUT = ROOT / "oof_exports" / "main342_coefficient_jackknife_stability_t1"
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_EVALUATION_FOLD = 1
DOSES = (0.25, 0.50, 0.75, 1.00)
SHRINKAGE = 0.20
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
    source = P1_SOURCE.read_text(encoding="utf-8")
    marker = 'base, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError("P1 execution marker not found")
    name = "m342_t1_p1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(P1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(P1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


P = _load_p1_definitions()
K = P["K"]
G = K["_load_g1"]()
V = G["_load_inventory"]()
base = V["base"].reset_index(drop=True)
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
base_error = base_prediction - truth
folds = base["fold"].to_numpy(np.int8)

active_names = tuple(K["ACTIVE_ENDPOINTS"])
active_indices = [names.index(name) for name in active_names]
active_delta = paths[active_indices] - base_prediction[None, :]
raw_factor = K["_raw_gates"](active_delta, base)["magnitude_soft_mean1"]
candidate_indices = [index for index in range(len(names)) if index != base_index]
all_delta = paths[candidate_indices] - base_prediction[None, :]

expected = P["_endpoint_table"]().set_index("id")["k2_parent"]
expected = expected.loc[base["id"].astype(str)].to_numpy(np.float64)


def _fold_payload(held_fold):
    train = folds != held_fold
    valid = folds == held_fold
    standard_alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
    standard_move = SHRINKAGE * (standard_alpha @ all_delta[:, valid])
    denominator = max(float(np.mean(raw_factor[train])), 1e-6)
    gate = np.clip(
        raw_factor[valid] / denominator,
        0.0,
        float(K["MAX_NORMALISED_FACTOR"]),
    )
    parent = base_prediction[valid] + gate * standard_move
    gap = float(np.max(np.abs(parent - expected[valid])))
    if gap > 1e-8:
        raise RuntimeError(f"Fold {held_fold} K2 reproduction gap: {gap}")

    jackknife = []
    donor_folds = [fold for fold in range(1, 6) if fold != held_fold]
    for donor_fold in donor_folds:
        local_train = train & (folds != donor_fold)
        alpha, _ = G["_fit_coefficients"](all_delta, base_error, local_train)
        jackknife.append(SHRINKAGE * (alpha @ all_delta[:, valid]))
    jackknife = np.asarray(jackknife, np.float64)
    ordered = np.sort(jackknife, axis=0)
    mean = np.mean(jackknife, axis=0)
    median = np.median(jackknife, axis=0)
    trimmed = np.mean(ordered[1:3], axis=0)
    sign_stability = np.abs(np.mean(np.sign(jackknife), axis=0))
    ratio = np.std(jackknife, axis=0) / (np.abs(mean) + 1.0)
    alternatives = {
        "jackknife_mean": mean,
        "jackknife_median": median,
        "jackknife_trimmed_mean": trimmed,
        "median_blend50": 0.50 * standard_move + 0.50 * median,
        "sign_stability_soft": standard_move * (0.25 + 0.75 * sign_stability),
        "variance_shrink_0p5": standard_move / (1.0 + 0.50 * ratio),
        "variance_shrink_1": standard_move / (1.0 + ratio),
        "variance_shrink_2": standard_move / (1.0 + 2.0 * ratio),
    }
    return {
        "valid": valid,
        "gate": gate,
        "parent": parent,
        "standard_move": standard_move,
        "alternatives": alternatives,
        "mean_jackknife_std": float(np.mean(np.std(jackknife, axis=0))),
        "mean_sign_stability": float(np.mean(sign_stability)),
    }


def _metric(payload, prediction, endpoint):
    local = base.loc[payload["valid"]].reset_index(drop=True)
    return K["_metrics"](local, prediction, endpoint)


discovery_payload = {fold: _fold_payload(fold) for fold in DISCOVERY_FOLDS}
rows = []
for formula in next(iter(discovery_payload.values()))["alternatives"]:
    for dose in DOSES:
        sse = 0.0
        count = 0
        fold_reports = []
        for fold, payload in discovery_payload.items():
            alternative = payload["alternatives"][formula]
            move = payload["standard_move"] + float(dose) * (
                alternative - payload["standard_move"]
            )
            prediction = base_prediction[payload["valid"]] + payload["gate"] * move
            candidate_metric = _metric(payload, prediction, formula)
            parent_metric = _metric(payload, payload["parent"], "k2_parent")
            sse += float(candidate_metric["exact_sse"])
            count += int(candidate_metric["rows"])
            fold_reports.append({
                "fold": fold,
                "gain": parent_metric["rmse"] - candidate_metric["rmse"],
                "far_gain": parent_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"],
                "p95_change": candidate_metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
            })
        pooled_rmse = float(np.sqrt(sse / count))
        parent_sse = sum(
            _metric(payload, payload["parent"], "k2_parent")["exact_sse"]
            for payload in discovery_payload.values()
        )
        parent_rmse = float(np.sqrt(parent_sse / count))
        rows.append({
            "formula": formula,
            "dose": float(dose),
            "pooled_rmse": pooled_rmse,
            "pooled_gain": parent_rmse - pooled_rmse,
            "minimum_fold_gain": min(item["gain"] for item in fold_reports),
            "minimum_fold_far_gain": min(item["far_gain"] for item in fold_reports),
            "maximum_fold_p95_change": max(item["p95_change"] for item in fold_reports),
            "fold4_gain": fold_reports[0]["gain"],
            "fold5_gain": fold_reports[1]["gain"],
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


def _evaluate_held(held_fold):
    payload = _fold_payload(held_fold)
    alternative = payload["alternatives"][formula]
    move = payload["standard_move"] + dose * (alternative - payload["standard_move"])
    prediction = base_prediction[payload["valid"]] + payload["gate"] * move
    parent_metric = _metric(payload, payload["parent"], "k2_parent")
    candidate_metric = _metric(payload, prediction, formula)
    return {
        "fold": held_fold,
        "parent": parent_metric,
        "candidate": candidate_metric,
        "gain": parent_metric["rmse"] - candidate_metric["rmse"],
        "far_gain": parent_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"],
        "p95_change": candidate_metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        "mean_jackknife_std": payload["mean_jackknife_std"],
        "mean_sign_stability": payload["mean_sign_stability"],
    }


confirmation = _evaluate_held(CONFIRMATION_FOLD)
mechanism_pass = bool(
    len(safe) > 0
    and confirmation["gain"] >= 0.01
    and confirmation["far_gain"] >= 0.0
    and confirmation["p95_change"] <= MAX_P95_CHANGE
)
application = _evaluate_held(APPLICATION_FOLD) if mechanism_pass else None
checks = {
    "discovery_has_safe_aggregation": len(safe) > 0,
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
    "experiment": "main342_coefficient_jackknife_stability_t1",
    "stage": "fold45_jackknife_selection_fold3_fold2_confirmation",
    "selection_or_tuning_count": 1,
    "row_level_residual_training": False,
    "protocol": {
        "discovery_folds": DISCOVERY_FOLDS,
        "confirmation_fold": CONFIRMATION_FOLD,
        "application_fold": APPLICATION_FOLD if mechanism_pass else None,
        "untouched_evaluation_fold": UNTOUCHED_EVALUATION_FOLD,
    },
    "coverage": base.groupby("fold")["well"].nunique().astype(int).to_dict(),
    "selected": selected,
    "confirmation": confirmation,
    "application": application,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> fit the deployment coefficient jackknife and audit its full-data equivalence"
        if strict_pass else
        "FAIL -> coefficient uncertainty does not explain K2 residual error; close T1"
    ),
}

scan.to_csv(OUT / "discovery_scan.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-T1 DISCOVERY")
print(scan.head(20).to_string(index=False), flush=True)
print("\nM342-T1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
