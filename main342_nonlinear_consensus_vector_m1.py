"""M342-M1: nonlinear consensus-vector endpoint after frozen K2.

Three active endpoint deltas are converted into truth-free robust consensus
directions per complete well. Fold4/5 cross-transfer selects one fixed formula;
Fold3 is the sole confirmation fold. Fold1/2 remain unused.
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
K1_SOURCE = ROOT / "main342_consensus_rejection_stack_k1.py"
OUT = ROOT / "oof_exports" / "main342_nonlinear_consensus_vector_m1"
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
UNTOUCHED_FOLDS = (1, 2)
FROZEN_SHRINKAGE = 0.20
MAX_ABS_DOSE = 0.05
K2_EXPECTED_RMSE = 7.689693565766868
FORMULAS = (
    "raw_median",
    "normalised_median",
    "normalised_mean",
    "agreement_median",
    "unanimous_minimum",
    "agreement_harmonic",
)


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_k1():
    source = K1_SOURCE.read_text(encoding="utf-8")
    marker = "G = _load_g1()"
    if marker not in source:
        raise RuntimeError("K1 execution marker not found")
    name = "m342_m1_k1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(K1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(K1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _dose(direction, error, mask):
    x = direction[mask]
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(x, error[mask]) / denominator, -MAX_ABS_DOSE, MAX_ABS_DOSE))


def _metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, wells = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    well_sse = np.bincount(codes, weights=np.square(error), minlength=len(wells))
    well_rmse = np.sqrt(well_sse / counts)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint, "rows": int(len(frame)), "wells": int(len(wells)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


def _directions(delta, frame):
    output = {name: np.empty(delta.shape[1], np.float64) for name in FORMULAS}
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        local = delta[:, positions]
        rms = np.maximum(np.sqrt(np.mean(np.square(local), axis=1)), 1e-6)
        z = local / rms[:, None]
        typical = float(np.median(rms))
        agreement = np.abs(np.sum(z, axis=0)) / np.maximum(np.sum(np.abs(z), axis=0), 1e-12)
        sign_sum = np.sign(np.sum(z, axis=0))
        unanimous = (np.abs(np.sum(np.sign(z), axis=0)) == z.shape[0]).astype(np.float64)
        harmonic = z.shape[0] / np.maximum(np.sum(1.0 / np.maximum(np.abs(z), 1e-3), axis=0), 1e-12)
        normalised_median = np.median(z, axis=0) * typical
        output["raw_median"][positions] = np.median(local, axis=0)
        output["normalised_median"][positions] = normalised_median
        output["normalised_mean"][positions] = np.mean(z, axis=0) * typical
        output["agreement_median"][positions] = agreement * normalised_median
        output["unanimous_minimum"][positions] = (
            unanimous * sign_sum * np.min(np.abs(local), axis=0)
        )
        output["agreement_harmonic"][positions] = agreement * sign_sum * harmonic * typical
    # Put every formula on the same global RMS scale. Scalar doses then have a
    # comparable interpretation and cannot win through arbitrary amplitude.
    target_rms = float(np.median([
        np.sqrt(np.mean(np.square(delta[index]))) for index in range(delta.shape[0])
    ]))
    for name, values in output.items():
        scale = max(float(np.sqrt(np.mean(np.square(values)))), 1e-8)
        output[name] = values * (target_rms / scale)
    return output


K = _load_k1()
G = K["_load_g1"]()
V = G["_load_inventory"]()
base = V["base"].reset_index(drop=True)
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
base_error = base_prediction - truth
fold = base["fold"].to_numpy(np.int8)
active_indices = [names.index(name) for name in K["ACTIVE_ENDPOINTS"]]
active_delta = paths[active_indices] - base_prediction[None, :]
raw_factor = K["_raw_gates"](active_delta, base)["magnitude_soft_mean1"]
all_indices = [index for index in range(len(names)) if index != base_index]
all_delta = paths[all_indices] - base_prediction[None, :]

k2 = base_prediction.copy()
for held in range(1, 6):
    train = fold != held
    valid = fold == held
    alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
    move = FROZEN_SHRINKAGE * (alpha @ all_delta[:, valid])
    factor = np.clip(
        raw_factor[valid] / max(float(np.mean(raw_factor[train])), 1e-6),
        0.0, float(K["MAX_NORMALISED_FACTOR"]),
    )
    k2[valid] += factor * move
k2_error = k2 - truth
k2_all_metric = _metric(base, k2, "k2_reproduction")
if abs(k2_all_metric["rmse"] - K2_EXPECTED_RMSE) > 1e-9:
    raise RuntimeError(f"K2 reproduction gap: {k2_all_metric['rmse']}")

directions = _directions(active_delta, base)
rolled = {}
for name, direction in directions.items():
    control = np.empty_like(direction)
    for positions in base.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        control[positions] = np.roll(direction[positions], max(len(positions) // 4, 1))
    rolled[name] = control

rows = []
for name in FORMULAS:
    total_sse, total_rows = 0.0, 0
    gains, doses = [], []
    for held in DISCOVERY_FOLDS:
        donor = DISCOVERY_FOLDS[1] if held == DISCOVERY_FOLDS[0] else DISCOVERY_FOLDS[0]
        donor_mask, held_mask = fold == donor, fold == held
        dose = _dose(directions[name], k2_error, donor_mask)
        prediction = k2[held_mask] + dose * directions[name][held_mask]
        local = base.loc[held_mask].reset_index(drop=True)
        parent = _metric(local, k2[held_mask], "parent")
        candidate = _metric(local, prediction, "candidate")
        gains.append(parent["rmse"] - candidate["rmse"])
        doses.append(dose)
        error = prediction - truth[held_mask]
        total_sse += float(np.dot(error, error))
        total_rows += int(held_mask.sum())
    rows.append({
        "formula": name,
        "cross_transfer_rmse": float(np.sqrt(total_sse / total_rows)),
        "minimum_fold_gain": float(min(gains)),
        "mean_gain": float(np.mean(gains)),
        "dose_fold4_from5": doses[0], "dose_fold5_from4": doses[1],
        "dose_sign_stable": bool(np.sign(doses[0]) == np.sign(doses[1])),
        "safe": bool(min(gains) >= 0.0 and np.sign(doses[0]) == np.sign(doses[1])),
    })
discovery = pd.DataFrame(rows).sort_values(
    ["safe", "cross_transfer_rmse"], ascending=[False, True], ignore_index=True
)
safe = discovery.loc[discovery["safe"]]
selected = (safe.iloc[0] if len(safe) else discovery.iloc[0]).to_dict()
selected_name = str(selected["formula"])

donor_mask = np.isin(fold, DISCOVERY_FOLDS)
valid = fold == CONFIRMATION_FOLD
dose = _dose(directions[selected_name], k2_error, donor_mask)
prediction = k2[valid] + dose * directions[selected_name][valid]
roll_prediction = k2[valid] + dose * rolled[selected_name][valid]
local = base.loc[valid].reset_index(drop=True)
parent = _metric(local, k2[valid], "k2_parent")
candidate = _metric(local, prediction, "consensus_vector")
roll_control = _metric(local, roll_prediction, "depth_roll_control")
gain = parent["rmse"] - candidate["rmse"]
far_gain = parent["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]
roll_loss = roll_control["rmse"] - candidate["rmse"]
p95_change = candidate["p95_well_rmse"] - parent["p95_well_rmse"]
checks = {
    "k2_reproduction_exact": abs(k2_all_metric["rmse"] - K2_EXPECTED_RMSE) <= 1e-9,
    "discovery_has_safe_formula": bool(len(safe)),
    "confirmation_gain_ge_0p01": gain >= 0.01,
    "confirmation_far_gain_ge_0": far_gain >= 0.0,
    "roll_loses_ge_0p005": roll_loss >= 0.005,
    "p95_change_le_0p20": p95_change <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_nonlinear_consensus_vector_m1",
    "stage": "fold45_formula_selection_fold3_confirmation_after_frozen_k2",
    "selection_or_tuning_count": 1,
    "protocol": {
        "discovery_folds": DISCOVERY_FOLDS,
        "confirmation_fold": CONFIRMATION_FOLD,
        "untouched_folds": UNTOUCHED_FOLDS,
        "maximum_abs_dose": MAX_ABS_DOSE,
    },
    "formulas": FORMULAS,
    "selected": selected,
    "selected_dose": dose,
    "confirmation": {
        "parent": parent, "candidate": candidate, "roll_control": roll_control,
        "gain": gain, "far_gain": far_gain, "roll_loss": roll_loss,
        "p95_change": p95_change,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze formula and dose, then apply once to Fold2"
        if strict_pass else
        "FAIL -> consensus magnitude is useful only as a gate, not a new endpoint; close M1"
    ),
}

discovery.to_csv(OUT / "discovery.csv", index=False)
pd.DataFrame([parent, candidate, roll_control]).to_csv(OUT / "confirmation.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-M1 DISCOVERY")
print(discovery.to_string(index=False), flush=True)
print("\nM342-M1 CONFIRMATION")
print(pd.DataFrame([parent, candidate, roll_control]).to_string(index=False), flush=True)
print("\nM342-M1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
