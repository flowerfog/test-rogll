"""M342-L1: dormant endpoint residual screen after frozen K2.

Fold4 and Fold5 cross-transfer select one endpoint that had zero full-fit G3
weight. Its signed scalar dose is then fitted on Fold4+5 and applied once to
Fold3. Fold1/2 are not used in candidate selection or confirmation statistics.
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
OUT = ROOT / "oof_exports" / "main342_dormant_endpoint_residual_l1"
OUT.mkdir(parents=True, exist_ok=True)

ACTIVE_ENDPOINTS = {
    "v144/coarse_crossfit",
    "psg/psg_tvt",
    "neighbor/fle_neighbor",
}
DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
UNTOUCHED_FOLDS = (1, 2)
FROZEN_SHRINKAGE = 0.20
MAX_ABS_INCREMENT_DOSE = 0.03
K2_EXPECTED_RMSE = 7.689693565766868


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
    if not K1_SOURCE.is_file():
        raise FileNotFoundError(K1_SOURCE)
    source = K1_SOURCE.read_text(encoding="utf-8")
    marker = "G = _load_g1()"
    if marker not in source:
        raise RuntimeError("K1 execution marker not found")
    name = "m342_l1_k1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(K1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(K1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _dose(direction, parent_error, mask):
    x = direction[mask]
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return 0.0
    value = -float(np.dot(x, parent_error[mask])) / denominator
    return float(np.clip(value, -MAX_ABS_INCREMENT_DOSE, MAX_ABS_INCREMENT_DOSE))


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, wells = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    well_sse = np.bincount(codes, weights=np.square(error), minlength=len(wells))
    well_rmse = np.sqrt(well_sse / counts)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(len(wells)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


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
raw_gates = K["_raw_gates"](active_delta, base)
raw_factor = raw_gates["magnitude_soft_mean1"]
all_indices = [index for index in range(len(names)) if index != base_index]
all_delta = paths[all_indices] - base_prediction[None, :]

# Reconstruct the exact frozen K2 fivefold parent before any L1 screening.
k2 = base_prediction.copy()
for held in range(1, 6):
    train = fold != held
    valid = fold == held
    raw_alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
    move = FROZEN_SHRINKAGE * (raw_alpha @ all_delta[:, valid])
    normalizer = max(float(np.mean(raw_factor[train])), 1e-6)
    factor = np.clip(
        raw_factor[valid] / normalizer,
        0.0,
        float(K["MAX_NORMALISED_FACTOR"]),
    )
    k2[valid] += factor * move
k2_error = k2 - truth
k2_metric = _metrics(base, k2, "k2_reproduction")
if abs(k2_metric["rmse"] - K2_EXPECTED_RMSE) > 1e-9:
    raise RuntimeError(f"K2 reproduction gap: {k2_metric['rmse']}")

# Only endpoints with zero full-data G3 weight are eligible. Recompute the
# full fit once to avoid relying on a copied manifest.
full_raw_alpha, _ = G["_fit_coefficients"](
    all_delta, base_error, np.ones(len(base), dtype=bool)
)
full_alpha = FROZEN_SHRINKAGE * full_raw_alpha
weight_by_name = dict(zip([names[index] for index in all_indices], full_alpha))
dormant_names = [
    name for name in weight_by_name
    if name not in ACTIVE_ENDPOINTS and abs(weight_by_name[name]) <= 1e-8
]
if not dormant_names:
    raise RuntimeError("No zero-weight dormant endpoints in the full G3 bank")

directions = {
    name: paths[names.index(name)] - base_prediction
    for name in dormant_names
}
rolled_directions = {}
for name, direction in directions.items():
    rolled = np.empty_like(direction)
    for positions in base.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        shift = max(len(positions) // 4, 1)
        rolled[positions] = np.roll(direction[positions], shift)
    rolled_directions[name] = rolled

discovery_rows = []
for name in dormant_names:
    total_sse = 0.0
    total_rows = 0
    gains = []
    doses = []
    for held in DISCOVERY_FOLDS:
        donor = DISCOVERY_FOLDS[1] if held == DISCOVERY_FOLDS[0] else DISCOVERY_FOLDS[0]
        donor_mask = fold == donor
        held_mask = fold == held
        dose = _dose(directions[name], k2_error, donor_mask)
        candidate = k2[held_mask] + dose * directions[name][held_mask]
        local = base.loc[held_mask].reset_index(drop=True)
        parent_metric = _metrics(local, k2[held_mask], "parent")
        metric = _metrics(local, candidate, "candidate")
        gains.append(parent_metric["rmse"] - metric["rmse"])
        doses.append(dose)
        error = candidate - truth[held_mask]
        total_sse += float(np.dot(error, error))
        total_rows += int(held_mask.sum())
    discovery_rows.append({
        "candidate": name,
        "cross_transfer_rmse": float(np.sqrt(total_sse / total_rows)),
        "minimum_fold_gain": float(min(gains)),
        "mean_gain": float(np.mean(gains)),
        "dose_fold4_from5": doses[0],
        "dose_fold5_from4": doses[1],
        "dose_sign_stable": bool(np.sign(doses[0]) == np.sign(doses[1])),
        "safe": bool(min(gains) >= 0.0 and np.sign(doses[0]) == np.sign(doses[1])),
    })
discovery = pd.DataFrame(discovery_rows).sort_values(
    ["safe", "cross_transfer_rmse"], ascending=[False, True], ignore_index=True
)
safe = discovery.loc[discovery["safe"]]
selected_row = (safe.iloc[0] if len(safe) else discovery.iloc[0]).to_dict()
selected_name = str(selected_row["candidate"])

donor_mask = np.isin(fold, DISCOVERY_FOLDS)
confirmation_mask = fold == CONFIRMATION_FOLD
selected_dose = _dose(directions[selected_name], k2_error, donor_mask)
confirmation = k2[confirmation_mask] + selected_dose * directions[selected_name][confirmation_mask]
roll_confirmation = (
    k2[confirmation_mask]
    + selected_dose * rolled_directions[selected_name][confirmation_mask]
)
local = base.loc[confirmation_mask].reset_index(drop=True)
parent_metric = _metrics(local, k2[confirmation_mask], "k2_parent")
candidate_metric = _metrics(local, confirmation, "dormant_increment")
roll_metric = _metrics(local, roll_confirmation, "depth_roll_control")
gain = parent_metric["rmse"] - candidate_metric["rmse"]
far_gain = parent_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"]
roll_loss = roll_metric["rmse"] - candidate_metric["rmse"]
p95_change = candidate_metric["p95_well_rmse"] - parent_metric["p95_well_rmse"]

checks = {
    "k2_reproduction_exact": abs(k2_metric["rmse"] - K2_EXPECTED_RMSE) <= 1e-9,
    "discovery_has_safe_candidate": bool(len(safe)),
    "confirmation_gain_ge_0p01": gain >= 0.01,
    "confirmation_far_gain_ge_0": far_gain >= 0.0,
    "roll_loses_ge_0p005": roll_loss >= 0.005,
    "p95_change_le_0p20": p95_change <= 0.20,
    "selected_abs_dose_le_0p03": abs(selected_dose) <= MAX_ABS_INCREMENT_DOSE + 1e-12,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_dormant_endpoint_residual_l1",
    "stage": "fold45_cross_transfer_selection_fold3_confirmation_after_frozen_k2",
    "selection_or_tuning_count": 1,
    "protocol": {
        "discovery_folds": DISCOVERY_FOLDS,
        "confirmation_fold": CONFIRMATION_FOLD,
        "untouched_folds": UNTOUCHED_FOLDS,
        "maximum_abs_increment_dose": MAX_ABS_INCREMENT_DOSE,
    },
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "dormant_endpoints": dormant_names,
    "selected": selected_row,
    "selected_dose": selected_dose,
    "confirmation": {
        "parent": parent_metric,
        "candidate": candidate_metric,
        "roll_control": roll_metric,
        "gain": gain,
        "far_gain": far_gain,
        "roll_loss": roll_loss,
        "p95_change": p95_change,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze candidate/sign/dose and apply once to Fold2"
        if strict_pass else
        "FAIL -> no dormant endpoint adds transferable residual information after K2; close L1"
    ),
}

discovery.to_csv(OUT / "discovery.csv", index=False)
pd.DataFrame([parent_metric, candidate_metric, roll_metric]).to_csv(
    OUT / "confirmation_summary.csv", index=False
)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-L1 DISCOVERY")
print(discovery.to_string(index=False), flush=True)
print("\nM342-L1 CONFIRMATION")
print(pd.DataFrame([parent_metric, candidate_metric, roll_metric]).to_string(index=False), flush=True)
print("\nM342-L1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
