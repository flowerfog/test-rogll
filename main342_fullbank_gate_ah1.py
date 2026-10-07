"""M342-AH1: full-bank agreement gate for the frozen K2 move."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_fullbank_gate_ah1"
DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
SCALES = np.asarray([0.50, 0.75, 1.0, 1.25, 1.50], np.float64)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_ah1"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _metric(B, frame, pred, name):
    return B["_metrics"](frame.reset_index(drop=True), pred, name)


started = time.time()
OUT.mkdir(parents=True, exist_ok=True)
P = _bootstrap(SOURCE)
B = P["B"]
O = P["O"]
K = P["K"]
G = K["_load_g1"]()
V = G["_load_inventory"]()
frame = V["base"].reset_index(drop=True)
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
base_error = base_prediction - truth
folds = frame["fold"].to_numpy(np.int8)
candidate_indices = [i for i in range(len(names)) if i != base_index]
all_delta = paths[candidate_indices] - base_prediction[None, :]
active_indices = [names.index(name) for name in K["ACTIVE_ENDPOINTS"]]
active_delta = paths[active_indices] - base_prediction[None, :]
gates = {
    "active3": K["_raw_gates"](active_delta, frame)["magnitude_soft_mean1"],
    "full11": K["_raw_gates"](all_delta, frame)["magnitude_soft_mean1"],
}
max_factor = float(K["MAX_NORMALISED_FACTOR"])

records = []
for gate_name, gate_values in gates.items():
    for held_fold in (CONFIRMATION_FOLD,):
        held = folds == held_fold
        train = ~held
        alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
        factor = np.clip(gate_values[held] / max(float(np.mean(gate_values[train])), 1e-6), 0.0, max_factor)
        move = 0.20 * factor * (alpha @ all_delta[:, held])
        local = frame.loc[held].reset_index(drop=True)
        parent = base_prediction[held]
        parent_metric = _metric(B, local, parent, "k2_parent")
        for scale in SCALES:
            pred = parent + float(scale) * move
            metric = _metric(B, local, pred, f"{gate_name}_{scale:g}")
            records.append({
                "gate": gate_name,
                "fold": held_fold,
                "scale": float(scale),
                "rmse": metric["rmse"],
                "gain": parent_metric["rmse"] - metric["rmse"],
                "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
            })

discovery_rows = []
for gate_name, gate_values in gates.items():
    donor_scores = []
    for scale in SCALES:
        fold_scores = []
        for held_fold in DISCOVERY_FOLDS:
            held = folds == held_fold
            train = ~held
            alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
            factor = np.clip(gate_values[held] / max(float(np.mean(gate_values[train])), 1e-6), 0.0, max_factor)
            move = 0.20 * factor * (alpha @ all_delta[:, held])
            local = frame.loc[held].reset_index(drop=True)
            parent = base_prediction[held]
            parent_metric = _metric(B, local, parent, "k2_parent")
            metric = _metric(B, local, parent + float(scale) * move, f"{gate_name}_{scale:g}")
            fold_scores.append({
                "gain": parent_metric["rmse"] - metric["rmse"],
                "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
            })
        donor = frame[frame["fold"].isin(DISCOVERY_FOLDS)].reset_index(drop=True)
        held = np.isin(folds, DISCOVERY_FOLDS)
        alpha, _ = G["_fit_coefficients"](all_delta, base_error, held)
        factor = np.clip(gate_values[held] / max(float(np.mean(gate_values[held])), 1e-6), 0.0, max_factor)
        move = 0.20 * factor * (alpha @ all_delta[:, held])
        donor_parent = base_prediction[held]
        donor_metric = _metric(B, donor, donor_parent + float(scale) * move, f"{gate_name}_{scale:g}")
        discovery_rows.append({
            "gate": gate_name,
            "scale": float(scale),
            "pooled_rmse": donor_metric["rmse"],
            "minimum_fold_gain": min(item["gain"] for item in fold_scores),
            "minimum_fold_far_gain": min(item["far_gain"] for item in fold_scores),
            "maximum_fold_p95_change": max(item["p95_change"] for item in fold_scores),
        })

discovery = pd.DataFrame(discovery_rows)
safe = discovery[
    (discovery["minimum_fold_gain"] >= 0.0)
    & (discovery["minimum_fold_far_gain"] >= 0.0)
    & (discovery["maximum_fold_p95_change"] <= 0.20)
]
selected = safe.sort_values(["pooled_rmse", "scale"]).iloc[0] if not safe.empty else discovery.sort_values(["pooled_rmse", "scale"]).iloc[0]
confirmation = pd.DataFrame(records)
chosen = confirmation[
    confirmation["gate"].eq(selected["gate"]) & confirmation["scale"].eq(float(selected["scale"]))
].iloc[0]
checks = {
    "complete_773_wells": frame["well"].nunique() == 773,
    "discovery_has_safe_gate": not safe.empty,
    "confirmation_gain_ge_0p03": float(chosen["gain"]) >= 0.03,
    "confirmation_far_gain_ge_0": float(chosen["far_gain"]) >= 0.0,
    "confirmation_p95_change_le_0p20": float(chosen["p95_change"]) <= 0.20,
}
verdict = {
    "experiment": "main342_fullbank_gate_ah1",
    "stage": "fold45_gate_selection_fold3_confirmation_fullbank_vs_active3",
    "selection_or_tuning_count": 1,
    "candidate_bank_size": len(candidate_indices),
    "active_bank_size": len(active_indices),
    "discovery": discovery.to_dict("records"),
    "selected": selected.to_dict(),
    "confirmation": confirmation.to_dict("records"),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> frozen full-bank gate Fold2 confirmation" if all(checks.values()) else "FAIL -> full-bank agreement gate does not transfer; close AH1",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AH1 DISCOVERY", flush=True)
print(discovery.sort_values("pooled_rmse").to_string(index=False), flush=True)
print("M342-AH1 CONFIRMATION", flush=True)
print(confirmation.to_string(index=False), flush=True)
print("M342-AH1 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
