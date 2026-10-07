"""M342-K1: nested truth-free endpoint-consensus rejection.

The G3 move is retained only where three independently generated endpoints
agree on its row-wise direction.  Gate choice is nested inside each outer
complete-well fold.  Gate features use endpoint predictions only.
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
G1_SOURCE = ROOT / "main342_fullbank_convex_stacking_g1.py"
OUT = ROOT / "oof_exports" / "main342_consensus_rejection_stack_k1"
OUT.mkdir(parents=True, exist_ok=True)

ACTIVE_ENDPOINTS = (
    "v144/coarse_crossfit",
    "psg/psg_tvt",
    "neighbor/fle_neighbor",
)
PROFILES = (
    "none",
    "sign_soft",
    "sign_hard",
    "magnitude_soft",
    "magnitude_hard",
    "magnitude_soft_mean1",
    "sign_hard_mean1",
)
FROZEN_SHRINKAGE = 0.20
MAX_NORMALISED_FACTOR = 2.0
INNER_MINIMUM_FOLD_GAIN = -0.005
INNER_MAXIMUM_P95_CHANGE = 0.20
G3_EXPECTED_RMSE = 7.7079303941645545


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


def _load_g1():
    if not G1_SOURCE.is_file():
        raise FileNotFoundError(G1_SOURCE)
    source = G1_SOURCE.read_text(encoding="utf-8")
    marker = "V = _load_inventory()"
    if marker not in source:
        raise RuntimeError("G1 execution marker not found")
    name = "m342_k1_g1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(G1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(G1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _raw_gates(active_delta, frame):
    normalised = np.empty_like(active_delta)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        local = active_delta[:, positions]
        scale = np.sqrt(np.mean(np.square(local), axis=1))
        scale = np.maximum(scale, 1e-6)
        normalised[:, positions] = local / scale[:, None]
    signs = np.sign(normalised)
    nonzero = np.sum(np.abs(signs) > 0.0, axis=0)
    sign_sum = np.abs(np.sum(signs, axis=0))
    unanimous = np.where(nonzero > 0, sign_sum == nonzero, False).astype(np.float64)
    magnitude = np.abs(np.sum(normalised, axis=0)) / np.maximum(
        np.sum(np.abs(normalised), axis=0), 1e-12
    )
    magnitude = np.clip(magnitude, 0.0, 1.0)
    return {
        "none": np.ones(active_delta.shape[1], np.float64),
        "sign_soft": 0.25 + 0.75 * unanimous,
        "sign_hard": unanimous,
        "magnitude_soft": 0.25 + 0.75 * magnitude,
        "magnitude_hard": (magnitude >= 0.60).astype(np.float64),
        "magnitude_soft_mean1": 0.25 + 0.75 * magnitude,
        "sign_hard_mean1": unanimous,
    }


def _gate(profile, raw_gates, fit_mask):
    factor = raw_gates[profile]
    if not profile.endswith("_mean1"):
        return factor
    denominator = max(float(np.mean(factor[fit_mask])), 1e-6)
    return np.clip(factor / denominator, 0.0, MAX_NORMALISED_FACTOR)


def _p95(frame, prediction):
    error = np.asarray(prediction, np.float64) - frame["tvt_true"].to_numpy(np.float64)
    local = pd.DataFrame({
        "well": frame["well"].astype(str).to_numpy(),
        "e2": np.square(error),
    }).groupby("well", sort=False)["e2"].mean()
    return float(np.sqrt(local).quantile(0.95))


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


G = _load_g1()
V = G["_load_inventory"]()
base = V["base"].reset_index(drop=True)
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
base_error = base_prediction - truth
fold = base["fold"].to_numpy(np.int8)

missing = [name for name in ACTIVE_ENDPOINTS if name not in names]
if missing:
    raise RuntimeError("Missing active endpoints: " + repr(missing))
active_indices = [names.index(name) for name in ACTIVE_ENDPOINTS]
active_delta = paths[active_indices] - base_prediction[None, :]
raw_gates = _raw_gates(active_delta, base)
all_indices = [index for index in range(len(names)) if index != base_index]
all_delta = paths[all_indices] - base_prediction[None, :]


def _select_profile(outer_held):
    outer_folds = [value for value in range(1, 6) if value != outer_held]
    payload = []
    for inner_held in outer_folds:
        fit_folds = [value for value in outer_folds if value != inner_held]
        fit_mask = np.isin(fold, fit_folds)
        valid_mask = fold == inner_held
        raw_alpha, _ = G["_fit_coefficients"](all_delta, base_error, fit_mask)
        move = FROZEN_SHRINKAGE * (raw_alpha @ all_delta[:, valid_mask])
        local = base.loc[valid_mask].reset_index(drop=True)
        base_local = base_prediction[valid_mask]
        base_rmse = float(np.sqrt(np.mean(np.square(base_local - truth[valid_mask]))))
        payload.append({
            "fold": inner_held,
            "fit_mask": fit_mask,
            "valid_mask": valid_mask,
            "local": local,
            "base": base_local,
            "truth": truth[valid_mask],
            "move": move,
            "base_rmse": base_rmse,
            "base_p95": _p95(local, base_local),
        })
    scan = []
    for profile in PROFILES:
        sse = 0.0
        rows = 0
        gains = []
        p95_changes = []
        activation = []
        for item in payload:
            factor = _gate(profile, raw_gates, item["fit_mask"])[item["valid_mask"]]
            prediction = item["base"] + factor * item["move"]
            error = prediction - item["truth"]
            rmse = float(np.sqrt(np.mean(np.square(error))))
            gains.append(item["base_rmse"] - rmse)
            p95_changes.append(_p95(item["local"], prediction) - item["base_p95"])
            activation.append(float(np.mean(factor)))
            sse += float(np.dot(error, error))
            rows += len(error)
        scan.append({
            "profile": profile,
            "inner_rmse": float(np.sqrt(sse / rows)),
            "minimum_inner_gain": float(min(gains)),
            "maximum_inner_p95_change": float(max(p95_changes)),
            "mean_factor": float(np.mean(activation)),
            "safe": bool(
                min(gains) >= INNER_MINIMUM_FOLD_GAIN
                and max(p95_changes) <= INNER_MAXIMUM_P95_CHANGE
            ),
        })
    safe = [row for row in scan if row["safe"]]
    selected = min(safe or scan, key=lambda row: (row["inner_rmse"], PROFILES.index(row["profile"])))
    return selected, scan


g3 = base_prediction.copy()
k1 = base_prediction.copy()
gate_roll = base_prediction.copy()
fold_reports = []
inner_scans = []
for held in range(1, 6):
    train = fold != held
    valid = fold == held
    selected, scan = _select_profile(held)
    inner_scans.extend({"outer_fold": held, **row} for row in scan)
    raw_alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
    move = FROZEN_SHRINKAGE * (raw_alpha @ all_delta[:, valid])
    factor_all = _gate(selected["profile"], raw_gates, train)
    factor = factor_all[valid]
    rolled_factor = factor.copy()
    local_frame = base.loc[valid].reset_index(drop=True)
    for positions in local_frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        shift = max(len(positions) // 4, 1)
        rolled_factor[positions] = np.roll(factor[positions], shift)
    g3[valid] += move
    k1[valid] += factor * move
    gate_roll[valid] += rolled_factor * move

    base_metric = _metrics(local_frame, base_prediction[valid], "base")
    g3_metric = _metrics(local_frame, g3[valid], "g3")
    k1_metric = _metrics(local_frame, k1[valid], "k1")
    roll_metric = _metrics(local_frame, gate_roll[valid], "gate_roll")
    fold_reports.append({
        "fold": held,
        "selected_profile": selected["profile"],
        "inner_rmse": selected["inner_rmse"],
        "minimum_inner_gain": selected["minimum_inner_gain"],
        "mean_factor": float(np.mean(factor)),
        "base_rmse": base_metric["rmse"],
        "g3_rmse": g3_metric["rmse"],
        "rmse": k1_metric["rmse"],
        "gain_vs_base": base_metric["rmse"] - k1_metric["rmse"],
        "increment_vs_g3": g3_metric["rmse"] - k1_metric["rmse"],
        "far_increment_vs_g3": g3_metric["far_gt2000_rmse"] - k1_metric["far_gt2000_rmse"],
        "gate_alignment_increment": roll_metric["rmse"] - k1_metric["rmse"],
        "p95_change_vs_g3": k1_metric["p95_well_rmse"] - g3_metric["p95_well_rmse"],
    })
    print("M342-K1 FOLD", _clean(fold_reports[-1]), flush=True)

summary = pd.DataFrame([
    _metrics(base, base_prediction, "v168"),
    _metrics(base, g3, "g3_reproduction"),
    _metrics(base, k1, "nested_consensus_rejection"),
    _metrics(base, gate_roll, "within_well_gate_roll_control"),
]).sort_values("rmse", ignore_index=True)
fold_frame = pd.DataFrame(fold_reports)
scan_frame = pd.DataFrame(inner_scans)
lookup = summary.set_index("endpoint")
g3_rmse = float(lookup.loc["g3_reproduction", "rmse"])
k1_rmse = float(lookup.loc["nested_consensus_rejection", "rmse"])
increment = g3_rmse - k1_rmse
gate_alignment = float(lookup.loc["within_well_gate_roll_control", "rmse"] - k1_rmse)
checks = {
    "g3_reproduction_abs_gap_le_1e_6": abs(g3_rmse - G3_EXPECTED_RMSE) <= 1e-6,
    "increment_vs_g3_ge_0p015": increment >= 0.015,
    "positive_increment_folds_ge_4": int((fold_frame["increment_vs_g3"] > 0).sum()) >= 4,
    "minimum_fold_increment_ge_minus0p01": float(fold_frame["increment_vs_g3"].min()) >= -0.01,
    "minimum_far_increment_ge_minus0p01": float(fold_frame["far_increment_vs_g3"].min()) >= -0.01,
    "gate_alignment_increment_ge_0p01": gate_alignment >= 0.01,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change_vs_g3"].max()) <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_consensus_rejection_stack_k1",
    "stage": "nested_truth_free_endpoint_consensus_rejection_gateway",
    "sample_unit": "complete physical well; all official hidden rows",
    "gate_inputs": "three endpoint deltas only; no truth, GR or held-fold statistics",
    "active_endpoints": ACTIVE_ENDPOINTS,
    "profiles": PROFILES,
    "frozen_shrinkage": FROZEN_SHRINKAGE,
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "summary": summary.to_dict("records"),
    "folds": fold_reports,
    "values": {"increment_vs_g3": increment, "gate_alignment_increment": gate_alignment},
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the modal gate profile and run a no-retuning confirmation"
        if strict_pass else
        "FAIL -> endpoint agreement cannot safely identify G3 error activation; close K1"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
scan_frame.to_csv(OUT / "inner_scans.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-K1 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-K1 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-K1 INNER SCANS")
print(scan_frame.to_string(index=False), flush=True)
print("\nM342-K1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
