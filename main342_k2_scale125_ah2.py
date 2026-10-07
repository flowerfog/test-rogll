"""M342-AH2: zero-retuning confirmation of frozen active3 K2 scale 1.25."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_k2_scale125_ah2"
FROZEN_SCALE = 1.25


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_ah2"
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


def _roll(frame, values):
    output = np.asarray(values, np.float64).copy()
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        if len(positions) > 1:
            output[positions] = np.roll(output[positions], max(1, len(positions) // 3))
    return output


started = time.time()
OUT.mkdir(parents=True, exist_ok=True)
P = _bootstrap(SOURCE)
B, O, K = P["B"], P["O"], P["K"]
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
raw_gate = K["_raw_gates"](active_delta, frame)["magnitude_soft_mean1"]
max_factor = float(K["MAX_NORMALISED_FACTOR"])

rows = []
for held_fold in (2, 1):
    held = folds == held_fold
    train = ~held
    alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
    factor = np.clip(raw_gate[held] / max(float(np.mean(raw_gate[train])), 1e-6), 0.0, max_factor)
    move = 0.20 * factor * (alpha @ all_delta[:, held])
    local = frame.loc[held].reset_index(drop=True)
    parent = base_prediction[held]
    candidate = parent + (FROZEN_SCALE - 1.0) * move
    reverse = parent - (FROZEN_SCALE - 1.0) * move
    rolled = parent + _roll(local, (FROZEN_SCALE - 1.0) * move)
    parent_metric = B["_metrics"](local, parent, "current_k2")
    candidate_metric = B["_metrics"](local, candidate, "frozen_scale125")
    reverse_metric = B["_metrics"](local, reverse, "reverse_control")
    roll_metric = B["_metrics"](local, rolled, "depth_roll_control")
    rows.append({
        "fold": held_fold,
        "rows": len(local),
        "wells": local["well"].nunique(),
        "base_rmse": parent_metric["rmse"],
        "rmse": candidate_metric["rmse"],
        "gain": parent_metric["rmse"] - candidate_metric["rmse"],
        "far_gain": parent_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"],
        "p95_change": candidate_metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        "reverse_loss": reverse_metric["rmse"] - candidate_metric["rmse"],
        "roll_loss": roll_metric["rmse"] - candidate_metric["rmse"],
    })
result = pd.DataFrame(rows)
result.to_csv(OUT / "confirmation_folds.csv", index=False)
checks = {
    "fold2_wells_ge_145": int(result.loc[result["fold"].eq(2), "wells"].iloc[0]) >= 145,
    "fold1_wells_ge_145": int(result.loc[result["fold"].eq(1), "wells"].iloc[0]) >= 145,
    "fold2_gain_ge_0p005": float(result.loc[result["fold"].eq(2), "gain"].iloc[0]) >= 0.005,
    "fold2_far_gain_ge_0": float(result.loc[result["fold"].eq(2), "far_gain"].iloc[0]) >= 0.0,
    "fold1_gain_ge_0": float(result.loc[result["fold"].eq(1), "gain"].iloc[0]) >= 0.0,
    "fold1_far_gain_ge_0": float(result.loc[result["fold"].eq(1), "far_gain"].iloc[0]) >= 0.0,
    "p95_changes_le_0p20": bool((result["p95_change"] <= 0.20).all()),
}
verdict = {
    "experiment": "main342_k2_scale125_ah2",
    "stage": "zero_retuning_fold2_fold1_confirmation_after_ah1",
    "selection_or_tuning_count": 0,
    "frozen_recipe": {"gate": "active3", "scale": FROZEN_SCALE},
    "folds": result.to_dict("records"),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> build one scale125 deployment candidate" if all(checks.values()) else "FAIL -> retain K2 scale1; close AH2",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AH2 FOLDS", flush=True)
print(result.to_string(index=False), flush=True)
print("M342-AH2 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
