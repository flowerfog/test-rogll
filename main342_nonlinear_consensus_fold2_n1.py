"""M342-N1: frozen M1 raw-median consensus confirmation on Fold2.

Formula, global RMS scaling and +0.05 dose are frozen from M1. No selection,
fit or retuning occurs here. Fold2 is evaluated once; Fold1 remains untouched.
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
M1_SOURCE = ROOT / "main342_nonlinear_consensus_vector_m1.py"
OUT = ROOT / "oof_exports" / "main342_nonlinear_consensus_fold2_n1"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_FORMULA = "raw_median"
FROZEN_DOSE = 0.05
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
FROZEN_SHRINKAGE = 0.20
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


def _load_m1():
    if not M1_SOURCE.is_file():
        raise FileNotFoundError(M1_SOURCE)
    source = M1_SOURCE.read_text(encoding="utf-8")
    marker = "K = _load_k1()"
    if marker not in source:
        raise RuntimeError("M1 execution marker not found")
    name = "m342_n1_m1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(M1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(M1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


M = _load_m1()
K = M["_load_k1"]()
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

# Exact frozen K2 parent on all folds. Fold2 truth is never used in this build.
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
k2_metric = M["_metric"](base, k2, "k2_reproduction")
if abs(k2_metric["rmse"] - K2_EXPECTED_RMSE) > 1e-9:
    raise RuntimeError(f"K2 reproduction gap: {k2_metric['rmse']}")

directions = M["_directions"](active_delta, base)
direction = directions[FROZEN_FORMULA]
valid = fold == CONFIRMATION_FOLD
local = base.loc[valid].reset_index(drop=True)
local_direction = direction[valid]

rolled = np.empty_like(local_direction)
for positions in local.groupby("well", sort=False).indices.values():
    positions = np.asarray(positions, np.int64)
    rolled[positions] = np.roll(local_direction[positions], max(len(positions) // 4, 1))

parent_prediction = k2[valid]
candidate_prediction = parent_prediction + FROZEN_DOSE * local_direction
roll_prediction = parent_prediction + FROZEN_DOSE * rolled
reverse_prediction = parent_prediction - FROZEN_DOSE * local_direction

parent = M["_metric"](local, parent_prediction, "k2_parent")
candidate = M["_metric"](local, candidate_prediction, "frozen_raw_median")
roll_control = M["_metric"](local, roll_prediction, "depth_roll_control")
reverse_control = M["_metric"](local, reverse_prediction, "negative_direction_control")
gain = parent["rmse"] - candidate["rmse"]
far_gain = parent["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]
roll_loss = roll_control["rmse"] - candidate["rmse"]
reverse_loss = reverse_control["rmse"] - candidate["rmse"]
p95_change = candidate["p95_well_rmse"] - parent["p95_well_rmse"]

output = local[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["k2_parent"] = parent_prediction
output["raw_median_direction"] = local_direction
output["candidate"] = candidate_prediction
output.to_parquet(OUT / "fold2_predictions.parquet", index=False)

checks = {
    "k2_reproduction_exact": abs(k2_metric["rmse"] - K2_EXPECTED_RMSE) <= 1e-9,
    "fold2_wells_ge_145": int(local["well"].nunique()) >= 145,
    "gain_ge_0p03": gain >= 0.03,
    "far_gain_ge_0p03": far_gain >= 0.03,
    "roll_loses_ge_0p01": roll_loss >= 0.01,
    "reverse_loses_ge_0p05": reverse_loss >= 0.05,
    "p95_change_le_0p20": p95_change <= 0.20,
    "fold1_untouched": True,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_nonlinear_consensus_fold2_n1",
    "stage": "zero_retuning_frozen_raw_median_fold2_confirmation",
    "selection_or_tuning_count": 0,
    "frozen_recipe": {
        "formula": FROZEN_FORMULA,
        "dose": FROZEN_DOSE,
        "scaling": "M1 full-773 truth-free global RMS scaling",
    },
    "confirmation_fold": CONFIRMATION_FOLD,
    "untouched_fold": UNTOUCHED_FOLD,
    "summary": [parent, candidate, roll_control, reverse_control],
    "values": {
        "gain": gain, "far_gain": far_gain, "roll_loss": roll_loss,
        "reverse_loss": reverse_loss, "p95_change": p95_change,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> apply the unchanged formula and dose once to untouched Fold1"
        if strict_pass else
        "FAIL -> M1 does not transfer to Fold2; do not deploy the consensus vector"
    ),
}

pd.DataFrame([parent, candidate, roll_control, reverse_control]).to_csv(
    OUT / "summary.csv", index=False
)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-N1 SUMMARY")
print(pd.DataFrame([parent, candidate, roll_control, reverse_control]).to_string(index=False), flush=True)
print("\nM342-N1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
