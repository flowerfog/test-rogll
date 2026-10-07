"""M342-AC4: zero-retuning Fold-1 confirmation of frozen 2x K2 move."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_k2_double_fold1_ac4"
HELD_FOLD = 1
FROZEN_SCALE = 2.0


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_ac4"
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
B = P["B"]
O = P["O"]
base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()[["id", "k2_parent", "k2_move"]].copy()
base["id"] = base["id"].astype(str)
endpoints["id"] = endpoints["id"].astype(str)
frame = base.merge(endpoints, on="id", how="inner", validate="one_to_one")
frame = frame[frame["fold"].eq(HELD_FOLD)].reset_index(drop=True)
if frame["well"].nunique() != 155:
    raise RuntimeError(f"Fold1 coverage is {frame['well'].nunique()}, expected 155")

parent = frame["k2_parent"].to_numpy(np.float64)
move = frame["k2_move"].to_numpy(np.float64)
candidate = parent + (FROZEN_SCALE - 1.0) * move
reverse = parent - move
rolled = parent + _roll(frame, move)
metrics = {
    "parent": B["_metrics"](frame, parent, "current_k2"),
    "candidate": B["_metrics"](frame, candidate, "frozen_k2_scale_2"),
    "reverse": B["_metrics"](frame, reverse, "scale_0_control"),
    "depth_roll": B["_metrics"](frame, rolled, "depth_roll_control"),
}
gain = metrics["parent"]["rmse"] - metrics["candidate"]["rmse"]
far_gain = metrics["parent"]["far_gt2000_rmse"] - metrics["candidate"]["far_gt2000_rmse"]
p95_change = metrics["candidate"]["p95_well_rmse"] - metrics["parent"]["p95_well_rmse"]
reverse_loss = metrics["reverse"]["rmse"] - metrics["candidate"]["rmse"]
roll_loss = metrics["depth_roll"]["rmse"] - metrics["candidate"]["rmse"]
checks = {
    "fold1_wells_exact_155": frame["well"].nunique() == 155,
    "gain_ge_0p005": gain >= 0.005,
    "far_gain_ge_0": far_gain >= 0.0,
    "p95_change_le_0p20": p95_change <= 0.20,
    "reverse_loses_ge_0p01": reverse_loss >= 0.01,
    "depth_roll_loses_ge_0p005": roll_loss >= 0.005,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_k2_double_fold1_ac4",
    "stage": "zero_retuning_frozen_scale2_fold1_confirmation",
    "selection_or_tuning_count": 0,
    "frozen_from": "M342-AC3 Fold4/5 discovery and Fold3 confirmation",
    "frozen_scale": FROZEN_SCALE,
    "protocol": {"confirmation_fold": 1, "fold2_used_only_as_prior_support": 1},
    "coverage": {"rows": len(frame), "wells": frame["well"].nunique()},
    "metrics": metrics,
    "values": {
        "gain": gain,
        "far_gain": far_gain,
        "p95_change": p95_change,
        "reverse_loss": reverse_loss,
        "roll_loss": roll_loss,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> build exact K2 scale-2 deployment delta" if strict_pass else "FAIL -> retain exact K2 scale 1 and LB 6.290",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AC4 METRICS", flush=True)
for value in metrics.values():
    print(value, flush=True)
print("M342-AC4 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
