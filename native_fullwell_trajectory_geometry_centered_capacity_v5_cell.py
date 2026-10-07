"""Uncensored 24-well convergence audit for the trajectory head.

Run in the same kernel after geometry-centred v3 stopped at its 24-well gate.
The v3 best score occurred at the final epoch, so this cell distinguishes a
budget-censored run from a true capacity failure.  It never starts Fold 1.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_V4 = {
    "capacity_pool_fast",
    "coefficient_center_fast",
    "coefficient_mean_fast",
    "_fixed_coefficient_metrics_fast",
    "_oracle_metrics_fast",
    "_seed_everything",
    "FastTrajectorySHead",
    "_train_fast",
    "device_fast",
    "OOF_ROOT",
}
missing_v4 = sorted(REQUIRED_V4.difference(globals()))
if missing_v4:
    raise RuntimeError(
        "Run this in the live kernel after geometry-centred v3; missing "
        + repr(missing_v4)
    )


EXPERIMENT_V4 = "native_fullwell_trajectory_geometry_centered_capacity_v5"
OUT_DIR_V4 = OOF_ROOT / EXPERIMENT_V4
MODEL_DIR_V4 = OUT_DIR_V4 / "models"
for directory in (OUT_DIR_V4, MODEL_DIR_V4):
    directory.mkdir(parents=True, exist_ok=True)

# This is the exact v3 24-well seed that reached 8.226 ft at epoch 180.
SEED_V4 = 20260724
EPOCHS_V4 = 600
REPORT_EVERY_V4 = 20
MARGIN_FT_V4 = 0.75


def _json_v4(path, payload):
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


started_v4 = time.time()
records_v4 = list(capacity_pool_fast)
if len(records_v4) != 24:
    raise RuntimeError(f"Expected 24 capacity wells, found {len(records_v4)}")

baseline_v4 = _fixed_coefficient_metrics_fast(
    records_v4, coefficient_center_fast
)
oracle_v4 = _oracle_metrics_fast(records_v4)
gate_v4 = float(oracle_v4["rmse"] + MARGIN_FT_V4)

_seed_everything(SEED_V4)
model_v4 = FastTrajectorySHead(len(coefficient_mean_fast)).to(device_fast)
parameters_v4 = int(sum(parameter.numel() for parameter in model_v4.parameters()))

print(
    "GEOMETRY-CENTRED 24-WELL CONVERGENCE CONFIG",
    {
        "wells": len(records_v4),
        "parameters": parameters_v4,
        "epochs": EPOCHS_V4,
        "carry_forward_tvt_baseline": baseline_v4,
        "representation_oracle": oracle_v4,
        "gate_rmse": gate_v4,
    },
    flush=True,
)

model_v4, best_v4, history_v4, runtime_v4 = _train_fast(
    model_v4,
    records_v4,
    records_v4,
    device_fast,
    EPOCHS_V4,
    REPORT_EVERY_V4,
    "capacity",
    capacity_gate=gate_v4,
    # Disable the inherited epoch-60 baseline fail-fast.  The unchanged RMSE
    # gate can still stop a successful run early, otherwise all 600 epochs run.
    capacity_baseline={"rmse": float("inf")},
)
history_v4.insert(0, "scenario", "twentyfour_wells_convergence")
history_v4.to_csv(OUT_DIR_V4 / "capacity_history.csv", index=False)
best_v4["bywell"].to_csv(OUT_DIR_V4 / "capacity_bywell.csv", index=False)

checkpoint_v4 = MODEL_DIR_V4 / "best_24well_capacity.pt"
torch.save(
    {
        "experiment": EXPERIMENT_V4,
        "seed": SEED_V4,
        "epoch": int(best_v4["epoch"]),
        "model": best_v4["state"],
        "metrics": best_v4["metrics"],
        "free_dimensions": len(coefficient_mean_fast),
        "coefficient_center": "carry_forward_tvt",
        "representation_oracle": oracle_v4,
        "gate_rmse": gate_v4,
    },
    checkpoint_v4,
)

pass_v4 = bool(best_v4["metrics"]["rmse"] <= gate_v4)
verdict_v4 = {
    "experiment": EXPERIMENT_V4,
    "stage": "24_well_capacity_convergence",
    "parameters": parameters_v4,
    "epochs_budget": EPOCHS_V4,
    "epochs_executed": int(history_v4["epoch"].max()),
    "capacity_fail_fast_disabled": True,
    "best_epoch": int(best_v4["epoch"]),
    "carry_forward_tvt_baseline": baseline_v4,
    "representation_oracle": oracle_v4,
    "gate_rmse": gate_v4,
    "result": best_v4["metrics"],
    "runtime_minutes": runtime_v4,
    "checkpoint": str(checkpoint_v4),
    "strict_pass": pass_v4,
    "next": (
        "PASS -> capacity is proven; build a separate Fold-1 transfer cell"
        if pass_v4
        else "FAIL -> close the low-dimensional trajectory coefficient head"
    ),
    "elapsed_minutes": (time.time() - started_v4) / 60.0,
}
_json_v4(OUT_DIR_V4 / "verdict.json", verdict_v4)

print("\n24-WELL CONVERGENCE HISTORY")
print(history_v4.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_v4, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_V4, flush=True)
