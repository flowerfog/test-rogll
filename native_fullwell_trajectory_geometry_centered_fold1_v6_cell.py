"""Held-out Fold-1 gateway for the geometry-centred trajectory head.

Prerequisite: run geometry-centred v3 and the v5 24-well capacity audit in the
same kernel.  The model is trained from scratch on all 618 outer-train wells;
the capacity checkpoint is deliberately not loaded.  Validation uses only the
155 untouched official-mask wells.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_V6 = {
    "FastTrajectorySHead",
    "_train_fast",
    "_evaluate_fast",
    "_seed_everything",
    "coefficient_mean_fast",
    "coefficient_std_fast",
    "coefficient_center_fast",
    "train_records",
    "valid_records",
    "train_wells",
    "representation_oracle_metrics_fast",
    "zero_s_metrics_fast",
    "population_mean_metrics_fast",
    "device_fast",
    "OOF_ROOT",
}
missing_v6 = sorted(REQUIRED_V6.difference(globals()))
if missing_v6:
    raise RuntimeError(
        "Run this in the live kernel after geometry-centred v3/v5; missing "
        + repr(missing_v6)
    )


EXPERIMENT_V6 = "native_fullwell_trajectory_geometry_centered_fold1_v6"
OUT_DIR_V6 = OOF_ROOT / EXPERIMENT_V6
MODEL_DIR_V6 = OUT_DIR_V6 / "models"
for directory in (OUT_DIR_V6, MODEL_DIR_V6):
    directory.mkdir(parents=True, exist_ok=True)

SEED_V6 = 20260724
EPOCHS_V6 = 36
REPORT_EVERY_V6 = 2
ARCHITECTURE_GATE_FT_V6 = 7.0
TARGET_GATE_FT_V6 = 5.5
CAUSAL_DEGRADATION_GATE_FT_V6 = 0.25


def _json_v6(path, payload):
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


started_v6 = time.time()
_seed_everything(SEED_V6)

# A fixed train-only diagnostic subset distinguishes optimisation failure from
# held-out transfer failure.  It never participates in early-stopping choice.
diagnostic_records_v6 = [
    train_records[index]
    for index in np.linspace(0, len(train_records) - 1, 24).round().astype(int)
]

model_v6 = FastTrajectorySHead(len(coefficient_mean_fast)).to(device_fast)
parameters_v6 = int(sum(parameter.numel() for parameter in model_v6.parameters()))
print(
    "GEOMETRY-CENTRED FOLD1 CONFIG",
    {
        "seed": SEED_V6,
        "train_wells": len(train_records),
        "valid_wells": len(valid_records),
        "parameters": parameters_v6,
        "epochs": EPOCHS_V6,
        "representation_oracle": representation_oracle_metrics_fast,
        "carry_forward_tvt_baseline": zero_s_metrics_fast,
        "capacity_checkpoint_loaded": False,
    },
    flush=True,
)

model_v6, best_v6, history_v6, runtime_v6 = _train_fast(
    model_v6,
    train_records,
    valid_records,
    device_fast,
    EPOCHS_V6,
    REPORT_EVERY_V6,
    "fold1",
    diagnostic_records=diagnostic_records_v6,
)
history_v6.to_csv(OUT_DIR_V6 / "fold1_history.csv", index=False)

full_metrics_v6, bywell_v6, payload_v6 = _evaluate_fast(
    model_v6, valid_records, device_fast, mode="full", save_rows=True
)
bywell_v6.to_csv(OUT_DIR_V6 / "fold1_bywell.csv", index=False)
np.savez_compressed(OUT_DIR_V6 / "fold1_oof.npz", **payload_v6)

control_rows_v6 = []
for mode in ("full", "no_gr", "no_postps_xyz"):
    metrics, _, _ = _evaluate_fast(
        model_v6, valid_records, device_fast, mode=mode, save_rows=False
    )
    control_rows_v6.append(
        {
            "mode": mode,
            **metrics,
            "degradation_vs_full": metrics["rmse"] - full_metrics_v6["rmse"],
        }
    )
controls_v6 = pd.DataFrame(control_rows_v6)
controls_v6.to_csv(OUT_DIR_V6 / "causal_controls.csv", index=False)

no_gr_degradation_v6 = float(
    controls_v6.loc[
        controls_v6["mode"] == "no_gr", "degradation_vs_full"
    ].iloc[0]
)
no_xyz_degradation_v6 = float(
    controls_v6.loc[
        controls_v6["mode"] == "no_postps_xyz", "degradation_vs_full"
    ].iloc[0]
)

checkpoint_v6 = MODEL_DIR_V6 / "best_fold1.pt"
torch.save(
    {
        "experiment": EXPERIMENT_V6,
        "seed": SEED_V6,
        "epoch": int(best_v6["epoch"]),
        "model": best_v6["state"],
        "metrics": full_metrics_v6,
        "free_dimensions": len(coefficient_mean_fast),
        "coefficient_mean": coefficient_mean_fast,
        "coefficient_std": coefficient_std_fast,
        "coefficient_center": coefficient_center_fast,
        "representation_oracle": representation_oracle_metrics_fast,
    },
    checkpoint_v6,
)

architecture_pass_v6 = bool(full_metrics_v6["rmse"] <= ARCHITECTURE_GATE_FT_V6)
target_pass_v6 = bool(full_metrics_v6["rmse"] <= TARGET_GATE_FT_V6)
causal_pass_v6 = bool(
    no_gr_degradation_v6 >= CAUSAL_DEGRADATION_GATE_FT_V6
    and no_xyz_degradation_v6 >= CAUSAL_DEGRADATION_GATE_FT_V6
)

verdict_v6 = {
    "experiment": EXPERIMENT_V6,
    "stage": "heldout_fold1_official_mask",
    "sample_unit": "complete_well",
    "standalone": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "capacity_checkpoint_loaded": False,
    "seed": SEED_V6,
    "train_wells": len(train_records),
    "valid_wells": len(valid_records),
    "parameters": parameters_v6,
    "epochs_budget": EPOCHS_V6,
    "best_epoch": int(best_v6["epoch"]),
    "representation_oracle": representation_oracle_metrics_fast,
    "carry_forward_tvt_baseline": zero_s_metrics_fast,
    "population_mean_baseline": population_mean_metrics_fast,
    "result": full_metrics_v6,
    "causal_controls": {
        "no_gr_degradation": no_gr_degradation_v6,
        "no_postps_xyz_degradation": no_xyz_degradation_v6,
    },
    "gates": {
        "architecture_rmse_le": ARCHITECTURE_GATE_FT_V6,
        "target_rmse_le": TARGET_GATE_FT_V6,
        "each_causal_degradation_ge": CAUSAL_DEGRADATION_GATE_FT_V6,
    },
    "architecture_pass": architecture_pass_v6,
    "target_5p5_pass": target_pass_v6,
    "causal_pass": causal_pass_v6,
    "strict_pass": bool(architecture_pass_v6 and causal_pass_v6),
    "runtime_minutes": runtime_v6,
    "elapsed_minutes": (time.time() - started_v6) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "checkpoint": str(checkpoint_v6),
    "next": (
        "PASS -> add pseudo-PS episodes without changing this architecture"
        if architecture_pass_v6 and causal_pass_v6
        else "FAIL -> capacity does not transfer; close this trajectory-head architecture"
    ),
}
_json_v6(OUT_DIR_V6 / "verdict.json", verdict_v6)

print("\nFOLD1 HISTORY")
print(history_v6.to_string(index=False), flush=True)
print("\nCAUSAL CONTROLS")
print(controls_v6.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_v6, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_V6, flush=True)
