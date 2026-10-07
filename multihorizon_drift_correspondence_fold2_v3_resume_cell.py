"""Resume V3 after the FP16 CUDA LU-solver failure.

Run this in the same live kernel that already contains ``core_matcher_v3``,
``core_volumes_v3`` and ``fold2_volumes_v3``.  It patches the global solve to
construct and solve the normal equations in an autocast-disabled FP32 block,
then restarts only the drift model training.  Matcher training and volume
construction are deliberately not repeated.
"""

import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_RESUME_V3 = {
    "core_matcher_v3",
    "core_wells",
    "core_volumes_v3",
    "fold2_volumes_v3",
    "MultiHorizonDriftCorrespondenceV3",
    "_new_training_v3",
    "_train_epoch_v3",
    "_evaluate_v3",
    "_build_volume_bank_v2",
    "_write_json",
    "_json_default",
    "OUT_DIR_V3",
    "MODEL_DIR_V3",
    "EXPERIMENT_V3",
    "MAX_EPOCHS_V3",
    "MIN_EPOCHS_V3",
    "EARLY_STOPPING_V3",
    "FOLD2_RMSE_GATE_V3",
    "FOLD2_FAR_GATE_V3",
    "FOLD2_MEDIAN_GATE_V3",
    "NO_GR_DEGRADATION_GATE_V3",
    "SHUFFLED_TYPEWELL_GATE_V3",
    "HORIZONS_V3",
    "HORIZON_FT_V3",
    "NODE_STEP_FT_V2",
    "mixed_epochs_v1",
    "tune_wells",
    "fold_path",
    "device",
    "OOF_ROOT",
}
missing_resume_v3 = sorted(REQUIRED_RESUME_V3.difference(globals()))
if missing_resume_v3:
    raise RuntimeError(
        "Run this immediately after the failed V3 cell; missing: "
        + repr(missing_resume_v3)
    )

if len(core_volumes_v3) != len(core_wells) or len(fold2_volumes_v3) != len(tune_wells):
    raise RuntimeError(
        f"Incomplete cached volumes: {len(core_volumes_v3)}/{len(fold2_volumes_v3)}"
    )


def _global_solve_fp32_resume_v3(
    absolute_s, absolute_precision, horizon_payload, columns
):
    dtype = absolute_s.dtype
    with torch.amp.autocast("cuda", enabled=False):
        rows = []
        observations = []
        precisions = []
        identity = torch.eye(
            columns, device=absolute_s.device, dtype=torch.float32
        )
        rows.append(identity)
        observations.append(absolute_s.float())
        precisions.append(absolute_precision.float())

        anchor = torch.zeros(
            1, columns, device=absolute_s.device, dtype=torch.float32
        )
        anchor[0, 0] = 1.0
        rows.append(anchor)
        observations.append(
            torch.zeros(1, device=absolute_s.device, dtype=torch.float32)
        )
        precisions.append(
            torch.full(
                (1,), 256.0, device=absolute_s.device, dtype=torch.float32
            )
        )

        for horizon, (delta_s, precision) in horizon_payload.items():
            horizon = int(horizon)
            if columns <= horizon:
                continue
            count = columns - horizon
            edge = torch.zeros(
                count, columns, device=absolute_s.device, dtype=torch.float32
            )
            index = torch.arange(count, device=absolute_s.device)
            edge[index, index] = -1.0
            edge[index, index + horizon] = 1.0
            rows.append(edge)
            observations.append(delta_s[:count].float())
            precisions.append(precision[:count].float())

        design = torch.cat(rows, dim=0)
        observation = torch.cat(observations, dim=0)
        precision = torch.cat(precisions, dim=0).clamp(0.03, 30.0)
        system = design.T @ (precision[:, None] * design)
        system = system + 1e-4 * torch.eye(
            columns, device=system.device, dtype=torch.float32
        )
        rhs = design.T @ (precision * observation)
        solution = torch.linalg.solve(system, rhs)
    solution = solution - solution[0]
    return solution.to(dtype)


MultiHorizonDriftCorrespondenceV3._global_solve = staticmethod(
    _global_solve_fp32_resume_v3
)

for stale_name_resume_v3 in (
    "model_v3",
    "optimizer_v3",
    "scheduler_v3",
    "scaler_v3",
):
    if stale_name_resume_v3 in globals():
        del globals()[stale_name_resume_v3]
gc.collect()
torch.cuda.empty_cache()

resume_started_v3 = time.time()
print(
    "DRIFT-V3 RESUME",
    {
        "core_volumes": len(core_volumes_v3),
        "fold2_volumes": len(fold2_volumes_v3),
        "solver_dtype": "float32 with autocast disabled",
        "rebuild_matcher": False,
        "rebuild_volumes": False,
    },
    flush=True,
)

model_v3, optimizer_v3, scheduler_v3, scaler_v3 = _new_training_v3()
parameter_count_v3 = int(sum(parameter.numel() for parameter in model_v3.parameters()))
history_v3 = []
best_v3 = None
best_state_v3 = None
stale_v3 = 0

for epoch_v3 in range(1, MAX_EPOCHS_V3 + 1):
    train_metrics_v3 = _train_epoch_v3(
        model_v3,
        core_volumes_v3,
        optimizer_v3,
        scaler_v3,
        epoch_v3,
    )
    scheduler_v3.step()
    valid_metrics_v3, _, _ = _evaluate_v3(model_v3, fold2_volumes_v3)
    row_v3 = {
        "epoch": epoch_v3,
        **train_metrics_v3,
        **{f"valid_{key}": value for key, value in valid_metrics_v3.items()},
        "learning_rate": optimizer_v3.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - resume_started_v3) / 60.0,
    }
    history_v3.append(row_v3)
    print("DRIFT-V3 FOLD2 EPOCH", row_v3, flush=True)
    if best_v3 is None or valid_metrics_v3["rmse"] < best_v3["rmse"] - 1e-4:
        best_v3 = {
            "epoch": epoch_v3,
            "rmse": valid_metrics_v3["rmse"],
            "metrics": valid_metrics_v3,
        }
        best_state_v3 = {
            key: value.detach().cpu().clone()
            for key, value in model_v3.state_dict().items()
        }
        stale_v3 = 0
    else:
        stale_v3 += 1
    if epoch_v3 >= MIN_EPOCHS_V3 and stale_v3 >= EARLY_STOPPING_V3:
        break

pd.DataFrame(history_v3).to_csv(
    OUT_DIR_V3 / "fold2_training_history.csv", index=False
)
model_v3.load_state_dict(best_state_v3)
best_metrics_v3, best_bywell_v3, best_payload_v3 = _evaluate_v3(
    model_v3, fold2_volumes_v3
)
best_bywell_v3.to_csv(OUT_DIR_V3 / "fold2_bywell.csv", index=False)
np.savez_compressed(OUT_DIR_V3 / "fold2_predictions.npz", **best_payload_v3)

no_gr_volumes_v3 = _build_volume_bank_v2(
    core_matcher_v3, tune_wells, "v3_fold2_no_gr", mode="no_gr"
)
no_gr_metrics_v3, _, _ = _evaluate_v3(model_v3, no_gr_volumes_v3)
del no_gr_volumes_v3
gc.collect()

shuffle_volumes_v3 = _build_volume_bank_v2(
    core_matcher_v3,
    tune_wells,
    "v3_fold2_shuffled_typewell",
    mode="shuffled_typewell",
)
shuffle_metrics_v3, _, _ = _evaluate_v3(model_v3, shuffle_volumes_v3)
del shuffle_volumes_v3
gc.collect()

no_gr_degradation_v3 = no_gr_metrics_v3["rmse"] - best_metrics_v3["rmse"]
shuffle_degradation_v3 = shuffle_metrics_v3["rmse"] - best_metrics_v3["rmse"]
path_pass_v3 = bool(
    best_metrics_v3["rmse"] <= FOLD2_RMSE_GATE_V3
    and best_metrics_v3["far_gt2000_rmse"] <= FOLD2_FAR_GATE_V3
    and best_metrics_v3["median_well_rmse"] <= FOLD2_MEDIAN_GATE_V3
)
causal_pass_v3 = bool(
    no_gr_degradation_v3 >= NO_GR_DEGRADATION_GATE_V3
    and shuffle_degradation_v3 >= SHUFFLED_TYPEWELL_GATE_V3
)
strict_pass_v3 = bool(path_pass_v3 and causal_pass_v3)

checkpoint_v3 = MODEL_DIR_V3 / "best_fold2_drift_correspondence.pt"
torch.save(
    {
        "model": best_state_v3,
        "epoch": int(best_v3["epoch"]),
        "metrics": best_metrics_v3,
        "horizons": HORIZONS_V3,
        "horizon_ft": HORIZON_FT_V3,
        "node_step_ft": NODE_STEP_FT_V2,
        "matcher_epochs": int(mixed_epochs_v1),
        "matcher_source": "folds345 freshly trained real correspondence v1",
        "prefix_adapter_dose": 0.0,
        "fold_source": str(fold_path),
    },
    checkpoint_v3,
)

controls_v3 = pd.DataFrame(
    [
        {"mode": "full", **best_metrics_v3, "degradation_vs_full": 0.0},
        {
            "mode": "no_gr",
            **no_gr_metrics_v3,
            "degradation_vs_full": no_gr_degradation_v3,
        },
        {
            "mode": "shuffled_typewell",
            **shuffle_metrics_v3,
            "degradation_vs_full": shuffle_degradation_v3,
        },
    ]
)
controls_v3.to_csv(OUT_DIR_V3 / "fold2_controls.csv", index=False)

verdict_v3 = {
    "experiment": EXPERIMENT_V3,
    "stage": "fold2_multihorizon_global_drift_gateway",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "autoregressive": False,
    "train_folds": [3, 4, 5],
    "validation_fold": 2,
    "parameters": parameter_count_v3,
    "selected_epoch": int(best_v3["epoch"]),
    "horizons_ft": HORIZON_FT_V3,
    "fold2": best_metrics_v3,
    "controls": {
        "no_gr": no_gr_metrics_v3,
        "shuffled_typewell": shuffle_metrics_v3,
        "no_gr_degradation": no_gr_degradation_v3,
        "shuffled_typewell_degradation": shuffle_degradation_v3,
    },
    "gates": {
        "rmse_le": FOLD2_RMSE_GATE_V3,
        "far_gt2000_rmse_le": FOLD2_FAR_GATE_V3,
        "median_well_rmse_le": FOLD2_MEDIAN_GATE_V3,
        "each_control_degradation_ge": 0.30,
    },
    "path_pass": path_pass_v3,
    "causal_pass": causal_pass_v3,
    "strict_pass": strict_pass_v3,
    "checkpoint": str(checkpoint_v3),
    "elapsed_minutes": (time.time() - resume_started_v3) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze epoch and build a separate folds2345-to-Fold1 cell"
        if strict_pass_v3
        else "FAIL -> drift target is insufficient; upgrade the local correspondence emission before another path model"
    ),
}
_write_json(OUT_DIR_V3 / "verdict.json", verdict_v3)

print("\nDRIFT-V3 FOLD2 CONTROLS")
print(controls_v3.to_string(index=False), flush=True)
print("\nDRIFT-V3 FINAL VERDICT")
print(json.dumps(verdict_v3, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_V3, flush=True)
