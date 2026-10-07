"""Final four-well E1 implementation audit using joint gradients.

The previous capacity cell stepped the optimiser after every well, allowing
sequential wells to overwrite each other.  Here all four losses are accumulated
before one optimiser step.  The exact-path phase uses a low learning rate and
does not repeat the destructive high-LR phase transition.  The 1-ft gate is
unchanged; this cell never trains 24 wells or Fold 1.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_V3 = {
    "pool_e1",
    "target_cache_e1",
    "coefficient_center_e1",
    "coefficient_scale_e1",
    "PCA_DIM_E1",
    "FastTrajectorySHead",
    "_input_fast",
    "_loss_e1",
    "_evaluate_e1",
    "_fixed_metrics_e1",
    "_seed_everything",
    "device_e1",
    "OOF_ROOT",
}
missing_v3 = sorted(REQUIRED_V3.difference(globals()))
if missing_v3:
    raise RuntimeError(
        "Run this in the E1 live kernel; missing " + repr(missing_v3)
    )


EXPERIMENT_V3 = "deltacorr_steerspline_e1_pca16_fourwell_jointgrad_v3"
OUT_DIR_V3 = OOF_ROOT / EXPERIMENT_V3
MODEL_DIR_V3 = OUT_DIR_V3 / "models"
for directory in (OUT_DIR_V3, MODEL_DIR_V3):
    directory.mkdir(parents=True, exist_ok=True)

SEED_V3 = 20260728
WARMUP_CYCLES_V3 = 600
PATH_CYCLES_V3 = 600
REPORT_EVERY_V3 = 20
GATE_V3 = 1.0


def _json_v3(path, payload):
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


indices_v3 = np.linspace(0, len(pool_e1) - 1, 4).round().astype(int)
records_v3 = [pool_e1[index] for index in indices_v3]
oracle_v3 = _fixed_metrics_e1(
    records_v3, np.zeros(PCA_DIM_E1), oracle=True
)
baseline_v3 = _fixed_metrics_e1(
    records_v3, coefficient_center_e1, oracle=False
)
if oracle_v3["rmse"] >= GATE_V3:
    raise RuntimeError(
        f"PCA16 oracle {oracle_v3['rmse']:.6f} makes the 1-ft gate impossible"
    )


def _target_standard_v3(record):
    coefficients = torch.from_numpy(
        target_cache_e1[record.well]["coefficients"]
    ).to(device_e1)
    center = torch.from_numpy(coefficient_center_e1).to(device_e1)
    scale = torch.from_numpy(coefficient_scale_e1).to(device_e1)
    return (coefficients - center) / scale


def _joint_phase_v3(
    model,
    phase,
    cycles,
    learning_rate,
    minimum_learning_rate,
    gradient_clip,
    history,
    best,
    started,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cycles, eta_min=minimum_learning_rate
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device_e1.type == "cuda", init_scale=512.0
    )
    mean_rows = float(np.mean([record.eval_rows for record in records_v3]))
    rng = np.random.default_rng(SEED_V3 + (phase == "exact_path"))

    for cycle in range(1, cycles + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = coefficient_sum = path_sum = 0.0
        before = float(scaler.get_scale())
        for index in rng.permutation(len(records_v3)):
            record = records_v3[int(index)]
            values = _input_fast(record, device_e1)
            with torch.amp.autocast("cuda", enabled=device_e1.type == "cuda"):
                standard = model(values)
                target_standard = _target_standard_v3(record)
                coefficient_loss = torch.mean(
                    (standard[0] - target_standard) ** 2
                )
                if phase == "coefficient_warmup":
                    loss = coefficient_loss / len(records_v3)
                    path_loss = torch.zeros((), device=device_e1)
                else:
                    path_objective, path_loss, _ = _loss_e1(
                        standard,
                        record,
                        device_e1,
                        record.eval_rows / max(mean_rows, 1.0),
                    )
                    loss = path_objective / len(records_v3)
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach().cpu()) * len(records_v3)
            coefficient_sum += float(coefficient_loss.detach().cpu())
            path_sum += float(path_loss.detach().cpu())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if cycle != 1 and cycle % REPORT_EVERY_V3 != 0 and cycle != cycles:
            continue
        metrics, bywell = _evaluate_e1(model, records_v3, device_e1)
        row = {
            "phase": phase,
            "cycle": cycle,
            "optimizer_steps": cycle,
            "well_forwards": cycle * len(records_v3),
            "loss": loss_sum / len(records_v3),
            "coefficient_loss": coefficient_sum / len(records_v3),
            "path_mse_scaled": path_sum / len(records_v3),
            **metrics,
            "scale_decreased": int(float(scaler.get_scale()) < before),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("E1 PCA16 FOUR-WELL JOINT", row, flush=True)
        if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
            best = {
                "phase": phase,
                "cycle": cycle,
                "metrics": metrics,
                "bywell": bywell.copy(),
                "state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }
        if cycle >= 40 and metrics["rmse"] <= GATE_V3:
            break
    return model, history, best


started_v3 = time.time()
_seed_everything(SEED_V3)
model_v3 = FastTrajectorySHead(PCA_DIM_E1).to(device_e1)
parameters_v3 = int(sum(parameter.numel() for parameter in model_v3.parameters()))
history_v3, best_v3 = [], None

print(
    "E1 PCA16 FOUR-WELL JOINT CONFIG",
    {
        "wells": [record.well for record in records_v3],
        "parameters": parameters_v3,
        "baseline": baseline_v3,
        "representation_oracle": oracle_v3,
        "gate_rmse": GATE_V3,
        "warmup_cycles": WARMUP_CYCLES_V3,
        "path_cycles": PATH_CYCLES_V3,
    },
    flush=True,
)

model_v3, history_v3, best_v3 = _joint_phase_v3(
    model_v3,
    "coefficient_warmup",
    WARMUP_CYCLES_V3,
    8e-4,
    1.5e-4,
    10.0,
    history_v3,
    best_v3,
    started_v3,
)

# Preserve the warm-up optimum before a low-LR path polish.  If path polish
# degrades, final evaluation returns to the best saved state.
model_v3.load_state_dict(best_v3["state"])
model_v3, history_v3, best_v3 = _joint_phase_v3(
    model_v3,
    "exact_path",
    PATH_CYCLES_V3,
    5e-5,
    1e-5,
    2.0,
    history_v3,
    best_v3,
    started_v3,
)
model_v3.load_state_dict(best_v3["state"])

history_frame_v3 = pd.DataFrame(history_v3)
history_frame_v3.to_csv(OUT_DIR_V3 / "history.csv", index=False)
best_v3["bywell"].to_csv(OUT_DIR_V3 / "bywell.csv", index=False)
checkpoint_v3 = MODEL_DIR_V3 / "best_fourwell_joint.pt"
torch.save(
    {
        "experiment": EXPERIMENT_V3,
        "phase": best_v3["phase"],
        "cycle": int(best_v3["cycle"]),
        "model": best_v3["state"],
        "metrics": best_v3["metrics"],
        "pca_dimensions": PCA_DIM_E1,
        "coefficient_center": coefficient_center_e1,
        "coefficient_scale": coefficient_scale_e1,
    },
    checkpoint_v3,
)

pass_v3 = bool(best_v3["metrics"]["rmse"] <= GATE_V3)
verdict_v3 = {
    "experiment": EXPERIMENT_V3,
    "stage": "E1_fourwell_joint_gradient_audit",
    "parameters": parameters_v3,
    "baseline": baseline_v3,
    "representation_oracle": oracle_v3,
    "gate_rmse_le": GATE_V3,
    "best_phase": best_v3["phase"],
    "best_cycle": int(best_v3["cycle"]),
    "result": best_v3["metrics"],
    "strict_pass": pass_v3,
    "checkpoint": str(checkpoint_v3),
    "elapsed_minutes": (time.time() - started_v3) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> run one frozen 24-well joint-gradient capacity gate"
        if pass_v3
        else "FAIL -> close deterministic PCA SteerSpline before DeltaCorr"
    ),
}
_json_v3(OUT_DIR_V3 / "verdict.json", verdict_v3)

print("\nE1 FOUR-WELL JOINT HISTORY")
print(history_frame_v3.to_string(index=False), flush=True)
print("\nE1 FOUR-WELL JOINT FINAL VERDICT")
print(json.dumps(verdict_v3, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_V3, flush=True)
