"""One-well E1 loss/optimisation diagnostic for the PCA16 S head.

This is an implementation audit, not another architecture search.  A free
16-vector first proves that the cached target, basis, decoder, and exact path
loss can recover the representation oracle.  The network then receives a
coefficient-only warm-up before exact native-row path fine-tuning.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REQUIRED_DIAG = {
    "pool_e1",
    "target_cache_e1",
    "coefficient_center_e1",
    "coefficient_scale_e1",
    "PCA_DIM_E1",
    "FastTrajectorySHead",
    "_input_fast",
    "_decode_e1",
    "_fixed_metrics_e1",
    "_rmse",
    "_seed_everything",
    "device_e1",
    "OOF_ROOT",
}
missing_diag = sorted(REQUIRED_DIAG.difference(globals()))
if missing_diag:
    raise RuntimeError(
        "Run this in the E1 live kernel; missing " + repr(missing_diag)
    )


EXPERIMENT_DIAG = "deltacorr_steerspline_e1_pca16_loss_diagnostic_v1"
OUT_DIR_DIAG = OOF_ROOT / EXPERIMENT_DIAG
MODEL_DIR_DIAG = OUT_DIR_DIAG / "models"
for directory in (OUT_DIR_DIAG, MODEL_DIR_DIAG):
    directory.mkdir(parents=True, exist_ok=True)

SEED_DIAG = 20260726
FREE_STEPS_DIAG = 1000
WARMUP_STEPS_DIAG = 600
PATH_STEPS_DIAG = 1200
REPORT_EVERY_DIAG = 100
NETWORK_GATE_DIAG = 1.0


def _json_diag(path, payload):
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


record_diag = pool_e1[0]
target_diag = target_cache_e1[record_diag.well]
oracle_diag = _fixed_metrics_e1(
    [record_diag], np.zeros(PCA_DIM_E1), oracle=True
)
baseline_diag = _fixed_metrics_e1(
    [record_diag], coefficient_center_e1, oracle=False
)

offset_diag = torch.from_numpy(target_diag["offset"]).to(device_e1)
design_diag = torch.from_numpy(target_diag["design"]).to(device_e1)
target_delta_s_diag = torch.from_numpy(target_diag["target_delta_s"]).to(device_e1)
target_coefficients_diag = torch.from_numpy(target_diag["coefficients"]).to(device_e1)
center_diag = torch.from_numpy(coefficient_center_e1).to(device_e1)
scale_diag = torch.from_numpy(coefficient_scale_e1).to(device_e1)
target_standard_diag = (target_coefficients_diag - center_diag) / scale_diag


def _metrics_from_standard_diag(standard):
    coefficients = (
        coefficient_center_e1
        + coefficient_scale_e1 * standard.detach().float().cpu().numpy()
    )
    truth, prediction = _decode_e1(record_diag, coefficients)
    return {
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "coefficient_standard_mse": float(
            np.mean(
                (
                    standard.detach().float().cpu().numpy()
                    - target_standard_diag.detach().cpu().numpy()
                )
                ** 2
            )
        ),
    }


started_diag = time.time()
print(
    "E1 PCA16 LOSS DIAGNOSTIC CONFIG",
    {
        "well": record_diag.well,
        "rows": record_diag.eval_rows,
        "baseline": baseline_diag,
        "representation_oracle": oracle_diag,
        "free_steps": FREE_STEPS_DIAG,
        "network_warmup_steps": WARMUP_STEPS_DIAG,
        "network_path_steps": PATH_STEPS_DIAG,
    },
    flush=True,
)

# ---------------------------------------------------------------------------
# A. Free-vector loss audit.
# ---------------------------------------------------------------------------

free_standard_diag = torch.nn.Parameter(
    torch.zeros(PCA_DIM_E1, device=device_e1)
)
free_optimizer_diag = torch.optim.Adam([free_standard_diag], lr=0.05)
free_history_diag = []
for step in range(1, FREE_STEPS_DIAG + 1):
    free_optimizer_diag.zero_grad(set_to_none=True)
    coefficients = center_diag + scale_diag * free_standard_diag
    prediction_delta_s = offset_diag + design_diag @ coefficients
    path_loss = torch.mean(
        ((prediction_delta_s - target_delta_s_diag) / 8.0) ** 2
    )
    coefficient_loss = torch.mean(
        (free_standard_diag - target_standard_diag) ** 2
    )
    loss = path_loss + 0.10 * coefficient_loss
    loss.backward()
    free_optimizer_diag.step()
    if step == 1 or step % REPORT_EVERY_DIAG == 0:
        metrics = _metrics_from_standard_diag(free_standard_diag)
        row = {
            "arm": "free_vector",
            "step": step,
            "loss": float(loss.detach().cpu()),
            "path_loss": float(path_loss.detach().cpu()),
            **metrics,
        }
        free_history_diag.append(row)
        print("E1 FREE-VECTOR", row, flush=True)

free_metrics_diag = _metrics_from_standard_diag(free_standard_diag)
free_pass_diag = bool(
    free_metrics_diag["rmse"] <= oracle_diag["rmse"] + 0.10
)

# ---------------------------------------------------------------------------
# B. Network coefficient warm-up followed by exact path fine-tuning.
# ---------------------------------------------------------------------------

_seed_everything(SEED_DIAG)
model_diag = FastTrajectorySHead(PCA_DIM_E1).to(device_e1)
optimizer_diag = torch.optim.AdamW(
    model_diag.parameters(), lr=8e-4, weight_decay=1e-5
)
total_steps_diag = WARMUP_STEPS_DIAG + PATH_STEPS_DIAG
scheduler_diag = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_diag, T_max=total_steps_diag, eta_min=5e-5
)
scaler_diag = torch.amp.GradScaler(
    "cuda", enabled=device_e1.type == "cuda", init_scale=512.0
)
values_diag = _input_fast(record_diag, device_e1)
network_history_diag = []
best_diag = None

for step in range(1, total_steps_diag + 1):
    model_diag.train()
    optimizer_diag.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device_e1.type == "cuda"):
        standard = model_diag(values_diag)[0]
        coefficient_loss = torch.mean(
            (standard - target_standard_diag) ** 2
        )
        coefficients = center_diag + scale_diag * standard
        prediction_delta_s = offset_diag + design_diag @ coefficients
        path_loss = torch.mean(
            ((prediction_delta_s - target_delta_s_diag) / 8.0) ** 2
        )
        if step <= WARMUP_STEPS_DIAG:
            loss = coefficient_loss
            phase = "coefficient_warmup"
        else:
            slope_loss = F.smooth_l1_loss(
                torch.diff(prediction_delta_s) / 2.0,
                torch.diff(target_delta_s_diag) / 2.0,
            )
            loss = path_loss + 0.25 * coefficient_loss + 0.02 * slope_loss
            phase = "exact_path"
    scaler_diag.scale(loss).backward()
    scaler_diag.unscale_(optimizer_diag)
    torch.nn.utils.clip_grad_norm_(model_diag.parameters(), 10.0)
    scaler_diag.step(optimizer_diag)
    scaler_diag.update()
    scheduler_diag.step()

    if step == 1 or step % REPORT_EVERY_DIAG == 0:
        model_diag.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=device_e1.type == "cuda"
        ):
            evaluation_standard = model_diag(values_diag)[0].float()
        metrics = _metrics_from_standard_diag(evaluation_standard)
        row = {
            "arm": "network",
            "phase": phase,
            "step": step,
            "loss": float(loss.detach().cpu()),
            "path_loss": float(path_loss.detach().cpu()),
            "coefficient_loss": float(coefficient_loss.detach().cpu()),
            **metrics,
            "learning_rate": optimizer_diag.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started_diag) / 60.0,
        }
        network_history_diag.append(row)
        print("E1 NETWORK", row, flush=True)
        if best_diag is None or metrics["rmse"] < best_diag["metrics"]["rmse"]:
            best_diag = {
                "step": step,
                "metrics": metrics,
                "state": {
                    key: value.detach().cpu()
                    for key, value in model_diag.state_dict().items()
                },
            }
        if (
            step >= WARMUP_STEPS_DIAG
            and metrics["rmse"] <= NETWORK_GATE_DIAG
        ):
            break

free_history_frame_diag = pd.DataFrame(free_history_diag)
network_history_frame_diag = pd.DataFrame(network_history_diag)
free_history_frame_diag.to_csv(
    OUT_DIR_DIAG / "free_vector_history.csv", index=False
)
network_history_frame_diag.to_csv(
    OUT_DIR_DIAG / "network_history.csv", index=False
)

checkpoint_diag = MODEL_DIR_DIAG / "best_onewell_loss_repair.pt"
torch.save(
    {
        "experiment": EXPERIMENT_DIAG,
        "well": record_diag.well,
        "step": int(best_diag["step"]),
        "model": best_diag["state"],
        "metrics": best_diag["metrics"],
        "pca_dimensions": PCA_DIM_E1,
        "coefficient_center": coefficient_center_e1,
        "coefficient_scale": coefficient_scale_e1,
    },
    checkpoint_diag,
)

network_pass_diag = bool(best_diag["metrics"]["rmse"] <= NETWORK_GATE_DIAG)
verdict_diag = {
    "experiment": EXPERIMENT_DIAG,
    "stage": "E1_onewell_loss_implementation_audit",
    "well": record_diag.well,
    "baseline": baseline_diag,
    "representation_oracle": oracle_diag,
    "free_vector_result": free_metrics_diag,
    "free_vector_pass": free_pass_diag,
    "network_best_step": int(best_diag["step"]),
    "network_result": best_diag["metrics"],
    "network_gate_rmse_le": NETWORK_GATE_DIAG,
    "network_pass": network_pass_diag,
    "strict_pass": bool(free_pass_diag and network_pass_diag),
    "checkpoint": str(checkpoint_diag),
    "elapsed_minutes": (time.time() - started_diag) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> rerun 1/4/24 capacity with coefficient warm-up and fixed updates"
        if free_pass_diag and network_pass_diag
        else "FAIL -> close deterministic PCA SteerSpline before DeltaCorr"
    ),
}
_json_diag(OUT_DIR_DIAG / "verdict.json", verdict_diag)

print("\nE1 LOSS DIAGNOSTIC VERDICT")
print(json.dumps(verdict_diag, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_DIAG, flush=True)
