"""Metric-aligned rescue for a completed heel-conditioned CorrNet v4 run.

Run this in the same kernel immediately after
``tvt_corrnet_heel_conditioned_onefold_v4_cell.py``.  It resumes the frozen
best v4 checkpoint and replaces SmoothL1-dominated training with pooled MSE
plus a top-quartile hard-column term.  This is one bounded rescue experiment,
not an open-ended hyperparameter scan.
"""

from pathlib import Path
import copy
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REQUIRED_GLOBALS = (
    "OUT_DIR",
    "MODEL_DIR",
    "SEED",
    "PREFIX_COLUMNS",
    "EVAL_COLUMNS",
    "STATE_OFFSETS",
    "TARGET_SIGMA_FT",
    "PanelDataset",
    "HeelConditionedCorrNet",
    "train_store",
    "valid_store",
    "input_channels",
    "device",
    "state_offsets_tensor",
    "_evaluate",
)
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the complete v4 cell in this kernel first; missing: " + repr(missing)
    )

RESCUE_DIR = Path(OUT_DIR) / "rmse_finetune_v1"
RESCUE_DIR.mkdir(parents=True, exist_ok=True)
RESCUE_EPOCHS = 12
RESCUE_LR = 1e-4
RESCUE_MIN_LR = 2e-5
RESCUE_BATCH_SIZE = 2
RESCUE_PATIENCE = 4
CE_WEIGHT = 0.20
MSE_SCALE = 64.0
HARD_FRACTION = 0.25
HARD_WEIGHT = 0.15
DIP_WEIGHT = 0.02


def _panel_weighted_mean(values, panel_weight):
    weight = panel_weight.float().clamp_min(1e-6)
    return (values * weight).sum() / weight.sum()


def _metric_loss(
    logits,
    state_valid,
    truth_residual,
    center_tvt,
    column_md,
    column_z,
    eval_mask,
    panel_weight,
    state_offsets,
):
    logits = logits.float().masked_fill(~state_valid, -30.0)
    distance = state_offsets[None, :, None] - truth_residual[:, None, :]
    target = torch.softmax(-0.5 * (distance / TARGET_SIGMA_FT) ** 2, dim=1)
    target = target * state_valid
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    probability = torch.softmax(logits, dim=1)
    prediction = (probability * state_offsets[None, :, None]).sum(dim=1)

    # Prefix columns remain a weak anchor; the official metric is eval-only.
    column_weight = 0.05 + 0.95 * eval_mask
    first_eval = torch.argmax(eval_mask, dim=1)
    column_weight[
        torch.arange(len(column_weight), device=column_weight.device), first_eval
    ] = 2.0
    denominator = column_weight.sum(dim=1).clamp_min(1e-6)

    ce_columns = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    ce = _panel_weighted_mean(
        (ce_columns * column_weight).sum(dim=1) / denominator, panel_weight
    )

    squared = (prediction - truth_residual) ** 2
    mse_by_panel = (squared * column_weight).sum(dim=1) / denominator
    mse = _panel_weighted_mean(mse_by_panel, panel_weight)

    eval_squared = squared[:, PREFIX_COLUMNS : PREFIX_COLUMNS + EVAL_COLUMNS]
    hard_count = max(1, int(round(HARD_FRACTION * EVAL_COLUMNS)))
    hard_by_panel = torch.topk(eval_squared, hard_count, dim=1).values.mean(dim=1)
    hard = _panel_weighted_mean(hard_by_panel, panel_weight)

    predicted_s = center_tvt + prediction + column_z
    true_s = center_tvt + truth_residual + column_z
    dmd = (column_md[:, 1:] - column_md[:, :-1]).clamp_min(1e-3)
    predicted_dip = (predicted_s[:, 1:] - predicted_s[:, :-1]) / dmd
    true_dip = (true_s[:, 1:] - true_s[:, :-1]) / dmd
    transition_weight = torch.maximum(eval_mask[:, 1:], eval_mask[:, :-1])
    dip_by_panel = (
        ((predicted_dip - true_dip) ** 2 * transition_weight).sum(dim=1)
        / transition_weight.sum(dim=1).clamp_min(1e-6)
    )
    dip = _panel_weighted_mean(dip_by_panel, panel_weight)

    total = (
        CE_WEIGHT * ce
        + mse / MSE_SCALE
        + HARD_WEIGHT * hard / MSE_SCALE
        + DIP_WEIGHT * dip
    )
    return total, {
        "ce": ce,
        "mse": mse,
        "hard_mse": hard,
        "dip_mse": dip,
    }


source_checkpoint = Path(MODEL_DIR) / f"seed_{SEED}.pt"
if not source_checkpoint.exists():
    raise FileNotFoundError(source_checkpoint)
checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=False)
rescue_model = HeelConditionedCorrNet(input_channels).to(device)
rescue_model.load_state_dict(checkpoint["model"])
rescue_model.eval()
source_metrics, _, _ = _evaluate(
    rescue_model,
    valid_store,
    "full",
    device,
    state_offsets_tensor,
    return_rows=False,
)
print("SOURCE", source_metrics, flush=True)

optimizer = torch.optim.AdamW(
    rescue_model.parameters(), lr=RESCUE_LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=RESCUE_EPOCHS, eta_min=RESCUE_MIN_LR
)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

dataset = PanelDataset(train_store, "full")
loader = DataLoader(
    dataset,
    batch_size=RESCUE_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
best_rmse = float(source_metrics["rmse"])
best_epoch = 0
best_state = copy.deepcopy(checkpoint["model"])
stale = 0
history = []
started = time.time()

for epoch in range(1, RESCUE_EPOCHS + 1):
    rescue_model.train()
    batch_rows = []
    for batch in loader:
        values = batch["input"].to(device, non_blocking=True)
        state_valid = batch["state_valid"].to(device, non_blocking=True)
        truth_residual = batch["truth_residual"].to(device, non_blocking=True)
        center_tvt = batch["center_tvt"].to(device, non_blocking=True)
        column_md = batch["column_md"].to(device, non_blocking=True)
        column_z = batch["column_z"].to(device, non_blocking=True)
        eval_mask = batch["eval_mask"].to(device, non_blocking=True)
        panel_weight = batch["panel_weight"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = rescue_model(values, state_offsets_tensor)
            loss, components = _metric_loss(
                logits,
                state_valid,
                truth_residual,
                center_tvt,
                column_md,
                column_z,
                eval_mask,
                panel_weight,
                state_offsets_tensor,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(rescue_model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        batch_rows.append(
            {
                "loss": float(loss.detach().cpu()),
                **{
                    name: float(value.detach().cpu())
                    for name, value in components.items()
                },
            }
        )
    scheduler.step()
    rescue_model.eval()
    metrics, by_well, _ = _evaluate(
        rescue_model,
        valid_store,
        "full",
        device,
        state_offsets_tensor,
        return_rows=False,
    )
    row = {
        "epoch": epoch,
        **{
            f"train_{name}": float(np.mean([item[name] for item in batch_rows]))
            for name in batch_rows[0]
        },
        **{f"valid_{name}": value for name, value in metrics.items()},
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    history.append(row)
    print("RMSE FINE-TUNE", row, flush=True)
    if metrics["rmse"] < best_rmse - 1e-4:
        best_rmse = float(metrics["rmse"])
        best_epoch = epoch
        best_state = copy.deepcopy(rescue_model.state_dict())
        stale = 0
        by_well.to_csv(RESCUE_DIR / "best_bywell.csv", index=False)
    else:
        stale += 1
    if epoch >= 4 and stale >= RESCUE_PATIENCE:
        print(f"RMSE fine-tune early stopping at epoch {epoch}", flush=True)
        break

pd.DataFrame(history).to_csv(RESCUE_DIR / "history.csv", index=False)
rescue_model.load_state_dict(best_state)
rescue_model.eval()
final_metrics, final_bywell, final_rows = _evaluate(
    rescue_model,
    valid_store,
    "full",
    device,
    state_offsets_tensor,
    return_rows=True,
)
final_bywell.to_csv(RESCUE_DIR / "final_bywell.csv", index=False)
final_rows.to_pickle(RESCUE_DIR / "fold1_predictions.pkl")
torch.save(
    {
        "model": best_state,
        "epoch": best_epoch,
        "metrics": final_metrics,
        "input_channels": input_channels,
        "seed": SEED,
        "source_checkpoint": str(source_checkpoint),
    },
    RESCUE_DIR / "best.pt",
)

dose_rows = []
truth = final_rows["truth"].to_numpy(float)
center = final_rows["center"].to_numpy(float)
correction = final_rows["prediction"].to_numpy(float) - center
for dose in np.round(np.arange(0.0, 1.251, 0.05), 2):
    prediction = center + dose * correction
    dose_rows.append(
        {
            "dose": float(dose),
            "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
            "mae": float(np.mean(np.abs(prediction - truth))),
        }
    )
dose_report = pd.DataFrame(dose_rows).sort_values("rmse")
dose_report.to_csv(RESCUE_DIR / "dose_scan_diagnostic.csv", index=False)

intervention_rows = []
for mode in ("full", "no_gr", "shuffled_typewell"):
    metrics, _, _ = _evaluate(
        rescue_model,
        valid_store,
        mode,
        device,
        state_offsets_tensor,
        return_rows=False,
    )
    intervention_rows.append({"mode": mode, **metrics})
interventions = pd.DataFrame(intervention_rows)
full_rmse = float(interventions.loc[interventions["mode"] == "full", "rmse"].iloc[0])
interventions["degradation_vs_full"] = interventions["rmse"] - full_rmse
interventions.to_csv(RESCUE_DIR / "interventions.csv", index=False)

no_gr_degradation = float(
    interventions.loc[interventions["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
shuffle_degradation = float(
    interventions.loc[
        interventions["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)
rescue_pass = bool(
    final_metrics["rmse"] <= 9.2
    and no_gr_degradation >= 0.20
    and shuffle_degradation >= 0.20
)
verdict = {
    "experiment": "tvt_corrnet_v4_rmse_finetune_v1",
    "source_rmse": float(source_metrics["rmse"]),
    "best_epoch": best_epoch,
    "final_metrics": final_metrics,
    "best_dose_diagnostic": dose_report.iloc[0].to_dict(),
    "no_gr_degradation": no_gr_degradation,
    "shuffled_typewell_degradation": shuffle_degradation,
    "runtime_minutes": (time.time() - started) / 60.0,
    "strict_pass": rescue_pass,
    "next": (
        "PASS -> inspect tail reliability and then expand seeds/folds"
        if rescue_pass
        else "FAIL -> close CorrNet; the remaining gap is architectural, not optimisation"
    ),
}
with open(RESCUE_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nFINAL METRICS")
print(json.dumps(final_metrics, indent=2))
print("\nDOSE SCAN TOP")
print(dose_report.head(10).to_string(index=False))
print("\nINTERVENTIONS")
print(interventions.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", RESCUE_DIR)

del optimizer, scheduler, loader, dataset
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

