"""Zero-S anchored capacity repair for the failed trajectory S-head v0.

Run in the same kernel after interrupting S-head v0.  This cell fixes the two
observed failures before spending another full fold run:

* outer-train S coefficients are scale-standardised, but the model predicts
  residuals around the physical zero-S trajectory.  Cross-well coefficient
  means are not a valid trajectory centre because signed well responses cancel
  in coefficient space;
* only one local convolution operates at native 1-ft resolution, followed
  immediately by stride-4/8 tokenisation.  All expensive TCN/attention work is
  performed on tokens.

The cell first runs a 24-well overfit gate.  Fold-1 training starts only if the
fast model reaches representation-oracle + 0.75 ft on those 24 wells.
"""

from pathlib import Path
import gc
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REQUIRED_FAST = {
    "records",
    "train_wells",
    "valid_wells",
    "train_records",
    "valid_records",
    "z_arrays_s",
    "selected_controls_s",
    "selected_oracle_s",
    "_target_s_s",
    "_rmse",
    "_seed_everything",
    "OOF_ROOT",
}
missing_fast = sorted(REQUIRED_FAST.difference(globals()))
if missing_fast:
    raise RuntimeError(
        "Interrupt native_fullwell_trajectory_shead_onefold_v0_cell.py only "
        f"after its representation audit; missing {missing_fast}"
    )


EXPERIMENT_FAST = "native_fullwell_trajectory_shead_zero_anchor_v2"
OUT_DIR_FAST = OOF_ROOT / EXPERIMENT_FAST
MODEL_DIR_FAST = OUT_DIR_FAST / "models"
SEED_FAST = 20260722

CAPACITY_WELLS_FAST = 24
CAPACITY_EPOCHS_FAST = 180
CAPACITY_REPORT_EVERY_FAST = 10
CAPACITY_MARGIN_FT_FAST = 0.75

FOLD_EPOCHS_FAST = 28
FOLD_REPORT_EVERY_FAST = 2
FOLD_MIN_EPOCHS_FAST = 10
FOLD_EARLY_REPORTS_FAST = 5
LEARNING_RATE_FAST = 3e-4
MIN_LEARNING_RATE_FAST = 3e-5
WEIGHT_DECAY_FAST = 2e-5
GRADIENT_CLIP_FAST = 3.0

LOSS_PATH_FAST = 1.0
LOSS_STANDARDISED_COEF_FAST = 0.08
LOSS_SLOPE_FAST = 0.025
LOSS_CURVATURE_FAST = 0.008
PATH_SCALE_FAST = 8.0

ARCHITECTURE_GATE_FAST = 10.0
TARGET_GATE_FAST = 7.0
FAIL_FAST_EPOCH_FAST = 12
FAIL_FAST_VALID_FAST = 12.0
FAIL_FAST_TRAIN_FAST = 4.0

for directory in (OUT_DIR_FAST, MODEL_DIR_FAST):
    directory.mkdir(parents=True, exist_ok=True)


def _json_fast(path, payload):
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


# ---------------------------------------------------------------------------
# Fold-clean coefficient standardisation and cheap baselines.
# ---------------------------------------------------------------------------


def _make_targets_fast(all_records):
    cache = {}
    for record in all_records:
        _, basis, delta_s, coefficients, oracle = _target_s_s(
            record,
            z_arrays_s[record.well],
            selected_controls_s,
        )
        cache[record.well] = {
            "basis": basis.astype(np.float32),
            "delta_s": delta_s.astype(np.float32),
            "coefficients": coefficients.astype(np.float32),
            "oracle": oracle.astype(np.float32),
        }
    return cache


target_cache_fast = _make_targets_fast(train_records + valid_records)
train_coefficient_matrix_fast = np.stack(
    [target_cache_fast[well]["coefficients"] for well in train_wells]
).astype(np.float64)
coefficient_mean_fast = train_coefficient_matrix_fast.mean(axis=0).astype(np.float32)
coefficient_std_fast = train_coefficient_matrix_fast.std(axis=0).astype(np.float32)
coefficient_std_fast = np.maximum(coefficient_std_fast, 1.0).astype(np.float32)
coefficient_center_fast = np.zeros_like(coefficient_mean_fast)

coefficient_audit_fast = pd.DataFrame(
    {
        "coefficient": np.arange(len(coefficient_mean_fast)),
        "mean": coefficient_mean_fast,
        "std": coefficient_std_fast,
        "min": train_coefficient_matrix_fast.min(axis=0),
        "max": train_coefficient_matrix_fast.max(axis=0),
    }
)
coefficient_audit_fast.to_csv(OUT_DIR_FAST / "coefficient_audit.csv", index=False)


def _decode_coefficients_fast(record, coefficients):
    target = target_cache_fast[record.well]
    delta_s = target["basis"] @ np.asarray(coefficients, np.float32)
    z = z_arrays_s[record.well]
    z_delta = z[record.eval_index].astype(np.float64) - float(z[record.anchor_index])
    prediction = record.anchor_tvt + delta_s.astype(np.float64) - z_delta
    truth = record.truth[record.eval_index].astype(np.float64)
    return truth, prediction


def _fixed_coefficient_metrics_fast(evaluation_records, coefficients):
    truths, predictions = [], []
    bywell = []
    for record in evaluation_records:
        truth, prediction = _decode_coefficients_fast(record, coefficients)
        truths.append(truth)
        predictions.append(prediction)
        bywell.append(_rmse(truth, prediction))
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    return {
        "rmse": _rmse(truth, prediction),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }


population_mean_metrics_fast = _fixed_coefficient_metrics_fast(
    valid_records, coefficient_mean_fast
)
zero_s_metrics_fast = _fixed_coefficient_metrics_fast(
    valid_records, np.zeros_like(coefficient_mean_fast)
)
print(
    "FAST S TARGET AUDIT",
    {
        "controls": selected_controls_s,
        "free_dimensions": len(coefficient_mean_fast),
        "representation_oracle_rmse": selected_oracle_s,
        "coefficient_abs_mean_max": float(np.max(np.abs(coefficient_mean_fast))),
        "coefficient_std_max": float(np.max(coefficient_std_fast)),
        "zero_s_baseline": zero_s_metrics_fast,
        "outer_train_mean_baseline": population_mean_metrics_fast,
    },
    flush=True,
)


# ---------------------------------------------------------------------------
# Fast native local stem and token-space full-well control decoder.
# ---------------------------------------------------------------------------


class FastResidual1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class FastControlAttention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.q_norm = nn.LayerNorm(channels)
        self.m_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, 8, dropout=0.03, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, query, memory):
        normalised_memory = self.m_norm(memory)
        update, _ = self.attention(
            self.q_norm(query),
            normalised_memory,
            normalised_memory,
            need_weights=False,
        )
        query = query + update
        return query + self.ff(self.ff_norm(query))


class FastTrajectorySHead(nn.Module):
    def __init__(self, free_dimensions):
        super().__init__()
        self.free_dimensions = int(free_dimensions)
        # One native-resolution local convolution, then immediate tokenisation.
        self.local = nn.Sequential(
            nn.Conv1d(14, 64, 15, padding=7),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.tokenise = nn.Sequential(
            nn.Conv1d(64, 96, 11, stride=4, padding=5),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            FastResidual1D(96, 1),
            FastResidual1D(96, 2),
            FastResidual1D(96, 4),
            nn.Conv1d(96, 128, 7, stride=2, padding=3),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            FastResidual1D(128, 1),
            FastResidual1D(128, 2),
            FastResidual1D(128, 4),
            FastResidual1D(128, 8),
            FastResidual1D(128, 16),
            FastResidual1D(128, 32),
            FastResidual1D(128, 64),
        )
        self.query = nn.Parameter(
            torch.randn(1, self.free_dimensions, 128) * 0.02
        )
        positions = torch.linspace(0.0, 1.0, self.free_dimensions)
        position_features = torch.stack(
            [
                positions,
                positions**2,
                torch.sin(math.pi * positions),
                torch.cos(math.pi * positions),
            ],
            dim=1,
        )
        self.register_buffer("position_features", position_features, persistent=False)
        self.position_project = nn.Linear(4, 128)
        self.attention = nn.ModuleList([FastControlAttention() for _ in range(2)])
        self.output = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Start exactly from the physical zero-S anchored trajectory.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, values):
        memory = self.tokenise(self.local(values)).transpose(1, 2)
        query = self.query.expand(values.shape[0], -1, -1)
        query = query + self.position_project(self.position_features)[None]
        for layer in self.attention:
            query = layer(query, memory)
        return self.output(query).squeeze(-1)


def _input_fast(record, device, mode="full"):
    horizontal = record.horizontal.copy()
    condition = record.condition.copy()
    if mode == "no_gr":
        horizontal[:] = 0.0
        condition[9] = 0.0
    elif mode == "no_postps_xyz":
        condition[4:9, record.anchor_index + 1 :] = 0.0
    values = np.concatenate([horizontal, condition], axis=0).astype(np.float32)
    return torch.from_numpy(values)[None].to(device)


def _standard_target_fast(record):
    coefficients = target_cache_fast[record.well]["coefficients"]
    return (coefficients - coefficient_center_fast) / coefficient_std_fast


def _loss_fast(standardised_prediction, record, device, row_weight):
    target = target_cache_fast[record.well]
    basis = torch.from_numpy(target["basis"]).to(device)
    delta_s = torch.from_numpy(target["delta_s"]).to(device)
    center = torch.from_numpy(coefficient_center_fast).to(device)
    std = torch.from_numpy(coefficient_std_fast).to(device)
    target_standard = torch.from_numpy(_standard_target_fast(record)).to(device)
    prediction_standard = standardised_prediction[0]
    coefficients = center + std * prediction_standard
    path = basis @ coefficients
    path_mse = torch.mean(((path - delta_s) / PATH_SCALE_FAST) ** 2)
    coefficient_loss = torch.mean((prediction_standard - target_standard) ** 2)
    slope = F.smooth_l1_loss(
        torch.diff(path) / 2.0,
        torch.diff(delta_s) / 2.0,
    )
    curvature = F.smooth_l1_loss(
        torch.diff(path, n=2) / 2.0,
        torch.diff(delta_s, n=2) / 2.0,
    )
    loss = row_weight * (
        LOSS_PATH_FAST * path_mse
        + LOSS_STANDARDISED_COEF_FAST * coefficient_loss
        + LOSS_SLOPE_FAST * slope
        + LOSS_CURVATURE_FAST * curvature
    )
    return loss, path_mse.detach(), coefficient_loss.detach()


@torch.no_grad()
def _predict_fast(model, record, device, mode="full"):
    model.eval()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        standard = model(_input_fast(record, device, mode=mode))[0].float().cpu().numpy()
    coefficients = coefficient_center_fast + coefficient_std_fast * standard
    return _decode_coefficients_fast(record, coefficients)


@torch.no_grad()
def _evaluate_fast(model, evaluation_records, device, mode="full", save_rows=False):
    truths, predictions, labels, bywell = [], [], [], []
    for record in evaluation_records:
        truth, prediction = _predict_fast(model, record, device, mode=mode)
        truths.append(truth)
        predictions.append(prediction)
        if save_rows:
            labels.append(np.full(len(truth), record.well, dtype="U16"))
        bywell.append(
            {
                "well": record.well,
                "rows": len(truth),
                "rmse": _rmse(truth, prediction),
                "mae": float(np.mean(np.abs(prediction - truth))),
                "mean_error": float(np.mean(prediction - truth)),
            }
        )
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": len(truth),
        "wells": len(evaluation_records),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "wells_above_12ft": int((bywell["rmse"] > 12).sum()),
    }
    payload = None
    if save_rows:
        payload = {
            "well": np.concatenate(labels),
            "truth": truth.astype(np.float32),
            "prediction": prediction.astype(np.float32),
        }
    return metrics, bywell, payload


def _select_capacity_fast():
    # Deterministic spread across eval length and GR missingness, using only
    # legal input metadata.
    rows = []
    for well in train_wells:
        record = records[well]
        rows.append(
            {
                "well": well,
                "rows": len(record.md),
                "eval_rows": record.eval_rows,
                "gr_missing": float(1.0 - record.horizontal[1].mean()),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["gr_missing", "eval_rows"])
    indices = np.linspace(0, len(frame) - 1, CAPACITY_WELLS_FAST).round().astype(int)
    return [records[well] for well in frame.iloc[indices]["well"]]


def _train_fast(
    model,
    training_records,
    evaluation_records,
    device,
    epochs,
    report_every,
    phase,
    diagnostic_records=None,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_FAST, weight_decay=WEIGHT_DECAY_FAST
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE_FAST
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=512.0, growth_interval=500
    )
    mean_rows = float(np.mean([record.eval_rows for record in training_records]))
    rng = np.random.default_rng(SEED_FAST + len(training_records))
    history, best = [], None
    reports_without_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"loss": 0.0, "path": 0.0, "coefficient": 0.0, "skipped": 0}
        for batch_number, index in enumerate(rng.permutation(len(training_records)), 1):
            record = training_records[int(index)]
            values = _input_fast(record, device)
            row_weight = record.eval_rows / max(mean_rows, 1.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                standard = model(values)
                loss, path_loss, coefficient_loss = _loss_fast(
                    standard, record, device, row_weight
                )
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_FAST)
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < before:
                sums["skipped"] += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["path"] += float(path_loss.cpu())
            sums["coefficient"] += float(coefficient_loss.cpu())
            del values, standard, loss
            if len(training_records) > 100 and batch_number % 100 == 0:
                print(
                    f"FAST {phase} epoch {epoch}/{epochs} wells {batch_number}/{len(training_records)}",
                    flush=True,
                )
        scheduler.step()
        if epoch != 1 and epoch % report_every != 0 and epoch != epochs:
            continue
        metrics, bywell, _ = _evaluate_fast(model, evaluation_records, device)
        diagnostic_rmse = np.nan
        if diagnostic_records is not None:
            diagnostic_metrics, _, _ = _evaluate_fast(
                model, diagnostic_records, device
            )
            diagnostic_rmse = diagnostic_metrics["rmse"]
        row = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": sums["loss"] / len(training_records),
            "train_path_mse_scaled": sums["path"] / len(training_records),
            "train_coefficient_loss": sums["coefficient"] / len(training_records),
            "train_subset_rmse": diagnostic_rmse,
            **metrics,
            "skipped_steps": sums["skipped"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("FAST S-HEAD EPOCH", row, flush=True)
        if best is None or metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
            best = {
                "epoch": epoch,
                "metrics": metrics,
                "bywell": bywell.copy(),
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            reports_without_improvement = 0
        else:
            reports_without_improvement += 1
        if phase == "capacity":
            capacity_gate = selected_oracle_s + CAPACITY_MARGIN_FT_FAST
            if epoch >= 60 and metrics["rmse"] <= capacity_gate:
                print("FAST capacity gate cleared", flush=True)
                break
            if (
                epoch >= 80
                and best["metrics"]["rmse"]
                >= capacity_zero_baseline_fast["rmse"] * 0.95
            ):
                print(
                    "FAST capacity fail-fast: no material gain over zero-S baseline",
                    flush=True,
                )
                break
        elif epoch >= FOLD_MIN_EPOCHS_FAST and reports_without_improvement >= FOLD_EARLY_REPORTS_FAST:
            print(f"FAST fold early stopping at epoch {epoch}", flush=True)
            break
        if (
            phase == "fold1"
            and epoch >= FAIL_FAST_EPOCH_FAST
            and metrics["rmse"] > FAIL_FAST_VALID_FAST
            and diagnostic_rmse <= FAIL_FAST_TRAIN_FAST
        ):
            print(
                "FAST fold fail-fast: training fit exists but held-out transfer is absent",
                flush=True,
            )
            break
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


# ---------------------------------------------------------------------------
# Execute the repair ladder.
# ---------------------------------------------------------------------------


STARTED_FAST = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("Fast S-head repair requires CUDA")
device_fast = torch.device("cuda")
_seed_everything(SEED_FAST)

capacity_records_fast = _select_capacity_fast()
capacity_zero_baseline_fast = _fixed_coefficient_metrics_fast(
    capacity_records_fast, coefficient_center_fast
)
capacity_model_fast = FastTrajectorySHead(len(coefficient_mean_fast)).to(device_fast)
with torch.no_grad(), torch.amp.autocast("cuda"):
    smoke_output_fast = capacity_model_fast(_input_fast(capacity_records_fast[0], device_fast))
parameters_fast = int(sum(parameter.numel() for parameter in capacity_model_fast.parameters()))
print(
    "FAST MODEL SMOKE",
    {
        "parameters": parameters_fast,
        "output": list(smoke_output_fast.shape),
        "finite": bool(torch.isfinite(smoke_output_fast).all()),
        "native_full_resolution_blocks": 1,
        "token_stride": 8,
        "capacity_zero_s_baseline": capacity_zero_baseline_fast,
    },
    flush=True,
)
del smoke_output_fast

capacity_model_fast, capacity_best_fast, capacity_history_fast, capacity_minutes_fast = _train_fast(
    capacity_model_fast,
    capacity_records_fast,
    capacity_records_fast,
    device_fast,
    CAPACITY_EPOCHS_FAST,
    CAPACITY_REPORT_EVERY_FAST,
    "capacity",
)
capacity_history_fast.to_csv(OUT_DIR_FAST / "capacity_history.csv", index=False)
capacity_gate_fast = selected_oracle_s + CAPACITY_MARGIN_FT_FAST
capacity_pass_fast = bool(capacity_best_fast["metrics"]["rmse"] <= capacity_gate_fast)

if not capacity_pass_fast:
    verdict_fast = {
        "experiment": EXPERIMENT_FAST,
        "stage": "capacity",
        "parameters": parameters_fast,
        "representation_oracle_rmse": selected_oracle_s,
        "capacity_gate_rmse": capacity_gate_fast,
        "capacity_result": capacity_best_fast["metrics"],
        "capacity_zero_s_baseline": capacity_zero_baseline_fast,
        "zero_s_baseline": zero_s_metrics_fast,
        "population_mean_baseline": population_mean_metrics_fast,
        "strict_pass": False,
        "next": "FAIL -> stop S-head optimisation; do not run 618-well fold",
    }
    _json_fast(OUT_DIR_FAST / "verdict.json", verdict_fast)
    print("FAST FINAL VERDICT", json.dumps(verdict_fast, indent=2), flush=True)
    raise RuntimeError(json.dumps(verdict_fast, indent=2))

del capacity_model_fast
gc.collect()
torch.cuda.empty_cache()

fold_model_fast = FastTrajectorySHead(len(coefficient_mean_fast)).to(device_fast)
fold_train_subset_fast = [
    records[well]
    for well in train_wells[:: max(1, len(train_wells) // 24)][:24]
]
fold_model_fast, fold_best_fast, fold_history_fast, fold_minutes_fast = _train_fast(
    fold_model_fast,
    train_records,
    valid_records,
    device_fast,
    FOLD_EPOCHS_FAST,
    FOLD_REPORT_EVERY_FAST,
    "fold1",
    diagnostic_records=fold_train_subset_fast,
)
fold_history_fast.to_csv(OUT_DIR_FAST / "fold1_history.csv", index=False)

full_metrics_fast, bywell_fast, payload_fast = _evaluate_fast(
    fold_model_fast, valid_records, device_fast, save_rows=True
)
bywell_fast.to_csv(OUT_DIR_FAST / "fold1_bywell.csv", index=False)
np.savez_compressed(OUT_DIR_FAST / "fold1_oof.npz", **payload_fast)

controls_rows_fast = []
for mode in ("full", "no_gr", "no_postps_xyz"):
    metrics, _, _ = _evaluate_fast(fold_model_fast, valid_records, device_fast, mode=mode)
    controls_rows_fast.append(
        {
            "mode": mode,
            **metrics,
            "degradation_vs_full": metrics["rmse"] - full_metrics_fast["rmse"],
        }
    )
controls_fast = pd.DataFrame(controls_rows_fast)
controls_fast.to_csv(OUT_DIR_FAST / "causal_controls.csv", index=False)

no_gr_degradation_fast = float(
    controls_fast.loc[controls_fast["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
trajectory_degradation_fast = float(
    controls_fast.loc[
        controls_fast["mode"] == "no_postps_xyz", "degradation_vs_full"
    ].iloc[0]
)
architecture_pass_fast = bool(full_metrics_fast["rmse"] <= ARCHITECTURE_GATE_FAST)
target_pass_fast = bool(full_metrics_fast["rmse"] <= TARGET_GATE_FAST)
causal_pass_fast = bool(no_gr_degradation_fast >= 0.25 and trajectory_degradation_fast >= 0.25)

verdict_fast = {
    "experiment": EXPERIMENT_FAST,
    "stage": "capacity_then_fold1",
    "controls": selected_controls_s,
    "free_dimensions": len(coefficient_mean_fast),
    "parameters": parameters_fast,
    "representation_oracle_rmse": selected_oracle_s,
    "coefficient_center": "zero_s_anchored",
    "zero_s_baseline": zero_s_metrics_fast,
    "population_mean_baseline": population_mean_metrics_fast,
    "capacity_result": capacity_best_fast["metrics"],
    "capacity_zero_s_baseline": capacity_zero_baseline_fast,
    "capacity_minutes": capacity_minutes_fast,
    "best_epoch": int(fold_best_fast["epoch"]),
    "result": full_metrics_fast,
    "causal_controls": {
        "no_gr_degradation": no_gr_degradation_fast,
        "no_postps_xyz_degradation": trajectory_degradation_fast,
    },
    "architecture_pass": architecture_pass_fast,
    "target_7ft_pass": target_pass_fast,
    "causal_pass": causal_pass_fast,
    "strict_pass": bool(architecture_pass_fast and causal_pass_fast),
    "training_minutes": fold_minutes_fast,
    "elapsed_minutes": (time.time() - STARTED_FAST) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> add heel reference while freezing this trajectory recipe"
        if architecture_pass_fast and causal_pass_fast
        else "FAIL -> close anchored S-head route; do not add references"
    ),
}
_json_fast(OUT_DIR_FAST / "verdict.json", verdict_fast)

print("\nFAST CAPACITY HISTORY")
print(capacity_history_fast.to_string(index=False), flush=True)
print("\nFAST FOLD HISTORY")
print(fold_history_fast.to_string(index=False), flush=True)
print("\nFAST CAUSAL CONTROLS")
print(controls_fast.to_string(index=False), flush=True)
print("\nFAST FINAL VERDICT")
print(json.dumps(verdict_fast, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_FAST, flush=True)
