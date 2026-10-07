"""Fold-clean anchored-S trajectory-head gateway from the full-well v3 plan.

Run in the same notebook kernel after the native one-fold cells have prepared
``records``, ``train_records`` and ``valid_records``.  This stage deliberately
removes heel/typewell references and the Delta-TVT cost volume.  It isolates
the missing mechanism identified by the plan:

    complete native-resolution XYZ/GR trajectory -> anchored low-dimensional
    structural curve S(MD), with TVT = S - Z.

The cell first audits fixed cubic B-spline representations with 8/12/16/24/32
controls.  It then selects the smallest representation whose fold-1 oracle is
at most 1.5 ft and trains one complete-well coefficient model.  Validation uses
only the untouched official fold-1 mask.  No neighbour, PF, CP12, Gold,
typewell, heel reference, or historical prediction is consumed.
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


# ---------------------------------------------------------------------------
# Live-kernel prerequisites from the already completed native data pipeline.
# ---------------------------------------------------------------------------


REQUIRED_LIVE_S = {
    "records",
    "train_wells",
    "valid_wells",
    "train_records",
    "valid_records",
    "DATA_ROOT",
    "OOF_ROOT",
    "_rmse",
    "_seed_everything",
}
missing_live_s = sorted(REQUIRED_LIVE_S.difference(globals()))
if missing_live_s:
    raise RuntimeError(
        "Run/interrupt native_resolution_dualref_onefold_v0_cell.py in this "
        f"kernel first; missing {missing_live_s}"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


EXPERIMENT_S = "native_fullwell_trajectory_shead_onefold_v0"
OUT_DIR_S = OOF_ROOT / EXPERIMENT_S
MODEL_DIR_S = OUT_DIR_S / "models"
SEED_S = 20260721

CONTROL_COUNTS_S = (8, 12, 16, 24, 32)
REPRESENTATION_GATE_FT_S = 1.50
MAX_SELECTED_CONTROLS_S = 24
BSPLINE_DEGREE_S = 3
BSPLINE_RIDGE_S = 1e-4

TOKEN_CHANNELS_S = 128
CONTROL_ATTENTION_HEADS_S = 8
ATTENTION_LAYERS_S = 3
EPOCHS_S = 40
REPORT_EVERY_S = 2
MIN_EPOCHS_S = 10
EARLY_STOPPING_REPORTS_S = 6
LEARNING_RATE_S = 2.5e-4
MIN_LEARNING_RATE_S = 3e-5
WEIGHT_DECAY_S = 3e-5
GRADIENT_CLIP_S = 3.0
GRAD_SCALER_INITIAL_SCALE_S = 512.0

LOSS_PATH_MSE_S = 1.00
LOSS_COEFFICIENT_S = 0.10
LOSS_SLOPE_S = 0.035
LOSS_CURVATURE_S = 0.010
PATH_ERROR_SCALE_FT_S = 8.0
COEFFICIENT_SCALE_FT_S = 16.0
GR_BLOCK_DROPOUT_PROBABILITY_S = 0.30
GR_BLOCK_DROPOUT_RANGE_S = (0.05, 0.15)

ARCHITECTURE_GATE_FT_S = 10.0
TARGET_GATE_FT_S = 7.0
POSTPS_XYZ_DEGRADATION_GATE_S = 0.25
NO_GR_DEGRADATION_GATE_S = 0.25
FAIL_FAST_EPOCH_S = 14
FAIL_FAST_VALID_RMSE_S = 12.0
FAIL_FAST_TRAIN_SUBSET_RMSE_S = 4.0

for directory in (OUT_DIR_S, MODEL_DIR_S):
    directory.mkdir(parents=True, exist_ok=True)


def _write_json_s(path, payload):
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
# Exact Z arrays and fixed anchored cubic B-spline representation.
# ---------------------------------------------------------------------------


def _load_z_arrays_s(wells):
    output = {}
    rows = []
    for index, well in enumerate(wells, 1):
        frame = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=["Z"],
        )
        z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(np.float32)
        if len(z) != len(records[well].md) or not np.isfinite(z).all():
            raise RuntimeError(f"{well}: invalid exact Z array")
        output[well] = z
        rows.append(
            {
                "well": well,
                "rows": len(z),
                "z_span": float(np.max(z) - np.min(z)),
            }
        )
        if index % 100 == 0 or index == len(wells):
            print(f"load exact Z {index}/{len(wells)}", flush=True)
    return output, pd.DataFrame(rows)


def _clamped_knots_s(n_controls, degree=BSPLINE_DEGREE_S):
    if n_controls <= degree:
        raise ValueError("n_controls must exceed spline degree")
    internal_count = n_controls - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
        if internal_count > 0
        else np.empty(0, np.float64)
    )
    return np.r_[
        np.zeros(degree + 1, np.float64),
        internal,
        np.ones(degree + 1, np.float64),
    ]


def _bspline_basis_s(progress, n_controls, degree=BSPLINE_DEGREE_S):
    """Cox-de Boor basis; first column is removed to hard-anchor delta S(0)=0."""

    u = np.clip(np.asarray(progress, np.float64), 0.0, 1.0)
    knots = _clamped_knots_s(n_controls, degree)
    basis = np.zeros((len(u), n_controls + degree), np.float64)
    for index in range(n_controls + degree):
        basis[:, index] = (
            (u >= knots[index]) & (u < knots[index + 1])
        ).astype(np.float64)
    basis[u >= 1.0, :] = 0.0
    basis[u >= 1.0, n_controls - 1] = 1.0
    for order in range(1, degree + 1):
        updated = np.zeros((len(u), n_controls + degree - order), np.float64)
        for index in range(updated.shape[1]):
            left_denominator = knots[index + order] - knots[index]
            right_denominator = knots[index + order + 1] - knots[index + 1]
            if left_denominator > 0:
                updated[:, index] += (
                    (u - knots[index]) / left_denominator * basis[:, index]
                )
            if right_denominator > 0:
                updated[:, index] += (
                    (knots[index + order + 1] - u)
                    / right_denominator
                    * basis[:, index + 1]
                )
        basis = updated
    basis[u >= 1.0, :] = 0.0
    basis[u >= 1.0, -1] = 1.0
    # At u=0 the clamped basis is [1, 0, ...].  Removing the first column
    # makes every decoded curve exactly zero at the prediction-start anchor.
    return basis[:, 1:].astype(np.float32)


def _target_s_s(record, z, n_controls):
    eval_index = record.eval_index
    denominator = max(float(record.md[-1] - record.md[record.anchor_index]), 1.0)
    progress = (
        record.md[eval_index] - record.md[record.anchor_index]
    ) / denominator
    basis = _bspline_basis_s(progress, n_controls)
    z_delta = z[eval_index].astype(np.float64) - float(z[record.anchor_index])
    delta_s = (
        record.truth[eval_index].astype(np.float64)
        - record.anchor_tvt
        + z_delta
    )
    design = basis.astype(np.float64)
    gram = design.T @ design
    rhs = design.T @ delta_s
    coefficients = np.linalg.solve(
        gram + BSPLINE_RIDGE_S * np.eye(gram.shape[0]), rhs
    )
    reconstruction = design @ coefficients
    return (
        progress.astype(np.float32),
        basis.astype(np.float32),
        delta_s.astype(np.float32),
        coefficients.astype(np.float32),
        reconstruction.astype(np.float32),
    )


def _representation_audit_s(evaluation_records, z_arrays):
    rows = []
    bywell_rows = []
    for controls in CONTROL_COUNTS_S:
        truths, predictions = [], []
        well_scores = []
        for record in evaluation_records:
            _, _, truth, _, prediction = _target_s_s(
                record, z_arrays[record.well], controls
            )
            truths.append(truth)
            predictions.append(prediction)
            well_rmse = _rmse(truth, prediction)
            well_scores.append(well_rmse)
            bywell_rows.append(
                {
                    "controls": controls,
                    "free_dimensions": controls - 1,
                    "well": record.well,
                    "rows": len(truth),
                    "rmse": well_rmse,
                }
            )
        truth = np.concatenate(truths)
        prediction = np.concatenate(predictions)
        rows.append(
            {
                "controls": controls,
                "free_dimensions": controls - 1,
                "rows": len(truth),
                "wells": len(evaluation_records),
                "pooled_rmse": _rmse(truth, prediction),
                "median_well_rmse": float(np.median(well_scores)),
                "p95_well_rmse": float(np.quantile(well_scores, 0.95)),
                "max_well_rmse": float(np.max(well_scores)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bywell_rows)


def _select_controls_s(audit):
    eligible = audit[
        (audit["pooled_rmse"] <= REPRESENTATION_GATE_FT_S)
        & (audit["controls"] <= MAX_SELECTED_CONTROLS_S)
    ].sort_values("controls")
    if len(eligible):
        return int(eligible.iloc[0]["controls"])
    fallback = audit[audit["controls"] <= MAX_SELECTED_CONTROLS_S].sort_values(
        "pooled_rmse"
    )
    return int(fallback.iloc[0]["controls"])


# ---------------------------------------------------------------------------
# Native-resolution trajectory encoder and control-query decoder.
# ---------------------------------------------------------------------------


class ResidualTCNS(nn.Module):
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


class ControlCrossAttentionS(nn.Module):
    def __init__(self, channels=TOKEN_CHANNELS_S):
        super().__init__()
        self.norm_query = nn.LayerNorm(channels)
        self.norm_memory = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            CONTROL_ATTENTION_HEADS_S,
            dropout=0.05,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 3),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(channels * 3, channels),
        )

    def forward(self, query, memory):
        attended, _ = self.attention(
            self.norm_query(query),
            self.norm_memory(memory),
            self.norm_memory(memory),
            need_weights=False,
        )
        query = query + attended
        return query + self.ff(self.norm_ff(query))


class FullWellTrajectorySHead(nn.Module):
    def __init__(self, free_dimensions):
        super().__init__()
        self.free_dimensions = int(free_dimensions)
        self.stem = nn.Sequential(
            nn.Conv1d(14, 64, 11, padding=5),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualTCNS(64, 1),
            ResidualTCNS(64, 2),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 96, 7, stride=2, padding=3),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            ResidualTCNS(96, 1),
            ResidualTCNS(96, 2),
            ResidualTCNS(96, 4),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(96, 128, 7, stride=2, padding=3),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResidualTCNS(128, 1),
            ResidualTCNS(128, 2),
            ResidualTCNS(128, 4),
            ResidualTCNS(128, 8),
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(128, TOKEN_CHANNELS_S, 7, stride=2, padding=3),
            nn.GroupNorm(8, TOKEN_CHANNELS_S),
            nn.GELU(),
            ResidualTCNS(TOKEN_CHANNELS_S, 1),
            ResidualTCNS(TOKEN_CHANNELS_S, 2),
            ResidualTCNS(TOKEN_CHANNELS_S, 4),
            ResidualTCNS(TOKEN_CHANNELS_S, 8),
            ResidualTCNS(TOKEN_CHANNELS_S, 16),
            ResidualTCNS(TOKEN_CHANNELS_S, 32),
        )
        self.control_queries = nn.Parameter(
            torch.randn(1, self.free_dimensions, TOKEN_CHANNELS_S) * 0.02
        )
        positions = torch.linspace(0.0, 1.0, self.free_dimensions)
        positional = torch.stack(
            [
                positions,
                positions**2,
                torch.sin(math.pi * positions),
                torch.cos(math.pi * positions),
            ],
            dim=1,
        )
        self.register_buffer("control_positions", positional, persistent=False)
        self.position_project = nn.Linear(4, TOKEN_CHANNELS_S)
        self.attention_layers = nn.ModuleList(
            [ControlCrossAttentionS() for _ in range(ATTENTION_LAYERS_S)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(TOKEN_CHANNELS_S),
            nn.Linear(TOKEN_CHANNELS_S, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, values):
        memory = self.down3(self.down2(self.down1(self.stem(values))))
        memory = memory.transpose(1, 2)
        query = self.control_queries.expand(values.shape[0], -1, -1)
        query = query + self.position_project(self.control_positions)[None]
        for layer in self.attention_layers:
            query = layer(query, memory)
        return self.output(query).squeeze(-1)


# ---------------------------------------------------------------------------
# Input interventions, loss and decoding.
# ---------------------------------------------------------------------------


def _input_s(record, device, mode="full", training=False, rng=None):
    horizontal = record.horizontal.copy()
    condition = record.condition.copy()
    if mode == "no_gr":
        horizontal[:] = 0.0
        condition[9] = 0.0
    elif mode == "no_postps_xyz":
        start = record.anchor_index + 1
        condition[4:9, start:] = 0.0
    elif mode != "full":
        raise ValueError(mode)

    if (
        training
        and rng is not None
        and rng.random() < GR_BLOCK_DROPOUT_PROBABILITY_S
    ):
        start = record.anchor_index + 1
        available = max(len(record.md) - start, 1)
        fraction = rng.uniform(*GR_BLOCK_DROPOUT_RANGE_S)
        length = max(8, int(round(fraction * available)))
        block_start = int(rng.integers(start, max(start + 1, len(record.md) - length + 1)))
        block_end = min(len(record.md), block_start + length)
        horizontal[:, block_start:block_end] = 0.0
        condition[9, block_start:block_end] = 0.0

    values = np.concatenate([horizontal, condition], axis=0).astype(np.float32)
    return torch.from_numpy(values)[None].to(device)


def _loss_s(model_coefficients, target_payload, device, row_weight):
    basis, target_delta_s, target_coefficients = target_payload
    basis = torch.from_numpy(basis).to(device)
    target_delta_s = torch.from_numpy(target_delta_s).to(device)
    target_coefficients = torch.from_numpy(target_coefficients).to(device)
    coefficients = model_coefficients[0]
    prediction = basis @ coefficients
    path_mse = torch.mean(
        ((prediction - target_delta_s) / PATH_ERROR_SCALE_FT_S) ** 2
    )
    coefficient_loss = torch.mean(
        ((coefficients - target_coefficients) / COEFFICIENT_SCALE_FT_S) ** 2
    )
    slope = F.smooth_l1_loss(
        torch.diff(prediction) / 2.0,
        torch.diff(target_delta_s) / 2.0,
    )
    curvature = F.smooth_l1_loss(
        torch.diff(prediction, n=2) / 2.0,
        torch.diff(target_delta_s, n=2) / 2.0,
    )
    loss = row_weight * (
        LOSS_PATH_MSE_S * path_mse
        + LOSS_COEFFICIENT_S * coefficient_loss
        + LOSS_SLOPE_S * slope
        + LOSS_CURVATURE_S * curvature
    )
    return loss, path_mse.detach(), coefficient_loss.detach()


def _target_payload_s(record, z_arrays, controls):
    _, basis, delta_s, coefficients, _ = _target_s_s(
        record, z_arrays[record.well], controls
    )
    return basis, delta_s, coefficients


@torch.no_grad()
def _predict_s(model, record, z_arrays, controls, device, mode="full"):
    model.eval()
    values = _input_s(record, device, mode=mode)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        coefficients = model(values)[0].float().cpu().numpy()
    _, basis, _, _, _ = _target_s_s(record, z_arrays[record.well], controls)
    delta_s = basis @ coefficients
    z = z_arrays[record.well]
    z_delta = z[record.eval_index].astype(np.float64) - float(z[record.anchor_index])
    prediction = record.anchor_tvt + delta_s.astype(np.float64) - z_delta
    truth = record.truth[record.eval_index].astype(np.float64)
    return truth, prediction


@torch.no_grad()
def _evaluate_s(model, evaluation_records, z_arrays, controls, device, mode="full", save_rows=False):
    truths, predictions, labels, bywell = [], [], [], []
    for record in evaluation_records:
        truth, prediction = _predict_s(
            model, record, z_arrays, controls, device, mode=mode
        )
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
        "wells_above_12ft": int((bywell["rmse"] > 12.0).sum()),
    }
    payload = None
    if save_rows:
        payload = {
            "well": np.concatenate(labels),
            "truth": truth.astype(np.float32),
            "prediction": prediction.astype(np.float32),
        }
    return metrics, bywell, payload


def _train_s(z_arrays, controls, device):
    free_dimensions = controls - 1
    model = FullWellTrajectorySHead(free_dimensions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_S,
        weight_decay=WEIGHT_DECAY_S,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_S, eta_min=MIN_LEARNING_RATE_S
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=GRAD_SCALER_INITIAL_SCALE_S,
        growth_interval=500,
    )
    mean_rows = float(np.mean([record.eval_rows for record in train_records]))
    train_subset = [
        records[well]
        for well in train_wells[:: max(1, len(train_wells) // 24)][:24]
    ]
    rng = np.random.default_rng(SEED_S)
    history, best = [], None
    reports_without_improvement = 0
    started = time.time()

    for epoch in range(1, EPOCHS_S + 1):
        model.train()
        order = rng.permutation(len(train_records))
        sums = {"loss": 0.0, "path": 0.0, "coefficient": 0.0, "skipped": 0}
        for batch_number, index in enumerate(order, 1):
            record = train_records[int(index)]
            target = _target_payload_s(record, z_arrays, controls)
            values = _input_s(record, device, training=True, rng=rng)
            row_weight = record.eval_rows / max(mean_rows, 1.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                coefficients = model(values)
                loss, path_loss, coefficient_loss = _loss_s(
                    coefficients, target, device, row_weight
                )
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_S)
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < before:
                sums["skipped"] += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["path"] += float(path_loss.cpu())
            sums["coefficient"] += float(coefficient_loss.cpu())
            del target, values, coefficients, loss
            if batch_number % 100 == 0:
                print(
                    f"S-HEAD epoch {epoch}/{EPOCHS_S} wells {batch_number}/{len(train_records)}",
                    flush=True,
                )
        scheduler.step()

        should_report = epoch <= 2 or epoch % REPORT_EVERY_S == 0 or epoch == EPOCHS_S
        if not should_report:
            continue
        valid_metrics, valid_bywell, _ = _evaluate_s(
            model, valid_records, z_arrays, controls, device
        )
        train_metrics, _, _ = _evaluate_s(
            model, train_subset, z_arrays, controls, device
        )
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / len(train_records),
            "train_path_mse_scaled": sums["path"] / len(train_records),
            "train_coefficient_loss_scaled": sums["coefficient"] / len(train_records),
            "train_subset_rmse": train_metrics["rmse"],
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "skipped_steps": sums["skipped"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("S-HEAD EPOCH", row, flush=True)
        pd.DataFrame(history).to_csv(OUT_DIR_S / "training_history.csv", index=False)

        if best is None or valid_metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
            best = {
                "epoch": epoch,
                "metrics": valid_metrics,
                "bywell": valid_bywell.copy(),
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            torch.save(
                {
                    "experiment": EXPERIMENT_S,
                    "epoch": epoch,
                    "controls": controls,
                    "metrics": valid_metrics,
                    "model": best["state"],
                },
                MODEL_DIR_S / "best_fold1.pt",
            )
            reports_without_improvement = 0
        else:
            reports_without_improvement += 1

        if (
            epoch >= FAIL_FAST_EPOCH_S
            and valid_metrics["rmse"] > FAIL_FAST_VALID_RMSE_S
            and train_metrics["rmse"] <= FAIL_FAST_TRAIN_SUBSET_RMSE_S
        ):
            print("S-HEAD FAIL-FAST: training fit exists but trajectory transfer is absent", flush=True)
            break
        if (
            epoch >= MIN_EPOCHS_S
            and reports_without_improvement >= EARLY_STOPPING_REPORTS_S
        ):
            print(f"S-HEAD early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), train_subset, (time.time() - started) / 60.0


# ---------------------------------------------------------------------------
# Execute representation audit and one-fold trajectory baseline.
# ---------------------------------------------------------------------------


STARTED_S = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("The trajectory S-head gateway requires CUDA")
device_s = torch.device("cuda")
_seed_everything(SEED_S)

all_wells_s = sorted(records)
z_arrays_s, z_audit_s = _load_z_arrays_s(all_wells_s)
z_audit_s.to_csv(OUT_DIR_S / "z_input_audit.csv", index=False)

representation_s, representation_bywell_s = _representation_audit_s(
    valid_records, z_arrays_s
)
representation_s.to_csv(OUT_DIR_S / "representation_oracle.csv", index=False)
representation_bywell_s.to_csv(
    OUT_DIR_S / "representation_oracle_bywell.csv", index=False
)
selected_controls_s = _select_controls_s(representation_s)
selected_oracle_s = float(
    representation_s.loc[
        representation_s["controls"] == selected_controls_s, "pooled_rmse"
    ].iloc[0]
)
print("S REPRESENTATION ORACLE")
print(representation_s.to_string(index=False), flush=True)
print(
    "S REPRESENTATION SELECTION",
    {
        "controls": selected_controls_s,
        "free_dimensions": selected_controls_s - 1,
        "oracle_rmse": selected_oracle_s,
        "gate": REPRESENTATION_GATE_FT_S,
    },
    flush=True,
)
if selected_oracle_s > REPRESENTATION_GATE_FT_S:
    verdict_s = {
        "experiment": EXPERIMENT_S,
        "stage": "representation_oracle",
        "selected_controls": selected_controls_s,
        "oracle_rmse": selected_oracle_s,
        "strict_pass": False,
        "next": "FAIL -> anchored S representation is too lossy; do not train",
    }
    _write_json_s(OUT_DIR_S / "verdict.json", verdict_s)
    raise RuntimeError(json.dumps(verdict_s, indent=2))

probe_s = FullWellTrajectorySHead(selected_controls_s - 1).to(device_s)
probe_record_s = train_records[0]
with torch.no_grad(), torch.amp.autocast("cuda"):
    probe_output_s = probe_s(_input_s(probe_record_s, device_s))
parameters_s = int(sum(parameter.numel() for parameter in probe_s.parameters()))
smoke_s = {
    "parameters": parameters_s,
    "input": [1, 14, len(probe_record_s.md)],
    "output": list(probe_output_s.shape),
    "finite": bool(torch.isfinite(probe_output_s).all()),
}
print("S-HEAD MODEL SMOKE", smoke_s, flush=True)
del probe_s, probe_output_s
gc.collect()
torch.cuda.empty_cache()

model_s, best_s, history_s, train_subset_s, training_minutes_s = _train_s(
    z_arrays_s, selected_controls_s, device_s
)

full_metrics_s, bywell_s, payload_s = _evaluate_s(
    model_s,
    valid_records,
    z_arrays_s,
    selected_controls_s,
    device_s,
    save_rows=True,
)
bywell_s.to_csv(OUT_DIR_S / "fold1_bywell.csv", index=False)
np.savez_compressed(OUT_DIR_S / "fold1_oof.npz", **payload_s)

control_rows_s = []
for mode in ("full", "no_gr", "no_postps_xyz"):
    metrics, _, _ = _evaluate_s(
        model_s, valid_records, z_arrays_s, selected_controls_s, device_s, mode=mode
    )
    control_rows_s.append(
        {
            "mode": mode,
            **metrics,
            "degradation_vs_full": metrics["rmse"] - full_metrics_s["rmse"],
        }
    )
controls_s = pd.DataFrame(control_rows_s)
controls_s.to_csv(OUT_DIR_S / "causal_controls.csv", index=False)

no_gr_degradation_s = float(
    controls_s.loc[controls_s["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
trajectory_degradation_s = float(
    controls_s.loc[
        controls_s["mode"] == "no_postps_xyz", "degradation_vs_full"
    ].iloc[0]
)
architecture_pass_s = bool(full_metrics_s["rmse"] <= ARCHITECTURE_GATE_FT_S)
target_pass_s = bool(full_metrics_s["rmse"] <= TARGET_GATE_FT_S)
trajectory_signal_pass_s = bool(
    trajectory_degradation_s >= POSTPS_XYZ_DEGRADATION_GATE_S
)
causal_pass_s = bool(
    trajectory_signal_pass_s and no_gr_degradation_s >= NO_GR_DEGRADATION_GATE_S
)

verdict_s = {
    "experiment": EXPERIMENT_S,
    "stage": "main_well_trajectory_s_head",
    "standalone": True,
    "uses_neighbor": False,
    "uses_typewell": False,
    "uses_heel_reference": False,
    "selected_controls": selected_controls_s,
    "free_dimensions": selected_controls_s - 1,
    "representation_oracle_rmse": selected_oracle_s,
    "parameters": parameters_s,
    "best_epoch": int(best_s["epoch"]),
    "result": full_metrics_s,
    "causal_controls": {
        "no_gr_degradation": no_gr_degradation_s,
        "no_postps_xyz_degradation": trajectory_degradation_s,
    },
    "gates": {
        "architecture_rmse_le": ARCHITECTURE_GATE_FT_S,
        "target_rmse_le": TARGET_GATE_FT_S,
        "no_gr_degradation_ge": NO_GR_DEGRADATION_GATE_S,
        "no_postps_xyz_degradation_ge": POSTPS_XYZ_DEGRADATION_GATE_S,
    },
    "architecture_pass": architecture_pass_s,
    "target_7ft_pass": target_pass_s,
    "trajectory_signal_pass": trajectory_signal_pass_s,
    "causal_pass": causal_pass_s,
    "strict_pass": bool(architecture_pass_s and causal_pass_s),
    "training_minutes": training_minutes_s,
    "elapsed_minutes": (time.time() - STARTED_S) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> add heel reference only, keeping the S head/backbone frozen"
        if architecture_pass_s and causal_pass_s
        else "FAIL -> trajectory S head did not transfer; do not add references yet"
    ),
}
_write_json_s(OUT_DIR_S / "verdict.json", verdict_s)

print("\nS-HEAD TRAINING HISTORY")
print(history_s.to_string(index=False), flush=True)
print("\nS-HEAD CAUSAL CONTROLS")
print(controls_s.to_string(index=False), flush=True)
print("\nS-HEAD FINAL VERDICT")
print(json.dumps(verdict_s, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_S, flush=True)

