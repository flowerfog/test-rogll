"""Fold-2 gateway for a multi-horizon drift-correspondence full-well model.

This experiment follows the failed pretrained PathNet v2.  V2 proved that a
complete-well head can improve the anchor by about 2 ft, but its direct state
posterior still drifts from 3.3 ft near the heel to 15.8 ft beyond 2,000 ft.

V3 makes drift the primary structured target.  It consumes the same legal,
pretrained real-correspondence volume and jointly predicts:

* an absolute TVT state posterior at every 32-ft node;
* Delta-S constraints at 32/64/128/256/512 ft, where S = TVT + Z;
* learned precision for each absolute and interval constraint.

All constraints are decoded together by one differentiable global weighted
least-squares solve.  There is no autoregressive use of the previous predicted
segment, no neighbor data, no CP12/Gold, and no test-time optimisation.

Run this file directly after uploading these source files to /root/autodl-tmp:

* real_correspondence_multiscale_matcher_gateway_v0_cell.py
* prefix_contrastive_matcher_adaptation_gateway_v1_cell.py
* pretrained_matcher_pathnet_nested_gateway_v2_cell.py
* this file

The v2 source is loaded only through its definition/bootstrap prefix.  Its
failed PathNet experiment tail is not executed.
"""

from __future__ import annotations

from pathlib import Path
import ast
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
# Restart-safe import of all validated record/matcher/volume definitions.
# ---------------------------------------------------------------------------


def _source_v3(filename):
    candidates = (Path("/root/autodl-tmp") / filename, Path.cwd() / filename)
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Upload {filename} to /root/autodl-tmp")
    return path


def _exec_v2_definitions_v3():
    path = _source_v3("pretrained_matcher_pathnet_nested_gateway_v2_cell.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    definitions = []
    found = False
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if "started_v2" in names:
                found = True
                break
        definitions.append(node)
    if not found:
        raise RuntimeError(f"Cannot find v2 execution marker in {path}")
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(path) + "::<definitions_only_for_v3>",
            "exec",
        ),
        globals(),
    )
    return path


V2_SOURCE_V3 = _exec_v2_definitions_v3()

REQUIRED_V3 = {
    "OOF_ROOT",
    "device",
    "core_wells",
    "tune_wells",
    "samples",
    "prefix_queries_v1",
    "mixed_epochs_v1",
    "selected_lambda_v1",
    "selected_dose_v1",
    "_fit_mixed_v1",
    "_build_volume_bank_v2",
    "_load_raw_record",
    "_path_payload_v2",
    "_summarise_payloads_v2",
    "STATE_OFFSETS_V2",
    "N_STATES_V2",
    "INPUT_CHANNELS_V2",
    "NODE_STEP_FT_V2",
    "_write_json",
    "_json_default",
    "fold_path",
}
missing_v3 = sorted(REQUIRED_V3.difference(globals()))
if missing_v3:
    raise RuntimeError("V3 bootstrap is incomplete: " + repr(missing_v3))


# ---------------------------------------------------------------------------
# Frozen protocol and model configuration.
# ---------------------------------------------------------------------------


EXPERIMENT_V3 = "multihorizon_drift_correspondence_fold2_v3"
OUT_DIR_V3 = Path(OOF_ROOT) / EXPERIMENT_V3
MODEL_DIR_V3 = OUT_DIR_V3 / "models"
for directory_v3 in (OUT_DIR_V3, MODEL_DIR_V3):
    directory_v3.mkdir(parents=True, exist_ok=True)

SEED_V3 = 20260810
MAX_EPOCHS_V3 = 18
MIN_EPOCHS_V3 = 7
EARLY_STOPPING_V3 = 5
LEARNING_RATE_V3 = 2.5e-4
MIN_LEARNING_RATE_V3 = 2.5e-5
WEIGHT_DECAY_V3 = 2.0e-5
GRAD_ACCUM_WELLS_V3 = 4
GRADIENT_CLIP_V3 = 4.0

BASE_CHANNELS_V3 = 32
TEMPORAL_CHANNELS_V3 = 64
TEMPORAL_DILATIONS_V3 = (1, 2, 4, 8, 16, 32, 64)
HORIZONS_V3 = (1, 2, 4, 8, 16)
HORIZON_FT_V3 = tuple(int(horizon * NODE_STEP_FT_V2) for horizon in HORIZONS_V3)

TARGET_SIGMA_FT_V3 = 1.5
LOSS_PATH_V3 = 0.50
LOSS_ABSOLUTE_CE_V3 = 0.20
LOSS_DRIFT_V3 = 0.25
LOSS_CYCLE_V3 = 0.05

FOLD2_RMSE_GATE_V3 = 10.0
FOLD2_FAR_GATE_V3 = 12.0
FOLD2_MEDIAN_GATE_V3 = 6.5
NO_GR_DEGRADATION_GATE_V3 = 0.30
SHUFFLED_TYPEWELL_GATE_V3 = 0.30

STATE_OFFSETS_TENSOR_V3 = torch.from_numpy(
    np.asarray(STATE_OFFSETS_V2, np.float32)
).to(device)


def _seed_v3(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _distance_weights_v3(node_md):
    node_md = np.asarray(node_md, np.float64)
    distance = node_md - node_md[0]
    bins = np.digitize(distance, [500.0, 1000.0, 2000.0], right=False)
    weights = np.zeros(len(distance), np.float32)
    present = np.unique(bins)
    for group in present:
        mask = bins == group
        weights[mask] = 1.0 / max(int(mask.sum()), 1)
    weights *= len(weights) / max(float(weights.sum()), 1e-8)
    return weights


# ---------------------------------------------------------------------------
# Multi-horizon global drift model.
# ---------------------------------------------------------------------------


class AxialDriftBlockV3(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.state = nn.Sequential(
            nn.Conv2d(channels, channels, (9, 1), padding=(4, 0), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )
        self.time = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                (1, 5),
                padding=(0, 2 * dilation),
                dilation=(1, dilation),
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
        )
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GLU(dim=1),
        )

    def forward(self, values):
        hidden = values + self.time(self.state(values))
        return hidden + self.mix(hidden)


class TemporalResidualV3(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return values + self.network(values)


class MultiHorizonDriftCorrespondenceV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS_V2, BASE_CHANNELS_V3, 1),
            nn.GroupNorm(4, BASE_CHANNELS_V3),
            nn.GELU(),
        )
        self.axial = nn.ModuleList(
            [
                AxialDriftBlockV3(BASE_CHANNELS_V3, dilation)
                for dilation in TEMPORAL_DILATIONS_V3
            ]
        )
        self.unary = nn.Sequential(
            nn.GroupNorm(4, BASE_CHANNELS_V3),
            nn.GELU(),
            nn.Conv2d(BASE_CHANNELS_V3, 1, 1),
        )
        summary_channels = 2 * BASE_CHANNELS_V3 + 2
        self.temporal_input = nn.Sequential(
            nn.Conv1d(summary_channels, TEMPORAL_CHANNELS_V3, 1),
            nn.GroupNorm(4, TEMPORAL_CHANNELS_V3),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[
                TemporalResidualV3(TEMPORAL_CHANNELS_V3, dilation)
                for dilation in TEMPORAL_DILATIONS_V3
            ]
        )
        self.absolute_precision = nn.Conv1d(TEMPORAL_CHANNELS_V3, 1, 1)
        self.horizon_heads = nn.ModuleDict(
            {
                str(horizon): nn.Conv1d(TEMPORAL_CHANNELS_V3, 2, 1)
                for horizon in HORIZONS_V3
            }
        )
        self.base_scale = nn.Parameter(torch.tensor(0.75))

    @staticmethod
    def _global_solve(absolute_s, absolute_precision, horizon_payload, columns):
        dtype = absolute_s.dtype
        # CUDA's dense LU solver does not implement FP16.  Autocast must be
        # disabled around both normal-equation construction and the solve;
        # calling .float() alone is insufficient because autocast converts the
        # matrix multiplications back to half precision.
        with torch.amp.autocast("cuda", enabled=False):
            solve_dtype = torch.float32
            rows = []
            observations = []
            precisions = []

            identity = torch.eye(
                columns, device=absolute_s.device, dtype=solve_dtype
            )
            rows.append(identity)
            observations.append(absolute_s.float())
            precisions.append(absolute_precision.float())

            anchor = torch.zeros(
                1, columns, device=absolute_s.device, dtype=solve_dtype
            )
            anchor[0, 0] = 1.0
            rows.append(anchor)
            observations.append(
                torch.zeros(1, device=absolute_s.device, dtype=solve_dtype)
            )
            precisions.append(
                torch.full(
                    (1,), 256.0, device=absolute_s.device, dtype=solve_dtype
                )
            )

            for horizon, (delta_s, precision) in horizon_payload.items():
                horizon = int(horizon)
                if columns <= horizon:
                    continue
                count = columns - horizon
                design_edge = torch.zeros(
                    count,
                    columns,
                    device=absolute_s.device,
                    dtype=solve_dtype,
                )
                index = torch.arange(count, device=absolute_s.device)
                design_edge[index, index] = -1.0
                design_edge[index, index + horizon] = 1.0
                rows.append(design_edge)
                observations.append(delta_s[:count].float())
                precisions.append(precision[:count].float())

            design = torch.cat(rows, dim=0)
            observation = torch.cat(observations, dim=0)
            precision = torch.cat(precisions, dim=0).clamp(0.03, 30.0)
            system = design.T @ (precision[:, None] * design)
            system = system + 1e-4 * torch.eye(
                columns, device=system.device, dtype=solve_dtype
            )
            rhs = design.T @ (precision * observation)
            solution = torch.linalg.solve(system, rhs)
        solution = solution - solution[0]
        return solution.to(dtype)

    def forward(self, values, z_relative):
        if values.shape[0] != 1:
            raise RuntimeError("V3 currently trains one complete well per forward")
        hidden = self.input(values)
        for block in self.axial:
            hidden = block(hidden)

        unary = self.unary(hidden).squeeze(1)
        unary = unary + F.softplus(self.base_scale) * values[:, 0]
        valid = values[:, -1] > 0.5
        unary = unary.masked_fill(~valid, -30.0)
        probability = torch.softmax(unary.float(), dim=1).to(unary.dtype)
        absolute_tvt = (
            probability * STATE_OFFSETS_TENSOR_V3[None, :, None]
        ).sum(dim=1)

        state_mean = hidden.mean(dim=2)
        state_max = hidden.amax(dim=2)
        progress = values[:, 9, 0]
        z_channel = (z_relative / 128.0).clamp(-4.0, 4.0)[None, None]
        summary = torch.cat(
            [state_mean, state_max, progress[:, None], z_channel], dim=1
        )
        temporal = self.temporal(self.temporal_input(summary))
        absolute_precision = F.softplus(
            self.absolute_precision(temporal)[:, 0]
        ) + 0.05

        horizon_payload = {}
        for horizon in HORIZONS_V3:
            raw = self.horizon_heads[str(horizon)](temporal)
            max_delta = 8.0 + 0.08 * float(horizon * NODE_STEP_FT_V2)
            delta_s = max_delta * torch.tanh(raw[:, 0])
            precision = F.softplus(raw[:, 1]) + 0.05
            horizon_payload[horizon] = (delta_s[0], precision[0])

        solved_s = self._global_solve(
            absolute_tvt[0] + z_relative,
            absolute_precision[0],
            horizon_payload,
            values.shape[-1],
        )
        solved_tvt = solved_s - z_relative
        solved_tvt = solved_tvt - solved_tvt[0]
        return {
            "unary": unary[0],
            "absolute_tvt": absolute_tvt[0],
            "absolute_precision": absolute_precision[0],
            "horizons": horizon_payload,
            "solved_tvt": solved_tvt,
            "solved_s": solved_s,
        }


def _tensors_v3(volume):
    values = torch.from_numpy(volume.input.astype(np.float32))[None].to(
        device, non_blocking=True
    )
    truth = torch.from_numpy(volume.truth_offset).to(device, non_blocking=True)
    z_relative = torch.from_numpy(volume.node_z_relative).to(
        device, non_blocking=True
    )
    distance_weight = torch.from_numpy(_distance_weights_v3(volume.node_md)).to(
        device, non_blocking=True
    )
    return values, truth, z_relative, distance_weight


def _weighted_mean_v3(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _loss_v3(output, truth_tvt, z_relative, node_weight):
    error = (output["solved_tvt"] - truth_tvt) / 8.0
    path_loss = _weighted_mean_v3(error.square(), node_weight)

    target = torch.exp(
        -0.5
        * (
            (STATE_OFFSETS_TENSOR_V3[:, None] - truth_tvt[None])
            / TARGET_SIGMA_FT_V3
        ).square()
    )
    target = target / target.sum(dim=0, keepdim=True).clamp_min(1e-8)
    column_ce = -(target * F.log_softmax(output["unary"], dim=0)).sum(dim=0)
    absolute_ce = _weighted_mean_v3(column_ce, node_weight)

    truth_s = truth_tvt + z_relative
    drift_losses = []
    drift_nll = []
    for horizon, (prediction, precision) in output["horizons"].items():
        if len(truth_s) <= horizon:
            continue
        target_delta = truth_s[horizon:] - truth_s[:-horizon]
        predicted_delta = prediction[: len(target_delta)]
        edge_weight = 0.5 * (
            node_weight[horizon:] + node_weight[:-horizon]
        )
        scale = max(4.0, 0.025 * float(horizon * NODE_STEP_FT_V2))
        residual = (predicted_delta - target_delta) / scale
        drift_losses.append(
            _weighted_mean_v3(F.smooth_l1_loss(
                residual, torch.zeros_like(residual), reduction="none"
            ), edge_weight)
        )
        edge_precision = precision[: len(target_delta)].clamp(0.03, 30.0)
        drift_nll.append(
            _weighted_mean_v3(
                0.5 * edge_precision * residual.square()
                - 0.5 * torch.log(edge_precision),
                edge_weight,
            )
        )
    drift_loss = torch.stack(drift_losses).mean()
    drift_precision_loss = 0.02 * torch.stack(drift_nll).mean()

    cycle_losses = []
    for small, large in zip(HORIZONS_V3[:-1], HORIZONS_V3[1:]):
        if large != 2 * small or len(truth_s) <= large:
            continue
        small_prediction = output["horizons"][small][0]
        large_prediction = output["horizons"][large][0]
        count = len(truth_s) - large
        composed = small_prediction[:count] + small_prediction[small : small + count]
        cycle_losses.append(
            F.smooth_l1_loss(
                large_prediction[:count] / 8.0,
                composed / 8.0,
            )
        )
    cycle_loss = (
        torch.stack(cycle_losses).mean()
        if cycle_losses
        else path_loss.new_zeros(())
    )

    loss = (
        LOSS_PATH_V3 * path_loss
        + LOSS_ABSOLUTE_CE_V3 * absolute_ce
        + LOSS_DRIFT_V3 * (drift_loss + drift_precision_loss)
        + LOSS_CYCLE_V3 * cycle_loss
    )
    return loss, {
        "path": path_loss.detach(),
        "absolute_ce": absolute_ce.detach(),
        "drift": drift_loss.detach(),
        "drift_precision": drift_precision_loss.detach(),
        "cycle": cycle_loss.detach(),
    }


@torch.no_grad()
def _predict_v3(model, volume):
    model.eval()
    values, _, z_relative, _ = _tensors_v3(volume)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(values, z_relative)
    prediction = output["solved_tvt"].float().cpu().numpy()
    prediction[0] = 0.0
    return prediction


def _evaluate_v3(model, volumes):
    payloads = []
    for well in sorted(volumes):
        prediction = _predict_v3(model, volumes[well])
        payloads.append(_path_payload_v2(volumes[well], prediction))
    return _summarise_payloads_v2(payloads)


def _train_epoch_v3(model, volumes, optimizer, scaler, epoch):
    model.train()
    wells = list(volumes)
    mean_eval_rows = float(np.mean([volumes[well].eval_rows for well in wells]))
    order = np.random.default_rng(SEED_V3 + 1009 * epoch).permutation(len(wells))
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "path": 0.0,
        "absolute_ce": 0.0,
        "drift": 0.0,
        "drift_precision": 0.0,
        "cycle": 0.0,
        "skipped": 0,
    }
    for step, position in enumerate(order, 1):
        volume = volumes[wells[int(position)]]
        values, truth, z_relative, node_weight = _tensors_v3(volume)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(values, z_relative)
            loss, components = _loss_v3(output, truth, z_relative, node_weight)
            loss = loss * volume.eval_rows / max(mean_eval_rows, 1.0)
            scaled_loss = loss / GRAD_ACCUM_WELLS_V3
        scaler.scale(scaled_loss).backward()
        update = step % GRAD_ACCUM_WELLS_V3 == 0 or step == len(order)
        if update:
            before = float(scaler.get_scale())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_V3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            totals["skipped"] += int(float(scaler.get_scale()) < before)
        totals["loss"] += float(loss.detach().cpu())
        for key, value in components.items():
            totals[key] += float(value.cpu())
        if step % 100 == 0 or step == len(order):
            print(
                f"DRIFT-V3 epoch={epoch} wells={step}/{len(order)} "
                f"loss={totals['loss']/step:.4f}",
                flush=True,
            )
    return {
        "train_loss": totals["loss"] / len(order),
        "train_path": totals["path"] / len(order),
        "train_absolute_ce": totals["absolute_ce"] / len(order),
        "train_drift": totals["drift"] / len(order),
        "train_drift_precision": totals["drift_precision"] / len(order),
        "train_cycle": totals["cycle"] / len(order),
        "skipped_steps": totals["skipped"],
    }


def _new_training_v3():
    _seed_v3(SEED_V3)
    model = MultiHorizonDriftCorrespondenceV3().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_V3, weight_decay=WEIGHT_DECAY_V3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_V3, eta_min=MIN_LEARNING_RATE_V3
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=512.0
    )
    return model, optimizer, scheduler, scaler


# ---------------------------------------------------------------------------
# Execute the folds 3/4/5 -> Fold 2 gateway only.
# ---------------------------------------------------------------------------


started_v3 = time.time()
if device.type != "cuda":
    raise RuntimeError("V3 requires CUDA")

# Remove Fold-2-selected per-well adapter from the evidence volume.  The v3
# gateway must not reuse Fold 2 to tune an upstream component.
selected_dose_original_v3 = float(selected_dose_v1)
selected_dose_v1 = 0.0

print(
    "DRIFT-V3 CONFIG",
    {
        "experiment": EXPERIMENT_V3,
        "sample_unit": "complete physical well",
        "train_folds": [3, 4, 5],
        "validation_fold": 2,
        "node_step_ft": NODE_STEP_FT_V2,
        "horizons_ft": HORIZON_FT_V3,
        "distance_balanced_loss": True,
        "prefix_adapter_dose": selected_dose_v1,
        "historical_adapter_dose": selected_dose_original_v3,
        "v2_source": str(V2_SOURCE_V3),
    },
    flush=True,
)

core_matcher_v3, matcher_history_v3 = _fit_mixed_v1(
    core_wells,
    mixed_epochs_v1,
    SEED_V3 + 10,
    "drift_v3_core_matcher",
)
matcher_history_v3.to_csv(OUT_DIR_V3 / "core_matcher_history.csv", index=False)
core_volumes_v3 = _build_volume_bank_v2(core_matcher_v3, core_wells, "v3_core345")
fold2_volumes_v3 = _build_volume_bank_v2(core_matcher_v3, tune_wells, "v3_fold2")

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
        "elapsed_minutes": (time.time() - started_v3) / 60.0,
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

pd.DataFrame(history_v3).to_csv(OUT_DIR_V3 / "fold2_training_history.csv", index=False)
model_v3.load_state_dict(best_state_v3)
best_metrics_v3, best_bywell_v3, best_payload_v3 = _evaluate_v3(
    model_v3, fold2_volumes_v3
)
best_bywell_v3.to_csv(OUT_DIR_V3 / "fold2_bywell.csv", index=False)
np.savez_compressed(OUT_DIR_V3 / "fold2_predictions.npz", **best_payload_v3)

# Matched controls use the same core matcher and frozen best drift model.
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
    "elapsed_minutes": (time.time() - started_v3) / 60.0,
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
