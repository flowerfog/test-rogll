"""SEGN-min v5: paired anchor-jitter training with restricted evidence maps.

V4 measured only 7.4% recovery of injected anchor error, with no causal gain
over no-GR or shuffled-typewell controls.  V5 addresses that measured failure:

* remove geometry, recursion, and absolute-anchor channels from each evidence
  refiner; those legal inputs remain available only to the channel gate;
* train matched zero/jitter views of the same chunk;
* require the predicted relative path to translate by exactly -jitter;
* select epochs on 32 fixed Fold-3 wells and confirm on the remaining Fold-3
  wells before any rollout evaluation;
* never load Fold 2 or Fold 1.

This is still a mechanism gateway, not a deployment model.
"""

from __future__ import annotations

from pathlib import Path
import copy
import gc
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bootstrap V4/V3/V2/V1 definitions without running an earlier experiment.
# ---------------------------------------------------------------------------

V4_SOURCE_CANDIDATES = (
    Path("/root/autodl-tmp/segn_min_anchor_recovery_fold3_v4_cell.py"),
    Path.cwd() / "segn_min_anchor_recovery_fold3_v4_cell.py",
)
V4_SOURCE = next((path for path in V4_SOURCE_CANDIDATES if path.exists()), None)
if V4_SOURCE is None:
    raise FileNotFoundError(
        "Upload segn_min_anchor_recovery_fold3_v4_cell.py to /root/autodl-tmp."
    )
v4_source = V4_SOURCE.read_text(encoding="utf-8")
v4_marker = "\n# Fold-3-only execution."
if v4_marker not in v4_source:
    raise RuntimeError(f"Cannot find V4 execution marker in {V4_SOURCE}")
exec(compile(v4_source.split(v4_marker, 1)[0], str(V4_SOURCE), "exec"), globals())


EXPERIMENT_V5 = "segn_min_anchor_equivariance_fold3_v5"
OUT_DIR = OOF_ROOT / EXPERIMENT_V5
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS_V5 = 10
TRAIN_CHUNKS_PER_WELL_V5 = 2
SELECTION_WELLS_V5 = 32
SELECTION_CHUNKS_PER_WELL_V5 = 2
SELECTION_OFFSETS_V5 = np.asarray([-24.0, 0.0, 24.0], np.float64)
TRAIN_JITTERS_V5 = np.asarray([-48.0, -24.0, -12.0, 12.0, 24.0, 48.0], np.float64)
LEARNING_RATE_V5 = 1.0e-4
MIN_LEARNING_RATE_V5 = 2.0e-5
EQUIVARIANCE_WEIGHT_V5 = 1.0
CHANNEL_EQUIVARIANCE_WEIGHT_V5 = 0.25
PATIENCE_V5 = 3

CONFIRM_ZERO_RMSE_GATE_V5 = 5.5
CONFIRM_RECOVERY_SLOPE_GATE_V5 = 0.50
CONFIRM_CONTROL_DROP_GATE_V5 = 0.10
ROLLOUT_BASELINE_V3 = 14.737753252342435
ROLLOUT_GAIN_GATE_V5 = 0.50


class SEGNMinCostMapNet(nn.Module):
    """Two GR evidence maps; physical/recursive inputs only route the gate."""

    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleGREncoder()

        self.query_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)
        self.reference_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)
        self.bilinear_matrix = nn.Parameter(torch.eye(HEAD_DIM))
        self.query_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)
        self.reference_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)

        # Evidence refiners see only pairwise GR evidence, validity, and local
        # state/time coordinates.  They cannot reconstruct a path from the
        # recursive anchor or drilled geometry alone.
        common_channels = 2 + 3 + 2
        self.bilinear_refiner = RefinedCostChannel(1 + common_channels)
        self.cosine_refiner = RefinedCostChannel(HEADS + common_channels)

        self.gate = nn.Sequential(
            nn.Linear(12, GATE_HIDDEN),
            nn.GELU(),
            nn.Linear(GATE_HIDDEN, 12),
            nn.GELU(),
            nn.Linear(12, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.channel_log_scale = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _distribution_features(logits, states):
        probability = F.softmax(logits, dim=0)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=0)
        entropy = entropy / math.log(max(logits.shape[0], 2))
        sharpness = 1.0 - entropy
        confidence = probability.max(dim=0).values
        expected = (probability * states[:, None]).sum(dim=0)
        return sharpness, confidence, expected

    def forward(
        self,
        horizontal,
        typewell,
        node_positions,
        candidate_positions,
        geometry,
        channel_control="full",
        return_aux=False,
    ):
        horizontal_embedding = self.encoder(horizontal)[0]
        typewell_embedding = self.encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(typewell_embedding, candidate_positions)

        query_a = self.query_bilinear(query[None])[0]
        reference_a = self.reference_bilinear(reference[None])[0]
        transformed_query = self.bilinear_matrix @ query_a
        bilinear = torch.einsum("dt,ds->st", transformed_query, reference_a)
        bilinear = bilinear[None] / math.sqrt(HEAD_DIM)

        query_b = self.query_heads(query[None])[0].reshape(HEADS, HEAD_DIM, -1)
        reference_b = self.reference_heads(reference[None])[0].reshape(
            HEADS, HEAD_DIM, -1
        )
        query_b = F.normalize(query_b, dim=1, eps=1e-5)
        reference_b = F.normalize(reference_b, dim=1, eps=1e-5)
        cosine = torch.einsum("hdt,hds->hst", query_b, reference_b)

        query_raw = horizontal[0, :, node_positions]
        reference_raw, raw_axis_valid = _sample_axis(typewell[0], candidate_positions)
        value_difference = -torch.abs(
            reference_raw[0, :, None] - query_raw[0, None, :]
        )[None]
        d1_difference = -torch.abs(
            reference_raw[2, :, None] - query_raw[2, None, :]
        )[None]
        query_valid = query_raw[1][None, None, :].expand(
            1, len(candidate_positions), -1
        )
        reference_valid = reference_raw[1][None, :, None].expand(
            1, -1, query.shape[-1]
        )
        joint_valid = query_valid * reference_valid
        candidate_valid = axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)
        valid_volume = candidate_valid[:, None]

        state_coordinate = (
            torch.from_numpy(STATE_OFFSETS).to(query.device, query.dtype)[None, :, None]
            / STATE_HALF_WIDTH_FT
        ).expand(1, -1, query.shape[-1])
        chunk_coordinate = torch.linspace(
            0.0, 1.0, query.shape[-1], device=query.device, dtype=query.dtype
        )[None, None, :].expand(1, len(candidate_positions), -1)
        common = torch.cat(
            [
                value_difference,
                d1_difference,
                query_valid,
                reference_valid,
                joint_valid,
                state_coordinate,
                chunk_coordinate,
            ],
            dim=0,
        )[None]

        logits_a = self.bilinear_refiner(
            torch.cat([bilinear[None], common], dim=1)
        )
        logits_b = self.cosine_refiner(torch.cat([cosine[None], common], dim=1))
        logits_a = logits_a.masked_fill(~valid_volume, -1.0e4)
        logits_b = logits_b.masked_fill(~valid_volume, -1.0e4)

        states = torch.from_numpy(STATE_OFFSETS).to(query.device, query.dtype)
        sharp_a, conf_a, expected_a = self._distribution_features(logits_a, states)
        sharp_b, conf_b, expected_b = self._distribution_features(logits_b, states)
        provisional = 0.5 * (logits_a + logits_b)
        provisional_conf = F.softmax(provisional, dim=0).max(dim=0).values
        disagreement = torch.abs(expected_a - expected_b) / STATE_HALF_WIDTH_FT
        query_coverage = query_raw[1].clamp(0.0, 1.0)
        reference_coverage = candidate_valid.float().mean().expand_as(query_coverage)
        gate_features = torch.stack(
            [
                sharp_a,
                sharp_b,
                conf_a,
                conf_b,
                provisional_conf,
                disagreement,
                geometry[1].clamp(-1.0, 1.0),
                query_coverage,
                reference_coverage,
                geometry[3].clamp(-2.0, 2.0),
                geometry[6].clamp(-4.0, 4.0),
                geometry[7].clamp(-4.0, 4.0),
            ],
            dim=1,
        )
        gate_weights = F.softmax(self.gate(gate_features), dim=1).transpose(0, 1)
        channel_logits = torch.stack([logits_a, logits_b], dim=0)
        channel_logits = channel_logits * torch.exp(self.channel_log_scale).clamp(
            0.5, 10.0
        )[:, None, None]

        if channel_control == "no_bilinear":
            gate_weights = torch.stack(
                [torch.zeros_like(gate_weights[0]), torch.ones_like(gate_weights[1])]
            )
        elif channel_control == "no_cosine":
            gate_weights = torch.stack(
                [torch.ones_like(gate_weights[0]), torch.zeros_like(gate_weights[1])]
            )
        logits = (gate_weights[:, None, :] * channel_logits).sum(dim=0)
        logits = logits.masked_fill(~valid_volume, -1.0e4)
        if not return_aux:
            return logits
        return logits, {
            "gate_weights": gate_weights,
            "channel_logits": channel_logits,
            "channel_expected": torch.stack([expected_a, expected_b]),
        }


def _single_loss_v5(model, payload, device):
    tensors = _tensor_payload(payload, device)
    logits, auxiliary = model(*tensors, return_aux=True)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    target_distribution = torch.exp(
        -0.5 * ((states - target[None]) / TARGET_SIGMA_FT_MIN) ** 2
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=0, keepdim=True
    ).clamp_min(1e-8)
    ce = _soft_ce(logits, target_distribution)
    ce_a = _soft_ce(auxiliary["channel_logits"][0], target_distribution)
    ce_b = _soft_ce(auxiliary["channel_logits"][1], target_distribution)
    probability = F.softmax(logits, dim=0)
    expected = (probability * states).sum(dim=0)
    mean_loss = F.smooth_l1_loss(expected / 8.0, target / 8.0)
    evidence = ce + AUXILIARY_CHANNEL_CE * 0.5 * (ce_a + ce_b) + LOSS_MEAN * mean_loss
    return evidence, expected, auxiliary, {
        "ce": ce,
        "mean": mean_loss,
        "ce_a": ce_a,
        "ce_b": ce_b,
    }


def _paired_loss_v5(model, zero_payload, jitter_payload, jitter_ft, device):
    zero_loss, zero_expected, zero_aux, zero_parts = _single_loss_v5(
        model, zero_payload, device
    )
    jitter_loss, jitter_expected, jitter_aux, jitter_parts = _single_loss_v5(
        model, jitter_payload, device
    )
    target_translation = torch.full_like(zero_expected, -float(jitter_ft))
    equivariance = F.smooth_l1_loss(
        (jitter_expected - zero_expected) / 8.0,
        target_translation / 8.0,
    )
    channel_equivariance = F.smooth_l1_loss(
        (jitter_aux["channel_expected"] - zero_aux["channel_expected"]) / 8.0,
        target_translation[None] / 8.0,
    )
    total = (
        0.5 * (zero_loss + jitter_loss)
        + EQUIVARIANCE_WEIGHT_V5 * equivariance
        + CHANNEL_EQUIVARIANCE_WEIGHT_V5 * channel_equivariance
    )
    gate = 0.5 * (
        zero_aux["gate_weights"].mean(dim=1)
        + jitter_aux["gate_weights"].mean(dim=1)
    )
    return total, {
        "ce": 0.5 * (zero_parts["ce"] + jitter_parts["ce"]),
        "mean": 0.5 * (zero_parts["mean"] + jitter_parts["mean"]),
        "ce_bilinear": 0.5 * (zero_parts["ce_a"] + jitter_parts["ce_a"]),
        "ce_cosine": 0.5 * (zero_parts["ce_b"] + jitter_parts["ce_b"]),
        "equivariance": equivariance,
        "channel_equivariance": channel_equivariance,
        "gate": gate,
    }


def _training_starts_v5(record, rng):
    starts = np.asarray(record.chunk_starts, np.int64)
    starts = starts[starts > 0]
    if not len(starts):
        return starts
    count = min(TRAIN_CHUNKS_PER_WELL_V5, len(starts))
    return np.sort(rng.choice(starts, size=count, replace=False))


def _train_epoch_v5(model, records, optimizer, scaler, device, epoch):
    model.train()
    rng = np.random.default_rng(SEED + 1709 * epoch)
    order = rng.permutation(len(records))
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "mean": 0.0,
        "ce_bilinear": 0.0,
        "ce_cosine": 0.0,
        "equivariance": 0.0,
        "channel_equivariance": 0.0,
        "gate_bilinear": 0.0,
        "gate_cosine": 0.0,
        "pairs": 0,
        "skipped": 0,
    }
    optimizer_steps = 0
    amp_enabled = device.type == "cuda"

    for number, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        for start_value in _training_starts_v5(record, rng):
            start = int(start_value)
            start_row = int(record.node_rows[start])
            previous_start = _previous_start_v4(record, start)
            previous_row = int(record.node_rows[previous_start])
            md_delta = max(float(record.md[start_row] - record.md[previous_row]), 1.0)
            previous_delta = float(record.truth[start_row] - record.truth[previous_row])
            previous_rate = previous_delta / md_delta
            zero_payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]),
                previous_delta=previous_delta,
                previous_rate=previous_rate,
            )

            jitter_order = rng.permutation(TRAIN_JITTERS_V5)
            jitter_payload = None
            jitter_ft = None
            for candidate_jitter in jitter_order:
                candidate_payload = _chunk_payload(
                    record,
                    start,
                    deployed_anchor=float(record.truth[start_row])
                    + float(candidate_jitter),
                    previous_delta=previous_delta,
                    previous_rate=previous_rate,
                )
                if float(np.max(np.abs(candidate_payload["target_state"]))) <= STATE_HALF_WIDTH_FT - 0.5:
                    jitter_payload = candidate_payload
                    jitter_ft = float(candidate_jitter)
                    break
            if jitter_payload is None:
                totals["skipped"] += 1
                continue

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, parts = _paired_loss_v5(
                    model, zero_payload, jitter_payload, jitter_ft, device
                )
                scaled_loss = loss / GRADIENT_ACCUMULATION
            if not torch.isfinite(loss):
                totals["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(scaled_loss).backward()
            totals["loss"] += float(loss.detach())
            for key in (
                "ce",
                "mean",
                "ce_bilinear",
                "ce_cosine",
                "equivariance",
                "channel_equivariance",
            ):
                totals[key] += float(parts[key].detach())
            totals["gate_bilinear"] += float(parts["gate"][0].detach())
            totals["gate_cosine"] += float(parts["gate"][1].detach())
            totals["pairs"] += 1

            if totals["pairs"] % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        if number % 100 == 0 or number == len(order):
            print(
                f"ANCHOR-EQUIV train epoch={epoch} wells={number}/{len(order)} "
                f"loss={totals['loss'] / max(totals['pairs'], 1):.4f}",
                flush=True,
            )

    if totals["pairs"] % GRADIENT_ACCUMULATION:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    denominator = max(totals["pairs"], 1)
    return {
        key: totals[key] / denominator
        for key in (
            "loss",
            "ce",
            "mean",
            "ce_bilinear",
            "ce_cosine",
            "equivariance",
            "channel_equivariance",
            "gate_bilinear",
            "gate_cosine",
        )
    } | {
        "pairs": totals["pairs"],
        "skipped": totals["skipped"],
        "optimizer_steps": optimizer_steps,
    }


@torch.no_grad()
def _quick_recovery_v5(model, records, device):
    model.eval()
    residuals = []
    corrections = []
    offsets = []
    zero_residuals = []
    corrupted_residuals = []
    for record in records:
        starts = _select_chunk_starts_v4(record)
        if len(starts) > SELECTION_CHUNKS_PER_WELL_V5:
            indices = np.rint(
                np.linspace(0, len(starts) - 1, SELECTION_CHUNKS_PER_WELL_V5)
            ).astype(np.int64)
            starts = starts[np.unique(indices)]
        for start_value in starts:
            start = int(start_value)
            start_row = int(record.node_rows[start])
            previous_start = _previous_start_v4(record, start)
            previous_row = int(record.node_rows[previous_start])
            md_delta = max(float(record.md[start_row] - record.md[previous_row]), 1.0)
            previous_delta = float(record.truth[start_row] - record.truth[previous_row])
            previous_rate = previous_delta / md_delta
            for offset_ft in SELECTION_OFFSETS_V5:
                payload = _chunk_payload(
                    record,
                    start,
                    deployed_anchor=float(record.truth[start_row]) + float(offset_ft),
                    previous_delta=previous_delta,
                    previous_rate=previous_rate,
                )
                if float(np.max(np.abs(payload["target_state"]))) > STATE_HALF_WIDTH_FT - 0.5:
                    continue
                expected, _, _, _ = _predict_chunk_min(model, payload, device)
                target_rows = payload["target_rows"]
                true_delta = record.truth[target_rows] - float(record.truth[start_row])
                residual = float(offset_ft) + expected - true_delta
                correction = expected - true_delta
                residuals.extend(residual.tolist())
                corrections.extend(correction.tolist())
                offsets.extend([float(offset_ft)] * len(residual))
                if float(offset_ft) == 0.0:
                    zero_residuals.extend(residual.tolist())
                else:
                    corrupted_residuals.extend(residual.tolist())
    residuals = np.asarray(residuals, np.float64)
    corrections = np.asarray(corrections, np.float64)
    offsets = np.asarray(offsets, np.float64)
    nonzero = offsets != 0.0
    slope, intercept = _ols_slope_v4(-offsets[nonzero], corrections[nonzero])
    zero_values = np.asarray(zero_residuals, np.float64)
    corrupt_values = np.asarray(corrupted_residuals, np.float64)
    return {
        "rows": int(len(residuals)),
        "recovery_slope": slope,
        "recovery_intercept": intercept,
        "zero_offset_rmse": float(np.sqrt(np.mean(zero_values**2))),
        "corrupted_rmse": float(np.sqrt(np.mean(corrupt_values**2))),
        "objective": float(
            np.sqrt(np.mean(corrupt_values**2))
            + 0.25 * np.sqrt(np.mean(zero_values**2))
        ),
    }


def _load_partial_v3_v5(model, checkpoint):
    source = checkpoint["model"]
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    result = model.load_state_dict(compatible, strict=False)
    # The gate must relearn routing for the restricted evidence channels.
    nn.init.zeros_(model.gate[-1].weight)
    nn.init.zeros_(model.gate[-1].bias)
    return {
        "source_tensors": len(source),
        "loaded_tensors": len(compatible),
        "missing_tensors": len(result.missing_keys),
        "unexpected_tensors": len(result.unexpected_keys),
    }


def _final_recovery_summary_v5(nodes):
    control_summary = {
        control: _control_summary_v4(nodes.loc[nodes["control"].eq(control)])
        for control in CONTROLS_V4
    }
    offset_rows = []
    for (control, offset_ft), frame in nodes.groupby(
        ["control", "offset_ft"], sort=True
    ):
        offset_rows.append(
            {
                "control": control,
                "offset_ft": float(offset_ft),
                **_metric_row_v4(frame),
                "first_rmse": _position_rmse_v4(frame, "first"),
                "rebase_rmse": _position_rmse_v4(frame, "rebase"),
                "endpoint_rmse": _position_rmse_v4(frame, "endpoint"),
            }
        )
    return control_summary, pd.DataFrame(offset_rows)


# ---------------------------------------------------------------------------
# Nested Fold-3 mechanism gateway.
# ---------------------------------------------------------------------------

started_v5 = time.time()
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for V5")
device_v5 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_v5, fold_path_v5 = _load_folds()
train_wells_v5 = sorted(folds_v5.loc[folds_v5["fold"].isin(AB_TRAIN_FOLDS), "well"])
fold3_wells_v5 = sorted(folds_v5.loc[folds_v5["fold"].eq(AB_VALID_FOLD), "well"])
needed_v5 = sorted(set(train_wells_v5 + fold3_wells_v5))
records_v5 = {record.well: record for record in _load_records(needed_v5, "fold345_v5")}
train_records_v5 = [records_v5[well] for well in train_wells_v5]
fold3_records_v5 = [records_v5[well] for well in fold3_wells_v5]
if len(train_records_v5) < 300 or len(fold3_records_v5) < 150:
    raise RuntimeError(
        f"Unexpected V5 fold coverage: {len(train_records_v5)}/{len(fold3_records_v5)}"
    )

split_rng_v5 = np.random.default_rng(SEED + 5505)
selection_indices_v5 = set(
    split_rng_v5.choice(
        len(fold3_records_v5), size=SELECTION_WELLS_V5, replace=False
    ).tolist()
)
selection_records_v5 = [
    record
    for index, record in enumerate(fold3_records_v5)
    if index in selection_indices_v5
]
confirmation_records_v5 = [
    record
    for index, record in enumerate(fold3_records_v5)
    if index not in selection_indices_v5
]

checkpoint_v5 = torch.load(CHECKPOINT_PATH_V4, map_location="cpu", weights_only=False)
_seed_everything(SEED + 5505)
model_v5 = SEGNMinCostMapNet().to(device_v5)
warm_start_v5 = _load_partial_v3_v5(model_v5, checkpoint_v5)

optimizer_v5 = torch.optim.AdamW(
    model_v5.parameters(), lr=LEARNING_RATE_V5, weight_decay=WEIGHT_DECAY
)
scheduler_v5 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v5, T_max=EPOCHS_V5, eta_min=MIN_LEARNING_RATE_V5
)
scaler_v5 = torch.amp.GradScaler("cuda", enabled=device_v5.type == "cuda")

guards_v5 = {
    "source": str(V4_SOURCE),
    "warm_start_checkpoint": str(CHECKPOINT_PATH_V4),
    "warm_start": warm_start_v5,
    "fold_source": str(fold_path_v5),
    "train_folds": list(AB_TRAIN_FOLDS),
    "train_wells": len(train_records_v5),
    "selection_fold": AB_VALID_FOLD,
    "selection_wells": len(selection_records_v5),
    "confirmation_wells": len(confirmation_records_v5),
    "parameters": int(sum(parameter.numel() for parameter in model_v5.parameters())),
    "train_jitters_ft": TRAIN_JITTERS_V5.tolist(),
    "fold2_touched": False,
    "fold1_touched": False,
}
print("ANCHOR-EQUIV GUARDS", json.dumps(guards_v5, indent=2), flush=True)

history_v5 = []
best_state_v5 = copy.deepcopy(model_v5.state_dict())
best_epoch_v5 = 0
best_objective_v5 = math.inf
bad_epochs_v5 = 0

for epoch_v5 in range(1, EPOCHS_V5 + 1):
    training_v5 = _train_epoch_v5(
        model_v5,
        train_records_v5,
        optimizer_v5,
        scaler_v5,
        device_v5,
        epoch_v5,
    )
    scheduler_v5.step()
    selection_v5 = _quick_recovery_v5(
        model_v5, selection_records_v5, device_v5
    )
    report_v5 = {
        "epoch": epoch_v5,
        **{f"train_{key}": value for key, value in training_v5.items()},
        **{f"selection_{key}": value for key, value in selection_v5.items()},
        "learning_rate": float(optimizer_v5.param_groups[0]["lr"]),
        "elapsed_minutes": (time.time() - started_v5) / 60.0,
    }
    history_v5.append(report_v5)
    pd.DataFrame(history_v5).to_csv(OUT_DIR / "training_history.csv", index=False)
    print("ANCHOR-EQUIV EPOCH", report_v5, flush=True)
    if selection_v5["objective"] < best_objective_v5 - 1e-4:
        best_objective_v5 = selection_v5["objective"]
        best_epoch_v5 = epoch_v5
        best_state_v5 = copy.deepcopy(model_v5.state_dict())
        bad_epochs_v5 = 0
    else:
        bad_epochs_v5 += 1
    if bad_epochs_v5 >= PATIENCE_V5:
        print(f"ANCHOR-EQUIV early stopping at epoch {epoch_v5}", flush=True)
        break

model_v5.load_state_dict(best_state_v5)
checkpoint_out_v5 = MODEL_DIR / "best_anchor_equivariance.pt"
torch.save(
    {
        "model": best_state_v5,
        "epoch": best_epoch_v5,
        "state_offsets": STATE_OFFSETS,
        "architecture": "restricted_evidence_paired_anchor_jitter",
        "source": str(V4_SOURCE),
    },
    checkpoint_out_v5,
)

(
    confirmation_nodes_v5,
    confirmation_chunks_v5,
    confirmation_requested_v5,
    confirmation_successful_v5,
    _,
    _,
) = _run_anchor_recovery_v4(model_v5, confirmation_records_v5, device_v5)
confirmation_coverage_v5 = confirmation_successful_v5 / max(
    confirmation_requested_v5, 1
)
control_summary_v5, offset_summary_v5 = _final_recovery_summary_v5(
    confirmation_nodes_v5
)
confirmation_chunks_v5.to_csv(
    OUT_DIR / "confirmation_anchor_recovery_chunks.csv", index=False
)
offset_summary_v5.to_csv(
    OUT_DIR / "confirmation_anchor_recovery_by_offset.csv", index=False
)

full_recovery_v5 = control_summary_v5["full"]
no_gr_recovery_v5 = control_summary_v5["no_gr"]
shuffled_recovery_v5 = control_summary_v5["shuffled_typewell"]
mechanism_checks_v5 = {
    "payload_coverage_ge_0p80": confirmation_coverage_v5 >= 0.80,
    "zero_offset_rmse_le_5p5": full_recovery_v5["zero_offset_rmse"]
    <= CONFIRM_ZERO_RMSE_GATE_V5,
    "full_recovery_slope_ge_0p50": full_recovery_v5["recovery_slope"]
    >= CONFIRM_RECOVERY_SLOPE_GATE_V5,
    "no_gr_recovery_slope_drop_ge_0p10": full_recovery_v5["recovery_slope"]
    - no_gr_recovery_v5["recovery_slope"]
    >= CONFIRM_CONTROL_DROP_GATE_V5,
    "shuffled_recovery_slope_drop_ge_0p10": full_recovery_v5["recovery_slope"]
    - shuffled_recovery_v5["recovery_slope"]
    >= CONFIRM_CONTROL_DROP_GATE_V5,
}
mechanism_pass_v5 = all(mechanism_checks_v5.values())

mechanism_verdict_v5 = {
    "experiment": EXPERIMENT_V5,
    "stage": "fold3_confirmation_anchor_equivariance_gate",
    "sample_unit": "paired zero/jitter chunks nested within complete wells",
    "truth_usage": "training labels plus controlled diagnostic anchor jitter",
    "legal_model_inputs_only": True,
    "guards": guards_v5,
    "selected_epoch": best_epoch_v5,
    "selection_objective": best_objective_v5,
    "confirmation_payload_coverage": confirmation_coverage_v5,
    "confirmation": control_summary_v5,
    "checks": mechanism_checks_v5,
    "strict_pass": mechanism_pass_v5,
    "fold2_touched": False,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - started_v5) / 60.0,
    "next": (
        "PASS -> evaluate complete Fold-3 recursive rollout"
        if mechanism_pass_v5
        else "FAIL -> absolute GR correspondence remains unlearned; stop SEGN recurrence work"
    ),
}
_json_v4(OUT_DIR / "mechanism_verdict.json", mechanism_verdict_v5)
print("\nANCHOR-EQUIV CONFIRMATION")
print(offset_summary_v5.to_string(index=False))
print("\nANCHOR-EQUIV MECHANISM VERDICT")
print(json.dumps(mechanism_verdict_v5, indent=2), flush=True)

if not mechanism_pass_v5:
    raise RuntimeError(json.dumps(mechanism_verdict_v5, indent=2))

# Only a confirmed recovery mechanism is allowed into recursive evaluation.
rollout_v5, _ = _evaluate_min(
    model_v5,
    fold3_records_v5,
    device_v5,
    anchor_mode="rollout",
    readout="mean",
    save_prefix="fold3_rollout_mean",
)
teacher_v5, _ = _evaluate_min(
    model_v5,
    fold3_records_v5,
    device_v5,
    anchor_mode="teacher",
    readout="mean",
    save_prefix="fold3_teacher_mean",
)
rollout_no_gr_v5, _ = _evaluate_min(
    model_v5,
    fold3_records_v5,
    device_v5,
    anchor_mode="rollout",
    readout="mean",
    control="no_gr",
    save_prefix="fold3_rollout_no_gr",
)
rollout_shuffled_v5, _ = _evaluate_min(
    model_v5,
    fold3_records_v5,
    device_v5,
    anchor_mode="rollout",
    readout="mean",
    control="shuffled_typewell",
    save_prefix="fold3_rollout_shuffled_typewell",
)
rollout_gain_v5 = ROLLOUT_BASELINE_V3 - rollout_v5["rmse"]
rollout_checks_v5 = {
    "gain_vs_v3_ge_0p50": rollout_gain_v5 >= ROLLOUT_GAIN_GATE_V5,
    "no_gr_degradation_ge_0p30": rollout_no_gr_v5["rmse"]
    - rollout_v5["rmse"]
    >= CONTROL_GATE_FT,
    "shuffled_degradation_ge_0p30": rollout_shuffled_v5["rmse"]
    - rollout_v5["rmse"]
    >= CONTROL_GATE_FT,
}
final_verdict_v5 = {
    **mechanism_verdict_v5,
    "stage": "fold3_recursive_rollout_gateway",
    "mechanism_strict_pass": True,
    "rollout": rollout_v5,
    "teacher_anchor": teacher_v5,
    "rollout_no_gr": rollout_no_gr_v5,
    "rollout_shuffled_typewell": rollout_shuffled_v5,
    "gain_vs_v3": rollout_gain_v5,
    "rollout_checks": rollout_checks_v5,
    "strict_pass": all(rollout_checks_v5.values()),
    "elapsed_minutes": (time.time() - started_v5) / 60.0,
    "next": (
        "PASS -> anchor correction transfers to rollout; prepare untouched Fold-2 refit"
        if all(rollout_checks_v5.values())
        else "FAIL -> correction is local but does not survive recursive rollout"
    ),
}
_json_v4(OUT_DIR / "verdict.json", final_verdict_v5)
print("\nANCHOR-EQUIV FINAL VERDICT")
print(json.dumps(final_verdict_v5, indent=2), flush=True)
print("saved:", OUT_DIR)
