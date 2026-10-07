"""SEGN-R1: five-slot complete-well cost-map observability gateway.

This is a clean rewrite of the local evidence layer described in the SEGN
work note.  It deliberately stops before recursive rollout.  Folds 4/5 train
the cost map, a length-stratified subset of Fold 3 selects the epoch, and the
remaining Fold-3 wells confirm the frozen model.  Fold 2 and Fold 1 are never
loaded.

The five matching slots are:

1. learned bilinear TCN x typewell interaction;
2. pair-feature MLP;
3. learned multi-head cosine mixture;
4. raw candidate-energy MLP;
5. local empirical-CDF Wasserstein proxy.

An independent geometry-only prior is the sixth slot.  Geometry and recursive
state never enter the matching slots.  A learned soft gate combines normalized
slot maps.  Posterior mean is the primary readout and mode is diagnostic.

Prerequisites
-------------
1. Run ``build_native_dualref_all_records_v0_cell.py`` once.
2. Upload ``segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py`` to
   ``/root/autodl-tmp``.  Only its validated data/mask/chunk definitions are
   executed; its model and training tail are not run.
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
# Bootstrap only validated raw-record, official-mask, and coordinate plumbing.
# ---------------------------------------------------------------------------

SOURCE_CANDIDATES_R1 = (
    Path("/root/autodl-tmp/segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py"),
    Path.cwd() / "segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py",
)
SOURCE_PATH_R1 = next((path for path in SOURCE_CANDIDATES_R1 if path.exists()), None)
if SOURCE_PATH_R1 is None:
    raise FileNotFoundError(
        "Upload segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py to "
        "/root/autodl-tmp before running SEGN-R1."
    )

source_r1 = SOURCE_PATH_R1.read_text(encoding="utf-8")
marker_r1 = "\nstarted = time.time()\n_seed_everything(SEED)\nif REQUIRE_CUDA"
if marker_r1 not in source_r1:
    raise RuntimeError(f"Cannot find bootstrap marker in {SOURCE_PATH_R1}")
exec(compile(source_r1.split(marker_r1, 1)[0], str(SOURCE_PATH_R1), "exec"), globals())


EXPERIMENT_R1 = "segn_r1_fiveslot_costmap_fold3_gateway_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT_R1
MODEL_DIR_R1 = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR_R1.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS_R1 = (4, 5)
SELECTION_FOLD_R1 = 3
UNTOUCHED_FOLDS_R1 = (1, 2)
SELECTION_WELLS_R1 = 32

MAX_EPOCHS_R1 = 48
MIN_EPOCHS_R1 = 16
REPORT_EVERY_R1 = 4
PATIENCE_REPORTS_R1 = 4
TRAIN_CHUNKS_PER_WELL_R1 = 3
GRADIENT_ACCUMULATION_R1 = 6
LEARNING_RATE_R1 = 3.0e-4
MIN_LEARNING_RATE_R1 = 3.0e-5
WEIGHT_DECAY_R1 = 2.0e-4
GRADIENT_CLIP_R1 = 2.0
GATE_WARMUP_EPOCHS_R1 = 4

TARGET_SIGMA_R1 = 2.0
LOSS_MEAN_R1 = 0.10
LOSS_STRONG_SLOT_R1 = 0.08
LOSS_GATE_R1 = 0.05
GATE_MAX_MEAN_R1 = 0.80
GATE_MIN_EFFECTIVE_SLOTS_R1 = 2.0

EMBEDDING_R1 = 32
HEADS_R1 = 4
HEAD_DIM_R1 = 8
SLOT_NAMES_R1 = (
    "bilinear",
    "pair_mlp",
    "costmix",
    "evidence",
    "w1",
    "prior",
)
STRONG_SLOT_INDICES_R1 = (0, 2)
CDF_THRESHOLDS_R1 = np.linspace(-2.5, 2.5, 13, dtype=np.float32)
CDF_WINDOW_R1 = 33


# ---------------------------------------------------------------------------
# Five matching maps, one independent prior, and learned soft fusion.
# ---------------------------------------------------------------------------


class ResidualEncoderBlockR1(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                7,
                padding=3 * dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )

    def forward(self, values):
        return values + self.network(values)


class SharedGREncoderR1(nn.Module):
    """Shared native-resolution encoder with roughly 400-ft context."""

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(4, EMBEDDING_R1, 9, padding=4),
            nn.GroupNorm(4, EMBEDDING_R1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualEncoderBlockR1(EMBEDDING_R1, 1),
            ResidualEncoderBlockR1(EMBEDDING_R1, 4),
            ResidualEncoderBlockR1(EMBEDDING_R1, 16),
            ResidualEncoderBlockR1(EMBEDDING_R1, 32),
        )

    def forward(self, values):
        return self.blocks(self.input(values))


def _local_cdf_features_r1(values: torch.Tensor) -> torch.Tensor:
    """Sliding empirical CDF of normalized GR for a 1-D W1 proxy."""

    gr = values[:, 0]
    valid = values[:, 1].clamp(0.0, 1.0)
    thresholds = torch.as_tensor(
        CDF_THRESHOLDS_R1, device=values.device, dtype=values.dtype
    )
    indicator = (gr[:, None, :] <= thresholds[None, :, None]).to(values.dtype)
    indicator = indicator * valid[:, None, :]
    mass = F.avg_pool1d(
        indicator,
        kernel_size=CDF_WINDOW_R1,
        stride=1,
        padding=CDF_WINDOW_R1 // 2,
    )
    coverage = F.avg_pool1d(
        valid[:, None, :],
        kernel_size=CDF_WINDOW_R1,
        stride=1,
        padding=CDF_WINDOW_R1 // 2,
    )
    return mass / coverage.clamp_min(0.10)


class FiveSlotCostMapR1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedGREncoderR1()

        self.query_bilinear = nn.Conv1d(EMBEDDING_R1, HEAD_DIM_R1, 1)
        self.reference_bilinear = nn.Conv1d(EMBEDDING_R1, HEAD_DIM_R1, 1)
        self.bilinear_matrix = nn.Parameter(torch.eye(HEAD_DIM_R1))

        self.pair_mlp = nn.Sequential(
            nn.Conv2d(2 * EMBEDDING_R1, 48, 1),
            nn.GELU(),
            nn.Conv2d(48, 24, 1),
            nn.GELU(),
            nn.Conv2d(24, 1, 1),
        )

        self.query_heads = nn.Conv1d(EMBEDDING_R1, HEADS_R1 * HEAD_DIM_R1, 1)
        self.reference_heads = nn.Conv1d(EMBEDDING_R1, HEADS_R1 * HEAD_DIM_R1, 1)
        self.head_weights = nn.Parameter(torch.zeros(HEADS_R1))

        self.evidence_mlp = nn.Sequential(
            nn.Conv2d(8, 24, 1),
            nn.GELU(),
            nn.Conv2d(24, 12, 1),
            nn.GELU(),
            nn.Conv2d(12, 1, 1),
        )

        # Geometry is confined to this prior and the gate.  It cannot alter
        # any GR matching map.
        self.prior_head = nn.Sequential(
            nn.Linear(8, 24),
            nn.GELU(),
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )

        # For each slot: normalized entropy/sharpness and peak confidence,
        # followed by six legal per-step context features.
        self.gate = nn.Sequential(
            nn.Linear(2 * len(SLOT_NAMES_R1) + 6, 48),
            nn.GELU(),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, len(SLOT_NAMES_R1)),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.slot_log_scale = nn.Parameter(
            torch.log(torch.tensor([2.0, 1.0, 4.0, 2.0, 8.0, 1.0]))
        )

    @staticmethod
    def _normalise_map(logits, candidate_valid):
        valid = candidate_valid[:, None]
        values = logits.masked_fill(~valid, 0.0)
        count = valid.sum(dim=0).clamp_min(1).to(logits.dtype)
        mean = values.sum(dim=0) / count
        centred = (values - mean[None]) * valid
        variance = centred.square().sum(dim=0) / count
        normalised = centred / torch.sqrt(variance + 1.0e-4)[None]
        return normalised.masked_fill(~valid, -1.0e4)

    @staticmethod
    def _distribution_features(logits, states):
        probability = F.softmax(logits, dim=0)
        entropy = -(probability * torch.log(probability.clamp_min(1.0e-8))).sum(dim=0)
        entropy = entropy / math.log(max(logits.shape[0], 2))
        confidence = probability.max(dim=0).values
        expected = (probability * states[:, None]).sum(dim=0)
        return probability, entropy, confidence, expected

    def forward(
        self,
        horizontal,
        typewell,
        node_positions,
        candidate_positions,
        geometry,
        force_uniform_gate=False,
        return_aux=False,
    ):
        horizontal_embedding = self.encoder(horizontal)[0]
        typewell_embedding = self.encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(typewell_embedding, candidate_positions)

        query_raw = horizontal[0, :, node_positions]
        reference_raw, raw_axis_valid = _sample_axis(typewell[0], candidate_positions)
        candidate_valid = axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)

        # Slot 0: true bilinear interaction.
        query_b = self.query_bilinear(query[None])[0]
        reference_b = self.reference_bilinear(reference[None])[0]
        transformed_query = self.bilinear_matrix @ query_b
        bilinear = torch.einsum("dt,ds->st", transformed_query, reference_b)
        bilinear = bilinear / math.sqrt(HEAD_DIM_R1)

        # Slot 1: pair-only learned interaction.  Neither geometry nor a
        # candidate coordinate is present in this tensor.
        product = reference[:, :, None] * query[:, None, :]
        difference = torch.abs(reference[:, :, None] - query[:, None, :])
        pair_volume = torch.cat([product, difference], dim=0)[None]
        pair_mlp = self.pair_mlp(pair_volume).squeeze(0).squeeze(0)

        # Slot 2: a learned mixture of independent cosine heads.
        query_h = self.query_heads(query[None])[0].reshape(HEADS_R1, HEAD_DIM_R1, -1)
        reference_h = self.reference_heads(reference[None])[0].reshape(
            HEADS_R1, HEAD_DIM_R1, -1
        )
        query_h = F.normalize(query_h, dim=1, eps=1.0e-5)
        reference_h = F.normalize(reference_h, dim=1, eps=1.0e-5)
        cosine_heads = torch.einsum("hdt,hds->hst", query_h, reference_h)
        costmix = (
            F.softmax(self.head_weights, dim=0)[:, None, None] * cosine_heads
        ).sum(dim=0)

        # Slot 3: raw value/derivative candidate energy.
        signed_value = reference_raw[0, :, None] - query_raw[0, None, :]
        signed_d1 = reference_raw[2, :, None] - query_raw[2, None, :]
        signed_d2 = reference_raw[3, :, None] - query_raw[3, None, :]
        joint_valid = reference_raw[1, :, None] * query_raw[1, None, :]
        evidence_features = torch.stack(
            [
                signed_value,
                torch.abs(signed_value),
                signed_value.square(),
                signed_d1,
                torch.abs(signed_d1),
                signed_d2,
                torch.abs(signed_d2),
                joint_valid,
            ],
            dim=0,
        )[None]
        evidence = self.evidence_mlp(evidence_features).squeeze(0).squeeze(0)

        # Slot 4: local empirical-CDF W1 proxy.  For 1-D distributions the
        # Wasserstein-1 distance is the integral of absolute CDF difference.
        query_cdf = _local_cdf_features_r1(horizontal)[0, :, node_positions]
        reference_cdf_full = _local_cdf_features_r1(typewell)[0]
        reference_cdf, cdf_axis_valid = _sample_axis(
            reference_cdf_full, candidate_positions
        )
        candidate_valid = candidate_valid & cdf_axis_valid
        w1 = -torch.mean(
            torch.abs(reference_cdf[:, :, None] - query_cdf[:, None, :]), dim=0
        )

        # Slot 5: independent geometry-only Gaussian prior.
        prior_parameters = self.prior_head(geometry.transpose(0, 1))
        prior_mean = torch.tanh(prior_parameters[:, 0]) * (0.80 * STATE_HALF_WIDTH_FT)
        prior_scale = F.softplus(prior_parameters[:, 1]) * 12.0 + 4.0
        states = torch.as_tensor(STATE_OFFSETS, device=query.device, dtype=query.dtype)
        prior = -0.5 * (
            (states[:, None] - prior_mean[None]) / prior_scale[None]
        ).square() - torch.log(prior_scale[None])

        raw_maps = [bilinear, pair_mlp, costmix, evidence, w1, prior]
        slot_maps = torch.stack(
            [self._normalise_map(values, candidate_valid) for values in raw_maps], dim=0
        )
        slot_scales = torch.exp(self.slot_log_scale).clamp(0.25, 20.0)
        slot_maps = slot_maps * slot_scales[:, None, None]

        slot_features = [
            self._distribution_features(values, states) for values in slot_maps
        ]
        entropies = torch.stack([item[1] for item in slot_features], dim=1)
        confidences = torch.stack([item[2] for item in slot_features], dim=1)
        expected = torch.stack([item[3] for item in slot_features], dim=0)

        query_coverage = query_raw[1].clamp(0.0, 1.0)
        reference_coverage = candidate_valid.float().mean().expand_as(query_coverage)
        context = torch.stack(
            [
                geometry[1].clamp(-1.0, 1.0),
                geometry[3].clamp(-2.0, 2.0),
                geometry[4].clamp(-2.0, 2.0),
                geometry[5].clamp(0.0, 1.0),
                query_coverage,
                reference_coverage,
            ],
            dim=1,
        )
        gate_input = torch.cat([1.0 - entropies, confidences, context], dim=1)
        learned_gate = F.softmax(self.gate(gate_input), dim=1).transpose(0, 1)
        if force_uniform_gate:
            gate_weights = torch.full_like(learned_gate, 1.0 / len(SLOT_NAMES_R1))
        else:
            gate_weights = learned_gate

        fused = (gate_weights[:, None, :] * slot_maps).sum(dim=0)
        fused = fused.masked_fill(~candidate_valid[:, None], -1.0e4)
        if not return_aux:
            return fused
        return fused, {
            "slot_maps": slot_maps,
            "slot_expected": expected,
            "slot_entropy": entropies.transpose(0, 1),
            "slot_confidence": confidences.transpose(0, 1),
            "gate_weights": gate_weights,
            "candidate_valid": candidate_valid,
        }


# ---------------------------------------------------------------------------
# Loss, training, and posterior diagnostics.
# ---------------------------------------------------------------------------


def _target_distribution_r1(target, states):
    distribution = torch.exp(-0.5 * ((states[:, None] - target[None]) / TARGET_SIGMA_R1) ** 2)
    return distribution / distribution.sum(dim=0, keepdim=True).clamp_min(1.0e-8)


def _loss_r1(model, payload, device, force_uniform_gate=False):
    tensors = _tensor_payload(payload, device)
    fused, auxiliary = model(
        *tensors,
        force_uniform_gate=force_uniform_gate,
        return_aux=True,
    )
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=target.dtype)
    target_distribution = _target_distribution_r1(target, states)

    fused_log_probability = F.log_softmax(fused, dim=0)
    fused_ce = -(target_distribution * fused_log_probability).sum(dim=0).mean()
    fused_probability = F.softmax(fused, dim=0)
    fused_mean = (fused_probability * states[:, None]).sum(dim=0)
    mean_loss = F.smooth_l1_loss(fused_mean / 8.0, target / 8.0)

    strong_ce = []
    for index in STRONG_SLOT_INDICES_R1:
        slot_log_probability = F.log_softmax(auxiliary["slot_maps"][index], dim=0)
        strong_ce.append(
            -(target_distribution * slot_log_probability).sum(dim=0).mean()
        )
    strong_slot_loss = torch.stack(strong_ce).mean()

    gate_mean = auxiliary["gate_weights"].mean(dim=1)
    gate_entropy = -(gate_mean * torch.log(gate_mean.clamp_min(1.0e-8))).sum()
    minimum_entropy = math.log(GATE_MIN_EFFECTIVE_SLOTS_R1)
    entropy_penalty = F.relu(
        torch.as_tensor(minimum_entropy, device=device, dtype=gate_entropy.dtype)
        - gate_entropy
    ).square()
    concentration_penalty = F.relu(gate_mean.max() - GATE_MAX_MEAN_R1).square()
    gate_penalty = entropy_penalty + concentration_penalty

    total = (
        fused_ce
        + LOSS_MEAN_R1 * mean_loss
        + LOSS_STRONG_SLOT_R1 * strong_slot_loss
        + LOSS_GATE_R1 * gate_penalty
    )
    return total, {
        "fused_ce": float(fused_ce.detach()),
        "mean": float(mean_loss.detach()),
        "strong_slot": float(strong_slot_loss.detach()),
        "gate": float(gate_penalty.detach()),
        "effective_slots": float(torch.exp(gate_entropy).detach()),
    }


def _training_starts_r1(record, rng):
    starts = np.asarray(record.chunk_starts, np.int32)
    if len(starts) <= TRAIN_CHUNKS_PER_WELL_R1:
        return starts
    # Stratify over heel/middle/toe, then jitter within each stratum.
    boundaries = np.linspace(0, len(starts), TRAIN_CHUNKS_PER_WELL_R1 + 1)
    selected = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        lo = int(math.floor(left))
        hi = max(lo + 1, int(math.ceil(right)))
        selected.append(int(starts[int(rng.integers(lo, min(hi, len(starts))))]))
    return np.asarray(selected, np.int32)


def _train_epoch_r1(model, records, optimizer, scaler, device, epoch):
    model.train()
    rng = np.random.default_rng(SEED + 7919 * epoch)
    order = rng.permutation(len(records))
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "fused_ce": 0.0,
        "mean": 0.0,
        "strong_slot": 0.0,
        "gate": 0.0,
        "effective_slots": 0.0,
        "samples": 0,
        "skipped": 0,
        "optimizer_steps": 0,
    }
    amp_enabled = device.type == "cuda"
    force_uniform = epoch <= GATE_WARMUP_EPOCHS_R1

    for progress, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        for start in _training_starts_r1(record, rng):
            start = int(start)
            start_row = int(record.node_rows[start])
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]),
                previous_delta=0.0,
                previous_rate=0.0,
            )
            if np.max(np.abs(payload["target_state"])) > STATE_HALF_WIDTH_FT - 0.5:
                totals["skipped"] += 1
                continue
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, parts = _loss_r1(
                    model,
                    payload,
                    device,
                    force_uniform_gate=force_uniform,
                )
                scaled_loss = loss / GRADIENT_ACCUMULATION_R1
            if not torch.isfinite(loss):
                totals["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(scaled_loss).backward()
            totals["loss"] += float(loss.detach())
            for key in ("fused_ce", "mean", "strong_slot", "gate", "effective_slots"):
                totals[key] += parts[key]
            totals["samples"] += 1

            if totals["samples"] % GRADIENT_ACCUMULATION_R1 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                totals["optimizer_steps"] += 1

        if progress % 100 == 0 or progress == len(records):
            mean_loss = totals["loss"] / max(totals["samples"], 1)
            print(
                f"SEGN-R1 epoch={epoch} wells={progress}/{len(records)} "
                f"loss={mean_loss:.4f}",
                flush=True,
            )

    if totals["samples"] % GRADIENT_ACCUMULATION_R1:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        totals["optimizer_steps"] += 1

    denominator = max(totals["samples"], 1)
    return {
        key: (value / denominator if key not in {"samples", "skipped", "optimizer_steps"} else value)
        for key, value in totals.items()
    } | {"uniform_gate": bool(force_uniform)}


@torch.no_grad()
def _posterior_payload_r1(model, payload, device):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        fused, auxiliary = model(*tensors, return_aux=True)
    fused = fused.float()
    slot_maps = auxiliary["slot_maps"].float()
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=torch.float32)
    maps = torch.cat([fused[None], slot_maps], dim=0)
    probabilities = F.softmax(maps, dim=1)
    means = (probabilities * states[None, :, None]).sum(dim=1)
    modes = states[torch.argmax(probabilities, dim=1)]
    entropies = -(
        probabilities * torch.log(probabilities.clamp_min(1.0e-8))
    ).sum(dim=1)
    return {
        "means": means.cpu().numpy().astype(np.float64),
        "modes": modes.cpu().numpy().astype(np.float64),
        "entropies": entropies.cpu().numpy().astype(np.float64),
        "maps": maps.cpu().numpy().astype(np.float32),
        "gate": auxiliary["gate_weights"].float().cpu().numpy().astype(np.float64),
    }


def _metric_payload_r1(wells, truths, predictions, distances):
    truth = np.concatenate(truths).astype(np.float64)
    prediction = np.concatenate(predictions).astype(np.float64)
    distance = np.concatenate(distances).astype(np.float64)
    row_wells = np.concatenate(
        [np.repeat(well, len(values)) for well, values in zip(wells, truths)]
    )
    squared = (prediction - truth) ** 2
    per_well = (
        pd.DataFrame({"well": row_wells, "squared": squared})
        .groupby("well", sort=False)["squared"]
        .mean()
        .pow(0.5)
    )
    result = {
        "rows": int(len(truth)),
        "wells": int(len(per_well)),
        "exact_sse": float(squared.sum()),
        "rmse": float(np.sqrt(squared.mean())),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
    }
    strata = (
        ("near_0_500_rmse", distance <= 500.0),
        ("mid_500_1000_rmse", (distance > 500.0) & (distance <= 1000.0)),
        ("mid_1000_2000_rmse", (distance > 1000.0) & (distance <= 2000.0)),
        ("far_gt2000_rmse", distance > 2000.0),
    )
    for name, mask in strata:
        result[name] = float(np.sqrt(squared[mask].mean())) if int(mask.sum()) else math.nan
    return result


@torch.no_grad()
def _evaluate_r1(model, records, device, control="full", save_prefix=None):
    model.eval()
    variants = ["fused_mean", "fused_mode"]
    for slot in SLOT_NAMES_R1:
        variants.extend([f"{slot}_mean", f"{slot}_mode"])
    collectors = {
        variant: {"wells": [], "truths": [], "predictions": [], "distances": []}
        for variant in variants
    }
    local = {
        name: {"count": 0, "within5": 0, "ranks": []}
        for name in ("fused",) + SLOT_NAMES_R1
    }
    gate_sum = np.zeros(len(SLOT_NAMES_R1), np.float64)
    gate_columns = 0
    entropy_sum = {name: 0.0 for name in ("fused",) + SLOT_NAMES_R1}
    entropy_columns = {name: 0 for name in ("fused",) + SLOT_NAMES_R1}

    shuffled = list(records[1:]) + list(records[:1]) if len(records) > 1 else list(records)
    for number, (record, shuffled_record) in enumerate(zip(records, shuffled), 1):
        node_count = len(record.node_rows)
        sums = {variant: np.zeros(node_count, np.float64) for variant in variants}
        counts = {variant: np.zeros(node_count, np.float64) for variant in variants}
        for variant in variants:
            sums[variant][0] = float(record.truth[record.node_rows[0]])
            counts[variant][0] = 1.0

        for raw_start in record.chunk_starts:
            start = int(raw_start)
            start_row = int(record.node_rows[start])
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]),
                previous_delta=0.0,
                previous_rate=0.0,
                control=control,
                shuffled_typewell=shuffled_record,
            )
            posterior = _posterior_payload_r1(model, payload, device)
            target = payload["target_state"].astype(np.float64)
            end = start + len(target)
            node_slice = slice(start + 1, end + 1)
            anchor = float(payload["deployed_anchor"])

            map_names = ("fused",) + SLOT_NAMES_R1
            for map_index, map_name in enumerate(map_names):
                for readout, values in (
                    ("mean", posterior["means"][map_index]),
                    ("mode", posterior["modes"][map_index]),
                ):
                    variant = f"{map_name}_{readout}"
                    sums[variant][node_slice] += anchor + values
                    counts[variant][node_slice] += 1.0

                mode = posterior["modes"][map_index]
                local[map_name]["count"] += len(target)
                local[map_name]["within5"] += int(np.sum(np.abs(mode - target) <= 5.0))
                map_values = posterior["maps"][map_index]
                true_index = np.abs(STATE_OFFSETS[:, None] - target[None]).argmin(axis=0)
                true_score = map_values[true_index, np.arange(len(target))]
                rank = 1 + np.sum(map_values > true_score[None], axis=0)
                local[map_name]["ranks"].extend(rank.tolist())
                entropy_sum[map_name] += float(posterior["entropies"][map_index].sum())
                entropy_columns[map_name] += int(len(target))

            gate_sum += posterior["gate"].sum(axis=1)
            gate_columns += posterior["gate"].shape[1]

        eval_rows = record.eval_index
        support_md = record.md[record.node_rows].astype(np.float64)
        truth = record.truth[eval_rows].astype(np.float64)
        distance = (
            record.md[eval_rows].astype(np.float64)
            - float(record.md[record.anchor_index])
        )
        for variant in variants:
            valid = counts[variant] > 0
            if int(valid.sum()) < 2:
                raise RuntimeError(f"{record.well}: {variant} has too few supported nodes")
            node_prediction = sums[variant][valid] / counts[variant][valid]
            prediction = np.interp(
                record.md[eval_rows].astype(np.float64),
                support_md[valid],
                node_prediction,
            )
            collectors[variant]["wells"].append(record.well)
            collectors[variant]["truths"].append(truth)
            collectors[variant]["predictions"].append(prediction)
            collectors[variant]["distances"].append(distance)

        if number % 25 == 0 or number == len(records):
            print(f"SEGN-R1 evaluate {control} {number}/{len(records)}", flush=True)

    metrics = {}
    for variant, values in collectors.items():
        metrics[variant] = _metric_payload_r1(**values)
    for map_name, values in local.items():
        count = max(values["count"], 1)
        metrics[f"{map_name}_mean"]["local_top1_within5"] = float(
            values["within5"] / count
        )
        metrics[f"{map_name}_mean"]["median_truth_rank"] = float(
            np.median(values["ranks"])
        )
        metrics[f"{map_name}_mean"]["p90_truth_rank"] = float(
            np.quantile(values["ranks"], 0.90)
        )
        metrics[f"{map_name}_mean"]["mean_entropy"] = float(
            entropy_sum[map_name] / max(entropy_columns[map_name], 1)
        )

    gate_mean = gate_sum / max(gate_columns, 1)
    gate_entropy = -np.sum(gate_mean * np.log(np.clip(gate_mean, 1.0e-12, None)))
    diagnostics = {
        "gate_mean": {
            name: float(value) for name, value in zip(SLOT_NAMES_R1, gate_mean)
        },
        "gate_max_mean": float(np.max(gate_mean)),
        "effective_slots": float(np.exp(gate_entropy)),
        "mean_minus_mode_gain": float(
            metrics["fused_mode"]["rmse"] - metrics["fused_mean"]["rmse"]
        ),
    }

    if save_prefix is not None:
        rows = []
        # Keep exact row-level artifacts for the two deployment readouts only.
        # Per-slot path metrics are already preserved in the CSV; materialising
        # every slot/readout combination would multiply this parquet by 14.
        for variant in ("fused_mean", "fused_mode"):
            values = collectors[variant]
            for well, truth, prediction, distance in zip(
                values["wells"],
                values["truths"],
                values["predictions"],
                values["distances"],
            ):
                rows.append(
                    pd.DataFrame(
                        {
                            "well": well,
                            "variant": variant,
                            "truth": truth,
                            "prediction": prediction,
                            "distance": distance,
                        }
                    )
                )
        row_frame = pd.concat(rows, ignore_index=True)
        try:
            row_frame.to_parquet(
                OUT_DIR / f"{save_prefix}_rows.parquet", index=False
            )
        except Exception as error:
            # Some notebook images load two incompatible pandas/pyarrow
            # extension registries.  Audit serialization must never discard
            # an otherwise completed confirmation evaluation.
            print(
                f"SEGN-R1 parquet warning: {error!r}; saving compressed NPZ",
                flush=True,
            )
            np.savez_compressed(
                OUT_DIR / f"{save_prefix}_rows.npz",
                well=row_frame["well"].astype(str).to_numpy(dtype="U"),
                variant=row_frame["variant"].astype(str).to_numpy(dtype="U"),
                truth=row_frame["truth"].to_numpy(np.float32),
                prediction=row_frame["prediction"].to_numpy(np.float32),
                distance=row_frame["distance"].to_numpy(np.float32),
            )
        pd.DataFrame(metrics).T.to_csv(
            OUT_DIR / f"{save_prefix}_metrics.csv", index_label="variant"
        )
    return metrics, diagnostics


def _fit_r1(train_records, selection_records, device):
    model = FiveSlotCostMapR1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_R1, weight_decay=WEIGHT_DECAY_R1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_R1, eta_min=MIN_LEARNING_RATE_R1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_rmse = math.inf
    bad_reports = 0
    history = []
    started = time.time()

    for epoch in range(1, MAX_EPOCHS_R1 + 1):
        training = _train_epoch_r1(
            model, train_records, optimizer, scaler, device, epoch
        )
        scheduler.step()
        if epoch % REPORT_EVERY_R1 != 0 and epoch != MAX_EPOCHS_R1:
            continue
        selection_metrics, selection_diagnostics = _evaluate_r1(
            model, selection_records, device, control="full"
        )
        report = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            "selection_fused_mean_rmse": selection_metrics["fused_mean"]["rmse"],
            "selection_fused_mode_rmse": selection_metrics["fused_mode"]["rmse"],
            "selection_bilinear_rmse": selection_metrics["bilinear_mean"]["rmse"],
            "selection_costmix_rmse": selection_metrics["costmix_mean"]["rmse"],
            "selection_mean_minus_mode_gain": selection_diagnostics["mean_minus_mode_gain"],
            "selection_effective_slots": selection_diagnostics["effective_slots"],
            "selection_gate_max_mean": selection_diagnostics["gate_max_mean"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(report)
        print("SEGN-R1 EPOCH", report, flush=True)

        score = float(selection_metrics["fused_mean"]["rmse"])
        if score < best_rmse - 1.0e-5:
            best_rmse = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_reports = 0
            torch.save(
                {
                    "model": best_state,
                    "epoch": best_epoch,
                    "selection_rmse": best_rmse,
                    "state_offsets": STATE_OFFSETS,
                    "slot_names": SLOT_NAMES_R1,
                    "source": str(SOURCE_PATH_R1),
                },
                MODEL_DIR_R1 / "best_fold3_selection.pt",
            )
        else:
            bad_reports += 1
        if epoch >= MIN_EPOCHS_R1 and bad_reports >= PATIENCE_REPORTS_R1:
            print("SEGN-R1 early stopping", flush=True)
            break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    return model, best_epoch, best_rmse, history


# ---------------------------------------------------------------------------
# Fold-clean teacher-anchor gateway.
# ---------------------------------------------------------------------------


started_r1 = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R1")
device_r1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r1, fold_path_r1 = _load_folds()
train_wells_r1 = folds_r1.loc[
    folds_r1["fold"].isin(TRAIN_FOLDS_R1), "well"
].tolist()
fold3_wells_r1 = folds_r1.loc[
    folds_r1["fold"].eq(SELECTION_FOLD_R1), "well"
].tolist()
if len(train_wells_r1) != 309 or len(fold3_wells_r1) != 154:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3 sizes: {len(train_wells_r1)}/{len(fold3_wells_r1)}"
    )

records_r1 = {
    record.well: record
    for record in _load_records(
        sorted(set(train_wells_r1 + fold3_wells_r1)), "r1_fold345"
    )
}
train_records_r1 = [records_r1[well] for well in train_wells_r1]
fold3_records_r1 = [records_r1[well] for well in fold3_wells_r1]

# Length-stratified deterministic split: selection sees the full suffix-length
# range but confirmation remains the larger, untouched Fold-3 population.
order_r1 = np.argsort([record.eval_rows for record in fold3_records_r1])
selection_positions_r1 = np.unique(
    np.rint(np.linspace(0, len(order_r1) - 1, SELECTION_WELLS_R1)).astype(int)
)
selection_indices_r1 = set(order_r1[selection_positions_r1].tolist())
selection_records_r1 = [
    record for index, record in enumerate(fold3_records_r1) if index in selection_indices_r1
]
confirmation_records_r1 = [
    record for index, record in enumerate(fold3_records_r1) if index not in selection_indices_r1
]

guards_r1 = {
    "source": str(SOURCE_PATH_R1),
    "fold_source": str(fold_path_r1),
    "train_folds": list(TRAIN_FOLDS_R1),
    "train_wells": len(train_records_r1),
    "selection_fold": SELECTION_FOLD_R1,
    "selection_wells": len(selection_records_r1),
    "confirmation_wells": len(confirmation_records_r1),
    "fold2_touched": False,
    "fold1_touched": False,
    "state_candidates": len(STATE_OFFSETS),
    "state_half_width_ft": STATE_HALF_WIDTH_FT,
    "node_spacing_md_ft": NODE_SPACING_MD_FT,
    "chunk_target_ft": CHUNK_TARGET_NODES * NODE_SPACING_MD_FT,
    "target_axis_coverage": _target_axis_coverage(fold3_records_r1),
    "slots": list(SLOT_NAMES_R1),
    "parameters": int(sum(p.numel() for p in FiveSlotCostMapR1().parameters())),
}
print("SEGN-R1 GUARDS", json.dumps(guards_r1, indent=2), flush=True)

model_r1, best_epoch_r1, best_selection_rmse_r1, history_r1 = _fit_r1(
    train_records_r1, selection_records_r1, device_r1
)

confirmation_full_r1, confirmation_diagnostics_r1 = _evaluate_r1(
    model_r1,
    confirmation_records_r1,
    device_r1,
    control="full",
    save_prefix="fold3_confirmation_full",
)
confirmation_no_gr_r1, no_gr_diagnostics_r1 = _evaluate_r1(
    model_r1, confirmation_records_r1, device_r1, control="no_gr"
)
confirmation_shuffled_r1, shuffled_diagnostics_r1 = _evaluate_r1(
    model_r1, confirmation_records_r1, device_r1, control="shuffled_typewell"
)

slot_report_r1 = []
for slot_r1 in ("fused",) + SLOT_NAMES_R1:
    mean_metrics_r1 = confirmation_full_r1[f"{slot_r1}_mean"]
    mode_metrics_r1 = confirmation_full_r1[f"{slot_r1}_mode"]
    slot_report_r1.append(
        {
            "slot": slot_r1,
            "mean_rmse": mean_metrics_r1["rmse"],
            "mode_rmse": mode_metrics_r1["rmse"],
            "mean_gain_vs_mode": mode_metrics_r1["rmse"] - mean_metrics_r1["rmse"],
            "local_top1_within5": mean_metrics_r1.get("local_top1_within5"),
            "median_truth_rank": mean_metrics_r1.get("median_truth_rank"),
            "mean_entropy": mean_metrics_r1.get("mean_entropy"),
            "no_gr_mean_rmse": confirmation_no_gr_r1[f"{slot_r1}_mean"]["rmse"],
            "no_gr_degradation": (
                confirmation_no_gr_r1[f"{slot_r1}_mean"]["rmse"]
                - mean_metrics_r1["rmse"]
            ),
            "shuffled_mean_rmse": confirmation_shuffled_r1[f"{slot_r1}_mean"]["rmse"],
            "shuffled_degradation": (
                confirmation_shuffled_r1[f"{slot_r1}_mean"]["rmse"]
                - mean_metrics_r1["rmse"]
            ),
        }
    )
slot_report_frame_r1 = pd.DataFrame(slot_report_r1).sort_values("mean_rmse")
slot_report_frame_r1.to_csv(OUT_DIR / "slot_confirmation_report.csv", index=False)
print("SEGN-R1 SLOT REPORT")
print(slot_report_frame_r1.to_string(index=False), flush=True)

strong_slots_r1 = [slot_report_r1[index + 1] for index in STRONG_SLOT_INDICES_R1]
best_strong_slot_r1 = min(strong_slots_r1, key=lambda row: row["mean_rmse"])
checks_r1 = {
    "target_axis_coverage_ge_0p99": guards_r1["target_axis_coverage"] >= 0.99,
    "fused_teacher_anchor_rmse_le_6": confirmation_full_r1["fused_mean"]["rmse"] <= 6.0,
    "best_strong_slot_rmse_le_8": best_strong_slot_r1["mean_rmse"] <= 8.0,
    "best_strong_slot_no_gr_degradation_ge_0p5": (
        best_strong_slot_r1["no_gr_degradation"] >= 0.5
    ),
    "best_strong_slot_shuffled_degradation_ge_0p5": (
        best_strong_slot_r1["shuffled_degradation"] >= 0.5
    ),
    "posterior_mean_gain_vs_mode_ge_0p5": (
        confirmation_diagnostics_r1["mean_minus_mode_gain"] >= 0.5
    ),
    "gate_max_mean_le_0p8": (
        confirmation_diagnostics_r1["gate_max_mean"] <= GATE_MAX_MEAN_R1
    ),
    "effective_slots_ge_2": (
        confirmation_diagnostics_r1["effective_slots"]
        >= GATE_MIN_EFFECTIVE_SLOTS_R1
    ),
}
strict_pass_r1 = bool(all(checks_r1.values()))

verdict_r1 = {
    "experiment": EXPERIMENT_R1,
    "stage": "fold3_five_slot_teacher_anchor_costmap_gateway",
    "sample_unit": "chunks nested within complete physical wells",
    "deployable": False,
    "truth_usage": "teacher anchor for observability only",
    "legal_model_inputs_only": True,
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R1),
        "selection_fold": SELECTION_FOLD_R1,
        "selection_wells": len(selection_records_r1),
        "confirmation_wells": len(confirmation_records_r1),
        "fold2_touched": False,
        "fold1_touched": False,
        "selected_epoch": best_epoch_r1,
        "selected_rmse": best_selection_rmse_r1,
    },
    "architecture": {
        "matching_slots": list(SLOT_NAMES_R1[:-1]),
        "independent_prior_slot": SLOT_NAMES_R1[-1],
        "geometry_in_matching_slots": False,
        "fusion": "learned per-step soft gate over normalized cost maps",
        "primary_readout": "posterior mean",
        "diagnostic_readout": "posterior mode",
        "cdf_window_ft": CDF_WINDOW_R1,
    },
    "guards": guards_r1,
    "confirmation_fused_mean": confirmation_full_r1["fused_mean"],
    "confirmation_fused_mode": confirmation_full_r1["fused_mode"],
    "confirmation_diagnostics": confirmation_diagnostics_r1,
    "best_strong_matching_slot": best_strong_slot_r1,
    "slot_report": slot_report_r1,
    "controls": {
        "no_gr_fused_mean": confirmation_no_gr_r1["fused_mean"],
        "shuffled_typewell_fused_mean": confirmation_shuffled_r1["fused_mean"],
        "no_gr_diagnostics": no_gr_diagnostics_r1,
        "shuffled_diagnostics": shuffled_diagnostics_r1,
    },
    "checks": checks_r1,
    "strict_pass": strict_pass_r1,
    "checkpoint": str(MODEL_DIR_R1 / "best_fold3_selection.pt"),
    "elapsed_minutes": (time.time() - started_r1) / 60.0,
    "next": (
        "PASS -> freeze the five-slot emission and run anchor-jitter before recurrence"
        if strict_pass_r1
        else "FAIL -> inspect slot table; do not add recurrence to an unproven emission"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict_r1)
print("SEGN-R1 FINAL VERDICT", json.dumps(verdict_r1, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

if not strict_pass_r1:
    raise RuntimeError(json.dumps(verdict_r1, indent=2))
