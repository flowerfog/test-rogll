"""SEGN-R4: work-note-faithful scheduled cost-map gateway.

This experiment fills the one combination missing from the previous SEGN
audits: the two matching channels retained by the work note, an independent
geometry/drilled-trend prior, a learned continuous gate, posterior-mean
decoding, and online scheduled autoregression are trained together.

Protocol
--------
* Folds 4/5 train matched teacher-forcing and scheduled-sampling arms.
* A deterministic 32-well subset of Fold 3 selects each arm's epoch.
* The remaining Fold-3 wells are a frozen confirmation population.
* Fold 2 and Fold 1 are not loaded by this cell.
* All model inputs are same-well horizontal GR/geometry, visible-prefix TVT,
  and that well's assigned prefix-calibrated typewell.  No PF, SP45, spatial
  neighbour, strong, CP12, Gold, or validation-truth feature is used at
  deployment time.

Prerequisites
-------------
1. Run ``build_native_dualref_all_records_v0_cell.py`` once.
2. Upload ``segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py`` and its
   prerequisite ``segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py`` to
   ``/root/autodl-tmp`` (or keep both in the current directory).
"""

from __future__ import annotations

from pathlib import Path
import copy
import gc
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reuse only validated record, official-mask, chunk, encoder, and metric code.
# The R1 training tail is deliberately not executed.
# ---------------------------------------------------------------------------

SOURCE_CANDIDATES_R4 = (
    Path("/root/autodl-tmp/segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py"),
    Path.cwd() / "segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py",
)
SOURCE_PATH_R4 = next((path for path in SOURCE_CANDIDATES_R4 if path.exists()), None)
if SOURCE_PATH_R4 is None:
    raise FileNotFoundError(
        "Upload segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py to "
        "/root/autodl-tmp before running SEGN-R4."
    )

source_r4 = SOURCE_PATH_R4.read_text(encoding="utf-8")
marker_r4 = "\nstarted_r1 = time.time()"
if marker_r4 not in source_r4:
    raise RuntimeError(f"Cannot find bootstrap marker in {SOURCE_PATH_R4}")
exec(
    compile(source_r4.split(marker_r4, 1)[0], str(SOURCE_PATH_R4), "exec"),
    globals(),
)


EXPERIMENT_R4 = "segn_r4_worknote_scheduled_costmap_fold3_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT_R4
MODEL_DIR_R4 = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR_R4.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS_R4 = (4, 5)
VALID_FOLD_R4 = 3
UNTOUCHED_FOLDS_R4 = (1, 2)
SELECTION_WELLS_R4 = 32

MAX_EPOCHS_R4 = 24
MIN_EPOCHS_R4 = 12
REPORT_EVERY_R4 = 4
PATIENCE_REPORTS_R4 = 3
SEQUENCE_CHUNKS_R4 = 6
SEQUENCES_PER_WELL_R4 = 1
SCHEDULE_MAX_EPSILON_R4 = 0.50
SCHEDULE_WARMUP_EPOCHS_R4 = 12
GRADIENT_ACCUMULATION_R4 = 6
LEARNING_RATE_R4 = 3.0e-4
MIN_LEARNING_RATE_R4 = 3.0e-5
WEIGHT_DECAY_R4 = 2.0e-4
GRADIENT_CLIP_R4 = 2.0

TARGET_SIGMA_R4 = 2.0
LOSS_MEAN_R4 = 0.10
LOSS_MATCHING_R4 = 0.05
PRIOR_GATE_BIAS_R4 = 2.0

SLOT_NAMES_R4 = ("bilinear", "cosine", "prior")
MATCHING_SLOT_INDICES_R4 = (0, 1)


# ---------------------------------------------------------------------------
# Work-note core: two localisers, independent prior, continuous soft trust.
# ---------------------------------------------------------------------------


class SEGNWorknoteR4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedGREncoderR1()

        self.query_bilinear = nn.Conv1d(EMBEDDING_R1, HEAD_DIM_R1, 1)
        self.reference_bilinear = nn.Conv1d(EMBEDDING_R1, HEAD_DIM_R1, 1)
        self.bilinear_matrix = nn.Parameter(torch.eye(HEAD_DIM_R1))

        self.query_heads = nn.Conv1d(
            EMBEDDING_R1, HEADS_R1 * HEAD_DIM_R1, 1
        )
        self.reference_heads = nn.Conv1d(
            EMBEDDING_R1, HEADS_R1 * HEAD_DIM_R1, 1
        )
        self.head_weights = nn.Parameter(torch.zeros(HEADS_R1))

        # The prior sees legal geometry and recursive state only.  It never
        # sees GR and therefore remains a matched fallback for the controls.
        self.prior_head = nn.Sequential(
            nn.Linear(8, 24),
            nn.GELU(),
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )

        # Six posterior-quality features plus agreement, distance, geometry,
        # coverage, and recursive-state features.  No hard routing is used.
        self.gate = nn.Sequential(
            nn.Linear(14, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, len(SLOT_NAMES_R4)),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.gate[-1].bias[-1] = PRIOR_GATE_BIAS_R4

        self.slot_log_scale = nn.Parameter(
            torch.log(torch.tensor([2.0, 4.0, 1.0], dtype=torch.float32))
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
        entropy = -(
            probability * torch.log(probability.clamp_min(1.0e-8))
        ).sum(dim=0)
        entropy = entropy / math.log(max(logits.shape[0], 2))
        confidence = probability.max(dim=0).values
        expected = (probability * states[:, None]).sum(dim=0)
        return 1.0 - entropy, confidence, expected

    def forward(
        self,
        horizontal,
        typewell,
        node_positions,
        candidate_positions,
        geometry,
        return_aux=False,
    ):
        horizontal_embedding = self.encoder(horizontal)[0]
        typewell_embedding = self.encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(
            typewell_embedding, candidate_positions
        )

        query_raw = horizontal[0, :, node_positions]
        reference_raw, raw_axis_valid = _sample_axis(
            typewell[0], candidate_positions
        )
        candidate_valid = (
            axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)
        )

        query_b = self.query_bilinear(query[None])[0]
        reference_b = self.reference_bilinear(reference[None])[0]
        transformed_query = self.bilinear_matrix @ query_b
        bilinear = torch.einsum("dt,ds->st", transformed_query, reference_b)
        bilinear = bilinear / math.sqrt(HEAD_DIM_R1)

        query_h = self.query_heads(query[None])[0].reshape(
            HEADS_R1, HEAD_DIM_R1, -1
        )
        reference_h = self.reference_heads(reference[None])[0].reshape(
            HEADS_R1, HEAD_DIM_R1, -1
        )
        query_h = F.normalize(query_h, dim=1, eps=1.0e-5)
        reference_h = F.normalize(reference_h, dim=1, eps=1.0e-5)
        cosine_heads = torch.einsum("hdt,hds->hst", query_h, reference_h)
        cosine = (
            F.softmax(self.head_weights, dim=0)[:, None, None] * cosine_heads
        ).sum(dim=0)

        prior_parameters = self.prior_head(geometry.transpose(0, 1))
        prior_mean = torch.tanh(prior_parameters[:, 0]) * (
            0.80 * STATE_HALF_WIDTH_FT
        )
        prior_scale = F.softplus(prior_parameters[:, 1]) * 12.0 + 4.0
        states = torch.as_tensor(
            STATE_OFFSETS, device=query.device, dtype=query.dtype
        )
        prior = -0.5 * (
            (states[:, None] - prior_mean[None]) / prior_scale[None]
        ).square() - torch.log(prior_scale[None])

        raw_maps = (bilinear, cosine, prior)
        slot_maps = torch.stack(
            [
                self._normalise_map(values, candidate_valid)
                for values in raw_maps
            ],
            dim=0,
        )
        scales = torch.exp(self.slot_log_scale).clamp(0.25, 20.0)
        slot_maps = slot_maps * scales[:, None, None]

        features = [
            self._distribution_features(values, states)
            for values in slot_maps
        ]
        sharpness = torch.stack([item[0] for item in features], dim=1)
        confidence = torch.stack([item[1] for item in features], dim=1)
        expected = torch.stack([item[2] for item in features], dim=0)
        matching_disagreement = (
            torch.abs(expected[0] - expected[1]) / STATE_HALF_WIDTH_FT
        )

        query_coverage = query_raw[1].clamp(0.0, 1.0)
        reference_coverage = candidate_valid.float().mean().expand_as(
            query_coverage
        )
        gate_context = torch.stack(
            [
                matching_disagreement,
                geometry[1].clamp(-1.0, 1.0),
                geometry[3].clamp(-2.0, 2.0),
                geometry[4].clamp(-2.0, 2.0),
                query_coverage,
                reference_coverage,
                geometry[6].clamp(-4.0, 4.0),
                geometry[7].clamp(-4.0, 4.0),
            ],
            dim=1,
        )
        gate_input = torch.cat([sharpness, confidence, gate_context], dim=1)
        gate_weights = F.softmax(self.gate(gate_input), dim=1).transpose(0, 1)
        fused = (gate_weights[:, None, :] * slot_maps).sum(dim=0)
        fused = fused.masked_fill(~candidate_valid[:, None], -1.0e4)

        if not return_aux:
            return fused
        return fused, {
            "slot_maps": slot_maps,
            "slot_expected": expected,
            "gate_weights": gate_weights,
            "candidate_valid": candidate_valid,
        }


def _target_distribution_r4(target, states):
    distribution = torch.exp(
        -0.5 * ((states[:, None] - target[None]) / TARGET_SIGMA_R4) ** 2
    )
    return distribution / distribution.sum(
        dim=0, keepdim=True
    ).clamp_min(1.0e-8)


def _loss_r4(model, payload, device):
    tensors = _tensor_payload(payload, device)
    fused, auxiliary = model(*tensors, return_aux=True)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.as_tensor(
        STATE_OFFSETS, device=device, dtype=target.dtype
    )
    target_distribution = _target_distribution_r4(target, states)

    fused_log_probability = F.log_softmax(fused, dim=0)
    fused_ce = -(
        target_distribution * fused_log_probability
    ).sum(dim=0).mean()
    fused_probability = F.softmax(fused, dim=0)
    fused_mean = (fused_probability * states[:, None]).sum(dim=0)
    mean_loss = F.smooth_l1_loss(fused_mean / 8.0, target / 8.0)

    matching_ce = []
    for index in MATCHING_SLOT_INDICES_R4:
        log_probability = F.log_softmax(
            auxiliary["slot_maps"][index], dim=0
        )
        matching_ce.append(
            -(target_distribution * log_probability).sum(dim=0).mean()
        )
    matching_loss = torch.stack(matching_ce).mean()
    total = (
        fused_ce
        + LOSS_MEAN_R4 * mean_loss
        + LOSS_MATCHING_R4 * matching_loss
    )
    return total, {
        "fused_ce": float(fused_ce.detach()),
        "mean": float(mean_loss.detach()),
        "matching": float(matching_loss.detach()),
        "expected": fused_mean.detach(),
        "gate_weights": auxiliary["gate_weights"].detach(),
    }


@torch.no_grad()
def _predict_chunk_r4(model, payload, device, variant="fused"):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        fused, auxiliary = model(*tensors, return_aux=True)
    if variant == "fused":
        logits = fused.float()
    elif variant == "prior":
        logits = auxiliary["slot_maps"][2].float()
    elif variant == "bilinear":
        logits = auxiliary["slot_maps"][0].float()
    elif variant == "cosine":
        logits = auxiliary["slot_maps"][1].float()
    else:
        raise KeyError(variant)
    probability = F.softmax(logits, dim=0)
    states = torch.as_tensor(
        STATE_OFFSETS, device=device, dtype=torch.float32
    )
    expected = (probability * states[:, None]).sum(dim=0)
    mode = STATE_OFFSETS[torch.argmax(probability, dim=0).cpu().numpy()]
    entropy = -(
        probability * torch.log(probability.clamp_min(1.0e-8))
    ).sum(dim=0)
    return (
        expected.cpu().numpy().astype(np.float64),
        mode.astype(np.float64),
        entropy.cpu().numpy().astype(np.float64),
        auxiliary["gate_weights"].float().cpu().numpy().astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Matched teacher/scheduled training.  Both arms see identical wells, starts,
# initial weights, and optimiser hyperparameters; only anchor sampling differs.
# ---------------------------------------------------------------------------


def _sequence_starts_r4(record, rng):
    starts = np.asarray(record.chunk_starts, np.int32)
    if len(starts) <= SEQUENCE_CHUNKS_R4:
        return starts
    first = int(rng.integers(0, len(starts) - SEQUENCE_CHUNKS_R4 + 1))
    return starts[first : first + SEQUENCE_CHUNKS_R4]


def _train_epoch_r4(
    model, records, optimizer, scaler, device, epoch, scheduled
):
    model.train()
    rng = np.random.default_rng(SEED + 1009 * epoch)
    schedule_rng = np.random.default_rng(SEED + 1009 * epoch + 7919)
    order = rng.permutation(len(records))
    epsilon = (
        min(
            SCHEDULE_MAX_EPSILON_R4,
            SCHEDULE_MAX_EPSILON_R4
            * epoch
            / SCHEDULE_WARMUP_EPOCHS_R4,
        )
        if scheduled
        else 0.0
    )
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "fused_ce": 0.0,
        "mean": 0.0,
        "matching": 0.0,
        "samples": 0,
        "model_anchor_samples": 0,
        "skipped": 0,
        "optimizer_steps": 0,
        **{f"gate_{name}": 0.0 for name in SLOT_NAMES_R4},
    }
    amp_enabled = device.type == "cuda"

    for progress, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        for _ in range(SEQUENCES_PER_WELL_R4):
            starts = _sequence_starts_r4(record, rng)
            predicted_nodes = np.full(
                len(record.node_rows), np.nan, np.float64
            )
            first_start = int(starts[0])
            predicted_nodes[first_start] = float(
                record.truth[int(record.node_rows[first_start])]
            )
            if first_start > 0:
                prior_candidates = np.asarray(record.chunk_starts, np.int32)
                prior_candidates = prior_candidates[prior_candidates < first_start]
                previous_start = (
                    int(prior_candidates[-1]) if len(prior_candidates) else 0
                )
                previous_anchor = float(
                    record.truth[int(record.node_rows[previous_start])]
                )
            else:
                previous_start = first_start
                previous_anchor = float(predicted_nodes[first_start])

            for sequence_step, raw_start in enumerate(starts):
                start = int(raw_start)
                start_row = int(record.node_rows[start])
                true_anchor = float(record.truth[start_row])
                use_model_anchor = (
                    sequence_step > 0
                    and np.isfinite(predicted_nodes[start])
                    and schedule_rng.random() < epsilon
                )
                deployed_anchor = (
                    float(predicted_nodes[start])
                    if use_model_anchor
                    else true_anchor
                )
                if use_model_anchor:
                    totals["model_anchor_samples"] += 1

                previous_row = int(record.node_rows[previous_start])
                md_delta = max(
                    float(record.md[start_row] - record.md[previous_row]), 1.0
                )
                previous_delta = (
                    deployed_anchor - previous_anchor
                    if start > 0
                    else 0.0
                )
                previous_rate = (
                    previous_delta / md_delta if start > 0 else 0.0
                )
                payload = _chunk_payload(
                    record,
                    start,
                    deployed_anchor=deployed_anchor,
                    previous_delta=previous_delta,
                    previous_rate=previous_rate,
                )
                if np.max(np.abs(payload["target_state"])) > (
                    STATE_HALF_WIDTH_FT - 0.5
                ):
                    totals["skipped"] += 1
                    previous_start = start
                    previous_anchor = deployed_anchor
                    continue

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss, parts = _loss_r4(model, payload, device)
                    scaled_loss = loss / GRADIENT_ACCUMULATION_R4
                if not torch.isfinite(loss):
                    totals["skipped"] += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(scaled_loss).backward()

                expected = parts["expected"].float().cpu().numpy()
                end = start + len(expected)
                predicted_nodes[start + 1 : end + 1] = (
                    deployed_anchor + expected
                )
                gate = parts["gate_weights"].float().cpu().numpy()
                totals["loss"] += float(loss.detach())
                for key in ("fused_ce", "mean", "matching"):
                    totals[key] += parts[key]
                for index, name in enumerate(SLOT_NAMES_R4):
                    totals[f"gate_{name}"] += float(gate[index].mean())
                totals["samples"] += 1

                if totals["samples"] % GRADIENT_ACCUMULATION_R4 == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRADIENT_CLIP_R4
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    totals["optimizer_steps"] += 1
                previous_start = start
                previous_anchor = deployed_anchor

        if progress % 100 == 0 or progress == len(order):
            print(
                f"SEGN-R4 train arm={'scheduled' if scheduled else 'teacher'} "
                f"epoch={epoch} epsilon={epsilon:.3f} "
                f"wells={progress}/{len(order)} "
                f"loss={totals['loss'] / max(totals['samples'], 1):.4f}",
                flush=True,
            )

    if totals["samples"] % GRADIENT_ACCUMULATION_R4:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R4)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        totals["optimizer_steps"] += 1

    denominator = max(totals["samples"], 1)
    result = {
        key: value / denominator
        for key, value in totals.items()
        if key
        in {
            "loss",
            "fused_ce",
            "mean",
            "matching",
            *{f"gate_{name}" for name in SLOT_NAMES_R4},
        }
    }
    result.update(
        {
            "samples": totals["samples"],
            "model_anchor_samples": totals["model_anchor_samples"],
            "model_anchor_fraction": (
                totals["model_anchor_samples"] / denominator
            ),
            "skipped": totals["skipped"],
            "optimizer_steps": totals["optimizer_steps"],
            "epsilon": epsilon,
        }
    )
    return result


@torch.no_grad()
def _predict_record_r4(
    model,
    record,
    device,
    anchor_mode="rollout",
    readout="mean",
    variant="fused",
    control="full",
    shuffled_record=None,
):
    model.eval()
    node_prediction = np.full(len(record.node_rows), np.nan, np.float64)
    node_prediction[0] = float(record.anchor_tvt)
    previous_start = 0
    previous_anchor = float(record.anchor_tvt)
    entropy_values = []
    disagreement_values = []
    gate_values = []

    for raw_start in record.chunk_starts:
        start = int(raw_start)
        start_row = int(record.node_rows[start])
        if anchor_mode == "teacher":
            deployed_anchor = float(record.truth[start_row])
        elif np.isfinite(node_prediction[start]):
            deployed_anchor = float(node_prediction[start])
        else:
            prior_nodes = np.flatnonzero(
                np.isfinite(node_prediction[: start + 1])
            )
            if not len(prior_nodes):
                raise RuntimeError(f"{record.well}: rollout lost its anchor")
            deployed_anchor = float(node_prediction[prior_nodes[-1]])

        previous_row = int(record.node_rows[previous_start])
        md_delta = max(
            float(record.md[start_row] - record.md[previous_row]), 1.0
        )
        previous_delta = (
            deployed_anchor - previous_anchor if start > 0 else 0.0
        )
        previous_rate = previous_delta / md_delta if start > 0 else 0.0
        payload = _chunk_payload(
            record,
            start,
            deployed_anchor=deployed_anchor,
            previous_delta=previous_delta,
            previous_rate=previous_rate,
            control=control,
            shuffled_typewell=shuffled_record,
        )
        expected, mode, entropy, gate = _predict_chunk_r4(
            model, payload, device, variant=variant
        )
        selected = expected if readout == "mean" else mode
        end = start + len(selected)
        node_prediction[start + 1 : end + 1] = deployed_anchor + selected
        entropy_values.extend(entropy.tolist())
        disagreement_values.extend(np.abs(expected - mode).tolist())
        gate_values.append(gate.mean(axis=1))
        previous_start = start
        previous_anchor = deployed_anchor

    valid = np.isfinite(node_prediction)
    if int(valid.sum()) < 2:
        raise RuntimeError(f"{record.well}: too few rollout nodes")
    eval_rows = record.eval_index
    query_md = record.md[eval_rows].astype(np.float64)
    prediction = np.interp(
        query_md,
        record.md[record.node_rows[valid]].astype(np.float64),
        node_prediction[valid],
    )
    truth = record.truth[eval_rows].astype(np.float64)
    distance = query_md - float(record.md[record.anchor_index])
    gate_mean = (
        np.mean(np.stack(gate_values), axis=0)
        if gate_values
        else np.full(len(SLOT_NAMES_R4), np.nan)
    )
    diagnostics = {
        "mean_entropy": (
            float(np.mean(entropy_values)) if entropy_values else math.nan
        ),
        "mean_decode_disagreement": (
            float(np.mean(disagreement_values))
            if disagreement_values
            else math.nan
        ),
        **{
            f"gate_{name}": float(value)
            for name, value in zip(SLOT_NAMES_R4, gate_mean)
        },
    }
    return truth, prediction, distance, diagnostics


def _shape_metrics_r4(wells, truths, predictions, anchors):
    truth = np.concatenate(truths).astype(np.float64)
    prediction = np.concatenate(predictions).astype(np.float64)
    anchor = np.concatenate(
        [np.full(len(values), value) for values, value in zip(truths, anchors)]
    )
    true_offset = truth - anchor
    predicted_offset = prediction - anchor

    def correlation(left, right):
        if np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
            return math.nan
        return float(np.corrcoef(left, right)[0, 1])

    denominator = float(np.dot(true_offset, true_offset))
    slope_pred_on_truth = (
        float(np.dot(true_offset, predicted_offset) / denominator)
        if denominator > 1.0e-12
        else math.nan
    )
    design = np.column_stack([np.ones(len(predicted_offset)), predicted_offset])
    coefficients, _, _, _ = np.linalg.lstsq(design, true_offset, rcond=None)
    affine_offset = design @ coefficients
    affine_rmse = float(np.sqrt(np.mean((affine_offset - true_offset) ** 2)))

    centred_truth = []
    centred_prediction = []
    per_well_correlation = []
    for true_values, predicted_values in zip(truths, predictions):
        true_values = np.asarray(true_values, np.float64)
        predicted_values = np.asarray(predicted_values, np.float64)
        centred_truth.append(true_values - true_values.mean())
        centred_prediction.append(predicted_values - predicted_values.mean())
        value = correlation(true_values, predicted_values)
        if np.isfinite(value):
            per_well_correlation.append(value)

    return {
        "pooled_offset_correlation": correlation(
            true_offset, predicted_offset
        ),
        "slope_predicted_offset_on_true_offset": slope_pred_on_truth,
        "offset_bias_ft": float(np.mean(predicted_offset - true_offset)),
        "within_well_shape_correlation": correlation(
            np.concatenate(centred_truth),
            np.concatenate(centred_prediction),
        ),
        "median_well_shape_correlation": (
            float(np.median(per_well_correlation))
            if per_well_correlation
            else math.nan
        ),
        "affine_oracle_intercept_ft": float(coefficients[0]),
        "affine_oracle_gain": float(coefficients[1]),
        "affine_oracle_rmse": affine_rmse,
    }


@torch.no_grad()
def _evaluate_r4(
    model,
    records,
    device,
    anchor_mode="rollout",
    readout="mean",
    variant="fused",
    control="full",
    save_prefix=None,
):
    wells = []
    truths = []
    predictions = []
    distances = []
    anchors = []
    bywell_rows = []
    shuffled = (
        list(records[1:]) + list(records[:1])
        if len(records) > 1
        else list(records)
    )

    for number, (record, shuffled_record) in enumerate(
        zip(records, shuffled), 1
    ):
        truth, prediction, distance, diagnostics = _predict_record_r4(
            model,
            record,
            device,
            anchor_mode=anchor_mode,
            readout=readout,
            variant=variant,
            control=control,
            shuffled_record=shuffled_record,
        )
        wells.append(record.well)
        truths.append(truth)
        predictions.append(prediction)
        distances.append(distance)
        anchors.append(float(record.anchor_tvt))
        bywell_rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "exact_sse": float(np.sum((prediction - truth) ** 2)),
                "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                "mae": float(np.mean(np.abs(prediction - truth))),
                **diagnostics,
            }
        )
        if number % 25 == 0 or number == len(records):
            print(
                f"SEGN-R4 evaluate {anchor_mode}/{readout}/{variant}/{control} "
                f"{number}/{len(records)}",
                flush=True,
            )

    metrics = _metric_payload_r1(wells, truths, predictions, distances)
    metrics.update(_shape_metrics_r4(wells, truths, predictions, anchors))
    bywell = pd.DataFrame(bywell_rows)
    for name in SLOT_NAMES_R4:
        metrics[f"gate_{name}"] = float(bywell[f"gate_{name}"].mean())
    metrics["mean_entropy"] = float(bywell["mean_entropy"].mean())
    metrics["mean_decode_disagreement"] = float(
        bywell["mean_decode_disagreement"].mean()
    )

    if save_prefix:
        bywell.to_csv(OUT_DIR / f"{save_prefix}_bywell.csv", index=False)
        np.savez_compressed(
            OUT_DIR / f"{save_prefix}_rows.npz",
            well=np.concatenate(
                [np.repeat(well, len(values)) for well, values in zip(wells, truths)]
            ).astype("U"),
            truth=np.concatenate(truths).astype(np.float32),
            prediction=np.concatenate(predictions).astype(np.float32),
            distance_past_anchor=np.concatenate(distances).astype(np.float32),
            anchor_tvt=np.concatenate(
                [np.full(len(values), value) for values, value in zip(truths, anchors)]
            ).astype(np.float32),
        )
    return metrics, bywell


def _state_checksum_r4(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _fit_arm_r4(train_records, selection_records, device, scheduled, phase):
    # Both A/B arms deliberately reset to the same initial state.
    _seed_everything(SEED + 4242)
    model = SEGNWorknoteR4().to(device)
    initial_checksum = _state_checksum_r4(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_R4, weight_decay=WEIGHT_DECAY_R4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_R4, eta_min=MIN_LEARNING_RATE_R4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_rmse = math.inf
    bad_reports = 0
    history = []
    started_arm = time.time()

    for epoch in range(1, MAX_EPOCHS_R4 + 1):
        training = _train_epoch_r4(
            model,
            train_records,
            optimizer,
            scaler,
            device,
            epoch,
            scheduled,
        )
        scheduler.step()
        report = {
            "phase": phase,
            "arm": "scheduled" if scheduled else "teacher",
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started_arm) / 60.0,
        }
        if epoch % REPORT_EVERY_R4 == 0 or epoch == MAX_EPOCHS_R4:
            selection, _ = _evaluate_r4(
                model,
                selection_records,
                device,
                anchor_mode="rollout",
                readout="mean",
                variant="fused",
            )
            report.update(
                {f"selection_{key}": value for key, value in selection.items()}
            )
            if selection["rmse"] < best_rmse - 1.0e-4:
                best_rmse = float(selection["rmse"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_reports = 0
            else:
                bad_reports += 1
        history.append(report)
        pd.DataFrame(history).to_csv(
            OUT_DIR / f"{phase}_history.csv", index=False
        )
        print("SEGN-R4 EPOCH", report, flush=True)
        if epoch >= MIN_EPOCHS_R4 and bad_reports >= PATIENCE_REPORTS_R4:
            print(f"SEGN-R4 {phase} early stopping at epoch {epoch}", flush=True)
            break

    if best_epoch == 0:
        raise RuntimeError(f"{phase}: no selection checkpoint was produced")
    model.load_state_dict(best_state)
    checkpoint = MODEL_DIR_R4 / f"{phase}_best.pt"
    torch.save(
        {
            "model": best_state,
            "epoch": best_epoch,
            "scheduled": scheduled,
            "selection_rmse": best_rmse,
            "initial_checksum": initial_checksum,
            "state_offsets": STATE_OFFSETS,
            "slot_names": SLOT_NAMES_R4,
            "source": str(SOURCE_PATH_R4),
        },
        checkpoint,
    )
    return model, best_epoch, best_rmse, initial_checksum, checkpoint


# ---------------------------------------------------------------------------
# Fold-3 matched confirmation.  This cell intentionally stops before Fold 2.
# ---------------------------------------------------------------------------


started_r4 = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R4")
device_r4 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r4, fold_path_r4 = _load_folds()
train_wells_r4 = folds_r4.loc[
    folds_r4["fold"].isin(TRAIN_FOLDS_R4), "well"
].tolist()
fold3_wells_r4 = folds_r4.loc[
    folds_r4["fold"].eq(VALID_FOLD_R4), "well"
].tolist()
if len(train_wells_r4) != 309 or len(fold3_wells_r4) != 154:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3 sizes: "
        f"{len(train_wells_r4)}/{len(fold3_wells_r4)}"
    )

records_r4 = {
    record.well: record
    for record in _load_records(
        sorted(set(train_wells_r4 + fold3_wells_r4)), "r4_fold345"
    )
}
train_records_r4 = [records_r4[well] for well in train_wells_r4]
fold3_records_r4 = [records_r4[well] for well in fold3_wells_r4]

order_r4 = np.argsort([record.eval_rows for record in fold3_records_r4])
selection_positions_r4 = np.unique(
    np.rint(
        np.linspace(0, len(order_r4) - 1, SELECTION_WELLS_R4)
    ).astype(int)
)
selection_indices_r4 = set(order_r4[selection_positions_r4].tolist())
selection_records_r4 = [
    record
    for index, record in enumerate(fold3_records_r4)
    if index in selection_indices_r4
]
confirmation_records_r4 = [
    record
    for index, record in enumerate(fold3_records_r4)
    if index not in selection_indices_r4
]

guards_r4 = {
    "source": str(SOURCE_PATH_R4),
    "fold_source": str(fold_path_r4),
    "train_folds": list(TRAIN_FOLDS_R4),
    "train_wells": len(train_records_r4),
    "selection_fold": VALID_FOLD_R4,
    "selection_wells": len(selection_records_r4),
    "confirmation_wells": len(confirmation_records_r4),
    "fold2_touched": False,
    "fold1_touched": False,
    "state_candidates": len(STATE_OFFSETS),
    "state_half_width_ft": STATE_HALF_WIDTH_FT,
    "node_spacing_md_ft": NODE_SPACING_MD_FT,
    "chunk_target_ft": CHUNK_TARGET_NODES * NODE_SPACING_MD_FT,
    "chunk_stride_ft": CHUNK_STRIDE_NODES * NODE_SPACING_MD_FT,
    "target_axis_coverage": _target_axis_coverage(fold3_records_r4),
    "slots": list(SLOT_NAMES_R4),
    "parameters": int(sum(p.numel() for p in SEGNWorknoteR4().parameters())),
    "typewell_calibration": "native-record visible-prefix affine calibration",
}
print("SEGN-R4 GUARDS", json.dumps(guards_r4, indent=2), flush=True)

teacher_model_r4, teacher_epoch_r4, teacher_selection_r4, teacher_init_r4, teacher_ckpt_r4 = _fit_arm_r4(
    train_records_r4,
    selection_records_r4,
    device_r4,
    scheduled=False,
    phase="teacher45_to_3",
)
teacher_confirmation_r4, _ = _evaluate_r4(
    teacher_model_r4,
    confirmation_records_r4,
    device_r4,
    anchor_mode="rollout",
    readout="mean",
    variant="fused",
    save_prefix="fold3_teacher_confirmation_rollout_mean",
)
del teacher_model_r4
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

scheduled_model_r4, scheduled_epoch_r4, scheduled_selection_r4, scheduled_init_r4, scheduled_ckpt_r4 = _fit_arm_r4(
    train_records_r4,
    selection_records_r4,
    device_r4,
    scheduled=True,
    phase="scheduled45_to_3",
)
if teacher_init_r4 != scheduled_init_r4:
    raise RuntimeError(
        "Matched A/B guard failed: teacher and scheduled initial states differ"
    )

scheduled_confirmation_r4, _ = _evaluate_r4(
    scheduled_model_r4,
    confirmation_records_r4,
    device_r4,
    anchor_mode="rollout",
    readout="mean",
    variant="fused",
    save_prefix="fold3_scheduled_confirmation_rollout_mean",
)
scheduled_mode_r4, _ = _evaluate_r4(
    scheduled_model_r4,
    confirmation_records_r4,
    device_r4,
    anchor_mode="rollout",
    readout="mode",
    variant="fused",
)
scheduled_teacher_anchor_r4, _ = _evaluate_r4(
    scheduled_model_r4,
    confirmation_records_r4,
    device_r4,
    anchor_mode="teacher",
    readout="mean",
    variant="fused",
)
prior_confirmation_r4, _ = _evaluate_r4(
    scheduled_model_r4,
    confirmation_records_r4,
    device_r4,
    anchor_mode="rollout",
    readout="mean",
    variant="prior",
)
no_gr_confirmation_r4, _ = _evaluate_r4(
    scheduled_model_r4,
    confirmation_records_r4,
    device_r4,
    anchor_mode="rollout",
    readout="mean",
    variant="fused",
    control="no_gr",
)
shuffled_confirmation_r4, _ = _evaluate_r4(
    scheduled_model_r4,
    confirmation_records_r4,
    device_r4,
    anchor_mode="rollout",
    readout="mean",
    variant="fused",
    control="shuffled_typewell",
)

schedule_gain_r4 = (
    teacher_confirmation_r4["rmse"] - scheduled_confirmation_r4["rmse"]
)
matching_gain_r4 = (
    prior_confirmation_r4["rmse"] - scheduled_confirmation_r4["rmse"]
)
mean_gain_r4 = scheduled_mode_r4["rmse"] - scheduled_confirmation_r4["rmse"]
rollout_gap_r4 = (
    scheduled_confirmation_r4["rmse"] - scheduled_teacher_anchor_r4["rmse"]
)
no_gr_degradation_r4 = (
    no_gr_confirmation_r4["rmse"] - scheduled_confirmation_r4["rmse"]
)
shuffled_degradation_r4 = (
    shuffled_confirmation_r4["rmse"] - scheduled_confirmation_r4["rmse"]
)

checks_r4 = {
    "matched_initial_state": teacher_init_r4 == scheduled_init_r4,
    "target_axis_coverage_ge_0p99": guards_r4["target_axis_coverage"] >= 0.99,
    "scheduled_confirmation_rmse_le_10p5": scheduled_confirmation_r4["rmse"] <= 10.5,
    "scheduled_gain_vs_teacher_ge_0p10": schedule_gain_r4 >= 0.10,
    "matching_gain_vs_prior_ge_0p20": matching_gain_r4 >= 0.20,
    "posterior_mean_better_than_mode": mean_gain_r4 > 0.0,
    "pooled_offset_correlation_ge_0p75": scheduled_confirmation_r4[
        "pooled_offset_correlation"
    ] >= 0.75,
    "no_gr_degradation_ge_0p20": no_gr_degradation_r4 >= 0.20,
    "shuffled_typewell_degradation_ge_0p30": shuffled_degradation_r4 >= 0.30,
}
strict_pass_r4 = bool(all(checks_r4.values()))

verdict_r4 = {
    "experiment": EXPERIMENT_R4,
    "stage": "fold3_worknote_complete_scheduled_costmap_gateway",
    "sample_unit": "autoregressive chunk sequences nested within complete wells",
    "deployable_inputs": True,
    "legal_model_inputs_only": True,
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "worknote_contract": {
        "candidate_grid": "193 anchor-relative 1-ft states over +/-96 ft",
        "matching_channels": ["bilinear TCN x typewell", "multi-head cosine"],
        "base_prior": "independent same-well geometry/drilled-trend Gaussian prior",
        "typewell_calibration": "visible-prefix affine calibration inside cost-map reference",
        "fusion": "learned per-step continuous soft gate",
        "decode": "posterior mean primary; mode diagnostic",
        "recurrence": "512-ft prediction chunks with 256-ft stride",
        "exposure_control": "online scheduled sampling to epsilon=0.5",
        "loss": "soft-bin CE plus small posterior-mean Huber; no CRPS",
        "dp_viterbi": False,
    },
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R4),
        "selection_fold": VALID_FOLD_R4,
        "selection_wells": len(selection_records_r4),
        "confirmation_wells": len(confirmation_records_r4),
        "fold2_touched": False,
        "fold1_touched": False,
        "teacher_epoch": teacher_epoch_r4,
        "scheduled_epoch": scheduled_epoch_r4,
        "teacher_selection_rmse": teacher_selection_r4,
        "scheduled_selection_rmse": scheduled_selection_r4,
        "matched_initial_checksum": teacher_init_r4,
    },
    "guards": guards_r4,
    "confirmation": {
        "teacher_rollout_mean": teacher_confirmation_r4,
        "scheduled_rollout_mean": scheduled_confirmation_r4,
        "scheduled_rollout_mode": scheduled_mode_r4,
        "scheduled_teacher_anchor_mean": scheduled_teacher_anchor_r4,
        "scheduled_prior_only_rollout": prior_confirmation_r4,
        "scheduled_no_gr_rollout": no_gr_confirmation_r4,
        "scheduled_shuffled_typewell_rollout": shuffled_confirmation_r4,
    },
    "diagnostics": {
        "scheduled_gain_vs_teacher": schedule_gain_r4,
        "matching_gain_vs_prior": matching_gain_r4,
        "posterior_mean_gain_vs_mode": mean_gain_r4,
        "rollout_gap_vs_teacher_anchor": rollout_gap_r4,
        "no_gr_degradation": no_gr_degradation_r4,
        "shuffled_typewell_degradation": shuffled_degradation_r4,
    },
    "checks": checks_r4,
    "strict_pass": strict_pass_r4,
    "checkpoints": {
        "teacher": str(teacher_ckpt_r4),
        "scheduled": str(scheduled_ckpt_r4),
    },
    "elapsed_minutes": (time.time() - started_r4) / 60.0,
    "next": (
        "PASS -> freeze R4, refit the scheduled recipe on Folds 3/4/5, and apply once to untouched Fold 2"
        if strict_pass_r4
        else "FAIL -> use the failed check to classify evidence, recurrence, or fusion; do not touch Fold 2"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict_r4)
print("SEGN-R4 FINAL VERDICT", json.dumps(verdict_r4, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

if not strict_pass_r4:
    raise RuntimeError(json.dumps(verdict_r4, indent=2))
