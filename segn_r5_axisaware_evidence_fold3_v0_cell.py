"""SEGN-R5: axis-aware, query-causal evidence gateway.

R4 proved that scheduled sampling and posterior-mean decoding help, but its
no-GR control was better than the full model.  The matching channels therefore
did not establish causal horizontal-GR correspondence.  This cell returns to
steps 1-2 of the SEGN work note and tests evidence before recurrence:

* horizontal GR is encoded on the MD axis with about 423-ft context;
* typewell GR is encoded separately on the TVT axis with about 99-ft context;
* all signal convolutions/projections are bias-free and normalization has no
  affine offset, so a zero horizontal-GR query gives uniform valid logits;
* only the work note's bilinear and multi-head cosine channels are retained;
* a learned continuous two-channel gate and posterior mean are evaluated under
  teacher anchors; no prior or recursive rollout can hide weak evidence.

Folds 4/5 train, a deterministic 32-well Fold-3 subset selects the epoch, and
the remaining Fold-3 wells confirm it.  Folds 1/2 are not loaded.
"""

from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE_CANDIDATES_R5 = (
    Path("/root/autodl-tmp/segn_r4_worknote_scheduled_costmap_fold3_v0_cell.py"),
    Path.cwd() / "segn_r4_worknote_scheduled_costmap_fold3_v0_cell.py",
)
SOURCE_PATH_R5 = next((path for path in SOURCE_CANDIDATES_R5 if path.exists()), None)
if SOURCE_PATH_R5 is None:
    raise FileNotFoundError(
        "Upload segn_r4_worknote_scheduled_costmap_fold3_v0_cell.py to "
        "/root/autodl-tmp before running SEGN-R5."
    )

source_r5 = SOURCE_PATH_R5.read_text(encoding="utf-8")
marker_r5 = "\nstarted_r4 = time.time()"
if marker_r5 not in source_r5:
    raise RuntimeError(f"Cannot find bootstrap marker in {SOURCE_PATH_R5}")
exec(
    compile(source_r5.split(marker_r5, 1)[0], str(SOURCE_PATH_R5), "exec"),
    globals(),
)


EXPERIMENT_R5 = "segn_r5_axisaware_evidence_fold3_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT_R5
MODEL_DIR_R5 = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR_R5.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS_R5 = (4, 5)
VALID_FOLD_R5 = 3
SELECTION_WELLS_R5 = 32
MAX_EPOCHS_R5 = 32
MIN_EPOCHS_R5 = 12
REPORT_EVERY_R5 = 4
PATIENCE_REPORTS_R5 = 4
TRAIN_CHUNKS_PER_WELL_R5 = 3
GRADIENT_ACCUMULATION_R5 = 6
LEARNING_RATE_R5 = 3.0e-4
MIN_LEARNING_RATE_R5 = 3.0e-5
WEIGHT_DECAY_R5 = 2.0e-4
GRADIENT_CLIP_R5 = 2.0
TARGET_SIGMA_R5 = 2.0
LOSS_MEAN_R5 = 0.10
LOSS_CHANNEL_R5 = 0.10

SIGNAL_CHANNELS_R5 = 24
OUTPUT_CHANNELS_R5 = 32
QUERY_DILATIONS_R5 = (1, 4, 16, 48)
REFERENCE_DILATIONS_R5 = (1, 2, 4, 8)
HEADS_R5 = 4
HEAD_DIM_R5 = 8
SLOT_NAMES_R5 = ("bilinear", "cosine")


class AxisResidualR5(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            7,
            padding=3 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(4, channels, affine=False)

    def forward(self, values):
        update = self.norm(self.pointwise(self.depthwise(values)))
        return values + F.gelu(update)


class AxisPyramidEncoderR5(nn.Module):
    def __init__(self, dilations):
        super().__init__()
        self.input = nn.Conv1d(
            3, SIGNAL_CHANNELS_R5, 9, padding=4, bias=False
        )
        self.input_norm = nn.GroupNorm(4, SIGNAL_CHANNELS_R5, affine=False)
        self.blocks = nn.ModuleList(
            [AxisResidualR5(SIGNAL_CHANNELS_R5, value) for value in dilations]
        )
        self.output = nn.Conv1d(
            SIGNAL_CHANNELS_R5 * (len(dilations) + 1),
            OUTPUT_CHANNELS_R5,
            1,
            bias=False,
        )
        self.output_norm = nn.GroupNorm(4, OUTPUT_CHANNELS_R5, affine=False)

    def forward(self, values):
        valid = values[:, 1:2].clamp(0.0, 1.0)
        signal = values[:, (0, 2, 3), :] * valid
        encoded = F.gelu(self.input_norm(self.input(signal)))
        pyramid = [encoded]
        for block in self.blocks:
            encoded = block(encoded)
            pyramid.append(encoded)
        output = self.output_norm(self.output(torch.cat(pyramid, dim=1)))
        return F.gelu(output) * valid


class AxisAwareEvidenceR5(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = AxisPyramidEncoderR5(QUERY_DILATIONS_R5)
        self.reference_encoder = AxisPyramidEncoderR5(REFERENCE_DILATIONS_R5)

        self.query_bilinear = nn.Conv1d(
            OUTPUT_CHANNELS_R5, HEAD_DIM_R5, 1, bias=False
        )
        self.reference_bilinear = nn.Conv1d(
            OUTPUT_CHANNELS_R5, HEAD_DIM_R5, 1, bias=False
        )
        self.bilinear_matrix = nn.Parameter(torch.eye(HEAD_DIM_R5))

        self.query_heads = nn.Conv1d(
            OUTPUT_CHANNELS_R5, HEADS_R5 * HEAD_DIM_R5, 1, bias=False
        )
        self.reference_heads = nn.Conv1d(
            OUTPUT_CHANNELS_R5, HEADS_R5 * HEAD_DIM_R5, 1, bias=False
        )
        self.head_weights = nn.Parameter(torch.zeros(HEADS_R5))
        self.slot_log_scale = nn.Parameter(
            torch.log(torch.tensor([2.0, 4.0], dtype=torch.float32))
        )

        # Sharpness/confidence for both channels, their disagreement, then
        # distance, Z geometry, and legal coverage.  The gate changes trust
        # between maps but cannot create state-dependent candidate evidence.
        self.gate = nn.Sequential(
            nn.Linear(10, 24),
            nn.GELU(),
            nn.Linear(24, 12),
            nn.GELU(),
            nn.Linear(12, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _normalise(logits, candidate_valid):
        valid = candidate_valid[:, None]
        values = logits.masked_fill(~valid, 0.0)
        count = valid.sum(dim=0).clamp_min(1).to(logits.dtype)
        mean = values.sum(dim=0) / count
        centred = (values - mean[None]) * valid
        variance = centred.square().sum(dim=0) / count
        output = centred / torch.sqrt(variance + 1.0e-4)[None]
        return output.masked_fill(~valid, -1.0e4)

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
        horizontal_embedding = self.query_encoder(horizontal)[0]
        typewell_embedding = self.reference_encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(
            typewell_embedding, candidate_positions
        )
        reference_raw, raw_axis_valid = _sample_axis(
            typewell[0], candidate_positions
        )
        candidate_valid = (
            axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)
        )

        query_b = self.query_bilinear(query[None])[0]
        reference_b = self.reference_bilinear(reference[None])[0]
        bilinear = torch.einsum(
            "dt,ds->st", self.bilinear_matrix @ query_b, reference_b
        ) / math.sqrt(HEAD_DIM_R5)

        query_h = self.query_heads(query[None])[0].reshape(
            HEADS_R5, HEAD_DIM_R5, -1
        )
        reference_h = self.reference_heads(reference[None])[0].reshape(
            HEADS_R5, HEAD_DIM_R5, -1
        )
        query_h = F.normalize(query_h, dim=1, eps=1.0e-5)
        reference_h = F.normalize(reference_h, dim=1, eps=1.0e-5)
        heads = torch.einsum("hdt,hds->hst", query_h, reference_h)
        cosine = (
            F.softmax(self.head_weights, dim=0)[:, None, None] * heads
        ).sum(dim=0)

        maps = torch.stack(
            [
                self._normalise(bilinear, candidate_valid),
                self._normalise(cosine, candidate_valid),
            ],
            dim=0,
        )
        scales = torch.exp(self.slot_log_scale).clamp(0.25, 20.0)
        maps = maps * scales[:, None, None]
        states = torch.as_tensor(
            STATE_OFFSETS, device=query.device, dtype=query.dtype
        )
        features = [
            self._distribution_features(values, states) for values in maps
        ]
        sharpness = torch.stack([item[0] for item in features], dim=1)
        confidence = torch.stack([item[1] for item in features], dim=1)
        expected = torch.stack([item[2] for item in features], dim=0)
        disagreement = (
            torch.abs(expected[0] - expected[1]) / STATE_HALF_WIDTH_FT
        )

        query_raw = horizontal[0, :, node_positions]
        query_coverage = query_raw[1].clamp(0.0, 1.0)
        reference_coverage = candidate_valid.float().mean().expand_as(
            query_coverage
        )
        gate_context = torch.stack(
            [
                disagreement,
                geometry[1].clamp(-1.0, 1.0),
                geometry[3].clamp(-2.0, 2.0),
                geometry[4].clamp(-2.0, 2.0),
                query_coverage,
                reference_coverage,
            ],
            dim=1,
        )
        gate_input = torch.cat([sharpness, confidence, gate_context], dim=1)
        gate_weights = F.softmax(self.gate(gate_input), dim=1).transpose(0, 1)
        fused = (gate_weights[:, None, :] * maps).sum(dim=0)
        fused = fused.masked_fill(~candidate_valid[:, None], -1.0e4)
        if not return_aux:
            return fused
        return fused, {
            "maps": maps,
            "expected": expected,
            "gate_weights": gate_weights,
            "candidate_valid": candidate_valid,
        }


def _query_zero_guard_r5(model, device):
    model.eval()
    horizontal = torch.zeros(1, 4, 900, device=device)
    horizontal[:, 1] = 1.0
    typewell = torch.randn(1, 4, 385, device=device)
    typewell[:, 1] = 1.0
    nodes = torch.arange(200, 216, device=device)
    candidates = torch.as_tensor(STATE_OFFSETS, device=device)
    geometry = torch.zeros(8, 16, device=device)
    with torch.no_grad():
        logits, auxiliary = model(
            horizontal,
            typewell,
            nodes,
            candidates,
            geometry,
            return_aux=True,
        )
    valid = auxiliary["candidate_valid"]
    variations = []
    for column in range(logits.shape[1]):
        variations.append(float(logits[:, column][valid].std()))
    return {
        "maximum_no_gr_logit_std": float(max(variations)),
        "finite": bool(torch.isfinite(logits).all()),
    }


def _loss_r5(model, payload, device):
    tensors = _tensor_payload(payload, device)
    fused, auxiliary = model(*tensors, return_aux=True)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.as_tensor(
        STATE_OFFSETS, device=device, dtype=target.dtype
    )
    target_distribution = _target_distribution_r4(target, states)
    fused_ce = -(
        target_distribution * F.log_softmax(fused, dim=0)
    ).sum(dim=0).mean()
    probability = F.softmax(fused, dim=0)
    expected = (probability * states[:, None]).sum(dim=0)
    mean_loss = F.smooth_l1_loss(expected / 8.0, target / 8.0)
    channel_ce = []
    for values in auxiliary["maps"]:
        channel_ce.append(
            -(target_distribution * F.log_softmax(values, dim=0))
            .sum(dim=0)
            .mean()
        )
    channel_loss = torch.stack(channel_ce).mean()
    total = fused_ce + LOSS_MEAN_R5 * mean_loss + LOSS_CHANNEL_R5 * channel_loss
    return total, {
        "fused_ce": float(fused_ce.detach()),
        "mean": float(mean_loss.detach()),
        "channel": float(channel_loss.detach()),
        "gate": auxiliary["gate_weights"].detach(),
    }


def _training_starts_r5(record, rng):
    starts = np.asarray(record.chunk_starts, np.int32)
    if len(starts) <= TRAIN_CHUNKS_PER_WELL_R5:
        return starts
    boundaries = np.linspace(0, len(starts), TRAIN_CHUNKS_PER_WELL_R5 + 1)
    selected = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        lo = int(math.floor(left))
        hi = max(lo + 1, int(math.ceil(right)))
        selected.append(int(starts[int(rng.integers(lo, min(hi, len(starts))))]))
    return np.asarray(selected, np.int32)


def _teacher_history_r5(record, start):
    start = int(start)
    if start <= 0:
        return 0.0, 0.0
    candidates = np.asarray(record.chunk_starts, np.int32)
    candidates = candidates[candidates < start]
    previous_start = int(candidates[-1]) if len(candidates) else 0
    current_row = int(record.node_rows[start])
    previous_row = int(record.node_rows[previous_start])
    delta = float(record.truth[current_row] - record.truth[previous_row])
    md_delta = max(float(record.md[current_row] - record.md[previous_row]), 1.0)
    return delta, delta / md_delta


def _train_epoch_r5(model, records, optimizer, scaler, device, epoch):
    model.train()
    rng = np.random.default_rng(SEED + 7919 * epoch)
    order = rng.permutation(len(records))
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "fused_ce": 0.0,
        "mean": 0.0,
        "channel": 0.0,
        "gate_bilinear": 0.0,
        "gate_cosine": 0.0,
        "samples": 0,
        "skipped": 0,
        "optimizer_steps": 0,
    }
    for progress, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        for start in _training_starts_r5(record, rng):
            start = int(start)
            start_row = int(record.node_rows[start])
            previous_delta, previous_rate = _teacher_history_r5(record, start)
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]),
                previous_delta=previous_delta,
                previous_rate=previous_rate,
            )
            if np.max(np.abs(payload["target_state"])) > STATE_HALF_WIDTH_FT - 0.5:
                totals["skipped"] += 1
                continue
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, parts = _loss_r5(model, payload, device)
                scaled_loss = loss / GRADIENT_ACCUMULATION_R5
            if not torch.isfinite(loss):
                totals["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(scaled_loss).backward()
            totals["loss"] += float(loss.detach())
            for key in ("fused_ce", "mean", "channel"):
                totals[key] += parts[key]
            gate = parts["gate"].float().cpu().numpy()
            totals["gate_bilinear"] += float(gate[0].mean())
            totals["gate_cosine"] += float(gate[1].mean())
            totals["samples"] += 1
            if totals["samples"] % GRADIENT_ACCUMULATION_R5 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_R5
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                totals["optimizer_steps"] += 1
        if progress % 100 == 0 or progress == len(order):
            print(
                f"SEGN-R5 train epoch={epoch} wells={progress}/{len(order)} "
                f"loss={totals['loss'] / max(totals['samples'], 1):.4f}",
                flush=True,
            )
    if totals["samples"] % GRADIENT_ACCUMULATION_R5:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        totals["optimizer_steps"] += 1
    denominator = max(totals["samples"], 1)
    return {
        **{
            key: totals[key] / denominator
            for key in (
                "loss",
                "fused_ce",
                "mean",
                "channel",
                "gate_bilinear",
                "gate_cosine",
            )
        },
        "samples": totals["samples"],
        "skipped": totals["skipped"],
        "optimizer_steps": totals["optimizer_steps"],
    }


@torch.no_grad()
def _chunk_posterior_r5(model, payload, device):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, auxiliary = model(*tensors, return_aux=True)
    logits = logits.float()
    probability = F.softmax(logits, dim=0)
    states = torch.as_tensor(
        STATE_OFFSETS, device=device, dtype=torch.float32
    )
    expected = (probability * states[:, None]).sum(dim=0)
    mode = STATE_OFFSETS[torch.argmax(probability, dim=0).cpu().numpy()]
    return {
        "logits": logits.cpu().numpy().astype(np.float64),
        "probability": probability.cpu().numpy().astype(np.float64),
        "expected": expected.cpu().numpy().astype(np.float64),
        "mode": mode.astype(np.float64),
        "valid": auxiliary["candidate_valid"].cpu().numpy().astype(bool),
        "gate": auxiliary["gate_weights"].float().cpu().numpy().astype(np.float64),
    }


@torch.no_grad()
def _evaluate_r5(model, records, device, control="full", save_prefix=None):
    model.eval()
    wells = []
    truths = []
    mean_predictions = []
    mode_predictions = []
    distances = []
    anchors = []
    bywell_rows = []
    local_target = []
    local_expected = []
    local_mode = []
    local_mass5 = []
    local_rank = []
    local_nll = []
    gate_sum = np.zeros(2, np.float64)
    gate_columns = 0
    shuffled = (
        list(records[1:]) + list(records[:1])
        if len(records) > 1
        else list(records)
    )

    for number, (record, shuffled_record) in enumerate(zip(records, shuffled), 1):
        node_count = len(record.node_rows)
        mean_sum = np.zeros(node_count, np.float64)
        mode_sum = np.zeros(node_count, np.float64)
        count = np.zeros(node_count, np.float64)
        mean_sum[0] = mode_sum[0] = float(record.anchor_tvt)
        count[0] = 1.0
        well_local_target = []
        well_local_expected = []

        for raw_start in record.chunk_starts:
            start = int(raw_start)
            start_row = int(record.node_rows[start])
            previous_delta, previous_rate = _teacher_history_r5(record, start)
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]),
                previous_delta=previous_delta,
                previous_rate=previous_rate,
                control=control,
                shuffled_typewell=shuffled_record,
            )
            posterior = _chunk_posterior_r5(model, payload, device)
            target = payload["target_state"].astype(np.float64)
            end = start + len(target)
            node_slice = slice(start + 1, end + 1)
            anchor = float(payload["deployed_anchor"])
            mean_sum[node_slice] += anchor + posterior["expected"]
            mode_sum[node_slice] += anchor + posterior["mode"]
            count[node_slice] += 1.0

            local_target.extend(target.tolist())
            local_expected.extend(posterior["expected"].tolist())
            local_mode.extend(posterior["mode"].tolist())
            well_local_target.extend(target.tolist())
            well_local_expected.extend(posterior["expected"].tolist())
            for column, target_value in enumerate(target):
                within5 = np.abs(STATE_OFFSETS - target_value) <= 5.0
                local_mass5.append(
                    float(posterior["probability"][within5, column].sum())
                )
                true_index = int(np.argmin(np.abs(STATE_OFFSETS - target_value)))
                true_score = posterior["logits"][true_index, column]
                valid = posterior["valid"]
                scores = posterior["logits"][:, column]
                better = int(np.sum(valid & (scores > true_score + 1.0e-8)))
                tied = int(np.sum(valid & (np.abs(scores - true_score) <= 1.0e-8)))
                local_rank.append(float(1 + better + 0.5 * max(tied - 1, 0)))
                local_nll.append(
                    float(-np.log(max(posterior["probability"][true_index, column], 1.0e-12)))
                )
            gate_sum += posterior["gate"].sum(axis=1)
            gate_columns += posterior["gate"].shape[1]

        supported = count > 0
        mean_nodes = mean_sum[supported] / count[supported]
        mode_nodes = mode_sum[supported] / count[supported]
        support_md = record.md[record.node_rows[supported]].astype(np.float64)
        eval_rows = record.eval_index
        query_md = record.md[eval_rows].astype(np.float64)
        mean_prediction = np.interp(query_md, support_md, mean_nodes)
        mode_prediction = np.interp(query_md, support_md, mode_nodes)
        truth = record.truth[eval_rows].astype(np.float64)
        distance = query_md - float(record.md[record.anchor_index])
        wells.append(record.well)
        truths.append(truth)
        mean_predictions.append(mean_prediction)
        mode_predictions.append(mode_prediction)
        distances.append(distance)
        anchors.append(float(record.anchor_tvt))
        bywell_rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "mean_rmse": float(np.sqrt(np.mean((mean_prediction - truth) ** 2))),
                "mode_rmse": float(np.sqrt(np.mean((mode_prediction - truth) ** 2))),
                "local_expected_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                np.asarray(well_local_expected)
                                - np.asarray(well_local_target)
                            )
                            ** 2
                        )
                    )
                ),
            }
        )
        if number % 25 == 0 or number == len(records):
            print(f"SEGN-R5 evaluate {control} {number}/{len(records)}", flush=True)

    mean_metrics = _metric_payload_r1(wells, truths, mean_predictions, distances)
    mean_metrics.update(_shape_metrics_r4(wells, truths, mean_predictions, anchors))
    mode_metrics = _metric_payload_r1(wells, truths, mode_predictions, distances)
    local_target_array = np.asarray(local_target, np.float64)
    local_expected_array = np.asarray(local_expected, np.float64)
    local_mode_array = np.asarray(local_mode, np.float64)
    local_metrics = {
        "columns": int(len(local_target_array)),
        "expected_rmse": float(
            np.sqrt(np.mean((local_expected_array - local_target_array) ** 2))
        ),
        "mode_rmse": float(
            np.sqrt(np.mean((local_mode_array - local_target_array) ** 2))
        ),
        "expected_correlation": (
            float(np.corrcoef(local_expected_array, local_target_array)[0, 1])
            if np.std(local_expected_array) > 1.0e-12
            else math.nan
        ),
        "mode_within5": float(np.mean(np.abs(local_mode_array - local_target_array) <= 5.0)),
        "posterior_mass_within5": float(np.mean(local_mass5)),
        "median_truth_rank": float(np.median(local_rank)),
        "p90_truth_rank": float(np.quantile(local_rank, 0.90)),
        "mean_true_state_nll": float(np.mean(local_nll)),
    }
    gate_mean = gate_sum / max(gate_columns, 1)
    diagnostics = {
        "gate_bilinear": float(gate_mean[0]),
        "gate_cosine": float(gate_mean[1]),
        "posterior_mean_gain_vs_mode": float(
            mode_metrics["rmse"] - mean_metrics["rmse"]
        ),
    }
    bywell = pd.DataFrame(bywell_rows)
    if save_prefix:
        bywell.to_csv(OUT_DIR / f"{save_prefix}_bywell.csv", index=False)
        np.savez_compressed(
            OUT_DIR / f"{save_prefix}_rows.npz",
            well=np.concatenate(
                [np.repeat(well, len(values)) for well, values in zip(wells, truths)]
            ).astype("U"),
            truth=np.concatenate(truths).astype(np.float32),
            prediction=np.concatenate(mean_predictions).astype(np.float32),
            distance_past_anchor=np.concatenate(distances).astype(np.float32),
        )
    return {
        "path_mean": mean_metrics,
        "path_mode": mode_metrics,
        "local": local_metrics,
        "diagnostics": diagnostics,
    }, bywell


def _fit_r5(train_records, selection_records, device):
    _seed_everything(SEED + 5151)
    model = AxisAwareEvidenceR5().to(device)
    zero_guard = _query_zero_guard_r5(model, device)
    if (
        not zero_guard["finite"]
        or zero_guard["maximum_no_gr_logit_std"] > 1.0e-6
    ):
        raise RuntimeError(f"Query-zero guard failed before training: {zero_guard}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_R5, weight_decay=WEIGHT_DECAY_R5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_R5, eta_min=MIN_LEARNING_RATE_R5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = math.inf
    bad_reports = 0
    history = []
    started_fit = time.time()
    for epoch in range(1, MAX_EPOCHS_R5 + 1):
        training = _train_epoch_r5(
            model, train_records, optimizer, scaler, device, epoch
        )
        scheduler.step()
        report = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started_fit) / 60.0,
        }
        if epoch % REPORT_EVERY_R5 == 0 or epoch == MAX_EPOCHS_R5:
            selection, _ = _evaluate_r5(model, selection_records, device)
            report.update(
                {
                    "selection_local_expected_rmse": selection["local"]["expected_rmse"],
                    "selection_path_mean_rmse": selection["path_mean"]["rmse"],
                    "selection_shape_correlation": selection["path_mean"][
                        "within_well_shape_correlation"
                    ],
                    "selection_mass_within5": selection["local"][
                        "posterior_mass_within5"
                    ],
                }
            )
            score = float(selection["local"]["expected_rmse"])
            if score < best_score - 1.0e-4:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_reports = 0
            else:
                bad_reports += 1
        history.append(report)
        pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
        print("SEGN-R5 EPOCH", report, flush=True)
        if epoch >= MIN_EPOCHS_R5 and bad_reports >= PATIENCE_REPORTS_R5:
            print(f"SEGN-R5 early stopping at epoch {epoch}", flush=True)
            break
    if best_epoch == 0:
        raise RuntimeError("SEGN-R5 produced no selection checkpoint")
    model.load_state_dict(best_state)
    checkpoint = MODEL_DIR_R5 / "best_axisaware_evidence.pt"
    torch.save(
        {
            "model": best_state,
            "epoch": best_epoch,
            "selection_local_expected_rmse": best_score,
            "state_offsets": STATE_OFFSETS,
            "query_dilations": QUERY_DILATIONS_R5,
            "reference_dilations": REFERENCE_DILATIONS_R5,
            "source": str(SOURCE_PATH_R5),
        },
        checkpoint,
    )
    return model, best_epoch, best_score, zero_guard, checkpoint


started_r5 = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R5")
device_r5 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r5, fold_path_r5 = _load_folds()
train_wells_r5 = folds_r5.loc[
    folds_r5["fold"].isin(TRAIN_FOLDS_R5), "well"
].tolist()
fold3_wells_r5 = folds_r5.loc[
    folds_r5["fold"].eq(VALID_FOLD_R5), "well"
].tolist()
if len(train_wells_r5) != 309 or len(fold3_wells_r5) != 154:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3 sizes: {len(train_wells_r5)}/{len(fold3_wells_r5)}"
    )
records_r5 = {
    record.well: record
    for record in _load_records(
        sorted(set(train_wells_r5 + fold3_wells_r5)), "r5_fold345"
    )
}
train_records_r5 = [records_r5[well] for well in train_wells_r5]
fold3_records_r5 = [records_r5[well] for well in fold3_wells_r5]
order_r5 = np.argsort([record.eval_rows for record in fold3_records_r5])
selection_positions_r5 = np.unique(
    np.rint(np.linspace(0, len(order_r5) - 1, SELECTION_WELLS_R5)).astype(int)
)
selection_indices_r5 = set(order_r5[selection_positions_r5].tolist())
selection_records_r5 = [
    record
    for index, record in enumerate(fold3_records_r5)
    if index in selection_indices_r5
]
confirmation_records_r5 = [
    record
    for index, record in enumerate(fold3_records_r5)
    if index not in selection_indices_r5
]

guards_r5 = {
    "source": str(SOURCE_PATH_R5),
    "fold_source": str(fold_path_r5),
    "train_folds": list(TRAIN_FOLDS_R5),
    "train_wells": len(train_records_r5),
    "selection_fold": VALID_FOLD_R5,
    "selection_wells": len(selection_records_r5),
    "confirmation_wells": len(confirmation_records_r5),
    "fold2_touched": False,
    "fold1_touched": False,
    "query_receptive_field_md_ft": 9 + 6 * sum(QUERY_DILATIONS_R5),
    "reference_receptive_field_tvt_ft": 9 + 6 * sum(REFERENCE_DILATIONS_R5),
    "target_axis_coverage": _target_axis_coverage(fold3_records_r5),
    "parameters": int(sum(p.numel() for p in AxisAwareEvidenceR5().parameters())),
}
print("SEGN-R5 GUARDS", json.dumps(guards_r5, indent=2), flush=True)

model_r5, best_epoch_r5, best_score_r5, zero_guard_r5, checkpoint_r5 = _fit_r5(
    train_records_r5, selection_records_r5, device_r5
)
confirmation_full_r5, _ = _evaluate_r5(
    model_r5,
    confirmation_records_r5,
    device_r5,
    control="full",
    save_prefix="fold3_confirmation_full",
)
confirmation_no_gr_r5, _ = _evaluate_r5(
    model_r5, confirmation_records_r5, device_r5, control="no_gr"
)
confirmation_shuffled_r5, _ = _evaluate_r5(
    model_r5, confirmation_records_r5, device_r5, control="shuffled_typewell"
)
post_zero_guard_r5 = _query_zero_guard_r5(model_r5, device_r5)

local_gain_vs_no_gr_r5 = (
    confirmation_no_gr_r5["local"]["expected_rmse"]
    - confirmation_full_r5["local"]["expected_rmse"]
)
path_gain_vs_no_gr_r5 = (
    confirmation_no_gr_r5["path_mean"]["rmse"]
    - confirmation_full_r5["path_mean"]["rmse"]
)
shuffled_local_degradation_r5 = (
    confirmation_shuffled_r5["local"]["expected_rmse"]
    - confirmation_full_r5["local"]["expected_rmse"]
)
shuffled_path_degradation_r5 = (
    confirmation_shuffled_r5["path_mean"]["rmse"]
    - confirmation_full_r5["path_mean"]["rmse"]
)

checks_r5 = {
    "target_axis_coverage_ge_0p99": guards_r5["target_axis_coverage"] >= 0.99,
    "query_zero_logit_std_le_1e_6": (
        post_zero_guard_r5["maximum_no_gr_logit_std"] <= 1.0e-6
    ),
    "local_expected_gain_vs_no_gr_ge_0p5": local_gain_vs_no_gr_r5 >= 0.5,
    "path_mean_gain_vs_no_gr_ge_0p3": path_gain_vs_no_gr_r5 >= 0.3,
    "shuffled_local_degradation_ge_0p3": shuffled_local_degradation_r5 >= 0.3,
    "shuffled_path_degradation_ge_0p3": shuffled_path_degradation_r5 >= 0.3,
    "teacher_anchor_path_rmse_le_8p5": confirmation_full_r5["path_mean"]["rmse"] <= 8.5,
    "within_well_shape_correlation_ge_0p75": confirmation_full_r5[
        "path_mean"
    ]["within_well_shape_correlation"] >= 0.75,
    "posterior_mass_within5_ge_0p15": confirmation_full_r5["local"][
        "posterior_mass_within5"
    ] >= 0.15,
    "posterior_mean_better_than_mode": confirmation_full_r5["diagnostics"][
        "posterior_mean_gain_vs_mode"
    ] > 0.0,
}
strict_pass_r5 = bool(all(checks_r5.values()))

verdict_r5 = {
    "experiment": EXPERIMENT_R5,
    "stage": "fold3_axisaware_querycausal_teacher_anchor_evidence_gateway",
    "sample_unit": "teacher-anchored chunks nested within complete wells",
    "deployable": False,
    "truth_usage": "teacher anchors for evidence observability only",
    "legal_model_inputs_only": True,
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R5),
        "selection_fold": VALID_FOLD_R5,
        "selection_wells": len(selection_records_r5),
        "confirmation_wells": len(confirmation_records_r5),
        "fold2_touched": False,
        "fold1_touched": False,
        "selected_epoch": best_epoch_r5,
        "selected_local_expected_rmse": best_score_r5,
    },
    "architecture": {
        "matching_channels": list(SLOT_NAMES_R5),
        "query_axis": "MD",
        "reference_axis": "TVT",
        "query_dilations": list(QUERY_DILATIONS_R5),
        "reference_dilations": list(REFERENCE_DILATIONS_R5),
        "query_causal_by_construction": True,
        "prior_present": False,
        "recurrence_present": False,
        "primary_readout": "posterior mean",
    },
    "guards": guards_r5,
    "query_zero_guard_before_training": zero_guard_r5,
    "query_zero_guard_after_training": post_zero_guard_r5,
    "confirmation": {
        "full": confirmation_full_r5,
        "no_gr": confirmation_no_gr_r5,
        "shuffled_typewell": confirmation_shuffled_r5,
    },
    "diagnostics": {
        "local_gain_vs_no_gr": local_gain_vs_no_gr_r5,
        "path_gain_vs_no_gr": path_gain_vs_no_gr_r5,
        "shuffled_local_degradation": shuffled_local_degradation_r5,
        "shuffled_path_degradation": shuffled_path_degradation_r5,
    },
    "checks": checks_r5,
    "strict_pass": strict_pass_r5,
    "checkpoint": str(checkpoint_r5),
    "elapsed_minutes": (time.time() - started_r5) / 60.0,
    "next": (
        "PASS -> transplant this frozen evidence encoder into the R4 scheduled prior/gate recurrence"
        if strict_pass_r5
        else "FAIL -> axis-aware bilinear/cosine evidence is still insufficient; do not resume recurrence"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict_r5)
print("SEGN-R5 FINAL VERDICT", json.dumps(verdict_r5, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

if not strict_pass_r5:
    raise RuntimeError(json.dumps(verdict_r5, indent=2))
