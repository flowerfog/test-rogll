"""Alias-aware long-context absolute matcher gateway on untouched Fold 2.

This is an emission experiment, not another path decoder.  It differs from the
earlier real-correspondence matcher in four material ways:

* horizontal context extends to 1600 ft and typewell context to 96 ft;
* every candidate state is conditioned on a deployment-legal prefix/geometry
  prior rather than forcing GR to recover the absolute gauge alone;
* online hardest negatives supplement explicit +/- cycle aliases;
* prior jitter and distance-balanced row weights prevent a prior-only shortcut.

Folds 3/4/5 train the matcher.  Fold 2 selects the epoch and is evaluated with
no-GR and shuffled-typewell controls.  Fold 1 is not touched by this cell.
"""

from dataclasses import dataclass
from pathlib import Path
import gc
import json
import math
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REQUIRED_ALIAS_V2 = {
    "OOF_ROOT",
    "folds",
    "device",
    "_load_raw_record",
    "_native_z_relative_v2",
    "_robust_prefix_slope_v2",
    "STATE_OFFSETS_V2",
    "_write_json",
    "_json_default",
}
missing_alias_v2 = sorted(REQUIRED_ALIAS_V2.difference(globals()))
if missing_alias_v2:
    raise RuntimeError(
        "Run in the Drift-V3 live kernel; missing: " + repr(missing_alias_v2)
    )

EXPERIMENT_ALIAS_V2 = "alias_aware_longcontext_absolute_matcher_fold2_v2"
OUT_DIR_ALIAS_V2 = Path(OOF_ROOT) / EXPERIMENT_ALIAS_V2
MODEL_DIR_ALIAS_V2 = OUT_DIR_ALIAS_V2 / "models"
for directory_alias_v2 in (OUT_DIR_ALIAS_V2, MODEL_DIR_ALIAS_V2):
    directory_alias_v2.mkdir(parents=True, exist_ok=True)

SEED_ALIAS_V2 = 20260720
CORE_FOLDS_ALIAS_V2 = (3, 4, 5)
VALID_FOLD_ALIAS_V2 = 2

QUERY_HALF_WIDTHS_ALIAS_V2 = np.asarray(
    [64.0, 128.0, 256.0, 512.0, 1024.0, 1600.0], np.float32
)
REFERENCE_HALF_WIDTHS_ALIAS_V2 = np.asarray(
    [4.0, 8.0, 16.0, 32.0, 64.0, 96.0], np.float32
)
WINDOW_POINTS_ALIAS_V2 = 65
WINDOW_UNIT_ALIAS_V2 = np.linspace(-1.0, 1.0, WINDOW_POINTS_ALIAS_V2, dtype=np.float32)
STATE_OFFSETS_ALIAS_V2 = np.asarray(STATE_OFFSETS_V2, np.float32)
N_STATES_ALIAS_V2 = len(STATE_OFFSETS_ALIAS_V2)
ZERO_STATE_ALIAS_V2 = int(np.argmin(np.abs(STATE_OFFSETS_ALIAS_V2)))
REFERENCE_CONTEXT_ALIAS_V2 = (
    _load_raw_record(str(folds.iloc[0]["well"])).typewell.shape[1]
    - N_STATES_ALIAS_V2
) // 2

TRAIN_POSITIONS_ALIAS_V2 = 128
VALID_POSITIONS_ALIAS_V2 = 224
MIN_REFERENCE_COVERAGE_ALIAS_V2 = 0.42
TARGET_SIGMA_ALIAS_V2 = 1.5
ALIAS_OFFSETS_ALIAS_V2 = (12, 24, 36, 48, 72)
ALIAS_MARGIN_ALIAS_V2 = 0.55
ALIAS_WEIGHT_ALIAS_V2 = 0.65
HARDEST_NEGATIVES_ALIAS_V2 = 6
HARD_MARGIN_ALIAS_V2 = 0.40
HARD_WEIGHT_ALIAS_V2 = 0.45
PRIOR_JITTER_STD_FT_ALIAS_V2 = 8.0
PRIOR_DROPOUT_ALIAS_V2 = 0.20

EMBEDDING_ALIAS_V2 = 56
MAX_EPOCHS_ALIAS_V2 = 12
MIN_EPOCHS_ALIAS_V2 = 4
EARLY_STOP_ALIAS_V2 = 3
LEARNING_RATE_ALIAS_V2 = 3e-4
MIN_LEARNING_RATE_ALIAS_V2 = 3e-5
WEIGHT_DECAY_ALIAS_V2 = 3e-5
GRADIENT_CLIP_ALIAS_V2 = 5.0


def _seed_alias_v2(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _select_even_alias_v2(indices, count):
    indices = np.asarray(indices, np.int64)
    if not len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, min(count, len(indices)))
    return np.unique(indices[np.rint(positions).astype(np.int64)])


def _window_bank_alias_v2(axis, value, valid, centres, half_widths):
    axis = np.asarray(axis, np.float64)
    value = np.asarray(value, np.float64)
    valid = np.asarray(valid, np.float64)
    centres = np.asarray(centres, np.float64)
    half_widths = np.asarray(half_widths, np.float64)
    query = (
        centres[:, None, None]
        + half_widths[None, :, None] * WINDOW_UNIT_ALIAS_V2[None, None]
    )
    flat = query.reshape(-1)
    sampled = np.interp(flat, axis, value).reshape(query.shape)
    sampled_valid = np.interp(flat, axis, valid).reshape(query.shape)
    mask = (
        (query >= axis[0])
        & (query <= axis[-1])
        & (sampled_valid >= 0.5)
    )
    output = np.empty((*query.shape[:2], 2, WINDOW_POINTS_ALIAS_V2), np.float16)
    output[:, :, 0] = np.where(mask, sampled, 0.0).astype(np.float16)
    output[:, :, 1] = mask.astype(np.float16)
    return output


@dataclass
class AliasSampleV2:
    well: str
    fold: int
    reference: np.ndarray
    state_valid: np.ndarray
    query: np.ndarray
    labels: np.ndarray
    label_index: np.ndarray
    prior_offset: np.ndarray
    distance_ft: np.ndarray
    z_relative: np.ndarray
    row_weight: np.ndarray
    requested_rows: int
    eval_rows: int


def _build_alias_sample_v2(well, fold):
    record = _load_raw_record(well)
    reference_axis = (
        np.arange(record.typewell.shape[1], dtype=np.float32)
        - float(record.typewell.shape[1] // 2)
    )
    reference = _window_bank_alias_v2(
        reference_axis,
        record.typewell[0],
        record.typewell[1],
        STATE_OFFSETS_ALIAS_V2,
        REFERENCE_HALF_WIDTHS_ALIAS_V2,
    )
    scale_coverage = reference[:, :, 1].astype(np.float32).mean(axis=-1)
    state_valid = np.max(scale_coverage, axis=1) >= MIN_REFERENCE_COVERAGE_ALIAS_V2

    count = TRAIN_POSITIONS_ALIAS_V2 if fold in CORE_FOLDS_ALIAS_V2 else VALID_POSITIONS_ALIAS_V2
    rows = _select_even_alias_v2(record.eval_index, count)
    labels = record.truth[rows].astype(np.float64) - float(record.anchor_tvt)
    label_index = np.rint(labels - float(STATE_OFFSETS_ALIAS_V2[0])).astype(np.int64)
    valid_label = (
        np.isfinite(labels)
        & (label_index >= 0)
        & (label_index < N_STATES_ALIAS_V2)
    )
    safe_index = np.clip(label_index, 0, N_STATES_ALIAS_V2 - 1)
    valid_label &= state_valid[safe_index]
    original_rows = len(rows)
    rows = rows[valid_label]
    labels = labels[valid_label]
    label_index = label_index[valid_label]

    full_z = _native_z_relative_v2(well).astype(np.float64)
    anchor_md = float(record.md[record.anchor_index])
    distance = record.md[rows].astype(np.float64) - anchor_md
    z_relative = full_z[rows]
    prefix_slope = _robust_prefix_slope_v2(record, full_z)
    prior_offset = prefix_slope * distance - z_relative
    query = _window_bank_alias_v2(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        record.md[rows],
        QUERY_HALF_WIDTHS_ALIAS_V2,
    )
    row_weight = 1.0 + 0.50 * np.clip(distance / 2500.0, 0.0, 1.0)
    return AliasSampleV2(
        well=well,
        fold=int(fold),
        reference=reference,
        state_valid=state_valid,
        query=query,
        labels=labels.astype(np.float32),
        label_index=label_index.astype(np.int64),
        prior_offset=prior_offset.astype(np.float32),
        distance_ft=distance.astype(np.float32),
        z_relative=z_relative.astype(np.float32),
        row_weight=row_weight.astype(np.float32),
        requested_rows=original_rows,
        eval_rows=int(record.eval_rows),
    )


class ChannelNormAliasV2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ResidualPatchAliasV2(nn.Module):
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
            ChannelNormAliasV2(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.network(values))


class PatchEncoderAliasV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 48, 9, padding=4),
            ChannelNormAliasV2(48),
            nn.GELU(),
            ResidualPatchAliasV2(48, 1),
            ResidualPatchAliasV2(48, 2),
            ResidualPatchAliasV2(48, 4),
            ResidualPatchAliasV2(48, 8),
        )
        self.output = nn.Sequential(
            nn.Linear(96, 112),
            nn.GELU(),
            nn.Linear(112, EMBEDDING_ALIAS_V2),
        )

    def forward(self, values):
        hidden = self.network(values)
        pooled = torch.cat([hidden.mean(-1), hidden.amax(-1)], dim=1)
        return F.normalize(self.output(pooled), dim=1, eps=1e-6)


class AliasAwareAbsoluteMatcherV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = PatchEncoderAliasV2()
        self.reference_encoder = PatchEncoderAliasV2()
        q_scales = len(QUERY_HALF_WIDTHS_ALIAS_V2)
        r_scales = len(REFERENCE_HALF_WIDTHS_ALIAS_V2)
        correlation_features = q_scales * r_scales
        summary_features = 2 + q_scales + r_scales
        coverage_features = q_scales + r_scales
        prior_features = 11
        self.scorer = nn.Sequential(
            nn.Linear(
                correlation_features + summary_features + coverage_features + prior_features,
                128,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        query,
        reference,
        state_valid,
        prior_offset,
        distance_ft,
        z_relative,
        prior_jitter=False,
    ):
        batch, q_scales, channels, points = query.shape
        states, r_scales, ref_channels, ref_points = reference.shape
        query_embedding = self.query_encoder(
            query.reshape(batch * q_scales, channels, points)
        ).reshape(batch, q_scales, -1)
        reference_embedding = self.reference_encoder(
            reference.reshape(states * r_scales, ref_channels, ref_points)
        ).reshape(states, r_scales, -1)
        correlation = torch.einsum(
            "bqe,ste->bsqt", query_embedding, reference_embedding
        )
        correlation = F.softplus(self.logit_scale) * correlation
        flat = correlation.flatten(2)
        maximum = flat.amax(dim=2, keepdim=True)
        mean = flat.mean(dim=2, keepdim=True)
        max_by_query = correlation.amax(dim=3)
        max_by_reference = correlation.amax(dim=2)

        query_coverage = query[:, :, 1].mean(dim=-1)
        reference_coverage = reference[:, :, 1].mean(dim=-1)
        query_coverage = query_coverage[:, None].expand(-1, states, -1)
        reference_coverage = reference_coverage[None].expand(batch, -1, -1)

        prior = prior_offset
        if prior_jitter:
            prior = prior + torch.randn_like(prior) * PRIOR_JITTER_STD_FT_ALIAS_V2
            keep = (
                torch.rand(len(prior), device=prior.device) >= PRIOR_DROPOUT_ALIAS_V2
            ).to(prior.dtype)
        else:
            keep = torch.ones_like(prior)
        state = torch.from_numpy(STATE_OFFSETS_ALIAS_V2).to(
            query.device, dtype=query.dtype
        )[None].expand(batch, -1)
        delta = (state - prior[:, None]) / 32.0
        absolute_delta = delta.abs()
        prior_features = torch.stack(
            [
                delta,
                absolute_delta,
                delta.square(),
                torch.exp(-0.5 * (delta * 4.0).square()),
                torch.exp(-0.5 * (delta * 2.0).square()),
                torch.exp(-0.5 * delta.square()),
                state / 128.0,
                prior[:, None].expand(-1, states) / 128.0,
                distance_ft[:, None].expand(-1, states) / 3000.0,
                z_relative[:, None].expand(-1, states) / 128.0,
                keep[:, None].expand(-1, states),
            ],
            dim=2,
        )
        prior_features = prior_features * keep[:, None, None]
        features = torch.cat(
            [
                flat,
                maximum,
                mean,
                max_by_query,
                max_by_reference,
                query_coverage,
                reference_coverage,
                prior_features,
            ],
            dim=2,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits.masked_fill(~state_valid[None], -1e4)


def _sample_tensors_alias_v2(sample, no_gr=False, reference=None, state_valid=None):
    query = sample.query.astype(np.float32)
    if no_gr:
        query = query.copy()
        query[:, :, 0] = 0.0
    if reference is None:
        reference = sample.reference
    if state_valid is None:
        state_valid = sample.state_valid
    return (
        torch.from_numpy(query).to(device, non_blocking=True),
        torch.from_numpy(np.asarray(reference, np.float32)).to(device, non_blocking=True),
        torch.from_numpy(np.asarray(state_valid, bool)).to(device, non_blocking=True),
        torch.from_numpy(sample.labels).to(device, non_blocking=True),
        torch.from_numpy(sample.label_index).to(device, non_blocking=True),
        torch.from_numpy(sample.prior_offset).to(device, non_blocking=True),
        torch.from_numpy(sample.distance_ft).to(device, non_blocking=True),
        torch.from_numpy(sample.z_relative).to(device, non_blocking=True),
        torch.from_numpy(sample.row_weight).to(device, non_blocking=True),
    )


def _loss_alias_v2(logits, labels, label_index, row_weight, state_valid):
    states = torch.from_numpy(STATE_OFFSETS_ALIAS_V2).to(logits.device)
    target = torch.exp(-0.5 * ((states[None] - labels[:, None]) / TARGET_SIGMA_ALIAS_V2) ** 2)
    target = target * state_valid[None]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    nll = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    ce = (nll * row_weight).sum() / row_weight.sum().clamp_min(1e-6)

    positive = logits.gather(1, label_index[:, None])[:, 0]
    alias_terms = []
    for offset in ALIAS_OFFSETS_ALIAS_V2:
        for sign in (-1, 1):
            candidate = label_index + sign * int(offset)
            valid = (candidate >= 0) & (candidate < N_STATES_ALIAS_V2)
            safe = candidate.clamp(0, N_STATES_ALIAS_V2 - 1)
            valid &= state_valid[safe]
            if valid.any():
                negative = logits.gather(1, safe[:, None])[:, 0]
                alias_terms.append(
                    F.relu(ALIAS_MARGIN_ALIAS_V2 - positive[valid] + negative[valid]).mean()
                )
    alias = torch.stack(alias_terms).mean() if alias_terms else logits.new_tensor(0.0)

    state_index = torch.arange(N_STATES_ALIAS_V2, device=logits.device)[None]
    hard_valid = state_valid[None] & ((state_index - label_index[:, None]).abs() > 5)
    hard_logits = logits.masked_fill(~hard_valid, -1e4)
    hard = hard_logits.topk(HARDEST_NEGATIVES_ALIAS_V2, dim=1).values
    hard_loss = F.relu(HARD_MARGIN_ALIAS_V2 - positive[:, None] + hard).mean()
    total = ce + ALIAS_WEIGHT_ALIAS_V2 * alias + HARD_WEIGHT_ALIAS_V2 * hard_loss
    return total, ce.detach(), alias.detach(), hard_loss.detach()


def _retrieval_payload_alias_v2(scores, sample):
    scores = np.asarray(scores, np.float64)
    labels = sample.labels.astype(np.float64)
    prediction_index = np.argmax(scores, axis=1)
    prediction = STATE_OFFSETS_ALIAS_V2[prediction_index].astype(np.float64)
    probability = np.exp(scores - scores.max(axis=1, keepdims=True))
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    posterior_mean = probability @ STATE_OFFSETS_ALIAS_V2.astype(np.float64)
    truth_score = scores[np.arange(len(scores)), sample.label_index]
    rank = 1 + np.sum(scores > truth_score[:, None], axis=1)
    top5 = np.argpartition(scores, -5, axis=1)[:, -5:]
    top8 = np.argpartition(scores, -8, axis=1)[:, -8:]
    alias_wins = np.full(len(scores), np.nan, np.float64)
    for row, truth_index in enumerate(sample.label_index):
        aliases = []
        for sign in (-1, 1):
            alias_index = int(truth_index + sign * 24)
            if 0 <= alias_index < N_STATES_ALIAS_V2 and sample.state_valid[alias_index]:
                aliases.append(scores[row, alias_index])
        if aliases:
            alias_wins[row] = float(truth_score[row] > max(aliases))
    return {
        "well": np.full(len(labels), sample.well, dtype="U16"),
        "label": labels,
        "prediction": prediction,
        "posterior_mean": posterior_mean,
        "rank": rank.astype(np.int32),
        "top1_hit": np.abs(prediction - labels) <= 5.0,
        "top5_hit": np.any(
            np.abs(STATE_OFFSETS_ALIAS_V2[top5] - labels[:, None]) <= 5.0, axis=1
        ),
        "top8_hit": np.any(
            np.abs(STATE_OFFSETS_ALIAS_V2[top8] - labels[:, None]) <= 5.0, axis=1
        ),
        "alias_win": alias_wins,
        "deep": sample.distance_ft >= 2000.0,
        "distance_ft": sample.distance_ft.astype(np.float64),
    }


def _merge_alias_v2(parts):
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _metrics_alias_v2(payload, requested_rows):
    label = payload["label"]
    prediction = payload["prediction"]
    posterior = payload["posterior_mean"]
    deep = payload["deep"]
    alias = np.isfinite(payload["alias_win"])
    metrics = {
        "rows": len(label),
        "requested_rows": int(requested_rows),
        "coverage": float(len(label) / max(requested_rows, 1)),
        "wells": int(len(np.unique(payload["well"]))),
        "argmax_rmse": float(np.sqrt(np.mean((prediction - label) ** 2))),
        "posterior_mean_rmse": float(np.sqrt(np.mean((posterior - label) ** 2))),
        "top1_within5": float(np.mean(payload["top1_hit"])),
        "top5_within5": float(np.mean(payload["top5_hit"])),
        "top8_within5": float(np.mean(payload["top8_hit"])),
        "median_rank": float(np.median(payload["rank"])),
        "p90_rank": float(np.quantile(payload["rank"], 0.90)),
        "alias24_win_rate": float(np.mean(payload["alias_win"][alias])) if alias.any() else np.nan,
        "deep_rows": int(deep.sum()),
        "deep_top8_within5": float(np.mean(payload["top8_hit"][deep])) if deep.any() else np.nan,
        "deep_median_rank": float(np.median(payload["rank"][deep])) if deep.any() else np.nan,
    }
    return metrics


@torch.no_grad()
def _evaluate_alias_v2(model, samples, wells, mode="full"):
    model.eval()
    parts = []
    requested = 0
    donor_wells = list(wells[1:]) + [wells[0]]
    for well, donor_well in zip(wells, donor_wells):
        sample = samples[well]
        reference = sample.reference
        state_valid = sample.state_valid
        no_gr = mode == "no_gr"
        if mode == "shuffled_typewell":
            donor = samples[donor_well]
            reference = donor.reference
            state_valid = donor.state_valid
        tensors = _sample_tensors_alias_v2(
            sample, no_gr=no_gr, reference=reference, state_valid=state_valid
        )
        query, reference_t, valid_t, _, _, prior, distance, z, _ = tensors
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(
                query,
                reference_t,
                valid_t,
                prior,
                distance,
                z,
                prior_jitter=False,
            )
        parts.append(_retrieval_payload_alias_v2(logits.float().cpu().numpy(), sample))
        requested += sample.requested_rows
        del tensors, query, reference_t, valid_t, prior, distance, z, logits
    payload = _merge_alias_v2(parts)
    return _metrics_alias_v2(payload, requested), payload


def _prior_baseline_alias_v2(samples, wells):
    parts = []
    requested = 0
    for well in wells:
        sample = samples[well]
        scores = -0.5 * (
            (STATE_OFFSETS_ALIAS_V2[None] - sample.prior_offset[:, None]) / 16.0
        ) ** 2
        scores[:, ~sample.state_valid] = -1e4
        parts.append(_retrieval_payload_alias_v2(scores, sample))
        requested += sample.requested_rows
    payload = _merge_alias_v2(parts)
    return _metrics_alias_v2(payload, requested)


def _train_epoch_alias_v2(model, samples, wells, optimizer, scaler, epoch):
    model.train()
    rng = np.random.default_rng(SEED_ALIAS_V2 + epoch * 1009)
    order = rng.permutation(len(wells))
    mean_rows = float(np.mean([samples[well].eval_rows for well in wells]))
    totals = {"loss": 0.0, "ce": 0.0, "alias": 0.0, "hard": 0.0, "skipped": 0}
    for step, position in enumerate(order, 1):
        sample = samples[wells[int(position)]]
        tensors = _sample_tensors_alias_v2(sample)
        query, reference, valid, labels, label_index, prior, distance, z, row_weight = tensors
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(
                query, reference, valid, prior, distance, z, prior_jitter=True
            )
            loss, ce, alias, hard = _loss_alias_v2(
                logits, labels, label_index, row_weight, valid
            )
            loss = loss * sample.eval_rows / max(mean_rows, 1.0)
        before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_ALIAS_V2)
        scaler.step(optimizer)
        scaler.update()
        totals["skipped"] += int(float(scaler.get_scale()) < before)
        totals["loss"] += float(loss.detach().cpu())
        totals["ce"] += float(ce.cpu())
        totals["alias"] += float(alias.cpu())
        totals["hard"] += float(hard.cpu())
        if step % 100 == 0 or step == len(wells):
            print(
                f"ALIAS-V2 epoch={epoch} wells={step}/{len(wells)} "
                f"loss={totals['loss']/step:.4f}",
                flush=True,
            )
        del tensors, query, reference, valid, labels, label_index
        del prior, distance, z, row_weight, logits, loss, ce, alias, hard
    return {
        "train_loss": totals["loss"] / len(wells),
        "train_ce": totals["ce"] / len(wells),
        "train_alias": totals["alias"] / len(wells),
        "train_hard": totals["hard"] / len(wells),
        "skipped_steps": totals["skipped"],
    }


started_alias_v2 = time.time()
_seed_alias_v2(SEED_ALIAS_V2)
fold_frame_alias_v2 = folds[["well", "fold"]].copy()
fold_frame_alias_v2["well"] = fold_frame_alias_v2["well"].astype(str)
fold_frame_alias_v2["fold"] = pd.to_numeric(
    fold_frame_alias_v2["fold"], errors="raise"
).astype(int)
core_wells_alias_v2 = sorted(
    fold_frame_alias_v2.loc[
        fold_frame_alias_v2["fold"].isin(CORE_FOLDS_ALIAS_V2), "well"
    ].tolist()
)
valid_wells_alias_v2 = sorted(
    fold_frame_alias_v2.loc[
        fold_frame_alias_v2["fold"].eq(VALID_FOLD_ALIAS_V2), "well"
    ].tolist()
)

samples_alias_v2 = {}
all_wells_alias_v2 = core_wells_alias_v2 + valid_wells_alias_v2
for index, well in enumerate(all_wells_alias_v2, 1):
    fold = int(fold_frame_alias_v2.set_index("well").loc[well, "fold"])
    samples_alias_v2[well] = _build_alias_sample_v2(well, fold)
    if index % 50 == 0 or index == len(all_wells_alias_v2):
        print(f"build alias samples {index}/{len(all_wells_alias_v2)}", flush=True)

guards_alias_v2 = {
    "core_wells": len(core_wells_alias_v2),
    "valid_wells": len(valid_wells_alias_v2),
    "minimum_rows": int(min(len(sample.labels) for sample in samples_alias_v2.values())),
    "mean_coverage": float(
        np.mean([
            len(sample.labels) / max(sample.requested_rows, 1)
            for sample in samples_alias_v2.values()
        ])
    ),
    "minimum_valid_states": int(
        min(sample.state_valid.sum() for sample in samples_alias_v2.values())
    ),
    "query_scales_md": QUERY_HALF_WIDTHS_ALIAS_V2.tolist(),
    "reference_scales_tvt": REFERENCE_HALF_WIDTHS_ALIAS_V2.tolist(),
}
print("ALIAS-V2 GUARDS", guards_alias_v2, flush=True)
if (
    guards_alias_v2["minimum_rows"] < 64
    or guards_alias_v2["mean_coverage"] < 0.95
    or guards_alias_v2["minimum_valid_states"] < 129
):
    raise RuntimeError(f"Alias-V2 representation guard failed: {guards_alias_v2}")

prior_metrics_alias_v2 = _prior_baseline_alias_v2(samples_alias_v2, valid_wells_alias_v2)
model_alias_v2 = AliasAwareAbsoluteMatcherV2().to(device)
optimizer_alias_v2 = torch.optim.AdamW(
    model_alias_v2.parameters(),
    lr=LEARNING_RATE_ALIAS_V2,
    weight_decay=WEIGHT_DECAY_ALIAS_V2,
)
scheduler_alias_v2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_alias_v2,
    T_max=MAX_EPOCHS_ALIAS_V2,
    eta_min=MIN_LEARNING_RATE_ALIAS_V2,
)
scaler_alias_v2 = torch.amp.GradScaler(
    "cuda", enabled=device.type == "cuda", init_scale=512.0
)

history_alias_v2 = []
best_alias_v2 = None
best_state_alias_v2 = None
stale_alias_v2 = 0
for epoch_alias_v2 in range(1, MAX_EPOCHS_ALIAS_V2 + 1):
    train_metrics_alias_v2 = _train_epoch_alias_v2(
        model_alias_v2,
        samples_alias_v2,
        core_wells_alias_v2,
        optimizer_alias_v2,
        scaler_alias_v2,
        epoch_alias_v2,
    )
    scheduler_alias_v2.step()
    valid_metrics_alias_v2, _ = _evaluate_alias_v2(
        model_alias_v2, samples_alias_v2, valid_wells_alias_v2, mode="full"
    )
    row_alias_v2 = {
        "epoch": epoch_alias_v2,
        **train_metrics_alias_v2,
        **{f"valid_{key}": value for key, value in valid_metrics_alias_v2.items()},
        "learning_rate": optimizer_alias_v2.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started_alias_v2) / 60.0,
    }
    history_alias_v2.append(row_alias_v2)
    print("ALIAS-V2 EPOCH", row_alias_v2, flush=True)
    objective_alias_v2 = (
        valid_metrics_alias_v2["top8_within5"],
        -valid_metrics_alias_v2["median_rank"],
        valid_metrics_alias_v2["alias24_win_rate"],
        -valid_metrics_alias_v2["argmax_rmse"],
    )
    if best_alias_v2 is None or objective_alias_v2 > best_alias_v2["objective"]:
        best_alias_v2 = {
            "epoch": epoch_alias_v2,
            "objective": objective_alias_v2,
            "metrics": valid_metrics_alias_v2,
        }
        best_state_alias_v2 = {
            key: value.detach().cpu().clone()
            for key, value in model_alias_v2.state_dict().items()
        }
        stale_alias_v2 = 0
    else:
        stale_alias_v2 += 1
    if epoch_alias_v2 >= MIN_EPOCHS_ALIAS_V2 and stale_alias_v2 >= EARLY_STOP_ALIAS_V2:
        print("ALIAS-V2 early stopping", flush=True)
        break

history_frame_alias_v2 = pd.DataFrame(history_alias_v2)
history_frame_alias_v2.to_csv(OUT_DIR_ALIAS_V2 / "training_history.csv", index=False)
model_alias_v2.load_state_dict(best_state_alias_v2)
full_metrics_alias_v2, full_payload_alias_v2 = _evaluate_alias_v2(
    model_alias_v2, samples_alias_v2, valid_wells_alias_v2, mode="full"
)
no_gr_metrics_alias_v2, _ = _evaluate_alias_v2(
    model_alias_v2, samples_alias_v2, valid_wells_alias_v2, mode="no_gr"
)
shuffle_metrics_alias_v2, _ = _evaluate_alias_v2(
    model_alias_v2, samples_alias_v2, valid_wells_alias_v2, mode="shuffled_typewell"
)

checkpoint_alias_v2 = MODEL_DIR_ALIAS_V2 / "best_fold2_alias_matcher.pt"
torch.save(
    {
        "model": best_state_alias_v2,
        "epoch": int(best_alias_v2["epoch"]),
        "config": {
            "query_half_widths_md": QUERY_HALF_WIDTHS_ALIAS_V2,
            "reference_half_widths_tvt": REFERENCE_HALF_WIDTHS_ALIAS_V2,
            "state_offsets": STATE_OFFSETS_ALIAS_V2,
        },
        "metrics": full_metrics_alias_v2,
    },
    checkpoint_alias_v2,
)

no_gr_drop_alias_v2 = float(
    full_metrics_alias_v2["top8_within5"] - no_gr_metrics_alias_v2["top8_within5"]
)
shuffle_drop_alias_v2 = float(
    full_metrics_alias_v2["top8_within5"] - shuffle_metrics_alias_v2["top8_within5"]
)
retrieval_pass_alias_v2 = bool(
    full_metrics_alias_v2["top8_within5"] >= 0.75
    and full_metrics_alias_v2["median_rank"] <= 5.0
    and full_metrics_alias_v2["alias24_win_rate"] >= 0.75
    and full_metrics_alias_v2["deep_top8_within5"] >= 0.72
)
causal_pass_alias_v2 = bool(
    no_gr_drop_alias_v2 >= 0.08 and shuffle_drop_alias_v2 >= 0.15
)

verdict_alias_v2 = {
    "experiment": EXPERIMENT_ALIAS_V2,
    "stage": "fold2_alias_aware_longcontext_absolute_emission_gateway",
    "sample_unit": "sampled stations nested within complete physical wells",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "protocol": {
        "train_folds": list(CORE_FOLDS_ALIAS_V2),
        "validation_fold": VALID_FOLD_ALIAS_V2,
        "fold1_touched": False,
        "selected_epoch": int(best_alias_v2["epoch"]),
    },
    "parameters": int(sum(parameter.numel() for parameter in model_alias_v2.parameters())),
    "guards": guards_alias_v2,
    "prior_only": prior_metrics_alias_v2,
    "full": full_metrics_alias_v2,
    "no_gr": no_gr_metrics_alias_v2,
    "shuffled_typewell": shuffle_metrics_alias_v2,
    "controls": {
        "no_gr_top8_drop": no_gr_drop_alias_v2,
        "shuffled_typewell_top8_drop": shuffle_drop_alias_v2,
    },
    "gates": {
        "top8_within5_ge": 0.75,
        "median_rank_le": 5.0,
        "alias24_win_ge": 0.75,
        "deep_top8_within5_ge": 0.72,
        "no_gr_top8_drop_ge": 0.08,
        "shuffled_typewell_top8_drop_ge": 0.15,
    },
    "retrieval_pass": retrieval_pass_alias_v2,
    "causal_pass": causal_pass_alias_v2,
    "strict_pass": bool(retrieval_pass_alias_v2 and causal_pass_alias_v2),
    "checkpoint": str(checkpoint_alias_v2),
    "elapsed_minutes": (time.time() - started_alias_v2) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> replace Drift-V3 absolute volume and rerun frozen solver"
        if retrieval_pass_alias_v2 and causal_pass_alias_v2
        else "FAIL -> long-context local correspondence is still insufficient; switch to complete-well direct alignment"
    ),
}
_write_json(OUT_DIR_ALIAS_V2 / "verdict.json", verdict_alias_v2)

print("\nALIAS-V2 COMPARISON")
print(
    pd.DataFrame(
        [
            {"mode": "prior_only", **prior_metrics_alias_v2},
            {"mode": "full", **full_metrics_alias_v2},
            {"mode": "no_gr", **no_gr_metrics_alias_v2},
            {"mode": "shuffled_typewell", **shuffle_metrics_alias_v2},
        ]
    ).to_string(index=False),
    flush=True,
)
print("\nALIAS-V2 FINAL VERDICT")
print(json.dumps(verdict_alias_v2, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_ALIAS_V2, flush=True)

del optimizer_alias_v2, scheduler_alias_v2, scaler_alias_v2
gc.collect()
torch.cuda.empty_cache()

