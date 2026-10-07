"""Complete-well neural GR matcher with an anchored linear-chain CRF.

Run this in the same kernel after ``fullwell_global_tvt_unet_onefold_v0``.
It reuses only that cell's fold-clean data store and utility functions.  The
failed 2-D U-Net is discarded.  Every sample remains one complete well.

The model has separate 1-D encoders for the horizontal-well sequence and the
typewell/visible-heel reference sequence.  Their dot product is the learned
GR emission volume.  A differentiable, anchor-conditioned CRF integrates the
emissions over the complete well; the same transition law is used by the
Viterbi decoder.  No PF, tabular base, CP12, Gold, Neighbor, or neighbor-well
data is consumed.
"""

from pathlib import Path
import copy
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


REQUIRED_GLOBALS = (
    "OOF_ROOT",
    "fold_path",
    "train_store",
    "valid_store",
    "TRAIN_TAGS",
    "STATE_OFFSETS",
    "_interp_finite",
    "_robust_normalise",
    "_seed_everything",
)
missing_globals = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run fullwell_global_tvt_unet_onefold_v0 through data-store creation "
        "first; missing: " + repr(missing_globals)
    )

# Release the failed image model while preserving the loaded well records.
for stale_name in ("model", "audit_model", "parameter_probe"):
    stale_object = globals().pop(stale_name, None)
    if stale_object is not None:
        del stale_object
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


OUT_DIR = Path(OOF_ROOT) / "fullwell_neural_crf_onefold_v0"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

VALID_FOLD = 1
SEED = 20260720
N_ALIGN_COLUMNS = 384
STATE_STEP_FT = float(np.median(np.diff(STATE_OFFSETS)))
MAX_TRANSITION_STATES = 16
TARGET_SIGMA_FT = 1.5
WIDTH = 64
EMBEDDING = 48
DROPOUT = 0.08
BATCH_SIZE = 6
GRADIENT_ACCUMULATION = 2
EPOCHS = 30
EARLY_STOPPING = 7
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 2e-4
AUXILIARY_CE_WEIGHT = 0.20
NEGATIVE_SCORE = -1.0e4
MEAN_SAMPLE_WEIGHT = float(
    np.mean([record["official_n_eval"] for record in train_store.records])
)

ALIGN_INDEX = np.unique(
    np.rint(np.linspace(0, len(train_store.records[0]["column_md"]) - 1, N_ALIGN_COLUMNS))
    .astype(np.int32)
)
if len(ALIGN_INDEX) != N_ALIGN_COLUMNS:
    raise RuntimeError(f"alignment index has {len(ALIGN_INDEX)} columns")

HORIZONTAL_CHANNELS = (
    "horizontal_gr",
    "horizontal_gr_valid",
    "horizontal_gr_d1",
    "horizontal_gr_d2",
    "md_fraction",
    "x_relative_kft",
    "y_relative_kft",
    "z_relative_kft",
    "dx_dmd",
    "dy_dmd",
    "dz_dmd",
    "static_gr_valid",
    "head_x_scaled",
    "head_y_scaled",
    "global_missing_fraction",
    "known_mask",
    "eval_mask",
    "md_since_anchor_kft",
    "known_fraction",
    "known_tvt_offset",
    "known_structure_offset",
)
REFERENCE_CHANNELS = (
    "typewell_gr",
    "typewell_valid",
    "typewell_gr_d1",
    "typewell_gr_d2",
    "heel_gr",
    "heel_valid",
    "heel_gr_d1",
    "heel_gr_d2",
    "state_offset",
    "typewell_position",
    "typewell_edge_margin",
)
ELIGIBLE_TAGS_BY_WELL = {}


def _finite_derivatives(values, valid):
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    filled = np.zeros(len(values), np.float64)
    if int(valid.sum()) >= 2:
        index = np.arange(len(values), dtype=np.float64)
        filled = np.interp(index, index[valid], values[valid])
    first = _robust_normalise(np.gradient(filled))
    second = _robust_normalise(np.gradient(np.gradient(filled)))
    first[~valid] = 0.0
    second[~valid] = 0.0
    return first.astype(np.float32), second.astype(np.float32)


def _compose_alignment(record, tag="official", mode="full", donor=None):
    cut = int(record["cuts"][tag])
    anchor = cut - 1
    anchor_md = float(record["md"][anchor])
    anchor_tvt = float(record["truth"][anchor])
    state_tvt = anchor_tvt + STATE_OFFSETS

    if mode == "shuffled_typewell":
        if donor is None:
            raise RuntimeError("shuffled_typewell requires a donor")
        tw_tvt = donor["tw_tvt"]
        tw_gr_source = donor["tw_gr"]
    else:
        tw_tvt = record["tw_tvt"]
        tw_gr_source = record["tw_gr"]
    typewell = _interp_finite(
        tw_tvt, tw_gr_source, state_tvt, left=np.nan, right=np.nan
    )
    typewell_valid = np.isfinite(typewell)
    typewell = np.nan_to_num(typewell, nan=0.0).astype(np.float32)
    typewell_d1, typewell_d2 = _finite_derivatives(typewell, typewell_valid)

    visible_valid = record["gr_finite"][:cut] & np.isfinite(record["truth"][:cut])
    if int(visible_valid.sum()) >= 16:
        heel = _interp_finite(
            record["truth"][:cut][visible_valid],
            record["gr"][:cut][visible_valid],
            state_tvt,
            left=np.nan,
            right=np.nan,
        )
    else:
        heel = np.full(len(state_tvt), np.nan, np.float64)
    heel_valid = np.isfinite(heel)
    heel = np.nan_to_num(heel, nan=0.0).astype(np.float32)
    heel_d1, heel_d2 = _finite_derivatives(heel, heel_valid)

    index = ALIGN_INDEX
    column_md = record["column_md"][index].astype(np.float32)
    column_z = record["column_z"][index].astype(np.float32)
    horizontal_gr = record["column_gr"][index].astype(np.float32)
    horizontal_valid = record["column_gr_valid"][index] > 0.5
    horizontal_d1, horizontal_d2 = _finite_derivatives(
        horizontal_gr, horizontal_valid
    )
    static_geometry = record["static_geometry"][:, index].astype(np.float32)
    target_offset = record["column_truth"][index] - anchor_tvt
    target_state = np.argmin(
        np.abs(STATE_OFFSETS[:, None] - target_offset[None, :]), axis=0
    ).astype(np.int64)
    target_coverage = np.isfinite(target_offset) & (
        np.abs(target_offset) <= float(np.max(np.abs(STATE_OFFSETS)))
    )
    known_mask = column_md <= anchor_md
    eval_mask = ~known_mask
    known_offset = np.where(
        known_mask, np.clip(target_offset / 128.0, -2.0, 2.0), 0.0
    )
    anchor_z = float(record["column_z"][np.argmin(np.abs(record["column_md"] - anchor_md))])
    structure_offset = target_offset + column_z - anchor_z
    known_structure = np.where(
        known_mask, np.clip(structure_offset / 256.0, -2.0, 2.0), 0.0
    )
    dynamic = np.stack(
        [
            known_mask.astype(np.float32),
            eval_mask.astype(np.float32),
            (column_md - anchor_md) / 1000.0,
            np.full(len(index), cut / len(record["md"]), np.float32),
            known_offset.astype(np.float32),
            known_structure.astype(np.float32),
        ],
        axis=0,
    )
    horizontal = np.concatenate(
        [
            np.stack(
                [
                    horizontal_gr,
                    horizontal_valid.astype(np.float32),
                    horizontal_d1,
                    horizontal_d2,
                ],
                axis=0,
            ),
            static_geometry,
            dynamic,
        ],
        axis=0,
    ).astype(np.float32)

    tw_span = max(float(tw_tvt[-1] - tw_tvt[0]), 1.0)
    typewell_position = (
        state_tvt - 0.5 * (float(tw_tvt[0]) + float(tw_tvt[-1]))
    ) / tw_span
    edge_margin = np.minimum(state_tvt - tw_tvt[0], tw_tvt[-1] - state_tvt)
    reference = np.stack(
        [
            typewell,
            typewell_valid.astype(np.float32),
            typewell_d1,
            typewell_d2,
            heel,
            heel_valid.astype(np.float32),
            heel_d1,
            heel_d2,
            STATE_OFFSETS / 128.0,
            typewell_position,
            np.clip(edge_margin / 128.0, -2.0, 2.0),
        ],
        axis=0,
    ).astype(np.float32)

    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        reference[[0, 2, 3, 4, 6, 7]] = 0.0
    elif mode not in ("full", "shuffled_typewell"):
        raise ValueError(mode)

    if horizontal.shape[0] != len(HORIZONTAL_CHANNELS):
        raise RuntimeError(f"horizontal channels: {horizontal.shape}")
    if reference.shape[0] != len(REFERENCE_CHANNELS):
        raise RuntimeError(f"reference channels: {reference.shape}")
    if not np.isfinite(horizontal).all() or not np.isfinite(reference).all():
        raise RuntimeError(f"non-finite alignment input for {record['well']}:{tag}")
    return {
        "horizontal": horizontal,
        "reference": reference,
        "target_offset": target_offset.astype(np.float32),
        "target_state": target_state,
        "target_coverage": target_coverage,
        "known_mask": known_mask,
        "eval_mask": eval_mask,
        "column_md": column_md,
        "column_z": column_z,
        "anchor_tvt": anchor_tvt,
        "anchor_md": anchor_md,
        "sample_weight": float(record["official_n_eval"]),
        "tag": tag,
    }


def _panel_crf_eligible(panel):
    eval_mask = panel["eval_mask"]
    if not bool(np.all(panel["target_coverage"][eval_mask])):
        return False
    states = panel["target_state"][eval_mask]
    zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    transitions = np.abs(np.diff(np.concatenate([[zero_state], states])))
    return bool(np.all(transitions <= MAX_TRANSITION_STATES))


class AlignmentDataset(Dataset):
    def __init__(self, store, training):
        self.store = store
        self.training = bool(training)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.store.records)

    def __getitem__(self, index):
        record = self.store.records[index]
        if self.training:
            preferred = (record["stable_slot"] + self.epoch) % len(TRAIN_TAGS)
            eligible = ELIGIBLE_TAGS_BY_WELL.get(record["well"], ("official",))
            ordered = [
                TRAIN_TAGS[(preferred + offset) % len(TRAIN_TAGS)]
                for offset in range(len(TRAIN_TAGS))
            ]
            tag = next((candidate for candidate in ordered if candidate in eligible), None)
            if tag is None:
                raise RuntimeError(f"{record['well']}: no CRF-eligible cut")
        else:
            tag = "official"
        panel = _compose_alignment(record, tag=tag)
        if not _panel_crf_eligible(panel):
            raise RuntimeError(f"{record['well']}:{tag}: ineligible CRF panel selected")
        return {
            "horizontal": torch.from_numpy(panel["horizontal"]),
            "reference": torch.from_numpy(panel["reference"]),
            "target_offset": torch.from_numpy(panel["target_offset"]),
            "target_state": torch.from_numpy(panel["target_state"]),
            "known_mask": torch.from_numpy(panel["known_mask"]),
            "eval_mask": torch.from_numpy(panel["eval_mask"]),
            "sample_weight": torch.tensor(panel["sample_weight"], dtype=torch.float32),
        }


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, 1)
        self.output = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = values
        values = self.depthwise(values)
        first, gate = self.pointwise(values).chunk(2, dim=1)
        values = first * torch.sigmoid(gate)
        values = self.dropout(self.output(values))
        return F.gelu(self.norm(values + residual))


class SequenceEncoder(nn.Module):
    def __init__(self, input_channels, width, embedding, dilations):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, 5, padding=2),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [TCNBlock(width, dilation, DROPOUT) for dilation in dilations]
        )
        self.global_projection = nn.Sequential(
            nn.Conv1d(width * 2, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 1),
        )
        self.embedding = nn.Conv1d(width, embedding, 1)

    def forward(self, values):
        hidden = self.stem(values)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = torch.cat(
            [hidden.mean(dim=-1, keepdim=True), hidden.amax(dim=-1, keepdim=True)],
            dim=1,
        )
        hidden = hidden + self.global_projection(pooled)
        embedding = F.normalize(self.embedding(hidden), dim=1, eps=1e-6)
        return hidden, embedding


class FullWellNeuralCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.horizontal_encoder = SequenceEncoder(
            len(HORIZONTAL_CHANNELS), WIDTH, EMBEDDING, (1, 2, 4, 8, 16, 32, 64)
        )
        self.reference_encoder = SequenceEncoder(
            len(REFERENCE_CHANNELS), WIDTH, EMBEDDING, (1, 2, 4, 8, 16, 32)
        )
        self.reference_bias = nn.Conv1d(WIDTH, 1, 1)
        self.raw_weights = nn.Parameter(torch.tensor([0.7, 0.3, 0.25, 0.10, 0.20, 0.08]))
        self.log_similarity_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        self.log_transition_penalty = nn.Parameter(torch.tensor(-2.0))
        self.log_prior_precision = nn.Parameter(torch.tensor(-2.5))

    def forward(self, horizontal, reference):
        horizontal_hidden, query = self.horizontal_encoder(horizontal)
        reference_hidden, key = self.reference_encoder(reference)
        similarity_scale = torch.exp(self.log_similarity_scale).clamp(0.5, 30.0)
        similarity = torch.einsum("bdw,bdh->bhw", query, key) * similarity_scale

        h_gr = horizontal[:, 0, None, :]
        h_d1 = horizontal[:, 2, None, :]
        h_d2 = horizontal[:, 3, None, :]
        tw_gr = reference[:, 0, :, None]
        tw_d1 = reference[:, 2, :, None]
        tw_d2 = reference[:, 3, :, None]
        heel_gr = reference[:, 4, :, None]
        weights = F.softplus(self.raw_weights)
        raw = (
            weights[0] * -torch.abs(h_gr - tw_gr)
            + weights[1] * h_gr * tw_gr
            + weights[2] * -torch.abs(h_d1 - tw_d1)
            + weights[3] * h_d1 * tw_d1
            + weights[4] * -torch.abs(h_gr - heel_gr)
            + weights[5] * -torch.abs(h_d2 - tw_d2)
        )
        state_bias = self.reference_bias(reference_hidden).transpose(1, 2)
        state_offset = reference[:, 8, :, None]
        prior_precision = F.softplus(self.log_prior_precision)
        prior = -prior_precision * state_offset.square()
        return similarity + raw + state_bias + prior

    def transition_penalty(self):
        return F.softplus(self.log_transition_penalty).clamp(1e-4, 4.0)


def _shift_alpha(alpha, delta):
    if delta == 0:
        return alpha
    if delta > 0:
        return F.pad(alpha[:, :-delta], (delta, 0), value=NEGATIVE_SCORE)
    return F.pad(alpha[:, -delta:], (0, -delta), value=NEGATIVE_SCORE)


def _clamp_known(alpha, target_state, known):
    selected = alpha.gather(1, target_state[:, None])
    clamped = torch.full_like(alpha, NEGATIVE_SCORE).scatter(
        1, target_state[:, None], selected
    )
    return torch.where(known[:, None], clamped, alpha)


def _crf_loss(
    emission,
    target_state,
    known_mask,
    eval_mask,
    sample_weight,
    transition_penalty,
):
    emission = emission.float()
    target_state = target_state.long()
    eval_mask = eval_mask.float()
    batch, states, columns = emission.shape
    penalty = transition_penalty.float()
    deltas = list(range(-MAX_TRANSITION_STATES, MAX_TRANSITION_STATES + 1))
    transition_scores = emission.new_tensor(
        [-float(delta * delta) for delta in deltas]
    ) * penalty

    zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    alpha = torch.full(
        (batch, states), NEGATIVE_SCORE, dtype=emission.dtype, device=emission.device
    )
    alpha[:, zero_state] = 0.0
    truth_emission = emission.new_zeros(batch)
    truth_transition = emission.new_zeros(batch)
    previous_truth_state = torch.full(
        (batch,), zero_state, dtype=torch.long, device=emission.device
    )
    previously_active = torch.zeros(batch, dtype=torch.bool, device=emission.device)

    for column in range(columns):
        candidates = torch.stack(
            [
                _shift_alpha(alpha, delta) + transition_scores[index]
                for index, delta in enumerate(deltas)
            ],
            dim=0,
        )
        advanced = emission[:, :, column] + torch.logsumexp(candidates, dim=0)
        active = eval_mask[:, column] > 0.5
        alpha = torch.where(active[:, None], advanced, alpha)

        current_truth_state = target_state[:, column]
        truth_emission = truth_emission + torch.where(
            active,
            emission[:, :, column].gather(
                1, current_truth_state[:, None]
            ).squeeze(1),
            torch.zeros_like(truth_emission),
        )
        truth_delta = current_truth_state - previous_truth_state
        if torch.any(active & (torch.abs(truth_delta) > MAX_TRANSITION_STATES)):
            raise RuntimeError("truth eval transition escaped the CRF band")
        truth_transition = truth_transition + torch.where(
            active,
            -penalty * truth_delta.float().square(),
            torch.zeros_like(truth_transition),
        )
        previous_truth_state = torch.where(
            active, current_truth_state, previous_truth_state
        )
        previously_active = previously_active | active
    log_partition = torch.logsumexp(alpha, dim=1)
    if not torch.all(previously_active):
        raise RuntimeError("a CRF sample has no eval columns")
    eval_columns = eval_mask.sum(dim=1).clamp_min(1.0)
    nll = (log_partition - truth_emission - truth_transition) / eval_columns

    log_probability = F.log_softmax(emission, dim=1)
    point_nll = -log_probability.gather(
        1, target_state[:, None, :]
    ).squeeze(1)
    point_nll = (point_nll * eval_mask).sum(dim=1) / eval_columns
    panel_scale = (sample_weight / MEAN_SAMPLE_WEIGHT).clamp(0.25, 4.0)
    crf = (nll * panel_scale).mean()
    auxiliary = (point_nll * panel_scale).mean()
    total = crf + AUXILIARY_CE_WEIGHT * auxiliary
    return total, crf, auxiliary


def _viterbi_numpy(emission, start_column, start_state, penalty):
    emission = np.asarray(emission, np.float64)
    states, columns = emission.shape
    length = columns - int(start_column)
    if length <= 0:
        raise RuntimeError("empty eval suffix")
    deltas = np.arange(-MAX_TRANSITION_STATES, MAX_TRANSITION_STATES + 1)
    previous = np.full(states, -np.inf, np.float64)
    previous[int(start_state)] = 0.0
    back = np.zeros((length, states), np.int16)
    state_index = np.arange(states)
    for local_column, column in enumerate(range(int(start_column), columns)):
        candidates = np.full((len(deltas), states), -np.inf, np.float64)
        predecessors = np.zeros((len(deltas), states), np.int16)
        for row, delta in enumerate(deltas):
            source = state_index - delta
            valid = (source >= 0) & (source < states)
            candidates[row, valid] = (
                previous[source[valid]] - float(penalty) * float(delta * delta)
            )
            predecessors[row, valid] = source[valid]
        best_row = np.argmax(candidates, axis=0)
        previous = emission[:, column] + candidates[best_row, state_index]
        back[local_column] = predecessors[best_row, state_index]
    path = np.full(columns, int(start_state), np.int32)
    state = int(np.argmax(previous))
    for local_column in range(length - 1, -1, -1):
        column = int(start_column) + local_column
        path[column] = state
        state = int(back[local_column, state])
    return path


@torch.no_grad()
def _predict_record(model, store, index, mode):
    record = store.records[index]
    donor = store.records[(index + max(1, len(store.records) // 3)) % len(store.records)]
    panel = _compose_alignment(
        record,
        tag="official",
        mode=mode,
        donor=donor if mode == "shuffled_typewell" else None,
    )
    horizontal = torch.from_numpy(panel["horizontal"])[None].to(device)
    reference = torch.from_numpy(panel["reference"])[None].to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        emission = model(horizontal, reference)
    emission = emission[0].float().cpu().numpy()
    first_eval = int(np.argmax(panel["eval_mask"]))
    if first_eval <= 0:
        raise RuntimeError(f"{record['well']}: no visible alignment column")
    # The latent path is defined relative to the exact prediction-start
    # anchor, so both training and deployment start from offset state zero.
    start_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    penalty = float(model.transition_penalty().detach().cpu())
    state_path = _viterbi_numpy(emission, first_eval, start_state, penalty)
    offset = STATE_OFFSETS[state_path].astype(np.float64)
    offset[panel["known_mask"]] = panel["target_offset"][panel["known_mask"]]
    column_tvt = float(panel["anchor_tvt"]) + offset
    column_s = column_tvt + panel["column_z"]
    prediction = (
        np.interp(record["eval_md"], panel["column_md"], column_s)
        - record["eval_z"]
    )
    anchor_prediction = np.full(len(prediction), panel["anchor_tvt"], np.float64)
    return prediction, anchor_prediction


@torch.no_grad()
def _evaluate(model, store, mode="full", return_rows=False):
    model.eval()
    errors = []
    anchor_errors = []
    well_rows = []
    row_frames = []
    for index, record in enumerate(store.records):
        prediction, anchor = _predict_record(model, store, index, mode)
        truth = record["eval_truth"].astype(np.float64)
        error = prediction - truth
        anchor_error = anchor - truth
        errors.append(error)
        anchor_errors.append(anchor_error)
        well_rows.append(
            {
                "well": record["well"],
                "rows": len(error),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "anchor_rmse": float(np.sqrt(np.mean(anchor_error**2))),
            }
        )
        if return_rows:
            row_frames.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": record["eval_rows"],
                        "truth": truth,
                        "prediction": prediction,
                    }
                )
            )
    errors = np.concatenate(errors)
    anchor_errors = np.concatenate(anchor_errors)
    by_well = pd.DataFrame(well_rows)
    metrics = {
        "rows": int(len(errors)),
        "wells": int(len(by_well)),
        "anchor_rmse": float(np.sqrt(np.mean(anchor_errors**2))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "gain_vs_anchor": float(
            np.sqrt(np.mean(anchor_errors**2)) - np.sqrt(np.mean(errors**2))
        ),
        "mae": float(np.mean(np.abs(errors))),
        "median_well_rmse": float(by_well["rmse"].median()),
        "p95_well_rmse": float(by_well["rmse"].quantile(0.95)),
    }
    rows = pd.concat(row_frames, ignore_index=True) if return_rows else None
    return metrics, by_well, rows


def _transition_guards():
    global ELIGIBLE_TAGS_BY_WELL
    absolute_offsets = []
    transitions = []
    by_tag = {
        tag: {
            "rows": 0,
            "covered_rows": 0,
            "transitions": 0,
            "covered_transitions": 0,
            "panels": 0,
            "eligible_panels": 0,
        }
        for tag in TRAIN_TAGS
    }
    eligible_by_well = {record["well"]: [] for record in train_store.records}
    train_panel_count = 0
    eligible_train_panel_count = 0
    tasks = [
        (record, TRAIN_TAGS, True) for record in train_store.records
    ] + [
        (record, ("official",), False) for record in valid_store.records
    ]
    for record, tags, is_train in tasks:
        for tag in tags:
            panel = _compose_alignment(record, tag=tag)
            eval_mask = panel["eval_mask"]
            eval_offsets = panel["target_offset"][eval_mask]
            eval_states = panel["target_state"][eval_mask]
            zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
            transition = np.abs(
                np.diff(np.concatenate([[zero_state], eval_states]))
            )
            absolute_offsets.append(np.abs(eval_offsets))
            transitions.append(transition)
            report = by_tag[tag]
            report["rows"] += int(len(eval_offsets))
            report["covered_rows"] += int(
                np.sum(np.abs(eval_offsets) <= np.max(np.abs(STATE_OFFSETS)))
            )
            report["transitions"] += int(len(transition))
            report["covered_transitions"] += int(
                np.sum(transition <= MAX_TRANSITION_STATES)
            )
            report["panels"] += 1
            eligible = _panel_crf_eligible(panel)
            report["eligible_panels"] += int(eligible)
            if is_train:
                train_panel_count += 1
                eligible_train_panel_count += int(eligible)
                if eligible:
                    eligible_by_well[record["well"]].append(tag)
    absolute_offsets = np.concatenate(absolute_offsets)
    transitions = np.concatenate(transitions)
    ELIGIBLE_TAGS_BY_WELL = {
        well: tuple(tags) for well, tags in eligible_by_well.items()
    }
    tag_summary = {}
    for tag, report in by_tag.items():
        tag_summary[tag] = {
            "row_axis_coverage": report["covered_rows"] / max(report["rows"], 1),
            "transition_coverage": report["covered_transitions"]
            / max(report["transitions"], 1),
            "eligible_panel_fraction": report["eligible_panels"]
            / max(report["panels"], 1),
            "panels": report["panels"],
        }
    official = tag_summary["official"]
    wells_without_eligible = sum(
        not len(tags) for tags in ELIGIBLE_TAGS_BY_WELL.values()
    )
    return {
        "axis_coverage": float(np.mean(absolute_offsets <= np.max(np.abs(STATE_OFFSETS)))),
        "offset_p99_ft": float(np.quantile(absolute_offsets, 0.99)),
        "transition_band_coverage": float(
            np.mean(transitions <= MAX_TRANSITION_STATES)
        ),
        "transition_p99_states": float(np.quantile(transitions, 0.99)),
        "transition_max_states": int(np.max(transitions)),
        "official_axis_coverage": float(official["row_axis_coverage"]),
        "official_transition_coverage": float(official["transition_coverage"]),
        "official_eligible_panel_fraction": float(official["eligible_panel_fraction"]),
        "eligible_train_panel_fraction": float(
            eligible_train_panel_count / max(train_panel_count, 1)
        ),
        "wells_without_eligible_cut": int(wells_without_eligible),
        "by_tag": tag_summary,
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
_seed_everything(SEED)
guards = _transition_guards()
if (
    guards["official_axis_coverage"] < 0.9999
    or guards["official_transition_coverage"] < 0.9999
    or guards["official_eligible_panel_fraction"] < 0.99
    or guards["eligible_train_panel_fraction"] < 0.90
    or guards["wells_without_eligible_cut"] > 0
):
    raise RuntimeError(f"CRF representation guard failed: {guards}")

probe_panel = _compose_alignment(train_store.records[0], tag="official")
probe_model = FullWellNeuralCRF().to(device)
parameters = int(sum(parameter.numel() for parameter in probe_model.parameters()))
active_model = probe_model
probe_emission = probe_model(
    torch.from_numpy(probe_panel["horizontal"])[None].to(device),
    torch.from_numpy(probe_panel["reference"])[None].to(device),
)
probe_loss, probe_crf, probe_ce = _crf_loss(
    probe_emission,
    torch.from_numpy(probe_panel["target_state"])[None].to(device),
    torch.from_numpy(probe_panel["known_mask"])[None].to(device),
    torch.from_numpy(probe_panel["eval_mask"])[None].to(device),
    torch.ones(1, device=device),
    probe_model.transition_penalty(),
)
if not torch.isfinite(probe_loss):
    raise RuntimeError("non-finite differentiable CRF smoke loss")
probe_loss.backward()
guards.update(
    {
        "input_horizontal": list(probe_panel["horizontal"].shape),
        "input_reference": list(probe_panel["reference"].shape),
        "finite_input": bool(
            np.isfinite(probe_panel["horizontal"]).all()
            and np.isfinite(probe_panel["reference"]).all()
        ),
        "smoke_loss": float(probe_loss.detach().cpu()),
        "smoke_crf": float(probe_crf.detach().cpu()),
        "smoke_ce": float(probe_ce.detach().cpu()),
    }
)
del probe_model, probe_emission, probe_loss, probe_crf, probe_ce
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

print(
    {
        "experiment": "fullwell_neural_crf_onefold_v0",
        "sample_unit": "complete well",
        "standalone": True,
        "device": str(device),
        "train_wells": len(train_store.records),
        "valid_wells": len(valid_store.records),
        "columns": N_ALIGN_COLUMNS,
        "states": len(STATE_OFFSETS),
        "max_transition_states": MAX_TRANSITION_STATES,
        "parameters": parameters,
        "guards": guards,
    },
    flush=True,
)


def _train_model():
    global active_model
    _seed_everything(SEED)
    active_model = FullWellNeuralCRF().to(device)
    optimizer = torch.optim.AdamW(
        active_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    dataset = AlignmentDataset(train_store, training=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    checkpoint_path = MODEL_DIR / f"seed_{SEED}.pt"
    best_rmse = np.inf
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch - 1)
        active_model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_rows = []
        for batch_index, batch in enumerate(loader, 1):
            horizontal = batch["horizontal"].to(device, non_blocking=True)
            reference = batch["reference"].to(device, non_blocking=True)
            target_state = batch["target_state"].to(device, non_blocking=True)
            known_mask = batch["known_mask"].to(device, non_blocking=True)
            eval_mask = batch["eval_mask"].to(device, non_blocking=True)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                emission = active_model(horizontal, reference)
            loss, crf, auxiliary = _crf_loss(
                emission,
                target_state,
                known_mask,
                eval_mask,
                sample_weight,
                active_model.transition_penalty(),
            )
            scaler.scale(loss / GRADIENT_ACCUMULATION).backward()
            should_step = (
                batch_index % GRADIENT_ACCUMULATION == 0
                or batch_index == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(active_model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "crf": float(crf.detach().cpu()),
                    "auxiliary_ce": float(auxiliary.detach().cpu()),
                }
            )
        scheduler.step()
        metrics, by_well, _ = _evaluate(active_model, valid_store, mode="full")
        row = {
            "epoch": epoch,
            **{
                f"train_{column}": float(np.mean([item[column] for item in batch_rows]))
                for column in batch_rows[0]
            },
            **{f"valid_{key}": value for key, value in metrics.items()},
            "transition_penalty": float(
                active_model.transition_penalty().detach().cpu()
            ),
            "similarity_scale": float(
                torch.exp(active_model.log_similarity_scale).detach().cpu()
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("NEURAL-CRF EPOCH", row, flush=True)
        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": active_model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "seed": SEED,
                    "horizontal_channels": HORIZONTAL_CHANNELS,
                    "reference_channels": REFERENCE_CHANNELS,
                },
                checkpoint_path,
            )
            by_well.to_csv(OUT_DIR / "best_bywell.csv", index=False)
        else:
            stale += 1
        if epoch >= 10 and stale >= EARLY_STOPPING:
            print(f"neural CRF early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    active_model.load_state_dict(checkpoint["model"])
    active_model.eval()
    metrics, by_well, rows = _evaluate(
        active_model, valid_store, mode="full", return_rows=True
    )
    by_well.to_csv(OUT_DIR / "final_bywell.csv", index=False)
    rows.to_pickle(OUT_DIR / "fold1_predictions.pkl")
    return active_model, {
        "best_epoch": int(checkpoint["epoch"]),
        **metrics,
        "runtime_minutes": (time.time() - started) / 60.0,
    }


active_model, result = _train_model()
intervention_rows = []
for mode in ("full", "no_gr", "shuffled_typewell"):
    metrics, _, rows = _evaluate(
        active_model, valid_store, mode=mode, return_rows=True
    )
    rows.to_pickle(OUT_DIR / f"intervention_{mode}.pkl")
    intervention_rows.append({"mode": mode, **metrics})
interventions = pd.DataFrame(intervention_rows)
full_rmse = float(interventions.loc[interventions["mode"] == "full", "rmse"].iloc[0])
interventions["degradation_vs_full"] = interventions["rmse"] - full_rmse
interventions.to_csv(OUT_DIR / "interventions.csv", index=False)
no_gr_degradation = float(
    interventions.loc[interventions["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
shuffle_degradation = float(
    interventions.loc[
        interventions["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)
architecture_pass = bool(result["rmse"] <= 7.0)
target_pass = bool(result["rmse"] <= 5.5)
causal_pass = bool(no_gr_degradation >= 0.30 and shuffle_degradation >= 0.30)
strict_pass = bool(architecture_pass and causal_pass)
verdict = {
    "experiment": "fullwell_neural_crf_onefold_v0",
    "sample_unit": "complete well",
    "standalone": True,
    "fold": VALID_FOLD,
    "fold_source": str(fold_path),
    "train_wells": len(train_store.records),
    "valid_wells": len(valid_store.records),
    "columns": N_ALIGN_COLUMNS,
    "states": len(STATE_OFFSETS),
    "parameters": parameters,
    "guards": guards,
    "result": result,
    "no_gr_degradation": no_gr_degradation,
    "shuffled_typewell_degradation": shuffle_degradation,
    "architecture_pass": architecture_pass,
    "target_5p5_pass": target_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "next": (
        "TARGET PASS -> run all five folds immediately"
        if target_pass and causal_pass
        else (
            "ARCHITECTURE PASS -> strengthen matcher, then run five folds"
            if strict_pass
            else "FAIL -> stop this neural-emission/CRF formulation"
        )
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nRESULT")
print(json.dumps(result, indent=2))
print("\nINTERVENTIONS")
print(interventions.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
