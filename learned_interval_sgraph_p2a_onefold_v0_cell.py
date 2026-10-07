"""P2a learned dual-reference interval residuals on the frozen SGraph.

Run this cell in the same live kernel after:

1. ``prefixridge_longrange_gateway_v0_cell.py``
2. ``global_longedge_sgraph_nested_gateway_v0_cell.py``

The previous experiments established two useful facts:

* the fold-clean global edge model carries real long-range signal; and
* the multiscale-star SGraph reconstructs the path almost exactly when its
  edges are truthful.

This gateway changes only the edge evidence.  A complete-well neural encoder
sees the native horizontal GR, legal geometry/known-mask channels, calibrated
typewell GR and visible heel GR.  It predicts a bounded residual on top of a
fold-clean global edge prediction.  The graph topology and Fold-2 reliability
weights remain frozen.

Nested protocol
---------------
* Global residual targets for neural training are cross-fitted by well fold.
* Folds 3/4/5 train the neural residual; Fold 2 selects only its epoch.
* The network is reinitialised and trained on Folds 2/3/4/5 for that epoch.
* Fold 1 is evaluated once, including no-GR, shuffled-typewell and
  shuffled-heel interventions.

This is a representation gateway, not yet the final single-checkpoint model.
If it passes, the frozen global branch can be distilled into the neural edge
head before five-fold expansion.
"""

from dataclasses import dataclass
from functools import lru_cache
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
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Live-kernel contract and frozen configuration
# ---------------------------------------------------------------------------

REQUIRED_P2A = {
    "OOF_ROOT",
    "RECORD_DIR",
    "FEATURE_COLUMNS",
    "best_iteration",
    "_fit_global",
    "core",
    "tune",
    "outer_train",
    "test",
    "core_prediction_tune",
    "test_prediction",
    "tune_sg",
    "test_sg",
    "tune_wells_sg",
    "test_wells_sg",
    "reliability_sg",
    "_evaluate_graph_sg",
    "_positive_far_fraction_sg",
    "_json_default",
}
missing_p2a = sorted(REQUIRED_P2A.difference(globals()))
if missing_p2a:
    raise RuntimeError(
        "Run PrefixRidge and global-longedge SGraph in this live kernel first; "
        "missing globals: " + repr(missing_p2a)
    )


EXPERIMENT_P2A = "learned_interval_sgraph_p2a_onefold_v0"
OUT_DIR_P2A = OOF_ROOT / EXPERIMENT_P2A
MODEL_DIR_P2A = OUT_DIR_P2A / "models"
CACHE_DIR_P2A = OUT_DIR_P2A / "cache"
for directory_p2a in (OUT_DIR_P2A, MODEL_DIR_P2A, CACHE_DIR_P2A):
    directory_p2a.mkdir(parents=True, exist_ok=True)

SEED_P2A = 20260803
TUNE_FOLD_P2A = 2
TEST_FOLD_P2A = 1
CORE_FOLDS_P2A = (3, 4, 5)
OUTER_FOLDS_P2A = (2, 3, 4, 5)

GRAPH_VARIANT_P2A = "multiscale_star"
GRAPH_WEIGHT_P2A = "inverse_variance"
HORIZONTAL_POOL_P2A = 4
EMBEDDING_P2A = 32
CONDITION_EMBEDDING_P2A = 16
MAX_EPOCHS_P2A = 12
EARLY_STOPPING_P2A = 4
ACCUMULATE_WELLS_P2A = 4
LEARNING_RATE_P2A = 3.0e-4
MIN_LEARNING_RATE_P2A = 5.0e-5
WEIGHT_DECAY_P2A = 1.0e-5
GRADIENT_CLIP_P2A = 5.0
CORRECTION_CAP_FT_P2A = 24.0

TUNE_PATH_GAIN_GATE_P2A = 0.10
LONG_EDGE_MSE_GAIN_GATE_P2A = 0.12
PATH_GAIN_GATE_P2A = 0.30
FAR_GAIN_GATE_P2A = 0.50
POSITIVE_FAR_WELL_GATE_P2A = 0.60
NEAR_DEGRADATION_GATE_P2A = 0.10
NO_GR_DEGRADATION_GATE_P2A = 0.20
SHUFFLED_TYPEWELL_DEGRADATION_GATE_P2A = 0.10
SHUFFLED_HEEL_DEGRADATION_GATE_P2A = 0.05


def _seed_p2a(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_p2a(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _rmse_p2a(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


# ---------------------------------------------------------------------------
# Fold-clean global predictions used as neural residual targets
# ---------------------------------------------------------------------------

def _crossfit_global_p2a(frame, held_folds, label):
    frame = frame.reset_index(drop=True)
    cache_path = CACHE_DIR_P2A / f"{label}_global_oof.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            prediction = data["prediction"].astype(np.float64)
            cached_rows = int(data["rows"][0])
            cached_folds = tuple(data["folds"].astype(int).tolist())
        if cached_rows == len(frame) and cached_folds == tuple(held_folds):
            print(f"loading cached {label} global OOF", flush=True)
            return prediction

    prediction = np.full(len(frame), np.nan, np.float64)
    for held_fold in held_folds:
        valid_mask = frame["fold"].to_numpy(int) == int(held_fold)
        train_frame = frame.loc[~valid_mask]
        valid_frame = frame.loc[valid_mask]
        print(
            "crossfit global edge",
            {
                "phase": label,
                "held_fold": int(held_fold),
                "train_edges": int(len(train_frame)),
                "valid_edges": int(len(valid_frame)),
            },
            flush=True,
        )
        model = _fit_global(train_frame, valid=None, rounds=int(best_iteration))
        prediction[valid_mask] = model.predict(
            valid_frame[FEATURE_COLUMNS], num_iteration=int(best_iteration)
        )
        del model, train_frame, valid_frame
        gc.collect()

    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{label}: non-finite cross-fitted global predictions")
    np.savez_compressed(
        cache_path,
        prediction=prediction.astype(np.float32),
        rows=np.asarray([len(frame)], np.int64),
        folds=np.asarray(held_folds, np.int16),
    )
    return prediction


# ---------------------------------------------------------------------------
# Compact phase data and native-record access
# ---------------------------------------------------------------------------

@dataclass
class PhaseDataP2A:
    name: str
    frame: pd.DataFrame
    base_prediction: np.ndarray
    residual_target: np.ndarray
    manual: np.ndarray
    manual_mean: np.ndarray
    manual_scale: np.ndarray
    base_mean: float
    base_scale: float
    residual_scale: float
    groups: dict
    wells: list


def _phase_p2a(name, frame, base_prediction, manual_stats=None, residual_scale=None):
    frame = frame.reset_index(drop=True).copy()
    base_prediction = np.asarray(base_prediction, np.float64)
    if len(frame) != len(base_prediction):
        raise RuntimeError(
            f"{name}: frame/prediction length mismatch {len(frame)} != {len(base_prediction)}"
        )
    target = frame["target_delta_s"].to_numpy(np.float64)
    residual = target - base_prediction

    manual_raw = frame[FEATURE_COLUMNS].to_numpy(np.float32, copy=True)
    if manual_stats is None:
        manual_mean = np.nanmean(manual_raw, axis=0).astype(np.float32)
        manual_scale = np.nanstd(manual_raw, axis=0).astype(np.float32)
        manual_scale = np.where(manual_scale < 1e-4, 1.0, manual_scale).astype(np.float32)
        base_mean = float(np.mean(base_prediction))
        base_scale = float(max(np.std(base_prediction), 1.0))
    else:
        manual_mean, manual_scale, base_mean, base_scale = manual_stats
    manual = np.nan_to_num(
        (manual_raw - manual_mean[None]) / manual_scale[None],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    del manual_raw

    if residual_scale is None:
        suffix = frame["region"].eq("suffix").to_numpy()
        weight = frame["edge_weight"].to_numpy(np.float64)[suffix]
        squared = residual[suffix] ** 2
        residual_scale = float(
            np.clip(
                np.sqrt(np.sum(weight * squared) / max(np.sum(weight), 1e-12)),
                2.0,
                24.0,
            )
        )
    groups = {
        str(well): positions.to_numpy(np.int64)
        for well, positions in frame.groupby("well", sort=True).groups.items()
    }
    wells = sorted(groups)
    return PhaseDataP2A(
        name=name,
        frame=frame,
        base_prediction=base_prediction.astype(np.float32),
        residual_target=residual.astype(np.float32),
        manual=manual,
        manual_mean=np.asarray(manual_mean, np.float32),
        manual_scale=np.asarray(manual_scale, np.float32),
        base_mean=float(base_mean),
        base_scale=float(base_scale),
        residual_scale=float(residual_scale),
        groups=groups,
        wells=wells,
    )


@lru_cache(maxsize=48)
def _native_record_p2a(well):
    path = RECORD_DIR / f"{well}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {
            "horizontal": data["horizontal"].astype(np.float32),
            "condition": data["condition"].astype(np.float32),
            "typewell": data["typewell"].astype(np.float32),
            "heel": data["heel"].astype(np.float32),
            "md": data["md"].astype(np.float32),
        }


def _node_token_index_p2a(group, md, encoded_length=None):
    node_count = int(group["node_j"].max()) + 1
    node_md = np.full(node_count, np.nan, np.float64)
    node_i = group["node_i"].to_numpy(int)
    node_j = group["node_j"].to_numpy(int)
    node_md[node_i] = group["start_md"].to_numpy(np.float64)
    node_md[node_j] = group["end_md"].to_numpy(np.float64)
    if not np.isfinite(node_md).all():
        missing = np.flatnonzero(~np.isfinite(node_md))
        raise RuntimeError(f"{group['well'].iloc[0]}: missing node MD {missing[:10].tolist()}")
    raw_index = np.searchsorted(md, node_md, side="left")
    raw_index = np.clip(raw_index, 0, len(md) - 1)
    token_index = raw_index // HORIZONTAL_POOL_P2A
    if encoded_length is not None:
        token_index = np.clip(token_index, 0, encoded_length - 1)
    return token_index.astype(np.int64)


def _donor_map_p2a(wells, shift=17):
    wells = sorted(map(str, wells))
    if len(wells) < 2:
        return {well: well for well in wells}
    shift = int(shift) % len(wells)
    if shift == 0:
        shift = 1
    return {well: wells[(index + shift) % len(wells)] for index, well in enumerate(wells)}


# ---------------------------------------------------------------------------
# Complete-well dual-reference interval network
# ---------------------------------------------------------------------------

class ResidualBlock1DP2A(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = 2 * dilation
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(4, channels),
        )

    def forward(self, value):
        return F.gelu(value + self.network(value))


class LearnedIntervalEdgeP2A(nn.Module):
    def __init__(self, manual_features, residual_scale):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.gr_stem = nn.Sequential(
            nn.Conv1d(4, EMBEDDING_P2A, 9, padding=4, bias=False),
            nn.GroupNorm(4, EMBEDDING_P2A),
            nn.GELU(),
            ResidualBlock1DP2A(EMBEDDING_P2A, 1),
            ResidualBlock1DP2A(EMBEDDING_P2A, 2),
            ResidualBlock1DP2A(EMBEDDING_P2A, 4),
            ResidualBlock1DP2A(EMBEDDING_P2A, 8),
            ResidualBlock1DP2A(EMBEDDING_P2A, 16),
            ResidualBlock1DP2A(EMBEDDING_P2A, 32),
            ResidualBlock1DP2A(EMBEDDING_P2A, 64),
        )
        self.condition_stem = nn.Sequential(
            nn.Conv1d(10, CONDITION_EMBEDDING_P2A, 7, padding=3, bias=False),
            nn.GroupNorm(4, CONDITION_EMBEDDING_P2A),
            nn.GELU(),
            ResidualBlock1DP2A(CONDITION_EMBEDDING_P2A, 1),
            ResidualBlock1DP2A(CONDITION_EMBEDDING_P2A, 2),
            ResidualBlock1DP2A(CONDITION_EMBEDDING_P2A, 4),
        )
        self.log_similarity_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        node_features = EMBEDDING_P2A + CONDITION_EMBEDDING_P2A + 10
        edge_features = 7 * node_features + int(manual_features) + 1
        self.edge_head = nn.Sequential(
            nn.Linear(edge_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)

    def _reference_summary(self, query, reference_embedding, reference_valid):
        query = F.normalize(query.float(), dim=1, eps=1e-5)
        reference_embedding = F.normalize(reference_embedding.float(), dim=1, eps=1e-5)
        similarity = torch.einsum("bct,bcr->btr", query, reference_embedding)
        valid = reference_valid > 0.5
        valid_count = valid.sum(dim=-1, keepdim=True)
        safe_valid = torch.where(valid_count >= 2, valid, torch.ones_like(valid))
        similarity = similarity.masked_fill(~safe_valid[:, None, :], -1.0e4)
        scale = torch.exp(self.log_similarity_scale).clamp(2.0, 30.0)
        probability = F.softmax(scale * similarity, dim=-1)

        states = torch.linspace(
            -1.5,
            1.5,
            reference_embedding.shape[-1],
            device=query.device,
            dtype=probability.dtype,
        )[None, None, :]
        expected = (probability * states).sum(dim=-1)
        variance = (probability * (states - expected[..., None]) ** 2).sum(dim=-1)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
        normaliser = torch.log(safe_valid.sum(dim=-1).clamp_min(2).float())
        entropy = entropy / normaliser
        top = torch.topk(probability, k=2, dim=-1).values
        return torch.stack(
            [expected, torch.sqrt(variance.clamp_min(1e-8)), entropy, top[..., 0], top[..., 0] - top[..., 1]],
            dim=1,
        )

    def forward(
        self,
        horizontal,
        condition,
        typewell,
        heel,
        node_token_index,
        edge_i,
        edge_j,
        manual,
        base_standard,
    ):
        horizontal_embedding_native = self.gr_stem(horizontal)
        horizontal_embedding = F.avg_pool1d(
            horizontal_embedding_native,
            kernel_size=HORIZONTAL_POOL_P2A,
            stride=HORIZONTAL_POOL_P2A,
            ceil_mode=True,
        )
        condition_embedding_native = self.condition_stem(condition)
        condition_embedding = F.avg_pool1d(
            condition_embedding_native,
            kernel_size=HORIZONTAL_POOL_P2A,
            stride=HORIZONTAL_POOL_P2A,
            ceil_mode=True,
        )
        columns = min(horizontal_embedding.shape[-1], condition_embedding.shape[-1])
        horizontal_embedding = horizontal_embedding[..., :columns]
        condition_embedding = condition_embedding[..., :columns]

        node_token_index = node_token_index.clamp(0, columns - 1)
        horizontal_node = horizontal_embedding[..., node_token_index]
        condition_node = condition_embedding[..., node_token_index]

        type_embedding = self.gr_stem(typewell)
        heel_embedding = self.gr_stem(heel)
        type_summary = self._reference_summary(
            horizontal_node, type_embedding, typewell[:, 1]
        )
        heel_summary = self._reference_summary(
            horizontal_node, heel_embedding, heel[:, 1]
        )
        node_sequence = torch.cat(
            [horizontal_node.float(), condition_node.float(), type_summary, heel_summary],
            dim=1,
        )
        node = node_sequence[0].transpose(0, 1)
        cumulative = torch.cat(
            [torch.zeros(1, node.shape[1], device=node.device), node.cumsum(dim=0)],
            dim=0,
        )
        count = (edge_j - edge_i + 1).clamp_min(1).float()[:, None]
        interval_mean = (cumulative[edge_j + 1] - cumulative[edge_i]) / count
        start = node[edge_i]
        end = node[edge_j]
        span = edge_j - edge_i
        quarter = node[edge_i + torch.round(span.float() * 0.25).long()]
        midpoint = node[edge_i + torch.round(span.float() * 0.50).long()]
        three_quarter = node[edge_i + torch.round(span.float() * 0.75).long()]
        edge_context = torch.cat(
            [
                start,
                end,
                end - start,
                interval_mean,
                quarter,
                midpoint,
                three_quarter,
                manual,
                base_standard[:, None],
            ],
            dim=1,
        )
        raw = self.edge_head(edge_context).squeeze(1)
        return CORRECTION_CAP_FT_P2A * torch.tanh(raw)


# ---------------------------------------------------------------------------
# Tensor preparation, fitting and prediction
# ---------------------------------------------------------------------------

def _model_inputs_p2a(phase, well, device, mode="full", donor=None):
    positions = phase.groups[str(well)]
    group = phase.frame.iloc[positions]
    record = _native_record_p2a(str(well))
    horizontal = record["horizontal"].copy()
    condition = record["condition"].copy()
    typewell = record["typewell"].copy()
    heel = record["heel"].copy()

    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    elif mode == "shuffled_typewell":
        if donor is None:
            raise ValueError("shuffled_typewell requires a donor")
        typewell = _native_record_p2a(str(donor))["typewell"].copy()
    elif mode == "shuffled_heel":
        if donor is None:
            raise ValueError("shuffled_heel requires a donor")
        heel = _native_record_p2a(str(donor))["heel"].copy()
    elif mode != "full":
        raise ValueError(mode)

    node_token = _node_token_index_p2a(group, record["md"])
    edge_i = group["node_i"].to_numpy(np.int64)
    edge_j = group["node_j"].to_numpy(np.int64)
    base_standard = (
        phase.base_prediction[positions].astype(np.float32) - phase.base_mean
    ) / phase.base_scale

    tensors = {
        "horizontal": torch.from_numpy(horizontal)[None].to(device),
        "condition": torch.from_numpy(condition)[None].to(device),
        "typewell": torch.from_numpy(typewell)[None].to(device),
        "heel": torch.from_numpy(heel)[None].to(device),
        "node_token": torch.from_numpy(node_token).to(device),
        "edge_i": torch.from_numpy(edge_i).to(device),
        "edge_j": torch.from_numpy(edge_j).to(device),
        "manual": torch.from_numpy(phase.manual[positions]).to(device),
        "base_standard": torch.from_numpy(base_standard).to(device),
        "target": torch.from_numpy(phase.residual_target[positions]).to(device),
    }

    suffix = group["region"].eq("suffix").to_numpy()
    weight = group["edge_weight"].to_numpy(np.float32)
    end_distance = group["end_distance"].to_numpy(np.float32)
    weight *= np.where(end_distance > 1000.0, 1.25, 1.0).astype(np.float32)
    weight *= np.where(end_distance > 2000.0, 1.20, 1.0).astype(np.float32)
    weight = weight / max(float(np.mean(weight[suffix])), 1e-8)
    tensors["suffix"] = torch.from_numpy(suffix).to(device)
    tensors["weight"] = torch.from_numpy(weight).to(device)
    return tensors, positions


def _forward_inputs_p2a(model, tensors):
    return model(
        tensors["horizontal"],
        tensors["condition"],
        tensors["typewell"],
        tensors["heel"],
        tensors["node_token"],
        tensors["edge_i"],
        tensors["edge_j"],
        tensors["manual"],
        tensors["base_standard"],
    )


def _train_epoch_p2a(model, phase, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    order = phase.wells.copy()
    random.Random(SEED_P2A + 1009 * epoch + len(order)).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    loss_sum = 0.0
    wells_seen = 0
    pending = 0
    for index, well in enumerate(order, 1):
        tensors, _ = _model_inputs_p2a(phase, well, device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            correction = _forward_inputs_p2a(model, tensors)
            suffix = tensors["suffix"]
            predicted = correction[suffix] / phase.residual_scale
            target = tensors["target"][suffix] / phase.residual_scale
            element = F.smooth_l1_loss(predicted, target, beta=0.5, reduction="none")
            weight = tensors["weight"][suffix]
            loss = (element * weight).sum() / weight.sum().clamp_min(1e-8)
            loss = loss + 1e-4 * torch.mean((correction[suffix] / CORRECTION_CAP_FT_P2A) ** 2)
            scaled_loss = loss / ACCUMULATE_WELLS_P2A
        scaler.scale(scaled_loss).backward()
        pending += 1
        loss_sum += float(loss.detach().cpu())
        wells_seen += 1

        if pending == ACCUMULATE_WELLS_P2A or index == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_P2A)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        if index % 100 == 0 or index == len(order):
            print(
                f"P2A epoch {epoch}/{total_epochs} wells {index}/{len(order)}",
                flush=True,
            )
        del tensors, correction, suffix, predicted, target, element, weight, loss, scaled_loss

    return float(loss_sum / max(wells_seen, 1))


@torch.no_grad()
def _predict_phase_p2a(model, phase, device, mode="full", donor_map=None, desc=None):
    model.eval()
    correction = np.zeros(len(phase.frame), np.float32)
    amp_enabled = device.type == "cuda"
    iterator = tqdm(phase.wells, desc=desc or f"predict {phase.name} {mode}")
    for well in iterator:
        donor = None if donor_map is None else donor_map[str(well)]
        tensors, positions = _model_inputs_p2a(phase, well, device, mode=mode, donor=donor)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            value = _forward_inputs_p2a(model, tensors)
        correction[positions] = value.float().cpu().numpy()
        del tensors, value
    return correction


def _edge_report_p2a(frame, base_prediction, candidate_prediction):
    length = frame["end_md"].to_numpy(np.float64) - frame["start_md"].to_numpy(np.float64)
    suffix = frame["region"].eq("suffix").to_numpy()
    subsets = {
        "suffix_all": suffix,
        "length_ge1000": suffix & (length >= 1000.0),
        "length_ge2000": suffix & (length >= 2000.0),
        "end_gt2000": suffix & (frame["end_distance"].to_numpy(np.float64) > 2000.0),
    }
    truth = frame["target_delta_s"].to_numpy(np.float64)
    base_prediction = np.asarray(base_prediction, np.float64)
    candidate_prediction = np.asarray(candidate_prediction, np.float64)
    edge_weight = frame["edge_weight"].to_numpy(np.float64)
    rows = []
    for name, mask in subsets.items():
        weight = edge_weight[mask]
        base_mse = float(
            np.sum(weight * (truth[mask] - base_prediction[mask]) ** 2)
            / max(np.sum(weight), 1e-12)
        )
        candidate_mse = float(
            np.sum(weight * (truth[mask] - candidate_prediction[mask]) ** 2)
            / max(np.sum(weight), 1e-12)
        )
        rows.append(
            {
                "subset": name,
                "edges": int(mask.sum()),
                "base_rmse": math.sqrt(base_mse),
                "candidate_rmse": math.sqrt(candidate_mse),
                "relative_mse_gain": (base_mse - candidate_mse) / max(base_mse, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _graph_p2a(phase, prediction):
    return _evaluate_graph_sg(
        phase.frame,
        np.asarray(prediction, np.float64),
        phase.wells,
        GRAPH_VARIANT_P2A,
        GRAPH_WEIGHT_P2A,
        reliability_sg,
    )


def _new_model_p2a(phase, device, seed):
    _seed_p2a(seed)
    model = LearnedIntervalEdgeP2A(len(FEATURE_COLUMNS), phase.residual_scale).to(device)
    return model


def _optimizer_p2a(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_P2A,
        weight_decay=WEIGHT_DECAY_P2A,
    )


def _set_lr_p2a(optimizer, epoch, total_epochs):
    fraction = (epoch - 1) / max(total_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    learning_rate = MIN_LEARNING_RATE_P2A + (
        LEARNING_RATE_P2A - MIN_LEARNING_RATE_P2A
    ) * cosine
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return float(learning_rate)


# ---------------------------------------------------------------------------
# Execute nested P2a gateway
# ---------------------------------------------------------------------------

started_p2a = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("P2a requires CUDA")
device_p2a = torch.device("cuda")
_seed_p2a(SEED_P2A)

core_p2a_frame = core.reset_index(drop=True).copy()
tune_p2a_frame = tune_sg.reset_index(drop=True).copy()
outer_p2a_frame = outer_train.reset_index(drop=True).copy()
test_p2a_frame = test_sg.reset_index(drop=True).copy()

print("building fold-clean residual targets", flush=True)
core_oof_p2a = _crossfit_global_p2a(core_p2a_frame, CORE_FOLDS_P2A, "core345")
outer_oof_p2a = _crossfit_global_p2a(outer_p2a_frame, OUTER_FOLDS_P2A, "outer2345")

core_phase_p2a = _phase_p2a("core345", core_p2a_frame, core_oof_p2a)
core_stats_p2a = (
    core_phase_p2a.manual_mean,
    core_phase_p2a.manual_scale,
    core_phase_p2a.base_mean,
    core_phase_p2a.base_scale,
)
tune_phase_p2a = _phase_p2a(
    "fold2",
    tune_p2a_frame,
    np.asarray(core_prediction_tune, np.float64),
    manual_stats=core_stats_p2a,
    residual_scale=core_phase_p2a.residual_scale,
)

base_tune_metrics_p2a, base_tune_bywell_p2a, _, _ = _graph_p2a(
    tune_phase_p2a, tune_phase_p2a.base_prediction
)
print("P2A FOLD2 BASE", base_tune_metrics_p2a, flush=True)

model_p2a = _new_model_p2a(core_phase_p2a, device_p2a, SEED_P2A)
parameter_count_p2a = int(sum(parameter.numel() for parameter in model_p2a.parameters()))
optimizer_p2a = _optimizer_p2a(model_p2a)
scaler_p2a = torch.amp.GradScaler("cuda", enabled=True)

history_rows_p2a = [
    {
        "epoch": 0,
        "train_loss": np.nan,
        "learning_rate": 0.0,
        "path_rmse": base_tune_metrics_p2a["rmse"],
        "path_gain_vs_base": 0.0,
        "far_rmse": base_tune_metrics_p2a["far_gt2000_rmse"],
        "near_rmse": base_tune_metrics_p2a["near_0_500_rmse"],
        "long_edge_mse_gain": 0.0,
    }
]
best_epoch_p2a = 0
best_path_rmse_p2a = float(base_tune_metrics_p2a["rmse"])
best_state_p2a = copy.deepcopy(model_p2a.state_dict())
epochs_without_improvement_p2a = 0

for epoch_p2a in range(1, MAX_EPOCHS_P2A + 1):
    learning_rate_p2a = _set_lr_p2a(optimizer_p2a, epoch_p2a, MAX_EPOCHS_P2A)
    train_loss_p2a = _train_epoch_p2a(
        model_p2a,
        core_phase_p2a,
        optimizer_p2a,
        scaler_p2a,
        device_p2a,
        epoch_p2a,
        MAX_EPOCHS_P2A,
    )
    tune_correction_p2a = _predict_phase_p2a(
        model_p2a,
        tune_phase_p2a,
        device_p2a,
        desc=f"Fold2 P2A epoch {epoch_p2a}",
    )
    tune_candidate_p2a = tune_phase_p2a.base_prediction + tune_correction_p2a
    tune_metrics_p2a, _, _, _ = _graph_p2a(tune_phase_p2a, tune_candidate_p2a)
    tune_edge_report_p2a = _edge_report_p2a(
        tune_phase_p2a.frame,
        tune_phase_p2a.base_prediction,
        tune_candidate_p2a,
    )
    long_gain_p2a = float(
        tune_edge_report_p2a.loc[
            tune_edge_report_p2a["subset"].eq("length_ge1000"),
            "relative_mse_gain",
        ].iloc[0]
    )
    row_p2a = {
        "epoch": epoch_p2a,
        "train_loss": train_loss_p2a,
        "learning_rate": learning_rate_p2a,
        "path_rmse": tune_metrics_p2a["rmse"],
        "path_gain_vs_base": base_tune_metrics_p2a["rmse"] - tune_metrics_p2a["rmse"],
        "far_rmse": tune_metrics_p2a["far_gt2000_rmse"],
        "near_rmse": tune_metrics_p2a["near_0_500_rmse"],
        "long_edge_mse_gain": long_gain_p2a,
    }
    history_rows_p2a.append(row_p2a)
    print("P2A FOLD2 EPOCH", row_p2a, flush=True)

    if tune_metrics_p2a["rmse"] < best_path_rmse_p2a - 1e-4:
        best_path_rmse_p2a = float(tune_metrics_p2a["rmse"])
        best_epoch_p2a = int(epoch_p2a)
        best_state_p2a = copy.deepcopy(model_p2a.state_dict())
        epochs_without_improvement_p2a = 0
    else:
        epochs_without_improvement_p2a += 1
    if epoch_p2a >= 4 and epochs_without_improvement_p2a >= EARLY_STOPPING_P2A:
        print("P2A Fold2 early stopping", flush=True)
        break

history_p2a = pd.DataFrame(history_rows_p2a)
history_p2a.to_csv(OUT_DIR_P2A / "fold2_epoch_selection.csv", index=False)
selected_tune_gain_p2a = float(base_tune_metrics_p2a["rmse"] - best_path_rmse_p2a)

torch.save(
    {
        "model": best_state_p2a,
        "epoch": best_epoch_p2a,
        "parameters": parameter_count_p2a,
        "manual_mean": core_phase_p2a.manual_mean,
        "manual_scale": core_phase_p2a.manual_scale,
        "base_mean": core_phase_p2a.base_mean,
        "base_scale": core_phase_p2a.base_scale,
        "residual_scale": core_phase_p2a.residual_scale,
    },
    MODEL_DIR_P2A / "fold2_selected.pt",
)

if best_epoch_p2a == 0 or selected_tune_gain_p2a < TUNE_PATH_GAIN_GATE_P2A:
    verdict_p2a = {
        "experiment": EXPERIMENT_P2A,
        "stage": "fold2_epoch_gateway",
        "parameters": parameter_count_p2a,
        "fold2_base": base_tune_metrics_p2a,
        "selected_epoch": best_epoch_p2a,
        "selected_rmse": best_path_rmse_p2a,
        "selected_gain": selected_tune_gain_p2a,
        "gate_gain_ge": TUNE_PATH_GAIN_GATE_P2A,
        "strict_pass": False,
        "next": "FAIL -> learned interval residual does not transfer; stop before Fold1",
        "elapsed_minutes": (time.time() - started_p2a) / 60.0,
    }
    _json_p2a(OUT_DIR_P2A / "verdict.json", verdict_p2a)
    print("P2A FINAL VERDICT", json.dumps(verdict_p2a, indent=2), flush=True)
    raise RuntimeError(json.dumps(verdict_p2a, indent=2))


# Reinitialise and train the untouched-Fold1 model for the frozen epoch count.
outer_phase_p2a = _phase_p2a("outer2345", outer_p2a_frame, outer_oof_p2a)
outer_stats_p2a = (
    outer_phase_p2a.manual_mean,
    outer_phase_p2a.manual_scale,
    outer_phase_p2a.base_mean,
    outer_phase_p2a.base_scale,
)
test_phase_p2a = _phase_p2a(
    "fold1",
    test_p2a_frame,
    np.asarray(test_prediction, np.float64),
    manual_stats=outer_stats_p2a,
    residual_scale=outer_phase_p2a.residual_scale,
)

del model_p2a, optimizer_p2a, scaler_p2a, best_state_p2a
gc.collect()
torch.cuda.empty_cache()

final_model_p2a = _new_model_p2a(outer_phase_p2a, device_p2a, SEED_P2A + 1)
final_optimizer_p2a = _optimizer_p2a(final_model_p2a)
final_scaler_p2a = torch.amp.GradScaler("cuda", enabled=True)
final_train_rows_p2a = []
for epoch_p2a in range(1, best_epoch_p2a + 1):
    learning_rate_p2a = _set_lr_p2a(final_optimizer_p2a, epoch_p2a, best_epoch_p2a)
    train_loss_p2a = _train_epoch_p2a(
        final_model_p2a,
        outer_phase_p2a,
        final_optimizer_p2a,
        final_scaler_p2a,
        device_p2a,
        epoch_p2a,
        best_epoch_p2a,
    )
    final_train_rows_p2a.append(
        {
            "epoch": epoch_p2a,
            "train_loss": train_loss_p2a,
            "learning_rate": learning_rate_p2a,
        }
    )
pd.DataFrame(final_train_rows_p2a).to_csv(
    OUT_DIR_P2A / "outer_training_history.csv", index=False
)

torch.save(
    {
        "model": final_model_p2a.state_dict(),
        "epoch": best_epoch_p2a,
        "parameters": parameter_count_p2a,
        "manual_mean": outer_phase_p2a.manual_mean,
        "manual_scale": outer_phase_p2a.manual_scale,
        "base_mean": outer_phase_p2a.base_mean,
        "base_scale": outer_phase_p2a.base_scale,
        "residual_scale": outer_phase_p2a.residual_scale,
        "correction_cap_ft": CORRECTION_CAP_FT_P2A,
    },
    MODEL_DIR_P2A / "fold1_final.pt",
)

base_fold1_metrics_p2a, base_fold1_bywell_p2a, _, base_payload_p2a = _graph_p2a(
    test_phase_p2a, test_phase_p2a.base_prediction
)
donor_map_p2a = _donor_map_p2a(test_phase_p2a.wells)
mode_rows_p2a = []
mode_outputs_p2a = {}
mode_bywell_p2a = {}
for mode_p2a in ("full", "no_gr", "shuffled_typewell", "shuffled_heel"):
    correction_p2a = _predict_phase_p2a(
        final_model_p2a,
        test_phase_p2a,
        device_p2a,
        mode=mode_p2a,
        donor_map=donor_map_p2a if mode_p2a.startswith("shuffled") else None,
        desc=f"Fold1 P2A {mode_p2a}",
    )
    prediction_p2a = test_phase_p2a.base_prediction + correction_p2a
    metrics_p2a, bywell_p2a, diagnostics_p2a, payload_p2a = _graph_p2a(
        test_phase_p2a, prediction_p2a
    )
    metrics_p2a["mode"] = mode_p2a
    metrics_p2a["mean_abs_edge_correction"] = float(np.mean(np.abs(correction_p2a)))
    metrics_p2a["p95_abs_edge_correction"] = float(np.quantile(np.abs(correction_p2a), 0.95))
    metrics_p2a["max_abs_edge_correction"] = float(np.max(np.abs(correction_p2a)))
    mode_rows_p2a.append(metrics_p2a)
    mode_outputs_p2a[mode_p2a] = {
        "correction": correction_p2a,
        "prediction": prediction_p2a,
        "payload": payload_p2a,
    }
    mode_bywell_p2a[mode_p2a] = bywell_p2a
    diagnostics_p2a.to_csv(
        OUT_DIR_P2A / f"fold1_graph_diagnostics_{mode_p2a}.csv", index=False
    )
    print("P2A FOLD1 MODE", metrics_p2a, flush=True)

mode_report_p2a = pd.DataFrame(mode_rows_p2a).sort_values("rmse")
mode_report_p2a.to_csv(OUT_DIR_P2A / "fold1_interventions.csv", index=False)
full_metrics_p2a = next(row for row in mode_rows_p2a if row["mode"] == "full")
full_bywell_p2a = mode_bywell_p2a["full"]
full_prediction_p2a = mode_outputs_p2a["full"]["prediction"]

edge_report_p2a = _edge_report_p2a(
    test_phase_p2a.frame,
    test_phase_p2a.base_prediction,
    full_prediction_p2a,
)
edge_report_p2a.to_csv(OUT_DIR_P2A / "fold1_edge_report.csv", index=False)
base_fold1_bywell_p2a.to_csv(OUT_DIR_P2A / "fold1_base_bywell.csv", index=False)
full_bywell_p2a.to_csv(OUT_DIR_P2A / "fold1_full_bywell.csv", index=False)

np.savez_compressed(
    OUT_DIR_P2A / "fold1_predictions.npz",
    base_edge=test_phase_p2a.base_prediction.astype(np.float32),
    full_edge=full_prediction_p2a.astype(np.float32),
    full_correction=mode_outputs_p2a["full"]["correction"].astype(np.float32),
    row_truth=mode_outputs_p2a["full"]["payload"]["truth"],
    row_base=base_payload_p2a["prediction"],
    row_full=mode_outputs_p2a["full"]["payload"]["prediction"],
    row_distance=base_payload_p2a["distance"],
    row_well=base_payload_p2a["well"],
)

long_edge_gain_p2a = float(
    edge_report_p2a.loc[
        edge_report_p2a["subset"].eq("length_ge1000"), "relative_mse_gain"
    ].iloc[0]
)
path_gain_p2a = float(base_fold1_metrics_p2a["rmse"] - full_metrics_p2a["rmse"])
far_gain_p2a = float(
    base_fold1_metrics_p2a["far_gt2000_rmse"] - full_metrics_p2a["far_gt2000_rmse"]
)
near_degradation_p2a = float(
    full_metrics_p2a["near_0_500_rmse"] - base_fold1_metrics_p2a["near_0_500_rmse"]
)
positive_far_fraction_p2a = _positive_far_fraction_sg(
    base_fold1_bywell_p2a, full_bywell_p2a
)
metrics_by_mode_p2a = {row["mode"]: row for row in mode_rows_p2a}
no_gr_degradation_p2a = float(
    metrics_by_mode_p2a["no_gr"]["rmse"] - full_metrics_p2a["rmse"]
)
shuffled_typewell_degradation_p2a = float(
    metrics_by_mode_p2a["shuffled_typewell"]["rmse"] - full_metrics_p2a["rmse"]
)
shuffled_heel_degradation_p2a = float(
    metrics_by_mode_p2a["shuffled_heel"]["rmse"] - full_metrics_p2a["rmse"]
)

gates_p2a = {
    "long_edge_mse_gain": long_edge_gain_p2a >= LONG_EDGE_MSE_GAIN_GATE_P2A,
    "path_gain": path_gain_p2a >= PATH_GAIN_GATE_P2A,
    "far_gain": far_gain_p2a >= FAR_GAIN_GATE_P2A,
    "positive_far_well_fraction": positive_far_fraction_p2a >= POSITIVE_FAR_WELL_GATE_P2A,
    "near_degradation": near_degradation_p2a <= NEAR_DEGRADATION_GATE_P2A,
    "no_gr_control": no_gr_degradation_p2a >= NO_GR_DEGRADATION_GATE_P2A,
    "shuffled_typewell_control": shuffled_typewell_degradation_p2a
    >= SHUFFLED_TYPEWELL_DEGRADATION_GATE_P2A,
    "shuffled_heel_control": shuffled_heel_degradation_p2a
    >= SHUFFLED_HEEL_DEGRADATION_GATE_P2A,
}
strict_pass_p2a = bool(all(gates_p2a.values()))

verdict_p2a = {
    "experiment": EXPERIMENT_P2A,
    "stage": "nested_fold1_learned_interval_residual_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "global_edge_branch": "cross-fitted for neural targets; Fold1 base trained on Folds 2-5",
    "graph": {
        "variant": GRAPH_VARIANT_P2A,
        "weight_scheme": GRAPH_WEIGHT_P2A,
        "weights": "frozen from Fold2 global-edge residual variance",
        "learned_precision": False,
    },
    "model": {
        "parameters": parameter_count_p2a,
        "horizontal_pool": HORIZONTAL_POOL_P2A,
        "embedding": EMBEDDING_P2A,
        "correction_cap_ft": CORRECTION_CAP_FT_P2A,
    },
    "protocol": {
        "epoch_train_folds": list(CORE_FOLDS_P2A),
        "epoch_validation_fold": TUNE_FOLD_P2A,
        "selected_epoch": best_epoch_p2a,
        "fold2_base_rmse": base_tune_metrics_p2a["rmse"],
        "fold2_selected_rmse": best_path_rmse_p2a,
        "fold2_gain": selected_tune_gain_p2a,
        "final_train_folds": list(OUTER_FOLDS_P2A),
        "untouched_test_fold": TEST_FOLD_P2A,
    },
    "fold1": {
        "base": base_fold1_metrics_p2a,
        "full": full_metrics_p2a,
        "path_gain": path_gain_p2a,
        "far_gain": far_gain_p2a,
        "near_degradation": near_degradation_p2a,
        "positive_far_well_fraction": positive_far_fraction_p2a,
        "long_edge_relative_mse_gain": long_edge_gain_p2a,
    },
    "controls": {
        "no_gr_degradation": no_gr_degradation_p2a,
        "shuffled_typewell_degradation": shuffled_typewell_degradation_p2a,
        "shuffled_heel_degradation": shuffled_heel_degradation_p2a,
    },
    "gate_thresholds": {
        "long_edge_mse_gain_ge": LONG_EDGE_MSE_GAIN_GATE_P2A,
        "path_gain_ge_ft": PATH_GAIN_GATE_P2A,
        "far_gain_ge_ft": FAR_GAIN_GATE_P2A,
        "positive_far_well_fraction_ge": POSITIVE_FAR_WELL_GATE_P2A,
        "near_degradation_le_ft": NEAR_DEGRADATION_GATE_P2A,
        "no_gr_degradation_ge_ft": NO_GR_DEGRADATION_GATE_P2A,
        "shuffled_typewell_degradation_ge_ft": SHUFFLED_TYPEWELL_DEGRADATION_GATE_P2A,
        "shuffled_heel_degradation_ge_ft": SHUFFLED_HEEL_DEGRADATION_GATE_P2A,
    },
    "gates": gates_p2a,
    "strict_pass": strict_pass_p2a,
    "next": (
        "PASS -> add differentiable path loss, then distil the global branch into one checkpoint"
        if strict_pass_p2a
        else "FAIL -> close learned long-edge residual before end-to-end SGraph training"
    ),
    "elapsed_minutes": (time.time() - started_p2a) / 60.0,
}
_json_p2a(OUT_DIR_P2A / "verdict.json", verdict_p2a)

print("\nP2A FOLD1 EDGE REPORT")
print(edge_report_p2a.to_string(index=False), flush=True)
print("\nP2A FOLD1 INTERVENTIONS")
print(mode_report_p2a.to_string(index=False), flush=True)
print("\nP2A FINAL VERDICT")
print(json.dumps(verdict_p2a, indent=2), flush=True)
print("saved:", OUT_DIR_P2A, flush=True)
