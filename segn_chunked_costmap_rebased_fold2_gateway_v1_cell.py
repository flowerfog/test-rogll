"""Fold-2 gateway for an overlap-rebased autoregressive TVT cost map.

This experiment isolates the most testable claim in the SEGN working note:
local cost-map evidence can be useful even when complete-well decoding fails,
and much of the deployed loss may come from recursive anchor drift.

Protocol
--------
* Fold 4/5 train a teacher-anchor model.
* That model rolls out on Fold 3 to create a deployment-shaped anchor-error
  library.  No Fold-2 labels enter this process.
* Fold 3/4/5 train the final model with matched joint anchor/delta corruption.
  Fold 2 selects the epoch and is evaluated once in full.
* Fold 1 is never loaded.

Every model input is legal at deployment: native horizontal GR/geometry, the
visible prefix embedded in the cached record, the assigned calibrated
typewell, and previous model predictions.  No neighbour, spatial strong,
CP12, Gold, PF path, or validation truth is used during rollout.

Prerequisite
------------
Run ``build_native_dualref_all_records_v0_cell.py`` once so that all 773
native-resolution records exist on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "segn_chunked_costmap_rebased_fold2_gateway_v1"
ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() else DATA_ROOT / "oof_exports"
RECORD_DIR_CANDIDATES = (
    OOF_ROOT / "native_resolution_dualref_capacity_v0" / "records",
    OOF_ROOT / "native_resolution_dualref_onefold_v0" / "records",
)
RECORD_DIRS = tuple(path for path in RECORD_DIR_CANDIDATES if path.exists())
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260721
REQUIRE_CUDA = True

NOISE_TRAIN_FOLDS = (4, 5)
NOISE_ROLLOUT_FOLD = 3
FINAL_TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2

NODE_SPACING_MD_FT = 32.0
CHUNK_TARGET_NODES = 16
CHUNK_STRIDE_NODES = 8
CHUNK_CONTEXT_MD_FT = 256.0
STATE_HALF_WIDTH_FT = 96.0
STATE_OFFSETS = np.arange(-96.0, 97.0, 1.0, dtype=np.float32)
REFERENCE_HALF_WIDTH_FT = 192.0
TARGET_SIGMA_FT = 1.5

EMBEDDING = 32
HEADS = 4
HEAD_DIM = 8
BASE_CHANNELS = 24
AXIAL_DILATIONS = (1, 2, 4, 8)

NOISE_STAGE_EPOCHS = 6
FINAL_EPOCHS = 14
SAMPLES_PER_WELL = 3
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 3.0e-4
MIN_LEARNING_RATE = 3.0e-5
WEIGHT_DECAY = 1.0e-5
GRADIENT_CLIP = 5.0
REPORT_EVERY = 2
TUNE_WELLS = 48
PATIENCE_REPORTS = 3

LOSS_CE = 1.0
LOSS_OFFSET = 0.50
LOSS_SLOPE = 0.08

EVIDENCE_GATE_RMSE = 7.0
ROLLOUT_GATE_RMSE = 8.0
ROLLOUT_GAP_GATE = 2.0
FAR_GATE_RMSE = 10.0
CONTROL_DEGRADATION_GATE = 1.0
TARGET_RMSE = 6.7


# ---------------------------------------------------------------------------
# Utilities and records
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _rmse(truth, prediction) -> float:
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _mae(truth, prediction) -> float:
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.mean(np.abs(prediction - truth)))


@dataclass
class Record:
    well: str
    horizontal: np.ndarray
    condition: np.ndarray
    typewell: np.ndarray
    md: np.ndarray
    eval_index: np.ndarray
    truth: np.ndarray
    anchor_tvt: float
    anchor_index: int
    eval_rows: int
    node_rows: np.ndarray
    chunk_starts: np.ndarray


def _load_folds():
    fold_path = next((path for path in FOLD_CANDIDATES if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError("Cannot find fold-clean well_metadata.csv")
    frame = pd.read_csv(fold_path, dtype=str)
    well_col = next((column for column in ("well", "well_id", "WELL") if column in frame), None)
    fold_col = next((column for column in ("fold", "outer_fold", "Fold") if column in frame), None)
    if well_col is None or fold_col is None:
        raise RuntimeError(f"Unexpected fold columns: {frame.columns.tolist()}")
    frame = frame[[well_col, fold_col]].rename(columns={well_col: "well", fold_col: "fold"})
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Unexpected folds: {sorted(frame['fold'].unique().tolist())}")
    return frame.sort_values(["fold", "well"]).reset_index(drop=True), fold_path


def _record_path(well):
    return next(
        (directory / f"{well}.npz" for directory in RECORD_DIRS if (directory / f"{well}.npz").exists()),
        None,
    )


def _node_rows(md, anchor_index, eval_index):
    anchor_md = float(md[anchor_index])
    toe_md = float(md[eval_index[-1]])
    targets = np.arange(anchor_md + NODE_SPACING_MD_FT, toe_md, NODE_SPACING_MD_FT)
    rows = [int(anchor_index)]
    for target in targets:
        position = int(np.searchsorted(md, target))
        candidates = np.asarray([max(0, position - 1), min(len(md) - 1, position)], np.int32)
        row = int(candidates[np.argmin(np.abs(md[candidates] - target))])
        if row > rows[-1] and row <= int(eval_index[-1]):
            rows.append(row)
    if int(eval_index[-1]) > rows[-1]:
        rows.append(int(eval_index[-1]))
    return np.asarray(rows, np.int32)


def _eligible_chunk_starts(record):
    # Chunk placement is a deterministic function of MD nodes.  It must not
    # depend on whether validation truth happens to fit inside the state axis.
    # Predict 512 ft but advance only 256 ft.  The next anchor is therefore
    # taken from the better-supported middle of the previous prediction.
    return np.arange(0, len(record.node_rows) - 1, CHUNK_STRIDE_NODES, dtype=np.int32)


def _load_record(well) -> Record:
    path = _record_path(well)
    if path is None:
        raise FileNotFoundError(
            f"Missing native record for {well}. Run build_native_dualref_all_records_v0_cell.py first."
        )
    with np.load(path, allow_pickle=False) as data:
        md = data["md"].astype(np.float32)
        eval_index = data["eval_index"].astype(np.int32)
        anchor_index = int(data["anchor_index"][0])
        record = Record(
            well=str(well),
            horizontal=data["horizontal"].astype(np.float32),
            condition=data["condition"].astype(np.float32),
            typewell=data["typewell"].astype(np.float32),
            md=md,
            eval_index=eval_index,
            truth=data["truth"].astype(np.float32),
            anchor_tvt=float(data["anchor_tvt"][0]),
            anchor_index=anchor_index,
            eval_rows=int(data["eval_rows"][0]),
            node_rows=_node_rows(md, anchor_index, eval_index),
            chunk_starts=np.empty(0, np.int32),
        )
    record.chunk_starts = _eligible_chunk_starts(record)
    # Very short official suffixes (notably fba7683c) legitimately form a
    # single partial chunk.  Keep them in the fold instead of changing the
    # metric population.
    if len(record.chunk_starts) < 1:
        raise RuntimeError(f"{well}: no deployable chunks")
    return record


def _load_records(wells, label):
    records = []
    failures = []
    started = time.time()
    for number, well in enumerate(wells, 1):
        try:
            records.append(_load_record(well))
        except Exception as error:
            failures.append({"well": well, "reason": repr(error)})
        if number % 50 == 0 or number == len(wells):
            print(f"load {label} records {number}/{len(wells)}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_record_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} {label} records failed: {failures[:5]}")
    print(f"loaded {label}: {len(records)} wells in {(time.time() - started) / 60.0:.2f}m", flush=True)
    return records


def _target_axis_coverage(records):
    covered = 0
    total = 0
    for record in records:
        for start in record.chunk_starts:
            start = int(start)
            end = min(start + CHUNK_TARGET_NODES, len(record.node_rows) - 1)
            rows = record.node_rows[start + 1 : end + 1]
            target = record.truth[rows] - record.truth[record.node_rows[start]]
            covered += int(np.sum(np.abs(target) <= STATE_HALF_WIDTH_FT - 0.5))
            total += len(target)
    return float(covered / max(total, 1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Residual1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )

    def forward(self, values):
        return values + self.network(values)


class SharedGREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(4, EMBEDDING, 9, padding=4),
            nn.GroupNorm(4, EMBEDDING),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            Residual1D(EMBEDDING, 1),
            Residual1D(EMBEDDING, 2),
            Residual1D(EMBEDDING, 4),
            Residual1D(EMBEDDING, 8),
        )

    def forward(self, values):
        return self.blocks(self.input(values))


class AxialBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.state = nn.Sequential(
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels),
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

    def forward(self, values):
        return F.gelu(values + self.time(self.state(values)))


def _sample_axis(features, positions):
    """Linearly sample [C,L] features on the fixed [-192,192] ft axis."""
    coordinate = positions + REFERENCE_HALF_WIDTH_FT
    low = torch.floor(coordinate).long()
    high = low + 1
    valid = (low >= 0) & (high < features.shape[-1])
    low_safe = low.clamp(0, features.shape[-1] - 1)
    high_safe = high.clamp(0, features.shape[-1] - 1)
    fraction = (coordinate - low.float()).clamp(0.0, 1.0)
    sampled = features[:, low_safe] * (1.0 - fraction)[None] + features[:, high_safe] * fraction[None]
    return sampled, valid


class ChunkedCostMapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedGREncoder()
        self.query_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)
        self.reference_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)
        self.query_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)
        self.reference_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)

        # 4 cosine heads, bilinear, value/d1 differences, three validity
        # channels, state/chunk coordinates, eight geometry/recursion
        # channels, and the current recursive anchor offset.
        input_channels = HEADS + 1 + 2 + 3 + 2 + 8 + 1
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, BASE_CHANNELS, 1),
            nn.GroupNorm(4, BASE_CHANNELS),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[AxialBlock(BASE_CHANNELS, dilation) for dilation in AXIAL_DILATIONS]
        )
        self.output = nn.Conv2d(BASE_CHANNELS, 1, 1)
        self.log_scale = nn.Parameter(torch.tensor(math.log(4.0)))

    def forward(
        self,
        horizontal,
        typewell,
        node_positions,
        candidate_positions,
        geometry,
        channel_control="full",
    ):
        horizontal_embedding = self.encoder(horizontal)[0]
        typewell_embedding = self.encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(typewell_embedding, candidate_positions)

        qh = self.query_heads(query[None])[0].reshape(HEADS, HEAD_DIM, -1)
        kh = self.reference_heads(reference[None])[0].reshape(HEADS, HEAD_DIM, -1)
        qh = F.normalize(qh, dim=1, eps=1e-5)
        kh = F.normalize(kh, dim=1, eps=1e-5)
        cosine = torch.einsum("hdt,hds->hst", qh, kh)

        qb = F.normalize(self.query_bilinear(query[None])[0], dim=0, eps=1e-5)
        kb = F.normalize(self.reference_bilinear(reference[None])[0], dim=0, eps=1e-5)
        bilinear = torch.einsum("dt,ds->st", qb, kb)[None]
        if channel_control == "no_cosine":
            cosine = torch.zeros_like(cosine)
        elif channel_control == "no_bilinear":
            bilinear = torch.zeros_like(bilinear)
        scale = torch.exp(self.log_scale).clamp(1.0, 20.0)

        query_raw = horizontal[0, :, node_positions]
        reference_raw, raw_axis_valid = _sample_axis(typewell[0], candidate_positions)
        value_difference = -torch.abs(reference_raw[0, :, None] - query_raw[0, None, :])[None]
        d1_difference = -torch.abs(reference_raw[2, :, None] - query_raw[2, None, :])[None]

        query_valid = query_raw[1][None, None, :].expand(1, len(candidate_positions), -1)
        reference_valid = reference_raw[1][None, :, None].expand(1, -1, query.shape[-1])
        joint_valid = query_valid * reference_valid
        state_coordinate = (
            torch.from_numpy(STATE_OFFSETS).to(query.device, query.dtype)[None, :, None]
            / STATE_HALF_WIDTH_FT
        ).expand(1, -1, query.shape[-1])
        chunk_coordinate = torch.linspace(0.0, 1.0, query.shape[-1], device=query.device, dtype=query.dtype)
        chunk_coordinate = chunk_coordinate[None, None, :].expand(1, len(candidate_positions), -1)
        geometry_volume = geometry[None, :, None, :].expand(1, -1, len(candidate_positions), -1)[0]
        anchor_offset = candidate_positions[len(candidate_positions) // 2] / REFERENCE_HALF_WIDTH_FT
        anchor_volume = anchor_offset.reshape(1, 1, 1).expand(1, len(candidate_positions), query.shape[-1])

        volume = torch.cat(
            [
                scale * cosine,
                scale * bilinear,
                value_difference,
                d1_difference,
                query_valid,
                reference_valid,
                joint_valid,
                state_coordinate,
                chunk_coordinate,
                geometry_volume,
                anchor_volume,
            ],
            dim=0,
        )[None]
        logits = self.output(self.blocks(self.input(volume))).squeeze(0).squeeze(0)
        candidate_valid = axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)
        logits = logits.masked_fill(~candidate_valid[:, None], -1.0e4)
        return logits


# ---------------------------------------------------------------------------
# Chunk construction, loss, and prediction
# ---------------------------------------------------------------------------


def _chunk_payload(
    record,
    start,
    deployed_anchor=None,
    previous_delta=0.0,
    previous_rate=0.0,
    control="full",
    shuffled_typewell=None,
):
    nodes = record.node_rows
    end = min(int(start) + CHUNK_TARGET_NODES, len(nodes) - 1)
    start_row = int(nodes[int(start)])
    target_rows = nodes[int(start) + 1 : end + 1]
    start_md = float(record.md[start_row])
    end_md = float(record.md[target_rows[-1]])
    crop_start = int(np.searchsorted(record.md, start_md - CHUNK_CONTEXT_MD_FT, side="left"))
    crop_end = int(np.searchsorted(record.md, end_md + CHUNK_CONTEXT_MD_FT, side="right"))
    crop_end = max(crop_end, int(target_rows[-1]) + 1)

    horizontal = record.horizontal[:, crop_start:crop_end].copy()
    if control == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
    typewell = (
        shuffled_typewell.typewell.copy()
        if control == "shuffled_typewell" and shuffled_typewell is not None
        else record.typewell.copy()
    )
    node_positions = (target_rows - crop_start).astype(np.int64)

    true_anchor = float(record.truth[start_row])
    if deployed_anchor is None:
        deployed_anchor = true_anchor
    deployed_anchor = float(deployed_anchor)
    anchor_offset = deployed_anchor - float(record.anchor_tvt)
    candidate_positions = anchor_offset + STATE_OFFSETS
    target_state = record.truth[target_rows].astype(np.float32) - deployed_anchor

    previous_delta_channel = np.full(
        len(target_rows), np.clip(float(previous_delta) / 32.0, -4.0, 4.0), np.float32
    )
    previous_rate_channel = np.full(
        len(target_rows), np.clip(float(previous_rate) / 0.05, -4.0, 4.0), np.float32
    )
    # The two recursive channels use exactly the same definition in teacher,
    # corrupted-anchor training, and rollout.
    condition = record.condition
    geometry = np.stack(
        [
            np.clip(condition[2, target_rows], 0.0, 1.0),
            np.clip(condition[3, target_rows], -1.0, 1.0),
            np.clip(condition[4, target_rows], -4.0, 4.0),
            np.clip(condition[7, target_rows], -2.0, 2.0),
            np.clip(condition[8, target_rows], -2.0, 2.0),
            np.clip(condition[9, target_rows], 0.0, 1.0),
            previous_delta_channel,
            previous_rate_channel,
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "horizontal": horizontal,
        "typewell": typewell,
        "node_positions": node_positions,
        "candidate_positions": candidate_positions.astype(np.float32),
        "geometry": geometry,
        "target_state": target_state,
        "target_rows": target_rows,
        "deployed_anchor": deployed_anchor,
        "previous_delta": float(previous_delta),
        "previous_rate": float(previous_rate),
        "distance": start_md - float(record.md[record.anchor_index]),
        "channel_control": (
            control if control in {"no_bilinear", "no_cosine"} else "full"
        ),
    }


def _tensor_payload(payload, device):
    return (
        torch.from_numpy(payload["horizontal"])[None].to(device),
        torch.from_numpy(payload["typewell"])[None].to(device),
        torch.from_numpy(payload["node_positions"]).long().to(device),
        torch.from_numpy(payload["candidate_positions"]).to(device),
        torch.from_numpy(payload["geometry"]).to(device),
    )


def _loss(model, payload, device):
    tensors = _tensor_payload(payload, device)
    logits = model(*tensors)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    target_distribution = torch.exp(-0.5 * ((states - target[None]) / TARGET_SIGMA_FT) ** 2)
    target_distribution = target_distribution / target_distribution.sum(dim=0, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(logits, dim=0)
    ce = -(target_distribution * log_probability).sum(dim=0).mean()
    probability = F.softmax(logits, dim=0)
    expected = (probability * states).sum(dim=0)
    offset = F.smooth_l1_loss(expected / 8.0, target / 8.0)
    expected_path = torch.cat([torch.zeros(1, device=device), expected])
    target_path = torch.cat([torch.zeros(1, device=device), target])
    slope = F.smooth_l1_loss(torch.diff(expected_path) / 2.0, torch.diff(target_path) / 2.0)
    total = LOSS_CE * ce + LOSS_OFFSET * offset + LOSS_SLOPE * slope
    return total, {"ce": float(ce.detach()), "offset": float(offset.detach()), "slope": float(slope.detach())}


@torch.no_grad()
def _predict_chunk(model, payload, device):
    tensors = _tensor_payload(payload, device)
    amp_enabled = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        logits = model(*tensors, channel_control=payload["channel_control"])
    probability = F.softmax(logits.float(), dim=0)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    expected = (probability * states).sum(dim=0).cpu().numpy()
    entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=0).cpu().numpy()
    mode = STATE_OFFSETS[torch.argmax(probability, dim=0).cpu().numpy()]
    return expected.astype(np.float64), mode.astype(np.float64), entropy.astype(np.float64)


def _anchor_noise(noise_library, distance, rng):
    if not noise_library:
        return {"anchor_error": 0.0, "previous_delta_error": 0.0}
    if distance <= 500.0:
        key = "0_500"
    elif distance <= 1000.0:
        key = "500_1000"
    elif distance <= 2000.0:
        key = "1000_2000"
    else:
        key = "gt2000"
    candidates = noise_library.get(key, [])
    if not candidates:
        candidates = [value for values in noise_library.values() for value in values]
    if not candidates:
        return {"anchor_error": 0.0, "previous_delta_error": 0.0}
    sample = candidates[int(rng.integers(0, len(candidates)))]
    return {
        "anchor_error": float(np.clip(sample["anchor_error"], -64.0, 64.0)),
        "previous_delta_error": float(
            np.clip(sample["previous_delta_error"], -64.0, 64.0)
        ),
    }


def _train_epoch(model, records, optimizer, scaler, device, epoch, noise_library=None, corruption_probability=0.0):
    model.train()
    rng = np.random.default_rng(SEED + 1000 * epoch + int(100 * corruption_probability))
    order = rng.permutation(len(records))
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "ce": 0.0, "offset": 0.0, "slope": 0.0, "samples": 0, "skipped": 0}
    steps = 0
    amp_enabled = device.type == "cuda"
    for sequence_number, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        count = min(SAMPLES_PER_WELL, len(record.chunk_starts))
        starts = rng.choice(record.chunk_starts, size=count, replace=False)
        for start in starts:
            start = int(start)
            start_row = int(record.node_rows[int(start)])
            distance = float(record.md[start_row] - record.md[record.anchor_index])
            previous_start = max(0, start - CHUNK_STRIDE_NODES)
            previous_row = int(record.node_rows[previous_start])
            previous_md_delta = max(float(record.md[start_row] - record.md[previous_row]), 1.0)
            true_previous_delta = float(record.truth[start_row] - record.truth[previous_row])
            noise = {"anchor_error": 0.0, "previous_delta_error": 0.0}
            if noise_library and rng.random() < corruption_probability:
                noise = _anchor_noise(noise_library, distance, rng)
            try:
                deployed_anchor = float(record.truth[start_row]) + noise["anchor_error"]
                previous_delta = true_previous_delta + noise["previous_delta_error"]
                previous_rate = previous_delta / previous_md_delta
                payload = _chunk_payload(
                    record,
                    start,
                    deployed_anchor=deployed_anchor,
                    previous_delta=previous_delta,
                    previous_rate=previous_rate,
                )
                if float(np.max(np.abs(payload["target_state"]))) > STATE_HALF_WIDTH_FT - 0.5:
                    totals["skipped"] += 1
                    continue
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss, parts = _loss(model, payload, device)
                    scaled_loss = loss / GRADIENT_ACCUMULATION
                if not torch.isfinite(loss):
                    totals["skipped"] += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(scaled_loss).backward()
                totals["loss"] += float(loss.detach())
                for key in ("ce", "offset", "slope"):
                    totals[key] += parts[key]
                totals["samples"] += 1
                if totals["samples"] % GRADIENT_ACCUMULATION == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
            except RuntimeError:
                raise
        if sequence_number % 100 == 0 or sequence_number == len(order):
            print(
                f"SEGN train epoch={epoch} wells={sequence_number}/{len(order)} "
                f"loss={totals['loss'] / max(totals['samples'], 1):.4f}",
                flush=True,
            )
    if totals["samples"] % GRADIENT_ACCUMULATION:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    result = {
        key: (value / max(totals["samples"], 1) if key in {"loss", "ce", "offset", "slope"} else value)
        for key, value in totals.items()
    }
    result["optimizer_steps"] = steps
    return result


@torch.no_grad()
def _predict_record(model, record, device, mode="rollout", control="full", shuffled_typewell=None, return_trace=False):
    model.eval()
    node_prediction = np.full(len(record.node_rows), np.nan, np.float64)
    node_prediction[0] = float(record.anchor_tvt)
    entropy_values = []
    disagreement_values = []
    trace = []
    prior_deployed_anchor = float(record.anchor_tvt)
    prior_start = 0
    for start in record.chunk_starts:
        start = int(start)
        start_row = int(record.node_rows[start])
        if mode == "teacher":
            deployed_anchor = float(record.truth[start_row])
        else:
            if not np.isfinite(node_prediction[start]):
                prior = np.flatnonzero(np.isfinite(node_prediction[: start + 1]))
                if not len(prior):
                    raise RuntimeError(f"{record.well}: rollout lost its anchor")
                deployed_anchor = float(node_prediction[prior[-1]])
            else:
                deployed_anchor = float(node_prediction[start])
        prior_row = int(record.node_rows[prior_start])
        previous_md_delta = max(float(record.md[start_row] - record.md[prior_row]), 1.0)
        previous_delta = float(deployed_anchor - prior_deployed_anchor) if start > 0 else 0.0
        previous_rate = previous_delta / previous_md_delta if start > 0 else 0.0
        true_previous_delta = (
            float(record.truth[start_row] - record.truth[prior_row]) if start > 0 else 0.0
        )
        # Truth is used below only to score the already-chosen deployed
        # anchor.  It never changes or clips the inference state.
        anchor_error = deployed_anchor - float(record.truth[start_row])
        payload = _chunk_payload(
            record,
            start,
            deployed_anchor=deployed_anchor,
            previous_delta=previous_delta,
            previous_rate=previous_rate,
            control=control,
            shuffled_typewell=shuffled_typewell,
        )
        expected, mode_state, entropy = _predict_chunk(model, payload, device)
        target_rows = payload["target_rows"]
        end = start + len(target_rows)
        node_prediction[start + 1 : end + 1] = deployed_anchor + expected
        entropy_values.extend(entropy.tolist())
        disagreement_values.extend(np.abs(expected - mode_state).tolist())
        trace.append(
            {
                "well": record.well,
                "start": start,
                "distance": payload["distance"],
                "anchor_error": anchor_error,
                "previous_delta": previous_delta,
                "previous_delta_error": previous_delta - true_previous_delta,
                "endpoint_error": float(node_prediction[end] - record.truth[target_rows[-1]]),
            }
        )
        prior_deployed_anchor = deployed_anchor
        prior_start = start
    valid_nodes = np.flatnonzero(np.isfinite(node_prediction))
    support_md = record.md[record.node_rows[valid_nodes]]
    support_tvt = node_prediction[valid_nodes]
    prediction = np.interp(record.md[record.eval_index], support_md, support_tvt)
    truth = record.truth[record.eval_index].astype(np.float64)
    diagnostics = {
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else math.nan,
        "mean_decode_disagreement": float(np.mean(disagreement_values)) if disagreement_values else math.nan,
    }
    return truth, prediction, diagnostics, trace if return_trace else []


def _metrics_from_rows(rows, truths, predictions):
    bywell = pd.DataFrame(rows)
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    distance = np.concatenate([row.pop("_distance") for row in rows]) if rows and "_distance" in rows[0] else None
    metrics = {
        "rows": len(truth),
        "wells": len(bywell),
        "rmse": _rmse(truth, prediction),
        "mae": _mae(truth, prediction),
        "median_well_rmse": float(bywell["well_rmse"].median()),
        "p75_well_rmse": float(bywell["well_rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["well_rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["well_rmse"].max()),
        "mean_entropy": float(bywell["mean_entropy"].mean()),
        "mean_decode_disagreement": float(bywell["mean_decode_disagreement"].mean()),
    }
    if distance is not None:
        strata = (
            ("near_0_500_rmse", distance <= 500.0),
            ("mid_500_1000_rmse", (distance > 500.0) & (distance <= 1000.0)),
            ("mid_1000_2000_rmse", (distance > 1000.0) & (distance <= 2000.0)),
            ("far_gt2000_rmse", distance > 2000.0),
        )
        for name, mask in strata:
            metrics[name] = _rmse(truth[mask], prediction[mask]) if int(mask.sum()) else math.nan
    return metrics, bywell


@torch.no_grad()
def _evaluate(model, records, device, mode="rollout", control="full", save_prefix=None):
    rows = []
    truths = []
    predictions = []
    permutation = np.roll(np.arange(len(records)), 1)
    for number, record in enumerate(records, 1):
        shuffled = records[int(permutation[number - 1])] if control == "shuffled_typewell" else None
        truth, prediction, diagnostics, _ = _predict_record(
            model,
            record,
            device,
            mode=mode,
            control=control,
            shuffled_typewell=shuffled,
        )
        distance = record.md[record.eval_index] - float(record.md[record.anchor_index])
        rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "well_rmse": _rmse(truth, prediction),
                "well_mae": _mae(truth, prediction),
                "mean_entropy": diagnostics["mean_entropy"],
                "mean_decode_disagreement": diagnostics["mean_decode_disagreement"],
                "_distance": distance.astype(np.float64),
            }
        )
        truths.append(truth)
        predictions.append(prediction)
        if number % 50 == 0 or number == len(records):
            print(f"SEGN evaluate {mode}/{control} {number}/{len(records)}", flush=True)
    metrics, bywell = _metrics_from_rows(rows, truths, predictions)
    if save_prefix:
        clean = bywell.drop(columns=["_distance"], errors="ignore")
        clean.to_csv(OUT_DIR / f"{save_prefix}_bywell.csv", index=False)
        np.savez_compressed(
            OUT_DIR / f"{save_prefix}_rows.npz",
            truth=np.concatenate(truths).astype(np.float32),
            prediction=np.concatenate(predictions).astype(np.float32),
            well=np.concatenate([np.repeat(record.well, len(record.eval_index)) for record in records]),
        )
    return metrics, bywell


def _build_noise_library(model, records, device):
    library = {"0_500": [], "500_1000": [], "1000_2000": [], "gt2000": []}
    traces = []
    for number, record in enumerate(records, 1):
        _, _, _, trace = _predict_record(model, record, device, mode="rollout", return_trace=True)
        for item in trace:
            distance = float(item["distance"])
            if distance <= 500.0:
                key = "0_500"
            elif distance <= 1000.0:
                key = "500_1000"
            elif distance <= 2000.0:
                key = "1000_2000"
            else:
                key = "gt2000"
            library[key].append(
                {
                    "anchor_error": float(item["anchor_error"]),
                    "previous_delta_error": float(item["previous_delta_error"]),
                }
            )
            traces.append(item)
        if number % 50 == 0 or number == len(records):
            print(f"SEGN rollout-noise {number}/{len(records)}", flush=True)
    pd.DataFrame(traces).to_csv(OUT_DIR / "fold3_rollout_anchor_errors.csv", index=False)
    summary = {}
    for key, values in library.items():
        anchor_values = np.asarray([value["anchor_error"] for value in values], np.float64)
        delta_values = np.asarray(
            [value["previous_delta_error"] for value in values], np.float64
        )
        summary[key] = {
            "count": len(values),
            "anchor_mean": float(np.mean(anchor_values)) if len(values) else math.nan,
            "anchor_std": float(np.std(anchor_values)) if len(values) else math.nan,
            "anchor_p05": float(np.quantile(anchor_values, 0.05)) if len(values) else math.nan,
            "anchor_p50": float(np.quantile(anchor_values, 0.50)) if len(values) else math.nan,
            "anchor_p95": float(np.quantile(anchor_values, 0.95)) if len(values) else math.nan,
            "previous_delta_error_mean": float(np.mean(delta_values)) if len(values) else math.nan,
            "previous_delta_error_std": float(np.std(delta_values)) if len(values) else math.nan,
            "error_correlation": (
                float(np.corrcoef(anchor_values, delta_values)[0, 1])
                if len(values) > 2 and np.std(anchor_values) > 0 and np.std(delta_values) > 0
                else math.nan
            ),
        }
    _write_json(OUT_DIR / "rollout_noise_summary.json", summary)
    return library, summary


def _fit_model(records, device, epochs, phase, initial_state=None, noise_library=None, tune_records=None):
    _seed_everything(SEED + sum(ord(char) for char in phase))
    model = ChunkedCostMapNet().to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=MIN_LEARNING_RATE,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_rmse = math.inf
    bad_reports = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        if noise_library:
            corruption_probability = min(0.80, 0.20 + 0.60 * epoch / max(epochs, 1))
        else:
            corruption_probability = 0.0
        train = _train_epoch(
            model,
            records,
            optimizer,
            scaler,
            device,
            epoch,
            noise_library=noise_library,
            corruption_probability=corruption_probability,
        )
        scheduler.step()
        report = {
            "phase": phase,
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            "corruption_probability": corruption_probability,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        should_report = epoch == 1 or epoch % REPORT_EVERY == 0 or epoch == epochs
        if tune_records and should_report:
            teacher, _ = _evaluate(model, tune_records, device, mode="teacher")
            rollout, _ = _evaluate(model, tune_records, device, mode="rollout")
            report.update({f"tune_teacher_{key}": value for key, value in teacher.items()})
            report.update({f"tune_rollout_{key}": value for key, value in rollout.items()})
            score = rollout["rmse"]
            if score < best_rmse - 1e-4:
                best_rmse = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_reports = 0
            else:
                bad_reports += 1
        elif not tune_records:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(report)
        print("SEGN EPOCH", report, flush=True)
        pd.DataFrame(history).to_csv(OUT_DIR / f"{phase}_history.csv", index=False)
        if tune_records and bad_reports >= PATIENCE_REPORTS:
            print(f"SEGN {phase} early stopping at epoch {epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    checkpoint = MODEL_DIR / f"{phase}_best.pt"
    torch.save(
        {
            "model": best_state,
            "best_epoch": best_epoch,
            "best_rollout_rmse": best_rmse,
            "config": {
                "state_half_width_ft": STATE_HALF_WIDTH_FT,
                "node_spacing_md_ft": NODE_SPACING_MD_FT,
                "chunk_target_nodes": CHUNK_TARGET_NODES,
                "chunk_stride_nodes": CHUNK_STRIDE_NODES,
            },
        },
        checkpoint,
    )
    return model, best_epoch, history, checkpoint


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


started = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this cost-map gateway")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not RECORD_DIRS:
    raise FileNotFoundError(
        "No native record directory exists. Run build_native_dualref_all_records_v0_cell.py first."
    )
record_files = {path.stem for directory in RECORD_DIRS for path in directory.glob("*.npz")}
if len(record_files) < 773:
    raise RuntimeError(
        f"Only {len(record_files)} native records exist; expected 773. "
        "Run build_native_dualref_all_records_v0_cell.py first."
    )

folds, fold_path = _load_folds()
noise_train_wells = folds.loc[folds["fold"].isin(NOISE_TRAIN_FOLDS), "well"].tolist()
noise_rollout_wells = folds.loc[folds["fold"].eq(NOISE_ROLLOUT_FOLD), "well"].tolist()
final_train_wells = folds.loc[folds["fold"].isin(FINAL_TRAIN_FOLDS), "well"].tolist()
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(final_train_wells) != 463 or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected Fold 3/4/5 or Fold 2 sizes: {len(final_train_wells)}/{len(valid_wells)}")

all_needed = sorted(set(noise_train_wells + noise_rollout_wells + final_train_wells + valid_wells))
records_by_well = {record.well: record for record in _load_records(all_needed, "fold2345")}
noise_train_records = [records_by_well[well] for well in noise_train_wells]
noise_rollout_records = [records_by_well[well] for well in noise_rollout_wells]
final_train_records = [records_by_well[well] for well in final_train_wells]
valid_records = [records_by_well[well] for well in valid_wells]

valid_order = np.argsort([record.eval_rows for record in valid_records])
tune_positions = np.rint(np.linspace(0, len(valid_order) - 1, TUNE_WELLS)).astype(int)
tune_records = [valid_records[int(valid_order[position])] for position in np.unique(tune_positions)]

guards = {
    "device": str(device),
    "fold_source": str(fold_path),
    "record_directories": [str(path) for path in RECORD_DIRS],
    "record_files": len(record_files),
    "noise_train_wells": len(noise_train_records),
    "noise_rollout_wells": len(noise_rollout_records),
    "final_train_wells": len(final_train_records),
    "valid_wells": len(valid_records),
    "tune_wells": len(tune_records),
    "minimum_chunks": int(min(len(record.chunk_starts) for record in records_by_well.values())),
    "mean_chunks": float(np.mean([len(record.chunk_starts) for record in records_by_well.values()])),
    "outer_train_target_axis_coverage": _target_axis_coverage(final_train_records),
    "fold2_target_axis_coverage_diagnostic": _target_axis_coverage(valid_records),
    "fold1_loaded": False,
}
print("SEGN CONFIG", guards, flush=True)
if guards["outer_train_target_axis_coverage"] < 0.95:
    raise RuntimeError(f"Outer-train state-axis coverage is too low: {guards}")

# Stage A: learn evidence with exact chunk anchors using only Folds 4/5.
noise_model, noise_epoch, noise_history, noise_checkpoint = _fit_model(
    noise_train_records,
    device,
    NOISE_STAGE_EPOCHS,
    "noise45_teacher",
)
noise_library, noise_summary = _build_noise_library(noise_model, noise_rollout_records, device)

# Stage B: warm-start and expose the final model to deployment-shaped anchor
# errors from the untouched-with-respect-to-Fold2 Fold-3 rollout.
final_model, best_epoch, final_history, final_checkpoint = _fit_model(
    final_train_records,
    device,
    FINAL_EPOCHS,
    "final345_rollout",
    initial_state=noise_model.state_dict(),
    noise_library=noise_library,
    tune_records=tune_records,
)

teacher_metrics, teacher_bywell = _evaluate(
    final_model,
    valid_records,
    device,
    mode="teacher",
    save_prefix="fold2_teacher",
)
rollout_metrics, rollout_bywell = _evaluate(
    final_model,
    valid_records,
    device,
    mode="rollout",
    save_prefix="fold2_rollout",
)
no_gr_metrics, _ = _evaluate(
    final_model,
    valid_records,
    device,
    mode="rollout",
    control="no_gr",
    save_prefix="fold2_no_gr",
)
shuffled_metrics, _ = _evaluate(
    final_model,
    valid_records,
    device,
    mode="rollout",
    control="shuffled_typewell",
    save_prefix="fold2_shuffled_typewell",
)
no_bilinear_metrics, _ = _evaluate(
    final_model,
    valid_records,
    device,
    mode="rollout",
    control="no_bilinear",
    save_prefix="fold2_no_bilinear",
)
no_cosine_metrics, _ = _evaluate(
    final_model,
    valid_records,
    device,
    mode="rollout",
    control="no_cosine",
    save_prefix="fold2_no_cosine",
)

comparison = pd.DataFrame(
    [
        {"mode": "teacher", **teacher_metrics},
        {"mode": "rollout", **rollout_metrics},
        {"mode": "no_gr", **no_gr_metrics},
        {"mode": "shuffled_typewell", **shuffled_metrics},
        {"mode": "no_bilinear", **no_bilinear_metrics},
        {"mode": "no_cosine", **no_cosine_metrics},
    ]
)
comparison.to_csv(OUT_DIR / "fold2_comparison.csv", index=False)
print("SEGN FOLD2 COMPARISON")
print(comparison.to_string(index=False), flush=True)

rollout_gap = rollout_metrics["rmse"] - teacher_metrics["rmse"]
no_gr_degradation = no_gr_metrics["rmse"] - rollout_metrics["rmse"]
shuffled_degradation = shuffled_metrics["rmse"] - rollout_metrics["rmse"]
no_bilinear_degradation = no_bilinear_metrics["rmse"] - rollout_metrics["rmse"]
no_cosine_degradation = no_cosine_metrics["rmse"] - rollout_metrics["rmse"]
evidence_pass = teacher_metrics["rmse"] <= EVIDENCE_GATE_RMSE
rollout_pass = (
    rollout_metrics["rmse"] <= ROLLOUT_GATE_RMSE
    and rollout_gap <= ROLLOUT_GAP_GATE
    and rollout_metrics.get("far_gt2000_rmse", math.inf) <= FAR_GATE_RMSE
)
causal_pass = (
    no_gr_degradation >= CONTROL_DEGRADATION_GATE
    and shuffled_degradation >= CONTROL_DEGRADATION_GATE
)
target_pass = rollout_metrics["rmse"] < TARGET_RMSE

if not evidence_pass:
    next_step = "FAIL EVIDENCE -> close this SEGN representation"
elif not rollout_pass:
    next_step = "EVIDENCE PASS / ROLLOUT FAIL -> keep encoder; repair recursive anchor propagation"
elif not causal_pass:
    next_step = "ROLLOUT PASS / CONTROL FAIL -> inspect geometry shortcut before Fold1"
elif target_pass:
    next_step = "TARGET PASS -> freeze recipe and run untouched Fold1"
else:
    next_step = "GATE PASS -> add one cross-fitted rowwise combiner before untouched Fold1"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_overlap_rebased_autoregressive_costmap_gateway",
    "sample_unit": "physical-MD chunk nested within one complete well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_spatial_strong_cp12_gold_pf": False,
    "protocol": {
        "noise_model_train_folds": list(NOISE_TRAIN_FOLDS),
        "noise_rollout_fold": NOISE_ROLLOUT_FOLD,
        "final_train_folds": list(FINAL_TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "noise_model_epoch": noise_epoch,
        "selected_epoch": best_epoch,
    },
    "representation": {
        "node_spacing_md_ft": NODE_SPACING_MD_FT,
        "chunk_target_nodes": CHUNK_TARGET_NODES,
        "chunk_stride_nodes": CHUNK_STRIDE_NODES,
        "nominal_chunk_md_ft": NODE_SPACING_MD_FT * CHUNK_TARGET_NODES,
        "nominal_rebase_md_ft": NODE_SPACING_MD_FT * CHUNK_STRIDE_NODES,
        "state_half_width_ft": STATE_HALF_WIDTH_FT,
        "decode": "posterior_mean",
        "previous_delta_input": True,
        "previous_rate_input": True,
        "matched_joint_anchor_delta_corruption": True,
    },
    "parameters": int(sum(parameter.numel() for parameter in final_model.parameters())),
    "guards": guards,
    "rollout_noise": noise_summary,
    "teacher": teacher_metrics,
    "rollout": rollout_metrics,
    "controls": {
        "no_gr": no_gr_metrics,
        "shuffled_typewell": shuffled_metrics,
        "no_bilinear": no_bilinear_metrics,
        "no_cosine": no_cosine_metrics,
        "no_gr_degradation": no_gr_degradation,
        "shuffled_typewell_degradation": shuffled_degradation,
        "no_bilinear_degradation": no_bilinear_degradation,
        "no_cosine_degradation": no_cosine_degradation,
    },
    "diagnostics": {
        "rollout_minus_teacher_rmse": rollout_gap,
    },
    "gates": {
        "teacher_rmse_le": EVIDENCE_GATE_RMSE,
        "rollout_rmse_le": ROLLOUT_GATE_RMSE,
        "rollout_gap_le": ROLLOUT_GAP_GATE,
        "far_gt2000_rmse_le": FAR_GATE_RMSE,
        "each_control_degradation_ge": CONTROL_DEGRADATION_GATE,
        "target_rmse_lt": TARGET_RMSE,
    },
    "evidence_pass": evidence_pass,
    "rollout_pass": rollout_pass,
    "causal_pass": causal_pass,
    "target_6p7_pass": target_pass,
    "strict_pass": bool(evidence_pass and rollout_pass and causal_pass),
    "checkpoint": str(final_checkpoint),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": float(__import__("shutil").disk_usage(OUT_DIR).free / 2**30),
    "next": next_step,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("SEGN FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)

del noise_model, final_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
