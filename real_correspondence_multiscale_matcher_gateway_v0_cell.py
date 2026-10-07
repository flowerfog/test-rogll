"""Fold-honest gateway for a real-correspondence multiscale GR matcher.

This is deliberately not another trajectory head.  It trains a local matcher
from real horizontal/typewell correspondences and asks one narrow question:
can a held-out well's true TVT state be ranked above the +/-24 ft aliases?

Protocol
--------
1. Reuse the legal-input records produced by
   ``native_resolution_dualref_capacity_v0_cell.py``.  Those records calibrate
   the assigned typewell from the visible prefix only.
   Training correspondences are sampled only from each record's official
   hidden suffix, matching the state axis and deployment task.
2. Tune the training epoch on Fold 2 while training on Folds 3-5.
3. Reinitialise and train on Folds 2-5 for the frozen epoch count.
4. Evaluate Fold 1 exactly once, including NCC, no-GR and shuffled-typewell
   controls.

The matcher sees four MD-context scales and four TVT-context scales.  It has no
state coordinate and therefore cannot solve the task from an absolute-position
shortcut.  The full 257-state loss includes explicit +/-24 and +/-48 ft alias
margins.
"""

from dataclasses import dataclass
from pathlib import Path
import gc
import json
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "real_correspondence_multiscale_matcher_gateway_v0"
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
RECORD_DIR = next((path for path in RECORD_DIR_CANDIDATES if path.exists()), RECORD_DIR_CANDIDATES[0])
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260731
TUNE_FOLD = 2
TEST_FOLD = 1
REQUIRE_CUDA = True

STATE_HALF_WIDTH_FT = 128
STATE_OFFSETS = np.arange(-STATE_HALF_WIDTH_FT, STATE_HALF_WIDTH_FT + 1, dtype=np.float32)
N_STATES = len(STATE_OFFSETS)
REFERENCE_CONTEXT_FT = 64

# Horizontal windows are sampled in MD.  Typewell windows are sampled in TVT.
# The scale bank lets the scorer infer the unknown local MD-to-TVT compression.
QUERY_HALF_WIDTHS_MD = np.asarray([64.0, 128.0, 256.0, 512.0], np.float32)
REFERENCE_HALF_WIDTHS_TVT = np.asarray([4.0, 8.0, 16.0, 32.0], np.float32)
WINDOW_POINTS = 65
WINDOW_UNIT_GRID = np.linspace(-1.0, 1.0, WINDOW_POINTS, dtype=np.float32)
# A state at the typewell edge has about half of a symmetric window.  The
# validity channel makes that one-sided context explicit, so rejecting it at
# 80% coverage discards legal truth rows and biases retrieval upward.
MIN_REFERENCE_SCALE_COVERAGE = 0.45

TRAIN_POSITIONS = 96
TUNE_POSITIONS = 128
TEST_POSITIONS = 192
TARGET_SIGMA_FT = 1.5
ALIAS_OFFSETS_FT = (24, 48)
ALIAS_MARGIN = 0.35
ALIAS_LOSS_WEIGHT = 0.50

EMBEDDING = 48
MAX_TUNING_EPOCHS = 10
MIN_TUNING_EPOCHS = 4
EARLY_STOP_REPORTS = 3
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 2e-5
GRADIENT_CLIP = 5.0
REPORT_EVERY_WELLS = 100

# These are gateway thresholds, not claims about final path RMSE.
TOP8_WITHIN5_ABSOLUTE_GATE = 0.50
TOP8_GAIN_VS_NCC_GATE = 0.15
MEDIAN_RANK_GATE = 5.0
ALIAS_WIN_GATE = 0.65
CONTROL_TOP8_DROP_GATE = 0.10


# ---------------------------------------------------------------------------
# Utilities and fold loading
# ---------------------------------------------------------------------------

def _seed_everything(seed):
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


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _horizontal_wells():
    paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
    wells = [path.name.split("__", 1)[0] for path in paths]
    if len(wells) != 773:
        raise FileNotFoundError(
            f"Expected 773 train wells under {DATA_ROOT / 'train'}; found {len(wells)}"
        )
    return wells


def _load_folds(wells):
    fold_path = next((path for path in FOLD_CANDIDATES if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError(
            "Cannot find fold-clean well_metadata.csv. Expected one of: "
            + repr([str(path) for path in FOLD_CANDIDATES])
        )
    frame = pd.read_csv(fold_path, dtype=str)
    well_column = next((column for column in ("well", "well_id", "WELL") if column in frame), None)
    fold_column = next((column for column in ("fold", "outer_fold", "Fold") if column in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Fold file columns are {frame.columns.tolist()}")
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    frame = frame[frame["well"].isin(set(wells))].copy()
    missing = sorted(set(wells) - set(frame["well"]))
    if missing:
        raise RuntimeError(f"Fold metadata misses {len(missing)} wells: {missing[:10]}")
    if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Unexpected folds: {sorted(frame['fold'].unique().tolist())}")
    return frame.sort_values(["fold", "well"]).reset_index(drop=True), fold_path


# ---------------------------------------------------------------------------
# Compact legal-input records and multiscale patch banks
# ---------------------------------------------------------------------------

@dataclass
class RawRecord:
    well: str
    horizontal: np.ndarray
    typewell: np.ndarray
    md: np.ndarray
    eval_index: np.ndarray
    truth: np.ndarray
    anchor_tvt: float
    anchor_index: int
    eval_rows: int


@dataclass
class QuerySet:
    patches: np.ndarray
    labels: np.ndarray
    label_index: np.ndarray
    row_weight: np.ndarray
    deep_mask: np.ndarray
    sample_rows: np.ndarray
    requested_rows: int


@dataclass
class WellSample:
    well: str
    fold: int
    reference: np.ndarray
    state_valid: np.ndarray
    train: QuerySet | None
    evaluation: QuerySet | None
    eval_rows: int
    label_coverage: float


def _load_raw_record(well):
    path = RECORD_DIR / f"{well}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run native_resolution_dualref_capacity_v0_cell.py "
            "once to build the legal-input record cache."
        )
    with np.load(path, allow_pickle=False) as data:
        horizontal = data["horizontal"].astype(np.float32)
        typewell = data["typewell"].astype(np.float32)
        if horizontal.ndim != 2 or horizontal.shape[0] < 2:
            raise RuntimeError(f"{well}: bad horizontal shape {horizontal.shape}")
        if typewell.ndim != 2 or typewell.shape[0] < 2:
            raise RuntimeError(f"{well}: bad typewell shape {typewell.shape}")
        return RawRecord(
            well=well,
            horizontal=horizontal[:2],
            typewell=typewell[:2],
            md=data["md"].astype(np.float32),
            eval_index=data["eval_index"].astype(np.int32),
            truth=data["truth"].astype(np.float32),
            anchor_tvt=float(data["anchor_tvt"][0]),
            anchor_index=int(data["anchor_index"][0]),
            eval_rows=int(data["eval_rows"][0]),
        )


def _select_evenly(indices, count):
    indices = np.asarray(indices, np.int64)
    if not len(indices):
        return indices
    selected = np.linspace(0, len(indices) - 1, min(count, len(indices))).round().astype(int)
    return np.unique(indices[selected])


def _window_bank(axis, value, valid, centres, half_widths):
    axis = np.asarray(axis, np.float64)
    value = np.asarray(value, np.float64)
    valid = np.asarray(valid, np.float64)
    centres = np.asarray(centres, np.float64)
    half_widths = np.asarray(half_widths, np.float64)
    queries = (
        centres[:, None, None]
        + half_widths[None, :, None] * WINDOW_UNIT_GRID[None, None, :]
    )
    flat = queries.reshape(-1)
    interpolated = np.interp(flat, axis, value).reshape(queries.shape)
    interpolated_valid = np.interp(flat, axis, valid).reshape(queries.shape)
    in_range = (queries >= axis[0]) & (queries <= axis[-1])
    mask = in_range & (interpolated_valid >= 0.5)
    output = np.empty((*queries.shape[:2], 2, WINDOW_POINTS), dtype=np.float16)
    output[:, :, 0] = np.where(mask, interpolated, 0.0).astype(np.float16)
    output[:, :, 1] = mask.astype(np.float16)
    return output


def _reference_bank(record):
    length = record.typewell.shape[1]
    expected = N_STATES + 2 * REFERENCE_CONTEXT_FT
    if length != expected:
        raise RuntimeError(
            f"{record.well}: expected typewell length {expected}, found {length}"
        )
    reference_axis = np.arange(length, dtype=np.float32) - float(length // 2)
    bank = _window_bank(
        reference_axis,
        record.typewell[0],
        record.typewell[1],
        STATE_OFFSETS,
        REFERENCE_HALF_WIDTHS_TVT,
    )
    coverage_by_scale = bank[:, :, 1].astype(np.float32).mean(axis=-1)
    state_valid = np.max(coverage_by_scale, axis=1) >= MIN_REFERENCE_SCALE_COVERAGE
    return bank, state_valid


def _query_set(record, rows, state_valid, include_known):
    rows = np.asarray(rows, np.int64)
    labels = record.truth[rows].astype(np.float64) - record.anchor_tvt
    label_index = np.rint(labels + STATE_HALF_WIDTH_FT).astype(np.int64)
    axis_valid = (
        np.isfinite(labels)
        & (label_index >= 0)
        & (label_index < N_STATES)
    )
    safe_index = np.clip(label_index, 0, N_STATES - 1)
    axis_valid &= state_valid[safe_index]
    original_rows = len(rows)
    rows = rows[axis_valid]
    labels = labels[axis_valid]
    label_index = label_index[axis_valid]
    centres_md = record.md[rows]
    patches = _window_bank(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        centres_md,
        QUERY_HALF_WIDTHS_MD,
    )
    is_eval = rows > record.anchor_index
    row_weight = np.where(is_eval, 1.0, 0.50 if include_known else 1.0).astype(np.float32)
    deep_mask = (record.md[rows] - record.md[record.anchor_index]) >= 1500.0
    return QuerySet(
        patches=patches,
        labels=labels.astype(np.float32),
        label_index=label_index.astype(np.int64),
        row_weight=row_weight,
        deep_mask=deep_mask,
        sample_rows=rows.astype(np.int32),
        requested_rows=int(original_rows),
    ), original_rows


def _build_well_sample(well, fold):
    record = _load_raw_record(well)
    reference, state_valid = _reference_bank(record)

    train_set = None
    if fold != TEST_FOLD:
        # The anchor-centred state axis describes the official hidden suffix.
        # Sampling the early heel adds rows that are legitimately far outside
        # +/-128 ft and creates a train/deploy mismatch.
        train_rows = _select_evenly(record.eval_index, TRAIN_POSITIONS)
        train_set, _ = _query_set(
            record, train_rows, state_valid, include_known=False
        )

    evaluation = None
    original_eval_rows = 0
    if fold in (TUNE_FOLD, TEST_FOLD):
        positions = TUNE_POSITIONS if fold == TUNE_FOLD else TEST_POSITIONS
        eval_rows = _select_evenly(record.eval_index, positions)
        evaluation, original_eval_rows = _query_set(
            record, eval_rows, state_valid, include_known=False
        )

    retained = (
        len(evaluation.labels) if evaluation is not None else len(train_set.labels)
    )
    expected = original_eval_rows if evaluation is not None else min(TRAIN_POSITIONS, len(record.truth))
    return WellSample(
        well=well,
        fold=int(fold),
        reference=reference,
        state_valid=state_valid,
        train=train_set,
        evaluation=evaluation,
        eval_rows=record.eval_rows,
        label_coverage=float(retained / max(expected, 1)),
    )


# ---------------------------------------------------------------------------
# Matcher model: no state coordinate, only cross-domain multiscale evidence
# ---------------------------------------------------------------------------

class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ResidualPatchBlock(nn.Module):
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
            ChannelNorm(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.network(values))


class PatchEncoder(nn.Module):
    def __init__(self, embedding=EMBEDDING):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 40, 9, padding=4),
            ChannelNorm(40),
            nn.GELU(),
            ResidualPatchBlock(40, 1),
            ResidualPatchBlock(40, 2),
            ResidualPatchBlock(40, 4),
        )
        self.output = nn.Sequential(
            nn.Linear(80, 96),
            nn.GELU(),
            nn.Linear(96, embedding),
        )

    def forward(self, values):
        hidden = self.network(values)
        pooled = torch.cat([hidden.mean(-1), hidden.amax(-1)], dim=1)
        return F.normalize(self.output(pooled), dim=1, eps=1e-6)


class RealCorrespondenceMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = PatchEncoder()
        self.reference_encoder = PatchEncoder()
        q_scales = len(QUERY_HALF_WIDTHS_MD)
        r_scales = len(REFERENCE_HALF_WIDTHS_TVT)
        correlation_features = q_scales * r_scales
        summary_features = 2 + q_scales + r_scales
        coverage_features = q_scales + r_scales
        pair_features = correlation_features + summary_features + coverage_features
        self.scorer = nn.Sequential(
            nn.Linear(pair_features, 80),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(80, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, query, reference, state_valid):
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
        scale = F.softplus(self.logit_scale)
        correlation = scale * correlation
        flat = correlation.flatten(2)
        maximum = flat.amax(dim=2, keepdim=True)
        mean = flat.mean(dim=2, keepdim=True)
        max_by_query = correlation.amax(dim=3)
        max_by_reference = correlation.amax(dim=2)

        query_coverage = query[:, :, 1].mean(dim=-1)
        reference_coverage = reference[:, :, 1].mean(dim=-1)
        query_coverage = query_coverage[:, None].expand(-1, states, -1)
        reference_coverage = reference_coverage[None].expand(batch, -1, -1)
        features = torch.cat(
            [
                flat,
                maximum,
                mean,
                max_by_query,
                max_by_reference,
                query_coverage,
                reference_coverage,
            ],
            dim=2,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits.masked_fill(~state_valid[None], -1e4)


def _to_tensors(sample, query_set, device, no_gr=False, reference=None, state_valid=None):
    query = query_set.patches.astype(np.float32)
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
        torch.from_numpy(query_set.labels).to(device, non_blocking=True),
        torch.from_numpy(query_set.label_index).to(device, non_blocking=True),
        torch.from_numpy(query_set.row_weight).to(device, non_blocking=True),
    )


def _matcher_loss(logits, labels, label_index, row_weight, state_valid):
    states = torch.from_numpy(STATE_OFFSETS).to(logits.device)
    target = torch.exp(-0.5 * ((states[None] - labels[:, None]) / TARGET_SIGMA_FT) ** 2)
    target = target * state_valid[None]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    nll = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    ce = (nll * row_weight).sum() / row_weight.sum().clamp_min(1e-6)

    positive = logits.gather(1, label_index[:, None])[:, 0]
    alias_terms = []
    for offset in ALIAS_OFFSETS_FT:
        for sign in (-1, 1):
            alias_index = label_index + sign * int(offset)
            valid = (alias_index >= 0) & (alias_index < N_STATES)
            safe_index = alias_index.clamp(0, N_STATES - 1)
            valid = valid & state_valid[safe_index]
            if valid.any():
                alias = logits.gather(1, safe_index[:, None])[:, 0]
                alias_terms.append(F.relu(ALIAS_MARGIN - positive[valid] + alias[valid]).mean())
    alias_loss = torch.stack(alias_terms).mean() if alias_terms else logits.new_tensor(0.0)
    return ce + ALIAS_LOSS_WEIGHT * alias_loss, ce.detach(), alias_loss.detach()


# ---------------------------------------------------------------------------
# Retrieval metrics and NCC baseline
# ---------------------------------------------------------------------------

def _normalised_patch_vectors(patches):
    patches = np.asarray(patches, np.float32)
    value = patches[..., 0, :]
    mask = patches[..., 1, :]
    count = mask.sum(axis=-1, keepdims=True)
    mean = (value * mask).sum(axis=-1, keepdims=True) / np.maximum(count, 1.0)
    centred = (value - mean) * mask
    norm = np.sqrt((centred * centred).sum(axis=-1, keepdims=True))
    return centred / np.maximum(norm, 1e-5)


def _ncc_scores(query, reference, state_valid):
    q = _normalised_patch_vectors(query)
    r = _normalised_patch_vectors(reference)
    correlation = np.einsum("bql,stl->bsqt", q, r, optimize=True)
    score = correlation.max(axis=(2, 3))
    score[:, ~state_valid] = -1e4
    return score.astype(np.float32)


def _retrieval_arrays(scores, labels, label_index, deep_mask):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.float64)
    label_index = np.asarray(label_index, np.int64)
    prediction_index = np.argmax(scores, axis=1)
    prediction = STATE_OFFSETS[prediction_index].astype(np.float64)
    top8_index = np.argpartition(scores, -8, axis=1)[:, -8:]
    top5_index = np.argpartition(scores, -5, axis=1)[:, -5:]
    top8_values = STATE_OFFSETS[top8_index]
    top5_values = STATE_OFFSETS[top5_index]
    truth_score = scores[np.arange(len(scores)), label_index]
    rank = 1 + np.sum(scores > truth_score[:, None], axis=1)

    alias_wins = np.full(len(scores), np.nan, np.float64)
    for row in range(len(scores)):
        aliases = []
        for offset in ALIAS_OFFSETS_FT[:1]:
            for sign in (-1, 1):
                index = int(label_index[row] + sign * offset)
                if 0 <= index < N_STATES and scores[row, index] > -1e3:
                    aliases.append(scores[row, index])
        if aliases:
            alias_wins[row] = float(truth_score[row] > max(aliases))

    return {
        "label": labels,
        "prediction": prediction,
        "abs_error": np.abs(prediction - labels),
        "rank": rank.astype(np.int32),
        "top5_hit": np.any(np.abs(top5_values - labels[:, None]) <= 5.0, axis=1),
        "top8_hit": np.any(np.abs(top8_values - labels[:, None]) <= 5.0, axis=1),
        "alias_win": alias_wins,
        "deep": np.asarray(deep_mask, bool),
    }


def _summarise_payload(payload, requested_rows=None):
    label = payload["label"]
    prediction = payload["prediction"]
    deep = payload["deep"]
    finite_alias = np.isfinite(payload["alias_win"])
    covered_rows = int(len(label))
    if requested_rows is None:
        requested_rows = covered_rows
    requested_rows = int(requested_rows)
    return {
        "rows": covered_rows,
        "covered_rows": covered_rows,
        "requested_rows": requested_rows,
        "coverage": float(covered_rows / max(requested_rows, 1)),
        "wells": int(len(np.unique(payload["well"]))),
        "argmax_rmse": _rmse(label, prediction),
        "argmax_mae": float(np.mean(np.abs(label - prediction))),
        "top1_within5": float(np.mean(payload["abs_error"] <= 5.0)),
        "top5_within5": float(np.mean(payload["top5_hit"])),
        "top8_within5": float(np.mean(payload["top8_hit"])),
        "effective_top1_within5": float(
            np.sum(payload["abs_error"] <= 5.0) / max(requested_rows, 1)
        ),
        "effective_top5_within5": float(
            np.sum(payload["top5_hit"]) / max(requested_rows, 1)
        ),
        "effective_top8_within5": float(
            np.sum(payload["top8_hit"]) / max(requested_rows, 1)
        ),
        "median_rank": float(np.median(payload["rank"])),
        "p90_rank": float(np.quantile(payload["rank"], 0.90)),
        "alias24_win_rate": (
            float(np.mean(payload["alias_win"][finite_alias])) if finite_alias.any() else np.nan
        ),
        "deep_rows": int(deep.sum()),
        "deep_top8_within5": float(np.mean(payload["top8_hit"][deep])) if deep.any() else np.nan,
        "deep_median_rank": float(np.median(payload["rank"][deep])) if deep.any() else np.nan,
    }


def _merge_payload(parts):
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


@torch.no_grad()
def _evaluate(model, samples, wells, device, mode="full", include_ncc=False):
    model.eval()
    learned_parts = []
    ncc_parts = []
    bywell_rows = []
    requested_rows_total = 0
    shuffled = list(wells[1:]) + [wells[0]]
    for well, shuffled_well in zip(wells, shuffled):
        sample = samples[well]
        query_set = sample.evaluation
        if query_set is None:
            raise RuntimeError(f"{well}: evaluation query set is missing")
        requested_rows_total += int(query_set.requested_rows)
        reference = sample.reference
        state_valid = sample.state_valid
        no_gr = mode == "no_gr"
        if mode == "shuffled_typewell":
            donor = samples[shuffled_well]
            reference = donor.reference
            state_valid = donor.state_valid
        query, reference_tensor, valid_tensor, _, _, _ = _to_tensors(
            sample,
            query_set,
            device,
            no_gr=no_gr,
            reference=reference,
            state_valid=state_valid,
        )
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(query, reference_tensor, valid_tensor)
        scores = logits.float().cpu().numpy()
        learned = _retrieval_arrays(
            scores, query_set.labels, query_set.label_index, query_set.deep_mask
        )
        learned["well"] = np.full(len(query_set.labels), well, dtype="U16")
        learned_parts.append(learned)
        well_metrics = _summarise_payload(
            learned, requested_rows=query_set.requested_rows
        )
        well_metrics.update({"well": well, "mode": mode, "eval_rows": sample.eval_rows})
        bywell_rows.append(well_metrics)

        if include_ncc:
            query_np = query_set.patches.astype(np.float32)
            if no_gr:
                query_np = query_np.copy()
                query_np[:, :, 0] = 0.0
            ncc_score = _ncc_scores(query_np, reference, state_valid)
            ncc = _retrieval_arrays(
                ncc_score, query_set.labels, query_set.label_index, query_set.deep_mask
            )
            ncc["well"] = np.full(len(query_set.labels), well, dtype="U16")
            ncc_parts.append(ncc)

    learned_payload = _merge_payload(learned_parts)
    learned_metrics = _summarise_payload(
        learned_payload, requested_rows=requested_rows_total
    )
    ncc_payload = _merge_payload(ncc_parts) if ncc_parts else None
    ncc_metrics = (
        _summarise_payload(ncc_payload, requested_rows=requested_rows_total)
        if ncc_payload is not None
        else None
    )
    return learned_metrics, pd.DataFrame(bywell_rows), learned_payload, ncc_metrics, ncc_payload


# ---------------------------------------------------------------------------
# Two-stage fold-honest training
# ---------------------------------------------------------------------------

def _train_epoch(model, samples, wells, device, optimizer, scaler, rng, epoch, phase):
    model.train()
    mean_eval_rows = float(np.mean([samples[well].eval_rows for well in wells]))
    sums = {"loss": 0.0, "ce": 0.0, "alias": 0.0, "skipped": 0}
    order = rng.permutation(len(wells))
    for step, position in enumerate(order, 1):
        well = wells[int(position)]
        sample = samples[well]
        query_set = sample.train
        if query_set is None:
            raise RuntimeError(f"{well}: train query set is missing")
        query, reference, state_valid, labels, label_index, row_weight = _to_tensors(
            sample, query_set, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(query, reference, state_valid)
            loss, ce, alias = _matcher_loss(
                logits, labels, label_index, row_weight, state_valid
            )
            loss = loss * sample.eval_rows / max(mean_eval_rows, 1.0)
        before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        sums["skipped"] += int(float(scaler.get_scale()) < before)
        sums["loss"] += float(loss.detach().cpu())
        sums["ce"] += float(ce.cpu())
        sums["alias"] += float(alias.cpu())
        if step % REPORT_EVERY_WELLS == 0 or step == len(wells):
            print(
                f"MATCHER {phase} epoch={epoch} wells={step}/{len(wells)} "
                f"loss={sums['loss']/step:.4f}",
                flush=True,
            )
    return {
        "train_loss": sums["loss"] / len(wells),
        "train_ce": sums["ce"] / len(wells),
        "train_alias": sums["alias"] / len(wells),
        "skipped_steps": sums["skipped"],
    }


def _new_training_objects(device, epochs, seed):
    _seed_everything(seed)
    model = RealCorrespondenceMatcher().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=512.0
    )
    return model, optimizer, scheduler, scaler


def _tune_epoch(samples, core_wells, tune_wells, device):
    model, optimizer, scheduler, scaler = _new_training_objects(
        device, MAX_TUNING_EPOCHS, SEED
    )
    rng = np.random.default_rng(SEED)
    history = []
    best = None
    reports_without_improvement = 0
    started = time.time()
    for epoch in range(1, MAX_TUNING_EPOCHS + 1):
        train_metrics = _train_epoch(
            model, samples, core_wells, device, optimizer, scaler, rng, epoch, "tune"
        )
        scheduler.step()
        tune_metrics, _, _, _, _ = _evaluate(
            model, samples, tune_wells, device, mode="full", include_ncc=False
        )
        row = {
            "phase": "epoch_selection",
            "epoch": epoch,
            **train_metrics,
            **{f"tune_{key}": value for key, value in tune_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("MATCHER TUNE", row, flush=True)
        objective = (
            tune_metrics["effective_top8_within5"],
            -tune_metrics["median_rank"],
            -tune_metrics["argmax_rmse"],
        )
        if best is None or objective > best["objective"]:
            best = {"epoch": epoch, "objective": objective}
            reports_without_improvement = 0
        else:
            reports_without_improvement += 1
        if epoch >= MIN_TUNING_EPOCHS and reports_without_improvement >= EARLY_STOP_REPORTS:
            break
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return int(best["epoch"]), pd.DataFrame(history)


def _fit_final(samples, train_wells, epochs, device):
    model, optimizer, scheduler, scaler = _new_training_objects(
        device, epochs, SEED + 1000
    )
    rng = np.random.default_rng(SEED + 1000)
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = _train_epoch(
            model, samples, train_wells, device, optimizer, scaler, rng, epoch, "final"
        )
        scheduler.step()
        row = {
            "phase": "frozen_final_fit",
            "epoch": epoch,
            **train_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("MATCHER FINAL", row, flush=True)
    return model, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

started = time.time()
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("This matcher gateway requires a CUDA GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_everything(SEED)

wells = _horizontal_wells()
folds, fold_path = _load_folds(wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
missing_records = [well for well in wells if not (RECORD_DIR / f"{well}.npz").exists()]
if missing_records:
    raise FileNotFoundError(
        f"Native legal-input cache covers {len(wells)-len(missing_records)}/{len(wells)} wells. "
        f"First missing: {missing_records[:10]}. Run native_resolution_dualref_capacity_v0_cell.py."
    )

print(
    {
        "experiment": EXPERIMENT,
        "device": str(device),
        "data_root": str(DATA_ROOT),
        "record_dir": str(RECORD_DIR),
        "fold_source": str(fold_path),
        "protocol": "folds3-5 train/fold2 tune -> folds2-5 refit -> fold1 once",
        "query_scales_md": QUERY_HALF_WIDTHS_MD.tolist(),
        "reference_scales_tvt": REFERENCE_HALF_WIDTHS_TVT.tolist(),
        "states": N_STATES,
    },
    flush=True,
)

samples = {}
build_rows = []
for index, well in enumerate(tqdm(wells, desc="build real-correspondence banks"), 1):
    sample = _build_well_sample(well, fold_by_well[well])
    samples[well] = sample
    build_rows.append(
        {
            "well": well,
            "fold": sample.fold,
            "train_rows": len(sample.train.labels) if sample.train is not None else 0,
            "requested_train_rows": sample.train.requested_rows if sample.train is not None else 0,
            "evaluation_rows": len(sample.evaluation.labels) if sample.evaluation is not None else 0,
            "requested_evaluation_rows": (
                sample.evaluation.requested_rows if sample.evaluation is not None else 0
            ),
            "valid_states": int(sample.state_valid.sum()),
            "label_coverage": sample.label_coverage,
            "memory_mb": (
                sample.reference.nbytes
                + (sample.train.patches.nbytes if sample.train is not None else 0)
                + (sample.evaluation.patches.nbytes if sample.evaluation is not None else 0)
            ) / 1024**2,
        }
    )
build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "bank_build_report.csv", index=False)
fold1_bank = build_report[build_report["fold"].eq(TEST_FOLD)]
fold2_bank = build_report[build_report["fold"].eq(TUNE_FOLD)]
guards = {
    "wells": len(samples),
    "minimum_label_coverage": float(build_report["label_coverage"].min()),
    "mean_label_coverage": float(build_report["label_coverage"].mean()),
    "fold1_pooled_label_coverage": float(
        fold1_bank["evaluation_rows"].sum()
        / max(fold1_bank["requested_evaluation_rows"].sum(), 1)
    ),
    "fold2_pooled_label_coverage": float(
        fold2_bank["evaluation_rows"].sum()
        / max(fold2_bank["requested_evaluation_rows"].sum(), 1)
    ),
    "minimum_valid_states": int(build_report["valid_states"].min()),
    "in_memory_bank_mb": float(build_report["memory_mb"].sum()),
}
print("BANK GUARDS", guards, flush=True)
if (
    guards["minimum_label_coverage"] < 0.90
    or guards["fold1_pooled_label_coverage"] < 0.98
    or guards["fold2_pooled_label_coverage"] < 0.98
    or guards["minimum_valid_states"] < 129
):
    raise RuntimeError(f"Real-correspondence bank guard failed: {guards}")

core_wells = folds.loc[folds["fold"].isin([3, 4, 5]), "well"].tolist()
tune_wells = folds.loc[folds["fold"].eq(TUNE_FOLD), "well"].tolist()
outer_train_wells = folds.loc[folds["fold"].ne(TEST_FOLD), "well"].tolist()
test_wells = folds.loc[folds["fold"].eq(TEST_FOLD), "well"].tolist()
if (
    len(outer_train_wells) != 618
    or len(test_wells) != 155
    or len(core_wells) + len(tune_wells) != len(outer_train_wells)
    or set(core_wells).intersection(tune_wells)
):
    raise RuntimeError(
        "Unexpected fold sizes: "
        + repr((len(core_wells), len(tune_wells), len(outer_train_wells), len(test_wells)))
    )

selected_epoch, tune_history = _tune_epoch(
    samples, core_wells, tune_wells, device
)
tune_history.to_csv(OUT_DIR / "epoch_selection_history.csv", index=False)
print("FROZEN EPOCH", selected_epoch, flush=True)

model, final_history = _fit_final(
    samples, outer_train_wells, selected_epoch, device
)
final_history.to_csv(OUT_DIR / "final_fit_history.csv", index=False)
checkpoint_path = MODEL_DIR / "fold1_gateway_matcher.pt"
torch.save(
    {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "selected_epoch": selected_epoch,
        "query_half_widths_md": QUERY_HALF_WIDTHS_MD,
        "reference_half_widths_tvt": REFERENCE_HALF_WIDTHS_TVT,
        "state_offsets": STATE_OFFSETS,
        "fold_source": str(fold_path),
    },
    checkpoint_path,
)

full_metrics, full_bywell, full_payload, ncc_metrics, ncc_payload = _evaluate(
    model, samples, test_wells, device, mode="full", include_ncc=True
)
no_gr_metrics, no_gr_bywell, no_gr_payload, _, _ = _evaluate(
    model, samples, test_wells, device, mode="no_gr", include_ncc=False
)
shuffle_metrics, shuffle_bywell, shuffle_payload, _, _ = _evaluate(
    model, samples, test_wells, device, mode="shuffled_typewell", include_ncc=False
)

full_bywell.to_csv(OUT_DIR / "fold1_bywell_full.csv", index=False)
no_gr_bywell.to_csv(OUT_DIR / "fold1_bywell_no_gr.csv", index=False)
shuffle_bywell.to_csv(OUT_DIR / "fold1_bywell_shuffled_typewell.csv", index=False)
np.savez_compressed(OUT_DIR / "fold1_retrieval_full.npz", **full_payload)
np.savez_compressed(OUT_DIR / "fold1_retrieval_ncc.npz", **ncc_payload)

comparison = pd.DataFrame(
    [
        {"method": "learned_full", **full_metrics},
        {"method": "ncc", **ncc_metrics},
        {"method": "learned_no_gr", **no_gr_metrics},
        {"method": "learned_shuffled_typewell", **shuffle_metrics},
    ]
)
comparison.to_csv(OUT_DIR / "fold1_method_comparison.csv", index=False)

absolute_pass = (
    full_metrics["effective_top8_within5"] >= TOP8_WITHIN5_ABSOLUTE_GATE
)
ncc_gain = (
    full_metrics["effective_top8_within5"]
    - ncc_metrics["effective_top8_within5"]
)
ncc_pass = ncc_gain >= TOP8_GAIN_VS_NCC_GATE
rank_pass = full_metrics["median_rank"] <= MEDIAN_RANK_GATE
alias_pass = full_metrics["alias24_win_rate"] >= ALIAS_WIN_GATE
no_gr_drop = (
    full_metrics["effective_top8_within5"]
    - no_gr_metrics["effective_top8_within5"]
)
shuffle_drop = (
    full_metrics["effective_top8_within5"]
    - shuffle_metrics["effective_top8_within5"]
)
causal_pass = (
    no_gr_drop >= CONTROL_TOP8_DROP_GATE
    and shuffle_drop >= CONTROL_TOP8_DROP_GATE
)
strict_pass = bool(absolute_pass and ncc_pass and rank_pass and alias_pass and causal_pass)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold1_real_correspondence_retrieval_gateway",
    "sample_unit": "complete physical well; sampled rows only within each well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "record_prefix_calibration": "visible TVT_input rows only",
    "fold_source": str(fold_path),
    "protocol": {
        "epoch_selection_train_folds": [3, 4, 5],
        "epoch_selection_valid_fold": TUNE_FOLD,
        "final_train_folds": [2, 3, 4, 5],
        "untouched_test_fold": TEST_FOLD,
        "selected_epoch": selected_epoch,
    },
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "guards": guards,
    "learned_full": full_metrics,
    "ncc": ncc_metrics,
    "no_gr": no_gr_metrics,
    "shuffled_typewell": shuffle_metrics,
    "diagnostics": {
        "top8_gain_vs_ncc": ncc_gain,
        "top8_drop_without_gr": no_gr_drop,
        "top8_drop_shuffled_typewell": shuffle_drop,
    },
    "gates": {
        "effective_top8_within5_ge": TOP8_WITHIN5_ABSOLUTE_GATE,
        "effective_top8_gain_vs_ncc_ge": TOP8_GAIN_VS_NCC_GATE,
        "median_rank_le": MEDIAN_RANK_GATE,
        "alias24_win_rate_ge": ALIAS_WIN_GATE,
        "each_control_top8_drop_ge": CONTROL_TOP8_DROP_GATE,
    },
    "absolute_pass": bool(absolute_pass),
    "ncc_pass": bool(ncc_pass),
    "rank_pass": bool(rank_pass),
    "alias_pass": bool(alias_pass),
    "causal_pass": bool(causal_pass),
    "strict_pass": strict_pass,
    "checkpoint": str(checkpoint_path),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze matcher and build per-well PCA/B-spline registration"
        if strict_pass
        else "FAIL -> do not build a trajectory solver; inspect scale-pair retrieval and aliases"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nFOLD1 METHOD COMPARISON")
print(comparison.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
