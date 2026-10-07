"""Native-resolution, complete-well dual-reference capacity gateway.

This is the first stage of the standalone <5.5 ft route.  It deliberately
uses no neighbour well, PF path, tabular prediction, CP12, Gold, or submission
feedback.  Every sample is one complete well and every input is available at
test time:

* the full native-resolution horizontal MD/X/Y/Z/GR sequence;
* the visible TVT_input heel prefix and its horizontal GR;
* the assigned typewell TVT/GR curve.

Unlike the earlier CorrNet/CRF experiments, horizontal GR is encoded at its
native ~1 ft spacing before any stride is applied.  The model builds a learned
TVT-drift cost volume from two references (typewell and the known heel), then
decodes one complete path.  The state axis is

    TVT - last_known_TVT in [-128, 128] ft at 1 ft spacing.

This cell is only a representation/capacity audit.  It trains nested 1, 4 and
24-well overfit scenarios.  A pass authorises a held-out fold experiment; it
is not a CV claim.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "native_resolution_dualref_capacity_v0"
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
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "records"
MODEL_DIR = OUT_DIR / "models"

SEED = 20260718
REQUIRE_CUDA = True
N_CAPACITY_WELLS = 24

STATE_HALF_WIDTH_FT = 128.0
STATE_STEP_FT = 1.0
STATE_OFFSETS = np.arange(
    -STATE_HALF_WIDTH_FT,
    STATE_HALF_WIDTH_FT + 0.5 * STATE_STEP_FT,
    STATE_STEP_FT,
    dtype=np.float32,
)
N_STATES = len(STATE_OFFSETS)
REFERENCE_CONTEXT_FT = 64.0
REFERENCE_OFFSETS = np.arange(
    -STATE_HALF_WIDTH_FT - REFERENCE_CONTEXT_FT,
    STATE_HALF_WIDTH_FT + REFERENCE_CONTEXT_FT + 0.5,
    1.0,
    dtype=np.float32,
)
REFERENCE_STATE_START = int(round(REFERENCE_CONTEXT_FT))

HORIZONTAL_STRIDE = 4
TARGET_SIGMA_FT = 1.5
EMBEDDING = 24
CONDITION_CHANNELS = 16
COST_CHANNELS = 25
AXIAL_BLOCKS = 6
BASE_CHANNELS = 24

LEARNING_RATE = 4e-4
MIN_LEARNING_RATE = 4e-5
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 5.0

LOSS_CE = 1.0
LOSS_OFFSET = 0.45
LOSS_SLOPE = 0.08
LOSS_CURVATURE = 0.015

SCENARIOS = (
    {
        "name": "one_well",
        "wells": 1,
        "epochs": 450,
        "pooled_gate": 0.50,
        "p95_gate": 0.75,
    },
    {
        "name": "four_wells",
        "wells": 4,
        "epochs": 550,
        "pooled_gate": 0.75,
        "p95_gate": 1.25,
    },
    {
        "name": "twentyfour_wells",
        "wells": 24,
        "epochs": 700,
        "pooled_gate": 1.00,
        "p95_gate": 2.00,
    },
)
REPORT_EVERY = 20
EARLY_SUCCESS_FRACTION = 0.90


# ---------------------------------------------------------------------------
# Utilities
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


def _robust_location_scale(values) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) < 8:
        return 0.0, 1.0
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    return centre, scale


def _normalise(values, centre, scale):
    output = (np.asarray(values, np.float64) - float(centre)) / float(scale)
    return np.clip(output, -8.0, 8.0)


def _interp_finite(x, y, query, left=np.nan, right=np.nan):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    xv = x[valid]
    yv = y[valid]
    order = np.argsort(xv, kind="mergesort")
    xv = xv[order]
    yv = yv[order]
    unique, inverse = np.unique(xv, return_inverse=True)
    if len(unique) != len(xv):
        sums = np.bincount(inverse, weights=yv)
        counts = np.bincount(inverse)
        xv = unique
        yv = sums / np.maximum(counts, 1)
    return np.interp(query, xv, yv, left=left, right=right)


def _nearest_distance(x, query):
    x = np.sort(np.asarray(x, np.float64)[np.isfinite(x)])
    query = np.asarray(query, np.float64)
    if not len(x):
        return np.full(query.shape, np.inf, np.float64)
    positions = np.searchsorted(x, query)
    left = np.abs(query - x[np.clip(positions - 1, 0, len(x) - 1)])
    right = np.abs(query - x[np.clip(positions, 0, len(x) - 1)])
    return np.minimum(left, right)


def _fill_1d(x, values):
    x = np.asarray(x, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.isfinite(x) & np.isfinite(values)
    if int(valid.sum()) < 2:
        return np.zeros(values.shape, np.float64)
    return np.interp(x, x[valid], values[valid])


def _gradient(values, x):
    values = np.asarray(values, np.float64)
    x = np.asarray(x, np.float64)
    if len(values) < 3:
        return np.zeros(values.shape, np.float64)
    dx = np.gradient(x)
    dx = np.where(np.abs(dx) < 1e-6, 1.0, dx)
    return np.gradient(values) / dx


def _robust_affine(source, target) -> tuple[float, float, int, float]:
    """Fit target ~= a * source + b using iterative residual clipping."""

    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    valid = np.isfinite(source) & np.isfinite(target)
    x = source[valid]
    y = target[valid]
    if len(x) < 32:
        shift = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, shift, int(len(x)), np.nan
    keep = np.ones(len(x), dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(4):
        design = np.c_[x[keep], np.ones(int(keep.sum()))]
        a, b = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (a * x + b)
        centre, scale = _robust_location_scale(residual[keep])
        new_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if int(new_keep.sum()) < 32 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    residual = y[keep] - (a * x[keep] + b)
    return float(a), float(b), int(keep.sum()), float(_rmse(y[keep], a * x[keep] + b))


def _reference_features(values, valid):
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, np.float64)
    filled = values.copy()
    finite = np.isfinite(filled) & (valid > 0.5)
    if int(finite.sum()) >= 2:
        positions = np.arange(len(filled), dtype=np.float64)
        filled = np.interp(positions, positions[finite], filled[finite])
    else:
        filled[:] = 0.0
    filled = np.nan_to_num(filled, nan=0.0)
    d1 = np.gradient(filled)
    d2 = np.gradient(d1)
    return np.stack(
        [
            np.clip(filled, -8.0, 8.0),
            valid,
            np.clip(d1, -4.0, 4.0),
            np.clip(d2, -4.0, 4.0),
        ],
        axis=0,
    ).astype(np.float32)


def _heel_reference(known_tvt, known_gr, query_tvt, radius=0.75):
    known_tvt = np.asarray(known_tvt, np.float64)
    known_gr = np.asarray(known_gr, np.float64)
    query_tvt = np.asarray(query_tvt, np.float64)
    valid = np.isfinite(known_tvt) & np.isfinite(known_gr)
    tvt = known_tvt[valid]
    gr = known_gr[valid]
    if len(tvt) < 8:
        return np.full(query_tvt.shape, np.nan), np.zeros(query_tvt.shape)
    order = np.argsort(tvt, kind="mergesort")
    tvt = tvt[order]
    gr = gr[order]
    output = np.full(query_tvt.shape, np.nan, np.float64)
    available = np.zeros(query_tvt.shape, np.float64)
    for index, query in enumerate(query_tvt):
        lo = np.searchsorted(tvt, query - radius, side="left")
        hi = np.searchsorted(tvt, query + radius, side="right")
        if hi > lo:
            output[index] = float(np.median(gr[lo:hi]))
            available[index] = 1.0
    return output, available


# ---------------------------------------------------------------------------
# Input audit and capacity-well selection
# ---------------------------------------------------------------------------


def _horizontal_paths():
    paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
    if len(paths) < 700:
        raise FileNotFoundError(
            f"Expected about 773 train horizontal files under {DATA_ROOT / 'train'}; found {len(paths)}"
        )
    return paths


def _scan_well(path: Path):
    frame = pd.read_csv(path, usecols=["TVT_input", "TVT", "GR"])
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
    truth = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    contiguous = bool(
        len(known)
        and len(hidden)
        and known[-1] + 1 == hidden[0]
        and np.all(np.diff(known) == 1)
    )
    return {
        "well": path.name.split("__", 1)[0],
        "rows": len(frame),
        "known_rows": len(known),
        "eval_rows": len(hidden),
        "known_fraction": len(known) / max(len(frame), 1),
        "gr_missing_fraction": float(np.mean(~np.isfinite(gr))),
        "contiguous_mask": contiguous,
        "target_finite_fraction": float(np.mean(np.isfinite(truth[hidden]))) if len(hidden) else 0.0,
    }


def _farthest_capacity_wells(metadata: pd.DataFrame, count: int):
    columns = ["known_fraction", "eval_rows", "gr_missing_fraction", "rows"]
    values = metadata[columns].to_numpy(np.float64)
    lo = np.quantile(values, 0.05, axis=0)
    hi = np.quantile(values, 0.95, axis=0)
    scaled = np.clip((values - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    centre = np.full(len(columns), 0.5)
    selected = [int(np.argmin(np.sum((scaled - centre) ** 2, axis=1)))]
    while len(selected) < count:
        distance = np.min(
            np.sum((scaled[:, None, :] - scaled[selected][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return metadata.iloc[selected].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Native-resolution record construction
# ---------------------------------------------------------------------------


@dataclass
class Record:
    well: str
    horizontal: np.ndarray
    condition: np.ndarray
    typewell: np.ndarray
    heel: np.ndarray
    latent_index: np.ndarray
    latent_truth_offset: np.ndarray
    latent_eval_mask: np.ndarray
    md: np.ndarray
    eval_index: np.ndarray
    truth: np.ndarray
    anchor_tvt: float
    anchor_index: int
    eval_rows: int
    calibration_a: float
    calibration_b: float
    calibration_pairs: int
    calibration_rmse: float
    teacher_rmse: float
    axis_coverage: float


def _record_path(well):
    return CACHE_DIR / f"{well}.npz"


def _save_record(record: Record) -> None:
    np.savez_compressed(
        _record_path(record.well),
        well=np.asarray(record.well),
        horizontal=record.horizontal.astype(np.float16),
        condition=record.condition.astype(np.float16),
        typewell=record.typewell.astype(np.float16),
        heel=record.heel.astype(np.float16),
        latent_index=record.latent_index.astype(np.int32),
        latent_truth_offset=record.latent_truth_offset.astype(np.float32),
        latent_eval_mask=record.latent_eval_mask.astype(np.uint8),
        md=record.md.astype(np.float32),
        eval_index=record.eval_index.astype(np.int32),
        truth=record.truth.astype(np.float32),
        anchor_tvt=np.asarray([record.anchor_tvt], np.float32),
        anchor_index=np.asarray([record.anchor_index], np.int32),
        eval_rows=np.asarray([record.eval_rows], np.int32),
        calibration=np.asarray(
            [record.calibration_a, record.calibration_b, record.calibration_pairs, record.calibration_rmse],
            np.float32,
        ),
        teacher_rmse=np.asarray([record.teacher_rmse], np.float32),
        axis_coverage=np.asarray([record.axis_coverage], np.float32),
        cache_version=np.asarray([1], np.int16),
    )


def _load_record(well) -> Record:
    with np.load(_record_path(well), allow_pickle=False) as data:
        calibration = data["calibration"].astype(np.float64)
        return Record(
            well=str(data["well"].item()),
            horizontal=data["horizontal"].astype(np.float32),
            condition=data["condition"].astype(np.float32),
            typewell=data["typewell"].astype(np.float32),
            heel=data["heel"].astype(np.float32),
            latent_index=data["latent_index"].astype(np.int32),
            latent_truth_offset=data["latent_truth_offset"].astype(np.float32),
            latent_eval_mask=data["latent_eval_mask"].astype(bool),
            md=data["md"].astype(np.float32),
            eval_index=data["eval_index"].astype(np.int32),
            truth=data["truth"].astype(np.float32),
            anchor_tvt=float(data["anchor_tvt"][0]),
            anchor_index=int(data["anchor_index"][0]),
            eval_rows=int(data["eval_rows"][0]),
            calibration_a=float(calibration[0]),
            calibration_b=float(calibration[1]),
            calibration_pairs=int(round(calibration[2])),
            calibration_rmse=float(calibration[3]),
            teacher_rmse=float(data["teacher_rmse"][0]),
            axis_coverage=float(data["axis_coverage"][0]),
        )


def _build_record(well: str) -> Record:
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)

    required_h = {"MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"}
    required_t = {"TVT", "GR"}
    if not required_h.issubset(horizontal.columns):
        raise RuntimeError(f"{well}: horizontal misses {sorted(required_h - set(horizontal.columns))}")
    if not required_t.issubset(typewell.columns):
        raise RuntimeError(f"{well}: typewell misses TVT/GR")

    def numeric(frame, column):
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)

    md = numeric(horizontal, "MD")
    x = numeric(horizontal, "X")
    y = numeric(horizontal, "Y")
    z = numeric(horizontal, "Z")
    gr = numeric(horizontal, "GR")
    truth = numeric(horizontal, "TVT")
    tvt_input = numeric(horizontal, "TVT_input")
    tw_tvt = numeric(typewell, "TVT")
    tw_gr = numeric(typewell, "GR")

    finite_md = np.isfinite(md)
    if not finite_md.all() or not np.all(np.diff(md) > 0):
        raise RuntimeError(f"{well}: MD must be finite and strictly increasing")
    known = np.flatnonzero(np.isfinite(tvt_input))
    eval_index = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    if len(known) < 64 or len(eval_index) < 128 or known[-1] + 1 != eval_index[0]:
        raise RuntimeError(f"{well}: invalid official visible-prefix mask")
    anchor_index = int(known[-1])
    anchor_tvt = float(tvt_input[anchor_index])
    if not np.isfinite(anchor_tvt):
        raise RuntimeError(f"{well}: non-finite anchor")

    # Calibrate the typewell GR sensor to the horizontal prefix.  This uses
    # only visible TVT_input rows and is therefore legal at deployment.
    tw_at_prefix = _interp_finite(tw_tvt, tw_gr, tvt_input[known])
    a, b, pairs, calibration_rmse = _robust_affine(tw_at_prefix, gr[known])
    tw_gr_calibrated = a * tw_gr + b

    gr_centre, gr_scale = _robust_location_scale(gr)
    gr_normalised = _normalise(gr, gr_centre, gr_scale)
    tw_normalised = _normalise(tw_gr_calibrated, gr_centre, gr_scale)

    gr_valid = np.isfinite(gr).astype(np.float64)
    gr_filled = _fill_1d(md, gr_normalised)
    gr_d1 = _gradient(gr_filled, md)
    gr_d2 = _gradient(gr_d1, md)
    horizontal_features = np.stack(
        [
            np.clip(gr_filled, -8.0, 8.0),
            gr_valid,
            np.clip(gr_d1, -4.0, 4.0),
            np.clip(gr_d2, -4.0, 4.0),
        ],
        axis=0,
    ).astype(np.float32)

    dx = _gradient(_fill_1d(md, x), md)
    dy = _gradient(_fill_1d(md, y), md)
    dz = _gradient(_fill_1d(md, z), md)
    d2z = _gradient(dz, md)
    progress = (md - md[0]) / max(md[-1] - md[0], 1.0)
    ps_progress = (md - md[anchor_index]) / max(md[-1] - md[anchor_index], 1.0)
    z_relative = (_fill_1d(md, z) - z[anchor_index]) / 512.0
    known_mask = np.isfinite(tvt_input).astype(np.float64)
    known_offset = np.where(known_mask > 0, (tvt_input - anchor_tvt) / STATE_HALF_WIDTH_FT, 0.0)
    condition = np.stack(
        [
            known_mask,
            np.clip(known_offset, -2.0, 2.0),
            np.clip(progress, 0.0, 1.0),
            np.clip(ps_progress, -1.0, 1.0),
            np.clip(z_relative, -4.0, 4.0),
            np.clip(dx, -2.0, 2.0),
            np.clip(dy, -2.0, 2.0),
            np.clip(dz, -2.0, 2.0),
            np.clip(d2z * 32.0, -2.0, 2.0),
            gr_valid,
        ],
        axis=0,
    ).astype(np.float32)

    reference_tvt = anchor_tvt + REFERENCE_OFFSETS
    typewell_values = _interp_finite(tw_tvt, tw_normalised, reference_tvt)
    typewell_valid = (
        np.isfinite(typewell_values)
        & (_nearest_distance(tw_tvt[np.isfinite(tw_gr)], reference_tvt) <= 1.25)
    ).astype(np.float64)
    typewell_features = _reference_features(typewell_values, typewell_valid)

    known_gr_normalised = gr_normalised[known]
    heel_values, heel_valid = _heel_reference(
        tvt_input[known], known_gr_normalised, reference_tvt
    )
    heel_features = _reference_features(heel_values, heel_valid)

    latent_index = np.arange(0, len(md), HORIZONTAL_STRIDE, dtype=np.int32)
    latent_truth = truth[latent_index] - anchor_tvt
    latent_eval_mask = (
        (latent_index > anchor_index)
        & np.isfinite(latent_truth)
    )
    if int(latent_eval_mask.sum()) < 64:
        raise RuntimeError(f"{well}: too few latent eval columns")

    eval_offset = truth[eval_index] - anchor_tvt
    axis_coverage = float(np.mean(np.abs(eval_offset) <= STATE_HALF_WIDTH_FT))
    latent_eval_index = latent_index[latent_eval_mask]
    teacher_md = np.r_[md[anchor_index], md[latent_eval_index]]
    teacher_tvt = np.r_[anchor_tvt, truth[latent_eval_index]]
    teacher_prediction = np.interp(md[eval_index], teacher_md, teacher_tvt)
    teacher_rmse = _rmse(truth[eval_index], teacher_prediction)

    return Record(
        well=well,
        horizontal=horizontal_features,
        condition=condition,
        typewell=typewell_features,
        heel=heel_features,
        latent_index=latent_index,
        latent_truth_offset=latent_truth.astype(np.float32),
        latent_eval_mask=latent_eval_mask,
        md=md.astype(np.float32),
        eval_index=eval_index.astype(np.int32),
        truth=truth.astype(np.float32),
        anchor_tvt=anchor_tvt,
        anchor_index=anchor_index,
        eval_rows=len(eval_index),
        calibration_a=a,
        calibration_b=b,
        calibration_pairs=pairs,
        calibration_rmse=calibration_rmse,
        teacher_rmse=teacher_rmse,
        axis_coverage=axis_coverage,
    )


# ---------------------------------------------------------------------------
# Native-resolution dual-reference network
# ---------------------------------------------------------------------------


class Residual1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class SharedGREncoder(nn.Module):
    def __init__(self, embedding=EMBEDDING):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(4, embedding, 9, padding=4),
            nn.GroupNorm(4, embedding),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            Residual1D(embedding, 1),
            Residual1D(embedding, 2),
            Residual1D(embedding, 4),
            Residual1D(embedding, 8),
        )

    def forward(self, values):
        return self.blocks(self.input(values))


class ConditionEncoder(nn.Module):
    def __init__(self, input_channels=10, output_channels=CONDITION_CHANNELS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 24, 7, padding=3),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 24, 5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, output_channels, 5, stride=2, padding=2),
            nn.GroupNorm(4, output_channels),
            nn.GELU(),
        )

    def forward(self, values):
        return self.network(values)


class QueryDownsample(nn.Module):
    def __init__(self, channels=EMBEDDING):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, stride=2, padding=2),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, stride=2, padding=2),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )

    def forward(self, values):
        return self.network(values)


class AxialResidual(nn.Module):
    def __init__(self, channels, temporal_dilation):
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
                (1, 9),
                padding=(0, 4 * temporal_dilation),
                dilation=(1, temporal_dilation),
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.time(self.state(values)))


class NativeDualReferenceWarpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gr_encoder = SharedGREncoder()
        self.query_downsample = QueryDownsample()
        self.condition_encoder = ConditionEncoder()

        # 2 correlations, 2 absolute-difference channels, 3 validity channels,
        # 1 state coordinate, 1 progress coordinate and 16 condition channels.
        input_channels = 2 + 2 + 3 + 1 + 1 + CONDITION_CHANNELS
        if input_channels != COST_CHANNELS:
            raise RuntimeError(
                f"cost channel config mismatch: calculated={input_channels}, expected={COST_CHANNELS}"
            )
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, BASE_CHANNELS, 1),
            nn.GroupNorm(4, BASE_CHANNELS),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8, 16, 32][:AXIAL_BLOCKS]
        self.blocks = nn.Sequential(
            *[AxialResidual(BASE_CHANNELS, dilation) for dilation in dilations]
        )
        self.output = nn.Conv2d(BASE_CHANNELS, 1, 1)
        self.correlation_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, horizontal, condition, typewell, heel):
        horizontal_embedding_full = self.gr_encoder(horizontal)
        query = self.query_downsample(horizontal_embedding_full)
        condition_embedding = self.condition_encoder(condition)
        columns = min(query.shape[-1], condition_embedding.shape[-1])
        query = query[..., :columns]
        condition_embedding = condition_embedding[..., :columns]

        type_embedding_full = self.gr_encoder(typewell)
        heel_embedding_full = self.gr_encoder(heel)
        state_slice = slice(REFERENCE_STATE_START, REFERENCE_STATE_START + N_STATES)
        type_key = type_embedding_full[..., state_slice]
        heel_key = heel_embedding_full[..., state_slice]

        query_normalised = F.normalize(query, dim=1, eps=1e-5)
        type_normalised = F.normalize(type_key, dim=1, eps=1e-5)
        heel_normalised = F.normalize(heel_key, dim=1, eps=1e-5)
        scale = F.softplus(self.correlation_scale)
        type_correlation = scale * torch.einsum(
            "bct,bcs->bst", query_normalised, type_normalised
        )
        heel_correlation = scale * torch.einsum(
            "bct,bcs->bst", query_normalised, heel_normalised
        )

        horizontal_value = horizontal[:, 0:1, ::HORIZONTAL_STRIDE][..., :columns]
        horizontal_valid = horizontal[:, 1:2, ::HORIZONTAL_STRIDE][..., :columns]
        type_value = typewell[:, 0:1, state_slice]
        heel_value = heel[:, 0:1, state_slice]
        type_valid = typewell[:, 1:2, state_slice]
        heel_valid = heel[:, 1:2, state_slice]

        h_value = horizontal_value.unsqueeze(2)
        type_value_volume = type_value.unsqueeze(-1)
        heel_value_volume = heel_value.unsqueeze(-1)
        type_difference = -torch.abs(h_value - type_value_volume)
        heel_difference = -torch.abs(h_value - heel_value_volume)

        states = N_STATES
        h_valid = horizontal_valid.unsqueeze(2).expand(-1, -1, states, -1)
        type_valid_volume = type_valid.unsqueeze(-1).expand(-1, -1, -1, columns)
        heel_valid_volume = heel_valid.unsqueeze(-1).expand(-1, -1, -1, columns)
        state_coordinate = torch.linspace(
            -1.0, 1.0, states, device=query.device, dtype=query.dtype
        )[None, None, :, None].expand(query.shape[0], -1, -1, columns)
        progress = torch.linspace(
            0.0, 1.0, columns, device=query.device, dtype=query.dtype
        )[None, None, None, :].expand(query.shape[0], -1, states, -1)
        condition_volume = condition_embedding.unsqueeze(2).expand(-1, -1, states, -1)

        cost = torch.cat(
            [
                type_correlation[:, None],
                heel_correlation[:, None],
                type_difference,
                heel_difference,
                h_valid,
                type_valid_volume,
                heel_valid_volume,
                state_coordinate,
                progress,
                condition_volume,
            ],
            dim=1,
        )
        logits = self.output(self.blocks(self.input(cost))).squeeze(1)
        return logits


# ---------------------------------------------------------------------------
# Loss, decoding and capacity training
# ---------------------------------------------------------------------------


def _tensor_record(record: Record, device, mode="full"):
    horizontal = record.horizontal.copy()
    typewell = record.typewell.copy()
    heel = record.heel.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    return (
        torch.from_numpy(horizontal)[None].to(device),
        torch.from_numpy(record.condition)[None].to(device),
        torch.from_numpy(typewell)[None].to(device),
        torch.from_numpy(heel)[None].to(device),
    )


def _alignment_loss(logits, record: Record, device, row_weight):
    columns = logits.shape[-1]
    target = torch.from_numpy(record.latent_truth_offset[:columns]).to(device)
    mask = torch.from_numpy(record.latent_eval_mask[:columns]).to(device)
    if int(mask.sum()) < 8:
        raise RuntimeError(f"{record.well}: too few supervised latent columns")
    state = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    target_eval = target[mask][None, :]
    target_distribution = torch.exp(
        -0.5 * ((state - target_eval) / TARGET_SIGMA_FT) ** 2
    )
    target_distribution = target_distribution / target_distribution.sum(dim=0, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(logits[0, :, mask], dim=0)
    ce = -(target_distribution * log_probability).sum(dim=0).mean()

    probability = F.softmax(logits[0, :, mask], dim=0)
    expected = (probability * state).sum(dim=0)
    offset = F.smooth_l1_loss(expected / 8.0, target[mask] / 8.0)

    anchor = torch.zeros(1, device=device, dtype=expected.dtype)
    expected_with_anchor = torch.cat([anchor, expected])
    target_with_anchor = torch.cat([anchor, target[mask]])
    slope = F.smooth_l1_loss(
        torch.diff(expected_with_anchor) / 2.0,
        torch.diff(target_with_anchor) / 2.0,
    )
    if len(expected_with_anchor) >= 3:
        curvature = F.smooth_l1_loss(
            torch.diff(expected_with_anchor, n=2) / 2.0,
            torch.diff(target_with_anchor, n=2) / 2.0,
        )
    else:
        curvature = torch.zeros((), device=device)

    loss = row_weight * (
        LOSS_CE * ce
        + LOSS_OFFSET * offset
        + LOSS_SLOPE * slope
        + LOSS_CURVATURE * curvature
    )
    return loss, ce.detach(), offset.detach(), expected.detach()


@torch.no_grad()
def _predict_record(model, record: Record, device, mode="full"):
    model.eval()
    horizontal, condition, typewell, heel = _tensor_record(record, device, mode=mode)
    amp_enabled = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        logits = model(horizontal, condition, typewell, heel)
    probability = F.softmax(logits[0].float(), dim=0)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    expected = (probability * states).sum(dim=0).cpu().numpy()

    columns = min(len(expected), len(record.latent_index))
    latent_index = record.latent_index[:columns]
    latent_eval = record.latent_eval_mask[:columns]
    eval_latent_index = latent_index[latent_eval]
    eval_expected = expected[:columns][latent_eval]
    if len(eval_latent_index) < 4:
        raise RuntimeError(f"{record.well}: prediction has too few eval columns")
    support_md = np.r_[record.md[record.anchor_index], record.md[eval_latent_index]]
    support_tvt = np.r_[record.anchor_tvt, record.anchor_tvt + eval_expected]
    prediction = np.interp(record.md[record.eval_index], support_md, support_tvt)
    truth = record.truth[record.eval_index]
    return truth.astype(np.float64), prediction.astype(np.float64)


@torch.no_grad()
def _evaluate(model, records, device, mode="full"):
    rows = []
    truths = []
    predictions = []
    for record in records:
        truth, prediction = _predict_record(model, record, device, mode=mode)
        truths.append(truth)
        predictions.append(prediction)
        rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "rmse": _rmse(truth, prediction),
                "mae": float(np.mean(np.abs(prediction - truth))),
            }
        )
    bywell = pd.DataFrame(rows)
    pooled_truth = np.concatenate(truths)
    pooled_prediction = np.concatenate(predictions)
    metrics = {
        "rows": len(pooled_truth),
        "wells": len(records),
        "pooled_rmse": _rmse(pooled_truth, pooled_prediction),
        "pooled_mae": float(np.mean(np.abs(pooled_prediction - pooled_truth))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
    }
    return metrics, bywell


def _train_scenario(config, records, device):
    name = config["name"]
    scenario_records = records[: int(config["wells"])]
    row_mean = float(np.mean([record.eval_rows for record in scenario_records]))
    row_weights = {
        record.well: record.eval_rows / max(row_mean, 1.0)
        for record in scenario_records
    }

    _seed_everything(SEED + len(scenario_records))
    model = NativeDualReferenceWarpNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=MIN_LEARNING_RATE,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    rng = np.random.default_rng(SEED + len(scenario_records))

    best = None
    history = []
    started = time.time()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        order = rng.permutation(len(scenario_records))
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_offset = 0.0
        for index in order:
            record = scenario_records[int(index)]
            horizontal, condition, typewell, heel = _tensor_record(record, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(horizontal, condition, typewell, heel)
                loss, ce, offset, _ = _alignment_loss(
                    logits,
                    record,
                    device,
                    row_weights[record.well],
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            epoch_ce += float(ce.cpu())
            epoch_offset += float(offset.cpu())
            del horizontal, condition, typewell, heel, logits, loss
        scheduler.step()

        should_report = epoch == 1 or epoch % REPORT_EVERY == 0 or epoch == int(config["epochs"])
        if not should_report:
            continue
        metrics, bywell = _evaluate(model, scenario_records, device)
        row = {
            "scenario": name,
            "epoch": epoch,
            "train_loss": epoch_loss / len(scenario_records),
            "train_ce": epoch_ce / len(scenario_records),
            "train_offset": epoch_offset / len(scenario_records),
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "correlation_scale": float(F.softplus(model.correlation_scale).detach().cpu()),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("CAPACITY EPOCH", row, flush=True)
        if best is None or metrics["pooled_rmse"] < best["metrics"]["pooled_rmse"]:
            best = {
                "epoch": epoch,
                "metrics": metrics,
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "bywell": bywell.copy(),
            }
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "scenario": name,
                    "epoch": epoch,
                    "metrics": metrics,
                    "model": best["state"],
                    "state_offsets": STATE_OFFSETS,
                    "configuration": {
                        "horizontal_stride": HORIZONTAL_STRIDE,
                        "embedding": EMBEDDING,
                        "base_channels": BASE_CHANNELS,
                        "reference_context_ft": REFERENCE_CONTEXT_FT,
                    },
                },
                MODEL_DIR / f"{name}.pt",
            )
        success = (
            metrics["pooled_rmse"] <= float(config["pooled_gate"]) * EARLY_SUCCESS_FRACTION
            and metrics["p95_well_rmse"] <= float(config["p95_gate"]) * EARLY_SUCCESS_FRACTION
        )
        if success and epoch >= 80:
            print(f"{name}: early capacity gate cleared", flush=True)
            break

    if best is None:
        raise RuntimeError(f"{name}: no checkpoint was evaluated")
    model.load_state_dict(best["state"])
    full_metrics, full_bywell = _evaluate(model, scenario_records, device)
    no_gr_metrics, _ = _evaluate(model, scenario_records, device, mode="no_gr")
    full_bywell.to_csv(OUT_DIR / f"{name}_bywell.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / f"{name}_history.csv", index=False)
    result = {
        "scenario": name,
        "wells": len(scenario_records),
        "best_epoch": int(best["epoch"]),
        **full_metrics,
        "no_gr_rmse": no_gr_metrics["pooled_rmse"],
        "no_gr_degradation": no_gr_metrics["pooled_rmse"] - full_metrics["pooled_rmse"],
        "pooled_gate": float(config["pooled_gate"]),
        "p95_gate": float(config["p95_gate"]),
        "pass": bool(
            full_metrics["pooled_rmse"] <= float(config["pooled_gate"])
            and full_metrics["p95_well_rmse"] <= float(config["p95_gate"])
        ),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


"""Fold-2 gateway for a standalone complete-well surface posterior model.

This addon is appended to the definition prefix of
native_resolution_dualref_capacity_v0_cell.py by the local build script.  The
resulting merged cell is self-contained on AutoDL and consumes only raw train
horizontal/typewell CSV files plus fold-clean well metadata.

The native dual-reference network produces a 257-state TVT posterior at every
latent MD column.  A bidirectional complete-well decoder then consumes a
low-rank view of the entire posterior, geometry and prefix channels.  It is
trained jointly on absolute TVT offset, structural surface (TVT + Z), local
slope, and 32/128/512-ft displacement targets.
"""

from pathlib import Path
import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_FWSP = "native_fullwell_surface_posterior_fold2_v0"
OUT_DIR_FWSP = OOF_ROOT / EXPERIMENT_FWSP
MODEL_DIR_FWSP = OUT_DIR_FWSP / "models"
OUT_DIR_FWSP.mkdir(parents=True, exist_ok=True)
MODEL_DIR_FWSP.mkdir(parents=True, exist_ok=True)

SEED_FWSP = 20260724
TRAIN_FOLDS_FWSP = (3, 4, 5)
VALID_FOLD_FWSP = 2
MAX_EPOCHS_FWSP = 36
MIN_EPOCHS_FWSP = 10
REPORT_EVERY_FWSP = 2
PATIENCE_REPORTS_FWSP = 5
LEARNING_RATE_FWSP = 2.5e-4
MIN_LEARNING_RATE_FWSP = 3.0e-5
WEIGHT_DECAY_FWSP = 3.0e-5
GRADIENT_CLIP_FWSP = 3.0
POSTERIOR_CHANNELS_FWSP = 40
GLOBAL_HIDDEN_FWSP = 96
GLOBAL_LAYERS_FWSP = 2
GLOBAL_DROPOUT_FWSP = 0.12
PATH_SCALE_FWSP = 24.0
TARGET_SIGMA_FWSP = 1.5
HORIZONS_FWSP = (8, 32, 128)  # latent columns = about 32/128/512 ft
LOSS_CE_FWSP = 0.20
LOSS_PATH_FWSP = 1.00
LOSS_SURFACE_SLOPE_FWSP = 0.20
LOSS_SURFACE_CURVATURE_FWSP = 0.04
LOSS_HORIZON_FWSP = 0.22
ARCHITECTURE_GATE_FWSP = 8.0
TARGET_GATE_FWSP = 5.5

FOLD_CANDIDATES_FWSP = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    DATA_ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
)


def _write_json_fwsp(path, payload):
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def _load_folds_fwsp(wells):
    fold_path = next((path for path in FOLD_CANDIDATES_FWSP if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError(
            "Missing fold-clean well_metadata.csv; run foldclean_full_fleongg_lgb_v0 first"
        )
    frame = pd.read_csv(fold_path)
    well_column = next((c for c in ("well", "well_id", "Well") if c in frame), None)
    fold_column = next((c for c in ("fold", "Fold") if c in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify well/fold columns in {fold_path}")
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well")
    missing = sorted(set(wells).difference(frame["well"]))
    if missing:
        raise RuntimeError(f"Fold metadata misses {len(missing)} wells: {missing[:5]}")
    return frame.loc[frame["well"].isin(wells)].copy(), fold_path


def _ensure_records_fwsp(wells):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for number, well in enumerate(wells, 1):
        path = _record_path(well)
        try:
            if not path.exists():
                _save_record(_build_record(well))
            record = _load_record(well)
            reports.append(
                {
                    "well": well,
                    "status": "ok",
                    "eval_rows": record.eval_rows,
                    "teacher_rmse": record.teacher_rmse,
                    "axis_coverage": record.axis_coverage,
                }
            )
        except Exception as exc:
            reports.append({"well": well, "status": "failed", "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"build/load native records {number}/{len(wells)}", flush=True)
    report = pd.DataFrame(reports)
    report.to_csv(OUT_DIR_FWSP / "record_build_report.csv", index=False)
    failed = report.loc[report["status"].eq("failed")]
    if len(failed):
        raise RuntimeError(
            f"{len(failed)} native records failed: "
            + repr(failed.head(5).to_dict(orient="records"))
        )
    return report


class FullWellPosteriorSurfaceNetFWSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = NativeDualReferenceWarpNet()
        self.posterior_projection = nn.Sequential(
            nn.Conv1d(N_STATES, POSTERIOR_CHANNELS_FWSP, 1),
            nn.GELU(),
            nn.Conv1d(POSTERIOR_CHANNELS_FWSP, POSTERIOR_CHANNELS_FWSP, 3, padding=1),
            nn.GELU(),
        )
        global_inputs = POSTERIOR_CHANNELS_FWSP + 10 + 5
        self.global_input = nn.Sequential(
            nn.Conv1d(global_inputs, GLOBAL_HIDDEN_FWSP, 5, padding=2),
            nn.GroupNorm(8, GLOBAL_HIDDEN_FWSP),
            nn.GELU(),
        )
        self.global_gru = nn.GRU(
            GLOBAL_HIDDEN_FWSP,
            GLOBAL_HIDDEN_FWSP,
            num_layers=GLOBAL_LAYERS_FWSP,
            batch_first=True,
            bidirectional=True,
            dropout=GLOBAL_DROPOUT_FWSP,
        )
        self.path_head = nn.Sequential(
            nn.Linear(2 * GLOBAL_HIDDEN_FWSP, GLOBAL_HIDDEN_FWSP),
            nn.GELU(),
            nn.Dropout(GLOBAL_DROPOUT_FWSP),
            nn.Linear(GLOBAL_HIDDEN_FWSP, 1),
        )
        self.mix_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, horizontal, condition, typewell, heel):
        logits = self.local(horizontal, condition, typewell, heel)
        probability = torch.softmax(logits, dim=1)
        states = torch.as_tensor(
            STATE_OFFSETS, device=logits.device, dtype=logits.dtype
        )[None, :, None]
        local_mean = torch.sum(probability * states, dim=1)
        local_var = torch.sum(probability * (states - local_mean[:, None]) ** 2, dim=1)
        local_std = torch.sqrt(local_var.clamp_min(1e-5))
        local_entropy = -torch.sum(
            probability * torch.log(probability.clamp_min(1e-8)), dim=1
        ) / np.log(N_STATES)
        local_peak = probability.amax(dim=1)
        local_margin = torch.topk(probability, 2, dim=1).values
        local_margin = local_margin[:, 0] - local_margin[:, 1]

        posterior = self.posterior_projection(probability)
        columns = posterior.shape[-1]
        condition_latent = condition[..., ::HORIZONTAL_STRIDE][..., :columns]
        summary = torch.stack(
            [
                local_mean / STATE_HALF_WIDTH_FT,
                local_std / STATE_HALF_WIDTH_FT,
                local_entropy,
                local_peak,
                local_margin,
            ],
            dim=1,
        )
        features = torch.cat([posterior, condition_latent, summary], dim=1)
        encoded = self.global_input(features).transpose(1, 2)
        encoded, _ = self.global_gru(encoded)
        global_offset = STATE_HALF_WIDTH_FT * torch.tanh(
            self.path_head(encoded).squeeze(-1)
        )
        mix = torch.sigmoid(self.mix_logit)
        path = mix * global_offset + (1.0 - mix) * local_mean
        return logits, path, {
            "local_mean": local_mean,
            "global_offset": global_offset,
            "mix": mix,
            "entropy": local_entropy,
        }


def _tensor_record_fwsp(record, device, mode="full", shuffled_typewell=None):
    horizontal = record.horizontal.copy()
    typewell = record.typewell.copy()
    heel = record.heel.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    elif mode == "shuffled_typewell":
        if shuffled_typewell is None:
            raise ValueError("shuffled_typewell record is required")
        typewell = shuffled_typewell.typewell.copy()
    return (
        torch.from_numpy(horizontal)[None].to(device),
        torch.from_numpy(record.condition)[None].to(device),
        torch.from_numpy(typewell)[None].to(device),
        torch.from_numpy(heel)[None].to(device),
    )


def _soft_targets_fwsp(target, device):
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=target.dtype)[None]
    distribution = torch.exp(-0.5 * ((states - target[:, None]) / TARGET_SIGMA_FWSP) ** 2)
    return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _record_targets_fwsp(record, columns, device):
    target = torch.from_numpy(record.latent_truth_offset[:columns]).to(device)
    mask = torch.from_numpy(record.latent_eval_mask[:columns]).to(device)
    latent_index = record.latent_index[:columns]
    z_relative = torch.from_numpy(record.condition[4, latent_index]).to(device) * 512.0
    surface = target + z_relative
    return target, surface, mask


def _loss_fwsp(logits, path, record, device):
    columns = min(logits.shape[-1], path.shape[-1])
    logits = logits[..., :columns]
    path = path[..., :columns]
    target, surface_target, mask = _record_targets_fwsp(record, columns, device)
    valid = mask & torch.isfinite(target) & (torch.abs(target) <= STATE_HALF_WIDTH_FT)
    if int(valid.sum()) < 32:
        raise RuntimeError(f"{record.well}: too few supervised columns")

    soft = _soft_targets_fwsp(target[valid], device)
    ce = -(soft * torch.log_softmax(logits[0, :, valid].T, dim=1)).sum(dim=1).mean()
    path_error = (path[0, valid] - target[valid]) / PATH_SCALE_FWSP
    path_mse = path_error.square().mean()

    z_relative = surface_target - target
    surface_prediction = path[0] + z_relative
    supervised_indices = torch.nonzero(valid, as_tuple=False).flatten()
    pred_surface = surface_prediction[supervised_indices]
    true_surface = surface_target[supervised_indices]
    slope = F.smooth_l1_loss(
        torch.diff(pred_surface) / 4.0,
        torch.diff(true_surface) / 4.0,
        beta=0.25,
    )
    if len(pred_surface) > 2:
        curvature = F.smooth_l1_loss(
            torch.diff(pred_surface, n=2) / 2.0,
            torch.diff(true_surface, n=2) / 2.0,
            beta=0.20,
        )
    else:
        curvature = path_mse.new_tensor(0.0)

    horizon_losses = []
    for horizon in HORIZONS_FWSP:
        if columns <= horizon:
            continue
        pair_mask = valid[horizon:] & valid[:-horizon]
        if int(pair_mask.sum()) < 8:
            continue
        pred_delta = surface_prediction[horizon:] - surface_prediction[:-horizon]
        true_delta = surface_target[horizon:] - surface_target[:-horizon]
        scale = max(float(np.sqrt(horizon) * 4.0), 8.0)
        horizon_losses.append(
            F.smooth_l1_loss(
                pred_delta[pair_mask] / scale,
                true_delta[pair_mask] / scale,
                beta=0.25,
            )
        )
    horizon_loss = (
        torch.stack(horizon_losses).mean()
        if horizon_losses
        else path_mse.new_tensor(0.0)
    )
    total = (
        LOSS_CE_FWSP * ce
        + LOSS_PATH_FWSP * path_mse
        + LOSS_SURFACE_SLOPE_FWSP * slope
        + LOSS_SURFACE_CURVATURE_FWSP * curvature
        + LOSS_HORIZON_FWSP * horizon_loss
    )
    return total, {
        "ce": ce.detach(),
        "path_mse": path_mse.detach(),
        "slope": slope.detach(),
        "curvature": curvature.detach(),
        "horizon": horizon_loss.detach(),
    }


def _decode_record_fwsp(record, latent_path):
    columns = min(len(record.latent_index), len(latent_path))
    latent_md = record.md[record.latent_index[:columns]]
    latent_tvt = record.anchor_tvt + np.asarray(latent_path[:columns], np.float64)
    support_md = np.r_[record.md[record.anchor_index], latent_md]
    support_tvt = np.r_[record.anchor_tvt, latent_tvt]
    order = np.argsort(support_md, kind="mergesort")
    support_md = support_md[order]
    support_tvt = support_tvt[order]
    unique, indices = np.unique(support_md, return_index=True)
    return np.interp(record.md[record.eval_index], unique, support_tvt[indices])


def _metrics_fwsp(rows):
    truth = rows["truth"].to_numpy(np.float64)
    prediction = rows["prediction"].to_numpy(np.float64)
    error = prediction - truth
    bywell = rows.assign(error2=error**2).groupby("well", sort=False)["error2"].mean().pow(0.5)
    output = {
        "rows": int(len(rows)),
        "wells": int(rows["well"].nunique()),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p75_well_rmse": float(bywell.quantile(0.75)),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    bands = (
        ("near_0_500_rmse", 0.0, 500.0),
        ("mid_500_1000_rmse", 500.0, 1000.0),
        ("mid_1000_2000_rmse", 1000.0, 2000.0),
        ("far_gt2000_rmse", 2000.0, np.inf),
    )
    distance = rows["md_since"].to_numpy(np.float64)
    for name, low, high in bands:
        selected = (distance >= low) & (distance < high)
        output[name] = (
            float(np.sqrt(np.mean(error[selected] ** 2))) if selected.any() else float("nan")
        )
    return output, bywell.reset_index(name="rmse")


@torch.no_grad()
def _evaluate_fwsp(model, records, device, mode="full", save_rows=False):
    model.eval()
    rows = []
    shuffled = list(reversed(records))
    entropies = []
    mixes = []
    for number, record in enumerate(records):
        other = shuffled[number]
        inputs = _tensor_record_fwsp(record, device, mode, other)
        logits, path, diagnostics = model(*inputs)
        prediction = _decode_record_fwsp(record, path[0].cpu().numpy())
        md_since = record.md[record.eval_index] - record.md[record.anchor_index]
        rows.append(
            pd.DataFrame(
                {
                    "well": record.well,
                    "row_idx": record.eval_index,
                    "truth": record.truth[record.eval_index],
                    "prediction": prediction,
                    "md_since": md_since,
                }
            )
        )
        entropies.append(float(diagnostics["entropy"].mean().cpu()))
        mixes.append(float(diagnostics["mix"].cpu()))
    frame = pd.concat(rows, ignore_index=True)
    metrics, bywell = _metrics_fwsp(frame)
    metrics["mean_entropy"] = float(np.mean(entropies))
    metrics["mean_global_mix"] = float(np.mean(mixes))
    return metrics, bywell, frame if save_rows else None


def _train_fwsp(train_records, valid_records, device):
    _seed_everything(SEED_FWSP)
    model = FullWellPosteriorSurfaceNetFWSP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_FWSP,
        weight_decay=WEIGHT_DECAY_FWSP,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_FWSP, eta_min=MIN_LEARNING_RATE_FWSP
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=512.0)
    rng = np.random.default_rng(SEED_FWSP)
    best = None
    bad_reports = 0
    history = []
    started = time.time()
    for epoch in range(1, MAX_EPOCHS_FWSP + 1):
        model.train()
        order = rng.permutation(len(train_records))
        sums = {key: 0.0 for key in ("loss", "ce", "path_mse", "slope", "curvature", "horizon")}
        skipped = 0
        for step, index in enumerate(order, 1):
            record = train_records[int(index)]
            inputs = _tensor_record_fwsp(record, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits, path, _ = model(*inputs)
                loss, parts = _loss_fwsp(logits, path, record, device)
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_FWSP)
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < before:
                skipped += 1
            sums["loss"] += float(loss.detach().cpu())
            for key in parts:
                sums[key] += float(parts[key].cpu())
            if step % 100 == 0 or step == len(order):
                print(
                    f"FWSP epoch {epoch}/{MAX_EPOCHS_FWSP} wells {step}/{len(order)} "
                    f"loss={sums['loss']/step:.4f}",
                    flush=True,
                )
            del inputs, logits, path, loss
        scheduler.step()

        if epoch not in (1, 2, MAX_EPOCHS_FWSP) and epoch % REPORT_EVERY_FWSP:
            continue
        valid_metrics, _, _ = _evaluate_fwsp(model, valid_records, device)
        report = {
            "epoch": epoch,
            **{f"train_{key}": value / len(order) for key, value in sums.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "skipped_steps": skipped,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(report)
        print("FWSP EPOCH", report, flush=True)
        if best is None or valid_metrics["rmse"] < best["metrics"]["rmse"] - 1e-4:
            best = {
                "epoch": epoch,
                "metrics": valid_metrics,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            torch.save(
                {"model": best["state"], "epoch": epoch, "metrics": valid_metrics},
                MODEL_DIR_FWSP / "best_fold2.pt",
            )
            bad_reports = 0
        else:
            bad_reports += 1
        pd.DataFrame(history).to_csv(OUT_DIR_FWSP / "training_history.csv", index=False)
        if epoch >= MIN_EPOCHS_FWSP and bad_reports >= PATIENCE_REPORTS_FWSP:
            print(f"FWSP early stopping: best epoch {best['epoch']}", flush=True)
            break
    if best is None:
        raise RuntimeError("No FWSP checkpoint was produced")
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


STARTED_FWSP = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a CUDA GPU")
DEVICE_FWSP = torch.device("cuda")
_seed_everything(SEED_FWSP)

wells_fwsp = [path.name.split("__", 1)[0] for path in _horizontal_paths()]
folds_fwsp, fold_path_fwsp = _load_folds_fwsp(wells_fwsp)
build_report_fwsp = _ensure_records_fwsp(wells_fwsp)
train_wells_fwsp = folds_fwsp.loc[
    folds_fwsp["fold"].isin(TRAIN_FOLDS_FWSP), "well"
].tolist()
valid_wells_fwsp = folds_fwsp.loc[
    folds_fwsp["fold"].eq(VALID_FOLD_FWSP), "well"
].tolist()
if len(train_wells_fwsp) < 460 or len(valid_wells_fwsp) < 154:
    raise RuntimeError(
        f"Unexpected Fold345/Fold2 sizes: {len(train_wells_fwsp)}/{len(valid_wells_fwsp)}"
    )
train_records_fwsp = [_load_record(well) for well in train_wells_fwsp]
valid_records_fwsp = [_load_record(well) for well in valid_wells_fwsp]

guards_fwsp = {
    "data_root": str(DATA_ROOT),
    "fold_source": str(fold_path_fwsp),
    "train_folds": list(TRAIN_FOLDS_FWSP),
    "validation_fold": VALID_FOLD_FWSP,
    "train_wells": len(train_records_fwsp),
    "valid_wells": len(valid_records_fwsp),
    "train_rows": int(sum(record.eval_rows for record in train_records_fwsp)),
    "valid_rows": int(sum(record.eval_rows for record in valid_records_fwsp)),
    "minimum_axis_coverage": float(build_report_fwsp["axis_coverage"].min()),
    "maximum_teacher_rmse": float(build_report_fwsp["teacher_rmse"].max()),
    "uses_historical_prediction": False,
    "uses_neighbor_spatial_cp12_gold": False,
    "sample_unit": "one complete physical well",
}
print("FWSP CONFIG", json.dumps(guards_fwsp, indent=2), flush=True)

model_fwsp, best_fwsp, history_fwsp, training_minutes_fwsp = _train_fwsp(
    train_records_fwsp, valid_records_fwsp, DEVICE_FWSP
)
full_fwsp, bywell_fwsp, rows_fwsp = _evaluate_fwsp(
    model_fwsp, valid_records_fwsp, DEVICE_FWSP, save_rows=True
)
rows_fwsp.to_parquet(OUT_DIR_FWSP / "fold2_oof.parquet", index=False)
bywell_fwsp.sort_values("rmse", ascending=False).to_csv(
    OUT_DIR_FWSP / "fold2_bywell.csv", index=False
)

controls_fwsp = []
for mode_fwsp in ("full", "no_gr", "shuffled_typewell"):
    metrics_fwsp, _, _ = _evaluate_fwsp(
        model_fwsp, valid_records_fwsp, DEVICE_FWSP, mode=mode_fwsp
    )
    controls_fwsp.append(
        {
            "mode": mode_fwsp,
            **metrics_fwsp,
            "degradation_vs_full": metrics_fwsp["rmse"] - full_fwsp["rmse"],
        }
    )
controls_frame_fwsp = pd.DataFrame(controls_fwsp)
controls_frame_fwsp.to_csv(OUT_DIR_FWSP / "causal_controls.csv", index=False)
no_gr_drop_fwsp = float(
    controls_frame_fwsp.loc[controls_frame_fwsp["mode"].eq("no_gr"), "degradation_vs_full"].iloc[0]
)
shuffle_drop_fwsp = float(
    controls_frame_fwsp.loc[
        controls_frame_fwsp["mode"].eq("shuffled_typewell"), "degradation_vs_full"
    ].iloc[0]
)

checks_fwsp = {
    "architecture_rmse_le_8": full_fwsp["rmse"] <= ARCHITECTURE_GATE_FWSP,
    "target_rmse_le_5p5": full_fwsp["rmse"] <= TARGET_GATE_FWSP,
    "far_rmse_le_10": full_fwsp["far_gt2000_rmse"] <= 10.0,
    "no_gr_degradation_ge_0p5": no_gr_drop_fwsp >= 0.5,
    "shuffled_typewell_degradation_ge_0p5": shuffle_drop_fwsp >= 0.5,
}
strict_pass_fwsp = bool(
    checks_fwsp["architecture_rmse_le_8"]
    and checks_fwsp["no_gr_degradation_ge_0p5"]
    and checks_fwsp["shuffled_typewell_degradation_ge_0p5"]
)
verdict_fwsp = {
    "experiment": EXPERIMENT_FWSP,
    "stage": "fold2_native_costmap_completewell_surface_gateway",
    "architecture": "native dual-reference 257-state posterior + bidirectional whole-well decoder",
    "legal_inputs_only": True,
    "guards": guards_fwsp,
    "parameters": int(sum(parameter.numel() for parameter in model_fwsp.parameters())),
    "selected_epoch": int(best_fwsp["epoch"]),
    "fold2": full_fwsp,
    "controls": {
        "no_gr_degradation": no_gr_drop_fwsp,
        "shuffled_typewell_degradation": shuffle_drop_fwsp,
    },
    "checks": checks_fwsp,
    "strict_pass": strict_pass_fwsp,
    "training_minutes": training_minutes_fwsp,
    "elapsed_minutes": (time.time() - STARTED_FWSP) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "TARGET PASS -> repeat on all five folds"
        if checks_fwsp["target_rmse_le_5p5"] and strict_pass_fwsp
        else (
            "ARCHITECTURE PASS -> inspect worst wells and add official-mask augmentation"
            if strict_pass_fwsp
            else "FAIL -> posterior ordering still does not transfer; stop this evidence family"
        )
    ),
}
_write_json_fwsp(OUT_DIR_FWSP / "verdict.json", verdict_fwsp)
print("\nFWSP CONTROLS")
print(controls_frame_fwsp.to_string(index=False), flush=True)
print("\nFWSP FINAL VERDICT")
print(json.dumps(verdict_fwsp, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_FWSP, flush=True)
