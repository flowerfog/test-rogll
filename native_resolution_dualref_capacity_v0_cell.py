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


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


STARTED = time.time()
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("This capacity audit requires a CUDA GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_everything(SEED)

print(
    {
        "experiment": EXPERIMENT,
        "data_root": str(DATA_ROOT),
        "device": str(device),
        "sample_unit": "one complete well",
        "horizontal_resolution": "native before learned stride",
        "horizontal_stride": HORIZONTAL_STRIDE,
        "states": N_STATES,
        "state_axis": "TVT - last_known_TVT",
        "uses_neighbor": False,
        "uses_existing_prediction": False,
    },
    flush=True,
)

scan_rows = []
horizontal_paths = _horizontal_paths()
for index, path in enumerate(horizontal_paths, 1):
    scan_rows.append(_scan_well(path))
    if index % 100 == 0 or index == len(horizontal_paths):
        print(f"scan input wells {index}/{len(horizontal_paths)}", flush=True)
metadata = pd.DataFrame(scan_rows)
metadata.to_csv(OUT_DIR / "input_well_audit.csv", index=False)
eligible = metadata[
    metadata["contiguous_mask"]
    & (metadata["known_rows"] >= 64)
    & (metadata["eval_rows"] >= 128)
    & (metadata["target_finite_fraction"] >= 0.999)
].reset_index(drop=True)
if len(eligible) < 700:
    raise RuntimeError(f"Only {len(eligible)} wells pass the legal-input guards")

selection = _farthest_capacity_wells(eligible, N_CAPACITY_WELLS)
selection["selection_order"] = np.arange(1, len(selection) + 1)
selection.to_csv(OUT_DIR / "capacity_well_selection.csv", index=False)
print("CAPACITY WELL SELECTION")
print(selection.to_string(index=False), flush=True)

build_rows = []
for index, row in enumerate(selection.itertuples(index=False), 1):
    well = str(row.well)
    path = _record_path(well)
    try:
        rebuild = True
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                rebuild = "cache_version" not in data.files or int(data["cache_version"][0]) != 1
        if rebuild:
            record = _build_record(well)
            _save_record(record)
        else:
            record = _load_record(well)
        build_rows.append(
            {
                "well": well,
                "status": "ok",
                "rows": len(record.md),
                "eval_rows": record.eval_rows,
                "latent_columns": len(record.latent_index),
                "teacher_rmse": record.teacher_rmse,
                "axis_coverage": record.axis_coverage,
                "calibration_pairs": record.calibration_pairs,
                "calibration_rmse": record.calibration_rmse,
                "typewell_valid": float(
                    record.typewell[1, REFERENCE_STATE_START : REFERENCE_STATE_START + N_STATES].mean()
                ),
                "heel_valid": float(
                    record.heel[1, REFERENCE_STATE_START : REFERENCE_STATE_START + N_STATES].mean()
                ),
            }
        )
    except Exception as exc:
        build_rows.append({"well": well, "status": "failed", "reason": repr(exc)})
    print(f"build native record {index}/{len(selection)} {well}", flush=True)

build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "record_build_report.csv", index=False)
failed = build_report[build_report["status"] != "ok"]
if len(failed):
    raise RuntimeError(f"Failed to build {len(failed)} records; see record_build_report.csv")

guards = {
    "eligible_wells": len(eligible),
    "capacity_wells": len(build_report),
    "teacher_rmse_mean": float(build_report["teacher_rmse"].mean()),
    "teacher_rmse_max": float(build_report["teacher_rmse"].max()),
    "axis_coverage_min": float(build_report["axis_coverage"].min()),
    "typewell_valid_mean": float(build_report["typewell_valid"].mean()),
    "heel_valid_mean": float(build_report["heel_valid"].mean()),
    "horizontal_rows_mean": float(build_report["rows"].mean()),
    "latent_columns_mean": float(build_report["latent_columns"].mean()),
    "native_points_retained_before_stride": 1.0,
}
print("REPRESENTATION GUARDS", json.dumps(guards, indent=2), flush=True)
if (
    guards["teacher_rmse_max"] > 0.35
    or guards["axis_coverage_min"] < 0.999
    or guards["typewell_valid_mean"] < 0.90
):
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "representation_guards",
        "guards": guards,
        "strict_pass": False,
        "next": "FAIL -> repair native TVT-axis construction before training",
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    raise RuntimeError(json.dumps(verdict, indent=2))

records = [_load_record(str(well)) for well in selection["well"]]
probe = NativeDualReferenceWarpNet().to(device)
with torch.no_grad():
    inputs = _tensor_record(records[0], device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = probe(*inputs)
if output.shape[1] != N_STATES or not torch.isfinite(output).all():
    raise RuntimeError(f"model smoke output invalid: {tuple(output.shape)}")
parameter_count = int(sum(parameter.numel() for parameter in probe.parameters()))
smoke = {
    "parameters": parameter_count,
    "input_horizontal": list(inputs[0].shape),
    "input_condition": list(inputs[1].shape),
    "input_typewell": list(inputs[2].shape),
    "output": list(output.shape),
    "finite": bool(torch.isfinite(output).all()),
}
print("MODEL SMOKE", smoke, flush=True)
del probe, inputs, output
gc.collect()
torch.cuda.empty_cache()

results = []
for config in SCENARIOS:
    result = _train_scenario(config, records, device)
    results.append(result)
    pd.DataFrame(results).to_csv(OUT_DIR / "capacity_results.csv", index=False)
    print("SCENARIO RESULT", result, flush=True)
    if not result["pass"]:
        print(f"STOP LADDER: {config['name']} failed its capacity gate", flush=True)
        break

result_frame = pd.DataFrame(results)
strict_pass = bool(
    len(results) == len(SCENARIOS)
    and all(bool(result["pass"]) for result in results)
)
verdict = {
    "experiment": EXPERIMENT,
    "sample_unit": "one complete well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "device": str(device),
    "states": N_STATES,
    "state_half_width_ft": STATE_HALF_WIDTH_FT,
    "horizontal_stride_after_native_encoding": HORIZONTAL_STRIDE,
    "parameters": parameter_count,
    "guards": guards,
    "model_smoke": smoke,
    "results": results,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build held-out fold-1 pooled-RMSE experiment with official-mask validation and causal controls"
        if strict_pass
        else "FAIL -> repair the native dual-reference representation; do not start held-out OOF"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nCAPACITY RESULTS")
print(result_frame.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
