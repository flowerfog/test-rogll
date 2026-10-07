"""Fold-1 gateway for a complete-well real-warp energy model.

This is a standalone, no-neighbour experiment.  It does not consume strong,
strong_bias, CP12, Gold, Neighbor, or validation labels at inference.  The
only cached prerequisite is the fold-clean v4 panel bank, which supplies the
official/pseudo prediction starts and deployment-legal PF centre trajectory.

Unlike the previous direct decoders, the network does not emit TVT columns.
It learns a scalar energy for any smooth complete-well candidate path.  During
training, candidates are made by perturbing the true trajectory while keeping
the real horizontal and typewell GR curves unchanged.  At validation time a
fold-clean template bank and stochastic beam search generate candidates
without using validation truth.
"""

from pathlib import Path
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
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
PANEL_ROOT = OOF_ROOT / "tvt_corrnet_heel_conditioned_onefold_v4" / "panels"
PANEL_GUARDS = OOF_ROOT / "tvt_corrnet_heel_conditioned_onefold_v4" / "panel_guards.csv"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
OUT_DIR = OOF_ROOT / "fullwell_realwarp_energy_onefold_v0"
MODEL_DIR = OUT_DIR / "models"

VALID_FOLD = 1
TRAIN_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
PREFIX_COLUMNS = 64
EVAL_COLUMNS = 256

SPLINE_CONTROL_OPTIONS = (8, 12, 16, 24, 32)
SPLINE_CONTROLS = 16
COEFFICIENT_CAP_FT = 96.0
SPLINE_RIDGE = 1e-4
TEMPLATE_CLUSTERS = 96

TRAIN_CANDIDATES = 48
TRAIN_EPOCHS = 12
TRAIN_BATCH_SIZE = 1
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 2e-4
PATIENCE = 4
SEED = 20260725

FIXED_BANK_RANDOM = 24
SEARCH_ELITES = 12
SEARCH_CHILDREN = 8
SEARCH_SCALES_FT = (16.0, 8.0, 4.0, 2.0)
SEARCH_SCORE_CHUNK = 96
POSTERIOR_TOPK = 8
POSTERIOR_TEMPERATURE = 0.10

# A representation floor below 30% of the 5.5-ft target leaves enough SSE
# budget while keeping the deployment search only 15-dimensional.  Chasing a
# sub-foot spline fit with 24-32 controls makes candidate search much harder.
REPRESENTATION_GATE_FT = 1.65
FIXED_BANK_ORACLE_GATE_FT = 7.0
ARCHITECTURE_GATE_FT = 7.0
TARGET_GATE_FT = 5.5
CAUSAL_GATE_FT = 0.30


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _robust_normalise(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    output = np.zeros(values.shape, np.float32)
    if int(valid.sum()) < 4:
        return output
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    output[valid] = np.clip((values[valid] - centre) / scale, -8.0, 8.0)
    return output


def _unique_curve(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    order = np.argsort(x[valid], kind="mergesort")
    x = x[valid][order]
    y = y[valid][order]
    unique, inverse = np.unique(x, return_inverse=True)
    if len(unique) != len(x):
        y = np.bincount(inverse, weights=y) / np.maximum(
            np.bincount(inverse), 1
        )
        x = unique
    return x.astype(np.float32), y.astype(np.float32)


def _interp_curve(x, y, query):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    if len(x) < 2:
        return np.full(query.shape, np.nan, np.float32)
    return np.interp(query, x, y, left=np.nan, right=np.nan).astype(np.float32)


def _safe_affine(x, y, valid):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 12:
        return 1.0, 0.0
    xv = x[valid]
    yv = y[valid]
    design = np.column_stack([xv, np.ones(len(xv))])
    slope, intercept = np.linalg.lstsq(design, yv, rcond=None)[0]
    residual = yv - (slope * xv + intercept)
    scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    if np.isfinite(scale) and scale > 1e-4:
        keep = np.abs(residual - np.median(residual)) <= 3.5 * scale
        if int(keep.sum()) >= 12:
            slope, intercept = np.linalg.lstsq(design[keep], yv[keep], rcond=None)[0]
    return float(np.clip(slope, -3.0, 3.0)), float(np.clip(intercept, -4.0, 4.0))


def _spline_basis(length, controls):
    column_progress = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
    knots = np.linspace(0.0, 1.0, int(controls), dtype=np.float64)
    eye = np.eye(int(controls), dtype=np.float64)
    basis = CubicSpline(knots, eye, axis=0, bc_type="natural")(column_progress)
    # Control point zero is fixed at zero so every candidate is anchored at PS.
    return basis[:, 1:].astype(np.float32)


def _fit_coefficients(target, basis, cap=COEFFICIENT_CAP_FT):
    target = np.asarray(target, np.float64)
    design = np.asarray(basis, np.float64)
    lhs = design.T @ design + SPLINE_RIDGE * np.eye(design.shape[1])
    rhs = design.T @ target
    coefficients = np.linalg.solve(lhs, rhs)
    return np.clip(coefficients, -cap, cap).astype(np.float32)


def _smooth_noise(rng, dimensions, scale):
    values = rng.normal(size=int(dimensions)).astype(np.float32)
    for _ in range(2):
        padded = np.pad(values, (1, 1), mode="edge")
        values = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    std = float(np.std(values))
    if std > 1e-5:
        values /= std
    return values * float(scale)


def _deduplicate_coefficients(coefficients, limit=None):
    coefficients = np.asarray(coefficients, np.float32)
    if not len(coefficients):
        return coefficients
    rounded = np.round(coefficients, 2)
    _, first = np.unique(rounded, axis=0, return_index=True)
    first = np.sort(first)
    output = coefficients[first]
    return output if limit is None else output[: int(limit)]


def _weighted_ncc(observed, candidate, valid):
    observed = np.asarray(observed, np.float32)
    candidate = np.asarray(candidate, np.float32)
    valid = np.asarray(valid, bool)
    weight = valid.astype(np.float32)
    count = np.maximum(weight.sum(axis=1), 1.0)
    mean_x = (candidate * weight).sum(axis=1) / count
    mean_y = (observed[None, :] * weight).sum(axis=1) / count
    x = (candidate - mean_x[:, None]) * weight
    y = (observed[None, :] - mean_y[:, None]) * weight
    numerator = (x * y).sum(axis=1)
    denominator = np.sqrt(
        np.maximum((x * x).sum(axis=1) * (y * y).sum(axis=1), 1e-8)
    )
    output = numerator / denominator
    output[count < 16] = -1.0
    return output.astype(np.float32)


def _well_tail_metrics(bywell, rmse_column):
    """Report well quantiles plus row-weighted SSE concentration."""
    frame = bywell[["rows", rmse_column]].copy()
    frame = frame[np.isfinite(frame[rmse_column]) & (frame["rows"] > 0)]
    if frame.empty:
        return {}
    frame["sse"] = frame["rows"] * frame[rmse_column] ** 2
    frame = frame.sort_values("sse", ascending=False).reset_index(drop=True)
    total_sse = float(frame["sse"].sum())
    total_rows = float(frame["rows"].sum())
    output = {
        "p50_well_rmse": float(frame[rmse_column].quantile(0.50)),
        "p75_well_rmse": float(frame[rmse_column].quantile(0.75)),
        "p90_well_rmse": float(frame[rmse_column].quantile(0.90)),
        "p95_well_rmse": float(frame[rmse_column].quantile(0.95)),
    }
    for fraction, label in ((0.05, "top5pct"), (0.10, "top10pct"), (0.20, "top20pct")):
        count = max(1, int(math.ceil(len(frame) * fraction)))
        removed_sse = float(frame.iloc[:count]["sse"].sum())
        removed_rows = float(frame.iloc[:count]["rows"].sum())
        output[f"{label}_well_sse_share"] = removed_sse / max(total_sse, 1e-12)
        output[f"rmse_without_{label}_wells"] = float(
            np.sqrt(
                max(total_sse - removed_sse, 0.0)
                / max(total_rows - removed_rows, 1.0)
            )
        )
    return output


def _oracle_shrink(center, prediction, truth):
    """Hindsight-only per-well convex dose, used strictly as a ceiling audit."""
    center = np.asarray(center, np.float64)
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    movement = prediction - center
    denominator = float(np.dot(movement, movement))
    if denominator <= 1e-12:
        alpha = 0.0
    else:
        alpha = float(
            np.clip(-np.dot(center - truth, movement) / denominator, 0.0, 1.0)
        )
    shrunk = center + alpha * movement
    return shrunk.astype(np.float32), alpha


def _local_ncc(observed, candidate, valid, window):
    observed = np.asarray(observed, np.float32)[None, :]
    candidate = np.asarray(candidate, np.float32)
    weight = np.asarray(valid, np.float32)
    count = uniform_filter1d(weight, int(window), axis=1, mode="nearest")
    mean_x = uniform_filter1d(candidate * weight, int(window), axis=1, mode="nearest")
    mean_y = uniform_filter1d(observed * weight, int(window), axis=1, mode="nearest")
    denominator = np.maximum(count, 1e-3)
    mean_x /= denominator
    mean_y /= denominator
    dx = (candidate - mean_x) * weight
    dy = (observed - mean_y) * weight
    covariance = uniform_filter1d(dx * dy, int(window), axis=1, mode="nearest")
    variance_x = uniform_filter1d(dx * dx, int(window), axis=1, mode="nearest")
    variance_y = uniform_filter1d(dy * dy, int(window), axis=1, mode="nearest")
    ncc = covariance / np.sqrt(np.maximum(variance_x * variance_y, 1e-6))
    ncc[count < 0.35] = 0.0
    return np.clip(ncc, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Panel and reference loading
# ---------------------------------------------------------------------------


class ReferenceStore:
    def __init__(self, wells):
        self.rows = {}
        for well in tqdm(sorted(set(map(str, wells))), desc="load typewell references"):
            path = DATA_ROOT / "train" / f"{well}__typewell.csv"
            frame = pd.read_csv(path, usecols=["TVT", "GR"])
            tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
            gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)
            valid = np.isfinite(tvt) & np.isfinite(gr)
            tvt, gr = _unique_curve(tvt[valid], _robust_normalise(gr[valid]))
            if len(tvt) < 32:
                raise RuntimeError(f"typewell too short: {well}")
            self.rows[well] = {"tvt": tvt, "gr": gr}


def _record_from_panel(path, fold_value):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "well",
            "tag",
            "horizontal_gr",
            "horizontal_valid",
            "geometry",
            "center_tvt",
            "truth_residual",
            "column_md",
            "column_z",
            "eval_md",
            "eval_z",
            "eval_truth",
            "eval_rows",
            "panel_weight",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"{path.name} misses {missing}")
        record = {
            "well": str(data["well"][0]),
            "tag": str(data["tag"][0]),
            "fold": int(fold_value),
            "horizontal_gr": data["horizontal_gr"].astype(np.float32),
            "horizontal_valid": data["horizontal_valid"].astype(np.float32),
            "geometry": data["geometry"].astype(np.float32),
            "center_tvt": data["center_tvt"].astype(np.float32),
            "truth_residual": data["truth_residual"].astype(np.float32),
            "column_md": data["column_md"].astype(np.float32),
            "column_z": data["column_z"].astype(np.float32),
            "eval_md": data["eval_md"].astype(np.float32),
            "eval_z": data["eval_z"].astype(np.float32),
            "eval_truth": data["eval_truth"].astype(np.float32),
            "eval_rows": data["eval_rows"].astype(np.int32),
            "panel_weight": float(data["panel_weight"][0]),
        }
    if len(record["column_md"]) != PREFIX_COLUMNS + EVAL_COLUMNS:
        raise RuntimeError(f"unexpected panel width: {path}")
    return record


def _prepare_record(record, references, basis):
    prefix = PREFIX_COLUMNS
    well = record["well"]
    reference = references.rows[well]
    record["reference_tvt"] = reference["tvt"]
    record["reference_gr"] = reference["gr"]

    prefix_tvt = record["center_tvt"][:prefix]
    prefix_horizontal = record["horizontal_gr"][:prefix]
    prefix_valid = record["horizontal_valid"][:prefix] > 0.5
    prefix_typewell = _interp_curve(reference["tvt"], reference["gr"], prefix_tvt)
    calibration_valid = prefix_valid & np.isfinite(prefix_typewell)
    slope, intercept = _safe_affine(prefix_typewell, prefix_horizontal, calibration_valid)
    record["calibration_slope"] = slope
    record["calibration_intercept"] = intercept
    self_tvt, self_gr = _unique_curve(
        prefix_tvt[calibration_valid], prefix_horizontal[calibration_valid]
    )
    record["self_tvt"] = self_tvt
    record["self_gr"] = self_gr

    record["eval_center_tvt"] = record["center_tvt"][prefix:].astype(np.float32)
    record["eval_column_md"] = record["column_md"][prefix:].astype(np.float32)
    record["eval_column_z"] = record["column_z"][prefix:].astype(np.float32)
    record["eval_horizontal_gr"] = record["horizontal_gr"][prefix:].astype(np.float32)
    record["eval_horizontal_valid"] = (
        record["horizontal_valid"][prefix:] > 0.5
    )
    record["eval_truth_residual"] = record["truth_residual"][prefix:].astype(np.float32)
    record["truth_coefficients"] = _fit_coefficients(
        record["eval_truth_residual"], basis
    )

    center_s = record["center_tvt"] + record["column_z"]
    record["center_prediction"] = (
        np.interp(record["eval_md"], record["column_md"], center_s)
        - record["eval_z"]
    ).astype(np.float32)
    anchor_md = float(record["column_md"][prefix - 1])
    correction_md = np.r_[anchor_md, record["eval_column_md"]].astype(np.float32)
    position = np.searchsorted(correction_md, record["eval_md"], side="right")
    right = np.clip(position, 1, len(correction_md) - 1)
    left = right - 1
    denominator = np.maximum(correction_md[right] - correction_md[left], 1e-6)
    alpha = (record["eval_md"] - correction_md[left]) / denominator
    record["interp_left"] = left.astype(np.int16)
    record["interp_right"] = right.astype(np.int16)
    record["interp_alpha"] = alpha.astype(np.float32)
    return record


def _coefficients_to_column_correction(coefficients, basis):
    correction = np.asarray(coefficients, np.float32) @ np.asarray(
        basis, np.float32
    ).T
    return np.clip(correction, -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT)


def _candidate_row_predictions(record, coefficients, basis):
    correction = _coefficients_to_column_correction(coefficients, basis)
    extended = np.concatenate(
        [np.zeros((len(correction), 1), np.float32), correction], axis=1
    )
    left = record["interp_left"]
    right = record["interp_right"]
    alpha = record["interp_alpha"][None, :]
    row_correction = (1.0 - alpha) * extended[:, left] + alpha * extended[:, right]
    return record["center_prediction"][None, :] + row_correction


def _candidate_exact_rmse(record, coefficients, basis):
    prediction = _candidate_row_predictions(record, coefficients, basis)
    return np.sqrt(
        np.mean((prediction - record["eval_truth"][None, :]) ** 2, axis=1)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Candidate generation and candidate-aligned GR features
# ---------------------------------------------------------------------------


def _training_coefficients(record, templates, epoch, index):
    rng = np.random.default_rng(SEED + 1000003 * int(epoch) + 7919 * int(index))
    truth = record["truth_coefficients"].astype(np.float32)
    dimensions = len(truth)
    candidates = [np.zeros(dimensions, np.float32), truth.copy()]

    if len(templates):
        take = min(8, len(templates))
        chosen = rng.choice(len(templates), size=take, replace=False)
        candidates.extend(templates[chosen])

    for scale in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        candidates.append(truth + _smooth_noise(rng, dimensions, scale))
        candidates.append(truth + _smooth_noise(rng, dimensions, scale))

    control_progress = np.linspace(1.0 / SPLINE_CONTROLS, 1.0, dimensions)
    for shift in (-48.0, -24.0, -12.0, 12.0, 24.0, 48.0):
        candidates.append(truth + shift * control_progress)
    for endpoint in (-64.0, -32.0, 32.0, 64.0):
        candidates.append(endpoint * control_progress)

    while len(candidates) < TRAIN_CANDIDATES * 2:
        mode = int(rng.integers(0, 4))
        scale = float(rng.choice([4.0, 8.0, 16.0, 24.0, 32.0, 48.0]))
        if mode == 0:
            candidate = truth + _smooth_noise(rng, dimensions, scale)
        elif mode == 1 and len(templates):
            candidate = templates[int(rng.integers(len(templates)))].copy()
            candidate += _smooth_noise(rng, dimensions, scale * 0.5)
        elif mode == 2:
            endpoint = float(rng.uniform(-80.0, 80.0))
            power = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
            candidate = endpoint * control_progress**power
            candidate += _smooth_noise(rng, dimensions, scale * 0.25)
        else:
            candidate = _smooth_noise(rng, dimensions, scale)
        candidates.append(candidate)

    candidates = np.clip(np.asarray(candidates), -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT)
    candidates = _deduplicate_coefficients(candidates)
    if len(candidates) < TRAIN_CANDIDATES:
        extra = np.stack(
            [truth + _smooth_noise(rng, dimensions, 8.0) for _ in range(TRAIN_CANDIDATES)]
        )
        candidates = _deduplicate_coefficients(
            np.vstack([candidates, np.clip(extra, -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT)])
        )
    # Preserve the exact fitted-truth and zero candidates, randomise the rest.
    fixed = candidates[:2]
    tail = candidates[2:]
    rng.shuffle(tail)
    return np.vstack([fixed, tail])[:TRAIN_CANDIDATES].astype(np.float32)


def _initial_coefficients(templates, dimensions, seed, random_count=FIXED_BANK_RANDOM):
    rng = np.random.default_rng(int(seed))
    progress = np.linspace(1.0 / SPLINE_CONTROLS, 1.0, int(dimensions), dtype=np.float32)
    candidates = [np.zeros(dimensions, np.float32)]
    candidates.extend(np.asarray(templates, np.float32))
    for endpoint in np.arange(-80.0, 80.1, 8.0):
        candidates.append(float(endpoint) * progress)
    for endpoint in (-64.0, -40.0, -24.0, 24.0, 40.0, 64.0):
        candidates.append(float(endpoint) * progress**0.5)
        candidates.append(float(endpoint) * progress**1.5)
        candidates.append(float(endpoint) * progress**2.0)
    for _ in range(int(random_count)):
        scale = float(rng.choice([8.0, 16.0, 24.0, 32.0, 48.0]))
        candidates.append(_smooth_noise(rng, dimensions, scale))
    candidates = np.clip(np.asarray(candidates), -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT)
    return _deduplicate_coefficients(candidates)


def _candidate_features(record, coefficients, basis, mode="full", reference_override=None):
    coefficients = np.asarray(coefficients, np.float32)
    correction = _coefficients_to_column_correction(coefficients, basis).astype(np.float32)
    candidate_tvt = record["eval_center_tvt"][None, :] + correction
    horizontal = record["eval_horizontal_gr"].astype(np.float32)
    horizontal_valid = record["eval_horizontal_valid"].astype(bool)

    reference_tvt = record["reference_tvt"]
    reference_gr = record["reference_gr"]
    if reference_override is not None:
        reference_tvt = reference_override["tvt"]
        reference_gr = reference_override["gr"]
    typewell = _interp_curve(
        reference_tvt, reference_gr, candidate_tvt.ravel()
    ).reshape(candidate_tvt.shape)
    typewell_valid = np.isfinite(typewell)
    typewell = np.nan_to_num(typewell, nan=0.0).astype(np.float32)
    calibrated = (
        record["calibration_slope"] * typewell + record["calibration_intercept"]
    ).astype(np.float32)

    self_reference = _interp_curve(
        record["self_tvt"], record["self_gr"], candidate_tvt.ravel()
    ).reshape(candidate_tvt.shape)
    self_valid = np.isfinite(self_reference)
    self_reference = np.nan_to_num(self_reference, nan=0.0).astype(np.float32)

    observed = np.broadcast_to(horizontal[None, :], candidate_tvt.shape).copy()
    observed_valid = np.broadcast_to(horizontal_valid[None, :], candidate_tvt.shape)
    match_valid = observed_valid & typewell_valid
    self_match_valid = observed_valid & self_valid
    if mode == "no_gr":
        observed[:] = 0.0
        observed_valid = np.zeros_like(observed_valid)
        match_valid = np.zeros_like(match_valid)
        self_match_valid = np.zeros_like(self_match_valid)
    elif mode not in ("full", "shuffled_typewell"):
        raise ValueError(mode)

    difference = (calibrated - observed) * match_valid
    absolute_difference = np.abs(difference)
    self_difference = (self_reference - observed) * self_match_valid
    self_absolute_difference = np.abs(self_difference)
    derivative_observed = np.gradient(observed, axis=1)
    derivative_typewell = np.gradient(calibrated, axis=1)
    derivative_difference = derivative_typewell - derivative_observed
    correction_d1 = np.gradient(correction, axis=1)
    correction_d2 = np.gradient(correction_d1, axis=1)
    progress = np.broadcast_to(
        np.linspace(0.0, 1.0, EVAL_COLUMNS, dtype=np.float32)[None, :],
        candidate_tvt.shape,
    )
    z_relative = np.broadcast_to(
        (
            (record["eval_column_z"] - record["column_z"][PREFIX_COLUMNS - 1])
            / 128.0
        )[None, :],
        candidate_tvt.shape,
    )
    center_dip = np.gradient(record["eval_center_tvt"], record["eval_column_md"])
    center_dip = np.broadcast_to(center_dip[None, :], candidate_tvt.shape)

    sequence = np.stack(
        [
            observed,
            observed_valid.astype(np.float32),
            typewell,
            typewell_valid.astype(np.float32),
            calibrated,
            difference,
            absolute_difference,
            observed * calibrated,
            self_reference,
            self_valid.astype(np.float32),
            self_difference,
            self_absolute_difference,
            derivative_observed,
            derivative_typewell,
            derivative_difference,
            uniform_filter1d(absolute_difference, 8, axis=1, mode="nearest"),
            uniform_filter1d(absolute_difference, 32, axis=1, mode="nearest"),
            uniform_filter1d(absolute_difference, 64, axis=1, mode="nearest"),
            _local_ncc(observed[0], calibrated, match_valid, 16),
            _local_ncc(observed[0], calibrated, match_valid, 64),
            correction / COEFFICIENT_CAP_FT,
            correction_d1 / 8.0,
            correction_d2 / 4.0,
            progress,
            z_relative,
            center_dip,
        ],
        axis=1,
    ).astype(np.float32)

    count = np.maximum(match_valid.sum(axis=1), 1)
    self_count = np.maximum(self_match_valid.sum(axis=1), 1)
    ncc = _weighted_ncc(observed[0], calibrated, match_valid)
    self_ncc = _weighted_ncc(observed[0], self_reference, self_match_valid)
    mae = absolute_difference.sum(axis=1) / count
    mismatch_rmse = np.sqrt((difference * difference).sum(axis=1) / count)
    self_mae = self_absolute_difference.sum(axis=1) / self_count
    abs_sorted = np.sort(absolute_difference, axis=1)
    q50 = abs_sorted[:, int(0.50 * (EVAL_COLUMNS - 1))]
    q90 = abs_sorted[:, int(0.90 * (EVAL_COLUMNS - 1))]
    total_variation = np.sum(np.abs(np.diff(correction, axis=1)), axis=1)
    curvature = np.sum(np.abs(np.diff(correction, n=2, axis=1)), axis=1)
    geometry = record["geometry"]
    known_fraction = float(geometry[15, 0]) if geometry.shape[0] > 15 else 0.25
    gr_missing = float(geometry[16, 0]) if geometry.shape[0] > 16 else 0.0
    summary = np.column_stack(
        [
            ncc,
            mae / 4.0,
            mismatch_rmse / 4.0,
            match_valid.mean(axis=1),
            self_ncc,
            self_mae / 4.0,
            self_match_valid.mean(axis=1),
            q50 / 4.0,
            q90 / 4.0,
            correction.mean(axis=1) / 64.0,
            correction.std(axis=1) / 64.0,
            np.max(np.abs(correction), axis=1) / 96.0,
            correction[:, -1] / 96.0,
            total_variation / 256.0,
            curvature / 128.0,
            coefficients.mean(axis=1) / 64.0,
            coefficients.std(axis=1) / 64.0,
            np.max(np.abs(coefficients), axis=1) / 96.0,
            np.full(len(coefficients), record["calibration_slope"]),
            np.full(len(coefficients), record["calibration_intercept"] / 4.0),
            np.full(len(coefficients), known_fraction),
            np.full(len(coefficients), gr_missing),
        ]
    ).astype(np.float32)
    sequence = np.nan_to_num(sequence, nan=0.0, posinf=8.0, neginf=-8.0)
    summary = np.nan_to_num(summary, nan=0.0, posinf=8.0, neginf=-8.0)
    return sequence, summary, correction


# ---------------------------------------------------------------------------
# Model and loss
# ---------------------------------------------------------------------------


class EnergyResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * int(dilation),
                dilation=int(dilation),
                groups=channels,
            ),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class WholeWellEnergyModel(nn.Module):
    def __init__(self, sequence_channels, summary_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(sequence_channels, 48, 7, stride=2, padding=3),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            EnergyResidualBlock(48, 1),
            EnergyResidualBlock(48, 2),
            EnergyResidualBlock(48, 4),
            nn.Conv1d(48, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            EnergyResidualBlock(64, 2),
            EnergyResidualBlock(64, 4),
        )
        self.summary = nn.Sequential(
            nn.Linear(summary_channels, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, 48),
            nn.GELU(),
        )
        self.energy = nn.Sequential(
            nn.Linear(64 * 3 + 48, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, sequence, summary):
        if sequence.ndim == 4:
            batch, candidates, channels, length = sequence.shape
            sequence = sequence.reshape(batch * candidates, channels, length)
            summary = summary.reshape(batch * candidates, summary.shape[-1])
            reshape = (batch, candidates)
        else:
            reshape = None
        hidden = self.sequence(sequence)
        pooled = torch.cat(
            [hidden.mean(dim=-1), hidden.amax(dim=-1), hidden.std(dim=-1)], dim=1
        )
        summary_hidden = self.summary(summary)
        energy = self.energy(torch.cat([pooled, summary_hidden], dim=1)).squeeze(-1)
        return energy if reshape is None else energy.reshape(*reshape)


def _energy_loss(energy, candidate_rmse, panel_weight):
    energy = energy.float()
    candidate_rmse = candidate_rmse.float()
    target_log = torch.log1p(candidate_rmse)
    regression = F.smooth_l1_loss(energy, target_log, beta=0.25)

    relative = candidate_rmse - candidate_rmse.amin(dim=1, keepdim=True)
    target_probability = torch.softmax(-relative / 2.0, dim=1)
    listwise = -(
        target_probability * F.log_softmax(-energy, dim=1)
    ).sum(dim=1)

    best = torch.argmin(candidate_rmse, dim=1)
    best_energy = energy.gather(1, best[:, None])
    margin = torch.clamp(relative / 4.0, 0.0, 2.0)
    pairwise = F.relu(margin - (energy - best_energy))
    pair_mask = torch.ones_like(pairwise).scatter(1, best[:, None], 0.0)
    pairwise = (pairwise * pair_mask).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)

    weight = panel_weight.float().clamp(0.20, 5.0)
    ranking = ((listwise + 0.60 * pairwise) * weight).mean()
    total = 0.45 * regression + ranking
    return total, {
        "regression": regression,
        "listwise": listwise.mean(),
        "pairwise": pairwise.mean(),
    }


class RealWarpDataset(Dataset):
    def __init__(self, records, templates, basis):
        self.records = records
        self.templates = templates
        self.basis = basis
        self.epoch = 0
        weights = np.asarray([record["panel_weight"] for record in records], np.float64)
        self.weight_scale = float(np.median(weights[weights > 0])) if np.any(weights > 0) else 1.0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        coefficients = _training_coefficients(
            record, self.templates, self.epoch, index
        )
        sequence, summary, correction = _candidate_features(
            record, coefficients, self.basis, mode="full"
        )
        candidate_rmse = np.sqrt(
            np.mean(
                (correction - record["eval_truth_residual"][None, :]) ** 2,
                axis=1,
            )
        ).astype(np.float32)
        panel_weight = np.float32(
            np.clip(record["panel_weight"] / max(self.weight_scale, 1e-6), 0.20, 5.0)
        )
        return {
            "sequence": torch.from_numpy(sequence),
            "summary": torch.from_numpy(summary),
            "candidate_rmse": torch.from_numpy(candidate_rmse),
            "panel_weight": torch.tensor(panel_weight, dtype=torch.float32),
        }


def _make_scaler(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


@torch.no_grad()
def _score_coefficients(
    model,
    record,
    coefficients,
    basis,
    device,
    mode="full",
    reference_override=None,
):
    model.eval()
    coefficients = np.asarray(coefficients, np.float32)
    energies = []
    for start in range(0, len(coefficients), SEARCH_SCORE_CHUNK):
        chunk = coefficients[start : start + SEARCH_SCORE_CHUNK]
        sequence, summary, _ = _candidate_features(
            record,
            chunk,
            basis,
            mode=mode,
            reference_override=reference_override,
        )
        sequence_tensor = torch.from_numpy(sequence).to(device)
        summary_tensor = torch.from_numpy(summary).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            energy = model(sequence_tensor, summary_tensor)
        energies.append(energy.float().cpu().numpy())
    return np.concatenate(energies).astype(np.float32)


def _fixed_bank_metrics(model, records, templates, basis, device, mode="full"):
    errors = []
    oracle_sse = 0.0
    total_rows = 0
    rank_correlations = []
    rows = []
    for index, record in enumerate(records):
        coefficients = _initial_coefficients(
            templates,
            basis.shape[1],
            SEED + 1709 * index,
            random_count=FIXED_BANK_RANDOM,
        )
        reference_override = None
        if mode == "shuffled_typewell":
            donor = records[(index + max(1, len(records) // 3)) % len(records)]
            reference_override = {
                "tvt": donor["reference_tvt"],
                "gr": donor["reference_gr"],
            }
        energy = _score_coefficients(
            model,
            record,
            coefficients,
            basis,
            device,
            mode=mode,
            reference_override=reference_override,
        )
        candidate_rmse = _candidate_exact_rmse(record, coefficients, basis)
        selected = int(np.argmin(energy))
        selected_prediction = _candidate_row_predictions(
            record, coefficients[selected : selected + 1], basis
        )[0]
        error = selected_prediction - record["eval_truth"]
        errors.append(error)
        oracle_sse += float(np.min(candidate_rmse) ** 2 * len(error))
        total_rows += len(error)
        correlation = spearmanr(energy, candidate_rmse).statistic
        if np.isfinite(correlation):
            rank_correlations.append(float(correlation))
        rows.append(
            {
                "well": record["well"],
                "rows": len(error),
                "selected_rmse": float(np.sqrt(np.mean(error**2))),
                "oracle_rmse": float(np.min(candidate_rmse)),
                "center_rmse": _rmse(record["eval_truth"], record["center_prediction"]),
                "rank_spearman": float(correlation) if np.isfinite(correlation) else np.nan,
                "candidates": len(coefficients),
            }
        )
    error = np.concatenate(errors)
    bywell = pd.DataFrame(rows)
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "oracle_rmse": float(np.sqrt(oracle_sse / max(total_rows, 1))),
        "median_well_rmse": float(bywell["selected_rmse"].median()),
        "p75_well_rmse": float(bywell["selected_rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["selected_rmse"].quantile(0.95)),
        "mean_rank_spearman": float(np.mean(rank_correlations)) if rank_correlations else np.nan,
    }
    metrics.update(_well_tail_metrics(bywell, "selected_rmse"))
    return metrics, bywell


def _train_energy_model(train_records, valid_records, templates, basis, device):
    _seed_everything(SEED)
    probe_coefficients = _training_coefficients(train_records[0], templates, 0, 0)
    probe_sequence, probe_summary, _ = _candidate_features(
        train_records[0], probe_coefficients, basis
    )
    sequence_channels = int(probe_sequence.shape[1])
    summary_channels = int(probe_summary.shape[1])
    model = WholeWellEnergyModel(sequence_channels, summary_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    scaler = _make_scaler(device)
    dataset = RealWarpDataset(train_records, templates, basis)
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    checkpoint_path = MODEL_DIR / "best_fold1.pt"
    best_rmse = np.inf
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, TRAIN_EPOCHS + 1):
        dataset.set_epoch(epoch)
        model.train()
        loss_rows = []
        epoch_started = time.time()
        for batch_index, batch in enumerate(loader, start=1):
            sequence = batch["sequence"].to(device, non_blocking=True)
            summary = batch["summary"].to(device, non_blocking=True)
            candidate_rmse = batch["candidate_rmse"].to(device, non_blocking=True)
            panel_weight = batch["panel_weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                energy = model(sequence, summary)
                loss, components = _energy_loss(
                    energy, candidate_rmse, panel_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in components.items()
                    },
                }
            )
            if batch_index % 500 == 0 or batch_index == len(loader):
                print(
                    "ENERGY TRAIN",
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(loader),
                        "mean_loss": float(
                            np.mean([row["loss"] for row in loss_rows[-500:]])
                        ),
                        "epoch_minutes": (time.time() - epoch_started) / 60.0,
                    },
                    flush=True,
                )
        scheduler.step()

        fixed_metrics, fixed_bywell = _fixed_bank_metrics(
            model, valid_records, templates, basis, device, mode="full"
        )
        epoch_row = {
            "epoch": epoch,
            **{
                f"train_{name}": float(np.mean([row[name] for row in loss_rows]))
                for name in loss_rows[0]
            },
            **{f"valid_fixed_{name}": value for name, value in fixed_metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(epoch_row)
        print("ENERGY EPOCH", epoch_row, flush=True)
        if fixed_metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = fixed_metrics["rmse"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": fixed_metrics,
                    "sequence_channels": sequence_channels,
                    "summary_channels": summary_channels,
                    "spline_controls": SPLINE_CONTROLS,
                    "templates": templates,
                    "seed": SEED,
                },
                checkpoint_path,
            )
            fixed_bywell.to_csv(OUT_DIR / "best_fixed_bank_bywell.csv", index=False)
        else:
            stale += 1
        if epoch >= 5 and stale >= PATIENCE:
            print(f"early stopping energy model at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    fixed_metrics, fixed_bywell = _fixed_bank_metrics(
        model, valid_records, templates, basis, device, mode="full"
    )
    fixed_bywell.to_csv(OUT_DIR / "fixed_bank_bywell.csv", index=False)
    return model, checkpoint, fixed_metrics, {
        "sequence_channels": sequence_channels,
        "summary_channels": summary_channels,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "runtime_minutes": (time.time() - started) / 60.0,
    }


# ---------------------------------------------------------------------------
# Deployment-style stochastic beam search
# ---------------------------------------------------------------------------


@torch.no_grad()
def _search_record(model, record, templates, basis, device, seed, mode="full", reference_override=None):
    rng = np.random.default_rng(int(seed))
    current = _initial_coefficients(
        templates, basis.shape[1], seed, random_count=FIXED_BANK_RANDOM
    )
    all_coefficients = []
    all_energies = []

    for stage, scale in enumerate((None,) + SEARCH_SCALES_FT):
        energy = _score_coefficients(
            model,
            record,
            current,
            basis,
            device,
            mode=mode,
            reference_override=reference_override,
        )
        all_coefficients.append(current)
        all_energies.append(energy)
        if scale is None:
            elite_indices = np.argsort(energy)[:SEARCH_ELITES]
        else:
            elite_indices = np.argsort(energy)[:SEARCH_ELITES]
        elite = current[elite_indices]
        if stage == len(SEARCH_SCALES_FT):
            break
        next_scale = SEARCH_SCALES_FT[stage]
        proposals = [elite]
        for candidate in elite:
            for _ in range(SEARCH_CHILDREN):
                proposals.append(
                    candidate[None, :]
                    + _smooth_noise(rng, basis.shape[1], next_scale)[None, :]
                )
        current = np.clip(
            np.vstack(proposals), -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT
        )
        current = _deduplicate_coefficients(current)

    coefficients = np.vstack(all_coefficients).astype(np.float32)
    energies = np.concatenate(all_energies).astype(np.float32)
    key = np.round(coefficients, 2)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    coefficients = coefficients[first]
    energies = energies[first]
    order = np.argsort(energies)
    selected = int(order[0])
    selected_prediction = _candidate_row_predictions(
        record, coefficients[selected : selected + 1], basis
    )[0]

    top = order[: min(POSTERIOR_TOPK, len(order))]
    relative_energy = energies[top] - energies[top[0]]
    posterior_weight = np.exp(-relative_energy / POSTERIOR_TEMPERATURE)
    posterior_weight /= max(float(posterior_weight.sum()), 1e-8)
    top_predictions = _candidate_row_predictions(record, coefficients[top], basis)
    posterior_prediction = posterior_weight @ top_predictions
    return {
        "coefficients": coefficients,
        "energies": energies,
        "prediction": selected_prediction.astype(np.float32),
        "posterior_prediction": posterior_prediction.astype(np.float32),
        "selected_coefficients": coefficients[selected],
    }


def _search_validation(model, records, templates, basis, device):
    selected_errors = []
    posterior_errors = []
    center_errors = []
    selected_shrunk_errors = []
    posterior_shrunk_errors = []
    oracle_sse = 0.0
    total_rows = 0
    bywell = []
    row_frames = []
    for index, record in enumerate(tqdm(records, desc="deployment-style real-warp search")):
        result = _search_record(
            model,
            record,
            templates,
            basis,
            device,
            seed=SEED + 10007 * index,
            mode="full",
        )
        truth = record["eval_truth"].astype(np.float64)
        selected_error = result["prediction"].astype(np.float64) - truth
        posterior_error = result["posterior_prediction"].astype(np.float64) - truth
        center_error = record["center_prediction"].astype(np.float64) - truth
        selected_shrunk, selected_alpha = _oracle_shrink(
            record["center_prediction"], result["prediction"], truth
        )
        posterior_shrunk, posterior_alpha = _oracle_shrink(
            record["center_prediction"], result["posterior_prediction"], truth
        )
        selected_shrunk_error = selected_shrunk.astype(np.float64) - truth
        posterior_shrunk_error = posterior_shrunk.astype(np.float64) - truth
        candidate_rmse = _candidate_exact_rmse(
            record, result["coefficients"], basis
        )
        oracle_sse += float(np.min(candidate_rmse) ** 2 * len(truth))
        total_rows += len(truth)
        selected_errors.append(selected_error)
        posterior_errors.append(posterior_error)
        center_errors.append(center_error)
        selected_shrunk_errors.append(selected_shrunk_error)
        posterior_shrunk_errors.append(posterior_shrunk_error)
        bywell.append(
            {
                "well": record["well"],
                "rows": len(truth),
                "center_rmse": float(np.sqrt(np.mean(center_error**2))),
                "selected_rmse": float(np.sqrt(np.mean(selected_error**2))),
                "posterior_rmse": float(np.sqrt(np.mean(posterior_error**2))),
                "selected_oracle_shrunk_rmse": float(
                    np.sqrt(np.mean(selected_shrunk_error**2))
                ),
                "posterior_oracle_shrunk_rmse": float(
                    np.sqrt(np.mean(posterior_shrunk_error**2))
                ),
                "selected_oracle_alpha": selected_alpha,
                "posterior_oracle_alpha": posterior_alpha,
                "search_oracle_rmse": float(np.min(candidate_rmse)),
                "candidates_scored": len(result["coefficients"]),
                "selected_energy": float(np.min(result["energies"])),
            }
        )
        row_frames.append(
            pd.DataFrame(
                {
                    "well": record["well"],
                    "row_idx": record["eval_rows"],
                    "truth": truth,
                    "center": record["center_prediction"],
                    "prediction": result["prediction"],
                    "posterior_prediction": result["posterior_prediction"],
                }
            )
        )
    selected_error = np.concatenate(selected_errors)
    posterior_error = np.concatenate(posterior_errors)
    center_error = np.concatenate(center_errors)
    selected_shrunk_error = np.concatenate(selected_shrunk_errors)
    posterior_shrunk_error = np.concatenate(posterior_shrunk_errors)
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": int(len(selected_error)),
        "wells": int(len(bywell)),
        "center_rmse": float(np.sqrt(np.mean(center_error**2))),
        "selected_rmse": float(np.sqrt(np.mean(selected_error**2))),
        "posterior_rmse": float(np.sqrt(np.mean(posterior_error**2))),
        "selected_oracle_shrunk_rmse": float(
            np.sqrt(np.mean(selected_shrunk_error**2))
        ),
        "posterior_oracle_shrunk_rmse": float(
            np.sqrt(np.mean(posterior_shrunk_error**2))
        ),
        "search_oracle_rmse": float(np.sqrt(oracle_sse / max(total_rows, 1))),
        "median_well_rmse": float(bywell["selected_rmse"].median()),
        "p75_well_rmse": float(bywell["selected_rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["selected_rmse"].quantile(0.95)),
        "selected_oracle_alpha_median": float(
            bywell["selected_oracle_alpha"].median()
        ),
        "posterior_oracle_alpha_median": float(
            bywell["posterior_oracle_alpha"].median()
        ),
        "positive_vs_center_fraction": float(
            np.mean(bywell["selected_rmse"] < bywell["center_rmse"])
        ),
    }
    metrics.update(_well_tail_metrics(bywell, "selected_rmse"))
    metrics.update(
        {
            f"posterior_{name}": value
            for name, value in _well_tail_metrics(bywell, "posterior_rmse").items()
        }
    )
    return metrics, bywell, pd.concat(row_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if not PANEL_ROOT.exists():
    raise FileNotFoundError(
        f"Missing {PANEL_ROOT}. Run only the panel-build portion of "
        "tvt_corrnet_heel_conditioned_onefold_v4_cell.py first."
    )
fold_path = _first_existing(FOLD_CANDIDATES)
if fold_path is None:
    raise FileNotFoundError("fold metadata not found")

fold_frame = pd.read_csv(fold_path, dtype={"well": str})
fold_frame = fold_frame.groupby("well", as_index=False)["fold"].first()
fold_frame["well"] = fold_frame["well"].astype(str)
fold_frame["fold"] = fold_frame["fold"].astype(int)
fold_map = dict(zip(fold_frame["well"], fold_frame["fold"]))

usable = None
if PANEL_GUARDS.exists():
    guard_frame = pd.read_csv(PANEL_GUARDS, dtype={"well": str, "tag": str})
    if "usable" in guard_frame.columns:
        guard_frame["usable"] = guard_frame["usable"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        usable = set(
            guard_frame.loc[guard_frame["usable"], ["well", "tag"]]
            .astype(str)
            .agg("__".join, axis=1)
        )

panel_rows = []
for path in sorted(PANEL_ROOT.glob("*.npz")):
    stem = path.stem
    if "__" not in stem:
        continue
    well, tag = stem.rsplit("__", 1)
    if well not in fold_map or tag not in TRAIN_TAGS:
        continue
    if usable is not None and f"{well}__{tag}" not in usable:
        continue
    if fold_map[well] == VALID_FOLD and tag != "official":
        continue
    panel_rows.append(
        {"path": path, "well": well, "tag": tag, "fold": fold_map[well]}
    )
panel_manifest = pd.DataFrame(panel_rows)
if panel_manifest.empty:
    raise RuntimeError("no usable v4 panels found")

train_manifest = panel_manifest[panel_manifest["fold"] != VALID_FOLD].copy()
valid_manifest = panel_manifest[
    (panel_manifest["fold"] == VALID_FOLD) & (panel_manifest["tag"] == "official")
].copy()
expected_valid = int((fold_frame["fold"] == VALID_FOLD).sum())
if len(valid_manifest) != expected_valid:
    raise RuntimeError(
        f"valid official panels {len(valid_manifest)} != expected {expected_valid}"
    )

print(
    {
        "experiment": "fullwell_realwarp_energy_onefold_v0",
        "sample_unit": "complete well/pseudo-PS panel",
        "standalone": True,
        "data_root": str(DATA_ROOT),
        "panel_root": str(PANEL_ROOT),
        "fold_source": str(fold_path),
        "train_panels": len(train_manifest),
        "train_wells": train_manifest["well"].nunique(),
        "valid_wells": len(valid_manifest),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    flush=True,
)

references = ReferenceStore(panel_manifest["well"].unique())
basis = _spline_basis(EVAL_COLUMNS, SPLINE_CONTROLS)
train_records = [
    _prepare_record(_record_from_panel(row.path, row.fold), references, basis)
    for row in tqdm(train_manifest.itertuples(index=False), total=len(train_manifest), desc="load train panels")
]
valid_records = [
    _prepare_record(_record_from_panel(row.path, row.fold), references, basis)
    for row in tqdm(valid_manifest.itertuples(index=False), total=len(valid_manifest), desc="load valid panels")
]

# Exact representational oracle, measured on untouched fold-1 wells.
oracle_rows = []
for controls in SPLINE_CONTROL_OPTIONS:
    option_basis = _spline_basis(EVAL_COLUMNS, controls)
    errors = []
    bywell = []
    for record in valid_records:
        coefficients = _fit_coefficients(record["eval_truth_residual"], option_basis)
        prediction = _candidate_row_predictions(
            record, coefficients[None, :], option_basis
        )[0]
        error = prediction - record["eval_truth"]
        errors.append(error)
        bywell.append(float(np.sqrt(np.mean(error**2))))
    error = np.concatenate(errors)
    oracle_rows.append(
        {
            "controls": controls,
            "free_dimensions": controls - 1,
            "pooled_rmse": float(np.sqrt(np.mean(error**2))),
            "median_well_rmse": float(np.median(bywell)),
            "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        }
    )
spline_oracle = pd.DataFrame(oracle_rows)
spline_oracle.to_csv(OUT_DIR / "spline_representation_oracle.csv", index=False)
print("\nSPLINE REPRESENTATION ORACLE")
print(spline_oracle.to_string(index=False))
selected_representation_rmse = float(
    spline_oracle.loc[spline_oracle["controls"] == SPLINE_CONTROLS, "pooled_rmse"].iloc[0]
)
if selected_representation_rmse > REPRESENTATION_GATE_FT:
    verdict = {
        "experiment": "fullwell_realwarp_energy_onefold_v0",
        "stage": "representation_audit",
        "spline_controls": SPLINE_CONTROLS,
        "representation_oracle_rmse": selected_representation_rmse,
        "strict_pass": False,
        "next": "FAIL -> increase spline controls before training",
    }
    with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)
    raise RuntimeError(json.dumps(verdict, indent=2))

# The global path templates use only outer-train truth coefficients.
train_coefficients = np.stack(
    [record["truth_coefficients"] for record in train_records]
).astype(np.float32)
template_model = MiniBatchKMeans(
    n_clusters=min(TEMPLATE_CLUSTERS, max(16, len(train_coefficients) // 20)),
    random_state=SEED,
    n_init=8,
    batch_size=256,
    max_iter=500,
).fit(train_coefficients)
templates = np.clip(
    template_model.cluster_centers_, -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT
).astype(np.float32)
np.save(OUT_DIR / "foldclean_spline_templates.npy", templates)

# Truth-independent template-bank ceiling before any learned search.
initial_oracle_sse = 0.0
initial_rows = 0
initial_bywell = []
for index, record in enumerate(tqdm(valid_records, desc="initial template-bank oracle")):
    coefficients = _initial_coefficients(
        templates, basis.shape[1], SEED + 1709 * index
    )
    candidate_rmse = _candidate_exact_rmse(record, coefficients, basis)
    best = float(np.min(candidate_rmse))
    initial_oracle_sse += best**2 * len(record["eval_truth"])
    initial_rows += len(record["eval_truth"])
    initial_bywell.append(
        {
            "well": record["well"],
            "oracle_rmse": best,
            "candidates": len(coefficients),
        }
    )
initial_bank_oracle_rmse = float(
    np.sqrt(initial_oracle_sse / max(initial_rows, 1))
)
pd.DataFrame(initial_bywell).to_csv(
    OUT_DIR / "initial_template_bank_oracle_bywell.csv", index=False
)
print(
    "\nINITIAL TEMPLATE BANK",
    {
        "templates": len(templates),
        "oracle_rmse": initial_bank_oracle_rmse,
        "gate_reference": FIXED_BANK_ORACLE_GATE_FT,
    },
    flush=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
model, checkpoint, fixed_metrics, model_info = _train_energy_model(
    train_records, valid_records, templates, basis, device
)
print("\nBEST FIXED-BANK METRICS")
print(json.dumps(fixed_metrics, indent=2))

search_started = time.time()
search_metrics, search_bywell, search_rows = _search_validation(
    model, valid_records, templates, basis, device
)
search_bywell.to_csv(OUT_DIR / "search_bywell.csv", index=False)
search_rows.to_pickle(OUT_DIR / "fold1_predictions.pkl")
search_metrics["runtime_minutes"] = (time.time() - search_started) / 60.0
print("\nDEPLOYMENT SEARCH")
print(json.dumps(search_metrics, indent=2))

primary_rmse = min(search_metrics["selected_rmse"], search_metrics["posterior_rmse"])
primary_decoder = (
    "selected" if search_metrics["selected_rmse"] <= search_metrics["posterior_rmse"] else "posterior"
)
primary_prefix = "" if primary_decoder == "selected" else "posterior_"
primary_median_well_rmse = float(
    search_metrics[f"{primary_prefix}p50_well_rmse"]
)
primary_p75_well_rmse = float(
    search_metrics[f"{primary_prefix}p75_well_rmse"]
)
primary_p95_well_rmse = float(
    search_metrics[f"{primary_prefix}p95_well_rmse"]
)

# Fixed-bank controls are enough to establish whether the scorer actually uses
# matching evidence; running three complete stochastic searches would triple
# the gateway cost without changing this causal question.
control_rows = [{"mode": "full", **fixed_metrics}]
if primary_rmse < 8.5 or fixed_metrics["rmse"] < 8.5:
    for mode in ("no_gr", "shuffled_typewell"):
        metrics, bywell = _fixed_bank_metrics(
            model, valid_records, templates, basis, device, mode=mode
        )
        bywell.to_csv(OUT_DIR / f"fixed_bank_{mode}_bywell.csv", index=False)
        control_rows.append({"mode": mode, **metrics})
controls = pd.DataFrame(control_rows)
full_fixed_rmse = float(controls.loc[controls["mode"] == "full", "rmse"].iloc[0])
controls["degradation_vs_full"] = controls["rmse"] - full_fixed_rmse
controls.to_csv(OUT_DIR / "fixed_bank_interventions.csv", index=False)
print("\nFIXED-BANK INTERVENTIONS")
print(controls.to_string(index=False))

no_gr_degradation = (
    float(controls.loc[controls["mode"] == "no_gr", "degradation_vs_full"].iloc[0])
    if "no_gr" in set(controls["mode"])
    else None
)
shuffled_degradation = (
    float(
        controls.loc[
            controls["mode"] == "shuffled_typewell", "degradation_vs_full"
        ].iloc[0]
    )
    if "shuffled_typewell" in set(controls["mode"])
    else None
)
causal_pass = bool(
    no_gr_degradation is not None
    and shuffled_degradation is not None
    and no_gr_degradation >= CAUSAL_GATE_FT
    and shuffled_degradation >= CAUSAL_GATE_FT
)
architecture_pass = bool(
    primary_rmse <= ARCHITECTURE_GATE_FT
    and search_metrics["search_oracle_rmse"] <= TARGET_GATE_FT
)
core_pass = bool(primary_median_well_rmse <= 5.0)
tail_pass = bool(primary_p95_well_rmse <= 12.0)
target_pass = bool(primary_rmse <= TARGET_GATE_FT)
strict_pass = bool(architecture_pass and causal_pass)
verdict = {
    "experiment": "fullwell_realwarp_energy_onefold_v0",
    "sample_unit": "complete well/pseudo-PS panel",
    "standalone": True,
    "fold": VALID_FOLD,
    "train_wells": int(train_manifest["well"].nunique()),
    "train_panels": int(len(train_records)),
    "valid_wells": int(len(valid_records)),
    "spline_controls": SPLINE_CONTROLS,
    "free_spline_dimensions": int(basis.shape[1]),
    "representation_oracle_rmse": selected_representation_rmse,
    "initial_template_bank_oracle_rmse": initial_bank_oracle_rmse,
    "model": model_info,
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "fixed_bank": fixed_metrics,
    "search": search_metrics,
    "primary_decoder": primary_decoder,
    "primary_rmse": float(primary_rmse),
    "primary_median_well_rmse": primary_median_well_rmse,
    "primary_p75_well_rmse": primary_p75_well_rmse,
    "primary_p95_well_rmse": primary_p95_well_rmse,
    "no_gr_degradation": no_gr_degradation,
    "shuffled_typewell_degradation": shuffled_degradation,
    "core_pass_median_le_5": core_pass,
    "tail_pass_p95_le_12": tail_pass,
    "architecture_pass": architecture_pass,
    "target_5p5_pass": target_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "next": (
        "PASS <=5.5 -> repeat seeds, then five outer folds"
        if target_pass and causal_pass
        else (
            "PASS architecture -> add one hard-negative mining round"
            if strict_pass
            else (
                "SEARCH ORACLE PASS / SELECTOR FAIL -> hard-negative mining"
                if search_metrics["search_oracle_rmse"] <= TARGET_GATE_FT
                else "FAIL -> candidate generation still limits the standalone model"
            )
        )
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)

del train_records, valid_records, references
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
