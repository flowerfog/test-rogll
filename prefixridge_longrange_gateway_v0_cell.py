"""Fold-honest PrefixRidge long-range edge gateway.

This cell implements the P0.75/P1 decision gate from PrefixRidge-SGraph v1.1.
It does not build the final graph solver.  It first tests the premise that a
closed-form same-well prefix adapter transfers to long-range suffix Delta-S
edges.

Protocol
--------
* Global edge mean: LightGBM trained on Folds 3-5, tuned on Fold 2.
* Ridge dose: selected on Fold 2 only.
* Final global mean: refit on Folds 2-5 for the frozen iteration count.
* Final decision: untouched Fold 1 with no-adapter, same-well, label-shuffle,
  and matched-other-well controls.

All inputs are legal at test time.  S = TVT + Z is used for every edge target;
Z is read from the existing native-resolution legal-input record cache.
"""

from dataclasses import dataclass
from pathlib import Path
import gc
import json
import random
import shutil
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "prefixridge_longrange_gateway_v0"
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

SEED = 20260801
TUNE_FOLD = 2
TEST_FOLD = 1
NODE_STEP_FT = 32.0
EDGE_STEPS = (1, 2, 4, 8, 16, 32, 64)
PREFIX_CONTEXT_FRACTION = 0.70
RIDGE_LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
RIDGE_JITTER = 1e-8

LGB_MAX_ROUNDS = 2400
LGB_LEARNING_RATE = 0.025
LGB_NUM_LEAVES = 31
LGB_MIN_CHILD_SAMPLES = 160

LATE_PREFIX_MSE_GAIN_GATE = 0.10
FAR_SUFFIX_MSE_GAIN_GATE = 0.08
POSITIVE_WELL_FRACTION_GATE = 0.60
GLOBAL_VS_ZERO_MSE_GAIN_GATE = 0.05


FEATURE_COLUMNS = [
    "length_kft",
    "log_length",
    "start_progress",
    "end_progress",
    "start_ps_progress",
    "end_ps_progress",
    "delta_z_hft",
    "z_rate",
    "gr_start",
    "gr_end",
    "gr_delta",
    "gr_abs_delta",
    "gr_mean",
    "gr_std",
    "gr_coverage",
    "dx_start",
    "dx_end",
    "dx_delta",
    "dy_start",
    "dy_end",
    "dy_delta",
    "dz_start",
    "dz_end",
    "dz_delta",
    "d2z_start",
    "d2z_end",
    "d2z_delta",
    "family_star",
    "scale_log2",
]

ADAPTER_FEATURE_COLUMNS = [
    "length_kft",
    "log_length",
    "start_ps_progress",
    "end_ps_progress",
    "delta_z_hft",
    "z_rate",
    "gr_delta",
    "gr_mean",
    "gr_std",
    "gr_coverage",
    "dz_start",
    "dz_end",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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


def _rmse(truth, prediction, weight=None):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    squared = (truth - prediction) ** 2
    if weight is None:
        return float(np.sqrt(np.mean(squared)))
    weight = np.asarray(weight, np.float64)
    return float(np.sqrt(np.sum(weight * squared) / np.maximum(np.sum(weight), 1e-12)))


def _horizontal_wells():
    paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
    wells = [path.name.split("__", 1)[0] for path in paths]
    if len(wells) != 773:
        raise FileNotFoundError(f"Expected 773 wells; found {len(wells)} under {DATA_ROOT / 'train'}")
    return wells


def _load_folds(wells):
    path = next((candidate for candidate in FOLD_CANDIDATES if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Cannot find fold-clean well metadata: "
            + repr([str(candidate) for candidate in FOLD_CANDIDATES])
        )
    frame = pd.read_csv(path, dtype=str)
    well_column = next((column for column in ("well", "well_id", "WELL") if column in frame), None)
    fold_column = next((column for column in ("fold", "outer_fold", "Fold") if column in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Unexpected fold columns: {frame.columns.tolist()}")
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
    return frame.sort_values(["fold", "well"]).reset_index(drop=True), path


# ---------------------------------------------------------------------------
# Record and edge construction
# ---------------------------------------------------------------------------

@dataclass
class WellRecord:
    well: str
    md: np.ndarray
    gr: np.ndarray
    gr_valid: np.ndarray
    condition: np.ndarray
    truth_tvt: np.ndarray
    anchor_index: int
    anchor_tvt: float
    anchor_md: float
    s_relative: np.ndarray


@dataclass
class WellMeta:
    well: str
    fold: int
    prefix_span: float
    eval_span: float
    prefix_gr_iqr: float
    prefix_coverage: float
    known_fraction: float
    split_md: float


def _load_record(well):
    path = RECORD_DIR / f"{well}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run native_resolution_dualref_capacity_v0_cell.py first."
        )
    with np.load(path, allow_pickle=False) as data:
        horizontal = data["horizontal"].astype(np.float32)
        condition = data["condition"].astype(np.float32)
        md = data["md"].astype(np.float64)
        truth = data["truth"].astype(np.float64)
        anchor_index = int(data["anchor_index"][0])
        anchor_tvt = float(data["anchor_tvt"][0])
    if condition.shape[0] < 10 or horizontal.shape[0] < 2:
        raise RuntimeError(f"{well}: stale native record shapes {horizontal.shape}, {condition.shape}")
    z_relative = condition[4].astype(np.float64) * 512.0
    s_relative = (truth - anchor_tvt) + z_relative
    if not np.isfinite(s_relative).all():
        raise RuntimeError(f"{well}: non-finite anchored S truth")
    return WellRecord(
        well=well,
        md=md,
        gr=horizontal[0].astype(np.float64),
        gr_valid=horizontal[1].astype(np.float64),
        condition=condition.astype(np.float64),
        truth_tvt=truth,
        anchor_index=anchor_index,
        anchor_tvt=anchor_tvt,
        anchor_md=float(md[anchor_index]),
        s_relative=s_relative,
    )


def _node_md(record):
    before = int(np.floor((record.anchor_md - record.md[0]) / NODE_STEP_FT))
    after = int(np.floor((record.md[-1] - record.anchor_md) / NODE_STEP_FT))
    prefix = record.anchor_md - np.arange(before, 0, -1, dtype=np.float64) * NODE_STEP_FT
    suffix = record.anchor_md + np.arange(1, after + 1, dtype=np.float64) * NODE_STEP_FT
    nodes = np.r_[prefix, record.anchor_md, suffix]
    if len(nodes) < 40:
        raise RuntimeError(f"{record.well}: only {len(nodes)} 32-ft nodes")
    return nodes, len(prefix)


def _interp(axis, values, queries):
    return np.interp(queries, axis, values).astype(np.float64)


def _interval_stats(record, start_md, end_md):
    start_row = np.searchsorted(record.md, start_md, side="left")
    end_row = np.searchsorted(record.md, end_md, side="right") - 1
    start_row = np.clip(start_row, 0, len(record.md) - 1)
    end_row = np.clip(end_row, start_row, len(record.md) - 1)
    valid = record.gr_valid
    value = record.gr
    cumulative_count = np.r_[0.0, np.cumsum(valid)]
    cumulative_sum = np.r_[0.0, np.cumsum(valid * value)]
    cumulative_square = np.r_[0.0, np.cumsum(valid * value * value)]
    count = cumulative_count[end_row + 1] - cumulative_count[start_row]
    total = cumulative_sum[end_row + 1] - cumulative_sum[start_row]
    square = cumulative_square[end_row + 1] - cumulative_square[start_row]
    rows = np.maximum(end_row - start_row + 1, 1)
    mean = total / np.maximum(count, 1.0)
    variance = np.maximum(square / np.maximum(count, 1.0) - mean * mean, 0.0)
    coverage = count / rows
    return mean, np.sqrt(variance), coverage


def _build_edges(record, fold):
    node_md, anchor_node = _node_md(record)
    n_nodes = len(node_md)
    node_gr = _interp(record.md, record.gr, node_md)
    node_condition = np.stack(
        [_interp(record.md, record.condition[channel], node_md) for channel in range(10)],
        axis=1,
    )
    node_s = _interp(record.md, record.s_relative, node_md)

    edge_i = []
    edge_j = []
    edge_family = []
    edge_step = []
    for step in EDGE_STEPS:
        count = n_nodes - step
        if count <= 0:
            continue
        edge_i.extend(np.arange(count).tolist())
        edge_j.extend((np.arange(count) + step).tolist())
        edge_family.extend([f"skip_{step}"] * count)
        edge_step.extend([step] * count)

    existing_anchor_steps = set(EDGE_STEPS)
    for destination in range(anchor_node + 1, n_nodes):
        if destination - anchor_node in existing_anchor_steps:
            continue
        edge_i.append(anchor_node)
        edge_j.append(destination)
        edge_family.append("anchor_star")
        edge_step.append(0)

    edge_i = np.asarray(edge_i, np.int32)
    edge_j = np.asarray(edge_j, np.int32)
    start_md = node_md[edge_i]
    end_md = node_md[edge_j]
    length = end_md - start_md
    if np.any(length <= 0):
        raise RuntimeError(f"{record.well}: non-positive edge length")
    gr_mean, gr_std, gr_coverage = _interval_stats(record, start_md, end_md)

    full_span = max(record.md[-1] - record.md[0], 1.0)
    eval_span = max(record.md[-1] - record.anchor_md, 1.0)
    start_progress = (start_md - record.md[0]) / full_span
    end_progress = (end_md - record.md[0]) / full_span
    start_ps_progress = (start_md - record.anchor_md) / eval_span
    end_ps_progress = (end_md - record.anchor_md) / eval_span
    z_start = node_condition[edge_i, 4] * 512.0
    z_end = node_condition[edge_j, 4] * 512.0
    delta_z = z_end - z_start

    dx_start = node_condition[edge_i, 5]
    dx_end = node_condition[edge_j, 5]
    dy_start = node_condition[edge_i, 6]
    dy_end = node_condition[edge_j, 6]
    dz_start = node_condition[edge_i, 7]
    dz_end = node_condition[edge_j, 7]
    d2z_start = node_condition[edge_i, 8]
    d2z_end = node_condition[edge_j, 8]

    family_star = np.asarray([family == "anchor_star" for family in edge_family], np.float64)
    scale_log2 = np.log2(np.maximum(length / NODE_STEP_FT, 1.0))
    split_md = record.md[0] + PREFIX_CONTEXT_FRACTION * (record.anchor_md - record.md[0])
    region = np.where(
        end_md <= record.anchor_md + 1e-6,
        "prefix",
        np.where(start_md >= record.anchor_md - 1e-6, "suffix", "cross"),
    )
    prefix_role = np.where(
        end_md <= split_md + 1e-6,
        "early_context",
        np.where(
            (start_md >= split_md - 1e-6) & (end_md <= record.anchor_md + 1e-6),
            "late_query",
            "other",
        ),
    )

    frame = pd.DataFrame(
        {
            "well": record.well,
            "fold": int(fold),
            "node_i": edge_i,
            "node_j": edge_j,
            "family": edge_family,
            "edge_step": edge_step,
            "region": region,
            "prefix_role": prefix_role,
            "start_md": start_md,
            "end_md": end_md,
            "end_distance": end_md - record.anchor_md,
            "target_delta_s": node_s[edge_j] - node_s[edge_i],
            "length_kft": length / 1000.0,
            "log_length": np.log1p(length) / np.log(2049.0),
            "start_progress": start_progress,
            "end_progress": end_progress,
            "start_ps_progress": start_ps_progress,
            "end_ps_progress": end_ps_progress,
            "delta_z_hft": delta_z / 100.0,
            "z_rate": delta_z / length,
            "gr_start": node_gr[edge_i],
            "gr_end": node_gr[edge_j],
            "gr_delta": node_gr[edge_j] - node_gr[edge_i],
            "gr_abs_delta": np.abs(node_gr[edge_j] - node_gr[edge_i]),
            "gr_mean": gr_mean,
            "gr_std": gr_std,
            "gr_coverage": gr_coverage,
            "dx_start": dx_start,
            "dx_end": dx_end,
            "dx_delta": dx_end - dx_start,
            "dy_start": dy_start,
            "dy_end": dy_end,
            "dy_delta": dy_end - dy_start,
            "dz_start": dz_start,
            "dz_end": dz_end,
            "dz_delta": dz_end - dz_start,
            "d2z_start": d2z_start,
            "d2z_end": d2z_end,
            "d2z_delta": d2z_end - d2z_start,
            "family_star": family_star,
            "scale_log2": scale_log2,
        }
    )
    frame["scale_group"] = np.where(
        frame["family_star"].eq(1.0),
        "anchor_star",
        frame["edge_step"].map(lambda value: f"skip_{int(value)}"),
    )
    group_count = frame.groupby(["well", "scale_group"])["well"].transform("size")
    frame["edge_weight"] = 1.0 / group_count.to_numpy(np.float64)
    frame["edge_weight"] *= len(frame) / frame["edge_weight"].sum()

    prefix_rows = np.arange(0, record.anchor_index + 1)
    prefix_values = record.gr[prefix_rows][record.gr_valid[prefix_rows] > 0.5]
    gr_iqr = float(np.quantile(prefix_values, 0.75) - np.quantile(prefix_values, 0.25)) if len(prefix_values) else 0.0
    meta = WellMeta(
        well=record.well,
        fold=int(fold),
        prefix_span=float(record.anchor_md - record.md[0]),
        eval_span=float(record.md[-1] - record.anchor_md),
        prefix_gr_iqr=gr_iqr,
        prefix_coverage=float(np.mean(record.gr_valid[prefix_rows] > 0.5)),
        known_fraction=float((record.anchor_index + 1) / len(record.md)),
        split_md=float(split_md),
    )
    return frame, meta


# ---------------------------------------------------------------------------
# Global mean, adapter and controls
# ---------------------------------------------------------------------------

def _fit_global(train, valid=None, rounds=None):
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=int(rounds or LGB_MAX_ROUNDS),
        learning_rate=LGB_LEARNING_RATE,
        num_leaves=LGB_NUM_LEAVES,
        min_child_samples=LGB_MIN_CHILD_SAMPLES,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {
        "sample_weight": train["edge_weight"].to_numpy(np.float64),
    }
    if valid is not None:
        fit_kwargs.update(
            {
                "eval_set": [(valid[FEATURE_COLUMNS], valid["target_delta_s"])],
                "eval_sample_weight": [valid["edge_weight"].to_numpy(np.float64)],
                "callbacks": [lgb.early_stopping(120, verbose=False)],
            }
        )
    model.fit(train[FEATURE_COLUMNS], train["target_delta_s"], **fit_kwargs)
    return model


def _fit_adapter_statistics(frame):
    values = frame[ADAPTER_FEATURE_COLUMNS].to_numpy(np.float64)
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean, scale


def _phi(frame, mean, scale):
    values = frame[ADAPTER_FEATURE_COLUMNS].to_numpy(np.float64)
    values = np.nan_to_num((values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([np.ones(len(values), np.float64), values])


def _ridge_adapter(frame, base_prediction, mask, ridge_lambda, mean, scale, shuffle=False):
    positions = np.flatnonzero(np.asarray(mask, bool))
    dimension = 1 + len(ADAPTER_FEATURE_COLUMNS)
    if len(positions) < max(24, 2 * dimension):
        return np.zeros(dimension, np.float64), len(positions), np.nan
    design = _phi(frame.iloc[positions], mean, scale)
    residual = (
        frame["target_delta_s"].to_numpy(np.float64)[positions]
        - np.asarray(base_prediction, np.float64)[positions]
    )
    if shuffle:
        seed = (SEED + zlib.crc32(str(frame["well"].iloc[0]).encode("ascii"))) & 0xFFFFFFFF
        residual = residual[np.random.default_rng(seed).permutation(len(residual))]
    weight = frame["edge_weight"].to_numpy(np.float64)[positions]
    weight = weight / max(np.mean(weight), 1e-12)
    normal = design.T @ (weight[:, None] * design)
    rhs = design.T @ (weight * residual)
    penalty = np.eye(dimension, dtype=np.float64) * float(ridge_lambda)
    penalty[0, 0] *= 0.10
    condition = float(np.linalg.cond(normal + penalty))
    try:
        coefficient = np.linalg.solve(
            normal + penalty + RIDGE_JITTER * np.eye(dimension), rhs
        )
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(
            normal + penalty + 1e-5 * np.eye(dimension), rhs, rcond=None
        )[0]
    return coefficient, len(positions), condition


def _adapter_prediction(frame, base_prediction, coefficient, mean, scale):
    return np.asarray(base_prediction, np.float64) + _phi(frame, mean, scale) @ coefficient


def _metadata_matrix(meta_frame):
    columns = ["prefix_span", "eval_span", "prefix_gr_iqr", "prefix_coverage", "known_fraction"]
    values = meta_frame[columns].to_numpy(np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (values - mean) / scale


def _matched_donors(meta_frame):
    values = _metadata_matrix(meta_frame)
    distance = np.sum((values[:, None] - values[None]) ** 2, axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    wells = meta_frame["well"].astype(str).tolist()
    return {well: wells[int(index)] for well, index in zip(wells, nearest)}


def _build_variant_predictions(frame, base_prediction, ridge_lambda, mean, scale, meta_frame):
    predictions = {}
    coefficients_full = {}
    coefficients_early = {}
    coefficients_shuffle_full = {}
    coefficients_shuffle_early = {}
    diagnostics = []

    for well, group in frame.groupby("well", sort=False):
        index = group.index.to_numpy()
        local_base = np.asarray(base_prediction[index], np.float64)
        prefix = group["region"].eq("prefix").to_numpy()
        early = group["prefix_role"].eq("early_context").to_numpy()
        coefficient_full, count_full, condition_full = _ridge_adapter(
            group, local_base, prefix, ridge_lambda, mean, scale, shuffle=False
        )
        coefficient_early, count_early, condition_early = _ridge_adapter(
            group, local_base, early, ridge_lambda, mean, scale, shuffle=False
        )
        shuffle_full, _, _ = _ridge_adapter(
            group, local_base, prefix, ridge_lambda, mean, scale, shuffle=True
        )
        shuffle_early, _, _ = _ridge_adapter(
            group, local_base, early, ridge_lambda, mean, scale, shuffle=True
        )
        coefficients_full[well] = coefficient_full
        coefficients_early[well] = coefficient_early
        coefficients_shuffle_full[well] = shuffle_full
        coefficients_shuffle_early[well] = shuffle_early
        diagnostics.append(
            {
                "well": well,
                "full_context_edges": count_full,
                "early_context_edges": count_early,
                "full_condition_number": condition_full,
                "early_condition_number": condition_early,
            }
        )

    donors = _matched_donors(meta_frame.reset_index(drop=True))
    for well, group in frame.groupby("well", sort=False):
        index = group.index.to_numpy()
        local_base = np.asarray(base_prediction[index], np.float64)
        predictions.setdefault("global", np.zeros(len(frame), np.float64))[index] = local_base
        predictions.setdefault("same_full", np.zeros(len(frame), np.float64))[index] = _adapter_prediction(
            group, local_base, coefficients_full[well], mean, scale
        )
        predictions.setdefault("same_early", np.zeros(len(frame), np.float64))[index] = _adapter_prediction(
            group, local_base, coefficients_early[well], mean, scale
        )
        predictions.setdefault("shuffle_full", np.zeros(len(frame), np.float64))[index] = _adapter_prediction(
            group, local_base, coefficients_shuffle_full[well], mean, scale
        )
        predictions.setdefault("shuffle_early", np.zeros(len(frame), np.float64))[index] = _adapter_prediction(
            group, local_base, coefficients_shuffle_early[well], mean, scale
        )
        donor = donors[well]
        predictions.setdefault("other_full", np.zeros(len(frame), np.float64))[index] = _adapter_prediction(
            group, local_base, coefficients_full[donor], mean, scale
        )
        predictions.setdefault("other_early", np.zeros(len(frame), np.float64))[index] = _adapter_prediction(
            group, local_base, coefficients_early[donor], mean, scale
        )
    return predictions, pd.DataFrame(diagnostics), donors


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(frame, prediction, mask):
    mask = np.asarray(mask, bool)
    subset = frame.loc[mask]
    truth = subset["target_delta_s"].to_numpy(np.float64)
    pred = np.asarray(prediction, np.float64)[mask]
    weight = subset["edge_weight"].to_numpy(np.float64)
    squared = (truth - pred) ** 2
    return {
        "edges": int(len(subset)),
        "wells": int(subset["well"].nunique()),
        "rmse": _rmse(truth, pred),
        "balanced_rmse": _rmse(truth, pred, weight),
        "mse": float(np.mean(squared)),
        "balanced_mse": float(np.sum(weight * squared) / np.maximum(np.sum(weight), 1e-12)),
    }


def _positive_well_fraction(frame, baseline, candidate, mask):
    mask = np.asarray(mask, bool)
    rows = []
    truth_all = frame["target_delta_s"].to_numpy(np.float64)
    for well, group in frame.loc[mask].groupby("well"):
        index = group.index.to_numpy()
        truth = truth_all[index]
        base_rmse = _rmse(truth, np.asarray(baseline)[index])
        candidate_rmse = _rmse(truth, np.asarray(candidate)[index])
        rows.append(
            {
                "well": well,
                "base_rmse": base_rmse,
                "candidate_rmse": candidate_rmse,
                "gain": base_rmse - candidate_rmse,
            }
        )
    bywell = pd.DataFrame(rows)
    return float(np.mean(bywell["gain"] > 0)), bywell


def _distance_masks(frame):
    suffix = frame["region"].eq("suffix").to_numpy()
    distance = frame["end_distance"].to_numpy(np.float64)
    return {
        "late_prefix": frame["prefix_role"].eq("late_query").to_numpy(),
        "suffix_all": suffix,
        "suffix_0_500": suffix & (distance <= 500),
        "suffix_500_1000": suffix & (distance > 500) & (distance <= 1000),
        "suffix_1000_2000": suffix & (distance > 1000) & (distance <= 2000),
        "suffix_gt2000": suffix & (distance > 2000),
        "suffix_ge1000": suffix & (distance > 1000),
    }


def _variant_report(frame, predictions):
    rows = []
    masks = _distance_masks(frame)
    zero = np.zeros(len(frame), np.float64)
    for subset_name, mask in masks.items():
        rows.append({"variant": "zero_delta_s", "subset": subset_name, **_metrics(frame, zero, mask)})
        for variant, prediction in predictions.items():
            rows.append({"variant": variant, "subset": subset_name, **_metrics(frame, prediction, mask)})
    return pd.DataFrame(rows)


def _mse_gain(report, subset, baseline, candidate):
    base = report[(report["subset"] == subset) & (report["variant"] == baseline)].iloc[0]
    test = report[(report["subset"] == subset) & (report["variant"] == candidate)].iloc[0]
    return float((base["balanced_mse"] - test["balanced_mse"]) / max(base["balanced_mse"], 1e-12))


# ---------------------------------------------------------------------------
# Execute nested gateway
# ---------------------------------------------------------------------------

started = time.time()
_seed_everything(SEED)
wells = _horizontal_wells()
folds, fold_path = _load_folds(wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
missing_records = [well for well in wells if not (RECORD_DIR / f"{well}.npz").exists()]
if missing_records:
    raise FileNotFoundError(
        f"Native record cache covers {len(wells)-len(missing_records)}/{len(wells)} wells; "
        f"first missing: {missing_records[:10]}"
    )

edge_parts = []
meta_rows = []
for well in tqdm(wells, desc="build 32-ft multiscale S edges"):
    record = _load_record(well)
    edges, meta = _build_edges(record, fold_by_well[well])
    edge_parts.append(edges)
    meta_rows.append(meta.__dict__)
edges = pd.concat(edge_parts, ignore_index=True)
meta = pd.DataFrame(meta_rows).sort_values(["fold", "well"]).reset_index(drop=True)
del edge_parts
gc.collect()

if not np.isfinite(edges[FEATURE_COLUMNS + ["target_delta_s", "edge_weight"]].to_numpy(np.float64)).all():
    raise RuntimeError("Edge bank contains non-finite values")

bank_summary = {
    "wells": int(edges["well"].nunique()),
    "edges": int(len(edges)),
    "prefix_edges": int(edges["region"].eq("prefix").sum()),
    "suffix_edges": int(edges["region"].eq("suffix").sum()),
    "late_prefix_edges": int(edges["prefix_role"].eq("late_query").sum()),
    "far_suffix_edges": int(((edges["region"] == "suffix") & (edges["end_distance"] > 1000)).sum()),
    "max_edge_length_ft": float((edges["end_md"] - edges["start_md"]).max()),
    "memory_mb": float(edges.memory_usage(deep=True).sum() / 1024**2),
}
print("EDGE BANK", bank_summary, flush=True)
if (
    bank_summary["wells"] != 773
    or bank_summary["late_prefix_edges"] < 10000
    or bank_summary["far_suffix_edges"] < 50000
    or bank_summary["max_edge_length_ft"] < 2000
):
    raise RuntimeError(f"Long-range edge bank guard failed: {bank_summary}")

core = edges[edges["fold"].isin([3, 4, 5])].copy()
tune = edges[edges["fold"].eq(TUNE_FOLD)].copy()
outer_train = edges[edges["fold"].ne(TEST_FOLD)].copy()
test = edges[edges["fold"].eq(TEST_FOLD)].copy()

print("training core global edge mean", flush=True)
core_model = _fit_global(core, tune)
best_iteration = int(core_model.best_iteration_ or core_model.n_estimators_)
core_prediction_tune = core_model.predict(tune[FEATURE_COLUMNS], num_iteration=best_iteration)
adapter_mean, adapter_scale = _fit_adapter_statistics(core)
tune_meta = meta[meta["fold"].eq(TUNE_FOLD)].copy()

lambda_rows = []
for ridge_lambda in RIDGE_LAMBDAS:
    prediction, diagnostics, _ = _build_variant_predictions(
        tune.reset_index(drop=True),
        np.asarray(core_prediction_tune, np.float64),
        ridge_lambda,
        adapter_mean,
        adapter_scale,
        tune_meta,
    )
    tune_reset = tune.reset_index(drop=True)
    report = _variant_report(tune_reset, prediction)
    far_gain = _mse_gain(report, "suffix_ge1000", "global", "same_full")
    late_gain = _mse_gain(report, "late_prefix", "global", "same_early")
    far_row = report[(report["subset"] == "suffix_ge1000") & (report["variant"] == "same_full")].iloc[0]
    lambda_rows.append(
        {
            "ridge_lambda": ridge_lambda,
            "far_balanced_rmse": far_row["balanced_rmse"],
            "far_mse_gain": far_gain,
            "late_prefix_mse_gain": late_gain,
            "median_condition": float(diagnostics["full_condition_number"].median()),
            "p95_condition": float(diagnostics["full_condition_number"].quantile(0.95)),
        }
    )
lambda_scan = pd.DataFrame(lambda_rows).sort_values(
    ["far_balanced_rmse", "late_prefix_mse_gain"], ascending=[True, False]
)
lambda_scan.to_csv(OUT_DIR / "fold2_ridge_lambda_scan.csv", index=False)
selected_lambda = float(lambda_scan.iloc[0]["ridge_lambda"])
print("RIDGE LAMBDA SCAN")
print(lambda_scan.to_string(index=False), flush=True)
print("FROZEN RIDGE LAMBDA", selected_lambda, flush=True)

print("refitting global edge mean on folds 2-5", flush=True)
final_model = _fit_global(outer_train, valid=None, rounds=best_iteration)
final_model.booster_.save_model(str(MODEL_DIR / "global_edge_mean_fold1.txt"))
test = test.reset_index(drop=True)
test_prediction = final_model.predict(test[FEATURE_COLUMNS], num_iteration=best_iteration)
final_adapter_mean, final_adapter_scale = _fit_adapter_statistics(outer_train)
test_meta = meta[meta["fold"].eq(TEST_FOLD)].copy()
predictions, adapter_diagnostics, donors = _build_variant_predictions(
    test,
    np.asarray(test_prediction, np.float64),
    selected_lambda,
    final_adapter_mean,
    final_adapter_scale,
    test_meta,
)
report = _variant_report(test, predictions)
report.to_csv(OUT_DIR / "fold1_edge_report.csv", index=False)
adapter_diagnostics.to_csv(OUT_DIR / "fold1_adapter_diagnostics.csv", index=False)
pd.DataFrame([{"well": well, "donor": donor} for well, donor in donors.items()]).to_csv(
    OUT_DIR / "fold1_matched_other_wells.csv", index=False
)

masks = _distance_masks(test)
late_gain = _mse_gain(report, "late_prefix", "global", "same_early")
far_gain = _mse_gain(report, "suffix_ge1000", "global", "same_full")
global_vs_zero_gain = _mse_gain(report, "suffix_ge1000", "zero_delta_s", "global")
late_positive, late_bywell = _positive_well_fraction(
    test, predictions["global"], predictions["same_early"], masks["late_prefix"]
)
far_positive, far_bywell = _positive_well_fraction(
    test, predictions["global"], predictions["same_full"], masks["suffix_ge1000"]
)
late_bywell.to_csv(OUT_DIR / "fold1_late_prefix_bywell.csv", index=False)
far_bywell.to_csv(OUT_DIR / "fold1_far_suffix_bywell.csv", index=False)

same_far = report[(report["subset"] == "suffix_ge1000") & (report["variant"] == "same_full")].iloc[0]
shuffle_far = report[(report["subset"] == "suffix_ge1000") & (report["variant"] == "shuffle_full")].iloc[0]
other_far = report[(report["subset"] == "suffix_ge1000") & (report["variant"] == "other_full")].iloc[0]
same_late = report[(report["subset"] == "late_prefix") & (report["variant"] == "same_early")].iloc[0]
shuffle_late = report[(report["subset"] == "late_prefix") & (report["variant"] == "shuffle_early")].iloc[0]
other_late = report[(report["subset"] == "late_prefix") & (report["variant"] == "other_early")].iloc[0]

late_pass = bool(
    late_gain >= LATE_PREFIX_MSE_GAIN_GATE
    and late_positive >= POSITIVE_WELL_FRACTION_GATE
    and same_late["balanced_rmse"] < shuffle_late["balanced_rmse"]
    and same_late["balanced_rmse"] < other_late["balanced_rmse"]
)
far_pass = bool(
    far_gain >= FAR_SUFFIX_MSE_GAIN_GATE
    and far_positive >= POSITIVE_WELL_FRACTION_GATE
    and same_far["balanced_rmse"] < shuffle_far["balanced_rmse"]
    and same_far["balanced_rmse"] < other_far["balanced_rmse"]
)
global_pass = bool(global_vs_zero_gain >= GLOBAL_VS_ZERO_MSE_GAIN_GATE)
condition_pass = bool(
    np.isfinite(adapter_diagnostics["full_condition_number"]).all()
    and adapter_diagnostics["full_condition_number"].quantile(0.95) < 1e8
)
strict_pass = bool(late_pass and far_pass and global_pass and condition_pass)

importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": final_model.booster_.feature_importance(importance_type="split"),
    }
).sort_values("importance_gain", ascending=False)
importance.to_csv(OUT_DIR / "global_edge_feature_importance.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "P0.75_P1_prefixridge_longrange_gateway",
    "target": "anchored Delta S; TVT is not predicted in this gateway",
    "fold_source": str(fold_path),
    "protocol": {
        "global_mean_train_folds": [3, 4, 5],
        "ridge_lambda_valid_fold": TUNE_FOLD,
        "final_global_train_folds": [2, 3, 4, 5],
        "untouched_test_fold": TEST_FOLD,
        "best_iteration": best_iteration,
        "selected_ridge_lambda": selected_lambda,
    },
    "edge_bank": bank_summary,
    "fold1": {
        "late_prefix_mse_gain": late_gain,
        "late_prefix_positive_well_fraction": late_positive,
        "far_1000plus_mse_gain": far_gain,
        "far_1000plus_positive_well_fraction": far_positive,
        "global_vs_zero_far_mse_gain": global_vs_zero_gain,
        "same_far_balanced_rmse": float(same_far["balanced_rmse"]),
        "shuffle_far_balanced_rmse": float(shuffle_far["balanced_rmse"]),
        "other_far_balanced_rmse": float(other_far["balanced_rmse"]),
        "same_late_balanced_rmse": float(same_late["balanced_rmse"]),
        "shuffle_late_balanced_rmse": float(shuffle_late["balanced_rmse"]),
        "other_late_balanced_rmse": float(other_late["balanced_rmse"]),
    },
    "gates": {
        "late_prefix_mse_gain_ge": LATE_PREFIX_MSE_GAIN_GATE,
        "far_1000plus_mse_gain_ge": FAR_SUFFIX_MSE_GAIN_GATE,
        "each_positive_well_fraction_ge": POSITIVE_WELL_FRACTION_GATE,
        "global_vs_zero_far_mse_gain_ge": GLOBAL_VS_ZERO_MSE_GAIN_GATE,
        "same_well_better_than_shuffle_and_other": True,
        "adapter_condition_p95_lt": 1e8,
    },
    "late_prefix_pass": late_pass,
    "far_suffix_pass": far_pass,
    "global_edge_signal_pass": global_pass,
    "condition_pass": condition_pass,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> build fixed-weight multiscale anchored SGraph on the frozen edge model"
        if strict_pass
        else "FAIL -> close PrefixRidge-SGraph before graph solve/backbone training"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nFOLD1 EDGE REPORT")
print(report.to_string(index=False), flush=True)
print("\nTOP FEATURE IMPORTANCE")
print(importance.head(30).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)

