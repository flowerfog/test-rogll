"""Strong-base-centred complete-well residual gateway.

This experiment restores input parity with the mature baseline family.  It
does not ask a network to recreate PF/candidate generation from raw GR.  The
frozen centre is the honest ``strong`` OOF path (SP45/FLE lineage before
strong-bias, CP12, Gold or Neighbor).  A complete-well TCN predicts only a
bounded residual using:

* SP45/FLE/ridge/selector candidate trajectories and their disagreement;
* complete suffix GR and trajectory geometry;
* candidate-specific horizontal/typewell GR mismatch channels;
* visible-prefix-only typewell sensor calibration.

Folds 3/4/5 train and Fold 2 is the untouched gateway.  The output head is
zero-initialised, so epoch 0 exactly reproduces ``strong``.  Model selection
includes that zero model and can never promote a degraded epoch.
"""

from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
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
from torch.utils.data import DataLoader, Dataset


ROOT_SB_V0 = Path("/root/autodl-tmp")
DATA_ROOT_SB_V0 = ROOT_SB_V0 / "rogii-wellbore-geology-prediction"
OOF_ROOT_SB_V0 = ROOT_SB_V0 / "oof_exports"
POOL_CANDIDATES_SB_V0 = (
    OOF_ROOT_SB_V0 / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT_SB_V0 / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH_SB_V0 = next(
    (path for path in POOL_CANDIDATES_SB_V0 if path.exists()), None
)
FOLD_CANDIDATES_SB_V0 = (
    OOF_ROOT_SB_V0 / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT_SB_V0 / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH_SB_V0 = next(
    (path for path in FOLD_CANDIDATES_SB_V0 if path.exists()), None
)

EXPERIMENT_SB_V0 = "strongbase_completewell_residual_gateway_v0"
OUT_DIR_SB_V0 = OOF_ROOT_SB_V0 / EXPERIMENT_SB_V0
CACHE_DIR_SB_V0 = OUT_DIR_SB_V0 / "well_cache"
MODEL_DIR_SB_V0 = OUT_DIR_SB_V0 / "models"
for directory_sb_v0 in (OUT_DIR_SB_V0, CACHE_DIR_SB_V0, MODEL_DIR_SB_V0):
    directory_sb_v0.mkdir(parents=True, exist_ok=True)

if POOL_PATH_SB_V0 is None:
    raise FileNotFoundError("candidate pool not found")
if FOLD_PATH_SB_V0 is None:
    raise FileNotFoundError("fold metadata not found")
if not (DATA_ROOT_SB_V0 / "train").exists():
    raise FileNotFoundError(DATA_ROOT_SB_V0 / "train")

SEED_SB_V0 = 20260721
TRAIN_FOLDS_SB_V0 = (3, 4, 5)
VALID_FOLD_SB_V0 = 2
SEQUENCE_POINTS_SB_V0 = 512
MAX_CORRECTION_FT_SB_V0 = 32.0
ANCHOR_RAMP_FT_SB_V0 = 400.0

CANDIDATE_WANTED_SB_V0 = (
    "sp45",
    "fleongg",
    "ridge",
    "selector",
    "strong_raw",
    "sp45_raw",
)
FORBIDDEN_COLUMNS_SB_V0 = {
    "strong_bias",
    "cp12",
    "cp_pack",
    "cp_single12",
}

HIDDEN_SB_V0 = 96
DILATIONS_SB_V0 = (1, 2, 4, 8, 16, 32, 64)
DROPOUT_SB_V0 = 0.10
BATCH_SIZE_SB_V0 = 8
MAX_EPOCHS_SB_V0 = 24
MIN_EPOCHS_SB_V0 = 5
PATIENCE_SB_V0 = 5
LEARNING_RATE_SB_V0 = 5e-4
MIN_LEARNING_RATE_SB_V0 = 5e-5
WEIGHT_DECAY_SB_V0 = 1e-4


def _seed_sb_v0(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _rmse_sb_v0(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _robust_scale_sb_v0(values, valid=None):
    values = np.asarray(values, np.float64)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, bool) & np.isfinite(values)
    finite = values[valid]
    if len(finite) < 8:
        return 0.0, 1.0
    centre = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(finite), 1.0))
    return centre, scale


def _normalise_sb_v0(values, centre=None, scale=None, clip=8.0):
    values = np.asarray(values, np.float64)
    if centre is None or scale is None:
        centre, scale = _robust_scale_sb_v0(values)
    output = np.nan_to_num((values - centre) / max(scale, 1e-6), nan=0.0)
    return np.clip(output, -clip, clip).astype(np.float32)


def _interp_sb_v0(x, y, query, left=np.nan, right=np.nan):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full(query.shape, np.nan, np.float64)
    order = np.argsort(x[valid], kind="mergesort")
    xv = x[valid][order]
    yv = y[valid][order]
    xv, unique = np.unique(xv, return_index=True)
    yv = yv[unique]
    return np.interp(query, xv, yv, left=left, right=right)


def _calibrate_typewell_sb_v0(horizontal, typewell):
    hgr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
    tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
    expected = _interp_sb_v0(tw_tvt, tw_gr, tvt_input)
    valid = np.isfinite(hgr) & np.isfinite(tvt_input) & np.isfinite(expected)
    if valid.sum() < 32:
        return tw_tvt, tw_gr, 1.0, 0.0
    x = expected[valid]
    y = hgr[valid]
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, 0.0
    for _ in range(5):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre, scale = _robust_scale_sb_v0(residual)
        next_keep = np.abs(residual - centre) <= 3.5 * max(scale, 1e-3)
        if next_keep.sum() < 24 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    slope = float(np.clip(slope, 0.25, 4.0))
    return tw_tvt, slope * tw_gr + float(intercept), slope, float(intercept)


def _gradient_sb_v0(values, md):
    values = np.asarray(values, np.float64)
    md = np.asarray(md, np.float64)
    if len(values) < 3:
        return np.zeros(len(values), np.float32)
    return np.nan_to_num(np.gradient(values, md), nan=0.0).astype(np.float32)


def _smooth_sb_v0(values, window):
    values = np.asarray(values, np.float64)
    if window <= 1:
        return values.astype(np.float32)
    kernel = np.ones(int(window), np.float64) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _node_indices_sb_v0(length):
    if length < SEQUENCE_POINTS_SB_V0:
        return None
    return np.rint(
        np.linspace(0, length - 1, SEQUENCE_POINTS_SB_V0)
    ).astype(np.int64)


def _node_weights_sb_v0(node_index, length):
    node_index = np.asarray(node_index, np.int64)
    boundaries = np.r_[
        0,
        np.rint((node_index[:-1] + node_index[1:]) / 2.0).astype(np.int64),
        length,
    ]
    weight = np.diff(boundaries).astype(np.float32)
    weight /= max(float(weight.mean()), 1e-6)
    return weight


header_sb_v0 = pd.read_csv(POOL_PATH_SB_V0, nrows=0).columns.tolist()
required_sb_v0 = {"id", "tvt_true", "strong"}
if not required_sb_v0.issubset(header_sb_v0):
    raise RuntimeError(
        f"pool misses {sorted(required_sb_v0 - set(header_sb_v0))}"
    )
candidate_columns_sb_v0 = [
    column
    for column in CANDIDATE_WANTED_SB_V0
    if column in header_sb_v0 and column not in FORBIDDEN_COLUMNS_SB_V0
]
if not {"sp45", "fleongg"}.issubset(candidate_columns_sb_v0):
    raise RuntimeError(
        f"SP45/FLE are required; available candidates={candidate_columns_sb_v0}"
    )
usecols_sb_v0 = [
    column
    for column in (
        "id",
        "well",
        "row_idx",
        "tvt_true",
        "strong",
        *candidate_columns_sb_v0,
    )
    if column in header_sb_v0
]
pool_sb_v0 = pd.read_csv(
    POOL_PATH_SB_V0,
    usecols=usecols_sb_v0,
    dtype={"id": str, "well": str},
)
pool_sb_v0["id"] = pool_sb_v0["id"].astype(str)
if "well" not in pool_sb_v0:
    pool_sb_v0["well"] = pool_sb_v0["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool_sb_v0:
    pool_sb_v0["row_idx"] = (
        pool_sb_v0["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    )
pool_sb_v0["well"] = pool_sb_v0["well"].astype(str)
pool_sb_v0["row_idx"] = pool_sb_v0["row_idx"].astype(np.int32)
for column_sb_v0 in ("tvt_true", "strong", *candidate_columns_sb_v0):
    pool_sb_v0[column_sb_v0] = pd.to_numeric(
        pool_sb_v0[column_sb_v0], errors="coerce"
    ).astype(np.float32)
pool_sb_v0 = pool_sb_v0.sort_values(["well", "row_idx"]).reset_index(drop=True)

fold_frame_sb_v0 = pd.read_csv(FOLD_PATH_SB_V0, dtype={"well": str})
fold_frame_sb_v0 = fold_frame_sb_v0.groupby("well", as_index=False)["fold"].first()
fold_map_sb_v0 = dict(
    zip(
        fold_frame_sb_v0["well"].astype(str),
        pd.to_numeric(fold_frame_sb_v0["fold"], errors="raise").astype(int),
    )
)
pool_sb_v0["fold"] = pool_sb_v0["well"].map(fold_map_sb_v0)
if pool_sb_v0["fold"].isna().any():
    raise RuntimeError("candidate pool contains wells missing from fold metadata")
pool_sb_v0["fold"] = pool_sb_v0["fold"].astype(np.int8)


def _build_well_sb_v0(well, frame):
    output = CACHE_DIR_SB_V0 / f"{well}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as data:
            return {
                "well": well,
                "status": "cached",
                "features": int(data["features"].shape[0]),
                "rows": int(len(data["truth_full"])),
            }
    horizontal_path = DATA_ROOT_SB_V0 / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT_SB_V0 / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    row_idx = frame["row_idx"].to_numpy(np.int64)
    if row_idx.max() >= len(horizontal):
        raise RuntimeError(f"{well}: row index exceeds horizontal file")

    md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    x_all = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(np.float64)
    y_all = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr_all = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise RuntimeError(f"{well}: no visible TVT prefix")
    anchor = int(known[-1])
    anchor_md = float(md_all[anchor])
    anchor_tvt = float(tvt_input[anchor])

    md = md_all[row_idx]
    x = x_all[row_idx]
    y = y_all[row_idx]
    z = z_all[row_idx]
    gr = gr_all[row_idx]
    truth = frame["tvt_true"].to_numpy(np.float64)
    strong = frame["strong"].to_numpy(np.float64)
    candidate = {
        column: frame[column].to_numpy(np.float64)
        for column in candidate_columns_sb_v0
    }
    if not np.isfinite(truth).all() or not np.isfinite(strong).all():
        raise RuntimeError(f"{well}: non-finite truth/strong")

    tw_tvt, tw_gr_calibrated, calibration_slope, calibration_intercept = (
        _calibrate_typewell_sb_v0(horizontal, typewell)
    )
    gr_centre, gr_scale = _robust_scale_sb_v0(gr_all)
    gr_norm = _normalise_sb_v0(gr, gr_centre, gr_scale)

    parts = []
    names = []

    def add(name, values):
        values = np.asarray(values, np.float32)
        if values.ndim == 1:
            values = values[None]
        parts.append(values)
        names.extend(
            [name] if values.shape[0] == 1 else [f"{name}_{i}" for i in range(values.shape[0])]
        )

    distance = md - anchor_md
    progress = np.clip(distance / max(float(distance[-1]), 1.0), 0.0, 1.0)
    ramp = 1.0 - np.exp(-np.maximum(distance, 0.0) / ANCHOR_RAMP_FT_SB_V0)
    add("geometry_progress", progress)
    add("geometry_ramp", ramp)
    add("geometry_z_relative", (z - z_all[anchor]) / 128.0)
    add("geometry_dx_dmd", _gradient_sb_v0(x, md))
    add("geometry_dy_dmd", _gradient_sb_v0(y, md))
    add("geometry_dz_dmd", _gradient_sb_v0(z, md))

    add("gr_value", gr_norm)
    add("gr_d1", _gradient_sb_v0(gr_norm, md) * 32.0)
    add("gr_d2", _gradient_sb_v0(_gradient_sb_v0(gr_norm, md), md) * 1024.0)
    for window in (17, 65, 257):
        smooth = _smooth_sb_v0(gr_norm, window)
        add(f"gr_smooth{window}", smooth)
        add(f"gr_highpass{window}", gr_norm - smooth)

    add("base_relative_anchor", (strong - anchor_tvt) / 128.0)
    add("base_d1", _gradient_sb_v0(strong, md) * 32.0)
    add("base_d2", _gradient_sb_v0(_gradient_sb_v0(strong, md), md) * 1024.0)

    candidate_stack = []
    path_for_match = {"strong": strong, **candidate}
    for column, values in candidate.items():
        delta = values - strong
        candidate_stack.append(delta)
        add(f"candidate_{column}_delta", delta / 32.0)
        add(f"candidate_{column}_d1", _gradient_sb_v0(delta, md) * 32.0 / 16.0)
        add(
            f"candidate_{column}_smooth",
            _smooth_sb_v0(delta, 65) / 32.0,
        )
    candidate_stack = np.stack(candidate_stack, axis=0)
    add("disagreement_mean", np.mean(candidate_stack, axis=0) / 32.0)
    add("disagreement_std", np.std(candidate_stack, axis=0) / 32.0)
    add("disagreement_range", np.ptp(candidate_stack, axis=0) / 32.0)

    tw_d1 = np.gradient(tw_gr_calibrated, tw_tvt)
    for column, values in path_for_match.items():
        tw_at_path = _interp_sb_v0(tw_tvt, tw_gr_calibrated, values)
        tw_at_path_norm = _normalise_sb_v0(
            tw_at_path, gr_centre, gr_scale
        )
        tw_gradient = _interp_sb_v0(tw_tvt, tw_d1, values)
        plus24 = _interp_sb_v0(tw_tvt, tw_gr_calibrated, values + 24.0)
        minus24 = _interp_sb_v0(tw_tvt, tw_gr_calibrated, values - 24.0)
        plus24_norm = _normalise_sb_v0(plus24, gr_centre, gr_scale)
        minus24_norm = _normalise_sb_v0(minus24, gr_centre, gr_scale)
        add(f"tw_{column}_value", tw_at_path_norm)
        add(f"tw_{column}_gradient", np.nan_to_num(tw_gradient / max(gr_scale, 1e-6)))
        add(f"match_{column}_residual", gr_norm - tw_at_path_norm)
        add(
            f"match_{column}_alias_margin",
            np.minimum(np.abs(gr_norm - plus24_norm), np.abs(gr_norm - minus24_norm))
            - np.abs(gr_norm - tw_at_path_norm),
        )

    feature_full = np.concatenate(parts, axis=0).astype(np.float32)
    node_index = _node_indices_sb_v0(len(frame))
    if node_index is None:
        # Keep the complete well instead of dropping rare short suffixes.
        # The network still receives the same fixed-length representation;
        # interpolation is on physical MD and therefore preserves trajectory
        # progress when row spacing is irregular.
        node_md = np.linspace(
            float(md[0]), float(md[-1]), SEQUENCE_POINTS_SB_V0
        ).astype(np.float32)
        feature_node = np.stack(
            [
                np.interp(node_md, md, channel).astype(np.float32)
                for channel in feature_full
            ],
            axis=0,
        )
        target_node = np.interp(
            node_md, md, truth - strong
        ).astype(np.float32)
        ramp_node = np.interp(node_md, md, ramp).astype(np.float32)
        node_weight = np.full(
            SEQUENCE_POINTS_SB_V0,
            float(len(frame)) / SEQUENCE_POINTS_SB_V0,
            np.float32,
        )
        node_weight /= max(float(node_weight.mean()), 1e-6)
    else:
        node_weight = _node_weights_sb_v0(node_index, len(frame))
        feature_node = feature_full[:, node_index]
        target_node = (truth[node_index] - strong[node_index]).astype(np.float32)
        node_md = md[node_index].astype(np.float32)
        ramp_node = ramp[node_index].astype(np.float32)

    if len(names) != feature_node.shape[0]:
        raise RuntimeError(f"{well}: feature name mismatch")
    np.savez_compressed(
        output,
        features=feature_node.astype(np.float16),
        feature_names=np.asarray(names, str),
        target=target_node,
        node_weight=node_weight,
        node_md=node_md,
        ramp=ramp_node,
        eval_md=md.astype(np.float32),
        truth_full=truth.astype(np.float32),
        base_full=strong.astype(np.float32),
        distance_full=distance.astype(np.float32),
        fold=np.asarray([int(frame["fold"].iloc[0])], np.int8),
        calibration=np.asarray([calibration_slope, calibration_intercept], np.float32),
    )
    return {
        "well": well,
        "status": "built",
        "features": feature_node.shape[0],
        "rows": len(frame),
    }


pool_groups_sb_v0 = {
    well: frame.copy()
    for well, frame in pool_sb_v0.groupby("well", sort=True)
    if int(frame["fold"].iloc[0]) in (*TRAIN_FOLDS_SB_V0, VALID_FOLD_SB_V0)
}
started_sb_v0 = time.time()
build_rows_sb_v0 = []
for index_sb_v0, (well_sb_v0, frame_sb_v0) in enumerate(pool_groups_sb_v0.items(), 1):
    try:
        build_rows_sb_v0.append(_build_well_sb_v0(well_sb_v0, frame_sb_v0))
    except Exception as exc:
        build_rows_sb_v0.append(
            {"well": well_sb_v0, "status": "failed", "reason": repr(exc)}
        )
    if index_sb_v0 % 50 == 0 or index_sb_v0 == len(pool_groups_sb_v0):
        print(
            f"build strong-base wells {index_sb_v0}/{len(pool_groups_sb_v0)}",
            flush=True,
        )
build_report_sb_v0 = pd.DataFrame(build_rows_sb_v0)
build_report_sb_v0.to_csv(OUT_DIR_SB_V0 / "build_report.csv", index=False)
failed_sb_v0 = build_report_sb_v0[build_report_sb_v0["status"].eq("failed")]
if len(failed_sb_v0):
    raise RuntimeError(
        f"{len(failed_sb_v0)} well builds failed: "
        + failed_sb_v0.head(10).to_dict(orient="records").__repr__()
    )

train_wells_sb_v0 = sorted(
    well
    for well, frame in pool_groups_sb_v0.items()
    if int(frame["fold"].iloc[0]) in TRAIN_FOLDS_SB_V0
)
valid_wells_sb_v0 = sorted(
    well
    for well, frame in pool_groups_sb_v0.items()
    if int(frame["fold"].iloc[0]) == VALID_FOLD_SB_V0
)

with np.load(CACHE_DIR_SB_V0 / f"{train_wells_sb_v0[0]}.npz", allow_pickle=False) as probe_sb_v0:
    feature_names_sb_v0 = probe_sb_v0["feature_names"].astype(str).tolist()
input_channels_sb_v0 = len(feature_names_sb_v0)
candidate_channel_sb_v0 = np.asarray(
    [
        name.startswith(("candidate_", "disagreement_", "match_", "tw_"))
        for name in feature_names_sb_v0
    ],
    bool,
)
gr_channel_sb_v0 = np.asarray(
    [name.startswith(("gr_", "match_", "tw_")) for name in feature_names_sb_v0],
    bool,
)


class StrongBaseDatasetV0(Dataset):
    def __init__(self, wells):
        self.wells = list(wells)

    def __len__(self):
        return len(self.wells)

    def __getitem__(self, index):
        well = self.wells[index]
        with np.load(CACHE_DIR_SB_V0 / f"{well}.npz", allow_pickle=False) as data:
            return {
                "well": well,
                "features": torch.from_numpy(data["features"].astype(np.float32)),
                "target": torch.from_numpy(data["target"].astype(np.float32)),
                "weight": torch.from_numpy(data["node_weight"].astype(np.float32)),
                "ramp": torch.from_numpy(data["ramp"].astype(np.float32)),
            }


class ResidualBlockSBV0(nn.Module):
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
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(DROPOUT_SB_V0),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.network(values))


class StrongBaseCompleteWellV0(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(input_channels, HIDDEN_SB_V0, 1),
            nn.GroupNorm(8, HIDDEN_SB_V0),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlockSBV0(HIDDEN_SB_V0, dilation)
                for dilation in DILATIONS_SB_V0
            ]
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(HIDDEN_SB_V0, HIDDEN_SB_V0, 1),
            nn.GELU(),
        )
        self.output = nn.Conv1d(HIDDEN_SB_V0 * 2, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, ramp):
        hidden = self.blocks(self.input(features))
        context = self.global_context(hidden).expand_as(hidden)
        raw = self.output(torch.cat([hidden, context], dim=1))[:, 0]
        return MAX_CORRECTION_FT_SB_V0 * torch.tanh(raw) * ramp


def _loss_sb_v0(prediction, target, weight):
    squared = (prediction - target) ** 2
    pooled = (squared * weight).sum() / weight.sum().clamp_min(1e-6)
    curvature = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
    smooth = torch.mean(curvature**2)
    return pooled + 0.01 * smooth, pooled.detach(), smooth.detach()


@torch.no_grad()
def _evaluate_sb_v0(model, wells, device, mode="full"):
    model.eval()
    frames = []
    for well in wells:
        with np.load(CACHE_DIR_SB_V0 / f"{well}.npz", allow_pickle=False) as data:
            features = data["features"].astype(np.float32)
            if mode == "no_gr":
                features[gr_channel_sb_v0] = 0.0
            elif mode == "no_candidates":
                features[candidate_channel_sb_v0] = 0.0
            tensor = torch.from_numpy(features[None]).to(device)
            ramp = torch.from_numpy(data["ramp"].astype(np.float32)[None]).to(device)
            correction_node = model(tensor, ramp)[0].float().cpu().numpy()
            correction = np.interp(
                data["eval_md"].astype(np.float64),
                data["node_md"].astype(np.float64),
                correction_node.astype(np.float64),
            )
            base = data["base_full"].astype(np.float64)
            truth = data["truth_full"].astype(np.float64)
            frames.append(
                pd.DataFrame(
                    {
                        "well": well,
                        "truth": truth,
                        "base": base,
                        "prediction": base + correction,
                        "distance_ft": data["distance_full"].astype(np.float64),
                        "correction": correction,
                    }
                )
            )
    frame = pd.concat(frames, ignore_index=True)
    error = frame["prediction"].to_numpy(float) - frame["truth"].to_numpy(float)
    base_error = frame["base"].to_numpy(float) - frame["truth"].to_numpy(float)
    bywell = []
    for well, group in frame.groupby("well", sort=True):
        local = group["prediction"].to_numpy(float) - group["truth"].to_numpy(float)
        local_base = group["base"].to_numpy(float) - group["truth"].to_numpy(float)
        bywell.append(
            {
                "well": well,
                "rows": len(group),
                "base_rmse": float(np.sqrt(np.mean(local_base**2))),
                "rmse": float(np.sqrt(np.mean(local**2))),
            }
        )
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": len(frame),
        "wells": len(bywell),
        "base_rmse": float(np.sqrt(np.mean(base_error**2))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "gain": float(np.sqrt(np.mean(base_error**2)) - np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "mean_abs_move": float(np.mean(np.abs(frame["correction"]))),
        "p95_abs_move": float(np.quantile(np.abs(frame["correction"]), 0.95)),
    }
    for label, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (
            (frame["distance_ft"].to_numpy(float) >= low)
            & (frame["distance_ft"].to_numpy(float) < high)
        )
        metrics[f"{label}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return metrics, bywell, frame


def _meta_fold_sb_v0(well):
    digest = hashlib.sha256(str(well).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 + 1


def _meta_report_sb_v0(frame):
    rows = []
    for meta_fold in range(1, 6):
        wells = [well for well in frame["well"].unique() if _meta_fold_sb_v0(well) == meta_fold]
        group = frame[frame["well"].isin(wells)]
        base_rmse = _rmse_sb_v0(group["truth"], group["base"])
        rmse = _rmse_sb_v0(group["truth"], group["prediction"])
        rows.append(
            {
                "meta_fold": meta_fold,
                "wells": len(wells),
                "rows": len(group),
                "base_rmse": base_rmse,
                "rmse": rmse,
                "gain": base_rmse - rmse,
            }
        )
    return pd.DataFrame(rows)


device_sb_v0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_sb_v0(SEED_SB_V0)
model_sb_v0 = StrongBaseCompleteWellV0(input_channels_sb_v0).to(device_sb_v0)
train_loader_sb_v0 = DataLoader(
    StrongBaseDatasetV0(train_wells_sb_v0),
    batch_size=BATCH_SIZE_SB_V0,
    shuffle=True,
    num_workers=0,
    pin_memory=device_sb_v0.type == "cuda",
)
optimizer_sb_v0 = torch.optim.AdamW(
    model_sb_v0.parameters(),
    lr=LEARNING_RATE_SB_V0,
    weight_decay=WEIGHT_DECAY_SB_V0,
)
scheduler_sb_v0 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_sb_v0,
    T_max=MAX_EPOCHS_SB_V0,
    eta_min=MIN_LEARNING_RATE_SB_V0,
)

base_metrics_sb_v0, _, base_frame_sb_v0 = _evaluate_sb_v0(
    model_sb_v0, valid_wells_sb_v0, device_sb_v0
)
best_sb_v0 = {
    "epoch": 0,
    "metrics": base_metrics_sb_v0,
    "state": {
        key: value.detach().cpu().clone()
        for key, value in model_sb_v0.state_dict().items()
    },
}
history_sb_v0 = [{"epoch": 0, **base_metrics_sb_v0}]
stale_sb_v0 = 0

for epoch_sb_v0 in range(1, MAX_EPOCHS_SB_V0 + 1):
    model_sb_v0.train()
    totals_sb_v0 = {"loss": 0.0, "pooled": 0.0, "smooth": 0.0}
    for batch_sb_v0 in train_loader_sb_v0:
        features_sb_v0 = batch_sb_v0["features"].to(
            device_sb_v0, non_blocking=True
        )
        target_sb_v0 = batch_sb_v0["target"].to(
            device_sb_v0, non_blocking=True
        )
        weight_sb_v0 = batch_sb_v0["weight"].to(
            device_sb_v0, non_blocking=True
        )
        ramp_sb_v0 = batch_sb_v0["ramp"].to(
            device_sb_v0, non_blocking=True
        )
        optimizer_sb_v0.zero_grad(set_to_none=True)
        prediction_sb_v0 = model_sb_v0(features_sb_v0, ramp_sb_v0)
        loss_sb_v0, pooled_loss_sb_v0, smooth_loss_sb_v0 = _loss_sb_v0(
            prediction_sb_v0, target_sb_v0, weight_sb_v0
        )
        loss_sb_v0.backward()
        torch.nn.utils.clip_grad_norm_(model_sb_v0.parameters(), 5.0)
        optimizer_sb_v0.step()
        totals_sb_v0["loss"] += float(loss_sb_v0.detach().cpu())
        totals_sb_v0["pooled"] += float(pooled_loss_sb_v0.cpu())
        totals_sb_v0["smooth"] += float(smooth_loss_sb_v0.cpu())
    scheduler_sb_v0.step()
    metrics_sb_v0, _, _ = _evaluate_sb_v0(
        model_sb_v0, valid_wells_sb_v0, device_sb_v0
    )
    row_sb_v0 = {
        "epoch": epoch_sb_v0,
        "train_loss": totals_sb_v0["loss"] / len(train_loader_sb_v0),
        "train_pooled": totals_sb_v0["pooled"] / len(train_loader_sb_v0),
        "train_smooth": totals_sb_v0["smooth"] / len(train_loader_sb_v0),
        **metrics_sb_v0,
        "learning_rate": optimizer_sb_v0.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started_sb_v0) / 60.0,
    }
    history_sb_v0.append(row_sb_v0)
    print("STRONGBASE EPOCH", row_sb_v0, flush=True)
    if metrics_sb_v0["rmse"] < best_sb_v0["metrics"]["rmse"] - 1e-5:
        best_sb_v0 = {
            "epoch": epoch_sb_v0,
            "metrics": metrics_sb_v0,
            "state": {
                key: value.detach().cpu().clone()
                for key, value in model_sb_v0.state_dict().items()
            },
        }
        stale_sb_v0 = 0
    else:
        stale_sb_v0 += 1
    if epoch_sb_v0 >= MIN_EPOCHS_SB_V0 and stale_sb_v0 >= PATIENCE_SB_V0:
        print("STRONGBASE early stopping", flush=True)
        break

pd.DataFrame(history_sb_v0).to_csv(
    OUT_DIR_SB_V0 / "training_history.csv", index=False
)
model_sb_v0.load_state_dict(best_sb_v0["state"])
full_metrics_sb_v0, full_bywell_sb_v0, full_frame_sb_v0 = _evaluate_sb_v0(
    model_sb_v0, valid_wells_sb_v0, device_sb_v0, mode="full"
)
no_gr_metrics_sb_v0, _, _ = _evaluate_sb_v0(
    model_sb_v0, valid_wells_sb_v0, device_sb_v0, mode="no_gr"
)
no_candidate_metrics_sb_v0, _, _ = _evaluate_sb_v0(
    model_sb_v0, valid_wells_sb_v0, device_sb_v0, mode="no_candidates"
)
meta_report_sb_v0 = _meta_report_sb_v0(full_frame_sb_v0)
meta_report_sb_v0.to_csv(OUT_DIR_SB_V0 / "meta_fold_scores.csv", index=False)
full_bywell_sb_v0.to_csv(OUT_DIR_SB_V0 / "fold2_bywell.csv", index=False)
full_frame_sb_v0.to_pickle(OUT_DIR_SB_V0 / "fold2_predictions.pkl")

checkpoint_sb_v0 = MODEL_DIR_SB_V0 / "best_fold2_strongbase_residual.pt"
torch.save(
    {
        "model": best_sb_v0["state"],
        "epoch": int(best_sb_v0["epoch"]),
        "feature_names": feature_names_sb_v0,
        "candidate_columns": candidate_columns_sb_v0,
        "metrics": full_metrics_sb_v0,
    },
    checkpoint_sb_v0,
)

gain_sb_v0 = float(full_metrics_sb_v0["gain"])
positive_meta_sb_v0 = int((meta_report_sb_v0["gain"] > 0).sum())
minimum_meta_gain_sb_v0 = float(meta_report_sb_v0["gain"].min())
no_gr_degradation_sb_v0 = float(
    no_gr_metrics_sb_v0["rmse"] - full_metrics_sb_v0["rmse"]
)
no_candidate_degradation_sb_v0 = float(
    no_candidate_metrics_sb_v0["rmse"] - full_metrics_sb_v0["rmse"]
)
strict_pass_sb_v0 = bool(
    gain_sb_v0 >= 0.30
    and positive_meta_sb_v0 >= 4
    and minimum_meta_gain_sb_v0 >= -0.10
    and no_gr_degradation_sb_v0 >= 0.10
    and no_candidate_degradation_sb_v0 >= 0.10
)

verdict_sb_v0 = {
    "experiment": EXPERIMENT_SB_V0,
    "stage": "strong_base_complete_well_residual_fold2_gateway",
    "centre": "strong OOF; excludes strong_bias, CP12, Gold and Neighbor",
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_SB_V0),
        "validation_fold": VALID_FOLD_SB_V0,
        "fold1_touched": False,
        "selected_epoch": int(best_sb_v0["epoch"]),
        "epoch0_exactly_reproduces_base": True,
    },
    "inputs": {
        "candidate_columns": candidate_columns_sb_v0,
        "feature_channels": input_channels_sb_v0,
        "sequence_points": SEQUENCE_POINTS_SB_V0,
        "visible_prefix_typewell_calibration": True,
    },
    "parameters": int(sum(parameter.numel() for parameter in model_sb_v0.parameters())),
    "base": base_metrics_sb_v0,
    "full": full_metrics_sb_v0,
    "no_gr": no_gr_metrics_sb_v0,
    "no_candidates": no_candidate_metrics_sb_v0,
    "meta_folds": meta_report_sb_v0.to_dict(orient="records"),
    "gains": {
        "overall": gain_sb_v0,
        "positive_meta_folds": positive_meta_sb_v0,
        "minimum_meta_fold_gain": minimum_meta_gain_sb_v0,
        "no_gr_degradation": no_gr_degradation_sb_v0,
        "no_candidate_degradation": no_candidate_degradation_sb_v0,
    },
    "gates": {
        "overall_gain_ge": 0.30,
        "positive_meta_folds_ge": 4,
        "minimum_meta_fold_gain_ge": -0.10,
        "no_gr_degradation_ge": 0.10,
        "no_candidate_degradation_ge": 0.10,
    },
    "strict_pass": strict_pass_sb_v0,
    "checkpoint": str(checkpoint_sb_v0),
    "elapsed_minutes": (time.time() - started_sb_v0) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT_SB_V0).free / 1024**3,
    "next": (
        "PASS -> train a proper five-fold strong-base complete-well residual model"
        if strict_pass_sb_v0
        else "FAIL -> candidate-complete-well residual learning is too weak; retain the frozen strong lineage"
    ),
}
(OUT_DIR_SB_V0 / "verdict.json").write_text(
    json.dumps(verdict_sb_v0, indent=2), encoding="utf-8"
)

print("\nMETA FOLDS")
print(meta_report_sb_v0.to_string(index=False), flush=True)
print("\nCONTROLS")
print(
    pd.DataFrame(
        [
            {"mode": "full", **full_metrics_sb_v0},
            {"mode": "no_gr", **no_gr_metrics_sb_v0},
            {"mode": "no_candidates", **no_candidate_metrics_sb_v0},
        ]
    ).to_string(index=False),
    flush=True,
)
print("\nFINAL VERDICT")
print(json.dumps(verdict_sb_v0, indent=2), flush=True)
print("saved:", OUT_DIR_SB_V0, flush=True)

del optimizer_sb_v0, scheduler_sb_v0
gc.collect()
torch.cuda.empty_cache()
