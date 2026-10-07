"""Nested PF-posterior complete-well BiGRU gateway.

The model does not invert raw GR from scratch.  A deterministic, fold-clean
family of per-well PF paths supplies a physical proposal coordinate system;
the BiGRU consumes the complete hidden-tail sequence and predicts a bounded
correction to the PF posterior centre.  All raw inputs are available for the
same well at test time.  No neighbour, CP12, Gold, or spatial feature is used.

Protocol
--------
* folds 4/5 train the epoch/dose-selection model;
* fold 3 selects epoch and scalar dose;
* folds 3/4/5 are refit for the frozen number of epochs;
* fold 2 is touched once for the final gateway and interventions;
* fold 1 is never loaded.
"""

from pathlib import Path
import copy
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------
EXPERIMENT = "pfposterior_bigru_nested_gateway_v0"
DATA_CANDIDATES = (
    Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    Path("/root/autodl-tmp"),
    Path.cwd(),
)
DATA_ROOT = next(
    (
        root for root in DATA_CANDIDATES
        if (root / "train").exists()
    ),
    None,
)
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof_honest.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
PFMM_DIR_CANDIDATES = (
    OOF_ROOT / "pf_multimodal_input_ablation_v0_s16_p200" / "pf_seed_features",
    OOF_ROOT / "pf_multimodal_input_ablation_v0" / "pf_seed_features",
)
PFMM_DIR = next((path for path in PFMM_DIR_CANDIDATES if path.exists()), None)
PF_FEATURE_DIRS = (
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "features",
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "features",
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "features",
)

OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "records_v0"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CORE_FOLDS = (4, 5)
SELECTION_FOLD = 3
OUTER_FOLDS = (3, 4, 5)
VALID_FOLD = 2
NODE_STEP_FT = 4.0
PSEUDO_HEEL_WEIGHT = 0.80
PSEUDO_BANDWIDTH_FT = 16.0
ANCHOR_RAMP_FT = 500.0
MAX_CORRECTION_FT = 64.0

PF_WANTED = (
    "pf_ancc",
    "pf_z",
    "likpf_scale_3",
    "likpf_scale_5",
    "likpf_scale_8",
    "likpf_scale_12",
    "likpf_mean",
)
BASE_PREFERENCE = (
    "likpf_mean",
    "likpf_scale_5",
    "likpf_scale_8",
    "pf_ancc",
)
PFMM_WANTED = (
    "pfmm_weighted_mean_d",
    "pfmm_q10_d",
    "pfmm_q25_d",
    "pfmm_q50_d",
    "pfmm_q75_d",
    "pfmm_q90_d",
    "pfmm_std",
    "pfmm_iqr",
    "pfmm_q90_q10",
    "pfmm_mode1_d",
    "pfmm_mode2_d",
    "pfmm_mode3_d",
    "pfmm_mode1_weight",
    "pfmm_mode2_weight",
    "pfmm_mode3_weight",
    "pfmm_entropy",
    "pfmm_ess_fraction",
    "pfmm_ll_gap12_per_row",
    "pfmm_ll_std_per_row",
)

SEED = 20260721
HIDDEN = 96
LAYERS = 2
DROPOUT = 0.12
BATCH_SIZE = 4
GRAD_ACCUM = 2
MAX_EPOCHS = 14
PATIENCE = 4
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 2e-4
DOSES = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
STRATA = (
    ("near_0_500", 0.0, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

if DATA_ROOT is None:
    raise FileNotFoundError("competition train directory not found")
if POOL_PATH is None:
    raise FileNotFoundError("fold-clean candidate pool not found")
if FOLD_PATH is None:
    raise FileNotFoundError("well fold metadata not found")


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


def _fill(values):
    return (
        pd.Series(np.asarray(values, float))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(float)
    )


def _centre_scale(values, valid=None):
    values = np.asarray(values, float)
    mask = np.isfinite(values)
    if valid is not None:
        mask &= np.asarray(valid, bool)
    finite = values[mask]
    if len(finite) < 4:
        return 0.0, 1.0
    centre = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(finite), 1e-3))
    return centre, scale


def _normalise(values, centre=None, scale=None, clip=8.0):
    values = np.asarray(values, float)
    if centre is None or scale is None:
        centre, scale = _centre_scale(values)
    return np.clip((np.nan_to_num(values, nan=centre) - centre) / max(scale, 1e-6), -clip, clip)


def _gradient(values, coordinate):
    values = np.asarray(values, float)
    coordinate = np.asarray(coordinate, float)
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, coordinate, edge_order=1)


def _smooth(values, window):
    return (
        pd.Series(np.asarray(values, float))
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(float)
    )


def _node_indices(md):
    md = np.asarray(md, float)
    grid = np.arange(md[0], md[-1] + 0.5 * NODE_STEP_FT, NODE_STEP_FT)
    indices = np.searchsorted(md, grid, side="left")
    indices = np.clip(indices, 0, len(md) - 1)
    indices = np.unique(np.r_[0, indices, len(md) - 1]).astype(np.int32)
    return indices


def _calibrated_pseudo_typewell(horizontal, typewell):
    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT")
    tw = tw.groupby("TVT", as_index=False)["GR"].mean()
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = _fill(tw["GR"].to_numpy(float))

    known = horizontal[horizontal["TVT_input"].notna()].copy()
    known_tvt = pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(float)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").to_numpy(float)
    expected = np.interp(known_tvt, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    valid = np.isfinite(known_tvt) & np.isfinite(known_gr) & np.isfinite(expected)
    if valid.sum() >= 16:
        x = expected[valid]
        y = known_gr[valid]
        slope = float(np.clip(np.cov(x, y, ddof=0)[0, 1] / max(np.var(x), 1e-6), 0.25, 4.0))
        intercept = float(np.median(y - slope * x))
    else:
        slope, intercept = 1.0, 0.0
    calibrated = slope * tw_gr + intercept

    # Inject only locally-supported heel residuals; outside the visible TVT
    # span the supplied typewell remains authoritative.
    if valid.sum() >= 8:
        residual = known_gr[valid] - np.interp(known_tvt[valid], tw_tvt, calibrated)
        distance = (tw_tvt[:, None] - known_tvt[valid][None, :]) / PSEUDO_BANDWIDTH_FT
        weight = np.exp(-0.5 * distance * distance)
        support = weight.sum(axis=1)
        local = (weight @ residual) / np.maximum(support, 1e-8)
        confidence = np.clip(support / 3.0, 0.0, 1.0)
        pseudo = calibrated + PSEUDO_HEEL_WEIGHT * confidence * local
    else:
        pseudo = calibrated.copy()
    return tw_tvt, calibrated, pseudo, slope, intercept


# ---------------------------------------------------------------------------
# Pool and folds
# ---------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
required = {"id", "tvt_true"}
if not required.issubset(header):
    raise RuntimeError(f"pool misses {sorted(required - set(header))}")


def _pf_feature_path(well):
    names = (
        f"{well}__official_full.pkl",
        f"{well}__official__full.pkl",
        f"{well}__full.pkl",
    )
    for directory in PF_FEATURE_DIRS:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None


def _frame_row_index(frame):
    if "_row_idx" in frame:
        return pd.to_numeric(frame["_row_idx"], errors="coerce").to_numpy(float)
    if "row_idx" in frame:
        return pd.to_numeric(frame["row_idx"], errors="coerce").to_numpy(float)
    if "id" in frame:
        return pd.to_numeric(
            frame["id"].astype(str).str.rsplit("_", n=1).str[-1],
            errors="coerce",
        ).to_numpy(float)
    raise RuntimeError("PF feature frame has no row identity")


feature_probe = None
for directory in PF_FEATURE_DIRS:
    if directory.exists():
        feature_probe = next(iter(directory.glob("*.pkl")), None)
        if feature_probe is not None:
            break
feature_columns = []
if feature_probe is not None:
    feature_columns = pd.read_pickle(feature_probe).columns.astype(str).tolist()
pf_columns = [
    column for column in PF_WANTED
    if column in header
    or column in feature_columns
    or column + "_y" in feature_columns
    or column + "_x" in feature_columns
    or column + "_d" in feature_columns
]
if len(pf_columns) < 3:
    raise RuntimeError(
        "too few PF paths across pool and fold-clean feature pickles: "
        + repr(
            {
                "pool_matches": [column for column in PF_WANTED if column in header],
                "feature_probe": str(feature_probe),
                "feature_matches": pf_columns,
            }
        )
    )
base_column = next((column for column in BASE_PREFERENCE if column in pf_columns), None)
if base_column is None:
    raise RuntimeError("no usable PF centre column")
usecols = [
    column for column in ("id", "well", "row_idx", "tvt_true", *pf_columns)
    if column in header
]
pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": str, "well": str})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pd.to_numeric(pool["row_idx"], errors="raise").astype(np.int32)
for column in ("tvt_true", *[column for column in pf_columns if column in pool]):
    pool[column] = pd.to_numeric(pool[column], errors="coerce").astype(np.float32)

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
folds = folds.groupby("well", as_index=False)["fold"].first()
fold_map = dict(zip(folds["well"].astype(str), pd.to_numeric(folds["fold"], errors="raise").astype(int)))
pool["fold"] = pool["well"].map(fold_map)
if pool["fold"].isna().any():
    raise RuntimeError("pool contains wells missing from fold metadata")
pool["fold"] = pool["fold"].astype(np.int8)
pool = pool[pool["fold"].isin((*CORE_FOLDS, SELECTION_FOLD, VALID_FOLD))]
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool_groups = {well: frame.copy() for well, frame in pool.groupby("well", sort=False)}

core_wells = sorted(pool.loc[pool["fold"].isin(CORE_FOLDS), "well"].unique())
selection_wells = sorted(pool.loc[pool["fold"].eq(SELECTION_FOLD), "well"].unique())
outer_wells = sorted(pool.loc[pool["fold"].isin(OUTER_FOLDS), "well"].unique())
valid_wells = sorted(pool.loc[pool["fold"].eq(VALID_FOLD), "well"].unique())
expected_sizes = (309, 154, 463, 155)
actual_sizes = (len(core_wells), len(selection_wells), len(outer_wells), len(valid_wells))
if actual_sizes != expected_sizes:
    raise RuntimeError(f"unexpected nested fold sizes: {actual_sizes}")


def _load_pfmm(well, row_idx):
    if PFMM_DIR is None:
        return {}, []
    path = PFMM_DIR / f"{well}.npz"
    if not path.exists():
        return {}, []
    with np.load(path, allow_pickle=False) as data:
        source_rows = data["eval_pos"].astype(np.int32)
        names = data["names"].astype(str).tolist()
        matrix = data["features"].astype(np.float32)
    locations = np.searchsorted(source_rows, row_idx)
    if np.any(locations >= len(source_rows)) or not np.array_equal(source_rows[locations], row_idx):
        return {}, []
    selected = [name for name in PFMM_WANTED if name in names]
    index = {name: names.index(name) for name in selected}
    return {name: matrix[locations, index[name]].astype(float) for name in selected}, selected


def _load_pf_paths(well, row_idx, pool_frame, anchor_tvt):
    paths = {}
    missing = [column for column in pf_columns if column not in pool_frame]
    indexed = None
    if missing:
        path = _pf_feature_path(well)
        if path is None:
            raise FileNotFoundError(f"{well}: fold-clean PF feature pickle not found")
        source = pd.read_pickle(path)
        source_rows = _frame_row_index(source)
        valid = np.isfinite(source_rows)
        indexed = source.loc[valid].copy()
        indexed["_pf_row_idx"] = source_rows[valid].astype(np.int32)
        indexed = indexed.drop_duplicates("_pf_row_idx", keep="last").set_index("_pf_row_idx")
        absent_rows = np.setdiff1d(row_idx, indexed.index.to_numpy(np.int32))
        if len(absent_rows):
            raise RuntimeError(f"{well}: PF feature pickle misses {len(absent_rows)} eval rows")

    for column in pf_columns:
        if column in pool_frame:
            values = pool_frame[column].to_numpy(float)
        else:
            candidates = (column, column + "_y", column + "_x", column + "_d")
            source_column = next((name for name in candidates if name in indexed.columns), None)
            if source_column is None:
                raise RuntimeError(f"{well}: PF column {column} is unavailable")
            values = pd.to_numeric(
                indexed.loc[row_idx, source_column], errors="coerce"
            ).to_numpy(float)
            if source_column.endswith("_d"):
                if "last_known_tvt" in indexed.columns:
                    last = pd.to_numeric(
                        indexed.loc[row_idx, "last_known_tvt"], errors="coerce"
                    ).to_numpy(float)
                    values = values + last
                else:
                    values = values + float(anchor_tvt)
        paths[column] = _fill(values)
    return paths


def _build_record(well):
    output = CACHE_DIR / f"{well}.npz"
    cache_version = f"pfbigru_v0b_pfmm{int(PFMM_DIR is not None)}"
    if output.exists():
        try:
            with np.load(output, allow_pickle=False) as data:
                if str(data["cache_version"][0]) == cache_version:
                    return {"well": well, "status": "cached", "nodes": len(data["node_md"])}
        except Exception:
            pass

    frame = pool_groups[well].sort_values("row_idx")
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    row_idx = frame["row_idx"].to_numpy(np.int32)
    if row_idx.max() >= len(horizontal):
        raise RuntimeError(f"{well}: pool row exceeds horizontal data")

    md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z_all = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    gr_all_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) < 16:
        raise RuntimeError(f"{well}: insufficient visible prefix")
    anchor = int(known[-1])
    anchor_md = float(md_all[anchor])
    anchor_tvt = float(tvt_input[anchor])
    anchor_z = float(z_all[anchor])

    md = md_all[row_idx]
    z = z_all[row_idx]
    gr_raw = gr_all_raw[row_idx]
    gr_filled_all = _fill(gr_all_raw)
    gr = gr_filled_all[row_idx]
    truth = frame["tvt_true"].to_numpy(float)
    paths = _load_pf_paths(well, row_idx, frame, anchor_tvt)
    if not np.isfinite(truth).all():
        raise RuntimeError(f"{well}: non-finite truth")
    base = paths[base_column]

    tw_tvt, tw_cal, tw_pseudo, cal_slope, cal_intercept = _calibrated_pseudo_typewell(horizontal, typewell)
    gr_centre, gr_scale = _centre_scale(gr_all_raw, np.isfinite(gr_all_raw))
    gr_norm = _normalise(gr, gr_centre, gr_scale)
    gr_valid = np.isfinite(gr_raw).astype(float)

    parts = []
    names = []
    pf_names = []
    gr_names = []

    def add(name, values, family="context"):
        values = np.asarray(values, np.float32)
        if values.ndim == 1:
            values = values[None]
        parts.append(values)
        local_names = [name] if values.shape[0] == 1 else [f"{name}_{i}" for i in range(values.shape[0])]
        names.extend(local_names)
        if family == "pf":
            pf_names.extend(local_names)
        if family == "gr":
            gr_names.extend(local_names)

    distance = md - anchor_md
    progress = np.clip(distance / max(float(distance[-1]), 1.0), 0.0, 1.0)
    ramp = 1.0 - np.exp(-np.maximum(distance, 0.0) / ANCHOR_RAMP_FT)
    add("geometry_progress", progress)
    add("geometry_sqrt_progress", np.sqrt(progress))
    add("geometry_distance", distance / 5000.0)
    add("geometry_distance_to_toe", (md[-1] - md) / 5000.0)
    add("geometry_z_relative", (z - anchor_z) / 128.0)
    add("geometry_dz_dmd", _gradient(z, md) * 16.0)
    add("geometry_d2z", _gradient(_gradient(z, md), md) * 1024.0)

    add("gr_value", gr_norm, "gr")
    add("gr_valid", gr_valid, "gr")
    add("gr_d1", _gradient(gr_norm, md) * 16.0, "gr")
    for window in (9, 33, 125):
        smooth = _smooth(gr_norm, window)
        add(f"gr_smooth{window}", smooth, "gr")
        add(f"gr_highpass{window}", gr_norm - smooth, "gr")
        local_std = np.sqrt(np.maximum(_smooth(gr_norm**2, window) - smooth**2, 0.0))
        add(f"gr_std{window}", local_std, "gr")

    path_stack = np.stack([paths[column] for column in pf_columns], axis=0)
    path_mean = np.mean(path_stack, axis=0)
    path_std = np.std(path_stack, axis=0)
    add("pf_family_mean", (path_mean - anchor_tvt) / 128.0, "pf")
    add("pf_family_std", path_std / 32.0, "pf")
    add("pf_family_range", np.ptp(path_stack, axis=0) / 64.0, "pf")
    for column, values in paths.items():
        add(f"pf_{column}_anchor", (values - anchor_tvt) / 128.0, "pf")
        add(f"pf_{column}_d1", _gradient(values + z, md) * 16.0, "pf")
        add(f"pf_{column}_vs_centre", (values - base) / 32.0, "pf")
        expected = np.interp(values, tw_tvt, tw_pseudo, left=np.nan, right=np.nan)
        plus = np.interp(values + 24.0, tw_tvt, tw_pseudo, left=np.nan, right=np.nan)
        minus = np.interp(values - 24.0, tw_tvt, tw_pseudo, left=np.nan, right=np.nan)
        expected_norm = _normalise(expected, gr_centre, gr_scale)
        plus_norm = _normalise(plus, gr_centre, gr_scale)
        minus_norm = _normalise(minus, gr_centre, gr_scale)
        residual = gr_norm - expected_norm
        alias = np.minimum(np.abs(gr_norm - plus_norm), np.abs(gr_norm - minus_norm)) - np.abs(residual)
        add(f"match_{column}_residual", residual, "pf")
        add(f"match_{column}_alias24", alias, "pf")

    sidecar, sidecar_names = _load_pfmm(well, row_idx)
    for name in sidecar_names:
        values = sidecar[name]
        if name.endswith("_d"):
            values = values / 128.0
        elif "std" in name or "iqr" in name or "q90_q10" in name:
            values = values / 32.0
        add(name, values, "pf")

    full_features = np.concatenate(parts, axis=0).astype(np.float32)
    nodes = _node_indices(md)
    feature_node = full_features[:, nodes]
    target_node = (truth[nodes] - base[nodes]).astype(np.float32)
    node_md = md[nodes].astype(np.float32)
    node_z = z[nodes].astype(np.float32)
    ramp_node = ramp[nodes].astype(np.float32)

    np.savez_compressed(
        output,
        cache_version=np.asarray([cache_version]),
        features=feature_node.astype(np.float16),
        feature_names=np.asarray(names),
        pf_feature_names=np.asarray(pf_names),
        gr_feature_names=np.asarray(gr_names),
        target=target_node,
        node_md=node_md,
        node_z=node_z,
        ramp=ramp_node,
        eval_md=md.astype(np.float32),
        distance_full=distance.astype(np.float32),
        truth_full=truth.astype(np.float32),
        base_full=base.astype(np.float32),
        fold=np.asarray([int(frame["fold"].iloc[0])], np.int8),
        calibration=np.asarray([cal_slope, cal_intercept], np.float32),
        sidecar_channels=np.asarray([len(sidecar_names)], np.int16),
    )
    return {
        "well": well,
        "status": "built",
        "nodes": len(nodes),
        "features": len(names),
        "sidecar_channels": len(sidecar_names),
    }


build_started = time.time()
all_wells = sorted(set(core_wells + selection_wells + valid_wells))
if PFMM_DIR is not None:
    pfmm_coverage = float(
        np.mean([(PFMM_DIR / f"{well}.npz").exists() for well in all_wells])
    )
    if pfmm_coverage < 1.0:
        print(
            "PF-BIGRU disabling incomplete PFMM sidecar",
            {"coverage": pfmm_coverage, "directory": str(PFMM_DIR)},
            flush=True,
        )
        PFMM_DIR = None
build_rows = []
for index, well in enumerate(all_wells, 1):
    try:
        build_rows.append(_build_record(well))
    except Exception as exc:
        build_rows.append({"well": well, "status": "failed", "reason": repr(exc)})
    if index % 50 == 0 or index == len(all_wells):
        print(f"PF-BIGRU build records {index}/{len(all_wells)}", flush=True)
build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "record_build_report.csv", index=False)
failures = build_report[build_report["status"].eq("failed")]
if len(failures):
    raise RuntimeError(f"{len(failures)} record builds failed: {failures.head(10).to_dict(orient='records')}")
fresh_schema = build_report.loc[build_report["status"].eq("built"), "features"].dropna().unique()
if len(fresh_schema) > 1:
    raise RuntimeError(f"record feature schema mismatch: {fresh_schema.tolist()}")


def _load_record(well):
    with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


probe = _load_record(all_wells[0])
feature_names = probe["feature_names"].astype(str).tolist()
pf_feature_names = set(probe["pf_feature_names"].astype(str).tolist())
gr_feature_names = set(probe["gr_feature_names"].astype(str).tolist())
pf_indices = np.asarray([i for i, name in enumerate(feature_names) if name in pf_feature_names], np.int64)
gr_indices = np.asarray([i for i, name in enumerate(feature_names) if name in gr_feature_names], np.int64)
if len(pf_indices) < 15:
    raise RuntimeError(f"PF feature block unexpectedly small: {len(pf_indices)}")


def _fit_scaler(wells):
    total = np.zeros(len(feature_names), np.float64)
    total2 = np.zeros(len(feature_names), np.float64)
    count = 0
    for well in wells:
        values = _load_record(well)["features"].astype(np.float64)
        total += values.sum(axis=1)
        total2 += np.square(values).sum(axis=1)
        count += values.shape[1]
    mean = total / max(count, 1)
    variance = np.maximum(total2 / max(count, 1) - mean**2, 1e-4)
    scale = np.sqrt(variance)
    scale = np.clip(scale, 0.05, 20.0)
    return mean.astype(np.float32), scale.astype(np.float32)


class WellDataset(Dataset):
    def __init__(self, wells, mean, scale):
        self.wells = list(wells)
        self.mean = mean[:, None]
        self.scale = scale[:, None]

    def __len__(self):
        return len(self.wells)

    def __getitem__(self, index):
        well = self.wells[index]
        data = _load_record(well)
        features = (data["features"].astype(np.float32) - self.mean) / self.scale
        return {
            "well": well,
            "features": features.T.astype(np.float32),
            "target": data["target"].astype(np.float32),
            "ramp": data["ramp"].astype(np.float32),
        }


def _collate(batch):
    lengths = np.asarray([len(item["target"]) for item in batch], np.int64)
    maximum = int(lengths.max())
    features = np.zeros((len(batch), maximum, len(feature_names)), np.float32)
    target = np.zeros((len(batch), maximum), np.float32)
    ramp = np.zeros((len(batch), maximum), np.float32)
    mask = np.zeros((len(batch), maximum), np.float32)
    wells = []
    for i, item in enumerate(batch):
        length = len(item["target"])
        features[i, :length] = item["features"]
        target[i, :length] = item["target"]
        ramp[i, :length] = item["ramp"]
        mask[i, :length] = 1.0
        wells.append(item["well"])
    return {
        "wells": wells,
        "features": torch.from_numpy(features),
        "target": torch.from_numpy(target),
        "ramp": torch.from_numpy(ramp),
        "mask": torch.from_numpy(mask),
        "lengths": torch.from_numpy(lengths),
    }


def _loader(wells, mean, scale, shuffle):
    return DataLoader(
        WellDataset(wells, mean, scale),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=_collate,
    )


class PFPPosteriorBiGRU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(channels, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            HIDDEN,
            HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, lengths, ramp):
        hidden = self.input(features)
        packed = pack_padded_sequence(
            hidden, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(
            packed, batch_first=True, total_length=features.shape[1]
        )
        correction = MAX_CORRECTION_FT * torch.tanh(self.head(hidden).squeeze(-1))
        return correction * ramp


def _loss(prediction, target, mask):
    denominator = torch.clamp(mask.sum(), min=1.0)
    squared = torch.square(prediction - target)
    point_mse = torch.sum(squared * mask) / denominator
    huber = F.smooth_l1_loss(prediction, target, beta=4.0, reduction="none")
    point_huber = torch.sum(huber * mask) / denominator

    delta_mask = mask[:, 1:] * mask[:, :-1]
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    delta = F.smooth_l1_loss(pred_delta, target_delta, beta=1.0, reduction="none")
    delta_loss = torch.sum(delta * delta_mask) / torch.clamp(delta_mask.sum(), min=1.0)

    lengths = torch.clamp(mask.sum(dim=1), min=1.0)
    mean_error = torch.sum((prediction - target) * mask, dim=1) / lengths
    datum_loss = torch.mean(torch.square(mean_error))
    return 0.70 * point_mse + 0.30 * point_huber + 0.10 * delta_loss + 0.20 * datum_loss


def _train_epoch(model, loader, optimizer, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    batches = 0
    for step, batch in enumerate(loader, 1):
        features = batch["features"].to(DEVICE, non_blocking=True)
        target = batch["target"].to(DEVICE, non_blocking=True)
        ramp = batch["ramp"].to(DEVICE, non_blocking=True)
        mask = batch["mask"].to(DEVICE, non_blocking=True)
        lengths = batch["lengths"]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            prediction = model(features, lengths, ramp)
            loss = _loss(prediction, target, mask) / GRAD_ACCUM
        scaler.scale(loss).backward()
        if step % GRAD_ACCUM == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu()) * GRAD_ACCUM
        batches += 1
    return total / max(batches, 1)


def _intervention_features(features, mode, donor=None):
    output = features.copy()
    if mode == "no_pf":
        output[:, pf_indices] = 0.0
    elif mode == "no_gr":
        output[:, gr_indices] = 0.0
    elif mode == "shuffled_pf":
        if donor is None:
            raise ValueError("shuffled_pf requires donor")
        source = donor[:, pf_indices]
        if len(source) == len(output):
            output[:, pf_indices] = source
        else:
            old = np.linspace(0.0, 1.0, len(source))
            new = np.linspace(0.0, 1.0, len(output))
            for local, channel in enumerate(pf_indices):
                output[:, channel] = np.interp(new, old, source[:, local])
    return output


@torch.no_grad()
def _predict_nodes(model, wells, mean, scale, mode="full"):
    model.eval()
    predictions = {}
    donors = {well: wells[(index + 37) % len(wells)] for index, well in enumerate(wells)}
    for well in wells:
        data = _load_record(well)
        features = (data["features"].astype(np.float32) - mean[:, None]) / scale[:, None]
        features = features.T
        donor = None
        if mode == "shuffled_pf":
            donor_data = _load_record(donors[well])
            donor = ((donor_data["features"].astype(np.float32) - mean[:, None]) / scale[:, None]).T
        features = _intervention_features(features, mode, donor)
        tensor = torch.from_numpy(features[None]).to(DEVICE)
        ramp = torch.from_numpy(data["ramp"].astype(np.float32)[None]).to(DEVICE)
        lengths = torch.as_tensor([features.shape[0]], dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            correction = model(tensor, lengths, ramp)[0].float().cpu().numpy()
        predictions[well] = correction
    return predictions


def _full_frame(wells, node_predictions, dose):
    rows = []
    for well in wells:
        data = _load_record(well)
        node_md = data["node_md"].astype(float)
        eval_md = data["eval_md"].astype(float)
        correction = np.interp(eval_md, node_md, node_predictions[well]) * float(dose)
        base = data["base_full"].astype(float)
        truth = data["truth_full"].astype(float)
        rows.append(
            pd.DataFrame(
                {
                    "well": well,
                    "truth": truth,
                    "base": base,
                    "prediction": base + correction,
                    "correction": correction,
                    "distance_ft": data["distance_full"].astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _metrics(frame):
    truth = frame["truth"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    error = prediction - truth
    bywell_rows = []
    for well, group in frame.groupby("well", sort=False):
        local = group["prediction"].to_numpy(float) - group["truth"].to_numpy(float)
        bywell_rows.append({"well": well, "rows": len(group), "sse": float(np.sum(local**2)), "rmse": float(np.sqrt(np.mean(local**2)))})
    bywell = pd.DataFrame(bywell_rows)
    result = {
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
    }
    distance = frame["distance_ft"].to_numpy(float)
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result, bywell


def _dose_scan(frame_zero, node_predictions, wells):
    rows = []
    for dose in DOSES:
        frame = _full_frame(wells, node_predictions, dose)
        metrics, _ = _metrics(frame)
        rows.append({"dose": dose, **metrics})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def _make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")


started = time.time()
core_mean, core_scale = _fit_scaler(core_wells)
core_loader = _loader(core_wells, core_mean, core_scale, shuffle=True)
selection_base = _full_frame(
    selection_wells,
    {well: np.zeros(len(_load_record(well)["node_md"]), np.float32) for well in selection_wells},
    0.0,
)
selection_base_metrics, _ = _metrics(selection_base)

model = PFPPosteriorBiGRU(len(feature_names)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=4e-5)
scaler = _make_scaler()
history = []
best = None
bad_epochs = 0

print(
    "PF-BIGRU CONFIG",
    {
        "device": str(DEVICE),
        "core_wells": len(core_wells),
        "selection_wells": len(selection_wells),
        "valid_wells": len(valid_wells),
        "features": len(feature_names),
        "pf_features": len(pf_indices),
        "gr_features": len(gr_indices),
        "base_column": base_column,
        "pfmm_dir": str(PFMM_DIR) if PFMM_DIR else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "record_build_minutes": (time.time() - build_started) / 60.0,
    },
    flush=True,
)
print("PF-BIGRU SELECTION BASE", selection_base_metrics, flush=True)

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = _train_epoch(model, core_loader, optimizer, scaler)
    scheduler.step()
    node_prediction = _predict_nodes(model, selection_wells, core_mean, core_scale)
    scan = _dose_scan(selection_base, node_prediction, selection_wells)
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "dose": float(scan.iloc[0]["dose"]),
        "rmse": float(scan.iloc[0]["rmse"]),
        "median_well_rmse": float(scan.iloc[0]["median_well_rmse"]),
        "p95_well_rmse": float(scan.iloc[0]["p95_well_rmse"]),
        "far_gt2000_rmse": float(scan.iloc[0]["far_gt2000_rmse"]),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    history.append(row)
    print("PF-BIGRU SELECTION EPOCH", row, flush=True)
    if best is None or row["rmse"] < best["rmse"] - 1e-4:
        best = {**row, "state": copy.deepcopy(model.state_dict())}
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs >= PATIENCE:
        print("PF-BIGRU selection early stopping", flush=True)
        break

history_frame = pd.DataFrame(history)
history_frame.to_csv(OUT_DIR / "selection_history.csv", index=False)
selected_epoch = int(best["epoch"])
selected_dose = float(best["dose"])
torch.save(
    {
        "model": best["state"],
        "epoch": selected_epoch,
        "dose": selected_dose,
        "feature_names": feature_names,
        "mean": core_mean,
        "scale": core_scale,
    },
    MODEL_DIR / "selection_model.pt",
)

# Frozen outer refit.  Fold 2 has not been loaded through a model before this.
del model, optimizer, scheduler, scaler, core_loader
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
_seed_everything(SEED + 101)
outer_mean, outer_scale = _fit_scaler(outer_wells)
outer_loader = _loader(outer_wells, outer_mean, outer_scale, shuffle=True)
outer_model = PFPPosteriorBiGRU(len(feature_names)).to(DEVICE)
outer_optimizer = torch.optim.AdamW(outer_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
outer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    outer_optimizer, T_max=max(selected_epoch, 1), eta_min=4e-5
)
outer_scaler = _make_scaler()
outer_history = []
for epoch in range(1, selected_epoch + 1):
    train_loss = _train_epoch(outer_model, outer_loader, outer_optimizer, outer_scaler)
    outer_scheduler.step()
    report = {"epoch": epoch, "train_loss": train_loss, "learning_rate": outer_optimizer.param_groups[0]["lr"]}
    outer_history.append(report)
    print("PF-BIGRU OUTER EPOCH", report, flush=True)

torch.save(
    {
        "model": outer_model.state_dict(),
        "epoch": selected_epoch,
        "dose": selected_dose,
        "feature_names": feature_names,
        "mean": outer_mean,
        "scale": outer_scale,
        "base_column": base_column,
        "pf_columns": pf_columns,
    },
    MODEL_DIR / "outer345_fold2_gateway.pt",
)
pd.DataFrame(outer_history).to_csv(OUT_DIR / "outer_training_history.csv", index=False)

intervention_rows = []
frames = {}
for mode in ("full", "no_pf", "no_gr", "shuffled_pf"):
    node_prediction = _predict_nodes(outer_model, valid_wells, outer_mean, outer_scale, mode=mode)
    frame = _full_frame(valid_wells, node_prediction, selected_dose)
    metrics, bywell = _metrics(frame)
    frames[mode] = frame
    bywell.to_csv(OUT_DIR / f"fold2_bywell_{mode}.csv", index=False)
    intervention_rows.append({"mode": mode, **metrics})
interventions = pd.DataFrame(intervention_rows)
full_metrics = interventions.loc[interventions["mode"].eq("full")].iloc[0]
for column in ("rmse", "far_gt2000_rmse"):
    interventions[f"degradation_{column}_vs_full"] = interventions[column] - float(full_metrics[column])
interventions.to_csv(OUT_DIR / "fold2_interventions.csv", index=False)
frames["full"].to_pickle(OUT_DIR / "fold2_predictions.pkl")

base_rmse = float(interventions.loc[interventions["mode"].eq("full"), "rmse"].iloc[0] + 0.0)
base_frame = frames["full"].copy()
base_frame["prediction"] = base_frame["base"]
base_metrics, _ = _metrics(base_frame)
full = interventions.loc[interventions["mode"].eq("full")].iloc[0].to_dict()
no_pf = interventions.loc[interventions["mode"].eq("no_pf")].iloc[0].to_dict()
shuffled = interventions.loc[interventions["mode"].eq("shuffled_pf")].iloc[0].to_dict()
no_gr = interventions.loc[interventions["mode"].eq("no_gr")].iloc[0].to_dict()

positive_control_pass = bool(base_metrics["rmse"] <= 12.0)
path_pass = bool(full["rmse"] <= 8.5 and full["median_well_rmse"] <= 6.5)
pf_causal_pass = bool(
    no_pf["rmse"] >= full["rmse"] + 0.30
    and shuffled["rmse"] >= full["rmse"] + 0.30
)
gr_causal_pass = bool(no_gr["rmse"] >= full["rmse"] + 0.20)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_fold2_pfposterior_completewell_bigru_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_cp12_gold_spatial": False,
    "protocol": {
        "core_train_folds": list(CORE_FOLDS),
        "epoch_dose_selection_fold": SELECTION_FOLD,
        "outer_train_folds": list(OUTER_FOLDS),
        "untouched_validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_epoch": selected_epoch,
        "selected_dose": selected_dose,
    },
    "inputs": {
        "base_column": base_column,
        "pf_columns": pf_columns,
        "pfmm_sidecar": str(PFMM_DIR) if PFMM_DIR else None,
        "feature_channels": len(feature_names),
        "pf_feature_channels": len(pf_indices),
        "gr_feature_channels": len(gr_indices),
        "pseudo_heel_weight": PSEUDO_HEEL_WEIGHT,
        "node_step_ft": NODE_STEP_FT,
    },
    "parameters": sum(parameter.numel() for parameter in outer_model.parameters()),
    "positive_control_base": base_metrics,
    "fold3_best": {key: value for key, value in best.items() if key != "state"},
    "fold2": full,
    "fold2_interventions": interventions.to_dict(orient="records"),
    "gates": {
        "PF_base_rmse_le": 12.0,
        "model_rmse_le": 8.5,
        "model_median_well_rmse_le": 6.5,
        "no_pf_degradation_ge": 0.30,
        "shuffled_pf_degradation_ge": 0.30,
        "no_gr_degradation_ge": 0.20,
    },
    "positive_control_pass": positive_control_pass,
    "path_pass": path_pass,
    "pf_causal_pass": pf_causal_pass,
    "gr_causal_pass": gr_causal_pass,
    "strict_pass": bool(positive_control_pass and path_pass and pf_causal_pass and gr_causal_pass),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> add quadratic dip-consistency head and expand to five-fold multi-seed"
        if positive_control_pass and path_pass and pf_causal_pass and gr_causal_pass
        else "FAIL -> PF posterior features do not close the complete-well transfer gap"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSELECTION HISTORY")
print(history_frame.to_string(index=False), flush=True)
print("\nFOLD2 INTERVENTIONS")
print(interventions.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
