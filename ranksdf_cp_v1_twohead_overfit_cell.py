"""RankSDF-CP v1 phase 2: exact 12-well two-head overfit gate.

Implements the PDF specification rather than the earlier simplified v0:

- S = TVT + Z coordinate system;
- 193 candidate states at +/-96 ft and 1 ft spacing;
- horizontal aggregation near 16 ft, padded to 768 columns;
- the frozen 24-channel stage-A panel;
- four-level base-24 GroupNorm U-Net;
- signed-distance and ranking-energy heads;
- 1.0 SDF + 0.5 rank + 0.2 softargmin + 0.05 piecewise-S loss.

This is deliberately an overfit test on the same 12 wells used for training.
It must pass before pseudo-PS or synthetic pretraining is attempted.
"""

from pathlib import Path
import json
import random
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
AUDIT_DIR = OOF_ROOT / "ranksdf_cp_v1_exact_panel_audit"
MANIFEST_PATH = AUDIT_DIR / "panel_manifest.json"
OUT_DIR = OOF_ROOT / "ranksdf_cp_v1_twohead_overfit"
CACHE_DIR = OUT_DIR / "panels"
MODEL_DIR = OUT_DIR / "models"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]

N_WELLS = 12
STATE_HALF_WIDTH_FT = 96.0
STATE_STEP_FT = 1.0
OFFSETS = np.arange(
    -STATE_HALF_WIDTH_FT,
    STATE_HALF_WIDTH_FT + 0.25 * STATE_STEP_FT,
    STATE_STEP_FT,
    dtype=np.float32,
)
TARGET_MD_STEP_FT = 16.0
MAX_COLUMNS = 768
SDF_SCALE_FT = 40.0
SOFT_LABEL_SIGMA_FT = 3.0

BASE_CHANNELS = 24
BATCH_SIZE = 1
EPOCHS = 120
EVAL_EVERY = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 20260716

LOSS_SDF = 1.0
LOSS_RANK = 0.5
LOSS_SOFTARGMIN = 0.2
LOSS_PIECEWISE = 0.05

PASS_EVAL_RMSE = 2.0
PASS_RAW_KNOWN_RMSE = 2.0
PASS_HARD_KNOWN_RMSE = 0.05

CHANNELS = [
    "typewell_gr",
    "horizontal_gr",
    "gr_difference",
    "gr_abs_difference",
    "gr_difference_smooth4ft",
    "gr_difference_smooth8ft",
    "gr_difference_smooth16ft",
    "gr_difference_smooth32ft",
    "typewell_valid",
    "horizontal_gr_valid",
    "known_path_sdf",
    "known_mask",
    "state_offset_coordinate",
    "md_relative",
    "dz_dmd",
    "d2z_dmd2",
    "curvature_xy",
    "azimuth_sin",
    "azimuth_cos",
    "self_reference_gr",
    "self_reference_difference",
    "self_reference_abs_difference",
    "self_reference_valid",
    "self_reference_coverage",
]
TYPEWELL_VALID_CHANNEL = CHANNELS.index("typewell_valid")


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _fill(values):
    return (
        pd.Series(np.asarray(values, np.float64))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _normalise(values, valid=None):
    values = np.asarray(values, np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, bool) & np.isfinite(values)
    output = np.zeros_like(values, np.float32)
    sample = values[valid]
    if len(sample) < 8:
        return output
    median = float(np.median(sample))
    scale = float(1.4826 * np.median(np.abs(sample - median)))
    if scale < 1e-3:
        scale = float(np.std(sample) + 1e-3)
    output[valid] = np.clip((values[valid] - median) / scale, -6.0, 6.0)
    return output


def _discover(wells, suffix):
    wells = set(map(str, wells))
    output = {}
    for root in (DATA_ROOT / "train", DATA_ROOT):
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            if not path.name.endswith(suffix):
                continue
            well = path.name[: -len(suffix)]
            if well in wells and "test" not in {part.lower() for part in path.parts}:
                output.setdefault(well, path)
    missing = sorted(wells - set(output))
    if missing:
        raise FileNotFoundError(f"missing {suffix}: {missing[:20]}")
    return output


def _prefix_reference(tvt_input, gr_normalised, gr_valid):
    valid = np.isfinite(tvt_input) & np.asarray(gr_valid, bool)
    if valid.sum() < 32:
        return None
    tvt = np.asarray(tvt_input[valid], np.float64)
    gr = np.asarray(gr_normalised[valid], np.float64)
    bins = np.round(tvt * 2.0) / 2.0
    curve = (
        pd.DataFrame({"tvt": bins, "gr": gr})
        .groupby("tvt", as_index=False)["gr"]
        .median()
        .sort_values("tvt")
    )
    if len(curve) < 24 or float(curve["tvt"].max() - curve["tvt"].min()) < 16.0:
        return None
    return curve["tvt"].to_numpy(float), curve["gr"].to_numpy(float)


def _geometry(md, x, y, z):
    md = np.asarray(md, np.float64)
    edge = 2 if len(md) >= 3 else 1
    dz = np.gradient(z, md, edge_order=edge)
    d2z = np.gradient(dz, md, edge_order=edge)
    dx = np.gradient(x, md, edge_order=edge)
    dy = np.gradient(y, md, edge_order=edge)
    azimuth = np.unwrap(np.arctan2(dy, dx))
    dazimuth = np.gradient(azimuth, md, edge_order=edge)
    speed_xy = np.sqrt(dx * dx + dy * dy)
    curvature = np.abs(dazimuth) * speed_xy
    return (
        np.clip(dz, -1.0, 1.0).astype(np.float32),
        np.clip(d2z * 100.0, -5.0, 5.0).astype(np.float32),
        np.clip(curvature * 1000.0, 0.0, 5.0).astype(np.float32),
        np.sin(azimuth).astype(np.float32),
        np.cos(azimuth).astype(np.float32),
    )


def _select_wells(pool):
    meta = (
        pool.assign(
            se=(pool["strong"].to_numpy(float) - pool["tvt_true"].to_numpy(float)) ** 2
        )
        .groupby("well", as_index=False)
        .agg(rows=("id", "size"), sse=("se", "sum"))
    )
    meta["rmse"] = np.sqrt(meta["sse"] / meta["rows"])
    meta["error_bin"] = pd.qcut(meta["rmse"].rank(method="first"), 4, labels=False).astype(int)
    meta["length_bin"] = pd.qcut(meta["rows"].rank(method="first"), 3, labels=False).astype(int)
    selected = []
    for error_bin in range(4):
        for length_bin in range(3):
            cell = meta[(meta["error_bin"] == error_bin) & (meta["length_bin"] == length_bin)].copy()
            cell["hash"] = cell["well"].map(
                lambda value: zlib.crc32(f"{value}|{SEED}".encode())
            )
            selected.append(str(cell.sort_values("hash").iloc[0]["well"]))
    return selected[:N_WELLS], meta


def _build_panel(well, horizontal_path, typewell_path, pool_well):
    horizontal = pd.read_csv(horizontal_path)
    required = {"MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"}
    if not required.issubset(horizontal.columns):
        raise RuntimeError(f"{well} horizontal missing {sorted(required - set(horizontal.columns))}")
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    typewell = typewell[np.isfinite(typewell["TVT"])].sort_values("TVT")
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean()

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md) & np.isfinite(z))
    if len(known) < 32:
        raise RuntimeError(f"{well} has insufficient known heel")
    anchor = int(known[-1])
    anchor_md = float(md[anchor])

    pool_well = pool_well.sort_values("row_idx")
    eval_rows = np.clip(pool_well["row_idx"].to_numpy(int), 0, len(horizontal) - 1)
    center_tvt_rows = np.full(len(horizontal), np.nan, np.float64)
    center_tvt_rows[known] = tvt_input[known]
    center_tvt_rows[eval_rows] = pool_well["strong"].to_numpy(float)
    finite_center = np.isfinite(center_tvt_rows) & np.isfinite(md)
    if finite_center.sum() < 32:
        raise RuntimeError(f"{well} has insufficient fold-safe centre")
    center_tvt_rows = np.interp(md, md[finite_center], center_tvt_rows[finite_center])

    span = float(md[-1] - md[0])
    n_columns = int(np.ceil(span / TARGET_MD_STEP_FT)) + 1
    if n_columns > MAX_COLUMNS:
        raise RuntimeError(f"{well} requires {n_columns} columns > {MAX_COLUMNS}")
    sample_md = np.linspace(float(md[0]), float(md[-1]), n_columns)
    sample_x = np.interp(sample_md, md, x)
    sample_y = np.interp(sample_md, md, y)
    sample_z = np.interp(sample_md, md, z)
    sample_tvt = np.interp(sample_md, md, tvt)
    sample_center_tvt = np.interp(sample_md, md, center_tvt_rows)
    true_s = sample_tvt + sample_z
    center_s = sample_center_tvt + sample_z
    target_offset = true_s - center_s
    known_columns = sample_md <= anchor_md + 1e-6

    original_gr_valid = np.isfinite(gr_raw)
    gr_normalised_rows = _normalise(_fill(gr_raw), original_gr_valid)
    sample_gr = np.interp(sample_md, md, gr_normalised_rows).astype(np.float32)
    sample_gr_valid = np.interp(sample_md, md, original_gr_valid.astype(float)) >= 0.5

    candidate_s = center_s[None, :] + OFFSETS[:, None]
    candidate_tvt = candidate_s - sample_z[None, :]
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr_raw = typewell["GR"].to_numpy(float)
    tw_gr = _normalise(_fill(tw_gr_raw), np.isfinite(tw_gr_raw))
    typewell_valid = (candidate_tvt >= tw_tvt[0]) & (candidate_tvt <= tw_tvt[-1])
    tw_grid = np.interp(candidate_tvt.ravel(), tw_tvt, tw_gr).reshape(candidate_tvt.shape).astype(np.float32)
    tw_grid[~typewell_valid] = 0.0

    horizontal_grid = np.broadcast_to(sample_gr[None, :], candidate_tvt.shape).astype(np.float32)
    difference = tw_grid - horizontal_grid
    smooth = [
        uniform_filter1d(difference, size=size, axis=0, mode="nearest")
        for size in (4, 8, 16, 32)
    ]

    known_sdf = np.zeros_like(difference, np.float32)
    known_sdf[:, known_columns] = np.clip(
        (true_s[None, known_columns] - candidate_s[:, known_columns]) / SDF_SCALE_FT,
        -3.0,
        3.0,
    ).astype(np.float32)
    known_mask = np.broadcast_to(known_columns[None, :], difference.shape).astype(np.float32)
    offset_grid = np.broadcast_to((OFFSETS[:, None] / STATE_HALF_WIDTH_FT), difference.shape).astype(np.float32)
    md_relative = 2.0 * (sample_md - sample_md[0]) / max(span, 1.0) - 1.0

    dz, d2z, curvature, azimuth_sin, azimuth_cos = _geometry(
        sample_md, sample_x, sample_y, sample_z
    )

    prefix = _prefix_reference(tvt_input, gr_normalised_rows, original_gr_valid)
    if prefix is None:
        self_grid = np.zeros_like(difference, np.float32)
        self_valid = np.zeros_like(typewell_valid, bool)
    else:
        prefix_tvt, prefix_gr = prefix
        self_valid = (candidate_tvt >= prefix_tvt[0]) & (candidate_tvt <= prefix_tvt[-1])
        self_grid = np.interp(candidate_tvt.ravel(), prefix_tvt, prefix_gr).reshape(candidate_tvt.shape).astype(np.float32)
        self_grid[~self_valid] = 0.0
    self_difference = self_grid - horizontal_grid
    self_coverage = np.broadcast_to(self_valid.mean(axis=0)[None, :], difference.shape).astype(np.float32)

    def broadcast(values):
        return np.broadcast_to(np.asarray(values, np.float32)[None, :], difference.shape).astype(np.float32)

    panel = np.stack(
        [
            tw_grid,
            horizontal_grid,
            difference,
            np.abs(difference),
            *smooth,
            typewell_valid.astype(np.float32),
            broadcast(sample_gr_valid),
            known_sdf,
            known_mask,
            offset_grid,
            broadcast(md_relative),
            broadcast(dz),
            broadcast(d2z),
            broadcast(curvature),
            broadcast(azimuth_sin),
            broadcast(azimuth_cos),
            self_grid,
            self_difference,
            np.abs(self_difference),
            self_valid.astype(np.float32),
            self_coverage,
        ],
        axis=0,
    ).astype(np.float16)
    if panel.shape[0] != len(CHANNELS):
        raise RuntimeError(f"channel mismatch: {panel.shape}")

    padded = np.zeros((len(CHANNELS), len(OFFSETS), MAX_COLUMNS), np.float16)
    padded[:, :, :n_columns] = panel
    padded_target = np.zeros(MAX_COLUMNS, np.float32)
    padded_target[:n_columns] = target_offset.astype(np.float32)
    padded_center_s = np.zeros(MAX_COLUMNS, np.float32)
    padded_center_s[:n_columns] = center_s.astype(np.float32)
    column_valid = np.zeros(MAX_COLUMNS, np.uint8)
    column_valid[:n_columns] = 1
    padded_known = np.zeros(MAX_COLUMNS, np.uint8)
    padded_known[:n_columns] = known_columns.astype(np.uint8)

    return {
        "panel": padded,
        "target_offset": padded_target,
        "center_s": padded_center_s,
        "column_valid": column_valid,
        "known_columns": padded_known,
        "n_columns": np.asarray([n_columns], np.int32),
        "sample_md": sample_md.astype(np.float32),
        "sample_z": sample_z.astype(np.float32),
        "eval_md": md[eval_rows].astype(np.float32),
        "eval_tvt": pool_well["tvt_true"].to_numpy(np.float32),
        "eval_strong": pool_well["strong"].to_numpy(np.float32),
        "coverage": np.asarray([np.mean(np.abs(target_offset) <= STATE_HALF_WIDTH_FT)], np.float32),
    }


class PanelDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            return (
                torch.from_numpy(data["panel"].astype(np.float32)),
                torch.from_numpy(data["target_offset"].astype(np.float32)),
                torch.from_numpy(data["center_s"].astype(np.float32)),
                torch.from_numpy(data["column_valid"].astype(bool)),
                torch.from_numpy(data["known_columns"].astype(bool)),
            )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, values):
        return self.net(values)


class RankSDFUNet(nn.Module):
    def __init__(self, in_channels, base=24):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 12)
        self.d4 = ConvBlock(base * 20, base * 8)
        self.d3 = ConvBlock(base * 12, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.sdf_head = nn.Conv2d(base, 1, 1)
        self.rank_head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def up(values, skip):
        return F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, values):
        e1 = self.e1(values)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        mid = self.mid(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.up(mid, e4), e4], dim=1))
        d3 = self.d3(torch.cat([self.up(d4, e3), e3], dim=1))
        d2 = self.d2(torch.cat([self.up(d3, e2), e2], dim=1))
        d1 = self.d1(torch.cat([self.up(d2, e1), e1], dim=1))
        return self.sdf_head(d1).squeeze(1), self.rank_head(d1).squeeze(1)


def _loss(model, panel, target_offset, center_s, column_valid, offsets):
    sdf_prediction, rank_logits = model(panel)
    typewell_valid = panel[:, TYPEWELL_VALID_CHANNEL] > 0.5
    state_valid = typewell_valid & column_valid[:, None, :]
    target_sdf = torch.clamp(
        (target_offset[:, None, :] - offsets[None, :, None]) / SDF_SCALE_FT,
        -3.0,
        3.0,
    )
    sdf_loss = F.smooth_l1_loss(
        sdf_prediction[state_valid], target_sdf[state_valid]
    )

    masked_logits = rank_logits.masked_fill(~state_valid, -30.0)
    soft_target = torch.exp(
        -0.5
        * ((offsets[None, :, None] - target_offset[:, None, :]) / SOFT_LABEL_SIGMA_FT) ** 2
    ) * state_valid.float()
    soft_target = soft_target / torch.clamp(soft_target.sum(dim=1, keepdim=True), min=1e-8)
    log_probability = F.log_softmax(masked_logits, dim=1)
    per_column_rank = -(soft_target * log_probability).sum(dim=1)
    rank_loss = per_column_rank[column_valid].mean()

    probability = torch.softmax(masked_logits, dim=1)
    expected_offset = (probability * offsets[None, :, None]).sum(dim=1)
    softargmin_loss = F.smooth_l1_loss(
        expected_offset[column_valid] / 10.0,
        target_offset[column_valid] / 10.0,
    )

    predicted_s = center_s + expected_offset
    d2 = predicted_s[:, 2:] - 2.0 * predicted_s[:, 1:-1] + predicted_s[:, :-2]
    triple_valid = column_valid[:, 2:] & column_valid[:, 1:-1] & column_valid[:, :-2]
    piecewise_loss = F.smooth_l1_loss(
        d2[triple_valid] / 4.0,
        torch.zeros_like(d2[triple_valid]),
    )
    total = (
        LOSS_SDF * sdf_loss
        + LOSS_RANK * rank_loss
        + LOSS_SOFTARGMIN * softargmin_loss
        + LOSS_PIECEWISE * piecewise_loss
    )
    components = {
        "sdf": float(sdf_loss.detach().cpu()),
        "rank": float(rank_loss.detach().cpu()),
        "softargmin": float(softargmin_loss.detach().cpu()),
        "piecewise": float(piecewise_loss.detach().cpu()),
    }
    return total, components


@torch.no_grad()
def _predict_panel(model, path, device):
    with np.load(path, allow_pickle=False) as data:
        panel = data["panel"].astype(np.float32)
        target_offset = data["target_offset"].astype(float)
        center_s = data["center_s"].astype(float)
        column_valid = data["column_valid"].astype(bool)
        known = data["known_columns"].astype(bool)
        n_columns = int(data["n_columns"][0])
        sample_md = data["sample_md"].astype(float)
        sample_z = data["sample_z"].astype(float)
        eval_md = data["eval_md"].astype(float)
        eval_tvt = data["eval_tvt"].astype(float)
        eval_strong = data["eval_strong"].astype(float)
    tensor = torch.from_numpy(panel)[None].to(device)
    sdf_prediction, rank_logits = model(tensor)
    valid = tensor[:, TYPEWELL_VALID_CHANNEL] > 0.5
    valid &= torch.from_numpy(column_valid)[None, None].to(device)
    logits = rank_logits.masked_fill(~valid, -30.0)
    probability = torch.softmax(logits, dim=1)
    offset_tensor = torch.from_numpy(OFFSETS).to(device)
    rank_offset = (probability * offset_tensor[None, :, None]).sum(dim=1)[0].cpu().numpy()
    sdf_abs = torch.abs(sdf_prediction).masked_fill(~valid, 1e6)
    sdf_index = torch.argmin(sdf_abs, dim=1)[0].cpu().numpy()
    sdf_offset = OFFSETS[sdf_index]

    raw_known_rank_error = rank_offset[known] - target_offset[known]
    rank_offset_hard = rank_offset.copy()
    sdf_offset_hard = sdf_offset.copy()
    rank_offset_hard[known] = target_offset[known]
    sdf_offset_hard[known] = target_offset[known]
    sample_rank_tvt = center_s[:n_columns] + rank_offset_hard[:n_columns] - sample_z
    sample_sdf_tvt = center_s[:n_columns] + sdf_offset_hard[:n_columns] - sample_z
    eval_rank = np.interp(eval_md, sample_md, sample_rank_tvt)
    eval_sdf = np.interp(eval_md, sample_md, sample_sdf_tvt)
    return {
        "truth": eval_tvt,
        "strong": eval_strong,
        "rank": eval_rank,
        "sdf": eval_sdf,
        "raw_known_error": raw_known_rank_error,
        "hard_known_error": rank_offset_hard[known] - target_offset[known],
    }


def _rmse(error):
    error = np.asarray(error, np.float64)
    return float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan


@torch.no_grad()
def _evaluate(model, panel_paths, device):
    model.eval()
    records = []
    for well, path in panel_paths.items():
        prediction = _predict_panel(model, path, device)
        records.append(
            {
                "well": well,
                "rows": len(prediction["truth"]),
                "rank_sse": float(np.sum((prediction["rank"] - prediction["truth"]) ** 2)),
                "sdf_sse": float(np.sum((prediction["sdf"] - prediction["truth"]) ** 2)),
                "strong_sse": float(np.sum((prediction["strong"] - prediction["truth"]) ** 2)),
                "raw_known_sse": float(np.sum(prediction["raw_known_error"] ** 2)),
                "raw_known_rows": len(prediction["raw_known_error"]),
                "hard_known_sse": float(np.sum(prediction["hard_known_error"] ** 2)),
            }
        )
    frame = pd.DataFrame(records)
    rows = int(frame["rows"].sum())
    known_rows = int(frame["raw_known_rows"].sum())
    metrics = {
        "rank_rmse": float(np.sqrt(frame["rank_sse"].sum() / rows)),
        "sdf_rmse": float(np.sqrt(frame["sdf_sse"].sum() / rows)),
        "strong_rmse": float(np.sqrt(frame["strong_sse"].sum() / rows)),
        "raw_known_rmse": float(np.sqrt(frame["raw_known_sse"].sum() / max(known_rows, 1))),
        "hard_known_rmse": float(np.sqrt(frame["hard_known_sse"].sum() / max(known_rows, 1))),
        "rows": rows,
    }
    return metrics, frame


_seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Run phase-1 audit first: {MANIFEST_PATH}")
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
if manifest.get("version") != "ranksdf_cp_v1_pdf_exact":
    raise RuntimeError("unexpected panel manifest")
if manifest.get("channels_stage_a") != CHANNELS:
    raise RuntimeError("channel manifest mismatch")

pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
required_pool = {"id", "tvt_true", "strong"}
if not required_pool.issubset(header):
    raise RuntimeError(f"pool missing {sorted(required_pool - set(header))}")
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
selected_wells, selection_meta = _select_wells(pool)
horizontal_files = _discover(selected_wells, "__horizontal_well.csv")
typewell_files = _discover(selected_wells, "__typewell.csv")

panel_paths = {}
build_rows = []
for well in tqdm(selected_wells, desc="build exact two-head panels"):
    path = CACHE_DIR / f"{well}.npz"
    if not path.exists():
        record = _build_panel(
            well,
            horizontal_files[well],
            typewell_files[well],
            pool[pool["well"] == well],
        )
        np.savez_compressed(path, **record)
    with np.load(path, allow_pickle=False) as data:
        build_rows.append(
            {
                "well": well,
                "columns": int(data["n_columns"][0]),
                "coverage": float(data["coverage"][0]),
                "cache_mb": path.stat().st_size / 2**20,
            }
        )
    panel_paths[well] = path
build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
selection_meta.to_csv(OUT_DIR / "well_selection_population.csv", index=False)

print(
    {
        "experiment": "ranksdf_cp_v1_twohead_overfit",
        "device": str(DEVICE),
        "wells": selected_wells,
        "channels": len(CHANNELS),
        "states": len(OFFSETS),
        "max_columns": MAX_COLUMNS,
        "base_channels": BASE_CHANNELS,
        "loss_weights": {
            "sdf": LOSS_SDF,
            "rank": LOSS_RANK,
            "softargmin": LOSS_SOFTARGMIN,
            "piecewise_s": LOSS_PIECEWISE,
        },
        "cache_mb": float(build_report["cache_mb"].sum()),
    }
)

loader = DataLoader(
    PanelDataset(panel_paths.values()),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=NUM_WORKERS > 0,
)
model = RankSDFUNet(len(CHANNELS), BASE_CHANNELS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
offset_tensor = torch.from_numpy(OFFSETS).to(DEVICE)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

history = []
best_rmse = np.inf
best_path = MODEL_DIR / "ranksdf_cp_v1_twohead_overfit_best.pt"
started = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []
    component_rows = []
    for panel, target_offset, center_s, column_valid, _ in loader:
        panel = panel.to(DEVICE, non_blocking=True)
        target_offset = target_offset.to(DEVICE, non_blocking=True)
        center_s = center_s.to(DEVICE, non_blocking=True)
        column_valid = column_valid.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, components = _loss(
                model, panel, target_offset, center_s, column_valid, offset_tensor
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_losses.append(float(loss.detach().cpu()))
        component_rows.append(components)

    record = {
        "epoch": epoch,
        "train_loss": float(np.mean(epoch_losses)),
        **{
            f"loss_{name}": float(np.mean([row[name] for row in component_rows]))
            for name in ("sdf", "rank", "softargmin", "piecewise")
        },
    }
    if epoch == 1 or epoch % EVAL_EVERY == 0:
        metrics, _ = _evaluate(model, panel_paths, DEVICE)
        record.update(metrics)
        print("epoch", record, flush=True)
        if metrics["rank_rmse"] < best_rmse:
            best_rmse = metrics["rank_rmse"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "channels": CHANNELS,
                    "manifest": manifest,
                },
                best_path,
            )
        if (
            metrics["rank_rmse"] <= PASS_EVAL_RMSE
            and metrics["raw_known_rmse"] <= PASS_RAW_KNOWN_RMSE
            and metrics["hard_known_rmse"] <= PASS_HARD_KNOWN_RMSE
        ):
            print("overfit gate reached", flush=True)
            history.append(record)
            break
    history.append(record)

if not best_path.exists():
    raise RuntimeError("no model checkpoint was saved")
checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
metrics, by_well = _evaluate(model, panel_paths, DEVICE)
strict_pass = bool(
    metrics["rank_rmse"] <= PASS_EVAL_RMSE
    and metrics["raw_known_rmse"] <= PASS_RAW_KNOWN_RMSE
    and metrics["hard_known_rmse"] <= PASS_HARD_KNOWN_RMSE
)
verdict = {
    "experiment": "ranksdf_cp_v1_twohead_overfit",
    "selected_wells": selected_wells,
    "selected_epoch": int(checkpoint["epoch"]),
    **metrics,
    "sdf_diagnostic_rmse": metrics["sdf_rmse"],
    "gates": {
        "rank_eval_rmse_max": PASS_EVAL_RMSE,
        "raw_known_rmse_max": PASS_RAW_KNOWN_RMSE,
        "hard_known_rmse_max": PASS_HARD_KNOWN_RMSE,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build real-well pseudo-PS one-fold three-seed baseline"
        if strict_pass
        else "FAIL -> debug exact two-head tensor/loss before pseudo-PS"
    ),
    "runtime_minutes": (time.time() - started) / 60.0,
    "model": str(best_path),
}

pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
by_well.to_csv(OUT_DIR / "overfit_bywell.csv", index=False)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nBY WELL")
display(by_well)
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
