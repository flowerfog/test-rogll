"""RankSDF-CP v2 gateway: fixed absolute-S panel, one fold, one seed.

Unlike v1, each image row has one geological meaning across every MD column:
S_j = S at the last known row + offset_j.  Validation reports both pointwise
posterior means and an anchored beam decoder with a slope-change penalty.
This is a bounded architecture diagnostic, not a submission builder.
"""

from pathlib import Path
import gc
import json
import random
import shutil
import time

import joblib
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
AXIS_AUDIT = OOF_ROOT / "ranksdf_cp_v2_absolute_s_axis_audit" / "verdict.json"
BANK_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
EXTENSION_DIR = OOF_ROOT / "multicut_fleongg_matched_k5_extension_v1"
OUT_DIR = OOF_ROOT / "ranksdf_cp_v2_absolute_s_fold1_1seed"
CACHE_DIR = OUT_DIR / "panels"
MODEL_DIR = OUT_DIR / "models"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]

VALID_FOLD = 1
TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
TRAIN_KNOWN_FRAC_MIN = 0.10
TRAIN_KNOWN_FRAC_MAX = 0.40

if not AXIS_AUDIT.exists():
    raise FileNotFoundError(f"Run the absolute-S audit first: {AXIS_AUDIT}")
AXIS_SPEC = json.loads(AXIS_AUDIT.read_text(encoding="utf-8"))
if not AXIS_SPEC.get("strict_pass") or AXIS_SPEC.get("recommended_half_width_ft") is None:
    raise RuntimeError(f"Absolute-S audit did not pass: {AXIS_SPEC}")
STATE_HALF_WIDTH_FT = float(AXIS_SPEC["recommended_half_width_ft"])
# The +/-320 ft gateway uses 321 rows, close to the cost of a +/-160 ft grid
# at 1-ft resolution. A positive result can later be refined to 1-1.5 ft.
STATE_STEP_FT = 2.0
OFFSETS = np.arange(
    -STATE_HALF_WIDTH_FT,
    STATE_HALF_WIDTH_FT + 0.25 * STATE_STEP_FT,
    STATE_STEP_FT,
    dtype=np.float32,
)
TARGET_MD_STEP_FT = 16.0
MAX_COLUMNS = 768
SDF_SCALE_FT = 40.0
SOFT_LABEL_SIGMA_FT = 4.0

BASE_CHANNELS = 24
BATCH_SIZE = 1
EPOCHS = 12
EVAL_EVERY = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
N_JOBS_BUILD = 4
MIN_FREE_GB_BEFORE_BUILD = 10.5
SEEDS = (20260716,)

DP_BEAM_SIZE = 64
DP_MAX_STEP_STATES = 6
DP_SLOPE_CHANGE_PENALTY = 0.08
DP_SDF_WEIGHT = 0.10

LOSS_SDF = 1.0
LOSS_RANK = 0.5
LOSS_SOFTARGMIN = 0.2
LOSS_PIECEWISE = 0.05

PASS_GAIN_VS_STRONG = 0.20
PASS_POSITIVE_WELL_FRACTION = 0.55
PASS_MEDIAN_WELL_GAIN = 0.10
PASS_ERROR_COSINE = 0.85

CHANNELS = [
    "typewell_gr", "horizontal_gr", "gr_difference", "gr_abs_difference",
    "local_ncc_112ft", "local_ncc_208ft", "local_ncc_400ft", "local_ncc_816ft",
    "typewell_valid", "horizontal_gr_valid", "known_path_sdf", "known_mask",
    "state_offset_coordinate", "md_relative", "dz_dmd", "d2z_dmd2",
    "curvature_xy", "azimuth_sin", "azimuth_cos", "self_reference_gr",
    "self_reference_difference", "self_reference_abs_difference",
    "self_reference_valid", "self_reference_coverage",
]
TYPEWELL_VALID_CHANNEL = CHANNELS.index("typewell_valid")
NCC_WINDOW_COLUMNS = (7, 13, 25, 51)


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
    valid = np.isfinite(values) if valid is None else (np.asarray(valid, bool) & np.isfinite(values))
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


def _masked_local_ncc(reference, observed, valid, window_columns):
    """Missing-aware local NCC along MD for every absolute-S candidate row."""
    reference = np.asarray(reference, np.float32)
    observed = np.asarray(observed, np.float32)
    valid = np.asarray(valid, bool)
    weight = valid.astype(np.float32)

    def window_sum(values, size):
        return uniform_filter1d(
            values, size=size, axis=1, mode="constant", cval=0.0
        ) * float(size)

    outputs = []
    for size in window_columns:
        count = window_sum(weight, size)
        sum_x = window_sum(reference * weight, size)
        sum_y = window_sum(observed * weight, size)
        sum_x2 = window_sum(reference * reference * weight, size)
        sum_y2 = window_sum(observed * observed * weight, size)
        sum_xy = window_sum(reference * observed * weight, size)
        denominator_count = np.maximum(count, 1.0)
        mean_x = sum_x / denominator_count
        mean_y = sum_y / denominator_count
        covariance = sum_xy / denominator_count - mean_x * mean_y
        variance_x = np.maximum(sum_x2 / denominator_count - mean_x * mean_x, 0.0)
        variance_y = np.maximum(sum_y2 / denominator_count - mean_y * mean_y, 0.0)
        ncc = covariance / np.sqrt(np.maximum(variance_x * variance_y, 1e-6))
        ncc[count < max(4, int(np.ceil(0.5 * size)))] = 0.0
        outputs.append(np.clip(ncc, -1.0, 1.0).astype(np.float32))
    return outputs


def _prefix_reference(tvt_input, gr_normalised, gr_valid):
    valid = np.isfinite(tvt_input) & np.asarray(gr_valid, bool)
    if valid.sum() < 32:
        return None
    bins = np.round(np.asarray(tvt_input[valid], float) * 2.0) / 2.0
    curve = (
        pd.DataFrame({"tvt": bins, "gr": np.asarray(gr_normalised[valid], float)})
        .groupby("tvt", as_index=False)["gr"].median().sort_values("tvt")
    )
    if len(curve) < 24 or float(curve["tvt"].max() - curve["tvt"].min()) < 16.0:
        return None
    return curve["tvt"].to_numpy(float), curve["gr"].to_numpy(float)


def _geometry(md, x, y, z):
    edge = 2 if len(md) >= 3 else 1
    dz = np.gradient(z, md, edge_order=edge)
    d2z = np.gradient(dz, md, edge_order=edge)
    dx = np.gradient(x, md, edge_order=edge)
    dy = np.gradient(y, md, edge_order=edge)
    azimuth = np.unwrap(np.arctan2(dy, dx))
    curvature = np.abs(np.gradient(azimuth, md, edge_order=edge)) * np.sqrt(dx * dx + dy * dy)
    return (
        np.clip(dz, -1.0, 1.0).astype(np.float32),
        np.clip(d2z * 100.0, -5.0, 5.0).astype(np.float32),
        np.clip(curvature * 1000.0, 0.0, 5.0).astype(np.float32),
        np.sin(azimuth).astype(np.float32),
        np.cos(azimuth).astype(np.float32),
    )


def _panel_path(well, tag):
    return CACHE_DIR / f"{well}__{tag}.npz"


def _build_panel(task):
    well = str(task["well"])
    tag = str(task["tag"])
    output_path = _panel_path(well, tag)
    if output_path.exists():
        return {"well": well, "tag": tag, "status": "cached", "path": str(output_path)}
    try:
        horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
        horizontal = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        required_hw = {"MD", "X", "Y", "Z", "GR", "TVT"}
        if not required_hw.issubset(horizontal.columns):
            raise RuntimeError(f"horizontal missing {sorted(required_hw - set(horizontal.columns))}")

        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(float)
        z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
        gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
        tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
        cut = int(task["cut_position"])
        cut = int(np.clip(cut, 32, len(horizontal) - 32))
        tvt_input = np.full(len(horizontal), np.nan, float)
        tvt_input[:cut] = tvt[:cut]
        anchor_md = float(md[cut - 1])
        anchor_s = float(tvt[cut - 1] + z[cut - 1])

        span = float(md[-1] - md[0])
        n_columns = int(np.ceil(span / TARGET_MD_STEP_FT)) + 1
        if n_columns > MAX_COLUMNS:
            raise RuntimeError(f"requires {n_columns} columns")
        sample_md = np.linspace(float(md[0]), float(md[-1]), n_columns)
        sample_x = np.interp(sample_md, md, x)
        sample_y = np.interp(sample_md, md, y)
        sample_z = np.interp(sample_md, md, z)
        sample_tvt = np.interp(sample_md, md, tvt)
        true_s = sample_tvt + sample_z
        center_s = np.full(n_columns, anchor_s, dtype=np.float64)
        target_offset = true_s - center_s
        known_columns = sample_md <= anchor_md + 1e-6

        gr_valid_rows = np.isfinite(gr_raw)
        gr_norm_rows = _normalise(_fill(gr_raw), gr_valid_rows)
        sample_gr = np.interp(sample_md, md, gr_norm_rows).astype(np.float32)
        sample_gr_valid = np.interp(sample_md, md, gr_valid_rows.astype(float)) >= 0.5

        candidate_s = np.broadcast_to(
            anchor_s + OFFSETS[:, None], (len(OFFSETS), n_columns)
        ).astype(np.float32)
        axis_invariance = float(np.max(np.abs(candidate_s - candidate_s[:, :1])))
        if axis_invariance > 1e-6:
            raise RuntimeError(f"absolute-S row axis moved by {axis_invariance} ft")
        candidate_tvt = candidate_s - sample_z[None, :]
        typewell = typewell[np.isfinite(typewell["TVT"])].sort_values("TVT")
        typewell = typewell.groupby("TVT", as_index=False)["GR"].mean()
        tw_tvt = typewell["TVT"].to_numpy(float)
        tw_gr_raw = typewell["GR"].to_numpy(float)
        tw_gr = _normalise(_fill(tw_gr_raw), np.isfinite(tw_gr_raw))
        tw_valid = (candidate_tvt >= tw_tvt[0]) & (candidate_tvt <= tw_tvt[-1])
        tw_grid = np.interp(candidate_tvt.ravel(), tw_tvt, tw_gr).reshape(candidate_tvt.shape).astype(np.float32)
        tw_grid[~tw_valid] = 0.0
        horizontal_grid = np.broadcast_to(sample_gr[None], candidate_tvt.shape).astype(np.float32)
        difference = tw_grid - horizontal_grid
        match_valid = tw_valid & sample_gr_valid[None, :]
        local_ncc = _masked_local_ncc(
            tw_grid, horizontal_grid, match_valid, NCC_WINDOW_COLUMNS
        )

        known_sdf = np.zeros_like(difference, np.float32)
        known_sdf[:, known_columns] = np.clip(
            (true_s[None, known_columns] - candidate_s[:, known_columns]) / SDF_SCALE_FT,
            -3.0, 3.0,
        ).astype(np.float32)
        known_mask = np.broadcast_to(known_columns[None], difference.shape).astype(np.float32)
        offset_grid = np.broadcast_to((OFFSETS[:, None] / STATE_HALF_WIDTH_FT), difference.shape).astype(np.float32)
        md_relative = 2.0 * (sample_md - sample_md[0]) / max(span, 1.0) - 1.0
        dz, d2z, curvature, az_sin, az_cos = _geometry(sample_md, sample_x, sample_y, sample_z)

        prefix = _prefix_reference(tvt_input, gr_norm_rows, gr_valid_rows)
        if prefix is None:
            self_grid = np.zeros_like(difference, np.float32)
            self_valid = np.zeros_like(tw_valid, bool)
        else:
            prefix_tvt, prefix_gr = prefix
            self_valid = (candidate_tvt >= prefix_tvt[0]) & (candidate_tvt <= prefix_tvt[-1])
            self_grid = np.interp(candidate_tvt.ravel(), prefix_tvt, prefix_gr).reshape(candidate_tvt.shape).astype(np.float32)
            self_grid[~self_valid] = 0.0
        self_difference = self_grid - horizontal_grid
        self_coverage = np.broadcast_to(self_valid.mean(axis=0)[None], difference.shape).astype(np.float32)

        def broadcast(values):
            return np.broadcast_to(np.asarray(values, np.float32)[None], difference.shape).astype(np.float32)

        panel = np.stack(
            [
                tw_grid, horizontal_grid, difference, np.abs(difference), *local_ncc,
                tw_valid.astype(np.float32), broadcast(sample_gr_valid), known_sdf,
                known_mask, offset_grid, broadcast(md_relative), broadcast(dz),
                broadcast(d2z), broadcast(curvature), broadcast(az_sin),
                broadcast(az_cos), self_grid, self_difference,
                np.abs(self_difference), self_valid.astype(np.float32), self_coverage,
            ], axis=0,
        ).astype(np.float16)
        if panel.shape[0] != len(CHANNELS):
            raise RuntimeError(f"channel mismatch {panel.shape}")

        padded = np.zeros((len(CHANNELS), len(OFFSETS), MAX_COLUMNS), np.float16)
        padded[:, :, :n_columns] = panel
        padded_target = np.zeros(MAX_COLUMNS, np.float32)
        padded_target[:n_columns] = target_offset.astype(np.float32)
        padded_center = np.zeros(MAX_COLUMNS, np.float32)
        padded_center[:n_columns] = center_s.astype(np.float32)
        column_valid = np.zeros(MAX_COLUMNS, np.uint8)
        column_valid[:n_columns] = 1
        padded_known = np.zeros(MAX_COLUMNS, np.uint8)
        padded_known[:n_columns] = known_columns.astype(np.uint8)

        record = {
            "panel": padded,
            "target_offset": padded_target,
            "center_s": padded_center,
            "column_valid": column_valid,
            "known_columns": padded_known,
            "n_columns": np.asarray([n_columns], np.int32),
            "sample_md": sample_md.astype(np.float32),
            "sample_z": sample_z.astype(np.float32),
            "coverage": np.asarray([np.mean(np.abs(target_offset) <= STATE_HALF_WIDTH_FT)], np.float32),
        }
        if bool(task["validation"]):
            pool_well = POOL_GROUPS[well].sort_values("row_idx")
            eval_rows = np.clip(pool_well["row_idx"].to_numpy(int), 0, len(horizontal) - 1)
            record.update(
                {
                    "eval_md": md[eval_rows].astype(np.float32),
                    "eval_tvt": pool_well["tvt_true"].to_numpy(np.float32),
                    "eval_strong": pool_well["strong"].to_numpy(np.float32),
                    "eval_cp12": pool_well["cp12"].to_numpy(np.float32),
                    "eval_center": (anchor_s - z[eval_rows]).astype(np.float32),
                }
            )
        np.savez_compressed(output_path, **record)
        return {
            "well": well, "tag": tag, "status": "built", "path": str(output_path),
            "coverage": float(record["coverage"][0]), "columns": n_columns,
            "axis_invariance_ft": axis_invariance,
        }
    except Exception as exc:
        return {"well": well, "tag": tag, "status": "failed", "reason": repr(exc)}


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
            )


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        groups = min(8, outputs)
        while outputs % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs), nn.SiLU(),
        )
    def forward(self, values):
        return self.net(values)


class RankSDFUNet(nn.Module):
    def __init__(self, inputs, base=24):
        super().__init__()
        self.e1 = ConvBlock(inputs, base)
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
        d4 = self.d4(torch.cat([self.up(mid, e4), e4], 1))
        d3 = self.d3(torch.cat([self.up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2, e1), e1], 1))
        return self.sdf_head(d1).squeeze(1), self.rank_head(d1).squeeze(1)


def _loss(model, panel, target, center, columns, offsets):
    sdf, logits = model(panel)
    label_columns = columns & (torch.abs(target) <= STATE_HALF_WIDTH_FT)
    state_valid = (panel[:, TYPEWELL_VALID_CHANNEL] > 0.5) & label_columns[:, None]
    target_sdf = torch.clamp((target[:, None] - offsets[None, :, None]) / SDF_SCALE_FT, -3.0, 3.0)
    sdf_loss = F.smooth_l1_loss(sdf[state_valid], target_sdf[state_valid])
    masked = logits.masked_fill(~state_valid, -30.0)
    soft = torch.exp(-0.5 * ((offsets[None, :, None] - target[:, None]) / SOFT_LABEL_SIGMA_FT) ** 2) * state_valid.float()
    soft = soft / torch.clamp(soft.sum(1, keepdim=True), min=1e-8)
    rank_loss = (-(soft * F.log_softmax(masked, 1)).sum(1))[label_columns].mean()
    probability = torch.softmax(masked, 1)
    expected = (probability * offsets[None, :, None]).sum(1)
    soft_loss = F.smooth_l1_loss(expected[label_columns] / 10.0, target[label_columns] / 10.0)
    predicted_s = center + expected
    d2 = predicted_s[:, 2:] - 2 * predicted_s[:, 1:-1] + predicted_s[:, :-2]
    triple = label_columns[:, 2:] & label_columns[:, 1:-1] & label_columns[:, :-2]
    piecewise = F.smooth_l1_loss(d2[triple] / 4.0, torch.zeros_like(d2[triple]))
    total = LOSS_SDF * sdf_loss + LOSS_RANK * rank_loss + LOSS_SOFTARGMIN * soft_loss + LOSS_PIECEWISE * piecewise
    return total


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(prediction, float) - np.asarray(truth, float)) ** 2)))


def _cosine(truth, first, second):
    a = np.asarray(first, float) - np.asarray(truth, float)
    b = np.asarray(second, float) - np.asarray(truth, float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denominator, 1e-12))


def _beam_decode(logits, sdf, valid, known, target_offsets):
    """Decode a fixed-S path while carrying the previous slope in the beam."""
    logits = np.asarray(logits, np.float64)
    sdf = np.asarray(sdf, np.float64)
    valid = np.asarray(valid, bool)
    known = np.asarray(known, bool)
    target_offsets = np.asarray(target_offsets, float)
    height, width = logits.shape
    masked_logits = np.where(valid, logits, -30.0)
    maxima = np.max(masked_logits, axis=0, keepdims=True)
    normalizer = maxima + np.log(
        np.maximum(np.exp(masked_logits - maxima).sum(axis=0, keepdims=True), 1e-12)
    )
    emission = -(masked_logits - normalizer) + DP_SDF_WEIGHT * np.abs(sdf)
    emission[~valid] = np.inf

    known_indices = np.flatnonzero(known)
    if not len(known_indices):
        raise RuntimeError("beam decoder has no known anchor column")
    anchor_column = int(known_indices[-1])
    path_states = np.full(width, -1, np.int32)
    known_states = np.clip(
        np.rint((target_offsets - OFFSETS[0]) / STATE_STEP_FT).astype(int),
        0,
        height - 1,
    )
    path_states[known] = known_states[known]
    start_state = int(known_states[anchor_column])
    start_slope = 0
    if anchor_column > 0 and known[anchor_column - 1]:
        start_slope = int(np.clip(
            start_state - known_states[anchor_column - 1],
            -DP_MAX_STEP_STATES,
            DP_MAX_STEP_STATES,
        ))

    beam_states = np.asarray([start_state], np.int32)
    beam_slopes = np.asarray([start_slope], np.int32)
    beam_costs = np.asarray([0.0], np.float64)
    history = []
    deltas = np.arange(-DP_MAX_STEP_STATES, DP_MAX_STEP_STATES + 1, dtype=np.int32)

    for column in range(anchor_column + 1, width):
        next_states = beam_states[:, None] + deltas[None, :]
        parent = np.broadcast_to(np.arange(len(beam_states))[:, None], next_states.shape)
        next_slopes = np.broadcast_to(deltas[None, :], next_states.shape)
        feasible = (next_states >= 0) & (next_states < height)
        clipped_states = np.clip(next_states, 0, height - 1)
        feasible &= valid[clipped_states, column]
        costs = (
            beam_costs[:, None]
            + emission[clipped_states, column]
            + DP_SLOPE_CHANGE_PENALTY
            * (next_slopes - beam_slopes[:, None]).astype(float) ** 2
        )
        flat_valid = np.flatnonzero(feasible.ravel() & np.isfinite(costs.ravel()))
        if not len(flat_valid):
            # Keep the diagnostic defined even if the typewell mask briefly
            # disconnects the path. The pointwise state is used for this column.
            candidates = np.flatnonzero(valid[:, column])
            fallback = start_state if not len(candidates) else int(
                candidates[np.argmin(emission[candidates, column])]
            )
            parent_index = int(np.argmin(beam_costs))
            previous_state = int(beam_states[parent_index])
            beam_states = np.asarray([fallback], np.int32)
            beam_slopes = np.asarray([
                int(np.clip(fallback - previous_state, -DP_MAX_STEP_STATES, DP_MAX_STEP_STATES))
            ], np.int32)
            beam_costs = np.asarray([beam_costs[parent_index] + emission[fallback, column]])
            history.append((np.asarray([parent_index]), beam_states.copy()))
            continue

        order = flat_valid[np.argsort(costs.ravel()[flat_valid])]
        selected = []
        seen = set()
        flat_states = clipped_states.ravel()
        flat_slopes = next_slopes.ravel()
        for index in order:
            key = (int(flat_states[index]), int(flat_slopes[index]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(int(index))
            if len(selected) >= DP_BEAM_SIZE:
                break
        selected = np.asarray(selected, np.int64)
        selected_parent = parent.ravel()[selected].astype(np.int32)
        beam_states = flat_states[selected].astype(np.int32)
        beam_slopes = flat_slopes[selected].astype(np.int32)
        beam_costs = costs.ravel()[selected]
        history.append((selected_parent, beam_states.copy()))

    if history:
        beam_index = int(np.argmin(beam_costs))
        for relative_column in range(len(history) - 1, -1, -1):
            parents, states = history[relative_column]
            column = anchor_column + 1 + relative_column
            path_states[column] = states[beam_index]
            beam_index = int(parents[beam_index])
    path_states[anchor_column] = start_state
    missing = path_states < 0
    if missing.any():
        pointwise = np.argmin(emission, axis=0)
        path_states[missing] = pointwise[missing]
    return OFFSETS[path_states]


@torch.no_grad()
def _evaluate(model, paths, device):
    model.eval()
    parts = []
    offset_tensor = torch.from_numpy(OFFSETS).to(device)
    for well, path in tqdm(paths.items(), desc="validate RankSDF", leave=False):
        with np.load(path, allow_pickle=False) as data:
            panel = data["panel"].astype(np.float32)
            target = data["target_offset"].astype(float)
            center_s = data["center_s"].astype(float)
            column_valid = data["column_valid"].astype(bool)
            known = data["known_columns"].astype(bool)
            n_columns = int(data["n_columns"][0])
            sample_md = data["sample_md"].astype(float)
            sample_z = data["sample_z"].astype(float)
            eval_md = data["eval_md"].astype(float)
            truth = data["eval_tvt"].astype(float)
            strong = data["eval_strong"].astype(float)
            cp12 = data["eval_cp12"].astype(float)
            centre = data["eval_center"].astype(float)
        tensor = torch.from_numpy(panel)[None].to(device)
        sdf, logits = model(tensor)
        valid = (tensor[:, TYPEWELL_VALID_CHANNEL] > 0.5) & torch.from_numpy(column_valid)[None, None].to(device)
        probability = torch.softmax(logits.masked_fill(~valid, -30.0), 1)
        rank_offset = (probability * offset_tensor[None, :, None]).sum(1)[0].cpu().numpy()
        sdf_index = torch.argmin(torch.abs(sdf).masked_fill(~valid, 1e6), 1)[0].cpu().numpy()
        sdf_offset = OFFSETS[sdf_index]
        dp_offset = _beam_decode(
            logits[0, :, :n_columns].float().cpu().numpy(),
            sdf[0, :, :n_columns].float().cpu().numpy(),
            valid[0, :, :n_columns].cpu().numpy(),
            known[:n_columns],
            target[:n_columns],
        )
        rank_offset[known] = target[known]
        sdf_offset[known] = target[known]
        dp_offset[known[:n_columns]] = target[:n_columns][known[:n_columns]]
        rank_curve = center_s[:n_columns] + rank_offset[:n_columns] - sample_z
        sdf_curve = center_s[:n_columns] + sdf_offset[:n_columns] - sample_z
        dp_curve = center_s[:n_columns] + dp_offset - sample_z
        parts.append(
            pd.DataFrame(
                {
                    "well": well, "truth": truth, "rank": np.interp(eval_md, sample_md, rank_curve),
                    "sdf": np.interp(eval_md, sample_md, sdf_curve), "centre": centre,
                    "dp": np.interp(eval_md, sample_md, dp_curve),
                    "strong": strong, "cp12": cp12,
                }
            )
        )
    rows = pd.concat(parts, ignore_index=True)
    truth = rows["truth"].to_numpy(float)
    rank = rows["rank"].to_numpy(float)
    strong = rows["strong"].to_numpy(float)
    bywell = []
    for well, group in rows.groupby("well", sort=False):
        bywell.append(
            {
                "well": well, "rows": len(group),
                "rank_rmse": _rmse(group["truth"], group["rank"]),
                "dp_rmse": _rmse(group["truth"], group["dp"]),
                "strong_rmse": _rmse(group["truth"], group["strong"]),
            }
        )
    bywell = pd.DataFrame(bywell)
    bywell["gain"] = bywell["strong_rmse"] - bywell["rank_rmse"]
    bywell["dp_gain"] = bywell["strong_rmse"] - bywell["dp_rmse"]
    metrics = {
        "rank_rmse": _rmse(truth, rank), "sdf_rmse": _rmse(truth, rows["sdf"]),
        "dp_rmse": _rmse(truth, rows["dp"]),
        "centre_rmse": _rmse(truth, rows["centre"]), "strong_rmse": _rmse(truth, strong),
        "cp12_rmse": _rmse(truth, rows["cp12"]),
        "gain_vs_strong": _rmse(truth, strong) - _rmse(truth, rank),
        "dp_gain_vs_strong": _rmse(truth, strong) - _rmse(truth, rows["dp"]),
        "error_cosine_vs_strong": _cosine(truth, rank, strong),
        "dp_error_cosine_vs_strong": _cosine(truth, rows["dp"], strong),
        "positive_well_fraction": float((bywell["gain"] > 0).mean()),
        "dp_positive_well_fraction": float((bywell["dp_gain"] > 0).mean()),
        "median_well_gain": float(bywell["gain"].median()), "rows": len(rows),
        "dp_median_well_gain": float(bywell["dp_gain"].median()),
    }
    return metrics, rows, bywell


for path in (MANIFEST_PATH, AXIS_AUDIT, BANK_DIR / "well_metadata.csv"):
    if not path.exists():
        raise FileNotFoundError(path)
manifest_spec = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(OOF_ROOT).free / 2**30
print("disk free before panel build (GB):", free_gb)
if free_gb < MIN_FREE_GB_BEFORE_BUILD:
    raise RuntimeError(
        f"Need at least {MIN_FREE_GB_BEFORE_BUILD:.1f} GB free before building "
        f"the RankSDF pseudo-PS cache; found {free_gb:.2f} GB"
    )

meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
fold_map = meta.set_index("well")["fold"].astype(int).to_dict()
manifests = [pd.read_csv(BANK_DIR / "cut_manifest.csv", dtype={"well": str})]
extension_manifest = EXTENSION_DIR / "new_cut_manifest.csv"
if extension_manifest.exists():
    manifests.append(pd.read_csv(extension_manifest, dtype={"well": str}))
cut_manifest = pd.concat(manifests, ignore_index=True).drop_duplicates(["well", "tag"], keep="last")
cut_manifest = cut_manifest[cut_manifest["tag"].isin(TAGS)].copy()
cut_manifest["fold"] = cut_manifest["well"].map(fold_map).astype(int)
if "actual_frac" not in cut_manifest:
    cut_manifest["actual_frac"] = cut_manifest["cut_position"] / cut_manifest["n_rows"]

pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
required_pool = {"id", "tvt_true", "strong", "cp12"}
if not required_pool.issubset(header):
    raise RuntimeError(f"pool missing {sorted(required_pool - set(header))}")
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong", "cp12") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
POOL_GROUPS = {well: frame.copy() for well, frame in pool.groupby("well", sort=False)}

train_manifest = cut_manifest[
    (cut_manifest["fold"] != VALID_FOLD)
    & cut_manifest["actual_frac"].between(TRAIN_KNOWN_FRAC_MIN, TRAIN_KNOWN_FRAC_MAX)
].copy()
valid_manifest = cut_manifest[(cut_manifest["fold"] == VALID_FOLD) & (cut_manifest["tag"] == "official")].copy()
tasks = []
for row in train_manifest.itertuples(index=False):
    tasks.append({"well": str(row.well), "tag": str(row.tag), "cut_position": int(row.cut_position), "validation": False})
for row in valid_manifest.itertuples(index=False):
    tasks.append({"well": str(row.well), "tag": str(row.tag), "cut_position": int(row.cut_position), "validation": True})

print(
    {
        "experiment": "ranksdf_cp_v2_absolute_s_onefold",
        "valid_fold": VALID_FOLD, "train_states": len(train_manifest),
        "train_wells": train_manifest["well"].nunique(), "valid_wells": len(valid_manifest),
        "seeds": SEEDS, "epochs": EPOCHS,
        "axis": "fixed absolute S at last-known anchor",
        "half_width_ft": STATE_HALF_WIDTH_FT, "state_step_ft": STATE_STEP_FT,
    }
)
print("TRAIN CUTS")
print(train_manifest.groupby("tag")["actual_frac"].describe()[["count", "mean", "min", "50%", "max"]].to_string())

build_started = time.time()
results = joblib.Parallel(n_jobs=N_JOBS_BUILD, prefer="threads")(
    joblib.delayed(_build_panel)(task) for task in tqdm(tasks, desc="build RankSDF pseudo-PS panels")
)
build_report = pd.DataFrame(results)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
failures = build_report[build_report["status"] == "failed"]
if len(failures):
    display(failures.head(30))
    raise RuntimeError(f"panel failures: {len(failures)}")
print("panel cache:", {"states": len(build_report), "elapsed_min": (time.time() - build_started) / 60.0})

train_paths = [_panel_path(str(row.well), str(row.tag)) for row in train_manifest.itertuples(index=False)]
valid_paths = {str(row.well): _panel_path(str(row.well), "official") for row in valid_manifest.itertuples(index=False)}

# The exact label path must reconstruct TVT before any network is involved.
teacher_truth = []
teacher_prediction = []
for path in valid_paths.values():
    with np.load(path, allow_pickle=False) as data:
        n_columns = int(data["n_columns"][0])
        teacher_curve = (
            data["center_s"][:n_columns].astype(float)
            + data["target_offset"][:n_columns].astype(float)
            - data["sample_z"].astype(float)
        )
        teacher_truth.append(data["eval_tvt"].astype(float))
        teacher_prediction.append(
            np.interp(data["eval_md"].astype(float), data["sample_md"].astype(float), teacher_curve)
        )
teacher_truth = np.concatenate(teacher_truth)
teacher_prediction = np.concatenate(teacher_prediction)
teacher_decode_rmse = _rmse(teacher_truth, teacher_prediction)
axis_invariance_max = float(build_report.get("axis_invariance_ft", pd.Series([np.nan])).max())
print(
    "PANEL GUARDS",
    {
        "teacher_decode_rmse": teacher_decode_rmse,
        "axis_invariance_max_ft": axis_invariance_max,
        "mean_label_coverage": float(build_report["coverage"].mean()),
        "min_label_coverage": float(build_report["coverage"].min()),
    },
)
if teacher_decode_rmse > 0.10 or axis_invariance_max > 1e-6:
    raise RuntimeError("fixed absolute-S panel guard failed")

loader = DataLoader(
    PanelDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
offset_tensor = torch.from_numpy(OFFSETS).to(DEVICE)
seed_reports = []
seed_predictions = []
all_history = []
training_started = time.time()

for seed in SEEDS:
    _seed_everything(seed)
    model = RankSDFUNet(len(CHANNELS), BASE_CHANNELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_rmse = np.inf
    best_path = MODEL_DIR / f"ranksdf_cp_v2_absolute_s_fold{VALID_FOLD}_seed{seed}.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for panel, target, center, columns in tqdm(loader, desc=f"seed {seed} epoch {epoch}", leave=False):
            panel = panel.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            center = center.to(DEVICE, non_blocking=True)
            columns = columns.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = _loss(model, panel, target, center, columns, offset_tensor)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history = {"seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            metrics, _, _ = _evaluate(model, valid_paths, DEVICE)
            history.update(metrics)
            print("epoch", history, flush=True)
            if metrics["dp_rmse"] < best_rmse:
                best_rmse = metrics["dp_rmse"]
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch, "seed": seed,
                     "metrics": metrics, "channels": CHANNELS, "manifest": manifest_spec},
                    best_path,
                )
        all_history.append(history)

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, predictions, bywell = _evaluate(model, valid_paths, DEVICE)
    metrics.update({"seed": seed, "selected_epoch": int(checkpoint["epoch"])})
    seed_reports.append(metrics)
    predictions = predictions.rename(
        columns={"rank": f"rank_{seed}", "dp": f"dp_{seed}"}
    )
    seed_predictions.append(
        predictions[["well", "truth", "strong", "cp12", f"rank_{seed}", f"dp_{seed}"]]
    )
    bywell.to_csv(OUT_DIR / f"bywell_seed{seed}.csv", index=False)
    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

merged = seed_predictions[0]
for seed, frame in zip(SEEDS[1:], seed_predictions[1:]):
    if len(frame) != len(merged):
        raise RuntimeError(f"seed {seed} prediction row mismatch")
    if not np.array_equal(frame["well"].to_numpy(str), merged["well"].to_numpy(str)):
        raise RuntimeError(f"seed {seed} well ordering mismatch")
    if not np.allclose(frame["truth"], merged["truth"], atol=1e-6):
        raise RuntimeError(f"seed {seed} truth ordering mismatch")
    merged[f"rank_{seed}"] = frame[f"rank_{seed}"].to_numpy(float)
    merged[f"dp_{seed}"] = frame[f"dp_{seed}"].to_numpy(float)
rank_columns = [f"rank_{seed}" for seed in SEEDS]
dp_columns = [f"dp_{seed}" for seed in SEEDS]
merged["ensemble"] = merged[rank_columns].mean(axis=1)
merged["dp_ensemble"] = merged[dp_columns].mean(axis=1)
truth = merged["truth"].to_numpy(float)
strong = merged["strong"].to_numpy(float)
ensemble = merged["ensemble"].to_numpy(float)
dp_ensemble = merged["dp_ensemble"].to_numpy(float)

bywell_rows = []
for well, group in merged.groupby("well", sort=False):
    bywell_rows.append(
        {
            "well": well, "rows": len(group),
            "strong_rmse": _rmse(group["truth"], group["strong"]),
            "ensemble_rmse": _rmse(group["truth"], group["ensemble"]),
            "dp_ensemble_rmse": _rmse(group["truth"], group["dp_ensemble"]),
        }
    )
ensemble_bywell = pd.DataFrame(bywell_rows)
ensemble_bywell["gain"] = ensemble_bywell["strong_rmse"] - ensemble_bywell["ensemble_rmse"]
ensemble_bywell["dp_gain"] = (
    ensemble_bywell["strong_rmse"] - ensemble_bywell["dp_ensemble_rmse"]
)
ensemble_report = {
    "rank_rmse": _rmse(truth, ensemble), "strong_rmse": _rmse(truth, strong),
    "cp12_rmse": _rmse(truth, merged["cp12"]),
    "gain_vs_strong": _rmse(truth, strong) - _rmse(truth, ensemble),
    "dp_rmse": _rmse(truth, dp_ensemble),
    "dp_gain_vs_strong": _rmse(truth, strong) - _rmse(truth, dp_ensemble),
    "error_cosine_vs_strong": _cosine(truth, ensemble, strong),
    "dp_error_cosine_vs_strong": _cosine(truth, dp_ensemble, strong),
    "positive_well_fraction": float((ensemble_bywell["gain"] > 0).mean()),
    "dp_positive_well_fraction": float((ensemble_bywell["dp_gain"] > 0).mean()),
    "median_well_gain": float(ensemble_bywell["gain"].median()),
    "dp_median_well_gain": float(ensemble_bywell["dp_gain"].median()),
    "rows": len(merged),
}
seed_report = pd.DataFrame(seed_reports)
strict_pass = bool(
    (seed_report["dp_gain_vs_strong"] > 0).all()
    and ensemble_report["dp_gain_vs_strong"] >= PASS_GAIN_VS_STRONG
    and ensemble_report["dp_positive_well_fraction"] >= PASS_POSITIVE_WELL_FRACTION
    and ensemble_report["dp_median_well_gain"] >= PASS_MEDIAN_WELL_GAIN
    and ensemble_report["dp_error_cosine_vs_strong"] <= PASS_ERROR_COSINE
)
verdict = {
    "experiment": "ranksdf_cp_v2_absolute_s_onefold",
    "valid_fold": VALID_FOLD, "train_states": len(train_manifest),
    "train_wells": int(train_manifest["well"].nunique()), "valid_wells": len(valid_manifest),
    "seeds": list(SEEDS),
    "seed_positive_count": int((seed_report["dp_gain_vs_strong"] > 0).sum()),
    "axis": {
        "definition": "fixed S_anchor + offset",
        "half_width_ft": STATE_HALF_WIDTH_FT,
        "step_ft": STATE_STEP_FT,
        "rows": len(OFFSETS),
    },
    "decoder": {
        "beam_size": DP_BEAM_SIZE,
        "max_step_states": DP_MAX_STEP_STATES,
        "slope_change_penalty": DP_SLOPE_CHANGE_PENALTY,
        "sdf_weight": DP_SDF_WEIGHT,
    },
    "panel_guards": {
        "teacher_decode_rmse": teacher_decode_rmse,
        "axis_invariance_max_ft": axis_invariance_max,
        "mean_label_coverage": float(build_report["coverage"].mean()),
        "min_label_coverage": float(build_report["coverage"].min()),
    },
    "ensemble": ensemble_report,
    "gates": {
        "all_seeds_positive": bool((seed_report["dp_gain_vs_strong"] > 0).all()),
        "ensemble_gain_vs_strong_min": PASS_GAIN_VS_STRONG,
        "positive_well_fraction_min": PASS_POSITIVE_WELL_FRACTION,
        "median_well_gain_min": PASS_MEDIAN_WELL_GAIN,
        "error_cosine_max": PASS_ERROR_COSINE,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> add synthetic pretraining and hard negatives"
        if strict_pass else "FAIL -> stop and diagnose the real-to-overfit generalisation gap"
    ),
    "panel_build_minutes": (training_started - build_started) / 60.0,
    "training_hours": (time.time() - training_started) / 3600.0,
}

pd.DataFrame(all_history).to_csv(OUT_DIR / "training_history.csv", index=False)
seed_report.to_csv(OUT_DIR / "seed_scores.csv", index=False)
merged.to_pickle(OUT_DIR / "fold1_predictions.pkl")
ensemble_bywell.to_csv(OUT_DIR / "ensemble_bywell.csv", index=False)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nSEEDS")
print(seed_report.to_string(index=False))
print("\nENSEMBLE")
print(json.dumps(ensemble_report, indent=2))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
