"""Strong-centred complete-well set-posterior scorer, nested Fold-2 gateway.

The existing 235-path experiment encoded and scored every candidate
independently.  This experiment keeps the fold-clean strong-centred path bank,
but compares all candidate modes jointly with a set transformer and decodes a
multiplicity-corrected posterior mean.

Protocol
--------
* Candidate paths are the already exported fold-clean OOF matrices.
* Folds 3/4 train the inner model; Fold 5 selects epoch and temperature.
* A fresh model trains on Folds 3/4/5 for the frozen epoch count.
* Fold 2 is evaluated exactly once. Fold 1 is never read.
* One training sample is one complete physical well.
* Strong defines the candidate coordinate system but is not an input channel.
* No neighbour, CP12, Gold, or test-time search is used.
"""

from __future__ import annotations

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
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
EXPERIMENT = "strong_centered_setposterior_fold2_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "panel_cache"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "research_handoff_bundle_v1" / "metadata" / "fold_clean_well_map.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
CANDIDATE_DIR = OOF_ROOT / "research_handoff_bundle_v1" / "candidate_paths"

INNER_TRAIN_FOLDS = (3, 4)
INNER_VALID_FOLD = 5
FINAL_TRAIN_FOLDS = (3, 4, 5)
FINAL_VALID_FOLD = 2
SEQUENCE_LENGTH = 256
MAX_EPOCHS = 16
MIN_EPOCHS = 4
PATIENCE = 4
LEARNING_RATE = 2.0e-4
MIN_LEARNING_RATE = 2.0e-5
WEIGHT_DECAY = 3.0e-5
GRAD_ACCUM_WELLS = 4
GRADIENT_CLIP = 4.0
TARGET_TEMPERATURE_FT = 1.5
TEMPERATURES = (0.50, 0.75, 1.00, 1.50, 2.00)
MULTIPLICITY_STRENGTH = 1.0
LOSS_LISTWISE = 1.0
LOSS_POSTERIOR_PATH = 0.80
LOSS_EXPECTED_MSE = 0.15
LOSS_ALIAS_MARGIN = 0.10
ALIAS_MARGIN = 0.75
SEED = 20260720

ORACLE_GATE_RMSE = 4.75
FOLD2_GAIN_GATE = 0.30
FOLD2_RMSE_GATE = 8.30
FAR_GAIN_GATE = 0.50
CONTROL_DEGRADATION_GATE = 0.20

CHANNELS = (
    "observed_gr",
    "candidate_gr",
    "difference",
    "absolute_difference",
    "candidate_valid",
    "correction",
    "smooth64_difference",
    "smooth256_difference",
    "progress",
)
GR_SEQUENCE_CHANNELS = (0, 1, 2, 3, 6, 7)
CANDIDATE_ALIGNMENT_CHANNELS = (1, 2, 3, 4, 6, 7)

SUMMARY_NAMES = (
    "match_mae",
    "match_rmse",
    "match_bias",
    "match_ncc",
    "match_coverage",
    "correction_end",
    "correction_mean",
    "correction_std",
    "correction_max_abs",
    "correction_tv",
    "correction_curvature",
    "log_multiplicity",
    "is_zero",
    "family_zero",
    "family_linear_endpoint_quantile",
    "family_raw_shape_kmeans",
    "family_length_normalised_kmeans",
    "family_pca_mean",
    "family_pca_component",
    "family_other",
)
GR_SUMMARY_CHANNELS = (0, 1, 2, 3, 4)

if POOL_PATH is None:
    raise FileNotFoundError("candidate pool not found")
if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
for path in (DATA_ROOT / "train", CANDIDATE_DIR):
    if not path.exists():
        raise FileNotFoundError(path)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )


def _rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _fit_affine(horizontal_gr, typewell_gr):
    x = np.asarray(horizontal_gr, np.float64)
    y = np.asarray(typewell_gr, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        return 1.0, float(np.nanmedian(y) - np.nanmedian(x))
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, float(np.median(y) - np.median(x))
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(int(keep.sum()))])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - centre)) + 1e-6)
        updated = np.abs(residual - centre) <= 3.5 * scale
        if updated.sum() < 24:
            break
        keep = updated
    return float(np.clip(slope, -5.0, 5.0)), float(intercept)


def _moving_average(values, valid, size):
    size = max(3, int(size))
    weight = valid.astype(np.float32)
    count = uniform_filter1d(
        weight, size=size, axis=1, mode="constant", cval=0.0
    ) * size
    total = uniform_filter1d(
        values * weight, size=size, axis=1, mode="constant", cval=0.0
    ) * size
    output = total / np.maximum(count, 1.0)
    output[count < max(4, int(np.ceil(0.5 * size)))] = 0.0
    return output.astype(np.float32)


def _masked_alignment_summary(reference, observed, valid):
    weight = valid.astype(np.float32)
    count = np.maximum(weight.sum(axis=1), 1.0)
    difference = (reference - observed) * weight
    bias = difference.sum(axis=1) / count
    mae = np.abs(difference).sum(axis=1) / count
    rmse = np.sqrt((difference * difference).sum(axis=1) / count)
    reference_mean = (reference * weight).sum(axis=1) / count
    observed_mean = (observed * weight).sum(axis=1) / count
    reference_centre = (reference - reference_mean[:, None]) * weight
    observed_centre = (observed - observed_mean[:, None]) * weight
    numerator = (reference_centre * observed_centre).sum(axis=1)
    denominator = np.sqrt(
        (reference_centre**2).sum(axis=1)
        * (observed_centre**2).sum(axis=1)
    )
    ncc = numerator / np.maximum(denominator, 1e-6)
    coverage = weight.mean(axis=1)
    return mae, rmse, bias, ncc, coverage


def _multiplicity(corrections):
    sample = corrections[:, :: max(1, corrections.shape[1] // 32)].astype(np.float32)
    endpoint = corrections[:, -1].astype(np.float32)
    output = np.ones(len(corrections), np.float32)
    block = 64
    for left in range(0, len(corrections), block):
        right = min(left + block, len(corrections))
        distance = np.sqrt(
            np.mean((sample[left:right, None] - sample[None]) ** 2, axis=2)
        )
        endpoint_distance = np.abs(endpoint[left:right, None] - endpoint[None])
        output[left:right] = np.sum(
            (distance <= 3.0) & (endpoint_distance <= 4.0), axis=1
        )
    return np.maximum(output, 1.0)


# ---------------------------------------------------------------------------
# Common OOF rows and per-well panel cache
# ---------------------------------------------------------------------------


header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", "strong")
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
for column in ("tvt_true", "strong"):
    pool[column] = pd.to_numeric(pool[column], errors="coerce").astype(np.float32)

fold_frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
fold_frame = fold_frame.groupby("well", as_index=False)["fold"].first()
fold_map = dict(
    zip(
        fold_frame["well"].astype(str),
        pd.to_numeric(fold_frame["fold"], errors="raise").astype(int),
    )
)
pool["fold"] = pool["well"].map(fold_map)
if pool["fold"].isna().any():
    raise RuntimeError("candidate pool contains wells missing from fold map")
pool = pool.loc[pool["fold"].isin((*FINAL_TRAIN_FOLDS, FINAL_VALID_FOLD))]
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool_groups = {
    well: frame.reset_index(drop=True)
    for well, frame in pool.groupby("well", sort=True)
}


def _candidate_path(well):
    return CANDIDATE_DIR / f"{well}.npz"


def _cache_path(well):
    return CACHE_DIR / f"{well}.npz"


def _build_panel(well, frame):
    output = _cache_path(well)
    if output.exists():
        with np.load(output, allow_pickle=False) as data:
            return {
                "well": well,
                "status": "cached",
                "candidates": int(data["sequence"].shape[0]),
                "rows": int(data["eval_rows"][0]),
            }
    source_path = _candidate_path(well)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    with np.load(source_path, allow_pickle=False) as source:
        source_rows = source["row_idx"].astype(np.int64)
        correction_native = (
            source["correction_q01"].astype(np.float32)
            * float(source["quantisation_ft"][0])
        )
        candidate_names = source["candidate_names"].astype(str)
        source_family = source["source_family"].astype(str)
    frame_rows = frame["row_idx"].to_numpy(np.int64)
    if not np.array_equal(source_rows, frame_rows):
        raise RuntimeError(f"{well}: candidate and pool row order differ")
    if len(candidate_names) != len(correction_native):
        raise RuntimeError(f"{well}: candidate metadata mismatch")

    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=["MD", "GR", "TVT_input"],
    )
    typewell = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"]
    )
    for column in ("MD", "GR", "TVT_input"):
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    typewell = typewell.apply(pd.to_numeric, errors="coerce").dropna(subset=["TVT"])
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    typewell["GR"] = typewell["GR"].interpolate(limit_direction="both").fillna(0.0)

    md = horizontal["MD"].to_numpy(np.float64)
    gr_raw = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    eval_md = md[frame_rows]
    anchor = int(frame_rows.min() - 1)
    if anchor < 0:
        raise RuntimeError(f"{well}: invalid anchor row")
    anchor_md = float(md[anchor])
    eval_span = float(max(eval_md[-1] - anchor_md, 1.0))
    sequence_md = np.linspace(eval_md[0], eval_md[-1], SEQUENCE_LENGTH)
    progress = np.clip((sequence_md - anchor_md) / eval_span, 0.0, 1.0).astype(np.float32)
    strong_native = frame["strong"].to_numpy(np.float64)
    truth_native = frame["tvt_true"].to_numpy(np.float64)
    strong_sequence = np.interp(sequence_md, eval_md, strong_native)
    truth_residual_sequence = np.interp(
        sequence_md, eval_md, truth_native - strong_native
    ).astype(np.float32)

    correction_sequence = np.stack(
        [np.interp(sequence_md, eval_md, values) for values in correction_native]
    ).astype(np.float32)
    candidate_tvt = strong_sequence[None] + correction_sequence

    tw_tvt = typewell["TVT"].to_numpy(np.float64)
    tw_gr_raw = typewell["GR"].to_numpy(np.float64)
    known = np.isfinite(tvt_input) & np.isfinite(gr_raw)
    known &= (tvt_input >= tw_tvt[0]) & (tvt_input <= tw_tvt[-1])
    if known.sum() < 16:
        raise RuntimeError(f"{well}: insufficient visible-prefix GR calibration")
    known_tw = np.interp(tvt_input[known], tw_tvt, tw_gr_raw)
    slope, intercept = _fit_affine(gr_raw[known], known_tw)
    calibrated = slope * gr_raw + intercept
    observed_valid_rows = np.isfinite(calibrated)
    observed_fill = np.interp(
        md, md[observed_valid_rows], calibrated[observed_valid_rows]
    )
    observed = np.interp(sequence_md, md, observed_fill)
    observed_valid = (
        np.interp(sequence_md, md, observed_valid_rows.astype(float)) >= 0.5
    )
    location = float(np.median(known_tw))
    scale = float(1.4826 * np.median(np.abs(known_tw - location)) + 1e-3)
    observed = np.clip((observed - location) / scale, -8.0, 8.0).astype(np.float32)
    tw_gr = np.clip((tw_gr_raw - location) / scale, -8.0, 8.0).astype(np.float32)

    candidate_valid = (
        (candidate_tvt >= tw_tvt[0])
        & (candidate_tvt <= tw_tvt[-1])
        & observed_valid[None]
    )
    candidate_gr = np.interp(
        candidate_tvt.ravel(), tw_tvt, tw_gr
    ).reshape(candidate_tvt.shape).astype(np.float32)
    candidate_gr[~candidate_valid] = 0.0
    observed_grid = np.broadcast_to(observed[None], candidate_gr.shape).astype(np.float32)
    difference = (candidate_gr - observed_grid) * candidate_valid
    step = eval_span / max(SEQUENCE_LENGTH - 1, 1)
    smooth64 = _moving_average(
        difference, candidate_valid, max(3, round(64.0 / max(step, 1e-3)))
    )
    smooth256 = _moving_average(
        difference, candidate_valid, max(5, round(256.0 / max(step, 1e-3)))
    )
    sequence = np.stack(
        [
            observed_grid,
            candidate_gr,
            difference,
            np.abs(difference),
            candidate_valid.astype(np.float32),
            correction_sequence / 64.0,
            smooth64,
            smooth256,
            np.broadcast_to(progress[None], candidate_gr.shape),
        ],
        axis=1,
    ).astype(np.float16)

    mae, match_rmse, bias, ncc, coverage = _masked_alignment_summary(
        candidate_gr, observed_grid, candidate_valid
    )
    multiplicity = _multiplicity(correction_sequence)
    zero_mask = np.asarray(
        [(name == "zero") or (family == "zero") for name, family in zip(candidate_names, source_family)],
        np.float32,
    )
    family_names = (
        "zero",
        "linear_endpoint_quantile",
        "raw_shape_kmeans",
        "length_normalised_kmeans",
        "pca_mean",
        "pca_component",
    )
    summary_parts = [
        mae,
        match_rmse,
        bias,
        ncc,
        coverage,
        correction_sequence[:, -1] / 64.0,
        correction_sequence.mean(axis=1) / 64.0,
        correction_sequence.std(axis=1) / 64.0,
        np.max(np.abs(correction_sequence), axis=1) / 64.0,
        np.mean(np.abs(np.diff(correction_sequence, axis=1)), axis=1) / 8.0,
        np.mean(np.abs(np.diff(correction_sequence, n=2, axis=1)), axis=1) / 4.0,
        np.log(multiplicity),
        zero_mask,
    ]
    for family in family_names:
        summary_parts.append((source_family == family).astype(np.float32))
    summary_parts.append((~np.isin(source_family, family_names)).astype(np.float32))
    summary = np.stack(summary_parts, axis=1).astype(np.float32)
    if summary.shape[1] != len(SUMMARY_NAMES):
        raise RuntimeError(f"{well}: summary shape mismatch {summary.shape}")

    error = correction_native - (truth_native - strong_native)[None]
    candidate_mse = np.mean(error * error, axis=1).astype(np.float32)
    zero_indices = np.flatnonzero(zero_mask > 0.5)
    if len(zero_indices) != 1:
        raise RuntimeError(f"{well}: expected one zero candidate, found {len(zero_indices)}")

    np.savez_compressed(
        output,
        sequence=sequence,
        summary=summary.astype(np.float16),
        correction_sequence=correction_sequence.astype(np.float16),
        truth_residual_sequence=truth_residual_sequence,
        candidate_mse=candidate_mse,
        log_multiplicity=np.log(multiplicity).astype(np.float32),
        candidate_names=candidate_names,
        source_family=source_family,
        zero_index=np.asarray([int(zero_indices[0])], np.int16),
        eval_rows=np.asarray([len(frame)], np.int32),
        fold=np.asarray([int(frame["fold"].iloc[0])], np.int8),
        calibration=np.asarray([slope, intercept, location, scale], np.float32),
    )
    return {
        "well": well,
        "status": "built",
        "candidates": len(candidate_names),
        "rows": len(frame),
    }


started = time.time()
build_rows = []
for index, (well, frame) in enumerate(pool_groups.items(), 1):
    try:
        build_rows.append(_build_panel(well, frame))
    except Exception as exc:
        build_rows.append({"well": well, "status": "failed", "reason": repr(exc)})
    if index % 50 == 0 or index == len(pool_groups):
        print(f"build set-posterior panels {index}/{len(pool_groups)}", flush=True)
build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
failed = build_report.loc[build_report["status"].eq("failed")]
if len(failed):
    raise RuntimeError(
        f"{len(failed)} panel builds failed: " + repr(failed.head(10).to_dict(orient="records"))
    )

inner_train_wells = sorted(
    well for well, frame in pool_groups.items()
    if int(frame["fold"].iloc[0]) in INNER_TRAIN_FOLDS
)
inner_valid_wells = sorted(
    well for well, frame in pool_groups.items()
    if int(frame["fold"].iloc[0]) == INNER_VALID_FOLD
)
final_train_wells = sorted(
    well for well, frame in pool_groups.items()
    if int(frame["fold"].iloc[0]) in FINAL_TRAIN_FOLDS
)
final_valid_wells = sorted(
    well for well, frame in pool_groups.items()
    if int(frame["fold"].iloc[0]) == FINAL_VALID_FOLD
)
if (
    len(inner_train_wells) not in (308, 309, 310)
    or len(inner_valid_wells) not in (154, 155)
    or len(final_train_wells) != 463
    or len(final_valid_wells) != 155
):
    raise RuntimeError(
        "unexpected nested fold sizes: "
        + repr(
            (
                len(inner_train_wells), len(inner_valid_wells),
                len(final_train_wells), len(final_valid_wells),
            )
        )
    )


def _load_panel(well):
    with np.load(_cache_path(well), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _oracle_metrics(wells):
    rows = 0
    oracle_sse = 0.0
    base_sse = 0.0
    bywell = []
    for well in wells:
        panel = _load_panel(well)
        candidate_mse = panel["candidate_mse"].astype(np.float64)
        n_rows = int(panel["eval_rows"][0])
        best = float(candidate_mse.min())
        zero = float(candidate_mse[int(panel["zero_index"][0])])
        rows += n_rows
        oracle_sse += n_rows * best
        base_sse += n_rows * zero
        bywell.append(
            {
                "well": well,
                "rows": n_rows,
                "base_rmse": math.sqrt(zero),
                "oracle_rmse": math.sqrt(best),
                "gain": math.sqrt(zero) - math.sqrt(best),
            }
        )
    return {
        "rows": rows,
        "wells": len(wells),
        "base_rmse": math.sqrt(base_sse / rows),
        "oracle_rmse": math.sqrt(oracle_sse / rows),
        "oracle_gain": math.sqrt(base_sse / rows) - math.sqrt(oracle_sse / rows),
    }, pd.DataFrame(bywell)


inner_oracle, inner_oracle_bywell = _oracle_metrics(inner_valid_wells)
fold2_oracle, fold2_oracle_bywell = _oracle_metrics(final_valid_wells)
inner_oracle_bywell.to_csv(OUT_DIR / "inner_fold5_oracle_bywell.csv", index=False)
fold2_oracle_bywell.to_csv(OUT_DIR / "fold2_oracle_bywell.csv", index=False)
print("ORACLE GATE", {"fold5": inner_oracle, "fold2": fold2_oracle}, flush=True)
if fold2_oracle["oracle_rmse"] > ORACLE_GATE_RMSE:
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "candidate_oracle_gate",
        "fold2_oracle": fold2_oracle,
        "gate": ORACLE_GATE_RMSE,
        "strict_pass": False,
        "next": "FAIL -> exported strong-centred path bank lost the required ceiling",
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    raise RuntimeError(json.dumps(verdict, indent=2))


# ---------------------------------------------------------------------------
# Complete-well candidate set model
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels, channels, 5, padding=2 * dilation,
                dilation=dilation, groups=channels,
            ),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.network(values))


class StrongCentredSetPosterior(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(len(CHANNELS), 48, 7, stride=2, padding=3),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            ResidualBlock(48, 1),
            ResidualBlock(48, 2),
            ResidualBlock(48, 4),
            nn.Conv1d(48, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualBlock(64, 2),
            ResidualBlock(64, 4),
        )
        self.summary = nn.Sequential(
            nn.Linear(len(SUMMARY_NAMES), 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, 48),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(176, 128), nn.LayerNorm(128), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 1)
        )

    def forward(self, sequence, summary):
        candidates = sequence.shape[0]
        hidden = self.sequence(sequence)
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)
        summary_hidden = self.summary(summary)
        candidate = self.fusion(torch.cat([pooled, summary_hidden], dim=1))
        candidate = self.candidate_set(candidate[None])[0]
        return self.score(candidate).squeeze(-1)


def _panel_tensors(panel, device, control="full"):
    sequence = panel["sequence"].astype(np.float32)
    summary = panel["summary"].astype(np.float32)
    if control == "no_gr":
        sequence[:, GR_SEQUENCE_CHANNELS, :] = 0.0
        summary[:, GR_SUMMARY_CHANNELS] = 0.0
    elif control == "shuffled_alignment":
        shift = max(1, len(sequence) // 7)
        original_sequence = sequence.copy()
        original_summary = summary.copy()
        sequence[:, CANDIDATE_ALIGNMENT_CHANNELS, :] = np.roll(
            original_sequence[:, CANDIDATE_ALIGNMENT_CHANNELS, :], shift, axis=0
        )
        summary[:, GR_SUMMARY_CHANNELS] = np.roll(
            original_summary[:, GR_SUMMARY_CHANNELS], shift, axis=0
        )
    return (
        torch.from_numpy(sequence).to(device, non_blocking=True),
        torch.from_numpy(summary).to(device, non_blocking=True),
        torch.from_numpy(panel["correction_sequence"].astype(np.float32)).to(
            device, non_blocking=True
        ),
        torch.from_numpy(panel["truth_residual_sequence"].astype(np.float32)).to(
            device, non_blocking=True
        ),
        torch.from_numpy(panel["candidate_mse"].astype(np.float32)).to(
            device, non_blocking=True
        ),
        torch.from_numpy(panel["log_multiplicity"].astype(np.float32)).to(
            device, non_blocking=True
        ),
    )


def _loss(model, panel, device):
    sequence, summary, corrections, truth_residual, candidate_mse, log_mult = (
        _panel_tensors(panel, device)
    )
    raw_scores = model(sequence, summary)
    scores = raw_scores - MULTIPLICITY_STRENGTH * log_mult
    candidate_rmse = torch.sqrt(candidate_mse.clamp_min(1e-6))
    target_logits = (
        -(candidate_rmse - candidate_rmse.amin()) / TARGET_TEMPERATURE_FT
        - MULTIPLICITY_STRENGTH * log_mult
    )
    target = torch.softmax(target_logits.float(), dim=0)
    probability = torch.softmax(scores.float(), dim=0)
    listwise = -(target * F.log_softmax(scores.float(), dim=0)).sum()
    posterior = probability @ corrections
    path_mse = torch.mean((posterior - truth_residual) ** 2) / 64.0
    expected_mse = torch.sum(probability * candidate_mse) / 64.0

    best = int(torch.argmin(candidate_mse).item())
    endpoint = corrections[:, -1]
    endpoint_gap = torch.abs(endpoint - endpoint[best])
    alias = (
        ((endpoint_gap >= 16.0) & (endpoint_gap <= 32.0))
        | ((endpoint_gap >= 40.0) & (endpoint_gap <= 56.0))
    ) & (candidate_rmse >= candidate_rmse[best] + 4.0)
    if bool(alias.any()):
        alias_score = torch.logsumexp(scores[alias], dim=0)
        alias_loss = F.relu(ALIAS_MARGIN - scores[best] + alias_score)
    else:
        alias_loss = scores.new_zeros(())
    total = (
        LOSS_LISTWISE * listwise
        + LOSS_POSTERIOR_PATH * path_mse
        + LOSS_EXPECTED_MSE * expected_mse
        + LOSS_ALIAS_MARGIN * alias_loss
    )
    return total, {
        "listwise": listwise.detach(),
        "path_mse": (path_mse * 64.0).detach(),
        "expected_mse": (expected_mse * 64.0).detach(),
        "alias": alias_loss.detach(),
    }


def _new_training(device, epochs):
    _seed_everything(SEED + epochs)
    model = StrongCentredSetPosterior().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    return model, optimizer, scheduler, scaler


def _train_epoch(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = np.random.default_rng(SEED + 1009 * epoch).permutation(len(wells))
    mean_rows = float(
        np.mean([int(_load_panel(well)["eval_rows"][0]) for well in wells])
    )
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "listwise": 0.0, "path_mse": 0.0, "expected_mse": 0.0, "alias": 0.0}
    for step, position in enumerate(order, 1):
        well = wells[int(position)]
        panel = _load_panel(well)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss, parts = _loss(model, panel, device)
            row_weight = int(panel["eval_rows"][0]) / max(mean_rows, 1.0)
            scaled_loss = loss * row_weight / GRAD_ACCUM_WELLS
        scaler.scale(scaled_loss).backward()
        if step % GRAD_ACCUM_WELLS == 0 or step == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        totals["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            totals[key] += float(value.cpu())
        if step % 100 == 0 or step == len(order):
            print(
                f"SET-POSTERIOR epoch={epoch} wells={step}/{len(order)} "
                f"loss={totals['loss']/step:.4f}",
                flush=True,
            )
        del panel
    return {key: value / len(order) for key, value in totals.items()}


@torch.no_grad()
def _scores_for_well(model, well, device, temperature=1.0, control="full"):
    model.eval()
    panel = _load_panel(well)
    sequence, summary, _, _, _, log_mult = _panel_tensors(panel, device, control)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        raw = model(sequence, summary)
    adjusted = raw.float() - MULTIPLICITY_STRENGTH * log_mult.float()
    probability = torch.softmax(adjusted / float(temperature), dim=0).cpu().numpy()
    return probability, panel


def _well_prediction(well, probability):
    frame = pool_groups[well]
    with np.load(_candidate_path(well), allow_pickle=False) as source:
        correction = (
            source["correction_q01"].astype(np.float32)
            * float(source["quantisation_ft"][0])
        )
    move = np.asarray(probability, np.float64) @ correction.astype(np.float64)
    return frame["strong"].to_numpy(np.float64) + move, move


def _summarise_rows(rows):
    frame = pd.concat(rows, ignore_index=True)
    error = frame["prediction"].to_numpy(float) - frame["truth"].to_numpy(float)
    base_error = frame["strong"].to_numpy(float) - frame["truth"].to_numpy(float)
    bywell = []
    for well, group in frame.groupby("well", sort=True):
        local_error = group["prediction"].to_numpy(float) - group["truth"].to_numpy(float)
        local_base = group["strong"].to_numpy(float) - group["truth"].to_numpy(float)
        bywell.append(
            {
                "well": well,
                "rows": len(group),
                "base_rmse": float(np.sqrt(np.mean(local_base**2))),
                "rmse": float(np.sqrt(np.mean(local_error**2))),
                "gain": float(np.sqrt(np.mean(local_base**2)) - np.sqrt(np.mean(local_error**2))),
                "entropy": float(group["entropy"].iloc[0]),
                "mean_abs_move": float(np.mean(np.abs(group["move"]))),
            }
        )
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": len(frame),
        "wells": len(bywell),
        "base_rmse": float(np.sqrt(np.mean(base_error**2))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "gain_vs_base": float(np.sqrt(np.mean(base_error**2)) - np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "positive_well_fraction": float((bywell["gain"] > 0).mean()),
        "median_well_gain": float(bywell["gain"].median()),
        "mean_entropy": float(bywell["entropy"].mean()),
        "mean_abs_move": float(np.mean(np.abs(frame["move"]))),
        "p95_abs_move": float(np.quantile(np.abs(frame["move"]), 0.95)),
    }
    for label, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (frame["distance"] >= low) & (frame["distance"] < high)
        metrics[f"{label}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
        metrics[f"{label}_base_rmse"] = float(np.sqrt(np.mean(base_error[mask] ** 2)))
        metrics[f"{label}_gain"] = metrics[f"{label}_base_rmse"] - metrics[f"{label}_rmse"]
    return metrics, bywell, frame


@torch.no_grad()
def _evaluate(model, wells, device, temperature=1.0, control="full"):
    rows = []
    for index, well in enumerate(wells, 1):
        probability, _ = _scores_for_well(
            model, well, device, temperature=temperature, control=control
        )
        prediction, move = _well_prediction(well, probability)
        frame = pool_groups[well]
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["MD"]
        )
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        eval_rows = frame["row_idx"].to_numpy(int)
        anchor = int(eval_rows.min() - 1)
        entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        rows.append(
            pd.DataFrame(
                {
                    "well": well,
                    "row_idx": eval_rows,
                    "truth": frame["tvt_true"].to_numpy(float),
                    "strong": frame["strong"].to_numpy(float),
                    "prediction": prediction,
                    "move": move,
                    "distance": md[eval_rows] - md[anchor],
                    "entropy": entropy,
                }
            )
        )
        if index % 50 == 0 or index == len(wells):
            print(
                f"evaluate {control} {index}/{len(wells)} temp={temperature:.2f}",
                flush=True,
            )
    return _summarise_rows(rows)


# ---------------------------------------------------------------------------
# Nested epoch/temperature selection and untouched Fold-2 application
# ---------------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("set-posterior gateway requires CUDA")

print(
    "SET-POSTERIOR CONFIG",
    {
        "device": str(device),
        "inner_train_wells": len(inner_train_wells),
        "inner_valid_wells": len(inner_valid_wells),
        "final_train_wells": len(final_train_wells),
        "fold2_wells": len(final_valid_wells),
        "sample_unit": "one complete physical well",
        "sequence_length": SEQUENCE_LENGTH,
        "summary_features": len(SUMMARY_NAMES),
    },
    flush=True,
)

model, optimizer, scheduler, scaler = _new_training(device, MAX_EPOCHS)
parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
history = []
best_state = None
best_epoch = None
best_rmse = np.inf
stale = 0
for epoch in range(1, MAX_EPOCHS + 1):
    train_metrics = _train_epoch(
        model, inner_train_wells, optimizer, scaler, device, epoch
    )
    scheduler.step()
    valid_metrics, _, _ = _evaluate(
        model, inner_valid_wells, device, temperature=1.0
    )
    row = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"valid_{key}": value for key, value in valid_metrics.items()},
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    history.append(row)
    print("SET-POSTERIOR INNER EPOCH", row, flush=True)
    if valid_metrics["rmse"] < best_rmse - 1e-4:
        best_rmse = valid_metrics["rmse"]
        best_epoch = epoch
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        stale = 0
    else:
        stale += 1
    if epoch >= MIN_EPOCHS and stale >= PATIENCE:
        break

pd.DataFrame(history).to_csv(OUT_DIR / "inner_training_history.csv", index=False)
if best_state is None:
    raise RuntimeError("inner training did not produce a checkpoint")
model.load_state_dict(best_state)

temperature_rows = []
for temperature in TEMPERATURES:
    metrics, _, _ = _evaluate(
        model, inner_valid_wells, device, temperature=temperature
    )
    temperature_rows.append({"temperature": temperature, **metrics})
temperature_scan = pd.DataFrame(temperature_rows).sort_values("rmse")
temperature_scan.to_csv(OUT_DIR / "inner_temperature_scan.csv", index=False)
selected_temperature = float(temperature_scan.iloc[0]["temperature"])

torch.save(
    {
        "model": best_state,
        "epoch": int(best_epoch),
        "temperature": selected_temperature,
        "channels": CHANNELS,
        "summary_names": SUMMARY_NAMES,
        "candidate_source": str(CANDIDATE_DIR),
        "fold_source": str(FOLD_PATH),
    },
    MODEL_DIR / "inner_fold5_selected.pt",
)

del model, optimizer, scheduler, scaler
gc.collect()
torch.cuda.empty_cache()

final_model, final_optimizer, final_scheduler, final_scaler = _new_training(
    device, int(best_epoch)
)
final_history = []
for epoch in range(1, int(best_epoch) + 1):
    train_metrics = _train_epoch(
        final_model,
        final_train_wells,
        final_optimizer,
        final_scaler,
        device,
        100 + epoch,
    )
    final_scheduler.step()
    final_history.append({"epoch": epoch, **train_metrics})
    print("SET-POSTERIOR FINAL TRAIN", final_history[-1], flush=True)
pd.DataFrame(final_history).to_csv(OUT_DIR / "final_training_history.csv", index=False)

fold2_metrics, fold2_bywell, fold2_rows = _evaluate(
    final_model,
    final_valid_wells,
    device,
    temperature=selected_temperature,
    control="full",
)
no_gr_metrics, _, _ = _evaluate(
    final_model,
    final_valid_wells,
    device,
    temperature=selected_temperature,
    control="no_gr",
)
shuffle_metrics, _, _ = _evaluate(
    final_model,
    final_valid_wells,
    device,
    temperature=selected_temperature,
    control="shuffled_alignment",
)

fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
fold2_rows.to_pickle(OUT_DIR / "fold2_predictions.pkl")
controls = pd.DataFrame(
    [
        {"mode": "full", **fold2_metrics, "degradation_vs_full": 0.0},
        {
            "mode": "no_gr",
            **no_gr_metrics,
            "degradation_vs_full": no_gr_metrics["rmse"] - fold2_metrics["rmse"],
        },
        {
            "mode": "shuffled_alignment",
            **shuffle_metrics,
            "degradation_vs_full": shuffle_metrics["rmse"] - fold2_metrics["rmse"],
        },
    ]
)
controls.to_csv(OUT_DIR / "fold2_controls.csv", index=False)

final_state = {
    key: value.detach().cpu().clone()
    for key, value in final_model.state_dict().items()
}
checkpoint_path = MODEL_DIR / "fold2_gateway_setposterior.pt"
torch.save(
    {
        "model": final_state,
        "epoch": int(best_epoch),
        "temperature": selected_temperature,
        "channels": CHANNELS,
        "summary_names": SUMMARY_NAMES,
        "candidate_source": str(CANDIDATE_DIR),
        "fold_source": str(FOLD_PATH),
        "fold2_metrics": fold2_metrics,
    },
    checkpoint_path,
)

no_gr_degradation = no_gr_metrics["rmse"] - fold2_metrics["rmse"]
shuffle_degradation = shuffle_metrics["rmse"] - fold2_metrics["rmse"]
strict_pass = bool(
    fold2_metrics["rmse"] <= FOLD2_RMSE_GATE
    and fold2_metrics["gain_vs_base"] >= FOLD2_GAIN_GATE
    and fold2_metrics["far_gt2000_gain"] >= FAR_GAIN_GATE
    and no_gr_degradation >= CONTROL_DEGRADATION_GATE
    and shuffle_degradation >= CONTROL_DEGRADATION_GATE
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_fold2_strong_centered_complete_well_set_posterior",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "strong_role": "candidate coordinate only; not an input channel",
    "protocol": {
        "inner_train_folds": INNER_TRAIN_FOLDS,
        "inner_valid_fold": INNER_VALID_FOLD,
        "final_train_folds": FINAL_TRAIN_FOLDS,
        "final_valid_fold": FINAL_VALID_FOLD,
        "fold1_touched": False,
        "selected_epoch": int(best_epoch),
        "selected_temperature": selected_temperature,
    },
    "parameters": parameter_count,
    "candidate_oracle": {"fold5": inner_oracle, "fold2": fold2_oracle},
    "fold2": fold2_metrics,
    "controls": {
        "no_gr": no_gr_metrics,
        "shuffled_alignment": shuffle_metrics,
        "no_gr_degradation": no_gr_degradation,
        "shuffled_alignment_degradation": shuffle_degradation,
    },
    "gates": {
        "rmse_le": FOLD2_RMSE_GATE,
        "gain_vs_base_ge": FOLD2_GAIN_GATE,
        "far_gt2000_gain_ge": FAR_GAIN_GATE,
        "each_control_degradation_ge": CONTROL_DEGRADATION_GATE,
    },
    "strict_pass": strict_pass,
    "checkpoint": str(checkpoint_path),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> repeat on a second outer fold before five-fold expansion"
        if strict_pass
        else "FAIL -> close strong-centred path-set selection despite its oracle"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nINNER TEMPERATURE SCAN")
print(temperature_scan.to_string(index=False), flush=True)
print("\nFOLD2 CONTROLS")
print(controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
