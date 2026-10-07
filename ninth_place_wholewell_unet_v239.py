"""V239: reproducible baseline for the ninth-place whole-well U-Net.

The script is intentionally staged. ``audit`` validates the representation,
``smoke`` trains a small honest well-held-out gateway, and ``full`` trains one
complete outer fold. It consumes only raw horizontal/typewell files and a well
fold map; no historical prediction endpoint is used.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(os.environ.get("ROGII_ROOT", "/root/autodl-tmp"))
DATA_ROOT = next(
    path for path in (
        ROOT / "rogii-wellbore-geology-prediction",
        Path.cwd(),
    )
    if (path / "train").is_dir()
)
OOF_ROOT = ROOT / "oof_exports"
GRID_MODE = os.environ.get("V239_GRID_MODE", "fixed16").lower()
MODEL_VARIANT = os.environ.get("V239_MODEL_VARIANT", "stock").lower()
if MODEL_VARIANT not in {"stock", "relief"}:
    raise ValueError(f"unknown V239_MODEL_VARIANT={MODEL_VARIANT}")
EXPERIMENT = (
    f"ninth_place_{MODEL_VARIANT}_wholewell_unet_v241"
    if GRID_MODE == "fixed16"
    else "ninth_place_wholewell_unet_v239"
)
OUT_ROOT = Path(os.environ.get("V239_OUT_ROOT", str(OOF_ROOT)))
OUT = OUT_ROOT / EXPERIMENT
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "likpf_residual_gbdt_full5_v215" / "well_metadata.csv",
)

MODE = os.environ.get("V239_MODE", "audit").lower()
VALID_FOLD = int(os.environ.get("V239_VALID_FOLD", "1"))
SEED = int(os.environ.get("V239_SEED", "239"))
N_HISTORY = 128
N_EVAL = 640
N_COLUMNS = N_HISTORY + N_EVAL
N_STATES = 384
STATE_STEP = 0.75
OFFSETS = (np.arange(N_STATES, dtype=np.float32) - 192.0) * STATE_STEP
HISTORY_FT = float(os.environ.get("V239_HISTORY_FT", "2048"))
MD_POOL_FT = float(os.environ.get("V239_MD_POOL_FT", "16"))
TARGET_SIGMA = 1.5
CHANNELS = (
    "signed_gr_difference",
    "absolute_gr_difference",
    "gr_product",
    "horizontal_gr",
    "typewell_gr",
    "candidate_offset",
    "md_relative",
    "z_relative",
    "known_tvt_heatmap",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def robust_normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    valid = np.isfinite(values)
    output = np.zeros_like(values)
    if valid.sum() < 8:
        return output
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    output[valid] = np.clip((values[valid] - centre) / scale, -8.0, 8.0)
    return output


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) < 8:
        return 0.0, 1.0
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    return centre, scale


def robust_affine(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    """Fit observed ~= slope * reference + intercept on visible prefix GR."""
    reference = np.asarray(reference, np.float64)
    observed = np.asarray(observed, np.float64)
    valid = np.isfinite(reference) & np.isfinite(observed)
    x, y = reference[valid], observed[valid]
    if len(x) < 24 or np.std(x) < 1e-4:
        return 1.0, float(np.nanmedian(y - x)) if len(x) else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    for _ in range(5):
        residual = y - (slope * x + intercept)
        centre = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - centre)) + 1e-3)
        weights = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        design = np.column_stack([x, np.ones_like(x)])
        lhs = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * y)
        try:
            slope, intercept = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
    return float(np.clip(slope, 0.25, 4.0)), float(intercept)


def fill(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, np.float64))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


@dataclass
class Record:
    well: str
    md: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    truth: np.ndarray
    tvt_input: np.ndarray
    tw_tvt: np.ndarray
    tw_gr: np.ndarray
    anchor: int

    @property
    def anchor_md(self) -> float:
        return float(self.md[self.anchor])

    @property
    def anchor_tvt(self) -> float:
        return float(self.tvt_input[self.anchor])


def load_record(well: str) -> Record:
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    required = {"MD", "Z", "GR", "TVT", "TVT_input"}
    if not required.issubset(horizontal):
        raise RuntimeError(f"{well}: missing horizontal columns {sorted(required - set(horizontal))}")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    if len(known) < 32:
        raise RuntimeError(f"{well}: insufficient visible prefix")
    tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    return Record(
        well=well,
        md=md.astype(np.float32),
        z=fill(z),
        gr=fill(gr),
        truth=truth.astype(np.float32),
        tvt_input=tvt_input.astype(np.float32),
        tw_tvt=tw["TVT"].to_numpy(np.float32),
        tw_gr=tw["GR"].to_numpy(np.float32),
        anchor=int(known[-1]),
    )


def sample_md(record: Record) -> np.ndarray:
    if GRID_MODE == "fixed16":
        relative = np.concatenate(
            [
                np.arange(-N_HISTORY, 0, dtype=np.float32),
                np.arange(1, N_EVAL + 1, dtype=np.float32),
            ]
        )
        return (record.anchor_md + MD_POOL_FT * relative).astype(np.float32)
    history_start = max(float(record.md[0]), record.anchor_md - HISTORY_FT)
    history = np.linspace(history_start, record.anchor_md, N_HISTORY, endpoint=False)
    future = np.linspace(record.anchor_md, float(record.md[-1]), N_EVAL + 1)[1:]
    return np.concatenate([history, future]).astype(np.float32)


def compose_panel(
    record: Record,
    centre_shift: float = 0.0,
    horizontal_flip: bool = False,
    path_relief_scale: float = 1.0,
    thickness_scale: float = 1.0,
    thickness_bend: float = 0.0,
) -> dict[str, np.ndarray | float]:
    columns_md = sample_md(record)
    column_valid = (columns_md >= float(record.md[0])) & (columns_md <= float(record.md[-1]))
    columns_z = np.interp(columns_md, record.md, record.z).astype(np.float32)
    columns_gr_raw = np.interp(columns_md, record.md, record.gr).astype(np.float32)
    columns_truth = np.interp(columns_md, record.md, record.truth).astype(np.float32)

    # Calibrate the typewell to the current well using only the visible TVT
    # prefix. This is the legal affine calibration used by the competition's
    # strongest alignment pipelines and is essential for cost-volume quality.
    known = np.flatnonzero(np.isfinite(record.tvt_input))
    tw_at_known = np.interp(
        record.tvt_input[known], record.tw_tvt, record.tw_gr, left=np.nan, right=np.nan
    )
    slope, intercept = robust_affine(tw_at_known, record.gr[known])
    calibrated_tw_raw = slope * record.tw_gr + intercept
    if abs(path_relief_scale - 1.0) > 1e-6:
        augmented_truth = record.anchor_tvt + path_relief_scale * (columns_truth - record.anchor_tvt)
        original_reference = np.interp(
            columns_truth, record.tw_tvt, calibrated_tw_raw, left=np.nan, right=np.nan
        )
        augmented_reference = np.interp(
            augmented_truth, record.tw_tvt, calibrated_tw_raw, left=np.nan, right=np.nan
        )
        residual = columns_gr_raw - original_reference
        synthetic = augmented_reference + residual
        replace = np.isfinite(synthetic)
        columns_gr_raw = np.where(replace, synthetic, columns_gr_raw).astype(np.float32)
        columns_truth = augmented_truth.astype(np.float32)
    centre, scale = robust_location_scale(record.gr)
    horizontal_gr = np.clip((columns_gr_raw - centre) / scale, -8.0, 8.0).astype(np.float32)
    horizontal_gr[~column_valid] = 0.0
    typewell_gr = np.clip((calibrated_tw_raw - centre) / scale, -8.0, 8.0).astype(np.float32)
    centre_tvt = record.anchor_tvt + float(centre_shift)
    candidate_tvt = centre_tvt + OFFSETS
    relative_tw = record.tw_tvt - record.anchor_tvt
    warped_tw_tvt = (
        record.anchor_tvt
        + thickness_scale * relative_tw
        + thickness_bend * 144.0 * np.tanh(relative_tw / 144.0)
    ).astype(np.float32)
    if np.any(np.diff(warped_tw_tvt) <= 0):
        raise RuntimeError("non-monotone thickness warp")
    type_values = np.interp(
        candidate_tvt,
        warped_tw_tvt,
        typewell_gr,
        left=np.nan,
        right=np.nan,
    ).astype(np.float32)
    state_valid = np.isfinite(type_values)
    type_values = np.nan_to_num(type_values, nan=0.0)

    h = np.broadcast_to(horizontal_gr[None], (N_STATES, N_COLUMNS))
    tw = np.broadcast_to(type_values[:, None], (N_STATES, N_COLUMNS))
    valid = np.broadcast_to(state_valid[:, None], (N_STATES, N_COLUMNS)).copy()
    signed = h - tw
    target_offset = columns_truth - centre_tvt
    history_mask = (np.arange(N_COLUMNS) < N_HISTORY) & column_valid
    known_heat = np.zeros((N_STATES, N_COLUMNS), np.float32)
    known_heat[:, history_mask] = np.exp(
        -0.5 * ((OFFSETS[:, None] - target_offset[None, history_mask]) / TARGET_SIGMA) ** 2
    )
    md_relative = (columns_md - record.anchor_md) / max(float(record.md[-1] - record.anchor_md), 1.0)
    z_relative = (columns_z - float(record.z[record.anchor])) / 256.0
    input_values = np.stack(
        [
            signed,
            np.abs(signed),
            h * tw,
            h,
            tw,
            np.broadcast_to(((OFFSETS + centre_shift) / 144.0)[:, None], h.shape),
            np.broadcast_to(md_relative[None], h.shape),
            np.broadcast_to(z_relative[None], h.shape),
            known_heat,
        ],
        axis=0,
    ).astype(np.float32)
    input_values[:, :, ~column_valid] = 0.0
    nearest = np.clip(np.rint((target_offset - OFFSETS[0]) / STATE_STEP).astype(int), 0, N_STATES - 1)
    target_valid = (
        np.isfinite(target_offset)
        & column_valid
        & (target_offset >= OFFSETS[0])
        & (target_offset <= OFFSETS[-1])
        & state_valid[nearest]
    )
    loss_mask = target_valid & ~history_mask

    if horizontal_flip:
        input_values = input_values[..., ::-1].copy()
        target_offset = target_offset[::-1].copy()
        loss_mask = loss_mask[::-1].copy()
        columns_md = columns_md[::-1].copy()
        columns_z = columns_z[::-1].copy()
        column_valid = column_valid[::-1].copy()

    return {
        "input": input_values,
        "state_valid": valid,
        "target_offset": target_offset.astype(np.float32),
        "loss_mask": loss_mask.astype(np.float32),
        "column_md": columns_md,
        "column_z": columns_z,
        "column_valid": column_valid,
        "centre_tvt": float(centre_tvt),
    }


class WellDataset(Dataset):
    def __init__(self, records: list[Record], training: bool):
        self.records = records
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(SEED + self.epoch * 100003 + index)
        shift = float(rng.choice([-18.0, -9.0, 0.0, 9.0, 18.0])) if self.training else 0.0
        flip = bool(rng.integers(0, 2)) if self.training else False
        if self.training and MODEL_VARIANT == "relief":
            path_scale = float(rng.uniform(0.82, 1.18))
            thickness_scale = float(rng.uniform(0.90, 1.10))
            thickness_bend = float(rng.uniform(-0.05, 0.05))
        else:
            path_scale = 1.0
            thickness_scale = 1.0
            thickness_bend = 0.0
        panel = compose_panel(
            self.records[index],
            shift,
            flip,
            path_relief_scale=path_scale,
            thickness_scale=thickness_scale,
            thickness_bend=thickness_bend,
        )
        return {
            "input": torch.from_numpy(panel["input"]),
            "state_valid": torch.from_numpy(panel["state_valid"]),
            "target_offset": torch.from_numpy(panel["target_offset"]),
            "loss_mask": torch.from_numpy(panel["loss_mask"]),
        }


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.GroupNorm(8, cout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(self.norm2(self.conv2(x)) + residual)


def make_layer(cin: int, cout: int, blocks: int, stride: int) -> nn.Sequential:
    return nn.Sequential(BasicBlock(cin, cout, stride), *[BasicBlock(cout, cout) for _ in range(blocks - 1)])


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(BasicBlock(cin + skip, cout), BasicBlock(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResNet34UNet(nn.Module):
    def __init__(self, input_channels: int = len(CHANNELS)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = make_layer(64, 64, 3, 1)
        self.layer2 = make_layer(64, 128, 4, 2)
        self.layer3 = make_layer(128, 256, 6, 2)
        self.layer4 = make_layer(256, 512, 3, 2)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 64, 48)
        self.head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        e1 = self.layer1(self.pool(s0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        x = self.dec3(e4, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        x = self.dec0(x, s0)
        x = F.interpolate(x, size=(N_STATES, N_COLUMNS), mode="bilinear", align_corners=False)
        return self.head(x).squeeze(1)


def objective(logits, valid, target, mask):
    logits = logits.float().masked_fill(~valid[:, :, :1], -30.0)
    offsets = torch.as_tensor(OFFSETS, device=logits.device)[None, :, None]
    distance = offsets - target[:, None]
    soft_target = torch.softmax(-0.5 * (distance / TARGET_SIGMA).square(), dim=1)
    log_probability = F.log_softmax(logits, dim=1)
    ce = -(soft_target * log_probability).sum(dim=1)
    probability = torch.softmax(logits, dim=1)
    expected = (probability * offsets).sum(dim=1)
    huber = F.smooth_l1_loss(expected, target, reduction="none", beta=3.0)
    denominator = mask.sum().clamp_min(1.0)
    loss = ((ce + 0.08 * huber) * mask).sum() / denominator
    return loss, expected


@torch.no_grad()
def predict_record(model, record, device, views=((0.0, False),)):
    predictions = []
    for shift, flip in views:
        panel = compose_panel(record, shift, flip)
        values = torch.from_numpy(panel["input"])[None].to(device)
        valid = torch.from_numpy(panel["state_valid"])[None, :, :1].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(values).float().masked_fill(~valid, -30.0)
        probability = torch.softmax(logits, dim=1)
        offsets = torch.as_tensor(OFFSETS, device=device)[None, :, None]
        expected = (probability * offsets).sum(dim=1)[0].cpu().numpy()
        if flip:
            expected = expected[::-1].copy()
        predictions.append(record.anchor_tvt + shift + expected)
    column_tvt = np.mean(predictions, axis=0)
    columns_md = sample_md(record)
    columns_z = np.interp(columns_md, record.md, record.z)
    column_valid = (columns_md >= float(record.md[0])) & (columns_md <= float(record.md[-1]))
    eval_rows = np.arange(record.anchor + 1, len(record.md))
    surface = column_tvt[column_valid] + columns_z[column_valid]
    prediction = np.interp(record.md[eval_rows], columns_md[column_valid], surface) - record.z[eval_rows]
    return eval_rows, prediction


@torch.no_grad()
def evaluate(model, records, device, views=((0.0, False),)):
    model.eval()
    errors, by_well = [], []
    for record in records:
        rows, prediction = predict_record(model, record, device, views)
        truth = record.truth[rows]
        finite = np.isfinite(truth) & np.isfinite(prediction)
        error = prediction[finite] - truth[finite]
        errors.append(error)
        by_well.append({"well": record.well, "rows": int(finite.sum()), "rmse": float(np.sqrt(np.mean(error**2)))})
    error = np.concatenate(errors)
    by_well = pd.DataFrame(by_well)
    return {
        "rows": int(len(error)),
        "wells": int(len(records)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(by_well.rmse.median()),
        "p95_well_rmse": float(by_well.rmse.quantile(0.95)),
    }, by_well


@torch.no_grad()
def export_predictions(model, records, device) -> pd.DataFrame:
    model.eval()
    parts = []
    for record in records:
        rows, prediction = predict_record(model, record, device)
        parts.append(
            pd.DataFrame(
                {
                    "well": record.well,
                    "row_idx": rows.astype(np.int32),
                    "truth": record.truth[rows].astype(np.float32),
                    "prediction": prediction.astype(np.float32),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def load_fold_map():
    path = next((path for path in FOLD_CANDIDATES if path.is_file()), None)
    if path is None:
        raise FileNotFoundError({"fold_candidates": [str(path) for path in FOLD_CANDIDATES]})
    frame = pd.read_csv(path, dtype={"well": str})
    frame = frame.groupby("well", as_index=False).fold.first()
    return frame, path


def main():
    if MODE not in {"audit", "smoke", "full"}:
        raise ValueError(f"unknown V239_MODE={MODE}")
    started = time.time()
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    fold, fold_path = load_fold_map()
    train_wells = fold.loc[fold.fold.ne(VALID_FOLD), "well"].astype(str).tolist()
    valid_wells = fold.loc[fold.fold.eq(VALID_FOLD), "well"].astype(str).tolist()
    rng = np.random.default_rng(SEED)
    if MODE == "audit":
        train_wells = sorted(rng.choice(train_wells, min(32, len(train_wells)), replace=False))
        valid_wells = sorted(rng.choice(valid_wells, min(16, len(valid_wells)), replace=False))
    elif MODE == "smoke":
        train_count = int(os.environ.get("V239_TRAIN_WELLS", "96"))
        valid_count = int(os.environ.get("V239_VALID_WELLS", "24"))
        train_wells = sorted(rng.choice(train_wells, min(train_count, len(train_wells)), replace=False))
        valid_wells = sorted(rng.choice(valid_wells, min(valid_count, len(valid_wells)), replace=False))
    print("V239 LOAD", {"mode": MODE, "train": len(train_wells), "valid": len(valid_wells)}, flush=True)
    train_records = [load_record(well) for well in train_wells]
    valid_records = [load_record(well) for well in valid_wells]
    coverages = []
    for record in train_records + valid_records:
        panel = compose_panel(record)
        future = np.arange(N_COLUMNS) >= N_HISTORY
        valid_future = future & panel["column_valid"]
        coverages.append(float(np.mean(panel["loss_mask"][valid_future] > 0)))
    audit = {
        "experiment": EXPERIMENT,
        "mode": MODE,
        "fold_source": str(fold_path),
        "train_wells": len(train_records),
        "valid_wells": len(valid_records),
        "input_shape": [len(CHANNELS), N_STATES, N_COLUMNS],
        "history_columns": N_HISTORY,
        "eval_columns": N_EVAL,
        "grid_mode": GRID_MODE,
        "model_variant": MODEL_VARIANT,
        "md_pool_ft": MD_POOL_FT if GRID_MODE == "fixed16" else None,
        "state_step_ft": STATE_STEP,
        "axis_min_ft": float(OFFSETS[0]),
        "axis_max_ft": float(OFFSETS[-1]),
        "mean_future_axis_coverage": float(np.mean(coverages)),
        "p05_future_axis_coverage": float(np.quantile(coverages, 0.05)),
    }
    print("V239 AUDIT", json.dumps(audit, indent=2), flush=True)
    if audit["p05_future_axis_coverage"] < 0.98:
        raise RuntimeError(f"state axis coverage failed: {audit}")
    if MODE == "audit":
        (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V239 training requires CUDA")
    torch.set_float32_matmul_precision("high")
    model = ResNet34UNet().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    epochs = int(os.environ.get("V239_EPOCHS", "4" if MODE == "smoke" else "18"))
    accumulation = int(os.environ.get("V239_ACCUMULATION", "4"))
    dataset = WellDataset(train_records, training=True)
    # Workers are recreated each epoch so they receive the epoch-dependent
    # deterministic shift/flip augmentation state.
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best = {"rmse": float("inf"), "epoch": 0}
    history = []
    checkpoint = OUT / f"fold{VALID_FOLD}_{MODE}_seed{SEED}.pt"
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loader, 1):
            values = batch["input"].to(device, non_blocking=True)
            valid = batch["state_valid"].to(device, non_blocking=True)
            target = batch["target_offset"].to(device, non_blocking=True)
            mask = batch["loss_mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(values)
                loss, _ = objective(logits, valid, target, mask)
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        metrics, by_well = evaluate(model, valid_records, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics, "elapsed_minutes": (time.time() - started) / 60.0}
        history.append(row)
        print("V239 EPOCH", row, flush=True)
        if metrics["rmse"] < best["rmse"]:
            best = {"rmse": metrics["rmse"], "epoch": epoch, "metrics": metrics}
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, checkpoint)
            by_well.to_csv(OUT / f"fold{VALID_FOLD}_{MODE}_best_bywell.csv", index=False)
    pd.DataFrame(history).to_csv(OUT / f"fold{VALID_FOLD}_{MODE}_history.csv", index=False)
    best_checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    best_rows = export_predictions(model, valid_records, device)
    try:
        best_rows.to_parquet(OUT / f"fold{VALID_FOLD}_{MODE}_seed{SEED}_oof.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        best_rows.to_pickle(OUT / f"fold{VALID_FOLD}_{MODE}_seed{SEED}_oof.pkl")
    verdict = {**audit, "parameters": parameters, "epochs": epochs, "best": best, "elapsed_minutes": (time.time() - started) / 60.0}
    (OUT / f"fold{VALID_FOLD}_{MODE}_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("V239 FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
