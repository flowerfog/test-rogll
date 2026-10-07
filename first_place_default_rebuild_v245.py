"""V245: audited reproduction of the disclosed first-place default model.

Scope:
  - Implements only the disclosed default alignment model (no PF/XY channels).
  - Separates disclosed mechanics from inferred training hyperparameters.
  - Refuses to train until representation, target, decoder, and encoder audits pass.

This is not presented as the winner's unreleased source code. It is a clean,
testable reconstruction of the public description.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(os.environ.get("ROGII_ROOT", "/root/autodl-tmp"))
DATA_ROOT = next(
    path for path in (ROOT / "rogii-wellbore-geology-prediction", Path.cwd())
    if (path / "train").is_dir()
)
OOF_ROOT = ROOT / "oof_exports"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
FOLD_CANDIDATES = (
    FOLD_PATH,
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
    OOF_ROOT / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet",
    OOF_ROOT / "canonical_endpoint_registry_v181" / "v168_fullrow_oof.parquet",
)
RUN_TAG = os.environ.get("V245_RUN_TAG", "default")
OUT = OOF_ROOT / f"first_place_default_rebuild_v245_{RUN_TAG}"

MODE = os.environ.get("V245_MODE", "audit").lower()
VALID_FOLD = int(os.environ.get("V245_VALID_FOLD", "1"))
SEED = int(os.environ.get("V245_SEED", "245"))

HEIGHT = 400
WIDTH = 345
TVT_STEP_FT = 0.5
TVT_RADIUS_FT = 100.0
MD_BIN_FT = 32.0
VISIBLE_FT = 1024.0
TARGET_FT = 10000.0


CHANNELS = (
    "typewell_gr",
    "typewell_gr_nan",
    "typewell_tvt_relative",
    "horizontal_gr_mean",
    "horizontal_gr_nan_fraction",
    "horizontal_gr_std",
    "horizontal_gr_slope",
    "horizontal_gr_last_minus_first",
    "horizontal_gr_quadratic",
    "horizontal_gr_linear",
    "horizontal_gr_intercept",
    "horizontal_gr_quadratic_rmse",
    "visible_tvt_relative",
    "visible_tvt_mask",
    "typewell_horizontal_gr_abs_gap",
    "typewell_visible_tvt_abs_gap",
    "z_diff_from_anchor",
)


@dataclass(frozen=True)
class Recipe:
    # Publicly disclosed geometry.
    height: int = HEIGHT
    width: int = WIDTH
    tvt_step_ft: float = TVT_STEP_FT
    visible_ft: float = VISIBLE_FT
    target_ft: float = TARGET_FT
    md_bin_ft: float = MD_BIN_FT
    encoder: str = "convnext_small.in12k_ft_in1k_384"
    precision: str = "bfloat16"
    # Inferred because the write-up does not disclose exact values.
    target_tau_ft: float = 1.5
    path_huber_weight: float = 0.15
    gr_penalty_weight: float = 0.01
    calibration_blend: float = 0.5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 18
    z_shift_probability: float = 0.5
    gr_transform_probability: float = 0.7
    channel_mask_probability: float = 0.08


RECIPE = Recipe(
    target_tau_ft=float(os.environ.get("V245_TARGET_TAU", "1.5")),
    path_huber_weight=float(os.environ.get("V245_PATH_WEIGHT", "0.15")),
    gr_penalty_weight=float(os.environ.get("V245_GR_WEIGHT", "0.01")),
    calibration_blend=float(os.environ.get("V245_CALIBRATION_BLEND", "0.5")),
    learning_rate=float(os.environ.get("V245_LR", "2e-4")),
    epochs=int(os.environ.get("V245_EPOCHS", "18")),
    z_shift_probability=float(os.environ.get("V245_Z_SHIFT_PROB", "0.0")),
    gr_transform_probability=float(os.environ.get("V245_GR_TRANSFORM_PROB", "0.7")),
    channel_mask_probability=float(os.environ.get("V245_CHANNEL_MASK_PROB", "0.08")),
)


@dataclass
class WellRecord:
    well: str
    md: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt: np.ndarray
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float32)


def fill_coordinate(values: np.ndarray) -> np.ndarray:
    return pd.Series(values.astype(float)).interpolate(limit_direction="both").to_numpy(np.float32)


def robust_normalize(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, np.float32)
    finite = np.isfinite(values) if valid is None else (np.isfinite(values) & valid)
    output = np.zeros_like(values, np.float32)
    if finite.sum() < 4:
        return output
    centre = float(np.median(values[finite]))
    mad = float(1.4826 * np.median(np.abs(values[finite] - centre)))
    scale = max(mad, float(np.std(values[finite])), 1.0)
    output[finite] = np.clip((values[finite] - centre) / scale, -8.0, 8.0)
    return output


def load_record(well: str, split: str = "train") -> WellRecord:
    horizontal_path = DATA_ROOT / split / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / split / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    md = numeric(horizontal, "MD")
    z = fill_coordinate(numeric(horizontal, "Z"))
    gr = numeric(horizontal, "GR")  # Preserve NaNs; they are an explicit feature.
    tvt = numeric(horizontal, "TVT") if "TVT" in horizontal else np.full(len(md), np.nan, np.float32)
    tvt_input = numeric(horizontal, "TVT_input")
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    if len(known) < 16:
        raise RuntimeError(f"{well}: visible TVT prefix is too short")
    tw_tvt = numeric(typewell, "TVT")
    tw_gr = numeric(typewell, "GR")
    order = np.argsort(np.nan_to_num(tw_tvt, nan=np.inf), kind="stable")
    tw_tvt, tw_gr = tw_tvt[order], tw_gr[order]
    valid_tvt = np.isfinite(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[valid_tvt], tw_gr[valid_tvt]
    if np.isfinite(tw_gr).sum() < 16:
        raise RuntimeError(f"{well}: typewell GR is too sparse")
    return WellRecord(str(well), md, z, gr, tvt, tvt_input, tw_tvt, tw_gr, int(known[-1]))


def alignment_axes(record: WellRecord) -> tuple[np.ndarray, np.ndarray]:
    md_centres = record.anchor_md - VISIBLE_FT + (np.arange(WIDTH, dtype=np.float32) + 0.5) * MD_BIN_FT
    tvt_axis = record.anchor_tvt + (np.arange(HEIGHT, dtype=np.float32) - HEIGHT / 2.0 + 0.5) * TVT_STEP_FT
    return md_centres, tvt_axis


def horizontal_statistics(md: np.ndarray, gr: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean, nan fraction, std, slope, delta, quadratic terms, RMSE."""
    edges = np.r_[centres - MD_BIN_FT / 2.0, centres[-1] + MD_BIN_FT / 2.0]
    groups = np.searchsorted(edges, md, side="right") - 1
    result = np.zeros((9, len(centres)), np.float32)
    result[1] = 1.0
    for column in range(len(centres)):
        rows = np.flatnonzero((groups == column) & np.isfinite(md))
        if len(rows) == 0:
            continue
        values = gr[rows]
        finite = np.isfinite(values)
        result[1, column] = 1.0 - float(finite.mean())
        if finite.sum() == 0:
            continue
        x = (md[rows][finite] - centres[column]) / MD_BIN_FT
        y = values[finite].astype(np.float64)
        result[0, column] = float(np.mean(y))
        result[2, column] = float(np.std(y))
        if len(y) >= 2:
            result[3, column] = float(np.polyfit(x, y, 1)[0])
            result[4, column] = float(y[-1] - y[0])
        degree = min(2, len(y) - 1)
        coefficient = np.polyfit(x, y, degree)
        coefficient = np.pad(coefficient, (3 - len(coefficient), 0))
        fitted = np.polyval(coefficient, x)
        result[5:8, column] = coefficient.astype(np.float32)
        result[8, column] = float(np.sqrt(np.mean((y - fitted) ** 2)))
    valid_bin = result[1] < 1.0
    for row in (0, 2, 3, 4, 5, 6, 7, 8):
        result[row] = robust_normalize(result[row], valid_bin)
    return result, valid_bin


def interpolate_typewell(record: WellRecord, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(record.tw_tvt) & np.isfinite(record.tw_gr)
    tvt, gr = record.tw_tvt[valid], record.tw_gr[valid]
    unique_tvt, inverse = np.unique(tvt, return_inverse=True)
    sums = np.bincount(inverse, weights=gr)
    counts = np.bincount(inverse)
    unique_gr = sums / np.maximum(counts, 1)
    values = np.interp(axis, unique_tvt, unique_gr, left=np.nan, right=np.nan).astype(np.float32)
    missing = ~np.isfinite(values)
    return values, missing.astype(np.float32)


def calibrate_typewell(record: WellRecord, axis: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Aggregate visible (TVT, GR) pairs into sorted TVT bins and blend."""
    visible = (
        np.isfinite(record.tvt_input[: record.anchor + 1])
        & np.isfinite(record.gr[: record.anchor + 1])
    )
    tvt = record.tvt_input[: record.anchor + 1][visible]
    gr = record.gr[: record.anchor + 1][visible]
    index = np.rint((tvt - axis[0]) / TVT_STEP_FT).astype(int)
    inside = (index >= 0) & (index < len(axis))
    index, gr = index[inside], gr[inside]
    if len(index) < 16:
        return base
    sums = np.bincount(index, weights=gr, minlength=len(axis))
    counts = np.bincount(index, minlength=len(axis))
    observed = np.divide(sums, counts, out=np.full(len(axis), np.nan), where=counts > 0)
    support = np.flatnonzero(np.isfinite(observed))
    if len(support) < 4:
        return base
    interpolated = np.interp(np.arange(len(axis)), support, observed[support])
    calibrated = base.copy()
    span = np.arange(len(axis)) >= support[0]
    span &= np.arange(len(axis)) <= support[-1]
    finite = span & np.isfinite(base)
    calibrated[finite] = (
        RECIPE.calibration_blend * interpolated[finite]
        + (1.0 - RECIPE.calibration_blend) * base[finite]
    )
    return calibrated.astype(np.float32)


class DifferenceBlockBank:
    def __init__(self, records: list[WellRecord], block: int = 64):
        self.blocks: list[np.ndarray] = []
        for record in records:
            difference = np.diff(record.tvt.astype(float))
            for start in range(0, max(0, len(difference) - block + 1), block):
                item = difference[start : start + block]
                if len(item) == block and np.isfinite(item).all():
                    self.blocks.append(item.astype(np.float32))

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        if not self.blocks:
            raise RuntimeError("No finite TVT-difference blocks available")
        output: list[np.ndarray] = []
        while sum(len(item) for item in output) < length:
            output.append(self.blocks[int(rng.integers(len(self.blocks)))])
        return np.concatenate(output)[:length]


def augmented_record(record: WellRecord, bank: DifferenceBlockBank, rng: np.random.Generator) -> WellRecord:
    result = replace(record)
    if rng.random() < RECIPE.z_shift_probability:
        sampled = bank.sample(len(record.md) - 1, rng)
        path = np.empty(len(record.md), np.float32)
        path[record.anchor] = record.anchor_tvt
        path[record.anchor + 1 :] = record.anchor_tvt + np.cumsum(sampled[record.anchor :])
        path[: record.anchor] = record.anchor_tvt - np.cumsum(sampled[: record.anchor][::-1])[::-1]
        surface = record.tvt + record.z
        z = surface - path
        noise = rng.normal(0.0, 1.5, len(path))
        kernel = np.ones(31, float) / 31.0
        noise = np.convolve(noise, kernel, mode="same")
        valid_tw = np.isfinite(record.tw_tvt) & np.isfinite(record.tw_gr)
        gr = np.interp(path + noise, record.tw_tvt[valid_tw], record.tw_gr[valid_tw], left=np.nan, right=np.nan)
        tvt_input = np.full_like(record.tvt_input, np.nan)
        tvt_input[: record.anchor + 1] = path[: record.anchor + 1]
        result = replace(result, z=z.astype(np.float32), gr=gr.astype(np.float32), tvt=path, tvt_input=tvt_input)
    # Fault augmentation: move the geological TVT path after a future cut,
    # keep the physical well trajectory Z fixed, and regenerate horizontal GR
    # from the shifted typewell correspondence.
    fault_probability = float(os.environ.get("V245_FAULT_PROB", "0.20"))
    if rng.random() < fault_probability and result.anchor + 64 < len(result.md):
        cut = int(rng.integers(result.anchor + 32, len(result.md) - 16))
        jump = float(rng.choice((-1.0, 1.0)) * rng.uniform(4.0, 24.0))
        tvt = result.tvt.copy()
        tvt[cut:] += jump
        valid_tw = np.isfinite(result.tw_tvt) & np.isfinite(result.tw_gr)
        gr = result.gr.copy()
        regenerated = np.interp(
            tvt[cut:] + rng.normal(0.0, 1.0, len(tvt) - cut),
            result.tw_tvt[valid_tw], result.tw_gr[valid_tw],
            left=np.nan, right=np.nan,
        )
        gr[cut:] = regenerated.astype(np.float32)
        result = replace(result, tvt=tvt.astype(np.float32), gr=gr.astype(np.float32))
    return result


def make_panel(
    record: WellRecord,
    training: bool = False,
    bank: DifferenceBlockBank | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    rng = rng or np.random.default_rng(SEED)
    if training and bank is not None:
        record = augmented_record(record, bank, rng)
    md_axis, tvt_axis = alignment_axes(record)
    inside_md = (md_axis >= record.md[0]) & (md_axis <= record.md[-1])
    horizontal, valid_horizontal = horizontal_statistics(record.md, record.gr, md_axis)
    tw_raw, tw_missing = interpolate_typewell(record, tvt_axis)
    tw_raw = calibrate_typewell(record, tvt_axis, tw_raw)
    if training and rng.random() < RECIPE.gr_transform_probability:
        tw_raw = tw_raw * float(rng.uniform(0.75, 1.25)) + float(rng.normal(0.0, 8.0))
    tw_gr = robust_normalize(tw_raw, np.isfinite(tw_raw))
    horizontal_gr = horizontal[0]

    z_axis = np.interp(md_axis, record.md, record.z).astype(np.float32)
    truth = np.interp(md_axis, record.md, record.tvt, left=np.nan, right=np.nan).astype(np.float32)
    known = inside_md & (md_axis <= record.anchor_md)
    visible = np.interp(
        md_axis,
        record.md[: record.anchor + 1],
        record.tvt_input[: record.anchor + 1],
        left=np.nan,
        right=np.nan,
    ).astype(np.float32)
    visible_mask = known & np.isfinite(visible)
    visible_relative = np.where(visible_mask, visible - record.anchor_tvt, 0.0).astype(np.float32)

    tw_grid = np.broadcast_to(tw_gr[:, None], (HEIGHT, WIDTH))
    h_grid = np.broadcast_to(horizontal_gr[None, :], (HEIGHT, WIDTH))
    tvt_relative = (tvt_axis - record.anchor_tvt) / TVT_RADIUS_FT
    visible_grid = np.broadcast_to((visible_relative / TVT_RADIUS_FT)[None, :], (HEIGHT, WIDTH))
    visible_mask_grid = np.broadcast_to(visible_mask.astype(np.float32)[None, :], (HEIGHT, WIDTH))
    tvt_gap = np.abs(tvt_relative[:, None] - visible_grid) * visible_mask_grid

    channels = [
        tw_grid,
        np.broadcast_to(tw_missing[:, None], (HEIGHT, WIDTH)),
        np.broadcast_to(tvt_relative[:, None], (HEIGHT, WIDTH)),
        h_grid,
        np.broadcast_to(horizontal[1][None, :], (HEIGHT, WIDTH)),
        *[np.broadcast_to(horizontal[row][None, :], (HEIGHT, WIDTH)) for row in range(2, 9)],
        visible_grid,
        visible_mask_grid,
        np.abs(tw_grid - h_grid),
        tvt_gap,
        np.broadcast_to(((z_axis - record.z[record.anchor]) / TVT_RADIUS_FT)[None, :], (HEIGHT, WIDTH)),
    ]
    image = np.stack(channels).astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=8.0, neginf=-8.0)
    if training and RECIPE.channel_mask_probability > 0:
        for channel in range(len(CHANNELS)):
            if channel not in (1, 2, 4, 13) and rng.random() < RECIPE.channel_mask_probability:
                image[channel] = 0.0

    target_offset_raw = truth - record.anchor_tvt
    loss_mask = (
        inside_md
        & ~known
        & np.isfinite(target_offset_raw)
        & (target_offset_raw >= -TVT_RADIUS_FT)
        & (target_offset_raw <= TVT_RADIUS_FT)
    )
    target_offset = np.where(np.isfinite(target_offset_raw), target_offset_raw, 0.0).astype(np.float32)
    return {
        "image": image,
        "target_offset": target_offset,
        "mask": loss_mask.astype(np.float32),
        "md": md_axis.astype(np.float32),
        "z": z_axis,
        "truth": truth,
    }


def target_distribution(target_offset: torch.Tensor) -> torch.Tensor:
    grid = (
        torch.arange(HEIGHT, device=target_offset.device, dtype=torch.float32)
        - HEIGHT / 2.0
        + 0.5
    ) * TVT_STEP_FT
    distribution = torch.exp(-torch.abs(grid[None, :, None] - target_offset[:, None, :]) / RECIPE.target_tau_ft)
    return distribution / distribution.sum(1, keepdim=True).clamp_min(1e-8)


def alignment_loss(
    logits: torch.Tensor,
    target_offset: torch.Tensor,
    mask: torch.Tensor,
    gr_gap: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    distribution = target_distribution(target_offset)
    log_probability = F.log_softmax(logits.float(), dim=1)
    probability = log_probability.exp()
    soft_ce = -(distribution * log_probability).sum(1)
    grid = (
        torch.arange(HEIGHT, device=logits.device, dtype=torch.float32)
        - HEIGHT / 2.0
        + 0.5
    ) * TVT_STEP_FT
    expected = (probability * grid[None, :, None]).sum(1)
    path = F.smooth_l1_loss(expected, target_offset, reduction="none", beta=2.0)
    gr = (probability * gr_gap).sum(1)
    total = soft_ce + RECIPE.path_huber_weight * path + RECIPE.gr_penalty_weight * gr
    denominator = mask.sum().clamp_min(1.0)
    loss = (total * mask).sum() / denominator
    parts = {
        "soft_ce": float((soft_ce * mask).sum().detach().cpu() / denominator.detach().cpu()),
        "path": float((path * mask).sum().detach().cpu() / denominator.detach().cpu()),
        "gr": float((gr * mask).sum().detach().cpu() / denominator.detach().cpu()),
    }
    return loss, parts


class ResidualDecoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.residual = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.block(values) + self.residual(values))


def load_encoder_checkpoint(encoder: nn.Module, path: Path, input_channels: int) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        source = load_file(str(path), device="cpu")
    else:
        source = torch.load(path, map_location="cpu", weights_only=False)
        source = source.get("state_dict", source.get("model", source))
    source = {key.removeprefix("module."): value for key, value in source.items()}
    current = encoder.state_dict()
    normalized = {key.replace(".", "").replace("_", ""): key for key in current}
    compatible = {}
    for source_key, value in source.items():
        key = source_key.removeprefix("model.")
        target_key = key if key in current else normalized.get(key.replace(".", "").replace("_", ""))
        if target_key is None:
            continue
        target = current[target_key]
        if value.ndim == 4 and value.shape[1] == 3 and target.shape[1] == input_channels:
            value = value.mean(1, keepdim=True).repeat(1, input_channels, 1, 1) * (3.0 / input_channels)
        if value.shape == target.shape:
            compatible[target_key] = value
    missing, unexpected = encoder.load_state_dict(compatible, strict=False)
    parameter_keys = set(dict(encoder.named_parameters()))
    loaded_parameters = len(parameter_keys.intersection(compatible))
    coverage = loaded_parameters / max(len(parameter_keys), 1)
    report = {
        "path": str(path),
        "loaded_tensors": len(compatible),
        "parameter_coverage": coverage,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
    if coverage < 0.70:
        raise RuntimeError({"message": "Encoder checkpoint coverage is too low", **report})
    return report


class ConvNeXtBNUNet(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        import timm

        try:
            self.encoder = timm.create_model(
                RECIPE.encoder,
                pretrained=False,
                features_only=True,
                in_chans=input_channels,
                conv_mlp=True,
                norm_layer=nn.BatchNorm2d,
            )
        except TypeError as error:
            raise RuntimeError(
                "Installed timm cannot construct ConvNeXt with channel-first BatchNorm; "
                "upgrade timm instead of silently using LayerNorm"
            ) from error
        checkpoint = os.environ.get("V245_ENCODER_CHECKPOINT", "").strip()
        if not checkpoint and os.environ.get("V245_ALLOW_RANDOM_ENCODER", "0") != "1":
            raise RuntimeError(
                "V245 training requires the disclosed ImageNet ConvNeXt checkpoint; "
                "set V245_ENCODER_CHECKPOINT or explicitly opt into random weights"
            )
        self.checkpoint_report = None
        if checkpoint:
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(path)
            self.checkpoint_report = load_encoder_checkpoint(self.encoder, path, input_channels)
        encoder_channels = list(self.encoder.feature_info.channels())
        widths = [64, 128, 256, 512][-len(encoder_channels) :]
        self.lateral = nn.ModuleList(
            nn.Conv2d(source, target, 1, bias=False)
            for source, target in zip(encoder_channels, widths)
        )
        self.decoder = nn.ModuleList()
        for level in range(len(widths) - 1, 0, -1):
            self.decoder.append(ResidualDecoderBlock(widths[level] + widths[level - 1], widths[level - 1]))
        self.head = nn.Conv2d(widths[0], 1, 1)
        # Autodl runs this whole-well model with batch=1. Running BN statistics
        # then make train/eval predictions materially different. Keep BN as the
        # disclosed normalization family, but use current spatial statistics in
        # both modes until a larger training batch is available.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        values = self.lateral[-1](features[-1])
        block = 0
        for level in range(len(features) - 2, -1, -1):
            skip = self.lateral[level](features[level])
            values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            values = self.decoder[block](torch.cat((values, skip), dim=1))
            block += 1
        values = F.interpolate(values, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
        return self.head(values).squeeze(1)


class AlignmentDataset(Dataset):
    def __init__(self, records: list[WellRecord], training: bool, bank: DifferenceBlockBank | None):
        self.records = records
        self.training = training
        self.bank = bank
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(SEED + self.epoch * 100003 + index)
        panel = make_panel(self.records[index], self.training, self.bank, rng)
        return {
            key: torch.from_numpy(panel[key])
            for key in ("image", "target_offset", "mask")
        }


@torch.no_grad()
def decode_panel(logits: torch.Tensor) -> np.ndarray:
    probability = torch.softmax(logits.float(), dim=1)
    grid = (
        torch.arange(HEIGHT, device=logits.device, dtype=torch.float32)
        - HEIGHT / 2.0
        + 0.5
    ) * TVT_STEP_FT
    return (probability * grid[None, :, None]).sum(1).cpu().numpy()


@torch.no_grad()
def predict_record(model: nn.Module, record: WellRecord, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    panel = make_panel(record)
    image = torch.from_numpy(panel["image"])[None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(image)
    offset = decode_panel(logits)[0]
    rows = np.arange(record.anchor + 1, len(record.md))
    prediction = record.anchor_tvt + np.interp(record.md[rows], panel["md"], offset)
    return rows, prediction.astype(np.float32)


def representation_audit(record: WellRecord, model: nn.Module | None = None) -> dict:
    panel = make_panel(record)
    if panel["image"].shape != (len(CHANNELS), HEIGHT, WIDTH):
        raise RuntimeError(f"Panel shape mismatch: {panel['image'].shape}")
    if not np.isfinite(panel["image"]).all():
        raise RuntimeError("Panel contains non-finite values")
    valid = panel["mask"].astype(bool)
    if valid.sum() < 8:
        raise RuntimeError(f"{record.well}: only {valid.sum()} valid target bins")
    target = torch.from_numpy(panel["target_offset"])[None]
    distribution = target_distribution(target)
    sums = distribution.sum(1).numpy()[0, valid]
    max_sum_gap = float(np.max(np.abs(sums - 1.0)))
    oracle_offset = decode_panel(torch.log(distribution.clamp_min(1e-12)))[0]
    oracle_rmse = float(np.sqrt(np.mean((oracle_offset[valid] - panel["target_offset"][valid]) ** 2)))
    if max_sum_gap > 1e-5 or oracle_rmse > 0.10:
        raise RuntimeError({"target_sum_gap": max_sum_gap, "oracle_rmse": oracle_rmse})
    layer_norm_modules = []
    batch_norm_modules = []
    if model is not None:
        layer_norm_modules = [name for name, module in model.named_modules() if "LayerNorm" in type(module).__name__]
        batch_norm_modules = [name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)]
        if layer_norm_modules:
            raise RuntimeError({"message": "LayerNorm remains in ConvNeXt", "modules": layer_norm_modules[:10]})
        if not batch_norm_modules:
            raise RuntimeError("No BatchNorm2d modules found")
    return {
        "well": record.well,
        "shape": list(panel["image"].shape),
        "valid_columns": int(valid.sum()),
        "visible_columns": int(panel["image"][13, 0].sum()),
        "target_distribution_max_sum_gap": max_sum_gap,
        "oracle_decode_rmse": oracle_rmse,
        "layer_norm_modules": len(layer_norm_modules),
        "batch_norm_modules": len(batch_norm_modules),
        "channel_min": panel["image"].min(axis=(1, 2)).tolist(),
        "channel_max": panel["image"].max(axis=(1, 2)).tolist(),
    }


def load_folds() -> pd.DataFrame:
    for path in FOLD_CANDIDATES:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, dtype={"well": str})
        else:
            frame = pd.read_parquet(path)
        if "well" not in frame.columns or "fold" not in frame.columns:
            continue
        result = frame[["well", "fold"]].copy()
        result["well"] = result["well"].astype(str)
        return result.groupby("well", as_index=False).fold.first()

    wells = sorted(
        path.name.removesuffix("__horizontal_well.csv")
        for path in (DATA_ROOT / "train").glob("*__horizontal_well.csv")
    )
    if len(wells) < 5:
        raise FileNotFoundError({"fold_candidates": [str(path) for path in FOLD_CANDIDATES], "train": str(DATA_ROOT / "train")})
    # Stable fallback only. It is reproducible, but it is not claimed to be the
    # deleted historical fold assignment.
    result = pd.DataFrame(
        {
            "well": wells,
            "fold": [
                1 + int(hashlib.sha1(well.encode("utf-8")).hexdigest()[:8], 16) % 5
                for well in wells
            ],
        }
    )
    fallback = OUT / "fallback_well_folds.csv"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(fallback, index=False)
    print("V245 FOLD FALLBACK", {"path": str(fallback), "wells": len(result), "historical_assignment": False}, flush=True)
    return result


def evaluate(model: nn.Module, records: list[WellRecord], device: torch.device) -> dict:
    model.eval()
    errors: list[np.ndarray] = []
    well_rmse: list[float] = []
    for record in records:
        rows, prediction = predict_record(model, record, device)
        valid = np.isfinite(prediction) & np.isfinite(record.tvt[rows])
        error = prediction[valid] - record.tvt[rows][valid]
        errors.append(error)
        well_rmse.append(float(np.sqrt(np.mean(error * error))))
    error = np.concatenate(errors)
    return {
        "rows": int(len(error)),
        "wells": int(len(records)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.percentile(well_rmse, 95)),
    }


@torch.no_grad()
def recalibrate_batch_norm(model: nn.Module, dataset: Dataset, device: torch.device) -> None:
    """Re-estimate BN running statistics before an overfit inference check."""
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    if not batch_norms or any(not module.track_running_stats for module in batch_norms):
        return
    previous = [(module.momentum, module.num_batches_tracked.clone()) for module in batch_norms]
    model.train()
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        model(batch["image"].to(device))
    for module, (momentum, count) in zip(batch_norms, previous):
        module.momentum = momentum
        module.num_batches_tracked.copy_(count)
    model.eval()


def main() -> None:
    if MODE not in {"audit", "overfit", "smoke", "full"}:
        raise ValueError(MODE)
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    folds = load_folds()
    train_ids = folds.loc[folds.fold.ne(VALID_FOLD), "well"].tolist()
    valid_ids = folds.loc[folds.fold.eq(VALID_FOLD), "well"].tolist()
    rng = np.random.default_rng(SEED)
    if MODE == "audit":
        train_ids, valid_ids = train_ids[:2], valid_ids[:1]
    elif MODE == "overfit":
        train_ids = sorted(rng.choice(train_ids, 4, replace=False))
        valid_ids = train_ids
    elif MODE == "smoke":
        train_ids = sorted(rng.choice(train_ids, min(96, len(train_ids)), replace=False))
        valid_ids = sorted(rng.choice(valid_ids, min(24, len(valid_ids)), replace=False))
    train_records = [load_record(well) for well in train_ids]
    valid_records = train_records if MODE == "overfit" else [load_record(well) for well in valid_ids]
    bank = DifferenceBlockBank(train_records)

    disclosed = {
        "grid": "345 MD bins x 400 TVT bins; 32-ft MD and 0.5-ft TVT resolution",
        "features": list(CHANNELS),
        "loss": "exponential soft CE + expected-path Huber + GR gap penalty",
        "model": "ConvNeXt-small U-Net; BatchNorm encoder; residual decoder; interpolation upsampling",
        "augmentations_implemented": ["TVT-difference block-bootstrap Z-shift", "typewell GR affine", "2D channel masking"],
        "augmentations_not_yet_implemented": [
            "fault jumps", "reverse path", "GR residual shift", "MD stretching", "tail cropping",
            "sequential typewell-axis masking", "typewell GR jitter coupling",
        ],
    }
    inferred = {
        key: value
        for key, value in asdict(RECIPE).items()
        if key in {
            "target_tau_ft", "path_huber_weight", "gr_penalty_weight", "calibration_blend",
            "learning_rate", "weight_decay", "epochs", "z_shift_probability",
            "gr_transform_probability", "channel_mask_probability",
        }
    }
    config = {
        "experiment": "first_place_default_rebuild_v245",
        "mode": MODE,
        "fold": VALID_FOLD,
        "seed": SEED,
        "train_wells": len(train_records),
        "valid_wells": len(valid_records),
        "disclosed": disclosed,
        "inferred_not_disclosed": inferred,
        "excluded_from_default_stage": ["particle-filter channels", "XY-neighbor channels", "ensemble gating"],
    }
    print("V245 CONFIG", json.dumps(config), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if MODE != "audit":
        if device.type != "cuda":
            raise RuntimeError("V245 training requires CUDA")
        model = ConvNeXtBNUNet(len(CHANNELS)).to(device)
    audit = representation_audit(valid_records[0], model)
    (OUT / "channel_manifest.json").write_text(json.dumps({"channels": list(CHANNELS)}, indent=2), encoding="utf-8")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("V245 AUDIT", json.dumps(audit), flush=True)
    if MODE == "audit":
        return

    # The overfit gate is a representation test. Augmentation must be disabled;
    # otherwise the target changes on every epoch and the gate is meaningless.
    overfit_gate = MODE == "overfit"
    dataset = AlignmentDataset(train_records, not overfit_gate, None if overfit_gate else bank)
    loader = DataLoader(
        dataset,
        batch_size=int(os.environ.get("V245_BATCH", "1")),
        shuffle=True,
        num_workers=int(os.environ.get("V245_WORKERS", "0")),
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE.learning_rate, weight_decay=RECIPE.weight_decay)
    epochs = int(os.environ.get("V245_EPOCHS", "100" if MODE == "overfit" else str(RECIPE.epochs)))
    accumulation = int(os.environ.get("V245_ACCUM", "2"))
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            target_offset = batch["target_offset"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image)
                loss, _ = alignment_loss(logits, target_offset, mask, image[:, 14])
            if not torch.isfinite(loss):
                raise RuntimeError({"message": "non-finite loss", "epoch": epoch, "step": step})
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if overfit_gate:
            recalibrate_batch_norm(model, dataset, device)
        metrics = evaluate(model, valid_records, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print("V245 EPOCH", row, flush=True)
        if best is None or row["rmse"] < best["rmse"]:
            best = row
            torch.save({"model": model.state_dict(), "channels": CHANNELS, "recipe": asdict(RECIPE), "metrics": row}, OUT / f"fold{VALID_FOLD}_seed{SEED}.pt")
    if MODE == "overfit" and best["rmse"] > 2.0:
        raise RuntimeError({"message": "overfit gate failed", "best_rmse": best["rmse"]})
    verdict = {**config, "audit": audit, "best": best, "history": history}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("V245 FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
