"""V243: first-place-style whole-well ConvNeXt alignment reproduction.

This is an independently scored endpoint, not a modification of the old GBDT
chain. It implements the disclosed, reproducible core: 345 MD bins, 400 TVT
bins at 0.5 ft, ConvNeXt U-Net, GR statistics, soft CE, path Huber, GR penalty,
and TVT+Z-preserving / GR affine augmentation.
"""

from __future__ import annotations

import copy
import json
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
DATA_ROOT = next(p for p in (ROOT / "rogii-wellbore-geology-prediction", Path.cwd()) if (p / "train").is_dir())
OOF_ROOT = ROOT / "oof_exports"
RUN_TAG = os.environ.get("V243_RUN_TAG", "default")
OUT = OOF_ROOT / ("first_place_convnext_alignment_v243" if RUN_TAG == "default" else f"first_place_convnext_alignment_v243_{RUN_TAG}")
FOLD_PATHS = (OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",)

MODE = os.environ.get("V243_MODE", "audit").lower()
VALID_FOLD = int(os.environ.get("V243_VALID_FOLD", "1"))
SEED = int(os.environ.get("V243_SEED", "243"))
H, W = 400, 345
TVT_STEP = 0.5
TVT_RADIUS = 100.0
MD_BIN_FT = 32.0
VISIBLE_FT = 1024.0
TARGET_TAU = float(os.environ.get("V243_TARGET_TAU", "1.5"))
GR_PENALTY = float(os.environ.get("V243_GR_PENALTY", "0.01"))
EMA_DECAY = 0.997


@dataclass
class Record:
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
    def anchor_tvt(self):
        return float(self.tvt_input[self.anchor])

    @property
    def anchor_md(self):
        return float(self.md[self.anchor])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def fill(x):
    return pd.Series(np.asarray(x, float)).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)


def norm(x):
    x = np.asarray(x, np.float32)
    valid = np.isfinite(x)
    if valid.sum() < 8:
        return np.zeros_like(x)
    med = np.median(x[valid])
    scale = max(1.4826 * np.median(np.abs(x[valid] - med)), float(np.std(x[valid])), 1.0)
    out = np.zeros_like(x)
    out[valid] = np.clip((x[valid] - med) / scale, -8.0, 8.0)
    return out


def bin_stats(x, value, centres):
    x = np.asarray(x, float)
    value = np.asarray(value, float)
    edges = np.r_[[-np.inf], (centres[:-1] + centres[1:]) / 2.0, [np.inf]]
    group = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(centres) - 1)
    valid = np.isfinite(x) & np.isfinite(value)
    total = np.bincount(group[valid], weights=value[valid], minlength=len(centres))
    count = np.bincount(group[valid], minlength=len(centres))
    fallback = np.interp(centres, x[valid], value[valid])
    mean = np.divide(total, count, out=fallback.copy(), where=count > 0)
    sq = np.bincount(group[valid], weights=(value[valid] - mean[group[valid]]) ** 2, minlength=len(centres))
    std = np.sqrt(np.divide(sq, count, out=np.zeros_like(sq), where=count > 0))
    return mean.astype(np.float32), std.astype(np.float32), count.astype(np.float32)


def horizontal_bin_features(md, gr, centres):
    """Published 32-ft bin summaries, including the missing polynomial terms."""
    edges = np.r_[centres - MD_BIN_FT / 2.0, centres[-1] + MD_BIN_FT / 2.0]
    group = np.searchsorted(edges, md, side="right") - 1
    output = np.full((8, len(centres)), np.nan, np.float32)
    count = np.zeros(len(centres), np.float32)
    for col in range(len(centres)):
        take = (group == col) & np.isfinite(md) & np.isfinite(gr)
        if take.sum() == 0:
            continue
        x = (md[take] - centres[col]) / MD_BIN_FT
        y = gr[take].astype(float)
        count[col] = len(y)
        output[0, col] = np.mean(y)
        output[1, col] = np.std(y)
        output[2, col] = (y[-1] - y[0]) / max(md[take][-1] - md[take][0], 1.0)
        output[3, col] = y[-1] - y[0]
        degree = min(2, len(y) - 1)
        coefficient = np.polyfit(x, y, degree)
        coefficient = np.pad(coefficient, (3 - len(coefficient), 0))
        fitted = np.polyval(coefficient, x)
        output[4:7, col] = coefficient
        output[7, col] = np.sqrt(np.mean((y - fitted) ** 2))
    return output, count


def load_record(well):
    h = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    t = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    md = pd.to_numeric(h["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(h["Z"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(h["GR"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(h["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    tw = t[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna().groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    if len(known) < 32 or len(tw) < 32:
        raise RuntimeError(f"{well}: insufficient prefix/typewell")
    return Record(well, md.astype(np.float32), fill(z), fill(gr), tvt.astype(np.float32), tvt_input.astype(np.float32), tw.TVT.to_numpy(np.float32), tw.GR.to_numpy(np.float32), int(known[-1]))


def panel(record, augment=None):
    augment = augment or {}
    md = record.anchor_md - VISIBLE_FT + (np.arange(W, dtype=np.float32) + 0.5) * MD_BIN_FT
    inside = (md >= record.md[0]) & (md <= record.md[-1])
    z = np.interp(md, record.md, record.z).astype(np.float32)
    hfeature_raw, hcount = horizontal_bin_features(record.md, record.gr, md)
    hfeature_raw[0] = hfeature_raw[0] * float(augment.get("gr_scale", 1.0)) + float(augment.get("gr_bias", 0.0))
    hmean_raw = hfeature_raw[0]
    hnan = (hcount == 0).astype(np.float32)
    tw_raw = record.tw_gr.copy()
    tw_raw = tw_raw * float(augment.get("gr_scale", 1.0)) + float(augment.get("gr_bias", 0.0))
    axis = record.anchor_tvt + (np.arange(H, dtype=np.float32) - H / 2.0 + 0.5) * TVT_STEP
    known_ids = np.flatnonzero(np.isfinite(record.tvt_input) & np.isfinite(record.gr))
    if len(known_ids) >= 16:
        observed, observed_count = bin_stats(record.tvt_input[known_ids], record.gr[known_ids], axis)
        observed_valid = observed_count > 0
        if observed_valid.sum() >= 8:
            observed_interp = np.interp(axis, axis[observed_valid], observed[observed_valid])
            original = np.interp(axis, record.tw_tvt, tw_raw)
            blend = np.clip(observed_count / 8.0, 0.0, 1.0)
            calibrated_axis = blend * observed_interp + (1.0 - blend) * original
            tw_raw = np.interp(record.tw_tvt, axis, calibrated_axis, left=tw_raw[0], right=tw_raw[-1])
    hmean = norm(hmean_raw)
    tw = norm(tw_raw)
    tw_row = np.interp(axis, record.tw_tvt, tw, left=0.0, right=0.0).astype(np.float32)
    tw_valid = ((axis >= record.tw_tvt.min()) & (axis <= record.tw_tvt.max())).astype(np.float32)
    candidate = axis[:, None]
    target = np.interp(md, record.md, record.tvt, left=np.nan, right=np.nan)
    target_bin = np.clip(np.rint((target - record.anchor_tvt + TVT_RADIUS) / TVT_STEP), 0, H - 1).astype(np.int64)
    known = (md <= record.anchor_md) & inside
    hint = np.zeros((H, W), np.float32)
    for col in np.flatnonzero(known):
        row = int(np.clip(np.rint((np.interp(md[col], record.md, record.tvt_input) - record.anchor_tvt + TVT_RADIUS) / TVT_STEP), 0, H - 1))
        if col == np.flatnonzero(known)[-1] or random.random() >= float(augment.get("hint_drop", 0.0)):
            hint[row, col] = 1.0
    cost = np.square(tw_row[:, None] - hmean[None, :]).astype(np.float32)
    abs_cost = np.abs(tw_row[:, None] - hmean[None, :]).astype(np.float32)
    hfeature = np.stack([norm(row) for row in hfeature_raw[1:]]).astype(np.float32)
    visible_raw = np.interp(md, record.md[:record.anchor + 1], record.tvt_input[:record.anchor + 1], left=np.nan, right=np.nan)
    visible = np.where(known & np.isfinite(visible_raw), visible_raw - record.anchor_tvt, 0.0).astype(np.float32)
    visible_grid = np.broadcast_to(visible[None], (H, W)).copy()
    tvt_grid = np.broadcast_to((axis - record.anchor_tvt)[:, None] / TVT_RADIUS, (H, W)).copy()
    z_diff = np.broadcast_to(((z - record.z[record.anchor]) / TVT_RADIUS)[None], (H, W)).copy()
    tvt_visible_gap = np.abs((axis[:, None] - record.anchor_tvt) - visible[None, :]).astype(np.float32) / TVT_RADIUS
    channels = [
        np.broadcast_to(tw_row[:, None], (H, W)),
        np.broadcast_to(tw_valid[:, None], (H, W)),
        tvt_grid,
        np.broadcast_to(hmean[None], (H, W)),
        np.broadcast_to(hnan[None], (H, W)),
        *[np.broadcast_to(row[None], (H, W)) for row in hfeature],
        visible_grid,
        hint,
        abs_cost,
        tvt_visible_gap,
        z_diff,
    ]
    image = np.stack(channels).astype(np.float32)
    valid = inside & np.isfinite(target) & ~known & (target - record.anchor_tvt >= -TVT_RADIUS) & (target - record.anchor_tvt <= TVT_RADIUS)
    return {"image": image, "target": target_bin, "target_offset": target - record.anchor_tvt, "mask": valid.astype(np.float32), "md": md, "z": z, "anchor": record.anchor_tvt}


class WellDataset(Dataset):
    def __init__(self, records, training):
        self.records, self.training, self.epoch = records, training, 0

    def __len__(self): return len(self.records)
    def set_epoch(self, epoch): self.epoch = epoch

    def __getitem__(self, index):
        rng = np.random.default_rng(SEED + self.epoch * 100003 + index)
        aug = {}
        if self.training:
            aug = {"hint_drop": float(rng.uniform(0.0, 0.2)), "gr_scale": float(rng.uniform(0.85, 1.15)), "gr_bias": float(rng.normal(0.0, 0.05))}
        p = panel(self.records[index], aug)
        return {k: torch.from_numpy(v) for k, v in p.items() if k in {"image", "target", "target_offset", "mask", "md", "z"}}


class EMA:
    def __init__(self, model):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.shadow.parameters(), model.parameters()): a.mul_(EMA_DECAY).add_(b, alpha=1 - EMA_DECAY)


def loss_fn(logits, target_offset, mask, tw_gap):
    grid = (torch.arange(H, device=logits.device, dtype=torch.float32) - H / 2.0 + 0.5) * TVT_STEP
    target_distribution = torch.exp(-torch.abs(grid[None, :, None] - target_offset[:, None, :]) / TARGET_TAU)
    target_distribution = target_distribution / target_distribution.sum(1, keepdim=True).clamp_min(1e-8)
    ce = -(target_distribution * F.log_softmax(logits, 1)).sum(1)
    prob = torch.softmax(logits, 1)
    expected = (prob * grid[None, :, None]).sum(1)
    path = F.smooth_l1_loss(expected, target_offset, reduction="none", beta=2.0)
    gr = (prob * tw_gap).sum(1)
    total = ce + 0.15 * path + GR_PENALTY * gr
    return (total * mask).sum() / mask.sum().clamp_min(1.0)


class TimmConvNeXtUNet(nn.Module):
    """Direct timm ConvNeXt feature pyramid with the disclosed wide decoder."""

    def __init__(self, in_channels):
        super().__init__()
        import timm

        name = os.environ.get("V243_TIMM_MODEL", "convnext_small.in12k_ft_in1k_384")
        local_checkpoint = os.environ.get("V243_ENCODER_CHECKPOINT", "").strip()
        pretrained = os.environ.get("V243_ENCODER_WEIGHTS", "imagenet") == "imagenet" and not local_checkpoint
        self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        if local_checkpoint:
            checkpoint = Path(local_checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if checkpoint.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(checkpoint), device="cpu")
            else:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                state = state.get("state_dict", state.get("model", state))
            state = {k.removeprefix("module."): v for k, v in state.items()}
            current = self.encoder.state_dict()
            expanded = dict(state)
            for key, value in state.items():
                expanded.setdefault(f"model.{key}", value)
                expanded.setdefault(f"encoder.{key}", value)
                if key.startswith("model."):
                    expanded.setdefault(key.removeprefix("model."), value)
            state = expanded
            normalized_current = {}
            for key in current:
                normalized_current.setdefault(key.replace(".", "").replace("_", ""), []).append(key)
            for key, value in list(state.items()):
                candidates = [key] if key in current else normalized_current.get(key.replace(".", "").replace("_", ""), [])
                for candidate in candidates:
                    if value.ndim == 4 and current[candidate].shape != value.shape:
                        if value.shape[1] == 3 and current[candidate].shape[1] == in_channels:
                            value = value.mean(1, keepdim=True).repeat(1, in_channels, 1, 1)
                            state[key] = value
                    break
            compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
            current_normalized = {}
            for key in current:
                normalized = key.replace(".", "").replace("_", "")
                current_normalized.setdefault(normalized, []).append(key)
            for key, value in state.items():
                normalized = key.replace(".", "").replace("_", "")
                for candidate in current_normalized.get(normalized, []):
                    if candidate not in compatible and current[candidate].shape == value.shape:
                        compatible[candidate] = value
                        break
            missing, unexpected = self.encoder.load_state_dict(compatible, strict=False)
            report = {"path": str(checkpoint), "loaded": len(compatible), "missing": len(missing), "unexpected": len(unexpected)}
            print("V243 LOCAL ENCODER", {**report, "missing_keys": missing[:5]}, flush=True)
            if len(compatible) < 100:
                raise RuntimeError({"message": "Local ConvNeXt checkpoint did not match timm encoder", **report, "checkpoint_keys": list(state)[:5], "model_keys": list(current)[:5]})
        channels = list(self.encoder.feature_info.channels())
        decoder = [512, 256, 128, 64, 32]
        level_channels = list(reversed(decoder[:len(channels)]))
        self.level_channels = level_channels
        self.lateral = nn.ModuleList([nn.Conv2d(c, d, 1) for c, d in zip(channels[-len(level_channels):], level_channels)])
        self.blocks = nn.ModuleList()
        for i in range(len(self.lateral) - 1, 0, -1):
            out_channels = level_channels[i - 1]
            in_channels_block = level_channels[i] + out_channels
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels_block, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.GELU(),
            ))
        self.head = nn.Conv2d(level_channels[0], 1, 1)

    def forward(self, x):
        feats = self.encoder(x)
        feats = feats[-len(self.lateral):]
        y = self.lateral[-1](feats[-1])
        block_index = 0
        for i in range(len(feats) - 2, -1, -1):
            skip = self.lateral[i](feats[i])
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = self.blocks[block_index](torch.cat([y, skip], 1))
            block_index += 1
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(y)


def load_folds():
    path = next((p for p in FOLD_PATHS if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(FOLD_PATHS)
    return pd.read_csv(path, dtype={"well": str}).groupby("well", as_index=False).fold.first(), path


def make_model():
    import segmentation_models_pytorch as smp
    if os.environ.get("V243_DIRECT_TIMM", "1") == "1":
        return TimmConvNeXtUNet(17), os.environ.get("V243_TIMM_MODEL", "convnext_small.in12k_ft_in1k_384")
    names = [
        os.environ.get("V243_ENCODER", "tu-convnext_small.in12k_ft_in1k_384"),
        "timm-convnext_small.in12k_ft_in1k_384",
        "tu-convnext_small.in12k_ft_in1k_384",
    ]
    available = set(smp.encoders.get_encoder_names())
    encoder = next((name for name in names if name in available), None)
    if encoder is None:
        raise RuntimeError({"message": "ConvNeXt encoder unavailable", "requested": names, "available_sample": sorted(available)[:30]})
    weights = os.environ.get("V243_ENCODER_WEIGHTS", "imagenet")
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=17, classes=1, activation=None, decoder_channels=(512, 256, 128, 64, 32)), encoder


@torch.no_grad()
def predict(model, record, device):
    p = panel(record)
    x = torch.from_numpy(p["image"])[None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x).float().squeeze(1)
    bins = torch.arange(H, device=device, dtype=torch.float32)[None, :, None]
    expected = ((torch.softmax(logits, 1) * bins).sum(1) - H / 2.0 + 0.5).cpu().numpy()[0] * TVT_STEP
    surface = record.anchor_tvt + expected + p["z"]
    rows = np.arange(record.anchor + 1, len(record.md))
    prediction = np.interp(record.md[rows], p["md"], surface) - record.z[rows]
    return rows, prediction


def main():
    if MODE not in {"audit", "smoke", "full"}:
        raise ValueError(MODE)
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    fold, fold_path = load_folds()
    train_ids = fold.loc[fold.fold.ne(VALID_FOLD), "well"].astype(str).tolist()
    valid_ids = fold.loc[fold.fold.eq(VALID_FOLD), "well"].astype(str).tolist()
    rng = np.random.default_rng(SEED)
    if MODE == "audit":
        train_ids, valid_ids = train_ids[:16], valid_ids[:8]
    elif MODE == "smoke":
        train_ids = sorted(rng.choice(train_ids, min(48, len(train_ids)), False))
        valid_ids = sorted(rng.choice(valid_ids, min(12, len(valid_ids)), False))
    train_records = [load_record(w) for w in train_ids]
    valid_records = [load_record(w) for w in valid_ids]
    audit = {"experiment": "first_place_convnext_alignment_v243", "fold": VALID_FOLD, "train_wells": len(train_records), "valid_wells": len(valid_records), "shape": [17, H, W], "encoder": os.environ.get("V243_ENCODER", "tu-convnext_small.in12k_ft_in1k_384"), "encoder_weights": os.environ.get("V243_ENCODER_WEIGHTS", "imagenet"), "loss": "exponential_soft_ce_plus_path_huber_plus_gr_penalty", "synthetic_pretraining": False, "fold_source": str(fold_path)}
    print("V243 CONFIG", json.dumps(audit), flush=True)
    if MODE == "audit":
        (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V243 requires CUDA")
    model, encoder = make_model()
    model = model.to(device)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("V243_LR", "2e-4")), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ds = WellDataset(train_records, True)
    loader = DataLoader(ds, batch_size=int(os.environ.get("V243_BATCH", "1")), shuffle=True, num_workers=2, pin_memory=True)
    epochs = int(os.environ.get("V243_EPOCHS", "4" if MODE == "smoke" else "18"))
    accum = int(os.environ.get("V243_ACCUM", "2"))
    best, history = None, []
    for epoch in range(1, epochs + 1):
        ds.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            target_offset = batch["target_offset"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            tw_gap = image[:, 14]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image).squeeze(1)
                loss = loss_fn(logits, target_offset, mask, tw_gap)
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); ema.update(model)
            losses.append(float(loss.detach().cpu()))
        errors, well_metrics = [], []
        for record in valid_records:
            rows, pred = predict(ema.shadow, record, device)
            finite = np.isfinite(record.tvt[rows]) & np.isfinite(pred)
            err = pred[finite] - record.tvt[rows][finite]
            errors.append(err); well_metrics.append(float(np.sqrt(np.mean(err * err))))
        err = np.concatenate(errors)
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)), "rmse": float(np.sqrt(np.mean(err * err))), "mae": float(np.mean(np.abs(err))), "median_well_rmse": float(np.median(well_metrics)), "p95_well_rmse": float(np.percentile(well_metrics, 95))}
        history.append(metrics); print("V243 EPOCH", metrics, flush=True)
        if best is None or metrics["rmse"] < best["rmse"]:
            best = metrics
            torch.save({"model": ema.shadow.state_dict(), "encoder": encoder, "metrics": metrics}, OUT / f"fold{VALID_FOLD}_seed{SEED}.pt")
    verdict = {**audit, "encoder": encoder, "best": best, "history": history, "next": "Only after the single-fold gate passes, run frozen folds and seeds"}
    (OUT / f"fold{VALID_FOLD}_seed{SEED}_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("V243 FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
