"""V244: literal 26th-place disclosed-core reproduction.

Only documented components are enabled: SMP ResNet34 ImageNet U-Net, 5-channel
256x512 panels, detached stacked stage 2, EMA, and self re-feed. The undisclosed
synthetic generator is deliberately off in this baseline.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

from stacked_unet_real_v242 import load_folds, load_record, panel


ROOT = Path(os.environ.get("ROGII_ROOT", "/root/autodl-tmp"))
OUT = ROOT / "oof_exports" / "literal_26th_resnet34_v244"
MODE = os.environ.get("V244_MODE", "audit").lower()
VALID_FOLD = int(os.environ.get("V244_VALID_FOLD", "1"))
SEED = int(os.environ.get("V244_SEED", "244"))
H, W = 256, 512
STEP_FT = 1.0
TEMPERATURE = 1.5
EMA_DECAY = 0.997


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class PanelDataset(Dataset):
    def __init__(self, records, training):
        self.records = records
        self.training = training
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        rng = np.random.default_rng(SEED + self.epoch * 100003 + index)
        if self.training:
            aug = {
                "md_factor": float(rng.uniform(0.90, 1.10)),
                "bed_factor": float(rng.uniform(0.90, 1.10)),
                "bed_bend": float(rng.uniform(-0.5, 0.5)),
                "hint_drop": float(rng.uniform(0.0, 0.20)),
                "anchor_jitter": float(rng.uniform(-12.0, 12.0)),
            }
            flip = bool(rng.integers(0, 2))
        else:
            aug, flip = {}, False
        p = panel(self.records[index], aug, flip)
        return {
            "image": torch.from_numpy(p["image"]),
            "target": torch.from_numpy(np.rint(p["target"] / STEP_FT + H / 2).astype(np.int64)),
            "target_offset": torch.from_numpy(p["target"]),
            "mask": torch.from_numpy(p["mask"]),
        }


class EMA:
    def __init__(self, model):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.shadow.parameters(), model.parameters()):
            a.mul_(EMA_DECAY).add_(b, alpha=1.0 - EMA_DECAY)


def make_model(in_channels):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=os.environ.get("V244_ENCODER_WEIGHTS", "imagenet"),
        in_channels=in_channels,
        classes=1,
        activation=None,
        decoder_channels=(512, 256, 128, 64, 32),
    )


def loss_fn(logits, target, mask):
    offsets = torch.arange(H, device=logits.device, dtype=torch.float32)[None, :, None]
    target_offset = target.float()
    target_bin = (target_offset / STEP_FT + H / 2.0).clamp(0.0, H - 1.0)
    soft_target = torch.exp(-torch.abs(offsets - target_bin[:, None, :]) / 1.5)
    soft_target = soft_target / soft_target.sum(1, keepdim=True).clamp_min(1e-6)
    ce = -(soft_target * F.log_softmax(logits, 1)).sum(1)
    probability = torch.softmax(logits, 1)
    expected = ((probability * offsets).sum(1) - H / 2.0) * STEP_FT
    path = F.smooth_l1_loss(expected, target_offset, reduction="none", beta=2.0)
    return ((ce + 0.15 * path) * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def predict(stage1, stage2, record, device, rounds=2):
    p = panel(record)
    base = torch.from_numpy(p["image"])[None].to(device)
    first = stage1(base).squeeze(1)
    belief = torch.softmax(first / TEMPERATURE, 1)[:, None]
    logits = stage2(torch.cat([base, belief], 1)).squeeze(1)
    for _ in range(max(0, rounds - 1)):
        belief = torch.softmax(logits / TEMPERATURE, 1)[:, None]
        logits = stage2(torch.cat([base, belief], 1)).squeeze(1)
    offsets = (torch.arange(H, device=device) - H // 2) * STEP_FT
    expected = (torch.softmax(logits, 1) * offsets[None, :, None]).sum(1)[0].cpu().numpy()
    surface = record.anchor_tvt + expected + p["z"]
    rows = np.arange(record.anchor + 1, len(record.md))
    return rows, np.interp(record.md[rows], p["md"], surface) - record.z[rows]


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
    verdict = {
        "experiment": "literal_26th_resnet34_v244",
        "fold": VALID_FOLD,
        "train_wells": len(train_records),
        "valid_wells": len(valid_records),
        "input_shape": [5, H, W],
        "backbone": "resnet34",
        "encoder_weights": os.environ.get("V244_ENCODER_WEIGHTS", "imagenet"),
        "decoder_channels": [512, 256, 128, 64, 32],
        "stacked": True,
        "detach": True,
        "ema": EMA_DECAY,
        "synthetic_pretraining": False,
        "fold_source": str(fold_path),
    }
    print("V244 CONFIG", json.dumps(verdict), flush=True)
    if MODE == "audit":
        (OUT / "audit.json").write_text(json.dumps(verdict, indent=2))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V244 requires CUDA")
    stage1, stage2 = make_model(5).to(device), make_model(6).to(device)
    ema1, ema2 = EMA(stage1), EMA(stage2)
    optimizer = torch.optim.AdamW(list(stage1.parameters()) + list(stage2.parameters()), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ds = PanelDataset(train_records, True)
    loader = DataLoader(ds, batch_size=int(os.environ.get("V244_BATCH", "8")), shuffle=True, num_workers=2, pin_memory=True)
    epochs = int(os.environ.get("V244_EPOCHS", "4" if MODE == "smoke" else "18"))
    accum = int(os.environ.get("V244_ACCUM", "1"))
    best, history = None, []
    for epoch in range(1, epochs + 1):
        ds.set_epoch(epoch); stage1.train(); stage2.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target_offset"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                l1 = stage1(image).squeeze(1)
                belief = torch.softmax(l1 / TEMPERATURE, 1).detach()[:, None]
                l2 = stage2(torch.cat([image, belief], 1)).squeeze(1)
                if random.random() < 0.5:
                    belief = torch.softmax(l2 / TEMPERATURE, 1).detach()[:, None]
                    l2 = stage2(torch.cat([image, belief], 1)).squeeze(1)
                loss = loss_fn(l2, target, mask) + 0.5 * loss_fn(l1, target, mask)
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); ema1.update(stage1); ema2.update(stage2)
            losses.append(float(loss.detach().cpu()))
        errors, well_rmse = [], []
        for record in valid_records:
            rows, prediction = predict(ema1.shadow, ema2.shadow, record, device, rounds=2)
            finite = np.isfinite(record.tvt[rows]) & np.isfinite(prediction)
            err = prediction[finite] - record.tvt[rows][finite]
            errors.append(err); well_rmse.append(float(np.sqrt(np.mean(err * err))))
        err = np.concatenate(errors)
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)), "rmse": float(np.sqrt(np.mean(err * err))), "mae": float(np.mean(np.abs(err))), "median_well_rmse": float(np.median(well_rmse)), "p95_well_rmse": float(np.percentile(well_rmse, 95))}
        history.append(metrics); print("V244 EPOCH", metrics, flush=True)
        if best is None or metrics["rmse"] < best["rmse"]:
            best = metrics
            torch.save({"stage1": ema1.shadow.state_dict(), "stage2": ema2.shadow.state_dict(), "metrics": metrics}, OUT / f"fold{VALID_FOLD}_seed{SEED}.pt")
    verdict.update({"best": best, "history": history, "epochs": epochs})
    (OUT / f"fold{VALID_FOLD}_seed{SEED}_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("V244 FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
