"""V242: disclosed 26th-place stacked U-Net reproduction.

This implements the disclosed real-well core and an explicit, statistics-driven
approximation of the undisclosed synthetic-well pretraining.
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
DATA_ROOT = next(path for path in (ROOT / "rogii-wellbore-geology-prediction", Path.cwd()) if (path / "train").is_dir())
OOF_ROOT = ROOT / "oof_exports"
OUT = OOF_ROOT / "stacked_unet_real_v242"
RUN_TAG = os.environ.get("V242_RUN_TAG", "default")
if RUN_TAG != "default":
    OUT = OOF_ROOT / f"stacked_unet_real_v242_{RUN_TAG}"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "likpf_residual_gbdt_full5_v215" / "well_metadata.csv",
)

MODE = os.environ.get("V242_MODE", "audit").lower()
VALID_FOLD = int(os.environ.get("V242_VALID_FOLD", "1"))
SEED = int(os.environ.get("V242_SEED", "242"))
H = 256
W = 512
WINDOW_FT = 256.0
STEP_FT = WINDOW_FT / H
ANCHOR_JITTER_FT = 12.0
REFEED_TEMPERATURE = 1.5
EMA_DECAY = 0.997
REFEED_TRAIN_PROB = float(os.environ.get("V242_REFEED_TRAIN_PROB", "0.5"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def norm(x):
    x = np.asarray(x, np.float32)
    valid = np.isfinite(x)
    if valid.sum() < 8:
        return np.zeros_like(x)
    med = np.median(x[valid])
    scale = 1.4826 * np.median(np.abs(x[valid] - med))
    scale = max(float(scale), float(np.std(x[valid])), 1.0)
    out = np.zeros_like(x)
    out[valid] = np.clip((x[valid] - med) / scale, -8.0, 8.0)
    return out


def fill(x):
    return pd.Series(np.asarray(x, float)).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)


def binned_mean(x, values, centres):
    x = np.asarray(x, float)
    values = np.asarray(values, float)
    edges = np.empty(len(centres) + 1, float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0], edges[-1] = -np.inf, np.inf
    group = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(centres) - 1)
    valid = np.isfinite(values) & np.isfinite(x)
    total = np.bincount(group[valid], weights=values[valid], minlength=len(centres))
    count = np.bincount(group[valid], minlength=len(centres))
    fallback = np.interp(centres, x[valid], values[valid])
    return np.divide(total, count, out=fallback, where=count > 0).astype(np.float32)


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
    def anchor_md(self):
        return float(self.md[self.anchor])

    @property
    def anchor_tvt(self):
        return float(self.tvt_input[self.anchor])


def load_record(well):
    h = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    t = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    md = pd.to_numeric(h["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(h["Z"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(h["GR"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(h["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    if len(known) < 32:
        raise RuntimeError(f"{well}: insufficient prefix")
    tw = t[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna().groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    return Record(well, md.astype(np.float32), fill(z), fill(gr), tvt.astype(np.float32), tvt_input.astype(np.float32), tw.TVT.to_numpy(np.float32), tw.GR.to_numpy(np.float32), int(known[-1]))


def make_synthetic_record(source, index, seed):
    """Generate a legal-looking lateral from a real typewell template.

    The published generator is unavailable, so this uses only training-side
    typewell GR and trajectory statistics. No held-out record is used as a
    synthetic source during validation.
    """
    rng = np.random.default_rng(seed + index * 7919)
    md = source.md.copy()
    z = source.z.copy()
    anchor = source.anchor
    x = (md - md[anchor]) / 1000.0
    anchor_surface = source.anchor_tvt + float(z[anchor])
    slope = float(rng.normal(0.0, 0.012))
    curvature = float(rng.normal(0.0, 0.004))
    knots_x = np.linspace(x.min(), x.max(), 10)
    knots_y = rng.normal(0.0, 2.5, len(knots_x))
    smooth = np.interp(x, knots_x, knots_y)
    surface = anchor_surface + slope * x * 1000.0 + curvature * x * x * 1000.0 + smooth
    tvt = (surface - z).astype(np.float32)
    lo, hi = float(source.tw_tvt.min()), float(source.tw_tvt.max())
    lookup = np.interp(np.clip(tvt, lo, hi), source.tw_tvt, source.tw_gr)
    tw_scale = max(float(np.std(source.tw_gr)), 1.0)
    drift = np.cumsum(rng.normal(0.0, tw_scale * 0.002, len(md)))
    drift -= drift[anchor]
    noise = rng.normal(0.0, tw_scale * rng.uniform(0.10, 0.28), len(md))
    gr = lookup * rng.uniform(0.88, 1.12) + drift + noise
    tvt_input = np.full(len(md), np.nan, np.float32)
    tvt_input[: anchor + 1] = tvt[: anchor + 1]
    return Record(f"synthetic_{source.well}_{index:04d}", md, z, gr.astype(np.float32), tvt, tvt_input, source.tw_tvt.copy(), source.tw_gr.copy(), anchor)


def panel(record, augment=None, flip=False):
    augment = augment or {}
    md_u = np.linspace(0.0, 1.0, W, dtype=np.float32)
    md_factor = float(augment.get("md_factor", 1.0))
    md_u = np.clip(md_u ** md_factor, 0.0, 1.0)
    columns_md = np.interp(md_u, [0.0, 1.0], [float(record.md[0]), float(record.md[-1])])
    z = np.interp(columns_md, record.md, record.z)
    gr_raw = binned_mean(record.md, record.gr, columns_md)
    truth = np.interp(columns_md, record.md, record.tvt)
    anchor_tvt = record.anchor_tvt + float(augment.get("anchor_jitter", 0.0))
    tw_axis = record.tw_tvt.copy()
    bed_factor = float(augment.get("bed_factor", 1.0))
    bed_bend = float(augment.get("bed_bend", 0.0))
    tw_axis = record.anchor_tvt + bed_factor * (tw_axis - record.anchor_tvt) + bed_bend * 32.0 * np.tanh((tw_axis - record.anchor_tvt) / 64.0)
    tw_values = norm(record.tw_gr)
    tw_grid = np.interp(anchor_tvt + (np.arange(H) - H // 2) * STEP_FT, tw_axis, tw_values, left=np.nan, right=np.nan)
    tw_grid = np.nan_to_num(tw_grid).astype(np.float32)
    h_grid = norm(gr_raw).astype(np.float32)
    candidate = anchor_tvt + (np.arange(H) - H // 2) * STEP_FT
    target_offset = truth - anchor_tvt
    nearest = np.clip(np.rint((target_offset - (-WINDOW_FT / 2)) / STEP_FT).astype(int), 0, H - 1)
    known_columns = columns_md <= record.anchor_md
    hint = np.zeros((H, W), np.float32)
    hint_drop = float(augment.get("hint_drop", 0.0))
    known_ids = np.flatnonzero(known_columns)
    for col in known_ids:
        if hint_drop and random.random() < hint_drop and col != known_ids[-1]:
            continue
        hint[nearest[col], col] = 1.0
    signed = tw_grid[:, None] - h_grid[None, :]
    cost = np.square(signed, dtype=np.float32)
    dz = np.gradient(z, columns_md, edge_order=1)
    dz = norm(dz)
    dz_grid = np.broadcast_to(dz[None], (H, W)).copy()
    image = np.stack([np.broadcast_to(tw_grid[:, None], (H, W)), np.broadcast_to(h_grid[None], (H, W)), hint, cost, dz_grid], axis=0).astype(np.float32)
    valid = np.isfinite(truth) & (target_offset >= -WINDOW_FT / 2) & (target_offset <= WINDOW_FT / 2)
    loss_mask = valid & ~known_columns
    if flip:
        image = image[..., ::-1].copy()
        image[4] *= -1.0
        target_offset = target_offset[::-1].copy()
        loss_mask = loss_mask[::-1].copy()
        columns_md = columns_md[::-1].copy()
        z = z[::-1].copy()
    return {"image": image, "target": target_offset.astype(np.float32), "mask": loss_mask.astype(np.float32), "md": columns_md, "z": z, "anchor": float(record.anchor_tvt)}


class WellDataset(Dataset):
    def __init__(self, records, training):
        self.records = records
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rng = np.random.default_rng(SEED + self.epoch * 100003 + index)
        if self.training:
            aug = {"md_factor": float(rng.uniform(0.90, 1.10)), "bed_factor": float(rng.uniform(0.90, 1.10)), "bed_bend": float(rng.uniform(-0.5, 0.5)), "hint_drop": float(rng.uniform(0.0, 0.20)), "anchor_jitter": float(rng.uniform(-ANCHOR_JITTER_FT, ANCHOR_JITTER_FT))}
            flip = bool(rng.integers(0, 2))
        else:
            aug, flip = {}, False
        p = panel(self.records[index], aug, flip)
        return {"image": torch.from_numpy(p["image"]), "target": torch.from_numpy(p["target"]), "mask": torch.from_numpy(p["mask"]), "md": torch.from_numpy(p["md"]), "z": torch.from_numpy(p["z"])}


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.GELU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout))
        self.s = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        return F.gelu(self.a(x) + self.s(x))


def layer(cin, cout, n, stride):
    return nn.Sequential(Block(cin, cout, stride), *[Block(cout, cout) for _ in range(n - 1)])


class StackedUNet(nn.Module):
    def __init__(self, cin=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, 64, 7, 2, 3, bias=False), nn.GroupNorm(8, 64), nn.GELU())
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.l1, self.l2, self.l3, self.l4 = layer(64, 64, 3, 1), layer(64, 128, 4, 2), layer(128, 256, 6, 2), layer(256, 512, 3, 2)
        self.d3 = Block(768, 256); self.d2 = Block(384, 128); self.d1 = Block(192, 64); self.d0 = Block(128, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s = self.stem(x); e1 = self.l1(self.pool(s)); e2 = self.l2(e1); e3 = self.l3(e2); e4 = self.l4(e3)
        x = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False); x = self.d3(torch.cat([x, e3], 1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False); x = self.d2(torch.cat([x, e2], 1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False); x = self.d1(torch.cat([x, e1], 1))
        x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False); x = self.d0(torch.cat([x, s], 1))
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(x).squeeze(1)


class SMPUNetAdapter(nn.Module):
    """SMP U-Net wrapper matching the one-channel V242 logits contract."""

    def __init__(self, cin, encoder_name):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "V242_BACKBONE requires segmentation_models_pytorch on Autodl"
            ) from exc
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=os.environ.get("V242_ENCODER_WEIGHTS", "imagenet"),
            in_channels=cin,
            classes=1,
            activation=None,
            decoder_channels=(512, 256, 128, 64, 32),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_model(cin):
    backbone = os.environ.get("V242_BACKBONE", "custom").strip()
    if backbone == "custom":
        return StackedUNet(cin)
    return SMPUNetAdapter(cin, backbone)


def loss_fn(logits, target, mask):
    target_bin = torch.round(target / STEP_FT + H // 2).long().clamp(0, H - 1)
    ce = F.cross_entropy(logits, target_bin, reduction="none")
    offsets = (torch.arange(H, device=logits.device, dtype=torch.float32) - H // 2)[None, :, None] * STEP_FT
    expected = (torch.softmax(logits, 1) * offsets).sum(1)
    return (ce * mask).sum() / mask.sum().clamp_min(1), expected


class EMA:
    def __init__(self, model):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.shadow.parameters(), model.parameters()): a.mul_(EMA_DECAY).add_(b, alpha=1 - EMA_DECAY)


def load_folds():
    path = next((p for p in FOLD_CANDIDATES if p.is_file()), None)
    if path is None: raise FileNotFoundError(FOLD_CANDIDATES)
    return pd.read_csv(path, dtype={"well": str}).groupby("well", as_index=False).fold.first(), path


@torch.no_grad()
def predict(stage1, stage2, record, device, rounds=3):
    p = panel(record)
    base = torch.from_numpy(p["image"])[None].to(device)
    first_logits = stage1(base)
    belief = torch.softmax(first_logits / REFEED_TEMPERATURE, 1)[:, None]
    logits = stage2(torch.cat([base, belief], 1))
    for _ in range(max(0, int(rounds) - 1)):
        belief = torch.softmax(logits / REFEED_TEMPERATURE, 1)[:, None]
        logits = stage2(torch.cat([base, belief], 1))
    offsets = (torch.arange(H, device=device) - H // 2)[None, :, None] * STEP_FT
    expected = (torch.softmax(logits, 1) * offsets).sum(1)[0].cpu().numpy()
    expected = np.asarray(expected, np.float32).reshape(-1)
    md = np.asarray(p["md"], np.float32).reshape(-1)
    z = np.asarray(p["z"], np.float32).reshape(-1)
    if not (len(expected) == len(md) == len(z)):
        raise RuntimeError({"message": "V242 prediction grid shape mismatch", "expected": len(expected), "md": len(md), "z": len(z)})
    rows = np.arange(record.anchor + 1, len(record.md)); surface = record.anchor_tvt + expected + z
    return rows, np.interp(record.md[rows], md, surface) - record.z[rows]


def main():
    if MODE not in {"audit", "smoke", "full"}: raise ValueError(MODE)
    seed_all(SEED); OUT.mkdir(parents=True, exist_ok=True); fold, fold_path = load_folds()
    train = fold.loc[fold.fold.ne(VALID_FOLD), "well"].astype(str).tolist(); valid = fold.loc[fold.fold.eq(VALID_FOLD), "well"].astype(str).tolist()
    rng = np.random.default_rng(SEED)
    if MODE == "audit": train, valid = train[:16], valid[:8]
    if MODE == "smoke": train, valid = sorted(rng.choice(train, 48, False)), sorted(rng.choice(valid, 12, False))
    train_r, valid_r = [load_record(w) for w in train], [load_record(w) for w in valid]
    use_synthetic = os.environ.get("V242_SYNTH", "1") not in {"0", "false", "False"}
    audit = {"experiment": "stacked_unet_real_v242", "fold": VALID_FOLD, "train_wells": len(train_r), "valid_wells": len(valid_r), "input_shape": [5, H, W], "backbone": os.environ.get("V242_BACKBONE", "custom"), "encoder_weights": os.environ.get("V242_ENCODER_WEIGHTS", "imagenet"), "decoder_channels": [512, 256, 128, 64, 32], "loss": "vanilla_hard_cross_entropy", "horizontal_gr": "per_MD_bin_mean", "synthetic_pretraining": use_synthetic, "synthetic_generator": "typewell_template_plus_smooth_surface_noise_and_GR_calibration", "fold_source": str(fold_path)}
    print("V242 CONFIG", json.dumps(audit), flush=True)
    if MODE == "audit": (OUT / "audit.json").write_text(json.dumps(audit, indent=2)); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("V242 requires CUDA")
    m1, m2 = make_model(5).to(device), make_model(6).to(device)
    ema1, ema2 = EMA(m1), EMA(m2); opt = torch.optim.AdamW(list(m1.parameters()) + list(m2.parameters()), lr=2e-4, weight_decay=1e-4); scaler = torch.amp.GradScaler("cuda")
    batch_size = int(os.environ.get("V242_BATCH", "2"))
    accum = int(os.environ.get("V242_ACCUM", "2"))
    epochs = int(os.environ.get("V242_EPOCHS", "4" if MODE == "smoke" else "18"))
    synth_epochs = int(os.environ.get("V242_SYNTH_EPOCHS", "2" if MODE == "smoke" else "8"))
    synth_multiplier = int(os.environ.get("V242_SYNTH_MULTIPLIER", "2"))
    real_ds = WellDataset(train_r, True)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    synth_records = []
    if use_synthetic:
        for source in train_r:
            for replica in range(synth_multiplier):
                synth_records.append(make_synthetic_record(source, replica, SEED))
    synth_ds = WellDataset(synth_records, True) if synth_records else None
    synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) if synth_ds else None
    best = None; history = []; phase_history = []

    def fit_loader(loader, dataset, phase, epoch):
        dataset.set_epoch(epoch)
        m1.train(); m2.train(); opt.zero_grad(set_to_none=True); losses = []
        for step, b in enumerate(loader, 1):
            image, target, mask = b["image"].to(device), b["target"].to(device), b["mask"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                l1 = m1(image)
                belief = torch.softmax(l1 / REFEED_TEMPERATURE, 1)[:, None].detach()
                l2 = m2(torch.cat([image, belief], 1))
                if random.random() < REFEED_TRAIN_PROB:
                    extra_rounds = random.choice((1, 2))
                    for _ in range(extra_rounds):
                        belief = torch.softmax(l2 / REFEED_TEMPERATURE, 1)[:, None].detach()
                        l2 = m2(torch.cat([image, belief], 1))
                loss, _ = loss_fn(l2, target, mask)
                loss = loss + 0.5 * loss_fn(l1, target, mask)[0]
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(loader): scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); ema1.update(m1); ema2.update(m2)
            losses.append(float(loss.detach().cpu()))
        phase_history.append({"phase": phase, "epoch": epoch, "train_loss": float(np.mean(losses)), "batches": len(losses)})

    for epoch in range(1, synth_epochs + 1):
        if synth_loader is not None:
            fit_loader(synth_loader, synth_ds, "synthetic_pretrain", epoch)
            print("V242 SYNTH EPOCH", phase_history[-1], flush=True)

    for epoch in range(1, epochs + 1):
        fit_loader(real_loader, real_ds, "real_finetune", epoch)
        rows, errors = [], []
        for r in valid_r:
            rr, pp = predict(ema1.shadow, ema2.shadow, r, device, 3); finite = np.isfinite(r.tvt[rr]) & np.isfinite(pp); errors.append(pp[finite] - r.tvt[rr][finite]); rows.append({"well": r.well, "rmse": float(np.sqrt(np.mean(errors[-1] ** 2)))})
        err = np.concatenate(errors); metrics = {"epoch": epoch, "train_loss": phase_history[-1]["train_loss"], "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))), "median_well_rmse": float(pd.DataFrame(rows).rmse.median())}; history.append(metrics); print("V242 EPOCH", metrics, flush=True)
        if best is None or metrics["rmse"] < best["rmse"]: best = metrics; torch.save({"stage1": ema1.shadow.state_dict(), "stage2": ema2.shadow.state_dict(), "metrics": metrics}, OUT / f"fold{VALID_FOLD}_seed{SEED}.pt")
    verdict = {**audit, "best": best, "epochs": epochs, "synthetic_epochs": synth_epochs if use_synthetic else 0, "synthetic_wells": len(synth_records), "phase_history": phase_history, "history": history, "next": "Run the same frozen recipe across folds/seeds before blending"}; (OUT / f"fold{VALID_FOLD}_seed{SEED}_verdict.json").write_text(json.dumps(verdict, indent=2)); print("V242 FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__": main()
