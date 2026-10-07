from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import get_worker_info

from .config import ReproductionConfig, VERSION_RECIPES
from .data import load_record
from .dataset import AlignmentDataset
from .folds import build_well_metadata, make_repeated_folds
from .loss import AlignmentLoss
from .model import ARCHITECTURE, AlignmentUNet, ModelEMA
from .xy import XYSurfacePrior


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(torch.initial_seed() % (2**32))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    squared, absolute, count, losses = 0.0, 0.0, 0, []
    for batch in loader:
        values = batch["values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(values)
        state = batch["state_offsets"].to(device)
        loss, parts = criterion(
            logits,
            batch["target_probability"].to(device),
            batch["target_offset"].to(device),
            batch["target_mask"].to(device),
            state,
            batch["gr_gap"].to(device),
        )
        prediction = parts["expected"]
        mask = batch["target_mask"].to(device).bool()
        error = prediction[mask] - batch["target_offset"].to(device)[mask]
        squared += float((error**2).sum())
        absolute += float(error.abs().sum())
        count += int(mask.sum())
        losses.append(float(loss))
    return {
        "loss": float(np.mean(losses)),
        "bin_rmse": math.sqrt(squared / max(count, 1)),
        "bin_mae": absolute / max(count, 1),
        "bins": count,
    }


def train_member(args) -> dict:
    config = ReproductionConfig()
    recipe = VERSION_RECIPES[args.version]
    seed = args.seed + args.repeat * 10_000 + args.fold * 100
    seed_everything(seed)
    run_id = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / args.version / f"repeat{args.repeat}_fold{args.fold}_seed{args.seed}_{run_id}"
    output.mkdir(parents=True, exist_ok=True)
    fold_path = Path(args.fold_file) if args.fold_file else Path(args.output_root) / "folds.csv"
    if fold_path.is_file():
        folds = pd.read_csv(fold_path, dtype={"well": str})
    else:
        folds = make_repeated_folds(build_well_metadata(args.data_root))
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(fold_path, index=False)
    column = f"repeat_{args.repeat}"
    train_wells = folds.loc[folds[column] != args.fold, "well"].tolist()
    valid_wells = folds.loc[folds[column] == args.fold, "well"].tolist()
    if args.limit_train:
        train_wells = train_wells[: args.limit_train]
    if args.limit_valid:
        valid_wells = valid_wells[: args.limit_valid]
    xy_prior = None
    if recipe.use_xy:
        xy_prior = XYSurfacePrior(config.xy).fit_records(
            [load_record(args.data_root, "train", well) for well in train_wells]
        )
    train_data = AlignmentDataset(
        args.data_root, "train", train_wells, recipe, config, xy_prior, True, seed, args.cache_root
    )
    valid_data = AlignmentDataset(
        args.data_root, "train", valid_wells, recipe, config, xy_prior, False, seed, args.cache_root
    )
    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_data, batch_size=max(1, args.batch_size), shuffle=False, **loader_kwargs)
    first = train_data[0]
    model = AlignmentUNet(
        len(first["values"]),
        pretrained=not args.no_pretrained,
        encoder_checkpoint=args.encoder_checkpoint,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, memory_format=torch.channels_last)
    ema = ModelEMA(model, config.train.ema_decay)
    ema.module.to(device)
    criterion = AlignmentLoss(config.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=config.train.min_learning_rate
    )
    best, stale, history = float("inf"), 0, []
    start_epoch = 1
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["raw_model"], strict=True)
        ema.module.load_state_dict(resume["ema_model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        best = float(resume["best"])
        stale = int(resume["stale"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = []
        for step, batch in enumerate(train_loader, 1):
            values = batch["values"].to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(values)
                loss, _ = criterion(
                    logits,
                    batch["target_probability"].to(device),
                    batch["target_offset"].to(device),
                    batch["target_mask"].to(device),
                    batch["state_offsets"].to(device),
                    batch["gr_gap"].to(device),
                )
                loss = loss / args.accumulation_steps
            loss.backward()
            if step % args.accumulation_steps == 0 or step == len(train_loader):
                clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            running.append(float(loss.detach()) * args.accumulation_steps)
        scheduler.step()
        raw_metrics = evaluate(model, valid_loader, criterion, device)
        ema_metrics = evaluate(ema.module, valid_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "lr": scheduler.get_last_lr()[0],
            "raw": raw_metrics,
            "ema": ema_metrics,
        }
        history.append(row)
        print("EPOCH", json.dumps(row), flush=True)
        best_payload = {
            "model": ema.module.state_dict(),
            "channels": first["values"].shape[0],
            "version": args.version,
            "repeat": args.repeat,
            "fold": args.fold,
            "seed": args.seed,
            "run_id": run_id,
            "epoch": epoch,
            "metrics": ema_metrics,
            "raw_metrics": raw_metrics,
            "architecture": ARCHITECTURE,
        }
        if ema_metrics["bin_rmse"] < best:
            best, stale = ema_metrics["bin_rmse"], 0
            torch.save(
                best_payload,
                output / "best.pt",
            )
        else:
            stale += 1
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "raw_model": model.state_dict(),
                    "ema_model": ema.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "stale": stale,
                    "history": history,
                    "run_id": run_id,
                    "architecture": ARCHITECTURE,
                },
                output / "last.pt",
            )
        if stale >= args.patience:
            break
    verdict = {
        "level": "behavioral reproduction from writeup only",
        "version": args.version,
        "repeat": args.repeat,
        "fold": args.fold,
        "seed": args.seed,
        "run_id": run_id,
        "architecture": ARCHITECTURE,
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "best_bin_rmse": best,
        "history": history,
    }
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data-root", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--cache-root")
    result.add_argument("--fold-file")
    result.add_argument("--version", choices=VERSION_RECIPES, required=True)
    result.add_argument("--repeat", type=int, choices=(0, 1, 2), required=True)
    result.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5), required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=220)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--accumulation-steps", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--patience", type=int, default=35)
    result.add_argument("--limit-train", type=int)
    result.add_argument("--limit-valid", type=int)
    result.add_argument("--no-pretrained", action="store_true")
    result.add_argument("--encoder-checkpoint")
    result.add_argument("--run-tag")
    result.add_argument("--resume")
    result.add_argument("--save-every", type=int, default=5)
    return result


def main() -> None:
    train_member(parser().parse_args())


if __name__ == "__main__":
    main()
