from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import get_worker_info

from .config import ReproductionConfig, VERSION_RECIPES
from .data import load_record
from .dataset import AlignmentDataset
from .folds import build_well_metadata, make_repeated_folds, validate_fold_contract
from .loss import AlignmentLoss
from .model import ARCHITECTURE, CONTRACT_VERSION, AlignmentUNet, ModelEMA
from .normalization import fit_population_stats, read_population_stats, write_population_stats
from .xy import XYSurfacePrior


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    info = get_worker_info()
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(worker_seed)


def source_member_seeds(base_seed: int, repeat: int, fold: int) -> tuple[int, int, int, int]:
    """Return source global-fold, model, train-loader, and validation-loader seeds."""
    global_fold = int(repeat) * 5 + (int(fold) - 1)
    return (
        global_fold,
        int(base_seed) + global_fold,
        int(base_seed) + global_fold * 1000 + 1,
        int(base_seed) + global_fold * 1000 + 2,
    )


def decode_row_offsets(offsets: np.ndarray, z: np.ndarray, anchor_tvt: float,
                       window: int, blend: float, apply_smooth: bool = False) -> np.ndarray:
    """Expand fixed-window bins; checkpoint selection uses unsmoothed suffix output."""
    prediction = np.full(len(z), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(z))
    if not len(valid):
        return prediction
    expanded = anchor_tvt + np.repeat(np.asarray(offsets, dtype=np.float64), 32)
    trajectory = slice(int(valid[0]), int(valid[-1]) + 1)
    local_tvt = expanded[trajectory].copy()
    local_z = np.asarray(z[trajectory], dtype=np.float64)
    local_valid = np.isfinite(local_z)
    if not local_valid.all():
        positions = np.arange(len(local_z))
        local_z = np.interp(positions, positions[local_valid], local_z[local_valid])
    if apply_smooth and len(local_tvt) >= 3:
        actual_window = min(window, len(local_tvt) if len(local_tvt) % 2 else len(local_tvt) - 1)
        actual_window = max(3, actual_window)
        surface = local_tvt + local_z
        smoothed_tvt = savgol_filter(surface, actual_window, 1, mode="interp") - local_z
        local_tvt = blend * smoothed_tvt + (1.0 - blend) * local_tvt
    prediction[trajectory] = local_tvt
    return prediction


@torch.no_grad()
def evaluate(model, loader, criterion, device, recipe) -> dict[str, float]:
    model.eval()
    squared, absolute, count, losses = 0.0, 0.0, 0, []
    row_squared, row_absolute, row_count = 0.0, 0.0, 0
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
        for index in range(len(prediction)):
            offsets = prediction[index].float().cpu().numpy()
            truth = batch["row_truth"][index].numpy()
            z = batch["row_z"][index].numpy()
            row_mask = batch["row_eval_mask"][index].numpy().astype(bool)
            row_mask &= np.isfinite(truth) & np.isfinite(z)
            tvt = decode_row_offsets(
                offsets, z, float(batch["anchor_tvt"][index]),
                recipe.sg_window, recipe.sg_blend, apply_smooth=False,
            )
            if row_mask.any():
                row_error = tvt[row_mask] - truth[row_mask]
                row_squared += float(row_error @ row_error)
                row_absolute += float(np.abs(row_error).sum())
                row_count += int(row_mask.sum())
    return {
        "loss": float(np.mean(losses)),
        "bin_rmse": math.sqrt(squared / max(count, 1)),
        "bin_mae": absolute / max(count, 1),
        "bins": count,
        "row_rmse": math.sqrt(row_squared / max(row_count, 1)),
        "row_mae": row_absolute / max(row_count, 1),
        "rows": row_count,
    }


def train_member(args) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal training requires a CUDA device with BF16 support")
    if args.batch_size != 16:
        raise ValueError("the disclosed training batch size is 16; use fewer wells, not a smaller batch")
    if args.version == "0719_V1" and args.seed != 7:
        raise ValueError("0719_V1 source contract requires base seed 7")
    if args.version == "0719_V1" and args.disable_augmentation and args.overfit_wells is None:
        raise ValueError(
            "0719_V1 formal training requires source augmentation; "
            "--disable-augmentation is only allowed with --overfit-wells"
        )
    config = ReproductionConfig()
    recipe = VERSION_RECIPES[args.version]
    # Source folds are enumerated repeat-major with zero-based fold indices.
    # cfg.seed is 7; each of the 15 members receives cfg.seed + global_fold.
    global_fold, model_seed, train_loader_seed, valid_loader_seed = source_member_seeds(
        args.seed, args.repeat, args.fold
    )
    seed_everything(model_seed, deterministic=False)
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
    validate_fold_contract(folds)
    print("TRAIN CONTRACT", json.dumps({
        "contract_version": CONTRACT_VERSION,
        "fold_contract": str(folds["fold_contract"].iloc[0]),
        "repeat": args.repeat,
        "fold_one_based": args.fold,
        "global_fold_zero_based": global_fold,
        "model_seed": model_seed,
        "train_loader_seed": train_loader_seed,
        "valid_loader_seed": valid_loader_seed,
        "augmentation": not (args.overfit_wells is not None or args.disable_augmentation),
        "activation_offload": args.activation_offload,
        "train_batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
    }), flush=True)
    column = f"repeat_{args.repeat}"
    train_wells = folds.loc[folds[column] != args.fold, "well"].tolist()
    valid_wells = folds.loc[folds[column] == args.fold, "well"].tolist()
    overfit = args.overfit_wells is not None
    if overfit:
        train_wells = train_wells[:args.overfit_wells]
        valid_wells = list(train_wells)
    if args.limit_train:
        train_wells = train_wells[: args.limit_train]
    if args.limit_valid:
        valid_wells = valid_wells[: args.limit_valid]
    stats_path = Path(args.output_root) / "population_stats.json"
    if stats_path.is_file():
        population_stats = read_population_stats(stats_path)
    else:
        # The writeup only states population statistics; it does not disclose
        # outer-fold fitting. Keep the reconstruction's original all-train
        # interpretation instead of silently changing the reference contract.
        population_stats = fit_population_stats(args.data_root, folds.well.tolist())
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        write_population_stats(stats_path, population_stats)
    xy_prior = None
    if recipe.use_xy:
        xy_prior = XYSurfacePrior(config.xy).fit_records(
            [load_record(args.data_root, "train", well) for well in train_wells]
        )
    train_data = AlignmentDataset(
        args.data_root, "train", train_wells, recipe, config, xy_prior,
        not (overfit or args.disable_augmentation), train_loader_seed,
        args.cache_root, population_stats
    )
    valid_data = AlignmentDataset(
        args.data_root, "train", valid_wells, recipe, config, xy_prior, False, valid_loader_seed,
        args.cache_root, population_stats
    )
    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    loader_generator = torch.Generator()
    loader_generator.manual_seed(train_loader_seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.validation_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    first = train_data[0]
    model = AlignmentUNet(
        len(first["values"]),
        pretrained=not args.no_pretrained,
        encoder_checkpoint=args.encoder_checkpoint,
        dropout=recipe.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, memory_format=torch.channels_last)
    ema = ModelEMA(model, config.train.ema_decay)
    ema.module.to(device)
    criterion = AlignmentLoss(config.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    epochs = args.epochs or recipe.epochs
    schedule_epochs = args.schedule_epochs or recipe.epochs
    if schedule_epochs < epochs:
        raise ValueError("schedule epochs cannot be shorter than run epochs")
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=schedule_epochs * optimizer_steps_per_epoch,
        eta_min=config.train.min_learning_rate,
    )
    best, best_epoch, history = float("inf"), 0, []
    start_epoch = 1
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("contract_version") != CONTRACT_VERSION:
            raise RuntimeError({
                "message": "resume checkpoint predates the corrected 0719 contract",
                "expected": CONTRACT_VERSION,
                "found": resume.get("contract_version"),
            })
        model.load_state_dict(resume["raw_model"], strict=True)
        ema.module.load_state_dict(resume["ema_model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        best = float(resume["best"])
        best_epoch = int(resume["best_epoch"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = []
        for step, batch in enumerate(train_loader, 1):
            values = batch["values"].to(device, non_blocking=True, memory_format=torch.channels_last)
            activation_context = (
                # Some PyTorch/CUDA builds reject pinned saved-tensor buffers
                # for channels-last BF16 activations. Pageable CPU offload has
                # identical numerical semantics and a lower compatibility risk.
                torch.autograd.graph.save_on_cpu(pin_memory=False)
                if args.activation_offload
                else nullcontext()
            )
            with activation_context:
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
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            running.append(float(loss.detach()) * args.accumulation_steps)
        # The source logs and validates epoch 1, then every log_freq epochs.
        should_validate = epoch == 1 or epoch % 5 == 0 or epoch == epochs
        if not should_validate:
            print("TRAIN EPOCH", json.dumps({
                "epoch": epoch,
                "train_loss": float(np.mean(running)),
                "lr": scheduler.get_last_lr()[0],
            }), flush=True)
            continue
        raw_metrics = evaluate(model, valid_loader, criterion, device, recipe)
        ema_metrics = evaluate(ema.module, valid_loader, criterion, device, recipe)
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
            "global_fold": global_fold,
            "model_seed": model_seed,
            "train_loader_seed": train_loader_seed,
            "valid_loader_seed": valid_loader_seed,
            "run_id": run_id,
            "epoch": epoch,
            "metrics": ema_metrics,
            "raw_metrics": raw_metrics,
            "architecture": ARCHITECTURE,
            "contract_version": CONTRACT_VERSION,
            "schedule_epochs": schedule_epochs,
            "population_stats": population_stats.to_dict(),
        }
        selection_model = model if overfit else ema.module
        selection_metrics = raw_metrics if overfit else ema_metrics
        best_payload["model"] = selection_model.state_dict()
        best_payload["metrics"] = selection_metrics
        best_payload["selected_model"] = "raw" if overfit else "ema"
        if selection_metrics["row_rmse"] < best:
            best, best_epoch = selection_metrics["row_rmse"], epoch
            torch.save(
                best_payload,
                output / "best.pt",
            )
        if epoch % args.save_every == 0 or epoch == epochs:
            torch.save(
                {
                    "raw_model": model.state_dict(),
                    "ema_model": ema.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "best_epoch": best_epoch,
                    "history": history,
                    "run_id": run_id,
                    "architecture": ARCHITECTURE,
                    "contract_version": CONTRACT_VERSION,
                    "schedule_epochs": schedule_epochs,
                },
                output / "last.pt",
            )
        # Source patience is measured in epochs since the best checkpoint, not
        # in validation events. With validation every five epochs these differ
        # by a factor of five.
        if (
            epoch >= config.train.minimum_epochs
            and epoch - best_epoch >= args.patience
        ):
            break
    verdict = {
        "level": "partial clean-room reconstruction; metric parity not verified",
        "version": args.version,
        "repeat": args.repeat,
        "fold": args.fold,
        "seed": args.seed,
        "global_fold": global_fold,
        "model_seed": model_seed,
        "train_loader_seed": train_loader_seed,
        "valid_loader_seed": valid_loader_seed,
        "run_id": run_id,
        "architecture": ARCHITECTURE,
        "schedule_epochs": schedule_epochs,
        "activation_offload": args.activation_offload,
        "validation_batch_size": args.validation_batch_size,
        "overfit": overfit,
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "best_row_rmse": best,
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
    result.add_argument("--seed", type=int, default=7,
                        help="disclosed cfg.seed; 0719 uses 7")
    result.add_argument("--epochs", type=int)
    result.add_argument(
        "--schedule-epochs",
        type=int,
        help="cosine schedule horizon; defaults to the disclosed version epoch count",
    )
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--validation-batch-size", type=int, default=32)
    result.add_argument("--accumulation-steps", type=int, default=1)
    result.add_argument(
        "--activation-offload",
        action="store_true",
        help="offload saved autograd tensors to CPU while preserving the disclosed batch size",
    )
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--patience", type=int, default=30)
    result.add_argument("--limit-train", type=int)
    result.add_argument("--limit-valid", type=int)
    result.add_argument("--overfit-wells", type=int)
    result.add_argument("--disable-augmentation", action="store_true")
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
