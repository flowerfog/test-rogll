from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import SEEDS, TRAIN, VERSION_BY_NAME
from .data import AlignmentDataset
from .loss import decode_expected, weighted_huber_ce
from .model import AlignmentUNet, load_encoder_checkpoint
from .infer_utils import smooth_raw


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_auxiliary(cache_root: Path, split: str, wells: list[str], channels: tuple[str, ...],
                   outer_fold: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    needs_pf = any(name.startswith("pf_") for name in channels)
    needs_xy = any(name.startswith("geo_") for name in channels)
    result: dict[str, dict[str, np.ndarray]] = {}
    for well in wells:
        values: dict[str, np.ndarray] = {}
        for component, needed in (("pf", needs_pf), ("xy", needs_xy)):
            if not needed:
                continue
            folder = cache_root / component / split
            if component == "xy" and split == "train":
                if outer_fold is None:
                    raise ValueError("Fold-clean train XY features require outer_fold")
                folder = folder / f"fold{outer_fold}"
            path = folder / f"{well}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as source:
                values.update({key: source[key] for key in source.files if not key.startswith("stat__")})
        result[well] = values
    return result


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.mul_(self.decay).add_(current.detach(), alpha=1 - self.decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             sg_window: int, sg_blend: float) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    squared, absolute, rows, nonfinite = 0.0, 0.0, 0, 0
    predictions: dict[str, np.ndarray] = {}
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        _, expected = decode_expected(logits, batch["state_tvt"].to(device))
        for index, well in enumerate(batch["well"]):
            truth = batch["raw_truth_tvt"][index].numpy()
            mask = batch["raw_eval_mask"][index].numpy().astype(bool)
            binned = expected[index].float().cpu().numpy()
            nonfinite += int((~np.isfinite(binned)).sum())
            centers = np.minimum(np.arange(345) * 32 + 31, 11023)
            prediction = np.interp(np.arange(len(truth)), centers, binned)
            prediction = smooth_raw(prediction, sg_window, sg_blend)
            error = prediction[mask] - truth[mask]
            squared += float(error @ error)
            absolute += float(np.abs(error).sum())
            rows += int(mask.sum())
            predictions[well] = prediction.astype(np.float32)
    return {"rows": rows, "rmse": math.sqrt(squared / max(rows, 1)), "mae": absolute / max(rows, 1),
            "nonfinite_prediction_bins": nonfinite}, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--version", choices=tuple(VERSION_BY_NAME), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, help="Program-check override; model/batch semantics remain unchanged.")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--overfit-wells", type=int,
                        help="Use the same wells for train/validation and disable augmentation.")
    parser.add_argument("--disable-augmentation", action="store_true",
                        help="Keep the honest fold split but disable all training simulation/augmentation.")
    parser.add_argument("--network-layout", choices=("state_horizontal", "horizontal_state"),
                        default="state_horizontal",
                        help="Spatial layout presented to the encoder; output is always state x horizontal.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA BF16; CPU fallback would change disclosed behavior")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The disclosed BN ConvNeXt training requires BF16 support")
    seed_all(args.seed)
    version = VERSION_BY_NAME[args.version]
    folds = pd.read_csv(args.cache_root / "folds.csv", dtype={"well": str})
    train_wells = folds.loc[folds.fold.ne(args.fold), "well"].tolist()
    valid_wells = folds.loc[folds.fold.eq(args.fold), "well"].tolist()
    overfit = args.overfit_wells is not None
    disable_augmentation = overfit or args.disable_augmentation
    if overfit:
        train_wells = train_wells[:args.overfit_wells]
        valid_wells = list(train_wells)
    if args.limit_train:
        train_wells = train_wells[:args.limit_train]
    if args.limit_valid:
        valid_wells = valid_wells[:args.limit_valid]
    auxiliary = load_auxiliary(args.cache_root, "train", train_wells + valid_wells,
                               version.channels, outer_fold=args.fold)
    train_set = AlignmentDataset(args.data_root, train_wells, "train", version.channels, auxiliary,
                                 training=not disable_augmentation, seed=args.seed,
                                 z_shift_prob=0.0 if disable_augmentation else version.z_shift_prob,
                                 version_name=version.name)
    valid_set = AlignmentDataset(args.data_root, valid_wells, "train", version.channels, auxiliary,
                                 version_name=version.name)
    loader_args = dict(num_workers=TRAIN.workers, pin_memory=True, persistent_workers=True, prefetch_factor=1)
    train_loader = DataLoader(train_set, batch_size=TRAIN.batch_size, shuffle=True, drop_last=True, **loader_args)
    valid_loader = DataLoader(valid_set, batch_size=version.validation_batch_size,
                              shuffle=False, **loader_args)
    if len(train_wells) < TRAIN.batch_size:
        raise ValueError("Program-check still requires at least 16 training wells to preserve BatchNorm semantics")
    model = AlignmentUNet(len(version.channels), version.dropout, layout=args.network_layout).cuda()
    checkpoint_report = None
    if args.encoder_checkpoint:
        raise ValueError(
            "Released configuration declares unet_pretrained=None; "
            "--encoder-checkpoint is not valid for the faithful run"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.learning_rate, weight_decay=TRAIN.weight_decay,
                                  betas=(.9, .999), eps=1e-8)
    epochs = args.epochs or version.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=TRAIN.min_learning_rate)
    ema = EMA(model, TRAIN.ema_decay)
    scaler = None
    output = args.output_root / version.name / f"fold{args.fold}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = {"version": version.name, "fold": args.fold, "seed": args.seed, "epochs": epochs,
              "batch_size": TRAIN.batch_size, "validation_batch_size": version.validation_batch_size,
              "channels": version.channels, "checkpoint": checkpoint_report,
              "train_wells": len(train_wells), "valid_wells": len(valid_wells), "overfit": overfit,
              "augmentation_disabled": disable_augmentation, "network_layout": args.network_layout}
    print("TRAIN CONFIG", json.dumps(config), flush=True)
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].float().cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss, _ = weighted_huber_ce(logits, batch["target"].cuda(non_blocking=True),
                                             batch["mask"].cuda(non_blocking=True),
                                             batch["state_tvt"].cuda(non_blocking=True),
                                             batch["truth_tvt"].cuda(non_blocking=True),
                                             batch["gr_gap"].cuda(non_blocking=True))
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError({
                    "message": "nonfinite training forward/loss",
                    "epoch": epoch,
                    "input_min": float(x.nan_to_num().min()),
                    "input_max": float(x.nan_to_num().max()),
                    "logits_finite": bool(torch.isfinite(logits).all()),
                    "loss": float(loss.detach().nan_to_num()),
                })
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ema.update(model)
            losses.append(float(loss.detach()))
        scheduler.step()
        evaluation_model = model if overfit else ema.model
        metrics, predictions = evaluate(evaluation_model, valid_loader, torch.device("cuda"),
                                        version.sg_window, version.sg_blend)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": scheduler.get_last_lr()[0], **metrics}
        history.append(row)
        print("EPOCH", row, flush=True)
        if metrics["rmse"] < best:
            best = metrics["rmse"]
            torch.save({"model": evaluation_model.state_dict(), "config": config, "metrics": metrics}, output / "best.pt")
            np.savez_compressed(output / "valid_predictions.npz", **predictions)
    (output / "verdict.json").write_text(json.dumps({"config": config, "best_rmse": best, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
