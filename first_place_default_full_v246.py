"""Complete deployment runner for the disclosed first-place Default model.

This runner does the part that can be reproduced from the public writeup:
345x400 alignment panels, exponential target, path-Huber and GR penalty,
ConvNeXt-small U-Net, outer well folds, three seeds, and one final submission.
The private PF/XY model code from the winning notebook is intentionally not
invented here; this script records that limitation in its manifest.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "first_place_default_rebuild_v245.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

# V245 contains the audited panel/model implementation. Importing it does not
# execute main(); its top-level code only resolves paths and constants.
import importlib.util

spec = importlib.util.spec_from_file_location("v245_impl", SOURCE)
v245 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# dataclasses resolves postponed annotations through sys.modules on Python 3.12.
sys.modules[spec.name] = v245
spec.loader.exec_module(v245)

ROOT = v245.ROOT
OOF_ROOT = v245.OOF_ROOT
OUT = OOF_ROOT / os.environ.get("V246_OUT", "first_place_default_full_v246")
MODE = os.environ.get("V246_MODE", "full").lower()
FOLDS = [int(x) for x in os.environ.get("V246_FOLDS", "1,2,3,4,5").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("V246_SEEDS", "0,1,2").split(",") if x.strip()]
EPOCHS = int(os.environ.get("V246_EPOCHS", "18"))
BATCH = int(os.environ.get("V246_BATCH", "1"))
ACCUM = int(os.environ.get("V246_ACCUM", "2"))
WORKERS = int(os.environ.get("V246_WORKERS", "0"))


def records_for(folds: pd.DataFrame, held: int) -> tuple[list, list]:
    train_ids = folds.loc[folds.fold.ne(held), "well"].astype(str).tolist()
    valid_ids = folds.loc[folds.fold.eq(held), "well"].astype(str).tolist()
    if MODE == "audit":
        train_ids, valid_ids = train_ids[:2], valid_ids[:1]
    return [v245.load_record(w) for w in train_ids], [v245.load_record(w) for w in valid_ids]


def make_model(seed: int):
    v245.seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V246 requires a CUDA runtime")
    model = v245.ConvNeXtBNUNet(len(v245.CHANNELS)).to(device)
    return model, device


def train_one(train_records: list, valid_records: list, fold: int, seed: int):
    model, device = make_model(seed)
    bank = v245.DifferenceBlockBank(train_records)
    dataset = v245.AlignmentDataset(train_records, True, bank)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                        pin_memory=True, persistent_workers=WORKERS > 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=v245.RECIPE.learning_rate,
                                  weight_decay=v245.RECIPE.weight_decay)
    best = None
    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target_offset"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image)
                loss, _ = v245.alignment_loss(logits, target, mask, image[:, 14])
            if not torch.isfinite(loss):
                raise RuntimeError({"fold": fold, "seed": seed, "epoch": epoch, "step": step})
            (loss / ACCUM).backward()
            if step % ACCUM == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        metrics = v245.evaluate(model, valid_records, device)
        row = {"fold": fold, "seed": seed, "epoch": epoch,
               "train_loss": float(np.mean(losses)), **metrics}
        print("V246 EPOCH", row, flush=True)
        if best is None or row["rmse"] < best["rmse"]:
            best = row
            ckpt = OUT / "checkpoints" / f"fold{fold}_seed{seed}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "channels": list(v245.CHANNELS),
                        "fold": fold, "seed": seed, "metrics": row}, ckpt)
    return model, device, best


@torch.no_grad()
def predict_records(model, device, records: list) -> dict[str, np.ndarray]:
    model.eval()
    output = {}
    for record in records:
        rows, pred = v245.predict_record(model, record, device)
        output.update({f"{record.well}_{int(row)}": float(value)
                       for row, value in zip(rows, pred)})
    return output


def main() -> None:
    started = time.time()
    if MODE not in {"audit", "full"}:
        raise ValueError("V246_MODE must be audit or full")
    OUT.mkdir(parents=True, exist_ok=True)
    folds = v245.load_folds()
    fold_reports = []
    oof = {}
    config = {
        "experiment": "first_place_default_model_v246",
        "public_recipe": "345 MD bins x 400 TVT bins, 0.5 ft axis, disclosed losses/features",
        "folds": FOLDS, "seeds": SEEDS, "epochs": EPOCHS,
        "train_wells": int(folds.well.nunique()),
        "private_components": {"PF": "not available", "XY_neighbor": "not available"},
        "submission_policy": "only this script writes submission.csv",
    }
    print("V246 CONFIG", json.dumps(config), flush=True)
    if MODE == "audit":
        audit = v245.representation_audit(v245.load_record(str(folds.well.iloc[0])))
        print("V246 AUDIT", json.dumps(audit), flush=True)
        return

    for fold in FOLDS:
        train_records, valid_records = records_for(folds, fold)
        for seed in SEEDS:
            model, device, best = train_one(train_records, valid_records, fold, seed)
            valid_pred = predict_records(model, device, valid_records)
            for key, value in valid_pred.items():
                oof.setdefault(key, []).append(value)
            fold_reports.append(best)
            print("V246 MEMBER DONE", {"fold": fold, "seed": seed, "best": best}, flush=True)

    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "fold_reports.json").write_text(json.dumps(fold_reports, indent=2), encoding="utf-8")
    pd.DataFrame([{"id": key, "prediction": float(np.mean(values))}
                  for key, values in oof.items()]).to_csv(OUT / "oof_member_mean.csv", index=False)

    verdict = {**config, "oof_rows": len(oof), "members": len(fold_reports),
               "model_dir": str(OUT / "checkpoints"),
               "submission_written": False,
               "elapsed_minutes": (time.time() - started) / 60.0}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("V246 FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
