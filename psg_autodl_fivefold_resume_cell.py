# Direct AutoDL resume cell.
# Run in the SAME kernel after the previous cell stopped with KeyError: 'TH'.
# It reuses only loaded raw wells and definitions; training starts from scratch.

import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REQUIRED = {
    "CACHE", "train_ids", "build_cloud", "struct_curve", "build_features",
    "PSG", "well_forward", "SCALES", "GRID", "N_FEAT", "FEATS", "DEVICE",
}
missing = sorted(REQUIRED.difference(globals()))
if missing:
    raise RuntimeError(
        f"Missing live definitions: {missing}. Run the PSG definition/data cell first."
    )
if int(N_FEAT) != 26:
    raise RuntimeError(
        f"Expected the inference-compatible 26-channel feature contract, got {N_FEAT}: {FEATS}"
    )
if DEVICE == "cpu":
    raise RuntimeError("Five-fold PSG training requires an AutoDL CUDA instance")

ARTIFACT_DIR = Path("/root/autodl-tmp/psg_5fold_dataset")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SEED_5F = 42
EPOCHS_5F = int(os.environ.get("PSG_EPOCHS", "45"))
PATIENCE_5F = int(os.environ.get("PSG_PATIENCE", "10"))
ACC_5F = 6
fivefold_started = time.time()


def _pseudo_ps_5f(well, rng):
    return int(rng.uniform(0.20, 0.45) * well["n"])


def _make_folds_5f(ids):
    ordered = np.asarray(sorted(ids), dtype=object)
    rng = np.random.RandomState(SEED_5F)
    ordered = ordered[rng.permutation(len(ordered))]
    result = {}
    for fold, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        for index in indices:
            result[str(ordered[index])] = fold
    return result


FOLD_BY_WELL_5F = _make_folds_5f(train_ids)
fold_table_5f = pd.DataFrame(
    {
        "well": sorted(train_ids),
        "fold": [FOLD_BY_WELL_5F[well] for well in sorted(train_ids)],
    }
)
fold_table_5f.to_csv(ARTIFACT_DIR / "psg_folds.csv", index=False)


def _prepare_structure_5f(fit_ids, valid_ids):
    cloud = build_cloud(fit_ids)
    for number, well_id in enumerate(fit_ids, start=1):
        well = CACHE[well_id]
        well["TH"], well["THd"], well["THn"], well["THdp"] = struct_curve(
            cloud, well, cloud["wid2idx"][well_id]
        )
        if number % 100 == 0 or number == len(fit_ids):
            print(f"structure train {number}/{len(fit_ids)}", flush=True)
    for number, well_id in enumerate(valid_ids, start=1):
        well = CACHE[well_id]
        well["TH"], well["THd"], well["THn"], well["THdp"] = struct_curve(
            cloud, well, None
        )
        if number % 50 == 0 or number == len(valid_ids):
            print(f"structure valid {number}/{len(valid_ids)}", flush=True)
    return cloud


def _evaluate_5f(ids):
    MODEL.eval()
    squared = 0.0
    absolute = 0.0
    rows_total = 0
    wells_total = 0
    with torch.no_grad():
        for well_id in ids:
            well = CACHE[well_id]
            ps = int(well["real_ps"])
            if ps >= well["n"] or ps < 300:
                continue
            features, _ = build_features(well, ps)
            prediction, rows, *_ = well_forward(well, ps, features)
            error = prediction.numpy() - well["TVT"][rows]
            error = error[np.isfinite(error)]
            squared += float(np.sum(error**2))
            absolute += float(np.sum(np.abs(error)))
            rows_total += len(error)
            wells_total += 1
    return {
        "rows": rows_total,
        "wells": wells_total,
        "rmse": float(np.sqrt(squared / max(rows_total, 1))),
        "mae": float(absolute / max(rows_total, 1)),
    }


def _train_fold_5f(fold, fit_ids, valid_ids):
    global MODEL
    fold_seed = SEED_5F + 1000 * fold
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)
    rng = np.random.RandomState(fold_seed)

    MODEL = PSG(N_FEAT).to(DEVICE)
    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_5F)
    best_rmse = np.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS_5F + 1):
        MODEL.train()
        order = list(fit_ids)
        rng.shuffle(order)
        optimizer.zero_grad()
        pending = 0
        train_loss_sum = 0.0
        wells_used = 0

        for number, well_id in enumerate(order, start=1):
            well = CACHE[well_id]
            ps = _pseudo_ps_5f(well, rng)
            if ps >= well["n"] - 100 or ps < 300:
                continue
            features, _ = build_features(well, ps)
            prediction, rows, edge_prediction, edge_truth, edge_scale = well_forward(
                well, ps, features
            )
            truth = torch.tensor(well["TVT"][rows])
            finite = torch.isfinite(truth)
            if not finite.any():
                continue
            path_loss = ((prediction - truth)[finite] ** 2).mean()
            edge_loss = torch.tensor(0.0, dtype=torch.float64)
            edge_scale_tensor = torch.tensor(edge_scale)
            for scale in SCALES:
                selected = edge_scale_tensor == scale
                if selected.any():
                    residual = (
                        edge_prediction[selected] - edge_truth[selected]
                    ) / (scale * GRID)
                    residual = residual[torch.isfinite(residual)]
                    if len(residual):
                        edge_loss = edge_loss + (residual**2).mean()
            loss = 0.7 * path_loss / 400.0 + 0.3 * edge_loss * 100.0
            (loss / ACC_5F).backward()
            pending += 1
            wells_used += 1
            train_loss_sum += float(loss.detach())

            if pending == ACC_5F:
                nn.utils.clip_grad_norm_(MODEL.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad()
                pending = 0
            if number % 100 == 0:
                print(
                    f"fold {fold} epoch {epoch} train {number}/{len(order)}",
                    flush=True,
                )

        if pending:
            correction = ACC_5F / pending
            for parameter in MODEL.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            nn.utils.clip_grad_norm_(MODEL.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        valid_metrics = _evaluate_5f(valid_ids)
        report = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss_sum / max(wells_used, 1),
            "train_wells": wells_used,
            "valid_rows": valid_metrics["rows"],
            "valid_wells": valid_metrics["wells"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_mae": valid_metrics["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - fivefold_started) / 60.0,
        }
        history.append(report)
        print("PSG FOLD EPOCH", report, flush=True)

        if valid_metrics["rmse"] < best_rmse:
            best_rmse = valid_metrics["rmse"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in MODEL.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE_5F:
                print(f"fold {fold} early stopping", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"fold {fold} produced no valid checkpoint")
    MODEL.load_state_dict(best_state)
    checkpoint_path = ARTIFACT_DIR / f"psg_oof_fold{fold}.pt"
    torch.save(best_state, checkpoint_path)
    pd.DataFrame(history).to_csv(
        ARTIFACT_DIR / f"psg_fold{fold}_training_history.csv", index=False
    )
    return checkpoint_path, best_epoch, best_rmse


def _predict_oof_5f(fold, valid_ids):
    output = []
    MODEL.eval()
    with torch.no_grad():
        for number, well_id in enumerate(valid_ids, start=1):
            well = CACHE[well_id]
            ps = int(well["real_ps"])
            if ps >= well["n"]:
                continue
            features, _ = build_features(well, ps)
            prediction, rows, *_ = well_forward(well, ps, features)
            values = prediction.numpy()
            truth = well["TVT"][rows]
            finite = np.isfinite(values) & np.isfinite(truth)
            for row_idx, predicted_tvt, true_tvt in zip(
                rows[finite], values[finite], truth[finite]
            ):
                output.append(
                    {
                        "well": well_id,
                        "row_idx": int(row_idx),
                        "md_since": float(well["MD"][row_idx] - well["MD"][ps]),
                        "psg_tvt": float(predicted_tvt),
                        "true_tvt": float(true_tvt),
                        "fold": int(fold),
                    }
                )
            if number % 50 == 0 or number == len(valid_ids):
                print(f"OOF fold {fold}: {number}/{len(valid_ids)}", flush=True)
    return output


all_oof_5f = []
fold_metrics_5f = []
for fold in range(1, 6):
    fold_started = time.time()
    valid_ids = sorted(
        [well for well in train_ids if FOLD_BY_WELL_5F[well] == fold]
    )
    fit_ids = sorted(
        [well for well in train_ids if FOLD_BY_WELL_5F[well] != fold]
    )
    print(
        f"\n===== PSG FOLD {fold}/5 train={len(fit_ids)} valid={len(valid_ids)} =====",
        flush=True,
    )
    fold_cloud = _prepare_structure_5f(fit_ids, valid_ids)
    checkpoint_path, best_epoch, selected_rmse = _train_fold_5f(
        fold, fit_ids, valid_ids
    )
    fold_oof = _predict_oof_5f(fold, valid_ids)
    all_oof_5f.extend(fold_oof)
    error = np.asarray(
        [row["psg_tvt"] - row["true_tvt"] for row in fold_oof], dtype=float
    )
    metrics = {
        "fold": fold,
        "train_wells": len(fit_ids),
        "valid_wells": len(valid_ids),
        "rows": len(fold_oof),
        "best_epoch": best_epoch,
        "selection_rmse": selected_rmse,
        "oof_rmse": float(np.sqrt(np.mean(error**2))),
        "oof_mae": float(np.mean(np.abs(error))),
        "checkpoint": str(checkpoint_path),
        "runtime_minutes": (time.time() - fold_started) / 60.0,
    }
    fold_metrics_5f.append(metrics)
    print("PSG FOLD COMPLETE", metrics, flush=True)
    del fold_cloud
    gc.collect()
    torch.cuda.empty_cache()

oof_5f = pd.DataFrame(all_oof_5f).sort_values(["well", "row_idx"]).reset_index(drop=True)
if oof_5f.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("OOF contains duplicate well/row_idx keys")
expected_rows = int(
    sum(CACHE[well]["n"] - CACHE[well]["real_ps"] for well in train_ids)
)
if len(oof_5f) != expected_rows:
    raise RuntimeError(f"OOF rows {len(oof_5f)} != expected {expected_rows}")
if oof_5f["well"].nunique() != len(train_ids):
    raise RuntimeError(
        f"OOF wells {oof_5f['well'].nunique()} != expected {len(train_ids)}"
    )
if not np.isfinite(oof_5f[["psg_tvt", "true_tvt"]].to_numpy()).all():
    raise RuntimeError("OOF contains non-finite prediction/truth values")

try:
    oof_5f.to_parquet(ARTIFACT_DIR / "psg_oof.parquet", index=False)
except Exception as exc:
    oof_5f.to_csv(ARTIFACT_DIR / "psg_oof.csv", index=False)
    raise RuntimeError(
        "Parquet export failed. Install pyarrow in AutoDL, then rerun only the "
        f"final export block. Original error: {exc!r}"
    )
oof_5f.to_csv(ARTIFACT_DIR / "psg_oof.csv", index=False)
pd.DataFrame(fold_metrics_5f).to_csv(
    ARTIFACT_DIR / "psg_fold_metrics.csv", index=False
)
overall_error = (
    oof_5f["psg_tvt"].to_numpy() - oof_5f["true_tvt"].to_numpy()
)
manifest = {
    "experiment": "psg_fivefold_oof_export_v1",
    "seed": SEED_5F,
    "device": DEVICE,
    "folds": 5,
    "train_wells": len(train_ids),
    "oof_rows": len(oof_5f),
    "oof_rmse": float(np.sqrt(np.mean(overall_error**2))),
    "oof_mae": float(np.mean(np.abs(overall_error))),
    "feature_count": int(N_FEAT),
    "features": list(FEATS),
    "epochs_budget": EPOCHS_5F,
    "patience": PATIENCE_5F,
    "fold_metrics": fold_metrics_5f,
    "elapsed_minutes": (time.time() - fivefold_started) / 60.0,
}
(ARTIFACT_DIR / "psg_training_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
archive = shutil.make_archive(
    "/root/autodl-tmp/psg_5fold_dataset", "zip", ARTIFACT_DIR
)
print("\nPSG FIVEFOLD EXPORT COMPLETE")
print(json.dumps(manifest, indent=2))
print("dataset folder:", ARTIFACT_DIR)
print("uploadable zip:", archive)
