# Direct AutoDL cell: rebuild PSG OOF on the exact fold-clean spatial folds.
# Run in a kernel containing the validated PSG raw-well cache and definitions.
# Training starts from scratch; no checkpoint that saw a held-out fold is reused.

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
    source_candidates_5f = [
        Path("/root/autodl-tmp/psg_fivefold_train_export_cell.py"),
        Path("psg_fivefold_train_export_cell.py"),
    ]
    source_path_5f = next(
        (path for path in source_candidates_5f if path.exists()), None
    )
    if source_path_5f is None:
        raise RuntimeError(
            f"Missing live definitions: {missing}. Upload "
            "psg_fivefold_train_export_cell.py to /root/autodl-tmp, or run its "
            "definition/data prefix first."
        )
    source_5f = source_path_5f.read_text(encoding="utf-8")
    training_marker_5f = "rng = np.random.RandomState(SEED)\ntr = list(train_ids)"
    if training_marker_5f not in source_5f:
        raise RuntimeError(
            f"Cannot find the training marker in {source_path_5f}; "
            "refusing to execute an unknown source layout."
        )
    print("PSG FOLDCLEAN BOOTSTRAP", str(source_path_5f), flush=True)
    exec(
        compile(
            source_5f.split(training_marker_5f, 1)[0],
            str(source_path_5f) + "::<definitions>",
            "exec",
        ),
        globals(),
    )
    missing = sorted(REQUIRED.difference(globals()))
    if missing:
        raise RuntimeError(f"PSG bootstrap still misses definitions: {missing}")
if int(N_FEAT) != 26:
    raise RuntimeError(
        f"Expected the inference-compatible 26-channel feature contract, got {N_FEAT}: {FEATS}"
    )
if DEVICE == "cpu":
    raise RuntimeError("Five-fold PSG training requires an AutoDL CUDA instance")

FOLD_META_PATH = Path(
    "/root/autodl-tmp/oof_exports/foldclean_full_fleongg_lgb_v0/well_metadata.csv"
)
if not FOLD_META_PATH.exists():
    raise FileNotFoundError(
        f"Missing {FOLD_META_PATH}. Run/upload foldclean_full_fleongg_lgb_v0 first."
    )

ARTIFACT_DIR = Path("/root/autodl-tmp/oof_exports/psg_foldclean_fivefold_oof_v0")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SEED_5F = 20260722
EPOCHS_5F = int(os.environ.get("PSG_EPOCHS", "45"))
PATIENCE_5F = int(os.environ.get("PSG_PATIENCE", "10"))
ACC_5F = 6
fivefold_started = time.time()


def _pseudo_ps_5f(well, rng):
    return int(rng.uniform(0.20, 0.45) * well["n"])


fold_metadata_5f = pd.read_csv(FOLD_META_PATH)
required_fold_columns_5f = {"well", "fold"}
if not required_fold_columns_5f.issubset(fold_metadata_5f.columns):
    raise RuntimeError(
        f"Fold metadata must contain {sorted(required_fold_columns_5f)}; "
        f"got {fold_metadata_5f.columns.tolist()}"
    )
fold_metadata_5f = fold_metadata_5f[["well", "fold"]].copy()
fold_metadata_5f["well"] = fold_metadata_5f["well"].astype(str).str.lower()
fold_metadata_5f["fold"] = pd.to_numeric(
    fold_metadata_5f["fold"], errors="raise"
).astype(int)
if fold_metadata_5f["well"].duplicated().any():
    raise RuntimeError("Fold metadata contains duplicate wells")
if not fold_metadata_5f["fold"].isin(range(1, 6)).all():
    raise RuntimeError("Fold metadata contains values outside 1..5")

metadata_fold_by_well_5f = dict(
    zip(fold_metadata_5f["well"], fold_metadata_5f["fold"])
)


def _recover_full_training_cache_5f(expected_wells):
    """Recover original labelled wells when bootstrap selected a masked copy."""
    search_roots = []
    configured_root = os.environ.get("ROGII_DATA_ROOT")
    if configured_root:
        search_roots.append(Path(configured_root))
    search_roots.extend(
        [
            Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
            Path("/kaggle/input/rogii-wellbore-geology-prediction"),
            Path("/kaggle/input"),
            Path("/root/autodl-tmp"),
        ]
    )
    unique_roots = []
    seen_roots = set()
    for root in search_roots:
        try:
            resolved = str(root.resolve())
        except Exception:
            resolved = str(root)
        if root.exists() and resolved not in seen_roots:
            unique_roots.append(root)
            seen_roots.add(resolved)

    expected = {str(well).lower() for well in expected_wells}
    paths_by_well = {well: [] for well in expected}
    scanned_paths = set()
    root_counts = {}
    for root in unique_roots:
        count = 0
        for path in root.rglob("*__horizontal_well.csv"):
            key = str(path.resolve())
            if key in scanned_paths:
                continue
            scanned_paths.add(key)
            well = path.name.split("__", 1)[0].lower()
            if well in paths_by_well:
                paths_by_well[well].append(path)
                count += 1
        root_counts[str(root)] = count
    print(
        "PSG RAW RECOVERY SCAN",
        {"roots": root_counts, "unique_files": len(scanned_paths)},
        flush=True,
    )

    selected_paths = {}
    rejected = []
    for number, well in enumerate(sorted(expected), start=1):
        best = None
        best_score = None
        for path in paths_by_well.get(well, []):
            try:
                columns = pd.read_csv(
                    path,
                    usecols=lambda column: column in {"TVT", "TVT_input"},
                )
                if "TVT" not in columns:
                    continue
                truth_count = int(
                    np.isfinite(pd.to_numeric(columns["TVT"], errors="coerce")).sum()
                )
                input_count = (
                    int(
                        np.isfinite(
                            pd.to_numeric(columns["TVT_input"], errors="coerce")
                        ).sum()
                    )
                    if "TVT_input" in columns
                    else 0
                )
                score = (truth_count > input_count, truth_count, len(columns))
                if best_score is None or score > best_score:
                    best = path
                    best_score = score
            except Exception as exc:
                rejected.append({"well": well, "path": str(path), "reason": repr(exc)})
        if best is not None and best_score[0]:
            selected_paths[well] = best
        if number % 100 == 0 or number == len(expected):
            print(
                f"PSG raw recovery {number}/{len(expected)} "
                f"complete_truth={len(selected_paths)}",
                flush=True,
            )

    if len(selected_paths) < 760:
        missing = sorted(expected.difference(selected_paths))
        diagnostic_path = ARTIFACT_DIR / "raw_recovery_failures.csv"
        pd.DataFrame(rejected).to_csv(diagnostic_path, index=False)
        raise RuntimeError(
            f"Only {len(selected_paths)} original complete-truth wells were found. "
            f"Missing examples={missing[:20]}. Set ROGII_DATA_ROOT to the directory "
            "containing all original training CSVs, then rerun. "
            f"Diagnostics: {diagnostic_path}"
        )

    recovered_cache = {}
    load_failures = []
    for number, well in enumerate(sorted(selected_paths), start=1):
        try:
            record = load_well(str(selected_paths[well]))
            if not record["is_train"]:
                raise RuntimeError("selected file is not complete-truth training data")
            recovered_cache[str(record["wid"]).lower()] = record
        except Exception as exc:
            load_failures.append(
                {"well": well, "path": str(selected_paths[well]), "reason": repr(exc)}
            )
        if number % 100 == 0 or number == len(selected_paths):
            print(
                f"PSG raw reload {number}/{len(selected_paths)} "
                f"success={len(recovered_cache)}",
                flush=True,
            )
    if len(recovered_cache) < 760:
        pd.DataFrame(load_failures).to_csv(
            ARTIFACT_DIR / "raw_reload_failures.csv", index=False
        )
        raise RuntimeError(
            f"Only {len(recovered_cache)} complete wells reloaded; "
            f"failures={load_failures[:10]}"
        )
    return recovered_cache


initial_metadata_matches_5f = sum(
    str(well).lower() in metadata_fold_by_well_5f for well in train_ids
)
if initial_metadata_matches_5f < 760:
    print(
        "PSG BOOTSTRAP CACHE IS INCOMPLETE",
        {
            "bootstrap_train_wells": len(train_ids),
            "metadata_matches": initial_metadata_matches_5f,
            "action": "recover original complete-truth CSVs",
        },
        flush=True,
    )
    del CACHE
    gc.collect()
    CACHE = _recover_full_training_cache_5f(metadata_fold_by_well_5f)
    train_ids = sorted(CACHE)

eligible_train_ids_5f = sorted(
    str(well).lower()
    for well in train_ids
    if str(well).lower() in metadata_fold_by_well_5f
)
missing_metadata_wells_5f = sorted(
    str(well).lower()
    for well in train_ids
    if str(well).lower() not in metadata_fold_by_well_5f
)
if missing_metadata_wells_5f:
    raise RuntimeError(
        "PSG wells missing from fold metadata: " + repr(missing_metadata_wells_5f[:20])
    )
if len(eligible_train_ids_5f) < 760:
    raise RuntimeError(
        f"Only {len(eligible_train_ids_5f)} PSG wells have fold-clean assignments"
    )

FOLD_BY_WELL_5F = {
    well: int(metadata_fold_by_well_5f[well]) for well in eligible_train_ids_5f
}
fold_table_5f = pd.DataFrame(
    {
        "well": eligible_train_ids_5f,
        "fold": [FOLD_BY_WELL_5F[well] for well in eligible_train_ids_5f],
    }
)
fold_table_5f.to_csv(ARTIFACT_DIR / "psg_folds.csv", index=False)
fold_counts_5f = fold_table_5f.groupby("fold")["well"].nunique().to_dict()
print(
    "PSG FOLDCLEAN CONFIG",
    {
        "fold_source": str(FOLD_META_PATH),
        "eligible_wells": len(eligible_train_ids_5f),
        "fold_counts": fold_counts_5f,
        "feature_count": int(N_FEAT),
        "epochs": EPOCHS_5F,
        "patience": PATIENCE_5F,
    },
    flush=True,
)


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
        [well for well in eligible_train_ids_5f if FOLD_BY_WELL_5F[well] == fold]
    )
    fit_ids = sorted(
        [well for well in eligible_train_ids_5f if FOLD_BY_WELL_5F[well] != fold]
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
    sum(
        CACHE[well]["n"] - CACHE[well]["real_ps"]
        for well in eligible_train_ids_5f
    )
)
if len(oof_5f) != expected_rows:
    raise RuntimeError(f"OOF rows {len(oof_5f)} != expected {expected_rows}")
if oof_5f["well"].nunique() != len(eligible_train_ids_5f):
    raise RuntimeError(
        f"OOF wells {oof_5f['well'].nunique()} != expected "
        f"{len(eligible_train_ids_5f)}"
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
    "experiment": "psg_foldclean_fivefold_oof_v0",
    "fold_source": str(FOLD_META_PATH),
    "fold_contract": "each PSG model and spatial cloud exclude the complete held-out fold",
    "seed": SEED_5F,
    "device": DEVICE,
    "folds": 5,
    "train_wells": len(eligible_train_ids_5f),
    "fold_counts": {str(key): int(value) for key, value in fold_counts_5f.items()},
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
    "/root/autodl-tmp/psg_foldclean_fivefold_oof_v0", "zip", ARTIFACT_DIR
)
print("\nPSG FIVEFOLD EXPORT COMPLETE")
print(json.dumps(manifest, indent=2))
print("dataset folder:", ARTIFACT_DIR)
print("uploadable zip:", archive)
