"""Competition-only nuisance augmentation A/B for the real GR matcher.

Chronolog V81/V82 showed a large external-domain gap, so no Chronolog samples
are consumed here.  This cell asks the narrower remaining question: can
sensor-style nuisance augmentation, calibrated from competition known-prefix
overlaps, improve held-out same-well alias retrieval?

Protocol: Fold 4/5 train, Fold 3 selects one epoch independently per arm,
Fold 3/4/5 refit, Fold 2 confirms once, Fold 1 remains untouched.  The baseline
and augmented arms share the exact matcher, samples, loss and initialization.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE_CANDIDATES = (
    ROOT / "real_correspondence_multiscale_matcher_gateway_v0_cell.py",
    Path("real_correspondence_multiscale_matcher_gateway_v0_cell.py"),
)
SOURCE = next((path for path in SOURCE_CANDIDATES if path.is_file()), None)
if SOURCE is None:
    raise FileNotFoundError(
        "Upload real_correspondence_multiscale_matcher_gateway_v0_cell.py "
        "to /root/autodl-tmp"
    )


def _load_definitions(path: Path):
    source = path.read_text(encoding="utf-8")
    marker = "# ---------------------------------------------------------------------------\n# Execute"
    if marker not in source:
        raise RuntimeError(f"Execution marker not found in {path}")
    module_name = "real_correspondence_v83_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    # Python 3.12 dataclasses resolve postponed/string annotations through the
    # defining module in sys.modules while the class decorator is running.
    sys.modules[module_name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module.__dict__


V = _load_definitions(SOURCE)
OOF_ROOT = V["OOF_ROOT"]
EXPERIMENT = "competition_nuisance_matcher_ab_v83"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260730
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
MAX_EPOCHS = 10
MIN_EPOCHS = 4
PATIENCE = 3
REPORT_EVERY = 100

# Fold roles must be changed before samples are built because the inherited
# builder decides which wells receive train/evaluation query sets from them.
V["SEED"] = SEED
V["TUNE_FOLD"] = SELECTION_FOLD
V["TEST_FOLD"] = APPLICATION_FOLD
V["MAX_TUNING_EPOCHS"] = MAX_EPOCHS
V["MIN_TUNING_EPOCHS"] = MIN_EPOCHS
V["REPORT_EVERY_WELLS"] = REPORT_EVERY


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _robust_affine(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 32 or np.std(x) < 1e-3:
        return None
    design = np.column_stack([x, np.ones(len(x))])
    for _ in range(3):
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ coef
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        keep = np.abs(residual - np.median(residual)) <= 3.5 * scale
        if keep.sum() < 24:
            break
        design, y = design[keep], y[keep]
    gain, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.asarray([gain, offset])
    x_scale = 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6
    residual_scale = 1.4826 * np.median(
        np.abs(residual - np.median(residual))
    )
    return float(gain), float(offset / x_scale), float(residual_scale / x_scale)


def _fit_nuisance(wells):
    rows = []
    for number, well in enumerate(wells, 1):
        record = V["_load_raw_record"](well)
        last = int(record.anchor_index)
        indices = np.arange(last + 1, dtype=np.int64)
        offset = record.truth[indices].astype(np.float64) - float(record.anchor_tvt)
        horizontal = record.horizontal[0, indices].astype(np.float64)
        horizontal_valid = record.horizontal[1, indices] >= 0.5
        ref_axis = (
            np.arange(record.typewell.shape[1], dtype=np.float64)
            - record.typewell.shape[1] // 2
        )
        typewell = np.interp(offset, ref_axis, record.typewell[0].astype(np.float64))
        typewell_valid = np.interp(
            offset, ref_axis, record.typewell[1].astype(np.float64)
        ) >= 0.5
        inside = (offset >= ref_axis[0]) & (offset <= ref_axis[-1])
        fit = _robust_affine(typewell[inside & horizontal_valid & typewell_valid],
                             horizontal[inside & horizontal_valid & typewell_valid])
        if fit is not None:
            rows.append({"well": well, "gain": fit[0], "offset_scale": fit[1],
                         "residual_scale": fit[2],
                         "horizontal_missing": float(1.0 - horizontal_valid.mean())})
        if number % 100 == 0 or number == len(wells):
            print(f"V83 nuisance {number}/{len(wells)} fits={len(rows)}", flush=True)
    frame = pd.DataFrame(rows)
    positive_gain = frame.loc[frame["gain"] > 0.05, "gain"].to_numpy(np.float64)
    if len(positive_gain) < 100:
        raise RuntimeError(f"Only {len(positive_gain)} valid positive affine gains")
    config = {
        "fit_wells": int(len(frame)),
        "log_gain_sigma": float(np.clip(np.std(np.log(positive_gain)), 0.02, 0.25)),
        "offset_sigma_scale": float(np.clip(frame["offset_scale"].std(), 0.02, 0.50)),
        "noise_sigma_scale": float(np.clip(frame["residual_scale"].median(), 0.01, 0.30)),
        "extra_dropout_probability": float(
            np.clip(frame["horizontal_missing"].quantile(0.50) * 0.15, 0.0, 0.08)
        ),
        "smooth_probability": 0.25,
    }
    frame.to_csv(OUT_DIR / "known_prefix_nuisance_fits.csv", index=False)
    _write_json(OUT_DIR / "nuisance_config.json", config)
    return config


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _augment_tensor(tensor, config, rng, dose=1.0):
    output = tensor.clone()
    values = output[..., 0, :]
    mask = output[..., 1, :] > 0.5
    finite = values[mask]
    if finite.numel() < 8:
        return output
    centre = finite.median()
    scale = (finite - centre).abs().median() * 1.4826
    scale = scale.clamp_min(1e-3)
    gain = math.exp(float(rng.normal(0.0, dose * config["log_gain_sigma"])))
    offset = float(rng.normal(0.0, dose * config["offset_sigma_scale"])) * scale
    transformed = gain * values + offset
    noise_sigma = dose * config["noise_sigma_scale"] * scale
    if float(noise_sigma) > 0:
        transformed = transformed + torch.randn_like(transformed) * noise_sigma
    if rng.random() < dose * config["smooth_probability"]:
        shape = transformed.shape
        smooth = F.avg_pool1d(
            transformed.reshape(-1, 1, shape[-1]), 3, stride=1, padding=1
        ).reshape(shape)
        transformed = 0.5 * transformed + 0.5 * smooth
    drop_probability = dose * config["extra_dropout_probability"]
    if drop_probability > 0:
        keep = torch.rand_like(transformed) >= drop_probability
        mask = mask & keep
    output[..., 0, :] = torch.where(mask, transformed, torch.zeros_like(transformed))
    output[..., 1, :] = mask.to(output.dtype)
    return output


def _train_epoch(model, samples, wells, device, optimizer, scaler, rng, config, augmented):
    model.train()
    mean_rows = float(np.mean([samples[well].eval_rows for well in wells]))
    totals = {"loss": 0.0, "ce": 0.0, "alias": 0.0, "skipped": 0}
    order = rng.permutation(len(wells))
    for step, position in enumerate(order, 1):
        well = wells[int(position)]
        sample = samples[well]
        query_set = sample.train
        query, reference, state_valid, labels, label_index, row_weight = V["_to_tensors"](
            sample, query_set, device
        )
        if augmented:
            # Retain clean observations in half the steps.  This keeps the A/B
            # focused on invariance rather than replacing the target domain.
            dose = float(rng.uniform(0.25, 1.0)) if rng.random() < 0.50 else 0.0
            if dose > 0:
                query = _augment_tensor(query, config, rng, dose)
                reference = _augment_tensor(reference, config, rng, dose)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(query, reference, state_valid)
            loss, ce, alias = V["_matcher_loss"](
                logits, labels, label_index, row_weight, state_valid
            )
            loss = loss * sample.eval_rows / max(mean_rows, 1.0)
        before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), V["GRADIENT_CLIP"])
        scaler.step(optimizer)
        scaler.update()
        totals["skipped"] += int(float(scaler.get_scale()) < before)
        totals["loss"] += float(loss.detach().cpu())
        totals["ce"] += float(ce.cpu())
        totals["alias"] += float(alias.cpu())
        if step % REPORT_EVERY == 0 or step == len(wells):
            print(
                f"V83 train augmented={augmented} wells={step}/{len(wells)} "
                f"loss={totals['loss']/step:.4f}", flush=True
            )
    return {
        "train_loss": totals["loss"] / len(wells),
        "train_ce": totals["ce"] / len(wells),
        "train_alias": totals["alias"] / len(wells),
        "skipped_steps": totals["skipped"],
    }


def _objects(device, epochs, seed):
    _seed(seed)
    model = V["RealCorrespondenceMatcher"]().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=V["LEARNING_RATE"], weight_decay=V["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=V["MIN_LEARNING_RATE"]
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=512.0
    )
    return model, optimizer, scheduler, scaler


def _objective(metrics):
    return (
        metrics["effective_top8_within5"],
        metrics["deep_top8_within5"],
        -metrics["median_rank"],
        -metrics["argmax_rmse"],
    )


def _select_epoch(arm, samples, train_wells, selection_wells, device, config, augmented):
    # Matched initialization and well order make augmentation the only A/B
    # difference. Torch's stream may then diverge only when augmentation draws
    # noise, which is part of the intervention being tested.
    seed = SEED
    model, optimizer, scheduler, scaler = _objects(device, MAX_EPOCHS, seed)
    rng = np.random.default_rng(seed)
    history, best, stale = [], None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train = _train_epoch(
            model, samples, train_wells, device, optimizer, scaler, rng, config, augmented
        )
        scheduler.step()
        metrics, _, _, _, _ = V["_evaluate"](
            model, samples, selection_wells, device, mode="full", include_ncc=False
        )
        row = {"arm": arm, "epoch": epoch, **train,
               **{f"selection_{key}": value for key, value in metrics.items()}}
        history.append(row)
        print("V83 EPOCH", row, flush=True)
        score = _objective(metrics)
        if best is None or score > best["score"]:
            best = {"epoch": epoch, "score": score}
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return int(best["epoch"]), pd.DataFrame(history)


def _fit_arm(arm, samples, wells, epochs, device, config, augmented):
    seed = SEED + 5000
    model, optimizer, scheduler, scaler = _objects(device, epochs, seed)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        row = _train_epoch(
            model, samples, wells, device, optimizer, scaler, rng, config, augmented
        )
        scheduler.step()
        history.append({"arm": arm, "epoch": epoch, **row})
    torch.save(
        {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "arm": arm, "epochs": epochs, "train_folds": [3, 4, 5],
         "nuisance": config if augmented else None},
        MODEL_DIR / f"{arm}_outer345.pt",
    )
    return model, pd.DataFrame(history)


def _ensure_record_cache(required_wells):
    record_dir = Path(V["RECORD_DIR"])
    record_dir.mkdir(parents=True, exist_ok=True)
    missing = [well for well in required_wells if not (record_dir / f"{well}.npz").is_file()]
    if not missing:
        print(f"V83 record cache ready: {len(required_wells)} wells", flush=True)
        return

    builder_candidates = (
        ROOT / "native_resolution_dualref_capacity_v0_cell.py",
        Path("native_resolution_dualref_capacity_v0_cell.py"),
    )
    builder_path = next((path for path in builder_candidates if path.is_file()), None)
    if builder_path is None:
        raise FileNotFoundError(
            f"Missing {len(missing)} native records and no builder source was found. "
            "Upload native_resolution_dualref_capacity_v0_cell.py to /root/autodl-tmp."
        )
    source = builder_path.read_text(encoding="utf-8")
    marker = "# Execute"
    if marker not in source:
        raise RuntimeError(f"Execution marker not found in {builder_path}")
    module_name = "native_dualref_v83_record_builder"
    module = types.ModuleType(module_name)
    module.__file__ = str(builder_path)
    sys.modules[module_name] = module
    exec(compile(source.split(marker, 1)[0], str(builder_path), "exec"), module.__dict__)
    builder = module.__dict__
    # Keep the inherited matcher's cache location authoritative even if the
    # builder source was configured with another OOF root.
    builder["CACHE_DIR"] = record_dir
    print(
        f"V83 rebuilding {len(missing)}/{len(required_wells)} native records "
        f"into {record_dir}", flush=True
    )
    failures = []
    for number, well in enumerate(missing, 1):
        try:
            record = builder["_build_record"](well)
            builder["_save_record"](record)
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 25 == 0 or number == len(missing):
            print(
                f"V83 records {number}/{len(missing)} failures={len(failures)}",
                flush=True,
            )
    pd.DataFrame(failures).to_csv(OUT_DIR / "record_build_failures.csv", index=False)
    remaining = [well for well in required_wells if not (record_dir / f"{well}.npz").is_file()]
    if remaining:
        raise RuntimeError(
            f"Failed to build {len(remaining)} required records: {failures[:5]}"
        )


if not torch.cuda.is_available():
    raise RuntimeError("V83 requires CUDA")
device = torch.device("cuda")
wells = V["_horizontal_wells"]()
folds, fold_path = V["_load_folds"](wells)
folds = folds[folds["fold"].isin([2, 3, 4, 5])].copy()
_ensure_record_cache(folds["well"].astype(str).tolist())
samples, failures = {}, []
for number, row in enumerate(folds.itertuples(index=False), 1):
    try:
        samples[row.well] = V["_build_well_sample"](row.well, int(row.fold))
    except Exception as exc:
        failures.append({"well": row.well, "fold": int(row.fold), "reason": repr(exc)})
    if number % 50 == 0 or number == len(folds):
        print(f"V83 samples {number}/{len(folds)} success={len(samples)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "sample_failures.csv", index=False)
if len(samples) < 610:
    raise RuntimeError(f"Only {len(samples)} usable wells: {failures[:5]}")

fold_by_well = dict(zip(folds["well"].astype(str), folds["fold"].astype(int)))
train_wells = sorted(w for w in samples if fold_by_well[w] in TRAIN_FOLDS)
selection_wells = sorted(w for w in samples if fold_by_well[w] == SELECTION_FOLD)
application_wells = sorted(w for w in samples if fold_by_well[w] == APPLICATION_FOLD)
if len(train_wells) < 307 or len(selection_wells) < 153 or len(application_wells) < 154:
    raise RuntimeError(
        f"Unexpected fold coverage: {len(train_wells)}/{len(selection_wells)}/{len(application_wells)}"
    )

config = _fit_nuisance(train_wells)
epoch_rows, fit_rows, results = [], [], {}
for arm, augmented in (("baseline", False), ("nuisance_augmented", True)):
    selected_epoch, history = _select_epoch(
        arm, samples, train_wells, selection_wells, device, config, augmented
    )
    epoch_rows.append(history)
    outer_wells = sorted(train_wells + selection_wells)
    model, fit_history = _fit_arm(
        arm, samples, outer_wells, selected_epoch, device, config, augmented
    )
    fit_rows.append(fit_history)
    full, bywell, payload, _, _ = V["_evaluate"](
        model, samples, application_wells, device, mode="full", include_ncc=False
    )
    no_gr, _, _, _, _ = V["_evaluate"](
        model, samples, application_wells, device, mode="no_gr", include_ncc=False
    )
    shuffled, _, _, _, _ = V["_evaluate"](
        model, samples, application_wells, device, mode="shuffled_typewell", include_ncc=False
    )
    bywell.to_csv(OUT_DIR / f"{arm}_fold2_bywell.csv", index=False)
    results[arm] = {
        "selected_epoch": selected_epoch,
        "full": full,
        "no_gr": no_gr,
        "shuffled_typewell": shuffled,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()

pd.concat(epoch_rows, ignore_index=True).to_csv(OUT_DIR / "epoch_selection.csv", index=False)
pd.concat(fit_rows, ignore_index=True).to_csv(OUT_DIR / "outer_fit_history.csv", index=False)

base = results["baseline"]["full"]
aug = results["nuisance_augmented"]["full"]
gains = {
    "effective_top8_within5": aug["effective_top8_within5"] - base["effective_top8_within5"],
    "deep_top8_within5": aug["deep_top8_within5"] - base["deep_top8_within5"],
    "median_rank": base["median_rank"] - aug["median_rank"],
    "argmax_rmse": base["argmax_rmse"] - aug["argmax_rmse"],
    "alias24_win_rate": aug["alias24_win_rate"] - base["alias24_win_rate"],
}
no_gr_drop = (
    results["nuisance_augmented"]["full"]["effective_top8_within5"]
    - results["nuisance_augmented"]["no_gr"]["effective_top8_within5"]
)
shuffle_drop = (
    results["nuisance_augmented"]["full"]["effective_top8_within5"]
    - results["nuisance_augmented"]["shuffled_typewell"]["effective_top8_within5"]
)
checks = {
    "top8_gain_ge_0p03": gains["effective_top8_within5"] >= 0.03,
    "deep_top8_gain_ge_0p03": gains["deep_top8_within5"] >= 0.03,
    "argmax_rmse_gain_ge_2ft": gains["argmax_rmse"] >= 2.0,
    "no_gr_top8_drop_ge_0p10": no_gr_drop >= 0.10,
    "typewell_shuffle_top8_drop_ge_0p10": shuffle_drop >= 0.10,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "competition_only_nuisance_augmentation_fold2_gateway",
    "external_samples_used": False,
    "source": str(SOURCE),
    "fold_source": str(fold_path),
    "protocol": {
        "train_folds": [4, 5], "selection_fold": 3, "application_fold": 2,
        "untouched_fold": 1, "fold1_touched": False,
    },
    "coverage": {
        "train_wells": len(train_wells), "selection_wells": len(selection_wells),
        "application_wells": len(application_wells), "failures": len(failures),
    },
    "nuisance": config,
    "results": results,
    "gains": gains,
    "causal_controls": {"no_gr_top8_drop": no_gr_drop, "typewell_shuffle_top8_drop": shuffle_drop},
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> run the augmented emission through one frozen global path decoder"
        if strict_pass
        else "FAIL -> close learned GR correspondence and nuisance-augmentation branches"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV83 FINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
