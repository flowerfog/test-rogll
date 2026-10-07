"""Held-out fold-1 gateway for the native-resolution dual-reference model.

Prerequisite
------------
Keep ``native_resolution_dualref_capacity_v0_cell.py`` at
``/root/autodl-tmp``.  This cell imports only its definitions (it does not
rerun the capacity ladder), builds/reuses official-mask records for all 773
wells, trains on folds 2-5, and evaluates untouched fold 1.

The metric is pooled row-level RMSE.  Each optimisation step is one complete
well and its loss is weighted by eval-row count, so the training objective
matches the competition metric rather than mean-per-well RMSE.
"""

from __future__ import annotations

from pathlib import Path
import ast
import gc
import json
import math
import random
import shutil
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bootstrap the validated capacity definitions without executing that file.
# ---------------------------------------------------------------------------


def _bootstrap_capacity_namespace():
    candidates = (
        Path("/root/autodl-tmp/native_resolution_dualref_capacity_v0_cell.py"),
        Path.cwd() / "native_resolution_dualref_capacity_v0_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        # In Jupyter the capacity cell may have been pasted/executed directly,
        # so its validated definitions exist in the live kernel even though no
        # .py source file was materialised on disk.
        live = globals()
        required = {
            "Record",
            "NativeDualReferenceWarpNet",
            "_build_record",
            "_horizontal_paths",
            "_load_record",
            "_record_path",
            "_save_record",
            "_seed_everything",
            "_alignment_loss",
            "_rmse",
            "STATE_OFFSETS",
            "N_STATES",
            "HORIZONTAL_STRIDE",
            "OOF_ROOT",
            "DATA_ROOT",
            "CACHE_DIR",
        }
        missing = sorted(required.difference(live))
        if missing:
            raise FileNotFoundError(
                "Cannot find the capacity source file or its live-kernel "
                f"definitions; missing {missing}. Run the capacity cell in "
                "this kernel or upload native_resolution_dualref_capacity_v0_cell.py."
            )
        print(
            "capacity bootstrap: reusing validated live-kernel definitions",
            flush=True,
        )
        return live, Path("live_kernel_capacity_definitions")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        definitions.append(node)
    module_name = "native_resolution_dualref_capacity_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    namespace = module.__dict__
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace, source_path


V, CAPACITY_SOURCE = _bootstrap_capacity_namespace()

Record = V["Record"]
NativeDualReferenceWarpNet = V["NativeDualReferenceWarpNet"]
_build_record = V["_build_record"]
_horizontal_paths = V["_horizontal_paths"]
_load_record = V["_load_record"]
_record_path = V["_record_path"]
_save_record = V["_save_record"]
_seed_everything = V["_seed_everything"]
_alignment_loss = V["_alignment_loss"]
_rmse = V["_rmse"]
STATE_OFFSETS = V["STATE_OFFSETS"]
N_STATES = V["N_STATES"]
HORIZONTAL_STRIDE = V["HORIZONTAL_STRIDE"]
OOF_ROOT = V["OOF_ROOT"]
DATA_ROOT = V["DATA_ROOT"]
CAPACITY_CACHE_DIR = V["CACHE_DIR"]


# ---------------------------------------------------------------------------
# One-fold configuration
# ---------------------------------------------------------------------------


EXPERIMENT = "native_resolution_dualref_onefold_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
VALID_FOLD = 1
SEED = 20260719
REQUIRE_CUDA = True

EPOCHS = 60
REPORT_EVERY = 2
EARLY_STOPPING_REPORTS = 7
MIN_EPOCHS_BEFORE_STOP = 12
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 2e-5
GRADIENT_CLIP = 5.0
GRAD_SCALER_INITIAL_SCALE = 1024.0

ARCHITECTURE_GATE_FT = 7.0
TARGET_GATE_FT = 5.5
NO_GR_DEGRADATION_GATE_FT = 1.0
SHUFFLED_TYPEWELL_DEGRADATION_GATE_FT = 0.20

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)


def _write_json(path: Path, payload: dict) -> None:
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def _load_folds(wells):
    fold_path = next((path for path in FOLD_CANDIDATES if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError(
            "Cannot find fold-clean well_metadata.csv. Expected one of: "
            + repr([str(path) for path in FOLD_CANDIDATES])
        )
    frame = pd.read_csv(fold_path, dtype={"well": str})
    well_column = next(
        (column for column in ("well", "well_id", "WELL") if column in frame),
        None,
    )
    fold_column = next(
        (column for column in ("fold", "outer_fold", "Fold") if column in frame),
        None,
    )
    if well_column is None or fold_column is None:
        raise RuntimeError(
            f"Fold metadata needs well/fold columns; found {frame.columns.tolist()}"
        )
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    frame = frame[frame["well"].isin(set(map(str, wells)))]
    if len(frame) != len(wells):
        missing = sorted(set(map(str, wells)) - set(frame["well"]))
        raise RuntimeError(
            f"Fold metadata covers {len(frame)}/{len(wells)} wells; missing {missing[:10]}"
        )
    unique_folds = sorted(frame["fold"].unique().tolist())
    if unique_folds != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Expected folds 1-5; found {unique_folds}")
    return frame.sort_values(["fold", "well"]).reset_index(drop=True), fold_path


def _ensure_all_records(wells):
    CAPACITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(CAPACITY_CACHE_DIR.glob("*.npz"))
    existing_bytes = sum(path.stat().st_size for path in existing)
    if existing:
        estimated_total = existing_bytes / len(existing) * len(wells)
        free = shutil.disk_usage(OOF_ROOT).free
        additional = max(0.0, estimated_total - existing_bytes)
        print(
            "CACHE ESTIMATE",
            {
                "existing_records": len(existing),
                "existing_mb": existing_bytes / 1024**2,
                "estimated_total_mb": estimated_total / 1024**2,
                "additional_mb": additional / 1024**2,
                "free_gb": free / 1024**3,
            },
            flush=True,
        )
        if additional + 0.75 * 1024**3 > free:
            raise RuntimeError(
                "Insufficient disk for the native record cache plus a 0.75 GB safety reserve"
            )

    rows = []
    for index, well in enumerate(wells, 1):
        path = _record_path(well)
        try:
            valid_cache = False
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    valid_cache = (
                        "cache_version" in data.files
                        and int(data["cache_version"][0]) == 1
                    )
            if not valid_cache:
                record = _build_record(well)
                _save_record(record)
                source = "built"
            else:
                record = _load_record(well)
                source = "cached"
            rows.append(
                {
                    "well": well,
                    "status": "ok",
                    "source": source,
                    "rows": len(record.md),
                    "eval_rows": record.eval_rows,
                    "teacher_rmse": record.teacher_rmse,
                    "axis_coverage": record.axis_coverage,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "well": well,
                    "status": "failed",
                    "source": None,
                    "reason": repr(exc),
                }
            )
        if index % 50 == 0 or index == len(wells):
            print(f"build/reuse native records {index}/{len(wells)}", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "record_build_report.csv", index=False)
    failures = report[report["status"] != "ok"]
    if len(failures):
        raise RuntimeError(
            f"Failed to prepare {len(failures)} wells; see record_build_report.csv"
        )
    return report


def _tensor_inputs(record, device, mode="full", donor=None):
    horizontal = record.horizontal.copy()
    condition = record.condition.copy()
    typewell = record.typewell.copy()
    heel = record.heel.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    elif mode == "shuffled_typewell":
        if donor is None:
            raise ValueError("shuffled_typewell needs a donor")
        typewell = donor.typewell.copy()
    elif mode == "shuffled_heel":
        if donor is None:
            raise ValueError("shuffled_heel needs a donor")
        heel = donor.heel.copy()
    elif mode != "full":
        raise ValueError(mode)
    return (
        torch.from_numpy(horizontal)[None].to(device),
        torch.from_numpy(condition)[None].to(device),
        torch.from_numpy(typewell)[None].to(device),
        torch.from_numpy(heel)[None].to(device),
    )


@torch.no_grad()
def _predict(model, record, device, mode="full", donor=None):
    model.eval()
    inputs = _tensor_inputs(record, device, mode=mode, donor=donor)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(*inputs)
    probability = F.softmax(logits[0].float(), dim=0)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    expected = (probability * states).sum(dim=0).cpu().numpy()
    columns = min(len(expected), len(record.latent_index))
    latent_index = record.latent_index[:columns]
    latent_eval = record.latent_eval_mask[:columns]
    eval_latent_index = latent_index[latent_eval]
    eval_expected = expected[:columns][latent_eval]
    support_md = np.r_[record.md[record.anchor_index], record.md[eval_latent_index]]
    support_tvt = np.r_[record.anchor_tvt, record.anchor_tvt + eval_expected]
    prediction = np.interp(record.md[record.eval_index], support_md, support_tvt)
    truth = record.truth[record.eval_index].astype(np.float64)
    return truth, prediction.astype(np.float64)


@torch.no_grad()
def _evaluate(model, records, device, mode="full", save_rows=False):
    bywell_rows = []
    truths = []
    predictions = []
    well_labels = []
    for index, record in enumerate(records):
        donor = None
        if mode in ("shuffled_typewell", "shuffled_heel"):
            donor = records[(index + max(1, len(records) // 2)) % len(records)]
        truth, prediction = _predict(
            model,
            record,
            device,
            mode=mode,
            donor=donor,
        )
        truths.append(truth)
        predictions.append(prediction)
        if save_rows:
            well_labels.append(np.full(len(truth), record.well, dtype="U16"))
        bywell_rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "rmse": _rmse(truth, prediction),
                "mae": float(np.mean(np.abs(prediction - truth))),
                "mean_error": float(np.mean(prediction - truth)),
            }
        )
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    bywell = pd.DataFrame(bywell_rows)
    metrics = {
        "rows": len(truth),
        "wells": len(records),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "wells_above_12ft": int((bywell["rmse"] > 12.0).sum()),
    }
    row_payload = None
    if save_rows:
        row_payload = {
            "well": np.concatenate(well_labels),
            "truth": truth.astype(np.float32),
            "prediction": prediction.astype(np.float32),
        }
    return metrics, bywell, row_payload


def _anchor_baseline(records):
    truths = []
    predictions = []
    bywell_rows = []
    for record in records:
        truth = record.truth[record.eval_index].astype(np.float64)
        prediction = np.full(len(truth), record.anchor_tvt, np.float64)
        truths.append(truth)
        predictions.append(prediction)
        bywell_rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "rmse": _rmse(truth, prediction),
            }
        )
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    bywell = pd.DataFrame(bywell_rows)
    return {
        "rmse": _rmse(truth, prediction),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
    }


def _train(train_records, valid_records, device):
    _seed_everything(SEED)
    model = NativeDualReferenceWarpNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=GRAD_SCALER_INITIAL_SCALE,
        growth_interval=500,
    )
    mean_rows = float(np.mean([record.eval_rows for record in train_records]))
    row_weights = {
        record.well: record.eval_rows / max(mean_rows, 1.0)
        for record in train_records
    }
    rng = np.random.default_rng(SEED)
    history = []
    best = None
    reports_without_improvement = 0
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_records))
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_offset = 0.0
        successful_steps = 0
        skipped_steps = 0
        for batch_number, index in enumerate(order, 1):
            record = train_records[int(index)]
            inputs = _tensor_inputs(record, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(*inputs)
                loss, ce, offset, _ = _alignment_loss(
                    logits,
                    record,
                    device,
                    row_weights[record.well],
                )
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                skipped_steps += 1
            else:
                successful_steps += 1
            epoch_loss += float(loss.detach().cpu())
            epoch_ce += float(ce.cpu())
            epoch_offset += float(offset.cpu())
            del inputs, logits, loss
            if batch_number % 150 == 0:
                print(
                    f"epoch {epoch}/{EPOCHS} train wells {batch_number}/{len(train_records)}",
                    flush=True,
                )
        if successful_steps:
            scheduler.step()

        should_report = (
            epoch == 1
            or epoch == 2
            or epoch % REPORT_EVERY == 0
            or epoch == EPOCHS
        )
        if not should_report:
            continue
        valid_metrics, valid_bywell, _ = _evaluate(
            model,
            valid_records,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_records),
            "train_ce": epoch_ce / len(train_records),
            "train_offset": epoch_offset / len(train_records),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "successful_steps": successful_steps,
            "skipped_steps": skipped_steps,
            "correlation_scale": float(
                F.softplus(model.correlation_scale).detach().cpu()
            ),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("ONEFOLD EPOCH", row, flush=True)

        if best is None or valid_metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
            best = {
                "epoch": epoch,
                "metrics": valid_metrics,
                "bywell": valid_bywell.copy(),
                "state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "epoch": epoch,
                    "metrics": valid_metrics,
                    "model": best["state"],
                    "state_offsets": STATE_OFFSETS,
                    "capacity_source": str(CAPACITY_SOURCE),
                },
                MODEL_DIR / "best_fold1.pt",
            )
            reports_without_improvement = 0
        else:
            reports_without_improvement += 1

        pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
        if (
            epoch >= MIN_EPOCHS_BEFORE_STOP
            and reports_without_improvement >= EARLY_STOPPING_REPORTS
        ):
            print(
                f"early stopping at epoch {epoch}; best epoch={best['epoch']}",
                flush=True,
            )
            break

    if best is None:
        raise RuntimeError("No valid checkpoint was produced")
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


STARTED = time.time()
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("This one-fold experiment requires a CUDA GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_everything(SEED)

wells = [path.name.split("__", 1)[0] for path in _horizontal_paths()]
folds, fold_path = _load_folds(wells)
build_report = _ensure_all_records(wells)
records = {well: _load_record(well) for well in wells}

train_wells = folds.loc[folds["fold"] != VALID_FOLD, "well"].tolist()
valid_wells = folds.loc[folds["fold"] == VALID_FOLD, "well"].tolist()
train_records = [records[well] for well in train_wells]
valid_records = [records[well] for well in valid_wells]
if len(train_records) != 618 or len(valid_records) != 155:
    raise RuntimeError(
        f"Expected fold1 split 618/155; found {len(train_records)}/{len(valid_records)}"
    )

guards = {
    "capacity_verdict": str(OOF_ROOT / V["EXPERIMENT"] / "verdict.json"),
    "capacity_source": str(CAPACITY_SOURCE),
    "fold_source": str(fold_path),
    "train_wells": len(train_records),
    "valid_wells": len(valid_records),
    "train_rows": int(sum(record.eval_rows for record in train_records)),
    "valid_rows": int(sum(record.eval_rows for record in valid_records)),
    "teacher_rmse_max": float(build_report["teacher_rmse"].max()),
    "axis_coverage_min": float(build_report["axis_coverage"].min()),
    "sample_unit": "one complete well",
    "validation_mask": "untouched official TVT_input mask",
    "loss_weighting": "eval-row weighted pooled SSE surrogate",
    "uses_neighbor": False,
    "uses_historical_prediction": False,
}
print("ONEFOLD CONFIG", json.dumps(guards, indent=2), flush=True)
if guards["teacher_rmse_max"] > 0.35 or guards["axis_coverage_min"] < 0.999:
    raise RuntimeError(f"One-fold representation guard failed: {guards}")

anchor_metrics = _anchor_baseline(valid_records)
print("ANCHOR BASELINE", anchor_metrics, flush=True)

model, best, history, training_minutes = _train(
    train_records,
    valid_records,
    device,
)

full_metrics, full_bywell, row_payload = _evaluate(
    model,
    valid_records,
    device,
    mode="full",
    save_rows=True,
)
full_bywell.to_csv(OUT_DIR / "fold1_bywell.csv", index=False)
np.savez_compressed(OUT_DIR / "fold1_oof.npz", **row_payload)

control_rows = []
for mode in ("full", "no_gr", "shuffled_typewell", "shuffled_heel"):
    metrics, _, _ = _evaluate(model, valid_records, device, mode=mode)
    control_rows.append(
        {
            "mode": mode,
            **metrics,
            "degradation_vs_full": metrics["rmse"] - full_metrics["rmse"],
        }
    )
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "causal_controls.csv", index=False)

no_gr_degradation = float(
    controls.loc[controls["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
typewell_degradation = float(
    controls.loc[
        controls["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)
heel_degradation = float(
    controls.loc[
        controls["mode"] == "shuffled_heel", "degradation_vs_full"
    ].iloc[0]
)

architecture_pass = bool(full_metrics["rmse"] <= ARCHITECTURE_GATE_FT)
target_pass = bool(full_metrics["rmse"] <= TARGET_GATE_FT)
causal_pass = bool(
    no_gr_degradation >= NO_GR_DEGRADATION_GATE_FT
    and typewell_degradation >= SHUFFLED_TYPEWELL_DEGRADATION_GATE_FT
)
strict_pass = bool(architecture_pass and causal_pass)

verdict = {
    "experiment": EXPERIMENT,
    "standalone": True,
    "sample_unit": "one complete well",
    "legal_test_inputs_only": True,
    "fold": VALID_FOLD,
    "guards": guards,
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "best_epoch": int(best["epoch"]),
    "anchor_baseline": anchor_metrics,
    "result": full_metrics,
    "causal_controls": {
        "no_gr_degradation": no_gr_degradation,
        "shuffled_typewell_degradation": typewell_degradation,
        "shuffled_heel_degradation": heel_degradation,
    },
    "gates": {
        "architecture_rmse_le": ARCHITECTURE_GATE_FT,
        "target_rmse_le": TARGET_GATE_FT,
        "no_gr_degradation_ge": NO_GR_DEGRADATION_GATE_FT,
        "shuffled_typewell_degradation_ge": SHUFFLED_TYPEWELL_DEGRADATION_GATE_FT,
    },
    "architecture_pass": architecture_pass,
    "target_5p5_pass": target_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "training_minutes": training_minutes,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "TARGET PASS -> freeze recipe and run all five folds"
        if target_pass and causal_pass
        else (
            "ARCHITECTURE PASS -> diagnose fold1 residuals, then tune only the native whole-well model"
            if strict_pass
            else "FAIL -> do not scale; inspect native-resolution generalisation gap and causal controls"
        )
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nTRAINING HISTORY")
print(history.to_string(index=False), flush=True)
print("\nCAUSAL CONTROLS")
print(controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR, flush=True)
