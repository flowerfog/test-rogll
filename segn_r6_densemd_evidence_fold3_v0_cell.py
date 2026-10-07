"""SEGN-R6: dense-MD work-note evidence observability gateway.

R5 used one cost-map column every 32 ft of measured depth.  The SEGN work note
instead defines a candidate cost map at every measured-depth step.  R5D proved
that the coarse model's logits cannot be rescued by temperature calibration.
This experiment therefore changes only the observation resolution:

* teacher-anchored chunks remain about 512 ft long and overlap every 256 ft;
* cost-map columns and labels are sampled every 4 ft of MD (about 128/chunk);
* the R5 axis-specific, query-causal bilinear/cosine evidence model is trained
  from scratch; no prior, recurrence, PF, spatial path, or strong base is used;
* posterior mean remains primary, while mode and matched controls are reported.

Folds 4/5 train, 32 deterministic Fold-3 wells select the epoch, and the other
122 Fold-3 wells confirm it.  Folds 1/2 are not loaded.  This is deliberately
an evidence test, not a deployable recursive decoder.
"""

from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SOURCE_CANDIDATES_R6 = (
    Path("/root/autodl-tmp/segn_r5_axisaware_evidence_fold3_v0_cell.py"),
    Path.cwd() / "segn_r5_axisaware_evidence_fold3_v0_cell.py",
)
SOURCE_PATH_R6 = next((path for path in SOURCE_CANDIDATES_R6 if path.exists()), None)
if SOURCE_PATH_R6 is None:
    raise FileNotFoundError(
        "Upload segn_r5_axisaware_evidence_fold3_v0_cell.py to "
        "/root/autodl-tmp before running SEGN-R6."
    )

source_r6 = SOURCE_PATH_R6.read_text(encoding="utf-8")
marker_r6 = "\nstarted_r5 = time.time()"
if marker_r6 not in source_r6:
    raise RuntimeError(f"Cannot find bootstrap marker in {SOURCE_PATH_R6}")
exec(
    compile(source_r6.split(marker_r6, 1)[0], str(SOURCE_PATH_R6), "exec"),
    globals(),
)


EXPERIMENT_R6 = "segn_r6_densemd_evidence_fold3_v0"
OUT_DIR_R6 = OOF_ROOT / EXPERIMENT_R6
MODEL_DIR_R6 = OUT_DIR_R6 / "models"
OUT_DIR_R6.mkdir(parents=True, exist_ok=True)
MODEL_DIR_R6.mkdir(parents=True, exist_ok=True)
# Bootstrapped record loaders write failure reports through this global.
OUT_DIR = OUT_DIR_R6

TRAIN_FOLDS_R6 = (4, 5)
VALID_FOLD_R6 = 3
SELECTION_WELLS_R6 = 32
DENSE_MD_SPACING_FT_R6 = 4.0
CHUNK_TARGET_MD_FT_R6 = 512.0
TRAIN_CHUNKS_PER_WELL_R6 = 3
MAX_EPOCHS_R6 = 24
MIN_EPOCHS_R6 = 12
REPORT_EVERY_R6 = 4
PATIENCE_REPORTS_R6 = 3
GRADIENT_ACCUMULATION_R6 = 6
LEARNING_RATE_R6 = 3.0e-4
MIN_LEARNING_RATE_R6 = 3.0e-5
WEIGHT_DECAY_R6 = 2.0e-4
GRADIENT_CLIP_R6 = 2.0
TARGET_SIGMA_R6 = 2.0
LOSS_MEAN_R6 = 0.15
LOSS_CHANNEL_R6 = 0.08


def _dense_rows_r6(record, start):
    """Nearest native eval rows on a deterministic 4-ft MD grid."""

    start = int(start)
    start_row = int(record.node_rows[start])
    start_md = float(record.md[start_row])
    toe_md = float(record.md[record.eval_index[-1]])
    end_md = min(start_md + CHUNK_TARGET_MD_FT_R6, toe_md)
    candidates = record.eval_index[
        (record.md[record.eval_index] > start_md)
        & (record.md[record.eval_index] <= end_md + 1.0e-4)
    ]
    if not len(candidates):
        return np.empty(0, np.int32)
    candidate_md = record.md[candidates].astype(np.float64)
    targets = np.arange(
        start_md + DENSE_MD_SPACING_FT_R6,
        end_md + 0.5 * DENSE_MD_SPACING_FT_R6,
        DENSE_MD_SPACING_FT_R6,
        dtype=np.float64,
    )
    if not len(targets):
        targets = np.asarray([end_md], np.float64)
    positions = np.searchsorted(candidate_md, targets)
    rows = []
    for target, position in zip(targets, positions):
        choices = []
        if position > 0:
            choices.append(int(position - 1))
        if position < len(candidates):
            choices.append(int(position))
        if not choices:
            continue
        choice = min(choices, key=lambda index: abs(candidate_md[index] - target))
        row = int(candidates[choice])
        if row > start_row and (not rows or row > rows[-1]):
            rows.append(row)
    final_row = int(candidates[-1])
    if final_row > start_row and (not rows or final_row > rows[-1]):
        rows.append(final_row)
    return np.asarray(rows, np.int32)


def _dense_payload_r6(
    record,
    start,
    control="full",
    shuffled_typewell=None,
):
    start = int(start)
    start_row = int(record.node_rows[start])
    target_rows = _dense_rows_r6(record, start)
    if not len(target_rows):
        raise RuntimeError(f"{record.well}: empty dense chunk at {start}")
    start_md = float(record.md[start_row])
    end_md = float(record.md[target_rows[-1]])
    crop_start = int(
        np.searchsorted(
            record.md, start_md - CHUNK_CONTEXT_MD_FT, side="left"
        )
    )
    crop_end = int(
        np.searchsorted(
            record.md, end_md + CHUNK_CONTEXT_MD_FT, side="right"
        )
    )
    crop_end = max(crop_end, int(target_rows[-1]) + 1)
    horizontal = record.horizontal[:, crop_start:crop_end].copy()
    if control == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
    typewell = (
        shuffled_typewell.typewell.copy()
        if control == "shuffled_typewell" and shuffled_typewell is not None
        else record.typewell.copy()
    )
    deployed_anchor = float(record.truth[start_row])
    anchor_offset = deployed_anchor - float(record.anchor_tvt)
    condition = record.condition
    geometry = np.stack(
        [
            np.clip(condition[2, target_rows], 0.0, 1.0),
            np.clip(condition[3, target_rows], -1.0, 1.0),
            np.clip(condition[4, target_rows], -4.0, 4.0),
            np.clip(condition[7, target_rows], -2.0, 2.0),
            np.clip(condition[8, target_rows], -2.0, 2.0),
            np.clip(condition[9, target_rows], 0.0, 1.0),
            np.zeros(len(target_rows), np.float32),
            np.zeros(len(target_rows), np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "horizontal": horizontal,
        "typewell": typewell,
        "node_positions": (target_rows - crop_start).astype(np.int64),
        "candidate_positions": (anchor_offset + STATE_OFFSETS).astype(np.float32),
        "geometry": geometry,
        "target_state": (
            record.truth[target_rows].astype(np.float32) - deployed_anchor
        ),
        "target_rows": target_rows,
        "deployed_anchor": deployed_anchor,
        "distance": start_md - float(record.md[record.anchor_index]),
    }


def _dense_axis_coverage_r6(records):
    covered = 0
    total = 0
    minimum_columns = math.inf
    maximum_columns = 0
    for record in records:
        for start in record.chunk_starts:
            rows = _dense_rows_r6(record, int(start))
            if not len(rows):
                continue
            target = record.truth[rows] - record.truth[record.node_rows[int(start)]]
            covered += int(np.sum(np.abs(target) <= STATE_HALF_WIDTH_FT - 0.5))
            total += len(target)
            minimum_columns = min(minimum_columns, len(target))
            maximum_columns = max(maximum_columns, len(target))
    return {
        "coverage": float(covered / max(total, 1)),
        "columns": int(total),
        "minimum_chunk_columns": int(minimum_columns if total else 0),
        "maximum_chunk_columns": int(maximum_columns),
    }


def _loss_r6(model, payload, device):
    tensors = _tensor_payload(payload, device)
    fused, auxiliary = model(*tensors, return_aux=True)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=target.dtype)
    target_distribution = torch.exp(
        -0.5 * ((states[:, None] - target[None]) / TARGET_SIGMA_R6) ** 2
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=0, keepdim=True
    ).clamp_min(1.0e-8)
    fused_ce = -(
        target_distribution * F.log_softmax(fused, dim=0)
    ).sum(dim=0).mean()
    probability = F.softmax(fused, dim=0)
    expected = (probability * states[:, None]).sum(dim=0)
    mean_loss = F.smooth_l1_loss(expected / 8.0, target / 8.0)
    channel_ce = torch.stack(
        [
            -(
                target_distribution * F.log_softmax(values, dim=0)
            ).sum(dim=0).mean()
            for values in auxiliary["maps"]
        ]
    ).mean()
    total = fused_ce + LOSS_MEAN_R6 * mean_loss + LOSS_CHANNEL_R6 * channel_ce
    return total, {
        "fused_ce": float(fused_ce.detach()),
        "mean": float(mean_loss.detach()),
        "channel": float(channel_ce.detach()),
        "gate": auxiliary["gate_weights"].detach(),
    }


def _training_starts_r6(record, rng):
    starts = np.asarray(record.chunk_starts, np.int32)
    if len(starts) <= TRAIN_CHUNKS_PER_WELL_R6:
        return starts
    boundaries = np.linspace(0, len(starts), TRAIN_CHUNKS_PER_WELL_R6 + 1)
    selected = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        lo = int(math.floor(left))
        hi = max(lo + 1, int(math.ceil(right)))
        index = int(rng.integers(lo, min(hi, len(starts))))
        selected.append(int(starts[index]))
    return np.asarray(selected, np.int32)


def _train_epoch_r6(model, records, optimizer, scaler, device, epoch):
    model.train()
    rng = np.random.default_rng(SEED + 6151 * epoch)
    order = rng.permutation(len(records))
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "fused_ce": 0.0,
        "mean": 0.0,
        "channel": 0.0,
        "gate_bilinear": 0.0,
        "gate_cosine": 0.0,
        "samples": 0,
        "columns": 0,
        "skipped": 0,
        "optimizer_steps": 0,
    }
    for progress, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        for start in _training_starts_r6(record, rng):
            payload = _dense_payload_r6(record, int(start))
            target = payload["target_state"]
            if np.max(np.abs(target)) > STATE_HALF_WIDTH_FT - 0.5:
                totals["skipped"] += 1
                continue
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, parts = _loss_r6(model, payload, device)
                scaled_loss = loss / GRADIENT_ACCUMULATION_R6
            if not torch.isfinite(loss):
                totals["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(scaled_loss).backward()
            totals["loss"] += float(loss.detach())
            for key in ("fused_ce", "mean", "channel"):
                totals[key] += parts[key]
            gate = parts["gate"].float().cpu().numpy()
            totals["gate_bilinear"] += float(gate[0].mean())
            totals["gate_cosine"] += float(gate[1].mean())
            totals["samples"] += 1
            totals["columns"] += len(target)
            if totals["samples"] % GRADIENT_ACCUMULATION_R6 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R6)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                totals["optimizer_steps"] += 1
        if progress % 100 == 0 or progress == len(order):
            print(
                f"SEGN-R6 train epoch={epoch} wells={progress}/{len(order)} "
                f"loss={totals['loss'] / max(totals['samples'], 1):.4f} "
                f"columns={totals['columns']}",
                flush=True,
            )
    if totals["samples"] % GRADIENT_ACCUMULATION_R6:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R6)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        totals["optimizer_steps"] += 1
    denominator = max(totals["samples"], 1)
    return {
        **{
            key: totals[key] / denominator
            for key in (
                "loss",
                "fused_ce",
                "mean",
                "channel",
                "gate_bilinear",
                "gate_cosine",
            )
        },
        "samples": totals["samples"],
        "columns": totals["columns"],
        "skipped": totals["skipped"],
        "optimizer_steps": totals["optimizer_steps"],
    }


@torch.no_grad()
def _dense_posterior_r6(model, payload, device):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, auxiliary = model(*tensors, return_aux=True)
    logits = logits.float()
    probability = F.softmax(logits, dim=0)
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=torch.float32)
    expected = (probability * states[:, None]).sum(dim=0)
    return {
        "logits": logits.cpu().numpy().astype(np.float64),
        "probability": probability.cpu().numpy().astype(np.float64),
        "expected": expected.cpu().numpy().astype(np.float64),
        "mode": STATE_OFFSETS[torch.argmax(probability, dim=0).cpu().numpy()].astype(
            np.float64
        ),
        "valid": auxiliary["candidate_valid"].cpu().numpy().astype(bool),
        "gate": auxiliary["gate_weights"].float().cpu().numpy().astype(np.float64),
    }


@torch.no_grad()
def _evaluate_r6(model, records, device, control="full", save_prefix=None):
    model.eval()
    wells = []
    truths = []
    mean_predictions = []
    mode_predictions = []
    distances = []
    anchors = []
    local_target = []
    local_expected = []
    local_mode = []
    local_mass5 = []
    local_rank = []
    local_nll = []
    bywell_rows = []
    gate_sum = np.zeros(2, np.float64)
    gate_columns = 0
    shuffled = (
        list(records[1:]) + list(records[:1])
        if len(records) > 1
        else list(records)
    )

    for number, (record, shuffled_record) in enumerate(zip(records, shuffled), 1):
        row_count = len(record.md)
        mean_sum = np.zeros(row_count, np.float64)
        mode_sum = np.zeros(row_count, np.float64)
        count = np.zeros(row_count, np.float64)
        mean_sum[record.anchor_index] = float(record.anchor_tvt)
        mode_sum[record.anchor_index] = float(record.anchor_tvt)
        count[record.anchor_index] = 1.0
        well_target = []
        well_expected = []

        for raw_start in record.chunk_starts:
            payload = _dense_payload_r6(
                record,
                int(raw_start),
                control=control,
                shuffled_typewell=shuffled_record,
            )
            posterior = _dense_posterior_r6(model, payload, device)
            rows = payload["target_rows"]
            target = payload["target_state"].astype(np.float64)
            anchor = float(payload["deployed_anchor"])
            mean_sum[rows] += anchor + posterior["expected"]
            mode_sum[rows] += anchor + posterior["mode"]
            count[rows] += 1.0
            local_target.extend(target.tolist())
            local_expected.extend(posterior["expected"].tolist())
            local_mode.extend(posterior["mode"].tolist())
            well_target.extend(target.tolist())
            well_expected.extend(posterior["expected"].tolist())

            state_distance = np.abs(STATE_OFFSETS[:, None] - target[None])
            within5 = state_distance <= 5.0
            local_mass5.extend(
                np.sum(posterior["probability"] * within5, axis=0).tolist()
            )
            true_index = np.argmin(state_distance, axis=0)
            columns = np.arange(len(target))
            true_scores = posterior["logits"][true_index, columns]
            better = np.sum(
                posterior["valid"][:, None]
                & (posterior["logits"] > true_scores[None] + 1.0e-8),
                axis=0,
            )
            tied = np.sum(
                posterior["valid"][:, None]
                & (np.abs(posterior["logits"] - true_scores[None]) <= 1.0e-8),
                axis=0,
            )
            local_rank.extend((1.0 + better + 0.5 * np.maximum(tied - 1, 0)).tolist())
            local_nll.extend(
                (
                    -np.log(
                        np.maximum(
                            posterior["probability"][true_index, columns],
                            1.0e-12,
                        )
                    )
                ).tolist()
            )
            gate_sum += posterior["gate"].sum(axis=1)
            gate_columns += posterior["gate"].shape[1]

        supported = count > 0
        support_md = record.md[supported].astype(np.float64)
        mean_values = mean_sum[supported] / count[supported]
        mode_values = mode_sum[supported] / count[supported]
        eval_rows = record.eval_index
        query_md = record.md[eval_rows].astype(np.float64)
        mean_prediction = np.interp(query_md, support_md, mean_values)
        mode_prediction = np.interp(query_md, support_md, mode_values)
        truth = record.truth[eval_rows].astype(np.float64)
        distance = query_md - float(record.md[record.anchor_index])
        wells.append(record.well)
        truths.append(truth)
        mean_predictions.append(mean_prediction)
        mode_predictions.append(mode_prediction)
        distances.append(distance)
        anchors.append(float(record.anchor_tvt))
        bywell_rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "mean_rmse": float(np.sqrt(np.mean((mean_prediction - truth) ** 2))),
                "mode_rmse": float(np.sqrt(np.mean((mode_prediction - truth) ** 2))),
                "local_expected_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                np.asarray(well_expected, np.float64)
                                - np.asarray(well_target, np.float64)
                            )
                            ** 2
                        )
                    )
                ),
            }
        )
        if number % 25 == 0 or number == len(records):
            print(f"SEGN-R6 evaluate {control} {number}/{len(records)}", flush=True)

    mean_metrics = _metric_payload_r1(wells, truths, mean_predictions, distances)
    mean_metrics.update(_shape_metrics_r4(wells, truths, mean_predictions, anchors))
    mode_metrics = _metric_payload_r1(wells, truths, mode_predictions, distances)
    target_array = np.asarray(local_target, np.float64)
    expected_array = np.asarray(local_expected, np.float64)
    mode_array = np.asarray(local_mode, np.float64)
    local_metrics = {
        "columns": int(len(target_array)),
        "expected_rmse": float(np.sqrt(np.mean((expected_array - target_array) ** 2))),
        "mode_rmse": float(np.sqrt(np.mean((mode_array - target_array) ** 2))),
        "expected_correlation": (
            float(np.corrcoef(expected_array, target_array)[0, 1])
            if np.std(expected_array) > 1.0e-12
            else math.nan
        ),
        "mode_within5": float(np.mean(np.abs(mode_array - target_array) <= 5.0)),
        "posterior_mass_within5": float(np.mean(local_mass5)),
        "median_truth_rank": float(np.median(local_rank)),
        "p90_truth_rank": float(np.quantile(local_rank, 0.90)),
        "mean_true_state_nll": float(np.mean(local_nll)),
    }
    gate_mean = gate_sum / max(gate_columns, 1)
    diagnostics = {
        "gate_bilinear": float(gate_mean[0]),
        "gate_cosine": float(gate_mean[1]),
        "posterior_mean_gain_vs_mode": float(mode_metrics["rmse"] - mean_metrics["rmse"]),
    }
    bywell = pd.DataFrame(bywell_rows)
    if save_prefix:
        bywell.to_csv(OUT_DIR_R6 / f"{save_prefix}_bywell.csv", index=False)
        np.savez_compressed(
            OUT_DIR_R6 / f"{save_prefix}_rows.npz",
            well=np.concatenate(
                [np.repeat(well, len(values)) for well, values in zip(wells, truths)]
            ).astype("U"),
            truth=np.concatenate(truths).astype(np.float32),
            prediction=np.concatenate(mean_predictions).astype(np.float32),
            distance_past_anchor=np.concatenate(distances).astype(np.float32),
        )
    return {
        "path_mean": mean_metrics,
        "path_mode": mode_metrics,
        "local": local_metrics,
        "diagnostics": diagnostics,
    }, bywell


def _fit_r6(train_records, selection_records, device):
    _seed_everything(SEED + 6161)
    model = AxisAwareEvidenceR5().to(device)
    zero_guard = _query_zero_guard_r5(model, device)
    if not zero_guard["finite"] or zero_guard["maximum_no_gr_logit_std"] > 1.0e-6:
        raise RuntimeError(f"R6 query-zero guard failed before training: {zero_guard}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_R6, weight_decay=WEIGHT_DECAY_R6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_R6, eta_min=MIN_LEARNING_RATE_R6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = math.inf
    bad_reports = 0
    history = []
    started_fit = time.time()
    for epoch in range(1, MAX_EPOCHS_R6 + 1):
        training = _train_epoch_r6(model, train_records, optimizer, scaler, device, epoch)
        scheduler.step()
        report = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started_fit) / 60.0,
        }
        if epoch % REPORT_EVERY_R6 == 0 or epoch == MAX_EPOCHS_R6:
            selection, _ = _evaluate_r6(model, selection_records, device)
            report.update(
                {
                    "selection_local_expected_rmse": selection["local"]["expected_rmse"],
                    "selection_local_expected_correlation": selection["local"][
                        "expected_correlation"
                    ],
                    "selection_path_mean_rmse": selection["path_mean"]["rmse"],
                    "selection_shape_correlation": selection["path_mean"][
                        "within_well_shape_correlation"
                    ],
                    "selection_mass_within5": selection["local"][
                        "posterior_mass_within5"
                    ],
                }
            )
            score = float(selection["local"]["expected_rmse"])
            if score < best_score - 1.0e-4:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_reports = 0
            else:
                bad_reports += 1
        history.append(report)
        pd.DataFrame(history).to_csv(OUT_DIR_R6 / "training_history.csv", index=False)
        print("SEGN-R6 EPOCH", report, flush=True)
        if epoch >= MIN_EPOCHS_R6 and bad_reports >= PATIENCE_REPORTS_R6:
            print(f"SEGN-R6 early stopping at epoch {epoch}", flush=True)
            break
    if best_epoch == 0:
        raise RuntimeError("SEGN-R6 produced no selection checkpoint")
    model.load_state_dict(best_state)
    checkpoint = MODEL_DIR_R6 / "best_densemd_evidence.pt"
    torch.save(
        {
            "model": best_state,
            "epoch": best_epoch,
            "selection_local_expected_rmse": best_score,
            "state_offsets": STATE_OFFSETS,
            "dense_md_spacing_ft": DENSE_MD_SPACING_FT_R6,
            "source": str(SOURCE_PATH_R6),
        },
        checkpoint,
    )
    return model, best_epoch, best_score, zero_guard, checkpoint


started_r6 = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R6")
device_r6 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r6, fold_path_r6 = _load_folds()
train_wells_r6 = folds_r6.loc[folds_r6["fold"].isin(TRAIN_FOLDS_R6), "well"].tolist()
fold3_wells_r6 = folds_r6.loc[folds_r6["fold"].eq(VALID_FOLD_R6), "well"].tolist()
if len(train_wells_r6) != 309 or len(fold3_wells_r6) != 154:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3 sizes: {len(train_wells_r6)}/{len(fold3_wells_r6)}"
    )
records_r6 = {
    record.well: record
    for record in _load_records(sorted(set(train_wells_r6 + fold3_wells_r6)), "r6_fold345")
}
train_records_r6 = [records_r6[well] for well in train_wells_r6]
fold3_records_r6 = [records_r6[well] for well in fold3_wells_r6]
order_r6 = np.argsort([record.eval_rows for record in fold3_records_r6])
selection_positions_r6 = np.unique(
    np.rint(np.linspace(0, len(order_r6) - 1, SELECTION_WELLS_R6)).astype(int)
)
selection_indices_r6 = set(order_r6[selection_positions_r6].tolist())
selection_records_r6 = [
    record for index, record in enumerate(fold3_records_r6) if index in selection_indices_r6
]
confirmation_records_r6 = [
    record for index, record in enumerate(fold3_records_r6) if index not in selection_indices_r6
]

coverage_r6 = _dense_axis_coverage_r6(fold3_records_r6)
guards_r6 = {
    "source": str(SOURCE_PATH_R6),
    "fold_source": str(fold_path_r6),
    "train_folds": list(TRAIN_FOLDS_R6),
    "train_wells": len(train_records_r6),
    "selection_fold": VALID_FOLD_R6,
    "selection_wells": len(selection_records_r6),
    "confirmation_wells": len(confirmation_records_r6),
    "fold2_touched": False,
    "fold1_touched": False,
    "dense_md_spacing_ft": DENSE_MD_SPACING_FT_R6,
    "chunk_target_md_ft": CHUNK_TARGET_MD_FT_R6,
    "coarse_columns_per_full_chunk": CHUNK_TARGET_NODES,
    "dense_columns_per_full_chunk": int(CHUNK_TARGET_MD_FT_R6 / DENSE_MD_SPACING_FT_R6),
    "target_axis": coverage_r6,
    "parameters": int(sum(p.numel() for p in AxisAwareEvidenceR5().parameters())),
}
print("SEGN-R6 GUARDS", json.dumps(guards_r6, indent=2), flush=True)
if coverage_r6["coverage"] < 0.99 or coverage_r6["minimum_chunk_columns"] < 1:
    raise RuntimeError(f"SEGN-R6 dense target guard failed: {coverage_r6}")

model_r6, best_epoch_r6, best_score_r6, zero_guard_r6, checkpoint_r6 = _fit_r6(
    train_records_r6, selection_records_r6, device_r6
)
confirmation_full_r6, _ = _evaluate_r6(
    model_r6,
    confirmation_records_r6,
    device_r6,
    control="full",
    save_prefix="fold3_confirmation_full",
)
confirmation_no_gr_r6, _ = _evaluate_r6(
    model_r6, confirmation_records_r6, device_r6, control="no_gr"
)
confirmation_shuffled_r6, _ = _evaluate_r6(
    model_r6, confirmation_records_r6, device_r6, control="shuffled_typewell"
)
post_zero_guard_r6 = _query_zero_guard_r5(model_r6, device_r6)

local_gain_r6 = (
    confirmation_no_gr_r6["local"]["expected_rmse"]
    - confirmation_full_r6["local"]["expected_rmse"]
)
path_gain_r6 = (
    confirmation_no_gr_r6["path_mean"]["rmse"]
    - confirmation_full_r6["path_mean"]["rmse"]
)
shuffled_local_degradation_r6 = (
    confirmation_shuffled_r6["local"]["expected_rmse"]
    - confirmation_full_r6["local"]["expected_rmse"]
)
shuffled_path_degradation_r6 = (
    confirmation_shuffled_r6["path_mean"]["rmse"]
    - confirmation_full_r6["path_mean"]["rmse"]
)
checks_r6 = {
    "target_axis_coverage_ge_0p99": coverage_r6["coverage"] >= 0.99,
    "query_zero_logit_std_le_1e_6": post_zero_guard_r6[
        "maximum_no_gr_logit_std"
    ]
    <= 1.0e-6,
    "local_expected_gain_vs_uniform_ge_0p20": local_gain_r6 >= 0.20,
    "path_mean_gain_vs_uniform_ge_0p20": path_gain_r6 >= 0.20,
    "local_expected_correlation_ge_0p20": confirmation_full_r6["local"][
        "expected_correlation"
    ]
    >= 0.20,
    "shuffled_local_degradation_ge_0p30": shuffled_local_degradation_r6 >= 0.30,
    "shuffled_path_degradation_ge_0p30": shuffled_path_degradation_r6 >= 0.30,
    "teacher_anchor_path_rmse_le_8p5": confirmation_full_r6["path_mean"]["rmse"]
    <= 8.5,
    "within_well_shape_correlation_ge_0p75": confirmation_full_r6["path_mean"][
        "within_well_shape_correlation"
    ]
    >= 0.75,
    "posterior_mean_better_than_mode": confirmation_full_r6["diagnostics"][
        "posterior_mean_gain_vs_mode"
    ]
    > 0.0,
}
strict_pass_r6 = bool(all(checks_r6.values()))

verdict_r6 = {
    "experiment": EXPERIMENT_R6,
    "stage": "fold3_densemd_axisaware_teacher_anchor_evidence_gateway",
    "sample_unit": "dense teacher-anchored chunks nested within complete wells",
    "deployable": False,
    "truth_usage": "teacher anchors for evidence observability only",
    "legal_model_inputs_only": True,
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R6),
        "selection_fold": VALID_FOLD_R6,
        "selection_wells": len(selection_records_r6),
        "confirmation_wells": len(confirmation_records_r6),
        "fold2_touched": False,
        "fold1_touched": False,
        "selected_epoch": best_epoch_r6,
        "selected_local_expected_rmse": best_score_r6,
    },
    "architecture": {
        "matching_channels": list(SLOT_NAMES_R5),
        "query_axis": "dense MD",
        "reference_axis": "TVT",
        "query_spacing_md_ft": DENSE_MD_SPACING_FT_R6,
        "query_receptive_field_md_ft": 9 + 6 * sum(QUERY_DILATIONS_R5),
        "reference_receptive_field_tvt_ft": 9 + 6 * sum(REFERENCE_DILATIONS_R5),
        "query_causal_by_construction": True,
        "prior_present": False,
        "recurrence_present": False,
        "primary_readout": "posterior mean",
    },
    "guards": guards_r6,
    "query_zero_guard_before_training": zero_guard_r6,
    "query_zero_guard_after_training": post_zero_guard_r6,
    "confirmation": {
        "full": confirmation_full_r6,
        "uniform_teacher_anchor": confirmation_no_gr_r6,
        "shuffled_typewell": confirmation_shuffled_r6,
    },
    "diagnostics": {
        "local_gain_vs_uniform": local_gain_r6,
        "path_gain_vs_uniform": path_gain_r6,
        "shuffled_local_degradation": shuffled_local_degradation_r6,
        "shuffled_path_degradation": shuffled_path_degradation_r6,
    },
    "checks": checks_r6,
    "strict_pass": strict_pass_r6,
    "checkpoint": str(checkpoint_r6),
    "elapsed_minutes": (time.time() - started_r6) / 60.0,
    "next": (
        "PASS -> calibrate the frozen dense evidence and transplant it into scheduled chunk recurrence"
        if strict_pass_r6
        else "FAIL -> dense sampling does not rescue bilinear/cosine evidence; close this evidence family"
    ),
}
_write_json(OUT_DIR_R6 / "verdict.json", verdict_r6)
print("SEGN-R6 FINAL VERDICT", json.dumps(verdict_r6, indent=2), flush=True)
print("saved:", OUT_DIR_R6, flush=True)

if not strict_pass_r6:
    raise RuntimeError(json.dumps(verdict_r6, indent=2))
