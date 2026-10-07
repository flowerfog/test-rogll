"""SEGN-R2 absolute matching gateway with controlled anchor jitter.

R1's geometry prior reached 4.585 ft under true chunk anchors but drifted to
19.656 ft in recursive rollout.  This experiment removes geometry/prior/gate
from the prediction being scored and trains only the bilinear and multi-head
cosine GR/typewell maps to undo hidden anchor perturbations.  The perturbation
value is never a model input.

Folds 4/5 train.  Thirty-two length-stratified Fold-3 wells select the epoch;
the remaining 122 wells confirm it with matched no-GR and shuffled-typewell
controls.  Fold 1 and Fold 2 are never loaded.  This is still an emission
gateway, not a recursive model.
"""

from pathlib import Path
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE_CANDIDATES_R2 = (
    Path("/root/autodl-tmp/segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py"),
    Path.cwd() / "segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py",
)
SOURCE_PATH_R2 = next((path for path in SOURCE_CANDIDATES_R2 if path.exists()), None)
if SOURCE_PATH_R2 is None:
    raise FileNotFoundError(
        "Upload segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py to "
        "/root/autodl-tmp before running SEGN-R2."
    )

source_r2 = SOURCE_PATH_R2.read_text(encoding="utf-8")
marker_r2 = "\nstarted_r1 = time.time()"
if marker_r2 not in source_r2:
    raise RuntimeError(f"Cannot find gateway marker in {SOURCE_PATH_R2}")
exec(
    compile(source_r2.split(marker_r2, 1)[0], str(SOURCE_PATH_R2), "exec"),
    globals(),
)


EXPERIMENT_R2 = "segn_r2_absolute_matching_jitter_gateway_v0"
OUT_DIR_R2 = OOF_ROOT / EXPERIMENT_R2
MODEL_DIR_R2 = OUT_DIR_R2 / "models"
OUT_DIR_R2.mkdir(parents=True, exist_ok=True)
MODEL_DIR_R2.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS_R2 = (4, 5)
VALID_FOLD_R2 = 3
SELECTION_WELLS_R2 = 32
JITTERS_R2 = (-48.0, -24.0, -12.0, 0.0, 12.0, 24.0, 48.0)
TRAIN_JITTERS_R2 = np.asarray((-48.0, -24.0, -12.0, 0.0, 0.0, 0.0, 12.0, 24.0, 48.0))
ALIAS_OFFSETS_R2 = (-24.0, 24.0)

MAX_EPOCHS_R2 = 36
MIN_EPOCHS_R2 = 16
REPORT_EVERY_R2 = 4
PATIENCE_REPORTS_R2 = 4
TRAIN_CHUNKS_PER_WELL_R2 = 3
GRADIENT_ACCUMULATION_R2 = 6
LEARNING_RATE_R2 = 2.0e-4
MIN_LEARNING_RATE_R2 = 2.0e-5
WEIGHT_DECAY_R2 = 1.0e-4
GRADIENT_CLIP_R2 = 2.0
TARGET_SIGMA_R2 = 2.0
LOSS_MEAN_R2 = 0.10
LOSS_ALIAS_R2 = 0.15
ALIAS_MARGIN_R2 = 1.0


class AbsoluteMatcherR2(nn.Module):
    """Only GR/typewell matching maps contribute to the output."""

    def __init__(self, warm_state=None):
        super().__init__()
        self.backbone = FiveSlotCostMapR1()
        if warm_state is not None:
            self.backbone.load_state_dict(warm_state)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        trainable_prefixes = (
            "encoder.",
            "query_bilinear.",
            "reference_bilinear.",
            "bilinear_matrix",
            "query_heads.",
            "reference_heads.",
            "head_weights",
            "slot_log_scale",
        )
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

    def forward(self, *tensors):
        _, auxiliary = self.backbone(*tensors, return_aux=True)
        weight = torch.sigmoid(self.mix_logit)
        logits = weight * auxiliary["slot_maps"][0] + (1.0 - weight) * auxiliary["slot_maps"][2]
        return logits, weight


def _loss_r2(model, payload, device):
    tensors = _tensor_payload(payload, device)
    logits, mix_weight = model(*tensors)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=target.dtype)
    target_distribution = torch.exp(
        -0.5 * ((states[:, None] - target[None]) / TARGET_SIGMA_R2) ** 2
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=0, keepdim=True
    ).clamp_min(1.0e-8)
    log_probability = F.log_softmax(logits, dim=0)
    ce = -(target_distribution * log_probability).sum(dim=0).mean()
    probability = F.softmax(logits, dim=0)
    expected = (probability * states[:, None]).sum(dim=0)
    mean_loss = F.smooth_l1_loss(expected / 8.0, target / 8.0)

    columns = torch.arange(len(target), device=device)
    true_index = torch.argmin(torch.abs(states[:, None] - target[None]), dim=0)
    true_score = logits[true_index, columns]
    alias_scores = []
    for alias_offset in ALIAS_OFFSETS_R2:
        alias_target = target + float(alias_offset)
        alias_index = torch.argmin(
            torch.abs(states[:, None] - alias_target[None]), dim=0
        )
        alias_scores.append(logits[alias_index, columns])
    hardest_alias = torch.stack(alias_scores, dim=0).max(dim=0).values
    alias_loss = F.relu(ALIAS_MARGIN_R2 - true_score + hardest_alias).mean()
    total = ce + LOSS_MEAN_R2 * mean_loss + LOSS_ALIAS_R2 * alias_loss
    return total, {
        "ce": float(ce.detach()),
        "mean": float(mean_loss.detach()),
        "alias": float(alias_loss.detach()),
        "bilinear_weight": float(mix_weight.detach()),
    }


def _training_starts_r2(record, rng):
    starts = np.asarray(record.chunk_starts, np.int32)
    if len(starts) <= TRAIN_CHUNKS_PER_WELL_R2:
        return starts
    positions = np.linspace(0, len(starts), TRAIN_CHUNKS_PER_WELL_R2 + 1)
    selected = []
    for left, right in zip(positions[:-1], positions[1:]):
        lo = int(math.floor(left))
        hi = max(lo + 1, int(math.ceil(right)))
        selected.append(int(starts[int(rng.integers(lo, min(hi, len(starts))))]))
    return np.asarray(selected, np.int32)


def _train_epoch_r2(model, records, optimizer, scaler, device, epoch):
    model.train()
    rng = np.random.default_rng(SEED + 104729 * epoch)
    order = rng.permutation(len(records))
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "mean": 0.0,
        "alias": 0.0,
        "bilinear_weight": 0.0,
        "samples": 0,
        "skipped": 0,
        "optimizer_steps": 0,
    }
    amp_enabled = device.type == "cuda"

    for progress, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        for raw_start in _training_starts_r2(record, rng):
            start = int(raw_start)
            start_row = int(record.node_rows[start])
            jitter = float(rng.choice(TRAIN_JITTERS_R2))
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]) + jitter,
                previous_delta=0.0,
                previous_rate=0.0,
            )
            if np.max(np.abs(payload["target_state"])) > STATE_HALF_WIDTH_FT - 0.5:
                totals["skipped"] += 1
                continue
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, parts = _loss_r2(model, payload, device)
                scaled_loss = loss / GRADIENT_ACCUMULATION_R2
            if not torch.isfinite(loss):
                totals["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(scaled_loss).backward()
            totals["loss"] += float(loss.detach())
            for key in ("ce", "mean", "alias", "bilinear_weight"):
                totals[key] += parts[key]
            totals["samples"] += 1
            if totals["samples"] % GRADIENT_ACCUMULATION_R2 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                totals["optimizer_steps"] += 1

        if progress % 100 == 0 or progress == len(records):
            print(
                f"SEGN-R2 epoch={epoch} wells={progress}/{len(records)} "
                f"loss={totals['loss'] / max(totals['samples'], 1):.4f}",
                flush=True,
            )

    if totals["samples"] % GRADIENT_ACCUMULATION_R2:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_R2)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        totals["optimizer_steps"] += 1

    denominator = max(totals["samples"], 1)
    return {
        key: (value / denominator if key not in {"samples", "skipped", "optimizer_steps"} else value)
        for key, value in totals.items()
    }


@torch.no_grad()
def _predict_payload_r2(model, payload, device):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, mix_weight = model(*tensors)
    logits = logits.float()
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=torch.float32)
    probability = F.softmax(logits, dim=0)
    expected = (probability * states[:, None]).sum(dim=0)
    mode = states[torch.argmax(probability, dim=0)]
    return (
        expected.cpu().numpy().astype(np.float64),
        mode.cpu().numpy().astype(np.float64),
        float(mix_weight),
    )


@torch.no_grad()
def _evaluate_offset_r2(
    model,
    records,
    device,
    offset,
    control="full",
):
    model.eval()
    wells = []
    truths = []
    predictions = []
    distances = []
    biases = []
    mix_weights = []
    shuffled = list(records[1:]) + list(records[:1]) if len(records) > 1 else list(records)

    for record, shuffled_record in zip(records, shuffled):
        node_count = len(record.node_rows)
        sums = np.zeros(node_count, np.float64)
        counts = np.zeros(node_count, np.float64)
        sums[0] = float(record.anchor_tvt)
        counts[0] = 1.0
        for raw_start in record.chunk_starts:
            start = int(raw_start)
            start_row = int(record.node_rows[start])
            deployed_anchor = float(record.truth[start_row]) + float(offset)
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=deployed_anchor,
                previous_delta=0.0,
                previous_rate=0.0,
                control=control,
                shuffled_typewell=shuffled_record,
            )
            expected, _, mix_weight = _predict_payload_r2(model, payload, device)
            end = start + len(expected)
            sums[start + 1 : end + 1] += deployed_anchor + expected
            counts[start + 1 : end + 1] += 1.0
            mix_weights.append(mix_weight)

        valid = counts > 0
        eval_rows = record.eval_index
        query_md = record.md[eval_rows].astype(np.float64)
        prediction = np.interp(
            query_md,
            record.md[record.node_rows[valid]].astype(np.float64),
            sums[valid] / counts[valid],
        )
        truth = record.truth[eval_rows].astype(np.float64)
        wells.append(record.well)
        truths.append(truth)
        predictions.append(prediction)
        distances.append(query_md - float(record.md[record.anchor_index]))
        biases.append(float(np.mean(prediction - truth)))

    metrics = _metric_payload_r1(wells, truths, predictions, distances)
    metrics["bias"] = float(
        np.average(biases, weights=[len(values) for values in truths])
    )
    metrics["offset_ft"] = float(offset)
    metrics["bilinear_weight"] = float(np.mean(mix_weights))
    return metrics


def _offset_suite_r2(model, records, device, control="full"):
    rows = []
    for offset in JITTERS_R2:
        metrics = _evaluate_offset_r2(
            model, records, device, offset=offset, control=control
        )
        rows.append(metrics)
        print(
            f"SEGN-R2 {control} offset={offset:+.0f} rmse={metrics['rmse']:.4f}",
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values("offset_ft").reset_index(drop=True)
    jittered = frame[frame["offset_ft"].ne(0.0)]
    jittered_rmse = float(
        np.sqrt(
            jittered["exact_sse"].sum()
            / jittered["rows"].sum()
        )
    )
    zero = frame.loc[frame["offset_ft"].eq(0.0)].iloc[0].to_dict()
    slope = float(np.polyfit(frame["offset_ft"], frame["bias"], 1)[0])
    return {
        "zero": zero,
        "jittered_rmse": jittered_rmse,
        "residual_response_slope": slope,
        "recovery_slope": 1.0 - slope,
        "offsets": frame.to_dict(orient="records"),
    }


def _fit_r2(model, train_records, selection_records, device):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE_R2, weight_decay=WEIGHT_DECAY_R2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_R2, eta_min=MIN_LEARNING_RATE_R2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_objective = math.inf
    bad_reports = 0
    history = []
    started = time.time()

    for epoch in range(1, MAX_EPOCHS_R2 + 1):
        training = _train_epoch_r2(
            model, train_records, optimizer, scaler, device, epoch
        )
        scheduler.step()
        if epoch % REPORT_EVERY_R2 != 0 and epoch != MAX_EPOCHS_R2:
            continue
        validation = _offset_suite_r2(model, selection_records, device, control="full")
        objective = validation["jittered_rmse"] + 0.25 * validation["zero"]["rmse"]
        report = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            "selection_zero_rmse": validation["zero"]["rmse"],
            "selection_jittered_rmse": validation["jittered_rmse"],
            "selection_recovery_slope": validation["recovery_slope"],
            "selection_objective": objective,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(report)
        print("SEGN-R2 EPOCH", report, flush=True)
        if objective < best_objective - 1.0e-5:
            best_objective = objective
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_reports = 0
            torch.save(
                {
                    "model": best_state,
                    "epoch": best_epoch,
                    "selection_objective": best_objective,
                    "jitters_ft": JITTERS_R2,
                    "state_offsets": STATE_OFFSETS,
                },
                MODEL_DIR_R2 / "best_absolute_matcher.pt",
            )
        else:
            bad_reports += 1
        if epoch >= MIN_EPOCHS_R2 and bad_reports >= PATIENCE_REPORTS_R2:
            print("SEGN-R2 early stopping", flush=True)
            break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT_DIR_R2 / "training_history.csv", index=False)
    return model, best_epoch, best_objective


started_r2 = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R2")
device_r2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r1_checkpoint_path_r2 = (
    OOF_ROOT
    / "segn_r1_fiveslot_costmap_fold3_gateway_v0"
    / "models"
    / "best_fold3_selection.pt"
)
if not r1_checkpoint_path_r2.exists():
    raise FileNotFoundError(r1_checkpoint_path_r2)
r1_checkpoint_r2 = torch.load(
    r1_checkpoint_path_r2, map_location=device_r2, weights_only=False
)

folds_r2, fold_path_r2 = _load_folds()
train_wells_r2 = folds_r2.loc[folds_r2["fold"].isin(TRAIN_FOLDS_R2), "well"].tolist()
fold3_wells_r2 = folds_r2.loc[folds_r2["fold"].eq(VALID_FOLD_R2), "well"].tolist()
if len(train_wells_r2) != 309 or len(fold3_wells_r2) != 154:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3 sizes: {len(train_wells_r2)}/{len(fold3_wells_r2)}"
    )
records_r2 = {
    record.well: record
    for record in _load_records(
        sorted(set(train_wells_r2 + fold3_wells_r2)), "r2_fold345"
    )
}
train_records_r2 = [records_r2[well] for well in train_wells_r2]
fold3_records_r2 = [records_r2[well] for well in fold3_wells_r2]

order_r2 = np.argsort([record.eval_rows for record in fold3_records_r2])
selection_positions_r2 = np.unique(
    np.rint(np.linspace(0, len(order_r2) - 1, SELECTION_WELLS_R2)).astype(int)
)
selection_indices_r2 = set(order_r2[selection_positions_r2].tolist())
selection_records_r2 = [
    record for index, record in enumerate(fold3_records_r2) if index in selection_indices_r2
]
confirmation_records_r2 = [
    record for index, record in enumerate(fold3_records_r2) if index not in selection_indices_r2
]

model_r2 = AbsoluteMatcherR2(warm_state=r1_checkpoint_r2["model"]).to(device_r2)
trainable_parameters_r2 = int(
    sum(parameter.numel() for parameter in model_r2.parameters() if parameter.requires_grad)
)
print(
    "SEGN-R2 CONFIG",
    {
        "train_wells": len(train_records_r2),
        "selection_wells": len(selection_records_r2),
        "confirmation_wells": len(confirmation_records_r2),
        "trainable_parameters": trainable_parameters_r2,
        "jitters_ft": JITTERS_R2,
        "fold2_touched": False,
        "fold1_touched": False,
    },
    flush=True,
)

model_r2, best_epoch_r2, best_objective_r2 = _fit_r2(
    model_r2, train_records_r2, selection_records_r2, device_r2
)

confirmation_full_r2 = _offset_suite_r2(
    model_r2, confirmation_records_r2, device_r2, control="full"
)
confirmation_no_gr_r2 = _offset_suite_r2(
    model_r2, confirmation_records_r2, device_r2, control="no_gr"
)
confirmation_shuffled_r2 = _offset_suite_r2(
    model_r2, confirmation_records_r2, device_r2, control="shuffled_typewell"
)

pd.DataFrame(confirmation_full_r2["offsets"]).to_csv(
    OUT_DIR_R2 / "confirmation_offsets_full.csv", index=False
)
pd.DataFrame(confirmation_no_gr_r2["offsets"]).to_csv(
    OUT_DIR_R2 / "confirmation_offsets_no_gr.csv", index=False
)
pd.DataFrame(confirmation_shuffled_r2["offsets"]).to_csv(
    OUT_DIR_R2 / "confirmation_offsets_shuffled_typewell.csv", index=False
)

checks_r2 = {
    "zero_offset_rmse_le_8": confirmation_full_r2["zero"]["rmse"] <= 8.0,
    "jittered_rmse_le_10": confirmation_full_r2["jittered_rmse"] <= 10.0,
    "recovery_slope_ge_0p5": confirmation_full_r2["recovery_slope"] >= 0.50,
    "no_gr_jittered_degradation_ge_1": (
        confirmation_no_gr_r2["jittered_rmse"]
        - confirmation_full_r2["jittered_rmse"]
        >= 1.0
    ),
    "shuffle_jittered_degradation_ge_1": (
        confirmation_shuffled_r2["jittered_rmse"]
        - confirmation_full_r2["jittered_rmse"]
        >= 1.0
    ),
    "no_gr_recovery_drop_ge_0p2": (
        confirmation_full_r2["recovery_slope"]
        - confirmation_no_gr_r2["recovery_slope"]
        >= 0.20
    ),
}
strict_pass_r2 = bool(all(checks_r2.values()))

verdict_r2 = {
    "experiment": EXPERIMENT_R2,
    "stage": "fold3_absolute_gr_matching_anchor_jitter_gateway",
    "sample_unit": "anchor-jitter chunks nested within complete physical wells",
    "deployable": False,
    "truth_usage": "training labels and controlled diagnostic anchor jitter",
    "model_inputs": "horizontal GR and assigned calibrated typewell only",
    "anchor_jitter_is_model_input": False,
    "geometry_prior_gate_used_in_output": False,
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R2),
        "selection_fold": VALID_FOLD_R2,
        "selection_wells": len(selection_records_r2),
        "confirmation_wells": len(confirmation_records_r2),
        "fold2_touched": False,
        "fold1_touched": False,
        "selected_epoch": best_epoch_r2,
        "selection_objective": best_objective_r2,
    },
    "trainable_parameters": trainable_parameters_r2,
    "confirmation": {
        "full": confirmation_full_r2,
        "no_gr": confirmation_no_gr_r2,
        "shuffled_typewell": confirmation_shuffled_r2,
    },
    "checks": checks_r2,
    "strict_pass": strict_pass_r2,
    "checkpoint": str(MODEL_DIR_R2 / "best_absolute_matcher.pt"),
    "elapsed_minutes": (time.time() - started_r2) / 60.0,
    "next": (
        "PASS -> fuse this absolute emission with the R1 prior, then retest recursive rollout"
        if strict_pass_r2
        else "FAIL -> bilinear/cosine typewell matching cannot recover anchor aliases"
    ),
}
_write_json(OUT_DIR_R2 / "verdict.json", verdict_r2)
print("SEGN-R2 FINAL VERDICT", json.dumps(verdict_r2, indent=2), flush=True)
print("saved:", OUT_DIR_R2, flush=True)

if not strict_pass_r2:
    raise RuntimeError(json.dumps(verdict_r2, indent=2))

