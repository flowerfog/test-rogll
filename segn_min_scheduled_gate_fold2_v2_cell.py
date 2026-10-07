"""SEGN-min v2: explicit two-channel cost maps with online scheduled sampling.

This cell implements the highest-value deltas in SEGN_minimal_spec.md while
reusing the validated native-record and official-mask plumbing from
``segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py``.

Protocol
--------
* Folds 4/5 train two matched arms: pure teacher forcing and online scheduled
  sampling. Fold 3 selects the arm and epoch using deployed autoregressive
  posterior-mean rollout.
* Fold 2 remains untouched unless scheduled sampling beats teacher forcing.
* The selected recipe is refit from scratch on Folds 3/4/5, then applied once
  to Fold 2. Fold 1 is never loaded.
* Inputs are same-well horizontal GR/geometry, visible-prefix TVT_input and the
  assigned prefix-calibrated typewell. No PF, spatial, neighbour, strong,
  CP12, Gold or validation-truth feature is used at inference.

Prerequisites
-------------
1. Run ``build_native_dualref_all_records_v0_cell.py`` once.
2. Keep or upload ``segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py`` to
   /root/autodl-tmp (or the current working directory). Only its definitions
   are executed; its training tail is not run.
"""

from __future__ import annotations

from pathlib import Path
import copy
import gc
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bootstrap validated records, masks, chunk geometry, and metric helpers.
# ---------------------------------------------------------------------------

SOURCE_CANDIDATES = (
    Path("/root/autodl-tmp/segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py"),
    Path.cwd() / "segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(
        "Upload segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py to "
        "/root/autodl-tmp before running SEGN-min v2."
    )

source = SOURCE_PATH.read_text(encoding="utf-8")
marker = "\nstarted = time.time()\n_seed_everything(SEED)\nif REQUIRE_CUDA"
if marker not in source:
    raise RuntimeError(f"Cannot find bootstrap marker in {SOURCE_PATH}")
bootstrap = source.split(marker, 1)[0]
exec(compile(bootstrap, str(SOURCE_PATH), "exec"), globals())


EXPERIMENT = "segn_min_scheduled_gate_fold2_v2"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Matched A/B on Folds 4/5 -> Fold 3. Fold 2 is application-only.
AB_TRAIN_FOLDS = (4, 5)
AB_VALID_FOLD = 3
FINAL_TRAIN_FOLDS = (3, 4, 5)
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1

AB_EPOCHS = 12
FINAL_MAX_EPOCHS = 12
SEQUENCE_CHUNKS = 6
SEQUENCES_PER_WELL = 1
SCHEDULE_MAX_EPSILON = 0.50
SCHEDULE_WARMUP_EPOCHS = 8
REPORT_EVERY_MIN = 2
PATIENCE_REPORTS_MIN = 3
LOSS_MEAN = 0.10
TARGET_SIGMA_FT_MIN = 2.0
SCHEDULE_GAIN_GATE_FT = 0.10
FOLD3_TEACHER_ANCHOR_GATE_RMSE = 7.0
STANDALONE_GATE_RMSE = 8.5
CONTROL_GATE_FT = 0.30


# ---------------------------------------------------------------------------
# Explicit cost-map channels and learned soft fusion gate.
# ---------------------------------------------------------------------------


class MultiScaleGREncoder(nn.Module):
    """About 400 ft receptive field at native one-foot sampling."""

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(4, EMBEDDING, 9, padding=4),
            nn.GroupNorm(4, EMBEDDING),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            Residual1D(EMBEDDING, 1),
            Residual1D(EMBEDDING, 4),
            Residual1D(EMBEDDING, 16),
            Residual1D(EMBEDDING, 32),
        )

    def forward(self, values):
        return self.blocks(self.input(values))


class SEGNMinCostMapNet(nn.Module):
    """Bilinear and multi-head-cosine maps joined by a per-step soft gate."""

    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleGREncoder()

        # Channel A: a true learned bilinear interaction (no cosine norm).
        self.query_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)
        self.reference_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)
        self.bilinear_matrix = nn.Parameter(torch.eye(HEAD_DIM))

        # Channel B: independent multi-head cosine projections.
        self.query_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)
        self.reference_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)

        # Gate features: sharpness A/B, confidence A/B, provisional confidence,
        # channel disagreement, distance, query/reference coverage, Z slope,
        # previous delta and previous rate.
        self.gate = nn.Sequential(
            nn.Linear(12, 24),
            nn.GELU(),
            nn.Linear(24, 12),
            nn.GELU(),
            nn.Linear(12, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.channel_log_scale = nn.Parameter(torch.tensor([2.0, 2.0]).log())

    @staticmethod
    def _distribution_features(logits, states):
        probability = F.softmax(logits, dim=0)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=0)
        entropy = entropy / math.log(max(logits.shape[0], 2))
        sharpness = 1.0 - entropy
        confidence = probability.max(dim=0).values
        expected = (probability * states[:, None]).sum(dim=0)
        return probability, sharpness, confidence, expected

    def forward(
        self,
        horizontal,
        typewell,
        node_positions,
        candidate_positions,
        geometry,
        channel_control="full",
        return_aux=False,
    ):
        horizontal_embedding = self.encoder(horizontal)[0]
        typewell_embedding = self.encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(typewell_embedding, candidate_positions)

        query_a = self.query_bilinear(query[None])[0]
        reference_a = self.reference_bilinear(reference[None])[0]
        transformed_query = self.bilinear_matrix @ query_a
        logits_a = torch.einsum("dt,ds->st", transformed_query, reference_a)
        logits_a = logits_a / math.sqrt(HEAD_DIM)

        query_b = self.query_heads(query[None])[0].reshape(HEADS, HEAD_DIM, -1)
        reference_b = self.reference_heads(reference[None])[0].reshape(HEADS, HEAD_DIM, -1)
        query_b = F.normalize(query_b, dim=1, eps=1e-5)
        reference_b = F.normalize(reference_b, dim=1, eps=1e-5)
        logits_b = torch.einsum("hdt,hds->hst", query_b, reference_b).mean(dim=0)

        query_raw = horizontal[0, :, node_positions]
        reference_raw, raw_axis_valid = _sample_axis(typewell[0], candidate_positions)
        candidate_valid = axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)
        valid_volume = candidate_valid[:, None]
        logits_a = logits_a.masked_fill(~valid_volume, -1.0e4)
        logits_b = logits_b.masked_fill(~valid_volume, -1.0e4)

        states = torch.from_numpy(STATE_OFFSETS).to(query.device, query.dtype)
        _, sharp_a, conf_a, expected_a = self._distribution_features(logits_a, states)
        _, sharp_b, conf_b, expected_b = self._distribution_features(logits_b, states)
        provisional = 0.5 * (logits_a + logits_b)
        provisional_conf = F.softmax(provisional, dim=0).max(dim=0).values
        disagreement = torch.abs(expected_a - expected_b) / STATE_HALF_WIDTH_FT

        query_coverage = query_raw[1].clamp(0.0, 1.0)
        reference_coverage = candidate_valid.float().mean().expand_as(query_coverage)
        gate_features = torch.stack(
            [
                sharp_a,
                sharp_b,
                conf_a,
                conf_b,
                provisional_conf,
                disagreement,
                geometry[1].clamp(-1.0, 1.0),
                query_coverage,
                reference_coverage,
                geometry[3].clamp(-2.0, 2.0),
                geometry[6].clamp(-4.0, 4.0),
                geometry[7].clamp(-4.0, 4.0),
            ],
            dim=1,
        )
        gate_weights = F.softmax(self.gate(gate_features), dim=1).transpose(0, 1)
        scales = torch.exp(self.channel_log_scale).clamp(0.5, 20.0)[:, None, None]
        channel_logits = torch.stack([logits_a, logits_b], dim=0) * scales

        if channel_control == "no_bilinear":
            gate_weights = torch.stack(
                [torch.zeros_like(gate_weights[0]), torch.ones_like(gate_weights[1])]
            )
        elif channel_control == "no_cosine":
            gate_weights = torch.stack(
                [torch.ones_like(gate_weights[0]), torch.zeros_like(gate_weights[1])]
            )

        logits = (gate_weights[:, None, :] * channel_logits).sum(dim=0)
        logits = logits.masked_fill(~valid_volume, -1.0e4)
        if not return_aux:
            return logits
        return logits, {
            "gate_weights": gate_weights,
            "channel_logits": channel_logits,
            "channel_expected": torch.stack([expected_a, expected_b]),
        }


def _loss_min(model, payload, device):
    tensors = _tensor_payload(payload, device)
    logits, model_aux = model(*tensors, return_aux=True)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    target_distribution = torch.exp(
        -0.5 * ((states - target[None]) / TARGET_SIGMA_FT_MIN) ** 2
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=0, keepdim=True
    ).clamp_min(1e-8)
    log_probability = F.log_softmax(logits, dim=0)
    ce = -(target_distribution * log_probability).sum(dim=0).mean()
    probability = F.softmax(logits, dim=0)
    expected = (probability * states).sum(dim=0)
    mean_loss = F.smooth_l1_loss(expected / 8.0, target / 8.0)
    total = ce + LOSS_MEAN * mean_loss
    return total, {
        "ce": float(ce.detach()),
        "mean": float(mean_loss.detach()),
        "expected": expected.detach(),
        "gate_weights": model_aux["gate_weights"].detach(),
    }


@torch.no_grad()
def _predict_chunk_min(model, payload, device):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, auxiliary = model(
            *tensors,
            channel_control=payload["channel_control"],
            return_aux=True,
        )
    probability = F.softmax(logits.float(), dim=0)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    expected = (probability * states).sum(dim=0)
    entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=0)
    mode = STATE_OFFSETS[torch.argmax(probability, dim=0).cpu().numpy()]
    return (
        expected.cpu().numpy().astype(np.float64),
        mode.astype(np.float64),
        entropy.cpu().numpy().astype(np.float64),
        auxiliary["gate_weights"].float().cpu().numpy().astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Online scheduled sampling: recursive anchors come from this model's current
# predictions, not from an offline error/noise library.
# ---------------------------------------------------------------------------


def _sequence_starts(record, rng):
    starts = record.chunk_starts
    if len(starts) <= SEQUENCE_CHUNKS:
        return starts
    first = int(rng.integers(0, len(starts) - SEQUENCE_CHUNKS + 1))
    return starts[first : first + SEQUENCE_CHUNKS]


def _train_epoch_min(model, records, optimizer, scaler, device, epoch, scheduled):
    model.train()
    # The A/B arms see the same well order and sequence starts. A separate RNG
    # controls only the scheduled-sampling coin flips.
    rng = np.random.default_rng(SEED + 1009 * epoch)
    schedule_rng = np.random.default_rng(SEED + 1009 * epoch + 7919)
    order = rng.permutation(len(records))
    epsilon = (
        min(SCHEDULE_MAX_EPSILON, SCHEDULE_MAX_EPSILON * epoch / SCHEDULE_WARMUP_EPOCHS)
        if scheduled
        else 0.0
    )
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "mean": 0.0,
        "samples": 0,
        "model_anchor_samples": 0,
        "skipped": 0,
        "gate_bilinear": 0.0,
        "gate_cosine": 0.0,
    }
    optimizer_steps = 0
    amp_enabled = device.type == "cuda"

    for sequence_number, record_index in enumerate(order, 1):
        record = records[int(record_index)]
        for _ in range(SEQUENCES_PER_WELL):
            starts = _sequence_starts(record, rng)
            predicted_nodes = np.full(len(record.node_rows), np.nan, np.float64)
            first_start = int(starts[0])
            predicted_nodes[first_start] = float(
                record.truth[int(record.node_rows[first_start])]
            )
            previous_start = first_start
            previous_anchor = float(predicted_nodes[first_start])

            for sequence_step, raw_start in enumerate(starts):
                start = int(raw_start)
                start_row = int(record.node_rows[start])
                true_anchor = float(record.truth[start_row])
                use_model_anchor = (
                    sequence_step > 0
                    and np.isfinite(predicted_nodes[start])
                    and schedule_rng.random() < epsilon
                )
                deployed_anchor = (
                    float(predicted_nodes[start]) if use_model_anchor else true_anchor
                )
                if use_model_anchor:
                    totals["model_anchor_samples"] += 1

                previous_row = int(record.node_rows[previous_start])
                previous_md_delta = max(
                    float(record.md[start_row] - record.md[previous_row]), 1.0
                )
                previous_delta = (
                    deployed_anchor - previous_anchor if sequence_step > 0 else 0.0
                )
                previous_rate = (
                    previous_delta / previous_md_delta if sequence_step > 0 else 0.0
                )
                payload = _chunk_payload(
                    record,
                    start,
                    deployed_anchor=deployed_anchor,
                    previous_delta=previous_delta,
                    previous_rate=previous_rate,
                )
                if np.max(np.abs(payload["target_state"])) > STATE_HALF_WIDTH_FT - 0.5:
                    totals["skipped"] += 1
                    previous_start = start
                    previous_anchor = deployed_anchor
                    continue

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss, parts = _loss_min(model, payload, device)
                    scaled_loss = loss / GRADIENT_ACCUMULATION
                if not torch.isfinite(loss):
                    totals["skipped"] += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(scaled_loss).backward()

                expected = parts["expected"].float().cpu().numpy()
                end = start + len(expected)
                predicted_nodes[start + 1 : end + 1] = deployed_anchor + expected
                gate = parts["gate_weights"].float().cpu().numpy()
                totals["loss"] += float(loss.detach())
                totals["ce"] += parts["ce"]
                totals["mean"] += parts["mean"]
                totals["gate_bilinear"] += float(gate[0].mean())
                totals["gate_cosine"] += float(gate[1].mean())
                totals["samples"] += 1

                if totals["samples"] % GRADIENT_ACCUMULATION == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                previous_start = start
                previous_anchor = deployed_anchor

        if sequence_number % 100 == 0 or sequence_number == len(order):
            print(
                f"SEGN-MIN train arm={'scheduled' if scheduled else 'teacher'} "
                f"epoch={epoch} epsilon={epsilon:.3f} wells={sequence_number}/{len(order)} "
                f"loss={totals['loss'] / max(totals['samples'], 1):.4f}",
                flush=True,
            )

    if totals["samples"] % GRADIENT_ACCUMULATION:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    denominator = max(totals["samples"], 1)
    return {
        "loss": totals["loss"] / denominator,
        "ce": totals["ce"] / denominator,
        "mean": totals["mean"] / denominator,
        "samples": totals["samples"],
        "model_anchor_samples": totals["model_anchor_samples"],
        "model_anchor_fraction": totals["model_anchor_samples"] / denominator,
        "skipped": totals["skipped"],
        "gate_bilinear": totals["gate_bilinear"] / denominator,
        "gate_cosine": totals["gate_cosine"] / denominator,
        "optimizer_steps": optimizer_steps,
        "epsilon": epsilon,
    }


@torch.no_grad()
def _predict_record_min(
    model,
    record,
    device,
    anchor_mode="rollout",
    readout="mean",
    control="full",
    shuffled_typewell=None,
):
    model.eval()
    node_prediction = np.full(len(record.node_rows), np.nan, np.float64)
    node_prediction[0] = float(record.anchor_tvt)
    entropy_values = []
    disagreement_values = []
    gate_values = []
    prior_anchor = float(record.anchor_tvt)
    prior_start = 0

    for raw_start in record.chunk_starts:
        start = int(raw_start)
        start_row = int(record.node_rows[start])
        if anchor_mode == "teacher":
            deployed_anchor = float(record.truth[start_row])
        elif np.isfinite(node_prediction[start]):
            deployed_anchor = float(node_prediction[start])
        else:
            prior = np.flatnonzero(np.isfinite(node_prediction[: start + 1]))
            if not len(prior):
                raise RuntimeError(f"{record.well}: rollout lost its anchor")
            deployed_anchor = float(node_prediction[prior[-1]])

        prior_row = int(record.node_rows[prior_start])
        md_delta = max(float(record.md[start_row] - record.md[prior_row]), 1.0)
        previous_delta = deployed_anchor - prior_anchor if start > 0 else 0.0
        previous_rate = previous_delta / md_delta if start > 0 else 0.0
        payload = _chunk_payload(
            record,
            start,
            deployed_anchor=deployed_anchor,
            previous_delta=previous_delta,
            previous_rate=previous_rate,
            control=control,
            shuffled_typewell=shuffled_typewell,
        )
        expected, mode_state, entropy, gate = _predict_chunk_min(
            model, payload, device
        )
        selected = expected if readout == "mean" else mode_state
        end = start + len(selected)
        node_prediction[start + 1 : end + 1] = deployed_anchor + selected
        entropy_values.extend(entropy.tolist())
        disagreement_values.extend(np.abs(expected - mode_state).tolist())
        gate_values.append(gate.mean(axis=1))
        prior_anchor = deployed_anchor
        prior_start = start

    valid_nodes = np.flatnonzero(np.isfinite(node_prediction))
    support_md = record.md[record.node_rows[valid_nodes]]
    support_tvt = node_prediction[valid_nodes]
    prediction = np.interp(record.md[record.eval_index], support_md, support_tvt)
    truth = record.truth[record.eval_index].astype(np.float64)
    gate_mean = np.mean(np.stack(gate_values), axis=0) if gate_values else np.full(2, np.nan)
    diagnostics = {
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else math.nan,
        "mean_decode_disagreement": (
            float(np.mean(disagreement_values)) if disagreement_values else math.nan
        ),
        "gate_bilinear": float(gate_mean[0]),
        "gate_cosine": float(gate_mean[1]),
    }
    return truth, prediction, diagnostics


@torch.no_grad()
def _evaluate_min(
    model,
    records,
    device,
    anchor_mode="rollout",
    readout="mean",
    control="full",
    save_prefix=None,
):
    rows = []
    truths = []
    predictions = []
    row_wells = []
    row_indices = []
    row_distances = []
    permutation = np.roll(np.arange(len(records)), 1)
    for number, record in enumerate(records, 1):
        shuffled = (
            records[int(permutation[number - 1])]
            if control == "shuffled_typewell"
            else None
        )
        truth, prediction, diagnostics = _predict_record_min(
            model,
            record,
            device,
            anchor_mode=anchor_mode,
            readout=readout,
            control=control,
            shuffled_typewell=shuffled,
        )
        distance = record.md[record.eval_index] - float(record.md[record.anchor_index])
        rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "well_rmse": _rmse(truth, prediction),
                "well_mae": _mae(truth, prediction),
                "exact_sse": float(np.sum((prediction - truth) ** 2)),
                **diagnostics,
                "_distance": distance.astype(np.float64),
            }
        )
        truths.append(truth)
        predictions.append(prediction)
        row_wells.append(np.repeat(record.well, len(record.eval_index)))
        row_indices.append(record.eval_index.astype(np.int32))
        row_distances.append(distance.astype(np.float32))
        if number % 50 == 0 or number == len(records):
            print(
                f"SEGN-MIN evaluate {anchor_mode}/{readout}/{control} "
                f"{number}/{len(records)}",
                flush=True,
            )

    metrics, bywell = _metrics_from_rows(rows, truths, predictions)
    metrics["exact_sse"] = float(
        np.sum((np.concatenate(predictions) - np.concatenate(truths)) ** 2)
    )
    metrics["gate_bilinear"] = float(bywell["gate_bilinear"].mean())
    metrics["gate_cosine"] = float(bywell["gate_cosine"].mean())
    if save_prefix:
        bywell.drop(columns=["_distance"], errors="ignore").to_csv(
            OUT_DIR / f"{save_prefix}_bywell.csv", index=False
        )
        np.savez_compressed(
            OUT_DIR / f"{save_prefix}_rows.npz",
            truth=np.concatenate(truths).astype(np.float32),
            prediction=np.concatenate(predictions).astype(np.float32),
            well=np.concatenate(row_wells),
            row_index=np.concatenate(row_indices),
            distance_past_anchor=np.concatenate(row_distances),
        )
    return metrics, bywell


def _fit_arm(records, tune_records, device, scheduled, epochs, phase):
    # Teacher and scheduled A/B arms share an exact initial state. The final
    # refit uses a separate, fixed seed because it has a different train set.
    _seed_everything(SEED + (9001 if phase.startswith("final") else 4242))
    model = SEGNMinCostMapNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_rmse = math.inf
    bad_reports = 0
    started_arm = time.time()

    for epoch in range(1, epochs + 1):
        train = _train_epoch_min(
            model, records, optimizer, scaler, device, epoch, scheduled
        )
        scheduler.step()
        report = {
            "phase": phase,
            "arm": "scheduled" if scheduled else "teacher",
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started_arm) / 60.0,
        }
        should_report = epoch == 1 or epoch % REPORT_EVERY_MIN == 0 or epoch == epochs
        if tune_records is not None and should_report:
            rollout, _ = _evaluate_min(
                model, tune_records, device, anchor_mode="rollout", readout="mean"
            )
            report.update({f"tune_{key}": value for key, value in rollout.items()})
            if rollout["rmse"] < best_rmse - 1e-4:
                best_rmse = rollout["rmse"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_reports = 0
            else:
                bad_reports += 1
        elif tune_records is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        history.append(report)
        pd.DataFrame(history).to_csv(OUT_DIR / f"{phase}_history.csv", index=False)
        print("SEGN-MIN EPOCH", report, flush=True)
        if tune_records is not None and bad_reports >= PATIENCE_REPORTS_MIN:
            print(f"SEGN-MIN {phase} early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    checkpoint = MODEL_DIR / f"{phase}_best.pt"
    torch.save(
        {
            "model": best_state,
            "epoch": best_epoch,
            "scheduled": scheduled,
            "state_offsets": STATE_OFFSETS,
            "source": str(SOURCE_PATH),
        },
        checkpoint,
    )
    return model, best_epoch, best_rmse, history, checkpoint


# ---------------------------------------------------------------------------
# Nested gateway.
# ---------------------------------------------------------------------------


started = time.time()
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-min v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds, fold_path = _load_folds()
fold_map = dict(zip(folds["well"], folds["fold"]))
ab_train_wells = sorted(folds.loc[folds["fold"].isin(AB_TRAIN_FOLDS), "well"])
ab_valid_wells = sorted(folds.loc[folds["fold"].eq(AB_VALID_FOLD), "well"])
final_train_wells = sorted(
    folds.loc[folds["fold"].isin(FINAL_TRAIN_FOLDS), "well"]
)
application_wells = sorted(folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"])
if min(
    len(ab_train_wells),
    len(ab_valid_wells),
    len(final_train_wells),
    len(application_wells),
) < 150:
    raise RuntimeError(
        "Unexpected fold sizes: "
        + repr(
            (
                len(ab_train_wells),
                len(ab_valid_wells),
                len(final_train_wells),
                len(application_wells),
            )
        )
    )

all_needed = sorted(
    set(ab_train_wells + ab_valid_wells + final_train_wells + application_wells)
)
records_by_well = {
    record.well: record for record in _load_records(all_needed, "fold2345")
}
ab_train_records = [records_by_well[well] for well in ab_train_wells]
ab_valid_records = [records_by_well[well] for well in ab_valid_wells]
final_train_records = [records_by_well[well] for well in final_train_wells]
application_records = [records_by_well[well] for well in application_wells]

guards = {
    "source": str(SOURCE_PATH),
    "fold_source": str(fold_path),
    "ab_train_wells": len(ab_train_records),
    "ab_valid_wells": len(ab_valid_records),
    "final_train_wells": len(final_train_records),
    "application_wells": len(application_records),
    "state_candidates": len(STATE_OFFSETS),
    "state_half_width_ft": STATE_HALF_WIDTH_FT,
    "node_spacing_md_ft": NODE_SPACING_MD_FT,
    "chunk_target_ft": CHUNK_TARGET_NODES * NODE_SPACING_MD_FT,
    "chunk_advance_ft": CHUNK_STRIDE_NODES * NODE_SPACING_MD_FT,
    "ab_target_axis_coverage": _target_axis_coverage(ab_valid_records),
    "fold1_touched": False,
}
print("SEGN-MIN GUARDS", json.dumps(guards, indent=2), flush=True)

teacher_model, teacher_epoch, teacher_best, teacher_history, teacher_checkpoint = _fit_arm(
    ab_train_records,
    ab_valid_records,
    device,
    scheduled=False,
    epochs=AB_EPOCHS,
    phase="ab_teacher45_to_3",
)
teacher_rollout, _ = _evaluate_min(
    teacher_model,
    ab_valid_records,
    device,
    anchor_mode="rollout",
    readout="mean",
    save_prefix="fold3_teacher_arm_rollout_mean",
)
del teacher_model
gc.collect()
torch.cuda.empty_cache()

scheduled_model, scheduled_epoch, scheduled_best, scheduled_history, scheduled_checkpoint = _fit_arm(
    ab_train_records,
    ab_valid_records,
    device,
    scheduled=True,
    epochs=AB_EPOCHS,
    phase="ab_scheduled45_to_3",
)
scheduled_rollout, _ = _evaluate_min(
    scheduled_model,
    ab_valid_records,
    device,
    anchor_mode="rollout",
    readout="mean",
    save_prefix="fold3_scheduled_arm_rollout_mean",
)
scheduled_mode, _ = _evaluate_min(
    scheduled_model,
    ab_valid_records,
    device,
    anchor_mode="rollout",
    readout="mode",
    save_prefix="fold3_scheduled_arm_rollout_mode",
)
scheduled_teacher_anchor, _ = _evaluate_min(
    scheduled_model,
    ab_valid_records,
    device,
    anchor_mode="teacher",
    readout="mean",
    save_prefix="fold3_scheduled_arm_teacher_mean",
)
fold3_no_gr, _ = _evaluate_min(
    scheduled_model,
    ab_valid_records,
    device,
    anchor_mode="teacher",
    readout="mean",
    control="no_gr",
    save_prefix="fold3_scheduled_arm_teacher_no_gr",
)
fold3_shuffled, _ = _evaluate_min(
    scheduled_model,
    ab_valid_records,
    device,
    anchor_mode="teacher",
    readout="mean",
    control="shuffled_typewell",
    save_prefix="fold3_scheduled_arm_teacher_shuffled_typewell",
)

ab_gain = teacher_rollout["rmse"] - scheduled_rollout["rmse"]
readout_gain = scheduled_mode["rmse"] - scheduled_rollout["rmse"]
anchor_gap = scheduled_rollout["rmse"] - scheduled_teacher_anchor["rmse"]
ab_summary = {
    "teacher_epoch": teacher_epoch,
    "scheduled_epoch": scheduled_epoch,
    "teacher_rollout": teacher_rollout,
    "scheduled_rollout": scheduled_rollout,
    "scheduled_mode": scheduled_mode,
    "scheduled_teacher_anchor": scheduled_teacher_anchor,
    "scheduled_teacher_no_gr": fold3_no_gr,
    "scheduled_teacher_shuffled_typewell": fold3_shuffled,
    "scheduled_gain_vs_teacher": ab_gain,
    "posterior_mean_gain_vs_mode": readout_gain,
    "rollout_gap_vs_teacher_anchor": anchor_gap,
}
_write_json(OUT_DIR / "fold3_ab_verdict.json", ab_summary)
print("SEGN-MIN FOLD3 A/B", json.dumps(ab_summary, indent=2), flush=True)

if (
    ab_gain < SCHEDULE_GAIN_GATE_FT
    or scheduled_teacher_anchor["rmse"] > FOLD3_TEACHER_ANCHOR_GATE_RMSE
    or fold3_no_gr["rmse"] - scheduled_teacher_anchor["rmse"] < CONTROL_GATE_FT
    or fold3_shuffled["rmse"] - scheduled_teacher_anchor["rmse"] < CONTROL_GATE_FT
):
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "fold3_scheduled_sampling_gate",
        "guards": guards,
        "ab": ab_summary,
        "gates": {
            "scheduled_gain_ge": SCHEDULE_GAIN_GATE_FT,
            "teacher_anchor_rmse_le": FOLD3_TEACHER_ANCHOR_GATE_RMSE,
            "teacher_no_gr_degradation_ge": CONTROL_GATE_FT,
            "teacher_shuffled_degradation_ge": CONTROL_GATE_FT,
            "posterior_mean_better_than_mode": readout_gain > 0.0,
        },
        "strict_pass": False,
        "fold2_touched": False,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "next": "FAIL -> scheduled sampling or teacher-anchor evidence gate failed; inspect channel maps before Fold2",
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    print("SEGN-MIN FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
    raise RuntimeError(json.dumps(verdict, indent=2))

# Refit the winning scheduled recipe from scratch on Folds 3/4/5. The epoch
# count is selected exclusively on Fold 3 above; Fold 2 remains untouched.
del scheduled_model
gc.collect()
torch.cuda.empty_cache()
selected_epochs = int(min(max(scheduled_epoch, 1), FINAL_MAX_EPOCHS))
final_model, final_epoch, _, final_history, final_checkpoint = _fit_arm(
    final_train_records,
    None,
    device,
    scheduled=True,
    epochs=selected_epochs,
    phase="final345_scheduled",
)

fold2_mean, _ = _evaluate_min(
    final_model,
    application_records,
    device,
    anchor_mode="rollout",
    readout="mean",
    save_prefix="fold2_rollout_mean",
)
fold2_mode, _ = _evaluate_min(
    final_model,
    application_records,
    device,
    anchor_mode="rollout",
    readout="mode",
    save_prefix="fold2_rollout_mode",
)
fold2_teacher_anchor, _ = _evaluate_min(
    final_model,
    application_records,
    device,
    anchor_mode="teacher",
    readout="mean",
    save_prefix="fold2_teacher_anchor_mean",
)
fold2_no_gr, _ = _evaluate_min(
    final_model,
    application_records,
    device,
    anchor_mode="rollout",
    readout="mean",
    control="no_gr",
    save_prefix="fold2_no_gr",
)
fold2_shuffled, _ = _evaluate_min(
    final_model,
    application_records,
    device,
    anchor_mode="rollout",
    readout="mean",
    control="shuffled_typewell",
    save_prefix="fold2_shuffled_typewell",
)
fold2_no_bilinear, _ = _evaluate_min(
    final_model,
    application_records,
    device,
    anchor_mode="rollout",
    readout="mean",
    control="no_bilinear",
    save_prefix="fold2_no_bilinear",
)
fold2_no_cosine, _ = _evaluate_min(
    final_model,
    application_records,
    device,
    anchor_mode="rollout",
    readout="mean",
    control="no_cosine",
    save_prefix="fold2_no_cosine",
)

comparison = pd.DataFrame(
    [
        {"variant": "rollout_mean", **fold2_mean},
        {"variant": "rollout_mode", **fold2_mode},
        {"variant": "teacher_anchor_mean", **fold2_teacher_anchor},
        {"variant": "no_gr", **fold2_no_gr},
        {"variant": "shuffled_typewell", **fold2_shuffled},
        {"variant": "no_bilinear", **fold2_no_bilinear},
        {"variant": "no_cosine", **fold2_no_cosine},
    ]
)
comparison.to_csv(OUT_DIR / "fold2_comparison.csv", index=False)
print("SEGN-MIN FOLD2 COMPARISON")
print(comparison.to_string(index=False), flush=True)

gates = {
    "fold3_scheduled_gain_ge_0p10": ab_gain >= SCHEDULE_GAIN_GATE_FT,
    "fold2_standalone_rmse_le_8p5": fold2_mean["rmse"] <= STANDALONE_GATE_RMSE,
    "fold2_posterior_mean_better_than_mode": fold2_mean["rmse"] < fold2_mode["rmse"],
    "fold2_no_gr_degradation_ge_0p30": (
        fold2_no_gr["rmse"] - fold2_mean["rmse"] >= CONTROL_GATE_FT
    ),
    "fold2_shuffled_typewell_degradation_ge_0p30": (
        fold2_shuffled["rmse"] - fold2_mean["rmse"] >= CONTROL_GATE_FT
    ),
}
strict_pass = all(gates.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_online_scheduled_sampling_costmap_gateway",
    "sample_unit": "complete physical well with truncated sequential training rollouts",
    "legal_inputs_only": True,
    "uses_pf_spatial_neighbor_strong_cp12_gold": False,
    "protocol": {
        "ab_train_folds": list(AB_TRAIN_FOLDS),
        "ab_validation_fold": AB_VALID_FOLD,
        "final_train_folds": list(FINAL_TRAIN_FOLDS),
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "selected_epoch": selected_epochs,
    },
    "model": {
        "parameters": int(sum(parameter.numel() for parameter in final_model.parameters())),
        "channels": ["bilinear", "multihead_cosine"],
        "fusion": "learned per-step soft gate",
        "readout": "posterior mean",
        "scheduled_sampling_max_epsilon": SCHEDULE_MAX_EPSILON,
        "sequence_chunks_per_training_sample": SEQUENCE_CHUNKS,
    },
    "guards": guards,
    "fold3_ab": ab_summary,
    "fold2": {
        "rollout_mean": fold2_mean,
        "rollout_mode": fold2_mode,
        "teacher_anchor_mean": fold2_teacher_anchor,
        "no_gr": fold2_no_gr,
        "shuffled_typewell": fold2_shuffled,
        "no_bilinear": fold2_no_bilinear,
        "no_cosine": fold2_no_cosine,
        "anchor_gap": fold2_mean["rmse"] - fold2_teacher_anchor["rmse"],
        "posterior_mean_gain": fold2_mode["rmse"] - fold2_mean["rmse"],
    },
    "gates": gates,
    "strict_pass": strict_pass,
    "checkpoint": str(final_checkpoint),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> run five-fold OOF and detrended distance-stratified correlation audit"
        if strict_pass
        else "FAIL -> return to channel-map observability; do not add K-anchor or fusion yet"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("SEGN-MIN FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
