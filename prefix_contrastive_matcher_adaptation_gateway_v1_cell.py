"""Prefix-supervised contrastive matcher plus legal per-well adaptation.

Run in the same live kernel after
``real_correspondence_multiscale_matcher_gateway_v0_cell.py``.

The v0 matcher already used explicit +/-24 and +/-48 ft alias margins, but its
global training rows came only from outer-train hidden suffixes.  A validation
well's visible prefix calibrated the sensor and then disappeared from the
matching objective.  This gateway adds the missing mechanisms:

* global training mixes real visible-prefix and hidden-suffix correspondences;
* a wrong-well typewell is used as an explicit cross-well negative;
* at inference, visible-prefix pairwise ranking deficits fit one 16-dimensional
  analytic Ridge residual over the scale-pair correlations;
* no network parameters are updated for a validation/test well.

The v0 Fold2-selected epoch is reused.  Fold2 selects only Ridge lambda and
dose; Folds 2-5 are then refit and Fold1 is evaluated once.
"""

from dataclasses import dataclass
from pathlib import Path
import gc
import json
import math
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Live-kernel contract and configuration
# ---------------------------------------------------------------------------

REQUIRED_V1 = {
    "OOF_ROOT",
    "MODEL_DIR",
    "samples",
    "folds",
    "fold_path",
    "core_wells",
    "tune_wells",
    "outer_train_wells",
    "test_wells",
    "device",
    "selected_epoch",
    "RealCorrespondenceMatcher",
    "QuerySet",
    "_load_raw_record",
    "_select_evenly",
    "_query_set",
    "_to_tensors",
    "_matcher_loss",
    "_new_training_objects",
    "_retrieval_arrays",
    "_summarise_payload",
    "_merge_payload",
    "_write_json",
    "_json_default",
    "STATE_OFFSETS",
    "N_STATES",
    "STATE_HALF_WIDTH_FT",
    "ALIAS_OFFSETS_FT",
    "GRADIENT_CLIP",
    "REPORT_EVERY_WELLS",
}
missing_v1 = sorted(REQUIRED_V1.difference(globals()))
if missing_v1:
    raise RuntimeError(
        "Run real_correspondence_multiscale_matcher_gateway_v0_cell.py in "
        "this live kernel first; missing: " + repr(missing_v1)
    )

EXPERIMENT_V1 = "prefix_contrastive_matcher_adaptation_gateway_v1"
OUT_DIR_V1 = OOF_ROOT / EXPERIMENT_V1
MODEL_DIR_V1 = OUT_DIR_V1 / "models"
for directory_v1 in (OUT_DIR_V1, MODEL_DIR_V1):
    directory_v1.mkdir(parents=True, exist_ok=True)

SEED_V1 = 20260805
PREFIX_POSITIONS_V1 = 160
GLOBAL_PREFIX_POSITIONS_V1 = 96
PREFIX_CONTEXT_FRACTION_V1 = 0.70
CROSS_WELL_MARGIN_V1 = 0.20
CROSS_WELL_WEIGHT_V1 = 0.10
PAIR_MARGIN_V1 = 0.50
TOP_GLOBAL_NEGATIVES_V1 = 4
ADAPTER_DIMENSION_V1 = 16  # the 4x4 scale-pair correlations only
ADAPTER_CORRECTION_CAP_V1 = 4.0
RIDGE_LAMBDAS_V1 = (0.1, 1.0, 10.0, 100.0)
ADAPTER_DOSES_V1 = (0.25, 0.50, 1.00)

TOP8_GATE_V1 = 0.70
MEDIAN_RANK_GATE_V1 = 8.0
ALIAS_GATE_V1 = 0.72
ADAPTER_GAIN_GATE_V1 = 0.03
LATE_PREFIX_GAIN_GATE_V1 = 0.03
NO_GR_DROP_GATE_V1 = 0.15
TYPEWELL_SHUFFLE_DROP_GATE_V1 = 0.20
WRONG_PREFIX_DROP_GATE_V1 = 0.03


def _seed_v1(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subset_query_v1(query, positions):
    positions = np.asarray(positions, np.int64)
    return QuerySet(
        patches=query.patches[positions],
        labels=query.labels[positions],
        label_index=query.label_index[positions],
        row_weight=query.row_weight[positions],
        deep_mask=query.deep_mask[positions],
        sample_rows=query.sample_rows[positions],
        requested_rows=int(len(positions)),
    )


def _concat_query_v1(first, second):
    return QuerySet(
        patches=np.concatenate([first.patches, second.patches], axis=0),
        labels=np.concatenate([first.labels, second.labels], axis=0),
        label_index=np.concatenate([first.label_index, second.label_index], axis=0),
        row_weight=np.concatenate([first.row_weight, second.row_weight], axis=0),
        deep_mask=np.concatenate([first.deep_mask, second.deep_mask], axis=0),
        sample_rows=np.concatenate([first.sample_rows, second.sample_rows], axis=0),
        requested_rows=int(first.requested_rows + second.requested_rows),
    )


def _legal_prefix_query_v1(well, state_valid):
    record = _load_raw_record(well)
    rows = np.arange(record.anchor_index + 1, dtype=np.int64)
    labels = record.truth[rows].astype(np.float64) - float(record.anchor_tvt)
    label_index = np.rint(labels + STATE_HALF_WIDTH_FT).astype(np.int64)
    legal = np.isfinite(labels) & (label_index >= 0) & (label_index < N_STATES)
    safe = np.clip(label_index, 0, N_STATES - 1)
    legal &= state_valid[safe]
    rows = rows[legal]
    if len(rows) < 64:
        raise RuntimeError(f"{well}: only {len(rows)} legal prefix correspondences")
    rows = _select_evenly(rows, PREFIX_POSITIONS_V1)
    query, _ = _query_set(record, rows, state_valid, include_known=True)
    if len(query.labels) < 48:
        raise RuntimeError(f"{well}: only {len(query.labels)} retained prefix queries")
    order = np.argsort(query.sample_rows)
    query = _subset_query_v1(query, order)
    query.row_weight[:] = 0.50
    return query


def _global_prefix_subset_v1(query):
    positions = _select_evenly(np.arange(len(query.labels)), GLOBAL_PREFIX_POSITIONS_V1)
    return _subset_query_v1(query, positions)


def _prefix_split_v1(query):
    split = int(round(len(query.labels) * PREFIX_CONTEXT_FRACTION_V1))
    split = int(np.clip(split, 32, len(query.labels) - 24))
    return (
        _subset_query_v1(query, np.arange(split)),
        _subset_query_v1(query, np.arange(split, len(query.labels))),
    )


# ---------------------------------------------------------------------------
# Mixed prefix/suffix global training with wrong-reference negatives
# ---------------------------------------------------------------------------

def _train_mixed_epoch_v1(model, wells, optimizer, scheduler, scaler, epoch, phase):
    model.train()
    rng = np.random.default_rng(SEED_V1 + epoch * 1009 + (0 if phase == "core" else 50000))
    order = rng.permutation(len(wells))
    donor_shift = max(1, len(wells) // 3)
    mean_eval_rows = float(np.mean([samples[well].eval_rows for well in wells]))
    totals = {"loss": 0.0, "ce": 0.0, "alias": 0.0, "cross": 0.0, "skipped": 0}

    for step, order_position in enumerate(order, 1):
        source_position = int(order_position)
        well = wells[source_position]
        donor = wells[(source_position + donor_shift) % len(wells)]
        sample = samples[well]
        query_set = sample.train
        if query_set is None:
            raise RuntimeError(f"{well}: missing mixed training query")
        query, reference, state_valid, labels, label_index, row_weight = _to_tensors(
            sample, query_set, device
        )
        donor_sample = samples[donor]
        donor_reference = torch.from_numpy(
            donor_sample.reference.astype(np.float32)
        ).to(device, non_blocking=True)
        donor_valid = torch.from_numpy(donor_sample.state_valid.astype(bool)).to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(query, reference, state_valid)
            primary, ce, alias = _matcher_loss(
                logits, labels, label_index, row_weight, state_valid
            )
            wrong_logits = model(query, donor_reference, donor_valid)
            positive = logits.gather(1, label_index[:, None])[:, 0]
            wrong_best = wrong_logits.amax(dim=1)
            cross = F.relu(CROSS_WELL_MARGIN_V1 - positive + wrong_best).mean()
            loss = primary + CROSS_WELL_WEIGHT_V1 * cross
            loss = loss * sample.eval_rows / max(mean_eval_rows, 1.0)

        before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        totals["skipped"] += int(float(scaler.get_scale()) < before)
        totals["loss"] += float(loss.detach().cpu())
        totals["ce"] += float(ce.cpu())
        totals["alias"] += float(alias.cpu())
        totals["cross"] += float(cross.detach().cpu())

        if step % REPORT_EVERY_WELLS == 0 or step == len(wells):
            print(
                f"PREFIX MATCHER {phase} epoch={epoch} wells={step}/{len(wells)} "
                f"loss={totals['loss']/step:.4f} cross={totals['cross']/step:.4f}",
                flush=True,
            )
        del query, reference, state_valid, labels, label_index, row_weight
        del donor_reference, donor_valid, logits, wrong_logits, loss, primary, ce, alias, cross

    scheduler.step()
    return {
        "epoch": int(epoch),
        "loss": totals["loss"] / len(wells),
        "ce": totals["ce"] / len(wells),
        "alias": totals["alias"] / len(wells),
        "cross": totals["cross"] / len(wells),
        "skipped_steps": totals["skipped"],
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


def _fit_mixed_v1(wells, epochs, seed, phase):
    model, optimizer, scheduler, scaler = _new_training_objects(device, epochs, seed)
    rows = []
    for epoch in range(1, epochs + 1):
        row = _train_mixed_epoch_v1(
            model, wells, optimizer, scheduler, scaler, epoch, phase
        )
        rows.append(row)
        print("PREFIX MATCHER EPOCH", {"phase": phase, **row}, flush=True)
    del optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return model, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expose the 4x4 scale-pair evidence and fit an analytic ranking adapter
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_features_v1(model, sample, query_set, mode="full", donor_sample=None):
    reference_np = sample.reference
    valid_np = sample.state_valid
    no_gr = mode == "no_gr"
    if mode == "shuffled_typewell":
        if donor_sample is None:
            raise ValueError("shuffled_typewell requires donor_sample")
        reference_np = donor_sample.reference
        valid_np = donor_sample.state_valid
    query, reference, state_valid, _, _, _ = _to_tensors(
        sample,
        query_set,
        device,
        no_gr=no_gr,
        reference=reference_np,
        state_valid=valid_np,
    )
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        batch, q_scales, channels, points = query.shape
        states, r_scales, ref_channels, ref_points = reference.shape
        query_embedding = model.query_encoder(
            query.reshape(batch * q_scales, channels, points)
        ).reshape(batch, q_scales, -1)
        reference_embedding = model.reference_encoder(
            reference.reshape(states * r_scales, ref_channels, ref_points)
        ).reshape(states, r_scales, -1)
        correlation = torch.einsum(
            "bqe,ste->bsqt", query_embedding, reference_embedding
        )
        correlation = F.softplus(model.logit_scale) * correlation
        flat = correlation.flatten(2)
        maximum = flat.amax(dim=2, keepdim=True)
        mean = flat.mean(dim=2, keepdim=True)
        max_by_query = correlation.amax(dim=3)
        max_by_reference = correlation.amax(dim=2)
        query_coverage = query[:, :, 1].mean(dim=-1)
        reference_coverage = reference[:, :, 1].mean(dim=-1)
        query_coverage = query_coverage[:, None].expand(-1, states, -1)
        reference_coverage = reference_coverage[None].expand(batch, -1, -1)
        features = torch.cat(
            [
                flat,
                maximum,
                mean,
                max_by_query,
                max_by_reference,
                query_coverage,
                reference_coverage,
            ],
            dim=2,
        )
        logits = model.scorer(features).squeeze(-1)
        logits = logits.masked_fill(~state_valid[None], -1e4)
    return (
        logits.float().cpu().numpy(),
        features[:, :, :ADAPTER_DIMENSION_V1].float().cpu().numpy(),
        np.asarray(valid_np, bool),
    )


@dataclass
class PairStatsV1:
    xtx: np.ndarray
    xty: np.ndarray
    scale: np.ndarray
    pairs: int


@dataclass
class RetrievalBlockV1:
    logits: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    label_index: np.ndarray
    deep_mask: np.ndarray
    requested_rows: int


@dataclass
class WellEvidenceV1:
    well: str
    state_valid: np.ndarray
    full_stats: PairStatsV1
    context_stats: PairStatsV1
    wrong_stats: PairStatsV1
    late: RetrievalBlockV1
    suffix: RetrievalBlockV1


def _pair_stats_v1(logits, features, label_index, state_valid, shuffle_labels=False):
    logits = np.asarray(logits, np.float64)
    features = np.asarray(features, np.float64)
    label_index = np.asarray(label_index, np.int64).copy()
    if shuffle_labels and len(label_index) > 1:
        label_index = np.roll(label_index, max(1, len(label_index) // 3))
    design = []
    target = []
    pair_weight = []

    for row in range(len(label_index)):
        truth = int(label_index[row])
        if truth < 0 or truth >= N_STATES or not state_valid[truth]:
            continue
        negatives = []
        for offset in ALIAS_OFFSETS_FT:
            for sign in (-1, 1):
                candidate = truth + sign * int(offset)
                if 0 <= candidate < N_STATES and state_valid[candidate]:
                    negatives.append((candidate, 2.0))
        order = np.argsort(logits[row])[::-1]
        added = 0
        used = {candidate for candidate, _ in negatives}
        for candidate in order:
            candidate = int(candidate)
            if not state_valid[candidate] or abs(candidate - truth) <= 5 or candidate in used:
                continue
            negatives.append((candidate, 1.0))
            used.add(candidate)
            added += 1
            if added >= TOP_GLOBAL_NEGATIVES_V1:
                break
        for negative, weight in negatives:
            difference = features[row, truth] - features[row, negative]
            current_gap = logits[row, truth] - logits[row, negative]
            needed = max(PAIR_MARGIN_V1 - current_gap, 0.0)
            design.append(difference)
            target.append(needed)
            pair_weight.append(weight)

    dimension = features.shape[-1]
    if len(design) < dimension:
        return PairStatsV1(
            xtx=np.zeros((dimension, dimension), np.float64),
            xty=np.zeros(dimension, np.float64),
            scale=np.ones(dimension, np.float64),
            pairs=len(design),
        )
    design = np.asarray(design, np.float64)
    target = np.asarray(target, np.float64)
    pair_weight = np.sqrt(np.asarray(pair_weight, np.float64))
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale = np.where(scale < 1e-4, 1.0, scale)
    design = design / scale[None]
    weighted_design = design * pair_weight[:, None]
    weighted_target = target * pair_weight
    return PairStatsV1(
        xtx=weighted_design.T @ weighted_design,
        xty=weighted_design.T @ weighted_target,
        scale=scale,
        pairs=len(design),
    )


def _solve_adapter_v1(stats, ridge_lambda):
    dimension = len(stats.xty)
    if stats.pairs < dimension:
        return np.zeros(dimension, np.float64)
    system = stats.xtx + float(ridge_lambda) * np.eye(dimension)
    return np.linalg.solve(system + 1e-8 * np.eye(dimension), stats.xty)


def _adapt_scores_v1(block, stats, ridge_lambda, dose):
    coefficient = _solve_adapter_v1(stats, ridge_lambda)
    features = block.features.astype(np.float32) / stats.scale.astype(np.float32)[None, None]
    correction = np.einsum("bsd,d->bs", features, coefficient, optimize=True)
    correction = np.clip(correction, -ADAPTER_CORRECTION_CAP_V1, ADAPTER_CORRECTION_CAP_V1)
    return block.logits.astype(np.float32) + float(dose) * correction.astype(np.float32)


def _retrieval_block_v1(logits, features, query):
    return RetrievalBlockV1(
        logits=logits.astype(np.float16),
        features=features.astype(np.float16),
        labels=query.labels.astype(np.float32),
        label_index=query.label_index.astype(np.int64),
        deep_mask=query.deep_mask.astype(bool),
        requested_rows=int(query.requested_rows),
    )


def _build_evidence_v1(model, wells, prefix_queries, mode="full"):
    output = {}
    donor_wells = list(wells[1:]) + [wells[0]]
    for well, donor_well in tqdm(
        zip(wells, donor_wells), total=len(wells), desc=f"build evidence {mode}"
    ):
        sample = samples[well]
        donor_sample = samples[donor_well] if mode == "shuffled_typewell" else None
        prefix = prefix_queries[well]
        context, late = _prefix_split_v1(prefix)

        prefix_logits, prefix_features, state_valid = _score_features_v1(
            model, sample, prefix, mode=mode, donor_sample=donor_sample
        )
        context_count = len(context.labels)
        context_logits = prefix_logits[:context_count]
        context_features = prefix_features[:context_count]
        late_logits = prefix_logits[context_count:]
        late_features = prefix_features[context_count:]

        suffix = sample.evaluation
        if suffix is None:
            raise RuntimeError(f"{well}: missing suffix evaluation")
        suffix_logits, suffix_features, suffix_valid = _score_features_v1(
            model, sample, suffix, mode=mode, donor_sample=donor_sample
        )
        if not np.array_equal(state_valid, suffix_valid):
            raise RuntimeError(f"{well}: inconsistent state validity")
        output[well] = WellEvidenceV1(
            well=well,
            state_valid=state_valid,
            full_stats=_pair_stats_v1(
                prefix_logits, prefix_features, prefix.label_index, state_valid
            ),
            context_stats=_pair_stats_v1(
                context_logits, context_features, context.label_index, state_valid
            ),
            wrong_stats=_pair_stats_v1(
                prefix_logits,
                prefix_features,
                prefix.label_index,
                state_valid,
                shuffle_labels=True,
            ),
            late=_retrieval_block_v1(late_logits, late_features, late),
            suffix=_retrieval_block_v1(suffix_logits, suffix_features, suffix),
        )
    return output


def _summarise_evidence_v1(evidence, ridge_lambda, dose, target="suffix", stats="full_stats"):
    parts = []
    requested = 0
    bywell = []
    for well in sorted(evidence):
        item = evidence[well]
        block = getattr(item, target)
        pair_stats = getattr(item, stats)
        scores = _adapt_scores_v1(block, pair_stats, ridge_lambda, dose)
        payload = _retrieval_arrays(
            scores, block.labels, block.label_index, block.deep_mask
        )
        payload["well"] = np.full(len(block.labels), well, dtype="U16")
        parts.append(payload)
        requested += block.requested_rows
        metrics = _summarise_payload(payload, requested_rows=block.requested_rows)
        metrics.update(
            {
                "well": well,
                "target": target,
                "pairs": pair_stats.pairs,
                "coefficient_norm": float(
                    np.linalg.norm(_solve_adapter_v1(pair_stats, ridge_lambda))
                ),
            }
        )
        bywell.append(metrics)
    merged = _merge_payload(parts)
    return (
        _summarise_payload(merged, requested_rows=requested),
        pd.DataFrame(bywell),
        merged,
    )


# ---------------------------------------------------------------------------
# Execute nested mixed-training and adaptation gateway
# ---------------------------------------------------------------------------

started_v1 = time.time()
_seed_v1(SEED_V1)
mixed_epochs_v1 = max(3, int(selected_epoch))

print("building legal prefix correspondence banks", flush=True)
prefix_queries_v1 = {}
prefix_rows_v1 = []
for well in tqdm(sorted(samples), desc="prefix correspondence banks"):
    prefix = _legal_prefix_query_v1(well, samples[well].state_valid)
    prefix_queries_v1[well] = prefix
    context, late = _prefix_split_v1(prefix)
    prefix_rows_v1.append(
        {
            "well": well,
            "fold": samples[well].fold,
            "prefix_rows": len(prefix.labels),
            "context_rows": len(context.labels),
            "late_rows": len(late.labels),
            "label_min": float(prefix.labels.min()),
            "label_max": float(prefix.labels.max()),
        }
    )
prefix_report_v1 = pd.DataFrame(prefix_rows_v1)
prefix_report_v1.to_csv(OUT_DIR_V1 / "prefix_bank_report.csv", index=False)
guards_v1 = {
    "wells": int(len(prefix_report_v1)),
    "minimum_prefix_rows": int(prefix_report_v1["prefix_rows"].min()),
    "median_prefix_rows": float(prefix_report_v1["prefix_rows"].median()),
    "minimum_context_rows": int(prefix_report_v1["context_rows"].min()),
    "minimum_late_rows": int(prefix_report_v1["late_rows"].min()),
}
print("PREFIX BANK GUARDS", guards_v1, flush=True)
if (
    guards_v1["wells"] != 773
    or guards_v1["minimum_context_rows"] < 32
    or guards_v1["minimum_late_rows"] < 24
):
    raise RuntimeError(f"Prefix correspondence bank guard failed: {guards_v1}")

# Replace only outer-training query sets.  Fold1 remains untouched evaluation.
for well in outer_train_wells:
    suffix = samples[well].train
    if suffix is None:
        raise RuntimeError(f"{well}: missing original suffix training queries")
    prefix = _global_prefix_subset_v1(prefix_queries_v1[well])
    samples[well].train = _concat_query_v1(suffix, prefix)

print(
    "MIXED MATCHER CONFIG",
    {
        "epochs": mixed_epochs_v1,
        "suffix_rows_per_well": int(np.median([len(samples[w].train.labels) - GLOBAL_PREFIX_POSITIONS_V1 for w in outer_train_wells])),
        "prefix_rows_per_well": GLOBAL_PREFIX_POSITIONS_V1,
        "cross_well_weight": CROSS_WELL_WEIGHT_V1,
        "adapter_dimension": ADAPTER_DIMENSION_V1,
    },
    flush=True,
)

core_model_v1, core_history_v1 = _fit_mixed_v1(
    core_wells, mixed_epochs_v1, SEED_V1, "core"
)
core_history_v1.to_csv(OUT_DIR_V1 / "core_training_history.csv", index=False)

fold2_evidence_v1 = _build_evidence_v1(
    core_model_v1, tune_wells, prefix_queries_v1, mode="full"
)
scan_rows_v1 = []
for ridge_lambda_v1 in RIDGE_LAMBDAS_V1:
    for dose_v1 in ADAPTER_DOSES_V1:
        suffix_metrics_v1, _, _ = _summarise_evidence_v1(
            fold2_evidence_v1, ridge_lambda_v1, dose_v1, "suffix", "full_stats"
        )
        late_metrics_v1, _, _ = _summarise_evidence_v1(
            fold2_evidence_v1, ridge_lambda_v1, dose_v1, "late", "context_stats"
        )
        scan_rows_v1.append(
            {
                "ridge_lambda": ridge_lambda_v1,
                "dose": dose_v1,
                "suffix_top8": suffix_metrics_v1["effective_top8_within5"],
                "suffix_median_rank": suffix_metrics_v1["median_rank"],
                "suffix_alias24_win": suffix_metrics_v1["alias24_win_rate"],
                "suffix_argmax_rmse": suffix_metrics_v1["argmax_rmse"],
                "late_top8": late_metrics_v1["effective_top8_within5"],
                "late_median_rank": late_metrics_v1["median_rank"],
                "late_alias24_win": late_metrics_v1["alias24_win_rate"],
            }
        )
scan_v1 = pd.DataFrame(scan_rows_v1).sort_values(
    ["suffix_top8", "suffix_median_rank", "late_top8", "suffix_alias24_win"],
    ascending=[False, True, False, False],
)
scan_v1.to_csv(OUT_DIR_V1 / "fold2_adapter_grid.csv", index=False)
selected_v1 = scan_v1.iloc[0].to_dict()
selected_lambda_v1 = float(selected_v1["ridge_lambda"])
selected_dose_v1 = float(selected_v1["dose"])
print("FOLD2 ADAPTER GRID")
print(scan_v1.to_string(index=False), flush=True)
print("FROZEN ADAPTER", selected_v1, flush=True)

del fold2_evidence_v1, core_model_v1
gc.collect()
torch.cuda.empty_cache()

final_model_v1, final_history_v1 = _fit_mixed_v1(
    outer_train_wells, mixed_epochs_v1, SEED_V1 + 1000, "final"
)
final_history_v1.to_csv(OUT_DIR_V1 / "final_training_history.csv", index=False)
checkpoint_v1 = MODEL_DIR_V1 / "fold1_prefix_contrastive_matcher.pt"
torch.save(
    {
        "model": {key: value.detach().cpu() for key, value in final_model_v1.state_dict().items()},
        "epochs": mixed_epochs_v1,
        "ridge_lambda": selected_lambda_v1,
        "adapter_dose": selected_dose_v1,
        "adapter_dimension": ADAPTER_DIMENSION_V1,
        "fold_source": str(fold_path),
    },
    checkpoint_v1,
)

fold1_full_evidence_v1 = _build_evidence_v1(
    final_model_v1, test_wells, prefix_queries_v1, mode="full"
)
frozen_metrics_v1, frozen_bywell_v1, frozen_payload_v1 = _summarise_evidence_v1(
    fold1_full_evidence_v1, selected_lambda_v1, 0.0, "suffix", "full_stats"
)
adapted_metrics_v1, adapted_bywell_v1, adapted_payload_v1 = _summarise_evidence_v1(
    fold1_full_evidence_v1,
    selected_lambda_v1,
    selected_dose_v1,
    "suffix",
    "full_stats",
)
wrong_metrics_v1, wrong_bywell_v1, wrong_payload_v1 = _summarise_evidence_v1(
    fold1_full_evidence_v1,
    selected_lambda_v1,
    selected_dose_v1,
    "suffix",
    "wrong_stats",
)
late_frozen_metrics_v1, _, _ = _summarise_evidence_v1(
    fold1_full_evidence_v1, selected_lambda_v1, 0.0, "late", "context_stats"
)
late_adapted_metrics_v1, late_bywell_v1, _ = _summarise_evidence_v1(
    fold1_full_evidence_v1,
    selected_lambda_v1,
    selected_dose_v1,
    "late",
    "context_stats",
)

adapted_bywell_v1.to_csv(OUT_DIR_V1 / "fold1_bywell_adapted.csv", index=False)
frozen_bywell_v1.to_csv(OUT_DIR_V1 / "fold1_bywell_frozen.csv", index=False)
wrong_bywell_v1.to_csv(OUT_DIR_V1 / "fold1_bywell_wrong_prefix.csv", index=False)
late_bywell_v1.to_csv(OUT_DIR_V1 / "fold1_bywell_late_prefix.csv", index=False)

# Controls are built and released one at a time to keep host memory bounded.
fold1_no_gr_evidence_v1 = _build_evidence_v1(
    final_model_v1, test_wells, prefix_queries_v1, mode="no_gr"
)
no_gr_metrics_v1, _, no_gr_payload_v1 = _summarise_evidence_v1(
    fold1_no_gr_evidence_v1,
    selected_lambda_v1,
    selected_dose_v1,
    "suffix",
    "full_stats",
)
del fold1_no_gr_evidence_v1
gc.collect()

fold1_shuffle_evidence_v1 = _build_evidence_v1(
    final_model_v1, test_wells, prefix_queries_v1, mode="shuffled_typewell"
)
shuffle_metrics_v1, _, shuffle_payload_v1 = _summarise_evidence_v1(
    fold1_shuffle_evidence_v1,
    selected_lambda_v1,
    selected_dose_v1,
    "suffix",
    "full_stats",
)
del fold1_shuffle_evidence_v1
gc.collect()

comparison_v1 = pd.DataFrame(
    [
        {"method": "mixed_frozen", **frozen_metrics_v1},
        {"method": "prefix_adapted", **adapted_metrics_v1},
        {"method": "wrong_prefix_adapter", **wrong_metrics_v1},
        {"method": "adapted_no_gr", **no_gr_metrics_v1},
        {"method": "adapted_shuffled_typewell", **shuffle_metrics_v1},
    ]
)
comparison_v1.to_csv(OUT_DIR_V1 / "fold1_method_comparison.csv", index=False)
np.savez_compressed(
    OUT_DIR_V1 / "fold1_retrieval_payloads.npz",
    label=adapted_payload_v1["label"],
    well=adapted_payload_v1["well"],
    adapted_prediction=adapted_payload_v1["prediction"],
    adapted_rank=adapted_payload_v1["rank"],
    frozen_prediction=frozen_payload_v1["prediction"],
    frozen_rank=frozen_payload_v1["rank"],
    wrong_prediction=wrong_payload_v1["prediction"],
    no_gr_prediction=no_gr_payload_v1["prediction"],
    shuffled_typewell_prediction=shuffle_payload_v1["prediction"],
)

adapter_gain_v1 = (
    adapted_metrics_v1["effective_top8_within5"]
    - frozen_metrics_v1["effective_top8_within5"]
)
late_gain_v1 = (
    late_adapted_metrics_v1["effective_top8_within5"]
    - late_frozen_metrics_v1["effective_top8_within5"]
)
no_gr_drop_v1 = (
    adapted_metrics_v1["effective_top8_within5"]
    - no_gr_metrics_v1["effective_top8_within5"]
)
shuffle_drop_v1 = (
    adapted_metrics_v1["effective_top8_within5"]
    - shuffle_metrics_v1["effective_top8_within5"]
)
wrong_drop_v1 = (
    adapted_metrics_v1["effective_top8_within5"]
    - wrong_metrics_v1["effective_top8_within5"]
)
gates_v1 = {
    "top8": adapted_metrics_v1["effective_top8_within5"] >= TOP8_GATE_V1,
    "median_rank": adapted_metrics_v1["median_rank"] <= MEDIAN_RANK_GATE_V1,
    "alias24": adapted_metrics_v1["alias24_win_rate"] >= ALIAS_GATE_V1,
    "adapter_gain": adapter_gain_v1 >= ADAPTER_GAIN_GATE_V1,
    "late_prefix_gain": late_gain_v1 >= LATE_PREFIX_GAIN_GATE_V1,
    "no_gr_control": no_gr_drop_v1 >= NO_GR_DROP_GATE_V1,
    "typewell_shuffle_control": shuffle_drop_v1 >= TYPEWELL_SHUFFLE_DROP_GATE_V1,
    "wrong_prefix_control": wrong_drop_v1 >= WRONG_PREFIX_DROP_GATE_V1,
}
strict_pass_v1 = bool(all(gates_v1.values()))

verdict_v1 = {
    "experiment": EXPERIMENT_V1,
    "stage": "nested_prefix_supervised_retrieval_and_analytic_adaptation",
    "sample_unit": "complete physical well; sampled correspondences within well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "protocol": {
        "global_epoch_source": "v0 Fold2-selected epoch",
        "epochs": mixed_epochs_v1,
        "core_train_folds": [3, 4, 5],
        "adapter_selection_fold": 2,
        "final_train_folds": [2, 3, 4, 5],
        "untouched_test_fold": 1,
        "selected_lambda": selected_lambda_v1,
        "selected_dose": selected_dose_v1,
    },
    "training": {
        "prefix_positions": GLOBAL_PREFIX_POSITIONS_V1,
        "suffix_positions": "v0 official suffix bank",
        "alias_offsets_ft": list(ALIAS_OFFSETS_FT),
        "cross_well_negative_weight": CROSS_WELL_WEIGHT_V1,
    },
    "adapter": {
        "dimension": ADAPTER_DIMENSION_V1,
        "fit_inputs": "visible prefix truth-state pairwise deficits only",
        "network_updates_at_inference": 0,
        "correction_cap": ADAPTER_CORRECTION_CAP_V1,
    },
    "guards": guards_v1,
    "fold2_selection": selected_v1,
    "fold1": {
        "mixed_frozen": frozen_metrics_v1,
        "prefix_adapted": adapted_metrics_v1,
        "wrong_prefix_adapter": wrong_metrics_v1,
        "adapted_no_gr": no_gr_metrics_v1,
        "adapted_shuffled_typewell": shuffle_metrics_v1,
        "late_prefix_frozen": late_frozen_metrics_v1,
        "late_prefix_adapted": late_adapted_metrics_v1,
    },
    "diagnostics": {
        "adapter_top8_gain": adapter_gain_v1,
        "late_prefix_top8_gain": late_gain_v1,
        "top8_drop_no_gr": no_gr_drop_v1,
        "top8_drop_shuffled_typewell": shuffle_drop_v1,
        "top8_drop_wrong_prefix": wrong_drop_v1,
    },
    "gate_thresholds": {
        "effective_top8_within5_ge": TOP8_GATE_V1,
        "median_rank_le": MEDIAN_RANK_GATE_V1,
        "alias24_win_ge": ALIAS_GATE_V1,
        "adapter_top8_gain_ge": ADAPTER_GAIN_GATE_V1,
        "late_prefix_top8_gain_ge": LATE_PREFIX_GAIN_GATE_V1,
        "no_gr_top8_drop_ge": NO_GR_DROP_GATE_V1,
        "typewell_shuffle_top8_drop_ge": TYPEWELL_SHUFFLE_DROP_GATE_V1,
        "wrong_prefix_top8_drop_ge": WRONG_PREFIX_DROP_GATE_V1,
    },
    "gates": gates_v1,
    "strict_pass": strict_pass_v1,
    "checkpoint": str(checkpoint_v1),
    "elapsed_minutes": (time.time() - started_v1) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze matcher and integrate its posterior into one fixed global path solver"
        if strict_pass_v1
        else "FAIL -> explicit prefix supervision/adaptation cannot reach the retrieval gate"
    ),
}
_write_json(OUT_DIR_V1 / "verdict.json", verdict_v1)

print("\nPREFIX MATCHER FOLD1 COMPARISON")
print(comparison_v1.to_string(index=False), flush=True)
print("\nPREFIX MATCHER FINAL VERDICT")
print(json.dumps(verdict_v1, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR_V1, flush=True)
