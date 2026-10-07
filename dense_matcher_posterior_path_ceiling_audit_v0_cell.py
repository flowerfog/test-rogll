"""Dense posterior path ceiling audit for the prefix-contrastive matcher.

Run in the same live kernel after
``prefix_contrastive_matcher_adaptation_gateway_v1_cell.py``.

No new neural parameters are introduced.  The cell exports 257-state matcher
scores at deterministic 32-ft nodes, freezes one physical transition dose on
Fold 2, and applies it once to Fold 1.  It also computes physically constrained
K=8/16/32/64 trajectory oracles to distinguish a bad menu from a bad selector.

Transitions are calibrated in S = TVT + Z space.  This prevents known vertical
geometry from being mistaken for a geological path jump.
"""

from dataclasses import dataclass
from pathlib import Path
import gc
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Live-kernel contract and configuration
# ---------------------------------------------------------------------------

REQUIRED_DENSE = {
    "OOF_ROOT",
    "RECORD_DIR",
    "samples",
    "core_wells",
    "tune_wells",
    "outer_train_wells",
    "test_wells",
    "final_model_v1",
    "prefix_queries_v1",
    "mixed_epochs_v1",
    "selected_lambda_v1",
    "selected_dose_v1",
    "_fit_mixed_v1",
    "_load_raw_record",
    "_window_bank",
    "_score_features_v1",
    "_pair_stats_v1",
    "_adapt_scores_v1",
    "RetrievalBlockV1",
    "QuerySet",
    "QUERY_HALF_WIDTHS_MD",
    "STATE_OFFSETS",
    "N_STATES",
    "STATE_HALF_WIDTH_FT",
    "_write_json",
    "_json_default",
}
missing_dense = sorted(REQUIRED_DENSE.difference(globals()))
if missing_dense:
    raise RuntimeError(
        "Run the prefix-contrastive matcher v1 cell in this live kernel first; "
        "missing: " + repr(missing_dense)
    )

EXPERIMENT_DENSE = "dense_matcher_posterior_path_ceiling_audit_v0"
OUT_DIR_DENSE = OOF_ROOT / EXPERIMENT_DENSE
OUT_DIR_DENSE.mkdir(parents=True, exist_ok=True)

SEED_DENSE = 20260806
NODE_STEP_FT_DENSE = 32.0
ZERO_STATE_INDEX_DENSE = int(STATE_HALF_WIDTH_FT)
TRANSITION_LAMBDAS_DENSE = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0)
EMISSION_MODES_DENSE = ("frozen", "adapted")
TOPK_VALUES_DENSE = (8, 16, 32, 64)

K32_ORACLE_GATE_DENSE = 5.5
K64_ORACLE_GATE_DENSE = 5.0
TOP64_NODE_COVERAGE_GATE_DENSE = 0.90
VITERBI_RMSE_GATE_DENSE = 10.0
FAR_GAIN_GATE_DENSE = 1.0
POSITIVE_WELL_GATE_DENSE = 0.60


@dataclass
class DenseQueryD0:
    well: str
    query: QuerySet
    node_rows: np.ndarray
    node_md: np.ndarray
    node_z: np.ndarray
    state_valid: np.ndarray
    axis_coverage: float


@dataclass
class DenseVolumeD0:
    well: str
    node_rows: np.ndarray
    node_md: np.ndarray
    node_z: np.ndarray
    truth_offset: np.ndarray
    state_valid: np.ndarray
    frozen_logits: np.ndarray
    adapted_logits: np.ndarray


@dataclass
class TransitionPriorD0:
    slope_centre: float
    residual_scale: float
    residual_cap: int
    edges: int
    residual_p99: float
    residual_p999: float


def _rmse_dense(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _native_z_dense(well):
    with np.load(RECORD_DIR / f"{well}.npz", allow_pickle=False) as data:
        condition = data["condition"].astype(np.float32)
    if condition.shape[0] < 5:
        raise RuntimeError(f"{well}: native condition has {condition.shape[0]} channels")
    return condition[4].astype(np.float64) * 512.0


def _dense_query_d0(well):
    record = _load_raw_record(well)
    anchor_md = float(record.md[record.anchor_index])
    final_md = float(record.md[-1])
    node_md = np.arange(
        anchor_md,
        final_md + 0.25 * NODE_STEP_FT_DENSE,
        NODE_STEP_FT_DENSE,
        dtype=np.float64,
    )
    node_md = node_md[node_md <= final_md + 1e-6]
    if final_md - node_md[-1] > 0.5 * NODE_STEP_FT_DENSE:
        node_md = np.r_[node_md, final_md]
    node_rows = np.searchsorted(record.md, node_md, side="left")
    node_rows = np.clip(node_rows, record.anchor_index, len(record.md) - 1)
    node_rows = np.unique(node_rows)
    if node_rows[0] != record.anchor_index:
        node_rows = np.r_[record.anchor_index, node_rows]
    node_md = record.md[node_rows].astype(np.float64)

    labels = record.truth[node_rows].astype(np.float64) - float(record.anchor_tvt)
    label_index = np.rint(labels + STATE_HALF_WIDTH_FT).astype(np.int64)
    axis_valid = np.isfinite(labels) & (label_index >= 0) & (label_index < N_STATES)
    safe_index = np.clip(label_index, 0, N_STATES - 1)
    axis_valid &= samples[well].state_valid[safe_index]
    axis_coverage = float(np.mean(axis_valid))
    if axis_coverage < 0.98:
        raise RuntimeError(f"{well}: dense state-axis coverage={axis_coverage:.4f}")

    patches = _window_bank(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        node_md,
        QUERY_HALF_WIDTHS_MD,
    )
    distance = node_md - anchor_md
    query = QuerySet(
        patches=patches,
        labels=labels.astype(np.float32),
        label_index=np.clip(label_index, 0, N_STATES - 1).astype(np.int64),
        row_weight=np.ones(len(node_rows), np.float32),
        deep_mask=(distance >= 1500.0),
        sample_rows=node_rows.astype(np.int32),
        requested_rows=int(len(node_rows)),
    )
    z = _native_z_dense(well)[node_rows]
    return DenseQueryD0(
        well=well,
        query=query,
        node_rows=node_rows.astype(np.int32),
        node_md=node_md,
        node_z=z,
        state_valid=samples[well].state_valid.astype(bool),
        axis_coverage=axis_coverage,
    )


def _build_volume_d0(model, well):
    dense = _dense_query_d0(well)
    sample = samples[well]
    frozen_logits, dense_features, state_valid = _score_features_v1(
        model, sample, dense.query, mode="full"
    )
    prefix = prefix_queries_v1[well]
    prefix_logits, prefix_features, prefix_valid = _score_features_v1(
        model, sample, prefix, mode="full"
    )
    if not np.array_equal(state_valid, prefix_valid):
        raise RuntimeError(f"{well}: prefix/dense validity mismatch")
    stats = _pair_stats_v1(
        prefix_logits, prefix_features, prefix.label_index, state_valid
    )
    block = RetrievalBlockV1(
        logits=frozen_logits,
        features=dense_features,
        labels=dense.query.labels,
        label_index=dense.query.label_index,
        deep_mask=dense.query.deep_mask,
        requested_rows=dense.query.requested_rows,
    )
    adapted_logits = _adapt_scores_v1(
        block, stats, selected_lambda_v1, selected_dose_v1
    )
    return DenseVolumeD0(
        well=well,
        node_rows=dense.node_rows,
        node_md=dense.node_md,
        node_z=dense.node_z,
        truth_offset=dense.query.labels.astype(np.float32),
        state_valid=state_valid,
        frozen_logits=frozen_logits.astype(np.float16),
        adapted_logits=adapted_logits.astype(np.float16),
    )


def _build_volumes_d0(model, wells, label):
    output = {}
    rows = []
    for well in tqdm(wells, desc=f"dense matcher volumes {label}"):
        volume = _build_volume_d0(model, well)
        output[well] = volume
        rows.append(
            {
                "well": well,
                "nodes": len(volume.node_md),
                "axis_coverage": float(
                    np.mean(
                        (volume.truth_offset >= -STATE_HALF_WIDTH_FT)
                        & (volume.truth_offset <= STATE_HALF_WIDTH_FT)
                    )
                ),
                "span_ft": float(volume.node_md[-1] - volume.node_md[0]),
            }
        )
    return output, pd.DataFrame(rows)


def _fit_transition_prior_d0(wells):
    slopes = []
    edge_payload = []
    for well in wells:
        dense = _dense_query_d0(well)
        s = dense.query.labels.astype(np.float64) + dense.node_z
        delta_md = np.diff(dense.node_md)
        delta_s = np.diff(s)
        valid = np.isfinite(delta_s) & (delta_md > 0)
        slopes.append(delta_s[valid] / delta_md[valid])
        edge_payload.append((delta_md[valid], delta_s[valid]))
    slopes = np.concatenate(slopes)
    slope_centre = float(np.median(slopes))
    residual = np.concatenate(
        [delta_s - slope_centre * delta_md for delta_md, delta_s in edge_payload]
    )
    abs_residual = np.abs(residual)
    residual_scale = float(max(np.quantile(abs_residual, 0.68), 0.75))
    p99 = float(np.quantile(abs_residual, 0.99))
    p999 = float(np.quantile(abs_residual, 0.999))
    residual_cap = int(np.clip(math.ceil(p999 + 2.0), 8, 40))
    return TransitionPriorD0(
        slope_centre=slope_centre,
        residual_scale=residual_scale,
        residual_cap=residual_cap,
        edges=len(residual),
        residual_p99=p99,
        residual_p999=p999,
    )


def _emission_d0(volume, mode):
    logits = (
        volume.frozen_logits.astype(np.float64)
        if mode == "frozen"
        else volume.adapted_logits.astype(np.float64)
    )
    logits[:, ~volume.state_valid] = -1e4
    logits -= logits.max(axis=1, keepdims=True)
    return logits


def _shift_slices_d0(shift, states=N_STATES):
    if shift >= 0:
        return slice(0, states - shift), slice(shift, states)
    return slice(-shift, states), slice(0, states + shift)


def _viterbi_d0(volume, mode, prior, transition_lambda):
    emission = _emission_d0(volume, mode)
    nodes, states = emission.shape
    dp = np.full(states, -np.inf, np.float64)
    dp[ZERO_STATE_INDEX_DENSE] = 0.0
    back = np.full((nodes, states), -1, np.int16)

    for node in range(1, nodes):
        delta_md = float(volume.node_md[node] - volume.node_md[node - 1])
        delta_z = float(volume.node_z[node] - volume.node_z[node - 1])
        expected_delta_tvt = prior.slope_centre * delta_md - delta_z
        centre_shift = int(round(expected_delta_tvt))
        new = np.full(states, -np.inf, np.float64)
        new_back = np.full(states, -1, np.int16)
        for shift in range(
            centre_shift - prior.residual_cap,
            centre_shift + prior.residual_cap + 1,
        ):
            if abs(shift) >= states:
                continue
            source, target = _shift_slices_d0(shift, states)
            residual_s = shift + delta_z - prior.slope_centre * delta_md
            penalty = float(transition_lambda) * (residual_s / prior.residual_scale) ** 2
            candidate = dp[source] - penalty + emission[node, target]
            better = candidate > new[target]
            if np.any(better):
                target_index = np.arange(states)[target][better]
                source_index = target_index - shift
                new[target_index] = candidate[better]
                new_back[target_index] = source_index.astype(np.int16)
        new[~volume.state_valid] = -np.inf
        if not np.isfinite(new).any():
            raise RuntimeError(f"{volume.well}: disconnected Viterbi at node {node}")
        dp = new
        back[node] = new_back

    path = np.empty(nodes, np.int16)
    path[-1] = int(np.argmax(dp))
    for node in range(nodes - 1, 0, -1):
        previous = int(back[node, path[node]])
        if previous < 0:
            raise RuntimeError(f"{volume.well}: invalid backpointer at node {node}")
        path[node - 1] = previous
    if path[0] != ZERO_STATE_INDEX_DENSE:
        raise RuntimeError(f"{volume.well}: Viterbi lost the anchor")
    return STATE_OFFSETS[path].astype(np.float64)


def _oracle_attempt_d0(volume, mode, prior, topk, residual_cap):
    logits = _emission_d0(volume, mode)
    nodes, states = logits.shape
    top_index = np.argpartition(logits, -topk, axis=1)[:, -topk:]
    allowed = np.zeros((nodes, states), bool)
    allowed[np.arange(nodes)[:, None], top_index] = True
    allowed[0] = False
    allowed[0, ZERO_STATE_INDEX_DENSE] = True
    unary = (STATE_OFFSETS[None].astype(np.float64) - volume.truth_offset[:, None]) ** 2
    unary[~allowed] = np.inf

    dp = np.full(states, np.inf, np.float64)
    dp[ZERO_STATE_INDEX_DENSE] = 0.0
    back = np.full((nodes, states), -1, np.int16)
    for node in range(1, nodes):
        delta_md = float(volume.node_md[node] - volume.node_md[node - 1])
        delta_z = float(volume.node_z[node] - volume.node_z[node - 1])
        expected_delta_tvt = prior.slope_centre * delta_md - delta_z
        centre_shift = int(round(expected_delta_tvt))
        new = np.full(states, np.inf, np.float64)
        new_back = np.full(states, -1, np.int16)
        for shift in range(centre_shift - residual_cap, centre_shift + residual_cap + 1):
            if abs(shift) >= states:
                continue
            source, target = _shift_slices_d0(shift, states)
            residual_s = shift + delta_z - prior.slope_centre * delta_md
            penalty = 0.01 * (residual_s / prior.residual_scale) ** 2
            candidate = dp[source] + penalty + unary[node, target]
            better = candidate < new[target]
            if np.any(better):
                target_index = np.arange(states)[target][better]
                source_index = target_index - shift
                new[target_index] = candidate[better]
                new_back[target_index] = source_index.astype(np.int16)
        if not np.isfinite(new).any():
            return None
        dp = new
        back[node] = new_back
    path = np.empty(nodes, np.int16)
    path[-1] = int(np.argmin(dp))
    for node in range(nodes - 1, 0, -1):
        previous = int(back[node, path[node]])
        if previous < 0:
            return None
        path[node - 1] = previous
    return STATE_OFFSETS[path].astype(np.float64)


def _topk_oracle_d0(volume, mode, prior, topk):
    for cap in (max(48, prior.residual_cap), 80, 128, 256):
        path = _oracle_attempt_d0(volume, mode, prior, topk, cap)
        if path is not None:
            return path, cap
    raise RuntimeError(f"{volume.well}: top{topk} oracle graph disconnected")


def _path_payload_d0(volume, node_offset):
    record = _load_raw_record(volume.well)
    eval_rows = np.arange(record.anchor_index + 1, len(record.md), dtype=np.int64)
    eval_md = record.md[eval_rows].astype(np.float64)
    prediction_offset = np.interp(eval_md, volume.node_md, node_offset)
    prediction = float(record.anchor_tvt) + prediction_offset
    truth = record.truth[eval_rows].astype(np.float64)
    distance = eval_md - float(record.md[record.anchor_index])
    return {
        "well": np.full(len(eval_rows), volume.well, dtype="U16"),
        "truth": truth.astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "distance": distance.astype(np.float32),
    }


def _summarise_paths_d0(payloads):
    merged = {
        key: np.concatenate([payload[key] for payload in payloads])
        for key in payloads[0]
    }
    truth = merged["truth"].astype(np.float64)
    prediction = merged["prediction"].astype(np.float64)
    distance = merged["distance"].astype(np.float64)
    bywell = []
    for well in np.unique(merged["well"]):
        mask = merged["well"] == well
        far = mask & (distance > 2000)
        bywell.append(
            {
                "well": well,
                "rmse": _rmse_dense(truth[mask], prediction[mask]),
                "far_rmse": _rmse_dense(truth[far], prediction[far]) if far.any() else np.nan,
            }
        )
    bywell = pd.DataFrame(bywell)

    def subset(mask):
        return _rmse_dense(truth[mask], prediction[mask]) if np.any(mask) else np.nan

    metrics = {
        "rows": len(truth),
        "wells": int(len(bywell)),
        "rmse": _rmse_dense(truth, prediction),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "near_0_500_rmse": subset(distance <= 500),
        "mid_500_1000_rmse": subset((distance > 500) & (distance <= 1000)),
        "mid_1000_2000_rmse": subset((distance > 1000) & (distance <= 2000)),
        "far_gt2000_rmse": subset(distance > 2000),
    }
    return metrics, bywell, merged


def _evaluate_viterbi_d0(volumes, mode, prior, transition_lambda):
    payloads = []
    for well in sorted(volumes):
        path = _viterbi_d0(volumes[well], mode, prior, transition_lambda)
        payloads.append(_path_payload_d0(volumes[well], path))
    return _summarise_paths_d0(payloads)


def _evaluate_anchor_d0(volumes):
    payloads = []
    for well in sorted(volumes):
        zeros = np.zeros(len(volumes[well].node_md), np.float64)
        payloads.append(_path_payload_d0(volumes[well], zeros))
    return _summarise_paths_d0(payloads)


def _evaluate_oracle_d0(volumes, mode, prior, topk):
    payloads = []
    caps = []
    for well in sorted(volumes):
        path, cap = _topk_oracle_d0(volumes[well], mode, prior, topk)
        caps.append(cap)
        payloads.append(_path_payload_d0(volumes[well], path))
    metrics, bywell, payload = _summarise_paths_d0(payloads)
    metrics["topk"] = int(topk)
    metrics["median_solver_cap"] = float(np.median(caps))
    metrics["max_solver_cap"] = int(np.max(caps))
    return metrics, bywell, payload


def _node_coverage_d0(volumes, mode, topk):
    hits = []
    distances = []
    ranks = []
    for volume in volumes.values():
        logits = _emission_d0(volume, mode)
        index = np.argpartition(logits, -topk, axis=1)[:, -topk:]
        values = STATE_OFFSETS[index]
        hits.append(np.any(np.abs(values - volume.truth_offset[:, None]) <= 5.0, axis=1))
        distances.append(volume.node_md - volume.node_md[0])
        truth_index = np.clip(
            np.rint(volume.truth_offset + STATE_HALF_WIDTH_FT).astype(int),
            0,
            N_STATES - 1,
        )
        truth_score = logits[np.arange(len(logits)), truth_index]
        ranks.append(1 + np.sum(logits > truth_score[:, None], axis=1))
    hit = np.concatenate(hits)
    distance = np.concatenate(distances)
    rank = np.concatenate(ranks)

    def fraction(mask):
        return float(np.mean(hit[mask])) if np.any(mask) else np.nan

    return {
        "topk": int(topk),
        "nodes": int(len(hit)),
        "within5": float(np.mean(hit)),
        "median_rank": float(np.median(rank)),
        "p90_rank": float(np.quantile(rank, 0.90)),
        "near_0_500": fraction(distance <= 500),
        "mid_500_1000": fraction((distance > 500) & (distance <= 1000)),
        "mid_1000_2000": fraction((distance > 1000) & (distance <= 2000)),
        "far_gt2000": fraction(distance > 2000),
    }


def _positive_well_fraction_d0(base_bywell, candidate_bywell):
    merged = base_bywell[["well", "rmse"]].merge(
        candidate_bywell[["well", "rmse"]], on="well", suffixes=("_base", "_candidate")
    )
    return float(np.mean(merged["rmse_candidate"] < merged["rmse_base"]))


# ---------------------------------------------------------------------------
# Execute nested selection and untouched Fold1 audit
# ---------------------------------------------------------------------------

started_dense = time.time()
print("fitting S-space transition priors", flush=True)
core_prior_dense = _fit_transition_prior_d0(core_wells)
outer_prior_dense = _fit_transition_prior_d0(outer_train_wells)
print("CORE TRANSITION PRIOR", core_prior_dense, flush=True)
print("OUTER TRANSITION PRIOR", outer_prior_dense, flush=True)

print("refitting core mixed matcher for Fold2 path selection", flush=True)
core_model_dense, core_history_dense = _fit_mixed_v1(
    core_wells, mixed_epochs_v1, SEED_DENSE, "dense_core"
)
core_history_dense.to_csv(OUT_DIR_DENSE / "core_matcher_history.csv", index=False)
fold2_volumes_dense, fold2_volume_report_dense = _build_volumes_d0(
    core_model_dense, tune_wells, "fold2"
)
fold2_volume_report_dense.to_csv(OUT_DIR_DENSE / "fold2_volume_report.csv", index=False)

grid_rows_dense = []
for mode_dense in EMISSION_MODES_DENSE:
    for transition_lambda_dense in TRANSITION_LAMBDAS_DENSE:
        metrics_dense, _, _ = _evaluate_viterbi_d0(
            fold2_volumes_dense,
            mode_dense,
            core_prior_dense,
            transition_lambda_dense,
        )
        grid_rows_dense.append(
            {
                "mode": mode_dense,
                "transition_lambda": transition_lambda_dense,
                **metrics_dense,
            }
        )
        print(
            "FOLD2 DENSE PATH",
            {
                "mode": mode_dense,
                "lambda": transition_lambda_dense,
                "rmse": metrics_dense["rmse"],
                "far": metrics_dense["far_gt2000_rmse"],
            },
            flush=True,
        )
grid_dense = pd.DataFrame(grid_rows_dense).sort_values(
    ["rmse", "far_gt2000_rmse", "p95_well_rmse"]
)
grid_dense.to_csv(OUT_DIR_DENSE / "fold2_viterbi_grid.csv", index=False)
selected_dense = grid_dense.iloc[0].to_dict()
selected_mode_dense = str(selected_dense["mode"])
selected_lambda_dense = float(selected_dense["transition_lambda"])
print("FROZEN DENSE PATH CONFIG", selected_dense, flush=True)

del fold2_volumes_dense, core_model_dense
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass

fold1_volumes_dense, fold1_volume_report_dense = _build_volumes_d0(
    final_model_v1, test_wells, "fold1"
)
fold1_volume_report_dense.to_csv(OUT_DIR_DENSE / "fold1_volume_report.csv", index=False)
anchor_metrics_dense, anchor_bywell_dense, anchor_payload_dense = _evaluate_anchor_d0(
    fold1_volumes_dense
)
viterbi_metrics_dense, viterbi_bywell_dense, viterbi_payload_dense = _evaluate_viterbi_d0(
    fold1_volumes_dense,
    selected_mode_dense,
    outer_prior_dense,
    selected_lambda_dense,
)
positive_fraction_dense = _positive_well_fraction_d0(
    anchor_bywell_dense, viterbi_bywell_dense
)

oracle_rows_dense = []
oracle_payloads_dense = {}
for topk_dense in TOPK_VALUES_DENSE:
    metrics_dense, bywell_dense, payload_dense = _evaluate_oracle_d0(
        fold1_volumes_dense,
        selected_mode_dense,
        outer_prior_dense,
        topk_dense,
    )
    oracle_rows_dense.append(metrics_dense)
    oracle_payloads_dense[topk_dense] = payload_dense
    bywell_dense.to_csv(OUT_DIR_DENSE / f"fold1_top{topk_dense}_oracle_bywell.csv", index=False)
    print("FOLD1 TOPK ORACLE", metrics_dense, flush=True)
oracle_report_dense = pd.DataFrame(oracle_rows_dense).sort_values("topk")
oracle_report_dense.to_csv(OUT_DIR_DENSE / "fold1_topk_trajectory_oracles.csv", index=False)

coverage_report_dense = pd.DataFrame(
    [
        _node_coverage_d0(fold1_volumes_dense, selected_mode_dense, topk_dense)
        for topk_dense in TOPK_VALUES_DENSE
    ]
)
coverage_report_dense.to_csv(OUT_DIR_DENSE / "fold1_node_topk_coverage.csv", index=False)
anchor_bywell_dense.to_csv(OUT_DIR_DENSE / "fold1_anchor_bywell.csv", index=False)
viterbi_bywell_dense.to_csv(OUT_DIR_DENSE / "fold1_viterbi_bywell.csv", index=False)

np.savez_compressed(
    OUT_DIR_DENSE / "fold1_path_payloads.npz",
    well=viterbi_payload_dense["well"],
    truth=viterbi_payload_dense["truth"],
    distance=viterbi_payload_dense["distance"],
    anchor=anchor_payload_dense["prediction"],
    viterbi=viterbi_payload_dense["prediction"],
    top32_oracle=oracle_payloads_dense[32]["prediction"],
    top64_oracle=oracle_payloads_dense[64]["prediction"],
)

top32_metrics_dense = next(row for row in oracle_rows_dense if row["topk"] == 32)
top64_metrics_dense = next(row for row in oracle_rows_dense if row["topk"] == 64)
top64_coverage_dense = float(
    coverage_report_dense.loc[coverage_report_dense["topk"].eq(64), "within5"].iloc[0]
)
far_gain_dense = float(
    anchor_metrics_dense["far_gt2000_rmse"] - viterbi_metrics_dense["far_gt2000_rmse"]
)
gates_dense = {
    "top32_oracle": top32_metrics_dense["rmse"] <= K32_ORACLE_GATE_DENSE,
    "top64_oracle": top64_metrics_dense["rmse"] <= K64_ORACLE_GATE_DENSE,
    "top64_node_coverage": top64_coverage_dense >= TOP64_NODE_COVERAGE_GATE_DENSE,
    "viterbi_path": viterbi_metrics_dense["rmse"] <= VITERBI_RMSE_GATE_DENSE,
    "far_gain": far_gain_dense >= FAR_GAIN_GATE_DENSE,
    "positive_wells": positive_fraction_dense >= POSITIVE_WELL_GATE_DENSE,
}
menu_pass_dense = bool(
    gates_dense["top32_oracle"]
    and gates_dense["top64_oracle"]
    and gates_dense["top64_node_coverage"]
)
selector_pass_dense = bool(
    gates_dense["viterbi_path"]
    and gates_dense["far_gain"]
    and gates_dense["positive_wells"]
)
strict_pass_dense = bool(menu_pass_dense and selector_pass_dense)

verdict_dense = {
    "experiment": EXPERIMENT_DENSE,
    "stage": "nested_dense_posterior_menu_and_path_ceiling_audit",
    "sample_unit": "complete physical well; deterministic 32-ft nodes",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "protocol": {
        "core_matcher_train_folds": [3, 4, 5],
        "path_selection_fold": 2,
        "final_matcher_train_folds": [2, 3, 4, 5],
        "untouched_test_fold": 1,
        "selected_emission": selected_mode_dense,
        "selected_transition_lambda": selected_lambda_dense,
    },
    "transition_prior": {
        "core": core_prior_dense.__dict__,
        "outer": outer_prior_dense.__dict__,
        "space": "S = TVT + Z",
    },
    "fold2_selection": selected_dense,
    "fold1": {
        "anchor": anchor_metrics_dense,
        "viterbi": viterbi_metrics_dense,
        "positive_well_fraction": positive_fraction_dense,
        "far_gain": far_gain_dense,
        "topk_oracles": oracle_rows_dense,
        "node_coverage": coverage_report_dense.to_dict(orient="records"),
    },
    "gate_thresholds": {
        "top32_oracle_rmse_le": K32_ORACLE_GATE_DENSE,
        "top64_oracle_rmse_le": K64_ORACLE_GATE_DENSE,
        "top64_node_within5_ge": TOP64_NODE_COVERAGE_GATE_DENSE,
        "viterbi_rmse_le": VITERBI_RMSE_GATE_DENSE,
        "far_gain_ge": FAR_GAIN_GATE_DENSE,
        "positive_well_fraction_ge": POSITIVE_WELL_GATE_DENSE,
    },
    "gates": gates_dense,
    "menu_pass": menu_pass_dense,
    "selector_pass": selector_pass_dense,
    "strict_pass": strict_pass_dense,
    "elapsed_minutes": (time.time() - started_dense) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> train one frozen-emission complete-well CRF residual"
        if strict_pass_dense
        else (
            "MENU PASS / SELECTOR FAIL -> only a frozen-emission CRF is authorised"
            if menu_pass_dense
            else "MENU FAIL -> close this matcher posterior before any path training"
        )
    ),
}
_write_json(OUT_DIR_DENSE / "verdict.json", verdict_dense)

print("\nFOLD2 VITERBI GRID")
print(grid_dense.to_string(index=False), flush=True)
print("\nFOLD1 NODE COVERAGE")
print(coverage_report_dense.to_string(index=False), flush=True)
print("\nFOLD1 TOPK TRAJECTORY ORACLES")
print(oracle_report_dense.to_string(index=False), flush=True)
print("\nDENSE POSTERIOR FINAL VERDICT")
print(json.dumps(verdict_dense, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR_DENSE, flush=True)
