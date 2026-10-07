"""Zero-training audit after the completed multi-horizon Drift-V3 gateway.

Run in the same live kernel after
``multihorizon_drift_correspondence_fold2_v3_resume_cell.py``.  The audit asks
four pre-registered questions before another model is trained:

1. How much do absolute observations, drift edges and their WLS combination
   contribute independently?
2. Does each learned Delta-S horizon beat the legal visible-prefix trend prior?
3. Can cross-correlation of posterior profiles estimate relative displacement
   more accurately than the same prior?
4. Do typewell-unique GR markers provide enough sparse absolute closures to
   reduce the far-toe oracle error?

No model parameters, thresholds or paths are fit to Fold 2 truth.  Marker
crossing locations are allowed to use truth only in the explicitly labelled
oracle ceiling; deployable marker detection is not claimed by this audit.
"""

from pathlib import Path
import gc
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_AUDIT_V0 = {
    "OOF_ROOT",
    "device",
    "model_v3",
    "fold2_volumes_v3",
    "_tensors_v3",
    "_load_raw_record",
    "_native_z_relative_v2",
    "_robust_prefix_slope_v2",
    "_path_payload_v2",
    "_summarise_payloads_v2",
    "STATE_OFFSETS_V2",
    "HORIZONS_V3",
    "NODE_STEP_FT_V2",
    "_write_json",
    "_json_default",
}
missing_audit_v0 = sorted(REQUIRED_AUDIT_V0.difference(globals()))
if missing_audit_v0:
    raise RuntimeError(
        "Run this in the completed Drift-V3 live kernel; missing: "
        + repr(missing_audit_v0)
    )

EXPERIMENT_AUDIT_V0 = "v3_constraint_profile_marker_audit_v0"
OUT_DIR_AUDIT_V0 = Path(OOF_ROOT) / EXPERIMENT_AUDIT_V0
OUT_DIR_AUDIT_V0.mkdir(parents=True, exist_ok=True)

PROFILE_TOPK_AUDIT_V0 = 16
PROFILE_MAX_SHIFT_AUDIT_V0 = 96
MARKERS_PER_WELL_AUDIT_V0 = 8
MARKER_EXCLUSION_FT_AUDIT_V0 = 8
MARKER_SEPARATION_FT_AUDIT_V0 = 16
MARKER_CROSSING_TOLERANCE_FT_AUDIT_V0 = 4.0
MARKER_WINDOW_RADII_AUDIT_V0 = (4, 8, 16, 32)
MARKER_WINDOW_POINTS_AUDIT_V0 = 33


def _rmse_audit_v0(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


@torch.no_grad()
def _forward_audit_v0(volume):
    model_v3.eval()
    values, truth, z_relative, _ = _tensors_v3(volume)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model_v3(values, z_relative)
    return {
        "truth": truth.float().cpu().numpy(),
        "z": z_relative.float().cpu().numpy(),
        "absolute": output["absolute_tvt"].float().cpu().numpy(),
        "absolute_precision": output["absolute_precision"].float().cpu().numpy(),
        "combined": output["solved_tvt"].float().cpu().numpy(),
        "horizons": {
            int(horizon): (
                delta.float().cpu().numpy(),
                precision.float().cpu().numpy(),
            )
            for horizon, (delta, precision) in output["horizons"].items()
        },
    }


def _graph_solve_audit_v0(
    absolute_s,
    absolute_precision,
    horizons,
    columns,
    use_absolute,
    use_drift,
    learned_precision,
):
    rows = []
    observations = []
    weights = []
    if use_absolute:
        rows.append(np.eye(columns, dtype=np.float64))
        observations.append(np.asarray(absolute_s, np.float64))
        weights.append(
            np.clip(np.asarray(absolute_precision, np.float64), 0.03, 30.0)
            if learned_precision
            else np.ones(columns, np.float64)
        )

    anchor = np.zeros((1, columns), np.float64)
    anchor[0, 0] = 1.0
    rows.append(anchor)
    observations.append(np.zeros(1, np.float64))
    weights.append(np.full(1, 256.0, np.float64))

    if use_drift:
        for horizon, (delta, precision) in horizons.items():
            if columns <= horizon:
                continue
            count = columns - horizon
            edge = np.zeros((count, columns), np.float64)
            index = np.arange(count)
            edge[index, index] = -1.0
            edge[index, index + horizon] = 1.0
            rows.append(edge)
            observations.append(np.asarray(delta[:count], np.float64))
            weights.append(
                np.clip(np.asarray(precision[:count], np.float64), 0.03, 30.0)
                if learned_precision
                else np.ones(count, np.float64)
            )

    design = np.concatenate(rows, axis=0)
    observation = np.concatenate(observations)
    weight = np.concatenate(weights)
    system = design.T @ (weight[:, None] * design)
    system.flat[:: columns + 1] += 1e-4
    rhs = design.T @ (weight * observation)
    solution = np.linalg.solve(system, rhs)
    return solution - solution[0]


def _topk_profile_shift_audit_v0(first, second):
    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    top_first = np.argpartition(first, -PROFILE_TOPK_AUDIT_V0)[
        -PROFILE_TOPK_AUDIT_V0:
    ]
    top_second = np.argpartition(second, -PROFILE_TOPK_AUDIT_V0)[
        -PROFILE_TOPK_AUDIT_V0:
    ]
    first_score = first[top_first]
    second_score = second[top_second]
    first_probability = np.exp(first_score - np.max(first_score))
    second_probability = np.exp(second_score - np.max(second_score))
    first_probability /= max(first_probability.sum(), 1e-12)
    second_probability /= max(second_probability.sum(), 1e-12)

    difference = top_second[:, None] - top_first[None, :]
    pair_weight = second_probability[:, None] * first_probability[None, :]
    valid = np.abs(difference) <= PROFILE_MAX_SHIFT_AUDIT_V0
    bins = np.arange(
        -PROFILE_MAX_SHIFT_AUDIT_V0,
        PROFILE_MAX_SHIFT_AUDIT_V0 + 1,
        dtype=np.int64,
    )
    histogram = np.zeros(len(bins), np.float64)
    np.add.at(
        histogram,
        difference[valid].astype(np.int64) + PROFILE_MAX_SHIFT_AUDIT_V0,
        pair_weight[valid],
    )
    order = np.argsort(histogram)[::-1]
    best = int(bins[order[0]])
    margin = float(
        histogram[order[0]] - histogram[order[1]] if len(order) > 1 else histogram[order[0]]
    )
    return best, margin


def _marker_features_audit_v0(record):
    values = np.asarray(record.typewell[0], np.float64)
    valid = np.asarray(record.typewell[1], np.float64) > 0.5
    state_offsets = np.asarray(STATE_OFFSETS_V2, np.float64)
    centre = len(values) // 2
    centres = centre + state_offsets.astype(np.int64)
    feature_parts = []
    coverage_parts = []
    unit = np.linspace(-1.0, 1.0, MARKER_WINDOW_POINTS_AUDIT_V0)
    axis = np.arange(len(values), dtype=np.float64)
    for radius in MARKER_WINDOW_RADII_AUDIT_V0:
        query = centres[:, None] + float(radius) * unit[None]
        sampled = np.interp(query.reshape(-1), axis, values).reshape(query.shape)
        sampled_valid = (
            np.interp(query.reshape(-1), axis, valid.astype(float)).reshape(query.shape)
            >= 0.5
        )
        sampled = np.where(sampled_valid, sampled, 0.0)
        count = sampled_valid.sum(axis=1, keepdims=True)
        mean = np.divide(
            (sampled * sampled_valid).sum(axis=1, keepdims=True),
            np.maximum(count, 1),
        )
        variance = np.divide(
            ((sampled - mean) ** 2 * sampled_valid).sum(axis=1, keepdims=True),
            np.maximum(count, 1),
        )
        normalised = (sampled - mean) / np.sqrt(np.maximum(variance, 1e-4))
        normalised *= sampled_valid
        normalised /= np.sqrt(
            np.maximum((normalised**2).sum(axis=1, keepdims=True), 1e-8)
        )
        feature_parts.append(normalised)
        coverage_parts.append(sampled_valid.mean(axis=1))
    features = np.concatenate(feature_parts, axis=1)
    features /= np.sqrt(np.maximum((features**2).sum(axis=1, keepdims=True), 1e-8))
    coverage = np.min(np.stack(coverage_parts, axis=1), axis=1)
    state_valid = coverage >= 0.75
    similarity = features @ features.T
    return features, similarity, state_valid, coverage


def _select_markers_audit_v0(record):
    _, similarity, valid, coverage = _marker_features_audit_v0(record)
    states = len(STATE_OFFSETS_V2)
    uniqueness = np.full(states, -np.inf, np.float64)
    alias24 = np.full(states, np.nan, np.float64)
    for state in range(states):
        if not valid[state]:
            continue
        allowed = valid.copy()
        lo = max(0, state - MARKER_EXCLUSION_FT_AUDIT_V0)
        hi = min(states, state + MARKER_EXCLUSION_FT_AUDIT_V0 + 1)
        allowed[lo:hi] = False
        second = float(np.max(similarity[state, allowed])) if allowed.any() else 1.0
        aliases = []
        for shift in (-24, 24):
            candidate = state + shift
            if 0 <= candidate < states and valid[candidate]:
                aliases.append(similarity[state, candidate])
        alias24[state] = max(aliases) if aliases else np.nan
        competitor = max(second, alias24[state] if np.isfinite(alias24[state]) else -1.0)
        uniqueness[state] = 1.0 - competitor

    selected = []
    for state in np.argsort(uniqueness)[::-1]:
        if not np.isfinite(uniqueness[state]):
            continue
        if all(abs(int(state) - previous) >= MARKER_SEPARATION_FT_AUDIT_V0 for previous in selected):
            selected.append(int(state))
        if len(selected) >= MARKERS_PER_WELL_AUDIT_V0:
            break
    return np.asarray(selected, np.int64), uniqueness, alias24, coverage


def _marker_crossings_audit_v0(truth_offset, marker_states):
    truth_index = np.asarray(truth_offset, np.float64) + 128.0
    crossings = []
    for marker in marker_states:
        distance = np.abs(truth_index - marker)
        inside = distance <= MARKER_CROSSING_TOLERANCE_FT_AUDIT_V0
        starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
        ends = np.flatnonzero(inside & ~np.r_[inside[1:], False])
        for start, end in zip(starts, ends):
            segment = np.arange(start, end + 1)
            best = int(segment[np.argmin(distance[segment])])
            crossings.append((best, int(marker)))
    crossings.sort()
    return crossings


started_audit_v0 = time.time()
forward_audit_v0 = {}
for index, well in enumerate(sorted(fold2_volumes_v3), 1):
    forward_audit_v0[well] = _forward_audit_v0(fold2_volumes_v3[well])
    if index % 50 == 0 or index == len(fold2_volumes_v3):
        print(f"constraint forward {index}/{len(fold2_volumes_v3)}", flush=True)


# ---------------------------------------------------------------------------
# Solver component ablation.
# ---------------------------------------------------------------------------


variant_payloads_audit_v0 = {
    "absolute_only": [],
    "drift_only": [],
    "combined_fixed_precision": [],
    "combined_learned_precision": [],
}
node_predictions_audit_v0 = {}
for well in sorted(fold2_volumes_v3):
    volume = fold2_volumes_v3[well]
    output = forward_audit_v0[well]
    z = output["z"].astype(np.float64)
    absolute = output["absolute"].astype(np.float64)
    absolute = absolute - absolute[0]
    drift_s = _graph_solve_audit_v0(
        absolute + z,
        output["absolute_precision"],
        output["horizons"],
        len(z),
        use_absolute=False,
        use_drift=True,
        learned_precision=True,
    )
    drift = drift_s - z
    drift -= drift[0]
    fixed_s = _graph_solve_audit_v0(
        absolute + z,
        output["absolute_precision"],
        output["horizons"],
        len(z),
        use_absolute=True,
        use_drift=True,
        learned_precision=False,
    )
    fixed = fixed_s - z
    fixed -= fixed[0]
    learned = output["combined"].astype(np.float64)
    learned -= learned[0]
    predictions = {
        "absolute_only": absolute,
        "drift_only": drift,
        "combined_fixed_precision": fixed,
        "combined_learned_precision": learned,
    }
    node_predictions_audit_v0[well] = predictions
    for variant, prediction in predictions.items():
        variant_payloads_audit_v0[variant].append(
            _path_payload_v2(volume, prediction)
        )

solver_rows_audit_v0 = []
solver_bywell_audit_v0 = {}
solver_merged_audit_v0 = {}
for variant, payloads in variant_payloads_audit_v0.items():
    metrics, bywell, merged = _summarise_payloads_v2(payloads)
    solver_rows_audit_v0.append({"variant": variant, **metrics})
    solver_bywell_audit_v0[variant] = bywell
    solver_merged_audit_v0[variant] = merged
    bywell.to_csv(OUT_DIR_AUDIT_V0 / f"solver_{variant}_bywell.csv", index=False)
solver_report_audit_v0 = pd.DataFrame(solver_rows_audit_v0).sort_values("rmse")
solver_report_audit_v0.to_csv(OUT_DIR_AUDIT_V0 / "solver_ablation.csv", index=False)


# ---------------------------------------------------------------------------
# Learned Delta-S and posterior-profile shift versus prefix trend.
# ---------------------------------------------------------------------------


horizon_rows_audit_v0 = []
precision_rows_audit_v0 = []
for well in sorted(fold2_volumes_v3):
    volume = fold2_volumes_v3[well]
    output = forward_audit_v0[well]
    truth_s = output["truth"].astype(np.float64) + output["z"].astype(np.float64)
    record = _load_raw_record(well)
    full_z = _native_z_relative_v2(well)
    slope = _robust_prefix_slope_v2(record, full_z)
    profile = volume.input[0].astype(np.float32).T  # [nodes, states]
    for horizon in HORIZONS_V3:
        if len(truth_s) <= horizon:
            continue
        count = len(truth_s) - horizon
        target = truth_s[horizon:] - truth_s[:-horizon]
        learned_delta, precision = output["horizons"][horizon]
        learned_delta = learned_delta[:count].astype(np.float64)
        precision = precision[:count].astype(np.float64)
        delta_md = (
            volume.node_md[horizon:].astype(np.float64)
            - volume.node_md[:-horizon].astype(np.float64)
        )
        prior = slope * delta_md
        z_delta = (
            output["z"][horizon:].astype(np.float64)
            - output["z"][:-horizon].astype(np.float64)
        )
        for edge in range(count):
            profile_delta_tvt, profile_margin = _topk_profile_shift_audit_v0(
                profile[edge], profile[edge + horizon]
            )
            horizon_rows_audit_v0.append(
                {
                    "well": well,
                    "horizon": int(horizon),
                    "horizon_ft": float(delta_md[edge]),
                    "distance_ft": float(volume.node_md[edge] - volume.node_md[0]),
                    "truth_delta_s": float(target[edge]),
                    "learned_delta_s": float(learned_delta[edge]),
                    "prior_delta_s": float(prior[edge]),
                    "profile_delta_s": float(profile_delta_tvt + z_delta[edge]),
                    "profile_margin": profile_margin,
                    "precision": float(precision[edge]),
                }
            )
            precision_rows_audit_v0.append(
                {
                    "well": well,
                    "horizon": int(horizon),
                    "precision": float(precision[edge]),
                    "squared_error": float((learned_delta[edge] - target[edge]) ** 2),
                }
            )

horizon_edges_audit_v0 = pd.DataFrame(horizon_rows_audit_v0)
horizon_edges_audit_v0.to_pickle(OUT_DIR_AUDIT_V0 / "horizon_edges.pkl")
horizon_summary_rows_audit_v0 = []
for horizon, frame in horizon_edges_audit_v0.groupby("horizon", sort=True):
    truth = frame["truth_delta_s"].to_numpy(float)
    learned_rmse = _rmse_audit_v0(truth, frame["learned_delta_s"])
    prior_rmse = _rmse_audit_v0(truth, frame["prior_delta_s"])
    profile_rmse = _rmse_audit_v0(truth, frame["profile_delta_s"])
    squared_error = (frame["learned_delta_s"].to_numpy(float) - truth) ** 2
    precision = frame["precision"].to_numpy(float)
    precision_spearman = pd.Series(precision).corr(
        pd.Series(-squared_error), method="spearman"
    )
    horizon_summary_rows_audit_v0.append(
        {
            "horizon": int(horizon),
            "median_horizon_ft": float(frame["horizon_ft"].median()),
            "edges": len(frame),
            "learned_rmse": learned_rmse,
            "prior_rmse": prior_rmse,
            "learned_gain_fraction_vs_prior": float(
                (prior_rmse - learned_rmse) / max(prior_rmse, 1e-8)
            ),
            "profile_rmse": profile_rmse,
            "profile_gain_fraction_vs_prior": float(
                (prior_rmse - profile_rmse) / max(prior_rmse, 1e-8)
            ),
            "precision_error_spearman": float(precision_spearman),
            "precision_mean": float(np.mean(precision)),
        }
    )
horizon_summary_audit_v0 = pd.DataFrame(horizon_summary_rows_audit_v0)
horizon_summary_audit_v0.to_csv(
    OUT_DIR_AUDIT_V0 / "horizon_summary.csv", index=False
)

precision_frame_audit_v0 = pd.DataFrame(precision_rows_audit_v0)
precision_bins_audit_v0 = []
for horizon, frame in precision_frame_audit_v0.groupby("horizon", sort=True):
    ranked = frame.copy()
    ranked["precision_bin"] = pd.qcut(
        ranked["precision"].rank(method="first"), 5, labels=False
    )
    for precision_bin, group in ranked.groupby("precision_bin"):
        precision_bins_audit_v0.append(
            {
                "horizon": int(horizon),
                "precision_bin": int(precision_bin),
                "edges": len(group),
                "mean_precision": float(group["precision"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
            }
        )
precision_calibration_audit_v0 = pd.DataFrame(precision_bins_audit_v0)
precision_calibration_audit_v0.to_csv(
    OUT_DIR_AUDIT_V0 / "precision_reliability_bins.csv", index=False
)


# ---------------------------------------------------------------------------
# Typewell-unique marker audit and explicit oracle ceiling.
# ---------------------------------------------------------------------------


marker_rows_audit_v0 = []
marker_oracle_payloads_audit_v0 = []
marker_well_rows_audit_v0 = []
for index, well in enumerate(sorted(fold2_volumes_v3), 1):
    volume = fold2_volumes_v3[well]
    record = _load_raw_record(well)
    markers, uniqueness, alias24, marker_coverage = _select_markers_audit_v0(record)
    crossings = _marker_crossings_audit_v0(volume.truth_offset, markers)
    profile = volume.input[0].astype(np.float32).T
    crossing_hits_top1 = []
    crossing_hits_top3 = []
    crossing_margins = []
    closure_points = [(0, 0.0)]
    base_prediction = node_predictions_audit_v0[well][
        "combined_learned_precision"
    ].astype(np.float64)

    for node, marker in crossings:
        marker_scores = profile[node, markers]
        order = np.argsort(marker_scores)[::-1]
        top_states = markers[order]
        hit_top1 = int(abs(int(top_states[0]) - marker) <= 4)
        hit_top3 = int(np.any(np.abs(top_states[:3] - marker) <= 4))
        margin = float(
            marker_scores[order[0]] - marker_scores[order[1]]
            if len(order) > 1
            else marker_scores[order[0]]
        )
        crossing_hits_top1.append(hit_top1)
        crossing_hits_top3.append(hit_top3)
        crossing_margins.append(margin)
        marker_offset = float(STATE_OFFSETS_V2[marker])
        closure_points.append((node, marker_offset - base_prediction[node]))
        marker_rows_audit_v0.append(
            {
                "well": well,
                "node": int(node),
                "distance_ft": float(volume.node_md[node] - volume.node_md[0]),
                "marker_state": int(marker),
                "marker_offset": marker_offset,
                "truth_offset": float(volume.truth_offset[node]),
                "top1_hit": hit_top1,
                "top3_hit": hit_top3,
                "score_margin": margin,
                "uniqueness": float(uniqueness[marker]),
                "alias24_similarity": float(alias24[marker]),
            }
        )

    closure_by_node = {}
    for node, correction in closure_points:
        closure_by_node.setdefault(int(node), []).append(float(correction))
    closure_nodes = np.asarray(sorted(closure_by_node), np.int64)
    closure_correction = np.asarray(
        [np.mean(closure_by_node[node]) for node in closure_nodes], np.float64
    )
    all_nodes = np.arange(len(base_prediction), dtype=np.float64)
    if len(closure_nodes) > 1:
        correction = np.interp(
            all_nodes,
            closure_nodes.astype(float),
            closure_correction,
            left=0.0,
            right=float(closure_correction[-1]),
        )
    else:
        correction = np.zeros_like(base_prediction)
    marker_oracle_prediction = base_prediction + correction
    marker_oracle_prediction[0] = 0.0
    marker_oracle_payloads_audit_v0.append(
        _path_payload_v2(volume, marker_oracle_prediction)
    )
    marker_well_rows_audit_v0.append(
        {
            "well": well,
            "markers": len(markers),
            "crossings": len(crossings),
            "has_crossing": bool(crossings),
            "top1_accuracy": float(np.mean(crossing_hits_top1)) if crossings else np.nan,
            "top3_accuracy": float(np.mean(crossing_hits_top3)) if crossings else np.nan,
            "median_margin": float(np.median(crossing_margins)) if crossings else np.nan,
            "median_uniqueness": float(np.median(uniqueness[markers])) if len(markers) else np.nan,
            "median_alias24_similarity": float(np.nanmedian(alias24[markers])) if len(markers) else np.nan,
        }
    )
    if index % 50 == 0 or index == len(fold2_volumes_v3):
        print(f"marker audit {index}/{len(fold2_volumes_v3)}", flush=True)

marker_crossings_audit_v0 = pd.DataFrame(marker_rows_audit_v0)
marker_bywell_audit_v0 = pd.DataFrame(marker_well_rows_audit_v0)
marker_crossings_audit_v0.to_csv(
    OUT_DIR_AUDIT_V0 / "marker_crossings.csv", index=False
)
marker_bywell_audit_v0.to_csv(OUT_DIR_AUDIT_V0 / "marker_bywell.csv", index=False)
marker_oracle_metrics_audit_v0, marker_oracle_bywell_audit_v0, marker_oracle_merged_audit_v0 = (
    _summarise_payloads_v2(marker_oracle_payloads_audit_v0)
)
marker_oracle_bywell_audit_v0.to_csv(
    OUT_DIR_AUDIT_V0 / "marker_oracle_bywell.csv", index=False
)

base_metrics_audit_v0 = next(
    row for row in solver_rows_audit_v0 if row["variant"] == "combined_learned_precision"
)
marker_wells_with_crossing_audit_v0 = int(marker_bywell_audit_v0["has_crossing"].sum())
marker_coverage_fraction_audit_v0 = float(marker_bywell_audit_v0["has_crossing"].mean())
marker_top1_audit_v0 = (
    float(marker_crossings_audit_v0["top1_hit"].mean())
    if len(marker_crossings_audit_v0)
    else np.nan
)
marker_top3_audit_v0 = (
    float(marker_crossings_audit_v0["top3_hit"].mean())
    if len(marker_crossings_audit_v0)
    else np.nan
)

learned_horizon_pass_audit_v0 = bool(
    np.all(horizon_summary_audit_v0["learned_gain_fraction_vs_prior"] >= 0.30)
)
profile_horizon_pass_audit_v0 = bool(
    np.any(horizon_summary_audit_v0["profile_gain_fraction_vs_prior"] >= 0.30)
)
precision_pass_audit_v0 = bool(
    np.all(horizon_summary_audit_v0["precision_error_spearman"] >= 0.10)
)
marker_ceiling_gain_audit_v0 = float(
    base_metrics_audit_v0["rmse"] - marker_oracle_metrics_audit_v0["rmse"]
)
marker_far_gain_audit_v0 = float(
    base_metrics_audit_v0["far_gt2000_rmse"]
    - marker_oracle_metrics_audit_v0["far_gt2000_rmse"]
)
marker_pass_audit_v0 = bool(
    marker_coverage_fraction_audit_v0 >= 0.40
    and marker_top1_audit_v0 >= 0.70
    and marker_far_gain_audit_v0 >= 1.0
)

verdict_audit_v0 = {
    "experiment": EXPERIMENT_AUDIT_V0,
    "stage": "zero_training_constraint_profile_marker_audit",
    "validation_fold": 2,
    "wells": len(fold2_volumes_v3),
    "solver_ablation": solver_report_audit_v0.to_dict(orient="records"),
    "horizon_summary": horizon_summary_audit_v0.to_dict(orient="records"),
    "precision_gate": {
        "minimum_precision_error_spearman_ge": 0.10,
        "pass": precision_pass_audit_v0,
    },
    "learned_drift_gate": {
        "every_horizon_gain_fraction_vs_prior_ge": 0.30,
        "pass": learned_horizon_pass_audit_v0,
    },
    "profile_shift_gate": {
        "any_horizon_gain_fraction_vs_prior_ge": 0.30,
        "pass": profile_horizon_pass_audit_v0,
    },
    "marker": {
        "markers_per_well": MARKERS_PER_WELL_AUDIT_V0,
        "crossing_wells": marker_wells_with_crossing_audit_v0,
        "well_coverage": marker_coverage_fraction_audit_v0,
        "crossings": int(len(marker_crossings_audit_v0)),
        "top1_accuracy": marker_top1_audit_v0,
        "top3_accuracy": marker_top3_audit_v0,
        "base_rmse": float(base_metrics_audit_v0["rmse"]),
        "oracle_rmse": marker_oracle_metrics_audit_v0["rmse"],
        "oracle_gain": marker_ceiling_gain_audit_v0,
        "base_far_rmse": float(base_metrics_audit_v0["far_gt2000_rmse"]),
        "oracle_far_rmse": marker_oracle_metrics_audit_v0["far_gt2000_rmse"],
        "oracle_far_gain": marker_far_gain_audit_v0,
        "gates": {
            "well_coverage_ge": 0.40,
            "top1_accuracy_ge": 0.70,
            "far_oracle_gain_ge": 1.0,
        },
        "pass": marker_pass_audit_v0,
    },
    "elapsed_minutes": (time.time() - started_audit_v0) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "BUILD ROBUST GRAPH V4"
        if marker_pass_audit_v0 or profile_horizon_pass_audit_v0
        else (
            "UPGRADE ABSOLUTE EMISSION; freeze decoder and graph"
            if not learned_horizon_pass_audit_v0
            else "DELTA-S IS VALID BUT NO DEPLOYABLE ABSOLUTE CLOSURE; audit learned markers"
        )
    ),
}
_write_json(OUT_DIR_AUDIT_V0 / "verdict.json", verdict_audit_v0)

print("\nSOLVER ABLATION")
print(solver_report_audit_v0.to_string(index=False), flush=True)
print("\nHORIZON SUMMARY")
print(horizon_summary_audit_v0.to_string(index=False), flush=True)
print("\nPRECISION RELIABILITY BINS")
print(precision_calibration_audit_v0.to_string(index=False), flush=True)
print("\nMARKER SUMMARY")
print(json.dumps(verdict_audit_v0["marker"], indent=2), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_audit_v0, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_AUDIT_V0, flush=True)

gc.collect()
torch.cuda.empty_cache()
