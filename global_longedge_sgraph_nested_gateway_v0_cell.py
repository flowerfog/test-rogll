"""Nested fixed-weight SGraph gateway for the global long-edge model.

Run in the same live kernel after ``prefixridge_longrange_gateway_v0_cell.py``.
PrefixRidge is intentionally excluded: the previous gateway disproved its
incremental value, while validating the fold-clean global Delta-S edge model.

Fold 2 selects only the graph edge set and a fixed scale-weight rule.  The
frozen graph is then applied once to Fold 1.  No validation-well routing,
learned precision, neighbour, CP12, Gold, or per-well optimisation is used.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd


REQUIRED_SG = {
    "edges",
    "meta",
    "tune",
    "test",
    "core_prediction_tune",
    "test_prediction",
    "_load_record",
    "_node_md",
    "OOF_ROOT",
    "_rmse",
    "_json_default",
    "NODE_STEP_FT",
}
missing_sg = sorted(REQUIRED_SG.difference(globals()))
if missing_sg:
    raise RuntimeError(
        "Run this in the live PrefixRidge gateway kernel; missing globals: "
        + repr(missing_sg)
    )


EXPERIMENT_SG = "global_longedge_sgraph_nested_gateway_v0"
OUT_DIR_SG = OOF_ROOT / EXPERIMENT_SG
OUT_DIR_SG.mkdir(parents=True, exist_ok=True)

GRAPH_VARIANTS_SG = {
    "chain": {"max_skip": 1, "star": False},
    "short256": {"max_skip": 8, "star": False},
    "multiscale2048": {"max_skip": 64, "star": False},
    "chain_star": {"max_skip": 1, "star": True},
    "multiscale_star": {"max_skip": 64, "star": True},
}
WEIGHT_SCHEMES_SG = ("equal_scale", "inverse_variance")
ORACLE_RMSE_GATE_SG = 0.20
PATH_RMSE_GATE_SG = 9.0
FAR_GAIN_VS_CHAIN_GATE_SG = 1.0
NEAR_DEGRADATION_GATE_SG = 0.10
POSITIVE_WELL_FRACTION_GATE_SG = 0.60


def _json_sg(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _solve_group_sg(frame):
    length = (frame["end_md"] - frame["start_md"]).to_numpy(np.float64)
    step = frame["edge_step"].to_numpy(int)
    star = frame["family"].eq("anchor_star").to_numpy()
    star_power = np.clip(
        np.rint(np.log2(np.maximum(length / NODE_STEP_FT, 1.0))).astype(int),
        0,
        10,
    )
    group = np.asarray([f"skip_{value}" for value in step], dtype="U24")
    group[star] = np.asarray([f"star_{value}" for value in star_power[star]])
    return group


def _reliability_sg(frame, prediction):
    local = frame.reset_index(drop=True).copy()
    local["solve_group"] = _solve_group_sg(local)
    local["squared_error"] = (
        local["target_delta_s"].to_numpy(np.float64)
        - np.asarray(prediction, np.float64)
    ) ** 2
    stats = (
        local.groupby("solve_group", as_index=False)
        .agg(edges=("well", "size"), mse=("squared_error", "mean"))
    )
    stats["rmse"] = np.sqrt(stats["mse"])
    stats["inverse_variance"] = 1.0 / np.maximum(stats["mse"], 0.25)
    reliability = dict(zip(stats["solve_group"], stats["inverse_variance"]))
    return reliability, stats


def _edge_mask_sg(frame, variant):
    config = GRAPH_VARIANTS_SG[variant]
    suffix = frame["region"].eq("suffix").to_numpy()
    star = frame["family"].eq("anchor_star").to_numpy()
    skip = frame["edge_step"].to_numpy(int)
    allowed_skip = (~star) & (skip <= int(config["max_skip"]))
    allowed_star = star & bool(config["star"])
    return suffix & (allowed_skip | allowed_star)


def _graph_weights_sg(frame, mask, scheme, reliability):
    subset = frame.loc[mask]
    groups = _solve_group_sg(subset)
    counts = pd.Series(groups).value_counts().to_dict()
    weight = np.empty(len(subset), np.float64)
    for index, group in enumerate(groups):
        base = 1.0 / max(counts[group], 1)
        if scheme == "inverse_variance":
            base *= float(reliability.get(group, 1.0))
        elif scheme != "equal_scale":
            raise ValueError(scheme)
        weight[index] = base
    weight /= max(np.mean(weight), 1e-12)
    return weight


def _solve_one_well_sg(frame, prediction, variant, scheme, reliability):
    mask = _edge_mask_sg(frame, variant)
    selected = frame.loc[mask]
    if not len(selected):
        raise RuntimeError(f"{frame['well'].iloc[0]}: no graph edges for {variant}")
    values = np.asarray(prediction, np.float64)[mask]
    anchor_node = int(selected["node_i"].min())
    maximum_node = int(selected["node_j"].max())
    nodes = maximum_node - anchor_node + 1
    if nodes < 3:
        raise RuntimeError(f"{frame['well'].iloc[0]}: graph has only {nodes} nodes")
    weights = _graph_weights_sg(frame, mask, scheme, reliability)
    normal = np.zeros((nodes - 1, nodes - 1), np.float64)
    rhs = np.zeros(nodes - 1, np.float64)
    edge_i = selected["node_i"].to_numpy(int) - anchor_node
    edge_j = selected["node_j"].to_numpy(int) - anchor_node
    for i, j, target, weight in zip(edge_i, edge_j, values, weights):
        if j <= 0 or i < 0:
            continue
        j_free = j - 1
        normal[j_free, j_free] += weight
        rhs[j_free] += weight * target
        if i > 0:
            i_free = i - 1
            normal[i_free, i_free] += weight
            normal[i_free, j_free] -= weight
            normal[j_free, i_free] -= weight
            rhs[i_free] -= weight * target
    diagonal = np.diag(normal)
    if np.any(diagonal <= 0):
        raise RuntimeError(
            f"{frame['well'].iloc[0]}: disconnected {variant} graph, "
            f"zero diagonals={int(np.sum(diagonal <= 0))}"
        )
    jitter = max(float(np.median(diagonal)) * 1e-8, 1e-10)
    condition = float(np.linalg.cond(normal + jitter * np.eye(nodes - 1)))
    solved = np.linalg.solve(normal + jitter * np.eye(nodes - 1), rhs)
    return np.r_[0.0, solved], anchor_node, condition, len(selected)


def _evaluate_graph_sg(frame, prediction, wells, variant, scheme, reliability):
    truth_parts = []
    prediction_parts = []
    distance_parts = []
    well_parts = []
    diagnostics = []
    bywell = []
    for well in wells:
        group = frame[frame["well"].eq(well)]
        index = group.index.to_numpy()
        path, anchor_node, condition, used_edges = _solve_one_well_sg(
            group,
            np.asarray(prediction, np.float64)[index],
            variant,
            scheme,
            reliability,
        )
        record = _load_record(well)
        node_md, expected_anchor = _node_md(record)
        if anchor_node != expected_anchor:
            raise RuntimeError(
                f"{well}: graph anchor {anchor_node} != node-grid anchor {expected_anchor}"
            )
        suffix_node_md = node_md[anchor_node : anchor_node + len(path)]
        eval_rows = np.arange(record.anchor_index + 1, len(record.md))
        eval_md = record.md[eval_rows]
        predicted_s = np.interp(eval_md, suffix_node_md, path)
        z_relative = record.condition[4, eval_rows] * 512.0
        predicted_tvt = record.anchor_tvt + predicted_s - z_relative
        truth_tvt = record.truth_tvt[eval_rows]
        distance = eval_md - record.anchor_md
        truth_parts.append(truth_tvt)
        prediction_parts.append(predicted_tvt)
        distance_parts.append(distance)
        well_parts.append(np.full(len(eval_rows), well, dtype="U16"))
        well_rmse = _rmse(truth_tvt, predicted_tvt)
        far = distance > 2000
        near = distance <= 500
        bywell.append(
            {
                "well": well,
                "rows": len(eval_rows),
                "rmse": well_rmse,
                "near_rmse": _rmse(truth_tvt[near], predicted_tvt[near]) if near.any() else np.nan,
                "far_rmse": _rmse(truth_tvt[far], predicted_tvt[far]) if far.any() else np.nan,
            }
        )
        diagnostics.append(
            {
                "well": well,
                "variant": variant,
                "scheme": scheme,
                "nodes": len(path),
                "edges": used_edges,
                "condition_number": condition,
            }
        )
    truth = np.concatenate(truth_parts)
    predicted = np.concatenate(prediction_parts)
    distance = np.concatenate(distance_parts)
    well_labels = np.concatenate(well_parts)
    bywell = pd.DataFrame(bywell)

    def subset_metrics(mask):
        return {
            "rows": int(mask.sum()),
            "rmse": _rmse(truth[mask], predicted[mask]),
            "mae": float(np.mean(np.abs(truth[mask] - predicted[mask]))),
        }

    metrics = {
        "variant": variant,
        "weight_scheme": scheme,
        "rows": len(truth),
        "wells": len(wells),
        "rmse": _rmse(truth, predicted),
        "mae": float(np.mean(np.abs(truth - predicted))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "near_0_500_rmse": subset_metrics(distance <= 500)["rmse"],
        "mid_500_1000_rmse": subset_metrics((distance > 500) & (distance <= 1000))["rmse"],
        "mid_1000_2000_rmse": subset_metrics((distance > 1000) & (distance <= 2000))["rmse"],
        "far_gt2000_rmse": subset_metrics(distance > 2000)["rmse"],
    }
    payload = {
        "truth": truth.astype(np.float32),
        "prediction": predicted.astype(np.float32),
        "distance": distance.astype(np.float32),
        "well": well_labels,
    }
    return metrics, bywell, pd.DataFrame(diagnostics), payload


def _anchor_baseline_sg(wells):
    truth_parts = []
    prediction_parts = []
    distance_parts = []
    for well in wells:
        record = _load_record(well)
        rows = np.arange(record.anchor_index + 1, len(record.md))
        truth_parts.append(record.truth_tvt[rows])
        prediction_parts.append(np.full(len(rows), record.anchor_tvt, np.float64))
        distance_parts.append(record.md[rows] - record.anchor_md)
    truth = np.concatenate(truth_parts)
    prediction = np.concatenate(prediction_parts)
    distance = np.concatenate(distance_parts)
    return {
        "rmse": _rmse(truth, prediction),
        "near_0_500_rmse": _rmse(truth[distance <= 500], prediction[distance <= 500]),
        "far_gt2000_rmse": _rmse(truth[distance > 2000], prediction[distance > 2000]),
    }


def _positive_far_fraction_sg(base_bywell, candidate_bywell):
    merged = base_bywell[["well", "far_rmse"]].merge(
        candidate_bywell[["well", "far_rmse"]], on="well", suffixes=("_base", "_candidate")
    )
    valid = np.isfinite(merged["far_rmse_base"]) & np.isfinite(merged["far_rmse_candidate"])
    return float(
        np.mean(
            merged.loc[valid, "far_rmse_candidate"].to_numpy()
            < merged.loc[valid, "far_rmse_base"].to_numpy()
        )
    )


# ---------------------------------------------------------------------------
# Execute nested graph selection
# ---------------------------------------------------------------------------

started_sg = time.time()
tune_sg = tune.reset_index(drop=True).copy()
test_sg = test.reset_index(drop=True).copy()
tune_prediction_sg = np.asarray(core_prediction_tune, np.float64)
test_prediction_sg = np.asarray(test_prediction, np.float64)
tune_wells_sg = sorted(tune_sg["well"].unique().tolist())
test_wells_sg = sorted(test_sg["well"].unique().tolist())

reliability_sg, reliability_report_sg = _reliability_sg(
    tune_sg, tune_prediction_sg
)
reliability_report_sg.to_csv(OUT_DIR_SG / "fold2_scale_reliability.csv", index=False)
print("FOLD2 SCALE RELIABILITY")
print(reliability_report_sg.to_string(index=False), flush=True)

grid_rows_sg = []
for variant_sg in GRAPH_VARIANTS_SG:
    for scheme_sg in WEIGHT_SCHEMES_SG:
        metrics_sg, _, diagnostics_sg, _ = _evaluate_graph_sg(
            tune_sg,
            tune_prediction_sg,
            tune_wells_sg,
            variant_sg,
            scheme_sg,
            reliability_sg,
        )
        grid_rows_sg.append(metrics_sg)
        print("FOLD2 SGRAPH", metrics_sg, flush=True)
grid_sg = pd.DataFrame(grid_rows_sg).sort_values(
    ["rmse", "far_gt2000_rmse", "p95_well_rmse"]
)
grid_sg.to_csv(OUT_DIR_SG / "fold2_graph_grid.csv", index=False)
selected_sg = grid_sg.iloc[0].to_dict()
selected_variant_sg = str(selected_sg["variant"])
selected_scheme_sg = str(selected_sg["weight_scheme"])
print("FROZEN SGRAPH", selected_sg, flush=True)

# Representation/solver guard: true Delta-S edges should recover the node path;
# remaining error is only 32-ft interpolation.
oracle_metrics_sg, _, oracle_diagnostics_sg, _ = _evaluate_graph_sg(
    test_sg,
    test_sg["target_delta_s"].to_numpy(np.float64),
    test_wells_sg,
    "multiscale_star",
    "equal_scale",
    reliability_sg,
)
print("SGRAPH ORACLE", oracle_metrics_sg, flush=True)

primary_metrics_sg, primary_bywell_sg, primary_diagnostics_sg, primary_payload_sg = _evaluate_graph_sg(
    test_sg,
    test_prediction_sg,
    test_wells_sg,
    selected_variant_sg,
    selected_scheme_sg,
    reliability_sg,
)
chain_metrics_sg, chain_bywell_sg, chain_diagnostics_sg, chain_payload_sg = _evaluate_graph_sg(
    test_sg,
    test_prediction_sg,
    test_wells_sg,
    "chain",
    selected_scheme_sg,
    reliability_sg,
)
anchor_metrics_sg = _anchor_baseline_sg(test_wells_sg)

primary_bywell_sg.to_csv(OUT_DIR_SG / "fold1_primary_bywell.csv", index=False)
chain_bywell_sg.to_csv(OUT_DIR_SG / "fold1_chain_bywell.csv", index=False)
pd.concat([primary_diagnostics_sg, chain_diagnostics_sg], ignore_index=True).to_csv(
    OUT_DIR_SG / "fold1_solver_diagnostics.csv", index=False
)
np.savez_compressed(OUT_DIR_SG / "fold1_primary_oof.npz", **primary_payload_sg)
np.savez_compressed(OUT_DIR_SG / "fold1_chain_oof.npz", **chain_payload_sg)

far_gain_sg = chain_metrics_sg["far_gt2000_rmse"] - primary_metrics_sg["far_gt2000_rmse"]
near_degradation_sg = primary_metrics_sg["near_0_500_rmse"] - chain_metrics_sg["near_0_500_rmse"]
positive_far_sg = _positive_far_fraction_sg(chain_bywell_sg, primary_bywell_sg)
oracle_pass_sg = oracle_metrics_sg["rmse"] <= ORACLE_RMSE_GATE_SG
path_pass_sg = primary_metrics_sg["rmse"] <= PATH_RMSE_GATE_SG
far_pass_sg = (
    far_gain_sg >= FAR_GAIN_VS_CHAIN_GATE_SG
    and positive_far_sg >= POSITIVE_WELL_FRACTION_GATE_SG
)
near_pass_sg = near_degradation_sg <= NEAR_DEGRADATION_GATE_SG
condition_pass_sg = bool(
    np.isfinite(primary_diagnostics_sg["condition_number"]).all()
    and primary_diagnostics_sg["condition_number"].quantile(0.95) < 1e12
)
strict_pass_sg = bool(
    oracle_pass_sg and path_pass_sg and far_pass_sg and near_pass_sg and condition_pass_sg
)

comparison_sg = pd.DataFrame(
    [
        {"method": "anchor", **anchor_metrics_sg},
        {"method": "chain", **chain_metrics_sg},
        {"method": "selected", **primary_metrics_sg},
        {"method": "oracle", **oracle_metrics_sg},
    ]
)
comparison_sg.to_csv(OUT_DIR_SG / "fold1_comparison.csv", index=False)

verdict_sg = {
    "experiment": EXPERIMENT_SG,
    "stage": "nested_fixed_weight_global_DeltaS_SGraph",
    "prefixridge_used": False,
    "decoder_selection": {
        "validation_fold": 2,
        "selected_variant": selected_variant_sg,
        "selected_weight_scheme": selected_scheme_sg,
        "fold2_metrics": selected_sg,
    },
    "fold1": {
        "anchor": anchor_metrics_sg,
        "chain": chain_metrics_sg,
        "selected": primary_metrics_sg,
        "oracle": oracle_metrics_sg,
        "far_gain_vs_chain": far_gain_sg,
        "near_degradation_vs_chain": near_degradation_sg,
        "positive_far_well_fraction": positive_far_sg,
    },
    "gates": {
        "oracle_rmse_le": ORACLE_RMSE_GATE_SG,
        "path_rmse_le": PATH_RMSE_GATE_SG,
        "far_gain_vs_chain_ge": FAR_GAIN_VS_CHAIN_GATE_SG,
        "positive_far_well_fraction_ge": POSITIVE_WELL_FRACTION_GATE_SG,
        "near_degradation_vs_chain_le": NEAR_DEGRADATION_GATE_SG,
        "solver_condition_p95_lt": 1e12,
    },
    "oracle_pass": oracle_pass_sg,
    "path_pass": path_pass_sg,
    "far_pass": far_pass_sg,
    "near_pass": near_pass_sg,
    "condition_pass": condition_pass_sg,
    "strict_pass": strict_pass_sg,
    "elapsed_minutes": (time.time() - started_sg) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> train learned edge trunk with the frozen graph topology"
        if strict_pass_sg
        else (
            "HOLD -> retain global edge model but stop SGraph topology expansion"
            if path_pass_sg
            else "FAIL -> close global edge SGraph as a standalone path"
        )
    ),
}
_json_sg(OUT_DIR_SG / "verdict.json", verdict_sg)

print("\nFOLD2 GRAPH GRID")
print(grid_sg.to_string(index=False), flush=True)
print("\nFOLD1 COMPARISON")
print(comparison_sg.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_sg, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR_SG, flush=True)

