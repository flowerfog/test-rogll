"""Full-sequence STRIDE beam using the validated Figure-1 posterior.

Prerequisite
------------
Run ``stride_figure1_posterior_parity_v3_cell.py`` first.  This cell consumes
only its outer-train prior and the Fold-3-selected implementation assumptions;
it does not consume any prediction or validation label from that experiment.

The decoder keeps complete sequence hypotheses.  Each extension is an
explicit (h_len, delta_TVT) state, scored by the fitted log-normal segment
length prior, dip persistence with dynamic drilled-trend centering, and the
visible-prefix calibrated Cauchy GR likelihood.  Fold 2 is a single frozen
application and Fold 1 remains untouched.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


EXPERIMENT = "stride_fullsequence_beam_v4"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "stride_figure1_posterior_parity_v3_cell.py",
    Path.cwd() / "stride_figure1_posterior_parity_v3_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
V3_DIR = OOF_ROOT / "stride_figure1_posterior_parity_v3"
V3_VERDICT = V3_DIR / "verdict.json"
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload stride_figure1_posterior_parity_v3_cell.py")
if not V3_VERDICT.exists():
    raise FileNotFoundError(f"Run Figure-1 parity first: {V3_VERDICT}")

NODE_STEP_FT = 32.0
BEAM_SIZE = 48
MAX_EMISSION_ROWS = 48
MODES = (
    "full",
    "raw_reference",
    "gaussian",
    "no_persistence",
    "previous_dip_centre",
)
STRATA = (
    ("near_0_500", 0.0, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "FOLD_PATH is not None" in test:
                break
        body.append(node)
    module_name = "stride_figure1_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module


V = _bootstrap(SOURCE_PATH)
load_record = V._load_record
state_grids = V._state_grids


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


@njit(cache=False)
def _interp_sorted(axis, values, query):
    if query < axis[0] or query > axis[-1]:
        return 0.0, False
    if query == axis[0]:
        return values[0], True
    if query == axis[-1]:
        return values[-1], True
    low = 0
    high = len(axis) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if axis[middle] <= query:
            low = middle
        else:
            high = middle
    fraction = (query - axis[low]) / max(axis[high] - axis[low], 1e-9)
    return values[low] * (1.0 - fraction) + values[high] * fraction, True


@njit(cache=False)
def _segment_emission(
    md,
    gr,
    start_row,
    end_row,
    start_tvt,
    delta_tvt,
    tw_tvt,
    tw_gr,
    gamma,
    gaussian,
    maximum_rows,
    effective_spacing_ft,
):
    count = end_row - start_row
    if count <= 0:
        return 1e6
    samples = min(count, maximum_rows)
    total = 0.0
    valid = 0
    length = md[end_row] - md[start_row]
    for sample in range(samples):
        row = start_row + 1 + ((sample + 1) * count - 1) // samples
        fraction = (md[row] - md[start_row]) / max(length, 1e-9)
        candidate_tvt = start_tvt + fraction * delta_tvt
        expected, inside = _interp_sorted(tw_tvt, tw_gr, candidate_tvt)
        if not inside:
            total += 50.0
            continue
        residual = (gr[row] - expected) / max(gamma, 1e-6)
        square = min(residual * residual, 1e4)
        if gaussian:
            total += 0.5 * min(square, 400.0)
        else:
            total += np.log1p(square)
        valid += 1
    if valid < max(4, samples // 2):
        return 1e6
    # Approximate the disclosed product over row likelihoods while accounting
    # for strongly autocorrelated 1-ft GR samples.  Unlike a segment mean, this
    # makes total evidence approximately invariant to how a fixed-length path
    # is partitioned into segments.
    return (total * count / samples) / max(effective_spacing_ft, 1.0)


@njit(cache=False)
def _beam_decode(
    md,
    z,
    gr,
    node_rows,
    anchor_tvt,
    previous_dip,
    previous_structure_rate,
    tw_tvt,
    tw_gr,
    gamma,
    h_nodes,
    d_grid,
    length_mu,
    length_sigma,
    lam,
    sigma_persist,
    dip_low,
    dip_high,
    temperature,
    beam_size,
    maximum_rows,
    effective_spacing_ft,
    persistence_weight,
    mode_code,
):
    nodes = len(node_rows)
    costs = np.full((nodes, beam_size), 1e30)
    tvt_state = np.zeros((nodes, beam_size))
    dip_state = np.zeros((nodes, beam_size))
    structure_state = np.zeros((nodes, beam_size))
    parent_node = np.full((nodes, beam_size), -1, np.int64)
    parent_slot = np.full((nodes, beam_size), -1, np.int64)
    costs[0, 0] = 0.0
    tvt_state[0, 0] = anchor_tvt
    dip_state[0, 0] = previous_dip
    structure_state[0, 0] = previous_structure_rate

    gaussian = mode_code == 2
    no_persistence = mode_code == 3
    previous_only = mode_code == 4

    for target in range(1, nodes):
        filled = 0
        worst_slot = 0
        worst_cost = 1e30
        for length_position in range(len(h_nodes)):
            source = target - h_nodes[length_position]
            if source < 0:
                continue
            start_row = node_rows[source]
            end_row = node_rows[target]
            length = md[end_row] - md[start_row]
            # A final residual shorter than H_MIN is a normal consequence of
            # tiling a finite lateral.  Permit it only at the terminal node;
            # interior segments still respect the disclosed length range.
            if (length < 0.75 * 50.0 and target != nodes - 1) or length > 1.10 * 600.0:
                continue
            log_length = np.log(max(length, 1.0))
            length_z = (log_length - length_mu) / max(length_sigma, 1e-6)
            # Candidate lengths are discrete bins on an approximately
            # log-spaced grid.  Bin mass cancels the continuous log-normal
            # density's 1/h Jacobian, matching the Figure-1 score.
            length_cost = 0.5 * length_z * length_z
            z_rate = (z[end_row] - z[start_row]) / max(length, 1e-9)

            for source_slot in range(beam_size):
                source_cost = costs[source, source_slot]
                if source_cost >= 1e29:
                    continue
                old_tvt = tvt_state[source, source_slot]
                old_dip = dip_state[source, source_slot]
                old_structure = structure_state[source, source_slot]
                drilled_trend = old_structure - z_rate
                centre = old_dip if previous_only else lam * old_dip + (1.0 - lam) * drilled_trend

                for d_position in range(len(d_grid)):
                    delta_tvt = d_grid[d_position]
                    dip = delta_tvt / max(length, 1e-9)
                    if dip < dip_low - 0.02 or dip > dip_high + 0.02:
                        continue
                    persistence_z = (dip - centre) / max(sigma_persist, 1e-6)
                    # A Gaussian prior has support everywhere.  Earlier code
                    # used a hard |z| cutoff as a speed optimisation, which
                    # could delete every route through a difficult well.
                    # Retain the full finite D grid and let the probability
                    # cost, rather than an artificial gate, reject bad dips.
                    persistence_cost = 0.0 if no_persistence else 0.5 * persistence_z * persistence_z
                    emission = _segment_emission(
                        md,
                        gr,
                        start_row,
                        end_row,
                        old_tvt,
                        delta_tvt,
                        tw_tvt,
                        tw_gr,
                        gamma,
                        gaussian,
                        maximum_rows,
                        effective_spacing_ft,
                    )
                    new_cost = (
                        source_cost
                        + length_cost
                        + persistence_weight * persistence_cost
                        + temperature * emission
                    )
                    if filled < beam_size:
                        slot = filled
                        filled += 1
                    elif new_cost < worst_cost:
                        slot = worst_slot
                    else:
                        continue
                    costs[target, slot] = new_cost
                    tvt_state[target, slot] = old_tvt + delta_tvt
                    dip_state[target, slot] = dip
                    structure_state[target, slot] = dip + z_rate
                    parent_node[target, slot] = source
                    parent_slot[target, slot] = source_slot
                    worst_slot = 0
                    worst_cost = costs[target, 0]
                    for candidate_slot in range(1, filled):
                        if costs[target, candidate_slot] > worst_cost:
                            worst_cost = costs[target, candidate_slot]
                            worst_slot = candidate_slot

    final_slot = 0
    for slot in range(1, beam_size):
        if costs[nodes - 1, slot] < costs[nodes - 1, final_slot]:
            final_slot = slot
    if costs[nodes - 1, final_slot] >= 1e29:
        return np.full(nodes, np.nan), 1e30

    path = np.full(nodes, np.nan)
    node = nodes - 1
    slot = final_slot
    while node >= 0 and slot >= 0:
        path[node] = tvt_state[node, slot]
        previous_node = parent_node[node, slot]
        previous_slot = parent_slot[node, slot]
        if previous_node < 0:
            break
        left_tvt = tvt_state[previous_node, previous_slot]
        right_tvt = tvt_state[node, slot]
        left_md = md[node_rows[previous_node]]
        right_md = md[node_rows[node]]
        for fill_node in range(previous_node + 1, node):
            fraction = (md[node_rows[fill_node]] - left_md) / max(right_md - left_md, 1e-9)
            path[fill_node] = left_tvt + fraction * (right_tvt - left_tvt)
        node = previous_node
        slot = previous_slot
    path[0] = anchor_tvt
    return path, costs[nodes - 1, final_slot]


def _node_rows(record):
    anchor = record["anchor"]
    last = int(record["evaluation"][-1])
    targets = np.arange(record["md"][anchor], record["md"][last] + NODE_STEP_FT, NODE_STEP_FT)
    rows = [anchor]
    for target in targets[1:]:
        row = int(np.searchsorted(record["md"], target, side="left"))
        row = min(max(row, anchor + 1), last)
        if row > rows[-1]:
            rows.append(row)
    if rows[-1] != last:
        rows.append(last)
    return np.asarray(rows, np.int64)


def _decode(
    record,
    prior,
    temperature,
    mode,
    effective_spacing_ft=16.0,
    persistence_weight=1.0,
    lambda_override=None,
):
    h_grid, d_grid = state_grids(prior)
    h_nodes = np.unique(np.maximum(np.rint(h_grid / NODE_STEP_FT).astype(np.int64), 1))
    # Consecutive small lengths guarantee every terminal node is reachable;
    # their actual MD lengths still receive the fitted log-normal cost.
    h_nodes = np.unique(np.r_[1, 2, 3, h_nodes]).astype(np.int64)
    tw_gr = record["tw_gr_raw"] if mode == "raw_reference" else record["tw_gr_affine"]
    mode_code = {
        "full": 0,
        "raw_reference": 1,
        "gaussian": 2,
        "no_persistence": 3,
        "previous_dip_centre": 4,
    }[mode]
    node_rows = _node_rows(record)
    path, objective = _beam_decode(
        record["md"],
        record["z"],
        record["gr"],
        node_rows,
        record["anchor_tvt"],
        record["previous_dip"],
        record["previous_dip"] + record["previous_z_rate"],
        record["tw_tvt"],
        tw_gr,
        record["gamma"],
        h_nodes,
        d_grid,
        prior["length_log_mu"],
        prior["length_log_sigma"],
        (
            prior["lambda_previous_dip"]
            if lambda_override is None
            else float(lambda_override)
        ),
        prior["sigma_persist"],
        prior["dip_low"],
        prior["dip_high"],
        temperature,
        BEAM_SIZE,
        MAX_EMISSION_ROWS,
        float(effective_spacing_ft),
        float(persistence_weight),
        mode_code,
    )
    if not np.isfinite(path).all():
        raise RuntimeError(f"{record['well']}: beam did not reach terminal node")
    evaluation = record["evaluation"]
    prediction = np.interp(record["md"][evaluation], record["md"][node_rows], path)
    return prediction, float(objective), int(len(node_rows))


def _metric_row(record, prediction, mode, objective, nodes):
    rows = record["evaluation"]
    truth = record["truth"][rows]
    error = prediction - truth
    distance = record["md"][rows] - record["anchor_md"]
    output = {
        "well": record["well"],
        "mode": mode,
        "rows": int(len(rows)),
        "exact_sse": float(np.sum(error**2)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
        "objective": objective,
        "nodes": nodes,
    }
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        output[f"{name}_rows"] = int(mask.sum())
        output[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return output


def _summarise(frame):
    rows = int(frame["rows"].sum())
    sse = float(frame["exact_sse"].sum())
    well_rmse = np.sqrt(frame["exact_sse"] / frame["rows"])
    output = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "exact_sse": sse,
        "rmse": float(np.sqrt(sse / rows)),
        "mae": float(frame["absolute_error_sum"].sum() / rows),
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    }
    for name, _, _ in STRATA:
        count = int(frame[f"{name}_rows"].sum())
        local_sse = float(frame[f"{name}_sse"].sum())
        output[f"{name}_rmse"] = float(np.sqrt(local_sse / count)) if count else np.nan
    return output


v3 = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
prior = v3["selected"]["prior"]
temperature = float(v3["selected"]["emission_temperature"])

folds = pd.read_csv(V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
valid_wells = sorted(folds.loc[folds["fold"].eq(V.APPLICATION_FOLD), "well"])
if len(valid_wells) != 155:
    raise RuntimeError(f"Expected 155 Fold2 wells, got {len(valid_wells)}")

records = []
for number, well in enumerate(valid_wells, start=1):
    records.append(load_record(well))
    if number % 50 == 0 or number == len(valid_wells):
        print(f"STRIDE v4 load Fold2 {number}/{len(valid_wells)}", flush=True)

started = time.time()
_decode(records[0], prior, temperature, "full")
bywell_rows = []
prediction_rows = []
for mode in MODES:
    for number, record in enumerate(records, start=1):
        prediction, objective, nodes = _decode(record, prior, temperature, mode)
        bywell_rows.append(_metric_row(record, prediction, mode, objective, nodes))
        if mode == "full":
            rows = record["evaluation"]
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "id": [f"{record['well']}_{int(row)}" for row in rows],
                        "well": record["well"],
                        "row_idx": rows,
                        "truth": record["truth"][rows],
                        "stride_v4": prediction,
                        "distance_ft": record["md"][rows] - record["anchor_md"],
                    }
                )
            )
        if number % 25 == 0 or number == len(records):
            print(f"STRIDE v4 Fold2 {mode} {number}/{len(records)}", flush=True)

bywell = pd.DataFrame(bywell_rows)
bywell.to_csv(OUT_DIR / "fold2_bywell_components.csv", index=False)
predictions = pd.concat(prediction_rows, ignore_index=True)
predictions.to_csv(OUT_DIR / "fold2_rows.csv.gz", index=False, compression="gzip")

summary_rows = []
for mode, frame in bywell.groupby("mode", sort=False):
    summary_rows.append({"mode": mode, **_summarise(frame)})
summary = pd.DataFrame(summary_rows)
full = summary.loc[summary["mode"].eq("full")].iloc[0].to_dict()
summary["degradation_vs_full"] = summary["rmse"] - float(full["rmse"])
summary.to_csv(OUT_DIR / "fold2_component_summary.csv", index=False)


def _mode(name):
    return summary.loc[summary["mode"].eq(name)].iloc[0].to_dict()


raw = _mode("raw_reference")
gaussian = _mode("gaussian")
no_persistence = _mode("no_persistence")
previous_only = _mode("previous_dip_centre")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_full_sequence_explicit_hlen_deltaTVT_beam",
    "sample_unit": "one complete physical well",
    "raw_inputs_only": True,
    "uses_sp45_track8_mainline_strong_cp12_gold_neighbor": False,
    "protocol": {
        "prior_folds": list(V.PRIOR_FOLDS),
        "selection_fold": V.SELECTION_FOLD,
        "application_fold": V.APPLICATION_FOLD,
        "untouched_fold": V.UNTOUCHED_FOLD,
        "figure1_source": str(V3_VERDICT),
    },
    "decoder": {
        "state": "explicit h_len x delta_TVT",
        "node_step_ft": NODE_STEP_FT,
        "delta_tvt_step_ft": float(V.D_STEP_FT),
        "beam_size": BEAM_SIZE,
        "maximum_emission_rows": MAX_EMISSION_ROWS,
        "full_sequence_backtracking": True,
        "terminal_selection": "global MAP",
    },
    "selected_prior": prior,
    "selected_emission_temperature": temperature,
    "fold2": full,
    "controls": {
        "raw_reference": raw,
        "gaussian": gaussian,
        "no_persistence": no_persistence,
        "previous_dip_centre": previous_only,
    },
    "component_effects_ft": {
        "affine_reference": float(raw["rmse"] - full["rmse"]),
        "cauchy_vs_gaussian": float(gaussian["rmse"] - full["rmse"]),
        "persistence": float(no_persistence["rmse"] - full["rmse"]),
        "dynamic_drilled_trend": float(previous_only["rmse"] - full["rmse"]),
    },
    "author_reference_not_a_gate": {
        "backbone_pooled_all_773": 9.99,
        "persistence_ablation_cost": 2.30,
        "typewell_reference_ablation_cost": 1.26,
        "drilled_trend_value": 0.93,
    },
    "component_direction_pass": bool(
        raw["rmse"] > full["rmse"]
        and gaussian["rmse"] > full["rmse"]
        and no_persistence["rmse"] > full["rmse"]
        and previous_only["rmse"] > full["rmse"]
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "Component directions match -> freeze assumptions and generate five-fold OOF"
        if (
            raw["rmse"] > full["rmse"]
            and gaussian["rmse"] > full["rmse"]
            and no_persistence["rmse"] > full["rmse"]
            and previous_only["rmse"] > full["rmse"]
        )
        else "Repair only the component whose matched ablation has the wrong direction"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSTRIDE V4 FOLD2 COMPONENT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
