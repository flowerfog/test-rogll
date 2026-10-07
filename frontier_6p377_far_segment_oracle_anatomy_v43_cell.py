"""Anatomy of V40's label-only far-segment endpoint oracle.

The audit measures how much of the 256-ft oracle survives when each well may
change endpoint mode only 0, 1, 2, or 4 times.  It is deliberately
non-deployable: truth supplies segment costs.  Its purpose is to distinguish a
coherent, low-switch path ceiling from fragmented label-by-label cherry-picking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_far_segment_oracle_anatomy_v43"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
SEGMENT_FT = 256.0
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
SWITCH_BUDGETS = (0, 1, 2, 4)
started = time.time()


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


def _smoothstep(values):
    t = np.clip((np.asarray(values, np.float64) - FAR_START_FT) / FADE_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _centred_mean(x, values, width):
    x = np.asarray(x, np.float64)
    values = np.asarray(values, np.float64)
    left = np.searchsorted(x, x - width / 2.0, side="left")
    right = np.searchsorted(x, x + width / 2.0, side="right")
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[right] - prefix[left]) / np.maximum(right - left, 1)


def _metrics(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local["tvt_true"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    wells = local["well"].astype(str).to_numpy()
    per_well = pd.Series(error**2).groupby(wells, sort=False).mean().pow(0.5)
    result = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
    }
    for name, mask in {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
    return result


def _decode_budget(costs, budget):
    """Return the minimum-cost state path with at most ``budget`` switches."""
    costs = np.asarray(costs, np.float64)
    steps, states = costs.shape
    if not steps:
        return np.empty(0, np.int16)
    value = np.full((steps, budget + 1, states), np.inf, np.float64)
    parent_state = np.full((steps, budget + 1, states), -1, np.int16)
    parent_budget = np.full((steps, budget + 1, states), -1, np.int16)
    value[0, 0] = costs[0]
    for step in range(1, steps):
        for used in range(budget + 1):
            for state in range(states):
                best_value = value[step - 1, used, state]
                best_state = state
                best_used = used
                if used > 0:
                    previous = value[step - 1, used - 1].copy()
                    previous[state] = np.inf
                    candidate_state = int(np.argmin(previous))
                    if previous[candidate_state] < best_value:
                        best_value = previous[candidate_state]
                        best_state = candidate_state
                        best_used = used - 1
                value[step, used, state] = costs[step, state] + best_value
                parent_state[step, used, state] = best_state
                parent_budget[step, used, state] = best_used
    flat = int(np.argmin(value[-1]))
    used, state = np.unravel_index(flat, value[-1].shape)
    path = np.empty(steps, np.int16)
    path[-1] = state
    for step in range(steps - 1, 0, -1):
        previous_state = int(parent_state[step, used, state])
        previous_used = int(parent_budget[step, used, state])
        state, used = previous_state, previous_used
        path[step - 1] = state
    return path


def _run_lengths(path):
    path = np.asarray(path, np.int16)
    if not len(path):
        return []
    boundaries = np.flatnonzero(path[1:] != path[:-1]) + 1
    return np.diff(np.concatenate([[0], boundaries, [len(path)]])).tolist()


for required in (V33_PATH, V28_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v33_columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", "pre_neighbor_cp8", BASE_COLUMN]
frame = pd.read_parquet(V33_PATH, columns=v33_columns)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

tail_columns = sorted(column for column in pq.ParquetFile(V28_PATH).schema.names if column.startswith("tail_"))
tail = pd.read_parquet(V28_PATH, columns=["id"] + tail_columns)
tail["id"] = tail["id"].astype(str)
frame = frame.merge(tail, on="id", how="left", validate="one_to_one")

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fade = _smoothstep(distance)
prefix = distance <= FAR_START_FT

candidate_paths = [base]
usable_names = ["base_6p377"]
for name in ["pre_neighbor_cp8"] + tail_columns:
    absolute = frame[name].to_numpy(np.float64)
    finite = np.isfinite(absolute)
    if finite.mean() < 0.95:
        continue
    correction = np.where(finite, np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[prefix] = base[prefix]
    candidate_paths.append(candidate)
    usable_names.append(name)
candidates = np.column_stack(candidate_paths)
if candidates.shape[1] < 3 or not np.isfinite(candidates).all():
    raise RuntimeError(f"Invalid candidate bank: {usable_names}")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}
policies = {f"max_switches_{budget}": budget for budget in SWITCH_BUDGETS}
policies["unbounded"] = None
predictions = {name: base.copy() for name in policies}
choice_rows = []

for well_number, (well, indices) in enumerate(groups.items(), 1):
    local_distance = distance[indices]
    far = local_distance > FAR_START_FT
    if not np.any(far):
        continue
    block_for_row = np.floor(np.maximum(local_distance - FAR_START_FT, 0.0) / SEGMENT_FT).astype(np.int32)
    blocks = np.unique(block_for_row[far])
    costs = np.zeros((len(blocks), candidates.shape[1]), np.float64)
    block_rows = []
    for step, block in enumerate(blocks):
        rows = np.flatnonzero(far & (block_for_row == block))
        global_rows = indices[rows]
        errors = candidates[global_rows] - truth[global_rows, None]
        costs[step] = np.sum(errors**2, axis=0)
        block_rows.append(rows)

    for policy, budget in policies.items():
        path = np.argmin(costs, axis=1).astype(np.int16) if budget is None else _decode_budget(costs, budget)
        selected = base[indices].copy()
        for step, (block, rows, state) in enumerate(zip(blocks, block_rows, path)):
            selected[rows] = candidates[indices[rows], int(state)]
            choice_rows.append({
                "well": well,
                "fold": int(frame.loc[indices[0], "fold"]),
                "policy": policy,
                "block": int(block),
                "step": int(step),
                "state": int(state),
                "candidate": usable_names[int(state)],
                "selected_sse": float(costs[step, state]),
                "base_sse": float(costs[step, 0]),
            })
        correction = _centred_mean(local_distance, selected - base[indices], SMOOTH_FT)
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        local_prediction = base[indices] + fade[indices] * correction
        local_prediction[local_distance <= FAR_START_FT] = base[indices][local_distance <= FAR_START_FT]
        predictions[policy][indices] = local_prediction
    if well_number % 100 == 0 or well_number == len(groups):
        print(f"V43 wells {well_number}/{len(groups)}", flush=True)

base_result = _metrics(frame, base, "base_6p377")
summary_rows = [base_result]
fold_rows = []
for policy, prediction in predictions.items():
    result = _metrics(frame, prediction, policy)
    result["gain"] = base_result["rmse"] - result["rmse"]
    result["far_gain"] = base_result["far_gt2000_rmse"] - result["far_gt2000_rmse"]
    result["p95_change"] = result["p95_well_rmse"] - base_result["p95_well_rmse"]
    summary_rows.append(result)
    for fold in (1, 2, 3):
        mask = frame["fold"].to_numpy(np.int8) == fold
        local = frame.loc[mask].reset_index(drop=True)
        local_base = _metrics(local, base[mask], "base")
        local_result = _metrics(local, prediction[mask], policy)
        fold_rows.append({
            "fold": fold,
            "policy": policy,
            "base_rmse": local_base["rmse"],
            "rmse": local_result["rmse"],
            "gain": local_base["rmse"] - local_result["rmse"],
            "far_gain": local_base["far_gt2000_rmse"] - local_result["far_gt2000_rmse"],
            "p95_change": local_result["p95_well_rmse"] - local_base["p95_well_rmse"],
        })

choices = pd.DataFrame(choice_rows)
anatomy_rows = []
for policy, local in choices.groupby("policy", sort=False):
    switch_counts = []
    run_lengths = []
    nonbase_rates = []
    for _, well_rows in local.groupby("well", sort=False):
        path = well_rows.sort_values("step")["state"].to_numpy(np.int16)
        switch_counts.append(int(np.sum(path[1:] != path[:-1])))
        run_lengths.extend(_run_lengths(path))
        nonbase_rates.append(float(np.mean(path != 0)))
    anatomy_rows.append({
        "policy": policy,
        "mean_switches_per_well": float(np.mean(switch_counts)),
        "median_switches_per_well": float(np.median(switch_counts)),
        "p90_switches_per_well": float(np.quantile(switch_counts, 0.90)),
        "median_run_blocks": float(np.median(run_lengths)),
        "mean_nonbase_rate": float(np.mean(nonbase_rates)),
    })

summary = pd.DataFrame(summary_rows)
fold_summary = pd.DataFrame(fold_rows)
anatomy = pd.DataFrame(anatomy_rows)
unbounded_gain = float(summary.loc[summary["endpoint"].eq("unbounded"), "gain"].iloc[0])
budget2_gain = float(summary.loc[summary["endpoint"].eq("max_switches_2"), "gain"].iloc[0])
retention = budget2_gain / max(unbounded_gain, 1e-12)
checks = {
    "unbounded_matches_v40_gain_approximately": bool(abs(unbounded_gain - 0.9086) <= 0.10),
    "two_switch_gain_ge_0p30": bool(budget2_gain >= 0.30),
    "two_switch_retains_half_oracle": bool(retention >= 0.50),
    "near_is_bitwise_frozen": bool(all(np.array_equal(prediction[prefix], base[prefix]) for prediction in predictions.values())),
}

summary.to_csv(OUT_DIR / "summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
anatomy.to_csv(OUT_DIR / "switch_anatomy.csv", index=False)
choices.to_csv(OUT_DIR / "oracle_choices.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "label_only_switch_budget_anatomy",
    "deployable": False,
    "truth_usage": "truth supplies candidate SSE for each far segment",
    "lineage": {"base": str(V33_PATH), "candidate_source": str(V28_PATH)},
    "candidate_names": usable_names,
    "base": base_result,
    "summary": summary.to_dict(orient="records"),
    "switch_anatomy": anatomy.to_dict(orient="records"),
    "two_switch_oracle_retention": retention,
    "checks": checks,
    "coherent_oracle": bool(checks["two_switch_gain_ge_0p30"] and checks["two_switch_retains_half_oracle"]),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "COHERENT -> seek a well-level change-point observation before another selector"
        if checks["two_switch_gain_ge_0p30"] and checks["two_switch_retains_half_oracle"]
        else "FRAGMENTED -> V40 gain is label-cherry-picked; abandon this candidate bank"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV43 SUMMARY")
print(summary.to_string(index=False))
print("\nV43 SWITCH ANATOMY")
print(anatomy.to_string(index=False))
print("\nV43 FOLDS")
print(fold_summary.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default))
print("saved:", OUT_DIR)
