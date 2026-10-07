"""Diagnose why the raw stochastic-alignment endpoint selects cycle skips.

V30 proved that the assigned typewell is causal but its 64-ft point emission
still selects wrong long-lived modes.  This audit does not train a model and
does not read historical predictions.  On frozen Fold 2 it separates:

1. absolute-state coverage / quantisation ceiling;
2. local emission observability;
3. whether accumulating 128--512 ft of evidence improves truth rank;
4. whether the transition posterior suppresses otherwise-visible truth.

The result chooses the next raw-data architecture: wider state support,
segment likelihood, or a different observation representation.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


EXPERIMENT = "raw_stochastic_alignment_observability_v31"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V30_VERDICT = V30_DIR / "verdict.json"
SOURCE_CANDIDATES = (
    ROOT / "raw_stochastic_alignment_standalone_v30_cell.py",
    Path.cwd() / "raw_stochastic_alignment_standalone_v30_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload raw_stochastic_alignment_standalone_v30_cell.py")
if not V30_VERDICT.exists():
    raise FileNotFoundError(V30_VERDICT)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module_name = "raw_stochastic_v30_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module


V = _bootstrap(SOURCE_PATH)
OFFSETS = V.OFFSETS_FT
ZERO_STATE = V.ZERO_STATE
NODE_SPACING_FT = V.NODE_SPACING_FT
WINDOW_NODES = (1, 2, 4, 8)


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


def _posterior(costs, distance, prior, config):
    emission = -costs / float(config["temperature"])
    nodes, states = emission.shape
    transitions = [None]
    alpha = np.full((nodes, states), -np.inf, np.float64)
    alpha[0, ZERO_STATE] = 0.0
    for node in range(1, nodes):
        transition = V._transition_log(
            distance[node] - distance[node - 1],
            prior,
            config["transition_multiplier"],
        )
        transitions.append(transition)
        alpha[node] = emission[node] + logsumexp(
            alpha[node - 1, :, None] + transition, axis=0
        )
        alpha[node] -= logsumexp(alpha[node])
    beta = np.zeros((nodes, states), np.float64)
    for node in range(nodes - 2, -1, -1):
        beta[node] = logsumexp(
            transitions[node + 1]
            + emission[node + 1][None, :]
            + beta[node + 1][None, :],
            axis=1,
        )
        beta[node] -= logsumexp(beta[node])
    value = alpha + beta
    value -= logsumexp(value, axis=1, keepdims=True)
    return np.exp(value)


def _trailing_sum(costs, width):
    if width <= 1:
        return costs.copy()
    cumulative = np.vstack([np.zeros((1, costs.shape[1])), np.cumsum(costs, axis=0)])
    output = np.empty_like(costs)
    for node in range(len(costs)):
        left = max(0, node - int(width) + 1)
        output[node] = cumulative[node + 1] - cumulative[left]
        output[node] /= node - left + 1
    return output


def _rank_features(score, true_offset):
    true_state = np.argmin(np.abs(OFFSETS[None, :] - true_offset[:, None]), axis=1)
    order = np.argsort(score, axis=1)
    inverse = np.argsort(order, axis=1)
    rank = inverse[np.arange(len(score)), true_state] + 1
    top1 = OFFSETS[order[:, 0]]
    top8 = OFFSETS[order[:, :8]]
    truth_cost = score[np.arange(len(score)), true_state]
    alias_wins = []
    for node, offset in enumerate(true_offset):
        aliases = []
        for signed in (-24.0, 24.0):
            target = offset + signed
            if OFFSETS[0] <= target <= OFFSETS[-1]:
                state = int(np.argmin(np.abs(OFFSETS - target)))
                aliases.append(score[node, state])
        if aliases:
            alias_wins.append(float(truth_cost[node] <= min(aliases)))
    return {
        "top1_within5": float(np.mean(np.abs(top1 - true_offset) <= 5.0)),
        "top8_within5": float(
            np.mean(np.min(np.abs(top8 - true_offset[:, None]), axis=1) <= 5.0)
        ),
        "median_truth_rank": float(np.median(rank)),
        "p90_truth_rank": float(np.quantile(rank, 0.90)),
        "alias24_win_rate": float(np.mean(alias_wins)) if alias_wins else np.nan,
    }


def _path_metric(record, rows, node_surface):
    eval_rows = record["evaluation"]
    surface = np.interp(record["md"][eval_rows], record["md"][rows], node_surface)
    prediction = surface - record["z"][eval_rows]
    error = prediction - record["truth"][eval_rows]
    distance = record["md"][eval_rows] - record["anchor_md"]
    far = distance > 2000.0
    return {
        "rows": int(len(error)),
        "sse": float(np.dot(error, error)),
        "far_rows": int(far.sum()),
        "far_sse": float(np.dot(error[far], error[far])) if far.any() else 0.0,
    }


started = time.time()
verdict_v30 = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
prior = verdict_v30["outer_train_prior"]
config = verdict_v30["selected_config"]
folds = pd.read_csv(V30_DIR / "folds.csv", dtype={"well": str})
fold2_wells = sorted(folds.loc[folds["fold"].eq(2), "well"].astype(str))

node_rows = []
path_rows = []
failures = []
for number, well in enumerate(fold2_wells, 1):
    try:
        record = V._load_record(well)
        rows, distance, state_surface, costs = V._emission(record, prior, config)
        true_surface = record["truth"][rows] + record["z"][rows]
        true_offset = true_surface - state_surface[:, ZERO_STATE]
        covered = (true_offset >= OFFSETS[0]) & (true_offset <= OFFSETS[-1])
        quantised_state = np.argmin(
            np.abs(OFFSETS[None, :] - true_offset[:, None]), axis=1
        )
        oracle_surface = state_surface[np.arange(len(rows)), quantised_state]
        centre_surface = state_surface[:, ZERO_STATE]
        posterior = _posterior(costs, distance, prior, config)
        posterior_score = -np.log(np.maximum(posterior, 1e-12))

        common = {
            "well": well,
            "nodes": int(len(rows)),
            "coverage": float(np.mean(covered)),
            "max_abs_true_offset": float(np.max(np.abs(true_offset))),
        }
        for label, score in (("posterior", posterior_score),):
            node_rows.append({**common, "evidence": label, **_rank_features(score, true_offset)})
        for width in WINDOW_NODES:
            aggregate = _trailing_sum(costs, width)
            label = f"emission_{int(width * NODE_SPACING_FT)}ft"
            node_rows.append({**common, "evidence": label, **_rank_features(aggregate, true_offset)})
            argmin_surface = state_surface[np.arange(len(rows)), np.argmin(aggregate, axis=1)]
            path_rows.append(
                {"well": well, "path": f"argmin_{int(width * NODE_SPACING_FT)}ft", **_path_metric(record, rows, argmin_surface)}
            )
        path_rows.append(
            {"well": well, "path": "state_oracle", **_path_metric(record, rows, oracle_surface)}
        )
        path_rows.append(
            {"well": well, "path": "prior_centre", **_path_metric(record, rows, centre_surface)}
        )
        posterior_surface = np.sum(posterior * state_surface, axis=1)
        path_rows.append(
            {"well": well, "path": "posterior_mean", **_path_metric(record, rows, posterior_surface)}
        )
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 25 == 0 or number == len(fold2_wells):
        print(f"V31 Fold2 audit {number}/{len(fold2_wells)} failures={len(failures)}", flush=True)

pd.DataFrame(failures, columns=["well", "reason"]).to_csv(
    OUT_DIR / "failures.csv", index=False
)
if failures:
    raise RuntimeError(f"{len(failures)} V31 failures: {failures[:5]}")

node_frame = pd.DataFrame(node_rows)
path_frame = pd.DataFrame(path_rows)
node_frame.to_csv(OUT_DIR / "bywell_observability.csv", index=False)
path_frame.to_csv(OUT_DIR / "bywell_paths.csv", index=False)

node_summary = (
    node_frame.groupby("evidence", as_index=False)
    .agg(
        wells=("well", "nunique"),
        mean_coverage=("coverage", "mean"),
        p95_max_abs_true_offset=("max_abs_true_offset", lambda x: np.quantile(x, 0.95)),
        top1_within5=("top1_within5", "mean"),
        top8_within5=("top8_within5", "mean"),
        median_truth_rank=("median_truth_rank", "median"),
        p90_truth_rank=("p90_truth_rank", "median"),
        alias24_win_rate=("alias24_win_rate", "mean"),
    )
    .sort_values("top8_within5", ascending=False)
)
path_summary_rows = []
for path, local in path_frame.groupby("path", sort=False):
    rows = int(local["rows"].sum())
    far_rows = int(local["far_rows"].sum())
    path_summary_rows.append(
        {
            "path": path,
            "rows": rows,
            "wells": int(local["well"].nunique()),
            "rmse": float(np.sqrt(local["sse"].sum() / rows)),
            "far_gt2000_rmse": (
                float(np.sqrt(local["far_sse"].sum() / far_rows)) if far_rows else np.nan
            ),
            "median_well_rmse": float(np.median(np.sqrt(local["sse"] / local["rows"]))),
        }
    )
path_summary = pd.DataFrame(path_summary_rows).sort_values("rmse")
node_summary.to_csv(OUT_DIR / "observability_summary.csv", index=False)
path_summary.to_csv(OUT_DIR / "path_summary.csv", index=False)


def _node(label):
    return node_summary[node_summary["evidence"].eq(label)].iloc[0]


def _path(label):
    return path_summary[path_summary["path"].eq(label)].iloc[0]


point = _node("emission_64ft")
long = _node("emission_512ft")
posterior_node = _node("posterior")
oracle = _path("state_oracle")
gates = {
    "state_oracle_rmse_le_3": bool(oracle["rmse"] <= 3.0),
    "mean_state_coverage_ge_0p99": bool(node_frame.groupby("well")["coverage"].first().mean() >= 0.99),
    "long_context_top8_gain_ge_0p10": bool(long["top8_within5"] - point["top8_within5"] >= 0.10),
    "long_context_median_rank_improves": bool(long["median_truth_rank"] < point["median_truth_rank"]),
    "posterior_does_not_destroy_top8": bool(
        posterior_node["top8_within5"] >= 0.8 * long["top8_within5"]
    ),
}
if not gates["state_oracle_rmse_le_3"] or not gates["mean_state_coverage_ge_0p99"]:
    next_action = "Repair absolute state support/centre before changing the observation model"
elif gates["long_context_top8_gain_ge_0p10"] and gates["long_context_median_rank_improves"]:
    next_action = "Build v32 segment-likelihood stochastic DP; long context resolves local aliases"
else:
    next_action = "Stop scale aggregation; build a learned raw/typewell pair representation before another decoder"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_raw_absolute_state_and_long_context_observability",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "source_v30": str(SOURCE_PATH),
    "frozen_config": config,
    "outer_train_prior": prior,
    "node_summary": node_summary.to_dict(orient="records"),
    "path_summary": path_summary.to_dict(orient="records"),
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": next_action,
    "elapsed_minutes": float((time.time() - started) / 60.0),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV31 OBSERVABILITY SUMMARY", flush=True)
print(node_summary.to_string(index=False), flush=True)
print("\nV31 PATH SUMMARY", flush=True)
print(path_summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT", flush=True)
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)

