"""Sparse joint offset/dip posterior for raw GR-typewell alignment.

V51 already represented surface and previous dip inside a pruned MAP beam.
This experiment tests the missing inference form: a full forward-backward
posterior on a sparse (offset, dip) state graph.  No historical prediction,
CP, Neighbor, Gold, spatial feature, or leaderboard artifact is consumed.

Fold 4/5 provide the frozen structural prior and V30 emission recipe.  A
deterministic 24-well subset of Fold 3 selects process noise; the remaining
Fold-3 wells confirm it.  Fold 2 is decoded only after confirmation passes.
Fold 1 is untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


EXPERIMENT = "raw_joint_offset_dip_posterior_v53"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V30_VERDICT = V30_DIR / "verdict.json"
SOURCE = next(
    (
        path
        for path in (
            ROOT / "raw_stochastic_alignment_standalone_v30_cell.py",
            Path.cwd() / "raw_stochastic_alignment_standalone_v30_cell.py",
        )
        if path.is_file()
    ),
    None,
)
if SOURCE is None:
    raise FileNotFoundError("Upload raw_stochastic_alignment_standalone_v30_cell.py")
for required in (V30_VERDICT, V30_DIR / "folds.csv"):
    if not required.is_file():
        raise FileNotFoundError(required)

PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
OFFSET_HALF_WIDTH_FT = 160.0
OFFSET_STEP_FT = 4.0
OFFSETS = np.arange(
    -OFFSET_HALF_WIDTH_FT,
    OFFSET_HALF_WIDTH_FT + 0.5 * OFFSET_STEP_FT,
    OFFSET_STEP_FT,
    dtype=np.float64,
)
DIPS = np.arange(-0.08, 0.0801, 0.01, dtype=np.float64)
ZERO_OFFSET = int(np.argmin(np.abs(OFFSETS)))
ZERO_DIP = int(np.argmin(np.abs(DIPS)))
CONFIGS = [
    {"id": "student_d0.01_o2", "dip_scale": 0.01, "offset_sigma": 2.0},
    {"id": "student_d0.01_o4", "dip_scale": 0.01, "offset_sigma": 4.0},
    {"id": "student_d0.02_o2", "dip_scale": 0.02, "offset_sigma": 2.0},
    {"id": "student_d0.02_o4", "dip_scale": 0.02, "offset_sigma": 4.0},
    {"id": "student_d0.04_o2", "dip_scale": 0.04, "offset_sigma": 2.0},
    {"id": "student_d0.04_o4", "dip_scale": 0.04, "offset_sigma": 4.0},
]
DISTANCE_EDGES = (-np.inf, 500.0, 1000.0, 2000.0, np.inf)
DISTANCE_LABELS = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")


def _bootstrap(path: Path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_stochastic_v30_joint_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


V = _bootstrap(SOURCE)
V.OFFSETS_FT = OFFSETS.copy()
V.ZERO_STATE = ZERO_OFFSET
EDGE_CACHE = {}


def _records(wells, label):
    records, failures = [], []
    for number, well in enumerate(wells, 1):
        try:
            records.append(V._load_record(str(well)))
        except Exception as exc:
            failures.append({"well": str(well), "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"V53 load {label} {number}/{len(wells)} failures={len(failures)}", flush=True)
    pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
    if len(records) < max(len(wells) - 2, 1):
        raise RuntimeError(f"Only {len(records)}/{len(wells)} usable {label} records")
    return records


def _group_logsumexp(groups, values, size):
    output_max = np.full(size, -np.inf, np.float64)
    np.maximum.at(output_max, groups, values)
    finite = np.isfinite(output_max[groups]) & np.isfinite(values)
    total = np.bincount(
        groups[finite],
        weights=np.exp(values[finite] - output_max[groups[finite]]),
        minlength=size,
    )
    output = np.full(size, -np.inf, np.float64)
    valid = total > 0.0
    output[valid] = output_max[valid] + np.log(total[valid])
    return output


def _edges(delta_md, config):
    key = (round(float(delta_md), 3), config["id"])
    cached = EDGE_CACHE.get(key)
    if cached is not None:
        return cached
    sources, destinations, scores = [], [], []
    dip_scale = float(config["dip_scale"])
    offset_sigma = float(config["offset_sigma"])
    dip_count = len(DIPS)
    for offset_index, offset in enumerate(OFFSETS):
        for dip_index, dip in enumerate(DIPS):
            source_state = offset_index * dip_count + dip_index
            for next_dip_index, next_dip in enumerate(DIPS):
                predicted = offset + 0.5 * (dip + next_dip) * float(delta_md)
                centre = int(np.rint((predicted - OFFSETS[0]) / OFFSET_STEP_FT))
                for next_offset_index in range(centre - 1, centre + 2):
                    if not 0 <= next_offset_index < len(OFFSETS):
                        continue
                    innovation = OFFSETS[next_offset_index] - predicted
                    dip_change = (next_dip - dip) / dip_scale
                    score = -math.log1p(dip_change * dip_change)
                    score -= 0.5 * (innovation / offset_sigma) ** 2
                    sources.append(source_state)
                    destinations.append(next_offset_index * dip_count + next_dip_index)
                    scores.append(score)
    source = np.asarray(sources, np.int32)
    destination = np.asarray(destinations, np.int32)
    log_probability = np.asarray(scores, np.float64)
    state_count = len(OFFSETS) * len(DIPS)
    normalizer = _group_logsumexp(source, log_probability, state_count)
    log_probability -= normalizer[source]
    cached = (source, destination, log_probability)
    EDGE_CACHE[key] = cached
    return cached


def _joint_decode(record, prior, config, emission_config, shuffled=False):
    rows, distance, state_surface, costs = V._emission(record, prior, emission_config)
    if shuffled:
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(f"v53-shuffle:{record['well']}".encode("ascii")).digest()[:8],
                "little",
            )
        )
        for node in range(1, len(costs)):
            costs[node] = costs[node, rng.permutation(len(OFFSETS))]
    log_offset_emission = -costs / float(emission_config["temperature"])
    state_count = len(OFFSETS) * len(DIPS)
    log_emission = np.repeat(log_offset_emission, len(DIPS), axis=1)
    alpha = np.full((len(rows), state_count), -np.inf, np.float64)
    alpha[0, ZERO_OFFSET * len(DIPS) + ZERO_DIP] = 0.0
    transitions = [None]
    for node in range(1, len(rows)):
        edge = _edges(distance[node] - distance[node - 1], config)
        transitions.append(edge)
        source, destination, log_probability = edge
        alpha[node] = log_emission[node] + _group_logsumexp(
            destination, alpha[node - 1, source] + log_probability, state_count
        )
        if not np.isfinite(alpha[node]).any():
            raise RuntimeError(f"{record['well']}: joint posterior died at node {node}")
        alpha[node] -= logsumexp(alpha[node])

    beta = np.zeros((len(rows), state_count), np.float64)
    for node in range(len(rows) - 2, -1, -1):
        source, destination, log_probability = transitions[node + 1]
        values = log_probability + log_emission[node + 1, destination] + beta[node + 1, destination]
        beta[node] = _group_logsumexp(source, values, state_count)
        finite = np.isfinite(beta[node])
        beta[node, finite] -= logsumexp(beta[node, finite])

    log_marginal = alpha + beta
    log_marginal -= logsumexp(log_marginal, axis=1, keepdims=True)
    marginal = np.exp(log_marginal).reshape(len(rows), len(OFFSETS), len(DIPS))
    offset_probability = marginal.sum(axis=2)
    dip_probability = marginal.sum(axis=1)
    mean_offset = offset_probability @ OFFSETS
    mean_dip = dip_probability @ DIPS
    centre_surface = state_surface[:, ZERO_OFFSET]
    mean_surface = centre_surface + mean_offset
    evaluation = record["evaluation"]
    prediction_surface = np.interp(record["md"][evaluation], record["md"][rows], mean_surface)
    prediction = prediction_surface - record["z"][evaluation]
    entropy = -np.sum(marginal * np.log(np.maximum(marginal, 1e-12)), axis=(1, 2))
    entropy /= math.log(state_count)
    return prediction, {
        "mean_entropy": float(np.mean(entropy[1:])),
        "mean_abs_dip": float(np.mean(np.abs(mean_dip[1:]))),
        "terminal_dip": float(mean_dip[-1]),
        "nodes": int(len(rows)),
    }


def _offset_only_decode(record, prior, emission_config):
    predictions, diagnostics = V._decode(
        record, prior, emission_config, no_gr=False, sample_paths=False
    )
    return predictions["posterior_mean"], diagnostics


def _decode_many(records, prior, emission_config, joint_config, label, shuffled=False):
    rows = []
    for number, record in enumerate(records, 1):
        if joint_config is None:
            prediction, diagnostics = _offset_only_decode(record, prior, emission_config)
            endpoint = "offset_only"
        else:
            prediction, diagnostics = _joint_decode(
                record, prior, joint_config, emission_config, shuffled=shuffled
            )
            endpoint = joint_config["id"] + ("_shuffle" if shuffled else "")
        truth = record["truth"][record["evaluation"]]
        distance = record["md"][record["evaluation"]] - record["anchor_md"]
        error = prediction - truth
        payload = {
            "well": record["well"],
            "endpoint": endpoint,
            "rows": int(len(error)),
            "sse": float(np.dot(error, error)),
            "sae": float(np.abs(error).sum()),
            **diagnostics,
        }
        for left, right, name in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_LABELS):
            mask = (distance > left) & (distance <= right)
            payload[f"{name}_rows"] = int(mask.sum())
            payload[f"{name}_sse"] = float(np.dot(error[mask], error[mask])) if mask.any() else 0.0
        rows.append(payload)
        if number % 12 == 0 or number == len(records):
            print(f"V53 decode {label} {number}/{len(records)}", flush=True)
    return pd.DataFrame(rows)


def _summary(frame):
    total_rows = int(frame["rows"].sum())
    well_rmse = np.sqrt(frame["sse"] / frame["rows"])
    output = {
        "endpoint": str(frame.iloc[0]["endpoint"]),
        "rows": total_rows,
        "wells": int(len(frame)),
        "exact_sse": float(frame["sse"].sum()),
        "rmse": float(np.sqrt(frame["sse"].sum() / total_rows)),
        "mae": float(frame["sae"].sum() / total_rows),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "mean_entropy": float(frame["mean_entropy"].mean()),
    }
    for name in DISTANCE_LABELS:
        count = int(frame[f"{name}_rows"].sum())
        output[f"{name}_rmse"] = (
            float(np.sqrt(frame[f"{name}_sse"].sum() / count)) if count else np.nan
        )
    return output


def _true_dip_coverage(records):
    covered, total, values = 0, 0, []
    for record in records:
        rows = V._node_rows(record)
        surface = record["truth"][rows] + record["z"][rows]
        delta_md = np.diff(record["md"][rows])
        rate = np.diff(surface) / np.maximum(delta_md, 1e-6)
        centre_rate = float(np.clip(
            record["prefix_surface_rate"], prior["rate_low"], prior["rate_high"]
        ))
        dip = rate - centre_rate
        values.extend(dip.tolist())
        covered += int(np.sum((dip >= DIPS[0]) & (dip <= DIPS[-1])))
        total += len(dip)
    values = np.asarray(values, np.float64)
    return {
        "segments": int(total),
        "coverage": float(covered / total),
        "p95_abs_true_dip": float(np.quantile(np.abs(values), 0.95)),
        "max_abs_true_dip": float(np.max(np.abs(values))),
    }


started = time.time()
v30 = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
prior = dict(v30["outer_train_prior"])
emission_config = dict(v30["selected_config"])
probe_edges = _edges(float(V.NODE_SPACING_FT), CONFIGS[0])
print(
    "V53 CONFIG",
    {
        "offset_states": len(OFFSETS),
        "dip_states": len(DIPS),
        "joint_states": len(OFFSETS) * len(DIPS),
        "sparse_edges_per_typical_step": len(probe_edges[0]),
        "discovery_configs": len(CONFIGS),
        "fold1_touched": False,
        "historical_prediction_dependencies": [],
    },
    flush=True,
)
folds = pd.read_csv(V30_DIR / "folds.csv", dtype={"well": str})
fold3_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].astype(str))
fold3_records = _records(fold3_wells, "fold3")
fold3_records = sorted(
    fold3_records,
    key=lambda record: hashlib.sha256(f"v53:{record['well']}".encode("ascii")).digest(),
)
discovery = fold3_records[:DISCOVERY_WELLS]
confirmation = fold3_records[DISCOVERY_WELLS:]

coverage = _true_dip_coverage(fold3_records)
baseline_discovery = _decode_many(
    discovery, prior, emission_config, None, "discovery offset-only"
)
baseline_discovery_summary = _summary(baseline_discovery)
scan_rows = []
discovery_parts = [baseline_discovery]
for config in CONFIGS:
    result = _decode_many(discovery, prior, emission_config, config, f"discovery {config['id']}")
    discovery_parts.append(result)
    summary = _summary(result)
    scan_rows.append({
        **config,
        **summary,
        "gain_vs_offset_only": baseline_discovery_summary["rmse"] - summary["rmse"],
        "far_gain_vs_offset_only": (
            baseline_discovery_summary["far_gt2000_rmse"] - summary["far_gt2000_rmse"]
        ),
    })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "far_gt2000_rmse"], kind="stable")
selected_config = next(config for config in CONFIGS if config["id"] == scan.iloc[0]["id"])
scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
pd.concat(discovery_parts, ignore_index=True).to_csv(
    OUT_DIR / "fold3_discovery_bywell.csv", index=False
)

baseline_confirmation = _decode_many(
    confirmation, prior, emission_config, None, "confirmation offset-only"
)
joint_confirmation = _decode_many(
    confirmation, prior, emission_config, selected_config, "confirmation joint"
)
shuffle_confirmation = _decode_many(
    confirmation,
    prior,
    emission_config,
    selected_config,
    "confirmation shuffled-emission",
    shuffled=True,
)
baseline_summary = _summary(baseline_confirmation)
joint_summary = _summary(joint_confirmation)
shuffle_summary = _summary(shuffle_confirmation)
confirmation_gain = baseline_summary["rmse"] - joint_summary["rmse"]
confirmation_far_gain = (
    baseline_summary["far_gt2000_rmse"] - joint_summary["far_gt2000_rmse"]
)
shuffle_loss = shuffle_summary["rmse"] - joint_summary["rmse"]
confirmation_gates = {
    "dip_state_coverage_ge_0p95": coverage["coverage"] >= 0.95,
    "confirmation_gain_ge_0p30": confirmation_gain >= 0.30,
    "confirmation_far_gain_ge_0p30": confirmation_far_gain >= 0.30,
    "shuffled_emission_loses_ge_2": shuffle_loss >= 2.0,
}
mechanism_pass = bool(all(confirmation_gates.values()))

summary_rows = [
    {"stage": "fold3_confirmation", **baseline_summary},
    {"stage": "fold3_confirmation", **joint_summary},
    {"stage": "fold3_confirmation", **shuffle_summary},
]
application = None
if mechanism_pass:
    fold2_records = _records(fold2_wells, "fold2")
    base2_frame = _decode_many(fold2_records, prior, emission_config, None, "Fold2 offset-only")
    joint2_frame = _decode_many(
        fold2_records, prior, emission_config, selected_config, "Fold2 joint"
    )
    base2 = _summary(base2_frame)
    joint2 = _summary(joint2_frame)
    summary_rows.extend(
        ({"stage": "fold2_application", **base2}, {"stage": "fold2_application", **joint2})
    )
    application = {
        "baseline": base2,
        "joint": joint2,
        "gain": base2["rmse"] - joint2["rmse"],
        "far_gain": base2["far_gt2000_rmse"] - joint2["far_gt2000_rmse"],
    }
    pd.concat([base2_frame, joint2_frame], ignore_index=True).to_csv(
        OUT_DIR / "fold2_bywell.csv", index=False
    )

summary_frame = pd.DataFrame(summary_rows)
summary_frame.to_csv(OUT_DIR / "summary.csv", index=False)
pd.concat(
    [baseline_confirmation, joint_confirmation, shuffle_confirmation], ignore_index=True
).to_csv(OUT_DIR / "fold3_confirmation_bywell.csv", index=False)
strict_pass = bool(
    mechanism_pass
    and application is not None
    and application["gain"] >= 0.20
    and application["far_gain"] >= 0.20
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_joint_offset_dip_posterior_selection_confirmation_Fold2_conditional",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "architecture": "sparse forward-backward posterior over absolute offset x dip",
    "state_space": {
        "offset_states": len(OFFSETS),
        "offset_half_width_ft": OFFSET_HALF_WIDTH_FT,
        "offset_step_ft": OFFSET_STEP_FT,
        "dip_states": len(DIPS),
        "dip_min": float(DIPS[0]),
        "dip_max": float(DIPS[-1]),
        "joint_states": int(len(OFFSETS) * len(DIPS)),
    },
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "discovery_wells": len(discovery),
        "confirmation_wells": len(confirmation),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "frozen_emission": emission_config,
    "dip_observability": coverage,
    "selected_config": selected_config,
    "confirmation": {
        "baseline": baseline_summary,
        "joint": joint_summary,
        "shuffled_emission": shuffle_summary,
        "gain": confirmation_gain,
        "far_gain": confirmation_far_gain,
        "shuffle_loss": shuffle_loss,
    },
    "confirmation_gates": confirmation_gates,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain joint posterior as a raw endpoint and audit against 6.377"
        if strict_pass
        else "FAIL -> explicit joint posterior does not resolve the raw GR alias; close literature state-space branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV53 DISCOVERY SCAN")
print(scan.to_string(index=False), flush=True)
print("\nV53 SUMMARY")
print(summary_frame.to_string(index=False), flush=True)
print("\nV53 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
