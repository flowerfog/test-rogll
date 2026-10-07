"""Stochastic alignment confidence gate for the v27/CP8 tail blend.

The endpoint family is frozen by v28.  This experiment asks one narrower
question: can a deployable path-posterior confidence signal improve the fixed
0.35 blend between v27 Neighbor and the Fold3-selected CP8 tail endpoint?

Fold 3 is the only training/selection fold.  Five grouped splits inside Fold 3
select shrinkage around the fixed blend.  Fold 2 confirms once and Fold 1 is the
final check.  Hidden predictions and hidden labels are never observed.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold


EXPERIMENT = "frontier_6p39_stochastic_dtw_confidence_v29"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
FIXED_WEIGHT_CP8 = 0.35
NODE_SPACING_FT = 64.0
WINDOW_HALF_FT = 32.0
MAX_WINDOW_ROWS = 17
OFFSETS_FT = np.arange(-48.0, 48.01, 3.0, dtype=np.float64)
TRANSITION_SIGMAS_FT = (4.0, 8.0, 16.0)
EMISSION_TEMPERATURE = 1.0
PATH_SAMPLES = 32
SHRINKAGES = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))
SEED = 20260726
started = time.time()


def _find_data_root() -> Path:
    candidates = (
        ROOT / "rogii-wellbore-geology-prediction",
        ROOT,
        Path.cwd(),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    )
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate original competition train directory")


DATA_ROOT = _find_data_root()


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


def _deduplicate(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna().groupby("axis", as_index=False)["value"].median().sort_values("axis")
    if len(frame) < 32:
        raise RuntimeError("typewell has too few finite unique TVT rows")
    return frame["axis"].to_numpy(np.float64), frame["value"].to_numpy(np.float64)


def _robust_scale(values, floor=5.0):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = float(np.median(values))
    return float(max(1.4826 * np.median(np.abs(values - centre)), floor))


def _load_raw(well, local):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    horizontal = horizontal.sort_values("MD").reset_index(drop=True)
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(
        limit_direction="both"
    ).to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    row_idx = local["row_idx"].to_numpy(np.int64)
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise RuntimeError(f"{well}: OOF row index outside horizontal file")
    raw_distance = md[row_idx] - md[row_idx[0]] + float(local["md_since"].iloc[0])
    distance_gap = np.abs(raw_distance - local["md_since"].to_numpy(np.float64))
    if not np.isfinite(distance_gap).all() or float(distance_gap.max()) > 0.1:
        raise RuntimeError(f"{well}: OOF/raw MD mismatch {np.nanmax(distance_gap)}")

    tw_tvt, tw_gr = _deduplicate(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(gr))
    expected = np.interp(tvt_input[known], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    valid = np.isfinite(expected)
    residual = gr[known][valid] - expected[valid]
    shift = float(np.median(residual)) if len(residual) else 0.0
    gamma = _robust_scale(residual - shift, 5.0)
    return {
        "distance": local["md_since"].to_numpy(np.float64),
        "gr": gr[row_idx],
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr + shift,
        "gamma": gamma,
        "prefix_shift": shift,
    }


def _node_positions(distance):
    distance = np.asarray(distance, np.float64)
    targets = np.arange(max(0.0, float(distance[0])), float(distance[-1]) + 0.5, NODE_SPACING_FT)
    positions = np.searchsorted(distance, targets, side="left")
    positions = np.clip(positions, 0, len(distance) - 1)
    positions = np.unique(np.concatenate(([0], positions, [len(distance) - 1]))).astype(np.int64)
    return positions


def _emission_cost(raw, parent, node_positions):
    distance = raw["distance"]
    gr = raw["gr"]
    costs = np.empty((len(node_positions), len(OFFSETS_FT)), np.float64)
    for node_number, position in enumerate(node_positions):
        centre = distance[position]
        rows = np.flatnonzero(np.abs(distance - centre) <= WINDOW_HALF_FT)
        if len(rows) > MAX_WINDOW_ROWS:
            rows = rows[np.linspace(0, len(rows) - 1, MAX_WINDOW_ROWS).round().astype(int)]
        candidate = parent[rows, None] + OFFSETS_FT[None, :]
        expected = np.interp(
            candidate.ravel(), raw["tw_tvt"], raw["tw_gr"], left=np.nan, right=np.nan
        ).reshape(candidate.shape)
        residual = (gr[rows, None] - expected) / max(raw["gamma"], 1e-6)
        value = np.log1p(np.minimum(residual * residual, 1e4))
        value[~np.isfinite(value)] = 20.0
        costs[node_number] = np.mean(value, axis=0)
        costs[node_number] -= np.min(costs[node_number])
    return costs


def _transition_log(sigma):
    difference = OFFSETS_FT[None, :] - OFFSETS_FT[:, None]
    log_transition = -0.5 * (difference / float(sigma)) ** 2
    log_transition -= logsumexp(log_transition, axis=1, keepdims=True)
    return log_transition


def _posterior(costs, sigma, seed):
    emission = -np.asarray(costs, np.float64) / EMISSION_TEMPERATURE
    transition = _transition_log(sigma)
    nodes, states = emission.shape
    alpha = np.empty((nodes, states), np.float64)
    alpha[0] = emission[0] - logsumexp(emission[0])
    for node in range(1, nodes):
        alpha[node] = emission[node] + logsumexp(alpha[node - 1, :, None] + transition, axis=0)
        alpha[node] -= logsumexp(alpha[node])
    beta = np.zeros((nodes, states), np.float64)
    for node in range(nodes - 2, -1, -1):
        beta[node] = logsumexp(
            transition + emission[node + 1][None, :] + beta[node + 1][None, :], axis=1
        )
        beta[node] -= logsumexp(beta[node])
    log_marginal = alpha + beta
    log_marginal -= logsumexp(log_marginal, axis=1, keepdims=True)
    marginal = np.exp(log_marginal)

    samples = None
    if abs(float(sigma) - 8.0) < 1e-9:
        rng = np.random.default_rng(seed)
        samples = np.empty((PATH_SAMPLES, nodes), np.int16)
        probability = np.exp(alpha[-1] - logsumexp(alpha[-1]))
        samples[:, -1] = rng.choice(states, PATH_SAMPLES, p=probability)
        for node in range(nodes - 2, -1, -1):
            for sample in range(PATH_SAMPLES):
                next_state = int(samples[sample, node + 1])
                log_probability = alpha[node] + transition[:, next_state]
                probability = np.exp(log_probability - logsumexp(log_probability))
                samples[sample, node] = rng.choice(states, p=probability)
    return marginal, samples


def _alignment_features(costs, sigma, marginal, samples=None):
    entropy = -np.sum(marginal * np.log(np.maximum(marginal, 1e-12)), axis=1) / math.log(
        marginal.shape[1]
    )
    mean = marginal @ OFFSETS_FT
    std = np.sqrt(np.maximum(marginal @ (OFFSETS_FT ** 2) - mean * mean, 0.0))
    ordered = np.sort(marginal, axis=1)
    mode_index = np.argmax(marginal, axis=1)
    output = {
        f"align_entropy_s{int(sigma)}": entropy,
        f"align_std_s{int(sigma)}": std,
        f"align_maxprob_s{int(sigma)}": ordered[:, -1],
        f"align_topgap_s{int(sigma)}": ordered[:, -1] - ordered[:, -2],
        f"align_mean_offset_s{int(sigma)}": mean,
        f"align_mode_offset_s{int(sigma)}": OFFSETS_FT[mode_index],
    }
    if samples is not None:
        values = OFFSETS_FT[samples]
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
        output["align_sample_iqr_s8"] = q75 - q25
        output["align_sample_median_s8"] = median
        output["align_sample_consensus_s8"] = np.mean(np.abs(values - median[None, :]) <= 3.0, axis=0)
    output[f"align_cost_range_s{int(sigma)}"] = np.quantile(costs, 0.9, axis=1) - np.min(costs, axis=1)
    return output


def _block_target(local, node_positions, neighbor, cp8):
    truth = local["tvt_true"].to_numpy(np.float64)
    target = np.empty(len(node_positions), np.float64)
    counts = np.empty(len(node_positions), np.int32)
    boundaries = np.empty(len(node_positions) + 1, np.int64)
    boundaries[0] = 0
    boundaries[-1] = len(local)
    if len(node_positions) > 1:
        boundaries[1:-1] = ((node_positions[:-1] + node_positions[1:]) // 2).astype(np.int64)
    for node in range(len(node_positions)):
        rows = np.arange(boundaries[node], boundaries[node + 1], dtype=np.int64)
        error = neighbor[rows] - truth[rows]
        direction = cp8[rows] - neighbor[rows]
        denominator = float(np.dot(direction, direction))
        target[node] = (
            float(np.clip(-np.dot(error, direction) / denominator, 0.0, 1.0))
            if denominator > 1e-9 else FIXED_WEIGHT_CP8
        )
        counts[node] = len(rows)
    return target, counts


def _metric(local, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = local["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    well_mse = pd.Series(error * error).groupby(local["well"].to_numpy(), sort=False).mean()
    output = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
    }
    distance = local["md_since"].to_numpy(np.float64)
    for name, mask in {
        "near_0_1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        output[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
        )
    return output


def _model(seed):
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=350,
        learning_rate=0.025,
        num_leaves=9,
        max_depth=4,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )


def _row_weights(frame, nodes, node_prediction, shrinkage):
    weights = np.full(len(frame), FIXED_WEIGHT_CP8, np.float64)
    prediction_map = dict(zip(nodes.index.to_numpy(), node_prediction))
    for well, row_indices in frame.groupby("well", sort=False).indices.items():
        row_indices = np.asarray(row_indices, np.int64)
        local_nodes = nodes[nodes["well"].eq(well)].sort_values("node_distance")
        values = np.array([prediction_map[index] for index in local_nodes.index], np.float64)
        values = FIXED_WEIGHT_CP8 + float(shrinkage) * (values - FIXED_WEIGHT_CP8)
        values = np.clip(values, 0.0, 0.80)
        weights[row_indices] = np.interp(
            frame.loc[row_indices, "md_since"].to_numpy(np.float64),
            local_nodes["node_distance"].to_numpy(np.float64),
            values,
        )
    return weights


if not V28_PATH.is_file():
    raise FileNotFoundError(V28_PATH)

frame = pd.read_parquet(V28_PATH)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since", "parent",
    "neighbor_v27", "tail_cp8_endpoint",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V28 OOF missing columns: {missing}")
frame = frame[list(required)].copy()
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any() or len(frame) != 2270643:
    raise RuntimeError(f"Unexpected V28 OOF coverage: {len(frame)}")

neighbor = frame["neighbor_v27"].to_numpy(np.float64)
cp8 = frame["tail_cp8_endpoint"].to_numpy(np.float64)
fixed_prediction = neighbor + FIXED_WEIGHT_CP8 * (cp8 - neighbor)
if not np.array_equal(neighbor[frame["md_since"].to_numpy() <= 1000.0], fixed_prediction[frame["md_since"].to_numpy() <= 1000.0]):
    raise RuntimeError("V28 endpoint family does not exactly freeze the first 1000 ft")

print(
    "V29 CONFIG",
    {
        "rows": len(frame), "wells": frame["well"].nunique(),
        "states": len(OFFSETS_FT), "node_spacing_ft": NODE_SPACING_FT,
        "transition_sigmas_ft": TRANSITION_SIGMAS_FT,
        "path_samples": PATH_SAMPLES, "data_root": str(DATA_ROOT),
    },
    flush=True,
)

node_parts = []
failures = []
well_groups = list(frame.groupby("well", sort=True))
for number, (well, local_source) in enumerate(well_groups, 1):
    local = local_source.reset_index()
    try:
        raw = _load_raw(well, local)
        node_positions = _node_positions(raw["distance"])
        local_parent = local["parent"].to_numpy(np.float64)
        costs = _emission_cost(raw, local_parent, node_positions)
        feature_values = {}
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()[:8], "little"
        )
        for sigma in TRANSITION_SIGMAS_FT:
            marginal, samples = _posterior(costs, sigma, stable_seed + int(sigma * 10))
            feature_values.update(_alignment_features(costs, sigma, marginal, samples))
        local_neighbor = local["neighbor_v27"].to_numpy(np.float64)
        local_cp8 = local["tail_cp8_endpoint"].to_numpy(np.float64)
        target, counts = _block_target(local, node_positions, local_neighbor, local_cp8)
        part = pd.DataFrame({
            "well": well,
            "fold": int(local["fold"].iloc[0]),
            "node_position": node_positions,
            "node_distance": raw["distance"][node_positions],
            "node_rows": counts,
            "oracle_weight_cp8": target,
            "neighbor_minus_parent": local_neighbor[node_positions] - local_parent[node_positions],
            "cp8_minus_parent": local_cp8[node_positions] - local_parent[node_positions],
            "cp8_minus_neighbor": local_cp8[node_positions] - local_neighbor[node_positions],
            "abs_cp8_minus_neighbor": np.abs(local_cp8[node_positions] - local_neighbor[node_positions]),
            "prefix_gr_shift": raw["prefix_shift"],
            "prefix_gr_scale": raw["gamma"],
        })
        for name, values in feature_values.items():
            part[name] = values
        node_parts.append(part)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 25 == 0 or number == len(well_groups):
        print(f"V29 alignment {number}/{len(well_groups)} failures={len(failures)}", flush=True)

pd.DataFrame(failures, columns=["well", "reason"]).to_csv(
    OUT_DIR / "record_failures.csv", index=False
)
if failures:
    raise RuntimeError(f"{len(failures)} stochastic alignment failures: {failures[:5]}")
nodes = pd.concat(node_parts, ignore_index=True)
nodes = nodes[nodes["node_distance"] > 1000.0].reset_index(drop=True)
nodes.to_parquet(OUT_DIR / "node_features.parquet", index=False, compression="zstd")

generic_features = [
    "node_distance", "neighbor_minus_parent", "cp8_minus_parent",
    "cp8_minus_neighbor", "abs_cp8_minus_neighbor", "prefix_gr_shift", "prefix_gr_scale",
]
alignment_features = [column for column in nodes.columns if column.startswith("align_")]
full_features = generic_features + alignment_features
for columns in (generic_features, full_features):
    nodes[columns] = nodes[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _crossfit_feature_set(feature_names, label):
    selection = nodes[nodes["fold"].eq(SELECTION_FOLD)].copy()
    prediction = np.full(len(selection), np.nan, np.float64)
    groups = selection["well"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    for split, (train_index, valid_index) in enumerate(
        splitter.split(selection, groups=groups), 1
    ):
        model = _model(SEED + split)
        model.fit(
            selection.iloc[train_index][feature_names],
            selection.iloc[train_index]["oracle_weight_cp8"],
            sample_weight=selection.iloc[train_index]["node_rows"],
        )
        prediction[valid_index] = model.predict(selection.iloc[valid_index][feature_names])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Missing Fold3 crossfit predictions for {label}")
    return selection, np.clip(prediction, 0.0, 1.0)


crossfit = {}
shrinkage_rows = []
selection_frame = frame[frame["fold"].eq(SELECTION_FOLD)].reset_index(drop=True)
selection_neighbor = selection_frame["neighbor_v27"].to_numpy(np.float64)
selection_cp8 = selection_frame["tail_cp8_endpoint"].to_numpy(np.float64)
fixed_selection = selection_neighbor + FIXED_WEIGHT_CP8 * (selection_cp8 - selection_neighbor)
fixed_selection_metric = _metric(selection_frame, fixed_selection, "fixed_035")
neighbor_selection_metric = _metric(selection_frame, selection_neighbor, "neighbor_v27")

for label, feature_names in (("generic", generic_features), ("alignment", full_features)):
    local_nodes, node_prediction = _crossfit_feature_set(feature_names, label)
    for shrinkage in SHRINKAGES:
        row_weight = _row_weights(selection_frame, local_nodes, node_prediction, shrinkage)
        candidate = selection_neighbor + row_weight * (selection_cp8 - selection_neighbor)
        metric = _metric(selection_frame, candidate, label)
        shrinkage_rows.append({
            "model": label,
            "shrinkage": float(shrinkage),
            "gain_vs_fixed": fixed_selection_metric["rmse"] - metric["rmse"],
            "p95_change_vs_fixed": metric["p95_well_rmse"] - fixed_selection_metric["p95_well_rmse"],
            "p95_change_vs_neighbor": metric["p95_well_rmse"] - neighbor_selection_metric["p95_well_rmse"],
            **metric,
        })
    crossfit[label] = (local_nodes, node_prediction)

shrinkage_scan = pd.DataFrame(shrinkage_rows)
shrinkage_scan["risk_pass"] = shrinkage_scan["p95_change_vs_neighbor"] <= 0.10
selected_rows = []
for label in ("generic", "alignment"):
    local = shrinkage_scan[shrinkage_scan["model"].eq(label)].sort_values(
        ["risk_pass", "rmse", "p95_well_rmse"], ascending=[False, True, True], kind="stable"
    )
    selected_rows.append(local.iloc[0])
selected_scan = pd.DataFrame(selected_rows).reset_index(drop=True)
shrinkage_scan.to_csv(OUT_DIR / "fold3_shrinkage_scan.csv", index=False)

node_predictions = {}
for label, feature_names in (("generic", generic_features), ("alignment", full_features)):
    train = nodes[nodes["fold"].eq(SELECTION_FOLD)]
    model = _model(SEED + (100 if label == "alignment" else 200))
    model.fit(
        train[feature_names], train["oracle_weight_cp8"], sample_weight=train["node_rows"]
    )
    node_predictions[label] = np.clip(model.predict(nodes[feature_names]), 0.0, 1.0)
    model.booster_.save_model(str(OUT_DIR / f"{label}_weight_model.txt"))

summary_rows = []
row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["neighbor_v27"] = neighbor.astype(np.float32)
row_output["cp8_tail"] = cp8.astype(np.float32)
row_output["fixed_035"] = fixed_prediction.astype(np.float32)

for held_fold, role in ((3, "selection_crossfit"), (2, "confirmation"), (1, "final")):
    local = frame[frame["fold"].eq(held_fold)].reset_index()
    local_neighbor = local["neighbor_v27"].to_numpy(np.float64)
    local_cp8 = local["tail_cp8_endpoint"].to_numpy(np.float64)
    local_fixed = local_neighbor + FIXED_WEIGHT_CP8 * (local_cp8 - local_neighbor)
    metrics = {"neighbor_v27": _metric(local, local_neighbor, "neighbor_v27")}
    metrics["fixed_035"] = _metric(local, local_fixed, "fixed_035")
    for label in ("generic", "alignment"):
        shrinkage = float(selected_scan.loc[selected_scan["model"].eq(label), "shrinkage"].iloc[0])
        if held_fold == SELECTION_FOLD:
            local_nodes, local_node_prediction = crossfit[label]
        else:
            local_nodes = nodes[nodes["fold"].eq(held_fold)]
            local_node_prediction = node_predictions[label][nodes["fold"].eq(held_fold).to_numpy()]
        row_weight = _row_weights(local, local_nodes, local_node_prediction, shrinkage)
        prediction = local_neighbor + row_weight * (local_cp8 - local_neighbor)
        metrics[label] = _metric(local, prediction, label)
        row_output.loc[local["index"].to_numpy(), f"weight_{label}"] = row_weight.astype(np.float32)
        row_output.loc[local["index"].to_numpy(), f"prediction_{label}"] = prediction.astype(np.float32)

    # Matched control: preserve each Fold2/Fold1 well's weight shape but assign
    # it to another well.  This tests whether alignment evidence is tied to the
    # correct physical well rather than merely regularizing weights.
    if held_fold in (1, 2):
        alignment_weight = row_output.loc[
            local["index"].to_numpy(), "weight_alignment"
        ].to_numpy(np.float64)
        shuffled_weight = alignment_weight.copy()
        wells = sorted(local["well"].unique())
        rng = np.random.default_rng(SEED + held_fold)
        permuted = list(rng.permutation(wells))
        source_curves = {}
        for well in wells:
            mask = local["well"].eq(well).to_numpy()
            source_curves[well] = (
                local.loc[mask, "md_since"].to_numpy(np.float64), alignment_weight[mask]
            )
        for target_well, source_well in zip(wells, permuted):
            mask = local["well"].eq(target_well).to_numpy()
            source_distance, source_weight = source_curves[source_well]
            shuffled_weight[mask] = np.interp(
                local.loc[mask, "md_since"].to_numpy(np.float64),
                source_distance, source_weight,
            )
        shuffled = local_neighbor + shuffled_weight * (local_cp8 - local_neighbor)
        metrics["shuffled_alignment"] = _metric(local, shuffled, "shuffled_alignment")

    for endpoint, metric in metrics.items():
        summary_rows.append({"fold": held_fold, "role": role, **metric})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
row_output.to_parquet(OUT_DIR / "stochastic_weight_oof.parquet", index=False, compression="zstd")


def _m(fold, endpoint):
    return summary[summary["fold"].eq(fold) & summary["endpoint"].eq(endpoint)].iloc[0]


alignment_selection = selected_scan[selected_scan["model"].eq("alignment")].iloc[0]
generic_selection = selected_scan[selected_scan["model"].eq("generic")].iloc[0]
gates = {
    "fold3_crossfit_alignment_beats_fixed": bool(alignment_selection["gain_vs_fixed"] > 0.0),
    "fold3_alignment_beats_generic": bool(
        alignment_selection["rmse"] < generic_selection["rmse"]
    ),
    "fold2_alignment_beats_fixed": bool(_m(2, "alignment")["rmse"] < _m(2, "fixed_035")["rmse"]),
    "fold1_alignment_beats_fixed": bool(_m(1, "alignment")["rmse"] < _m(1, "fixed_035")["rmse"]),
    "fold2_far_nonworse": bool(
        _m(2, "alignment")["far_gt2000_rmse"] <= _m(2, "fixed_035")["far_gt2000_rmse"] + 0.01
    ),
    "fold1_far_nonworse": bool(
        _m(1, "alignment")["far_gt2000_rmse"] <= _m(1, "fixed_035")["far_gt2000_rmse"] + 0.01
    ),
    "fold2_p95_change_le_0p10": bool(
        _m(2, "alignment")["p95_well_rmse"] <= _m(2, "neighbor_v27")["p95_well_rmse"] + 0.10
    ),
    "fold1_p95_change_le_0p10": bool(
        _m(1, "alignment")["p95_well_rmse"] <= _m(1, "neighbor_v27")["p95_well_rmse"] + 0.10
    ),
    "shuffle_destroys_half_fold2_increment": bool(
        (_m(2, "fixed_035")["rmse"] - _m(2, "shuffled_alignment")["rmse"])
        <= 0.5 * max(_m(2, "fixed_035")["rmse"] - _m(2, "alignment")["rmse"], 0.0) + 1e-9
    ),
}

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_grouped_crossfit_stochastic_alignment_weight_Fold2_confirmation_Fold1_final",
    "literature_basis": {
        "stochastic_dtw": "10.30632/pjv63n6-2022a11",
        "principle": "path concentration indicates stable log correlation; path dispersion indicates ambiguity",
    },
    "hidden_predictions_observed": False,
    "endpoint_family": {
        "first": "v27 Neighbor lowpass tail",
        "second": "v28 Fold3-selected CP8 tail dose 0.40",
        "fixed_weight_cp8": FIXED_WEIGHT_CP8,
    },
    "posterior": {
        "offset_states_ft": OFFSETS_FT.tolist(),
        "node_spacing_ft": NODE_SPACING_FT,
        "window_half_ft": WINDOW_HALF_FT,
        "transition_sigmas_ft": list(TRANSITION_SIGMAS_FT),
        "path_samples": PATH_SAMPLES,
        "prefix_reference": "median shift and robust scale from visible prefix",
        "emission": "Cauchy GR cost",
    },
    "coverage": {
        "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
        "nodes": int(len(nodes)), "features": int(len(full_features)),
    },
    "fold3_selection": selected_scan.to_dict(orient="records"),
    "metrics": summary.to_dict(orient="records"),
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> freeze the stochastic confidence model and port its raw-data feature builder to Kaggle"
        if all(gates.values()) else
        "FAIL -> deploy the fixed v28 CP8 blend candidate or retain v27; close stochastic confidence"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV29 FOLD3 SHRINKAGE SELECTION", flush=True)
print(selected_scan.to_string(index=False), flush=True)
print("\nV29 SUMMARY", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT", flush=True)
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
