"""Fold-2 gateway for strong-as-a-prior absolute-edge graph correction.

This is a restart-safe, CPU-only audit.  It consumes only artifacts already
written by the Drift-V3 and marker cross-fit experiments:

* candidate_pool_full_merged_oof_honest_fle.csv
* v3_constraint_profile_marker_audit_v0/horizon_edges.pkl
* v3_marker_event_crossfit_gateway_v1/crossfit_candidates.pkl

The optimisation variable is a correction relative to the fold-clean ``strong``
OOF path.  Therefore the strong-only arm reproduces the input prediction
exactly at row level.  Deployable Delta-S and cross-fitted marker observations
may move the path through a robust graph solve.  Exact Geology formation tops
are evaluated only as an explicitly labelled oracle; they never participate in
the deployable recipe selection or strict verdict.

Protocol
--------
* V3 and the marker classifier were trained on complete wells only.
* Fold 2 is split into five deterministic complete-well meta folds.
* For each held meta fold, graph hyperparameters are selected on the other four.
* Fold 1 is not read.
* No truth is used to generate Delta-S, marker, or strong-prior observations.
* The corridor arm uses hidden Geology labels and is oracle-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Paths and frozen protocol
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
EXPERIMENT = "strong_prior_absolute_edge_graph_gateway_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
MARKER_PATH = (
    OOF_ROOT
    / "v3_marker_event_crossfit_gateway_v1"
    / "crossfit_candidates.pkl"
)
EDGE_PATH = (
    OOF_ROOT
    / "v3_constraint_profile_marker_audit_v0"
    / "horizon_edges.pkl"
)

VALID_FOLD = 2
META_FOLDS = 5
EVENT_NMS_FT = 128.0
MAX_IRLS_ITERATIONS = 6
ANCHOR_WEIGHT = 1.0e4
RIDGE = 1.0e-6
SEED = 20260720

MAJOR_FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")

if POOL_PATH is None:
    raise FileNotFoundError(f"candidate pool not found under {OOF_ROOT}")
if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
for required_path in (MARKER_PATH, EDGE_PATH, DATA_ROOT / "train"):
    if not required_path.exists():
        raise FileNotFoundError(required_path)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _stable_meta_fold(well):
    digest = hashlib.sha256(str(well).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % META_FOLDS + 1


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(max(float(sse), 0.0) / max(int(rows), 1)))


def _huber_weight(residual, delta):
    residual = np.abs(np.asarray(residual, np.float64))
    return np.minimum(1.0, float(delta) / np.maximum(residual, 1e-8))


# ---------------------------------------------------------------------------
# Load the exact Fold-2 strong lineage and deployable evidence.
# ---------------------------------------------------------------------------


header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
required_columns = {"id", "tvt_true", "strong"}
if not required_columns.issubset(header):
    raise RuntimeError(f"pool misses {sorted(required_columns - set(header))}")
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong") if column in header]
pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": str, "well": str})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pd.to_numeric(pool["row_idx"], errors="raise").astype(np.int32)
for column in ("tvt_true", "strong"):
    pool[column] = pd.to_numeric(pool[column], errors="coerce").astype(np.float64)
if not np.isfinite(pool[["tvt_true", "strong"]].to_numpy()).all():
    raise RuntimeError("non-finite truth or strong values")

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
folds = folds.groupby("well", as_index=False)["fold"].first()
fold_map = dict(zip(folds["well"], pd.to_numeric(folds["fold"], errors="raise").astype(int)))
pool["fold"] = pool["well"].map(fold_map)
if pool["fold"].isna().any():
    raise RuntimeError("pool contains wells missing from fold metadata")
pool = (
    pool.loc[pool["fold"].eq(VALID_FOLD)]
    .sort_values(["well", "row_idx"])
    .reset_index(drop=True)
)
pool["meta_fold"] = pool["well"].map(_stable_meta_fold).astype(np.int8)

marker_candidates = pd.read_pickle(MARKER_PATH)
marker_candidates["well"] = marker_candidates["well"].astype(str)
edge_frame = pd.read_pickle(EDGE_PATH)
edge_frame["well"] = edge_frame["well"].astype(str)

fold2_wells = sorted(pool["well"].unique())
marker_wells = set(marker_candidates["well"].unique())
edge_wells = set(edge_frame["well"].unique())
missing_marker = sorted(set(fold2_wells) - marker_wells)
missing_edges = sorted(set(fold2_wells) - edge_wells)
if missing_marker or missing_edges:
    raise RuntimeError(
        f"evidence is incomplete; marker_missing={missing_marker[:10]}, "
        f"edge_missing={missing_edges[:10]}"
    )


@dataclass
class WellRecord:
    well: str
    meta_fold: int
    row_index: np.ndarray
    eval_md: np.ndarray
    distance: np.ndarray
    truth: np.ndarray
    strong: np.ndarray
    node_md: np.ndarray
    node_distance: np.ndarray
    strong_node: np.ndarray
    strong_offset: np.ndarray
    z_node: np.ndarray
    anchor_tvt: float
    marker: pd.DataFrame
    edges: pd.DataFrame
    corridor_low: float
    corridor_high: float


def _formation_tops(typewell):
    if "Geology" not in typewell or "TVT" not in typewell:
        return {}
    geology = typewell["Geology"].astype("string").str.strip()
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce")
    tops = {}
    for formation in MAJOR_FORMATIONS:
        values = tvt[geology.eq(formation)].dropna()
        if len(values):
            tops[formation] = float(values.min())
    return tops


def _anchor_corridor(anchor_tvt, tops):
    ordered = sorted((value, name) for name, value in tops.items())
    if len(ordered) < 2:
        return np.nan, np.nan
    values = np.asarray([value for value, _ in ordered], np.float64)
    position = int(np.searchsorted(values, anchor_tvt, side="right") - 1)
    if position < 0 or position >= len(values) - 1:
        return np.nan, np.nan
    return float(values[position]), float(values[position + 1])


def _build_record(well, frame):
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    row_index = frame["row_idx"].to_numpy(np.int64)
    if row_index.max() >= len(horizontal):
        raise RuntimeError(f"{well}: row index exceeds horizontal file")

    md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise RuntimeError(f"{well}: no visible TVT prefix")
    anchor = int(known[-1])
    anchor_md = float(md_all[anchor])
    anchor_tvt = float(tvt_input[anchor])

    eval_md = md_all[row_index]
    strong = frame["strong"].to_numpy(np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    marker = marker_candidates.loc[marker_candidates["well"].eq(well)].copy()
    node_table = marker[["node", "node_md"]].drop_duplicates().sort_values("node")
    expected_nodes = np.arange(len(node_table), dtype=np.int64)
    if not np.array_equal(node_table["node"].to_numpy(np.int64), expected_nodes):
        raise RuntimeError(f"{well}: marker nodes are not contiguous")
    node_md = node_table["node_md"].to_numpy(np.float64)
    interpolation_md = np.r_[anchor_md, eval_md]
    if np.any(np.diff(interpolation_md) < 0):
        raise RuntimeError(f"{well}: MD is not monotonic")
    strong_node = np.interp(node_md, interpolation_md, np.r_[anchor_tvt, strong])
    z_node = np.interp(node_md, md_all, z_all)
    edges = edge_frame.loc[edge_frame["well"].eq(well)].copy()
    low, high = _anchor_corridor(anchor_tvt, _formation_tops(typewell))
    return WellRecord(
        well=well,
        meta_fold=int(frame["meta_fold"].iloc[0]),
        row_index=row_index,
        eval_md=eval_md,
        distance=eval_md - anchor_md,
        truth=truth,
        strong=strong,
        node_md=node_md,
        node_distance=node_md - node_md[0],
        strong_node=strong_node,
        strong_offset=strong_node - anchor_tvt,
        z_node=z_node,
        anchor_tvt=anchor_tvt,
        marker=marker,
        edges=edges,
        corridor_low=low,
        corridor_high=high,
    )


started = time.time()
records = {}
for index, (well, frame) in enumerate(pool.groupby("well", sort=True), 1):
    records[well] = _build_record(well, frame)
    if index % 50 == 0 or index == len(fold2_wells):
        print(f"build graph records {index}/{len(fold2_wells)}", flush=True)


# ---------------------------------------------------------------------------
# Robust correction graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConfig:
    config_id: str
    arm: str
    edge_source: str = "none"
    edge_weight: float = 0.0
    marker_threshold: float = 1.1
    marker_top_events: int = 0
    marker_weight: float = 0.0
    marker_cap_ft: float = 0.0
    prior_tau_ft: float = 1600.0
    prior_floor: float = 0.20
    smooth_weight: float = 0.03
    corridor_weight: float = 0.0


def _config_id(arm, edge_source="none", edge_weight=0.0, threshold=1.1, top=0, marker_weight=0.0, cap=0.0):
    return (
        f"{arm}|e={edge_source}:{edge_weight:.3f}|"
        f"m={threshold:.2f}:{top}:{marker_weight:.2f}:{cap:.0f}"
    )


configs = [GraphConfig(config_id="strong_only", arm="strong")]
for source, weight in (("prior", 0.02), ("prior", 0.06), ("learned", 0.01), ("learned", 0.03)):
    configs.append(
        GraphConfig(
            config_id=_config_id("delta", source, weight),
            arm="delta",
            edge_source=source,
            edge_weight=weight,
        )
    )
for threshold in (0.75, 0.85, 0.92):
    for top_events in (1, 2):
        for marker_weight in (0.15, 0.35):
            for cap in (8.0, 16.0):
                marker_config = dict(
                    marker_threshold=threshold,
                    marker_top_events=top_events,
                    marker_weight=marker_weight,
                    marker_cap_ft=cap,
                )
                configs.append(
                    GraphConfig(
                        config_id=_config_id(
                            "marker", threshold=threshold, top=top_events,
                            marker_weight=marker_weight, cap=cap,
                        ),
                        arm="marker",
                        **marker_config,
                    )
                )
                for source, edge_weight in (("prior", 0.03), ("learned", 0.015)):
                    configs.append(
                        GraphConfig(
                            config_id=_config_id(
                                "delta_marker", source, edge_weight, threshold,
                                top_events, marker_weight, cap,
                            ),
                            arm="delta_marker",
                            edge_source=source,
                            edge_weight=edge_weight,
                            **marker_config,
                        )
                    )

config_by_id = {config.config_id: config for config in configs}
if len(config_by_id) != len(configs):
    raise RuntimeError("duplicate graph config ids")


def _select_marker_events(record, config, shuffled=False):
    if config.marker_top_events <= 0 or config.marker_weight <= 0:
        return []
    frame = record.marker.copy()
    frame = frame.loc[frame["event_probability"].ge(config.marker_threshold)]
    if not len(frame):
        return []
    marker_offsets = (
        frame[["marker", "marker_offset"]]
        .drop_duplicates("marker")
        .sort_values("marker")
    )
    marker_ids = marker_offsets["marker"].to_numpy(np.int64)
    marker_values = marker_offsets["marker_offset"].to_numpy(np.float64)
    shuffled_map = {
        int(marker_ids[index]): float(marker_values[(index + 1) % len(marker_values)])
        for index in range(len(marker_ids))
    }
    selected = []
    for row in frame.sort_values("event_probability", ascending=False).itertuples(index=False):
        node = int(row.node)
        if node <= 0 or node >= len(record.node_md):
            continue
        marker_offset = (
            shuffled_map[int(row.marker)] if shuffled else float(row.marker_offset)
        )
        raw = marker_offset - float(record.strong_offset[node])
        if abs(raw) > 64.0:
            continue
        if any(abs(float(row.node_md) - event[1]) < EVENT_NMS_FT for event in selected):
            continue
        if any(int(row.marker) == event[2] for event in selected):
            continue
        confidence = max(
            0.05,
            min(
                1.0,
                (float(row.event_probability) - config.marker_threshold)
                / max(1.0 - config.marker_threshold, 1e-6),
            ),
        )
        selected.append(
            (
                node,
                float(row.node_md),
                int(row.marker),
                float(np.clip(raw, -config.marker_cap_ft, config.marker_cap_ft)),
                config.marker_weight * confidence,
                float(row.event_probability),
            )
        )
        if len(selected) >= config.marker_top_events:
            break
    return selected


def _edge_observations(record, config):
    if config.edge_source == "none" or config.edge_weight <= 0:
        return []
    source_column = f"{config.edge_source}_delta_s"
    if source_column not in record.edges:
        raise RuntimeError(f"missing edge source {source_column}")
    output = []
    strong_s = record.strong_offset + (record.z_node - record.z_node[0])
    for horizon, frame in record.edges.groupby("horizon", sort=True):
        horizon = int(horizon)
        frame = frame.sort_values("distance_ft")
        expected = len(record.node_md) - horizon
        if len(frame) != expected:
            raise RuntimeError(
                f"{record.well}: horizon {horizon} has {len(frame)}/{expected} edges"
            )
        desired = frame[source_column].to_numpy(np.float64)
        strong_delta = strong_s[horizon:] - strong_s[:-horizon]
        correction_delta = np.clip(desired - strong_delta, -24.0, 24.0)
        per_edge_weight = config.edge_weight / math.sqrt(max(float(horizon), 1.0))
        for index, observation in enumerate(correction_delta):
            output.append((index, index + horizon, float(observation), per_edge_weight))
    return output


def _solve_graph(record, config, shuffled_marker=False, corridor=False):
    columns = len(record.node_md)
    if config.arm == "strong" and not corridor:
        return np.zeros(columns, np.float64), []

    row_indices = []
    col_indices = []
    values = []
    observations = []
    base_weights = []
    kinds = []
    row = 0

    def add(entries, observation, weight, kind):
        nonlocal row
        for column, value in entries:
            row_indices.append(row)
            col_indices.append(int(column))
            values.append(float(value))
        observations.append(float(observation))
        base_weights.append(float(weight))
        kinds.append(kind)
        row += 1

    prior_weight = config.prior_floor + np.exp(
        -record.node_distance / max(config.prior_tau_ft, 1.0)
    )
    for node, weight in enumerate(prior_weight):
        add(((node, 1.0),), 0.0, weight, "strong")
    add(((0, 1.0),), 0.0, ANCHOR_WEIGHT, "anchor")

    if columns >= 3 and config.smooth_weight > 0:
        for node in range(1, columns - 1):
            add(
                ((node - 1, 1.0), (node, -2.0), (node + 1, 1.0)),
                0.0,
                config.smooth_weight,
                "smooth",
            )

    for left, right, observation, weight in _edge_observations(record, config):
        add(((left, -1.0), (right, 1.0)), observation, weight, "delta")

    selected_events = _select_marker_events(record, config, shuffled=shuffled_marker)
    for node, _, _, observation, weight, _ in selected_events:
        add(((node, 1.0),), observation, weight, "marker")

    design = sparse.csr_matrix(
        (values, (row_indices, col_indices)), shape=(row, columns), dtype=np.float64
    )
    observation = np.asarray(observations, np.float64)
    base_weight = np.asarray(base_weights, np.float64)
    kind = np.asarray(kinds, dtype="U10")
    correction = np.zeros(columns, np.float64)

    for _ in range(MAX_IRLS_ITERATIONS):
        residual = design @ correction - observation
        robust = np.ones(len(residual), np.float64)
        robust[kind == "strong"] = _huber_weight(residual[kind == "strong"], 6.0)
        robust[kind == "delta"] = _huber_weight(residual[kind == "delta"], 4.0)
        robust[kind == "marker"] = _huber_weight(residual[kind == "marker"], 4.0)

        extra_design = None
        extra_observation = None
        extra_weight = None
        if corridor and np.isfinite(record.corridor_low) and np.isfinite(record.corridor_high):
            absolute = record.strong_node + correction
            low_violation = absolute < record.corridor_low
            high_violation = absolute > record.corridor_high
            violated = low_violation | high_violation
            if violated.any():
                nodes = np.flatnonzero(violated)
                target = np.where(
                    low_violation[nodes],
                    record.corridor_low - record.strong_node[nodes],
                    record.corridor_high - record.strong_node[nodes],
                )
                extra_design = sparse.csr_matrix(
                    (
                        np.ones(len(nodes), np.float64),
                        (np.arange(len(nodes)), nodes),
                    ),
                    shape=(len(nodes), columns),
                )
                extra_observation = target
                extra_weight = np.full(len(nodes), config.corridor_weight, np.float64)

        weight = np.maximum(base_weight * robust, 1e-8)
        weighted_design = design.multiply(weight[:, None])
        system = design.T @ weighted_design
        rhs = design.T @ (weight * observation)
        if extra_design is not None:
            system = system + extra_design.T @ extra_design.multiply(extra_weight[:, None])
            rhs = rhs + extra_design.T @ (extra_weight * extra_observation)
        system = system + RIDGE * sparse.eye(columns, format="csr")
        updated = np.asarray(spsolve(system.tocsc(), rhs), np.float64)
        updated -= updated[0]
        if np.max(np.abs(updated - correction)) < 1e-5:
            correction = updated
            break
        correction = updated
    return correction, selected_events


def _predict_rows(record, config, shuffled_marker=False, corridor=False):
    correction_node, events = _solve_graph(
        record, config, shuffled_marker=shuffled_marker, corridor=corridor
    )
    correction = np.interp(
        record.eval_md,
        record.node_md,
        correction_node,
        left=float(correction_node[0]),
        right=float(correction_node[-1]),
    )
    return record.strong + correction, correction, events


def _well_score(record, prediction):
    prediction = np.asarray(prediction, np.float64)
    error = prediction - record.truth
    base_error = record.strong - record.truth
    output = {
        "well": record.well,
        "meta_fold": record.meta_fold,
        "rows": len(error),
        "sse": float(np.sum(error * error)),
        "base_sse": float(np.sum(base_error * base_error)),
        "mae_sum": float(np.sum(np.abs(error))),
        "base_mae_sum": float(np.sum(np.abs(base_error))),
    }
    for label, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (record.distance >= low) & (record.distance < high)
        output[f"{label}_rows"] = int(mask.sum())
        output[f"{label}_sse"] = float(np.sum(error[mask] ** 2))
        output[f"{label}_base_sse"] = float(np.sum(base_error[mask] ** 2))
    return output


def _aggregate_scores(frame):
    rows = int(frame["rows"].sum())
    sse = float(frame["sse"].sum())
    base_sse = float(frame["base_sse"].sum())
    result = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "base_rmse": _rmse_from_sse(base_sse, rows),
        "rmse": _rmse_from_sse(sse, rows),
        "gain": _rmse_from_sse(base_sse, rows) - _rmse_from_sse(sse, rows),
        "mae": float(frame["mae_sum"].sum() / max(rows, 1)),
    }
    well_rmse = np.sqrt(frame["sse"].to_numpy(float) / frame["rows"].to_numpy(float))
    result.update(
        {
            "median_well_rmse": float(np.median(well_rmse)),
            "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "max_well_rmse": float(np.max(well_rmse)),
        }
    )
    for label in ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"):
        group_rows = int(frame[f"{label}_rows"].sum())
        group_sse = float(frame[f"{label}_sse"].sum())
        group_base_sse = float(frame[f"{label}_base_sse"].sum())
        result[f"{label}_rmse"] = _rmse_from_sse(group_sse, group_rows)
        result[f"{label}_gain"] = (
            _rmse_from_sse(group_base_sse, group_rows)
            - _rmse_from_sse(group_sse, group_rows)
        )
    return result


# ---------------------------------------------------------------------------
# Scan configurations, then select each held meta fold from the other four.
# ---------------------------------------------------------------------------


scan_rows = []
for config_index, config in enumerate(configs, 1):
    for record in records.values():
        prediction, correction, events = _predict_rows(record, config)
        score = _well_score(record, prediction)
        score.update(
            {
                "config_id": config.config_id,
                "arm": config.arm,
                "mean_abs_move": float(np.mean(np.abs(correction))),
                "p95_abs_move": float(np.quantile(np.abs(correction), 0.95)),
                "events": len(events),
            }
        )
        scan_rows.append(score)
    if config_index % 10 == 0 or config_index == len(configs):
        print(f"graph config scan {config_index}/{len(configs)}", flush=True)

scan = pd.DataFrame(scan_rows)
scan.to_pickle(OUT_DIR / "config_scan_bywell.pkl")


def _select_config(train_frame, allowed_arms):
    candidates = []
    for config_id, frame in train_frame.loc[train_frame["arm"].isin(allowed_arms)].groupby("config_id"):
        metrics = _aggregate_scores(frame)
        fold_gains = []
        for _, fold_frame in frame.groupby("meta_fold"):
            fold_gains.append(_aggregate_scores(fold_frame)["gain"])
        near_ok = metrics["near_0_500_gain"] >= -0.05
        safe = (
            metrics["gain"] >= 0.05
            and sum(gain > 0 for gain in fold_gains) >= 3
            and min(fold_gains) >= -0.10
            and near_ok
        )
        config = config_by_id[config_id]
        complexity = (
            int(config.marker_top_events > 0)
            + int(config.edge_source != "none")
            + 0.01 * config.marker_top_events
        )
        candidates.append(
            {
                "config_id": config_id,
                "safe": bool(safe),
                "train_gain": metrics["gain"],
                "train_far_gain": metrics["far_gt2000_gain"],
                "train_near_gain": metrics["near_0_500_gain"],
                "train_min_meta_gain": min(fold_gains),
                "complexity": complexity,
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    passing = candidate_frame.loc[candidate_frame["safe"]].copy()
    if not len(passing):
        return "strong_only", candidate_frame
    passing = passing.sort_values(
        ["train_gain", "train_far_gain", "complexity"],
        ascending=[False, False, True],
    )
    return str(passing.iloc[0]["config_id"]), candidate_frame


arm_menus = {
    "strong": {"strong"},
    "delta": {"strong", "delta"},
    "marker": {"strong", "marker"},
    "delta_marker": {"strong", "delta", "marker", "delta_marker"},
}
selection_rows = []
selected_predictions = {}
selected_controls = {}
selected_events_rows = []

for arm_name, allowed_arms in arm_menus.items():
    selected_predictions[arm_name] = {}
    selected_controls[arm_name] = {}
    for held_meta in range(1, META_FOLDS + 1):
        training_scan = scan.loc[scan["meta_fold"].ne(held_meta)]
        config_id, _ = _select_config(training_scan, allowed_arms)
        config = config_by_id[config_id]
        held_wells = [
            well for well, record in records.items() if record.meta_fold == held_meta
        ]
        held_rows = scan.loc[
            scan["config_id"].eq(config_id) & scan["meta_fold"].eq(held_meta)
        ]
        held_metrics = _aggregate_scores(held_rows)
        selection_rows.append(
            {
                "arm": arm_name,
                "held_meta_fold": held_meta,
                "selected_config": config_id,
                **held_metrics,
            }
        )
        for well in held_wells:
            record = records[well]
            prediction, correction, events = _predict_rows(record, config)
            selected_predictions[arm_name][well] = prediction
            if config.marker_top_events > 0:
                control_prediction, _, _ = _predict_rows(
                    record, config, shuffled_marker=True
                )
            else:
                control_prediction = prediction.copy()
            selected_controls[arm_name][well] = control_prediction
            for event in events:
                selected_events_rows.append(
                    {
                        "arm": arm_name,
                        "held_meta_fold": held_meta,
                        "well": well,
                        "config_id": config_id,
                        "node": event[0],
                        "node_md": event[1],
                        "marker": event[2],
                        "correction_observation": event[3],
                        "weight": event[4],
                        "event_probability": event[5],
                    }
                )

selection = pd.DataFrame(selection_rows)
selection.to_csv(OUT_DIR / "nested_selection.csv", index=False)
pd.DataFrame(selected_events_rows).to_csv(
    OUT_DIR / "selected_marker_events.csv", index=False
)


def _score_prediction_map(prediction_map):
    rows = []
    for well, prediction in prediction_map.items():
        rows.append(_well_score(records[well], prediction))
    frame = pd.DataFrame(rows)
    return _aggregate_scores(frame), frame


arm_scores = []
arm_bywell = {}
control_scores = []
for arm_name in arm_menus:
    metrics, bywell = _score_prediction_map(selected_predictions[arm_name])
    control_metrics, _ = _score_prediction_map(selected_controls[arm_name])
    arm_scores.append({"arm": arm_name, **metrics})
    control_scores.append(
        {
            "arm": arm_name,
            **control_metrics,
            "degradation_vs_identity": control_metrics["rmse"] - metrics["rmse"],
        }
    )
    arm_bywell[arm_name] = bywell
    bywell.to_csv(OUT_DIR / f"nested_{arm_name}_bywell.csv", index=False)

arm_scores = pd.DataFrame(arm_scores).sort_values("rmse")
control_scores = pd.DataFrame(control_scores).sort_values("rmse")
arm_scores.to_csv(OUT_DIR / "nested_arm_scores.csv", index=False)
control_scores.to_csv(OUT_DIR / "shuffled_marker_controls.csv", index=False)


# ---------------------------------------------------------------------------
# Exact-top corridor is a diagnostic ceiling only.
# ---------------------------------------------------------------------------


best_deployable_arm = str(arm_scores.iloc[0]["arm"])
corridor_rows = []
corridor_prediction_maps = {}
for corridor_weight in (0.10, 0.30, 1.00):
    label = f"{best_deployable_arm}+oracle_corridor_w{corridor_weight:.2f}"
    corridor_prediction_maps[label] = {}
    for held_meta in range(1, META_FOLDS + 1):
        selected_config_id = str(
            selection.loc[
                selection["arm"].eq(best_deployable_arm)
                & selection["held_meta_fold"].eq(held_meta),
                "selected_config",
            ].iloc[0]
        )
        base_config = config_by_id[selected_config_id]
        corridor_config = GraphConfig(
            **{
                **base_config.__dict__,
                "corridor_weight": corridor_weight,
            }
        )
        for well, record in records.items():
            if record.meta_fold != held_meta:
                continue
            prediction, _, _ = _predict_rows(
                record, corridor_config, corridor=True
            )
            corridor_prediction_maps[label][well] = prediction
    metrics, _ = _score_prediction_map(corridor_prediction_maps[label])
    corridor_rows.append(
        {
            "variant": label,
            "oracle_only": True,
            "corridor_weight": corridor_weight,
            **metrics,
        }
    )

corridor_scores = pd.DataFrame(corridor_rows).sort_values("rmse")
corridor_scores.to_csv(OUT_DIR / "oracle_corridor_scores.csv", index=False)


# ---------------------------------------------------------------------------
# Final gates and compact row-level handoff.
# ---------------------------------------------------------------------------


strong_metrics = arm_scores.loc[arm_scores["arm"].eq("strong")].iloc[0].to_dict()
best_row = arm_scores.iloc[0].to_dict()
best_arm = str(best_row["arm"])
best_selection = selection.loc[selection["arm"].eq(best_arm)]
positive_meta = int((best_selection["gain"] > 0).sum())
minimum_meta_gain = float(best_selection["gain"].min())
best_control = control_scores.loc[control_scores["arm"].eq(best_arm)].iloc[0].to_dict()

strong_exact_max_abs = 0.0
for well, prediction in selected_predictions["strong"].items():
    strong_exact_max_abs = max(
        strong_exact_max_abs,
        float(np.max(np.abs(prediction - records[well].strong))),
    )

strict_pass = bool(
    strong_exact_max_abs <= 1e-12
    and float(best_row["gain"]) >= 0.30
    and float(best_row["far_gt2000_gain"]) >= 0.50
    and float(best_row["near_0_500_gain"]) >= -0.05
    and positive_meta >= 4
    and minimum_meta_gain >= -0.10
    and (
        best_arm in {"strong", "delta"}
        or float(best_control["degradation_vs_identity"]) >= 0.10
    )
)

row_payload = {
    "well": [],
    "row_idx": [],
    "truth": [],
    "strong": [],
    "prediction": [],
    "meta_fold": [],
}
for well in sorted(records):
    record = records[well]
    row_payload["well"].append(np.full(len(record.truth), well, dtype="U16"))
    row_payload["row_idx"].append(record.row_index.astype(np.int32))
    row_payload["truth"].append(record.truth.astype(np.float32))
    row_payload["strong"].append(record.strong.astype(np.float32))
    row_payload["prediction"].append(
        selected_predictions[best_arm][well].astype(np.float32)
    )
    row_payload["meta_fold"].append(
        np.full(len(record.truth), record.meta_fold, dtype=np.int8)
    )
np.savez_compressed(
    OUT_DIR / "best_nested_fold2_predictions.npz",
    **{key: np.concatenate(value) for key, value in row_payload.items()},
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_nested_strong_prior_absolute_edge_graph_gateway",
    "centre": "fold-clean strong OOF, used only as a robust graph prior",
    "protocol": {
        "validation_fold": VALID_FOLD,
        "meta_folds": META_FOLDS,
        "selection": "leave-one-complete-well-meta-fold-out",
        "fold1_touched": False,
        "test_time_search": 0,
    },
    "sources": {
        "pool": str(POOL_PATH),
        "folds": str(FOLD_PATH),
        "marker_crossfit": str(MARKER_PATH),
        "delta_s_edges": str(EDGE_PATH),
    },
    "guards": {
        "wells": len(records),
        "rows": len(pool),
        "strong_only_max_abs_reproduction_error": strong_exact_max_abs,
        "configs": len(configs),
        "marker_wells": len(marker_wells),
        "edge_wells": len(edge_wells),
    },
    "arm_scores": arm_scores.to_dict(orient="records"),
    "shuffled_marker_controls": control_scores.to_dict(orient="records"),
    "best_deployable": {
        **best_row,
        "positive_meta_folds": positive_meta,
        "minimum_meta_fold_gain": minimum_meta_gain,
    },
    "oracle_corridor": {
        "uses_validation Geology labels": True,
        "participates_in_selection": False,
        "scores": corridor_scores.to_dict(orient="records"),
    },
    "gates": {
        "overall_gain_ge": 0.30,
        "far_gt2000_gain_ge": 0.50,
        "near_0_500_gain_ge": -0.05,
        "positive_meta_folds_ge": 4,
        "minimum_meta_fold_gain_ge": -0.10,
        "marker_shuffle_degradation_ge": 0.10,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train outer-fold marker/corridor stack and evaluate untouched Fold1"
        if strict_pass
        else "FAIL -> absolute edges are not reliable enough even with a strong prior"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nNESTED SELECTION")
print(selection.to_string(index=False), flush=True)
print("\nDEPLOYABLE ARM SCORES")
print(arm_scores.to_string(index=False), flush=True)
print("\nSHUFFLED MARKER CONTROLS")
print(control_scores.to_string(index=False), flush=True)
print("\nORACLE CORRIDOR (NOT DEPLOYABLE)")
print(corridor_scores.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
