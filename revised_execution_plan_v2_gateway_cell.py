"""Artifact-first Day 1-5 gateway for the revised execution plan V2.

Run as one AutoDL notebook cell after the historical OOF artifacts have been
restored under ``/root/autodl-tmp/oof_exports``.  This cell does not train a
new model.  It inventories lineage, assembles only row-aligned OOF endpoints,
runs one-degree-of-freedom nested portfolio tests, measures the whole-well
router oracle, and records the status of G3/G4/T1 from surviving artifacts.

The hidden-LB production endpoint is 6.792 (CP12 -> Gold -> full-progress
geometric Neighbor).  Because hidden predictions/truth are unavailable, the
cell never labels a local surrogate as an exact reproduction of that endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STARTED = time.time()
EXPERIMENT = "revised_execution_plan_v2_gateway_v0"
CURRENT_SAFE_LB = 6.792
LEGACY_SAFE_LB = 6.882
N_FOLDS = 5

D_GAIN_GATE = 0.05
C_GAIN_GATE = 0.05
P_GAIN_GATE = 0.10
P_ADAPTIVE_GAIN_GATE = 0.15
G2_ORACLE_GAIN_GATE = 0.20
G2_TAIL_RECOVERY_GATE = 0.15
G3_SPEARMAN_GATE = 0.25
G3_AUC_GATE = 0.60

NEIGHBOR_WEIGHT = 0.25
NEIGHBOR_RAW_CAP = 24.0
MIN_ABS_DIRECTION_DOT = 0.70
MAX_PORTFOLIO_ENDPOINTS = 12
MAX_SEQUENTIAL_STEPS = 3


def _first_existing(paths):
    return next((Path(path) for path in paths if Path(path).exists()), None)


OOF_ROOT = _first_existing(
    [
        "/root/autodl-tmp/oof_exports",
        "/kaggle/working/oof_exports",
        Path.cwd() / "oof_exports",
    ]
)
DATA_ROOT = _first_existing(
    [
        "/root/autodl-tmp/rogii-wellbore-geology-prediction",
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
        Path.cwd(),
    ]
)
if OOF_ROOT is None:
    raise FileNotFoundError("Cannot locate oof_exports")
if DATA_ROOT is None or not (DATA_ROOT / "train").exists():
    raise FileNotFoundError("Cannot locate competition train directory")

OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

POOL_PATH = _first_existing(
    [
        OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
        OOF_ROOT / "candidate_pool_full_merged_oof.csv",
    ]
)
if POOL_PATH is None:
    raise FileNotFoundError("Candidate pool is required for row ownership")

FOLD_SOURCE = _first_existing(
    [
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
        OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "well_metadata.csv",
    ]
)
if FOLD_SOURCE is None:
    raise FileNotFoundError("No stable fold metadata found")

print(
    {
        "experiment": EXPERIMENT,
        "data_root": str(DATA_ROOT),
        "oof_root": str(OOF_ROOT),
        "pool": str(POOL_PATH),
        "fold_source": str(FOLD_SOURCE),
        "current_safe_lb": CURRENT_SAFE_LB,
    },
    flush=True,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    valid = np.isfinite(y) & np.isfinite(prediction)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((prediction[valid] - y[valid]) ** 2)))


def _error_cosine(y, first, second):
    first_error = np.asarray(first, np.float64) - np.asarray(y, np.float64)
    second_error = np.asarray(second, np.float64) - np.asarray(y, np.float64)
    valid = np.isfinite(first_error) & np.isfinite(second_error)
    if not valid.any():
        return np.nan
    denominator = np.linalg.norm(first_error[valid]) * np.linalg.norm(second_error[valid])
    return float(np.dot(first_error[valid], second_error[valid]) / max(denominator, 1e-12))


def _partial_sha256(path, block=1024 * 1024):
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(block))
        if size > block:
            handle.seek(max(0, size - block))
            digest.update(handle.read(block))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _well_metrics(frame, prediction, truth):
    work = pd.DataFrame(
        {
            "well": frame["well"].to_numpy(),
            "rows": 1,
            "se": (np.asarray(prediction, np.float64) - truth) ** 2,
        }
    )
    grouped = work.groupby("well", sort=False).agg(rows=("rows", "sum"), sse=("se", "sum"))
    grouped["rmse"] = np.sqrt(grouped["sse"] / grouped["rows"])
    return grouped


def _tail_metrics(frame, truth, safe, prediction):
    safe_well = _well_metrics(frame, safe, truth).rename(
        columns={"sse": "safe_sse", "rmse": "safe_rmse"}
    )
    new_well = _well_metrics(frame, prediction, truth).rename(
        columns={"sse": "new_sse", "rmse": "new_rmse"}
    )
    merged = safe_well.join(new_well[["new_sse", "new_rmse"]], how="inner")
    worst_count = max(1, int(math.ceil(0.20 * len(merged))))
    worst = merged.nlargest(worst_count, "safe_sse")
    p95_delta = float(merged["new_rmse"].quantile(0.95) - merged["safe_rmse"].quantile(0.95))
    worst_safe = float(np.sqrt(worst["safe_sse"].sum() / worst["rows"].sum()))
    worst_new = float(np.sqrt(worst["new_sse"].sum() / worst["rows"].sum()))
    return {
        "improved_well_fraction": float((merged["new_rmse"] < merged["safe_rmse"]).mean()),
        "median_well_delta": float((merged["new_rmse"] - merged["safe_rmse"]).median()),
        "p95_well_delta": p95_delta,
        "worst20_safe_rmse": worst_safe,
        "worst20_new_rmse": worst_new,
        "worst20_gain": worst_safe - worst_new,
    }, merged


# ---------------------------------------------------------------------------
# 1. Artifact census
# ---------------------------------------------------------------------------
inventory_rows = []
for path in OOF_ROOT.rglob("*"):
    if not path.is_file() or OUT_DIR in path.parents:
        continue
    relative = path.relative_to(OOF_ROOT)
    text = str(relative).replace("\\", "/")
    fold_match = re.search(r"(?:fold|f)[_-]?(\d+)", text, flags=re.I)
    seed_match = re.search(r"seed[_-]?(\d+)", text, flags=re.I)
    suffix = path.suffix.lower()
    kind = (
        "checkpoint" if suffix in {".pt", ".pth", ".joblib", ".cbm", ".json"} and "model" in text.lower()
        else "row_oof" if any(token in path.name.lower() for token in ("oof", "prediction")) and suffix in {".pkl", ".parquet", ".npz", ".csv"}
        else "verdict" if path.name.lower() in {"verdict.json", "result.json"}
        else "table" if suffix in {".csv", ".parquet", ".pkl"}
        else "array" if suffix in {".npz", ".npy"}
        else "other"
    )
    stat = path.stat()
    should_hash = kind in {"checkpoint", "row_oof", "verdict"}
    inventory_rows.append(
        {
            "experiment": relative.parts[0] if relative.parts else "root",
            "relative_path": text,
            "kind": kind,
            "suffix": suffix,
            "size_mb": stat.st_size / 1024**2,
            "mtime_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
            "fold_hint": int(fold_match.group(1)) if fold_match else np.nan,
            "seed_hint": int(seed_match.group(1)) if seed_match else np.nan,
            "is_checkpoint": kind == "checkpoint",
            "is_oof_hint": kind == "row_oof",
            "hash_mode": "head_tail_size" if should_hash else "metadata_only",
            "partial_sha256": _partial_sha256(path) if should_hash else None,
        }
    )

inventory = pd.DataFrame(inventory_rows).sort_values(
    ["experiment", "relative_path"], kind="stable"
)
inventory.to_csv(OUT_DIR / "artifact_inventory.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Canonical row keys, truth, predictions, and fold ownership
# ---------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
prediction_allowlist = [
    "strong",
    "strong_bias",
    "cp12",
    "sp45",
    "fleongg",
    "ridge",
    "selector",
    "cp_pack",
    "cp_single12",
]
key_columns = [column for column in ("id", "well", "row_idx", "tvt_true") if column in header]
prediction_columns = [column for column in prediction_allowlist if column in header]
if "tvt_true" not in key_columns:
    raise RuntimeError("Candidate pool lacks tvt_true")

pool = pd.read_csv(
    POOL_PATH,
    usecols=key_columns + prediction_columns,
    dtype={"id": "string", "well": "string"},
)
if "id" not in pool:
    if not {"well", "row_idx"}.issubset(pool.columns):
        raise RuntimeError("Cannot construct canonical row id")
    pool["id"] = pool["well"].astype(str) + "_" + pool["row_idx"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].astype(str).str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["id"] = pool["id"].astype(str)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if pool["id"].duplicated().any():
    raise RuntimeError("Duplicate canonical row ids")

truth = pool["tvt_true"].to_numpy(np.float64)
if not np.isfinite(truth).all():
    raise RuntimeError("Non-finite truth in candidate pool")

meta = pd.read_csv(FOLD_SOURCE, dtype={"well": str})
if not {"well", "fold"}.issubset(meta.columns):
    raise RuntimeError("Fold source lacks well/fold")
fold_map = meta.groupby("well")["fold"].first().astype(int).to_dict()
pool["fold"] = pool["well"].map(fold_map)
if pool["fold"].isna().any():
    missing = sorted(pool.loc[pool["fold"].isna(), "well"].unique())
    raise RuntimeError(f"Missing fold ownership for {len(missing)} wells")
pool["fold"] = pool["fold"].astype(np.int8)

fold_ownership = (
    pool.groupby(["well", "fold"], as_index=False)
    .agg(rows=("id", "size"), row_idx_min=("row_idx", "min"), row_idx_max=("row_idx", "max"))
)
fold_ownership["owner_rule"] = "all pseudo-PS states remain inside the source well fold"
fold_ownership["source"] = str(FOLD_SOURCE)
fold_ownership.to_csv(OUT_DIR / "fold_ownership.csv", index=False)

predictions = {}
prediction_sources = {}
for column in prediction_columns:
    values = pool[column].to_numpy(np.float64)
    if np.isfinite(values).all():
        predictions[column] = values.astype(np.float32)
        prediction_sources[column] = f"pool:{POOL_PATH.name}:{column}"


def _eligible_neighbor_rows(source_dir):
    selected_path = source_dir / "selected_neighbors.csv"
    if not selected_path.exists():
        return np.zeros(len(pool), bool)
    selected = pd.read_csv(selected_path, dtype={"well": str}).set_index("well")
    direction_column = "abs_direction_dot" if "abs_direction_dot" in selected else "direction_dot"
    direction = selected[direction_column].abs()
    eligible_well = (
        (selected["median_distance"] <= 600.0)
        & (selected["p90_distance"] <= 1200.0)
        & (selected["anchor_distance_full"] <= 1200.0)
        & (direction >= MIN_ABS_DIRECTION_DOT)
    )
    return pool["well"].map(eligible_well.to_dict()).fillna(False).to_numpy(bool)


neighbor_source = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
if neighbor_source.exists() and "cp12" in predictions:
    transfer_path = neighbor_source / "neighbor_transfer_oof.npz"
    if transfer_path.exists():
        transfer = np.load(transfer_path, allow_pickle=False)
        eligible = _eligible_neighbor_rows(neighbor_source)
        for method, label in (
            ("structure_progress", "neighbor_frozen_cp12"),
            ("structure_xy", "neighbor_xy_cp12"),
        ):
            if method not in transfer:
                continue
            candidate = transfer[method].astype(np.float64)
            if len(candidate) != len(pool):
                continue
            base = predictions["cp12"].astype(np.float64)
            move = np.zeros(len(pool), np.float64)
            valid = eligible & np.isfinite(candidate)
            move[valid] = NEIGHBOR_WEIGHT * np.clip(
                candidate[valid] - base[valid], -NEIGHBOR_RAW_CAP, NEIGHBOR_RAW_CAP
            )
            predictions[label] = (base + move).astype(np.float32)
            prediction_sources[label] = f"{transfer_path.name}:{method};base=cp12"

geometry_dir = OOF_ROOT / "neighbor_fullcurve_projection_direction_v1"
geometry_path = geometry_dir / "primary_candidate_oof.npz"
if geometry_path.exists() and "cp12" in predictions:
    candidate = np.load(geometry_path, allow_pickle=False)["candidate"].astype(np.float64)
    if len(candidate) == len(pool):
        eligible = _eligible_neighbor_rows(neighbor_source)
        base = predictions["cp12"].astype(np.float64)
        move = np.zeros(len(pool), np.float64)
        valid = eligible & np.isfinite(candidate)
        move[valid] = NEIGHBOR_WEIGHT * np.clip(
            candidate[valid] - base[valid], -NEIGHBOR_RAW_CAP, NEIGHBOR_RAW_CAP
        )
        predictions["neighbor_geom_cp12"] = (base + move).astype(np.float32)
        prediction_sources["neighbor_geom_cp12"] = (
            f"{geometry_path.name}:full_progress_geom;base=cp12"
        )


def _merge_keyed_pickle(path, candidates):
    try:
        frame = pd.read_pickle(path)
    except Exception:
        return
    if "id" not in frame:
        return
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    for source_column, target_name in candidates:
        if source_column not in frame:
            continue
        joined = pool[["id"]].merge(
            frame[["id", source_column]], on="id", how="left", validate="one_to_one"
        )
        values = joined[source_column].to_numpy(np.float64)
        if len(values) == len(pool) and np.isfinite(values).all():
            predictions[target_name] = values.astype(np.float32)
            prediction_sources[target_name] = str(path)


strict_lgb = OOF_ROOT / "multicut_k5_strict_outer_spatial_v1" / "strict_outer_spatial_oof.pkl"
if strict_lgb.exists():
    _merge_keyed_pickle(strict_lgb, [("pred_spatial", "strict_spatial_lgb")])

strict_three = OOF_ROOT / "multicut_k5_strict_spatial_three_model_v1" / "three_model_oof.pkl"
if strict_three.exists():
    _merge_keyed_pickle(
        strict_three,
        [
            ("pred_lgb", "strict_spatial_lgb_3m"),
            ("pred_cat", "strict_spatial_cat"),
            ("pred_xgb", "strict_spatial_xgb"),
            ("pred_equal_3", "strict_spatial_equal3"),
        ],
    )

selector_path = OOF_ROOT / "dsse_shape_selector_nested_v0" / "selector_oof.npz"
if selector_path.exists():
    selector = np.load(selector_path, allow_pickle=False)["prediction"].astype(np.float64)
    if len(selector) == len(pool) and np.isfinite(selector).all():
        predictions["dsse_selector"] = selector.astype(np.float32)
        prediction_sources["dsse_selector"] = str(selector_path)

if len(predictions) > MAX_PORTFOLIO_ENDPOINTS:
    priority = [
        "neighbor_geom_cp12",
        "neighbor_frozen_cp12",
        "cp12",
        "strong_bias",
        "strong",
        "strict_spatial_equal3",
        "strict_spatial_lgb",
        "strict_spatial_cat",
        "strict_spatial_xgb",
        "sp45",
        "fleongg",
        "dsse_selector",
    ]
    keep = [name for name in priority if name in predictions][:MAX_PORTFOLIO_ENDPOINTS]
    predictions = {name: predictions[name] for name in keep}
    prediction_sources = {name: prediction_sources[name] for name in keep}

if not predictions:
    raise RuntimeError("No complete row-level OOF predictions were discovered")

local_safe_name = (
    "neighbor_geom_cp12"
    if "neighbor_geom_cp12" in predictions
    else "neighbor_frozen_cp12"
    if "neighbor_frozen_cp12" in predictions
    else "cp12"
    if "cp12" in predictions
    else min(predictions, key=lambda name: _rmse(truth, predictions[name]))
)
local_safe = predictions[local_safe_name].astype(np.float64)
current_mechanism_oof_available = "neighbor_geom_cp12" in predictions


# ---------------------------------------------------------------------------
# 3. Persist the canonical residual matrix and model-level diagnostics
# ---------------------------------------------------------------------------
row_matrix = pool[["id", "well", "row_idx", "fold", "tvt_true"]].copy()
for name, values in predictions.items():
    row_matrix[f"pred__{name}"] = values.astype(np.float32)
    row_matrix[f"err__{name}"] = (values.astype(np.float64) - truth).astype(np.float32)

residual_path = OUT_DIR / "prediction_residual_matrix.parquet"
try:
    row_matrix.to_parquet(residual_path, index=False, compression="zstd")
except Exception as exc:
    residual_path = OUT_DIR / "prediction_residual_matrix.pkl"
    row_matrix.to_pickle(residual_path)
    print("Parquet unavailable; wrote pickle:", repr(exc), flush=True)

model_rows = []
per_well_parts = []
for name, values in predictions.items():
    values64 = values.astype(np.float64)
    bywell = _well_metrics(pool, values64, truth).reset_index()
    bywell["model"] = name
    per_well_parts.append(bywell)
    model_rows.append(
        {
            "model": name,
            "rmse": _rmse(truth, values64),
            "median_well_rmse": float(bywell["rmse"].median()),
            "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
            "error_cosine_vs_local_safe": _error_cosine(truth, local_safe, values64),
            "source": prediction_sources[name],
        }
    )
model_scores = pd.DataFrame(model_rows).sort_values("rmse")
model_scores.to_csv(OUT_DIR / "model_scores.csv", index=False)
per_well_sse = pd.concat(per_well_parts, ignore_index=True)
per_well_sse.to_csv(OUT_DIR / "prediction_per_well_sse.csv", index=False)

names = list(predictions)
row_cosine = pd.DataFrame(index=names, columns=names, dtype=float)
well_sse_corr = pd.DataFrame(index=names, columns=names, dtype=float)
well_sse_wide = per_well_sse.pivot(index="well", columns="model", values="sse")
for first in names:
    for second in names:
        row_cosine.loc[first, second] = _error_cosine(
            truth, predictions[first], predictions[second]
        )
        well_sse_corr.loc[first, second] = well_sse_wide[first].corr(
            well_sse_wide[second], method="spearman"
        )
row_cosine.to_csv(OUT_DIR / "row_error_cosine.csv")
well_sse_corr.to_csv(OUT_DIR / "well_sse_spearman.csv")


# ---------------------------------------------------------------------------
# 4. G1: exact one-degree-of-freedom nested endpoint tests
# ---------------------------------------------------------------------------
fold_values = pool["fold"].to_numpy(np.int8)


def _fit_dose(y, current, endpoint, mask):
    error = current[mask] - y[mask]
    delta = endpoint[mask] - current[mask]
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(error, delta) / denominator, 0.0, 1.0))


def _nested_endpoint(current, endpoint, label):
    output = np.empty(len(truth), np.float64)
    fold_rows = []
    for held_fold in range(1, N_FOLDS + 1):
        train_mask = fold_values != held_fold
        valid_mask = fold_values == held_fold
        dose = _fit_dose(truth, current, endpoint, train_mask)
        output[valid_mask] = current[valid_mask] + dose * (
            endpoint[valid_mask] - current[valid_mask]
        )
        fold_rows.append(
            {
                "endpoint": label,
                "fold": held_fold,
                "dose": dose,
                "current_rmse": _rmse(truth[valid_mask], current[valid_mask]),
                "prediction_rmse": _rmse(truth[valid_mask], output[valid_mask]),
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame["gain"] = fold_frame["current_rmse"] - fold_frame["prediction_rmse"]
    tail, _ = _tail_metrics(pool, truth, current, output)
    summary = {
        "endpoint": label,
        "current_rmse": _rmse(truth, current),
        "prediction_rmse": _rmse(truth, output),
        "gain": _rmse(truth, current) - _rmse(truth, output),
        "mean_dose": float(fold_frame["dose"].mean()),
        "min_fold_gain": float(fold_frame["gain"].min()),
        "positive_folds": int((fold_frame["gain"] > 0).sum()),
        "error_cosine_vs_current": _error_cosine(truth, current, endpoint),
        **tail,
    }
    summary["d_gate_pass"] = bool(
        summary["gain"] >= D_GAIN_GATE
        and summary["p95_well_delta"] <= 0.0
        and summary["worst20_gain"] >= 0.0
    )
    return output, summary, fold_frame


g1_summaries = []
g1_folds = []
g1_predictions = {}
for name, endpoint in predictions.items():
    if name == local_safe_name:
        continue
    output, summary, folds = _nested_endpoint(
        local_safe, endpoint.astype(np.float64), name
    )
    g1_summaries.append(summary)
    g1_folds.append(folds)
    g1_predictions[name] = output

g1_summary = pd.DataFrame(g1_summaries).sort_values(
    ["gain", "min_fold_gain"], ascending=False
)
g1_fold_report = pd.concat(g1_folds, ignore_index=True) if g1_folds else pd.DataFrame()
g1_summary.to_csv(OUT_DIR / "g1_nested_endpoints.csv", index=False)
g1_fold_report.to_csv(OUT_DIR / "g1_nested_folds.csv", index=False)

# Sequential development audit: at most three endpoints, each fitted with one
# new scalar dose. This is not a production portfolio or a confirmation split.
sequential_rows = []
current = local_safe.copy()
remaining = g1_summary["endpoint"].tolist()
for step in range(1, min(MAX_SEQUENTIAL_STEPS, len(remaining)) + 1):
    candidates = []
    for name in remaining:
        output, summary, folds = _nested_endpoint(
            current, predictions[name].astype(np.float64), name
        )
        candidates.append((summary["gain"], name, output, summary, folds))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, selected_name, selected_output, selected_summary, selected_folds = candidates[0]
    sequential_rows.append(
        {
            "step": step,
            "selected_endpoint": selected_name,
            **selected_summary,
        }
    )
    if selected_summary["gain"] < D_GAIN_GATE:
        break
    current = selected_output
    remaining.remove(selected_name)

g1_sequential = pd.DataFrame(sequential_rows)
g1_sequential.to_csv(OUT_DIR / "g1_sequential.csv", index=False)


# ---------------------------------------------------------------------------
# 5. G2: whole-well action oracle over a small legal endpoint basket
# ---------------------------------------------------------------------------
expert_names = [local_safe_name]
expert_names.extend(
    name
    for name in g1_summary.head(7)["endpoint"].tolist()
    if name not in expert_names
)
expert_predictions = {name: predictions[name].astype(np.float64) for name in expert_names}

well_indices = pool.groupby("well", sort=False).indices
oracle_prediction = local_safe.copy()
g2_rows = []
for well, indices in well_indices.items():
    indices = np.asarray(indices, np.int64)
    safe_sse = float(np.sum((local_safe[indices] - truth[indices]) ** 2))
    candidate_sse = {
        name: float(np.sum((values[indices] - truth[indices]) ** 2))
        for name, values in expert_predictions.items()
    }
    best_name = min(candidate_sse, key=candidate_sse.get)
    oracle_prediction[indices] = expert_predictions[best_name][indices]
    g2_rows.append(
        {
            "well": well,
            "rows": len(indices),
            "safe_sse": safe_sse,
            "oracle_sse": candidate_sse[best_name],
            "delta_sse": safe_sse - candidate_sse[best_name],
            "oracle_endpoint": best_name,
        }
    )

g2_bywell = pd.DataFrame(g2_rows)
g2_bywell.to_csv(OUT_DIR / "g2_router_oracle_bywell.csv", index=False)
safe_rmse = _rmse(truth, local_safe)
oracle_rmse = _rmse(truth, oracle_prediction)
worst_count = max(1, int(math.ceil(0.20 * len(g2_bywell))))
worst = g2_bywell.nlargest(worst_count, "safe_sse")
tail_recovery = float(
    worst["delta_sse"].sum() / max(worst["safe_sse"].sum(), 1e-12)
)
g2_summary = {
    "local_safe": local_safe_name,
    "experts": expert_names,
    "safe_rmse": safe_rmse,
    "oracle_rmse": oracle_rmse,
    "oracle_gain": safe_rmse - oracle_rmse,
    "tail_sse_recovery_fraction": tail_recovery,
    "positive_well_fraction": float((g2_bywell["delta_sse"] > 0).mean()),
    "strict_pass": bool(
        safe_rmse - oracle_rmse >= G2_ORACLE_GAIN_GATE
        and tail_recovery >= G2_TAIL_RECOVERY_GATE
    ),
}
_write_json(OUT_DIR / "g2_router_oracle.json", g2_summary)


# ---------------------------------------------------------------------------
# 6. G3/G4/T1 and closed-branch evidence from surviving verdicts
# ---------------------------------------------------------------------------
evidence_specs = [
    (
        "G3a_prefix_adapter",
        OOF_ROOT / "tvt_corrnet_prefix_meta_adapter_v0" / "verdict.json",
        "existing weak online adaptation evidence",
    ),
    (
        "G3a_forward_chaining",
        OOF_ROOT / "block_paired_gold_forward_chaining" / "verdict.json",
        "existing paired cutback evidence",
    ),
    (
        "G4_S3_mode_audit",
        OOF_ROOT / "s3_s5_zero_retrain_repair_audit_v1" / "verdict.json",
        "iteration and reachability audit",
    ),
    (
        "P3_absolute_energy",
        OOF_ROOT / "fullwell_absolute_energy_e1_onefold_v0" / "verdict.json",
        "menu-independent energy result",
    ),
    (
        "G2_learned_selector",
        OOF_ROOT / "dsse_shape_selector_nested_v0" / "verdict.json",
        "learned whole-well action selector",
    ),
]
evidence_rows = []
for gateway, path, meaning in evidence_specs:
    verdict = _read_json(path) if path.exists() else None
    evidence_rows.append(
        {
            "gateway": gateway,
            "artifact": str(path),
            "exists": bool(path.exists()),
            "meaning": meaning,
            "reported_strict_pass": (
                verdict.get("strict_pass") if isinstance(verdict, dict) else None
            ),
            "reported_next": (
                verdict.get("next") if isinstance(verdict, dict) else None
            ),
            "verdict_json": (
                json.dumps(verdict, ensure_ascii=False, default=_json_default)
                if verdict is not None
                else None
            ),
        }
    )
gateway_evidence = pd.DataFrame(evidence_rows)
gateway_evidence.to_csv(OUT_DIR / "existing_gateway_evidence.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Preregistration, reproduction note, and Day-5 decision
# ---------------------------------------------------------------------------
preregistration = f"""experiment: {EXPERIMENT}
production_baseline:
  lineage: CP12 -> Gold -> full-progress geometric Neighbor
  public_lb: {CURRENT_SAFE_LB}
  exact_hidden_oof_available: false
local_development_baseline:
  name: {local_safe_name}
  rmse: {safe_rmse:.9f}
  warning: local surrogate only; Gold-after-CP12 hidden endpoint is unavailable
gates:
  D:
    nested_gain_min: {D_GAIN_GATE}
    p95_well_delta_max: 0.0
    worst20_gain_min: 0.0
  C:
    frozen_new_split_gain_min: {C_GAIN_GATE}
    retain_development_gain_fraction_min: 0.50
  P:
    joint_gain_min: {P_GAIN_GATE}
    adaptive_portfolio_gain_min: {P_ADAPTIVE_GAIN_GATE}
    positive_folds_min: 4
    improved_well_fraction_min: 0.55
  G2:
    whole_well_oracle_gain_min: {G2_ORACLE_GAIN_GATE}
    tail_sse_recovery_fraction_min: {G2_TAIL_RECOVERY_GATE}
  G3a:
    spearman_min: {G3_SPEARMAN_GATE}
    auc_min: {G3_AUC_GATE}
confirmation_split:
  status: not_consumed
  rule: freeze one D-pass recipe before selecting a never-used well split
negative_controls:
  - wrong-well prefix
  - shuffled typewell
  - no-signal endpoint
"""
(OUT_DIR / "gateway_preregistration.yaml").write_text(preregistration, encoding="utf-8")

safe_note = f"""# Safe Endpoint Reproduction Status

## Production endpoint

- Current public LB: **{CURRENT_SAFE_LB}**
- Lineage: `CP12 -> Gold -> full-progress geometric Neighbor`
- Legacy frozen-neighbor LB: `{LEGACY_SAFE_LB}`
- Hidden truth and the exact hidden endpoint vector are unavailable by design.

## Local OOF status

- Canonical pool: `{POOL_PATH}`
- Fold ownership: `{FOLD_SOURCE}`
- Local development surrogate: `{local_safe_name}`
- Surrogate pooled OOF RMSE: `{safe_rmse:.9f}`

The surrogate is used only for D-stage discovery. It is not called a numerical
reproduction of LB {CURRENT_SAFE_LB}, because the Gold-after-CP12 hidden vector
does not exist locally. Any D-pass challenger must be frozen and rerun on a new
confirmation split before it can be compared with the production endpoint.

## Raw-to-submission production order

1. Produce the upstream blended base.
2. Apply CP12.
3. Apply Gold.
4. Snapshot `submission_before_neighbor.csv`.
5. Apply full-progress geometric Neighbor with weight 0.25, raw cap 24 ft and
   applied cap 6 ft.
6. Preserve the independent 6.792 safe submission.
"""
(OUT_DIR / "safe_6792_reproduction.md").write_text(safe_note, encoding="utf-8")

g1_passes = (
    g1_summary.loc[g1_summary["d_gate_pass"], "endpoint"].tolist()
    if len(g1_summary)
    else []
)
existing_e1 = _read_json(
    OOF_ROOT / "fullwell_absolute_energy_e1_onefold_v0" / "verdict.json"
)
e1_failed = isinstance(existing_e1, dict) and not bool(existing_e1.get("strict_pass", False))

if not current_mechanism_oof_available:
    decision = "INCOMPLETE_BASELINE"
    selected_challenger = None
    next_action = (
        "Rebuild neighbor_fullcurve_projection_direction_v1/primary_candidate_oof.npz "
        "and rerun this gateway before consuming a confirmation split."
    )
elif g1_passes:
    decision = "GO-A"
    selected_challenger = g1_passes[0]
    next_action = "Freeze the top one-dof endpoint and run one untouched confirmation split."
elif g2_summary["strict_pass"]:
    decision = "G2_ORACLE_ONLY"
    selected_challenger = None
    next_action = "Oracle exists but no deployable action model passed; do not train a router without new evidence."
else:
    decision = "FALLBACK"
    selected_challenger = None
    next_action = "No short D-gate winner; preserve 6.792 and return research capacity to the native-resolution single-well model."

decision_record = {
    "experiment": EXPERIMENT,
    "production_safe_lb": CURRENT_SAFE_LB,
    "production_safe_lineage": "CP12 -> Gold -> full-progress geometric Neighbor",
    "local_safe_surrogate": local_safe_name,
    "local_safe_rmse": safe_rmse,
    "current_mechanism_oof_available": current_mechanism_oof_available,
    "g1_d_gate_endpoints": g1_passes,
    "g2": g2_summary,
    "e1_already_failed": e1_failed,
    "day5_decision": decision,
    "selected_short_challenger": selected_challenger,
    "next": next_action,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR / "verdict.json", decision_record)

decision_log = f"""# Revised V2 Decision Log

## Frozen facts

- Production safe endpoint is LB {CURRENT_SAFE_LB}, not the PDF's older {LEGACY_SAFE_LB} endpoint.
- Exact hidden residuals are unavailable; `{local_safe_name}` is only a local OOF surrogate.
- Absolute-energy E1 is already closed: `{e1_failed}`.

## G1

- D-gate endpoints: `{g1_passes}`
- The detailed nested fold table is `g1_nested_folds.csv`.
- Sequential results are development diagnostics, not additive score claims.

## G2

- Whole-well oracle gain: `{g2_summary['oracle_gain']:.6f}`
- Tail SSE recovery: `{g2_summary['tail_sse_recovery_fraction']:.6f}`
- Oracle gate: `{g2_summary['strict_pass']}`
- Oracle success does not authorize a learned router by itself.

## Day-5 decision

- Decision: **{decision}**
- Selected challenger: `{selected_challenger}`
- Next action: {next_action}

## Prohibited reopenings

- E1 absolute energy, Set96/listwise selector and dSSE selector.
- S-space fixed-axis RankSDF/CRF recipes already rejected by coverage and OOF.
- Neighbor multi-k/local-plane/jitter and multidimensional LB response fitting.
"""
(OUT_DIR / "decision_log.md").write_text(decision_log, encoding="utf-8")

print("\nMODEL SCORES")
print(model_scores.to_string(index=False))
print("\nG1 NESTED ENDPOINTS")
print(g1_summary.to_string(index=False))
print("\nG1 SEQUENTIAL")
print(g1_sequential.to_string(index=False))
print("\nG2 ROUTER ORACLE")
print(json.dumps(g2_summary, indent=2, ensure_ascii=False, default=_json_default))
print("\nDAY-5 VERDICT")
print(json.dumps(decision_record, indent=2, ensure_ascii=False, default=_json_default))
print("saved:", OUT_DIR)
