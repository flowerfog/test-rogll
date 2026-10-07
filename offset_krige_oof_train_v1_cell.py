"""Train and validate a per-well spatial residual lookup.

This reproduces the final post-processing cell from rogii-kim-om020.ipynb,
then runs a strictly nested variant search.  The target is one scalar per well:

    b_w = robust_location(tvt_true - honest_oof_prediction)

Outer validation wells are never present in the spatial reference lookup.  The
script writes one lookup/config pair per available base and promotes a single
pair only when the nested OOF gate passes.

Run as one AutoDL notebook cell, or as a Python script.  It only consumes
existing OOF artifacts and the original train horizontal-well CSV files.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "offset_krige_oof_train_v1"

POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
# Add an exact final-chain OOF column here if one exists in your pool.  A lookup
# is only validated for the base lineage from which its residuals were built.
BASE_COLUMNS = ("strong", "strong_bias", "cp12")
INCLUDE_NEIGHBOR_VARIANTS = True

N_FOLDS = 5
RANDOM_STATE = 20260716
N_BOOTSTRAP = 3000

# Frozen notebook configuration.  It is always reported independently.
NOTEBOOK_GEOMETRY = {
    "name": "notebook_exact",
    "theta": 2.2166,
    "anisotropy": 2.0,
    "range_ft": 500.0,
    "k": 25,
}
NOTEBOOK_APPLICATION = {
    "weight": 0.30,
    "ramp": "constant",
    "applied_cap_ft": None,
}

LABEL_ESTIMATORS = ("mean", "trim10", "huber")
APPLICATION_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50)
APPLICATION_RAMPS = ("constant", "tau85", "tau250", "tau500", "tau1000")
APPLICATION_CAPS = (3.0, 6.0, 12.0, None)

PASS_GAIN = 0.08
PASS_TOP40_GAIN = 0.05
PASS_MIN_FOLD_GAIN = 0.0


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(max(float(sse), 0.0) / max(int(rows), 1)))


def _huber_location(values, tuning=1.5, iterations=20):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)) + 1e-6)
    for _ in range(iterations):
        residual = (values - location) / scale
        weight = np.ones(len(values), np.float64)
        outside = np.abs(residual) > tuning
        weight[outside] = tuning / np.abs(residual[outside])
        updated = float(np.sum(weight * values) / max(np.sum(weight), 1e-12))
        if abs(updated - location) < 1e-7:
            break
        location = updated
    return location


def _trimmed_mean(values, fraction=0.10):
    values = np.sort(np.asarray(values, np.float64))
    trim = int(np.floor(len(values) * fraction))
    if trim and len(values) > 2 * trim:
        values = values[trim:-trim]
    return float(np.mean(values))


def _balanced_fallback_folds(well_table):
    table = well_table[["well", "n_eval"]].copy()
    table["stratum"] = pd.qcut(
        table["n_eval"].rank(method="first"), q=10, labels=False
    ).astype(int)
    table["fold"] = -1
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (_, valid) in enumerate(splitter.split(table, table["stratum"]), 1):
        table.loc[valid, "fold"] = fold
    return dict(zip(table["well"].astype(str), table["fold"].astype(int)))


def _load_fold_map(wells, well_table):
    candidates = [
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
        OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "well_metadata.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not {"well", "fold"}.issubset(frame.columns):
            continue
        frame["well"] = frame["well"].astype(str)
        mapping = frame.groupby("well")["fold"].first().astype(int).to_dict()
        if set(wells).issubset(mapping):
            return mapping, str(path)
    return _balanced_fallback_folds(well_table), "deterministic row-count StratifiedKFold"


def _well_id_from_horizontal_path(path):
    name = Path(path).name
    for suffix in ("__horizontal_well.csv", "_horizontal_well.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _discover_train_horizontal_files(required_wells):
    """Find official train wells even when the dataset has an extra directory level."""
    required_wells = set(map(str, required_wells))
    candidates = {well: [] for well in required_wells}
    roots = [DATA_ROOT / "train", DATA_ROOT]

    def add_from_root(root):
        if not root.exists():
            return
        for path in root.rglob("*horizontal_well.csv"):
            well = _well_id_from_horizontal_path(path)
            if well not in required_wells:
                continue
            lower_parts = {part.lower() for part in path.parts}
            if "test" in lower_parts:
                continue
            path_text = str(path).lower()
            if any(
                token in path_text
                for token in ("masked_tmp", "oof_exports", ".trash", "__pycache__")
            ):
                continue
            priority = (
                0 if "train" in lower_parts else 1,
                0 if DATA_ROOT in path.parents else 1,
                len(path.parts),
                str(path),
            )
            candidates[well].append((priority, path))

    for root in roots:
        add_from_root(root)

    missing = [well for well, rows in candidates.items() if not rows]
    if missing:
        # Last-resort discovery handles renamed/extracted competition roots.
        add_from_root(Path("/root/autodl-tmp"))

    selected = {}
    invalid = {}
    for well, rows in candidates.items():
        for _, path in sorted(rows, key=lambda item: item[0]):
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
                if {"MD", "X", "Y", "TVT_input"}.issubset(columns):
                    selected[well] = path
                    break
                invalid.setdefault(well, []).append(
                    {"path": str(path), "missing_columns": sorted({"MD", "X", "Y", "TVT_input"} - columns)}
                )
            except Exception as exc:
                invalid.setdefault(well, []).append(
                    {"path": str(path), "error": repr(exc)}
                )

    missing = sorted(required_wells - set(selected))
    if missing:
        examples = {
            well: invalid.get(well, [])[:3]
            for well in missing[:10]
        }
        raise FileNotFoundError(
            "Could not locate official train horizontal files for "
            f"{len(missing)} wells; examples={missing[:20]}; candidate diagnostics={examples}"
        )
    return selected


def _geometry_grid():
    rows = [dict(NOTEBOOK_GEOMETRY)]

    # A deliberately small structured search.  Nested CV, not full-OOF score,
    # decides whether any of these is better than the frozen notebook geometry.
    for theta_index, theta in enumerate(np.arange(0.0, np.pi, np.pi / 8.0)):
        for range_ft in (350.0, 500.0, 750.0, 1000.0):
            rows.append(
                {
                    "name": f"angle{theta_index}_a2_r{int(range_ft)}_k25",
                    "theta": float(theta),
                    "anisotropy": 2.0,
                    "range_ft": range_ft,
                    "k": 25,
                }
            )
    for range_ft in (350.0, 500.0, 750.0, 1000.0, 1500.0):
        rows.append(
            {
                "name": f"isotropic_r{int(range_ft)}_k25",
                "theta": 0.0,
                "anisotropy": 1.0,
                "range_ft": range_ft,
                "k": 25,
            }
        )
    for anisotropy in (1.5, 2.5, 3.0):
        rows.append(
            {
                "name": f"notebook_theta_a{anisotropy:g}_r500_k25",
                "theta": 2.2166,
                "anisotropy": anisotropy,
                "range_ft": 500.0,
                "k": 25,
            }
        )
    for k in (12, 40, 64):
        rows.append(
            {
                "name": f"notebook_theta_a2_r500_k{k}",
                "theta": 2.2166,
                "anisotropy": 2.0,
                "range_ft": 500.0,
                "k": k,
            }
        )

    unique = {}
    for row in rows:
        key = (row["theta"], row["anisotropy"], row["range_ft"], row["k"])
        unique.setdefault(key, row)
    return list(unique.values())


def _transform_xy(xy, geometry):
    theta = float(geometry["theta"])
    cosine, sine = np.cos(theta), np.sin(theta)
    x = xy[:, 0]
    y = xy[:, 1]
    return np.column_stack(
        [
            x * cosine + y * sine,
            (-x * sine + y * cosine) / float(geometry["anisotropy"]),
        ]
    )


def _predict_offsets(xy, labels, query_index, reference_index, geometry):
    query_index = np.asarray(query_index, dtype=int)
    reference_index = np.asarray(reference_index, dtype=int)
    transformed = _transform_xy(xy, geometry)
    query = transformed[query_index]
    reference = transformed[reference_index]
    reference_labels = np.asarray(labels, np.float64)[reference_index]

    distance2 = np.sum((query[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    k = min(int(geometry["k"]), len(reference_index))
    nearest_position = np.argpartition(distance2, kth=k - 1, axis=1)[:, :k]
    nearest_distance2 = np.take_along_axis(distance2, nearest_position, axis=1)
    nearest_labels = reference_labels[nearest_position]
    weights = np.exp(
        -nearest_distance2 / (2.0 * float(geometry["range_ft"]) ** 2)
    )
    weight_sum = np.sum(weights, axis=1)
    fallback = float(np.mean(reference_labels))
    prediction = np.full(len(query_index), fallback, np.float64)
    valid = weight_sum > 1e-12
    prediction[valid] = (
        np.sum(weights[valid] * nearest_labels[valid], axis=1) / weight_sum[valid]
    )
    nearest = np.sqrt(np.min(distance2, axis=1))
    ess = weight_sum**2 / np.maximum(np.sum(weights**2, axis=1), 1e-12)
    return prediction, nearest, ess


def _score_indices(stats, indices, offsets, application):
    indices = np.asarray(indices, dtype=int)
    offsets = np.asarray(offsets, np.float64)
    shift = float(application["weight"]) * offsets
    cap = application.get("applied_cap_ft")
    if cap is not None:
        shift = np.clip(shift, -float(cap), float(cap))
    ramp = str(application["ramp"])
    sse = stats["sse"][indices]
    sum_error_ramp = stats["sum_error_ramp"][ramp][indices]
    sum_ramp2 = stats["sum_ramp2"][ramp][indices]
    candidate_sse = sse + 2.0 * shift * sum_error_ramp + shift**2 * sum_ramp2
    rows = stats["n"][indices]
    return {
        "base_sse": float(np.sum(sse)),
        "prediction_sse": float(np.sum(candidate_sse)),
        "rows": int(np.sum(rows)),
        "shift": shift,
        "per_well_prediction_sse": candidate_sse,
    }


def _fold_score_table(stats, folds, offsets, application, allowed=None):
    if allowed is None:
        allowed = np.ones(len(folds), dtype=bool)
    rows = []
    for fold in sorted(np.unique(folds[allowed])):
        index = np.flatnonzero(allowed & (folds == fold))
        result = _score_indices(stats, index, offsets[index], application)
        base_rmse = _rmse_from_sse(result["base_sse"], result["rows"])
        prediction_rmse = _rmse_from_sse(result["prediction_sse"], result["rows"])
        rows.append(
            {
                "fold": int(fold),
                "rows": result["rows"],
                "base_rmse": base_rmse,
                "prediction_rmse": prediction_rmse,
                "gain": base_rmse - prediction_rmse,
            }
        )
    return pd.DataFrame(rows)


def _application_grid():
    for weight in APPLICATION_WEIGHTS:
        for ramp in APPLICATION_RAMPS:
            for cap in APPLICATION_CAPS:
                yield {
                    "weight": float(weight),
                    "ramp": ramp,
                    "applied_cap_ft": cap,
                }


def _select_application(stats, folds, offsets, allowed):
    candidates = []
    for application in _application_grid():
        report = _fold_score_table(stats, folds, offsets, application, allowed)
        index = np.flatnonzero(allowed)
        total = _score_indices(stats, index, offsets[index], application)
        candidates.append(
            {
                **application,
                "rmse": _rmse_from_sse(total["prediction_sse"], total["rows"]),
                "mean_fold_gain": float(report["gain"].mean()),
                "min_fold_gain": float(report["gain"].min()),
                "positive_folds": int((report["gain"] > 0).sum()),
                "folds": int(len(report)),
            }
        )
    table = pd.DataFrame(candidates)
    robust = table[
        (table["positive_folds"] == table["folds"])
        & (table["min_fold_gain"] >= -1e-10)
    ]
    if robust.empty:
        robust = table.sort_values(
            ["positive_folds", "min_fold_gain", "rmse"],
            ascending=[False, False, True],
        ).head(1)
    else:
        robust = robust.sort_values("rmse").head(1)
    selected = robust.iloc[0].to_dict()
    return {
        "weight": float(selected["weight"]),
        "ramp": str(selected["ramp"]),
        "applied_cap_ft": (
            None
            if pd.isna(selected["applied_cap_ft"])
            else float(selected["applied_cap_ft"])
        ),
    }, table


def _cv_offsets(xy, labels, folds, geometry, allowed=None):
    if allowed is None:
        allowed = np.ones(len(folds), dtype=bool)
    output = np.full(len(folds), np.nan, np.float64)
    nearest = np.full(len(folds), np.nan, np.float64)
    ess = np.full(len(folds), np.nan, np.float64)
    for fold in sorted(np.unique(folds[allowed])):
        query = np.flatnonzero(allowed & (folds == fold))
        reference = np.flatnonzero(allowed & (folds != fold))
        pred, dist, effective = _predict_offsets(
            xy, labels, query, reference, geometry
        )
        output[query] = pred
        nearest[query] = dist
        ess[query] = effective
    return output, nearest, ess


def _select_geometry(stats, xy, labels_by_estimator, folds, allowed, geometries):
    frozen_application = dict(NOTEBOOK_APPLICATION)
    candidates = []
    cache = {}
    for estimator, labels in labels_by_estimator.items():
        for geometry in geometries:
            offsets, nearest, ess = _cv_offsets(
                xy, labels, folds, geometry, allowed=allowed
            )
            index = np.flatnonzero(allowed)
            result = _score_indices(
                stats, index, offsets[index], frozen_application
            )
            fold_report = _fold_score_table(
                stats, folds, offsets, frozen_application, allowed
            )
            key = (estimator, geometry["name"])
            cache[key] = (offsets, nearest, ess)
            candidates.append(
                {
                    "estimator": estimator,
                    **geometry,
                    "rmse": _rmse_from_sse(
                        result["prediction_sse"], result["rows"]
                    ),
                    "mean_fold_gain": float(fold_report["gain"].mean()),
                    "min_fold_gain": float(fold_report["gain"].min()),
                    "positive_folds": int((fold_report["gain"] > 0).sum()),
                    "folds": int(len(fold_report)),
                }
            )
    table = pd.DataFrame(candidates)
    robust = table[table["positive_folds"] == table["folds"]]
    if robust.empty:
        robust = table.sort_values(
            ["positive_folds", "min_fold_gain", "rmse"],
            ascending=[False, False, True],
        ).head(1)
    else:
        robust = robust.sort_values("rmse").head(1)
    selected = robust.iloc[0]
    geometry = {
        "name": str(selected["name"]),
        "theta": float(selected["theta"]),
        "anisotropy": float(selected["anisotropy"]),
        "range_ft": float(selected["range_ft"]),
        "k": int(selected["k"]),
    }
    estimator = str(selected["estimator"])
    offsets, nearest, ess = cache[(estimator, geometry["name"])]
    return estimator, geometry, offsets, nearest, ess, table


def _summary_row(name, stats, folds, offsets, application, top40):
    index = np.arange(len(folds))
    result = _score_indices(stats, index, offsets, application)
    base_rmse = _rmse_from_sse(result["base_sse"], result["rows"])
    prediction_rmse = _rmse_from_sse(result["prediction_sse"], result["rows"])
    fold_report = _fold_score_table(stats, folds, offsets, application)
    top = _score_indices(
        stats, np.flatnonzero(top40), offsets[top40], application
    )
    top_base = _rmse_from_sse(top["base_sse"], top["rows"])
    top_prediction = _rmse_from_sse(top["prediction_sse"], top["rows"])
    return {
        "variant": name,
        "base_rmse": base_rmse,
        "prediction_rmse": prediction_rmse,
        "gain": base_rmse - prediction_rmse,
        "mean_fold_gain": float(fold_report["gain"].mean()),
        "min_fold_gain": float(fold_report["gain"].min()),
        "positive_folds": int((fold_report["gain"] > 0).sum()),
        "top40_gain": top_base - top_prediction,
        "mean_abs_shift": float(np.mean(np.abs(result["shift"]))),
        "p95_abs_shift": float(np.quantile(np.abs(result["shift"]), 0.95)),
        "max_abs_shift": float(np.max(np.abs(result["shift"]))),
    }, fold_report, result


def _bootstrap_gain(stats, prediction_sse, rng):
    gains = np.empty(N_BOOTSTRAP, np.float64)
    n_wells = len(stats["n"])
    for iteration in range(N_BOOTSTRAP):
        sample = rng.integers(0, n_wells, size=n_wells)
        rows = int(np.sum(stats["n"][sample]))
        base_rmse = _rmse_from_sse(np.sum(stats["sse"][sample]), rows)
        pred_rmse = _rmse_from_sse(np.sum(prediction_sse[sample]), rows)
        gains[iteration] = base_rmse - pred_rmse
    return {
        "iterations": N_BOOTSTRAP,
        "mean_gain": float(np.mean(gains)),
        "p01_gain": float(np.quantile(gains, 0.01)),
        "p05_gain": float(np.quantile(gains, 0.05)),
        "p50_gain": float(np.quantile(gains, 0.50)),
        "p95_gain": float(np.quantile(gains, 0.95)),
        "positive_probability": float(np.mean(gains > 0)),
    }


# ---------------------------------------------------------------------------
# Load honest row-level predictions.
# ---------------------------------------------------------------------------
started = time.time()
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError(f"No candidate pool under {OOF_ROOT}")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
available_bases = [column for column in BASE_COLUMNS if column in header]
if not available_bases:
    raise RuntimeError(f"None of {BASE_COLUMNS} found in pool: {header}")

usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", *available_bases)
    if column in header
]
print("loading:", pool_path)
pool = pd.read_csv(
    pool_path,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)

# Optional neighbor-adjusted bases.  These are separate lineages and therefore
# receive separate lookup files.
base_arrays = {
    column: pool[column].to_numpy(np.float64) for column in available_bases
}
neighbor_source = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
if INCLUDE_NEIGHBOR_VARIANTS and (
    (neighbor_source / "neighbor_transfer_oof.npz").exists()
    and (neighbor_source / "selected_neighbors.csv").exists()
    and "strong" in base_arrays
):
    sidecar = np.load(neighbor_source / "neighbor_transfer_oof.npz")
    neighbor_candidate = sidecar["structure_progress"].astype(np.float64)
    if len(neighbor_candidate) != len(pool):
        raise RuntimeError(
            f"neighbor sidecar/pool mismatch: {len(neighbor_candidate)} != {len(pool)}"
        )
    neighbor = pd.read_csv(neighbor_source / "selected_neighbors.csv")
    neighbor["well"] = neighbor["well"].astype(str)
    meta = neighbor.set_index("well")
    eligible = (
        (pool["well"].map(meta["median_distance"]).to_numpy(float) <= 600.0)
        & (pool["well"].map(meta["p90_distance"]).to_numpy(float) <= 1200.0)
        & (pool["well"].map(meta["anchor_distance_full"]).to_numpy(float) <= 1200.0)
        & (pool["well"].map(meta["abs_direction_dot"]).to_numpy(float) >= 0.70)
        & np.isfinite(neighbor_candidate)
    )
    for stage in ("strong", "strong_bias", "cp12"):
        if stage not in base_arrays:
            continue
        base = base_arrays[stage]
        move = np.zeros(len(pool), np.float64)
        move[eligible] = 0.25 * np.clip(
            neighbor_candidate[eligible] - base[eligible], -24.0, 24.0
        )
        base_arrays[f"{stage}_neighbor_recentered"] = base + move

truth = pool["tvt_true"].to_numpy(np.float64)
wells = sorted(pool["well"].unique())
well_to_position = {well: position for position, well in enumerate(wells)}


# ---------------------------------------------------------------------------
# Build coordinates, ramp statistics, and robust per-well residual labels.
# ---------------------------------------------------------------------------
ramp_names = APPLICATION_RAMPS
well_rows = []
stats_by_base = {
    base: {
        "n": np.zeros(len(wells), np.int64),
        "sse": np.zeros(len(wells), np.float64),
        "sum_error_ramp": {
            ramp: np.zeros(len(wells), np.float64) for ramp in ramp_names
        },
        "sum_ramp2": {
            ramp: np.zeros(len(wells), np.float64) for ramp in ramp_names
        },
        "labels": {
            estimator: np.zeros(len(wells), np.float64)
            for estimator in LABEL_ESTIMATORS
        },
    }
    for base in base_arrays
}

train_file_map = _discover_train_horizontal_files(wells)
print(
    "train horizontal discovery:",
    {
        "required": len(wells),
        "found": len(train_file_map),
        "example": str(train_file_map[wells[0]]),
    },
)

for count, (well, frame) in enumerate(pool.groupby("well", sort=False), 1):
    position = well_to_position[well]
    path = train_file_map[well]
    horizontal = pd.read_csv(path, usecols=["MD", "X", "Y", "TVT_input"])
    row_index = frame["row_idx"].to_numpy(int)
    if row_index.max() >= len(horizontal):
        raise RuntimeError(f"row index outside horizontal file for {well}")
    known = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(float)))
    if not len(known):
        raise RuntimeError(f"no known anchor for {well}")
    anchor = int(known[-1])
    md = horizontal["MD"].to_numpy(float)
    md_since = np.maximum(md[row_index] - md[anchor], 0.0)
    x = horizontal["X"].to_numpy(float)[row_index]
    y = horizontal["Y"].to_numpy(float)[row_index]
    ramps = {"constant": np.ones(len(frame), np.float64)}
    for tau in (85.0, 250.0, 500.0, 1000.0):
        ramps[f"tau{int(tau)}"] = 1.0 - np.exp(-md_since / tau)

    well_rows.append(
        {
            "well": well,
            "cx": float(np.mean(x)),
            "cy": float(np.mean(y)),
            "anchor_x": float(horizontal["X"].iloc[anchor]),
            "anchor_y": float(horizontal["Y"].iloc[anchor]),
            "n_eval": int(len(frame)),
        }
    )
    row_position = frame.index.to_numpy(int)
    well_truth = truth[row_position]
    for base_name, full_prediction in base_arrays.items():
        error = full_prediction[row_position] - well_truth
        residual = -error
        stats = stats_by_base[base_name]
        stats["n"][position] = len(error)
        stats["sse"][position] = float(np.sum(error**2))
        for ramp_name, ramp in ramps.items():
            stats["sum_error_ramp"][ramp_name][position] = float(
                np.sum(error * ramp)
            )
            stats["sum_ramp2"][ramp_name][position] = float(np.sum(ramp**2))
        stats["labels"]["mean"][position] = float(np.mean(residual))
        stats["labels"]["trim10"][position] = _trimmed_mean(residual)
        stats["labels"]["huber"][position] = _huber_location(residual)
    if count % 100 == 0:
        print(f"well statistics {count}/{len(wells)}", flush=True)

well_table = pd.DataFrame(well_rows).sort_values("well").reset_index(drop=True)
if well_table["well"].tolist() != wells:
    raise RuntimeError("well metadata order mismatch")
fold_map, fold_source = _load_fold_map(wells, well_table)
well_table["fold"] = well_table["well"].map(fold_map).astype(int)
folds = well_table["fold"].to_numpy(int)
xy = well_table[["cx", "cy"]].to_numpy(np.float64)

OUT_DIR.mkdir(parents=True, exist_ok=True)
well_table.to_csv(OUT_DIR / "well_coordinates_and_folds.csv", index=False)
print("fold source:", fold_source)
print("bases:", list(base_arrays))


# ---------------------------------------------------------------------------
# Frozen reproduction, strict nested selection, and production lookup.
# ---------------------------------------------------------------------------
geometries = _geometry_grid()
all_score_rows = []
all_fold_rows = []
all_nested_rows = []
base_verdicts = {}
rng = np.random.default_rng(RANDOM_STATE)

for base_name, stats in stats_by_base.items():
    print("\n" + "=" * 78)
    print("BASE:", base_name)
    base_dir = OUT_DIR / base_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Top 40 is frozen from this base before any spatial correction.
    top40_index = np.argsort(stats["sse"])[-40:]
    top40 = np.zeros(len(wells), dtype=bool)
    top40[top40_index] = True

    exact_offsets, exact_nearest, exact_ess = _cv_offsets(
        xy, stats["labels"]["mean"], folds, NOTEBOOK_GEOMETRY
    )
    exact_summary, exact_folds, exact_result = _summary_row(
        "notebook_exact_frozen",
        stats,
        folds,
        exact_offsets,
        NOTEBOOK_APPLICATION,
        top40,
    )
    exact_summary.update(
        {
            "base": base_name,
            "estimator": "mean",
            "geometry": NOTEBOOK_GEOMETRY["name"],
            **NOTEBOOK_APPLICATION,
        }
    )
    all_score_rows.append(exact_summary)
    exact_folds.insert(0, "variant", "notebook_exact_frozen")
    exact_folds.insert(0, "base", base_name)
    all_fold_rows.append(exact_folds)

    # Strict nested outer loop.  Geometry and application are selected using
    # only the four outer-training folds; all references exclude the outer fold.
    nested_offsets = np.full(len(wells), np.nan, np.float64)
    nested_prediction_sse = np.full(len(wells), np.nan, np.float64)
    nested_shift = np.full(len(wells), np.nan, np.float64)
    nested_fold_reports = []

    for outer_fold in range(1, N_FOLDS + 1):
        outer_train = folds != outer_fold
        outer_valid = folds == outer_fold
        (
            selected_estimator,
            selected_geometry,
            inner_offsets,
            _,
            _,
            geometry_table,
        ) = _select_geometry(
            stats,
            xy,
            stats["labels"],
            folds,
            outer_train,
            geometries,
        )
        selected_application, application_table = _select_application(
            stats, folds, inner_offsets, outer_train
        )
        query = np.flatnonzero(outer_valid)
        reference = np.flatnonzero(outer_train)
        outer_offsets, outer_nearest, outer_ess = _predict_offsets(
            xy,
            stats["labels"][selected_estimator],
            query,
            reference,
            selected_geometry,
        )
        result = _score_indices(
            stats, query, outer_offsets, selected_application
        )
        nested_offsets[query] = outer_offsets
        nested_shift[query] = result["shift"]
        nested_prediction_sse[query] = result["per_well_prediction_sse"]
        base_rmse = _rmse_from_sse(result["base_sse"], result["rows"])
        pred_rmse = _rmse_from_sse(result["prediction_sse"], result["rows"])
        row = {
            "base": base_name,
            "outer_fold": outer_fold,
            "estimator": selected_estimator,
            **{f"geometry_{k}": v for k, v in selected_geometry.items()},
            **{f"application_{k}": v for k, v in selected_application.items()},
            "valid_rows": result["rows"],
            "base_rmse": base_rmse,
            "prediction_rmse": pred_rmse,
            "gain": base_rmse - pred_rmse,
            "nearest_p95": float(np.quantile(outer_nearest, 0.95)),
            "ess_p05": float(np.quantile(outer_ess, 0.05)),
        }
        all_nested_rows.append(row)
        nested_fold_reports.append(
            {
                "base": base_name,
                "variant": "nested_optimized",
                "fold": outer_fold,
                "rows": result["rows"],
                "base_rmse": base_rmse,
                "prediction_rmse": pred_rmse,
                "gain": base_rmse - pred_rmse,
            }
        )
        geometry_table.to_csv(
            base_dir / f"outer{outer_fold}_geometry_search.csv", index=False
        )
        application_table.to_csv(
            base_dir / f"outer{outer_fold}_application_search.csv", index=False
        )
        print("nested fold:", row)

    if not np.isfinite(nested_prediction_sse).all():
        raise RuntimeError(f"incomplete nested predictions for {base_name}")
    nested_rows = int(np.sum(stats["n"]))
    nested_base_rmse = _rmse_from_sse(np.sum(stats["sse"]), nested_rows)
    nested_rmse = _rmse_from_sse(np.sum(nested_prediction_sse), nested_rows)
    nested_fold_table = pd.DataFrame(nested_fold_reports)
    top_base = _rmse_from_sse(
        np.sum(stats["sse"][top40]), np.sum(stats["n"][top40])
    )
    top_pred = _rmse_from_sse(
        np.sum(nested_prediction_sse[top40]), np.sum(stats["n"][top40])
    )
    bootstrap = _bootstrap_gain(stats, nested_prediction_sse, rng)
    nested_summary = {
        "base": base_name,
        "variant": "nested_optimized",
        "estimator": "selected independently inside each outer fold",
        "geometry": "selected independently inside each outer fold",
        "weight": np.nan,
        "ramp": "selected independently inside each outer fold",
        "applied_cap_ft": np.nan,
        "base_rmse": nested_base_rmse,
        "prediction_rmse": nested_rmse,
        "gain": nested_base_rmse - nested_rmse,
        "mean_fold_gain": float(nested_fold_table["gain"].mean()),
        "min_fold_gain": float(nested_fold_table["gain"].min()),
        "positive_folds": int((nested_fold_table["gain"] > 0).sum()),
        "top40_gain": top_base - top_pred,
        "mean_abs_shift": float(np.mean(np.abs(nested_shift))),
        "p95_abs_shift": float(np.quantile(np.abs(nested_shift), 0.95)),
        "max_abs_shift": float(np.max(np.abs(nested_shift))),
        "bootstrap_p05_gain": bootstrap["p05_gain"],
        "bootstrap_positive_probability": bootstrap["positive_probability"],
    }
    all_score_rows.append(nested_summary)
    all_fold_rows.append(nested_fold_table)

    # Select one production configuration from all five cross-fitted folds.
    (
        production_estimator,
        production_geometry,
        production_offsets,
        production_nearest,
        production_ess,
        production_geometry_table,
    ) = _select_geometry(
        stats,
        xy,
        stats["labels"],
        folds,
        np.ones(len(wells), dtype=bool),
        geometries,
    )
    production_application, production_application_table = _select_application(
        stats,
        folds,
        production_offsets,
        np.ones(len(wells), dtype=bool),
    )
    production_summary, production_folds, production_result = _summary_row(
        "production_selected_cv",
        stats,
        folds,
        production_offsets,
        production_application,
        top40,
    )
    production_summary.update(
        {
            "base": base_name,
            "estimator": production_estimator,
            "geometry": production_geometry["name"],
            **production_application,
        }
    )
    all_score_rows.append(production_summary)
    production_folds.insert(0, "variant", "production_selected_cv")
    production_folds.insert(0, "base", base_name)
    all_fold_rows.append(production_folds)
    production_geometry_table.to_csv(
        base_dir / "production_geometry_search.csv", index=False
    )
    production_application_table.to_csv(
        base_dir / "production_application_search.csv", index=False
    )

    labels = stats["labels"][production_estimator]
    lookup = well_table[["well", "cx", "cy"]].copy()
    lookup["b_w"] = labels
    lookup[["cx", "cy", "b_w", "well"]].to_csv(
        base_dir / "offset_krige_lookup.csv", index=False
    )
    diagnostics = lookup.copy()
    diagnostics["fold"] = folds
    diagnostics["n_eval"] = stats["n"]
    diagnostics["base_rmse"] = np.sqrt(stats["sse"] / stats["n"])
    for estimator in LABEL_ESTIMATORS:
        diagnostics[f"b_w_{estimator}"] = stats["labels"][estimator]
    diagnostics.to_csv(base_dir / "offset_krige_lookup_diagnostics.csv", index=False)

    guard_nearest_max = float(max(1500.0, np.quantile(production_nearest, 0.99) * 1.25))
    guard_ess_min = float(max(1.0, np.quantile(production_ess, 0.01) * 0.50))
    config = {
        "experiment": "offset_krige_oof_train_v1",
        "pool": str(pool_path),
        "base_lineage": base_name,
        "label_definition": "robust_location(tvt_true - honest_oof_prediction)",
        "label_estimator": production_estimator,
        "coordinate_definition": "mean X,Y over official eval rows",
        "geometry": production_geometry,
        "application": production_application,
        "coverage_guard": {
            "nearest_distance_max": guard_nearest_max,
            "effective_neighbors_min": guard_ess_min,
        },
        "nested_validation": nested_summary,
        "bootstrap": bootstrap,
        "frozen_notebook_validation": exact_summary,
        "fold_source": fold_source,
        "training_wells": len(wells),
    }
    with (base_dir / "offset_krige_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    strict_pass = bool(
        nested_summary["gain"] >= PASS_GAIN
        and nested_summary["positive_folds"] == N_FOLDS
        and nested_summary["min_fold_gain"] > PASS_MIN_FOLD_GAIN
        and nested_summary["top40_gain"] >= PASS_TOP40_GAIN
        and bootstrap["p05_gain"] > 0.0
    )
    base_verdicts[base_name] = {
        "strict_pass": strict_pass,
        "nested_prediction_rmse": nested_summary["prediction_rmse"],
        "nested_gain": nested_summary["gain"],
        "nested_min_fold_gain": nested_summary["min_fold_gain"],
        "nested_top40_gain": nested_summary["top40_gain"],
        "bootstrap_p05_gain": bootstrap["p05_gain"],
        "production_estimator": production_estimator,
        "production_geometry": production_geometry,
        "production_application": production_application,
        "lookup": str(base_dir / "offset_krige_lookup.csv"),
        "config": str(base_dir / "offset_krige_config.json"),
    }
    print("FROZEN:", exact_summary)
    print("NESTED:", nested_summary)
    print("PRODUCTION CONFIG:", config)
    print("STRICT PASS:", strict_pass)


score_report = pd.DataFrame(all_score_rows).sort_values(
    ["variant", "gain"], ascending=[True, False]
)
fold_report = pd.concat(all_fold_rows, ignore_index=True)
nested_report = pd.DataFrame(all_nested_rows)
score_report.to_csv(OUT_DIR / "offset_krige_scores.csv", index=False)
fold_report.to_csv(OUT_DIR / "offset_krige_folds.csv", index=False)
nested_report.to_csv(OUT_DIR / "offset_krige_nested_selection.csv", index=False)

passing = [
    (name, verdict) for name, verdict in base_verdicts.items() if verdict["strict_pass"]
]
promoted = None
if passing:
    promoted, promoted_verdict = min(
        passing, key=lambda item: item[1]["nested_prediction_rmse"]
    )
    source_dir = OUT_DIR / promoted
    # pandas/json rewrites keep this cell self-contained and portable.
    pd.read_csv(source_dir / "offset_krige_lookup.csv").to_csv(
        OUT_DIR / "offset_krige_lookup.csv", index=False
    )
    with (source_dir / "offset_krige_config.json").open(encoding="utf-8") as handle:
        promoted_config = json.load(handle)
    with (OUT_DIR / "offset_krige_config.json").open("w", encoding="utf-8") as handle:
        json.dump(promoted_config, handle, indent=2, sort_keys=True)

verdict = {
    "experiment": "offset_krige_oof_train_v1",
    "pool": str(pool_path),
    "fold_source": fold_source,
    "bases": base_verdicts,
    "promoted_base": promoted,
    "strict_pass": promoted is not None,
    "upload_files": (
        [
            str(OUT_DIR / "offset_krige_lookup.csv"),
            str(OUT_DIR / "offset_krige_config.json"),
        ]
        if promoted is not None
        else []
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nSCORES")
print(score_report.to_string(index=False))
print("\nNESTED FOLDS")
print(nested_report.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
