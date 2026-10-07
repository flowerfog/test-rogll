"""Raw active-steering / trajectory-tortuosity observability gateway.

This experiment asks whether the drilled trajectory contains information about
the future formation surface after controlling for distance and the visible
prefix trend. It does not consume GR matching outputs, historical predictions,
CP, Neighbor, Gold, or leaderboard artifacts.

Fold 4/5 train compact tabular residual models. A deterministic 24-well subset
of Fold 3 selects capacity and smoothing; the remaining Fold-3 wells provide a
matched confirmation. Fold 2 is touched only after the mechanism gates pass.
Fold 1 remains untouched.
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
from scipy.ndimage import gaussian_filter1d


EXPERIMENT = "raw_active_steering_tortuosity_v54"
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
WINDOWS_FT = (32.0, 64.0, 128.0, 256.0, 512.0)
MODEL_CONFIGS = (
    {
        "id": "leaf15_child80",
        "num_leaves": 15,
        "min_child_samples": 80,
        "reg_lambda": 4.0,
    },
    {
        "id": "leaf31_child120",
        "num_leaves": 31,
        "min_child_samples": 120,
        "reg_lambda": 8.0,
    },
    {
        "id": "leaf15_child240",
        "num_leaves": 15,
        "min_child_samples": 240,
        "reg_lambda": 12.0,
    },
)
SMOOTH_NODES = (0.0, 1.0, 2.0)
DISTANCE_EDGES = (-np.inf, 500.0, 1000.0, 2000.0, np.inf)
DISTANCE_LABELS = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
SEED = 54027


def _bootstrap(path: Path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_stochastic_v30_trajectory_definitions"
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


def _numeric(frame, names):
    lower = {str(column).lower(): column for column in frame.columns}
    column = next((lower[name.lower()] for name in names if name.lower() in lower), None)
    if column is None:
        raise KeyError(f"Missing one of {names}; columns={list(frame.columns)}")
    return pd.to_numeric(frame[column], errors="coerce").interpolate(
        limit_direction="both"
    ).to_numpy(np.float64)


def _load_record(well):
    record = V._load_record(str(well))
    path = V.DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path).sort_values("MD").reset_index(drop=True)
    if len(frame) != len(record["md"]):
        raise RuntimeError(f"{well}: raw/V30 row mismatch {len(frame)}/{len(record['md'])}")
    record["x"] = _numeric(frame, ("X", "x"))
    record["y"] = _numeric(frame, ("Y", "y"))
    raw_z = _numeric(frame, ("Z", "z"))
    if np.nanmax(np.abs(raw_z - record["z"])) > 1e-5:
        raise RuntimeError(f"{well}: Z alignment mismatch")
    return record


def _load_records(wells, label):
    records, failures = [], []
    for number, well in enumerate(wells, 1):
        try:
            records.append(_load_record(str(well)))
        except Exception as exc:
            failures.append({"well": str(well), "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"V54 load {label} {number}/{len(wells)} failures={len(failures)}", flush=True)
    pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
    if len(records) < max(len(wells) - 2, 1):
        raise RuntimeError(f"Only {len(records)}/{len(wells)} usable {label} records")
    return records


def _interp(axis, values, query):
    return np.interp(query, axis, values, left=values[0], right=values[-1])


def _trajectory_features(record):
    md = record["md"]
    x, y, z = record["x"], record["y"], record["z"]
    node_rows = V._node_rows(record)
    node_md = md[node_rows]
    distance = node_md - record["anchor_md"]
    total_distance = max(float(distance[-1]), 1.0)
    anchor = int(record["anchor"])
    spacing = max(float(np.median(np.diff(md))), 1e-3)

    dx = np.gradient(x, md)
    dy = np.gradient(y, md)
    dz = np.gradient(z, md)
    tangent_norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    tangent_norm = np.maximum(tangent_norm, 1e-8)
    tx, ty, tz = dx / tangent_norm, dy / tangent_norm, dz / tangent_norm

    features = {
        "distance_kft": distance / 1000.0,
        "log_distance": np.log1p(np.maximum(distance, 0.0)) / 8.0,
        "progress": distance / total_distance,
        "total_length_kft": np.full(len(node_rows), total_distance / 1000.0),
        "prefix_surface_rate": np.full(len(node_rows), record["prefix_surface_rate"]),
    }
    trajectory_columns = []
    z_columns = []

    def add(name, value, z_feature=False):
        features[name] = np.asarray(value, np.float64)
        trajectory_columns.append(name)
        if z_feature:
            z_columns.append(name)

    add("delta_x_kft", (x[node_rows] - x[anchor]) / 1000.0)
    add("delta_y_kft", (y[node_rows] - y[anchor]) / 1000.0)
    add("delta_z_kft", (z[node_rows] - z[anchor]) / 1000.0, True)
    chord = np.sqrt(
        (x[node_rows] - x[anchor]) ** 2
        + (y[node_rows] - y[anchor]) ** 2
        + (z[node_rows] - z[anchor]) ** 2
    )
    add("anchor_chord_ratio", distance / np.maximum(chord, 1.0))
    add("anchor_vertical_rate", (z[node_rows] - z[anchor]) / np.maximum(distance, 1.0), True)

    for scale in WINDOWS_FT:
        sigma = max(scale / spacing, 0.25)
        sx = gaussian_filter1d(tx, sigma=sigma, mode="nearest")
        sy = gaussian_filter1d(ty, sigma=sigma, mode="nearest")
        sz = gaussian_filter1d(tz, sigma=sigma, mode="nearest")
        curvature = np.sqrt(
            np.gradient(sx, md) ** 2
            + np.gradient(sy, md) ** 2
            + np.gradient(sz, md) ** 2
        )
        tag = str(int(scale))
        add(f"tangent_x_s{tag}", sx[node_rows])
        add(f"tangent_y_s{tag}", sy[node_rows])
        add(f"tangent_z_s{tag}", sz[node_rows], True)
        add(f"curvature_s{tag}", np.log1p(10000.0 * curvature[node_rows]))
        add(
            f"vertical_curvature_s{tag}",
            1000.0 * np.gradient(sz, md)[node_rows],
            True,
        )

        left_md = np.maximum(node_md - scale, md[0])
        right_md = np.minimum(node_md + scale, md[-1])
        left_x, right_x = _interp(md, x, left_md), _interp(md, x, right_md)
        left_y, right_y = _interp(md, y, left_md), _interp(md, y, right_md)
        left_z, right_z = _interp(md, z, left_md), _interp(md, z, right_md)
        span = np.maximum(right_md - left_md, 1.0)
        local_chord = np.sqrt(
            (right_x - left_x) ** 2
            + (right_y - left_y) ** 2
            + (right_z - left_z) ** 2
        )
        # A transparent 3-D arc/chord tortuosity proxy, not a claim to
        # reproduce a proprietary or paper-specific Q-3D implementation.
        add(f"arc_chord_tortuosity_w{tag}", span / np.maximum(local_chord, 1.0) - 1.0)
        add(f"window_dx_w{tag}", (right_x - left_x) / span)
        add(f"window_dy_w{tag}", (right_y - left_y) / span)
        add(f"window_dz_w{tag}", (right_z - left_z) / span, True)

        past_md = np.maximum(node_md - scale, md[0])
        past_x, past_y, past_z = (
            _interp(md, x, past_md),
            _interp(md, y, past_md),
            _interp(md, z, past_md),
        )
        past_span = np.maximum(node_md - past_md, 1.0)
        past_chord = np.sqrt(
            (x[node_rows] - past_x) ** 2
            + (y[node_rows] - past_y) ** 2
            + (z[node_rows] - past_z) ** 2
        )
        add(f"trailing_arc_chord_w{tag}", past_span / np.maximum(past_chord, 1.0) - 1.0)
        add(f"trailing_dz_w{tag}", (z[node_rows] - past_z) / past_span, True)

    frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    surface = record["truth"][node_rows] + z[node_rows]
    baseline_surface = record["anchor_surface"] + record["prefix_surface_rate"] * distance
    target = surface - baseline_surface
    weights = np.empty(len(node_rows), np.float64)
    weights[:-1] = np.maximum(np.diff(node_rows), 1)
    weights[-1] = weights[-2] if len(weights) > 1 else 1.0
    return {
        "well": record["well"],
        "record": record,
        "node_rows": node_rows,
        "distance": distance,
        "baseline_surface": baseline_surface,
        "features": frame,
        "target": target,
        "weights": weights,
        "base_columns": [name for name in frame if name not in trajectory_columns],
        "z_columns": sorted(set([name for name in trajectory_columns if name in z_columns])),
        "trajectory_columns": trajectory_columns,
    }


def _build_cache(records, label):
    cache = []
    for number, record in enumerate(records, 1):
        cache.append(_trajectory_features(record))
        if number % 50 == 0 or number == len(records):
            print(f"V54 features {label} {number}/{len(records)}", flush=True)
    return cache


def _matrix(cache, columns, train=False):
    matrices, targets, weights = [], [], []
    for item in cache:
        start = 1 if train else 0
        matrices.append(item["features"].iloc[start:][columns].to_numpy(np.float32))
        targets.append(item["target"][start:].astype(np.float32))
        weights.append(item["weights"][start:].astype(np.float32))
    x = np.concatenate(matrices)
    if not train:
        return x
    return x, np.concatenate(targets), np.concatenate(weights)


def _fit_model(train_cache, valid_cache, columns, config):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    x_train, y_train, w_train = _matrix(train_cache, columns, train=True)
    x_valid, y_valid, w_valid = _matrix(valid_cache, columns, train=True)
    model = LGBMRegressor(
        objective="huber",
        alpha=0.9,
        n_estimators=1600,
        learning_rate=0.025,
        num_leaves=int(config["num_leaves"]),
        min_child_samples=int(config["min_child_samples"]),
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=float(config["reg_lambda"]),
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(x_valid, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric="l2",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    return model


def _predict_cache(model, cache, columns, shuffle_trajectory=False):
    x = _matrix(cache, columns, train=False)
    if shuffle_trajectory:
        trajectory_indices = [i for i, name in enumerate(columns) if name not in cache[0]["base_columns"]]
        distance_index = columns.index("distance_kft")
        rng = np.random.default_rng(SEED + 991)
        distance = x[:, distance_index]
        bins = np.digitize(distance, (0.5, 1.0, 2.0, 3.0, 4.0))
        for group in np.unique(bins):
            rows = np.flatnonzero(bins == group)
            permutation = rng.permutation(rows)
            x[np.ix_(rows, trajectory_indices)] = x[np.ix_(permutation, trajectory_indices)]
    return model.predict(x).astype(np.float64)


def _evaluate(cache, residual, endpoint, smooth_nodes):
    reports = []
    cursor = 0
    for item in cache:
        count = len(item["node_rows"])
        node_residual = residual[cursor:cursor + count].copy()
        cursor += count
        node_residual -= node_residual[0]
        if smooth_nodes > 0.0:
            node_residual = gaussian_filter1d(node_residual, sigma=smooth_nodes, mode="nearest")
            node_residual -= node_residual[0]
        node_surface = item["baseline_surface"] + node_residual
        record = item["record"]
        evaluation = record["evaluation"]
        prediction_surface = np.interp(
            record["md"][evaluation], record["md"][item["node_rows"]], node_surface
        )
        prediction = prediction_surface - record["z"][evaluation]
        truth = record["truth"][evaluation]
        error = prediction - truth
        distance = record["md"][evaluation] - record["anchor_md"]
        report = {
            "well": record["well"],
            "endpoint": endpoint,
            "rows": int(len(error)),
            "sse": float(np.dot(error, error)),
            "sae": float(np.abs(error).sum()),
        }
        for left, right, name in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_LABELS):
            mask = (distance > left) & (distance <= right)
            report[f"{name}_rows"] = int(mask.sum())
            report[f"{name}_sse"] = float(np.dot(error[mask], error[mask])) if mask.any() else 0.0
        reports.append(report)
    if cursor != len(residual):
        raise RuntimeError(f"Prediction cursor mismatch {cursor}/{len(residual)}")
    return pd.DataFrame(reports)


def _baseline(cache):
    residual = np.zeros(sum(len(item["node_rows"]) for item in cache), np.float64)
    return _evaluate(cache, residual, "prefix_trend", 0.0)


def _summary(frame):
    rows = int(frame["rows"].sum())
    well_rmse = np.sqrt(frame["sse"] / frame["rows"])
    result = {
        "endpoint": str(frame.iloc[0]["endpoint"]),
        "rows": rows,
        "wells": int(len(frame)),
        "exact_sse": float(frame["sse"].sum()),
        "rmse": float(np.sqrt(frame["sse"].sum() / rows)),
        "mae": float(frame["sae"].sum() / rows),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    for name in DISTANCE_LABELS:
        count = int(frame[f"{name}_rows"].sum())
        result[f"{name}_rmse"] = (
            float(np.sqrt(frame[f"{name}_sse"].sum() / count)) if count else np.nan
        )
    return result


def _prediction_reports(model, cache, columns, endpoint, smooth, shuffled=False):
    prediction = _predict_cache(model, cache, columns, shuffle_trajectory=shuffled)
    return _evaluate(cache, prediction, endpoint, smooth)


started = time.time()
folds = pd.read_csv(V30_DIR / "folds.csv", dtype={"well": str})
train_wells = sorted(folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"].astype(str))
fold3_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].astype(str))
train_records = _load_records(train_wells, "fold45")
fold3_records = _load_records(fold3_wells, "fold3")
fold3_records.sort(
    key=lambda record: hashlib.sha256(f"v54:{record['well']}".encode("ascii")).digest()
)
discovery_records = fold3_records[:DISCOVERY_WELLS]
confirmation_records = fold3_records[DISCOVERY_WELLS:]
train_cache = _build_cache(train_records, "fold45")
discovery_cache = _build_cache(discovery_records, "fold3_discovery")
confirmation_cache = _build_cache(confirmation_records, "fold3_confirmation")

base_columns = train_cache[0]["base_columns"]
z_columns = base_columns + train_cache[0]["z_columns"]
full_columns = list(train_cache[0]["features"].columns)
schemas = {
    "distance_only": base_columns,
    "z_trajectory": z_columns,
    "full_3d_tortuosity": full_columns,
}
print(
    "V54 CONFIG",
    {
        "train_wells": len(train_cache),
        "discovery_wells": len(discovery_cache),
        "confirmation_wells": len(confirmation_cache),
        "features": {key: len(value) for key, value in schemas.items()},
        "fold1_touched": False,
    },
    flush=True,
)

model_bank = {}
scan_rows = []
discovery_outputs = [_baseline(discovery_cache)]
for config in MODEL_CONFIGS:
    full_model = _fit_model(train_cache, discovery_cache, full_columns, config)
    model_bank[config["id"]] = full_model
    raw_prediction = _predict_cache(full_model, discovery_cache, full_columns)
    for smooth in SMOOTH_NODES:
        frame = _evaluate(
            discovery_cache,
            raw_prediction,
            f"{config['id']}_smooth{smooth:g}",
            smooth,
        )
        discovery_outputs.append(frame)
        scan_rows.append({
            **config,
            "smooth_nodes": smooth,
            "best_iteration": int(full_model.best_iteration_),
            **_summary(frame),
        })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "far_gt2000_rmse"], kind="stable")
selected_id = str(scan.iloc[0]["id"])
selected_smooth = float(scan.iloc[0]["smooth_nodes"])
selected_config = next(config for config in MODEL_CONFIGS if config["id"] == selected_id)
full_model = model_bank[selected_id]

distance_model = _fit_model(train_cache, discovery_cache, base_columns, selected_config)
z_model = _fit_model(train_cache, discovery_cache, z_columns, selected_config)
confirmation_frames = {
    "prefix_trend": _baseline(confirmation_cache),
    "distance_only": _prediction_reports(
        distance_model, confirmation_cache, base_columns, "distance_only", selected_smooth
    ),
    "z_trajectory": _prediction_reports(
        z_model, confirmation_cache, z_columns, "z_trajectory", selected_smooth
    ),
    "full_3d_tortuosity": _prediction_reports(
        full_model, confirmation_cache, full_columns, "full_3d_tortuosity", selected_smooth
    ),
    "shuffled_3d_tortuosity": _prediction_reports(
        full_model,
        confirmation_cache,
        full_columns,
        "shuffled_3d_tortuosity",
        selected_smooth,
        shuffled=True,
    ),
}
confirmation_summary = {key: _summary(value) for key, value in confirmation_frames.items()}
full = confirmation_summary["full_3d_tortuosity"]
distance_control = confirmation_summary["distance_only"]
z_control = confirmation_summary["z_trajectory"]
shuffle_control = confirmation_summary["shuffled_3d_tortuosity"]
gates = {
    "full_gain_vs_distance_ge_0p25": distance_control["rmse"] - full["rmse"] >= 0.25,
    "full_far_gain_vs_distance_ge_0p30": (
        distance_control["far_gt2000_rmse"] - full["far_gt2000_rmse"] >= 0.30
    ),
    "full_gain_vs_z_ge_0p05": z_control["rmse"] - full["rmse"] >= 0.05,
    "shuffle_loses_ge_0p20": shuffle_control["rmse"] - full["rmse"] >= 0.20,
}
mechanism_pass = bool(all(gates.values()))

summary_rows = [
    {"stage": "fold3_confirmation", **summary}
    for summary in confirmation_summary.values()
]
application = None
if mechanism_pass:
    fold2_records = _load_records(fold2_wells, "fold2")
    fold2_cache = _build_cache(fold2_records, "fold2")
    fold2_distance = _prediction_reports(
        distance_model, fold2_cache, base_columns, "distance_only", selected_smooth
    )
    fold2_full = _prediction_reports(
        full_model, fold2_cache, full_columns, "full_3d_tortuosity", selected_smooth
    )
    distance2, full2 = _summary(fold2_distance), _summary(fold2_full)
    summary_rows.extend((
        {"stage": "fold2_application", **distance2},
        {"stage": "fold2_application", **full2},
    ))
    application = {
        "distance_only": distance2,
        "full_3d_tortuosity": full2,
        "gain": distance2["rmse"] - full2["rmse"],
        "far_gain": distance2["far_gt2000_rmse"] - full2["far_gt2000_rmse"],
    }
    pd.concat([fold2_distance, fold2_full], ignore_index=True).to_csv(
        OUT_DIR / "fold2_bywell.csv", index=False
    )

pd.DataFrame(scan_rows).sort_values("rmse").to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
pd.concat(discovery_outputs, ignore_index=True).to_csv(
    OUT_DIR / "fold3_discovery_bywell.csv", index=False
)
pd.concat(confirmation_frames.values(), ignore_index=True).to_csv(
    OUT_DIR / "fold3_confirmation_bywell.csv", index=False
)
summary_frame = pd.DataFrame(summary_rows)
summary_frame.to_csv(OUT_DIR / "summary.csv", index=False)

strict_pass = bool(
    mechanism_pass
    and application is not None
    and application["gain"] >= 0.20
    and application["far_gain"] >= 0.20
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_training_Fold3_active_steering_attribution_Fold2_conditional",
    "sample_unit": "complete physical well",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "uses_gr_in_model": False,
    "uses_absolute_xy_location": False,
    "feature_families": {
        "distance_only": len(base_columns),
        "z_trajectory": len(z_columns),
        "full_3d_tortuosity": len(full_columns),
        "tortuosity_definition": "MD span / 3-D chord - 1 (transparent proxy)",
        "windows_ft": list(WINDOWS_FT),
    },
    "protocol": {
        "train_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "discovery_wells": len(discovery_cache),
        "confirmation_wells": len(confirmation_cache),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "selected_config": {
        **selected_config,
        "smooth_nodes": selected_smooth,
        "best_iteration": int(full_model.best_iteration_),
    },
    "confirmation": confirmation_summary,
    "confirmation_effects": {
        "gain_vs_distance": distance_control["rmse"] - full["rmse"],
        "far_gain_vs_distance": distance_control["far_gt2000_rmse"] - full["far_gt2000_rmse"],
        "gain_vs_z": z_control["rmse"] - full["rmse"],
        "shuffle_loss": shuffle_control["rmse"] - full["rmse"],
    },
    "gates": gates,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> cross-fit the trajectory endpoint and audit decorrelation from 6.377"
        if strict_pass
        else "FAIL -> drilled geometry does not expose future formation trend; close active-steering branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV54 DISCOVERY SCAN")
print(scan.to_string(index=False), flush=True)
print("\nV54 SUMMARY")
print(summary_frame.to_string(index=False), flush=True)
print("\nV54 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
