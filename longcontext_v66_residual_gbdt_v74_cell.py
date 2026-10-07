"""Long-context candidate-conditioned residual GBDT on the complete V66 OOF.

This is the missing experiment between the old zero-training alignment audit
and the old eight-control-point CP8 ablations.  It starts from the exact
773-well V66 incumbent, samples 24 control stations per complete well, and
asks whether long within-well context can predict a conservative correction.

Protocol
--------
* outer predictions are five-fold, complete-well OOF;
* Fold 4/5 select one correction recipe;
* Fold 1/2/3 are frozen confirmation folds;
* truth is used only as the residual training label and for metrics;
* all model features are available from the horizontal well, typewell,
  known prefix, geometry, and the already-produced parent trajectory;
* geometry-only and station-shuffled long-context controls are matched.

Run on AutoDL after V66.  The feature cache is resumable.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.ndimage import gaussian_filter1d


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "longcontext_v66_residual_gbdt_v74"
OUT_DIR = OOF_ROOT / EXPERIMENT
FEATURE_DIR = OUT_DIR / "features"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, FEATURE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V66_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
DATA_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    ROOT / "replay_data",
    ROOT / "data",
)
DATA_ROOT = next(
    (path for path in DATA_CANDIDATES if (path / "train").is_dir()), None
)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Cannot find raw train directory under: "
        + repr([str(path) for path in DATA_CANDIDATES])
    )
if not V66_PATH.is_file():
    raise FileNotFoundError(f"Run V66 first: {V66_PATH}")

EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
SELECTION_FOLDS = (4, 5)
CONFIRMATION_FOLDS = (1, 2, 3)
BASE_COLUMN = "incumbent_6p377_replay"
N_CONTROLS = 24
OFFSETS_FT = np.array(
    [-64, -48, -32, -24, -16, -8, 0, 8, 16, 24, 32, 48, 64],
    dtype=np.float64,
)
WINDOWS_FT = (64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0)
PROFILE_WINDOWS_FT = (128.0, 512.0, 2048.0)
DERIVATIVE_WINDOWS_FT = (256.0, 1024.0)
MIN_WINDOW_ROWS = 12
MIN_WINDOW_FRACTION = 0.15
SEED = 20260729
CAPS_FT = (2.0, 4.0, 8.0, 12.0)
DOSES = (0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
STARTS_FT = (0.0, 500.0, 1000.0)
TRANSITION_FT = 512.0
P95_TOLERANCE_FT = 0.20


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _robust_location_scale(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-3)
    return centre, scale


def _robust_affine(source, target):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    valid = np.isfinite(source) & np.isfinite(target)
    x, y = source[valid], target[valid]
    if len(x) < 32:
        shift = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, shift, int(len(x)), np.nan
    keep = np.ones(len(x), bool)
    gain, intercept = 1.0, 0.0
    for _ in range(5):
        design = np.c_[x[keep], np.ones(int(keep.sum()))]
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (gain * x + intercept)
        centre, scale = _robust_location_scale(residual[keep])
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if int(next_keep.sum()) < 32 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    gain = float(np.clip(gain, 0.15, 6.0))
    residual = y[keep] - (gain * x[keep] + intercept)
    rmse = float(np.sqrt(np.mean(residual * residual))) if len(residual) else np.nan
    return gain, float(intercept), int(keep.sum()), rmse


def _deduplicate_axis(axis, values):
    table = pd.DataFrame({"axis": axis, "value": values}).dropna()
    table = table.groupby("axis", as_index=False)["value"].mean().sort_values("axis")
    return table["axis"].to_numpy(np.float64), table["value"].to_numpy(np.float64)


def _fill_1d(axis, values):
    axis = np.asarray(axis, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.isfinite(axis) & np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros(len(values), np.float64)
    order = np.argsort(axis[valid])
    return np.interp(axis, axis[valid][order], values[valid][order])


def _gradient(values, axis):
    values = np.asarray(values, np.float64)
    axis = np.asarray(axis, np.float64).copy()
    spacing = np.diff(axis)
    positive = spacing[np.isfinite(spacing) & (spacing > 1e-6)]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    for index in range(1, len(axis)):
        if not np.isfinite(axis[index]) or axis[index] <= axis[index - 1]:
            axis[index] = axis[index - 1] + fallback
    return np.gradient(values, axis, axis=0)


def _ncc(x, y, valid):
    if int(valid.sum()) < MIN_WINDOW_ROWS:
        return np.nan
    a = x[valid].astype(np.float64)
    b = y[valid].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator < 1e-8:
        return np.nan
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _costs(observed, expected, valid):
    count = int(valid.sum())
    if count < MIN_WINDOW_ROWS:
        return np.nan, np.nan, np.nan, count
    difference = np.clip(observed[valid] - expected[valid], -8.0, 8.0)
    mae = float(np.mean(np.abs(difference)))
    cauchy = float(np.mean(np.log1p(difference * difference)))
    return mae, cauchy, _ncc(observed, expected, valid), count


def _offset_name(value):
    value = int(value)
    return f"m{abs(value)}" if value < 0 else (f"p{value}" if value > 0 else "z0")


def _weighted_offset_summary(cost):
    cost = np.asarray(cost, np.float64)
    valid = np.isfinite(cost)
    if valid.sum() < 3:
        return (np.nan,) * 8
    local_cost = cost[valid]
    local_offset = OFFSETS_FT[valid]
    order = np.argsort(local_cost)
    best = float(local_offset[order[0]])
    minimum = float(local_cost[order[0]])
    gap = float(local_cost[order[1]] - local_cost[order[0]])
    temperature = max(float(np.nanmedian(np.abs(local_cost - np.nanmedian(local_cost)))), 0.05)
    weight = np.exp(-(local_cost - minimum) / temperature)
    weight /= max(float(weight.sum()), 1e-12)
    mean = float(np.dot(weight, local_offset))
    std = float(np.sqrt(np.dot(weight, (local_offset - mean) ** 2)))
    zero = float(cost[np.flatnonzero(OFFSETS_FT == 0)[0]])
    alias24 = float(np.nanmin(cost[np.isin(OFFSETS_FT, (-24.0, 24.0))]))
    edge = float(np.nanmin(cost[np.isin(OFFSETS_FT, (-64.0, 64.0))]))
    return best, minimum, gap, mean, std, zero, alias24 - zero, edge - zero


def _trailing_indices(md, centre, window_ft):
    centre_md = float(md[centre])
    left = int(np.searchsorted(md, centre_md - float(window_ft), side="left"))
    right = centre + 1
    return np.arange(max(0, left), right, dtype=np.int64)


def _slope(values, md, indices):
    if len(indices) < 2:
        return 0.0
    denominator = float(md[indices[-1]] - md[indices[0]])
    if abs(denominator) < 1e-6:
        return 0.0
    return float((values[indices[-1]] - values[indices[0]]) / denominator)


def _control_positions(n_rows):
    if n_rows < 64:
        return np.array([], np.int64)
    start = min(16, max(0, n_rows // 50))
    positions = np.linspace(start, n_rows - 1, N_CONTROLS)
    return np.unique(np.round(positions).astype(np.int64))


def _build_well_features(well, local):
    cache_path = FEATURE_DIR / f"{well}.parquet"
    if cache_path.is_file():
        cached = pd.read_parquet(cache_path)
        if len(cached) >= max(16, N_CONTROLS - 2):
            return cached

    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    required_horizontal = {"MD", "Z", "GR", "TVT", "TVT_input"}
    if not required_horizontal.issubset(horizontal.columns):
        raise RuntimeError(f"horizontal missing {sorted(required_horizontal - set(horizontal.columns))}")
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise RuntimeError("typewell missing TVT/GR")

    row_idx = local["row_idx"].to_numpy(np.int64)
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise RuntimeError("row_idx outside horizontal file")
    md_full = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z_full = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr_full = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    truth_full = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.isfinite(visible)
    if known.sum() < 32:
        raise RuntimeError("too few known-prefix rows")
    truth_gap = np.abs(truth_full[row_idx] - local["tvt_true"].to_numpy(np.float64))
    if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.02:
        raise RuntimeError(f"truth alignment gap {float(np.nanmax(truth_gap))}")

    md = md_full[row_idx]
    z = z_full[row_idx]
    observed_gr = gr_full[row_idx]
    truth = local["tvt_true"].to_numpy(np.float64)
    parent = local[BASE_COLUMN].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    if np.any(np.diff(md) < -1e-6):
        raise RuntimeError("evaluation MD is not monotone")

    tw_tvt, tw_gr = _deduplicate_axis(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    if len(tw_tvt) < 32:
        raise RuntimeError("too few typewell rows")
    tw_at_known = np.interp(visible[known], tw_tvt, tw_gr)
    affine_gain, affine_intercept, affine_pairs, affine_rmse = _robust_affine(
        tw_at_known, gr_full[known]
    )
    calibrated_tw_gr = affine_gain * tw_gr + affine_intercept
    gr_centre, gr_scale = _robust_location_scale(gr_full[known])
    observed_norm = (observed_gr - gr_centre) / gr_scale
    tw_norm = (calibrated_tw_gr - gr_centre) / gr_scale

    expected = np.column_stack(
        [
            np.interp(parent + offset, tw_tvt, tw_norm, left=np.nan, right=np.nan)
            for offset in OFFSETS_FT
        ]
    ).astype(np.float64)
    observed_filled = _fill_1d(md, observed_norm)
    observed_derivative = _gradient(observed_filled, md)
    expected_derivative = _gradient(expected, md)
    parent_surface = parent + z
    controls = _control_positions(len(local))
    if len(controls) < 16:
        raise RuntimeError("too few control positions")

    records = []
    for order, centre in enumerate(controls):
        record = {
            "well": str(well),
            "fold": int(local["fold"].iloc[0]),
            "control_order": int(order),
            "center_local": int(centre),
            "row_idx": int(row_idx[centre]),
            "target_residual": float(truth[centre] - parent[centre]),
            "parent_tvt": float(parent[centre]),
            "GEO_md_since": float(distance[centre]),
            "GEO_log1p_md_since": float(np.log1p(max(distance[centre], 0.0))),
            "GEO_progress": float(centre / max(len(local) - 1, 1)),
            "GEO_eval_rows": float(len(local)),
            "GEO_parent_from_start": float(parent[centre] - parent[0]),
            "GEO_surface_from_start": float(parent_surface[centre] - parent_surface[0]),
            "GEO_z_from_start": float(z[centre] - z[0]),
            "CAL_affine_gain": float(affine_gain),
            "CAL_affine_intercept_scaled": float(affine_intercept / gr_scale),
            "CAL_pairs": float(affine_pairs),
            "CAL_rmse_scaled": float(affine_rmse / gr_scale) if np.isfinite(affine_rmse) else np.nan,
            "CAL_eval_gr_coverage": float(np.isfinite(observed_gr).mean()),
            "CAL_prefix_gr_coverage": float(np.isfinite(gr_full[known]).mean()),
        }
        for window_ft in WINDOWS_FT:
            indices = _trailing_indices(md, centre, window_ft)
            record[f"GEO_parent_slope_w{int(window_ft)}"] = _slope(parent, md, indices)
            record[f"GEO_surface_slope_w{int(window_ft)}"] = _slope(parent_surface, md, indices)
            record[f"GEO_z_slope_w{int(window_ft)}"] = _slope(z, md, indices)
            observed = observed_norm[indices]
            profile_mae, profile_cauchy, profile_ncc = [], [], []
            counts = []
            for offset_index, offset in enumerate(OFFSETS_FT):
                predicted = expected[indices, offset_index]
                valid = np.isfinite(observed) & np.isfinite(predicted)
                minimum = max(MIN_WINDOW_ROWS, int(math.ceil(MIN_WINDOW_FRACTION * len(indices))))
                if int(valid.sum()) < minimum:
                    mae = cauchy = ncc = np.nan
                    count = int(valid.sum())
                else:
                    mae, cauchy, ncc, count = _costs(observed, predicted, valid)
                profile_mae.append(mae)
                profile_cauchy.append(cauchy)
                profile_ncc.append(ncc)
                counts.append(count)
                if window_ft in PROFILE_WINDOWS_FT:
                    suffix = _offset_name(offset)
                    record[f"LC_mae_w{int(window_ft)}_{suffix}"] = mae
                    record[f"LC_cauchy_w{int(window_ft)}_{suffix}"] = cauchy
                    record[f"LC_ncc_w{int(window_ft)}_{suffix}"] = ncc
            summary = _weighted_offset_summary(profile_cauchy)
            names = ("best", "min", "gap", "mean", "std", "zero", "alias24_delta", "edge_delta")
            for name, value in zip(names, summary):
                record[f"LC_cauchy_{name}_w{int(window_ft)}"] = value
            mae_summary = _weighted_offset_summary(profile_mae)
            for name, value in zip(names, mae_summary):
                record[f"LC_mae_{name}_w{int(window_ft)}"] = value
            ncc_cost = -np.asarray(profile_ncc, np.float64)
            ncc_summary = _weighted_offset_summary(ncc_cost)
            for name, value in zip(names, ncc_summary):
                record[f"LC_ncc_{name}_w{int(window_ft)}"] = value
            record[f"LC_support_w{int(window_ft)}"] = float(np.max(counts) / max(len(indices), 1))

        for window_ft in DERIVATIVE_WINDOWS_FT:
            indices = _trailing_indices(md, centre, window_ft)
            derivative_cost = []
            for offset_index in range(len(OFFSETS_FT)):
                predicted = expected_derivative[indices, offset_index]
                observed = observed_derivative[indices]
                valid = np.isfinite(predicted) & np.isfinite(observed) & np.isfinite(observed_gr[indices])
                if int(valid.sum()) < MIN_WINDOW_ROWS:
                    derivative_cost.append(np.nan)
                else:
                    difference = np.clip(observed[valid] - predicted[valid], -1.0, 1.0)
                    derivative_cost.append(float(np.mean(np.log1p(16.0 * difference * difference))))
            summary = _weighted_offset_summary(derivative_cost)
            names = ("best", "min", "gap", "mean", "std", "zero", "alias24_delta", "edge_delta")
            for name, value in zip(names, summary):
                record[f"LC_derivative_{name}_w{int(window_ft)}"] = value
        records.append(record)

    result = pd.DataFrame(records)
    result.to_parquet(cache_path, index=False, compression="zstd")
    return result


def _metrics(local, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = local["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    distance = local["md_since"].to_numpy(np.float64)
    per_well = (
        pd.DataFrame({"well": local["well"].astype(str), "sse": error * error})
        .groupby("well", sort=False)["sse"]
        .mean()
        .pow(0.5)
    )
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def _fit_arm(controls, feature_columns, arm):
    matrix = controls[feature_columns].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    target = controls["target_residual"].to_numpy(np.float32)
    folds = controls["fold"].to_numpy(np.int8)
    output = np.full(len(controls), np.nan, np.float32)
    reports = []
    for held_fold in EXPECTED_FOLDS:
        train = np.flatnonzero(folds != held_fold)
        valid = np.flatnonzero(folds == held_fold)
        model = LGBMRegressor(
            objective="huber",
            alpha=0.90,
            n_estimators=1800,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=8.0,
            random_state=SEED + held_fold,
            n_jobs=8,
            verbosity=-1,
        )
        model.fit(
            matrix.iloc[train], target[train],
            eval_set=[(matrix.iloc[valid], target[valid])],
            eval_metric="rmse",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        output[valid] = model.predict(matrix.iloc[valid]).astype(np.float32)
        joblib.dump(
            {"model": model, "features": feature_columns, "arm": arm, "held_fold": held_fold},
            MODEL_DIR / f"{arm}_hold{held_fold}.joblib",
            compress=3,
        )
        reports.append({
            "arm": arm,
            "fold": int(held_fold),
            "features": int(len(feature_columns)),
            "controls": int(len(valid)),
            "best_iteration": int(model.best_iteration_ or 0),
            "control_rmse": float(np.sqrt(np.mean((output[valid] - target[valid]) ** 2))),
        })
    if not np.isfinite(output).all():
        raise RuntimeError(f"Non-finite crossfit controls for {arm}")
    return output, reports


def _expand_controls(frame, controls, values):
    output = np.zeros(len(frame), np.float64)
    frame_groups = {
        str(well): np.asarray(indices, np.int64)
        for well, indices in frame.groupby("well", sort=False).indices.items()
    }
    for well, local_controls in controls.groupby("well", sort=False):
        positions = frame_groups.get(str(well), np.array([], np.int64))
        if not len(positions):
            continue
        local_controls = local_controls.sort_values("center_local")
        x = local_controls["center_local"].to_numpy(np.float64)
        y = values[local_controls.index.to_numpy(np.int64)].astype(np.float64)
        x = np.r_[0.0, x]
        y = np.r_[0.0, y]
        keep = np.r_[True, np.diff(x) > 1e-9]
        curve = np.interp(np.arange(len(positions), dtype=np.float64), x[keep], y[keep])
        sigma = max(4.0, len(positions) / N_CONTROLS / 6.0)
        output[positions] = gaussian_filter1d(curve, sigma=sigma, mode="nearest")
    return output


def _gate(distance, start_ft):
    if start_ft <= 0:
        return np.ones(len(distance), np.float64)
    t = np.clip((distance - start_ft) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply_recipe(base, raw_correction, distance, recipe):
    correction = np.clip(raw_correction, -recipe["cap_ft"], recipe["cap_ft"])
    return base + recipe["dose"] * _gate(distance, recipe["start_ft"]) * correction


frame = pd.read_parquet(V66_PATH)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V66 missing columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
if len(frame) != EXPECTED_ROWS or frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"V66 coverage mismatch: {len(frame)}/{frame['well'].nunique()}")
if frame["id"].duplicated().any():
    raise RuntimeError("V66 has duplicate ids")
if frame[["tvt_true", "md_since", BASE_COLUMN]].isna().any().any():
    raise RuntimeError("V66 contains non-finite core values")

feature_tables, failures = [], []
groups = frame.groupby("well", sort=False)
for number, (well, local) in enumerate(groups, 1):
    try:
        feature_tables.append(_build_well_features(str(well), local.reset_index(drop=True)))
    except Exception as exc:
        failures.append({"well": str(well), "reason": repr(exc)})
    if number % 50 == 0 or number == EXPECTED_WELLS:
        print(
            f"V74 features {number}/{EXPECTED_WELLS} success={len(feature_tables)} failures={len(failures)}",
            flush=True,
        )
pd.DataFrame(failures).to_csv(OUT_DIR / "feature_failures.csv", index=False)
controls = pd.concat(feature_tables, ignore_index=True)
controls["well"] = controls["well"].astype(str)
fold_wells = controls.groupby("fold")["well"].nunique().astype(int).to_dict()
if controls["well"].nunique() < 760 or min(fold_wells.values()) < 150:
    raise RuntimeError(f"Insufficient feature coverage: wells={controls['well'].nunique()} folds={fold_wells}")
controls = controls.sort_values(["well", "control_order"], kind="stable").reset_index(drop=True)
controls.to_parquet(OUT_DIR / "control_features.parquet", index=False, compression="zstd")

metadata_columns = {
    "well", "fold", "control_order", "center_local", "row_idx",
    "target_residual", "parent_tvt",
}
all_features = [
    column for column in controls.columns
    if column not in metadata_columns and pd.api.types.is_numeric_dtype(controls[column])
]
long_features = [column for column in all_features if column.startswith("LC_")]
geometry_features = [column for column in all_features if not column.startswith("LC_")]
if len(long_features) < 100 or len(geometry_features) < 20:
    raise RuntimeError(
        f"Unexpected feature schema: geometry={len(geometry_features)} long={len(long_features)}"
    )

shuffled = controls.copy()
for well, indices in shuffled.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    shift = max(1, len(indices) // 3 + (int(str(well)[:4], 16) % max(1, len(indices) // 3)))
    shuffled.loc[indices, long_features] = np.roll(
        shuffled.loc[indices, long_features].to_numpy(copy=True), shift=shift, axis=0
    )

arm_controls = {
    "geometry_only": (controls, geometry_features),
    "aligned_longcontext": (controls, geometry_features + long_features),
    "station_shuffled_longcontext": (shuffled, geometry_features + long_features),
}
raw_predictions, model_reports = {}, []
for arm, (table, features) in arm_controls.items():
    print(f"V74 train {arm}: controls={len(table):,} features={len(features)}", flush=True)
    control_prediction, reports = _fit_arm(table, features, arm)
    model_reports.extend(reports)
    raw_predictions[arm] = _expand_controls(frame, controls, control_prediction)
pd.DataFrame(model_reports).to_csv(OUT_DIR / "model_folds.csv", index=False)

base = frame[BASE_COLUMN].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold_values = frame["fold"].to_numpy(np.int8)
selection_mask = np.isin(fold_values, SELECTION_FOLDS)
selection_rows = []
for start_ft in STARTS_FT:
    for cap_ft in CAPS_FT:
        for dose in DOSES:
            recipe = {"start_ft": start_ft, "cap_ft": cap_ft, "dose": dose}
            prediction = _apply_recipe(
                base, raw_predictions["aligned_longcontext"], distance, recipe
            )
            local_gains, local_p95 = [], []
            for fold in SELECTION_FOLDS:
                mask = fold_values == fold
                local = frame.loc[mask].reset_index(drop=True)
                base_metric = _metrics(local, base[mask], "base")
                metric = _metrics(local, prediction[mask], "candidate")
                local_gains.append(base_metric["rmse"] - metric["rmse"])
                local_p95.append(metric["p95_well_rmse"] - base_metric["p95_well_rmse"])
            base_metric = _metrics(frame.loc[selection_mask], base[selection_mask], "base")
            metric = _metrics(frame.loc[selection_mask], prediction[selection_mask], "candidate")
            selection_rows.append({
                **recipe,
                "rmse": metric["rmse"],
                "gain": base_metric["rmse"] - metric["rmse"],
                "minimum_fold_gain": float(min(local_gains)),
                "maximum_p95_change": float(max(local_p95)),
            })
selection = pd.DataFrame(selection_rows)
safe = selection[
    (selection["minimum_fold_gain"] >= 0.0)
    & (selection["maximum_p95_change"] <= P95_TOLERANCE_FT)
]
if safe.empty:
    selected = selection[selection["dose"].eq(0.0)].iloc[0]
else:
    selected = safe.sort_values(
        ["minimum_fold_gain", "gain", "dose"], ascending=[False, False, True]
    ).iloc[0]
recipe = {
    "start_ft": float(selected["start_ft"]),
    "cap_ft": float(selected["cap_ft"]),
    "dose": float(selected["dose"]),
}
selection.to_csv(OUT_DIR / "selection_scan.csv", index=False)
print("V74 SELECTED", recipe, flush=True)

endpoint_predictions = {"base": base}
for arm, raw in raw_predictions.items():
    endpoint_predictions[arm] = _apply_recipe(base, raw, distance, recipe)

metric_rows, fold_reports = [], []
for scope, scope_mask in {
    "selection45": np.isin(fold_values, SELECTION_FOLDS),
    "confirmation123": np.isin(fold_values, CONFIRMATION_FOLDS),
    "all": np.ones(len(frame), bool),
}.items():
    local = frame.loc[scope_mask].reset_index(drop=True)
    for endpoint, prediction in endpoint_predictions.items():
        metric_rows.append({"scope": scope, **_metrics(local, prediction[scope_mask], endpoint)})

for fold in EXPECTED_FOLDS:
    mask = fold_values == fold
    local = frame.loc[mask].reset_index(drop=True)
    base_metric = _metrics(local, base[mask], "base")
    aligned_metric = _metrics(local, endpoint_predictions["aligned_longcontext"][mask], "aligned")
    geometry_metric = _metrics(local, endpoint_predictions["geometry_only"][mask], "geometry")
    shuffle_metric = _metrics(local, endpoint_predictions["station_shuffled_longcontext"][mask], "shuffle")
    fold_reports.append({
        "fold": int(fold),
        "base_rmse": base_metric["rmse"],
        "aligned_rmse": aligned_metric["rmse"],
        "gain": base_metric["rmse"] - aligned_metric["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - aligned_metric["far_gt2000_rmse"],
        "p95_change": aligned_metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
        "increment_vs_geometry": geometry_metric["rmse"] - aligned_metric["rmse"],
        "increment_vs_shuffle": shuffle_metric["rmse"] - aligned_metric["rmse"],
    })
metrics = pd.DataFrame(metric_rows)
fold_report = pd.DataFrame(fold_reports)
metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)

confirmation = metrics[
    metrics["scope"].eq("confirmation123")
].set_index("endpoint")
base_confirmation = confirmation.loc["base"]
aligned_confirmation = confirmation.loc["aligned_longcontext"]
geometry_confirmation = confirmation.loc["geometry_only"]
shuffle_confirmation = confirmation.loc["station_shuffled_longcontext"]
confirmation_folds = fold_report[fold_report["fold"].isin(CONFIRMATION_FOLDS)]
confirmation_gain = float(base_confirmation["rmse"] - aligned_confirmation["rmse"])
confirmation_far_gain = float(
    base_confirmation["far_gt2000_rmse"] - aligned_confirmation["far_gt2000_rmse"]
)
increment_vs_geometry = float(geometry_confirmation["rmse"] - aligned_confirmation["rmse"])
increment_vs_shuffle = float(shuffle_confirmation["rmse"] - aligned_confirmation["rmse"])
checks = {
    "confirmation_gain_ge_0p10": confirmation_gain >= 0.10,
    "confirmation_far_gain_ge_0p10": confirmation_far_gain >= 0.10,
    "minimum_confirmation_fold_gain_ge_0": float(confirmation_folds["gain"].min()) >= 0.0,
    "aligned_increment_vs_geometry_ge_0p05": increment_vs_geometry >= 0.05,
    "aligned_increment_vs_shuffle_ge_0p05": increment_vs_shuffle >= 0.05,
    "confirmation_p95_change_le_0p20": float(confirmation_folds["p95_change"].max()) <= 0.20,
}
strict_pass = bool(all(checks.values()) and recipe["dose"] > 0.0)

if strict_pass:
    output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN]].copy()
    output["v74_raw_correction"] = raw_predictions["aligned_longcontext"].astype(np.float32)
    output["v74_prediction"] = endpoint_predictions["aligned_longcontext"].astype(np.float32)
    output.to_parquet(OUT_DIR / "v74_oof.parquet", index=False, compression="zstd")

print("\nV74 METRICS")
print(metrics.to_string(index=False))
print("\nV74 FOLDS")
print(fold_report.to_string(index=False))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "V66_long_context_candidate_conditioned_residual_GBDT",
    "sample_unit": "complete physical well with 24 trajectory control stations",
    "protocol": {
        "outer_folds": list(EXPECTED_FOLDS),
        "selection_folds": list(SELECTION_FOLDS),
        "confirmation_folds": list(CONFIRMATION_FOLDS),
        "selection_reused_on_confirmation": False,
        "parent": str(V66_PATH),
        "raw_data": str(DATA_ROOT),
    },
    "coverage": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "feature_wells": int(controls["well"].nunique()),
        "controls": int(len(controls)),
        "fold_wells": {str(key): int(value) for key, value in fold_wells.items()},
        "failures": int(len(failures)),
    },
    "features": {
        "geometry": int(len(geometry_features)),
        "long_context": int(len(long_features)),
        "windows_ft": list(WINDOWS_FT),
        "profile_windows_ft": list(PROFILE_WINDOWS_FT),
        "offsets_ft": OFFSETS_FT.tolist(),
    },
    "selected_recipe": recipe,
    "confirmation": {
        "base_rmse": float(base_confirmation["rmse"]),
        "rmse": float(aligned_confirmation["rmse"]),
        "gain": confirmation_gain,
        "base_far_rmse": float(base_confirmation["far_gt2000_rmse"]),
        "far_rmse": float(aligned_confirmation["far_gt2000_rmse"]),
        "far_gain": confirmation_far_gain,
        "increment_vs_geometry": increment_vs_geometry,
        "increment_vs_station_shuffle": increment_vs_shuffle,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": float((time.time() - STARTED) / 60.0),
    "next": (
        "PASS -> refit five deployable fold models and build the Kaggle feature generator bundle"
        if strict_pass
        else "FAIL -> long causal context is not consumed reliably by a residual GBDT; close this route"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_jsonable(verdict), indent=2), encoding="utf-8"
)
print("\nV74 FINAL VERDICT")
print(json.dumps(_jsonable(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
