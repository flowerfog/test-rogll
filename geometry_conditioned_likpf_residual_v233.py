"""V233: test anchor-relative geometry as conditioning for the likPF residual.

This experiment deliberately keeps the V215 likPF parent and five well folds.
Only legal inference-time geometry and visible-prefix S=TVT_input+Z summaries are
added.  The model predicts truth-minus-likPF, never absolute TVT.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
V215_DIR = ROOT / "oof_exports" / "likpf_residual_gbdt_full5_v215"
OOF_PATH = V215_DIR / "oof.parquet"
OUT_DIR = ROOT / "oof_exports" / "geometry_conditioned_likpf_residual_v233"

SEED = 233
SAMPLE_ROWS_PER_WELL = 1024
WARM_TAU = 85.0
WARM_FLOOR = 0.15
SMOOTH_WINDOWS = (1, 61, 181)
CAPS = (4.0, 8.0, 12.0, 20.0)
DOSES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - np.asarray(prediction)))))


def _locate_horizontal(well: str) -> Path:
    candidates = (
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        ROOT / "data" / "train" / f"{well}__horizontal_well.csv",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"horizontal file missing for {well}: {candidates}")
    return path


def _slope(md: np.ndarray, value: np.ndarray, window_ft: float) -> float:
    if len(md) < 2:
        return 0.0
    keep = md >= md[-1] - window_ft
    x = md[keep].astype(np.float64)
    y = value[keep].astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.ptp(x) < 1e-6:
        return 0.0
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0 else 0.0


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    return (
        pd.Series(values.astype(np.float64, copy=False))
        .rolling(window, center=True, min_periods=max(3, window // 5))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(np.float64)
    )


def _smooth_by_well(frame: pd.DataFrame, values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    result = np.empty(len(frame), np.float64)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        result[positions] = _rolling_mean(values[positions], window)
    return result


def _well_features(group: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    well = str(group["well"].iloc[0])
    horizontal = pd.read_csv(
        _locate_horizontal(well),
        usecols=lambda name: name in {"MD", "X", "Y", "Z", "TVT_input"},
    )
    required = {"MD", "Z", "TVT_input"}
    if not required.issubset(horizontal.columns):
        raise RuntimeError(f"{well}: missing columns {sorted(required - set(horizontal.columns))}")

    input_tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(input_tvt))
    if not len(known):
        raise RuntimeError(f"{well}: no visible TVT_input")
    anchor = int(known[-1])

    row_idx = group["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int64).to_numpy()
    if np.any(row_idx < 0) or np.any(row_idx >= len(horizontal)):
        raise RuntimeError(f"{well}: OOF row index outside horizontal file")

    md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(md_all).all() or not np.isfinite(z_all).all():
        raise RuntimeError(f"{well}: non-finite MD/Z")

    prefix_md = md_all[: anchor + 1]
    prefix_s = input_tvt[: anchor + 1] + z_all[: anchor + 1]
    slopes = {window: _slope(prefix_md, prefix_s, float(window)) for window in (128, 256, 512, 1024)}
    slope_values = np.asarray(list(slopes.values()), np.float64)

    md = md_all[row_idx]
    z = z_all[row_idx]
    md_since = md - md_all[anchor]
    dz = z - z_all[anchor]
    span = max(float(md_since[-1]) if len(md_since) else 0.0, 1.0)
    likpf = group["likpf"].to_numpy(np.float64)
    last_tvt = float(input_tvt[anchor])
    lik_delta = likpf - last_tvt

    z_gradient = np.gradient(z_all, md_all, edge_order=1)
    slope32 = _rolling_mean(z_gradient, 33)[row_idx]
    slope128 = _rolling_mean(z_gradient, 129)[row_idx]
    slope512 = _rolling_mean(z_gradient, 513)[row_idx]
    curvature128 = np.gradient(_rolling_mean(z_gradient, 129), md_all, edge_order=1)[row_idx]

    data = {
        "md_since": md_since,
        "progress": md_since / span,
        "dz_anchor": dz,
        "dz_rate_anchor": dz / np.maximum(md_since, 1.0),
        "z_slope_32": slope32,
        "z_slope_128": slope128,
        "z_slope_512": slope512,
        "z_slope_spread": np.maximum.reduce((slope32, slope128, slope512))
        - np.minimum.reduce((slope32, slope128, slope512)),
        "z_curvature_128": curvature128,
        "prefix_s_slope_mean": np.full(len(group), slope_values.mean()),
        "prefix_s_slope_std": np.full(len(group), slope_values.std()),
        "prefix_s_slope_range": np.full(len(group), np.ptp(slope_values)),
        "likpf_delta": lik_delta,
    }
    for window, slope in slopes.items():
        extrapolated_ds = slope * md_since
        geometry_delta = extrapolated_ds - dz
        data[f"prefix_s_slope_{window}"] = np.full(len(group), slope)
        data[f"surface_extrap_{window}"] = extrapolated_ds
        data[f"geometry_tvt_delta_{window}"] = geometry_delta
        data[f"likpf_minus_geometry_{window}"] = lik_delta - geometry_delta

    if {"X", "Y"}.issubset(horizontal.columns):
        x_all = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(np.float64)
        y_all = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(np.float64)
        dx, dy = x_all[row_idx] - x_all[anchor], y_all[row_idx] - y_all[anchor]
        data["dx_anchor"] = dx
        data["dy_anchor"] = dy
        data["dxy_anchor"] = np.hypot(dx, dy)
        data["z_over_dxy"] = dz / np.maximum(np.hypot(dx, dy), 1.0)

    features = pd.DataFrame(data, index=group.index).astype(np.float32)
    audit = {
        "well": well,
        "rows": len(group),
        "corr_dz_likpf_delta": float(np.corrcoef(dz, lik_delta)[0, 1]) if len(group) > 2 else np.nan,
        **{f"prefix_s_slope_{window}": slope for window, slope in slopes.items()},
    }
    return features, audit


def _sample_positions(group: pd.DataFrame, maximum: int) -> np.ndarray:
    if len(group) <= maximum:
        return group.index.to_numpy(np.int64)
    local = np.unique(np.linspace(0, len(group) - 1, maximum).round().astype(np.int64))
    return group.index.to_numpy(np.int64)[local]


def _summary(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    well_rmse = frame.assign(_se=np.square(error)).groupby("well", sort=False)["_se"].mean().pow(0.5)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "far_gt2000_rmse": _rmse(truth[far], prediction[far]) if far.any() else np.nan,
    }


def main():
    started = time.time()
    if not OOF_PATH.is_file():
        raise FileNotFoundError(OOF_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(OOF_PATH)
    required = {"id", "well", "fold", "md_since", "tvt_true", "likpf"}
    if not required.issubset(base.columns):
        raise RuntimeError(f"V215 OOF missing {sorted(required - set(base.columns))}")
    base = base.sort_values(["well", "md_since", "id"], kind="stable").reset_index(drop=True)
    if len(base) != 3_783_989 or base["well"].nunique() != 773:
        raise RuntimeError(f"Unexpected V215 coverage: rows={len(base)} wells={base['well'].nunique()}")
    base["well"] = base["well"].astype(str)

    print("V233 building legal geometry features", flush=True)
    feature_parts, audits = [], []
    groups = list(base.groupby("well", sort=False))
    for number, (_, group) in enumerate(groups, 1):
        features, audit = _well_features(group)
        feature_parts.append(features)
        audits.append(audit)
        if number % 100 == 0 or number == len(groups):
            print(f"V233 features {number}/{len(groups)}", flush=True)
    feature_frame = pd.concat(feature_parts).sort_index()
    if len(feature_frame) != len(base) or feature_frame.isna().any().any():
        raise RuntimeError("V233 feature alignment or finite-value failure")
    feature_names = feature_frame.columns.tolist()
    feature_hash = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()

    sample_parts, sample_weight_parts = [], []
    for _, group in base.groupby("well", sort=False):
        positions = _sample_positions(group, SAMPLE_ROWS_PER_WELL)
        sample_parts.append(positions)
        sample_weight_parts.append(np.full(len(positions), len(group) / len(positions), np.float32))
    sample_indices = np.concatenate(sample_parts)
    sample_weights = np.concatenate(sample_weight_parts)
    train_x_all = feature_frame.iloc[sample_indices].to_numpy(np.float32)
    residual = base["tvt_true"].to_numpy(np.float64) - base["likpf"].to_numpy(np.float64)
    warm = 1.0 - np.exp(-np.maximum(base["md_since"].to_numpy(np.float64), 0.0) / WARM_TAU)
    normalized_residual = np.clip(residual / np.maximum(warm, WARM_FLOOR), -80.0, 80.0)
    train_y_all = normalized_residual[sample_indices].astype(np.float32)
    sample_folds = base["fold"].to_numpy(np.int8)[sample_indices]

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("lightgbm is required") from error

    use_gpu = os.environ.get("V233_DEVICE", "gpu").lower() == "gpu"
    params = dict(
        objective="regression_l2",
        learning_rate=0.025,
        n_estimators=2500,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=8.0,
        max_bin=255,
        force_col_wise=True,
        n_jobs=8,
        verbosity=-1,
        random_state=SEED,
    )
    if use_gpu:
        params.update(device_type="gpu", gpu_use_dp=False)

    correction = np.full(len(base), np.nan, np.float32)
    fold_models = []
    for held_fold in range(1, 6):
        train_mask = sample_folds != held_fold
        valid_mask = base["fold"].to_numpy(np.int8) == held_fold
        valid_x = feature_frame.loc[valid_mask, feature_names].to_numpy(np.float32)
        valid_y = normalized_residual[valid_mask].astype(np.float32)
        model = lgb.LGBMRegressor(**params)
        try:
            model.fit(
                train_x_all[train_mask],
                train_y_all[train_mask],
                sample_weight=sample_weights[train_mask],
                eval_set=[(valid_x, valid_y)],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
        except Exception as error:
            if not use_gpu:
                raise
            print(f"V233 GPU fallback on fold {held_fold}: {error!r}", flush=True)
            cpu_params = {key: value for key, value in params.items() if key not in {"device_type", "gpu_use_dp"}}
            model = lgb.LGBMRegressor(**cpu_params)
            model.fit(
                train_x_all[train_mask],
                train_y_all[train_mask],
                sample_weight=sample_weights[train_mask],
                eval_set=[(valid_x, valid_y)],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
        predicted = model.predict(valid_x, num_iteration=model.best_iteration_).astype(np.float32)
        correction[valid_mask] = predicted
        fold_models.append(
            {
                "fold": held_fold,
                "best_iteration": int(model.best_iteration_),
                "residual_rmse": _rmse(valid_y, predicted),
            }
        )
        print("V233 fold", fold_models[-1], flush=True)
        del valid_x, valid_y, model
        gc.collect()

    if not np.isfinite(correction).all():
        raise RuntimeError("Incomplete V233 OOF correction")

    effective_correction = warm * correction.astype(np.float64)
    correction_by_window = {
        window: _smooth_by_well(base, effective_correction, window) for window in SMOOTH_WINDOWS
    }
    truth = base["tvt_true"].to_numpy(np.float64)
    parent = base["likpf"].to_numpy(np.float64)
    folds = base["fold"].to_numpy(np.int8)

    scan_rows = []
    predictions = {}
    for window, smooth in correction_by_window.items():
        for cap in CAPS:
            bounded = np.clip(smooth, -cap, cap)
            for dose in DOSES:
                key = (window, cap, dose)
                prediction = parent + dose * bounded
                predictions[key] = prediction
                row = {"window": window, "cap": cap, "dose": dose, "rmse": _rmse(truth, prediction)}
                for fold in range(1, 6):
                    mask = folds == fold
                    row[f"fold{fold}_rmse"] = _rmse(truth[mask], prediction[mask])
                scan_rows.append(row)
    scan = pd.DataFrame(scan_rows).sort_values("rmse").reset_index(drop=True)

    nested = np.empty(len(base), np.float64)
    selected = []
    for held_fold in range(1, 6):
        donor_folds = [fold for fold in range(1, 6) if fold != held_fold]
        donor_sse = np.zeros(len(scan), np.float64)
        donor_rows = 0
        for fold in donor_folds:
            count = int((folds == fold).sum())
            donor_rows += count
            donor_sse += np.square(scan[f"fold{fold}_rmse"].to_numpy(np.float64)) * count
        chosen = scan.iloc[int(np.argmin(donor_sse / donor_rows))]
        key = (int(chosen.window), float(chosen.cap), float(chosen.dose))
        mask = folds == held_fold
        nested[mask] = predictions[key][mask]
        selected.append(
            {
                "fold": held_fold,
                "window": key[0],
                "cap": key[1],
                "dose": key[2],
                "base_rmse": _rmse(truth[mask], parent[mask]),
                "candidate_rmse": _rmse(truth[mask], nested[mask]),
            }
        )
        selected[-1]["gain"] = selected[-1]["base_rmse"] - selected[-1]["candidate_rmse"]

    base_summary = _summary(base, parent, "likpf_parent")
    nested_summary = _summary(base, nested, "geometry_conditioned_nested")
    gain = base_summary["rmse"] - nested_summary["rmse"]
    positive_folds = sum(item["gain"] > 0 for item in selected)

    output = base[["id", "well", "fold", "md_since", "tvt_true", "likpf"]].copy()
    output["geometry_residual_normalized_raw"] = correction
    output["geometry_residual_effective_raw"] = effective_correction.astype(np.float32)
    output["geometry_conditioned_nested"] = nested.astype(np.float32)
    output.to_parquet(OUT_DIR / "oof.parquet", index=False)
    scan.to_csv(OUT_DIR / "dose_scan.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT_DIR / "well_geometry_audit.csv", index=False)
    pd.DataFrame(selected).to_csv(OUT_DIR / "folds.csv", index=False)
    pd.Series(feature_names, name="feature").to_csv(OUT_DIR / "features.csv", index=False)

    verdict = {
        "experiment": "geometry_conditioned_likpf_residual_v233",
        "stage": "fivefold_anchor_relative_geometry_conditioning_of_V215_likPF_residual",
        "target": "truth minus frozen likPF",
        "legal_new_information": [
            "future X/Y/Z trajectory relative to visible anchor",
            "visible-prefix S=TVT_input+Z multiscale slopes",
            "bit-versus-prefix-bed extrapolation disagreements",
        ],
        "features": len(feature_names),
        "feature_sha256": feature_hash,
        "sample_rows_per_well": SAMPLE_ROWS_PER_WELL,
        "training_rows": len(sample_indices),
        "coverage": {"rows": len(base), "wells": int(base.well.nunique())},
        "models": fold_models,
        "base": base_summary,
        "candidate": nested_summary,
        "gain": gain,
        "folds": selected,
        "checks": {
            "gain_ge_0p10": gain >= 0.10,
            "positive_folds_ge_4": positive_folds >= 4,
            "minimum_fold_gain_ge_minus0p03": min(item["gain"] for item in selected) >= -0.03,
            "candidate_rmse_le_9": nested_summary["rmse"] <= 9.0,
        },
        "strict_pass": bool(
            gain >= 0.10
            and positive_folds >= 4
            and min(item["gain"] for item in selected) >= -0.03
        ),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    verdict["next"] = (
        "PASS -> merge these exact geometry columns into the stronger V215/V217 representation"
        if verdict["strict_pass"]
        else "FAIL -> explicit cumulative geometry does not explain the standalone GBDT gap"
    )
    (OUT_DIR / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )

    print("\nV233 TOP 20")
    print(scan.head(20).to_string(index=False))
    print("\nV233 FOLDS")
    print(pd.DataFrame(selected).to_string(index=False))
    print("\nV233 FINAL VERDICT")
    print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default))
    print("saved:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
