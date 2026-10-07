"""V171: independent functional surface-trajectory engine.

Run after V166 in the same kernel. Unlike V76/V144 and all prior correction
heads, this engine consumes no historical prediction. It learns the complete
post-anchor geological surface trajectory S=TVT+Z from legal well descriptors:
geometry, visible-prefix state, horizontal GR and assigned typewell GR.

The output curve is represented by PCA coefficients and predicted in a strict
fivefold cross-fit using a linear and a nonlinear regressor. This is an
investment gateway for a genuinely independent second engine, not a deployable
blend search.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge


STARTED_V171 = time.time()
ROOT_V171 = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OUT_V171 = ROOT_V171 / "oof_exports" / "functional_surface_trajectory_engine_v171"
MODEL_V171 = OUT_V171 / "models"
for directory_v171 in (OUT_V171, MODEL_V171):
    directory_v171.mkdir(parents=True, exist_ok=True)

required_v171 = ("records", "wells", "base_frame", "prediction_column", "_metric")
missing_v171 = [name for name in required_v171 if name not in globals()]
if missing_v171:
    raise RuntimeError("Run V166 first. Missing: " + repr(missing_v171))

SEED_V171 = 20260804
CURVE_NODES_V171 = 192
CURVE_COMPONENTS_V171 = 24


def _clean_v171(value):
    if isinstance(value, dict):
        return {str(key): _clean_v171(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v171(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _descriptor_v171(record):
    return np.concatenate([
        np.asarray(record["anchor_xy"], np.float64) / 10000.0,
        np.asarray(record["direction"], np.float64),
        [float(record["length"]) / 5000.0],
        np.asarray(record["slopes"], np.float64) * 1000.0,
        np.asarray(record["surface_shape"], np.float64) / 100.0,
        np.asarray(record["z_shape"], np.float64) / 100.0,
        np.asarray(record["gr_shape"], np.float64),
        np.asarray(record["typewell_shape"], np.float64),
        np.asarray(record["future_gr_shape"], np.float64),
        np.asarray(record["future_gr_gradient"], np.float64),
    ])


def _curve_v171(record):
    progress = np.asarray(record["post_progress"], np.float64)
    delta = np.asarray(record["post_surface_delta"], np.float64)
    fraction = progress / max(float(progress[-1]), 1.0)
    curve = np.interp(np.linspace(0.0, 1.0, CURVE_NODES_V171), fraction, delta)
    curve -= curve[0]
    return curve


well_order_v171 = sorted(records)
features_v171 = np.stack([_descriptor_v171(records[well]) for well in well_order_v171])
curves_v171 = np.stack([_curve_v171(records[well]) for well in well_order_v171])
folds_v171 = np.asarray([records[well]["fold"] for well in well_order_v171], np.int8)
if len(well_order_v171) != 773 or not np.isfinite(curves_v171).all():
    raise RuntimeError("V171 record coverage or curve target is invalid")

prediction_curves_v171 = {
    "functional_ridge": np.full_like(curves_v171, np.nan),
    "functional_extra": np.full_like(curves_v171, np.nan),
    "functional_ensemble": np.full_like(curves_v171, np.nan),
}
model_reports_v171 = []
for held_fold_v171 in range(1, 6):
    train_v171 = folds_v171 != held_fold_v171
    valid_v171 = ~train_v171
    x_train_v171 = features_v171[train_v171]
    center_v171 = np.nanmedian(x_train_v171, axis=0)
    center_v171[~np.isfinite(center_v171)] = 0.0
    x_train_v171 = np.where(np.isfinite(x_train_v171), x_train_v171, center_v171)
    x_valid_v171 = np.where(
        np.isfinite(features_v171[valid_v171]), features_v171[valid_v171], center_v171
    )
    mean_v171 = np.mean(x_train_v171, axis=0)
    scale_v171 = np.maximum(np.std(x_train_v171, axis=0), 1e-3)
    x_train_v171 = np.clip((x_train_v171 - mean_v171) / scale_v171, -8.0, 8.0)
    x_valid_v171 = np.clip((x_valid_v171 - mean_v171) / scale_v171, -8.0, 8.0)

    pca_v171 = PCA(
        n_components=min(CURVE_COMPONENTS_V171, int(train_v171.sum()) - 1),
        random_state=SEED_V171 + held_fold_v171,
    )
    coefficient_train_v171 = pca_v171.fit_transform(curves_v171[train_v171])
    ridge_v171 = Ridge(alpha=100.0)
    ridge_v171.fit(x_train_v171, coefficient_train_v171)
    extra_v171 = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=6,
        max_features=0.65,
        n_jobs=-1,
        random_state=SEED_V171 + held_fold_v171,
    )
    extra_v171.fit(x_train_v171, coefficient_train_v171)
    ridge_curve_v171 = pca_v171.inverse_transform(ridge_v171.predict(x_valid_v171))
    extra_curve_v171 = pca_v171.inverse_transform(extra_v171.predict(x_valid_v171))
    ridge_curve_v171 -= ridge_curve_v171[:, :1]
    extra_curve_v171 -= extra_curve_v171[:, :1]
    prediction_curves_v171["functional_ridge"][valid_v171] = ridge_curve_v171
    prediction_curves_v171["functional_extra"][valid_v171] = extra_curve_v171
    prediction_curves_v171["functional_ensemble"][valid_v171] = (
        0.5 * ridge_curve_v171 + 0.5 * extra_curve_v171
    )
    report_v171 = {
        "heldout_fold": held_fold_v171,
        "train_wells": int(train_v171.sum()),
        "valid_wells": int(valid_v171.sum()),
        "pca_components": int(pca_v171.n_components_),
        "pca_variance": float(np.sum(pca_v171.explained_variance_ratio_)),
    }
    model_reports_v171.append(report_v171)
    joblib.dump(
        {
            "feature_fill": center_v171, "feature_mean": mean_v171,
            "feature_scale": scale_v171, "pca": pca_v171,
            "ridge": ridge_v171, "extra": extra_v171,
        },
        MODEL_V171 / f"heldout_fold{held_fold_v171}.joblib",
        compress=3,
    )
    print("V171 MODEL", report_v171, flush=True)

for endpoint_v171, value_v171 in prediction_curves_v171.items():
    if not np.isfinite(value_v171).all():
        raise RuntimeError(f"Incomplete {endpoint_v171} cross-fit")

frame_v171 = base_frame.copy()
frame_v171["well"] = frame_v171["well"].astype(str)
frame_v171 = frame_v171.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
engine_prediction_v171 = {
    endpoint: np.full(len(frame_v171), np.nan, np.float64)
    for endpoint in prediction_curves_v171
}
well_to_index_v171 = {well: index for index, well in enumerate(well_order_v171)}
for well_v171, positions_v171 in frame_v171.groupby("well", sort=False).indices.items():
    positions_v171 = np.asarray(positions_v171, np.int64)
    record_v171 = records[str(well_v171)]
    row_idx_v171 = frame_v171.iloc[positions_v171]["row_idx"].to_numpy(np.int64)
    fraction_v171 = (
        record_v171["md"][row_idx_v171] - record_v171["md"][record_v171["anchor"]]
    ) / max(float(record_v171["length"]), 1.0)
    fraction_v171 = np.clip(fraction_v171, 0.0, 1.0)
    grid_v171 = np.linspace(0.0, 1.0, CURVE_NODES_V171)
    z_delta_v171 = record_v171["z"][row_idx_v171] - record_v171["anchor_z"]
    index_v171 = well_to_index_v171[str(well_v171)]
    for endpoint_v171, curves_endpoint_v171 in prediction_curves_v171.items():
        surface_delta_v171 = np.interp(
            fraction_v171, grid_v171, curves_endpoint_v171[index_v171]
        )
        engine_prediction_v171[endpoint_v171][positions_v171] = (
            record_v171["anchor_tvt"] + surface_delta_v171 - z_delta_v171
        )

base_v171 = frame_v171[prediction_column].to_numpy(np.float64)
truth_v171 = frame_v171["tvt_true"].to_numpy(np.float64)
distance_v171 = frame_v171["md_since"].to_numpy(np.float64)


def _detrended_corr_v171(left, right):
    bins_v171 = np.clip((distance_v171 / 250.0).astype(int), 0, 31)
    left_v171 = np.asarray(left, np.float64).copy()
    right_v171 = np.asarray(right, np.float64).copy()
    for bin_v171 in np.unique(bins_v171):
        selected_v171 = bins_v171 == bin_v171
        left_v171[selected_v171] -= np.mean(left_v171[selected_v171])
        right_v171[selected_v171] -= np.mean(right_v171[selected_v171])
    return float(np.corrcoef(left_v171, right_v171)[0, 1])


summary_rows_v171 = [_metric(frame_v171, base_v171, "base_current")]
gateway_rows_v171 = []
output_v171 = frame_v171[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output_v171["base_current"] = base_v171.astype(np.float32)
for endpoint_v171, standalone_v171 in engine_prediction_v171.items():
    standalone_metric_v171 = _metric(frame_v171, standalone_v171, endpoint_v171)
    summary_rows_v171.append(standalone_metric_v171)
    move_v171 = standalone_v171 - base_v171
    residual_v171 = truth_v171 - base_v171
    denominator_v171 = float(np.dot(move_v171, move_v171))
    alpha_v171 = float(np.dot(move_v171, residual_v171) / denominator_v171)
    blend_v171 = base_v171 + alpha_v171 * move_v171
    blend_metric_v171 = _metric(frame_v171, blend_v171, f"{endpoint_v171}_oracle_global_blend")
    summary_rows_v171.append(blend_metric_v171)
    fold_gains_v171 = []
    for fold_v171 in range(1, 6):
        selected_v171 = frame_v171["fold"].to_numpy(int) == fold_v171
        fold_base_v171 = _metric(frame_v171.loc[selected_v171], base_v171[selected_v171], "base")
        fold_blend_v171 = _metric(frame_v171.loc[selected_v171], blend_v171[selected_v171], "blend")
        fold_gains_v171.append(fold_base_v171["rmse"] - fold_blend_v171["rmse"])
    gateway_rows_v171.append({
        "endpoint": endpoint_v171,
        "standalone_rmse": standalone_metric_v171["rmse"],
        "raw_error_corr": float(np.corrcoef(standalone_v171 - truth_v171, base_v171 - truth_v171)[0, 1]),
        "detrended_error_corr": _detrended_corr_v171(standalone_v171 - truth_v171, base_v171 - truth_v171),
        "oracle_global_alpha": alpha_v171,
        "oracle_global_gain": summary_rows_v171[0]["rmse"] - blend_metric_v171["rmse"],
        "positive_blend_folds": int(np.sum(np.asarray(fold_gains_v171) > 0)),
        "minimum_blend_fold_gain": float(np.min(fold_gains_v171)),
    })
    output_v171[endpoint_v171] = standalone_v171.astype(np.float32)

summary_v171 = pd.DataFrame(summary_rows_v171)
gateway_v171 = pd.DataFrame(gateway_rows_v171).sort_values(
    ["oracle_global_gain", "standalone_rmse"], ascending=[False, True], kind="stable"
)
eligible_v171 = gateway_v171[
    (gateway_v171["standalone_rmse"] <= 10.5)
    & (gateway_v171["detrended_error_corr"] <= 0.60)
    & (gateway_v171["oracle_global_gain"] >= 0.20)
    & (gateway_v171["positive_blend_folds"] >= 4)
]
verdict_v171 = {
    "experiment": "functional_surface_trajectory_engine_v171",
    "stage": "complete_fivefold_independent_functional_trajectory_gateway",
    "historical_prediction_inputs": [],
    "features": int(features_v171.shape[1]),
    "curve_nodes": CURVE_NODES_V171,
    "curve_components": CURVE_COMPONENTS_V171,
    "coverage": {"rows": len(frame_v171), "wells": frame_v171["well"].nunique()},
    "models": gateway_rows_v171,
    "investment_pass": bool(len(eligible_v171)),
    "selected_for_nested_blend": (
        str(eligible_v171.iloc[0]["endpoint"]) if len(eligible_v171) else None
    ),
    "elapsed_minutes": (time.time() - STARTED_V171) / 60.0,
    "next": (
        "PASS -> nested blend and refit one all-773 deployment engine"
        if len(eligible_v171)
        else "FAIL -> descriptor-to-trajectory mapping is not a CV<6 engine; stop this family"
    ),
}

summary_v171.to_csv(OUT_V171 / "summary.csv", index=False)
gateway_v171.to_csv(OUT_V171 / "gateway.csv", index=False)
pd.DataFrame(model_reports_v171).to_csv(OUT_V171 / "models.csv", index=False)
output_v171.to_parquet(OUT_V171 / "crossfit_oof.parquet", index=False, compression="zstd")
(OUT_V171 / "verdict.json").write_text(
    json.dumps(_clean_v171(verdict_v171), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nV171 SUMMARY")
print(summary_v171.to_string(index=False), flush=True)
print("\nV171 GATEWAY")
print(gateway_v171.to_string(index=False), flush=True)
print("\nV171 FINAL VERDICT")
print(json.dumps(_clean_v171(verdict_v171), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V171, flush=True)
