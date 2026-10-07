"""Cumulative segment-surface GBDT endpoint and nested blend gateway.

V2 proved that segment-rate prediction changes the failure mechanism, but its
independent rate errors accumulate from heel to toe.  This cell predicts an
absolute endpoint quantity instead:

    average_rate(x) = (surface(x) - surface(heel)) / x,
    surface = TVT + Z.

Each segment endpoint is therefore tied directly to the known heel anchor.
The decoded path interpolates between absolute endpoint predictions instead
of integrating an unconstrained chain of local rates.  The current mainline
or proxy remains evaluation-only and is never a model input or label.

Run V2 first and leave its source and OOF directory on AutoDL.  V2's local-rate
endpoint is included only as a separately measured candidate and in fixed
dual-target candidate paths; no validation truth is used to build them.
"""

from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


V2_SOURCE_CANDIDATES = [
    Path("/root/autodl-tmp/segment_surface_gbdt_frontier_v2_cell.py"),
    Path("segment_surface_gbdt_frontier_v2_cell.py"),
]
V2_SOURCE = next((path for path in V2_SOURCE_CANDIDATES if path.exists()), None)
if V2_SOURCE is None:
    raise FileNotFoundError(
        "Upload segment_surface_gbdt_frontier_v2_cell.py to /root/autodl-tmp"
    )
source_v3 = V2_SOURCE.read_text(encoding="utf-8")
marker_v3 = "started = time.time()"
if marker_v3 not in source_v3:
    raise RuntimeError(f"Cannot locate definition boundary in {V2_SOURCE}")
exec(
    compile(source_v3.split(marker_v3, 1)[0], str(V2_SOURCE) + "::<definitions>", "exec"),
    globals(),
)


EXPERIMENT = "segment_surface_cumulative_gbdt_frontier_v3"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LENGTHS_FT = [256.0]
SEGMENT_LENGTH_FT = 256.0
MAX_TREES_V3 = 1800
PATIENCE_V3 = 140
MAX_ENDPOINT_DETRENDED_RHO_V3 = 0.75
# _selection_scan is inherited from V2 and reads this global.
MAX_ENDPOINT_DETRENDED_RHO = MAX_ENDPOINT_DETRENDED_RHO_V3
PASS_STANDALONE_RMSE_V3 = 11.50
PASS_BLEND_GAIN_FT_V3 = 0.10
PASS_MIN_FOLD_GAIN_FT_V3 = -0.05
PASS_CONTROL_LOSS_FT_V3 = 0.03
V2_OOF = (
    OOF_ROOT / "segment_surface_gbdt_frontier_v2" / "segment_surface_gbdt_oof.parquet"
)


def _add_cumulative_features_v3(segments, broad_features, moment_features):
    segments = segments.copy().sort_values("segment_uid").reset_index(drop=True)
    mean_columns = [f"mean__{column}" for column in broad_features]
    moment_columns = [
        f"mean__{column}" for column in moment_features if f"mean__{column}" in segments
    ]
    mean_values = segments[mean_columns].to_numpy(np.float64)
    moment_positions = [mean_columns.index(column) for column in moment_columns]
    cumulative_mean = np.empty_like(mean_values)
    cumulative_std = np.empty((len(segments), len(moment_positions)), np.float64)
    target_delta = np.empty(len(segments), np.float64)

    for _, indices in segments.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        order = np.argsort(segments.iloc[indices]["seg_index"].to_numpy())
        ordered = indices[order]
        weights = segments.iloc[ordered]["seg_rows"].to_numpy(np.float64)
        values = mean_values[ordered]
        cumulative_weight = np.cumsum(weights)
        cumulative_sum = np.cumsum(values * weights[:, None], axis=0)
        cumulative_mean[ordered] = cumulative_sum / cumulative_weight[:, None]
        if moment_positions:
            selected = values[:, moment_positions]
            cumulative_square = np.cumsum(selected**2 * weights[:, None], axis=0)
            selected_mean = cumulative_sum[:, moment_positions] / cumulative_weight[:, None]
            variance = cumulative_square / cumulative_weight[:, None] - selected_mean**2
            cumulative_std[ordered] = np.sqrt(np.maximum(variance, 0.0))
        local_delta = (
            segments.iloc[ordered]["target_rate"].to_numpy(np.float64)
            * segments.iloc[ordered]["seg_length_ft"].to_numpy(np.float64)
        )
        target_delta[ordered] = np.cumsum(local_delta)

    cumulative_frame = pd.DataFrame(
        cumulative_mean,
        columns=[f"cummean__{column}" for column in broad_features],
        index=segments.index,
    )
    if moment_positions:
        history_frame = pd.DataFrame(
            cumulative_std,
            columns=[
                f"history_std__{mean_columns[position].split('mean__', 1)[1]}"
                for position in moment_positions
            ],
            index=segments.index,
        )
        cumulative_frame = pd.concat([cumulative_frame, history_frame], axis=1)
    segments = pd.concat([segments, cumulative_frame], axis=1)
    endpoint = np.maximum(segments["seg_end_ft"].to_numpy(np.float64), 1.0)
    segments["target_avg_rate"] = target_delta / endpoint
    return segments, cumulative_frame.columns.tolist()


def _new_model_v3(outer_fold, n_estimators, device_type=None):
    kwargs = dict(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.25,
        reg_lambda=24.0,
        max_bin=255,
        n_jobs=-1,
        random_state=SEED + 1000 + int(outer_fold),
        verbosity=-1,
    )
    if device_type is not None:
        kwargs["device_type"] = device_type
    return LGBMRegressor(**kwargs)


def _fit_with_fallback_v3(model_gpu, model_cpu, x, y, fit_kwargs):
    try:
        model_gpu.fit(x, y, **fit_kwargs)
        return model_gpu
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("LightGBM GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model_cpu.fit(x, y, **fit_kwargs)
        return model_cpu


def _fit_refit_v3(fit, tune, final_train, features, outer_fold):
    x_fit = _matrix(fit, features)
    y_fit = fit["target_avg_rate"].to_numpy(np.float32) * RATE_SCALE
    x_tune = _matrix(tune, features)
    y_tune = tune["target_avg_rate"].to_numpy(np.float32) * RATE_SCALE
    fit_kwargs = dict(
        sample_weight=fit["seg_rows"].to_numpy(np.float32),
        eval_set=[(x_tune, y_tune)],
        eval_sample_weight=[tune["seg_rows"].to_numpy(np.float32)],
        eval_metric="rmse",
        callbacks=[early_stopping(PATIENCE_V3, verbose=False), log_evaluation(0)],
    )
    stage = _fit_with_fallback_v3(
        _new_model_v3(outer_fold, MAX_TREES_V3, "gpu"),
        _new_model_v3(outer_fold, MAX_TREES_V3),
        x_fit,
        y_fit,
        fit_kwargs,
    )
    selected_trees = max(20, int(stage.best_iteration_ or MAX_TREES_V3))
    del stage, x_fit, y_fit, x_tune, y_tune
    gc.collect()

    x_final = _matrix(final_train, features)
    y_final = final_train["target_avg_rate"].to_numpy(np.float32) * RATE_SCALE
    final_kwargs = dict(
        sample_weight=final_train["seg_rows"].to_numpy(np.float32)
    )
    final = _fit_with_fallback_v3(
        _new_model_v3(outer_fold, selected_trees, "gpu"),
        _new_model_v3(outer_fold, selected_trees),
        x_final,
        y_final,
        final_kwargs,
    )
    del x_final, y_final
    gc.collect()
    return final, selected_trees


def _decode_average_rates_v3(rows, segments, average_rates):
    average_rates = np.asarray(average_rates, np.float64)
    if len(average_rates) != len(segments) or not np.isfinite(average_rates).all():
        raise RuntimeError("Invalid cumulative average-rate vector")
    rate_by_uid = pd.Series(
        average_rates, index=segments["segment_uid"].to_numpy(np.int32)
    )
    end_by_uid = segments.set_index("segment_uid")["seg_end_ft"]
    anchor_by_well = segments.groupby("well")["anchor_surface"].first()
    output = np.empty(len(rows), np.float64)
    uid_column = f"segment_uid_{int(SEGMENT_LENGTH_FT)}"
    for well, indices in rows.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        order = np.argsort(rows.iloc[indices]["md_since"].to_numpy(np.float64))
        ordered = indices[order]
        well_rows = rows.iloc[ordered]
        anchor = float(anchor_by_well[well])
        previous_x = 0.0
        previous_surface = anchor
        for uid in well_rows[uid_column].drop_duplicates().astype(int):
            local = np.flatnonzero(well_rows[uid_column].to_numpy(np.int32) == uid)
            global_indices = ordered[local]
            x = rows.iloc[global_indices]["md_since"].to_numpy(np.float64)
            endpoint_x = float(end_by_uid.loc[uid])
            endpoint_surface = anchor + float(rate_by_uid.loc[uid]) * endpoint_x
            slope = (endpoint_surface - previous_surface) / max(
                endpoint_x - previous_x, 1e-3
            )
            surface = previous_surface + slope * (x - previous_x)
            output[global_indices] = (
                surface - rows.iloc[global_indices]["z"].to_numpy(np.float64)
            )
            previous_x = endpoint_x
            previous_surface = endpoint_surface
    return output


def _average_to_guarded_local_v3(segments, average_rates, clip_low, clip_high):
    average_rates = np.asarray(average_rates, np.float64)
    clip_low = np.asarray(clip_low, np.float64)
    clip_high = np.asarray(clip_high, np.float64)
    output = np.empty(len(segments), np.float64)
    for _, indices in segments.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        order = np.argsort(segments.iloc[indices]["seg_index"].to_numpy())
        ordered = indices[order]
        accumulated_delta = 0.0
        for index in ordered:
            endpoint_x = float(segments.iloc[index]["seg_end_ft"])
            segment_length = float(segments.iloc[index]["seg_length_ft"])
            requested_delta = float(average_rates[index]) * endpoint_x
            local_rate = (requested_delta - accumulated_delta) / max(segment_length, 1e-3)
            local_rate = float(np.clip(local_rate, clip_low[index], clip_high[index]))
            output[index] = local_rate
            accumulated_delta += local_rate * segment_length
    return output


started_v3 = time.time()
for required_v3 in (FEATURE_DIR, FEATURE_LIST_PATH, FOLD_PATH):
    if not required_v3.exists():
        raise FileNotFoundError(required_v3)

reference_path_v3, reference_column_v3, reference_is_exact_v3, reference_table_v3 = (
    _load_reference()
)
reference_groups_v3 = {
    well: group.set_index("row_idx").sort_index()
    for well, group in reference_table_v3.groupby("well", sort=False)
}
fold_metadata_v3 = pd.read_csv(FOLD_PATH, usecols=["well", "fold"])
fold_metadata_v3["well"] = fold_metadata_v3["well"].astype(str).str.lower()
fold_metadata_v3["fold"] = fold_metadata_v3["fold"].astype(int)
fold_by_well_v3 = (
    fold_metadata_v3.drop_duplicates("well").set_index("well")["fold"]
)
eligible_wells_v3 = sorted(
    well
    for well in fold_by_well_v3.index
    if well in reference_groups_v3 and _feature_paths(well)[1].exists()
)
if len(eligible_wells_v3) < MIN_WELLS:
    raise RuntimeError(f"Only {len(eligible_wells_v3)} eligible wells")

listed_features_v3 = pd.read_csv(FEATURE_LIST_PATH)["feature"].astype(str).tolist()
schema_v3 = pd.read_pickle(_feature_paths(eligible_wells_v3[0])[1])
_, _, broad_features_v3, moment_features_v3 = _feature_partitions(
    listed_features_v3, schema_v3
)
rows_v3, segment_tables_v3 = _build_segment_bank(
    eligible_wells_v3,
    fold_by_well_v3,
    reference_groups_v3,
    broad_features_v3,
    moment_features_v3,
)
segments_v3, cumulative_features_v3 = _add_cumulative_features_v3(
    segment_tables_v3[SEGMENT_LENGTH_FT], broad_features_v3, moment_features_v3
)
base_segment_features_v3 = _segment_feature_columns(
    broad_features_v3, moment_features_v3
)
model_features_v3 = base_segment_features_v3 + cumulative_features_v3

print(
    "CUMULATIVE-SURFACE CONFIG",
    {
        "reference": str(reference_path_v3),
        "reference_column": reference_column_v3,
        "reference_is_exact_6p781": reference_is_exact_v3,
        "wells": len(eligible_wells_v3),
        "segments": len(segments_v3),
        "base_features": len(base_segment_features_v3),
        "cumulative_features": len(cumulative_features_v3),
        "model_features": len(model_features_v3),
    },
    flush=True,
)

oracle_v3 = _decode_average_rates_v3(
    rows_v3, segments_v3, segments_v3["target_avg_rate"].to_numpy(np.float64)
)
oracle_metrics_v3 = _metrics(rows_v3, oracle_v3)
print("CUMULATIVE REPRESENTATION ORACLE", oracle_metrics_v3, flush=True)
if oracle_metrics_v3["rmse"] > ORACLE_GATE_RMSE:
    raise RuntimeError(f"Cumulative representation oracle failed: {oracle_metrics_v3}")

average_oof_v3 = np.full(len(segments_v3), np.nan, np.float64)
clip_low_v3 = np.full(len(segments_v3), np.nan, np.float64)
clip_high_v3 = np.full(len(segments_v3), np.nan, np.float64)
model_report_v3 = []
for outer_fold_v3 in FOLDS:
    tune_fold_v3 = FOLDS[outer_fold_v3 % len(FOLDS)]
    fit_v3 = segments_v3.loc[
        ~segments_v3["fold"].isin([outer_fold_v3, tune_fold_v3])
    ]
    tune_v3 = segments_v3.loc[segments_v3["fold"].eq(tune_fold_v3)]
    final_train_v3 = segments_v3.loc[segments_v3["fold"].ne(outer_fold_v3)]
    valid_mask_v3 = segments_v3["fold"].eq(outer_fold_v3).to_numpy()
    valid_v3 = segments_v3.loc[valid_mask_v3].reset_index(drop=True)
    print(
        f"cumulative model outer={outer_fold_v3} tune={tune_fold_v3} "
        f"fit={len(fit_v3)} final={len(final_train_v3)} valid={len(valid_v3)}",
        flush=True,
    )
    model_v3, selected_trees_v3 = _fit_refit_v3(
        fit_v3,
        tune_v3,
        final_train_v3,
        model_features_v3,
        outer_fold_v3,
    )
    prediction_v3 = (
        model_v3.predict(_matrix(valid_v3, model_features_v3)).astype(np.float64)
        / RATE_SCALE
    )
    avg_low_v3, avg_high_v3 = np.quantile(
        final_train_v3["target_avg_rate"].to_numpy(np.float64), [0.001, 0.999]
    )
    avg_padding_v3 = max(0.003, 0.05 * float(avg_high_v3 - avg_low_v3))
    prediction_v3 = np.clip(
        prediction_v3, avg_low_v3 - avg_padding_v3, avg_high_v3 + avg_padding_v3
    )
    average_oof_v3[valid_mask_v3] = prediction_v3
    local_low_v3, local_high_v3 = np.quantile(
        final_train_v3["target_rate"].to_numpy(np.float64), [0.001, 0.999]
    )
    local_padding_v3 = max(0.005, 0.05 * float(local_high_v3 - local_low_v3))
    clip_low_v3[valid_mask_v3] = local_low_v3 - local_padding_v3
    clip_high_v3[valid_mask_v3] = local_high_v3 + local_padding_v3
    joblib.dump(model_v3, MODEL_DIR / f"cumulative_broad_fold{outer_fold_v3}.joblib")
    model_report_v3.append(
        {
            "outer_fold": outer_fold_v3,
            "tune_fold": tune_fold_v3,
            "selected_trees": selected_trees_v3,
            "segments_fit": len(fit_v3),
            "segments_final_train": len(final_train_v3),
            "segments_valid": len(valid_v3),
            "avg_clip_low": avg_low_v3 - avg_padding_v3,
            "avg_clip_high": avg_high_v3 + avg_padding_v3,
            "local_clip_low": local_low_v3 - local_padding_v3,
            "local_clip_high": local_high_v3 + local_padding_v3,
        }
    )
    del fit_v3, tune_v3, final_train_v3, valid_v3, model_v3
    gc.collect()

if not (
    np.isfinite(average_oof_v3).all()
    and np.isfinite(clip_low_v3).all()
    and np.isfinite(clip_high_v3).all()
):
    raise RuntimeError("Incomplete cumulative OOF")

smoothed_average_v3 = _smooth_rates(segments_v3, average_oof_v3)
rows_v3["c256_raw"] = _decode_average_rates_v3(
    rows_v3, segments_v3, average_oof_v3
).astype(np.float32)
rows_v3["c256_smooth"] = _decode_average_rates_v3(
    rows_v3, segments_v3, smoothed_average_v3
).astype(np.float32)
guarded_local_v3 = _average_to_guarded_local_v3(
    segments_v3, smoothed_average_v3, clip_low_v3, clip_high_v3
)
rows_v3["c256_guarded"] = _decode_rates(
    rows_v3, segments_v3, SEGMENT_LENGTH_FT, guarded_local_v3
).astype(np.float32)

candidate_arms_v3 = ["c256_raw", "c256_smooth", "c256_guarded"]
v2_loaded_v3 = False
if V2_OOF.exists():
    v2_table_v3 = pd.read_parquet(V2_OOF, columns=["id", "s256_broad"])
    v2_table_v3["id"] = v2_table_v3["id"].astype(str).str.lower()
    if v2_table_v3["id"].duplicated().any():
        raise RuntimeError("V2 OOF has duplicate ids")
    aligned_v2_v3 = v2_table_v3.set_index("id").reindex(rows_v3["id"])
    if aligned_v2_v3["s256_broad"].notna().mean() < MIN_ALIGNMENT:
        raise RuntimeError("V2 endpoint alignment failed")
    rows_v3["v2_s256_rate"] = aligned_v2_v3["s256_broad"].to_numpy(np.float32)
    candidate_arms_v3.append("v2_s256_rate")
    for cumulative_dose_v3 in (0.25, 0.50, 0.75):
        name_v3 = f"dual_c{int(cumulative_dose_v3 * 100):02d}"
        rows_v3[name_v3] = (
            cumulative_dose_v3 * rows_v3["c256_guarded"].to_numpy(np.float64)
            + (1.0 - cumulative_dose_v3)
            * rows_v3["v2_s256_rate"].to_numpy(np.float64)
        ).astype(np.float32)
        candidate_arms_v3.append(name_v3)
    v2_loaded_v3 = True

reference_metrics_v3 = _metrics(rows_v3, rows_v3["reference"])
endpoint_records_v3 = []
for arm_v3 in candidate_arms_v3:
    metrics_v3 = _metrics(rows_v3, rows_v3[arm_v3])
    correlation_v3 = _correlations(rows_v3, rows_v3[arm_v3])
    limit_v3 = reference_metrics_v3["rmse"] / max(metrics_v3["rmse"], 1e-12)
    endpoint_records_v3.append(
        {
            "arm": arm_v3,
            **metrics_v3,
            **correlation_v3,
            "blendability_limit": limit_v3,
            "blendability_margin": limit_v3
            - correlation_v3["distance_detrended_error_rho"],
        }
    )
endpoint_report_v3 = pd.DataFrame(endpoint_records_v3).sort_values(
    "rmse", ignore_index=True
)

nested_blend_v3 = np.empty(len(rows_v3), np.float64)
selected_candidate_v3 = np.empty(len(rows_v3), np.float64)
fold_records_v3 = []
scan_parts_v3 = []
for held_fold_v3 in FOLDS:
    selection_v3 = rows_v3.loc[rows_v3["fold"].ne(held_fold_v3)]
    held_mask_v3 = rows_v3["fold"].eq(held_fold_v3).to_numpy()
    held_v3 = rows_v3.loc[held_mask_v3]
    selected_v3, scan_v3 = _selection_scan(selection_v3, candidate_arms_v3)
    scan_v3.insert(0, "held_fold", held_fold_v3)
    scan_parts_v3.append(scan_v3)
    selected_arm_v3 = str(selected_v3["arm"])
    selected_weight_v3 = float(selected_v3["weight_candidate"])
    candidate_v3 = held_v3[selected_arm_v3].to_numpy(np.float64)
    reference_v3 = held_v3["reference"].to_numpy(np.float64)
    blend_v3 = reference_v3 + selected_weight_v3 * (candidate_v3 - reference_v3)
    selected_candidate_v3[held_mask_v3] = candidate_v3
    nested_blend_v3[held_mask_v3] = blend_v3
    base_rmse_v3 = _metrics(held_v3, reference_v3)["rmse"]
    blend_rmse_v3 = _metrics(held_v3, blend_v3)["rmse"]
    fold_records_v3.append(
        {
            "fold": held_fold_v3,
            "selected_arm": selected_arm_v3,
            "weight_candidate": selected_weight_v3,
            "reference_rmse": base_rmse_v3,
            "candidate_rmse": _metrics(held_v3, candidate_v3)["rmse"],
            "blend_rmse": blend_rmse_v3,
            "gain": base_rmse_v3 - blend_rmse_v3,
            **_correlations(held_v3, candidate_v3),
        }
    )

rows_v3["selected_candidate"] = selected_candidate_v3.astype(np.float32)
rows_v3["nested_blend"] = nested_blend_v3.astype(np.float32)
fold_report_v3 = pd.DataFrame(fold_records_v3)
fold_weights_v3 = fold_report_v3.set_index("fold")["weight_candidate"].to_dict()
direction_v3 = selected_candidate_v3 - rows_v3["reference"].to_numpy(np.float64)
rows_v3["matched_direction_flip"] = (
    rows_v3["reference"].to_numpy(np.float64)
    + rows_v3["fold"].map(fold_weights_v3).to_numpy(np.float64)
    * _matched_direction_flip(rows_v3, direction_v3)
).astype(np.float32)

candidate_metrics_v3 = _metrics(rows_v3, rows_v3["selected_candidate"])
blend_metrics_v3 = _metrics(rows_v3, rows_v3["nested_blend"])
control_metrics_v3 = _metrics(rows_v3, rows_v3["matched_direction_flip"])
candidate_correlations_v3 = _correlations(rows_v3, rows_v3["selected_candidate"])
blend_gain_v3 = reference_metrics_v3["rmse"] - blend_metrics_v3["rmse"]
control_loss_v3 = control_metrics_v3["rmse"] - blend_metrics_v3["rmse"]
selected_counts_v3 = fold_report_v3["selected_arm"].value_counts().to_dict()
checks_v3 = {
    "representation_oracle_rmse_le_1p5": oracle_metrics_v3["rmse"]
    <= ORACLE_GATE_RMSE,
    "selected_standalone_rmse_le_11p5": candidate_metrics_v3["rmse"]
    <= PASS_STANDALONE_RMSE_V3,
    "selected_detrended_rho_le_0p75": candidate_correlations_v3[
        "distance_detrended_error_rho"
    ]
    <= MAX_ENDPOINT_DETRENDED_RHO_V3,
    "nested_blend_gain_ge_0p10": blend_gain_v3 >= PASS_BLEND_GAIN_FT_V3,
    "minimum_fold_gain_ge_minus0p05": float(fold_report_v3["gain"].min())
    >= PASS_MIN_FOLD_GAIN_FT_V3,
    "positive_folds_ge_4": int((fold_report_v3["gain"] > 0.0).sum()) >= 4,
    "minimum_selected_weight_ge_0p02": float(
        fold_report_v3["weight_candidate"].min()
    )
    >= 0.02,
    "matched_control_worse_by_ge_0p03": control_loss_v3
    >= PASS_CONTROL_LOSS_FT_V3,
    "same_arm_selected_in_at_least_3_folds": max(selected_counts_v3.values()) >= 3,
}
strict_pass_v3 = bool(all(checks_v3.values()))

output_columns_v3 = [
    "id",
    "well",
    "row_idx",
    "fold",
    "truth",
    "md_since",
    "reference",
    *candidate_arms_v3,
    "selected_candidate",
    "nested_blend",
    "matched_direction_flip",
]
rows_v3[output_columns_v3].to_parquet(
    OUT_DIR / "cumulative_segment_surface_oof.parquet",
    index=False,
    compression="zstd",
)
endpoint_report_v3.to_csv(OUT_DIR / "endpoint_report.csv", index=False)
fold_report_v3.to_csv(OUT_DIR / "nested_fold_report.csv", index=False)
pd.concat(scan_parts_v3, ignore_index=True).to_csv(
    OUT_DIR / "nested_blend_scan.csv", index=False
)
pd.DataFrame(model_report_v3).to_csv(OUT_DIR / "model_report.csv", index=False)
pd.DataFrame({"feature": model_features_v3}).to_csv(
    OUT_DIR / "model_features.csv", index=False
)

verdict_v3 = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_absolute_segment_endpoint_gbdt_and_nested_blend_gateway",
    "reason_for_target_change": (
        "V2 local-rate errors accumulated from near RMSE 2.73 to far RMSE 16.33"
    ),
    "reference": {
        "path": str(reference_path_v3),
        "column": reference_column_v3,
        "is_exact_6p781_oof": reference_is_exact_v3,
    },
    "protocol": {
        "sample_unit": "256-ft segment nested within complete physical well",
        "target": "average surface rate from known heel to segment endpoint",
        "decode": "absolute endpoint surfaces plus piecewise interpolation",
        "outer_model_refit": "select trees on 3 folds + tune fold, refit on all 4 outer-train folds",
        "reference_used_as_model_input": False,
        "reference_used_as_training_label": False,
        "v2_endpoint_loaded": v2_loaded_v3,
    },
    "feature_counts": {
        "broad_rows": len(broad_features_v3),
        "segment_base": len(base_segment_features_v3),
        "cumulative": len(cumulative_features_v3),
        "total": len(model_features_v3),
    },
    "representation_oracle": oracle_metrics_v3,
    "endpoint_report": endpoint_report_v3.to_dict(orient="records"),
    "reference_metrics": reference_metrics_v3,
    "selected_candidate_metrics": candidate_metrics_v3,
    "nested_blend_metrics": blend_metrics_v3,
    "matched_direction_flip_metrics": control_metrics_v3,
    "selected_candidate_correlations": candidate_correlations_v3,
    "nested_blend_gain": blend_gain_v3,
    "matched_control_loss": control_loss_v3,
    "selected_arm_counts": selected_counts_v3,
    "folds": fold_report_v3.to_dict(orient="records"),
    "checks": checks_v3,
    "strict_pass": strict_pass_v3,
    "elapsed_minutes": (time.time() - started_v3) / 60.0,
    "next": (
        "PASS proxy -> rerun only fusion audit against exact 6.781 OOF"
        if strict_pass_v3 and not reference_is_exact_v3
        else "PASS exact mainline -> full refit and deployment packaging"
        if strict_pass_v3
        else "FAIL -> close segment GBDT; absolute endpoint target did not cure drift"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict_v3)

print("\nCUMULATIVE ENDPOINT REPORT")
print(endpoint_report_v3.to_string(index=False), flush=True)
print("\nCUMULATIVE NESTED FOLDS")
print(fold_report_v3.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_v3, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
