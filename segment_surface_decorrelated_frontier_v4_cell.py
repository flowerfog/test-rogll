"""Blend-aware, lineage-independent segment-surface endpoint.

This AutoDL cell trains a complete-well endpoint whose inputs exclude the
SP45/FLE/CP/Gold/neighbor lineage.  The current fold-clean mainline is used
only for held-fold correlation and fusion measurement.

For every 256-ft segment the model predicts the average structural-surface
rate from the visible heel anchor to that segment endpoint.  Three frozen,
reference-free objectives form an accuracy/tail-robustness frontier: ordinary
row-weighted L2, well-balanced Huber, and far-emphasised well-balanced Huber.
At inference the endpoint needs only raw deployable features.

The reference deliberately does not enter the loss.  A normal five-fold OOF
reference is not nested deeply enough to be a legal outer-fold training label:
an OOF model for an outer-train well can have seen the current outer-valid
fold.  Blend-aware loss must wait for nested-mainline OOF.

Prerequisites in /root/autodl-tmp:
  * segment_surface_gbdt_frontier_v2_cell.py
  * segment_surface_cumulative_gbdt_frontier_v3_cell.py
  * foldclean_full_fleongg_lgb_v0 feature cache
  * preferably frontier_6p39_existing_models_oof_v0 output

The cell emits five-fold OOF predictions, per-fold models, correlation tables,
nested blend weights, a matched direction-flip control, and a strict verdict.
It never raises merely because the research gate fails.
"""

from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


ROOT_V4 = Path("/root/autodl-tmp")
OOF_ROOT_V4 = ROOT_V4 / "oof_exports"
V3_SOURCE_CANDIDATES_V4 = [
    ROOT_V4 / "segment_surface_cumulative_gbdt_frontier_v3_cell.py",
    Path("segment_surface_cumulative_gbdt_frontier_v3_cell.py"),
]
V3_SOURCE_V4 = next(
    (path for path in V3_SOURCE_CANDIDATES_V4 if path.exists()), None
)
if V3_SOURCE_V4 is None:
    raise FileNotFoundError(
        "Upload segment_surface_cumulative_gbdt_frontier_v3_cell.py to "
        "/root/autodl-tmp"
    )

source_v4 = V3_SOURCE_V4.read_text(encoding="utf-8")
marker_v4 = "started_v3 = time.time()"
if marker_v4 not in source_v4:
    raise RuntimeError(f"Cannot locate definition boundary in {V3_SOURCE_V4}")
exec(
    compile(
        source_v4.split(marker_v4, 1)[0],
        str(V3_SOURCE_V4) + "::<definitions>",
        "exec",
    ),
    globals(),
)


EXPERIMENT_V4 = "segment_surface_decorrelated_frontier_v4"
OUT_DIR_V4 = OOF_ROOT_V4 / EXPERIMENT_V4
MODEL_DIR_V4 = OUT_DIR_V4 / "models"
OUT_DIR_V4.mkdir(parents=True, exist_ok=True)
MODEL_DIR_V4.mkdir(parents=True, exist_ok=True)

# Prefer the strongest reproducible model-core OOF.  The older references are
# retained only so the endpoint can still be trained before that replay exists.
REFERENCE_CANDIDATES = [
    OOF_ROOT_V4
    / "frontier_6p39_existing_models_oof_v0"
    / "frontier_6p39_core_oof.parquet",
    ROOT_V4 / "mainline_6p47_core_oof.parquet",
    ROOT_V4 / "mainline_6p781_oof.parquet",
    OOF_ROOT_V4 / "mainline_6p781_oof" / "mainline_6p781_oof.parquet",
    OOF_ROOT_V4
    / "track8_stride_strong_nested_fusion_v4"
    / "best_nested_predictions.parquet",
]
REFERENCE_COLUMN_PREFERENCE = [
    "frontier_6p39_core",
    "mainline_6p47",
    "mainline_6p781",
    "mainline_tvt",
    "mainline",
    "prediction",
    "tvt_pred",
    "strong_proxy",
]
TRUTH_COLUMN_PREFERENCE = ["truth", "tvt_true", "true_tvt"]

SEGMENT_LENGTHS_FT = [256.0]
SEGMENT_LENGTH_FT = 256.0
ARM_CONFIGS_V4 = [
    {"name": "l2_row", "objective": "regression", "weight": "row"},
    {"name": "huber_well", "objective": "huber", "weight": "well"},
    {"name": "huber_far", "objective": "huber", "weight": "well_far"},
]
MAX_TREES_V4 = 1800
PATIENCE_V4 = 140
BLEND_GRID = np.array(
    [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65],
    dtype=np.float64,
)

# Selection is permissive enough to expose the frontier.  Promotion remains
# tied to the stricter pre-registered checks below.
MAX_ENDPOINT_DETRENDED_RHO = 0.65
MIN_BLENDABILITY_MARGIN = 0.01
PASS_STANDALONE_RMSE_V4 = 10.50
PASS_FAR_RMSE_V4 = 13.00
PASS_DETRENDED_RHO_V4 = 0.45
PASS_BLEND_GAIN_V4 = 0.20
PASS_MIN_WEIGHT_V4 = 0.10
PASS_MIN_FOLD_GAIN_V4 = -0.05
PASS_POSITIVE_FOLDS_V4 = 4
PASS_CONTROL_LOSS_V4 = 0.03


def _new_model_v4(outer_fold, arm, n_estimators, device_type=None):
    kwargs = dict(
        objective=arm["objective"],
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
        random_state=(
            SEED
            + 4000
            + int(outer_fold) * 100
            + sum(ord(character) for character in arm["name"])
        ),
        verbosity=-1,
    )
    if device_type is not None:
        kwargs["device_type"] = device_type
    return LGBMRegressor(**kwargs)


def _training_weights_v4(frame, scheme):
    row_weight = frame["seg_rows"].to_numpy(np.float64)
    if scheme == "row":
        output = row_weight
    else:
        well_total = frame.groupby("well")["seg_rows"].transform("sum").to_numpy(
            np.float64
        )
        output = row_weight / np.maximum(well_total, 1.0)
        output *= float(np.mean(well_total))
        if scheme == "well_far":
            distance = frame["seg_end_ft"].to_numpy(np.float64)
            output *= 1.0 + 1.5 * np.clip(distance / 2500.0, 0.0, 1.0)
    output /= max(float(np.mean(output)), 1e-12)
    return output.astype(np.float32)


def _fit_model_v4(fit, tune, final_train, features, outer_fold, arm):
    x_fit = _matrix(fit, features)
    y_fit = fit["target_avg_rate"].to_numpy(np.float32) * RATE_SCALE
    x_tune = _matrix(tune, features)
    y_tune = tune["target_avg_rate"].to_numpy(np.float32) * RATE_SCALE
    fit_kwargs = dict(
        sample_weight=_training_weights_v4(fit, arm["weight"]),
        eval_set=[(x_tune, y_tune)],
        eval_sample_weight=[_training_weights_v4(tune, arm["weight"])],
        eval_metric="rmse",
        callbacks=[early_stopping(PATIENCE_V4, verbose=False), log_evaluation(0)],
    )
    stage = _fit_with_fallback_v3(
        _new_model_v4(outer_fold, arm, MAX_TREES_V4, "gpu"),
        _new_model_v4(outer_fold, arm, MAX_TREES_V4),
        x_fit,
        y_fit,
        fit_kwargs,
    )
    trees = max(20, int(stage.best_iteration_ or MAX_TREES_V4))
    del stage, x_fit, y_fit, x_tune, y_tune
    gc.collect()

    x_final = _matrix(final_train, features)
    y_final = final_train["target_avg_rate"].to_numpy(np.float32) * RATE_SCALE
    final = _fit_with_fallback_v3(
        _new_model_v4(outer_fold, arm, trees, "gpu"),
        _new_model_v4(outer_fold, arm, trees),
        x_final,
        y_final,
        {"sample_weight": _training_weights_v4(final_train, arm["weight"])},
    )
    del x_final, y_final
    gc.collect()
    return final, trees


def _distance_correlations_v4(frame, candidate):
    truth = frame["truth"].to_numpy(np.float64)
    reference_error = frame["reference"].to_numpy(np.float64) - truth
    candidate_error = np.asarray(candidate, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    bands = [
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, None),
    ]
    output = {}
    for name, low, high in bands:
        mask = distance >= low
        if high is not None:
            mask &= distance < high
        output[f"{name}_error_rho"] = (
            _safe_rho(reference_error[mask], candidate_error[mask])
            if int(mask.sum()) >= 3
            else float("nan")
        )
    return output


def _metrics_extended_v4(frame, prediction):
    output = _metrics(frame, prediction)
    truth = frame["truth"].to_numpy(np.float64)
    error2 = (np.asarray(prediction, np.float64) - truth) ** 2
    well_sse = pd.Series(error2).groupby(frame["well"].astype(str).to_numpy()).sum()
    worst_count = max(1, int(np.ceil(0.20 * len(well_sse))))
    output["worst20_well_sse_share"] = float(
        well_sse.nlargest(worst_count).sum() / max(float(well_sse.sum()), 1e-12)
    )
    return output


started_v4 = time.time()
for required_v4 in (FEATURE_DIR, FEATURE_LIST_PATH, FOLD_PATH):
    if not required_v4.exists():
        raise FileNotFoundError(required_v4)

reference_path_v4, reference_column_v4, _, reference_table_v4 = _load_reference()
reference_groups_v4 = {
    well: group.set_index("row_idx").sort_index()
    for well, group in reference_table_v4.groupby("well", sort=False)
}
fold_metadata_v4 = pd.read_csv(FOLD_PATH, usecols=["well", "fold"])
fold_metadata_v4["well"] = fold_metadata_v4["well"].astype(str).str.lower()
fold_metadata_v4["fold"] = fold_metadata_v4["fold"].astype(int)
fold_by_well_v4 = (
    fold_metadata_v4.drop_duplicates("well").set_index("well")["fold"]
)
eligible_wells_v4 = sorted(
    well
    for well in fold_by_well_v4.index
    if well in reference_groups_v4 and _feature_paths(well)[1].exists()
)
if len(eligible_wells_v4) < MIN_WELLS:
    raise RuntimeError(f"Only {len(eligible_wells_v4)} eligible wells")

listed_features_v4 = pd.read_csv(FEATURE_LIST_PATH)["feature"].astype(str).tolist()
schema_v4 = pd.read_pickle(_feature_paths(eligible_wells_v4[0])[1])
_, _, broad_features_v4, moment_features_v4 = _feature_partitions(
    listed_features_v4, schema_v4
)
rows_v4, segment_tables_v4 = _build_segment_bank(
    eligible_wells_v4,
    fold_by_well_v4,
    reference_groups_v4,
    broad_features_v4,
    moment_features_v4,
)
segments_v4, cumulative_features_v4 = _add_cumulative_features_v3(
    segment_tables_v4[SEGMENT_LENGTH_FT],
    broad_features_v4,
    moment_features_v4,
)
base_features_v4 = _segment_feature_columns(broad_features_v4, moment_features_v4)
model_features_v4 = base_features_v4 + cumulative_features_v4

reference_exact_v4 = (
    reference_column_v4 in {"frontier_6p39_core", "mainline_6p47", "mainline_6p781"}
)
print(
    "DECORRELATED-SURFACE CONFIG",
    {
        "reference": str(reference_path_v4),
        "reference_column": reference_column_v4,
        "reference_exact_model_core": reference_exact_v4,
        "wells": len(eligible_wells_v4),
        "segments": len(segments_v4),
        "features": len(model_features_v4),
        "arms": ARM_CONFIGS_V4,
    },
    flush=True,
)

oracle_v4 = _decode_average_rates_v3(
    rows_v4,
    segments_v4,
    segments_v4["target_avg_rate"].to_numpy(np.float64),
)
oracle_metrics_v4 = _metrics(rows_v4, oracle_v4)
print("DECORRELATED-SURFACE REPRESENTATION ORACLE", oracle_metrics_v4, flush=True)
if oracle_metrics_v4["rmse"] > ORACLE_GATE_RMSE:
    raise RuntimeError(f"Representation oracle failed: {oracle_metrics_v4}")

oof_rates_v4 = {
    arm["name"]: np.full(len(segments_v4), np.nan, np.float64)
    for arm in ARM_CONFIGS_V4
}
model_records_v4 = []
for outer_fold_v4 in FOLDS:
    tune_fold_v4 = FOLDS[outer_fold_v4 % len(FOLDS)]
    fit_v4 = segments_v4.loc[
        ~segments_v4["fold"].isin([outer_fold_v4, tune_fold_v4])
    ]
    tune_v4 = segments_v4.loc[segments_v4["fold"].eq(tune_fold_v4)]
    final_train_v4 = segments_v4.loc[segments_v4["fold"].ne(outer_fold_v4)]
    valid_mask_v4 = segments_v4["fold"].eq(outer_fold_v4).to_numpy()
    valid_v4 = segments_v4.loc[valid_mask_v4].reset_index(drop=True)
    print(
        f"decorrelated surface outer={outer_fold_v4} tune={tune_fold_v4} "
        f"fit={len(fit_v4)} final={len(final_train_v4)} valid={len(valid_v4)}",
        flush=True,
    )
    physical_low_v4, physical_high_v4 = np.quantile(
        final_train_v4["target_avg_rate"].to_numpy(np.float64), [0.001, 0.999]
    )
    padding_v4 = max(0.003, 0.05 * float(physical_high_v4 - physical_low_v4))
    for arm_config_v4 in ARM_CONFIGS_V4:
        model_v4, trees_v4 = _fit_model_v4(
            fit_v4,
            tune_v4,
            final_train_v4,
            model_features_v4,
            outer_fold_v4,
            arm_config_v4,
        )
        prediction_v4 = (
            model_v4.predict(_matrix(valid_v4, model_features_v4)).astype(np.float64)
            / RATE_SCALE
        )
        prediction_v4 = np.clip(
            prediction_v4,
            physical_low_v4 - padding_v4,
            physical_high_v4 + padding_v4,
        )
        arm_name_v4 = arm_config_v4["name"]
        oof_rates_v4[arm_name_v4][valid_mask_v4] = prediction_v4
        joblib.dump(
            model_v4,
            MODEL_DIR_V4 / f"surface_{arm_name_v4}_fold{outer_fold_v4}.joblib",
        )
        model_records_v4.append(
            {
                "outer_fold": outer_fold_v4,
                "tune_fold": tune_fold_v4,
                "arm": arm_name_v4,
                "objective": arm_config_v4["objective"],
                "weight_scheme": arm_config_v4["weight"],
                "selected_trees": trees_v4,
                "segments_fit": len(fit_v4),
                "segments_final_train": len(final_train_v4),
                "segments_valid": len(valid_v4),
            }
        )
        del model_v4, prediction_v4
        gc.collect()
    del fit_v4, tune_v4, final_train_v4, valid_v4
    gc.collect()

candidate_arms_v4 = []
for arm_name_v4, rates_v4 in oof_rates_v4.items():
    if not np.isfinite(rates_v4).all():
        raise RuntimeError(f"Incomplete OOF rates for arm={arm_name_v4}")
    arm_v4 = f"surface_{arm_name_v4}"
    smoothed_v4 = _smooth_rates(segments_v4, rates_v4)
    rows_v4[arm_v4] = _decode_average_rates_v3(
        rows_v4, segments_v4, smoothed_v4
    ).astype(np.float32)
    candidate_arms_v4.append(arm_v4)

reference_metrics_v4 = _metrics_extended_v4(rows_v4, rows_v4["reference"])
endpoint_records_v4 = []
for arm_v4 in candidate_arms_v4:
    metrics_v4 = _metrics_extended_v4(rows_v4, rows_v4[arm_v4])
    correlations_v4 = _correlations(rows_v4, rows_v4[arm_v4])
    endpoint_records_v4.append(
        {
            "arm": arm_v4,
            **metrics_v4,
            **correlations_v4,
            **_distance_correlations_v4(rows_v4, rows_v4[arm_v4]),
        }
    )
endpoint_report_v4 = pd.DataFrame(endpoint_records_v4).sort_values(
    "rmse", ignore_index=True
)

nested_blend_v4 = np.empty(len(rows_v4), np.float64)
selected_candidate_v4 = np.empty(len(rows_v4), np.float64)
fold_records_v4 = []
scan_parts_v4 = []
for held_fold_v4 in FOLDS:
    selection_v4 = rows_v4.loc[rows_v4["fold"].ne(held_fold_v4)]
    held_mask_v4 = rows_v4["fold"].eq(held_fold_v4).to_numpy()
    held_v4 = rows_v4.loc[held_mask_v4]
    selected_v4, scan_v4 = _selection_scan(selection_v4, candidate_arms_v4)
    scan_v4.insert(0, "held_fold", held_fold_v4)
    scan_parts_v4.append(scan_v4)
    arm_v4 = str(selected_v4["arm"])
    weight_v4 = float(selected_v4["weight_candidate"])
    candidate_v4 = held_v4[arm_v4].to_numpy(np.float64)
    reference_v4 = held_v4["reference"].to_numpy(np.float64)
    blend_v4 = reference_v4 + weight_v4 * (candidate_v4 - reference_v4)
    selected_candidate_v4[held_mask_v4] = candidate_v4
    nested_blend_v4[held_mask_v4] = blend_v4
    base_rmse_v4 = _metrics(held_v4, reference_v4)["rmse"]
    blend_rmse_v4 = _metrics(held_v4, blend_v4)["rmse"]
    fold_records_v4.append(
        {
            "fold": held_fold_v4,
            "selected_arm": arm_v4,
            "weight_candidate": weight_v4,
            "reference_rmse": base_rmse_v4,
            "candidate_rmse": _metrics(held_v4, candidate_v4)["rmse"],
            "blend_rmse": blend_rmse_v4,
            "gain": base_rmse_v4 - blend_rmse_v4,
            **_correlations(held_v4, candidate_v4),
            **_distance_correlations_v4(held_v4, candidate_v4),
        }
    )

rows_v4["selected_candidate"] = selected_candidate_v4.astype(np.float32)
rows_v4["nested_blend"] = nested_blend_v4.astype(np.float32)
fold_report_v4 = pd.DataFrame(fold_records_v4)
fold_weights_v4 = fold_report_v4.set_index("fold")["weight_candidate"].to_dict()
direction_v4 = selected_candidate_v4 - rows_v4["reference"].to_numpy(np.float64)
rows_v4["matched_direction_flip"] = (
    rows_v4["reference"].to_numpy(np.float64)
    + rows_v4["fold"].map(fold_weights_v4).to_numpy(np.float64)
    * _matched_direction_flip(rows_v4, direction_v4)
).astype(np.float32)

candidate_metrics_v4 = _metrics_extended_v4(
    rows_v4, rows_v4["selected_candidate"]
)
blend_metrics_v4 = _metrics_extended_v4(rows_v4, rows_v4["nested_blend"])
control_metrics_v4 = _metrics_extended_v4(
    rows_v4, rows_v4["matched_direction_flip"]
)
candidate_correlations_v4 = _correlations(
    rows_v4, rows_v4["selected_candidate"]
)
candidate_correlations_v4.update(
    _distance_correlations_v4(rows_v4, rows_v4["selected_candidate"])
)
blend_gain_v4 = reference_metrics_v4["rmse"] - blend_metrics_v4["rmse"]
control_loss_v4 = control_metrics_v4["rmse"] - blend_metrics_v4["rmse"]
selected_counts_v4 = fold_report_v4["selected_arm"].value_counts().to_dict()

checks_v4 = {
    "representation_oracle_rmse_le_1p5": oracle_metrics_v4["rmse"] <= 1.50,
    "standalone_rmse_le_10p5": candidate_metrics_v4["rmse"]
    <= PASS_STANDALONE_RMSE_V4,
    "far_gt2000_rmse_le_13": candidate_metrics_v4["far_gt2000_rmse"]
    <= PASS_FAR_RMSE_V4,
    "distance_detrended_rho_le_0p45": candidate_correlations_v4[
        "distance_detrended_error_rho"
    ]
    <= PASS_DETRENDED_RHO_V4,
    "nested_blend_gain_ge_0p20": blend_gain_v4 >= PASS_BLEND_GAIN_V4,
    "minimum_selected_weight_ge_0p10": float(
        fold_report_v4["weight_candidate"].min()
    )
    >= PASS_MIN_WEIGHT_V4,
    "minimum_fold_gain_ge_minus0p05": float(fold_report_v4["gain"].min())
    >= PASS_MIN_FOLD_GAIN_V4,
    "positive_folds_ge_4": int((fold_report_v4["gain"] > 0.0).sum())
    >= PASS_POSITIVE_FOLDS_V4,
    "matched_control_worse_by_ge_0p03": control_loss_v4
    >= PASS_CONTROL_LOSS_V4,
    "same_arm_selected_in_at_least_3_folds": max(selected_counts_v4.values()) >= 3,
}
strict_pass_v4 = bool(all(checks_v4.values()))

output_columns_v4 = [
    "id",
    "well",
    "row_idx",
    "fold",
    "truth",
    "md_since",
    "reference",
    *candidate_arms_v4,
    "selected_candidate",
    "nested_blend",
    "matched_direction_flip",
]
rows_v4[output_columns_v4].to_parquet(
    OUT_DIR_V4 / "decorrelated_segment_surface_oof.parquet",
    index=False,
    compression="zstd",
)
endpoint_report_v4.to_csv(OUT_DIR_V4 / "endpoint_report.csv", index=False)
fold_report_v4.to_csv(OUT_DIR_V4 / "nested_fold_report.csv", index=False)
pd.concat(scan_parts_v4, ignore_index=True).to_csv(
    OUT_DIR_V4 / "nested_blend_scan.csv", index=False
)
pd.DataFrame(model_records_v4).to_csv(
    OUT_DIR_V4 / "model_report.csv", index=False
)
pd.DataFrame({"feature": model_features_v4}).to_csv(
    OUT_DIR_V4 / "model_features.csv", index=False
)

verdict_v4 = {
    "experiment": EXPERIMENT_V4,
    "stage": "fivefold_blendaware_absolute_surface_endpoint_gateway",
    "reference": {
        "path": str(reference_path_v4),
        "column": reference_column_v4,
        "exact_reproducible_model_core": reference_exact_v4,
    },
    "protocol": {
        "sample_unit": "256-ft segments nested within complete physical wells",
        "model_inputs": "deployable non-endpoint features only",
        "reference_used_as_model_input": False,
        "reference_used_in_training_objective": False,
        "target": "average structural-surface rate from heel to endpoint",
        "arms": ARM_CONFIGS_V4,
        "nested_reference_warning": (
            "ordinary OOF reference residuals are evaluation-only; they are "
            "not legal outer-fold labels without nested-mainline replay"
        ),
        "outer_model_refit": (
            "tree count selected on 3 folds plus tune fold; final model refit "
            "on all 4 outer-train folds"
        ),
        "folds": FOLDS,
    },
    "guards": {
        "eligible_wells": len(eligible_wells_v4),
        "segments": len(segments_v4),
        "features": len(model_features_v4),
    },
    "representation_oracle": oracle_metrics_v4,
    "endpoint_report": endpoint_report_v4.to_dict(orient="records"),
    "reference_metrics": reference_metrics_v4,
    "selected_candidate_metrics": candidate_metrics_v4,
    "selected_candidate_correlations": candidate_correlations_v4,
    "nested_blend_metrics": blend_metrics_v4,
    "nested_blend_gain": blend_gain_v4,
    "matched_direction_flip_metrics": control_metrics_v4,
    "matched_control_loss": control_loss_v4,
    "selected_arm_counts": selected_counts_v4,
    "folds": fold_report_v4.to_dict(orient="records"),
    "checks": checks_v4,
    "strict_pass": strict_pass_v4,
    "elapsed_minutes": (time.time() - started_v4) / 60.0,
    "next": (
        "PASS -> package the selected independent endpoint and confirm against the "
        "frozen 6.47 submission path"
        if strict_pass_v4
        else "FAIL -> the independent feature family cannot learn a stable "
        "anti-correlated endpoint; change evidence, not blend dose"
    ),
}
_write_json(OUT_DIR_V4 / "verdict.json", verdict_v4)

print("\nDECORRELATED ENDPOINT REPORT")
print(endpoint_report_v4.to_string(index=False), flush=True)
print("\nDECORRELATED NESTED FOLDS")
print(fold_report_v4.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(
    json.dumps(verdict_v4, indent=2, ensure_ascii=False, default=_json_default),
    flush=True,
)
print(f"saved: {OUT_DIR_V4}", flush=True)
