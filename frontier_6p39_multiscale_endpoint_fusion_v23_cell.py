"""Nested multiscale endpoint fusion on the complete deployable 6.39 OOF tail.

This is a zero-forward postprocessing audit.  It decomposes existing, fold-clean
endpoint disagreements into frequency bands and learns a small linear correction
to the complete 6.39 parent.  Fold 3 selects the recipe, Fold 2 confirms it, and
Fold 1 is the untouched final check.
"""

from pathlib import Path
import json
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
CORE_PATH = (
    OOF_ROOT / "frontier_6p39_existing_models_oof_v0"
    / "frontier_6p39_core_oof.parquet"
)
FULL_PATH = (
    OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18"
    / "full_oof_cp_endpoints.parquet"
)
EXPERIMENT = "frontier_6p39_multiscale_endpoint_fusion_v23"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_COLUMN = "full_parent_matched_vp_replay"
ENDPOINTS = ("ridge_pp", "sp45_selector_raw", "fle")
SMOOTH_SCALES_FT = (32.0, 128.0, 512.0)
RIDGE_FRACTIONS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DOSES = (0.25, 0.50, 0.75, 1.00)
CAPS_FT = (2.0, 4.0, 8.0, 16.0)
FIT_STRIDE = 16
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
SEED = 20260725
started = time.time()


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    error = prediction - truth
    well_rmse = np.sqrt(
        pd.Series(error * error).groupby(frame["well"], sort=False).mean().to_numpy()
    )
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    bands = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in bands.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _read_inputs():
    for path in (CORE_PATH, FULL_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    full_columns = [
        "id", "well", "row_idx", "fold", "tvt_true", "md_since", PARENT_COLUMN,
    ]
    full = pd.read_parquet(FULL_PATH, columns=full_columns)
    core_columns = ["id", "truth", "frontier_6p39_core", *ENDPOINTS]
    core = pd.read_parquet(CORE_PATH, columns=core_columns)
    if full["id"].duplicated().any() or core["id"].duplicated().any():
        raise RuntimeError("Duplicate ids in an OOF input")

    if len(full) == len(core) and np.array_equal(
        full["id"].astype(str).to_numpy(), core["id"].astype(str).to_numpy()
    ):
        frame = full.reset_index(drop=True)
        for column in core_columns[1:]:
            frame[column] = core[column].to_numpy()
    else:
        frame = full.merge(core, on="id", how="left", validate="one_to_one")
    missing = int(frame["frontier_6p39_core"].isna().sum())
    if missing:
        raise RuntimeError(f"Core alignment left {missing} rows unmatched")
    truth_gap = np.abs(
        frame["tvt_true"].to_numpy(np.float64)
        - frame["truth"].to_numpy(np.float64)
    )
    if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
        raise RuntimeError(f"Truth mismatch: max={float(np.nanmax(truth_gap))}")
    frame["well"] = frame["well"].astype(str)
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    return frame, float(truth_gap.max())


def _build_feature_cache(frame):
    core = frame["frontier_6p39_core"].to_numpy(np.float64)
    distance = frame["md_since"].to_numpy(np.float64)
    endpoint_values = {
        endpoint: frame[endpoint].to_numpy(np.float64) for endpoint in ENDPOINTS
    }
    cache = {}
    feature_names = []
    for endpoint in ENDPOINTS:
        feature_names.extend([
            f"{endpoint}_high_lt32",
            f"{endpoint}_mid_32_128",
            f"{endpoint}_long_128_512",
            f"{endpoint}_trend_gt512",
        ])

    groups = frame.groupby("well", sort=False).indices
    for number, (well, raw_indices) in enumerate(groups.items(), 1):
        indices = np.asarray(raw_indices, np.int64)
        order = np.argsort(frame.loc[indices, "row_idx"].to_numpy(), kind="stable")
        indices = indices[order]
        d = distance[indices]
        delta_d = np.diff(d)
        positive = delta_d[np.isfinite(delta_d) & (delta_d > 0.0)]
        spacing = float(np.median(positive)) if len(positive) else 1.0
        spacing = float(np.clip(spacing, 0.25, 4.0))
        columns = []
        for endpoint in ENDPOINTS:
            signal = endpoint_values[endpoint][indices] - core[indices]
            if not np.isfinite(signal).all():
                raise RuntimeError(f"{well}: non-finite {endpoint} disagreement")
            windows = [
                max(3, int(round(scale_ft / spacing)))
                for scale_ft in SMOOTH_SCALES_FT
            ]
            smooth32 = uniform_filter1d(signal, windows[0], mode="nearest")
            smooth128 = uniform_filter1d(signal, windows[1], mode="nearest")
            smooth512 = uniform_filter1d(signal, windows[2], mode="nearest")
            columns.extend([
                signal - smooth32,
                smooth32 - smooth128,
                smooth128 - smooth512,
                smooth512,
            ])
        features = np.column_stack(columns).astype(np.float32)
        cache[well] = (indices, features)
        if number % 100 == 0 or number == len(groups):
            print(f"MULTISCALE features {number}/{len(groups)}", flush=True)
    return cache, feature_names


class CorrectionModel:
    def __init__(self, ridge_fraction):
        self.ridge_fraction = float(ridge_fraction)
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, x, y):
        xs = self.scaler.fit_transform(x)
        alpha = max(1e-9, self.ridge_fraction * len(xs))
        self.model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
        self.model.fit(xs, y)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))


def _training_matrix(frame, cache, train_folds):
    folds = set(int(value) for value in train_folds)
    truth = frame["tvt_true"].to_numpy(np.float64)
    parent = frame[PARENT_COLUMN].to_numpy(np.float64)
    x_parts = []
    y_parts = []
    sampled_wells = 0
    for well, (indices, features) in cache.items():
        fold = int(frame.loc[indices[0], "fold"])
        if fold not in folds:
            continue
        selected = np.arange(0, len(indices), FIT_STRIDE, dtype=np.int64)
        if selected[-1] != len(indices) - 1:
            selected = np.r_[selected, len(indices) - 1]
        x_parts.append(features[selected])
        y_parts.append((truth[indices[selected]] - parent[indices[selected]]).astype(np.float32))
        sampled_wells += 1
    if sampled_wells == 0:
        raise RuntimeError(f"No training wells for folds {sorted(folds)}")
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite training matrix")
    return x, y, sampled_wells


def _fit_model(frame, cache, train_folds, ridge_fraction):
    x, y, wells = _training_matrix(frame, cache, train_folds)
    model = CorrectionModel(ridge_fraction).fit(x, y)
    report = {
        "train_folds": [int(value) for value in train_folds],
        "train_rows_sampled": int(len(x)),
        "train_wells": int(wells),
        "ridge_fraction": float(ridge_fraction),
        "alpha": float(model.model.alpha),
        "intercept": float(model.model.intercept_),
        "coefficient_l2": float(np.linalg.norm(model.model.coef_)),
    }
    return model, report


def _predict_fold(frame, cache, model, fold):
    output = np.full(len(frame), np.nan, np.float64)
    wells = 0
    for well, (indices, features) in cache.items():
        if int(frame.loc[indices[0], "fold"]) != int(fold):
            continue
        output[indices] = model.predict(features)
        wells += 1
    mask = frame["fold"].eq(int(fold)).to_numpy()
    if wells == 0 or not np.isfinite(output[mask]).all():
        raise RuntimeError(f"Incomplete prediction for Fold {fold}")
    return output, wells


def _apply(raw_correction, dose, cap_ft):
    return float(dose) * np.clip(
        np.asarray(raw_correction, np.float64), -float(cap_ft), float(cap_ft)
    )


def _rolled_control(frame, correction, folds):
    control = np.asarray(correction, np.float64).copy()
    target_folds = set(int(value) for value in folds)
    for _, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        if int(frame.loc[indices[0], "fold"]) not in target_folds:
            continue
        control[indices] = np.roll(control[indices], max(1, len(indices) // 2))
    return control


frame, maximum_truth_gap = _read_inputs()
required_folds = {1, 2, 3, 4, 5}
observed_folds = set(frame["fold"].astype(int).unique())
if observed_folds != required_folds:
    raise RuntimeError(f"Unexpected folds: {sorted(observed_folds)}")

print(
    "MULTISCALE CONFIG",
    {
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "endpoints": ENDPOINTS,
        "scales_ft": SMOOTH_SCALES_FT,
        "fit_stride": FIT_STRIDE,
    },
    flush=True,
)
cache, feature_names = _build_feature_cache(frame)
parent = frame[PARENT_COLUMN].to_numpy(np.float64)

# Fold 3 is the only hyperparameter selection surface.
selection_mask = frame["fold"].eq(SELECTION_FOLD).to_numpy()
selection_frame = frame.loc[selection_mask].reset_index(drop=True)
selection_models = {}
selection_model_reports = []
scan_rows = []
for ridge_fraction in RIDGE_FRACTIONS:
    model, report = _fit_model(frame, cache, (4, 5), ridge_fraction)
    raw, wells = _predict_fold(frame, cache, model, SELECTION_FOLD)
    report["predicted_wells"] = int(wells)
    selection_models[ridge_fraction] = model
    selection_model_reports.append(report)
    for dose in DOSES:
        for cap_ft in CAPS_FT:
            correction = _apply(raw[selection_mask], dose, cap_ft)
            prediction = parent[selection_mask] + correction
            scan_rows.append({
                "ridge_fraction": float(ridge_fraction),
                "dose": float(dose),
                "cap_ft": float(cap_ft),
                "mean_abs_move": float(np.mean(np.abs(correction))),
                **_metric(selection_frame, prediction, "candidate"),
            })
selection_scan = pd.DataFrame(scan_rows).sort_values(
    ["rmse", "p95_well_rmse", "mean_abs_move"], kind="stable"
).reset_index(drop=True)
selection_scan.to_csv(OUT_DIR / "fold3_selection_scan.csv", index=False)
selected = {
    "ridge_fraction": float(selection_scan.iloc[0]["ridge_fraction"]),
    "dose": float(selection_scan.iloc[0]["dose"]),
    "cap_ft": float(selection_scan.iloc[0]["cap_ft"]),
}

# Refit the frozen recipe before each genuinely held-out application fold.
fold_recipes = {
    CONFIRMATION_FOLD: (3, 4, 5),
    FINAL_FOLD: (2, 3, 4, 5),
}
raw_oof = np.full(len(frame), np.nan, np.float64)
correction_oof = np.full(len(frame), np.nan, np.float64)
fit_reports = []
for fold, train_folds in fold_recipes.items():
    model, report = _fit_model(
        frame, cache, train_folds, selected["ridge_fraction"]
    )
    raw, wells = _predict_fold(frame, cache, model, fold)
    mask = frame["fold"].eq(fold).to_numpy()
    raw_oof[mask] = raw[mask]
    correction_oof[mask] = _apply(raw[mask], selected["dose"], selected["cap_ft"])
    report.update({"heldout_fold": int(fold), "predicted_wells": int(wells)})
    fit_reports.append(report)

# Keep the Fold 3 selection prediction only for transparent reporting; it is not
# included in the primary Fold1+2 confirmation metric.
selection_model = selection_models[selected["ridge_fraction"]]
selection_raw, _ = _predict_fold(frame, cache, selection_model, SELECTION_FOLD)
raw_oof[selection_mask] = selection_raw[selection_mask]
correction_oof[selection_mask] = _apply(
    selection_raw[selection_mask], selected["dose"], selected["cap_ft"]
)

summary_rows = []
for fold, role in (
    (SELECTION_FOLD, "selection"),
    (CONFIRMATION_FOLD, "confirmation"),
    (FINAL_FOLD, "final"),
):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    summary_rows.append({"fold": fold, "role": role, **_metric(subset, parent[mask], "parent")})
    summary_rows.append({
        "fold": fold,
        "role": role,
        **_metric(subset, parent[mask] + correction_oof[mask], "multiscale"),
    })

primary_mask = frame["fold"].isin((FINAL_FOLD, CONFIRMATION_FOLD)).to_numpy()
primary_frame = frame.loc[primary_mask].reset_index(drop=True)
summary_rows.append({
    "fold": 0,
    "role": "fold1_plus_fold2_primary",
    **_metric(primary_frame, parent[primary_mask], "parent"),
})
summary_rows.append({
    "fold": 0,
    "role": "fold1_plus_fold2_primary",
    **_metric(
        primary_frame,
        parent[primary_mask] + correction_oof[primary_mask],
        "multiscale",
    ),
})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

rolled = _rolled_control(frame, correction_oof, (FINAL_FOLD, CONFIRMATION_FOLD))
control_rows = []
for fold, role in ((CONFIRMATION_FOLD, "confirmation"), (FINAL_FOLD, "final")):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in {
        "parent": parent[mask],
        "multiscale": parent[mask] + correction_oof[mask],
        "opposite_move": parent[mask] - correction_oof[mask],
        "rolled_within_well": parent[mask] + rolled[mask],
    }.items():
        control_rows.append({
            "fold": fold, "role": role, **_metric(subset, prediction, endpoint)
        })
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)

def _summary_metric(fold, endpoint, column):
    row = summary[(summary["fold"].eq(fold)) & (summary["endpoint"].eq(endpoint))]
    return float(row.iloc[0][column])


fold_gain = {
    fold: _summary_metric(fold, "parent", "rmse")
    - _summary_metric(fold, "multiscale", "rmse")
    for fold in (SELECTION_FOLD, CONFIRMATION_FOLD, FINAL_FOLD, 0)
}
fold_far_change = {
    fold: _summary_metric(fold, "multiscale", "far_gt2000_rmse")
    - _summary_metric(fold, "parent", "far_gt2000_rmse")
    for fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
fold_p95_change = {
    fold: _summary_metric(fold, "multiscale", "p95_well_rmse")
    - _summary_metric(fold, "parent", "p95_well_rmse")
    for fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
control_gain = {}
for fold in (CONFIRMATION_FOLD, FINAL_FOLD):
    table = controls[controls["fold"].eq(fold)].set_index("endpoint")
    control_gain[fold] = float(
        table.loc["parent", "rmse"] - table.loc["rolled_within_well", "rmse"]
    )

gates = {
    "fold2_gain_ge_0p08": bool(fold_gain[CONFIRMATION_FOLD] >= 0.08),
    "fold1_gain_ge_0p08": bool(fold_gain[FINAL_FOLD] >= 0.08),
    "fold12_gain_ge_0p08": bool(fold_gain[0] >= 0.08),
    "fold2_far_degradation_le_0p02": bool(fold_far_change[CONFIRMATION_FOLD] <= 0.02),
    "fold1_far_degradation_le_0p02": bool(fold_far_change[FINAL_FOLD] <= 0.02),
    "fold2_p95_degradation_le_0p10": bool(fold_p95_change[CONFIRMATION_FOLD] <= 0.10),
    "fold1_p95_degradation_le_0p10": bool(fold_p95_change[FINAL_FOLD] <= 0.10),
    "rolled_control_retains_le_30pct_fold2": bool(
        control_gain[CONFIRMATION_FOLD]
        <= 0.30 * max(fold_gain[CONFIRMATION_FOLD], 1e-9)
    ),
    "rolled_control_retains_le_30pct_fold1": bool(
        control_gain[FINAL_FOLD]
        <= 0.30 * max(fold_gain[FINAL_FOLD], 1e-9)
    ),
}

row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["parent"] = parent.astype(np.float32)
row_output["raw_multiscale_correction"] = raw_oof.astype(np.float32)
row_output["multiscale_correction"] = correction_oof.astype(np.float32)
row_output["multiscale_prediction"] = (parent + correction_oof).astype(np.float32)
row_output.to_parquet(
    OUT_DIR / "fold123_multiscale_oof.parquet", index=False, compression="zstd"
)

# Save an all-well refit for a later Kaggle integration cell.  This artifact is
# never used to score the nested audit above.
deployment_model, deployment_report = _fit_model(
    frame, cache, (1, 2, 3, 4, 5), selected["ridge_fraction"]
)
bundle = {
    "experiment": EXPERIMENT,
    "selected": selected,
    "endpoints": list(ENDPOINTS),
    "smooth_scales_ft": list(SMOOTH_SCALES_FT),
    "feature_names": feature_names,
    "scaler": deployment_model.scaler,
    "model": deployment_model.model,
    "parent_column": PARENT_COLUMN,
}
joblib.dump(bundle, OUT_DIR / "multiscale_fusion_bundle.joblib", compress=3)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_selection_Fold2_confirmation_Fold1_final_multiscale_endpoint_fusion",
    "zero_model_or_pf_forwards": True,
    "hidden_predictions_observed": False,
    "inputs": {"core": str(CORE_PATH), "full_parent": str(FULL_PATH)},
    "alignment": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "maximum_truth_gap": maximum_truth_gap,
    },
    "protocol": {
        "feature_endpoints": list(ENDPOINTS),
        "smooth_scales_ft": list(SMOOTH_SCALES_FT),
        "fit_stride": FIT_STRIDE,
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "Fold1_tuning_or_search_count": 0,
        "Fold2_tuning_or_search_count": 0,
    },
    "selected": selected,
    "selection_search_count": int(len(selection_scan)),
    "selection_model_reports": selection_model_reports,
    "heldout_fit_reports": fit_reports,
    "deployment_refit_report_not_used_for_audit": deployment_report,
    "fold_gain": {str(key): float(value) for key, value in fold_gain.items()},
    "fold_far_change": {str(key): float(value) for key, value in fold_far_change.items()},
    "fold_p95_change": {str(key): float(value) for key, value in fold_p95_change.items()},
    "rolled_control_gain": {str(key): float(value) for key, value in control_gain.items()},
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "test_well_optimizer_or_search_count": 0,
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> integrate the frozen bundle after the full 6.39 parent writer"
        if all(gates.values())
        else "FAIL -> close multiscale endpoint fusion and move to prefix-diagnostic calibration"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

display_columns = [
    "fold", "role", "endpoint", "rows", "wells", "rmse", "mae",
    "median_well_rmse", "p95_well_rmse", "near_0_500_rmse",
    "mid_500_1000_rmse", "mid_1000_2000_rmse", "far_gt2000_rmse",
]
print("MULTISCALE FOLD3 TOP 20")
print(selection_scan.head(20).to_string(index=False), flush=True)
print("\nMULTISCALE SUMMARY")
print(summary[display_columns].to_string(index=False), flush=True)
print("\nMATCHED CONTROLS")
print(controls[display_columns].to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
