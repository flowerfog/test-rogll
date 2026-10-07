"""Nested legal confidence gate for the frozen V34 tail-SGM correction.

V36 truth is used only as a training target.  Every reported held-fold dose is
produced by a model that never saw that well.  Model/config/mode selection is
performed on an inner cross-fit over the two remaining folds.
"""

from pathlib import Path
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_tail_sgm_legal_dose_gate_v37"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V34_DIR = OOF_ROOT / "frontier_6p377_tail_sgm_residual_v34"
V36_DIR = OOF_ROOT / "frontier_6p377_tail_sgm_dose_oracle_v36"
SOURCE = V34_DIR / "tail_sgm_oof.parquet"
DIAGNOSTICS = V34_DIR / "well_diagnostics.csv"
CHOICES = V36_DIR / "truth_assisted_well_choices.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_DOSE = 0.0375
MAX_DOSE = 0.10
START_FT = 1000.0
FOLDS = (1, 2, 3)
THRESHOLDS = tuple(np.round(np.arange(0.20, 0.801, 0.05), 2))
CONFIGS = (
    {"id": "leaf7_d2", "num_leaves": 7, "max_depth": 2, "min_child_samples": 25},
    {"id": "leaf11_d3", "num_leaves": 11, "max_depth": 3, "min_child_samples": 20},
    {"id": "leaf15_d3", "num_leaves": 15, "max_depth": 3, "min_child_samples": 15},
    {"id": "leaf15_d4", "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
)
started = time.time()


def metric(local, prediction):
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    well_mse = pd.Series(error**2).groupby(local["well"].to_numpy(), sort=False).mean()
    well_rmse = np.sqrt(well_mse.to_numpy(np.float64))
    near = distance <= START_FT
    far = distance > 2000.0
    return {
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) < 1e-6:
        return 0.0
    xc = x[valid] - np.mean(x[valid])
    return float(np.dot(xc, y[valid] - np.mean(y[valid])) / max(np.dot(xc, xc), 1e-12))


def summarize_well(local):
    local = local.sort_values("row_idx", kind="stable")
    d = local["md_since"].to_numpy(np.float64)
    c = local["v34_tail_correction"].to_numpy(np.float64)
    b = local["v33_current_dose_0p20"].to_numpy(np.float64)
    tail = d > START_FT
    tail_count = int(tail.sum())
    tail_c = c[tail] if tail_count else np.zeros(1, np.float64)
    tail_b = b[tail] if tail_count else np.zeros(1, np.float64)
    tail_d = d[tail] if tail_count else np.zeros(1, np.float64)
    dc = np.diff(tail_c) if tail_count > 1 else np.zeros(1)
    db = np.diff(tail_b) if tail_count > 1 else np.zeros(1)
    signs = np.sign(tail_c)
    nonzero = signs != 0
    sign_flips = np.sum(signs[nonzero][1:] != signs[nonzero][:-1]) if nonzero.sum() > 1 else 0
    near_rows = np.flatnonzero(d <= START_FT)
    reference_index = int(near_rows[-1]) if len(near_rows) else 0
    result = {
        "well": str(local["well"].iloc[0]),
        "fold": int(local["fold"].iloc[0]),
        "rows": int(len(local)),
        "max_md_since": float(np.max(d)),
        "tail_fraction": float(np.mean(tail)),
        "has_tail": float(tail_count > 0),
        "tail_rows": float(tail_count),
        "correction_mean": float(np.mean(tail_c)) if tail_count else 0.0,
        "correction_std": float(np.std(tail_c)) if tail_count else 0.0,
        "correction_rms": float(np.sqrt(np.mean(tail_c ** 2))) if tail_count else 0.0,
        "correction_mean_abs": float(np.mean(np.abs(tail_c))) if tail_count else 0.0,
        "correction_p95_abs": float(np.quantile(np.abs(tail_c), 0.95)) if tail_count else 0.0,
        "correction_max_abs": float(np.max(np.abs(tail_c))) if tail_count else 0.0,
        "correction_endpoint": float(c[-1]) if tail_count else 0.0,
        "correction_slope": slope(tail_d, tail_c) if tail_count else 0.0,
        "correction_total_variation": float(np.sum(np.abs(dc)) / max(tail_count, 1)),
        "correction_sign_flips": float(sign_flips / max(tail_count, 1)),
        "base_tail_slope": slope(tail_d, tail_b) if tail_count else 0.0,
        "base_tail_std": float(np.std(tail_b)) if tail_count else 0.0,
        "base_tail_total_variation": float(np.sum(np.abs(db)) / max(tail_count, 1)),
        "base_endpoint_minus_1000": float(b[-1] - b[reference_index]) if tail_count else 0.0,
    }
    for lo, hi, name in (
        (1000.0, 2000.0, "1000_2000"),
        (2000.0, 1e20, "gt2000"),
    ):
        mask = (d > lo) & (d <= hi)
        result[f"correction_rms_{name}"] = float(np.sqrt(np.mean(c[mask] ** 2))) if mask.any() else 0.0
        result[f"correction_mean_{name}"] = float(np.mean(c[mask])) if mask.any() else 0.0
        result[f"correction_endpoint_{name}"] = float(c[np.flatnonzero(mask)[-1]]) if mask.any() else 0.0
    return result


def fit_model(config, x, target, seed):
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=160,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
        **{key: value for key, value in config.items() if key != "id"},
    )
    model.fit(x, target)
    return model


def doses_from_score(score, mode, threshold):
    score = np.clip(np.asarray(score, np.float64), 0.0, 1.0)
    if mode == "soft":
        return MAX_DOSE * score
    if mode == "hard":
        return MAX_DOSE * (score >= threshold)
    if mode == "ternary":
        return np.where(score < threshold, 0.0, np.where(score < 0.5 * (1.0 + threshold), FIXED_DOSE, MAX_DOSE))
    raise ValueError(mode)


def row_prediction(local, dose_by_well):
    dose = local["well"].map(dose_by_well).to_numpy(np.float64)
    return (
        local["v33_current_dose_0p20"].to_numpy(np.float64)
        + dose * local["v34_tail_correction"].to_numpy(np.float64)
    )


for path in (SOURCE, DIAGNOSTICS, CHOICES):
    if not path.is_file():
        raise FileNotFoundError(path)
frame = pd.read_parquet(SOURCE).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
diagnostics = pd.read_csv(DIAGNOSTICS)
choices = pd.read_csv(CHOICES)
choices = choices[choices["label"].eq("primary_three_dose")].copy()
if choices["well"].duplicated().any():
    raise RuntimeError("Duplicate V36 primary choices")

feature_frame = pd.DataFrame([summarize_well(local) for _, local in frame.groupby("well", sort=True)])
feature_frame = feature_frame.merge(diagnostics, on=["well", "fold"], how="left", validate="one_to_one")
feature_frame = feature_frame.merge(
    choices[["well", "chosen_dose", "oracle_gain_vs_fixed_mse"]],
    on="well", how="left", validate="one_to_one",
)
if feature_frame[["chosen_dose", "oracle_gain_vs_fixed_mse"]].isna().any().any():
    raise RuntimeError("V36 choice alignment failed")
feature_frame["target"] = feature_frame["chosen_dose"] / MAX_DOSE

excluded = {"well", "fold", "chosen_dose", "oracle_gain_vs_fixed_mse", "target"}
features = [column for column in feature_frame.columns if column not in excluded]
feature_frame[features] = feature_frame[features].replace([np.inf, -np.inf], np.nan)
feature_frame[features] = feature_frame[features].fillna(feature_frame[features].median()).fillna(0.0)
feature_frame.to_csv(OUT_DIR / "legal_well_features.csv", index=False)

association_rows = []
for feature in features:
    association_rows.append({
        "feature": feature,
        "spearman_dose": float(feature_frame[feature].corr(feature_frame["chosen_dose"], method="spearman")),
        "spearman_oracle_gain": float(feature_frame[feature].corr(feature_frame["oracle_gain_vs_fixed_mse"], method="spearman")),
    })
associations = pd.DataFrame(association_rows).sort_values("spearman_dose", key=np.abs, ascending=False)
associations.to_csv(OUT_DIR / "feature_associations.csv", index=False)

outer_predictions = []
fold_reports = []
rng = np.random.default_rng(20260727)
for outer_fold in FOLDS:
    train_folds = [fold for fold in FOLDS if fold != outer_fold]
    train = feature_frame[feature_frame["fold"].isin(train_folds)].copy()
    valid = feature_frame[feature_frame["fold"].eq(outer_fold)].copy()
    inner_score = np.zeros(len(train), np.float64)
    config_rows = []
    for config in CONFIGS:
        score = np.zeros(len(train), np.float64)
        for inner_valid_fold in train_folds:
            inner_train = train[train["fold"].ne(inner_valid_fold)]
            inner_valid = train[train["fold"].eq(inner_valid_fold)]
            model = fit_model(config, inner_train[features], inner_train["target"], 1000 + outer_fold * 10 + inner_valid_fold)
            score[train.index.get_indexer(inner_valid.index)] = model.predict(inner_valid[features])
        for mode in ("soft", "hard", "ternary"):
            thresholds = (0.5,) if mode == "soft" else THRESHOLDS
            for threshold in thresholds:
                doses = doses_from_score(score, mode, threshold)
                prediction = row_prediction(frame[frame["fold"].isin(train_folds)], dict(zip(train["well"], doses)))
                value = metric(frame[frame["fold"].isin(train_folds)].reset_index(drop=True), prediction)
                config_rows.append({
                    "config": config["id"], "mode": mode, "threshold": threshold,
                    "rmse": value["rmse"], "mean_dose": float(np.mean(doses)),
                })
    inner_scan = pd.DataFrame(config_rows).sort_values("rmse", kind="stable")
    selected = inner_scan.iloc[0]
    selected_config = next(config for config in CONFIGS if config["id"] == selected["config"])
    final_model = fit_model(selected_config, train[features], train["target"], 2000 + outer_fold)
    valid_score = final_model.predict(valid[features])
    valid_dose = doses_from_score(valid_score, selected["mode"], float(selected["threshold"]))
    dose_by_well = dict(zip(valid["well"], valid_dose))
    local = frame[frame["fold"].eq(outer_fold)].reset_index(drop=True)
    prediction = row_prediction(local, dose_by_well)

    shuffled_dose = valid_dose.copy()
    rng.shuffle(shuffled_dose)
    shuffled_prediction = row_prediction(local, dict(zip(valid["well"], shuffled_dose)))
    base_prediction = local["v33_current_dose_0p20"].to_numpy(np.float64)
    fixed_prediction = base_prediction + FIXED_DOSE * local["v34_tail_correction"].to_numpy(np.float64)
    report = {
        "fold": outer_fold,
        "selected_config": str(selected["config"]),
        "selected_mode": str(selected["mode"]),
        "selected_threshold": float(selected["threshold"]),
        "inner_rmse": float(selected["rmse"]),
        "mean_dose": float(np.mean(valid_dose)),
        "base": metric(local, base_prediction),
        "fixed": metric(local, fixed_prediction),
        "adaptive": metric(local, prediction),
        "shuffle": metric(local, shuffled_prediction),
    }
    report["gain_vs_fixed"] = report["fixed"]["rmse"] - report["adaptive"]["rmse"]
    report["far_gain_vs_fixed"] = report["fixed"]["far_gt2000_rmse"] - report["adaptive"]["far_gt2000_rmse"]
    report["p95_change_vs_fixed"] = report["adaptive"]["p95_well_rmse"] - report["fixed"]["p95_well_rmse"]
    report["shuffle_loss"] = report["shuffle"]["rmse"] - report["adaptive"]["rmse"]
    fold_reports.append(report)
    outer_predictions.append(pd.DataFrame({
        "well": valid["well"].to_numpy(), "fold": outer_fold,
        "score": np.clip(valid_score, 0.0, 1.0), "dose": valid_dose,
    }))
    inner_scan.to_csv(OUT_DIR / f"fold{outer_fold}_inner_scan.csv", index=False)

well_prediction = pd.concat(outer_predictions, ignore_index=True)
well_prediction.to_csv(OUT_DIR / "crossfit_well_doses.csv", index=False)
dose_map = dict(zip(well_prediction["well"], well_prediction["dose"]))
adaptive_prediction = row_prediction(frame, dose_map)
base_prediction = frame["v33_current_dose_0p20"].to_numpy(np.float64)
fixed_prediction = base_prediction + FIXED_DOSE * frame["v34_tail_correction"].to_numpy(np.float64)
overall = {
    "base": metric(frame, base_prediction),
    "fixed": metric(frame, fixed_prediction),
    "adaptive": metric(frame, adaptive_prediction),
}
overall["gain_vs_fixed"] = overall["fixed"]["rmse"] - overall["adaptive"]["rmse"]
overall["far_gain_vs_fixed"] = overall["fixed"]["far_gt2000_rmse"] - overall["adaptive"]["far_gt2000_rmse"]
overall["p95_change_vs_fixed"] = overall["adaptive"]["p95_well_rmse"] - overall["fixed"]["p95_well_rmse"]

minimum_fold_gain = min(report["gain_vs_fixed"] for report in fold_reports)
minimum_shuffle_loss = min(report["shuffle_loss"] for report in fold_reports)
maximum_p95_change = max(report["p95_change_vs_fixed"] for report in fold_reports)
strict_pass = bool(
    overall["gain_vs_fixed"] >= 0.03
    and minimum_fold_gain >= -0.01
    and minimum_shuffle_loss >= 0.01
    and maximum_p95_change <= 0.10
)

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["v33_base"] = base_prediction.astype(np.float32)
output["v35_fixed"] = fixed_prediction.astype(np.float32)
output["v37_dose"] = output["well"].map(dose_map).to_numpy(np.float32)
output["v37_adaptive"] = adaptive_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "legal_dose_gate_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_outer_crossfit_twofold_inner_model_and_threshold_selection",
    "deployable_inputs_only": True,
    "truth_usage": "training target only; no held-fold truth used for model/config/mode/threshold selection",
    "features": features,
    "overall": overall,
    "folds": fold_reports,
    "checks": {
        "overall_gain_vs_fixed_ge_0p03": overall["gain_vs_fixed"] >= 0.03,
        "minimum_fold_gain_ge_minus0p01": minimum_fold_gain >= -0.01,
        "minimum_shuffle_loss_ge_0p01": minimum_shuffle_loss >= 0.01,
        "maximum_p95_change_le_0p10": maximum_p95_change <= 0.10,
    },
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> export the selected fold models and reproduce identical features in the Kaggle V35 addon"
        if strict_pass else
        "FAIL -> legal diagnostics cannot reach the V36 oracle; retain fixed V35 and close adaptive dosing"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("V37 TOP FEATURE ASSOCIATIONS")
print(associations.head(25).to_string(index=False), flush=True)
print("\nV37 FOLD REPORTS")
for report in fold_reports:
    print(json.dumps(report, indent=2), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
