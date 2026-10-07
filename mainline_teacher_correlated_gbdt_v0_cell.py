"""Fold-clean teacher-anchored GBDT residual gateway.

Purpose
-------
Train a deliberately high-error-correlation GBDT correction on top of the
current strongest endpoint.  This is a replacement/correction experiment, not
a decorrelated ensemble experiment.

The preferred teacher is an exact fold-clean OOF export of the 6.781 mainline.
When that artifact is unavailable, the cell explicitly falls back to the best
available fold-clean strong/neighbor proxy.  A proxy run can validate the
mechanism but cannot establish a leaderboard gain.

Protocol
--------
* outer validation is grouped by complete physical well;
* one additional training fold is reserved for early stopping in every outer
  fold, so the held outer fold is not used to choose tree count;
* the GBDT predicts truth - teacher, using the existing deployable full feature
  stack plus teacher disagreement channels;
* correction dose is selected on the other four OOF folds and then applied to
  the held fold;
* promotion requires both an RMSE gain and >=0.90 raw/distance-detrended error
  correlation with the teacher;
* a matched per-well sign-flip control must lose the measured gain.

Run this file as one AutoDL notebook cell after the feature cache and the V4
strong proxy (or exact mainline OOF) have been uploaded/generated.
"""

from pathlib import Path
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.signal import savgol_filter


warnings.filterwarnings("ignore")

EXPERIMENT = "mainline_teacher_correlated_gbdt_v0"
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = CACHE_ROOT / "features"
FEATURE_LIST_PATH = CACHE_ROOT / "all_features.csv"
FOLD_PATH = CACHE_ROOT / "well_metadata.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_CANDIDATES = [
    Path("/root/autodl-tmp/mainline_6p781_oof.parquet"),
    OOF_ROOT / "mainline_6p781_oof" / "mainline_6p781_oof.parquet",
    OOF_ROOT
    / "track8_stride_strong_nested_fusion_v4"
    / "best_nested_predictions.parquet",
]
TEACHER_PREDICTION_PREFERENCE = [
    "mainline_6p781",
    "mainline_tvt",
    "mainline",
    "prediction",
    "tvt_pred",
    "strong_proxy",
]
TRUTH_PREFERENCE = ["truth", "tvt_true", "true_tvt"]

SEED = 20260723
FOLDS = [1, 2, 3, 4, 5]
MAX_TREES = 2200
PATIENCE = 160
MODELS_PER_FOLD = 2
MIN_WELLS = 770
MIN_ALIGNMENT = 0.999
TARGET_CLIP_FT = 80.0
SMOOTH_WINDOW = 81
SMOOTH_POLY = 3

DOSE_GRID = np.array(
    [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0],
    dtype=np.float64,
)
MIN_RAW_RHO = 0.90
MIN_DETRENDED_RHO = 0.90
PASS_GAIN_FT = 0.05
PASS_MIN_FOLD_GAIN_FT = -0.05
PASS_POSITIVE_FOLDS = 4
PASS_CONTROL_LOSS_FT = 0.02

TEACHER_FEATURES = [
    "tc_teacher_delta_last",
    "tc_teacher_minus_pf",
    "tc_abs_teacher_minus_pf",
    "tc_teacher_minus_lik3",
    "tc_abs_teacher_minus_lik3",
    "tc_teacher_minus_lik5",
    "tc_abs_teacher_minus_lik5",
    "tc_teacher_minus_lik8",
    "tc_abs_teacher_minus_lik8",
    "tc_teacher_minus_lik12",
    "tc_abs_teacher_minus_lik12",
    "tc_teacher_delta_x_sqrt_progress",
]


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


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _safe_rho(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    left = left[valid]
    right = right[valid]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _error_cosine(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _distance_detrend(error, md_since):
    error = np.asarray(error, np.float64)
    distance = np.asarray(md_since, np.float64)
    x = np.log1p(np.maximum(distance, 0.0))
    scale = max(float(np.nanmax(x)), 1.0)
    x = x / scale
    valid = np.isfinite(error) & np.isfinite(x)
    if int(valid.sum()) < 16:
        return error.copy()
    coefficients = np.polyfit(x[valid], error[valid], deg=3)
    trend = np.polyval(coefficients, x)
    return error - trend


def _correlations(truth, teacher, candidate, md_since):
    teacher_error = np.asarray(teacher, np.float64) - np.asarray(truth, np.float64)
    candidate_error = np.asarray(candidate, np.float64) - np.asarray(truth, np.float64)
    return {
        "raw_signed_error_rho": _safe_rho(teacher_error, candidate_error),
        "raw_error_cosine": _error_cosine(teacher_error, candidate_error),
        "distance_detrended_error_rho": _safe_rho(
            _distance_detrend(teacher_error, md_since),
            _distance_detrend(candidate_error, md_since),
        ),
    }


def _metrics(frame, prediction):
    truth = frame["truth"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    codes, names = pd.factorize(frame["well"].astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error**2, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    distance = frame["md_since"].to_numpy(np.float64)

    def band(low, high=None):
        mask = distance >= low
        if high is not None:
            mask &= distance < high
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else float("nan")

    return {
        "rows": int(len(frame)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "near_0_500_rmse": band(0.0, 500.0),
        "mid_500_1000_rmse": band(500.0, 1000.0),
        "mid_1000_2000_rmse": band(1000.0, 2000.0),
        "far_gt2000_rmse": band(2000.0),
    }


def _matrix(frame, columns):
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _feature_paths(well):
    return FEATURE_DIR / f"{well}__sample.pkl", FEATURE_DIR / f"{well}__full.pkl"


def _load_teacher():
    path = next((candidate for candidate in TEACHER_CANDIDATES if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "No teacher OOF found. Expected exact mainline_6p781_oof.parquet or "
            "track8_stride_strong_nested_fusion_v4/best_nested_predictions.parquet"
        )
    table = pd.read_parquet(path)
    if "id" not in table:
        if "well" not in table or "row_idx" not in table:
            raise RuntimeError(f"Teacher lacks id or well/row_idx: {table.columns.tolist()}")
        table["id"] = [
            f"{well}_{int(row)}" for well, row in zip(table["well"], table["row_idx"])
        ]
    table["id"] = table["id"].astype(str).str.lower()
    if "well" not in table:
        table["well"] = table["id"].str.rsplit("_", n=1).str[0]
    if "row_idx" not in table:
        table["row_idx"] = table["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    table["well"] = table["well"].astype(str).str.lower()
    table["row_idx"] = table["row_idx"].astype(np.int32)
    prediction_column = next(
        (column for column in TEACHER_PREDICTION_PREFERENCE if column in table), None
    )
    if prediction_column is None:
        raise RuntimeError(
            "Cannot identify teacher prediction from columns: " + repr(table.columns.tolist())
        )
    truth_column = next((column for column in TRUTH_PREFERENCE if column in table), None)
    keep = ["id", "well", "row_idx", prediction_column]
    if truth_column is not None and truth_column not in keep:
        keep.append(truth_column)
    if "fold" in table:
        keep.append("fold")
    table = table[keep].copy()
    table = table.rename(columns={prediction_column: "teacher"})
    if truth_column is not None:
        table = table.rename(columns={truth_column: "teacher_truth"})
    if table["id"].duplicated().any():
        raise RuntimeError("Teacher OOF contains duplicate ids")
    if not np.isfinite(table["teacher"].to_numpy(np.float64)).all():
        raise RuntimeError("Teacher OOF contains non-finite predictions")
    exact = prediction_column != "strong_proxy" and "6p781" in str(path).lower()
    return path, prediction_column, exact, table


def _attach_teacher(frame, well):
    aligned = teacher_groups[str(well)].reindex(frame["_row_idx"].to_numpy(np.int32))
    coverage = float(aligned["teacher"].notna().mean())
    if coverage < MIN_ALIGNMENT:
        raise RuntimeError(f"{well}: teacher coverage={coverage:.6%}")
    truth = (
        frame["target"].to_numpy(np.float64)
        + frame["last_known_tvt"].to_numpy(np.float64)
    )
    if "teacher_truth" in aligned:
        gap = np.abs(truth - aligned["teacher_truth"].to_numpy(np.float64))
        if not np.isfinite(gap).all() or float(np.max(gap)) > 0.01:
            raise RuntimeError(f"{well}: teacher/cache truth gap={float(np.nanmax(gap)):.6f}")

    teacher = aligned["teacher"].to_numpy(np.float32)
    last = frame["last_known_tvt"].to_numpy(np.float32)
    pf = frame["pf_ancc"].to_numpy(np.float32)
    progress = np.sqrt(np.maximum(frame["md_since"].to_numpy(np.float32), 0.0) + 1.0)
    frame["teacher"] = teacher
    frame["correction_target"] = np.clip(truth - teacher, -TARGET_CLIP_FT, TARGET_CLIP_FT)
    frame["tc_teacher_delta_last"] = teacher - last
    frame["tc_teacher_minus_pf"] = teacher - pf
    frame["tc_abs_teacher_minus_pf"] = np.abs(teacher - pf)
    for suffix, source in (
        ("lik3", "likpf_scale_3"),
        ("lik5", "likpf_scale_5"),
        ("lik8", "likpf_scale_8"),
        ("lik12", "likpf_scale_12"),
    ):
        reference = frame[source].to_numpy(np.float32)
        difference = teacher - reference
        frame[f"tc_teacher_minus_{suffix}"] = difference
        frame[f"tc_abs_teacher_minus_{suffix}"] = np.abs(difference)
    frame["tc_teacher_delta_x_sqrt_progress"] = (teacher - last) * progress / 100.0
    return frame


def _load_frames(wells, kind):
    position = 0 if kind == "sample" else 1
    parts = []
    for well in wells:
        path = _feature_paths(str(well))[position]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path)
        frame["well"] = str(well)
        parts.append(_attach_teacher(frame, str(well)))
    return pd.concat(parts, ignore_index=True, sort=False)


def _new_model(outer_fold, seed_index):
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=MAX_TREES,
        learning_rate=0.025,
        num_leaves=48,
        min_child_samples=120,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.25,
        reg_lambda=20.0,
        max_bin=255,
        n_jobs=-1,
        random_state=SEED + 100 * int(outer_fold) + int(seed_index),
        verbosity=-1,
    )


def _fit_one(train, tune, columns, outer_fold, seed_index):
    model = _new_model(outer_fold, seed_index)
    x_train = _matrix(train, columns)
    x_tune = _matrix(tune, columns)
    y_train = train["correction_target"].to_numpy(np.float32)
    y_tune = tune["correction_target"].to_numpy(np.float32)
    base_weight = train.get(
        "sample_weight", pd.Series(np.ones(len(train), np.float32))
    ).to_numpy(np.float32)
    tail_weight = 1.0 + 0.5 * np.minimum(np.abs(y_train) / 20.0, 2.0)
    fit_kwargs = dict(
        sample_weight=base_weight * tail_weight,
        eval_set=[(x_tune, y_tune)],
        eval_metric="rmse",
        callbacks=[early_stopping(PATIENCE, verbose=False), log_evaluation(0)],
    )
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        candidate = LGBMRegressor(**params)
        candidate.fit(x_train, y_train, **fit_kwargs)
        model = candidate
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("LightGBM GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model.fit(x_train, y_train, **fit_kwargs)
    del x_train, x_tune
    gc.collect()
    return model


def _smooth_by_well(frame, correction):
    correction = np.asarray(correction, np.float64)
    output = correction.copy()
    for _, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        order = np.argsort(frame.iloc[indices]["md_since"].to_numpy(np.float64))
        ordered_indices = indices[order]
        values = correction[ordered_indices]
        window = min(SMOOTH_WINDOW, len(values))
        if window % 2 == 0:
            window -= 1
        if window >= SMOOTH_POLY + 2:
            output[ordered_indices] = savgol_filter(values, window, SMOOTH_POLY)
    return output


def _select_dose(selection):
    truth = selection["truth"].to_numpy(np.float64)
    teacher = selection["teacher"].to_numpy(np.float64)
    correction = selection["correction"].to_numpy(np.float64)
    md_since = selection["md_since"].to_numpy(np.float64)
    rows = []
    for dose in DOSE_GRID:
        prediction = teacher + float(dose) * correction
        rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        corr = _correlations(truth, teacher, prediction, md_since)
        rows.append({"dose": float(dose), "rmse": rmse, **corr})
    scan = pd.DataFrame(rows)
    allowed = scan[
        scan["raw_signed_error_rho"].ge(MIN_RAW_RHO)
        & scan["distance_detrended_error_rho"].ge(MIN_DETRENDED_RHO)
    ]
    if allowed.empty:
        selected = scan.loc[scan["dose"].eq(0.0)].iloc[0]
    else:
        selected = allowed.sort_values(["rmse", "dose"], ascending=[True, True]).iloc[0]
    return float(selected["dose"]), scan


def _matched_sign_flip(frame, correction):
    rng = np.random.default_rng(SEED + 991)
    wells = frame["well"].astype(str).drop_duplicates().tolist()
    signs = {well: float(rng.choice([-1.0, 1.0])) for well in wells}
    multiplier = frame["well"].astype(str).map(signs).to_numpy(np.float64)
    return np.asarray(correction, np.float64) * multiplier


started = time.time()
for required in (FEATURE_DIR, FEATURE_LIST_PATH, FOLD_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

teacher_path, teacher_column, teacher_is_exact_mainline, teacher_table = _load_teacher()
fold_metadata = pd.read_csv(FOLD_PATH, usecols=["well", "fold"])
fold_metadata["well"] = fold_metadata["well"].astype(str).str.lower()
fold_metadata["fold"] = fold_metadata["fold"].astype(int)
fold_by_well = fold_metadata.drop_duplicates("well").set_index("well")["fold"]

teacher_groups = {
    well: group.set_index("row_idx").sort_index()
    for well, group in teacher_table.groupby("well", sort=False)
}
eligible_wells = sorted(
    well
    for well in fold_by_well.index
    if well in teacher_groups and all(path.exists() for path in _feature_paths(well))
)
if len(eligible_wells) < MIN_WELLS:
    raise RuntimeError(f"Only {len(eligible_wells)} wells have teacher plus feature cache")
fold_counts = {
    fold: int(sum(int(fold_by_well[well]) == fold for well in eligible_wells))
    for fold in FOLDS
}
if min(fold_counts.values()) < 150:
    raise RuntimeError(f"Unexpected fold coverage: {fold_counts}")

listed_features = pd.read_csv(FEATURE_LIST_PATH)["feature"].astype(str).tolist()
schema = pd.read_pickle(_feature_paths(eligible_wells[0])[0])
base_features = [
    column
    for column in listed_features
    if column in schema
    and pd.api.types.is_numeric_dtype(schema[column])
    and column
    not in {
        "target",
        "TVT",
        "TVT_input",
        "_row_idx",
        "fold",
        "sample_weight",
    }
]
model_features = base_features + TEACHER_FEATURES
if len(base_features) < 100:
    raise RuntimeError(f"Too few base features: {len(base_features)}")

print(
    "TEACHER-CORRELATED GBDT CONFIG",
    {
        "teacher": str(teacher_path),
        "teacher_column": teacher_column,
        "teacher_is_exact_6p781": teacher_is_exact_mainline,
        "wells": len(eligible_wells),
        "fold_counts": fold_counts,
        "base_features": len(base_features),
        "teacher_features": len(TEACHER_FEATURES),
        "models_per_fold": MODELS_PER_FOLD,
    },
    flush=True,
)

oof_parts = []
model_rows = []
for outer_fold in FOLDS:
    tune_fold = FOLDS[outer_fold % len(FOLDS)]
    fit_wells = [
        well
        for well in eligible_wells
        if int(fold_by_well[well]) not in {outer_fold, tune_fold}
    ]
    tune_wells = [
        well for well in eligible_wells if int(fold_by_well[well]) == tune_fold
    ]
    valid_wells = [
        well for well in eligible_wells if int(fold_by_well[well]) == outer_fold
    ]
    print(
        f"\nouter={outer_fold} fit={len(fit_wells)} tune={len(tune_wells)} "
        f"valid={len(valid_wells)}",
        flush=True,
    )
    train = _load_frames(fit_wells, "sample")
    tune = _load_frames(tune_wells, "sample")
    valid = _load_frames(valid_wells, "full")
    corrections = []
    for seed_index in range(MODELS_PER_FOLD):
        model = _fit_one(train, tune, model_features, outer_fold, seed_index)
        correction = model.predict(
            _matrix(valid, model_features), num_iteration=model.best_iteration_
        ).astype(np.float64)
        corrections.append(correction)
        model_rows.append(
            {
                "outer_fold": int(outer_fold),
                "tune_fold": int(tune_fold),
                "seed_index": int(seed_index),
                "best_iteration": int(model.best_iteration_),
            }
        )
        joblib.dump(
            model,
            MODEL_DIR / f"teacher_correction_fold{outer_fold}_seed{seed_index}.joblib",
        )
    correction = np.mean(np.vstack(corrections), axis=0)
    correction = _smooth_by_well(valid, correction)
    truth = valid["target"].to_numpy(np.float64) + valid["last_known_tvt"].to_numpy(np.float64)
    oof_parts.append(
        pd.DataFrame(
            {
                "id": valid["id"].astype(str).str.lower().to_numpy(),
                "well": valid["well"].astype(str).str.lower().to_numpy(),
                "row_idx": valid["_row_idx"].to_numpy(np.int32),
                "fold": int(outer_fold),
                "truth": truth,
                "md_since": valid["md_since"].to_numpy(np.float32),
                "teacher": valid["teacher"].to_numpy(np.float32),
                "correction": correction.astype(np.float32),
            }
        )
    )
    del train, tune, valid, corrections
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
if oof["id"].duplicated().any() or oof["well"].nunique() < MIN_WELLS:
    raise RuntimeError("Invalid OOF coverage")

fold_scans = []
fold_rows = []
nested_prediction = np.empty(len(oof), dtype=np.float64)
for held_fold in FOLDS:
    selection = oof.loc[oof["fold"].ne(held_fold)]
    held_mask = oof["fold"].eq(held_fold).to_numpy()
    dose, scan = _select_dose(selection)
    scan.insert(0, "held_fold", int(held_fold))
    fold_scans.append(scan)
    nested_prediction[held_mask] = (
        oof.loc[held_mask, "teacher"].to_numpy(np.float64)
        + dose * oof.loc[held_mask, "correction"].to_numpy(np.float64)
    )
    held = oof.loc[held_mask]
    base_rmse = _metrics(held, held["teacher"])["rmse"]
    candidate_rmse = _metrics(held, nested_prediction[held_mask])["rmse"]
    corr = _correlations(
        held["truth"], held["teacher"], nested_prediction[held_mask], held["md_since"]
    )
    fold_rows.append(
        {
            "fold": int(held_fold),
            "selected_dose": float(dose),
            "teacher_rmse": base_rmse,
            "candidate_rmse": candidate_rmse,
            "gain": base_rmse - candidate_rmse,
            **corr,
        }
    )

oof["teacher_correlated_gbdt"] = nested_prediction.astype(np.float32)
oof["matched_sign_flip"] = (
    oof["teacher"].to_numpy(np.float64)
    + oof["fold"].map(
        {row["fold"]: row["selected_dose"] for row in fold_rows}
    ).to_numpy(np.float64)
    * _matched_sign_flip(oof, oof["correction"])
).astype(np.float32)

teacher_metrics = _metrics(oof, oof["teacher"])
candidate_metrics = _metrics(oof, oof["teacher_correlated_gbdt"])
control_metrics = _metrics(oof, oof["matched_sign_flip"])
correlations = _correlations(
    oof["truth"],
    oof["teacher"],
    oof["teacher_correlated_gbdt"],
    oof["md_since"],
)
fold_report = pd.DataFrame(fold_rows)
scan_report = pd.concat(fold_scans, ignore_index=True)
model_report = pd.DataFrame(model_rows)
gain = teacher_metrics["rmse"] - candidate_metrics["rmse"]
control_loss = control_metrics["rmse"] - candidate_metrics["rmse"]

checks = {
    "gain_ge_0p05": gain >= PASS_GAIN_FT,
    "minimum_fold_gain_ge_minus0p05": float(fold_report["gain"].min())
    >= PASS_MIN_FOLD_GAIN_FT,
    "positive_folds_ge_4": int((fold_report["gain"] > 0.0).sum())
    >= PASS_POSITIVE_FOLDS,
    "raw_error_rho_ge_0p90": correlations["raw_signed_error_rho"] >= MIN_RAW_RHO,
    "detrended_error_rho_ge_0p90": correlations["distance_detrended_error_rho"]
    >= MIN_DETRENDED_RHO,
    "median_selected_dose_gt_zero": float(fold_report["selected_dose"].median()) > 0.0,
    "matched_control_worse_by_ge_0p02": control_loss >= PASS_CONTROL_LOSS_FT,
}
strict_pass = bool(all(checks.values()))

oof.to_parquet(OUT_DIR / "teacher_correlated_gbdt_oof.parquet", index=False, compression="zstd")
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
scan_report.to_csv(OUT_DIR / "dose_scan.csv", index=False)
model_report.to_csv(OUT_DIR / "model_report.csv", index=False)
pd.Series(model_features, name="feature").to_csv(OUT_DIR / "model_features.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_teacher_anchored_high_correlation_gbdt_gateway",
    "role": "mainline correction/replacement; not a decorrelated fusion endpoint",
    "teacher": {
        "path": str(teacher_path),
        "column": teacher_column,
        "is_exact_6p781_oof": teacher_is_exact_mainline,
    },
    "protocol": {
        "outer_folds": FOLDS,
        "inner_early_stopping_fold": "cyclic next fold, excluded from fit",
        "training_rows": "up to 1024 cached rows per fit well",
        "validation_rows": "all official hidden rows",
        "dose_selection": "other four OOF folds only",
        "sample_unit": "complete physical well",
    },
    "features": {
        "base": len(base_features),
        "teacher_channels": len(TEACHER_FEATURES),
        "total": len(model_features),
    },
    "teacher_metrics": teacher_metrics,
    "candidate_metrics": candidate_metrics,
    "matched_sign_flip_metrics": control_metrics,
    "gain": gain,
    "matched_control_loss": control_loss,
    "correlations": correlations,
    "folds": fold_report.to_dict(orient="records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS proxy mechanism -> rerun with exact 6.781 fold-clean OOF before deployment"
        if strict_pass and not teacher_is_exact_mainline
        else "PASS exact mainline -> full refit and Kaggle packaging"
        if strict_pass
        else "FAIL -> high correlation alone is not useful; do not package this correction"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nTEACHER-CORRELATED GBDT SUMMARY")
print(
    pd.DataFrame(
        [
            {"arm": "teacher", **teacher_metrics},
            {"arm": "teacher_correlated_gbdt", **candidate_metrics},
            {"arm": "matched_sign_flip", **control_metrics},
        ]
    ).to_string(index=False),
    flush=True,
)
print("\nFOLD REPORT")
print(fold_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
