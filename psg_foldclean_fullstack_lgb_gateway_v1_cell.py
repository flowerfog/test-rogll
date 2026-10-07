"""Fold-clean full-stack LightGBM gateway using exact-fold PSG OOF features.

PSG and cached spatial features use the identical historical fold-clean folds.
Every validation well is excluded from both the PSG model/structure cloud and
the spatial feature reference pool.

Arms:
  baseline: cached deployable full feature stack, including spatial features;
  psg_features: the identical features plus exact-fold PSG disagreement features;
  psg_neutral: the trained psg_features model with all new channels zeroed.
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

EXPERIMENT = "psg_foldclean_fullstack_lgb_gateway_v1"
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = CACHE_ROOT / "features"
BASE_FEATURE_PATH = CACHE_ROOT / "all_features.csv"
FOLD_META_PATH = CACHE_ROOT / "well_metadata.csv"
PSG_PATH = OOF_ROOT / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260722
N_FOLDS = 5
MAX_TREES = 2400
PATIENCE = 180

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

MIN_WELLS = 760
MIN_MERGE_COVERAGE = 0.999
PASS_OVERALL_GAIN = 0.10
PASS_MIN_FOLD_GAIN = -0.05
PASS_POSITIVE_FOLDS = 4
PASS_NEUTRAL_DEGRADATION = 0.03

PSG_FEATURES = [
    "psgx_psg_delta_last",
    "psgx_psg_minus_pf",
    "psgx_abs_psg_minus_pf",
    "psgx_psg_minus_lik3",
    "psgx_abs_psg_minus_lik3",
    "psgx_psg_minus_lik5",
    "psgx_abs_psg_minus_lik5",
    "psgx_psg_minus_lik8",
    "psgx_abs_psg_minus_lik8",
    "psgx_psg_minus_lik12",
    "psgx_abs_psg_minus_lik12",
    "psgx_psg_delta_x_sqrt_progress",
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


def _rmse(truth, prediction):
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(error**2)))


def _metrics(truth, prediction, wells):
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    codes, names = pd.factorize(np.asarray(wells).astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error**2, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    return {
        "rows": int(len(error)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }


def _matrix(frame, columns):
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _new_model(fold, arm):
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=MAX_TREES,
        learning_rate=0.025,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=10.0,
        max_bin=255,
        n_jobs=-1,
        random_state=SEED + int(fold) + (0 if arm == "baseline" else 100),
        verbosity=-1,
    )


def _fit_model(train, valid, columns, fold, arm):
    model = _new_model(fold, arm)
    x_train = _matrix(train, columns)
    x_valid = _matrix(valid, columns)
    y_train = train["target"].to_numpy(np.float32)
    y_valid = valid["target"].to_numpy(np.float32)
    sample_weight = train.get(
        "sample_weight", pd.Series(np.ones(len(train), np.float32))
    ).to_numpy(np.float32)
    fit_kwargs = dict(
        sample_weight=sample_weight,
        eval_set=[(x_valid, y_valid)],
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
    del x_train, x_valid
    gc.collect()
    return model


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    md_since = frame["md_since"].to_numpy(np.float64)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(np.float64) - last
    if PP_LIKPF_COL in frame:
        lik_delta = frame[PP_LIKPF_COL].to_numpy(np.float64) - last
    else:
        lik_delta = pf_delta
    learned = PP_ALPHA * warm * (
        np.asarray(model_delta, np.float64) * (1.0 - PP_W_PF_ANCC)
        + pf_delta * PP_W_PF_ANCC
    )
    prediction = last + PP_W_MODEL * learned + (1.0 - PP_W_MODEL) * lik_delta

    output = prediction.copy()
    offset = 0
    for _, group in frame.groupby("well", sort=False):
        n_rows = len(group)
        values = prediction[offset : offset + n_rows]
        window = min(PP_SG_WIN, n_rows)
        if window % 2 == 0:
            window -= 1
        if window >= PP_SG_POLY + 2:
            output[offset : offset + n_rows] = savgol_filter(
                values, window, PP_SG_POLY
            )
        offset += n_rows
    return output


def _feature_paths(well):
    return FEATURE_DIR / f"{well}__sample.pkl", FEATURE_DIR / f"{well}__full.pkl"


def _attach_psg(frame, well):
    table = psg_groups[str(well)]
    rows = frame["_row_idx"].to_numpy(np.int32)
    aligned = table.reindex(rows)
    coverage = float(aligned["psg_tvt"].notna().mean())
    if coverage < MIN_MERGE_COVERAGE:
        raise RuntimeError(f"{well}: PSG feature coverage={coverage:.6%}")

    frame_truth = (
        frame["target"].to_numpy(np.float64)
        + frame["last_known_tvt"].to_numpy(np.float64)
    )
    psg_truth = aligned["true_tvt"].to_numpy(np.float64)
    truth_gap = np.abs(frame_truth - psg_truth)
    if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
        raise RuntimeError(
            f"{well}: PSG/cache truth mismatch max={float(np.nanmax(truth_gap)):.6f}"
        )

    last = frame["last_known_tvt"].to_numpy(np.float32)
    pf = frame["pf_ancc"].to_numpy(np.float32)
    progress = np.sqrt(np.maximum(frame["md_since"].to_numpy(np.float32), 0.0) + 1.0)
    psg = aligned["psg_tvt"].to_numpy(np.float32)
    lik3 = frame["likpf_scale_3"].to_numpy(np.float32)
    lik5 = frame["likpf_scale_5"].to_numpy(np.float32)
    lik8 = frame["likpf_scale_8"].to_numpy(np.float32)
    lik12 = frame["likpf_scale_12"].to_numpy(np.float32)

    frame["psgx_psg_delta_last"] = psg - last
    frame["psgx_psg_minus_pf"] = psg - pf
    frame["psgx_abs_psg_minus_pf"] = np.abs(psg - pf)
    for suffix, reference in (
        ("lik3", lik3),
        ("lik5", lik5),
        ("lik8", lik8),
        ("lik12", lik12),
    ):
        difference = psg - reference
        frame[f"psgx_psg_minus_{suffix}"] = difference
        frame[f"psgx_abs_psg_minus_{suffix}"] = np.abs(difference)
    frame["psgx_psg_delta_x_sqrt_progress"] = (psg - last) * progress / 100.0
    return frame


def _load_frames(wells, kind):
    frames = []
    position = 0 if kind == "sample" else 1
    for well in wells:
        path = _feature_paths(str(well))[position]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path)
        frame["well"] = str(well)
        frames.append(_attach_psg(frame, str(well)))
    return pd.concat(frames, ignore_index=True, sort=False)


started = time.time()
for required in (FEATURE_DIR, BASE_FEATURE_PATH, FOLD_META_PATH, PSG_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

psg = pd.read_parquet(
    PSG_PATH,
    columns=["well", "row_idx", "fold", "psg_tvt", "true_tvt"],
)
psg["well"] = psg["well"].astype(str).str.lower()
psg["row_idx"] = psg["row_idx"].astype(np.int32)
psg["fold"] = psg["fold"].astype(np.int8)
if psg.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("PSG feature table has duplicate keys")
fold_values = sorted(psg["fold"].unique().tolist())
if fold_values != [1, 2, 3, 4, 5]:
    raise RuntimeError(f"Expected PSG folds 1..5, got {fold_values}")

fold_metadata = pd.read_csv(FOLD_META_PATH, usecols=["well", "fold"])
fold_metadata["well"] = fold_metadata["well"].astype(str).str.lower()
fold_metadata["fold"] = fold_metadata["fold"].astype(int)
metadata_fold = fold_metadata.set_index("well")["fold"]
psg_well_fold = psg.groupby("well")["fold"].first().astype(int)
common_fold_wells = metadata_fold.index.intersection(psg_well_fold.index)
fold_mismatch = common_fold_wells[
    metadata_fold.loc[common_fold_wells].to_numpy()
    != psg_well_fold.loc[common_fold_wells].to_numpy()
]
if len(fold_mismatch):
    raise RuntimeError(
        "PSG/spatial fold mismatch for wells: " + repr(fold_mismatch[:20].tolist())
    )

psg_groups = {
    well: group.set_index("row_idx").sort_index()
    for well, group in psg.groupby("well", sort=False)
}
fold_by_well = psg.groupby("well")["fold"].first().astype(int)
eligible_wells = sorted(
    well
    for well in psg_groups
    if all(path.exists() for path in _feature_paths(well))
)
missing_cache = sorted(set(psg_groups).difference(eligible_wells))
if len(eligible_wells) < MIN_WELLS:
    raise RuntimeError(
        f"Only {len(eligible_wells)} PSG wells have feature cache; missing={missing_cache[:20]}"
    )

base_features = pd.read_csv(BASE_FEATURE_PATH)["feature"].astype(str).tolist()
schema = pd.read_pickle(_feature_paths(eligible_wells[0])[0])
base_features = [
    column
    for column in base_features
    if column in schema
    and pd.api.types.is_numeric_dtype(schema[column])
    and column not in {"target", "TVT", "TVT_input", "_row_idx", "fold", "sample_weight"}
]
if len(base_features) < 20:
    raise RuntimeError(f"Too few baseline features: {base_features}")
all_features = base_features + PSG_FEATURES

print(
    "PSG FEATURE GATEWAY CONFIG",
    {
        "wells": len(eligible_wells),
        "missing_cache_wells": len(missing_cache),
        "base_features": len(base_features),
        "psg_features": len(PSG_FEATURES),
        "fold_wells": {
            fold: int(sum(fold_by_well[well] == fold for well in eligible_wells))
            for fold in fold_values
        },
    },
    flush=True,
)

oof_parts = []
fold_rows = []
model_rows = []
for fold in fold_values:
    train_wells = [well for well in eligible_wells if int(fold_by_well[well]) != fold]
    valid_wells = [well for well in eligible_wells if int(fold_by_well[well]) == fold]
    print(
        f"\nPSG feature fold {fold}/5 train={len(train_wells)} valid={len(valid_wells)}",
        flush=True,
    )
    train = _load_frames(train_wells, "sample")
    valid = _load_frames(valid_wells, "full")
    truth = valid["target"].to_numpy(np.float64) + valid["last_known_tvt"].to_numpy(np.float64)

    baseline_model = _fit_model(train, valid, base_features, fold, "baseline")
    baseline_delta = baseline_model.predict(
        _matrix(valid, base_features), num_iteration=baseline_model.best_iteration_
    )
    baseline_prediction = _postprocess(valid, baseline_delta)

    psg_model = _fit_model(train, valid, all_features, fold, "psg_features")
    psg_delta = psg_model.predict(
        _matrix(valid, all_features), num_iteration=psg_model.best_iteration_
    )
    psg_prediction = _postprocess(valid, psg_delta)

    neutral = valid.copy()
    for column in PSG_FEATURES:
        neutral[column] = 0.0
    neutral_delta = psg_model.predict(
        _matrix(neutral, all_features), num_iteration=psg_model.best_iteration_
    )
    neutral_prediction = _postprocess(neutral, neutral_delta)

    baseline_rmse = _rmse(truth, baseline_prediction)
    psg_rmse = _rmse(truth, psg_prediction)
    neutral_rmse = _rmse(truth, neutral_prediction)
    fold_rows.append(
        {
            "fold": int(fold),
            "rows": int(len(valid)),
            "wells": int(len(valid_wells)),
            "baseline_rmse": baseline_rmse,
            "psg_feature_rmse": psg_rmse,
            "neutral_rmse": neutral_rmse,
            "gain": baseline_rmse - psg_rmse,
            "neutral_degradation": neutral_rmse - psg_rmse,
        }
    )
    model_rows.extend(
        [
            {
                "fold": int(fold),
                "arm": "baseline",
                "features": len(base_features),
                "best_iteration": int(baseline_model.best_iteration_),
            },
            {
                "fold": int(fold),
                "arm": "psg_features",
                "features": len(all_features),
                "best_iteration": int(psg_model.best_iteration_),
            },
        ]
    )
    print("PSG FEATURE FOLD", fold_rows[-1], flush=True)

    oof_parts.append(
        pd.DataFrame(
            {
                "id": valid["id"].astype(str).to_numpy(),
                "well": valid["well"].astype(str).to_numpy(),
                "row_idx": valid["_row_idx"].to_numpy(np.int32),
                "fold": int(fold),
                "tvt_true": truth,
                "baseline": baseline_prediction,
                "psg_features": psg_prediction,
                "psg_neutral": neutral_prediction,
            }
        )
    )
    joblib.dump(baseline_model, MODEL_DIR / f"baseline_fold{fold}.joblib")
    joblib.dump(psg_model, MODEL_DIR / f"psg_features_fold{fold}.joblib")
    del train, valid, neutral, baseline_model, psg_model
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
fold_report = pd.DataFrame(fold_rows)
model_report = pd.DataFrame(model_rows)
if oof.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("Model OOF has duplicate rows")

summary_rows = []
for arm in ("baseline", "psg_features", "psg_neutral"):
    summary_rows.append(
        {"arm": arm, **_metrics(oof["tvt_true"], oof[arm], oof["well"])}
    )
summary = pd.DataFrame(summary_rows)
baseline_rmse = float(summary.loc[summary["arm"].eq("baseline"), "rmse"].iloc[0])
psg_rmse = float(summary.loc[summary["arm"].eq("psg_features"), "rmse"].iloc[0])
neutral_rmse = float(summary.loc[summary["arm"].eq("psg_neutral"), "rmse"].iloc[0])
overall_gain = baseline_rmse - psg_rmse
neutral_degradation = neutral_rmse - psg_rmse

checks = {
    "overall_gain_ge_0p10": overall_gain >= PASS_OVERALL_GAIN,
    "minimum_fold_gain_ge_minus0p05": float(fold_report["gain"].min()) >= PASS_MIN_FOLD_GAIN,
    "positive_folds_ge_4": int((fold_report["gain"] > 0.0).sum()) >= PASS_POSITIVE_FOLDS,
    "neutral_degradation_ge_0p03": neutral_degradation >= PASS_NEUTRAL_DEGRADATION,
}
strict_pass = bool(all(checks.values()))

oof.to_parquet(OUT_DIR / "psg_feature_lgb_oof.parquet", index=False)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
model_report.to_csv(OUT_DIR / "model_report.csv", index=False)
pd.Series(base_features, name="feature").to_csv(OUT_DIR / "base_features.csv", index=False)
pd.Series(PSG_FEATURES, name="feature").to_csv(OUT_DIR / "psg_features.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_psg_oof_feature_lightgbm_ab",
    "protocol": {
        "outer_folds": "exact PSG OOF folds",
        "training_rows": "up to 1024 cached rows per well",
        "validation_rows": "all official hidden rows",
        "spatial_features_used": True,
        "fold_source": str(FOLD_META_PATH),
        "psg_fold_source": str(PSG_PATH),
        "psg_target_well_excluded": True,
    },
    "wells": len(eligible_wells),
    "missing_cache_wells": missing_cache,
    "base_feature_count": len(base_features),
    "psg_feature_count": len(PSG_FEATURES),
    "baseline_rmse": baseline_rmse,
    "psg_feature_rmse": psg_rmse,
    "neutral_rmse": neutral_rmse,
    "overall_gain": overall_gain,
    "neutral_degradation": neutral_degradation,
    "fold_gains": fold_report["gain"].tolist(),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> refit the full stack and package five PSG checkpoints for Kaggle inference"
        if strict_pass
        else "FAIL -> PSG helps as an endpoint but not as a downstream tabular feature; close this branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nPSG FEATURE MODEL SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nPSG FEATURE FOLDS")
print(fold_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
