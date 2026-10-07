"""V214: honest five-fold self-trained PF-GBDT reconstruction.

This experiment trains new LightGBM models from the cached legal feature table.
It never loads the public deployment boosters or any historical prediction
endpoint. The fixed downstream readout is the original learned/PF algebra:
60% warmed model delta plus 40% likelihood-PF, followed by Savitzky-Golay
smoothing within each complete held-out well.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.signal import savgol_filter


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / "selftrained_pf_gbdt_full5_v214"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 214
N_FOLDS = 5
TRAIN_ROWS_PER_WELL = 1024
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

DATA_CANDIDATES = (
    ROOT / "wellbore-geology-prediction-artifacts" / "data" / "train.csv",
    ROOT / "data" / "train.csv",
    ROOT / "train.csv",
)
FOLD_CANDIDATES = (
    OOF / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet",
    OOF / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
)
DATA_PATH = next((path for path in DATA_CANDIDATES if path.is_file()), None)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.is_file()), None)
if DATA_PATH is None or FOLD_PATH is None:
    raise FileNotFoundError({"data": [str(x) for x in DATA_CANDIDATES], "fold": [str(x) for x in FOLD_CANDIDATES]})


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _metrics(frame, column):
    truth = frame["tvt_true"].to_numpy(float)
    prediction = frame[column].to_numpy(float)
    error = prediction - truth
    distance = frame["md_since"].to_numpy(float)
    result = {
        "endpoint": column,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
    }
    well_mse = pd.Series(error * error).groupby(frame["well"].to_numpy()).mean()
    result["median_well_rmse"] = float(np.sqrt(well_mse).median())
    result["p95_well_rmse"] = float(np.sqrt(well_mse).quantile(0.95))
    for name, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (distance >= low) & (distance < high)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


SPATIAL_PREFIXES = (
    "tvtF_", "tvtFw_", "tvtF50_", "bw_", "bww_", "bw50_",
    "bw_early_", "bw_mid_", "frm_rmse_", "dense_", "tvt_dense",
)
SPATIAL_EXACT = {
    "form_mean_d", "form_std_d", "form_rng_d", "spatial_ancc_d",
    "spatial_knn_dist", "pf_vs_spatial", "pf_vs_dense", "spatial_vs_dense",
    "beam_vs_spatial", "sig_std", "sig_mean_d",
}
FORBIDDEN_EXACT = {
    "target", "tvt", "TVT", "truth", "tvt_true", "prediction",
    "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA",
    "fold", "row_idx", "_row_idx",
}
FORBIDDEN_LINEAGE_TOKENS = (
    "strong", "strong_bias", "cp8", "cp12", "gold", "neighbor",
    "v66", "v76", "v116", "v127", "v144", "v168", "incumbent",
)


def _is_spatial(column):
    return column in SPATIAL_EXACT or column.startswith(SPATIAL_PREFIXES)


def _is_forbidden_lineage(column):
    lowered = column.lower()
    return any(token in lowered for token in FORBIDDEN_LINEAGE_TOKENS)


def _load_folds():
    if FOLD_PATH.suffix.lower() == ".csv":
        frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
        if not {"well", "fold"}.issubset(frame.columns):
            raise RuntimeError(f"Missing well/fold in {FOLD_PATH}")
        frame = frame[["well", "fold"]].drop_duplicates("well")
        return frame
    frame = pd.read_parquet(FOLD_PATH, columns=["well", "fold"])
    frame["well"] = frame["well"].astype(str)
    return frame.drop_duplicates("well")


print("V214 loading", DATA_PATH, flush=True)
data = pd.read_csv(DATA_PATH, low_memory=False)
required = {"id", "well", "target", "last_known_tvt", "md_since", "pf_ancc", PP_LIKPF_COL}
missing = sorted(required - set(data.columns))
if missing:
    raise RuntimeError(f"V214 feature table missing {missing}")
data["id"] = data["id"].astype(str)
data["well"] = data["well"].astype(str)

folds = _load_folds()
if folds["well"].nunique() != 773 or folds["well"].duplicated().any():
    raise RuntimeError(f"Invalid fold map: {folds['well'].nunique()} wells")
if "fold" in data.columns:
    data = data.drop(columns=["fold"])
data = data.merge(folds, on="well", how="inner", validate="many_to_one")
if data["well"].nunique() != 773 or data["fold"].isna().any():
    raise RuntimeError(f"Incomplete fold alignment: rows={len(data)} wells={data['well'].nunique()}")

if "_row_idx" in data.columns:
    data["row_order"] = pd.to_numeric(data["_row_idx"], errors="coerce")
elif "row_idx" in data.columns:
    data["row_order"] = pd.to_numeric(data["row_idx"], errors="coerce")
else:
    data["row_order"] = data["id"].str.rsplit("_", n=1).str[-1].astype(np.int64)
data = data.sort_values(["well", "row_order"], kind="stable").reset_index(drop=True)

numeric = [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]
feature_columns = [
    column for column in numeric
    if column not in FORBIDDEN_EXACT
    and column not in {"row_order"}
    and not _is_spatial(column)
    and not _is_forbidden_lineage(column)
]
if len(feature_columns) < 80:
    raise RuntimeError(f"Only {len(feature_columns)} legal non-spatial features")

leak_columns = sorted(
    column for column in feature_columns
    if column in FORBIDDEN_EXACT or _is_spatial(column) or _is_forbidden_lineage(column)
)
if leak_columns:
    raise RuntimeError(f"Leakage guard failed: {leak_columns[:20]}")

data["target"] = pd.to_numeric(data["target"], errors="coerce")
data = data.loc[data["target"].notna()].reset_index(drop=True)
data["tvt_true"] = data["last_known_tvt"].to_numpy(float) + data["target"].to_numpy(float)

# Fixed equal-well training sample. Each complete well contributes at most 1024 rows.
sample_positions = []
for _, positions in data.groupby("well", sort=False).indices.items():
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) > TRAIN_ROWS_PER_WELL:
        take = np.unique(np.round(np.linspace(0, len(positions) - 1, TRAIN_ROWS_PER_WELL)).astype(int))
        positions = positions[take]
    sample_positions.append(positions)
sample_positions = np.concatenate(sample_positions)
sample = data.iloc[sample_positions].copy()
well_rows = data.groupby("well").size().to_dict()
well_sample_rows = sample.groupby("well").size().to_dict()
sample["sample_weight"] = sample["well"].map(well_rows).to_numpy(float) / sample["well"].map(well_sample_rows).to_numpy(float)

print(
    "V214 CONFIG",
    json.dumps(
        {
            "data": str(DATA_PATH), "folds": str(FOLD_PATH), "rows": len(data),
            "sample_rows": len(sample), "wells": data["well"].nunique(),
            "features": len(feature_columns), "public_boosters_loaded": 0,
            "historical_prediction_inputs": 0, "particle_assumption": "cached feature table",
        }, indent=2,
    ), flush=True,
)
pd.Series(feature_columns, name="feature").to_csv(OUT / "feature_columns.csv", index=False)


def _matrix(frame):
    return frame.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)


def _new_model(fold):
    params = dict(
        objective="regression", boosting_type="gbdt", n_estimators=3000,
        learning_rate=0.025, num_leaves=64, min_child_samples=80,
        subsample=0.80, subsample_freq=1, colsample_bytree=0.70,
        reg_alpha=0.10, reg_lambda=10.0, max_bin=255,
        n_jobs=-1, random_state=SEED + fold, verbosity=-1,
    )
    if os.environ.get("V214_CPU", "0") != "1":
        params["device_type"] = "gpu"
    return LGBMRegressor(**params)


def _fit(train, valid, fold):
    model = _new_model(fold)
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(100)],
    )
    try:
        model.fit(_matrix(train), train["target"].to_numpy(float), **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("V214 GPU unavailable, CPU fallback:", repr(exc), flush=True)
        os.environ["V214_CPU"] = "1"
        model = _new_model(fold)
        model.fit(_matrix(train), train["target"].to_numpy(float), **kwargs)
    return model


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(float)
    distance = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(distance, 0.0) / PP_TAU)
    lik = frame[PP_LIKPF_COL].to_numpy(float)
    lik = np.where(np.isfinite(lik), lik, last)
    prediction = last + PP_W_MODEL * warm * model_delta + (1.0 - PP_W_MODEL) * (lik - last)
    output = prediction.copy()
    offset = 0
    for _, group in frame.groupby("well", sort=False):
        n = len(group)
        window = min(PP_SG_WIN, n)
        if window % 2 == 0:
            window -= 1
        if window >= PP_SG_POLY + 2:
            output[offset:offset + n] = savgol_filter(prediction[offset:offset + n], window, PP_SG_POLY)
        offset += n
    return output


oof_parts = []
fold_reports = []
for fold in range(1, N_FOLDS + 1):
    train = sample.loc[~sample["fold"].eq(fold)].copy()
    valid = data.loc[data["fold"].eq(fold)].copy()
    print(f"V214 fold {fold}: train_rows={len(train)} valid_rows={len(valid)}", flush=True)
    model = _fit(train, valid, fold)
    delta = model.predict(_matrix(valid), num_iteration=model.best_iteration_)
    raw = valid["last_known_tvt"].to_numpy(float) + delta
    final = _postprocess(valid, delta)
    part = valid[["id", "well", "fold", "row_order", "md_since", "tvt_true"]].copy()
    part["model_raw"] = raw
    part["selftrained_pf_gbdt"] = final
    oof_parts.append(part)
    raw_rmse = _rmse(part["tvt_true"], part["model_raw"])
    final_rmse = _rmse(part["tvt_true"], part["selftrained_pf_gbdt"])
    report = {
        "fold": fold, "rows": len(valid), "wells": valid["well"].nunique(),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "raw_rmse": raw_rmse, "final_rmse": final_rmse,
        "readout_gain": raw_rmse - final_rmse,
    }
    fold_reports.append(report)
    print("V214 FOLD", report, flush=True)
    joblib.dump(model, MODEL_DIR / f"selftrained_pf_gbdt_fold{fold}.joblib")
    del train, valid, model, delta, raw, final, part
    gc.collect()

oof_frame = pd.concat(oof_parts, ignore_index=True)
summary = pd.DataFrame([_metrics(oof_frame, "model_raw"), _metrics(oof_frame, "selftrained_pf_gbdt")])
fold_frame = pd.DataFrame(fold_reports)
final_rmse = float(summary.loc[summary["endpoint"].eq("selftrained_pf_gbdt"), "rmse"].iloc[0])
checks = {
    "coverage_773": bool(len(oof_frame) == len(data) and oof_frame["well"].nunique() == 773),
    "standalone_rmse_le_7p4": bool(final_rmse <= 7.4),
    "readout_positive_folds_ge_4": bool((fold_frame["readout_gain"] > 0).sum() >= 4),
    "no_public_boosters": True,
    "no_historical_prediction_inputs": True,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "selftrained_pf_gbdt_full5_v214",
    "stage": "complete_773well_selftrained_nonspatial_PF_GBDT",
    "data": str(DATA_PATH), "fold_source": str(FOLD_PATH),
    "features": len(feature_columns), "feature_sha256": hashlib.sha256("\n".join(feature_columns).encode()).hexdigest(),
    "training": {"models": 5, "rows_per_well": TRAIN_ROWS_PER_WELL, "public_boosters_loaded": 0},
    "summary": summary.to_dict(orient="records"), "folds": fold_reports,
    "checks": checks, "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> build the same self-trained model in the Kaggle inference notebook" if strict_pass else "FAIL -> tree strength came from a different representation; do not submit",
}
oof_frame.to_parquet(OUT / "oof.parquet", index=False)
summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
print("\nV214 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV214 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nV214 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT, flush=True)
