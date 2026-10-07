"""V216: fold-clean spatial likelihood-PF residual GBDT.

Consumes the already completed 773-well fold-clean feature cache. This is an
exact target/readout transplant from V215: the only added information is the
cross-fitted FI/DI spatial feature family built from the other four folds.
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
SOURCE = OOF / "foldclean_full_fleongg_lgb_v0"
FEATURES = SOURCE / "features"
OUT = OOF / "foldclean_spatial_residual_gbdt_v216"
MODELS = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 216
N_FOLDS = 5
LIKPF_COLUMN = "likpf_scale_5"
WARM_TAU = 85.0
WARM_FLOOR = 0.15
SG_WINDOW = 61
SG_POLY = 3

required = (SOURCE / "well_metadata.csv", SOURCE / "all_features.csv")
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)
if len(list(FEATURES.glob("*__sample.pkl"))) != 773 or len(list(FEATURES.glob("*__full.pkl"))) != 773:
    raise RuntimeError("V216 requires all 773 fold-clean sample and full feature files")

meta = pd.read_csv(SOURCE / "well_metadata.csv", dtype={"well": str})
if meta["well"].nunique() != 773 or not {"well", "fold"}.issubset(meta.columns):
    raise RuntimeError("Invalid fold-clean metadata")

feature_columns = pd.read_csv(SOURCE / "all_features.csv")["feature"].astype(str).tolist()
FORBIDDEN_EXACT = {
    "target", "residual_target", "TVT", "tvt", "truth", "tvt_true",
    "well", "id", "fold", "_row_idx", "sample_weight",
    "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA",
}
FORBIDDEN_LINEAGE = (
    "strong", "cp8", "cp12", "gold", "neighbor", "v66", "v76",
    "v116", "v127", "v144", "v168", "incumbent",
)
feature_columns = [
    column for column in feature_columns
    if column not in FORBIDDEN_EXACT
    and not any(token in column.lower() for token in FORBIDDEN_LINEAGE)
]
if len(feature_columns) < 160:
    raise RuntimeError(f"Expected the spatial feature family; found only {len(feature_columns)} features")


def _paths(well):
    return FEATURES / f"{well}__sample.pkl", FEATURES / f"{well}__full.pkl"


def _load(wells, full):
    index = 1 if full else 0
    frames = [pd.read_pickle(_paths(str(well))[index]) for well in wells]
    return pd.concat(frames, ignore_index=True, sort=False)


def _prepare(frame, training):
    frame = frame.copy()
    last = frame["last_known_tvt"].to_numpy(float)
    lik = frame[LIKPF_COLUMN].to_numpy(float)
    lik = np.where(np.isfinite(lik), lik, last)
    distance = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(distance, 0.0) / WARM_TAU)
    divisor = np.maximum(warm, WARM_FLOOR)
    frame["residual_target"] = np.clip(
        (frame["target"].to_numpy(float) - (lik - last)) / divisor,
        -80.0,
        80.0,
    ).astype(np.float32)
    if training and "sample_weight" not in frame:
        counts = frame.groupby("well").size()
        frame["sample_weight"] = frame["well"].map((1.0 / counts).to_dict()).to_numpy(float).copy()
        frame["sample_weight"] /= float(frame["sample_weight"].mean())
    return frame


def _matrix(frame):
    return (
        frame.reindex(columns=feature_columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _new_model(fold):
    params = dict(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=3000,
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
        random_state=SEED + fold,
        verbosity=-1,
    )
    if os.environ.get("V216_CPU", "0") != "1":
        params["device_type"] = "gpu"
    return LGBMRegressor(**params)


def _fit(train, valid, fold):
    model = _new_model(fold)
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["residual_target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(100)],
    )
    try:
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    except Exception as exc:
        if "gpu" not in str(exc).lower() and "opencl" not in str(exc).lower():
            raise
        print("V216 GPU fallback", repr(exc), flush=True)
        os.environ["V216_CPU"] = "1"
        model = _new_model(fold)
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    return model


def _smooth(frame, prediction):
    output = prediction.copy()
    offset = 0
    for _, group in frame.groupby("well", sort=False):
        n = len(group)
        window = min(SG_WINDOW, n)
        if window % 2 == 0:
            window -= 1
        if window >= SG_POLY + 2:
            output[offset:offset + n] = savgol_filter(
                prediction[offset:offset + n], window, SG_POLY
            )
        offset += n
    return output


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


parts = []
fold_reports = []
for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[~meta["fold"].eq(fold), "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"].eq(fold), "well"].astype(str).tolist()
    train = _prepare(_load(train_wells, full=False), training=True)
    valid = _prepare(_load(valid_wells, full=True), training=False)
    valid = valid.sort_values(["well", "_row_idx"], kind="stable").reset_index(drop=True)
    print(f"V216 fold {fold}: train={len(train)} valid={len(valid)} features={len(feature_columns)}", flush=True)
    model = _fit(train, valid, fold)
    residual = model.predict(_matrix(valid), num_iteration=model.best_iteration_)
    last = valid["last_known_tvt"].to_numpy(float)
    lik = valid[LIKPF_COLUMN].to_numpy(float)
    lik = np.where(np.isfinite(lik), lik, last)
    warm = 1.0 - np.exp(-np.maximum(valid["md_since"].to_numpy(float), 0.0) / WARM_TAU)
    candidate = _smooth(valid, lik + warm * residual)
    truth = last + valid["target"].to_numpy(float)
    report = {
        "fold": fold,
        "rows": len(valid),
        "wells": len(valid_wells),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "likpf_rmse": _rmse(truth, lik),
        "candidate_rmse": _rmse(truth, candidate),
    }
    report["gain"] = report["likpf_rmse"] - report["candidate_rmse"]
    fold_reports.append(report)
    print("V216 FOLD", report, flush=True)
    parts.append(pd.DataFrame({
        "id": valid["id"].astype(str),
        "well": valid["well"].astype(str),
        "fold": fold,
        "md_since": valid["md_since"].to_numpy(float),
        "tvt_true": truth,
        "likpf": lik,
        "candidate": candidate,
    }))
    joblib.dump(model, MODELS / f"fold{fold}.joblib")
    del train, valid, model, residual, candidate
    gc.collect()

oof = pd.concat(parts, ignore_index=True)


def _metrics(column):
    error = oof[column].to_numpy(float) - oof["tvt_true"].to_numpy(float)
    well_rmse = pd.Series(error * error).groupby(oof["well"].to_numpy()).mean().pow(0.5)
    far = oof["md_since"].to_numpy(float) >= 2000.0
    return {
        "endpoint": column,
        "rows": len(oof),
        "wells": oof["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


summary = pd.DataFrame([_metrics("likpf"), _metrics("candidate")])
fold_frame = pd.DataFrame(fold_reports)
candidate_rmse = float(summary.loc[summary["endpoint"].eq("candidate"), "rmse"].iloc[0])

# Read-only matched comparison with V215 when its complete OOF is available.
v215_path = OOF / "likpf_residual_gbdt_full5_v215" / "oof.parquet"
matched_v215 = None
if v215_path.is_file():
    prior = pd.read_parquet(v215_path, columns=["id", "candidate"]).rename(columns={"candidate": "v215"})
    joined = oof[["id", "tvt_true", "candidate"]].merge(prior, on="id", how="inner", validate="one_to_one")
    if len(joined) == len(oof):
        matched_v215 = {
            "v215_rmse": _rmse(joined["tvt_true"], joined["v215"]),
            "v216_rmse": _rmse(joined["tvt_true"], joined["candidate"]),
        }
        matched_v215["spatial_gain"] = matched_v215["v215_rmse"] - matched_v215["v216_rmse"]

checks = {
    "coverage_773": bool(len(oof) == 3_783_989 and oof["well"].nunique() == 773),
    "candidate_rmse_le_9p5": bool(candidate_rmse <= 9.5),
    "positive_folds_5of5": bool((fold_frame["gain"] > 0).all()),
    "spatial_gain_vs_v215_ge_0p50": bool(matched_v215 is not None and matched_v215["spatial_gain"] >= 0.50),
    "no_public_boosters": True,
}
verdict = {
    "experiment": "foldclean_spatial_residual_gbdt_v216",
    "stage": "complete_773well_random_GroupKFold_foldclean_spatial_residual_GBDT",
    "source_cache": str(SOURCE),
    "features": len(feature_columns),
    "feature_sha256": hashlib.sha256("\n".join(feature_columns).encode()).hexdigest(),
    "summary": summary.to_dict(orient="records"),
    "folds": fold_reports,
    "matched_v215": matched_v215,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> train a second tree architecture and assess a Kaggle bundle" if all(checks.values()) else "FAIL -> spatial residual GBDT remains too weak; do not submit",
}
oof.to_parquet(OUT / "oof.parquet", index=False)
summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nV216 SUMMARY\n", summary.to_string(index=False), flush=True)
print("\nV216 FOLDS\n", fold_frame.to_string(index=False), flush=True)
print("\nV216 FINAL VERDICT\n", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT, flush=True)
