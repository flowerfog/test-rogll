"""V217: clean spatial A/B on the frozen V215 residual-GBDT representation.

The V216 result mixed two changes: fold-clean spatial features and an older
feature cache whose likelihood-PF baseline was much weaker.  V217 keeps every
V215 row, target, likelihood-PF value, fold, and non-spatial feature frozen,
then joins only the cross-fitted spatial columns from the V216 source cache.
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
BASE = OOF / "likpf_residual_gbdt_full5_v215"
SPATIAL = OOF / "foldclean_full_fleongg_lgb_v0"
OUT = OOF / "likpf_residual_foldclean_spatial_v217"
MODELS = OUT / "models"
for path in (OUT, MODELS):
    path.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SEED = 217
LIKPF_COLUMN = "likpf_scale_5"
WARM_TAU = 85.0
WARM_FLOOR = 0.15
SG_WINDOW = 61
SG_POLY = 3


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


required = (
    BASE / "well_metadata.csv",
    BASE / "features.csv",
    BASE / "oof.parquet",
    SPATIAL / "well_metadata.csv",
    SPATIAL / "all_features.csv",
    SPATIAL / "no_spatial_features.csv",
)
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

base_meta = pd.read_csv(BASE / "well_metadata.csv", dtype={"well": str})
spatial_meta = pd.read_csv(SPATIAL / "well_metadata.csv", dtype={"well": str})
fold_check = base_meta[["well", "fold"]].merge(
    spatial_meta[["well", "fold"]], on="well", how="outer", suffixes=("_base", "_spatial"), validate="one_to_one"
)
if len(fold_check) != 773 or not fold_check["fold_base"].eq(fold_check["fold_spatial"]).all():
    mismatch = fold_check.loc[~fold_check["fold_base"].eq(fold_check["fold_spatial"])].head().to_dict("records")
    raise RuntimeError(f"Fold maps differ; spatial features are not fold-clean for V215: {mismatch}")

base_features = pd.read_csv(BASE / "features.csv")["feature"].astype(str).tolist()
all_spatial_source = pd.read_csv(SPATIAL / "all_features.csv")["feature"].astype(str).tolist()
nonspatial_source = set(pd.read_csv(SPATIAL / "no_spatial_features.csv")["feature"].astype(str))
spatial_features = [name for name in all_spatial_source if name not in nonspatial_source]
if len(base_features) < 80 or len(spatial_features) < 20:
    raise RuntimeError({"base_features": len(base_features), "spatial_features": len(spatial_features)})
if set(base_features) & set(spatial_features):
    raise RuntimeError(f"Unexpected feature overlap: {sorted(set(base_features) & set(spatial_features))[:10]}")
features = base_features + spatial_features


def _paths(root, well, full):
    suffix = "full" if full else "sample"
    return root / "features" / f"{well}__{suffix}.pkl"


def _load_one(well, full):
    base = pd.read_pickle(_paths(BASE, well, full))
    spatial = pd.read_pickle(_paths(SPATIAL, well, full))
    key = "id" if "id" in base and "id" in spatial else "_row_idx"
    if base[key].duplicated().any() or spatial[key].duplicated().any():
        raise RuntimeError(f"{well}: duplicate {key}")
    lookup = spatial.set_index(key)[spatial_features]
    aligned = lookup.reindex(base[key].to_numpy())
    if len(aligned) != len(base) or aligned.isna().all(axis=1).any():
        missing = int(aligned.isna().all(axis=1).sum())
        raise RuntimeError(f"{well}: incomplete spatial alignment ({missing}/{len(base)})")
    base[spatial_features] = aligned.to_numpy(np.float32)
    return base


def _load(wells, full):
    return pd.concat([_load_one(str(well), full) for well in wells], ignore_index=True, sort=False)


def _prepare(frame, training):
    frame = frame.copy()
    last = frame["last_known_tvt"].to_numpy(float)
    lik = frame[LIKPF_COLUMN].to_numpy(float)
    lik = np.where(np.isfinite(lik), lik, last)
    warm = 1.0 - np.exp(-np.maximum(frame["md_since"].to_numpy(float), 0.0) / WARM_TAU)
    divisor = np.maximum(warm, WARM_FLOOR)
    frame["residual_target"] = np.clip(
        (frame["target"].to_numpy(float) - (lik - last)) / divisor, -80.0, 80.0
    ).astype(np.float32)
    if training:
        counts = frame.groupby("well").size()
        weight = frame["well"].map((1.0 / counts).to_dict()).to_numpy(float).copy()
        frame["sample_weight"] = weight / float(weight.mean())
    return frame


def _matrix(frame):
    return (
        frame.reindex(columns=features)
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
    if os.environ.get("V217_CPU", "0") != "1":
        params["device_type"] = "gpu"
    return LGBMRegressor(**params)


def _fit(train, valid, fold):
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["residual_target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(100)],
    )
    model = _new_model(fold)
    try:
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("V217 GPU fallback", repr(exc), flush=True)
        os.environ["V217_CPU"] = "1"
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
            output[offset:offset + n] = savgol_filter(prediction[offset:offset + n], window, SG_POLY)
        offset += n
    return output


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


parts = []
fold_reports = []
for fold in range(1, N_FOLDS + 1):
    train_wells = base_meta.loc[~base_meta["fold"].eq(fold), "well"].astype(str).tolist()
    valid_wells = base_meta.loc[base_meta["fold"].eq(fold), "well"].astype(str).tolist()
    train = _prepare(_load(train_wells, full=False), training=True)
    valid = _prepare(_load(valid_wells, full=True), training=False)
    valid = valid.sort_values(["well", "_row_idx"], kind="stable").reset_index(drop=True)
    print(
        f"V217 fold {fold}: train={len(train)} valid={len(valid)} "
        f"base={len(base_features)} spatial={len(spatial_features)} total={len(features)}",
        flush=True,
    )
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
    print("V217 FOLD", report, flush=True)
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
if len(oof) != 3_783_989 or oof["well"].nunique() != 773 or oof["id"].duplicated().any():
    raise RuntimeError({"rows": len(oof), "wells": oof["well"].nunique(), "duplicates": int(oof["id"].duplicated().sum())})

base_oof = pd.read_parquet(BASE / "oof.parquet")[["id", "candidate"]].rename(columns={"candidate": "v215"})
matched = oof.merge(base_oof, on="id", how="left", validate="one_to_one")
if matched["v215"].isna().any():
    raise RuntimeError("V215 OOF alignment incomplete")

base_rmse = _rmse(matched["tvt_true"], matched["v215"])
candidate_rmse = _rmse(matched["tvt_true"], matched["candidate"])
fold_compare = []
for fold, group in matched.groupby("fold", sort=True):
    fold_compare.append({
        "fold": int(fold),
        "v215_rmse": _rmse(group["tvt_true"], group["v215"]),
        "v217_rmse": _rmse(group["tvt_true"], group["candidate"]),
    })
for row in fold_compare:
    row["gain"] = row["v215_rmse"] - row["v217_rmse"]

checks = {
    "coverage_773": True,
    "v215_parity": abs(base_rmse - 10.172981622636218) <= 1e-6,
    "spatial_gain_ge_0p10": base_rmse - candidate_rmse >= 0.10,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in fold_compare) >= 4,
    "minimum_fold_gain_ge_minus0p05": min(row["gain"] for row in fold_compare) >= -0.05,
}
verdict = {
    "experiment": "likpf_residual_foldclean_spatial_v217",
    "stage": "paired_V215_representation_plus_foldclean_spatial_features",
    "features": {
        "v215_nonspatial": len(base_features),
        "foldclean_spatial": len(spatial_features),
        "total": len(features),
        "sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
    },
    "values": {
        "v215_rmse": base_rmse,
        "v217_rmse": candidate_rmse,
        "spatial_gain": base_rmse - candidate_rmse,
    },
    "folds": fold_compare,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> test CatBoost/XGBoost diversity on this representation" if all(checks.values()) else "FAIL -> fold-clean spatial inputs do not improve the stronger V215 representation",
}

oof.to_parquet(OUT / "oof.parquet", index=False)
pd.DataFrame(fold_reports).to_csv(OUT / "fold_metrics.csv", index=False)
pd.DataFrame(fold_compare).to_csv(OUT / "paired_v215_comparison.csv", index=False)
pd.Series(base_features, name="feature").to_csv(OUT / "base_features.csv", index=False)
pd.Series(spatial_features, name="feature").to_csv(OUT / "spatial_features.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")

print("\nV217 PAIRED FOLDS")
print(pd.DataFrame(fold_compare).to_string(index=False))
print("\nV217 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
