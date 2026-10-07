"""V215: self-trained likelihood-PF residual GBDT from raw well files.

Run with ``%run -i`` after the original notebook's function-definition cells
have defined ``build_well`` and ``lik_pf``. No pretrained booster, historical
endpoint, CP, Neighbor, or train-only formation surface is consumed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.signal import savgol_filter


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF = ROOT / "oof_exports"
OUT = OOF / "likpf_residual_gbdt_full5_v215"
CACHE = OUT / "features"
TEMP = OUT / "masked_tmp"
MODELS = OUT / "models"
for path in (OUT, CACHE, TEMP, MODELS):
    path.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SEED = 215
TRAIN_ROWS_PER_WELL = 1024
FEATURE_WORKERS = max(1, int(os.environ.get("V215_WORKERS", "4")))
LIKPF_COLUMN = "likpf_scale_5"
WARM_TAU = 85.0
WARM_FLOOR = 0.15
SG_WINDOW = 61
SG_POLY = 3

if not (DATA_ROOT / "train").is_dir():
    raise FileNotFoundError(DATA_ROOT / "train")
missing_globals = [name for name in ("build_well", "lik_pf") if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original notebook function-definition cells, then execute "
        "%run -i /root/autodl-tmp/likpf_residual_gbdt_full5_v215.py. Missing: "
        + repr(missing_globals)
    )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _balanced_folds(meta):
    ordered = meta.sort_values(["n_eval", "well"], ascending=[False, True])
    loads = np.zeros(N_FOLDS, np.int64)
    assignment = {}
    for row in ordered.itertuples(index=False):
        fold = int(np.argmin(loads)) + 1
        assignment[str(row.well)] = fold
        loads[fold - 1] += int(row.n_eval)
    return assignment, loads


def _scan_metadata():
    rows = []
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv")):
        well = path.name.replace("__horizontal_well.csv", "")
        typewell = DATA_ROOT / "train" / f"{well}__typewell.csv"
        if not typewell.is_file():
            continue
        values = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(values))
        hidden = np.flatnonzero(~np.isfinite(values))
        if len(known) >= 64 and len(hidden) >= 128 and hidden[0] == len(known):
            rows.append({"well": well, "n_eval": len(hidden)})
    meta = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    if len(meta) != 773:
        raise RuntimeError(f"Expected 773 eligible wells, found {len(meta)}")
    fold_candidates = (
        OOF / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        OOF / "frontier_6p377_full_fivefold_lineage_v66" / "well_metadata.csv",
    )
    fold_path = next((path for path in fold_candidates if path.is_file()), None)
    if fold_path is not None:
        prior = pd.read_csv(fold_path, dtype={"well": str})
        if {"well", "fold"}.issubset(prior.columns) and prior["well"].nunique() == 773:
            meta = meta.merge(prior[["well", "fold"]].drop_duplicates("well"), on="well", validate="one_to_one")
            return meta, str(fold_path)
    mapping, loads = _balanced_folds(meta)
    meta["fold"] = meta["well"].map(mapping).astype(int)
    return meta, f"deterministic_balanced_eval_rows:{loads.tolist()}"


meta, fold_source = _scan_metadata()
meta.to_csv(OUT / "well_metadata.csv", index=False)


class _NeutralFormationImputer:
    def impute(self, xy_q, self_wid=None):
        n = len(np.atleast_2d(xy_q))
        count = len(globals().get("FORMATIONS", [0, 1, 2, 3, 4, 5]))
        values = np.broadcast_to(np.arange(count, dtype=np.float32) * 100.0, (n, count)).copy()
        return values, np.full(n, 1e6, np.float32)


class _NeutralDenseImputer:
    def impute(self, xy_q, self_wid=None, k=None, nfetch=None):
        n = len(np.atleast_2d(xy_q))
        return np.zeros(n, np.float32), np.zeros(n, np.float32), np.full(n, 1e6, np.float32)


# The function was defined in the interactive notebook. Patch its own global
# namespace, not merely this script's namespace.
builder_globals = build_well.__globals__
old_fi = builder_globals.get("_FI")
old_di = builder_globals.get("_DI")
builder_globals["_FI"] = _NeutralFormationImputer()
builder_globals["_DI"] = _NeutralDenseImputer()


def _add_likpf(frame, horizontal, typewell):
    last = frame["last_known_tvt"].to_numpy(np.float32)
    defaults = ("likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12", "likpf_mean")
    for column in defaults:
        frame[column] = last
    output, row_index, _ = lik_pf(horizontal.copy(), typewell.copy())
    if len(row_index):
        table = pd.DataFrame({"_row_idx": np.asarray(row_index, int)})
        generated = []
        for key, values in output.items():
            column = "likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
            table[column] = np.asarray(values, np.float32)
            generated.append(column)
        frame = frame.drop(columns=[column for column in generated if column in frame.columns])
        frame = frame.merge(table, on="_row_idx", how="left", validate="one_to_one")
    for column in defaults:
        if column not in frame:
            frame[column] = frame["last_known_tvt"]
        frame[column] = frame[column].fillna(frame["last_known_tvt"]).astype(np.float32)
        frame[column + "_d"] = (frame[column] - frame["last_known_tvt"]).astype(np.float32)
    return frame


def _paths(well):
    return CACHE / f"{well}__sample.pkl", CACHE / f"{well}__full.pkl"


def _build_one(well, fold):
    sample_path, full_path = _paths(well)
    if sample_path.is_file() and full_path.is_file():
        return {"well": well, "status": "cached"}
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    temp_path = TEMP / f"{well}__horizontal_well.csv"
    try:
        source = pd.read_csv(horizontal_path)
        legal = source[["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]].copy()
        legal.to_csv(temp_path, index=False)
        typewell = pd.read_csv(typewell_path).sort_values("TVT")
        frame = build_well(str(temp_path), str(typewell_path), is_train=True)
        if frame is None or len(frame) < 64:
            raise RuntimeError("build_well returned too few rows")
        frame = frame.reset_index(drop=True)
        frame["_row_idx"] = frame["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
        frame = _add_likpf(frame, legal, typewell)
        frame["well"] = well
        frame["id"] = well + "_" + frame["_row_idx"].astype(str)
        frame["fold"] = int(fold)
        last = frame["last_known_tvt"].to_numpy(float)
        lik_delta = frame[LIKPF_COLUMN].to_numpy(float) - last
        warm = 1.0 - np.exp(-np.maximum(frame["md_since"].to_numpy(float), 0.0) / WARM_TAU)
        divisor = np.maximum(warm, WARM_FLOOR)
        frame["residual_target"] = np.clip((frame["target"].to_numpy(float) - lik_delta) / divisor, -80.0, 80.0).astype(np.float32)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame.to_pickle(full_path)
        positions = np.arange(len(frame))
        if len(frame) > TRAIN_ROWS_PER_WELL:
            positions = np.unique(np.round(np.linspace(0, len(frame) - 1, TRAIN_ROWS_PER_WELL)).astype(int))
        sample = frame.iloc[positions].copy()
        sample["sample_weight"] = len(frame) / len(sample)
        sample.to_pickle(sample_path)
        return {"well": well, "status": "built", "rows": len(frame)}
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}
    finally:
        if temp_path.is_file():
            temp_path.unlink()


print("V215 feature build", {"wells": len(meta), "workers": FEATURE_WORKERS, "fold_source": fold_source}, flush=True)
try:
    reports = Parallel(n_jobs=FEATURE_WORKERS, prefer="threads", verbose=10)(
        delayed(_build_one)(str(row.well), int(row.fold)) for row in meta.itertuples(index=False)
    )
finally:
    builder_globals["_FI"] = old_fi
    builder_globals["_DI"] = old_di
pd.DataFrame(reports).to_csv(OUT / "feature_build_report.csv", index=False)
failures = [row for row in reports if row["status"] == "failed"]
if failures:
    raise RuntimeError(f"V215 feature failures={len(failures)} first={failures[:3]}")


SPATIAL_PREFIXES = ("tvtF_", "tvtFw_", "tvtF50_", "bw_", "bww_", "bw50_", "bw_early_", "bw_mid_", "frm_rmse_", "dense_", "tvt_dense")
SPATIAL_EXACT = {"form_mean_d", "form_std_d", "form_rng_d", "spatial_ancc_d", "spatial_knn_dist", "pf_vs_spatial", "pf_vs_dense", "spatial_vs_dense", "beam_vs_spatial", "sig_std", "sig_mean_d"}
FORBIDDEN = {"target", "residual_target", "TVT", "tvt", "truth", "well", "id", "fold", "_row_idx", "sample_weight", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
LINEAGE = ("strong", "cp8", "cp12", "gold", "neighbor", "v66", "v76", "v116", "v127", "v144", "v168", "incumbent")


def _load(wells, full):
    index = 1 if full else 0
    return pd.concat([pd.read_pickle(_paths(str(well))[index]) for well in wells], ignore_index=True, sort=False)


schema = _load(meta["well"].head(8), full=False)
features = []
for column in schema.columns:
    lowered = column.lower()
    spatial = column in SPATIAL_EXACT or column.startswith(SPATIAL_PREFIXES)
    lineage = any(token in lowered for token in LINEAGE)
    if column not in FORBIDDEN and not spatial and not lineage and pd.api.types.is_numeric_dtype(schema[column]):
        features.append(column)
del schema
if len(features) < 80:
    raise RuntimeError(f"Only {len(features)} legal features")
pd.Series(features, name="feature").to_csv(OUT / "features.csv", index=False)


def _matrix(frame):
    return frame.reindex(columns=features).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)


def _model(fold):
    params = dict(objective="regression", n_estimators=3000, learning_rate=0.025, num_leaves=64, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=10.0, max_bin=255, n_jobs=-1, random_state=SEED + fold, verbosity=-1)
    if os.environ.get("V215_CPU", "0") != "1":
        params["device_type"] = "gpu"
    return LGBMRegressor(**params)


def _fit(train, valid, fold):
    model = _model(fold)
    kwargs = dict(sample_weight=train["sample_weight"].to_numpy(float), eval_set=[(_matrix(valid), valid["residual_target"].to_numpy(float))], eval_metric="rmse", callbacks=[early_stopping(200, verbose=False), log_evaluation(100)])
    try:
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    except Exception as exc:
        if "gpu" not in str(exc).lower() and "opencl" not in str(exc).lower():
            raise
        print("V215 GPU fallback", repr(exc), flush=True)
        os.environ["V215_CPU"] = "1"
        model = _model(fold)
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
            output[offset:offset+n] = savgol_filter(prediction[offset:offset+n], window, SG_POLY)
        offset += n
    return output


parts = []
fold_reports = []
for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[~meta["fold"].eq(fold), "well"].tolist()
    valid_wells = meta.loc[meta["fold"].eq(fold), "well"].tolist()
    train = _load(train_wells, full=False)
    valid = _load(valid_wells, full=True).sort_values(["well", "_row_idx"], kind="stable").reset_index(drop=True)
    print(f"V215 fold {fold} train={len(train)} valid={len(valid)}", flush=True)
    model = _fit(train, valid, fold)
    residual = model.predict(_matrix(valid), num_iteration=model.best_iteration_)
    last = valid["last_known_tvt"].to_numpy(float)
    lik = valid[LIKPF_COLUMN].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(valid["md_since"].to_numpy(float), 0.0) / WARM_TAU)
    candidate = _smooth(valid, lik + warm * residual)
    truth = last + valid["target"].to_numpy(float)
    report = {"fold": fold, "rows": len(valid), "wells": len(valid_wells), "best_iteration": int(model.best_iteration_ or model.n_estimators), "likpf_rmse": _rmse(truth, lik), "candidate_rmse": _rmse(truth, candidate)}
    report["gain"] = report["likpf_rmse"] - report["candidate_rmse"]
    fold_reports.append(report)
    print("V215 FOLD", report, flush=True)
    parts.append(pd.DataFrame({"id": valid["id"].astype(str), "well": valid["well"].astype(str), "fold": fold, "md_since": valid["md_since"], "tvt_true": truth, "likpf": lik, "candidate": candidate}))
    joblib.dump(model, MODELS / f"fold{fold}.joblib")
    del train, valid, model, residual, candidate
    gc.collect()

oof = pd.concat(parts, ignore_index=True)


def _metrics(column):
    error = oof[column].to_numpy(float) - oof["tvt_true"].to_numpy(float)
    well_rmse = pd.Series(error * error).groupby(oof["well"].to_numpy()).mean().pow(0.5)
    far = oof["md_since"].to_numpy(float) >= 2000.0
    return {"endpoint": column, "rows": len(oof), "wells": oof["well"].nunique(), "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(well_rmse.median()), "p95_well_rmse": float(well_rmse.quantile(0.95)), "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2)))}


summary = pd.DataFrame([_metrics("likpf"), _metrics("candidate")])
fold_frame = pd.DataFrame(fold_reports)
candidate_rmse = float(summary.loc[summary["endpoint"].eq("candidate"), "rmse"].iloc[0])
checks = {"coverage_773": len(oof) > 3_700_000 and oof["well"].nunique() == 773, "candidate_rmse_le_7p8": candidate_rmse <= 7.8, "positive_folds_ge_4": int((fold_frame["gain"] > 0).sum()) >= 4, "no_public_boosters": True}
verdict = {"experiment": "likpf_residual_gbdt_full5_v215", "stage": "rawfile_rebuilt_selftrained_likPF_residual_GBDT", "fold_source": fold_source, "features": len(features), "feature_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(), "summary": summary.to_dict(orient="records"), "folds": fold_reports, "checks": checks, "strict_pass": bool(all(checks.values())), "elapsed_minutes": (time.time() - STARTED) / 60.0, "next": "PASS -> build one Kaggle train-and-infer notebook" if all(checks.values()) else "FAIL -> independent residual GBDT is not a 7.x engine; do not submit"}
oof.to_parquet(OUT / "oof.parquet", index=False)
summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
print("\nV215 SUMMARY\n", summary.to_string(index=False), flush=True)
print("\nV215 FOLDS\n", fold_frame.to_string(index=False), flush=True)
print("\nV215 FINAL VERDICT\n", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT, flush=True)
