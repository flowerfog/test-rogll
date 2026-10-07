"""Cross-fitted full-spatial fleongg LightGBM A/B.

Run after the original fleongg function-definition cells have defined:
    build_well, lik_pf, FormationPlaneKNN, DenseANCCImputer.

Each well's official-start feature frame is rebuilt once with FI/DI references
excluding that well's entire validation fold. The model experiment compares:

    no_spatial: all deployable non-spatial features;
    foldclean_spatial: the same features plus cross-fitted FI/DI features.

Target, folds, row sampling, model parameters, and post-processing are shared.
Feature generation is cached per well and resumes safely.
"""

from pathlib import Path
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from joblib import Parallel, delayed
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = OUT_DIR / "features"
TEMP_DIR = OUT_DIR / "masked_tmp"
MODEL_DIR = OUT_DIR / "models"

N_FOLDS = 5
TRAIN_ROWS_PER_WELL = 1024
N_JOBS_FEATURES = 2
SEED = 20260713

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

PASS_OVERALL_GAIN = 0.25
PASS_MIN_FOLD_GAIN = 0.05

required_globals = [
    "build_well",
    "lik_pf",
    "FormationPlaneKNN",
    "DenseANCCImputer",
]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original fleongg function-definition cells first; missing: "
        + repr(missing_globals)
    )

for directory in (OOF_ROOT, OUT_DIR, FEATURE_DIR, TEMP_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _rmse(y_true, prediction):
    return float(
        np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(prediction)))
    )


def _balanced_folds(meta):
    ordered = meta.sort_values(["n_eval", "well"], ascending=[False, True])
    loads = np.zeros(N_FOLDS, dtype=np.int64)
    assignment = {}
    for row in ordered.itertuples(index=False):
        fold = int(np.argmin(loads)) + 1
        assignment[str(row.well)] = fold
        loads[fold - 1] += int(row.n_eval)
    return assignment, loads


def _evenly_spaced_positions(n_rows, n_take):
    if n_rows <= n_take:
        return np.arange(n_rows, dtype=int)
    return np.unique(
        np.round(np.linspace(0, n_rows - 1, int(n_take))).astype(int)
    )


def _feature_paths(wid):
    return (
        FEATURE_DIR / f"{wid}__sample.pkl",
        FEATURE_DIR / f"{wid}__full.pkl",
    )


def _add_likpf(frame, horizontal, typewell):
    defaults = [
        "likpf_scale_3",
        "likpf_scale_5",
        "likpf_scale_8",
        "likpf_scale_12",
        "likpf_mean",
    ]
    last = frame["last_known_tvt"].to_numpy(np.float32)
    for column in defaults:
        frame[column] = last
    try:
        output, row_index, _ = lik_pf(horizontal.copy(), typewell.copy())
        if not len(row_index):
            return frame
        generated = {}
        for key, values in output.items():
            column = "likpf_" + key.replace("pf_scale_", "scale_").replace(
                "pf_mean", "mean"
            )
            generated[column] = np.asarray(values, np.float32)
        # Remove fallback columns before merging the real likelihood-PF output.
        # Otherwise pandas suffixes both copies with _x/_y and the canonical
        # column is later recreated from last_known_tvt, silently disabling PF.
        frame = frame.drop(
            columns=[column for column in generated if column in frame.columns]
        )
        table = pd.DataFrame({"_row_idx": np.asarray(row_index, dtype=int)})
        for column, values in generated.items():
            table[column] = values
        frame = frame.merge(table, on="_row_idx", how="left", validate="one_to_one")
        for column in defaults:
            if column not in frame:
                frame[column] = frame["last_known_tvt"]
            frame[column] = (
                frame[column]
                .fillna(frame["last_known_tvt"])
                .astype(np.float32)
            )
            frame[column + "_d"] = (
                frame[column] - frame["last_known_tvt"]
            ).astype(np.float32)
        return frame
    except Exception as exc:
        warnings.warn(f"lik_pf fallback: {exc!r}")
        for column in defaults:
            frame[column + "_d"] = np.zeros(len(frame), dtype=np.float32)
        return frame


# ---------------------------------------------------------------------------
# Metadata and frozen folds.
# ---------------------------------------------------------------------------
meta_rows = []
train_paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
for path in tqdm(train_paths, desc="scan fold-clean metadata"):
    wid = path.name.replace("__horizontal_well.csv", "")
    try:
        tvt_input = pd.read_csv(path, usecols=["TVT_input"])[
            "TVT_input"
        ].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        hidden = np.flatnonzero(~np.isfinite(tvt_input))
        if len(known) < 64 or len(hidden) < 128:
            continue
        if hidden[0] != len(known):
            continue
        meta_rows.append(
            {
                "well": wid,
                "n_known": len(known),
                "n_eval": len(hidden),
            }
        )
    except Exception:
        continue

meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
if len(meta) < 700:
    raise RuntimeError(f"only {len(meta)} eligible wells")
fold_map, fold_loads = _balanced_folds(meta)
meta["fold"] = meta["well"].map(fold_map).astype(int)
meta.to_csv(OUT_DIR / "well_metadata.csv", index=False)
print(
    {
        "experiment": "foldclean_full_fleongg_lgb_v0",
        "wells": len(meta),
        "fold_loads": fold_loads.tolist(),
        "train_rows_per_well": TRAIN_ROWS_PER_WELL,
        "feature_jobs": N_JOBS_FEATURES,
    }
)


def _build_one_well(wid, fold):
    sample_path, full_path = _feature_paths(wid)
    if sample_path.exists() and full_path.exists():
        return {"well": wid, "fold": fold, "status": "cached"}

    horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
    temp_path = TEMP_DIR / f"{wid}__horizontal_well.csv"
    try:
        source = pd.read_csv(horizontal_path)
        # Test-schema enforcement: target is retained only so build_well can
        # write its label column; formation surfaces never enter target-well X.
        keep = ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]
        missing = [column for column in keep if column not in source.columns]
        if missing:
            raise RuntimeError(f"missing source columns: {missing}")
        sanitized = source[keep].copy()
        sanitized.to_csv(temp_path, index=False)
        typewell = pd.read_csv(typewell_path).sort_values("TVT")

        frame = build_well(str(temp_path), str(typewell_path), is_train=True)
        if frame is None or len(frame) < 64:
            raise RuntimeError("build_well returned too few rows")
        frame = frame.reset_index(drop=True)
        frame["_row_idx"] = (
            frame["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
        )
        frame = _add_likpf(frame, sanitized, typewell)
        frame["well"] = wid
        frame["id"] = wid + "_" + frame["_row_idx"].astype(str)
        frame["fold"] = int(fold)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame.to_pickle(full_path)

        positions = _evenly_spaced_positions(
            len(frame), TRAIN_ROWS_PER_WELL
        )
        sample = frame.iloc[positions].copy()
        sample["sample_weight"] = len(frame) / max(len(sample), 1)
        sample.to_pickle(sample_path)
        return {
            "well": wid,
            "fold": fold,
            "status": "built",
            "rows": len(frame),
        }
    except Exception as exc:
        return {
            "well": wid,
            "fold": fold,
            "status": "failed",
            "reason": repr(exc),
        }
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cross-fitted feature generation. For fold f, FI/DI contain only the other
# four folds; every well in f is then built against those references.
# ---------------------------------------------------------------------------
original_fi = globals().get("_FI")
original_di = globals().get("_DI")
feature_records = []
feature_started = time.time()
try:
    for fold in range(1, N_FOLDS + 1):
        fold_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
        missing_wells = [
            wid
            for wid in fold_wells
            if not all(path.exists() for path in _feature_paths(wid))
        ]
        print(
            f"\nfeature fold {fold}/{N_FOLDS}: "
            f"valid={len(fold_wells)} missing={len(missing_wells)}",
            flush=True,
        )
        if not missing_wells:
            feature_records.extend(
                {"well": wid, "fold": fold, "status": "cached"}
                for wid in fold_wells
            )
            continue

        reference_wells = meta.loc[
            meta["fold"] != fold, "well"
        ].astype(str).tolist()
        globals()["_FI"] = FormationPlaneKNN(
            reference_wells, DATA_ROOT / "train"
        )
        globals()["_DI"] = DenseANCCImputer(
            reference_wells, DATA_ROOT / "train"
        )

        results = Parallel(
            n_jobs=N_JOBS_FEATURES,
            prefer="threads",
        )(
            delayed(_build_one_well)(wid, fold)
            for wid in tqdm(missing_wells, desc=f"build fold {fold}")
        )
        feature_records.extend(results)
        failures = [
            result for result in results if result["status"] == "failed"
        ]
        if failures:
            pd.DataFrame(failures).to_csv(
                OUT_DIR / f"feature_failures_fold{fold}.csv", index=False
            )
            display(pd.DataFrame(failures).head(30))
            raise RuntimeError(
                f"feature fold {fold} failed for {len(failures)} wells"
            )
        print(
            f"feature fold {fold} complete; "
            f"elapsed_hours={(time.time()-feature_started)/3600:.2f}",
            flush=True,
        )
finally:
    globals()["_FI"] = original_fi
    globals()["_DI"] = original_di

pd.DataFrame(feature_records).to_csv(
    OUT_DIR / "feature_build_report.csv", index=False
)


def _load_samples(wells):
    frames = []
    for wid in wells:
        sample_path, _ = _feature_paths(str(wid))
        frames.append(pd.read_pickle(sample_path))
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_full(wells):
    frames = []
    for wid in wells:
        _, full_path = _feature_paths(str(wid))
        frames.append(pd.read_pickle(full_path))
    return pd.concat(frames, ignore_index=True, sort=False)


SPATIAL_PREFIXES = (
    "tvtF_",
    "tvtFw_",
    "tvtF50_",
    "bw_",
    "bww_",
    "bw50_",
    "bw_early_",
    "bw_mid_",
    "frm_rmse_",
    "dense_",
    "tvt_dense",
)
SPATIAL_EXACT = {
    "form_mean_d",
    "form_std_d",
    "form_rng_d",
    "spatial_ancc_d",
    "spatial_knn_dist",
    "pf_vs_spatial",
    "pf_vs_dense",
    "spatial_vs_dense",
    "beam_vs_spatial",
    "sig_std",
    "sig_mean_d",
}


def _is_spatial_feature(column):
    return column in SPATIAL_EXACT or column.startswith(SPATIAL_PREFIXES)


def _feature_columns(frame):
    excluded = {
        "well",
        "id",
        "target",
        "_row_idx",
        "fold",
        "sample_weight",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


schema_wells = meta["well"].head(10).astype(str).tolist()
schema = _load_samples(schema_wells)
all_features = _feature_columns(schema)
no_spatial_features = [
    column for column in all_features if not _is_spatial_feature(column)
]
spatial_features = [
    column for column in all_features if _is_spatial_feature(column)
]
del schema
gc.collect()
print(
    {
        "all_features": len(all_features),
        "no_spatial_features": len(no_spatial_features),
        "spatial_features": len(spatial_features),
    }
)
if len(spatial_features) < 20:
    raise RuntimeError(
        f"only {len(spatial_features)} spatial features detected"
    )


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
        n_estimators=2400,
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
        random_state=SEED + fold + (0 if arm == "no_spatial" else 100),
        verbosity=-1,
    )


def _fit_model(train, valid, columns, fold, arm):
    model = _new_model(fold, arm)
    fit_kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid, columns), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(0)],
    )
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        gpu_model = LGBMRegressor(**params)
        gpu_model.fit(
            _matrix(train, columns),
            train["target"].to_numpy(float),
            **fit_kwargs,
        )
        return gpu_model
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU LightGBM unavailable; CPU fallback:", repr(exc))
        model.fit(
            _matrix(train, columns),
            train["target"].to_numpy(float),
            **fit_kwargs,
        )
        return model


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(float)
    md_since = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(float) - last
    if PP_LIKPF_COL in frame:
        lik_delta = frame[PP_LIKPF_COL].to_numpy(float) - last
    else:
        lik_delta = pf_delta
    learned = PP_ALPHA * warm * (
        model_delta * (1.0 - PP_W_PF_ANCC)
        + pf_delta * PP_W_PF_ANCC
    )
    prediction = (
        last
        + PP_W_MODEL * learned
        + (1.0 - PP_W_MODEL) * lik_delta
    )

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


# ---------------------------------------------------------------------------
# Identical LightGBM A/B on the cross-fitted feature cache.
# ---------------------------------------------------------------------------
oof_parts = []
fold_rows = []
model_rows = []
training_started = time.time()
for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    print(
        f"\nmodel fold {fold}/{N_FOLDS}: "
        f"train={len(train_wells)} valid={len(valid_wells)}",
        flush=True,
    )
    train = _load_samples(train_wells)
    valid = _load_full(valid_wells)
    predictions = {}

    for arm, columns in (
        ("no_spatial", no_spatial_features),
        ("foldclean_spatial", all_features),
    ):
        model = _fit_model(train, valid, columns, fold, arm)
        delta = model.predict(
            _matrix(valid, columns),
            num_iteration=model.best_iteration_,
        )
        predictions[arm] = _postprocess(valid, delta)
        joblib.dump(model, MODEL_DIR / f"{arm}_fold{fold}.joblib")
        model_rows.append(
            {
                "arm": arm,
                "fold": fold,
                "features": len(columns),
                "best_iteration": int(model.best_iteration_),
            }
        )

    truth = (
        valid["target"].to_numpy(float)
        + valid["last_known_tvt"].to_numpy(float)
    )
    no_spatial_rmse = _rmse(truth, predictions["no_spatial"])
    clean_rmse = _rmse(truth, predictions["foldclean_spatial"])
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_wells),
            "no_spatial_rmse": no_spatial_rmse,
            "foldclean_spatial_rmse": clean_rmse,
            "gain": no_spatial_rmse - clean_rmse,
        }
    )
    print("fold result:", fold_rows[-1], flush=True)
    oof_parts.append(
        pd.DataFrame(
            {
                "id": valid["id"].astype(str).to_numpy(),
                "well": valid["well"].astype(str).to_numpy(),
                "row_idx": valid["_row_idx"].to_numpy(int),
                "fold": fold,
                "tvt_true": truth,
                "no_spatial": predictions["no_spatial"],
                "foldclean_spatial": predictions["foldclean_spatial"],
            }
        )
    )
    del train, valid, predictions, model
    gc.collect()
    print(
        f"model elapsed_hours={(time.time()-training_started)/3600:.2f}",
        flush=True,
    )

oof = pd.concat(oof_parts, ignore_index=True)
fold_report = pd.DataFrame(fold_rows)
no_spatial_rmse = _rmse(oof["tvt_true"], oof["no_spatial"])
clean_rmse = _rmse(oof["tvt_true"], oof["foldclean_spatial"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    truth_well = group["tvt_true"].to_numpy(float)
    base_well = group["no_spatial"].to_numpy(float)
    clean_well = group["foldclean_spatial"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "no_spatial_sse": float(np.sum((base_well - truth_well) ** 2)),
            "no_spatial_rmse": _rmse(truth_well, base_well),
            "foldclean_spatial_rmse": _rmse(truth_well, clean_well),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = set(bywell.nlargest(40, "no_spatial_sse")["well"])
top_mask = oof["well"].isin(top40).to_numpy()
top40_gain = (
    _rmse(
        oof.loc[top_mask, "tvt_true"],
        oof.loc[top_mask, "no_spatial"],
    )
    - _rmse(
        oof.loc[top_mask, "tvt_true"],
        oof.loc[top_mask, "foldclean_spatial"],
    )
)

strict_pass = bool(
    no_spatial_rmse - clean_rmse >= PASS_OVERALL_GAIN
    and (fold_report["gain"] >= PASS_MIN_FOLD_GAIN).all()
)
result = {
    "no_spatial_rmse": no_spatial_rmse,
    "foldclean_spatial_rmse": clean_rmse,
    "gain": no_spatial_rmse - clean_rmse,
    "mean_fold_gain": float(fold_report["gain"].mean()),
    "min_fold_gain": float(fold_report["gain"].min()),
    "positive_folds": int((fold_report["gain"] > 0).sum()),
    "top40_gain": top40_gain,
    "feature_hours": (time.time() - feature_started) / 3600.0,
    "training_hours": (time.time() - training_started) / 3600.0,
    "strict_pass": strict_pass,
    "next": (
        "train CatBoost/XGBoost and full deployment stack"
        if strict_pass
        else "stop fold-clean spatial tabular route"
    ),
}

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nA/B FOLDS")
display(fold_report)
print("\nOVERALL")
print(result)
print(
    "\nVERDICT:",
    "PASS -> 3-model ensemble"
    if strict_pass
    else "FAIL / STOP fold-clean spatial",
)

oof.to_pickle(OUT_DIR / "foldclean_fleongg_oof.pkl")
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
bywell.to_csv(OUT_DIR / "bywell_report.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)
pd.Series(all_features, name="feature").to_csv(
    OUT_DIR / "all_features.csv", index=False
)
pd.Series(no_spatial_features, name="feature").to_csv(
    OUT_DIR / "no_spatial_features.csv", index=False
)
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", OUT_DIR)
