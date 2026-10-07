"""Full K5 pseudo-prediction-start rebuild for fleongg.

Run this as one notebook cell after the original fleongg notebook has defined
``build_well`` and ``lik_pf``. Do not run the original model-training cell.

The experiment compares three otherwise identical LightGBM models:

* single_cut: trained only on each training well's official prediction start;
* k3: official plus fixed 50% and 75% pseudo prediction starts;
* k5: official plus fixed 35%, 50%, 65%, and 75% pseudo starts.

Both models are evaluated only on held-out wells at their official prediction
start. Grouping is always by the original well. Spatial FI/DI features are
neutralised during feature construction and removed before fitting, because
their historical OOF lineage is fold-contaminated.
"""

from pathlib import Path
import gc
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Fixed, pre-registered v0 configuration.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
CACHE_VERSION = "k5_full_rebuild_v2_ps35_50_65_75"
CACHE_DIR = OOF_DIR / f"multicut_fleongg_{CACHE_VERSION}"
FEATURE_DIR = CACHE_DIR / "features"
TEMP_DIR = CACHE_DIR / "masked_tmp"
MODEL_DIR = CACHE_DIR / "models"

N_FOLDS = 5
SYNTHETIC_FRACS = (0.35, 0.50, 0.65, 0.75)
N_SYNTHETIC_CUTS = len(SYNTHETIC_FRACS)
K3_TAGS = ("official", "synth2", "synth4")
K5_TAGS = ("official", "synth1", "synth2", "synth3", "synth4")
TRAIN_ROWS_PER_CUT = 768
N_JOBS_FEATURES = 4
RANDOM_STATE = 20260711

# One deliberately conservative model. Full stack is allowed only after v0 passes.
LGB_PARAMS = dict(
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
    random_state=RANDOM_STATE,
    verbosity=-1,
)

# Fixed fleongg post-processing, identical for the A/B comparison.
PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

PASS_K5_VS_SINGLE = 0.25
PASS_K5_VS_K3 = 0.10
PASS_CP12_NESTED = 0.05
ALPHA_GRID = np.round(np.arange(0.0, 0.5001, 0.025), 3)

required_globals = ["build_well", "lik_pf"]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original fleongg function-definition cells first; missing: "
        + repr(missing_globals)
    )

for directory in (OOF_DIR, CACHE_DIR, FEATURE_DIR, TEMP_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print(
    {
        "experiment": "multicut_fleongg_k5_full_rebuild_v2",
        "cache": str(CACHE_DIR),
        "cuts_per_well": 1 + N_SYNTHETIC_CUTS,
        "synthetic_fracs": SYNTHETIC_FRACS,
        "k3_tags": K3_TAGS,
        "train_rows_per_cut": TRAIN_ROWS_PER_CUT,
        "feature_jobs": N_JOBS_FEATURES,
        "validation": "official cut only",
        "spatial_features": "neutralised and dropped",
    }
)


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evenly_spaced_positions(n_rows, n_take):
    if n_rows <= n_take:
        return np.arange(n_rows, dtype=int)
    return np.unique(
        np.round(np.linspace(0, n_rows - 1, int(n_take))).astype(int)
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


class _NeutralFormationImputer:
    """Finite placeholders for build_well; every derived column is dropped."""

    def impute(self, xy_q, self_wid=None):
        n = len(np.atleast_2d(xy_q))
        n_formations = len(globals().get("FORMATIONS", [0, 1, 2, 3, 4, 5]))
        offsets = np.arange(n_formations, dtype=np.float32) * 100.0
        values = np.broadcast_to(offsets, (n, n_formations)).copy()
        return values, np.full(n, 1e6, dtype=np.float32)


class _NeutralDenseImputer:
    def impute(self, xy_q, self_wid=None, k=None, nfetch=None):
        n = len(np.atleast_2d(xy_q))
        return (
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.full(n, 1e6, dtype=np.float32),
        )


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


def _add_likpf_to_frame(frame, masked_hw, typewell, pseudo_wid):
    """Build the four lik-PF columns with ids matching build_well output."""
    defaults = [
        "likpf_scale_3",
        "likpf_scale_5",
        "likpf_scale_8",
        "likpf_mean",
    ]
    last = frame["last_known_tvt"].to_numpy(np.float32)
    for column in defaults:
        frame[column] = last

    try:
        output, row_index, _ = lik_pf(masked_hw.copy(), typewell.copy())
        if not len(row_index):
            return frame
        lp = pd.DataFrame({"_row_idx": np.asarray(row_index, dtype=int)})
        generated = []
        for key, value in output.items():
            column = "likpf_" + key.replace("pf_scale_", "scale_").replace(
                "pf_mean", "mean"
            )
            lp[column] = np.asarray(value, dtype=np.float32)
            generated.append(column)
        frame = frame.drop(columns=[c for c in generated if c in frame.columns])
        frame = frame.merge(lp, on="_row_idx", how="left", validate="one_to_one")
        for column in defaults:
            if column not in frame:
                frame[column] = frame["last_known_tvt"]
            frame[column] = frame[column].fillna(frame["last_known_tvt"]).astype(
                np.float32
            )
        return frame
    except Exception as exc:
        warnings.warn(f"lik_pf fallback for {pseudo_wid}: {exc!r}")
        return frame


def _build_masked_source(source, cut_position, official):
    keep = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]
    masked = source[keep].copy().reset_index(drop=True)
    if not official:
        truth = masked["TVT"].to_numpy(float)
        simulated = np.full(len(masked), np.nan, dtype=float)
        simulated[:cut_position] = truth[:cut_position]
        masked["TVT_input"] = simulated
    return masked


def _feature_paths(wid, tag):
    sample_path = FEATURE_DIR / f"{wid}__{tag}__sample.pkl"
    full_path = FEATURE_DIR / f"{wid}__official_full.pkl" if tag == "official" else None
    return sample_path, full_path


def _build_one_task(task):
    wid = str(task["well"])
    tag = str(task["tag"])
    cut_position = int(task["cut_position"])
    official = bool(task["official"])
    official_n_eval = int(task["official_n_eval"])
    sample_path, full_path = _feature_paths(wid, tag)

    if sample_path.exists() and (not official or full_path.exists()):
        return {"well": wid, "tag": tag, "status": "cached"}

    horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
    pseudo_wid = f"{wid}_mc_{tag}"
    temp_path = TEMP_DIR / f"{pseudo_wid}__horizontal_well.csv"

    try:
        source = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path).sort_values("TVT")
        masked = _build_masked_source(source, cut_position, official)
        masked.to_csv(temp_path, index=False)

        frame = build_well(str(temp_path), str(typewell_path), is_train=True)
        if frame is None or len(frame) < 64:
            raise RuntimeError("build_well returned too few rows")

        frame = frame.reset_index(drop=True)
        frame["_row_idx"] = (
            frame["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
        )
        frame = _add_likpf_to_frame(frame, masked, typewell, pseudo_wid)
        frame["well"] = wid
        frame["id"] = wid + "_" + frame["_row_idx"].astype(str)
        frame["_cut_tag"] = tag
        frame["_official_n_eval"] = official_n_eval
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

        if official:
            frame.to_pickle(full_path)

        positions = _evenly_spaced_positions(len(frame), TRAIN_ROWS_PER_CUT)
        sample = frame.iloc[positions].copy()
        # Every original well has the same total weight in every training arm.
        sample["_sample_weight_k3"] = official_n_eval / (
            len(K3_TAGS) * max(1, len(sample))
        )
        sample["_sample_weight_k5"] = official_n_eval / (
            len(K5_TAGS) * max(1, len(sample))
        )
        sample["_sample_weight_single"] = official_n_eval / max(1, len(sample))
        sample.to_pickle(sample_path)
        return {"well": wid, "tag": tag, "status": "built", "rows": len(frame)}
    except Exception as exc:
        return {"well": wid, "tag": tag, "status": "failed", "reason": repr(exc)}
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scan official starts and freeze four fixed pseudo prediction starts.
# ---------------------------------------------------------------------------
meta_rows = []
train_files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
for path in tqdm(train_files, desc="scan multi-cut metadata"):
    wid = path.name.replace("__horizontal_well.csv", "")
    try:
        tvt_input = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        hidden = np.flatnonzero(~np.isfinite(tvt_input))
        if len(known) < 64 or len(hidden) < 128:
            continue
        official_cut = int(hidden[0])
        if not np.all(np.isfinite(tvt_input[:official_cut])):
            continue
        if not np.all(~np.isfinite(tvt_input[official_cut:])):
            continue
        meta_rows.append(
            {
                "well": wid,
                "n_rows": len(tvt_input),
                "official_cut": official_cut,
                "visible_frac": official_cut / len(tvt_input),
                "n_eval": len(hidden),
            }
        )
    except Exception:
        continue

meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
if len(meta) < 700:
    raise RuntimeError(f"Only {len(meta)} eligible wells; expected roughly 773")

synthetic_fracs = np.asarray(SYNTHETIC_FRACS, dtype=float)
fold_map, fold_loads = _balanced_folds(meta)
meta["fold"] = meta["well"].map(fold_map).astype(int)
meta.to_csv(CACHE_DIR / "well_metadata.csv", index=False)

print("eligible wells:", len(meta))
print("fixed synthetic visible fractions:", synthetic_fracs.tolist())
print("fold row loads:", fold_loads.tolist())

tasks = []
for row in meta.itertuples(index=False):
    cuts = [("official", int(row.official_cut), True)]
    used = {int(row.official_cut)}
    for number, fraction in enumerate(synthetic_fracs, 1):
        cut = int(np.clip(round(float(fraction) * row.n_rows), 64, row.n_rows - 128))
        if cut in used:
            direction = -1 if number == 1 else 1
            cut = int(np.clip(cut + direction * max(32, row.n_rows // 20), 64, row.n_rows - 128))
        used.add(cut)
        cuts.append((f"synth{number}", cut, False))
    for tag, cut, official in cuts:
        tasks.append(
            {
                "well": str(row.well),
                "tag": tag,
                "cut_position": int(cut),
                "official": bool(official),
                "official_n_eval": int(row.n_eval),
            }
        )

pd.DataFrame(tasks).to_csv(CACHE_DIR / "cut_manifest.csv", index=False)


# ---------------------------------------------------------------------------
# Feature generation. Neutral globals make this independent of all-well FI/DI.
# ---------------------------------------------------------------------------
original_fi = globals().get("_FI", None)
original_di = globals().get("_DI", None)
globals()["_FI"] = _NeutralFormationImputer()
globals()["_DI"] = _NeutralDenseImputer()

feature_started = time.time()
try:
    from joblib import Parallel, delayed

    smoke_well = str(meta.iloc[0]["well"])
    smoke_tasks = [task for task in tasks if task["well"] == smoke_well]
    smoke_results = [_build_one_task(task) for task in smoke_tasks]
    smoke_failed = [row for row in smoke_results if row["status"] == "failed"]
    print("feature smoke test:", smoke_results, flush=True)
    if smoke_failed:
        raise RuntimeError("Feature smoke test failed: " + repr(smoke_failed))

    results = Parallel(n_jobs=N_JOBS_FEATURES, prefer="threads")(
        delayed(_build_one_task)(task)
        for task in tqdm(tasks, desc="multi-cut feature tasks")
    )
finally:
    globals()["_FI"] = original_fi
    globals()["_DI"] = original_di

feature_status = pd.DataFrame(results)
feature_status.to_csv(CACHE_DIR / "feature_status.csv", index=False)
print("feature status:", feature_status["status"].value_counts().to_dict())
print("feature elapsed hours:", (time.time() - feature_started) / 3600.0)

failed = feature_status[feature_status["status"] == "failed"]
if len(failed):
    display(failed.head(30))
if len(failed) > max(10, 0.02 * len(tasks)):
    raise RuntimeError(f"Too many feature failures: {len(failed)} / {len(tasks)}")

valid_wells = []
for wid in meta["well"].astype(str):
    required_paths = [_feature_paths(wid, tag)[0] for tag in K5_TAGS]
    official_full = _feature_paths(wid, "official")[1]
    if all(path.exists() for path in required_paths) and official_full.exists():
        valid_wells.append(wid)

meta = meta[meta["well"].isin(valid_wells)].reset_index(drop=True)
print("complete wells for training:", len(meta))
if len(meta) < 700:
    raise RuntimeError("Too few complete wells after feature generation")


def _load_samples(wells, tags, keep_columns=None):
    frames = []
    for wid in wells:
        for tag in tags:
            path, _ = _feature_paths(str(wid), tag)
            if path.exists():
                frame = pd.read_pickle(path)
                if keep_columns is not None:
                    for column in keep_columns:
                        if column not in frame.columns:
                            frame[column] = np.nan
                    frame = frame[list(keep_columns)].copy()
                frames.append(frame)
    if not frames:
        raise RuntimeError("No training samples loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_official_full(wells):
    frames = []
    for wid in wells:
        _, path = _feature_paths(str(wid), "official")
        frames.append(pd.read_pickle(path))
    return pd.concat(frames, ignore_index=True, sort=False)


def _feature_columns(frame):
    excluded = {
        "well",
        "id",
        "target",
        "_row_idx",
        "_cut_tag",
        "_official_n_eval",
        "_sample_weight_k3",
        "_sample_weight_k5",
        "_sample_weight_single",
    }
    columns = []
    for column in frame.columns:
        if column in excluded or _is_spatial_feature(column):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def _matrix(frame, columns):
    return frame.reindex(columns=columns).to_numpy(np.float32)


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
        model_delta * (1.0 - PP_W_PF_ANCC) + pf_delta * PP_W_PF_ANCC
    )
    prediction = last + PP_W_MODEL * learned + (1.0 - PP_W_MODEL) * lik_delta

    output = prediction.copy()
    offset = 0
    for _, group in frame.groupby("well", sort=False):
        n = len(group)
        values = prediction[offset : offset + n]
        window = min(PP_SG_WIN, n)
        if window % 2 == 0:
            window -= 1
        if window >= PP_SG_POLY + 2:
            output[offset : offset + n] = savgol_filter(values, window, PP_SG_POLY)
        offset += n
    return output


def _fit_lgb(train, valid, feature_columns, weight_column, model_name, fold):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(LGB_PARAMS)
    params["device_type"] = "gpu"
    model = LGBMRegressor(**params)
    fit_kwargs = dict(
        sample_weight=train[weight_column].to_numpy(float),
        eval_set=[(_matrix(valid, feature_columns), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(100)],
    )
    try:
        model.fit(
            _matrix(train, feature_columns),
            train["target"].to_numpy(float),
            **fit_kwargs,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU LightGBM unavailable; falling back to CPU:", repr(exc))
        params.pop("device_type", None)
        model = LGBMRegressor(**params)
        model.fit(
            _matrix(train, feature_columns),
            train["target"].to_numpy(float),
            **fit_kwargs,
        )
    joblib.dump(model, MODEL_DIR / f"{model_name}_fold{fold}.joblib")
    return model


# Freeze the feature list from all training sample schemas before CV.
schema_probe = _load_samples(meta["well"].astype(str).head(10), list(K5_TAGS))
feature_columns = _feature_columns(schema_probe)
del schema_probe
print("model features:", len(feature_columns))
print(feature_columns)


# ---------------------------------------------------------------------------
# Five-fold single/K3/K5 comparison. Validation is untouched official PS.
# ---------------------------------------------------------------------------
oof_parts = []
fold_records = []
training_started = time.time()

for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_fold_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    print(
        f"\nfold {fold}/{N_FOLDS}: train_wells={len(train_wells)} "
        f"valid_wells={len(valid_fold_wells)}",
        flush=True,
    )

    valid = _load_official_full(valid_fold_wells)

    single_columns = list(dict.fromkeys(feature_columns + ["target", "_sample_weight_single"]))
    train_single = _load_samples(train_wells, ["official"], single_columns)
    model_single = _fit_lgb(
        train_single, valid, feature_columns, "_sample_weight_single", "single", fold
    )
    del train_single
    gc.collect()

    k3_columns = list(dict.fromkeys(feature_columns + ["target", "_sample_weight_k3"]))
    train_k3 = _load_samples(train_wells, list(K3_TAGS), k3_columns)
    model_k3 = _fit_lgb(
        train_k3, valid, feature_columns, "_sample_weight_k3", "k3", fold
    )
    del train_k3
    gc.collect()

    k5_columns = list(dict.fromkeys(feature_columns + ["target", "_sample_weight_k5"]))
    train_k5 = _load_samples(train_wells, list(K5_TAGS), k5_columns)
    model_k5 = _fit_lgb(
        train_k5, valid, feature_columns, "_sample_weight_k5", "k5", fold
    )
    del train_k5
    gc.collect()

    matrix_valid = _matrix(valid, feature_columns)
    delta_single = model_single.predict(
        matrix_valid, num_iteration=model_single.best_iteration_
    )
    delta_k3 = model_k3.predict(matrix_valid, num_iteration=model_k3.best_iteration_)
    delta_k5 = model_k5.predict(matrix_valid, num_iteration=model_k5.best_iteration_)
    pred_single = _postprocess(valid, delta_single)
    pred_k3 = _postprocess(valid, delta_k3)
    pred_k5 = _postprocess(valid, delta_k5)
    truth = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)

    single_rmse = _rmse(truth, pred_single)
    k3_rmse = _rmse(truth, pred_k3)
    k5_rmse = _rmse(truth, pred_k5)
    fold_records.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_fold_wells),
            "single_rmse": single_rmse,
            "k3_rmse": k3_rmse,
            "k5_rmse": k5_rmse,
            "k3_gain_vs_single": single_rmse - k3_rmse,
            "k5_gain_vs_single": single_rmse - k5_rmse,
            "k5_gain_vs_k3": k3_rmse - k5_rmse,
            "single_best_iteration": int(model_single.best_iteration_),
            "k3_best_iteration": int(model_k3.best_iteration_),
            "k5_best_iteration": int(model_k5.best_iteration_),
        }
    )
    print("fold result:", fold_records[-1], flush=True)

    oof_parts.append(
        pd.DataFrame(
            {
                "id": valid["id"].astype(str).to_numpy(),
                "well": valid["well"].astype(str).to_numpy(),
                "row_idx": valid["_row_idx"].to_numpy(int),
                "fold": fold,
                "tvt_true": truth,
                "pred_single": pred_single,
                "pred_k3": pred_k3,
                "pred_k5": pred_k5,
            }
        )
    )

    del (
        valid,
        model_single,
        model_k3,
        model_k5,
        matrix_valid,
    )
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
fold_summary = pd.DataFrame(fold_records)
y = oof["tvt_true"].to_numpy(float)
p_single = oof["pred_single"].to_numpy(float)
p_k3 = oof["pred_k3"].to_numpy(float)
p_k5 = oof["pred_k5"].to_numpy(float)
single_rmse = _rmse(y, p_single)
k3_rmse = _rmse(y, p_k3)
k5_rmse = _rmse(y, p_k5)

bywell = []
for wid, group in oof.groupby("well", sort=False):
    wy = group["tvt_true"].to_numpy(float)
    ws = group["pred_single"].to_numpy(float)
    w3 = group["pred_k3"].to_numpy(float)
    w5 = group["pred_k5"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "single_sse": float(np.sum((ws - wy) ** 2)),
            "k3_sse": float(np.sum((w3 - wy) ** 2)),
            "k5_sse": float(np.sum((w5 - wy) ** 2)),
            "single_rmse": _rmse(wy, ws),
            "k3_rmse": _rmse(wy, w3),
            "k5_rmse": _rmse(wy, w5),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = bywell.nlargest(40, "single_sse")
top40_single = float(np.sqrt(top40["single_sse"].sum() / top40["rows"].sum()))
top40_k3 = float(np.sqrt(top40["k3_sse"].sum() / top40["rows"].sum()))
top40_k5 = float(np.sqrt(top40["k5_sse"].sum() / top40["rows"].sum()))


# ---------------------------------------------------------------------------
# Strong-base gateway: nested CP12/K5 blend, alpha selected on the other folds.
# ---------------------------------------------------------------------------
pool_path = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
if not pool_path.exists():
    raise FileNotFoundError(pool_path)
cp12 = pd.read_csv(pool_path, usecols=["id", "cp12"])
cp12["id"] = cp12["id"].astype(str)
if cp12["id"].duplicated().any():
    raise RuntimeError("candidate pool has duplicate ids")
oof = oof.merge(cp12, on="id", how="left", validate="one_to_one")
if oof["cp12"].isna().any():
    raise RuntimeError(f"Missing CP12 for {int(oof['cp12'].isna().sum())} rows")

nested_records = []
nested_prediction = np.full(len(oof), np.nan, dtype=float)
for fold in range(1, N_FOLDS + 1):
    train_mask = oof["fold"].to_numpy(int) != fold
    valid_mask = ~train_mask
    train_y = oof.loc[train_mask, "tvt_true"].to_numpy(float)
    train_cp = oof.loc[train_mask, "cp12"].to_numpy(float)
    train_k5 = oof.loc[train_mask, "pred_k5"].to_numpy(float)
    train_scores = [
        _rmse(train_y, (1.0 - alpha) * train_cp + alpha * train_k5)
        for alpha in ALPHA_GRID
    ]
    alpha = float(ALPHA_GRID[int(np.argmin(train_scores))])

    valid_y = oof.loc[valid_mask, "tvt_true"].to_numpy(float)
    valid_cp = oof.loc[valid_mask, "cp12"].to_numpy(float)
    valid_k5 = oof.loc[valid_mask, "pred_k5"].to_numpy(float)
    valid_blend = (1.0 - alpha) * valid_cp + alpha * valid_k5
    nested_prediction[valid_mask] = valid_blend
    cp_score = _rmse(valid_y, valid_cp)
    blend_score = _rmse(valid_y, valid_blend)
    nested_records.append(
        {
            "fold": fold,
            "alpha": alpha,
            "cp12_rmse": cp_score,
            "blend_rmse": blend_score,
            "gain": cp_score - blend_score,
        }
    )

oof["pred_cp12_k5_nested"] = nested_prediction
nested_folds = pd.DataFrame(nested_records)
cp12_rmse = _rmse(oof["tvt_true"], oof["cp12"])
nested_rmse = _rmse(oof["tvt_true"], oof["pred_cp12_k5_nested"])
nested_gain = cp12_rmse - nested_rmse

k5_gain_vs_single = single_rmse - k5_rmse
k5_gain_vs_k3 = k3_rmse - k5_rmse
k5_single_positive = int((fold_summary["k5_gain_vs_single"] > 0).sum())
k5_k3_positive = int((fold_summary["k5_gain_vs_k3"] > 0).sum())
nested_positive = int((nested_folds["gain"] > 0).sum())

strict_pass = bool(
    k5_gain_vs_single >= PASS_K5_VS_SINGLE
    and k5_gain_vs_k3 >= PASS_K5_VS_K3
    and k5_single_positive == N_FOLDS
    and k5_k3_positive == N_FOLDS
    and top40_k5 < top40_k3
    and nested_gain >= PASS_CP12_NESTED
    and nested_positive == N_FOLDS
    and nested_folds["gain"].min() > 0
)

result = {
    "rows": int(len(oof)),
    "wells": int(oof["well"].nunique()),
    "single_rmse": single_rmse,
    "k3_rmse": k3_rmse,
    "k5_rmse": k5_rmse,
    "k3_gain_vs_single": single_rmse - k3_rmse,
    "k5_gain_vs_single": k5_gain_vs_single,
    "k5_gain_vs_k3": k5_gain_vs_k3,
    "k5_vs_single_min_fold": float(fold_summary["k5_gain_vs_single"].min()),
    "k5_vs_k3_min_fold": float(fold_summary["k5_gain_vs_k3"].min()),
    "k5_vs_single_positive_folds": k5_single_positive,
    "k5_vs_k3_positive_folds": k5_k3_positive,
    "top40_single_rmse": top40_single,
    "top40_k3_rmse": top40_k3,
    "top40_k5_rmse": top40_k5,
    "top40_k5_gain_vs_k3": top40_k3 - top40_k5,
    "cp12_rmse": cp12_rmse,
    "cp12_k5_nested_rmse": nested_rmse,
    "cp12_k5_nested_gain": nested_gain,
    "cp12_nested_min_fold": float(nested_folds["gain"].min()),
    "cp12_nested_positive_folds": nested_positive,
    "mean_nested_alpha": float(nested_folds["alpha"].mean()),
    "feature_hours": (training_started - feature_started) / 3600.0,
    "training_hours": (time.time() - training_started) / 3600.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train K5 CatBoost/XGBoost ensemble, then CP12+Gold"
        if strict_pass
        else "FAIL -> stop pseudo-PS route; keep original 6.991 chain"
    ),
}

print("\nSINGLE / K3 / K5 FOLDS")
display(fold_summary)
print("\nCP12 NESTED BLEND FOLDS")
display(nested_folds)
print("\nOVERALL")
print(result)
print("\nVERDICT:", result["next"])

oof.to_pickle(CACHE_DIR / "k5_full_rebuild_oof.pkl")
fold_summary.to_csv(CACHE_DIR / "k5_full_rebuild_folds.csv", index=False)
nested_folds.to_csv(CACHE_DIR / "cp12_k5_nested_folds.csv", index=False)
bywell.to_csv(CACHE_DIR / "k5_full_rebuild_bywell.csv", index=False)
pd.DataFrame([result]).to_csv(CACHE_DIR / "k5_full_rebuild_summary.csv", index=False)
pd.Series(feature_columns, name="feature").to_csv(
    CACHE_DIR / "k5_full_rebuild_features.csv", index=False
)
print("saved:", CACHE_DIR)
