"""Multi-cut fleongg v0: honest base-model augmentation kill-test.

Run this as one notebook cell after the original fleongg notebook has defined
``build_well`` and ``lik_pf``. Do not run the original model-training cell.

The experiment compares two otherwise identical LightGBM models:

* single_cut: trained only on each training well's official prediction start;
* multi_cut: trained on the official start plus two synthetic starts.

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
CACHE_VERSION = "v0_k3_nospatial_r768_lgb1"
CACHE_DIR = OOF_DIR / f"multicut_fleongg_{CACHE_VERSION}"
FEATURE_DIR = CACHE_DIR / "features"
TEMP_DIR = CACHE_DIR / "masked_tmp"
MODEL_DIR = CACHE_DIR / "models"

N_FOLDS = 5
N_SYNTHETIC_CUTS = 2
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

PASS_OVERALL_GAIN = 0.20
PASS_MIN_FOLD_GAIN = 0.05

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
        "experiment": "multicut_fleongg_v0",
        "cache": str(CACHE_DIR),
        "cuts_per_well": 1 + N_SYNTHETIC_CUTS,
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
        # Every original well has the same total training weight in A and B.
        sample["_sample_weight_multi"] = official_n_eval / (
            (1 + N_SYNTHETIC_CUTS) * max(1, len(sample))
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
# Scan official starts and freeze two global synthetic visible-fraction cuts.
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

synthetic_fracs = meta["visible_frac"].quantile([0.25, 0.75]).to_numpy(float)
fold_map, fold_loads = _balanced_folds(meta)
meta["fold"] = meta["well"].map(fold_map).astype(int)
meta.to_csv(CACHE_DIR / "well_metadata.csv", index=False)

print("eligible wells:", len(meta))
print("synthetic visible fractions:", synthetic_fracs.tolist())
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
    required_paths = [_feature_paths(wid, tag)[0] for tag in ("official", "synth1", "synth2")]
    official_full = _feature_paths(wid, "official")[1]
    if all(path.exists() for path in required_paths) and official_full.exists():
        valid_wells.append(wid)

meta = meta[meta["well"].isin(valid_wells)].reset_index(drop=True)
print("complete wells for training:", len(meta))
if len(meta) < 700:
    raise RuntimeError("Too few complete wells after feature generation")


def _load_samples(wells, tags):
    frames = []
    for wid in wells:
        for tag in tags:
            path, _ = _feature_paths(str(wid), tag)
            if path.exists():
                frames.append(pd.read_pickle(path))
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
        "_sample_weight_multi",
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
schema_probe = _load_samples(meta["well"].astype(str).head(10), ["official", "synth1", "synth2"])
feature_columns = _feature_columns(schema_probe)
del schema_probe
print("model features:", len(feature_columns))
print(feature_columns)


# ---------------------------------------------------------------------------
# Five-fold A/B. Validation is always the untouched official cut.
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

    train_single = _load_samples(train_wells, ["official"])
    train_multi = _load_samples(train_wells, ["official", "synth1", "synth2"])
    valid = _load_official_full(valid_fold_wells)

    model_single = _fit_lgb(
        train_single,
        valid,
        feature_columns,
        "_sample_weight_single",
        "single_cut",
        fold,
    )
    model_multi = _fit_lgb(
        train_multi,
        valid,
        feature_columns,
        "_sample_weight_multi",
        "multi_cut",
        fold,
    )

    delta_single = model_single.predict(
        _matrix(valid, feature_columns), num_iteration=model_single.best_iteration_
    )
    delta_multi = model_multi.predict(
        _matrix(valid, feature_columns), num_iteration=model_multi.best_iteration_
    )
    pred_single = _postprocess(valid, delta_single)
    pred_multi = _postprocess(valid, delta_multi)
    truth = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)

    single_rmse = _rmse(truth, pred_single)
    multi_rmse = _rmse(truth, pred_multi)
    fold_records.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_fold_wells),
            "single_cut_rmse": single_rmse,
            "multi_cut_rmse": multi_rmse,
            "gain": single_rmse - multi_rmse,
            "single_best_iteration": int(model_single.best_iteration_),
            "multi_best_iteration": int(model_multi.best_iteration_),
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
                "pred_single_cut": pred_single,
                "pred_multi_cut": pred_multi,
            }
        )
    )

    del train_single, train_multi, valid, model_single, model_multi
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
fold_summary = pd.DataFrame(fold_records)
overall_single = _rmse(oof["tvt_true"], oof["pred_single_cut"])
overall_multi = _rmse(oof["tvt_true"], oof["pred_multi_cut"])
overall_gain = overall_single - overall_multi

bywell = []
for wid, group in oof.groupby("well", sort=False):
    y = group["tvt_true"].to_numpy(float)
    p0 = group["pred_single_cut"].to_numpy(float)
    p1 = group["pred_multi_cut"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "single_sse": float(np.sum((p0 - y) ** 2)),
            "multi_sse": float(np.sum((p1 - y) ** 2)),
            "single_rmse": _rmse(y, p0),
            "multi_rmse": _rmse(y, p1),
            "gain": _rmse(y, p0) - _rmse(y, p1),
        }
    )
bywell = pd.DataFrame(bywell)
bywell["single_sse_frac"] = bywell["single_sse"] / bywell["single_sse"].sum()
top40 = bywell.nlargest(40, "single_sse")
top40_single = float(np.sqrt(top40["single_sse"].sum() / top40["rows"].sum()))
top40_multi = float(np.sqrt(top40["multi_sse"].sum() / top40["rows"].sum()))
top40_gain = top40_single - top40_multi

positive_folds = int((fold_summary["gain"] > 0).sum())
min_fold_gain = float(fold_summary["gain"].min())
strict_pass = bool(
    overall_gain >= PASS_OVERALL_GAIN
    and positive_folds == N_FOLDS
    and min_fold_gain >= PASS_MIN_FOLD_GAIN
    and top40_gain >= 0.0
)

result = {
    "single_cut_rmse": overall_single,
    "multi_cut_rmse": overall_multi,
    "gain": overall_gain,
    "mean_fold_gain": float(fold_summary["gain"].mean()),
    "min_fold_gain": min_fold_gain,
    "positive_folds": positive_folds,
    "top40_single_rmse": top40_single,
    "top40_multi_rmse": top40_multi,
    "top40_gain": top40_gain,
    "strict_pass": strict_pass,
    "training_hours": (time.time() - training_started) / 3600.0,
}

print("\nFOLDS")
display(fold_summary)
print("\nOVERALL")
print(result)
print("\nVERDICT:", "PASS -> build full stack" if strict_pass else "FAIL / STOP multi-cut")

oof.to_pickle(CACHE_DIR / "multicut_v0_oof.pkl")
oof[["id", "well", "row_idx", "tvt_true", "pred_single_cut", "pred_multi_cut"]].to_csv(
    CACHE_DIR / "multicut_v0_oof.csv", index=False
)
fold_summary.to_csv(CACHE_DIR / "multicut_v0_folds.csv", index=False)
bywell.to_csv(CACHE_DIR / "multicut_v0_bywell.csv", index=False)
pd.DataFrame([result]).to_csv(CACHE_DIR / "multicut_v0_summary.csv", index=False)
pd.Series(feature_columns, name="feature").to_csv(
    CACHE_DIR / "multicut_v0_features.csv", index=False
)
print("saved:", CACHE_DIR)
