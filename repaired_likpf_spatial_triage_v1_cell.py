"""Repaired-likPF spatial A/B triage using existing foldclean feature caches.

Important: this is a cheap triage, not a final unbiased CV. Validation-well
features are fold-clean, but training-well spatial features were cross-fitted
against their own folds and can include the outer validation fold. A pass only
justifies the expensive nested outer-clean rebuild.
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
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BASE_DIR = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = BASE_DIR / "features"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "repaired_likpf_spatial_triage_v1"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260715
N_FOLDS = 5

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

LIKPF_COLUMNS = (
    "likpf_scale_3",
    "likpf_scale_5",
    "likpf_scale_8",
    "likpf_scale_12",
    "likpf_mean",
)

for path in (
    BASE_DIR / "well_metadata.csv",
    BASE_DIR / "all_features.csv",
    BASE_DIR / "no_spatial_features.csv",
    FEATURE_DIR,
    POOL_PATH,
):
    if not path.exists():
        raise FileNotFoundError(path)


def rmse(y, prediction):
    return float(
        np.sqrt(mean_squared_error(np.asarray(y, float), np.asarray(prediction, float)))
    )


def feature_paths(wid):
    return (
        FEATURE_DIR / f"{wid}__sample.pkl",
        FEATURE_DIR / f"{wid}__full.pkl",
    )


def canonicalize_feature_list(features):
    accidental = {
        f"{column}_{suffix}"
        for column in LIKPF_COLUMNS
        for suffix in ("x", "y")
    }
    output = []
    seen = set()
    for column in features:
        column = str(column)
        if column in accidental or column in seen:
            continue
        output.append(column)
        seen.add(column)
    for column in LIKPF_COLUMNS:
        for required in (column, column + "_d"):
            if required not in seen:
                output.append(required)
                seen.add(required)
    return output


def repair_likpf(frame):
    frame = frame.copy()
    last = frame["last_known_tvt"].to_numpy(np.float32)
    for column in LIKPF_COLUMNS:
        source = column + "_y" if column + "_y" in frame.columns else column
        values = (
            frame[source].to_numpy(np.float32)
            if source in frame.columns
            else last.copy()
        )
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = last[bad]
        frame[column] = values
        frame[column + "_d"] = values - last
    return frame


def load_frames(wells, kind):
    path_index = 0 if kind == "sample" else 1
    frames = []
    for wid in wells:
        path = feature_paths(str(wid))[path_index]
        frames.append(repair_likpf(pd.read_pickle(path)))
    return pd.concat(frames, ignore_index=True, sort=False)


def matrix(frame, columns):
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def new_model(fold):
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
        random_state=SEED + fold,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def fit_model(train, valid, columns, fold):
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(matrix(valid, columns), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(0)],
    )
    model = new_model(fold)
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        gpu_model = LGBMRegressor(**params)
        gpu_model.fit(
            matrix(train, columns), train["target"].to_numpy(float), **kwargs
        )
        return gpu_model
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU LightGBM unavailable; CPU fallback:", repr(exc))
        model.fit(matrix(train, columns), train["target"].to_numpy(float), **kwargs)
        return model


def postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(float)
    md_since = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(float) - last
    lik_delta = frame[PP_LIKPF_COL].to_numpy(float) - last
    learned = PP_ALPHA * warm * (
        model_delta * (1.0 - PP_W_PF_ANCC) + pf_delta * PP_W_PF_ANCC
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


meta = pd.read_csv(BASE_DIR / "well_metadata.csv")
meta["well"] = meta["well"].astype(str)
meta["fold"] = meta["fold"].astype(int)

no_spatial_features = canonicalize_feature_list(
    pd.read_csv(BASE_DIR / "no_spatial_features.csv")["feature"].tolist()
)
all_features = canonicalize_feature_list(
    pd.read_csv(BASE_DIR / "all_features.csv")["feature"].tolist()
)
spatial_only = [column for column in all_features if column not in no_spatial_features]
if len(spatial_only) < 20:
    raise RuntimeError(f"only {len(spatial_only)} spatial features detected")

arms = {
    "reference_fixed_likpf": no_spatial_features,
    "spatial_fixed_likpf": all_features,
}
print(
    {
        "experiment": "repaired_likpf_spatial_triage_v1",
        "wells": len(meta),
        "arm_sizes": {name: len(columns) for name, columns in arms.items()},
        "spatial_features": len(spatial_only),
        "warning": "triage only; training spatial X is not outer-fold clean",
    }
)

oof_parts = []
fold_rows = []
model_rows = []
started = time.time()
for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[meta["fold"] != fold, "well"].tolist()
    valid_wells = meta.loc[meta["fold"] == fold, "well"].tolist()
    print(
        f"\nfold {fold}/{N_FOLDS}: train={len(train_wells)} valid={len(valid_wells)}",
        flush=True,
    )
    train = load_frames(train_wells, "sample")
    valid = load_frames(valid_wells, "full")
    truth = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)
    predictions = {}

    for arm, columns in arms.items():
        missing = [column for column in columns if column not in train.columns]
        if missing:
            raise RuntimeError(f"{arm} missing columns: {missing[:20]}")
        model = fit_model(train, valid, columns, fold)
        delta = model.predict(matrix(valid, columns), num_iteration=model.best_iteration_)
        predictions[arm] = postprocess(valid, delta)
        joblib.dump(model, MODEL_DIR / f"{arm}_fold{fold}.joblib")
        model_rows.append(
            {
                "arm": arm,
                "fold": fold,
                "features": len(columns),
                "best_iteration": int(model.best_iteration_),
            }
        )
        del model, delta
        gc.collect()

    reference_score = rmse(truth, predictions["reference_fixed_likpf"])
    spatial_score = rmse(truth, predictions["spatial_fixed_likpf"])
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_wells),
            "reference_rmse": reference_score,
            "spatial_rmse": spatial_score,
            "gain": reference_score - spatial_score,
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
                "reference_fixed_likpf": predictions["reference_fixed_likpf"],
                "spatial_fixed_likpf": predictions["spatial_fixed_likpf"],
            }
        )
    )
    del train, valid, truth, predictions
    gc.collect()
    print(f"elapsed_hours={(time.time() - started) / 3600.0:.2f}", flush=True)

oof = pd.concat(oof_parts, ignore_index=True)
fold_report = pd.DataFrame(fold_rows)
reference_rmse = rmse(oof["tvt_true"], oof["reference_fixed_likpf"])
spatial_rmse = rmse(oof["tvt_true"], oof["spatial_fixed_likpf"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    y = group["tvt_true"].to_numpy(float)
    ref = group["reference_fixed_likpf"].to_numpy(float)
    pred = group["spatial_fixed_likpf"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "reference_sse": float(np.sum((ref - y) ** 2)),
            "reference_rmse": rmse(y, ref),
            "spatial_rmse": rmse(y, pred),
            "gain": rmse(y, ref) - rmse(y, pred),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = set(bywell.nlargest(40, "reference_sse")["well"])
top_mask = oof["well"].isin(top40).to_numpy()
top40_gain = rmse(
    oof.loc[top_mask, "tvt_true"], oof.loc[top_mask, "reference_fixed_likpf"]
) - rmse(oof.loc[top_mask, "tvt_true"], oof.loc[top_mask, "spatial_fixed_likpf"])

# Cheap strong-base triage. Alpha is selected on four folds and evaluated on the fifth.
print("\nloading CP12 for nested blend triage", flush=True)
pool = pd.read_csv(POOL_PATH, usecols=["id", "cp12"])
pool["id"] = pool["id"].astype(str)
oof = oof.merge(pool, on="id", how="left", validate="one_to_one")
if oof["cp12"].isna().any():
    raise RuntimeError(f"missing CP12 rows: {oof['cp12'].isna().sum()}")

alphas = np.arange(0.0, 0.501, 0.025)
y = oof["tvt_true"].to_numpy(float)
cp12 = oof["cp12"].to_numpy(float)
spatial = oof["spatial_fixed_likpf"].to_numpy(float)
fold_values = oof["fold"].to_numpy(int)
nested_prediction = cp12.copy()
blend_rows = []

for fold in sorted(np.unique(fold_values)):
    train_mask = fold_values != fold
    valid_mask = fold_values == fold
    train_scores = [
        rmse(
            y[train_mask],
            (1.0 - alpha) * cp12[train_mask] + alpha * spatial[train_mask],
        )
        for alpha in alphas
    ]
    alpha = float(alphas[int(np.argmin(train_scores))])
    nested_prediction[valid_mask] = (
        (1.0 - alpha) * cp12[valid_mask] + alpha * spatial[valid_mask]
    )
    base_score = rmse(y[valid_mask], cp12[valid_mask])
    blend_score = rmse(y[valid_mask], nested_prediction[valid_mask])
    blend_rows.append(
        {
            "fold": int(fold),
            "alpha": alpha,
            "cp12_rmse": base_score,
            "blend_rmse": blend_score,
            "gain": base_score - blend_score,
        }
    )

blend_report = pd.DataFrame(blend_rows)
cp12_rmse = rmse(y, cp12)
nested_blend_rmse = rmse(y, nested_prediction)
e_cp12 = cp12 - y
e_spatial = spatial - y
error_cosine = float(
    np.dot(e_cp12, e_spatial)
    / np.sqrt(np.dot(e_cp12, e_cp12) * np.dot(e_spatial, e_spatial))
)

result = {
    "warning": "triage only; spatial training features are not outer-fold clean",
    "reference_fixed_likpf_rmse": reference_rmse,
    "spatial_fixed_likpf_rmse": spatial_rmse,
    "spatial_gain_vs_reference": reference_rmse - spatial_rmse,
    "mean_fold_gain": float(fold_report["gain"].mean()),
    "min_fold_gain": float(fold_report["gain"].min()),
    "positive_folds": int((fold_report["gain"] > 0).sum()),
    "top40_gain": float(top40_gain),
    "cp12_rmse": cp12_rmse,
    "nested_blend_rmse": nested_blend_rmse,
    "nested_gain_vs_cp12": cp12_rmse - nested_blend_rmse,
    "nested_min_fold_gain": float(blend_report["gain"].min()),
    "nested_positive_folds": int((blend_report["gain"] > 0).sum()),
    "mean_nested_alpha": float(blend_report["alpha"].mean()),
    "error_cosine_vs_cp12": error_cosine,
    "runtime_hours": (time.time() - started) / 3600.0,
}
result["triage_pass"] = bool(
    result["spatial_gain_vs_reference"] >= 0.50
    and result["positive_folds"] == 5
    and result["min_fold_gain"] >= 0.20
    and result["spatial_fixed_likpf_rmse"] <= 9.50
    and result["nested_gain_vs_cp12"] >= 0.05
    and result["nested_positive_folds"] == 5
    and result["nested_min_fold_gain"] > 0.0
)
result["next"] = (
    "run fully nested outer-clean spatial rebuild"
    if result["triage_pass"]
    else "stop foldclean spatial route"
)

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nSPATIAL A/B FOLDS")
display(fold_report)
print("\nCP12 NESTED BLEND FOLDS")
display(blend_report)
print("\nOVERALL")
print(result)
print("\nVERDICT:", "TRIAGE PASS" if result["triage_pass"] else "FAIL / STOP")

oof["nested_cp12_spatial"] = nested_prediction
oof.to_pickle(OUT_DIR / "repaired_spatial_oof.pkl")
fold_report.to_csv(OUT_DIR / "spatial_fold_report.csv", index=False)
blend_report.to_csv(OUT_DIR / "cp12_nested_blend_report.csv", index=False)
bywell.to_csv(OUT_DIR / "bywell_report.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", OUT_DIR)
