"""Absolute-coordinate input A/B using the existing foldclean feature cache.

This cell does not rebuild fleongg features. It repairs the likelihood-PF column
suffixes in memory, then compares the same non-spatial LightGBM with and without
raw deployable X/Y/MD/Z and prediction-start coordinates.
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

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BASE_DIR = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = BASE_DIR / "features"
OUT_DIR = OOF_ROOT / "absolute_coordinate_ablation_v0"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260713
N_FOLDS = 5

# Keep post-processing identical to the preceding fleongg A/B.
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
COORD_COLUMNS = (
    "abs_x_kft",
    "abs_y_kft",
    "abs_md_kft",
    "abs_z_kft",
    "anchor_x_kft",
    "anchor_y_kft",
    "anchor_md_kft",
    "anchor_z_kft",
)

for path in (
    BASE_DIR / "well_metadata.csv",
    BASE_DIR / "no_spatial_features.csv",
):
    if not path.exists():
        raise FileNotFoundError(path)
if not FEATURE_DIR.exists():
    raise FileNotFoundError(FEATURE_DIR)


def rmse(y, prediction):
    return float(
        np.sqrt(mean_squared_error(np.asarray(y, float), np.asarray(prediction, float)))
    )


meta = pd.read_csv(BASE_DIR / "well_metadata.csv")
meta["well"] = meta["well"].astype(str)
meta["fold"] = meta["fold"].astype(int)

reference_features = (
    pd.read_csv(BASE_DIR / "no_spatial_features.csv")["feature"]
    .astype(str)
    .tolist()
)

# The old cache contains accidental likelihood-PF _x/_y duplicates. They are
# repaired to canonical columns while loading and excluded as model inputs.
suffix_columns = {
    f"{column}_{suffix}"
    for column in LIKPF_COLUMNS
    for suffix in ("x", "y")
}
reference_features = [
    column for column in reference_features if column not in suffix_columns
]
for column in LIKPF_COLUMNS:
    if column not in reference_features:
        reference_features.append(column)
    delta_column = column + "_d"
    if delta_column not in reference_features:
        reference_features.append(delta_column)


print("loading coordinate arrays", flush=True)
coordinate_arrays = {}
anchor_rows = []
t0 = time.time()
for i, wid in enumerate(meta["well"]):
    path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    horizontal = pd.read_csv(
        path,
        usecols=["MD", "X", "Y", "Z", "TVT_input"],
    )
    values = horizontal[["MD", "X", "Y", "Z"]].to_numpy(np.float32)
    known = np.flatnonzero(
        np.isfinite(horizontal["TVT_input"].to_numpy(float))
    )
    if not len(known):
        raise RuntimeError(f"no known TVT_input rows for {wid}")
    anchor = int(known[-1])
    coordinate_arrays[wid] = values
    anchor_rows.append(
        {
            "well": wid,
            "anchor_row": anchor,
            "anchor_md": float(values[anchor, 0]),
            "anchor_x": float(values[anchor, 1]),
            "anchor_y": float(values[anchor, 2]),
            "anchor_z": float(values[anchor, 3]),
        }
    )
    if (i + 1) % 100 == 0:
        print(f"coordinates {i + 1}/{len(meta)}", flush=True)

anchor_meta = pd.DataFrame(anchor_rows).set_index("well")
print(
    "coordinate metadata:",
    {"wells": len(anchor_meta), "elapsed_sec": time.time() - t0},
)


def feature_paths(wid):
    return (
        FEATURE_DIR / f"{wid}__sample.pkl",
        FEATURE_DIR / f"{wid}__full.pkl",
    )


def repair_likpf(frame):
    frame = frame.copy()
    last = frame["last_known_tvt"].to_numpy(np.float32)
    for column in LIKPF_COLUMNS:
        generated = column + "_y"
        if generated in frame.columns:
            values = frame[generated].to_numpy(np.float32)
        elif column in frame.columns:
            values = frame[column].to_numpy(np.float32)
        else:
            values = last.copy()
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = last[bad]
        frame[column] = values
        frame[column + "_d"] = values - last
    return frame


def add_absolute_coordinates(frame):
    frame = repair_likpf(frame)
    n = len(frame)
    current = np.empty((n, 4), dtype=np.float32)
    anchor = np.empty((n, 4), dtype=np.float32)

    for wid, positions in frame.groupby("well", sort=False).indices.items():
        wid = str(wid)
        positions = np.asarray(positions, dtype=np.int64)
        row_idx = frame.iloc[positions]["_row_idx"].to_numpy(np.int64)
        values = coordinate_arrays[wid]
        if row_idx.min() < 0 or row_idx.max() >= len(values):
            raise IndexError(f"row index outside source horizontal well: {wid}")
        current[positions] = values[row_idx]
        a = anchor_meta.loc[wid]
        anchor[positions, 0] = float(a["anchor_md"])
        anchor[positions, 1] = float(a["anchor_x"])
        anchor[positions, 2] = float(a["anchor_y"])
        anchor[positions, 3] = float(a["anchor_z"])

    # Convert large coordinate magnitudes to thousands of feet. Tree splits are
    # scale-invariant, but this keeps diagnostics and fallback models stable.
    frame["abs_md_kft"] = current[:, 0] / 1000.0
    frame["abs_x_kft"] = current[:, 1] / 1000.0
    frame["abs_y_kft"] = current[:, 2] / 1000.0
    frame["abs_z_kft"] = current[:, 3] / 1000.0
    frame["anchor_md_kft"] = anchor[:, 0] / 1000.0
    frame["anchor_x_kft"] = anchor[:, 1] / 1000.0
    frame["anchor_y_kft"] = anchor[:, 2] / 1000.0
    frame["anchor_z_kft"] = anchor[:, 3] / 1000.0
    return frame


def load_frames(wells, kind):
    frames = []
    path_index = 0 if kind == "sample" else 1
    for wid in wells:
        path = feature_paths(str(wid))[path_index]
        frames.append(add_absolute_coordinates(pd.read_pickle(path)))
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
            matrix(train, columns),
            train["target"].to_numpy(float),
            **kwargs,
        )
        return gpu_model
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU LightGBM unavailable; CPU fallback:", repr(exc))
        model.fit(
            matrix(train, columns),
            train["target"].to_numpy(float),
            **kwargs,
        )
        return model


def postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(float)
    md_since = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(float) - last
    lik_delta = frame[PP_LIKPF_COL].to_numpy(float) - last
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


arms = {
    "reference_fixed_likpf": reference_features,
    "absolute_coordinates": reference_features + list(COORD_COLUMNS),
}
print(
    {
        "experiment": "absolute_coordinate_ablation_v0",
        "wells": len(meta),
        "arm_sizes": {name: len(columns) for name, columns in arms.items()},
        "coordinate_features": list(COORD_COLUMNS),
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
    fold_predictions = {}

    for arm, columns in arms.items():
        missing = [column for column in columns if column not in train.columns]
        if missing:
            raise RuntimeError(f"{arm} missing columns: {missing[:20]}")
        model = fit_model(train, valid, columns, fold)
        delta = model.predict(
            matrix(valid, columns),
            num_iteration=model.best_iteration_,
        )
        fold_predictions[arm] = postprocess(valid, delta)
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

    reference_score = rmse(truth, fold_predictions["reference_fixed_likpf"])
    coordinate_score = rmse(truth, fold_predictions["absolute_coordinates"])
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_wells),
            "reference_rmse": reference_score,
            "coordinate_rmse": coordinate_score,
            "gain": reference_score - coordinate_score,
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
                "reference_fixed_likpf": fold_predictions["reference_fixed_likpf"],
                "absolute_coordinates": fold_predictions["absolute_coordinates"],
            }
        )
    )
    del train, valid, truth, fold_predictions
    gc.collect()
    print(f"elapsed_hours={(time.time() - started) / 3600.0:.2f}", flush=True)

oof = pd.concat(oof_parts, ignore_index=True)
fold_report = pd.DataFrame(fold_rows)
reference_rmse = rmse(oof["tvt_true"], oof["reference_fixed_likpf"])
coordinate_rmse = rmse(oof["tvt_true"], oof["absolute_coordinates"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    y = group["tvt_true"].to_numpy(float)
    ref = group["reference_fixed_likpf"].to_numpy(float)
    pred = group["absolute_coordinates"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "reference_sse": float(np.sum((ref - y) ** 2)),
            "reference_rmse": rmse(y, ref),
            "coordinate_rmse": rmse(y, pred),
            "gain": rmse(y, ref) - rmse(y, pred),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = set(bywell.nlargest(40, "reference_sse")["well"])
mask = oof["well"].isin(top40).to_numpy()
top40_gain = rmse(
    oof.loc[mask, "tvt_true"],
    oof.loc[mask, "reference_fixed_likpf"],
) - rmse(
    oof.loc[mask, "tvt_true"],
    oof.loc[mask, "absolute_coordinates"],
)

result = {
    "reference_fixed_likpf_rmse": reference_rmse,
    "absolute_coordinate_rmse": coordinate_rmse,
    "gain": reference_rmse - coordinate_rmse,
    "mean_fold_gain": float(fold_report["gain"].mean()),
    "min_fold_gain": float(fold_report["gain"].min()),
    "positive_folds": int((fold_report["gain"] > 0).sum()),
    "top40_gain": float(top40_gain),
    "runtime_hours": (time.time() - started) / 3600.0,
}
result["strict_pass"] = bool(
    result["gain"] >= 0.10
    and result["positive_folds"] == 5
    and result["min_fold_gain"] >= 0.02
    and result["top40_gain"] > 0.0
)
result["next"] = (
    "fit CP12-residual model with coordinates"
    if result["strict_pass"]
    else "stop absolute-coordinate route; build PF multimodal inputs"
)

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nA/B FOLDS")
display(fold_report)
print("\nOVERALL")
print(result)
print("\nVERDICT:", "PASS" if result["strict_pass"] else "FAIL / STOP")

oof.to_pickle(OUT_DIR / "absolute_coordinate_oof.pkl")
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
bywell.to_csv(OUT_DIR / "bywell_report.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", OUT_DIR)
