"""Forward NCC-profile input A/B on the repaired non-spatial fleongg baseline.

For each official eval row, score future horizontal GR windows against Typewell
GR along three existing candidate paths over a small TVT-offset grid. Only the
profile shape is added to LightGBM; no hard routing or candidate replacement is
performed. Existing foldclean feature pickles are reused.
"""

from pathlib import Path
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BASE_DIR = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = BASE_DIR / "features"
OUT_DIR = OOF_ROOT / "forward_ncc_profile_ablation_v0"
NCC_CACHE_DIR = OUT_DIR / "ncc_features"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, NCC_CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260714
N_FOLDS = 5
N_JOBS = 4
WINDOWS = (128, 512)
OFFSETS = np.arange(-32.0, 32.1, 8.0, dtype=np.float64)
CANDIDATES = ("pf_ancc", "likpf5", "beam_mean")

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


def feature_paths(wid):
    return (
        FEATURE_DIR / f"{wid}__sample.pkl",
        FEATURE_DIR / f"{wid}__full.pkl",
    )


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


def forward_ncc(observed, expected, window):
    observed = np.asarray(observed, np.float64)
    expected = np.asarray(expected, np.float64)
    n_rows = len(observed)
    valid = np.isfinite(observed) & np.isfinite(expected)
    x = np.where(valid, observed, 0.0)
    y = np.where(valid, expected, 0.0)

    arrays = [
        valid.astype(np.float64),
        x,
        y,
        x * x,
        y * y,
        x * y,
    ]
    cumulative = [np.concatenate([[0.0], np.cumsum(a)]) for a in arrays]
    start = np.arange(n_rows, dtype=np.int64)
    end = np.minimum(start + int(window), n_rows)
    sums = [values[end] - values[start] for values in cumulative]
    count, sx, sy, sxx, syy, sxy = sums
    safe_count = np.maximum(count, 1.0)
    covariance = sxy - sx * sy / safe_count
    variance_x = np.maximum(sxx - sx * sx / safe_count, 0.0)
    variance_y = np.maximum(syy - sy * sy / safe_count, 0.0)
    denominator = np.sqrt(variance_x * variance_y)
    correlation = np.zeros(n_rows, dtype=np.float64)
    enough = count >= np.maximum(12.0, 0.30 * (end - start))
    good = enough & (denominator > 1e-8)
    correlation[good] = covariance[good] / denominator[good]
    return np.clip(correlation, -1.0, 1.0)


def profile_summary(scores):
    # scores shape: n_offsets x n_rows
    scores = np.asarray(scores, np.float64)
    n_offsets, n_rows = scores.shape
    best_index = np.argmax(scores, axis=0)
    columns = np.arange(n_rows)
    peak = scores[best_index, columns]
    zero_index = int(np.argmin(np.abs(OFFSETS)))
    score_zero = scores[zero_index]
    best_offset = OFFSETS[best_index]

    far_second = np.full(n_rows, -1.0, dtype=np.float64)
    second_offset = np.zeros(n_rows, dtype=np.float64)
    for index in range(n_offsets):
        rows = np.flatnonzero(best_index == index)
        if not len(rows):
            continue
        allowed = np.flatnonzero(np.abs(OFFSETS - OFFSETS[index]) >= 16.0)
        if not len(allowed):
            allowed = np.flatnonzero(np.arange(n_offsets) != index)
        local = scores[np.ix_(allowed, rows)]
        local_best = np.argmax(local, axis=0)
        far_second[rows] = local[local_best, np.arange(len(rows))]
        second_offset[rows] = OFFSETS[allowed[local_best]]

    prominence = peak - far_second
    width = np.sum(scores >= (peak[None, :] - 0.05), axis=0).astype(np.float64)
    boundary = np.isin(best_index, (0, n_offsets - 1)).astype(np.float64)
    return {
        "best_offset": best_offset,
        "peak": peak,
        "score0": score_zero,
        "gain": peak - score_zero,
        "prominence": prominence,
        "width": width,
        "boundary": boundary,
        "second_gap": np.abs(best_offset - second_offset),
    }


def candidate_curves(frame):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    if "pf_ancc" not in frame.columns:
        raise RuntimeError("cached frame has no pf_ancc")
    if "beam_mean_d" not in frame.columns:
        raise RuntimeError("cached frame has no beam_mean_d")
    return {
        "pf_ancc": frame["pf_ancc"].to_numpy(np.float64),
        "likpf5": frame["likpf_scale_5"].to_numpy(np.float64),
        "beam_mean": last + frame["beam_mean_d"].to_numpy(np.float64),
    }


def build_ncc_cache(wid):
    started = time.time()
    output_path = NCC_CACHE_DIR / f"{wid}.npz"
    if output_path.exists():
        try:
            cached = np.load(output_path, allow_pickle=False)
            if (
                tuple(cached["windows"].astype(int)) == WINDOWS
                and np.array_equal(cached["offsets"], OFFSETS)
            ):
                return {
                    "well": wid,
                    "status": "cached",
                    "rows": len(cached["eval_pos"]),
                    "elapsed_sec": time.time() - started,
                }
        except Exception:
            pass

    try:
        _, full_path = feature_paths(wid)
        frame = repair_likpf(pd.read_pickle(full_path)).reset_index(drop=True)
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
            usecols=["GR"],
        )
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__typewell.csv",
            usecols=["TVT", "GR"],
        ).sort_values("TVT")
        tw_tvt = typewell["TVT"].to_numpy(np.float64)
        tw_gr_series = typewell["GR"].interpolate(limit_direction="both")
        tw_gr = tw_gr_series.fillna(float(tw_gr_series.mean())).to_numpy(np.float64)
        if len(tw_gr) >= 7:
            tw_gr = gaussian_filter1d(tw_gr, sigma=1.0, mode="nearest")

        eval_pos = frame["_row_idx"].to_numpy(np.int32)
        observed = horizontal["GR"].to_numpy(np.float64)[eval_pos]
        curves = candidate_curves(frame)
        feature_data = {}

        for candidate_name in CANDIDATES:
            curve = curves[candidate_name]
            window_summaries = {}
            for window in WINDOWS:
                scores = []
                for offset in OFFSETS:
                    expected = np.interp(
                        curve + float(offset),
                        tw_tvt,
                        tw_gr,
                        left=tw_gr[0],
                        right=tw_gr[-1],
                    )
                    scores.append(forward_ncc(observed, expected, window))
                summary = profile_summary(np.vstack(scores))
                window_summaries[window] = summary
                prefix = f"fncc_{candidate_name}_w{window}"
                for name, values in summary.items():
                    feature_data[f"{prefix}_{name}"] = values

            short = window_summaries[WINDOWS[0]]
            long = window_summaries[WINDOWS[1]]
            feature_data[f"fncc_{candidate_name}_offset_consistency"] = np.abs(
                short["best_offset"] - long["best_offset"]
            )
            feature_data[f"fncc_{candidate_name}_min_prominence"] = np.minimum(
                short["prominence"], long["prominence"]
            )
            feature_data[f"fncc_{candidate_name}_min_peak"] = np.minimum(
                short["peak"], long["peak"]
            )

        names = list(feature_data)
        values = np.column_stack([feature_data[name] for name in names])
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        np.savez_compressed(
            output_path,
            eval_pos=eval_pos,
            features=values.astype(np.float32),
            names=np.asarray(names),
            windows=np.asarray(WINDOWS, np.int32),
            offsets=OFFSETS,
        )
        return {
            "well": wid,
            "status": "built",
            "rows": len(eval_pos),
            "features": len(names),
            "elapsed_sec": time.time() - started,
        }
    except Exception as exc:
        return {
            "well": wid,
            "status": "failed",
            "reason": repr(exc),
            "elapsed_sec": time.time() - started,
        }


meta = pd.read_csv(BASE_DIR / "well_metadata.csv")
meta["well"] = meta["well"].astype(str)
meta["fold"] = meta["fold"].astype(int)
wells = meta["well"].tolist()

print(
    {
        "experiment": "forward_ncc_profile_ablation_v0",
        "wells": len(wells),
        "windows": WINDOWS,
        "offsets": OFFSETS.tolist(),
        "candidates": CANDIDATES,
    }
)

feature_started = time.time()
build_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(build_ncc_cache)(wid)
    for wid in tqdm(wells, desc="forward NCC profiles")
)
build_report = pd.DataFrame(build_results)
build_report.to_csv(OUT_DIR / "feature_build_report.csv", index=False)
failures = build_report[build_report["status"] == "failed"]
if len(failures):
    display(failures.head(30))
    raise RuntimeError(f"forward NCC failed for {len(failures)} wells")
print(
    "forward NCC cache complete:",
    {
        "built": int((build_report["status"] == "built").sum()),
        "cached": int((build_report["status"] == "cached").sum()),
        "elapsed_min": (time.time() - feature_started) / 60.0,
    },
)

ncc_cache = {}
ncc_feature_names = None
for wid in tqdm(wells, desc="load NCC sidecars"):
    cached = np.load(NCC_CACHE_DIR / f"{wid}.npz", allow_pickle=False)
    names = cached["names"].astype(str).tolist()
    if ncc_feature_names is None:
        ncc_feature_names = names
    elif names != ncc_feature_names:
        raise RuntimeError(f"NCC feature schema mismatch for {wid}")
    ncc_cache[wid] = (
        cached["eval_pos"].astype(np.int32),
        cached["features"].astype(np.float32),
    )


reference_features = (
    pd.read_csv(BASE_DIR / "no_spatial_features.csv")["feature"]
    .astype(str)
    .tolist()
)
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
    if column + "_d" not in reference_features:
        reference_features.append(column + "_d")


def attach_ncc(frame):
    frame = repair_likpf(frame)
    output = np.empty((len(frame), len(ncc_feature_names)), np.float32)
    for wid, positions in frame.groupby("well", sort=False).indices.items():
        wid = str(wid)
        positions = np.asarray(positions, np.int64)
        row_idx = frame.iloc[positions]["_row_idx"].to_numpy(np.int32)
        eval_pos, features = ncc_cache[wid]
        locations = np.searchsorted(eval_pos, row_idx)
        if (
            np.any(locations >= len(eval_pos))
            or not np.array_equal(eval_pos[locations], row_idx)
        ):
            raise RuntimeError(f"NCC sidecar row mismatch for {wid}")
        output[positions] = features[locations]
    for index, name in enumerate(ncc_feature_names):
        frame[name] = output[:, index]
    return frame


def load_frames(well_ids, kind):
    path_index = 0 if kind == "sample" else 1
    frames = []
    for wid in well_ids:
        path = feature_paths(str(wid))[path_index]
        frames.append(attach_ncc(pd.read_pickle(path)))
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


arms = {
    "reference_fixed_likpf": reference_features,
    "forward_ncc": reference_features + ncc_feature_names,
}
print(
    {
        "arm_sizes": {name: len(columns) for name, columns in arms.items()},
        "ncc_features": len(ncc_feature_names),
    }
)

oof_parts = []
fold_rows = []
model_rows = []
training_started = time.time()
for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[meta["fold"] != fold, "well"].tolist()
    valid_wells = meta.loc[meta["fold"] == fold, "well"].tolist()
    print(
        f"\nmodel fold {fold}/{N_FOLDS}: train={len(train_wells)} valid={len(valid_wells)}",
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

    ref_score = rmse(truth, predictions["reference_fixed_likpf"])
    ncc_score = rmse(truth, predictions["forward_ncc"])
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_wells),
            "reference_rmse": ref_score,
            "forward_ncc_rmse": ncc_score,
            "gain": ref_score - ncc_score,
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
                "forward_ncc": predictions["forward_ncc"],
            }
        )
    )
    del train, valid, truth, predictions
    gc.collect()
    print(
        f"model elapsed_hours={(time.time() - training_started) / 3600.0:.2f}",
        flush=True,
    )

oof = pd.concat(oof_parts, ignore_index=True)
fold_report = pd.DataFrame(fold_rows)
reference_rmse = rmse(oof["tvt_true"], oof["reference_fixed_likpf"])
ncc_rmse = rmse(oof["tvt_true"], oof["forward_ncc"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    y = group["tvt_true"].to_numpy(float)
    ref = group["reference_fixed_likpf"].to_numpy(float)
    pred = group["forward_ncc"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "reference_sse": float(np.sum((ref - y) ** 2)),
            "reference_rmse": rmse(y, ref),
            "forward_ncc_rmse": rmse(y, pred),
            "gain": rmse(y, ref) - rmse(y, pred),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = set(bywell.nlargest(40, "reference_sse")["well"])
mask = oof["well"].isin(top40).to_numpy()
top40_gain = rmse(
    oof.loc[mask, "tvt_true"], oof.loc[mask, "reference_fixed_likpf"]
) - rmse(oof.loc[mask, "tvt_true"], oof.loc[mask, "forward_ncc"])

result = {
    "reference_fixed_likpf_rmse": reference_rmse,
    "forward_ncc_rmse": ncc_rmse,
    "gain": reference_rmse - ncc_rmse,
    "mean_fold_gain": float(fold_report["gain"].mean()),
    "min_fold_gain": float(fold_report["gain"].min()),
    "positive_folds": int((fold_report["gain"] > 0).sum()),
    "top40_gain": float(top40_gain),
    "feature_minutes": (training_started - feature_started) / 60.0,
    "training_hours": (time.time() - training_started) / 3600.0,
}
result["strict_pass"] = bool(
    result["gain"] >= 0.10
    and result["positive_folds"] == 5
    and result["min_fold_gain"] >= 0.02
    and result["top40_gain"] > 0.0
)
result["next"] = (
    "fit CP12-residual model with forward-NCC inputs"
    if result["strict_pass"]
    else "stop forward-NCC input route"
)

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nA/B FOLDS")
display(fold_report)
print("\nOVERALL")
print(result)
print("\nVERDICT:", "PASS" if result["strict_pass"] else "FAIL / STOP")

oof.to_pickle(OUT_DIR / "forward_ncc_oof.pkl")
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
bywell.to_csv(OUT_DIR / "bywell_report.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", OUT_DIR)
