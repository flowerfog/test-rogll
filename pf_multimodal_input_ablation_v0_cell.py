"""PF multimodal-input A/B on the repaired non-spatial fleongg baseline.

Run after the original tracker-definition cells so `_grid` and
`_pf_lik_allseeds` exist. Existing foldclean feature pickles are reused; the
new low-budget seed trajectories are cached independently and are deployable.
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
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BASE_DIR = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = BASE_DIR / "features"
OUT_DIR = OOF_ROOT / "pf_multimodal_input_ablation_v0_s16_p200"
PF_CACHE_DIR = OUT_DIR / "pf_seed_features"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, PF_CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260714
N_FOLDS = 5
N_SEEDS = 16
N_PARTICLES = 200
N_JOBS = 4
LIK_SCALE = 5.0
TOP_K = 5
N_MODES = 3

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

missing_globals = [
    name for name in ("_grid", "_pf_lik_allseeds") if name not in globals()
]
if missing_globals:
    raise RuntimeError(
        "Run the original tracker-definition cell first; missing: "
        + repr(missing_globals)
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


def likelihood_weights(log_lik, scale=LIK_SCALE):
    values = np.asarray(log_lik, np.float64)
    z = (values - np.max(values)) / float(scale)
    weights = np.exp(np.clip(z, -700.0, 0.0))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(len(values), 1.0 / len(values), np.float64)
    return weights / total


def weighted_quantile_rows(predictions, weights, quantiles):
    order = np.argsort(predictions, axis=0)
    sorted_prediction = np.take_along_axis(predictions, order, axis=0)
    weight_matrix = np.broadcast_to(weights[:, None], predictions.shape)
    sorted_weight = np.take_along_axis(weight_matrix, order, axis=0)
    cumulative = np.cumsum(sorted_weight, axis=0)
    columns = np.arange(predictions.shape[1])
    output = []
    for quantile in quantiles:
        positions = np.argmax(cumulative >= float(quantile), axis=0)
        output.append(sorted_prediction[positions, columns])
    return output


def trajectory_modes(predictions, weights, weighted_mean, n_modes=N_MODES):
    n_seeds, n_rows = predictions.shape
    take = np.unique(
        np.round(np.linspace(0, n_rows - 1, min(48, n_rows))).astype(int)
    )
    profile = predictions[:, take] - weighted_mean[None, take]
    profile_scale = np.maximum(np.std(profile, axis=0, keepdims=True), 2.0)
    profile = profile / profile_scale
    unique_count = len(np.unique(np.round(profile, 3), axis=0))
    k = min(int(n_modes), n_seeds, unique_count)
    if k <= 1:
        labels = np.zeros(n_seeds, dtype=int)
        k = 1
    else:
        labels = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10,
        ).fit_predict(profile)

    modes = []
    for label in range(k):
        mask = labels == label
        mode_weight = float(weights[mask].sum())
        local_weight = weights[mask] / max(mode_weight, 1e-12)
        path = np.sum(local_weight[:, None] * predictions[mask], axis=0)
        modes.append((mode_weight, path))
    modes.sort(key=lambda item: item[0], reverse=True)
    while len(modes) < n_modes:
        modes.append((0.0, weighted_mean.copy()))
    return modes[:n_modes]


def build_seed_feature_cache(wid):
    started = time.time()
    cache_path = PF_CACHE_DIR / f"{wid}.npz"
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if (
                int(cached["n_seeds"]) == N_SEEDS
                and int(cached["n_particles"]) == N_PARTICLES
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
        hw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__typewell.csv")
        tw = tw.sort_values("TVT").reset_index(drop=True)
        tw_tvt = tw["TVT"].to_numpy(float)
        tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(float)
        known = hw[hw["TVT_input"].notna()]
        evaluated = hw[hw["TVT_input"].isna()]
        if len(known) < 30 or len(evaluated) < 64:
            raise RuntimeError("too few known/eval rows")

        last = known.iloc[-1]
        last_tvt = float(last["TVT_input"])
        last_surface = last_tvt + float(last["Z"])
        type_at_known = np.interp(
            known["TVT_input"].to_numpy(float), tw_tvt, tw_gr
        )
        gr_sigma = float(
            np.clip(
                np.nanstd(
                    known["GR"].fillna(0.0).to_numpy(float) - type_at_known
                ),
                10.0,
                60.0,
            )
        )
        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float))
        dz = np.diff(tail["Z"].to_numpy(float))
        dm = np.diff(tail["MD"].to_numpy(float))
        positive = dm > 0
        initial_rate = (
            float(np.median((dt + dz)[positive] / dm[positive]))
            if positive.sum() >= 3
            else 0.0
        )
        grid_gr, grid_min, grid_step = _grid(tw_tvt, tw_gr)
        all_gr = (
            hw["GR"]
            .interpolate(limit_direction="both")
            .fillna(float(np.mean(tw_gr)))
            .to_numpy(float)
        )
        eval_pos = evaluated.index.to_numpy(int)
        observed_gr = all_gr[eval_pos]

        predictions, log_lik = _pf_lik_allseeds(
            evaluated["MD"].to_numpy(float),
            evaluated["Z"].to_numpy(float),
            observed_gr,
            grid_gr,
            grid_min,
            grid_step,
            gr_sigma,
            last_surface,
            initial_rate,
            N_PARTICLES,
            N_SEEDS,
            0,
            0.998,
            0.002,
            0.005,
            0.1,
            0.001,
            0.5,
            4.5,
        )
        predictions = np.asarray(predictions, np.float64)
        log_lik = np.asarray(log_lik, np.float64)
        weights = likelihood_weights(log_lik)
        weighted_mean = np.sum(weights[:, None] * predictions, axis=0)
        order = np.argsort(log_lik)[::-1]
        top_indices = order[: min(TOP_K, len(order))]
        quantiles = weighted_quantile_rows(
            predictions, weights, (0.10, 0.25, 0.50, 0.75, 0.90)
        )
        centered = predictions - weighted_mean[None, :]
        posterior_std = np.sqrt(
            np.sum(weights[:, None] * centered * centered, axis=0)
        )
        modes = trajectory_modes(predictions, weights, weighted_mean)

        feature_data = {
            "pfmm_weighted_mean_d": weighted_mean - last_tvt,
            "pfmm_q10_d": quantiles[0] - last_tvt,
            "pfmm_q25_d": quantiles[1] - last_tvt,
            "pfmm_q50_d": quantiles[2] - last_tvt,
            "pfmm_q75_d": quantiles[3] - last_tvt,
            "pfmm_q90_d": quantiles[4] - last_tvt,
            "pfmm_std": posterior_std,
            "pfmm_iqr": quantiles[3] - quantiles[1],
            "pfmm_q90_q10": quantiles[4] - quantiles[0],
        }
        for rank in range(TOP_K):
            if rank < len(top_indices):
                path = predictions[top_indices[rank]]
            else:
                path = weighted_mean
            feature_data[f"pfmm_top{rank + 1}_d"] = path - last_tvt
            feature_data[f"pfmm_top{rank + 1}_vs_mean"] = path - weighted_mean
        for mode_index, (mode_weight, path) in enumerate(modes, 1):
            feature_data[f"pfmm_mode{mode_index}_d"] = path - last_tvt
            feature_data[f"pfmm_mode{mode_index}_vs_mean"] = path - weighted_mean
            feature_data[f"pfmm_mode{mode_index}_weight"] = np.full(
                len(eval_pos), mode_weight
            )

        entropy = float(
            -np.sum(weights * np.log(np.maximum(weights, 1e-12)))
            / np.log(max(len(weights), 2))
        )
        ess_fraction = float(1.0 / np.sum(weights**2) / len(weights))
        sorted_weights = np.sort(weights)[::-1]
        ll_sorted = np.sort(log_lik)[::-1]
        n_eval = max(len(eval_pos), 1)
        well_scalars = {
            "pfmm_entropy": entropy,
            "pfmm_ess_fraction": ess_fraction,
            "pfmm_top1_weight": float(sorted_weights[0]),
            "pfmm_top2_weight": float(sorted_weights[1]) if len(weights) > 1 else 0.0,
            "pfmm_top12_weight": float(sorted_weights[:2].sum()),
            "pfmm_ll_gap12_per_row": (
                float((ll_sorted[0] - ll_sorted[1]) / n_eval)
                if len(ll_sorted) > 1
                else 0.0
            ),
            "pfmm_ll_std_per_row": float(np.std(log_lik) / n_eval),
            "pfmm_gr_sigma": gr_sigma,
        }
        for name, value in well_scalars.items():
            feature_data[name] = np.full(len(eval_pos), value)

        names = list(feature_data)
        matrix = np.column_stack([feature_data[name] for name in names])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.savez_compressed(
            cache_path,
            eval_pos=eval_pos.astype(np.int32),
            features=matrix.astype(np.float32),
            names=np.asarray(names),
            n_seeds=np.asarray(N_SEEDS, np.int32),
            n_particles=np.asarray(N_PARTICLES, np.int32),
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
        "experiment": "pf_multimodal_input_ablation_v0",
        "wells": len(wells),
        "n_seeds": N_SEEDS,
        "n_particles": N_PARTICLES,
        "n_jobs": N_JOBS,
    }
)

build_started = time.time()
build_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(build_seed_feature_cache)(wid)
    for wid in tqdm(wells, desc="PF multimodal cache")
)
build_report = pd.DataFrame(build_results)
build_report.to_csv(OUT_DIR / "pf_feature_build_report.csv", index=False)
failures = build_report[build_report["status"] == "failed"]
if len(failures):
    display(failures.head(30))
    raise RuntimeError(f"PF multimodal cache failed for {len(failures)} wells")
print(
    "PF feature cache complete:",
    {
        "built": int((build_report["status"] == "built").sum()),
        "cached": int((build_report["status"] == "cached").sum()),
        "elapsed_min": (time.time() - build_started) / 60.0,
    },
)

# Preload sidecars once; this avoids decompressing each well four times while
# assembling the five training folds.
pfmm_cache = {}
pfmm_feature_names = None
for wid in tqdm(wells, desc="load PF sidecars"):
    cached = np.load(PF_CACHE_DIR / f"{wid}.npz", allow_pickle=False)
    names = cached["names"].astype(str).tolist()
    if pfmm_feature_names is None:
        pfmm_feature_names = names
    elif names != pfmm_feature_names:
        raise RuntimeError(f"PF feature schema mismatch for {wid}")
    pfmm_cache[wid] = (
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


def attach_pfmm(frame):
    frame = repair_likpf(frame)
    output = np.empty((len(frame), len(pfmm_feature_names)), np.float32)
    for wid, positions in frame.groupby("well", sort=False).indices.items():
        wid = str(wid)
        positions = np.asarray(positions, np.int64)
        row_idx = frame.iloc[positions]["_row_idx"].to_numpy(np.int32)
        eval_pos, features = pfmm_cache[wid]
        locations = np.searchsorted(eval_pos, row_idx)
        if (
            np.any(locations >= len(eval_pos))
            or not np.array_equal(eval_pos[locations], row_idx)
        ):
            raise RuntimeError(f"PF sidecar row mismatch for {wid}")
        output[positions] = features[locations]
    for index, name in enumerate(pfmm_feature_names):
        frame[name] = output[:, index]
    frame["pfmm_wmean_vs_full_scale5"] = (
        frame["pfmm_weighted_mean_d"] - frame["likpf_scale_5_d"]
    ).astype(np.float32)
    return frame


def load_frames(well_ids, kind):
    path_index = 0 if kind == "sample" else 1
    frames = []
    for wid in well_ids:
        path = feature_paths(str(wid))[path_index]
        frames.append(attach_pfmm(pd.read_pickle(path)))
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


pfmm_model_features = pfmm_feature_names + ["pfmm_wmean_vs_full_scale5"]
arms = {
    "reference_fixed_likpf": reference_features,
    "pf_multimodal": reference_features + pfmm_model_features,
}
print(
    {
        "arm_sizes": {name: len(columns) for name, columns in arms.items()},
        "pf_multimodal_features": pfmm_model_features,
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
    pfmm_score = rmse(truth, predictions["pf_multimodal"])
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_wells),
            "reference_rmse": ref_score,
            "pf_multimodal_rmse": pfmm_score,
            "gain": ref_score - pfmm_score,
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
                "pf_multimodal": predictions["pf_multimodal"],
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
pfmm_rmse = rmse(oof["tvt_true"], oof["pf_multimodal"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    y = group["tvt_true"].to_numpy(float)
    ref = group["reference_fixed_likpf"].to_numpy(float)
    pred = group["pf_multimodal"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "reference_sse": float(np.sum((ref - y) ** 2)),
            "reference_rmse": rmse(y, ref),
            "pf_multimodal_rmse": rmse(y, pred),
            "gain": rmse(y, ref) - rmse(y, pred),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = set(bywell.nlargest(40, "reference_sse")["well"])
top_mask = oof["well"].isin(top40).to_numpy()
top40_gain = rmse(
    oof.loc[top_mask, "tvt_true"], oof.loc[top_mask, "reference_fixed_likpf"]
) - rmse(
    oof.loc[top_mask, "tvt_true"], oof.loc[top_mask, "pf_multimodal"]
)

result = {
    "reference_fixed_likpf_rmse": reference_rmse,
    "pf_multimodal_rmse": pfmm_rmse,
    "gain": reference_rmse - pfmm_rmse,
    "mean_fold_gain": float(fold_report["gain"].mean()),
    "min_fold_gain": float(fold_report["gain"].min()),
    "positive_folds": int((fold_report["gain"] > 0).sum()),
    "top40_gain": float(top40_gain),
    "pf_feature_minutes": (training_started - build_started) / 60.0,
    "training_hours": (time.time() - training_started) / 3600.0,
}
result["strict_pass"] = bool(
    result["gain"] >= 0.10
    and result["positive_folds"] == 5
    and result["min_fold_gain"] >= 0.02
    and result["top40_gain"] > 0.0
)
result["next"] = (
    "fit CP12-residual model with PF multimodal inputs"
    if result["strict_pass"]
    else "stop PF multimodal tabular route"
)

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nA/B FOLDS")
display(fold_report)
print("\nOVERALL")
print(result)
print("\nVERDICT:", "PASS" if result["strict_pass"] else "FAIL / STOP")

oof.to_pickle(OUT_DIR / "pf_multimodal_oof.pkl")
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
bywell.to_csv(OUT_DIR / "bywell_report.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", OUT_DIR)
