"""Strict outer-fold spatial input A/B on the matched K5 cut bank.

For model fold f, FI/DI are fitted only on folds != f. The same outer-fold
imputers generate features for every training and validation well used by that
model; training wells exclude themselves through ``self_wid``. Existing
PF/beam/NCC features are reused, so this cell does not rerun those engines.

Run after the original fleongg definition cells have defined:
    FormationPlaneKNN, DenseANCCImputer, seg_b_well, FORMATIONS
and after matched K5 has completed.
"""

from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
BANK_DIR = OOF_DIR / "multicut_fleongg_matched_k3_gateway_v1"
FEATURE_DIR = BANK_DIR / "features"
K5_DIR = OOF_DIR / "multicut_fleongg_matched_k5_extension_v1"
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
RUN_DIR = OOF_DIR / "multicut_k5_strict_outer_spatial_v1"
RAW_DIR = RUN_DIR / "raw_spatial"
SPATIAL_DIR = RUN_DIR / "spatial_states"
MODEL_DIR = RUN_DIR / "models"

K5_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
N_FOLDS = 5
N_JOBS = 2
RANDOM_STATE = 20260715

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

LGB_PARAMS = dict(
    objective="regression",
    boosting_type="gbdt",
    n_estimators=4000,
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

required_globals = [
    "FormationPlaneKNN",
    "DenseANCCImputer",
    "seg_b_well",
    "FORMATIONS",
]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original fleongg definition cells first; missing: "
        + repr(missing_globals)
    )

required_paths = (
    BANK_DIR / "well_metadata.csv",
    BANK_DIR / "cut_manifest.csv",
    BANK_DIR / "matched_k3_gateway_features.csv",
    K5_DIR / "new_cut_manifest.csv",
    K5_DIR / "matched_k5_oof.pkl",
    POOL_PATH,
    FEATURE_DIR,
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError("Required K5 artifacts missing: " + repr(missing_paths))

for directory in (RUN_DIR, RAW_DIR, SPATIAL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
manifest_a = pd.read_csv(BANK_DIR / "cut_manifest.csv", dtype={"well": str})
manifest_b = pd.read_csv(K5_DIR / "new_cut_manifest.csv", dtype={"well": str})
manifest = pd.concat([manifest_a, manifest_b], ignore_index=True, sort=False)
manifest = manifest[manifest["tag"].isin(K5_TAGS)].copy()
cut_lookup = {
    (str(row.well), str(row.tag)): int(row.cut_position)
    for row in manifest.itertuples(index=False)
}
missing_cuts = [
    (wid, tag)
    for wid in meta["well"].astype(str)
    for tag in K5_TAGS
    if (wid, tag) not in cut_lookup
]
if missing_cuts:
    raise RuntimeError(
        f"Incomplete K5 cut manifest: {len(missing_cuts)} missing states; "
        f"first={missing_cuts[:10]}"
    )
base_features = (
    pd.read_csv(BANK_DIR / "matched_k3_gateway_features.csv")["feature"]
    .astype(str)
    .tolist()
)
base_oof = pd.read_pickle(K5_DIR / "matched_k5_oof.pkl")
base_oof["id"] = base_oof["id"].astype(str)
base_oof["well"] = base_oof["well"].astype(str)

print(
    {
        "experiment": "multicut_k5_strict_outer_spatial_v1",
        "wells": int(meta["well"].nunique()),
        "base_features": len(base_features),
        "states": K5_TAGS,
        "spatial_jobs": N_JOBS,
        "protocol": "each model fold owns one FI/DI built only from its train folds",
    }
)


def _rmse(y_true, prediction):
    y_true = np.asarray(y_true, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y_true) ** 2)))


def _error_cosine(y_true, prediction_a, prediction_b):
    y_true = np.asarray(y_true, float)
    error_a = np.asarray(prediction_a, float) - y_true
    error_b = np.asarray(prediction_b, float) - y_true
    denominator = np.sqrt(np.dot(error_a, error_a) * np.dot(error_b, error_b))
    return float(np.dot(error_a, error_b) / denominator) if denominator else np.nan


def _simplex_weights(step=0.025):
    units = int(round(1.0 / step))
    rows = []
    for a in range(units + 1):
        for b in range(units - a + 1):
            rows.append((a / units, b / units, (units - a - b) / units))
    return np.asarray(rows, dtype=np.float64)


def _sample_path(wid, tag):
    return FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


def _official_full_path(wid):
    return FEATURE_DIR / f"{wid}__official_full.pkl"


def _state_path(fold, wid, tag):
    directory = SPATIAL_DIR / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{wid}__{tag}.pkl"


def _raw_path(fold, wid):
    directory = RAW_DIR / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{wid}.npz"


def _load_base_state(wid, tag, full=False):
    path = _official_full_path(wid) if full else _sample_path(wid, tag)
    return pd.read_pickle(path)


_OUTER_FI = None
_OUTER_DI = None


def _build_raw_spatial(fold, wid):
    path = _raw_path(fold, wid)
    if path.exists():
        return path
    source_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    source = pd.read_csv(source_path, usecols=["X", "Y"])
    xy = source[["X", "Y"]].to_numpy(np.float64)
    form_all, knn_all = _OUTER_FI.impute(xy, self_wid=wid)
    dense_all, dense_std_all, dense_dist_all = _OUTER_DI.impute(xy, self_wid=wid)
    temp_path = path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(
            temp_path,
            form=np.asarray(form_all, np.float32),
            knn=np.asarray(knn_all, np.float32),
            dense=np.asarray(dense_all, np.float32),
            dense_std=np.asarray(dense_std_all, np.float32),
            dense_dist=np.asarray(dense_dist_all, np.float32),
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def _build_raw_spatial_record(fold, wid):
    try:
        path = _build_raw_spatial(fold, wid)
        return {"well": wid, "status": "built", "path": str(path)}
    except Exception as exc:
        return {"well": wid, "status": "failed", "reason": repr(exc)}


def _derive_spatial_state(fold, wid, tag, full):
    output_path = _state_path(fold, wid, tag)
    if output_path.exists():
        return {"well": wid, "tag": tag, "status": "cached"}

    try:
        frame = _load_base_state(wid, tag, full=full).reset_index(drop=True)
        source = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__typewell.csv"
        ).sort_values("TVT")
        cut = int(cut_lookup[(wid, tag)])
        row_index = frame["_row_idx"].to_numpy(int)
        raw_path = _raw_path(fold, wid)
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing prebuilt raw spatial cache: {raw_path}")
        with np.load(raw_path) as raw:
            form_all = raw["form"].copy()
            knn_all = raw["knn"].copy()
            dense_all = raw["dense"].copy()
            dense_std_all = raw["dense_std"].copy()
            dense_dist_all = raw["dense_dist"].copy()

        ktvt = source["TVT"].to_numpy(np.float32)[:cut]
        z_all = source["Z"].to_numpy(np.float32)
        z_known = z_all[:cut]
        z_eval = z_all[row_index]
        form_known = form_all[:cut]
        form_eval = form_all[row_index]
        dense_known = dense_all[:cut]
        dense_eval = dense_all[row_index]
        dense_std_eval = dense_std_all[row_index]
        dense_dist_eval = dense_dist_all[row_index]
        knn_eval = knn_all[row_index]
        last_tvt = float(ktvt[-1])
        n_rows = len(frame)

        def scalar(value):
            return np.full(n_rows, np.float32(value), dtype=np.float32)

        tvt_features = {}
        formation_rmse = {}
        formation_paths = []
        for index, formation in enumerate(FORMATIONS):
            b_full, b_early, b_mid, b_late, b_wls = seg_b_well(
                ktvt, z_known, form_known[:, index]
            )
            tvt_f = (-z_eval + form_eval[:, index] + b_full).astype(np.float32)
            tvt_fw = (-z_eval + form_eval[:, index] + b_wls).astype(np.float32)
            tvt_f50 = (-z_eval + form_eval[:, index] + b_late).astype(np.float32)
            tvt_features[f"tvtF_{formation}"] = tvt_f
            tvt_features[f"tvtFw_{formation}"] = tvt_fw
            tvt_features[f"tvtF50_{formation}"] = tvt_f50
            tvt_features[f"bw_{formation}"] = scalar(b_full)
            tvt_features[f"bww_{formation}"] = scalar(b_wls)
            tvt_features[f"bw50_{formation}"] = scalar(b_late)
            tvt_features[f"bw_early_{formation}"] = scalar(b_early)
            tvt_features[f"bw_mid_{formation}"] = scalar(b_mid)
            known_prediction = -z_known + form_known[:, index] + b_full
            formation_rmse[formation] = float(
                np.sqrt(np.mean((ktvt - known_prediction) ** 2))
            )
            formation_paths.append(tvt_f)

        formation_stack = np.stack(formation_paths, axis=1)
        _, _, _, dense_late, dense_wls = seg_b_well(
            ktvt, z_known, dense_known
        )
        dense_bias = float(np.median(ktvt + z_known - dense_known))
        tvt_dense = (-z_eval + dense_eval + dense_bias).astype(np.float32)
        tvt_densew = (-z_eval + dense_eval + dense_wls).astype(np.float32)
        tvt_dense50 = (-z_eval + dense_eval + dense_late).astype(np.float32)
        dense_residual = ktvt + z_known - dense_known

        pf = frame["pf_ancc"].to_numpy(np.float32)
        beam_columns = [
            column
            for column in frame.columns
            if column.startswith("beam_")
            and column.endswith("_d")
            and column not in {"beam_mean_d", "beam_std_d", "beam_med_d"}
        ]
        beam_paths = [
            frame[column].to_numpy(np.float32) + np.float32(last_tvt)
            for column in beam_columns
        ]
        sc8 = frame["sc8_d"].to_numpy(np.float32) + np.float32(last_tvt)
        sc15 = frame["sc15_d"].to_numpy(np.float32) + np.float32(last_tvt)
        sc25 = frame["sc25_d"].to_numpy(np.float32) + np.float32(last_tvt)
        sc_ens = frame["sc_ens_d"].to_numpy(np.float32) + np.float32(last_tvt)
        ancc_path = tvt_features["tvtF_ANCC"]
        signal_stack = np.stack(
            [pf] + beam_paths + [sc8, sc15, sc25, sc_ens, ancc_path, tvt_dense],
            axis=1,
        )
        beam_cons = (
            frame["beam_cons_d"].to_numpy(np.float32) + np.float32(last_tvt)
        )
        tw_tvt = typewell["TVT"].to_numpy(np.float32)
        tw_gr = typewell["GR"].to_numpy(np.float32)

        features = {
            "id": frame["id"].astype(str).to_numpy(),
            **tvt_features,
            **{
                f"frm_rmse_{formation}": scalar(formation_rmse[formation])
                for formation in FORMATIONS
            },
            "form_mean_d": (
                formation_stack.mean(axis=1) - last_tvt
            ).astype(np.float32),
            "form_std_d": formation_stack.std(axis=1).astype(np.float32),
            "form_rng_d": (
                formation_stack.max(axis=1) - formation_stack.min(axis=1)
            ).astype(np.float32),
            "spatial_ancc_d": (
                form_eval[:, 0]
                - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))
            ).astype(np.float32),
            "spatial_knn_dist": knn_eval.astype(np.float32),
            "dense_ancc": dense_eval.astype(np.float32),
            "dense_std": dense_std_eval.astype(np.float32),
            "dense_dist": dense_dist_eval.astype(np.float32),
            "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
            "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
            "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
            "dense_rmse": scalar(np.sqrt(np.mean(dense_residual**2))),
            "dense_bias": scalar(np.mean(dense_residual)),
            "dense_nb_std": scalar(np.mean(dense_std_all[:cut])),
            "pf_vs_spatial": (pf - ancc_path).astype(np.float32),
            "pf_vs_dense": (pf - tvt_dense).astype(np.float32),
            "spatial_vs_dense": (ancc_path - tvt_dense).astype(np.float32),
            "beam_vs_spatial": (beam_cons - ancc_path).astype(np.float32),
            "sig_std": signal_stack.std(axis=1).astype(np.float32),
            "sig_mean_d": (
                signal_stack.mean(axis=1) - last_tvt
            ).astype(np.float32),
        }
        result = pd.DataFrame(features)
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        temp_output_path = output_path.with_suffix(".tmp.pkl")
        try:
            result.to_pickle(temp_output_path)
            temp_output_path.replace(output_path)
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink()
        return {
            "well": wid,
            "tag": tag,
            "status": "built",
            "rows": len(result),
        }
    except Exception as exc:
        return {
            "well": wid,
            "tag": tag,
            "status": "failed",
            "reason": repr(exc),
        }


feature_records = []
feature_started = time.time()
from joblib import Parallel, delayed

for fold in range(1, N_FOLDS + 1):
    reference_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_wells = set(meta.loc[meta["fold"] == fold, "well"].astype(str))
    print(
        f"\nouter spatial fold {fold}: references={len(reference_wells)} "
        f"valid={len(valid_wells)}",
        flush=True,
    )
    _OUTER_FI = FormationPlaneKNN(reference_wells, DATA_ROOT / "train")
    _OUTER_DI = DenseANCCImputer(reference_wells, DATA_ROOT / "train")

    raw_wells = [
        wid
        for wid in meta["well"].astype(str)
        if not _raw_path(fold, wid).exists()
    ]
    print("missing raw spatial wells:", len(raw_wells))
    if raw_wells:
        raw_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_build_raw_spatial_record)(fold, wid)
            for wid in tqdm(raw_wells, desc=f"raw spatial fold {fold}")
        )
        raw_failures = [row for row in raw_results if row["status"] == "failed"]
        pd.DataFrame(raw_results).to_csv(
            RUN_DIR / f"raw_build_report_fold{fold}.csv", index=False
        )
        if raw_failures:
            pd.DataFrame(raw_failures).to_csv(
                RUN_DIR / f"raw_failures_fold{fold}.csv", index=False
            )
            display(pd.DataFrame(raw_failures).head(30))
            raise RuntimeError(
                f"Raw spatial fold {fold}: {len(raw_failures)} failures"
            )

    tasks = []
    for wid in meta["well"].astype(str):
        tags = ("official",) if wid in valid_wells else K5_TAGS
        for tag in tags:
            output_path = _state_path(fold, wid, tag)
            if not output_path.exists():
                tasks.append((wid, tag, wid in valid_wells))
    print("missing spatial states:", len(tasks))
    if tasks:
        results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_derive_spatial_state)(fold, wid, tag, full)
            for wid, tag, full in tqdm(tasks, desc=f"spatial fold {fold}")
        )
        feature_records.extend(results)
        failures = [row for row in results if row["status"] == "failed"]
        if failures:
            pd.DataFrame(failures).to_csv(
                RUN_DIR / f"failures_fold{fold}.csv", index=False
            )
            display(pd.DataFrame(failures).head(30))
            raise RuntimeError(f"Spatial fold {fold}: {len(failures)} failures")
    print(
        f"spatial fold {fold} done; elapsed_hours="
        f"{(time.time()-feature_started)/3600:.2f}"
    )
    del _OUTER_FI, _OUTER_DI
    _OUTER_FI = None
    _OUTER_DI = None
    gc.collect()

pd.DataFrame(feature_records).to_csv(RUN_DIR / "feature_build_report.csv", index=False)


first_spatial = pd.read_pickle(
    _state_path(1, meta.loc[meta["fold"] != 1, "well"].astype(str).iloc[0], "official")
)
spatial_features = [column for column in first_spatial.columns if column != "id"]
all_features = base_features + [
    column for column in spatial_features if column not in base_features
]
print(
    {
        "base_features": len(base_features),
        "spatial_features": len(spatial_features),
        "all_features": len(all_features),
    }
)
if len(spatial_features) < 50:
    raise RuntimeError(f"Only {len(spatial_features)} spatial features derived")


def _attach_spatial(base, spatial_path):
    spatial = pd.read_pickle(spatial_path)
    if len(base) != len(spatial):
        raise RuntimeError(f"Row mismatch for {spatial_path}")
    if not np.array_equal(
        base["id"].astype(str).to_numpy(), spatial["id"].astype(str).to_numpy()
    ):
        raise RuntimeError(f"ID order mismatch for {spatial_path}")
    for column in spatial_features:
        base[column] = spatial[column].to_numpy()
    return base


def _load_train(fold, wells):
    keep = list(dict.fromkeys(all_features + ["target", "_weight", "id"]))
    frames = []
    pseudo_mass = 0.60 / (len(K5_TAGS) - 1)
    for wid in tqdm(wells, desc=f"load spatial train fold {fold}", leave=False):
        for tag in K5_TAGS:
            base = pd.read_pickle(_sample_path(str(wid), tag))
            base = _attach_spatial(base, _state_path(fold, str(wid), tag))
            n_eval = float(base["_official_n_eval"].iloc[0])
            mass = 0.40 if tag == "official" else pseudo_mass
            base["_weight"] = mass * n_eval / max(1, len(base))
            for column in keep:
                if column not in base:
                    base[column] = np.nan
            frames.append(base[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_valid(fold, wells):
    keep = list(
        dict.fromkeys(
            all_features
            + [
                "target",
                "well",
                "id",
                "_row_idx",
                "last_known_tvt",
                "md_since",
                "pf_ancc",
                PP_LIKPF_COL,
            ]
        )
    )
    frames = []
    for wid in tqdm(wells, desc=f"load spatial valid fold {fold}", leave=False):
        base = pd.read_pickle(_official_full_path(str(wid)))
        base = _attach_spatial(base, _state_path(fold, str(wid), "official"))
        for column in keep:
            if column not in base:
                base[column] = np.nan
        frames.append(base[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


def _matrix(frame):
    matrix = frame.reindex(columns=all_features).to_numpy(np.float32)
    matrix[np.isinf(matrix)] = np.nan
    return matrix


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(float)
    md_since = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(float) - last
    likelihood = frame[PP_LIKPF_COL].to_numpy(float)
    likelihood = np.where(np.isfinite(likelihood), likelihood, last)
    learned = PP_ALPHA * warm * (
        model_delta * (1.0 - PP_W_PF_ANCC) + pf_delta * PP_W_PF_ANCC
    )
    prediction = (
        last
        + PP_W_MODEL * learned
        + (1.0 - PP_W_MODEL) * (likelihood - last)
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


def _fit(train, valid, fold):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model_path = MODEL_DIR / f"spatial_lgb_fold{fold}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    params = dict(LGB_PARAMS)
    params["random_state"] = RANDOM_STATE + fold
    params["device_type"] = "gpu"
    model = LGBMRegressor(**params)
    kwargs = dict(
        sample_weight=train["_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(300, verbose=False), log_evaluation(100)],
    )
    try:
        model.fit(_matrix(train), train["target"].to_numpy(float), **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU unavailable; CPU fallback:", repr(exc))
        params.pop("device_type", None)
        model = LGBMRegressor(**params)
        model.fit(_matrix(train), train["target"].to_numpy(float), **kwargs)
    joblib.dump(model, model_path)
    return model


oof_parts = []
fold_rows = []
training_started = time.time()
for fold in range(1, N_FOLDS + 1):
    prediction_path = RUN_DIR / f"prediction_fold{fold}.pkl"
    if prediction_path.exists():
        oof_parts.append(pd.read_pickle(prediction_path))
        continue
    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    train = _load_train(fold, train_wells)
    valid = _load_valid(fold, valid_wells)
    model = _fit(train, valid, fold)
    prediction = _postprocess(
        valid, model.predict(_matrix(valid), num_iteration=model.best_iteration_)
    )
    truth = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)
    part = pd.DataFrame(
        {
            "id": valid["id"].astype(str).to_numpy(),
            "well": valid["well"].astype(str).to_numpy(),
            "fold": fold,
            "tvt_true_new": truth,
            "pred_spatial": prediction,
        }
    )
    base_fold = base_oof.loc[
        base_oof["fold"] == fold,
        ["id", "well", "fold", "tvt_true", "pred_k5_4000"],
    ].rename(columns={"pred_k5_4000": "pred_base"})
    part = base_fold.merge(part, on=["id", "well", "fold"], validate="one_to_one")
    if float(np.max(np.abs(part["tvt_true"] - part["tvt_true_new"]))) > 1e-3:
        raise RuntimeError(f"Truth mismatch in fold {fold}")
    part = part.drop(columns="tvt_true_new")
    part.to_pickle(prediction_path)
    oof_parts.append(part)
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(part),
            "base_rmse": _rmse(part["tvt_true"], part["pred_base"]),
            "spatial_rmse": _rmse(part["tvt_true"], part["pred_spatial"]),
            "gain": _rmse(part["tvt_true"], part["pred_base"])
            - _rmse(part["tvt_true"], part["pred_spatial"]),
            "best_iteration": int(model.best_iteration_),
        }
    )
    print("fold result:", fold_rows[-1])
    del train, valid, model
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
fold_report = []
for fold in range(1, N_FOLDS + 1):
    group = oof[oof["fold"] == fold]
    fold_report.append(
        {
            "fold": fold,
            "rows": len(group),
            "base_rmse": _rmse(group["tvt_true"], group["pred_base"]),
            "spatial_rmse": _rmse(group["tvt_true"], group["pred_spatial"]),
            "gain": _rmse(group["tvt_true"], group["pred_base"])
            - _rmse(group["tvt_true"], group["pred_spatial"]),
        }
    )
fold_report = pd.DataFrame(fold_report)
base_rmse = _rmse(oof["tvt_true"], oof["pred_base"])
spatial_rmse = _rmse(oof["tvt_true"], oof["pred_spatial"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    truth = group["tvt_true"].to_numpy(float)
    base = group["pred_base"].to_numpy(float)
    spatial = group["pred_spatial"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "base_sse": float(np.sum((base - truth) ** 2)),
            "spatial_sse": float(np.sum((spatial - truth) ** 2)),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = bywell.nlargest(40, "base_sse")
top40_base = float(np.sqrt(top40["base_sse"].sum() / top40["rows"].sum()))
top40_spatial = float(
    np.sqrt(top40["spatial_sse"].sum() / top40["rows"].sum())
)

gain = base_rmse - spatial_rmse
positive_folds = int((fold_report["gain"] > 0).sum())
architecture_pass = bool(
    gain >= 0.30 and positive_folds >= 4 and top40_spatial <= top40_base
)

# External-base check: a new base is useful only if it is competitive by
# itself or survives held-fold fusion with SP45/fleongg.
pool_columns = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
sp45_column = "sp45_like" if "sp45_like" in pool_columns else "sp45"
fleongg_column = "fleongg" if "fleongg" in pool_columns else "fle"
if sp45_column not in pool_columns or fleongg_column not in pool_columns:
    raise RuntimeError("SP45/fleongg columns missing from honest candidate pool")
context_columns = [
    column for column in ("strong_bias", "cp12") if column in pool_columns
]
read_columns = ["id", sp45_column, fleongg_column] + context_columns
pool = pd.read_csv(
    POOL_PATH,
    usecols=read_columns,
    dtype={
        "id": str,
        **{column: np.float32 for column in read_columns if column != "id"},
    },
).rename(
    columns={sp45_column: "pred_sp45", fleongg_column: "pred_fleongg"}
)
pool["id"] = pool["id"].astype(str)
external = oof.merge(pool, on="id", how="inner", validate="one_to_one")
if len(external) != len(oof):
    raise RuntimeError(f"External merge mismatch: {len(oof)} -> {len(external)}")

external_models = {
    "strict_spatial": "pred_spatial",
    "sp45": "pred_sp45",
    "fleongg": "pred_fleongg",
    **{column: column for column in context_columns},
}
external_report = pd.DataFrame(
    [
        {
            "model": name,
            "rmse": _rmse(external["tvt_true"], external[column]),
            "error_cosine_vs_spatial": _error_cosine(
                external["tvt_true"],
                external["pred_spatial"],
                external[column],
            ),
        }
        for name, column in external_models.items()
    ]
).sort_values("rmse").reset_index(drop=True)

blend_columns = ["pred_spatial", "pred_sp45", "pred_fleongg"]
weight_grid = _simplex_weights(0.025)
nested_prediction = np.full(len(external), np.nan, dtype=np.float64)
blend_fold_rows = []
fold_values = external["fold"].to_numpy(int)
for fold in range(1, N_FOLDS + 1):
    train_mask = fold_values != fold
    valid_mask = ~train_mask
    train_truth = external.loc[train_mask, "tvt_true"].to_numpy(float)
    train_predictions = external.loc[train_mask, blend_columns].to_numpy(float)
    train_errors = train_predictions - train_truth[:, None]
    gram = train_errors.T @ train_errors / len(train_errors)
    grid_mse = np.einsum("ij,jk,ik->i", weight_grid, gram, weight_grid)
    weights = weight_grid[int(np.argmin(grid_mse))]
    valid_truth = external.loc[valid_mask, "tvt_true"].to_numpy(float)
    valid_components = external.loc[valid_mask, blend_columns].to_numpy(float)
    valid_prediction = valid_components @ weights
    nested_prediction[valid_mask] = valid_prediction
    component_scores = [
        _rmse(valid_truth, valid_components[:, index]) for index in range(3)
    ]
    blend_score = _rmse(valid_truth, valid_prediction)
    blend_fold_rows.append(
        {
            "fold": fold,
            "weight_spatial": float(weights[0]),
            "weight_sp45": float(weights[1]),
            "weight_fleongg": float(weights[2]),
            "spatial_rmse": component_scores[0],
            "sp45_rmse": component_scores[1],
            "fleongg_rmse": component_scores[2],
            "blend_rmse": blend_score,
            "gain_vs_best_component": min(component_scores) - blend_score,
        }
    )
blend_folds = pd.DataFrame(blend_fold_rows)
external_truth = external["tvt_true"].to_numpy(float)
sp45_rmse = _rmse(external_truth, external["pred_sp45"])
fleongg_rmse = _rmse(external_truth, external["pred_fleongg"])
nested_blend_rmse = _rmse(external_truth, nested_prediction)
best_external_rmse = min(sp45_rmse, fleongg_rmse)
nested_gain_vs_best = best_external_rmse - nested_blend_rmse
fusion_pass = bool(
    nested_gain_vs_best >= 0.05
    and int((blend_folds["gain_vs_best_component"] > 0).sum()) >= 4
)
external_competitive = bool(spatial_rmse <= best_external_rmse)
strict_pass = bool(architecture_pass and (external_competitive or fusion_pass))
result = {
    "base_rmse": base_rmse,
    "strict_outer_spatial_rmse": spatial_rmse,
    "gain": gain,
    "mean_fold_gain": float(fold_report["gain"].mean()),
    "min_fold_gain": float(fold_report["gain"].min()),
    "positive_folds": positive_folds,
    "top40_base_rmse": top40_base,
    "top40_spatial_rmse": top40_spatial,
    "top40_gain": top40_base - top40_spatial,
    "feature_hours": (training_started - feature_started) / 3600.0,
    "training_hours": (time.time() - training_started) / 3600.0,
    "architecture_pass": architecture_pass,
    "sp45_rmse": sp45_rmse,
    "fleongg_rmse": fleongg_rmse,
    "external_competitive": external_competitive,
    "nested_external_blend_rmse": nested_blend_rmse,
    "nested_external_gain_vs_best": nested_gain_vs_best,
    "nested_external_min_fold_gain": float(
        blend_folds["gain_vs_best_component"].min()
    ),
    "nested_external_positive_folds": int(
        (blend_folds["gain_vs_best_component"] > 0).sum()
    ),
    "fusion_pass": fusion_pass,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train CatBoost/XGBoost on strict-spatial K5"
        if strict_pass
        else "strict-spatial K5 is not competitive with SP45/fleongg"
    ),
}

print("\nFOLDS")
display(fold_report)
print("\nEXTERNAL MODELS")
display(external_report)
print("\nNESTED SPATIAL / SP45 / FLEONGG")
display(blend_folds)
print("\nRESULT")
print(result)
print("\nVERDICT:", result["next"])

oof.to_pickle(RUN_DIR / "strict_outer_spatial_oof.pkl")
fold_report.to_csv(RUN_DIR / "fold_report.csv", index=False)
bywell.to_csv(RUN_DIR / "bywell.csv", index=False)
external_report.to_csv(RUN_DIR / "external_models.csv", index=False)
blend_folds.to_csv(RUN_DIR / "nested_external_blend_folds.csv", index=False)
pd.Series(all_features, name="feature").to_csv(
    RUN_DIR / "all_features.csv", index=False
)
(RUN_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", RUN_DIR)
