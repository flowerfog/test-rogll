"""Build the full-data strict-spatial K5 LightGBM deployment bundle.

For full training, FI/DI are fitted on all 773 training wells. Every training
well is queried leave-self-out, matching test deployment where all training
wells are references and the target test well is not among them. Cached K5
PF/beam/likelihood states are reused; only spatial columns are rebuilt.

Run after the original fleongg definition cells have defined:
    FormationPlaneKNN, DenseANCCImputer, seg_b_well, FORMATIONS
"""

from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
BANK_DIR = OOF_DIR / "multicut_fleongg_matched_k3_gateway_v1"
FEATURE_DIR = BANK_DIR / "features"
K5_DIR = OOF_DIR / "multicut_fleongg_matched_k5_extension_v1"
CV_RUN = OOF_DIR / "multicut_k5_strict_outer_spatial_v1"
RUN_DIR = OOF_DIR / "multicut_k5_full_spatial_lgb_bundle_v1"
RAW_DIR = RUN_DIR / "raw_spatial"
STATE_DIR = RUN_DIR / "spatial_states"
MODEL_DIR = RUN_DIR / "models"
BUNDLE_PATH = RUN_DIR / "strict_spatial_full_lgb_bundle_v1.joblib"

K5_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
N_JOBS = 2
RANDOM_STATE = 20260715

PP_CONFIG = {
    "alpha": 1.0,
    "tau": 85.0,
    "w_model": 0.60,
    "w_pf_ancc": 0.0,
    "likpf_col": "likpf_scale_5",
    "sg_win": 61,
    "sg_poly": 3,
}

LGB_PARAMS = dict(
    objective="regression",
    boosting_type="gbdt",
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
        "Run original fleongg definition cells first; missing: "
        + repr(missing_globals)
    )

required_paths = (
    BANK_DIR / "well_metadata.csv",
    BANK_DIR / "cut_manifest.csv",
    K5_DIR / "new_cut_manifest.csv",
    FEATURE_DIR,
    CV_RUN / "all_features.csv",
    CV_RUN / "models",
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError("Missing full-bundle inputs: " + repr(missing_paths))
for directory in (RUN_DIR, RAW_DIR, STATE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
manifest = pd.concat(
    [
        pd.read_csv(BANK_DIR / "cut_manifest.csv", dtype={"well": str}),
        pd.read_csv(K5_DIR / "new_cut_manifest.csv", dtype={"well": str}),
    ],
    ignore_index=True,
    sort=False,
)
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
    raise RuntimeError(f"Missing {len(missing_cuts)} K5 cuts: {missing_cuts[:10]}")

feature_columns = (
    pd.read_csv(CV_RUN / "all_features.csv")["feature"].astype(str).tolist()
)


def _sample_path(wid, tag):
    return FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


def _raw_path(wid):
    return RAW_DIR / f"{wid}.npz"


def _state_path(wid, tag):
    return STATE_DIR / f"{wid}__{tag}.pkl"


def _load_base_state(wid, tag):
    return pd.read_pickle(_sample_path(wid, tag))


print(
    {
        "experiment": "multicut_k5_full_spatial_lgb_bundle_v1",
        "wells": int(meta["well"].nunique()),
        "features": len(feature_columns),
        "states": K5_TAGS,
        "jobs": N_JOBS,
        "bundle": str(BUNDLE_PATH),
    }
)


# All training wells are deployment references. Querying with self_wid removes
# only the target well, not its whole CV fold.
all_wells = meta["well"].astype(str).tolist()
print("building full FI/DI references:", len(all_wells))
FULL_FI = FormationPlaneKNN(all_wells, DATA_ROOT / "train")
FULL_DI = DenseANCCImputer(all_wells, DATA_ROOT / "train")


def _build_raw_record(wid):
    path = _raw_path(wid)
    if path.exists():
        return {"well": wid, "status": "cached"}
    temp_path = path.with_suffix(".tmp.npz")
    try:
        source = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
            usecols=["X", "Y"],
        )
        xy = source[["X", "Y"]].to_numpy(np.float64)
        form, knn = FULL_FI.impute(xy, self_wid=wid)
        dense, dense_std, dense_dist = FULL_DI.impute(xy, self_wid=wid)
        np.savez_compressed(
            temp_path,
            form=np.asarray(form, np.float32),
            knn=np.asarray(knn, np.float32),
            dense=np.asarray(dense, np.float32),
            dense_std=np.asarray(dense_std, np.float32),
            dense_dist=np.asarray(dense_dist, np.float32),
        )
        temp_path.replace(path)
        return {"well": wid, "status": "built"}
    except Exception as exc:
        return {"well": wid, "status": "failed", "reason": repr(exc)}
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _derive_state(wid, tag):
    output_path = _state_path(wid, tag)
    if output_path.exists():
        return {"well": wid, "tag": tag, "status": "cached"}
    temp_path = output_path.with_suffix(".tmp.pkl")
    try:
        frame = _load_base_state(wid, tag).reset_index(drop=True)
        source = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__typewell.csv"
        ).sort_values("TVT")
        cut = int(cut_lookup[(wid, tag)])
        row_index = frame["_row_idx"].to_numpy(int)
        if row_index.min() < cut or row_index.max() >= len(source):
            raise RuntimeError(
                f"Bad row index range {row_index.min()}..{row_index.max()} cut={cut}"
            )
        with np.load(_raw_path(wid)) as raw:
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
        _, _, _, dense_late, dense_wls = seg_b_well(ktvt, z_known, dense_known)
        dense_residual = ktvt + z_known - dense_known
        dense_bias = float(np.median(dense_residual))
        tvt_dense = (-z_eval + dense_eval + dense_bias).astype(np.float32)
        tvt_densew = (-z_eval + dense_eval + dense_wls).astype(np.float32)
        tvt_dense50 = (-z_eval + dense_eval + dense_late).astype(np.float32)

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
            [pf]
            + beam_paths
            + [sc8, sc15, sc25, sc_ens, ancc_path, tvt_dense],
            axis=1,
        )
        beam_cons = frame["beam_cons_d"].to_numpy(np.float32) + np.float32(
            last_tvt
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
            "form_mean_d": (formation_stack.mean(axis=1) - last_tvt).astype(
                np.float32
            ),
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
            "sig_mean_d": (signal_stack.mean(axis=1) - last_tvt).astype(
                np.float32
            ),
        }
        result = pd.DataFrame(features)
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.to_pickle(temp_path)
        temp_path.replace(output_path)
        return {"well": wid, "tag": tag, "status": "built", "rows": len(result)}
    except Exception as exc:
        return {"well": wid, "tag": tag, "status": "failed", "reason": repr(exc)}
    finally:
        if temp_path.exists():
            temp_path.unlink()


feature_started = time.time()
missing_raw = [wid for wid in all_wells if not _raw_path(wid).exists()]
print("missing full raw spatial wells:", len(missing_raw))
if missing_raw:
    raw_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_build_raw_record)(wid)
        for wid in tqdm(missing_raw, desc="full raw spatial")
    )
    pd.DataFrame(raw_results).to_csv(RUN_DIR / "raw_build_report.csv", index=False)
    failures = [row for row in raw_results if row["status"] == "failed"]
    if failures:
        display(pd.DataFrame(failures).head(30))
        raise RuntimeError(f"Full raw spatial failed: {len(failures)}")

state_tasks = [
    (wid, tag)
    for wid in all_wells
    for tag in K5_TAGS
    if not _state_path(wid, tag).exists()
]
print("missing full spatial states:", len(state_tasks))
if state_tasks:
    state_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_derive_state)(wid, tag)
        for wid, tag in tqdm(state_tasks, desc="full spatial states")
    )
    pd.DataFrame(state_results).to_csv(
        RUN_DIR / "state_build_report.csv", index=False
    )
    failures = [row for row in state_results if row["status"] == "failed"]
    if failures:
        display(pd.DataFrame(failures).head(30))
        raise RuntimeError(f"Full spatial states failed: {len(failures)}")

del FULL_FI, FULL_DI
gc.collect()

schema = pd.read_pickle(_state_path(all_wells[0], "official"))
spatial_features = [column for column in schema.columns if column != "id"]
del schema
gc.collect()
if len(spatial_features) < 50:
    raise RuntimeError(f"Only {len(spatial_features)} full spatial features")


def _attach_spatial(base, path):
    spatial = pd.read_pickle(path)
    if len(base) != len(spatial):
        raise RuntimeError(f"Row mismatch: {path}")
    if not np.array_equal(
        base["id"].astype(str).to_numpy(), spatial["id"].astype(str).to_numpy()
    ):
        raise RuntimeError(f"ID mismatch: {path}")
    for column in spatial_features:
        base[column] = spatial[column].to_numpy()
    return base


def _load_full_training():
    keep = list(dict.fromkeys(feature_columns + ["target", "_weight"]))
    frames = []
    pseudo_mass = 0.60 / (len(K5_TAGS) - 1)
    for wid in tqdm(all_wells, desc="load full spatial K5 training"):
        for tag in K5_TAGS:
            frame = pd.read_pickle(_sample_path(wid, tag))
            frame = _attach_spatial(frame, _state_path(wid, tag))
            n_eval = float(frame["_official_n_eval"].iloc[0])
            mass = 0.40 if tag == "official" else pseudo_mass
            frame["_weight"] = mass * n_eval / max(1, len(frame))
            for column in keep:
                if column not in frame:
                    frame[column] = np.nan
            frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


fold_iterations = []
for fold in range(1, 6):
    model_path = CV_RUN / "models" / f"spatial_lgb_fold{fold}.joblib"
    model = joblib.load(model_path)
    iteration = getattr(model, "best_iteration_", None)
    if iteration is None or int(iteration) <= 0:
        iteration = getattr(model, "n_estimators_", None)
    if iteration is None or int(iteration) <= 0:
        raise RuntimeError(f"Cannot read best iteration: {model_path}")
    fold_iterations.append(int(iteration))
    del model
full_iterations = int(round(float(np.median(fold_iterations))))
print({"fold_best_iterations": fold_iterations, "full_iterations": full_iterations})

training_started = time.time()
train = _load_full_training()
x_train = train.reindex(columns=feature_columns).to_numpy(np.float32)
x_train[np.isinf(x_train)] = np.nan
y_train = train["target"].to_numpy(np.float32)
weights = train["_weight"].to_numpy(np.float32)
print(
    {
        "training_rows": len(train),
        "matrix_shape": x_train.shape,
        "target_rmse_zero": float(np.sqrt(np.mean(y_train.astype(float) ** 2))),
    }
)
del train
gc.collect()

from lightgbm import LGBMRegressor

params = dict(LGB_PARAMS)
params["n_estimators"] = full_iterations
params["device_type"] = "gpu"
model = LGBMRegressor(**params)
try:
    model.fit(x_train, y_train, sample_weight=weights)
except Exception as exc:
    message = str(exc).lower()
    if "gpu" not in message and "opencl" not in message:
        raise
    print("GPU unavailable; CPU fallback:", repr(exc))
    params.pop("device_type", None)
    model = LGBMRegressor(**params)
    model.fit(x_train, y_train, sample_weight=weights)

bundle = {
    "version": "strict_spatial_full_lgb_bundle_v1",
    "model": model,
    "feature_columns": feature_columns,
    "spatial_features": spatial_features,
    "k5_tags": K5_TAGS,
    "fold_best_iterations": fold_iterations,
    "full_iterations": full_iterations,
    "lgb_params": params,
    "postprocess": PP_CONFIG,
    "recommended_blend": {
        "strict_spatial": 0.45,
        "strong_bias": 0.55,
        "oof_nested_rmse": 8.775219576328515,
    },
    "training_wells": len(all_wells),
    "training_rows": len(y_train),
}
joblib.dump(bundle, BUNDLE_PATH, compress=3)

importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance_gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "importance_split": model.booster_.feature_importance(
            importance_type="split"
        ),
    }
).sort_values("importance_gain", ascending=False)
importance.to_csv(RUN_DIR / "feature_importance.csv", index=False)

summary = {
    "bundle": str(BUNDLE_PATH),
    "bundle_mb": BUNDLE_PATH.stat().st_size / 2**20,
    "features": len(feature_columns),
    "spatial_features": len(spatial_features),
    "training_wells": len(all_wells),
    "training_rows": len(y_train),
    "fold_best_iterations": fold_iterations,
    "full_iterations": full_iterations,
    "feature_hours": (training_started - feature_started) / 3600.0,
    "training_hours": (time.time() - training_started) / 3600.0,
}
print("\nTOP IMPORTANCE")
display(importance.head(40))
print("\nSUMMARY")
print(summary)
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print("saved bundle:", BUNDLE_PATH)
