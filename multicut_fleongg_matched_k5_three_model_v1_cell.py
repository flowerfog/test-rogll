"""Matched-K5 three-model tabular base.

This cell reuses the completed matched K5 cut bank and LightGBM OOF. It trains
only CatBoost and XGBoost, validates on untouched official prediction starts,
and evaluates fixed and nested three-model ensembles. CP12/Gold are excluded.

Run on AutoDL after the matched-K5 extension has completed. No fleongg
function-definition globals are required because all features are cached.
"""

from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm.auto import tqdm


OOF_DIR = Path("/root/autodl-tmp/oof_exports")
BANK_DIR = OOF_DIR / "multicut_fleongg_matched_k3_gateway_v1"
FEATURE_DIR = BANK_DIR / "features"
K5_DIR = OOF_DIR / "multicut_fleongg_matched_k5_extension_v1"
RUN_DIR = OOF_DIR / "multicut_fleongg_matched_k5_three_model_v1"
MODEL_DIR = RUN_DIR / "models"
FOLD_PRED_DIR = RUN_DIR / "fold_predictions"

K5_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
N_FOLDS = 5
RANDOM_STATE = 20260715

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

# Fixed before seeing CatBoost/XGBoost OOF.
ENSEMBLE_WEIGHTS = {
    "equal_3": (1 / 3, 1 / 3, 1 / 3),
    "lgb_half": (0.50, 0.25, 0.25),
    "cat_xgb": (0.00, 0.50, 0.50),
    "lgb_cat": (0.50, 0.50, 0.00),
    "lgb_xgb": (0.50, 0.00, 0.50),
}
PRIMARY_ENSEMBLE = "equal_3"
PASS_PRIMARY_GAIN = 0.15

for directory in (RUN_DIR, MODEL_DIR, FOLD_PRED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

required = (
    BANK_DIR / "well_metadata.csv",
    BANK_DIR / "matched_k3_gateway_features.csv",
    K5_DIR / "matched_k5_oof.pkl",
    FEATURE_DIR,
)
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Required K5 cache is incomplete: " + repr(missing))

meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
feature_columns = (
    pd.read_csv(BANK_DIR / "matched_k3_gateway_features.csv")["feature"]
    .astype(str)
    .tolist()
)
lgb_oof = pd.read_pickle(K5_DIR / "matched_k5_oof.pkl")
lgb_oof["id"] = lgb_oof["id"].astype(str)
lgb_oof["well"] = lgb_oof["well"].astype(str)

required_lgb = {"id", "well", "fold", "tvt_true", "pred_k5_4000"}
missing_lgb = required_lgb - set(lgb_oof.columns)
if missing_lgb:
    raise RuntimeError(f"K5 OOF missing columns: {sorted(missing_lgb)}")

print(
    {
        "experiment": "matched_k5_three_model_v1",
        "wells": int(meta["well"].nunique()),
        "features": len(feature_columns),
        "k5_tags": K5_TAGS,
        "lgb_oof_rmse": float(
            np.sqrt(np.mean((lgb_oof["pred_k5_4000"] - lgb_oof["tvt_true"]) ** 2))
        ),
        "ensembles": ENSEMBLE_WEIGHTS,
        "primary": PRIMARY_ENSEMBLE,
    }
)


def _rmse(y_true, prediction):
    y_true = np.asarray(y_true, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y_true) ** 2)))


def _error_cosine(y_true, pred_a, pred_b):
    error_a = np.asarray(pred_a, float) - np.asarray(y_true, float)
    error_b = np.asarray(pred_b, float) - np.asarray(y_true, float)
    denominator = np.sqrt(np.dot(error_a, error_a) * np.dot(error_b, error_b))
    return float(np.dot(error_a, error_b) / denominator) if denominator > 0 else np.nan


def _sample_path(wid, tag):
    return FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


def _official_full_path(wid):
    return FEATURE_DIR / f"{wid}__official_full.pkl"


def _check_feature_bank():
    complete = []
    for wid in meta["well"].astype(str):
        paths = [_sample_path(wid, tag) for tag in K5_TAGS]
        paths.append(_official_full_path(wid))
        if all(path.exists() for path in paths):
            complete.append(wid)
    if len(complete) < 700:
        raise RuntimeError(f"Only {len(complete)} complete K5 wells")
    return complete


complete_wells = _check_feature_bank()
meta = meta[meta["well"].isin(complete_wells)].copy().reset_index(drop=True)
lgb_oof = lgb_oof[lgb_oof["well"].isin(complete_wells)].copy().reset_index(drop=True)
print("complete wells:", len(meta))


def _matrix(frame):
    matrix = frame.reindex(columns=feature_columns).to_numpy(np.float32)
    matrix[np.isinf(matrix)] = np.nan
    return matrix


def _load_train(wells):
    keep = list(dict.fromkeys(feature_columns + ["target", "_weight"]))
    frames = []
    pseudo_mass = 0.60 / (len(K5_TAGS) - 1)
    for wid in tqdm(wells, desc="load K5 train", leave=False):
        for tag in K5_TAGS:
            frame = pd.read_pickle(_sample_path(str(wid), tag))
            n_eval = float(frame["_official_n_eval"].iloc[0])
            mass = 0.40 if tag == "official" else pseudo_mass
            frame["_weight"] = mass * n_eval / max(1, len(frame))
            for column in keep:
                if column not in frame.columns:
                    frame[column] = np.nan
            frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_valid(wells):
    keep = list(
        dict.fromkeys(
            feature_columns
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
    for wid in tqdm(wells, desc="load official valid", leave=False):
        frame = pd.read_pickle(_official_full_path(str(wid)))
        for column in keep:
            if column not in frame.columns:
                frame[column] = np.nan
        frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


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


def _fit_catboost(x_train, y_train, weights, x_valid, y_valid, fold):
    from catboost import CatBoostRegressor, Pool

    model_path = MODEL_DIR / f"catboost_fold{fold}.cbm"
    if model_path.exists():
        model = CatBoostRegressor()
        model.load_model(model_path)
        print(f"loaded cached CatBoost fold {fold}")
        return model

    train_pool = Pool(x_train, label=y_train, weight=weights)
    valid_pool = Pool(x_valid, label=y_valid)
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=10.0,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.80,
        random_seed=RANDOM_STATE + fold,
        task_type="GPU",
        devices="0",
        od_type="Iter",
        od_wait=250,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(model_path)
    del train_pool, valid_pool
    gc.collect()
    return model


def _fit_xgboost(x_train, y_train, weights, x_valid, y_valid, fold):
    import xgboost as xgb
    from xgboost import XGBRegressor

    model_path = MODEL_DIR / f"xgboost_fold{fold}.json"
    if model_path.exists():
        model = XGBRegressor()
        model.load_model(model_path)
        print(f"loaded cached XGBoost fold {fold}")
        return model

    major = int(str(xgb.__version__).split(".")[0])
    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=3000,
        learning_rate=0.025,
        max_depth=8,
        min_child_weight=20.0,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=10.0,
        max_bin=256,
        random_state=RANDOM_STATE + 100 + fold,
        n_jobs=-1,
        early_stopping_rounds=250,
    )
    if major >= 2:
        params.update(tree_method="hist", device="cuda")
    else:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    model = XGBRegressor(**params)
    fit_kwargs = dict(
        sample_weight=weights,
        eval_set=[(x_valid, y_valid)],
        verbose=100,
    )
    if major < 2:
        # Older sklearn wrappers expect early stopping in fit rather than init.
        model.set_params(early_stopping_rounds=None)
        fit_kwargs["early_stopping_rounds"] = 250
    model.fit(x_train, y_train, **fit_kwargs)
    model.save_model(model_path)
    return model


def _blend(frame, weights):
    wl, wc, wx = weights
    return (
        wl * frame["pred_lgb"].to_numpy(float)
        + wc * frame["pred_cat"].to_numpy(float)
        + wx * frame["pred_xgb"].to_numpy(float)
    )


fold_parts = []
fold_rows = []
started = time.time()

for fold in range(1, N_FOLDS + 1):
    prediction_path = FOLD_PRED_DIR / f"fold{fold}.pkl"
    if prediction_path.exists():
        part = pd.read_pickle(prediction_path)
        fold_parts.append(part)
        print(f"loaded cached fold predictions: {fold}")
        continue

    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    print(
        f"\nthree-model fold {fold}/{N_FOLDS}: "
        f"train={len(train_wells)} valid={len(valid_wells)}",
        flush=True,
    )

    train = _load_train(train_wells)
    valid = _load_valid(valid_wells)
    x_train = _matrix(train)
    y_train = train["target"].to_numpy(np.float32)
    weights = train["_weight"].to_numpy(np.float32)
    x_valid = _matrix(valid)
    y_valid_delta = valid["target"].to_numpy(np.float32)
    del train
    gc.collect()

    cat = _fit_catboost(
        x_train, y_train, weights, x_valid, y_valid_delta, fold
    )
    cat_delta = np.asarray(cat.predict(x_valid), dtype=float)
    pred_cat = _postprocess(valid, cat_delta)
    del cat
    gc.collect()

    xgb_model = _fit_xgboost(
        x_train, y_train, weights, x_valid, y_valid_delta, fold
    )
    xgb_delta = np.asarray(xgb_model.predict(x_valid), dtype=float)
    pred_xgb = _postprocess(valid, xgb_delta)
    del xgb_model
    gc.collect()

    truth = (
        valid["target"].to_numpy(float)
        + valid["last_known_tvt"].to_numpy(float)
    )
    part = pd.DataFrame(
        {
            "id": valid["id"].astype(str).to_numpy(),
            "well": valid["well"].astype(str).to_numpy(),
            "fold": fold,
            "tvt_true_new": truth,
            "pred_cat": pred_cat,
            "pred_xgb": pred_xgb,
        }
    )
    lgb_fold = lgb_oof.loc[
        lgb_oof["fold"] == fold,
        ["id", "well", "fold", "tvt_true", "pred_k5_4000"],
    ].rename(columns={"pred_k5_4000": "pred_lgb"})
    part = lgb_fold.merge(part, on=["id", "well", "fold"], validate="one_to_one")
    if len(part) != len(valid):
        raise RuntimeError(f"Fold {fold}: LGB/new row mismatch")
    truth_diff = float(np.max(np.abs(part["tvt_true"] - part["tvt_true_new"])))
    if truth_diff > 1e-3:
        raise RuntimeError(f"Fold {fold}: truth mismatch max={truth_diff}")
    part = part.drop(columns="tvt_true_new")
    part.to_pickle(prediction_path)
    fold_parts.append(part)

    del valid, x_train, y_train, weights, x_valid, y_valid_delta
    gc.collect()
    print(f"fold {fold} saved; elapsed_hours={(time.time()-started)/3600:.2f}")


oof = pd.concat(fold_parts, ignore_index=True)
for name, weights in ENSEMBLE_WEIGHTS.items():
    oof[f"pred_{name}"] = _blend(oof, weights)

# Nested menu selection is diagnostic. The fixed equal ensemble remains the
# pre-registered primary result.
nested_prediction = np.full(len(oof), np.nan, dtype=float)
nested_rows = []
for fold in range(1, N_FOLDS + 1):
    train_mask = oof["fold"].to_numpy(int) != fold
    valid_mask = ~train_mask
    train_scores = {
        name: _rmse(
            oof.loc[train_mask, "tvt_true"],
            oof.loc[train_mask, f"pred_{name}"],
        )
        for name in ENSEMBLE_WEIGHTS
    }
    selected = min(train_scores, key=train_scores.get)
    nested_prediction[valid_mask] = oof.loc[
        valid_mask, f"pred_{selected}"
    ].to_numpy(float)
    base_score = _rmse(
        oof.loc[valid_mask, "tvt_true"], oof.loc[valid_mask, "pred_lgb"]
    )
    selected_score = _rmse(
        oof.loc[valid_mask, "tvt_true"], nested_prediction[valid_mask]
    )
    nested_rows.append(
        {
            "fold": fold,
            "selected": selected,
            "lgb_rmse": base_score,
            "selected_rmse": selected_score,
            "gain": base_score - selected_score,
        }
    )
oof["pred_nested_menu"] = nested_prediction
nested_folds = pd.DataFrame(nested_rows)

model_columns = {
    "lgb": "pred_lgb",
    "cat": "pred_cat",
    "xgb": "pred_xgb",
    **{name: f"pred_{name}" for name in ENSEMBLE_WEIGHTS},
    "nested_menu": "pred_nested_menu",
}
overall_rows = []
fold_score_rows = []
for name, column in model_columns.items():
    overall_rows.append(
        {
            "model": name,
            "rmse": _rmse(oof["tvt_true"], oof[column]),
            "gain_vs_lgb": _rmse(oof["tvt_true"], oof["pred_lgb"])
            - _rmse(oof["tvt_true"], oof[column]),
        }
    )
    for fold in range(1, N_FOLDS + 1):
        mask = oof["fold"] == fold
        base_score = _rmse(oof.loc[mask, "tvt_true"], oof.loc[mask, "pred_lgb"])
        score = _rmse(oof.loc[mask, "tvt_true"], oof.loc[mask, column])
        fold_score_rows.append(
            {
                "model": name,
                "fold": fold,
                "lgb_rmse": base_score,
                "rmse": score,
                "gain_vs_lgb": base_score - score,
            }
        )

overall = pd.DataFrame(overall_rows).sort_values("rmse").reset_index(drop=True)
fold_scores = pd.DataFrame(fold_score_rows)

bywell = []
primary_column = f"pred_{PRIMARY_ENSEMBLE}"
for wid, group in oof.groupby("well", sort=False):
    truth = group["tvt_true"].to_numpy(float)
    lgb = group["pred_lgb"].to_numpy(float)
    primary = group[primary_column].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "lgb_sse": float(np.sum((lgb - truth) ** 2)),
            "primary_sse": float(np.sum((primary - truth) ** 2)),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = bywell.nlargest(40, "lgb_sse")
top40_lgb = float(np.sqrt(top40["lgb_sse"].sum() / top40["rows"].sum()))
top40_primary = float(
    np.sqrt(top40["primary_sse"].sum() / top40["rows"].sum())
)

primary_folds = fold_scores[fold_scores["model"] == PRIMARY_ENSEMBLE]
primary_gain = float(
    overall.loc[overall["model"] == PRIMARY_ENSEMBLE, "gain_vs_lgb"].iloc[0]
)
strict_pass = bool(
    primary_gain >= PASS_PRIMARY_GAIN
    and int((primary_folds["gain_vs_lgb"] > 0).sum()) >= 4
    and top40_primary <= top40_lgb
)

cosines = pd.DataFrame(
    [
        {
            "pair": "lgb_cat",
            "error_cosine": _error_cosine(
                oof["tvt_true"], oof["pred_lgb"], oof["pred_cat"]
            ),
        },
        {
            "pair": "lgb_xgb",
            "error_cosine": _error_cosine(
                oof["tvt_true"], oof["pred_lgb"], oof["pred_xgb"]
            ),
        },
        {
            "pair": "cat_xgb",
            "error_cosine": _error_cosine(
                oof["tvt_true"], oof["pred_cat"], oof["pred_xgb"]
            ),
        },
    ]
)

result = {
    "lgb_rmse": float(overall.loc[overall["model"] == "lgb", "rmse"].iloc[0]),
    "cat_rmse": float(overall.loc[overall["model"] == "cat", "rmse"].iloc[0]),
    "xgb_rmse": float(overall.loc[overall["model"] == "xgb", "rmse"].iloc[0]),
    "equal_3_rmse": float(
        overall.loc[overall["model"] == PRIMARY_ENSEMBLE, "rmse"].iloc[0]
    ),
    "equal_3_gain_vs_lgb": primary_gain,
    "equal_3_min_fold_gain": float(primary_folds["gain_vs_lgb"].min()),
    "equal_3_positive_folds": int((primary_folds["gain_vs_lgb"] > 0).sum()),
    "top40_lgb_rmse": top40_lgb,
    "top40_equal_3_rmse": top40_primary,
    "top40_gain": top40_lgb - top40_primary,
    "nested_menu_rmse": float(
        overall.loc[overall["model"] == "nested_menu", "rmse"].iloc[0]
    ),
    "training_hours": (time.time() - started) / 3600.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> test target representation, then strict outer-clean spatial"
        if strict_pass
        else "model-family diversity is small -> move directly to new input features"
    ),
}

print("\nOVERALL MODELS")
display(overall)
print("\nFOLD SCORES")
display(fold_scores)
print("\nNESTED MENU")
display(nested_folds)
print("\nERROR COSINES")
display(cosines)
print("\nRESULT")
print(result)
print("\nVERDICT:", result["next"])

oof.to_pickle(RUN_DIR / "three_model_oof.pkl")
overall.to_csv(RUN_DIR / "overall_models.csv", index=False)
fold_scores.to_csv(RUN_DIR / "fold_scores.csv", index=False)
nested_folds.to_csv(RUN_DIR / "nested_menu_folds.csv", index=False)
cosines.to_csv(RUN_DIR / "error_cosines.csv", index=False)
bywell.to_csv(RUN_DIR / "bywell.csv", index=False)
(RUN_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", RUN_DIR)
