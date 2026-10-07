"""Train CatBoost/XGBoost on the cached strict-spatial K5 feature bank.

The strict-spatial LightGBM OOF is frozen as the baseline. This cell reuses
all cached K5 and outer-fold spatial states, trains only CatBoost/XGBoost, and
then evaluates both internal ensembles and four-fold-selected blends with
strong_bias. No FI/DI, PF, beam, or likelihood features are rebuilt.
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
SPATIAL_RUN = OOF_DIR / "multicut_k5_strict_outer_spatial_v1"
SPATIAL_STATE_DIR = SPATIAL_RUN / "spatial_states"
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
RUN_DIR = OOF_DIR / "multicut_k5_strict_spatial_three_model_v1"
MODEL_DIR = RUN_DIR / "models"
FOLD_PRED_DIR = RUN_DIR / "fold_predictions"

K5_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
N_FOLDS = 5
RANDOM_STATE = 20260715
PAIR_STEP = 0.025

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

# Fixed before seeing strict-spatial CatBoost/XGBoost OOF.
ENSEMBLE_WEIGHTS = {
    "equal3": (1 / 3, 1 / 3, 1 / 3),
    "lgb_half": (0.50, 0.25, 0.25),
    "lgb_cat": (0.50, 0.50, 0.00),
    "lgb_xgb": (0.50, 0.00, 0.50),
    "cat_xgb": (0.00, 0.50, 0.50),
}
PRIMARY_ENSEMBLE = "equal3"

for directory in (RUN_DIR, MODEL_DIR, FOLD_PRED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

required_paths = (
    BANK_DIR / "well_metadata.csv",
    FEATURE_DIR,
    SPATIAL_RUN / "all_features.csv",
    SPATIAL_RUN / "strict_outer_spatial_oof.pkl",
    SPATIAL_STATE_DIR,
    POOL_PATH,
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError("Missing strict-spatial artifacts: " + repr(missing_paths))

meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
feature_columns = (
    pd.read_csv(SPATIAL_RUN / "all_features.csv")["feature"].astype(str).tolist()
)
lgb_oof = pd.read_pickle(SPATIAL_RUN / "strict_outer_spatial_oof.pkl")
lgb_oof["id"] = lgb_oof["id"].astype(str)
lgb_oof["well"] = lgb_oof["well"].astype(str)
required_lgb = {"id", "well", "fold", "tvt_true", "pred_spatial"}
missing_lgb = required_lgb - set(lgb_oof.columns)
if missing_lgb:
    raise RuntimeError(f"Strict-spatial OOF missing: {sorted(missing_lgb)}")


def _sample_path(wid, tag):
    return FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


def _official_full_path(wid):
    return FEATURE_DIR / f"{wid}__official_full.pkl"


def _spatial_path(fold, wid, tag):
    return SPATIAL_STATE_DIR / f"fold{fold}" / f"{wid}__{tag}.pkl"


schema_well = meta.loc[meta["fold"] != 1, "well"].astype(str).iloc[0]
schema_spatial = pd.read_pickle(_spatial_path(1, schema_well, "official"))
spatial_features = [column for column in schema_spatial.columns if column != "id"]
del schema_spatial
gc.collect()
if len(spatial_features) < 50:
    raise RuntimeError(f"Only {len(spatial_features)} strict spatial features")

print(
    {
        "experiment": "multicut_k5_strict_spatial_three_model_v1",
        "wells": int(meta["well"].nunique()),
        "features": len(feature_columns),
        "spatial_features": len(spatial_features),
        "ensembles": ENSEMBLE_WEIGHTS,
        "primary": PRIMARY_ENSEMBLE,
    }
)


def _rmse(y_true, prediction):
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - y_true) ** 2)))


def _error_cosine(y_true, prediction_a, prediction_b):
    y_true = np.asarray(y_true, dtype=np.float64)
    error_a = np.asarray(prediction_a, dtype=np.float64) - y_true
    error_b = np.asarray(prediction_b, dtype=np.float64) - y_true
    denominator = np.sqrt(np.dot(error_a, error_a) * np.dot(error_b, error_b))
    return float(np.dot(error_a, error_b) / denominator) if denominator else np.nan


def _attach_spatial(base, spatial_path):
    spatial = pd.read_pickle(spatial_path)
    if len(base) != len(spatial):
        raise RuntimeError(f"Row mismatch: {spatial_path}")
    if not np.array_equal(
        base["id"].astype(str).to_numpy(), spatial["id"].astype(str).to_numpy()
    ):
        raise RuntimeError(f"ID mismatch: {spatial_path}")
    for column in spatial_features:
        base[column] = spatial[column].to_numpy()
    return base


def _load_train(fold, wells):
    keep = list(dict.fromkeys(feature_columns + ["target", "_weight"]))
    frames = []
    pseudo_mass = 0.60 / (len(K5_TAGS) - 1)
    for wid in tqdm(wells, desc=f"load strict-spatial train fold {fold}", leave=False):
        for tag in K5_TAGS:
            frame = pd.read_pickle(_sample_path(str(wid), tag))
            frame = _attach_spatial(frame, _spatial_path(fold, str(wid), tag))
            n_eval = float(frame["_official_n_eval"].iloc[0])
            mass = 0.40 if tag == "official" else pseudo_mass
            frame["_weight"] = mass * n_eval / max(1, len(frame))
            for column in keep:
                if column not in frame:
                    frame[column] = np.nan
            frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_valid(fold, wells):
    keep = list(
        dict.fromkeys(
            feature_columns
            + [
                "target",
                "well",
                "id",
                "last_known_tvt",
                "md_since",
                "pf_ancc",
                PP_LIKPF_COL,
            ]
        )
    )
    frames = []
    for wid in tqdm(wells, desc=f"load strict-spatial valid fold {fold}", leave=False):
        frame = pd.read_pickle(_official_full_path(str(wid)))
        frame = _attach_spatial(frame, _spatial_path(fold, str(wid), "official"))
        for column in keep:
            if column not in frame:
                frame[column] = np.nan
        frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


def _matrix(frame):
    matrix = frame.reindex(columns=feature_columns).to_numpy(np.float32)
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
    prediction = last + PP_W_MODEL * learned + (1.0 - PP_W_MODEL) * (
        likelihood - last
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
started = time.time()
for fold in range(1, N_FOLDS + 1):
    prediction_path = FOLD_PRED_DIR / f"fold{fold}.pkl"
    if prediction_path.exists():
        fold_parts.append(pd.read_pickle(prediction_path))
        print(f"loaded cached fold predictions: {fold}")
        continue
    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    print(f"\nstrict-spatial three-model fold {fold}/5", flush=True)
    train = _load_train(fold, train_wells)
    valid = _load_valid(fold, valid_wells)
    x_train = _matrix(train)
    y_train = train["target"].to_numpy(np.float32)
    weights = train["_weight"].to_numpy(np.float32)
    x_valid = _matrix(valid)
    y_valid = valid["target"].to_numpy(np.float32)
    del train
    gc.collect()

    cat = _fit_catboost(x_train, y_train, weights, x_valid, y_valid, fold)
    pred_cat = _postprocess(valid, np.asarray(cat.predict(x_valid), dtype=float))
    del cat
    gc.collect()
    xgb = _fit_xgboost(x_train, y_train, weights, x_valid, y_valid, fold)
    pred_xgb = _postprocess(valid, np.asarray(xgb.predict(x_valid), dtype=float))
    del xgb
    gc.collect()

    truth_new = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)
    new = pd.DataFrame(
        {
            "id": valid["id"].astype(str).to_numpy(),
            "well": valid["well"].astype(str).to_numpy(),
            "fold": fold,
            "tvt_true_new": truth_new,
            "pred_cat": pred_cat,
            "pred_xgb": pred_xgb,
        }
    )
    lgb_fold = lgb_oof.loc[
        lgb_oof["fold"] == fold,
        ["id", "well", "fold", "tvt_true", "pred_spatial"],
    ].rename(columns={"pred_spatial": "pred_lgb"})
    part = lgb_fold.merge(new, on=["id", "well", "fold"], validate="one_to_one")
    truth_difference = float(
        np.max(np.abs(part["tvt_true"] - part["tvt_true_new"]))
    )
    if truth_difference > 1e-3:
        raise RuntimeError(f"Fold {fold}: truth mismatch {truth_difference}")
    part = part.drop(columns="tvt_true_new")
    part.to_pickle(prediction_path)
    fold_parts.append(part)
    del valid, x_train, y_train, weights, x_valid, y_valid
    gc.collect()
    print(f"fold {fold} saved; elapsed_hours={(time.time()-started)/3600:.2f}")


oof = pd.concat(fold_parts, ignore_index=True)
for name, weights in ENSEMBLE_WEIGHTS.items():
    oof[f"pred_{name}"] = _blend(oof, weights)

pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "strong_bias"],
    dtype={"id": str, "strong_bias": np.float32},
)
pool["id"] = pool["id"].astype(str)
oof = oof.merge(pool, on="id", how="inner", validate="one_to_one")
if len(oof) != len(lgb_oof):
    raise RuntimeError(f"Strong-bias merge mismatch: {len(oof)} vs {len(lgb_oof)}")

candidate_columns = {
    "lgb": "pred_lgb",
    "cat": "pred_cat",
    "xgb": "pred_xgb",
    **{name: f"pred_{name}" for name in ENSEMBLE_WEIGHTS},
}
truth = oof["tvt_true"].to_numpy(float)
standalone = pd.DataFrame(
    [
        {
            "model": name,
            "rmse": _rmse(truth, oof[column]),
            "cosine_vs_strong_bias": _error_cosine(
                truth, oof[column], oof["strong_bias"]
            ),
        }
        for name, column in candidate_columns.items()
    ]
).sort_values("rmse").reset_index(drop=True)

# For each fixed spatial model, select only its strong-bias blend weight on
# four folds. A separate nested menu jointly selects the model and its weight.
pair_weights = np.arange(0.0, 1.0 + PAIR_STEP / 2.0, PAIR_STEP)
fold_values = oof["fold"].to_numpy(int)
nested_predictions = {
    name: np.full(len(oof), np.nan, dtype=np.float64) for name in candidate_columns
}
menu_prediction = np.full(len(oof), np.nan, dtype=np.float64)
blend_fold_rows = []
menu_rows = []
for fold in range(1, N_FOLDS + 1):
    train_mask = fold_values != fold
    valid_mask = ~train_mask
    train_truth = oof.loc[train_mask, "tvt_true"].to_numpy(float)
    train_strong = oof.loc[train_mask, "strong_bias"].to_numpy(float)
    valid_truth = oof.loc[valid_mask, "tvt_true"].to_numpy(float)
    valid_strong = oof.loc[valid_mask, "strong_bias"].to_numpy(float)
    choices = []
    for name, column in candidate_columns.items():
        train_model = oof.loc[train_mask, column].to_numpy(float)
        train_error_model = train_model - train_truth
        train_error_strong = train_strong - train_truth
        a = float(np.mean(train_error_model**2))
        b = float(np.mean(train_error_strong**2))
        c = float(np.mean(train_error_model * train_error_strong))
        mse_grid = (
            pair_weights**2 * a
            + (1.0 - pair_weights) ** 2 * b
            + 2.0 * pair_weights * (1.0 - pair_weights) * c
        )
        index = int(np.argmin(mse_grid))
        weight_model = float(pair_weights[index])
        valid_model = oof.loc[valid_mask, column].to_numpy(float)
        valid_prediction = weight_model * valid_model + (1.0 - weight_model) * valid_strong
        nested_predictions[name][valid_mask] = valid_prediction
        valid_score = _rmse(valid_truth, valid_prediction)
        choices.append((float(mse_grid[index]), name, weight_model, valid_prediction))
        blend_fold_rows.append(
            {
                "model": name,
                "fold": fold,
                "weight_model": weight_model,
                "weight_strong_bias": 1.0 - weight_model,
                "train_blend_rmse": float(np.sqrt(mse_grid[index])),
                "model_rmse": _rmse(valid_truth, valid_model),
                "strong_bias_rmse": _rmse(valid_truth, valid_strong),
                "blend_rmse": valid_score,
            }
        )
    selected = min(choices, key=lambda row: row[0])
    menu_prediction[valid_mask] = selected[3]
    menu_rows.append(
        {
            "fold": fold,
            "selected_model": selected[1],
            "weight_model": selected[2],
            "weight_strong_bias": 1.0 - selected[2],
            "train_rmse": float(np.sqrt(selected[0])),
            "valid_rmse": _rmse(valid_truth, selected[3]),
        }
    )

blend_folds = pd.DataFrame(blend_fold_rows)
menu_folds = pd.DataFrame(menu_rows)
blend_summary_rows = []
for name, prediction in nested_predictions.items():
    folds = blend_folds[blend_folds["model"] == name]
    score = _rmse(truth, prediction)
    blend_summary_rows.append(
        {
            "model": name,
            "nested_strong_rmse": score,
            "gain_vs_strong_bias": _rmse(truth, oof["strong_bias"]) - score,
            "mean_weight_model": float(folds["weight_model"].mean()),
            "min_weight_model": float(folds["weight_model"].min()),
            "max_weight_model": float(folds["weight_model"].max()),
        }
    )
blend_summary = pd.DataFrame(blend_summary_rows).sort_values("nested_strong_rmse").reset_index(drop=True)

baseline_prediction = nested_predictions["lgb"]
baseline_fold_scores = {}
for fold in range(1, N_FOLDS + 1):
    mask = fold_values == fold
    baseline_fold_scores[fold] = _rmse(truth[mask], baseline_prediction[mask])

comparison_rows = []
for name, prediction in {**nested_predictions, "nested_menu": menu_prediction}.items():
    fold_gains = []
    for fold in range(1, N_FOLDS + 1):
        mask = fold_values == fold
        fold_gains.append(
            baseline_fold_scores[fold] - _rmse(truth[mask], prediction[mask])
        )
    comparison_rows.append(
        {
            "model": name,
            "rmse": _rmse(truth, prediction),
            "gain_vs_lgb_strong": _rmse(truth, baseline_prediction)
            - _rmse(truth, prediction),
            "mean_fold_gain": float(np.mean(fold_gains)),
            "min_fold_gain": float(np.min(fold_gains)),
            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
        }
    )
comparison = pd.DataFrame(comparison_rows).sort_values("rmse").reset_index(drop=True)

# Tail guard: compare candidates on the same 40 wells with the largest
# strong_bias SSE. A model-family gain must not come from sacrificing them.
bywell_rows = []
for wid, index_labels in oof.groupby("well", sort=False).groups.items():
    index = np.asarray(list(index_labels), dtype=int)
    row_truth = truth[index]
    strong_error = oof["strong_bias"].to_numpy(float)[index] - row_truth
    baseline_error = baseline_prediction[index] - row_truth
    primary_error = nested_predictions[PRIMARY_ENSEMBLE][index] - row_truth
    menu_error = menu_prediction[index] - row_truth
    bywell_rows.append(
        {
            "well": str(wid),
            "rows": len(index),
            "strong_sse": float(np.dot(strong_error, strong_error)),
            "baseline_sse": float(np.dot(baseline_error, baseline_error)),
            "primary_sse": float(np.dot(primary_error, primary_error)),
            "menu_sse": float(np.dot(menu_error, menu_error)),
        }
    )
bywell = pd.DataFrame(bywell_rows)
top40 = bywell.nlargest(40, "strong_sse")
top40_rows = float(top40["rows"].sum())
top40_scores = {
    "lgb_strong": float(np.sqrt(top40["baseline_sse"].sum() / top40_rows)),
    "equal3_strong": float(np.sqrt(top40["primary_sse"].sum() / top40_rows)),
    "nested_menu": float(np.sqrt(top40["menu_sse"].sum() / top40_rows)),
}

primary = comparison[comparison["model"] == PRIMARY_ENSEMBLE].iloc[0]
menu_result = comparison[comparison["model"] == "nested_menu"].iloc[0]
primary_pass = bool(
    primary["gain_vs_lgb_strong"] >= 0.02
    and primary["positive_folds"] >= 4
    and top40_scores["equal3_strong"] <= top40_scores["lgb_strong"]
)
menu_pass = bool(
    menu_result["gain_vs_lgb_strong"] >= 0.03
    and menu_result["positive_folds"] >= 4
    and top40_scores["nested_menu"] <= top40_scores["lgb_strong"]
)
strict_pass = bool(primary_pass or menu_pass)
if primary_pass and menu_pass:
    chosen = (
        PRIMARY_ENSEMBLE
        if primary["gain_vs_lgb_strong"] >= menu_result["gain_vs_lgb_strong"]
        else "nested_menu"
    )
elif primary_pass:
    chosen = PRIMARY_ENSEMBLE
elif menu_pass:
    chosen = "nested_menu"
else:
    chosen = "lgb"
result = {
    "spatial_lgb_rmse": _rmse(truth, oof["pred_lgb"]),
    "spatial_equal3_rmse": _rmse(truth, oof[f"pred_{PRIMARY_ENSEMBLE}"]),
    "lgb_strong_nested_rmse": _rmse(truth, baseline_prediction),
    "equal3_strong_nested_rmse": float(primary["rmse"]),
    "equal3_gain_vs_lgb_strong": float(primary["gain_vs_lgb_strong"]),
    "equal3_min_fold_gain": float(primary["min_fold_gain"]),
    "equal3_positive_folds": int(primary["positive_folds"]),
    "nested_menu_rmse": float(menu_result["rmse"]),
    "nested_menu_gain_vs_lgb_strong": float(menu_result["gain_vs_lgb_strong"]),
    "top40_lgb_strong_rmse": top40_scores["lgb_strong"],
    "top40_equal3_strong_rmse": top40_scores["equal3_strong"],
    "top40_nested_menu_rmse": top40_scores["nested_menu"],
    "primary_pass": primary_pass,
    "menu_pass": menu_pass,
    "selected_deployment_family": chosen,
    "training_hours": (time.time() - started) / 3600.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train full strict-spatial models and package Kaggle blend"
        if strict_pass
        else "keep the simpler strict-spatial LGB + strong_bias blend"
    ),
}

print("\nSTANDALONE")
display(standalone)
print("\nNESTED STRONG FOLDS")
display(blend_folds)
print("\nNESTED STRONG SUMMARY")
display(blend_summary)
print("\nNESTED MODEL MENU")
display(menu_folds)
print("\nGAIN VS LGB+STRONG")
display(comparison)
print("\nSTRONG-BIAS TOP40")
display(pd.DataFrame([top40_scores]))
print("\nRESULT")
print(result)
print("\nVERDICT:", result["next"])

oof.to_pickle(RUN_DIR / "three_model_oof.pkl")
standalone.to_csv(RUN_DIR / "standalone.csv", index=False)
blend_folds.to_csv(RUN_DIR / "nested_strong_folds.csv", index=False)
blend_summary.to_csv(RUN_DIR / "nested_strong_summary.csv", index=False)
menu_folds.to_csv(RUN_DIR / "nested_model_menu.csv", index=False)
comparison.to_csv(RUN_DIR / "comparison_vs_lgb_strong.csv", index=False)
bywell.to_csv(RUN_DIR / "bywell.csv", index=False)
(RUN_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", RUN_DIR)
