"""Nested Track6-style independent-PF GBDT gateway.

This is deliberately different from the prior Track8 approximation. It uses:

* the independently constructed ``pf_z`` particle-filter path;
* multi-scale GR/correlation features;
* fold-clean formation-plane and nearest-neighbour geometry features;
* a small LightGBM residual model.

It explicitly excludes the SP45 family: pf_ancc, likelihood-PF, beam, slope,
hybrid, selector, ridge, strong, CP12, and Gold features. The primary gate is
error decorrelation and two-way oracle headroom against SP45, not standalone
RMSE alone.
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


warnings.filterwarnings("ignore")

OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = CACHE_ROOT / "features"
META_PATH = CACHE_ROOT / "well_metadata.csv"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "track6_independent_pf_gbdt_gateway_v1"
MODEL_DIR = OUT_DIR / "models"

SEED = 20260722
CORE_FOLDS = (3, 4, 5)
SELECTION_FOLD = 2
OUTER_FOLDS = (2, 3, 4, 5)
TEST_FOLD = 1
MAX_FEATURES = 180
MAX_TREES = 1800

for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _metrics(frame, prediction, name):
    truth = frame["tvt_true"].to_numpy(float)
    error = np.asarray(prediction, float) - truth
    work = pd.DataFrame(
        {"well": frame["well"].astype(str).to_numpy(), "error2": error * error}
    )
    per_well = work.groupby("well", sort=False)["error2"].mean().pow(0.5)
    return {
        "variant": name,
        "rows": int(len(frame)),
        "wells": int(len(per_well)),
        "exact_sse": float(np.sum(error * error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(per_well.mean()),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
    }


def _feature_paths(well):
    return (
        FEATURE_DIR / f"{well}__sample.pkl",
        FEATURE_DIR / f"{well}__full.pkl",
    )


def _load(wells, kind):
    position = 0 if kind == "sample" else 1
    frames = []
    for well in wells:
        path = _feature_paths(str(well))[position]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path)
        frame["well"] = str(well)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


EXCLUDED = {
    "well",
    "id",
    "target",
    "TVT",
    "TVT_input",
    "_row_idx",
    "fold",
    "sample_weight",
}
SP45_FAMILY_TOKENS = (
    "pf_ancc",
    "likpf",
    "beam_",
    "hyb",
    "slp",
    "selector",
    "ridge",
    "sp45",
    "strong",
    "cp12",
    "gold",
)
TRACK6_PREFIXES = (
    "pf_z",
    "gr",
    "ncc",
    "corr",
    "tvtf_",
    "tvtfw_",
    "tvtf50_",
    "bw_",
    "bww_",
    "bw50_",
    "frm_",
    "form_",
    "dense_",
    "spatial_",
)
TRACK6_CONTAINS = (
    "corr",
    "ncc",
    "neighbor",
    "neighbour",
    "distance",
    "entropy",
    "margin",
    "spectral",
    "resid",
    "typewell",
)
CORE_GEOMETRY = {
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "last_known_tvt",
    "md_since",
    "progress",
}


def _legal_track6_feature(frame, column):
    if column in EXCLUDED or not pd.api.types.is_numeric_dtype(frame[column]):
        return False
    lower = column.lower()
    if any(token in lower for token in ("truth", "oracle", "label")):
        return False
    if any(token in lower for token in SP45_FAMILY_TOKENS):
        return False
    if column in CORE_GEOMETRY:
        return True
    if lower.startswith(TRACK6_PREFIXES):
        return True
    return any(token in lower for token in TRACK6_CONTAINS)


def _priority(column):
    lower = column.lower()
    if lower.startswith("pf_z") or column in CORE_GEOMETRY:
        return 0
    if "corr" in lower or "ncc" in lower or lower.startswith("gr"):
        return 1
    if lower.startswith(("tvtf", "bw", "frm", "form", "dense", "spatial")):
        return 2
    return 3


def _engineer(frame, spatial_paths):
    frame = frame.copy()
    last = frame["last_known_tvt"].to_numpy(np.float32)
    pf_z = frame["pf_z"].to_numpy(np.float32)
    frame["track6_pf_z_delta"] = pf_z - last
    md_since = np.maximum(frame["md_since"].to_numpy(np.float32), 0.0)
    frame["track6_progress_log"] = np.log1p(md_since)
    frame["track6_pf_z_delta_x_progress"] = (pf_z - last) * np.sqrt(md_since + 1.0)

    available = [column for column in spatial_paths if column in frame]
    if available:
        values = frame[available].to_numpy(np.float32)
        values = np.where(np.isfinite(values), values, pf_z[:, None])
        frame["track6_spatial_mean_minus_pf_z"] = np.mean(values, axis=1) - pf_z
        frame["track6_spatial_std"] = np.std(values, axis=1)
        frame["track6_spatial_range"] = np.max(values, axis=1) - np.min(values, axis=1)
        for column in available[:24]:
            frame[f"track6_{column}_minus_pf_z"] = (
                frame[column].to_numpy(np.float32) - pf_z
            )
    else:
        frame["track6_spatial_mean_minus_pf_z"] = 0.0
        frame["track6_spatial_std"] = 0.0
        frame["track6_spatial_range"] = 0.0
    return frame


def _select_features(schema):
    base = [column for column in schema.columns if _legal_track6_feature(schema, column)]
    order = {column: index for index, column in enumerate(base)}
    base = sorted(base, key=lambda column: (_priority(column), order[column]))[:MAX_FEATURES]

    spatial_paths = []
    for column in base:
        lower = column.lower()
        if not lower.startswith(("tvtf", "bw", "dense", "spatial", "form")):
            continue
        values = schema[column].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) and float(np.nanmedian(np.abs(values.to_numpy(float)))) > 1000:
            spatial_paths.append(column)
    engineered = _engineer(schema, spatial_paths)
    additions = [column for column in engineered.columns if column.startswith("track6_")]
    return list(dict.fromkeys(base + additions)), spatial_paths


def _matrix(frame, features):
    return (
        frame.reindex(columns=features)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _model(trees, seed):
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=int(trees),
        learning_rate=0.025,
        num_leaves=48,
        min_child_samples=96,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.20,
        reg_lambda=14.0,
        max_bin=255,
        n_jobs=-1,
        random_state=int(seed),
        verbosity=-1,
    )


def _fit_core(train, valid, features):
    x_train = _matrix(train, features)
    x_valid = _matrix(valid, features)
    y_train = train["target"].to_numpy(np.float32)
    y_valid = valid["target"].to_numpy(np.float32)
    weights = train.get("sample_weight", pd.Series(np.ones(len(train)))).to_numpy(float)
    model = _model(MAX_TREES, SEED)
    fit_kwargs = dict(
        sample_weight=weights,
        eval_set=[(x_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        gpu = LGBMRegressor(**params)
        gpu.fit(x_train, y_train, **fit_kwargs)
        model = gpu
    except Exception as exc:
        if "gpu" not in str(exc).lower() and "opencl" not in str(exc).lower():
            raise
        print("Track6 GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model.fit(x_train, y_train, **fit_kwargs)
    del x_train, x_valid
    gc.collect()
    return model


def _fit_outer(train, features, trees):
    x_train = _matrix(train, features)
    y_train = train["target"].to_numpy(np.float32)
    weights = train.get("sample_weight", pd.Series(np.ones(len(train)))).to_numpy(float)
    model = _model(trees, SEED + 100)
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        gpu = LGBMRegressor(**params)
        gpu.fit(x_train, y_train, sample_weight=weights)
        model = gpu
    except Exception as exc:
        if "gpu" not in str(exc).lower() and "opencl" not in str(exc).lower():
            raise
        print("Track6 outer GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model.fit(x_train, y_train, sample_weight=weights)
    del x_train
    gc.collect()
    return model


def _smooth(frame, prediction, window):
    prediction = np.asarray(prediction, float)
    if window <= 0:
        return prediction.copy()
    output = prediction.copy()
    for indices in frame.groupby("well", sort=False).indices.values():
        indices = np.asarray(indices, int)
        width = min(int(window), len(indices))
        if width % 2 == 0:
            width -= 1
        if width >= 5:
            output[indices] = savgol_filter(prediction[indices], width, 3)
    return output


def _decode(frame, model_delta, recipe):
    last = frame["last_known_tvt"].to_numpy(float)
    pf_delta = frame["pf_z"].to_numpy(float) - last
    md_since = np.maximum(frame["md_since"].to_numpy(float), 0.0)
    warm = 1.0 - np.exp(-md_since / float(recipe["tau"]))
    delta = (
        (1.0 - float(recipe["w_pf_z"])) * np.asarray(model_delta, float)
        + float(recipe["w_pf_z"]) * pf_delta
    )
    prediction = last + float(recipe["alpha"]) * warm * delta
    return _smooth(frame, prediction, int(recipe["sg_window"]))


def _scan(valid, model_delta):
    reports = []
    for alpha in (0.85, 1.00, 1.15):
        for tau in (40.0, 85.0, 160.0):
            for w_pf_z in (0.0, 0.25, 0.50, 0.75):
                for sg_window in (17, 61):
                    recipe = {
                        "alpha": alpha,
                        "tau": tau,
                        "w_pf_z": w_pf_z,
                        "sg_window": sg_window,
                    }
                    prediction = _decode(valid, model_delta, recipe)
                    reports.append({**recipe, **_metrics(valid, prediction, "selection")})
    report = pd.DataFrame(reports).sort_values(
        ["mean_well_rmse", "rmse"], ignore_index=True
    )
    row = report.iloc[0]
    recipe = {
        "alpha": float(row["alpha"]),
        "tau": float(row["tau"]),
        "w_pf_z": float(row["w_pf_z"]),
        "sg_window": int(row["sg_window"]),
    }
    return recipe, report


def _pair(truth, base, candidate, wells):
    truth = np.asarray(truth, float)
    base = np.asarray(base, float)
    candidate = np.asarray(candidate, float)
    base_error = base - truth
    candidate_error = candidate - truth
    cosine = float(
        np.dot(base_error, candidate_error)
        / np.sqrt(np.dot(base_error, base_error) * np.dot(candidate_error, candidate_error))
    )
    rows = pd.DataFrame(
        {
            "well": np.asarray(wells, str),
            "truth": truth,
            "base": base,
            "candidate": candidate,
        }
    )
    fixed_rows = []
    for weight in np.linspace(0.0, 1.0, 101):
        prediction = base + weight * (candidate - base)
        metric = _metrics(
            pd.DataFrame({"well": rows["well"], "tvt_true": truth}), prediction, "fixed"
        )
        fixed_rows.append((weight, metric["rmse"], metric["mean_well_rmse"]))
    fixed = min(fixed_rows, key=lambda item: (item[2], item[1]))

    oracle = np.empty(len(rows), float)
    oracle_weights = []
    for well, indices in rows.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, int)
        direction = candidate[indices] - base[indices]
        denominator = float(np.dot(direction, direction))
        weight = 0.0
        if denominator > 1e-12:
            weight = float(
                np.clip(-np.dot(base_error[indices], direction) / denominator, 0.0, 1.0)
            )
        oracle[indices] = base[indices] + weight * direction
        oracle_weights.append(weight)
    oracle_metrics = _metrics(
        pd.DataFrame({"well": rows["well"], "tvt_true": truth}), oracle, "oracle"
    )
    return {
        "error_cosine": cosine,
        "fixed_weight_track6": float(fixed[0]),
        "fixed_rmse": float(fixed[1]),
        "fixed_mean_well_rmse": float(fixed[2]),
        "oracle_rmse": oracle_metrics["rmse"],
        "oracle_mean_well_rmse": oracle_metrics["mean_well_rmse"],
        "oracle_p95_well_rmse": oracle_metrics["p95_well_rmse"],
        "mean_oracle_weight_track6": float(np.mean(oracle_weights)),
    }


started = time.time()
meta = pd.read_csv(META_PATH, dtype={"well": str})
core_wells = meta.loc[meta["fold"].isin(CORE_FOLDS), "well"].tolist()
selection_wells = meta.loc[meta["fold"].eq(SELECTION_FOLD), "well"].tolist()
outer_wells = meta.loc[meta["fold"].isin(OUTER_FOLDS), "well"].tolist()
test_wells = meta.loc[meta["fold"].eq(TEST_FOLD), "well"].tolist()

schema = _load(core_wells[:8], "sample")
if "pf_z" not in schema:
    raise RuntimeError("fold-clean feature cache has no independent pf_z path")
features, spatial_paths = _select_features(schema)
schema = _engineer(schema, spatial_paths)
if len(features) < 30:
    raise RuntimeError(f"only {len(features)} Track6 features selected")
del schema
gc.collect()
print(
    "TRACK6 CONFIG",
    {
        "core_wells": len(core_wells),
        "selection_wells": len(selection_wells),
        "outer_wells": len(outer_wells),
        "test_wells": len(test_wells),
        "features": len(features),
        "spatial_paths": spatial_paths,
    },
    flush=True,
)

core = _engineer(_load(core_wells, "sample"), spatial_paths)
selection = _engineer(_load(selection_wells, "full"), spatial_paths)
selection["tvt_true"] = selection["last_known_tvt"] + selection["target"]
core_model = _fit_core(core, selection, features)
selected_trees = int(core_model.best_iteration_ or MAX_TREES)
selection_delta = core_model.predict(
    _matrix(selection, features), num_iteration=selected_trees
)
recipe, scan = _scan(selection, selection_delta)
selection_prediction = _decode(selection, selection_delta, recipe)
scan.to_csv(OUT_DIR / "fold2_recipe_scan.csv", index=False)
joblib.dump(core_model, MODEL_DIR / "track6_core345.joblib")
print(
    "TRACK6 FOLD2 FROZEN",
    {
        "selected_trees": selected_trees,
        "recipe": recipe,
        "metrics": _metrics(selection, selection_prediction, "track6"),
    },
    flush=True,
)
del core, core_model, selection, selection_delta, selection_prediction
gc.collect()

outer = _engineer(_load(outer_wells, "sample"), spatial_paths)
test = _engineer(_load(test_wells, "full"), spatial_paths)
test["tvt_true"] = test["last_known_tvt"] + test["target"]
outer_model = _fit_outer(outer, features, selected_trees)
test_delta = outer_model.predict(_matrix(test, features))
track6 = _decode(test, test_delta, recipe)

# Matched controls: remove or shuffle the independent evidence at inference.
no_pf = test.copy()
for column in features:
    if column == "pf_z":
        no_pf[column] = no_pf["last_known_tvt"].to_numpy(float)
    elif column.lower().startswith("pf_z") or column.startswith("track6_pf_z"):
        no_pf[column] = 0.0
no_pf_delta = outer_model.predict(_matrix(no_pf, features))
no_pf_prediction = _decode(test, no_pf_delta, {**recipe, "w_pf_z": 0.0})

shuffled = test.copy()
rng = np.random.default_rng(SEED + 999)
shuffle_columns = [
    column
    for column in features
    if column.lower().startswith(("pf_z", "tvtf", "bw", "frm", "form", "dense", "spatial"))
    or column.startswith("track6_")
]
permutation = rng.permutation(len(shuffled))
for column in shuffle_columns:
    shuffled[column] = shuffled[column].to_numpy()[permutation]
shuffle_delta = outer_model.predict(_matrix(shuffled, features))
shuffle_prediction = _decode(test, shuffle_delta, {**recipe, "w_pf_z": 0.0})

joblib.dump(outer_model, MODEL_DIR / "track6_outer2345.joblib")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
sp45_source = next((column for column in ("sp45", "sp45_like") if column in header), None)
if sp45_source is None:
    raise RuntimeError("candidate pool has no SP45 endpoint")
context = [column for column in ("cp12", "strong_bias", "strong", "fleongg") if column in header]
pool = pd.read_csv(POOL_PATH, usecols=["id", sp45_source] + context, dtype={"id": str})
pool = pool.rename(columns={sp45_source: "sp45"})
joined = test[["id"]].copy()
joined["id"] = joined["id"].astype(str)
joined = joined.merge(pool, on="id", how="left", validate="one_to_one")
if joined["sp45"].isna().any():
    raise RuntimeError("SP45 did not cover all Fold1 rows")

sp45 = joined["sp45"].to_numpy(float)
truth = test["tvt_true"].to_numpy(float)
pair = _pair(truth, sp45, track6, test["well"].astype(str).to_numpy())
score_rows = [
    _metrics(test, track6, "track6"),
    _metrics(test, no_pf_prediction, "no_pf_z"),
    _metrics(test, shuffle_prediction, "shuffled_independent_evidence"),
    _metrics(test, sp45, "sp45"),
]
for column in context:
    score_rows.append(_metrics(test, joined[column].to_numpy(float), column))
scores = pd.DataFrame(score_rows).sort_values("rmse", ignore_index=True)

track6_metrics = _metrics(test, track6, "track6")
no_pf_metrics = _metrics(test, no_pf_prediction, "no_pf_z")
shuffle_metrics = _metrics(test, shuffle_prediction, "shuffled")
strict_pass = bool(
    pair["error_cosine"] <= 0.50
    and pair["oracle_mean_well_rmse"] <= 5.30
    and pair["fixed_mean_well_rmse"] <= 6.40
    and track6_metrics["rmse"] < no_pf_metrics["rmse"] - 0.20
    and track6_metrics["rmse"] < shuffle_metrics["rmse"] - 0.20
)

oof = test[["id", "well", "fold", "tvt_true", "md_since"]].copy()
oof["track6"] = track6.astype(np.float32)
oof["sp45"] = sp45.astype(np.float32)
oof["no_pf_z"] = no_pf_prediction.astype(np.float32)
oof["shuffled_independent_evidence"] = shuffle_prediction.astype(np.float32)
oof.to_pickle(OUT_DIR / "fold1_track6_oof.pkl")
scores.to_csv(OUT_DIR / "fold1_scores.csv", index=False)
pd.Series(features, name="feature").to_csv(OUT_DIR / "features.csv", index=False)

verdict = {
    "experiment": "track6_independent_pf_gbdt_gateway_v1",
    "stage": "fold2_nested_selection_to_untouched_fold1",
    "protocol": {
        "core_folds": list(CORE_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "outer_folds": list(OUTER_FOLDS),
        "test_fold": TEST_FOLD,
        "selected_trees": selected_trees,
    },
    "features": len(features),
    "spatial_paths": spatial_paths,
    "excluded_sp45_family": list(SP45_FAMILY_TOKENS),
    "recipe": recipe,
    "track6": track6_metrics,
    "sp45": _metrics(test, sp45, "sp45"),
    "pair": pair,
    "controls": {
        "no_pf_z_rmse": no_pf_metrics["rmse"],
        "shuffled_independent_evidence_rmse": shuffle_metrics["rmse"],
    },
    "gates": {
        "error_cosine_le": 0.50,
        "two_way_oracle_mean_well_rmse_le": 5.30,
        "fixed_blend_mean_well_rmse_le": 6.40,
        "each_control_degradation_ge": 0.20,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> add CatBoost member and cross-fit the published disagreement weight"
        if strict_pass
        else "FAIL -> cached pf_z/correlation/NN features do not reproduce Track6 independence"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nTRACK6 FOLD1 SCORES")
print(scores.to_string(index=False), flush=True)
print("\nTRACK6/SP45 PAIR")
print(json.dumps(pair, indent=2), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
