"""Fast nested Track8/SP45 reproduction gateway.

This cell reuses the fold-clean feature cache produced by
``foldclean_full_fleongg_lgb_v0_cell.py``. It does not rebuild PF features.

Protocol
--------
1. Train a Track8-like non-spatial GBDT on folds 3/4/5 and select its tree
   count, PF dose, warm-up, smoothing, and fixed blend on fold 2.
2. Fit a well-level soft confidence gate on fold 2 only.
3. Refit the GBDT on folds 2/3/4/5 with the frozen tree count and evaluate
   the untouched fold 1.
4. Report both a matched PF-centred endpoint and transfer to the historical
   SP45 endpoint from the candidate pool. A shuffled gate is mandatory.

The target unit is TVT feet. No neighbour, spatial, CP12, Gold, or validation
truth feature enters either model.
"""

from pathlib import Path
import gc
import json
import math
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings("ignore")

OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = CACHE_ROOT / "features"
META_PATH = CACHE_ROOT / "well_metadata.csv"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "track8_sp45_confidence_nested_gateway_v0"
MODEL_DIR = OUT_DIR / "models"

SEED = 20260721
CORE_TRAIN_FOLDS = (3, 4, 5)
SELECTION_FOLD = 2
OUTER_TRAIN_FOLDS = (2, 3, 4, 5)
TEST_FOLD = 1
MAX_FEATURES = 160
MAX_TREES = 1600

for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _rmse(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _mae(truth, prediction):
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction, dtype=np.float64)
                - np.asarray(truth, dtype=np.float64)
            )
        )
    )


def _metrics(rows, prediction, label):
    truth = rows["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    work = pd.DataFrame(
        {
            "well": rows["well"].astype(str).to_numpy(),
            "error2": error * error,
            "abs_error": np.abs(error),
        }
    )
    per_well = work.groupby("well", sort=False).agg(
        rows=("error2", "size"),
        mse=("error2", "mean"),
        mae=("abs_error", "mean"),
    )
    per_well["rmse"] = np.sqrt(per_well["mse"])
    return {
        "variant": label,
        "rows": int(len(rows)),
        "wells": int(len(per_well)),
        "exact_sse": float(np.sum(error * error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(per_well["rmse"].mean()),
        "median_well_rmse": float(per_well["rmse"].median()),
        "p75_well_rmse": float(per_well["rmse"].quantile(0.75)),
        "p95_well_rmse": float(per_well["rmse"].quantile(0.95)),
        "max_well_rmse": float(per_well["rmse"].max()),
    }


def _feature_paths(well):
    return (
        FEATURE_DIR / f"{well}__sample.pkl",
        FEATURE_DIR / f"{well}__full.pkl",
    )


def _load_frames(wells, kind):
    frames = []
    position = 0 if kind == "sample" else 1
    for well in wells:
        path = _feature_paths(str(well))[position]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run foldclean_full_fleongg_lgb_v0_cell.py "
                "through feature-cache generation first."
            )
        frame = pd.read_pickle(path)
        frame["well"] = str(well)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


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
EXCLUDED_EXACT = {
    "well",
    "id",
    "target",
    "TVT",
    "TVT_input",
    "_row_idx",
    "fold",
    "sample_weight",
}


def _is_spatial(column):
    return column in SPATIAL_EXACT or column.startswith(SPATIAL_PREFIXES)


def _is_legal_numeric(frame, column):
    lower = column.lower()
    if column in EXCLUDED_EXACT or _is_spatial(column):
        return False
    if not pd.api.types.is_numeric_dtype(frame[column]):
        return False
    if any(token in lower for token in ("truth", "oracle", "label")):
        return False
    return True


def _feature_priority(column):
    lower = column.lower()
    exact = {
        "md",
        "z",
        "gr",
        "last_known_tvt",
        "md_since",
        "pf_ancc",
        "pf_ancc_std",
        "pf_ancc_delta",
    }
    if lower in exact:
        return 0
    high_signal = (
        "pf_",
        "likpf_",
        "beam_",
        "sc",
        "hyb",
        "slp",
        "gr_",
        "corr",
        "ncc",
        "typewell",
        "resid",
        "progress",
        "md_since",
    )
    if lower.startswith(high_signal) or any(
        token in lower for token in ("corr", "ncc", "entropy", "margin")
    ):
        return 1
    return 2


def _select_base_features(schema):
    saved = CACHE_ROOT / "no_spatial_features.csv"
    if saved.exists():
        listed = pd.read_csv(saved)["feature"].astype(str).tolist()
        candidates = [
            column
            for column in listed
            if column in schema and _is_legal_numeric(schema, column)
        ]
    else:
        candidates = [
            column for column in schema.columns if _is_legal_numeric(schema, column)
        ]
    order = {column: index for index, column in enumerate(candidates)}
    candidates = sorted(
        candidates, key=lambda column: (_feature_priority(column), order[column])
    )
    return candidates[:MAX_FEATURES]


def _path_columns(schema):
    prefixes = ("pf_", "likpf_", "beam_", "sc", "hyb", "slp")
    output = []
    for column in schema.columns:
        if not column.lower().startswith(prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(schema[column]):
            continue
        values = schema[column].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(values):
            continue
        # Absolute TVT paths are around 10,000 ft; uncertainty and deltas are not.
        if float(np.nanmedian(np.abs(values.to_numpy(float)))) > 1000.0:
            output.append(column)
    if "pf_ancc" in schema and "pf_ancc" not in output:
        output.append("pf_ancc")
    return list(dict.fromkeys(output))


def _engineer(frame, path_columns):
    frame = frame.copy()
    last = frame["last_known_tvt"].to_numpy(np.float32)
    pf = frame["pf_ancc"].to_numpy(np.float32)
    frame["track8_pf_delta"] = pf - last
    available = [column for column in path_columns if column in frame]
    if available:
        paths = frame[available].to_numpy(np.float32)
        paths = np.where(np.isfinite(paths), paths, pf[:, None])
        frame["track8_path_mean_delta"] = np.mean(paths, axis=1) - last
        frame["track8_path_std"] = np.std(paths, axis=1)
        frame["track8_path_range"] = np.max(paths, axis=1) - np.min(paths, axis=1)
        frame["track8_path_mad_pf"] = np.mean(np.abs(paths - pf[:, None]), axis=1)
        for column in available[:24]:
            frame[f"track8_{column}_minus_pf"] = (
                frame[column].to_numpy(np.float32) - pf
            )
    else:
        for column in (
            "track8_path_mean_delta",
            "track8_path_std",
            "track8_path_range",
            "track8_path_mad_pf",
        ):
            frame[column] = 0.0
    md_since = np.maximum(frame["md_since"].to_numpy(np.float32), 0.0)
    frame["track8_progress_log"] = np.log1p(md_since)
    frame["track8_pf_std_x_progress"] = (
        frame.get("pf_ancc_std", pd.Series(np.zeros(len(frame))))
        .to_numpy(np.float32)
        * np.sqrt(md_since + 1.0)
    )
    return frame


def _matrix(frame, columns):
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _new_model(n_estimators, seed):
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        learning_rate=0.025,
        num_leaves=48,
        min_child_samples=96,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=12.0,
        max_bin=255,
        n_jobs=-1,
        random_state=int(seed),
        verbosity=-1,
    )


def _fit_core(train, valid, columns):
    x_train = _matrix(train, columns)
    x_valid = _matrix(valid, columns)
    y_train = train["target"].to_numpy(np.float32)
    y_valid = valid["target"].to_numpy(np.float32)
    weights = train.get("sample_weight", pd.Series(np.ones(len(train)))).to_numpy(float)
    model = _new_model(MAX_TREES, SEED)
    fit_kwargs = dict(
        sample_weight=weights,
        eval_set=[(x_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        candidate = LGBMRegressor(**params)
        candidate.fit(x_train, y_train, **fit_kwargs)
        model = candidate
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("Track8 GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model.fit(x_train, y_train, **fit_kwargs)
    del x_train, x_valid
    gc.collect()
    return model


def _fit_outer(train, columns, n_estimators):
    x_train = _matrix(train, columns)
    y_train = train["target"].to_numpy(np.float32)
    weights = train.get("sample_weight", pd.Series(np.ones(len(train)))).to_numpy(float)
    model = _new_model(n_estimators, SEED + 100)
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        candidate = LGBMRegressor(**params)
        candidate.fit(x_train, y_train, sample_weight=weights)
        model = candidate
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("Track8 outer GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model.fit(x_train, y_train, sample_weight=weights)
    del x_train
    gc.collect()
    return model


def _smooth(frame, values, window):
    values = np.asarray(values, dtype=np.float64).copy()
    if int(window) <= 0:
        return values
    output = values.copy()
    for indices in frame.groupby("well", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        width = min(int(window), len(indices))
        if width % 2 == 0:
            width -= 1
        if width >= 5:
            output[indices] = savgol_filter(values[indices], width, 3)
    return output


def _decode(frame, model_delta, recipe):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    pf_delta = frame["pf_ancc"].to_numpy(np.float64) - last
    md_since = np.maximum(frame["md_since"].to_numpy(np.float64), 0.0)
    tau = float(recipe["tau"])
    warm = np.ones(len(frame)) if tau <= 0 else 1.0 - np.exp(-md_since / tau)
    mixed_delta = (
        (1.0 - float(recipe["w_pf"])) * np.asarray(model_delta, dtype=np.float64)
        + float(recipe["w_pf"]) * pf_delta
    )
    prediction = last + float(recipe["alpha"]) * warm * mixed_delta
    return _smooth(frame, prediction, int(recipe["sg_window"]))


def _with_truth(frame):
    frame = frame.copy()
    frame["tvt_true"] = (
        frame["last_known_tvt"].to_numpy(np.float64)
        + frame["target"].to_numpy(np.float64)
    )
    return frame


def _scan_recipes(valid, model_delta):
    rows = _with_truth(valid)
    reports = []
    for alpha in (0.80, 1.00, 1.15):
        for tau in (40.0, 85.0, 160.0):
            for w_pf in (0.0, 0.25, 0.50, 0.75):
                for sg_window in (0, 17, 61):
                    recipe = {
                        "alpha": alpha,
                        "tau": tau,
                        "w_pf": w_pf,
                        "sg_window": sg_window,
                    }
                    prediction = _decode(valid, model_delta, recipe)
                    metrics = _metrics(rows, prediction, "selection")
                    reports.append({**recipe, **metrics})
    report = pd.DataFrame(reports).sort_values(
        ["mean_well_rmse", "rmse"], ignore_index=True
    )
    winner = report.iloc[0]
    recipe = {
        "alpha": float(winner["alpha"]),
        "tau": float(winner["tau"]),
        "w_pf": float(winner["w_pf"]),
        "sg_window": int(winner["sg_window"]),
    }
    return recipe, report


def _optimal_weight(truth, base, candidate):
    truth = np.asarray(truth, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    direction = np.asarray(candidate, dtype=np.float64) - base
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(base - truth, direction) / denominator, 0.0, 1.0))


def _fixed_weight_scan(rows, base, candidate):
    reports = []
    truth = rows["tvt_true"].to_numpy(np.float64)
    for weight in np.linspace(0.0, 1.0, 41):
        prediction = base + float(weight) * (candidate - base)
        metrics = _metrics(rows, prediction, "fixed")
        reports.append({"weight_track8": float(weight), **metrics})
    report = pd.DataFrame(reports).sort_values(
        ["mean_well_rmse", "rmse"], ignore_index=True
    )
    return float(report.iloc[0]["weight_track8"]), report


def _well_gate_table(frame, base, candidate, include_target):
    work = pd.DataFrame(
        {
            "well": frame["well"].astype(str).to_numpy(),
            "fold": frame["fold"].to_numpy(int),
            "md_since": frame["md_since"].to_numpy(float),
            "base": np.asarray(base, dtype=float),
            "candidate": np.asarray(candidate, dtype=float),
            "disagreement": np.asarray(candidate, dtype=float) - np.asarray(base, dtype=float),
            "pf_std": frame.get("pf_ancc_std", pd.Series(np.zeros(len(frame)))).to_numpy(float),
            "path_std": frame.get("track8_path_std", pd.Series(np.zeros(len(frame)))).to_numpy(float),
            "path_range": frame.get("track8_path_range", pd.Series(np.zeros(len(frame)))).to_numpy(float),
            "path_mad": frame.get("track8_path_mad_pf", pd.Series(np.zeros(len(frame)))).to_numpy(float),
        }
    )
    if include_target:
        work["truth"] = frame["tvt_true"].to_numpy(float)
    rows = []
    for well, group in work.groupby("well", sort=False):
        disagreement = group["disagreement"].to_numpy(float)
        record = {
            "well": str(well),
            "fold": int(group["fold"].iloc[0]),
            "rows": int(len(group)),
            "md_since_max": float(group["md_since"].max()),
            "md_since_mean": float(group["md_since"].mean()),
            "disagreement_mean": float(np.mean(disagreement)),
            "disagreement_abs_mean": float(np.mean(np.abs(disagreement))),
            "disagreement_rms": float(np.sqrt(np.mean(disagreement ** 2))),
            "disagreement_p95": float(np.quantile(np.abs(disagreement), 0.95)),
            "disagreement_end": float(np.mean(disagreement[-min(128, len(disagreement)) :])),
            "pf_std_mean": float(group["pf_std"].mean()),
            "pf_std_p95": float(group["pf_std"].quantile(0.95)),
            "path_std_mean": float(group["path_std"].mean()),
            "path_range_mean": float(group["path_range"].mean()),
            "path_mad_mean": float(group["path_mad"].mean()),
        }
        if include_target:
            record["optimal_weight"] = _optimal_weight(
                group["truth"], group["base"], group["candidate"]
            )
            record["base_rmse"] = _rmse(group["truth"], group["base"])
            record["candidate_rmse"] = _rmse(group["truth"], group["candidate"])
        rows.append(record)
    return pd.DataFrame(rows)


GATE_FEATURES = [
    "rows",
    "md_since_max",
    "md_since_mean",
    "disagreement_mean",
    "disagreement_abs_mean",
    "disagreement_rms",
    "disagreement_p95",
    "disagreement_end",
    "pf_std_mean",
    "pf_std_p95",
    "path_std_mean",
    "path_range_mean",
    "path_mad_mean",
]


def _fit_gate(selection_wells, shuffled=False):
    target = selection_wells["optimal_weight"].to_numpy(float).copy()
    if shuffled:
        target = np.random.default_rng(SEED + 999).permutation(target)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=8,
        max_features=0.8,
        random_state=SEED + (1000 if shuffled else 0),
        n_jobs=-1,
    )
    model.fit(selection_wells[GATE_FEATURES].fillna(0.0), target)
    return model


def _map_well_weights(frame, well_table, weights):
    mapping = dict(zip(well_table["well"].astype(str), np.asarray(weights, float)))
    return frame["well"].astype(str).map(mapping).to_numpy(float)


started = time.time()
if not META_PATH.exists():
    raise FileNotFoundError(META_PATH)
if not POOL_PATH.exists():
    raise FileNotFoundError(POOL_PATH)

meta = pd.read_csv(META_PATH, dtype={"well": str})
required_meta = {"well", "fold"}
if not required_meta.issubset(meta.columns):
    raise RuntimeError(f"metadata missing {sorted(required_meta - set(meta.columns))}")

core_wells = meta.loc[meta["fold"].isin(CORE_TRAIN_FOLDS), "well"].tolist()
selection_wells = meta.loc[meta["fold"].eq(SELECTION_FOLD), "well"].tolist()
outer_wells = meta.loc[meta["fold"].isin(OUTER_TRAIN_FOLDS), "well"].tolist()
test_wells = meta.loc[meta["fold"].eq(TEST_FOLD), "well"].tolist()
print(
    "TRACK8 CONFIG",
    {
        "core_wells": len(core_wells),
        "selection_wells": len(selection_wells),
        "outer_wells": len(outer_wells),
        "test_wells": len(test_wells),
        "cache": str(FEATURE_DIR),
    },
    flush=True,
)

schema = _load_frames(core_wells[: min(8, len(core_wells))], "sample")
base_features = _select_base_features(schema)
path_columns = _path_columns(schema)
schema = _engineer(schema, path_columns)
engineered_features = [
    column for column in schema.columns if column.startswith("track8_")
]
feature_columns = list(dict.fromkeys(base_features + engineered_features))
del schema
gc.collect()

if "pf_ancc" not in feature_columns or len(feature_columns) < 40:
    raise RuntimeError(
        f"invalid Track8 schema: features={len(feature_columns)}, "
        f"pf_ancc={'pf_ancc' in feature_columns}"
    )
print(
    "TRACK8 FEATURES",
    {
        "features": len(feature_columns),
        "base_features": len(base_features),
        "engineered_features": len(engineered_features),
        "path_columns": path_columns,
    },
    flush=True,
)

# Nested selection phase: folds 3/4/5 -> fold 2.
core_train = _engineer(_load_frames(core_wells, "sample"), path_columns)
selection = _engineer(_load_frames(selection_wells, "full"), path_columns)
selection["tvt_true"] = (
    selection["last_known_tvt"].to_numpy(float)
    + selection["target"].to_numpy(float)
)
core_model = _fit_core(core_train, selection, feature_columns)
selected_trees = int(core_model.best_iteration_ or MAX_TREES)
selection_delta = core_model.predict(
    _matrix(selection, feature_columns), num_iteration=selected_trees
)
recipe, recipe_scan = _scan_recipes(selection, selection_delta)
selection_track8 = _decode(selection, selection_delta, recipe)
selection_pf = _smooth(selection, selection["pf_ancc"].to_numpy(float), recipe["sg_window"])
fixed_weight, fixed_scan = _fixed_weight_scan(selection, selection_pf, selection_track8)

selection_gate_table = _well_gate_table(
    selection, selection_pf, selection_track8, include_target=True
)
gate_model = _fit_gate(selection_gate_table, shuffled=False)
shuffle_gate_model = _fit_gate(selection_gate_table, shuffled=True)

joblib.dump(core_model, MODEL_DIR / "track8_core345.joblib")
joblib.dump(gate_model, MODEL_DIR / "confidence_gate_fold2.joblib")
joblib.dump(shuffle_gate_model, MODEL_DIR / "confidence_gate_shuffled_fold2.joblib")
recipe_scan.to_csv(OUT_DIR / "fold2_recipe_scan.csv", index=False)
fixed_scan.to_csv(OUT_DIR / "fold2_fixed_weight_scan.csv", index=False)
selection_gate_table.to_csv(OUT_DIR / "fold2_gate_training_wells.csv", index=False)
print(
    "FOLD2 FROZEN",
    {
        "selected_trees": selected_trees,
        "recipe": recipe,
        "fixed_weight_track8": fixed_weight,
        "pf": _metrics(selection, selection_pf, "pf"),
        "track8": _metrics(selection, selection_track8, "track8"),
    },
    flush=True,
)

del core_train, core_model, selection_delta
gc.collect()

# Untouched Fold1 application: folds 2/3/4/5 -> fold 1.
outer_train = _engineer(_load_frames(outer_wells, "sample"), path_columns)
test = _engineer(_load_frames(test_wells, "full"), path_columns)
test["tvt_true"] = (
    test["last_known_tvt"].to_numpy(float) + test["target"].to_numpy(float)
)
outer_model = _fit_outer(outer_train, feature_columns, selected_trees)
test_delta = outer_model.predict(_matrix(test, feature_columns))
test_track8 = _decode(test, test_delta, recipe)
test_pf = _smooth(test, test["pf_ancc"].to_numpy(float), recipe["sg_window"])
joblib.dump(outer_model, MODEL_DIR / "track8_outer2345.joblib")

# Historical SP45/strong context exists for the Fold1 OOF rows only.
pool_header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
sp45_source = next(
    (column for column in ("sp45", "sp45_like") if column in pool_header), None
)
if sp45_source is None:
    raise RuntimeError(f"candidate pool has no SP45 endpoint: {pool_header[:40]}")
context_sources = {
    name: next((column for column in aliases if column in pool_header), None)
    for name, aliases in {
        "strong": ("strong",),
        "strong_bias": ("strong_bias",),
        "fleongg": ("fleongg", "fle"),
        "cp12": ("cp12",),
    }.items()
}
context_sources = {name: source for name, source in context_sources.items() if source}
pool_columns = ["id", sp45_source] + list(context_sources.values())
pool = pd.read_csv(
    POOL_PATH,
    usecols=list(dict.fromkeys(pool_columns)),
    dtype={"id": str},
)
if pool["id"].duplicated().any():
    raise RuntimeError("candidate pool contains duplicate ids")
pool = pool.rename(columns={sp45_source: "sp45"})
pool = pool.rename(columns={source: name for name, source in context_sources.items()})

test_context = test[["id"]].copy()
test_context["id"] = test_context["id"].astype(str)
test_context = test_context.merge(pool, on="id", how="left", validate="one_to_one")
if test_context["sp45"].isna().any():
    missing = int(test_context["sp45"].isna().sum())
    raise RuntimeError(f"SP45 merge missed {missing}/{len(test_context)} Fold1 rows")
test_sp45 = test_context["sp45"].to_numpy(float)

# Apply the Fold2-trained gate to both matched PF and transferred SP45 bases.
test_gate_pf = _well_gate_table(test, test_pf, test_track8, include_target=False)
weight_real_pf = np.clip(
    gate_model.predict(test_gate_pf[GATE_FEATURES].fillna(0.0)), 0.0, 1.0
)
weight_shuffle_pf = np.clip(
    shuffle_gate_model.predict(test_gate_pf[GATE_FEATURES].fillna(0.0)), 0.0, 1.0
)
row_weight_real_pf = _map_well_weights(test, test_gate_pf, weight_real_pf)
row_weight_shuffle_pf = _map_well_weights(test, test_gate_pf, weight_shuffle_pf)

matched_fixed = test_pf + fixed_weight * (test_track8 - test_pf)
matched_gate = test_pf + row_weight_real_pf * (test_track8 - test_pf)
matched_shuffle = test_pf + row_weight_shuffle_pf * (test_track8 - test_pf)

test_gate_sp45 = _well_gate_table(test, test_sp45, test_track8, include_target=False)
weight_real_sp45 = np.clip(
    gate_model.predict(test_gate_sp45[GATE_FEATURES].fillna(0.0)), 0.0, 1.0
)
weight_shuffle_sp45 = np.clip(
    shuffle_gate_model.predict(test_gate_sp45[GATE_FEATURES].fillna(0.0)), 0.0, 1.0
)
row_weight_real_sp45 = _map_well_weights(test, test_gate_sp45, weight_real_sp45)
row_weight_shuffle_sp45 = _map_well_weights(test, test_gate_sp45, weight_shuffle_sp45)

sp45_fixed = test_sp45 + fixed_weight * (test_track8 - test_sp45)
sp45_gate = test_sp45 + row_weight_real_sp45 * (test_track8 - test_sp45)
sp45_shuffle = test_sp45 + row_weight_shuffle_sp45 * (test_track8 - test_sp45)

# Same-fold oracle is diagnostic only and quantifies remaining confidence headroom.
oracle_weights = []
oracle_prediction = np.empty(len(test), dtype=float)
for well, indices in test.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, dtype=int)
    weight = _optimal_weight(
        test["tvt_true"].to_numpy(float)[indices],
        test_sp45[indices],
        test_track8[indices],
    )
    oracle_weights.append({"well": str(well), "oracle_weight_track8": weight})
    oracle_prediction[indices] = test_sp45[indices] + weight * (
        test_track8[indices] - test_sp45[indices]
    )

predictions = {
    "pf_ancc": test_pf,
    "track8": test_track8,
    "matched_pf_fixed": matched_fixed,
    "matched_pf_gate": matched_gate,
    "matched_pf_shuffled_gate": matched_shuffle,
    "sp45": test_sp45,
    "sp45_fixed_transfer": sp45_fixed,
    "sp45_confidence_transfer": sp45_gate,
    "sp45_shuffled_transfer": sp45_shuffle,
    "sp45_track8_well_oracle": oracle_prediction,
}
for name in context_sources:
    predictions[name] = test_context[name].to_numpy(float)

score_table = pd.DataFrame(
    [_metrics(test, prediction, name) for name, prediction in predictions.items()]
).sort_values("rmse", ignore_index=True)

base_metrics = _metrics(test, test_sp45, "sp45")
gate_metrics = _metrics(test, sp45_gate, "sp45_confidence_transfer")
shuffle_metrics = _metrics(test, sp45_shuffle, "sp45_shuffled_transfer")
oracle_metrics = _metrics(test, oracle_prediction, "sp45_track8_well_oracle")
strict_pass = bool(
    gate_metrics["rmse"] < base_metrics["rmse"] - 0.10
    and gate_metrics["mean_well_rmse"] < base_metrics["mean_well_rmse"] - 0.15
    and gate_metrics["rmse"] < shuffle_metrics["rmse"] - 0.05
)

oof = test[["id", "well", "fold", "tvt_true", "md_since"]].copy()
for name, prediction in predictions.items():
    oof[name] = np.asarray(prediction, dtype=np.float32)
oof.to_pickle(OUT_DIR / "fold1_track8_sp45_oof.pkl")
score_table.to_csv(OUT_DIR / "fold1_scores.csv", index=False)
pd.DataFrame(oracle_weights).to_csv(OUT_DIR / "fold1_oracle_weights.csv", index=False)
test_gate_pf.assign(
    predicted_weight=weight_real_pf,
    shuffled_weight=weight_shuffle_pf,
).to_csv(OUT_DIR / "fold1_matched_pf_gate.csv", index=False)
test_gate_sp45.assign(
    predicted_weight=weight_real_sp45,
    shuffled_weight=weight_shuffle_sp45,
).to_csv(OUT_DIR / "fold1_sp45_transfer_gate.csv", index=False)
pd.Series(feature_columns, name="feature").to_csv(
    OUT_DIR / "track8_features.csv", index=False
)

verdict = {
    "experiment": "track8_sp45_confidence_nested_gateway_v0",
    "stage": "fold2_nested_selection_to_untouched_fold1",
    "sample_unit": "rows sampled within complete physical wells",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_spatial": False,
    "uses_cp12_gold": False,
    "protocol": {
        "core_train_folds": list(CORE_TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "outer_train_folds": list(OUTER_TRAIN_FOLDS),
        "untouched_test_fold": TEST_FOLD,
        "selected_trees": selected_trees,
    },
    "features": len(feature_columns),
    "path_columns": path_columns,
    "frozen_recipe": recipe,
    "frozen_fixed_weight_track8": fixed_weight,
    "sp45_source": sp45_source,
    "fold1_scores": score_table.to_dict(orient="records"),
    "sp45_rmse": base_metrics["rmse"],
    "track8_rmse": _metrics(test, test_track8, "track8")["rmse"],
    "confidence_transfer_rmse": gate_metrics["rmse"],
    "confidence_transfer_gain_vs_sp45": base_metrics["rmse"] - gate_metrics["rmse"],
    "confidence_transfer_mean_well_gain": (
        base_metrics["mean_well_rmse"] - gate_metrics["mean_well_rmse"]
    ),
    "shuffled_transfer_rmse": shuffle_metrics["rmse"],
    "well_oracle_rmse": oracle_metrics["rmse"],
    "mean_predicted_weight_track8": float(np.mean(weight_real_sp45)),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> repeat 5 seeds and export Kaggle Track8 endpoint"
        if strict_pass
        else "FAIL -> the published SP45/Track8 confidence mechanism does not transfer on this lineage"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nFOLD1 SCORES")
print(score_table.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

