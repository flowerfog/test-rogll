"""Fold-clean low-error-correlation GBDT endpoint discovery.

This experiment intentionally avoids distillation.  The current mainline (or
the strongest available fold-clean proxy) is used only for OOF measurement and
nested blend-weight selection.  It is never a model feature or a training
label.

Three GBDT arms are trained from disjoint, deployable evidence partitions:

* trajectory: local drilling geometry and visible-prefix trend;
* gr_reference: raw GR/typewell/correlation summaries without PF/beam outputs;
* local_union: all non-lineage local features.

Every arm excludes PF, likelihood-PF, beam, SP45, CP, strong, selector, Gold,
neighbor, spatial-KNN and dense-surface feature families.  The aim is not the
best standalone score.  The aim is a competent endpoint whose errors are low
correlation with the mainline and whose nested blend improves held-out RMSE.
"""

from pathlib import Path
import gc
import json
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.signal import savgol_filter


warnings.filterwarnings("ignore")

EXPERIMENT = "decorrelated_evidence_partition_gbdt_v0"
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = CACHE_ROOT / "features"
FEATURE_LIST_PATH = CACHE_ROOT / "all_features.csv"
FOLD_PATH = CACHE_ROOT / "well_metadata.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CANDIDATES = [
    Path("/root/autodl-tmp/mainline_6p781_oof.parquet"),
    OOF_ROOT / "mainline_6p781_oof" / "mainline_6p781_oof.parquet",
    OOF_ROOT
    / "track8_stride_strong_nested_fusion_v4"
    / "best_nested_predictions.parquet",
]
REFERENCE_COLUMN_PREFERENCE = [
    "mainline_6p781",
    "mainline_tvt",
    "mainline",
    "prediction",
    "tvt_pred",
    "strong_proxy",
]
TRUTH_COLUMN_PREFERENCE = ["truth", "tvt_true", "true_tvt"]

SEED = 20260723
FOLDS = [1, 2, 3, 4, 5]
MAX_TREES = 2200
PATIENCE = 160
MIN_WELLS = 770
MIN_ALIGNMENT = 0.999
SMOOTH_WINDOW = 81
SMOOTH_POLY = 3
ANCHOR_TAU_FT = 55.0

BLEND_GRID = np.array(
    [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65],
    dtype=np.float64,
)
MAX_ENDPOINT_DETRENDED_RHO = 0.75
PASS_BLEND_GAIN_FT = 0.10
PASS_MIN_FOLD_GAIN_FT = -0.05
PASS_POSITIVE_FOLDS = 4
PASS_CONTROL_LOSS_FT = 0.03
PASS_MIN_SELECTED_WEIGHT = 0.02

# These tokens identify the current PF/GBDT/neighbor lineage.  Features in
# these families are excluded before any model matrix is built.
LINEAGE_BANNED_TOKENS = (
    "pf",
    "likpf",
    "beam",
    "sp45",
    "cp12",
    "cp_",
    "strong",
    "selector",
    "ridge",
    "gold",
    "neighbor",
    "neighbour",
    "knn",
    "spatial",
    "dense",
    "ancc",
    "formation",
    "form_",
    "fi_",
    "di_",
    "tvtf_",
    "tvtfw_",
    "tvtf50_",
    "frm_rmse_",
    "bw_",
    "bww_",
    "bw50_",
    "bw_early_",
    "bw_mid_",
    "sig_mean",
    "sig_std",
)

NON_FEATURE_EXACT = {
    "well",
    "id",
    "target",
    "tvt",
    "tvt_input",
    "_row_idx",
    "fold",
    "sample_weight",
}

# Absolute map coordinates are deliberately omitted.  Local trajectory
# derivatives remain legal; map-position evidence overlaps the neighbor route.
MAP_COORDINATE_EXACT = {"x", "y", "easting", "northing"}

TRAJECTORY_TOKENS = (
    "md",
    "z",
    "tvd",
    "inc",
    "incl",
    "azi",
    "azim",
    "dip",
    "slope",
    "trend",
    "curv",
    "dogleg",
    "dls",
    "progress",
    "since",
    "anchor",
    "last_known",
    "heel",
    "prefix",
    "dz",
    "dmd",
)

GR_REFERENCE_TOKENS = (
    "gr",
    "gamma",
    "typewell",
    "type_",
    "tw_",
    "corr",
    "xcorr",
    "ncc",
    "cos",
    "similar",
    "window",
    "rolling",
    "mean",
    "std",
    "median",
    "quant",
    "skew",
    "kurt",
    "range",
    "min",
    "max",
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _safe_rho(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    left = left[valid]
    right = right[valid]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _error_cosine(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _distance_detrend(error, md_since):
    error = np.asarray(error, np.float64)
    x = np.log1p(np.maximum(np.asarray(md_since, np.float64), 0.0))
    x /= max(float(np.nanmax(x)), 1.0)
    valid = np.isfinite(error) & np.isfinite(x)
    if int(valid.sum()) < 16:
        return error.copy()
    coefficients = np.polyfit(x[valid], error[valid], deg=3)
    return error - np.polyval(coefficients, x)


def _correlations(frame, candidate):
    truth = frame["truth"].to_numpy(np.float64)
    reference = frame["reference"].to_numpy(np.float64)
    candidate = np.asarray(candidate, np.float64)
    reference_error = reference - truth
    candidate_error = candidate - truth
    md_since = frame["md_since"].to_numpy(np.float64)
    return {
        "raw_signed_error_rho": _safe_rho(reference_error, candidate_error),
        "raw_error_cosine": _error_cosine(reference_error, candidate_error),
        "distance_detrended_error_rho": _safe_rho(
            _distance_detrend(reference_error, md_since),
            _distance_detrend(candidate_error, md_since),
        ),
    }


def _metrics(frame, prediction):
    prediction = np.asarray(prediction, np.float64)
    truth = frame["truth"].to_numpy(np.float64)
    error = prediction - truth
    codes, names = pd.factorize(frame["well"].astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error**2, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    distance = frame["md_since"].to_numpy(np.float64)

    def band(low, high=None):
        mask = distance >= low
        if high is not None:
            mask &= distance < high
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else float("nan")

    return {
        "rows": int(len(frame)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "near_0_500_rmse": band(0.0, 500.0),
        "mid_500_1000_rmse": band(500.0, 1000.0),
        "mid_1000_2000_rmse": band(1000.0, 2000.0),
        "far_gt2000_rmse": band(2000.0),
    }


def _matrix(frame, columns):
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _feature_paths(well):
    return FEATURE_DIR / f"{well}__sample.pkl", FEATURE_DIR / f"{well}__full.pkl"


def _load_reference():
    path = next((candidate for candidate in REFERENCE_CANDIDATES if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Upload mainline_6p781_oof.parquet or run the V4 strong-proxy export first"
        )
    table = pd.read_parquet(path)
    if "id" not in table:
        if "well" not in table or "row_idx" not in table:
            raise RuntimeError(f"Reference lacks id or well/row_idx: {table.columns.tolist()}")
        table["id"] = [
            f"{well}_{int(row)}" for well, row in zip(table["well"], table["row_idx"])
        ]
    table["id"] = table["id"].astype(str).str.lower()
    if "well" not in table:
        table["well"] = table["id"].str.rsplit("_", n=1).str[0]
    if "row_idx" not in table:
        table["row_idx"] = table["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    table["well"] = table["well"].astype(str).str.lower()
    table["row_idx"] = table["row_idx"].astype(np.int32)
    prediction_column = next(
        (column for column in REFERENCE_COLUMN_PREFERENCE if column in table), None
    )
    if prediction_column is None:
        raise RuntimeError(
            "Cannot identify reference prediction from columns: "
            + repr(table.columns.tolist())
        )
    truth_column = next(
        (column for column in TRUTH_COLUMN_PREFERENCE if column in table), None
    )
    keep = ["id", "well", "row_idx", prediction_column]
    if truth_column is not None and truth_column not in keep:
        keep.append(truth_column)
    if "fold" in table:
        keep.append("fold")
    table = table[keep].copy().rename(columns={prediction_column: "reference"})
    if truth_column is not None:
        table = table.rename(columns={truth_column: "reference_truth"})
    if table["id"].duplicated().any():
        raise RuntimeError("Reference OOF has duplicate ids")
    exact = prediction_column != "strong_proxy" and "6p781" in str(path).lower()
    return path, prediction_column, exact, table


def _attach_reference(frame, well):
    aligned = reference_groups[str(well)].reindex(frame["_row_idx"].to_numpy(np.int32))
    coverage = float(aligned["reference"].notna().mean())
    if coverage < MIN_ALIGNMENT:
        raise RuntimeError(f"{well}: reference coverage={coverage:.6%}")
    truth = (
        frame["target"].to_numpy(np.float64)
        + frame["last_known_tvt"].to_numpy(np.float64)
    )
    if "reference_truth" in aligned:
        gap = np.abs(truth - aligned["reference_truth"].to_numpy(np.float64))
        if not np.isfinite(gap).all() or float(np.max(gap)) > 0.01:
            raise RuntimeError(f"{well}: reference/cache truth gap={float(np.nanmax(gap)):.6f}")
    frame["reference"] = aligned["reference"].to_numpy(np.float32)
    return frame


def _load_frames(wells, kind):
    position = 0 if kind == "sample" else 1
    parts = []
    for well in wells:
        path = _feature_paths(str(well))[position]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path)
        frame["well"] = str(well)
        parts.append(_attach_reference(frame, str(well)))
    return pd.concat(parts, ignore_index=True, sort=False)


def _is_lineage_feature(column):
    lower = str(column).lower()
    return any(token in lower for token in LINEAGE_BANNED_TOKENS)


def _contains_any(column, tokens):
    lower = str(column).lower()
    padded = "_" + re.sub(r"[^a-z0-9]+", "_", lower).strip("_") + "_"
    for token in tokens:
        token = str(token).lower()
        if len(token) <= 2:
            if lower == token or f"_{token}_" in padded:
                return True
        elif token in lower:
            return True
    return False


def _partition_features(listed, schema):
    numeric = [
        column
        for column in listed
        if column in schema
        and pd.api.types.is_numeric_dtype(schema[column])
        and str(column).lower() not in NON_FEATURE_EXACT
    ]
    safe_local = [
        column
        for column in numeric
        if not _is_lineage_feature(column)
        and str(column).lower() not in MAP_COORDINATE_EXACT
    ]
    context = [
        column
        for column in safe_local
        if str(column).lower()
        in {"last_known_tvt", "md_since", "md", "z", "gr"}
    ]
    trajectory = [
        column for column in safe_local if _contains_any(column, TRAJECTORY_TOKENS)
    ]
    gr_reference = [
        column for column in safe_local if _contains_any(column, GR_REFERENCE_TOKENS)
    ]

    def unique(values):
        return list(dict.fromkeys(values))

    partitions = {
        "trajectory": unique(context + trajectory),
        "gr_reference": unique(context + gr_reference),
        "local_union": unique(safe_local),
    }
    partitions = {
        name: columns for name, columns in partitions.items() if len(columns) >= 8
    }
    if len(partitions) < 2:
        raise RuntimeError(
            "Feature partitioning produced too few arms: "
            + repr({name: len(columns) for name, columns in partitions.items()})
        )
    return numeric, safe_local, partitions


def _new_model(outer_fold, arm):
    arm_seed = sum(ord(character) for character in arm)
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=MAX_TREES,
        learning_rate=0.025,
        num_leaves=48,
        min_child_samples=120,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.20,
        reg_lambda=18.0,
        max_bin=255,
        n_jobs=-1,
        random_state=SEED + 100 * int(outer_fold) + arm_seed,
        verbosity=-1,
    )


def _fit_model(train, tune, columns, outer_fold, arm):
    model = _new_model(outer_fold, arm)
    x_train = _matrix(train, columns)
    x_tune = _matrix(tune, columns)
    y_train = train["target"].to_numpy(np.float32)
    y_tune = tune["target"].to_numpy(np.float32)
    sample_weight = train.get(
        "sample_weight", pd.Series(np.ones(len(train), np.float32))
    ).to_numpy(np.float32)
    fit_kwargs = dict(
        sample_weight=sample_weight,
        eval_set=[(x_tune, y_tune)],
        eval_metric="rmse",
        callbacks=[early_stopping(PATIENCE, verbose=False), log_evaluation(0)],
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
        print("LightGBM GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model.fit(x_train, y_train, **fit_kwargs)
    del x_train, x_tune
    gc.collect()
    return model


def _decode(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    md_since = frame["md_since"].to_numpy(np.float64)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / ANCHOR_TAU_FT)
    prediction = last + warm * np.asarray(model_delta, np.float64)
    output = prediction.copy()
    for _, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        order = np.argsort(frame.iloc[indices]["md_since"].to_numpy(np.float64))
        ordered = indices[order]
        values = prediction[ordered]
        window = min(SMOOTH_WINDOW, len(values))
        if window % 2 == 0:
            window -= 1
        if window >= SMOOTH_POLY + 2:
            output[ordered] = savgol_filter(values, window, SMOOTH_POLY)
    return output


def _selection_scan(selection, arm_columns):
    reference = selection["reference"].to_numpy(np.float64)
    truth = selection["truth"].to_numpy(np.float64)
    rows = []
    for arm in arm_columns:
        candidate = selection[arm].to_numpy(np.float64)
        endpoint_corr = _correlations(selection, candidate)
        for weight in BLEND_GRID:
            blend = reference + float(weight) * (candidate - reference)
            rows.append(
                {
                    "arm": arm,
                    "weight_candidate": float(weight),
                    "rmse": float(np.sqrt(np.mean((blend - truth) ** 2))),
                    **endpoint_corr,
                }
            )
    scan = pd.DataFrame(rows)
    allowed = scan[
        scan["distance_detrended_error_rho"].le(MAX_ENDPOINT_DETRENDED_RHO)
    ]
    if allowed.empty:
        # Weight zero is retained as the safe no-op when no endpoint is truly
        # decorrelated enough.
        selected = scan.loc[scan["weight_candidate"].eq(0.0)].iloc[0]
    else:
        selected = allowed.sort_values(
            ["rmse", "weight_candidate"], ascending=[True, True]
        ).iloc[0]
    return selected, scan


def _matched_direction_flip(frame, direction):
    rng = np.random.default_rng(SEED + 404)
    wells = frame["well"].astype(str).drop_duplicates().tolist()
    signs = {well: float(rng.choice([-1.0, 1.0])) for well in wells}
    multiplier = frame["well"].astype(str).map(signs).to_numpy(np.float64)
    return np.asarray(direction, np.float64) * multiplier


started = time.time()
for required in (FEATURE_DIR, FEATURE_LIST_PATH, FOLD_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

reference_path, reference_column, reference_is_exact, reference_table = _load_reference()
reference_groups = {
    well: group.set_index("row_idx").sort_index()
    for well, group in reference_table.groupby("well", sort=False)
}
fold_metadata = pd.read_csv(FOLD_PATH, usecols=["well", "fold"])
fold_metadata["well"] = fold_metadata["well"].astype(str).str.lower()
fold_metadata["fold"] = fold_metadata["fold"].astype(int)
fold_by_well = fold_metadata.drop_duplicates("well").set_index("well")["fold"]
eligible_wells = sorted(
    well
    for well in fold_by_well.index
    if well in reference_groups and all(path.exists() for path in _feature_paths(well))
)
if len(eligible_wells) < MIN_WELLS:
    raise RuntimeError(f"Only {len(eligible_wells)} wells have reference plus cache")
fold_counts = {
    fold: int(sum(int(fold_by_well[well]) == fold for well in eligible_wells))
    for fold in FOLDS
}
if min(fold_counts.values()) < 150:
    raise RuntimeError(f"Unexpected fold coverage: {fold_counts}")

listed_features = pd.read_csv(FEATURE_LIST_PATH)["feature"].astype(str).tolist()
schema = pd.read_pickle(_feature_paths(eligible_wells[0])[0])
numeric_features, safe_local_features, partitions = _partition_features(
    listed_features, schema
)
partition_report = pd.DataFrame(
    [
        {"arm": arm, "feature": feature}
        for arm, features in partitions.items()
        for feature in features
    ]
)
partition_report.to_csv(OUT_DIR / "feature_partitions.csv", index=False)

print(
    "DECORRELATED GBDT CONFIG",
    {
        "reference": str(reference_path),
        "reference_column": reference_column,
        "reference_is_exact_6p781": reference_is_exact,
        "wells": len(eligible_wells),
        "fold_counts": fold_counts,
        "numeric_features": len(numeric_features),
        "safe_local_features": len(safe_local_features),
        "partitions": {arm: len(features) for arm, features in partitions.items()},
    },
    flush=True,
)

oof_parts = []
model_rows = []
for outer_fold in FOLDS:
    tune_fold = FOLDS[outer_fold % len(FOLDS)]
    fit_wells = [
        well
        for well in eligible_wells
        if int(fold_by_well[well]) not in {outer_fold, tune_fold}
    ]
    tune_wells = [
        well for well in eligible_wells if int(fold_by_well[well]) == tune_fold
    ]
    valid_wells = [
        well for well in eligible_wells if int(fold_by_well[well]) == outer_fold
    ]
    print(
        f"\nouter={outer_fold} fit={len(fit_wells)} tune={len(tune_wells)} "
        f"valid={len(valid_wells)}",
        flush=True,
    )
    train = _load_frames(fit_wells, "sample")
    tune = _load_frames(tune_wells, "sample")
    valid = _load_frames(valid_wells, "full")
    truth = valid["target"].to_numpy(np.float64) + valid["last_known_tvt"].to_numpy(np.float64)
    part = pd.DataFrame(
        {
            "id": valid["id"].astype(str).str.lower().to_numpy(),
            "well": valid["well"].astype(str).str.lower().to_numpy(),
            "row_idx": valid["_row_idx"].to_numpy(np.int32),
            "fold": int(outer_fold),
            "truth": truth,
            "md_since": valid["md_since"].to_numpy(np.float32),
            "reference": valid["reference"].to_numpy(np.float32),
        }
    )
    for arm, features in partitions.items():
        model = _fit_model(train, tune, features, outer_fold, arm)
        delta = model.predict(
            _matrix(valid, features), num_iteration=model.best_iteration_
        )
        part[arm] = _decode(valid, delta).astype(np.float32)
        model_rows.append(
            {
                "outer_fold": int(outer_fold),
                "tune_fold": int(tune_fold),
                "arm": arm,
                "features": len(features),
                "best_iteration": int(model.best_iteration_),
            }
        )
        joblib.dump(model, MODEL_DIR / f"{arm}_fold{outer_fold}.joblib")
    oof_parts.append(part)
    del train, tune, valid, part
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
if oof["id"].duplicated().any() or oof["well"].nunique() < MIN_WELLS:
    raise RuntimeError("Invalid model OOF coverage")

arm_columns = list(partitions)
endpoint_rows = []
for arm in arm_columns:
    endpoint_rows.append(
        {
            "arm": arm,
            **_metrics(oof, oof[arm]),
            **_correlations(oof, oof[arm]),
        }
    )
endpoint_report = pd.DataFrame(endpoint_rows).sort_values("rmse", ignore_index=True)

nested_blend = np.empty(len(oof), dtype=np.float64)
selected_candidate = np.empty(len(oof), dtype=np.float64)
fold_rows = []
scan_parts = []
for held_fold in FOLDS:
    selection = oof.loc[oof["fold"].ne(held_fold)]
    held_mask = oof["fold"].eq(held_fold).to_numpy()
    held = oof.loc[held_mask]
    selected, scan = _selection_scan(selection, arm_columns)
    scan.insert(0, "held_fold", int(held_fold))
    scan_parts.append(scan)
    arm = str(selected["arm"])
    weight = float(selected["weight_candidate"])
    candidate = held[arm].to_numpy(np.float64)
    reference = held["reference"].to_numpy(np.float64)
    blend = reference + weight * (candidate - reference)
    selected_candidate[held_mask] = candidate
    nested_blend[held_mask] = blend
    reference_rmse = _metrics(held, reference)["rmse"]
    blend_rmse = _metrics(held, blend)["rmse"]
    fold_rows.append(
        {
            "fold": int(held_fold),
            "selected_arm": arm,
            "weight_candidate": weight,
            "reference_rmse": reference_rmse,
            "candidate_rmse": _metrics(held, candidate)["rmse"],
            "blend_rmse": blend_rmse,
            "gain": reference_rmse - blend_rmse,
            **_correlations(held, candidate),
        }
    )

oof["selected_candidate"] = selected_candidate.astype(np.float32)
oof["nested_blend"] = nested_blend.astype(np.float32)
fold_report = pd.DataFrame(fold_rows)
weight_by_fold = fold_report.set_index("fold")["weight_candidate"].to_dict()
direction = oof["selected_candidate"].to_numpy(np.float64) - oof["reference"].to_numpy(np.float64)
oof["matched_direction_flip"] = (
    oof["reference"].to_numpy(np.float64)
    + oof["fold"].map(weight_by_fold).to_numpy(np.float64)
    * _matched_direction_flip(oof, direction)
).astype(np.float32)

reference_metrics = _metrics(oof, oof["reference"])
candidate_metrics = _metrics(oof, oof["selected_candidate"])
blend_metrics = _metrics(oof, oof["nested_blend"])
control_metrics = _metrics(oof, oof["matched_direction_flip"])
candidate_correlations = _correlations(oof, oof["selected_candidate"])
blend_gain = reference_metrics["rmse"] - blend_metrics["rmse"]
control_loss = control_metrics["rmse"] - blend_metrics["rmse"]
selected_counts = fold_report["selected_arm"].value_counts().to_dict()

checks = {
    "selected_endpoint_detrended_rho_le_0p75": candidate_correlations[
        "distance_detrended_error_rho"
    ]
    <= MAX_ENDPOINT_DETRENDED_RHO,
    "nested_blend_gain_ge_0p10": blend_gain >= PASS_BLEND_GAIN_FT,
    "minimum_fold_gain_ge_minus0p05": float(fold_report["gain"].min())
    >= PASS_MIN_FOLD_GAIN_FT,
    "positive_folds_ge_4": int((fold_report["gain"] > 0.0).sum())
    >= PASS_POSITIVE_FOLDS,
    "minimum_selected_weight_ge_0p02": float(fold_report["weight_candidate"].min())
    >= PASS_MIN_SELECTED_WEIGHT,
    "matched_control_worse_by_ge_0p03": control_loss >= PASS_CONTROL_LOSS_FT,
    "same_arm_selected_in_at_least_3_folds": max(selected_counts.values()) >= 3,
}
strict_pass = bool(all(checks.values()))

oof.to_parquet(OUT_DIR / "decorrelated_gbdt_oof.parquet", index=False, compression="zstd")
endpoint_report.to_csv(OUT_DIR / "endpoint_report.csv", index=False)
fold_report.to_csv(OUT_DIR / "nested_fold_report.csv", index=False)
pd.concat(scan_parts, ignore_index=True).to_csv(OUT_DIR / "nested_blend_scan.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_disjoint_evidence_gbdt_endpoint_and_nested_blend_gateway",
    "reference": {
        "path": str(reference_path),
        "column": reference_column,
        "is_exact_6p781_oof": reference_is_exact,
    },
    "protocol": {
        "sample_unit": "complete physical well",
        "outer_folds": FOLDS,
        "inner_early_stopping_fold": "cyclic next fold, excluded from fit",
        "reference_used_as_model_input": False,
        "reference_used_as_training_label": False,
        "blend_selection": "other four OOF folds only",
    },
    "feature_partitions": {arm: len(features) for arm, features in partitions.items()},
    "banned_lineage_tokens": list(LINEAGE_BANNED_TOKENS),
    "endpoint_report": endpoint_report.to_dict(orient="records"),
    "reference_metrics": reference_metrics,
    "selected_candidate_metrics": candidate_metrics,
    "nested_blend_metrics": blend_metrics,
    "matched_direction_flip_metrics": control_metrics,
    "selected_candidate_correlations": candidate_correlations,
    "nested_blend_gain": blend_gain,
    "matched_control_loss": control_loss,
    "selected_arm_counts": selected_counts,
    "folds": fold_report.to_dict(orient="records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS proxy -> repeat against exact 6.781 OOF, then full refit"
        if strict_pass and not reference_is_exact
        else "PASS exact mainline -> full refit and deployment blend"
        if strict_pass
        else "FAIL -> these disjoint tabular feature partitions do not provide a useful independent endpoint"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nDECORRELATED ENDPOINTS")
print(endpoint_report.to_string(index=False), flush=True)
print("\nNESTED FOLD REPORT")
print(fold_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
