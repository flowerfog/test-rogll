"""Segment-surface GBDT endpoint and nested blend gateway.

The row-level feature-partition frontier was tangent to the current mainline:
accuracy and error correlation increased at essentially the same rate.  This
cell changes the sample unit and target instead of tuning the same row model.

For each complete well and each 128/256 ft segment:

1. aggregate deployable row features within the segment;
2. predict the segment rate of surface = TVT + Z;
3. integrate predicted rates from the known heel anchor;
4. subtract the exactly known Z path to recover row-level TVT.

The mainline/strong proxy is used only for OOF correlation measurement and
nested blend selection.  It is never a GBDT input or training label.
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


warnings.filterwarnings("ignore")

EXPERIMENT = "segment_surface_gbdt_frontier_v2"
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
SEGMENT_LENGTHS_FT = [128.0, 256.0]
RATE_SCALE = 100.0
MAX_TREES = 1800
PATIENCE = 140
MIN_WELLS = 770
MIN_ALIGNMENT = 0.999
MAX_MOMENT_FEATURES = 56

BLEND_GRID = np.array(
    [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65],
    dtype=np.float64,
)
ORACLE_GATE_RMSE = 1.50
MAX_ENDPOINT_DETRENDED_RHO = 0.80
MIN_BLENDABILITY_MARGIN = 0.01
PASS_BLEND_GAIN_FT = 0.10
PASS_STANDALONE_RMSE = 11.50
PASS_MIN_FOLD_GAIN_FT = -0.05
PASS_POSITIVE_FOLDS = 4
PASS_CONTROL_LOSS_FT = 0.03
PASS_MIN_SELECTED_WEIGHT = 0.02

NON_FEATURE_EXACT = {
    "well",
    "id",
    "target",
    "tvt",
    "tvt_input",
    "_row_idx",
    "fold",
    "sample_weight",
    "truth",
    "true_tvt",
    "tvt_true",
    "label",
}
MAP_COORDINATE_EXACT = {"x", "y", "easting", "northing"}

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
    "sig_mean",
    "sig_std",
)

UNCERTAINTY_TOKENS = (
    "std",
    "sigma",
    "entropy",
    "uncert",
    "spread",
    "dispersion",
    "range",
    "gap",
    "margin",
    "confidence",
    "score",
    "rank",
    "iqr",
    "p05",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
)

MOMENT_TOKENS = (
    "gr",
    "gamma",
    "md",
    "z",
    "tvd",
    "inc",
    "azi",
    "dip",
    "slope",
    "trend",
    "curv",
    "corr",
    "ncc",
    "type",
    "prefix",
    "heel",
    "progress",
    "since",
    *UNCERTAINTY_TOKENS,
)

DIRECT_ENDPOINT_PATTERNS = (
    r"(^|_)cp(12)?($|_)",
    r"strong",
    r"selector",
    r"ridge",
    r"gold",
    r"neighbor",
    r"neighbour",
    r"sp45",
    r"^pf_ancc$",
    r"^likpf_(scale|mean)",
    r"^beam_.*(_d|prediction|tvt)$",
    r"^tvtf",
    r"^tvt_dense",
    r"^spatial_ancc",
)

CONTEXT_FEATURES = [
    "seg_start_ft",
    "seg_end_ft",
    "seg_length_ft",
    "seg_rows",
    "seg_index",
    "log1p_seg_start",
    "sqrt_seg_start",
]


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
    reference_error = frame["reference"].to_numpy(np.float64) - truth
    candidate_error = np.asarray(candidate, np.float64) - truth
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


def _feature_paths(well):
    return FEATURE_DIR / f"{well}__sample.pkl", FEATURE_DIR / f"{well}__full.pkl"


def _matrix(frame, columns):
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


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


def _is_lineage_feature(column):
    lower = str(column).lower()
    return any(token in lower for token in LINEAGE_BANNED_TOKENS)


def _is_direct_endpoint(column):
    lower = str(column).lower()
    hard_tokens = (
        "cp12",
        "strong",
        "selector",
        "ridge",
        "gold",
        "neighbor",
        "neighbour",
        "sp45",
    )
    if any(token in lower for token in hard_tokens):
        return True
    if _contains_any(column, UNCERTAINTY_TOKENS):
        return False
    return any(re.search(pattern, lower) is not None for pattern in DIRECT_ENDPOINT_PATTERNS)


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
            "Cannot identify reference prediction: " + repr(table.columns.tolist())
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
        raise RuntimeError("Reference OOF contains duplicate ids")
    exact = prediction_column != "strong_proxy" and "6p781" in str(path).lower()
    return path, prediction_column, exact, table


def _feature_partitions(listed, schema):
    numeric = [
        column
        for column in listed
        if column in schema
        and pd.api.types.is_numeric_dtype(schema[column])
        and str(column).lower() not in NON_FEATURE_EXACT
    ]
    local = [
        column
        for column in numeric
        if not _is_lineage_feature(column)
        and str(column).lower() not in MAP_COORDINATE_EXACT
    ]
    broad = [
        column
        for column in numeric
        if not _is_direct_endpoint(column)
        and str(column).lower() not in MAP_COORDINATE_EXACT
    ]
    moment = [
        column for column in broad if _contains_any(column, MOMENT_TOKENS)
    ][:MAX_MOMENT_FEATURES]
    if len(local) < 20 or len(broad) < 40:
        raise RuntimeError(f"Too few features: local={len(local)} broad={len(broad)}")
    return numeric, local, broad, moment


def _anchor_z(md_since, z):
    md_since = np.asarray(md_since, np.float64)
    z = np.asarray(z, np.float64)
    take = np.flatnonzero(np.isfinite(md_since) & np.isfinite(z) & (md_since <= 64.0))
    if len(take) >= 4 and float(np.ptp(md_since[take])) > 1e-6:
        coefficients = np.polyfit(md_since[take], z[take], deg=1)
        return float(np.polyval(coefficients, 0.0))
    return float(z[np.flatnonzero(np.isfinite(z))[0]])


def _aggregate(values):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    safe = np.where(finite, values, 0.0)
    mean = safe.sum(axis=0) / np.maximum(counts, 1)
    centred = np.where(finite, values - mean, 0.0)
    std = np.sqrt((centred**2).sum(axis=0) / np.maximum(counts, 1))
    first = np.zeros(values.shape[1], np.float64)
    last = np.zeros(values.shape[1], np.float64)
    for column in range(values.shape[1]):
        valid = np.flatnonzero(finite[:, column])
        if len(valid):
            first[column] = values[valid[0], column]
            last[column] = values[valid[-1], column]
    return mean, std, last - first


def _build_segment_bank(wells, fold_by_well, reference_groups, broad_features, moment_features):
    row_parts = []
    segment_records = {length: [] for length in SEGMENT_LENGTHS_FT}
    next_uid = {length: 0 for length in SEGMENT_LENGTHS_FT}
    moment_positions = [broad_features.index(column) for column in moment_features]
    for number, well in enumerate(wells, start=1):
        _, path = _feature_paths(well)
        frame = pd.read_pickle(path).reset_index(drop=True)
        frame["well"] = well
        if "Z" in frame:
            z_column = "Z"
        elif "z" in frame:
            z_column = "z"
        else:
            raise RuntimeError(f"{well}: feature cache has no Z column")
        row_idx = frame["_row_idx"].to_numpy(np.int32)
        aligned = reference_groups[well].reindex(row_idx)
        coverage = float(aligned["reference"].notna().mean())
        if coverage < MIN_ALIGNMENT:
            raise RuntimeError(f"{well}: reference coverage={coverage:.6%}")
        last_known = frame["last_known_tvt"].to_numpy(np.float64)
        truth = frame["target"].to_numpy(np.float64) + last_known
        if "reference_truth" in aligned:
            truth_gap = np.abs(
                aligned["reference_truth"].to_numpy(np.float64) - truth
            )
            if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
                raise RuntimeError(
                    f"{well}: reference truth mismatch max={float(np.nanmax(truth_gap))}"
                )
        z = frame[z_column].to_numpy(np.float64)
        md_since = frame["md_since"].to_numpy(np.float64)
        surface = truth + z
        anchor_surface = float(last_known[0] + _anchor_z(md_since, z))
        feature_values = _matrix(frame, broad_features).astype(np.float64)
        row_part = pd.DataFrame(
            {
                "id": frame["id"].astype(str).str.lower().to_numpy(),
                "well": well,
                "row_idx": row_idx,
                "fold": int(fold_by_well[well]),
                "truth": truth.astype(np.float32),
                "z": z.astype(np.float32),
                "md_since": md_since.astype(np.float32),
                "reference": aligned["reference"].to_numpy(np.float32),
            }
        )
        for length in SEGMENT_LENGTHS_FT:
            bins = np.floor(np.maximum(md_since - 1e-6, 0.0) / length).astype(np.int32)
            uids = np.empty(len(frame), np.int32)
            previous_x = 0.0
            previous_surface = anchor_surface
            for segment_index, bin_value in enumerate(np.unique(bins)):
                indices = np.flatnonzero(bins == bin_value)
                if not len(indices):
                    continue
                uid = next_uid[length]
                next_uid[length] += 1
                uids[indices] = uid
                end_index = int(indices[-1])
                end_x = float(md_since[end_index])
                delta_x = max(end_x - previous_x, 1e-3)
                end_surface = float(surface[end_index])
                target_rate = (end_surface - previous_surface) / delta_x
                mean, std, delta = _aggregate(feature_values[indices])
                record = {
                    "segment_uid": uid,
                    "well": well,
                    "fold": int(fold_by_well[well]),
                    "seg_index": int(segment_index),
                    "seg_start_ft": previous_x,
                    "seg_end_ft": end_x,
                    "seg_length_ft": delta_x,
                    "seg_rows": int(len(indices)),
                    "log1p_seg_start": float(np.log1p(previous_x)),
                    "sqrt_seg_start": float(np.sqrt(max(previous_x, 0.0))),
                    "anchor_surface": anchor_surface,
                    "target_rate": target_rate,
                }
                for position, column in enumerate(broad_features):
                    record[f"mean__{column}"] = float(mean[position])
                for position in moment_positions:
                    column = broad_features[position]
                    record[f"std__{column}"] = float(std[position])
                    record[f"delta__{column}"] = float(delta[position])
                segment_records[length].append(record)
                previous_x = end_x
                previous_surface = end_surface
            row_part[f"segment_uid_{int(length)}"] = uids
        row_parts.append(row_part)
        if number % 50 == 0 or number == len(wells):
            print(f"build segment bank {number}/{len(wells)}", flush=True)
        del frame, feature_values
        if number % 100 == 0:
            gc.collect()
    rows = pd.concat(row_parts, ignore_index=True)
    segments = {
        length: pd.DataFrame(records).sort_values("segment_uid").reset_index(drop=True)
        for length, records in segment_records.items()
    }
    return rows, segments


def _segment_feature_columns(base_features, moment_features):
    return (
        CONTEXT_FEATURES
        + [f"mean__{column}" for column in base_features]
        + [f"std__{column}" for column in moment_features if column in base_features]
        + [f"delta__{column}" for column in moment_features if column in base_features]
    )


def _new_model(outer_fold, arm):
    arm_seed = sum(ord(character) for character in arm)
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=MAX_TREES,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.25,
        reg_lambda=20.0,
        max_bin=255,
        n_jobs=-1,
        random_state=SEED + 100 * int(outer_fold) + arm_seed,
        verbosity=-1,
    )


def _fit_model(train, tune, columns, outer_fold, arm):
    model = _new_model(outer_fold, arm)
    x_train = _matrix(train, columns)
    x_tune = _matrix(tune, columns)
    y_train = (train["target_rate"].to_numpy(np.float32) * RATE_SCALE)
    y_tune = (tune["target_rate"].to_numpy(np.float32) * RATE_SCALE)
    weights = train["seg_rows"].to_numpy(np.float32)
    fit_kwargs = dict(
        sample_weight=weights,
        eval_set=[(x_tune, y_tune)],
        eval_sample_weight=[tune["seg_rows"].to_numpy(np.float32)],
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


def _smooth_rates(segment_frame, rates):
    output = np.asarray(rates, np.float64).copy()
    for _, indices in segment_frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        order = np.argsort(segment_frame.iloc[indices]["seg_index"].to_numpy())
        ordered = indices[order]
        values = output[ordered]
        if len(values) >= 3:
            padded = np.pad(values, (1, 1), mode="edge")
            output[ordered] = (
                0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
            )
    return output


def _decode_rates(rows, segments, length, rates):
    rates = np.asarray(rates, np.float64)
    if len(rates) != len(segments) or not np.isfinite(rates).all():
        raise RuntimeError(f"Invalid rate vector for {length}")
    rate_by_uid = pd.Series(rates, index=segments["segment_uid"].to_numpy(np.int32))
    anchor_by_well = segments.groupby("well")["anchor_surface"].first()
    output = np.empty(len(rows), np.float64)
    uid_column = f"segment_uid_{int(length)}"
    for well, indices in rows.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        order = np.argsort(rows.iloc[indices]["md_since"].to_numpy(np.float64))
        ordered = indices[order]
        well_rows = rows.iloc[ordered]
        current_surface = float(anchor_by_well[well])
        previous_x = 0.0
        for uid in well_rows[uid_column].drop_duplicates().astype(int):
            local = np.flatnonzero(well_rows[uid_column].to_numpy(np.int32) == uid)
            global_indices = ordered[local]
            x = rows.iloc[global_indices]["md_since"].to_numpy(np.float64)
            rate = float(rate_by_uid.loc[uid])
            surface = current_surface + rate * (x - previous_x)
            output[global_indices] = surface - rows.iloc[global_indices]["z"].to_numpy(np.float64)
            end_x = float(np.max(x))
            current_surface += rate * (end_x - previous_x)
            previous_x = end_x
    return output


def _selection_scan(selection, arms):
    reference = selection["reference"].to_numpy(np.float64)
    truth = selection["truth"].to_numpy(np.float64)
    reference_rmse = float(np.sqrt(np.mean((reference - truth) ** 2)))
    records = []
    for arm in arms:
        candidate = selection[arm].to_numpy(np.float64)
        candidate_rmse = float(np.sqrt(np.mean((candidate - truth) ** 2)))
        corr = _correlations(selection, candidate)
        limit = reference_rmse / max(candidate_rmse, 1e-12)
        margin = limit - corr["distance_detrended_error_rho"]
        for weight in BLEND_GRID:
            blend = reference + float(weight) * (candidate - reference)
            records.append(
                {
                    "arm": arm,
                    "weight_candidate": float(weight),
                    "rmse": float(np.sqrt(np.mean((blend - truth) ** 2))),
                    "candidate_rmse": candidate_rmse,
                    "blendability_limit": limit,
                    "blendability_margin": margin,
                    **corr,
                }
            )
    scan = pd.DataFrame(records)
    allowed = scan[
        scan["distance_detrended_error_rho"].le(MAX_ENDPOINT_DETRENDED_RHO)
        & scan["blendability_margin"].ge(MIN_BLENDABILITY_MARGIN)
    ]
    if allowed.empty:
        selected = scan.loc[scan["weight_candidate"].eq(0.0)].iloc[0]
    else:
        selected = allowed.sort_values(
            ["rmse", "weight_candidate"], ascending=[True, True]
        ).iloc[0]
    return selected, scan


def _matched_direction_flip(frame, direction):
    rng = np.random.default_rng(SEED + 808)
    wells = frame["well"].astype(str).drop_duplicates().tolist()
    signs = np.ones(len(wells), np.float64)
    signs[: len(wells) // 2] = -1.0
    rng.shuffle(signs)
    mapping = dict(zip(wells, signs))
    return np.asarray(direction, np.float64) * frame["well"].map(mapping).to_numpy(np.float64)


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
    if well in reference_groups and _feature_paths(well)[1].exists()
)
if len(eligible_wells) < MIN_WELLS:
    raise RuntimeError(f"Only {len(eligible_wells)} wells have reference plus cache")

listed_features = pd.read_csv(FEATURE_LIST_PATH)["feature"].astype(str).tolist()
schema = pd.read_pickle(_feature_paths(eligible_wells[0])[1])
numeric_features, local_features, broad_features, moment_features = _feature_partitions(
    listed_features, schema
)
local_segment_features = _segment_feature_columns(local_features, moment_features)
broad_segment_features = _segment_feature_columns(broad_features, moment_features)

print(
    "SEGMENT-SURFACE CONFIG",
    {
        "reference": str(reference_path),
        "reference_column": reference_column,
        "reference_is_exact_6p781": reference_is_exact,
        "wells": len(eligible_wells),
        "numeric_features": len(numeric_features),
        "local_features": len(local_features),
        "broad_features": len(broad_features),
        "moment_features": len(moment_features),
        "segment_lengths_ft": SEGMENT_LENGTHS_FT,
    },
    flush=True,
)

rows, segment_tables = _build_segment_bank(
    eligible_wells,
    fold_by_well,
    reference_groups,
    broad_features,
    moment_features,
)
if rows["id"].duplicated().any() or rows["well"].nunique() < MIN_WELLS:
    raise RuntimeError("Invalid row bank")

oracle_rows = []
for length, segments in segment_tables.items():
    oracle_prediction = _decode_rates(
        rows, segments, length, segments["target_rate"].to_numpy(np.float64)
    )
    oracle_rows.append(
        {"segment_length_ft": length, **_metrics(rows, oracle_prediction)}
    )
oracle_report = pd.DataFrame(oracle_rows).sort_values("rmse", ignore_index=True)
oracle_report.to_csv(OUT_DIR / "representation_oracle.csv", index=False)
print("\nSEGMENT REPRESENTATION ORACLE")
print(oracle_report.to_string(index=False), flush=True)
if float(oracle_report["rmse"].min()) > ORACLE_GATE_RMSE:
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "segment_surface_representation_oracle",
        "representation_oracle": oracle_report.to_dict(orient="records"),
        "strict_pass": False,
        "next": "FAIL -> segment representation is too coarse; stop before training",
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    raise RuntimeError(json.dumps(verdict, indent=2, default=_json_default))

arm_specs = {}
for length in SEGMENT_LENGTHS_FT:
    arm_specs[f"s{int(length)}_local"] = (length, local_segment_features)
    arm_specs[f"s{int(length)}_broad"] = (length, broad_segment_features)

rate_oof = {
    arm: np.full(len(segment_tables[length]), np.nan, np.float64)
    for arm, (length, _) in arm_specs.items()
}
model_rows = []
for outer_fold in FOLDS:
    tune_fold = FOLDS[outer_fold % len(FOLDS)]
    print(f"\nsegment models outer={outer_fold} tune={tune_fold}", flush=True)
    for arm, (length, features) in arm_specs.items():
        segments = segment_tables[length]
        fit = segments.loc[~segments["fold"].isin([outer_fold, tune_fold])]
        tune = segments.loc[segments["fold"].eq(tune_fold)]
        valid_mask = segments["fold"].eq(outer_fold).to_numpy()
        valid = segments.loc[valid_mask].reset_index(drop=True)
        model = _fit_model(fit, tune, features, outer_fold, arm)
        predicted = model.predict(
            _matrix(valid, features), num_iteration=model.best_iteration_
        ).astype(np.float64) / RATE_SCALE
        low, high = np.quantile(fit["target_rate"].to_numpy(np.float64), [0.005, 0.995])
        padding = max(0.005, 0.10 * float(high - low))
        predicted = np.clip(predicted, low - padding, high + padding)
        predicted = _smooth_rates(valid, predicted)
        rate_oof[arm][valid_mask] = predicted
        model_rows.append(
            {
                "outer_fold": outer_fold,
                "tune_fold": tune_fold,
                "arm": arm,
                "segments_fit": len(fit),
                "segments_tune": len(tune),
                "segments_valid": len(valid),
                "features": len(features),
                "best_iteration": int(model.best_iteration_),
                "rate_clip_low": float(low - padding),
                "rate_clip_high": float(high + padding),
            }
        )
        joblib.dump(model, MODEL_DIR / f"{arm}_fold{outer_fold}.joblib")
        del fit, tune, valid, model
        gc.collect()

for arm, (length, _) in arm_specs.items():
    if not np.isfinite(rate_oof[arm]).all():
        raise RuntimeError(f"Incomplete rate OOF for {arm}")
    rows[arm] = _decode_rates(rows, segment_tables[length], length, rate_oof[arm]).astype(
        np.float32
    )

endpoint_rows = []
reference_rmse = _metrics(rows, rows["reference"])["rmse"]
for arm in arm_specs:
    metrics = _metrics(rows, rows[arm])
    corr = _correlations(rows, rows[arm])
    limit = reference_rmse / max(metrics["rmse"], 1e-12)
    endpoint_rows.append(
        {
            "arm": arm,
            **metrics,
            **corr,
            "blendability_limit": limit,
            "blendability_margin": limit - corr["distance_detrended_error_rho"],
        }
    )
endpoint_report = pd.DataFrame(endpoint_rows).sort_values("rmse", ignore_index=True)

arms = list(arm_specs)
nested_blend = np.empty(len(rows), np.float64)
selected_candidate = np.empty(len(rows), np.float64)
fold_rows = []
scan_parts = []
for held_fold in FOLDS:
    selection = rows.loc[rows["fold"].ne(held_fold)]
    held_mask = rows["fold"].eq(held_fold).to_numpy()
    held = rows.loc[held_mask]
    selected, scan = _selection_scan(selection, arms)
    scan.insert(0, "held_fold", held_fold)
    scan_parts.append(scan)
    arm = str(selected["arm"])
    weight = float(selected["weight_candidate"])
    candidate = held[arm].to_numpy(np.float64)
    reference = held["reference"].to_numpy(np.float64)
    blend = reference + weight * (candidate - reference)
    selected_candidate[held_mask] = candidate
    nested_blend[held_mask] = blend
    base_rmse = _metrics(held, reference)["rmse"]
    blend_rmse = _metrics(held, blend)["rmse"]
    fold_rows.append(
        {
            "fold": held_fold,
            "selected_arm": arm,
            "weight_candidate": weight,
            "reference_rmse": base_rmse,
            "candidate_rmse": _metrics(held, candidate)["rmse"],
            "blend_rmse": blend_rmse,
            "gain": base_rmse - blend_rmse,
            **_correlations(held, candidate),
        }
    )

rows["selected_candidate"] = selected_candidate.astype(np.float32)
rows["nested_blend"] = nested_blend.astype(np.float32)
fold_report = pd.DataFrame(fold_rows)
weights = fold_report.set_index("fold")["weight_candidate"].to_dict()
direction = selected_candidate - rows["reference"].to_numpy(np.float64)
rows["matched_direction_flip"] = (
    rows["reference"].to_numpy(np.float64)
    + rows["fold"].map(weights).to_numpy(np.float64)
    * _matched_direction_flip(rows, direction)
).astype(np.float32)

reference_metrics = _metrics(rows, rows["reference"])
candidate_metrics = _metrics(rows, rows["selected_candidate"])
blend_metrics = _metrics(rows, rows["nested_blend"])
control_metrics = _metrics(rows, rows["matched_direction_flip"])
candidate_correlations = _correlations(rows, rows["selected_candidate"])
blendability_limit = reference_metrics["rmse"] / max(candidate_metrics["rmse"], 1e-12)
blendability_margin = (
    blendability_limit - candidate_correlations["distance_detrended_error_rho"]
)
blend_gain = reference_metrics["rmse"] - blend_metrics["rmse"]
control_loss = control_metrics["rmse"] - blend_metrics["rmse"]
selected_counts = fold_report["selected_arm"].value_counts().to_dict()

checks = {
    "representation_oracle_rmse_le_1p5": float(oracle_report["rmse"].min())
    <= ORACLE_GATE_RMSE,
    "selected_endpoint_detrended_rho_le_0p80": candidate_correlations[
        "distance_detrended_error_rho"
    ]
    <= MAX_ENDPOINT_DETRENDED_RHO,
    "selected_blendability_margin_ge_0p01": blendability_margin
    >= MIN_BLENDABILITY_MARGIN,
    "selected_standalone_rmse_le_11p5": candidate_metrics["rmse"]
    <= PASS_STANDALONE_RMSE,
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

output_columns = [
    "id",
    "well",
    "row_idx",
    "fold",
    "truth",
    "md_since",
    "reference",
    *arms,
    "selected_candidate",
    "nested_blend",
    "matched_direction_flip",
]
rows[output_columns].to_parquet(
    OUT_DIR / "segment_surface_gbdt_oof.parquet", index=False, compression="zstd"
)
endpoint_report.to_csv(OUT_DIR / "endpoint_report.csv", index=False)
fold_report.to_csv(OUT_DIR / "nested_fold_report.csv", index=False)
pd.concat(scan_parts, ignore_index=True).to_csv(OUT_DIR / "nested_blend_scan.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)
pd.DataFrame(
    [
        {"family": "local", "feature": feature}
        for feature in local_segment_features
    ]
    + [
        {"family": "broad", "feature": feature}
        for feature in broad_segment_features
    ]
).to_csv(OUT_DIR / "segment_features.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_segment_surface_rate_gbdt_and_nested_blend_gateway",
    "reference": {
        "path": str(reference_path),
        "column": reference_column,
        "is_exact_6p781_oof": reference_is_exact,
    },
    "protocol": {
        "sample_unit": "fixed-length segment nested within complete physical well",
        "target": "segment rate of surface = TVT + Z",
        "decode": "integrate segment rates from known heel anchor, then subtract Z",
        "reference_used_as_model_input": False,
        "reference_used_as_training_label": False,
        "outer_folds": FOLDS,
        "inner_early_stopping_fold": "cyclic next fold, excluded from fit",
        "blend_selection": "other four OOF folds only",
    },
    "feature_counts": {
        "row_local": len(local_features),
        "row_broad": len(broad_features),
        "moments": len(moment_features),
        "segment_local": len(local_segment_features),
        "segment_broad": len(broad_segment_features),
    },
    "representation_oracle": oracle_report.to_dict(orient="records"),
    "endpoint_report": endpoint_report.to_dict(orient="records"),
    "reference_metrics": reference_metrics,
    "selected_candidate_metrics": candidate_metrics,
    "nested_blend_metrics": blend_metrics,
    "matched_direction_flip_metrics": control_metrics,
    "selected_candidate_correlations": candidate_correlations,
    "selected_blendability_limit": blendability_limit,
    "selected_blendability_margin": blendability_margin,
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
        else "FAIL -> close low-correlation GBDT; segment target did not create useful independence"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSEGMENT-SURFACE ENDPOINTS")
print(endpoint_report.to_string(index=False), flush=True)
print("\nNESTED FOLD REPORT")
print(fold_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
