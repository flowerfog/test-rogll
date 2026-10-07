"""V198: corrected multiengine correlation and soft-fusion audit.

This read-only/full-row experiment fixes the V187 STRIDE artifact locator,
reports raw, uncentred and distance-detrended error dependence against V168,
and evaluates two legal five-fold soft consumers:

1. non-negative global residual simplex weights trained on the other folds;
2. distance-band inverse-MSE posterior means, with and without a truth-free
   robust centrality adjustment.

No row/block endpoint is selected. Truth is used for diagnostics and for
outer-training-fold reliability/weights only; held-fold truth is metrics only.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / "canonical_multiengine_soft_fusion_audit_v198"
OUT.mkdir(parents=True, exist_ok=True)

BASE_PATH = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
BASE_COLUMN = "metadata_analog_w0p05"
BASE_NAME = "v168/metadata_analog_w0p05"
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
EXPECTED_BASE_SHA = "43f23cb1f53212e8da77b38e873b24ae53ddd092a8f2fa16fa761e7a1536e8c6"
DISTANCE_EDGES = np.array([0.0, 500.0, 1000.0, 2000.0, np.inf], np.float64)
DISTANCE_NAMES = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")

# alias, path, source prediction columns. The V155 entry intentionally points
# to the file and column actually written by stride_independent_engine_v155.
SOURCES = (
    ("v169", OOF / "historical_analog_ranked_consensus_v169" / "fullrow_oof.parquet", ("nested_consensus",)),
    ("v144", OOF / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet",
     ("consensus_center", "nested_sequence_coefficient", "exact_v76", "coarse_crossfit")),
    ("v66", OOF / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
     ("incumbent_6p377_replay", "full_cp8_ordered")),
    ("v96", OOF / "rawdata_lattice_v76_full773_training_v96" / "fivefold_coarse_oof.parquet",
     ("coarse_crossfit", "v96_773_crossfit")),
    ("psg", OOF / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet", ("psg_tvt",)),
    ("track8", OOF / "track8_sp45_foldclean_fivefold_v2" / "track8_oof.parquet", ("track8", "common_only")),
    ("legacy", OOF / "frontier_6p39_existing_models_oof_v0" / "frontier_6p39_core_oof.parquet",
     ("sp45", "strong_bias", "strong", "cp12", "fleongg")),
    ("neighbor", OOF / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet",
     ("fle_neighbor", "fle_parent")),
    ("stride", OOF / "stride_independent_engine_v155" / "stride_v155_predictions.parquet", ("prediction",)),
)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _load_aligned(path: Path, columns: tuple[str, ...], canonical: pd.DataFrame) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    schema = set(pq.ParquetFile(path).schema.names)
    available = [column for column in columns if column in schema]
    if not available or not {"well", "row_idx"}.issubset(schema):
        return {}
    local = pd.read_parquet(path, columns=["well", "row_idx", *available])
    local["well"] = local["well"].astype(str)
    local["row_idx"] = local["row_idx"].astype(np.int64)
    local = local.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if len(local) != len(canonical):
        return {}
    if not np.array_equal(local["well"].to_numpy(), canonical["well"].to_numpy()):
        return {}
    if not np.array_equal(local["row_idx"].to_numpy(), canonical["row_idx"].to_numpy()):
        return {}
    output = {}
    for column in available:
        values = local[column].to_numpy(np.float32)
        if np.isfinite(values).all():
            output[column] = values
    return output


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, names = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error * error, minlength=len(names))
    well_rmse = np.sqrt(sse / counts)
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    for index, name in enumerate(DISTANCE_NAMES):
        mask = (distance >= DISTANCE_EDGES[index]) & (distance < DISTANCE_EDGES[index + 1])
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-20 else np.nan


def _distance_detrend(error: np.ndarray, distance: np.ndarray, bins: int = 64) -> np.ndarray:
    """Subtract a piecewise distance-error regression curve."""
    finite_distance = distance[np.isfinite(distance)]
    edges = np.unique(np.quantile(finite_distance, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return error - np.mean(error)
    labels = np.clip(np.searchsorted(edges, distance, side="right") - 1, 0, len(edges) - 2)
    sums = np.bincount(labels, weights=error, minlength=len(edges) - 1)
    counts = np.bincount(labels, minlength=len(edges) - 1)
    means = sums / np.maximum(counts, 1)
    return error - means[labels]


def _pair_report(
    name: str,
    base_error: np.ndarray,
    candidate_error: np.ndarray,
    distance: np.ndarray,
) -> dict:
    m1 = float(np.mean(base_error * base_error))
    m2 = float(np.mean(candidate_error * candidate_error))
    cross = float(np.mean(base_error * candidate_error))
    denominator = m1 + m2 - 2.0 * cross
    alpha_unclipped = (m1 - cross) / denominator if denominator > 1e-20 else 0.0
    alpha = float(np.clip(alpha_unclipped, 0.0, 1.0))
    blend_error = (1.0 - alpha) * base_error + alpha * candidate_error
    formula_mse = (
        (m1 * m2 - cross * cross) / denominator if denominator > 1e-20 else m1
    )
    base_detrended = _distance_detrend(base_error, distance)
    candidate_detrended = _distance_detrend(candidate_error, distance)
    report = {
        "endpoint": name,
        "base_rmse": float(np.sqrt(m1)),
        "standalone_rmse": float(np.sqrt(m2)),
        "raw_pearson": _corr(base_error, candidate_error),
        "uncentred_error_cosine": _cosine(base_error, candidate_error),
        "distance_detrended_pearson": _corr(base_detrended, candidate_detrended),
        "analytic_alpha_unclipped": float(alpha_unclipped),
        "analytic_alpha_clipped": alpha,
        "exact_clipped_blend_rmse": float(np.sqrt(np.mean(blend_error * blend_error))),
        "exact_clipped_gain": float(np.sqrt(m1) - np.sqrt(np.mean(blend_error * blend_error))),
        "formula_unclipped_rmse": float(np.sqrt(max(formula_mse, 0.0))),
    }
    for index, band in enumerate(DISTANCE_NAMES):
        mask = (distance >= DISTANCE_EDGES[index]) & (distance < DISTANCE_EDGES[index + 1])
        report[f"{band}_raw_pearson"] = _corr(base_error[mask], candidate_error[mask])
        report[f"{band}_detrended_pearson"] = _corr(base_detrended[mask], candidate_detrended[mask])
    return report


def _project_l1_ball(values: np.ndarray, radius: float = 1.0) -> np.ndarray:
    values = np.maximum(np.asarray(values, np.float64), 0.0)
    if values.sum() <= radius:
        return values
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(values) + 1)
    valid = ordered - (cumulative - radius) / indices > 0.0
    rho = int(np.flatnonzero(valid)[-1])
    threshold = (cumulative[rho] - radius) / (rho + 1)
    return np.maximum(values - threshold, 0.0)


def _fit_residual_simplex(gram: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Minimize ||base + D*a - truth|| with a>=0 and sum(a)<=1."""
    if len(linear) == 0:
        return np.empty(0, np.float64)
    eigen_max = float(np.max(np.linalg.eigvalsh(gram))) if len(linear) else 0.0
    step = 1.0 / max(2.0 * eigen_max, 1e-12)
    weights = np.zeros(len(linear), np.float64)
    for _ in range(2000):
        updated = _project_l1_ball(weights - step * 2.0 * (gram @ weights - linear))
        if np.max(np.abs(updated - weights)) < 1e-10:
            weights = updated
            break
        weights = updated
    return weights


base = pd.read_parquet(BASE_PATH)
required = {"well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN}
missing = sorted(required.difference(base.columns))
if missing:
    raise RuntimeError(f"V168 missing columns: {missing}")
base["well"] = base["well"].astype(str)
base["row_idx"] = base["row_idx"].astype(np.int64)
base = base.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Invalid canonical coverage: {len(base)}/{base['well'].nunique()}")
base_values = base[BASE_COLUMN].to_numpy(np.float32)
base_sha = hashlib.sha256(base_values.tobytes()).hexdigest()
if base_sha != EXPECTED_BASE_SHA:
    raise RuntimeError(f"V168 prediction SHA mismatch: {base_sha}")

candidate_arrays: dict[str, np.ndarray] = {BASE_NAME: base_values}
source_report = []
for alias, path, columns in SOURCES:
    loaded = _load_aligned(path, columns, base)
    source_report.append({
        "alias": alias,
        "path": str(path),
        "status": "loaded" if loaded else ("missing" if not path.is_file() else "incompatible"),
        "columns": list(loaded),
    })
    for column, values in loaded.items():
        name = f"{alias}/v155_full" if alias == "stride" and column == "prediction" else f"{alias}/{column}"
        candidate_arrays[name] = values

# Truth-independent exact duplicate removal.
unique_arrays: dict[str, np.ndarray] = {}
seen_hashes: set[str] = set()
for name, values in candidate_arrays.items():
    digest = hashlib.sha256(np.asarray(values, np.float32).tobytes()).hexdigest()
    if digest not in seen_hashes:
        unique_arrays[name] = np.asarray(values, np.float32)
        seen_hashes.add(digest)
candidate_arrays = unique_arrays
if BASE_NAME not in candidate_arrays or len(candidate_arrays) < 5:
    raise RuntimeError(f"Only {len(candidate_arrays)} unique aligned endpoints")

names = list(candidate_arrays)
matrix = np.stack([candidate_arrays[name] for name in names]).astype(np.float32)
base_index = names.index(BASE_NAME)
if base_index != 0:
    raise RuntimeError("V168 must remain the first endpoint")
truth = base["tvt_true"].to_numpy(np.float64)
distance = base["md_since"].to_numpy(np.float64)
fold_values = base["fold"].to_numpy(np.int8)
base_error = matrix[0].astype(np.float64) - truth

standalone = pd.DataFrame([
    _metrics(base, matrix[index], name) for index, name in enumerate(names)
]).sort_values("rmse", ignore_index=True)
pairwise = pd.DataFrame([
    _pair_report(name, base_error, matrix[index].astype(np.float64) - truth, distance)
    for index, name in enumerate(names) if index != base_index
]).sort_values("exact_clipped_gain", ascending=False, ignore_index=True)

# Fold sufficient statistics for residual-simplex fitting.
deltas = matrix[1:].astype(np.float64) - matrix[0:1].astype(np.float64)
endpoint_count = deltas.shape[0]
fold_stats: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
for fold in range(1, 6):
    positions = np.flatnonzero(fold_values == fold)
    local_deltas = deltas[:, positions]
    local_error = base_error[positions]
    fold_stats[fold] = (
        local_deltas @ local_deltas.T,
        -(local_deltas @ local_error),
        len(positions),
    )

total_gram = sum(item[0] for item in fold_stats.values())
total_linear = sum(item[1] for item in fold_stats.values())
total_rows = sum(item[2] for item in fold_stats.values())
nested_simplex = matrix[0].astype(np.float64).copy()
nested_band = matrix[0].astype(np.float64).copy()
nested_robust = matrix[0].astype(np.float64).copy()
weight_reports = []

for held_fold in range(1, 6):
    held = fold_values == held_fold
    train = ~held
    train_rows = total_rows - fold_stats[held_fold][2]
    gram = (total_gram - fold_stats[held_fold][0]) / train_rows
    linear = (total_linear - fold_stats[held_fold][1]) / train_rows
    residual_weights = _fit_residual_simplex(gram, linear)
    full_weights = np.concatenate(([1.0 - residual_weights.sum()], residual_weights))
    nested_simplex[held] = full_weights @ matrix[:, held].astype(np.float64)

    for band_index, band_name in enumerate(DISTANCE_NAMES):
        train_band = train & (distance >= DISTANCE_EDGES[band_index]) & (distance < DISTANCE_EDGES[band_index + 1])
        held_band = held & (distance >= DISTANCE_EDGES[band_index]) & (distance < DISTANCE_EDGES[band_index + 1])
        if not held_band.any():
            continue
        errors = matrix[:, train_band].astype(np.float64) - truth[train_band][None, :]
        mse = np.mean(errors * errors, axis=1)
        base_mse = mse[base_index]
        eligible = mse <= 2.0 * base_mse
        reliability = np.where(eligible, 1.0 / np.maximum(mse, 1e-9), 0.0)
        reliability /= reliability.sum()
        local_matrix = matrix[:, held_band].astype(np.float64)
        centre = reliability @ local_matrix
        nested_band[held_band] = centre

        absolute = np.abs(local_matrix - centre[None, :])
        scale = reliability @ absolute
        scale = np.maximum(scale, 0.25)
        huber = np.minimum(1.0, (2.0 * scale[None, :]) / np.maximum(absolute, 1e-9))
        local_weights = reliability[:, None] * huber
        local_weights /= np.maximum(local_weights.sum(axis=0, keepdims=True), 1e-12)
        nested_robust[held_band] = np.sum(local_weights * local_matrix, axis=0)

        weight_reports.append({
            "held_fold": held_fold,
            "distance_band": band_name,
            "consumer": "inverse_mse_posterior_mean",
            "weights": {names[index]: float(reliability[index]) for index in range(len(names)) if reliability[index] > 1e-8},
        })
    weight_reports.append({
        "held_fold": held_fold,
        "distance_band": "all",
        "consumer": "residual_simplex",
        "weights": {names[index]: float(full_weights[index]) for index in range(len(names)) if full_weights[index] > 1e-8},
    })

summary = pd.DataFrame([
    _metrics(base, matrix[0], "v168"),
    _metrics(base, nested_simplex, "nested_residual_simplex"),
    _metrics(base, nested_band, "nested_band_posterior_mean"),
    _metrics(base, nested_robust, "nested_band_robust_mean"),
])
base_rmse = float(summary.loc[summary.endpoint.eq("v168"), "rmse"].iloc[0])
for endpoint in summary.endpoint:
    endpoint_rmse = float(summary.loc[summary.endpoint.eq(endpoint), "rmse"].iloc[0])
    summary.loc[summary.endpoint.eq(endpoint), "gain"] = base_rmse - endpoint_rmse

fold_rows = []
predictions = {
    "v168": matrix[0],
    "nested_residual_simplex": nested_simplex,
    "nested_band_posterior_mean": nested_band,
    "nested_band_robust_mean": nested_robust,
}
for fold in range(1, 6):
    mask = fold_values == fold
    local = base.loc[mask].reset_index(drop=True)
    base_fold_rmse = _metrics(local, matrix[0, mask], "v168")["rmse"]
    for endpoint, prediction in predictions.items():
        metric = _metrics(local, np.asarray(prediction)[mask], endpoint)
        fold_rows.append({
            "fold": fold,
            "endpoint": endpoint,
            "rmse": metric["rmse"],
            "gain": base_fold_rmse - metric["rmse"],
            "far_gt2000_rmse": metric["far_gt2000_rmse"],
        })
fold_summary = pd.DataFrame(fold_rows)

candidate_endpoints = [name for name in predictions if name != "v168"]
consumer_checks = {}
for endpoint in candidate_endpoints:
    local = fold_summary[fold_summary.endpoint.eq(endpoint)]
    gain = float(summary.loc[summary.endpoint.eq(endpoint), "gain"].iloc[0])
    consumer_checks[endpoint] = {
        "gain": gain,
        "positive_folds": int(np.sum(local["gain"] > 0.0)),
        "minimum_fold_gain": float(local["gain"].min()),
        "pass": bool(gain >= 0.10 and np.sum(local["gain"] > 0.0) >= 4 and local["gain"].min() >= -0.03),
    }

stride_loaded = "stride/v155_full" in names
stride_pair = pairwise[pairwise.endpoint.eq("stride/v155_full")]
best_pair = pairwise.iloc[0].to_dict()
strict_pass = bool(any(report["pass"] for report in consumer_checks.values()))
verdict = {
    "experiment": "canonical_multiengine_soft_fusion_audit_v198",
    "stage": "corrected_lineage_dependence_and_fivefold_soft_fusion_audit",
    "base": {"path": str(BASE_PATH), "column": BASE_COLUMN, "rmse": base_rmse},
    "candidate_count": len(names),
    "candidate_endpoints": names,
    "source_report": source_report,
    "lineage_correction": {
        "v187_stride_locator": str(OOF / "stride_independent_engine_v155" / "stride_oof.parquet"),
        "actual_v155_locator": str(OOF / "stride_independent_engine_v155" / "stride_v155_predictions.parquet"),
        "actual_v155_column": "prediction",
        "stride_loaded": stride_loaded,
    },
    "correlation_note": (
        "The closed-form MSE formula uses the uncentred error cross moment. "
        "Distance-detrended Pearson is an independence diagnostic and cannot "
        "replace that cross moment in the raw pooled-MSE formula."
    ),
    "best_pairwise_global_blend": best_pair,
    "v168_v155": stride_pair.iloc[0].to_dict() if len(stride_pair) else None,
    "soft_consumers": consumer_checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the passing soft consumer and audit deployable endpoint lineage"
        if strict_pass else
        "FAIL -> corrected endpoint diversity still does not yield a stable 0.10-ft soft-fusion gain"
    ),
}

standalone.to_csv(OUT / "standalone_endpoints.csv", index=False)
pairwise.to_csv(OUT / "v168_pairwise_dependence.csv", index=False)
summary.to_csv(OUT / "soft_fusion_summary.csv", index=False)
fold_summary.to_csv(OUT / "soft_fusion_by_fold.csv", index=False)
_write_json(OUT / "weights.json", {"reports": weight_reports})
_write_json(OUT / "verdict.json", verdict)
print("\nV198 STANDALONE")
print(standalone.to_string(index=False), flush=True)
print("\nV198 V168 PAIRWISE")
print(pairwise.to_string(index=False), flush=True)
print("\nV198 SOFT FUSION")
print(summary.to_string(index=False), flush=True)
print("\nV198 FOLDS")
print(fold_summary.to_string(index=False), flush=True)
print("\nV198 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT, flush=True)
