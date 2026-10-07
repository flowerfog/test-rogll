"""Route 4 Phase 0: multi-well regional geological-surface oracle.

This is deliberately not another single-neighbour curve transfer.  Every
query location is estimated jointly from one nearby surface observation per
reference well, where surface = TVT + Z.  The target well is never available
as a reference.  Two reference protocols are evaluated:

* loo_all_other: deployment-equivalent leave-one-well-out with 772 references;
* outer_fold_excluded: excludes the target's complete canonical fold.

No selector is trained.  The experiment measures standalone accuracy,
candidate-bank oracle, fixed blends with the complete V66 incumbent, error
correlation, and uncertainty/error association.
Run this file as one AutoDL notebook cell or as a script.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "regional_surface_multiwell_oracle_v75"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V66_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
DATA_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    ROOT / "replay_data",
    ROOT / "data",
)
DATA_ROOT = next((path for path in DATA_CANDIDATES if (path / "train").is_dir()), None)

EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
REFERENCE_STRIDE = 12
QUERY_ROWS_PER_WELL = 192
COARSE_POINTS_PER_WELL = 16
COARSE_QUERY_K = 256
COARSE_QUERY_K_MAX = 2048
SHORTLIST_WELLS = 64
MIN_DISTINCT_WELLS = 24
DISTANCE_FLOOR_FT = 40.0
RIDGE_LINEAR = 0.10
RIDGE_QUADRATIC = 1.0
ROBUST_ITERATIONS = 3
RANDOM_SEED = 20260729

PROTOCOLS = ("loo_all_other", "outer_fold_excluded")
CANDIDATES = (
    "idw_k4",
    "idw_k8",
    "idw_k16",
    "plane_k8",
    "plane_k16",
    "plane_k24",
    "robust_plane_k16",
    "robust_plane_k24",
    "quadratic_k24",
)
BLEND_WEIGHTS = np.round(np.linspace(0.0, 0.60, 13), 2)

if DATA_ROOT is None:
    raise FileNotFoundError(f"No train directory under {[str(p) for p in DATA_CANDIDATES]}")
for required in (V66_PATH, FOLD_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)


def _evenly_spaced(index: np.ndarray, maximum: int) -> np.ndarray:
    index = np.asarray(index, np.int32)
    if len(index) <= maximum:
        return index
    take = np.unique(np.round(np.linspace(0, len(index) - 1, maximum)).astype(int))
    return index[take]


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not valid.all():
        raise RuntimeError(f"{endpoint}: {int((~valid).sum())} non-finite rows")
    error = prediction - truth
    well_mse = (
        pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
        .groupby("well", sort=False)["e2"]
        .mean()
        .to_numpy(np.float64)
    )
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
        "far_gt2000_rmse": (
            float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan
        ),
    }


def _weighted_average(values: np.ndarray, distance: np.ndarray) -> float:
    weight = 1.0 / np.maximum(distance + DISTANCE_FLOOR_FT, DISTANCE_FLOOR_FT) ** 2
    return float(np.dot(weight, values) / np.sum(weight))


def _design(offset: np.ndarray, quadratic: bool) -> np.ndarray:
    x = offset[:, 0] / 1000.0
    y = offset[:, 1] / 1000.0
    if quadratic:
        return np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y))
    return np.column_stack((np.ones(len(x)), x, y))


def _weighted_regression(
    xy: np.ndarray,
    values: np.ndarray,
    distance: np.ndarray,
    query_xy: np.ndarray,
    *,
    quadratic: bool = False,
    robust: bool = False,
) -> tuple[float, float]:
    offset = xy - query_xy[None, :]
    matrix = _design(offset, quadratic)
    base_weight = 1.0 / np.maximum(
        distance + DISTANCE_FLOOR_FT, DISTANCE_FLOOR_FT
    ) ** 2
    weight = base_weight.copy()
    ridge = RIDGE_QUADRATIC if quadratic else RIDGE_LINEAR
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 1e-8
    coefficient = None
    for iteration in range(ROBUST_ITERATIONS if robust else 1):
        lhs = matrix.T @ (weight[:, None] * matrix) + penalty
        rhs = matrix.T @ (weight * values)
        try:
            coefficient = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if robust and iteration + 1 < ROBUST_ITERATIONS:
            residual = values - matrix @ coefficient
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 0.25
            huber = np.minimum(1.0, (1.5 * scale) / np.maximum(np.abs(residual), 1e-9))
            weight = base_weight * huber
    fitted = matrix @ coefficient
    residual_scale = float(np.sqrt(np.average((values - fitted) ** 2, weights=base_weight)))
    return float(coefficient[0]), residual_scale


def _shortlist_reference_codes(
    query_xy: np.ndarray,
    target_code: int,
    target_fold: int,
    protocol: str,
    coarse_tree: cKDTree,
    coarse_well_code: np.ndarray,
    code_fold: np.ndarray,
) -> np.ndarray:
    """Select wells with equal per-well representation before exact lookup."""
    query_k = COARSE_QUERY_K
    score: dict[int, float] = {}
    while True:
        distances, indices = coarse_tree.query(
            query_xy, k=min(query_k, len(coarse_well_code)), workers=-1
        )
        distances = np.asarray(distances, np.float64).reshape(-1)
        indices = np.asarray(indices, np.int64).reshape(-1)
        codes = coarse_well_code[indices]
        for code, distance in zip(codes, distances):
            code = int(code)
            if code == target_code:
                continue
            if protocol == "outer_fold_excluded" and int(code_fold[code]) == target_fold:
                continue
            previous = score.get(code)
            if previous is None or distance < previous:
                score[code] = float(distance)
        if (
            len(score) >= SHORTLIST_WELLS
            or query_k >= COARSE_QUERY_K_MAX
            or query_k >= len(coarse_well_code)
        ):
            break
        query_k = min(query_k * 2, COARSE_QUERY_K_MAX, len(coarse_well_code))
    if len(score) < MIN_DISTINCT_WELLS:
        raise RuntimeError(
            f"Only {len(score)} coarse reference wells for protocol={protocol}"
        )
    ordered = sorted(score, key=lambda code: (score[code], code))[:SHORTLIST_WELLS]
    return np.asarray(ordered, np.int16)


def _fit_query_candidates(
    query_xy: np.ndarray,
    xy: np.ndarray,
    values: np.ndarray,
    chosen_distance: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], int, float]:
    if len(values) < MIN_DISTINCT_WELLS:
        raise RuntimeError(f"Only {len(values)} exact reference wells")
    output: dict[str, float] = {}
    residual: dict[str, float] = {}
    for k in (4, 8, 16):
        output[f"idw_k{k}"] = _weighted_average(values[:k], chosen_distance[:k])
        residual[f"idw_k{k}"] = float(
            np.sqrt(np.average(
                (values[:k] - output[f"idw_k{k}"]) ** 2,
                weights=1.0 / np.maximum(chosen_distance[:k] + DISTANCE_FLOOR_FT, DISTANCE_FLOOR_FT) ** 2,
            ))
        )
    for k in (8, 16, 24):
        value, scale = _weighted_regression(
            xy[:k], values[:k], chosen_distance[:k], query_xy
        )
        output[f"plane_k{k}"] = value
        residual[f"plane_k{k}"] = scale
    for k in (16, 24):
        value, scale = _weighted_regression(
            xy[:k], values[:k], chosen_distance[:k], query_xy, robust=True
        )
        output[f"robust_plane_k{k}"] = value
        residual[f"robust_plane_k{k}"] = scale
    value, scale = _weighted_regression(
        xy[:24], values[:24], chosen_distance[:24], query_xy, quadratic=True, robust=True
    )
    output["quadratic_k24"] = value
    residual["quadratic_k24"] = scale
    return output, residual, len(values), float(chosen_distance[0])


def _detrended_correlation(error_a: np.ndarray, error_b: np.ndarray, distance: np.ndarray) -> float:
    bins = np.clip((distance // 250.0).astype(int), 0, 20)
    a = error_a.copy()
    b = error_b.copy()
    for value in np.unique(bins):
        mask = bins == value
        a[mask] -= np.mean(a[mask])
        b[mask] -= np.mean(b[mask])
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Canonical V66 sampled target rows.
# ---------------------------------------------------------------------------
v66_header = pq.read_schema(V66_PATH).names
prediction_column = next(
    (
        column
        for column in (
            "incumbent_6p377_replay",
            "prediction",
            "current_dose_0p20",
            "tvt_pred",
        )
        if column in v66_header
    ),
    None,
)
if prediction_column is None:
    raise RuntimeError(f"Cannot identify V66 prediction: {v66_header}")

columns = [
    column
    for column in ("id", "well", "row_idx", "fold", "tvt_true", "md_since", prediction_column)
    if column in v66_header
]
v66 = pd.read_parquet(V66_PATH, columns=columns)
v66["well"] = v66["well"].astype(str)
v66 = v66.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(v66) != EXPECTED_ROWS or v66["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"V66 coverage mismatch: rows={len(v66)} wells={v66['well'].nunique()}")

fold_frame = pd.read_csv(FOLD_PATH)
fold_frame["well"] = fold_frame["well"].astype(str)
fold_by_well = fold_frame.groupby("well")["fold"].first().astype(int).to_dict()
if len(fold_by_well) != EXPECTED_WELLS:
    raise RuntimeError(f"Fold metadata has {len(fold_by_well)} wells")

sample_parts = []
for well, group in v66.groupby("well", sort=True):
    positions = _evenly_spaced(np.arange(len(group), dtype=np.int32), QUERY_ROWS_PER_WELL)
    sample_parts.append(group.iloc[positions])
sample = pd.concat(sample_parts, ignore_index=True)
sample = sample.rename(columns={prediction_column: "base"})


# ---------------------------------------------------------------------------
# Load complete trajectories and form a sparse regional surface observation set.
# ---------------------------------------------------------------------------
wells = sorted(sample["well"].unique())
well_to_code = {well: code for code, well in enumerate(wells)}
code_fold = np.asarray([fold_by_well[well] for well in wells], np.int8)
records: dict[str, dict] = {}
point_xy_parts = []
point_surface_parts = []
point_code_parts = []
coarse_xy_parts = []
coarse_code_parts = []
failures = []

for number, well in enumerate(wells, 1):
    try:
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        frame = frame.apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(frame[["X", "Y", "Z", "TVT"]].to_numpy()).all(axis=1)
        known = np.flatnonzero(np.isfinite(frame["TVT_input"].to_numpy(np.float64)))
        if len(known) == 0:
            raise RuntimeError("no visible prefix")
        anchor = int(known[-1])
        xy = frame[["X", "Y"]].to_numpy(np.float64)
        z = frame["Z"].to_numpy(np.float64)
        surface = frame["TVT"].to_numpy(np.float64) + z
        reference_rows = np.flatnonzero(finite)[::REFERENCE_STRIDE]
        if len(reference_rows) < 40:
            raise RuntimeError(f"only {len(reference_rows)} reference rows")
        reference_xy = xy[reference_rows]
        reference_surface = surface[reference_rows]
        records[well] = {
            "xy": xy,
            "z": z,
            "surface": surface,
            "anchor": anchor,
            "anchor_tvt_input": float(frame["TVT_input"].iloc[anchor]),
            "reference_xy": reference_xy,
            "reference_surface": reference_surface,
            "reference_tree": cKDTree(reference_xy, leafsize=32),
        }
        code = well_to_code[well]
        point_xy_parts.append(reference_xy)
        point_surface_parts.append(reference_surface)
        point_code_parts.append(np.full(len(reference_rows), code, np.int16))
        coarse_positions = _evenly_spaced(
            np.arange(len(reference_xy), dtype=np.int32), COARSE_POINTS_PER_WELL
        )
        coarse_xy_parts.append(reference_xy[coarse_positions])
        coarse_code_parts.append(np.full(len(coarse_positions), code, np.int16))
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == len(wells):
        print(f"V75 load {number}/{len(wells)} failures={len(failures)}", flush=True)

pd.DataFrame(failures).to_csv(OUT_DIR / "load_failures.csv", index=False)
if failures:
    raise RuntimeError(f"V75 requires exact 773-well coverage: {failures[:5]}")

point_xy = np.concatenate(point_xy_parts).astype(np.float64)
point_surface = np.concatenate(point_surface_parts).astype(np.float64)
point_well_code = np.concatenate(point_code_parts).astype(np.int16)
coarse_xy = np.concatenate(coarse_xy_parts).astype(np.float64)
coarse_well_code = np.concatenate(coarse_code_parts).astype(np.int16)
coarse_tree = cKDTree(coarse_xy, leafsize=64)
print(
    "V75 CONFIG",
    {
        "data": str(DATA_ROOT),
        "v66": str(V66_PATH),
        "sample_rows": len(sample),
        "sample_wells": sample["well"].nunique(),
        "reference_points": len(point_xy),
        "coarse_points": len(coarse_xy),
        "protocols": PROTOCOLS,
        "candidates": CANDIDATES,
    },
    flush=True,
)


# ---------------------------------------------------------------------------
# Evaluate every surface family on identical target rows.
# ---------------------------------------------------------------------------
protocol_frames = []
diagnostic_rows = []
for protocol in PROTOCOLS:
    parts = []
    for number, (well, group) in enumerate(sample.groupby("well", sort=True), 1):
        record = records[well]
        target_code = well_to_code[well]
        target_fold = int(fold_by_well[well])
        row_idx = group["row_idx"].to_numpy(np.int32)
        anchor = int(record["anchor"])
        if np.any(row_idx < 0) or np.any(row_idx >= len(record["xy"])):
            raise RuntimeError(f"{well}: V66 row indices outside source trajectory")
        query_rows = np.r_[anchor, row_idx]
        query_xy = record["xy"][query_rows]
        shortlist_codes = _shortlist_reference_codes(
            query_xy,
            target_code,
            target_fold,
            protocol,
            coarse_tree,
            coarse_well_code,
            code_fold,
        )
        exact_distance = np.empty((len(shortlist_codes), len(query_rows)), np.float64)
        exact_xy = np.empty((len(shortlist_codes), len(query_rows), 2), np.float64)
        exact_surface = np.empty((len(shortlist_codes), len(query_rows)), np.float64)
        for reference_position, reference_code in enumerate(shortlist_codes):
            reference = records[wells[int(reference_code)]]
            distance, local_index = reference["reference_tree"].query(
                query_xy, k=1, workers=-1
            )
            exact_distance[reference_position] = distance
            exact_xy[reference_position] = reference["reference_xy"][local_index]
            exact_surface[reference_position] = reference["reference_surface"][local_index]
        surface_predictions = {candidate: np.empty(len(query_rows)) for candidate in CANDIDATES}
        residual_predictions = {candidate: np.empty(len(query_rows)) for candidate in CANDIDATES}
        distinct_counts = np.empty(len(query_rows), np.int16)
        nearest_distance = np.empty(len(query_rows), np.float64)
        for position, source_row in enumerate(query_rows):
            order = np.argsort(exact_distance[:, position])[:MIN_DISTINCT_WELLS]
            output, residual, count, nearest = _fit_query_candidates(
                record["xy"][source_row],
                exact_xy[order, position],
                exact_surface[order, position],
                exact_distance[order, position],
            )
            for candidate in CANDIDATES:
                surface_predictions[candidate][position] = output[candidate]
                residual_predictions[candidate][position] = residual[candidate]
            distinct_counts[position] = count
            nearest_distance[position] = nearest

        output_frame = group.copy()
        target_z = record["z"][row_idx]
        anchor_z = float(record["z"][anchor])
        anchor_tvt = float(record["anchor_tvt_input"])
        candidate_matrix = []
        residual_matrix = []
        for candidate in CANDIDATES:
            surface_delta = surface_predictions[candidate][1:] - surface_predictions[candidate][0]
            prediction = anchor_tvt + surface_delta - (target_z - anchor_z)
            output_frame[candidate] = prediction.astype(np.float32)
            output_frame[f"residual_{candidate}"] = residual_predictions[candidate][1:].astype(np.float32)
            candidate_matrix.append(prediction)
            residual_matrix.append(residual_predictions[candidate][1:])
        output_frame["candidate_spread"] = np.std(np.asarray(candidate_matrix), axis=0).astype(np.float32)
        output_frame["mean_plane_residual"] = np.mean(np.asarray(residual_matrix), axis=0).astype(np.float32)
        output_frame["nearest_reference_distance"] = nearest_distance[1:].astype(np.float32)
        output_frame["distinct_references"] = distinct_counts[1:]
        output_frame["protocol"] = protocol
        parts.append(output_frame)
        if number % 50 == 0 or number == len(wells):
            print(f"V75 {protocol} {number}/{len(wells)}", flush=True)
    protocol_frame = pd.concat(parts, ignore_index=True)
    protocol_frames.append(protocol_frame)

    truth = protocol_frame["tvt_true"].to_numpy(np.float64)
    base = protocol_frame["base"].to_numpy(np.float64)
    base_error = base - truth
    for candidate in CANDIDATES:
        prediction = protocol_frame[candidate].to_numpy(np.float64)
        error = prediction - truth
        distance = protocol_frame["md_since"].to_numpy(np.float64)
        spread = protocol_frame["candidate_spread"].to_numpy(np.float64)
        residual = protocol_frame[f"residual_{candidate}"].to_numpy(np.float64)
        diagnostic_rows.append(
            {
                "protocol": protocol,
                "candidate": candidate,
                "candidate_rmse": float(np.sqrt(np.mean(error**2))),
                "raw_error_rho": float(np.corrcoef(base_error, error)[0, 1]),
                "distance_detrended_rho": _detrended_correlation(base_error, error, distance),
                "spread_abs_error_spearman": float(spearmanr(spread, np.abs(error)).statistic),
                "residual_abs_error_spearman": float(spearmanr(residual, np.abs(error)).statistic),
            }
        )

all_predictions = pd.concat(protocol_frames, ignore_index=True)
diagnostics = pd.DataFrame(diagnostic_rows)
all_predictions.to_parquet(
    OUT_DIR / "sampled_regional_surface_predictions.parquet",
    index=False,
    compression="zstd",
)


# ---------------------------------------------------------------------------
# Candidate and blend ceilings. Oracle selection is per complete well.
# ---------------------------------------------------------------------------
summary_rows = []
blend_rows = []
oracle_payload = {}
for protocol, frame in all_predictions.groupby("protocol", sort=False):
    frame = frame.reset_index(drop=True)
    truth = frame["tvt_true"].to_numpy(np.float64)
    base = frame["base"].to_numpy(np.float64)
    summary_rows.append({"protocol": protocol, **_metric(frame, base, "base_v66")})
    for candidate in CANDIDATES:
        prediction = frame[candidate].to_numpy(np.float64)
        summary_rows.append({"protocol": protocol, **_metric(frame, prediction, candidate)})
        for weight in BLEND_WEIGHTS:
            blend = base + float(weight) * (prediction - base)
            metric = _metric(frame, blend, f"blend_{candidate}_w{weight:g}")
            blend_rows.append({"protocol": protocol, "candidate": candidate, "weight": float(weight), **metric})

    # Per-well candidate-bank oracle and candidate+blend oracle.
    candidate_arrays = {
        candidate: frame[candidate].to_numpy(np.float64) for candidate in CANDIDATES
    }
    bank_prediction = np.empty(len(frame), np.float64)
    fusion_prediction = np.empty(len(frame), np.float64)
    bank_choices = []
    for well, positions in frame.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        y = truth[positions]
        best_bank = None
        best_bank_sse = np.inf
        best_fusion = None
        best_fusion_sse = np.inf
        for candidate in CANDIDATES:
            value = candidate_arrays[candidate][positions]
            sse = float(np.sum((value - y) ** 2))
            if sse < best_bank_sse:
                best_bank_sse = sse
                best_bank = (candidate, value)
            for weight in BLEND_WEIGHTS:
                blend = base[positions] + float(weight) * (value - base[positions])
                blend_sse = float(np.sum((blend - y) ** 2))
                if blend_sse < best_fusion_sse:
                    best_fusion_sse = blend_sse
                    best_fusion = (candidate, float(weight), blend)
        bank_prediction[positions] = best_bank[1]
        fusion_prediction[positions] = best_fusion[2]
        bank_choices.append(
            {
                "protocol": protocol,
                "well": str(well),
                "bank_candidate": best_bank[0],
                "bank_sse": best_bank_sse,
                "fusion_candidate": best_fusion[0],
                "fusion_weight": best_fusion[1],
                "fusion_sse": best_fusion_sse,
            }
        )
    bank_metric = _metric(frame, bank_prediction, "per_well_bank_oracle")
    fusion_metric = _metric(frame, fusion_prediction, "per_well_fusion_oracle")
    summary_rows.extend(
        [
            {"protocol": protocol, **bank_metric},
            {"protocol": protocol, **fusion_metric},
        ]
    )
    oracle_payload[protocol] = {
        "bank": bank_metric,
        "fusion": fusion_metric,
        "choices": bank_choices,
    }

summary = pd.DataFrame(summary_rows).sort_values(["protocol", "rmse"])
blend_scan = pd.DataFrame(blend_rows).sort_values(["protocol", "rmse"])
best_fixed = blend_scan.groupby("protocol", sort=False).head(1).reset_index(drop=True)

print("\nV75 REGIONAL SURFACE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV75 BEST FIXED BLENDS")
print(best_fixed.to_string(index=False), flush=True)
print("\nV75 ERROR / UNCERTAINTY DIAGNOSTICS")
print(diagnostics.sort_values(["protocol", "candidate_rmse"]).to_string(index=False), flush=True)

summary.to_csv(OUT_DIR / "summary.csv", index=False)
blend_scan.to_csv(OUT_DIR / "fixed_blend_scan.csv", index=False)
diagnostics.to_csv(OUT_DIR / "diagnostics.csv", index=False)
pd.DataFrame(
    [row for value in oracle_payload.values() for row in value["choices"]]
).to_csv(OUT_DIR / "oracle_choices.csv", index=False)

loo_summary = summary[summary["protocol"] == "loo_all_other"].set_index("endpoint")
outer_summary = summary[summary["protocol"] == "outer_fold_excluded"].set_index("endpoint")
loo_best_candidate = diagnostics[diagnostics["protocol"] == "loo_all_other"].sort_values("candidate_rmse").iloc[0]
outer_best_candidate = diagnostics[diagnostics["protocol"] == "outer_fold_excluded"].sort_values("candidate_rmse").iloc[0]
loo_blend = best_fixed[best_fixed["protocol"] == "loo_all_other"].iloc[0]
outer_blend = best_fixed[best_fixed["protocol"] == "outer_fold_excluded"].iloc[0]

checks = {
    "loo_bank_oracle_rmse_le_7": bool(loo_summary.loc["per_well_bank_oracle", "rmse"] <= 7.0),
    "outer_bank_oracle_rmse_le_7p5": bool(outer_summary.loc["per_well_bank_oracle", "rmse"] <= 7.5),
    "loo_fixed_blend_gain_ge_0p10": bool(
        loo_summary.loc["base_v66", "rmse"] - float(loo_blend["rmse"]) >= 0.10
    ),
    "outer_fixed_blend_gain_ge_0p05": bool(
        outer_summary.loc["base_v66", "rmse"] - float(outer_blend["rmse"]) >= 0.05
    ),
    "loo_best_error_rho_le_0p75": bool(float(loo_best_candidate["distance_detrended_rho"]) <= 0.75),
    "outer_best_error_rho_le_0p80": bool(float(outer_best_candidate["distance_detrended_rho"]) <= 0.80),
}
strict_pass = bool(all(checks.values()))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


verdict = {
    "experiment": EXPERIMENT,
    "stage": "route4_phase0_multiwell_regional_surface_oracle",
    "sample_unit": "192 fixed hidden-suffix rows per complete physical well",
    "truth_usage": "other-well regional surface observations; target truth only for audit/oracle",
    "protocols": {
        "loo_all_other": "target well excluded; deployment-equivalent 772-well library",
        "outer_fold_excluded": "target canonical fold entirely excluded",
    },
    "sources": {
        "v66": str(V66_PATH),
        "folds": str(FOLD_PATH),
        "data": str(DATA_ROOT),
    },
    "coverage": {
        "rows": int(len(sample)),
        "wells": int(sample["well"].nunique()),
        "reference_points": int(len(point_xy)),
    },
    "loo": {
        "best_candidate": loo_best_candidate.to_dict(),
        "bank_oracle": oracle_payload["loo_all_other"]["bank"],
        "fusion_oracle": oracle_payload["loo_all_other"]["fusion"],
        "best_fixed_blend": loo_blend.to_dict(),
    },
    "outer_fold_excluded": {
        "best_candidate": outer_best_candidate.to_dict(),
        "bank_oracle": oracle_payload["outer_fold_excluded"]["bank"],
        "fusion_oracle": oracle_payload["outer_fold_excluded"]["fusion"],
        "best_fixed_blend": outer_blend.to_dict(),
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build a nested uncertainty-weighted mixture of local planes"
        if strict_pass
        else "FAIL -> close direct regional-surface interpolation; proceed to Route 6 analog-bank oracle"
    ),
}
(OUT_DIR / "result.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)
print("\nV75 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
