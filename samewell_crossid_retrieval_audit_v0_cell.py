"""Cross-ID same-physical-well retrieval audit.

This cell is read-only with respect to submission.csv. It searches the provided
training bank for renamed, shifted, reversed, or partially overlapping copies
of each test well using only deployable evidence:

* typewell GR fingerprint;
* complete horizontal GR fingerprint;
* relative MD/X/Y/Z trajectory shape;
* the test well's visible TVT_input prefix as a final guard.

The training TVT column is used only after a candidate train well has been
retrieved, to measure whether its path transfers consistently to the visible
test prefix. No hidden/test TVT is accessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, savgol_filter


EXPERIMENT = "samewell_crossid_retrieval_audit_v0"
GR_POINTS = 384
TYPEWELL_POINTS = 256
GEOMETRY_POINTS = 128
PRELIMINARY_TOPK = 10
SEARCH_POINTS = 900
MIN_SEARCH_COVERAGE = 0.45


def _find_data_root() -> Path:
    explicit = globals().get("SAMEWELL_DATA_ROOT")
    candidates = [
        Path(explicit) if explicit else None,
        Path.cwd(),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "train").exists() and (candidate / "test").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find train/ and test/ directories")


def _default_output_dir(data_root: Path) -> Path:
    explicit = globals().get("SAMEWELL_OUTPUT_DIR")
    if explicit:
        return Path(explicit)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / EXPERIMENT
    autodl = Path("/root/autodl-tmp/oof_exports")
    if autodl.exists():
        return autodl / EXPERIMENT
    return data_root / "output" / EXPERIMENT


DATA_ROOT = _find_data_root()
OUTPUT_DIR = _default_output_dir(DATA_ROOT)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _well_id(path: Path) -> str:
    return path.stem.replace("__horizontal_well", "").replace("__typewell", "")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    positions = np.arange(len(values), dtype=np.float64)
    values[~finite] = np.interp(positions[~finite], positions[finite], values[finite])
    return values


def _aggregate_axis(axis: np.ndarray, *values: np.ndarray):
    frame = pd.DataFrame({"axis": axis})
    for index, value in enumerate(values):
        frame[f"v{index}"] = value
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["axis"])
    if frame.empty:
        return np.empty(0), [np.empty(0) for _ in values]
    frame = frame.groupby("axis", sort=True, as_index=False).mean(numeric_only=True)
    return frame["axis"].to_numpy(np.float64), [
        _fill(frame[f"v{index}"].to_numpy(np.float64)) for index in range(len(values))
    ]


def _resample(axis: np.ndarray, values: np.ndarray, points: int) -> np.ndarray:
    axis, packed = _aggregate_axis(np.asarray(axis, np.float64), np.asarray(values, np.float64))
    if len(axis) < 2:
        return np.zeros(points, dtype=np.float32)
    values = packed[0]
    span = float(axis[-1] - axis[0])
    if not np.isfinite(span) or span <= 0:
        return np.zeros(points, dtype=np.float32)
    grid = np.linspace(axis[0], axis[-1], points)
    return np.interp(grid, axis, values).astype(np.float32)


def _standardise(values: np.ndarray) -> np.ndarray:
    values = _fill(np.asarray(values, np.float64))
    centre = float(np.median(values))
    scale = float(np.median(np.abs(values - centre)) * 1.4826)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - centre) / scale).astype(np.float32)


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32)


def _safe_ncc(left: np.ndarray, right: np.ndarray) -> float:
    left = _standardise(left).astype(np.float64)
    right = _standardise(right).astype(np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = _fill(values)
    selected = min(int(window), len(values))
    if selected % 2 == 0:
        selected -= 1
    if selected < 5:
        return values
    return savgol_filter(values, selected, min(3, selected - 2))


def _canonical_typewell_hash(tvt: np.ndarray, gr: np.ndarray) -> str:
    tvt, packed = _aggregate_axis(tvt, gr)
    if len(tvt) < 2:
        return ""
    gr = packed[0]
    relative = tvt - tvt[0]
    payload = np.column_stack(
        [np.round(relative, 3), np.round(gr, 3)]
    ).astype("<f4", copy=False)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _relative_geometry(md, x, y, z, reverse: bool = False) -> np.ndarray:
    md, packed = _aggregate_axis(md, x, y, z)
    if len(md) < 2:
        return np.zeros(GEOMETRY_POINTS * 3, dtype=np.float32)
    x, y, z = packed
    if reverse:
        md = -md[::-1]
        x, y, z = x[::-1], y[::-1], z[::-1]
    grid = np.linspace(md[0], md[-1], GEOMETRY_POINTS)
    coordinates = []
    scale = max(float(abs(md[-1] - md[0])), 1.0)
    for values in (x, y, z):
        sampled = np.interp(grid, md, values)
        sampled = (sampled - sampled[0]) / scale
        coordinates.append(sampled)
    return _unit(np.concatenate(coordinates))


@dataclass
class Fingerprint:
    well: str
    horizontal_path: Path
    typewell_path: Path | None
    rows: int
    md_min: float
    md_max: float
    gr: np.ndarray
    gr_reverse: np.ndarray
    geometry: np.ndarray
    geometry_reverse: np.ndarray
    typewell: np.ndarray
    typewell_hash: str


def _fingerprint(horizontal_path: Path) -> Fingerprint:
    well = _well_id(horizontal_path)
    typewell_path = horizontal_path.with_name(f"{well}__typewell.csv")
    horizontal = pd.read_csv(horizontal_path, usecols=lambda c: c in {"MD", "X", "Y", "Z", "GR"})
    md = _numeric(horizontal, "MD")
    gr_raw = _numeric(horizontal, "GR")
    gr = _unit(_standardise(_resample(md, gr_raw, GR_POINTS)))
    gr_reverse = _unit(_standardise(_resample(-md[::-1], gr_raw[::-1], GR_POINTS)))
    geometry = _relative_geometry(
        md,
        _numeric(horizontal, "X"),
        _numeric(horizontal, "Y"),
        _numeric(horizontal, "Z"),
        reverse=False,
    )
    geometry_reverse = _relative_geometry(
        md,
        _numeric(horizontal, "X"),
        _numeric(horizontal, "Y"),
        _numeric(horizontal, "Z"),
        reverse=True,
    )

    typewell_vector = np.zeros(TYPEWELL_POINTS, dtype=np.float32)
    typewell_hash = ""
    if typewell_path.exists():
        typewell_frame = pd.read_csv(typewell_path, usecols=lambda c: c in {"TVT", "GR"})
        tvt = _numeric(typewell_frame, "TVT")
        tw_gr = _numeric(typewell_frame, "GR")
        typewell_vector = _unit(_standardise(_resample(tvt, tw_gr, TYPEWELL_POINTS)))
        typewell_hash = _canonical_typewell_hash(tvt, tw_gr)

    finite_md = md[np.isfinite(md)]
    return Fingerprint(
        well=well,
        horizontal_path=horizontal_path,
        typewell_path=typewell_path if typewell_path.exists() else None,
        rows=len(horizontal),
        md_min=float(np.min(finite_md)),
        md_max=float(np.max(finite_md)),
        gr=gr,
        gr_reverse=gr_reverse,
        geometry=geometry,
        geometry_reverse=geometry_reverse,
        typewell=typewell_vector,
        typewell_hash=typewell_hash,
    )


def _build_fingerprints(directory: Path, label: str) -> list[Fingerprint]:
    paths = sorted(directory.glob("*__horizontal_well.csv"))
    records = []
    for index, path in enumerate(paths, 1):
        records.append(_fingerprint(path))
        if index % 100 == 0 or index == len(paths):
            print(f"fingerprint {label}: {index}/{len(paths)}", flush=True)
    return records


def _preliminary_scores(query: Fingerprint, bank: list[Fingerprint]) -> pd.DataFrame:
    bank_gr = np.stack([record.gr for record in bank])
    bank_gr_reverse = np.stack([record.gr_reverse for record in bank])
    bank_geometry = np.stack([record.geometry for record in bank])
    bank_geometry_reverse = np.stack([record.geometry_reverse for record in bank])
    bank_typewell = np.stack([record.typewell for record in bank])

    gr_forward = bank_gr @ query.gr
    gr_reverse = bank_gr_reverse @ query.gr
    geometry_forward = bank_geometry @ query.geometry
    geometry_reverse = bank_geometry_reverse @ query.geometry
    forward = 0.75 * gr_forward + 0.25 * geometry_forward
    reverse = 0.75 * gr_reverse + 0.25 * geometry_reverse
    orientation = np.where(forward >= reverse, 1, -1)
    horizontal_score = np.maximum(forward, reverse)
    typewell_score = bank_typewell @ query.typewell
    score = 0.52 * typewell_score + 0.48 * horizontal_score

    exact_hash = np.asarray(
        [
            bool(query.typewell_hash)
            and query.typewell_hash == record.typewell_hash
            for record in bank
        ]
    )
    score = score + exact_hash.astype(np.float32) * 0.50
    return pd.DataFrame(
        {
            "train_well": [record.well for record in bank],
            "preliminary_score": score,
            "typewell_ncc": typewell_score,
            "horizontal_score": horizontal_score,
            "gr_fingerprint_ncc": np.maximum(gr_forward, gr_reverse),
            "geometry_fingerprint_ncc": np.where(
                orientation > 0, geometry_forward, geometry_reverse
            ),
            "fingerprint_orientation": orientation,
            "typewell_hash_match": exact_hash,
            "exact_id": [query.well == record.well for record in bank],
        }
    ).sort_values("preliminary_score", ascending=False, ignore_index=True)


def _alignment_score(query_gr, train_md, train_gr, mapped_md):
    valid = (
        np.isfinite(mapped_md)
        & (mapped_md >= train_md[0])
        & (mapped_md <= train_md[-1])
    )
    coverage = float(np.mean(valid))
    if valid.sum() < 80 or coverage < MIN_SEARCH_COVERAGE:
        return -2.0, coverage
    transferred = np.interp(mapped_md[valid], train_md, train_gr)
    score = _safe_ncc(query_gr[valid], transferred)
    return score, coverage


def _horizontal_arrays(frame: pd.DataFrame, tvt_column: str):
    columns = ["MD", "GR", "X", "Y", "Z", tvt_column]
    work = pd.DataFrame(
        {
            column: pd.to_numeric(
                frame[column] if column in frame else np.nan,
                errors="coerce",
            )
            for column in columns
        }
    )
    work = (
        work.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["MD"])
        .groupby("MD", sort=True, as_index=False)
        .mean(numeric_only=True)
    )
    md = work["MD"].to_numpy(np.float64)
    values = [
        _fill(work[column].to_numpy(np.float64))
        for column in ("GR", "X", "Y", "Z")
    ]
    # This mask is the deployment contract and must never be interpolated.
    values.append(work[tvt_column].to_numpy(np.float64))
    return md, values


def _xcorr_alignment_starts(qmd, qgr, tmd, tgr, orientation):
    q_step = np.nanmedian(np.diff(qmd))
    t_step = np.nanmedian(np.diff(tmd))
    step = float(max(q_step, t_step, 0.5))
    q_grid = np.arange(qmd[0], qmd[-1] + 0.5 * step, step)
    t_grid = np.arange(tmd[0], tmd[-1] + 0.5 * step, step)
    if len(q_grid) < 80 or len(t_grid) < 80:
        return np.empty(0)
    q_values = np.interp(q_grid, qmd, qgr)
    t_values = np.interp(t_grid, tmd, tgr)
    if orientation < 0:
        q_values = q_values[::-1]
    q_values = _standardise(q_values).astype(np.float64)
    t_values = _standardise(t_values).astype(np.float64)
    correlation = correlate(t_values, q_values, mode="full", method="fft")
    lags = correlation_lags(len(t_values), len(q_values), mode="full")
    order = np.argsort(correlation)[::-1]
    starts = []
    minimum_gap = max(2, int(round(10.0 / step)))
    selected_lags = []
    for index in order:
        lag = int(lags[index])
        if any(abs(lag - previous) < minimum_gap for previous in selected_lags):
            continue
        segment_start = float(t_grid[0] + lag * step)
        start = segment_start if orientation > 0 else segment_start + (q_grid[-1] - q_grid[0])
        starts.append(start)
        selected_lags.append(lag)
        if len(starts) >= 12:
            break
    return np.asarray(starts, dtype=np.float64)


def _search_alignment(query: pd.DataFrame, train: pd.DataFrame, preferred_orientation: int):
    qmd, packed = _horizontal_arrays(query, "TVT_input")
    if len(qmd) > SEARCH_POINTS:
        indices = np.linspace(0, len(qmd) - 1, SEARCH_POINTS).round().astype(int)
        qmd_search = qmd[indices]
        qgr_search = packed[0][indices]
    else:
        qmd_search = qmd
        qgr_search = packed[0]

    tmd, train_packed = _horizontal_arrays(train, "TVT")
    if len(qmd_search) < 80 or len(tmd) < 80:
        raise RuntimeError("too few horizontal rows for alignment")

    qspan = max(float(qmd_search[-1] - qmd_search[0]), 1.0)
    tspan = max(float(tmd[-1] - tmd[0]), 1.0)
    scale_candidates = np.unique(
        np.clip(
            np.asarray([1.0, tspan / qspan, 0.98, 0.99, 1.01, 1.02]),
            0.75,
            1.35,
        )
    )
    orientations = [preferred_orientation, -preferred_orientation]
    candidates = []
    for orientation in orientations:
        xcorr_starts = _xcorr_alignment_starts(
            qmd_search,
            qgr_search,
            tmd,
            train_packed[0],
            orientation,
        )
        for absolute_scale in scale_candidates:
            signed_scale = float(orientation * absolute_scale)
            mapped_span = abs(signed_scale) * qspan
            low = float(tmd[0] - mapped_span * (1.0 - MIN_SEARCH_COVERAGE))
            high = float(tmd[-1] + mapped_span * (1.0 - MIN_SEARCH_COVERAGE))
            starts = np.linspace(low, high, 31)
            starts = np.r_[starts, qmd_search[0], tmd[0], tmd[-1]]
            if abs(absolute_scale - 1.0) <= 0.02:
                starts = np.r_[starts, xcorr_starts]
            for start in np.unique(starts):
                mapped = start + signed_scale * (qmd_search - qmd_search[0])
                score, coverage = _alignment_score(
                    qgr_search, tmd, train_packed[0], mapped
                )
                candidates.append((score, coverage, start, signed_scale))

    ordered = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)
    coarse_step = max(tspan, qspan) / 30.0
    seeds = []
    for candidate in ordered:
        if any(
            candidate[3] == previous[3]
            and abs(candidate[2] - previous[2]) < max(coarse_step * 0.20, 2.0)
            for previous in seeds
        ):
            continue
        seeds.append(candidate)
        if len(seeds) >= 8:
            break

    refined = []
    for seed in seeds:
        for start in np.linspace(
            seed[2] - 0.6 * coarse_step,
            seed[2] + 0.6 * coarse_step,
            41,
        ):
            mapped = start + seed[3] * (qmd_search - qmd_search[0])
            score, coverage = _alignment_score(
                qgr_search, tmd, train_packed[0], mapped
            )
            refined.append((score, coverage, start, seed[3]))

    alignment_candidates = ordered[:30] + refined
    top_gr_score = max(item[0] for item in alignment_candidates)

    def prefix_guard(item):
        mapped = item[2] + item[3] * (qmd - qmd[0])
        known = (
            np.isfinite(packed[4])
            & (mapped >= tmd[0])
            & (mapped <= tmd[-1])
        )
        if known.sum() < 20:
            return math.inf
        transferred = np.interp(mapped[known], tmd, train_packed[4])
        residual = packed[4][known] - transferred
        residual = residual[np.isfinite(residual)]
        if len(residual) < 20:
            return math.inf
        residual = residual - np.mean(residual)
        return float(np.sqrt(np.mean(residual**2)))

    near_optimal = [
        item for item in alignment_candidates if item[0] >= top_gr_score - 0.06
    ]
    guarded = [(prefix_guard(item), item) for item in near_optimal]
    finite_guarded = [pair for pair in guarded if np.isfinite(pair[0])]
    if finite_guarded:
        _, provisional = min(
            finite_guarded,
            key=lambda pair: (pair[0], -pair[1][0], -pair[1][1]),
        )
    else:
        provisional = max(
            alignment_candidates, key=lambda item: (item[0], item[1])
        )

    fine_offsets = []
    for start in np.linspace(provisional[2] - 8.0, provisional[2] + 8.0, 65):
        mapped = start + provisional[3] * (qmd_search - qmd_search[0])
        score, coverage = _alignment_score(qgr_search, tmd, train_packed[0], mapped)
        fine_offsets.append((score, coverage, start, provisional[3]))
    fine_top = max(item[0] for item in fine_offsets)
    fine_near = [item for item in fine_offsets if item[0] >= fine_top - 0.03]
    fine_guarded = [(prefix_guard(item), item) for item in fine_near]
    fine_finite = [pair for pair in fine_guarded if np.isfinite(pair[0])]
    if fine_finite:
        _, best = min(
            fine_finite,
            key=lambda pair: (pair[0], -pair[1][0], -pair[1][1]),
        )
    else:
        best = max(fine_offsets + [provisional], key=lambda item: (item[0], item[1]))

    mapped_full = best[2] + best[3] * (qmd - qmd[0])
    valid = (mapped_full >= tmd[0]) & (mapped_full <= tmd[-1])
    transferred_gr = np.full(len(qmd), np.nan)
    transferred_gr[valid] = np.interp(mapped_full[valid], tmd, train_packed[0])
    query_gr = packed[0]
    raw_ncc = _safe_ncc(query_gr[valid], transferred_gr[valid])
    smooth21_ncc = _safe_ncc(
        _smooth(query_gr, 21)[valid], _smooth(transferred_gr, 21)[valid]
    )
    smooth101_ncc = _safe_ncc(
        _smooth(query_gr, 101)[valid], _smooth(transferred_gr, 101)[valid]
    )
    if valid.sum() >= 5:
        derivative_ncc = _safe_ncc(
            np.gradient(query_gr)[valid], np.gradient(transferred_gr)[valid]
        )
    else:
        derivative_ncc = 0.0

    geometry_errors = []
    for qvalues, tvalues in zip(packed[1:4], train_packed[1:4]):
        transferred = np.full(len(qmd), np.nan)
        transferred[valid] = np.interp(mapped_full[valid], tmd, tvalues)
        qrelative = qvalues[valid] - qvalues[valid][0]
        trelative = transferred[valid] - transferred[valid][0]
        geometry_errors.append(np.mean((qrelative - trelative) ** 2))
    geometry_scale = max(float(np.ptp(qmd[valid])), 1.0)
    geometry_rmse = float(np.sqrt(np.mean(geometry_errors)) / geometry_scale)
    geometry_similarity = float(np.exp(-30.0 * geometry_rmse))

    known = np.isfinite(packed[4]) & valid
    prefix_rows = int(known.sum())
    prefix_raw_rmse = math.nan
    prefix_bias_rmse = math.nan
    prefix_linear_rmse = math.nan
    prefix_bias = math.nan
    prefix_slope = math.nan
    if prefix_rows >= 5:
        transferred_tvt = np.interp(mapped_full[known], tmd, train_packed[4])
        observed = packed[4][known]
        residual = observed - transferred_tvt
        prefix_raw_rmse = float(np.sqrt(np.mean(residual**2)))
        prefix_bias = float(np.mean(residual))
        prefix_bias_rmse = float(np.sqrt(np.mean((residual - prefix_bias) ** 2)))
        progress = qmd[known] - qmd[known][0]
        if prefix_rows >= 20 and np.ptp(progress) > 1.0:
            prefix_slope, intercept = np.polyfit(progress, residual, 1)
            fitted = intercept + prefix_slope * progress
            prefix_linear_rmse = float(np.sqrt(np.mean((residual - fitted) ** 2)))
        else:
            prefix_slope = 0.0
            prefix_linear_rmse = prefix_bias_rmse

    eval_mask = ~np.isfinite(packed[4])
    eval_coverage = float(np.mean(valid[eval_mask])) if eval_mask.any() else 0.0
    return {
        "alignment_start_train_md": float(best[2]),
        "alignment_signed_scale": float(best[3]),
        "alignment_orientation": int(np.sign(best[3])),
        "alignment_search_ncc": float(best[0]),
        "horizontal_coverage": float(np.mean(valid)),
        "eval_overlap_coverage": eval_coverage,
        "gr_raw_ncc": raw_ncc,
        "gr_smooth21_ncc": smooth21_ncc,
        "gr_smooth101_ncc": smooth101_ncc,
        "gr_derivative_ncc": derivative_ncc,
        "geometry_rmse_normalised": geometry_rmse,
        "geometry_similarity": geometry_similarity,
        "prefix_rows": prefix_rows,
        "prefix_raw_rmse": prefix_raw_rmse,
        "prefix_bias": prefix_bias,
        "prefix_bias_rmse": prefix_bias_rmse,
        "prefix_linear_slope": prefix_slope,
        "prefix_linear_rmse": prefix_linear_rmse,
    }


def _fine_score(row: dict) -> float:
    return float(
        0.42 * row["gr_smooth21_ncc"]
        + 0.18 * row["gr_smooth101_ncc"]
        + 0.10 * row["gr_raw_ncc"]
        + 0.08 * row["gr_derivative_ncc"]
        + 0.10 * row["geometry_similarity"]
        + 0.12 * row["typewell_ncc"]
        + (0.20 if row["typewell_hash_match"] else 0.0)
    )


def _suggest_action(best: pd.Series, margin: float) -> str:
    prefix = float(best["prefix_bias_rmse"])
    gr = float(best["gr_smooth21_ncc"])
    coverage = float(best["eval_overlap_coverage"])
    hash_match = bool(best["typewell_hash_match"])
    exact_id = bool(best["exact_id"])
    if (
        np.isfinite(prefix)
        and prefix <= 0.25
        and gr >= 0.990
        and coverage >= 0.80
        and (hash_match or exact_id or margin >= 0.08)
    ):
        return "full"
    if (
        np.isfinite(prefix)
        and prefix <= 0.75
        and gr >= 0.970
        and coverage >= 0.50
        and margin >= 0.02
    ):
        return "graded"
    if np.isfinite(prefix) and prefix <= 2.0 and gr >= 0.90 and coverage >= 0.30:
        return "review"
    return "no_match"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


started = time.time()
print({"experiment": EXPERIMENT, "data_root": str(DATA_ROOT)}, flush=True)
train_bank = _build_fingerprints(DATA_ROOT / "train", "train")
test_queries = _build_fingerprints(DATA_ROOT / "test", "test")
if not train_bank or not test_queries:
    raise RuntimeError("Empty train or test fingerprint bank")

train_gr = np.stack([record.gr for record in train_bank])
train_geometry = np.stack([record.geometry for record in train_bank])
train_typewell = np.stack([record.typewell for record in train_bank])
bank_similarity = (
    0.48 * (train_gr @ train_gr.T)
    + 0.20 * (train_geometry @ train_geometry.T)
    + 0.32 * (train_typewell @ train_typewell.T)
)
np.fill_diagonal(bank_similarity, -np.inf)
best_impostor = np.max(bank_similarity, axis=1)
hash_counts = pd.Series([record.typewell_hash for record in train_bank]).value_counts()
hash_counts = hash_counts[hash_counts.index != ""]
bank_calibration = {
    "train_wells": len(train_bank),
    "typewell_hash_groups": int(len(hash_counts)),
    "typewell_hash_collision_groups": int((hash_counts > 1).sum()),
    "typewell_hash_max_group": int(hash_counts.max()) if len(hash_counts) else 0,
    "best_other_similarity_p50": float(np.quantile(best_impostor, 0.50)),
    "best_other_similarity_p95": float(np.quantile(best_impostor, 0.95)),
    "best_other_similarity_p99": float(np.quantile(best_impostor, 0.99)),
    "best_other_similarity_max": float(np.max(best_impostor)),
}
print("BANK CALIBRATION", json.dumps(_jsonable(bank_calibration), indent=2), flush=True)

candidate_rows = []
summary_rows = []
bank_lookup = {record.well: record for record in train_bank}
for query_index, query in enumerate(test_queries, 1):
    query_frame = pd.read_csv(query.horizontal_path)
    preliminary = _preliminary_scores(query, train_bank)
    hash_matches = preliminary[preliminary["typewell_hash_match"]]
    preliminary_count = 3 if len(hash_matches) else PRELIMINARY_TOPK
    selected = preliminary.head(preliminary_count).copy()
    selected = (
        pd.concat([selected, hash_matches], ignore_index=True)
        .drop_duplicates("train_well")
        .sort_values("preliminary_score", ascending=False)
        .head(max(preliminary_count, len(hash_matches)))
    )

    fine_rows = []
    for candidate in selected.itertuples(index=False):
        bank_record = bank_lookup[candidate.train_well]
        train_frame = pd.read_csv(bank_record.horizontal_path)
        try:
            aligned = _search_alignment(
                query_frame, train_frame, int(candidate.fingerprint_orientation)
            )
            row = candidate._asdict()
            row.update(aligned)
            row["query_well"] = query.well
            row["fine_score"] = _fine_score(row)
            row["status"] = "ok"
        except Exception as error:
            row = candidate._asdict()
            row.update(
                {
                    "query_well": query.well,
                    "fine_score": -np.inf,
                    "status": f"failed:{type(error).__name__}:{error}",
                }
            )
        fine_rows.append(row)

    fine = pd.DataFrame(fine_rows).sort_values("fine_score", ascending=False)
    finite = fine[np.isfinite(fine["fine_score"])]
    if finite.empty:
        summary_rows.append(
            {
                "query_well": query.well,
                "top1_train_well": None,
                "suggested_action": "no_match",
                "reason": "all candidate alignments failed",
            }
        )
    else:
        best = finite.iloc[0]
        second_score = float(finite.iloc[1]["fine_score"]) if len(finite) > 1 else -np.inf
        margin = float(best["fine_score"] - second_score) if np.isfinite(second_score) else math.inf
        action = _suggest_action(best, margin)
        summary = {
            "query_well": query.well,
            "top1_train_well": str(best["train_well"]),
            "suggested_action": action,
            "exact_id": bool(best["exact_id"]),
            "typewell_hash_match": bool(best["typewell_hash_match"]),
            "fine_score": float(best["fine_score"]),
            "top2_fine_score": second_score,
            "fine_margin": margin,
            "gr_smooth21_ncc": float(best["gr_smooth21_ncc"]),
            "gr_smooth101_ncc": float(best["gr_smooth101_ncc"]),
            "geometry_similarity": float(best["geometry_similarity"]),
            "prefix_rows": int(best["prefix_rows"]),
            "prefix_raw_rmse": float(best["prefix_raw_rmse"]),
            "prefix_bias_rmse": float(best["prefix_bias_rmse"]),
            "prefix_linear_rmse": float(best["prefix_linear_rmse"]),
            "eval_overlap_coverage": float(best["eval_overlap_coverage"]),
            "alignment_orientation": int(best["alignment_orientation"]),
            "alignment_signed_scale": float(best["alignment_signed_scale"]),
        }
        summary_rows.append(summary)
        print("QUERY RESULT", json.dumps(_jsonable(summary), indent=2), flush=True)

    fine["rank"] = np.arange(1, len(fine) + 1)
    candidate_rows.extend(fine.to_dict(orient="records"))
    print(f"query progress: {query_index}/{len(test_queries)}", flush=True)

candidate_frame = pd.DataFrame(candidate_rows)
summary_frame = pd.DataFrame(summary_rows)
candidate_frame.to_csv(OUTPUT_DIR / "samewell_candidate_scores.csv", index=False)
summary_frame.to_csv(OUTPUT_DIR / "samewell_retrieval_summary.csv", index=False)

actions = summary_frame["suggested_action"].value_counts(dropna=False).to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "read_only_cross_id_same_physical_well_retrieval",
    "data_root": str(DATA_ROOT),
    "train_wells": len(train_bank),
    "test_wells": len(test_queries),
    "bank_calibration": bank_calibration,
    "actions": actions,
    "full_or_graded_wells": int(
        summary_frame["suggested_action"].isin(["full", "graded"]).sum()
    ),
    "cross_id_full_or_graded_wells": int(
        (
            summary_frame["suggested_action"].isin(["full", "graded"])
            & ~summary_frame.get("exact_id", pd.Series(False, index=summary_frame.index)).fillna(False)
        ).sum()
    ),
    "submission_modified": False,
    "thresholds": {
        "full": {
            "prefix_bias_rmse_le": 0.25,
            "gr_smooth21_ncc_ge": 0.99,
            "eval_overlap_coverage_ge": 0.80,
        },
        "graded": {
            "prefix_bias_rmse_le": 0.75,
            "gr_smooth21_ncc_ge": 0.97,
            "eval_overlap_coverage_ge": 0.50,
            "fine_margin_ge": 0.02,
        },
    },
    "outputs": {
        "summary": str(OUTPUT_DIR / "samewell_retrieval_summary.csv"),
        "candidates": str(OUTPUT_DIR / "samewell_candidate_scores.csv"),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "Inspect cross-ID full/graded matches; do not apply overrides until pseudo-rerun controls pass.",
}
verdict = _jsonable(verdict)
(OUTPUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)

print("\nRETRIEVAL SUMMARY")
print(summary_frame.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUTPUT_DIR)
