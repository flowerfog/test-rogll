"""OOF audit for the deterministic PF bimodal midpoint hedge.

This distribution-matches the only materially new mechanism in
``rogii-mha160-sep3-r2-20260720.ipynb``:

* 128 PF seeds, 500 particles per seed;
* likelihood temperature 5;
* weighted two-means over each seed's mean evaluation TVT;
* frozen gate: minor mass >= 0.22 and separation in [3, 40] ft;
* frozen move: clip(1.6 * (branch midpoint - weighted mean), -4, 4).

The PF evidence uses only the same well's typewell, full horizontal GR, visible
TVT_input prefix, and trajectory. Truth is used only after the PF run to score
the frozen move against row-level OOF bases.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr
from tqdm.auto import tqdm


EXPERIMENT = "pf_bimodal_midpoint_hedge_oof_v0"
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
FOLD_SOURCE = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT

N_SEEDS = 128
N_PARTICLES = 500
LIKELIHOOD_TEMPERATURE = 5.0
PF_INIT_SPREAD_FT = 4.5
MAX_WORKERS = int(
    os.environ.get(
        "ROGII_PF_WORKERS", str(min(8, max(1, os.cpu_count() or 1)))
    )
)

FIXED_ALPHA = 1.6
FIXED_MINOR_MASS = 0.22
FIXED_SEPARATION_LOW_FT = 3.0
FIXED_SEPARATION_HIGH_FT = 40.0
FIXED_MOVE_CAP_FT = 4.0

NEIGHBOR_WEIGHT = 0.25
NEIGHBOR_RAW_CAP_FT = 24.0
MIN_ABS_DIRECTION_DOT = 0.70

# Set to a positive integer for a cheap smoke run. Leave at zero for the real
# audit. Existing per-well PF statistics are resumed from disk.
WELL_LIMIT = int(os.environ.get("ROGII_PF_WELL_LIMIT", "0"))
ALPHA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.6, 2.0)
BOOTSTRAP_ITERATIONS = 4000
SHUFFLE_ITERATIONS = 1000

CACHE_SIGNATURE = (
    f"seeds{N_SEEDS}_particles{N_PARTICLES}_temp{LIKELIHOOD_TEMPERATURE:g}_"
    f"spread{PF_INIT_SPREAD_FT:g}_seedbase0_rng_numba_np_random"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
if not DATA_ROOT.exists():
    raise FileNotFoundError(DATA_ROOT)
if not POOL_PATH.exists():
    raise FileNotFoundError(POOL_PATH)


def _rmse_from_sse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / max(int(rows), 1)))


def _rmse(truth, prediction) -> float:
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


# ---------------------------------------------------------------------------
# Load row-level OOF bases without changing their persisted row order.
# ---------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
required_columns = ["id", "tvt_true"]
optional_columns = ["well", "row_idx", "fold", "strong", "strong_bias", "cp12"]
usecols = [column for column in required_columns + optional_columns if column in header]
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
    low_memory=False,
)
if not set(required_columns).issubset(pool.columns):
    raise RuntimeError(f"Missing required pool columns: {required_columns}")
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
truth = pool["tvt_true"].to_numpy(np.float64)
if not np.isfinite(truth).all():
    raise RuntimeError("Non-finite truth in candidate pool")

if "fold" not in pool:
    if not FOLD_SOURCE.exists():
        raise FileNotFoundError(FOLD_SOURCE)
    fold_frame = pd.read_csv(FOLD_SOURCE, dtype={"well": str})
    fold_map = fold_frame.groupby("well")["fold"].first().astype(int).to_dict()
    pool["fold"] = pool["well"].map(fold_map)
if pool["fold"].isna().any():
    raise RuntimeError("Fold mapping is incomplete")
pool["fold"] = pool["fold"].astype(np.int8)

wells = sorted(pool["well"].unique())
if WELL_LIMIT:
    wells = wells[: int(WELL_LIMIT)]
    selected = pool["well"].isin(wells).to_numpy(bool)
    pool = pool.loc[selected].reset_index(drop=True)
    truth = pool["tvt_true"].to_numpy(np.float64)

pool_indices = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in pool.groupby("well", sort=False).indices.items()
}
predictions = {}
for column in ("strong", "strong_bias", "cp12"):
    if column in pool:
        values = pool[column].to_numpy(np.float64)
        if np.isfinite(values).all():
            predictions[column] = values


def _eligible_neighbor_rows(source_dir: Path) -> np.ndarray:
    selected_path = source_dir / "selected_neighbors.csv"
    if not selected_path.exists():
        return np.zeros(len(pool), bool)
    selected = pd.read_csv(selected_path, dtype={"well": str}).set_index("well")
    direction_column = (
        "abs_direction_dot" if "abs_direction_dot" in selected else "direction_dot"
    )
    direction = selected[direction_column].abs()
    eligible_well = (
        (selected["median_distance"] <= 600.0)
        & (selected["p90_distance"] <= 1200.0)
        & (selected["anchor_distance_full"] <= 1200.0)
        & (direction >= MIN_ABS_DIRECTION_DOT)
    )
    return pool["well"].map(eligible_well.to_dict()).fillna(False).to_numpy(bool)


neighbor_source = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
eligible_neighbor = _eligible_neighbor_rows(neighbor_source)
transfer_path = neighbor_source / "neighbor_transfer_oof.npz"
if transfer_path.exists() and "cp12" in predictions:
    transfer = np.load(transfer_path, allow_pickle=False)
    for method, name in (
        ("structure_progress", "neighbor_frozen_cp12"),
        ("structure_xy", "neighbor_xy_cp12"),
    ):
        if method not in transfer:
            continue
        candidate = transfer[method].astype(np.float64)
        if len(candidate) != len(pool):
            continue
        base = predictions["cp12"]
        move = np.zeros(len(pool), np.float64)
        valid = eligible_neighbor & np.isfinite(candidate)
        move[valid] = NEIGHBOR_WEIGHT * np.clip(
            candidate[valid] - base[valid],
            -NEIGHBOR_RAW_CAP_FT,
            NEIGHBOR_RAW_CAP_FT,
        )
        predictions[name] = base + move

geometry_path = (
    OOF_ROOT / "neighbor_fullcurve_projection_direction_v1" / "primary_candidate_oof.npz"
)
if geometry_path.exists() and "cp12" in predictions:
    candidate = np.load(geometry_path, allow_pickle=False)["candidate"].astype(np.float64)
    if len(candidate) == len(pool):
        base = predictions["cp12"]
        move = np.zeros(len(pool), np.float64)
        valid = eligible_neighbor & np.isfinite(candidate)
        move[valid] = NEIGHBOR_WEIGHT * np.clip(
            candidate[valid] - base[valid],
            -NEIGHBOR_RAW_CAP_FT,
            NEIGHBOR_RAW_CAP_FT,
        )
        predictions["neighbor_geom_cp12"] = base + move

if not predictions:
    raise RuntimeError("No complete OOF base predictions were found")
primary_base = (
    "neighbor_geom_cp12"
    if "neighbor_geom_cp12" in predictions
    else "neighbor_frozen_cp12"
    if "neighbor_frozen_cp12" in predictions
    else "cp12"
    if "cp12" in predictions
    else min(predictions, key=lambda name: _rmse(truth, predictions[name]))
)
print(
    {
        "experiment": EXPERIMENT,
        "rows": len(pool),
        "wells": len(wells),
        "bases": {name: _rmse(truth, values) for name, values in predictions.items()},
        "primary_base": primary_base,
        "workers": MAX_WORKERS,
        "cache_signature": CACHE_SIGNATURE,
    },
    flush=True,
)


# ---------------------------------------------------------------------------
# Distribution-matched seed-cloud calculation from the referenced notebook.
# The public notebook uses numpy.default_rng while this accelerated Numba loop
# uses Numba's seeded np.random implementation. The equations and seed count
# match, but individual particles are not bitwise-identical.
# ---------------------------------------------------------------------------
@njit(cache=False)
def _interp1(grid, value, minimum, step):
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    maximum_index = len(grid) - 1
    if index >= maximum_index:
        return grid[maximum_index]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=False, nogil=True)
def _pf_lik_allseeds(
    md_values,
    z_values,
    gr_values,
    gr_grid,
    tvt_minimum,
    tvt_step,
    gr_sigma,
    initial_structure,
    initial_rate,
    particles,
    seeds,
    seed_base,
    momentum,
    rate_noise,
    position_noise,
    rough_position,
    rough_rate,
    resample_fraction,
    initial_spread,
):
    rows = len(md_values)
    seed_predictions = np.empty((seeds, rows))
    seed_likelihoods = np.empty(seeds)
    tvt_maximum = tvt_minimum + len(gr_grid) * tvt_step
    for seed in range(seeds):
        np.random.seed(seed_base + seed)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = initial_structure + initial_spread * np.random.randn()
            rate[particle] = initial_rate + 0.01 * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_values[0] - 1.0
        for row in range(rows):
            delta_md = md_values[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = momentum * rate[particle] + rate_noise * np.random.randn()
                position[particle] += (
                    rate[particle] * delta_md + position_noise * np.random.randn()
                )
                particle_tvt = position[particle] - z_values[row]
                if particle_tvt < tvt_minimum - 100.0:
                    particle_tvt = tvt_minimum - 100.0
                if particle_tvt > tvt_maximum + 100.0:
                    particle_tvt = tvt_maximum + 100.0
                position[particle] = particle_tvt + z_values[row]

            average_likelihood = 0.0
            for particle in range(particles):
                expected_gr = _interp1(
                    gr_grid,
                    position[particle] - z_values[row],
                    tvt_minimum,
                    tvt_step,
                )
                residual = (gr_values[row] - expected_gr) / gr_sigma
                squared = residual * residual
                if squared > 600.0:
                    squared = 600.0
                likelihood = np.exp(-0.5 * squared)
                if likelihood < 1e-300:
                    likelihood = 1e-300
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            if average_likelihood < 1e-300:
                average_likelihood = 1e-300
            log_likelihood += np.log(average_likelihood)

            weight_sum = 0.0
            for particle in range(particles):
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(particles):
                    weights[particle] = 1.0 / particles

            inverse_effective = 0.0
            for particle in range(particles):
                inverse_effective += weights[particle] * weights[particle]
            effective = 1.0 / inverse_effective
            if effective < resample_fraction * particles:
                cumulative = np.empty(particles)
                cumulative_value = 0.0
                for particle in range(particles):
                    cumulative_value += weights[particle]
                    cumulative[particle] = cumulative_value
                start = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles)
                new_rate = np.empty(particles)
                cumulative_index = 0
                for particle in range(particles):
                    threshold = start + particle / particles
                    while (
                        cumulative_index < particles - 1
                        and cumulative[cumulative_index] < threshold
                    ):
                        cumulative_index += 1
                    new_position[particle] = (
                        position[cumulative_index] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = rate[cumulative_index] + rough_rate * np.random.randn()
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles

            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (position[particle] - z_values[row])
            seed_predictions[seed, row] = estimate
            previous_md = md_values[row]
        seed_likelihoods[seed] = log_likelihood
    return seed_predictions, seed_likelihoods


def _typewell_grid(tvt, gr, step=0.2):
    minimum = float(np.min(tvt))
    maximum = float(np.max(tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    return np.interp(grid_tvt, tvt, gr).astype(np.float64), minimum, float(step)


def _weighted_two_means(values, weights):
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    center_one = float(np.quantile(values, 0.1))
    center_two = float(np.quantile(values, 0.9))
    for _ in range(25):
        first = np.abs(values - center_one) <= np.abs(values - center_two)
        weight_one = float(weights[first].sum())
        weight_two = float(weights[~first].sum())
        if weight_one <= 0.0 or weight_two <= 0.0:
            break
        next_one = float(np.sum(weights[first] * values[first]) / weight_one)
        next_two = float(np.sum(weights[~first] * values[~first]) / weight_two)
        movement = abs(next_one - center_one) + abs(next_two - center_two)
        center_one, center_two = next_one, next_two
        if movement < 1e-9:
            break
    first = np.abs(values - center_one) <= np.abs(values - center_two)
    return (
        center_one,
        float(weights[first].sum()),
        center_two,
        float(weights[~first].sum()),
    )


def _compute_well_statistics(well: str) -> dict:
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)

    known_mask = horizontal["TVT_input"].notna().to_numpy(bool)
    eval_mask = ~known_mask
    if known_mask.sum() < 30 or eval_mask.sum() < 10:
        raise RuntimeError(
            f"insufficient prefix/eval rows: {known_mask.sum()}/{eval_mask.sum()}"
        )

    tw_valid = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
    if len(tw_valid) < 20:
        raise RuntimeError("insufficient typewell rows")
    tw_tvt = tw_valid["TVT"].to_numpy(np.float64)
    tw_gr = tw_valid["GR"].to_numpy(np.float64)

    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last = known.iloc[-1]
    initial_structure = float(last["TVT_input"]) + float(last["Z"])
    expected_known_gr = np.interp(
        known["TVT_input"].to_numpy(np.float64), tw_tvt, tw_gr
    )
    observed_known_gr = known["GR"].fillna(0.0).to_numpy(np.float64)
    gr_sigma = float(
        np.clip(np.nanstd(observed_known_gr - expected_known_gr), 10.0, 60.0)
    )

    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(tail["Z"].to_numpy(np.float64))
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    valid_delta = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[valid_delta] / delta_md[valid_delta]))
        if valid_delta.sum() >= 3
        else 0.0
    )

    gr_grid, tvt_minimum, tvt_step = _typewell_grid(tw_tvt, tw_gr)
    horizontal_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.mean(tw_gr)))
        .to_numpy(np.float64)
    )
    seed_predictions, seed_likelihoods = _pf_lik_allseeds(
        evaluation["MD"].to_numpy(np.float64),
        evaluation["Z"].to_numpy(np.float64),
        horizontal_gr[eval_mask],
        gr_grid,
        tvt_minimum,
        tvt_step,
        gr_sigma,
        initial_structure,
        initial_rate,
        N_PARTICLES,
        N_SEEDS,
        0,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        PF_INIT_SPREAD_FT,
    )

    centered_likelihood = seed_likelihoods - np.max(seed_likelihoods)
    weights = np.exp(centered_likelihood / LIKELIHOOD_TEMPERATURE)
    weights /= weights.sum()
    seed_levels = seed_predictions.mean(axis=1)
    weighted_level = float(np.sum(weights * seed_levels))
    center_one, weight_one, center_two, weight_two = _weighted_two_means(
        seed_levels, weights
    )
    midpoint_delta = 0.5 * (center_one + center_two) - weighted_level

    weighted_prediction = np.sum(weights[:, None] * seed_predictions, axis=0)
    truth_values = evaluation["TVT"].to_numpy(np.float64)
    truth_valid = np.isfinite(truth_values)
    pf_rmse = (
        _rmse(truth_values[truth_valid], weighted_prediction[truth_valid])
        if truth_valid.any()
        else np.nan
    )
    truth_level = float(np.mean(truth_values[truth_valid])) if truth_valid.any() else np.nan

    return {
        "well": str(well),
        "cache_signature": CACHE_SIGNATURE,
        "known_rows": int(known_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
        "gr_sigma": gr_sigma,
        "likelihood_max_per_row": float(np.max(seed_likelihoods) / eval_mask.sum()),
        "likelihood_spread": float(np.std(seed_likelihoods)),
        "center_one": center_one,
        "center_two": center_two,
        "weight_one": weight_one,
        "weight_two": weight_two,
        "minor_mass": min(weight_one, weight_two),
        "separation_ft": abs(center_one - center_two),
        "weighted_level": weighted_level,
        "midpoint_delta_ft": midpoint_delta,
        "frozen_shift_ft": float(
            np.clip(FIXED_ALPHA * midpoint_delta, -FIXED_MOVE_CAP_FT, FIXED_MOVE_CAP_FT)
        ),
        "pf_scale5_rmse": pf_rmse,
        "pf_scale5_optimal_datum_shift_ft": (
            truth_level - weighted_level if np.isfinite(truth_level) else np.nan
        ),
    }


# Compile before launching concurrent calls. This avoids compilation races.
_warm_md = np.arange(1.0, 9.0)
_warm_z = np.zeros(8)
_warm_gr = np.full(8, 50.0)
_warm_grid = np.linspace(40.0, 60.0, 32)
_pf_lik_allseeds(
    _warm_md,
    _warm_z,
    _warm_gr,
    _warm_grid,
    0.0,
    1.0,
    20.0,
    50.0,
    0.0,
    8,
    2,
    0,
    0.998,
    0.002,
    0.005,
    0.1,
    0.001,
    0.5,
    4.5,
)

statistics_path = OUT_DIR / "bimodal_well_statistics.csv"
failure_path = OUT_DIR / "bimodal_failures.csv"
statistics_by_well = {}
if statistics_path.exists():
    cached = pd.read_csv(statistics_path, dtype={"well": str})
    if "cache_signature" in cached:
        cached = cached[cached["cache_signature"].eq(CACHE_SIGNATURE)]
    statistics_by_well = {
        str(row["well"]): row for row in cached.to_dict(orient="records")
    }

missing_wells = [well for well in wells if well not in statistics_by_well]
failures = []
started = time.time()
if missing_wells:
    print(
        f"PF seed clouds: cached={len(statistics_by_well)} "
        f"missing={len(missing_wells)} workers={MAX_WORKERS}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_well = {
            executor.submit(_compute_well_statistics, well): well for well in missing_wells
        }
        for completed, future in enumerate(
            tqdm(
                as_completed(future_to_well),
                total=len(future_to_well),
                desc="PF bimodal statistics",
            ),
            1,
        ):
            well = future_to_well[future]
            try:
                result = future.result()
                statistics_by_well[well] = result
            except Exception as exc:
                failures.append({"well": well, "reason": repr(exc)})
            if completed % 10 == 0 or completed == len(missing_wells):
                pd.DataFrame(statistics_by_well.values()).sort_values("well").to_csv(
                    statistics_path, index=False
                )
                pd.DataFrame(failures).to_csv(failure_path, index=False)

statistics = pd.DataFrame(statistics_by_well.values())
if len(statistics):
    statistics = statistics.sort_values("well").reset_index(drop=True)
statistics.to_csv(statistics_path, index=False)
pd.DataFrame(failures).to_csv(failure_path, index=False)
if len(statistics) < max(20, int(0.95 * len(wells))):
    raise RuntimeError(f"Only {len(statistics)} / {len(wells)} PF statistics succeeded")


# ---------------------------------------------------------------------------
# Score constant per-well moves using sufficient SSE statistics.
# ---------------------------------------------------------------------------
def _frozen_shift(frame: pd.DataFrame, alpha=FIXED_ALPHA) -> np.ndarray:
    qualified = (
        (frame["minor_mass"].to_numpy(float) >= FIXED_MINOR_MASS)
        & (frame["separation_ft"].to_numpy(float) >= FIXED_SEPARATION_LOW_FT)
        & (frame["separation_ft"].to_numpy(float) <= FIXED_SEPARATION_HIGH_FT)
    )
    shift = np.zeros(len(frame), np.float64)
    proposed = alpha * frame["midpoint_delta_ft"].to_numpy(np.float64)
    shift[qualified] = np.clip(
        proposed[qualified], -FIXED_MOVE_CAP_FT, FIXED_MOVE_CAP_FT
    )
    shift[np.abs(shift) <= 0.01] = 0.0
    return shift


def _sse_after(frame: pd.DataFrame, shift: np.ndarray) -> np.ndarray:
    return (
        frame["base_sse"].to_numpy(np.float64)
        + 2.0 * shift * frame["error_sum"].to_numpy(np.float64)
        + frame["rows"].to_numpy(np.float64) * shift * shift
    )


def _base_well_frame(base_name: str) -> pd.DataFrame:
    prediction = predictions[base_name]
    records = []
    fold_by_well = pool.groupby("well")["fold"].first().astype(int).to_dict()
    for well in wells:
        indices = pool_indices[well]
        error = prediction[indices] - truth[indices]
        records.append(
            {
                "well": well,
                "fold": int(fold_by_well[well]),
                "rows": int(len(indices)),
                "base_sse": float(np.sum(error * error)),
                "error_sum": float(np.sum(error)),
                "optimal_constant_shift_ft": float(-np.mean(error)),
            }
        )
    frame = pd.DataFrame(records).merge(statistics, on="well", how="left", validate="one_to_one")
    return frame


base_summaries = []
fold_reports = []
per_base_frames = {}
for base_name in predictions:
    frame = _base_well_frame(base_name)
    shift = _frozen_shift(frame)
    candidate_sse = _sse_after(frame, shift)
    base_sse = frame["base_sse"].to_numpy(np.float64)
    rows = frame["rows"].to_numpy(np.int64)
    total_rows = int(rows.sum())
    base_rmse = _rmse_from_sse(base_sse.sum(), total_rows)
    candidate_rmse = _rmse_from_sse(candidate_sse.sum(), total_rows)
    qualified = np.abs(shift) > 0.0
    per_well_gain = np.sqrt(base_sse / rows) - np.sqrt(candidate_sse / rows)

    if qualified.sum() >= 3:
        corr = float(
            spearmanr(
                frame.loc[qualified, "midpoint_delta_ft"],
                frame.loc[qualified, "optimal_constant_shift_ft"],
            ).statistic
        )
        sign_accuracy = float(
            np.mean(
                np.sign(shift[qualified])
                == np.sign(frame.loc[qualified, "optimal_constant_shift_ft"].to_numpy(float))
            )
        )
    else:
        corr = np.nan
        sign_accuracy = np.nan

    reverse_sse = _sse_after(frame, -shift)
    reverse_gain = base_rmse - _rmse_from_sse(reverse_sse.sum(), total_rows)
    base_summaries.append(
        {
            "base": base_name,
            "rows": total_rows,
            "wells": len(frame),
            "qualified_wells": int(qualified.sum()),
            "base_rmse": base_rmse,
            "prediction_rmse": candidate_rmse,
            "gain": base_rmse - candidate_rmse,
            "positive_qualified_fraction": (
                float(np.mean(per_well_gain[qualified] > 0.0)) if qualified.any() else 0.0
            ),
            "spearman_midpoint_vs_optimal_shift": corr,
            "sign_accuracy": sign_accuracy,
            "mean_abs_shift": float(np.mean(np.abs(shift))),
            "p95_abs_shift": float(np.quantile(np.abs(shift), 0.95)),
            "max_abs_shift": float(np.max(np.abs(shift))),
            "reverse_sign_gain": reverse_gain,
        }
    )
    frame["frozen_shift_ft"] = shift
    frame["candidate_sse"] = candidate_sse
    frame["well_gain"] = per_well_gain
    per_base_frames[base_name] = frame

    for fold in sorted(frame["fold"].unique()):
        mask = frame["fold"].eq(fold).to_numpy(bool)
        fold_rows = int(rows[mask].sum())
        fold_base = _rmse_from_sse(base_sse[mask].sum(), fold_rows)
        fold_prediction = _rmse_from_sse(candidate_sse[mask].sum(), fold_rows)
        fold_reports.append(
            {
                "base": base_name,
                "fold": int(fold),
                "rows": fold_rows,
                "wells": int(mask.sum()),
                "qualified_wells": int(qualified[mask].sum()),
                "base_rmse": fold_base,
                "prediction_rmse": fold_prediction,
                "gain": fold_base - fold_prediction,
            }
        )

base_summary = pd.DataFrame(base_summaries).sort_values("gain", ascending=False)
fold_report = pd.DataFrame(fold_reports)
base_summary.to_csv(OUT_DIR / "fixed_recipe_bases.csv", index=False)
fold_report.to_csv(OUT_DIR / "fixed_recipe_folds.csv", index=False)
print("\nFIXED RECIPE BASES")
print(base_summary.to_string(index=False))
print("\nFIXED RECIPE FOLDS")
print(fold_report.to_string(index=False))


primary = per_base_frames[primary_base].copy()
primary_shift = primary["frozen_shift_ft"].to_numpy(np.float64)
primary_base_sse = primary["base_sse"].to_numpy(np.float64)
primary_rows = primary["rows"].to_numpy(np.int64)
primary_candidate_sse = primary["candidate_sse"].to_numpy(np.float64)
total_rows = int(primary_rows.sum())
primary_base_rmse = _rmse_from_sse(primary_base_sse.sum(), total_rows)
primary_prediction_rmse = _rmse_from_sse(primary_candidate_sse.sum(), total_rows)
primary_gain = primary_base_rmse - primary_prediction_rmse

# Alpha response is cheap once the PF seed clouds have been cached.
alpha_rows = []
for alpha in ALPHA_GRID:
    shift = _frozen_shift(primary, alpha=float(alpha))
    candidate_sse = _sse_after(primary, shift)
    alpha_rows.append(
        {
            "alpha": float(alpha),
            "rmse": _rmse_from_sse(candidate_sse.sum(), total_rows),
            "gain": primary_base_rmse - _rmse_from_sse(candidate_sse.sum(), total_rows),
            "qualified_wells": int((np.abs(shift) > 0.0).sum()),
            "mean_abs_shift": float(np.mean(np.abs(shift))),
            "max_abs_shift": float(np.max(np.abs(shift))),
        }
    )
alpha_scan = pd.DataFrame(alpha_rows)
alpha_scan.to_csv(OUT_DIR / "alpha_scan_diagnostic.csv", index=False)

# Nested alpha selection: each outer fold receives an alpha chosen exclusively
# from the other four folds. The gate thresholds remain frozen.
nested_rows = []
nested_candidate_sse = np.zeros(len(primary), np.float64)
for outer_fold in sorted(primary["fold"].unique()):
    train_mask = primary["fold"].ne(outer_fold).to_numpy(bool)
    valid_mask = ~train_mask
    candidates = []
    for alpha in ALPHA_GRID:
        shift = _frozen_shift(primary, alpha=float(alpha))
        candidate_sse = _sse_after(primary, shift)
        train_rows = int(primary_rows[train_mask].sum())
        train_gain = _rmse_from_sse(primary_base_sse[train_mask].sum(), train_rows) - _rmse_from_sse(
            candidate_sse[train_mask].sum(), train_rows
        )
        candidates.append((train_gain, -float(alpha), float(alpha), candidate_sse, shift))
    selected = max(candidates, key=lambda item: (item[0], item[1]))
    selected_alpha = selected[2]
    candidate_sse = selected[3]
    nested_candidate_sse[valid_mask] = candidate_sse[valid_mask]
    valid_rows = int(primary_rows[valid_mask].sum())
    valid_base = _rmse_from_sse(primary_base_sse[valid_mask].sum(), valid_rows)
    valid_prediction = _rmse_from_sse(candidate_sse[valid_mask].sum(), valid_rows)
    nested_rows.append(
        {
            "fold": int(outer_fold),
            "selected_alpha": selected_alpha,
            "train_gain": selected[0],
            "valid_base_rmse": valid_base,
            "valid_prediction_rmse": valid_prediction,
            "valid_gain": valid_base - valid_prediction,
        }
    )
nested_report = pd.DataFrame(nested_rows)
nested_report.to_csv(OUT_DIR / "nested_alpha_folds.csv", index=False)
nested_rmse = _rmse_from_sse(nested_candidate_sse.sum(), total_rows)

# Well bootstrap and well-assignment shuffle controls.
rng = np.random.default_rng(20260720)
bootstrap_gains = np.empty(BOOTSTRAP_ITERATIONS, np.float64)
for iteration in range(BOOTSTRAP_ITERATIONS):
    sample = rng.integers(0, len(primary), len(primary))
    sampled_rows = int(primary_rows[sample].sum())
    bootstrap_gains[iteration] = _rmse_from_sse(
        primary_base_sse[sample].sum(), sampled_rows
    ) - _rmse_from_sse(primary_candidate_sse[sample].sum(), sampled_rows)

shuffle_gains = np.empty(SHUFFLE_ITERATIONS, np.float64)
for iteration in range(SHUFFLE_ITERATIONS):
    shuffled_shift = rng.permutation(primary_shift)
    shuffled_sse = _sse_after(primary, shuffled_shift)
    shuffle_gains[iteration] = primary_base_rmse - _rmse_from_sse(
        shuffled_sse.sum(), total_rows
    )

primary.to_csv(OUT_DIR / "primary_per_well.csv", index=False)
primary_folds = fold_report[fold_report["base"].eq(primary_base)]
primary_summary = base_summary[base_summary["base"].eq(primary_base)].iloc[0]

pf_shift_valid = (
    np.isfinite(primary["midpoint_delta_ft"])
    & np.isfinite(primary["pf_scale5_optimal_datum_shift_ft"])
)
pf_shift_spearman = (
    float(
        spearmanr(
            primary.loc[pf_shift_valid, "midpoint_delta_ft"],
            primary.loc[pf_shift_valid, "pf_scale5_optimal_datum_shift_ft"],
        ).statistic
    )
    if pf_shift_valid.sum() >= 3
    else np.nan
)

gates = {
    "fixed_gain_ge": 0.03,
    "min_fold_gain_ge": -0.02,
    "positive_folds_ge": 4,
    "qualified_wells_ge": 30,
    "positive_qualified_fraction_ge": 0.55,
    "spearman_ge": 0.15,
    "reverse_sign_gain_le": 0.0,
    "fixed_gain_gt_shuffle_p95": True,
}
checks = {
    "fixed_gain": bool(primary_gain >= gates["fixed_gain_ge"]),
    "min_fold_gain": bool(primary_folds["gain"].min() >= gates["min_fold_gain_ge"]),
    "positive_folds": bool((primary_folds["gain"] > 0.0).sum() >= gates["positive_folds_ge"]),
    "qualified_wells": bool(
        int(primary_summary["qualified_wells"]) >= gates["qualified_wells_ge"]
    ),
    "positive_qualified_fraction": bool(
        float(primary_summary["positive_qualified_fraction"])
        >= gates["positive_qualified_fraction_ge"]
    ),
    "spearman": bool(
        np.isfinite(primary_summary["spearman_midpoint_vs_optimal_shift"])
        and float(primary_summary["spearman_midpoint_vs_optimal_shift"])
        >= gates["spearman_ge"]
    ),
    "reverse_sign": bool(
        float(primary_summary["reverse_sign_gain"]) <= gates["reverse_sign_gain_le"]
    ),
    "shuffle": bool(primary_gain > float(np.quantile(shuffle_gains, 0.95))),
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": EXPERIMENT,
    "source_notebook": "rogii-mha160-sep3-r2-20260720.ipynb",
    "pool": str(POOL_PATH),
    "fold_source": str(FOLD_SOURCE),
    "primary_base": primary_base,
    "pf_config": {
        "seeds": N_SEEDS,
        "particles": N_PARTICLES,
        "likelihood_temperature": LIKELIHOOD_TEMPERATURE,
        "initial_spread_ft": PF_INIT_SPREAD_FT,
        "seed_base": 0,
        "rng": "numba seeded np.random; distribution-matched, not bitwise default_rng",
    },
    "frozen_recipe": {
        "alpha": FIXED_ALPHA,
        "minor_mass_min": FIXED_MINOR_MASS,
        "separation_low_ft": FIXED_SEPARATION_LOW_FT,
        "separation_high_ft": FIXED_SEPARATION_HIGH_FT,
        "move_cap_ft": FIXED_MOVE_CAP_FT,
    },
    "statistics": {
        "requested_wells": len(wells),
        "successful_wells": len(statistics),
        "failed_wells": len(failures),
        "qualified_wells": int(primary_summary["qualified_wells"]),
        "pf_midpoint_vs_pf_optimal_shift_spearman": pf_shift_spearman,
    },
    "fixed_primary": {
        "base_rmse": primary_base_rmse,
        "prediction_rmse": primary_prediction_rmse,
        "gain": primary_gain,
        "min_fold_gain": float(primary_folds["gain"].min()),
        "positive_folds": int((primary_folds["gain"] > 0.0).sum()),
        "positive_qualified_fraction": float(
            primary_summary["positive_qualified_fraction"]
        ),
        "spearman_midpoint_vs_optimal_shift": float(
            primary_summary["spearman_midpoint_vs_optimal_shift"]
        ),
        "sign_accuracy": float(primary_summary["sign_accuracy"]),
        "reverse_sign_gain": float(primary_summary["reverse_sign_gain"]),
    },
    "nested_alpha": {
        "rmse": nested_rmse,
        "gain": primary_base_rmse - nested_rmse,
        "min_fold_gain": float(nested_report["valid_gain"].min()),
        "positive_folds": int((nested_report["valid_gain"] > 0.0).sum()),
    },
    "bootstrap": {
        "iterations": BOOTSTRAP_ITERATIONS,
        "p01_gain": float(np.quantile(bootstrap_gains, 0.01)),
        "p05_gain": float(np.quantile(bootstrap_gains, 0.05)),
        "p50_gain": float(np.quantile(bootstrap_gains, 0.50)),
        "p95_gain": float(np.quantile(bootstrap_gains, 0.95)),
        "positive_probability": float(np.mean(bootstrap_gains > 0.0)),
    },
    "shuffle_control": {
        "iterations": SHUFFLE_ITERATIONS,
        "p50_gain": float(np.quantile(shuffle_gains, 0.50)),
        "p95_gain": float(np.quantile(shuffle_gains, 0.95)),
        "fixed_gain_gt_shuffle_p95": bool(
            primary_gain > float(np.quantile(shuffle_gains, 0.95))
        ),
    },
    "gates": gates,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> port the frozen midpoint statistic into the final Kaggle PF cell and run one guarded LB probe"
        if strict_pass
        else "FAIL -> do not add midpoint hedge to the 6.781 mainline"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
alpha_scan.to_csv(OUT_DIR / "alpha_scan_diagnostic.csv", index=False)
print("\nALPHA SCAN")
print(alpha_scan.to_string(index=False))
print("\nNESTED ALPHA")
print(nested_report.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
