"""Nested paired audit of the learned lik-PF GR-scale multiplier.

The 6.56 notebook differs from the earlier source at one important line:
``gr_sigma *= 1.3`` inside its learned trajectory lik-PF. This cell tests that
mechanism before rebuilding the expensive learned endpoint.

It reuses the complete seed trajectories from
``likpf_blockposterior_globaldp_v0`` as the multiplier=1.0 baseline. Fold 3 has
a deterministic 24-well discovery subset and an independent 131-well
confirmation subset. The selected non-unit multiplier is then applied once to
Fold 2. Fold 1 remains untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


EXPERIMENT = "likpf_grscale_multiplier_nested_gateway_v1"
SOURCE_EXPERIMENT = "likpf_blockposterior_globaldp_v0"

ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").exists() else Path.cwd()
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
if not DATA_ROOT.exists():
    DATA_ROOT = Path.cwd()
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() or ROOT.name == "autodl-tmp" else Path.cwd() / "output"
SOURCE_DIR = OOF_ROOT / SOURCE_EXPERIMENT
SOURCE_CACHE = SOURCE_DIR / "pf_cache"
OUT_DIR = OOF_ROOT / EXPERIMENT
RECORD_DIR = OUT_DIR / "records"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECORD_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260721  # Exact source-cache fold seed.
N_SEEDS = 128
N_PARTICLES = 500
SCALES = (3.0, 5.0, 8.0, 12.0)
MULTIPLIERS = (1.0, 1.15, 1.30, 1.50)
DISCOVERY_WELLS = 24
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
MAX_WORKERS = int(os.environ.get("ROGII_GS_WORKERS", "4"))
WELL_LIMIT = int(os.environ.get("ROGII_GS_WELL_LIMIT", "0"))


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _stable_folds(wells):
    scored = []
    for well in wells:
        digest = hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    fold = np.empty(len(ordered), dtype=np.int64)
    for number, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        fold[indices] = number
    return pd.DataFrame({"well": ordered, "fold": fold})


def _discovery_order(wells):
    return sorted(
        wells,
        key=lambda well: hashlib.sha256(
            f"gs-discovery:{well}".encode("ascii")
        ).digest(),
    )


def _softmax(values, temperature):
    values = np.asarray(values, np.float64)
    scaled = (values - np.max(values)) / float(temperature)
    weight = np.exp(np.clip(scaled, -700.0, 0.0))
    return weight / max(float(weight.sum()), 1e-300)


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
def _pf_allseeds(
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
    initial_spread,
):
    rows = len(md_values)
    predictions = np.empty((seeds, rows))
    likelihoods = np.empty(seeds)
    tvt_maximum = tvt_minimum + len(gr_grid) * tvt_step
    for seed in range(seeds):
        np.random.seed(seed_base + seed)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = initial_structure + initial_spread * np.random.randn()
            rate[particle] = initial_rate + 0.01 * np.random.randn()
        total_loglik = 0.0
        previous_md = md_values[0] - 1.0
        for row in range(rows):
            delta_md = md_values[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = 0.998 * rate[particle] + 0.002 * np.random.randn()
                position[particle] += rate[particle] * delta_md + 0.005 * np.random.randn()
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
            total_loglik += np.log(average_likelihood)

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
            if effective < 0.5 * particles:
                cumulative = np.empty(particles)
                running = 0.0
                for particle in range(particles):
                    running += weights[particle]
                    cumulative[particle] = running
                start = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles)
                new_rate = np.empty(particles)
                cumulative_index = 0
                for particle in range(particles):
                    threshold = start + particle / particles
                    while cumulative_index < particles - 1 and cumulative[cumulative_index] < threshold:
                        cumulative_index += 1
                    new_position[particle] = position[cumulative_index] + 0.1 * np.random.randn()
                    new_rate[particle] = rate[cumulative_index] + 0.001 * np.random.randn()
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles

            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (position[particle] - z_values[row])
            predictions[seed, row] = estimate
            previous_md = md_values[row]
        likelihoods[seed] = total_loglik
    return predictions, likelihoods


def _typewell_grid(tvt, gr, step=0.2):
    minimum = float(np.min(tvt))
    maximum = float(np.max(tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    return np.interp(grid_tvt, tvt, gr).astype(np.float64), minimum, float(step)


def _raw_payload(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    horizontal = horizontal.sort_values("MD").reset_index(drop=True)
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    typewell = typewell[["TVT", "GR"]].dropna().sort_values("TVT")

    known_mask = horizontal["TVT_input"].notna().to_numpy(bool)
    evaluation_mask = (~known_mask) & horizontal["TVT"].notna().to_numpy(bool)
    if known_mask.sum() < 30 or evaluation_mask.sum() < 10:
        raise RuntimeError(
            f"{well}: known/eval={known_mask.sum()}/{evaluation_mask.sum()}"
        )

    tw_tvt = typewell["TVT"].to_numpy(np.float64)
    tw_gr = typewell["GR"].to_numpy(np.float64)
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[evaluation_mask]
    last = known.iloc[-1]
    anchor_md = float(last["MD"])
    anchor_surface = float(last["TVT_input"] + last["Z"])

    # Match the source cache exactly; only the multiplier changes downstream.
    expected_known = np.interp(known["TVT_input"], tw_tvt, tw_gr)
    observed_known = (
        known["GR"].interpolate(limit_direction="both").to_numpy(np.float64)
    )
    base_sigma = float(
        np.clip(np.nanstd(observed_known - expected_known), 10.0, 60.0)
    )
    tail = known.tail(30)
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    delta_surface = np.diff(
        (tail["TVT_input"] + tail["Z"]).to_numpy(np.float64)
    )
    valid = delta_md > 0
    initial_rate = (
        float(np.median(delta_surface[valid] / delta_md[valid]))
        if valid.sum() >= 3
        else 0.0
    )
    gr_grid, tvt_minimum, tvt_step = _typewell_grid(tw_tvt, tw_gr)
    horizontal_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.mean(tw_gr)))
        .to_numpy(np.float64)
    )
    return {
        "row_idx": np.flatnonzero(evaluation_mask).astype(np.int32),
        "md": evaluation["MD"].to_numpy(np.float64),
        "z": evaluation["Z"].to_numpy(np.float64),
        "gr": horizontal_gr[evaluation_mask],
        "truth": evaluation["TVT"].to_numpy(np.float64),
        "gr_grid": gr_grid,
        "tvt_minimum": tvt_minimum,
        "tvt_step": tvt_step,
        "base_sigma": base_sigma,
        "anchor_md": anchor_md,
        "anchor_surface": anchor_surface,
        "initial_rate": initial_rate,
    }


def _source_cache_path(well):
    return SOURCE_CACHE / f"{well}.npz"


def _variant_cache_path(well, multiplier):
    tag = f"{float(multiplier):.2f}".replace(".", "p")
    return RECORD_DIR / f"m{tag}_{well}.npz"


def _aggregate(predictions, likelihoods, truth):
    output = {}
    effective = {}
    entropy_effective = {}
    top_weight = {}
    for scale in SCALES:
        weight = _softmax(likelihoods, scale)
        key = f"pf_scale_{int(scale)}"
        output[key] = np.sum(weight[:, None] * predictions, axis=0).astype(np.float32)
        effective[key] = float(1.0 / np.sum(weight * weight))
        entropy_effective[key] = float(
            np.exp(-np.sum(weight * np.log(np.maximum(weight, 1e-300))))
        )
        top_weight[key] = float(np.max(weight))
    output["pf_mean"] = predictions.mean(axis=0).astype(np.float32)
    output["pf_std"] = predictions.std(axis=0).astype(np.float32)
    seed_mse = np.mean((predictions - truth[None, :]) ** 2, axis=1)
    seed_index = int(np.argmin(seed_mse))
    output["seed_oracle"] = predictions[seed_index].astype(np.float32)
    return output, effective, entropy_effective, top_weight, seed_index


def _build_variant(well, multiplier):
    path = _variant_cache_path(well, multiplier)
    if path.exists():
        return {"well": well, "multiplier": multiplier, "status": "cached"}
    payload = _raw_payload(well)
    predictions, likelihoods = _pf_allseeds(
        payload["md"],
        payload["z"],
        payload["gr"],
        payload["gr_grid"],
        payload["tvt_minimum"],
        payload["tvt_step"],
        payload["base_sigma"] * float(multiplier),
        payload["anchor_surface"],
        payload["initial_rate"],
        N_PARTICLES,
        N_SEEDS,
        0,
        4.5,
    )
    aggregate, effective, entropy_effective, top_weight, seed_index = _aggregate(
        predictions,
        likelihoods,
        payload["truth"],
    )
    save = {
        "well": np.asarray([well]),
        "multiplier": np.asarray([multiplier], np.float32),
        "row_idx": payload["row_idx"],
        "md": payload["md"].astype(np.float32),
        "truth": payload["truth"].astype(np.float32),
        "anchor_md": np.asarray([payload["anchor_md"]], np.float64),
        "base_sigma": np.asarray([payload["base_sigma"]], np.float32),
        "used_sigma": np.asarray([payload["base_sigma"] * multiplier], np.float32),
        "likelihood_max": np.asarray([np.max(likelihoods)], np.float32),
        "likelihood_std": np.asarray([np.std(likelihoods)], np.float32),
        "likelihood_top_gap": np.asarray(
            [np.sort(likelihoods)[-1] - np.sort(likelihoods)[-2]], np.float32
        ),
        "seed_oracle_index": np.asarray([seed_index], np.int16),
    }
    save.update(aggregate)
    for key, value in effective.items():
        save[f"effective_{key}"] = np.asarray([value], np.float32)
    for key, value in entropy_effective.items():
        save[f"entropy_effective_{key}"] = np.asarray([value], np.float32)
    for key, value in top_weight.items():
        save[f"top_weight_{key}"] = np.asarray([value], np.float32)
    np.savez_compressed(path, **save)
    return {"well": well, "multiplier": multiplier, "status": "built"}


def _load_baseline(well):
    path = _source_cache_path(well)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        center = data["center"].astype(np.float64)
        predictions = center[None, :] + data["seed_delta"].astype(np.float64)
        likelihoods = data["total_loglik"].astype(np.float64)
        truth = data["truth"].astype(np.float64)
        aggregate, effective, entropy_effective, top_weight, seed_index = _aggregate(
            predictions,
            likelihoods,
            truth,
        )
        return {
            "well": well,
            "multiplier": 1.0,
            "row_idx": np.arange(len(truth), dtype=np.int32),
            "md": data["md"].astype(np.float64),
            "truth": truth,
            "anchor_md": float(data["anchor_md"][0]),
            "base_sigma": float(data["gr_sigma"][0]),
            "used_sigma": float(data["gr_sigma"][0]),
            "paths": aggregate,
            "effective": effective,
            "entropy_effective": entropy_effective,
            "top_weight": top_weight,
            "likelihood_max": float(np.max(likelihoods)),
            "likelihood_std": float(np.std(likelihoods)),
            "likelihood_top_gap": float(
                np.sort(likelihoods)[-1] - np.sort(likelihoods)[-2]
            ),
            "seed_oracle_index": seed_index,
        }


def _load_variant(well, multiplier):
    if abs(float(multiplier) - 1.0) < 1e-12:
        return _load_baseline(well)
    path = _variant_cache_path(well, multiplier)
    with np.load(path, allow_pickle=False) as data:
        paths = {
            key: data[key].astype(np.float64)
            for key in [
                "pf_scale_3",
                "pf_scale_5",
                "pf_scale_8",
                "pf_scale_12",
                "pf_mean",
                "pf_std",
                "seed_oracle",
            ]
        }
        effective = {
            key: float(data[f"effective_{key}"][0])
            for key in ("pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12")
        }
        entropy_effective = {
            key: float(data[f"entropy_effective_{key}"][0])
            for key in ("pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12")
        }
        top_weight = {
            key: float(data[f"top_weight_{key}"][0])
            for key in ("pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12")
        }
        return {
            "well": well,
            "multiplier": float(multiplier),
            "row_idx": data["row_idx"].astype(np.int32),
            "md": data["md"].astype(np.float64),
            "truth": data["truth"].astype(np.float64),
            "anchor_md": float(data["anchor_md"][0]),
            "base_sigma": float(data["base_sigma"][0]),
            "used_sigma": float(data["used_sigma"][0]),
            "paths": paths,
            "effective": effective,
            "entropy_effective": entropy_effective,
            "top_weight": top_weight,
            "likelihood_max": float(data["likelihood_max"][0]),
            "likelihood_std": float(data["likelihood_std"][0]),
            "likelihood_top_gap": float(data["likelihood_top_gap"][0]),
            "seed_oracle_index": int(data["seed_oracle_index"][0]),
        }


def _error_row(payload, variant, prediction, **extra):
    truth = payload["truth"]
    error = np.asarray(prediction, np.float64) - truth
    distance = payload["md"] - payload["anchor_md"]
    row = {
        "well": payload["well"],
        "multiplier": payload["multiplier"],
        "variant": variant,
        "rows": len(error),
        "sse": float(np.sum(error**2)),
        "sae": float(np.sum(np.abs(error))),
        "effective_scale8": float(payload["effective"]["pf_scale_8"]),
        "entropy_effective_scale8": float(
            payload["entropy_effective"]["pf_scale_8"]
        ),
        "top_weight_scale8": float(payload["top_weight"]["pf_scale_8"]),
        "likelihood_top_gap": float(payload["likelihood_top_gap"]),
    }
    for label, mask in {
        "near_0_500": (distance >= 0.0) & (distance <= 500.0),
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        row[f"{label}_rows"] = int(mask.sum())
        row[f"{label}_sse"] = float(np.sum(error[mask] ** 2))
    row.update(extra)
    return row


def _evaluate(wells, multipliers, label):
    rows = []
    for number, well in enumerate(wells, start=1):
        for multiplier in multipliers:
            payload = _load_variant(well, multiplier)
            for variant in (
                "pf_scale_3",
                "pf_scale_5",
                "pf_scale_8",
                "pf_scale_12",
                "pf_mean",
                "seed_oracle",
            ):
                rows.append(_error_row(payload, variant, payload["paths"][variant]))

            scale_candidates = np.stack(
                [payload["paths"][f"pf_scale_{int(scale)}"] for scale in SCALES]
            )
            candidate_mse = np.mean(
                (scale_candidates - payload["truth"][None, :]) ** 2,
                axis=1,
            )
            selected = int(np.argmin(candidate_mse))
            rows.append(
                _error_row(
                    payload,
                    "scale_bank_oracle",
                    scale_candidates[selected],
                    selected_scale=float(SCALES[selected]),
                )
            )
        if number % 25 == 0 or number == len(wells):
            print(f"GS EVALUATE {label} {number}/{len(wells)}", flush=True)
    return pd.DataFrame(rows)


def _summary(detail):
    rows = []
    for (multiplier, variant), group in detail.groupby(
        ["multiplier", "variant"], sort=False
    ):
        count = int(group["rows"].sum())
        sse = float(group["sse"].sum())
        well_rmse = np.sqrt(group["sse"] / group["rows"])
        row = {
            "multiplier": float(multiplier),
            "variant": variant,
            "rows": count,
            "wells": int(group["well"].nunique()),
            "exact_sse": sse,
            "rmse": math.sqrt(sse / count),
            "mae": float(group["sae"].sum() / count),
            "median_well_rmse": float(np.median(well_rmse)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "positive_well_fraction_vs_m1": np.nan,
            "mean_effective_scale8": float(group["effective_scale8"].mean()),
            "mean_entropy_effective_scale8": float(
                group["entropy_effective_scale8"].mean()
            ),
            "mean_top_weight_scale8": float(group["top_weight_scale8"].mean()),
            "mean_likelihood_top_gap": float(group["likelihood_top_gap"].mean()),
        }
        for label in (
            "near_0_500",
            "mid_500_1000",
            "mid_1000_2000",
            "far_gt2000",
        ):
            strata_rows = int(group[f"{label}_rows"].sum())
            row[f"{label}_rmse"] = (
                math.sqrt(float(group[f"{label}_sse"].sum()) / strata_rows)
                if strata_rows
                else np.nan
            )
        rows.append(row)
    output = pd.DataFrame(rows)

    baseline = detail.loc[detail["multiplier"].eq(1.0), ["well", "variant", "sse", "rows"]]
    baseline = baseline.rename(columns={"sse": "base_sse", "rows": "base_rows"})
    for index, row in output.iterrows():
        if row["multiplier"] == 1.0:
            output.loc[index, "positive_well_fraction_vs_m1"] = 0.0
            continue
        candidate = detail.loc[
            detail["multiplier"].eq(row["multiplier"])
            & detail["variant"].eq(row["variant"]),
            ["well", "sse", "rows"],
        ]
        paired = candidate.merge(
            baseline.loc[baseline["variant"].eq(row["variant"])],
            on="well",
            validate="one_to_one",
        )
        candidate_rmse = np.sqrt(paired["sse"] / paired["rows"])
        baseline_rmse = np.sqrt(paired["base_sse"] / paired["base_rows"])
        output.loc[index, "positive_well_fraction_vs_m1"] = float(
            np.mean(candidate_rmse < baseline_rmse)
        )
    return output


def _one(summary, multiplier, variant):
    rows = summary.loc[
        np.isclose(summary["multiplier"], float(multiplier))
        & summary["variant"].eq(variant)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for {multiplier}/{variant}, got {len(rows)}")
    return rows.iloc[0].to_dict()


def _run_jobs(wells, multipliers, label):
    tasks = [
        (well, multiplier)
        for well in wells
        for multiplier in multipliers
        if abs(float(multiplier) - 1.0) > 1e-12
        and not _variant_cache_path(well, multiplier).exists()
    ]
    if not tasks:
        print(f"GS BUILD {label}: all cached", flush=True)
        return
    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_build_variant, well, multiplier): (well, multiplier)
            for well, multiplier in tasks
        }
        for number, future in enumerate(as_completed(futures), start=1):
            try:
                report = future.result()
                if number % 10 == 0 or number == len(futures):
                    print(
                        f"GS BUILD {label} {number}/{len(futures)} "
                        f"last={report['well']} m={report['multiplier']}",
                        flush=True,
                    )
            except Exception as exc:
                well, multiplier = futures[future]
                failures.append(
                    {"well": well, "multiplier": multiplier, "reason": repr(exc)}
                )
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} {label} jobs failed: {failures[:5]}")


def _paired_bootstrap(detail, multiplier, variant, draws=2000):
    candidate = detail.loc[
        np.isclose(detail["multiplier"], float(multiplier))
        & detail["variant"].eq(variant),
        ["well", "rows", "sse"],
    ]
    baseline = detail.loc[
        detail["multiplier"].eq(1.0) & detail["variant"].eq(variant),
        ["well", "rows", "sse"],
    ].rename(columns={"rows": "base_rows", "sse": "base_sse"})
    paired = candidate.merge(baseline, on="well", validate="one_to_one")
    rng = np.random.default_rng(20260722)
    gains = []
    for _ in range(draws):
        index = rng.integers(0, len(paired), len(paired))
        sampled = paired.iloc[index]
        candidate_rmse = math.sqrt(float(sampled["sse"].sum() / sampled["rows"].sum()))
        baseline_rmse = math.sqrt(
            float(sampled["base_sse"].sum() / sampled["base_rows"].sum())
        )
        gains.append(baseline_rmse - candidate_rmse)
    gains = np.asarray(gains)
    return {
        "draws": draws,
        "p05_gain": float(np.quantile(gains, 0.05)),
        "p50_gain": float(np.quantile(gains, 0.50)),
        "p95_gain": float(np.quantile(gains, 0.95)),
        "positive_probability": float(np.mean(gains > 0.0)),
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

started = time.time()
if not SOURCE_CACHE.exists():
    raise FileNotFoundError(
        f"Missing {SOURCE_CACHE}. Run {SOURCE_EXPERIMENT}_cell.py first."
    )

all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = _stable_folds(all_wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
fold3 = [well for well in all_wells if fold_by_well[well] == SELECTION_FOLD]
fold2 = [well for well in all_wells if fold_by_well[well] == APPLICATION_FOLD]
ordered_fold3 = _discovery_order(fold3)
discovery_wells = ordered_fold3[:DISCOVERY_WELLS]
fold3_confirmation_wells = ordered_fold3[DISCOVERY_WELLS:]
if WELL_LIMIT:
    discovery_wells = discovery_wells[: min(WELL_LIMIT, len(discovery_wells))]
    fold3_confirmation_wells = fold3_confirmation_wells[:WELL_LIMIT]
    fold2 = fold2[:WELL_LIMIT]

for well in discovery_wells + fold3_confirmation_wells + fold2:
    if not _source_cache_path(well).exists():
        raise FileNotFoundError(_source_cache_path(well))

print(
    "GS-MULTIPLIER CONFIG",
    {
        "source": str(SOURCE_DIR),
        "discovery_wells": len(discovery_wells),
        "fold3_confirmation_wells": len(fold3_confirmation_wells),
        "fold2_wells": len(fold2),
        "multipliers": MULTIPLIERS,
        "seeds": N_SEEDS,
        "particles": N_PARTICLES,
        "workers": MAX_WORKERS,
        "well_limit": WELL_LIMIT,
    },
    flush=True,
)

# JIT warm-up before threaded calls.
_pf_allseeds(
    np.arange(1.0, 5.0),
    np.zeros(4),
    np.full(4, 50.0),
    np.linspace(40.0, 60.0, 32),
    0.0,
    1.0,
    20.0,
    50.0,
    0.0,
    8,
    2,
    0,
    4.5,
)

_run_jobs(discovery_wells, MULTIPLIERS, "fold3_discovery")
discovery_detail = _evaluate(discovery_wells, MULTIPLIERS, "fold3_discovery")
discovery_summary = _summary(discovery_detail)
discovery_detail.to_csv(OUT_DIR / "fold3_discovery_detail.csv", index=False)
discovery_summary.sort_values(["variant", "rmse"]).to_csv(
    OUT_DIR / "fold3_discovery_summary.csv", index=False
)

bank_scan = discovery_summary.loc[
    discovery_summary["variant"].eq("scale_bank_oracle")
].sort_values(["rmse", "multiplier"])
selected_multiplier = float(bank_scan.iloc[0]["multiplier"])
discovery_base = _one(discovery_summary, 1.0, "scale_bank_oracle")
discovery_selected = _one(
    discovery_summary, selected_multiplier, "scale_bank_oracle"
)
discovery_gain = float(discovery_base["rmse"] - discovery_selected["rmse"])

confirmation_multipliers = (
    (1.0, selected_multiplier)
    if selected_multiplier != 1.0
    else (1.0,)
)
if selected_multiplier != 1.0:
    _run_jobs(
        fold3_confirmation_wells,
        confirmation_multipliers,
        "fold3_confirmation",
    )
    _run_jobs(fold2, confirmation_multipliers, "fold2_application")

fold3_detail = _evaluate(
    fold3_confirmation_wells,
    confirmation_multipliers,
    "fold3_confirmation",
)
fold2_detail = _evaluate(fold2, confirmation_multipliers, "fold2_application")
fold3_summary = _summary(fold3_detail)
fold2_summary = _summary(fold2_detail)
fold3_detail.to_csv(OUT_DIR / "fold3_confirmation_detail.csv", index=False)
fold2_detail.to_csv(OUT_DIR / "fold2_application_detail.csv", index=False)
fold3_summary.sort_values(["variant", "rmse"]).to_csv(
    OUT_DIR / "fold3_confirmation_summary.csv", index=False
)
fold2_summary.sort_values(["variant", "rmse"]).to_csv(
    OUT_DIR / "fold2_application_summary.csv", index=False
)
folds.to_csv(OUT_DIR / "folds.csv", index=False)

fold3_base = _one(fold3_summary, 1.0, "scale_bank_oracle")
fold3_selected = _one(
    fold3_summary, selected_multiplier, "scale_bank_oracle"
)
fold2_base = _one(fold2_summary, 1.0, "scale_bank_oracle")
fold2_selected = _one(fold2_summary, selected_multiplier, "scale_bank_oracle")
fold3_gain = float(fold3_base["rmse"] - fold3_selected["rmse"])
fold2_gain = float(fold2_base["rmse"] - fold2_selected["rmse"])

fixed_scale_gains = {}
for variant in ("pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"):
    base = _one(fold2_summary, 1.0, variant)
    selected = _one(fold2_summary, selected_multiplier, variant)
    fixed_scale_gains[variant] = float(base["rmse"] - selected["rmse"])
best_fixed_scale_gain = float(max(fixed_scale_gains.values()))
fold2_bootstrap = _paired_bootstrap(
    fold2_detail,
    selected_multiplier,
    "scale_bank_oracle",
)

base_effective = float(fold2_base["mean_entropy_effective_scale8"])
selected_effective = float(fold2_selected["mean_entropy_effective_scale8"])
effective_ratio = selected_effective / max(base_effective, 1e-9)

checks = {
    "selected_multiplier_is_nonunit": selected_multiplier != 1.0,
    "discovery_bank_gain_ge_0p05": discovery_gain >= 0.05,
    "fold3_confirmation_bank_gain_ge_0p05": fold3_gain >= 0.05,
    "fold2_bank_gain_ge_0p05": fold2_gain >= 0.05,
    "fold2_best_fixed_scale_gain_ge_0p02": best_fixed_scale_gain >= 0.02,
    "fold2_positive_well_fraction_ge_0p52": (
        float(fold2_selected["positive_well_fraction_vs_m1"]) >= 0.52
    ),
    "entropy_effective_seed_ratio_ge_1p10": effective_ratio >= 1.10,
    "bootstrap_positive_probability_ge_0p90": (
        fold2_bootstrap["positive_probability"] >= 0.90
    ),
}
strict_pass = bool(all(checks.values())) and WELL_LIMIT == 0

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_exact_likpf_grscale_multiplier_mechanism_gate",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "truth_usage": "multiplier selection and oracle pricing only",
    "uses_neighbor_spatial_cp12_gold": False,
    "source": str(SOURCE_DIR),
    "protocol": {
        "discovery": "24 deterministic wells from Fold3",
        "confirmation": "remaining Fold3 wells",
        "application": "Fold2",
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "pf": {
        "seeds": N_SEEDS,
        "particles": N_PARTICLES,
        "scales": list(SCALES),
        "multipliers_scanned": list(MULTIPLIERS),
        "selected_multiplier": selected_multiplier,
    },
    "scope_warning": (
        "This prices the exact lik-PF feature bank, not the final pretrained "
        "learned booster endpoint. A pass authorizes that rebuild; a fail does "
        "not rewrite historical leaderboard evidence."
    ),
    "fold3_discovery": {
        "base": discovery_base,
        "selected": discovery_selected,
        "gain": discovery_gain,
    },
    "fold3_confirmation": {
        "base": fold3_base,
        "selected": fold3_selected,
        "gain": fold3_gain,
    },
    "fold2_application": {
        "base": fold2_base,
        "selected": fold2_selected,
        "gain": fold2_gain,
        "fixed_scale_gains": fixed_scale_gains,
        "best_fixed_scale_gain": best_fixed_scale_gain,
        "entropy_effective_seed_ratio": effective_ratio,
        "bootstrap": fold2_bootstrap,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> regenerate fold-clean learned features at the selected multiplier and retrain the endpoint"
        if strict_pass
        else "FAIL -> GR-scale softening is not a transferable raw lik-PF improvement; do not rebuild the learned endpoint yet"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

_write_json(OUT_DIR / "verdict.json", verdict)
print("\nGS DISCOVERY SUMMARY")
print(discovery_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)
print("\nGS FOLD3 CONFIRMATION")
print(fold3_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)
print("\nGS FOLD2 APPLICATION")
print(fold2_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)
print("\nGS-MULTIPLIER FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
