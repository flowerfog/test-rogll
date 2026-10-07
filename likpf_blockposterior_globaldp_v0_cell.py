"""128-seed lik-PF block posterior followed by global structural DP.

This is a zero-learning, single-well gateway.  Each PF seed exposes both its
row trajectory and row likelihood increments.  Those trajectories are
clustered into a posterior over block offsets; a first-order global DP then
selects a structurally plausible sequence of modes.  Fold 4/5 fit only the
transition prior, Fold 3 selects decoder constants, Fold 2 confirms them, and
Fold 1 remains untouched.
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


EXPERIMENT = "likpf_blockposterior_globaldp_v0"
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").exists() else Path.cwd()
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
if not DATA_ROOT.exists():
    DATA_ROOT = Path.cwd()
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() or ROOT.name == "autodl-tmp" else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "pf_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260721
N_SEEDS = 128
N_PARTICLES = 500
BLOCK_FT = 128.0
OFFSET_GRID = np.arange(-64.0, 64.01, 1.0, dtype=np.float64)
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
MAX_WORKERS = int(os.environ.get("ROGII_PF_WORKERS", "4"))
WELL_LIMIT = int(os.environ.get("ROGII_PFDP_WELL_LIMIT", "0"))

TEMPERATURES = (1.0, 3.0, 5.0, 12.0)
KDE_SIGMAS = (1.0, 2.0, 4.0)
TRANSITION_LAMBDAS = (0.0, 0.01, 0.05, 0.20, 1.0)
ANCHOR_LAMBDA = 0.20


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
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


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
def _pf_allseeds_rowlik(
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
    seed_predictions = np.empty((seeds, rows))
    seed_row_loglik = np.empty((seeds, rows))
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
        total_loglik = 0.0
        previous_md = md_values[0] - 1.0
        for row in range(rows):
            delta_md = max(md_values[row] - previous_md, 1.0)
            for particle in range(particles):
                rate[particle] = 0.998 * rate[particle] + 0.002 * np.random.randn()
                position[particle] += rate[particle] * delta_md + 0.005 * np.random.randn()
                particle_tvt = position[particle] - z_values[row]
                particle_tvt = min(max(particle_tvt, tvt_minimum - 100.0), tvt_maximum + 100.0)
                position[particle] = particle_tvt + z_values[row]

            average_likelihood = 0.0
            for particle in range(particles):
                expected_gr = _interp1(gr_grid, position[particle] - z_values[row], tvt_minimum, tvt_step)
                residual = (gr_values[row] - expected_gr) / gr_sigma
                squared = min(residual * residual, 600.0)
                likelihood = max(np.exp(-0.5 * squared), 1e-300)
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            average_likelihood = max(average_likelihood, 1e-300)
            row_loglik = np.log(average_likelihood)
            seed_row_loglik[seed, row] = row_loglik
            total_loglik += row_loglik

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
            if 1.0 / inverse_effective < 0.5 * particles:
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
            seed_predictions[seed, row] = estimate
            previous_md = md_values[row]
        seed_likelihoods[seed] = total_loglik
    return seed_predictions, seed_row_loglik, seed_likelihoods


def _typewell_grid(tvt, gr, step=0.2):
    minimum = float(np.min(tvt))
    maximum = float(np.max(tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    return np.interp(grid_tvt, tvt, gr).astype(np.float64), minimum, float(step)


def _load_raw(well, include_typewell=True):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    horizontal = horizontal.sort_values("MD").reset_index(drop=True)
    known = horizontal["TVT_input"].notna().to_numpy(bool)
    evaluation = (~known) & horizontal["TVT"].notna().to_numpy(bool)
    if known.sum() < 30 or evaluation.sum() < 10:
        raise RuntimeError(f"{well}: known/eval={known.sum()}/{evaluation.sum()}")
    result = {"horizontal": horizontal, "known": known, "evaluation": evaluation}
    if include_typewell:
        typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        result["typewell"] = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
    return result


def _cache_path(well):
    return CACHE_DIR / f"{well}.npz"


def _run_well_pf(well):
    path = _cache_path(well)
    if path.exists():
        return well, "cached", 0.0
    started = time.time()
    raw = _load_raw(well)
    horizontal = raw["horizontal"]
    known_mask = raw["known"]
    eval_mask = raw["evaluation"]
    typewell = raw["typewell"]
    tw_tvt = typewell["TVT"].to_numpy(np.float64)
    tw_gr = typewell["GR"].to_numpy(np.float64)
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last = known.iloc[-1]
    anchor_md = float(last["MD"])
    anchor_surface = float(last["TVT_input"] + last["Z"])
    expected_known = np.interp(known["TVT_input"], tw_tvt, tw_gr)
    observed_known = known["GR"].interpolate(limit_direction="both").to_numpy(np.float64)
    gr_sigma = float(np.clip(np.nanstd(observed_known - expected_known), 10.0, 60.0))
    tail = known.tail(30)
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    delta_surface = np.diff((tail["TVT_input"] + tail["Z"]).to_numpy(np.float64))
    valid = delta_md > 0
    initial_rate = float(np.median(delta_surface[valid] / delta_md[valid])) if valid.sum() >= 3 else 0.0
    gr_grid, tvt_minimum, tvt_step = _typewell_grid(tw_tvt, tw_gr)
    horizontal_gr = horizontal["GR"].interpolate(limit_direction="both").fillna(float(np.mean(tw_gr))).to_numpy(np.float64)
    seed_prediction, row_loglik, total_loglik = _pf_allseeds_rowlik(
        evaluation["MD"].to_numpy(np.float64),
        evaluation["Z"].to_numpy(np.float64),
        horizontal_gr[eval_mask],
        gr_grid,
        tvt_minimum,
        tvt_step,
        gr_sigma,
        anchor_surface,
        initial_rate,
        N_PARTICLES,
        N_SEEDS,
        0,
        4.5,
    )
    global_weight = _softmax(total_loglik, 5.0)
    center = np.sum(global_weight[:, None] * seed_prediction, axis=0)
    delta = seed_prediction - center[None, :]
    np.savez_compressed(
        path,
        md=evaluation["MD"].to_numpy(np.float32),
        z=evaluation["Z"].to_numpy(np.float32),
        truth=evaluation["TVT"].to_numpy(np.float32),
        center=center.astype(np.float32),
        seed_delta=delta.astype(np.float16),
        row_loglik=row_loglik.astype(np.float16),
        total_loglik=total_loglik.astype(np.float32),
        anchor_md=np.asarray([anchor_md], np.float64),
        anchor_surface=np.asarray([anchor_surface], np.float64),
        initial_rate=np.asarray([initial_rate], np.float64),
        gr_sigma=np.asarray([gr_sigma], np.float32),
    )
    return well, "built", (time.time() - started) / 60.0


def _load_cache(well):
    with np.load(_cache_path(well), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _fit_transition_prior(wells):
    rates = []
    for well in wells:
        raw = _load_raw(well, include_typewell=False)
        frame = raw["horizontal"].loc[raw["evaluation"]]
        md = frame["MD"].to_numpy(np.float64)
        surface = (frame["TVT"] + frame["Z"]).to_numpy(np.float64)
        block = np.floor((md - md[0]) / BLOCK_FT).astype(int)
        block_md = np.asarray([np.mean(md[block == key]) for key in np.unique(block)])
        block_surface = np.asarray([np.mean(surface[block == key]) for key in np.unique(block)])
        valid = np.diff(block_md) > 0
        rates.extend((np.diff(block_surface)[valid] / np.diff(block_md)[valid]).tolist())
    rates = np.asarray(rates, np.float64)
    centre = float(np.median(rates))
    scale = float(1.4826 * np.median(np.abs(rates - centre)))
    return {"rate_centre": centre, "rate_scale": max(scale, 0.005), "count": len(rates)}


def _block_posterior(cache, temperature, kde_sigma):
    md = cache["md"].astype(np.float64)
    center = cache["center"].astype(np.float64)
    delta = cache["seed_delta"].astype(np.float64)
    row_loglik = cache["row_loglik"].astype(np.float64)
    block_id = np.floor((md - float(cache["anchor_md"][0])) / BLOCK_FT).astype(int)
    unique = np.unique(block_id)
    block_md, block_center, block_z, posterior = [], [], [], []
    for key in unique:
        rows = np.flatnonzero(block_id == key)
        if len(rows) < 4:
            continue
        offsets = np.median(delta[:, rows], axis=1)
        local_loglik = np.sum(row_loglik[:, rows], axis=1)
        weight = _softmax(local_loglik, temperature)
        kernel = np.exp(-0.5 * ((OFFSET_GRID[None, :] - offsets[:, None]) / kde_sigma) ** 2)
        probability = np.sum(weight[:, None] * kernel, axis=0)
        probability = np.maximum(probability, 1e-12)
        probability /= probability.sum()
        block_md.append(float(np.mean(md[rows])))
        block_center.append(float(np.mean(center[rows])))
        block_z.append(float(np.mean(cache["z"][rows])))
        posterior.append(probability)
    return {
        "md": np.asarray(block_md),
        "center": np.asarray(block_center),
        "z": np.asarray(block_z),
        "posterior": np.asarray(posterior),
    }


def _decode(blocks, prior, transition_lambda, posterior_override=None):
    posterior = blocks["posterior"] if posterior_override is None else posterior_override
    energy = -np.log(np.maximum(posterior, 1e-12))
    block_surface = blocks["center"][:, None] + OFFSET_GRID[None, :] + blocks["z"][:, None]
    count, states = energy.shape
    cost = np.full((count, states), np.inf, np.float64)
    back = np.full((count, states), -1, np.int16)
    anchor_md = blocks["anchor_md"]
    anchor_surface = blocks["anchor_surface"]
    first_delta = max(blocks["md"][0] - anchor_md, 1.0)
    first_rate = (block_surface[0] - anchor_surface) / first_delta
    cost[0] = energy[0] + ANCHOR_LAMBDA * ((first_rate - prior["rate_centre"]) / prior["rate_scale"]) ** 2
    for block in range(1, count):
        delta_md = max(blocks["md"][block] - blocks["md"][block - 1], 1.0)
        rate = (block_surface[block][None, :] - block_surface[block - 1][:, None]) / delta_md
        transition = transition_lambda * ((rate - prior["rate_centre"]) / prior["rate_scale"]) ** 2
        candidate = cost[block - 1][:, None] + transition
        parent = np.argmin(candidate, axis=0)
        cost[block] = energy[block] + candidate[parent, np.arange(states)]
        back[block] = parent.astype(np.int16)
    path = np.empty(count, np.int16)
    path[-1] = int(np.argmin(cost[-1]))
    for block in range(count - 1, 0, -1):
        path[block - 1] = back[block, path[block]]
    return OFFSET_GRID[path]


def _prediction(cache, block_md, offsets):
    md = cache["md"].astype(np.float64)
    correction = np.interp(md, block_md, offsets, left=offsets[0], right=offsets[-1])
    return cache["center"].astype(np.float64) + correction


def _evaluate(wells, prior, config, control="full"):
    rows = []
    for well in wells:
        cache = _load_cache(well)
        blocks = _block_posterior(cache, config["temperature"], config["kde_sigma"])
        blocks["anchor_md"] = float(cache["anchor_md"][0])
        blocks["anchor_surface"] = float(cache["anchor_surface"][0])
        posterior_override = None
        if control == "shuffled_blocks":
            seed = int.from_bytes(hashlib.sha256(f"shuffle:{well}".encode()).digest()[:8], "little")
            posterior_override = blocks["posterior"][np.random.default_rng(seed).permutation(len(blocks["posterior"]))]
        elif control == "transition_only":
            posterior_override = np.full_like(blocks["posterior"], 1.0 / len(OFFSET_GRID))
        elif control != "full":
            raise ValueError(control)
        offsets = _decode(blocks, prior, config["transition_lambda"], posterior_override)
        independent = OFFSET_GRID[np.argmax(blocks["posterior"], axis=1)]
        truth = cache["truth"].astype(np.float64)
        md = cache["md"].astype(np.float64)
        distance = md - float(cache["anchor_md"][0])
        far = distance > 2000.0
        variants = {
            "pf_weighted": cache["center"].astype(np.float64),
            "block_argmax": _prediction(cache, blocks["md"], independent),
            "global_dp": _prediction(cache, blocks["md"], offsets),
        }
        # Truth is used only to price the representation ceiling.
        truth_block_offset = []
        for block_md in blocks["md"]:
            near = np.abs(md - block_md) <= BLOCK_FT / 2
            truth_block_offset.append(float(np.mean(truth[near] - cache["center"][near])))
        oracle_offset = OFFSET_GRID[np.argmin(np.abs(OFFSET_GRID[None, :] - np.asarray(truth_block_offset)[:, None]), axis=1)]
        variants["block_oracle"] = _prediction(cache, blocks["md"], oracle_offset)
        for variant, prediction in variants.items():
            error = prediction - truth
            rows.append(
                {
                    "well": well,
                    "variant": variant,
                    "rows": len(error),
                    "sse": float(np.sum(error**2)),
                    "sae": float(np.sum(np.abs(error))),
                    "far_rows": int(far.sum()),
                    "far_sse": float(np.sum(error[far] ** 2)),
                    "blocks": len(blocks["md"]),
                }
            )
    return pd.DataFrame(rows)


def _summaries(frame):
    output = []
    for variant, group in frame.groupby("variant"):
        count = int(group["rows"].sum())
        sse = float(group["sse"].sum())
        far_rows = int(group["far_rows"].sum())
        well_rmse = np.sqrt(group["sse"] / group["rows"])
        output.append(
            {
                "variant": variant,
                "rows": count,
                "wells": int(group["well"].nunique()),
                "exact_sse": sse,
                "rmse": math.sqrt(sse / count),
                "mae": float(group["sae"].sum() / count),
                "median_well_rmse": float(np.median(well_rmse)),
                "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
                "far_gt2000_rmse": math.sqrt(float(group["far_sse"].sum()) / far_rows),
                "mean_blocks": float(group["blocks"].mean()),
            }
        )
    return pd.DataFrame(output)


# --- EXECUTION ---
started = time.time()
all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = _stable_folds(all_wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
prior_wells = [well for well in all_wells if fold_by_well[well] in PRIOR_FOLDS]
selection_wells = [well for well in all_wells if fold_by_well[well] == SELECTION_FOLD]
application_wells = [well for well in all_wells if fold_by_well[well] == APPLICATION_FOLD]
if WELL_LIMIT:
    selection_wells = selection_wells[:WELL_LIMIT]
    application_wells = application_wells[:WELL_LIMIT]
prior = _fit_transition_prior(prior_wells)
print("PF-DP CONFIG", {"prior": prior, "selection": len(selection_wells), "application": len(application_wells), "workers": MAX_WORKERS}, flush=True)

# Warm Numba before threaded calls.
_pf_allseeds_rowlik(
    np.arange(1.0, 5.0), np.zeros(4), np.full(4, 50.0), np.linspace(40, 60, 32),
    0.0, 1.0, 20.0, 50.0, 0.0, 8, 2, 0, 4.5,
)
needed = selection_wells + application_wells
failures = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(_run_well_pf, well): well for well in needed}
    completed = 0
    for future in as_completed(futures):
        completed += 1
        try:
            well, status, minutes = future.result()
            if completed % 10 == 0 or completed == len(futures):
                print(f"PF cache {completed}/{len(futures)} last={well} {status} {minutes:.2f}m", flush=True)
        except Exception as exc:
            failures.append({"well": futures[future], "reason": repr(exc)})
if failures:
    pd.DataFrame(failures).to_csv(OUT_DIR / "pf_failures.csv", index=False)
    raise RuntimeError(f"{len(failures)} PF failures: {failures[:5]}")

grid_rows = []
selection_details = {}
for temperature in TEMPERATURES:
    for sigma in KDE_SIGMAS:
        for transition_lambda in TRANSITION_LAMBDAS:
            config = {"temperature": temperature, "kde_sigma": sigma, "transition_lambda": transition_lambda}
            detail = _evaluate(selection_wells, prior, config)
            summary = _summaries(detail)
            metrics = summary.set_index("variant").loc["global_dp"].to_dict()
            grid_rows.append({**config, **metrics})
            selection_details[(temperature, sigma, transition_lambda)] = (detail, summary)
            print("PF-DP GRID", temperature, sigma, transition_lambda, metrics["rmse"], flush=True)
grid = pd.DataFrame(grid_rows).sort_values("rmse")
selected_row = grid.iloc[0]
selected_config = {
    "temperature": float(selected_row["temperature"]),
    "kde_sigma": float(selected_row["kde_sigma"]),
    "transition_lambda": float(selected_row["transition_lambda"]),
}
grid.to_csv(OUT_DIR / "fold3_decoder_grid.csv", index=False)
selection_detail, selection_summary = selection_details[
    (selected_config["temperature"], selected_config["kde_sigma"], selected_config["transition_lambda"])
]
selection_detail.to_csv(OUT_DIR / "fold3_bywell.csv", index=False)
selection_summary.to_csv(OUT_DIR / "fold3_summary.csv", index=False)

application_detail = _evaluate(application_wells, prior, selected_config)
application_summary = _summaries(application_detail)
shuffle_summary = _summaries(_evaluate(application_wells, prior, selected_config, "shuffled_blocks"))
transition_summary = _summaries(_evaluate(application_wells, prior, selected_config, "transition_only"))
application_detail.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
application_summary.to_csv(OUT_DIR / "fold2_summary.csv", index=False)

full = application_summary.set_index("variant")
baseline = full.loc["pf_weighted"].to_dict()
independent = full.loc["block_argmax"].to_dict()
dp = full.loc["global_dp"].to_dict()
oracle = full.loc["block_oracle"].to_dict()
shuffle_dp = shuffle_summary.set_index("variant").loc["global_dp"].to_dict()
transition_dp = transition_summary.set_index("variant").loc["global_dp"].to_dict()
gain = baseline["rmse"] - dp["rmse"]
far_gain = baseline["far_gt2000_rmse"] - dp["far_gt2000_rmse"]
shuffle_retained_gain = baseline["rmse"] - shuffle_dp["rmse"]
gates = {
    "block_oracle_rmse_le_5p5": oracle["rmse"] <= 5.5,
    "dp_gain_vs_pf_ge_0p5": gain >= 0.5,
    "far_gain_vs_pf_ge_1": far_gain >= 1.0,
    "shuffle_retains_le_30pct_gain": shuffle_retained_gain <= 0.30 * max(gain, 1e-9),
    "dp_rmse_le_10": dp["rmse"] <= 10.0,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_128seed_blockposterior_global_structural_dp",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "pf": {"seeds": N_SEEDS, "particles": N_PARTICLES},
    "block_ft": BLOCK_FT,
    "offset_states": len(OFFSET_GRID),
    "selected_config": selected_config,
    "transition_prior": prior,
    "fold2": {
        "pf_weighted": baseline,
        "block_argmax": independent,
        "global_dp": dp,
        "block_oracle": oracle,
        "shuffled_blocks_dp": shuffle_dp,
        "transition_only_dp": transition_dp,
        "gain_vs_pf": gain,
        "far_gain_vs_pf": far_gain,
    },
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": (
        "PASS -> integrate the block posterior DP as the independent raw-data endpoint"
        if all(gates.values())
        else "FAIL -> blockwise PF posterior does not resolve the alias; close this decoder"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nPF-DP FOLD2 SUMMARY")
print(application_summary.to_string(index=False), flush=True)
print("\nPF-DP FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
