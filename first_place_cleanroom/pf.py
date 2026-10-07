from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import HEIGHT, PF, STATE_AXIS, WIDTH
from .data import Well, _bin_last, _bin_stats, _take, _window_indices, calibrated_typewell_gr


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _transition_logpdf(next_state: float, previous: np.ndarray, center: float, sigma: float,
                       jump_prob: float, jump_sigma: float) -> np.ndarray:
    delta = next_state - previous - center
    local = np.exp(-.5 * (delta / sigma) ** 2) / sigma
    jump = np.exp(-.5 * (delta / jump_sigma) ** 2) / jump_sigma
    return np.log((1 - jump_prob) * local + jump_prob * jump + 1e-20)


def build_pf_features(well: Well, cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    cache = None if cache_dir is None else cache_dir / f"{well.well}.npz"
    if cache is not None and cache.is_file():
        with np.load(cache) as data:
            return {name: data[name] for name in data.files}
    indices = _window_indices(well)
    gr = _take(well.horizontal.GR.to_numpy(float), indices)
    z = _take(well.horizontal.Z.to_numpy(float), indices)
    obs = _bin_stats(gr)["gr"]
    z_bin = _bin_last(z)
    z_delta = np.diff(z_bin, prepend=z_bin[0])
    axis = np.asarray(STATE_AXIS, np.float32)
    tw_gr, _ = calibrated_typewell_gr(well, axis + well.anchor_tvt)
    density = np.zeros((HEIGHT, WIDTH), np.float64)
    smooth_density = np.zeros_like(density)
    smooth_path_sum = np.zeros(WIDTH, np.float64)
    smooth_path_count = np.zeros(WIDTH, np.float64)
    for seed_index in range(PF.seeds):
        rng = np.random.default_rng(PF.base_seed + seed_index)
        profile = seed_index % len(PF.transition_sigmas)
        sigma = PF.transition_sigmas[profile]
        jump_probability = PF.jump_probabilities[profile]
        jump_sigma = PF.jump_sigmas[profile]
        particles = np.zeros(PF.particles, np.float32)
        weights = np.full(PF.particles, 1 / PF.particles, np.float64)
        history_particles = np.empty((WIDTH, PF.particles), np.float32)
        history_weights = np.empty((WIDTH, PF.particles), np.float32)
        for column in range(WIDTH):
            center = -float(np.nan_to_num(z_delta[column]))
            jumps = rng.random(PF.particles) < jump_probability
            noise = rng.normal(0, sigma, PF.particles)
            noise[jumps] = rng.normal(0, jump_sigma, jumps.sum())
            particles = np.clip(particles + center + noise, axis[0], axis[-1])
            # PF emissions are updated in 64 raw-sample bins. One network
            # column contains 32 samples, hence one update per two columns.
            update = column % 2 == 1 or column == WIDTH - 1
            if update:
                left = max(0, column - 1)
                observation = float(np.nanmean(obs[left:column + 1]))
                predicted_gr = np.interp(particles, axis, tw_gr)
                residual = predicted_gr - observation
                likelihood = np.exp(-.5 * (residual / PF.likelihood_scale) ** 2)
                weights = likelihood + 1e-12
                weights /= weights.sum()
            state_index = np.clip(np.rint((particles - axis[0]) / .5).astype(int), 0, HEIGHT - 1)
            np.add.at(density[:, column], state_index, weights)
            history_particles[column] = particles
            history_weights[column] = weights
            if update:
                selected = systematic_resample(weights, rng)
                particles = particles[selected]
                weights.fill(1 / PF.particles)
        # Clean-room FFBSi: sample terminal states then apply the backward transition kernel.
        for _ in range(4):
            path = np.empty(WIDTH, np.float32)
            path[-1] = rng.choice(history_particles[-1], p=history_weights[-1] / history_weights[-1].sum())
            for column in range(WIDTH - 2, -1, -1):
                center = -float(np.nan_to_num(z_delta[column + 1]))
                logw = np.log(history_weights[column] + 1e-20) + _transition_logpdf(
                    float(path[column + 1]), history_particles[column], center, sigma,
                    jump_probability, jump_sigma)
                probability = np.exp(logw - logw.max())
                probability /= probability.sum()
                path[column] = rng.choice(history_particles[column], p=probability)
            state_index = np.clip(np.rint((path - axis[0]) / .5).astype(int), 0, HEIGHT - 1)
            smooth_density[state_index, np.arange(WIDTH)] += 1
            smooth_path_sum += path
            smooth_path_count += 1
    density /= np.maximum(density.sum(0, keepdims=True), 1e-12)
    smooth_density /= np.maximum(smooth_density.sum(0, keepdims=True), 1e-12)
    smooth_path = smooth_path_sum / np.maximum(smooth_path_count, 1)
    state_grid = axis[:, None]
    result = {
        "pf_particle_density_prob": density.astype(np.float32),
        "pf_prob": density.astype(np.float32),
        "pf_prob_ffbsi": smooth_density.astype(np.float32),
        "pf_tvt_ffbsi": (np.abs(state_grid - smooth_path[None]) / 100.0).astype(np.float32),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **result)
    return result
