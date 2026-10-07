from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .data import PanelSample
from .inferred_defaults import PFDefaults, PFProfile


@dataclass(frozen=True)
class PFResult:
    density: np.ndarray
    filtered_offset: np.ndarray
    ffbsi_offset: np.ndarray
    entropy: np.ndarray
    profile_paths: np.ndarray


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = len(weights)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions).astype(np.int32)


def _student_log_likelihood(residual: np.ndarray, scale: float, degrees: float) -> np.ndarray:
    return -0.5 * (degrees + 1.0) * np.log1p((residual / scale) ** 2 / degrees)


class BootstrapParticleFilter:
    def __init__(self, defaults: PFDefaults, profile: PFProfile, seed: int):
        self.defaults = defaults
        self.profile = profile
        self.rng = np.random.default_rng(seed)

    def run(self, sample: PanelSample) -> PFResult:
        h, states = sample.gr_gap.shape
        offsets = sample.state_offsets.astype(np.float64)
        tw = sample.values[sample.channels.index("tw_gr")][0].astype(np.float64)
        hw = sample.values[sample.channels.index("hw_gr_mean")][:, 0].astype(np.float64)
        z = sample.values[sample.channels.index("z_diff")][:, 0].astype(np.float64) * 100.0
        base = -z
        particles = self.rng.normal(0.0, 2.0, self.defaults.particles)
        weights = np.full(self.defaults.particles, 1.0 / self.defaults.particles)
        clouds = np.empty((h, self.defaults.particles), dtype=np.float32)
        cloud_weights = np.empty_like(clouds)
        filtered = np.empty(h, dtype=np.float32)
        density = np.zeros((h, states), dtype=np.float32)
        entropy = np.empty(h, dtype=np.float32)
        update_bins = max(1, int(round(self.defaults.update_rows / 32)))
        for time in range(h):
            drift = base[time] - (base[time - 1] if time else 0.0)
            particles += drift + self.rng.normal(0.0, self.profile.process_sigma_ft, len(particles))
            jumps = self.rng.random(len(particles)) < self.profile.jump_probability
            particles[jumps] += self.rng.normal(0.0, self.profile.jump_sigma_ft, jumps.sum())
            particles = np.clip(particles, offsets[0], offsets[-1])
            if time % update_bins == 0 and np.isfinite(hw[time]):
                expected_gr = np.interp(particles, offsets, tw)
                log_weight = np.log(np.maximum(weights, 1e-30)) + _student_log_likelihood(
                    hw[time] - expected_gr,
                    self.profile.likelihood_scale_gr / 20.0,
                    self.defaults.student_df,
                )
                log_weight -= np.max(log_weight)
                weights = np.exp(log_weight)
                weights /= max(weights.sum(), 1e-30)
            clouds[time] = particles
            cloud_weights[time] = weights
            filtered[time] = float(np.sum(weights * particles))
            histogram, _ = np.histogram(
                particles,
                bins=np.r_[offsets - 0.25, offsets[-1] + 0.25],
                weights=weights,
            )
            histogram = gaussian_filter1d(histogram.astype(np.float32), sigma=1.0)
            density[time] = histogram / max(float(histogram.sum()), 1e-12)
            entropy[time] = float(-np.sum(weights * np.log(np.maximum(weights, 1e-30))))
            ess = 1.0 / max(float(np.sum(weights**2)), 1e-30)
            if ess < self.defaults.resample_ess_fraction * len(weights):
                indices = systematic_resample(weights, self.rng)
                particles = particles[indices]
                weights.fill(1.0 / len(weights))
        paths = self._ffbsi(clouds, cloud_weights, base)
        ffbsi = np.median(paths, axis=0).astype(np.float32)
        return PFResult(density, filtered, ffbsi, entropy, paths)

    def _ffbsi(self, clouds: np.ndarray, weights: np.ndarray, base: np.ndarray) -> np.ndarray:
        h, _ = clouds.shape
        paths = np.empty((self.defaults.ffbsi_paths, h), dtype=np.float32)
        variance = max(self.profile.process_sigma_ft**2, 0.25)
        for draw in range(self.defaults.ffbsi_paths):
            index = int(self.rng.choice(len(weights[-1]), p=weights[-1] / weights[-1].sum()))
            paths[draw, -1] = clouds[-1, index]
            for time in range(h - 2, -1, -1):
                drift = base[time + 1] - base[time]
                residual = paths[draw, time + 1] - (clouds[time] + drift)
                backward = weights[time].astype(np.float64) * np.exp(-0.5 * residual**2 / variance)
                if not np.isfinite(backward).all() or backward.sum() <= 0:
                    backward = weights[time].astype(np.float64)
                backward /= backward.sum()
                index = int(self.rng.choice(len(backward), p=backward))
                paths[draw, time] = clouds[time, index]
        return paths


class MultiProfileParticleFilter:
    def __init__(self, defaults: PFDefaults = PFDefaults(), seed: int = 0):
        self.defaults = defaults
        self.seed = seed

    def run(self, sample: PanelSample) -> PFResult:
        # The disclosed bank uses 256 stochastic trajectories across diverse
        # profiles. Split the 64 FFBSi traces over those runs, then aggregate.
        per_run = replace(self.defaults, ffbsi_paths=max(1, self.defaults.ffbsi_paths // self.defaults.seeds))
        profiles = self.defaults.profiles
        profile_probability = np.asarray([profile.weight for profile in profiles], dtype=np.float64)
        profile_probability /= profile_probability.sum()
        profile_index = np.random.default_rng(self.seed).choice(
            len(profiles), size=self.defaults.seeds, p=profile_probability
        )
        results = [
            BootstrapParticleFilter(per_run, profiles[int(which)], self.seed + 1009 * index).run(sample)
            for index, which in enumerate(profile_index)
        ]
        density = np.mean(np.stack([result.density for result in results]), axis=0)
        filtered = np.mean(np.stack([result.filtered_offset for result in results]), axis=0)
        ffbsi = np.median(np.stack([result.ffbsi_offset for result in results]), axis=0)
        entropy = -np.sum(density * np.log(np.maximum(density, 1e-12)), axis=1)
        return PFResult(
            density.astype(np.float32),
            filtered.astype(np.float32),
            ffbsi.astype(np.float32),
            entropy.astype(np.float32),
            np.stack([result.ffbsi_offset for result in results]),
        )
