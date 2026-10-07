from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import VERSIONS
from .data import PanelSample, WellRecord, _interp_nan, load_record


# Source-derived 0719 augmentation contract. These values were previously
# replaced by generic guessed probabilities, which materially changed training.
REVERSE_PROBABILITY = 0.25
GR_NOISE_SHIFT_PROBABILITY = 0.10
MD_STRETCH_PROBABILITY = 0.40
TAIL_CUT_PROBABILITY = 0.20
GR_TRANSFORM_PROBABILITY = 0.60
CHANNEL_MASK_PROBABILITY = 0.70
CHANNEL_DROP_PROBABILITY = 0.10
PF_CHANNEL_DROP_PROBABILITY = 0.05
SEQUENCE_MASK_PROBABILITY = 0.10
DIFF_BLOCK_REVERSE_PROBABILITY = 0.0


def build_trend_pool(data_root: str | Path, split: str, wells: list[str]) -> list[tuple[str, np.ndarray]]:
    """Build the cross-well suffix dTVT bank used by diff-block bootstrap."""
    pool: list[tuple[str, np.ndarray]] = []
    for well in wells:
        record = load_record(data_root, split, well)
        if record.truth is None:
            continue
        suffix = np.flatnonzero(~np.isfinite(record.tvt_input))
        if suffix.size < 1000:
            continue
        path = record.truth[suffix]
        increments = np.diff(path)
        increments = increments[np.isfinite(increments) & (np.abs(increments) < 0.15)]
        if increments.size:
            pool.append((well, increments.astype(np.float32)))
    if not pool:
        raise ValueError("cross-well trend pool is empty")
    return pool


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - values.mean()
    scale = float(values.std())
    return values / scale if scale > 1e-6 else np.zeros_like(values)


def _boxcar_random_component(
    rng: np.random.Generator, length: int, window: int
) -> np.ndarray:
    """Generate the source implementation's valid-convolution noise component."""
    window = max(1, int(window))
    raw = rng.normal(size=length + window - 1).astype(np.float32)
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return _standardize(np.convolve(raw, kernel, mode="valid"))


def _multiscale_tvt_noise(rng: np.random.Generator, length: int) -> np.ndarray:
    components = []
    for window, weight in zip((1536, 256, 2), (0.68, 0.20, 0.12)):
        components.append(np.sqrt(weight) * _boxcar_random_component(rng, length, window))
    return 0.95 * _standardize(np.sum(components, axis=0))


class RawWellAugmenter:
    """Source-derived 0719 physical simulation and raw-data augmentation."""

    def __init__(self, seed: int, profile: str, trend_pool: list[tuple[str, np.ndarray]]):
        self.rng = np.random.default_rng(seed)
        self.profile = profile
        self.trend_pool = trend_pool

    def __call__(self, record: WellRecord) -> WellRecord:
        output = record
        probability = VERSIONS.get(self.profile, VERSIONS["0719_V1"]).z_shift_probability
        if self.rng.random() < probability:
            output = self._z_shift(output)
        if self.rng.random() < REVERSE_PROBABILITY:
            output = self._reverse(output)
        if self.rng.random() < GR_NOISE_SHIFT_PROBABILITY:
            output = self._residual_shift(output)
        if self.rng.random() < GR_TRANSFORM_PROBABILITY:
            output = self._gr_transform(output)
        if self.rng.random() < MD_STRETCH_PROBABILITY:
            output = self._md_stretch(output)
        if self.rng.random() < TAIL_CUT_PROBABILITY:
            output = self._tail_crop(output)
        return output

    def _sample_bootstrap_trend(self, length: int, current_well: str) -> np.ndarray:
        increments = np.empty(max(0, length - 1), dtype=np.float32)
        position = 0
        while position < len(increments):
            block_length = int(self.rng.choice((1024, 2048, 4096), p=(0.45, 0.40, 0.15)))
            # sample_from_same_well=False means use the global bank. It does not
            # mean explicitly excluding the current well from that bank.
            _, donor = self.trend_pool[int(self.rng.integers(len(self.trend_pool)))]
            use = min(block_length, len(donor), len(increments) - position)
            start = int(self.rng.integers(0, len(donor) - use + 1))
            block = donor[start : start + use].copy()
            if use > 1 and DIFF_BLOCK_REVERSE_PROBABILITY > 0.0 and self.rng.random() < DIFF_BLOCK_REVERSE_PROBABILITY:
                block = -block[::-1].copy()
            if position and use:
                join = min(16, use)
                weights = (np.arange(join, dtype=np.float32) + 1) / (join + 1)
                block[:join] = (1 - weights) * increments[position - 1] + weights * block[:join]
            increments[position : position + use] = block
            position += use
        trend = np.empty(length, dtype=np.float32)
        trend[0] = 0.0
        if length > 1:
            trend[1:] = np.cumsum(increments)
        return trend

    def _z_shift(self, record: WellRecord) -> WellRecord:
        if record.truth is None:
            return record
        suffix = ~np.isfinite(record.tvt_input)
        suffix_indices = np.flatnonzero(suffix)
        if suffix_indices.size < 2:
            return record
        original_truth = record.truth.astype(np.float32, copy=True)
        truth = original_truth.copy()
        original_suffix = original_truth[suffix]

        # Faults are sparse surface jumps. They intentionally change the
        # geological surface seen by the independent prior before trajectory
        # resampling; they are not cancelled by an opposite Z adjustment.
        faulted_suffix = original_suffix.copy()
        if self.rng.random() < 0.05 and len(faulted_suffix) > 1:
            count = int(self.rng.integers(0, min(3, len(faulted_suffix) - 1) + 1))
            if count:
                points = self.rng.choice(np.arange(1, len(faulted_suffix)), count, replace=False)
                fault = np.zeros(len(faulted_suffix), dtype=np.float32)
                fault[points] = self.rng.laplace(0.0, 1.75, count)
                faulted_suffix += np.cumsum(fault)

        trend = self._sample_bootstrap_trend(len(original_suffix), record.well)
        simulated_suffix = faulted_suffix[0] + trend

        # Preserve sparse surface jumps from the original path.
        jumps = np.diff(faulted_suffix)
        jumps[np.abs(jumps) < 0.15] = 0.0
        simulated_suffix[1:] += np.cumsum(jumps)
        truth[suffix] = simulated_suffix

        z = record.z.astype(np.float32, copy=True)
        z[suffix] = faulted_suffix + record.z[suffix] - simulated_suffix

        # Apply the disclosed whole-well vertical translation while retaining
        # a 100-ft margin inside the typewell TVT support.
        anchor_tvt = float(record.anchor_tvt)
        shift_min = anchor_tvt - (float(record.typewell_tvt[-1]) - 100.0)
        shift_max = anchor_tvt - (float(record.typewell_tvt[0]) + 100.0)
        low = max(-40.0, shift_min)
        high = min(40.0, shift_max)
        range_shift = float(self.rng.uniform(low, high)) if low <= high else 0.0
        tvt_input = record.tvt_input.astype(np.float32, copy=True)
        if range_shift:
            truth -= range_shift
            z += range_shift
            visible = np.isfinite(tvt_input)
            tvt_input[visible] -= range_shift

        # TVT-bias-v2 residual simulation around the newly matched typewell GR.
        matched = _interp_nan(truth, record.typewell_tvt, record.typewell_gr)
        noise = np.zeros(len(truth), dtype=np.float32)
        tvt_noise = _multiscale_tvt_noise(self.rng, suffix_indices.size)
        corrupted_tvt = np.clip(
            truth[suffix] + tvt_noise,
            float(np.nanmin(record.typewell_tvt)),
            float(np.nanmax(record.typewell_tvt)),
        )
        corrupted_gr = _interp_nan(corrupted_tvt, record.typewell_tvt, record.typewell_gr)
        suffix_noise = corrupted_gr - matched[suffix]
        finite = np.isfinite(suffix_noise)
        if finite.any():
            ranges = ((5.6, 8.4), (8.4, 12.2), (12.2, 17.5))
            selected = int(self.rng.choice(3, p=(0.24, 0.51, 0.25)))
            target_std = float(self.rng.uniform(*ranges[selected]))
            current_std = float(np.std(suffix_noise[finite]))
            if current_std > 1e-6:
                suffix_noise[finite] *= target_std / current_std
            suffix_noise[finite] += float(self.rng.normal(0.0, 2.0))
        noise[suffix] = np.clip(suffix_noise, -36.0, 36.0)
        gr = matched + noise
        gr[np.isnan(record.gr)] = np.nan
        if not range_shift:
            gr[~suffix] = record.gr[~suffix]
        else:
            original_matched = _interp_nan(original_truth, record.typewell_tvt, record.typewell_gr)
            base_noise = record.gr - original_matched
            gr[~suffix] = matched[~suffix] + base_noise[~suffix]
        return replace(record, z=z, gr=gr.astype(np.float32), truth=truth, tvt_input=tvt_input)

    def _residual_shift(self, record: WellRecord) -> WellRecord:
        suffix = ~np.isfinite(record.tvt_input)
        if not suffix.any():
            return record
        matched = _interp_nan(record.truth, record.typewell_tvt, record.typewell_gr)
        residual = record.gr[suffix] - matched[suffix]
        shift = int(self.rng.uniform(-len(residual), len(residual)))
        gr = record.gr.copy()
        gr[suffix] = matched[suffix] + np.roll(residual, shift)
        return replace(record, gr=gr.astype(np.float32))

    def _gr_transform(self, record: WellRecord) -> WellRecord:
        # The disclosed affine/texture transform applies to typewell GR only.
        output = record.typewell_gr.astype(np.float32, copy=True)
        output = (self.rng.normal(1.0, 0.03) * output + self.rng.normal(0.0, 2.0)).astype(np.float32)
        draw = self.rng.random()
        smooth = uniform_filter1d(output, size=5, mode="nearest")
        if draw < 0.10:
            output = smooth
        elif draw < 0.20:
            output = output + self.rng.uniform(0.0, 0.3) * (output - smooth)
        return replace(record, typewell_gr=output.astype(np.float32))

    def _md_stretch(self, record: WellRecord) -> WellRecord:
        factor = float(self.rng.uniform(0.7, 1.3))
        length = max(2, int(len(record.md) * factor))
        source_axis = record.md.astype(np.float64)
        target_axis = np.linspace(source_axis[0], source_axis[-1], length)
        fields = {}
        for name in ("md", "x", "y", "z", "gr", "tvt_input", "truth"):
            value = getattr(record, name)
            fields[name] = None if value is None else np.interp(
                target_axis, source_axis, value, left=np.nan, right=np.nan
            ).astype(np.float32)
        fields["md"] = target_axis.astype(np.float32)
        return replace(record, **fields)

    def _tail_crop(self, record: WellRecord) -> WellRecord:
        suffix_length = len(record.md) - record.anchor_index - 1
        if suffix_length <= 1:
            return record
        cut = min(int(suffix_length * self.rng.uniform(0.0, 0.5)), suffix_length - 1)
        if not cut:
            return record
        end = len(record.md) - cut
        fields = {}
        for name in ("md", "x", "y", "z", "gr", "tvt_input", "truth"):
            value = getattr(record, name)
            fields[name] = None if value is None else value[:end]
        return replace(record, **fields)

    def _reverse(self, record: WellRecord) -> WellRecord:
        if record.truth is None:
            return record
        anchor_tvt = record.anchor_tvt
        visible = np.flatnonzero(np.isfinite(record.tvt_input))
        far = visible[np.abs(record.tvt_input[visible] - anchor_tvt) > 10.0]
        start = int(far[-1] + 1) if far.size else 0
        fields = {
            name: getattr(record, name)[start:][::-1].copy()
            for name in ("md", "x", "y", "z", "gr", "truth")
        }
        tvt_input = fields["truth"].copy()
        tvt_input[256:] = np.nan
        fields["tvt_input"] = tvt_input.astype(np.float32)
        fields["md"] = np.arange(len(tvt_input), dtype=np.float32)
        return replace(record, **fields)


class PanelAugmenter:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: PanelSample) -> PanelSample:
        values = sample.values.copy()
        if self.rng.random() < CHANNEL_MASK_PROBABILITY:
            pf_mask = self.rng.random() < PF_CHANNEL_DROP_PROBABILITY
            for index, name in enumerate(sample.channels):
                if name.startswith("pf_"):
                    if pf_mask:
                        values[index] = 0.0
                elif self.rng.random() < CHANNEL_DROP_PROBABILITY:
                    values[index] = 0.0
        if self.rng.random() < SEQUENCE_MASK_PROBABILITY:
            suffix = np.flatnonzero(sample.target_mask)
            if suffix.size:
                length = max(1, int(np.ceil(self.rng.uniform(0.0, 0.3) * suffix.size)))
                offset = int(self.rng.integers(0, suffix.size - length + 1))
                start, end = int(suffix[offset]), int(suffix[offset + length - 1]) + 1
                values[:, start:end, :] = 0.0
        return replace(sample, values=np.nan_to_num(values).astype(np.float32))
