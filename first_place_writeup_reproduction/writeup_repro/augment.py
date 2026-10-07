from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .data import PanelSample, WellRecord, _interp_nan


class RawWellAugmenter:
    """Writeup-disclosed physical and GR augmentations applied before rasterization."""

    def __init__(self, seed: int = 0, profile: str = "base_a"):
        self.rng = np.random.default_rng(seed)
        self.profile = profile

    def __call__(self, record: WellRecord) -> WellRecord:
        output = record
        if self.rng.random() < 0.45:
            output = self._z_shift(output)
        if self.rng.random() < 0.12:
            output = self._fault_jump(output)
        if self.rng.random() < 0.75:
            output = self._gr_transform(output)
        if self.rng.random() < 0.18:
            output = self._residual_shift(output)
        if self.rng.random() < 0.12:
            output = self._md_stretch(output)
        if self.rng.random() < 0.10:
            output = self._tail_crop(output)
        if self.rng.random() < 0.08:
            output = self._reverse(output)
        return output

    def _z_shift(self, record: WellRecord) -> WellRecord:
        truth = record.truth
        if truth is None or len(truth) < 64:
            return record
        increments = np.diff(truth).astype(np.float32)
        block = int(self.rng.integers(32, min(257, max(33, len(increments)))))
        sampled = []
        while sum(map(len, sampled)) < len(increments):
            start = int(self.rng.integers(0, max(1, len(increments) - block + 1)))
            sampled.append(increments[start : start + block])
        sampled_increment = np.concatenate(sampled)[: len(increments)]
        synthetic = np.r_[truth[0], truth[0] + np.cumsum(sampled_increment)].astype(np.float32)
        synthetic += gaussian_filter1d(self.rng.normal(0, 1.5, len(synthetic)), sigma=24)
        surface = truth + record.z
        z = surface - synthetic
        gr = _interp_nan(synthetic, record.typewell_tvt, record.typewell_gr)
        residual = gaussian_filter1d(self.rng.normal(0, 4.0, len(gr)), sigma=3)
        gr = np.where(np.isfinite(gr), gr + residual, record.gr)
        visible = np.isfinite(record.tvt_input)
        tvt_input = np.where(visible, synthetic, np.nan).astype(np.float32)
        return replace(record, z=z.astype(np.float32), gr=gr.astype(np.float32), truth=synthetic, tvt_input=tvt_input)

    def _fault_jump(self, record: WellRecord) -> WellRecord:
        if record.truth is None or len(record.truth) < 128:
            return record
        point = int(self.rng.integers(record.anchor_index + 1, len(record.truth))) if record.anchor_index + 1 < len(record.truth) else len(record.truth)
        if point >= len(record.truth):
            return record
        jump = float(self.rng.normal(0, 18.0))
        truth = record.truth.copy()
        truth[point:] += jump
        gr = record.gr.copy()
        rematched = _interp_nan(truth[point:], record.typewell_tvt, record.typewell_gr)
        gr[point:] = np.where(np.isfinite(rematched), rematched, gr[point:])
        return replace(record, truth=truth, gr=gr)

    def _gr_transform(self, record: WellRecord) -> WellRecord:
        scale = float(self.rng.lognormal(0.0, 0.12))
        shift = float(self.rng.normal(0, 8.0))
        jitter = float(self.rng.normal(0, 2.0))
        return replace(
            record,
            gr=(scale * record.gr + shift).astype(np.float32),
            typewell_gr=(scale * record.typewell_gr + shift + jitter).astype(np.float32),
        )

    def _residual_shift(self, record: WellRecord) -> WellRecord:
        matched = _interp_nan(record.tvt_input, record.typewell_tvt, record.typewell_gr)
        residual = np.nan_to_num(record.gr - matched)
        shift = int(self.rng.integers(-128, 129))
        gr = np.where(np.isfinite(matched), matched + np.roll(residual, shift), record.gr)
        return replace(record, gr=gr.astype(np.float32))

    def _md_stretch(self, record: WellRecord) -> WellRecord:
        factor = float(self.rng.uniform(0.92, 1.08))
        md = record.md[0] + (record.md - record.md[0]) * factor
        return replace(record, md=md.astype(np.float32))

    def _tail_crop(self, record: WellRecord) -> WellRecord:
        minimum = record.anchor_index + 256
        if len(record.md) <= minimum:
            return record
        end = int(self.rng.integers(minimum, len(record.md) + 1))
        fields = {}
        for name in ("md", "x", "y", "z", "gr", "tvt_input", "truth"):
            value = getattr(record, name)
            fields[name] = None if value is None else value[:end]
        return replace(record, **fields)

    def _reverse(self, record: WellRecord) -> WellRecord:
        if record.truth is None:
            return record
        fields = {}
        for name in ("md", "x", "y", "z", "gr", "truth"):
            fields[name] = getattr(record, name)[::-1].copy()
        visible_count = min(record.anchor_index + 1, len(record.truth))
        tvt_input = np.full(len(record.truth), np.nan, dtype=np.float32)
        tvt_input[:visible_count] = fields["truth"][:visible_count]
        fields["tvt_input"] = tvt_input
        return replace(record, **fields)


class PanelAugmenter:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: PanelSample) -> PanelSample:
        values = sample.values.copy()
        if self.rng.random() < 0.20:
            count = int(self.rng.integers(1, max(2, values.shape[0] // 4)))
            values[self.rng.choice(values.shape[0], count, replace=False)] = 0
        if self.rng.random() < 0.20:
            maximum_width = min(64, values.shape[2])
            minimum_width = min(4, maximum_width)
            start = int(self.rng.integers(0, values.shape[2] - minimum_width + 1))
            available = min(maximum_width, values.shape[2] - start)
            width = int(self.rng.integers(minimum_width, available + 1))
            values[:, :, start : start + width] = 0
        if self.rng.random() < 0.20:
            h = int(self.rng.integers(8, 48))
            w = int(self.rng.integers(8, 64))
            y = int(self.rng.integers(0, max(1, values.shape[1] - h)))
            x = int(self.rng.integers(0, max(1, values.shape[2] - w)))
            values[:, y : y + h, x : x + w] = 0
        if "pf_particle_density_prob" in sample.channels and self.rng.random() < 0.25:
            channel = sample.channels.index("pf_particle_density_prob")
            values[channel] = np.roll(
                values[channel],
                (int(self.rng.integers(-8, 9)), int(self.rng.integers(-12, 13))),
                axis=(0, 1),
            )
        return replace(sample, values=np.nan_to_num(values).astype(np.float32))
