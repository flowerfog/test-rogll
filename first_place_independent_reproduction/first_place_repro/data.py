from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BASE_CHANNELS, GridConfig

PF_CHANNELS = (
    "pf_particle_density_prob", "pf_prob_ffbsi", "pf_tvt_abs_gap",
    "pf_accumulated_tvt",
)
XY_CHANNELS = (
    "xy_tvt_diff",
    "xy_tvt_rel",
    "xy_tvt_abs_gap",
    "xy_mean_distance",
    "xy_p10_distance",
    "xy_prefix_weight_ratio",
    "xy_center_distance_ratio",
)


# Fixed train-population constants disclosed for the 0719 sequence tensor.
HORIZONTAL_GR = (87.43, 21.09)
TYPEWELL_GR = (78.62, 28.81)
TYPEWELL_TVT_REL = (0.0, 57.73)
SEEN_TVT_REL = (-18.08, 48.21)
Z_DIFF = (-0.016892, 0.106500)
QUADRATIC_STATS = {
    "hw_gr_quadratic_a": (0.000585828, 0.382492540),
    "hw_gr_quadratic_b": (0.001914565, 0.338615701),
    "hw_gr_quadratic_c": (-0.006968186, 1.025529919),
    "hw_gr_quadratic_rmse": (0.212819023, 0.131140903),
}


@dataclass(frozen=True)
class WellRecord:
    well: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt_input: np.ndarray
    truth: np.ndarray | None
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray

    @property
    def anchor_index(self) -> int:
        visible = np.flatnonzero(np.isfinite(self.tvt_input))
        if not len(visible):
            raise ValueError(f"{self.well}: no visible TVT_input")
        return int(visible[-1])

    @property
    def anchor_tvt(self) -> float:
        return float(self.tvt_input[self.anchor_index])


@dataclass(frozen=True)
class PanelSample:
    well: str
    values: np.ndarray
    channels: tuple[str, ...]
    target_probability: np.ndarray
    target_offset: np.ndarray
    target_mask: np.ndarray
    gr_gap: np.ndarray
    state_offsets: np.ndarray
    row_bins: np.ndarray
    selected_rows: np.ndarray
    eval_rows: np.ndarray
    anchor_tvt: float
    anchor_index: int

    def append(self, values: np.ndarray, names: Iterable[str]) -> "PanelSample":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 2:
            values = values[None]
        if values.shape[1:] != self.values.shape[1:]:
            raise ValueError((values.shape, self.values.shape))
        names = tuple(names)
        if values.shape[0] != len(names):
            raise ValueError("channel name count mismatch")
        return replace(
            self,
            values=np.concatenate([self.values, values], axis=0),
            channels=self.channels + names,
        )

    def select(self, names: Iterable[str]) -> "PanelSample":
        names = tuple(names)
        missing = set(names) - set(self.channels)
        if missing:
            raise KeyError(f"panel is missing channels: {sorted(missing)}")
        index = [self.channels.index(name) for name in names]
        return replace(self, values=self.values[index], channels=names)


def normalize_panel(sample: PanelSample, population_stats) -> PanelSample:
    # Base channels are constructed in their final disclosed numerical space.
    # Auxiliary branches are likewise responsible for their own fixed scaling.
    del population_stats
    return sample


def _column(frame: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)


def load_record(data_root: str | Path, split: str, well: str) -> WellRecord:
    folder = Path(data_root) / split
    horizontal = pd.read_csv(folder / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(folder / f"{well}__typewell.csv")
    return WellRecord(
        well=str(well),
        md=_column(horizontal, "MD"),
        x=_column(horizontal, "X"),
        y=_column(horizontal, "Y"),
        z=_column(horizontal, "Z"),
        gr=_column(horizontal, "GR"),
        tvt_input=_column(horizontal, "TVT_input"),
        truth=_column(horizontal, "TVT") if "TVT" in horizontal else None,
        typewell_tvt=_column(typewell, "TVT"),
        typewell_gr=_column(typewell, "GR"),
    )


def list_wells(data_root: str | Path, split: str) -> list[str]:
    suffix = "__horizontal_well.csv"
    return sorted(
        path.name[: -len(suffix)] for path in (Path(data_root) / split).glob(f"*{suffix}")
    )


def _robust_center_scale(values: np.ndarray, fallback: float = 1.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, fallback
    center = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(np.std(finite))
    return center, max(scale, fallback)


def _interp_nan(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    valid = np.isfinite(xp) & np.isfinite(fp)
    if valid.sum() < 2:
        return np.full_like(x, np.nan, dtype=np.float32)
    order = np.argsort(xp[valid])
    return np.interp(x, xp[valid][order], fp[valid][order], left=np.nan, right=np.nan).astype(
        np.float32
    )


def calibrated_typewell_gr(
    record: WellRecord, state_tvt: np.ndarray, blend: float = 0.7
) -> np.ndarray:
    # The disclosed implementation calibrates the native typewell samples and
    # only then interpolates that calibrated curve onto the candidate axis.
    calibrated_native = record.typewell_gr.astype(np.float32, copy=True)
    anchor_tvt = record.anchor_tvt
    visible = (
        np.isfinite(record.tvt_input)
        & np.isfinite(record.gr)
        & (record.tvt_input >= anchor_tvt - 100.0)
        & (record.tvt_input <= anchor_tvt + 100.0)
    )
    # Visible GR is aggregated on the disclosed 0.25-ft TVT grid before it is
    # interpolated to the candidate states. This avoids letting raw sampling
    # density dominate calibration.
    visible_tvt = record.tvt_input[visible]
    visible_gr = record.gr[visible]
    if len(visible_tvt):
        bins = np.rint(visible_tvt / 0.25).astype(np.int64)
        order = np.argsort(bins)
        bins = bins[order]
        visible_gr = visible_gr[order]
        unique, start = np.unique(bins, return_index=True)
        sums = np.add.reduceat(visible_gr, start)
        counts = np.diff(np.r_[start, len(visible_gr)])
        reference_tvt = unique.astype(np.float32) * 0.25
        reference_gr = (sums / counts).astype(np.float32)
        support = (
            (record.typewell_tvt >= reference_tvt.min())
            & (record.typewell_tvt <= reference_tvt.max())
        )
        if support.any():
            reference_at_native = np.interp(
                record.typewell_tvt[support], reference_tvt, reference_gr,
                left=np.nan, right=np.nan,
            ).astype(np.float32)
            finite = np.isfinite(reference_at_native)
            native_indices = np.flatnonzero(support)[finite]
            calibrated_native[native_indices] = (
                blend * reference_at_native[finite]
                + (1.0 - blend) * calibrated_native[native_indices]
            )
    return _interp_nan(state_tvt, record.typewell_tvt, calibrated_native)


def select_rows(record: WellRecord, grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    anchor = record.anchor_index
    start = max(0, anchor - grid.visible_rows + 1)
    selected = np.full(grid.horizontal_bins * grid.horizontal_downsample, -1, dtype=np.int32)
    prefix = np.arange(start, anchor + 1, dtype=np.int32)
    prefix_destination = grid.visible_rows - len(prefix)
    selected[prefix_destination : grid.visible_rows] = prefix
    suffix = np.arange(
        anchor + 1,
        min(len(record.md), anchor + grid.target_rows + 1),
        dtype=np.int32,
    )
    selected[grid.visible_rows : grid.visible_rows + len(suffix)] = suffix
    row_bins = np.repeat(np.arange(grid.horizontal_bins, dtype=np.int32), grid.horizontal_downsample)
    return selected, row_bins


def horizontal_bin_features(record: WellRecord, selected: np.ndarray, grid: GridConfig) -> dict[str, np.ndarray]:
    valid_rows = selected >= 0
    gr_rows = np.full(len(selected), np.nan, dtype=np.float32)
    z_rows = np.full(len(selected), np.nan, dtype=np.float32)
    tvt_rows = np.full(len(selected), np.nan, dtype=np.float32)
    md_rows = np.full(len(selected), np.nan, dtype=np.float32)
    gr_rows[valid_rows] = record.gr[selected[valid_rows]]
    z_rows[valid_rows] = record.z[selected[valid_rows]]
    tvt_rows[valid_rows] = record.tvt_input[selected[valid_rows]]
    md_rows[valid_rows] = record.md[selected[valid_rows]]
    gr = gr_rows.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    z = z_rows.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    tvt = tvt_rows.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    md = md_rows.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    result = {name: np.zeros(grid.horizontal_bins, dtype=np.float32) for name in (
        "mean", "nan", "std", "slope", "last_first", "qa", "qb", "qc", "qrmse", "z", "z_mask", "visible_tvt", "visible_mask"
    )}
    for name in ("qa", "qb", "qc", "qrmse", "visible_tvt"):
        result[name].fill(np.nan)
    for index in range(grid.horizontal_bins):
        values = gr[index]
        actual = selected.reshape(grid.horizontal_bins, grid.horizontal_downsample)[index] >= 0
        finite = actual & np.isfinite(values)
        result["nan"][index] = (
            1.0 - float(finite.sum() / actual.sum()) if actual.any() else 1.0
        )
        if finite.any():
            result["mean"][index] = float(np.nanmean(values))
            result["std"][index] = float(np.nanstd(values))
            locations = np.flatnonzero(finite)
            result["last_first"][index] = float(values[locations[-1]] - values[locations[0]])
        if finite.sum() >= 2:
            linear = np.polyfit(md[index][finite], values[finite], 1)
            result["slope"][index] = float(linear[0])
        if finite.sum() >= 6:
            locations = np.flatnonzero(finite).astype(np.float64)
            center = 0.5 * (locations[0] + locations[-1])
            half_span = 0.5 * (locations[-1] - locations[0])
            fit_x = (locations - center) / max(half_span, 1.0)
            coefficient = np.polyfit(fit_x, values[finite], 2)
            fitted = np.polyval(coefficient, fit_x)
            result["qa"][index], result["qb"][index], result["qc"][index] = coefficient
            result["qrmse"][index] = float(np.sqrt(np.mean((values[finite] - fitted) ** 2)))
        z_finite = z[index][np.isfinite(z[index])]
        if len(z_finite):
            result["z"][index] = float(z_finite.mean())
            result["z_mask"][index] = 1.0
        visible = tvt[index][np.isfinite(tvt[index])]
        if len(visible):
            result["visible_tvt"][index] = float(visible.mean())
            result["visible_mask"][index] = 1.0
    return result


def _bin_truth(record: WellRecord, selected: np.ndarray, grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    target = np.full(grid.horizontal_bins, np.nan, dtype=np.float32)
    mask = np.zeros(grid.horizontal_bins, dtype=bool)
    if record.truth is None:
        return target, mask
    rows = selected.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    for index, group in enumerate(rows):
        group = group[group >= 0]
        if not len(group):
            continue
        values = record.truth[group]
        finite = np.isfinite(values)
        if finite.any():
            # The reference loss supervises only the hidden suffix. Visible
            # prefix bins are context, never easy target rows.
            suffix = group > record.anchor_index
            finite &= suffix
            if finite.any():
                target[index] = float(values[finite].mean() - record.anchor_tvt)
                mask[index] = True
    return target, mask


def build_target(offsets: np.ndarray, truth: np.ndarray, mask: np.ndarray, sigma_ft: float) -> np.ndarray:
    classification_truth = np.clip(truth, offsets[0], offsets[-1])
    distance = offsets[None, :] - classification_truth[:, None]
    probability = np.exp(-0.5 * (distance / sigma_ft) ** 2).astype(np.float32)
    probability[~mask] = 0.0
    denominator = probability.sum(axis=1, keepdims=True)
    return np.divide(probability, denominator, out=np.zeros_like(probability), where=denominator > 0)


def build_base_panel(
    record: WellRecord,
    grid: GridConfig = GridConfig(),
    target_sigma_ft: float = 1.25,
    calibration_blend: float = 0.7,
    population_stats=None,
    apply_channel_normalization: bool = True,
) -> PanelSample:
    selected, row_bins = select_rows(record, grid)
    features = horizontal_bin_features(record, selected, grid)
    offsets = grid.state_offsets
    state_tvt = record.anchor_tvt + offsets
    tw_gr_raw = calibrated_typewell_gr(record, state_tvt, calibration_blend)
    if population_stats is None:
        raise ValueError("build_base_panel requires outer-train population statistics")
    horizontal_gr_mean, horizontal_gr_std = HORIZONTAL_GR
    typewell_gr_mean, typewell_gr_std = TYPEWELL_GR
    tw_nan = ~np.isfinite(tw_gr_raw)
    tw_gr = np.nan_to_num((tw_gr_raw - typewell_gr_mean) / typewell_gr_std, nan=0.0)
    hw_mean = (features["mean"] - horizontal_gr_mean) / horizontal_gr_std
    hw_mean[features["nan"] >= 1.0] = 0.0
    hw_std = np.nan_to_num(features["std"] / horizontal_gr_std)
    slope = np.clip(features["slope"] * grid.horizontal_downsample / horizontal_gr_std, -8.0, 8.0)
    quadratic = {}
    for key, source in (
        ("hw_gr_quadratic_a", features["qa"] / horizontal_gr_std),
        ("hw_gr_quadratic_b", features["qb"] / horizontal_gr_std),
        ("hw_gr_quadratic_c", (features["qc"] - horizontal_gr_mean) / horizontal_gr_std),
        ("hw_gr_quadratic_rmse", features["qrmse"] / horizontal_gr_std),
    ):
        mean, std = QUADRATIC_STATS[key]
        normalized = np.nan_to_num(np.clip((source - mean) / std, -8.0, 8.0))
        normalized[features["nan"] >= 1.0] = 0.0
        quadratic[key] = normalized
    seen_tvt_rel = features["visible_tvt"] - record.anchor_tvt
    # _finish_feature is applied before normalization in the source pipeline.
    seen_tvt_rel_finished = np.nan_to_num(seen_tvt_rel, nan=0.0)
    horizontal = {
        "hw_gr_mean": hw_mean,
        "hw_gr_nan": features["nan"],
        "hw_gr_std": hw_std,
        "hw_gr_slope": np.nan_to_num(slope),
        "hw_gr_last_first": np.nan_to_num(features["last_first"] / horizontal_gr_std),
        **quadratic,
        "visible_tvt_rel": (
            (seen_tvt_rel_finished - SEEN_TVT_REL[0]) / SEEN_TVT_REL[1]
        ),
        "visible_tvt_mask": features["visible_mask"],
    }
    panel: dict[str, np.ndarray] = {
        "tw_gr": np.broadcast_to(tw_gr[None, :], (grid.horizontal_bins, grid.state_bins)),
        "tw_gr_nan": np.broadcast_to(tw_nan.astype(np.float32)[None, :], (grid.horizontal_bins, grid.state_bins)),
        "tw_tvt_rel": np.broadcast_to(((offsets - TYPEWELL_TVT_REL[0]) / TYPEWELL_TVT_REL[1])[None, :], (grid.horizontal_bins, grid.state_bins)),
    }
    for name, values in horizontal.items():
        panel[name] = np.broadcast_to(values[:, None], (grid.horizontal_bins, grid.state_bins))
    z_step = np.full(len(selected), np.nan, dtype=np.float32)
    valid = selected >= 0
    previous_valid = valid & np.r_[False, valid[:-1]]
    contiguous = previous_valid & (selected == np.r_[-2, selected[:-1]] + 1)
    z_step[contiguous] = record.z[selected[contiguous]] - record.z[selected[contiguous] - 1]
    z_groups = z_step.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    z_finite = np.isfinite(z_groups)
    z_counts = z_finite.sum(axis=1)
    z_mean = np.divide(
        np.nansum(z_groups, axis=1), z_counts,
        out=np.full(grid.horizontal_bins, np.nan, dtype=np.float32),
        where=z_counts > 0,
    )
    z_diff = np.nan_to_num((z_mean - Z_DIFF[0]) / Z_DIFF[1])
    panel["z_diff"] = np.broadcast_to(z_diff[:, None], (grid.horizontal_bins, grid.state_bins))
    raw_hw = features["mean"]
    raw_gap = np.abs(raw_hw[:, None] - tw_gr_raw[None, :]) / horizontal_gr_std
    invalid_gap = (features["nan"] >= 1.0)[:, None] | tw_nan[None, :]
    panel["tw_hw_gr_abs_gap"] = np.where(invalid_gap, 0.0, raw_gap)
    raw_tvt_gap = np.clip(
        np.abs(offsets[None, :] - seen_tvt_rel_finished[:, None]), 0.0, 100.0
    ) / TYPEWELL_TVT_REL[1]
    panel["tw_visible_tvt_abs_gap"] = raw_tvt_gap
    values = np.stack([panel[name] for name in BASE_CHANNELS]).astype(np.float32)
    truth, target_mask = _bin_truth(record, selected, grid)
    probability = build_target(offsets, truth, target_mask, target_sigma_ft)
    gr_gap = np.square((tw_gr_raw[None, :] - raw_hw[:, None]) / horizontal_gr_std)
    gr_gap = np.minimum(gr_gap, 25.0).astype(np.float32)
    valid_horizontal_gr = features["nan"] < 1.0
    gr_gap[~valid_horizontal_gr, :] = 0.0
    gr_gap[valid_horizontal_gr[:, None] & tw_nan[None, :]] = 25.0
    eval_rows = selected[(selected >= 0) & (selected > record.anchor_index)]
    sample = PanelSample(
        well=record.well,
        values=values,
        channels=BASE_CHANNELS,
        target_probability=probability,
        target_offset=np.nan_to_num(truth, nan=0.0),
        target_mask=target_mask,
        gr_gap=gr_gap,
        state_offsets=offsets,
        row_bins=row_bins,
        selected_rows=selected,
        eval_rows=eval_rows,
        anchor_tvt=record.anchor_tvt,
        anchor_index=record.anchor_index,
    )
    if apply_channel_normalization:
        sample = normalize_panel(sample, population_stats)
    return sample


def append_pf_channels(sample: PanelSample, density: np.ndarray, path_offset: np.ndarray) -> PanelSample:
    density = np.asarray(density, dtype=np.float32)
    path_offset = np.asarray(path_offset, dtype=np.float32)
    if density.shape != sample.gr_gap.shape or path_offset.shape != (density.shape[0],):
        raise ValueError("PF shape mismatch")
    ffbsi_probability = np.exp(
        -0.5 * ((sample.state_offsets[None, :] - path_offset[:, None]) / 1.5) ** 2
    ).astype(np.float32)
    ffbsi_probability /= np.maximum(ffbsi_probability.sum(axis=1, keepdims=True), 1e-12)
    channels = np.stack(
        [
            density,
            ffbsi_probability,
            np.abs(sample.state_offsets[None, :] - path_offset[:, None]) / 100.0,
            np.broadcast_to((path_offset / 100.0)[:, None], density.shape),
        ]
    )
    return sample.append(channels, PF_CHANNELS)


def append_xy_channels(sample: PanelSample, xy) -> PanelSample:
    path = np.asarray(xy.tvt_offset, dtype=np.float32)
    height, width = sample.gr_gap.shape
    if path.shape != (height,):
        raise ValueError("XY shape mismatch")
    quality = (
        xy.mean_distance,
        xy.p10_distance,
        xy.prefix_weight_ratio,
        xy.center_distance_ratio,
    )
    channels = [
        np.broadcast_to(np.gradient(path)[:, None] / 100.0, (height, width)),
        np.broadcast_to((path / 100.0)[:, None], (height, width)),
        np.abs(sample.state_offsets[None, :] - path[:, None]) / 100.0,
    ]
    channels.extend(np.full((height, width), float(value), dtype=np.float32) for value in quality)
    return sample.append(np.stack(channels), XY_CHANNELS)
