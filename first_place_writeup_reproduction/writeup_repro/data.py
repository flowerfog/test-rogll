from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import GridConfig


BASE_CHANNELS = (
    "tw_gr",
    "tw_gr_nan",
    "tw_tvt_rel",
    "hw_gr_mean",
    "hw_gr_nan",
    "hw_gr_std",
    "hw_gr_slope",
    "hw_gr_last_first",
    "hw_gr_quadratic_a",
    "hw_gr_quadratic_b",
    "hw_gr_quadratic_c",
    "hw_gr_quadratic_rmse",
    "visible_tvt_rel",
    "visible_tvt_mask",
    "tw_hw_gr_abs_gap",
    "tw_visible_tvt_abs_gap",
    "z_diff",
)

PF_CHANNELS = ("pf_particle_density_prob", "pf_tvt_abs_gap", "pf_accumulated_tvt")
XY_CHANNELS = (
    "xy_tvt_diff",
    "xy_tvt_rel",
    "xy_tvt_abs_gap",
    "xy_mean_distance",
    "xy_p10_distance",
    "xy_prefix_weight_ratio",
    "xy_center_distance_ratio",
)


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
    record: WellRecord, state_tvt: np.ndarray, blend: float = 0.5
) -> np.ndarray:
    original = _interp_nan(state_tvt, record.typewell_tvt, record.typewell_gr)
    visible = np.isfinite(record.tvt_input) & np.isfinite(record.gr)
    calibrated = _interp_nan(state_tvt, record.tvt_input[visible], record.gr[visible])
    both = np.isfinite(original) & np.isfinite(calibrated)
    output = original.copy()
    output[both] = blend * calibrated[both] + (1.0 - blend) * original[both]
    output[~np.isfinite(output) & np.isfinite(calibrated)] = calibrated[
        ~np.isfinite(output) & np.isfinite(calibrated)
    ]
    return output


def select_rows(record: WellRecord, grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    anchor = record.anchor_index
    start = max(0, anchor - grid.visible_rows + 1)
    rows = np.arange(start, min(len(record.md), anchor + grid.target_rows + 1), dtype=np.int32)
    selected = np.full(grid.horizontal_bins * grid.horizontal_downsample, -1, dtype=np.int32)
    selected[: min(len(selected), len(rows))] = rows[: len(selected)]
    row_bins = np.repeat(np.arange(grid.horizontal_bins, dtype=np.int32), grid.horizontal_downsample)
    return selected, row_bins


def horizontal_bin_features(record: WellRecord, selected: np.ndarray, grid: GridConfig) -> dict[str, np.ndarray]:
    valid_rows = selected >= 0
    gr_rows = np.full(len(selected), np.nan, dtype=np.float32)
    z_rows = np.full(len(selected), np.nan, dtype=np.float32)
    tvt_rows = np.full(len(selected), np.nan, dtype=np.float32)
    gr_rows[valid_rows] = record.gr[selected[valid_rows]]
    z_rows[valid_rows] = record.z[selected[valid_rows]]
    tvt_rows[valid_rows] = record.tvt_input[selected[valid_rows]]
    gr = gr_rows.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    z = z_rows.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    tvt = tvt_rows.reshape(grid.horizontal_bins, grid.horizontal_downsample)
    position = np.linspace(-1.0, 1.0, grid.horizontal_downsample, dtype=np.float32)
    result = {name: np.zeros(grid.horizontal_bins, dtype=np.float32) for name in (
        "mean", "nan", "std", "slope", "last_first", "qa", "qb", "qc", "qrmse", "z", "visible_tvt", "visible_mask"
    )}
    for index in range(grid.horizontal_bins):
        values = gr[index]
        finite = np.isfinite(values)
        result["nan"][index] = 1.0 - float(finite.mean())
        if finite.any():
            result["mean"][index] = float(np.nanmean(values))
            result["std"][index] = float(np.nanstd(values))
            locations = np.flatnonzero(finite)
            result["last_first"][index] = float(values[locations[-1]] - values[locations[0]])
        if finite.sum() >= 2:
            linear = np.polyfit(position[finite], values[finite], 1)
            result["slope"][index] = float(linear[0])
        if finite.sum() >= 3:
            coefficient = np.polyfit(position[finite], values[finite], 2)
            fitted = np.polyval(coefficient, position[finite])
            result["qa"][index], result["qb"][index], result["qc"][index] = coefficient
            result["qrmse"][index] = float(np.sqrt(np.mean((values[finite] - fitted) ** 2)))
        z_finite = z[index][np.isfinite(z[index])]
        if len(z_finite):
            result["z"][index] = float(z_finite.mean())
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
            target[index] = float(values[finite].mean() - record.anchor_tvt)
            mask[index] = True
    return target, mask


def build_target(offsets: np.ndarray, truth: np.ndarray, mask: np.ndarray, sigma_ft: float) -> np.ndarray:
    distance = offsets[None, :] - truth[:, None]
    probability = np.exp(-0.5 * (distance / sigma_ft) ** 2).astype(np.float32)
    probability[~mask] = 0.0
    denominator = probability.sum(axis=1, keepdims=True)
    return np.divide(probability, denominator, out=np.zeros_like(probability), where=denominator > 0)


def build_base_panel(
    record: WellRecord,
    grid: GridConfig = GridConfig(),
    target_sigma_ft: float = 1.5,
    calibration_blend: float = 0.5,
) -> PanelSample:
    selected, row_bins = select_rows(record, grid)
    features = horizontal_bin_features(record, selected, grid)
    offsets = grid.state_offsets
    state_tvt = record.anchor_tvt + offsets
    tw_gr_raw = calibrated_typewell_gr(record, state_tvt, calibration_blend)
    gr_center, gr_scale = _robust_center_scale(
        np.concatenate([record.typewell_gr, record.gr[np.isfinite(record.gr)]])
    )
    tw_nan = ~np.isfinite(tw_gr_raw)
    tw_gr = np.nan_to_num((tw_gr_raw - gr_center) / gr_scale, nan=0.0)
    hw_mean = (features["mean"] - gr_center) / gr_scale
    hw_mean[features["nan"] >= 1.0] = 0.0
    hw_std = features["std"] / gr_scale
    horizontal = {
        "hw_gr_mean": hw_mean,
        "hw_gr_nan": features["nan"],
        "hw_gr_std": hw_std,
        "hw_gr_slope": features["slope"] / gr_scale,
        "hw_gr_last_first": features["last_first"] / gr_scale,
        "hw_gr_quadratic_a": features["qa"] / gr_scale,
        "hw_gr_quadratic_b": features["qb"] / gr_scale,
        "hw_gr_quadratic_c": (features["qc"] - gr_center) / gr_scale,
        "hw_gr_quadratic_rmse": features["qrmse"] / gr_scale,
        "visible_tvt_rel": np.where(
            features["visible_mask"] > 0,
            (features["visible_tvt"] - record.anchor_tvt) / grid.state_radius_ft,
            0.0,
        ),
        "visible_tvt_mask": features["visible_mask"],
        "z_diff": (features["z"] - float(record.z[record.anchor_index])) / grid.state_radius_ft,
    }
    panel: dict[str, np.ndarray] = {
        "tw_gr": np.broadcast_to(tw_gr[None, :], (grid.horizontal_bins, grid.state_bins)),
        "tw_gr_nan": np.broadcast_to(tw_nan.astype(np.float32)[None, :], (grid.horizontal_bins, grid.state_bins)),
        "tw_tvt_rel": np.broadcast_to((offsets / grid.state_radius_ft)[None, :], (grid.horizontal_bins, grid.state_bins)),
    }
    for name, values in horizontal.items():
        panel[name] = np.broadcast_to(values[:, None], (grid.horizontal_bins, grid.state_bins))
    panel["tw_hw_gr_abs_gap"] = np.abs(panel["tw_gr"] - panel["hw_gr_mean"])
    panel["tw_visible_tvt_abs_gap"] = np.abs(
        panel["tw_tvt_rel"] - panel["visible_tvt_rel"]
    ) * panel["visible_tvt_mask"]
    values = np.stack([panel[name] for name in BASE_CHANNELS]).astype(np.float32)
    truth, target_mask = _bin_truth(record, selected, grid)
    target_mask &= (truth >= offsets[0]) & (truth <= offsets[-1])
    probability = build_target(offsets, truth, target_mask, target_sigma_ft)
    gr_gap = np.abs(tw_gr[None, :] - hw_mean[:, None]).astype(np.float32)
    eval_rows = selected[(selected >= 0) & (selected > record.anchor_index)]
    return PanelSample(
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


def append_pf_channels(sample: PanelSample, density: np.ndarray, path_offset: np.ndarray) -> PanelSample:
    density = np.asarray(density, dtype=np.float32)
    path_offset = np.asarray(path_offset, dtype=np.float32)
    if density.shape != sample.gr_gap.shape or path_offset.shape != (density.shape[0],):
        raise ValueError("PF shape mismatch")
    channels = np.stack(
        [
            density,
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
