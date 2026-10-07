from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from .config import HEIGHT, PADDED_ROWS, RAW_ROWS, STATE_AXIS, STATE_STEP, VISIBLE_ROWS, WIDTH


_AUGMENTATION = {
    "0719_V1": {"gr_noise": .10, "tw_jitter": 0.0, "tw_drift": 0.0, "pf_rotate": .10, "pf_mask": .05, "pf_corrupt": 0.0, "a_std": .03},
    "0724_V1": {"gr_noise": 0.0, "tw_jitter": 0.0, "tw_drift": 0.0, "pf_rotate": .10, "pf_mask": .05, "pf_corrupt": 0.0, "a_std": .03},
    "0729_V3": {"gr_noise": 0.0, "tw_jitter": 0.0, "tw_drift": 0.0, "pf_rotate": .10, "pf_mask": .05, "pf_corrupt": 0.0, "a_std": .03},
    "0801_V1": {"gr_noise": 0.0, "tw_jitter": .20, "tw_drift": .30, "pf_rotate": .10, "pf_mask": .05, "pf_corrupt": 0.0, "a_std": .05},
    "0801_V2": {"gr_noise": 0.0, "tw_jitter": .20, "tw_drift": .30, "pf_rotate": .10, "pf_mask": .05, "pf_corrupt": 0.0, "a_std": .05},
    "0802_V2": {"gr_noise": 0.0, "tw_jitter": .05, "tw_drift": 0.0, "pf_rotate": .05, "pf_mask": .15, "pf_corrupt": .05, "a_std": .03},
    "0803_V2": {"gr_noise": 0.0, "tw_jitter": .20, "tw_drift": .30, "pf_rotate": .10, "pf_mask": .05, "pf_corrupt": 0.0, "a_std": .05},
}


@dataclass
class Well:
    well: str
    horizontal: pd.DataFrame
    typewell: pd.DataFrame
    anchor_index: int
    anchor_tvt: float


@dataclass
class Panel:
    well: str
    values: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    eval_mask: np.ndarray
    gr_gap: np.ndarray
    state_tvt: np.ndarray
    column_rows: np.ndarray
    truth_tvt: np.ndarray
    raw_truth_tvt: np.ndarray
    raw_eval_mask: np.ndarray
    anchor_tvt: float


def list_wells(data_root: Path, split: str = "train") -> list[str]:
    suffix = "__horizontal_well.csv"
    return sorted(p.name.removesuffix(suffix) for p in (data_root / split).glob(f"*{suffix}"))


def load_well(data_root: Path, well: str, split: str = "train") -> Well:
    folder = data_root / split
    hz = pd.read_csv(folder / f"{well}__horizontal_well.csv")
    tw = pd.read_csv(folder / f"{well}__typewell.csv")
    visible = np.flatnonzero(np.isfinite(hz["TVT_input"].to_numpy(float)))
    if not len(visible):
        raise ValueError(f"{well}: no visible TVT_input")
    anchor_index = int(visible[-1])
    return Well(well, hz, tw, anchor_index, float(hz.TVT_input.iloc[anchor_index]))


def _window_indices(well: Well) -> np.ndarray:
    start = well.anchor_index - VISIBLE_ROWS + 1
    return np.arange(start, start + RAW_ROWS, dtype=np.int64)


def _take(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.full(len(indices), np.nan, dtype=np.float32)
    valid = (indices >= 0) & (indices < len(array))
    result[valid] = np.asarray(array, dtype=np.float32)[indices[valid]]
    return result


def _pad(values: np.ndarray) -> np.ndarray:
    result = np.full(PADDED_ROWS, np.nan, dtype=np.float32)
    result[: min(len(values), PADDED_ROWS)] = values[:PADDED_ROWS]
    return result


def _bin_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    x = _pad(values).reshape(WIDTH, 32)
    valid = np.isfinite(x)
    count = valid.sum(1)
    safe = np.where(valid, x, 0.0)
    mean = safe.sum(1) / np.maximum(count, 1)
    centered = np.where(valid, x - mean[:, None], 0.0)
    std = np.sqrt((centered * centered).sum(1) / np.maximum(count, 1))
    first = np.argmax(valid, axis=1)
    last = 31 - np.argmax(valid[:, ::-1], axis=1)
    rows = np.arange(WIDTH)
    delta = np.where(count > 1, x[rows, last] - x[rows, first], 0.0)
    u = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    out = {"gr": mean, "gr_isnan_rate": 1.0 - count / 32.0, "gr_std": std,
           "gr_first_last_delta": delta}
    coeff = np.zeros((WIDTH, 3), np.float32)
    rmse = np.zeros(WIDTH, np.float32)
    for j in range(WIDTH):
        mask = valid[j]
        if mask.sum() >= 3:
            coeff[j] = np.polyfit(u[mask], x[j, mask], 2).astype(np.float32)
            fit = np.polyval(coeff[j], u[mask])
            rmse[j] = np.sqrt(np.mean((fit - x[j, mask]) ** 2))
    out.update(gr_slope=coeff[:, 1], gr_quadratic_a=coeff[:, 0],
               gr_quadratic_b=coeff[:, 1], gr_quadratic_c=coeff[:, 2],
               gr_quadratic_rmse=rmse)
    return {k: np.nan_to_num(v).astype(np.float32) for k, v in out.items()}


def _bin_last(values: np.ndarray) -> np.ndarray:
    x = _pad(values).reshape(WIDTH, 32)
    result = np.full(WIDTH, np.nan, np.float32)
    for j in range(WIDTH):
        finite = x[j, np.isfinite(x[j])]
        if len(finite):
            result[j] = finite[-1]
    return result


def calibrated_typewell_gr(well: Well, state_tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tw_tvt = well.typewell.TVT.to_numpy(float)
    tw_gr = well.typewell.GR.to_numpy(float)
    original = np.interp(state_tvt, tw_tvt, np.nan_to_num(tw_gr, nan=np.nanmedian(tw_gr)))
    prefix = well.horizontal.iloc[: well.anchor_index + 1]
    mask = np.isfinite(prefix.TVT_input) & np.isfinite(prefix.GR)
    if mask.sum() < 32:
        return original.astype(np.float32), ~np.isfinite(original)
    tvt = prefix.loc[mask, "TVT_input"].to_numpy(float)
    gr = prefix.loc[mask, "GR"].to_numpy(float)
    bins = np.round(tvt * 2.0) / 2.0
    grouped = pd.DataFrame({"tvt": bins, "gr": gr}).groupby("tvt", as_index=False).gr.mean()
    visible = np.interp(state_tvt, grouped.tvt, grouped.gr, left=np.nan, right=np.nan)
    support = (state_tvt >= grouped.tvt.min()) & (state_tvt <= grouped.tvt.max())
    # INFERRED: the writeup specifies blending but not its coefficient.
    calibrated = original.copy()
    calibrated[support] = .5 * original[support] + .5 * visible[support]
    return calibrated.astype(np.float32), ~np.isfinite(calibrated)


def _repeat_row(x: np.ndarray) -> np.ndarray:
    return np.broadcast_to(x[None, :], (HEIGHT, WIDTH)).copy().astype(np.float32)


def _repeat_col(x: np.ndarray) -> np.ndarray:
    return np.broadcast_to(x[:, None], (HEIGHT, WIDTH)).copy().astype(np.float32)


def _norm_gr(x: np.ndarray) -> np.ndarray:
    # INFERRED shared physical normalization, intentionally not per-panel.
    return np.clip((x - 80.0) / 40.0, -4.0, 4.0).astype(np.float32)


def build_panel(
    well: Well,
    channels: tuple[str, ...],
    auxiliary: Mapping[str, np.ndarray] | None = None,
) -> Panel:
    auxiliary = auxiliary or {}
    indices = _window_indices(well)
    hz = well.horizontal
    gr_raw = _take(hz.GR.to_numpy(float), indices)
    z_raw = _take(hz.Z.to_numpy(float), indices)
    input_raw = _take(hz.TVT_input.to_numpy(float), indices)
    truth_raw = _take(hz.TVT.to_numpy(float), indices) if "TVT" in hz else np.full(RAW_ROWS, np.nan)
    stats = _bin_stats(gr_raw)
    seen = _bin_last(input_raw) - well.anchor_tvt
    z_diff = _bin_last(z_raw) - float(hz.Z.iloc[well.anchor_index])
    state_rel = np.asarray(STATE_AXIS, np.float32)
    state_tvt = state_rel + well.anchor_tvt
    tw_gr, tw_nan = calibrated_typewell_gr(well, state_tvt)
    gr_grid = _repeat_row(stats["gr"])
    tw_grid = _repeat_col(tw_gr)
    seen_grid = _repeat_row(np.nan_to_num(seen))
    state_grid = _repeat_col(state_rel)
    base: dict[str, np.ndarray] = {
        "tw_gr": _norm_gr(tw_grid),
        "gr": _norm_gr(gr_grid),
        "gr_abs_diff": np.abs(tw_grid - gr_grid) / 40.0,
        "tw_gr_is_nan": _repeat_col(tw_nan.astype(np.float32)),
        "tw_tvt_rel": state_grid / 100.0,
        "seen_tvt_rel": seen_grid / 100.0,
        "tw_seen_tvt_abs_diff": np.abs(state_grid - seen_grid) / 100.0,
        "z_diff": _repeat_row(np.nan_to_num(z_diff)) / 100.0,
    }
    for key, values in stats.items():
        if key != "gr":
            scale = 40.0 if key not in ("gr_isnan_rate",) else 1.0
            base[key] = _repeat_row(values / scale)
    for key, value in auxiliary.items():
        arr = np.asarray(value, np.float32)
        if arr.shape == (WIDTH,):
            arr = _repeat_row(arr)
        if arr.shape != (HEIGHT, WIDTH):
            raise ValueError(f"{well.well}: auxiliary {key} has shape {arr.shape}")
        base[key] = arr
    missing = [name for name in channels if name not in base]
    if missing:
        raise KeyError(f"{well.well}: missing channels {missing}")
    values = np.stack([np.nan_to_num(base[name]) for name in channels]).astype(np.float32)
    truth_bin = _bin_last(truth_raw)
    valid = np.isfinite(truth_bin)
    distance = state_tvt[:, None] - truth_bin[None, :]
    target = np.exp(-np.abs(distance) / 1.25).astype(np.float32)
    target[:, ~valid] = 0.0
    target /= np.maximum(target.sum(0, keepdims=True), 1e-12)
    target_mask = valid & (truth_bin >= state_tvt.min()) & (truth_bin <= state_tvt.max())
    gr_gap = np.abs(tw_grid - gr_grid).astype(np.float32)
    column_rows = indices[0] + np.minimum(np.arange(WIDTH) * 32 + 31, RAW_ROWS - 1)
    eval_mask = target_mask & (column_rows > well.anchor_index) & (column_rows < len(hz))
    raw_eval_mask = np.isfinite(truth_raw) & (indices > well.anchor_index)
    return Panel(well.well, values, target, target_mask, eval_mask, gr_gap, state_tvt.astype(np.float32),
                 column_rows, truth_bin.astype(np.float32), truth_raw.astype(np.float32),
                 raw_eval_mask, well.anchor_tvt)


def simulate_z_shift(well: Well, donor: Well, rng: np.random.Generator) -> Well:
    """0719 diff-block bootstrap plus the disclosed TVT_bias_v2 noise profile."""
    hz = well.horizontal.copy()
    # Bootstrap only the donor's post-anchor lateral. Sampling the full well
    # leaks the high-curvature build section into the simulated target region
    # and creates physically impossible hundred-foot excursions.
    donor_tvt = donor.horizontal.TVT.to_numpy(float)[donor.anchor_index:]
    diffs = np.diff(donor_tvt)
    need = len(hz)
    generated: list[np.ndarray] = []
    while sum(map(len, generated)) < need - 1:
        length = int(rng.choice((1024, 2048, 4096), p=(.45, .40, .15)))
        if len(diffs) <= length:
            block = np.resize(diffs, length)
        else:
            start = int(rng.integers(0, len(diffs) - length))
            block = diffs[start:start + length].copy()
        block *= rng.uniform(.9, 1.1)
        generated.append(block)
    increments = np.concatenate(generated)[:need - 1]
    # rule_reverse_prob=.5 applies to the assembled rule. The disclosed
    # diff_block_reverse_prob is zero, so individual blocks are never reversed.
    if rng.random() < .5:
        increments = increments[::-1]
    if len(increments) > 32:
        increments = gaussian_filter1d(increments, sigma=16 / 3)
    path = np.r_[0.0, np.cumsum(increments)]
    path += well.anchor_tvt - path[well.anchor_index]

    # Active 0719 TVT_bias_v2: windows=(1536,256,2), weights=(.68,.20,.12),
    # target std=.95. Window-to-sigma conversion is implementation-inferred.
    bias = np.zeros(need, np.float64)
    for window, weight in zip((1536, 256, 2), (.68, .20, .12)):
        component = gaussian_filter1d(rng.standard_normal(need), sigma=max(1.0, window / 6))
        component -= component.mean()
        component /= max(component.std(), 1e-6)
        bias += weight * component
    bias *= .95 / max(bias.std(), 1e-6)
    path += np.clip(bias, -36.0, 36.0)
    path += well.anchor_tvt - path[well.anchor_index]
    surface = hz.TVT.to_numpy(float) + hz.Z.to_numpy(float)
    # Faults are offsets in the geological surface S=TVT+Z, not shifts that
    # are immediately cancelled by recomputing Z.
    if rng.random() < .05:
        for _ in range(int(rng.integers(1, 4))):
            at = int(rng.integers(max(1, well.anchor_index), need))
            surface[at:] += rng.normal(0, 1.75)
    hz["TVT"] = path
    hz["Z"] = surface - path
    visible = np.arange(need) <= well.anchor_index
    hz["TVT_input"] = np.where(visible, path, np.nan)
    tw_tvt = well.typewell.TVT.to_numpy(float)
    tw_gr = well.typewell.GR.to_numpy(float)
    matched = np.interp(path, tw_tvt, np.nan_to_num(tw_gr, nan=np.nanmedian(tw_gr)))
    low, high = ((5.6, 8.4), (8.4, 12.2), (12.2, 17.5))[rng.choice(3, p=(.24, .51, .25))]
    target_std = rng.uniform(low, high)
    residual = (
        .48 * gaussian_filter1d(rng.standard_t(6, need), sigma=1024 / 6)
        + .25 * gaussian_filter1d(rng.standard_t(6, need), sigma=256 / 6)
        + .20 * gaussian_filter1d(rng.standard_t(6, need), sigma=32 / 6)
        + .07 * rng.standard_t(6, need)
    )
    residual -= residual.mean()
    residual *= target_std / max(residual.std(), 1e-6)
    hz["GR"] = matched + np.clip(residual, -45.0, 45.0) + rng.normal(0, 2.0)
    return Well(well.well, hz, well.typewell.copy(), well.anchor_index, float(path[well.anchor_index]))


def warp_auxiliary_for_simulation(original: Well, simulated: Well,
                                  auxiliary: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep cached PF/XY evidence in the coordinate system of a simulated path."""
    if not auxiliary:
        return {}
    indices = _window_indices(original)
    original_tvt = _bin_last(_take(original.horizontal.TVT.to_numpy(float), indices))
    simulated_tvt = _bin_last(_take(simulated.horizontal.TVT.to_numpy(float), indices))
    delta = np.nan_to_num(simulated_tvt - original_tvt).astype(np.float32)
    rows = np.arange(HEIGHT, dtype=np.float32)
    shifted: dict[str, np.ndarray] = {}
    state_maps = {
        "pf_particle_density_prob", "pf_prob", "pf_prob_ffbsi",
        "pf_tvt_ffbsi", "geo_tvt_abs_diff",
    }
    for name, raw in auxiliary.items():
        value = np.asarray(raw, dtype=np.float32)
        if name in state_maps:
            output = np.empty_like(value)
            for column in range(WIDTH):
                source = rows - delta[column] / STATE_STEP
                output[:, column] = np.interp(
                    source, rows, value[:, column], left=0.0, right=0.0
                )
            if name in {"pf_particle_density_prob", "pf_prob", "pf_prob_ffbsi"}:
                output /= np.maximum(output.sum(0, keepdims=True), 1e-8)
            shifted[name] = output
        elif name == "geo_tvt_rel":
            shifted[name] = value + delta[None] / 100.0
        elif name == "geo_tvt_diff":
            delta_increment = np.diff(delta, prepend=0.0)
            shifted[name] = value + delta_increment[None] / 100.0
        else:
            shifted[name] = value.copy()
    return shifted


def augment_panel(panel: Panel, channels: tuple[str, ...], rng: np.random.Generator,
                  version_name: str) -> Panel:
    cfg = _AUGMENTATION[version_name]
    values = panel.values.copy()
    if rng.random() < .25:
        target, mask = panel.target.copy(), panel.target_mask.copy()
        eval_mask, gr_gap = panel.eval_mask.copy(), panel.gr_gap.copy()
        # Preserve the 1024-row visible head. Reverse only a physical segment
        # whose new first state remains within the disclosed +/-10 ft bound.
        future_start = VISIBLE_ROWS // 32
        candidates = np.flatnonzero(
            mask & (np.arange(WIDTH) >= future_start)
            & (np.abs(panel.truth_tvt - panel.anchor_tvt) <= 10.0)
        )
        if len(candidates):
            stop = int(candidates[-1]) + 1
            if stop > future_start + 1:
                sl = slice(future_start, stop)
                values[:, :, sl] = values[:, :, sl][:, :, ::-1]
                target[:, sl] = target[:, sl][:, ::-1]
                mask[sl] = mask[sl][::-1]
                eval_mask[sl] = eval_mask[sl][::-1]
                gr_gap[:, sl] = gr_gap[:, sl][:, ::-1]
    else:
        target, mask, eval_mask, gr_gap = panel.target.copy(), panel.target_mask.copy(), panel.eval_mask.copy(), panel.gr_gap.copy()
    if rng.random() < .4:
        ratio = rng.uniform(.7, 1.3)
        pivot = VISIBLE_ROWS / 32 - 1
        destination = np.arange(WIDTH)
        source = np.clip(pivot + (destination - pivot) / ratio, 0, WIDTH - 1)
        left = np.floor(source).astype(int)
        right = np.minimum(left + 1, WIDTH - 1)
        fraction = source - left
        values = values[:, :, left] * (1 - fraction)[None, None] + values[:, :, right] * fraction[None, None]
        target = target[:, left] * (1 - fraction)[None] + target[:, right] * fraction[None]
        target /= np.maximum(target.sum(0, keepdims=True), 1e-12)
        gr_gap = gr_gap[:, left] * (1 - fraction)[None] + gr_gap[:, right] * fraction[None]
        mask, eval_mask = mask[np.rint(source).astype(int)], eval_mask[np.rint(source).astype(int)]
    # GR_noise_shift=copy shifts the within-well residual while retaining the
    # truth-aligned typewell component. It is active only for released 0719.
    if rng.random() < cfg["gr_noise"] and {"tw_gr", "gr", "gr_abs_diff"}.issubset(channels):
        tw_index, gr_index, gap_index = map(channels.index, ("tw_gr", "gr", "gr_abs_diff"))
        matched = (target * values[tw_index]).sum(axis=0)
        horizontal = values[gr_index, 0].copy()
        residual = horizontal - matched
        shift = int(rng.integers(1, max(2, WIDTH)))
        shifted = np.roll(residual, shift)
        transformed = matched + shifted
        values[gr_index] = transformed[None]
        values[gap_index] = np.abs(values[tw_index] - transformed[None])

    # The writeup applies GR'=a*GR+b to typewell GR, not both curves.
    if rng.random() < .6 and {"tw_gr", "gr", "gr_abs_diff"}.issubset(channels):
        gain, offset = rng.normal(1, cfg["a_std"]), rng.normal(0, 2 / 40)
        tw_index, gr_index, gap_index = map(channels.index, ("tw_gr", "gr", "gr_abs_diff"))
        values[tw_index] = gain * values[tw_index] + offset
        values[gap_index] = np.abs(values[tw_index] - values[gr_index])
    if rng.random() < .2:
        cut = int(rng.uniform(0, .5) * WIDTH)
        if cut:
            values[:, :, -cut:] = 0
            target[:, -cut:] = 0
            mask[-cut:] = False
            eval_mask[-cut:] = False
    pf_names = {"pf_prob", "pf_tvt_ffbsi", "pf_particle_density_prob", "pf_prob_ffbsi"}
    if rng.random() < cfg["pf_rotate"]:
        shift = rng.uniform(-20.0, 20.0) / .5
        ratio = rng.uniform(.75, 1.25)
        rows = np.arange(HEIGHT, dtype=np.float32)
        for index, name in enumerate(channels):
            if name in pf_names:
                source = (rows - (HEIGHT - 1) / 2) / ratio + (HEIGHT - 1) / 2 - shift
                values[index] = np.stack([
                    np.interp(source, rows, values[index, :, column], left=0.0, right=0.0)
                    for column in range(WIDTH)
                ], axis=1)
    if rng.random() < cfg["pf_corrupt"]:
        for index, name in enumerate(channels):
            if name in {"pf_prob", "pf_particle_density_prob", "pf_prob_ffbsi"}:
                temperature = rng.uniform(.85, 1.35)
                probability = np.maximum(values[index], 1e-8) ** (1 / temperature)
                probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-8)
                dropout = rng.random(WIDTH) < .02
                probability[:, dropout] = 0.0
                values[index] = probability

    if rng.random() < cfg["tw_jitter"] and {"tw_gr", "gr", "gr_abs_diff"}.issubset(channels):
        tw_index, gr_index, gap_index = map(channels.index, ("tw_gr", "gr", "gr_abs_diff"))
        gain = np.clip(rng.normal(1.0, .04), .9, 1.1)
        offset = rng.normal(0, 2 / 40)
        values[tw_index] = gain * values[tw_index] + offset
        values[gr_index] = gain * values[gr_index] + offset
        values[gap_index] = np.abs(values[tw_index] - values[gr_index])

    if rng.random() < cfg["tw_drift"]:
        shift = rng.normal(0, 1.0) / .5
        rows = np.arange(HEIGHT, dtype=np.float32)
        source = rows - shift
        for name in ("tw_gr", "tw_gr_is_nan", "tw_tvt_rel", "tw_seen_tvt_abs_diff"):
            if name in channels:
                index = channels.index(name)
                values[index] = np.stack([
                    np.interp(source, rows, values[index, :, column], left=0.0, right=0.0)
                    for column in range(WIDTH)
                ], axis=1)
        if {"tw_gr", "gr", "gr_abs_diff"}.issubset(channels):
            values[channels.index("gr_abs_diff")] = np.abs(
                values[channels.index("tw_gr")] - values[channels.index("gr")]
            )
    if rng.random() < .7:
        for i, name in enumerate(channels):
            probability = cfg["pf_mask"] if name in pf_names else .1
            if rng.random() < probability:
                values[i] = 0
    if rng.random() < .1:
        # Sequential masking is along the typewell/state axis, not MD.
        height = max(1, int(HEIGHT * .3))
        start = int(rng.integers(0, HEIGHT - height + 1))
        values[:, start:start + height, :] = 0
    if "gr_abs_diff" in channels:
        gr_gap = np.maximum(values[channels.index("gr_abs_diff")], 0.0) * 40.0
    return Panel(panel.well, values, target, mask, eval_mask, gr_gap, panel.state_tvt,
                 panel.column_rows, panel.truth_tvt, panel.raw_truth_tvt,
                 panel.raw_eval_mask, panel.anchor_tvt)


class AlignmentDataset(Dataset):
    def __init__(self, data_root: Path, wells: list[str], split: str, channels: tuple[str, ...],
                 auxiliary_store: Mapping[str, Mapping[str, np.ndarray]] | None = None,
                 training: bool = False, seed: int = 0, z_shift_prob: float = 0.0,
                 version_name: str = "0719_V1"):
        self.data_root, self.wells, self.split, self.channels = data_root, wells, split, channels
        self.auxiliary_store = auxiliary_store or {}
        self.training, self.seed, self.z_shift_prob = training, seed, z_shift_prob
        self.version_name = version_name
        self._loaded: dict[str, Well] = {}
        self._calls = 0

    def __len__(self) -> int:
        return len(self.wells)

    def _well(self, name: str) -> Well:
        if name not in self._loaded:
            self._loaded[name] = load_well(self.data_root, name, self.split)
        return self._loaded[name]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        self._calls += 1
        rng = np.random.default_rng((torch.initial_seed() + self.seed * 1_000_003 + index * 1009 + self._calls) % (2**63 - 1))
        well = self._well(self.wells[index])
        auxiliary = self.auxiliary_store.get(well.well, {})
        if self.training and rng.random() < self.z_shift_prob:
            if len(self.wells) < 2:
                raise ValueError("z_shift requires at least two wells")
            donor_index = (index + int(rng.integers(1, len(self.wells)))) % len(self.wells)
            donor = self._well(self.wells[donor_index])
            original = well
            well = simulate_z_shift(original, donor, rng)
            auxiliary = warp_auxiliary_for_simulation(original, well, auxiliary)
        panel = build_panel(well, self.channels, auxiliary)
        if self.training:
            panel = augment_panel(panel, self.channels, rng, self.version_name)
        return {
            "well": panel.well,
            "x": torch.from_numpy(panel.values).float(),
            "target": torch.from_numpy(panel.target),
            "mask": torch.from_numpy(panel.target_mask),
            "eval_mask": torch.from_numpy(panel.eval_mask),
            "gr_gap": torch.from_numpy(panel.gr_gap),
            "state_tvt": torch.from_numpy(panel.state_tvt),
            "truth_tvt": torch.from_numpy(panel.truth_tvt),
            "raw_truth_tvt": torch.from_numpy(panel.raw_truth_tvt),
            "raw_eval_mask": torch.from_numpy(panel.raw_eval_mask),
            "column_rows": torch.from_numpy(panel.column_rows),
            "anchor_tvt": torch.tensor(panel.anchor_tvt, dtype=torch.float32),
        }
