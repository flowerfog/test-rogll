from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import PanelAugmenter, RawWellAugmenter, build_trend_pool
from .config import ReproductionConfig, VersionRecipe
from .data import append_pf_channels, append_xy_channels, build_base_panel, load_record, normalize_panel
from .pf import MultiProfileParticleFilter


class AlignmentDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        wells: list[str],
        recipe: VersionRecipe,
        config: ReproductionConfig = ReproductionConfig(),
        xy_prior=None,
        training: bool = False,
        seed: int = 0,
        cache_root: str | Path | None = None,
        population_stats=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.wells = list(wells)
        self.recipe = recipe
        self.config = config
        self.xy_prior = xy_prior
        self.training = training
        self.seed = seed
        self.cache_root = Path(cache_root) if cache_root else None
        if population_stats is None:
            raise ValueError("dataset requires fold-clean population statistics")
        self.population_stats = population_stats
        self.trend_pool = build_trend_pool(self.data_root, split, self.wells) if training else []
        self.raw_augmenter = RawWellAugmenter(seed, recipe.augmentation_profile, self.trend_pool)
        self.panel_augmenter = PanelAugmenter(seed + 1)
        if recipe.use_xy and xy_prior is None:
            raise ValueError("XY recipe requires a fitted XYSurfacePrior")

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.raw_augmenter = RawWellAugmenter(
            self.seed, self.recipe.augmentation_profile, self.trend_pool
        )
        self.panel_augmenter = PanelAugmenter(self.seed + 1)

    def __len__(self) -> int:
        return len(self.wells)

    def _cache_path(self, well: str) -> Path | None:
        if self.cache_root is None or self.training:
            return None
        return self.cache_root / self.recipe.name / self.split / f"{well}.npz"

    def __getitem__(self, index: int):
        well = self.wells[index]
        record = load_record(self.data_root, self.split, well)
        if self.training:
            record = self.raw_augmenter(record)
        sample = build_base_panel(
            record,
            self.config.grid,
            self.config.loss.target_sigma_ft,
            population_stats=self.population_stats,
            apply_channel_normalization=False,
        )
        if self.recipe.use_pf:
            result = MultiProfileParticleFilter(self.config.pf, self.seed + index).run(sample)
            sample = append_pf_channels(sample, result.density, result.ffbsi_offset)
        if self.recipe.use_xy:
            sample = append_xy_channels(sample, self.xy_prior.predict(record, sample))
        sample = sample.select(self.recipe.channels)
        sample = normalize_panel(sample, self.population_stats)
        if self.training:
            sample = self.panel_augmenter(sample)
        selected = sample.selected_rows
        valid_rows = selected >= 0
        row_truth = np.full(len(selected), np.nan, dtype=np.float32)
        row_z = np.full(len(selected), np.nan, dtype=np.float32)
        row_eval_mask = np.zeros(len(selected), dtype=bool)
        if record.truth is not None:
            row_truth[valid_rows] = record.truth[selected[valid_rows]]
        row_z[valid_rows] = record.z[selected[valid_rows]]
        row_eval_mask[valid_rows] = selected[valid_rows] > record.anchor_index
        return {
            "well": well,
            "values": torch.from_numpy(sample.values),
            "target_probability": torch.from_numpy(sample.target_probability),
            "target_offset": torch.from_numpy(sample.target_offset),
            "target_mask": torch.from_numpy(sample.target_mask),
            "gr_gap": torch.from_numpy(sample.gr_gap),
            "state_offsets": torch.from_numpy(sample.state_offsets),
            "anchor_tvt": sample.anchor_tvt,
            "anchor_index": sample.anchor_index,
            "selected_rows": sample.selected_rows,
            "row_truth": torch.from_numpy(row_truth),
            "row_z": torch.from_numpy(row_z),
            "row_eval_mask": torch.from_numpy(row_eval_mask),
        }
