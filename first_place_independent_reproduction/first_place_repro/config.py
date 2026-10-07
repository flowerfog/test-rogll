from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    horizontal_bins: int = 345
    rows_per_bin: int = 32
    visible_rows: int = 1024
    future_rows: int = 10_000
    state_bins: int = 400
    state_step_ft: float = 0.5
    state_radius_ft: float = 100.0

    @property
    def state_offsets(self) -> np.ndarray:
        return (-99.75 + 0.5 * np.arange(400, dtype=np.float32))

    @property
    def raster_rows(self) -> int:
        return self.horizontal_bins * self.rows_per_bin

    @property
    def horizontal_downsample(self) -> int:
        return self.rows_per_bin

    @property
    def target_rows(self) -> int:
        return self.future_rows


BASE_CHANNELS = (
    "tw_gr", "hw_gr_mean", "tw_hw_gr_abs_gap", "hw_gr_nan", "hw_gr_std",
    "hw_gr_last_first", "hw_gr_slope", "hw_gr_quadratic_a",
    "hw_gr_quadratic_b", "hw_gr_quadratic_c", "hw_gr_quadratic_rmse",
    "tw_gr_nan", "tw_tvt_rel", "visible_tvt_rel", "tw_visible_tvt_abs_gap",
    "z_diff",
)


@dataclass(frozen=True)
class Version:
    name: str
    channels: tuple[str, ...]
    epochs: int
    dropout: float
    z_shift_probability: float
    sg_window: int
    sg_blend: float
    general_weight: float
    no_xy_weight: float

    @property
    def use_pf(self) -> bool:
        return any(name.startswith("pf_") for name in self.channels)

    @property
    def use_xy(self) -> bool:
        return any(name.startswith("xy_") for name in self.channels)

    @property
    def augmentation_profile(self) -> str:
        return self.name


VERSIONS = {
    "0719_V1": Version("0719_V1", BASE_CHANNELS, 300, .05, .75, 1025, .6828376753426914, .07, .40),
    "0729_V3": Version("0729_V3", BASE_CHANNELS + ("pf_particle_density_prob", "pf_tvt_abs_gap"), 200, .05, .75, 1025, .6828376753426914, 0., .20),
    "0801_V1": Version("0801_V1", BASE_CHANNELS, 300, 0., .75, 577, .726196, .07, .40),
    "0724_V1": Version("0724_V1", BASE_CHANNELS + ("xy_tvt_diff", "xy_tvt_rel", "xy_tvt_abs_gap"), 300, .05, .75, 1025, .6828376753426914, .28, 0.),
    "0801_V2": Version("0801_V2", BASE_CHANNELS + ("xy_tvt_diff", "xy_tvt_rel", "xy_tvt_abs_gap"), 300, 0., .75, 577, .726196, .28, 0.),
    "0803_V2": Version("0803_V2", BASE_CHANNELS + ("xy_tvt_diff", "pf_particle_density_prob", "pf_prob_ffbsi", "xy_tvt_abs_gap", "xy_tvt_rel"), 200, 0., .85, 577, .726196, .28, 0.),
    # Produced in the disclosed lineage but assigned zero final weight.
    "0802_V2": Version("0802_V2", BASE_CHANNELS + ("xy_tvt_diff", "pf_particle_density_prob", "pf_prob_ffbsi"), 200, .05, .75, 577, .726196, 0., 0.),
}


@dataclass(frozen=True)
class Training:
    batch_size: int = 16
    validation_batch_size: int = 32
    learning_rate: float = 2e-4
    min_learning_rate: float = 5e-7
    weight_decay: float = .02
    ema_decay: float = .99
    gradient_clip: float = 10.
    minimum_epochs: int = 170
    patience: int = 30
    target_sigma_ft: float = 1.25
    target_mean_ft: float = 1.591691
    target_std_ft: float = 15.853182
    ce_weight: float = 1 / 15
    regression_weight: float = 1 / 60
    gr_weight: float = 1 / 60


GRID = Grid()
TRAINING = Training()
REPEATS = (33, 34, 35)
FOLDS = (1, 2, 3, 4, 5)

# Compatibility names used across the independent modules.
GridConfig = Grid
VersionRecipe = Version
VERSION_RECIPES = VERSIONS


@dataclass(frozen=True)
class ReproductionConfig:
    grid: Grid = GRID
    train: Training = TRAINING

    @property
    def loss(self):
        from .inferred_defaults import LossDefaults
        return LossDefaults()

    @property
    def pf(self):
        from .inferred_defaults import PFDefaults
        return PFDefaults()

    @property
    def xy(self):
        from .inferred_defaults import XYDefaults
        return XYDefaults()
