from __future__ import annotations

from dataclasses import dataclass


HEIGHT = 400
WIDTH = 345
RECONSTRUCTION_REVISION = "20260807_r2_nan_and_simulation_fix"
VISIBLE_ROWS = 1024
TARGET_ROWS = 10_000
RAW_ROWS = VISIBLE_ROWS + TARGET_ROWS
PADDED_ROWS = WIDTH * 32
STATE_STEP = 0.5
STATE_MIN = -100.0
STATE_AXIS = tuple(STATE_MIN + STATE_STEP * i for i in range(HEIGHT))

BASE = (
    "tw_gr", "gr", "gr_abs_diff", "gr_isnan_rate", "gr_std",
    "gr_first_last_delta", "gr_slope", "gr_quadratic_a",
    "gr_quadratic_b", "gr_quadratic_c", "gr_quadratic_rmse",
    "tw_gr_is_nan", "tw_tvt_rel", "seen_tvt_rel",
    "tw_seen_tvt_abs_diff",
)


@dataclass(frozen=True)
class Version:
    name: str
    channels: tuple[str, ...]
    epochs: int
    dropout: float
    z_shift_prob: float
    sg_window: int
    sg_blend: float
    general_weight: float
    no_xy_weight: float

    @property
    def validation_batch_size(self) -> int:
        # Released log: 0719 uses val=32; all later members use val=16.
        return 32 if self.name == "0719_V1" else 16


VERSIONS = (
    Version("0719_V1", BASE + ("z_diff",), 300, .05, .75, 1025, .682838, .07, .40),
    Version("0724_V1", BASE + ("z_diff", "geo_tvt_diff"), 300, .05, .75, 1025, .682838, .28, 0.0),
    Version("0729_V3", BASE + ("z_diff", "pf_particle_density_prob", "pf_tvt_ffbsi"), 200, .05, .75, 1025, .682838, 0.0, .20),
    Version("0801_V1", BASE + ("z_diff",), 300, 0.0, .75, 577, .726196, .07, .40),
    Version("0801_V2", BASE + ("geo_tvt_diff",), 300, 0.0, .75, 577, .726196, .28, 0.0),
    # This released member is produced although it is not selected by the final table.
    Version("0802_V2", BASE + ("pf_prob", "pf_tvt_ffbsi", "geo_tvt_diff"), 200, .05, .75, 577, .726196, 0.0, 0.0),
    Version("0803_V2", BASE + ("geo_tvt_diff", "pf_particle_density_prob", "pf_prob_ffbsi", "geo_tvt_abs_diff", "geo_tvt_rel"), 200, 0.0, .85, 577, .726196, .28, 0.0),
)

VERSION_BY_NAME = {v.name: v for v in VERSIONS}
FOLDS = (1, 2, 3, 4, 5)
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    val_batch_size: int = 16
    workers: int = 4
    learning_rate: float = 2e-4
    min_learning_rate: float = 5e-7
    weight_decay: float = .02
    ema_decay: float = .99
    target_sigma: float = 1.25
    regression_weight: float = 1 / 60
    alignment_weight: float = 1 / 15
    gr_penalty_weight: float = 1 / 60
    tvt_diff_clip: float = 100.0


@dataclass(frozen=True)
class PFConfig:
    particles: int = 500
    seeds: int = 128
    base_seed: int = 20_265_667
    likelihood_scale: float = 6.0
    update_rows: int = 64
    # INFERRED: the writeup discloses diverse profiles but not their values.
    transition_sigmas: tuple[float, ...] = (.35, .55, .8, 1.1)
    jump_probabilities: tuple[float, ...] = (.0025, .005, .01, .02)
    jump_sigmas: tuple[float, ...] = (4.0, 7.0, 11.0, 16.0)


@dataclass(frozen=True)
class XYConfig:
    neighbor_wells: int = 12
    point_neighbors: int = 112
    idw_power: float = 2.0
    anisotropy_angle_deg: float = 50.0
    anisotropy_ratio: float = 1.3
    train_bin_size: int = 64
    query_bin_size: int = 32
    prefix_bin_size: int = 32
    prefix_support_rows: int = 512
    filter_ds_threshold: float = .15
    ridge_fraction: float = .01
    min_eigen_ratio: float = .05
    max_gradient: float = .25
    tukey_scale: float = .1
    robust_iterations: int = 1
    max_train_points: int = 12_000


TRAIN = TrainConfig()
PF = PFConfig()
XY = XYConfig()
