from dataclasses import dataclass

from .config import TRAINING
from .inferred import INFERRED_PF_PROFILES


@dataclass(frozen=True)
class LossDefaults:
    target_sigma_ft: float = TRAINING.target_sigma_ft
    cross_entropy_weight: float = TRAINING.ce_weight
    path_huber_weight: float = TRAINING.regression_weight
    gr_penalty_weight: float = TRAINING.gr_weight
    huber_delta_ft: float = 1.0
    target_mean_ft: float = TRAINING.target_mean_ft
    target_std_ft: float = TRAINING.target_std_ft


PFProfile = type(INFERRED_PF_PROFILES[0])


@dataclass(frozen=True)
class PFDefaults:
    particles: int = 500
    seeds: int = 256
    ffbsi_paths: int = 64
    update_rows: int = 64
    resample_ess_fraction: float = .30
    student_df: float = 4.0
    profiles: tuple = INFERRED_PF_PROFILES


@dataclass(frozen=True)
class XYDefaults:
    neighbors: int = 12
    point_neighbors: int = 112
    anisotropy: float = 1.3
    anisotropy_angle_deg: float = 50.0
    ridge: float = .01
    robust_iterations: int = 1
    distance_floor: float = 1.0
    quality_quantile: float = .95
    max_gradient: float = .25


@dataclass(frozen=True)
class TrainDefaults:
    epochs: int = 300
    batch_size: int = 16
    accumulation_steps: int = 1
    learning_rate: float = TRAINING.learning_rate
    min_learning_rate: float = TRAINING.min_learning_rate
    weight_decay: float = TRAINING.weight_decay
    ema_decay: float = TRAINING.ema_decay
    gradient_clip: float = TRAINING.gradient_clip
    workers: int = 4
    patience: int = TRAINING.patience
    minimum_epochs: int = TRAINING.minimum_epochs
