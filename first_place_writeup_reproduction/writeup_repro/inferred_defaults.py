from dataclasses import dataclass


@dataclass(frozen=True)
class LossDefaults:
    target_sigma_ft: float = 1.5
    cross_entropy_weight: float = 1.0
    path_huber_weight: float = 0.10
    gr_penalty_weight: float = 0.02
    huber_delta_ft: float = 3.0


@dataclass(frozen=True)
class PFProfile:
    process_sigma_ft: float
    likelihood_scale_gr: float
    jump_probability: float
    jump_sigma_ft: float
    weight: float


@dataclass(frozen=True)
class PFDefaults:
    particles: int = 512
    ffbsi_paths: int = 32
    update_rows: int = 64
    resample_ess_fraction: float = 0.55
    student_df: float = 4.0
    profiles: tuple[PFProfile, ...] = (
        PFProfile(0.8, 8.0, 0.002, 18.0, 0.25),
        PFProfile(1.5, 12.0, 0.005, 28.0, 0.30),
        PFProfile(2.5, 18.0, 0.010, 42.0, 0.25),
        PFProfile(4.0, 26.0, 0.020, 65.0, 0.20),
    )


@dataclass(frozen=True)
class XYDefaults:
    neighbors: int = 32
    anisotropy: float = 2.0
    ridge: float = 1e-4
    robust_iterations: int = 3
    distance_floor: float = 100.0
    quality_quantile: float = 0.95


@dataclass(frozen=True)
class TrainDefaults:
    epochs: int = 220
    batch_size: int = 4
    accumulation_steps: int = 4
    learning_rate: float = 2e-4
    min_learning_rate: float = 5e-7
    weight_decay: float = 0.02
    ema_decay: float = 0.99
    gradient_clip: float = 10.0
    workers: int = 4
    patience: int = 35
