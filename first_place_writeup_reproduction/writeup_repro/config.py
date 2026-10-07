from dataclasses import dataclass

from .inferred_defaults import LossDefaults, PFDefaults, TrainDefaults, XYDefaults


@dataclass(frozen=True)
class GridConfig:
    horizontal_bins: int = 345
    horizontal_downsample: int = 32
    visible_rows: int = 1024
    target_rows: int = 10_000
    state_bins: int = 400
    state_radius_ft: float = 100.0
    state_step_ft: float = 0.5

    @property
    def total_rows(self) -> int:
        return self.visible_rows + self.target_rows

    @property
    def state_offsets(self):
        import numpy as np

        return (
            -self.state_radius_ft
            + self.state_step_ft * (np.arange(self.state_bins, dtype=np.float32) + 0.5)
        )


@dataclass(frozen=True)
class VersionRecipe:
    name: str
    use_pf: bool
    use_xy: bool
    augmentation_profile: str
    general_weight: float
    no_xy_weight: float


VERSION_RECIPES = {
    "0719_V1": VersionRecipe("0719_V1", False, False, "base_a", 0.07, 0.40),
    "0729_V3": VersionRecipe("0729_V3", True, False, "pf", 0.00, 0.20),
    "0801_V1": VersionRecipe("0801_V1", False, False, "base_b", 0.07, 0.40),
    "0724_V1": VersionRecipe("0724_V1", False, True, "xy_a", 0.28, 0.00),
    "0801_V2": VersionRecipe("0801_V2", False, True, "xy_b", 0.28, 0.00),
    "0803_V2": VersionRecipe("0803_V2", True, True, "xy_pf", 0.28, 0.00),
}


@dataclass(frozen=True)
class ReproductionConfig:
    grid: GridConfig = GridConfig()
    loss: LossDefaults = LossDefaults()
    pf: PFDefaults = PFDefaults()
    xy: XYDefaults = XYDefaults()
    train: TrainDefaults = TrainDefaults()
