import numpy as np

from writeup_repro.augment import PanelAugmenter
from writeup_repro.config import GridConfig
from writeup_repro.data import PanelSample


def test_sequential_mask_never_uses_invalid_bounds():
    grid = GridConfig(horizontal_bins=8, state_bins=5, state_radius_ft=1.25, state_step_ft=0.5)
    sample = PanelSample(
        "w", np.ones((3, 8, 5), np.float32), ("a", "b", "c"),
        np.zeros((8, 5), np.float32), np.zeros(8, np.float32), np.zeros(8, bool),
        np.zeros((8, 5), np.float32), grid.state_offsets, np.repeat(np.arange(8), 32),
        np.arange(256), np.arange(256), 0.0, 31,
    )
    augmenter = PanelAugmenter(9)
    for _ in range(1000):
        output = augmenter(sample)
        assert output.values.shape == sample.values.shape
        assert np.isfinite(output.values).all()
