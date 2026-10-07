from pathlib import Path

import numpy as np
import pandas as pd

from writeup_repro.config import GridConfig, VERSION_RECIPES
from writeup_repro.data import BASE_CHANNELS, build_base_panel, list_wells, load_record
from writeup_repro.ensemble import EnsembleRouter
from writeup_repro.folds import make_repeated_folds
from writeup_repro.loss import AlignmentLoss


ROOT = Path(__file__).resolve().parents[2]


def test_grid_contract():
    grid = GridConfig()
    assert (grid.horizontal_bins, grid.state_bins) == (345, 400)
    assert np.allclose(np.diff(grid.state_offsets), 0.5)
    assert np.isclose(grid.state_offsets.mean(), 0.0)


def test_real_well_panel_contract():
    well = list_wells(ROOT, "train")[0]
    record = load_record(ROOT, "train", well)
    sample = build_base_panel(record)
    assert sample.values.shape == (len(BASE_CHANNELS), 345, 400)
    assert sample.channels == BASE_CHANNELS
    assert np.isfinite(sample.values).all()
    assert np.isfinite(sample.target_probability).all()
    assert np.allclose(sample.target_probability[sample.target_mask].sum(axis=1), 1.0, atol=1e-5)
    assert np.isclose(sample.anchor_tvt, record.tvt_input[record.anchor_index])


def test_loss_prefers_perfect_alignment():
    import torch

    grid = GridConfig()
    target = torch.zeros(1, 3, grid.state_bins)
    indices = torch.tensor([[180, 200, 220]])
    target.scatter_(2, indices[..., None], 1.0)
    state = torch.from_numpy(grid.state_offsets)[None]
    truth = state[0, indices]
    mask = torch.ones(1, 3, dtype=torch.bool)
    gap = torch.zeros_like(target)
    criterion = AlignmentLoss()
    perfect = torch.full_like(target, -8.0).scatter(2, indices[..., None], 8.0)
    shifted = torch.full_like(target, -8.0).scatter(2, (indices + 20)[..., None], 8.0)
    good, _ = criterion(perfect, target, truth, mask, state, gap)
    bad, _ = criterion(shifted, target, truth, mask, state, gap)
    assert good < bad


def test_repeated_fold_coverage():
    rng = np.random.default_rng(4)
    metadata = pd.DataFrame(
        {
            "well": [f"w{i}" for i in range(100)],
            "x": rng.normal(size=100),
            "y": rng.normal(size=100),
            "rows": rng.integers(1000, 8000, 100),
            "target_rows": rng.integers(500, 6000, 100),
        }
    )
    folds = make_repeated_folds(metadata)
    for repeat in range(3):
        assert set(folds[f"repeat_{repeat}"]) == {1, 2, 3, 4, 5}


def test_version_and_weight_contract():
    assert set(VERSION_RECIPES) == {
        "0719_V1", "0729_V3", "0801_V1", "0724_V1", "0801_V2", "0803_V2"
    }
    router = EnsembleRouter()
    assert np.isclose(sum(router.general.values()), 1.0)
    assert np.isclose(sum(router.no_xy.values()), 1.0)
    assert all(not VERSION_RECIPES[name].use_xy for name in router.no_xy)
