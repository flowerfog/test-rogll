import numpy as np
import torch

from first_place_repro.config import BASE_CHANNELS, GRID, VERSION_RECIPES
from first_place_repro.data import WellRecord, build_base_panel, horizontal_bin_features, select_rows
from first_place_repro.loss import AlignmentLoss
from first_place_repro.normalization import PopulationStats


def record() -> WellRecord:
    rows = 1400
    anchor = 1099
    truth = np.linspace(10.0, 35.0, rows, dtype=np.float32)
    tvt_input = np.full(rows, np.nan, dtype=np.float32)
    tvt_input[: anchor + 1] = truth[: anchor + 1]
    z = np.arange(rows, dtype=np.float32) * -0.02
    gr = 80.0 + 10.0 * np.sin(np.arange(rows, dtype=np.float32) / 30.0)
    typewell_tvt = np.linspace(-100, 150, 1001, dtype=np.float32)
    typewell_gr = 80.0 + 10.0 * np.sin(typewell_tvt / 10.0)
    return WellRecord(
        well="synthetic", md=np.arange(rows, dtype=np.float32),
        x=np.zeros(rows, np.float32), y=np.zeros(rows, np.float32), z=z, gr=gr,
        tvt_input=tvt_input, truth=truth,
        typewell_tvt=typewell_tvt, typewell_gr=typewell_gr,
    )


def panel():
    return build_base_panel(
        record(), GRID, population_stats=PopulationStats(0.0, 1.0),
        apply_channel_normalization=True,
    )


def test_channel_order_and_layout():
    sample = panel()
    assert sample.values.shape == (16, 345, 400)
    assert sample.channels == VERSION_RECIPES["0719_V1"].channels
    assert sample.channels == BASE_CHANNELS


def test_supervision_is_suffix_only():
    sample = panel()
    assert not sample.target_mask[:32].any()
    assert sample.target_mask[32:].any()


def test_z_diff_is_horizontal_signal_broadcast_over_states():
    sample = panel()
    z = sample.values[sample.channels.index("z_diff")]
    assert np.allclose(z, z[:, :1])
    assert np.allclose(z[:32], 0.0)
    assert np.any(np.abs(z[32:]) > 0)


def test_short_prefix_is_right_aligned():
    short = record()
    tvt_input = np.full_like(short.tvt_input, np.nan)
    tvt_input[:513] = short.truth[:513]
    short = WellRecord(**{**short.__dict__, "tvt_input": tvt_input})
    sample = build_base_panel(
        short, GRID, population_stats=PopulationStats(0.0, 1.0),
        apply_channel_normalization=True,
    )
    assert np.all(sample.selected_rows[:511] == -1)
    assert sample.selected_rows[511] == 0
    assert sample.selected_rows[1023] == 512


def test_gr_penalty_is_squared_and_clipped():
    sample = panel()
    assert sample.gr_gap.min() >= 0.0
    assert sample.gr_gap.max() <= 25.0


def test_typewell_calibration_uses_only_anchor_neighborhood():
    source = record()
    changed_gr = source.gr.copy()
    far_visible = np.isfinite(source.tvt_input) & (
        np.abs(source.tvt_input - source.anchor_tvt) > 100.0
    )
    changed_gr[far_visible] += 1000.0
    changed = WellRecord(**{**source.__dict__, "gr": changed_gr})
    baseline = build_base_panel(
        source, GRID, population_stats=PopulationStats(0.0, 1.0)
    )
    candidate = build_base_panel(
        changed, GRID, population_stats=PopulationStats(0.0, 1.0)
    )
    index = baseline.channels.index("tw_gr")
    assert np.allclose(baseline.values[index], candidate.values[index])


def test_quadratic_features_require_six_observations():
    source = record()
    gr = source.gr.copy()
    gr[:] = np.nan
    gr[:5] = np.arange(5, dtype=np.float32)
    sparse = WellRecord(**{**source.__dict__, "gr": gr})
    selected, _ = select_rows(sparse, GRID)
    features = horizontal_bin_features(sparse, selected, GRID)
    assert features["qa"][0] == 0.0


def test_loss_ignores_prefix_bins():
    sample = panel()
    logits = torch.zeros(1, 345, 400, requires_grad=True)
    criterion = AlignmentLoss()
    loss, _ = criterion(
        logits,
        torch.from_numpy(sample.target_probability)[None],
        torch.from_numpy(sample.target_offset)[None],
        torch.from_numpy(sample.target_mask)[None],
        torch.from_numpy(sample.state_offsets)[None],
        torch.from_numpy(sample.gr_gap)[None],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[:, :32]) == 0
