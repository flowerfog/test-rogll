import numpy as np

from writeup_repro.config import GridConfig
from writeup_repro.data import PanelSample, WellRecord
from writeup_repro.inferred_defaults import PFDefaults, PFProfile, XYDefaults
from writeup_repro.pf import MultiProfileParticleFilter
from writeup_repro.xy import XYSurfacePrior


def synthetic_sample(height=24):
    grid = GridConfig(horizontal_bins=height)
    offsets = grid.state_offsets
    tw = np.sin(offsets / 8.0)
    path = np.linspace(-8, 8, height)
    hw = np.interp(path, offsets, tw)
    values = np.stack(
        [
            np.broadcast_to(tw, (height, 400)),
            np.broadcast_to(hw[:, None], (height, 400)),
            np.zeros((height, 400)),
        ]
    ).astype(np.float32)
    return PanelSample(
        "synthetic", values, ("tw_gr", "hw_gr_mean", "z_diff"),
        np.zeros((height, 400), np.float32), np.zeros(height, np.float32),
        np.zeros(height, bool), np.abs(tw[None] - hw[:, None]).astype(np.float32),
        offsets, np.repeat(np.arange(height), 32), np.arange(height * 32),
        np.arange(height * 32), 0.0, 31,
    )


def test_pf_density_and_ffbsi():
    defaults = PFDefaults(
        particles=64,
        ffbsi_paths=4,
        profiles=(PFProfile(1.0, 8.0, 0.01, 20.0, 1.0),),
    )
    result = MultiProfileParticleFilter(defaults, seed=3).run(synthetic_sample())
    assert result.density.shape == (24, 400)
    assert np.allclose(result.density.sum(axis=1), 1.0, atol=1e-4)
    assert np.isfinite(result.ffbsi_offset).all()


def make_record(well, x, y, surface):
    rows = 64
    z = np.linspace(-100, -110, rows).astype(np.float32)
    tvt = (surface - z).astype(np.float32)
    return WellRecord(
        well, np.arange(rows, dtype=np.float32), np.full(rows, x, np.float32),
        np.full(rows, y, np.float32), z, np.zeros(rows, np.float32), tvt, tvt,
        np.linspace(tvt.min()-10, tvt.max()+10, 100).astype(np.float32), np.zeros(100, np.float32)
    )


def test_xy_plane_recovery():
    records = []
    for x in (-1000.0, 0.0, 1000.0):
        for y in (-1000.0, 0.0, 1000.0):
            records.append(make_record(f"w{x}_{y}", x, y, 20 + 0.01*x - 0.02*y))
    prior = XYSurfacePrior(XYDefaults(neighbors=24, distance_floor=10)).fit_records(records, points_per_well=2)
    query = make_record("query", 250.0, -300.0, 20 + 2.5 + 6.0)
    sample = synthetic_sample()
    sample = PanelSample(**{**sample.__dict__, "selected_rows": np.arange(24*32) % 64})
    result = prior.predict(query, sample)
    assert np.isfinite(result.tvt_offset).all()
    expected = query.truth[sample.selected_rows.reshape(24, -1)].mean(axis=1) - query.anchor_tvt
    assert np.sqrt(np.mean((result.tvt_offset - expected) ** 2)) < 1.0
