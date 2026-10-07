from dataclasses import replace

import numpy as np

from first_place_repro.augment import DIFF_BLOCK_REVERSE_PROBABILITY, RawWellAugmenter
from first_place_repro.data import WellRecord


def _record(well: str = "a", length: int = 1400) -> WellRecord:
    md = np.arange(length, dtype=np.float32)
    truth = (20.0 + 0.02 * md + 0.5 * np.sin(md / 80.0)).astype(np.float32)
    tvt_input = truth.copy()
    tvt_input[400:] = np.nan
    typewell_tvt = np.linspace(-100, 200, 1201, dtype=np.float32)
    typewell_gr = (80 + 20 * np.sin(typewell_tvt / 8)).astype(np.float32)
    gr = np.interp(truth, typewell_tvt, typewell_gr).astype(np.float32)
    return WellRecord(
        well=well,
        md=md,
        x=md,
        y=np.zeros(length, dtype=np.float32),
        z=(1000 - truth).astype(np.float32),
        gr=gr,
        tvt_input=tvt_input,
        truth=truth,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
    )


def test_z_shift_uses_other_well_and_preserves_surface():
    record = _record()
    donor = np.full(2000, 0.01, dtype=np.float32)
    augmented = RawWellAugmenter(3, "0719_V1", [("donor", donor)])._z_shift(record)
    suffix = ~np.isfinite(record.tvt_input)
    np.testing.assert_allclose(
        augmented.truth[suffix] + augmented.z[suffix],
        record.truth[suffix] + record.z[suffix],
        atol=1e-4,
    )
    assert np.all(np.isnan(augmented.tvt_input[suffix]))
    assert np.allclose(augmented.tvt_input[~suffix], augmented.truth[~suffix])


def test_z_shift_applies_one_bounded_whole_well_translation():
    record = _record()
    donor = np.full(2000, 0.01, dtype=np.float32)
    augmented = RawWellAugmenter(5, "0719_V1", [("donor", donor)])._z_shift(record)
    visible = np.isfinite(record.tvt_input)
    shifts = record.tvt_input[visible] - augmented.tvt_input[visible]
    np.testing.assert_allclose(shifts, shifts[0], atol=1e-5)
    assert abs(float(shifts[0])) <= 40.0
    np.testing.assert_allclose(
        augmented.truth[visible] + augmented.z[visible],
        record.truth[visible] + record.z[visible],
        atol=1e-4,
    )


def test_md_stretch_resamples_all_horizontal_arrays():
    record = _record(length=1200)
    augmented = RawWellAugmenter(7, "0719_V1", [("donor", np.ones(1200, dtype=np.float32))])._md_stretch(record)
    assert len(augmented.md) != len(record.md)
    lengths = {len(getattr(augmented, name)) for name in ("md", "x", "y", "z", "gr", "tvt_input", "truth")}
    assert len(lengths) == 1


def test_gr_transform_leaves_horizontal_gr_unchanged():
    record = _record()
    augmented = RawWellAugmenter(11, "0719_V1", [("donor", np.ones(1200, dtype=np.float32))])._gr_transform(record)
    np.testing.assert_array_equal(augmented.gr, record.gr)
    assert not np.array_equal(augmented.typewell_gr, record.typewell_gr)


def test_active_0719_diff_blocks_do_not_reverse():
    assert DIFF_BLOCK_REVERSE_PROBABILITY == 0.0
