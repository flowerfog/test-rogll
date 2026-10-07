from pathlib import Path

import numpy as np

from writeup_repro.data import build_base_panel, list_wells, load_record
from writeup_repro.infer import decode_well


ROOT = Path(__file__).resolve().parents[2]


def test_decode_exports_future_official_contract():
    well = list_wells(ROOT, "train")[0]
    record = load_record(ROOT, "train", well)
    sample = build_base_panel(record)
    frame = decode_well(record, sample, np.zeros(345, dtype=np.float32), smooth_window=31)
    assert list(frame.columns) == ["well", "row", "id", "tvt"]
    assert (frame.row > record.anchor_index).all()
    assert frame.id.str.startswith(f"{well}_").all()
    assert np.isfinite(frame.tvt).all()
