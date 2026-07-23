import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from rogii_baseline.data import (
    DEFAULT_DATA_ROOT,
    discover_wells,
    read_well,
    resolve_data_root,
)


def _write_well(root: Path, well_id: str, tvt_input: list[float | None]) -> None:
    horizontal = pd.DataFrame(
        {
            "MD": [100.0, 101.0, 102.0, 103.0],
            "X": [10.0, 10.1, 10.3, 10.6],
            "Y": [20.0, 20.0, 20.1, 20.2],
            "Z": [-50.0, -51.0, -51.3, -51.4],
            "TVT": [1000.0, 1001.0, 1001.4, 1001.8],
            "GR": [80.0, 82.0, None, 90.0],
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame(
        {"TVT": [999.0, 1000.0, 1001.0], "GR": [79.0, 81.0, 84.0]}
    )
    horizontal.to_csv(root / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(root / f"{well_id}__typewell.csv", index=False)


class DataLoadingTest(unittest.TestCase):
    def test_data_root_defaults_only_when_environment_is_absent(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_data_root(), DEFAULT_DATA_ROOT)
        with patch.dict(os.environ, {"ROGII_DATA_ROOT": ""}, clear=True):
            self.assertEqual(resolve_data_root(), Path(""))

    def test_discovers_pair_and_loads_suffix_mask(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_well(root, "well_a", [1000.0, 1001.0, None, None])

            wells = discover_wells(root)
            horizontal, typewell = read_well(wells[0], require_target=True)

            self.assertEqual([well.well_id for well in wells], ["well_a"])
            self.assertEqual(
                horizontal["TVT_input"].notna().tolist(),
                [True, True, False, False],
            )
            self.assertEqual(typewell.columns.tolist(), ["TVT", "GR"])

    def test_rejects_noncontiguous_observed_prefix(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_well(root, "well_b", [1000.0, None, 1001.4, None])

            with self.assertRaisesRegex(ValueError, "not a contiguous prefix"):
                read_well(discover_wells(root)[0], require_target=True)


if __name__ == "__main__":
    unittest.main()
