import unittest

import numpy as np

from rogii_baseline.metrics import pooled_rmse


class PooledRmseTest(unittest.TestCase):
    def test_weights_rows_not_wells(self) -> None:
        # One short, bad well and one long, perfect well must be row-weighted.
        actual = np.zeros(10)
        predicted = np.array([10.0] + [0.0] * 9)

        score = pooled_rmse(actual, predicted)
        self.assertAlmostEqual(score, np.sqrt(100.0 / 10.0))
        self.assertNotAlmostEqual(score, (10.0 + 0.0) / 2.0)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            pooled_rmse(np.zeros(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
