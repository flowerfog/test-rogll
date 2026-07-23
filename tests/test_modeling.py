import unittest

import numpy as np

from rogii_baseline.modeling import ModelConfig, TrainingSample, fit_model


class TorchModelTest(unittest.TestCase):
    def test_small_cpu_fit_predict_is_finite(self) -> None:
        random = np.random.default_rng(42)
        x = random.normal(size=(32, 28)).astype(np.float32)
        residual = (0.5 * x[:, 0] - 0.2 * x[:, 1]).astype(np.float64)
        samples = {
            "well_a": TrainingSample(x=x[:16], residual=residual[:16]),
            "well_b": TrainingSample(x=x[16:], residual=residual[16:]),
        }
        config = ModelConfig(
            max_iter=2,
            batch_size=8,
            hidden_width=16,
            device="cpu",
        )

        model = fit_model(samples, ["well_a", "well_b"], config)
        prediction = model.predict(x[:5])

        self.assertEqual(prediction.shape, (5,))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
