"""Competition metrics."""

from __future__ import annotations

import numpy as np


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute row-level RMSE after pooling all supplied rows."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {predicted.shape}")
    if not len(actual):
        raise ValueError("RMSE requires at least one row")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RMSE inputs must be finite")
    return float(np.sqrt(np.mean(np.square(predicted - actual))))

