from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth_raw(raw: np.ndarray, window: int, blend: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    finite = np.isfinite(raw)
    if not finite.all():
        valid = np.flatnonzero(finite)
        if len(valid) < 2:
            return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.interp(np.arange(len(raw)), valid, raw[valid]).astype(np.float32)
    usable = min(window, len(raw) if len(raw) % 2 else len(raw) - 1)
    if usable < 3:
        return raw.astype(np.float32)
    smooth = savgol_filter(raw, usable, 1, mode="interp")
    return ((1 - blend) * raw + blend * smooth).astype(np.float32)
