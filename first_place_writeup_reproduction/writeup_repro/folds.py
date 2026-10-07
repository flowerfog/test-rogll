from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from .data import list_wells, load_record


def build_well_metadata(data_root: str | Path) -> pd.DataFrame:
    rows = []
    for well in list_wells(data_root, "train"):
        record = load_record(data_root, "train", well)
        anchor = record.anchor_index
        rows.append(
            {
                "well": well,
                "x": float(record.x[anchor]),
                "y": float(record.y[anchor]),
                "rows": len(record.md),
                "target_rows": max(0, len(record.md) - anchor - 1),
            }
        )
    return pd.DataFrame(rows)


def make_repeated_folds(metadata: pd.DataFrame, repeats: int = 3, folds: int = 5) -> pd.DataFrame:
    frame = metadata.copy().reset_index(drop=True)
    xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
    xy = (xy - np.nanmean(xy, axis=0)) / np.maximum(np.nanstd(xy, axis=0), 1.0)
    regions = KMeans(n_clusters=min(20, len(frame)), random_state=11, n_init=10).fit_predict(xy)
    length_bin = pd.qcut(frame["target_rows"], q=min(5, len(frame)), labels=False, duplicates="drop")
    strata = regions.astype(str) + "_" + length_bin.fillna(0).astype(int).astype(str)
    counts = pd.Series(strata).value_counts()
    rare = set(counts[counts < folds].index)
    strata = np.array([f"r{regions[i]}" if value in rare else value for i, value in enumerate(strata)])
    for repeat, seed in enumerate((17, 53, 97)[:repeats]):
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        assignment = np.zeros(len(frame), dtype=np.int8)
        try:
            splits = splitter.split(frame, strata)
            for fold, (_, valid) in enumerate(splits, 1):
                assignment[valid] = fold
        except ValueError:
            fallback = StratifiedKFold(folds, shuffle=True, random_state=seed)
            coarse = pd.qcut(frame["target_rows"], q=folds, labels=False, duplicates="drop")
            for fold, (_, valid) in enumerate(fallback.split(frame, coarse), 1):
                assignment[valid] = fold
        frame[f"repeat_{repeat}"] = assignment
    return frame
