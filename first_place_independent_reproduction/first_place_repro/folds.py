from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from .data import list_wells, load_record


FOLD_CONTRACT = "geo_skfold_mean_xy_kmeans20_seed33_35_v1"


def build_well_metadata(data_root: str | Path) -> pd.DataFrame:
    rows = []
    for well in list_wells(data_root, "train"):
        record = load_record(data_root, "train", well)
        anchor = record.anchor_index
        rows.append(
            {
                "well": well,
                "x": float(np.nanmean(record.x)),
                "y": float(np.nanmean(record.y)),
                "rows": len(record.md),
                "target_rows": max(0, len(record.md) - anchor - 1),
            }
        )
    return pd.DataFrame(rows)


def make_repeated_folds(metadata: pd.DataFrame, repeats: int = 3, folds: int = 5) -> pd.DataFrame:
    frame = metadata.copy().reset_index(drop=True)
    xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
    regions = KMeans(n_clusters=min(20, len(frame)), random_state=0, n_init="auto").fit_predict(xy)
    strata = regions
    for repeat, seed in enumerate((33, 34, 35)[:repeats]):
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        assignment = np.zeros(len(frame), dtype=np.int8)
        for fold, (_, valid) in enumerate(splitter.split(frame, strata), 1):
            assignment[valid] = fold
        frame[f"repeat_{repeat}"] = assignment
    frame["fold_contract"] = FOLD_CONTRACT
    return frame


def validate_fold_contract(frame: pd.DataFrame, repeats: int = 3) -> None:
    required = {"well", "x", "y", "fold_contract"} | {
        f"repeat_{repeat}" for repeat in range(repeats)
    }
    missing = sorted(required - set(frame.columns))
    contracts = set(frame.get("fold_contract", pd.Series(dtype=str)).dropna().astype(str))
    if missing or contracts != {FOLD_CONTRACT}:
        raise RuntimeError({
            "message": "fold file does not satisfy the disclosed geo_skfold contract",
            "expected_contract": FOLD_CONTRACT,
            "found_contracts": sorted(contracts),
            "missing_columns": missing,
            "action": "use a new output root or delete only the stale folds.csv and regenerate it",
        })
