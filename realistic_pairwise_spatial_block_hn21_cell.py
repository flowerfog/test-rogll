"""HN21: spatial-block stress test of the HN residual signal.

Random HN folds are distributionally indistinguishable (HN20 AUC ~= 0.5), so
they cannot test geographic extrapolation.  This cell assigns the 617 HN14
wells to five equal-size contiguous Morton-order XY blocks and refits the same
base-only and HN residual models while holding out one entire block.

The HN endpoint remains individually OOF under its original generator folds;
this gateway specifically tests portability of the residual consumer, not a
fully spatially-refit candidate generator.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
SOURCE = ROOT / "realistic_pairwise_residual_feature_hn14_cell.py"
EXPERIMENT = "realistic_pairwise_spatial_block_hn21"
OUT_DIR = OOF / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_BLOCKS = 5
MAX_TRAIN_ROWS_PER_WELL = 1024
CORRECTION_CAP_FT = 8.0
SEED = 20260803


def _load_hn14_definitions():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError("Cannot locate HN14 execution marker")
    name = "hn14_definitions_for_hn21"
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), module.__dict__)
    return module.__dict__


H = _load_hn14_definitions()


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _part1by1(value):
    value = np.asarray(value, np.uint32) & np.uint32(0x0000FFFF)
    value = (value | (value << np.uint32(8))) & np.uint32(0x00FF00FF)
    value = (value | (value << np.uint32(4))) & np.uint32(0x0F0F0F0F)
    value = (value | (value << np.uint32(2))) & np.uint32(0x33333333)
    value = (value | (value << np.uint32(1))) & np.uint32(0x55555555)
    return value


def _coordinates(wells):
    records = []
    failures = []
    for number, well in enumerate(wells, 1):
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        try:
            frame = pd.read_csv(path, usecols=["X", "Y"])
            x = pd.to_numeric(frame["X"], errors="coerce").to_numpy(np.float64)
            y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 32:
                raise RuntimeError(f"only {valid.sum()} finite XY rows")
            records.append({
                "well": str(well),
                "x": float(np.median(x[valid])),
                "y": float(np.median(y[valid])),
                "x_span": float(np.ptp(x[valid])),
                "y_span": float(np.ptp(y[valid])),
            })
        except Exception as error:
            failures.append({"well": str(well), "reason": repr(error)})
        if number % 100 == 0:
            print(f"HN21 XY {number}/{len(wells)} failures={len(failures)}", flush=True)
    pd.DataFrame(failures).to_csv(OUT_DIR / "coordinate_failures.csv", index=False)
    output = pd.DataFrame(records)
    if len(output) != len(wells):
        raise RuntimeError(f"Coordinate coverage {len(output)}/{len(wells)}: {failures[:5]}")
    return output


def _spatial_blocks(coordinates):
    output = coordinates.copy()
    codes = []
    for column in ("x", "y"):
        values = output[column].to_numpy(np.float64)
        ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(np.float64)
        codes.append(np.clip(np.floor(ranks * 65535.0), 0, 65535).astype(np.uint32))
    morton = _part1by1(codes[0]) | (_part1by1(codes[1]) << np.uint32(1))
    order = np.argsort(morton, kind="stable")
    block = np.empty(len(output), np.int8)
    for number, positions in enumerate(np.array_split(order, N_BLOCKS), 1):
        block[positions] = number
    output["morton"] = morton.astype(np.uint64)
    output["spatial_block"] = block
    return output


def _fit_model(x, y, sample_weight, feature_names, block, arm):
    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.85,
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=7,
        min_child_samples=80,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=4.0,
        random_state=SEED + block,
        n_jobs=max(1, min(16, os.cpu_count() or 1)),
        verbosity=-1,
    )
    model.fit(x[feature_names], y, sample_weight=sample_weight)
    model.booster_.save_model(str(MODEL_DIR / f"{arm}_block{block}.txt"))
    return model


def _weights(frame, indices):
    wells = frame.iloc[indices]["well"].astype(str)
    counts = wells.value_counts()
    result = wells.map(lambda well: 1.0 / counts[well]).to_numpy(np.float64, copy=True)
    result *= len(result) / result.sum()
    return result


frame, base_source, hn_source = H["_load_alignment"]()
features, base_features, hn_features = H["_features"](frame)
hn_only = [column for column in hn_features if column not in set(base_features)]
coordinates = _spatial_blocks(_coordinates(sorted(frame["well"].astype(str).unique())))
block_lookup = dict(zip(coordinates["well"], coordinates["spatial_block"]))
frame["spatial_block"] = frame["well"].astype(str).map(block_lookup).astype(np.int8)
target = frame["tvt_true"].to_numpy(np.float64) - frame["base"].to_numpy(np.float64)
base_prediction = frame["base"].to_numpy(np.float64)

sample_indices = np.concatenate([
    H["_stable_sample"](local, MAX_TRAIN_ROWS_PER_WELL)
    for _, local in frame.groupby("well", sort=False)
])
predictions = {
    "base_only": np.full(len(frame), np.nan, np.float64),
    "hn_features": np.full(len(frame), np.nan, np.float64),
    "alignment_shuffle": np.full(len(frame), np.nan, np.float64),
}
fold_rows = []

xy = coordinates[["x", "y"]].to_numpy(np.float64)
xy_scale = np.nanstd(xy, axis=0)
xy_scale = np.where(xy_scale > 1e-6, xy_scale, 1.0)
xy_normalised = (xy - np.nanmean(xy, axis=0)) / xy_scale

for block in range(1, N_BLOCKS + 1):
    train_indices = sample_indices[
        frame.iloc[sample_indices]["spatial_block"].to_numpy(int) != block
    ]
    held_indices = np.flatnonzero(frame["spatial_block"].to_numpy(int) == block)
    held_wells = frame.iloc[held_indices]["well"].astype(str).unique()
    if len(held_wells) < 115:
        raise RuntimeError(f"Spatial block {block}: only {len(held_wells)} wells")

    weight = _weights(frame, train_indices)
    base_model = _fit_model(
        features.iloc[train_indices], target[train_indices], weight,
        base_features, block, "base_only",
    )
    hn_model = _fit_model(
        features.iloc[train_indices], target[train_indices], weight,
        hn_features, block, "hn_features",
    )
    predictions["base_only"][held_indices] = np.clip(
        base_model.predict(features.iloc[held_indices][base_features]),
        -CORRECTION_CAP_FT, CORRECTION_CAP_FT,
    )
    predictions["hn_features"][held_indices] = np.clip(
        hn_model.predict(features.iloc[held_indices][hn_features]),
        -CORRECTION_CAP_FT, CORRECTION_CAP_FT,
    )

    shuffled = features.iloc[held_indices][hn_features].copy()
    held_well_rows = frame.iloc[held_indices]["well"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED + 1000 + block)
    for well in np.unique(held_well_rows):
        local = np.flatnonzero(held_well_rows == well)
        values = shuffled.iloc[local][hn_only].to_numpy(copy=True)
        shuffled.iloc[local, shuffled.columns.get_indexer(hn_only)] = values[
            rng.permutation(len(local))
        ]
    predictions["alignment_shuffle"][held_indices] = np.clip(
        hn_model.predict(shuffled), -CORRECTION_CAP_FT, CORRECTION_CAP_FT
    )

    held_coordinate_mask = coordinates["spatial_block"].to_numpy(int) == block
    train_coordinate_mask = ~held_coordinate_mask
    nearest = NearestNeighbors(n_neighbors=1).fit(xy_normalised[train_coordinate_mask])
    nearest_distance = nearest.kneighbors(
        xy_normalised[held_coordinate_mask], return_distance=True
    )[0][:, 0]

    local_frame = frame.iloc[held_indices]
    base_m = H["_metrics"](local_frame, base_prediction[held_indices])
    base_only_m = H["_metrics"](
        local_frame, base_prediction[held_indices] + predictions["base_only"][held_indices]
    )
    hn_m = H["_metrics"](
        local_frame, base_prediction[held_indices] + predictions["hn_features"][held_indices]
    )
    shuffle_m = H["_metrics"](
        local_frame, base_prediction[held_indices] + predictions["alignment_shuffle"][held_indices]
    )
    fold_rows.append({
        "spatial_block": block,
        "rows": int(len(local_frame)),
        "wells": int(len(held_wells)),
        "median_nearest_train_xy": float(np.median(nearest_distance)),
        "p90_nearest_train_xy": float(np.quantile(nearest_distance, 0.90)),
        "base_rmse": base_m["rmse"],
        "base_only_rmse": base_only_m["rmse"],
        "hn_rmse": hn_m["rmse"],
        "gain_vs_base": base_m["rmse"] - hn_m["rmse"],
        "increment_vs_base_only": base_only_m["rmse"] - hn_m["rmse"],
        "far_gain_vs_base_only": base_only_m["far_gt2000_rmse"] - hn_m["far_gt2000_rmse"],
        "shuffle_loss": shuffle_m["rmse"] - hn_m["rmse"],
        "p95_change_vs_base_only": hn_m["p95_well_rmse"] - base_only_m["p95_well_rmse"],
    })
    print("HN21 BLOCK", fold_rows[-1], flush=True)

if any(not np.isfinite(values).all() for values in predictions.values()):
    raise RuntimeError("Incomplete HN21 predictions")

final = {name: base_prediction + correction for name, correction in predictions.items()}
base_metrics = H["_metrics"](frame, base_prediction)
base_only_metrics = H["_metrics"](frame, final["base_only"])
hn_metrics = H["_metrics"](frame, final["hn_features"])
shuffle_metrics = H["_metrics"](frame, final["alignment_shuffle"])
folds = pd.DataFrame(fold_rows)
increment = base_only_metrics["rmse"] - hn_metrics["rmse"]
shuffle_loss = shuffle_metrics["rmse"] - hn_metrics["rmse"]
checks = {
    "aligned_wells_ge_600": bool(frame["well"].nunique() >= 600),
    "each_spatial_block_ge_115_wells": bool(folds["wells"].min() >= 115),
    "increment_vs_base_only_ge_0p05": bool(increment >= 0.05),
    "positive_blocks_ge_4": bool((folds["increment_vs_base_only"] > 0).sum() >= 4),
    "minimum_block_increment_ge_minus0p05": bool(folds["increment_vs_base_only"].min() >= -0.05),
    "shuffle_loses_ge_0p05": bool(shuffle_loss >= 0.05),
    "maximum_p95_change_le_0p50": bool(folds["p95_change_vs_base_only"].max() <= 0.50),
}
strict_pass = bool(all(checks.values()))

output = frame[["well", "row_idx", "fold", "spatial_block", "tvt_true", "md_since"]].copy()
output["base"] = base_prediction.astype(np.float32)
for name, values in final.items():
    output[name] = values.astype(np.float32)
output.to_parquet(OUT_DIR / "hn21_oof.parquet", index=False, compression="zstd")
coordinates.to_csv(OUT_DIR / "spatial_blocks.csv", index=False)
folds.to_csv(OUT_DIR / "blocks.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "five_equal_size_morton_xy_block_residual_stress_test",
    "sample_unit": "complete physical well",
    "scope": "residual consumer spatial portability; HN generator remains individually OOF",
    "sources": {
        "hn14_source": str(SOURCE),
        "v66": str(H["V66_PATH"]),
        "hn_rows": str(H["HN_PATH"]),
    },
    "coverage": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "block_wells": {str(k): int(v) for k, v in coordinates.groupby("spatial_block")["well"].nunique().items()},
    },
    "base": base_metrics,
    "base_only": base_only_metrics,
    "hn_features": hn_metrics,
    "alignment_shuffle": shuffle_metrics,
    "increment_vs_base_only": increment,
    "shuffle_loss": shuffle_loss,
    "blocks": fold_rows,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> HN signal survives geographic extrapolation; inspect exact Kaggle lineage"
        if strict_pass
        else "FAIL -> HN residual is interpolation-bound; retire HN deployment branch"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nHN21 BLOCKS")
print(folds.to_string(index=False), flush=True)
print("\nHN21 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
