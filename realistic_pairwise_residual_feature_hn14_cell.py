"""HN14: consume the crossfit HN path as residual-model features.

HN13 showed that directly blending the HN path does not transfer to LB.  This
gateway asks a narrower question: can a fold-clean row model learn when and
how much of the low-correlation HN endpoint to consume?  A matched base-only
model and an HN-alignment shuffle are mandatory controls.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("Install lightgbm before running HN14") from exc


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "realistic_pairwise_residual_feature_hn14"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

V66_PATH = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
HN_ROW_CANDIDATES = (
    OOF_ROOT / "realistic_pairwise_mainline_fusion_hn4" / "hn3_row_oof.parquet",
    OOF_ROOT / "realistic_pairwise_crossfit_hn3" / "hn3_row_oof.parquet",
)
HN_PATH = next((path for path in HN_ROW_CANDIDATES if path.is_file()), None)
if not V66_PATH.is_file():
    raise FileNotFoundError(V66_PATH)
if HN_PATH is None:
    raise FileNotFoundError(
        "HN3 row OOF is missing. Checked: " + repr([str(path) for path in HN_ROW_CANDIDATES])
    )

SEED = 20260801
# HN3 deliberately produced crossfit endpoints only for its evaluation folds
# 2..5; Fold 1 was untouched by that training protocol and has no HN endpoint.
FOLDS = (2, 3, 4, 5)
MAX_TRAIN_ROWS_PER_WELL = 1024
CORRECTION_CAP_FT = 8.0
DOSES = np.round(np.arange(0.0, 1.001, 0.10), 2)
HN_COLUMN_CANDIDATES = ("hn3_pairwise", "pairwise", "hn_prediction", "prediction")


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


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    tmp = pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
    well_rmse = tmp.groupby("well", sort=False)["e2"].mean().pow(0.5)
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[distance <= 1000.0] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _stable_sample(local, count):
    if len(local) <= count:
        return local.index.to_numpy(np.int64)
    positions = np.linspace(0, len(local) - 1, count).round().astype(np.int64)
    return local.index.to_numpy(np.int64)[np.unique(positions)]


def _load_alignment():
    v66 = pd.read_parquet(V66_PATH)
    hn = pd.read_parquet(HN_PATH)
    v66.columns = [str(column) for column in v66.columns]
    hn.columns = [str(column) for column in hn.columns]
    base_column = next(
        (column for column in ("incumbent_6p377_replay", "current_dose_0p20") if column in v66),
        None,
    )
    hn_column = next((column for column in HN_COLUMN_CANDIDATES if column in hn), None)
    if base_column is None or hn_column is None:
        raise RuntimeError(
            f"Cannot identify endpoints: v66={list(v66.columns)}, hn={list(hn.columns)}"
        )
    key = "id" if "id" in v66 and "id" in hn else None
    if key is None:
        required = {"well", "row_idx"}
        if not required.issubset(v66) or not required.issubset(hn):
            raise RuntimeError("Neither id nor (well,row_idx) alignment keys are available")
        keys = ["well", "row_idx"]
    else:
        keys = ["id"]
    vcols = keys + [
        column for column in ("well", "row_idx", "fold", "tvt_true", "md_since", base_column)
        if column not in keys
    ]
    hcols = keys + [column for column in ("well", "row_idx", "hn_fold", hn_column) if column not in keys]
    left = v66[vcols].copy()
    right = hn[hcols].copy()
    left = left.rename(columns={base_column: "base"})
    right = right.rename(columns={hn_column: "hn"})
    frame = left.merge(right, on=keys, how="inner", suffixes=("", "_hn"), validate="one_to_one")
    if "well_hn" in frame:
        mismatch = frame["well"].astype(str).ne(frame["well_hn"].astype(str))
        if mismatch.any():
            raise RuntimeError("Well mismatch after id alignment")
        frame = frame.drop(columns="well_hn")
    if "row_idx_hn" in frame:
        frame = frame.drop(columns="row_idx_hn")
    if "hn_fold" not in frame:
        raise RuntimeError("HN row OOF does not contain its model holdout fold")
    # V66 and HN3 were generated with different, independently legal well
    # partitions.  The residual gateway must split on HN3's holdout fold so
    # every HN feature is unseen for its target well.  V66 remains OOF under
    # its own partition, retained here for lineage auditing only.
    if "fold" in frame:
        frame = frame.rename(columns={"fold": "v66_fold"})
    frame["fold"] = frame["hn_fold"]
    numeric = [
        "tvt_true", "md_since", "base", "hn", "fold", "hn_fold",
        "row_idx",
    ]
    if "v66_fold" in frame:
        numeric.append("v66_fold")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric).copy()
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = frame["fold"].astype(np.int8)
    frame["hn_fold"] = frame["hn_fold"].astype(np.int8)
    if "v66_fold" in frame:
        frame["v66_fold"] = frame["v66_fold"].astype(np.int8)
    frame["row_idx"] = frame["row_idx"].astype(np.int32)
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if frame["well"].nunique() < 600:
        raise RuntimeError(f"Only {frame['well'].nunique()} aligned wells")
    return frame, base_column, hn_column


def _features(frame):
    out = pd.DataFrame(index=frame.index)
    group = frame.groupby("well", sort=False)
    distance = frame["md_since"].to_numpy(np.float64)
    base = frame["base"].to_numpy(np.float64)
    hn = frame["hn"].to_numpy(np.float64)
    delta = hn - base
    max_distance = group["md_since"].transform("max").to_numpy(np.float64)
    progress = distance / np.maximum(max_distance, 1.0)
    first_base = group["base"].transform("first").to_numpy(np.float64)
    first_hn = group["hn"].transform("first").to_numpy(np.float64)
    out["distance_kft"] = distance / 1000.0
    out["progress"] = progress
    out["progress2"] = progress * progress
    out["log1p_distance"] = np.log1p(np.maximum(distance, 0.0))
    out["base_relative"] = base - first_base
    out["base_relative_per_kft"] = (base - first_base) / np.maximum(distance / 1000.0, 0.25)
    for lag in (32, 128, 512):
        shifted_base = group["base"].shift(lag).to_numpy(np.float64)
        shifted_hn = group["hn"].shift(lag).to_numpy(np.float64)
        out[f"base_delta_lag{lag}"] = np.nan_to_num(base - shifted_base)
        out[f"hn_delta_lag{lag}"] = np.nan_to_num(hn - shifted_hn)
        out[f"disagreement_delta_lag{lag}"] = np.nan_to_num(delta - (shifted_hn - shifted_base))
    out["hn_relative"] = hn - first_hn
    out["disagreement"] = delta
    out["abs_disagreement"] = np.abs(delta)
    out["disagreement_per_kft"] = delta / np.maximum(distance / 1000.0, 0.25)
    well_delta_mean = pd.Series(delta, index=frame.index).groupby(frame["well"], sort=False).transform("mean")
    well_delta_std = pd.Series(delta, index=frame.index).groupby(frame["well"], sort=False).transform("std")
    well_delta_abs = pd.Series(np.abs(delta), index=frame.index).groupby(frame["well"], sort=False).transform("mean")
    out["well_delta_mean"] = well_delta_mean.to_numpy(np.float64)
    out["well_delta_std"] = well_delta_std.fillna(0.0).to_numpy(np.float64)
    out["well_abs_delta_mean"] = well_delta_abs.to_numpy(np.float64)
    out["delta_minus_well_mean"] = delta - out["well_delta_mean"].to_numpy(np.float64)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    base_names = [
        "distance_kft", "progress", "progress2", "log1p_distance",
        "base_relative", "base_relative_per_kft",
        "base_delta_lag32", "base_delta_lag128", "base_delta_lag512",
    ]
    return out, base_names, list(out.columns)


def _fit_model(x, y, sample_weight, feature_names, held_fold, arm):
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
        random_state=SEED + held_fold,
        n_jobs=max(1, min(16, os.cpu_count() or 1)),
        verbosity=-1,
    )
    model.fit(x[feature_names], y, sample_weight=sample_weight)
    model.booster_.save_model(str(MODEL_DIR / f"{arm}_hold{held_fold}.txt"))
    return model


started = time.time()
frame, base_source, hn_source = _load_alignment()
observed_hn_folds = tuple(sorted(frame["fold"].astype(int).unique().tolist()))
if observed_hn_folds != FOLDS:
    raise RuntimeError(
        f"Unexpected HN3 holdout folds: observed={observed_hn_folds}, expected={FOLDS}"
    )
feature_frame, base_features, hn_features = _features(frame)
target = frame["tvt_true"].to_numpy(np.float64) - frame["base"].to_numpy(np.float64)

sample_indices = []
for _, local in frame.groupby("well", sort=False):
    sample_indices.append(_stable_sample(local, MAX_TRAIN_ROWS_PER_WELL))
sample_indices = np.concatenate(sample_indices)

raw_predictions = {"base_only": np.full(len(frame), np.nan), "hn_features": np.full(len(frame), np.nan)}
shuffle_predictions = np.full(len(frame), np.nan)
fold_rows = []
hn_only_columns = [column for column in hn_features if column not in set(base_features)]

for held_fold in FOLDS:
    train_mask_sample = frame.loc[sample_indices, "fold"].to_numpy() != held_fold
    train_indices = sample_indices[train_mask_sample]
    held_indices = np.flatnonzero(frame["fold"].to_numpy() == held_fold)
    if frame.iloc[held_indices]["well"].nunique() < 90:
        raise RuntimeError(f"Fold {held_fold} has insufficient held wells")
    train_wells = frame.iloc[train_indices]["well"].astype(str)
    counts = train_wells.value_counts()
    sample_weight = train_wells.map(
        lambda well: 1.0 / counts[well]
    ).to_numpy(dtype=np.float64, copy=True)
    sample_weight *= len(sample_weight) / sample_weight.sum()

    base_model = _fit_model(
        feature_frame.iloc[train_indices], target[train_indices], sample_weight,
        base_features, held_fold, "base_only",
    )
    hn_model = _fit_model(
        feature_frame.iloc[train_indices], target[train_indices], sample_weight,
        hn_features, held_fold, "hn_features",
    )
    raw_predictions["base_only"][held_indices] = base_model.predict(
        feature_frame.iloc[held_indices][base_features]
    )
    raw_predictions["hn_features"][held_indices] = hn_model.predict(
        feature_frame.iloc[held_indices][hn_features]
    )

    shuffled = feature_frame.iloc[held_indices][hn_features].copy()
    held_wells = frame.iloc[held_indices]["well"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED + 1000 + held_fold)
    for well in np.unique(held_wells):
        positions = np.flatnonzero(held_wells == well)
        if len(positions) > 1:
            permutation = rng.permutation(len(positions))
            shuffled.iloc[positions, shuffled.columns.get_indexer(hn_only_columns)] = (
                shuffled.iloc[positions][hn_only_columns].to_numpy()[permutation]
            )
    shuffle_predictions[held_indices] = hn_model.predict(shuffled)
    print(
        "HN14 FIT",
        {
            "fold": held_fold,
            "train_rows": len(train_indices),
            "held_rows": len(held_indices),
            "held_wells": int(frame.iloc[held_indices]["well"].nunique()),
        },
        flush=True,
    )

base_prediction = frame["base"].to_numpy(np.float64)
for key in raw_predictions:
    raw_predictions[key] = np.clip(raw_predictions[key], -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
shuffle_predictions = np.clip(shuffle_predictions, -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
if any(not np.isfinite(values).all() for values in [*raw_predictions.values(), shuffle_predictions]):
    raise RuntimeError("Non-finite HN14 crossfit correction")


def _select_dose(arm, held_fold):
    inner = frame["fold"].to_numpy() != held_fold
    rows = []
    for dose in DOSES:
        prediction = base_prediction[inner] + dose * raw_predictions[arm][inner]
        metrics = _metrics(frame.loc[inner], prediction)
        base_metrics = _metrics(frame.loc[inner], base_prediction[inner])
        rows.append({
            "dose": float(dose),
            "rmse": metrics["rmse"],
            "gain": base_metrics["rmse"] - metrics["rmse"],
            "p95_change": metrics["p95_well_rmse"] - base_metrics["p95_well_rmse"],
        })
    scan = pd.DataFrame(rows)
    safe = scan[scan["p95_change"] <= 0.20]
    if not len(safe):
        return 0.0
    return float(safe.sort_values(["rmse", "dose"]).iloc[0]["dose"])


final_predictions = {
    "base_only": base_prediction.copy(),
    "hn_features": base_prediction.copy(),
    "hn_alignment_shuffle": base_prediction.copy(),
}
for held_fold in FOLDS:
    held = frame["fold"].to_numpy() == held_fold
    base_dose = _select_dose("base_only", held_fold)
    hn_dose = _select_dose("hn_features", held_fold)
    final_predictions["base_only"][held] += base_dose * raw_predictions["base_only"][held]
    final_predictions["hn_features"][held] += hn_dose * raw_predictions["hn_features"][held]
    final_predictions["hn_alignment_shuffle"][held] += hn_dose * shuffle_predictions[held]
    base_metric = _metrics(frame.loc[held], base_prediction[held])
    base_only_metric = _metrics(frame.loc[held], final_predictions["base_only"][held])
    hn_metric = _metrics(frame.loc[held], final_predictions["hn_features"][held])
    shuffle_metric = _metrics(frame.loc[held], final_predictions["hn_alignment_shuffle"][held])
    fold_rows.append({
        "fold": held_fold,
        "base_dose": base_dose,
        "hn_dose": hn_dose,
        "base_rmse": base_metric["rmse"],
        "base_only_rmse": base_only_metric["rmse"],
        "hn_rmse": hn_metric["rmse"],
        "hn_gain": base_metric["rmse"] - hn_metric["rmse"],
        "increment_vs_base_only": base_only_metric["rmse"] - hn_metric["rmse"],
        "shuffle_rmse": shuffle_metric["rmse"],
        "shuffle_loss": shuffle_metric["rmse"] - hn_metric["rmse"],
        "p95_change": hn_metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
    })

fold_report = pd.DataFrame(fold_rows)
base_metrics = _metrics(frame, base_prediction)
base_only_metrics = _metrics(frame, final_predictions["base_only"])
hn_metrics = _metrics(frame, final_predictions["hn_features"])
shuffle_metrics = _metrics(frame, final_predictions["hn_alignment_shuffle"])
gain = base_metrics["rmse"] - hn_metrics["rmse"]
increment = base_only_metrics["rmse"] - hn_metrics["rmse"]
shuffle_loss = shuffle_metrics["rmse"] - hn_metrics["rmse"]
p95_change = hn_metrics["p95_well_rmse"] - base_metrics["p95_well_rmse"]
checks = {
    "aligned_wells_ge_600": frame["well"].nunique() >= 600,
    "overall_gain_ge_0p10": gain >= 0.10,
    "increment_vs_base_only_ge_0p05": increment >= 0.05,
    "shuffle_loses_ge_0p05": shuffle_loss >= 0.05,
    "positive_hn_folds_ge_4": int((fold_report["hn_gain"] > 0).sum()) >= 4,
    "minimum_fold_increment_ge_minus0p03": float(fold_report["increment_vs_base_only"].min()) >= -0.03,
    "p95_change_le_0p20": p95_change <= 0.20,
}
strict_pass = all(checks.values())

output = frame[["well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["base"] = base_prediction.astype(np.float32)
output["hn3"] = frame["hn"].to_numpy(np.float32)
output["base_only_residual"] = final_predictions["base_only"].astype(np.float32)
output["hn_feature_residual"] = final_predictions["hn_features"].astype(np.float32)
output.to_parquet(OUT_DIR / "hn14_crossfit_oof.parquet", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "hn3_fourfold_crossfit_endpoint_as_residual_features",
    "sample_unit": "HN3 fourfold complete-well outer folds; sampled train rows and full held rows",
    "sources": {"v66": V66_PATH, "hn3_row_oof": HN_PATH},
    "coverage": {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "hn_fold_wells": frame.groupby("fold")["well"].nunique().astype(int).to_dict(),
        "v66_fold_wells": (
            frame.groupby("v66_fold")["well"].nunique().astype(int).to_dict()
            if "v66_fold" in frame else {}
        ),
        "hn_v66_fold_match_fraction": (
            float(np.mean(frame["fold"].to_numpy() == frame["v66_fold"].to_numpy()))
            if "v66_fold" in frame else None
        ),
    },
    "protocol": {
        "hn_evaluation_folds": list(FOLDS),
        "hn_fold1_endpoint_exists": False,
        "v66_predictions_remain_oof_under_their_own_folds": True,
    },
    "features": {
        "base_only": base_features,
        "with_hn": hn_features,
        "correction_cap_ft": CORRECTION_CAP_FT,
    },
    "base": base_metrics,
    "matched_base_only": base_only_metrics,
    "hn_features": hn_metrics,
    "hn_alignment_shuffle": shuffle_metrics,
    "gain": gain,
    "increment_vs_base_only": increment,
    "shuffle_loss": shuffle_loss,
    "p95_change": p95_change,
    "folds": fold_report.to_dict("records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> refit the residual feature model for Kaggle inference"
        if strict_pass
        else "FAIL -> HN information is not legally selectable; close HN after HN13"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nHN14 FOLDS")
print(fold_report.to_string(index=False), flush=True)
print("\nHN14 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
