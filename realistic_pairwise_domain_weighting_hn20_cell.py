"""HN20: fold-as-hidden covariate-shift weighting for the HN residual model.

For each held HN fold, a label-free domain classifier distinguishes held wells
from the remaining training wells using complete-well base/HN trajectory
summaries.  Its density-ratio estimate reweights only the residual model's
training wells.  The residual target, row features, model class, correction
cap, and dose are identical to HN14.  Unweighted and shuffled-weight models
are matched controls.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
SOURCE = ROOT / "realistic_pairwise_residual_feature_hn14_cell.py"
EXPERIMENT = "realistic_pairwise_domain_weighting_hn20"
OUT_DIR = OOF / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = (2, 3, 4, 5)
MAX_TRAIN_ROWS_PER_WELL = 1024
CORRECTION_CAP_FT = 8.0
SEED = 20260802
DOMAIN_WEIGHT_MIN = 0.20
DOMAIN_WEIGHT_MAX = 5.0


def _load_hn14_definitions():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError("Cannot locate HN14 execution marker")
    name = "hn14_definitions_for_hn20"
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


def _domain_features(frame, row_features):
    records = []
    values = row_features.copy()
    values["well"] = frame["well"].astype(str).to_numpy()
    values["fold"] = frame["fold"].astype(int).to_numpy()
    for well, local in values.groupby("well", sort=False):
        row = {
            "well": str(well),
            "fold": int(local["fold"].iloc[0]),
            "rows": int(len(local)),
            "max_md_since": float(frame.loc[local.index, "md_since"].max()),
        }
        numeric = local.drop(columns=["well", "fold"])
        # Means and spreads retain trajectory/disagreement shape while keeping
        # the domain classifier low-capacity relative to 617 wells.
        for column in numeric.columns:
            data = numeric[column].to_numpy(np.float64)
            row[f"mean__{column}"] = float(np.nanmean(data))
            row[f"std__{column}"] = float(np.nanstd(data))
        for column in (
            "base_relative", "base_relative_per_kft", "hn_relative",
            "disagreement", "abs_disagreement", "disagreement_per_kft",
        ):
            if column not in numeric:
                continue
            data = numeric[column].to_numpy(np.float64)
            row[f"p10__{column}"] = float(np.nanquantile(data, 0.10))
            row[f"p90__{column}"] = float(np.nanquantile(data, 0.90))
        records.append(row)
    output = pd.DataFrame(records)
    if output["well"].duplicated().any():
        raise RuntimeError("Duplicate domain-feature wells")
    return output


def _domain_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                C=0.20,
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ),
        ),
    ])


def _density_ratio_weights(domain, held_fold, feature_names):
    labels = domain["fold"].eq(held_fold).astype(np.int8).to_numpy()
    x = domain[feature_names]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + held_fold)
    oof_probability = cross_val_predict(
        _domain_pipeline(), x, labels, cv=splitter, method="predict_proba"
    )[:, 1]
    auc = float(roc_auc_score(labels, oof_probability))

    model = _domain_pipeline()
    model.fit(x, labels)
    probability = np.clip(model.predict_proba(x)[:, 1], 1e-3, 1.0 - 1e-3)
    n_train = int(np.sum(labels == 0))
    n_target = int(np.sum(labels == 1))
    ratio = probability / (1.0 - probability) * (n_train / max(n_target, 1))
    ratio = np.clip(ratio, DOMAIN_WEIGHT_MIN, DOMAIN_WEIGHT_MAX)
    ratio[labels == 1] = np.nan
    training_ratio = ratio[labels == 0]
    training_ratio /= np.mean(training_ratio)
    ratio[labels == 0] = training_ratio
    effective_n = float(np.sum(training_ratio) ** 2 / np.sum(training_ratio ** 2))
    return dict(zip(domain["well"].astype(str), ratio)), auc, effective_n


def _fit_residual(x, y, sample_weight, feature_names, held_fold, arm):
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


def _row_weights(frame, indices, well_ratio=None):
    wells = frame.iloc[indices]["well"].astype(str)
    counts = wells.value_counts()
    weights = wells.map(lambda well: 1.0 / counts[well]).to_numpy(np.float64, copy=True)
    if well_ratio is not None:
        weights *= wells.map(lambda well: float(well_ratio[well])).to_numpy(np.float64)
    weights *= len(weights) / weights.sum()
    return weights


frame, base_source, hn_source = H["_load_alignment"]()
row_features, base_features, hn_features = H["_features"](frame)
domain = _domain_features(frame, row_features)
domain_feature_names = [column for column in domain.columns if column not in {"well", "fold"}]
target = frame["tvt_true"].to_numpy(np.float64) - frame["base"].to_numpy(np.float64)
base_prediction = frame["base"].to_numpy(np.float64)

sample_indices = np.concatenate([
    H["_stable_sample"](local, MAX_TRAIN_ROWS_PER_WELL)
    for _, local in frame.groupby("well", sort=False)
])

predictions = {
    "unweighted": np.full(len(frame), np.nan, np.float64),
    "domain_weighted": np.full(len(frame), np.nan, np.float64),
    "weight_shuffle": np.full(len(frame), np.nan, np.float64),
}
fold_rows = []
rng = np.random.default_rng(SEED)

for held_fold in FOLDS:
    train_indices = sample_indices[
        frame.iloc[sample_indices]["fold"].to_numpy(int) != held_fold
    ]
    held_indices = np.flatnonzero(frame["fold"].to_numpy(int) == held_fold)
    held_wells = frame.iloc[held_indices]["well"].astype(str).unique()
    if len(held_wells) < 145:
        raise RuntimeError(f"Fold {held_fold}: only {len(held_wells)} held wells")

    ratio, domain_auc, effective_n = _density_ratio_weights(
        domain, held_fold, domain_feature_names
    )
    train_wells_unique = sorted(frame.iloc[train_indices]["well"].astype(str).unique())
    ratio_values = np.asarray([ratio[well] for well in train_wells_unique], np.float64)
    shuffled_values = ratio_values[rng.permutation(len(ratio_values))]
    shuffled_ratio = dict(zip(train_wells_unique, shuffled_values))

    unweighted_model = _fit_residual(
        row_features.iloc[train_indices], target[train_indices],
        _row_weights(frame, train_indices), hn_features, held_fold, "unweighted",
    )
    weighted_model = _fit_residual(
        row_features.iloc[train_indices], target[train_indices],
        _row_weights(frame, train_indices, ratio), hn_features, held_fold, "domain_weighted",
    )
    shuffle_model = _fit_residual(
        row_features.iloc[train_indices], target[train_indices],
        _row_weights(frame, train_indices, shuffled_ratio), hn_features, held_fold, "weight_shuffle",
    )

    for name, model in (
        ("unweighted", unweighted_model),
        ("domain_weighted", weighted_model),
        ("weight_shuffle", shuffle_model),
    ):
        correction = model.predict(row_features.iloc[held_indices][hn_features])
        predictions[name][held_indices] = np.clip(
            correction, -CORRECTION_CAP_FT, CORRECTION_CAP_FT
        )

    local = frame.iloc[held_indices]
    base_m = H["_metrics"](local, base_prediction[held_indices])
    unweighted_m = H["_metrics"](
        local, base_prediction[held_indices] + predictions["unweighted"][held_indices]
    )
    weighted_m = H["_metrics"](
        local, base_prediction[held_indices] + predictions["domain_weighted"][held_indices]
    )
    shuffle_m = H["_metrics"](
        local, base_prediction[held_indices] + predictions["weight_shuffle"][held_indices]
    )
    fold_rows.append({
        "fold": held_fold,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "domain_auc": domain_auc,
        "effective_training_wells": effective_n,
        "weight_min": float(np.min(ratio_values)),
        "weight_p50": float(np.median(ratio_values)),
        "weight_max": float(np.max(ratio_values)),
        "base_rmse": base_m["rmse"],
        "unweighted_rmse": unweighted_m["rmse"],
        "weighted_rmse": weighted_m["rmse"],
        "gain_vs_unweighted": unweighted_m["rmse"] - weighted_m["rmse"],
        "gain_vs_base": base_m["rmse"] - weighted_m["rmse"],
        "far_gain_vs_unweighted": unweighted_m["far_gt2000_rmse"] - weighted_m["far_gt2000_rmse"],
        "shuffle_rmse": shuffle_m["rmse"],
        "shuffle_loss": shuffle_m["rmse"] - weighted_m["rmse"],
        "p95_change_vs_unweighted": weighted_m["p95_well_rmse"] - unweighted_m["p95_well_rmse"],
    })
    print("HN20 FOLD", fold_rows[-1], flush=True)

if any(not np.isfinite(values).all() for values in predictions.values()):
    raise RuntimeError("Incomplete HN20 predictions")

final = {
    name: base_prediction + correction for name, correction in predictions.items()
}
base_metrics = H["_metrics"](frame, base_prediction)
unweighted_metrics = H["_metrics"](frame, final["unweighted"])
weighted_metrics = H["_metrics"](frame, final["domain_weighted"])
shuffle_metrics = H["_metrics"](frame, final["weight_shuffle"])
folds = pd.DataFrame(fold_rows)
gain = unweighted_metrics["rmse"] - weighted_metrics["rmse"]
shuffle_loss = shuffle_metrics["rmse"] - weighted_metrics["rmse"]
checks = {
    "aligned_wells_ge_600": bool(frame["well"].nunique() >= 600),
    "domain_auc_mean_ge_0p60": bool(folds["domain_auc"].mean() >= 0.60),
    "gain_vs_unweighted_ge_0p03": bool(gain >= 0.03),
    "positive_folds_ge_3": bool((folds["gain_vs_unweighted"] > 0).sum() >= 3),
    "minimum_fold_gain_ge_minus0p03": bool(folds["gain_vs_unweighted"].min() >= -0.03),
    "shuffle_loses_ge_0p02": bool(shuffle_loss >= 0.02),
    "minimum_effective_training_wells_ge_150": bool(folds["effective_training_wells"].min() >= 150),
    "maximum_p95_change_le_0p30": bool(folds["p95_change_vs_unweighted"].max() <= 0.30),
}
strict_pass = bool(all(checks.values()))

output = frame[["well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["base"] = base_prediction.astype(np.float32)
for name, values in final.items():
    output[name] = values.astype(np.float32)
output.to_parquet(OUT_DIR / "hn20_oof.parquet", index=False, compression="zstd")
domain.to_parquet(OUT_DIR / "domain_features.parquet", index=False)
folds.to_csv(OUT_DIR / "folds.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fourfold_fold_as_hidden_density_ratio_residual_ab",
    "sample_unit": "complete physical well",
    "label_free_domain_inputs": True,
    "residual_recipe": "HN14 features/model/cap/dose unchanged",
    "sources": {
        "hn14_source": str(SOURCE),
        "v66": str(H["V66_PATH"]),
        "hn_rows": str(H["HN_PATH"]),
    },
    "coverage": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "folds": {str(k): int(v) for k, v in frame.groupby("fold")["well"].nunique().items()},
    },
    "domain_features": len(domain_feature_names),
    "base": base_metrics,
    "unweighted": unweighted_metrics,
    "domain_weighted": weighted_metrics,
    "weight_shuffle": shuffle_metrics,
    "gain_vs_unweighted": gain,
    "shuffle_loss": shuffle_loss,
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build Kaggle hidden-domain weights before full residual refit"
        if strict_pass
        else "FAIL -> covariate weighting does not explain local/LB divergence; close branch"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nHN20 FOLDS")
print(folds.to_string(index=False), flush=True)
print("\nHN20 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
