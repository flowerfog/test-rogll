"""HN15: test HN14 residual features on the exact HN13 deployment endpoint.

The residual model is fitted only on HN3 crossfit wells (HN folds 2-5).  It is
then applied, without tuning, to HN12's 155 HN-fold-1 wells, where every HN
model used by the deployment aggregation is unseen on every evaluation well.
"""

from pathlib import Path
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
EXPERIMENT = "realistic_pairwise_deployment_residual_hn15"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = (
    OOF_ROOT / "realistic_pairwise_residual_feature_hn14" / "hn14_crossfit_oof.parquet"
)
DEPLOYMENT_PATH = (
    OOF_ROOT / "realistic_pairwise_deployment_aggregation_hn12"
    / "fold1_aggregation_endpoints.parquet"
)
V66_PATH = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)

PRIMARY_ENDPOINT = "score_mean"  # Exact HN13 aggregation; never selected here.
DIAGNOSTIC_ENDPOINTS = (
    "score_median",
    "score_borda",
    "path_mean",
    "path_median",
    "path_medoid",
    "score_majority_vote",
    "score_maximin",
    "score_margin_weighted",
)
MAX_TRAIN_ROWS_PER_WELL = 512
CORRECTION_CAP_FT = 8.0
DOSE = 1.0  # Frozen because all four HN14 outer folds selected 1.0.
SEED = 20260730


def _clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    well_rmse = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
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
        out[f"disagreement_delta_lag{lag}"] = np.nan_to_num(
            delta - (shifted_hn - shifted_base)
        )
    out["hn_relative"] = hn - first_hn
    out["disagreement"] = delta
    out["abs_disagreement"] = np.abs(delta)
    out["disagreement_per_kft"] = delta / np.maximum(distance / 1000.0, 0.25)
    delta_series = pd.Series(delta, index=frame.index)
    out["well_delta_mean"] = delta_series.groupby(frame["well"], sort=False).transform("mean")
    out["well_delta_std"] = (
        delta_series.groupby(frame["well"], sort=False).transform("std").fillna(0.0)
    )
    out["well_abs_delta_mean"] = (
        pd.Series(np.abs(delta), index=frame.index)
        .groupby(frame["well"], sort=False)
        .transform("mean")
    )
    out["delta_minus_well_mean"] = delta - out["well_delta_mean"].to_numpy(np.float64)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    base_names = [
        "distance_kft", "progress", "progress2", "log1p_distance",
        "base_relative", "base_relative_per_kft",
        "base_delta_lag32", "base_delta_lag128", "base_delta_lag512",
    ]
    return out, base_names, list(out.columns)


def _stable_sample(frame, count):
    blocks = []
    for _, local in frame.groupby("well", sort=False):
        if len(local) <= count:
            blocks.append(local.index.to_numpy(np.int64))
        else:
            positions = np.linspace(0, len(local) - 1, count).round().astype(np.int64)
            blocks.append(local.index.to_numpy(np.int64)[np.unique(positions)])
    return np.concatenate(blocks)


def _fit_model(x, y, sample_weight, features, name):
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
        random_state=SEED,
        n_jobs=max(1, min(16, os.cpu_count() or 1)),
        verbosity=-1,
    )
    model.fit(x[features], y, sample_weight=sample_weight)
    model.booster_.save_model(str(MODEL_DIR / f"{name}.txt"))
    return model


def _prepare_train():
    frame = pd.read_parquet(TRAIN_PATH)
    required = {"well", "row_idx", "tvt_true", "md_since", "base", "hn3"}
    if not required.issubset(frame):
        raise RuntimeError(f"HN14 train columns missing: {sorted(required - set(frame))}")
    frame = frame.rename(columns={"hn3": "hn"})
    frame["well"] = frame["well"].astype(str)
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    return frame


def _prepare_test(endpoint):
    columns = pd.read_parquet(DEPLOYMENT_PATH).columns.tolist()
    if endpoint not in columns:
        raise RuntimeError(f"Missing HN12 endpoint {endpoint}; available={columns}")
    hn = pd.read_parquet(DEPLOYMENT_PATH, columns=["well", "row_idx", endpoint])
    v66 = pd.read_parquet(
        V66_PATH,
        columns=[
            "well", "row_idx", "fold", "tvt_true", "md_since",
            "incumbent_6p377_replay",
        ],
    )
    hn["well"] = hn["well"].astype(str)
    v66["well"] = v66["well"].astype(str)
    frame = v66.merge(hn, on=["well", "row_idx"], how="inner", validate="one_to_one")
    frame = frame.rename(columns={"incumbent_6p377_replay": "base", endpoint: "hn"})
    numeric = ["row_idx", "fold", "tvt_true", "md_since", "base", "hn"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric).copy()
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    return frame


started = time.time()
for path in (TRAIN_PATH, DEPLOYMENT_PATH, V66_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)

train = _prepare_train()
test_primary = _prepare_test(PRIMARY_ENDPOINT)
overlap = sorted(set(train["well"]).intersection(test_primary["well"]))
if overlap:
    raise RuntimeError(f"Residual train/evaluation well overlap: {overlap[:10]}")
if train["well"].nunique() < 600 or test_primary["well"].nunique() < 145:
    raise RuntimeError(
        f"Insufficient coverage train/test={train['well'].nunique()}/{test_primary['well'].nunique()}"
    )

train_features, base_features, hn_features = _features(train)
sample = _stable_sample(train, MAX_TRAIN_ROWS_PER_WELL)
train_wells = train.iloc[sample]["well"].astype(str)
counts = train_wells.value_counts()
weights = train_wells.map(lambda well: 1.0 / counts[well]).to_numpy(
    dtype=np.float64, copy=True
)
weights *= len(weights) / weights.sum()
target = train["tvt_true"].to_numpy(np.float64) - train["base"].to_numpy(np.float64)
base_model = _fit_model(
    train_features.iloc[sample], target[sample], weights, base_features, "base_only_full"
)
hn_model = _fit_model(
    train_features.iloc[sample], target[sample], weights, hn_features, "hn_features_full"
)

reports = []
saved = None
for endpoint in (PRIMARY_ENDPOINT,) + DIAGNOSTIC_ENDPOINTS:
    test = _prepare_test(endpoint)
    test_features, check_base_features, check_hn_features = _features(test)
    if check_base_features != base_features or check_hn_features != hn_features:
        raise RuntimeError("HN15 feature schema drift")
    base = test["base"].to_numpy(np.float64)
    base_correction = np.clip(
        base_model.predict(test_features[base_features]),
        -CORRECTION_CAP_FT,
        CORRECTION_CAP_FT,
    )
    hn_correction = np.clip(
        hn_model.predict(test_features[hn_features]),
        -CORRECTION_CAP_FT,
        CORRECTION_CAP_FT,
    )
    shuffled_features = test_features[hn_features].copy()
    hn_only = [column for column in hn_features if column not in set(base_features)]
    rng = np.random.default_rng(SEED + 1000)
    well_values = test["well"].astype(str).to_numpy()
    for well in np.unique(well_values):
        positions = np.flatnonzero(well_values == well)
        if len(positions) > 1:
            values = shuffled_features.iloc[positions][hn_only].to_numpy(copy=True)
            shuffled_features.iloc[
                positions, shuffled_features.columns.get_indexer(hn_only)
            ] = values[rng.permutation(len(positions))]
    shuffle_correction = np.clip(
        hn_model.predict(shuffled_features), -CORRECTION_CAP_FT, CORRECTION_CAP_FT
    )
    predictions = {
        "base": base,
        "base_only": base + DOSE * base_correction,
        "hn_residual": base + DOSE * hn_correction,
        "alignment_shuffle": base + DOSE * shuffle_correction,
    }
    metrics = {name: _metrics(test, prediction) for name, prediction in predictions.items()}
    report = {
        "endpoint": endpoint,
        "candidate_rmse": _metrics(test, test["hn"].to_numpy(np.float64))["rmse"],
        "base_rmse": metrics["base"]["rmse"],
        "base_only_rmse": metrics["base_only"]["rmse"],
        "rmse": metrics["hn_residual"]["rmse"],
        "gain": metrics["base"]["rmse"] - metrics["hn_residual"]["rmse"],
        "increment_vs_base_only": metrics["base_only"]["rmse"] - metrics["hn_residual"]["rmse"],
        "shuffle_rmse": metrics["alignment_shuffle"]["rmse"],
        "shuffle_loss": metrics["alignment_shuffle"]["rmse"] - metrics["hn_residual"]["rmse"],
        "far_gain": metrics["base"]["far_gt2000_rmse"] - metrics["hn_residual"]["far_gt2000_rmse"],
        "p95_change": metrics["hn_residual"]["p95_well_rmse"] - metrics["base"]["p95_well_rmse"],
    }
    reports.append(report)
    if endpoint == PRIMARY_ENDPOINT:
        saved = test[["well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
        saved["base"] = predictions["base"].astype(np.float32)
        saved["hn_deployment"] = test["hn"].to_numpy(np.float32)
        saved["base_only_residual"] = predictions["base_only"].astype(np.float32)
        saved["hn_deployment_residual"] = predictions["hn_residual"].astype(np.float32)

summary = pd.DataFrame(reports)
primary = summary.loc[summary["endpoint"].eq(PRIMARY_ENDPOINT)].iloc[0]
checks = {
    "train_wells_ge_600": int(train["well"].nunique()) >= 600,
    "deployment_wells_ge_145": int(test_primary["well"].nunique()) >= 145,
    "physical_well_overlap_is_zero": len(overlap) == 0,
    "score_mean_gain_ge_0p10": float(primary["gain"]) >= 0.10,
    "score_mean_increment_vs_base_only_ge_0p05": float(primary["increment_vs_base_only"]) >= 0.05,
    "score_mean_shuffle_loses_ge_0p05": float(primary["shuffle_loss"]) >= 0.05,
    "score_mean_far_gain_ge_0p10": float(primary["far_gain"]) >= 0.10,
    "score_mean_p95_change_le_0p20": float(primary["p95_change"]) <= 0.20,
}
strict_pass = all(checks.values())
summary.to_csv(OUT_DIR / "deployment_endpoint_summary.csv", index=False)
saved.to_parquet(OUT_DIR / "hn15_score_mean_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "hn14_residual_transfer_to_exact_hn13_deployment_aggregation",
    "selection_or_tuning_count": 0,
    "protocol": {
        "residual_train": "HN3 folds 2-5 crossfit wells",
        "evaluation": "HN12 HN-fold-1 wells, unseen by all four deployment HN models",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "diagnostic_endpoints_not_selected": list(DIAGNOSTIC_ENDPOINTS),
        "correction_cap_ft": CORRECTION_CAP_FT,
        "dose": DOSE,
    },
    "sources": {"train": TRAIN_PATH, "deployment": DEPLOYMENT_PATH, "v66": V66_PATH},
    "coverage": {
        "train_rows": len(train),
        "train_wells": int(train["well"].nunique()),
        "evaluation_rows": len(test_primary),
        "evaluation_wells": int(test_primary["well"].nunique()),
        "physical_well_overlap": len(overlap),
    },
    "features": {"base_only": base_features, "with_hn": hn_features},
    "primary": primary.to_dict(),
    "diagnostics": summary.to_dict(orient="records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> package HN14 residual model with exact HN13 score_mean inference"
        if strict_pass
        else "FAIL -> HN14 gain does not transfer to deployment aggregation; close residual branch"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nHN15 DEPLOYMENT ENDPOINT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nHN15 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
