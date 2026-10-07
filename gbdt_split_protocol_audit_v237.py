"""V237: compare row-random, well-held-out, and within-well-tail CV protocols.

The model, features, residual target, and parent are frozen.  Only the split
protocol changes.  This is an audit of CV comparability, not a submission model.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


ROOT = Path("/root/autodl-tmp")
V217 = ROOT / "oof_exports" / "likpf_residual_foldclean_spatial_v217"
V215 = ROOT / "oof_exports" / "likpf_residual_gbdt_full5_v215"
SPATIAL = ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0"
OUT = ROOT / "oof_exports" / "gbdt_split_protocol_audit_v237"
SEED = 237
MAX_ROWS_PER_WELL = 1024
N_SPLITS = 5


def rmse(y, p):
    return float(np.sqrt(np.mean(np.square(np.asarray(y, float) - np.asarray(p, float)))))


def load_data():
    required = (
        V217 / "base_features.csv", V217 / "spatial_features.csv",
        V215 / "well_metadata.csv", V215 / "features", SPATIAL / "features",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    base_features = pd.read_csv(V217 / "base_features.csv")["feature"].astype(str).tolist()
    spatial_features = pd.read_csv(V217 / "spatial_features.csv")["feature"].astype(str).tolist()
    features = base_features + spatial_features
    meta = pd.read_csv(V215 / "well_metadata.csv", dtype={"well": str})
    parts = []
    for number, well in enumerate(meta.well.astype(str), 1):
        base_path = V215 / "features" / f"{well}__sample.pkl"
        spatial_path = SPATIAL / "features" / f"{well}__sample.pkl"
        for path in (base_path, spatial_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        frame = pd.read_pickle(base_path)
        spatial = pd.read_pickle(spatial_path)
        key = "id" if "id" in frame and "id" in spatial else "_row_idx"
        if frame[key].duplicated().any() or spatial[key].duplicated().any():
            raise RuntimeError(f"{well}: duplicate alignment key {key}")
        aligned = spatial.set_index(key)[spatial_features].reindex(frame[key].to_numpy())
        if aligned.isna().all(axis=1).any():
            raise RuntimeError(f"{well}: incomplete spatial feature alignment")
        frame[spatial_features] = aligned.to_numpy(np.float32)
        if len(frame) > MAX_ROWS_PER_WELL:
            positions = np.unique(np.round(np.linspace(0, len(frame) - 1, MAX_ROWS_PER_WELL)).astype(int))
            frame = frame.iloc[positions].copy()
        frame["well"] = str(well)
        frame["fold"] = int(meta.loc[meta.well.eq(well), "fold"].iloc[0])
        parts.append(frame)
        if number % 100 == 0 or number == len(meta):
            print(f"V237 loading {number}/{len(meta)}", flush=True)
    data = pd.concat(parts, ignore_index=True, sort=False)
    required_columns = {"well", "fold", "target", "last_known_tvt", "likpf_scale_5", "md_since", "residual_target"}
    if not required_columns.issubset(data.columns):
        raise RuntimeError(f"missing columns: {sorted(required_columns - set(data.columns))}")
    data = data.replace([np.inf, -np.inf], np.nan)
    data[features] = data[features].fillna(0.0).astype(np.float32)
    return data, features


def model_params():
    params = dict(
        objective="regression",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=10.0,
        max_bin=255,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
    )
    if os.environ.get("V237_CPU", "0") != "1":
        params["device_type"] = "gpu"
    return params


def evaluate_protocol(data, features, protocol, splits):
    from lightgbm import LGBMRegressor, early_stopping

    x = data[features].to_numpy(np.float32)
    y = data["residual_target"].to_numpy(np.float32)
    parent = data["likpf_scale_5"].to_numpy(float)
    last = data["last_known_tvt"].to_numpy(float)
    truth = last + data["target"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(data["md_since"].to_numpy(float), 0.0) / 85.0)
    warm = np.maximum(warm, 0.15)
    fold_rows = []
    all_predictions = np.full(len(data), np.nan, float)
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        model = LGBMRegressor(**model_params())
        try:
            model.fit(
                x[train_idx], y[train_idx],
                sample_weight=data["sample_weight"].to_numpy(float)[train_idx] if "sample_weight" in data else None,
                eval_set=[(x[valid_idx], y[valid_idx])],
                callbacks=[early_stopping(100, verbose=False)],
            )
        except Exception as error:
            if "gpu" not in str(error).lower() and "opencl" not in str(error).lower():
                raise
            print(f"V237 CPU fallback {protocol} fold {fold}: {error!r}", flush=True)
            os.environ["V237_CPU"] = "1"
            model = LGBMRegressor(**model_params())
            model.fit(
                x[train_idx], y[train_idx],
                sample_weight=data["sample_weight"].to_numpy(float)[train_idx] if "sample_weight" in data else None,
                eval_set=[(x[valid_idx], y[valid_idx])],
                callbacks=[early_stopping(100, verbose=False)],
            )
        correction = model.predict(x[valid_idx], num_iteration=model.best_iteration_)
        prediction = parent[valid_idx] + warm[valid_idx] * correction
        all_predictions[valid_idx] = prediction
        fold_rows.append({
            "protocol": protocol,
            "fold": fold,
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "valid_wells": int(data.iloc[valid_idx].well.nunique()),
            "best_iteration": int(model.best_iteration_),
            "parent_rmse": rmse(truth[valid_idx], parent[valid_idx]),
            "candidate_rmse": rmse(truth[valid_idx], prediction),
            "gain": rmse(truth[valid_idx], parent[valid_idx]) - rmse(truth[valid_idx], prediction),
        })
        print("V237 fold", fold_rows[-1], flush=True)
        del model
        gc.collect()
    summary = {
        "protocol": protocol,
        "rows": len(data),
        "wells": int(data.well.nunique()),
        "parent_rmse": rmse(truth, parent),
        "candidate_rmse": rmse(truth, all_predictions),
        "gain": rmse(truth, parent) - rmse(truth, all_predictions),
        "positive_folds": int(sum(row["gain"] > 0 for row in fold_rows)),
        "minimum_fold_gain": float(min(row["gain"] for row in fold_rows)),
    }
    return summary, fold_rows, all_predictions


def main():
    started = time.time()
    data, features = load_data()
    print("V237 DATA", {"rows": len(data), "wells": data.well.nunique(), "features": len(features)}, flush=True)
    groups = data.well.to_numpy()
    indices = np.arange(len(data))

    group_splits = list(GroupKFold(N_SPLITS).split(indices, groups=groups))
    random_splits = list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(indices))
    tail_splits = []
    for held in range(1, N_SPLITS + 1):
        valid_mask = data.groupby("well", sort=False)["md_since"].transform(lambda values: values >= values.quantile(0.80)).to_numpy() & data.fold.eq(held).to_numpy()
        valid_idx = np.flatnonzero(valid_mask)
        train_idx = np.flatnonzero(~valid_mask)
        tail_splits.append((train_idx, valid_idx))

    summaries, fold_rows = [], []
    for protocol, splits in (("random_rows", random_splits), ("well_group", group_splits), ("held_fold_tail20", tail_splits)):
        summary, folds, _ = evaluate_protocol(data, features, protocol, splits)
        summaries.append(summary)
        fold_rows.extend(folds)

    summary_frame = pd.DataFrame(summaries)
    fold_frame = pd.DataFrame(fold_rows)
    verdict = {
        "experiment": "gbdt_split_protocol_audit_v237",
        "stage": "same_V217_features_and_target_three_validation_protocols",
        "features": len(features),
        "sample_rows_per_well": MAX_ROWS_PER_WELL,
        "protocols": summary_frame.to_dict("records"),
        "folds": fold_frame.to_dict("records"),
        "checks": {
            "random_rows_rmse_le_7": bool(summary_frame.loc[summary_frame.protocol.eq("random_rows"), "candidate_rmse"].iloc[0] <= 7.0),
            "well_group_rmse_gt_random_by_0p5": bool(
                summary_frame.loc[summary_frame.protocol.eq("well_group"), "candidate_rmse"].iloc[0]
                - summary_frame.loc[summary_frame.protocol.eq("random_rows"), "candidate_rmse"].iloc[0] >= 0.5
            ),
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    verdict["next"] = "Random-row CV is not comparable to well-held-out CV" if verdict["checks"]["well_group_rmse_gt_random_by_0p5"] else "Split protocol does not explain the GBDT gap"
    OUT.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(OUT / "summary.csv", index=False)
    fold_frame.to_csv(OUT / "folds.csv", index=False)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("\nV237 SUMMARY\n", summary_frame.to_string(index=False))
    print("\nV237 FINAL VERDICT\n", json.dumps(verdict, indent=2))
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
