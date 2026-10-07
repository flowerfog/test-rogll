"""V158: independent full-feature direct TVT model on canonical 773 wells.

This is a new upstream predictor, not a correction to V76.  Training and
validation are split by well; no historical prediction column is used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise RuntimeError("V158 requires lightgbm") from exc

STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
DATA_CANDIDATES = (
    ROOT / "wellbore-geology-prediction-artifacts" / "data" / "train.csv",
    ROOT / "data" / "train.csv",
    Path.cwd() / "wellbore-geology-prediction-artifacts" / "data" / "train.csv",
)
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), None)
FOLD_CANDIDATES = (
    OOF_ROOT / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet",
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / "raw_full_feature_direct_v158"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if DATA_PATH is None or FOLD_PATH is None:
    raise FileNotFoundError({"data": DATA_CANDIDATES, "fold": FOLD_CANDIDATES})


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _metrics(frame, pred):
    error = frame[pred].to_numpy(float) - frame["target"].to_numpy(float)
    distance = frame["md_since"].to_numpy(float)
    result = {"rows": len(error), "wells": int(frame["well"].nunique()), "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error)))}
    well_rmse = pd.Series(error * error).groupby(frame["well"].to_numpy()).mean().pow(0.5)
    result["median_well_rmse"] = float(well_rmse.median())
    result["p95_well_rmse"] = float(well_rmse.quantile(0.95))
    for name, low, high in (("near_0_500", 0.0, 500.0), ("mid_500_1000", 500.0, 1000.0), ("mid_1000_2000", 1000.0, 2000.0), ("far_gt2000", 2000.0, np.inf)):
        mask = (distance >= low) & (distance < high)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


folds = pd.read_parquet(FOLD_PATH, columns=["id", "well", "fold"])
folds["id"] = folds["id"].astype(str)
folds["well"] = folds["well"].astype(str)
folds = folds[["id", "well", "fold"]].drop_duplicates("id")
data = pd.read_csv(DATA_PATH, low_memory=False)
data["id"] = data["id"].astype(str)
data["well"] = data["well"].astype(str)
if "target" not in data or "last_known_tvt" not in data:
    raise RuntimeError(f"Expected target and last_known_tvt in {DATA_PATH}")
data = data.merge(folds, on=["id", "well"], how="inner", validate="one_to_one")
if data["fold"].isna().any() or data["well"].nunique() != 773:
    raise RuntimeError(f"Incomplete canonical alignment: rows={len(data)} wells={data['well'].nunique()}")

excluded = {"id", "well", "fold", "target", "tvt", "TVT"}
feature_columns = [column for column in data.columns if column not in excluded and pd.api.types.is_numeric_dtype(data[column])]
if len(feature_columns) < 50:
    raise RuntimeError(f"Only {len(feature_columns)} numeric features")
data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
data["target"] = pd.to_numeric(data["target"], errors="coerce")
data = data.loc[data["target"].notna()].reset_index(drop=True)

pred = np.full(len(data), np.nan, np.float64)
fold_reports = []
for heldout in (1, 2, 3, 4, 5):
    train_mask = data["fold"].to_numpy(int) != heldout
    valid_mask = ~train_mask
    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.85,
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=8.0,
        random_state=158 + heldout,
        n_jobs=-1,
    )
    model.fit(
        data.loc[train_mask, feature_columns],
        data.loc[train_mask, "target"],
        eval_set=[(data.loc[valid_mask, feature_columns], data.loc[valid_mask, "target"])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred[valid_mask] = model.predict(data.loc[valid_mask, feature_columns])
    valid = data.loc[valid_mask, ["id", "well", "target", "md_since"]].copy()
    valid["prediction"] = pred[valid_mask]
    report = {"fold": heldout, "train_wells": int(data.loc[train_mask, "well"].nunique()), "valid_wells": int(data.loc[valid_mask, "well"].nunique()), "best_iteration": int(model.best_iteration_ or model.n_estimators), **_metrics(valid, "prediction")}
    fold_reports.append(report)
    print("V158 fold", heldout, report, flush=True)

output = data[["id", "well", "fold", "target", "last_known_tvt", "md_since"]].copy()
output["prediction"] = pred
output["prediction_absolute"] = output["last_known_tvt"].to_numpy(float) + pred
output.to_parquet(OUT_DIR / "raw_direct_oof.parquet", index=False)
summary = pd.DataFrame([{**{"endpoint": "raw_direct"}, **_metrics(output, "prediction")}])
summary.to_csv(OUT_DIR / "summary.csv", index=False)
verdict = {"experiment": "raw_full_feature_direct_v158", "stage": "canonical_773_well_group_crossfit", "data": str(DATA_PATH), "fold_source": str(FOLD_PATH), "features": len(feature_columns), "feature_names": feature_columns, "fold_reports": fold_reports, "summary": summary.to_dict(orient="records"), "gates": {"coverage_773": bool(output["well"].nunique() == 773), "standalone_rmse_le_10p5": bool(float(summary.loc[0, "rmse"]) <= 10.5)}, "next": "Compare raw_direct against V76 on exact IDs; only then test a nested blend" if float(summary.loc[0, "rmse"]) <= 10.5 else "FAIL -> full raw-feature direct model is not a viable independent generator", "elapsed_minutes": (time.time() - STARTED) / 60.0}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
print("V158 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("V158 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
