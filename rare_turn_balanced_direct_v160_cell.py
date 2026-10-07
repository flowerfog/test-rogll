"""V160: rare-turn and equal-well weighted matched A/B against V158."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
DATA_PATH = ROOT / "wellbore-geology-prediction-artifacts" / "data" / "train.csv"
FOLD_PATH = OOF_ROOT / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet"
V158_PATH = OOF_ROOT / "raw_full_feature_direct_v158" / "raw_direct_oof.parquet"
OUT_DIR = OOF_ROOT / "rare_turn_balanced_direct_v160"
OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (DATA_PATH, FOLD_PATH, V158_PATH):
    if not path.exists(): raise FileNotFoundError(path)


def _metrics(frame, column):
    error = frame[column].to_numpy(float) - frame["target"].to_numpy(float)
    distance = frame["md_since"].to_numpy(float)
    wr = pd.Series(error * error).groupby(frame["well"].to_numpy()).mean().pow(0.5)
    far = distance > 2000.0
    return {"rows": len(error), "wells": frame["well"].nunique(), "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(wr.median()), "p95_well_rmse": float(wr.quantile(0.95)), "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2)))}


folds = pd.read_parquet(FOLD_PATH, columns=["id", "well", "row_idx", "fold"])
folds["id"], folds["well"] = folds["id"].astype(str), folds["well"].astype(str)
data = pd.read_csv(DATA_PATH, low_memory=False)
data["id"], data["well"] = data["id"].astype(str), data["well"].astype(str)
data = data.merge(folds, on=["id", "well"], how="inner", validate="one_to_one")
if len(data) != 3_783_989 or data["well"].nunique() != 773: raise RuntimeError("Canonical alignment failed")
excluded = {"id", "well", "fold", "row_idx", "target", "tvt", "TVT"}
features = [c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]
data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
data = data.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

# Labels are deltas from the visible anchor.  Local absolute-TV T derivatives
# equal local delta derivatives because the anchor is constant within a well.
group = data.groupby("well", sort=False)
delta_target = group["target"].diff().fillna(0.0).to_numpy(float)
delta_md = group["md_since"].diff().replace(0.0, np.nan).fillna(1.0).to_numpy(float)
local_slope = delta_target / np.maximum(np.abs(delta_md), 1e-6)
local_change = pd.Series(local_slope).groupby(data["well"], sort=False).diff().fillna(0.0).to_numpy(float)
turn_score = np.maximum(np.abs(local_slope), 32.0 * np.abs(local_change))

reference = pd.read_parquet(V158_PATH, columns=["id", "prediction"])
reference["id"] = reference["id"].astype(str)
reference = data[["id"]].merge(reference, on="id", validate="one_to_one")
pred = np.full(len(data), np.nan, float)
reports = []
for heldout in (1, 2, 3, 4, 5):
    train = data["fold"].to_numpy(int) != heldout
    valid = ~train
    train_score = turn_score[train]
    q90, q98 = np.quantile(train_score, [0.90, 0.98])
    train_counts = data.loc[train].groupby("well").size()
    equal_well = np.array(
        data.loc[train, "well"].map((1.0 / train_counts).to_dict()),
        dtype=np.float64,
        copy=True,
    )
    equal_well /= np.mean(equal_well)
    score = train_score
    event_weight = 1.0 + 1.5 * (score >= q90) + 3.5 * (score >= q98)
    weight = np.array(
        np.clip(equal_well * event_weight, 0.10, 12.0),
        dtype=np.float64,
        copy=True,
    )
    weight /= np.mean(weight)
    model = lgb.LGBMRegressor(objective="huber", alpha=0.85, n_estimators=2500, learning_rate=0.03, num_leaves=127, min_child_samples=200, subsample=0.85, colsample_bytree=0.75, reg_lambda=8.0, random_state=160 + heldout, n_jobs=-1)
    model.fit(data.loc[train, features], data.loc[train, "target"], sample_weight=weight, eval_set=[(data.loc[valid, features], data.loc[valid, "target"])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pred[valid] = model.predict(data.loc[valid, features])
    local = data.loc[valid, ["id", "well", "target", "md_since"]].copy()
    local["v158"] = reference.loc[valid, "prediction"].to_numpy(float)
    local["v160"] = pred[valid]
    old, new = _metrics(local, "v158"), _metrics(local, "v160")
    reports.append({"fold": heldout, "q90": q90, "q98": q98, "best_iteration": int(model.best_iteration_ or model.n_estimators), "v158_rmse": old["rmse"], "v160_rmse": new["rmse"], "gain": old["rmse"] - new["rmse"], "far_gain": old["far_gt2000_rmse"] - new["far_gt2000_rmse"]})
    print("V160 fold", reports[-1], flush=True)

output = data[["id", "well", "row_idx", "fold", "target", "last_known_tvt", "md_since"]].copy()
output["v158"] = reference["prediction"].to_numpy(float)
output["v160"] = pred
output["prediction_absolute"] = output["last_known_tvt"].to_numpy(float) + pred
output.to_parquet(OUT_DIR / "balanced_direct_oof.parquet", index=False)
old, new = _metrics(output, "v158"), _metrics(output, "v160")
summary = pd.DataFrame([{"endpoint": "v158", **old}, {"endpoint": "v160", **new}])
summary.to_csv(OUT_DIR / "summary.csv", index=False)
gains = [x["gain"] for x in reports]
verdict = {"experiment": "rare_turn_balanced_direct_v160", "stage": "matched_V158_features_model_equalwell_rareturn_weighting", "features": len(features), "fold_reports": reports, "summary": summary.to_dict(orient="records"), "gain": old["rmse"] - new["rmse"], "far_gain": old["far_gt2000_rmse"] - new["far_gt2000_rmse"], "checks": {"gain_ge_0p30": bool(old["rmse"] - new["rmse"] >= 0.30), "positive_folds_4of5": bool(sum(x > 0 for x in gains) >= 4), "far_gain_positive": bool(old["far_gt2000_rmse"] - new["far_gt2000_rmse"] > 0)}, "strict_pass": bool(old["rmse"] - new["rmse"] >= 0.30 and sum(x > 0 for x in gains) >= 4 and old["far_gt2000_rmse"] - new["far_gt2000_rmse"] > 0), "next": "PASS -> run exact-ID V144 nested blend" if old["rmse"] - new["rmse"] >= 0.30 and sum(x > 0 for x in gains) >= 4 else "FAIL -> rare-turn imbalance is not the missing mechanism", "elapsed_minutes": (time.time() - STARTED) / 60.0}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding="utf-8")
print("V160 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("V160 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x), flush=True)
