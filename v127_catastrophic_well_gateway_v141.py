"""V141: cross-fit catastrophic-well detection and existing-endpoint rescue gate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
SEED = 20260802
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE_DIR = ROOT / "oof_exports" / "final_consensus_sequence_coefficient_crossfit_v127"
OOF_PATH = SOURCE_DIR / "crossfit_oof.parquet"
COEFFICIENT_PATH = SOURCE_DIR / "crossfit_coefficients.csv"
OUT_DIR = ROOT / "oof_exports" / "v127_catastrophic_well_gateway_v141"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRUTH = "tvt_true"
BASE = "nested_sequence_coefficient"
ARMS = ("consensus_center", "frozen_v126_recipe", "rolled_sequence_control")
TOP_K = (13, 25, 46)
RETAIN_FRACTIONS = (0.03, 0.05, 0.10)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _metric(frame, prediction, endpoint):
    truth = frame[TRUTH].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    well_mse = pd.Series(np.square(error), index=frame["well"]).groupby(level=0).mean()
    result = {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
    }
    for name, mask in {
        "near_0_500_rmse": distance < 500.0,
        "mid_500_1000_rmse": (distance >= 500.0) & (distance < 1000.0),
        "mid_1000_2000_rmse": (distance >= 1000.0) & (distance < 2000.0),
        "far_gt2000_rmse": distance >= 2000.0,
    }.items():
        result[name] = float(np.sqrt(np.mean(np.square(error[mask])))) if mask.any() else np.nan
    return result


def _summary(values, prefix):
    values = np.asarray(values, np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _well_features(local):
    distance = local["md_since"].to_numpy(np.float64)
    base = local[BASE].to_numpy(np.float64)
    result = {
        "well": str(local["well"].iloc[0]),
        "fold": int(local["fold"].iloc[0]),
        "rows": len(local),
        "maximum_distance": float(np.max(distance)),
        "base_range": float(np.ptp(base)),
        "base_diff_rms": float(np.sqrt(np.mean(np.square(np.diff(base))))) if len(base) > 1 else 0.0,
        "base_curvature_rms": float(np.sqrt(np.mean(np.square(np.diff(base, 2))))) if len(base) > 2 else 0.0,
    }
    for arm in ARMS:
        candidate = local[arm].to_numpy(np.float64)
        move = candidate - base
        result.update(_summary(move, f"{arm}_move"))
        far = distance >= 2000.0
        if far.any():
            result.update(_summary(move[far], f"{arm}_far_move"))
        result[f"{arm}_diff_rms"] = float(np.sqrt(np.mean(np.square(np.diff(move))))) if len(move) > 1 else 0.0
        result[f"{arm}_correlation"] = (
            float(np.corrcoef(base, candidate)[0, 1])
            if np.std(base) > 0 and np.std(candidate) > 0 else 0.0
        )
    truth = local[TRUTH].to_numpy(np.float64)
    error = base - truth
    result["target_sse"] = float(np.dot(error, error))
    result["target_mse"] = float(np.mean(np.square(error)))
    for arm in ARMS:
        arm_error = local[arm].to_numpy(np.float64) - truth
        result[f"gain_sse_{arm}"] = result["target_sse"] - float(np.dot(arm_error, arm_error))
    return result


def _models():
    return {
        "ridge100": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=100.0)),
        "hist_leaf16": make_pipeline(
            SimpleImputer(),
            HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=180, max_leaf_nodes=7,
                min_samples_leaf=16, l2_regularization=8.0, random_state=SEED,
            ),
        ),
        "extra_leaf10": make_pipeline(
            SimpleImputer(),
            ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=10, max_features=0.7,
                n_jobs=-1, random_state=SEED,
            ),
        ),
    }


for path in (OOF_PATH, COEFFICIENT_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)
frame = pd.read_parquet(OOF_PATH)
frame["well"] = frame["well"].astype(str)
required = {"id", "well", "row_idx", "fold", TRUTH, "md_since", BASE, *ARMS}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V127 OOF missing columns: {missing}")
if frame["id"].duplicated().any() or frame["well"].nunique() != 464:
    raise RuntimeError("Invalid V127 OOF coverage")
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)

panel = pd.DataFrame([_well_features(local) for _, local in frame.groupby("well", sort=True)])
coefficients = pd.read_csv(COEFFICIENT_PATH, dtype={"well": str})
coefficients["well"] = coefficients["well"].astype(str)
predicted_coefficients = [column for column in coefficients if column.startswith("pred_")]
if not predicted_coefficients:
    raise RuntimeError("V127 coefficient panel has no predicted coefficients")
panel = panel.merge(
    coefficients[["well", *predicted_coefficients]], on="well", how="left", validate="one_to_one"
)
if panel[predicted_coefficients].isna().any().any():
    raise RuntimeError("Incomplete V127 coefficient merge")

excluded = {
    "well", "fold", "target_sse", "target_mse",
    *(f"gain_sse_{arm}" for arm in ARMS),
}
features = [column for column in panel if column not in excluded]
X = panel[features].replace([np.inf, -np.inf], np.nan)
y = np.log1p(panel["target_mse"].to_numpy(np.float64))
folds = panel["fold"].to_numpy(np.int8)
scores = {name: np.full(len(panel), np.nan, np.float64) for name in _models()}
fit_rows = []
for heldout in (1, 2, 3):
    train = np.flatnonzero(folds != heldout)
    valid = np.flatnonzero(folds == heldout)
    for name, model in _models().items():
        model.fit(X.iloc[train], y[train])
        scores[name][valid] = model.predict(X.iloc[valid])
        fit_rows.append({
            "heldout_fold": heldout, "model": name,
            "train_wells": len(train), "valid_wells": len(valid),
        })
if any(not np.isfinite(value).all() for value in scores.values()):
    raise RuntimeError("Incomplete V141 cross-fit risk scores")

total_sse = float(panel["target_sse"].sum())
true_order = np.argsort(-panel["target_sse"].to_numpy(np.float64), kind="stable")
risk_rows = []
for name, score in scores.items():
    order = np.argsort(-score, kind="stable")
    for k in TOP_K:
        selected = order[:k]
        truth_set = set(true_order[:k])
        risk_rows.append({
            "model": name,
            "k": k,
            "recall_at_k": len(truth_set.intersection(selected)) / k,
            "selected_sse_fraction": float(panel.iloc[selected]["target_sse"].sum() / total_sse),
            "mean_selected_rmse": float(np.mean(np.sqrt(panel.iloc[selected]["target_mse"]))),
        })
risk = pd.DataFrame(risk_rows)

base_prediction = frame[BASE].to_numpy(np.float64)
base_metric = _metric(frame, base_prediction, BASE)
well_to_panel = {well: index for index, well in enumerate(panel["well"])}
row_panel = frame["well"].map(well_to_panel).to_numpy(np.int64)
rescue_rows = [dict(base_metric, model="base", arm="base", fraction=0.0, selected_wells=0, gain=0.0)]
for name, score in scores.items():
    for fraction in RETAIN_FRACTIONS:
        count = max(1, int(round(fraction * len(panel))))
        selected = set(np.argsort(-score, kind="stable")[:count].tolist())
        row_selected = np.fromiter((position in selected for position in row_panel), bool, len(frame))
        for arm in ARMS:
            prediction = base_prediction.copy()
            prediction[row_selected] = frame.loc[row_selected, arm].to_numpy(np.float64)
            metric = _metric(frame, prediction, f"{name}_{arm}_{fraction:g}")
            metric.update(
                model=name, arm=arm, fraction=fraction, selected_wells=count,
                gain=base_metric["rmse"] - metric["rmse"],
                far_gain=base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                p95_gain=base_metric["p95_well_rmse"] - metric["p95_well_rmse"],
            )
            rescue_rows.append(metric)
rescue = pd.DataFrame(rescue_rows).sort_values("rmse", kind="stable")

# Existing-arm oracle shows whether detection could matter even with perfect routing.
oracle = base_prediction.copy()
oracle_choices = []
for well, positions in frame.groupby("well", sort=True).indices.items():
    index = np.asarray(positions, np.int64)
    truth = frame.iloc[index][TRUTH].to_numpy(np.float64)
    choices = [(float(np.sum(np.square(base_prediction[index] - truth))), "base", base_prediction[index])]
    for arm in ARMS:
        candidate = frame.iloc[index][arm].to_numpy(np.float64)
        choices.append((float(np.sum(np.square(candidate - truth))), arm, candidate))
    loss, arm, prediction = min(choices, key=lambda item: item[0])
    oracle[index] = prediction
    oracle_choices.append({"well": well, "arm": arm, "sse": loss})
oracle_metric = _metric(frame, oracle, "existing_arm_perwell_oracle")
oracle_gain = base_metric["rmse"] - oracle_metric["rmse"]

best_risk = risk[risk["k"].eq(13)].sort_values(
    ["selected_sse_fraction", "recall_at_k"], ascending=False, kind="stable"
).iloc[0]
best_rescue = rescue[rescue["model"].ne("base")].iloc[0]
risk.to_csv(OUT_DIR / "risk_capture.csv", index=False)
rescue.to_csv(OUT_DIR / "rescue_summary.csv", index=False)
pd.DataFrame(fit_rows).to_csv(OUT_DIR / "fit_reports.csv", index=False)
pd.DataFrame(oracle_choices).to_csv(OUT_DIR / "existing_arm_oracle_choices.csv", index=False)
score_panel = panel[["well", "fold", "target_sse", "target_mse"]].copy()
for name, values in scores.items():
    score_panel[f"risk_{name}"] = values
score_panel.to_csv(OUT_DIR / "risk_oof.csv", index=False)

verdict = {
    "experiment": "v127_catastrophic_well_gateway_v141",
    "stage": "threefold_crossfit_catastrophic_detection_and_existing_endpoint_rescue",
    "deployable": False,
    "coverage": {"rows": len(frame), "wells": len(panel)},
    "features": len(features),
    "models": list(scores),
    "top13": {
        "selected_model": str(best_risk["model"]),
        "recall": float(best_risk["recall_at_k"]),
        "selected_sse_fraction": float(best_risk["selected_sse_fraction"]),
    },
    "best_existing_rescue": {
        "model": str(best_rescue["model"]),
        "arm": str(best_rescue["arm"]),
        "fraction": float(best_rescue["fraction"]),
        "gain": float(best_rescue["gain"]),
        "far_gain": float(best_rescue["far_gain"]),
        "p95_gain": float(best_rescue["p95_gain"]),
    },
    "existing_arm_oracle": {
        "rmse": oracle_metric["rmse"],
        "gain": oracle_gain,
    },
    "checks": {
        "top13_recall_ge_0p30": float(best_risk["recall_at_k"]) >= 0.30,
        "top13_sse_fraction_ge_0p20": float(best_risk["selected_sse_fraction"]) >= 0.20,
        "existing_arm_oracle_gain_ge_0p20": oracle_gain >= 0.20,
        "crossfit_rescue_gain_ge_0p10": float(best_rescue["gain"]) >= 0.10,
        "crossfit_rescue_far_gain_positive": float(best_rescue["far_gain"]) > 0.0,
        "crossfit_rescue_p95_not_worse": float(best_rescue["p95_gain"]) >= 0.0,
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> confirm frozen catastrophic rescue on an untouched endpoint lineage"
    if verdict["strict_pass"] else
    "FAIL -> existing endpoints cannot deliver a sub-6 rescue; train an independent catastrophic-well model"
)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(_clean(verdict), handle, indent=2, allow_nan=False)

print("V141 RISK CAPTURE")
print(risk.sort_values(["k", "selected_sse_fraction"], ascending=[True, False]).to_string(index=False), flush=True)
print("\nV141 RESCUE TOP 20")
print(rescue.head(20).to_string(index=False), flush=True)
print("\nV141 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
