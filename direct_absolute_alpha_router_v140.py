"""V140: well-crossfit observability gate for the V139 per-well direction.

The router consumes only deployable summaries of V127, V139, their difference,
and measured-distance progression. Truth defines the training alpha on outer
training wells only. V140 is an investment gate, not a deployment artifact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
SEED = 20260802
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = (
    ROOT / "oof_exports" / "prefix_conditioned_direct_absolute_trajectory_v139"
    / "fold3_confirmation_oof.parquet"
)
OUT_DIR = ROOT / "oof_exports" / "direct_absolute_alpha_router_v140"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRUTH = "tvt_true"
BASE = "nested_sequence_coefficient"
DIRECT = "direct_absolute_trajectory"
ALPHA_LIMIT = 0.15


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
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_{name}": 0.0 for name in (
            "mean", "std", "rms", "q10", "q25", "q50", "q75", "q90", "min", "max"
        )}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(finite)))),
        f"{prefix}_q10": float(np.quantile(finite, 0.10)),
        f"{prefix}_q25": float(np.quantile(finite, 0.25)),
        f"{prefix}_q50": float(np.quantile(finite, 0.50)),
        f"{prefix}_q75": float(np.quantile(finite, 0.75)),
        f"{prefix}_q90": float(np.quantile(finite, 0.90)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _slope(values, distance):
    values = np.asarray(values, np.float64)
    x = np.asarray(distance, np.float64)
    x = (x - x.mean()) / max(float(x.std()), 1.0)
    return float(np.dot(x, values - values.mean()) / max(np.dot(x, x), 1e-12))


def _well_features(local):
    base = local[BASE].to_numpy(np.float64)
    direct = local[DIRECT].to_numpy(np.float64)
    move = direct - base
    distance = local["md_since"].to_numpy(np.float64)
    result = {
        "well": str(local["well"].iloc[0]),
        "rows": len(local),
        "maximum_distance": float(np.max(distance)),
        "move_slope": _slope(move, distance),
        "base_slope": _slope(base, distance),
        "direct_slope": _slope(direct, distance),
        "move_diff_rms": float(np.sqrt(np.mean(np.square(np.diff(move))))) if len(move) > 1 else 0.0,
        "move_curvature_rms": float(np.sqrt(np.mean(np.square(np.diff(move, 2))))) if len(move) > 2 else 0.0,
        "base_diff_rms": float(np.sqrt(np.mean(np.square(np.diff(base))))) if len(base) > 1 else 0.0,
        "direct_diff_rms": float(np.sqrt(np.mean(np.square(np.diff(direct))))) if len(direct) > 1 else 0.0,
        "move_sign_change_fraction": float(np.mean(np.diff(np.signbit(move)) != 0)) if len(move) > 1 else 0.0,
        "base_direct_correlation": float(np.corrcoef(base, direct)[0, 1]) if np.std(base) > 0 and np.std(direct) > 0 else 0.0,
    }
    result.update(_summary(base - base[0], "base"))
    result.update(_summary(direct - direct[0], "direct"))
    result.update(_summary(move, "move"))
    for label, mask in {
        "near": distance < 500.0,
        "mid1": (distance >= 500.0) & (distance < 1000.0),
        "mid2": (distance >= 1000.0) & (distance < 2000.0),
        "far": distance >= 2000.0,
    }.items():
        result.update(_summary(move[mask], f"move_{label}"))
    return result


def _target(local):
    truth = local[TRUTH].to_numpy(np.float64)
    base = local[BASE].to_numpy(np.float64)
    move = local[DIRECT].to_numpy(np.float64) - base
    error = base - truth
    curvature = float(np.dot(move, move))
    alpha = 0.0 if curvature <= 1e-12 else float(np.clip(-np.dot(error, move) / curvature, -ALPHA_LIMIT, ALPHA_LIMIT))
    return alpha, curvature, float(np.dot(error, error))


def _models():
    return {
        "ridge10": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
        "hist_leaf12": make_pipeline(
            SimpleImputer(),
            HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.04, max_iter=160,
                max_leaf_nodes=7, min_samples_leaf=12, l2_regularization=4.0,
                random_state=SEED,
            ),
        ),
        "extra_leaf8": make_pipeline(
            SimpleImputer(),
            ExtraTreesRegressor(
                n_estimators=400, min_samples_leaf=8, max_features=0.7,
                random_state=SEED, n_jobs=-1,
            ),
        ),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE)
frame["well"] = frame["well"].astype(str)
required = {"id", "well", "row_idx", TRUTH, "md_since", BASE, DIRECT}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V139 OOF missing columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

panel_rows = []
for well, local in frame.groupby("well", sort=True):
    row = _well_features(local)
    row["oracle_alpha"], row["curvature"], row["base_sse"] = _target(local)
    panel_rows.append(row)
panel = pd.DataFrame(panel_rows)
feature_columns = [column for column in panel.columns if column not in {
    "well", "oracle_alpha", "curvature", "base_sse"
}]
X = panel[feature_columns].replace([np.inf, -np.inf], np.nan)
y = panel["oracle_alpha"].to_numpy(np.float64)
sample_weight = np.sqrt(np.maximum(panel["curvature"].to_numpy(np.float64), 1.0))

kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
predictions = {name: np.full(len(panel), np.nan, np.float64) for name in _models()}
fit_rows = []
for outer, (train, valid) in enumerate(kfold.split(panel), 1):
    for name, model in _models().items():
        fit_kwargs = {}
        final_name = model.steps[-1][0]
        fit_kwargs[f"{final_name}__sample_weight"] = sample_weight[train]
        model.fit(X.iloc[train], y[train], **fit_kwargs)
        predictions[name][valid] = np.clip(model.predict(X.iloc[valid]), -ALPHA_LIMIT, ALPHA_LIMIT)
        fit_rows.append({"outer": outer, "model": name, "train_wells": len(train), "valid_wells": len(valid)})

base = frame[BASE].to_numpy(np.float64)
move = frame[DIRECT].to_numpy(np.float64) - base
base_metric = _metric(frame, base, BASE)
well_to_position = {well: position for position, well in enumerate(panel["well"])}
row_position = frame["well"].map(well_to_position).to_numpy(np.int64)
summary_rows = [dict(base_metric, model="base", gain=0.0)]
candidate_predictions = {}
for name, alpha in predictions.items():
    prediction = base + alpha[row_position] * move
    candidate_predictions[name] = prediction
    metric = _metric(frame, prediction, name)
    metric.update(
        model=name,
        gain=base_metric["rmse"] - metric["rmse"],
        far_gain=base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        p95_gain=base_metric["p95_well_rmse"] - metric["p95_well_rmse"],
    )
    summary_rows.append(metric)

summary = pd.DataFrame(summary_rows).sort_values("rmse", kind="stable")
best_name = str(summary[summary["model"].ne("base")].iloc[0]["model"])
best_alpha = predictions[best_name]
rng = np.random.default_rng(SEED)
shuffle_gains = []
for draw in range(32):
    shuffled = rng.permutation(best_alpha)
    metric = _metric(frame, base + shuffled[row_position] * move, f"shuffle_{draw}")
    shuffle_gains.append(base_metric["rmse"] - metric["rmse"])

panel_output = panel[["well", "oracle_alpha", "curvature", "base_sse"]].copy()
for name, values in predictions.items():
    panel_output[f"predicted_alpha_{name}"] = values
panel_output.to_csv(OUT_DIR / "router_oof.csv", index=False)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(fit_rows).to_csv(OUT_DIR / "fit_reports.csv", index=False)

best_row = summary[summary["model"].eq(best_name)].iloc[0]
aligned_gain = float(best_row["gain"])
shuffle_p95 = float(np.quantile(shuffle_gains, 0.95))
verdict = {
    "experiment": "direct_absolute_alpha_router_v140",
    "stage": "fivefold_well_crossfit_V139_direction_observability_gate",
    "deployable": False,
    "source_wells": len(panel),
    "features": len(feature_columns),
    "alpha_limit": ALPHA_LIMIT,
    "candidate_models": list(predictions),
    "selected_model": best_name,
    "values": {
        "crossfit_gain": aligned_gain,
        "far_gain": float(best_row["far_gain"]),
        "p95_gain": float(best_row["p95_gain"]),
        "shuffle_gain_p95": shuffle_p95,
        "aligned_increment_vs_shuffle_p95": aligned_gain - shuffle_p95,
        "alpha_spearman": float(pd.Series(best_alpha).corr(pd.Series(y), method="spearman")),
        "alpha_sign_accuracy": float(np.mean(np.sign(best_alpha) == np.sign(y))),
    },
    "checks": {
        "crossfit_gain_ge_0p10": aligned_gain >= 0.10,
        "far_gain_positive": float(best_row["far_gain"]) > 0.0,
        "p95_not_worse_by_gt_0p10": float(best_row["p95_gain"]) >= -0.10,
        "aligned_beats_shuffle_p95_ge_0p05": aligned_gain - shuffle_p95 >= 0.05,
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> reconstruct V139 across canonical folds and confirm the frozen router"
    if verdict["strict_pass"] else
    "FAIL -> per-well V139 oracle direction is not observable; close V139 completely"
)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(_clean(verdict), handle, indent=2, allow_nan=False)

print("V140 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV140 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
