"""V95: fixed-complexity distance-specific dose on top of V76.

V74 showed that a per-well, per-distance oracle improves on the scalar
per-well oracle.  V76 only predicts one coarse-endpoint dose per well.  This
cell tests the remaining hypothesis directly: predict four residual doses
around V76 using the same deployable disagreement features, cross-fit by
complete well, and join the bands with smooth transitions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_v76_distance_dose_v95"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V74_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_oracle_v74" / "perwell_oracle_features.parquet"
V76_PATH = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
FOLDS = (1, 2, 3)
MIN_WEIGHT, MAX_WEIGHT = 0.0, 0.60
BAND_NAMES = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
BANDS = (
    ("near_0_500", -np.inf, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)
BOUNDARIES = (500.0, 1000.0, 2000.0)
TRANSITION_HALF_WIDTH = 64.0
BASE_FEATURES = (
    "maximum_distance_ft", "move_mean", "move_std", "move_rms",
    "move_abs_mean", "move_abs_p90", "move_abs_max", "move_slope_per_ft",
    "endpoint_correlation", "parent_slope_per_ft", "coarse_slope_per_ft",
    "parent_roughness", "coarse_roughness",
    "near_0_500_fraction", "near_0_500_move_rms", "near_0_500_move_mean",
    "mid_500_1000_fraction", "mid_500_1000_move_rms", "mid_500_1000_move_mean",
    "mid_1000_2000_fraction", "mid_1000_2000_move_rms", "mid_1000_2000_move_mean",
    "far_gt2000_fraction", "far_gt2000_move_rms", "far_gt2000_move_mean",
)
LOG_FEATURES = (
    "move_std", "move_rms", "move_abs_mean", "move_abs_p90", "move_abs_max",
    "parent_roughness", "coarse_roughness", "near_0_500_move_rms",
    "mid_500_1000_move_rms", "mid_1000_2000_move_rms", "far_gt2000_move_rms",
)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _optimal_weight(parent, move, truth):
    denominator = float(np.dot(move, move))
    if denominator <= 1e-10:
        return np.nan
    return float(np.clip(np.dot(move, truth - parent) / denominator, MIN_WEIGHT, MAX_WEIGHT))


def _augment(features):
    output = features.loc[:, BASE_FEATURES].astype(np.float64).copy()
    for column in LOG_FEATURES:
        output[f"log1p_{column}"] = np.log1p(np.maximum(output[column], 0.0))
    output["roughness_ratio"] = output["coarse_roughness"] / np.maximum(output["parent_roughness"], 1e-4)
    output["slope_difference"] = output["coarse_slope_per_ft"] - output["parent_slope_per_ft"]
    return output


def _hist():
    return HistGradientBoostingRegressor(
        max_iter=160, learning_rate=0.035, max_leaf_nodes=7,
        min_samples_leaf=45, l2_regularization=12.0, random_state=20260730,
    )


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


def _smoothstep(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _row_weights(distance, band_weights):
    distance = np.asarray(distance, np.float64)
    result = np.full(len(distance), band_weights[0], np.float64)
    for index, boundary in enumerate(BOUNDARIES):
        blend = _smoothstep((distance - (boundary - TRANSITION_HALF_WIDTH)) / (2.0 * TRANSITION_HALF_WIDTH))
        result = result * (1.0 - blend) + band_weights[index + 1] * blend
    return np.clip(result, MIN_WEIGHT, MAX_WEIGHT)


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - frame[TRUTH].to_numpy(np.float64)
    error2 = error * error
    bywell = np.sqrt(pd.DataFrame({"well": frame["well"], "e2": error2}).groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint, "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()), "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()), "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    for name, lower, upper in BANDS:
        mask = (distance > lower) & (distance <= upper)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error2[mask]))) if mask.any() else np.nan
    return result


for required in (V70_PATH, V74_PATH, V76_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
features = pd.read_parquet(V74_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
v76 = pd.read_parquet(V76_PATH).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
if not features[["fold", "well"]].equals(v76[["fold", "well"]]):
    raise RuntimeError("V74/V76 well alignment mismatch")
if frame.groupby("fold")["well"].nunique().astype(int).to_dict() != {1: 155, 2: 155, 3: 154}:
    raise RuntimeError("Unexpected V70 fold coverage")

scalar_map = dict(zip(v76["well"].astype(str), v76["ensemble_weight"].astype(float)))
base_x = _augment(features)
panel_rows = []
for row_index, row in features.iterrows():
    well = str(row["well"])
    local = frame[frame["well"].astype(str).eq(well)]
    parent = local[PARENT].to_numpy(np.float64)
    move = local[COARSE].to_numpy(np.float64) - parent
    truth = local[TRUTH].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    for band_index, (band, lower, upper) in enumerate(BANDS):
        mask = (distance > lower) & (distance <= upper)
        target = _optimal_weight(parent[mask], move[mask], truth[mask]) if mask.any() else np.nan
        panel_rows.append({
            "feature_row": row_index, "fold": int(row["fold"]), "well": well,
            "band": band, "band_index": band_index, "band_rows": int(mask.sum()),
            "scalar_weight": scalar_map[well], "oracle_band_weight": target,
        })
panel = pd.DataFrame(panel_rows)
panel = panel[np.isfinite(panel["oracle_band_weight"]) & panel["band_rows"].gt(20)].reset_index(drop=True)

x_rows = []
for item in panel.itertuples(index=False):
    values = base_x.iloc[int(item.feature_row)].copy()
    for band in BAND_NAMES:
        values[f"band_{band}"] = float(item.band == band)
    values["band_local_move_rms"] = float(features.iloc[int(item.feature_row)][f"{item.band}_move_rms"])
    values["band_local_move_mean"] = float(features.iloc[int(item.feature_row)][f"{item.band}_move_mean"])
    values["scalar_weight"] = float(item.scalar_weight)
    x_rows.append(values)
x = pd.DataFrame(x_rows).to_numpy(np.float64)
if not np.isfinite(x).all():
    raise RuntimeError("Non-finite distance-dose features")
y = panel["oracle_band_weight"].to_numpy(np.float64) - panel["scalar_weight"].to_numpy(np.float64)
sample_weight = np.sqrt(panel["band_rows"].to_numpy(np.float64))
oof_residual = np.full(len(panel), np.nan, np.float64)

for fold in FOLDS:
    valid = panel["fold"].to_numpy() == fold
    train = ~valid
    hist, ridge = _hist(), _ridge()
    hist.fit(x[train], y[train], sample_weight=sample_weight[train])
    ridge.fit(x[train], y[train])
    oof_residual[valid] = 0.5 * hist.predict(x[valid]) + 0.5 * ridge.predict(x[valid])
    print("V95 FOLD", {"fold": fold, "train": int(train.sum()), "valid": int(valid.sum()), "mean_residual": float(oof_residual[valid].mean())}, flush=True)

if not np.isfinite(oof_residual).all():
    raise RuntimeError("Incomplete V95 OOF residuals")
panel["predicted_band_weight"] = np.clip(panel["scalar_weight"] + oof_residual, MIN_WEIGHT, MAX_WEIGHT)

pred_scalar = np.empty(len(frame), np.float64)
pred_band = np.empty(len(frame), np.float64)
pred_shuffle = np.empty(len(frame), np.float64)
rng = np.random.default_rng(20260730)
for fold in FOLDS:
    fold_wells = sorted(frame.loc[frame["fold"].eq(fold), "well"].astype(str).unique())
    vectors = {}
    for well in fold_wells:
        local_panel = panel[panel["well"].eq(well)].set_index("band")
        scalar = scalar_map[well]
        vectors[well] = np.array([
            float(local_panel.loc[band, "predicted_band_weight"]) if band in local_panel.index else scalar
            for band in BAND_NAMES
        ], np.float64)
    shuffled = list(vectors.values())
    rng.shuffle(shuffled)
    shuffled = dict(zip(fold_wells, shuffled))
    for well in fold_wells:
        indices = frame.index[(frame["fold"].eq(fold)) & (frame["well"].astype(str).eq(well))].to_numpy()
        local = frame.loc[indices]
        parent = local[PARENT].to_numpy(np.float64)
        move = local[COARSE].to_numpy(np.float64) - parent
        scalar = scalar_map[well]
        pred_scalar[indices] = parent + scalar * move
        pred_band[indices] = parent + _row_weights(local["md_since"], vectors[well]) * move
        pred_shuffle[indices] = parent + _row_weights(local["md_since"], shuffled[well]) * move

summary_rows = []
for fold in (0, 1, 2, 3):
    mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].to_numpy() == fold
    local = frame.loc[mask].reset_index(drop=True)
    for name, prediction in (("v76", pred_scalar), ("distance_dose", pred_band), ("shuffled_band_weights", pred_shuffle)):
        summary_rows.append({"fold": fold, **_metric(local, prediction[mask], name)})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
panel.to_parquet(OUT_DIR / "distance_dose_weights_oof.parquet", index=False, compression="zstd")

def _row(fold, endpoint):
    return summary[(summary["fold"].eq(fold)) & (summary["endpoint"].eq(endpoint))].iloc[0]

fold_gains = {str(f): float(_row(f, "v76")["rmse"] - _row(f, "distance_dose")["rmse"]) for f in FOLDS}
overall = _row(0, "v76")
candidate = _row(0, "distance_dose")
shuffled = _row(0, "shuffled_band_weights")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fixed_family_threefold_distance_specific_dose_confirmation",
    "base": "V69 All773 coarse plus V76 scalar adaptive dose",
    "change": "four distance-band residual doses with 128-ft smooth transitions",
    "selection_or_tuning_count": 0,
    "fold_gains": fold_gains,
    "overall_gain": float(overall["rmse"] - candidate["rmse"]),
    "far_gain": float(overall["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
    "near_degradation": float(candidate["near_0_500_rmse"] - overall["near_0_500_rmse"]),
    "shuffle_increment": float(shuffled["rmse"] - candidate["rmse"]),
    "checks": {
        "overall_gain_ge_0p03": bool(overall["rmse"] - candidate["rmse"] >= 0.03),
        "positive_folds_3of3": bool(all(value > 0.0 for value in fold_gains.values())),
        "far_gain_nonnegative": bool(overall["far_gt2000_rmse"] >= candidate["far_gt2000_rmse"]),
        "near_degradation_le_0p03": bool(candidate["near_0_500_rmse"] - overall["near_0_500_rmse"] <= 0.03),
        "aligned_beats_shuffle_ge_0p02": bool(shuffled["rmse"] - candidate["rmse"] >= 0.02),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = "PASS -> fit the same fixed family on all honest wells and export a deployment bundle" if verdict["strict_pass"] else "FAIL -> retain scalar V76 and close distance-specific dosing"
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV95 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV95 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
