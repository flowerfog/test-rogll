"""Matched model-family ablation for the proven no-GR CP8 features.

The input table, folds, target, CP interpolation, shrink, and cap are fixed.
Only the estimator (or a pre-registered fixed estimator blend) changes.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CP_PATH = OOF_ROOT / "aux_aware_cp8_ablation_v0" / "control_point_table.pkl"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "cp8_no_gr_model_family_v0"

BASE_COL = "cp12"
N_CP = 8
N_FOLDS = 5
SEED = 20260713
CP_ALPHA = 0.50
CP_CAP = 6.0
EXPECTED_REFERENCE_GAIN = 0.10307155203761553
REFERENCE_TOLERANCE = 0.005

OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (CP_PATH, POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


cp = pd.read_pickle(CP_PATH).copy().reset_index(drop=True)
cp["well"] = cp["well"].astype(str)
metadata = {"well", "fold", "center_local", "target_residual"}
base_feature_columns = [
    column
    for column in cp.columns
    if column not in metadata
    and not column.startswith("AUX_")
    and pd.api.types.is_numeric_dtype(cp[column])
]
feature_columns = [
    column for column in base_feature_columns if "gr" not in column.lower()
]
if len(feature_columns) != 100:
    raise RuntimeError(f"expected 100 no-GR features, got {len(feature_columns)}")

matrix = (
    cp[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)
target = cp["target_residual"].to_numpy(np.float32)
folds = cp["fold"].to_numpy(int)
fold_map = cp.groupby("well")["fold"].first().astype(int).to_dict()

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [
    column
    for column in ["id", "well", "row_idx", "tvt_true", BASE_COL]
    if column in header
]
df = pd.read_csv(POOL_PATH, usecols=usecols)
df["id"] = df["id"].astype(str)
if "well" not in df.columns:
    df["well"] = df["id"].str.rsplit("_", n=1).str[0]
df["well"] = df["well"].astype(str)
if "row_idx" not in df.columns:
    df["row_idx"] = df["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
df = df.sort_values(["well", "row_idx"]).reset_index(drop=True)
if set(df["well"].unique()) != set(fold_map):
    raise RuntimeError("pool and CP well sets differ")

truth = df["tvt_true"].to_numpy(np.float32)
base = df[BASE_COL].to_numpy(np.float32)
groups = df.groupby("well", sort=False).indices
print(
    {
        "experiment": "cp8_no_gr_model_family_v0",
        "cp_rows": len(cp),
        "features": len(feature_columns),
        "base_rmse": _rmse(truth, base),
        "alpha": CP_ALPHA,
        "cap": CP_CAP,
    }
)


def _new_lgb_reference(fold):
    return LGBMRegressor(
        objective="huber",
        alpha=0.90,
        n_estimators=2000,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.20,
        reg_lambda=4.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1,
    )


def _new_lgb_conservative(fold):
    return LGBMRegressor(
        objective="huber",
        alpha=0.90,
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=SEED + 100 + fold,
        n_jobs=-1,
        verbosity=-1,
    )


def _new_extra_trees(fold):
    return ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.75,
        min_samples_leaf=8,
        max_depth=None,
        bootstrap=False,
        random_state=SEED + 200 + fold,
        n_jobs=-1,
    )


def _new_hist(fold):
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=4.0,
        early_stopping=False,
        random_state=SEED + 300 + fold,
    )


builders = {
    "lgb_reference": _new_lgb_reference,
    "lgb_conservative": _new_lgb_conservative,
    "extra_trees": _new_extra_trees,
    "hist_absolute": _new_hist,
}

control_predictions = {
    name: np.full(len(cp), np.nan, dtype=np.float32) for name in builders
}
model_rows = []
started = time.time()
for fold in range(1, N_FOLDS + 1):
    train_indices = np.flatnonzero(folds != fold)
    valid_indices = np.flatnonzero(folds == fold)
    for name, builder in builders.items():
        model = builder(fold)
        if name.startswith("lgb_"):
            model.fit(
                matrix.iloc[train_indices],
                target[train_indices],
                eval_set=[(matrix.iloc[valid_indices], target[valid_indices])],
                eval_metric="rmse",
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
        else:
            model.fit(matrix.iloc[train_indices], target[train_indices])
            best_iteration = int(getattr(model, "n_iter_", 0) or 0)

        prediction = model.predict(matrix.iloc[valid_indices]).astype(np.float32)
        control_predictions[name][valid_indices] = prediction
        model_rows.append(
            {
                "model": name,
                "fold": fold,
                "train_rows": len(train_indices),
                "valid_rows": len(valid_indices),
                "best_iteration": best_iteration,
                "target_rmse": _rmse(target[valid_indices], prediction),
            }
        )
    print(f"fold {fold}/{N_FOLDS} elapsed={time.time()-started:.1f}s")

for name, values in control_predictions.items():
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite control predictions: {name}")


# Fixed blends are specified before inspecting their row-level scores.
control_variants = {
    **control_predictions,
    "blend_ref_cons_50": (
        0.50 * control_predictions["lgb_reference"]
        + 0.50 * control_predictions["lgb_conservative"]
    ),
    "blend_ref_extra_75_25": (
        0.75 * control_predictions["lgb_reference"]
        + 0.25 * control_predictions["extra_trees"]
    ),
    "blend_ref_hist_75_25": (
        0.75 * control_predictions["lgb_reference"]
        + 0.25 * control_predictions["hist_absolute"]
    ),
    "blend_ref_cons_extra_50_25_25": (
        0.50 * control_predictions["lgb_reference"]
        + 0.25 * control_predictions["lgb_conservative"]
        + 0.25 * control_predictions["extra_trees"]
    ),
}


def _expand(control_prediction):
    move = np.zeros(len(df), dtype=np.float32)
    for wid, indices in groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        controls = cp[cp["well"].eq(wid)].sort_values("center_local")
        positions = controls["center_local"].to_numpy(float)
        values = control_prediction[controls.index.to_numpy(int)].astype(float)
        keep = np.r_[True, np.diff(positions) > 0]
        positions = positions[keep]
        values = values[keep]
        if len(positions) == 1:
            well_move = np.full(len(indices), values[0])
        else:
            well_move = np.interp(
                np.arange(len(indices), dtype=float), positions, values
            )
        sigma = max(4.0, len(indices) / N_CP / 8.0)
        well_move = gaussian_filter1d(well_move, sigma=sigma, mode="nearest")
        move[indices] = well_move.astype(np.float32)
    return np.clip(CP_ALPHA * move, -CP_CAP, CP_CAP)


moves = {}
predictions = {}
for name, control in tqdm(control_variants.items(), desc="expand model variants"):
    moves[name] = _expand(control)
    predictions[name] = base + moves[name]

base_sse_bywell = []
for wid, indices in groups.items():
    indices = np.asarray(indices, dtype=np.int64)
    error = base[indices] - truth[indices]
    base_sse_bywell.append({"well": wid, "sse": float(np.sum(error * error))})
top40 = set(pd.DataFrame(base_sse_bywell).nlargest(40, "sse")["well"].astype(str))
top40_mask = df["well"].isin(top40).to_numpy()
base_rmse = _rmse(truth, base)
top40_base_rmse = _rmse(truth[top40_mask], base[top40_mask])

score_rows = []
fold_rows = []
for name, prediction in predictions.items():
    fold_gains = []
    for fold in range(1, N_FOLDS + 1):
        valid_wells = {wid for wid, value in fold_map.items() if value == fold}
        mask = df["well"].isin(valid_wells).to_numpy()
        fold_base = _rmse(truth[mask], base[mask])
        fold_score = _rmse(truth[mask], prediction[mask])
        gain = fold_base - fold_score
        fold_gains.append(gain)
        fold_rows.append(
            {
                "variant": name,
                "fold": fold,
                "rows": int(mask.sum()),
                "base_rmse": fold_base,
                "rmse": fold_score,
                "gain": gain,
            }
        )

    score = _rmse(truth, prediction)
    score_rows.append(
        {
            "variant": name,
            "base_rmse": base_rmse,
            "rmse": score,
            "gain": base_rmse - score,
            "mean_fold_gain": float(np.mean(fold_gains)),
            "min_fold_gain": float(np.min(fold_gains)),
            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
            "top40_gain": (
                top40_base_rmse
                - _rmse(truth[top40_mask], prediction[top40_mask])
            ),
            "mean_abs_move": float(np.mean(np.abs(moves[name]))),
            "p95_abs_move": float(np.quantile(np.abs(moves[name]), 0.95)),
        }
    )

score_report = pd.DataFrame(score_rows).sort_values("gain", ascending=False)
fold_report = pd.DataFrame(fold_rows)
reference = score_report[score_report["variant"].eq("lgb_reference")].iloc[0]
reference_reproduced = (
    abs(float(reference["gain"]) - EXPECTED_REFERENCE_GAIN)
    <= REFERENCE_TOLERANCE
)
best = score_report.iloc[0]
best_folds = fold_report[fold_report["variant"].eq(best["variant"])]
strict_pass = bool(
    reference_reproduced
    and best["variant"] != "lgb_reference"
    and float(best["gain"]) >= 0.12
    and float(best["gain"]) - float(reference["gain"]) >= 0.015
    and (best_folds["gain"] > 0).all()
)

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nSCORES")
display(score_report)
print("\nFOLDS")
display(fold_report)
verdict = {
    "reference_gain": float(reference["gain"]),
    "reference_reproduced": reference_reproduced,
    "best_variant": str(best["variant"]),
    "best_gain": float(best["gain"]),
    "incremental_gain": float(best["gain"]) - float(reference["gain"]),
    "strict_pass": strict_pass,
    "next": (
        "deployability and full-chain audit"
        if strict_pass
        else "keep lgb_reference; stop CP8 model-family tuning"
    ),
}
print("\nVERDICT", verdict)

pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_folds.csv", index=False)
score_report.to_csv(OUT_DIR / "score_report.csv", index=False)
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
if strict_pass:
    output = df[["id", "well", "row_idx", "tvt_true"]].copy()
    output[BASE_COL] = base
    for name, prediction in predictions.items():
        output[name] = prediction
    output.to_pickle(OUT_DIR / "model_family_oof.pkl")
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
