"""Nested label-free per-well Neighbor dose after frozen CP8.

Folds 4/5 train small dose regressors, Fold 3 selects the model and shrinkage,
and Folds 2/1 confirm once.  Targets are complete-well optimal doses, while
features are prediction/geometry diagnostics available at inference time.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SEQUENCE_DIR = OOF_ROOT / "frontier_6p39_cp_neighbor_sequence_v3"
SEQUENCE_PATH = SEQUENCE_DIR / "cp_neighbor_oof.parquet"
SEQUENCE_VERDICT = SEQUENCE_DIR / "verdict.json"
NEIGHBOR_PATH = (
    OOF_ROOT / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet"
)
NEIGHBOR_META = OOF_ROOT / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"
OUT_DIR = OOF_ROOT / "frontier_6p39_neighbor_dose_regressor_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLDS = (2, 1)
MAX_DOSE = 1.25
SHRINKAGES = (0.25, 0.50, 0.75, 1.00)
SEED = 20260724

for required in (SEQUENCE_PATH, SEQUENCE_VERDICT, NEIGHBOR_PATH, NEIGHBOR_META):
    if not required.is_file():
        raise FileNotFoundError(required)

sequence_verdict = json.loads(SEQUENCE_VERDICT.read_text(encoding="utf-8"))
source_dose = float(sequence_verdict["selected_recipe"]["neighbor_dose"])
if source_dose <= 0.0:
    raise RuntimeError(f"Cannot recover Neighbor direction from dose={source_dose}")

frame = pd.read_parquet(SEQUENCE_PATH)
required = {"id", "well", "row_idx", "fold", "tvt_true", "base", "cp_only", "cp_neighbor"}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Sequence OOF missing {missing}")
frame = frame.sort_values(["well", "row_idx"]).reset_index(drop=True)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)

neighbor_rows = pd.read_parquet(NEIGHBOR_PATH, columns=["id", "md_since"])
neighbor_rows["id"] = neighbor_rows["id"].astype(str)
frame = frame.merge(neighbor_rows, on="id", how="left", validate="one_to_one")
if frame["md_since"].isna().any():
    raise RuntimeError("Neighbor OOF does not cover sequence rows")

meta = pd.read_csv(NEIGHBOR_META)
meta["well"] = meta["well"].astype(str)
meta_columns = [
    column for column in (
        "well", "min_distance", "median_distance", "p90_distance",
        "anchor_distance_full", "abs_direction_dot",
    ) if column in meta.columns
]
meta = meta[meta_columns].drop_duplicates("well")

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["base"].to_numpy(np.float64)
cp = frame["cp_only"].to_numpy(np.float64)
direction = (frame["cp_neighbor"].to_numpy(np.float64) - cp) / source_dose
distance = frame["md_since"].to_numpy(np.float64)
fold_rows = frame["fold"].to_numpy(np.int8)


def metrics(mask: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction[mask] - truth[mask]
    wells = frame.loc[mask, "well"].to_numpy()
    bywell = (
        pd.DataFrame({"well": wells, "error2": error * error})
        .groupby("well")["error2"].mean().pow(0.5)
    )
    result = {
        "rows": int(mask.sum()),
        "wells": int(pd.Series(wells).nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    local_distance = distance[mask]
    for name, local_mask in {
        "near_0_500_rmse": local_distance <= 500.0,
        "mid_500_1000_rmse": (local_distance > 500.0) & (local_distance <= 1000.0),
        "mid_1000_2000_rmse": (local_distance > 1000.0) & (local_distance <= 2000.0),
        "far_gt2000_rmse": local_distance > 2000.0,
    }.items():
        result[name] = float(np.sqrt(np.mean(error[local_mask] ** 2))) if local_mask.any() else None
    return result


feature_rows = []
for well, rows in frame.groupby("well", sort=False).indices.items():
    rows = np.asarray(rows, dtype=np.int64)
    d = direction[rows]
    cp_move = cp[rows] - base[rows]
    cp_error = cp[rows] - truth[rows]
    far = distance[rows] > 2000.0
    denominator = float(np.dot(d, d))
    optimal_dose = (
        float(np.clip(-np.dot(cp_error, d) / denominator, 0.0, MAX_DOSE))
        if denominator > 1e-10 else 0.0
    )
    corr = float(np.corrcoef(cp_move, d)[0, 1]) if np.std(cp_move) > 1e-8 and np.std(d) > 1e-8 else 0.0
    feature_rows.append({
        "well": str(well),
        "fold": int(frame.iloc[rows[0]]["fold"]),
        "rows": int(len(rows)),
        "optimal_dose": optimal_dose,
        "cp_move_rmse": float(np.sqrt(np.mean(cp_move ** 2))),
        "cp_move_abs_mean": float(np.mean(np.abs(cp_move))),
        "cp_move_signed_mean": float(np.mean(cp_move)),
        "neighbor_direction_rmse": float(np.sqrt(np.mean(d ** 2))),
        "neighbor_direction_abs_mean": float(np.mean(np.abs(d))),
        "neighbor_direction_abs_p95": float(np.quantile(np.abs(d), 0.95)),
        "neighbor_direction_signed_mean": float(np.mean(d)),
        "neighbor_direction_far_rmse": float(np.sqrt(np.mean(d[far] ** 2))) if far.any() else 0.0,
        "neighbor_active_fraction": float(np.mean(np.abs(d) > 1e-10)),
        "cp_neighbor_move_correlation": corr,
        "cp_neighbor_endpoint_gap": float(abs(cp_move[-1] - d[-1])),
    })

wells = pd.DataFrame(feature_rows).merge(meta, on="well", how="left", validate="one_to_one")
feature_names = [
    column for column in wells.columns
    if column not in {"well", "fold", "rows", "optimal_dose"}
]
X = wells[feature_names].replace([np.inf, -np.inf], np.nan)
y = wells["optimal_dose"].to_numpy(np.float64)
fold_wells = wells["fold"].to_numpy(np.int8)
train_mask = np.isin(fold_wells, TRAIN_FOLDS)
selection_well_mask = fold_wells == SELECTION_FOLD

models = {
    "ridge_1": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
    "ridge_10": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
    "ridge_100": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=100.0)),
    "extra_leaf10": make_pipeline(SimpleImputer(), ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=10, max_features=0.8,
        random_state=SEED, n_jobs=-1,
    )),
    "extra_leaf20": make_pipeline(SimpleImputer(), ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=20, max_features=0.8,
        random_state=SEED + 1, n_jobs=-1,
    )),
    "hist_leaf15": make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.04, max_leaf_nodes=7,
        min_samples_leaf=15, l2_regularization=5.0, random_state=SEED,
    )),
}

well_position = {well: position for position, well in enumerate(wells["well"])}
row_well_position = frame["well"].map(well_position).to_numpy(np.int64)


def prediction_from_dose(well_dose: np.ndarray) -> np.ndarray:
    return cp + well_dose[row_well_position] * direction


selection_row_mask = fold_rows == SELECTION_FOLD
cp_selection = metrics(selection_row_mask, cp)
scan_rows = []
fitted_predictions = {}
for name, estimator in models.items():
    fitted = clone(estimator).fit(X.loc[train_mask], y[train_mask])
    raw = np.clip(fitted.predict(X), 0.0, MAX_DOSE)
    fitted_predictions[name] = (fitted, raw)
    for shrinkage in SHRINKAGES:
        dose = np.clip(float(shrinkage) * raw, 0.0, MAX_DOSE)
        result = metrics(selection_row_mask, prediction_from_dose(dose))
        scan_rows.append({
            "model": name,
            "shrinkage": float(shrinkage),
            "mean_dose": float(np.mean(dose[selection_well_mask])),
            **result,
            "gain_vs_cp": cp_selection["rmse"] - result["rmse"],
        })

scan = pd.DataFrame(scan_rows).sort_values(
    ["rmse", "p95_well_rmse", "mean_dose", "model"],
    ascending=[True, True, True, True],
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_model_scan.csv", index=False)
selected = scan.iloc[0]
selected_model = str(selected["model"])
selected_shrinkage = float(selected["shrinkage"])
fitted, raw_dose = fitted_predictions[selected_model]
selected_dose = np.clip(selected_shrinkage * raw_dose, 0.0, MAX_DOSE)
candidate = prediction_from_dose(selected_dose)

# Matched target-shuffle control uses the selected architecture and shrinkage.
rng = np.random.default_rng(SEED)
shuffled_y = y[train_mask].copy()
rng.shuffle(shuffled_y)
shuffled_model = clone(models[selected_model]).fit(X.loc[train_mask], shuffled_y)
shuffled_dose = np.clip(selected_shrinkage * shuffled_model.predict(X), 0.0, MAX_DOSE)
shuffled_candidate = prediction_from_dose(shuffled_dose)

rows = []
details = {}
gains = {}
for fold_value, label in ((3, "selection_fold3"), (2, "confirmation_fold2"), (1, "final_fold1")):
    mask = fold_rows == fold_value
    local = {
        "cp_only": metrics(mask, cp),
        "dose_model": metrics(mask, candidate),
        "shuffled_target": metrics(mask, shuffled_candidate),
    }
    details[label] = local
    gains[label] = local["cp_only"]["rmse"] - local["dose_model"]["rmse"]
    for endpoint, result in local.items():
        rows.append({"subset": label, "endpoint": endpoint, **result})

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
wells["predicted_dose"] = selected_dose
wells["shuffled_predicted_dose"] = shuffled_dose
wells.to_csv(OUT_DIR / "well_doses.csv", index=False)

strict_pass = bool(
    gains["confirmation_fold2"] >= 0.05
    and gains["final_fold1"] >= 0.05
    and details["confirmation_fold2"]["dose_model"]["rmse"]
        < details["confirmation_fold2"]["shuffled_target"]["rmse"]
    and details["final_fold1"]["dose_model"]["rmse"]
        < details["final_fold1"]["shuffled_target"]["rmse"]
)
verdict = {
    "experiment": "frontier_6p39_neighbor_dose_regressor_v6",
    "stage": "Fold45_train_Fold3_select_Fold2_and_Fold1_confirm",
    "source_neighbor_dose": source_dose,
    "selected_model": selected_model,
    "selected_shrinkage": selected_shrinkage,
    "features": feature_names,
    "metrics": details,
    "gain_vs_cp": gains,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024 ** 3),
    "next": (
        "PASS -> freeze dose model and build deployment CP->Neighbor chain"
        if strict_pass
        else "FAIL -> confidence is not transferable; deploy frozen CP without Neighbor"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("FOLD3 DOSE MODEL SCAN")
print(scan.to_string(index=False), flush=True)
print("\nDOSE MODEL CONFIRMATION")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
