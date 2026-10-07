"""Nested whole-well gate audit for Neighbor after the confirmed CP8 endpoint.

Run after ``frontier_6p39_cp_neighbor_sequence_v3_cell.py``.  Fold 3 selects
one label-free threshold rule.  Fold 2 and Fold 1 are confirmation sets and
are never searched.  This cell does not train a model or regenerate paths.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SEQUENCE_PATH = (
    OOF_ROOT / "frontier_6p39_cp_neighbor_sequence_v3" / "cp_neighbor_oof.parquet"
)
NEIGHBOR_PATH = (
    OOF_ROOT / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet"
)
NEIGHBOR_META = OOF_ROOT / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"
OUT_DIR = OOF_ROOT / "frontier_6p39_neighbor_after_cp_gate_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLDS = (2, 1)
MIN_SELECTED_WELLS = 12
QUANTILES = np.linspace(0.10, 0.90, 17)

for required in (SEQUENCE_PATH, NEIGHBOR_PATH, NEIGHBOR_META):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(SEQUENCE_PATH)
required_columns = {
    "id", "well", "row_idx", "fold", "tvt_true", "base", "cp_only", "cp_neighbor"
}
missing = sorted(required_columns.difference(frame.columns))
if missing:
    raise RuntimeError(f"Sequence OOF is missing {missing}")
frame = frame.sort_values(["well", "row_idx"]).reset_index(drop=True)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)

try:
    neighbor_rows = pd.read_parquet(NEIGHBOR_PATH, columns=["id", "md_since"])
except Exception as exc:
    raise RuntimeError(f"Neighbor OOF has no readable id/md_since columns: {exc}") from exc
neighbor_rows["id"] = neighbor_rows["id"].astype(str)
frame = frame.merge(neighbor_rows, on="id", how="left", validate="one_to_one")
if frame["md_since"].isna().any():
    raise RuntimeError(f"Missing md_since for {int(frame['md_since'].isna().sum())} rows")

meta = pd.read_csv(NEIGHBOR_META)
meta["well"] = meta["well"].astype(str)
meta_columns = [
    column
    for column in (
        "well", "min_distance", "median_distance", "p90_distance",
        "anchor_distance_full", "abs_direction_dot",
    )
    if column in meta.columns
]
meta = meta[meta_columns].drop_duplicates("well")

truth = frame["tvt_true"].to_numpy(np.float64)
cp = frame["cp_only"].to_numpy(np.float64)
combo = frame["cp_neighbor"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
increment = combo - cp


def metrics(mask: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction[mask] - truth[mask]
    local_wells = frame.loc[mask, "well"].to_numpy()
    bywell = (
        pd.DataFrame({"well": local_wells, "error2": error * error})
        .groupby("well")["error2"]
        .mean()
        .pow(0.5)
    )
    result = {
        "rows": int(mask.sum()),
        "wells": int(pd.Series(local_wells).nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    local_distance = distance[mask]
    for label, local_mask in {
        "near_0_500_rmse": local_distance <= 500.0,
        "mid_500_1000_rmse": (local_distance > 500.0) & (local_distance <= 1000.0),
        "mid_1000_2000_rmse": (local_distance > 1000.0) & (local_distance <= 2000.0),
        "far_gt2000_rmse": local_distance > 2000.0,
    }.items():
        result[label] = (
            float(np.sqrt(np.mean(error[local_mask] ** 2))) if local_mask.any() else None
        )
    return result


# All gate features are available at inference time.  None uses truth.
feature_rows = []
for well, rows in frame.groupby("well", sort=False).indices.items():
    rows = np.asarray(rows, dtype=np.int64)
    local_increment = increment[rows]
    local_distance = distance[rows]
    far = local_distance > 2000.0
    feature_rows.append({
        "well": str(well),
        "fold": int(frame.iloc[rows[0]]["fold"]),
        "rows": int(len(rows)),
        "cp_move_rmse": float(np.sqrt(np.mean((cp[rows] - frame.iloc[rows]["base"].to_numpy(float)) ** 2))),
        "neighbor_increment_rmse": float(np.sqrt(np.mean(local_increment ** 2))),
        "neighbor_increment_abs_mean": float(np.mean(np.abs(local_increment))),
        "neighbor_increment_abs_p95": float(np.quantile(np.abs(local_increment), 0.95)),
        "neighbor_increment_signed_mean": float(np.mean(local_increment)),
        "neighbor_increment_far_rmse": (
            float(np.sqrt(np.mean(local_increment[far] ** 2))) if far.any() else 0.0
        ),
        "neighbor_active_fraction": float(np.mean(np.abs(local_increment) > 1e-10)),
    })

features = pd.DataFrame(feature_rows).merge(meta, on="well", how="left", validate="one_to_one")
feature_names = [
    column for column in features.columns
    if column not in {"well", "fold", "rows"} and np.isfinite(features[column]).sum() >= 100
]
well_to_position = {well: position for position, well in enumerate(features["well"])}
row_well_position = frame["well"].map(well_to_position).to_numpy(np.int64)


def prediction_from_well_choice(choose_combo: np.ndarray) -> np.ndarray:
    return np.where(choose_combo[row_well_position], combo, cp)


def perwell_oracle_prediction(fold_value: int) -> tuple[np.ndarray, np.ndarray]:
    choose = np.zeros(len(features), dtype=bool)
    fold_wells = features.loc[features["fold"].eq(fold_value), "well"]
    for well in fold_wells:
        rows = np.flatnonzero(frame["well"].to_numpy() == well)
        cp_sse = float(np.sum((cp[rows] - truth[rows]) ** 2))
        combo_sse = float(np.sum((combo[rows] - truth[rows]) ** 2))
        choose[well_to_position[well]] = combo_sse < cp_sse
    return prediction_from_well_choice(choose), choose


selection_row_mask = frame["fold"].to_numpy() == SELECTION_FOLD
selection_well_mask = features["fold"].to_numpy() == SELECTION_FOLD
cp_selection = metrics(selection_row_mask, cp)

scan_rows = []
for feature in feature_names:
    values = features[feature].to_numpy(np.float64)
    fit_values = values[selection_well_mask & np.isfinite(values)]
    if len(np.unique(fit_values)) < 5:
        continue
    for threshold in np.unique(np.quantile(fit_values, QUANTILES)):
        for direction in ("le", "ge"):
            finite = np.isfinite(values)
            choose = finite & ((values <= threshold) if direction == "le" else (values >= threshold))
            selected_count = int(np.sum(choose & selection_well_mask))
            if selected_count < MIN_SELECTED_WELLS:
                continue
            prediction = prediction_from_well_choice(choose)
            result = metrics(selection_row_mask, prediction)
            scan_rows.append({
                "feature": feature,
                "direction": direction,
                "threshold": float(threshold),
                "selected_wells": selected_count,
                **result,
                "gain_vs_cp": cp_selection["rmse"] - result["rmse"],
            })

scan = pd.DataFrame(scan_rows).sort_values(
    ["rmse", "p95_well_rmse", "selected_wells", "feature"],
    ascending=[True, True, True, True],
).reset_index(drop=True)
if scan.empty:
    raise RuntimeError("No valid threshold gates")
scan.to_csv(OUT_DIR / "fold3_gate_scan.csv", index=False)
selected = scan.iloc[0]
recipe = {
    "feature": str(selected["feature"]),
    "direction": str(selected["direction"]),
    "threshold": float(selected["threshold"]),
}

values = features[recipe["feature"]].to_numpy(np.float64)
finite = np.isfinite(values)
selected_choice = finite & (
    (values <= recipe["threshold"])
    if recipe["direction"] == "le"
    else (values >= recipe["threshold"])
)
gated = prediction_from_well_choice(selected_choice)

summary_rows = []
details = {}
oracle_gains = {}
gate_gains = {}
for fold_value, label in ((3, "selection_fold3"), (2, "confirmation_fold2"), (1, "final_fold1")):
    row_mask = frame["fold"].to_numpy() == fold_value
    oracle, oracle_choice = perwell_oracle_prediction(fold_value)
    local = {
        "cp_only": metrics(row_mask, cp),
        "fixed_combo": metrics(row_mask, combo),
        "threshold_gate": metrics(row_mask, gated),
        "perwell_oracle": metrics(row_mask, oracle),
    }
    local["selected_gate_wells"] = int(np.sum(selected_choice & (features["fold"].to_numpy() == fold_value)))
    local["oracle_combo_wells"] = int(np.sum(oracle_choice & (features["fold"].to_numpy() == fold_value)))
    details[label] = local
    oracle_gains[label] = local["cp_only"]["rmse"] - local["perwell_oracle"]["rmse"]
    gate_gains[label] = local["cp_only"]["rmse"] - local["threshold_gate"]["rmse"]
    for endpoint in ("cp_only", "fixed_combo", "threshold_gate", "perwell_oracle"):
        summary_rows.append({"subset": label, "endpoint": endpoint, **local[endpoint]})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "gate_summary.csv", index=False)
features.to_csv(OUT_DIR / "well_gate_features.csv", index=False)

oracle_viable = bool(
    oracle_gains["confirmation_fold2"] >= 0.15
    and oracle_gains["final_fold1"] >= 0.15
)
gate_pass = bool(
    gate_gains["confirmation_fold2"] >= 0.05
    and gate_gains["final_fold1"] >= 0.05
)
verdict = {
    "experiment": "frontier_6p39_neighbor_after_cp_gate_v4",
    "stage": "Fold3_label_free_threshold_selection_to_Fold2_and_Fold1_confirmation",
    "selected_recipe": recipe,
    "feature_count": len(feature_names),
    "metrics": details,
    "oracle_gain_vs_cp": oracle_gains,
    "gate_gain_vs_cp": gate_gains,
    "gates": {
        "each_confirmation_oracle_gain_ge_0p15": oracle_viable,
        "each_confirmation_gate_gain_ge_0p05": gate_pass,
    },
    "strict_pass": bool(oracle_viable and gate_pass),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024 ** 3),
    "next": (
        "PASS -> freeze this whole-well gate and reproduce CP->Neighbor inference"
        if oracle_viable and gate_pass
        else (
            "ORACLE ONLY -> train no gate yet; improve label-free Neighbor confidence"
            if oracle_viable
            else "CLOSE CONDITIONAL NEIGHBOR -> deploy CP only"
        )
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("FOLD3 GATE TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nGATE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
