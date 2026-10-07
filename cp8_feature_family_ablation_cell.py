"""Feature-family ablation for the stable no-aux CP8 residual model.

Uses the already-built control-point table from aux_aware_cp8_ablation_v0.
Every variant keeps the same folds, LightGBM settings, CP interpolation,
0.5 shrink, and 6 ft cap. The only changed variable is the input family.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from tqdm.auto import tqdm


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "aux_aware_cp8_ablation_v0"
CP_PATH = SOURCE_DIR / "control_point_table.pkl"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "cp8_feature_family_ablation_v0"

BASE_COL = "cp12"
N_CP = 8
N_FOLDS = 5
SEED = 20260713
CP_ALPHA = 0.50
CP_CAP = 6.0

OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (CP_PATH, POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


cp = pd.read_pickle(CP_PATH).copy()
cp["well"] = cp["well"].astype(str)
cp = cp.reset_index(drop=True)

metadata = {"well", "fold", "center_local", "target_residual"}
base_feature_columns = [
    column
    for column in cp.columns
    if column not in metadata
    and not column.startswith("AUX_")
    and pd.api.types.is_numeric_dtype(cp[column])
]


def _is_candidate(column):
    roots = ("ridge", "selector", "fleongg", "sp45", "strong_bias", "strong")
    return column.startswith("candidate_") or column.startswith(roots)


def _is_gr(column):
    lower = column.lower()
    return "gr" in lower


def _is_geometry(column):
    roots = (
        "MD_",
        "X_",
        "Y_",
        "Z_",
        "build_",
        "dogleg_",
        "fraction_",
    )
    return column.startswith(roots)


def _is_base_shape(column):
    return column.startswith("base_")


families = {
    "candidate": [column for column in base_feature_columns if _is_candidate(column)],
    "gr": [column for column in base_feature_columns if _is_gr(column)],
    "geometry": [column for column in base_feature_columns if _is_geometry(column)],
    "base_shape": [column for column in base_feature_columns if _is_base_shape(column)],
}
assigned = set().union(*[set(columns) for columns in families.values()])
families["context"] = [column for column in base_feature_columns if column not in assigned]

variants = {
    "all": base_feature_columns,
    "no_candidate": [column for column in base_feature_columns if column not in families["candidate"]],
    "no_gr": [column for column in base_feature_columns if column not in families["gr"]],
    "no_geometry": [column for column in base_feature_columns if column not in families["geometry"]],
    "no_base_shape": [column for column in base_feature_columns if column not in families["base_shape"]],
    "candidate_base_context": sorted(
        set(families["candidate"] + families["base_shape"] + families["context"])
    ),
    "raw_base_context": sorted(
        set(families["gr"] + families["geometry"] + families["base_shape"] + families["context"])
    ),
}

print("family sizes:", {name: len(columns) for name, columns in families.items()})
print("variant sizes:", {name: len(columns) for name, columns in variants.items()})


header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [column for column in ["id", "well", "row_idx", "tvt_true", BASE_COL] if column in header]
df = pd.read_csv(POOL_PATH, usecols=usecols)
df["id"] = df["id"].astype(str)
if "well" not in df.columns:
    df["well"] = df["id"].str.rsplit("_", n=1).str[0]
df["well"] = df["well"].astype(str)
if "row_idx" not in df.columns:
    df["row_idx"] = df["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
df = df.sort_values(["well", "row_idx"]).reset_index(drop=True)

truth = df["tvt_true"].to_numpy(np.float32)
base = df[BASE_COL].to_numpy(np.float32)
fold_map = cp.groupby("well")["fold"].first().astype(int).to_dict()


def _train_control_oof(name, feature_columns):
    matrix = (
        cp[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    target = cp["target_residual"].to_numpy(np.float32)
    output = np.full(len(cp), np.nan, dtype=np.float32)
    model_rows = []
    for fold in range(1, N_FOLDS + 1):
        train_indices = np.flatnonzero(cp["fold"].to_numpy(int) != fold)
        valid_indices = np.flatnonzero(cp["fold"].to_numpy(int) == fold)
        model = LGBMRegressor(
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
        model.fit(
            matrix.iloc[train_indices],
            target[train_indices],
            eval_set=[(matrix.iloc[valid_indices], target[valid_indices])],
            eval_metric="rmse",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        prediction = model.predict(matrix.iloc[valid_indices]).astype(np.float32)
        output[valid_indices] = prediction
        model_rows.append(
            {
                "variant": name,
                "fold": fold,
                "features": len(feature_columns),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "target_rmse": _rmse(target[valid_indices], prediction),
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError(f"non-finite control OOF: {name}")
    return output, model_rows


def _expand(control_prediction):
    move = np.zeros(len(df), dtype=np.float32)
    for wid, indices in df.groupby("well", sort=False).indices.items():
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
            well_move = np.interp(np.arange(len(indices), dtype=float), positions, values)
        sigma = max(4.0, len(indices) / N_CP / 8.0)
        well_move = gaussian_filter1d(well_move, sigma=sigma, mode="nearest")
        move[indices] = well_move.astype(np.float32)
    return np.clip(CP_ALPHA * move, -CP_CAP, CP_CAP)


predictions = {}
moves = {}
model_rows = []
for name, columns in tqdm(variants.items(), desc="feature-family variants"):
    control, report = _train_control_oof(name, columns)
    moves[name] = _expand(control)
    predictions[name] = base + moves[name]
    model_rows.extend(report)


top40_wells = set()
base_sse_bywell = []
for wid, indices in df.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, dtype=np.int64)
    error = base[indices] - truth[indices]
    base_sse_bywell.append({"well": wid, "sse": float(np.sum(error * error))})
top40_wells = set(pd.DataFrame(base_sse_bywell).nlargest(40, "sse")["well"].astype(str))

score_rows = []
fold_rows = []
base_rmse = _rmse(truth, base)
top40_mask = df["well"].isin(top40_wells).to_numpy()
top40_base = _rmse(truth[top40_mask], base[top40_mask])

for name in variants:
    prediction = predictions[name]
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
            "features": len(variants[name]),
            "base_rmse": base_rmse,
            "rmse": score,
            "gain": base_rmse - score,
            "mean_fold_gain": float(np.mean(fold_gains)),
            "min_fold_gain": float(np.min(fold_gains)),
            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
            "top40_gain": top40_base - _rmse(truth[top40_mask], prediction[top40_mask]),
            "mean_abs_move": float(np.mean(np.abs(moves[name]))),
        }
    )

score_report = pd.DataFrame(score_rows).sort_values("gain", ascending=False)
fold_report = pd.DataFrame(fold_rows)
all_gain = float(score_report.loc[score_report["variant"].eq("all"), "gain"].iloc[0])
drop_rows = []
for family in ("candidate", "gr", "geometry", "base_shape"):
    variant = f"no_{family}"
    variant_gain = float(
        score_report.loc[score_report["variant"].eq(variant), "gain"].iloc[0]
    )
    drop_rows.append(
        {
            "family": family,
            "features_removed": len(families[family]),
            "all_gain": all_gain,
            "gain_without_family": variant_gain,
            "family_value": all_gain - variant_gain,
        }
    )
family_report = pd.DataFrame(drop_rows).sort_values("family_value", ascending=False)

print("\nSCORES")
display(score_report)
print("\nFOLDS")
display(fold_report)
print("\nFAMILY VALUE")
display(family_report)
print(
    "\nCONCLUSION",
    {
        "best_variant": score_report.iloc[0]["variant"],
        "best_gain": float(score_report.iloc[0]["gain"]),
        "highest_value_family": family_report.iloc[0]["family"],
        "highest_family_value": float(family_report.iloc[0]["family_value"]),
    },
)

pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_folds.csv", index=False)
score_report.to_csv(OUT_DIR / "score_report.csv", index=False)
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
family_report.to_csv(OUT_DIR / "family_value_report.csv", index=False)
(OUT_DIR / "conclusion.json").write_text(
    json.dumps(
        {
            "best_variant": str(score_report.iloc[0]["variant"]),
            "best_gain": float(score_report.iloc[0]["gain"]),
            "highest_value_family": str(family_report.iloc[0]["family"]),
            "highest_family_value": float(family_report.iloc[0]["family_value"]),
        },
        indent=2,
    )
)
print("saved:", OUT_DIR)
