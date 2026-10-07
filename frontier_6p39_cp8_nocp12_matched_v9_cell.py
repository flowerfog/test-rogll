"""Matched five-fold audit and deployment refit for a CP12-free CP8 head.

Run on AutoDL after ``frontier_6p39_cp8_parent_audit_v2_cell.py`` has
persisted its control-point table and OOF output. This cell does not rebuild
or consume CP12 features. It also removes candidate dispersion features whose
original values included CP12 indirectly.
"""

from pathlib import Path
import hashlib
import json
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.ndimage import gaussian_filter1d


SEED = 20260724
N_SPLITS = 5
N_CP = 8
WIN_ROWS = 192
CP_ALPHA = 1.0
CP_CAP = 12.0

ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = ROOT / "frontier_6p39_cp8_parent_audit_v2"
OUT_DIR = ROOT / "frontier_6p39_cp8_nocp12_matched_v9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CP_TABLE = SOURCE_DIR / "cp_training_table.parquet"
SOURCE_OOF = SOURCE_DIR / "cp8_rebased_oof.pkl"

for required in (CP_TABLE, SOURCE_OOF):
    if not required.exists():
        raise FileNotFoundError(
            f"Missing {required}. Run frontier_6p39_cp8_parent_audit_v2_cell.py once."
        )

started = time.time()
cp = pd.read_parquet(CP_TABLE)
oof = pd.read_pickle(SOURCE_OOF)
oof["id"] = oof["id"].astype(str)
oof["well"] = oof["well"].astype(str)
oof = oof.sort_values(["well", "row_idx"]).reset_index(drop=True)

required_cp = {"well", "center_global", "center_local", "target_resid"}
required_oof = {
    "id", "well", "row_idx", "tvt_true", "frontier_6p39_core",
    "fold", "cp8_rebased",
}
if not required_cp.issubset(cp.columns):
    raise RuntimeError(f"Malformed CP table: {sorted(required_cp.difference(cp.columns))}")
if not required_oof.issubset(oof.columns):
    raise RuntimeError(f"Malformed source OOF: {sorted(required_oof.difference(oof.columns))}")
if cp["well"].astype(str).nunique() != 773 or oof["well"].nunique() != 773:
    raise RuntimeError(
        f"Expected 773 wells, got CP={cp['well'].nunique()} OOF={oof['well'].nunique()}"
    )

cp["well"] = cp["well"].astype(str)
fold_map = (
    oof[["well", "fold"]]
    .drop_duplicates()
    .set_index("well")["fold"]
    .astype(int)
    .to_dict()
)
cp_folds = cp["well"].map(fold_map)
if cp_folds.isna().any() or set(cp_folds.astype(int)) != set(range(1, 6)):
    raise RuntimeError("Frozen five-fold map does not cover the CP table")
cp_folds = cp_folds.to_numpy(np.int8)

# A strict CP12-free schema. The two dispersion columns are excluded because
# the original CP8 table computed them over a stack that included CP12.
forbidden_exact = {"pred_std_mean", "pred_range_mean", "oof_pred_resid"}
forbidden_prefixes = ("cp12_",)
feature_cols = [
    column
    for column in cp.columns
    if column not in {
        "well", "center_global", "center_local", "target_resid",
        *forbidden_exact,
    }
    and not column.startswith(forbidden_prefixes)
    and pd.api.types.is_numeric_dtype(cp[column])
]
if any("cp12" in column.lower() for column in feature_cols):
    raise RuntimeError("CP12 leaked into the no-CP12 feature schema")
for family in ("ridge", "selector", "fleongg", "sp45"):
    if f"{family}_minus_base_mean" not in feature_cols:
        raise RuntimeError(f"Missing deployable feature family: {family}")

X = (
    cp[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)
y = pd.to_numeric(cp["target_resid"], errors="coerce").to_numpy(np.float32)
if not np.isfinite(y).all():
    raise RuntimeError("Non-finite CP targets")

print(
    "NOCP12 MATRIX",
    {
        "cp_rows": len(cp),
        "wells": cp["well"].nunique(),
        "features": len(feature_cols),
        "removed": sorted(
            column for column in cp.columns
            if column.startswith("cp12_") or column in forbidden_exact
        ),
    },
    flush=True,
)


def make_model(seed, n_estimators=2000):
    return LGBMRegressor(
        objective="huber",
        alpha=0.90,
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.20,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


control_oof = np.full(len(cp), np.nan, np.float32)
fold_models = []
for fold in range(1, N_SPLITS + 1):
    train = np.flatnonzero(cp_folds != fold)
    valid = np.flatnonzero(cp_folds == fold)
    model = make_model(SEED + fold)
    model.fit(
        X.iloc[train],
        y[train],
        eval_set=[(X.iloc[valid], y[valid])],
        eval_metric="rmse",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    control_oof[valid] = model.predict(X.iloc[valid]).astype(np.float32)
    report = {
        "fold": fold,
        "train_cp_rows": int(len(train)),
        "valid_cp_rows": int(len(valid)),
        "valid_wells": int(cp.iloc[valid]["well"].nunique()),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "cp_target_rmse": float(np.sqrt(np.mean((control_oof[valid] - y[valid]) ** 2))),
    }
    fold_models.append(report)
    print("NOCP12 FOLD MODEL", report, flush=True)

if not np.isfinite(control_oof).all():
    raise RuntimeError("Incomplete no-CP12 control OOF")
cp["nocp12_oof_resid"] = control_oof

# Reconstruct the row-level move in exactly the same way as the original CP8.
raw_move = np.zeros(len(oof), np.float32)
oof_groups = {
    str(well): np.asarray(index, np.int64)
    for well, index in oof.groupby("well", sort=False).indices.items()
}
for well, index in oof_groups.items():
    part = cp[cp["well"].eq(well)].sort_values("center_local")
    if part.empty:
        raise RuntimeError(f"No CP rows for {well}")
    xp = part["center_local"].to_numpy(float)
    fp = part["nocp12_oof_resid"].to_numpy(float)
    keep = np.r_[True, np.diff(xp) > 0]
    xp, fp = xp[keep], fp[keep]
    n = len(index)
    if xp.min() < 0 or xp.max() >= n:
        raise RuntimeError(f"{well}: CP center outside scored rows")
    move = (
        np.full(n, fp[0], float)
        if len(xp) == 1
        else np.interp(np.arange(n, dtype=float), xp, fp)
    )
    sigma = max(4.0, n / N_CP / 8.0)
    raw_move[index] = gaussian_filter1d(move, sigma=sigma, mode="nearest").astype(np.float32)

move = np.clip(CP_ALPHA * raw_move, -CP_CAP, CP_CAP)
truth = oof["tvt_true"].to_numpy(np.float64)
parent = oof["frontier_6p39_core"].to_numpy(np.float64)
original_cp8 = oof["cp8_rebased"].to_numpy(np.float64)
candidate = parent + move.astype(np.float64)

def metric(name, prediction, mask):
    error = np.asarray(prediction, float)[mask] - truth[mask]
    return {
        "endpoint": name,
        "rows": int(mask.sum()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }


rows = []
all_rows = np.ones(len(oof), dtype=bool)
endpoints = {
    "parent_6p39": parent,
    "original_cp8_with_cp12": original_cp8,
    "cp8_nocp12": candidate,
}
for fold in range(1, 6):
    mask = oof["fold"].to_numpy(int) == fold
    for name, prediction in endpoints.items():
        rows.append({"fold": fold, **metric(name, prediction, mask)})
for name, prediction in endpoints.items():
    rows.append({"fold": 0, **metric(name, prediction, all_rows)})

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT_DIR / "matched_comparison.csv", index=False)
print("\nNOCP12 MATCHED COMPARISON")
print(comparison.to_string(index=False), flush=True)

overall = comparison[comparison["fold"].eq(0)].set_index("endpoint")
fold_wide = comparison[comparison["fold"].ne(0)].pivot(
    index="fold", columns="endpoint", values="rmse"
)
parent_rmse = float(overall.loc["parent_6p39", "rmse"])
original_rmse = float(overall.loc["original_cp8_with_cp12", "rmse"])
nocp12_rmse = float(overall.loc["cp8_nocp12", "rmse"])
fold_gains = fold_wide["parent_6p39"] - fold_wide["cp8_nocp12"]

gates = {
    "gain_vs_parent_ge_0p08": bool(parent_rmse - nocp12_rmse >= 0.08),
    "positive_folds_5of5": bool((fold_gains > 0).all()),
    "loss_vs_original_cp8_le_0p05": bool(nocp12_rmse - original_rmse <= 0.05),
    "cp12_absent_from_schema": bool(
        not any("cp12" in column.lower() for column in feature_cols)
    ),
}
strict_pass = bool(all(gates.values()))

pd.DataFrame(fold_models).to_csv(OUT_DIR / "model_folds.csv", index=False)
row_output = oof[["id", "well", "row_idx", "fold", "tvt_true", "frontier_6p39_core"]].copy()
row_output["cp8_original"] = original_cp8.astype(np.float32)
row_output["cp8_nocp12"] = candidate.astype(np.float32)
row_output["nocp12_move"] = move.astype(np.float32)
row_output.to_parquet(OUT_DIR / "cp8_nocp12_oof.parquet", index=False)

bundle_path = None
if strict_pass:
    iterations = [row["best_iteration"] for row in fold_models if row["best_iteration"] > 0]
    final_estimators = int(np.clip(round(np.median(iterations) * 1.15), 250, 1600))
    models = []
    for member, offset in enumerate((0, 101, 211, 307, 401), start=1):
        model = make_model(SEED + offset, n_estimators=final_estimators)
        model.fit(X, y)
        models.append(model)
        print("NOCP12 DEPLOYMENT MEMBER", member, flush=True)
    bundle = {
        "version": "frontier_6p39_cp8_nocp12_w192_huber_v9",
        "models": models,
        "feature_cols": feature_cols,
        "aux_cols": ["ridge", "selector", "fleongg", "sp45"],
        "base_col": "frontier_6p39_core",
        "reference_col": None,
        "n_cp": N_CP,
        "win_rows": WIN_ROWS,
        "cp_alpha": CP_ALPHA,
        "cp_cap": CP_CAP,
        "final_estimators": final_estimators,
        "training_cp_rows": int(len(cp)),
        "training_wells": int(cp["well"].nunique()),
        "excluded_features": sorted(
            column for column in cp.columns
            if column.startswith("cp12_") or column in forbidden_exact
        ),
        "placement": "after 6.39 core and before guarded/visible-prefix overlays",
    }
    deployment = OUT_DIR / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    bundle_path = deployment / "frontier_6p39_cp8_nocp12_bundle_v9.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    metadata = {key: value for key, value in bundle.items() if key != "models"}
    metadata.update({
        "members": len(models),
        "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    })
    (deployment / "frontier_6p39_cp8_nocp12_bundle_v9.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

verdict = {
    "experiment": "frontier_6p39_cp8_nocp12_matched_v9",
    "stage": "fivefold_matched_cp12_ablation_and_conditional_refit",
    "parent_rmse": parent_rmse,
    "original_cp8_rmse": original_rmse,
    "nocp12_rmse": nocp12_rmse,
    "gain_vs_parent": parent_rmse - nocp12_rmse,
    "loss_vs_original_cp8": nocp12_rmse - original_rmse,
    "fold_gain_vs_parent": {str(int(k)): float(v) for k, v in fold_gains.items()},
    "features": len(feature_cols),
    "aux_cols": ["ridge", "selector", "fleongg", "sp45"],
    "gates": gates,
    "strict_pass": strict_pass,
    "bundle": str(bundle_path) if bundle_path is not None else None,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(Path("/root/autodl-tmp")).free / (1024 ** 3),
    "next": (
        "PASS -> upload deployment folder and run the no-CP12 Kaggle inference cell"
        if strict_pass
        else "FAIL -> close CP8; keep the published 6.39 core without historical CP"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
