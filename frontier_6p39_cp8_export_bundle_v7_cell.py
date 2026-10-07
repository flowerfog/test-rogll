"""Fit and export the deployment CP8 model for the reconstructed 6.39 core.

Run this immediately after ``frontier_6p39_cp8_parent_audit_v2_cell.py`` in
the same AutoDL kernel. The audit builds the exact fold-clean feature table;
this cell only refits on all 773 training wells and writes a Kaggle artifact.
"""

from pathlib import Path
import hashlib
import json
import runpy

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


# Do not trust generic notebook globals such as X/y_cp/feature_cols here: later
# experiments commonly reuse those names. Rebuild the matrix from the CP8
# audit's own control-point table and validate its fingerprint.
out_dir = Path("/root/autodl-tmp/oof_exports/frontier_6p39_cp8_parent_audit_v2")
cp_table_path = out_dir / "cp_training_table.parquet"

if cp_table_path.exists():
    cp_fit = pd.read_parquet(cp_table_path)
    print("loaded persisted CP8 training table:", cp_table_path, flush=True)
elif isinstance(globals().get("cp"), pd.DataFrame):
    cp_fit = globals()["cp"].copy()
    print("using live CP8 training table", flush=True)
else:
    source_candidates = [
        Path("/root/autodl-tmp/frontier_6p39_cp8_parent_audit_v2_cell.py"),
        Path("frontier_6p39_cp8_parent_audit_v2_cell.py"),
    ]
    source_path = next((path for path in source_candidates if path.exists()), None)
    if source_path is None:
        raise RuntimeError(
            "The live `cp` variable was overwritten and no persisted CP table exists. "
            "Upload frontier_6p39_cp8_parent_audit_v2_cell.py to /root/autodl-tmp "
            "or rerun that audit cell once."
        )
    print("rebuilding CP8 audit in an isolated namespace:", source_path, flush=True)
    audit_namespace = runpy.run_path(str(source_path), run_name="frontier_6p39_cp8_rebuild")
    rebuilt = audit_namespace.get("cp")
    if not isinstance(rebuilt, pd.DataFrame):
        raise RuntimeError("Isolated CP8 audit did not produce a DataFrame named `cp`")
    cp_fit = rebuilt.copy()
    cp_fit.drop(columns=["oof_pred_resid"], errors="ignore").to_parquet(
        cp_table_path, index=False
    )
    print("persisted rebuilt CP8 training table:", cp_table_path, flush=True)
required_cp_columns = {
    "well", "center_global", "center_local", "target_resid",
    "base_center", "cp12_minus_base_mean", "sp45_minus_base_mean",
}
missing_cp = sorted(required_cp_columns.difference(cp_fit.columns))
if missing_cp:
    raise RuntimeError(
        "The live `cp` table is not the frontier_6p39 CP8 audit table; "
        f"missing columns: {missing_cp}. Rerun only the CP8 parent audit cell."
    )

training_wells = int(cp_fit["well"].astype(str).nunique())
if training_wells != 773 or not (6100 <= len(cp_fit) <= 6200):
    raise RuntimeError(
        "Unexpected CP8 table shape: "
        f"rows={len(cp_fit)}, wells={training_wells}; expected about 6184/773. "
        "Rerun only the CP8 parent audit cell."
    )

feature_cols_fit = [
    column for column in cp_fit.columns
    if column not in {
        "well", "center_global", "center_local", "target_resid",
        # Added only after OOF fitting for diagnostics. It is unavailable at
        # deployment and including it would leak the audit model prediction.
        "oof_pred_resid",
    }
    and pd.api.types.is_numeric_dtype(cp_fit[column])
]
x_fit = (
    cp_fit[feature_cols_fit]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)
y_fit = pd.to_numeric(cp_fit["target_resid"], errors="coerce").to_numpy(np.float32)
if len(x_fit) != len(y_fit) or not np.isfinite(y_fit).all():
    raise RuntimeError(f"Invalid reconstructed CP8 matrix: X={x_fit.shape}, y={y_fit.shape}")
if x_fit.shape[1] != 126:
    raise RuntimeError(
        f"Unexpected CP8 feature count {x_fit.shape[1]}; expected exactly 126"
    )

aux_cols_fit = ["cp12", "ridge", "selector", "fleongg", "sp45"]
for column in aux_cols_fit:
    if f"{column}_minus_base_mean" not in feature_cols_fit:
        raise RuntimeError(f"CP8 table is missing deployment feature family: {column}")

fold_report_path = out_dir / "model_folds.csv"
if not fold_report_path.exists():
    raise FileNotFoundError(fold_report_path)
fold_report = pd.read_csv(fold_report_path)
if "best_iteration" not in fold_report:
    raise RuntimeError(f"Malformed CP8 fold report: {fold_report_path}")

best_iterations = [
    int(value) for value in fold_report["best_iteration"].tolist()
    if int(value) > 0
]
if not best_iterations:
    raise RuntimeError("The OOF audit did not record any valid best_iteration values")

final_estimators = int(round(float(np.median(best_iterations)) * 1.15))
final_estimators = max(250, min(final_estimators, 1600))
seed_base = 20260711

print(
    "CP8 EXPORT MATRIX",
    {
        "rows": len(x_fit),
        "features": x_fit.shape[1],
        "wells": training_wells,
        "labels": len(y_fit),
        "trees": final_estimators,
    },
    flush=True,
)

models = []
for member, seed_offset in enumerate((0, 101, 211, 307, 401), start=1):
    model = LGBMRegressor(
        objective="huber",
        alpha=0.90,
        n_estimators=final_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.20,
        reg_lambda=4.0,
        random_state=seed_base + seed_offset,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_fit, y_fit)
    models.append(model)
    print(
        "CP8 deployment member",
        {"member": member, "seed": seed_base + seed_offset, "trees": final_estimators},
        flush=True,
    )

bundle = {
    "version": "frontier_6p39_cp8_w192_huber_v7",
    "models": models,
    "feature_cols": feature_cols_fit,
    "aux_cols": aux_cols_fit,
    "base_col": "frontier_6p39_core",
    "reference_col": "cp12",
    "n_cp": 8,
    "win_rows": 192,
    "cp_alpha": 1.0,
    "cp_cap": 12.0,
    "final_estimators": final_estimators,
    "training_cp_rows": int(len(x_fit)),
    "training_wells": training_wells,
    "parent": "reconstructed fold-clean frontier_6p39_core",
    "placement": "after 6.39 core blend and before guarded/visible-prefix overlays",
}

deployment_dir = out_dir / "deployment"
deployment_dir.mkdir(parents=True, exist_ok=True)
bundle_path = deployment_dir / "frontier_6p39_cp8_bundle_v7.joblib"
joblib.dump(bundle, bundle_path, compress=3)

sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
metadata = {key: value for key, value in bundle.items() if key != "models"}
metadata.update({
    "members": len(models),
    "bundle": str(bundle_path),
    "sha256": sha256,
})
metadata_path = deployment_dir / "frontier_6p39_cp8_bundle_v7.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("saved:", bundle_path, flush=True)
print("sha256:", sha256, flush=True)
print("upload the deployment directory as a Kaggle dataset", flush=True)
