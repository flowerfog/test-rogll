from pathlib import Path
import json

import joblib
import numpy as np
from lightgbm import LGBMRegressor


# Run after cpfamily_strong_exact_oof.py in the same notebook session.
required = [
    "X", "y_cp", "feature_cols", "aux_cols", "fold_model_rows",
    "N_CP", "WIN_ROWS", "CP_ALPHA", "CP_CAP", "OOF_DIR", "SEED",
]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(f"run exact-base OOF cell first; missing: {missing}")

best_iterations = [
    int(row["best_iteration"])
    for row in fold_model_rows
    if int(row["best_iteration"]) > 0
]
if not best_iterations:
    best_iterations = [500]

final_estimators = int(round(np.median(best_iterations) * 1.15))
final_estimators = max(250, min(final_estimators, 1600))

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
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)
model.fit(X, y_cp)

bundle = {
    "version": "strong_exact_cpfam_k8_w192_huber_v1",
    "model": model,
    "feature_cols": list(feature_cols),
    "aux_cols": list(aux_cols),
    "base_col": "strong",
    "reference_col": "cp12",
    "n_cp": int(N_CP),
    "win_rows": int(WIN_ROWS),
    "cp_alpha": float(CP_ALPHA),
    "cp_cap": float(CP_CAP),
    "blend_cp12": 0.50,
    "blend_k8": 0.50,
    "final_estimators": final_estimators,
}

bundle_path = Path(OOF_DIR) / "strong_exact_cpfam_k8_bundle.joblib"
joblib.dump(bundle, bundle_path, compress=3)

meta = {k: v for k, v in bundle.items() if k != "model"}
meta["bundle_path"] = str(bundle_path)
meta_path = Path(OOF_DIR) / "strong_exact_cpfam_k8_bundle.json"
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("saved bundle:", bundle_path)
print("saved metadata:", meta_path)
print(meta)
