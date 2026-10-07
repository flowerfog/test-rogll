"""Validate strict-spatial fusion with the rebuilt Huber bias lineage.

This deliberately ignores the historical pool.strong_bias curve. It uses the
newly reconstructed OOF curve, selects blend weights on four folds, and scores
the held-out fold. On pass, it promotes the rebuilt bundle for Kaggle use while
retaining lineage_match=False in its metadata.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


OOF_DIR = Path("/root/autodl-tmp/oof_exports")
SPATIAL_PATH = (
    OOF_DIR
    / "multicut_k5_strict_outer_spatial_v1"
    / "strict_outer_spatial_oof.pkl"
)
BIAS_RUN = OOF_DIR / "strong_bias_huber_rebuild_v1"
BIAS_OOF_PATH = BIAS_RUN / "strong_bias_rebuilt_oof.csv"
BUNDLE_PATH = BIAS_RUN / "strong_bias_huber_bundle_v1.joblib"
RUN_DIR = OOF_DIR / "strict_spatial_rebuilt_bias_blend_audit_v1"

WEIGHT_STEP = 0.025
FIXED_WEIGHTS = (0.40, 0.45, 0.50)  # weight on strict-spatial


def _rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _cosine(y, a, b):
    ea = np.asarray(a, float) - np.asarray(y, float)
    eb = np.asarray(b, float) - np.asarray(y, float)
    denominator = np.sqrt(np.dot(ea, ea) * np.dot(eb, eb))
    return float(np.dot(ea, eb) / denominator) if denominator else np.nan


for path in (SPATIAL_PATH, BIAS_OOF_PATH, BUNDLE_PATH):
    if not path.exists():
        raise FileNotFoundError(path)
RUN_DIR.mkdir(parents=True, exist_ok=True)

spatial = pd.read_pickle(SPATIAL_PATH)
spatial["id"] = spatial["id"].astype(str)
required_spatial = {"id", "well", "fold", "tvt_true", "pred_spatial"}
missing = required_spatial - set(spatial.columns)
if missing:
    raise RuntimeError(f"Spatial OOF missing: {sorted(missing)}")

bias = pd.read_csv(
    BIAS_OOF_PATH,
    usecols=["id", "tvt_true", "strong_bias_rebuilt"],
    dtype={"id": str, "tvt_true": np.float32, "strong_bias_rebuilt": np.float32},
)
bias["id"] = bias["id"].astype(str)

data = spatial[["id", "well", "fold", "tvt_true", "pred_spatial"]].merge(
    bias.rename(columns={"tvt_true": "tvt_true_bias"}),
    on="id",
    how="inner",
    validate="one_to_one",
)
if len(data) != len(spatial):
    raise RuntimeError(f"Merge mismatch: {len(spatial)} -> {len(data)}")
truth_difference = (
    data["tvt_true"].to_numpy(float) - data["tvt_true_bias"].to_numpy(float)
)
if np.max(np.abs(truth_difference)) > 1e-3:
    raise RuntimeError("Truth mismatch between spatial and rebuilt-bias OOF")

truth = data["tvt_true"].to_numpy(float)
spatial_prediction = data["pred_spatial"].to_numpy(float)
bias_prediction = data["strong_bias_rebuilt"].to_numpy(float)
folds = data["fold"].to_numpy(int)

weights = np.arange(0.0, 1.0 + WEIGHT_STEP / 2, WEIGHT_STEP)
nested = np.full(len(data), np.nan, dtype=np.float64)
fold_rows = []
for fold in sorted(np.unique(folds)):
    train_mask = folds != fold
    valid_mask = ~train_mask
    train_truth = truth[train_mask]
    train_spatial = spatial_prediction[train_mask]
    train_bias = bias_prediction[train_mask]
    train_scores = np.asarray(
        [
            _rmse(train_truth, weight * train_spatial + (1.0 - weight) * train_bias)
            for weight in weights
        ]
    )
    weight = float(weights[int(np.argmin(train_scores))])
    valid_prediction = weight * spatial_prediction[valid_mask] + (1.0 - weight) * bias_prediction[valid_mask]
    nested[valid_mask] = valid_prediction
    spatial_rmse = _rmse(truth[valid_mask], spatial_prediction[valid_mask])
    bias_rmse = _rmse(truth[valid_mask], bias_prediction[valid_mask])
    blend_rmse = _rmse(truth[valid_mask], valid_prediction)
    fold_rows.append(
        {
            "fold": int(fold),
            "weight_strict_spatial": weight,
            "weight_rebuilt_bias": 1.0 - weight,
            "spatial_rmse": spatial_rmse,
            "rebuilt_bias_rmse": bias_rmse,
            "blend_rmse": blend_rmse,
            "gain_vs_best": min(spatial_rmse, bias_rmse) - blend_rmse,
        }
    )

fold_frame = pd.DataFrame(fold_rows)
fixed_rows = []
for weight in FIXED_WEIGHTS:
    prediction = weight * spatial_prediction + (1.0 - weight) * bias_prediction
    fold_gains = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        score = _rmse(truth[mask], prediction[mask])
        best = min(
            _rmse(truth[mask], spatial_prediction[mask]),
            _rmse(truth[mask], bias_prediction[mask]),
        )
        fold_gains.append(best - score)
    fixed_rows.append(
        {
            "weight_strict_spatial": weight,
            "weight_rebuilt_bias": 1.0 - weight,
            "rmse": _rmse(truth, prediction),
            "gain_vs_best_overall": min(
                _rmse(truth, spatial_prediction), _rmse(truth, bias_prediction)
            )
            - _rmse(truth, prediction),
            "mean_fold_gain_vs_best": float(np.mean(fold_gains)),
            "min_fold_gain_vs_best": float(np.min(fold_gains)),
            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
        }
    )
fixed_frame = pd.DataFrame(fixed_rows).sort_values("rmse").reset_index(drop=True)

nested_rmse = _rmse(truth, nested)
best_component_rmse = min(
    _rmse(truth, spatial_prediction), _rmse(truth, bias_prediction)
)
positive_folds = int((fold_frame["gain_vs_best"] > 0).sum())
min_fold_gain = float(fold_frame["gain_vs_best"].min())
blend_pass = bool(
    best_component_rmse - nested_rmse >= 0.05
    and positive_folds == 5
    and min_fold_gain > 0
)

recommended = fixed_frame.iloc[0]
result = {
    "strict_spatial_rmse": _rmse(truth, spatial_prediction),
    "rebuilt_bias_rmse": _rmse(truth, bias_prediction),
    "error_cosine": _cosine(truth, spatial_prediction, bias_prediction),
    "nested_rmse": nested_rmse,
    "nested_gain_vs_best": best_component_rmse - nested_rmse,
    "nested_positive_folds": positive_folds,
    "nested_min_fold_gain": min_fold_gain,
    "mean_nested_weight_strict": float(fold_frame["weight_strict_spatial"].mean()),
    "recommended_fixed_weight_strict": float(recommended["weight_strict_spatial"]),
    "recommended_fixed_weight_bias": float(recommended["weight_rebuilt_bias"]),
    "recommended_fixed_rmse": float(recommended["rmse"]),
    "blend_pass": blend_pass,
}

print("\nNESTED FOLDS")
display(fold_frame)
print("\nFIXED WEIGHTS")
display(fixed_frame)
print("\nRESULT")
print(result)

fold_frame.to_csv(RUN_DIR / "nested_folds.csv", index=False)
fixed_frame.to_csv(RUN_DIR / "fixed_weights.csv", index=False)
(RUN_DIR / "result.json").write_text(json.dumps(result, indent=2))

bundle = joblib.load(BUNDLE_PATH)
summary = bundle["oof_summary"]
standalone_pass = bool(
    summary.get("gain", -np.inf) >= 0.10
    and summary.get("positive_folds", 0) == 5
    and summary.get("min_fold_gain", -np.inf) > 0
)
bundle["oof_summary"]["performance_pass"] = standalone_pass
if blend_pass:
    bundle["oof_summary"]["strict_pass"] = True
    bundle["oof_summary"]["lineage_match"] = False
    bundle["deployment_status"] = "validated_rebuild_variant"
    bundle["strict_spatial_blend_audit"] = result
    print("PROMOTED FOR STANDALONE + BLEND")
else:
    bundle["deployment_status"] = (
        "validated_standalone_rebuild_only"
        if standalone_pass
        else "not_validated"
    )
    print("STANDALONE ONLY; do not use the blend.")
joblib.dump(bundle, BUNDLE_PATH, compress=3)
print("UPDATED BUNDLE:", BUNDLE_PATH)
