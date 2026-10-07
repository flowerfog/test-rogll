"""Kaggle inference cell for the strict-spatial K5 full LGB bundle.

Run the original fleongg function-definition cells first. This cell rebuilds
the deployment features from the competition train/test files, loads the small
full-data bundle, and writes a standalone strict-spatial submission.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

BUNDLE_OVERRIDE = None
DATA_ROOT_OVERRIDE = None
OUTPUT_NAME = "submission_strict_spatial_lgb.csv"
WRITE_AS_FINAL_SUBMISSION = True
MAX_FEATURE_JOBS = 4


def _first_existing(paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def _find_data_root():
    candidates = []
    if DATA_ROOT_OVERRIDE:
        candidates.append(Path(DATA_ROOT_OVERRIDE))

    cfg = globals().get("CFG")
    if cfg is not None:
        for attr in ("DATA", "dataset_path", "data_path", "root"):
            value = getattr(cfg, attr, None)
            if value:
                candidates.append(Path(value))

    candidates.extend(
        [
            Path("/kaggle/input/rogii-wellbore-geology-prediction"),
            Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
            Path.cwd(),
        ]
    )
    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("Cannot locate competition data root with train/ and test/")


def _find_bundle():
    if BUNDLE_OVERRIDE:
        path = Path(BUNDLE_OVERRIDE)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    names = (
        "strict_spatial_full_lgb_bundle_v1.joblib",
        "strict_spatial_full_lgb_bundle.joblib",
    )
    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            matches.extend(root.rglob(name))
    matches = sorted(set(matches), key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    if not matches:
        raise FileNotFoundError(
            "Cannot find strict_spatial_full_lgb_bundle_v1.joblib. "
            "Upload it as a Kaggle Dataset or set BUNDLE_OVERRIDE."
        )
    return matches[0]


required_globals = [
    "FormationPlaneKNN",
    "DenseANCCImputer",
    "build_likpf",
    "build_features",
    "add_likpf_features",
]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original fleongg function-definition cells first; missing: "
        + repr(missing_globals)
    )


started = time.time()
DATA_ROOT = _find_data_root()
BUNDLE_PATH = _find_bundle()
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
WORK.mkdir(parents=True, exist_ok=True)

print("DATA_ROOT:", DATA_ROOT)
print("BUNDLE:", BUNDLE_PATH)

bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
feature_cols = list(bundle["feature_columns"])
post = dict(bundle["postprocess"])

if len(feature_cols) != len(set(feature_cols)):
    raise RuntimeError("Bundle contains duplicate feature names")

expected_spatial = [
    "tvt_dense_d",
    "tvt_densew_d",
    "tvt_dense50_d",
    "dense_dist",
    "dense_nb_std",
    "pf_vs_spatial",
    "pf_vs_dense",
    "spatial_vs_dense",
]
missing_bundle_spatial = [c for c in expected_spatial if c not in feature_cols]
if missing_bundle_spatial:
    raise RuntimeError(
        "Wrong bundle lineage: required strict-spatial columns absent: "
        + repr(missing_bundle_spatial)
    )


# The original helpers use CFG.DATA internally. Point them at the detected root.
cfg = globals().get("CFG")
if cfg is None:
    raise RuntimeError("CFG is missing; run the original notebook setup cells")
for attr in ("DATA", "dataset_path"):
    try:
        setattr(cfg, attr, DATA_ROOT)
    except Exception:
        pass
if hasattr(cfg, "n_jobs"):
    try:
        cfg.n_jobs = min(MAX_FEATURE_JOBS, max(1, int(cfg.n_jobs)))
    except Exception:
        cfg.n_jobs = MAX_FEATURE_JOBS


def _well_ids(directory):
    ids = []
    for path in sorted(Path(directory).glob("*.csv")):
        name = path.stem
        if name.endswith("__horizontal_well"):
            ids.append(name[: -len("__horizontal_well")])
    return sorted(set(ids))


train_wids = _well_ids(DATA_ROOT / "train")
test_wids = _well_ids(DATA_ROOT / "test")
if not train_wids or not test_wids:
    raise RuntimeError(
        f"Failed to enumerate wells: train={len(train_wids)}, test={len(test_wids)}"
    )

sample_path = _first_existing(
    [
        DATA_ROOT / "sample_submission.csv",
        DATA_ROOT / "sample_submission" / "sample_submission.csv",
    ]
)
if sample_path is None:
    candidates = list(DATA_ROOT.rglob("sample_submission.csv"))
    sample_path = candidates[0] if candidates else None
if sample_path is None:
    raise FileNotFoundError("Cannot find sample_submission.csv")

sample = pd.read_csv(sample_path)
if "id" not in sample.columns:
    raise RuntimeError(f"sample submission has no id column: {sample.columns.tolist()}")
target_col = "tvt" if "tvt" in sample.columns else [c for c in sample.columns if c != "id"][0]

sample_wids = set(sample["id"].astype(str).str.split("_", n=1).str[0])
if sample_wids != set(test_wids):
    only_sample = sorted(sample_wids - set(test_wids))[:10]
    only_files = sorted(set(test_wids) - sample_wids)[:10]
    raise RuntimeError(
        "Test-well lineage mismatch. "
        f"only_sample={only_sample}, only_files={only_files}"
    )


# Rebuild deployment spatial references from all train wells. Test wells are
# never part of these imputers, matching full-data deployment semantics.
print("initializing deployment FI/DI from", len(train_wids), "training wells")
globals()["_FI"] = FormationPlaneKNN(train_wids, DATA_ROOT / "train")
globals()["_DI"] = DenseANCCImputer(train_wids, DATA_ROOT / "train")

print("building likelihood-PF test features for", len(test_wids), "wells")
likpf_test = build_likpf(test_wids, "test")
if likpf_test is None or len(likpf_test) == 0:
    raise RuntimeError("build_likpf returned no test rows")

print("building strict-spatial test features")
test_features = build_features(test_wids, "test", is_train=False)
if test_features is None or len(test_features) == 0:
    raise RuntimeError("build_features returned no test rows")
test_features = add_likpf_features(test_features, likpf_test).reset_index(drop=True)

if "id" not in test_features.columns or "well" not in test_features.columns:
    raise RuntimeError("Built features are missing id/well columns")
if test_features["id"].duplicated().any():
    dup = test_features.loc[test_features["id"].duplicated(), "id"].head().tolist()
    raise RuntimeError(f"Duplicate feature IDs: {dup}")

missing_features = [c for c in feature_cols if c not in test_features.columns]
deployment_nan_features = {
    # Historical K5 cache leakage: this column was accidentally retained in
    # the model schema, while strict official-fold validation did not contain
    # it and therefore evaluated it as missing. Test must follow that exact
    # official validation path.
    "_sample_weight_k5",
}
allowed_missing = [c for c in missing_features if c in deployment_nan_features]
unexpected_missing = [c for c in missing_features if c not in deployment_nan_features]
if unexpected_missing:
    raise RuntimeError(
        f"Missing {len(unexpected_missing)} bundle features; "
        f"first 30: {unexpected_missing[:30]}"
    )
for column in allowed_missing:
    test_features[column] = np.nan
if allowed_missing:
    print("deployment-only missing features filled with NaN:", allowed_missing)

all_nan_features = [
    c
    for c in feature_cols
    if c not in deployment_nan_features and test_features[c].notna().sum() == 0
]
if all_nan_features:
    raise RuntimeError(
        f"Bundle features are entirely NaN on test; first 30: {all_nan_features[:30]}"
    )

X = test_features.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
delta = np.asarray(model.predict(X), dtype=np.float64)
if len(delta) != len(test_features) or not np.isfinite(delta).all():
    raise RuntimeError("Model produced invalid predictions")


def _col(name):
    if name not in test_features.columns:
        raise RuntimeError(f"Postprocess column missing: {name}")
    return test_features[name].to_numpy(dtype=np.float64)


last = _col("last_known_tvt")
md_since = np.maximum(_col("md_since"), 0.0)
alpha = float(post.get("alpha", 0.985))
tau = float(post.get("tau", 120.0))
w_pf = float(post.get("w_pf_ancc", post.get("w_pf", 0.05)))
w_model = float(post.get("w_model", post.get("w_sub1", 0.55)))
likpf_col = str(post.get("likpf_col", "likpf_scale_5"))
sg_win = int(post.get("sg_win", 61))
sg_poly = int(post.get("sg_poly", 3))

pf_delta = _col("pf_ancc") - last
lik = _col(likpf_col)
lik = np.where(np.isfinite(lik), lik, last)
warm = 1.0 - np.exp(-md_since / max(tau, 1e-6))
learned_delta = alpha * warm * ((1.0 - w_pf) * delta + w_pf * pf_delta)
prediction = last + w_model * learned_delta + (1.0 - w_model) * (lik - last)

# Match the training/deployment per-well Savitzky-Golay postprocess exactly.
smoothed = prediction.copy()
for _, idx in test_features.groupby("well", sort=False).indices.items():
    idx = np.asarray(idx, dtype=np.int64)
    n = len(idx)
    window = min(sg_win, n if n % 2 == 1 else n - 1)
    if window <= sg_poly:
        continue
    smoothed[idx] = savgol_filter(
        prediction[idx],
        window_length=window,
        polyorder=min(sg_poly, window - 1),
        mode="interp",
    )

pred_df = pd.DataFrame({"id": test_features["id"].astype(str), target_col: smoothed})
submission = sample[["id"]].copy()
submission["id"] = submission["id"].astype(str)
submission = submission.merge(pred_df, on="id", how="left", validate="one_to_one")

if submission[target_col].isna().any():
    missing_ids = submission.loc[submission[target_col].isna(), "id"].head().tolist()
    raise RuntimeError(f"Missing predictions after sample alignment: {missing_ids}")
if len(submission) != len(sample):
    raise RuntimeError(f"Submission row mismatch: {len(submission)} != {len(sample)}")

output_path = WORK / OUTPUT_NAME
submission.to_csv(output_path, index=False)
if WRITE_AS_FINAL_SUBMISSION:
    submission.to_csv(WORK / "submission.csv", index=False)

move = smoothed - last
audit = {
    "data_root": str(DATA_ROOT),
    "bundle": str(BUNDLE_PATH),
    "rows": int(len(submission)),
    "wells": int(len(test_wids)),
    "features": int(len(feature_cols)),
    "deployment_nan_features": allowed_missing,
    "spatial_features": int(bundle.get("spatial_feature_count", -1)),
    "postprocess": post,
    "prediction_min": float(np.min(smoothed)),
    "prediction_max": float(np.max(smoothed)),
    "mean_abs_move_from_anchor": float(np.mean(np.abs(move))),
    "p95_abs_move_from_anchor": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move_from_anchor": float(np.max(np.abs(move))),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "output": str(output_path),
}
with open(WORK / "strict_spatial_inference_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print("\nSTRICT-SPATIAL INFERENCE COMPLETE")
print(json.dumps(audit, indent=2))
print("saved:", output_path)
if WRITE_AS_FINAL_SUBMISSION:
    print("final Kaggle file:", WORK / "submission.csv")
