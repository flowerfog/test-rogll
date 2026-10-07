"""Nested calibration of the synthetic GR residual-transfer generator.

Run immediately after ``synthetic_gr_domain_audit_v0_cell.py`` in the same
kernel.  Hyperparameters are selected with folds 3-5 as generator sources and
fold 2 as the real calibration domain.  The selected recipe is then frozen and
confirmed once on fold 1 using folds 2-5 as generator sources.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm


REQUIRED_GLOBALS = (
    "OUT_DIR",
    "fold",
    "train_segments",
    "valid_segments",
    "_segment_features",
    "POINTS",
    "RANDOM_STATE",
)
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run synthetic_gr_domain_audit_v0_cell.py first in this kernel; missing: "
        + repr(missing)
    )

CALIBRATION_DIR = Path(OUT_DIR).parent / "synthetic_gr_generator_calibration_v1"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FOLD = 2
SOURCE_FOLDS = (3, 4, 5)
RESIDUAL_SCALES = (0.60, 0.75, 0.90, 1.00)
TRANSFER_MODES = ("raw", "orthogonal")
CALIBRATION_TREES = 220
CONFIRMATION_TREES = 450


def _fill_residual(sample):
    residual = np.asarray(sample["residual"], np.float64).copy()
    valid = np.asarray(sample["horizontal_valid"], bool) & np.asarray(
        sample["typewell_valid"], bool
    )
    if int(valid.sum()) >= 2:
        positions = np.arange(POINTS)
        residual = np.interp(positions, positions[valid], residual[valid])
    else:
        residual[:] = 0.0
    return residual, valid


def _transfer_residual(target, donor, scale, mode):
    target_reference = np.asarray(target["aligned_typewell"], np.float64)
    donor_reference = np.asarray(donor["aligned_typewell"], np.float64)
    donor_residual, donor_valid = _fill_residual(donor)
    if mode == "raw":
        transferred = donor_residual
    elif mode == "orthogonal":
        valid = donor_valid & np.isfinite(donor_reference)
        if int(valid.sum()) >= 12 and float(np.var(donor_reference[valid])) > 1e-6:
            reference = donor_reference[valid]
            residual = donor_residual[valid]
            beta = float(
                np.dot(reference - reference.mean(), residual - residual.mean())
                / np.dot(reference - reference.mean(), reference - reference.mean())
            )
            beta = float(np.clip(beta, -1.5, 1.5))
            intercept = float(np.mean(residual) - beta * np.mean(reference))
            orthogonal = donor_residual - (intercept + beta * donor_reference)
            transferred = intercept + beta * target_reference + orthogonal
        else:
            transferred = donor_residual
    else:
        raise ValueError(mode)
    target_valid = np.asarray(target["typewell_valid"], bool)
    synthetic_valid = np.asarray(donor["horizontal_valid"], bool) & target_valid
    synthetic = target_reference + float(scale) * transferred
    synthetic = np.clip(synthetic, -8.0, 8.0)
    synthetic[~synthetic_valid] = 0.0
    output = dict(target)
    output["horizontal"] = synthetic.astype(np.float32)
    output["horizontal_valid"] = synthetic_valid
    return output


def _span_matched_target(source_segments, desired_span, rng):
    distances = np.asarray(
        [abs(float(segment["span_ft"]) - float(desired_span)) for segment in source_segments]
    )
    nearest = np.flatnonzero(distances <= np.quantile(distances, 0.10))
    if not len(nearest):
        nearest = np.asarray([int(np.argmin(distances))])
    return source_segments[int(rng.choice(nearest))]


def _synthetic_feature_frame(real_segments, source_segments, scale, mode, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for index, real_sample in enumerate(real_segments):
        target = _span_matched_target(source_segments, real_sample["span_ft"], rng)
        donor = source_segments[int(rng.integers(0, len(source_segments)))]
        synthetic = _transfer_residual(target, donor, scale, mode)
        synthetic["segment_id"] = index
        rows.append(_segment_features(synthetic, "synthetic", f"{mode}_{scale:.2f}"))
    return pd.DataFrame(rows)


def _real_feature_frame(real_segments):
    return pd.DataFrame(
        [
            _segment_features(sample, "real", "real")
            for sample in tqdm(real_segments, desc="nested real-domain features")
        ]
    )


def _domain_metrics(real, synthetic, trees, seed, return_importance=False):
    metadata = {"well", "segment_id", "domain", "method"}
    feature_columns = [column for column in real.columns if column not in metadata]
    pair = pd.concat([real, synthetic], ignore_index=True)
    X = pair[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = (pair["domain"] == "synthetic").astype(int).to_numpy()
    groups = pair["domain"].astype(str) + ":" + pair["well"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    prediction = np.full(len(pair), np.nan, np.float64)
    fold_auc = []
    importances = []
    for fold_index, (train_index, valid_index) in enumerate(splitter.split(X, y, groups)):
        classifier = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=8,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + fold_index,
        )
        classifier.fit(X[train_index], y[train_index])
        prediction[valid_index] = classifier.predict_proba(X[valid_index])[:, 1]
        fold_auc.append(float(roc_auc_score(y[valid_index], prediction[valid_index])))
        importances.append(classifier.feature_importances_)
    distances = []
    for column in feature_columns:
        left = real[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        right = synthetic[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        scale = float(np.sqrt(0.5 * (np.var(left) + np.var(right))) + 1e-6)
        distances.append(
            {
                "feature": column,
                "standardised_mean_difference": float(
                    abs(np.mean(left) - np.mean(right)) / scale
                ),
                "ks_statistic": float(ks_2samp(left, right).statistic),
            }
        )
    distances = pd.DataFrame(distances)
    metrics = {
        "domain_auc": float(roc_auc_score(y, prediction)),
        "mean_fold_auc": float(np.mean(fold_auc)),
        "max_fold_auc": float(np.max(fold_auc)),
        "median_standardised_mean_difference": float(
            distances["standardised_mean_difference"].median()
        ),
        "p90_standardised_mean_difference": float(
            distances["standardised_mean_difference"].quantile(0.90)
        ),
        "median_ks": float(distances["ks_statistic"].median()),
        "real_truth_ncc_median": float(real["truth_ncc"].median()),
        "synthetic_truth_ncc_median": float(synthetic["truth_ncc"].median()),
        "truth_ncc_median_gap": float(
            abs(real["truth_ncc"].median() - synthetic["truth_ncc"].median())
        ),
        "real_peak_margin_median": float(real["peak_margin"].median()),
        "synthetic_peak_margin_median": float(synthetic["peak_margin"].median()),
        "missing_fraction_gap": float(
            abs(real["missing_fraction"].mean() - synthetic["missing_fraction"].mean())
        ),
    }
    importance = None
    if return_importance:
        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": np.mean(importances, axis=0),
            }
        ).sort_values("importance", ascending=False)
    return metrics, distances, importance


def _selection_score(metrics):
    # Zero means all four original preregistered domain gates pass.
    return float(
        4.0 * max(metrics["domain_auc"] - 0.75, 0.0)
        + 4.0 * max(metrics["median_standardised_mean_difference"] - 0.25, 0.0)
        + 5.0 * max(metrics["truth_ncc_median_gap"] - 0.10, 0.0)
        + 5.0 * max(metrics["missing_fraction_gap"] - 0.05, 0.0)
        + 0.05 * metrics["domain_auc"]
        + 0.05 * metrics["truth_ncc_median_gap"]
    )


fold_map = dict(zip(fold["well"].astype(str), fold["fold"].astype(int)))
calibration_real_segments = [
    sample for sample in train_segments if fold_map[str(sample["well"])] == CALIBRATION_FOLD
]
calibration_source_segments = [
    sample for sample in train_segments if fold_map[str(sample["well"])] in SOURCE_FOLDS
]
if len(calibration_real_segments) < 500 or len(calibration_source_segments) < 1200:
    raise RuntimeError(
        "insufficient nested segments: "
        f"real={len(calibration_real_segments)}, source={len(calibration_source_segments)}"
    )

started = time.time()
print(
    {
        "experiment": "synthetic_gr_generator_calibration_v1",
        "calibration_fold": CALIBRATION_FOLD,
        "source_folds": SOURCE_FOLDS,
        "calibration_real_segments": len(calibration_real_segments),
        "calibration_source_segments": len(calibration_source_segments),
        "confirmation_real_segments": len(valid_segments),
        "configs": len(RESIDUAL_SCALES) * len(TRANSFER_MODES),
    },
    flush=True,
)

calibration_real = _real_feature_frame(calibration_real_segments)
config_rows = []
config_synthetic = {}
for mode in TRANSFER_MODES:
    for scale in RESIDUAL_SCALES:
        name = f"{mode}_{scale:.2f}"
        synthetic = _synthetic_feature_frame(
            calibration_real_segments,
            calibration_source_segments,
            scale,
            mode,
            RANDOM_STATE + int(round(scale * 100)) + (1000 if mode == "orthogonal" else 0),
        )
        metrics, _, _ = _domain_metrics(
            calibration_real,
            synthetic,
            CALIBRATION_TREES,
            RANDOM_STATE + int(round(scale * 100)),
        )
        row = {
            "config": name,
            "mode": mode,
            "residual_scale": scale,
            **metrics,
        }
        row["selection_score"] = _selection_score(row)
        config_rows.append(row)
        config_synthetic[name] = synthetic
        print("CALIBRATION", row, flush=True)

config_report = pd.DataFrame(config_rows).sort_values(
    ["selection_score", "domain_auc", "truth_ncc_median_gap"]
)
config_report.to_csv(CALIBRATION_DIR / "config_scan_fold2.csv", index=False)
selected = config_report.iloc[0]
selected_mode = str(selected["mode"])
selected_scale = float(selected["residual_scale"])

# Freeze the nested-selected recipe before touching fold 1 again.
confirmation_real = _real_feature_frame(valid_segments)
confirmation_synthetic = _synthetic_feature_frame(
    valid_segments,
    train_segments,
    selected_scale,
    selected_mode,
    RANDOM_STATE + 9001,
)
confirmation_metrics, confirmation_distances, confirmation_importance = _domain_metrics(
    confirmation_real,
    confirmation_synthetic,
    CONFIRMATION_TREES,
    RANDOM_STATE + 777,
    return_importance=True,
)
confirmation_distances.to_csv(CALIBRATION_DIR / "confirmation_feature_distances.csv", index=False)
confirmation_importance.to_csv(CALIBRATION_DIR / "confirmation_importance.csv", index=False)
confirmation_features = pd.concat(
    [confirmation_real, confirmation_synthetic], ignore_index=True
)
confirmation_features.to_csv(CALIBRATION_DIR / "confirmation_features.csv", index=False)

strict_pass = bool(
    confirmation_metrics["domain_auc"] <= 0.75
    and confirmation_metrics["median_standardised_mean_difference"] <= 0.25
    and confirmation_metrics["truth_ncc_median_gap"] <= 0.10
    and confirmation_metrics["missing_fraction_gap"] <= 0.05
)
verdict = {
    "experiment": "synthetic_gr_generator_calibration_v1",
    "nested_selection": {
        "source_folds": list(SOURCE_FOLDS),
        "calibration_fold": CALIBRATION_FOLD,
        "selected_mode": selected_mode,
        "selected_residual_scale": selected_scale,
        "selection_metrics": selected.to_dict(),
    },
    "confirmation_fold": 1,
    "confirmation_metrics": confirmation_metrics,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> materialise 50k fold-clean synthetic samples and pretrain the matcher"
        if strict_pass
        else "FAIL -> synthetic domain remains distinguishable; stop large-scale pretraining"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(CALIBRATION_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nCONFIG SCAN")
print(config_report.to_string(index=False))
print("\nCONFIRMATION TOP FEATURES")
print(confirmation_importance.head(20).to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", CALIBRATION_DIR)

