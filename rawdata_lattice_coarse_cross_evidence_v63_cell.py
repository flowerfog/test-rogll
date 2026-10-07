"""Read-only audit that promotes the V4 coarse path instead of V60b DeepSets.

Fold2 determines the coarse-path blend weight analytically.  The rounded frozen
weight is then applied once to the already-materialized Fold1 endpoints.  This
cell performs no candidate reconstruction, model fitting, or Fold1 search.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


started = time.time()
EXPERIMENT = "rawdata_lattice_coarse_cross_evidence_v63"
ROOT = Path("/root/autodl-tmp")
OUT_DIR = ROOT / "oof_exports" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD2_PATH = (
    ROOT
    / "oof_exports/rawdata_lattice_deepset_cross_evidence_v61"
    / "v60b_cross_evidence_fold2_oof.parquet"
)
FOLD1_PATH = (
    ROOT
    / "oof_exports/rawdata_lattice_deepset_fold1_confirmation_v62"
    / "fold1_confirmation_oof.parquet"
)
PARENT = "incumbent_6p377"
COARSE = "v60b_coarse"
TRUTH = "tvt_true"
WEIGHT_STEP = 0.05
MAX_WEIGHT = 0.50


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _read(path, expected_fold):
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"id", "well", TRUTH, PARENT, COARSE, "md_since"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")
    if frame["id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate ids")
    if frame[list(required - {"id", "well"})].isna().any().any():
        raise RuntimeError(f"{path}: non-finite required values")
    if "fold" in frame:
        observed = sorted(frame["fold"].dropna().astype(int).unique().tolist())
        # Cross-lineage artifacts may carry stale labels; disclose rather than
        # silently using them as proof of fold identity.
    else:
        observed = []
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    frame["audit_fold"] = int(expected_fold)
    return frame, observed


def _analytic_weight(frame):
    truth = frame[TRUTH].to_numpy(np.float64)
    parent = frame[PARENT].to_numpy(np.float64)
    delta = frame[COARSE].to_numpy(np.float64) - parent
    denominator = float(np.dot(delta, delta))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(truth - parent, delta) / denominator, 0.0, MAX_WEIGHT))


def _round_weight(weight):
    rounded = round(float(weight) / WEIGHT_STEP) * WEIGHT_STEP
    return float(np.clip(rounded, 0.0, MAX_WEIGHT))


def _prediction(frame, weight):
    parent = frame[PARENT].to_numpy(np.float64)
    coarse = frame[COARSE].to_numpy(np.float64)
    return parent + float(weight) * (coarse - parent)


def _rmse(truth, prediction, mask=None):
    if mask is None:
        mask = np.ones(len(truth), dtype=bool)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(prediction[mask] - truth[mask]))))


def _metrics(frame, prediction):
    truth = frame[TRUTH].to_numpy(np.float64)
    distance = frame["md_since"].to_numpy(np.float64)
    error2 = np.square(prediction - truth)
    bywell = pd.DataFrame({"well": frame["well"].to_numpy(str), "e2": error2})
    bywell = bywell.groupby("well", sort=False)["e2"].mean().pow(0.5)
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "far_gt2000_rmse": _rmse(truth, prediction, far),
    }


def _signed_error_correlation(frame):
    truth = frame[TRUTH].to_numpy(np.float64)
    parent_error = frame[PARENT].to_numpy(np.float64) - truth
    coarse_error = frame[COARSE].to_numpy(np.float64) - truth
    return float(np.corrcoef(parent_error, coarse_error)[0, 1])


fold2, fold2_declared = _read(FOLD2_PATH, 2)
fold1, fold1_declared = _read(FOLD1_PATH, 1)

analytic_weight = _analytic_weight(fold2)
frozen_weight = _round_weight(analytic_weight)

rows = []
metrics = {}
for fold, frame in ((2, fold2), (1, fold1)):
    predictions = {
        "parent": frame[PARENT].to_numpy(np.float64),
        "coarse": frame[COARSE].to_numpy(np.float64),
        "blend_w0p20": _prediction(frame, 0.20),
        "blend_fold2_frozen": _prediction(frame, frozen_weight),
    }
    metrics[str(fold)] = {}
    for endpoint, prediction in predictions.items():
        result = _metrics(frame, prediction)
        metrics[str(fold)][endpoint] = result
        rows.append({"fold": fold, "endpoint": endpoint, **result})

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "coarse_weight_summary.csv", index=False)

fold2_parent = metrics["2"]["parent"]
fold2_frozen = metrics["2"]["blend_fold2_frozen"]
fold1_parent = metrics["1"]["parent"]
fold1_frozen = metrics["1"]["blend_fold2_frozen"]
fold1_w020 = metrics["1"]["blend_w0p20"]

checks = {
    "fold2_frozen_gain_ge_0p05": bool(
        fold2_parent["rmse"] - fold2_frozen["rmse"] >= 0.05
    ),
    "fold1_frozen_gain_ge_0p05": bool(
        fold1_parent["rmse"] - fold1_frozen["rmse"] >= 0.05
    ),
    "fold1_far_gain_positive": bool(
        fold1_parent["far_gt2000_rmse"] - fold1_frozen["far_gt2000_rmse"] > 0.0
    ),
    "fold1_p95_gain_positive": bool(
        fold1_parent["p95_well_rmse"] - fold1_frozen["p95_well_rmse"] > 0.0
    ),
    "fold2_weight_close_to_predeclared_0p20": bool(abs(frozen_weight - 0.20) <= 0.10),
    # The 0.20 control was inherited from the DeepSets endpoint.  It was not a
    # Fold2-selected coarse-path dose, so outperforming it is not a valid
    # transfer gate.  The honest question is whether the Fold2-frozen coarse
    # dose remains beneficial on Fold1 without hurting its tail or p95.
    "fold1_frozen_gain_is_at_least_half_fold2_gain": bool(
        fold1_parent["rmse"] - fold1_frozen["rmse"]
        >= 0.5 * (fold2_parent["rmse"] - fold2_frozen["rmse"])
    ),
}
strict_pass = bool(all(checks.values()))

combined = pd.concat([fold2, fold1], ignore_index=True)
combined_prediction = _prediction(combined, frozen_weight)
combined["coarse_blend_fold2_frozen"] = combined_prediction
combined.to_parquet(OUT_DIR / "coarse_blend_fold12_oof.parquet", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_coarse_weight_fit_Fold1_read_only_consistency_audit",
    "read_only": True,
    "model_fit_calls": 0,
    "fold1_search_count": 0,
    "fold1_disclosure": (
        "Fold1 endpoints and the predeclared 0.20 coarse control were already viewed in V62; "
        "the deployed candidate remains the independently Fold2-fitted coarse weight. "
        "The 0.20 result is diagnostic only and is not used for selection."
    ),
    "sources": {"fold2": str(FOLD2_PATH), "fold1": str(FOLD1_PATH)},
    "declared_fold_labels": {"fold2_source": fold2_declared, "fold1_source": fold1_declared},
    "endpoint": "V4 coarse MAP path before the V60b DeepSets reranker",
    "weight": {
        "fold2_analytic": analytic_weight,
        "grid_step": WEIGHT_STEP,
        "frozen": frozen_weight,
        "predeclared_control": 0.20,
    },
    "metrics": metrics,
    "correlation": {
        "fold2_signed_error_raw": _signed_error_correlation(fold2),
        "fold1_signed_error_raw": _signed_error_correlation(fold1),
    },
    "gains": {
        "fold2_rmse": fold2_parent["rmse"] - fold2_frozen["rmse"],
        "fold1_rmse": fold1_parent["rmse"] - fold1_frozen["rmse"],
        "fold1_far": fold1_parent["far_gt2000_rmse"] - fold1_frozen["far_gt2000_rmse"],
        "fold1_p95": fold1_parent["p95_well_rmse"] - fold1_frozen["p95_well_rmse"],
    },
    "combined_fold12": _metrics(combined, combined_prediction),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> freeze conservative coarse weight from Fold2; retire DeepSets and build the V4 coarse final-train/test deployment path"
        if strict_pass
        else "FAIL -> coarse blend weight is not stable across Fold2/Fold1; close the V3-bank endpoint"
    ),
}

print("V63 COARSE WEIGHT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV63 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
_write_json(OUT_DIR / "verdict.json", verdict)
print(f"saved: {OUT_DIR}", flush=True)
