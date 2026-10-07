"""Fast five-fold audit of STRIDE against existing Track8-like endpoints.

Reuse the compact V9 STRIDE fold artifacts.  For every available candidate
endpoint, report standalone accuracy, signed-error correlation, an all-row
oracle scalar blend, and a legal five-fold cross-fit scalar blend whose dose
for each held-out fold is fit on the other four folds only.  No decoder is
rerun and no endpoint is promoted by an oracle result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_existing_endpoints_audit_v10"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V9_DIR = OOF_ROOT / "stride_sp45_fivefold_oof_v9"
FOLD_DIR = V9_DIR / "folds"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
if POOL_PATH is None:
    raise FileNotFoundError("Missing candidate_pool_full_merged_oof_honest_fle.csv")

HELDOUT_FOLDS = (1, 2, 3, 4, 5)
ENDPOINT_PREFERENCE = (
    "sp45",
    "fleongg",
    "strong",
    "strong_raw",
    "strong_bias",
    "cp12",
    "ridge",
    "selector",
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _metrics(truth, prediction, wells):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    well_rmse = np.sqrt(pd.Series(error * error).groupby(np.asarray(wells)).mean())
    return {
        "rows": int(len(error)),
        "wells": int(pd.Series(wells).nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
    }


parts = []
for fold in HELDOUT_FOLDS:
    path = FOLD_DIR / f"fold{fold}_stride_raw.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        wells = data["wells"].astype(str)
        well_code = data["well_code"].astype(np.int64)
        row_idx = data["row_idx"].astype(np.int64)
        stride = data["stride_raw"].astype(np.float32)
    well = wells[well_code]
    parts.append(
        pd.DataFrame(
            {
                "id": [f"{name}_{int(row)}" for name, row in zip(well, row_idx)],
                "well": well,
                "fold": fold,
                "stride_raw": stride,
            }
        )
    )
frame = pd.concat(parts, ignore_index=True)
if frame["well"].nunique() != 773:
    raise RuntimeError(f"V9 artifacts contain {frame['well'].nunique()} wells")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
endpoints = [name for name in ENDPOINT_PREFERENCE if name in header]
if not {"id", "well", "tvt_true"}.issubset(header) or not endpoints:
    raise RuntimeError(f"Cannot identify endpoint pool columns: {header}")
pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "well", "tvt_true", *endpoints],
    dtype={"id": str, "well": str},
)
frame = frame.merge(pool, on=["id", "well"], how="inner", validate="one_to_one")
if frame["well"].nunique() != 773:
    raise RuntimeError(f"Pool alignment contains {frame['well'].nunique()} wells")

truth = frame["tvt_true"].to_numpy(np.float64)
stride = frame["stride_raw"].to_numpy(np.float64)
stride_error = stride - truth
fold_array = frame["fold"].to_numpy(np.int8)
wells = frame["well"].to_numpy()

summary_rows = []
fold_rows = []
for endpoint in endpoints:
    base = frame[endpoint].to_numpy(np.float64)
    if not np.isfinite(base).all():
        print(f"skip non-finite endpoint: {endpoint}", flush=True)
        continue
    base_error = base - truth
    direction = stride - base
    correlation = float(np.corrcoef(base_error, stride_error)[0, 1])

    denominator = float(np.dot(direction, direction))
    oracle_weight = float(np.clip(-np.dot(base_error, direction) / denominator, 0.0, 1.0))
    oracle_prediction = base + oracle_weight * direction
    oracle_metrics = _metrics(truth, oracle_prediction, wells)

    crossfit_prediction = np.empty(len(frame), np.float64)
    local_fold_rows = []
    for heldout_fold in HELDOUT_FOLDS:
        train_mask = fold_array != heldout_fold
        valid_mask = ~train_mask
        local_direction = direction[train_mask]
        local_denominator = float(np.dot(local_direction, local_direction))
        weight = float(
            np.clip(-np.dot(base_error[train_mask], local_direction) / local_denominator, 0.0, 1.0)
        )
        crossfit_prediction[valid_mask] = base[valid_mask] + weight * direction[valid_mask]
        base_rmse = _metrics(truth[valid_mask], base[valid_mask], wells[valid_mask])["rmse"]
        blend_rmse = _metrics(
            truth[valid_mask], crossfit_prediction[valid_mask], wells[valid_mask]
        )["rmse"]
        item = {
            "endpoint": endpoint,
            "heldout_fold": heldout_fold,
            "weight_stride": weight,
            "base_rmse": base_rmse,
            "blend_rmse": blend_rmse,
            "gain": base_rmse - blend_rmse,
        }
        fold_rows.append(item)
        local_fold_rows.append(item)

    base_metrics = _metrics(truth, base, wells)
    crossfit_metrics = _metrics(truth, crossfit_prediction, wells)
    positive_folds = int(sum(item["gain"] > 0.0 for item in local_fold_rows))
    summary_rows.append(
        {
            "endpoint": endpoint,
            "standalone_rmse": base_metrics["rmse"],
            "standalone_mean_well_rmse": base_metrics["mean_well_rmse"],
            "error_correlation_stride": correlation,
            "oracle_weight_stride": oracle_weight,
            "oracle_blend_rmse": oracle_metrics["rmse"],
            "crossfit_blend_rmse": crossfit_metrics["rmse"],
            "crossfit_gain": base_metrics["rmse"] - crossfit_metrics["rmse"],
            "positive_gain_folds": positive_folds,
            "minimum_fold_gain": float(min(item["gain"] for item in local_fold_rows)),
            "maximum_fold_gain": float(max(item["gain"] for item in local_fold_rows)),
        }
    )
    print(
        "V10",
        endpoint,
        base_metrics["rmse"],
        crossfit_metrics["rmse"],
        correlation,
        positive_folds,
        flush=True,
    )

summary = pd.DataFrame(summary_rows).sort_values(
    ["crossfit_blend_rmse", "crossfit_gain"], ascending=[True, False]
)
fold_report = pd.DataFrame(fold_rows)
summary.to_csv(OUT_DIR / "endpoint_summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "endpoint_fold_report.csv", index=False)

stable = summary.loc[
    (summary["crossfit_gain"] > 0.0)
    & (summary["positive_gain_folds"] >= 4)
]
selected = stable.iloc[0].to_dict() if len(stable) else None
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_existing_endpoint_role_audit",
    "source_stride": str(FOLD_DIR),
    "source_pool": str(POOL_PATH),
    "endpoints": endpoints,
    "summary": summary.to_dict(orient="records"),
    "selected_stable_endpoint": selected,
    "interpretation": {
        "fleongg": "closest existing GBDT analogue to Track8, not proven identical",
        "sp45": "raw SP45 path endpoint",
        "strong": "already-combined SP45/Fleongg family endpoint, not Track8 itself",
    },
    "next": (
        "A stable Fleongg-like endpoint exists -> freeze it for deployment-dose confirmation"
        if selected is not None and selected["endpoint"] in {"fleongg", "strong", "strong_raw", "strong_bias"}
        else "No existing endpoint supplies a stable Track8 role -> reconstruct SP45-feature GBDT explicitly"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV10 ENDPOINT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
