"""Cross-fit the SP45/FLE parent weight on the completed 6.39 model-core OOF."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
SOURCE_DIR = ROOT / "oof_exports" / "frontier_6p39_existing_models_oof_v0"
SOURCE = SOURCE_DIR / "frontier_6p39_core_oof.parquet"
OUT_DIR = ROOT / "oof_exports" / "frontier_6p39_crossfit_parent_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = np.round(np.linspace(0.0, 1.0, 41), 6)
PUBLISHED_WEIGHT_SP45 = 0.60

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

frame = pd.read_parquet(SOURCE)
required = {"id", "well", "row_idx", "fold", "truth", "sp45", "fle"}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"6.39 OOF lacks columns: {missing}")
frame = frame.sort_values(["well", "row_idx"]).reset_index(drop=True)
frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
if set(frame["fold"].unique()) != set(range(1, 6)):
    raise RuntimeError(f"Expected folds 1..5, found {sorted(frame['fold'].unique())}")
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate OOF ids")

truth = frame["truth"].to_numpy(np.float64)
sp45 = frame["sp45"].to_numpy(np.float64)
fle = frame["fle"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
fle_error = fle - truth
sp45_delta = sp45 - fle


def prediction(weight_sp45: float) -> np.ndarray:
    return float(weight_sp45) * sp45 + (1.0 - float(weight_sp45)) * fle


def sse_on(weight_sp45: float, mask: np.ndarray) -> float:
    base = fle_error[mask]
    delta = sp45_delta[mask]
    weight = float(weight_sp45)
    return float(
        np.dot(base, base)
        + 2.0 * weight * np.dot(base, delta)
        + weight * weight * np.dot(delta, delta)
    )


def metric(name: str, values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    error = values - truth
    bywell = (
        pd.DataFrame({"well": frame["well"].to_numpy(), "error2": error * error})
        .groupby("well")["error2"]
        .mean()
        .pow(0.5)
    )
    return {
        "endpoint": name,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(bywell.mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }


scan_rows = []
all_mask = np.ones(len(frame), dtype=bool)
for weight in WEIGHTS:
    exact_sse = sse_on(weight, all_mask)
    scan_rows.append(
        {
            "weight_sp45": float(weight),
            "weight_fle": float(1.0 - weight),
            "exact_sse": exact_sse,
            "rmse": float(np.sqrt(exact_sse / len(frame))),
        }
    )
scan = pd.DataFrame(scan_rows).sort_values(["exact_sse", "weight_sp45"])
scan.to_csv(OUT_DIR / "diagnostic_full_oof_weight_scan.csv", index=False)
diagnostic_weight = float(scan.iloc[0]["weight_sp45"])

crossfit = np.full(len(frame), np.nan, dtype=np.float64)
fold_rows = []
for heldout_fold in range(1, 6):
    train_mask = fold != heldout_fold
    valid_mask = fold == heldout_fold
    candidates = [
        (sse_on(weight, train_mask), float(weight))
        for weight in WEIGHTS
    ]
    _, selected_weight = min(candidates, key=lambda item: (item[0], item[1]))
    crossfit[valid_mask] = prediction(selected_weight)[valid_mask]
    base_error = fle[valid_mask] - truth[valid_mask]
    candidate_error = crossfit[valid_mask] - truth[valid_mask]
    fold_rows.append(
        {
            "heldout_fold": heldout_fold,
            "train_wells": int(frame.loc[train_mask, "well"].nunique()),
            "valid_wells": int(frame.loc[valid_mask, "well"].nunique()),
            "weight_sp45": selected_weight,
            "weight_fle": 1.0 - selected_weight,
            "fle_rmse": float(np.sqrt(np.mean(base_error * base_error))),
            "crossfit_rmse": float(np.sqrt(np.mean(candidate_error * candidate_error))),
            "gain": float(
                np.sqrt(np.mean(base_error * base_error))
                - np.sqrt(np.mean(candidate_error * candidate_error))
            ),
        }
    )
if not np.isfinite(crossfit).all():
    raise RuntimeError("Cross-fit endpoint has missing rows")

fold_report = pd.DataFrame(fold_rows)
fold_report.to_csv(OUT_DIR / "crossfit_weights.csv", index=False)
published = prediction(PUBLISHED_WEIGHT_SP45)
diagnostic = prediction(diagnostic_weight)

summary = pd.DataFrame(
    [
        metric("fle", fle),
        metric("sp45", sp45),
        metric("published_w0p60", published),
        metric("crossfit_weight", crossfit),
        metric("diagnostic_full_oof_optimum", diagnostic),
    ]
)
summary.to_csv(OUT_DIR / "endpoint_summary.csv", index=False)

output = frame[["id", "well", "row_idx", "fold", "truth", "sp45", "fle"]].copy()
output["published_6p39_core"] = published
output["crossfit_parent"] = crossfit
output.to_parquet(
    OUT_DIR / "frontier_6p39_crossfit_parent_oof.parquet",
    index=False,
    compression="zstd",
)

weights = fold_report["weight_sp45"].to_numpy(np.float64)
crossfit_metrics = summary.loc[summary["endpoint"].eq("crossfit_weight")].iloc[0].to_dict()
fle_metrics = summary.loc[summary["endpoint"].eq("fle")].iloc[0].to_dict()
verdict = {
    "experiment": "frontier_6p39_crossfit_parent_v1",
    "stage": "fivefold_crossfit_SP45_FLE_parent_weight",
    "source": str(SOURCE),
    "published_weight_sp45": PUBLISHED_WEIGHT_SP45,
    "diagnostic_full_oof_weight_sp45": diagnostic_weight,
    "crossfit_weights": fold_report.to_dict(orient="records"),
    "weight_stability": {
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
        "mean": float(weights.mean()),
        "all_positive": bool(np.all(weights > 0.0)),
        "range_le_0p15": bool(weights.max() - weights.min() <= 0.15),
    },
    "summary": summary.to_dict(orient="records"),
    "crossfit_gain_vs_fle": float(fle_metrics["rmse"] - crossfit_metrics["rmse"]),
    "strict_pass": bool(
        fle_metrics["rmse"] - crossfit_metrics["rmse"] >= 0.05
        and np.all(weights > 0.0)
        and weights.max() - weights.min() <= 0.15
    ),
    "next": (
        "PASS -> audit CP and neighbor on published, FLE, and crossfit parents"
        if fle_metrics["rmse"] - crossfit_metrics["rmse"] >= 0.05
        else "NO STABLE BLEND -> keep FLE as the local parent; still audit published 6.39 separately"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("CROSSFIT WEIGHTS")
print(fold_report.to_string(index=False), flush=True)
print("\nPARENT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
