"""V159: exact-ID nested blend audit for V158 against canonical V144 endpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
RAW_PATH = OOF_ROOT / "raw_full_feature_direct_v158" / "raw_direct_oof.parquet"
BASE_PATH = OOF_ROOT / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet"
FEATURE_CANDIDATES = (
    ROOT / "wellbore-geology-prediction-artifacts" / "data" / "train.csv",
    ROOT / "data" / "train.csv",
)
FEATURE_PATH = next((path for path in FEATURE_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / "raw_direct_v76_nested_blend_v159"
OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (RAW_PATH, BASE_PATH):
    if not path.exists(): raise FileNotFoundError(path)
if FEATURE_PATH is None: raise FileNotFoundError(FEATURE_CANDIDATES)


def _metrics(truth, pred, wells, distance):
    error = pred - truth
    wr = pd.Series(error * error).groupby(wells).mean().pow(0.5)
    far = distance > 2000.0
    return {"rows": len(error), "wells": pd.Series(wells).nunique(), "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(wr.median()), "p95_well_rmse": float(wr.quantile(0.95)), "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2)))}


def _alpha(base_error, candidate_error, mask):
    direction = candidate_error[mask] - base_error[mask]
    denom = float(np.dot(direction, direction))
    return float(np.clip(-np.dot(base_error[mask], direction) / max(denom, 1e-12), 0.0, 1.0))


def _detrend(error, distance, wells):
    out = np.empty(len(error), float)
    codes, unique = pd.factorize(wells, sort=False)
    for code in range(len(unique)):
        idx = np.flatnonzero(codes == code)
        if len(idx) < 3 or np.ptp(distance[idx]) < 1e-6:
            out[idx] = error[idx] - np.mean(error[idx])
        else:
            slope, intercept = np.polyfit(distance[idx], error[idx], 1)
            out[idx] = error[idx] - (slope * distance[idx] + intercept)
    return out


base_schema = pq.ParquetFile(BASE_PATH).schema.names
base_columns = [x for x in ("incumbent_6p377_replay", "exact_v76", "consensus_center", "nested_sequence_coefficient") if x in base_schema]
if not base_columns: raise RuntimeError("No canonical V144 endpoint columns")
base = pd.read_parquet(BASE_PATH, columns=["id", "well", "row_idx", "fold", "tvt_true", "md_since", *base_columns])
raw = pd.read_parquet(RAW_PATH, columns=["id", "well", "fold", "target", "prediction"])
anchor = pd.read_csv(FEATURE_PATH, usecols=["id", "last_known_tvt"], dtype={"id": str})
anchor = anchor.drop_duplicates("id")
base["id"], raw["id"] = base["id"].astype(str), raw["id"].astype(str)
raw = raw.merge(anchor, on="id", how="left", validate="one_to_one")
if raw["last_known_tvt"].isna().any(): raise RuntimeError("Missing V158 anchor values")
raw["prediction_absolute"] = raw["last_known_tvt"].to_numpy(float) + raw["prediction"].to_numpy(float)
frame = base.merge(raw[["id", "prediction_absolute"]], on="id", how="inner", validate="one_to_one")
if len(frame) != 3_783_989 or frame["well"].nunique() != 773: raise RuntimeError(f"Alignment failed {len(frame)}/{frame['well'].nunique()}")
truth = frame["tvt_true"].to_numpy(float)
candidate = frame["prediction_absolute"].to_numpy(float)
candidate_error = candidate - truth
distance = frame["md_since"].to_numpy(float)
wells = frame["well"].to_numpy(str)
folds = frame["fold"].to_numpy(int)

summary_rows, fold_rows = [], []
candidate_metrics = _metrics(truth, candidate, wells, distance)
summary_rows.append({"base": "standalone", "endpoint": "raw_direct", **candidate_metrics})
for column in base_columns:
    base_pred = frame[column].to_numpy(float)
    base_error = base_pred - truth
    base_detrended = _detrend(base_error, distance, wells)
    candidate_detrended = _detrend(candidate_error, distance, wells)
    nested = np.empty(len(frame), float)
    local = []
    for heldout in (1, 2, 3, 4, 5):
        valid = folds == heldout
        train = ~valid
        alpha = _alpha(base_error, candidate_error, train)
        nested[valid] = base_pred[valid] + alpha * (candidate[valid] - base_pred[valid])
        bm = _metrics(truth[valid], base_pred[valid], wells[valid], distance[valid])
        nm = _metrics(truth[valid], nested[valid], wells[valid], distance[valid])
        row = {"base": column, "fold": heldout, "alpha_raw_direct": alpha, "base_rmse": bm["rmse"], "blend_rmse": nm["rmse"], "gain": bm["rmse"] - nm["rmse"], "far_gain": bm["far_gt2000_rmse"] - nm["far_gt2000_rmse"]}
        fold_rows.append(row); local.append(row)
    bm = _metrics(truth, base_pred, wells, distance)
    nm = _metrics(truth, nested, wells, distance)
    summary_rows.append({"base": column, "endpoint": "base", **bm})
    summary_rows.append({"base": column, "endpoint": "nested_blend", **nm, "gain": bm["rmse"] - nm["rmse"], "far_gain": bm["far_gt2000_rmse"] - nm["far_gt2000_rmse"], "raw_error_corr": float(np.corrcoef(base_error, candidate_error)[0, 1]), "detrended_error_corr": float(np.corrcoef(base_detrended, candidate_detrended)[0, 1]), "alpha_mean": float(np.mean([x["alpha_raw_direct"] for x in local])), "alpha_min": float(np.min([x["alpha_raw_direct"] for x in local])), "positive_folds": int(sum(x["gain"] > 0 for x in local))})

summary = pd.DataFrame(summary_rows)
fold_report = pd.DataFrame(fold_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
blend = summary.loc[summary["endpoint"].eq("nested_blend")].sort_values("rmse").iloc[0]
verdict = {"experiment": "raw_direct_v76_nested_blend_v159", "stage": "exact_ID_leave_one_fold_out_blend", "candidate": str(RAW_PATH), "base": str(BASE_PATH), "best_base": str(blend["base"]), "best": blend.to_dict(), "folds": fold_report.loc[fold_report["base"].eq(blend["base"])].to_dict(orient="records"), "gates": {"detrended_corr_le_0p75": bool(abs(float(blend["detrended_error_corr"])) <= 0.75), "gain_ge_0p05": bool(float(blend["gain"]) >= 0.05), "positive_folds_4of5": bool(int(blend["positive_folds"]) >= 4), "alpha_min_ge_0p05": bool(float(blend["alpha_min"]) >= 0.05)}, "strict_pass": bool(abs(float(blend["detrended_error_corr"])) <= 0.75 and float(blend["gain"]) >= 0.05 and int(blend["positive_folds"]) >= 4 and float(blend["alpha_min"]) >= 0.05), "next": "PASS -> freeze V158 model family and build inference bundle" if abs(float(blend["detrended_error_corr"])) <= 0.75 and float(blend["gain"]) >= 0.05 and int(blend["positive_folds"]) >= 4 and float(blend["alpha_min"]) >= 0.05 else "FAIL -> V158 does not add stable independent value", "elapsed_minutes": (time.time() - STARTED) / 60.0}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding="utf-8")
print("V159 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("V159 FOLDS")
print(fold_report.to_string(index=False), flush=True)
print("V159 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x), flush=True)
