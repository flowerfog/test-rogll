"""V154: cheap diversity and analytic-blend pretest for existing OOF artifacts."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "independent_engine_diversity_pretest_v154"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CANDIDATES = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
    OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18" / "full_oof_cp_endpoints.parquet",
)
BASE_PATH = next((path for path in BASE_CANDIDATES if path.is_file()), None)
if BASE_PATH is None:
    raise FileNotFoundError(BASE_CANDIDATES)

BASE_COLUMNS = ["id", "well", "row_idx", "fold", "tvt_true", "md_since"]
base_schema = pq.ParquetFile(BASE_PATH).schema.names
BASE_COLUMN = next(
    (column for column in ("incumbent_6p377_replay", "current_dose_0p20", "full_cp8_ordered") if column in base_schema),
    None,
)
if BASE_COLUMN is None:
    raise RuntimeError(f"No deployed baseline column in {BASE_PATH}")
base = pd.read_parquet(BASE_PATH, columns=BASE_COLUMNS + [BASE_COLUMN])
base["id"] = base["id"].astype(str)
base["well"] = base["well"].astype(str)
base = base.sort_values("id", kind="stable").reset_index(drop=True)
if len(base) != 3_783_989 or base["id"].duplicated().any():
    raise RuntimeError(f"Invalid canonical baseline rows/ids: {len(base)}")
base_pred = base[BASE_COLUMN].to_numpy(float)
truth = base["tvt_true"].to_numpy(float)
distance = base["md_since"].to_numpy(float)
base_error = base_pred - truth
codes, wells = pd.factorize(base["well"], sort=True)


def _detrend(values):
    values = np.asarray(values, float)
    x = distance
    n = np.bincount(codes, minlength=len(wells)).astype(float)
    sx = np.bincount(codes, weights=x, minlength=len(wells))
    sy = np.bincount(codes, weights=values, minlength=len(wells))
    sxx = np.bincount(codes, weights=x * x, minlength=len(wells))
    sxy = np.bincount(codes, weights=x * values, minlength=len(wells))
    denom = np.maximum(n * sxx - sx * sx, 1.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / np.maximum(n, 1.0)
    return values - (intercept[codes] + slope[codes] * x)


def _alpha_fit(left, right, mask):
    e1, e2 = left[mask], right[mask]
    delta = e2 - e1
    denom = float(np.dot(delta, delta))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(e1, delta) / denom, 0.0, 1.0))


def _rmse(error, mask=None):
    value = np.asarray(error, float) if mask is None else np.asarray(error, float)[mask]
    return float(np.sqrt(np.mean(value * value))) if len(value) else np.nan


def _evaluate(name, candidate):
    candidate = np.asarray(candidate, float)
    if len(candidate) != len(base) or not np.isfinite(candidate).all():
        return None
    candidate_error = candidate - truth
    base_detrended = _detrend(base_error)
    candidate_detrended = _detrend(candidate_error)
    alpha = _alpha_fit(base_error, candidate_error, np.ones(len(base), bool))
    blended_error = base_error + alpha * (candidate_error - base_error)
    far = distance > 2000.0
    fold_rows = []
    for fold in sorted(base["fold"].unique()):
        valid = base["fold"].to_numpy(int) == int(fold)
        train = ~valid
        fold_alpha = _alpha_fit(base_error, candidate_error, train)
        fold_blend = base_error + fold_alpha * (candidate_error - base_error)
        fold_rows.append({
            "fold": int(fold),
            "alpha": fold_alpha,
            "base_rmse": _rmse(base_error, valid),
            "base_far_rmse": _rmse(base_error, valid & far),
            "candidate_rmse": _rmse(candidate_error, valid),
            "blend_rmse": _rmse(fold_blend, valid),
            "far_blend_rmse": _rmse(fold_blend, valid & far),
            "gain": _rmse(base_error, valid) - _rmse(fold_blend, valid),
        })
    fold_frame = pd.DataFrame(fold_rows)
    return {
        "endpoint": name,
        "rows": len(base),
        "wells": len(wells),
        "standalone_rmse": _rmse(candidate_error),
        "standalone_far_rmse": _rmse(candidate_error, far),
        "raw_error_corr": float(np.corrcoef(base_error, candidate_error)[0, 1]),
        "detrended_error_corr": float(np.corrcoef(base_detrended, candidate_detrended)[0, 1]),
        "oracle_global_alpha": alpha,
        "oracle_global_blend_rmse": _rmse(blended_error),
        "oracle_global_blend_far_rmse": _rmse(blended_error, far),
        "fold_mean_alpha": float(fold_frame["alpha"].mean()),
        "fold_alpha_min": float(fold_frame["alpha"].min()),
        "fold_alpha_max": float(fold_frame["alpha"].max()),
        "fold_positive_fraction": float(np.mean(fold_frame["alpha"] > 0.05)),
        "fold_mean_gain": float(fold_frame["gain"].mean()),
        "fold_min_gain": float(fold_frame["gain"].min()),
        "fold_far_mean_gain": float(
            np.mean([row["base_far_rmse"] - row["far_blend_rmse"] for row in fold_rows])
        ),
        "fold_rows": fold_rows,
    }


def _candidate_columns(path):
    schema = pq.ParquetFile(path).schema.names
    excluded = set(BASE_COLUMNS + [BASE_COLUMN, "truth", "prediction", "baseline", "endpoint", "fold"])
    hints = ("pred", "endpoint", "coarse", "v76", "cp", "neighbor", "blend", "consensus", "dose", "trajectory", "parent", "refined", "correct", "controller", "strong", "psg", "fleongg", "sp45")
    columns = [
        column for column in schema
        if column not in excluded and column not in {"well", "row_idx"}
        and any(token in column.lower() for token in hints)
    ]
    return columns[:32]


files = []
for path in OOF_ROOT.rglob("*.parquet"):
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows == len(base):
            files.append(path)
    except Exception:
        continue
files = sorted(set(files))
reports = []
loaded = []
for path in files:
    columns = _candidate_columns(path)
    if not columns:
        continue
    try:
        schema = pq.ParquetFile(path).schema.names
        source_bases = [name for name in ("base", "parent", "incumbent") if name in schema]
        if "id" in schema:
            table_columns = ["id", *source_bases, *columns]
            candidate_frame = pd.read_parquet(path, columns=table_columns)
            candidate_frame["id"] = candidate_frame["id"].astype(str)
            candidate_frame = candidate_frame.drop_duplicates("id").set_index("id").reindex(base["id"]).reset_index()
        elif {"well", "row_idx"}.issubset(schema):
            table_columns = ["well", "row_idx", *source_bases, *columns]
            candidate_frame = pd.read_parquet(path, columns=table_columns)
            candidate_frame["well"] = candidate_frame["well"].astype(str)
            candidate_frame["row_idx"] = candidate_frame["row_idx"].astype(np.int64)
            candidate_frame = candidate_frame.drop_duplicates(["well", "row_idx"])
            candidate_frame = base[["well", "row_idx"]].merge(
                candidate_frame, on=["well", "row_idx"], how="left", sort=False, validate="one_to_one"
            )
        else:
            continue
        if len(candidate_frame) != len(base):
            continue
        for column in columns:
            if not pd.api.types.is_numeric_dtype(candidate_frame[column]):
                continue
            values = candidate_frame[column].to_numpy(float)
            lower = column.lower()
            if any(token in lower for token in ("delta", "move", "offset")):
                base_candidates = [name for name in ("base", "parent", "incumbent") if name in candidate_frame.columns]
                if not base_candidates:
                    continue
                values = candidate_frame[base_candidates[0]].to_numpy(float) + values
            report = _evaluate(f"{path.parent.name}/{column}", values)
            if report is not None:
                report["source"] = str(path)
                reports.append(report)
                loaded.append((path, column))
    except Exception as exc:
        print(f"V154 skip {path}: {exc}", flush=True)

if not reports:
    raise RuntimeError("No exact-row candidate endpoints discovered")
report_frame = pd.DataFrame(reports).sort_values("fold_mean_gain", ascending=False).reset_index(drop=True)
report_frame.drop(columns=["fold_rows"]).to_csv(OUT_DIR / "endpoint_diversity.csv", index=False)
fold_rows = []
for report in reports:
    for row in report["fold_rows"]:
        fold_rows.append({"endpoint": report["endpoint"], **row})
pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_blends.csv", index=False)

best = report_frame.iloc[0]
verdict = {
    "experiment": "independent_engine_diversity_pretest_v154",
    "base": str(BASE_PATH),
    "base_column": BASE_COLUMN,
    "candidate_files_scanned": len(files),
    "candidate_endpoints_evaluated": len(report_frame),
    "best_endpoint": str(best["endpoint"]),
    "best": best.drop(labels=["fold_rows"]).to_dict(),
    "gates": {
        "candidate_standalone_rmse_le_10p5": bool(float(best["standalone_rmse"]) <= 10.5),
        "detrended_corr_le_0p60": bool(abs(float(best["detrended_error_corr"])) <= 0.60),
        "fold_positive_fraction_ge_0p80": bool(float(best["fold_positive_fraction"]) >= 0.80),
        "fold_mean_gain_ge_0p10": bool(float(best["fold_mean_gain"]) >= 0.10),
        "fold_min_gain_nonnegative": bool(float(best["fold_min_gain"]) >= 0.0),
    },
    "next": "PASS -> build independent STRIDE-like engine" if float(best["fold_mean_gain"]) >= 0.10 and abs(float(best["detrended_error_corr"])) <= 0.60 else "FAIL -> existing artifacts lack a usable independent blend channel",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
print("V154 TOP ENDPOINTS")
print(report_frame.drop(columns=["fold_rows"]).head(20).to_string(index=False))
print("V154 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=str))
