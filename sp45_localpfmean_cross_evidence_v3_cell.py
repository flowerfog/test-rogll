"""Zero-training cross-evidence audit for the V2 local-reference PF mean.

V2 showed that local_050 improves the unweighted PF population mean while
hurting likelihood-ranked seeds and the selector.  This audit treats that mean
as a deliberately weak, potentially decorrelated endpoint and measures nested
scalar blends against historical fold-clean endpoints on the same Fold-2 rows.
This is an investment audit, not an untouched confirmation: Fold 2 identified
the useful sub-endpoint, so any passing pair must be regenerated on Fold 3.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "sp45_localpfmean_cross_evidence_v3"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR = OOF_ROOT / "sp45_localref_fold2_gateway_v2"
RECORD_DIR = SOURCE_DIR / "records"
V2_VERDICT = SOURCE_DIR / "verdict.json"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
if not V2_VERDICT.exists():
    raise FileNotFoundError(V2_VERDICT)
if POOL_PATH is None:
    raise FileNotFoundError("Missing candidate_pool_full_merged_oof_honest_fle.csv")

BASE_PREFERENCE = ("strong_bias", "strong", "sp45", "cp12", "fleongg", "ridge", "selector")
INNER_FOLDS = 5


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


def _inner_fold(well):
    digest = hashlib.sha256(str(well).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % INNER_FOLDS


parts = []
for number, path in enumerate(sorted(RECORD_DIR.glob("*.npz")), start=1):
    with np.load(path, allow_pickle=False) as data:
        well = str(data["well"][0])
        rows = data["row_idx"].astype(np.int32)
        required = ("raw_pf_mean", "pf_mean", "truth", "md", "last_md")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RuntimeError(f"{path.name} lacks paired V2 arrays: {missing}")
        parts.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in rows],
                    "well": well,
                    "row_idx": rows,
                    "truth": data["truth"].astype(np.float32),
                    "distance_ft": (data["md"] - float(data["last_md"][0])).astype(np.float32),
                    "raw_pf_mean": data["raw_pf_mean"].astype(np.float32),
                    "local_pf_mean": data["pf_mean"].astype(np.float32),
                }
            )
        )
    if number % 50 == 0:
        print(f"load paired PF records {number}", flush=True)
if len(parts) != 155:
    raise RuntimeError(f"Expected 155 Fold2 records, found {len(parts)}")
frame = pd.concat(parts, ignore_index=True)
del parts

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
base_columns = [column for column in BASE_PREFERENCE if column in header]
if not base_columns:
    raise RuntimeError(f"No historical endpoint columns found in {header}")
pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "well", "tvt_true"] + base_columns,
    dtype={"id": str, "well": str},
)
frame = frame.merge(pool, on=["id", "well"], how="inner", validate="one_to_one")
if frame["well"].nunique() != 155:
    raise RuntimeError(f"Pool alignment lost wells: {frame['well'].nunique()}")
truth_gap = np.abs(frame["truth"].to_numpy(np.float64) - frame["tvt_true"].to_numpy(np.float64))
if not np.isfinite(truth_gap).all() or float(np.max(truth_gap)) > 0.01:
    raise RuntimeError(f"Truth mismatch: {float(np.max(truth_gap))}")
frame.drop(columns="tvt_true", inplace=True)
frame["inner_fold"] = frame["well"].map(_inner_fold).astype(np.int8)


def _metrics(prediction):
    prediction = np.asarray(prediction, np.float64)
    truth = frame["truth"].to_numpy(np.float64)
    error = prediction - truth
    bywell = np.sqrt(
        pd.DataFrame({"well": frame["well"], "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
    )
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(bywell.mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }
    distance = frame["distance_ft"].to_numpy(np.float64)
    for name, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (distance >= low) & (distance < high)
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


def _correlation(base, candidate):
    truth = frame["truth"].to_numpy(np.float64)
    base_error = np.asarray(base, np.float64) - truth
    candidate_error = np.asarray(candidate, np.float64) - truth
    raw = float(np.corrcoef(base_error, candidate_error)[0, 1])
    bins = pd.cut(
        frame["distance_ft"],
        bins=[-np.inf, 500.0, 1000.0, 2000.0, np.inf],
        labels=False,
    ).to_numpy()
    base_detrended = base_error.copy()
    candidate_detrended = candidate_error.copy()
    within = {}
    for index, name in enumerate(("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")):
        mask = bins == index
        within[name] = float(np.corrcoef(base_error[mask], candidate_error[mask])[0, 1])
        base_detrended[mask] -= np.mean(base_detrended[mask])
        candidate_detrended[mask] -= np.mean(candidate_detrended[mask])
    return {
        "raw": raw,
        "distance_strata_demeaned": float(np.corrcoef(base_detrended, candidate_detrended)[0, 1]),
        "within_distance_strata": within,
    }


def _crossfit_pair(base_column):
    truth = frame["truth"].to_numpy(np.float64)
    base = frame[base_column].to_numpy(np.float64)
    candidate = frame["local_pf_mean"].to_numpy(np.float64)
    direction = candidate - base
    base_error = base - truth
    prediction = np.empty(len(frame), np.float64)
    folds = frame["inner_fold"].to_numpy(np.int8)
    weights = []
    for heldout in range(INNER_FOLDS):
        train = folds != heldout
        valid = ~train
        denominator = float(np.dot(direction[train], direction[train]))
        weight = float(np.clip(-np.dot(base_error[train], direction[train]) / denominator, 0.0, 1.0))
        prediction[valid] = base[valid] + weight * direction[valid]
        valid_base = float(np.sqrt(np.mean(base_error[valid] ** 2)))
        valid_error = prediction[valid] - truth[valid]
        valid_blend = float(np.sqrt(np.mean(valid_error ** 2)))
        weights.append(
            {
                "heldout_inner_fold": heldout,
                "train_wells": int(frame.loc[train, "well"].nunique()),
                "valid_wells": int(frame.loc[valid, "well"].nunique()),
                "weight_local_pf_mean": weight,
                "base_rmse": valid_base,
                "blend_rmse": valid_blend,
                "gain": valid_base - valid_blend,
            }
        )
    return prediction, weights


endpoint_metrics = {
    "raw_pf_mean": _metrics(frame["raw_pf_mean"]),
    "local_pf_mean": _metrics(frame["local_pf_mean"]),
}
pair_rows = []
pair_payload = {}
for base_column in base_columns:
    if not np.isfinite(frame[base_column].to_numpy(np.float64)).all():
        continue
    base = frame[base_column].to_numpy(np.float64)
    candidate = frame["local_pf_mean"].to_numpy(np.float64)
    crossfit, weights = _crossfit_pair(base_column)
    base_metrics = _metrics(base)
    blend_metrics = _metrics(crossfit)
    correlation = _correlation(base, candidate)
    positive_folds = int(sum(row["gain"] > 0.0 for row in weights))
    minimum_weight = float(min(row["weight_local_pf_mean"] for row in weights))
    pair_rows.append(
        {
            "base": base_column,
            "base_rmse": base_metrics["rmse"],
            "local_pf_mean_rmse": endpoint_metrics["local_pf_mean"]["rmse"],
            "crossfit_blend_rmse": blend_metrics["rmse"],
            "crossfit_gain": base_metrics["rmse"] - blend_metrics["rmse"],
            "positive_inner_folds": positive_folds,
            "minimum_weight": minimum_weight,
            "raw_error_correlation": correlation["raw"],
            "detrended_error_correlation": correlation["distance_strata_demeaned"],
            "far_gain": base_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"],
        }
    )
    pair_payload[base_column] = {
        "base": base_metrics,
        "crossfit_blend": blend_metrics,
        "weights": weights,
        "correlation": correlation,
    }

summary = pd.DataFrame(pair_rows).sort_values(
    ["crossfit_blend_rmse", "crossfit_gain"], ascending=[True, False]
)
summary.to_csv(OUT_DIR / "pair_summary.csv", index=False)
winner = summary.iloc[0].to_dict()
best_standalone_rmse = float(summary["base_rmse"].min())
gain_vs_best_standalone = float(best_standalone_rmse - winner["crossfit_blend_rmse"])
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_discovery_cross_evidence_investment_audit",
    "selection_status": "discovery_only; local_pf_mean was identified using Fold2 truth",
    "source": str(SOURCE_DIR),
    "candidate_pool": str(POOL_PATH),
    "rows": int(len(frame)),
    "wells": int(frame["well"].nunique()),
    "endpoint_metrics": endpoint_metrics,
    "pairs": pair_payload,
    "winner": winner,
    "best_standalone_rmse": best_standalone_rmse,
    "winner_gain_vs_best_standalone": gain_vs_best_standalone,
    "gates": {
        "winner_gain_vs_best_standalone_ge_0p10": bool(gain_vs_best_standalone >= 0.10),
        "winner_positive_inner_folds_ge_4": bool(winner["positive_inner_folds"] >= 4),
        "winner_minimum_weight_ge_0p05": bool(winner["minimum_weight"] >= 0.05),
        "winner_detrended_correlation_le_0p75": bool(winner["detrended_error_correlation"] <= 0.75),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> regenerate raw/local PF means on Fold3 and confirm the frozen endpoint/base pair"
    if verdict["strict_pass"]
    else "FAIL -> the improved PF population mean has no stable cross-evidence value"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nLOCAL PF-MEAN ENDPOINTS")
print(pd.DataFrame(endpoint_metrics).T.to_string(), flush=True)
print("\nCROSS-EVIDENCE PAIRS")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
