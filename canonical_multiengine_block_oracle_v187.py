"""V187: canonical full-row block oracle over genuinely different engines.

This is a zero-training investment gate.  It asks whether the complete OOF
endpoint inventory contains enough local (within-well) diversity to reach a
pooled RMSE below 6 when endpoint choice may change only at coarse MD blocks.
Truth is used only by the oracle and metrics; no output is deployable.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / "canonical_multiengine_block_oracle_v187"
OUT.mkdir(parents=True, exist_ok=True)

BASE_PATH = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
BASE_COLUMN = "metadata_analog_w0p05"
EXPECTED_BASE_SHA = "43f23cb1f53212e8da77b38e873b24ae53ddd092a8f2fa16fa761e7a1536e8c6"
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
FREEZE_FT = 1000.0
BLOCK_FT = (128.0, 256.0, 512.0, 1024.0)
CONTINUITY_ROWS = (0.0, 8.0, 32.0, 128.0)

# Deliberately explicit: controls, oracle columns and same-model dose scans are
# excluded. Missing historical artifacts are reported and skipped.
SOURCES = (
    ("v168", OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet",
     ("metadata_analog_w0p05",)),
    ("v169", OOF / "historical_analog_ranked_consensus_v169" / "fullrow_oof.parquet",
     ("nested_consensus",)),
    ("v144", OOF / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet",
     ("consensus_center", "nested_sequence_coefficient", "exact_v76", "coarse_crossfit")),
    ("v66", OOF / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
     ("incumbent_6p377_replay", "full_cp8_ordered")),
    ("v96", OOF / "rawdata_lattice_v76_full773_training_v96" / "fivefold_coarse_oof.parquet",
     ("coarse_crossfit", "v96_773_crossfit")),
    ("psg", OOF / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet",
     ("psg_tvt",)),
    ("track8", OOF / "track8_sp45_foldclean_fivefold_v2" / "track8_oof.parquet",
     ("track8", "common_only")),
    ("legacy", OOF / "frontier_6p39_existing_models_oof_v0" / "frontier_6p39_core_oof.parquet",
     ("sp45", "strong_bias", "strong", "cp12", "fleongg")),
    ("neighbor", OOF / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet",
     ("fle_neighbor", "fle_parent")),
    ("stride", OOF / "stride_independent_engine_v155" / "stride_oof.parquet",
     ("stride_full",)),
)


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, names = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error * error, minlength=len(names))
    well_rmse = np.sqrt(sse / counts)
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _metrics_with_folds(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    result = _metrics(frame, prediction, endpoint)
    fold_values = frame["fold"].to_numpy(np.int8)
    for fold in range(1, 6):
        mask = fold_values == fold
        result[f"fold{fold}_rmse"] = _metrics(
            frame.loc[mask].reset_index(drop=True),
            np.asarray(prediction)[mask],
            endpoint,
        )["rmse"]
    result["maximum_fold_rmse"] = max(result[f"fold{fold}_rmse"] for fold in range(1, 6))
    return result


def _load_aligned(path: Path, columns: tuple[str, ...], canonical: pd.DataFrame) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    schema = set(pq.ParquetFile(path).schema.names)
    available = [column for column in columns if column in schema]
    if not available or not {"well", "row_idx"}.issubset(schema):
        return {}
    local = pd.read_parquet(path, columns=["well", "row_idx", *available])
    local["well"] = local["well"].astype(str)
    local["row_idx"] = local["row_idx"].astype(np.int64)
    local = local.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if len(local) != len(canonical):
        return {}
    if not np.array_equal(local["row_idx"].to_numpy(), canonical["row_idx"].to_numpy()):
        return {}
    if not np.array_equal(local["well"].to_numpy(), canonical["well"].to_numpy()):
        return {}
    output = {}
    for column in available:
        values = local[column].to_numpy(np.float32)
        if np.isfinite(values).all():
            output[column] = values
    return output


def _block_path(
    truth: np.ndarray,
    predictions: np.ndarray,
    distance: np.ndarray,
    block_ft: float,
    continuity_rows: float,
    base_index: int,
) -> tuple[np.ndarray, int]:
    """Return an oracle endpoint path with a first-order boundary penalty."""
    rows = len(truth)
    output = predictions[base_index].copy()
    active = distance > FREEZE_FT
    if not np.any(active):
        return output, 0
    labels = np.floor((distance[active] - FREEZE_FT) / block_ft).astype(np.int32)
    block_ids = np.unique(labels)
    k = predictions.shape[0]
    costs = np.empty((len(block_ids), k), np.float64)
    starts = np.empty(len(block_ids), np.int64)
    active_positions = np.flatnonzero(active)
    for b, block_id in enumerate(block_ids):
        positions = active_positions[labels == block_id]
        starts[b] = positions[0]
        error = predictions[:, positions] - truth[None, positions]
        costs[b] = np.sum(error * error, axis=1)

    dp = costs[0].copy()
    back = np.zeros((len(block_ids), k), np.int16)
    for b in range(1, len(block_ids)):
        boundary = starts[b]
        # Penalize a discontinuity introduced solely by changing engines.
        jump = predictions[:, boundary][None, :] - predictions[:, boundary][:, None]
        transition = continuity_rows * jump * jump
        total = dp[:, None] + transition
        back[b] = np.argmin(total, axis=0).astype(np.int16)
        dp = costs[b] + np.min(total, axis=0)

    states = np.empty(len(block_ids), np.int16)
    states[-1] = int(np.argmin(dp))
    for b in range(len(block_ids) - 1, 0, -1):
        states[b - 1] = back[b, states[b]]
    for b, block_id in enumerate(block_ids):
        positions = active_positions[labels == block_id]
        output[positions] = predictions[states[b], positions]
    return output, int(np.count_nonzero(states[1:] != states[:-1]))


if not BASE_PATH.is_file():
    raise FileNotFoundError(BASE_PATH)
base = pd.read_parquet(BASE_PATH)
required = {"well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN}
missing = sorted(required.difference(base.columns))
if missing:
    raise RuntimeError(f"V168 missing columns: {missing}")
base["well"] = base["well"].astype(str)
base["row_idx"] = base["row_idx"].astype(np.int64)
base = base.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Invalid canonical coverage: rows={len(base)} wells={base['well'].nunique()}")
base_sha = hashlib.sha256(base[BASE_COLUMN].to_numpy(np.float32).tobytes()).hexdigest()
if base_sha != EXPECTED_BASE_SHA:
    raise RuntimeError(f"V168 prediction SHA mismatch: {base_sha}")

candidate_arrays: dict[str, np.ndarray] = {"v168/metadata_analog_w0p05": base[BASE_COLUMN].to_numpy(np.float32)}
source_report = []
for family, path, columns in SOURCES:
    loaded = _load_aligned(path, columns, base)
    source_report.append({
        "family": family,
        "path": str(path),
        "status": "loaded" if loaded else ("missing" if not path.is_file() else "incompatible"),
        "columns": list(loaded),
    })
    for column, values in loaded.items():
        candidate_arrays[f"{family}/{column}"] = values

# Remove exact duplicate paths without using truth.
unique_arrays: dict[str, np.ndarray] = {}
seen_hashes: set[str] = set()
for name, values in candidate_arrays.items():
    digest = hashlib.sha256(np.asarray(values, np.float32).tobytes()).hexdigest()
    if digest not in seen_hashes:
        unique_arrays[name] = values
        seen_hashes.add(digest)
candidate_arrays = unique_arrays
if len(candidate_arrays) < 5:
    raise RuntimeError(f"Only {len(candidate_arrays)} unique complete endpoints available")

names = list(candidate_arrays)
base_index = names.index("v168/metadata_analog_w0p05")
matrix = np.stack([candidate_arrays[name] for name in names]).astype(np.float64)
truth_all = base["tvt_true"].to_numpy(np.float64)

standalone = pd.DataFrame([_metrics_with_folds(base, matrix[index], name) for index, name in enumerate(names)])
standalone = standalone.sort_values("rmse", ignore_index=True)
results = [_metrics_with_folds(base, matrix[base_index], "v168") | {
    "block_ft": np.nan, "continuity_rows": np.nan, "switches": 0,
}]

well_groups = list(base.groupby("well", sort=False).indices.items())
for block_ft in BLOCK_FT:
    for continuity_rows in CONTINUITY_ROWS:
        prediction = np.empty(len(base), np.float64)
        switches = 0
        for _, positions in well_groups:
            positions = np.asarray(positions, np.int64)
            local, local_switches = _block_path(
                truth_all[positions], matrix[:, positions],
                base.iloc[positions]["md_since"].to_numpy(np.float64),
                block_ft, continuity_rows, base_index,
            )
            prediction[positions] = local
            switches += local_switches
        result = _metrics_with_folds(
            base, prediction,
            f"block_oracle_b{int(block_ft)}_c{int(continuity_rows)}",
        )
        result.update({
            "block_ft": block_ft,
            "continuity_rows": continuity_rows,
            "switches": switches,
        })
        results.append(result)
        print("V187", {key: result[key] for key in ("endpoint", "rmse", "p95_well_rmse", "far_gt2000_rmse", "switches")}, flush=True)

summary = pd.DataFrame(results).sort_values("rmse", ignore_index=True)
continuous = summary[summary["continuity_rows"].fillna(-1).ge(32.0)]
raw = summary[summary["continuity_rows"].fillna(-1).eq(0.0)]
best = raw.iloc[0]
best_continuous = continuous.iloc[0]
base_rmse = float(summary.loc[summary["endpoint"].eq("v168"), "rmse"].iloc[0])
checks = {
    "coverage_exact": len(base) == EXPECTED_ROWS and base["well"].nunique() == EXPECTED_WELLS,
    "unique_engines_ge_5": len(names) >= 5,
    "raw_block_oracle_lt_6": float(best["rmse"]) < 6.0,
    "continuous_block_oracle_lt_6": float(best_continuous["rmse"]) < 6.0,
    "continuous_gain_ge_1": base_rmse - float(best_continuous["rmse"]) >= 1.0,
    "continuous_far_rmse_lt_7": float(best_continuous["far_gt2000_rmse"]) < 7.0,
    "continuous_maximum_fold_rmse_lt_6p5": float(best_continuous["maximum_fold_rmse"]) < 6.5,
}
investment_pass = bool(all(checks.values()))
verdict = {
    "experiment": "canonical_multiengine_block_oracle_v187",
    "stage": "complete_773well_fullrow_local_engine_support_gate",
    "deployable": 0,
    "truth_usage": "block endpoint selection and metrics only",
    "base": str(BASE_PATH),
    "base_column": BASE_COLUMN,
    "base_prediction_sha256": base_sha,
    "candidate_endpoints": names,
    "candidate_count": len(names),
    "source_report": source_report,
    "base_rmse": base_rmse,
    "best_raw": best.to_dict(),
    "best_continuous": best_continuous.to_dict(),
    "checks": checks,
    "investment_pass": investment_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train a legal blockwise engine selector with fixed blocks and continuity"
        if investment_pass else
        "FAIL -> existing engines cannot support CV<6 even locally; rebuild candidate generation"
    ),
}
standalone.to_csv(OUT / "standalone_endpoints.csv", index=False)
summary.to_csv(OUT / "block_oracle_summary.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
print("\nV187 STANDALONE")
print(standalone.to_string(index=False), flush=True)
print("\nV187 BLOCK ORACLE")
print(summary.to_string(index=False), flush=True)
print("\nV187 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=str), flush=True)
print("saved:", OUT, flush=True)
