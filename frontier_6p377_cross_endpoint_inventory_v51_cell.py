"""Cross-endpoint ceiling audit on the exact LB 6.377 lineage.

This is a zero-training investment gate. It aligns every available OOF
endpoint to the exact V33 rows, freezes the first 2000 ft to the current
6.377 prediction, and measures:

* standalone and far-tail RMSE;
* raw and distance-detrended signed-error correlation with 6.377;
* per-well and fixed-length segment oracle ceilings;
* fold stability of the oracle gain.

Truth is used only for diagnostics and oracle selection. No prediction from
this cell is deployable.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_cross_endpoint_inventory_v51"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
V28_PATH = (
    OOF_ROOT
    / "frontier_6p39_tail_endpoint_oracle_v28"
    / "tail_endpoint_oof.parquet"
)
OPTIONAL_SOURCES = (
    (
        "track8",
        OOF_ROOT / "track8_sp45_foldclean_fivefold_v2" / "track8_oof.parquet",
        ("common_only", "track8"),
    ),
    (
        "stride",
        OOF_ROOT / "stride_backbone_773_crossfit_v2" / "stride_backbone_oof.parquet",
        ("stride_backbone",),
    ),
    (
        "psg",
        OOF_ROOT / "psg_sp45_crossfit_audit_v0" / "psg_sp45_crossfit_oof.parquet",
        ("sp45", "psg_tvt", "psg_guarded", "crossfit_blend"),
    ),
)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FROZEN_THROUGH_FT = 2000.0
FADE_FT = 512.0
CAP_FT = 48.0
SEGMENT_LENGTHS_FT = (256.0, 512.0, 1024.0)
MIN_COVERAGE = 0.95
started = time.time()


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


def _pearson(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 10 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _detrend_distance(error, distance, bin_ft=256.0):
    error = np.asarray(error, np.float64)
    distance = np.asarray(distance, np.float64)
    bins = np.floor(np.maximum(distance, 0.0) / float(bin_ft)).astype(np.int32)
    output = error.copy()
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        output[mask] -= np.mean(error[mask])
    return output


def _smoothstep(distance):
    t = np.clip(
        (np.asarray(distance, np.float64) - FROZEN_THROUGH_FT) / FADE_FT,
        0.0,
        1.0,
    )
    return t * t * (3.0 - 2.0 * t)


def _metrics(prediction, name, mask=None):
    use = np.ones(len(frame), bool) if mask is None else np.asarray(mask, bool)
    error = np.asarray(prediction, np.float64)[use] - truth[use]
    local_wells = wells[use]
    local_distance = distance[use]
    codes, names = pd.factorize(local_wells, sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error**2, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    far = local_distance > FROZEN_THROUGH_FT
    return {
        "endpoint": name,
        "rows": int(use.sum()),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "far_gt2000_rmse": (
            float(np.sqrt(np.mean(error[far] ** 2))) if np.any(far) else math.nan
        ),
    }


def _oracle_pair(candidate, segment_ft=None, row_mask=None):
    prediction = base.copy()
    choices = []
    allowed = np.ones(len(frame), bool) if row_mask is None else np.asarray(row_mask, bool)
    for well, indices in groups.items():
        valid = allowed[indices] & (distance[indices] > FROZEN_THROUGH_FT)
        if not np.any(valid):
            continue
        if segment_ft is None:
            blocks = np.zeros(len(indices), np.int32)
        else:
            blocks = np.floor(
                np.maximum(distance[indices] - FROZEN_THROUGH_FT, 0.0)
                / float(segment_ft)
            ).astype(np.int32)
        for block in np.unique(blocks[valid]):
            local = np.flatnonzero(valid & (blocks == block))
            rows = indices[local]
            base_sse = float(np.sum((base[rows] - truth[rows]) ** 2))
            candidate_sse = float(np.sum((candidate[rows] - truth[rows]) ** 2))
            use_candidate = candidate_sse < base_sse
            if use_candidate:
                prediction[rows] = candidate[rows]
            choices.append((well, int(block), int(use_candidate), base_sse, candidate_sse))
    return prediction, choices


def _oracle_bank(segment_ft=None):
    prediction = base.copy()
    winning = np.zeros(len(candidate_names), np.int64)
    for well, indices in groups.items():
        valid = distance[indices] > FROZEN_THROUGH_FT
        if not np.any(valid):
            continue
        if segment_ft is None:
            blocks = np.zeros(len(indices), np.int32)
        else:
            blocks = np.floor(
                np.maximum(distance[indices] - FROZEN_THROUGH_FT, 0.0)
                / float(segment_ft)
            ).astype(np.int32)
        for block in np.unique(blocks[valid]):
            local = np.flatnonzero(valid & (blocks == block))
            rows = indices[local]
            errors = candidate_matrix[rows].astype(np.float64) - truth[rows, None]
            state = int(np.argmin(np.sum(errors**2, axis=0)))
            prediction[rows] = candidate_matrix[rows, state]
            winning[state] += 1
    return prediction, dict(zip(candidate_names, winning.tolist()))


def _merge_source(source_name, path, wanted_columns):
    if not path.is_file():
        source_inventory.append(
            {"source": source_name, "path": str(path), "status": "missing"}
        )
        return
    schema = pq.ParquetFile(path).schema.names
    key = "id" if "id" in schema else None
    if key is None:
        source_inventory.append(
            {"source": source_name, "path": str(path), "status": "no_id"}
        )
        return
    usable = [column for column in wanted_columns if column in schema]
    if not usable:
        source_inventory.append(
            {"source": source_name, "path": str(path), "status": "no_candidates"}
        )
        return
    local = pd.read_parquet(path, columns=[key] + usable)
    local[key] = local[key].astype(str).str.lower()
    if local[key].duplicated().any():
        raise RuntimeError(f"Duplicate ids in {path}")
    local = local.rename(columns={column: f"{source_name}__{column}" for column in usable})
    before = set(frame.columns)
    frame_merge = frame.merge(local, on="id", how="left", validate="one_to_one")
    frame.drop(columns=list(frame.columns), inplace=True)
    for column in frame_merge.columns:
        frame[column] = frame_merge[column]
    added = sorted(set(frame.columns).difference(before))
    source_inventory.append(
        {
            "source": source_name,
            "path": str(path),
            "status": "loaded",
            "columns": added,
        }
    )


for required in (V33_PATH, V28_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v33_columns = [
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "pre_neighbor_cp8", BASE_COLUMN,
]
frame = pd.read_parquet(V33_PATH, columns=v33_columns)
frame = frame[frame["fold"].isin(FOLDS)].copy()
frame["id"] = frame["id"].astype(str).str.lower()
frame["well"] = frame["well"].astype(str).str.lower()
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in V33")

source_inventory = []
v28_schema = pq.ParquetFile(V28_PATH).schema.names
v28_candidates = ["neighbor_v27"] + sorted(
    column for column in v28_schema if column.startswith("tail_")
)
_merge_source("v28", V28_PATH, tuple(v28_candidates))
for source_name, path, columns in OPTIONAL_SOURCES:
    _merge_source(source_name, path, columns)

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
wells = frame["well"].astype(str).to_numpy()
fold_values = frame["fold"].to_numpy(np.int8)
fade = _smoothstep(distance)
frozen = distance <= FROZEN_THROUGH_FT
far = ~frozen
groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}

raw_candidate_columns = ["pre_neighbor_cp8"]
for item in source_inventory:
    raw_candidate_columns.extend(item.get("columns", []))
raw_candidate_columns = list(dict.fromkeys(raw_candidate_columns))

candidate_names = ["base_6p377"]
candidate_paths = [base.astype(np.float32)]
coverage_rows = []
raw_metric_rows = []
for column in raw_candidate_columns:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(values)
    coverage = float(finite.mean())
    coverage_rows.append({"candidate": column, "coverage": coverage})
    if coverage < MIN_COVERAGE:
        continue
    filled = np.where(finite, values, base)
    raw_metric_rows.append(_metrics(filled, f"raw__{column}"))
    correction = np.clip(filled - base, -CAP_FT, CAP_FT)
    tail_path = base + fade * correction
    tail_path[frozen] = base[frozen]
    candidate_names.append(column)
    candidate_paths.append(tail_path.astype(np.float32))

candidate_matrix = np.column_stack(candidate_paths)
if candidate_matrix.shape[1] < 3:
    raise RuntimeError(f"Too few usable candidates: {candidate_names}")
if not np.isfinite(candidate_matrix).all():
    raise RuntimeError("Candidate matrix contains non-finite values")
if not np.array_equal(
    candidate_matrix[frozen],
    np.repeat(base[frozen, None].astype(np.float32), candidate_matrix.shape[1], axis=1),
):
    raise RuntimeError("A candidate changed rows through 2000 ft")

print(
    "V51 ENDPOINT CONFIG",
    {
        "rows": len(frame),
        "wells": len(groups),
        "candidates": candidate_names,
        "optional_sources": source_inventory,
    },
    flush=True,
)

base_error = base - truth
base_error_detrended = _detrend_distance(base_error, distance)
base_metrics = _metrics(base, "base_6p377")
summary_rows = []
fold_rows = []
choice_rows = []

for state, name in enumerate(candidate_names[1:], 1):
    candidate = candidate_matrix[:, state].astype(np.float64)
    error = candidate - truth
    error_detrended = _detrend_distance(error, distance)
    candidate_metrics = _metrics(candidate, name)

    whole_prediction, whole_choices = _oracle_pair(candidate, segment_ft=None)
    whole_metrics = _metrics(whole_prediction, f"{name}__wholewell_oracle")
    segment_metrics = {}
    for segment_ft in SEGMENT_LENGTHS_FT:
        prediction, choices = _oracle_pair(candidate, segment_ft=segment_ft)
        metrics = _metrics(prediction, f"{name}__segment_{int(segment_ft)}_oracle")
        segment_metrics[int(segment_ft)] = metrics
        for well, block, selected, base_sse, candidate_sse in choices:
            choice_rows.append(
                {
                    "candidate": name,
                    "segment_ft": float(segment_ft),
                    "well": well,
                    "block": block,
                    "selected_candidate": selected,
                    "base_sse": base_sse,
                    "candidate_sse": candidate_sse,
                }
            )

    fold_segment_gains = []
    for fold in FOLDS:
        fold_mask = fold_values == fold
        fold_base = _metrics(base, "base", fold_mask)
        fold_candidate = _metrics(candidate, name, fold_mask)
        fold_whole_prediction, _ = _oracle_pair(
            candidate, segment_ft=None, row_mask=fold_mask
        )
        fold_segment_prediction, _ = _oracle_pair(
            candidate, segment_ft=512.0, row_mask=fold_mask
        )
        fold_whole = _metrics(fold_whole_prediction, "whole", fold_mask)
        fold_segment = _metrics(fold_segment_prediction, "segment512", fold_mask)
        fold_gain = fold_base["rmse"] - fold_segment["rmse"]
        fold_segment_gains.append(fold_gain)
        fold_rows.append(
            {
                "candidate": name,
                "fold": fold,
                "base_rmse": fold_base["rmse"],
                "candidate_rmse": fold_candidate["rmse"],
                "wholewell_oracle_rmse": fold_whole["rmse"],
                "wholewell_oracle_gain": fold_base["rmse"] - fold_whole["rmse"],
                "segment512_oracle_rmse": fold_segment["rmse"],
                "segment512_oracle_gain": fold_gain,
            }
        )

    summary_rows.append(
        {
            "candidate": name,
            "candidate_rmse": candidate_metrics["rmse"],
            "candidate_far_rmse": candidate_metrics["far_gt2000_rmse"],
            "raw_error_rho": _pearson(base_error, error),
            "far_error_rho": _pearson(base_error[far], error[far]),
            "detrended_error_rho": _pearson(base_error_detrended, error_detrended),
            "detrended_far_error_rho": _pearson(
                base_error_detrended[far], error_detrended[far]
            ),
            "wholewell_oracle_rmse": whole_metrics["rmse"],
            "wholewell_oracle_gain": base_metrics["rmse"] - whole_metrics["rmse"],
            "segment256_oracle_rmse": segment_metrics[256]["rmse"],
            "segment256_oracle_gain": base_metrics["rmse"] - segment_metrics[256]["rmse"],
            "segment512_oracle_rmse": segment_metrics[512]["rmse"],
            "segment512_oracle_gain": base_metrics["rmse"] - segment_metrics[512]["rmse"],
            "segment1024_oracle_rmse": segment_metrics[1024]["rmse"],
            "segment1024_oracle_gain": base_metrics["rmse"] - segment_metrics[1024]["rmse"],
            "minimum_fold_segment512_gain": float(min(fold_segment_gains)),
            "candidate_selected_well_fraction": float(
                np.mean([choice[2] for choice in whole_choices])
            ),
        }
    )

summary = pd.DataFrame(summary_rows).sort_values(
    ["segment512_oracle_gain", "detrended_far_error_rho"],
    ascending=[False, True],
).reset_index(drop=True)
fold_table = pd.DataFrame(fold_rows)
choices = pd.DataFrame(choice_rows)

bank_rows = []
bank_predictions = {}
for segment_ft in (None,) + SEGMENT_LENGTHS_FT:
    prediction, winners = _oracle_bank(segment_ft)
    label = "wholewell" if segment_ft is None else f"segment_{int(segment_ft)}"
    metrics = _metrics(prediction, f"bank_{label}_oracle")
    metrics["gain_vs_base"] = base_metrics["rmse"] - metrics["rmse"]
    metrics["winning_counts"] = winners
    bank_rows.append(metrics)
    bank_predictions[label] = prediction

summary["oracle_gain_ge_0p30"] = summary[
    ["wholewell_oracle_gain", "segment512_oracle_gain"]
].max(axis=1) >= 0.30
summary["low_detrended_far_rho"] = summary["detrended_far_error_rho"] <= 0.85
summary["fold_stable_oracle"] = summary["minimum_fold_segment512_gain"] >= 0.05
summary["investable"] = (
    summary["oracle_gain_ge_0p30"]
    & summary["low_detrended_far_rho"]
    & summary["fold_stable_oracle"]
)

summary.to_csv(OUT_DIR / "endpoint_summary.csv", index=False)
fold_table.to_csv(OUT_DIR / "endpoint_fold_oracles.csv", index=False)
choices.to_csv(OUT_DIR / "segment512_choices.csv", index=False)
pd.DataFrame(coverage_rows).to_csv(OUT_DIR / "coverage.csv", index=False)
pd.DataFrame(raw_metric_rows).to_csv(OUT_DIR / "raw_endpoint_metrics.csv", index=False)

investable = summary.loc[summary["investable"]]
verdict = {
    "experiment": EXPERIMENT,
    "stage": "exact_lineage_cross_endpoint_error_correlation_and_oracle_inventory",
    "sample_unit": "complete physical well and fixed far segments",
    "truth_usage": "diagnostic oracle only; no deployable prediction",
    "parent": str(V33_PATH),
    "rows": int(len(frame)),
    "wells": int(len(groups)),
    "base": base_metrics,
    "sources": source_inventory,
    "candidates": candidate_names,
    "bank_oracles": bank_rows,
    "gates": {
        "oracle_gain_ge": 0.30,
        "detrended_far_error_rho_le": 0.85,
        "minimum_fold_segment512_gain_ge": 0.05,
    },
    "investable_candidates": investable["candidate"].tolist(),
    "strict_pass": bool(len(investable)),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> train only on the surviving endpoint identities"
        if len(investable)
        else "FAIL -> no existing endpoint has enough independent directional ceiling; generate new evidence"
    ),
}
_write_json(OUT_DIR / "result.json", verdict)

print("\nV51 ENDPOINT SUMMARY", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nV51 FOLD ORACLES", flush=True)
print(fold_table.to_string(index=False), flush=True)
print("\nV51 BANK ORACLES", flush=True)
print(pd.DataFrame(bank_rows).drop(columns=["winning_counts"]).to_string(index=False), flush=True)
print("\nV51 FINAL VERDICT", flush=True)
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
