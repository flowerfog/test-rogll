"""Far-only segment endpoint oracle on the exact 6.377 lineage.

This is an investment gate, not a deployable selector.  It asks whether
existing non-SGM endpoints contain useful local corrections after the current
CP8 -> Neighbor path.  Rows at or before 2000 ft are bitwise frozen.  Truth is
used only to select a candidate independently inside complete-well segments.

The raw segment oracle is reported together with a deployability-oriented
version whose correction is clipped and smoothed across segment boundaries.
Only the latter is allowed to pass the investment gate.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_far_segment_endpoint_oracle_v40"
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
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
SEGMENT_LENGTHS_FT = (256.0, 512.0, 1024.0)
FOLDS = (1, 2, 3)
started = time.time()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def _smoothstep(distance):
    t = np.clip((np.asarray(distance, np.float64) - FAR_START_FT) / FADE_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _centred_mean(md, values, width_ft):
    md = np.asarray(md, np.float64)
    values = np.asarray(values, np.float64)
    half = 0.5 * float(width_ft)
    left = np.searchsorted(md, md - half, side="left")
    right = np.searchsorted(md, md + half, side="right")
    csum = np.concatenate(([0.0], np.cumsum(values)))
    count = np.maximum(right - left, 1)
    return (csum[right] - csum[left]) / count


def _metrics(local, prediction, endpoint):
    truth = local["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if len(error) != len(local) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    per_well = (
        pd.DataFrame({"well": local["well"].astype(str), "error2": error**2})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    distance = local["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
    }
    masks = {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in masks.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
        )
    return result


for required in (V33_PATH, V28_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v33_columns = pq.ParquetFile(V33_PATH).schema.names
required_v33 = [
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "pre_neighbor_cp8", BASE_COLUMN,
]
missing_v33 = sorted(set(required_v33).difference(v33_columns))
if missing_v33:
    raise RuntimeError(f"V33 columns missing: {missing_v33}")

frame = pd.read_parquet(V33_PATH, columns=required_v33)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("V33 contains duplicate ids")

v28_columns = pq.ParquetFile(V28_PATH).schema.names
tail_columns = sorted(column for column in v28_columns if column.startswith("tail_"))
if not tail_columns:
    raise RuntimeError(f"No tail endpoint columns in {V28_PATH}: {v28_columns}")
candidate_frame = pd.read_parquet(V28_PATH, columns=["id"] + tail_columns)
candidate_frame["id"] = candidate_frame["id"].astype(str)
if candidate_frame["id"].duplicated().any():
    raise RuntimeError("V28 contains duplicate ids")
frame = frame.merge(candidate_frame, on="id", how="left", validate="one_to_one")

numeric_required = ["tvt_true", "md_since", "pre_neighbor_cp8", BASE_COLUMN]
if frame[numeric_required].isna().any().any():
    raise RuntimeError("Missing V33 values after alignment")

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fade = _smoothstep(distance)
prefix = distance <= FAR_START_FT

# V28 endpoints came from an earlier parent.  They enter only as bounded
# correction directions around the exact V33 path, never as replacement paths.
candidate_names = ["pre_neighbor_cp8"] + tail_columns
candidate_paths = [base]
usable_names = ["base_6p377"]
coverage = {}
for name in candidate_names:
    absolute = frame[name].to_numpy(np.float64)
    finite = np.isfinite(absolute)
    coverage[name] = float(finite.mean())
    if float(finite.mean()) < 0.95:
        continue
    correction = np.where(finite, np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[prefix] = base[prefix]
    candidate_paths.append(candidate)
    usable_names.append(name)

candidates = np.column_stack(candidate_paths)
if candidates.shape[1] < 3:
    raise RuntimeError(f"Too few aligned candidates: {usable_names}")
if not np.isfinite(candidates).all():
    raise RuntimeError("Non-finite candidate matrix")
if not np.array_equal(candidates[prefix], np.repeat(base[prefix, None], candidates.shape[1], axis=1)):
    raise RuntimeError("A candidate changed rows at or before 2000 ft")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}
print(
    "V40 CONFIG",
    {
        "rows": len(frame),
        "wells": len(groups),
        "base": BASE_COLUMN,
        "candidate_count": len(usable_names),
        "candidates": usable_names,
        "segment_lengths_ft": SEGMENT_LENGTHS_FT,
        "far_start_ft": FAR_START_FT,
        "smooth_ft": SMOOTH_FT,
    },
    flush=True,
)


def _segment_oracle(segment_ft):
    raw = base.copy()
    feasible = base.copy()
    choice_rows = []
    for well, indices in groups.items():
        local_distance = distance[indices]
        far_local = local_distance > FAR_START_FT
        selected = base[indices].copy()
        selected_state = np.zeros(len(indices), np.int16)
        if np.any(far_local):
            block = np.floor(
                np.maximum(local_distance - FAR_START_FT, 0.0) / float(segment_ft)
            ).astype(np.int32)
            for block_id in np.unique(block[far_local]):
                local_rows = np.flatnonzero(far_local & (block == block_id))
                global_rows = indices[local_rows]
                errors = candidates[global_rows] - truth[global_rows, None]
                sse = np.sum(errors**2, axis=0)
                state = int(np.argmin(sse))
                selected[local_rows] = candidates[global_rows, state]
                selected_state[local_rows] = state
                choice_rows.append(
                    {
                        "well": well,
                        "fold": int(frame.loc[global_rows[0], "fold"]),
                        "segment_ft": float(segment_ft),
                        "block": int(block_id),
                        "rows": int(len(global_rows)),
                        "state": state,
                        "candidate": usable_names[state],
                        "base_sse": float(sse[0]),
                        "selected_sse": float(sse[state]),
                    }
                )
        raw[indices] = selected
        correction = selected - base[indices]
        correction = _centred_mean(
            frame.loc[indices, "md_since"].to_numpy(np.float64), correction, SMOOTH_FT
        )
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        local_feasible = base[indices] + fade[indices] * correction
        local_feasible[local_distance <= FAR_START_FT] = base[indices][
            local_distance <= FAR_START_FT
        ]
        feasible[indices] = local_feasible
    return raw, feasible, pd.DataFrame(choice_rows)


summary_rows = []
fold_rows = []
choice_tables = []
base_metrics = _metrics(frame, base, "base_6p377")
for segment_ft in SEGMENT_LENGTHS_FT:
    raw, feasible, choices = _segment_oracle(segment_ft)
    choice_tables.append(choices)
    for endpoint, prediction in (("raw_oracle", raw), ("feasible_oracle", feasible)):
        result = _metrics(frame, prediction, endpoint)
        result.update(
            {
                "segment_ft": float(segment_ft),
                "gain": base_metrics["rmse"] - result["rmse"],
                "far_gain": base_metrics["far_gt2000_rmse"] - result["far_gt2000_rmse"],
                "p95_change": result["p95_well_rmse"] - base_metrics["p95_well_rmse"],
            }
        )
        summary_rows.append(result)
        for held_fold in FOLDS:
            mask = frame["fold"].to_numpy(np.int8) == held_fold
            local = frame.loc[mask].reset_index(drop=True)
            local_base = _metrics(local, base[mask], "base_6p377")
            local_result = _metrics(local, prediction[mask], endpoint)
            fold_rows.append(
                {
                    "fold": held_fold,
                    "segment_ft": float(segment_ft),
                    "endpoint": endpoint,
                    "base_rmse": local_base["rmse"],
                    "rmse": local_result["rmse"],
                    "gain": local_base["rmse"] - local_result["rmse"],
                    "base_far_rmse": local_base["far_gt2000_rmse"],
                    "far_rmse": local_result["far_gt2000_rmse"],
                    "far_gain": local_base["far_gt2000_rmse"] - local_result["far_gt2000_rmse"],
                    "p95_change": local_result["p95_well_rmse"] - local_base["p95_well_rmse"],
                }
            )
    print("V40 SEGMENT", segment_ft, summary_rows[-1], flush=True)

summary = pd.DataFrame(summary_rows)
fold_summary = pd.DataFrame(fold_rows)
choices = pd.concat(choice_tables, ignore_index=True)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
choices.to_csv(OUT_DIR / "oracle_segment_choices.csv", index=False)

feasible = summary[summary["endpoint"].eq("feasible_oracle")].copy()
feasible = feasible.sort_values(["gain", "far_gain"], ascending=False, kind="stable")
selected_segment_ft = float(feasible.iloc[0]["segment_ft"])
selected_folds = fold_summary[
    fold_summary["endpoint"].eq("feasible_oracle")
    & np.isclose(fold_summary["segment_ft"], selected_segment_ft)
]
selected = feasible.iloc[0].to_dict()

checks = {
    "feasible_gain_ge_0p15": bool(selected["gain"] >= 0.15),
    "feasible_far_gain_ge_0p25": bool(selected["far_gain"] >= 0.25),
    "each_fold_gain_ge_0p08": bool(selected_folds["gain"].min() >= 0.08),
    "each_fold_far_gain_ge_0p15": bool(selected_folds["far_gain"].min() >= 0.15),
    "near_le2000_bitwise_frozen": True,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "exact_6p377_far_only_segment_endpoint_oracle_gate",
    "deployable": False,
    "truth_usage": "truth selects one frozen endpoint per complete-well far segment",
    "lineage": {
        "base": str(V33_PATH),
        "base_column": BASE_COLUMN,
        "candidate_source": str(V28_PATH),
        "tail_sgm_used": False,
        "pseudo_typewell_used": False,
    },
    "policy": {
        "far_start_ft": FAR_START_FT,
        "fade_ft": FADE_FT,
        "cap_ft": CAP_FT,
        "boundary_smooth_ft": SMOOTH_FT,
        "segment_lengths_ft": SEGMENT_LENGTHS_FT,
    },
    "candidate_names": usable_names,
    "candidate_coverage": coverage,
    "base": base_metrics,
    "selected_segment_ft": selected_segment_ft,
    "selected_feasible_oracle": selected,
    "selected_folds": selected_folds.to_dict(orient="records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": (
        "PASS -> freeze segment scale and build V41 nested legal segment selector"
        if strict_pass
        else "FAIL -> existing endpoints do not contain enough far-segment complementarity; close routing"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV40 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV40 FOLDS")
print(fold_summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
