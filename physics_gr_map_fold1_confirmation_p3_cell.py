"""Untouched Fold-1 confirmation of the frozen P2 raw-GR blend.

The arm and global 0.05 dose are frozen from P1/P2.  Fold 1 is decoded once;
no parameter is selected here.  The exact 6.377 lineage supplies the base.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "physics_gr_map_fold1_confirmation_p3"
ROOT = Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT if os.name != "nt" else Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V3_SOURCE = ROOT / "rawdata_lattice_mode_union_v3_cell.py"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
BASE_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
BASE_COLUMN = "current_dose_0p20"
PRIOR_FOLDS = (4, 5)
CONFIRMATION_FOLD = 1
FROZEN_MODE = "global"
FROZEN_WEIGHT = 0.05
started = time.time()

ARM = {
    "id": "raw_clip_b512_shape4",
    "beam": 512,
    "diversity_ft": 4.0,
    "reference": "raw",
    "likelihood": "clipped_gaussian",
    "history_diversity": True,
}


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [clean(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    work = pd.DataFrame({"well": frame["well"].to_numpy(), "e2": error * error})
    bywell = np.sqrt(work.groupby("well")["e2"].mean().to_numpy(np.float64))
    distance = frame["md_since"].to_numpy(np.float64)
    near = distance <= 2000.0
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        "max_well_rmse": float(np.max(bywell)),
        "near_le2000_rmse": float(np.sqrt(np.mean(error[near] ** 2))) if near.any() else None,
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else None,
    }


if not V3_SOURCE.is_file():
    raise FileNotFoundError(V3_SOURCE)
if not FOLD_PATH.is_file():
    raise FileNotFoundError(FOLD_PATH)
if not BASE_PATH.is_file():
    raise FileNotFoundError(BASE_PATH)

source = V3_SOURCE.read_text(encoding="utf-8")
marker = "# --- V3 EXECUTION ---"
if marker not in source:
    raise RuntimeError("V3 execution marker not found")
N = {"__name__": "physics_p3_v3_definitions", "__file__": str(V3_SOURCE)}
exec(compile(source.split(marker, 1)[0], str(V3_SOURCE), "exec"), N)
B = N["B"]
V = N["V"]
DATA_ROOT = N["DATA_ROOT"]

folds = pd.read_csv(FOLD_PATH)
well_column = next((column for column in ("well", "well_id", "pair_id") if column in folds), None)
fold_column = next((column for column in ("fold", "outer_fold") if column in folds), None)
if well_column is None or fold_column is None:
    raise RuntimeError(f"Cannot identify fold columns: {folds.columns.tolist()}")
folds = folds[[well_column, fold_column]].rename(columns={well_column: "well", fold_column: "fold"})
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
if folds["well"].duplicated().any():
    raise RuntimeError("Duplicate wells in fold metadata")
fold_by_well = dict(zip(folds["well"], folds["fold"]))

needed_wells = sorted(
    well for well, fold in fold_by_well.items() if fold in (*PRIOR_FOLDS, CONFIRMATION_FOLD)
)
records = {}
failures = []
for number, well in enumerate(needed_wells, 1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "stage": "load", "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed_wells):
        print(f"P3 load {number}/{len(needed_wells)} success={len(records)} failures={len(failures)}", flush=True)

prior_records = [records[well] for well in needed_wells if well in records and fold_by_well[well] in PRIOR_FOLDS]
if len(prior_records) < 300:
    raise RuntimeError(f"Only {len(prior_records)} prior wells")
prior = V["_fit_prior"](prior_records, float(N["SEGMENT_LENGTH_FT"]))
prior_payload = clean(prior)
(OUT_DIR / "frozen_prior.json").write_text(
    json.dumps(prior_payload, indent=2, allow_nan=False), encoding="utf-8"
)

confirmation_wells = [
    well for well in needed_wells if well in records and fold_by_well[well] == CONFIRMATION_FOLD
]
frames = []
for number, well in enumerate(confirmation_wells, 1):
    record = records[well]
    try:
        scores, paths, segments = N["_beam_bank_v3"](record, prior, ARM)
        path = paths[int(np.argmin(scores))]
        prediction = B["_path_prediction"](record, prior, path, segments)
        rows = record["eval_rows"]
        frames.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in rows],
                    "well": well,
                    "row_idx": rows.astype(np.int32),
                    "physics_fold": CONFIRMATION_FOLD,
                    "physics_truth": record["truth"][rows].astype(np.float32),
                    "physics_md_since": (record["md"][rows] - record["anchor_md"]).astype(np.float32),
                    "gr_map": prediction[rows].astype(np.float32),
                }
            )
        )
    except Exception as exc:
        failures.append({"well": well, "stage": "decode", "reason": repr(exc)})
    if number % 12 == 0 or number == len(confirmation_wells):
        print(f"P3 decode {number}/{len(confirmation_wells)} success={len(frames)} failures={len(failures)}", flush=True)

if not frames:
    raise RuntimeError(f"No Fold1 predictions: {failures[:10]}")
physics = pd.concat(frames, ignore_index=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)

base = pd.read_parquet(
    BASE_PATH,
    columns=["id", "well", "fold", "tvt_true", "md_since", BASE_COLUMN],
)
base = base.loc[base["fold"].eq(CONFIRMATION_FOLD)].copy()
frame = base.merge(physics, on=["id", "well"], how="inner", validate="one_to_one")
if not frame["fold"].eq(frame["physics_fold"]).all():
    raise RuntimeError("Physics/base fold mismatch")
truth_gap = np.abs(frame["tvt_true"].to_numpy(np.float64) - frame["physics_truth"].to_numpy(np.float64))
distance_gap = np.abs(frame["md_since"].to_numpy(np.float64) - frame["physics_md_since"].to_numpy(np.float64))
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
    raise RuntimeError(f"Truth alignment failed: {np.nanmax(truth_gap)}")
if not np.isfinite(distance_gap).all() or float(distance_gap.max()) > 0.01:
    raise RuntimeError(f"Distance alignment failed: {np.nanmax(distance_gap)}")
if frame["well"].nunique() < 150:
    raise RuntimeError(f"Only {frame['well'].nunique()} aligned Fold1 wells")

base_prediction = frame[BASE_COLUMN].to_numpy(np.float64)
candidate_prediction = frame["gr_map"].to_numpy(np.float64)
blend_prediction = base_prediction + FROZEN_WEIGHT * (candidate_prediction - base_prediction)

frame["gr_map_reversed"] = candidate_prediction
for _, positions in frame.groupby("well", sort=False).indices.items():
    positions = np.asarray(positions, np.int64)
    local_base = base_prediction[positions]
    local_delta = candidate_prediction[positions] - local_base
    frame.iloc[positions, frame.columns.get_loc("gr_map_reversed")] = local_base + local_delta[::-1]
reverse_prediction = base_prediction + FROZEN_WEIGHT * (
    frame["gr_map_reversed"].to_numpy(np.float64) - base_prediction
)

base_metrics = metrics(frame, base_prediction)
candidate_metrics = metrics(frame, candidate_prediction)
blend_metrics = metrics(frame, blend_prediction)
reverse_metrics = metrics(frame, reverse_prediction)
gain = base_metrics["rmse"] - blend_metrics["rmse"]
far_gain = base_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"]
reverse_loss = reverse_metrics["rmse"] - blend_metrics["rmse"]
base_error = base_prediction - frame["tvt_true"].to_numpy(np.float64)
candidate_error = candidate_prediction - frame["tvt_true"].to_numpy(np.float64)
rho = float(np.corrcoef(base_error, candidate_error)[0, 1])

checks = {
    "gain_ge_0p03": gain >= 0.03,
    "far_gain_ge_0p03": far_gain >= 0.03,
    "p95_change_le_0p20": blend_metrics["p95_well_rmse"] <= base_metrics["p95_well_rmse"] + 0.20,
    "reversed_alignment_loses_ge_0p03": reverse_loss >= 0.03,
    "aligned_wells_ge_150": frame["well"].nunique() >= 150,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "untouched_fold1_frozen_raw_gr_map_blend_confirmation",
    "sample_unit": "complete physical well",
    "protocol": {
        "prior_folds": PRIOR_FOLDS,
        "confirmation_fold": CONFIRMATION_FOLD,
        "fold1_previously_touched": False,
        "selection_or_retuning_on_fold1": False,
        "frozen_mode": FROZEN_MODE,
        "frozen_weight": FROZEN_WEIGHT,
        "fold_source": FOLD_PATH,
    },
    "arm": ARM,
    "base": base_metrics,
    "candidate": candidate_metrics,
    "blend": blend_metrics,
    "reversed_delta_control": reverse_metrics,
    "raw_signed_error_rho": rho,
    "gain": gain,
    "far_gain": far_gain,
    "reversed_alignment_loss": reverse_loss,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["next"] = (
    "PASS -> package the frozen 0.05 raw-GR dose after the exact 6.377 endpoint"
    if verdict["strict_pass"]
    else "FAIL -> Fold2 gain did not survive untouched Fold1; do not deploy"
)

output = frame[["id", "well", "fold", "tvt_true", "md_since", BASE_COLUMN, "gr_map"]].copy()
output["blend"] = blend_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold1_confirmation_oof.parquet", index=False, compression="zstd")
(OUT_DIR / "result.json").write_text(json.dumps(clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("\nPHYSICS-P3 FINAL VERDICT")
print(json.dumps(clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
