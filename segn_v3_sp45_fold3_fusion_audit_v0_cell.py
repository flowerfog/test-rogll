"""Zero-training Fold-3 fusion audit for SEGN-min V3 versus fold-clean sp45.

This cell answers one narrow question: can the weak V3 rollout serve as a
useful low-correlation endpoint when blended with sp45, as in the published
multi-engine workflow?

The audit uses exact row-aligned signed residuals and reports:
* standalone and fixed-weight blend exact SSE / pooled RMSE;
* continuous global optimum and a 0.01 fixed-weight scan;
* five deterministic held-well meta-fold weights and gains;
* raw, distance-bin-detrended, and distance-standardised correlations;
* four distance strata;
* hard and continuous per-well two-way oracle ceilings;
* worst-20% well overlap and error-mass concentration.

Important limitation: V3 selected its epoch on Fold 3.  Therefore every result
here is discovery-only.  A promising result authorises one frozen Fold-2
confirmation; it is not a deployable blend weight.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd


EXPERIMENT = "segn_v3_sp45_fold3_fusion_audit_v0"
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SEGN_ROOT = OOF_ROOT / "segn_min_refined_channels_fold2_v3"
SEGN_CANDIDATES = (
    SEGN_ROOT / "fold3_scheduled_arm_rollout_mean_rows.npz",
    SEGN_ROOT / "fold3_scheduled_arm_rollout_mode_rows.npz",
)
SEGN_PATH = next((path for path in SEGN_CANDIDATES if path.exists()), None)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    Path("/root/autodl-tmp/candidate_pool_full_merged_oof_honest_fle.csv"),
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
RECORD_DIR_CANDIDATES = (
    OOF_ROOT / "native_resolution_dualref_capacity_v0" / "records",
    OOF_ROOT / "native_resolution_dualref_onefold_v0" / "records",
)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PREFERENCE = ("sp45", "sp45_raw")
DISTANCE_EDGES = np.asarray([0.0, 500.0, 1000.0, 2000.0, np.inf])
DISTANCE_LABELS = ("0_500", "500_1000", "1000_2000", "gt2000")
META_FOLDS = 5
WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
MIN_ALIGNMENT = 0.995
MAX_TRUTH_GAP = 1e-3

# Discovery gates only.  Passing authorises a frozen Fold-2 confirmation.
CONDITIONAL_RHO_GATE = 0.45
MIN_GLOBAL_GAIN = 0.05
MIN_NESTED_GAIN = 0.02
MIN_POSITIVE_META_FOLDS = 4
MIN_MEDIAN_WEIGHT = 0.02


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
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


def _rmse(error):
    error = np.asarray(error, np.float64)
    return float(np.sqrt(np.mean(error**2)))


def _mae(error):
    return float(np.mean(np.abs(np.asarray(error, np.float64))))


def _rho(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return math.nan
    left = left[valid] - float(np.mean(left[valid]))
    right = right[valid] - float(np.mean(right[valid]))
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def _cosine(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def _optimal_weight(base_error, candidate_error):
    base_error = np.asarray(base_error, np.float64)
    candidate_error = np.asarray(candidate_error, np.float64)
    direction = candidate_error - base_error
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(base_error, direction) / denominator, 0.0, 1.0))


def _blend_error(base_error, candidate_error, weight):
    return (1.0 - float(weight)) * np.asarray(
        base_error, np.float64
    ) + float(weight) * np.asarray(candidate_error, np.float64)


def _meta_fold(well):
    digest = hashlib.sha256(str(well).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % META_FOLDS


def _distance_bin(distance):
    return np.clip(
        np.digitize(distance, DISTANCE_EDGES[1:-1], right=True),
        0,
        len(DISTANCE_LABELS) - 1,
    )


def _crossfit_distance_residual(error, distance, meta_fold, standardise=False):
    """Remove each engine's held-out distance-band error trend separately."""

    error = np.asarray(error, np.float64)
    bins = _distance_bin(distance)
    output = np.full(len(error), np.nan, np.float64)
    for fold in range(META_FOLDS):
        train_fold = meta_fold != fold
        held_fold = meta_fold == fold
        for band in range(len(DISTANCE_LABELS)):
            train = train_fold & (bins == band)
            held = held_fold & (bins == band)
            if not int(held.sum()):
                continue
            if int(train.sum()) < 32:
                train = train_fold
            centre = float(np.mean(error[train]))
            values = error[held] - centre
            if standardise:
                scale = float(np.sqrt(np.mean((error[train] - centre) ** 2)))
                values = values / max(scale, 1e-3)
            output[held] = values
    return output


def _metrics(error, wells):
    error = np.asarray(error, np.float64)
    wells = np.asarray(wells).astype(str)
    bywell_rmse = []
    for well in pd.unique(wells):
        values = error[wells == well]
        bywell_rmse.append(_rmse(values))
    bywell_rmse = np.asarray(bywell_rmse, np.float64)
    return {
        "rows": int(len(error)),
        "wells": int(len(bywell_rmse)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": _rmse(error),
        "mae": _mae(error),
        "bias": float(np.mean(error)),
        "mean_well_rmse": float(np.mean(bywell_rmse)),
        "median_well_rmse": float(np.median(bywell_rmse)),
        "p95_well_rmse": float(np.quantile(bywell_rmse, 0.95)),
        "max_well_rmse": float(np.max(bywell_rmse)),
    }


def _reconstruct_row_index(wells):
    wells = np.asarray(wells).astype(str)
    output = np.full(len(wells), -1, np.int32)
    for well in pd.unique(wells):
        positions = np.flatnonzero(wells == well)
        record_path = next(
            (
                directory / f"{well}.npz"
                for directory in RECORD_DIR_CANDIDATES
                if (directory / f"{well}.npz").exists()
            ),
            None,
        )
        if record_path is None:
            raise FileNotFoundError(f"Missing native record for {well}")
        with np.load(record_path, allow_pickle=False) as record:
            expected = record["eval_index"].astype(np.int32)
        if len(expected) != len(positions):
            raise RuntimeError(
                f"{well}: V3 rows={len(positions)} but record rows={len(expected)}"
            )
        output[positions] = expected
    if np.any(output < 0):
        raise RuntimeError("Failed to reconstruct all V3 row indices")
    return output


def _reconstruct_distance(wells, row_index):
    wells = np.asarray(wells).astype(str)
    row_index = np.asarray(row_index, np.int32)
    output = np.full(len(wells), np.nan, np.float64)
    for well in pd.unique(wells):
        positions = np.flatnonzero(wells == well)
        record_path = next(
            (
                directory / f"{well}.npz"
                for directory in RECORD_DIR_CANDIDATES
                if (directory / f"{well}.npz").exists()
            ),
            None,
        )
        if record_path is None:
            raise FileNotFoundError(f"Missing native record for {well}")
        with np.load(record_path, allow_pickle=False) as record:
            md = record["md"].astype(np.float64)
            anchor_index = int(record["anchor_index"][0])
        rows = row_index[positions]
        output[positions] = md[rows] - md[anchor_index]
    if not np.isfinite(output).all():
        raise RuntimeError("Failed to reconstruct all distances")
    return output


def _per_well_table(frame, base_error, candidate_error):
    rows = []
    for well, positions in frame.groupby("well", sort=False).groups.items():
        indices = np.asarray(list(positions), np.int64)
        base_values = base_error[indices]
        candidate_values = candidate_error[indices]
        weight = _optimal_weight(base_values, candidate_values)
        blend_values = _blend_error(base_values, candidate_values, weight)
        rows.append(
            {
                "well": well,
                "rows": len(indices),
                "base_rmse": _rmse(base_values),
                "candidate_rmse": _rmse(candidate_values),
                "base_bias": float(np.mean(base_values)),
                "candidate_bias": float(np.mean(candidate_values)),
                "candidate_wins": bool(
                    np.dot(candidate_values, candidate_values)
                    < np.dot(base_values, base_values)
                ),
                "oracle_weight": weight,
                "oracle_blend_rmse": _rmse(blend_values),
                "base_sse": float(np.dot(base_values, base_values)),
                "candidate_sse": float(np.dot(candidate_values, candidate_values)),
            }
        )
    return pd.DataFrame(rows)


def _worst20_summary(bywell):
    tail_count = max(1, int(math.ceil(0.20 * len(bywell))))
    base_tail = set(bywell.nlargest(tail_count, "base_rmse")["well"])
    candidate_tail = set(bywell.nlargest(tail_count, "candidate_rmse")["well"])
    overlap = base_tail & candidate_tail
    union = base_tail | candidate_tail
    base_mask = bywell["well"].isin(base_tail)
    candidate_mask = bywell["well"].isin(candidate_tail)
    return {
        "tail_wells_each": tail_count,
        "overlap_wells": len(overlap),
        "overlap_fraction_of_each_tail": len(overlap) / tail_count,
        "jaccard": len(overlap) / max(len(union), 1),
        "base_tail_sse_fraction": float(
            bywell.loc[base_mask, "base_sse"].sum() / bywell["base_sse"].sum()
        ),
        "candidate_tail_sse_fraction": float(
            bywell.loc[candidate_mask, "candidate_sse"].sum()
            / bywell["candidate_sse"].sum()
        ),
        "overlap_well_ids": sorted(overlap),
    }


started = time.time()
if SEGN_PATH is None:
    raise FileNotFoundError(
        "Cannot find V3 Fold-3 row predictions. Checked: "
        + repr([str(path) for path in SEGN_CANDIDATES])
    )
if POOL_PATH is None:
    raise FileNotFoundError(
        "Cannot find candidate pool. Checked: "
        + repr([str(path) for path in POOL_CANDIDATES])
    )

with np.load(SEGN_PATH, allow_pickle=False) as data:
    keys = set(data.files)
    truth_v3 = data["truth"].astype(np.float64)
    prediction_v3 = data["prediction"].astype(np.float64)
    well_v3 = data["well"].astype(str)
    row_index_v3 = (
        data["row_index"].astype(np.int32)
        if "row_index" in keys
        else _reconstruct_row_index(well_v3)
    )
    distance_v3 = (
        data["distance_past_anchor"].astype(np.float64)
        if "distance_past_anchor" in keys
        else _reconstruct_distance(well_v3, row_index_v3)
    )
if not (
    len(truth_v3)
    == len(prediction_v3)
    == len(well_v3)
    == len(row_index_v3)
    == len(distance_v3)
):
    raise RuntimeError("V3 row arrays have inconsistent lengths")

v3 = pd.DataFrame(
    {
        "well": well_v3,
        "row_idx": row_index_v3,
        "truth_v3": truth_v3,
        "segn_v3": prediction_v3,
        "distance": distance_v3,
    }
)
if v3.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("V3 output has duplicate well/row_idx keys")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
well_column = next(
    (column for column in ("well", "well_id", "WELL") if column in header), None
)
row_column = next(
    (column for column in ("row_idx", "_row_idx", "row_index") if column in header),
    None,
)
truth_column = next(
    (column for column in ("tvt_true", "TVT", "target") if column in header), None
)
id_column = "id" if "id" in header else None
base_column = next((column for column in BASE_PREFERENCE if column in header), None)
base_columns = [base_column] if base_column is not None else []
if truth_column is None or base_column is None or (row_column is None and id_column is None):
    raise RuntimeError(f"Cannot identify pool truth/keys/sp45 from columns: {header}")

identity = [column for column in (id_column, well_column, row_column) if column]
use_columns = list(dict.fromkeys(identity + [truth_column] + base_columns))
pool = pd.read_csv(
    POOL_PATH,
    usecols=use_columns,
    dtype={column: str for column in identity},
)
rename = {truth_column: "truth"}
if well_column:
    rename[well_column] = "well"
if row_column:
    rename[row_column] = "row_idx"
pool = pool.rename(columns=rename)
if "well" not in pool:
    pool["well"] = pool[id_column].astype(str).str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pd.to_numeric(
        pool[id_column].astype(str).str.rsplit("_", n=1).str[-1], errors="raise"
    ).astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pd.to_numeric(pool["row_idx"], errors="raise").astype(np.int32)
for column in ["truth"] + base_columns:
    pool[column] = pd.to_numeric(pool[column], errors="coerce")
if pool.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("Candidate pool has duplicate well/row_idx keys")

frame = v3.merge(pool, on=["well", "row_idx"], how="left", validate="one_to_one")
alignment = float(frame[base_columns[0]].notna().mean())
if alignment < MIN_ALIGNMENT:
    missing = frame.loc[frame[base_columns[0]].isna(), ["well", "row_idx"]].head(20)
    raise RuntimeError(
        f"V3/pool alignment {alignment:.6%} < {MIN_ALIGNMENT:.3%}\n{missing}"
    )
frame = frame.dropna(subset=["truth"] + base_columns).reset_index(drop=True)
truth_gap = float(np.max(np.abs(frame["truth_v3"] - frame["truth"])))
if not np.isfinite(truth_gap) or truth_gap > MAX_TRUTH_GAP:
    raise RuntimeError(f"V3/pool truth mismatch: max={truth_gap:.9f}")
frame["meta_fold"] = frame["well"].map(_meta_fold).astype(np.int8)

print(
    "V3-SP45 FUSION CONFIG",
    {
        "v3": str(SEGN_PATH),
        "pool": str(POOL_PATH),
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "alignment": alignment,
        "truth_gap": truth_gap,
        "bases": base_columns,
    },
    flush=True,
)

all_summaries = []
all_scans = []
all_meta = []
all_strata = []
all_oracles = []
all_worst20 = {}

truth = frame["truth"].to_numpy(np.float64)
candidate = frame["segn_v3"].to_numpy(np.float64)
candidate_error = candidate - truth
wells = frame["well"].to_numpy(str)
distance = frame["distance"].to_numpy(np.float64)
meta_fold = frame["meta_fold"].to_numpy(np.int8)

for base_name in base_columns:
    base = frame[base_name].to_numpy(np.float64)
    base_error = base - truth
    if not np.isfinite(base_error).all():
        continue

    base_metrics = _metrics(base_error, wells)
    candidate_metrics = _metrics(candidate_error, wells)
    if not (0.1 < base_metrics["rmse"] < 20.0):
        continue

    global_weight = _optimal_weight(base_error, candidate_error)
    global_blend_error = _blend_error(base_error, candidate_error, global_weight)
    global_blend_metrics = _metrics(global_blend_error, wells)

    for weight in WEIGHT_GRID:
        error = _blend_error(base_error, candidate_error, weight)
        metrics = _metrics(error, wells)
        all_scans.append(
            {
                "base": base_name,
                "weight_segn": float(weight),
                **metrics,
                "gain_vs_base": base_metrics["rmse"] - metrics["rmse"],
            }
        )

    nested_error = np.full(len(frame), np.nan, np.float64)
    meta_rows = []
    for fold in range(META_FOLDS):
        train = meta_fold != fold
        held = meta_fold == fold
        selected_weight = _optimal_weight(base_error[train], candidate_error[train])
        held_error = _blend_error(
            base_error[held], candidate_error[held], selected_weight
        )
        held_base_rmse = _rmse(base_error[held])
        held_blend_rmse = _rmse(held_error)
        nested_error[held] = held_error
        row = {
            "base": base_name,
            "meta_fold": fold,
            "selected_weight_segn": selected_weight,
            "held_rows": int(held.sum()),
            "held_wells": int(frame.loc[held, "well"].nunique()),
            "held_base_rmse": held_base_rmse,
            "held_blend_rmse": held_blend_rmse,
            "held_gain": held_base_rmse - held_blend_rmse,
        }
        meta_rows.append(row)
        all_meta.append(row)
    nested_metrics = _metrics(nested_error, wells)

    detrended_base = _crossfit_distance_residual(
        base_error, distance, meta_fold, standardise=False
    )
    detrended_candidate = _crossfit_distance_residual(
        candidate_error, distance, meta_fold, standardise=False
    )
    standard_base = _crossfit_distance_residual(
        base_error, distance, meta_fold, standardise=True
    )
    standard_candidate = _crossfit_distance_residual(
        candidate_error, distance, meta_fold, standardise=True
    )

    bins = _distance_bin(distance)
    for band_index, band_name in enumerate(DISTANCE_LABELS):
        mask = bins == band_index
        band_weight = _optimal_weight(base_error[mask], candidate_error[mask])
        band_blend = _blend_error(
            base_error[mask], candidate_error[mask], band_weight
        )
        all_strata.append(
            {
                "base": base_name,
                "distance_band": band_name,
                "rows": int(mask.sum()),
                "wells": int(frame.loc[mask, "well"].nunique()),
                "base_rmse": _rmse(base_error[mask]),
                "candidate_rmse": _rmse(candidate_error[mask]),
                "signed_error_rho": _rho(base_error[mask], candidate_error[mask]),
                "error_cosine": _cosine(base_error[mask], candidate_error[mask]),
                "oracle_weight_segn": band_weight,
                "oracle_blend_rmse": _rmse(band_blend),
                "oracle_gain": _rmse(base_error[mask]) - _rmse(band_blend),
            }
        )

    bywell = _per_well_table(frame, base_error, candidate_error)
    bywell.to_csv(OUT_DIR / f"{base_name}_per_well.csv", index=False)
    worst20 = _worst20_summary(bywell)
    all_worst20[base_name] = worst20

    continuous_oracle_error = np.full(len(frame), np.nan, np.float64)
    hard_oracle_error = np.full(len(frame), np.nan, np.float64)
    for well, positions in frame.groupby("well", sort=False).groups.items():
        indices = np.asarray(list(positions), np.int64)
        weight = float(bywell.loc[bywell["well"].eq(well), "oracle_weight"].iloc[0])
        continuous_oracle_error[indices] = _blend_error(
            base_error[indices], candidate_error[indices], weight
        )
        use_candidate = bool(
            bywell.loc[bywell["well"].eq(well), "candidate_wins"].iloc[0]
        )
        hard_oracle_error[indices] = (
            candidate_error[indices] if use_candidate else base_error[indices]
        )
    continuous_oracle_metrics = _metrics(continuous_oracle_error, wells)
    hard_oracle_metrics = _metrics(hard_oracle_error, wells)
    oracle_row = {
        "base": base_name,
        "continuous_perwell_oracle": continuous_oracle_metrics,
        "hard_perwell_oracle": hard_oracle_metrics,
        "oracle_weight_mean": float(bywell["oracle_weight"].mean()),
        "oracle_weight_median": float(bywell["oracle_weight"].median()),
        "oracle_weight_zero_fraction": float(
            np.mean(bywell["oracle_weight"].to_numpy(float) <= 1e-8)
        ),
        "candidate_wins_well_fraction": float(bywell["candidate_wins"].mean()),
    }
    all_oracles.append(oracle_row)

    weights = np.asarray(
        [row["selected_weight_segn"] for row in meta_rows], np.float64
    )
    gains = np.asarray([row["held_gain"] for row in meta_rows], np.float64)
    summary = {
        "base": base_name,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "base_rmse": base_metrics["rmse"],
        "candidate_rmse": candidate_metrics["rmse"],
        "raw_signed_error_rho": _rho(base_error, candidate_error),
        "raw_error_cosine": _cosine(base_error, candidate_error),
        "distance_detrended_rho": _rho(detrended_base, detrended_candidate),
        "distance_standardised_rho": _rho(standard_base, standard_candidate),
        "continuous_weight_segn": global_weight,
        "continuous_blend_rmse": global_blend_metrics["rmse"],
        "continuous_gain": base_metrics["rmse"] - global_blend_metrics["rmse"],
        "nested_blend_rmse": nested_metrics["rmse"],
        "nested_gain": base_metrics["rmse"] - nested_metrics["rmse"],
        "meta_weight_min": float(weights.min()),
        "meta_weight_median": float(np.median(weights)),
        "meta_weight_max": float(weights.max()),
        "positive_meta_folds": int(np.sum(gains > 0.0)),
        "well_rmse_rho": _rho(bywell["base_rmse"], bywell["candidate_rmse"]),
        "well_bias_rho": _rho(bywell["base_bias"], bywell["candidate_bias"]),
        "worst20_overlap_fraction": worst20["overlap_fraction_of_each_tail"],
        "continuous_perwell_oracle_rmse": continuous_oracle_metrics["rmse"],
        "hard_perwell_oracle_rmse": hard_oracle_metrics["rmse"],
    }
    summary["discovery_pass"] = bool(
        summary["distance_standardised_rho"] <= CONDITIONAL_RHO_GATE
        and summary["continuous_gain"] >= MIN_GLOBAL_GAIN
        and summary["nested_gain"] >= MIN_NESTED_GAIN
        and summary["positive_meta_folds"] >= MIN_POSITIVE_META_FOLDS
        and summary["meta_weight_median"] >= MIN_MEDIAN_WEIGHT
    )
    all_summaries.append(summary)

if not all_summaries:
    raise RuntimeError("No valid sp45 endpoint survived alignment and RMSE guards")

summary_frame = pd.DataFrame(all_summaries).sort_values(
    ["nested_blend_rmse", "continuous_blend_rmse"], ignore_index=True
)
scan_frame = pd.DataFrame(all_scans)
meta_frame = pd.DataFrame(all_meta)
strata_frame = pd.DataFrame(all_strata)
summary_frame.to_csv(OUT_DIR / "summary.csv", index=False)
scan_frame.to_csv(OUT_DIR / "fixed_weight_scan.csv", index=False)
meta_frame.to_csv(OUT_DIR / "meta_fold_weights.csv", index=False)
strata_frame.to_csv(OUT_DIR / "distance_strata.csv", index=False)
_write_json(OUT_DIR / "perwell_oracles.json", all_oracles)
_write_json(OUT_DIR / "worst20_overlap.json", all_worst20)

best = summary_frame.iloc[0].to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold3_v3_sp45_fusion_discovery",
    "source": str(SEGN_PATH),
    "pool": str(POOL_PATH),
    "protocol_warning": (
        "V3 epoch selection used Fold 3; meta-fold blending is an exploratory "
        "stability diagnostic, not an untouched confirmation"
    ),
    "alignment": {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "rate": alignment,
        "maximum_truth_gap": truth_gap,
    },
    "best_pair": best,
    "worst20": all_worst20[str(best["base"])],
    "perwell_oracle": next(
        item for item in all_oracles if item["base"] == str(best["base"])
    ),
    "gates": {
        "distance_standardised_rho_le": CONDITIONAL_RHO_GATE,
        "continuous_gain_ge": MIN_GLOBAL_GAIN,
        "nested_gain_ge": MIN_NESTED_GAIN,
        "positive_meta_folds_ge": MIN_POSITIVE_META_FOLDS,
        "median_meta_weight_ge": MIN_MEDIAN_WEIGHT,
    },
    "strict_pass": bool(best["discovery_pass"]),
    "fold2_touched": False,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS DISCOVERY -> freeze V3 epoch/recipe, refit Folds 3/4/5, confirm once on Fold 2"
        if bool(best["discovery_pass"])
        else "FAIL -> V3 is not a useful sp45 fusion endpoint; continue V5 mechanism work"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV3-SP45 SUMMARY")
print(summary_frame.to_string(index=False), flush=True)
print("\nV3-SP45 DISTANCE STRATA")
print(strata_frame.to_string(index=False), flush=True)
print("\nV3-SP45 META-FOLD WEIGHTS")
print(meta_frame.to_string(index=False), flush=True)
print("\nV3-SP45 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
