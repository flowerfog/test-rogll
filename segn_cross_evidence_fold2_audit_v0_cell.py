"""Cross-evidence audit for the Fold-2 SEGN rollout endpoint.

This is a zero-training audit.  It aligns SEGN predictions with fold-clean
candidate-pool endpoints and measures complementarity without confusing
mechanism diagnostics with deployable blend gains.

Primary fixed-blend quantities use raw signed residual covariance and exact
SSE.  Distance-detrended and distance-standardised correlations are reported
only as conditional-independence diagnostics.  Five deterministic well-group
meta-folds select a blend weight on four groups and apply it to the held group.
"""

from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd


EXPERIMENT = "segn_cross_evidence_fold2_audit_v0"
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SEGN_ROOT = OOF_ROOT / "segn_chunked_costmap_fold2_gateway_v0"
SEGN_PATH = SEGN_ROOT / "fold2_rollout_rows.npz"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
RECORD_DIR_CANDIDATES = (
    OOF_ROOT / "native_resolution_dualref_capacity_v0" / "records",
    OOF_ROOT / "native_resolution_dualref_onefold_v0" / "records",
)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PREFERENCE = (
    "strong_bias",
    "strong",
    "cp12",
    "sp45",
    "sp45_like",
    "fleongg",
    "full_q50",
    "likpf_scale_12",
    "pf_ancc",
)
DISTANCE_EDGES = np.asarray([0.0, 500.0, 1000.0, 2000.0, np.inf])
DISTANCE_LABELS = ("0_500", "500_1000", "1000_2000", "gt2000")
META_FOLDS = 5
MIN_COVERAGE = 0.995
MAX_BASE_RMSE = 20.0

ENDPOINT_GATE_RMSE = 10.5
RAW_RHO_DIAGNOSTIC = 0.55
CONDITIONAL_RHO_GATE = 0.45
MIN_WEIGHT = 0.10


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def _rmse_from_error(error):
    error = np.asarray(error, np.float64)
    return float(np.sqrt(np.mean(error * error)))


def _rho(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return math.nan
    left = left[valid] - np.mean(left[valid])
    right = right[valid] - np.mean(right[valid])
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def _cosine(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def _weight(base_error, candidate_error):
    base_error = np.asarray(base_error, np.float64)
    direction = np.asarray(candidate_error, np.float64) - base_error
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(base_error, direction) / denominator, 0.0, 1.0))


def _blend_error(base_error, candidate_error, weight):
    return (1.0 - float(weight)) * np.asarray(base_error) + float(weight) * np.asarray(candidate_error)


def _meta_fold(well):
    digest = hashlib.sha256(str(well).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % META_FOLDS


def _distance_bin(distance):
    return np.clip(np.digitize(distance, DISTANCE_EDGES[1:-1], right=True), 0, len(DISTANCE_LABELS) - 1)


def _crossfit_distance_residual(error, distance, meta_fold, standardise):
    error = np.asarray(error, np.float64)
    bins = _distance_bin(distance)
    output = np.full(len(error), np.nan, np.float64)
    for fold in range(META_FOLDS):
        train = meta_fold != fold
        valid = meta_fold == fold
        for distance_bin in range(len(DISTANCE_LABELS)):
            train_rows = train & (bins == distance_bin)
            valid_rows = valid & (bins == distance_bin)
            if not int(valid_rows.sum()):
                continue
            if int(train_rows.sum()) < 16:
                train_rows = train
            centre = float(np.mean(error[train_rows]))
            residual = error[valid_rows] - centre
            if standardise:
                scale = float(np.sqrt(np.mean((error[train_rows] - centre) ** 2)))
                residual = residual / max(scale, 1e-3)
            output[valid_rows] = residual
    return output


def _reconstruct_row_index(wells):
    wells = np.asarray(wells).astype(str)
    row_index = np.full(len(wells), -1, np.int32)
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
                f"{well}: SEGN rows={len(positions)} but native eval rows={len(expected)}"
            )
        row_index[positions] = expected
    if np.any(row_index < 0):
        raise RuntimeError("Failed to reconstruct all SEGN row indices")
    return row_index


def _per_well_diagnostics(frame, base_error, candidate_error):
    work = pd.DataFrame(
        {
            "well": frame["well"].to_numpy(),
            "base_error": base_error,
            "candidate_error": candidate_error,
        }
    )
    rows = []
    for well, group in work.groupby("well", sort=False):
        rows.append(
            {
                "well": well,
                "rows": len(group),
                "base_bias": float(group["base_error"].mean()),
                "candidate_bias": float(group["candidate_error"].mean()),
                "base_rmse": _rmse_from_error(group["base_error"]),
                "candidate_rmse": _rmse_from_error(group["candidate_error"]),
            }
        )
    bywell = pd.DataFrame(rows)
    return {
        "well_bias_rho": _rho(bywell["base_bias"], bywell["candidate_bias"]),
        "well_rmse_rho": _rho(bywell["base_rmse"], bywell["candidate_rmse"]),
    }, bywell


def _pair_audit(frame, base_name):
    truth = frame["truth"].to_numpy(np.float64)
    base = frame[base_name].to_numpy(np.float64)
    candidate = frame["segn"].to_numpy(np.float64)
    base_error = base - truth
    candidate_error = candidate - truth
    distance = frame["distance"].to_numpy(np.float64)
    meta_fold = frame["meta_fold"].to_numpy(np.int32)

    raw_weight = _weight(base_error, candidate_error)
    raw_blend = _blend_error(base_error, candidate_error, raw_weight)
    detrended_base = _crossfit_distance_residual(base_error, distance, meta_fold, False)
    detrended_candidate = _crossfit_distance_residual(candidate_error, distance, meta_fold, False)
    standard_base = _crossfit_distance_residual(base_error, distance, meta_fold, True)
    standard_candidate = _crossfit_distance_residual(candidate_error, distance, meta_fold, True)

    nested_error = np.full(len(frame), np.nan, np.float64)
    nested_rows = []
    for fold in range(META_FOLDS):
        train = meta_fold != fold
        valid = meta_fold == fold
        selected_weight = _weight(base_error[train], candidate_error[train])
        held_base = _rmse_from_error(base_error[valid])
        held_blend_error = _blend_error(base_error[valid], candidate_error[valid], selected_weight)
        held_blend = _rmse_from_error(held_blend_error)
        nested_error[valid] = held_blend_error
        nested_rows.append(
            {
                "base": base_name,
                "meta_fold": fold,
                "train_selected_weight": selected_weight,
                "held_rows": int(valid.sum()),
                "held_wells": int(frame.loc[valid, "well"].nunique()),
                "held_base_rmse": held_base,
                "held_blend_rmse": held_blend,
                "held_gain": held_base - held_blend,
            }
        )

    stratum_rows = []
    bins = _distance_bin(distance)
    for index, label in enumerate(DISTANCE_LABELS):
        mask = bins == index
        weight = _weight(base_error[mask], candidate_error[mask])
        stratum_rows.append(
            {
                "base": base_name,
                "stratum": label,
                "rows": int(mask.sum()),
                "base_rmse": _rmse_from_error(base_error[mask]),
                "candidate_rmse": _rmse_from_error(candidate_error[mask]),
                "signed_error_rho": _rho(base_error[mask], candidate_error[mask]),
                "error_cosine": _cosine(base_error[mask], candidate_error[mask]),
                "oracle_fixed_weight": weight,
                "oracle_fixed_blend_rmse": _rmse_from_error(
                    _blend_error(base_error[mask], candidate_error[mask], weight)
                ),
            }
        )

    per_well, bywell = _per_well_diagnostics(frame, base_error, candidate_error)
    nested = pd.DataFrame(nested_rows)
    weights = nested["train_selected_weight"].to_numpy(float)
    gains = nested["held_gain"].to_numpy(float)
    result = {
        "base": base_name,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "base_rmse": _rmse_from_error(base_error),
        "candidate_rmse": _rmse_from_error(candidate_error),
        "raw_signed_error_rho": _rho(base_error, candidate_error),
        "raw_error_cosine": _cosine(base_error, candidate_error),
        "crossfit_distance_detrended_rho": _rho(detrended_base, detrended_candidate),
        "crossfit_distance_standardised_rho": _rho(standard_base, standard_candidate),
        "full_fold_oracle_weight": raw_weight,
        "full_fold_oracle_blend_rmse": _rmse_from_error(raw_blend),
        "nested_blend_rmse": _rmse_from_error(nested_error),
        "nested_gain": _rmse_from_error(base_error) - _rmse_from_error(nested_error),
        "nested_weight_min": float(np.min(weights)),
        "nested_weight_median": float(np.median(weights)),
        "nested_weight_max": float(np.max(weights)),
        "nested_weights_ge_0p1": int(np.sum(weights >= MIN_WEIGHT)),
        "positive_meta_folds": int(np.sum(gains > 0.0)),
        **per_well,
    }
    result["minimum_stability_pass"] = bool(
        np.all(weights >= 0.0)
        and np.median(weights) >= MIN_WEIGHT
        and np.sum(weights >= MIN_WEIGHT) >= 4
        and np.sum(gains > 0.0) >= 4
        and result["nested_gain"] > 0.0
    )
    result["strong_stability_pass"] = bool(
        np.all(weights >= MIN_WEIGHT)
        and np.all(gains > 0.0)
        and result["nested_gain"] > 0.0
    )
    result["conditional_independence_pass"] = bool(
        result["crossfit_distance_standardised_rho"] <= CONDITIONAL_RHO_GATE
    )
    return result, nested_rows, stratum_rows, bywell


started = time.time()
if not SEGN_PATH.exists():
    raise FileNotFoundError(
        f"Missing {SEGN_PATH}. Finish segn_chunked_costmap_fold2_gateway_v0 first."
    )
if not POOL_PATH.exists():
    raise FileNotFoundError(POOL_PATH)

with np.load(SEGN_PATH, allow_pickle=False) as data:
    segn_truth = data["truth"].astype(np.float64)
    segn_prediction = data["prediction"].astype(np.float64)
    segn_well = data["well"].astype(str)
if not (len(segn_truth) == len(segn_prediction) == len(segn_well)):
    raise RuntimeError("SEGN row arrays have inconsistent lengths")
row_index = _reconstruct_row_index(segn_well)
segn = pd.DataFrame(
    {
        "well": segn_well,
        "row_idx": row_index,
        "segn_truth": segn_truth,
        "segn": segn_prediction,
    }
)
if segn.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("SEGN output has duplicate well/row_idx keys")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
well_column = next((column for column in ("well", "well_id", "WELL") if column in header), None)
row_column = next((column for column in ("row_idx", "_row_idx", "row_index") if column in header), None)
truth_column = next((column for column in ("tvt_true", "TVT", "target") if column in header), None)
id_column = "id" if "id" in header else None
if truth_column is None or (row_column is None and id_column is None):
    raise RuntimeError(f"Cannot identify pool keys/truth from columns: {header}")
base_columns = [column for column in BASE_PREFERENCE if column in header]
if not base_columns:
    raise RuntimeError(f"No preferred base endpoint found in candidate pool: {BASE_PREFERENCE}")

identity_columns = [column for column in (id_column, well_column, row_column) if column is not None]
use_columns = list(dict.fromkeys(identity_columns + [truth_column] + base_columns))
pool = pd.read_csv(POOL_PATH, usecols=use_columns, dtype={column: str for column in identity_columns})
rename = {truth_column: "truth"}
if well_column is not None:
    rename[well_column] = "well"
if row_column is not None:
    rename[row_column] = "row_idx"
pool = pool.rename(columns=rename)
if "well" not in pool:
    pool["well"] = pool[id_column].astype(str).str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pd.to_numeric(
        pool[id_column].astype(str).str.rsplit("_", n=1).str[-1],
        errors="raise",
    ).astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pd.to_numeric(pool["row_idx"], errors="raise").astype(np.int32)
if pool.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("Candidate pool has duplicate well/row_idx keys")
frame = segn.merge(pool, on=["well", "row_idx"], how="left", validate="one_to_one")
if len(frame) != len(segn):
    raise RuntimeError(f"SEGN/pool row mismatch: {len(segn)}/{len(frame)}")
truth_gap = np.nanmax(np.abs(frame["segn_truth"].to_numpy(float) - frame["truth"].to_numpy(float)))
if not np.isfinite(truth_gap) or truth_gap > 1e-3:
    raise RuntimeError(f"SEGN/pool truth mismatch: max={truth_gap}")
frame["distance"] = np.nan
for well, positions in frame.groupby("well", sort=False).groups.items():
    rows = frame.loc[positions, "row_idx"].to_numpy(np.int32)
    record_path = next(
        (directory / f"{well}.npz" for directory in RECORD_DIR_CANDIDATES if (directory / f"{well}.npz").exists()),
        None,
    )
    with np.load(record_path, allow_pickle=False) as record:
        md = record["md"].astype(np.float64)
        anchor_index = int(record["anchor_index"][0])
    frame.loc[positions, "distance"] = md[rows] - md[anchor_index]
frame["meta_fold"] = frame["well"].map(_meta_fold).astype(np.int8)

print(
    "CROSS-EVIDENCE CONFIG",
    {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "bases": base_columns,
        "segn_rmse": _rmse_from_error(frame["segn"] - frame["truth"]),
    },
    flush=True,
)

summary_rows = []
nested_rows = []
stratum_rows = []
for base in base_columns:
    values = pd.to_numeric(frame[base], errors="coerce").to_numpy(np.float64)
    coverage = float(np.mean(np.isfinite(values)))
    if coverage < MIN_COVERAGE:
        continue
    if not np.isfinite(values).all():
        values = np.where(np.isfinite(values), values, np.nanmedian(values))
        frame[base] = values
    base_rmse = _rmse_from_error(values - frame["truth"].to_numpy(float))
    if not (0.05 < base_rmse <= MAX_BASE_RMSE):
        continue
    result, local_nested, local_strata, bywell = _pair_audit(frame, base)
    summary_rows.append(result)
    nested_rows.extend(local_nested)
    stratum_rows.extend(local_strata)
    bywell.to_csv(OUT_DIR / f"{base}_bywell.csv", index=False)

if not summary_rows:
    raise RuntimeError("No base endpoint survived coverage/RMSE validation")
summary = pd.DataFrame(summary_rows).sort_values(
    ["nested_blend_rmse", "base_rmse"], ignore_index=True
)
nested = pd.DataFrame(nested_rows)
strata = pd.DataFrame(stratum_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
nested.to_csv(OUT_DIR / "meta_fold_weights.csv", index=False)
strata.to_csv(OUT_DIR / "distance_strata.csv", index=False)
print("CROSS-EVIDENCE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("DISTANCE STRATA")
print(strata.to_string(index=False), flush=True)
print("META-FOLD WEIGHTS")
print(nested.to_string(index=False), flush=True)

best = summary.iloc[0].to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_cross_evidence_conditional_independence_audit",
    "source": str(SEGN_PATH),
    "pool": str(POOL_PATH),
    "important_interpretation": {
        "fixed_global_blend": "raw signed residual covariance and exact SSE are primary",
        "distance_standardised_rho": "mechanism diagnostic for conditional blending, not a replacement for raw covariance",
        "meta_fold_weights": "selected on four deterministic well groups and applied to the held group",
        "outer_fold_limit": "this is Fold2 only; five true outer folds are required before deployment",
    },
    "best_pair": best,
    "gates": {
        "candidate_rmse_le": ENDPOINT_GATE_RMSE,
        "raw_rho_diagnostic_le": RAW_RHO_DIAGNOSTIC,
        "crossfit_distance_standardised_rho_le": CONDITIONAL_RHO_GATE,
        "minimum_weight": MIN_WEIGHT,
        "minimum_stability": "median weight >=0.1, >=4/5 weights >=0.1, >=4/5 held gains positive",
        "strong_stability": "all five weights >=0.1 and all five held gains positive",
    },
    "endpoint_accuracy_pass": bool(best["candidate_rmse"] <= ENDPOINT_GATE_RMSE),
    "raw_independence_diagnostic_pass": bool(best["raw_signed_error_rho"] <= RAW_RHO_DIAGNOSTIC),
    "conditional_independence_pass": bool(best["conditional_independence_pass"]),
    "minimum_stability_pass": bool(best["minimum_stability_pass"]),
    "strong_stability_pass": bool(best["strong_stability_pass"]),
    "strict_pass": bool(
        best["candidate_rmse"] <= ENDPOINT_GATE_RMSE
        and best["nested_gain"] > 0.0
        and best["minimum_stability_pass"]
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
if verdict["strict_pass"]:
    verdict["next"] = "PASS FOLD2 -> produce SEGN OOF for all five outer folds before any deployment blend"
elif best["conditional_independence_pass"] and not best["minimum_stability_pass"]:
    verdict["next"] = "CONDITIONAL SIGNAL ONLY -> no deployment; inspect distance-conditioned weights"
else:
    verdict["next"] = "FAIL -> SEGN is not a stable complementary endpoint on Fold2"
_write_json(OUT_DIR / "verdict.json", verdict)
print("CROSS-EVIDENCE FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
