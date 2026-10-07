"""Audit the deployed HN7 dose against the complete V66 mainline lineage.

This is a frozen, read-only endpoint comparison. It does not retune the HN
model or the deployed 0.275 blend weight.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
MAINLINE_PATH = (
    ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
HN_PATH = (
    ROOT
    / "realistic_pairwise_mainline_fusion_hn4"
    / "hn3_row_oof.parquet"
)
OUT_DIR = ROOT / "realistic_pairwise_v66_lineage_audit_hn9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_WEIGHT = 0.275
EXPECTED_MAINLINE_ROWS = 3_783_989
EXPECTED_MAINLINE_WELLS = 773
DISTANCE_BINS = np.asarray([0.0, 500.0, 1000.0, 2000.0, np.inf])


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    well_rmse = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    return {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _correlation(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return np.nan
    if np.std(left[finite]) <= 1e-12 or np.std(right[finite]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _distance_detrend(error, distance):
    error = np.asarray(error, np.float64)
    distance = np.asarray(distance, np.float64)
    output = np.empty_like(error)
    labels = np.digitize(distance, DISTANCE_BINS[1:-1], right=True)
    for label in range(len(DISTANCE_BINS) - 1):
        mask = labels == label
        output[mask] = error[mask] - np.mean(error[mask])
    return output


started = time.time()
for path in (MAINLINE_PATH, HN_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)

mainline_columns = [
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "incumbent_6p377_replay",
]
mainline = pd.read_parquet(MAINLINE_PATH, columns=mainline_columns)
mainline["id"] = mainline["id"].astype(str)
mainline["well"] = mainline["well"].astype(str)
if len(mainline) != EXPECTED_MAINLINE_ROWS or mainline["well"].nunique() != EXPECTED_MAINLINE_WELLS:
    raise RuntimeError(
        f"V66 is not complete: rows={len(mainline)} wells={mainline['well'].nunique()}"
    )
if mainline["id"].duplicated().any():
    raise RuntimeError("V66 contains duplicate ids")

hn = pd.read_parquet(HN_PATH)
required_hn = {"id", "well", "row_idx", "tvt_true", "md_since", "hn3_pairwise"}
missing_hn = sorted(required_hn.difference(hn.columns))
if missing_hn:
    raise RuntimeError(f"HN endpoint misses columns: {missing_hn}")
hn = hn[list(required_hn)].rename(
    columns={"tvt_true": "hn_truth", "md_since": "hn_md_since"}
)
hn["id"] = hn["id"].astype(str)
hn["well"] = hn["well"].astype(str)
if hn["id"].duplicated().any():
    raise RuntimeError("HN endpoint contains duplicate ids")

aligned = mainline.merge(
    hn,
    on=["id", "well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
truth_gap = np.abs(
    aligned["tvt_true"].to_numpy(np.float64)
    - aligned["hn_truth"].to_numpy(np.float64)
)
distance_gap = np.abs(
    aligned["md_since"].to_numpy(np.float64)
    - aligned["hn_md_since"].to_numpy(np.float64)
)
if not len(aligned) or truth_gap.max() > 0.01 or distance_gap.max() > 0.01:
    raise RuntimeError(
        f"V66/HN alignment failed: truth={truth_gap.max()} distance={distance_gap.max()}"
    )
aligned = aligned.drop(columns=["hn_truth", "hn_md_since"])

fold_wells = (
    aligned[["well", "fold"]]
    .drop_duplicates()
    .groupby("fold")["well"]
    .nunique()
    .astype(int)
    .to_dict()
)
coverage = {
    "mainline_rows": int(len(mainline)),
    "mainline_wells": int(mainline["well"].nunique()),
    "hn_rows": int(len(hn)),
    "hn_wells": int(hn["well"].nunique()),
    "aligned_rows": int(len(aligned)),
    "aligned_wells": int(aligned["well"].nunique()),
    "canonical_fold_wells": fold_wells,
    "truth_gap_max": float(truth_gap.max()),
    "distance_gap_max": float(distance_gap.max()),
}
print("HN9 ALIGNMENT", coverage, flush=True)

truth = aligned["tvt_true"].to_numpy(np.float64)
base = aligned["incumbent_6p377_replay"].to_numpy(np.float64)
candidate = aligned["hn3_pairwise"].to_numpy(np.float64)
blend = base + FROZEN_WEIGHT * (candidate - base)
distance = aligned["md_since"].to_numpy(np.float64)
base_error = base - truth
candidate_error = candidate - truth

base_metrics = _metrics(aligned, base)
candidate_metrics = _metrics(aligned, candidate)
blend_metrics = _metrics(aligned, blend)

fold_rows = []
for fold in sorted(aligned["fold"].unique()):
    mask = aligned["fold"].to_numpy(int) == int(fold)
    local = aligned.loc[mask].reset_index(drop=True)
    local_base = _metrics(local, base[mask])
    local_blend = _metrics(local, blend[mask])
    fold_rows.append(
        {
            "fold": int(fold),
            "rows": int(mask.sum()),
            "wells": int(local["well"].nunique()),
            "base_rmse": local_base["rmse"],
            "rmse": local_blend["rmse"],
            "gain": local_base["rmse"] - local_blend["rmse"],
            "base_far_rmse": local_base["far_gt2000_rmse"],
            "far_rmse": local_blend["far_gt2000_rmse"],
            "far_gain": local_base["far_gt2000_rmse"] - local_blend["far_gt2000_rmse"],
            "p95_change": local_blend["p95_well_rmse"] - local_base["p95_well_rmse"],
        }
    )
fold_frame = pd.DataFrame(fold_rows)
fold_frame.to_csv(OUT_DIR / "frozen_weight_folds.csv", index=False)

distance_rho = {}
for lower, upper in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
    mask = (distance > lower) & (distance <= upper)
    distance_rho[f"{lower:g}_{upper:g}"] = _correlation(
        base_error[mask], candidate_error[mask]
    )

correlation = {
    "raw": _correlation(base_error, candidate_error),
    "distance_detrended": _correlation(
        _distance_detrend(base_error, distance),
        _distance_detrend(candidate_error, distance),
    ),
    "distance_strata": distance_rho,
}

checks = {
    "v66_complete_773_wells": bool(mainline["well"].nunique() == 773),
    "aligned_hn_wells_ge_600": bool(aligned["well"].nunique() >= 600),
    "each_canonical_fold_ge_110_wells": bool(min(fold_wells.values()) >= 110),
    "overall_gain_ge_0p03": bool(base_metrics["rmse"] - blend_metrics["rmse"] >= 0.03),
    "far_gain_ge_0p05": bool(
        base_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"] >= 0.05
    ),
    "positive_folds_ge_4": bool((fold_frame["gain"] > 0).sum() >= 4),
    "minimum_fold_gain_ge_minus0p05": bool(fold_frame["gain"].min() >= -0.05),
    "maximum_p95_change_le_0p20": bool(fold_frame["p95_change"].max() <= 0.20),
}
verdict = {
    "experiment": "realistic_pairwise_v66_lineage_audit_hn9",
    "stage": "frozen_hn7_weight_against_complete_v66_canonical_lineage",
    "selection_or_tuning_count": 0,
    "sources": {"mainline": MAINLINE_PATH, "hn_endpoint": HN_PATH},
    "frozen_weight": FROZEN_WEIGHT,
    "coverage": coverage,
    "error_correlation": correlation,
    "base": base_metrics,
    "candidate": candidate_metrics,
    "blend": blend_metrics,
    "gain": base_metrics["rmse"] - blend_metrics["rmse"],
    "far_gain": base_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"],
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> HN7 local gain survives complete V66 lineage; investigate hidden shift"
        if all(checks.values())
        else "FAIL -> old HN gain was a partial-lineage artifact; close HN7"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
aligned[[
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "incumbent_6p377_replay", "hn3_pairwise",
]].assign(hn7_w0275=blend.astype(np.float32)).to_parquet(
    OUT_DIR / "aligned_frozen_oof.parquet", index=False, compression="zstd"
)

print("\nHN9 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nHN9 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
