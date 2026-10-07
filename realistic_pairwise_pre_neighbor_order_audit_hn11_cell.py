"""Frozen OOF audit of applying HN before rather than after Neighbor.

This cell changes only operator order. It aligns the complete V66 five-fold
lineage with the HN3 cross-fit endpoint, keeps the deployed HN weight at 0.275,
and replays the frozen Neighbor transform against the HN-modified parent.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STARTED = time.time()
ROOT = Path("/root/autodl-tmp/oof_exports")
V66_PATH = (
    ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
HN_PATH = (
    ROOT
    / "realistic_pairwise_mainline_fusion_hn4"
    / "hn3_row_oof.parquet"
)
OUT_DIR = ROOT / "realistic_pairwise_pre_neighbor_order_audit_hn11"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HN_WEIGHT = 0.275
NEIGHBOR_DOSE = 0.20
NEIGHBOR_CAP_FT = 24.0
NEAR_LIMIT_FT = 1000.0
FAR_LIMIT_FT = 2000.0
EXPECTED_V66_ROWS = 3_783_989
EXPECTED_V66_WELLS = 773


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _metrics(frame, prediction):
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    distance = frame["md_since"].to_numpy(np.float64)
    error = prediction - truth
    error2 = error * error
    well_rmse = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error2})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )

    def region_rmse(mask):
        return float(np.sqrt(np.mean(error2[mask]))) if np.any(mask) else np.nan

    return {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
        "near_le1000_rmse": region_rmse(distance <= NEAR_LIMIT_FT),
        "mid_1000_2000_rmse": region_rmse(
            (distance > NEAR_LIMIT_FT) & (distance <= FAR_LIMIT_FT)
        ),
        "far_gt2000_rmse": region_rmse(distance > FAR_LIMIT_FT),
    }


def _neighbor_replay(parent, neighbor, active, tail_profile):
    parent = np.asarray(parent, np.float64)
    output = parent.copy()
    move = np.zeros(len(parent), np.float64)
    move[active] = (
        NEIGHBOR_DOSE
        * tail_profile[active]
        * np.clip(
            neighbor[active] - parent[active],
            -NEIGHBOR_CAP_FT,
            NEIGHBOR_CAP_FT,
        )
    )
    output += move
    return output, move


for required in (V66_PATH, HN_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v66_columns = [
    "id",
    "well",
    "row_idx",
    "fold",
    "tvt_true",
    "md_since",
    "full_cp8_ordered",
    "neighbor_absolute",
    "neighbor_active",
    "tail_profile",
    "incumbent_6p377_replay",
]
missing_v66 = sorted(set(v66_columns).difference(pq.read_schema(V66_PATH).names))
if missing_v66:
    raise RuntimeError(f"V66 misses required columns: {missing_v66}")
v66 = pd.read_parquet(V66_PATH, columns=v66_columns)
v66["id"] = v66["id"].astype(str)
v66["well"] = v66["well"].astype(str)
if len(v66) != EXPECTED_V66_ROWS or v66["well"].nunique() != EXPECTED_V66_WELLS:
    raise RuntimeError(
        f"Incomplete V66: rows={len(v66)} wells={v66['well'].nunique()}"
    )
if v66["id"].duplicated().any():
    raise RuntimeError("V66 contains duplicate ids")

hn_header = pq.read_schema(HN_PATH).names
hn_prediction_column = next(
    (name for name in ("hn3_pairwise", "pairwise", "prediction") if name in hn_header),
    None,
)
if hn_prediction_column is None:
    raise RuntimeError(f"Cannot identify HN prediction column: {hn_header}")
hn_columns = ["id", "well", "row_idx", "tvt_true", "md_since", hn_prediction_column]
hn = pd.read_parquet(HN_PATH, columns=hn_columns).rename(
    columns={
        "tvt_true": "hn_truth",
        "md_since": "hn_md_since",
        hn_prediction_column: "hn_endpoint",
    }
)
hn["id"] = hn["id"].astype(str)
hn["well"] = hn["well"].astype(str)
if hn["id"].duplicated().any():
    raise RuntimeError("HN OOF contains duplicate ids")

frame = v66.merge(hn, on=["id", "well", "row_idx"], how="inner", validate="one_to_one")
truth_gap = np.abs(
    frame["tvt_true"].to_numpy(np.float64)
    - frame["hn_truth"].to_numpy(np.float64)
)
distance_gap = np.abs(
    frame["md_since"].to_numpy(np.float64)
    - frame["hn_md_since"].to_numpy(np.float64)
)
if not len(frame) or truth_gap.max() > 0.01 or distance_gap.max() > 0.01:
    raise RuntimeError(
        f"V66/HN alignment failed: truth={truth_gap.max()} distance={distance_gap.max()}"
    )
frame = frame.drop(columns=["hn_truth", "hn_md_since"])

cp8 = frame["full_cp8_ordered"].to_numpy(np.float64)
neighbor = frame["neighbor_absolute"].to_numpy(np.float64)
active = frame["neighbor_active"].to_numpy(bool)
tail_profile = frame["tail_profile"].to_numpy(np.float64)
incumbent = frame["incumbent_6p377_replay"].to_numpy(np.float64)
hn_endpoint = frame["hn_endpoint"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)

replayed_incumbent, incumbent_move = _neighbor_replay(
    cp8, neighbor, active, tail_profile
)
lineage_gap = np.abs(replayed_incumbent - incumbent)
if lineage_gap.max() > 0.001:
    raise RuntimeError(
        f"Frozen Neighbor replay does not reproduce V66: max={lineage_gap.max()}"
    )

# Existing deployment order: all downstream mainline operations, then HN.
hn_after_neighbor = incumbent + HN_WEIGHT * (hn_endpoint - incumbent)

# Proposed order: HN modifies the pre-Neighbor parent; Neighbor then sees that
# modified trajectory and can pull the tail back toward its independent path.
hn_parent = cp8 + HN_WEIGHT * (hn_endpoint - cp8)
hn_before_neighbor, proposed_neighbor_move = _neighbor_replay(
    hn_parent, neighbor, active, tail_profile
)

near = distance <= NEAR_LIMIT_FT
near_order_gap = np.abs(hn_before_neighbor[near] - hn_after_neighbor[near])
if near_order_gap.max(initial=0.0) > 1e-10:
    raise RuntimeError(
        f"Operator order changed the frozen first 1000 ft: max={near_order_gap.max()}"
    )

endpoints = {
    "base_v66": incumbent,
    "hn_after_neighbor": hn_after_neighbor,
    "hn_before_neighbor": hn_before_neighbor,
}
overall_metrics = {name: _metrics(frame, values) for name, values in endpoints.items()}

fold_rows = []
for fold in sorted(frame["fold"].unique()):
    mask = frame["fold"].to_numpy(np.int8) == int(fold)
    local = frame.loc[mask].reset_index(drop=True)
    local_metrics = {
        name: _metrics(local, values[mask]) for name, values in endpoints.items()
    }
    fold_rows.append(
        {
            "fold": int(fold),
            "rows": int(mask.sum()),
            "wells": int(local["well"].nunique()),
            "base_rmse": local_metrics["base_v66"]["rmse"],
            "post_rmse": local_metrics["hn_after_neighbor"]["rmse"],
            "pre_rmse": local_metrics["hn_before_neighbor"]["rmse"],
            "post_gain": local_metrics["base_v66"]["rmse"]
            - local_metrics["hn_after_neighbor"]["rmse"],
            "pre_gain": local_metrics["base_v66"]["rmse"]
            - local_metrics["hn_before_neighbor"]["rmse"],
            "pre_gain_vs_post": local_metrics["hn_after_neighbor"]["rmse"]
            - local_metrics["hn_before_neighbor"]["rmse"],
            "base_far_rmse": local_metrics["base_v66"]["far_gt2000_rmse"],
            "post_far_rmse": local_metrics["hn_after_neighbor"]["far_gt2000_rmse"],
            "pre_far_rmse": local_metrics["hn_before_neighbor"]["far_gt2000_rmse"],
            "pre_far_gain_vs_post": local_metrics["hn_after_neighbor"]["far_gt2000_rmse"]
            - local_metrics["hn_before_neighbor"]["far_gt2000_rmse"],
            "pre_p95_change_vs_base": local_metrics["hn_before_neighbor"]["p95_well_rmse"]
            - local_metrics["base_v66"]["p95_well_rmse"],
        }
    )
fold_frame = pd.DataFrame(fold_rows)
fold_frame.to_csv(OUT_DIR / "folds.csv", index=False)

order_delta = hn_before_neighbor - hn_after_neighbor
active_order_delta = order_delta[active]
coverage = {
    "v66_rows": int(len(v66)),
    "v66_wells": int(v66["well"].nunique()),
    "hn_rows": int(len(hn)),
    "hn_wells": int(hn["well"].nunique()),
    "aligned_rows": int(len(frame)),
    "aligned_wells": int(frame["well"].nunique()),
    "fold_wells": (
        frame[["well", "fold"]]
        .drop_duplicates()
        .groupby("fold")["well"]
        .nunique()
        .astype(int)
        .to_dict()
    ),
    "truth_gap_max": float(truth_gap.max()),
    "distance_gap_max": float(distance_gap.max()),
    "v66_replay_gap_max": float(lineage_gap.max()),
}
checks = {
    "aligned_wells_ge_600": bool(frame["well"].nunique() >= 600),
    "fold1_wells_ge_90": bool(coverage["fold_wells"].get(1, 0) >= 90),
    "folds2to5_each_ge_145": bool(
        all(coverage["fold_wells"].get(fold, 0) >= 145 for fold in (2, 3, 4, 5))
    ),
    "near_order_bitwise_unchanged": bool(near_order_gap.max(initial=0.0) <= 1e-10),
    "pre_neighbor_gain_ge_0p03": bool(
        overall_metrics["base_v66"]["rmse"]
        - overall_metrics["hn_before_neighbor"]["rmse"]
        >= 0.03
    ),
    "pre_not_worse_than_post_by_gt_0p02": bool(
        overall_metrics["hn_before_neighbor"]["rmse"]
        - overall_metrics["hn_after_neighbor"]["rmse"]
        <= 0.02
    ),
    "positive_pre_folds_ge_4": bool((fold_frame["pre_gain"] > 0).sum() >= 4),
    "minimum_pre_fold_gain_ge_minus0p05": bool(fold_frame["pre_gain"].min() >= -0.05),
    "maximum_pre_p95_change_le_0p20": bool(
        fold_frame["pre_p95_change_vs_base"].max() <= 0.20
    ),
}

summary = []
for name, metric in overall_metrics.items():
    summary.append({"endpoint": name, **metric})
summary_frame = pd.DataFrame(summary)
summary_frame.to_csv(OUT_DIR / "summary.csv", index=False)

verdict = {
    "experiment": "realistic_pairwise_pre_neighbor_order_audit_hn11",
    "stage": "frozen_hn_before_vs_after_neighbor_complete_v66_oof",
    "selection_or_tuning_count": 0,
    "sample_unit": "row-aligned OOF predictions from complete-well crossfit",
    "sources": {"v66": V66_PATH, "hn_endpoint": HN_PATH},
    "frozen_recipe": {
        "hn_weight": HN_WEIGHT,
        "neighbor_dose": NEIGHBOR_DOSE,
        "neighbor_cap_ft": NEIGHBOR_CAP_FT,
        "neighbor_start_ft": NEAR_LIMIT_FT,
        "neighbor_transition_ft": 250.0,
    },
    "coverage": coverage,
    "metrics": overall_metrics,
    "comparison": {
        "post_gain_vs_base": overall_metrics["base_v66"]["rmse"]
        - overall_metrics["hn_after_neighbor"]["rmse"],
        "pre_gain_vs_base": overall_metrics["base_v66"]["rmse"]
        - overall_metrics["hn_before_neighbor"]["rmse"],
        "pre_gain_vs_post": overall_metrics["hn_after_neighbor"]["rmse"]
        - overall_metrics["hn_before_neighbor"]["rmse"],
        "pre_far_gain_vs_post": overall_metrics["hn_after_neighbor"]["far_gt2000_rmse"]
        - overall_metrics["hn_before_neighbor"]["far_gt2000_rmse"],
        "order_delta_rms_ft": float(np.sqrt(np.mean(order_delta * order_delta))),
        "order_delta_active_rms_ft": float(
            np.sqrt(np.mean(active_order_delta * active_order_delta))
        )
        if len(active_order_delta)
        else 0.0,
        "order_delta_p95_abs_ft": float(np.quantile(np.abs(order_delta), 0.95)),
        "near_order_max_abs_ft": float(near_order_gap.max(initial=0.0)),
    },
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> put HN7 immediately before Neighbor in Kaggle and run one placement submission"
        if all(checks.values())
        else "FAIL -> retain HN after-final result as diagnostic only; do not move it before visible-prefix"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)

print("HN11 COVERAGE", coverage, flush=True)
print("\nHN11 SUMMARY")
print(summary_frame.to_string(index=False), flush=True)
print("\nHN11 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nHN11 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
