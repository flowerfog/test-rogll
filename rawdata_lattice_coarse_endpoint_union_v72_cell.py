"""Constrained canonical345/All773 coarse endpoint union on the 6.377 lineage.

This is a read-only AutoDL audit.  Fold 2 selects non-negative endpoint doses
on a frozen grid; Fold 1 is touched once for confirmation.  The primary
control is the already validated All773 endpoint at dose 0.30, not the parent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_coarse_endpoint_union_v72"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
CANONICAL_PATH = (
    OOF_ROOT
    / "rawdata_lattice_coarse_canonical_gateway_v67"
    / "canonical_fold12_coarse_oof.parquet"
)
ALL773_PATH = (
    OOF_ROOT
    / "rawdata_lattice_coarse_all773_crossfit_v70"
    / "all773_proxy_crossfit_oof.parquet"
)

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
CANONICAL = "coarse_canonical345"
ALL773 = "coarse_all773"
SELECTION_FOLD = 2
CONFIRMATION_FOLD = 1
DOSES = np.round(np.arange(0.0, 0.4001, 0.05), 2)
MAX_TOTAL_DOSE = 0.50
REFERENCE_DOSE = 0.30


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    error = prediction - truth
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    strata = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in strata.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _union(frame, weight_canonical, weight_all773):
    parent = frame[PARENT].to_numpy(np.float64)
    canonical = frame[CANONICAL].to_numpy(np.float64)
    all773 = frame[ALL773].to_numpy(np.float64)
    return (
        parent
        + float(weight_canonical) * (canonical - parent)
        + float(weight_all773) * (all773 - parent)
    )


for required in (PARENT_PATH, CANONICAL_PATH, ALL773_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

parent = pd.read_parquet(
    PARENT_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT],
)
canonical = pd.read_parquet(
    CANONICAL_PATH,
    columns=["id", "fold", TRUTH, "coarse_path"],
).rename(columns={TRUTH: "canonical_truth", "coarse_path": CANONICAL})
all773 = pd.read_parquet(
    ALL773_PATH,
    columns=["id", "fold", TRUTH, "coarse_crossfit"],
).rename(columns={TRUTH: "all773_truth", "coarse_crossfit": ALL773})

for label, source in (("parent", parent), ("canonical", canonical), ("all773", all773)):
    source["id"] = source["id"].astype(str)
    if source["id"].duplicated().any():
        raise RuntimeError(f"{label} contains duplicate ids")

frame = parent[parent["fold"].isin((SELECTION_FOLD, CONFIRMATION_FOLD))].merge(
    canonical, on=["id", "fold"], how="inner", validate="one_to_one"
).merge(
    all773, on=["id", "fold"], how="inner", validate="one_to_one"
)
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
truth_gap = float(
    max(
        np.max(np.abs(frame[TRUTH].to_numpy(np.float64) - frame["canonical_truth"].to_numpy(np.float64))),
        np.max(np.abs(frame[TRUTH].to_numpy(np.float64) - frame["all773_truth"].to_numpy(np.float64))),
    )
)
if truth_gap > 0.01:
    raise RuntimeError(f"Truth mismatch across endpoints: {truth_gap}")
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if fold_wells != {1: 155, 2: 155}:
    raise RuntimeError(f"Incomplete Fold1/2 coverage: {fold_wells}")
if frame[[TRUTH, "md_since", PARENT, CANONICAL, ALL773]].isna().any().any():
    raise RuntimeError("Matched endpoint frame contains missing values")

selection = frame[frame["fold"].eq(SELECTION_FOLD)].copy()
confirmation = frame[frame["fold"].eq(CONFIRMATION_FOLD)].copy()
scan_rows = []
for weight_canonical in DOSES:
    for weight_all773 in DOSES:
        total = float(weight_canonical + weight_all773)
        if total > MAX_TOTAL_DOSE + 1e-12:
            continue
        metric = _metric(
            selection,
            _union(selection, weight_canonical, weight_all773),
            "union",
        )
        scan_rows.append(
            {
                "weight_canonical": float(weight_canonical),
                "weight_all773": float(weight_all773),
                "total_weight": total,
                **metric,
            }
        )

scan = pd.DataFrame(scan_rows).sort_values(
    ["exact_sse", "total_weight", "weight_canonical"], kind="stable"
).reset_index(drop=True)
selected = scan.iloc[0]
selected_weights = {
    "canonical345": float(selected["weight_canonical"]),
    "all773": float(selected["weight_all773"]),
}
scan.to_csv(OUT_DIR / "fold2_weight_surface.csv", index=False)

summary_rows = []
metrics = {}
oof_parts = []
for fold, local in ((SELECTION_FOLD, selection), (CONFIRMATION_FOLD, confirmation)):
    predictions = {
        "parent": local[PARENT].to_numpy(np.float64),
        "canonical_w0p30": _union(local, REFERENCE_DOSE, 0.0),
        "all773_w0p30": _union(local, 0.0, REFERENCE_DOSE),
        "average_endpoints_w0p30": _union(local, 0.15, 0.15),
        "selected_union": _union(
            local,
            selected_weights["canonical345"],
            selected_weights["all773"],
        ),
    }
    metrics[str(fold)] = {}
    for endpoint, prediction in predictions.items():
        metric = _metric(local, prediction, endpoint)
        metrics[str(fold)][endpoint] = metric
        summary_rows.append({"fold": int(fold), **metric})
    output = local[["id", "well", "row_idx", "fold", TRUTH, "md_since"]].copy()
    output[PARENT] = local[PARENT].to_numpy(np.float32)
    output[CANONICAL] = local[CANONICAL].to_numpy(np.float32)
    output[ALL773] = local[ALL773].to_numpy(np.float32)
    output["selected_union"] = predictions["selected_union"].astype(np.float32)
    oof_parts.append(output)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.concat(oof_parts, ignore_index=True).to_parquet(
    OUT_DIR / "selected_union_fold12_oof.parquet", index=False, compression="zstd"
)

fold1 = metrics["1"]
fold2 = metrics["2"]
gains = {
    "fold2_union_vs_all773": float(
        fold2["all773_w0p30"]["rmse"] - fold2["selected_union"]["rmse"]
    ),
    "fold1_union_vs_all773": float(
        fold1["all773_w0p30"]["rmse"] - fold1["selected_union"]["rmse"]
    ),
    "fold1_far_union_vs_all773": float(
        fold1["all773_w0p30"]["far_gt2000_rmse"]
        - fold1["selected_union"]["far_gt2000_rmse"]
    ),
    "fold1_p95_union_vs_all773": float(
        fold1["all773_w0p30"]["p95_well_rmse"]
        - fold1["selected_union"]["p95_well_rmse"]
    ),
}
checks = {
    "fold2_union_gain_ge_0p03": bool(gains["fold2_union_vs_all773"] >= 0.03),
    "fold1_union_gain_ge_0p03": bool(gains["fold1_union_vs_all773"] >= 0.03),
    "fold1_far_nonworse": bool(gains["fold1_far_union_vs_all773"] >= 0.0),
    "fold1_p95_nonworse": bool(gains["fold1_p95_union_vs_all773"] >= 0.0),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_constrained_endpoint_union_selection_Fold1_confirmation",
    "read_only": True,
    "parent": "complete_6p377_CP8_plus_tail_only_Neighbor",
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "fold1_selection_or_tuning_count": 0,
        "dose_grid": DOSES.tolist(),
        "maximum_total_dose": MAX_TOTAL_DOSE,
        "strong_control": "All773 coarse at fixed dose 0.30",
    },
    "alignment": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "fold_wells": fold_wells,
        "maximum_truth_gap": truth_gap,
    },
    "selected_weights": selected_weights,
    "metrics": metrics,
    "gains": gains,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build a two-endpoint Kaggle deployment cell at the end of the 6.377 lineage"
        if strict_pass
        else "FAIL -> deploy All773 alone at dose 0.30; endpoint averaging adds no stable value"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV72 FOLD2 TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV72 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV72 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
