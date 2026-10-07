"""Diagnose the non-zero graph configurations hidden by nested abstention.

Run after ``strong_prior_absolute_edge_graph_gateway_v0_cell.py``.  This cell
does not solve paths, train models, or touch Fold 1.  It only aggregates the
already saved per-well configuration scan to distinguish lack of signal from
cross-meta-fold instability.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "strong_prior_absolute_edge_graph_gateway_v0"
SCAN_PATH = SOURCE_DIR / "config_scan_bywell.pkl"
OUT_DIR = OOF_ROOT / "strong_prior_graph_raw_scan_diagnostic_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not SCAN_PATH.exists():
    raise FileNotFoundError(SCAN_PATH)
scan = pd.read_pickle(SCAN_PATH)


def _rmse(sse, rows):
    return float(np.sqrt(float(sse) / max(int(rows), 1)))


def _aggregate(frame):
    rows = int(frame["rows"].sum())
    base = _rmse(frame["base_sse"].sum(), rows)
    score = _rmse(frame["sse"].sum(), rows)
    output = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "base_rmse": base,
        "rmse": score,
        "gain": base - score,
        "mean_abs_move": float(
            np.average(frame["mean_abs_move"], weights=frame["rows"])
        ),
        "p95_abs_move_well_median": float(frame["p95_abs_move"].median()),
        "events": int(frame["events"].sum()),
        "event_wells": int((frame["events"] > 0).sum()),
    }
    for label in ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"):
        group_rows = int(frame[f"{label}_rows"].sum())
        base_group = _rmse(frame[f"{label}_base_sse"].sum(), group_rows)
        score_group = _rmse(frame[f"{label}_sse"].sum(), group_rows)
        output[f"{label}_rmse"] = score_group
        output[f"{label}_gain"] = base_group - score_group
    return output


summary_rows = []
fold_rows = []
for (arm, config_id), frame in scan.groupby(["arm", "config_id"], sort=False):
    fold_gains = []
    for meta_fold, fold_frame in frame.groupby("meta_fold", sort=True):
        metrics = _aggregate(fold_frame)
        fold_gains.append(metrics["gain"])
        fold_rows.append(
            {
                "arm": arm,
                "config_id": config_id,
                "meta_fold": int(meta_fold),
                **metrics,
            }
        )
    metrics = _aggregate(frame)
    summary_rows.append(
        {
            "arm": arm,
            "config_id": config_id,
            **metrics,
            "positive_meta_folds": int(np.sum(np.asarray(fold_gains) > 0)),
            "minimum_meta_fold_gain": float(np.min(fold_gains)),
            "mean_meta_fold_gain": float(np.mean(fold_gains)),
        }
    )

summary = pd.DataFrame(summary_rows)
folds = pd.DataFrame(fold_rows)
summary["safety_pass"] = (
    (summary["gain"] >= 0.05)
    & (summary["positive_meta_folds"] >= 4)
    & (summary["minimum_meta_fold_gain"] >= -0.10)
    & (summary["near_0_500_gain"] >= -0.05)
)
summary["robust_rank"] = (
    summary["gain"]
    + np.minimum(summary["minimum_meta_fold_gain"], 0.0)
    + 0.25 * np.minimum(summary["near_0_500_gain"], 0.0)
)
summary = summary.sort_values(
    ["safety_pass", "robust_rank", "gain"], ascending=False
).reset_index(drop=True)

best_by_arm = (
    summary.sort_values(["gain", "minimum_meta_fold_gain"], ascending=False)
    .groupby("arm", as_index=False)
    .first()
    .sort_values("gain", ascending=False)
)
best_ids = set(best_by_arm["config_id"])
best_folds = folds.loc[folds["config_id"].isin(best_ids)].copy()

summary.to_csv(OUT_DIR / "all_config_summary.csv", index=False)
best_by_arm.to_csv(OUT_DIR / "best_unconstrained_by_arm.csv", index=False)
best_folds.to_csv(OUT_DIR / "best_unconstrained_meta_folds.csv", index=False)

nonzero = summary.loc[summary["arm"].ne("strong")]
best_nonzero = nonzero.iloc[0].to_dict() if len(nonzero) else None
verdict = {
    "experiment": "strong_prior_graph_raw_scan_diagnostic_v1",
    "source": str(SCAN_PATH),
    "configs": int(summary["config_id"].nunique()),
    "best_unconstrained_by_arm": best_by_arm.to_dict(orient="records"),
    "nonzero_safety_passing_configs": int(
        nonzero["safety_pass"].sum() if len(nonzero) else 0
    ),
    "best_nonzero": best_nonzero,
    "interpretation": (
        "NONZERO SIGNAL EXISTS -> inspect nested instability before closure"
        if best_nonzero is not None and float(best_nonzero["gain"]) >= 0.05
        else "NO MATERIAL NONZERO SIGNAL -> close strong-prior graph branch"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=lambda value: value.item()),
    encoding="utf-8",
)

print("BEST UNCONSTRAINED BY ARM")
print(best_by_arm.to_string(index=False), flush=True)
print("\nMETA FOLDS FOR RAW ARM WINNERS")
print(best_folds.to_string(index=False), flush=True)
print("\nTOP 20 RAW CONFIGS")
print(summary.head(20).to_string(index=False), flush=True)
print("\nVERDICT")
print(json.dumps(verdict, indent=2, default=lambda value: value.item()), flush=True)
print("saved:", OUT_DIR, flush=True)
