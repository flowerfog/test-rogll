"""Retrospective fixed-dose robustness audit for the frozen V34 SGM path.

This cell does not rebuild or retune the GR cost map.  It consumes the exact
V34 correction and asks whether a smaller global dose preserves the consistent
three-fold pooled/far gain while removing the Fold-1 p95 regression.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_tail_sgm_dose_robustness_v35"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE = OOF_ROOT / "frontier_6p377_tail_sgm_residual_v34" / "tail_sgm_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOSES = tuple(np.round(np.arange(0.0, 0.1001, 0.0125), 4))
START_FT = 1000.0
started = time.time()


def metric(local, prediction):
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    well_mse = pd.Series(error**2).groupby(local["well"].to_numpy(), sort=False).mean()
    well_rmse = np.sqrt(well_mse.to_numpy(np.float64))
    near = distance <= 1000.0
    far = distance > 2000.0
    return {
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "v33_current_dose_0p20", "v34_tail_correction",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V34 OOF missing columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
base = frame["v33_current_dose_0p20"].to_numpy(np.float64)
correction = frame["v34_tail_correction"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int64)
if np.any(correction[distance <= START_FT] != 0.0):
    raise RuntimeError("V34 correction violates the frozen <=1000-ft zone")

base_by_fold = {}
for held_fold in (1, 2, 3):
    mask = fold == held_fold
    base_by_fold[held_fold] = metric(frame.loc[mask].reset_index(drop=True), base[mask])

rows = []
for dose in DOSES:
    prediction = base + float(dose) * correction
    if not np.array_equal(prediction[distance <= START_FT], base[distance <= START_FT]):
        raise RuntimeError(f"Dose {dose} changed the frozen near zone")
    for held_fold in (1, 2, 3):
        mask = fold == held_fold
        value = metric(frame.loc[mask].reset_index(drop=True), prediction[mask])
        baseline = base_by_fold[held_fold]
        rows.append({
            "dose": float(dose),
            "fold": held_fold,
            **value,
            "gain": baseline["rmse"] - value["rmse"],
            "far_gain": baseline["far_gt2000_rmse"] - value["far_gt2000_rmse"],
            "p95_change": value["p95_well_rmse"] - baseline["p95_well_rmse"],
            "max_change": value["max_well_rmse"] - baseline["max_well_rmse"],
            "near_change": value["near_le1000_rmse"] - baseline["near_le1000_rmse"],
        })
scan = pd.DataFrame(rows)
scan.to_csv(OUT_DIR / "dose_by_fold.csv", index=False)

summary = scan.groupby("dose", as_index=False).agg(
    mean_gain=("gain", "mean"),
    minimum_gain=("gain", "min"),
    mean_far_gain=("far_gain", "mean"),
    minimum_far_gain=("far_gain", "min"),
    maximum_p95_change=("p95_change", "max"),
    maximum_max_change=("max_change", "max"),
    maximum_abs_near_change=("near_change", lambda x: float(np.max(np.abs(x)))),
)
summary["robust"] = (
    summary["minimum_gain"].ge(0.02)
    & summary["minimum_far_gain"].ge(0.03)
    & summary["maximum_p95_change"].le(0.10)
    & summary["maximum_abs_near_change"].le(1e-10)
)
summary = summary.sort_values(
    ["robust", "minimum_gain", "mean_gain"], ascending=[False, False, False], kind="stable"
).reset_index(drop=True)
summary.to_csv(OUT_DIR / "dose_summary.csv", index=False)

selected = summary.iloc[0]
selected_dose = float(selected["dose"])
selected_prediction = base + selected_dose * correction
selected_rows = scan[np.isclose(scan["dose"], selected_dose)].sort_values("fold")
strict_pass = bool(selected["robust"])

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["v33_base"] = base.astype(np.float32)
output["v34_full_correction"] = correction.astype(np.float32)
output["v35_selected"] = selected_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "tail_sgm_robust_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "retrospective_threefold_fixed_v34_dose_robustness",
    "source": str(SOURCE),
    "lineage": [
        "frontier_6p377_cp8_neighbor_dose_replay_v33",
        "frontier_6p377_tail_sgm_residual_v34",
    ],
    "validation_status": "retrospective_only_all_available_folds_already_observed",
    "selected_dose": selected_dose,
    "selection_rule": "maximize minimum fold gain among p95-safe fixed doses",
    "selected_summary": selected.to_dict(),
    "selected_by_fold": selected_rows.to_dict(orient="records"),
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> package as a conservative LB-only candidate; retain V34 dose 0.10 as an aggressive alternative"
        if strict_pass else
        "FAIL -> V34 gain cannot be made p95-safe by a smaller fixed dose"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("V35 DOSE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV35 SELECTED BY FOLD")
print(selected_rows.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
