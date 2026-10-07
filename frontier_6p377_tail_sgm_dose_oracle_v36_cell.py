"""Truth-assisted reachability audit for adaptive V34 tail-SGM dosing.

This is a diagnostic oracle, not a deployable model.  It uses each well's
hidden truth to choose a dose and asks whether legal confidence modelling is
worth building after the fixed V35 dose of 0.0375.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_tail_sgm_dose_oracle_v36"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE = OOF_ROOT / "frontier_6p377_tail_sgm_residual_v34" / "tail_sgm_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_DOSE = 0.0375
PRIMARY_DOSES = (0.0, FIXED_DOSE, 0.10)
FINE_DOSES = tuple(np.round(np.arange(0.0, 0.1001, 0.0125), 4))
START_FT = 1000.0
MIN_INCREMENT_TO_TRAIN_GATE = 0.10
started = time.time()


def metrics(local, prediction):
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    groups = pd.Series(error**2).groupby(local["well"].to_numpy(), sort=False).mean()
    well_rmse = np.sqrt(groups.to_numpy(np.float64))
    near = distance <= START_FT
    far = distance > 2000.0
    return {
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def choose_per_well(frame, base, correction, doses, label):
    truth = frame["tvt_true"].to_numpy(np.float64)
    wells = frame["well"].to_numpy()
    unique_wells, inverse = np.unique(wells, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    candidate_sse = []
    for dose in doses:
        error = base + float(dose) * correction - truth
        candidate_sse.append(np.bincount(inverse, weights=error**2) / counts)
    candidate_mse = np.stack(candidate_sse, axis=1)
    choice_index = np.argmin(candidate_mse, axis=1)
    chosen_dose = np.asarray(doses, np.float64)[choice_index]
    row_dose = chosen_dose[inverse]
    prediction = base + row_dose * correction

    choice = pd.DataFrame({
        "well": unique_wells,
        "fold": frame.groupby("well", sort=True)["fold"].first().to_numpy(np.int64),
        "chosen_dose": chosen_dose,
        "base_mse": candidate_mse[:, 0],
        "oracle_mse": candidate_mse[np.arange(len(unique_wells)), choice_index],
    })
    fixed_index = int(np.argmin(np.abs(np.asarray(doses, np.float64) - FIXED_DOSE)))
    if not np.isclose(doses[fixed_index], FIXED_DOSE):
        fixed_error = base + FIXED_DOSE * correction - truth
        fixed_mse = np.bincount(inverse, weights=fixed_error**2) / counts
    else:
        fixed_mse = candidate_mse[:, fixed_index]
    choice["fixed_mse"] = fixed_mse
    choice["oracle_gain_vs_fixed_mse"] = choice["fixed_mse"] - choice["oracle_mse"]
    choice["label"] = label
    return prediction, row_dose, choice


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
folds = frame["fold"].to_numpy(np.int64)
if not np.isfinite(base).all() or not np.isfinite(correction).all():
    raise RuntimeError("Non-finite V33 base or V34 correction")
if np.any(correction[distance <= START_FT] != 0.0):
    raise RuntimeError("V34 correction violates frozen <=1000-ft zone")

fixed_prediction = base + FIXED_DOSE * correction
primary_prediction, primary_row_dose, primary_choices = choose_per_well(
    frame, base, correction, PRIMARY_DOSES, "primary_three_dose"
)
fine_prediction, fine_row_dose, fine_choices = choose_per_well(
    frame, base, correction, FINE_DOSES, "fine_grid"
)
choices = pd.concat([primary_choices, fine_choices], ignore_index=True)
choices.to_csv(OUT_DIR / "truth_assisted_well_choices.csv", index=False)

rows = []
predictions = {
    "v33_base": base,
    "v35_fixed_0p0375": fixed_prediction,
    "primary_three_dose_oracle": primary_prediction,
    "fine_grid_oracle": fine_prediction,
}
for variant, prediction in predictions.items():
    for held_fold in (0, 1, 2, 3):
        mask = np.ones(len(frame), dtype=bool) if held_fold == 0 else folds == held_fold
        local = frame.loc[mask].reset_index(drop=True)
        row = metrics(local, prediction[mask])
        row.update({"variant": variant, "fold": held_fold})
        rows.append(row)
summary = pd.DataFrame(rows)

base_metrics = summary[(summary.variant == "v33_base") & (summary.fold == 0)].iloc[0]
fixed_metrics = summary[(summary.variant == "v35_fixed_0p0375") & (summary.fold == 0)].iloc[0]
base_rmse_by_fold = summary[summary.variant == "v33_base"].set_index("fold")["rmse"]
fixed_rmse_by_fold = summary[summary.variant == "v35_fixed_0p0375"].set_index("fold")["rmse"]
summary["gain_vs_v33"] = [
    float(base_rmse_by_fold.loc[held_fold] - rmse)
    for held_fold, rmse in zip(summary["fold"], summary["rmse"])
]
summary["gain_vs_fixed"] = [
    float(fixed_rmse_by_fold.loc[held_fold] - rmse)
    for held_fold, rmse in zip(summary["fold"], summary["rmse"])
]
summary.to_csv(OUT_DIR / "oracle_summary.csv", index=False)

primary_metrics = summary[
    (summary.variant == "primary_three_dose_oracle") & (summary.fold == 0)
].iloc[0]
fine_metrics = summary[(summary.variant == "fine_grid_oracle") & (summary.fold == 0)].iloc[0]
primary_increment = float(fixed_metrics.rmse - primary_metrics.rmse)
fine_increment = float(fixed_metrics.rmse - fine_metrics.rmse)

primary_counts = {
    f"{dose:.4f}": int((primary_choices["chosen_dose"] == dose).sum())
    for dose in PRIMARY_DOSES
}
fine_counts = {
    f"{dose:.4f}": int((fine_choices["chosen_dose"] == dose).sum())
    for dose in FINE_DOSES
}
train_gate = primary_increment >= MIN_INCREMENT_TO_TRAIN_GATE

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["v33_base"] = base.astype(np.float32)
output["v34_correction"] = correction.astype(np.float32)
output["v35_fixed"] = fixed_prediction.astype(np.float32)
output["primary_oracle_dose"] = primary_row_dose.astype(np.float32)
output["primary_oracle_prediction"] = primary_prediction.astype(np.float32)
output["fine_oracle_dose"] = fine_row_dose.astype(np.float32)
output["fine_oracle_prediction"] = fine_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "dose_oracle_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "truth_assisted_per_well_tail_dose_reachability",
    "deployable": False,
    "truth_usage": "hidden truth selects one dose per complete physical well",
    "source": str(SOURCE),
    "lineage": [
        "frontier_6p377_cp8_neighbor_dose_replay_v33",
        "frontier_6p377_tail_sgm_residual_v34",
        "frontier_6p377_tail_sgm_dose_robustness_v35",
    ],
    "fixed_dose": FIXED_DOSE,
    "primary_doses": list(PRIMARY_DOSES),
    "fine_doses": list(FINE_DOSES),
    "v33": base_metrics.to_dict(),
    "v35_fixed": fixed_metrics.to_dict(),
    "primary_oracle": primary_metrics.to_dict(),
    "fine_oracle": fine_metrics.to_dict(),
    "primary_increment_vs_fixed": primary_increment,
    "fine_increment_vs_fixed": fine_increment,
    "primary_dose_counts": primary_counts,
    "fine_dose_counts": fine_counts,
    "gate": {
        "minimum_increment_to_train_legal_confidence_gate": MIN_INCREMENT_TO_TRAIN_GATE,
        "pass": bool(train_gate),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> audit legal pre-decode confidence features, then cross-fit a conservative dose gate"
        if train_gate else
        "FAIL -> close adaptive dosing; freeze V35 and seek a new evidence source rather than tuning SGM dose"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("V36 ORACLE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV36 PRIMARY DOSE COUNTS", primary_counts, flush=True)
print("V36 FINE DOSE COUNTS", fine_counts, flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
