"""V168: full-row confirmation of the frozen V167 metadata analog move.

Run after V167/V167b in the same kernel. The candidate selector remains the
fivefold cross-fitted V167 ranker. The dose is frozen at 0.05, which belonged
to V167's original blend grid and was positive on all five sampled folds.
No model fitting or parameter selection occurs here.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


started_v168 = time.time()
experiment_v168 = "historical_analog_metadata_fullrow_v168"
out_v168 = Path("/root/autodl-tmp/oof_exports") / experiment_v168
out_v168.mkdir(parents=True, exist_ok=True)
FROZEN_WEIGHT_V168 = 0.05

required_v168 = (
    "base_frame", "prediction_column", "candidate_v167", "records",
    "_transfer", "_metric",
)
missing_v168 = [name for name in required_v168 if name not in globals()]
if missing_v168:
    raise RuntimeError("Run V166 and V167 first. Missing: " + repr(missing_v168))

selected_index_v168 = candidate_v167.groupby("well", sort=False)["score"].idxmax()
selected_lookup_v168 = (
    candidate_v167.loc[selected_index_v168, ["well", "reference_well"]]
    .set_index("well")["reference_well"].astype(str).to_dict()
)

# A deterministic within-fold reference roll preserves the reference-fold
# exclusion rule while destroying target/reference pairing.
control_lookup_v168 = {}
selected_table_v168 = pd.DataFrame({
    "well": list(selected_lookup_v168),
    "reference_well": list(selected_lookup_v168.values()),
})
selected_table_v168["fold"] = selected_table_v168["well"].map(
    {well: int(records[well]["fold"]) for well in records}
)
for _, local_v168 in selected_table_v168.groupby("fold", sort=True):
    target_wells_v168 = local_v168["well"].astype(str).tolist()
    references_v168 = local_v168["reference_well"].astype(str).tolist()
    references_v168 = references_v168[1:] + references_v168[:1]
    control_lookup_v168.update(dict(zip(target_wells_v168, references_v168)))

full_v168 = base_frame.copy()
full_v168["well"] = full_v168["well"].astype(str)
if prediction_column not in full_v168:
    raise RuntimeError(f"Missing base prediction {prediction_column}")
full_v168 = full_v168.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
base_prediction_v168 = full_v168[prediction_column].to_numpy(np.float64)
aligned_prediction_v168 = np.full(len(full_v168), np.nan, np.float64)
control_prediction_v168 = np.full(len(full_v168), np.nan, np.float64)

groups_v168 = list(full_v168.groupby("well", sort=False).indices.items())
for number_v168, (well_v168, positions_v168) in enumerate(groups_v168, 1):
    well_v168 = str(well_v168)
    positions_v168 = np.asarray(positions_v168, np.int64)
    if well_v168 not in selected_lookup_v168:
        raise RuntimeError(f"No V167 selection for {well_v168}")
    target_v168 = records[well_v168]
    row_idx_v168 = full_v168.iloc[positions_v168]["row_idx"].to_numpy(np.int64)
    query_distance_v168 = (
        target_v168["md"][row_idx_v168]
        - target_v168["md"][target_v168["anchor"]]
    )
    target_z_v168 = target_v168["z"][row_idx_v168]
    aligned_reference_v168 = records[selected_lookup_v168[well_v168]]
    control_reference_v168 = records[control_lookup_v168[well_v168]]
    aligned_prediction_v168[positions_v168], _ = _transfer(
        aligned_reference_v168, target_v168, query_distance_v168, target_z_v168
    )
    control_prediction_v168[positions_v168], _ = _transfer(
        control_reference_v168, target_v168, query_distance_v168, target_z_v168
    )
    if number_v168 % 50 == 0 or number_v168 == len(groups_v168):
        print(f"V168 reconstruct {number_v168}/{len(groups_v168)}", flush=True)

if not np.isfinite(aligned_prediction_v168).all() or not np.isfinite(control_prediction_v168).all():
    raise RuntimeError("Non-finite V168 reconstruction")

candidate_v168 = base_prediction_v168 + FROZEN_WEIGHT_V168 * (
    aligned_prediction_v168 - base_prediction_v168
)
control_v168 = base_prediction_v168 + FROZEN_WEIGHT_V168 * (
    control_prediction_v168 - base_prediction_v168
)

summary_rows_v168 = []
fold_rows_v168 = []
for fold_v168 in (0, 1, 2, 3, 4, 5):
    mask_v168 = np.ones(len(full_v168), bool) if fold_v168 == 0 else (
        full_v168["fold"].to_numpy(int) == fold_v168
    )
    local_v168 = full_v168.loc[mask_v168].reset_index(drop=True)
    base_m_v168 = _metric(local_v168, base_prediction_v168[mask_v168], "base_current")
    candidate_m_v168 = _metric(local_v168, candidate_v168[mask_v168], "metadata_analog_w0p05")
    control_m_v168 = _metric(local_v168, control_v168[mask_v168], "rolled_reference_control")
    for metric_v168 in (base_m_v168, candidate_m_v168, control_m_v168):
        summary_rows_v168.append({"fold": fold_v168, **metric_v168})
    if fold_v168:
        fold_rows_v168.append({
            "fold": fold_v168,
            "base_rmse": base_m_v168["rmse"],
            "rmse": candidate_m_v168["rmse"],
            "gain": base_m_v168["rmse"] - candidate_m_v168["rmse"],
            "far_gain": base_m_v168["far_gt2000_rmse"] - candidate_m_v168["far_gt2000_rmse"],
            "control_increment": control_m_v168["rmse"] - candidate_m_v168["rmse"],
            "p95_gain": base_m_v168["p95_well_rmse"] - candidate_m_v168["p95_well_rmse"],
        })

summary_v168 = pd.DataFrame(summary_rows_v168)
folds_v168 = pd.DataFrame(fold_rows_v168)
overall_v168 = summary_v168[summary_v168["fold"].eq(0)].set_index("endpoint")
base_overall_v168 = overall_v168.loc["base_current"]
candidate_overall_v168 = overall_v168.loc["metadata_analog_w0p05"]
control_overall_v168 = overall_v168.loc["rolled_reference_control"]
checks_v168 = {
    "fullrow_gain_ge_0p05": bool(base_overall_v168["rmse"] - candidate_overall_v168["rmse"] >= 0.05),
    "positive_folds_5of5": bool((folds_v168["gain"] > 0).all()),
    "far_gain_positive": bool(base_overall_v168["far_gt2000_rmse"] > candidate_overall_v168["far_gt2000_rmse"]),
    "p95_not_worse_by_gt_0p10": bool(candidate_overall_v168["p95_well_rmse"] - base_overall_v168["p95_well_rmse"] <= 0.10),
    "aligned_beats_rolled_reference_ge_0p03": bool(control_overall_v168["rmse"] - candidate_overall_v168["rmse"] >= 0.03),
}
strict_pass_v168 = bool(all(checks_v168.values()))
verdict_v168 = {
    "experiment": experiment_v168,
    "stage": "complete_773well_fullrow_confirmation_of_frozen_metadata_analog_move",
    "selection_or_tuning_count": 0,
    "frozen_weight": FROZEN_WEIGHT_V168,
    "candidate_selector": "V167 fivefold cross-fitted metadata ranker",
    "coverage": {"rows": int(len(full_v168)), "wells": int(full_v168["well"].nunique())},
    "base": base_overall_v168.to_dict(),
    "candidate": candidate_overall_v168.to_dict(),
    "rolled_reference_control": control_overall_v168.to_dict(),
    "gain": float(base_overall_v168["rmse"] - candidate_overall_v168["rmse"]),
    "far_gain": float(base_overall_v168["far_gt2000_rmse"] - candidate_overall_v168["far_gt2000_rmse"]),
    "folds": fold_rows_v168,
    "checks": checks_v168,
    "strict_pass": strict_pass_v168,
    "elapsed_minutes": (time.time() - started_v168) / 60.0,
    "next": (
        "PASS -> rebuild the metadata selector on deployment candidates and package a frozen 0.05 move"
        if strict_pass_v168
        else "FAIL -> sampled V167 metadata gain does not survive complete-row confirmation"
    ),
}

output_v168 = full_v168[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output_v168["base_current"] = base_prediction_v168.astype(np.float32)
output_v168["metadata_analog_w0p05"] = candidate_v168.astype(np.float32)
output_v168["rolled_reference_control"] = control_v168.astype(np.float32)
output_v168.to_parquet(out_v168 / "fullrow_oof.parquet", index=False, compression="zstd")
summary_v168.to_csv(out_v168 / "summary.csv", index=False)
folds_v168.to_csv(out_v168 / "folds.csv", index=False)
(out_v168 / "verdict.json").write_text(json.dumps(verdict_v168, indent=2, default=str), encoding="utf-8")
print("\nV168 MATCHED SUMMARY")
print(summary_v168.to_string(index=False), flush=True)
print("\nV168 FOLDS")
print(folds_v168.to_string(index=False), flush=True)
print("\nV168 FINAL VERDICT")
print(json.dumps(verdict_v168, indent=2, default=str), flush=True)
print("saved:", out_v168, flush=True)
