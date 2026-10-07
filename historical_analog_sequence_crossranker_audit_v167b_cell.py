"""V167b: read-only attribution and fixed-dose audit for V167.

Run after V167 in the same kernel. No feature generation or model fitting.
"""

from pathlib import Path
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


started_v167b = time.time()
out_v167b = Path("/root/autodl-tmp/oof_exports/historical_analog_sequence_crossranker_audit_v167b")
out_v167b.mkdir(parents=True, exist_ok=True)

required_v167b = (
    "candidate_v167", "frame_v167", "base_v167", "selected_v167",
    "rolled_v167", "score_v167", "rolled_score_v167", "FEATURE_NAMES_V167",
    "OUT_DIR_V167", "_metric",
)
missing_v167b = [name for name in required_v167b if name not in globals()]
if missing_v167b:
    raise RuntimeError("Run V167 first. Missing: " + repr(missing_v167b))

sequence_features_v167b = len(FEATURE_NAMES_V167) - 12
importance_rows_v167b = []
for fold_v167b in range(1, 6):
    model_path_v167b = OUT_DIR_V167 / f"ranker_fold{fold_v167b}.txt"
    booster_v167b = lgb.Booster(model_file=str(model_path_v167b))
    gain_v167b = booster_v167b.feature_importance(importance_type="gain").astype(float)
    split_v167b = booster_v167b.feature_importance(importance_type="split").astype(float)
    total_gain_v167b = max(float(gain_v167b.sum()), 1e-12)
    total_split_v167b = max(float(split_v167b.sum()), 1e-12)
    importance_rows_v167b.append({
        "fold": fold_v167b,
        "sequence_gain_fraction": float(gain_v167b[:sequence_features_v167b].sum() / total_gain_v167b),
        "metadata_gain_fraction": float(gain_v167b[sequence_features_v167b:].sum() / total_gain_v167b),
        "sequence_split_fraction": float(split_v167b[:sequence_features_v167b].sum() / total_split_v167b),
        "metadata_split_fraction": float(split_v167b[sequence_features_v167b:].sum() / total_split_v167b),
        "used_sequence_features": int(np.sum(gain_v167b[:sequence_features_v167b] > 0)),
        "used_metadata_features": int(np.sum(gain_v167b[sequence_features_v167b:] > 0)),
    })
importance_v167b = pd.DataFrame(importance_rows_v167b)

score_gap_v167b = np.asarray(score_v167, float) - np.asarray(rolled_score_v167, float)
aligned_choice_v167b = candidate_v167.groupby("well", sort=False)["score"].idxmax()
rolled_choice_v167b = candidate_v167.groupby("well", sort=False)["rolled_score"].idxmax()
aligned_refs_v167b = candidate_v167.loc[aligned_choice_v167b].set_index("well")["reference_well"]
rolled_refs_v167b = candidate_v167.loc[rolled_choice_v167b].set_index("well")["reference_well"]
choice_change_v167b = float((aligned_refs_v167b != rolled_refs_v167b).mean())

weights_v167b = np.round(np.arange(0.0, 0.151, 0.0125), 4)
scan_rows_v167b = []
for weight_v167b in weights_v167b:
    prediction_v167b = base_v167 + weight_v167b * (selected_v167 - base_v167)
    overall_v167b = _metric(frame_v167, prediction_v167b, f"fixed_w{weight_v167b:g}")
    row_v167b = {
        "weight": float(weight_v167b),
        "rmse": overall_v167b["rmse"],
        "gain": float(_metric(frame_v167, base_v167, "base")["rmse"] - overall_v167b["rmse"]),
        "far_gt2000_rmse": overall_v167b["far_gt2000_rmse"],
    }
    positive_v167b = 0
    minimum_v167b = np.inf
    for fold_v167b in range(1, 6):
        mask_v167b = frame_v167["fold"].to_numpy(int) == fold_v167b
        local_v167b = frame_v167.loc[mask_v167b].reset_index(drop=True)
        base_m_v167b = _metric(local_v167b, base_v167[mask_v167b], "base")
        pred_m_v167b = _metric(local_v167b, prediction_v167b[mask_v167b], "candidate")
        gain_fold_v167b = float(base_m_v167b["rmse"] - pred_m_v167b["rmse"])
        row_v167b[f"fold{fold_v167b}_gain"] = gain_fold_v167b
        positive_v167b += int(gain_fold_v167b > 0)
        minimum_v167b = min(minimum_v167b, gain_fold_v167b)
    row_v167b["positive_folds"] = positive_v167b
    row_v167b["minimum_fold_gain"] = float(minimum_v167b)
    scan_rows_v167b.append(row_v167b)
scan_v167b = pd.DataFrame(scan_rows_v167b).sort_values("rmse").reset_index(drop=True)
stable_v167b = scan_v167b[
    (scan_v167b["positive_folds"] >= 4)
    & (scan_v167b["minimum_fold_gain"] >= -0.02)
]

checks_v167b = {
    "score_changes_under_roll": bool(np.max(np.abs(score_gap_v167b)) > 1e-7),
    "choice_change_fraction_ge_0p05": bool(choice_change_v167b >= 0.05),
    "mean_sequence_gain_fraction_ge_0p20": bool(importance_v167b["sequence_gain_fraction"].mean() >= 0.20),
    "stable_fixed_weight_exists": bool(len(stable_v167b) > 0),
    "stable_fixed_gain_ge_0p05": bool(len(stable_v167b) > 0 and stable_v167b.iloc[0]["gain"] >= 0.05),
}
verdict_v167b = {
    "experiment": "historical_analog_sequence_crossranker_audit_v167b",
    "stage": "read_only_sequence_attribution_and_fixed_dose_audit",
    "sequence_features": sequence_features_v167b,
    "metadata_features": 12,
    "maximum_aligned_rolled_score_gap": float(np.max(np.abs(score_gap_v167b))),
    "aligned_rolled_choice_change_fraction": choice_change_v167b,
    "mean_sequence_gain_fraction": float(importance_v167b["sequence_gain_fraction"].mean()),
    "mean_metadata_gain_fraction": float(importance_v167b["metadata_gain_fraction"].mean()),
    "best_fixed": scan_v167b.iloc[0].to_dict(),
    "best_stable_fixed": stable_v167b.iloc[0].to_dict() if len(stable_v167b) else None,
    "checks": checks_v167b,
    "strict_pass": bool(all(checks_v167b.values())),
    "elapsed_minutes": (time.time() - started_v167b) / 60.0,
    "next": (
        "PASS -> confirm the frozen fixed dose on a separate endpoint lineage"
        if all(checks_v167b.values())
        else "FAIL -> V167 gain is metadata-only or fold-unstable; close analog retrieval"
    ),
}

importance_v167b.to_csv(out_v167b / "feature_block_importance.csv", index=False)
scan_v167b.to_csv(out_v167b / "fixed_weight_scan.csv", index=False)
(out_v167b / "verdict.json").write_text(json.dumps(verdict_v167b, indent=2), encoding="utf-8")
print("V167b IMPORTANCE")
print(importance_v167b.to_string(index=False), flush=True)
print("\nV167b FIXED WEIGHT SCAN")
print(scan_v167b.to_string(index=False), flush=True)
print("\nV167b FINAL VERDICT")
print(json.dumps(verdict_v167b, indent=2), flush=True)
print("saved:", out_v167b, flush=True)
