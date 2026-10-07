"""V169: nested top-K consensus for the V167 metadata analog ranker.

Run after V168 in the same kernel. Candidate scores are frozen cross-fitted
V167 scores. For each held fold, consensus recipe and dose are selected using
only the other four folds' OOF rows, then applied once to the held fold.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


started_v169 = time.time()
experiment_v169 = "historical_analog_ranked_consensus_v169"
out_v169 = Path("/root/autodl-tmp/oof_exports") / experiment_v169
out_v169.mkdir(parents=True, exist_ok=True)

TOPK_V169 = (1, 2, 4, 8, 16)
AGGREGATORS_V169 = ("mean", "median")
WEIGHTS_V169 = (0.025, 0.05, 0.075, 0.10, 0.15)

required_v169 = (
    "full_v168", "base_prediction_v168", "candidate_v167", "records",
    "_transfer", "_metric",
)
missing_v169 = [name for name in required_v169 if name not in globals()]
if missing_v169:
    raise RuntimeError("Run V167 and V168 first. Missing: " + repr(missing_v169))

ranked_refs_v169 = {}
for well_v169, local_v169 in candidate_v167.groupby("well", sort=False):
    ranked_refs_v169[str(well_v169)] = (
        local_v169.sort_values("score", ascending=False, kind="stable")
        ["reference_well"].astype(str).head(max(TOPK_V169)).tolist()
    )

endpoint_names_v169 = [
    f"{aggregator_v169}_top{topk_v169}"
    for aggregator_v169 in AGGREGATORS_V169
    for topk_v169 in TOPK_V169
    if not (topk_v169 == 1 and aggregator_v169 == "median")
]
paths_v169 = {
    endpoint_v169: np.full(len(full_v168), np.nan, np.float32)
    for endpoint_v169 in endpoint_names_v169
}

groups_v169 = list(full_v168.groupby("well", sort=False).indices.items())
for number_v169, (well_v169, positions_v169) in enumerate(groups_v169, 1):
    well_v169 = str(well_v169)
    positions_v169 = np.asarray(positions_v169, np.int64)
    target_v169 = records[well_v169]
    row_idx_v169 = full_v168.iloc[positions_v169]["row_idx"].to_numpy(np.int64)
    distance_v169 = (
        target_v169["md"][row_idx_v169]
        - target_v169["md"][target_v169["anchor"]]
    )
    z_v169 = target_v169["z"][row_idx_v169]
    references_v169 = ranked_refs_v169[well_v169]
    matrix_v169 = []
    for reference_well_v169 in references_v169:
        prediction_v169, _ = _transfer(
            records[reference_well_v169], target_v169, distance_v169, z_v169
        )
        matrix_v169.append(np.asarray(prediction_v169, np.float64))
    matrix_v169 = np.asarray(matrix_v169, np.float64)
    for topk_v169 in TOPK_V169:
        local_matrix_v169 = matrix_v169[:topk_v169]
        paths_v169[f"mean_top{topk_v169}"][positions_v169] = np.mean(
            local_matrix_v169, axis=0
        ).astype(np.float32)
        if topk_v169 > 1:
            paths_v169[f"median_top{topk_v169}"][positions_v169] = np.median(
                local_matrix_v169, axis=0
            ).astype(np.float32)
    if number_v169 % 50 == 0 or number_v169 == len(groups_v169):
        print(f"V169 reconstruct {number_v169}/{len(groups_v169)}", flush=True)

for endpoint_v169, prediction_v169 in paths_v169.items():
    if not np.isfinite(prediction_v169).all():
        raise RuntimeError(f"Non-finite V169 endpoint {endpoint_v169}")

truth_v169 = full_v168["tvt_true"].to_numpy(np.float64)
fold_array_v169 = full_v168["fold"].to_numpy(int)
base_rmse_v169 = float(np.sqrt(np.mean((base_prediction_v168 - truth_v169) ** 2)))
scan_rows_v169 = []
for endpoint_v169, path_v169 in paths_v169.items():
    path_v169 = path_v169.astype(np.float64)
    for weight_v169 in WEIGHTS_V169:
        prediction_v169 = base_prediction_v168 + weight_v169 * (
            path_v169 - base_prediction_v168
        )
        row_v169 = {
            "endpoint": endpoint_v169,
            "weight": float(weight_v169),
            "rmse": float(np.sqrt(np.mean((prediction_v169 - truth_v169) ** 2))),
        }
        row_v169["gain"] = base_rmse_v169 - row_v169["rmse"]
        for fold_v169 in range(1, 6):
            mask_v169 = fold_array_v169 == fold_v169
            base_fold_v169 = float(np.sqrt(np.mean(
                (base_prediction_v168[mask_v169] - truth_v169[mask_v169]) ** 2
            )))
            score_fold_v169 = float(np.sqrt(np.mean(
                (prediction_v169[mask_v169] - truth_v169[mask_v169]) ** 2
            )))
            row_v169[f"fold{fold_v169}_gain"] = base_fold_v169 - score_fold_v169
        scan_rows_v169.append(row_v169)
scan_v169 = pd.DataFrame(scan_rows_v169).sort_values("rmse").reset_index(drop=True)

# Strict second-level nested selection. Each held fold sees a recipe selected
# solely by pooled SSE from the other four cross-fitted folds.
nested_prediction_v169 = np.full(len(full_v168), np.nan, np.float64)
selection_rows_v169 = []
for held_fold_v169 in range(1, 6):
    train_mask_v169 = fold_array_v169 != held_fold_v169
    held_mask_v169 = ~train_mask_v169
    choices_v169 = []
    for endpoint_v169, path_v169 in paths_v169.items():
        path_v169 = path_v169.astype(np.float64)
        for weight_v169 in WEIGHTS_V169:
            candidate_train_v169 = base_prediction_v168[train_mask_v169] + weight_v169 * (
                path_v169[train_mask_v169] - base_prediction_v168[train_mask_v169]
            )
            score_v169 = float(np.sqrt(np.mean(
                (candidate_train_v169 - truth_v169[train_mask_v169]) ** 2
            )))
            choices_v169.append((score_v169, endpoint_v169, weight_v169))
    selection_rmse_v169, selected_endpoint_v169, selected_weight_v169 = min(choices_v169)
    selected_path_v169 = paths_v169[selected_endpoint_v169].astype(np.float64)
    nested_prediction_v169[held_mask_v169] = (
        base_prediction_v168[held_mask_v169]
        + selected_weight_v169 * (
            selected_path_v169[held_mask_v169] - base_prediction_v168[held_mask_v169]
        )
    )
    held_frame_v169 = full_v168.loc[held_mask_v169].reset_index(drop=True)
    base_m_v169 = _metric(
        held_frame_v169, base_prediction_v168[held_mask_v169], "base"
    )
    candidate_m_v169 = _metric(
        held_frame_v169, nested_prediction_v169[held_mask_v169], "nested_consensus"
    )
    selection_rows_v169.append({
        "fold": held_fold_v169,
        "selected_endpoint": selected_endpoint_v169,
        "selected_weight": float(selected_weight_v169),
        "selection_rmse": selection_rmse_v169,
        "base_rmse": base_m_v169["rmse"],
        "rmse": candidate_m_v169["rmse"],
        "gain": base_m_v169["rmse"] - candidate_m_v169["rmse"],
        "far_gain": base_m_v169["far_gt2000_rmse"] - candidate_m_v169["far_gt2000_rmse"],
        "p95_gain": base_m_v169["p95_well_rmse"] - candidate_m_v169["p95_well_rmse"],
    })

if not np.isfinite(nested_prediction_v169).all():
    raise RuntimeError("Incomplete V169 nested prediction")

base_m_v169 = _metric(full_v168, base_prediction_v168, "base_current")
v168_prediction_v169 = base_prediction_v168 + 0.05 * (
    paths_v169["mean_top1"].astype(np.float64) - base_prediction_v168
)
v168_m_v169 = _metric(full_v168, v168_prediction_v169, "v168_top1_w0p05")
nested_m_v169 = _metric(full_v168, nested_prediction_v169, "nested_consensus")
selections_v169 = pd.DataFrame(selection_rows_v169)
checks_v169 = {
    "gain_vs_v168_ge_0p03": bool(v168_m_v169["rmse"] - nested_m_v169["rmse"] >= 0.03),
    "positive_folds_ge_4": bool((selections_v169["gain"] > 0).sum() >= 4),
    "minimum_fold_gain_ge_minus0p03": bool(selections_v169["gain"].min() >= -0.03),
    "far_gain_positive": bool(base_m_v169["far_gt2000_rmse"] > nested_m_v169["far_gt2000_rmse"]),
    "p95_not_worse_by_gt_0p10": bool(nested_m_v169["p95_well_rmse"] - base_m_v169["p95_well_rmse"] <= 0.10),
}
strict_pass_v169 = bool(all(checks_v169.values()))
summary_v169 = pd.DataFrame([base_m_v169, v168_m_v169, nested_m_v169])
verdict_v169 = {
    "experiment": experiment_v169,
    "stage": "fivefold_nested_ranked_analog_consensus",
    "base": "V144 full-row current endpoint",
    "reference": "V168 frozen top1 weight 0.05",
    "candidate_recipes": len(paths_v169) * len(WEIGHTS_V169),
    "coverage": {"rows": int(len(full_v168)), "wells": int(full_v168["well"].nunique())},
    "base_rmse": base_m_v169["rmse"],
    "v168_rmse": v168_m_v169["rmse"],
    "nested_rmse": nested_m_v169["rmse"],
    "gain_vs_base": base_m_v169["rmse"] - nested_m_v169["rmse"],
    "gain_vs_v168": v168_m_v169["rmse"] - nested_m_v169["rmse"],
    "folds": selection_rows_v169,
    "checks": checks_v169,
    "strict_pass": strict_pass_v169,
    "elapsed_minutes": (time.time() - started_v169) / 60.0,
    "next": (
        "PASS -> freeze the nested-majority consensus recipe and build deployment bundle"
        if strict_pass_v169
        else "FAIL -> retain V168 top1 weight 0.05"
    ),
}

output_v169 = full_v168[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output_v169["base_current"] = base_prediction_v168.astype(np.float32)
output_v169["v168_top1_w0p05"] = v168_prediction_v169.astype(np.float32)
output_v169["nested_consensus"] = nested_prediction_v169.astype(np.float32)
output_v169.to_parquet(out_v169 / "fullrow_oof.parquet", index=False, compression="zstd")
scan_v169.to_csv(out_v169 / "development_scan.csv", index=False)
selections_v169.to_csv(out_v169 / "nested_selections.csv", index=False)
summary_v169.to_csv(out_v169 / "summary.csv", index=False)
(out_v169 / "verdict.json").write_text(json.dumps(verdict_v169, indent=2), encoding="utf-8")
print("\nV169 DEVELOPMENT TOP 20")
print(scan_v169.head(20).to_string(index=False), flush=True)
print("\nV169 MATCHED SUMMARY")
print(summary_v169.to_string(index=False), flush=True)
print("\nV169 NESTED SELECTIONS")
print(selections_v169.to_string(index=False), flush=True)
print("\nV169 FINAL VERDICT")
print(json.dumps(verdict_v169, indent=2), flush=True)
print("saved:", out_v169, flush=True)
