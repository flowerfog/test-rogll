"""V189: zero-training consumption and shortlist audit of V188 scores.

Run immediately after V188 in the same kernel.  This cell does not fit a new
model. It cross-fits a confidence quantile and dose for the V188 selected
trajectory, audits a rolled-match control, and measures truth-only top-K
shortlist ceilings to decide whether pairwise/listwise training is justified.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


STARTED_V189 = time.time()
REQUIRED_V189 = (
    "base", "paths", "predictions", "score", "panel", "well_slices",
    "names", "k", "base_index", "truth", "sse", "OOF", "_metrics",
)
missing_v189 = [name for name in REQUIRED_V189 if name not in globals()]
if missing_v189:
    raise RuntimeError("Run V188 first in this kernel. Missing: " + repr(missing_v189))

OUT_V189 = OOF / "canonical_block_selector_consumption_v189"
OUT_V189.mkdir(parents=True, exist_ok=True)
DOSES_V189 = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)
QUANTILES_V189 = (0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95)
TOPK_V189 = (1, 2, 3, 4, 6, 8, 12)
print("V189 CONFIG", {"blocks": int(panel["group"].max()) + 1, "candidates": int(k)}, flush=True)

base_prediction_v189 = paths[base_index].astype(np.float64)
selected_v189 = predictions["full"].astype(np.float64)
rolled_v189 = predictions["rolled_match"].astype(np.float64)
fold_array_v189 = base["fold"].to_numpy(np.int8)
base_error_v189 = base_prediction_v189 - truth
move_v189 = selected_v189 - base_prediction_v189

# One confidence margin per block: predicted base RMSE minus predicted best
# candidate RMSE. Positive values mean the model expects a switch to help.
groups_v189 = int(panel["group"].max()) + 1
group_fold_v189 = panel.groupby("group", sort=True)["fold"].first().to_numpy(np.int8)
score_matrix_v189 = score["full"].astype(np.float64).reshape(groups_v189, k)
margin_v189 = score_matrix_v189[:, base_index] - np.min(score_matrix_v189, axis=1)

row_group_v189 = np.full(len(base), -1, np.int32)
for slices_v189 in well_slices.values():
    for group_v189, _, positions_v189 in slices_v189:
        row_group_v189[positions_v189] = int(group_v189)
active_rows_v189 = row_group_v189 >= 0
row_margin_v189 = np.full(len(base), -np.inf, np.float64)
row_margin_v189[active_rows_v189] = margin_v189[row_group_v189[active_rows_v189]]

# Per-block SSE is quadratic in dose: SSE(d) = a + 2*d*b + d^2*c.
# This avoids materialising 3.78M-row predictions during recipe selection.
group_b_v189 = np.bincount(
    row_group_v189[active_rows_v189],
    weights=(base_error_v189 * move_v189)[active_rows_v189],
    minlength=groups_v189,
).astype(np.float64)
group_c_v189 = np.bincount(
    row_group_v189[active_rows_v189],
    weights=(move_v189 * move_v189)[active_rows_v189],
    minlength=groups_v189,
).astype(np.float64)
print("V189 block quadratics ready", flush=True)


def _prediction_v189(move, threshold, dose):
    eligible_v189 = active_rows_v189 & (row_margin_v189 >= threshold)
    return base_prediction_v189 + float(dose) * np.asarray(move, np.float64) * eligible_v189


def _sse_v189(prediction, mask):
    error_v189 = np.asarray(prediction, np.float64)[mask] - truth[mask]
    return float(np.dot(error_v189, error_v189))


# Nested recipe selection: each held fold receives a threshold and dose chosen
# only by the other four folds. Quantiles are converted to numeric thresholds
# on those training blocks, then frozen for the held fold.
nested_v189 = base_prediction_v189.copy()
nested_rolled_v189 = base_prediction_v189.copy()
selection_rows_v189 = []
for held_v189 in range(1, 6):
    train_rows_v189 = fold_array_v189 != held_v189
    valid_rows_v189 = fold_array_v189 == held_v189
    train_blocks_v189 = group_fold_v189 != held_v189
    base_train_sse_v189 = _sse_v189(base_prediction_v189, train_rows_v189)
    candidates_v189 = []
    for quantile_v189 in QUANTILES_V189:
        threshold_v189 = float(np.quantile(margin_v189[train_blocks_v189], quantile_v189))
        eligible_blocks_v189 = train_blocks_v189 & (margin_v189 >= threshold_v189)
        for dose_v189 in DOSES_V189:
            delta_sse_v189 = np.sum(
                2.0 * dose_v189 * group_b_v189[eligible_blocks_v189]
                + dose_v189 * dose_v189 * group_c_v189[eligible_blocks_v189]
            )
            candidates_v189.append((
                base_train_sse_v189 + float(delta_sse_v189),
                quantile_v189, threshold_v189, dose_v189,
            ))
    _, quantile_v189, threshold_v189, dose_v189 = min(candidates_v189, key=lambda value: value[0])
    aligned_v189 = _prediction_v189(move_v189, threshold_v189, dose_v189)
    rolled_candidate_v189 = _prediction_v189(rolled_v189 - base_prediction_v189, threshold_v189, dose_v189)
    nested_v189[valid_rows_v189] = aligned_v189[valid_rows_v189]
    nested_rolled_v189[valid_rows_v189] = rolled_candidate_v189[valid_rows_v189]
    selection_rows_v189.append({
        "fold": held_v189, "quantile": quantile_v189,
        "threshold": threshold_v189, "dose": dose_v189,
        "selected_block_fraction": float(np.mean(margin_v189[group_fold_v189 == held_v189] >= threshold_v189)),
    })
    print("V189 nested fold", selection_rows_v189[-1], flush=True)


# Analytic global dose is oracle-only attribution, never a deployable result.
denominator_v189 = float(np.dot(move_v189, move_v189))
analytic_alpha_v189 = float(np.clip(-np.dot(base_error_v189, move_v189) / max(denominator_v189, 1e-12), 0.0, 1.0))
analytic_v189 = base_prediction_v189 + analytic_alpha_v189 * move_v189


# Does the learned ordering at least preserve the right endpoint in a small
# shortlist? Truth chooses only inside each learned top-K set.
topk_predictions_v189 = {}
sse_matrix_v189 = np.asarray(sse, np.float64).reshape(groups_v189, k)
score_order_v189 = np.argsort(score_matrix_v189, axis=1, kind="stable")
active_indices_v189 = np.flatnonzero(active_rows_v189)
active_groups_v189 = row_group_v189[active_indices_v189]
for topk_v189 in TOPK_V189:
    shortlist_v189 = score_order_v189[:, :min(topk_v189, k)]
    shortlist_sse_v189 = np.take_along_axis(sse_matrix_v189, shortlist_v189, axis=1)
    best_position_v189 = np.argmin(shortlist_sse_v189, axis=1)
    best_candidate_v189 = shortlist_v189[np.arange(groups_v189), best_position_v189]
    output_v189 = base_prediction_v189.copy()
    output_v189[active_indices_v189] = paths[
        best_candidate_v189[active_groups_v189], active_indices_v189
    ]
    topk_predictions_v189[topk_v189] = output_v189
    print("V189 shortlist", {"topk": topk_v189}, flush=True)


summary_rows_v189 = []
for name_v189, prediction_v189 in (
    ("v168", base_prediction_v189),
    ("v188_full", selected_v189),
    ("nested_confidence_dose", nested_v189),
    ("nested_rolled_control", nested_rolled_v189),
    ("analytic_global_dose_oracle", analytic_v189),
):
    summary_rows_v189.append(_metrics(base, prediction_v189, name_v189))
for topk_v189, prediction_v189 in topk_predictions_v189.items():
    summary_rows_v189.append(_metrics(base, prediction_v189, f"learned_top{topk_v189}_oracle"))
summary_v189 = pd.DataFrame(summary_rows_v189).sort_values("rmse", ignore_index=True)

fold_rows_v189 = []
for fold_v189 in range(1, 6):
    mask_v189 = fold_array_v189 == fold_v189
    local_v189 = base.loc[mask_v189].reset_index(drop=True)
    base_metric_v189 = _metrics(local_v189, base_prediction_v189[mask_v189], "v168")
    candidate_metric_v189 = _metrics(local_v189, nested_v189[mask_v189], "nested")
    rolled_metric_v189 = _metrics(local_v189, nested_rolled_v189[mask_v189], "rolled")
    fold_rows_v189.append({
        "fold": fold_v189,
        "gain": base_metric_v189["rmse"] - candidate_metric_v189["rmse"],
        "far_gain": base_metric_v189["far_gt2000_rmse"] - candidate_metric_v189["far_gt2000_rmse"],
        "p95_gain": base_metric_v189["p95_well_rmse"] - candidate_metric_v189["p95_well_rmse"],
        "aligned_increment_vs_rolled": rolled_metric_v189["rmse"] - candidate_metric_v189["rmse"],
    })
folds_v189 = pd.DataFrame(fold_rows_v189)

lookup_v189 = summary_v189.set_index("endpoint")
gain_v189 = float(lookup_v189.loc["v168", "rmse"] - lookup_v189.loc["nested_confidence_dose", "rmse"])
rolled_increment_v189 = float(lookup_v189.loc["nested_rolled_control", "rmse"] - lookup_v189.loc["nested_confidence_dose", "rmse"])
top2_oracle_v189 = float(lookup_v189.loc["learned_top2_oracle", "rmse"])
top4_oracle_v189 = float(lookup_v189.loc["learned_top4_oracle", "rmse"])
checks_v189 = {
    "nested_gain_ge_0p03": gain_v189 >= .03,
    "positive_folds_ge_4": int((folds_v189["gain"] > 0).sum()) >= 4,
    "minimum_fold_gain_ge_minus0p02": float(folds_v189["gain"].min()) >= -.02,
    "aligned_beats_rolled_ge_0p02": rolled_increment_v189 >= .02,
    "learned_top2_oracle_lt_6": top2_oracle_v189 < 6.0,
    "learned_top4_oracle_lt_5p5": top4_oracle_v189 < 5.5,
}
selector_pass_v189 = bool(all(checks_v189[key] for key in (
    "nested_gain_ge_0p03", "positive_folds_ge_4",
    "minimum_fold_gain_ge_minus0p02", "aligned_beats_rolled_ge_0p02",
)))
pairwise_investment_v189 = bool(
    checks_v189["learned_top2_oracle_lt_6"] or checks_v189["learned_top4_oracle_lt_5p5"]
)
verdict_v189 = {
    "experiment": "canonical_block_selector_consumption_v189",
    "stage": "read_only_nested_consumption_and_learned_shortlist_gate",
    "selection_or_training_count": 0,
    "base": "V168 canonical full-row endpoint",
    "source_scores": "V188 fivefold crossfit",
    "analytic_global_alpha_oracle": analytic_alpha_v189,
    "nested_selections": selection_rows_v189,
    "summary": summary_v189.to_dict("records"),
    "folds": fold_rows_v189,
    "values": {
        "nested_gain": gain_v189,
        "aligned_increment_vs_rolled": rolled_increment_v189,
        "learned_top2_oracle_rmse": top2_oracle_v189,
        "learned_top4_oracle_rmse": top4_oracle_v189,
    },
    "checks": checks_v189,
    "selector_pass": selector_pass_v189,
    "pairwise_investment_pass": pairwise_investment_v189,
    "elapsed_minutes": (time.time() - STARTED_V189) / 60.0,
    "next": (
        "PASS SELECTOR -> freeze nested confidence dose"
        if selector_pass_v189 else
        "PASS SHORTLIST -> replace RMSE regression with pairwise/listwise ranking"
        if pairwise_investment_v189 else
        "FAIL -> learned ordering does not preserve the oracle; close block selector"
    ),
}
summary_v189.to_csv(OUT_V189 / "summary.csv", index=False)
folds_v189.to_csv(OUT_V189 / "folds.csv", index=False)
pd.DataFrame(selection_rows_v189).to_csv(OUT_V189 / "nested_selections.csv", index=False)
(OUT_V189 / "verdict.json").write_text(json.dumps(verdict_v189, indent=2, default=str), encoding="utf-8")
print("\nV189 SUMMARY")
print(summary_v189.to_string(index=False), flush=True)
print("\nV189 FOLDS")
print(folds_v189.to_string(index=False), flush=True)
print("\nV189 FINAL VERDICT")
print(json.dumps(verdict_v189, indent=2, default=str), flush=True)
print("saved:", OUT_V189, flush=True)
