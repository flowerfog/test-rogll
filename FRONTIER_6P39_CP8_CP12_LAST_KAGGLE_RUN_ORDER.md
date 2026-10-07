# Frontier 6.39 + CP12-aware CP8-last

This is a separate submission variant. Do not stack it with the no-CP12 CP8
cell.

## Required Kaggle inputs

- `frontier_6p39_cp8_bundle_v7.joblib` exported by
  `frontier_6p39_cp8_export_bundle_v7_cell.py`
- the historical `strong_base_projected_oof.csv` dataset consumed by the CP12
  side-curve cell

## Run order

1. Run the original 6.39 notebook through the cell that writes the immutable
   `submission_sp45_learned_w*.csv` core.
2. Run the complete `frontier_6p39_cp12_sidecurve_prep_v8_cell.py` cell. It
   saves `frontier_6p39_core_before_cp.csv` and
   `submission_controlpoint_before_gold.csv`, then restores the core.
3. Continue every enabled original notebook cell through its final
   `submission.csv`, including hidden overlays and package gates.
4. Run `frontier_6p39_cp8_cp12_last_kaggle_v19_cell.py` as the final writer.
5. Submit `/kaggle/working/submission.csv`. Run no later submission writer.

## Required final message

The v19 cell must print:

- `placement: cp12_feature_then_cp_last_increment`
- `cp12_role: auxiliary feature only; never the prediction parent`
- `additive_decomposition_max_error` no greater than `1e-9`
- `FINAL WRITER`

The named final artifact is
`/kaggle/working/submission_frontier_6p39_cp8_cp12_last_increment.csv`.

