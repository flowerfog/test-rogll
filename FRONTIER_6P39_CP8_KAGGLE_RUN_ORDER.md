# Frontier 6.39 + CP8 Kaggle Run Order

This is the submission path validated by the current OOF work. Neighbor is
excluded. Historical CP12 is built only as an auxiliary CP8 feature; it never
becomes the parent prediction.

## One-time AutoDL export

1. Keep the completed `frontier_6p39_cp8_parent_audit_v2` outputs.
2. Run the complete cell in `frontier_6p39_cp8_export_bundle_v7_cell.py`.
3. The output must print exactly `features: 126`, `wells: 773`, and five model
   members. A 127-feature bundle is invalid.
4. Upload this directory as a Kaggle dataset:

   `/root/autodl-tmp/oof_exports/frontier_6p39_cp8_parent_audit_v2/deployment/`

The required file is `frontier_6p39_cp8_bundle_v7.joblib`.

## Kaggle notebook

1. Run `rogii-public-score-frontier-lab-visuals.ipynb` normally through cell 39.
   Stop immediately after cell 39 prints the selected SP45/learned blend and
   writes `submission.csv`.
2. Run the complete code from
   `frontier_6p39_cp12_sidecurve_prep_v8_cell.py` as one new cell.
3. Run the complete code from
   `frontier_6p39_cp8_kaggle_infer_v7_cell.py` as the next new cell.
4. Continue the original notebook at cell 41. Cell 41 is read-only. Run cell 42
   (guarded same-well override), then cells 44 through 49 in their original
   order, subject to their existing enable flags.
5. Submit `/kaggle/working/submission.csv` only after the final audit cell passes.

## Required files and states

Before the CP12 side-curve cell:

- `sp45_projection_submission.csv`
- `learned_trajectory_submission.csv`
- `submission_sp45_learned_w0.60.csv` (or the weight selected by the notebook)
- notebook globals `sub_1` and `sub_2`
- Kaggle input dataset containing `strong_base_projected_oof.csv`

Before the CP8 inference cell:

- `frontier_6p39_core_before_cp.csv`
- `submission_controlpoint_before_gold.csv`
- `ridge_pp_submission.csv`
- `selector_submission.csv`
- `fleongg_pretrained_submission.csv`
- `sp45_projection_submission.csv`
- uploaded `frontier_6p39_cp8_bundle_v7.joblib`

## Cells to skip

- `frontier_6p39_cp_parent_prep_v7_cell.py`
- every old standalone CP12 cell
- every Neighbor cell
- any Gold/guarded/visible-prefix cell before CP8 inference

The CP12 side-curve cell reconstructs the historical 0.55 SP45/FLE projected
parent internally, runs CP12 there, saves only the CP12 curve, and restores the
6.39 core. This preserves the OOF feature bloodline used by the CP8 model.
