# Frontier 6.39 + CP8-noCP12 Kaggle Run Order

This is the paired LB test of CP8. It excludes Neighbor and the historical
CP12 side curve. It preserves the published Q0522 rule after rebasing its
obsolete source hash on the CP8 pre-Q artifact. A no-Q candidate is saved too.

## Dataset to upload

Upload the complete AutoDL directory:

`/root/autodl-tmp/oof_exports/frontier_6p39_cp8_nocp12_matched_v9/deployment/`

It must contain:

`frontier_6p39_cp8_nocp12_bundle_v9.joblib`

## Exact Kaggle order

Do not identify stages by notebook cell number. The source-only notebook has
seven extra Markdown cells, so its JSON cell indices are shifted even though
all 48 executable code cells are identical.

1. Run the frontier notebook from the start through the code cell beginning
   `from pathlib import Path as _FinalBlendPath`. Stop only after its output
   prints `wrote final submission.csv from submission_sp45_learned_w0.60.csv`
   and `/kaggle/working/submission_sp45_learned_w0.60.csv` exists. This is JSON
   cell 39 in `rogii-public-score-frontier-lab-visuals.ipynb`, but JSON cell 46
   in `rogii-public-score-frontier-lab-visuals-4c1212.ipynb`.
2. Insert and run the complete contents of
   `frontier_6p39_cp8_nocp12_kaggle_v9_cell.py`.
   Before continuing, require both
   `submission_frontier_6p39_cp8_nocp12_before_overlays.csv` and
   `frontier_6p39_cp8_nocp12_deployment.json` to exist.
3. Resume the original notebook with the guarded same-well overlap cell.
4. Run the original balanced visible-prefix cell.
5. Run the original model-package correction cell.
6. Run the original midpoint hedge cell.
7. Skip the old Q0522 cell and run
   `frontier_6p39_cp8_nocp12_q0522_rebased_v11_cell.py` instead.
8. Skip the old final audit cell because it is bound to the old pre-Q hash.
9. Run `frontier_6p39_cp8_nocp12_final_audit_v10_cell.py`.
10. Submit `/kaggle/working/submission.csv` only if the final cell prints
    `"ready": true`.

## Do not run

- Any historical CP12 preparation/inference cell.
- Any Neighbor cell.
- The old hash-bound Q0522 cell. Use the rebased guarded cell instead.
- The old hash-bound final audit.
- CP8 twice.

## Score interpretation

The Q0522 candidate is directly comparable with the 6.39 submission because
the same fixed hidden-ID rule is retained. The cell also saves
`submission_frontier_6p39_cp8_nocp12_no_q0522.csv`; that file is diagnostic
only and can be submitted separately if a clean pre-Q comparison is desired.
