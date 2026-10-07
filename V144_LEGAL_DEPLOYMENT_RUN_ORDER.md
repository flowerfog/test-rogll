# V144 Legal Deployment

## AutoDL bundle

1. Confirm that V144's `crossfit_oof.parquet` and `crossfit_coefficients.csv`
   still exist under `oof_exports/final_consensus_sequence_coefficient_full5_v144/`.
2. Run `final_consensus_sequence_coefficient_bundle_v144_cell.py`. It recovers
   from those artifacts automatically after a kernel restart; do not rerun the
   full five-fold V144 experiment.
3. Upload `/root/autodl-tmp/final_consensus_sequence_coefficient_bundle_v144.zip`
   as a Kaggle dataset.

## Kaggle inference

1. Run the exact parent pipeline.
2. Run `rawdata_lattice_coarse_all773_kaggle_v69_cell.py`.
3. Run `rawdata_lattice_coarse_adaptive_kaggle_v77_cell.py`.
4. Run `rawdata_lattice_consensus_center_kaggle_v116_cell.py`.
5. Run `final_consensus_sequence_coefficient_kaggle_v144_cell.py`.
6. Submit `/kaggle/working/submission_v144_sequence_coefficient.csv`.

V144 is legal because hidden-well inference uses only the V69/V77/V116 paths,
horizontal MD/Z/GR/TVT_input, and typewell TVT/GR. True TVT is used only to
construct coefficient labels for the 773 training wells during bundle fitting.

V144 failed its strict CV gate: pooled gain was 0.0169 ft and Folds 1 and 3
were negative. Treat this as a one-slot score candidate, not a replacement for
the frozen 6.342 submission.
