# V195 Seed-1 V69+V77 Kaggle Run Order

## Purpose

Produce one reproducible submission from the exact V69 All773 coarse plus V76
adaptive-dose family that previously reached Public LB 6.342. The only changed
variable is the upstream PF seed. Seed 1 is frozen from V97 and was evaluated
without tuning on the disjoint V98 48-well panel.

## Exact order

1. Start a fresh Kaggle rerun.
2. Run the original mainline notebook through its final `build_well` definition.
   Stop before the cell that calls `main()`.
3. Insert and run `kaggle_pf_seed1_patch_v195_cell.py`.
4. Run the original `main()` and all cells required to create the same parent
   submission used by the 6.342 run. Do not add experimental overlays.
5. Run `rawdata_lattice_coarse_all773_kaggle_v69_cell.py`.
6. Immediately run `rawdata_lattice_coarse_adaptive_kaggle_v77_cell.py`.
7. Immediately run `kaggle_pf_seed1_finalize_v195_cell.py`.
8. Stop the notebook. Do not run V116, V144, V168, or any other overlay.
   Kaggle automatically submits `/kaggle/working/submission.csv`.

The named V195 file is only a byte-identical audit copy. While authoring, the
visible test data contains three example wells (currently 14,151 rows). This is
expected; the Code Competition rerun replaces those files with hidden wells.

## Interpretation

This is a score-focused, one-slot candidate. V98 measured a 0.041-ft gain over
seed 0 and 0.064-ft far-tail gain, but the result failed its statistical gate.
Do not infer a mechanism from one Public LB result and do not tune another seed
from the Public leaderboard.
