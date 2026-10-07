# V196 Seed-1 Consensus Kaggle Run Order

## Why this route

V196 does not use V144. It combines two frozen findings:

- V98: PF seed 1 beat seed 0 by 0.041 ft on a disjoint 48-well panel.
- V115: V116 top-33 consensus improved exact V76 on Folds 1, 2, and 3 by
  0.043, 0.138, and 0.183 ft; pooled three-fold gain was 0.122 ft.

The consensus is weaker than its shuffled support control on Fold2, so V196 is
a score-focused submission candidate, not proof of a new inference mechanism.

## Exact order in one separate fresh Kaggle notebook version

1. Run the original notebook through its final `build_well` definition, but
   stop before the cell that invokes `main()`.
2. Run `kaggle_pf_seed1_patch_v195_cell.py`.
3. Run the original `main()` and the unchanged parent pipeline.
4. Run `rawdata_lattice_coarse_all773_kaggle_v69_cell.py`.
5. Run `rawdata_lattice_coarse_adaptive_kaggle_v77_cell.py`.
6. Run `rawdata_lattice_consensus_center_kaggle_v116_cell.py`.
7. Run `kaggle_pf_seed1_consensus_finalize_v196_cell.py`.
8. Stop the notebook. Do not run V144, V168, or any later overlay.

The finalizer guarantees that the V116 result is written to the only path that
Kaggle automatically submits:

- `/kaggle/working/submission.csv`

`submission_v196_seed1_v116_consensus.csv` is only a byte-identical archive.
This notebook version creates exactly one submission candidate. It must not run
the V195 finalizer. While authoring, three example test wells are expected; the
hidden Code Competition rerun supplies the actual test wells.
