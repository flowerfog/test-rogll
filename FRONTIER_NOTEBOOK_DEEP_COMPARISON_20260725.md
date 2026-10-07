# Frontier notebook deep comparison

Compared files:

1. `E:\rogii-public-score-frontier-lab-visuals.ipynb`
2. `E:\rogii-public-score-frontier-lab-visuals-4c1212.ipynb`
3. `E:\rogii-public-score-frontier-lab-visuals (1).ipynb`

## File identity

| File | Bytes | SHA256 |
|---|---:|---|
| `rogii-public-score-frontier-lab-visuals.ipynb` | 467919 | `da460516b1aeda7404b1b1f2abb5702c418c31f64f4088ecaf81ca08fc795438` |
| `rogii-public-score-frontier-lab-visuals-4c1212.ipynb` | 337465 | `488fe12d9538bb4953cdc17c2999aeb89209cebf81d30c267a13ee58cbcde847` |
| `rogii-public-score-frontier-lab-visuals (1).ipynb` | 467919 | `da460516b1aeda7404b1b1f2abb5702c418c31f64f4088ecaf81ca08fc795438` |

Files 1 and 3 are byte-for-byte identical. File 2 is a different notebook container.

## Executable-code comparison

All three notebooks contain 48 code cells. After preserving code-cell order and joining their exact source text, all three have the same SHA256:

`9710e42514daffc4d4912c8b65fe1f84c986b5cb9bc1e7a42a4951b8773b6c04`

The ordered list of all 48 individual code-cell hashes is also identical. Therefore:

- no model parameter differs in the current source;
- no profile setting differs;
- no cell was added, removed, or reordered in executable-code order;
- the final blend, overlap, visible-prefix, model-package, midpoint, Q0522, and audit code is identical.

The active profile in all three is `vp_balanced_modelpkg_005`, with SP45 blend weight `0.60`, 128 selector PF seeds, and the same post-processing switches.

## Container and run-history differences

| Property | Files 1 and 3 | File 2 (`4c1212`) |
|---|---|---|
| Total cells | 55 | 62 |
| Markdown cells | 7 | 14 |
| Code cells | 48 | 48 |
| Stored outputs | 28 | 0 |
| Code execution counts | 48 populated | 0 populated |
| Python metadata | 3.12.13 | 3.12.12 |
| Papermill run | 2026-07-23 | 2026-07-07 |
| Notebook format | 4.5 | 4.4 |

File 2 contains seven extra explanatory markdown cells. They shift physical JSON cell positions by seven after the inserted documentation, but do not change executable order.

Important position example:

| Stage | Code index | Files 1/3 physical index | File 2 physical index |
|---|---:|---:|---:|
| Final SP45/FLE blend | 37 | 39 | 46 |
| Midpoint hedge | 42 | 46 | 53 |
| Q0522 | 43 | 47 | 54 |
| Final audit | 44 | 48 | 55 |

Any external CP8 cell placed by physical cell number can therefore land before the final blend in file 2 and be overwritten. Placement must be identified by the code cell beginning `from pathlib import Path as _FinalBlendPath`, not by notebook cell number.

## Why identical source can still produce different submissions

The final SP45 selector PF is explicitly seeded (`seed=0..127`) and is intended to be deterministic. However, the FLE feature-generation path defines ANCC/Z particle filters that use Numba `np.random` without a per-well seed. When the notebook rebuilds FLE features from mounted models rather than loading a precomputed full submission, this endpoint can vary between runs. FLE carries 40% of the final blend.

The notebooks also resolve artifacts at runtime from mounted dataset paths, including:

- `/kaggle/input/datasets/fleongg/rogii-claude-models-pub`
- `/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts`
- optional model-package roots

The notebook file does not pin Kaggle dataset version IDs or artifact hashes. The same path can therefore refer to different mounted dataset revisions across runs. Python/Numba versions also differ between the recorded July runs.

Consequently, `6.39` and `6.465` can come from the same current source while using different runtime random states or mounted artifact revisions. The source files alone cannot identify which score belongs to which generated CSV.

## Recorded July 23 output

Files 1 and 3 contain a complete July 23 run. It records:

- FLE loaded from `/kaggle/input/datasets/fleongg/rogii-claude-models-pub`;
- final fixed blend from `submission_sp45_learned_w0.60.csv`;
- final audited `submission.csv` SHA256 `8328487738fbb7e1e89c733b80367ced2281e6dd19f4746804adfd77a923071d`;
- notebook metadata names `6.568` as the verified public parent and marks the Q0522 branch as not competition-submitted.

This metadata does not prove the LB score of the July 23 CSV. Only Kaggle's submitted output artifact can establish that mapping.

## Final verdict

The files are different as notebook artifacts, but not as current executable programs:

- files 1 and 3 are exact duplicates;
- file 2 has different documentation, execution metadata, format version, and no saved outputs;
- all executable source is exactly identical;
- score differences must come from generated-output lineage, runtime randomness, mounted artifact versions, or a temporary cell not saved into these files.

To identify the true `6.39` lineage, compare the actual `submission.csv` downloaded from that Kaggle run, its SHA256, the run's mounted dataset version IDs, and the run log showing whether FLE was rebuilt or loaded as a precomputed submission.
