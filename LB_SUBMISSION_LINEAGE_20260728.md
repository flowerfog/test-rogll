# LB Submission Lineage - 2026-07-28

This record groups public-LB scores by the actual final-cell recipe. Scores from
different tail-cell lineages must not be pooled as repeated runs of one recipe.

## Only Neighbor

Final recipe: 6.377-family mainline with the Neighbor tail, without the 0.05
Physics tail dose.

| Run | Public LB |
|---:|---:|
| 1 | 6.472 |
| 2 | 6.430 |
| 3 | 6.454 |
| 4 | 6.381 |

- Mean: `6.4343`
- Median: `6.4420`
- Sample standard deviation: `0.0394`
- Best observed: `6.381`

## Neighbor + 0.05 Physics

Final recipe: the same Neighbor lineage followed by the frozen 0.05 raw-GR
Physics tail cells.

| Run | Public LB |
|---:|---:|
| 1 | 6.413 |

- Only one confirmed run; no recipe mean or variance can be estimated.
- Best observed: `6.413`

## Current Interpretation

There is no longer evidence of a Physics deployment gain. Its only confirmed
score, `6.413`, is `0.0213` below the four-run Only-Neighbor mean but `0.0320`
worse than the best Only-Neighbor run. With one Physics run and stochastic
submissions, that difference is not identifiable as a treatment effect.

Therefore:

1. Keep `Only Neighbor` as the current deployed recipe and `6.381` as its best
   observed score.
2. Archive the Physics tail without further tuning or submission work.
3. Use the paired PF repeatability audit only as a reproducibility finding.
4. Continue with a genuinely new modelling direction.

## P5 PF Repeatability Audit

The paired 24-well, six-seed audit measured the pre-CP8/Neighbor learned core:

- Mean raw `pf_ancc` stochastic standard deviation: `3.4084 ft`
- Mean final core prediction stochastic standard deviation: `0.2784 ft`
- Mean pairwise prediction RMS difference: `0.7355 ft`
- Mean pairwise signed-error correlation: `0.99528`
- Six-seed mean gain versus median single seed: `0.0159 ft`
- Six-seed mean loss versus the best sampled seed: `0.0847 ft`

Multi-seed feature averaging failed its gate. The PF randomness is real but is
mostly suppressed by the learned core and does not provide a useful ensemble
gain. The accepted action is deterministic per-well seeding, not multi-seed
retraining. There is no longer a measured multi-run LB advantage attributable
to the 0.05 Physics tail.
