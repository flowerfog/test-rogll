# ROGII 6.47 Mainline Baseline

Effective date: 2026-07-23

## Frozen source

- Notebook: `E:\rogii-luck-is-all-you-need.ipynb`
- SHA256: `249019DFC160599AD7639C66173024DBA73A1C8CC6C8CE86FB73B0E65F502B71`
- Reported public LB: `6.47`
- Local status: exact full-stack fold-clean OOF not yet reconstructed
- Status: current deployment incumbent and default parent for local experiments

The former 6.781 notebook is retained only as a historical comparator. CP12 and geometric Neighbor are not part of the 6.47 mainline and must not be restored without an isolated ablation.

## Frozen scoring path

1. Ridge and SP45 selector blend: `0.30 / 0.70`.
2. SP45 projection: degree `3`, projection dose `0.75`.
3. Learned/FLE likelihood PF: prefix GR scale clipped to `[10, 60]`, then multiplied by `1.30`.
4. Learned/FLE path: `0.60 learned model + 0.40 likPF(scale_5)`.
5. SP45 and learned/FLE blend: `0.60 / 0.40`.
6. No second projection after the SP45/FLE blend.
7. Guarded same-well contact override.
8. Balanced visible-prefix calibration, cut fractions `(0.50, 0.65, 0.75)`, profile output selected.
9. Model-package gated correction with selected maximum weight `0.00425`; automatically disabled when prediction-difference p95 exceeds `25 ft`.
10. PF seed-branch midpoint hedge: strength `0.60`, minimum minor mass `0.25`, separation range `[4, 40] ft`, movement cap `2 ft`.
11. Final ID/order/finite-value audit.

Default-disabled components remain non-scoring:

- Bimodal selector detector
- Heel-calibrated selector hedge
- FFT denoising
- Sequence matcher
- Full-stack CV diagnostics

## Experiment contract

Every new candidate must:

1. Start from the frozen 6.47 notebook, not from 6.781 or an `ultimate-pf` notebook.
2. Change one scoring mechanism at a time.
3. Preserve an untouched baseline candidate in `/kaggle/working` before overwriting `submission.csv`.
4. Report changed wells, changed rows, mean/p95/max absolute movement and the final SHA256.
5. Treat the three visible test wells as implementation controls only; never infer hidden activation or benefit from them.
6. Keep any hard-coded visible-well experiment out of the hidden-scoring hypothesis.
7. Select parameters and promote changes only from fold-clean matched OOF confirmation. Diagnostic/oracle improvements and LB movement alone are insufficient.
8. Use LB only once after local promotion, as an external transfer check; never tune the next parameter from LB feedback.

## Required local protocol

The 6.47 local baseline must be regenerated on AutoDL with complete physical wells as groups:

1. Use the frozen five-fold whole-well assignment from `foldclean_full_fleongg_lgb_v0/well_metadata.csv` when available.
2. For each outer fold, fit every learned model, normalizer and spatial reference bank on the other four folds only.
3. Recreate each validation well with its organizer-provided `TVT_input` visible prefix and hidden suffix. Validation truth is used only for final scoring.
4. Disable same-ID train lookup for validation wells. The guarded same-well route is an implementation control for the three visible test examples, not a CV feature.
5. Generate fold-clean SP45 and learned/FLE endpoints before applying the frozen degree-3 projection and `0.60 / 0.40` blend.
6. Replay balanced visible-prefix calibration using only each validation well's visible prefix.
7. Replay PF seed-branch statistics from that validation well's 128 seed trajectories; no truth may enter branch qualification.
8. Exclude the full-data model package until it has a fold-clean retrained counterpart. Its tiny gated dose must not contaminate the baseline CV.

Mandatory reports:

- exact row count and exact SSE
- pooled RMSE and MAE
- per-fold pooled RMSE
- median, p75, p95 and maximum well RMSE
- `0-500`, `500-1000`, `1000-2000`, and `>2000 ft` RMSE
- worst-20% well SSE share
- changed rows/wells and movement statistics for each late-stage module
- matched no-op/control output hashes

## Current evidence

- The 6.594 source pipeline and 6.47 pipeline are code-identical except for learned/FLE likelihood `GS x 1.30`.
- The A24 `+1 ft` continuation targets visible example well `00e12e8b`; it is not a hidden-score mechanism.
- Transplanting `GS x 1.30` into the old 6.781 CP12/Neighbor chain via A1+A2+B scored `6.806`, so cross-pipeline transfer is closed.
- This does not weaken `GS x 1.30` inside the frozen 6.47 pipeline where it is part of the incumbent result.

## Next experiment order

1. Build the exact fold-clean learned/FLE `GS x 1.30` endpoint on AutoDL.
2. Build the matching fold-clean SP45 degree-3 endpoint on the same folds and rows.
3. Assemble the frozen `0.60 / 0.40` baseline and replay balanced Gold plus seed-branch.
4. Freeze the resulting full-stack OOF score as the local incumbent.
5. Only then scan one parameter family at a time. Keep an untouched confirmation fold or use nested selection.
6. Submit to LB only after a local gain survives confirmation.
