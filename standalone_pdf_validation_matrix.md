# Standalone PDF Validation Matrix

## Fixed protocol

- Objective: a standalone, no-neighbor model with pooled five-fold RMSE <= 5.5 ft.
- Split: the existing random GroupKFold by well; fold 1 is the common gateway.
- Forbidden model inputs: PF/beam predictions, strong, strong_bias, CP12, Gold,
  Neighbor, neighbor-well curves, and hidden/LB feedback.
- Allowed inputs: the target well's complete horizontal GR/geometry, its visible
  TVT prefix, and its assigned typewell.
- Gateway metrics: pooled RMSE, pooled MAE, median/p95 well RMSE, anchor baseline,
  no-GR intervention, and shuffled-typewell intervention.
- Architecture pass: fold-1 pooled RMSE <= 7.0 ft and both causal interventions
  degrade RMSE by at least 0.30 ft.
- Target pass: fold-1 pooled RMSE <= 5.5 ft. Run at least one second seed before
  expanding to all five folds.
- Final pass: five-fold pooled RMSE <= 5.5 ft, no negative causal control, and no
  single fold or seed responsible for the result.
- A failed implementation closes only its exact representation/decoder pair.

## Completed gateways

| Family | Best honest result | Status | Scope of conclusion |
|---|---:|---|---|
| Absolute-S RankSDF U-Net/DP v0-v2 | 15.801 | Closed | Fixed absolute-S panel and tested decoders only |
| Raw-window GBDT | 13.408 | Closed | Supplied raw-window table representation |
| Trajectory/geology-state model | 14.109 | Closed | State labels carry signal but do not yield a precise trajectory |
| TVT CorrNet v2 | 14.286 | Closed | Original dense CorrNet recipe |
| Heel-conditioned CorrNet v4 | 10.124 | Closed | Causal GR signal exists; architecture is too weak |
| Full-well 2-D U-Net | 15.554 | Closed | Image U-Net representation; all mean/MAP/DP decoders failed |
| Handcrafted zero-learning GR DP | Worse than baseline | Closed | Handcrafted emissions and fixed DP only |
| Synthetic local matcher | Small local gain only | Closed | Local descriptor is not a standalone trajectory model |
| Tree/listwise candidate selectors | No robust gain | Closed | Tested shallow candidate selectors |
| Shape basket + DSSE selector | Oracle about 7.09; selector 0 gain | Closed | Current basket/selector, not all learned path scorers |
| Whole-well dual-encoder neural CRF (S1) | Viterbi 18.045; optimistic posterior 14.969 | Closed | Current emission encoder, CRF training, and audited decoders |
| Prefix meta-adapted CorrNet (S2) | Full adapter 10.110 vs frozen 10.124 | Closed | Low-dimensional deployment-legal prefix adaptation; signal is real but too small |
| Axial/RAFT iterative CorrNet (S3) | 9.687 vs centre 10.944 | Closed | Better than v4 by about 0.44 ft, but above the 7.0 architecture gate and causal controls were below 0.30 ft |
| Calibrated synthetic pretraining on S3 (S4) | 9.788 vs matched scratch 9.847 | Closed | Strong causal use of GR/typewell, but only +0.060 ft and no architecture-level rescue |
| Physical-S path bank + deep listwise scorer (S5) | 11.883; pruned oracle 6.142 | Closed | GR/typewell causality is strong and posterior beats MAP, but deployable path selection remains far from the menu oracle |

## Mandatory remaining gateways

| ID | PDF-level hypothesis | Minimal honest implementation | State |
|---|---|---|---|
| S1 | Whole-well learned GR matching with structured decoding | Dual 1-D encoders, explicit matching matrix, anchor-conditioned differentiable CRF, Viterbi/posterior mean | Closed: 14.969 best optimistic audit |
| S2 | Prefix meta-adaptation resolves per-well acquisition/domain shift | Shared CorrNet plus a small heel-conditioned adapter fitted only on masked visible-prefix tasks | Closed: +0.014 full dose; +0.058 optimistic dose |
| S3 | Axial/RAFT-style iterative correspondence is stronger than a U-Net | Coarse-to-fine cost volume with axial context and recurrent offset refinement | Closed: 9.687; no-GR +0.237, shuffled-typewell +0.136 |
| S4 | Synthetic data helps once applied to a complete-well model | Calibrated residual-block generator pretraining for S1/S3, followed by real-only fine-tuning | Closed: +0.060; no-GR +1.483, shuffled-typewell +0.652 |
| S5 | Whole-path learned scoring can recover candidate oracle | Non-tabular sequence encoder with listwise loss over complete no-neighbor paths and posterior mean output | Closed: 11.883 vs oracle 6.142; no-GR +3.448, shuffled-typewell +4.129 |

## Execution order

1. S1 one-fold gateway.
2. S2 one-fold gateway, regardless of S1 outcome.
3. S3 one-fold gateway, regardless of S1/S2 outcome.
4. S4 on the stronger of S1 and S3; if both fail, run a bounded pretrain/fine-tune
   gateway anyway and record the interaction.
5. S5 one-fold gateway using a no-neighbor candidate bank.
6. Run second seeds for every architecture-pass model.
7. Expand surviving models to five folds and compare standalone pooled RMSE.

## Artifact discipline

- Persist only the best checkpoint, row-level validation predictions, by-well
  metrics, interventions, training history, and verdict JSON.
- Delete regenerable panel/logit caches after each verdict.
- Do not blend with the 6.882 chain until a standalone model has a complete
  five-fold result.
