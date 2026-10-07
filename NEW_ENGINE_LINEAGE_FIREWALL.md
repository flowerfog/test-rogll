# New Engine Lineage Firewall

## Objective

Build a deployable endpoint with canonical full-row, complete-well, five-fold pooled RMSE below 6.0.

## Hard lineage rule

Previous endpoints may be used only as read-only comparison baselines in the final metric table.
They must not be used as model inputs, labels, teachers, candidate generators, priors, residual targets,
routers, blend parents, confidence features, or post-processing parents.

This prohibition includes PF/CP/Neighbor artifacts, historical candidate banks, and every V-series
endpoint through V190b, including V66, V76, V127, V144, V168, V169, and V187.

## Permanently closed routes

- Dose calibration, endpoint blending, routing, fallback, and residual correction.
- Existing candidate-bank ranking, selection, barycenters, and block selection.
- Prefix-state prediction of future residual direction.
- Historical analog retrieval and metadata-only analog transfer.
- Spatial or transductive surface reconstruction from existing endpoint lineage.
- STRIDE/regime decoders and the attempted correlation CRF family.
- Direct absolute-trajectory TCN variants and their projection routers.
- Horizon-oracle distillation into the existing endpoint family.

A closed route cannot be reopened by renaming features, changing the learner, adding a dose, or using a
different old parent.

## New-route admission gate

A proposed route must satisfy all of the following before it receives a version number:

1. It consumes only raw legal competition inputs available for the target well at inference time.
2. It introduces a genuinely new observable or state representation, not a transformation of an old prediction.
3. Its validation uses complete physical wells and the canonical full-row pooled RMSE.
4. Its split construction is deterministic, explicit, and audited for well, typewell, row, and ordering leakage.
5. A small gateway establishes either standalone transfer or a useful candidate oracle before a full five-fold run.
6. Failure ends the route; no selector, router, blend, or post-processing rescue is allowed.

## Current evidence boundary

- Best frozen deployable comparison endpoint: V168, full-row pooled RMSE 7.756316.
- The apparent V172 RMSE 4.945 was ordering leakage and is invalid.
- V187 demonstrates a block oracle below 4, but V188/V189 show that its choice is not observable with the
  attempted legal evidence. It is not a model path.
- V190/V190b show that cross-well transductive surface information has negligible globally consumable gain.
- The inspected public notebook reports strict GroupKFold model RMSE around 10.4, not 5.0. Its three-well
  contact override is test-specific and cannot support a general CV claim.
- The public notebook constructs formation KNN and dense ANCC imputers from all 773 training wells before
  GroupKFold. Validation-fold formation labels therefore enter feature generation.
- Direct use of the train-only formation columns reconstructs hidden TVT with about 0.006 ft RMSE after a
  per-well prefix datum fit. This explains how leaked validation can look spectacular; those columns do not
  exist for competition test wells.
- Exact normalized typewell duplicates cover only about 4.4% of training wells (13 groups), so exact legal
  typewell reuse is not a global CV<6 route.
- V191 learned blockwise derivatives of S=TVT+Z without historical predictions. Its Fold2/Fold3 RMSE was
  19.99/19.73, with far-tail RMSE 24.65 on Fold3. The agreement shows a stable observability/integration
  limitation rather than selection noise. Structural block-gradient forecasting is permanently closed and
  must not be rescued with an old endpoint, spatial field, neighbour route, or prefix correction.
- V192 learned one datum-free global potential F(X,Y) from within-well differences of S=TVT+Z. Reversing
  each held-out XY path degraded Fold3 RMSE from 24.68 to 259.60, proving that the model used real spatial
  information. However, Fold2/Fold3 RMSE was 34.27/24.68 despite low training loss. A single globally shared
  structural potential does not transfer across held-out spatial/geological domains and is permanently closed.
- V193 conditioned the datum-free potential on the complete legal typewell GR curve. Fold2 improved from
  V192's 34.27 to 26.63 RMSE, but remained far outside the 8-ft gateway at its best epoch. Fold3 was never
  loaded. Conditional spatial-potential families are closed and cannot be repaired or blended.
- V194 used training-only Geology labels to learn separate horizontal- and typewell-GR facies probability
  models, then performed legal Fold45-to-Fold2 registration. Typewell facies recognition transferred
  (macro-F1 0.658), but horizontal facies recognition did not (macro-F1 0.156). More importantly, the
  retrieval support itself was unusable: top-1 RMSE was 84.71 ft and even the truth-selected top-32 oracle
  was 57.50 ft. Facies-probability registration is permanently closed; a sequence decoder cannot recover a
  path that is absent from its candidate support.
- The 251-experiment evidence matrix found 24 failed experiments containing a reported sub-6 oracle metric.
  None is an unconsumed deployable route. They are representation ceilings, truth-selected candidate/block
  ceilings, or branches whose legal learner subsequently failed. In particular, the promising V3 STRIDE
  first-segment posterior was consumed by the V5 scale repair and complete five-fold V9/V12 runs. Raw and
  local-reference STRIDE finished at 12.331 and 12.754 RMSE; their cross-fit gains over SP45 were only 0.090
  and 0.097 and were fold-unstable. The V3 entry must not be reopened as an allegedly missed branch.
- The other apparent continuation, V36, was a 3.971 near-1000-ft band metric rather than pooled RMSE. V37
  already ran its requested legal dose gate; overall RMSE stayed near 8.3 and aligned performance was
  indistinguishable from shuffle. V36 is consumed and closed.

## Required next decision

Do not allocate another version number to a model assembled from the currently exhausted observables. A new
route is admissible only after identifying either (a) an inference-time competition input not represented in
the inventory above, or (b) a concrete leakage or metric mismatch in the canonical five-fold protocol. The
first experiment must be a cheap information-support audit with a truth-selected oracle below 6 ft; otherwise
the route closes before model training. Without one of these two findings, the evidence does not support a
legal canonical-CV-below-6 claim.
