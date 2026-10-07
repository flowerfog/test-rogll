# Evidence Matrix Triage

## Result

The 251-experiment scan found 24 failed experiments with at least one scalar whose JSON key contains both
`oracle` and `rmse` and whose value is below 6.0. This is evidence that the target trajectories are often
present in a representation or candidate set. It is not evidence that 24 legal predictors were overlooked.
Several of the 24 values are not pooled endpoint metrics at all: they are near/far distance bands,
median-well summaries, parity gaps, or reconstruction diagnostics. The revised matrix reports
`sub6_primary_oracle` separately and uses only that field for the unresolved list.

## Classification

### Representation ceilings

- `absolute_rate_jump_representation_audit_v137c`
- `shape_datum_residual_decomposition_v0`
- `current_best_error_budget_v132`
- `final_consensus_residual_oracle_v123`

These use truth to project onto a grid, basis, datum, slope, or residual family. Subsequent legal coefficient
or trajectory predictors failed to infer the required latent variables.

### Truth-selected support ceilings

- Local cost-profile and edge/cliff audits
- V3/V6/V55/V60 candidate-bank selectors and hard-negative experiments
- V75/V163 regional-surface candidate audits
- V78/V80b/V107/V128 residual or block candidates
- V184/V185 rescue and routing audits
- HN2/HN23/HN26 alias and tournament audits
- V36 tail-SGM dose oracle

These answer whether a good answer exists among candidates when truth chooses it. Their legal selectors,
routers, doses, or regressors did not transfer. Replacing the selector model does not add information.

### Consumed continuation

`stride_figure1_posterior_parity_v3` passed only a first-hidden-segment posterior-geometry test. It was then
connected to full-sequence decoding:

- V5 repaired evidence scaling but Fold2 remained 13.144 RMSE.
- V9 raw STRIDE reached 12.331 RMSE over 773 wells; cross-fit blend gain was 0.090 and Fold5 lost 0.495.
- V12 local-reference STRIDE reached 12.754 RMSE; blend gain was 0.097 with only three positive folds.

Therefore V3 is closed, not an unconsumed sub-6 route.

V36 is the other JSON verdict that requested continuation. Its reported 3.971 value is only
`near_le1000_rmse`, not full-row RMSE. V37 performed the requested legal confidence-gate experiment; overall
RMSE remained about 8.3 and aligned predictions were effectively indistinguishable from the shuffled control.
V36 is also fully consumed.

## Decision

No entry in `failed_with_sub6_oracle.csv` justifies another model run. A new run requires a new legal
inference-time observable or a demonstrated validation defect, as specified by `NEW_ENGINE_LINEAGE_FIREWALL.md`.
With the currently inventoried inputs and two days remaining, the evidence supports freezing the best
deployable submission rather than assigning a new version to another oracle-consumption attempt.
