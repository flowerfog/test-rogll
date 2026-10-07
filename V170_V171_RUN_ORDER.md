# V170 deployment and V171 CV<6 gateway

## AutoDL validation and bundle

The current kernel should still contain the completed V166 and V167 globals.

1. Run `historical_analog_parent_lineage_v170_cell.py`.
2. Inspect `V170 PARENT SUMMARY`.
3. Continue only if `consensus_center` has `deployable_gate=True`.
4. Run `historical_analog_deployment_bundle_v170_cell.py`.
5. Upload `/root/autodl-tmp/historical_analog_deployment_bundle_v170.zip` as a Kaggle dataset.

Do not deploy V168 when V170 does not pass on `consensus_center`.

## Kaggle deployment

Run in this exact order:

1. Complete 6.377 parent pipeline.
2. `rawdata_lattice_coarse_all773_kaggle_v69_cell.py`
3. `rawdata_lattice_coarse_adaptive_kaggle_v77_cell.py`
4. `rawdata_lattice_consensus_center_kaggle_v116_cell.py`
5. `historical_analog_kaggle_v170_cell.py`

Submit `/kaggle/working/submission.csv`. The V170 cell preserves the exact V116
parent as `/kaggle/working/submission_v116_parent_before_v170.csv`.

## Independent CV<6 route

Run `functional_surface_trajectory_engine_v171_cell.py` after V166 in the same
AutoDL kernel. Proceed to nested blending only when `investment_pass=true`.
The pre-registered gateway requires:

- standalone RMSE <= 10.5
- detrended error correlation with V144 <= 0.60
- analytic global blend gain >= 0.20
- positive analytic blend gain on at least four folds

V171 consumes no V76/V144 prediction as a feature. It is a separate trajectory
generator, not another correction or routing layer.
