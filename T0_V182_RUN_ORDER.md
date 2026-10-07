# T0 V182 Run Order

## Kaggle deterministic replay

1. Restart the Kaggle kernel.
2. Run `kaggle_t0_determinism_bootstrap_v182_cell.py` before importing numpy,
   numba, LightGBM, or torch.
3. Run the frozen submission pipeline without changing parameters.
4. Run `kaggle_t0_finalize_override_audit_v182_cell.py` after the final
   `/kaggle/working/submission.csv` is written.
5. Rerun steps 1-4 with the same seed and uploaded reference submission. The
   second audit must report `repeatability.within_0p005=true`.

The audit distinguishes same-ID overlap from successful guarded override. The
visible test audit must not be used to infer hidden overlap unless Kaggle makes
the private rerun log available.

## AutoDL full-row OOF freeze

Run `canonical_endpoint_registry_v181.py` where the V144 and V168 artifacts
exist. Require `strict_pass=true`, then freeze the selected endpoint path,
prediction column, key hash, prediction hash, row count, well count, and pooled
full-row RMSE. Sampled 192-node and equal-well RMSE remain diagnostics only.

