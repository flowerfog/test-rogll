# T3 V186 Same-Well Override Run Order

T3 is already implemented and has passed its pseudo-rerun controls. Do not
retune its thresholds on the three visible test wells.

## Required uploaded files

- `samewell_crossid_retrieval_audit_v0_cell.py`
- `kaggle_samewell_crossid_guarded_apply_v1_cell.py`
- `kaggle_t0_finalize_override_audit_v182_cell.py`

## Final notebook order

1. Run `kaggle_t0_determinism_bootstrap_v182_cell.py` before the mainline.
2. Run the frozen mainline and all selected prediction stages.
3. Preserve the resulting `submission.csv` as the non-T3 control.
4. Run `kaggle_samewell_crossid_guarded_apply_v1_cell.py` as the final
   prediction-changing cell. It invokes the read-only retrieval audit itself.
5. Require `samewell_crossid_apply_verdict.json` to report:
   - `status=applied`;
   - `fail_closed=true`;
   - no failure;
   - finite submission with unchanged ID order.
6. Run `kaggle_t0_finalize_override_audit_v182_cell.py` after T3.
7. Submit only `/kaggle/working/submission.csv`.

## Frozen guards

- Full transfer: prefix RMSE <= 0.25, GR NCC >= 0.990, overlap >= 0.80.
- Graded transfer: prefix RMSE <= 0.75, GR NCC >= 0.970, overlap >= 0.50,
  retrieval margin >= 0.02, transfer weight 0.85.
- Out-of-range rows retain the current prediction plus the nearest valid
  boundary shift.
- Any exception restores the pre-T3 submission.

## Existing validation

- 96 renamed-full controls: 100% final recall and activation.
- 96 shifted/noisy controls: 100% final recall and activation.
- 96 partial/shifted controls: 100% final recall and activation.
- 96 true-well-excluded impostors: 0% false activation.
- Visible test: 3/3 exact-ID wells and 14,151/14,151 rows applied.

Visible activation is an implementation check only. Hidden overlap and hidden
activation must be read from the private rerun artifacts when available.

