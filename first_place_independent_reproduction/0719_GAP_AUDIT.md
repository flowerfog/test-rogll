# 0719 first-place gap audit

## Reproduction status

The package is a partial clean-room reconstruction of the disclosed 0719 Default
branch. It is not yet a behavioral, literal, or metric-verified reproduction.
The augmentation path remains unverified, and PF and XY are simplified inferred
implementations rather than reproduced engines.

## Confirmed defects in the previous run

1. The fixed input contract was duplicated in `config.py` and `data.py`, which
   allowed the channel list to drift. It now has one source of truth.
2. Typewell GR calibration used every visible prefix row. The disclosed method
   uses only visible rows within `TVT0 +/- 100 ft`, aggregates on a 0.25-ft TVT
   grid, and blends calibrated/original GR by 0.7/0.3.
3. Earlier builds used an incorrect `z_diff` candidate surface. The disclosed
   channel is the suffix rowwise `Z[i] - Z[i-1]`, downsampled and broadcast over
   the state axis.
4. Earlier builds supervised visible-prefix bins. The disclosed target mask is
   suffix-only.
5. Earlier builds entered the complete timm RGB stem. The disclosed network
   uses a custom full-resolution ResBlock, `(2,4)` average pooling, a 1x1
   32-to-96 projection, and then the four ConvNeXt stages.
6. Earlier builds used the wrong fold construction. The disclosed protocol is
   20 XY KMeans regions (seed 0), then fivefold stratification with repeat seeds
   33, 34, and 35.
7. The cosine scheduler was stepped once per epoch. The disclosed scheduler is
   stepped after every optimizer update with `T_max = epochs * steps_per_epoch`.
8. Training did not use `drop_last=True`, and validation used batch 16 instead
   of the disclosed batch 32.
9. Validation checkpoint selection incorrectly applied Savitzky-Golay smoothing.
   The reference selects checkpoints on raw suffix predictions (`apply_smooth=False`).
10. Older code used the wrong GR penalty and padding behavior. The corrected
    contract uses squared normalized GR error clipped at 25, suffix-only targets,
    and right-aligned short prefixes.
11. The largest remaining mismatch is augmentation. The previous code sampled
    TVT increments from the same well in 32-256-row blocks and added generic
    Gaussian noise. The 0719 branch uses other-well suffix trends, 1024/2048/4096
    diff blocks with 0.45/0.40/0.15 probabilities, 16-row joins, TVT-bias-v2 GR
    residual simulation, and a 0.75 application probability.
12. The previous MD augmentation only changed MD coordinates. The disclosed
    augmentation resamples every horizontal array to 0.7-1.3 times its length.
13. The previous GR affine transform changed horizontal and typewell GR together
    with much larger noise. The disclosed transform changes typewell GR only,
    with gain std 0.03, offset std 2, and mutually exclusive smooth/sharpen arms.
14. Reverse-path, residual-shift, tail-cut, channel masking, and sequence masking
    use the disclosed probabilities, but their seeded numerical behavior has not
    been shown equivalent. They remain a blocking reproduction gap.
15. The first cross-well simulation patch omitted the bounded whole-well
    vertical translation (`z_shift_range=40`) and cancelled synthetic faults by
    applying the opposite change to Z. The corrected code translates TVT/Z and
    visible TVT input together within the typewell support margin, while faults
    remain sparse geological-surface jumps. This implementation is still an
    approximation until a same-input seeded parity audit passes.

## Not defects after source verification

- The current local channel names differ from author names, but their 0719
  order is equivalent and ends with `z_diff`.
- The effective 0719 ConvNeXt BN overrides are `eps=1e-4`, `momentum=0.1`, and
  `track_running_stats=True`. The model-class defaults differ, but the active
  configuration overrides them.
- Computing the expected offset in feet and then normalizing is algebraically
  equivalent to taking the expectation on the normalized state axis.

## Additional corrected defects

- Geographic folds previously clustered anchor-point X/Y with `n_init=10`.
  The disclosed split clusters unscaled whole-well horizontal mean X/Y and
  uses the sklearn `n_init="auto"` behavior. Old fold CSVs are therefore
  incompatible.

## Consequence for existing results

The previous 13-15 ft validation curves used an incompatible fold file, seed 0,
and disabled augmentation, so they cannot be compared directly with the 5.091
reference. Also, source fold 0 is a slow-learning fold: its archived RMSE is
about 13.33 at epoch 15 and 10.19 at epoch 20, before reaching 5.63 at epoch 80.
An epoch-18 value around 14 is therefore not by itself proof of a broken model,
but the unresolved augmentation and split gaps still invalidate a parity claim.

## Remaining blockers

- Formal training remains blocked until the 0719 augmentation behavior and fold
  input contract pass their parity gates.
- Numerical PF/FFBSi and XY implementation details remain inferred. They cannot
  be called literal reproductions without exact disclosed values or independent
  metric parity.
