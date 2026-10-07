# 9th Place Solution Analysis

## Reported result

| Validation / leaderboard | RMSE |
|---|---:|
| Plain four-member NNLS, strict CV | 5.56 |
| Eight-member banded ensemble, strict CV | 5.28 |
| Plain ensemble, private LB | 6.503 |
| Banded ensemble, private LB | 6.251 |

The public leaderboard favored the plain blend, while strict CV selected the
banded blend that was materially better on the private leaderboard. This is a
direct example of why public-LB optimization was unsafe in this competition.

## Core mechanisms

### Autoregressive local classifier

- Builds a `9 x 184 x 41` cost-volume image.
- Predicts one of 41 TVT offset lanes spanning approximately +/-30 ft.
- Uses dilated CNN blocks followed by a bidirectional GRU.
- Trains on random chunks with anchor noise and prior walks so training matches
  autoregressive inference rollout.
- Averages six chunk-length and phase configurations.
- Reported member OOF: 7.38 ft.

### Whole-well U-Nets

- Represents each complete well as a candidate-TVT-by-MD image.
- Uses 384 vertical bins at 0.75 ft and 640 compressed MD columns, plus history.
- Predicts a categorical TVT posterior for every MD column.
- ResNet-34 U-Net consumes signed/absolute GR matching costs, trajectory
  geometry, and the known-TVT history mask.
- Decodes the expected candidate row rather than a hard argmax.

### Physical augmentation

- Vertical-shift and horizontal-flip augmentation.
- Path-relief scaling modifies the target path while preserving residual
  structure.
- Thickness-warp applies a smooth monotone warp to the typewell depth axis and
  recomputes the matching costs.
- These are physically consistent training examples, not image-space noise.

### Entropy-weighted TTA

- Runs ten views per well.
- Weights posterior views using their mean column entropy.
- Reported improvement for the main U-Net: 6.33 to 5.86 OOF.

### Ensemble and validation

- Three independent fold seeds times five GroupKFold folds: 15 checkpoints.
- Fits non-negative no-intercept blends separately in seven distance bands.
- Calibrates blend shrinkage from cross-member prediction spread.
- Uses chain OOF, held-fold combiner OOF, and stricter per-foldseed validation.

## Comparison with our experiments

### What was genuinely similar

- V135/V173/V202 created candidate-state or cost-volume representations.
- V202b proved that the true state was frequently inside the model support:
  top-32 oracle RMSE 5.86 and top-64 oracle RMSE 2.57.
- V198 explored posterior averaging, but only over an existing path bank.
- V238 created artificial prefixes to reduce the train/inference mismatch.

### What we did not reproduce

- We never trained a high-capacity whole-well ResNet-34 U-Net on the full cost
  surface.
- V202 used a small block model and then attempted explicit reranking; it did
  not learn full-well dense segmentation.
- We did not use path-relief or thickness-warp augmentation with cost channels
  recomputed after each physical transformation.
- We did not train 15 seed-structured checkpoints and average their posterior
  distributions.
- We did not perform entropy-weighted TTA on categorical cost-volume models.
- Our blending was generally global or based on a small set of endpoints; it
  was not seven-band NNLS plus spread calibration under strict foldseed OOF.

Therefore, "we tried cost maps" is not equivalent to having tried this method.
Our experiments established that candidate support existed, but our models did
not have the architecture, augmentation, scale, or validation structure used
to turn that support into a stable prediction.

## Main postmortem conclusion

The ninth-place solution did not discover a hidden scalar feature. It improved
generalization by preserving the complete candidate distribution, training a
whole-well image model, generating physically meaningful distribution shift,
and averaging a large seed/fold ensemble. Our pipeline repeatedly compressed
the cost surface into paths or summary features and then tried to repair or
route those outputs. That compression discarded the ambiguity information the
winning family used directly.

## Reproduction order

1. Reconstruct the exact whole-well cost-volume tensor and verify truth-lane
   coverage on the existing five well folds.
2. Train one small U-Net without augmentation as a parity baseline.
3. Add expected-row decoding and vertical-shift TTA.
4. Add path-relief augmentation and verify its benefit on a held fold.
5. Add thickness-warp augmentation and verify toe-distance improvement.
6. Scale to five folds and multiple fold seeds only after the single-model
   mechanics pass.
7. Reproduce strict held-fold blending before comparing leaderboard behavior.

