# 26th Place Solution: Reproducibility Boundary

## What is disclosed sufficiently

- Image formulation: TVT depth rows by MD columns.
- Main grid: 256 depth bins over a 256 ft window, 1 ft per bin.
- Horizontal grid: `W = 2H = 512` equal MD columns.
- Five channels: typewell GR, horizontal GR, known hint raster, squared GR cost,
  and dZ/dMD.
- Column-wise softmax classification with soft-argmax decoding.
- Stacked two-stage U-Net; stage 2 receives stage 1's detached softened belief
  plane.
- Refeed at inference, with the reported optimum at three rounds for the L/XL
  models.
- MD warp, bed warp, horizontal flip, hint drop, and anchor jitter as legal
  raw-data augmentations.
- EMA decay `0.997`, batch size `8`, and seed ensembling.
- Grouped well split stratified by bucketed TVT excursion range.

## What cannot be reproduced exactly from the writeup

- The synthetic-well generator: trajectory process, fault/unconformity process,
  noise distributions, calibration drift, and curriculum schedule are not
  specified numerically.
- Exact SMP/timm versions, EfficientNet-L/XL/L2 checkpoint names, ImageNet
  preprocessing, decoder details beyond the channel widths, and optimizer
  schedule are not fully specified.
- Exact ranges and probabilities for all five augmentations are absent.
- Exact stacked training schedule, refeed probability, softening temperature
  schedule, and stage loss weights are incomplete.
- The final ensemble member checkpoints and OOF predictions are not provided.
- The spatial-prior seventh-channel model is described but explicitly retired
  due to possible CV leakage; it should not be treated as a clean baseline.

## Practical verdict

An **honest architectural reproduction** is feasible. An **exact numerical
reproduction of 5.038 CV or 6.66 private LB is not currently possible** without
the missing synthetic generator, pretrained checkpoints, and hyperparameters.
Any claim otherwise would be false precision.

## Recommended reproduction order

1. Build the disclosed 5-channel `256 x 512` image with prefix hint raster.
2. Train one ResNet34/SMP U-Net with vanilla CE and soft-argmax.
3. Add EMA, range-stratified GroupKFold, and the five raw-data augmentations.
4. Add detached stage-2 stacking and three inference refeeds.
5. Compare ResNet34 to an available EfficientNet implementation.
6. Only then add approximate synthetic pretraining, clearly labeled as a new
   implementation rather than a faithful reproduction.
7. Build the seven-seed/five-fold ensemble after a single-model held-fold
   result is stable.

