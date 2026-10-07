# Archive (3) vs clean-room audit

## Scope

`archive (3)` is a literal artifact package for the author's `0719_V1`
sequence model, not merely a checkpoint:

- exact runtime configuration: `cfg.pkl`
- training/inference source: `seq_NN_*.py`
- 15 trained models: `models.pkl` (3 repeats x 5 folds)
- OOF predictions and training log
- test predictions and submission

The package reports OOF RMSE **5.0910** after averaging the 15 OOF members.
It is one first-place ensemble component, not the later seven-version final
ensemble.

## Conclusion

The existing `first_place_cleanroom` is not a behavioral reproduction of
`0719_V1`. It implements a related formulation, but differs in the tensor
contract, pretrained initialization, encoder/decoder graph, target
construction, feature normalization, loss units, fold protocol and
postprocessing. Its RMSE is therefore not directly comparable with the
author's RMSE.

## P0 differences

These differences are large enough to explain validation RMSE remaining in
the 9-15 range.

### 1. Tensor axes are reversed

- Author: `[channel, horizontal_bin=345, typewell_state=400]`.
- Clean-room native: `[channel, typewell_state=400, horizontal_bin=345]`.
- Author applies softmax over the final, 400-state axis.
- Clean-room was originally trained with the reversed layout by default.

This is not a cosmetic transpose. ConvNeXt has anisotropic stem stride
`(2, 4)`, so reversing the axes changes which physical dimension is reduced
fourfold.

### 2. Pretraining was interpreted backwards

The stored configuration has `unet_pretrained=None`. In the author's
`make_model()`, `None` resolves to **True during training**. It resolves to
False only when loading trained models for submission.

The clean-room code explicitly rejected an encoder checkpoint and trained the
encoder from scratch. That removed a disclosed and active initialization.

### 3. The model graph is different

The author uses timm's actual ConvNeXt-small stages, converts their LayerNorm
modules to FP32 BatchNorm, adds a full-resolution residual stem, then decodes
the native stage skips with bilinear residual up-blocks.

The clean-room model manually approximates ConvNeXt blocks, replaces native
stage downsampling with separate AvgPool/1x1 blocks, adds lateral projections
to `(32,64,128,128)`, and uses a different decoder/head. The author's
`updown_mode_2d='non-param'` does not mean replacing ConvNeXt's native
encoder downsampling as the clean-room implementation assumed.

### 4. Regression target and loss use different units

Author:

- predicts `TVT - TVT_anchor`
- downsample target is a masked **mean** per 32-row bin
- normalizes with `mean=1.591691, std=15.853182`
- converts the state expectation to this normalized target space
- computes Huber in normalized units

Clean-room:

- decodes absolute TVT states
- compares expected absolute TVT directly with absolute truth TVT
- computes Huber in feet

The nominal Huber weight is the same, but it multiplies a different numerical
objective.

### 5. Alignment targets are different

Author:

- bin truth: downsampled mean TVT
- state centers: `[-99.75, ..., 99.75]`
- target: Gaussian `exp(-0.5 * (distance / 1.25)^2)`

Clean-room:

- bin truth: final TVT row in each bin
- state centers: `[-100.0, ..., 99.5]`
- target: Laplace `exp(-abs(distance) / 1.25)`

### 6. Input channels are not numerically reproduced

A same-well deterministic audit on `000d7d20` found:

| Channel | MAE | RMSE | Correlation |
|---|---:|---:|---:|
| tw_gr | 0.198 | 0.245 | 0.984 |
| gr | 0.103 | 0.218 | 0.962 |
| gr_abs_diff | 0.241 | 0.468 | 0.990 |
| gr_slope | 0.166 | 0.343 | 0.976 |
| gr_quadratic_a | 0.218 | 0.428 | 0.983 |
| gr_quadratic_b | 0.233 | 0.475 | 0.976 |
| gr_quadratic_c | 0.884 | 1.349 | 0.561 |
| gr_quadratic_rmse | 0.272 | 0.718 | 0.731 |
| tw_tvt_rel | 0.366 | 0.423 | 1.000 |
| seen_tvt_rel | 0.379 | 0.384 | 0.999 |
| tw_seen_tvt_abs_diff | 0.367 | 0.424 | 0.989 |
| z_diff | 0.256 | 0.846 | **0.036** |

The author uses stored population statistics for every channel. The
clean-room uses approximate constants such as `(GR-80)/40`, `/100`, and
`/40`. The shape features also use a different fit/normalization contract.
`z_diff` is not even the same effective signal.

### 7. Typewell calibration differs

- Author: visible-prefix GR is binned at 0.25 ft and blended
  `0.7 * visible + 0.3 * original typewell`.
- Clean-room: bins at 0.5 ft and blends 0.5/0.5.

## P1 differences

These may not individually explain RMSE 15, but matter for reaching the
reference 5.091.

- Author CV is geographic stratified five-fold with 20 XY clusters, repeated
  three times using split seeds 33, 34 and 35. Clean-room uses a fixed fold
  table with three model seeds.
- Author uses ImageNet pretrained ConvNeXt, BF16, channels-last memory format,
  non-deterministic kernels, AdamW, gradient clip 10, EMA 0.99, minimum 170
  epochs and patience 30.
- Author expands each 32-row prediction bin piecewise-constantly before
  smoothing. Clean-room linearly interpolates bin-center predictions.
- Author smooths the inferred surface `TVT + Z`, then subtracts `Z`.
  Clean-room smooths TVT directly.
- Author's state centers are half-bin centers, not left edges.
- The exact simulation and augmentation order is materially more complex than
  the clean-room approximation.

## Reference learning curve

The supplied log disproves the idea that a correct fold should remain around
15:

- repeat0/fold1: 15.29 at epoch 1
- 13.75 at epoch 10
- 6.28 at epoch 20
- 5.36 at epoch 35
- 4.95 at epoch 45
- best raw 4.57 at epoch 160

If a run remains near 15 after epoch 20-40, its implementation is not following
the supplied `0719_V1` behavior.

## Reproduction status

- Reference artifact level: **literal**, for `0719_V1`.
- Existing clean-room level: **prototype/partial reconstruction**.
- Exact/source-derived components in clean-room: broad 2D alignment
  formulation, dimensions, channel names, optimizer family, headline loss
  weights.
- Missing or incorrect components: the P0 and P1 items above.
- Validation performed: source/config/log audit and deterministic same-well
  16-channel parity test.
- Reference metric: 5.0910 OOF.
- Clean-room observed metric: previously 9-15 depending on run; not comparable.

## Correct next action

Do not tune the current clean-room network. First run the supplied
`0719_V1` package in inference mode and verify that `models.pkl` reproduces
its stored test output. Then either:

1. use the supplied source as the literal implementation; or
2. port it module by module, requiring same-well tensor parity and trained-model
   output parity before retraining.

The machine-readable channel audit is
`postmortem/archive3_cleanroom_parity.json`; its generator is
`tools/audit_archive3_parity.py`.
