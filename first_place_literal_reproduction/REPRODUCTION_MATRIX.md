# First-place reproduction matrix

This package separates literal artifacts from source-derived reconstruction.
It never substitutes a smaller model, fewer wells, fewer folds, or fewer
ensemble members without an explicit command and label.

| Component | Evidence | Implementation status |
|---|---|---|
| 0719_V1 source | Supplied `seq_NN_*.py` | Literal byte copy, SHA256 locked |
| 0719_V1 cfg | Supplied `cfg.pkl` | Literal, SHA256 locked |
| 0719_V1 weights | Supplied `models.pkl`, 3 repeats x 5 folds | Literal, SHA256 locked |
| 0719_V1 OOF | Supplied `oof_df.pqt` and log | Literal, reference RMSE 5.0910 |
| 0719_V1 test output | Supplied submission/details | Literal parity gate |
| 2D alignment geometry | Write-up and source | Exact: 345 horizontal bins x 400 typewell states |
| ConvNeXt U-Net | Write-up and source | Exact author implementation |
| CE + Huber + GR loss | Write-up and source | Exact author implementation |
| Typewell calibration | Write-up and source | Exact author implementation |
| Augmentations | Write-up and source | Exact author implementation and ordering |
| PF | Write-up and source | Complete source implementation: bin updates, low-probability jumps, resampling, profile mixture, FFBSi |
| XY prior | Write-up and source | Complete source implementation: anisotropic local WLS plane, regularization, singular fallback, diagnostics |
| Six published versions | Write-up | Version/branch mapping and reported CV exact; only 0719 cfg/weights were supplied |
| Later five version configs | Not supplied | Source-derived from 0719 cfg plus disclosed branch channels; undisclosed augmentation deltas inherited from 0719 |
| Later five model weights | Not supplied | Must be retrained with exact author runtime |
| General ensemble weights | Write-up | Exact published values, normalized because displayed weights sum to 0.98 |
| No-XY ensemble weights | Write-up | Exact published values |
| XY quality routing | Write-up | Four disclosed diagnostics, training 95th-percentile rule |

The six published members are:

| Version | Branch | General | No XY | CV | Artifact status |
|---|---|---:|---:|---:|---|
| 0719_V1 | Default | 0.07 | 0.40 | 5.09 | Supplied literal model |
| 0729_V3 | PF | 0.00 | 0.20 | 5.53 | Retrain |
| 0801_V1 | Default | 0.07 | 0.40 | 5.16 | Retrain |
| 0724_V1 | XY | 0.28 | 0.00 | 4.86 | Retrain |
| 0801_V2 | XY | 0.28 | 0.00 | 4.80 | Retrain |
| 0803_V2 | XY + PF | 0.28 | 0.00 | 5.00 | Retrain |

The reported full ensemble metrics are CV 4.627, public LB 5.980 and private
LB 5.639. They are validation targets, not claims made by an untrained
reconstruction.
