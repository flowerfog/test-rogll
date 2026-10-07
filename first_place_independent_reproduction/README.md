# First-place writeup reproduction (clean room)

This package is currently a partial clean-room reconstruction built from the published
solution description and competition data. It does not read or require author
source code, author configurations, trained competition checkpoints, cached
predictions, or `archive (3)`.

The only optional external checkpoint is the generic ImageNet ConvNeXt-small
pretraining checkpoint. It is not a competition model.

The current corrected 0719 contract is `0719_contract_v11_pageable_activation_offload`. Checkpoints from
earlier package versions are intentionally rejected because their data,
scheduler, and validation contracts are incompatible. See
`0719_GAP_AUDIT.md` for the audited defects.

## Fidelity boundary

- The 0719 formulation, tensor geometry, no-augmentation channel construction,
  ConvNeXt/U-Net structure, loss, and core BF16 training protocol are implemented.
- Augmentation, exact geographic fold inputs, PF/FFBSi, XY local planes, routing,
  final decoding, and ensemble behavior are not yet parity verified. Several are
  still simplified implementations and are marked `missing` in the matrix.
- Numerical details absent from the writeup are isolated in
  `first_place_repro/inferred.py`.
- This is not yet a behavioral reproduction. Training results are diagnostic only
  until every required matrix row is resolved and the metric gate passes.
- The reference target is OOF RMSE 4.627 for the final ensemble and 5.091 for
  `0719_V1`. Those metrics must be reached before calling it validated.

## AutoDL notebook

Use the cells in `AUTODL_JUPYTER_CELLS.md`. The first command runs contract,
data, model, loss, checkpoint, and tiny-overfit gates. Formal training is
blocked until those gates pass.
