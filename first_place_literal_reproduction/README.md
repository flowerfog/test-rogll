# First-place literal reproduction

This is the faithful path for the first-place ROGII solution. The supplied
author source is preserved unchanged under `author_src/`; orchestration lives
under `literal_repro/`.

## What is exact

`archive (3)` contains the complete `0719_V1` component: exact source, cfg,
15 trained models, OOF, log and test predictions. The package verifies these
artifacts and reruns their inference literally. It also uses the author's
complete PF, FFBSi and XY-local-plane implementations already present in that
source.

## What must be retrained

The final solution has six published versions. Only `0719_V1` weights and cfg
were supplied. For the other five, this package uses the exact author runtime,
the disclosed version/feature mapping and all known channels. Their undisclosed
version-specific augmentation deltas cannot be recovered from the available
artifacts, so they inherit the literal 0719 defaults and are marked
source-derived in every manifest.

See `REPRODUCTION_MATRIX.md` and `AUTODL_JUPYTER_CELLS.md` before running.
