# ROGII Wellbore Geology Prediction

Research and competition code for predicting hidden true vertical thickness
(TVT) trajectories in horizontal wells.

The repository contains two related tracks:

- `first_place_independent_reproduction/`: a clean-room, source-independent
  reproduction of the disclosed whole-well alignment approach. It includes
  the data contract, state-space alignment model, training, validation,
  PF/XY auxiliary branches, ensemble scaffolding, and single-model inference.
- Root-level `*_cell.py` and `*_v*.py` files: dated experiment cells and
  audit scripts from the tabular, particle-filter, spatial, and sequence
  research tracks.

## Reproducibility

Competition data, trained weights, cached features, generated submissions,
large archives, and experiment outputs are intentionally excluded from Git.
See the package README and `first_place_independent_reproduction/AUTODL_JUPYTER_CELLS.md`
for the AutoDL/Jupyter workflow.

The clean-room package is designed to run without the author's private source
code or model files. An encoder checkpoint may be supplied separately when
available; it is never stored in this repository.

## Main package

```text
first_place_independent_reproduction/
  first_place_repro/       implementation
  tests/                    contract and smoke tests
  README.md                package-specific instructions
  pyproject.toml           test/package metadata
```

Typical local checks:

```bash
python -m pytest first_place_independent_reproduction/tests
```

This is an archived research repository rather than a turnkey Kaggle
submission. Paths, datasets, and accelerator settings must be configured for
the local environment.
