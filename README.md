# ROGII Wellbore Geology Prediction baseline

This repository contains one deliberately modest, reproducible baseline for the
ROGII Wellbore Geology Prediction competition. It prioritizes an honest
well-grouped validation score over model complexity.

## Hypothesis

At the last known heel row, `TVT_input` supplies a reliable TVT anchor. A first
trajectory estimate can be extrapolated from that anchor using the local
heel relationship between TVT and Z. A small histogram gradient-boosting model
can then learn the remaining correction from relative trajectory geometry,
absolute location, horizontal-well GR, and the paired typewell GR curve.

The model never uses a horizontal well's target `TVT` to create features. That
column is read only after feature construction to form the training residual.
The typewell `TVT` coordinate is part of the reference curve supplied at
inference and is therefore allowed.

## Data contract

Set `ROGII_DATA_ROOT` to a directory with this layout:

```text
<root>/
  train/
    <well_id>__horizontal_well.csv
    <well_id>__typewell.csv
  test/
    <well_id>__horizontal_well.csv
    <well_id>__typewell.csv
  sample_submission.csv
```

Horizontal files require `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input`.
Training horizontal files additionally require a complete `TVT` target.
`TVT_input` must contain a non-empty known prefix followed by a non-empty
missing suffix. Typewell files require `TVT` and `GR`. Rows must be ordered by
non-decreasing MD because submission IDs use the original zero-based row
number.

The code reads `ROGII_DATA_ROOT`. It defaults to `/root/autolab/data` only when
the environment variable is absent; an explicitly configured bad path fails
instead of silently switching datasets.

## Validation

The provided training censoring boundary is treated as the simulated test
condition: the known `TVT_input` heel prefix is retained and the complete toe
suffix is masked. Wells, rather than rows, are deterministically shuffled into
five folds. For each fold the model is fitted only on other wells and predicts
all masked rows of the held-out wells.

The final score is computed as
`sqrt(sum((prediction - TVT)^2) / number_of_held_out_toe_rows)` across all
folds. It is not an unweighted average of per-well RMSE values. To bound memory
and training time, fitting uses evenly spaced toe rows per training well;
scoring still includes every held-out toe row.

## Commands

Create an environment and run the lightweight synthetic tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the same first experiment configured for AutoDL:

```bash
export ROGII_DATA_ROOT=/root/autolab/data
python train_eval.py --folds 5 --max-train-rows-per-well 512 --max-iter 160
```

On Windows PowerShell, point at the observed local data without copying it:

```powershell
$env:ROGII_DATA_ROOT = "E:\rogii-wellbore-geology-prediction"
python train_eval.py --folds 5 --max-train-rows-per-well 512 --max-iter 160
```

Fit on all training wells and create predictions aligned to the supplied
sample submission. Keep the output outside this source repository:

```bash
python predict.py --output /root/autolab/output/rogii_suffix_hgb_v1.csv \
  --max-train-rows-per-well 512 --max-iter 160
```

Both entry points print a JSON summary of the configuration and result.
