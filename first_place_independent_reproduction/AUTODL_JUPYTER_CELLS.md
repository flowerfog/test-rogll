# AutoDL Jupyter cells

The package is self-contained. It never reads author competition artifacts.
`model.safetensors` may be a generic ImageNet ConvNeXt-small checkpoint.

## 1. Install and validate

```python
from pathlib import Path
import os, subprocess, sys

PROJECT = Path('/root/autodl-tmp/first_place_independent_reproduction')
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
subprocess.run([sys.executable, '-u', '-m', 'first_place_repro.validate'], check=True)
```

## 2. Tiny-overfit gate

```python
DATA = '/root/autodl-tmp/rogii-wellbore-geology-prediction'
OUT = '/root/autodl-tmp/first_place_independent_outputs_contract_v6'
ENCODER = '/root/autodl-tmp/model.safetensors'  # generic ImageNet pretraining

cmd = [
    sys.executable, '-u', '-m', 'first_place_repro.train',
    '--data-root', DATA, '--output-root', OUT,
    '--version', '0719_V1', '--repeat', '0', '--fold', '1', '--seed', '7',
    '--epochs', '100', '--overfit-wells', '16', '--batch-size', '16',
    '--workers', '0', '--encoder-checkpoint', ENCODER, '--run-tag', 'overfit_gate',
]
subprocess.run(cmd, check=True)
```

The row RMSE must fall below 2 ft. Otherwise the formal run is blocked.

## 3. Corrected 0719 fold discrepancy gate

This preserves the disclosed batch-16/BF16 semantics. It is a learning-direction
check, not a reference metric.

```python
cmd = [
    sys.executable, '-u', '-m', 'first_place_repro.train',
    '--data-root', DATA, '--output-root', OUT,
    '--version', '0719_V1', '--repeat', '0', '--fold', '1', '--seed', '7',
    '--epochs', '25', '--schedule-epochs', '300',
    '--batch-size', '16', '--workers', '4', '--encoder-checkpoint', ENCODER,
    '--activation-offload', '--validation-batch-size', '4',
    '--run-tag', 'contract_v11_fold_gate',
]
subprocess.run(cmd, check=True)
```

This uses the complete fold, not a reduced-well smoke run. Reference logs use
zero-based fold labels and show substantial early-fold dispersion at epoch 25:
8.58, 5.87, 7.86, 8.33, and 7.73 ft. Do not compare one local fold to the best
reference fold by label. The first hard discrepancy gate is epoch 80: the
matching reference folds are 5.63, 4.70, 6.24, 5.61, and 6.03 ft. Stop if the
corrected run remains above 7.3 ft at epoch 80.

## 4. Full 0719 member

```python
cmd = [
    sys.executable, '-u', '-m', 'first_place_repro.train',
    '--data-root', DATA, '--output-root', OUT,
    '--version', '0719_V1', '--repeat', '0', '--fold', '1', '--seed', '7',
    '--epochs', '300', '--batch-size', '16', '--workers', '4',
    '--encoder-checkpoint', ENCODER, '--run-tag', 'formal',
]
subprocess.run(cmd, check=True)
```

Do not launch the remaining 89 members until this member approaches the
reported `0719_V1` behavior.

## 5. Infer and gate the remaining version family

The writeup discloses the seven-version lineage but not every numerical recipe.
The non-0719 recipes are therefore labeled inferred in
`remaining-model-matrix.csv`. Start with one member per positive-weight version:

```python
cmd = [
    sys.executable, '-u', '-m', 'first_place_repro.train_all',
    '--data-root', DATA,
    '--output-root', '/root/autodl-tmp/first_place_remaining_outputs',
    '--fold-file', f'{OUT}/folds.csv',
    '--encoder-checkpoint', ENCODER,
    '--versions', '0801_V1',
    '--stage', 'gate', '--epochs', '80',
    '--batch-size', '16', '--validation-batch-size', '4', '--workers', '4',
    '--activation-offload', '--resume', '--execute',
]
subprocess.run(cmd, check=True)
```

Start with `0801_V1` because it uses the already validated Default inputs and
does not depend on inferred PF/XY engines. After that member passes, replace
the version with `0729_V3`, `0724_V1`, `0801_V2`, and finally `0803_V2`, one
at a time. `0802_V2` is excluded because its locked final weight is zero.

After each branch has a credible single-member curve, run its 15 members. For
example, the Default-v2 branch is:

```python
cmd = [
    sys.executable, '-u', '-m', 'first_place_repro.train_all',
    '--data-root', DATA,
    '--output-root', '/root/autodl-tmp/first_place_remaining_outputs',
    '--fold-file', f'{OUT}/folds.csv',
    '--encoder-checkpoint', ENCODER,
    '--versions', '0801_V1', '--stage', 'formal',
    '--batch-size', '16', '--validation-batch-size', '4', '--workers', '4',
    '--activation-offload', '--resume', '--execute',
]
subprocess.run(cmd, check=True)
```

`repeat=0..2` and `fold=1..5` produce 15 members. The source base seed remains
7; repeat and fold are converted internally into the member-specific seeds.
