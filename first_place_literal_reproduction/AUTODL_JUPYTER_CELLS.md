# AutoDL Jupyter cells

Upload this directory to `/root/autodl-tmp/first_place_literal_reproduction`
and keep the supplied author package at `/root/autodl-tmp/archive (3)`.

## Cell 1 - environment

```python
from pathlib import Path
import os, sys, zipfile

ROOT = Path('/root/autodl-tmp')
PROJECT = ROOT / 'first_place_literal_reproduction'
if not (PROJECT / 'literal_repro').is_dir():
    archives = sorted(ROOT.glob('first_place_literal_reproduction*.zip'))
    if not archives:
        raise FileNotFoundError({
            'message': 'Upload first_place_literal_reproduction.zip to /root/autodl-tmp',
            'searched': str(ROOT / 'first_place_literal_reproduction*.zip'),
        })
    with zipfile.ZipFile(archives[-1]) as archive:
        archive.extractall(ROOT)

# Handle zip tools that add one extra top-level directory.
if not (PROJECT / 'literal_repro').is_dir():
    matches = list(ROOT.glob('**/literal_repro/run_full.py'))
    if len(matches) != 1:
        raise RuntimeError({'message': 'could not uniquely locate extracted project', 'matches': [str(x) for x in matches]})
    PROJECT = matches[0].parents[1]

ARCHIVE = Path('/root/autodl-tmp/archive (3)')
DATA = Path('/root/autodl-tmp/rogii-wellbore-geology-prediction')
OUTPUT = Path('/root/autodl-tmp/first_place_literal_outputs')
for required in (PROJECT / 'literal_repro', ARCHIVE, DATA / 'train', DATA / 'test'):
    if not required.exists():
        raise FileNotFoundError(required)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print({'PROJECT': str(PROJECT), 'ARCHIVE': str(ARCHIVE), 'DATA': str(DATA)})
```

Install only if the AutoDL image does not already provide these packages:

```python
import subprocess
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'timm', 'safetensors', 'pyarrow', 'numba',
], check=True, cwd='/root/autodl-tmp')
```

## Cell 2 - audit every supplied artifact

The first pass skips hashing the 3.2 GB model file but checks its exact byte
size. Add `--hash-large` once when transferring machines.

```python
import subprocess
subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.audit',
    '--reference-dir', str(ARCHIVE),
    '--output', str(OUTPUT / 'archive_audit.json'),
], check=True)
```

## Cell 3 - literal 0719 inference

This loads all 15 supplied author models. It is not a smoke model.

```python
subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.run_author', 'infer-0719',
    '--data-root', str(DATA),
    '--reference-dir', str(ARCHIVE),
    '--output-dir', str(OUTPUT / '0719_V1'),
    '--device', 'cuda', '--workers', '4',
], check=True)
```

## Cell 4 - exact output parity

```python
subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.parity',
    '--reference', str(ARCHIVE / 'submission.csv'),
    '--candidate', str(OUTPUT / '0719_V1' / 'submission.csv'),
    '--atol', '1e-6',
], check=True)
```

Only continue if Cell 4 passes. This proves the data path, graph, weights and
postprocessing reproduce the supplied component.

## Cell 5 - generate all six cfgs

```python
subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.make_branch_cfg',
    '--reference-cfg', str(ARCHIVE / 'cfg.pkl'),
    '--output-root', str(OUTPUT),
], check=True)
```

## Cell 6 - retrain one missing version

Run this separately for `0729_V3`, `0801_V1`, `0724_V1`, `0801_V2`, and
`0803_V2`. Each run is the full author protocol: 3 geographic repeats x 5
folds, up to 300 epochs, EMA, BF16, exact augmentations and exact PF/XY code.

```python
VERSION = '0803_V2'
subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.train_from_cfg',
    '--cfg', str(OUTPUT / 'configs' / VERSION / 'cfg.pkl'),
    '--data-root', str(DATA),
    '--output-dir', str(OUTPUT / VERSION),
    '--encoder-checkpoint', '/root/autodl-tmp/model.safetensors',
], check=True)
```

## Cell 7 - final routed ensemble

This deliberately refuses to run until all six `submission.csv` files and the
four disclosed XY-quality statistics are present.

```python
subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.xy_quality',
    '--data-root', str(DATA),
    '--output', str(OUTPUT / 'xy_quality.csv'),
], check=True)

subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.ensemble',
    '--prediction-root', str(OUTPUT),
    '--quality-csv', str(OUTPUT / 'xy_quality.csv'),
    '--output', str(OUTPUT / 'first_place_ensemble.csv'),
], check=True)
```

## Optional single cell - run the complete known solution

This is the same sequence as Cells 2-7 and does not use well, epoch, fold,
seed or branch limits.

```python
subprocess.run([
    sys.executable, '-u', '-m', 'literal_repro.run_full',
    '--data-root', str(DATA),
    '--reference-dir', str(ARCHIVE),
    '--output-root', str(OUTPUT),
    '--encoder-checkpoint', '/root/autodl-tmp/model.safetensors',
    '--workers', '4',
], check=True)
```
