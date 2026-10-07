# Single-model Kaggle submission

This path evaluates exactly one trained `0719_V1` checkpoint. It does not load
PF, XY, another fold, another seed, or an existing submission.

## 1. Select and export the full-fold checkpoint on AutoDL

```python
from pathlib import Path
import json
import shutil
import torch

OUTPUT_ROOT = Path('/root/autodl-tmp/first_place_independent_outputs_v11')
EXPORT_ROOT = Path('/root/autodl-tmp/kaggle_single_0719')
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

candidates = []
for path in OUTPUT_ROOT.rglob('best.pt'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    metrics = checkpoint.get('metrics', {})
    if checkpoint.get('version') != '0719_V1':
        continue
    if checkpoint.get('contract_version') != '0719_contract_v11_pageable_activation_offload':
        continue
    if int(metrics.get('rows', 0)) < 500_000:
        continue  # Reject tiny-overfit and reduced-well gates.
    candidates.append((float(metrics['row_rmse']), path, checkpoint))

if not candidates:
    raise RuntimeError('No complete-fold 0719 v11 best.pt was found')

candidates.sort(key=lambda item: item[0])
for rmse, path, checkpoint in candidates:
    print({
        'row_rmse': rmse,
        'epoch': checkpoint.get('epoch'),
        'selected_model': checkpoint.get('selected_model'),
        'path': str(path),
    })

rmse, source, checkpoint = candidates[0]
destination = EXPORT_ROOT / 'best.pt'
shutil.copy2(source, destination)
(EXPORT_ROOT / 'checkpoint.json').write_text(json.dumps({
    'source': str(source),
    'row_rmse': rmse,
    'epoch': checkpoint.get('epoch'),
    'selected_model': checkpoint.get('selected_model'),
    'version': checkpoint.get('version'),
    'contract_version': checkpoint.get('contract_version'),
}, indent=2), encoding='utf-8')
print('UPLOAD THIS DIRECTORY AS A PRIVATE KAGGLE DATASET:', EXPORT_ROOT)
```

Upload `/root/autodl-tmp/kaggle_single_0719` as one private Kaggle dataset.
Upload `first_place_single_submission_v2.zip` as another private dataset.

## 2. Run inference in a Kaggle notebook

Enable a GPU and attach the competition data plus the two private datasets.

```python
from pathlib import Path
import os
import subprocess
import sys
import zipfile

working = Path('/kaggle/working/first_place_single_runtime')
working.mkdir(parents=True, exist_ok=True)

code_zips = list(Path('/kaggle/input').rglob('first_place_single_submission_v2.zip'))
if len(code_zips) != 1:
    raise RuntimeError({'message': 'expected exactly one code zip', 'found': [str(x) for x in code_zips]})
with zipfile.ZipFile(code_zips[0]) as archive:
    archive.extractall(working)

projects = list(working.rglob('first_place_independent_reproduction/pyproject.toml'))
if len(projects) != 1:
    raise RuntimeError({'message': 'project extraction failed', 'found': [str(x) for x in projects]})
project = projects[0].parent
os.chdir(project)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', '.'], check=True)

checkpoints = list(Path('/kaggle/input').rglob('best.pt'))
if len(checkpoints) != 1:
    raise RuntimeError({
        'message': 'attach a dataset containing exactly one best.pt',
        'found': [str(x) for x in checkpoints],
    })

data_candidates = [
    Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
    Path('/kaggle/input/rogii-wellbore-geology-prediction'),
]
data_root = next((path for path in data_candidates if (path / 'test').is_dir()), None)
if data_root is None:
    matches = [path.parent for path in Path('/kaggle/input').rglob('sample_submission.csv') if (path.parent / 'test').is_dir()]
    if len(matches) != 1:
        raise RuntimeError({'message': 'competition data root not found', 'matches': [str(x) for x in matches]})
    data_root = matches[0]

output = Path('/kaggle/working/submission.csv')
command = [
    sys.executable, '-u', '-m', 'first_place_repro.submit_single',
    '--data-root', str(data_root),
    '--checkpoint', str(checkpoints[0]),
    '--sample-submission', str(data_root / 'sample_submission.csv'),
    '--output', str(output),
]
print('RUN:', ' '.join(command), flush=True)
subprocess.run(command, check=True)

import pandas as pd
submission = pd.read_csv(output)
print(submission.shape)
print(submission.tvt.describe())
print('SUBMIT:', output)
print('AUDIT:', output.with_suffix('.audit.json'))
```
