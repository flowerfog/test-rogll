# AutoDL Jupyter validation

Upload `first_place_writeup_reproduction_avgpool_v2.zip` to `/root/autodl-tmp`, then run
the following cells in order. These cells never read the author package.

## Cell 1 - unpack and install the disclosed backbone

```python
from pathlib import Path
import os, shutil, subprocess, sys, zipfile

ROOT = Path("/root/autodl-tmp")
ZIP = ROOT / "first_place_writeup_reproduction_avgpool_v2.zip"
PROJECT = ROOT / "first_place_writeup_reproduction"

# A previous run may have left Jupyter inside PROJECT. Move out before replacing it.
os.chdir(ROOT)
if PROJECT.exists():
    shutil.rmtree(PROJECT)
with zipfile.ZipFile(ZIP) as archive:
    archive.extractall(ROOT)

try:
    import timm, safetensors
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "timm>=1.0", "safetensors>=0.4",
    ], cwd=ROOT)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("PROJECT", PROJECT)
```

## Cell 2 - subprocess helper

```python
import os, subprocess, sys

def run_module(module, *args):
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)
```

## Cell 3 - contract, full-resolution forward and real-well overfit gate

```python
DATA = "/root/autodl-tmp/rogii-wellbore-geology-prediction"
ENCODER = Path("/root/autodl-tmp/model.safetensors")

args = [
    "--data-root", DATA,
    "--steps", 20,
]
if ENCODER.is_file():
    args += ["--encoder-checkpoint", ENCODER]
run_module("writeup_repro.autodl_validate", *args)
```

Required terminal line:

```text
AUTODL VALIDATION PASS
```

Do not continue if the overfit ratio is `>= 0.8`, the output is not
`[1, 345, 400]`, or checkpoint parameter coverage is below `0.85`.

## Cell 4 - 96/24-well default-model learning gateway

```python
OUTPUT = "/root/autodl-tmp/first_place_writeup_models_avgpool_v2"

args = [
    "--data-root", DATA,
    "--output-root", OUTPUT,
    "--version", "0719_V1",
    "--repeat", 0,
    "--fold", 1,
    "--epochs", 60,
    "--batch-size", 2,
    "--accumulation-steps", 8,
    "--workers", 4,
    "--patience", 20,
    "--limit-train", 96,
    "--limit-valid", 24,
    "--run-tag", "avgpool_v2_gateway",
]
if ENCODER.is_file():
    args += ["--encoder-checkpoint", ENCODER]
run_module("writeup_repro.train", *args)
```

This gateway validates learning direction only. It is not a reference CV.
Stop and audit if validation stays flat after training loss falls.

If the gateway stops after writing `best.pt`, audit the saved best model without
retraining:

```python
run_module(
    "writeup_repro.audit_checkpoint",
    "--data-root", DATA,
    "--checkpoint", f"{OUTPUT}/0719_V1/repeat0_fold1_seed0_avgpool_v2_gateway/best.pt",
    "--fold-file", f"{OUTPUT}/folds.csv",
    "--repeat", 0,
    "--fold", 1,
    "--limit-valid", 24,
)
```

## Cell 5 - complete Fold 1 default member

```python
args = [
    "--data-root", DATA,
    "--output-root", OUTPUT,
    "--version", "0719_V1",
    "--repeat", 0,
    "--fold", 1,
    "--epochs", 220,
    "--batch-size", 2,
    "--accumulation-steps", 8,
    "--workers", 4,
    "--patience", 35,
    "--run-tag", "avgpool_v2_full",
]
if ENCODER.is_file():
    args += ["--encoder-checkpoint", ENCODER]
run_module("writeup_repro.train", *args)
```

Only after Fold 1 reaches the expected metric neighborhood should the remaining
repeats, folds, PF/XY versions and ensemble be opened.

## Cell 6 - all disclosed versions, three repeats and five folds

```python
args = [
    "--data-root", DATA,
    "--output-root", OUTPUT,
    "--epochs", 220,
    "--batch-size", 2,
    "--accumulation-steps", 8,
    "--workers", 4,
    "--patience", 35,
]
if ENCODER.is_file():
    args += ["--encoder-checkpoint", ENCODER]
run_module("writeup_repro.train_all", *args)
```

## Cell 7 - test inference and fixed routed ensemble

```python
run_module(
    "writeup_repro.ensemble",
    "--data-root", DATA,
    "--model-root", OUTPUT,
    "--output", "/root/autodl-tmp/submission_writeup_reproduction.csv",
)
```
