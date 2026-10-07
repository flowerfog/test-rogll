# V242 run order

`stacked_unet_real_v242.py` is the real-data reproduction of the disclosed 26th-place core plus an explicit inferred synthetic pretraining generator. It is not the original final score: the writeup does not provide the original synthetic generator, exact pretrained weights, or complete optimizer schedule.

Upload the Python file to `/root/autodl-tmp/` first. Run these commands in separate notebook cells, in order.

## 1. Audit only

```python
import os
os.environ["ROGII_ROOT"] = "/root/autodl-tmp"
os.environ["V242_MODE"] = "audit"
exec(open("/root/autodl-tmp/stacked_unet_real_v242.py", encoding="utf-8").read())
```

This checks the data root, fold file, prefix lengths, and 5-channel panel configuration without training.

## 2. Smoke run

```python
import os
os.environ["ROGII_ROOT"] = "/root/autodl-tmp"
os.environ["V242_MODE"] = "smoke"
os.environ["V242_EPOCHS"] = "4"
os.environ["V242_BATCH"] = "1"
os.environ["V242_SYNTH"] = "1"
exec(open("/root/autodl-tmp/stacked_unet_real_v242.py", encoding="utf-8").read())
```

The smoke result is only a wiring check. It must show finite loss and metrics, and produce a checkpoint under `oof_exports/stacked_unet_real_v242`.

## 3. One full held-fold member

```python
import os
os.environ["ROGII_ROOT"] = "/root/autodl-tmp"
os.environ["V242_MODE"] = "full"
os.environ["V242_VALID_FOLD"] = "1"
os.environ["V242_SEED"] = "242"
os.environ["V242_EPOCHS"] = "18"
os.environ["V242_BATCH"] = "1"
os.environ["V242_REFEED_TRAIN_PROB"] = "0.5"
os.environ["V242_SYNTH"] = "1"
exec(open("/root/autodl-tmp/stacked_unet_real_v242.py", encoding="utf-8").read())
```

For the real-only control, set `V242_SYNTH="0"`. Compare it against the synthetic-pretrain run using the same fold, seed, epochs, and re-feed settings. Only after this completes should we decide whether a 5-fold or multi-seed sweep is worth the disk and GPU time. A single fold is not a competition submission and is not evidence of the published result.

## 4. EfficientNet member

The 26th-place writeup used SMP U-Net encoders. V242 supports that path through `segmentation_models_pytorch`:

```python
import os
os.environ["ROGII_ROOT"] = "/root/autodl-tmp"
os.environ["V242_MODE"] = "full"
os.environ["V242_VALID_FOLD"] = "1"
os.environ["V242_SEED"] = "242"
os.environ["V242_BACKBONE"] = "timm-efficientnet-b2"
os.environ["V242_SYNTH"] = "0"
os.environ["V242_EPOCHS"] = "18"
os.environ["V242_BATCH"] = "1"
exec(open("/root/autodl-tmp/stacked_unet_real_v242.py", encoding="utf-8").read())
```

Run one Fold 1 member before launching the ensemble. The original EffNet L/XL sizes are too large to assume on a 3080 Ti; B2 is the first hardware-feasible architecture probe. `run_v242_members.py` launches frozen folds and seeds without overwriting checkpoints.
