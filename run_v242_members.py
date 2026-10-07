"""Run the disclosed-core V242 EfficientNet members without overwriting outputs."""

import os
import subprocess
import sys
from pathlib import Path


SOURCE = Path(os.environ.get("V242_SOURCE", "/root/autodl-tmp/stacked_unet_real_v242.py"))
FOLDS = [int(x) for x in os.environ.get("V242_FOLDS", "1,2,3,4,5").split(",")]
SEEDS = [int(x) for x in os.environ.get("V242_SEEDS", "242,243,244").split(",")]
BACKBONE = os.environ.get("V242_BACKBONE", "timm-efficientnet-b2")

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

for fold in FOLDS:
    for seed in SEEDS:
        env = os.environ.copy()
        env.update(
            {
                "V242_MODE": "full",
                "V242_VALID_FOLD": str(fold),
                "V242_SEED": str(seed),
                "V242_SYNTH": "0",
                "V242_BACKBONE": BACKBONE,
                "V242_RUN_TAG": f"{BACKBONE.replace('-', '_')}_f{fold}_s{seed}",
                "V242_BATCH": os.environ.get("V242_BATCH", "1"),
                "V242_EPOCHS": os.environ.get("V242_EPOCHS", "18"),
                "V242_REFEED_TRAIN_PROB": "0.5",
            }
        )
        print("V242 MEMBER", {"fold": fold, "seed": seed, "backbone": BACKBONE}, flush=True)
        subprocess.run([sys.executable, str(SOURCE)], env=env, check=True)
