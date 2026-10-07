from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(module: str, *arguments: object) -> None:
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    print("RUN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, check=True, env=environment)


def train_one(
    data_root: str,
    output_root: str,
    version: str = "0719_V1",
    repeat: int = 0,
    fold: int = 1,
    epochs: int = 220,
    batch_size: int = 4,
    accumulation_steps: int = 4,
    encoder_checkpoint: str | None = None,
) -> None:
    arguments = [
        "writeup_repro.train",
        "--data-root", data_root,
        "--output-root", output_root,
        "--version", version,
        "--repeat", repeat,
        "--fold", fold,
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--accumulation-steps", accumulation_steps,
    ]
    if encoder_checkpoint:
        arguments += ["--encoder-checkpoint", encoder_checkpoint]
    _run(*arguments)


def train_full(data_root: str, output_root: str, epochs: int = 220, encoder_checkpoint: str | None = None) -> None:
    arguments = [
        "writeup_repro.train_all",
        "--data-root", data_root,
        "--output-root", output_root,
        "--epochs", epochs,
    ]
    if encoder_checkpoint:
        arguments += ["--encoder-checkpoint", encoder_checkpoint]
    _run(*arguments)


def infer_full(data_root: str, model_root: str, output: str) -> None:
    _run(
        "writeup_repro.ensemble",
        "--data-root", data_root,
        "--model-root", model_root,
        "--output", output,
    )
