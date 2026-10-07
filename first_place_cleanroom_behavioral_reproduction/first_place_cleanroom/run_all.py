from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import FOLDS, SEEDS, VERSIONS


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("audit", "prepare", "train", "oof", "infer", "all"), default="all")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    cache = args.work_root / "cache"
    models = args.work_root / "models"
    python = sys.executable
    if args.stage in ("audit", "all"):
        run([python, "-m", "first_place_cleanroom.contract_audit",
             "--data-root", str(args.data_root), "--model-forward"])
        run([python, "-m", "first_place_cleanroom.simulation_audit",
             "--data-root", str(args.data_root), "--wells", "128"])
        run([python, "-m", "first_place_cleanroom.audit", "--data-root", str(args.data_root),
             "--version", "0719_V1", "--model-forward"])
    if args.stage in ("prepare", "all"):
        run([python, "-m", "first_place_cleanroom.prepare", "--data-root", str(args.data_root),
             "--output-root", str(cache), "--component", "all", "--split", "both"])
        run([python, "-m", "first_place_cleanroom.contract_audit",
             "--data-root", str(args.data_root), "--cache-root", str(cache)])
    if args.stage in ("train", "all"):
        for version in VERSIONS:
            for fold in FOLDS:
                for seed in SEEDS:
                    checkpoint = models / version.name / f"fold{fold}_seed{seed}" / "best.pt"
                    if checkpoint.is_file():
                        print("SKIP", checkpoint, flush=True)
                        continue
                    command = [python, "-m", "first_place_cleanroom.train",
                               "--data-root", str(args.data_root), "--cache-root", str(cache),
                               "--output-root", str(models), "--version", version.name,
                               "--fold", str(fold), "--seed", str(seed)]
                    run(command)
    if args.stage in ("oof", "all"):
        run([python, "-m", "first_place_cleanroom.evaluate_oof",
             "--data-root", str(args.data_root), "--cache-root", str(cache),
             "--model-root", str(models), "--output", str(args.work_root / "oof")])
    if args.stage in ("infer", "all"):
        if args.submission is None:
            raise ValueError("--submission is required for infer/all")
        run([python, "-m", "first_place_cleanroom.infer", "--data-root", str(args.data_root),
             "--cache-root", str(cache), "--model-root", str(models), "--output", str(args.submission)])


if __name__ == "__main__":
    main()
