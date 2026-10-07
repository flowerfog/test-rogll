from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import VERSION_RECIPES
from .train import parser as member_parser
from .train import train_member


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--fold-file")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--versions", nargs="*", choices=VERSION_RECIPES, default=list(VERSION_RECIPES))
    parser.add_argument("--encoder-checkpoint")
    args = parser.parse_args()
    results = []
    base_parser = member_parser()
    for version in args.versions:
        for repeat in range(3):
            for fold in range(1, 6):
                arguments = [
                    "--data-root", args.data_root,
                    "--output-root", args.output_root,
                    "--version", version,
                    "--repeat", str(repeat),
                    "--fold", str(fold),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--accumulation-steps", str(args.accumulation_steps),
                    "--workers", str(args.workers),
                    "--patience", str(args.patience),
                ]
                if args.cache_root:
                    arguments += ["--cache-root", args.cache_root]
                if args.fold_file:
                    arguments += ["--fold-file", args.fold_file]
                if args.encoder_checkpoint:
                    arguments += ["--encoder-checkpoint", args.encoder_checkpoint]
                results.append(train_member(base_parser.parse_args(arguments)))
    summary = Path(args.output_root) / "training_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
