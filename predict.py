"""Fit on all training wells and predict the hidden-test masked suffixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii_baseline.data import resolve_data_root
from rogii_baseline.modeling import ModelConfig, predict_hidden_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Overrides ROGII_DATA_ROOT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-rows-per-well", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_data_root(args.data_root)
    config = ModelConfig(
        max_train_rows_per_well=args.max_train_rows_per_well,
        max_iter=args.max_iter,
    )
    result = predict_hidden_test(root, args.output, config=config)
    result["data_root"] = str(root)
    result["config"] = config.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

