"""Run leakage-resistant grouped out-of-fold evaluation."""

from __future__ import annotations

import argparse
import json

from rogii_baseline.data import resolve_data_root
from rogii_baseline.modeling import ModelConfig, grouped_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Overrides ROGII_DATA_ROOT")
    parser.add_argument("--folds", type=int, default=5)
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
    result = grouped_oof(root / "train", folds=args.folds, config=config)
    result["data_root"] = str(root)
    result["config"] = config.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

