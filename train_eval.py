"""Run leakage-resistant grouped out-of-fold evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii_baseline.data import resolve_data_root
from rogii_baseline.modeling import ModelConfig, grouped_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Overrides ROGII_DATA_ROOT")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-train-rows-per-well", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("output/metrics.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_data_root(args.data_root)
    config = ModelConfig(
        max_train_rows_per_well=args.max_train_rows_per_well,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        hidden_width=args.hidden_width,
        device=args.device,
    )
    result = grouped_oof(root / "train", folds=args.folds, config=config)
    result["data_root"] = str(root)
    result["config"] = config.__dict__
    serialized = json.dumps(result, indent=2, sort_keys=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
