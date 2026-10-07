from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MISSING_WEIGHT_VERSIONS = ("0729_V3", "0801_V1", "0724_V1", "0801_V2", "0803_V2")


def execute(*arguments: str) -> None:
    command = [sys.executable, "-u", *map(str, arguments)]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def module(name: str, *arguments: str) -> None:
    execute("-m", name, *arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every known first-place component without reduced gates.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    module("literal_repro.audit", "--reference-dir", args.reference_dir, "--output", args.output_root / "archive_audit.json")
    module(
        "literal_repro.run_author", "infer-0719", "--data-root", args.data_root,
        "--reference-dir", args.reference_dir, "--output-dir", args.output_root / "0719_V1",
        "--device", "cuda", "--workers", str(args.workers),
    )
    module(
        "literal_repro.parity", "--reference", args.reference_dir / "submission.csv",
        "--candidate", args.output_root / "0719_V1" / "submission.csv", "--atol", "1e-6",
    )
    module(
        "literal_repro.make_branch_cfg", "--reference-cfg", args.reference_dir / "cfg.pkl",
        "--output-root", args.output_root,
    )
    for version in MISSING_WEIGHT_VERSIONS:
        module(
            "literal_repro.train_from_cfg", "--cfg", args.output_root / "configs" / version / "cfg.pkl",
            "--data-root", args.data_root, "--output-dir", args.output_root / version,
            "--encoder-checkpoint", args.encoder_checkpoint,
        )
    module(
        "literal_repro.xy_quality", "--data-root", args.data_root,
        "--output", args.output_root / "xy_quality.csv",
    )
    module(
        "literal_repro.ensemble", "--prediction-root", args.output_root,
        "--quality-csv", args.output_root / "xy_quality.csv",
        "--output", args.output_root / "first_place_ensemble.csv",
    )


if __name__ == "__main__":
    main()
