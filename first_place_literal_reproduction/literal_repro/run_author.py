from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .constants import AUTHOR_SRC


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(AUTHOR_SRC) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the byte-identical author implementation.")
    sub = parser.add_subparsers(dest="action", required=True)
    infer = sub.add_parser("infer-0719")
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--reference-dir", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--workers", type=int)
    train = sub.add_parser("train-0719")
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    args = parser.parse_args()

    command = [sys.executable, "-u", str(AUTHOR_SRC / "seq_NN_main.py")]
    if args.action == "infer-0719":
        command += [
            "--submit-mode", "--train-output-dir", str(args.reference_dir),
            "--data-dir", str(args.data_root), "--output-dir", str(args.output_dir),
            "--device", args.device,
        ]
        if args.output_dir.exists():
            command += ["--f", "1"]
        if args.workers is not None:
            command += ["--num-workers", str(args.workers)]
    else:
        command += [
            "--data-dir", str(args.data_root), "--output-dir", str(args.output_dir),
            "--device", args.device,
        ]
    run(command)


if __name__ == "__main__":
    main()
