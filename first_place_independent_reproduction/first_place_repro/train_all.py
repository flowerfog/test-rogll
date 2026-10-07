from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import FOLDS, VERSIONS


DEFAULT_REMAINING_ORDER = (
    "0801_V1",  # Default branch; closest transfer from the validated 0719 member.
    "0729_V3",  # PF branch.
    "0724_V1",  # XY branch.
    "0801_V2",  # Updated XY branch.
    "0803_V2",  # Joint PF + XY branch.
)
ZERO_WEIGHT_VERSIONS = ("0802_V2",)


def member_directory(
    output_root: str | Path,
    version: str,
    repeat: int,
    fold: int,
    seed: int,
    run_tag: str,
) -> Path:
    return Path(output_root) / version / f"repeat{repeat}_fold{fold}_seed{seed}_{run_tag}"


def build_command(args: argparse.Namespace, version: str, repeat: int, fold: int) -> list[str]:
    recipe = VERSIONS[version]
    command = [
        sys.executable, "-u", "-m", "first_place_repro.train",
        "--data-root", args.data_root,
        "--output-root", args.output_root,
        "--version", version,
        "--repeat", str(repeat),
        "--fold", str(fold),
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--validation-batch-size", str(args.validation_batch_size),
        "--workers", str(args.workers),
        "--schedule-epochs", str(recipe.epochs),
        "--save-every", str(args.save_every),
        "--run-tag", args.run_tag,
    ]
    if args.epochs is not None:
        command += ["--epochs", str(args.epochs)]
    if args.fold_file:
        command += ["--fold-file", args.fold_file]
    if args.encoder_checkpoint:
        command += ["--encoder-checkpoint", args.encoder_checkpoint]
    if args.activation_offload:
        command.append("--activation-offload")
    resume = member_directory(
        args.output_root, version, repeat, fold, args.seed, args.run_tag
    ) / "last.pt"
    if args.resume and resume.is_file():
        command += ["--resume", str(resume)]
    return command


def selected_versions(args: argparse.Namespace) -> tuple[str, ...]:
    versions = tuple(args.versions or DEFAULT_REMAINING_ORDER)
    unknown = sorted(set(versions) - set(VERSIONS))
    if unknown:
        raise ValueError(f"unknown versions: {unknown}")
    if not args.include_zero_weight and any(v in ZERO_WEIGHT_VERSIONS for v in versions):
        raise ValueError(
            "0802_V2 has zero disclosed final weight; pass --include-zero-weight "
            "to train it intentionally"
        )
    return versions


def selected_members(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.stage == "gate":
        return [(args.gate_repeat, args.gate_fold)]
    return [(repeat, fold) for repeat in args.repeats for fold in args.folds]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the inferred remaining first-place version family."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fold-file")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--versions", nargs="+", choices=tuple(VERSIONS))
    parser.add_argument("--stage", choices=("gate", "formal"), default="gate")
    parser.add_argument("--gate-repeat", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--gate-fold", type=int, choices=FOLDS, default=1)
    parser.add_argument("--repeats", nargs="+", type=int, choices=(0, 1, 2), default=(0, 1, 2))
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=FOLDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--run-tag", default="remaining_inferred_v1")
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-zero-weight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    versions = selected_versions(args)
    members = selected_members(args)
    plan = []
    for version in versions:
        for repeat, fold in members:
            command = build_command(args, version, repeat, fold)
            plan.append({"version": version, "repeat": repeat, "fold": fold, "command": command})

    print("REMAINING MODEL PLAN", json.dumps({
        "reproduction_level": "approximate; version-specific unpublished details inferred",
        "stage": args.stage,
        "versions": list(versions),
        "members": len(plan),
        "execute": args.execute,
        "zero_weight_excluded": not args.include_zero_weight,
    }), flush=True)
    for index, item in enumerate(plan, 1):
        print("MEMBER", json.dumps({"index": index, **item}), flush=True)
        if args.execute:
            subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
