from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import GRID, TRAINING
from .data import build_base_panel, load_record
from .data import list_wells
from .normalization import PopulationStats, fit_population_stats, write_population_stats


# Aggregate statistics recorded for the disclosed 16-channel 0719_V1 tensor.
# They are evidence targets, not values used by training or inference.
REFERENCE = {
    "tw_gr": (0.0289992280, 0.7677921653),
    "hw_gr_mean": (0.1688505709, 0.4991806448),
    "tw_hw_gr_abs_gap": (0.5021169782, 0.8372106552),
    "hw_gr_nan": (0.7604295611, 0.3022331297),
    "hw_gr_std": (0.1316640079, 0.1926901639),
    "hw_gr_last_first": (0.0013345600, 0.4324038327),
    "hw_gr_slope": (0.0023776572, 0.4556427300),
    "hw_gr_quadratic_a": (-0.0054805088, 0.5541797280),
    "hw_gr_quadratic_b": (-0.0047324449, 0.5874106288),
    "hw_gr_quadratic_c": (0.1694939137, 0.5105002522),
    "hw_gr_quadratic_rmse": (0.0273253676, 0.7757748365),
    "tw_gr_nan": (0.0, 0.0),
    "tw_tvt_rel": (-0.0000000033, 1.0000839233),
    "visible_tvt_rel": (0.3448083401, 0.2467621267),
    "tw_visible_tvt_abs_gap": (0.8720659614, 0.5038485527),
    "z_diff": (0.0505324751, 0.7820937634),
}


def load_stats(checkpoint: Path) -> PopulationStats:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = payload.get("population_stats")
    if not values:
        raise RuntimeError(f"checkpoint has no population_stats: {checkpoint}")
    return PopulationStats.from_dict(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--refit-stats", type=Path)
    parser.add_argument("--well", default="000d7d20")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-standardized-gap", type=float, default=0.35)
    args = parser.parse_args()

    record = load_record(args.data_root, args.split, args.well)
    if args.refit_stats:
        stats = fit_population_stats(args.data_root, list_wells(args.data_root, "train"))
        args.refit_stats.parent.mkdir(parents=True, exist_ok=True)
        write_population_stats(args.refit_stats, stats)
        print("REFIT STATS", args.refit_stats, flush=True)
    else:
        stats = load_stats(args.checkpoint)
    physical = build_base_panel(
        record,
        GRID,
        TRAINING.target_sigma_ft,
        population_stats=stats,
        apply_channel_normalization=False,
    )
    sample = build_base_panel(
        record, GRID, TRAINING.target_sigma_ft,
        population_stats=stats, apply_channel_normalization=True,
    )
    rows = []
    failures = []
    physical_by_name = dict(zip(physical.channels, physical.values))
    for name, values in zip(sample.channels, sample.values):
        if name not in REFERENCE:
            continue
        reference_mean, reference_std = REFERENCE[name]
        actual_mean = float(np.mean(values))
        actual_std = float(np.std(values))
        physical_values = physical_by_name[name]
        scale = max(reference_std, 0.1)
        mean_gap = abs(actual_mean - reference_mean) / scale
        std_gap = abs(actual_std - reference_std) / scale
        row = {
            "channel": name,
            "actual_mean": actual_mean,
            "reference_mean": reference_mean,
            "actual_std": actual_std,
            "reference_std": reference_std,
            "physical_mean": float(np.mean(physical_values)),
            "physical_std": float(np.std(physical_values)),
            "physical_min": float(np.min(physical_values)),
            "physical_max": float(np.max(physical_values)),
            "population_mean": stats.channel_mean.get(name),
            "population_scale": stats.channel_scale.get(name),
            "mean_gap_sigma": mean_gap,
            "std_gap_fraction": std_gap,
        }
        rows.append(row)
        if max(mean_gap, std_gap) > args.max_standardized_gap:
            failures.append(name)

    report = {
        "well": args.well,
        "split": args.split,
        "shape": list(sample.values.shape),
        "state_axis": [float(sample.state_offsets[0]), float(sample.state_offsets[-1])],
        "threshold": args.max_standardized_gap,
        "channels": rows,
        "failures": failures,
        "passed": not failures,
    }
    print("REPRESENTATION AUDIT", json.dumps(report, indent=2), flush=True)
    if failures:
        raise RuntimeError({
            "message": "representation parity failed; training is blocked",
            "channels": failures,
        })


if __name__ == "__main__":
    main()
