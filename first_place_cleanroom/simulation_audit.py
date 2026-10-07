from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import list_wells, load_well, simulate_z_shift


def summarize(values: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": dict(zip(
            ("q001", "q01", "q50", "q99", "q999"),
            map(float, np.quantile(values, (.001, .01, .5, .99, .999))),
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--wells", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    names = list_wells(args.data_root, "train")
    rng = np.random.default_rng(args.seed)
    real, simulated = [], []
    for name in names[: min(args.wells, len(names))]:
        record = load_well(args.data_root, name, "train")
        donor = load_well(args.data_root, names[int(rng.integers(len(names)))], "train")
        synthetic = simulate_z_shift(record, donor, rng)
        for source, output in ((record, real), (synthetic, simulated)):
            tvt = source.horizontal.TVT.to_numpy(float)
            output.append(tvt[source.anchor_index + 1:] - tvt[source.anchor_index])
    real_values = np.concatenate(real)
    simulated_values = np.concatenate(simulated)
    report = {"wells": min(args.wells, len(names)), "real": summarize(real_values),
              "simulated": summarize(simulated_values)}
    ratio = report["simulated"]["std"] / report["real"]["std"]
    report["std_ratio"] = ratio
    report["pass"] = .75 <= ratio <= 1.75 and abs(report["simulated"]["mean"]) <= 8.0
    print("SIMULATION AUDIT", json.dumps(report, indent=2), flush=True)
    if not report["pass"]:
        raise RuntimeError({"message": "synthetic target distribution gate failed", **report})


if __name__ == "__main__":
    main()
