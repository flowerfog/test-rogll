from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FOLDS, RECONSTRUCTION_REVISION, SEEDS, VERSIONS
from .data import load_well
from .xy import xy_is_safe


def load_version(model_root: Path, version_name: str) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    count: dict[str, int] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            folder = model_root / version_name / f"fold{fold}_seed{seed}"
            path = folder / "valid_predictions.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            verdict = json.loads((folder / "verdict.json").read_text(encoding="utf-8"))
            revision = verdict.get("config", {}).get("revision")
            if revision != RECONSTRUCTION_REVISION:
                raise RuntimeError({"message": "stale OOF member", "path": str(path),
                                    "found": revision, "expected": RECONSTRUCTION_REVISION})
            with np.load(path) as source:
                for well in source.files:
                    value = source[well].astype(np.float64)
                    total[well] = total.get(well, np.zeros_like(value)) + value
                    count[well] = count.get(well, 0) + 1
    wrong = {well: number for well, number in count.items() if number != len(SEEDS)}
    if wrong:
        raise RuntimeError({"message": "OOF member count mismatch", "sample": dict(list(wrong.items())[:10])})
    return {well: value / count[well] for well, value in total.items()}


def load_xy_stats(cache_root: Path, well: str, fold: int) -> dict[str, float]:
    path = cache_root / "xy" / "train" / f"fold{fold}" / f"{well}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as source:
        return {key.removeprefix("stat__"): float(source[key])
                for key in source.files if key.startswith("stat__")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = {
        version.name: load_version(args.model_root, version.name)
        for version in VERSIONS
        if version.general_weight or version.no_xy_weight
    }
    wells = sorted(next(iter(predictions.values())))
    folds = pd.read_csv(args.cache_root / "folds.csv", dtype={"well": str})
    well_fold = folds.set_index("well").fold.astype(int).to_dict()
    squared = absolute = 0.0
    rows = safe_wells = 0
    by_well = []
    for well in wells:
        record = load_well(args.data_root, well, "train")
        stats = load_xy_stats(args.cache_root, well, well_fold[well])
        safe = xy_is_safe(stats)
        safe_wells += int(safe)
        blend = np.zeros(len(record.horizontal), np.float64)
        weight_sum = 0.0
        for version in VERSIONS:
            if version.name not in predictions:
                continue
            weight = version.general_weight if safe else version.no_xy_weight
            blend += weight * predictions[version.name][well]
            weight_sum += weight
        if weight_sum <= 0:
            raise RuntimeError(f"{well}: zero ensemble weight")
        blend /= weight_sum
        truth = record.horizontal.TVT.to_numpy(float)
        mask = np.arange(len(truth)) > record.anchor_index
        mask &= np.isfinite(truth) & np.isfinite(blend)
        error = blend[mask] - truth[mask]
        sse = float(error @ error)
        squared += sse
        absolute += float(np.abs(error).sum())
        rows += int(mask.sum())
        by_well.append({"well": well, "rows": int(mask.sum()), "sse": sse,
                        "rmse": math.sqrt(sse / max(mask.sum(), 1)), "xy_safe": safe})
    summary = {"rows": rows, "wells": len(wells), "xy_safe_wells": safe_wells,
               "rmse": math.sqrt(squared / rows), "mae": absolute / rows,
               "reference_rmse": 4.627}
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(by_well).to_csv(args.output / "by_well.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("OOF SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
