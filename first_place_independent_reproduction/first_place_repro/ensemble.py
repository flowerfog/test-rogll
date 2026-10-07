from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import VERSION_RECIPES
from .data import list_wells
from .infer import predict_version
from .xy import XYSurfacePrior


def normalize_weights(values: dict[str, float]) -> dict[str, float]:
    positive = {name: float(value) for name, value in values.items() if value > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("ensemble has no positive weights")
    return {name: value / total for name, value in positive.items()}


class EnsembleRouter:
    def __init__(self):
        self.general = normalize_weights(
            {name: recipe.general_weight for name, recipe in VERSION_RECIPES.items()}
        )
        self.no_xy = normalize_weights(
            {name: recipe.no_xy_weight for name, recipe in VERSION_RECIPES.items()}
        )

    def weights(self, xy_reliable: bool) -> dict[str, float]:
        return self.general if xy_reliable else self.no_xy

    def blend(self, predictions: dict[str, np.ndarray], xy_reliable: bool) -> np.ndarray:
        weights = self.weights(xy_reliable)
        missing = set(weights) - set(predictions)
        if missing:
            raise KeyError(f"missing version predictions: {sorted(missing)}")
        return sum(weights[name] * predictions[name] for name in weights)


def run_ensemble(data_root: str, model_root: str, output: str) -> dict:
    router = EnsembleRouter()
    xy_prior = XYSurfacePrior().fit_root(data_root)
    required_versions = sorted(set(router.general) | set(router.no_xy))
    by_version, reliability = {}, {}
    for version in required_versions:
        predictions, version_reliability = predict_version(
            data_root,
            model_root,
            version,
            "test",
            xy_prior if VERSION_RECIPES[version].use_xy else None,
        )
        by_version[version] = predictions
        reliability.update(version_reliability)
    frames = []
    for well in list_wells(data_root, "test"):
        vectors = {
            version: by_version[version][well].sort_values("row")["tvt"].to_numpy()
            for version in required_versions
        }
        xy_reliable = reliability.get(well)
        if xy_reliable is None:
            raise RuntimeError(f"XY reliability was not produced for {well}")
        blended = router.blend(vectors, bool(xy_reliable))
        reference = by_version[required_versions[0]][well].sort_values("row").copy()
        reference["tvt"] = blended
        frames.append(reference)
    result = pd.concat(frames, ignore_index=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result[["id", "tvt"]].to_csv(destination, index=False)
    verdict = {
        "level": "behavioral reproduction from writeup only",
        "members": required_versions,
        "general_weights": router.general,
        "no_xy_weights": router.no_xy,
        "rows": len(result),
        "wells": result.well.nunique(),
        "output": str(destination),
    }
    destination.with_suffix(".json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run_ensemble(args.data_root, args.model_root, args.output), indent=2))


if __name__ == "__main__":
    main()
