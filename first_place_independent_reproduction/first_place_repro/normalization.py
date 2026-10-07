from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from .data import load_record


@dataclass(frozen=True)
class PopulationStats:
    gr_center: float
    gr_scale: float
    channel_mean: dict[str, float] = field(default_factory=dict)
    channel_scale: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "PopulationStats":
        return cls(
            gr_center=float(values["gr_center"]),
            gr_scale=float(values["gr_scale"]),
            channel_mean={str(k): float(v) for k, v in values.get("channel_mean", {}).items()},
            channel_scale={str(k): float(v) for k, v in values.get("channel_scale", {}).items()},
        )


def fit_population_stats(data_root: str | Path, wells: list[str]) -> PopulationStats:
    # Kept as an API compatibility shim. The 0719 representation uses the
    # fixed train-population constants embedded in data.py; it must not refit
    # them per fold or from a single parity well.
    del data_root, wells
    return PopulationStats(87.43, 21.09)


def write_population_stats(path: str | Path, stats: PopulationStats) -> None:
    Path(path).write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")


def read_population_stats(path: str | Path) -> PopulationStats:
    return PopulationStats.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
