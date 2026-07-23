"""Dataset discovery and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path("/root/autolab/data")
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
HORIZONTAL_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_COLUMNS = ("TVT", "GR")


@dataclass(frozen=True)
class WellPaths:
    well_id: str
    horizontal: Path
    typewell: Path


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """Resolve data root without falling back when an explicit value is bad."""
    if explicit is not None:
        return Path(explicit).expanduser()
    configured = os.environ.get("ROGII_DATA_ROOT")
    return Path(configured).expanduser() if configured is not None else DEFAULT_DATA_ROOT


def discover_wells(split_dir: Path) -> list[WellPaths]:
    """Return sorted horizontal/typewell pairs from a train or test directory."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Data split does not exist: {split_dir}")

    wells: list[WellPaths] = []
    for horizontal in sorted(split_dir.glob(f"*{HORIZONTAL_SUFFIX}")):
        well_id = horizontal.name.removesuffix(HORIZONTAL_SUFFIX)
        typewell = split_dir / f"{well_id}{TYPEWELL_SUFFIX}"
        if not typewell.is_file():
            raise FileNotFoundError(
                f"Missing typewell for {well_id}: expected {typewell}"
            )
        wells.append(WellPaths(well_id, horizontal, typewell))

    if not wells:
        raise FileNotFoundError(f"No *{HORIZONTAL_SUFFIX} files in {split_dir}")
    return wells


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        newly_missing = converted.isna() & frame[column].notna()
        if newly_missing.any():
            row = int(np.flatnonzero(newly_missing.to_numpy())[0])
            raise ValueError(f"Non-numeric {column} at row {row} in {path}")
        frame[column] = converted


def validate_suffix_mask(horizontal: pd.DataFrame, path: Path) -> int:
    """Validate known-prefix/missing-suffix TVT_input and return its cutoff."""
    observed = horizontal["TVT_input"].notna().to_numpy()
    if not observed.any():
        raise ValueError(f"TVT_input has no known heel prefix in {path}")
    if observed.all():
        raise ValueError(f"TVT_input has no masked toe suffix in {path}")
    cutoff = int(np.flatnonzero(~observed)[0])
    if observed[cutoff:].any():
        raise ValueError(f"TVT_input is not a contiguous prefix in {path}")
    return cutoff


def read_well(
    paths: WellPaths, *, require_target: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a paired well, checking the inference-time schema and suffix mask."""
    horizontal = pd.read_csv(paths.horizontal)
    typewell = pd.read_csv(paths.typewell)

    horizontal_required = HORIZONTAL_COLUMNS + (("TVT",) if require_target else ())
    _require_columns(horizontal, horizontal_required, paths.horizontal)
    _require_columns(typewell, TYPEWELL_COLUMNS, paths.typewell)
    _numeric(horizontal, horizontal_required, paths.horizontal)
    _numeric(typewell, TYPEWELL_COLUMNS, paths.typewell)

    if horizontal.empty:
        raise ValueError(f"Horizontal well is empty: {paths.horizontal}")
    geometry = horizontal[["MD", "X", "Y", "Z"]].to_numpy(dtype=float)
    if not np.isfinite(geometry).all():
        raise ValueError(f"MD/X/Y/Z must be finite in {paths.horizontal}")
    if not horizontal["MD"].is_monotonic_increasing:
        raise ValueError(f"MD must be non-decreasing in {paths.horizontal}")
    validate_suffix_mask(horizontal, paths.horizontal)

    reference = typewell[["TVT", "GR"]].to_numpy(dtype=float)
    if len(reference) < 2 or not np.isfinite(reference).all():
        raise ValueError(f"Typewell TVT/GR needs at least two finite rows: {paths.typewell}")
    if require_target and not np.isfinite(horizontal["TVT"].to_numpy(dtype=float)).all():
        raise ValueError(f"Training TVT must be complete and finite: {paths.horizontal}")

    return horizontal, typewell
