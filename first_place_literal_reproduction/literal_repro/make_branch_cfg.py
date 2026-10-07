from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
from copy import deepcopy
from pathlib import Path

from .constants import AUTHOR_SRC, VERSION_CHANNELS, VERSIONS


def ordered_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for group in groups for name in group))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create source-derived configs for the six published branches.")
    parser.add_argument("--reference-cfg", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(AUTHOR_SRC))
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        with args.reference_cfg.open("rb") as stream:
            reference = pickle.load(stream)
    finally:
        pathlib.PosixPath = original_posix_path

    reports = []
    for version, spec in VERSIONS.items():
        cfg = deepcopy(reference)
        branch = spec["branch"]
        channels = VERSION_CHANNELS[version]
        cfg.unet_static_channels = channels
        cfg.output_dir = args.output_root / version
        cfg.PF_heatmap_cache_dir = args.output_root / "cache" / version / "PF_cache"
        cfg.PF_sample_cache_dir = args.output_root / "cache" / version / "PF_sample_cache"
        cfg.refresh()
        version_dir = args.output_root / "configs" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        with (version_dir / "cfg.pkl").open("wb") as stream:
            pickle.dump(cfg, stream)
        report = {
            "version": version,
            "branch": branch,
            "channels": list(channels),
            "status": spec["status"],
            "config_basis": "literal cfg" if version == "0719_V1" else "0719 literal cfg plus write-up disclosed channels",
        }
        (version_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
