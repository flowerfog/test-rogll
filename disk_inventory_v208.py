"""Read-only disk inventory for the AutoDL ROGII workspace.

This script never deletes or modifies experiment artifacts.  It writes small
CSV/JSON reports under /root/autodl-tmp/disk_audit_v208 and prints the most
useful summaries for deciding what can be removed before rebuilding V4 panels.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import time

import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
REPORT_DIR = ROOT / "disk_audit_v208"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# These are needed to rebuild/run V208 and must not be proposed for deletion.
PROTECTED_RELATIVE = (
    "rogii-wellbore-geology-prediction",
    "tvt_corrnet_heel_conditioned_onefold_v4_cell.py",
    "tvt_corrnet_scaled_fullfold_gateway_v208.py",
    "oof_exports/foldclean_full_fleongg_lgb_v0",
    "oof_exports/multicut_fleongg_matched_k3_gateway_v1",
    "oof_exports/multicut_fleongg_matched_k5_extension_v1",
    "oof_exports/tvt_corrnet_heel_conditioned_onefold_v4",
    "oof_exports/tvt_corrnet_scaled_fullfold_gateway_v208",
)


def _human(size):
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _is_protected(path):
    resolved = path.resolve()
    for relative in PROTECTED_RELATIVE:
        protected = (ROOT / relative).resolve()
        if resolved == protected or protected in resolved.parents or resolved in protected.parents:
            return True
    return False


def _category(path):
    parts = {part.lower() for part in path.parts}
    suffix = path.suffix.lower()
    name = path.name.lower()
    if "panels" in parts or "matched_panels" in parts or "supplement_panels" in parts:
        return "panels"
    if "records" in parts or "record_cache" in parts:
        return "records"
    if "features" in parts:
        return "features"
    if "models" in parts or suffix in (".pt", ".pth", ".ckpt", ".bin", ".txt") and "model" in name:
        return "models"
    if suffix in (".parquet", ".pkl", ".pickle", ".feather"):
        return "tables"
    if suffix in (".npz", ".npy"):
        return "arrays"
    if suffix in (".csv", ".json", ".yaml", ".yml"):
        return "reports"
    return "other"


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(ROOT)
    started = time.time()
    usage = shutil.disk_usage(ROOT)
    experiment_totals = defaultdict(lambda: {"apparent": 0, "allocated": 0, "files": 0, "latest": 0.0})
    root_totals = defaultdict(lambda: {"apparent": 0, "allocated": 0, "files": 0, "latest": 0.0})
    category_totals = defaultdict(lambda: {"apparent": 0, "allocated": 0, "files": 0})
    largest = []
    checkpoints = []
    scanned = 0
    errors = []

    for directory, _, filenames in os.walk(ROOT):
        directory_path = Path(directory)
        try:
            relative_directory = directory_path.relative_to(ROOT)
        except ValueError:
            continue
        root_name = relative_directory.parts[0] if relative_directory.parts else "."
        experiment = None
        if len(relative_directory.parts) >= 2 and relative_directory.parts[0] == "oof_exports":
            experiment = relative_directory.parts[1]
        for filename in filenames:
            path = directory_path / filename
            try:
                stat = path.stat()
            except OSError as exc:
                errors.append({"path": str(path), "reason": repr(exc)})
                continue
            apparent = int(stat.st_size)
            allocated = int(getattr(stat, "st_blocks", 0) * 512) or apparent
            modified = float(stat.st_mtime)
            scanned += 1
            for target in (root_totals[root_name],):
                target["apparent"] += apparent
                target["allocated"] += allocated
                target["files"] += 1
                target["latest"] = max(target["latest"], modified)
            if experiment is not None:
                target = experiment_totals[experiment]
                target["apparent"] += apparent
                target["allocated"] += allocated
                target["files"] += 1
                target["latest"] = max(target["latest"], modified)
            category = _category(path)
            category_totals[category]["apparent"] += apparent
            category_totals[category]["allocated"] += allocated
            category_totals[category]["files"] += 1
            row = {
                "path": str(path),
                "apparent_bytes": apparent,
                "allocated_bytes": allocated,
                "category": category,
                "modified_utc": datetime.fromtimestamp(modified, timezone.utc).isoformat(),
                "protected": int(_is_protected(path)),
            }
            if apparent >= 20 * 1024**2:
                largest.append(row)
            if path.suffix.lower() in (".pt", ".pth", ".ckpt"):
                checkpoints.append(row)
            if scanned % 100_000 == 0:
                print(
                    "DISK AUDIT",
                    {"files": scanned, "elapsed_min": (time.time() - started) / 60.0},
                    flush=True,
                )

    def totals_frame(source, name):
        rows = []
        now = time.time()
        for key, value in source.items():
            path = ROOT / key if name == "root_entry" else OOF_ROOT / key
            rows.append(
                {
                    name: key,
                    "apparent_gb": value["apparent"] / 1024**3,
                    "allocated_gb": value["allocated"] / 1024**3,
                    "files": value["files"],
                    "age_days": (now - value["latest"]) / 86400 if value["latest"] else None,
                    "protected": int(_is_protected(path)),
                }
            )
        return pd.DataFrame(rows).sort_values("allocated_gb", ascending=False)

    root_frame = totals_frame(root_totals, "root_entry")
    experiment_frame = totals_frame(experiment_totals, "experiment")
    category_frame = pd.DataFrame(
        [
            {
                "category": key,
                "apparent_gb": value["apparent"] / 1024**3,
                "allocated_gb": value["allocated"] / 1024**3,
                "files": value["files"],
            }
            for key, value in category_totals.items()
        ]
    ).sort_values("allocated_gb", ascending=False)
    file_columns = (
        "path", "apparent_bytes", "allocated_bytes", "category",
        "modified_utc", "protected",
    )
    largest_frame = pd.DataFrame(largest, columns=file_columns).sort_values(
        "allocated_bytes", ascending=False
    )
    checkpoint_frame = pd.DataFrame(checkpoints, columns=file_columns).sort_values(
        "allocated_bytes", ascending=False
    )

    root_frame.to_csv(REPORT_DIR / "root_entries.csv", index=False)
    experiment_frame.to_csv(REPORT_DIR / "oof_experiments.csv", index=False)
    category_frame.to_csv(REPORT_DIR / "categories.csv", index=False)
    largest_frame.to_csv(REPORT_DIR / "largest_files.csv", index=False)
    checkpoint_frame.to_csv(REPORT_DIR / "checkpoints.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(REPORT_DIR / "scan_errors.csv", index=False)

    unprotected = experiment_frame[experiment_frame["protected"].eq(0)].copy()
    reclaim_top20 = float(unprotected.head(20)["allocated_gb"].sum())
    verdict = {
        "experiment": "disk_inventory_v208",
        "read_only": True,
        "root": str(ROOT),
        "disk": {
            "total_gb": usage.total / 1024**3,
            "used_gb": usage.used / 1024**3,
            "free_gb": usage.free / 1024**3,
        },
        "files_scanned": scanned,
        "scan_errors": len(errors),
        "protected_paths": [str(ROOT / item) for item in PROTECTED_RELATIVE],
        "largest_unprotected_oof_gb_top20": reclaim_top20,
        "report_dir": str(REPORT_DIR),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "next": "Review candidates with Codex before deleting anything",
    }
    (REPORT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print("\nDISK SUMMARY")
    print(json.dumps(verdict, indent=2), flush=True)
    print("\nTOP ROOT ENTRIES")
    print(root_frame.head(20).to_string(index=False), flush=True)
    print("\nSTORAGE CATEGORIES")
    print(category_frame.to_string(index=False), flush=True)
    print("\nTOP 40 OOF EXPERIMENTS (protected=1 must be retained)")
    print(experiment_frame.head(40).to_string(index=False), flush=True)
    print("\nTOP 30 LARGE FILES")
    display_columns = ["allocated_bytes", "category", "protected", "path"]
    print(largest_frame[display_columns].head(30).to_string(index=False), flush=True)
    print("saved:", REPORT_DIR, flush=True)


if __name__ == "__main__":
    main()
