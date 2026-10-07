"""Conservative cleanup after autodl_disk_cleanup_audit_v67.py.

Set EXECUTE=True only after reviewing the printed paths. The script removes
failed experiment outputs, regenerable caches outside protected experiments,
and the desktop trash. It never touches raw data or the current 6.377/V67
lineage.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


EXECUTE = False
ROOT = Path("/root/autodl-tmp").resolve()
OOF_ROOT = (ROOT / "oof_exports").resolve()
AUDIT = ROOT / "disk_cleanup_audit_v67" / "candidate_paths.txt"

PROTECTED_PARTS = {
    "frontier_6p377_full_fivefold_lineage_v66",
    "frontier_6p39_full_oof_cp_increment_v18",
    "neighbor_fullcurve_transfer_v0",
    "foldclean_full_fleongg_lgb_v0",
    "frontier_6p39_existing_models_oof_v0",
    "frontier_6p39_cp8_nocp12_matched_v9",
    "rawdata_lattice_coarse_canonical_gateway_v67",
}


def size_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for current, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                pass
    return total


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


if not AUDIT.is_file():
    raise FileNotFoundError("Run autodl_disk_cleanup_audit_v67.py first")

raw = [Path(line.strip()).resolve() for line in AUDIT.read_text().splitlines() if line.strip()]
raw.append((ROOT / ".Trash-0").resolve())

approved = []
for path in raw:
    if path == ROOT or path == OOF_ROOT:
        continue
    if not (inside(path, OOF_ROOT) or path == ROOT / ".Trash-0"):
        continue
    if any(part in PROTECTED_PARTS for part in path.parts):
        continue
    approved.append(path)

# If both an experiment and one of its caches are listed, keep only the parent.
approved = sorted(set(approved), key=lambda item: (len(item.parts), str(item)))
targets = []
for path in approved:
    if not any(inside(path, parent) for parent in targets):
        targets.append(path)

existing = [(path, size_bytes(path)) for path in targets if path.exists()]
total = sum(size for _, size in existing)
print(
    "CLEANUP PLAN",
    {
        "execute": EXECUTE,
        "targets": len(existing),
        "reclaim_gb": round(total / 1024**3, 3),
    },
)
for path, size in sorted(existing, key=lambda item: item[1], reverse=True):
    print(f"{size / 1024**3:8.3f} GB  {path}")

if not EXECUTE:
    print("DRY RUN ONLY. Set EXECUTE=True in this cell and rerun to delete.")
else:
    for path, _ in existing:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    usage = shutil.disk_usage(ROOT)
    print(
        "CLEANUP COMPLETE",
        {
            "removed_targets": len(existing),
            "reclaimed_gb_estimate": round(total / 1024**3, 3),
            "free_gb": round(usage.free / 1024**3, 3),
        },
    )
