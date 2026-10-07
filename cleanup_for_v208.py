"""Delete two audited, V208-independent AutoDL directories.

Targets are deliberately hard-coded.  The script verifies that the canonical
per-well dataset and every V208 upstream artifact still exist before deleting.
It refuses symlinks and paths outside /root/autodl-tmp.
"""

from pathlib import Path
import shutil


ROOT = Path("/root/autodl-tmp").resolve()
TARGETS = (
    ROOT / "wellbore-geology-prediction-artifacts",
    ROOT / ".Trash-0",
)
REQUIRED = (
    ROOT / "rogii-wellbore-geology-prediction" / "train",
    ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ROOT / "oof_exports" / "multicut_fleongg_matched_k3_gateway_v1",
    ROOT / "oof_exports" / "multicut_fleongg_matched_k5_extension_v1",
    ROOT / "tvt_corrnet_heel_conditioned_onefold_v4_cell.py",
    ROOT / "tvt_corrnet_scaled_fullfold_gateway_v208.py",
)


def allocated_bytes(path):
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            stat = item.stat()
            total += int(getattr(stat, "st_blocks", 0) * 512) or int(stat.st_size)
    return total


missing = [str(path) for path in REQUIRED if not path.exists()]
if missing:
    raise RuntimeError(f"Refusing cleanup; required V208 inputs are missing: {missing}")

before = shutil.disk_usage(ROOT).free
reports = []
for target in TARGETS:
    resolved = target.resolve()
    if resolved.parent != ROOT:
        raise RuntimeError(f"Refusing path outside root: {resolved}")
    if target.is_symlink():
        raise RuntimeError(f"Refusing symlink target: {target}")
    size = allocated_bytes(target)
    existed = target.exists()
    if existed:
        shutil.rmtree(target)
    reports.append(
        {
            "path": str(target),
            "existed": existed,
            "removed_gb": size / 1024**3,
        }
    )

after = shutil.disk_usage(ROOT).free
print("V208 CLEANUP")
for report in reports:
    print(report)
print(
    {
        "free_before_gb": before / 1024**3,
        "free_after_gb": after / 1024**3,
        "freed_gb": (after - before) / 1024**3,
        "required_inputs_rechecked": all(path.exists() for path in REQUIRED),
    }
)
