"""Safely reclaim the obsolete RankSDF v1 panel cache on AutoDL."""

from pathlib import Path
import shutil


ROOT = Path("/root/autodl-tmp").resolve()
OOF_ROOT = (ROOT / "oof_exports").resolve()

# These tensors belong to the failed moving-centre v1 experiment. Reports,
# predictions and model checkpoints in its parent directory are retained.
DELETE_DIRS = [
    OOF_ROOT / "ranksdf_cp_v1_real_pseudops_fold1_3seed" / "panels",
]


def directory_size(path):
    total = 0
    files = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
                files += 1
            except FileNotFoundError:
                pass
    return total, files


def assert_safe_directory(path):
    resolved = path.resolve()
    if resolved == OOF_ROOT or OOF_ROOT not in resolved.parents:
        raise RuntimeError(f"Refusing path outside the OOF root: {resolved}")
    if resolved.name != "panels":
        raise RuntimeError(f"Only an explicitly named panel cache may be removed: {resolved}")
    if resolved.parent.name != "ranksdf_cp_v1_real_pseudops_fold1_3seed":
        raise RuntimeError(f"Unexpected experiment directory: {resolved}")
    return resolved


before = shutil.disk_usage(ROOT)
print(
    "DISK BEFORE",
    {
        "free_gb": before.free / 2**30,
        "used_gb": before.used / 2**30,
    },
)

report = []
for candidate in DELETE_DIRS:
    path = assert_safe_directory(candidate)
    if not path.exists():
        report.append({"path": str(path), "exists": False, "files": 0, "size_gb": 0.0})
        continue
    size, files = directory_size(path)
    report.append(
        {
            "path": str(path),
            "exists": True,
            "files": files,
            "size_gb": size / 2**30,
        }
    )

print("DELETE PLAN")
for row in report:
    print(row)
print("reclaimable_gb:", sum(row["size_gb"] for row in report))

for row in report:
    if not row["exists"]:
        continue
    path = assert_safe_directory(Path(row["path"]))
    shutil.rmtree(path)
    print("deleted:", path)

after = shutil.disk_usage(ROOT)
print(
    "DISK AFTER",
    {
        "free_gb": after.free / 2**30,
        "used_gb": after.used / 2**30,
        "reclaimed_gb": (after.free - before.free) / 2**30,
    },
)

if after.free / 2**30 < 10.5:
    print("WARNING: less than 10.5 GB remains; do not start the v2 panel build yet.")
else:
    print("READY: enough free space for the bounded RankSDF v2 gateway.")
