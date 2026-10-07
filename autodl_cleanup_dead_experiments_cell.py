"""Safe AutoDL cleanup for experiments that have already failed their gates.

The cell preserves small diagnostic reports before deleting only explicitly
listed caches. It never touches the candidate pool, dataset, submissions, or
the retained multi-cut OOF pickle.
"""

from pathlib import Path
import shutil


OOF_ROOT = Path("/root/autodl-tmp/oof_exports").resolve()
REPORT_DIR = OOF_ROOT / "keep_reports_20260713"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def path_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def human_size(size):
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024


def assert_inside_oof(path):
    resolved = Path(path).resolve()
    if resolved == OOF_ROOT or OOF_ROOT not in resolved.parents:
        raise RuntimeError(f"Refusing unsafe path: {resolved}")
    return resolved


def preserve_reports(source_dir, prefix, names):
    source_dir = Path(source_dir)
    for name in names:
        source = source_dir / name
        if source.exists() and source.is_file():
            destination = REPORT_DIR / f"{prefix}__{name}"
            shutil.copy2(source, destination)
            print("preserved", destination)


MULTICUT_DIR = OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
DP_V0_DIR = OOF_ROOT / "global_dp_costvolume_v0_band80_step1_stride4_multiscale"
DP_V1_DIR = OOF_ROOT / "global_dp_costvolume_v1_fullgrid_step2_stride4_multiscale"
SURFACE_CACHE_DIR = OOF_ROOT / "global_surface_killtest_v2_last_anchor_all6_p80_q256"
PF_SEED_CACHE_DIR = OOF_ROOT / "pf_seed_cache_s16_p200"


# Keep the decisive, small reports outside directories that will be removed.
preserve_reports(
    MULTICUT_DIR,
    "multicut_v0",
    [
        "multicut_v0_summary.csv",
        "multicut_v0_folds.csv",
        "multicut_v0_bywell.csv",
        "multicut_v0_features.csv",
        "feature_status.csv",
        "well_metadata.csv",
        "cut_manifest.csv",
    ],
)
preserve_reports(
    DP_V0_DIR,
    "dp_v0",
    ["global_dp_summary.csv", "global_dp_folds.csv", "global_dp_verdict.csv"],
)
preserve_reports(
    DP_V1_DIR,
    "dp_v1",
    ["global_dp_summary.csv", "global_dp_folds.csv", "global_dp_verdict.csv"],
)


# Explicitly dead/rebuildable paths. The retained multi-cut OOF pickle is not
# listed here because it is still useful as an honest row-level benchmark.
DELETE_PATHS = [
    DP_V0_DIR,
    DP_V1_DIR,
    SURFACE_CACHE_DIR,
    PF_SEED_CACHE_DIR,
    MULTICUT_DIR / "features",
    MULTICUT_DIR / "masked_tmp",
    MULTICUT_DIR / "models",
    MULTICUT_DIR / "multicut_v0_oof.csv",  # duplicate of retained pickle
]

print("\nDELETE PLAN")
total_planned = 0
for path in DELETE_PATHS:
    size = path_size(path)
    total_planned += size
    print(f"{human_size(size):>12}  {path}  exists={path.exists()}")
print("planned free space:", human_size(total_planned))


# This list is deliberately explicit. Change to False for an audit-only run.
EXECUTE = True

if EXECUTE:
    freed = 0
    for raw_path in DELETE_PATHS:
        path = assert_inside_oof(raw_path)
        if not path.exists():
            continue
        size = path_size(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        freed += size
        print("deleted", human_size(size), path)
    print("\nfreed approximately:", human_size(freed))
else:
    print("Audit only; nothing deleted")


# Show remaining large files. Send this table back before deleting tier-2 OOFs.
large_files = []
for path in OOF_ROOT.rglob("*"):
    try:
        if path.is_file() and path.stat().st_size >= 100 * 1024**2:
            large_files.append(
                {
                    "size_gb": path.stat().st_size / 1024**3,
                    "path": str(path),
                }
            )
    except OSError:
        pass

large_files = sorted(large_files, key=lambda row: row["size_gb"], reverse=True)
print("\nREMAINING FILES >= 100 MB")
if large_files:
    display(__import__("pandas").DataFrame(large_files))
else:
    print("none")

print("\nretained multi-cut OOF:", MULTICUT_DIR / "multicut_v0_oof.pkl")
print("retained reports:", REPORT_DIR)
