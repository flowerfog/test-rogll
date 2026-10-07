"""Read-only AutoDL disk audit before the canonical V67 experiment.

This script never deletes files. It inventories /root/autodl-tmp, protects
the current 6.377 lineage and V67 dependencies, and writes ranked cleanup
candidates to CSV and text files.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
REPORT_DIR = ROOT / "disk_cleanup_audit_v67"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# These artifacts are required to reproduce or deploy the current incumbent,
# or to run/resume V67. Keep their complete directories.
PROTECTED_OOF_NAMES = {
    "frontier_6p377_full_fivefold_lineage_v66",
    "frontier_6p39_full_oof_cp_increment_v18",
    "neighbor_fullcurve_transfer_v0",
    "foldclean_full_fleongg_lgb_v0",
    "frontier_6p39_existing_models_oof_v0",
    "frontier_6p39_cp8_nocp12_matched_v9",
    "rawdata_lattice_coarse_canonical_gateway_v67",
}

PROTECTED_ROOT_NAMES = {
    "train",
    "test",
    "rogii-claude-models-pub",
    "oof_exports",
    "rawdata_lattice_coarse_canonical_gateway_v67_cell.py",
    "rawdata_lattice_coarse_ranker_v4_cell.py",
    "rawdata_lattice_mode_union_v3_cell.py",
    "rawdata_lattice_beam_survival_v2_cell.py",
    "rawdata_segment_lattice_nested_v1_cell.py",
}

# V67 reads these directly from oof_exports rather than from an experiment
# directory. Exact existence depends on the user's artifact layout.
PROTECTED_OOF_FILES = {
    "candidate_pool_full_merged_oof_honest_fle.csv",
}

CACHE_NAMES = {
    "cache",
    "caches",
    "panels",
    "records",
    "record_cache",
    "seed_banks",
    "banks",
    "train_samples",
    "eval_parts",
    "__pycache__",
    ".ipynb_checkpoints",
}


def tree_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def verdict_state(path: Path) -> tuple[str, str]:
    verdict = path / "verdict.json"
    if not verdict.is_file():
        return "missing", ""
    try:
        payload = json.loads(verdict.read_text(encoding="utf-8"))
    except Exception as exc:
        return "invalid", repr(exc)
    value = payload.get("strict_pass")
    if value is True:
        return "pass", str(payload.get("next", ""))
    if value is False:
        return "fail", str(payload.get("next", ""))
    return "unknown", str(payload.get("next", ""))


def classify_oof(path: Path, size: int) -> tuple[str, str]:
    if path.name in PROTECTED_OOF_NAMES:
        return "KEEP", "current lineage, deployment artifact, or active V67"
    if path.is_file() and path.name in PROTECTED_OOF_FILES:
        return "KEEP", "direct V67 dependency"
    state, _ = verdict_state(path) if path.is_dir() else ("file", "")
    if state == "fail":
        return "DELETE_CANDIDATE", "completed experiment with strict_pass=false"
    if path.is_dir() and any(child.name in CACHE_NAMES for child in path.iterdir()):
        return "REVIEW_CACHE", "contains regenerable cache directory"
    if state == "pass":
        return "REVIEW_PASSED", "passed historical experiment; archive if no longer needed"
    if size >= 256 * 1024**2:
        return "REVIEW_LARGE", "large output without a readable final verdict"
    return "REVIEW", "historical or incomplete output"


def nested_caches() -> list[dict]:
    rows = []
    if not OOF_ROOT.is_dir():
        return rows
    for experiment in OOF_ROOT.iterdir():
        if not experiment.is_dir() or experiment.name in PROTECTED_OOF_NAMES:
            continue
        for path in experiment.rglob("*"):
            if path.is_dir() and path.name in CACHE_NAMES:
                size = tree_size(path)
                if size:
                    rows.append(
                        {
                            "path": str(path),
                            "size_gb": size / 1024**3,
                            "action": "DELETE_CACHE_CANDIDATE",
                            "reason": "regenerable cache from a non-protected experiment",
                        }
                    )
    return rows


if not ROOT.is_dir():
    raise FileNotFoundError(ROOT)

usage = shutil.disk_usage(ROOT)
print(
    "DISK",
    {
        "total_gb": round(usage.total / 1024**3, 3),
        "used_gb": round(usage.used / 1024**3, 3),
        "free_gb": round(usage.free / 1024**3, 3),
    },
    flush=True,
)

root_rows = []
for path in ROOT.iterdir():
    size = tree_size(path)
    if path.name in PROTECTED_ROOT_NAMES:
        action, reason = "KEEP", "raw data, model package, OOF root, or V67 source"
    elif path.name in CACHE_NAMES:
        action, reason = "DELETE_CANDIDATE", "top-level regenerable cache"
    elif path.is_file() and path.suffix.lower() in {".zip", ".tar", ".gz"}:
        action, reason = "REVIEW_ARCHIVE", "uploaded/generated archive; often duplicated after extraction"
    elif size >= 256 * 1024**2:
        action, reason = "REVIEW_LARGE", "large top-level artifact"
    else:
        action, reason = "REVIEW", "small or unclassified top-level artifact"
    root_rows.append(
        {
            "path": str(path),
            "size_gb": size / 1024**3,
            "action": action,
            "reason": reason,
        }
    )

oof_rows = []
if OOF_ROOT.is_dir():
    for path in OOF_ROOT.iterdir():
        size = tree_size(path)
        action, reason = classify_oof(path, size)
        state, next_action = verdict_state(path) if path.is_dir() else ("file", "")
        oof_rows.append(
            {
                "path": str(path),
                "size_gb": size / 1024**3,
                "action": action,
                "verdict": state,
                "reason": reason,
                "next": next_action,
            }
        )

root_frame = pd.DataFrame(root_rows).sort_values("size_gb", ascending=False)
oof_frame = pd.DataFrame(oof_rows).sort_values("size_gb", ascending=False)
cache_frame = pd.DataFrame(nested_caches())
if len(cache_frame):
    cache_frame = cache_frame.sort_values("size_gb", ascending=False)

root_frame.to_csv(REPORT_DIR / "root_inventory.csv", index=False)
oof_frame.to_csv(REPORT_DIR / "oof_inventory.csv", index=False)
cache_frame.to_csv(REPORT_DIR / "nested_cache_candidates.csv", index=False)

delete_frame = oof_frame[oof_frame["action"].eq("DELETE_CANDIDATE")].copy()
delete_paths = delete_frame["path"].tolist()
if len(cache_frame):
    delete_paths.extend(cache_frame["path"].tolist())
(REPORT_DIR / "candidate_paths.txt").write_text(
    "\n".join(dict.fromkeys(delete_paths)) + ("\n" if delete_paths else ""),
    encoding="utf-8",
)

print("\nTOP LEVEL (largest 25)")
print(root_frame.head(25).to_string(index=False), flush=True)
print("\nOOF OUTPUTS (largest 40)")
print(oof_frame.head(40).to_string(index=False), flush=True)
if len(cache_frame):
    print("\nREGENERABLE CACHES (largest 30)")
    print(cache_frame.head(30).to_string(index=False), flush=True)

candidate_bytes = int(delete_frame["size_gb"].sum() * 1024**3)
cache_bytes = int(cache_frame["size_gb"].sum() * 1024**3) if len(cache_frame) else 0
print(
    "\nAUDIT SUMMARY",
    {
        "failed_experiment_candidate_gb": round(candidate_bytes / 1024**3, 3),
        "nested_cache_candidate_gb": round(cache_bytes / 1024**3, 3),
        "report_dir": str(REPORT_DIR),
        "deletion_performed": False,
    },
    flush=True,
)
print("Read-only audit complete. Paste OOF OUTPUTS and AUDIT SUMMARY before deleting.")
