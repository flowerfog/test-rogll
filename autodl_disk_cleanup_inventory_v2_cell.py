"""AutoDL disk inventory and guarded cleanup planner.

Run this cell after the interrupted PSG archive operation.  It removes only
the explicitly known interrupted root-level ZIP when enabled, then inventories
all OOF experiments and large files.  Experiment directories are never
deleted unless their exact names are copied into SELECTED_DELETE_NAMES and
EXECUTE_SELECTED_DELETE is changed to True.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import time
import zipfile

import pandas as pd


ROOT = Path("/root/autodl-tmp").resolve()
OOF_ROOT = (ROOT / "oof_exports").resolve()
INTERRUPTED_PSG_ZIP = (ROOT / "psg_foldclean_fivefold_oof_v0.zip").resolve()
PSG_ARTIFACT_DIR = (OOF_ROOT / "psg_foldclean_fivefold_oof_v0").resolve()

# This exact archive was left by the failed shutil.make_archive call shown in
# the traceback.  Deleting it does not touch the completed artifact directory.
DELETE_INTERRUPTED_PSG_ZIP = True

# The training cell writes both formats, while downstream calibration/inference
# uses parquet.  CSV is removed only after parquet metadata and manifest agree.
DELETE_VERIFIED_PSG_CSV = True

# Audit first.  Copy exact top-level names from the candidate table only after
# reviewing it, then change this switch to True in a second run.
EXECUTE_SELECTED_DELETE = False
SELECTED_DELETE_NAMES = [
    # "old_failed_experiment_name",
]

LARGE_FILE_MB = 100
REPORT_COPY_LIMIT_MB = 20


# These artifacts are active, expensive to rebuild, or part of the 6.781/PSG/
# current-SEGN lineage.  A selected deletion is refused even if a name appears
# below.  Prefix entries protect versioned descendants as well.
PROTECTED_EXACT = {
    "foldclean_full_fleongg_lgb_v0",
    "multicut_fleongg_matched_k3_gateway_v1",
    "native_resolution_dualref_capacity_v0",
    "neighbor_fullcurve_transfer_v0",
    "neighbor_fullcurve_projection_direction_v1",
    "psg_foldclean_fivefold_oof_v0",
    "segn_r4_worknote_scheduled_costmap_fold3_v0",
    "segn_r5_axisaware_evidence_fold3_v0",
    "segn_r5_frozen_logit_dose_audit_v1",
    "segn_r6_densemd_evidence_fold3_v0",
}
PROTECTED_PREFIXES = (
    "candidate_pool_full_merged",
    "keep_reports_",
)


def _human_size(size):
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _disk(label):
    usage = shutil.disk_usage(ROOT)
    row = {
        "label": label,
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
    }
    print("DISK", row, flush=True)
    return row


def _path_size(path):
    path = Path(path)
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total = 0
    if not path.exists():
        return total
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _is_protected(name):
    return name in PROTECTED_EXACT or any(
        name.startswith(prefix) for prefix in PROTECTED_PREFIXES
    )


def _inside_oof(path):
    resolved = Path(path).resolve()
    if resolved == OOF_ROOT or OOF_ROOT not in resolved.parents:
        raise RuntimeError(f"Refusing path outside an OOF child: {resolved}")
    return resolved


def _load_verdict(directory):
    preferred = (
        "verdict.json",
        "result.json",
        "final_verdict.json",
        "summary.json",
    )
    candidates = [directory / name for name in preferred]
    candidates.extend(sorted(directory.glob("*verdict*.json")))
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        strict = payload.get("strict_pass")
        if strict is None:
            strict = payload.get("pass")
        return path, payload, strict
    return None, None, None


def _classify(directory, strict):
    if _is_protected(directory.name):
        return "PROTECTED"
    if strict is False:
        return "FAILED_VERDICT_CANDIDATE"
    if strict is True:
        return "PASSED_REVIEW_BEFORE_DELETE"
    return "NO_VERDICT_REVIEW"


def _zip_status(path):
    if not path.exists():
        return "missing"
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            return "valid" if bad_member is None else f"corrupt:{bad_member}"
    except Exception as error:
        return f"unreadable:{type(error).__name__}"


def _preserve_small_reports(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    patterns = ("*.json", "*summary*.csv", "*verdict*.csv", "*metrics*.csv")
    copied = 0
    for pattern in patterns:
        for path in source.glob(pattern):
            try:
                if not path.is_file() or path.stat().st_size > REPORT_COPY_LIMIT_MB * 1024**2:
                    continue
                target = destination / f"{source.name}__{path.name}"
                shutil.copy2(path, target)
                copied += 1
            except OSError:
                pass
    return copied


before = _disk("before")

# Emergency recovery from the exact failed operation.  The source directory
# with parquet and five checkpoints remains protected.
zip_size = _path_size(INTERRUPTED_PSG_ZIP)
zip_state = _zip_status(INTERRUPTED_PSG_ZIP)
print(
    "INTERRUPTED ARCHIVE",
    {
        "path": str(INTERRUPTED_PSG_ZIP),
        "exists": INTERRUPTED_PSG_ZIP.exists(),
        "size": _human_size(zip_size),
        "zip_state": zip_state,
        "delete_enabled": DELETE_INTERRUPTED_PSG_ZIP,
    },
    flush=True,
)
if DELETE_INTERRUPTED_PSG_ZIP and INTERRUPTED_PSG_ZIP.exists():
    INTERRUPTED_PSG_ZIP.unlink()
    print("deleted interrupted archive:", _human_size(zip_size), flush=True)

after_archive = _disk("after_interrupted_archive")


def _verify_psg_parquet():
    parquet_path = PSG_ARTIFACT_DIR / "psg_oof.parquet"
    csv_path = PSG_ARTIFACT_DIR / "psg_oof.csv"
    manifest_path = PSG_ARTIFACT_DIR / "psg_training_manifest.json"
    result = {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "parquet_exists": parquet_path.is_file(),
        "csv_exists": csv_path.is_file(),
        "verified": False,
        "reason": "",
    }
    if not parquet_path.is_file() or not manifest_path.is_file():
        result["reason"] = "missing parquet or manifest"
        return result
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        parquet_rows = int(parquet_file.metadata.num_rows)
        columns = set(parquet_file.schema.names)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_rows = int(manifest["oof_rows"])
        required = {"well", "row_idx", "psg_tvt", "true_tvt"}
        result.update(
            {
                "parquet_rows": parquet_rows,
                "manifest_rows": manifest_rows,
                "columns": sorted(columns),
                "required_columns_present": required.issubset(columns),
            }
        )
        result["verified"] = bool(
            parquet_rows == manifest_rows and required.issubset(columns)
        )
        if not result["verified"]:
            result["reason"] = "parquet metadata does not match manifest/schema"
    except Exception as error:
        result["reason"] = repr(error)
    return result


psg_verification = _verify_psg_parquet()
print("PSG PARQUET VERIFICATION", psg_verification, flush=True)
psg_csv = PSG_ARTIFACT_DIR / "psg_oof.csv"
if DELETE_VERIFIED_PSG_CSV and psg_verification["verified"] and psg_csv.exists():
    csv_size = _path_size(psg_csv)
    psg_csv.unlink()
    print("deleted verified duplicate PSG CSV:", _human_size(csv_size), flush=True)
elif DELETE_VERIFIED_PSG_CSV and psg_csv.exists():
    print("kept PSG CSV because parquet verification did not pass", flush=True)

after_safe_duplicates = _disk("after_safe_duplicates")


# Top-level OOF inventory with verdict-aware classification.
experiment_rows = []
if OOF_ROOT.exists():
    for directory in sorted(OOF_ROOT.iterdir()):
        if not directory.is_dir():
            continue
        size = _path_size(directory)
        verdict_path, verdict, strict = _load_verdict(directory)
        experiment_rows.append(
            {
                "name": directory.name,
                "size_gb": size / 1024**3,
                "size": _human_size(size),
                "class": _classify(directory, strict),
                "strict_pass": strict,
                "verdict": str(verdict_path) if verdict_path else "",
                "next": str(verdict.get("next", "")) if verdict else "",
            }
        )

experiments = pd.DataFrame(experiment_rows)
if len(experiments):
    experiments = experiments.sort_values("size_gb", ascending=False).reset_index(drop=True)
    print("\nTOP OOF DIRECTORIES")
    print(experiments.head(80).to_string(index=False), flush=True)
    failed = experiments.loc[experiments["class"].eq("FAILED_VERDICT_CANDIDATE")]
    print("\nFAILED-VERDICT CANDIDATES")
    if len(failed):
        print(failed.to_string(index=False), flush=True)
        print(
            "failed-candidate total:",
            _human_size(int(failed["size_gb"].sum() * 1024**3)),
            flush=True,
        )
    else:
        print("none")


# Largest individual files reveal caches hidden inside otherwise useful dirs.
large_rows = []
if OOF_ROOT.exists():
    for path in OOF_ROOT.rglob("*"):
        try:
            if path.is_file() and path.stat().st_size >= LARGE_FILE_MB * 1024**2:
                large_rows.append(
                    {
                        "size_gb": path.stat().st_size / 1024**3,
                        "size": _human_size(path.stat().st_size),
                        "protected_parent": _is_protected(path.relative_to(OOF_ROOT).parts[0]),
                        "path": str(path),
                    }
                )
        except OSError:
            pass
large_files = pd.DataFrame(large_rows)
if len(large_files):
    large_files = large_files.sort_values("size_gb", ascending=False).reset_index(drop=True)
    print("\nFILES >=", LARGE_FILE_MB, "MB")
    print(large_files.head(100).to_string(index=False), flush=True)


# CSV is usually much larger than parquet.  These pairs are low-risk cleanup
# candidates only after the parquet columns/row count are verified.
duplicate_rows = []
if OOF_ROOT.exists():
    for csv_path in OOF_ROOT.rglob("*.csv"):
        parquet_path = csv_path.with_suffix(".parquet")
        if not parquet_path.is_file():
            continue
        try:
            duplicate_rows.append(
                {
                    "csv_gb": csv_path.stat().st_size / 1024**3,
                    "parquet_gb": parquet_path.stat().st_size / 1024**3,
                    "recoverable_gb": csv_path.stat().st_size / 1024**3,
                    "csv": str(csv_path),
                    "parquet": str(parquet_path),
                }
            )
        except OSError:
            pass
duplicates = pd.DataFrame(duplicate_rows)
if len(duplicates):
    duplicates = duplicates.sort_values("recoverable_gb", ascending=False).reset_index(drop=True)
    print("\nCSV/PARQUET DUPLICATE PAIRS - VERIFY PARQUET BEFORE DELETING CSV")
    print(duplicates.to_string(index=False), flush=True)


# Root-level caches and source uploads are listed separately.  This scan does
# not descend into the raw competition dataset.
root_rows = []
for path in sorted(ROOT.iterdir()):
    if path == OOF_ROOT or path.name in {"rogii-wellbore-geology-prediction"}:
        continue
    try:
        size = _path_size(path)
        root_rows.append(
            {
                "name": path.name,
                "kind": "dir" if path.is_dir() else "file",
                "size_gb": size / 1024**3,
                "size": _human_size(size),
                "path": str(path),
            }
        )
    except OSError:
        pass
root_table = pd.DataFrame(root_rows)
if len(root_table):
    root_table = root_table.sort_values("size_gb", ascending=False).reset_index(drop=True)
    print("\nROOT-LEVEL ITEMS EXCLUDING DATASET AND OOF_ROOT")
    print(root_table.head(80).to_string(index=False), flush=True)


if EXECUTE_SELECTED_DELETE:
    if not SELECTED_DELETE_NAMES:
        raise RuntimeError("EXECUTE_SELECTED_DELETE=True but no names were selected")
    report_dir = (ROOT / "cleanup_preserved_reports").resolve()
    planned = []
    for name in SELECTED_DELETE_NAMES:
        if Path(name).name != name:
            raise RuntimeError(f"Use only an exact top-level OOF name: {name!r}")
        if _is_protected(name):
            raise RuntimeError(f"Refusing protected experiment: {name}")
        path = _inside_oof(OOF_ROOT / name)
        if path.exists():
            planned.append((path, _path_size(path)))
    print("\nSELECTED DELETE PLAN")
    for path, size in planned:
        print(f"{_human_size(size):>12}  {path}")
    print("selected total:", _human_size(sum(size for _, size in planned)))
    for path, size in planned:
        copied = _preserve_small_reports(path, report_dir)
        shutil.rmtree(path)
        print("deleted", _human_size(size), path, "preserved_reports=", copied)
    print("preserved reports:", report_dir)
else:
    print(
        "\nAUDIT ONLY: no experiment directory deleted. Copy exact names into "
        "SELECTED_DELETE_NAMES and enable EXECUTE_SELECTED_DELETE after review.",
        flush=True,
    )

final_disk = _disk("final")

# Save small tables only after emergency archive removal has restored space.
report_root = ROOT / "disk_inventory"
try:
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if len(experiments):
        experiments.to_csv(report_root / f"experiments_{timestamp}.csv", index=False)
    if len(large_files):
        large_files.to_csv(report_root / f"large_files_{timestamp}.csv", index=False)
    if len(duplicates):
        duplicates.to_csv(report_root / f"duplicates_{timestamp}.csv", index=False)
    print("inventory tables:", report_root)
except OSError as error:
    print("could not save inventory tables:", repr(error))
