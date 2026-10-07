"""Inventory only the OOF artifacts explicitly declared by the model package.

Run on AutoDL after uploading the model package under /root/autodl-tmp. This
cell is read-only apart from its compact audit outputs. It deliberately does
not scan the rest of /root/autodl-tmp for files whose names contain "oof".
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "frontier_6p39_modelpkg_oof_inventory_v13"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sha256_head(path: Path, limit: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def _flatten_oof_paths(value, prefix="oof"):
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_oof_paths(child, child_prefix))
    elif isinstance(value, str):
        rows.append((prefix, value))
    return rows


def _parquet_metadata(path: Path) -> dict:
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(path).metadata
        schema = pq.ParquetFile(path).schema_arrow
        return {
            "rows": int(metadata.num_rows),
            "columns": [str(name) for name in schema.names],
            "column_count": int(len(schema.names)),
        }
    except Exception:
        frame = pd.read_parquet(path)
        return {
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "column_count": int(len(frame.columns)),
        }


def _npy_metadata(path: Path) -> dict:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    flat = values.reshape(-1)
    if len(flat) <= 200_000:
        sample = np.asarray(flat)
    else:
        index = np.linspace(0, len(flat) - 1, 200_000, dtype=np.int64)
        sample = np.asarray(flat[index])
    numeric = np.issubdtype(values.dtype, np.number)
    return {
        "shape": [int(value) for value in values.shape],
        "rows": int(values.shape[0]) if values.ndim else 1,
        "dtype": str(values.dtype),
        "sample_finite_fraction": (
            float(np.isfinite(sample).mean()) if numeric and len(sample) else None
        ),
        "sample_min": float(np.nanmin(sample)) if numeric and len(sample) else None,
        "sample_max": float(np.nanmax(sample)) if numeric and len(sample) else None,
    }


manifest_paths = sorted(ROOT.glob("**/metadata/model_package_manifest.json"))
if not manifest_paths:
    raise FileNotFoundError(
        "No metadata/model_package_manifest.json found below /root/autodl-tmp"
    )

package_reports = []
artifact_rows = []
for manifest_path in manifest_paths:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_root = manifest_path.parents[1]
    expected_rows = int(manifest.get("oof_rows_used", 0) or 0)
    declared = _flatten_oof_paths(manifest.get("oof", {}))

    package_artifacts = []
    for logical_name, relative in declared:
        raw_path = Path(relative)
        path = raw_path if raw_path.is_absolute() else package_root / raw_path
        row = {
            "manifest": str(manifest_path),
            "package_root": str(package_root),
            "logical_name": logical_name,
            "declared_path": relative,
            "resolved_path": str(path),
            "exists": bool(path.exists()),
            "size_mb": float(path.stat().st_size / 2**20) if path.exists() else None,
            "sha256_first_8mb": _sha256_head(path) if path.exists() else None,
        }
        if path.exists():
            try:
                if path.suffix.lower() == ".npy":
                    row.update(_npy_metadata(path))
                elif path.suffix.lower() in {".parquet", ".pq"}:
                    row.update(_parquet_metadata(path))
                else:
                    row["inspection_warning"] = "unsupported declared OOF format"
            except Exception as exc:
                row["inspection_error"] = repr(exc)
        package_artifacts.append(row)
        artifact_rows.append(row)

    existing = [row for row in package_artifacts if row["exists"]]
    row_matched = [
        row
        for row in existing
        if isinstance(row.get("rows"), int) and row["rows"] == expected_rows
    ]
    arrays = [row for row in package_artifacts if row["logical_name"] != "oof.train_gt"]
    train_gt = next(
        (row for row in package_artifacts if row["logical_name"] == "oof.train_gt"),
        None,
    )
    train_columns = set(train_gt.get("columns", [])) if train_gt else set()
    has_key_candidate = bool(
        {"id", "row_idx", "well", "well_id", "target", "tvt_true", "TVT"}
        & train_columns
    )
    all_declared_exist = bool(package_artifacts) and all(
        row["exists"] for row in package_artifacts
    )
    all_rows_match = bool(package_artifacts) and len(row_matched) == len(package_artifacts)
    arrays_usable = bool(arrays) and all(
        row["exists"]
        and row.get("rows") == expected_rows
        and row.get("sample_finite_fraction") == 1.0
        for row in arrays
    )

    imputer_mode = manifest.get("oof_imputer_mode")
    imputer_note = manifest.get("oof_imputer_note")
    status = "unusable_missing_or_misaligned_declared_oof"
    if all_declared_exist and all_rows_match and arrays_usable and has_key_candidate:
        status = "usable_matched_oof_with_imputer_caveat"

    package_reports.append(
        {
            "manifest": str(manifest_path),
            "package_root": str(package_root),
            "expected_oof_rows": expected_rows,
            "n_splits": manifest.get("n_splits"),
            "folds_run": manifest.get("folds_run"),
            "oof_complete_rate": manifest.get("oof_complete_rate"),
            "declared_artifacts": int(len(package_artifacts)),
            "all_declared_exist": all_declared_exist,
            "all_rows_match": all_rows_match,
            "arrays_usable": arrays_usable,
            "train_gt_has_alignment_key_candidate": has_key_candidate,
            "oof_imputer_mode": imputer_mode,
            "oof_imputer_note": imputer_note,
            "strict_outer_fold_rebuild": False,
            "matched_replay_status": status,
        }
    )

artifact_frame = pd.DataFrame(artifact_rows)
artifact_frame.to_csv(OUT_DIR / "declared_oof_artifacts.csv", index=False)

usable_packages = [
    report
    for report in package_reports
    if report["matched_replay_status"] == "usable_matched_oof_with_imputer_caveat"
]
verdict = {
    "experiment": "frontier_6p39_modelpkg_oof_inventory_v13",
    "scope": "manifest-declared model-package OOF only",
    "packages": package_reports,
    "usable_package_count": int(len(usable_packages)),
    "strict_foldclean_claim": False,
    "interpretation": (
        "Persisted five-fold predictions can be used for a matched incremental OOF "
        "replay, but the manifest says formation/KNN imputers were not rebuilt per "
        "validation fold, so they are not evidence for a strict outer-fold score."
    ),
    "next": (
        "build matched model-package replay on parent and CP8 endpoints"
        if usable_packages
        else "repair or upload the manifest-declared OOF artifacts"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("MODEL-PACKAGE DECLARED OOF ARTIFACTS")
print(artifact_frame.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
