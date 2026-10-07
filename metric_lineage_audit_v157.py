"""V157: read-only metric and prediction-lineage audit.

The audit does not train or alter submissions.  It detects the common failure
mode where a local OOF metric uses a different row universe or endpoint than
the file used for Kaggle inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None


def _schema(path):
    if pq is None:
        raise RuntimeError("pyarrow is unavailable; parquet audit requires pyarrow")
    return pq.ParquetFile(path).schema.names


def _parquet_report(path):
    if pq is None:
        raise RuntimeError("pyarrow is unavailable; parquet audit requires pyarrow")
    schema = _schema(path)
    metadata = pq.ParquetFile(path).metadata
    report = {
        "path": str(path),
        "kind": "parquet",
        "rows": int(metadata.num_rows),
        "columns": schema,
    }
    usecols = [c for c in ("id", "well", "row_idx", "fold", "tvt_true", "md_since") if c in schema]
    if usecols:
        frame = pd.read_parquet(path, columns=usecols)
        report["wells"] = int(frame["well"].nunique()) if "well" in frame else None
        report["id_unique"] = bool(frame["id"].is_unique) if "id" in frame else None
        report["fold_counts"] = (
            {str(key): int(value) for key, value in frame["fold"].value_counts(dropna=False).sort_index().items()}
            if "fold" in frame else None
        )
        report["has_truth"] = "tvt_true" in frame
    return report


def _csv_report(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtype = {"well": str} if "well" in header else None
    frame = pd.read_csv(path, usecols=header[: min(len(header), 8)], dtype=dtype, low_memory=False)
    report = {"path": str(path), "kind": "csv", "rows": int(len(frame)), "columns": header}
    id_column = next((c for c in ("id", "well_id", "Index") if c in header), None)
    if id_column:
        values = frame[id_column].astype(str)
        report["id_unique"] = bool(values.is_unique)
        report["id_missing"] = int(values.isna().sum())
    return report


def _find_files(root):
    patterns = ("*.parquet", "*.csv", "*.csv.gz")
    paths = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(set(p for p in paths if p.is_file()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sample", type=Path, default=None)
    args, _unknown = parser.parse_known_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    reports = []
    for path in _find_files(root):
        try:
            if path.suffix == ".parquet":
                reports.append(_parquet_report(path))
            elif path.name.endswith(".csv") or path.name.endswith(".csv.gz"):
                reports.append(_csv_report(path))
        except Exception as exc:
            reports.append({"path": str(path), "status": "error", "error": repr(exc)})

    valid = [r for r in reports if r.get("status") != "error"]
    canonical = [r for r in valid if r.get("rows") in (2_270_643, 3_783_989)]
    submission = None
    if args.sample and args.sample.exists():
        submission = {"path": str(args.sample), "rows": int(len(pd.read_csv(args.sample)))}

    verdict = {
        "experiment": "metric_lineage_audit_v157",
        "root": str(root),
        "files_scanned": len(reports),
        "files_with_errors": [r for r in reports if r.get("status") == "error"],
        "canonical_scale_files": canonical,
        "sample_or_submission": submission,
        "gates": {
            "canonical_773_file_exists": any(r.get("rows") == 3_783_989 for r in canonical),
            "canonical_464_file_exists": any(r.get("rows") == 2_270_643 for r in canonical),
            "no_scan_errors": not any(r.get("status") == "error" for r in reports),
        },
        "next": "Freeze the exact 773-row endpoint and evaluate all candidates on its IDs" if canonical else "Upload or point to the canonical OOF directory before modeling",
    }
    output = root / "v157_metric_lineage_audit.json"
    output.write_text(json.dumps(verdict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(verdict, indent=2, ensure_ascii=False, default=str))
    print("saved:", output)


if __name__ == "__main__":
    main()
