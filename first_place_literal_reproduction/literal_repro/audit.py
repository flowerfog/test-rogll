from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .constants import AUTHOR_SOURCE_HASHES, AUTHOR_SRC, REFERENCE_FILES, REFERENCE_OOF_RMSE


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def audit_sources() -> dict:
    rows = []
    for name, expected in AUTHOR_SOURCE_HASHES.items():
        path = AUTHOR_SRC / name
        actual = sha256(path) if path.is_file() else None
        rows.append({"file": name, "expected": expected, "actual": actual, "ok": actual == expected})
    return {"ok": all(row["ok"] for row in rows), "files": rows}


def audit_reference(reference_dir: Path, hash_large: bool) -> dict:
    rows = []
    for name, (expected_size, expected_hash) in REFERENCE_FILES.items():
        path = reference_dir / name
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        should_hash = exists and (hash_large or size < 256 * 1024 * 1024)
        actual_hash = sha256(path) if should_hash else None
        hash_ok = actual_hash == expected_hash if should_hash else None
        rows.append({
            "file": name,
            "exists": exists,
            "size": size,
            "expected_size": expected_size,
            "size_ok": size == expected_size,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "hash_ok": hash_ok,
        })
    log_path = reference_dir / "seq_nn.log"
    log_rmse = None
    if log_path.is_file():
        match = re.search(r"OOF RMSE:\s*([0-9.]+)", log_path.read_text(encoding="utf-8", errors="replace"))
        log_rmse = float(match.group(1)) if match else None
    ok = all(row["exists"] and row["size_ok"] and row["hash_ok"] is not False for row in rows)
    ok = ok and log_rmse is not None and abs(log_rmse - REFERENCE_OOF_RMSE) < 5e-5
    return {"ok": ok, "reference_dir": str(reference_dir), "log_oof_rmse": log_rmse, "files": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--hash-large", action="store_true", help="Also hash the 3.2 GB models.pkl.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"sources": audit_sources(), "reference": audit_reference(args.reference_dir, args.hash_large)}
    report["ok"] = report["sources"]["ok"] and report["reference"]["ok"]
    payload = json.dumps(report, indent=2)
    print(payload, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
