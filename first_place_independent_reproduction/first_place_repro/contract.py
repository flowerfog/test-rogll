from __future__ import annotations

import csv
from pathlib import Path


def validate_reproduction_matrix(path: str | Path) -> dict[str, int]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    for row in rows:
        if row["required"].lower() == "true" and row["status"] == "missing":
            failures.append(f"{row['component']}: missing")
        if row["disclosed"].lower() == "true" and row["status"] not in {"exact", "source-derived"}:
            failures.append(f"{row['component']}: disclosed but {row['status']}")
        if not row["implementation"] or not row["parity_test"]:
            failures.append(f"{row['component']}: no implementation/test")
    if failures:
        raise RuntimeError("REPRODUCTION CONTRACT FAILED\n" + "\n".join(failures))
    counts = {status: sum(row["status"] == status for row in rows) for status in {"exact", "source-derived", "inferred", "missing"}}
    return counts


def assert_clean_room(root: str | Path) -> None:
    forbidden = ("archive (3)", "seq_nn_", "cfg.pkl", "models.pkl", "author_output")
    hits = []
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".pt", ".pkl"}:
            continue
        if path.name in {"README.md", "contract.py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in forbidden:
            if token in text:
                hits.append(f"{path}: {token}")
    if hits:
        raise RuntimeError("AUTHOR DEPENDENCY DETECTED\n" + "\n".join(hits))
