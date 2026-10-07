from __future__ import annotations

import csv
from pathlib import Path


DISCLOSED = {"exact", "source-derived"}


def validate_contract(path: str | Path, gate: str = "training") -> dict[str, int]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("empty reproduction matrix")
    failures = []
    for row in rows:
        required = row["required"].lower() == "true"
        disclosed = row["disclosed"].lower() == "true"
        status = row["status"]
        if status in DISCLOSED and (not row["evidence"] or not row["parity_test"]):
            failures.append(f"{row['component']}: disclosed row lacks evidence/test")
        if gate == "training" and required:
            if status == "missing":
                failures.append(f"{row['component']}: missing")
            if disclosed and status not in DISCLOSED:
                failures.append(f"{row['component']}: disclosed but {status}")
            if status == "inferred" and (not row["implementation"] or not row["parity_test"]):
                failures.append(f"{row['component']}: inferred without implementation/test")
    if failures:
        raise RuntimeError("REPRODUCTION GATE FAILED\n" + "\n".join(failures))
    counts = {status: sum(row["status"] == status for row in rows) for status in DISCLOSED | {"inferred", "missing"}}
    print("REPRODUCTION GATE PASSED", counts, flush=True)
    return counts
