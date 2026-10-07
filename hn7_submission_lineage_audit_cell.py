"""Read-only Kaggle audit for HN7 submission placement and parent lineage.

Run this after the HN7 inference cell. It does not rewrite any submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working")
PARENT = WORK / "submission_6p377_before_hn7.csv"
HN7_NAMED = WORK / "submission_hn7_w0275.csv"
FINAL = WORK / "submission.csv"
ALTERNATIVE_PARENTS = [
    WORK / "submission_6p377.csv",
    WORK / "final_submission.csv",
]
BASE_WEIGHT = 0.725
HN_WEIGHT = 0.275


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty or len(frame.columns) < 2:
        raise RuntimeError(f"Invalid submission: {path}")
    id_column = "id" if "id" in frame.columns else frame.columns[0]
    value_columns = [column for column in frame.columns if column != id_column]
    if len(value_columns) != 1:
        raise RuntimeError(f"Cannot identify prediction column in {path}: {frame.columns.tolist()}")
    value_column = value_columns[0]
    frame[id_column] = frame[id_column].astype(str)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    if frame[id_column].duplicated().any() or not np.isfinite(frame[value_column]).all():
        raise RuntimeError(f"Invalid ids or values in {path}")
    return frame, id_column, value_column


def _aligned_values(reference: pd.DataFrame, reference_id: str, path: Path) -> np.ndarray:
    frame, id_column, value_column = _load(path)
    aligned = reference[[reference_id]].merge(
        frame[[id_column, value_column]],
        left_on=reference_id,
        right_on=id_column,
        how="left",
        validate="one_to_one",
    )
    values = aligned[value_column].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Row alignment failed for {path}")
    return values


parent, parent_id, parent_value = _load(PARENT)
parent_values = parent[parent_value].to_numpy(np.float64)
named_values = _aligned_values(parent, parent_id, HN7_NAMED)
final_values = _aligned_values(parent, parent_id, FINAL)
implied_hn = (named_values - BASE_WEIGHT * parent_values) / HN_WEIGHT
move = named_values - parent_values

alternatives = {}
for path in ALTERNATIVE_PARENTS:
    if not path.exists():
        continue
    values = _aligned_values(parent, parent_id, path)
    alternatives[path.name] = {
        "sha256": _sha256(path),
        "maximum_abs_difference_vs_snapshot": float(np.max(np.abs(values - parent_values))),
        "rms_difference_vs_snapshot": float(np.sqrt(np.mean(np.square(values - parent_values)))),
    }

report = {
    "experiment": "hn7_submission_lineage_audit_v0",
    "files": {
        path.name: {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "modified_time": path.stat().st_mtime,
        }
        for path in (PARENT, HN7_NAMED, FINAL)
    },
    "rows": int(len(parent)),
    "weights": {"parent": BASE_WEIGHT, "hn7": HN_WEIGHT},
    "named_vs_final": {
        "maximum_abs_difference": float(np.max(np.abs(named_values - final_values))),
        "rms_difference": float(np.sqrt(np.mean(np.square(named_values - final_values)))),
    },
    "hn7_move_vs_parent": {
        "changed_fraction_gt_1e_9": float(np.mean(np.abs(move) > 1e-9)),
        "mean_abs_ft": float(np.mean(np.abs(move))),
        "rms_ft": float(np.sqrt(np.mean(np.square(move)))),
        "p95_abs_ft": float(np.quantile(np.abs(move), 0.95)),
        "maximum_abs_ft": float(np.max(np.abs(move))),
    },
    "implied_hn_endpoint": {
        "finite": bool(np.isfinite(implied_hn).all()),
        "mean_abs_disagreement_vs_parent_ft": float(np.mean(np.abs(implied_hn - parent_values))),
        "rms_disagreement_vs_parent_ft": float(
            np.sqrt(np.mean(np.square(implied_hn - parent_values)))
        ),
    },
    "alternative_parent_comparisons": alternatives,
    "checks": {
        "named_hn7_equals_final_submission": bool(np.array_equal(named_values, final_values)),
        "final_submission_is_not_parent": bool(np.any(np.abs(final_values - parent_values) > 1e-9)),
        "snapshot_predates_named_hn7": bool(PARENT.stat().st_mtime <= HN7_NAMED.stat().st_mtime),
    },
}
report["strict_pass"] = bool(all(report["checks"].values()))
print(json.dumps(report, indent=2), flush=True)

if not report["strict_pass"]:
    raise RuntimeError("HN7 placement/lineage audit failed; inspect the report above")
