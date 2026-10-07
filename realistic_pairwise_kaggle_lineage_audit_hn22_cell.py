"""HN22: read-only audit of the exact 6.377 -> HN13 -> HN16 submission lineage.

Run after the revised HN13 and HN16 cells. This cell never writes submission.csv.
Set ROGII_EXPECTED_PARENT_SHA256 to the hash of an independently archived 6.377
submission when available; that turns parent identity from inferred to proven.
"""

from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
FILES = {
    "parent": WORK / "submission_6p377_before_hn13.csv",
    "hn13_mean": WORK / "submission_hn13_score_mean_w0275.csv",
    "hn13_median": WORK / "submission_hn13_score_median_w0275.csv",
    "hn16": WORK / "submission_hn16_score_mean_residual.csv",
    "final": WORK / "submission.csv",
}
HN_WEIGHT = 0.275


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path, reference_ids=None):
    frame = pd.read_csv(path)
    if "id" not in frame or frame["id"].duplicated().any():
        raise RuntimeError(f"Invalid id column in {path}")
    value_column = next((name for name in ("tvt", "TVT") if name in frame), None)
    if value_column is None:
        raise RuntimeError(f"No TVT column in {path}: {frame.columns.tolist()}")
    ids = frame["id"].astype(str)
    if reference_ids is not None and not ids.equals(reference_ids):
        raise RuntimeError(f"ID order differs in {path}")
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite TVT values in {path}")
    return ids, values


def _stats(delta):
    delta = np.asarray(delta, np.float64)
    return {
        "mean_abs_ft": float(np.mean(np.abs(delta))),
        "rms_ft": float(np.sqrt(np.mean(delta * delta))),
        "p95_abs_ft": float(np.quantile(np.abs(delta), 0.95)),
        "maximum_abs_ft": float(np.max(np.abs(delta))),
        "changed_fraction_gt_1e_9": float(np.mean(np.abs(delta) > 1e-9)),
    }


for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")

ids, parent = _load(FILES["parent"])
values = {"parent": parent}
for name in ("hn13_mean", "hn13_median", "hn16", "final"):
    _, values[name] = _load(FILES[name], ids)

hn_mean = (values["hn13_mean"] - (1.0 - HN_WEIGHT) * parent) / HN_WEIGHT
hn_median = (values["hn13_median"] - (1.0 - HN_WEIGHT) * parent) / HN_WEIGHT
if not np.isfinite(hn_mean).all() or not np.isfinite(hn_median).all():
    raise RuntimeError("Reconstructed HN endpoint is non-finite")

hashes = {name: _sha256(path) for name, path in FILES.items()}
expected_parent_hash = os.environ.get("ROGII_EXPECTED_PARENT_SHA256", "").strip().lower()
alternative_parent_comparisons = {}
for name in ("final_submission.csv", "submission_6p377.csv"):
    path = WORK / name
    if not path.is_file():
        continue
    try:
        _, candidate = _load(path, ids)
    except Exception as exc:
        alternative_parent_comparisons[name] = {"error": repr(exc)}
        continue
    alternative_parent_comparisons[name] = {
        "sha256": _sha256(path),
        "maximum_abs_difference_ft": float(np.max(np.abs(candidate - parent))),
        "rms_difference_ft": float(np.sqrt(np.mean((candidate - parent) ** 2))),
    }

hn13_summary_path = WORK / "hn13_inference_summary.json"
hn16_summary_path = WORK / "hn16_inference_summary.json"
hn13_summary = json.loads(hn13_summary_path.read_text()) if hn13_summary_path.is_file() else {}
hn16_summary = json.loads(hn16_summary_path.read_text()) if hn16_summary_path.is_file() else {}

reported_model = Path(hn16_summary.get("hn15_model", ""))
reported_model_hash = hn16_summary.get("hn15_model_sha256")
model_hash_matches = None
if reported_model.is_file() and reported_model_hash:
    model_hash_matches = _sha256(reported_model) == str(reported_model_hash).lower()

parent_matches_alternative = any(
    isinstance(item, dict) and item.get("maximum_abs_difference_ft") == 0.0
    for item in alternative_parent_comparisons.values()
)
parent_hash_verified = bool(expected_parent_hash) and hashes["parent"] == expected_parent_hash
parent_identity_verified = parent_hash_verified or parent_matches_alternative

checks = {
    "all_ids_aligned": True,
    "final_equals_hn16": hashes["final"] == hashes["hn16"],
    "parent_differs_from_hn13_mean": hashes["parent"] != hashes["hn13_mean"],
    "parent_differs_from_hn13_median": hashes["parent"] != hashes["hn13_median"],
    "parent_differs_from_hn16": hashes["parent"] != hashes["hn16"],
    "mean_and_median_endpoints_differ": bool(np.any(np.abs(hn_mean - hn_median) > 1e-9)),
    "parent_identity_verified": parent_identity_verified,
    "reported_model_hash_matches": model_hash_matches,
}

verdict = {
    "experiment": "realistic_pairwise_kaggle_lineage_audit_hn22",
    "stage": "read_only_exact_submission_lineage_audit",
    "rows": len(ids),
    "wells": int(ids.str.rsplit("_", n=1).str[0].nunique()),
    "files": {
        name: {
            "path": str(path),
            "sha256": hashes[name],
            "size_bytes": path.stat().st_size,
            "modified_time": path.stat().st_mtime,
        }
        for name, path in FILES.items()
    },
    "hn13_mean_move_vs_parent": _stats(values["hn13_mean"] - parent),
    "hn13_median_move_vs_parent": _stats(values["hn13_median"] - parent),
    "hn16_move_vs_parent": _stats(values["hn16"] - parent),
    "mean_vs_median_raw_hn_endpoint": _stats(hn_mean - hn_median),
    "alternative_parent_comparisons": alternative_parent_comparisons,
    "expected_parent_sha256_supplied": bool(expected_parent_hash),
    "hn13_parent_source_before_snapshot": hn13_summary.get("parent_source_before_snapshot"),
    "hn16_model": str(reported_model) if str(reported_model) else None,
    "hn16_model_sha256": reported_model_hash,
    "checks": checks,
    "strict_pass": bool(all(value is True for value in checks.values() if value is not None)),
    "next": (
        "PASS -> lineage is exact; treat the LB miss as hidden-distribution failure"
        if parent_identity_verified and checks["final_equals_hn16"]
        else "INCOMPLETE -> provide an archived 6.377 SHA256 or an exact parent copy before another submission"
    ),
}

(WORK / "hn22_lineage_audit.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("HN22 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
