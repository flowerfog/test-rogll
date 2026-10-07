"""Audit and preserve the deterministic seed-1 V116 consensus submission.

Run after, in order: the V195 seed patch, the original mainline, V69, V77,
the V195 finalizer, and the V116 consensus-center Kaggle cell.
"""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd


WORK_V196 = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
SOURCE_V196 = WORK_V196 / "submission.csv"
V116_OUTPUT_V196 = WORK_V196 / "submission_v116_consensus_center.csv"
V77_BASE_V196 = WORK_V196 / "submission_v116_v76_base.csv"
V116_SUMMARY_V196 = WORK_V196 / "v116_consensus_center_summary.json"
OUTPUT_V196 = WORK_V196 / "submission_v196_seed1_v116_consensus.csv"
SUMMARY_V196 = WORK_V196 / "v196_seed1_consensus_summary.json"


def _sha256_v196(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if not globals().get("PF_SEED1_PATCH_INSTALLED_V195", False):
    raise RuntimeError("V195 deterministic seed-1 patch was not installed")
for required_v196 in (SOURCE_V196, V116_OUTPUT_V196, V77_BASE_V196, V116_SUMMARY_V196):
    if not required_v196.is_file():
        raise FileNotFoundError(required_v196)

current_v196 = pd.read_csv(SOURCE_V196, dtype={"id": "string"})
named_v116_v196 = pd.read_csv(V116_OUTPUT_V196, dtype={"id": "string"})
base_v195_v196 = pd.read_csv(V77_BASE_V196, dtype={"id": "string"})
for label_v196, frame_v196 in (
    ("current", current_v196),
    ("v116", named_v116_v196),
    ("v195", base_v195_v196),
):
    if frame_v196.columns.tolist() != ["id", "tvt"]:
        raise RuntimeError(f"{label_v196}: unexpected columns {frame_v196.columns.tolist()}")
    if frame_v196["id"].duplicated().any():
        raise RuntimeError(f"{label_v196}: duplicate IDs")
    values_v196 = pd.to_numeric(frame_v196["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values_v196).all():
        raise RuntimeError(f"{label_v196}: non-finite predictions")

if not current_v196["id"].equals(named_v116_v196["id"]):
    raise RuntimeError("submission.csv and V116 named output IDs differ")
if not current_v196["id"].equals(base_v195_v196["id"]):
    raise RuntimeError("V195 and V116 output IDs differ")
current_values_v196 = current_v196["tvt"].to_numpy(float)
named_values_v196 = named_v116_v196["tvt"].to_numpy(float)
if not np.array_equal(current_values_v196, named_values_v196):
    gap_v196 = float(np.max(np.abs(current_values_v196 - named_values_v196)))
    raise RuntimeError(f"V116 named/current prediction mismatch: {gap_v196}")

calls_v196 = pd.DataFrame(globals().get("PF_SEED1_CALLS_V195", []))
if calls_v196.empty:
    raise RuntimeError("No deterministic PF calls were recorded")
test_calls_v196 = calls_v196.loc[~calls_v196["is_train"]].copy()
submission_wells_v196 = current_v196["id"].str.rsplit("_", n=1).str[0].nunique()
if test_calls_v196["well"].nunique() != submission_wells_v196:
    raise RuntimeError(
        "Seed audit/test coverage mismatch: "
        f"{test_calls_v196['well'].nunique()}/{submission_wells_v196}"
    )
if test_calls_v196.groupby("well")["seed"].nunique().max() != 1:
    raise RuntimeError("A test well received multiple deterministic seeds")

v116_summary_v196 = json.loads(V116_SUMMARY_V196.read_text(encoding="utf-8"))
if int(v116_summary_v196.get("fallback_wells") and len(v116_summary_v196["fallback_wells"]) or 0) > 0:
    raise RuntimeError(f"V116 used fallbacks: {v116_summary_v196['fallback_wells']}")
if float(v116_summary_v196.get("coverage", 0.0)) < 0.98:
    raise RuntimeError(f"V116 coverage too low: {v116_summary_v196.get('coverage')}")
if float(v116_summary_v196.get("frozen_prefix_max_abs_difference", np.inf)) != 0.0:
    raise RuntimeError("V116 changed its frozen prefix")

base_values_v196 = base_v195_v196["tvt"].to_numpy(float)
delta_v196 = current_values_v196 - base_values_v196
if not np.any(np.abs(delta_v196) > 1e-12):
    raise RuntimeError("V116 did not change any V195 prediction")

# Kaggle auto-submits submission.csv. Keep a byte-identical named archive.
current_v196.to_csv(SOURCE_V196, index=False)
shutil.copyfile(SOURCE_V196, OUTPUT_V196)
summary_v196 = {
    "experiment": "kaggle_pf_seed1_v116_consensus_v196",
    "parent": "deterministic seed-1 V69 All773 plus V77 adaptive dose",
    "overlay": "frozen V116 top33 uniform consensus",
    "rows": int(len(current_v196)),
    "wells": int(submission_wells_v196),
    "seeded_test_wells": int(test_calls_v196["well"].nunique()),
    "v116_coverage": float(v116_summary_v196["coverage"]),
    "v116_fallback_wells": list(v116_summary_v196["fallback_wells"]),
    "changed_rows": int(np.count_nonzero(np.abs(delta_v196) > 1e-12)),
    "changed_rows_fraction": float(np.mean(np.abs(delta_v196) > 1e-12)),
    "mean_abs_change_ft": float(np.mean(np.abs(delta_v196))),
    "p95_abs_change_ft": float(np.quantile(np.abs(delta_v196), 0.95)),
    "maximum_abs_change_ft": float(np.max(np.abs(delta_v196))),
    "v77_base_sha256": _sha256_v196(V77_BASE_V196),
    "submission_sha256": _sha256_v196(OUTPUT_V196),
    "interactive_rows_note": (
        "Three visible example wells are expected while authoring; Kaggle's "
        "hidden rerun replaces them with the competition test wells."
    ),
    "auto_submit": str(SOURCE_V196),
    "named_archive": str(OUTPUT_V196),
}
SUMMARY_V196.write_text(json.dumps(summary_v196, indent=2), encoding="utf-8")
print("V196 FINAL SUMMARY")
print(json.dumps(summary_v196, indent=2), flush=True)
print("KAGGLE AUTO-SUBMITS:", SOURCE_V196, flush=True)
