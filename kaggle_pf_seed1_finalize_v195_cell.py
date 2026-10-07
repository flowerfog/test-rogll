"""Audit and preserve the seed-1 V69+V77 submission candidate."""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd


WORK_V195 = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
SOURCE_V195 = WORK_V195 / "submission.csv"
OUTPUT_V195 = WORK_V195 / "submission_v195_seed1_v69_v77.csv"
SUMMARY_V195 = WORK_V195 / "v195_seed1_submission_summary.json"

if not globals().get("PF_SEED1_PATCH_INSTALLED_V195", False):
    raise RuntimeError("The V195 seed-1 patch was not installed")
if not SOURCE_V195.is_file():
    raise FileNotFoundError(SOURCE_V195)
if not (WORK_V195 / "v77_adaptive_dose_summary.json").is_file():
    raise RuntimeError("Run V69 and then V77 before the V195 final audit")

submission_v195 = pd.read_csv(SOURCE_V195, dtype={"id": "string"})
if submission_v195.columns.tolist() != ["id", "tvt"]:
    raise RuntimeError(f"Unexpected submission columns: {submission_v195.columns.tolist()}")
if submission_v195["id"].duplicated().any():
    raise RuntimeError("Duplicate submission IDs")
values_v195 = pd.to_numeric(submission_v195["tvt"], errors="coerce").to_numpy(float)
if not np.isfinite(values_v195).all():
    raise RuntimeError("Non-finite submission predictions")

calls_v195 = pd.DataFrame(globals().get("PF_SEED1_CALLS_V195", []))
if calls_v195.empty:
    raise RuntimeError("The patched build_well was never called")
test_calls_v195 = calls_v195.loc[~calls_v195["is_train"]].copy()
submission_wells_v195 = submission_v195["id"].str.rsplit("_", n=1).str[0].nunique()
if test_calls_v195["well"].nunique() != submission_wells_v195:
    raise RuntimeError(
        "Seed audit/test coverage mismatch: "
        f"{test_calls_v195['well'].nunique()}/{submission_wells_v195}"
    )
if test_calls_v195.groupby("well")["seed"].nunique().max() != 1:
    raise RuntimeError("A test well received more than one deterministic seed")

# Kaggle Code Competition auto-submits this exact path.  The named file is
# only an identical archival copy for debugging after the rerun.
submission_v195.to_csv(SOURCE_V195, index=False)
shutil.copyfile(SOURCE_V195, OUTPUT_V195)


def _sha256_v195(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


summary_v195 = {
    "experiment": "kaggle_pf_seed1_v69_v77_v195",
    "recipe_seed": PF_RECIPE_SEED_V195,
    "rows": int(len(submission_v195)),
    "wells": int(submission_wells_v195),
    "seeded_build_calls": int(len(calls_v195)),
    "seeded_test_wells": int(test_calls_v195["well"].nunique()),
    "submission_sha256": _sha256_v195(OUTPUT_V195),
    "source_v77_sha256": _sha256_v195(SOURCE_V195),
    "interactive_rows_note": (
        "Three visible example wells are expected while authoring; Kaggle's "
        "hidden rerun replaces them with the competition test wells."
    ),
    "auto_submit": str(SOURCE_V195),
    "named_archive": str(OUTPUT_V195),
}
SUMMARY_V195.write_text(json.dumps(summary_v195, indent=2), encoding="utf-8")
print("V195 FINAL SUMMARY")
print(json.dumps(summary_v195, indent=2), flush=True)
print("KAGGLE AUTO-SUBMITS:", SOURCE_V195, flush=True)
