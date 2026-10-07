"""Emit the validated standalone strong_bias submission on Kaggle."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working")


def _find_data_root():
    candidates = []
    if "CFG" in globals():
        for attribute in ("dataset_path", "DATA", "data_path"):
            if hasattr(CFG, attribute):
                candidates.append(Path(getattr(CFG, attribute)))
    candidates += [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    for path in candidates:
        if (path / "sample_submission.csv").exists():
            return path
    raise FileNotFoundError("Competition sample_submission.csv not found")


def _find_upload(name):
    matches = sorted(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}, found: {matches}")
    return matches[0]


DATA_ROOT = _find_data_root()
UPLOAD_CSV = _find_upload("strong_bias_submission.csv")
lineage_matches = sorted(Path("/kaggle/input").rglob("strong_bias_lineage.json"))

sample = pd.read_csv(DATA_ROOT / "sample_submission.csv", usecols=["id"])
sample["id"] = sample["id"].astype(str)
candidate = pd.read_csv(UPLOAD_CSV, usecols=["id", "tvt"])
candidate["id"] = candidate["id"].astype(str)

if candidate["id"].duplicated().any():
    raise RuntimeError("Uploaded strong_bias submission contains duplicate ids")
submission = sample.merge(candidate, on="id", how="left", validate="one_to_one")
if len(submission) != len(sample):
    raise RuntimeError("Submission row count changed during alignment")
if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
    raise RuntimeError("Missing/nonfinite strong_bias predictions")
if set(candidate["id"]) != set(sample["id"]):
    raise RuntimeError("Uploaded ids do not exactly equal competition test ids")

standalone_path = WORK / "submission_strong_bias_standalone.csv"
final_path = WORK / "submission.csv"
submission.to_csv(standalone_path, index=False)
submission.to_csv(final_path, index=False)

print(
    {
        "data_root": str(DATA_ROOT),
        "upload": str(UPLOAD_CSV),
        "rows": len(submission),
        "wells": int(submission["id"].str.rsplit("_", n=1).str[0].nunique()),
        "tvt_mean": float(submission["tvt"].mean()),
        "tvt_std": float(submission["tvt"].std()),
        "tvt_min": float(submission["tvt"].min()),
        "tvt_max": float(submission["tvt"].max()),
        "saved": str(final_path),
    }
)
if len(lineage_matches) == 1:
    lineage = json.loads(lineage_matches[0].read_text())
    print("LINEAGE:", lineage)
else:
    print("lineage json matches:", lineage_matches)

display(submission.head())
print("Submit /kaggle/working/submission.csv; run no later mutation cells.")
