"""Paste after the exact K2 submission endpoint is generated.

Required live arrays/dataframe:
  k2_base_prediction: pre-K2 endpoint in submission-row order
  k2_parent_prediction: exact current K2 endpoint in submission-row order
  submission: dataframe with id and tvt, or set SUBMISSION_PATH
"""

from pathlib import Path
import hashlib
import shutil

import numpy as np
import pandas as pd


SCALE = 1.25
SUBMISSION_PATH = Path(globals().get("OUTPUT_PATH", "/kaggle/working/submission.csv"))
PARENT_SNAPSHOT = Path("/kaggle/working/submission_k2_scale1_parent.csv")
OUTPUT_SCALE125 = Path("/kaggle/working/submission_k2_scale125.csv")

if "k2_base_prediction" not in globals() or "k2_parent_prediction" not in globals():
    raise RuntimeError(
        "Run the exact K2 endpoint cell first. Required live arrays: "
        "k2_base_prediction and k2_parent_prediction"
    )

base = np.asarray(k2_base_prediction, np.float64).reshape(-1)
parent = np.asarray(k2_parent_prediction, np.float64).reshape(-1)
if len(base) != len(parent) or not np.isfinite(base).all() or not np.isfinite(parent).all():
    raise RuntimeError("K2 arrays have different lengths or non-finite values")

if "submission" in globals() and isinstance(submission, pd.DataFrame):
    parent_frame = submission.copy()
else:
    if not SUBMISSION_PATH.is_file():
        raise FileNotFoundError(SUBMISSION_PATH)
    parent_frame = pd.read_csv(SUBMISSION_PATH)
if not {"id", "tvt"}.issubset(parent_frame.columns):
    raise RuntimeError(f"Expected id/tvt submission columns, found {list(parent_frame.columns)}")
if len(parent_frame) != len(parent) or parent_frame["id"].duplicated().any():
    raise RuntimeError("K2 endpoint length/id contract failed")

parent_frame = parent_frame[["id", "tvt"]].copy()
parent_frame["tvt"] = parent
parent_frame.to_csv(PARENT_SNAPSHOT, index=False)
candidate = base + SCALE * (parent - base)
if not np.isfinite(candidate).all():
    raise RuntimeError("Scale125 candidate contains non-finite values")
candidate_frame = parent_frame.copy()
candidate_frame["tvt"] = candidate
candidate_frame.to_csv(OUTPUT_SCALE125, index=False)
candidate_frame.to_csv(SUBMISSION_PATH, index=False)

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

audit = {
    "parent": str(PARENT_SNAPSHOT),
    "candidate": str(OUTPUT_SCALE125),
    "final": str(SUBMISSION_PATH),
    "rows": len(candidate_frame),
    "scale": SCALE,
    "parent_sha256": _sha256(PARENT_SNAPSHOT),
    "candidate_sha256": _sha256(OUTPUT_SCALE125),
    "mean_abs_move_ft": float(np.mean(np.abs(candidate - parent))),
    "rms_move_ft": float(np.sqrt(np.mean(np.square(candidate - parent)))),
    "max_abs_move_ft": float(np.max(np.abs(candidate - parent))),
}
print("M342-K2 SCALE125 DEPLOYMENT", audit, flush=True)
print("submission.csv:", SUBMISSION_PATH, flush=True)
