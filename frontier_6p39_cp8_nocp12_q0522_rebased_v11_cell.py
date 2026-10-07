"""Rebase the frozen Q0522 edit on the CP8-noCP12 pre-Q submission.

Run only after the original midpoint hedge cell. This preserves every causal
guard from the published Q0522 cell except its obsolete source-file hash.
The no-Q candidate is saved before the edit so both artifacts remain available.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


LABEL = "Q0522_CP8_REBASED"
EXTRA_SHIFT = 0.522
EXPECTED_WELL = "00e12e8b"
EXPECTED_ROWS = 4301
EXPECTED_BRANCH_SHIFT = 2.0

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
SUBMISSION = WORK / "submission.csv"
BRANCH_REPORT = WORK / "pf_seed_branch_hedge_report.csv"
CP_CHECKPOINT = WORK / "submission_frontier_6p39_cp8_nocp12_before_overlays.csv"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_sample():
    candidates = []
    cfg = globals().get("CFG")
    if cfg is not None:
        for attribute in ("DATA", "dataset_path"):
            if hasattr(cfg, attribute):
                candidates.append(Path(getattr(cfg, attribute)) / "sample_submission.csv")
    candidates.extend(
        [
            Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv"),
            Path("/kaggle/input/rogii-wellbore-geology-prediction/sample_submission.csv"),
            WORK / "sample_submission.csv",
        ]
    )
    return next((path for path in candidates if path.exists()), None)


for required in (SUBMISSION, BRANCH_REPORT, CP_CHECKPOINT):
    if not required.exists():
        raise FileNotFoundError(required)
sample_path = find_sample()
if sample_path is None:
    raise FileNotFoundError("sample_submission.csv")

sample = pd.read_csv(sample_path, dtype={"id": "string"})[["id"]]
base = pd.read_csv(SUBMISSION, dtype={"id": "string"})
if list(base.columns) != ["id", "tvt"]:
    raise RuntimeError(f"{LABEL}: unexpected columns {list(base.columns)}")
if len(base) != len(sample) or not base["id"].equals(sample["id"]):
    raise RuntimeError(f"{LABEL}: source submission does not match sample order")
base["tvt"] = pd.to_numeric(base["tvt"], errors="coerce")
base_values = base["tvt"].to_numpy(float)
if not np.isfinite(base_values).all():
    raise RuntimeError(f"{LABEL}: non-finite source predictions")

branch = pd.read_csv(BRANCH_REPORT)
required_columns = {"well", "reason", "shift", "moved_rows"}
if not required_columns.issubset(branch.columns):
    raise RuntimeError(
        f"{LABEL}: branch report missing {sorted(required_columns - set(branch.columns))}"
    )
applied = branch[branch["reason"].astype(str).str.lower().eq("applied")].copy()
if len(applied) != 1:
    raise RuntimeError(f"{LABEL}: expected exactly one applied branch, got {len(applied)}")
row = applied.iloc[0]
source_well = str(row["well"])
source_shift = float(row["shift"])
source_rows = int(row["moved_rows"])
if (
    source_well != EXPECTED_WELL
    or abs(source_shift - EXPECTED_BRANCH_SHIFT) > 1e-9
    or source_rows != EXPECTED_ROWS
):
    raise RuntimeError(
        f"{LABEL}: midpoint branch changed: well={source_well}, "
        f"shift={source_shift}, rows={source_rows}"
    )

wells = base["id"].astype(str).str.rsplit("_", n=1).str[0]
mask = wells.eq(EXPECTED_WELL).to_numpy()
if int(mask.sum()) != EXPECTED_ROWS:
    raise RuntimeError(f"{LABEL}: expected {EXPECTED_ROWS} target rows, got {int(mask.sum())}")

no_q_path = WORK / "submission_frontier_6p39_cp8_nocp12_no_q0522.csv"
base.to_csv(no_q_path, index=False)
final_values = base_values.copy()
final_values[mask] += EXTRA_SHIFT
delta = final_values - base_values
if not np.allclose(delta[mask], EXTRA_SHIFT, atol=1e-10, rtol=0.0):
    raise RuntimeError(f"{LABEL}: target shift audit failed")
if np.max(np.abs(delta[~mask])) != 0.0:
    raise RuntimeError(f"{LABEL}: changed non-target rows")

final = base.copy()
final["tvt"] = final_values
named_path = WORK / "submission_frontier_6p39_cp8_nocp12_q0522.csv"
final.to_csv(named_path, index=False)
final.to_csv(SUBMISSION, index=False)

report = {
    "experiment": LABEL,
    "rule_source": "published Q0522; source hash rebased after CP8",
    "source_sha256": sha256(no_q_path),
    "cp_checkpoint_sha256": sha256(CP_CHECKPOINT),
    "final_sha256": sha256(named_path),
    "target_well": EXPECTED_WELL,
    "target_rows": int(mask.sum()),
    "midpoint_shift_ft": source_shift,
    "additional_shift_ft": EXTRA_SHIFT,
    "total_branch_shift_ft": source_shift + EXTRA_SHIFT,
    "non_target_max_abs_move_ft": float(np.max(np.abs(delta[~mask]))),
    "no_q_candidate": str(no_q_path),
    "q0522_candidate": str(named_path),
}
(WORK / "q0522_cp8_rebased_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(LABEL, json.dumps(report, indent=2), flush=True)
