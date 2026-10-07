"""V182 T0 final audit: run after the final submission.csv is written."""

from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


WORK_V182 = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
SUBMISSION_V182 = WORK_V182 / "submission.csv"
REFERENCE_V182 = WORK_V182 / "submission_v182_repeat_reference.csv"
OVERRIDE_REPORT_V182 = WORK_V182 / "gold_contact_override_report.csv"
TOLERANCE_V182 = 0.005


def _clean_v182(value):
    if isinstance(value, dict):
        return {str(key): _clean_v182(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v182(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _sha256_v182(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_ids_v182(frame):
    split = frame["id"].astype(str).str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2:
        raise RuntimeError("submission IDs must end in _rowindex")
    out = frame.copy()
    out["well"] = split[0].astype(str)
    out["row_idx"] = pd.to_numeric(split[1], errors="raise").astype(np.int64)
    return out


def _find_data_root_v182():
    candidates = []
    explicit = globals().get("DATA_DIR") or globals().get("DATA_ROOT")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend((Path("/kaggle/input/rogii-wellbore-geology-prediction"), Path.cwd()))
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(path.parent for path in input_root.glob("*/*__horizontal_well.csv"))
        candidates.extend(path.parent.parent for path in input_root.glob("*/train/*__horizontal_well.csv"))
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate train/test data root")


def _well_ids_v182(directory):
    suffix = "__horizontal_well"
    return {
        path.stem[: -len(suffix)]
        for path in directory.glob("*__horizontal_well.csv")
        if path.stem.endswith(suffix)
    }


if not SUBMISSION_V182.is_file():
    raise FileNotFoundError(SUBMISSION_V182)

submission_v182 = pd.read_csv(SUBMISSION_V182)
if not {"id", "tvt"}.issubset(submission_v182.columns):
    raise RuntimeError(f"Invalid submission columns: {submission_v182.columns.tolist()}")
submission_v182 = _split_ids_v182(submission_v182[["id", "tvt"]])
if submission_v182["id"].duplicated().any() or not np.isfinite(submission_v182["tvt"]).all():
    raise RuntimeError("Submission has duplicate IDs or non-finite predictions")

data_root_v182 = _find_data_root_v182()
train_wells_v182 = _well_ids_v182(data_root_v182 / "train")
test_wells_v182 = _well_ids_v182(data_root_v182 / "test")
submission_wells_v182 = set(submission_v182["well"])
overlap_wells_v182 = sorted(train_wells_v182 & test_wells_v182 & submission_wells_v182)
overlap_mask_v182 = submission_v182["well"].isin(overlap_wells_v182)

override_rows_v182 = 0
override_wells_v182 = 0
override_status_v182 = {}
if OVERRIDE_REPORT_V182.is_file():
    report_v182 = pd.read_csv(OVERRIDE_REPORT_V182)
    report_v182["well"] = report_v182["well"].astype(str)
    override_status_v182 = report_v182["status"].value_counts(dropna=False).to_dict()
    applied_v182 = report_v182[report_v182["status"].eq("override")]
    override_wells_v182 = int(applied_v182["well"].nunique())
    if "rows_overridden" in applied_v182:
        override_rows_v182 = int(applied_v182["rows_overridden"].fillna(0).sum())

repeat_v182 = {"reference_available": REFERENCE_V182.is_file()}
if REFERENCE_V182.is_file():
    reference_v182 = pd.read_csv(REFERENCE_V182)[["id", "tvt"]]
    comparison_v182 = submission_v182[["id", "tvt"]].merge(
        reference_v182, on="id", how="inner", suffixes=("_current", "_reference"), validate="one_to_one"
    )
    delta_v182 = comparison_v182["tvt_current"].to_numpy(float) - comparison_v182["tvt_reference"].to_numpy(float)
    repeat_v182.update({
        "matched_rows": len(comparison_v182),
        "max_abs_difference": float(np.max(np.abs(delta_v182))) if len(delta_v182) else None,
        "rms_difference": float(np.sqrt(np.mean(delta_v182 * delta_v182))) if len(delta_v182) else None,
        "within_0p005": bool(len(comparison_v182) == len(submission_v182) and np.max(np.abs(delta_v182)) <= TOLERANCE_V182),
    })
else:
    submission_v182[["id", "tvt"]].to_csv(REFERENCE_V182, index=False)
    repeat_v182["next"] = "Rerun the identical pipeline, then rerun this cell"

checkpoint_names_v182 = (
    "submission_controlpoint_before_gold.csv",
    "submission_gold_prefix_conservative.csv",
    "submission_v69_all773_w030.csv",
    "submission.csv",
)
checkpoints_v182 = []
for name_v182 in checkpoint_names_v182:
    path_v182 = WORK_V182 / name_v182
    if path_v182.is_file():
        checkpoints_v182.append({"file": name_v182, "rows": sum(1 for _ in path_v182.open("rb")) - 1, "sha256": _sha256_v182(path_v182)})

verdict_v182 = {
    "experiment": "kaggle_t0_determinism_override_audit_v182",
    "data_root": data_root_v182,
    "submission": {"rows": len(submission_v182), "wells": len(submission_wells_v182), "sha256": _sha256_v182(SUBMISSION_V182)},
    "overlap": {
        "train_wells": len(train_wells_v182),
        "test_wells": len(test_wells_v182),
        "same_id_wells": len(overlap_wells_v182),
        "same_id_well_fraction": len(overlap_wells_v182) / max(1, len(submission_wells_v182)),
        "same_id_rows": int(overlap_mask_v182.sum()),
        "same_id_row_fraction": float(overlap_mask_v182.mean()),
        "well_ids": overlap_wells_v182,
    },
    "override": {
        "report_available": OVERRIDE_REPORT_V182.is_file(),
        "status_counts": override_status_v182,
        "overridden_wells": override_wells_v182,
        "overridden_rows": override_rows_v182,
        "overridden_row_fraction": override_rows_v182 / max(1, len(submission_v182)),
    },
    "repeatability": repeat_v182,
    "checkpoints": checkpoints_v182,
    "hidden_overlap_inference_allowed": False,
    "next": "Use private rerun logs for hidden overlap; visible logs are implementation controls only",
}

output_v182 = WORK_V182 / "v182_t0_audit.json"
output_v182.write_text(json.dumps(_clean_v182(verdict_v182), indent=2), encoding="utf-8")
print("V182 T0 FINAL AUDIT")
print(json.dumps(_clean_v182(verdict_v182), indent=2), flush=True)
print("saved:", output_v182, flush=True)

