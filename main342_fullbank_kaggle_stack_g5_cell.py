"""Exact Kaggle tail cell for the frozen M342-G3 four-endpoint stack.

Required run order in one kernel:
  1. complete mainline, including the structure-progress Neighbor cell
  2. V69 -> V77 -> V116 -> V144
  3. historical_analog_kaggle_v170_cell.py (patched for the V144 parent)
  4. kaggle_psg_v66_delta_tail_v5_cell.py (raw PSG variables are retained)
  5. this cell

The current root submission may have been overwritten by the PSG diagnostic;
the exact V168 parent is therefore read from V170's named immutable copy.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd


STARTED_G5 = time.time()
WORK_G5 = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
PARENT_G5 = WORK_G5 / "submission_v170_metadata_analog_w0p05.csv"
PARENT_SUMMARY_G5 = WORK_G5 / "v170_metadata_analog_summary.json"
OUTPUT_G5 = WORK_G5 / "submission.csv"
NAMED_G5 = WORK_G5 / "submission_m342_g3_fixed_stack.csv"
ENDPOINTS_G5 = WORK_G5 / "m342_g3_test_endpoints.csv"
AUDIT_G5 = WORK_G5 / "m342_g3_kaggle_stack_audit.json"
NEIGHBOR_AUDIT_G5 = WORK_G5 / "frontier_6p39_tail_neighbor_test_audit.csv"
PSG_RAW_AUDIT_G5 = WORK_G5 / "psg_v66_raw_for_g5_audit.json"

WEIGHTS_G5 = {
    "base_v168": 0.9399999999999695,
    "coarse_crossfit": 0.022155949463498307,
    "psg_tvt": 0.027592981219235015,
    "fle_neighbor": 0.010251069317290395,
}
NEIGHBOR_WEIGHT_G5 = 0.35
MAX_MEDIAN_DISTANCE_G5 = 600.0
MAX_P90_DISTANCE_G5 = 1200.0
MAX_ANCHOR_DISTANCE_G5 = 1200.0
MIN_ABS_DIRECTION_DOT_G5 = 0.70


def _sha256_g5(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_g5(frame, expected_ids, label):
    frame = frame.copy()
    if "id" not in frame or frame["id"].duplicated().any():
        raise RuntimeError(f"{label}: invalid or duplicate ids")
    value_column = next((name for name in ("tvt", "TVT") if name in frame), None)
    if value_column is None:
        candidates = [name for name in frame if name != "id"]
        if len(candidates) != 1:
            raise RuntimeError(f"{label}: cannot identify prediction column")
        value_column = candidates[0]
    local = frame[["id", value_column]].rename(columns={value_column: label})
    local["id"] = local["id"].astype(str)
    local = pd.DataFrame({"id": expected_ids}).merge(
        local, on="id", how="left", validate="one_to_one"
    )
    values = local[label].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}: incomplete/non-finite alignment")
    return values


required_live_g5 = (
    "sample", "v77_coarse", "test", "psg_tvt",
    "base_submission", "candidate_values", "MAPPING_MODE",
)
missing_live_g5 = [name for name in required_live_g5 if name not in globals()]
if missing_live_g5:
    raise RuntimeError(
        "Run Neighbor, V69, V77, V116, V144, patched V170 and PSG-V5 first. "
        "Missing live values: " + repr(missing_live_g5)
    )
if not PARENT_G5.is_file() or not PARENT_SUMMARY_G5.is_file():
    raise FileNotFoundError("The exact V170-on-V144 named parent is missing")
if MAPPING_MODE != "progress_v0":
    raise RuntimeError(f"G3 requires the structure-progress Neighbor mapper, got {MAPPING_MODE}")
if not NEIGHBOR_AUDIT_G5.is_file():
    raise FileNotFoundError(NEIGHBOR_AUDIT_G5)
if not PSG_RAW_AUDIT_G5.is_file():
    raise FileNotFoundError("Run PSG-V5 raw-only inference before G5")
psg_raw_audit_g5 = json.loads(PSG_RAW_AUDIT_G5.read_text(encoding="utf-8"))
if (
    psg_raw_audit_g5.get("mode") != "raw_only_for_main342_g5"
    or int(psg_raw_audit_g5.get("rows", -1)) != len(sample)
    or float(psg_raw_audit_g5.get("finite_fraction", 0.0)) != 1.0
    or bool(psg_raw_audit_g5.get("submission_modified", True))
):
    raise RuntimeError(f"Invalid PSG raw endpoint audit: {psg_raw_audit_g5}")

sample_g5 = sample.copy()
sample_g5["id"] = sample_g5["id"].astype(str)
if sample_g5["id"].duplicated().any():
    raise RuntimeError("sample_submission contains duplicate ids")
ids_g5 = sample_g5["id"]
value_column_g5 = next((name for name in ("tvt", "TVT") if name in sample_g5), None)
if value_column_g5 is None:
    raise RuntimeError(f"Cannot identify sample target: {sample_g5.columns.tolist()}")

# Exact V168 parent: V144 nested sequence plus the frozen 0.05 analog move.
parent_frame_g5 = pd.read_csv(PARENT_G5)
base_v168_g5 = _normalise_g5(parent_frame_g5, ids_g5, "base_v168")
parent_summary_g5 = json.loads(PARENT_SUMMARY_G5.read_text(encoding="utf-8"))
if parent_summary_g5.get("parent") != "V144 nested_sequence_coefficient":
    raise RuntimeError(f"V170 parent lineage is wrong: {parent_summary_g5.get('parent')}")
if _sha256_g5(PARENT_G5) != parent_summary_g5.get("submission_sha256"):
    raise RuntimeError("V170 named parent hash differs from its audit summary")

# V144 coarse_crossfit deploys as the raw all-773 coarse path retained by V77.
coarse_g5 = np.asarray(v77_coarse, np.float64)
if len(coarse_g5) != len(sample_g5) or not np.isfinite(coarse_g5).all():
    raise RuntimeError("raw v77_coarse is not fully aligned to sample")

# Raw PSG is aligned through the PSG feature frame IDs. Do not use the V5
# feature-model correction or its rewritten root submission.
test_ids_g5 = test["id"].astype(str)
if test_ids_g5.duplicated().any() or len(psg_tvt) != len(test_ids_g5):
    raise RuntimeError("raw PSG/test feature alignment is malformed")
psg_frame_g5 = pd.DataFrame({"id": test_ids_g5, "psg_tvt": np.asarray(psg_tvt, np.float64)})
psg_g5 = pd.DataFrame({"id": ids_g5}).merge(
    psg_frame_g5, on="id", how="left", validate="one_to_one"
)["psg_tvt"].to_numpy(np.float64)
if not np.isfinite(psg_g5).all():
    raise RuntimeError("raw PSG does not cover every sample row")

# Reconstruct the historical V2 fle_neighbor endpoint. candidate_values is the
# absolute structure-progress transfer generated before any mainline dose.
neighbor_base_g5 = base_submission.copy()
neighbor_base_g5["id"] = neighbor_base_g5["id"].astype(str)
if len(neighbor_base_g5) != len(candidate_values):
    raise RuntimeError("Neighbor raw candidate/base row mismatch")
raw_neighbor_frame_g5 = pd.DataFrame({
    "id": neighbor_base_g5["id"],
    "raw_neighbor": np.asarray(candidate_values, np.float64),
})
raw_neighbor_g5 = pd.DataFrame({"id": ids_g5}).merge(
    raw_neighbor_frame_g5, on="id", how="left", validate="one_to_one"
)["raw_neighbor"].to_numpy(np.float64)

fle_candidates_g5 = (
    WORK_G5 / "learned_trajectory_submission.csv",
    WORK_G5 / "fleongg_pretrained_submission.csv",
)
fle_path_g5 = next((path for path in fle_candidates_g5 if path.is_file()), None)
if fle_path_g5 is None:
    raise FileNotFoundError(
        "Missing raw FLE endpoint: learned_trajectory_submission.csv or "
        "fleongg_pretrained_submission.csv"
    )
fle_parent_g5 = _normalise_g5(pd.read_csv(fle_path_g5), ids_g5, "fle_parent")

audit_g5 = pd.read_csv(NEIGHBOR_AUDIT_G5)
if "well" not in audit_g5 or audit_g5["well"].astype(str).duplicated().any():
    raise RuntimeError("Neighbor audit must contain one row per well")
needed_audit_g5 = {
    "well", "median_distance", "p90_distance", "anchor_distance_full", "abs_direction_dot",
}
missing_audit_g5 = sorted(needed_audit_g5.difference(audit_g5.columns))
if missing_audit_g5:
    raise RuntimeError(f"Neighbor audit misses coherent600 fields: {missing_audit_g5}")
audit_g5["well"] = audit_g5["well"].astype(str)
gate_by_well_g5 = (
    audit_g5["median_distance"].le(MAX_MEDIAN_DISTANCE_G5)
    & audit_g5["p90_distance"].le(MAX_P90_DISTANCE_G5)
    & audit_g5["anchor_distance_full"].le(MAX_ANCHOR_DISTANCE_G5)
    & audit_g5["abs_direction_dot"].ge(MIN_ABS_DIRECTION_DOT_G5)
)
gate_map_g5 = dict(zip(audit_g5["well"], gate_by_well_g5))
sample_wells_g5 = ids_g5.str.rsplit("_", n=1).str[0]
eligible_g5 = sample_wells_g5.map(gate_map_g5).fillna(False).to_numpy(bool)
eligible_g5 &= np.isfinite(raw_neighbor_g5)
fle_neighbor_g5 = fle_parent_g5.copy()
fle_neighbor_g5[eligible_g5] += NEIGHBOR_WEIGHT_G5 * (
    raw_neighbor_g5[eligible_g5] - fle_parent_g5[eligible_g5]
)
if not np.isfinite(fle_neighbor_g5).all():
    raise RuntimeError("reconstructed fle_neighbor contains non-finite values")

if abs(sum(WEIGHTS_G5.values()) - 1.0) > 1e-10:
    raise RuntimeError(f"G3 weights no longer sum to one: {WEIGHTS_G5}")
final_g5 = (
    WEIGHTS_G5["base_v168"] * base_v168_g5
    + WEIGHTS_G5["coarse_crossfit"] * coarse_g5
    + WEIGHTS_G5["psg_tvt"] * psg_g5
    + WEIGHTS_G5["fle_neighbor"] * fle_neighbor_g5
)
if not np.isfinite(final_g5).all():
    raise RuntimeError("G3 final stack contains non-finite values")

endpoint_frame_g5 = pd.DataFrame({
    "id": ids_g5,
    "base_v168": base_v168_g5,
    "coarse_crossfit": coarse_g5,
    "psg_tvt": psg_g5,
    "fle_parent": fle_parent_g5,
    "raw_neighbor": raw_neighbor_g5,
    "fle_neighbor": fle_neighbor_g5,
    "neighbor_eligible": eligible_g5,
    "final": final_g5,
})
endpoint_frame_g5.to_csv(ENDPOINTS_G5, index=False)
output_g5 = sample_g5.copy()
output_g5[value_column_g5] = final_g5
output_g5.to_csv(NAMED_G5, index=False)
output_g5.to_csv(OUTPUT_G5, index=False)

move_g5 = final_g5 - base_v168_g5
audit_payload_g5 = {
    "experiment": "main342_fullbank_kaggle_stack_g5",
    "parent": "V168 exact = V144 nested_sequence_coefficient + frozen 0.05 analog",
    "weights": WEIGHTS_G5,
    "neighbor_recipe": {
        "method": "structure_progress", "gate": "coherent600",
        "weight": NEIGHBOR_WEIGHT_G5, "cap_ft": None,
    },
    "rows": int(len(output_g5)),
    "wells": int(sample_wells_g5.nunique()),
    "neighbor_eligible_wells": int(audit_g5.loc[gate_by_well_g5, "well"].nunique()),
    "neighbor_eligible_row_fraction": float(eligible_g5.mean()),
    "move_vs_v168": {
        "mean_abs_ft": float(np.mean(np.abs(move_g5))),
        "rms_ft": float(np.sqrt(np.mean(np.square(move_g5)))),
        "p95_abs_ft": float(np.quantile(np.abs(move_g5), 0.95)),
        "max_abs_ft": float(np.max(np.abs(move_g5))),
    },
    "files": {
        "v168_parent": str(PARENT_G5),
        "fle_parent": str(fle_path_g5),
        "endpoint_snapshot": str(ENDPOINTS_G5),
        "named_submission": str(NAMED_G5),
        "submission": str(OUTPUT_G5),
    },
    "hashes": {
        "v168_parent": _sha256_g5(PARENT_G5),
        "named_submission": _sha256_g5(NAMED_G5),
        "submission": _sha256_g5(OUTPUT_G5),
    },
    "checks": {
        "v168_parent_hash_verified": True,
        "four_absolute_endpoints_finite": True,
        "weights_sum_to_one": True,
        "named_equals_submission": _sha256_g5(NAMED_G5) == _sha256_g5(OUTPUT_G5),
    },
    "elapsed_minutes": (time.time() - STARTED_G5) / 60.0,
    "instruction": "Submit /kaggle/working/submission.csv",
}
AUDIT_G5.write_text(json.dumps(audit_payload_g5, indent=2), encoding="utf-8")
print("M342-G5 KAGGLE STACK COMPLETE")
print(json.dumps(audit_payload_g5, indent=2), flush=True)
