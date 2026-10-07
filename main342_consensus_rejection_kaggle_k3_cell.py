"""Kaggle tail cell for the frozen M342-K2 consensus gate.

Run immediately after main342_fullbank_kaggle_stack_g5_cell.py. This cell
does not rerun a model. It reads G5's immutable endpoint snapshot, gates only
the G5 move away from V168, archives the ungated parent, and writes the final
submission.csv.
"""

from pathlib import Path
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd


STARTED_K3 = time.time()
WORK_K3 = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
ENDPOINTS_K3 = WORK_K3 / "m342_g3_test_endpoints.csv"
G3_NAMED_K3 = WORK_K3 / "submission_m342_g3_fixed_stack.csv"
G3_ARCHIVE_K3 = WORK_K3 / "submission_m342_g3_before_consensus_k3.csv"
NAMED_K3 = WORK_K3 / "submission_m342_k3_consensus_gate.csv"
OUTPUT_K3 = WORK_K3 / "submission.csv"
AUDIT_K3 = WORK_K3 / "m342_k3_consensus_gate_audit.json"

NORMALIZER_K3 = 0.6446710948584246
MAX_FACTOR_K3 = 2.0
ACTIVE_COLUMNS_K3 = ("coarse_crossfit", "psg_tvt", "fle_neighbor")
EXPECTED_G3_WEIGHTS_K3 = {
    "base_v168": 0.9399999999999695,
    "coarse_crossfit": 0.022155949463498307,
    "psg_tvt": 0.027592981219235015,
    "fle_neighbor": 0.010251069317290395,
}


def _sha256_k3(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


for required_k3 in (ENDPOINTS_K3, G3_NAMED_K3):
    if not required_k3.is_file():
        raise FileNotFoundError(
            f"Missing {required_k3}. Run main342_fullbank_kaggle_stack_g5_cell.py first."
        )

endpoints_k3 = pd.read_csv(ENDPOINTS_K3)
required_columns_k3 = {
    "id", "base_v168", "final", *ACTIVE_COLUMNS_K3,
}
missing_columns_k3 = sorted(required_columns_k3.difference(endpoints_k3.columns))
if missing_columns_k3:
    raise RuntimeError(f"G5 endpoint snapshot misses: {missing_columns_k3}")
if endpoints_k3["id"].astype(str).duplicated().any():
    raise RuntimeError("G5 endpoint snapshot contains duplicate ids")

ids_k3 = endpoints_k3["id"].astype(str)
wells_k3 = ids_k3.str.rsplit("_", n=1).str[0]
base_k3 = endpoints_k3["base_v168"].to_numpy(np.float64)
g3_k3 = endpoints_k3["final"].to_numpy(np.float64)
active_k3 = endpoints_k3.loc[:, ACTIVE_COLUMNS_K3].to_numpy(np.float64).T
if not np.isfinite(np.concatenate([base_k3, g3_k3, active_k3.ravel()])).all():
    raise RuntimeError("G5 endpoint snapshot contains non-finite values")

# Verify that the stored G5 final is exactly the expected four-endpoint stack.
replayed_g3_k3 = EXPECTED_G3_WEIGHTS_K3["base_v168"] * base_k3
for index_k3, name_k3 in enumerate(ACTIVE_COLUMNS_K3):
    replayed_g3_k3 += EXPECTED_G3_WEIGHTS_K3[name_k3] * active_k3[index_k3]
g3_replay_gap_k3 = float(np.max(np.abs(replayed_g3_k3 - g3_k3)))
if g3_replay_gap_k3 > 1e-8:
    raise RuntimeError(f"G5 endpoint lineage mismatch: max gap={g3_replay_gap_k3}")

delta_k3 = active_k3 - base_k3[None, :]
normalised_k3 = np.empty_like(delta_k3)
rms_rows_k3 = []
well_groups_k3 = pd.Series(np.arange(len(endpoints_k3)), index=endpoints_k3.index).groupby(
    wells_k3, sort=False
)
for well_k3, positions_series_k3 in well_groups_k3:
    positions_k3 = positions_series_k3.to_numpy(np.int64)
    local_k3 = delta_k3[:, positions_k3]
    rms_k3 = np.sqrt(np.mean(np.square(local_k3), axis=1))
    rms_k3 = np.maximum(rms_k3, 1e-6)
    normalised_k3[:, positions_k3] = local_k3 / rms_k3[:, None]
    rms_rows_k3.append({
        "well": str(well_k3),
        **{f"rms__{name}": float(value) for name, value in zip(ACTIVE_COLUMNS_K3, rms_k3)},
    })

agreement_k3 = np.abs(np.sum(normalised_k3, axis=0)) / np.maximum(
    np.sum(np.abs(normalised_k3), axis=0), 1e-12
)
agreement_k3 = np.clip(agreement_k3, 0.0, 1.0)
raw_factor_k3 = 0.25 + 0.75 * agreement_k3
factor_k3 = np.clip(raw_factor_k3 / NORMALIZER_K3, 0.0, MAX_FACTOR_K3)
final_k3 = base_k3 + factor_k3 * (g3_k3 - base_k3)
if not np.isfinite(final_k3).all():
    raise RuntimeError("K3 consensus output contains non-finite values")

g3_submission_k3 = pd.read_csv(G3_NAMED_K3)
if "id" not in g3_submission_k3 or g3_submission_k3["id"].astype(str).duplicated().any():
    raise RuntimeError("Invalid named G3 submission")
value_column_k3 = next(
    (name for name in ("tvt", "TVT") if name in g3_submission_k3), None
)
if value_column_k3 is None:
    candidates_k3 = [name for name in g3_submission_k3 if name != "id"]
    if len(candidates_k3) != 1:
        raise RuntimeError("Cannot identify G3 submission value column")
    value_column_k3 = candidates_k3[0]
if not g3_submission_k3["id"].astype(str).reset_index(drop=True).equals(
    ids_k3.reset_index(drop=True)
):
    raise RuntimeError("G3 submission and endpoint snapshot id order differ")
named_g3_values_k3 = g3_submission_k3[value_column_k3].to_numpy(np.float64)
named_g3_gap_k3 = float(np.max(np.abs(named_g3_values_k3 - g3_k3)))
if named_g3_gap_k3 > 1e-8:
    raise RuntimeError(f"Named G3 values differ from endpoint snapshot: {named_g3_gap_k3}")

shutil.copy2(G3_NAMED_K3, G3_ARCHIVE_K3)
output_k3 = g3_submission_k3.copy()
output_k3[value_column_k3] = final_k3
output_k3.to_csv(NAMED_K3, index=False)
shutil.copy2(NAMED_K3, OUTPUT_K3)

move_g3_k3 = g3_k3 - base_k3
move_k3 = final_k3 - base_k3
audit_payload_k3 = {
    "experiment": "main342_consensus_rejection_kaggle_k3",
    "parent": "M342-G5 exact four-endpoint stack",
    "profile": "magnitude_soft_mean1",
    "normalizer": NORMALIZER_K3,
    "maximum_factor": MAX_FACTOR_K3,
    "active_endpoints": ACTIVE_COLUMNS_K3,
    "rows": int(len(output_k3)),
    "wells": int(wells_k3.nunique()),
    "g3_replay_max_abs_gap_ft": g3_replay_gap_k3,
    "named_g3_max_abs_gap_ft": named_g3_gap_k3,
    "factor": {
        "mean": float(np.mean(factor_k3)),
        "std": float(np.std(factor_k3)),
        "p05": float(np.quantile(factor_k3, 0.05)),
        "median": float(np.median(factor_k3)),
        "p95": float(np.quantile(factor_k3, 0.95)),
        "minimum": float(np.min(factor_k3)),
        "maximum": float(np.max(factor_k3)),
    },
    "move_vs_v168": {
        "g3_rms_ft": float(np.sqrt(np.mean(np.square(move_g3_k3)))),
        "k3_rms_ft": float(np.sqrt(np.mean(np.square(move_k3)))),
    },
    "change_vs_g3": {
        "mean_abs_ft": float(np.mean(np.abs(final_k3 - g3_k3))),
        "rms_ft": float(np.sqrt(np.mean(np.square(final_k3 - g3_k3)))),
        "p95_abs_ft": float(np.quantile(np.abs(final_k3 - g3_k3), 0.95)),
        "maximum_abs_ft": float(np.max(np.abs(final_k3 - g3_k3))),
    },
    "files": {
        "endpoint_snapshot": str(ENDPOINTS_K3),
        "ungated_parent": str(G3_ARCHIVE_K3),
        "named_submission": str(NAMED_K3),
        "submission": str(OUTPUT_K3),
    },
    "hashes": {
        "ungated_parent": _sha256_k3(G3_ARCHIVE_K3),
        "named_submission": _sha256_k3(NAMED_K3),
        "submission": _sha256_k3(OUTPUT_K3),
    },
    "checks": {
        "g3_lineage_exact": g3_replay_gap_k3 <= 1e-8,
        "named_g3_matches_snapshot": named_g3_gap_k3 <= 1e-8,
        "factor_finite": bool(np.isfinite(factor_k3).all()),
        "output_finite": bool(np.isfinite(final_k3).all()),
        "named_equals_submission": _sha256_k3(NAMED_K3) == _sha256_k3(OUTPUT_K3),
    },
    "per_well_rms": rms_rows_k3,
    "elapsed_minutes": (time.time() - STARTED_K3) / 60.0,
    "instruction": "Submit /kaggle/working/submission.csv",
}
if not all(audit_payload_k3["checks"].values()):
    raise RuntimeError(f"K3 audit failed: {audit_payload_k3['checks']}")
AUDIT_K3.write_text(json.dumps(audit_payload_k3, indent=2), encoding="utf-8")
print("M342-K3 CONSENSUS KAGGLE TAIL COMPLETE")
print(json.dumps(audit_payload_k3, indent=2), flush=True)
