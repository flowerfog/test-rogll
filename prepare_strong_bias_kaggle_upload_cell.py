"""Validate and export the exact standalone strong_bias test submission.

This cell deliberately does not recreate a bias model. It verifies that the
local strong-bias OOF artifact exactly matches the candidate-pool column, then
checks that the test candidate is a capped correction of the exact strong
test curve before creating the small Kaggle upload directory.
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_DIR = ROOT / "oof_exports"
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
OUTPUT_DIR = OOF_DIR / "strong_bias_kaggle_upload"
OUTPUT_CSV = OUTPUT_DIR / "strong_bias_submission.csv"
OUTPUT_JSON = OUTPUT_DIR / "strong_bias_lineage.json"

# Set either override only if the automatic path is not the correct artifact.
OOF_OVERRIDE = None
CANDIDATE_OVERRIDE = None
BASE_OVERRIDE = None

EXPECTED_MOVE_CAP = 6.0
MAX_OOF_DIFF = 1e-5


def _existing_override(value):
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _find_named(names, roots):
    direct = []
    for root in roots:
        root = Path(root)
        for name in names:
            path = root / name
            if path.exists():
                direct.append(path)
    if direct:
        return direct[0]
    found = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for name in names:
            found.extend(root.rglob(name))
    if not found:
        return None
    found = sorted(set(path.resolve() for path in found), key=lambda path: path.stat().st_mtime, reverse=True)
    return found[0]


def _prediction_column(columns, allow_tvt=True):
    candidates = [
        "strong_bias",
        "tvt_base",
        "prediction",
        "tvt_pred",
        "pred",
    ]
    if allow_tvt:
        candidates.append("tvt")
    for column in candidates:
        if column in columns:
            return column
    raise RuntimeError(f"Cannot identify prediction column from {list(columns)}")


def _read_submission(path, value_name):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    prediction_column = _prediction_column(header, allow_tvt=True)
    frame = pd.read_csv(
        path,
        usecols=["id", prediction_column],
        dtype={"id": str, prediction_column: np.float64},
    ).rename(columns={prediction_column: value_name})
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise RuntimeError(f"Duplicate ids in {path}")
    if frame[value_name].isna().any() or not np.isfinite(frame[value_name]).all():
        raise RuntimeError(f"Nonfinite predictions in {path}")
    return frame


for path in (POOL_PATH, DATA_ROOT / "sample_submission.csv"):
    if not path.exists():
        raise FileNotFoundError(path)

oof_path = _existing_override(OOF_OVERRIDE) or _find_named(
    [
        "strong_base_projected_bias_corrected_oof.csv",
        "strong_bias_oof.csv",
    ],
    [OOF_DIR, ROOT],
)
candidate_path = _existing_override(CANDIDATE_OVERRIDE) or _find_named(
    [
        "submission_huber_bias_corrected.csv",
        "submission_strong_bias.csv",
        "strong_bias_submission.csv",
        "submission_bias_corrected.csv",
    ],
    [OOF_DIR, ROOT],
)
base_path = _existing_override(BASE_OVERRIDE) or _find_named(
    [
        "submission_before_gold.csv",
        "submission_strong_exact.csv",
        "submission_strong.csv",
    ],
    [OOF_DIR, ROOT],
)

if candidate_path is None:
    raise FileNotFoundError(
        "Cannot find a strong-bias test submission. Set CANDIDATE_OVERRIDE."
    )
if base_path is None:
    raise FileNotFoundError(
        "Cannot find the exact strong test submission used before bias correction. "
        "Set BASE_OVERRIDE; do not use a CP12/Gold file."
    )

print(
    {
        "oof_path": str(oof_path) if oof_path is not None else None,
        "candidate_path": str(candidate_path),
        "base_path": str(base_path),
    }
)


# The pool is the authoritative OOF used by all prior audits. If the separate
# source artifact survived cleanup, additionally require an exact row match.
pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "strong", "strong_bias"],
    dtype={"id": str, "strong": np.float32, "strong_bias": np.float32},
)
pool["id"] = pool["id"].astype(str)
pool_move = pool["strong_bias"].to_numpy(float) - pool["strong"].to_numpy(float)
pool_move_stats = {
    "rows": len(pool),
    "mean_abs_move": float(np.mean(np.abs(pool_move))),
    "p95_abs_move": float(np.quantile(np.abs(pool_move), 0.95)),
    "max_abs_move": float(np.max(np.abs(pool_move))),
    "moved_rows": int(np.sum(np.abs(pool_move) > 1e-8)),
}
print("POOL OOF MOVE:", pool_move_stats)
if pool_move_stats["max_abs_move"] > EXPECTED_MOVE_CAP + 1e-3:
    raise RuntimeError("pool.strong_bias itself violates the expected move cap")

oof_prediction_column = None
if oof_path is not None:
    oof_header = pd.read_csv(oof_path, nrows=0).columns.tolist()
    oof_prediction_column = _prediction_column(oof_header, allow_tvt=False)
    source_oof = pd.read_csv(
        oof_path,
        usecols=["id", oof_prediction_column],
        dtype={"id": str, oof_prediction_column: np.float32},
    ).rename(columns={oof_prediction_column: "source_strong_bias"})
    source_oof["id"] = source_oof["id"].astype(str)
    oof_check = pool[["id", "strong_bias"]].merge(
        source_oof, on="id", how="inner", validate="one_to_one"
    )
    if len(oof_check) != len(pool):
        raise RuntimeError(
            f"OOF lineage row mismatch: {len(pool)} -> {len(oof_check)}"
        )
    oof_difference = (
        oof_check["strong_bias"].to_numpy(float)
        - oof_check["source_strong_bias"].to_numpy(float)
    )
    oof_stats = {
        "mode": "separate_artifact_exact_match",
        "rows": len(oof_check),
        "mean_abs_diff": float(np.mean(np.abs(oof_difference))),
        "p95_abs_diff": float(np.quantile(np.abs(oof_difference), 0.95)),
        "max_abs_diff": float(np.max(np.abs(oof_difference))),
    }
    if oof_stats["max_abs_diff"] > MAX_OOF_DIFF:
        raise RuntimeError(
            "OOF artifact does not match pool.strong_bias; do not submit this lineage"
        )
else:
    oof_stats = {
        "mode": "pool_authoritative_separate_artifact_missing_after_cleanup",
        "rows": len(pool),
    }
print("OOF LINEAGE:", oof_stats)


# Test lineage: candidate and exact strong base must cover the official IDs.
sample = pd.read_csv(DATA_ROOT / "sample_submission.csv", usecols=["id"])
sample["id"] = sample["id"].astype(str)
candidate = _read_submission(candidate_path, "strong_bias")
base = _read_submission(base_path, "strong")
test = sample.merge(candidate, on="id", how="left", validate="one_to_one")
test = test.merge(base, on="id", how="left", validate="one_to_one")
if test[["strong_bias", "strong"]].isna().any().any():
    raise RuntimeError("Candidate/base do not cover every sample submission id")

move = test["strong_bias"].to_numpy(float) - test["strong"].to_numpy(float)
move_stats = {
    "rows": len(test),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
    "moved_rows": int(np.sum(np.abs(move) > 1e-8)),
}
print("TEST MOVE:", move_stats)
if move_stats["moved_rows"] == 0:
    raise RuntimeError("Candidate is identical to strong base")
if move_stats["max_abs_move"] > EXPECTED_MOVE_CAP + 1e-3:
    raise RuntimeError(
        f"Test move exceeds expected {EXPECTED_MOVE_CAP} ft cap. "
        "Likely wrong base or wrong candidate; set overrides explicitly."
    )
for statistic in ("mean_abs_move", "p95_abs_move"):
    reference = max(pool_move_stats[statistic], 1e-9)
    ratio = move_stats[statistic] / reference
    if not 0.25 <= ratio <= 4.0:
        raise RuntimeError(
            f"Test/pool {statistic} ratio is {ratio:.3f}; candidate lineage is implausible"
        )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output = test[["id", "strong_bias"]].rename(columns={"strong_bias": "tvt"})
output.to_csv(OUTPUT_CSV, index=False)

lineage = {
    "version": "standalone_strong_bias_v1",
    "oof_source": str(oof_path) if oof_path is not None else None,
    "test_candidate_source": str(candidate_path),
    "test_strong_base_source": str(base_path),
    "oof_prediction_column": oof_prediction_column,
    "oof_check": oof_stats,
    "pool_oof_move": pool_move_stats,
    "test_move": move_stats,
    "expected_move_cap": EXPECTED_MOVE_CAP,
    "output": str(OUTPUT_CSV),
}
OUTPUT_JSON.write_text(json.dumps(lineage, indent=2))

roundtrip = pd.read_csv(OUTPUT_CSV)
if len(roundtrip) != len(sample) or not np.isfinite(roundtrip["tvt"]).all():
    raise RuntimeError("Export roundtrip failed")

print("\nUPLOAD THESE TWO FILES AS ONE PRIVATE KAGGLE DATASET:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)
print("Do not upload the OOF or candidate pool.")
