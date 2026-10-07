"""Discover renamed strong-bias test submissions or surviving model bundles."""

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_DIR = ROOT / "oof_exports"
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
OUTPUT = OOF_DIR / "strong_bias_artifact_discovery.csv"

# Set this only if automatic exact-strong discovery fails.
BASE_OVERRIDE = None


def _prediction_column(columns):
    for column in ("tvt", "strong", "tvt_base", "prediction", "tvt_pred", "pred"):
        if column in columns:
            return column
    return None


def _read_aligned(path, sample, value_name):
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if "id" not in header:
            return None
        prediction_column = _prediction_column(header)
        if prediction_column is None:
            return None
        frame = pd.read_csv(
            path,
            usecols=["id", prediction_column],
            dtype={"id": str, prediction_column: np.float64},
        ).rename(columns={prediction_column: value_name})
        frame["id"] = frame["id"].astype(str)
        if frame["id"].duplicated().any():
            return None
        aligned = sample.merge(frame, on="id", how="left", validate="one_to_one")
        if aligned[value_name].isna().any() or len(frame) != len(sample):
            return None
        if set(frame["id"]) != set(sample["id"]):
            return None
        if not np.isfinite(aligned[value_name]).all():
            return None
        return aligned[value_name].to_numpy(float)
    except Exception:
        return None


for path in (DATA_ROOT / "sample_submission.csv", POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

sample = pd.read_csv(DATA_ROOT / "sample_submission.csv", usecols=["id"])
sample["id"] = sample["id"].astype(str)
pool = pd.read_csv(
    POOL_PATH,
    usecols=["strong", "strong_bias"],
    dtype={"strong": np.float32, "strong_bias": np.float32},
)
pool_move = pool["strong_bias"].to_numpy(float) - pool["strong"].to_numpy(float)
reference = {
    "mean_abs": float(np.mean(np.abs(pool_move))),
    "p95_abs": float(np.quantile(np.abs(pool_move), 0.95)),
    "max_abs": float(np.max(np.abs(pool_move))),
    "moved_frac": float(np.mean(np.abs(pool_move) > 1e-8)),
}
print("POOL STRONG_BIAS MOVE:", reference)

base_names = [
    "submission_before_gold.csv",
    "submission_strong_exact.csv",
    "submission_strong.csv",
]
if BASE_OVERRIDE is not None:
    base_path = Path(BASE_OVERRIDE)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
else:
    base_path = None
    for name in base_names:
        direct = [OOF_DIR / name, ROOT / name]
        direct += list(ROOT.rglob(name))
        for path in direct:
            if path.exists() and _read_aligned(path, sample, "base") is not None:
                base_path = path
                break
        if base_path is not None:
            break

if base_path is None:
    raise FileNotFoundError(
        "Cannot locate exact strong base. Set BASE_OVERRIDE before discovery."
    )
base = _read_aligned(base_path, sample, "base")
print("EXACT STRONG BASE:", base_path)

# Scan plausible submission-sized CSV files. Avoid the 900 MB OOF pool.
csv_paths = []
for path in ROOT.rglob("*.csv"):
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        continue
    name = path.name.lower()
    if not any(token in name for token in ("submission", "submit", "huber", "bias", "strong")):
        continue
    if not 100_000 <= size <= 100 * 2**20:
        continue
    csv_paths.append(path)

rows = []
for path in sorted(set(csv_paths)):
    prediction = _read_aligned(path, sample, "candidate")
    if prediction is None:
        continue
    move = prediction - base
    mean_abs = float(np.mean(np.abs(move)))
    p95_abs = float(np.quantile(np.abs(move), 0.95))
    max_abs = float(np.max(np.abs(move)))
    moved_frac = float(np.mean(np.abs(move) > 1e-8))
    if mean_abs <= 1e-12:
        distribution_distance = np.inf
    else:
        distribution_distance = (
            abs(math.log(max(mean_abs, 1e-9) / max(reference["mean_abs"], 1e-9)))
            + abs(math.log(max(p95_abs, 1e-9) / max(reference["p95_abs"], 1e-9)))
            + abs(moved_frac - reference["moved_frac"])
        )
    rows.append(
        {
            "path": str(path),
            "size_mb": path.stat().st_size / 2**20,
            "mean_abs_move": mean_abs,
            "p95_abs_move": p95_abs,
            "max_abs_move": max_abs,
            "moved_frac": moved_frac,
            "cap6_compatible": bool(max_abs <= 6.001 and mean_abs > 1e-12),
            "distribution_distance": distribution_distance,
        }
    )

table = pd.DataFrame(rows)
if len(table):
    table = table.sort_values(
        ["cap6_compatible", "distribution_distance"],
        ascending=[False, True],
    ).reset_index(drop=True)
else:
    table = pd.DataFrame(
        columns=[
            "path",
            "size_mb",
            "mean_abs_move",
            "p95_abs_move",
            "max_abs_move",
            "moved_frac",
            "cap6_compatible",
            "distribution_distance",
        ]
    )
table.to_csv(OUTPUT, index=False)
print("\nRANKED TEST CURVES")
display(table.head(50))
print("saved:", OUTPUT)

model_rows = []
for suffix in ("*.joblib", "*.pkl", "*.json", "*.cbm"):
    for path in ROOT.rglob(suffix):
        name = path.name.lower()
        if not any(token in name for token in ("bias", "huber", "strong")):
            continue
        try:
            model_rows.append(
                {
                    "path": str(path),
                    "size_mb": path.stat().st_size / 2**20,
                    "modified": pd.Timestamp(path.stat().st_mtime, unit="s"),
                }
            )
        except FileNotFoundError:
            pass
models = pd.DataFrame(model_rows)
if len(models):
    models = models.sort_values("modified", ascending=False).reset_index(drop=True)
print("\nPOSSIBLE MODEL / LINEAGE ARTIFACTS")
display(models.head(100))

compatible = table[table["cap6_compatible"]] if len(table) else table
if len(compatible) == 0:
    print("\nNO CAP-6 TEST CURVE FOUND: strong_bias was OOF-only and must be rebuilt.")
else:
    print("\nDo not auto-submit yet. Use the top path as CANDIDATE_OVERRIDE only after reviewing this table.")
