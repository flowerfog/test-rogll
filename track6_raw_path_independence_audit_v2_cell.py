"""Audit raw fold-clean path features before the Track6 GBDT collapses them.

This zero-training audit asks whether any raw PF, likelihood-PF, formation
surface, or dense path already supplies the low-error-correlation endpoint
needed by the published Track6 blend. Both raw and 61-row smoothed paths are
tested against the exact historical SP45 Fold1 endpoint.
"""

from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = CACHE_ROOT / "features"
META_PATH = CACHE_ROOT / "well_metadata.csv"
TRACK6_PATH = OOF_ROOT / "track6_independent_pf_gbdt_gateway_v1" / "fold1_track6_oof.pkl"
OUT_DIR = OOF_ROOT / "track6_raw_path_independence_audit_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FOLD = 1
MAX_STANDALONE_RMSE = 120.0
MIN_COVERAGE = 0.95
ORACLE_MEAN_GATE = 5.30
COSINE_GATE = 0.50


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _smooth_by_well(values, codes, width=61):
    values = np.asarray(values, float)
    output = values.copy()
    for code in range(int(codes.max()) + 1):
        indices = np.flatnonzero(codes == code)
        window = min(int(width), len(indices))
        if window % 2 == 0:
            window -= 1
        if window >= 5:
            output[indices] = savgol_filter(values[indices], window, 3)
    return output


def _group_sum(values, codes, groups):
    return np.bincount(codes, weights=np.asarray(values, float), minlength=groups)


def _standalone(truth, prediction, codes, counts):
    error = np.asarray(prediction, float) - truth
    sse = _group_sum(error * error, codes, len(counts))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    return {
        "standalone_rmse": float(np.sqrt(np.sum(sse) / np.sum(counts))),
        "standalone_mean_well_rmse": float(np.mean(well_rmse)),
        "standalone_median_well_rmse": float(np.median(well_rmse)),
        "standalone_p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }


def _pair(truth, base, candidate, codes, counts):
    groups = len(counts)
    base_error = np.asarray(base, float) - truth
    candidate_error = np.asarray(candidate, float) - truth
    direction = np.asarray(candidate, float) - np.asarray(base, float)
    a = _group_sum(base_error * base_error, codes, groups)
    b = _group_sum(base_error * direction, codes, groups)
    c = _group_sum(direction * direction, codes, groups)

    denominator = float(
        np.sqrt(np.dot(base_error, base_error) * np.dot(candidate_error, candidate_error))
    )
    cosine = float(np.dot(base_error, candidate_error) / denominator) if denominator else 1.0

    weights = np.linspace(0.0, 1.0, 101)
    grid_sse = np.maximum(
        a[:, None] + 2.0 * b[:, None] * weights[None, :] + c[:, None] * weights[None, :] ** 2,
        0.0,
    )
    grid_mean = np.mean(np.sqrt(grid_sse / counts[:, None]), axis=0)
    grid_pooled = np.sqrt(np.sum(grid_sse, axis=0) / np.sum(counts))
    best = int(np.argmin(grid_mean))

    oracle_weight = np.zeros(groups, float)
    valid = c > 1e-12
    oracle_weight[valid] = np.clip(-b[valid] / c[valid], 0.0, 1.0)
    oracle_sse = np.maximum(
        a + 2.0 * b * oracle_weight + c * oracle_weight * oracle_weight, 0.0
    )
    oracle_well = np.sqrt(oracle_sse / counts)
    return {
        "error_cosine": cosine,
        "fixed_weight_candidate": float(weights[best]),
        "fixed_rmse": float(grid_pooled[best]),
        "fixed_mean_well_rmse": float(grid_mean[best]),
        "oracle_rmse": float(np.sqrt(np.sum(oracle_sse) / np.sum(counts))),
        "oracle_mean_well_rmse": float(np.mean(oracle_well)),
        "oracle_median_well_rmse": float(np.median(oracle_well)),
        "oracle_p95_well_rmse": float(np.quantile(oracle_well, 0.95)),
        "mean_oracle_weight_candidate": float(np.mean(oracle_weight)),
        "active_oracle_well_fraction": float(np.mean(oracle_weight > 0.05)),
    }


def _feature_path(well):
    return FEATURE_DIR / f"{well}__full.pkl"


started = time.time()
if not TRACK6_PATH.exists():
    raise FileNotFoundError(TRACK6_PATH)
meta = pd.read_csv(META_PATH, dtype={"well": str})
test_wells = meta.loc[meta["fold"].eq(TEST_FOLD), "well"].tolist()
if len(test_wells) != 155:
    raise RuntimeError(f"unexpected Fold1 wells: {len(test_wells)}")

oof = pd.read_pickle(TRACK6_PATH)
oof["id"] = oof["id"].astype(str)
if oof["id"].duplicated().any():
    raise RuntimeError("Track6 OOF contains duplicate ids")
truth = oof["tvt_true"].to_numpy(np.float64)
sp45 = oof["sp45"].to_numpy(np.float64)
codes, well_names = pd.factorize(oof["well"].astype(str), sort=False)
counts = np.bincount(codes).astype(float)

schema_frames = [pd.read_pickle(_feature_path(well)).head(512) for well in test_wells[:8]]
schema = pd.concat(schema_frames, ignore_index=True, sort=False)
excluded = {
    "id",
    "well",
    "fold",
    "target",
    "TVT",
    "TVT_input",
    "last_known_tvt",
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "_row_idx",
    "sample_weight",
}
path_columns = []
for column in schema.columns:
    if column in excluded or not pd.api.types.is_numeric_dtype(schema[column]):
        continue
    lower = column.lower()
    if any(token in lower for token in ("truth", "oracle", "target")):
        continue
    values = schema[column].replace([np.inf, -np.inf], np.nan).dropna()
    if not len(values):
        continue
    if float(np.nanmedian(np.abs(values.to_numpy(float)))) > 1000.0:
        path_columns.append(column)
path_columns = list(dict.fromkeys(path_columns))
del schema, schema_frames
gc.collect()
if not path_columns:
    raise RuntimeError("no absolute path columns found in Fold1 feature cache")

print(
    "RAW PATH AUDIT CONFIG",
    {"rows": len(oof), "wells": len(counts), "path_columns": path_columns},
    flush=True,
)

frames = []
for number, well in enumerate(test_wells, start=1):
    frame = pd.read_pickle(_feature_path(well))
    keep = ["id", "well"] + [column for column in path_columns if column in frame]
    frame = frame.reindex(columns=keep)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = str(well)
    frames.append(frame)
    if number % 50 == 0 or number == len(test_wells):
        print(f"load raw paths {number}/{len(test_wells)}", flush=True)
raw = pd.concat(frames, ignore_index=True, sort=False)
del frames
gc.collect()

data = oof[["id"]].merge(raw, on="id", how="left", validate="one_to_one")
if len(data) != len(oof):
    raise RuntimeError(f"row mismatch: {len(data)} vs {len(oof)}")

rows = []
for column in path_columns:
    if column not in data:
        continue
    raw_values = data[column].to_numpy(np.float64)
    finite = np.isfinite(raw_values)
    coverage = float(np.mean(finite))
    if coverage < MIN_COVERAGE:
        continue
    if not finite.all():
        raw_values = np.where(finite, raw_values, np.nanmedian(raw_values[finite]))
    for transform, values in (
        ("raw", raw_values),
        ("sg61", _smooth_by_well(raw_values, codes, 61)),
    ):
        standalone = _standalone(truth, values, codes, counts)
        if standalone["standalone_rmse"] > MAX_STANDALONE_RMSE:
            continue
        rows.append(
            {
                "candidate": column,
                "transform": transform,
                "coverage": coverage,
                **standalone,
                **_pair(truth, sp45, values, codes, counts),
            }
        )

# Carry the trained endpoint into the same table for reference.
for column in ("track6", "no_pf_z", "shuffled_independent_evidence"):
    if column not in oof:
        continue
    values = oof[column].to_numpy(np.float64)
    rows.append(
        {
            "candidate": column,
            "transform": "trained_reference",
            "coverage": 1.0,
            **_standalone(truth, values, codes, counts),
            **_pair(truth, sp45, values, codes, counts),
        }
    )

report = pd.DataFrame(rows).sort_values(
    ["oracle_mean_well_rmse", "error_cosine", "fixed_mean_well_rmse"],
    ignore_index=True,
)
report.to_csv(OUT_DIR / "raw_path_pair_scan.csv", index=False)

passing = report[
    report["oracle_mean_well_rmse"].le(ORACLE_MEAN_GATE)
    & report["error_cosine"].le(COSINE_GATE)
]
best = report.iloc[0].to_dict() if len(report) else None
verdict = {
    "experiment": "track6_raw_path_independence_audit_v2",
    "stage": "zero_training_pre_gbdt_path_independence",
    "rows": int(len(oof)),
    "wells": int(len(counts)),
    "raw_path_columns": len(path_columns),
    "evaluated_variants": int(len(report)),
    "gates": {
        "error_cosine_le": COSINE_GATE,
        "two_way_oracle_mean_well_rmse_le": ORACLE_MEAN_GATE,
    },
    "best": best,
    "passing_variants": int(len(passing)),
    "strict_pass": bool(len(passing) > 0),
    "next": (
        "PASS -> preserve the winning raw endpoint and train only its confidence/calibration layer"
        if len(passing) > 0
        else "FAIL -> generate a genuinely new multi-seed particle-filter family; cached paths lack Track6 independence"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nTOP RAW PATHS BY SP45 TWO-WAY ORACLE")
print(report.head(50).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

