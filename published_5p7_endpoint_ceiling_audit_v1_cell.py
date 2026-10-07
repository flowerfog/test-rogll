"""Zero-training endpoint ceiling audit for the published 5.7 recipe.

The published recipe needs two genuinely complementary complete-well
estimators before confidence gating can help. This audit scans every numeric
path-like column in the Fold1 candidate pool and measures that prerequisite.

Run after ``track8_sp45_confidence_nested_gateway_v0_cell.py`` so the exact
Fold1 truth, wells, Track8 prediction, and historical endpoints are already in
one OOF artifact. No model is trained and no candidate is promoted here.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
TRACK8_ROOT = OOF_ROOT / "track8_sp45_confidence_nested_gateway_v0"
TRACK8_PATH = TRACK8_ROOT / "fold1_track8_sp45_oof.pkl"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "published_5p7_endpoint_ceiling_audit_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUBLISHED_TWO_WAY_ORACLE_MEAN_WELL = 5.11
ORACLE_GATE_MEAN_WELL = 5.30
FIXED_BLEND_GATE_MEAN_WELL = 6.40
MIN_COVERAGE = 0.995
MAX_STANDALONE_RMSE = 40.0
BASES = ("sp45", "cp12", "strong_bias", "strong")


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _safe_name(value):
    return str(value).replace(" ", "_")


def _group_sums(values, codes, groups):
    return np.bincount(codes, weights=np.asarray(values, float), minlength=groups)


def _candidate_metrics(truth, prediction, codes, counts):
    error = np.asarray(prediction, float) - truth
    sse = _group_sums(error * error, codes, len(counts))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    return {
        "standalone_exact_sse": float(np.sum(sse)),
        "standalone_rmse": float(np.sqrt(np.sum(sse) / np.sum(counts))),
        "standalone_mean_well_rmse": float(np.mean(well_rmse)),
        "standalone_median_well_rmse": float(np.median(well_rmse)),
        "standalone_p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }


def _pair_metrics(truth, base, candidate, codes, counts):
    groups = len(counts)
    base_error = np.asarray(base, float) - truth
    candidate = np.asarray(candidate, float)
    direction = candidate - np.asarray(base, float)

    a = _group_sums(base_error * base_error, codes, groups)
    b = _group_sums(base_error * direction, codes, groups)
    c = _group_sums(direction * direction, codes, groups)

    weights = np.linspace(0.0, 1.0, 101)
    sse_grid = (
        a[:, None]
        + 2.0 * b[:, None] * weights[None, :]
        + c[:, None] * weights[None, :] ** 2
    )
    sse_grid = np.maximum(sse_grid, 0.0)
    mean_well_grid = np.mean(np.sqrt(sse_grid / counts[:, None]), axis=0)
    pooled_grid = np.sqrt(np.sum(sse_grid, axis=0) / np.sum(counts))
    fixed_index = int(np.argmin(mean_well_grid))

    oracle_weight = np.zeros(groups, dtype=float)
    valid = c > 1e-12
    oracle_weight[valid] = np.clip(-b[valid] / c[valid], 0.0, 1.0)
    oracle_sse = np.maximum(
        a + 2.0 * b * oracle_weight + c * oracle_weight * oracle_weight,
        0.0,
    )
    oracle_well_rmse = np.sqrt(oracle_sse / counts)

    candidate_sse = np.maximum(a + 2.0 * b + c, 0.0)
    hard_sse = np.minimum(a, candidate_sse)
    hard_well_rmse = np.sqrt(hard_sse / counts)

    candidate_error = candidate - truth
    denominator = float(
        np.sqrt(np.dot(base_error, base_error) * np.dot(candidate_error, candidate_error))
    )
    cosine = float(np.dot(base_error, candidate_error) / denominator) if denominator else 1.0

    return {
        "error_cosine": cosine,
        "fixed_weight_candidate": float(weights[fixed_index]),
        "fixed_rmse": float(pooled_grid[fixed_index]),
        "fixed_mean_well_rmse": float(mean_well_grid[fixed_index]),
        "oracle_rmse": float(np.sqrt(np.sum(oracle_sse) / np.sum(counts))),
        "oracle_mean_well_rmse": float(np.mean(oracle_well_rmse)),
        "oracle_median_well_rmse": float(np.median(oracle_well_rmse)),
        "oracle_p95_well_rmse": float(np.quantile(oracle_well_rmse, 0.95)),
        "hard_oracle_rmse": float(np.sqrt(np.sum(hard_sse) / np.sum(counts))),
        "hard_oracle_mean_well_rmse": float(np.mean(hard_well_rmse)),
        "mean_oracle_weight_candidate": float(np.mean(oracle_weight)),
        "oracle_active_well_fraction": float(np.mean(oracle_weight > 0.05)),
        "oracle_full_candidate_fraction": float(np.mean(oracle_weight > 0.95)),
    }


started = time.time()
if not TRACK8_PATH.exists():
    raise FileNotFoundError(
        f"Missing {TRACK8_PATH}. Run the Track8/SP45 nested gateway first."
    )
if not POOL_PATH.exists():
    raise FileNotFoundError(POOL_PATH)

oof = pd.read_pickle(TRACK8_PATH)
required = {"id", "well", "tvt_true", "sp45", "track8"}
if not required.issubset(oof.columns):
    raise RuntimeError(f"Track8 OOF missing {sorted(required - set(oof.columns))}")
if oof["id"].astype(str).duplicated().any():
    raise RuntimeError("Track8 OOF contains duplicate ids")
oof["id"] = oof["id"].astype(str)

truth = oof["tvt_true"].to_numpy(np.float64)
codes, well_names = pd.factorize(oof["well"].astype(str), sort=False)
counts = np.bincount(codes).astype(np.float64)
groups = len(counts)

probe = pd.read_csv(POOL_PATH, nrows=2048)
excluded_tokens = ("truth", "target", "oracle", "error", "rmse", "sse")
excluded_exact = {
    "id",
    "well",
    "fold",
    "tvt_true",
    "TVT",
    "TVT_input",
    "target",
    "row_idx",
    "_row_idx",
}
candidate_columns = []
for column in probe.columns:
    if column in excluded_exact or any(token in column.lower() for token in excluded_tokens):
        continue
    converted = pd.to_numeric(probe[column], errors="coerce")
    if float(converted.notna().mean()) < 0.98:
        continue
    candidate_columns.append(column)

print(
    "ENDPOINT AUDIT CONFIG",
    {
        "rows": len(oof),
        "wells": groups,
        "pool_numeric_columns": len(candidate_columns),
        "bases": list(BASES),
    },
    flush=True,
)

pool = pd.read_csv(
    POOL_PATH,
    usecols=["id"] + candidate_columns,
    dtype={"id": str, **{column: np.float32 for column in candidate_columns}},
)
if pool["id"].duplicated().any():
    raise RuntimeError("candidate pool contains duplicate ids")
data = oof.merge(pool, on="id", how="left", suffixes=("", "__pool"), validate="one_to_one")
if len(data) != len(oof):
    raise RuntimeError(f"row mismatch: oof={len(oof)} merged={len(data)}")

# Prefer the already-audited OOF values where names collide.
prediction_arrays = {}
for column in candidate_columns:
    merged_column = column if column not in oof.columns else column + "__pool"
    if merged_column not in data:
        continue
    values = data[merged_column].to_numpy(np.float64)
    finite = np.isfinite(values)
    coverage = float(np.mean(finite))
    if coverage < MIN_COVERAGE:
        continue
    if not finite.all():
        values = np.where(finite, values, np.nanmedian(values[finite]))
    score = _rmse(truth, values)
    if not (0.05 < score <= MAX_STANDALONE_RMSE):
        continue
    prediction_arrays[column] = values

for column in ("track8", "pf_ancc", "sp45", "cp12", "strong_bias", "strong", "fleongg"):
    if column in oof:
        values = oof[column].to_numpy(np.float64)
        if np.isfinite(values).mean() >= MIN_COVERAGE:
            prediction_arrays[column] = values

available_bases = [base for base in BASES if base in prediction_arrays]
if "sp45" not in available_bases:
    raise RuntimeError(f"SP45 missing after candidate validation: {sorted(prediction_arrays)}")

standalone_rows = []
for name, values in prediction_arrays.items():
    standalone_rows.append(
        {
            "candidate": name,
            "coverage": float(np.mean(np.isfinite(values))),
            **_candidate_metrics(truth, values, codes, counts),
        }
    )
standalone = pd.DataFrame(standalone_rows).sort_values("standalone_rmse", ignore_index=True)

pair_rows = []
for base_name in available_bases:
    base = prediction_arrays[base_name]
    base_metrics = _candidate_metrics(truth, base, codes, counts)
    for candidate_name, candidate in prediction_arrays.items():
        if candidate_name == base_name:
            continue
        metrics = _pair_metrics(truth, base, candidate, codes, counts)
        pair_rows.append(
            {
                "base": base_name,
                "candidate": candidate_name,
                "base_rmse": base_metrics["standalone_rmse"],
                "base_mean_well_rmse": base_metrics["standalone_mean_well_rmse"],
                "candidate_rmse": _rmse(truth, candidate),
                **metrics,
                "fixed_gain_mean_well": (
                    base_metrics["standalone_mean_well_rmse"]
                    - metrics["fixed_mean_well_rmse"]
                ),
                "oracle_gain_mean_well": (
                    base_metrics["standalone_mean_well_rmse"]
                    - metrics["oracle_mean_well_rmse"]
                ),
            }
        )

pairs = pd.DataFrame(pair_rows)
pairs = pairs.sort_values(
    ["oracle_mean_well_rmse", "fixed_mean_well_rmse"], ignore_index=True
)
sp45_pairs = pairs[pairs["base"].eq("sp45")].copy()

oracle_pass = pairs[
    pairs["oracle_mean_well_rmse"].le(ORACLE_GATE_MEAN_WELL)
]
fixed_pass = pairs[
    pairs["fixed_mean_well_rmse"].le(FIXED_BLEND_GATE_MEAN_WELL)
]
sp45_oracle_pass = sp45_pairs[
    sp45_pairs["oracle_mean_well_rmse"].le(ORACLE_GATE_MEAN_WELL)
]

standalone.to_csv(OUT_DIR / "standalone_endpoints.csv", index=False)
pairs.to_csv(OUT_DIR / "all_base_candidate_pairs.csv", index=False)
sp45_pairs.to_csv(OUT_DIR / "sp45_candidate_pairs.csv", index=False)

best_sp45 = sp45_pairs.iloc[0].to_dict() if len(sp45_pairs) else None
verdict = {
    "experiment": "published_5p7_endpoint_ceiling_audit_v1",
    "stage": "zero_training_complete_well_endpoint_ceiling",
    "rows": int(len(oof)),
    "wells": int(groups),
    "prediction_endpoints": int(len(prediction_arrays)),
    "available_bases": available_bases,
    "published_reference": {
        "standalone_mean_well_rmse": 7.28,
        "gated_mean_well_rmse": 6.21,
        "two_way_oracle_mean_well_rmse": PUBLISHED_TWO_WAY_ORACLE_MEAN_WELL,
    },
    "gates": {
        "two_way_oracle_mean_well_rmse_le": ORACLE_GATE_MEAN_WELL,
        "fixed_blend_mean_well_rmse_le": FIXED_BLEND_GATE_MEAN_WELL,
    },
    "best_sp45_pair": best_sp45,
    "all_base_oracle_pass_pairs": int(len(oracle_pass)),
    "sp45_oracle_pass_pairs": int(len(sp45_oracle_pass)),
    "fixed_blend_pass_pairs": int(len(fixed_pass)),
    "strict_pass": bool(len(sp45_oracle_pass) > 0),
    "next": (
        "PASS -> reproduce the top endpoint exactly, then cross-fit confidence gating"
        if len(sp45_oracle_pass) > 0
        else "FAIL -> the required Track6-like complementary endpoint is absent from the current pool"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nTOP STANDALONE ENDPOINTS")
print(standalone.head(30).to_string(index=False), flush=True)
print("\nTOP SP45 PAIRS BY TWO-WAY ORACLE")
print(sp45_pairs.head(40).to_string(index=False), flush=True)
print("\nTOP ALL-BASE PAIRS BY TWO-WAY ORACLE")
print(pairs.head(40).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

