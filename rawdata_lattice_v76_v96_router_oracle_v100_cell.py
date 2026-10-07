"""V100: oracle ceiling for routing between V76 and V96 dose models.

V76 and V96 share the same parent/coarse endpoints and differ only in their
per-well scalar dose. This read-only audit reconstructs both dose predictions
and measures whether a legal router could have enough headroom to matter.
Truth is used only for oracle metrics. No deployment model is produced.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_v76_v96_router_oracle_v100"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V96_SOURCE = ROOT / "rawdata_lattice_v76_full773_training_v96_cell.py"
V96_COARSE = (
    OOF_ROOT / "rawdata_lattice_v76_full773_training_v96"
    / "fivefold_coarse_oof.parquet"
)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _bootstrap_v96():
    if not V96_SOURCE.is_file():
        raise FileNotFoundError(V96_SOURCE)
    source = V96_SOURCE.read_text(encoding="utf-8")
    marker = "for path in (V4_SOURCE, V66_PATH, V70_PATH, V76_BUNDLE, V76_OOF_WEIGHTS):"
    if marker not in source:
        raise RuntimeError(f"Cannot identify V96 definition boundary in {V96_SOURCE}")
    module = types.ModuleType("v100_v96_definitions")
    module.__file__ = str(V96_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V96_SOURCE), "exec"), module.__dict__)
    return module


def _fixed_alpha(base, candidate, truth):
    move = candidate - base
    denominator = float(np.dot(move, move))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(truth - base, move) / denominator, 0.0, 1.0))


def _metrics(frame, prediction, endpoint):
    truth = frame[N.TRUTH].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    by_well = pd.DataFrame(
        {"well": frame["well"].astype(str), "e2": error ** 2}
    ).groupby("well")["e2"].mean()
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well.to_numpy()))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well.to_numpy()), 0.95)),
        "far_gt2000_rmse": (
            float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan
        ),
    }


N = _bootstrap_v96()
for required in (V96_COARSE, N.V76_BUNDLE, N.V76_OOF_WEIGHTS):
    if not Path(required).is_file():
        raise FileNotFoundError(required)

complete = pd.read_parquet(V96_COARSE).sort_values(
    ["fold", "well", "row_idx"], kind="stable"
).reset_index(drop=True)
if len(complete) != 3_783_989 or complete["well"].nunique() != 773:
    raise RuntimeError(
        f"Invalid V96 coarse coverage: rows={len(complete)} "
        f"wells={complete['well'].nunique()}"
    )

features = pd.DataFrame(
    [
        N._feature_row(local)
        for _, local in complete.groupby(["fold", "well"], sort=False)
    ]
).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
x, feature_names = N._augment(features)
y = features["oracle_weight"].to_numpy(np.float64)
sample_weight = np.sqrt(features["rows"].to_numpy(np.float64))

# Exact V96 crossfit reconstruction.
expanded_weights = np.full(len(features), np.nan, np.float64)
for heldout_fold in N.ALL_FOLDS:
    valid = features["fold"].to_numpy(np.int64) == int(heldout_fold)
    train = ~valid
    hist = N._hist()
    ridge = N._ridge()
    hist.fit(x[train], y[train], sample_weight=sample_weight[train])
    ridge.fit(x[train], y[train])
    expanded_weights[valid] = np.clip(
        0.5 * hist.predict(x[valid]) + 0.5 * ridge.predict(x[valid]),
        N.MIN_WEIGHT,
        N.MAX_WEIGHT,
    )

# Exact matched V76 reconstruction used by V96.
old_bundle = joblib.load(N.V76_BUNDLE)
baseline_weights = np.clip(
    old_bundle["hist_fraction"] * old_bundle["hist_model"].predict(x)
    + old_bundle["ridge_fraction"] * old_bundle["ridge_model"].predict(x),
    N.MIN_WEIGHT,
    N.MAX_WEIGHT,
)
old_oof = pd.read_parquet(
    N.V76_OOF_WEIGHTS, columns=["fold", "well", "ensemble_weight"]
)
old_oof_map = dict(
    zip(old_oof["well"].astype(str), old_oof["ensemble_weight"].astype(float))
)
honest_old = features["fold"].isin(N.PRIOR_FOLDS).to_numpy()
baseline_weights[honest_old] = (
    features.loc[honest_old, "well"].astype(str).map(old_oof_map).to_numpy(np.float64)
)
if not np.isfinite(baseline_weights).all() or not np.isfinite(expanded_weights).all():
    raise RuntimeError("Non-finite V76/V96 reconstructed weights")

baseline_map = dict(zip(features["well"].astype(str), baseline_weights))
expanded_map = dict(zip(features["well"].astype(str), expanded_weights))
row_baseline_weight = (
    complete["well"].astype(str).map(baseline_map).to_numpy(np.float64)
)
row_expanded_weight = (
    complete["well"].astype(str).map(expanded_map).to_numpy(np.float64)
)
parent = complete[N.PARENT].to_numpy(np.float64)
coarse_move = complete[N.COARSE].to_numpy(np.float64) - parent
truth = complete[N.TRUTH].to_numpy(np.float64)
v76 = parent + row_baseline_weight * coarse_move
v96 = parent + row_expanded_weight * coarse_move

summary_rows = []
by_well_rows = []
fixed_rows = []
oracle = np.empty(len(complete), np.float64)
for fold in (0, *N.ALL_FOLDS):
    mask = (
        np.ones(len(complete), dtype=bool)
        if fold == 0
        else complete["fold"].to_numpy(np.int64) == int(fold)
    )
    local = complete.loc[mask]
    local_truth = truth[mask]
    local_v76 = v76[mask]
    local_v96 = v96[mask]
    alpha = _fixed_alpha(local_v76, local_v96, local_truth)
    fixed = local_v76 + alpha * (local_v96 - local_v76)
    fixed_rows.append({"fold": int(fold), "fixed_alpha_v96": alpha})
    summary_rows.extend(
        [
            {"fold": int(fold), **_metrics(local, local_v76, "v76")},
            {"fold": int(fold), **_metrics(local, local_v96, "v96")},
            {"fold": int(fold), **_metrics(local, fixed, "fixed_oracle")},
        ]
    )

for well, indices in complete.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    local_truth = truth[indices]
    local_v76 = v76[indices]
    local_v96 = v96[indices]
    alpha = _fixed_alpha(local_v76, local_v96, local_truth)
    local_oracle = local_v76 + alpha * (local_v96 - local_v76)
    oracle[indices] = local_oracle
    v76_rmse = float(np.sqrt(np.mean((local_v76 - local_truth) ** 2)))
    v96_rmse = float(np.sqrt(np.mean((local_v96 - local_truth) ** 2)))
    oracle_rmse = float(np.sqrt(np.mean((local_oracle - local_truth) ** 2)))
    feature_index = int(np.flatnonzero(features["well"].astype(str).eq(str(well)))[0])
    by_well_rows.append(
        {
            "well": str(well),
            "fold": int(complete.iloc[indices[0]]["fold"]),
            "rows": len(indices),
            "v76_weight": float(baseline_weights[feature_index]),
            "v96_weight": float(expanded_weights[feature_index]),
            "weight_difference": float(expanded_weights[feature_index] - baseline_weights[feature_index]),
            "oracle_alpha_v96": alpha,
            "v76_rmse": v76_rmse,
            "v96_rmse": v96_rmse,
            "oracle_rmse": oracle_rmse,
            "oracle_gain_vs_best": min(v76_rmse, v96_rmse) - oracle_rmse,
        }
    )

summary_rows.append({"fold": 0, **_metrics(complete, oracle, "perwell_oracle")})
summary = pd.DataFrame(summary_rows)
by_well = pd.DataFrame(by_well_rows)
fixed_table = pd.DataFrame(fixed_rows)


def _row(endpoint):
    return summary.loc[
        summary["fold"].eq(0) & summary["endpoint"].eq(endpoint)
    ].iloc[0]


v76_metric = _row("v76")
v96_metric = _row("v96")
fixed_metric = _row("fixed_oracle")
oracle_metric = _row("perwell_oracle")
best_standalone_rmse = min(float(v76_metric["rmse"]), float(v96_metric["rmse"]))
fixed_gain = best_standalone_rmse - float(fixed_metric["rmse"])
oracle_gain = best_standalone_rmse - float(oracle_metric["rmse"])
error_correlation = float(np.corrcoef(v76 - truth, v96 - truth)[0, 1])
endpoint_disagreement_rms = float(np.sqrt(np.mean((v96 - v76) ** 2)))
meaningful_preference = float(
    np.mean(
        (by_well["oracle_alpha_v96"] > 0.10)
        & (by_well["oracle_alpha_v96"] < 0.90)
    )
)

checks = {
    "perwell_oracle_gain_ge_0p08": oracle_gain >= 0.08,
    "fixed_oracle_gain_ge_0p01": fixed_gain >= 0.01,
    "endpoint_disagreement_rms_ge_0p10": endpoint_disagreement_rms >= 0.10,
    "error_correlation_le_0p995": error_correlation <= 0.995,
    "interior_oracle_fraction_ge_0p15": meaningful_preference >= 0.15,
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": EXPERIMENT,
    "stage": "read_only_V76_V96_router_oracle_investment_gate",
    "selection_or_tuning_count": 0,
    "truth_usage": "oracle metrics only",
    "coverage": {"rows": len(complete), "wells": complete["well"].nunique()},
    "standalone": {
        "v76_rmse": float(v76_metric["rmse"]),
        "v96_rmse": float(v96_metric["rmse"]),
    },
    "fixed_oracle": {
        "rmse": float(fixed_metric["rmse"]),
        "gain_vs_best_standalone": fixed_gain,
        "weights_by_fold": fixed_table.to_dict("records"),
    },
    "perwell_oracle": {
        "rmse": float(oracle_metric["rmse"]),
        "gain_vs_best_standalone": oracle_gain,
        "interior_alpha_fraction": meaningful_preference,
    },
    "error_correlation": error_correlation,
    "endpoint_disagreement_rms": endpoint_disagreement_rms,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> test a fold-clean legal router between V76 and V96"
        if strict_pass
        else "FAIL -> fold reversal is not an exploitable endpoint difference; retain V76"
    ),
}

summary.to_csv(OUT_DIR / "summary.csv", index=False)
by_well.to_csv(OUT_DIR / "by_well.csv", index=False)
fixed_table.to_csv(OUT_DIR / "fixed_weights.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)

print("V100 SUMMARY")
print(summary.to_string(index=False))
print("\nV100 FIXED WEIGHTS")
print(fixed_table.to_string(index=False))
print("\nV100 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
