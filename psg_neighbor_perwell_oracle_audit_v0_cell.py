"""Zero-training per-well oracle audit for the PSG-SP45 branch.

This cell asks whether the small, unstable fixed gain left after the frozen
CP12 + full_progress_geom neighbor endpoint can be recovered by a deployable
per-well confidence gate. Truth is used only to measure the oracle ceiling and
feature associations; no prediction vector from this cell is deployable.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


EXPERIMENT = "psg_neighbor_perwell_oracle_audit_v0"
ROOT = Path("/root/autodl-tmp/oof_exports")
SURVIVAL_PATH = ROOT / "psg_sp45_branch_survival_audit_v0" / "neighbor_branch_crossfit_oof.parquet"
BRANCH_PATH = ROOT / "psg_sp45_crossfit_audit_v0" / "psg_sp45_crossfit_oof.parquet"
NEIGHBOR_META_PATH = ROOT / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"
OUT_DIR = ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORACLE_GAIN_GATE_FT = 0.30
MIN_HELPED_WELL_FRACTION = 0.20
FEATURE_SPEARMAN_GATE = 0.15
FEATURE_AUC_GATE = 0.60


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _metrics(truth, prediction, wells):
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    codes, names = pd.factorize(np.asarray(wells).astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error**2, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    return {
        "rows": int(len(error)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }


def _safe_spearman(feature, target):
    feature = np.asarray(feature, np.float64)
    target = np.asarray(target, np.float64)
    valid = np.isfinite(feature) & np.isfinite(target)
    if int(valid.sum()) < 10 or float(np.nanstd(feature[valid])) <= 1e-12:
        return math.nan
    return float(spearmanr(feature[valid], target[valid]).statistic)


def _safe_auc(feature, target):
    feature = np.asarray(feature, np.float64)
    target = np.asarray(target, bool)
    valid = np.isfinite(feature)
    if int(valid.sum()) < 10 or len(np.unique(target[valid])) < 2:
        return math.nan
    score = float(roc_auc_score(target[valid], feature[valid]))
    return max(score, 1.0 - score)


started = time.time()
for path in (SURVIVAL_PATH, BRANCH_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

survival = pd.read_parquet(SURVIVAL_PATH)
branch = pd.read_parquet(
    BRANCH_PATH,
    columns=["id", "md_since", "sp45", "psg_tvt", "psg_guarded", "crossfit_blend"],
)
survival["id"] = survival["id"].astype(str).str.lower()
branch["id"] = branch["id"].astype(str).str.lower()
if survival["id"].duplicated().any() or branch["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in oracle inputs")

data = survival.merge(branch, on="id", how="left", validate="one_to_one", suffixes=("", "__branch"))
required = {
    "well",
    "fold",
    "truth",
    "cp12_neighbor_full_progress_geom",
    "psg_sp45_branch",
    "md_since",
    "sp45",
    "psg_tvt",
}
missing = sorted(required.difference(data.columns))
if missing:
    raise RuntimeError(f"Oracle data missing columns: {missing}")
if not np.isfinite(data[list(required - {"well"})].to_numpy(np.float64)).all():
    raise RuntimeError("Oracle data contains non-finite numeric values")

data["well"] = data["well"].astype(str).str.lower()
truth = data["truth"].to_numpy(np.float64)
base = data["cp12_neighbor_full_progress_geom"].to_numpy(np.float64)
candidate = data["psg_sp45_branch"].to_numpy(np.float64)
wells = data["well"].to_numpy()
base_error = base - truth
candidate_error = candidate - truth
direction = candidate - base

codes, well_names = pd.factorize(data["well"], sort=False)
groups = len(well_names)
counts = np.bincount(codes).astype(np.float64)
base_sse = np.bincount(codes, weights=base_error**2, minlength=groups)
candidate_sse = np.bincount(codes, weights=candidate_error**2, minlength=groups)
cross = np.bincount(codes, weights=base_error * direction, minlength=groups)
direction_sse = np.bincount(codes, weights=direction**2, minlength=groups)

oracle_weight = np.zeros(groups, np.float64)
valid_direction = direction_sse > 1e-12
oracle_weight[valid_direction] = np.clip(
    -cross[valid_direction] / direction_sse[valid_direction], 0.0, 1.0
)
row_oracle_weight = oracle_weight[codes]
oracle_prediction = base + row_oracle_weight * direction
hard_use_candidate = candidate_sse < base_sse
hard_prediction = np.where(hard_use_candidate[codes], candidate, base)

first_position = np.full(groups, len(data), np.int64)
np.minimum.at(first_position, codes, np.arange(len(data), dtype=np.int64))
fold_by_well = data["fold"].to_numpy(np.int8)[first_position]
distance = data["md_since"].to_numpy(np.float64)
abs_disagreement = np.abs(direction)
signed_disagreement = direction
psg_sp45_disagreement = data["psg_tvt"].to_numpy(np.float64) - data["sp45"].to_numpy(np.float64)


def _group_mean(values):
    return np.bincount(codes, weights=np.asarray(values, np.float64), minlength=groups) / counts


def _group_rms(values):
    values = np.asarray(values, np.float64)
    return np.sqrt(np.bincount(codes, weights=values**2, minlength=groups) / counts)


def _group_quantile(values, quantile):
    frame = pd.DataFrame({"well": data["well"].to_numpy(), "value": np.asarray(values)})
    return frame.groupby("well", sort=False)["value"].quantile(quantile).reindex(well_names).to_numpy()


features = pd.DataFrame(
    {
        "well": well_names,
        "fold": fold_by_well,
        "rows": counts,
        "max_md_since": np.maximum.reduceat(
            distance[np.argsort(codes, kind="stable")], np.r_[0, np.cumsum(counts.astype(int))[:-1]]
        ),
        "mean_abs_base_branch_disagreement": _group_mean(abs_disagreement),
        "rms_base_branch_disagreement": _group_rms(direction),
        "p95_abs_base_branch_disagreement": _group_quantile(abs_disagreement, 0.95),
        "mean_signed_base_branch_disagreement": _group_mean(signed_disagreement),
        "endpoint_base_branch_disagreement": np.zeros(groups, np.float64),
        "mean_abs_psg_sp45_disagreement": _group_mean(np.abs(psg_sp45_disagreement)),
        "rms_psg_sp45_disagreement": _group_rms(psg_sp45_disagreement),
    }
)

# Replace the endpoint feature with an explicit last-row lookup; the expression
# above is only a shape-preserving initializer for irregular well lengths.
last_position = np.zeros(groups, np.int64)
np.maximum.at(last_position, codes, np.arange(len(data), dtype=np.int64))
features["endpoint_base_branch_disagreement"] = signed_disagreement[last_position]

for lower, upper, label in (
    (0.0, 500.0, "0_500"),
    (500.0, 1000.0, "500_1000"),
    (1000.0, 2000.0, "1000_2000"),
    (2000.0, np.inf, "gt2000"),
):
    selected = (distance >= lower) & (distance < upper)
    selected_counts = np.bincount(codes[selected], minlength=groups).astype(np.float64)
    sums = np.bincount(codes[selected], weights=abs_disagreement[selected], minlength=groups)
    features[f"mean_abs_disagreement_{label}"] = np.divide(
        sums, selected_counts, out=np.zeros(groups), where=selected_counts > 0
    )

if NEIGHBOR_META_PATH.exists():
    meta = pd.read_csv(NEIGHBOR_META_PATH)
    meta["well"] = meta["well"].astype(str).str.lower()
    meta = meta.drop_duplicates("well").set_index("well")
    for column in (
        "min_distance",
        "median_distance",
        "p90_distance",
        "anchor_distance_full",
        "abs_direction_dot",
        "direction_dot",
    ):
        if column in meta:
            features[f"neighbor_{column}"] = features["well"].map(meta[column])

features["base_rmse"] = np.sqrt(base_sse / counts)
features["candidate_rmse"] = np.sqrt(candidate_sse / counts)
features["candidate_benefit_mse"] = (base_sse - candidate_sse) / counts
features["candidate_wins"] = hard_use_candidate
features["oracle_weight"] = oracle_weight

excluded = {
    "well",
    "fold",
    "base_rmse",
    "candidate_rmse",
    "candidate_benefit_mse",
    "candidate_wins",
    "oracle_weight",
}
association_rows = []
for column in features.columns:
    if column in excluded:
        continue
    values = pd.to_numeric(features[column], errors="coerce").to_numpy(np.float64)
    association_rows.append(
        {
            "feature": column,
            "spearman_oracle_weight": _safe_spearman(values, oracle_weight),
            "spearman_candidate_benefit": _safe_spearman(values, features["candidate_benefit_mse"]),
            "abs_auc_candidate_wins": _safe_auc(values, hard_use_candidate),
        }
    )
associations = pd.DataFrame(association_rows)
associations["best_abs_spearman"] = associations[
    ["spearman_oracle_weight", "spearman_candidate_benefit"]
].abs().max(axis=1)
associations = associations.sort_values(
    ["best_abs_spearman", "abs_auc_candidate_wins"], ascending=False, ignore_index=True
)

base_metrics = _metrics(truth, base, wells)
candidate_metrics = _metrics(truth, candidate, wells)
oracle_metrics = _metrics(truth, oracle_prediction, wells)
hard_metrics = _metrics(truth, hard_prediction, wells)
oracle_gain = base_metrics["rmse"] - oracle_metrics["rmse"]
best_spearman = float(associations["best_abs_spearman"].max())
best_auc = float(associations["abs_auc_candidate_wins"].max())
helped_fraction = float(np.mean(hard_use_candidate))

checks = {
    "perwell_oracle_gain_ge_0p30": oracle_gain >= ORACLE_GAIN_GATE_FT,
    "candidate_helps_ge_20pct_wells": helped_fraction >= MIN_HELPED_WELL_FRACTION,
    "feature_spearman_ge_0p15_or_auc_ge_0p60": (
        best_spearman >= FEATURE_SPEARMAN_GATE or best_auc >= FEATURE_AUC_GATE
    ),
}
strict_pass = bool(all(checks.values()))

features.to_csv(OUT_DIR / "per_well_oracle_and_features.csv", index=False)
associations.to_csv(OUT_DIR / "feature_associations.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "zero_training_perwell_confidence_gate_ceiling",
    "rows": len(data),
    "wells": groups,
    "base": base_metrics,
    "candidate": candidate_metrics,
    "perwell_convex_oracle": oracle_metrics,
    "perwell_hard_oracle": hard_metrics,
    "oracle_gain": oracle_gain,
    "candidate_helped_well_fraction": helped_fraction,
    "mean_oracle_weight": float(np.mean(oracle_weight)),
    "median_oracle_weight": float(np.median(oracle_weight)),
    "best_feature_abs_spearman": best_spearman,
    "best_feature_abs_auc": best_auc,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> train a nested well-level confidence gate; never use oracle weights in deployment"
        if strict_pass
        else "FAIL -> no reachable per-well ceiling; close PSG after the SP45 diagnostic branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nPSG-NEIGHBOR ORACLE SUMMARY")
print(
    pd.DataFrame(
        [
            {"variant": "base", **base_metrics},
            {"variant": "psg_sp45_branch", **candidate_metrics},
            {"variant": "perwell_convex_oracle", **oracle_metrics},
            {"variant": "perwell_hard_oracle", **hard_metrics},
        ]
    ).to_string(index=False),
    flush=True,
)
print("\nTOP LABEL-FREE FEATURE ASSOCIATIONS")
print(associations.head(20).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
