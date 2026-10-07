"""Cheap ablation/rescore for the one-fold whole-path GR matcher.

Reuses saved candidate features.  It removes candidate identity, separates GR
evidence from path-shape priors, scans individual GR scores, and compares
LambdaRank with RMSE regression before deciding whether summary-feature path
matching is genuinely dead.
"""

from pathlib import Path
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "gr_path_matcher_onefold_v0"
FEATURE_PATH = SOURCE_DIR / "path_features.pkl"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
OUT_DIR = OOF_ROOT / "gr_path_matcher_ablation_rescore_v0"

TRAIN_FOLD = 5
VALID_FOLD = 1
PASS_GAIN = 0.10
PASS_POSITIVE_WELLS = 0.52
PASS_MEAN_REGRET = 3.00


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(truth, float) - np.asarray(prediction, float)) ** 2)))


def _selector_report(name, valid, selected, well_baseline):
    chosen = valid.loc[selected].copy()
    chosen = chosen.merge(well_baseline, on="well", how="left", validate="one_to_one")
    oracle = valid.groupby("well", as_index=False)["candidate_rmse"].min().rename(
        columns={"candidate_rmse": "oracle_rmse"}
    )
    chosen = chosen.merge(oracle, on="well", how="left", validate="one_to_one")
    chosen["gain"] = chosen["strong_rmse"] - chosen["candidate_rmse"]
    chosen["regret"] = chosen["candidate_rmse"] - chosen["oracle_rmse"]
    pooled_rmse = float(
        np.sqrt(np.sum(chosen["rows"] * chosen["candidate_rmse"] ** 2) / chosen["rows"].sum())
    )
    strong_pooled = float(
        np.sqrt(np.sum(chosen["rows"] * chosen["strong_rmse"] ** 2) / chosen["rows"].sum())
    )
    report = {
        "selector": name,
        "wells": len(chosen),
        "strong_rmse": strong_pooled,
        "rmse": pooled_rmse,
        "gain_vs_strong": strong_pooled - pooled_rmse,
        "mean_well_rmse": float(chosen["candidate_rmse"].mean()),
        "mean_well_gain": float(chosen["gain"].mean()),
        "median_well_gain": float(chosen["gain"].median()),
        "positive_well_fraction": float((chosen["gain"] > 0).mean()),
        "mean_regret": float(chosen["regret"].mean()),
        "median_regret": float(chosen["regret"].median()),
    }
    return report, chosen


def _safe_group_extreme(frame, column, maximize=False):
    """Select one row per well, falling back to the zero path for all-NA groups."""
    selected = []
    for _, group in frame.groupby("well", sort=False):
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if finite.any():
            local = int(np.nanargmax(values) if maximize else np.nanargmin(values))
            selected.append(group.index[local])
            continue
        zero = group.index[group["candidate_name"].eq("zero")]
        selected.append(int(zero[0]) if len(zero) else int(group.index[0]))
    return pd.Index(selected)


for path in (FEATURE_PATH,):
    if not path.exists():
        raise FileNotFoundError(path)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)

started = time.time()
features = pd.read_pickle(FEATURE_PATH)
features = features.replace([np.inf, -np.inf], np.nan)
train = features[features["fold"] == TRAIN_FOLD].sort_values(
    ["well", "candidate_index"]
).reset_index(drop=True)
valid = features[features["fold"] == VALID_FOLD].sort_values(
    ["well", "candidate_index"]
).reset_index(drop=True)

header = pd.read_csv(pool_path, nrows=0).columns.tolist()
usecols = [column for column in ("id", "well", "tvt_true", "strong") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
pool = pool[pool["well"].isin(valid["well"].unique())]
well_rows = []
for well, frame in pool.groupby("well", sort=False):
    well_rows.append(
        {
            "well": well,
            "rows": len(frame),
            "strong_rmse": _rmse(frame["tvt_true"], frame["strong"]),
        }
    )
well_baseline = pd.DataFrame(well_rows)

zero = valid[valid["candidate_name"] == "zero"].merge(
    well_baseline, on="well", validate="one_to_one"
)
zero_difference = float(np.max(np.abs(zero["candidate_rmse"] - zero["strong_rmse"])))
if zero_difference > 1e-3:
    raise RuntimeError(f"zero candidate does not reproduce strong: {zero_difference}")

metadata_columns = {
    "well", "candidate_name", "family", "fold", "candidate_rmse", "relevance",
    "candidate_index",
}
numeric_columns = [
    column for column in features.columns
    if column not in metadata_columns and pd.api.types.is_numeric_dtype(features[column])
]
gr_prefixes = (
    "raw_", "s32_", "s64_", "s128_", "s256_", "s512_", "s1024_",
    "segment_", "d64_", "d256_", "d512_", "rank_raw_", "rank_s32_",
    "rank_s64_", "rank_s128_", "rank_s256_", "rank_s512_", "rank_s1024_",
    "rank_segment_", "rank_d64_", "rank_d256_", "rank_d512_",
)
gr_columns = [column for column in numeric_columns if column.startswith(gr_prefixes)]
context_columns = [
    column for column in (
        "known_frac", "gr_missing_frac", "eval_span_kft", "calibration_slope",
        "calibration_intercept_scaled",
    ) if column in numeric_columns
]
shape_columns = [
    column for column in numeric_columns
    if column.startswith("correction_") or column.startswith("family_")
]
variant_columns = {
    "gr_only": sorted(set(gr_columns + context_columns)),
    "gr_plus_shape_no_identity": sorted(set(gr_columns + context_columns + shape_columns)),
    "shape_only": sorted(set(shape_columns + context_columns)),
}

train_group = train.groupby("well", sort=False).size().to_list()
valid_group = valid.groupby("well", sort=False).size().to_list()
reports = []
selected_parts = []
model_rows = []

# Oracle and zero establish the exact ceiling and base in the same evaluator.
for name, indices in (
    ("oracle", valid.groupby("well")["candidate_rmse"].idxmin()),
    ("zero", valid[valid["candidate_name"] == "zero"].index),
):
    report, selected = _selector_report(name, valid, indices, well_baseline)
    reports.append(report)
    selected_parts.append(selected.assign(selector=name))

# Scan every individual path-level GR statistic before fitting another model.
direct_candidates = []
for column in gr_columns:
    if column.endswith("_coverage"):
        continue
    if column.startswith("rank_"):
        indices = _safe_group_extreme(valid, column, maximize=False)
    elif column.endswith("_bias"):
        temporary = f"__abs_{column}"
        valid[temporary] = valid[column].abs()
        indices = _safe_group_extreme(valid, temporary, maximize=False)
    elif column.endswith("_ncc"):
        indices = _safe_group_extreme(valid, column, maximize=True)
    else:
        indices = _safe_group_extreme(valid, column, maximize=False)
    report, _ = _selector_report(f"direct:{column}", valid, indices, well_baseline)
    direct_candidates.append(report)
direct_scores = pd.DataFrame(direct_candidates).sort_values("rmse")
direct_scores.to_csv(OUT_DIR / "direct_gr_score_scan.csv", index=False)
if len(direct_scores):
    best_direct_name = str(direct_scores.iloc[0]["selector"])
    reports.append(direct_scores.iloc[0].to_dict())

for variant, columns in variant_columns.items():
    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=1800,
        learning_rate=0.025, num_leaves=31, min_child_samples=40,
        colsample_bytree=0.85, reg_lambda=2.0, random_state=20260716,
        n_jobs=-1, verbosity=-1,
    )
    ranker.fit(
        train[columns], train["relevance"].astype(int), group=train_group,
        eval_set=[(valid[columns], valid["relevance"].astype(int))],
        eval_group=[valid_group], eval_at=[1, 3, 5, 10],
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )
    score_column = f"score_rank_{variant}"
    valid[score_column] = ranker.predict(valid[columns], num_iteration=ranker.best_iteration_)
    indices = valid.groupby("well")[score_column].idxmax()
    name = f"rank:{variant}"
    report, selected = _selector_report(name, valid, indices, well_baseline)
    reports.append(report)
    selected_parts.append(selected.assign(selector=name))
    joblib.dump(
        {"model": ranker, "features": columns, "variant": variant},
        OUT_DIR / f"{variant}_ranker.joblib", compress=3,
    )
    model_rows.append(
        {"model": name, "features": len(columns), "best_iteration": int(ranker.best_iteration_)}
    )

    # Regression optimises candidate RMSE directly and upweights the useful
    # end of each candidate list rather than treating all 235 paths equally.
    train_rank = train.groupby("well")["candidate_rmse"].rank(method="first") - 1.0
    weights = 0.2 + 4.0 * np.exp(-train_rank.to_numpy(float) / 24.0)
    regressor = lgb.LGBMRegressor(
        objective="huber", n_estimators=2200, learning_rate=0.025,
        num_leaves=31, min_child_samples=40, colsample_bytree=0.85,
        reg_lambda=2.0, random_state=20260717, n_jobs=-1, verbosity=-1,
    )
    regressor.fit(
        train[columns], train["candidate_rmse"], sample_weight=weights,
        eval_set=[(valid[columns], valid["candidate_rmse"])],
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )
    score_column = f"score_reg_{variant}"
    valid[score_column] = regressor.predict(valid[columns], num_iteration=regressor.best_iteration_)
    indices = valid.groupby("well")[score_column].idxmin()
    name = f"reg:{variant}"
    report, selected = _selector_report(name, valid, indices, well_baseline)
    reports.append(report)
    selected_parts.append(selected.assign(selector=name))
    joblib.dump(
        {"model": regressor, "features": columns, "variant": variant},
        OUT_DIR / f"{variant}_regressor.joblib", compress=3,
    )
    model_rows.append(
        {"model": name, "features": len(columns), "best_iteration": int(regressor.best_iteration_)}
    )

reports = pd.DataFrame(reports).sort_values("rmse").reset_index(drop=True)
models = pd.DataFrame(model_rows)
selected_report = pd.concat(selected_parts, ignore_index=True)
reports.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
models.to_csv(OUT_DIR / "model_summary.csv", index=False)
selected_report.to_csv(OUT_DIR / "selected_bywell.csv", index=False)
valid.to_pickle(OUT_DIR / "valid_rescored_features.pkl")

deployable = reports[~reports["selector"].isin(["oracle"])]
best = deployable.iloc[0].to_dict()
strict_pass = bool(
    best["gain_vs_strong"] >= PASS_GAIN
    and best["positive_well_fraction"] >= PASS_POSITIVE_WELLS
    and best["mean_regret"] <= PASS_MEAN_REGRET
)
verdict = {
    "experiment": "gr_path_matcher_ablation_rescore_v0",
    "source": str(SOURCE_DIR),
    "zero_reproduction_max_rmse_difference": zero_difference,
    "gr_features": len(gr_columns),
    "shape_features": len(shape_columns),
    "best": best,
    "gates": {
        "gain_vs_strong_min": PASS_GAIN,
        "positive_well_fraction_min": PASS_POSITIVE_WELLS,
        "mean_regret_max": PASS_MEAN_REGRET,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> expand the winning identity-free objective to nested pseudo-PS folds"
        if strict_pass
        else "FAIL -> close summary-feature whole-path GR ranking; learned sequence scoring would be a new route"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nABLATIONS")
print(reports.to_string(index=False))
print("\nTOP DIRECT GR SCORES")
print(direct_scores.head(30).to_string(index=False))
print("\nMODELS")
print(models.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
