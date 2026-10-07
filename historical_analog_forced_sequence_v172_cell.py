"""V172: forced candidate-conditioned GR sequence ranker.

Run after V167 in the same kernel. V167b proved that the combined ranker used
zero sequence features, so this gate removes all metadata columns and asks the
only remaining high-value question: can candidate-conditioned future GR alone
narrow the V166 analog oracle to a useful shortlist?

This is a sampled-row observability gate. It does not tune a deployment dose.
"""

from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED_V172 = time.time()
ROOT_V172 = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OUT_V172 = ROOT_V172 / "oof_exports" / "historical_analog_forced_sequence_v172"
OUT_V172.mkdir(parents=True, exist_ok=True)

required_v172 = (
    "candidate_v167", "FEATURE_NAMES_V167", "RELEVANCE_LEVELS",
    "sample_v167", "records", "_transfer", "_metric",
)
missing_v172 = [name for name in required_v172 if name not in globals()]
if missing_v172:
    raise RuntimeError("Run V166 and V167 first. Missing: " + repr(missing_v172))

SEQUENCE_FEATURES_V172 = len(FEATURE_NAMES_V167) - 12
SEED_V172 = 20260805
if SEQUENCE_FEATURES_V172 <= 0:
    raise RuntimeError("Invalid V167 feature partition")


def _clean_v172(value):
    if isinstance(value, dict):
        return {str(key): _clean_v172(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v172(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _ordered_v172(frame, column):
    ordered = frame.sort_values(["well", "reference_well"], kind="stable")
    matrix = np.stack(ordered[column].to_numpy()).astype(np.float32)[:, :SEQUENCE_FEATURES_V172]
    groups = ordered.groupby("well", sort=False).size().to_numpy(np.int32)
    return ordered, matrix, groups


params_v172 = dict(
    objective="lambdarank",
    metric="ndcg",
    label_gain=list(range(RELEVANCE_LEVELS + 1)),
    n_estimators=1400,
    learning_rate=0.025,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.90,
    reg_alpha=1.0,
    reg_lambda=16.0,
    max_bin=127,
    n_jobs=-1,
    verbosity=-1,
)

score_v172 = np.full(len(candidate_v167), np.nan, np.float64)
rolled_score_v172 = np.full(len(candidate_v167), np.nan, np.float64)
model_rows_v172 = []
for held_fold_v172 in range(1, 6):
    train_v172 = candidate_v167[candidate_v167["fold"] != held_fold_v172]
    valid_v172 = candidate_v167[candidate_v167["fold"] == held_fold_v172]
    train_ordered_v172, train_x_v172, train_groups_v172 = _ordered_v172(train_v172, "features")
    valid_ordered_v172, valid_x_v172, valid_groups_v172 = _ordered_v172(valid_v172, "features")
    _, rolled_x_v172, _ = _ordered_v172(valid_v172, "rolled_features")
    model_v172 = lgb.LGBMRanker(**params_v172, random_state=SEED_V172 + held_fold_v172)
    model_v172.fit(
        train_x_v172,
        train_ordered_v172["relevance"].to_numpy(np.int32),
        group=train_groups_v172,
        eval_set=[(valid_x_v172, valid_ordered_v172["relevance"].to_numpy(np.int32))],
        eval_group=[valid_groups_v172],
        eval_at=(1, 4, 8, 16),
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    score_v172[valid_ordered_v172.index] = model_v172.predict(
        valid_x_v172, num_iteration=model_v172.best_iteration_
    )
    rolled_score_v172[valid_ordered_v172.index] = model_v172.predict(
        rolled_x_v172, num_iteration=model_v172.best_iteration_
    )
    model_v172.booster_.save_model(str(OUT_V172 / f"ranker_fold{held_fold_v172}.txt"))
    model_rows_v172.append({
        "fold": held_fold_v172,
        "train_wells": int(train_v172["well"].nunique()),
        "valid_wells": int(valid_v172["well"].nunique()),
        "best_iteration": int(model_v172.best_iteration_),
        "used_features": int(np.sum(model_v172.booster_.feature_importance("split") > 0)),
    })
    print("V172 MODEL", model_rows_v172[-1], flush=True)
    del train_x_v172, valid_x_v172, rolled_x_v172, model_v172
    gc.collect()

if not np.isfinite(score_v172).all() or not np.isfinite(rolled_score_v172).all():
    raise RuntimeError("Incomplete V172 scores")

panel_v172 = candidate_v167.copy()
panel_v172["sequence_score"] = score_v172
panel_v172["rolled_sequence_score"] = rolled_score_v172


def _reference_lookup_v172(score_column, oracle_k=None):
    ranked_v172 = panel_v172.sort_values(
        ["well", score_column], ascending=[True, False], kind="stable"
    )
    if oracle_k is None:
        selected_v172 = ranked_v172.groupby("well", sort=False).head(1)
    else:
        shortlisted_v172 = ranked_v172.groupby("well", sort=False).head(int(oracle_k))
        indices_v172 = shortlisted_v172.groupby("well", sort=False)["sse"].idxmin()
        selected_v172 = shortlisted_v172.loc[indices_v172]
    return selected_v172.set_index("well")["reference_well"].astype(str).to_dict()


def _expand_v172(lookup_v172):
    parts_v172 = []
    aligned_frame_v172 = []
    for well_v172, group_v172 in sample_v167.groupby("well", sort=False):
        target_v172 = records[str(well_v172)]
        reference_v172 = records[str(lookup_v172[str(well_v172)])]
        row_idx_v172 = group_v172["row_idx"].to_numpy(np.int64)
        query_v172 = target_v172["md"][row_idx_v172] - target_v172["md"][target_v172["anchor"]]
        prediction_v172, _ = _transfer(
            reference_v172, target_v172, query_v172, target_v172["z"][row_idx_v172]
        )
        parts_v172.append(np.asarray(prediction_v172, np.float64))
        aligned_frame_v172.append(group_v172)
    return pd.concat(aligned_frame_v172, ignore_index=True), np.concatenate(parts_v172)


frame_v172, selected_v172 = _expand_v172(_reference_lookup_v172("sequence_score"))
_, rolled_v172 = _expand_v172(_reference_lookup_v172("rolled_sequence_score"))
_, oracle4_v172 = _expand_v172(_reference_lookup_v172("sequence_score", 4))
_, oracle8_v172 = _expand_v172(_reference_lookup_v172("sequence_score", 8))
_, oracle16_v172 = _expand_v172(_reference_lookup_v172("sequence_score", 16))
base_v172 = frame_v172["base"].to_numpy(np.float64)

summary_v172 = pd.DataFrame([
    _metric(frame_v172, base_v172, "base_current"),
    _metric(frame_v172, selected_v172, "sequence_top1"),
    _metric(frame_v172, rolled_v172, "rolled_sequence_top1"),
    _metric(frame_v172, oracle4_v172, "sequence_oracle_top4"),
    _metric(frame_v172, oracle8_v172, "sequence_oracle_top8"),
    _metric(frame_v172, oracle16_v172, "sequence_oracle_top16"),
])
indexed_v172 = summary_v172.set_index("endpoint")
choice_v172 = _reference_lookup_v172("sequence_score")
rolled_choice_v172 = _reference_lookup_v172("rolled_sequence_score")
choice_change_v172 = float(np.mean([
    choice_v172[well_v172] != rolled_choice_v172[well_v172]
    for well_v172 in choice_v172
]))

checks_v172 = {
    "sequence_oracle_top8_rmse_le_7": bool(indexed_v172.loc["sequence_oracle_top8", "rmse"] <= 7.0),
    "sequence_oracle_top16_rmse_le_6p5": bool(indexed_v172.loc["sequence_oracle_top16", "rmse"] <= 6.5),
    "aligned_top1_beats_rolled_ge_0p20": bool(
        indexed_v172.loc["rolled_sequence_top1", "rmse"]
        - indexed_v172.loc["sequence_top1", "rmse"] >= 0.20
    ),
    "rolled_choice_change_fraction_ge_0p10": bool(choice_change_v172 >= 0.10),
}
verdict_v172 = {
    "experiment": "historical_analog_forced_sequence_v172",
    "stage": "fivefold_forced_candidate_conditioned_GR_sequence_gateway",
    "candidate_menu": "frozen V166/V167 outer-fold-excluded union",
    "metadata_features": 0,
    "sequence_features": SEQUENCE_FEATURES_V172,
    "coverage": {
        "rows": len(frame_v172), "wells": frame_v172["well"].nunique(),
        "candidate_rows": len(panel_v172),
    },
    "rolled_choice_change_fraction": choice_change_v172,
    "checks": checks_v172,
    "strict_pass": bool(all(checks_v172.values())),
    "elapsed_minutes": (time.time() - STARTED_V172) / 60.0,
    "next": (
        "PASS -> train a segmental sequence decoder on the learned shortlist"
        if all(checks_v172.values())
        else "FAIL -> close analog retrieval; rebuild the PF/DP trajectory state space"
    ),
}

summary_v172.to_csv(OUT_V172 / "summary.csv", index=False)
pd.DataFrame(model_rows_v172).to_csv(OUT_V172 / "models.csv", index=False)
panel_v172.drop(columns=["features", "rolled_features"]).to_parquet(
    OUT_V172 / "candidate_scores.parquet", index=False, compression="zstd"
)
(OUT_V172 / "verdict.json").write_text(
    json.dumps(_clean_v172(verdict_v172), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nV172 SUMMARY")
print(summary_v172.to_string(index=False), flush=True)
print("\nV172 FINAL VERDICT")
print(json.dumps(_clean_v172(verdict_v172), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V172, flush=True)
