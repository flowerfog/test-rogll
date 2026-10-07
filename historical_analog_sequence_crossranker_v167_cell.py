"""V167: candidate-conditioned GR sequence cross-ranker.

Run this cell immediately after V166 in the same kernel. V166's scalar
four-scale emission score showed aligned information, but discarded where in
the sequence each mismatch occurred. This gateway keeps the frozen V166
outer-fold-excluded analog menu and trains a grouped ranker from binned,
multi-scale candidate-conditioned GR residual sequences.

Held-out target wells never contribute labels to their ranker. Reference wells
from the target canonical fold are excluded by the frozen V166 protocol.
"""

from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED_V167 = time.time()
EXPERIMENT_V167 = "historical_analog_sequence_crossranker_v167"
OUT_DIR_V167 = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT_V167
OUT_DIR_V167.mkdir(parents=True, exist_ok=True)

REQUIRED_V166 = (
    "records", "sample", "wells", "well_index", "fold_array", "arrays",
    "FAMILIES", "_family_scores", "_transfer", "_smooth",
    "_robust_standardise", "_metric",
)
missing_v166 = [name for name in REQUIRED_V166 if name not in globals()]
if missing_v166:
    raise RuntimeError(
        "Run the complete V166 cell first in this kernel. Missing: "
        + repr(missing_v166)
    )

SEED_V167 = 20260803
MENU_PER_FAMILY = 64
MAX_CANDIDATES_PER_WELL = 256
RESIDUAL_BINS = 16
WIDTHS = (1, 5, 15, 31)
LAGS = (-16, -8, -4, 0, 4, 8, 16)
RELEVANCE_LEVELS = 31
BLEND_WEIGHTS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)


def _json_default_v167(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _corr_v167(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 8:
        return 0.0
    left = left[valid] - np.mean(left[valid])
    right = right[valid] - np.mean(right[valid])
    scale = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / scale) if scale > 1e-10 else 0.0


def _lag_corr_v167(left, right, lag):
    if lag < 0:
        return _corr_v167(left[-lag:], right[:lag])
    if lag > 0:
        return _corr_v167(left[:-lag], right[lag:])
    return _corr_v167(left, right)


def _bin_stats_v167(values, bins=RESIDUAL_BINS):
    values = np.asarray(values, np.float64)
    edges = np.linspace(0, len(values), bins + 1).round().astype(np.int64)
    output = []
    for start, stop in zip(edges[:-1], edges[1:]):
        local = values[start:max(stop, start + 1)]
        finite = local[np.isfinite(local)]
        if len(finite) == 0:
            output.extend((0.0, 0.0, 0.0))
        else:
            output.extend((
                float(np.mean(finite)),
                float(np.mean(np.abs(finite))),
                float(np.sqrt(np.mean(finite * finite))),
            ))
    return output


def _sequence_features_v167(target, row_idx, prediction, roll_fraction=0.0):
    observed = target["gr"][row_idx].astype(np.float64)
    if roll_fraction:
        observed = np.roll(
            observed, max(1, int(round(len(observed) * roll_fraction)))
        )
    expected = np.interp(
        prediction,
        target["typewell_tvt"],
        target["typewell_gr"],
        left=target["typewell_gr"][0],
        right=target["typewell_gr"][-1],
    )
    output = []
    for width in WIDTHS:
        left = _robust_standardise(_smooth(observed, width))
        right = _robust_standardise(_smooth(expected, width))
        residual = np.clip(left - right, -4.0, 4.0)
        output.extend(_bin_stats_v167(residual))
        output.extend([
            _corr_v167(left, right),
            _corr_v167(np.gradient(left), np.gradient(right)),
            float(np.mean(residual)),
            float(np.mean(np.abs(residual))),
            float(np.sqrt(np.mean(residual * residual))),
            float(np.quantile(np.abs(residual), 0.90)),
        ])
        output.extend(_lag_corr_v167(left, right, lag) for lag in LAGS)
    return np.asarray(output, np.float32)


def _feature_names_v167():
    names = []
    for width in WIDTHS:
        for index in range(RESIDUAL_BINS):
            names.extend([
                f"w{width}_b{index}_mean",
                f"w{width}_b{index}_absmean",
                f"w{width}_b{index}_rms",
            ])
        names.extend([
            f"w{width}_corr", f"w{width}_gradient_corr",
            f"w{width}_global_mean", f"w{width}_global_absmean",
            f"w{width}_global_rms", f"w{width}_global_p90abs",
        ])
        names.extend(f"w{width}_lag{lag}" for lag in LAGS)
    names.extend([
        "anchor_distance_kft", "direction_dot", "length_ratio",
        "reference_slope128", "reference_slope256", "reference_slope512",
        "target_slope128", "target_slope256", "target_slope512",
        "best_retrieval_rank", "mean_retrieval_rank", "family_support",
    ])
    return names


FEATURE_NAMES_V167 = _feature_names_v167()


def _candidate_menu_v167(target, target_index):
    eligible = np.ones(len(wells), dtype=bool)
    eligible[target_index] = False
    eligible &= fold_array != target["fold"]
    scores = _family_scores(target, eligible, arrays)
    references = {}
    for family in FAMILIES:
        ranking = np.argsort(scores[family], kind="stable")[:MENU_PER_FAMILY]
        for rank, reference_index in enumerate(ranking, 1):
            reference_index = int(reference_index)
            if not eligible[reference_index]:
                continue
            payload = references.setdefault(reference_index, {"ranks": []})
            payload["ranks"].append(rank)
    ordered = sorted(
        references.items(),
        key=lambda item: (
            min(item[1]["ranks"]),
            -len(item[1]["ranks"]),
            float(np.mean(item[1]["ranks"])),
            item[0],
        ),
    )
    return dict(ordered[:MAX_CANDIDATES_PER_WELL])


candidate_rows_v167 = []
failures_v167 = []
sample_v167 = sample[sample["well"].isin(records)].copy()
group_count_v167 = sample_v167["well"].nunique()
for number, (well, group) in enumerate(sample_v167.groupby("well", sort=False), 1):
    try:
        target = records[well]
        target_index = well_index[well]
        row_idx = group["row_idx"].to_numpy(np.int64)
        query_distance = target["md"][row_idx] - target["md"][target["anchor"]]
        target_z = target["z"][row_idx]
        truth = group["tvt_true"].to_numpy(np.float64)
        menu = _candidate_menu_v167(target, target_index)
        if len(menu) < 64:
            raise RuntimeError(f"only {len(menu)} unique candidates")
        local_rows = []
        for reference_index, payload in menu.items():
            reference_well = wells[reference_index]
            prediction, coverage = _transfer(
                records[reference_well], target, query_distance, target_z
            )
            aligned = _sequence_features_v167(target, row_idx, prediction)
            rolled = _sequence_features_v167(target, row_idx, prediction, 0.20)
            anchor_distance = float(np.linalg.norm(
                target["anchor_xy"] - records[reference_well]["anchor_xy"]
            )) / 1000.0
            direction_dot = float(np.dot(
                target["direction"], records[reference_well]["direction"]
            ))
            length_ratio = float(
                records[reference_well]["length"] / max(target["length"], 1.0)
            )
            metadata = np.asarray([
                anchor_distance, direction_dot, length_ratio,
                *records[reference_well]["slopes"], *target["slopes"],
                float(min(payload["ranks"])), float(np.mean(payload["ranks"])),
                float(len(payload["ranks"])),
            ], np.float32)
            error = prediction - truth
            local_rows.append({
                "well": well,
                "fold": int(target["fold"]),
                "reference_well": reference_well,
                "coverage": float(coverage),
                "sse": float(np.dot(error, error)),
                "path_rmse": float(np.sqrt(np.mean(error * error))),
                "features": np.concatenate([aligned, metadata]),
                "rolled_features": np.concatenate([rolled, metadata]),
            })
        ordered = sorted(local_rows, key=lambda row: row["sse"])
        denominator = max(len(ordered) - 1, 1)
        for rank, row in enumerate(ordered):
            row["relevance"] = int(round(
                RELEVANCE_LEVELS * (1.0 - rank / denominator)
            ))
            candidate_rows_v167.append(row)
    except Exception as exc:
        failures_v167.append({"well": well, "reason": repr(exc)})
    if number % 25 == 0 or number == group_count_v167:
        print(
            f"V167 features {number}/{group_count_v167} "
            f"candidates={len(candidate_rows_v167)} failures={len(failures_v167)}",
            flush=True,
        )

pd.DataFrame(failures_v167).to_csv(OUT_DIR_V167 / "failures.csv", index=False)
candidate_v167 = pd.DataFrame(candidate_rows_v167)
if candidate_v167["well"].nunique() < 760:
    raise RuntimeError(
        f"Only {candidate_v167['well'].nunique()} candidate-complete wells: "
        f"{failures_v167[:5]}"
    )


def _ordered_matrix_v167(frame, column):
    ordered = frame.sort_values(["well", "reference_well"], kind="stable")
    matrix = np.stack(ordered[column].to_numpy()).astype(np.float32)
    groups = ordered.groupby("well", sort=False).size().to_numpy(np.int32)
    return ordered, matrix, groups


params_v167 = dict(
    objective="lambdarank",
    metric="ndcg",
    # LightGBM's default table has only 31 entries (labels 0..30), while
    # V167 deliberately uses 32 relevance levels (0..31).
    label_gain=list(range(RELEVANCE_LEVELS + 1)),
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=31,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.5,
    reg_lambda=12.0,
    max_bin=127,
    n_jobs=-1,
    verbosity=-1,
)

score_v167 = np.full(len(candidate_v167), np.nan, np.float64)
rolled_score_v167 = np.full(len(candidate_v167), np.nan, np.float64)
model_rows_v167 = []
for held_fold in range(1, 6):
    train = candidate_v167[candidate_v167["fold"] != held_fold]
    valid = candidate_v167[candidate_v167["fold"] == held_fold]
    train_ordered, train_x, train_groups = _ordered_matrix_v167(train, "features")
    valid_ordered, valid_x, valid_groups = _ordered_matrix_v167(valid, "features")
    _, valid_rolled_x, _ = _ordered_matrix_v167(valid, "rolled_features")
    model = lgb.LGBMRanker(**params_v167, random_state=SEED_V167 + held_fold)
    model.fit(
        train_x,
        train_ordered["relevance"].to_numpy(np.int32),
        group=train_groups,
        eval_set=[(valid_x, valid_ordered["relevance"].to_numpy(np.int32))],
        eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    score_v167[valid_ordered.index] = model.predict(
        valid_x, num_iteration=model.best_iteration_
    )
    rolled_score_v167[valid_ordered.index] = model.predict(
        valid_rolled_x, num_iteration=model.best_iteration_
    )
    model.booster_.save_model(str(OUT_DIR_V167 / f"ranker_fold{held_fold}.txt"))
    model_rows_v167.append({
        "fold": held_fold,
        "train_wells": int(train["well"].nunique()),
        "valid_wells": int(valid["well"].nunique()),
        "train_candidates": int(len(train)),
        "valid_candidates": int(len(valid)),
        "best_iteration": int(model.best_iteration_),
    })
    print("V167 model", model_rows_v167[-1], flush=True)
    del train_x, valid_x, valid_rolled_x, model
    gc.collect()

if not np.isfinite(score_v167).all() or not np.isfinite(rolled_score_v167).all():
    raise RuntimeError("Incomplete V167 cross-fitted scores")
candidate_v167["score"] = score_v167
candidate_v167["rolled_score"] = rolled_score_v167


def _selected_reference_v167(score_column):
    indices = candidate_v167.groupby("well", sort=False)[score_column].idxmax()
    return candidate_v167.loc[indices, ["well", "reference_well"]].set_index("well")["reference_well"].to_dict()


def _topk_oracle_reference_v167(k):
    ranked = candidate_v167.sort_values(
        ["well", "score"], ascending=[True, False], kind="stable"
    ).groupby("well", sort=False).head(k)
    indices = ranked.groupby("well", sort=False)["sse"].idxmin()
    return ranked.loc[indices, ["well", "reference_well"]].set_index("well")["reference_well"].to_dict()


def _expand_references_v167(reference_lookup):
    parts = []
    for well, group in sample_v167.groupby("well", sort=False):
        target = records[well]
        reference = records[reference_lookup[well]]
        row_idx = group["row_idx"].to_numpy(np.int64)
        query_distance = target["md"][row_idx] - target["md"][target["anchor"]]
        prediction, _ = _transfer(reference, target, query_distance, target["z"][row_idx])
        parts.append(np.asarray(prediction, np.float64))
    return np.concatenate(parts)


selected_v167 = _expand_references_v167(_selected_reference_v167("score"))
rolled_v167 = _expand_references_v167(_selected_reference_v167("rolled_score"))
oracle8_v167 = _expand_references_v167(_topk_oracle_reference_v167(8))
frame_v167 = pd.concat(
    [group for _, group in sample_v167.groupby("well", sort=False)],
    ignore_index=True,
)
base_v167 = frame_v167["base"].to_numpy(np.float64)

# Select the blend dose for every held fold using only the other folds'
# cross-fitted predictions.
nested_v167 = np.empty(len(frame_v167), np.float64)
fold_rows_v167 = []
for held_fold in range(1, 6):
    held = frame_v167["fold"].to_numpy(int) == held_fold
    train = ~held
    train_scores = []
    for weight in BLEND_WEIGHTS:
        prediction = base_v167[train] + weight * (selected_v167[train] - base_v167[train])
        error = prediction - frame_v167.loc[train, "tvt_true"].to_numpy(np.float64)
        train_scores.append((float(np.sqrt(np.mean(error * error))), weight))
    _, selected_weight = min(train_scores)
    nested_v167[held] = base_v167[held] + selected_weight * (
        selected_v167[held] - base_v167[held]
    )
    held_frame = frame_v167.loc[held].reset_index(drop=True)
    base_metric = _metric(held_frame, base_v167[held], "base")
    candidate_metric = _metric(held_frame, nested_v167[held], "nested")
    fold_rows_v167.append({
        "fold": held_fold,
        "selected_weight": float(selected_weight),
        "base_rmse": base_metric["rmse"],
        "rmse": candidate_metric["rmse"],
        "gain": base_metric["rmse"] - candidate_metric["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"],
    })

base_metric_v167 = _metric(frame_v167, base_v167, "base_current")
selected_metric_v167 = _metric(frame_v167, selected_v167, "crossrank_top1")
rolled_metric_v167 = _metric(frame_v167, rolled_v167, "rolled_feature_control")
oracle8_metric_v167 = _metric(frame_v167, oracle8_v167, "crossrank_oracle_top8")
nested_metric_v167 = _metric(frame_v167, nested_v167, "nested_blend")
fold_report_v167 = pd.DataFrame(fold_rows_v167)

rank_associations_v167 = []
for well, local in candidate_v167.groupby("well", sort=False):
    rank_associations_v167.append(
        local[["score", "path_rmse"]].corr(method="spearman").iloc[0, 1]
    )
median_rank_rho_v167 = float(np.nanmedian(rank_associations_v167))

checks_v167 = {
    "crossrank_top8_oracle_rmse_le_7": bool(oracle8_metric_v167["rmse"] <= 7.0),
    "nested_gain_ge_0p05": bool(base_metric_v167["rmse"] - nested_metric_v167["rmse"] >= 0.05),
    "positive_folds_ge_4": bool((fold_report_v167["gain"] > 0).sum() >= 4),
    "minimum_fold_gain_ge_minus0p03": bool(fold_report_v167["gain"].min() >= -0.03),
    "aligned_top1_beats_rolled_ge_0p20": bool(rolled_metric_v167["rmse"] - selected_metric_v167["rmse"] >= 0.20),
    "median_rank_spearman_le_minus0p15": bool(median_rank_rho_v167 <= -0.15),
}
strict_pass_v167 = bool(all(checks_v167.values()))
summary_v167 = pd.DataFrame([
    base_metric_v167, selected_metric_v167, rolled_metric_v167,
    oracle8_metric_v167, nested_metric_v167,
])
verdict_v167 = {
    "experiment": EXPERIMENT_V167,
    "stage": "fivefold_candidate_conditioned_sequence_crossranking",
    "sample_unit": "complete physical target well",
    "base": "V144 sampled current endpoint",
    "candidate_menu": "V166 outer-fold-excluded top64 per family, deduplicated and capped at 256",
    "features": len(FEATURE_NAMES_V167),
    "coverage": {
        "rows": int(len(frame_v167)),
        "wells": int(frame_v167["well"].nunique()),
        "candidate_rows": int(len(candidate_v167)),
        "failures": int(len(failures_v167)),
    },
    "base_rmse": base_metric_v167["rmse"],
    "selected_top1_rmse": selected_metric_v167["rmse"],
    "learned_top8_oracle_rmse": oracle8_metric_v167["rmse"],
    "nested_blend_rmse": nested_metric_v167["rmse"],
    "nested_gain": base_metric_v167["rmse"] - nested_metric_v167["rmse"],
    "rolled_top1_loss": rolled_metric_v167["rmse"] - selected_metric_v167["rmse"],
    "median_within_well_score_pathrmse_spearman": median_rank_rho_v167,
    "folds": fold_rows_v167,
    "checks": checks_v167,
    "strict_pass": strict_pass_v167,
    "elapsed_minutes": (time.time() - STARTED_V167) / 60.0,
    "next": (
        "PASS -> train the same cross-ranker on all wells and package candidate generation"
        if strict_pass_v167
        else "FAIL -> complete candidate-conditioned GR sequences still cannot identify the analog; close analog retrieval"
    ),
}

summary_v167.to_csv(OUT_DIR_V167 / "summary.csv", index=False)
fold_report_v167.to_csv(OUT_DIR_V167 / "folds.csv", index=False)
pd.DataFrame(model_rows_v167).to_csv(OUT_DIR_V167 / "models.csv", index=False)
pd.DataFrame({"feature": FEATURE_NAMES_V167}).to_csv(
    OUT_DIR_V167 / "feature_names.csv", index=False
)
candidate_v167.drop(columns=["features", "rolled_features"]).to_parquet(
    OUT_DIR_V167 / "candidate_scores.parquet", index=False, compression="zstd"
)
(OUT_DIR_V167 / "verdict.json").write_text(
    json.dumps(verdict_v167, indent=2, default=_json_default_v167),
    encoding="utf-8",
)

print("\nV167 MATCHED SUMMARY")
print(summary_v167.to_string(index=False), flush=True)
print("\nV167 FOLDS")
print(fold_report_v167.to_string(index=False), flush=True)
print("\nV167 FINAL VERDICT")
print(json.dumps(verdict_v167, indent=2, default=_json_default_v167), flush=True)
print("saved:", OUT_DIR_V167, flush=True)
