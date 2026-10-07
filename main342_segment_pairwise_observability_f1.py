"""M342-F1: real-candidate 128-ft segment pairwise observability gateway.

Rebuild the frozen V188 legal block feature panel, but replace absolute-RMSE
regression with symmetric hard-pair classification. Each training block uses
its real oracle endpoint as the positive and the closest legal-feature
alternatives with at least 2 ft RMSE regret as negatives. Fivefold complete-
well crossfit and an in-block feature-alignment shuffle test observability.

This is an attribution gateway, not a deployable decoder. Truth is used only
for training labels and final held-fold metrics.
"""

from pathlib import Path
import json
import sys
import time
import types

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "canonical_block_engine_selector_v188.py"
OUT = ROOT / "oof_exports" / "main342_segment_pairwise_observability_f1"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260804
MIN_REGRET_FT = 2.0
MAX_REGRET_FT = 40.0
NEGATIVES_PER_BLOCK = 4


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_v188_panel():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = "score = {name: np.full"
    if marker not in source:
        raise RuntimeError("V188 model-training marker not found")
    module_name = "m342_f1_v188_panel"
    module = types.ModuleType(module_name)
    module.__file__ = str(SOURCE)
    sys.modules[module_name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    required = (
        "base", "paths", "truth", "names", "base_index", "k", "x", "panel",
        "y", "sse", "folds", "well_slices", "feature_names", "_metrics",
    )
    missing = [name for name in required if name not in module.__dict__]
    if missing:
        raise RuntimeError("Incomplete V188 panel definitions: " + repr(missing))
    return module.__dict__


def _group_rows(panel):
    groups = panel["group"].to_numpy(np.int64)
    unique, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise RuntimeError("V188 groups are not contiguous")
    return starts.astype(np.int64), counts.astype(np.int64)


def _training_pairs(x, y, panel, train_mask, starts, counts):
    train_rows = np.flatnonzero(train_mask)
    centre = x.iloc[train_rows].median(axis=0).to_numpy(np.float64)
    scale = x.iloc[train_rows].std(axis=0).to_numpy(np.float64)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    values = x.to_numpy(np.float64)
    pair_left, pair_right, pair_group = [], [], []
    for group, (start, count) in enumerate(zip(starts, counts)):
        indices = np.arange(start, start + count)
        if not train_mask[start]:
            continue
        local_y = y[indices]
        positive = int(indices[np.argmin(local_y)])
        regret = local_y - y[positive]
        eligible = indices[(regret >= MIN_REGRET_FT) & (regret <= MAX_REGRET_FT)]
        if not len(eligible):
            continue
        delta = (values[eligible] - values[positive]) / scale
        distance = np.sqrt(np.mean(np.square(delta), axis=1))
        order = np.argsort(distance, kind="stable")[:NEGATIVES_PER_BLOCK]
        for negative in eligible[order]:
            pair_left.append(positive)
            pair_right.append(int(negative))
            pair_group.append(group)
    if not pair_left:
        raise RuntimeError("No hard pairs constructed")
    left = np.asarray(pair_left, np.int64)
    right = np.asarray(pair_right, np.int64)
    group = np.asarray(pair_group, np.int64)
    delta = (values[left] - values[right]) / scale
    matrix = np.concatenate([delta, -delta], axis=0).astype(np.float32)
    target = np.concatenate([
        np.ones(len(delta), np.uint8), np.zeros(len(delta), np.uint8),
    ])
    # Equal total weight per physical block, then duplicate symmetrically.
    _, inverse, per_group = np.unique(group, return_inverse=True, return_counts=True)
    weight = 1.0 / per_group[inverse]
    weight = np.concatenate([weight, weight]).astype(np.float32)
    return matrix, target, weight, scale, len(np.unique(group)), len(left)


def _tournament(model, x_values, scale, starts, counts, valid_group_mask, shuffle=False, seed=0):
    choices = np.full(len(starts), -1, np.int16)
    margins = np.full(len(starts), np.nan, np.float32)
    rng = np.random.default_rng(seed)
    pair_targets, pair_probabilities = [], []
    for group, (start, count) in enumerate(zip(starts, counts)):
        if not valid_group_mask[group]:
            continue
        indices = np.arange(start, start + count)
        features = x_values[indices].copy()
        if shuffle:
            features = features[rng.permutation(count)]
        left, right = np.triu_indices(count, k=1)
        delta = (features[left] - features[right]) / scale
        probability = model.predict_proba(delta.astype(np.float32))[:, 1]
        score = np.zeros(count, np.float64)
        games = np.zeros(count, np.float64)
        np.add.at(score, left, probability)
        np.add.at(score, right, 1.0 - probability)
        np.add.at(games, left, 1.0)
        np.add.at(games, right, 1.0)
        score /= np.maximum(games, 1.0)
        order = np.argsort(-score, kind="stable")
        choices[group] = int(order[0])
        margins[group] = float(score[order[0]] - score[order[1]]) if count > 1 else 0.0
        if not shuffle:
            local_y = y[indices]
            pair_targets.extend((local_y[left] < local_y[right]).astype(np.uint8).tolist())
            pair_probabilities.extend(probability.tolist())
    auc = float(roc_auc_score(pair_targets, pair_probabilities)) if len(set(pair_targets)) == 2 else np.nan
    return choices, margins, auc


def _assemble(base, paths, base_index, well_slices, choices):
    output = paths[base_index].copy()
    for slices in well_slices.values():
        for group, _, positions in slices:
            choice = int(choices[group])
            if choice >= 0:
                output[positions] = paths[choice, positions]
    return output


V = _load_v188_panel()
base = V["base"]
paths = V["paths"]
truth = V["truth"]
names = V["names"]
base_index = int(V["base_index"])
k = int(V["k"])
x = V["x"]
panel = V["panel"]
y = V["y"]
sse = V["sse"]
folds = V["folds"]
well_slices = V["well_slices"]
feature_names = V["feature_names"]
metric_fn = V["_metrics"]

starts, counts = _group_rows(panel)
group_fold = panel.iloc[starts]["fold"].to_numpy(np.int8)
if len(panel) != len(x) or len(panel) != len(y) or len(panel) != len(sse):
    raise RuntimeError("V188 panel arrays are not aligned")
if np.any(counts != k):
    raise RuntimeError("Every block must contain every endpoint candidate")

x_values = x.to_numpy(np.float64)
crossfit_choices = np.full(len(starts), -1, np.int16)
shuffle_choices = np.full(len(starts), -1, np.int16)
crossfit_margins = np.full(len(starts), np.nan, np.float32)
fold_reports = []
for held in range(1, 6):
    train_row_mask = folds != held
    valid_group_mask = group_fold == held
    matrix, target, weight, scale, train_groups, train_pairs = _training_pairs(
        x, y, panel, train_row_mask, starts, counts
    )
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=450, learning_rate=0.025,
        num_leaves=31, min_child_samples=100, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
        reg_alpha=1.0, reg_lambda=20.0, random_state=SEED + held,
        n_jobs=-1, verbosity=-1, force_col_wise=True,
    )
    model.fit(matrix, target, sample_weight=weight)
    model.booster_.save_model(str(MODEL_DIR / f"pairwise_hold{held}.txt"))
    choice, margin, auc = _tournament(
        model, x_values, scale, starts, counts, valid_group_mask,
        shuffle=False, seed=SEED + held,
    )
    shuffled, _, _ = _tournament(
        model, x_values, scale, starts, counts, valid_group_mask,
        shuffle=True, seed=SEED + 100 + held,
    )
    crossfit_choices[valid_group_mask] = choice[valid_group_mask]
    shuffle_choices[valid_group_mask] = shuffled[valid_group_mask]
    crossfit_margins[valid_group_mask] = margin[valid_group_mask]
    fold_reports.append({
        "fold": held,
        "train_groups": train_groups,
        "train_hard_pairs": train_pairs,
        "valid_groups": int(valid_group_mask.sum()),
        "held_pair_auc": auc,
    })
    print("M342-F1 FOLD", fold_reports[-1], flush=True)

if np.any(crossfit_choices < 0) or np.any(shuffle_choices < 0):
    raise RuntimeError("Incomplete F1 crossfit choices")

candidate_prediction = _assemble(base, paths, base_index, well_slices, crossfit_choices)
shuffle_prediction = _assemble(base, paths, base_index, well_slices, shuffle_choices)
oracle_choices = np.argmin(sse.reshape(len(starts), k), axis=1).astype(np.int16)
oracle_prediction = _assemble(base, paths, base_index, well_slices, oracle_choices)

# Fold-specific constant candidate is selected only on the other four folds.
constant_choices = np.empty(len(starts), np.int16)
candidate_ids = panel["candidate"].to_numpy(np.int16)
for held in range(1, 6):
    train = folds != held
    totals = np.asarray([
        sse[train & (candidate_ids == candidate)].sum() for candidate in range(k)
    ])
    constant_choices[group_fold == held] = int(np.argmin(totals))
constant_prediction = _assemble(base, paths, base_index, well_slices, constant_choices)

predictions = {
    "v168": paths[base_index],
    "fold_constant": constant_prediction,
    "pairwise": candidate_prediction,
    "feature_alignment_shuffle": shuffle_prediction,
    "block_oracle": oracle_prediction,
}
summary = pd.DataFrame([
    metric_fn(base, prediction, name) for name, prediction in predictions.items()
]).sort_values("rmse", ignore_index=True)

fold_rows = []
for held in range(1, 6):
    mask = base["fold"].to_numpy(np.int8) == held
    local = base.loc[mask].reset_index(drop=True)
    metrics = {
        name: metric_fn(local, prediction[mask], name)
        for name, prediction in predictions.items()
    }
    fold_rows.append({
        "fold": held,
        "base_rmse": metrics["v168"]["rmse"],
        "constant_rmse": metrics["fold_constant"]["rmse"],
        "pairwise_rmse": metrics["pairwise"]["rmse"],
        "gain_vs_base": metrics["v168"]["rmse"] - metrics["pairwise"]["rmse"],
        "gain_vs_constant": metrics["fold_constant"]["rmse"] - metrics["pairwise"]["rmse"],
        "shuffle_rmse": metrics["feature_alignment_shuffle"]["rmse"],
        "shuffle_loss": metrics["feature_alignment_shuffle"]["rmse"] - metrics["pairwise"]["rmse"],
        "oracle_rmse": metrics["block_oracle"]["rmse"],
    })
fold_frame = pd.DataFrame(fold_rows)

lookup = summary.set_index("endpoint")
gain_vs_constant = float(lookup.loc["fold_constant", "rmse"] - lookup.loc["pairwise", "rmse"])
shuffle_loss = float(lookup.loc["feature_alignment_shuffle", "rmse"] - lookup.loc["pairwise", "rmse"])
mean_auc = float(np.mean([row["held_pair_auc"] for row in fold_reports]))
checks = {
    "mean_pair_auc_ge_0p60": mean_auc >= 0.60,
    "gain_vs_constant_ge_0p20": gain_vs_constant >= 0.20,
    "shuffle_loses_ge_0p20": shuffle_loss >= 0.20,
    "positive_folds_vs_constant_ge_4": int((fold_frame["gain_vs_constant"] > 0).sum()) >= 4,
    "minimum_fold_shuffle_loss_ge_0p05": float(fold_frame["shuffle_loss"].min()) >= 0.05,
    "block_oracle_rmse_lt_5": float(lookup.loc["block_oracle", "rmse"]) < 5.0,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_segment_pairwise_observability_f1",
    "stage": "fivefold_real_endpoint_segment_hardpair_observability_gateway",
    "sample_unit": "one real candidate endpoint on one 128-ft physical-well block",
    "truth_usage": "hard-pair training labels and held-fold metrics only",
    "candidate_endpoints": names,
    "candidate_count": k,
    "features": feature_names,
    "hard_pair_recipe": {
        "minimum_regret_ft": MIN_REGRET_FT,
        "maximum_regret_ft": MAX_REGRET_FT,
        "nearest_negatives_per_block": NEGATIVES_PER_BLOCK,
        "symmetric_pairs": True,
    },
    "coverage": {
        "rows": int(len(base)), "wells": int(base["well"].nunique()),
        "blocks": int(len(starts)), "candidate_rows": int(len(panel)),
    },
    "fold_reports": fold_reports,
    "summary": summary.to_dict("records"),
    "folds": fold_rows,
    "values": {
        "mean_pair_auc": mean_auc,
        "gain_vs_constant": gain_vs_constant,
        "shuffle_loss": shuffle_loss,
        "mean_selection_margin": float(np.nanmean(crossfit_margins)),
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze F1 scores and test continuity-DP consumption against V168"
        if strict_pass else
        "FAIL -> segment winner is not observable; close blockwise candidate selection"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
pd.DataFrame(fold_reports).to_csv(OUT / "fold_reports.csv", index=False)
pd.DataFrame({
    "group": np.arange(len(starts)), "fold": group_fold,
    "choice": crossfit_choices, "shuffle_choice": shuffle_choices,
    "oracle_choice": oracle_choices, "margin": crossfit_margins,
}).to_parquet(OUT / "block_choices.parquet", index=False)
(OUT / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-F1 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-F1 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-F1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
