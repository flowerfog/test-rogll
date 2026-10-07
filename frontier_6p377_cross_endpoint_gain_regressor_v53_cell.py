"""Nested gain regressor for the independent V51 endpoint bank.

V52 established weak but causal whole-well selection signal.  Its rank labels
discarded gain magnitude, so this follow-up predicts each non-base endpoint's
far-tail SSE improvement directly.  All threshold and dose choices remain
nested across folds 1/2/3.  Rows through 2000 ft are restored from float64 base
after every assembly to guarantee exact equality.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_cross_endpoint_gain_regressor_v53"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V52_SOURCE = ROOT / "frontier_6p377_cross_endpoint_wholewell_selector_v52_cell.py"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = (1, 2, 3)
SEED = 20260727
GAIN_SCALE = 100.0
THRESHOLDS = (-0.30, -0.15, -0.05, 0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40)
DOSES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)
started = time.time()


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
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


if not V52_SOURCE.is_file():
    raise FileNotFoundError(V52_SOURCE)
source = V52_SOURCE.read_text(encoding="utf-8")
marker = "\ndef _fit_model(local, seed):"
if marker not in source:
    raise RuntimeError("V52 definition marker not found")
V = {"__name__": "v52_gain_feature_builder", "__file__": str(V52_SOURCE)}
exec(compile(source.split(marker, 1)[0], str(V52_SOURCE), "exec"), V)

frame = V["frame"]
features = V["features"].copy()
feature_names = list(V["feature_names"])
candidate_feature_names = list(V["candidate_feature_names"])
candidate_names = list(V["candidate_names"])
candidate_matrix = np.asarray(V["candidate_matrix"], np.float64)
base = np.asarray(V["base"], np.float64)
truth = np.asarray(V["truth"], np.float64)
distance = np.asarray(V["distance"], np.float64)
wells = np.asarray(V["wells"], str)
fold_values = np.asarray(V["fold_values"], np.int8)
groups = V["groups"]
fold_by_well = V["fold_by_well"]
frozen = np.asarray(V["frozen"], bool)
state_count = len(candidate_names)
_metrics = V["_metrics"]

if state_count != 4:
    raise RuntimeError(f"Expected four V52 states, got {candidate_names}")

# Convert absolute candidate MSE into a robust signed gain label.  Base rows
# are excluded from fitting; they remain the explicit abstention state.
base_mse_by_well = (
    features.loc[features["state"].eq(0), ["well", "target_far_mse"]]
    .set_index("well")["target_far_mse"]
    .to_dict()
)
features["raw_gain_mse"] = features.apply(
    lambda row: float(base_mse_by_well[str(row["well"])] - row["target_far_mse"]),
    axis=1,
)
features["target_gain"] = np.arcsinh(features["raw_gain_mse"].to_numpy(np.float64) / GAIN_SCALE)
trainable = features[features["state"].gt(0)].copy()
if trainable["well"].nunique() < 460:
    raise RuntimeError("Incomplete V53 trainable feature matrix")


def _fit_model(local, seed):
    local = local.sort_values(["well", "state"], kind="stable")
    magnitude = np.abs(local["raw_gain_mse"].to_numpy(np.float64))
    positive_scale = np.median(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1.0
    importance = np.clip(magnitude / max(float(positive_scale), 1e-6), 0.5, 4.0)
    sample_weight = np.sqrt(np.maximum(local["far_rows"].to_numpy(np.float64), 1.0)) * importance
    model = lgb.LGBMRegressor(
        objective="huber", alpha=0.85,
        n_estimators=420, learning_rate=0.018,
        num_leaves=9, max_depth=4, min_child_samples=26,
        subsample=0.85, colsample_bytree=0.72,
        reg_alpha=2.5, reg_lambda=20.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(local[feature_names], local["target_gain"], sample_weight=sample_weight)
    return model


def _predict(model, local, shuffled=False, seed=SEED):
    local = local.sort_values(["well", "state"], kind="stable").copy()
    matrix = np.array(local[feature_names].to_numpy(np.float64), copy=True)
    if shuffled:
        rng = np.random.default_rng(seed)
        candidate_positions = [feature_names.index(name) for name in candidate_feature_names]
        for _, positions in local.groupby("well", sort=False).indices.items():
            positions = np.asarray(positions, np.int64)
            permutation = rng.permutation(len(positions))
            matrix[np.ix_(positions, candidate_positions)] = matrix[
                np.ix_(positions[permutation], candidate_positions)
            ]
    output = local[["well", "fold", "state"]].copy()
    output["predicted_gain"] = model.predict(matrix)
    return output


def _assemble(scores, threshold, dose, target_wells):
    prediction = base.copy()
    choices = []
    score_groups = {str(well): local for well, local in scores.groupby("well", sort=False)}
    for well in target_wells:
        local = score_groups.get(str(well))
        if local is None or len(local) != state_count - 1:
            continue
        local = local.sort_values("state", kind="stable")
        gains = local["predicted_gain"].to_numpy(np.float64)
        best_position = int(np.argmax(gains))
        best_gain = float(gains[best_position])
        state = int(local["state"].iloc[best_position]) if best_gain >= float(threshold) else 0
        indices = groups[str(well)]
        if state:
            prediction[indices] = base[indices] + float(dose) * (
                candidate_matrix[indices, state] - base[indices]
            )
        choices.append({
            "well": str(well), "state": state, "candidate": candidate_names[state],
            "predicted_gain": best_gain, "threshold": float(threshold), "dose": float(dose),
        })
    prediction[frozen] = base[frozen]
    return prediction, choices


def _scan(scores, folds_to_score):
    target_wells = sorted(well for well, fold in fold_by_well.items() if fold in folds_to_score)
    row_mask = np.isin(fold_values, list(folds_to_score))
    base_metrics = _metrics(base, "base", row_mask)
    rows = []
    for threshold in THRESHOLDS:
        for dose in DOSES:
            prediction, choices = _assemble(scores, threshold, dose, target_wells)
            metrics = _metrics(prediction, "candidate", row_mask)
            fold_gains = []
            for fold in folds_to_score:
                mask = fold_values == fold
                fold_gains.append(
                    _metrics(base, "base", mask)["rmse"]
                    - _metrics(prediction, "candidate", mask)["rmse"]
                )
            rows.append({
                "threshold": float(threshold), "dose": float(dose),
                "rmse": metrics["rmse"], "gain": base_metrics["rmse"] - metrics["rmse"],
                "far_gain": base_metrics["far_gt2000_rmse"] - metrics["far_gt2000_rmse"],
                "p95_change": metrics["p95_well_rmse"] - base_metrics["p95_well_rmse"],
                "minimum_fold_gain": float(min(fold_gains)),
                "activation_fraction": float(np.mean([item["state"] != 0 for item in choices])),
            })
    table = pd.DataFrame(rows)
    safe = table[(table["p95_change"] <= 0.20) & (table["minimum_fold_gain"] >= -0.05)]
    selected = (safe if len(safe) else table).sort_values(
        ["rmse", "minimum_fold_gain"], ascending=[True, False], kind="stable"
    ).iloc[0]
    return table, selected


oof_prediction = base.copy()
oof_shuffle_prediction = base.copy()
fold_reports = []
choice_rows = []
scan_rows = []
importance_rows = []
for held_fold in FOLDS:
    train_folds = [fold for fold in FOLDS if fold != held_fold]
    inner_scores = []
    for inner_valid in train_folds:
        inner_train_fold = next(fold for fold in train_folds if fold != inner_valid)
        inner_train = trainable[trainable["fold"].eq(inner_train_fold)]
        inner_valid_frame = trainable[trainable["fold"].eq(inner_valid)]
        inner_model = _fit_model(inner_train, SEED + held_fold * 10 + inner_valid)
        inner_scores.append(_predict(inner_model, inner_valid_frame))
    inner_scores = pd.concat(inner_scores, ignore_index=True)
    scan, selected = _scan(inner_scores, tuple(train_folds))
    scan["held_fold"] = held_fold
    scan_rows.append(scan)

    train = trainable[trainable["fold"].isin(train_folds)]
    held = trainable[trainable["fold"].eq(held_fold)]
    model = _fit_model(train, SEED + held_fold)
    model.booster_.save_model(str(MODEL_DIR / f"gain_regressor_fold{held_fold}.txt"))
    held_scores = _predict(model, held)
    shuffled_scores = _predict(model, held, shuffled=True, seed=SEED + held_fold)
    held_wells = sorted(well for well, fold in fold_by_well.items() if fold == held_fold)
    prediction, choices = _assemble(held_scores, selected["threshold"], selected["dose"], held_wells)
    shuffled_prediction, _ = _assemble(
        shuffled_scores, selected["threshold"], selected["dose"], held_wells
    )
    mask = fold_values == held_fold
    oof_prediction[mask] = prediction[mask]
    oof_shuffle_prediction[mask] = shuffled_prediction[mask]
    choice_rows.extend({**item, "fold": held_fold} for item in choices)
    base_fold = _metrics(base, "base", mask)
    candidate_fold = _metrics(prediction, "candidate", mask)
    shuffle_fold = _metrics(shuffled_prediction, "shuffle", mask)
    fold_reports.append({
        "fold": held_fold,
        "selected_threshold": float(selected["threshold"]),
        "selected_dose": float(selected["dose"]),
        "inner_gain": float(selected["gain"]),
        "inner_activation_fraction": float(selected["activation_fraction"]),
        "base_rmse": base_fold["rmse"], "rmse": candidate_fold["rmse"],
        "gain": base_fold["rmse"] - candidate_fold["rmse"],
        "base_far_rmse": base_fold["far_gt2000_rmse"],
        "far_rmse": candidate_fold["far_gt2000_rmse"],
        "far_gain": base_fold["far_gt2000_rmse"] - candidate_fold["far_gt2000_rmse"],
        "shuffle_rmse": shuffle_fold["rmse"],
        "shuffle_loss": shuffle_fold["rmse"] - candidate_fold["rmse"],
        "p95_change": candidate_fold["p95_well_rmse"] - base_fold["p95_well_rmse"],
    })
    importance_rows.extend(
        {"fold": held_fold, "feature": name, "gain": float(gain)}
        for name, gain in zip(
            feature_names, model.booster_.feature_importance(importance_type="gain")
        )
    )
    print("V53 FOLD", fold_reports[-1], flush=True)

oof_prediction[frozen] = base[frozen]
oof_shuffle_prediction[frozen] = base[frozen]
base_metrics = _metrics(base, "base")
candidate_metrics = _metrics(oof_prediction, "nested_gain_regressor")
shuffle_metrics = _metrics(oof_shuffle_prediction, "candidate_feature_shuffle")
fold_table = pd.DataFrame(fold_reports)
choices = pd.DataFrame(choice_rows)
checks = {
    "overall_gain_ge_0p05": bool(base_metrics["rmse"] - candidate_metrics["rmse"] >= 0.05),
    "far_gain_ge_0p075": bool(base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"] >= 0.075),
    "minimum_fold_gain_ge_minus0p03": bool(fold_table["gain"].min() >= -0.03),
    "positive_folds_ge_2": bool((fold_table["gain"] > 0).sum() >= 2),
    "shuffle_loses_ge_0p03": bool(shuffle_metrics["rmse"] - candidate_metrics["rmse"] >= 0.03),
    "maximum_p95_change_le_0p20": bool(fold_table["p95_change"].max() <= 0.20),
    "near_rows_bitwise_unchanged": bool(np.array_equal(oof_prediction[frozen], base[frozen])),
}
strict_pass = bool(all(checks.values()))

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["base_6p377"] = base.astype(np.float32)
output["nested_gain_regressor"] = oof_prediction.astype(np.float32)
output["candidate_feature_shuffle"] = oof_shuffle_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "gain_regressor_oof.parquet", index=False, compression="zstd")
choices.to_csv(OUT_DIR / "well_choices.csv", index=False)
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
pd.concat(scan_rows, ignore_index=True).to_csv(OUT_DIR / "inner_scan.csv", index=False)
pd.DataFrame(importance_rows).to_csv(OUT_DIR / "feature_importance.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_cross_endpoint_gain_regression",
    "sample_unit": "one complete physical well and one non-base endpoint",
    "source": str(V52_SOURCE),
    "candidates": candidate_names,
    "gain_scale": GAIN_SCALE,
    "base": base_metrics,
    "regressor": candidate_metrics,
    "candidate_feature_shuffle": shuffle_metrics,
    "gain": base_metrics["rmse"] - candidate_metrics["rmse"],
    "far_gain": base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"],
    "shuffle_loss": shuffle_metrics["rmse"] - candidate_metrics["rmse"],
    "activation_fraction": float((choices["state"] != 0).mean()),
    "choice_counts": choices["candidate"].value_counts().to_dict(),
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> freeze this whole-well gate and test 512-ft endpoint changes only inside activated wells"
        if strict_pass else
        "FAIL -> endpoint gain magnitude is not observable from legal whole-well features; close whole-well selection"
    ),
}
_write_json(OUT_DIR / "result.json", verdict)

print("\nV53 FOLDS", flush=True)
print(fold_table.to_string(index=False), flush=True)
print("\nV53 CHOICE COUNTS", flush=True)
print(choices["candidate"].value_counts(dropna=False).to_string(), flush=True)
print("\nV53 FINAL VERDICT", flush=True)
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
