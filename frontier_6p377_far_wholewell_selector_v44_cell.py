"""Whole-well selector for the coherent V43 far-endpoint oracle.

V43 showed that one frozen endpoint per physical well captures most of V40's
segment oracle.  This experiment therefore aggregates V41's legal segment
evidence to (well, candidate) records and predicts complete-well candidate
gain.  Selection and dose are nested across folds 1/2/3.  Rows through 2000 ft
remain exactly equal to the V33/6.377 lineage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_far_wholewell_selector_v44"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
FEATURE_PATH = OOF_ROOT / "frontier_6p377_far_segment_selector_v41" / "segment_candidate_features.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
SEED = 20260727
THRESHOLDS = (-0.05, 0.0, 0.025, 0.05, 0.10, 0.20, 0.35)
DOSES = (0.25, 0.50, 0.75, 1.00)
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


def _smoothstep(values):
    t = np.clip((np.asarray(values, np.float64) - FAR_START_FT) / FADE_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _centred_mean(x, values, width):
    x = np.asarray(x, np.float64)
    values = np.asarray(values, np.float64)
    left = np.searchsorted(x, x - width / 2.0, side="left")
    right = np.searchsorted(x, x + width / 2.0, side="right")
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[right] - prefix[left]) / np.maximum(right - left, 1)


def _metrics(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local["tvt_true"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    wells = local["well"].astype(str).to_numpy()
    per_well = pd.Series(error**2).groupby(wells, sort=False).mean().pow(0.5)
    result = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
    }
    for name, mask in {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
    return result


def _weighted_mean(values, weights):
    return float(np.average(np.asarray(values, np.float64), weights=np.asarray(weights, np.float64)))


def _trend(values):
    values = np.asarray(values, np.float64)
    if len(values) < 2:
        return 0.0
    x = np.linspace(-1.0, 1.0, len(values))
    return float(np.dot(x, values - values.mean()) / max(np.dot(x, x), 1e-9))


for required in (V33_PATH, V28_PATH, FEATURE_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v33_columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", "pre_neighbor_cp8", BASE_COLUMN]
frame = pd.read_parquet(V33_PATH, columns=v33_columns)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

tail_columns = sorted(column for column in pq.ParquetFile(V28_PATH).schema.names if column.startswith("tail_"))
tail = pd.read_parquet(V28_PATH, columns=["id"] + tail_columns)
tail["id"] = tail["id"].astype(str)
frame = frame.merge(tail, on="id", how="left", validate="one_to_one")

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fade = _smoothstep(distance)
prefix = distance <= FAR_START_FT

candidate_paths = [base]
usable_names = ["base_6p377"]
for name in ["pre_neighbor_cp8"] + tail_columns:
    absolute = frame[name].to_numpy(np.float64)
    finite = np.isfinite(absolute)
    if finite.mean() < 0.95:
        continue
    correction = np.where(finite, np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[prefix] = base[prefix]
    candidate_paths.append(candidate)
    usable_names.append(name)
candidates = np.column_stack(candidate_paths)
state_count = candidates.shape[1]
if state_count < 3 or not np.isfinite(candidates).all():
    raise RuntimeError(f"Invalid candidate bank: {usable_names}")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}
segments = pd.read_parquet(FEATURE_PATH)
segments["well"] = segments["well"].astype(str)
segments = segments[segments["state"].between(0, state_count - 1)].copy()

# Candidate-specific evidence is aggregated over the complete far section.
aggregate_columns = [
    "eval_gr_coverage", "path_spread_mean", "path_spread_max",
    "correction_mean", "correction_rms", "correction_std", "correction_p95abs",
    "correction_start", "correction_end", "correction_slope", "correction_tv",
    "consensus_mean", "consensus_rms", "consensus_std",
    "emission_coverage", "emission_ncc", "emission_cauchy", "emission_bias", "emission_std",
]
constant_columns = ["max_md_since", "prefix_gain", "prefix_bias", "prefix_gamma"]
missing = sorted(set(aggregate_columns + constant_columns).difference(segments.columns))
if missing:
    raise RuntimeError(f"V41 feature columns missing: {missing}")

records = []
for (well, state), local in segments.groupby(["well", "state"], sort=False):
    local = local.sort_values("block", kind="stable")
    weights = np.maximum(local["segment_rows"].to_numpy(np.float64), 1.0)
    record = {
        "well": str(well),
        "fold": int(local["fold"].iloc[0]),
        "state": int(state),
        "is_base": float(int(state) == 0),
        "segments": int(len(local)),
        "far_rows": int(weights.sum()),
        **{f"is_state_{slot}": float(int(state) == slot) for slot in range(state_count)},
    }
    for column in constant_columns:
        record[column] = float(local[column].iloc[0])
    for column in aggregate_columns:
        values = local[column].to_numpy(np.float64)
        record[f"{column}_mean"] = _weighted_mean(values, weights)
        record[f"{column}_std"] = float(np.std(values))
        record[f"{column}_p10"] = float(np.quantile(values, 0.10))
        record[f"{column}_p90"] = float(np.quantile(values, 0.90))
        record[f"{column}_first"] = float(values[0])
        record[f"{column}_last"] = float(values[-1])
        record[f"{column}_trend"] = _trend(values)
    indices = groups[str(well)]
    selected = candidates[indices, int(state)]
    correction = _centred_mean(distance[indices], selected - base[indices], SMOOTH_FT)
    correction = np.clip(correction, -CAP_FT, CAP_FT)
    feasible = base[indices] + fade[indices] * correction
    feasible[distance[indices] <= FAR_START_FT] = base[indices][distance[indices] <= FAR_START_FT]
    base_mse = float(np.mean((base[indices] - truth[indices]) ** 2))
    candidate_mse = float(np.mean((feasible - truth[indices]) ** 2))
    record["raw_gain_mse"] = base_mse - candidate_mse
    record["target_gain"] = float(np.arcsinh((base_mse - candidate_mse) / 25.0))
    records.append(record)

features = pd.DataFrame(records)
if features["well"].nunique() < 455:
    raise RuntimeError(f"Only {features['well'].nunique()} aggregate feature wells")
excluded = {"well", "fold", "state", "target_gain", "raw_gain_mse"}
feature_names = [column for column in features.columns if column not in excluded]
if features[feature_names].isna().any().any():
    raise RuntimeError("NaN in V44 feature matrix")
features.to_parquet(OUT_DIR / "well_candidate_features.parquet", index=False, compression="zstd")


def _fit_model(train):
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=350,
        learning_rate=0.02,
        num_leaves=9,
        max_depth=4,
        min_child_samples=24,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_alpha=2.0,
        reg_lambda=15.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train[feature_names],
        train["target_gain"],
        sample_weight=np.sqrt(np.maximum(train["far_rows"].to_numpy(np.float64), 1.0)),
    )
    return model


def _predict(model, local, shuffled=False, seed=SEED):
    matrix = np.array(local[feature_names].to_numpy(np.float64), copy=True)
    if shuffled:
        rng = np.random.default_rng(seed)
        wells = local["well"].to_numpy(str)
        for well in np.unique(wells):
            positions = np.flatnonzero(wells == well)
            if len(positions) > 1:
                matrix[positions] = matrix[rng.permutation(positions)]
    return model.predict(matrix)


def _assemble(scores, threshold, dose, wells):
    prediction = base.copy()
    choices = []
    score_groups = {str(well): local for well, local in scores.groupby("well", sort=False)}
    for well in wells:
        local_scores = score_groups.get(well)
        if local_scores is None or local_scores.empty:
            continue
        local_scores = local_scores.sort_values("state")
        state_values = np.full(state_count, -1e6, np.float64)
        states = local_scores["state"].to_numpy(np.int64)
        state_values[states] = local_scores["score"].to_numpy(np.float64)
        best = int(np.argmax(state_values[1:]) + 1)
        margin = float(state_values[best] - state_values[0])
        selected_state = best if margin >= float(threshold) else 0
        indices = groups[well]
        selected = candidates[indices, selected_state]
        correction = _centred_mean(distance[indices], selected - base[indices], SMOOTH_FT)
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        local_prediction = base[indices] + float(dose) * fade[indices] * correction
        local_prediction[distance[indices] <= FAR_START_FT] = base[indices][distance[indices] <= FAR_START_FT]
        prediction[indices] = local_prediction
        choices.append({
            "well": well,
            "state": selected_state,
            "candidate": usable_names[selected_state],
            "margin": margin,
        })
    return prediction, pd.DataFrame(choices)


def _select_config(scores, folds):
    wells = sorted(features.loc[features["fold"].isin(folds), "well"].unique())
    row_mask = frame["fold"].isin(folds).to_numpy()
    local = frame.loc[row_mask].reset_index(drop=True)
    base_metric = _metrics(local, base[row_mask], "base")
    reports = []
    for threshold in THRESHOLDS:
        for dose in DOSES:
            prediction, choices = _assemble(scores, threshold, dose, wells)
            metric = _metrics(local, prediction[row_mask], "candidate")
            reports.append({
                "threshold": threshold,
                "dose": dose,
                "rmse": metric["rmse"],
                "gain": base_metric["rmse"] - metric["rmse"],
                "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
                "nonbase_rate": float((choices["state"] != 0).mean()) if len(choices) else 0.0,
            })
    scan = pd.DataFrame(reports)
    safe = scan[scan["p95_change"] <= 0.15]
    selected = (safe if len(safe) else scan).sort_values(["rmse", "far_gain"], ascending=[True, False]).iloc[0]
    return selected, scan


oof_prediction = base.copy()
shuffle_prediction = base.copy()
fold_reports = []
scan_tables = []
choice_tables = []

for held_fold in FOLDS:
    train_folds = [fold for fold in FOLDS if fold != held_fold]
    inner_scores = []
    for inner_held in train_folds:
        inner_train_fold = next(fold for fold in train_folds if fold != inner_held)
        model = _fit_model(features[features["fold"].eq(inner_train_fold)])
        valid = features[features["fold"].eq(inner_held)].copy()
        valid["score"] = _predict(model, valid)
        inner_scores.append(valid)
    inner_scores = pd.concat(inner_scores, ignore_index=True)
    selected, scan = _select_config(inner_scores, train_folds)
    scan["held_fold"] = held_fold
    scan_tables.append(scan)

    model = _fit_model(features[features["fold"].isin(train_folds)])
    model.booster_.save_model(str(MODEL_DIR / f"fold{held_fold}.txt"))
    held = features[features["fold"].eq(held_fold)].copy()
    held["score"] = _predict(model, held)
    shuffled = held[["well", "state"]].copy()
    shuffled["score"] = _predict(model, held, shuffled=True, seed=SEED + held_fold)
    held_wells = sorted(held["well"].unique())
    prediction, choices = _assemble(held, float(selected["threshold"]), float(selected["dose"]), held_wells)
    shuffled_path, _ = _assemble(shuffled, float(selected["threshold"]), float(selected["dose"]), held_wells)
    row_mask = frame["fold"].to_numpy(np.int8) == held_fold
    oof_prediction[row_mask] = prediction[row_mask]
    shuffle_prediction[row_mask] = shuffled_path[row_mask]

    local = frame.loc[row_mask].reset_index(drop=True)
    base_metric = _metrics(local, base[row_mask], "base")
    metric = _metrics(local, oof_prediction[row_mask], "wholewell_selector")
    shuffle_metric = _metrics(local, shuffle_prediction[row_mask], "candidate_shuffle")
    choices["fold"] = held_fold
    choice_tables.append(choices)
    fold_reports.append({
        "fold": held_fold,
        "threshold": float(selected["threshold"]),
        "dose": float(selected["dose"]),
        "inner_rmse": float(selected["rmse"]),
        "nonbase_rate": float((choices["state"] != 0).mean()) if len(choices) else 0.0,
        "base_rmse": base_metric["rmse"],
        "rmse": metric["rmse"],
        "gain": base_metric["rmse"] - metric["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
        "shuffle_rmse": shuffle_metric["rmse"],
        "shuffle_loss": shuffle_metric["rmse"] - metric["rmse"],
    })
    print("V44 FOLD", fold_reports[-1], flush=True)

base_result = _metrics(frame, base, "base_6p377")
selector_result = _metrics(frame, oof_prediction, "nested_wholewell_selector")
shuffle_result = _metrics(frame, shuffle_prediction, "candidate_feature_shuffle")
gain = base_result["rmse"] - selector_result["rmse"]
far_gain = base_result["far_gt2000_rmse"] - selector_result["far_gt2000_rmse"]
shuffle_loss = shuffle_result["rmse"] - selector_result["rmse"]
fold_table = pd.DataFrame(fold_reports)
checks = {
    "gain_ge_0p10": bool(gain >= 0.10),
    "minimum_fold_gain_ge_minus0p02": bool(fold_table["gain"].min() >= -0.02),
    "positive_folds_ge_2": bool((fold_table["gain"] > 0).sum() >= 2),
    "far_gain_ge_0p15": bool(far_gain >= 0.15),
    "shuffle_loses_70pct_gain": bool(shuffle_loss >= 0.70 * max(gain, 0.0)),
    "maximum_fold_p95_change_le_0p15": bool(fold_table["p95_change"].max() <= 0.15),
    "near_is_bitwise_frozen": bool(np.array_equal(oof_prediction[prefix], base[prefix])),
}
strict_pass = bool(all(checks.values()))

pd.concat(scan_tables, ignore_index=True).to_csv(OUT_DIR / "inner_scans.csv", index=False)
pd.concat(choice_tables, ignore_index=True).to_csv(OUT_DIR / "well_choices.csv", index=False)
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    base_6p377=base.astype(np.float32),
    nested_wholewell_selector=oof_prediction.astype(np.float32),
    candidate_feature_shuffle=shuffle_prediction.astype(np.float32),
).to_parquet(OUT_DIR / "wholewell_selector_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_complete_well_far_endpoint_selector",
    "sample_unit": "complete physical well candidate",
    "deployable_inputs_only": True,
    "lineage": {
        "base": str(V33_PATH),
        "base_column": BASE_COLUMN,
        "candidate_source": str(V28_PATH),
        "segment_evidence": str(FEATURE_PATH),
    },
    "candidate_names": usable_names,
    "feature_count": len(feature_names),
    "feature_wells": int(features["well"].nunique()),
    "base": base_result,
    "nested_wholewell_selector": selector_result,
    "candidate_feature_shuffle": shuffle_result,
    "gain": gain,
    "far_gain": far_gain,
    "shuffle_loss": shuffle_loss,
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> refit and package the far-only whole-well endpoint selector"
        if strict_pass
        else "FAIL -> coherent oracle identity is not observable from current legal evidence; close bank"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV44 SUMMARY")
print(pd.DataFrame([base_result, selector_result, shuffle_result]).to_string(index=False))
print("\nV44 FOLDS")
print(fold_table.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default))
print("saved:", OUT_DIR)
