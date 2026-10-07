"""Sequence-constrained routing of V40's far-segment endpoint bank.

V41 established that independent 256-ft candidate selection cannot reach the
segment oracle.  This experiment keeps its legal cross-fitted unary scorer but
decodes every well jointly.  A small dynamic program penalises endpoint mode
switches and discontinuities between adjacent segments.  Rows at or before
2000 ft remain exactly equal to the V33/6.377 lineage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_far_segment_sequence_dp_v42"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
V41_DIR = OOF_ROOT / "frontier_6p377_far_segment_selector_v41"
FEATURE_PATH = V41_DIR / "segment_candidate_features.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
SEGMENT_FT = 256.0
SEED = 20260727

# Scores are in asinh(MSE-gain / 25) units.
BASE_BIASES = (0.00, 0.025, 0.075, 0.15)
SWITCH_PENALTIES = (0.00, 0.025, 0.075, 0.15)
JUMP_PENALTIES = (0.00, 0.01, 0.03)
DOSES = (0.25, 0.50, 0.75)
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
    if len(values) < 2 or width <= 1.0:
        return values.copy()
    left = np.searchsorted(x, x - width / 2.0, side="left")
    right = np.searchsorted(x, x + width / 2.0, side="right")
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[right] - prefix[left]) / np.maximum(right - left, 1)


def _metrics(local, prediction, endpoint):
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
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
    bands = {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in bands.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
    return result


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

candidate_names = ["base_6p377", "pre_neighbor_cp8"] + tail_columns
candidate_paths = [base]
usable_names = [candidate_names[0]]
for name in candidate_names[1:]:
    absolute = frame[name].to_numpy(np.float64)
    if np.isfinite(absolute).mean() < 0.95:
        continue
    correction = np.where(np.isfinite(absolute), np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[prefix] = base[prefix]
    candidate_paths.append(candidate)
    usable_names.append(name)
candidates = np.column_stack(candidate_paths)
state_count = candidates.shape[1]
if state_count < 3 or not np.isfinite(candidates).all():
    raise RuntimeError(f"Invalid candidate bank: {usable_names}")

features = pd.read_parquet(FEATURE_PATH)
features["well"] = features["well"].astype(str)
features = features[features["state"].between(0, state_count - 1)].copy()
excluded = {"well", "fold", "block", "state", "target_gain", "raw_gain_mse"}
feature_names = [column for column in features.columns if column not in excluded]
if features[feature_names].isna().any().any():
    raise RuntimeError("NaN in V41 feature matrix")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}


def _fit_model(train):
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=48,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    weight = np.sqrt(np.maximum(train["segment_rows"].to_numpy(np.float64), 1.0))
    model.fit(train[feature_names], train["target_gain"], sample_weight=weight)
    return model


def _predict(model, local):
    return model.predict(local[feature_names].to_numpy(np.float64))


def _decode_well(local_scores, base_bias, switch_penalty, jump_penalty, shuffle_blocks=False, seed=SEED):
    table = local_scores.sort_values(["block", "state"], kind="stable").copy()
    blocks = np.sort(table["block"].unique())
    if not len(blocks):
        return np.empty(0, np.int16)
    scores = np.full((len(blocks), state_count), -1e6, np.float64)
    starts = np.zeros_like(scores)
    ends = np.zeros_like(scores)
    for block_index, block in enumerate(blocks):
        block_rows = table[table["block"].eq(block)].sort_values("state")
        states = block_rows["state"].to_numpy(np.int64)
        scores[block_index, states] = block_rows["score"].to_numpy(np.float64)
        starts[block_index, states] = block_rows["correction_start"].to_numpy(np.float64)
        ends[block_index, states] = block_rows["correction_end"].to_numpy(np.float64)
    if shuffle_blocks and len(blocks) > 1:
        rng = np.random.default_rng(seed)
        scores = scores[rng.permutation(len(blocks))]
    scores = np.clip(scores - scores[:, [0]], -2.0, 2.0)
    scores[:, 1:] -= float(base_bias)

    value = np.full_like(scores, -np.inf)
    parent = np.zeros((len(blocks), state_count), np.int16)
    value[0] = scores[0]
    for step in range(1, len(blocks)):
        jump = np.abs(ends[step - 1, :, None] - starts[step, None, :]) / 4.0
        switch = (np.arange(state_count)[:, None] != np.arange(state_count)[None, :]).astype(np.float64)
        transition = value[step - 1, :, None] - float(switch_penalty) * switch - float(jump_penalty) * jump
        parent[step] = np.argmax(transition, axis=0)
        value[step] = scores[step] + np.max(transition, axis=0)
    path = np.empty(len(blocks), np.int16)
    path[-1] = int(np.argmax(value[-1]))
    for step in range(len(blocks) - 1, 0, -1):
        path[step - 1] = parent[step, path[step]]
    return path


def _assemble(score_frame, config, wells, shuffle_blocks=False, seed=SEED, score_groups=None):
    prediction = base.copy()
    choices = []
    if score_groups is None:
        score_groups = {
            str(well): local.copy()
            for well, local in score_frame.groupby("well", sort=False)
        }
    for well_number, well in enumerate(wells):
        indices = groups[well]
        local_distance = distance[indices]
        far = local_distance > FAR_START_FT
        blocks_for_rows = np.floor(np.maximum(local_distance - FAR_START_FT, 0.0) / SEGMENT_FT).astype(np.int32)
        local_scores = score_groups.get(well)
        if local_scores is None:
            continue
        if local_scores.empty:
            continue
        ordered_blocks = np.sort(local_scores["block"].unique())
        states = _decode_well(
            local_scores,
            config["base_bias"],
            config["switch_penalty"],
            config["jump_penalty"],
            shuffle_blocks=shuffle_blocks,
            seed=seed + well_number,
        )
        selected = base[indices].copy()
        for block, state in zip(ordered_blocks, states):
            rows = np.flatnonzero(far & (blocks_for_rows == block))
            selected[rows] = candidates[indices[rows], int(state)]
            choices.append({"well": well, "block": int(block), "state": int(state), "candidate": usable_names[int(state)]})
        correction = _centred_mean(local_distance, selected - base[indices], SMOOTH_FT)
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        local_prediction = base[indices] + float(config["dose"]) * fade[indices] * correction
        local_prediction[local_distance <= FAR_START_FT] = base[indices][local_distance <= FAR_START_FT]
        prediction[indices] = local_prediction
    return prediction, pd.DataFrame(choices)


def _scan(scores, folds):
    wells = sorted(features.loc[features["fold"].isin(folds), "well"].unique())
    score_groups = {
        str(well): local.copy()
        for well, local in scores.groupby("well", sort=False)
    }
    row_mask = frame["fold"].isin(folds).to_numpy()
    local = frame.loc[row_mask].reset_index(drop=True)
    base_metric = _metrics(local, base[row_mask], "base")
    reports = []
    for base_bias in BASE_BIASES:
        for switch_penalty in SWITCH_PENALTIES:
            for jump_penalty in JUMP_PENALTIES:
                for dose in DOSES:
                    config = {
                        "base_bias": base_bias,
                        "switch_penalty": switch_penalty,
                        "jump_penalty": jump_penalty,
                        "dose": dose,
                    }
                    prediction, choices = _assemble(scores, config, wells, score_groups=score_groups)
                    metric = _metrics(local, prediction[row_mask], "candidate")
                    reports.append({
                        **config,
                        "rmse": metric["rmse"],
                        "gain": base_metric["rmse"] - metric["rmse"],
                        "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                        "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
                        "switch_rate": float((choices["state"].to_numpy()[1:] != choices["state"].to_numpy()[:-1]).mean()) if len(choices) > 1 else 0.0,
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
    selected, scan = _scan(inner_scores, train_folds)
    config = {key: float(selected[key]) for key in ("base_bias", "switch_penalty", "jump_penalty", "dose")}
    scan["held_fold"] = held_fold
    scan_tables.append(scan)

    model = _fit_model(features[features["fold"].isin(train_folds)])
    model.booster_.save_model(str(MODEL_DIR / f"fold{held_fold}.txt"))
    held = features[features["fold"].eq(held_fold)].copy()
    held["score"] = _predict(model, held)
    held_wells = sorted(held["well"].unique())
    held_groups = {
        str(well): local.copy()
        for well, local in held.groupby("well", sort=False)
    }
    prediction, choices = _assemble(held, config, held_wells, score_groups=held_groups)
    shuffled, _ = _assemble(
        held,
        config,
        held_wells,
        shuffle_blocks=True,
        seed=SEED + held_fold,
        score_groups=held_groups,
    )
    row_mask = frame["fold"].to_numpy(np.int8) == held_fold
    oof_prediction[row_mask] = prediction[row_mask]
    shuffle_prediction[row_mask] = shuffled[row_mask]

    local = frame.loc[row_mask].reset_index(drop=True)
    base_metric = _metrics(local, base[row_mask], "base")
    metric = _metrics(local, oof_prediction[row_mask], "sequence_dp")
    shuffle_metric = _metrics(local, shuffle_prediction[row_mask], "block_order_shuffle")
    choices["fold"] = held_fold
    choice_tables.append(choices)
    fold_reports.append({
        "fold": held_fold,
        **config,
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
    print("V42 FOLD", fold_reports[-1], flush=True)

base_result = _metrics(frame, base, "base_6p377")
dp_result = _metrics(frame, oof_prediction, "nested_sequence_dp")
shuffle_result = _metrics(frame, shuffle_prediction, "block_order_shuffle")
gain = base_result["rmse"] - dp_result["rmse"]
far_gain = base_result["far_gt2000_rmse"] - dp_result["far_gt2000_rmse"]
shuffle_loss = shuffle_result["rmse"] - dp_result["rmse"]
fold_table = pd.DataFrame(fold_reports)
checks = {
    "gain_ge_0p10": bool(gain >= 0.10),
    "minimum_fold_gain_ge_minus0p02": bool(fold_table["gain"].min() >= -0.02),
    "positive_folds_ge_2": bool((fold_table["gain"] > 0).sum() >= 2),
    "far_gain_ge_0p15": bool(far_gain >= 0.15),
    "block_shuffle_loses_70pct_gain": bool(shuffle_loss >= 0.70 * max(gain, 0.0)),
    "maximum_fold_p95_change_le_0p15": bool(fold_table["p95_change"].max() <= 0.15),
    "near_is_bitwise_frozen": bool(np.array_equal(oof_prediction[distance <= 2000.0], base[distance <= 2000.0])),
}
strict_pass = bool(all(checks.values()))

pd.concat(scan_tables, ignore_index=True).to_csv(OUT_DIR / "inner_scans.csv", index=False)
pd.concat(choice_tables, ignore_index=True).to_csv(OUT_DIR / "segment_choices.csv", index=False)
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    base_6p377=base.astype(np.float32),
    nested_sequence_dp=oof_prediction.astype(np.float32),
    block_order_shuffle=shuffle_prediction.astype(np.float32),
).to_parquet(OUT_DIR / "sequence_dp_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_far_endpoint_sequence_dp",
    "sample_unit": "complete-well ordered 256-ft segment sequence",
    "deployable_inputs_only": True,
    "lineage": {
        "base": str(V33_PATH),
        "base_column": BASE_COLUMN,
        "candidate_source": str(V28_PATH),
        "unary_features": str(FEATURE_PATH),
    },
    "candidate_names": usable_names,
    "feature_count": len(feature_names),
    "feature_wells": int(features["well"].nunique()),
    "base": base_result,
    "nested_sequence_dp": dp_result,
    "block_order_shuffle": shuffle_result,
    "gain": gain,
    "far_gain": far_gain,
    "shuffle_loss": shuffle_loss,
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> package the far-only sequence correction for Kaggle"
        if strict_pass
        else "FAIL -> V40 oracle is not reachable from legal endpoint evidence; close endpoint routing"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV42 SUMMARY")
print(pd.DataFrame([base_result, dp_result, shuffle_result]).to_string(index=False))
print("\nV42 FOLDS")
print(fold_table.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default))
print("saved:", OUT_DIR)
