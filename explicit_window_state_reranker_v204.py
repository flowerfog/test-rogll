"""V204: explicit candidate-window reranking of V202 top-64 support.

The V202 network is used only as a high-recall candidate generator. A
LightGBM ranker consumes explicit candidate-conditioned GR/typewell window
statistics and Z-conditioned local registration features. Train Folds 4/5,
select one consumer on 24 Fold3 wells, confirm on the other 130, and only then
reuse Fold2. Fold1 remains untouched.
"""
import copy
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch


STARTED_V204 = time.time()
EXPERIMENT_V204 = "explicit_window_state_reranker_v204"
required_v204 = (
    "model", "train_records", "select_records", "gate_records", "device",
    "_block_nodes", "_inputs", "_logits_for_records", "_row_prediction",
    "_metrics", "STATE_OFFSETS", "REFERENCE_STATE_START", "OUT", "SEED",
)
missing_v204 = [name for name in required_v204 if name not in globals()]
if missing_v204:
    raise RuntimeError(
        "Run V202 in this live kernel first. Missing: " + repr(missing_v204)
    )

OUT_V204 = Path(OUT).parent / EXPERIMENT_V204
OUT_V204.mkdir(parents=True, exist_ok=True)
TOPK_SUPPORT_V204 = 64
RELEVANCE_LEVELS_V204 = 32
WINDOW_RADII_V204 = (8, 16, 32)
REGISTRATION_HALF_WINDOW_V204 = 64.0
REGISTRATION_OFFSETS_V204 = np.linspace(
    -REGISTRATION_HALF_WINDOW_V204, REGISTRATION_HALF_WINDOW_V204, 17
)
S_RATE_PERTURBATIONS_V204 = (-0.03, 0.0, 0.03)


def _clean_v204(value):
    if isinstance(value, dict):
        return {str(key): _clean_v204(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v204(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _restore_v202_v204():
    if "base_state_v203" in globals():
        model.load_state_dict(copy.deepcopy(base_state_v203))
        return "live_base_state_v203"
    checkpoint_path = Path(OUT) / "models" / "fold45_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state", checkpoint))
    model.load_state_dict(state)
    return str(checkpoint_path)


def _safe_corr_v204(left, right):
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 5:
        return 0.0, 8.0
    x = left[finite].astype(np.float64)
    y = right[finite].astype(np.float64)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    correlation = float(np.dot(x, y) / denominator) if denominator > 1e-8 else 0.0
    return correlation, float(np.mean(np.abs(x - y)))


def _candidate_features_v204(item, donor=None):
    record = item["record"]
    rows = item["rows"]
    logits = item["logits"].astype(np.float64)
    typewell = (donor.typewell if donor is not None else record.typewell).astype(np.float64)
    horizontal = record.horizontal.astype(np.float64)
    condition = record.condition.astype(np.float64)
    md = record.md.astype(np.float64)
    z_relative = condition[4] * 512.0
    candidate_indices = np.argpartition(logits, -TOPK_SUPPORT_V204, axis=0)[-TOPK_SUPPORT_V204:]
    feature_rows = []
    metadata_rows = []

    for block, row in enumerate(rows):
        candidates = candidate_indices[:, block]
        candidate_logits = logits[candidates, block]
        order = np.argsort(candidate_logits)[::-1]
        rank = np.empty(len(order), np.int16)
        rank[order] = np.arange(1, len(order) + 1)
        horizontal_query = np.interp(
            md[row] + REGISTRATION_OFFSETS_V204,
            md,
            horizontal[0],
            left=np.nan,
            right=np.nan,
        )
        z_query = np.interp(
            md[row] + np.array([-32.0, 32.0]), md, z_relative
        )
        z_slope = float((z_query[1] - z_query[0]) / 64.0)
        horizontal_stats = []
        for radius in WINDOW_RADII_V204:
            mask = np.abs(md - md[row]) <= radius
            values = horizontal[0, mask]
            horizontal_stats.extend([
                float(np.mean(values)), float(np.std(values)),
                float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75)),
            ])

        for local_index, state_index in enumerate(candidates):
            reference_index = REFERENCE_STATE_START + int(state_index)
            values = [
                float(candidate_logits[local_index]),
                float(rank[local_index]),
                float(candidate_logits[local_index] - np.max(candidate_logits)),
                float(STATE_OFFSETS[state_index] / 128.0),
                float(item["md"][block] / 5000.0),
                float(z_slope),
            ]
            values.extend(condition[:, row].tolist())
            values.extend(horizontal[:, row].tolist())
            values.extend(typewell[:, reference_index].tolist())
            values.extend(
                (typewell[:, reference_index] - horizontal[:, row]).tolist()
            )
            values.extend(horizontal_stats)

            for radius in WINDOW_RADII_V204:
                lo = max(0, reference_index - radius)
                hi = min(typewell.shape[1], reference_index + radius + 1)
                local = typewell[0, lo:hi]
                valid = typewell[1, lo:hi] > 0.5
                local = local[valid]
                if len(local):
                    tw_stats = [
                        float(np.mean(local)), float(np.std(local)),
                        float(np.quantile(local, 0.25)), float(np.quantile(local, 0.75)),
                    ]
                else:
                    tw_stats = [0.0, 0.0, 0.0, 0.0]
                h_stats = horizontal_stats[
                    4 * WINDOW_RADII_V204.index(radius):4 * (WINDOW_RADII_V204.index(radius) + 1)
                ]
                values.extend(tw_stats)
                values.extend((np.asarray(tw_stats) - np.asarray(h_stats)).tolist())

            best_corr = -1.0
            best_mae = 8.0
            best_rate = 0.0
            for s_rate in S_RATE_PERTURBATIONS_V204:
                tvt_rate = -z_slope + s_rate
                reference_positions = reference_index + tvt_rate * REGISTRATION_OFFSETS_V204
                type_query = np.interp(
                    reference_positions,
                    np.arange(typewell.shape[1], dtype=np.float64),
                    typewell[0],
                    left=np.nan,
                    right=np.nan,
                )
                correlation, mae = _safe_corr_v204(horizontal_query, type_query)
                if correlation > best_corr:
                    best_corr, best_mae, best_rate = correlation, mae, tvt_rate
            values.extend([best_corr, best_mae, best_rate])
            feature_rows.append(values)
            metadata_rows.append({
                "well": record.well,
                "block": block,
                "state_index": int(state_index),
                "state_offset": float(STATE_OFFSETS[state_index]),
                "label": float(item["label"][block]),
            })

    return np.asarray(feature_rows, np.float32), pd.DataFrame(metadata_rows)


def _build_table_v204(items, rolled=False):
    features = []
    metadata = []
    for index, item in enumerate(items):
        donor = None
        if rolled:
            donor = items[(index + max(1, len(items) // 2)) % len(items)]["record"]
        local_features, local_metadata = _candidate_features_v204(item, donor=donor)
        features.append(local_features)
        metadata.append(local_metadata)
        if (index + 1) % 50 == 0 or index + 1 == len(items):
            print(
                f"V204 features rolled={rolled} {index + 1}/{len(items)}",
                flush=True,
            )
    return np.concatenate(features), pd.concat(metadata, ignore_index=True)


def _groups_v204(metadata):
    return metadata.groupby(["well", "block"], sort=False).size().to_numpy(np.int32)


def _consume_v204(items, metadata, scores, recipe):
    lookup = metadata.copy()
    lookup["score"] = scores
    offsets_by_well = {}
    for (well, block), local in lookup.groupby(["well", "block"], sort=False):
        local = local.nlargest(int(recipe["topk"]), "score")
        if recipe["mode"] == "hard":
            offset = float(local.iloc[0]["state_offset"])
        elif recipe["mode"] == "median":
            offset = float(np.median(local["state_offset"]))
        else:
            score = local["score"].to_numpy(np.float64) / float(recipe["temperature"])
            score -= np.max(score)
            weight = np.exp(score)
            offset = float(np.dot(weight, local["state_offset"]) / np.sum(weight))
        offsets_by_well.setdefault(well, {})[int(block)] = offset

    parts = []
    for item in items:
        record = item["record"]
        mapping = offsets_by_well[record.well]
        offsets = np.asarray([mapping[index] for index in range(len(item["rows"]))])
        prediction = _row_prediction(record, item["rows"], offsets)
        eval_rows = record.eval_index
        parts.append({
            "well": record.well,
            "row_idx": eval_rows.astype(np.int64),
            "truth": record.truth[eval_rows].astype(np.float64),
            "prediction": prediction,
            "distance": record.md[eval_rows].astype(np.float64) - float(record.md[record.anchor_index]),
        })
    return parts


checkpoint_source_v204 = _restore_v202_v204()
train_items_v204 = _logits_for_records(model, train_records, device, mode="full")
select_items_v204 = _logits_for_records(model, select_records, device, mode="full")
train_x_v204, train_meta_v204 = _build_table_v204(train_items_v204)
select_x_v204, select_meta_v204 = _build_table_v204(select_items_v204)
select_rolled_x_v204, _ = _build_table_v204(select_items_v204, rolled=True)

absolute_error_v204 = np.abs(
    train_meta_v204["state_offset"].to_numpy(np.float64)
    - train_meta_v204["label"].to_numpy(np.float64)
)
train_y_v204 = np.clip(
    RELEVANCE_LEVELS_V204 - 1 - np.floor(absolute_error_v204),
    0,
    RELEVANCE_LEVELS_V204 - 1,
).astype(np.int32)
ranker_v204 = lgb.LGBMRanker(
    objective="lambdarank",
    n_estimators=500,
    learning_rate=0.035,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_lambda=2.0,
    random_state=int(SEED) + 204,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1,
    label_gain=list(range(RELEVANCE_LEVELS_V204)),
)
ranker_v204.fit(train_x_v204, train_y_v204, group=_groups_v204(train_meta_v204))
select_score_v204 = ranker_v204.predict(select_x_v204)
rolled_score_v204 = ranker_v204.predict(select_rolled_x_v204)

select_wells_v204 = sorted(select_meta_v204["well"].unique())
discovery_wells_v204 = set(select_wells_v204[:24])
recipes_v204 = [{"mode": "hard", "topk": 1, "temperature": 1.0}]
recipes_v204 += [
    {"mode": mode, "topk": topk, "temperature": temperature}
    for mode in ("median", "soft")
    for topk in (3, 5, 9)
    for temperature in ((1.0,) if mode == "median" else (0.25, 0.5, 1.0))
]

recipe_rows_v204 = []
for recipe_v204 in recipes_v204:
    parts_v204 = _consume_v204(
        select_items_v204, select_meta_v204, select_score_v204, recipe_v204
    )
    discovery_parts_v204 = [part for part in parts_v204 if part["well"] in discovery_wells_v204]
    metrics_v204 = _metrics(discovery_parts_v204, "fold3_discovery")
    recipe_rows_v204.append({**recipe_v204, **metrics_v204})
recipe_scan_v204 = pd.DataFrame(recipe_rows_v204).sort_values("rmse", ignore_index=True)
recipe_scan_v204.to_csv(OUT_V204 / "fold3_discovery_scan.csv", index=False)
selected_recipe_v204 = {
    "mode": str(recipe_scan_v204.iloc[0]["mode"]),
    "topk": int(recipe_scan_v204.iloc[0]["topk"]),
    "temperature": float(recipe_scan_v204.iloc[0]["temperature"]),
}

selected_parts_v204 = _consume_v204(
    select_items_v204, select_meta_v204, select_score_v204, selected_recipe_v204
)
rolled_parts_v204 = _consume_v204(
    select_items_v204, select_meta_v204, rolled_score_v204, selected_recipe_v204
)
confirmation_parts_v204 = [part for part in selected_parts_v204 if part["well"] not in discovery_wells_v204]
rolled_confirmation_v204 = [part for part in rolled_parts_v204 if part["well"] not in discovery_wells_v204]
confirmation_v204 = _metrics(confirmation_parts_v204, "fold3_confirmation")
rolled_v204 = _metrics(rolled_confirmation_v204, "rolled_typewell_control")

fold2_v204 = None
confirmation_pass_v204 = confirmation_v204["rmse"] <= 9.5
if confirmation_pass_v204:
    gate_items_v204 = _logits_for_records(model, gate_records, device, mode="full")
    gate_x_v204, gate_meta_v204 = _build_table_v204(gate_items_v204)
    gate_score_v204 = ranker_v204.predict(gate_x_v204)
    gate_parts_v204 = _consume_v204(
        gate_items_v204, gate_meta_v204, gate_score_v204, selected_recipe_v204
    )
    fold2_v204 = _metrics(gate_parts_v204, "fold2_confirmation")

checks_v204 = {
    "fold3_confirmation_rmse_le_9p5": confirmation_v204["rmse"] <= 9.5,
    "rolled_typewell_degradation_ge_0p30": rolled_v204["rmse"] - confirmation_v204["rmse"] >= 0.30,
    "fold2_rmse_le_9p5": fold2_v204 is not None and fold2_v204["rmse"] <= 9.5,
}
strict_pass_v204 = bool(all(checks_v204.values()))
verdict_v204 = {
    "experiment": EXPERIMENT_V204,
    "stage": "Fold45_explicit_state_ranking_Fold3_discovery_confirmation_Fold2_reconfirmation",
    "candidate_generator": "frozen V202 top64",
    "checkpoint_source": checkpoint_source_v204,
    "changed_information": "multiscale candidate GR windows and Z-conditioned local registration",
    "training_candidates": len(train_meta_v204),
    "features": int(train_x_v204.shape[1]),
    "protocol": {
        "train_folds": [4, 5],
        "fold3_discovery_wells": 24,
        "fold3_confirmation_wells": 130,
        "reused_confirmation_fold": 2 if fold2_v204 is not None else None,
        "untouched_fold": 1,
    },
    "selected_recipe": selected_recipe_v204,
    "fold3_confirmation": confirmation_v204,
    "rolled_typewell_control": rolled_v204,
    "fold2": fold2_v204,
    "checks": checks_v204,
    "strict_pass": strict_pass_v204,
    "elapsed_minutes": (time.time() - STARTED_V204) / 60.0,
    "next": (
        "PASS -> crossfit V204 and blend against V168"
        if strict_pass_v204
        else "FAIL -> explicit local registration cannot rank V202 support; close this cost-map family"
    ),
}
(OUT_V204 / "verdict.json").write_text(
    json.dumps(_clean_v204(verdict_v204), indent=2, allow_nan=False), encoding="utf-8"
)
ranker_v204.booster_.save_model(str(OUT_V204 / "state_ranker.txt"))

print("\nV204 DISCOVERY TOP")
print(recipe_scan_v204[["mode", "topk", "temperature", "rmse", "median_well_rmse", "p95_well_rmse", "far_gt2000_rmse"]].head(15).to_string(index=False))
print("\nV204 FINAL VERDICT")
print(json.dumps(_clean_v204(verdict_v204), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V204, flush=True)
