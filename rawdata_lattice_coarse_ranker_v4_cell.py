"""Fold-clean coarse ranker for the frozen V3 raw-data lattice union.

The V3 union reached a 4.556-ft Fold-2 oracle.  This gateway asks whether
deployable complete-path evidence can retain that ceiling before investing in
a neural set reranker.  Candidate generation and training selection are both
label-free; truth is used only as a whole-well ranking label and for metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("Install lightgbm before running V4: pip install lightgbm") from exc


def _bootstrap_v3():
    candidates = [
        Path.cwd() / "rawdata_lattice_mode_union_v3_cell.py",
        Path("/root/autodl-tmp/rawdata_lattice_mode_union_v3_cell.py"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Upload rawdata_lattice_mode_union_v3_cell.py to /root/autodl-tmp"
        )
    text = source.read_text(encoding="utf-8")
    marker = "# --- V3 EXECUTION ---"
    if marker not in text:
        raise RuntimeError(f"bootstrap marker missing from {source}")
    namespace = {"__name__": "raw_lattice_v3_definitions"}
    exec(compile(text.split(marker, 1)[0], str(source), "exec"), namespace)
    return namespace, source


C, V3_SOURCE = _bootstrap_v3()
B = C["B"]
V = C["V"]
DATA_ROOT = C["DATA_ROOT"]
EXPERIMENT = "rawdata_lattice_coarse_ranker_v4"
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OUT_DIR = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT
else:
    OUT_DIR = Path.cwd() / "output" / EXPERIMENT
CACHE_DIR = OUT_DIR / "panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
SEGMENT_LENGTH_FT = 128.0
SEQUENCE_BINS = 12
TRAIN_CANDIDATES_PER_WELL = 768
TOPK_VALUES = (32, 64, 128, 192, 256, 384)
RANDOM_SEED = 20260721

SELECTED_ARM_IDS = (
    "raw_g_b1024_e2",
    "raw_clip_b512_shape4",
    "raw_g_b512_e4",
)
ARM_BY_ID = {arm["id"]: arm for arm in C["ARMS"]}
SELECTED_ARMS = [ARM_BY_ID[arm_id] for arm_id in SELECTED_ARM_IDS]


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _robust_z(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    scale = max(scale, 1e-6)
    return np.clip((values - centre) / scale, -12.0, 12.0)


def _pool_sequence(values, bins=SEQUENCE_BINS):
    values = np.asarray(values, np.float64)
    n_candidates, length = values.shape
    pooled = []
    for indices in np.array_split(np.arange(length), bins):
        if len(indices):
            pooled.append(np.nanmean(values[:, indices], axis=1))
        else:
            pooled.append(np.zeros(n_candidates, dtype=np.float64))
    return np.stack(pooled, axis=1)


def _channel_features(values, prefix):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    safe = np.where(finite, values, np.nan)
    features = [
        np.nanmean(safe, axis=1),
        np.nanstd(safe, axis=1),
        np.nanquantile(safe, 0.50, axis=1),
        np.nanquantile(safe, 0.90, axis=1),
        np.nanmax(safe, axis=1),
        finite.mean(axis=1),
    ]
    names = [
        f"gr_{prefix}_mean",
        f"gr_{prefix}_std",
        f"gr_{prefix}_p50",
        f"gr_{prefix}_p90",
        f"gr_{prefix}_max",
        f"gr_{prefix}_coverage",
    ]
    pooled = _pool_sequence(safe)
    for index in range(pooled.shape[1]):
        features.append(pooled[:, index])
        names.append(f"gr_{prefix}_bin{index:02d}")
    return np.stack(features, axis=1), names


def _candidate_union(record, prior):
    segments = None
    path_rows = {}
    arm_scores = {}
    for arm_index, arm in enumerate(SELECTED_ARMS):
        scores, paths, current_segments = C["_beam_bank_v3"](record, prior, arm)
        if segments is None:
            segments = current_segments
        elif len(segments) != len(current_segments):
            raise RuntimeError(f"{record['well']}: inconsistent segment count")
        order = np.argsort(scores)
        rank = np.empty(len(scores), dtype=np.float64)
        rank[order] = np.arange(len(scores), dtype=np.float64) / max(len(scores) - 1, 1)
        zscore = _robust_z(scores)
        for path_index, path in enumerate(paths):
            key = path.tobytes()
            if key not in path_rows:
                path_rows[key] = {
                    "path": path.copy(),
                    "source_mask": 0,
                    "rank": np.ones(len(SELECTED_ARMS), dtype=np.float32),
                    "score_z": np.full(len(SELECTED_ARMS), 12.0, dtype=np.float32),
                }
            row = path_rows[key]
            row["source_mask"] |= 1 << arm_index
            row["rank"][arm_index] = min(row["rank"][arm_index], rank[path_index])
            row["score_z"][arm_index] = min(
                row["score_z"][arm_index], zscore[path_index]
            )
        arm_scores[arm["id"]] = {
            "generated": int(len(paths)),
            "score_min": float(np.min(scores)),
        }
    rows = list(path_rows.values())
    paths = np.stack([row["path"] for row in rows]).astype(np.int16)
    source_mask = np.asarray([row["source_mask"] for row in rows], np.uint8)
    arm_rank = np.stack([row["rank"] for row in rows]).astype(np.float32)
    arm_score_z = np.stack([row["score_z"] for row in rows]).astype(np.float32)
    return paths, source_mask, arm_rank, arm_score_z, segments, arm_scores


def _candidate_evidence(record, prior, paths, segments, arm_rank, arm_score_z, source_mask):
    rates_grid = prior["rate_grid"]
    n_candidates = len(paths)
    rates = rates_grid[paths]
    all_rows = np.concatenate(segments)
    candidate_tvt_parts = []
    start_surface = np.full(n_candidates, record["anchor_surface"], np.float64)
    start_md = float(record["anchor_md"])
    for segment_index, rows in enumerate(segments):
        rate = rates[:, segment_index]
        delta = record["md"][rows][None, :] - start_md
        surface = start_surface[:, None] + rate[:, None] * delta
        candidate_tvt_parts.append(surface - record["z"][rows][None, :])
        start_surface += rate * float(record["md"][rows][-1] - start_md)
        start_md = float(record["md"][rows][-1])
    candidate_tvt = np.concatenate(candidate_tvt_parts, axis=1)

    raw_reference = V["_interp_nan"](
        candidate_tvt.ravel(), record["type_tvt"], record["type_gr"]
    ).reshape(candidate_tvt.shape)
    shifted_type_gr = B["_reference_for_record"](record, "shift")
    shifted_reference = V["_interp_nan"](
        candidate_tvt.ravel(), record["type_tvt"], shifted_type_gr
    ).reshape(candidate_tvt.shape)
    observed = record["gr"][all_rows][None, :]
    raw_residual = (raw_reference - observed) / record["noise_scale"]
    shift_residual = (shifted_reference - observed) / record["noise_scale"]
    raw_valid = np.isfinite(raw_residual)
    shift_valid = np.isfinite(shift_residual)
    raw_residual[~raw_valid] = np.nan
    shift_residual[~shift_valid] = np.nan

    channels = [
        (raw_residual**2, "raw_sq"),
        (np.minimum(raw_residual**2, 9.0), "raw_clip"),
        (np.log1p(raw_residual**2), "raw_cauchy"),
        (np.abs(raw_residual), "raw_abs"),
        (shift_residual**2, "shift_sq"),
    ]
    feature_blocks = []
    feature_names = []
    for values, name in channels:
        block, names = _channel_features(values, name)
        feature_blocks.append(block)
        feature_names.extend(names)

    rate_change = np.diff(rates, axis=1, prepend=prior["rate_centre"])
    rate_features = np.stack(
        [
            rates.mean(axis=1),
            rates.std(axis=1),
            rates.min(axis=1),
            rates.max(axis=1),
            rates[:, -1],
            np.mean(np.abs(rate_change), axis=1),
            np.max(np.abs(rate_change), axis=1),
            np.mean(rate_change**2, axis=1),
        ],
        axis=1,
    )
    rate_names = [
        "geom_rate_mean",
        "geom_rate_std",
        "geom_rate_min",
        "geom_rate_max",
        "geom_rate_last",
        "geom_change_abs_mean",
        "geom_change_abs_max",
        "geom_change_sq_mean",
    ]
    for matrix, prefix in ((rates, "geom_rate"), (rate_change, "geom_change")):
        pooled = _pool_sequence(matrix)
        rate_features = np.concatenate([rate_features, pooled], axis=1)
        rate_names.extend([f"{prefix}_bin{index:02d}" for index in range(pooled.shape[1])])
    feature_blocks.append(rate_features)
    feature_names.extend(rate_names)

    generation = []
    generation_names = []
    for arm_index, arm_id in enumerate(SELECTED_ARM_IDS):
        generation.extend(
            [
                ((source_mask >> arm_index) & 1).astype(np.float64),
                arm_rank[:, arm_index].astype(np.float64),
                arm_score_z[:, arm_index].astype(np.float64),
            ]
        )
        generation_names.extend(
            [
                f"gen_{arm_id}_member",
                f"gen_{arm_id}_rank",
                f"gen_{arm_id}_score_z",
            ]
        )
    feature_blocks.append(np.stack(generation, axis=1))
    feature_names.extend(generation_names)

    features = np.concatenate(feature_blocks, axis=1)
    features = np.nan_to_num(features, nan=20.0, posinf=20.0, neginf=-20.0)
    features = np.clip(features, -50.0, 100.0).astype(np.float32)
    return features, feature_names


def _exact_targets(record, prior, paths, segments):
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows]
    distance = record["md"][eval_rows] - record["anchor_md"]
    far = distance > 2000.0
    sse = np.empty(len(paths), np.float64)
    sae = np.empty(len(paths), np.float64)
    far_sse = np.empty(len(paths), np.float64)
    for index, path in enumerate(paths):
        prediction = B["_path_prediction"](record, prior, path, segments)[eval_rows]
        error = prediction - truth
        sse[index] = np.sum(error**2)
        sae[index] = np.sum(np.abs(error))
        far_sse[index] = np.sum(error[far] ** 2)
    return {
        "sse": sse,
        "sae": sae,
        "far_sse": far_sse,
        "rows": int(len(eval_rows)),
        "far_rows": int(far.sum()),
        "mse": sse / len(eval_rows),
    }


def _panel_path(well):
    return CACHE_DIR / f"{well}.npz"


def _build_panel(record, prior, force=False):
    path = _panel_path(record["well"])
    if path.exists() and not force:
        return path
    paths, source_mask, arm_rank, arm_score_z, segments, arm_report = _candidate_union(
        record, prior
    )
    features, feature_names = _candidate_evidence(
        record, prior, paths, segments, arm_rank, arm_score_z, source_mask
    )
    targets = _exact_targets(record, prior, paths, segments)
    np.savez_compressed(
        path,
        features=features,
        source_mask=source_mask,
        arm_rank=arm_rank,
        mse=targets["mse"].astype(np.float32),
        sse=targets["sse"],
        sae=targets["sae"],
        far_sse=targets["far_sse"],
        rows=np.asarray([targets["rows"]], np.int64),
        far_rows=np.asarray([targets["far_rows"]], np.int64),
    )
    metadata = {
        "well": record["well"],
        "candidates_after_dedup": int(len(paths)),
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names,
        "arms": arm_report,
    }
    _write_json(path.with_suffix(".json"), metadata)
    return path


def _load_panel(well):
    with np.load(_panel_path(well), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _training_indices(panel, well):
    n = len(panel["mse"])
    deterministic = np.argsort(np.min(panel["arm_rank"], axis=1))[
        : min(TRAIN_CANDIDATES_PER_WELL // 2, n)
    ]
    remaining = np.setdiff1d(np.arange(n), deterministic, assume_unique=False)
    count = min(TRAIN_CANDIDATES_PER_WELL - len(deterministic), len(remaining))
    seed = int.from_bytes(hashlib.sha256(well.encode("ascii")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(remaining, size=count, replace=False) if count else np.empty(0, int)
    return np.unique(np.r_[deterministic, sampled]).astype(np.int64)


def _rank_labels(mse):
    order = np.argsort(mse)
    rank = np.empty(len(mse), np.int64)
    rank[order] = np.arange(len(mse))
    # LightGBM's default lambdarank label_gain maps relevance levels 0..30.
    relevance = 30 - np.floor(31.0 * rank / max(len(mse), 1)).astype(np.int64)
    return np.clip(relevance, 0, 30)


def _matrix(wells):
    x_parts, y_parts, groups = [], [], []
    for well in wells:
        panel = _load_panel(well)
        indices = _training_indices(panel, well)
        x_parts.append(panel["features"][indices])
        y_parts.append(_rank_labels(panel["mse"][indices]))
        groups.append(len(indices))
    return np.concatenate(x_parts), np.concatenate(y_parts), groups


def _score_panel(model, panel, control, seed):
    features = panel["features"].copy()
    gr_columns = np.asarray([name.startswith("gr_") or name.startswith("gen_") for name in FEATURE_NAMES])
    if control == "no_gr":
        features[:, gr_columns] = 0.0
    elif control == "shuffled_alignment":
        rng = np.random.default_rng(seed)
        features[:, gr_columns] = features[rng.permutation(len(features))][:, gr_columns]
    elif control != "full":
        raise ValueError(control)
    iteration = getattr(model, "best_iteration_", None)
    if iteration is not None and int(iteration) <= 0:
        iteration = None
    return model.predict(features, num_iteration=iteration)


def _balanced_topk(scores, source_mask, k):
    selected = []
    selected_set = set()
    quota = int(math.ceil(k / len(SELECTED_ARMS)))
    for arm_index in range(len(SELECTED_ARMS)):
        eligible = np.flatnonzero(((source_mask >> arm_index) & 1) == 1)
        order = eligible[np.argsort(-scores[eligible])]
        for index in order[:quota]:
            if int(index) not in selected_set:
                selected.append(int(index))
                selected_set.add(int(index))
    if len(selected) < k:
        for index in np.argsort(-scores):
            if int(index) not in selected_set:
                selected.append(int(index))
                selected_set.add(int(index))
                if len(selected) >= k:
                    break
    return np.asarray(selected[:k], np.int64)


def _evaluate(model, wells, control="full", topk=None, policy="global"):
    rows = []
    for well in wells:
        panel = _load_panel(well)
        seed = int.from_bytes(hashlib.sha256(f"{control}:{well}".encode()).digest()[:8], "little")
        scores = _score_panel(model, panel, control, seed)
        map_index = int(np.argmax(scores))
        if topk is None:
            selected = np.arange(len(scores))
        elif policy == "global":
            selected = np.argsort(-scores)[: min(topk, len(scores))]
        elif policy == "balanced":
            selected = _balanced_topk(scores, panel["source_mask"], min(topk, len(scores)))
        else:
            raise ValueError(policy)
        oracle_index = int(selected[np.argmin(panel["mse"][selected])])
        rows.append(
            {
                "well": well,
                "rows": int(panel["rows"][0]),
                "far_rows": int(panel["far_rows"][0]),
                "map_sse": float(panel["sse"][map_index]),
                "map_sae": float(panel["sae"][map_index]),
                "map_far_sse": float(panel["far_sse"][map_index]),
                "oracle_sse": float(panel["sse"][oracle_index]),
                "oracle_sae": float(panel["sae"][oracle_index]),
                "oracle_far_sse": float(panel["far_sse"][oracle_index]),
                "candidates": int(len(scores)),
                "retained": int(len(selected)),
            }
        )
    frame = pd.DataFrame(rows)

    def metrics(prefix):
        count = int(frame["rows"].sum())
        far_count = int(frame["far_rows"].sum())
        sse = float(frame[f"{prefix}_sse"].sum())
        well_rmse = np.sqrt(frame[f"{prefix}_sse"] / frame["rows"])
        return {
            "rows": count,
            "wells": int(frame["well"].nunique()),
            "exact_sse": sse,
            "rmse": math.sqrt(sse / count),
            "mae": float(frame[f"{prefix}_sae"].sum() / count),
            "median_well_rmse": float(np.median(well_rmse)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "far_gt2000_rmse": math.sqrt(
                float(frame[f"{prefix}_far_sse"].sum()) / far_count
            ),
        }

    return {"map": metrics("map"), "oracle": metrics("oracle")}, frame


# --- V4 EXECUTION ---
started = time.time()
np.random.seed(RANDOM_SEED)
wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = V["_stable_folds"](wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
train_wells = [well for well in wells if fold_by_well[well] in TRAIN_FOLDS]
selection_wells = [well for well in wells if fold_by_well[well] == SELECTION_FOLD]
application_wells = [well for well in wells if fold_by_well[well] == APPLICATION_FOLD]
needed = train_wells + selection_wells + application_wells

records = {}
failures = []
for number, well in enumerate(needed, start=1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed):
        print(f"V4 load {number}/{len(needed)} failures={len(failures)}", flush=True)
if failures:
    pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
    print(
        "V4 RECORD EXCLUSIONS",
        {"count": len(failures), "examples": failures[:5]},
        flush=True,
    )

# Match V3: input-ineligible short wells are excluded without consulting truth
# values or model performance.  The exclusion list and achieved fold coverage
# remain explicit in the verdict.
train_wells = [well for well in train_wells if well in records]
selection_wells = [well for well in selection_wells if well in records]
application_wells = [well for well in application_wells if well in records]
needed = train_wells + selection_wells + application_wells
coverage = {
    "requested_wells": int(len(fold_by_well) - sum(fold_by_well[w] == UNTOUCHED_FOLD for w in fold_by_well)),
    "eligible_wells": int(len(needed)),
    "excluded_wells": int(len(failures)),
    "train_wells": int(len(train_wells)),
    "selection_wells": int(len(selection_wells)),
    "application_wells": int(len(application_wells)),
}
if len(selection_wells) < 154 or len(application_wells) < 154 or len(train_wells) < 307:
    raise RuntimeError(f"Unexpected eligible fold coverage: {coverage}")

prior = V["_fit_prior"]([records[well] for well in train_wells], SEGMENT_LENGTH_FT)
for number, well in enumerate(needed, start=1):
    _build_panel(records[well], prior)
    if number % 20 == 0 or number == len(needed):
        print(f"V4 panels {number}/{len(needed)}", flush=True)

first_metadata = json.loads(_panel_path(needed[0]).with_suffix(".json").read_text())
FEATURE_NAMES = first_metadata["feature_names"]
_write_json(OUT_DIR / "feature_names.json", FEATURE_NAMES)

x_train, y_train, group_train = _matrix(train_wells)
x_selection, y_selection, group_selection = _matrix(selection_wells)
print(
    "V4 MATRICES",
    {
        "train": list(x_train.shape),
        "selection": list(x_selection.shape),
        "features": len(FEATURE_NAMES),
        "train_wells": len(group_train),
        "selection_wells": len(group_selection),
    },
    flush=True,
)

model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=(1, 8, 32),
    n_estimators=1200,
    learning_rate=0.025,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=RANDOM_SEED,
    n_jobs=max(1, min(16, os.cpu_count() or 1)),
    verbosity=-1,
)
model.fit(
    x_train,
    y_train,
    group=group_train,
    eval_set=[(x_selection, y_selection)],
    eval_group=[group_selection],
    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(25)],
)
model.booster_.save_model(str(OUT_DIR / "coarse_ranker.txt"))

selection_grid = []
for policy in ("global", "balanced"):
    for topk in TOPK_VALUES:
        result, _ = _evaluate(model, selection_wells, "full", topk, policy)
        selection_grid.append(
            {
                "policy": policy,
                "topk": topk,
                "map_rmse": result["map"]["rmse"],
                "oracle_rmse": result["oracle"]["rmse"],
                "oracle_p95": result["oracle"]["p95_well_rmse"],
                "oracle_far": result["oracle"]["far_gt2000_rmse"],
            }
        )
selection_grid = pd.DataFrame(selection_grid)
eligible = selection_grid[selection_grid["oracle_rmse"] <= 4.9]
if len(eligible):
    selected = eligible.sort_values(["topk", "map_rmse"]).iloc[0]
else:
    selected = selection_grid.sort_values(["oracle_rmse", "map_rmse"]).iloc[0]
selected_policy = str(selected["policy"])
selected_topk = int(selected["topk"])
selection_grid.to_csv(OUT_DIR / "fold3_topk_grid.csv", index=False)
print("\nV4 FOLD3 TOPK GRID")
print(selection_grid.sort_values(["oracle_rmse", "topk"]).to_string(index=False), flush=True)

# Fold 3 has now served its only tuning role.  Freeze the iteration count and
# top-K policy, then refit on Fold 3/4/5 before the untouched Fold-2 gateway.
# This recovers most of the available training wells without leaking Fold 2.
fold3_result, fold3_bywell = _evaluate(
    model, selection_wells, "full", selected_topk, selected_policy
)
x_outer = np.concatenate([x_train, x_selection], axis=0)
y_outer = np.concatenate([y_train, y_selection], axis=0)
group_outer = list(group_train) + list(group_selection)
outer_model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=(1, 8, 32),
    n_estimators=max(1, int(model.best_iteration_)),
    learning_rate=0.025,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=RANDOM_SEED,
    n_jobs=max(1, min(16, os.cpu_count() or 1)),
    verbosity=-1,
)
outer_model.fit(x_outer, y_outer, group=group_outer)
outer_model.booster_.save_model(str(OUT_DIR / "coarse_ranker_outer345.txt"))
print(
    "V4 OUTER REFIT",
    {
        "wells": len(group_outer),
        "candidate_rows": len(x_outer),
        "iterations": int(model.best_iteration_),
    },
    flush=True,
)

fold2_full, fold2_bywell = _evaluate(
    outer_model, application_wells, "full", selected_topk, selected_policy
)
fold2_no_gr, _ = _evaluate(
    outer_model, application_wells, "no_gr", selected_topk, selected_policy
)
fold2_shuffle, _ = _evaluate(
    outer_model, application_wells, "shuffled_alignment", selected_topk, selected_policy
)
fold3_bywell.to_csv(OUT_DIR / "fold3_bywell.csv", index=False)
fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

controls = {
    "no_gr_map_degradation": fold2_no_gr["map"]["rmse"] - fold2_full["map"]["rmse"],
    "shuffled_alignment_map_degradation": fold2_shuffle["map"]["rmse"] - fold2_full["map"]["rmse"],
}
gates = {
    "coarse_topk_oracle_rmse_le_4p9": fold2_full["oracle"]["rmse"] <= 4.9,
    "coarse_topk_p95_le_9p5": fold2_full["oracle"]["p95_well_rmse"] <= 9.5,
    "unary_map_rmse_le_8": fold2_full["map"]["rmse"] <= 8.0,
    "each_control_map_degradation_ge_0p5": min(controls.values()) >= 0.5,
}
if not gates["coarse_topk_oracle_rmse_le_4p9"]:
    next_action = "FAIL -> coarse retrieval destroys the V3 ceiling; redesign candidate features"
elif not gates["unary_map_rmse_le_8"]:
    next_action = "RETRIEVAL PASS -> freeze top-K and train the complete-well neural listwise reranker"
else:
    next_action = "PASS -> unary scorer is already competitive; add the frozen listwise reranker"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "foldclean_complete_path_coarse_ranking_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "candidate_union": list(SELECTED_ARM_IDS),
    "coverage": coverage,
    "record_exclusions": failures,
    "feature_count": len(FEATURE_NAMES),
    "selected_policy": selected_policy,
    "selected_topk": selected_topk,
    "best_iteration": int(model.best_iteration_),
    "inner_training_wells": int(len(group_train)),
    "outer_refit_wells": int(len(group_outer)),
    "fold3": fold3_result,
    "fold2_full": fold2_full,
    "fold2_no_gr": fold2_no_gr,
    "fold2_shuffled_alignment": fold2_shuffle,
    "controls": controls,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": next_action,
    "elapsed_minutes": (time.time() - started) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV4 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
