"""V107: parent-conditioned block residual lattice gateway.

This is deliberately not another complete-well ranker.  It freezes the V76
parent, converts the V4 candidate bank into local moves relative to that
parent, and learns the legal 128-ft evidence for the exact block SSE gain.
A small Viterbi decoder can select a candidate move or stay on the parent in
every block.  Fold45 trains, the first 24 Fold3 wells select decoder constants,
and the remaining Fold3 wells are a read-only confirmation.  Fold2/Fold1 are
not touched by this gateway.

Required AutoDL artifacts:
  /root/autodl-tmp/rawdata_lattice_structural_risk_ranker_v81b_cell.py
  V76 weights listed below and the raw dataset used by V81/V4.
Optional reusable panels (missing Fold45 panels are rebuilt as compact top64):
  /root/autodl-tmp/oof_exports/rawdata_lattice_blendaware_ranker_v103/matched_panels
Optional shortlist model (a legal three-arm rank fallback is used if absent):
  /root/autodl-tmp/oof_exports/rawdata_lattice_coarse_ranker_v4/coarse_ranker.txt
  /root/autodl-tmp/oof_exports/rawdata_lattice_coarse_adaptive_dose_bundle_v76/
      fixed_ensemble_weights_oof.parquet
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import types
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_parent_block_residual_v107"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
SHORTLIST_DIR = OUT_DIR / "compact_top64_panels"
for directory in (OUT_DIR, MODEL_DIR, SHORTLIST_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V81_SOURCE = ROOT / "rawdata_lattice_structural_risk_ranker_v81b_cell.py"
V103_PANELS = (
    OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103" / "matched_panels"
)
V4_MODEL = OOF_ROOT / "rawdata_lattice_coarse_ranker_v4" / "coarse_ranker.txt"
V76_WEIGHTS = (
    OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76"
    / "fixed_ensemble_weights_oof.parquet"
)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
DISCOVERY_WELLS = 24
BLOCK_FT = 128.0
TOPK = 64
RANDOM_SEED = 20260801
PARENT = "incumbent_6p377_replay"
TRUTH = "tvt_true"
TRAIN_LABEL_DOSE = 0.30

# Predicted gain is in ft^2 per row.  Threshold is therefore interpretable as
# the minimum expected per-row MSE reduction before leaving the V76 parent.
GAIN_THRESHOLDS = (0.0, 0.25, 0.50, 1.00)
CONTINUITY_WEIGHTS = (0.0, 0.03, 0.10, 0.30)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _bootstrap_v81():
    source = V81_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V81B EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V81b bootstrap marker missing from {V81_SOURCE}")
    module = types.ModuleType("raw_lattice_v81_v107_definitions")
    module.__file__ = str(V81_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V81_SOURCE), "exec"), module.__dict__)
    return module


for required in (V81_SOURCE, V76_WEIGHTS):
    if not required.exists():
        raise FileNotFoundError(required)

N = _bootstrap_v81()
Q, V, B = N.Q, N.V, N.B
prior = N.prior
incumbent = N.incumbent
fold_by_well = N.fold_by_well
global_feature_names = list(N.feature_names)

weights = pd.read_parquet(V76_WEIGHTS)
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))
base_ranker = lgb.Booster(model_file=str(V4_MODEL)) if V4_MODEL.is_file() else None
SHORTLIST_SOURCE = "v4_coarse_ranker" if base_ranker is not None else "legal_three_arm_rank"
if base_ranker is None:
    print(
        "V107 SHORTLIST FALLBACK",
        {"missing": str(V4_MODEL), "source": SHORTLIST_SOURCE},
        flush=True,
    )


def _panel_path(well):
    return V103_PANELS / f"{well}.npz"


def _load_panel(well):
    source = _panel_path(well)
    compact = SHORTLIST_DIR / f"{well}.npz"
    if source.is_file():
        with np.load(source, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    if compact.is_file():
        with np.load(compact, allow_pickle=False) as data:
            panel = {key: data[key] for key in data.files}
        if (
            panel["features"].shape[1] == len(global_feature_names)
            and len(panel["features"]) <= TOPK
        ):
            return panel

    # V103 panels may have been removed during disk cleanup.  Reconstruct the
    # legal bank, immediately reduce it to top64, and persist only that compact
    # subset so interrupted runs can resume without restoring the large cache.
    record = V["_load_record"](well)
    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(
        record, prior
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        features, names = Q._candidate_evidence(
            record, prior, paths, segments, arm_rank, arm_score_z, source_mask
        )
    if list(names) != global_feature_names:
        raise RuntimeError(f"{well}: rebuilt feature schema mismatch")
    scores = _shortlist_scores(features, arm_rank, source_mask)
    chosen = Q._balanced_topk(scores, source_mask, min(TOPK, len(paths)))
    panel = {
        "features": features[chosen].astype(np.float32),
        "paths": paths[chosen].astype(np.int16),
        "source_mask": source_mask[chosen].astype(np.uint8),
        "arm_rank": arm_rank[chosen].astype(np.float32),
    }
    np.savez_compressed(compact, **panel)
    return panel


def _shortlist_scores(features, arm_rank, source_mask):
    if base_ranker is not None:
        return np.asarray(base_ranker.predict(features), np.float64)
    source_count = np.unpackbits(
        np.asarray(source_mask, np.uint8)[:, None], axis=1
    ).sum(axis=1)
    return -np.min(np.asarray(arm_rank, np.float64), axis=1) + 0.01 * source_count


available = {
    well for well, fold in fold_by_well.items()
    if fold in (3, 4, 5)
}
coverage = {
    str(fold): sum(well in available for well, value in fold_by_well.items() if value == fold)
    for fold in range(1, 6)
}
if coverage["3"] < 153 or coverage["4"] < 153 or coverage["5"] < 153:
    raise RuntimeError(f"Incomplete V76/fold coverage: {coverage}")


def _safe_mean(values, axis=1):
    finite = np.isfinite(values)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return total / np.maximum(count, 1)


def _safe_std(values):
    mean = _safe_mean(values)
    centred = np.where(np.isfinite(values), values - mean[:, None], 0.0)
    count = np.isfinite(values).sum(axis=1)
    return np.sqrt((centred * centred).sum(axis=1) / np.maximum(count, 1))


def _row_correlation(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1)
    left_mean = np.where(valid, left, 0.0).sum(axis=1) / np.maximum(count, 1)
    right_mean = np.where(valid, right, 0.0).sum(axis=1) / np.maximum(count, 1)
    left_c = np.where(valid, left - left_mean[:, None], 0.0)
    right_c = np.where(valid, right - right_mean[:, None], 0.0)
    numerator = (left_c * right_c).sum(axis=1)
    denominator = np.sqrt((left_c * left_c).sum(axis=1) * (right_c * right_c).sum(axis=1))
    output = numerator / np.maximum(denominator, 1e-9)
    output[count < 4] = 0.0
    return np.clip(output, -1.0, 1.0)


LOCAL_FEATURE_NAMES = [
    "block_fraction", "distance_start_kft", "distance_end_kft", "block_rows",
    "v76_dose", "base_rank_score", "base_rank_fraction",
    "move_mean", "move_std", "move_abs_mean", "move_start", "move_end",
    "move_slope", "raw_gr_abs", "raw_gr_clip", "raw_gr_cauchy",
    "raw_gr_corr", "blend_gr_abs", "blend_gr_clip", "blend_gr_cauchy",
    "blend_gr_corr", "parent_gr_abs", "gr_abs_gain_vs_parent",
    "gr_clip_gain_vs_parent", "path_rate", "path_rate_change",
]
FEATURE_NAMES = global_feature_names + LOCAL_FEATURE_NAMES


def _candidate_predictions(record, paths, segments):
    eval_rows = record["eval_rows"]
    output = np.empty((len(paths), len(eval_rows)), np.float32)
    for index, path in enumerate(paths):
        output[index] = B["_path_prediction"](record, prior, path, segments)[eval_rows]
    if not np.isfinite(output).all():
        raise RuntimeError(f"{record['well']}: non-finite candidate predictions")
    return output


def _well_context(well, labelled, baseline_override=None):
    panel = _load_panel(well)
    features_all = panel["features"].astype(np.float32, copy=False)
    paths_all = panel["paths"].astype(np.int16, copy=False)
    if features_all.shape[1] != len(global_feature_names) or len(features_all) != len(paths_all):
        raise RuntimeError(f"{well}: panel schema mismatch")

    scores_all = _shortlist_scores(
        features_all, panel["arm_rank"], panel["source_mask"]
    )
    chosen = Q._balanced_topk(scores_all, panel["source_mask"], min(TOPK, len(paths_all)))
    features = features_all[chosen]
    paths = paths_all[chosen]
    scores = np.asarray(scores_all[chosen], np.float64)
    score_order = np.argsort(-scores)
    score_rank = np.empty(len(scores), np.float64)
    score_rank[score_order] = np.arange(len(scores), dtype=np.float64)
    score_rank /= max(len(scores) - 1, 1)

    record = V["_load_record"](well)
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    if paths.shape[1] != len(segments):
        raise RuntimeError(f"{well}: path/segment mismatch {paths.shape[1]}/{len(segments)}")
    candidate = _candidate_predictions(record, paths, segments).astype(np.float64)

    local = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    lineage_parent = local[PARENT].to_numpy(np.float64)
    parent = (
        lineage_parent.copy()
        if baseline_override is None
        else np.asarray(baseline_override, np.float64).copy()
    )
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    observed_gr = record["gr"][eval_rows].astype(np.float64)
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: incumbent/evaluation rows {len(local)}/{len(eval_rows)}")
    if np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: truth alignment")
    if np.max(np.abs(local["md_since"].to_numpy(np.float64) - distance)) > 1.01:
        raise RuntimeError(f"{well}: distance alignment")

    # V76 has honest OOF doses only for Fold1/2/3.  Fold45 labels retain the
    # preregistered 0.30 deployment dose used by V102/V103; Fold3 evaluation
    # always uses its actual cross-fitted V76 dose.
    if well in weight_map:
        dose = float(weight_map[well])
        dose_source = "v76_oof"
    elif int(fold_by_well[well]) in TRAIN_FOLDS:
        dose = float(TRAIN_LABEL_DOSE)
        dose_source = "fixed_train_label"
    else:
        raise RuntimeError(f"{well}: missing V76 OOF dose")
    if len(parent) != len(lineage_parent) or not np.isfinite(parent).all():
        raise RuntimeError(f"{well}: invalid baseline override")
    # Every candidate endpoint follows the deployed coarse formula.  The
    # optional baseline only changes the stay state and gain target.
    blended = lineage_parent[None, :] + dose * (candidate - lineage_parent[None, :])
    block_id = np.floor(np.maximum(distance, 0.0) / BLOCK_FT).astype(np.int32)
    block_values = np.unique(block_id)
    parent_reference = V["_interp_nan"](parent, record["type_tvt"], record["type_gr"])
    parent_gr = (parent_reference - observed_gr) / float(record["noise_scale"])
    rate_grid = np.asarray(prior["rate_grid"], np.float64)
    parent_error = parent - truth

    x_parts, y_parts, row_counts = [], [], []
    move_start, move_end = [], []
    actual_gain = []
    block_rows = []
    for block_position, identifier in enumerate(block_values):
        rows = np.flatnonzero(block_id == identifier)
        if len(rows) < 8:
            continue
        block_rows.append(rows)
        raw = candidate[:, rows]
        blend = blended[:, rows]
        move = blend - parent[rows][None, :]
        observed = np.broadcast_to(observed_gr[rows][None, :], raw.shape)
        raw_reference = V["_interp_nan"](
            raw.ravel(), record["type_tvt"], record["type_gr"]
        ).reshape(raw.shape)
        blend_reference = V["_interp_nan"](
            blend.ravel(), record["type_tvt"], record["type_gr"]
        ).reshape(blend.shape)
        raw_residual = (raw_reference - observed) / float(record["noise_scale"])
        blend_residual = (blend_reference - observed) / float(record["noise_scale"])
        parent_residual = parent_gr[rows]
        parent_abs = float(np.nanmean(np.abs(parent_residual)))
        parent_clip = float(np.nanmean(np.minimum(parent_residual * parent_residual, 9.0)))

        raw_abs = _safe_mean(np.abs(raw_residual))
        raw_clip = _safe_mean(np.minimum(raw_residual * raw_residual, 9.0))
        blend_abs = _safe_mean(np.abs(blend_residual))
        blend_clip = _safe_mean(np.minimum(blend_residual * blend_residual, 9.0))
        segment_index = min(block_position, paths.shape[1] - 1)
        rates = rate_grid[paths[:, segment_index]]
        if segment_index:
            previous_rates = rate_grid[paths[:, segment_index - 1]]
        else:
            previous_rates = np.full(len(paths), float(prior["rate_centre"]))
        local_features = np.column_stack([
            np.full(len(paths), block_position / max(len(block_values) - 1, 1)),
            np.full(len(paths), distance[rows[0]] / 1000.0),
            np.full(len(paths), distance[rows[-1]] / 1000.0),
            np.full(len(paths), len(rows)),
            np.full(len(paths), dose),
            scores,
            score_rank,
            _safe_mean(move),
            _safe_std(move),
            _safe_mean(np.abs(move)),
            move[:, 0],
            move[:, -1],
            (move[:, -1] - move[:, 0]) / max(distance[rows[-1]] - distance[rows[0]], 1.0),
            raw_abs,
            raw_clip,
            _safe_mean(np.log1p(raw_residual * raw_residual)),
            _row_correlation(raw_reference, observed),
            blend_abs,
            blend_clip,
            _safe_mean(np.log1p(blend_residual * blend_residual)),
            _row_correlation(blend_reference, observed),
            np.full(len(paths), parent_abs),
            parent_abs - blend_abs,
            parent_clip - blend_clip,
            rates,
            rates - previous_rates,
        ]).astype(np.float32)
        matrix = np.concatenate([features, local_features], axis=1)
        matrix[~np.isfinite(matrix)] = np.nan
        x_parts.append(matrix)

        base_e2 = parent_error[rows] ** 2
        candidate_e2 = (blend - truth[rows][None, :]) ** 2
        gain = base_e2.mean() - candidate_e2.mean(axis=1)
        actual_gain.append(gain.astype(np.float32))
        if labelled:
            y_parts.append(gain.astype(np.float32))
        row_counts.append(len(rows))
        move_start.append(move[:, 0].astype(np.float32))
        move_end.append(move[:, -1].astype(np.float32))

    if not x_parts:
        raise RuntimeError(f"{well}: no usable blocks")
    return {
        "well": well,
        "local": local,
        "parent": parent,
        "truth": truth,
        "distance": distance,
        "candidate": blended.astype(np.float32),
        "block_rows": block_rows,
        "x": x_parts,
        "y": y_parts,
        "actual_gain": actual_gain,
        "row_counts": np.asarray(row_counts, np.float64),
        "move_start": move_start,
        "move_end": move_end,
        "dose": dose,
        "dose_source": dose_source,
    }


# --- V107 EXECUTION ---

train_wells = sorted(w for w, fold in fold_by_well.items() if fold in TRAIN_FOLDS and w in available)
fold3_wells = sorted(w for w, fold in fold_by_well.items() if fold == SELECTION_FOLD and w in available)
if len(train_wells) < 306 or len(fold3_wells) < 153:
    raise RuntimeError(f"Unexpected coverage train={len(train_wells)} fold3={len(fold3_wells)}")

print(
    "V107 CONFIG",
    {
        "train_wells": len(train_wells), "fold3_wells": len(fold3_wells),
        "block_ft": BLOCK_FT, "topk": TOPK, "features": len(FEATURE_NAMES),
        "shortlist_source": SHORTLIST_SOURCE,
        "train_label_dose": TRAIN_LABEL_DOSE,
    },
    flush=True,
)

x_train, y_train, sample_weight = [], [], []
train_report = []
for number, well in enumerate(train_wells, 1):
    try:
        context = _well_context(well, labelled=True)
        for matrix, target, rows in zip(context["x"], context["y"], context["row_counts"]):
            x_train.append(matrix)
            y_train.append(target)
            sample_weight.append(np.full(len(target), rows, np.float32))
        train_report.append({"well": well, "status": "built", "blocks": len(context["x"])})
    except Exception as exc:
        train_report.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 20 == 0 or number == len(train_wells):
        failures = sum(row["status"] == "failed" for row in train_report)
        print(
            f"V107 train blocks {number}/{len(train_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

train_report_frame = pd.DataFrame(train_report)
train_report_frame.to_csv(OUT_DIR / "train_build_report.csv", index=False)
if sum(row["status"] == "built" for row in train_report) < 300:
    raise RuntimeError(f"Insufficient V107 training wells: {train_report_frame['status'].value_counts().to_dict()}")

x_train = np.concatenate(x_train).astype(np.float32, copy=False)
y_train = np.concatenate(y_train).astype(np.float32, copy=False)
sample_weight = np.concatenate(sample_weight).astype(np.float32, copy=False)
clip_low, clip_high = np.quantile(y_train, [0.005, 0.995])
y_train = np.clip(y_train, clip_low, clip_high)
print(
    "V107 MATRIX",
    {
        "rows": len(x_train), "features": x_train.shape[1],
        "target_p005": float(clip_low), "target_p995": float(clip_high),
    },
    flush=True,
)

model = lgb.LGBMRegressor(
    objective="huber", alpha=0.85, n_estimators=450, learning_rate=0.035,
    num_leaves=31, max_depth=-1, min_child_samples=160,
    subsample=0.85, colsample_bytree=0.80,
    reg_alpha=0.5, reg_lambda=8.0,
    random_state=RANDOM_SEED,
    n_jobs=max(1, min(16, os.cpu_count() or 1)), verbosity=-1,
)
model.fit(x_train, y_train, sample_weight=sample_weight)
model.booster_.save_model(str(MODEL_DIR / "fold45_block_gain.txt"))
pd.DataFrame({
    "feature": FEATURE_NAMES,
    "gain": model.booster_.feature_importance(importance_type="gain"),
}).sort_values("gain", ascending=False).to_csv(OUT_DIR / "feature_importance.csv", index=False)
del x_train, y_train, sample_weight
gc.collect()


def _prepare_eval(well):
    context = _well_context(well, labelled=False)
    context["predicted_gain"] = [model.predict(matrix) for matrix in context["x"]]
    del context["x"], context["y"]
    return context


prepared = []
eval_report = []
for number, well in enumerate(fold3_wells, 1):
    try:
        prepared.append(_prepare_eval(well))
        eval_report.append({"well": well, "status": "built"})
    except Exception as exc:
        eval_report.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 10 == 0 or number == len(fold3_wells):
        failures = sum(row["status"] == "failed" for row in eval_report)
        print(
            f"V107 prepare Fold3 {number}/{len(fold3_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )
pd.DataFrame(eval_report).to_csv(OUT_DIR / "fold3_build_report.csv", index=False)
if len(prepared) < 150:
    raise RuntimeError(f"Only {len(prepared)} prepared Fold3 wells")


def _decode(context, threshold, continuity, oracle=False):
    n_blocks = len(context["block_rows"])
    n_candidates = len(context["predicted_gain"][0])
    stay = n_candidates
    back = np.full((n_blocks, n_candidates + 1), -1, np.int16)
    score = np.zeros(n_candidates + 1, np.float64)

    for block in range(n_blocks):
        gain = (
            np.asarray(context["actual_gain"][block], np.float64)
            if oracle else np.asarray(context["predicted_gain"][block], np.float64)
        )
        emission = np.r_[(gain - float(threshold)) * context["row_counts"][block], 0.0]
        if block == 0:
            score = emission
            continue
        previous_end = np.r_[context["move_end"][block - 1], 0.0]
        current_start = np.r_[context["move_start"][block], 0.0]
        transition = (
            float(continuity) * 32.0
            * (previous_end[:, None] - current_start[None, :]) ** 2
        )
        values = score[:, None] - transition
        best_previous = np.argmax(values, axis=0)
        score = values[best_previous, np.arange(n_candidates + 1)] + emission
        back[block] = best_previous.astype(np.int16)

    state = int(np.argmax(score))
    states = np.empty(n_blocks, np.int16)
    states[-1] = state
    for block in range(n_blocks - 1, 0, -1):
        state = int(back[block, state])
        states[block - 1] = state

    prediction = context["parent"].copy()
    changed_rows = 0
    for block, (rows, state) in enumerate(zip(context["block_rows"], states)):
        if int(state) != stay:
            prediction[rows] = context["candidate"][int(state), rows]
            changed_rows += len(rows)
    return prediction, states, changed_rows


def _evaluate(contexts, threshold, continuity, stage, oracle=False):
    parts = []
    changed_wells = 0
    changed_rows = 0
    switches = []
    for context in contexts:
        prediction, states, local_changed = _decode(
            context, threshold, continuity, oracle=oracle
        )
        local = context["local"].copy()
        local["prediction"] = prediction.astype(np.float32)
        parts.append(local)
        changed_wells += int(local_changed > 0)
        changed_rows += local_changed
        switches.append(int(np.sum(states[1:] != states[:-1])))
    frame = pd.concat(parts, ignore_index=True)
    metric = N._metric(frame, frame["prediction"], "block_oracle" if oracle else "block_dp")
    metric.update({
        "stage": stage,
        "threshold": float(threshold),
        "continuity": float(continuity),
        "changed_wells": int(changed_wells),
        "changed_rows_fraction": float(changed_rows / len(frame)),
        "mean_switches": float(np.mean(switches)),
    })
    return metric, frame


discovery_names = set(fold3_wells[:DISCOVERY_WELLS])
discovery_contexts = [item for item in prepared if item["well"] in discovery_names]
confirmation_contexts = [item for item in prepared if item["well"] not in discovery_names]

parent_discovery_frame = pd.concat([item["local"] for item in discovery_contexts], ignore_index=True)
parent_confirmation_frame = pd.concat([item["local"] for item in confirmation_contexts], ignore_index=True)
parent_discovery = {"stage": "fold3_discovery", "threshold": None, "continuity": None,
                    "changed_wells": 0, "changed_rows_fraction": 0.0, "mean_switches": 0.0,
                    **N._metric(parent_discovery_frame, parent_discovery_frame[PARENT], "v76")}
parent_confirmation = {"stage": "fold3_confirmation", "threshold": None, "continuity": None,
                       "changed_wells": 0, "changed_rows_fraction": 0.0, "mean_switches": 0.0,
                       **N._metric(parent_confirmation_frame, parent_confirmation_frame[PARENT], "v76")}

scan_rows = [parent_discovery]
scan_frames = {}
for threshold in GAIN_THRESHOLDS:
    for continuity in CONTINUITY_WEIGHTS:
        row, frame = _evaluate(
            discovery_contexts, threshold, continuity, "fold3_discovery"
        )
        row["gain"] = float(parent_discovery["rmse"] - row["rmse"])
        row["far_gain"] = float(parent_discovery["far_gt2000_rmse"] - row["far_gt2000_rmse"])
        row["near_degradation"] = float(row["near_0_500_rmse"] - parent_discovery["near_0_500_rmse"])
        row["p95_gain"] = float(parent_discovery["p95_well_rmse"] - row["p95_well_rmse"])
        scan_rows.append(row)
        scan_frames[(threshold, continuity)] = frame

scan = pd.DataFrame(scan_rows)
eligible = scan[
    scan["threshold"].notna()
    & (scan["near_degradation"] <= 0.05)
    & (scan["p95_gain"] >= -0.50)
].sort_values(["rmse", "far_gt2000_rmse"])
if eligible.empty:
    candidates = scan[scan["threshold"].notna()].sort_values("rmse")
    selected_reason = "best_rmse_no_risk_eligible_config"
else:
    candidates = eligible
    selected_reason = "best_rmse_with_discovery_risk_constraints"
selected = candidates.iloc[0]
selected_threshold = float(selected["threshold"])
selected_continuity = float(selected["continuity"])

oracle_discovery, _ = _evaluate(
    discovery_contexts, 0.0, selected_continuity, "fold3_discovery", oracle=True
)
confirmation, confirmation_frame = _evaluate(
    confirmation_contexts, selected_threshold, selected_continuity, "fold3_confirmation"
)
oracle_confirmation, _ = _evaluate(
    confirmation_contexts, 0.0, selected_continuity, "fold3_confirmation", oracle=True
)
confirmation["gain"] = float(parent_confirmation["rmse"] - confirmation["rmse"])
confirmation["far_gain"] = float(
    parent_confirmation["far_gt2000_rmse"] - confirmation["far_gt2000_rmse"]
)
confirmation["near_degradation"] = float(
    confirmation["near_0_500_rmse"] - parent_confirmation["near_0_500_rmse"]
)
confirmation["p95_gain"] = float(
    parent_confirmation["p95_well_rmse"] - confirmation["p95_well_rmse"]
)

summary = pd.concat([
    scan,
    pd.DataFrame([parent_confirmation, confirmation]),
], ignore_index=True)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
confirmation_frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, "prediction"]].to_parquet(
    OUT_DIR / "fold3_confirmation_oof.parquet", index=False
)

checks = {
    "discovery_gain_ge_0p25": bool(float(selected["gain"]) >= 0.25),
    "confirmation_gain_ge_0p25": bool(confirmation["gain"] >= 0.25),
    "confirmation_far_gain_nonnegative": bool(confirmation["far_gain"] >= 0.0),
    "confirmation_near_degradation_le_0p05": bool(confirmation["near_degradation"] <= 0.05),
    "confirmation_p95_not_worse_by_gt_0p20": bool(confirmation["p95_gain"] >= -0.20),
    "confirmation_changed_rows_ge_0p05": bool(confirmation["changed_rows_fraction"] >= 0.05),
    "confirmation_oracle_gain_ge_1": bool(
        parent_confirmation["rmse"] - oracle_confirmation["rmse"] >= 1.0
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_block_gain_training_Fold3_discovery_confirmation_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "truth_usage": "Fold45 block labels and evaluation metrics only",
    "base": "V76 fixed Hist/Ridge adaptive dose endpoint",
    "training_label_dose": TRAIN_LABEL_DOSE,
    "candidate_bank": "V4 frozen three-arm union, balanced top64",
    "shortlist_source": SHORTLIST_SOURCE,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_contexts),
        "confirmation_wells": len(confirmation_contexts),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "model": {
        "family": "LightGBM Huber block-gain regressor plus Viterbi",
        "features": len(FEATURE_NAMES),
        "training_rows": int(sum(row.get("blocks", 0) * TOPK for row in train_report)),
        "block_ft": BLOCK_FT,
        "topk": TOPK,
    },
    "selected_config": {
        "gain_threshold": selected_threshold,
        "continuity_weight": selected_continuity,
        "reason": selected_reason,
    },
    "discovery": selected.to_dict(),
    "confirmation_parent": parent_confirmation,
    "confirmation": confirmation,
    "block_oracle": {
        "discovery": oracle_discovery,
        "confirmation": oracle_confirmation,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build Fold3 labels, refit on Fold345, and touch Fold2 once"
        if strict_pass else
        "FAIL -> block-level evidence cannot consume the candidate oracle; stop V107"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV107 DISCOVERY TOP 20")
print(
    scan[scan["threshold"].notna()].sort_values("rmse").head(20).to_string(index=False)
)
print("\nV107 CONFIRMATION")
print(pd.DataFrame([parent_confirmation, confirmation]).to_string(index=False))
print("\nV107 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
print("saved:", OUT_DIR)
