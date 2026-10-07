"""Nested whole-well selector for the independent V51 endpoints.

The exact 6.377 lineage is frozen through 2000 ft.  For each physical well the
model ranks four far-tail paths: base, PSG, Track8 common-only, and Track8.
All tuning is nested across folds 1/2/3.  Truth is used only as the training
label and for held-fold metrics; no truth-derived feature is admitted.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_cross_endpoint_wholewell_selector_v52"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
PSG_PATH = OOF_ROOT / "psg_sp45_crossfit_audit_v0" / "psg_sp45_crossfit_oof.parquet"
TRACK8_PATH = OOF_ROOT / "track8_sp45_foldclean_fivefold_v2" / "track8_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FROZEN_THROUGH_FT = 2000.0
FADE_FT = 512.0
CAP_FT = 48.0
SEED = 20260727
TEMPERATURES = (0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.5, 4.0)
DOSES = (0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
RAW_DIR_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction" / "train",
    ROOT / "data" / "train",
)
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
    t = np.clip((np.asarray(values, np.float64) - FROZEN_THROUGH_FT) / FADE_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _safe_stats(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.0] * 8
    return [
        float(np.mean(values)), float(np.std(values)),
        float(np.sqrt(np.mean(values**2))), float(np.mean(np.abs(values))),
        float(np.quantile(values, 0.10)), float(np.quantile(values, 0.90)),
        float(np.max(np.abs(values))), float(values[-1]),
    ]


STAT_SUFFIXES = ("mean", "std", "rms", "mae", "p10", "p90", "maxabs", "last")


def _add_stats(record, prefix, values, candidate_specific=True):
    for suffix, value in zip(STAT_SUFFIXES, _safe_stats(values)):
        name = f"{prefix}_{suffix}"
        record[name] = value
        if candidate_specific:
            candidate_feature_names.add(name)


def _gradient(values, coordinate):
    values = np.asarray(values, np.float64)
    coordinate = np.asarray(coordinate, np.float64)
    if len(values) < 3 or np.any(~np.isfinite(values)) or np.any(np.diff(coordinate) <= 0):
        return np.zeros(len(values), np.float64)
    return np.gradient(values, coordinate, edge_order=1)


def _metrics(prediction, endpoint, row_mask=None):
    use = np.ones(len(frame), bool) if row_mask is None else np.asarray(row_mask, bool)
    error = np.asarray(prediction, np.float64)[use] - truth[use]
    local_wells = wells[use]
    local_distance = distance[use]
    codes, names = pd.factorize(local_wells, sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error**2, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    result = {
        "endpoint": endpoint,
        "rows": int(use.sum()), "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    for label, mask in {
        "near_le2000": local_distance <= FROZEN_THROUGH_FT,
        "far_gt2000": local_distance > FROZEN_THROUGH_FT,
    }.items():
        result[f"{label}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else math.nan
    return result


def _merge_endpoint(path, columns, prefix):
    if not path.is_file():
        raise FileNotFoundError(path)
    schema = pq.ParquetFile(path).schema.names
    if "id" not in schema:
        raise RuntimeError(f"No id in {path}")
    missing = sorted(set(columns).difference(schema))
    if missing:
        raise RuntimeError(f"Missing {missing} in {path}")
    local = pd.read_parquet(path, columns=["id"] + list(columns))
    local["id"] = local["id"].astype(str).str.lower()
    if local["id"].duplicated().any():
        raise RuntimeError(f"Duplicate ids in {path}")
    local = local.rename(columns={column: f"{prefix}__{column}" for column in columns})
    return local


for required in (V33_PATH, PSG_PATH, TRACK8_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V33_PATH,
    columns=["id", "well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN],
)
frame = frame[frame["fold"].isin(FOLDS)].copy()
frame["id"] = frame["id"].astype(str).str.lower()
frame["well"] = frame["well"].astype(str).str.lower()
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in V33")

frame = frame.merge(_merge_endpoint(PSG_PATH, ("psg_tvt",), "psg"), on="id", how="left", validate="one_to_one")
frame = frame.merge(_merge_endpoint(TRACK8_PATH, ("common_only", "track8"), "track8"), on="id", how="left", validate="one_to_one")

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame[BASE_COLUMN].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
wells = frame["well"].astype(str).to_numpy()
fold_values = frame["fold"].to_numpy(np.int8)
fade = _smoothstep(distance)
frozen = distance <= FROZEN_THROUGH_FT

candidate_names = ["base_6p377", "psg_tvt", "track8_common", "track8"]
source_columns = [None, "psg__psg_tvt", "track8__common_only", "track8__track8"]
candidate_paths = [base.astype(np.float32)]
coverage = {}
for name, column in zip(candidate_names[1:], source_columns[1:]):
    raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(raw)
    coverage[name] = float(finite.mean())
    if finite.mean() < 0.95:
        raise RuntimeError(f"{name} coverage is only {finite.mean():.3%}")
    raw = np.where(finite, raw, base)
    correction = np.clip(raw - base, -CAP_FT, CAP_FT)
    path = base + fade * correction
    path[frozen] = base[frozen]
    candidate_paths.append(path.astype(np.float32))
candidate_matrix = np.column_stack(candidate_paths)
state_count = len(candidate_names)
if not np.isfinite(candidate_matrix).all():
    raise RuntimeError("Non-finite candidate matrix")
if not np.array_equal(
    candidate_matrix[frozen],
    np.repeat(base[frozen, None].astype(np.float32), state_count, axis=1),
):
    raise RuntimeError("A candidate changed rows through 2000 ft")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}
fold_by_well = frame.groupby("well", sort=False)["fold"].first().astype(int).to_dict()

raw_dir = next((path for path in RAW_DIR_CANDIDATES if path.is_dir()), None)
candidate_feature_names = set()
records = []
raw_loaded = 0
for number, (well, indices) in enumerate(groups.items(), 1):
    local_distance = distance[indices]
    far = local_distance > FROZEN_THROUGH_FT
    if not np.any(far):
        continue
    local_candidates = candidate_matrix[indices].astype(np.float64)
    local_base = base[indices]
    local_truth = truth[indices]
    local_row_idx = frame.loc[indices, "row_idx"].to_numpy(np.int64)

    md = local_distance.copy()
    z = np.zeros(len(indices), np.float64)
    gr = np.full(len(indices), np.nan, np.float64)
    if raw_dir is not None:
        path = raw_dir / f"{well}__horizontal_well.csv"
        if path.is_file():
            header = pd.read_csv(path, nrows=0).columns.tolist()
            usable = [column for column in ("MD", "Z", "GR") if column in header]
            raw = pd.read_csv(path, usecols=usable).apply(pd.to_numeric, errors="coerce")
            if len(raw) > int(local_row_idx.max()):
                if "MD" in raw:
                    md = raw["MD"].to_numpy(np.float64)[local_row_idx]
                if "Z" in raw:
                    z = raw["Z"].to_numpy(np.float64)[local_row_idx]
                if "GR" in raw:
                    gr = raw["GR"].to_numpy(np.float64)[local_row_idx]
                raw_loaded += 1

    bank_median = np.median(local_candidates, axis=1)
    bank_spread = np.std(local_candidates, axis=1)
    shared = {
        "well": well,
        "fold": int(fold_by_well[well]),
        "rows": int(len(indices)),
        "far_rows": int(far.sum()),
        "max_md_since": float(np.max(local_distance)),
        "gr_coverage_far": float(np.isfinite(gr[far]).mean()),
    }
    _add_stats(shared, "bank_spread_far", bank_spread[far], candidate_specific=False)
    _add_stats(shared, "gr_far", gr[far], candidate_specific=False)
    _add_stats(shared, "z_far", z[far], candidate_specific=False)

    far_state_mse = np.mean((local_candidates[far] - local_truth[far, None]) ** 2, axis=0)
    order = np.argsort(far_state_mse)
    relevance = np.empty(state_count, np.int8)
    relevance[order] = np.arange(state_count - 1, -1, -1, dtype=np.int8)

    for state, name in enumerate(candidate_names):
        path = local_candidates[:, state]
        correction = path - local_base
        consensus = path - bank_median
        other = np.delete(local_candidates, state, axis=1)
        nearest_rms = np.min(np.sqrt(np.mean((other[far] - path[far, None]) ** 2, axis=0)))
        record = dict(shared)
        record.update({
            "state": state,
            "candidate": name,
            "relevance": int(relevance[state]),
            "target_far_mse": float(far_state_mse[state]),
            "nearest_other_rms_far": float(nearest_rms),
        })
        candidate_feature_names.add("nearest_other_rms_far")
        for slot in range(state_count):
            record[f"is_state_{slot}"] = float(state == slot)
        _add_stats(record, "correction_far", correction[far])
        _add_stats(record, "consensus_far", consensus[far])
        for low, high, label in (
            (2000.0, 3000.0, "d2_3k"),
            (3000.0, 4000.0, "d3_4k"),
            (4000.0, np.inf, "d4k_plus"),
        ):
            mask = (local_distance > low) & (local_distance <= high)
            _add_stats(record, f"correction_{label}", correction[mask])
            _add_stats(record, f"consensus_{label}", consensus[mask])
        surface = path + z
        tvt_slope = _gradient(path[far], md[far])
        surface_slope = _gradient(surface[far], md[far])
        correction_slope = _gradient(correction[far], md[far])
        _add_stats(record, "tvt_slope_far", tvt_slope)
        _add_stats(record, "surface_slope_far", surface_slope)
        _add_stats(record, "correction_slope_far", correction_slope)
        _add_stats(record, "tvt_curvature_far", _gradient(tvt_slope, md[far]))
        _add_stats(record, "surface_curvature_far", _gradient(surface_slope, md[far]))
        records.append(record)
    if number % 100 == 0 or number == len(groups):
        print(f"V52 features {number}/{len(groups)} raw_loaded={raw_loaded}", flush=True)

features = pd.DataFrame(records).sort_values(["well", "state"], kind="stable").reset_index(drop=True)
if features["well"].nunique() < 460 or not features.groupby("well").size().eq(state_count).all():
    raise RuntimeError("Incomplete V52 well/state feature matrix")
excluded = {"well", "fold", "state", "candidate", "relevance", "target_far_mse"}
feature_names = [column for column in features.columns if column not in excluded]
identity_features = [column for column in feature_names if column.startswith("is_state_")]
candidate_feature_names = sorted(set(candidate_feature_names).intersection(feature_names))
shared_feature_names = sorted(set(feature_names).difference(candidate_feature_names).difference(identity_features))
features[feature_names] = features[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
features.to_parquet(OUT_DIR / "well_candidate_features.parquet", index=False, compression="zstd")


def _fit_model(local, seed):
    local = local.sort_values(["well", "state"], kind="stable")
    groups_local = local.groupby("well", sort=False).size().to_numpy(np.int32)
    if not np.all(groups_local == state_count):
        raise RuntimeError("Non-uniform rank groups")
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3, 7],
        n_estimators=240, learning_rate=0.025, num_leaves=7, max_depth=3,
        min_child_samples=28, subsample=0.85, colsample_bytree=0.72,
        reg_alpha=2.0, reg_lambda=18.0, random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        local[feature_names], local["relevance"].astype(int), group=groups_local,
        sample_weight=np.sqrt(np.maximum(local["far_rows"].to_numpy(np.float64), 1.0)),
    )
    return model


def _predict_scores(model, local, shuffled=False, seed=SEED):
    local = local.sort_values(["well", "state"], kind="stable").copy()
    matrix = np.array(local[feature_names].to_numpy(np.float64), copy=True)
    if shuffled:
        rng = np.random.default_rng(seed)
        candidate_positions = [feature_names.index(name) for name in candidate_feature_names]
        for _, positions in local.groupby("well", sort=False).indices.items():
            positions = np.asarray(positions, np.int64)
            permutation = rng.permutation(len(positions))
            matrix[np.ix_(positions, candidate_positions)] = matrix[np.ix_(positions[permutation], candidate_positions)]
    output = local[["well", "fold", "state"]].copy()
    output["score"] = model.predict(matrix)
    return output


def _normalised_weights(scores, temperature):
    scores = np.asarray(scores, np.float64)
    scores = (scores - np.mean(scores)) / max(float(np.std(scores)), 1e-6)
    logits = scores / float(temperature)
    logits -= np.max(logits)
    weights = np.exp(np.clip(logits, -40.0, 0.0))
    return weights / np.sum(weights)


def _assemble(scores, temperature, dose, target_wells):
    prediction = base.copy()
    choices = []
    score_groups = {str(well): local for well, local in scores.groupby("well", sort=False)}
    for well in target_wells:
        local = score_groups.get(str(well))
        if local is None or len(local) != state_count:
            continue
        local = local.sort_values("state", kind="stable")
        weights = _normalised_weights(local["score"].to_numpy(np.float64), temperature)
        weights *= float(dose)
        weights[0] += 1.0 - float(dose)
        indices = groups[str(well)]
        prediction[indices] = candidate_matrix[indices].astype(np.float64) @ weights
        choices.append({
            "well": str(well), "temperature": float(temperature), "dose": float(dose),
            "selected_state": int(np.argmax(weights)), "selected_candidate": candidate_names[int(np.argmax(weights))],
            **{f"weight_{name}": float(weights[state]) for state, name in enumerate(candidate_names)},
        })
    return prediction, choices


def _scan(scores, folds_to_score):
    rows = []
    target_wells = sorted(well for well, fold in fold_by_well.items() if fold in folds_to_score)
    row_mask = np.isin(fold_values, list(folds_to_score))
    base_metrics = _metrics(base, "base", row_mask)
    for temperature in TEMPERATURES:
        for dose in DOSES:
            prediction, _ = _assemble(scores, temperature, dose, target_wells)
            metrics = _metrics(prediction, "candidate", row_mask)
            fold_gains = []
            for fold in folds_to_score:
                mask = fold_values == fold
                fold_gains.append(_metrics(base, "base", mask)["rmse"] - _metrics(prediction, "candidate", mask)["rmse"])
            rows.append({
                "temperature": float(temperature), "dose": float(dose),
                "rmse": metrics["rmse"], "gain": base_metrics["rmse"] - metrics["rmse"],
                "far_gain": base_metrics["far_gt2000_rmse"] - metrics["far_gt2000_rmse"],
                "p95_change": metrics["p95_well_rmse"] - base_metrics["p95_well_rmse"],
                "minimum_fold_gain": float(min(fold_gains)),
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
        inner_train = features[features["fold"].eq(next(fold for fold in train_folds if fold != inner_valid))]
        inner_valid_frame = features[features["fold"].eq(inner_valid)]
        inner_model = _fit_model(inner_train, SEED + held_fold * 10 + inner_valid)
        inner_scores.append(_predict_scores(inner_model, inner_valid_frame))
    inner_scores = pd.concat(inner_scores, ignore_index=True)
    scan, selected = _scan(inner_scores, tuple(train_folds))
    scan["held_fold"] = held_fold
    scan_rows.append(scan)

    train = features[features["fold"].isin(train_folds)]
    held = features[features["fold"].eq(held_fold)]
    model = _fit_model(train, SEED + held_fold)
    model.booster_.save_model(str(MODEL_DIR / f"ranker_fold{held_fold}.txt"))
    held_scores = _predict_scores(model, held)
    shuffled_scores = _predict_scores(model, held, shuffled=True, seed=SEED + held_fold)
    held_wells = sorted(well for well, fold in fold_by_well.items() if fold == held_fold)
    prediction, choices = _assemble(held_scores, selected["temperature"], selected["dose"], held_wells)
    shuffled_prediction, _ = _assemble(shuffled_scores, selected["temperature"], selected["dose"], held_wells)
    mask = fold_values == held_fold
    oof_prediction[mask] = prediction[mask]
    oof_shuffle_prediction[mask] = shuffled_prediction[mask]
    choice_rows.extend({**item, "fold": held_fold} for item in choices)
    base_fold = _metrics(base, "base", mask)
    candidate_fold = _metrics(prediction, "candidate", mask)
    shuffle_fold = _metrics(shuffled_prediction, "shuffle", mask)
    fold_reports.append({
        "fold": held_fold,
        "selected_temperature": float(selected["temperature"]),
        "selected_dose": float(selected["dose"]),
        "inner_gain": float(selected["gain"]),
        "inner_minimum_fold_gain": float(selected["minimum_fold_gain"]),
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
        for name, gain in zip(feature_names, model.booster_.feature_importance(importance_type="gain"))
    )
    print("V52 FOLD", fold_reports[-1], flush=True)

base_metrics = _metrics(base, "base")
candidate_metrics = _metrics(oof_prediction, "nested_selector")
shuffle_metrics = _metrics(oof_shuffle_prediction, "candidate_feature_shuffle")
fold_table = pd.DataFrame(fold_reports)
choices = pd.DataFrame(choice_rows)
checks = {
    "overall_gain_ge_0p10": bool(base_metrics["rmse"] - candidate_metrics["rmse"] >= 0.10),
    "far_gain_ge_0p15": bool(base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"] >= 0.15),
    "minimum_fold_gain_ge_minus0p03": bool(fold_table["gain"].min() >= -0.03),
    "positive_folds_ge_2": bool((fold_table["gain"] > 0).sum() >= 2),
    "shuffle_loses_ge_0p05": bool(shuffle_metrics["rmse"] - candidate_metrics["rmse"] >= 0.05),
    "maximum_p95_change_le_0p20": bool(fold_table["p95_change"].max() <= 0.20),
    "near_rows_bitwise_unchanged": bool(np.array_equal(oof_prediction[frozen], base[frozen])),
}
strict_pass = bool(all(checks.values()))

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["base_6p377"] = base.astype(np.float32)
output["nested_selector"] = oof_prediction.astype(np.float32)
output["candidate_feature_shuffle"] = oof_shuffle_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "selector_oof.parquet", index=False, compression="zstd")
choices.to_csv(OUT_DIR / "well_choices.csv", index=False)
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
pd.concat(scan_rows, ignore_index=True).to_csv(OUT_DIR / "inner_scan.csv", index=False)
pd.DataFrame(importance_rows).to_csv(OUT_DIR / "feature_importance.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_cross_endpoint_complete_well_ranking",
    "sample_unit": "one complete physical well with four endpoint states",
    "parent": str(V33_PATH),
    "candidates": candidate_names,
    "coverage": coverage,
    "raw_geometry_directory": str(raw_dir) if raw_dir else None,
    "raw_geometry_wells_loaded": int(raw_loaded),
    "features": {
        "total": len(feature_names),
        "identity": len(identity_features),
        "candidate_specific": len(candidate_feature_names),
        "shared": len(shared_feature_names),
    },
    "base": base_metrics,
    "selector": candidate_metrics,
    "candidate_feature_shuffle": shuffle_metrics,
    "gain": base_metrics["rmse"] - candidate_metrics["rmse"],
    "far_gain": base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"],
    "shuffle_loss": shuffle_metrics["rmse"] - candidate_metrics["rmse"],
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> freeze the whole-well gate and test 512-ft segment refinement"
        if strict_pass else
        "FAIL -> whole-well endpoint benefit is not observable; do not train a segment selector on the same evidence"
    ),
}
_write_json(OUT_DIR / "result.json", verdict)

print("\nV52 FOLDS", flush=True)
print(fold_table.to_string(index=False), flush=True)
print("\nV52 CHOICE COUNTS", flush=True)
print(choices["selected_candidate"].value_counts(dropna=False).to_string(), flush=True)
print("\nV52 FINAL VERDICT", flush=True)
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
