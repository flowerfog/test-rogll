"""Canonical-fold audit for adding the V4 coarse path to the 6.377 incumbent.

The old V61-V63 audit joined two incompatible fold lineages.  This AutoDL
cell uses the complete V66 incumbent as the only fold authority, fits the
structural prior on canonical Folds 4/5, trains the frozen V4 ranker on
canonical Folds 3/4/5, selects one scalar blend dose on canonical Fold 2,
and applies it once to canonical Fold 1.

No architecture, feature, candidate-arm, or ranker hyperparameter search is
performed.  Per-well training samples and evaluation endpoints are cached.
"""

from __future__ import annotations

import hashlib
import json
import math
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
EXPERIMENT = "rawdata_lattice_coarse_canonical_gateway_v67"
OUT_DIR = OOF_ROOT / EXPERIMENT
TRAIN_CACHE = OUT_DIR / "train_samples"
EVAL_CACHE = OUT_DIR / "eval_parts"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, TRAIN_CACHE, EVAL_CACHE, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V4_VERDICT = OOF_ROOT / "rawdata_lattice_coarse_ranker_v4" / "verdict.json"
INCUMBENT_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
INCUMBENT_COLUMN = "incumbent_6p377_replay"
TRUTH_COLUMN = "tvt_true"
PRIOR_FOLDS = (4, 5)
TRAIN_FOLDS = (3, 4, 5)
SELECTION_FOLD = 2
CONFIRMATION_FOLD = 1
WEIGHT_STEP = 0.05
MAX_WEIGHT = 0.50
DEFAULT_ITERATIONS = 400
RANDOM_SEED = 20260721
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773


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


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_v4():
    if not V4_SOURCE.is_file():
        raise FileNotFoundError(V4_SOURCE)
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 bootstrap marker missing from {V4_SOURCE}")
    module = types.ModuleType("raw_lattice_v4_canonical_definitions")
    module.__file__ = str(V4_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V4_SOURCE), "exec"), module.__dict__)
    return module


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH_COLUMN].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid endpoint {endpoint}")
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    for name, mask in {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _analytic_weight(frame):
    parent = frame[INCUMBENT_COLUMN].to_numpy(np.float64)
    endpoint = frame["coarse_path"].to_numpy(np.float64)
    truth = frame[TRUTH_COLUMN].to_numpy(np.float64)
    delta = endpoint - parent
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(truth - parent, delta) / denominator, 0.0, MAX_WEIGHT))


def _round_weight(value):
    return float(np.clip(round(value / WEIGHT_STEP) * WEIGHT_STEP, 0.0, MAX_WEIGHT))


def _blend(frame, weight):
    parent = frame[INCUMBENT_COLUMN].to_numpy(np.float64)
    endpoint = frame["coarse_path"].to_numpy(np.float64)
    return parent + float(weight) * (endpoint - parent)


for required in (V4_SOURCE, INCUMBENT_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

Q = _bootstrap_v4()
V = Q.V
B = Q.B
DATA_ROOT = Path(Q.DATA_ROOT)
SEGMENT_LENGTH_FT = float(Q.SEGMENT_LENGTH_FT)

incumbent = pd.read_parquet(INCUMBENT_PATH)
required_incumbent = [
    "id", "well", "row_idx", "fold", TRUTH_COLUMN, "md_since", INCUMBENT_COLUMN
]
missing = sorted(set(required_incumbent).difference(incumbent.columns))
if missing:
    raise RuntimeError(f"V66 incumbent is missing columns: {missing}")
incumbent = incumbent[required_incumbent].copy()
incumbent["id"] = incumbent["id"].astype(str)
incumbent["well"] = incumbent["well"].astype(str)
incumbent["fold"] = incumbent["fold"].astype(np.int8)
incumbent = incumbent.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if (
    len(incumbent) != EXPECTED_ROWS
    or incumbent["well"].nunique() != EXPECTED_WELLS
    or incumbent["id"].duplicated().any()
):
    raise RuntimeError(
        f"V66 coverage invalid: rows={len(incumbent)} "
        f"wells={incumbent['well'].nunique()} duplicates={incumbent['id'].duplicated().sum()}"
    )
fold_by_well = incumbent.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
if any(incumbent.groupby("well")["fold"].nunique() != 1):
    raise RuntimeError("A physical well appears in multiple canonical folds")

raw_wells = {
    path.name.split("__", 1)[0]
    for path in (DATA_ROOT / "train").glob("*__horizontal_well.csv")
}
if raw_wells != set(fold_by_well):
    raise RuntimeError(
        f"Raw/V66 well mismatch: raw_only={len(raw_wells - set(fold_by_well))} "
        f"v66_only={len(set(fold_by_well) - raw_wells)}"
    )

prior_wells = sorted(well for well, fold in fold_by_well.items() if fold in PRIOR_FOLDS)
train_wells = sorted(well for well, fold in fold_by_well.items() if fold in TRAIN_FOLDS)
selection_wells = sorted(well for well, fold in fold_by_well.items() if fold == SELECTION_FOLD)
confirmation_wells = sorted(
    well for well, fold in fold_by_well.items() if fold == CONFIRMATION_FOLD
)

prior_records = []
record_failures = []
for number, well in enumerate(prior_wells, 1):
    try:
        prior_records.append(V["_load_record"](well))
    except Exception as exc:
        record_failures.append({"stage": "prior", "well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(prior_wells):
        print(
            f"V67 load canonical prior records {number}/{len(prior_wells)} "
            f"failures={len(record_failures)}",
            flush=True,
        )
if len(prior_records) < 307:
    raise RuntimeError(f"Only {len(prior_records)} eligible canonical Fold45 prior wells")
prior_eligible = len(prior_records)
prior = V["_fit_prior"](prior_records, SEGMENT_LENGTH_FT)
del prior_records
_write_json(
    OUT_DIR / "canonical_fold45_prior.json",
    {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in prior.items()
    },
)

feature_names = None
x_parts = []
y_parts = []
groups = []
training_report = []
for number, well in enumerate(train_wells, 1):
    cache_path = TRAIN_CACHE / f"{well}.npz"
    status = "reused"
    try:
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as data:
                x = data["x"].astype(np.float32, copy=False)
                y = data["y"].astype(np.int8, copy=False)
                names = json.loads(str(data["feature_names_json"].item()))
                bank_candidates = int(data["bank_candidates"][0])
        else:
            record = V["_load_record"](well)
            paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(
                record, prior
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Mean of empty slice", category=RuntimeWarning
                )
                features, names = Q._candidate_evidence(
                    record, prior, paths, segments, arm_rank, arm_score_z, source_mask
                )
            targets = Q._exact_targets(record, prior, paths, segments)
            panel = {
                "features": features,
                "mse": targets["mse"].astype(np.float32),
                "arm_rank": arm_rank,
            }
            indices = Q._training_indices(panel, well)
            x = features[indices].astype(np.float32, copy=False)
            y = Q._rank_labels(targets["mse"][indices]).astype(np.int8)
            bank_candidates = int(len(paths))
            np.savez_compressed(
                cache_path,
                x=x,
                y=y,
                feature_names_json=np.asarray(json.dumps(names)),
                bank_candidates=np.asarray([bank_candidates], np.int32),
            )
            status = "built"
        if feature_names is None:
            feature_names = list(names)
        elif list(names) != feature_names:
            raise RuntimeError("feature schema mismatch")
        x_parts.append(x)
        y_parts.append(y)
        groups.append(int(len(y)))
        training_report.append(
            {
                "well": well,
                "fold": int(fold_by_well[well]),
                "sampled_candidates": int(len(y)),
                "bank_candidates": bank_candidates,
                "status": status,
            }
        )
    except Exception as exc:
        record_failures.append({"stage": "ranker_train", "well": well, "reason": repr(exc)})
    if number % 25 == 0 or number == len(train_wells):
        print(
            f"V67 canonical 345 samples {number}/{len(train_wells)} "
            f"usable={len(groups)} elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )
if len(groups) < 460:
    raise RuntimeError(f"Only {len(groups)} usable canonical Fold345 training wells")
pd.DataFrame(training_report).to_csv(OUT_DIR / "training_report.csv", index=False)
pd.DataFrame(record_failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
_write_json(OUT_DIR / "feature_names.json", feature_names)

x_train = np.concatenate(x_parts, axis=0)
y_train = np.concatenate(y_parts, axis=0)
del x_parts, y_parts
if V4_VERDICT.is_file():
    v4_verdict = json.loads(V4_VERDICT.read_text(encoding="utf-8"))
    iterations = int(v4_verdict.get("best_iteration") or DEFAULT_ITERATIONS)
else:
    iterations = DEFAULT_ITERATIONS
iterations = max(1, iterations)

print(
    "V67 CANONICAL TRAIN MATRIX",
    {
        "rows": len(x_train),
        "features": x_train.shape[1],
        "wells": len(groups),
        "iterations": iterations,
    },
    flush=True,
)
model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=(1, 8, 32),
    n_estimators=iterations,
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
model.fit(x_train, y_train, group=groups)
model_path = MODEL_DIR / "coarse_ranker_canonical345.txt"
model.booster_.save_model(str(model_path))
del x_train, y_train


def _build_eval_part(well, fold):
    parent = incumbent[incumbent["well"].eq(well)].sort_values(
        "row_idx", kind="stable"
    ).copy()
    if len(parent) == 0:
        raise RuntimeError(f"{well}: no V66 parent rows")
    parent_prediction = parent[INCUMBENT_COLUMN].to_numpy(np.float64)
    status = "built"
    reason = ""
    try:
        record = V["_load_record"](well)
        paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(
            record, prior
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice", category=RuntimeWarning
            )
            features, names = Q._candidate_evidence(
                record, prior, paths, segments, arm_rank, arm_score_z, source_mask
            )
        if list(names) != feature_names:
            raise RuntimeError("feature schema mismatch")
        scores = model.predict(features)
        chosen = int(np.argmax(scores))
        eval_rows = record["eval_rows"]
        coarse = B["_path_prediction"](record, prior, paths[chosen], segments)[eval_rows]
        truth = record["truth"][eval_rows].astype(np.float64)
        distance = (
            record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
        )
        if len(parent) != len(eval_rows):
            raise RuntimeError(f"row count {len(parent)}/{len(eval_rows)}")
        truth_gap = float(
            np.max(np.abs(parent[TRUTH_COLUMN].to_numpy(np.float64) - truth))
        )
        distance_gap = float(
            np.max(np.abs(parent["md_since"].to_numpy(np.float64) - distance))
        )
        if truth_gap > 0.01 or distance_gap > 1.01:
            raise RuntimeError(
                f"alignment truth_gap={truth_gap} distance_gap={distance_gap}"
            )
        if not np.isfinite(coarse).all():
            raise RuntimeError("non-finite coarse path")
        parent["coarse_path"] = np.asarray(coarse, np.float32)
        candidate_count = int(len(paths))
        chosen_score = float(scores[chosen])
    except Exception as exc:
        parent["coarse_path"] = parent_prediction.astype(np.float32)
        status = "parent_fallback"
        reason = repr(exc)
        candidate_count = 0
        chosen_score = np.nan
    parent["coarse_status"] = status
    keep = [
        "id", "well", "row_idx", "fold", TRUTH_COLUMN, "md_since",
        INCUMBENT_COLUMN, "coarse_path", "coarse_status",
    ]
    report = {
        "well": well,
        "fold": int(fold),
        "rows": int(len(parent)),
        "status": status,
        "reason": reason,
        "candidate_count": candidate_count,
        "chosen_score": chosen_score,
    }
    return parent[keep], report


eval_parts = []
eval_reports = []
for fold, wells in (
    (SELECTION_FOLD, selection_wells),
    (CONFIRMATION_FOLD, confirmation_wells),
):
    fold_cache = EVAL_CACHE / f"fold{fold}"
    fold_cache.mkdir(parents=True, exist_ok=True)
    for number, well in enumerate(wells, 1):
        path = fold_cache / f"{well}.parquet"
        report_path = fold_cache / f"{well}.json"
        if path.is_file() and report_path.is_file():
            part = pd.read_parquet(path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            part, report = _build_eval_part(well, fold)
            part.to_parquet(path, index=False, compression="zstd")
            _write_json(report_path, report)
        eval_parts.append(part)
        eval_reports.append(report)
        if number % 10 == 0 or number == len(wells):
            print(
                f"V67 reconstruct canonical Fold{fold} {number}/{len(wells)} "
                f"fallbacks={sum(r['status'] != 'built' for r in eval_reports)} "
                f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
                flush=True,
            )

frame = pd.concat(eval_parts, ignore_index=True)
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
pd.DataFrame(eval_reports).to_csv(OUT_DIR / "evaluation_report.csv", index=False)
if frame["id"].duplicated().any():
    raise RuntimeError("V67 endpoint contains duplicate ids")
observed_fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if observed_fold_wells != {1: 155, 2: 155}:
    raise RuntimeError(f"Incomplete canonical heldout coverage: {observed_fold_wells}")

fold2 = frame[frame["fold"].eq(SELECTION_FOLD)].copy()
fold1 = frame[frame["fold"].eq(CONFIRMATION_FOLD)].copy()
analytic_weight = _analytic_weight(fold2)
frozen_weight = _round_weight(analytic_weight)

summary_rows = []
metrics = {}
for fold, local in ((2, fold2), (1, fold1)):
    predictions = {
        "parent": local[INCUMBENT_COLUMN].to_numpy(np.float64),
        "coarse": local["coarse_path"].to_numpy(np.float64),
        "blend_frozen": _blend(local, frozen_weight),
        "blend_w0p10_control": _blend(local, 0.10),
    }
    metrics[str(fold)] = {}
    for endpoint, prediction in predictions.items():
        result = _metric(local, prediction, endpoint)
        metrics[str(fold)][endpoint] = result
        summary_rows.append({"fold": fold, **result})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

frame["coarse_blend_frozen"] = _blend(frame, frozen_weight).astype(np.float32)
output_path = OUT_DIR / "canonical_fold12_coarse_oof.parquet"
frame.to_parquet(output_path, index=False, compression="zstd")

fold2_parent = metrics["2"]["parent"]
fold2_blend = metrics["2"]["blend_frozen"]
fold1_parent = metrics["1"]["parent"]
fold1_blend = metrics["1"]["blend_frozen"]
fold2_gain = fold2_parent["rmse"] - fold2_blend["rmse"]
fold1_gain = fold1_parent["rmse"] - fold1_blend["rmse"]
fold1_far_gain = fold1_parent["far_gt2000_rmse"] - fold1_blend["far_gt2000_rmse"]
fold1_p95_gain = fold1_parent["p95_well_rmse"] - fold1_blend["p95_well_rmse"]
fallback_wells = int(sum(report["status"] != "built" for report in eval_reports))
checks = {
    "canonical_fold2_155_wells": bool(fold2["well"].nunique() == 155),
    "canonical_fold1_155_wells": bool(fold1["well"].nunique() == 155),
    "fallback_wells_le_2": bool(fallback_wells <= 2),
    "selected_weight_positive": bool(frozen_weight > 0.0),
    "fold2_gain_ge_0p03": bool(fold2_gain >= 0.03),
    "fold1_gain_ge_0p03": bool(fold1_gain >= 0.03),
    "fold1_far_gain_nonnegative": bool(fold1_far_gain >= 0.0),
    "fold1_p95_gain_nonnegative": bool(fold1_p95_gain >= 0.0),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "canonical345_training_canonical2_weight_selection_canonical1_confirmation",
    "sample_unit": "one complete physical well",
    "candidate_generation_uses_truth": False,
    "truth_usage": "Fold345 rank labels and Fold2 scalar blend weight only",
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "ranker_train_folds": list(TRAIN_FOLDS),
        "weight_selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "fold1_search_count": 0,
        "canonical_fold_source": str(INCUMBENT_PATH),
    },
    "frozen_v4_recipe": {
        "candidate_union": list(Q.SELECTED_ARM_IDS),
        "segment_length_ft": SEGMENT_LENGTH_FT,
        "features": len(feature_names),
        "iterations": iterations,
        "ranker_hyperparameters": {
            "learning_rate": 0.025,
            "num_leaves": 31,
            "min_child_samples": 80,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
        },
    },
    "coverage": {
        "prior_requested": len(prior_wells),
        "prior_eligible": prior_eligible,
        "ranker_train_requested": len(train_wells),
        "ranker_train_eligible": len(groups),
        "heldout_fold_wells": {str(k): v for k, v in observed_fold_wells.items()},
        "fallback_wells": fallback_wells,
    },
    "weight": {
        "fold2_analytic": analytic_weight,
        "step": WEIGHT_STEP,
        "frozen": frozen_weight,
    },
    "metrics": metrics,
    "gains": {
        "fold2_rmse": fold2_gain,
        "fold1_rmse": fold1_gain,
        "fold1_far": fold1_far_gain,
        "fold1_p95": fold1_p95_gain,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "model": str(model_path),
    "model_sha256": _sha256(model_path),
    "output": str(output_path),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> retrain the identical coarse ranker on all eligible 773 wells and deploy only the frozen weight"
        if strict_pass
        else "FAIL -> retire V65; the coarse endpoint does not add to the complete 6.377 lineage"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV67 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV67 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
if not strict_pass:
    print("V67 did not pass; do not create another V65 submission.", flush=True)
