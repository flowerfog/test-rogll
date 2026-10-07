"""Honest local proxy for the V68 all-773 coarse-ranker refit.

The deployed all-773 model cannot be scored on the same 773 wells used to fit
its rank labels.  This AutoDL cell instead performs fixed-recipe cross-fitting
on canonical Folds 1/2/3.  For each held-out fold, the ranker is trained on the
other four folds and evaluated only on the held-out complete wells.

The structural prior remains the exact V67 Fold-4/5 prior used by deployment.
Consequently Folds 4/5 are prior-fitting data and are deliberately not reported
as honest validation folds.  No weight, iteration, feature, or arm search is
performed: the blend weight is fixed at 0.30 and the ranker at 67 iterations.
Run V68 first so all sampled candidate-label caches are available.
"""

from __future__ import annotations

import gc
import hashlib
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
EXPERIMENT = "rawdata_lattice_coarse_all773_crossfit_v70"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
EVAL_CACHE = OUT_DIR / "eval_parts"
for directory in (OUT_DIR, MODEL_DIR, EVAL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V66_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
V67_DIR = OOF_ROOT / "rawdata_lattice_coarse_canonical_gateway_v67"
V67_PRIOR = V67_DIR / "canonical_fold45_prior.json"
V67_CACHE = V67_DIR / "train_samples"
V67_OOF = V67_DIR / "canonical_fold12_coarse_oof.parquet"
V68_DIR = OOF_ROOT / "rawdata_lattice_coarse_canonical_final_bundle_v68"
V68_CACHE = V68_DIR / "sampled_training_rows"
V68_VERDICT = V68_DIR / "verdict.json"

INCUMBENT_COLUMN = "incumbent_6p377_replay"
TRUTH_COLUMN = "tvt_true"
HONEST_FOLDS = (1, 2, 3)
PRIOR_FOLDS = (4, 5)
FROZEN_ITERATIONS = 67
FROZEN_WEIGHT = 0.30
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
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_v4():
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 bootstrap marker missing from {V4_SOURCE}")
    module = types.ModuleType("raw_lattice_v4_all773_crossfit_definitions")
    module.__file__ = str(V4_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V4_SOURCE), "exec"), module.__dict__)
    return module


def _sample_path(well):
    candidates = (V68_CACHE / f"{well}.npz", V67_CACHE / f"{well}.npz")
    return next((path for path in candidates if path.is_file()), None)


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


for required in (V4_SOURCE, V66_PATH, V67_PRIOR, V68_VERDICT):
    if not required.is_file():
        raise FileNotFoundError(required)

v68 = json.loads(V68_VERDICT.read_text(encoding="utf-8"))
if not v68.get("strict_pass"):
    raise RuntimeError("V68 did not strictly pass")
if int(v68["iterations"]) != FROZEN_ITERATIONS:
    raise RuntimeError(f"Unexpected V68 iterations: {v68['iterations']}")
if abs(float(v68["blend_weight_coarse"]) - FROZEN_WEIGHT) > 1e-12:
    raise RuntimeError(f"Unexpected V68 blend weight: {v68['blend_weight_coarse']}")

Q = _bootstrap_v4()
V = Q.V
B = Q.B
DATA_ROOT = Path(Q.DATA_ROOT)
prior_payload = json.loads(V67_PRIOR.read_text(encoding="utf-8"))
prior = {
    key: np.asarray(value, np.float64) if key == "rate_grid" else value
    for key, value in prior_payload.items()
}

incumbent = pd.read_parquet(V66_PATH)
required_columns = {
    "id", "well", "row_idx", "fold", TRUTH_COLUMN, "md_since", INCUMBENT_COLUMN
}
missing = sorted(required_columns.difference(incumbent.columns))
if missing:
    raise RuntimeError(f"V66 incumbent missing columns: {missing}")
incumbent = incumbent[list(required_columns)].copy()
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
        f"Invalid V66 coverage: rows={len(incumbent)} wells={incumbent['well'].nunique()}"
    )
fold_by_well = incumbent.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
if any(incumbent.groupby("well")["fold"].nunique() != 1):
    raise RuntimeError("A physical well appears in multiple canonical folds")

sample_paths = {well: _sample_path(well) for well in sorted(fold_by_well)}
missing_samples = [well for well, path in sample_paths.items() if path is None]
if len(missing_samples) > 3:
    raise RuntimeError(
        f"Missing {len(missing_samples)} V68/V67 sampled panels. Rerun V68 first: "
        f"{missing_samples[:10]}"
    )

feature_names = None
for path in sample_paths.values():
    if path is None:
        continue
    with np.load(path, allow_pickle=False) as data:
        names = json.loads(str(data["feature_names_json"].item()))
    if feature_names is None:
        feature_names = list(names)
    elif list(names) != feature_names:
        raise RuntimeError(f"Feature schema mismatch in {path}")
if feature_names is None:
    raise RuntimeError("No sampled panels available")


def _fit_fold_model(heldout_fold):
    train_wells = sorted(
        well
        for well, fold in fold_by_well.items()
        if fold != heldout_fold and sample_paths[well] is not None
    )
    x_parts, y_parts, groups = [], [], []
    for number, well in enumerate(train_wells, 1):
        with np.load(sample_paths[well], allow_pickle=False) as data:
            x = data["x"].astype(np.float32, copy=False)
            y = data["y"].astype(np.int32, copy=False)
        x_parts.append(x)
        y_parts.append(y)
        groups.append(int(len(y)))
        if number % 100 == 0 or number == len(train_wells):
            print(
                f"V70 Fold{heldout_fold} load train panels {number}/{len(train_wells)}",
                flush=True,
            )
    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    del x_parts, y_parts
    print(
        f"V70 Fold{heldout_fold} train matrix",
        {"rows": len(x_train), "features": x_train.shape[1], "wells": len(groups)},
        flush=True,
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=(1, 8, 32),
        n_estimators=FROZEN_ITERATIONS,
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
    model_path = MODEL_DIR / f"coarse_ranker_leave_fold{heldout_fold}_out.txt"
    model.booster_.save_model(str(model_path))
    del x_train, y_train
    gc.collect()
    return model.booster_, train_wells, model_path


def _predict_well(well, fold, model):
    parent = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    parent_prediction = parent[INCUMBENT_COLUMN].to_numpy(np.float64)
    try:
        record = V["_load_record"](well)
        paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
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
        distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
        if len(parent) != len(eval_rows):
            raise RuntimeError(f"row count {len(parent)}/{len(eval_rows)}")
        truth_gap = float(np.max(np.abs(parent[TRUTH_COLUMN].to_numpy(np.float64) - truth)))
        distance_gap = float(np.max(np.abs(parent["md_since"].to_numpy(np.float64) - distance)))
        if truth_gap > 0.01 or distance_gap > 1.01:
            raise RuntimeError(f"alignment truth_gap={truth_gap} distance_gap={distance_gap}")
        if not np.isfinite(coarse).all():
            raise RuntimeError("non-finite coarse path")
        parent["coarse_crossfit"] = np.asarray(coarse, np.float32)
        status = "built"
        reason = ""
        candidate_count = int(len(paths))
        chosen_score = float(scores[chosen])
    except Exception as exc:
        parent["coarse_crossfit"] = parent_prediction.astype(np.float32)
        status = "parent_fallback"
        reason = repr(exc)
        candidate_count = 0
        chosen_score = np.nan
    parent["blend_w0p30"] = (
        parent_prediction
        + FROZEN_WEIGHT
        * (parent["coarse_crossfit"].to_numpy(np.float64) - parent_prediction)
    ).astype(np.float32)
    report = {
        "well": well,
        "fold": int(fold),
        "rows": int(len(parent)),
        "status": status,
        "reason": reason,
        "candidate_count": candidate_count,
        "chosen_score": chosen_score,
    }
    keep = [
        "id", "well", "row_idx", "fold", TRUTH_COLUMN, "md_since",
        INCUMBENT_COLUMN, "coarse_crossfit", "blend_w0p30",
    ]
    return parent[keep], report


parts = []
reports = []
model_manifest = []
for heldout_fold in HONEST_FOLDS:
    model, train_wells, model_path = _fit_fold_model(heldout_fold)
    valid_wells = sorted(well for well, fold in fold_by_well.items() if fold == heldout_fold)
    fold_cache = EVAL_CACHE / f"fold{heldout_fold}"
    fold_cache.mkdir(parents=True, exist_ok=True)
    for number, well in enumerate(valid_wells, 1):
        part_path = fold_cache / f"{well}.parquet"
        report_path = fold_cache / f"{well}.json"
        if part_path.is_file() and report_path.is_file():
            part = pd.read_parquet(part_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            part, report = _predict_well(well, heldout_fold, model)
            part.to_parquet(part_path, index=False, compression="zstd")
            _write_json(report_path, report)
        parts.append(part)
        reports.append(report)
        if number % 10 == 0 or number == len(valid_wells):
            fold_fallbacks = sum(
                item["status"] != "built" for item in reports if item["fold"] == heldout_fold
            )
            print(
                f"V70 Fold{heldout_fold} evaluate {number}/{len(valid_wells)} "
                f"fallbacks={fold_fallbacks} elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
                flush=True,
            )
    model_manifest.append(
        {
            "heldout_fold": heldout_fold,
            "train_wells": len(train_wells),
            "valid_wells": len(valid_wells),
            "model": str(model_path),
            "sha256": _sha256(model_path),
        }
    )
    del model
    gc.collect()

frame = pd.concat(parts, ignore_index=True)
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate cross-fit ids")
observed_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
expected_wells = {1: 155, 2: 155, 3: 154}
if observed_wells != expected_wells:
    raise RuntimeError(f"Incomplete honest-fold coverage: {observed_wells}")

summary_rows = []
metrics = {}
for fold in (*HONEST_FOLDS, 0):
    local = frame if fold == 0 else frame[frame["fold"].eq(fold)]
    metrics[str(fold)] = {}
    for endpoint, column in {
        "incumbent_6p377": INCUMBENT_COLUMN,
        "coarse_crossfit": "coarse_crossfit",
        "blend_w0p30": "blend_w0p30",
    }.items():
        result = _metric(local, local[column].to_numpy(np.float64), endpoint)
        metrics[str(fold)][endpoint] = result
        summary_rows.append({"fold": fold, **result})

v67_comparison = None
if V67_OOF.is_file():
    old = pd.read_parquet(V67_OOF, columns=["id", "fold", "coarse_blend_frozen"])
    matched = frame[frame["fold"].isin((1, 2))].merge(
        old, on=["id", "fold"], how="inner", validate="one_to_one"
    )
    if len(matched) == len(frame[frame["fold"].isin((1, 2))]):
        v67_comparison = {}
        for fold in (1, 2, 0):
            local = matched if fold == 0 else matched[matched["fold"].eq(fold)]
            old_metric = _metric(local, local["coarse_blend_frozen"], "v67_canonical345")
            new_metric = _metric(local, local["blend_w0p30"], "v70_leave_one_fold_out")
            v67_comparison[str(fold)] = {
                "v67_rmse": old_metric["rmse"],
                "v70_rmse": new_metric["rmse"],
                "v70_gain_vs_v67": old_metric["rmse"] - new_metric["rmse"],
            }

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(reports).to_csv(OUT_DIR / "evaluation_report.csv", index=False)
frame.to_parquet(OUT_DIR / "all773_proxy_crossfit_oof.parquet", index=False, compression="zstd")

fold_gains = {
    str(fold): metrics[str(fold)]["incumbent_6p377"]["rmse"]
    - metrics[str(fold)]["blend_w0p30"]["rmse"]
    for fold in HONEST_FOLDS
}
far_gains = {
    str(fold): metrics[str(fold)]["incumbent_6p377"]["far_gt2000_rmse"]
    - metrics[str(fold)]["blend_w0p30"]["far_gt2000_rmse"]
    for fold in HONEST_FOLDS
}
fallbacks = int(sum(report["status"] != "built" for report in reports))
checks = {
    "honest_fold_coverage_exact": bool(observed_wells == expected_wells),
    "fallback_wells_le_3": bool(fallbacks <= 3),
    "gain_positive_3of3": bool(all(value > 0.0 for value in fold_gains.values())),
    "overall_gain_ge_0p20": bool(
        metrics["0"]["incumbent_6p377"]["rmse"]
        - metrics["0"]["blend_w0p30"]["rmse"] >= 0.20
    ),
    "far_gain_nonnegative_3of3": bool(all(value >= 0.0 for value in far_gains.values())),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fixed_recipe_leave_one_of_folds123_out_proxy_for_all773_refit",
    "sample_unit": "one complete physical well",
    "truth_usage": "rank labels from the four non-heldout folds only",
    "protocol": {
        "structural_prior_folds": list(PRIOR_FOLDS),
        "honest_validation_folds": list(HONEST_FOLDS),
        "excluded_from_validation": {
            "4": "used to fit frozen structural prior",
            "5": "used to fit frozen structural prior",
        },
        "iterations": FROZEN_ITERATIONS,
        "blend_weight": FROZEN_WEIGHT,
        "selection_or_tuning_count": 0,
    },
    "coverage": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "fold_wells": observed_wells,
        "missing_training_samples": missing_samples,
        "fallback_wells": fallbacks,
    },
    "models": model_manifest,
    "metrics": metrics,
    "fold_gain": fold_gains,
    "fold_far_gain": far_gains,
    "v67_fold12_comparison": v67_comparison,
    "checks": checks,
    "strict_pass": strict_pass,
    "output": str(OUT_DIR / "all773_proxy_crossfit_oof.parquet"),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> submit canonical345 first and all773 as the second LB A/B"
        if strict_pass
        else "FAIL -> do not submit all773; retain the canonical345 deployment model"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV70 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
if v67_comparison is not None:
    print("\nV70 VS V67 ON MATCHED FOLD1/2")
    print(json.dumps(_clean(v67_comparison), indent=2), flush=True)
print("\nV70 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)

