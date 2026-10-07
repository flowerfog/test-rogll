"""V103: matched standalone-label versus blend-aware-label V4 ranker.

Candidate bank, features, sampled candidates, LightGBM family, seed and 67
iterations are frozen.  Only the complete-well ranking target changes:

    old: SSE(candidate, truth)
    new: SSE(parent + 0.30 * (candidate - parent), truth)

Fold45 trains the two matched rankers. Fold3 is split into the same 24-well
discovery and 130-well confirmation used by V102. Only after the frozen target
passes confirmation are Fold345 labels fit and Fold2 touched once. Fold1 is
never evaluated.
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
EXPERIMENT = "rawdata_lattice_blendaware_ranker_v103"
OUT_DIR = OOF_ROOT / EXPERIMENT
LABEL_DIR = OUT_DIR / "blend_labels"
PANEL_DIR = OUT_DIR / "matched_panels"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, LABEL_DIR, PANEL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V81_SOURCE = ROOT / "rawdata_lattice_structural_risk_ranker_v81b_cell.py"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
LABEL_DOSE = 0.30
FROZEN_ITERATIONS = 67
RANDOM_SEED = 20260721
PARENT = "incumbent_6p377_replay"
TRUTH = "tvt_true"


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


def _bootstrap_v81():
    if not V81_SOURCE.is_file():
        raise FileNotFoundError(V81_SOURCE)
    source = V81_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V81B EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V81b bootstrap marker missing from {V81_SOURCE}")
    module = types.ModuleType("raw_lattice_v81_blendaware_definitions")
    module.__file__ = str(V81_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V81_SOURCE), "exec"), module.__dict__)
    return module


N = _bootstrap_v81()
Q = N.Q
V = N.V
B = N.B
prior = N.prior
incumbent = N.incumbent
fold_by_well = N.fold_by_well
feature_names = list(N.feature_names)
available = set(N.available)

if not V76_WEIGHTS.is_file():
    raise FileNotFoundError(V76_WEIGHTS)
weights = pd.read_parquet(V76_WEIGHTS)
required_weight_columns = {"well", "fold", "ensemble_weight"}
missing = sorted(required_weight_columns.difference(weights.columns))
if missing:
    raise RuntimeError(f"V76 weights missing columns: {missing}")
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))


def _panel_path(well):
    return PANEL_DIR / f"{well}.npz"


def _load_panel(well):
    with np.load(_panel_path(well), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _build_blend_labels(well):
    destination = _panel_path(well)
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as data:
            if (
                int(data["panel_version"][0]) == 1
                and float(data["label_dose"][0]) == LABEL_DOSE
                and data["features"].shape[1] == len(feature_names)
            ):
                return {"well": well, "status": "reused", "reason": ""}

    record = V["_load_record"](well)
    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        features, names = Q._candidate_evidence(
            record, prior, paths, segments, arm_rank, arm_score_z, source_mask
        )
    if list(names) != feature_names:
        raise RuntimeError(f"{well}: feature schema mismatch")
    local = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable")
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    parent = local[PARENT].to_numpy(np.float64)
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: incumbent/eval rows {len(local)}/{len(eval_rows)}")
    if np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: truth alignment")

    standalone_mse = np.empty(len(paths), np.float64)
    blend_mse = np.empty(len(paths), np.float64)
    parent_error = parent - truth
    for index, path in enumerate(paths):
        prediction = B["_path_prediction"](record, prior, path, segments)[eval_rows].astype(np.float64)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{well}: non-finite candidate {index}")
        raw_error = prediction - truth
        move = prediction - parent
        blend_error = parent_error + LABEL_DOSE * move
        standalone_mse[index] = float(np.mean(raw_error * raw_error))
        blend_mse[index] = float(np.mean(blend_error * blend_error))

    np.savez_compressed(
        destination,
        panel_version=np.asarray([1], np.int8),
        candidate_count=np.asarray([len(paths)], np.int32),
        label_dose=np.asarray([LABEL_DOSE], np.float64),
        features=features.astype(np.float32),
        source_mask=source_mask.astype(np.uint8),
        arm_rank=arm_rank.astype(np.float32),
        paths=paths.astype(np.int16),
        mse=standalone_mse.astype(np.float32),
        blend_mse=blend_mse.astype(np.float32),
        rows=np.asarray([len(eval_rows)], np.int64),
        same_oracle=np.asarray([int(np.argmin(standalone_mse) == np.argmin(blend_mse))], np.int8),
    )
    return {"well": well, "status": "built", "reason": "", "candidates": len(paths)}


label_wells = sorted(
    well for well, fold in fold_by_well.items() if fold in (3, 4, 5) and well in available
)
label_reports = []
for number, well in enumerate(label_wells, 1):
    try:
        label_reports.append(_build_blend_labels(well))
    except Exception as exc:
        label_reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 10 == 0 or number == len(label_wells):
        failures = sum(row["status"] == "failed" for row in label_reports)
        print(
            f"V103 blend labels {number}/{len(label_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

label_report = pd.DataFrame(label_reports)
label_report.to_csv(OUT_DIR / "label_build_report.csv", index=False)
failed = label_report[label_report["status"].eq("failed")]
if len(failed):
    raise RuntimeError(f"{len(failed)} label failures: {failed.head().to_dict('records')}")


def _target(panel, well, target):
    if target == "standalone":
        return panel["mse"].astype(np.float64)
    if target == "blendaware":
        values = panel["blend_mse"].astype(np.float64)
        if len(values) != len(panel["mse"]):
            raise RuntimeError(f"{well}: blend label length mismatch")
        return values
    raise ValueError(target)


def _matrix(wells, target):
    x_parts, y_parts, groups = [], [], []
    for number, well in enumerate(wells, 1):
        panel = _load_panel(well)
        indices = Q._training_indices(panel, well)
        x_parts.append(panel["features"][indices].astype(np.float32, copy=False))
        y_parts.append(Q._rank_labels(_target(panel, well, target)[indices]))
        groups.append(int(len(indices)))
        if number % 100 == 0 or number == len(wells):
            print(f"V103 matrix {target} {number}/{len(wells)}", flush=True)
    return np.concatenate(x_parts), np.concatenate(y_parts), groups


def _fit(wells, target, label):
    x, y, groups = _matrix(wells, target)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=(1, 8, 32),
        n_estimators=FROZEN_ITERATIONS, learning_rate=0.025,
        num_leaves=31, max_depth=-1, min_child_samples=80,
        subsample=0.85, colsample_bytree=0.80,
        reg_alpha=0.2, reg_lambda=2.0, random_state=RANDOM_SEED,
        n_jobs=max(1, min(16, os.cpu_count() or 1)), verbosity=-1,
    )
    model.fit(x, y, group=groups)
    path = MODEL_DIR / f"{label}.txt"
    model.booster_.save_model(str(path))
    print("V103 MODEL", {"label": label, "rows": len(x), "wells": len(groups)}, flush=True)
    del x, y
    gc.collect()
    return model.booster_


def _evaluate(wells, models, label, use_panels):
    parts = []
    reports = []
    for number, well in enumerate(wells, 1):
        local = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        parent = local[PARENT].to_numpy(np.float64)
        report = {"well": well, "stage": label, "status": "built", "reason": ""}
        try:
            record = V["_load_record"](well)
            if use_panels:
                panel = _load_panel(well)
                features = panel["features"].astype(np.float32, copy=False)
                paths = panel["paths"].astype(np.int16, copy=False)
                segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
                if len(features) != len(paths) or paths.shape[1] != len(segments):
                    raise RuntimeError(
                        f"panel shape features={len(features)} paths={paths.shape} segments={len(segments)}"
                    )
            else:
                paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                    features, names = Q._candidate_evidence(
                        record, prior, paths, segments, arm_rank, arm_score_z, source_mask
                    )
                if list(names) != feature_names:
                    raise RuntimeError("feature schema mismatch")
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
            if len(local) != len(eval_rows):
                raise RuntimeError(f"row count {len(local)}/{len(eval_rows)}")
            if np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth alignment")
            if np.max(np.abs(local["md_since"].to_numpy(np.float64) - distance)) > 1.01:
                raise RuntimeError("distance alignment")
            if well not in weight_map:
                raise RuntimeError("missing V76 dose")
            dose = float(weight_map[well])
            report["dose"] = dose
            report["candidate_count"] = int(len(paths))
            for model_id, model in models.items():
                scores = model.predict(features)
                chosen = int(np.argmax(scores))
                prediction = B["_path_prediction"](record, prior, paths[chosen], segments)[eval_rows]
                if not np.isfinite(prediction).all():
                    raise RuntimeError(f"{model_id}: non-finite path")
                local[f"coarse_{model_id}"] = np.asarray(prediction, np.float32)
                local[f"blend_{model_id}"] = (
                    parent + dose * (np.asarray(prediction, np.float64) - parent)
                ).astype(np.float32)
                report[f"chosen_{model_id}"] = chosen
        except Exception as exc:
            report["status"] = "parent_fallback"
            report["reason"] = repr(exc)
            for model_id in models:
                local[f"coarse_{model_id}"] = parent.astype(np.float32)
                local[f"blend_{model_id}"] = parent.astype(np.float32)
        parts.append(local)
        reports.append(report)
        if number % 10 == 0 or number == len(wells):
            fallbacks = sum(row["status"] != "built" for row in reports)
            print(
                f"V103 evaluate {label} {number}/{len(wells)} fallbacks={fallbacks} "
                f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
            )
    return pd.concat(parts, ignore_index=True), reports


def _summaries(frame, models, stage):
    rows = [{"stage": stage, "model": "parent", **N._metric(frame, frame[PARENT], "parent")}]
    for model_id in models:
        rows.append({
            "stage": stage, "model": model_id,
            **N._metric(frame, frame[f"coarse_{model_id}"], "coarse"),
        })
        rows.append({
            "stage": stage, "model": model_id,
            **N._metric(frame, frame[f"blend_{model_id}"], "blend_v76"),
        })
    return rows


train45 = sorted(well for well, fold in fold_by_well.items() if fold in TRAIN_FOLDS and well in available)
fold3 = sorted(well for well, fold in fold_by_well.items() if fold == SELECTION_FOLD and well in available)
discovery_wells = fold3[:DISCOVERY_WELLS]
confirmation_wells = fold3[DISCOVERY_WELLS:]
fold2 = sorted(well for well, fold in fold_by_well.items() if fold == APPLICATION_FOLD)
if len(train45) < 307 or len(fold3) < 153 or len(fold2) < 154:
    raise RuntimeError(f"Insufficient coverage train45={len(train45)} fold3={len(fold3)} fold2={len(fold2)}")

print("V103 CONFIG", {
    "train45": len(train45), "fold3": len(fold3), "fold2": len(fold2),
    "features": len(feature_names), "iterations": FROZEN_ITERATIONS,
    "label_dose": LABEL_DOSE,
}, flush=True)

models45 = {
    "standalone": _fit(train45, "standalone", "fold45_standalone"),
    "blendaware": _fit(train45, "blendaware", "fold45_blendaware"),
}
fold3_frame, fold3_reports = _evaluate(fold3, models45, "fold3", use_panels=True)
pd.DataFrame(fold3_reports).to_csv(OUT_DIR / "fold3_evaluation_report.csv", index=False)
discovery_frame = fold3_frame[fold3_frame["well"].isin(discovery_wells)].copy()
confirmation_frame = fold3_frame[fold3_frame["well"].isin(confirmation_wells)].copy()
summary_rows = _summaries(discovery_frame, models45, "fold3_discovery")
summary_rows += _summaries(confirmation_frame, models45, "fold3_confirmation")
summary = pd.DataFrame(summary_rows)


def _row(table, stage, model):
    return table[
        table["stage"].eq(stage)
        & table["model"].eq(model)
        & table["endpoint"].eq("blend_v76")
    ].iloc[0]


discovery_base = _row(summary, "fold3_discovery", "standalone")
discovery_new = _row(summary, "fold3_discovery", "blendaware")
confirmation_base = _row(summary, "fold3_confirmation", "standalone")
confirmation_new = _row(summary, "fold3_confirmation", "blendaware")
discovery_gain = float(discovery_base["rmse"] - discovery_new["rmse"])
confirmation_gain = float(confirmation_base["rmse"] - confirmation_new["rmse"])
confirmation_far_gain = float(
    confirmation_base["far_gt2000_rmse"] - confirmation_new["far_gt2000_rmse"]
)
confirmation_p95_gain = float(
    confirmation_base["p95_well_rmse"] - confirmation_new["p95_well_rmse"]
)
confirmation_pass = bool(
    discovery_gain > 0.0
    and confirmation_gain >= 0.05
    and confirmation_far_gain >= 0.0
    and confirmation_p95_gain >= -0.10
)

fold2_frame = None
fold2_reports = []
application_pass = False
application_gain = None
application_far_gain = None
application_p95_gain = None
if confirmation_pass:
    train345 = sorted(
        well for well, fold in fold_by_well.items() if fold in (3, 4, 5) and well in available
    )
    models345 = {
        "standalone": _fit(train345, "standalone", "fold345_standalone"),
        "blendaware": _fit(train345, "blendaware", "fold345_blendaware"),
    }
    fold2_frame, fold2_reports = _evaluate(fold2, models345, "fold2", use_panels=False)
    summary_rows += _summaries(fold2_frame, models345, "fold2_application")
    summary = pd.DataFrame(summary_rows)
    fold2_base = _row(summary, "fold2_application", "standalone")
    fold2_new = _row(summary, "fold2_application", "blendaware")
    application_gain = float(fold2_base["rmse"] - fold2_new["rmse"])
    application_far_gain = float(fold2_base["far_gt2000_rmse"] - fold2_new["far_gt2000_rmse"])
    application_p95_gain = float(fold2_base["p95_well_rmse"] - fold2_new["p95_well_rmse"])
    application_pass = bool(
        application_gain >= 0.05
        and application_far_gain >= 0.0
        and application_p95_gain >= -0.10
    )
    pd.DataFrame(fold2_reports).to_csv(OUT_DIR / "fold2_evaluation_report.csv", index=False)
    keep = [
        "id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT,
        "coarse_standalone", "blend_standalone", "coarse_blendaware", "blend_blendaware",
    ]
    fold2_frame[keep].to_parquet(
        OUT_DIR / "fold2_blendaware_ranker_oof.parquet", index=False, compression="zstd"
    )

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
print("\nV103 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)

strict_pass = bool(confirmation_pass and application_pass)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_matched_label_AB_Fold3_confirmation_Fold2_single_application",
    "sample_unit": "one complete physical well",
    "matched_constants": {
        "candidate_bank": "V4 frozen three-arm union",
        "features": len(feature_names),
        "iterations": FROZEN_ITERATIONS,
        "model_family": "LightGBM LambdaRank",
        "training_candidates_per_well": int(Q.TRAIN_CANDIDATES_PER_WELL),
        "label_dose": LABEL_DOSE,
    },
    "changed_variable": "standalone path MSE labels versus parent-blended SSE labels",
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "application_fold": APPLICATION_FOLD if confirmation_pass else None,
        "fold1_touched": False,
    },
    "fold3": {
        "discovery_gain": discovery_gain,
        "confirmation_gain": confirmation_gain,
        "confirmation_far_gain": confirmation_far_gain,
        "confirmation_p95_gain": confirmation_p95_gain,
        "pass": confirmation_pass,
    },
    "fold2": None if not confirmation_pass else {
        "gain": application_gain,
        "far_gain": application_far_gain,
        "p95_gain": application_p95_gain,
        "pass": application_pass,
    },
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train the fixed blend-aware ranker on all 773 labels and export a Kaggle bundle"
        if strict_pass
        else "FAIL -> V102 oracle ceiling is not learnable with current candidate evidence; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV103 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
