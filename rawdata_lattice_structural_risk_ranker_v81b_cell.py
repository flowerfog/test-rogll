"""Matched structural-risk target A/B for the frozen V4 coarse endpoint.

The original V4 target already ranks candidates by complete-well MSE.  This
experiment keeps candidates, features, LightGBM and the 0.30 deployment dose
fixed, and changes only the ranking target by adding far-tail MSE and squared
whole-well MAE.  Fold-3 discovery selects a target, the remaining Fold-3 wells
confirm it, and Fold-2 is touched once after that confirmation passes.
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
EXPERIMENT = "rawdata_lattice_structural_risk_ranker_v81b"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
SUPPLEMENT_PANEL_DIR = OUT_DIR / "supplement_panels"
for directory in (OUT_DIR, MODEL_DIR, SUPPLEMENT_PANEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V4_PANEL_DIR = OOF_ROOT / "rawdata_lattice_coarse_ranker_v4" / "panels"
V66_PATH = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
V67_PRIOR = (
    OOF_ROOT / "rawdata_lattice_coarse_canonical_gateway_v67"
    / "canonical_fold45_prior.json"
)

INCUMBENT_COLUMN = "incumbent_6p377_replay"
TRUTH_COLUMN = "tvt_true"
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
FROZEN_ITERATIONS = 67
FROZEN_WEIGHT = 0.30
RANDOM_SEED = 20260721

# Both extra terms have units ft^2.  Baseline is included to make stopping a
# valid outcome rather than forcing a new target to win.
RECIPES = (
    {"id": "mse", "far_weight": 0.0, "coherent_weight": 0.0},
    {"id": "far025", "far_weight": 0.25, "coherent_weight": 0.0},
    {"id": "coh025", "far_weight": 0.0, "coherent_weight": 0.25},
    {"id": "far025_coh025", "far_weight": 0.25, "coherent_weight": 0.25},
    {"id": "far050_coh025", "far_weight": 0.50, "coherent_weight": 0.25},
    {"id": "far025_coh050", "far_weight": 0.25, "coherent_weight": 0.50},
)


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


def _bootstrap_v4():
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 bootstrap marker missing from {V4_SOURCE}")
    module = types.ModuleType("raw_lattice_v4_structural_target_definitions")
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
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


for required in (V4_SOURCE, V4_PANEL_DIR, V66_PATH, V67_PRIOR):
    if not required.exists():
        raise FileNotFoundError(required)

Q = _bootstrap_v4()
V = Q.V
B = Q.B
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
fold_by_well = incumbent.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
if len(incumbent) != 3_783_989 or incumbent["well"].nunique() != 773:
    raise RuntimeError("V66 is not the complete 773-well lineage")


def _panel_path(well):
    old = V4_PANEL_DIR / f"{well}.npz"
    supplement = SUPPLEMENT_PANEL_DIR / f"{well}.npz"
    if old.is_file():
        return old
    return supplement


# Recover the frozen schema before supplementing deleted V4 panels.
feature_names = None
for metadata_path in sorted(V4_PANEL_DIR.glob("*.json")):
    if metadata_path.is_file():
        feature_names = json.loads(metadata_path.read_text(encoding="utf-8"))["feature_names"]
        break
if feature_names is None:
    raise RuntimeError("No V4 feature metadata found")


def _build_supplement_panel(well):
    path = SUPPLEMENT_PANEL_DIR / f"{well}.npz"
    if path.is_file():
        return {"well": well, "status": "reused", "reason": ""}
    try:
        record = V["_load_record"](well)
        paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            features, names = Q._candidate_evidence(
                record, prior, paths, segments, arm_rank, arm_score_z, source_mask
            )
        if list(names) != list(feature_names):
            raise RuntimeError("feature schema mismatch")
        targets = Q._exact_targets(record, prior, paths, segments)
        np.savez_compressed(
            path,
            features=features.astype(np.float32),
            source_mask=source_mask,
            arm_rank=arm_rank,
            mse=targets["mse"].astype(np.float32),
            sse=targets["sse"],
            sae=targets["sae"],
            far_sse=targets["far_sse"],
            rows=np.asarray([targets["rows"]], np.int64),
            far_rows=np.asarray([targets["far_rows"]], np.int64),
        )
        _write_json(
            path.with_suffix(".json"),
            {"well": well, "candidates": len(paths), "feature_names": feature_names},
        )
        return {"well": well, "status": "built", "reason": ""}
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}


available_before = {
    well for well in fold_by_well if (V4_PANEL_DIR / f"{well}.npz").is_file()
}
required_training_wells = sorted(
    well for well, fold in fold_by_well.items() if fold in (3, 4, 5)
)
missing_training_wells = [well for well in required_training_wells if well not in available_before]
supplement_reports = []
for number, well in enumerate(missing_training_wells, 1):
    report = _build_supplement_panel(well)
    supplement_reports.append(report)
    if number % 10 == 0 or number == len(missing_training_wells):
        failures = sum(row["status"] == "failed" for row in supplement_reports)
        print(
            f"V81b supplement panels {number}/{len(missing_training_wells)} "
            f"failures={failures} elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )
pd.DataFrame(supplement_reports).to_csv(OUT_DIR / "supplement_panel_report.csv", index=False)

available = {well for well in fold_by_well if _panel_path(well).is_file()}
coverage = {
    str(fold): sum(well in available for well, value in fold_by_well.items() if value == fold)
    for fold in range(1, 6)
}
# Fold2 is evaluated by raw reconstruction and does not require a cached panel.
if coverage["3"] < 153 or coverage["4"] < 153 or coverage["5"] < 153:
    raise RuntimeError(f"Incomplete training panels after supplement: {coverage}")


def _load_panel(well):
    with np.load(_panel_path(well), allow_pickle=False) as data:
        panel = {key: data[key] for key in data.files}
    if panel["features"].shape[1] != len(feature_names):
        raise RuntimeError(f"{well}: feature schema mismatch")
    return panel


def _risk(panel, recipe, shuffle=False, well=""):
    rows = max(int(panel["rows"][0]), 1)
    far_rows = int(panel["far_rows"][0])
    base = panel["sse"].astype(np.float64) / rows
    far = panel["far_sse"].astype(np.float64) / far_rows if far_rows else base.copy()
    coherent = (panel["sae"].astype(np.float64) / rows) ** 2
    if shuffle:
        seed = int.from_bytes(hashlib.sha256(f"structural:{well}".encode()).digest()[:8], "little")
        order = np.random.default_rng(seed).permutation(len(base))
        far = far[order]
        coherent = coherent[order]
    return (
        base
        + float(recipe["far_weight"]) * far
        + float(recipe["coherent_weight"]) * coherent
    )


def _matrix(wells, recipe, shuffle=False):
    x_parts, y_parts, groups = [], [], []
    used = []
    for number, well in enumerate(wells, 1):
        if well not in available:
            continue
        panel = _load_panel(well)
        indices = Q._training_indices(panel, well)
        x_parts.append(panel["features"][indices].astype(np.float32, copy=False))
        y_parts.append(Q._rank_labels(_risk(panel, recipe, shuffle, well)[indices]))
        groups.append(int(len(indices)))
        used.append(well)
        if number % 100 == 0 or number == len(wells):
            print(f"V81b matrix {recipe['id']} {number}/{len(wells)}", flush=True)
    return np.concatenate(x_parts), np.concatenate(y_parts), groups, used


def _fit(wells, recipe, label, shuffle=False):
    x, y, groups, used = _matrix(wells, recipe, shuffle)
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
    print("V81b MODEL", {"label": label, "rows": len(x), "wells": len(groups)}, flush=True)
    del x, y
    gc.collect()
    return model.booster_, used


def _evaluate(wells, models, label):
    parts = []
    reports = []
    for number, well in enumerate(wells, 1):
        parent = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        parent_prediction = parent[INCUMBENT_COLUMN].to_numpy(np.float64)
        report = {"well": well, "stage": label, "status": "built", "reason": ""}
        try:
            record = V["_load_record"](well)
            paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                features, names = Q._candidate_evidence(
                    record, prior, paths, segments, arm_rank, arm_score_z, source_mask
                )
            if list(names) != list(feature_names):
                raise RuntimeError("feature schema mismatch")
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
            if len(parent) != len(eval_rows):
                raise RuntimeError(f"row count {len(parent)}/{len(eval_rows)}")
            if np.max(np.abs(parent[TRUTH_COLUMN].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth alignment")
            if np.max(np.abs(parent["md_since"].to_numpy(np.float64) - distance)) > 1.01:
                raise RuntimeError("distance alignment")
            chosen = {}
            for model_id, model in models.items():
                scores = model.predict(features)
                index = int(np.argmax(scores))
                prediction = B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
                if not np.isfinite(prediction).all():
                    raise RuntimeError(f"{model_id}: non-finite path")
                parent[f"coarse_{model_id}"] = np.asarray(prediction, np.float32)
                parent[f"blend_{model_id}"] = (
                    parent_prediction + FROZEN_WEIGHT * (prediction - parent_prediction)
                ).astype(np.float32)
                chosen[model_id] = index
            report["candidate_count"] = int(len(paths))
            report["chosen"] = chosen
        except Exception as exc:
            report["status"] = "parent_fallback"
            report["reason"] = repr(exc)
            for model_id in models:
                parent[f"coarse_{model_id}"] = parent_prediction.astype(np.float32)
                parent[f"blend_{model_id}"] = parent_prediction.astype(np.float32)
        parts.append(parent)
        reports.append(report)
        if number % 10 == 0 or number == len(wells):
            fallbacks = sum(row["status"] != "built" for row in reports)
            print(
                f"V81b evaluate {label} {number}/{len(wells)} fallbacks={fallbacks} "
                f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
            )
    return pd.concat(parts, ignore_index=True), reports


def _summaries(frame, models, stage):
    rows = []
    rows.append({"stage": stage, "model": "parent", **_metric(frame, frame[INCUMBENT_COLUMN], "parent")})
    for model_id in models:
        rows.append({"stage": stage, "model": model_id, **_metric(frame, frame[f"coarse_{model_id}"], "coarse")})
        rows.append({"stage": stage, "model": model_id, **_metric(frame, frame[f"blend_{model_id}"], "blend_w0p30")})
    return rows


# --- V81B EXECUTION ---

train45 = sorted(well for well, fold in fold_by_well.items() if fold in TRAIN_FOLDS and well in available)
fold3 = sorted(well for well, fold in fold_by_well.items() if fold == SELECTION_FOLD and well in available)
discovery_wells = fold3[:DISCOVERY_WELLS]
confirmation_wells = fold3[DISCOVERY_WELLS:]
fold2 = sorted(well for well, fold in fold_by_well.items() if fold == APPLICATION_FOLD and well in available)

print("V81b CONFIG", {
    "panel_coverage": coverage, "train45": len(train45), "fold3": len(fold3),
    "fold2": len(fold2), "recipes": [row["id"] for row in RECIPES],
}, flush=True)

# One model per target; all other training choices are exactly matched.
selection_models = {}
for recipe in RECIPES:
    selection_models[recipe["id"]], _ = _fit(train45, recipe, f"fold45_{recipe['id']}")

fold3_frame, fold3_reports = _evaluate(fold3, selection_models, "fold3")
discovery_frame = fold3_frame[fold3_frame["well"].isin(discovery_wells)].copy()
confirmation_frame = fold3_frame[fold3_frame["well"].isin(confirmation_wells)].copy()
selection_rows = _summaries(discovery_frame, selection_models, "fold3_discovery")
confirmation_rows = _summaries(confirmation_frame, selection_models, "fold3_confirmation")

selection_table = pd.DataFrame(selection_rows)
blend_selection = selection_table[selection_table["endpoint"].eq("blend_w0p30")].copy()
selected_id = str(blend_selection.sort_values(["rmse", "model"], kind="stable").iloc[0]["model"])
selected_recipe = next(row for row in RECIPES if row["id"] == selected_id)

confirmation_table = pd.DataFrame(confirmation_rows)
base_confirmation = confirmation_table[
    confirmation_table["model"].eq("mse") & confirmation_table["endpoint"].eq("blend_w0p30")
].iloc[0]
selected_confirmation = confirmation_table[
    confirmation_table["model"].eq(selected_id) & confirmation_table["endpoint"].eq("blend_w0p30")
].iloc[0]
fold3_gain = float(base_confirmation["rmse"] - selected_confirmation["rmse"])
fold3_far_gain = float(base_confirmation["far_gt2000_rmse"] - selected_confirmation["far_gt2000_rmse"])

confirmation_pass = bool(
    selected_id != "mse" and fold3_gain >= 0.03 and fold3_far_gain >= 0.05
    and float(selected_confirmation["near_0_500_rmse"] - base_confirmation["near_0_500_rmse"]) <= 0.03
)

application_rows = []
fold2_frame = None
fold2_reports = []
application_pass = False
if confirmation_pass:
    train345 = sorted(
        well for well, fold in fold_by_well.items() if fold in (3, 4, 5) and well in available
    )
    application_models = {}
    application_models["mse"], _ = _fit(train345, RECIPES[0], "fold345_mse")
    application_models[selected_id], _ = _fit(train345, selected_recipe, f"fold345_{selected_id}")
    application_models["shuffled_structural"], _ = _fit(
        train345, selected_recipe, "fold345_shuffled_structural", shuffle=True
    )
    fold2_frame, fold2_reports = _evaluate(fold2, application_models, "fold2")
    application_rows = _summaries(fold2_frame, application_models, "fold2_application")
    application_table = pd.DataFrame(application_rows)
    def _blend_row(model_id):
        return application_table[
            application_table["model"].eq(model_id) & application_table["endpoint"].eq("blend_w0p30")
        ].iloc[0]
    base2 = _blend_row("mse")
    selected2 = _blend_row(selected_id)
    shuffled2 = _blend_row("shuffled_structural")
    gain2 = float(base2["rmse"] - selected2["rmse"])
    far_gain2 = float(base2["far_gt2000_rmse"] - selected2["far_gt2000_rmse"])
    shuffle_loss = float(shuffled2["rmse"] - selected2["rmse"])
    application_pass = bool(
        gain2 >= 0.03 and far_gain2 >= 0.05
        and float(selected2["near_0_500_rmse"] - base2["near_0_500_rmse"]) <= 0.03
        and shuffle_loss >= 0.02
    )
else:
    gain2 = far_gain2 = shuffle_loss = np.nan

summary = pd.concat([
    selection_table, confirmation_table, pd.DataFrame(application_rows)
], ignore_index=True)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(fold3_reports + fold2_reports).to_json(
    OUT_DIR / "evaluation_report.jsonl", orient="records", lines=True
)
if fold2_frame is not None:
    keep = [
        "id", "well", "row_idx", "fold", TRUTH_COLUMN, "md_since", INCUMBENT_COLUMN,
        "blend_mse", f"blend_{selected_id}", "blend_shuffled_structural",
    ]
    fold2_frame[keep].to_parquet(OUT_DIR / "fold2_matched_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "matched_complete_path_target_decomposition_Fold3_selection_confirmation_Fold2_application",
    "sample_unit": "one complete physical well",
    "candidate_features_model_and_blend_weight_fixed": True,
    "original_target": "complete-well MSE",
    "candidate_target": "MSE + far_weight*far_MSE + coherent_weight*MAE^2",
    "protocol": {
        "train_folds": list(TRAIN_FOLDS), "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_wells), "confirmation_wells": len(confirmation_wells),
        "application_fold": APPLICATION_FOLD if confirmation_pass else None,
        "fold1_touched": False, "iterations": FROZEN_ITERATIONS,
        "blend_weight": FROZEN_WEIGHT,
    },
    "selected_recipe": selected_recipe,
    "fold3_confirmation": {
        "gain_vs_mse_target": fold3_gain, "far_gain_vs_mse_target": fold3_far_gain,
        "pass": confirmation_pass,
    },
    "fold2_application": None if not confirmation_pass else {
        "gain_vs_mse_target": gain2, "far_gain_vs_mse_target": far_gain2,
        "aligned_gain_vs_shuffled_target": shuffle_loss, "pass": application_pass,
    },
    "strict_pass": bool(confirmation_pass and application_pass),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild the V76 adaptive-dose endpoint from this coarse model"
        if confirmation_pass and application_pass
        else "FAIL -> retain V76 and close structural target reweighting"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV81b MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV81b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
