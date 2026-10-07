"""Top-K candidate classification versus the frozen V4 LambdaRank selector.

V4 optimizes a full candidate ordering although deployment consumes only the
argmax.  This matched experiment keeps candidates, 131 features, tree capacity,
67 boosting rounds and the 0.30 endpoint dose fixed.  Fold-3 selects K, Fold-2
confirms once, and Fold-1 is touched only after confirmation passes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_topk_selector_v82"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V81_SOURCE = ROOT / "rawdata_lattice_structural_risk_ranker_v81b_cell.py"
RANDOM_SEED = 20260721
ITERATIONS = 67
BLEND_WEIGHT = 0.30
RECIPES = (
    {"id": "lambdarank", "family": "rank", "topk": None},
    {"id": "top8", "family": "binary", "topk": 8},
    {"id": "top16", "family": "binary", "topk": 16},
    {"id": "top32", "family": "binary", "topk": 32},
    {"id": "top64", "family": "binary", "topk": 64},
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


def _bootstrap_v81():
    if not V81_SOURCE.is_file():
        raise FileNotFoundError(V81_SOURCE)
    source = V81_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V81B EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"Bootstrap marker missing from {V81_SOURCE}; upload the updated V81b file")
    module = types.ModuleType("raw_lattice_v81b_topk_definitions")
    module.__file__ = str(V81_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V81_SOURCE), "exec"), module.__dict__)
    return module


N = _bootstrap_v81()
Q = N.Q
incumbent = N.incumbent
fold_by_well = N.fold_by_well
available = set(N.available)


def _labels(panel, indices, recipe, shuffle, well):
    risk = panel["mse"].astype(np.float64)[indices]
    if recipe["family"] == "rank":
        labels = Q._rank_labels(risk)
    else:
        k = min(int(recipe["topk"]), max(len(indices) - 1, 1))
        order = np.argsort(risk, kind="stable")
        labels = np.zeros(len(indices), np.int32)
        labels[order[:k]] = 1
    if shuffle:
        seed = int.from_bytes(hashlib.sha256(f"topk:{well}".encode()).digest()[:8], "little")
        labels = labels[np.random.default_rng(seed).permutation(len(labels))]
    return labels


def _matrix(wells, recipe, shuffle=False):
    x_parts, y_parts, weight_parts, groups, used = [], [], [], [], []
    for number, well in enumerate(wells, 1):
        if well not in available:
            continue
        panel = N._load_panel(well)
        indices = Q._training_indices(panel, well)
        x = panel["features"][indices].astype(np.float32, copy=False)
        y = _labels(panel, indices, recipe, shuffle, well)
        if recipe["family"] == "binary":
            positive = int(y.sum())
            negative = int(len(y) - positive)
            if positive == 0 or negative == 0:
                continue
            weights = np.where(
                y == 1,
                0.5 * len(y) / positive,
                0.5 * len(y) / negative,
            ).astype(np.float32)
        else:
            weights = np.ones(len(y), np.float32)
        x_parts.append(x)
        y_parts.append(y)
        weight_parts.append(weights)
        groups.append(int(len(y)))
        used.append(well)
        if number % 100 == 0 or number == len(wells):
            print(f"V82 matrix {recipe['id']} {number}/{len(wells)}", flush=True)
    return (
        np.concatenate(x_parts), np.concatenate(y_parts),
        np.concatenate(weight_parts), groups, used,
    )


def _model_params():
    return dict(
        n_estimators=ITERATIONS, learning_rate=0.025, num_leaves=31,
        max_depth=-1, min_child_samples=80, subsample=0.85,
        colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
        random_state=RANDOM_SEED,
        n_jobs=max(1, min(16, os.cpu_count() or 1)), verbosity=-1,
    )


def _fit(wells, recipe, label, shuffle=False):
    x, y, weights, groups, used = _matrix(wells, recipe, shuffle)
    if recipe["family"] == "rank":
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=(1, 8, 32), **_model_params())
        model.fit(x, y, group=groups)
    else:
        model = lgb.LGBMClassifier(objective="binary", **_model_params())
        model.fit(x, y, sample_weight=weights)
    path = MODEL_DIR / f"{label}.txt"
    model.booster_.save_model(str(path))
    print("V82 MODEL", {
        "label": label, "family": recipe["family"], "rows": len(x),
        "wells": len(groups), "positive_fraction": float(np.mean(y == 1)) if recipe["family"] == "binary" else None,
    }, flush=True)
    del x, y, weights
    gc.collect()
    return model.booster_, used


def _summary(frame, models, stage):
    rows = [{"stage": stage, "model": "parent", **N._metric(frame, frame[N.INCUMBENT_COLUMN], "parent")}]
    for model_id in models:
        rows.append({"stage": stage, "model": model_id, **N._metric(frame, frame[f"coarse_{model_id}"], "coarse")})
        rows.append({"stage": stage, "model": model_id, **N._metric(frame, frame[f"blend_{model_id}"], "blend_w0p30")})
    return rows


def _blend_row(table, model):
    return table[table["model"].eq(model) & table["endpoint"].eq("blend_w0p30")].iloc[0]


def _gains(table, selected):
    base = _blend_row(table, "lambdarank")
    candidate = _blend_row(table, selected)
    return {
        "gain": float(base["rmse"] - candidate["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        "near_degradation": float(candidate["near_0_500_rmse"] - base["near_0_500_rmse"]),
        "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
    }


# --- V82 EXECUTION ---

train45 = sorted(well for well, fold in fold_by_well.items() if fold in (4, 5) and well in available)
fold3 = sorted(well for well, fold in fold_by_well.items() if fold == 3)
fold2 = sorted(well for well, fold in fold_by_well.items() if fold == 2)
fold1 = sorted(well for well, fold in fold_by_well.items() if fold == 1)
print("V82 CONFIG", {
    "train45": len(train45), "fold3": len(fold3), "fold2": len(fold2),
    "fold1": len(fold1), "recipes": [row["id"] for row in RECIPES],
}, flush=True)

# Full Fold-3 is the sole objective-selection fold.
selection_models = {}
for recipe in RECIPES:
    selection_models[recipe["id"]], _ = _fit(train45, recipe, f"fold45_{recipe['id']}")
fold3_frame, reports3 = N._evaluate(fold3, selection_models, "v82_fold3_selection")
selection_table = pd.DataFrame(_summary(fold3_frame, selection_models, "fold3_selection"))
blend_selection = selection_table[selection_table["endpoint"].eq("blend_w0p30")]
selected_id = str(blend_selection.sort_values(["rmse", "model"], kind="stable").iloc[0]["model"])
selected_recipe = next(row for row in RECIPES if row["id"] == selected_id)
fold3_result = _gains(selection_table, selected_id)
selection_pass = bool(selected_id != "lambdarank" and fold3_result["gain"] >= 0.05 and fold3_result["far_gain"] >= 0.0)

application_rows = []
reports2 = []
fold2_frame = None
confirmation_pass = False
if selection_pass:
    train345 = sorted(well for well, fold in fold_by_well.items() if fold in (3, 4, 5) and well in available)
    models2 = {}
    models2["lambdarank"], _ = _fit(train345, RECIPES[0], "fold345_lambdarank")
    models2[selected_id], _ = _fit(train345, selected_recipe, f"fold345_{selected_id}")
    models2["shuffled_topk"], _ = _fit(train345, selected_recipe, "fold345_shuffled_topk", shuffle=True)
    fold2_frame, reports2 = N._evaluate(fold2, models2, "v82_fold2_confirmation")
    application_rows = _summary(fold2_frame, models2, "fold2_confirmation")
    application_table = pd.DataFrame(application_rows)
    fold2_result = _gains(application_table, selected_id)
    selected2 = _blend_row(application_table, selected_id)
    shuffled2 = _blend_row(application_table, "shuffled_topk")
    fold2_result["shuffle_loss"] = float(shuffled2["rmse"] - selected2["rmse"])
    confirmation_pass = bool(
        fold2_result["gain"] >= 0.03 and fold2_result["far_gain"] >= 0.0
        and fold2_result["near_degradation"] <= 0.03
        and fold2_result["shuffle_loss"] >= 0.02
    )
else:
    fold2_result = None

final_rows = []
reports1 = []
fold1_frame = None
final_pass = False
supplement2 = []
if confirmation_pass:
    missing2 = [well for well in fold2 if well not in available]
    for number, well in enumerate(missing2, 1):
        report = N._build_supplement_panel(well)
        supplement2.append(report)
        if report["status"] != "failed":
            available.add(well)
        if number % 10 == 0 or number == len(missing2):
            print(f"V82 supplement Fold2 {number}/{len(missing2)}", flush=True)
    train2345 = sorted(well for well, fold in fold_by_well.items() if fold in (2, 3, 4, 5) and well in available)
    models1 = {}
    models1["lambdarank"], _ = _fit(train2345, RECIPES[0], "fold2345_lambdarank")
    models1[selected_id], _ = _fit(train2345, selected_recipe, f"fold2345_{selected_id}")
    fold1_frame, reports1 = N._evaluate(fold1, models1, "v82_fold1_final")
    final_rows = _summary(fold1_frame, models1, "fold1_final")
    final_table = pd.DataFrame(final_rows)
    fold1_result = _gains(final_table, selected_id)
    final_pass = bool(
        fold1_result["gain"] >= 0.03 and fold1_result["far_gain"] >= 0.0
        and fold1_result["near_degradation"] <= 0.03
    )
else:
    fold1_result = None

summary = pd.concat([
    selection_table, pd.DataFrame(application_rows), pd.DataFrame(final_rows)
], ignore_index=True)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(reports3 + reports2 + reports1).to_json(
    OUT_DIR / "evaluation_report.jsonl", orient="records", lines=True
)
pd.DataFrame(supplement2).to_csv(OUT_DIR / "fold2_supplement_report.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_topK_selection_Fold2_confirmation_Fold1_final",
    "sample_unit": "one complete physical well",
    "matched_constants": {
        "candidate_bank": "V4 frozen union", "feature_count": len(N.feature_names),
        "iterations": ITERATIONS, "blend_weight": BLEND_WEIGHT,
    },
    "selected_recipe": selected_recipe,
    "fold3_selection": {**fold3_result, "pass": selection_pass},
    "fold2_confirmation": None if fold2_result is None else {**fold2_result, "pass": confirmation_pass},
    "fold1_final": None if fold1_result is None else {**fold1_result, "pass": final_pass},
    "fold1_touched": bool(confirmation_pass),
    "strict_pass": bool(selection_pass and confirmation_pass and final_pass),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild V76 adaptive dose from the frozen top-K coarse endpoint"
        if selection_pass and confirmation_pass and final_pass
        else "FAIL -> retain V76; close top-K selector objective"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV82 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV82 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
