"""Zero-learning multiscale path-emission audit.

The pointwise typewell/heel emissions rank the true state poorly.  This cell
tests whether aggregating those same legal costs along short candidate paths
can move the true state into a deployable top-K menu before any decoder is
trained.

Protocol:
  * scan fixed recipes on 24 Fold-3 wells;
  * confirm the selected recipe on the other Fold-3 wells;
  * apply it once to Fold 2;
  * leave Fold 1 untouched.
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import shutil
import sys
import time
import types

import numpy as np
import pandas as pd


def _bootstrap_costmap():
    candidates = (
        Path("/root/autodl-tmp/inverted_costmap_transformer_fold2_v1_cell.py"),
        Path.cwd() / "inverted_costmap_transformer_fold2_v1_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload inverted_costmap_transformer_fold2_v1_cell.py, "
            "inverted_transformer_fullwell_fold2_v0_cell.py, and "
            "native_resolution_dualref_capacity_v0_cell.py to /root/autodl-tmp."
        )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    module_name = "path_emission_source_v0"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module.__dict__, source_path


V, SOURCE_PATH = _bootstrap_costmap()
OOF_ROOT = V["OOF_ROOT"]
PACK_DIR = V["PACK_DIR"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float32)
N_STATES = len(STATE_OFFSETS)
_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_pack = V["_load_pack"]
_metrics = V["_metrics"]

EXPERIMENT = "multiscale_path_emission_topk_gateway_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260725
DISCOVERY_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 24
TOPK_BUDGETS = (1, 8, 32, 64, 128)
SLOPE_STEPS = (-2, -1, 0, 1, 2)

# Radius is measured on the 257-point complete-well grid.  Typical laterals
# make these roughly 30-500 ft windows, while slope steps span the empirical
# local TVT-rate range.
RECIPES = (
    {"id": "point_type", "radii": (0,), "references": (0,), "clip": 40.0, "combine": "sum"},
    {"id": "point_dual_or", "radii": (0,), "references": (0, 1), "clip": 40.0, "combine": "max"},
    {"id": "type_r2", "radii": (2,), "references": (0,), "clip": 40.0, "combine": "sum"},
    {"id": "type_r4", "radii": (4,), "references": (0,), "clip": 40.0, "combine": "sum"},
    {"id": "type_ms248", "radii": (2, 4, 8), "references": (0,), "clip": 40.0, "combine": "sum"},
    {"id": "type_robust_ms248", "radii": (2, 4, 8), "references": (0,), "clip": 6.0, "combine": "sum"},
    {"id": "dual_or_ms248", "radii": (2, 4, 8), "references": (0, 1), "clip": 12.0, "combine": "max"},
    {"id": "dual_sum_ms248", "radii": (2, 4, 8), "references": (0, 1), "clip": 12.0, "combine": "sum"},
    {"id": "dual_or_ms4816", "radii": (4, 8, 16), "references": (0, 1), "clip": 12.0, "combine": "max"},
    {"id": "dual_sum_ms4816", "radii": (4, 8, 16), "references": (0, 1), "clip": 12.0, "combine": "sum"},
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _moving_nanmean(values, radius):
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0).astype(np.float64)
    count = valid.astype(np.int32)
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(clean, axis=0)])
    ccount = np.vstack([np.zeros((1, values.shape[1]), np.int32), np.cumsum(count, axis=0)])
    rows = np.arange(values.shape[0])
    lo = np.maximum(rows - radius, 0)
    hi = np.minimum(rows + radius + 1, values.shape[0])
    total = csum[hi] - csum[lo]
    number = ccount[hi] - ccount[lo]
    return np.where(number > 0, total / np.maximum(number, 1), np.nan).astype(np.float32)


def _line_mean_cost(cost_state_time, slope_step, radius):
    """Mean cost around every (time, state) along a fixed local slope."""
    cost = cost_state_time.T.astype(np.float32)  # [time, state]
    time_points, states = cost.shape
    q_min = min(0, -slope_step * (time_points - 1))
    q_max = states - 1 + max(0, -slope_step * (time_points - 1))
    width = q_max - q_min + 1
    sheared = np.full((time_points, width), np.nan, np.float32)
    state_index = np.arange(states)
    for t in range(time_points):
        q = state_index - slope_step * t - q_min
        sheared[t, q] = cost[t]
    smooth = _moving_nanmean(sheared, int(radius))
    output = np.empty_like(cost)
    for t in range(time_points):
        q = state_index - slope_step * t - q_min
        output[t] = smooth[t, q]
    return output.T


def _normalise_score(score):
    median = np.median(score, axis=0, keepdims=True)
    mad = np.median(np.abs(score - median), axis=0, keepdims=True)
    return ((score - median) / np.maximum(1.4826 * mad, 0.05)).astype(np.float32)


def _reference_path_score(emission, radii, clip):
    cost = np.clip(-emission.astype(np.float32), 0.0, float(clip))
    scale_scores = []
    for radius in radii:
        slope_costs = [
            _line_mean_cost(cost, step, radius) for step in SLOPE_STEPS
        ]
        best_cost = np.min(np.stack(slope_costs), axis=0)
        scale_scores.append(_normalise_score(-best_cost))
    return np.mean(scale_scores, axis=0).astype(np.float32)


def _recipe_score(pack, recipe, control="full"):
    reference_scores = [
        _reference_path_score(
            pack.emissions[reference], recipe["radii"], recipe["clip"]
        )
        for reference in recipe["references"]
    ]
    if recipe["combine"] == "max":
        score = np.max(np.stack(reference_scores), axis=0)
    elif recipe["combine"] == "sum":
        score = np.mean(np.stack(reference_scores), axis=0)
    else:
        raise ValueError(recipe["combine"])
    if control == "rolled_position":
        score = np.roll(score, max(1, score.shape[1] // 3), axis=1)
    elif control != "full":
        raise ValueError(control)
    return score.astype(np.float32)


def _interpolate_grid(pack, grid_offset):
    support = pack.grid_md > pack.anchor_md
    support_md = np.r_[pack.anchor_md, pack.grid_md[support]]
    support_offset = np.r_[0.0, np.asarray(grid_offset)[support]]
    return pack.anchor_tvt + np.interp(pack.eval_md, support_md, support_offset)


def _score_grid_predictions(packs, predictions):
    truths, estimates, distances, wells = [], [], [], []
    for pack, prediction in zip(packs, predictions):
        truths.append(pack.truth.astype(np.float64))
        estimates.append(_interpolate_grid(pack, prediction))
        distances.append(pack.eval_distance.astype(np.float64))
        wells.append(np.full(pack.eval_rows, pack.well, dtype="U16"))
    metrics, bywell, _ = _metrics(
        np.concatenate(truths),
        np.concatenate(estimates),
        np.concatenate(distances),
        np.concatenate(wells),
    )
    return metrics, bywell


def _topk_prediction(pack, score, budget):
    k = min(int(budget), N_STATES)
    indices = np.argpartition(score, -k, axis=0)[-k:]
    offsets = STATE_OFFSETS[indices]
    choice = np.argmin(np.abs(offsets - pack.target[None]), axis=0)
    prediction = offsets[choice, np.arange(score.shape[1])]
    prediction[: pack.anchor_grid + 1] = 0.0
    return prediction.astype(np.float32)


def _rank_diagnostics(pack, score):
    distance = np.abs(STATE_OFFSETS[:, None] - pack.target[None])
    truth_state = np.argmin(distance, axis=0)
    order = np.argsort(-score, axis=0)
    inverse = np.empty_like(order)
    inverse[order, np.arange(score.shape[1])] = np.arange(N_STATES)[:, None]
    rank = inverse[truth_state, np.arange(score.shape[1])] + 1
    mask = pack.supervised
    return rank[mask].astype(np.int16)


def _evaluate_recipe(packs, recipe, control="full", budgets=TOPK_BUDGETS):
    scores = []
    ranks = []
    for number, pack in enumerate(packs, 1):
        score = _recipe_score(pack, recipe, control=control)
        scores.append(score)
        ranks.append(_rank_diagnostics(pack, score))
        if number % 25 == 0 or number == len(packs):
            print(
                f"emission {recipe['id']} {control} {number}/{len(packs)}",
                flush=True,
            )
    rank = np.concatenate(ranks)
    summary = {
        "recipe": recipe["id"],
        "control": control,
        "median_truth_rank_grid": float(np.median(rank)),
        "p90_truth_rank_grid": float(np.quantile(rank, 0.90)),
        "top8_truth_recall_grid": float(np.mean(rank <= 8)),
        "top32_truth_recall_grid": float(np.mean(rank <= 32)),
    }
    curves = []
    bywell = {}
    for budget in budgets:
        predictions = [
            _topk_prediction(pack, score, budget) for pack, score in zip(packs, scores)
        ]
        metrics, well_metrics = _score_grid_predictions(packs, predictions)
        curves.append({"budget": int(budget), **metrics})
        bywell[int(budget)] = well_metrics
    summary["oracle_curve"] = curves
    return summary, bywell


def _metric_at(result, budget, key="rmse"):
    return next(row[key] for row in result["oracle_curve"] if row["budget"] == budget)


STARTED = time.time()
np.random.seed(SEED)
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
fold3_wells = folds.loc[folds["fold"].eq(DISCOVERY_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
if len(fold3_wells) not in (154, 155) or len(fold2_wells) != 155:
    raise RuntimeError(f"Unexpected Fold3/Fold2 sizes: {len(fold3_wells)}/{len(fold2_wells)}")

needed = sorted(set(fold3_wells + fold2_wells))
_ensure_records(needed)
_ensure_packs(needed)
packs = {well: _load_pack(well) for well in needed}

rng = np.random.default_rng(SEED)
shuffled_fold3 = np.asarray(fold3_wells)[rng.permutation(len(fold3_wells))]
discovery_wells = shuffled_fold3[:DISCOVERY_WELLS].tolist()
confirmation_wells = shuffled_fold3[DISCOVERY_WELLS:].tolist()
discovery_packs = [packs[well] for well in discovery_wells]
confirmation_packs = [packs[well] for well in confirmation_wells]
application_packs = [packs[well] for well in fold2_wells]

discovery_results = []
for recipe in RECIPES:
    result, _ = _evaluate_recipe(discovery_packs, recipe)
    discovery_results.append(result)
    print(
        "DISCOVERY",
        recipe["id"],
        "top32",
        _metric_at(result, 32),
        "rank",
        result["median_truth_rank_grid"],
        flush=True,
    )

discovery_table = pd.DataFrame(
    [
        {
            "recipe": result["recipe"],
            "top8_oracle_rmse": _metric_at(result, 8),
            "top32_oracle_rmse": _metric_at(result, 32),
            "top64_oracle_rmse": _metric_at(result, 64),
            "median_truth_rank_grid": result["median_truth_rank_grid"],
            "top32_truth_recall_grid": result["top32_truth_recall_grid"],
        }
        for result in discovery_results
    ]
).sort_values(["top32_oracle_rmse", "median_truth_rank_grid"])
discovery_table.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
print("DISCOVERY SCAN\n", discovery_table.to_string(index=False), flush=True)

selected_ids = discovery_table.head(2)["recipe"].tolist()
selected_recipes = [recipe for recipe in RECIPES if recipe["id"] in selected_ids]
confirmation_results = []
for recipe in selected_recipes:
    result, _ = _evaluate_recipe(confirmation_packs, recipe)
    confirmation_results.append(result)
    print("FOLD3 CONFIRMATION", recipe["id"], _metric_at(result, 32), flush=True)

selected_result = min(
    confirmation_results,
    key=lambda result: (_metric_at(result, 32), result["median_truth_rank_grid"]),
)
selected_recipe = next(
    recipe for recipe in RECIPES if recipe["id"] == selected_result["recipe"]
)

application_result, application_bywell = _evaluate_recipe(
    application_packs, selected_recipe
)
control_result, _ = _evaluate_recipe(
    application_packs, selected_recipe, control="rolled_position", budgets=(32,)
)
application_bywell[32].to_csv(OUT_DIR / "fold2_top32_oracle_bywell.csv", index=False)

fold2_top32 = _metric_at(application_result, 32)
control_top32 = _metric_at(control_result, 32)
checks = {
    "fold2_top32_oracle_rmse_le_5p5": fold2_top32 <= 5.5,
    "fold2_top32_truth_recall_ge_0p70": application_result["top32_truth_recall_grid"] >= 0.70,
    "fold2_median_truth_rank_le_16": application_result["median_truth_rank_grid"] <= 16,
    "rolled_control_top32_worse_by_ge_1": control_top32 - fold2_top32 >= 1.0,
}
strict_pass = all(checks.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_zero_learning_multiscale_path_emission_gateway",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "discovery_fold": DISCOVERY_FOLD,
        "discovery_wells": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "application_fold": APPLICATION_FOLD,
        "application_wells": len(application_packs),
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
        "fold_source": str(fold_path),
        "costmap_source": str(PACK_DIR),
    },
    "mechanism": {
        "slope_steps_state_bins_per_grid_step": list(SLOPE_STEPS),
        "recipes": list(RECIPES),
        "selected_recipe": selected_recipe,
    },
    "discovery": discovery_results,
    "fold3_confirmation": confirmation_results,
    "fold2": application_result,
    "fold2_rolled_position_control": control_result,
    "diagnostics": {
        "fold2_top32_oracle_rmse": fold2_top32,
        "rolled_control_top32_rmse": control_top32,
        "rolled_control_degradation": control_top32 - fold2_top32,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> use this emission to build a complete-well selector"
        if strict_pass
        else "FAIL -> path aggregation does not repair candidate ranking; change the reference representation"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
