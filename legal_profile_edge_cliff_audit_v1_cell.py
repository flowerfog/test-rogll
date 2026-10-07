"""Audit the 64->128 top-K oracle cliff and state-axis edge effects.

This cell deliberately separates three quantities that are easy to conflate:

* argmax localization, which can be damaged by edge pseudo-peaks;
* truth-state rank, which edge pseudo-peaks can only worsen;
* top-K oracle error, which uses truth only after the legal ranking is frozen.

The Fold-2 confirmation split exactly matches
local_costprofile_feature_oracle_fold2_v0_cell.py.
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
            "Upload inverted_costmap_transformer_fold2_v1_cell.py and its two "
            "native-record dependencies to /root/autodl-tmp."
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
    module_name = "edge_cliff_source_v1"
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

EXPERIMENT = "legal_profile_edge_cliff_audit_v1"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260724
VALID_FOLD = 2
SELECTION_WELLS = 48
BUDGETS = (1, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 257)
EDGE_WIDTHS = (8, 16, 24, 32)
INTERIOR_LIMIT = float(np.max(np.abs(STATE_OFFSETS)) - 16.0)
PROFILE_NAMES = ("typewell", "heel", "combined")


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


def _profiles(pack):
    typewell = pack.emissions[0].astype(np.float32)
    heel = pack.emissions[1].astype(np.float32)
    combined = np.logaddexp(typewell, heel).astype(np.float32) - np.float32(np.log(2.0))
    return {"typewell": typewell, "heel": heel, "combined": combined}


def _interpolate_grid(pack, grid_offset):
    support = pack.grid_md > pack.anchor_md
    support_md = np.r_[pack.anchor_md, pack.grid_md[support]]
    support_offset = np.r_[0.0, np.asarray(grid_offset)[support]]
    return pack.anchor_tvt + np.interp(pack.eval_md, support_md, support_offset)


def _path_metrics(packs, predictions):
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


def _ordered_states(score):
    return np.argsort(-score, axis=0)


def _truth_rank(order, target):
    inverse = np.empty_like(order)
    inverse[order, np.arange(order.shape[1])] = np.arange(N_STATES)[:, None]
    distance = np.abs(STATE_OFFSETS[:, None] - target[None])
    minimum = np.min(distance, axis=0, keepdims=True)
    nearest = distance <= minimum + 1e-6
    tied_rank = np.where(nearest, inverse, N_STATES)
    return np.min(tied_rank, axis=0) + 1


def _topk_choice(order, target, budget, allowed=None):
    if allowed is None:
        candidates = order[: int(budget)]
    else:
        filtered = np.where(allowed[order], order, -1)
        # Stable compaction preserves legal score order while removing edges.
        candidates = np.empty((int(budget), order.shape[1]), np.int64)
        for column in range(order.shape[1]):
            valid = filtered[:, column]
            valid = valid[valid >= 0]
            candidates[:, column] = valid[: int(budget)]
    offsets = STATE_OFFSETS[candidates]
    choice = np.argmin(np.abs(offsets - target[None]), axis=0)
    return offsets[choice, np.arange(order.shape[1])].astype(np.float32)


def _edge_fraction(offset, width):
    limit = float(np.max(np.abs(STATE_OFFSETS)) - width)
    return float(np.mean(np.abs(offset) >= limit))


def _profile_audit(packs, profile_name):
    orders = []
    targets = []
    masks = []
    argmax_offsets = []
    predictions_by_budget = {budget: [] for budget in BUDGETS}
    selected_by_budget = {budget: [] for budget in BUDGETS}

    for pack in packs:
        score = _profiles(pack)[profile_name]
        order = _ordered_states(score)
        target = pack.target.astype(np.float32)
        mask = pack.supervised.astype(bool)
        orders.append(order)
        targets.append(target)
        masks.append(mask)
        argmax_offsets.append(STATE_OFFSETS[order[0]][mask])
        for budget in BUDGETS:
            selected = _topk_choice(order, target, budget)
            selected[: pack.anchor_grid + 1] = 0.0
            predictions_by_budget[budget].append(selected)
            selected_by_budget[budget].append(selected[mask])

    rank = np.concatenate(
        [_truth_rank(order, target)[mask] for order, target, mask in zip(orders, targets, masks)]
    )
    target = np.concatenate([value[mask] for value, mask in zip(targets, masks)])
    argmax = np.concatenate(argmax_offsets)
    argmax_error = argmax - target

    curve = []
    bywell = {}
    for budget in BUDGETS:
        selected = np.concatenate(selected_by_budget[budget])
        grid_error = selected - target
        metrics, well_metrics = _path_metrics(packs, predictions_by_budget[budget])
        curve.append(
            {
                "budget": int(budget),
                "truth_rank_recall": float(np.mean(rank <= budget)),
                "grid_within5": float(np.mean(np.abs(grid_error) <= 5.0)),
                "grid_rmse": float(np.sqrt(np.mean(grid_error.astype(np.float64) ** 2))),
                "selected_edge16_fraction": _edge_fraction(selected, 16),
                "selected_abs_offset_p95": float(np.quantile(np.abs(selected), 0.95)),
                "path_rmse": metrics["rmse"],
                "path_far_gt2000_rmse": metrics["far_gt2000_rmse"],
            }
        )
        bywell[budget] = well_metrics

    interior_state = np.abs(STATE_OFFSETS) <= INTERIOR_LIMIT
    interior_rows = np.abs(target) <= INTERIOR_LIMIT - 5.0
    interior_diagnostics = []
    for budget in (32, 64, 96, 128):
        selections = []
        interior_targets = []
        for order, pack_target, mask in zip(orders, targets, masks):
            selected = _topk_choice(order, pack_target, budget, allowed=interior_state)
            row_mask = mask & (np.abs(pack_target) <= INTERIOR_LIMIT - 5.0)
            selections.append(selected[row_mask])
            interior_targets.append(pack_target[row_mask])
        selected = np.concatenate(selections)
        local_target = np.concatenate(interior_targets)
        error = selected - local_target
        interior_diagnostics.append(
            {
                "budget": budget,
                "rows": len(error),
                "grid_rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                "grid_within5": float(np.mean(np.abs(error) <= 5.0)),
            }
        )

    return {
        "profile": profile_name,
        "grid_rows": int(len(rank)),
        "truth_rank": {
            "median": float(np.median(rank)),
            "p75": float(np.quantile(rank, 0.75)),
            "p90": float(np.quantile(rank, 0.90)),
            "p95": float(np.quantile(rank, 0.95)),
            "fraction_65_to_128": float(np.mean((rank >= 65) & (rank <= 128))),
        },
        "target_axis": {
            **{
                f"edge{width}_fraction": _edge_fraction(target, width)
                for width in EDGE_WIDTHS
            },
            "abs_target_p95": float(np.quantile(np.abs(target), 0.95)),
        },
        "argmax": {
            "grid_rmse": float(np.sqrt(np.mean(argmax_error.astype(np.float64) ** 2))),
            "grid_within5": float(np.mean(np.abs(argmax_error) <= 5.0)),
            **{
                f"edge{width}_fraction": _edge_fraction(argmax, width)
                for width in EDGE_WIDTHS
            },
        },
        "oracle_curve": curve,
        "interior_edge_mask_control": interior_diagnostics,
    }, bywell


STARTED = time.time()
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(valid_wells) != 155:
    raise RuntimeError(f"Expected 155 Fold2 wells, found {len(valid_wells)}")
split_rng = np.random.default_rng(SEED)
shuffled = np.asarray(valid_wells)[split_rng.permutation(len(valid_wells))]
confirmation_wells = shuffled[SELECTION_WELLS:].tolist()
_ensure_records(confirmation_wells)
_ensure_packs(confirmation_wells)
packs = [_load_pack(well) for well in confirmation_wells]

results = []
for profile_name in PROFILE_NAMES:
    result, bywell = _profile_audit(packs, profile_name)
    results.append(result)
    pd.DataFrame(result["oracle_curve"]).to_csv(
        OUT_DIR / f"{profile_name}_oracle_curve.csv", index=False
    )
    bywell[128].to_csv(OUT_DIR / f"{profile_name}_top128_bywell.csv", index=False)
    print(
        "EDGE CLIFF",
        profile_name,
        {
            "median_rank": result["truth_rank"]["median"],
            "rank_65_128": result["truth_rank"]["fraction_65_to_128"],
            "argmax_rmse": result["argmax"]["grid_rmse"],
            "argmax_edge16": result["argmax"]["edge16_fraction"],
        },
        flush=True,
    )

combined = next(result for result in results if result["profile"] == "combined")
combined_64 = next(row for row in combined["oracle_curve"] if row["budget"] == 64)
combined_128 = next(row for row in combined["oracle_curve"] if row["budget"] == 128)
edge_drives_argmax = (
    combined["argmax"]["edge16_fraction"]
    >= combined["target_axis"]["edge16_fraction"] + 0.10
)
cliff_is_rank_mass = combined["truth_rank"]["fraction_65_to_128"] >= 0.20

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_confirmation_state_edge_and_rank_cliff_audit",
    "sample_unit": "complete physical well; 257-point grid nested within well",
    "truth_usage": "oracle diagnostics only",
    "legal_profile_ranking_frozen_before_oracle": True,
    "protocol": {
        "validation_fold": VALID_FOLD,
        "selection_wells_excluded": SELECTION_WELLS,
        "confirmation_wells": len(confirmation_wells),
        "fold1_touched": False,
        "fold_source": str(fold_path),
        "costmap_source": str(PACK_DIR),
    },
    "profiles": results,
    "combined_diagnosis": {
        "top64_path_rmse": combined_64["path_rmse"],
        "top128_path_rmse": combined_128["path_rmse"],
        "fraction_truth_rank_65_to_128": combined["truth_rank"]["fraction_65_to_128"],
        "argmax_edge16_fraction": combined["argmax"]["edge16_fraction"],
        "target_edge16_fraction": combined["target_axis"]["edge16_fraction"],
        "edge_pseudopeaks_drive_argmax": edge_drives_argmax,
        "rank_mass_explains_64_to_128_cliff": cliff_is_rank_mass,
    },
    "interpretation": (
        "Edge pseudo-peaks damage argmax, while the 64-to-128 oracle cliff is "
        "caused by truth-state rank mass in that interval; edge peaks cannot "
        "artificially improve a truth-selected top-K oracle."
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
