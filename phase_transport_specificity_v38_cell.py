"""Decompose positional-W2 evidence into generic-depth and exact-reference parts."""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


def _source_path() -> Path:
    configured = os.environ.get("ROGII_V37_SOURCE", "").strip()
    candidates = [
        Path("/root/autodl-tmp/phase_transport_observability_v37_cell.py"),
        Path("phase_transport_observability_v37_cell.py"),
    ]
    if configured:
        candidates.insert(0, Path(configured))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Upload phase_transport_observability_v37_cell.py to /root/autodl-tmp"
    )


def _bootstrap() -> tuple[dict, Path]:
    path = _source_path()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    name = "phase_transport_v37_definitions_for_v38"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.__dict__, path


V, SOURCE_PATH = _bootstrap()
OOF_ROOT = V["OOF_ROOT"]
STATE_OFFSETS = V["STATE_OFFSETS"]
REFERENCE_OFFSETS = V["REFERENCE_OFFSETS"]
N_STATES = V["N_STATES"]
ALIAS_FT = V["ALIAS_FT"]

_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_record = V["_load_record"]
_load_pack = V["_load_pack"]
_exact_profile = V["_exact_profile"]
_normalise_states = V["_normalise_states"]
_stable_order = V["_stable_order"]
_truth_rank = V["_truth_rank"]
_score_at_offset = V["_score_at_offset"]
_path_metrics = V["_path_metrics"]

EXPERIMENT = "phase_transport_specificity_v38"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260727
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
DOSES = (0.025, 0.05, 0.10, 0.20, 0.35)
LAMBDAS = (0.50, 0.75, 1.00)
CONFIG = {
    "name": "posw2_b05_m512_t16",
    "family": "posw2",
    "beta": 0.5,
    "md_half": 512.0,
    "tvt_half": 16.0,
    "bidirectional": True,
}


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


def _generic_typewell(records, wells):
    values = np.stack([records[well].typewell[0] for well in wells]).astype(np.float64)
    valid = np.stack([records[well].typewell[1] > 0.5 for well in wells])
    masked = np.where(valid & np.isfinite(values), values, np.nan)
    reference = np.nanmedian(masked, axis=0)
    coverage = np.mean(np.isfinite(masked), axis=0)
    finite = np.isfinite(reference) & (coverage >= 0.25)
    if int(finite.sum()) < 300:
        raise RuntimeError(f"Generic typewell has only {int(finite.sum())} valid samples")
    reference = np.interp(
        REFERENCE_OFFSETS,
        REFERENCE_OFFSETS[finite],
        reference[finite],
        left=np.nan,
        right=np.nan,
    )
    valid_output = np.isfinite(reference).astype(np.float32)
    return np.stack((np.nan_to_num(reference, nan=0.0), valid_output)).astype(np.float32)


def _shift_typewell(typewell, shift_ft):
    values = np.interp(
        REFERENCE_OFFSETS + float(shift_ft),
        REFERENCE_OFFSETS,
        typewell[0],
        left=np.nan,
        right=np.nan,
    )
    valid = np.interp(
        REFERENCE_OFFSETS + float(shift_ft),
        REFERENCE_OFFSETS,
        typewell[1],
        left=0.0,
        right=0.0,
    ) > 0.5
    valid &= np.isfinite(values)
    return np.stack((np.nan_to_num(values, nan=0.0), valid.astype(np.float32)))


_COMPONENT_CACHE = {}


def _components(record, pack, generic_reference, exact_override=None, generic_override=None):
    exact_key = (str(pack.well), "exact") if exact_override is None else None
    generic_key = (str(pack.well), "generic") if generic_override is None else None
    if exact_key is not None and exact_key in _COMPONENT_CACHE:
        exact_z = _COMPONENT_CACHE[exact_key]
    else:
        exact = _exact_profile(record, pack, CONFIG, exact_override)
        exact_z, _ = _normalise_states(exact)
        if exact_key is not None:
            _COMPONENT_CACHE[exact_key] = exact_z.astype(np.float32)
    if generic_key is not None and generic_key in _COMPONENT_CACHE:
        generic_z = _COMPONENT_CACHE[generic_key]
    else:
        reference = generic_reference if generic_override is None else generic_override
        generic = _exact_profile(record, pack, CONFIG, reference)
        generic_z, _ = _normalise_states(generic)
        if generic_key is not None:
            _COMPONENT_CACHE[generic_key] = generic_z.astype(np.float32)
    point_z, _ = _normalise_states(pack.emissions[0])
    return point_z, np.asarray(exact_z), np.asarray(generic_z)


def _recipe_name(recipe):
    return (
        f"{recipe['mode']}_d{recipe.get('dose', 0):g}"
        f"_l{recipe.get('lambda_generic', 0):g}"
        f"_gd{recipe.get('generic_dose', 0):g}"
        f"_rd{recipe.get('residual_dose', 0):g}"
    )


def _combine(point, exact, generic, recipe):
    mode = recipe["mode"]
    if mode == "baseline":
        return point.copy()
    if mode == "exact":
        return point + float(recipe["dose"]) * exact
    if mode == "generic":
        return point + float(recipe["dose"]) * generic
    residual = exact - float(recipe["lambda_generic"]) * generic
    if mode == "residual":
        return point + float(recipe["dose"]) * residual
    if mode == "joint":
        return (
            point
            + float(recipe["generic_dose"]) * generic
            + float(recipe["residual_dose"]) * residual
        )
    raise ValueError(mode)


def _evaluate(
    wells,
    records,
    packs,
    generic_reference,
    recipe,
    control="full",
    donor_by_well=None,
):
    ranks, distances, targets, argmax_values, alias_wins = [], [], [], [], []
    path_items = []
    permutation = None
    if control == "state_shuffle":
        permutation = np.random.default_rng(SEED + 381).permutation(N_STATES)
    for number, well in enumerate(wells, start=1):
        record, pack = records[well], packs[well]
        exact_override = None
        generic_override = None
        if control == "donor_exact":
            exact_override = records[donor_by_well[well]].typewell
        elif control == "shift_exact_64":
            direction = 1.0 if number % 2 else -1.0
            exact_override = _shift_typewell(record.typewell, direction * 64.0)
        elif control == "shift_generic_64":
            direction = 1.0 if number % 2 else -1.0
            generic_override = _shift_typewell(generic_reference, direction * 64.0)
        point, exact, generic = _components(
            record, pack, generic_reference, exact_override, generic_override
        )
        score = _combine(point, exact, generic, recipe)
        if permutation is not None:
            adjustment = score - point
            score = point + adjustment[permutation]
        order = _stable_order(score)
        target = pack.target.astype(np.float64)
        mask = pack.supervised.astype(bool) & (
            np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - 5.0
        )
        rank = _truth_rank(order, target)
        argmax = STATE_OFFSETS[order[0]]
        argmax[: pack.anchor_grid + 1] = 0.0
        true_score = _score_at_offset(score, target)
        plus = _score_at_offset(score, target + ALIAS_FT)
        minus = _score_at_offset(score, target - ALIAS_FT)
        alias_mask = mask & (
            np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - ALIAS_FT
        )
        ranks.append(rank[mask])
        distances.append((pack.grid_md - pack.anchor_md)[mask])
        targets.append(target[mask])
        argmax_values.append(argmax[mask])
        alias_wins.append((true_score > np.maximum(plus, minus))[alias_mask])
        path_items.append((pack, argmax))
        if number % 25 == 0 or number == len(wells):
            print(
                f"V38 {control} {_recipe_name(recipe)} {number}/{len(wells)}",
                flush=True,
            )
    rank = np.concatenate(ranks)
    distance = np.concatenate(distances)
    target = np.concatenate(targets)
    argmax = np.concatenate(argmax_values)
    alias = np.concatenate(alias_wins)
    deep = distance > 2000.0
    error = argmax - target
    path = _path_metrics(path_items)
    return {
        "recipe": _recipe_name(recipe),
        "mode": recipe["mode"],
        "control": control,
        "wells": int(len(wells)),
        "grid_rows": int(len(rank)),
        "median_rank": float(np.median(rank)),
        "top8_recall": float(np.mean(rank <= 8)),
        "top32_recall": float(np.mean(rank <= 32)),
        "deep_top32_recall": float(np.mean(rank[deep] <= 32)),
        "alias24_win_rate": float(np.mean(alias)),
        "argmax_grid_rmse": float(np.sqrt(np.mean(error * error))),
        "path_rmse": float(path["rmse"]),
        "path_far_gt2000_rmse": float(path["far_gt2000_rmse"]),
    }


def _objective(row):
    return (
        1.25 * row["top32_recall"]
        + row["deep_top32_recall"]
        + 0.50 * row["alias24_win_rate"]
        - 0.001 * row["argmax_grid_rmse"]
    )


def _menus():
    baseline = {"mode": "baseline"}
    exact = [{"mode": "exact", "dose": dose} for dose in DOSES]
    generic = [{"mode": "generic", "dose": dose} for dose in DOSES]
    residual = [
        {"mode": "residual", "dose": dose, "lambda_generic": lam}
        for lam in LAMBDAS for dose in DOSES
    ]
    joint = [
        {
            "mode": "joint",
            "generic_dose": generic_dose,
            "residual_dose": residual_dose,
            "lambda_generic": 1.0,
        }
        for generic_dose in (0.10, 0.20)
        for residual_dose in (0.05, 0.10, 0.20)
    ]
    return baseline, exact, generic, residual, joint


STARTED = time.time()
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
prior_wells = folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"].tolist()
fold3_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
if len(prior_wells) < 300 or len(fold3_wells) < 154 or len(fold2_wells) < 154:
    raise RuntimeError(
        f"Unexpected prior/Fold3/Fold2 sizes: {len(prior_wells)}/{len(fold3_wells)}/{len(fold2_wells)}"
    )

rng = np.random.default_rng(SEED)
fold3_order = np.asarray(fold3_wells)[rng.permutation(len(fold3_wells))]
discovery_wells = fold3_order[:DISCOVERY_WELLS].tolist()
confirmation_wells = fold3_order[DISCOVERY_WELLS:].tolist()
needed_records = sorted(set(prior_wells + fold3_wells + fold2_wells))
needed_packs = sorted(set(fold3_wells + fold2_wells))
_ensure_records(needed_records)
_ensure_packs(needed_packs)
records = {well: _load_record(well) for well in needed_records}
packs = {well: _load_pack(well) for well in needed_packs}
generic_reference = _generic_typewell(records, prior_wells)
np.save(OUT_DIR / "generic_typewell_fold45.npy", generic_reference)

baseline_recipe, exact_menu, generic_menu, residual_menu, joint_menu = _menus()
menu = [baseline_recipe] + exact_menu + generic_menu + residual_menu + joint_menu
discovery_rows = []
for recipe in menu:
    row = _evaluate(
        discovery_wells, records, packs, generic_reference, recipe
    )
    row["objective"] = _objective(row)
    discovery_rows.append(row)
discovery = pd.DataFrame(discovery_rows).sort_values("objective", ascending=False)
discovery.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)

recipe_by_name = {_recipe_name(recipe): recipe for recipe in menu}
selected_recipe = recipe_by_name[discovery.iloc[0]["recipe"]]
best_generic_name = discovery.loc[discovery["mode"].eq("generic")].iloc[0]["recipe"]
best_generic_recipe = recipe_by_name[best_generic_name]
best_residual_name = discovery.loc[discovery["mode"].eq("residual")].iloc[0]["recipe"]
best_residual_recipe = recipe_by_name[best_residual_name]
exact_fixed_recipe = {"mode": "exact", "dose": 0.20}

confirmation_recipes = [
    baseline_recipe,
    exact_fixed_recipe,
    best_generic_recipe,
    best_residual_recipe,
    selected_recipe,
]
confirmation_rows = []
for recipe in confirmation_recipes:
    confirmation_rows.append(
        {
            "stage": "fold3_confirmation",
            **_evaluate(
                confirmation_wells, records, packs, generic_reference, recipe
            ),
        }
    )
confirmation = pd.DataFrame(confirmation_rows)
confirmation.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)

# Freeze all recipes before the first Fold-2 score is computed.
fold2_rows = []
for recipe in confirmation_recipes:
    fold2_rows.append(
        {
            "stage": "fold2_application",
            **_evaluate(fold2_wells, records, packs, generic_reference, recipe),
        }
    )

donor_pool = np.asarray(prior_wells)[rng.permutation(len(prior_wells))]
donor_by_well = {
    well: str(donor_pool[index % len(donor_pool)])
    for index, well in enumerate(fold2_wells)
}
for control in ("state_shuffle", "donor_exact", "shift_exact_64", "shift_generic_64"):
    fold2_rows.append(
        {
            "stage": "fold2_control",
            **_evaluate(
                fold2_wells,
                records,
                packs,
                generic_reference,
                selected_recipe,
                control=control,
                donor_by_well=donor_by_well,
            ),
        }
    )
fold2 = pd.DataFrame(fold2_rows)
fold2.to_csv(OUT_DIR / "fold2_application_and_controls.csv", index=False)


def _row(frame, recipe):
    return frame.loc[frame["recipe"].eq(_recipe_name(recipe))].iloc[0].to_dict()


fold3_base = _row(confirmation, baseline_recipe)
fold3_selected = _row(confirmation, selected_recipe)
fold2_base = _row(fold2.loc[fold2["control"].eq("full")], baseline_recipe)
fold2_selected = _row(fold2.loc[fold2["control"].eq("full")], selected_recipe)
fold2_generic = _row(fold2.loc[fold2["control"].eq("full")], best_generic_recipe)
fold2_residual = _row(fold2.loc[fold2["control"].eq("full")], best_residual_recipe)

fold3_gain = fold3_selected["top32_recall"] - fold3_base["top32_recall"]
fold2_gain = fold2_selected["top32_recall"] - fold2_base["top32_recall"]
generic_gain = fold2_generic["top32_recall"] - fold2_base["top32_recall"]
residual_gain = fold2_residual["top32_recall"] - fold2_base["top32_recall"]
specific_fraction = residual_gain / max(fold2_gain, 1.0e-12)

gates = {
    "fold3_selected_gain_ge_0p02": bool(fold3_gain >= 0.02),
    "fold2_selected_gain_ge_0p02": bool(fold2_gain >= 0.02),
    "fold2_deep_gain_ge_0p02": bool(
        fold2_selected["deep_top32_recall"] - fold2_base["deep_top32_recall"] >= 0.02
    ),
    "exact_residual_gain_ge_0p01": bool(residual_gain >= 0.01),
    "selected_alias_nonworse": bool(
        fold2_selected["alias24_win_rate"] >= fold2_base["alias24_win_rate"] - 0.005
    ),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_generic_reference_Fold3_decomposition_selection_Fold2_confirmation",
    "source": str(SOURCE_PATH),
    "raw_inputs_only": True,
    "uses_cp_neighbor_gold_strong_mainline": False,
    "protocol": {
        "generic_reference_folds": PRIOR_FOLDS,
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_wells),
        "internal_confirmation_wells": len(confirmation_wells),
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "selected_recipe": selected_recipe,
    "best_generic_recipe": best_generic_recipe,
    "best_residual_recipe": best_residual_recipe,
    "fold3": {"baseline": fold3_base, "selected": fold3_selected},
    "fold2": {
        "baseline": fold2_base,
        "selected": fold2_selected,
        "generic": fold2_generic,
        "exact_minus_generic_residual": fold2_residual,
    },
    "decomposition": {
        "selected_top32_gain": fold2_gain,
        "generic_top32_gain": generic_gain,
        "exact_residual_top32_gain": residual_gain,
        "specific_fraction_of_selected_gain": specific_fraction,
    },
    "controls": fold2.loc[fold2["stage"].eq("fold2_control")].to_dict(orient="records"),
    "gates": gates,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> retain generic and exact-residual W2 as separate sequence-model channels"
        if strict_pass
        else "MIXED -> retain exact and generic W2 as separate candidate features; exact-reference specificity is only boundary-strength and must not be described as a standalone matcher"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV38 DISCOVERY TOP 25")
print(discovery.head(25).to_string(index=False))
print("\nV38 FOLD3 CONFIRMATION")
print(confirmation.to_string(index=False))
print("\nV38 FOLD2 APPLICATION AND CONTROLS")
print(fold2.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
