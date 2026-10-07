"""Gate nonlinear rank/distance-correlation emissions on raw GR windows."""

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
    configured = os.environ.get("ROGII_V38_SOURCE", "").strip()
    candidates = [
        Path("/root/autodl-tmp/phase_transport_specificity_v38_cell.py"),
        Path("phase_transport_specificity_v38_cell.py"),
    ]
    if configured:
        candidates.insert(0, Path(configured))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Upload phase_transport_specificity_v38_cell.py to /root/autodl-tmp"
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
    name = "phase_transport_v38_definitions_for_v41"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.__dict__, path


V38, SOURCE_PATH = _bootstrap()
V37 = V38["V"]
OOF_ROOT = V38["OOF_ROOT"]
STATE_OFFSETS = np.asarray(V38["STATE_OFFSETS"], np.float64)
N_STATES = int(V38["N_STATES"])
ALIAS_FT = float(V38["ALIAS_FT"])

_load_folds = V38["_load_folds"]
_ensure_records = V38["_ensure_records"]
_ensure_packs = V38["_ensure_packs"]
_load_record = V38["_load_record"]
_load_pack = V38["_load_pack"]
_generic_typewell = V38["_generic_typewell"]
_shift_typewell = V38["_shift_typewell"]
_components_w2 = V38["_components"]
_normalise_states = V38["_normalise_states"]
_stable_order = V38["_stable_order"]
_truth_rank = V38["_truth_rank"]
_score_at_offset = V38["_score_at_offset"]
_path_metrics = V38["_path_metrics"]
_window_arrays = V37["_window_arrays"]
_fill_detrend = V37["_fill_detrend"]

EXPERIMENT = "nonlinear_dependence_emission_v41"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260727
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
FROZEN_V38 = {"exact_dose": 0.10, "generic_dose": 0.10}
DOSES = (0.025, 0.05, 0.10, 0.20, 0.35)
CONFIGS = (
    {"name": "rank_m256", "family": "rank", "md_half": 256.0, "tvt_half": 16.0},
    {"name": "rank_m512", "family": "rank", "md_half": 512.0, "tvt_half": 16.0},
    {"name": "dcor_m256", "family": "dcor", "md_half": 256.0, "tvt_half": 16.0},
    {"name": "dcor_m512", "family": "dcor", "md_half": 512.0, "tvt_half": 16.0},
    {"name": "hybrid_m512", "family": "hybrid", "md_half": 512.0, "tvt_half": 16.0},
)

_projection_rng = np.random.default_rng(SEED + 410)
_DCOR_PROJECTION = (
    _projection_rng.choice((-1.0, 1.0), size=(17 * 17, 96)).astype(np.float64)
    / np.sqrt(96.0)
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


def _filled(values, valid):
    output, valid = _fill_detrend(values, valid)
    usable = valid.mean(axis=1) >= 0.75
    return np.asarray(output, np.float64), usable


def _rank_embedding(values, valid):
    values, usable = _filled(values, valid)
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    row = np.arange(len(values))[:, None]
    ranks[row, order] = np.arange(values.shape[1], dtype=np.float64)[None]
    ranks -= ranks.mean(axis=1, keepdims=True)
    ranks /= np.maximum(np.linalg.norm(ranks, axis=1, keepdims=True), 1.0e-12)
    return ranks, usable


def _dcor_embedding(values, valid):
    values, usable = _filled(values, valid)
    # Seventeen paired positions retain the window geometry while making the
    # exact O(n^2) distance-matrix embedding affordable for all candidates.
    index = np.linspace(0, values.shape[1] - 1, 17).round().astype(int)
    sample = values[:, index]
    distance = np.abs(sample[:, :, None] - sample[:, None, :])
    centred = (
        distance
        - distance.mean(axis=2, keepdims=True)
        - distance.mean(axis=1, keepdims=True)
        + distance.mean(axis=(1, 2), keepdims=True)
    )
    embedding = centred.reshape(len(values), -1) @ _DCOR_PROJECTION
    embedding /= np.maximum(
        np.linalg.norm(embedding, axis=1, keepdims=True), 1.0e-12
    )
    return embedding, usable


def _pair_score(query_embedding, query_usable, reference_embedding, reference_usable):
    score = reference_embedding @ query_embedding.T
    usable = reference_usable[:, None] & query_usable[None]
    return np.where(usable, score, -40.0)


def _nonlinear_profile(
    record,
    pack,
    config,
    typewell_override=None,
    sample_permutation=None,
):
    query, q_valid, reference, r_valid = _window_arrays(
        record,
        pack,
        config["md_half"],
        config["tvt_half"],
        typewell_override,
    )
    if sample_permutation is not None:
        reference = reference[:, sample_permutation]
        r_valid = r_valid[:, sample_permutation]

    def score(family, ref_values, ref_valid):
        embed = _rank_embedding if family == "rank" else _dcor_embedding
        qe, qu = embed(query, q_valid)
        re, ru = embed(ref_values, ref_valid)
        return _pair_score(qe, qu, re, ru)

    family = config["family"]
    families = ("rank", "dcor") if family == "hybrid" else (family,)
    outputs = []
    for item in families:
        forward = score(item, reference, r_valid)
        reverse = score(item, reference[:, ::-1], r_valid[:, ::-1])
        outputs.append(np.maximum(forward, reverse))
    if len(outputs) == 1:
        return outputs[0]
    first, first_usable = _normalise_states(outputs[0])
    second, second_usable = _normalise_states(outputs[1])
    output = np.full_like(first, -40.0)
    usable = first_usable & second_usable
    output[:, usable] = 0.5 * first[:, usable] + 0.5 * second[:, usable]
    return output


_PROFILE_CACHE = {}


def _components(
    record,
    pack,
    generic_reference,
    config,
    exact_override=None,
    generic_override=None,
    sample_permutation=None,
):
    # The frozen W2 baseline always uses the legal, unmodified references.
    point, exact_w2, generic_w2 = _components_w2(
        record, pack, generic_reference
    )
    exact_key = None
    generic_key = None
    if exact_override is None and sample_permutation is None:
        exact_key = (str(pack.well), config["name"], "exact")
    if generic_override is None and sample_permutation is None:
        generic_key = (str(pack.well), config["name"], "generic")
    if exact_key is not None and exact_key in _PROFILE_CACHE:
        exact = _PROFILE_CACHE[exact_key]
    else:
        exact, _ = _normalise_states(
            _nonlinear_profile(
                record, pack, config, exact_override, sample_permutation
            )
        )
        if exact_key is not None:
            _PROFILE_CACHE[exact_key] = exact.astype(np.float32)
    if generic_key is not None and generic_key in _PROFILE_CACHE:
        generic = _PROFILE_CACHE[generic_key]
    else:
        reference = generic_reference if generic_override is None else generic_override
        generic, _ = _normalise_states(
            _nonlinear_profile(
                record, pack, config, reference, sample_permutation
            )
        )
        if generic_key is not None:
            _PROFILE_CACHE[generic_key] = generic.astype(np.float32)
    return tuple(
        np.asarray(value, np.float64)
        for value in (point, exact_w2, generic_w2, exact, generic)
    )


def _recipe_name(recipe):
    return (
        f"{recipe['config']}_{recipe['mode']}"
        f"_ed{recipe.get('exact_dose', 0):g}"
        f"_gd{recipe.get('generic_dose', 0):g}"
    )


def _combine(components, recipe):
    point, exact_w2, generic_w2, exact, generic = components
    return (
        point
        + FROZEN_V38["exact_dose"] * exact_w2
        + FROZEN_V38["generic_dose"] * generic_w2
        + float(recipe.get("exact_dose", 0.0)) * exact
        + float(recipe.get("generic_dose", 0.0)) * generic
    )


def _evaluate(
    wells,
    records,
    packs,
    generic_reference,
    config,
    recipe,
    control="full",
    donor_by_well=None,
):
    ranks, distances, targets, argmax_values, alias_wins = [], [], [], [], []
    path_items = []
    state_permutation = None
    sample_permutation = None
    if control == "state_shuffle":
        state_permutation = np.random.default_rng(SEED + 411).permutation(N_STATES)
    elif control == "window_shuffle":
        sample_permutation = np.random.default_rng(SEED + 412).permutation(65)
    for number, well in enumerate(wells, start=1):
        record, pack = records[well], packs[well]
        exact_override = None
        generic_override = None
        if control == "donor_exact":
            exact_override = records[donor_by_well[well]].typewell
        elif control == "shift_exact_64":
            exact_override = _shift_typewell(
                record.typewell, 64.0 if number % 2 else -64.0
            )
        elif control == "shift_generic_64":
            generic_override = _shift_typewell(
                generic_reference, 64.0 if number % 2 else -64.0
            )
        components = _components(
            record,
            pack,
            generic_reference,
            config,
            exact_override,
            generic_override,
            sample_permutation,
        )
        score = _combine(components, recipe)
        if state_permutation is not None:
            baseline = _combine(
                components,
                {
                    "config": config["name"], "mode": "baseline",
                    "exact_dose": 0.0, "generic_dose": 0.0,
                },
            )
            score = baseline + (score - baseline)[state_permutation]
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
                f"V41 {control} {_recipe_name(recipe)} {number}/{len(wells)}",
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
        "config": config["name"],
        "family": config["family"],
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


def _menu(config):
    output = [
        {
            "config": config["name"], "mode": "baseline",
            "exact_dose": 0.0, "generic_dose": 0.0,
        }
    ]
    for dose in DOSES:
        output.extend(
            (
                {
                    "config": config["name"], "mode": "exact",
                    "exact_dose": dose, "generic_dose": 0.0,
                },
                {
                    "config": config["name"], "mode": "generic",
                    "exact_dose": 0.0, "generic_dose": dose,
                },
                {
                    "config": config["name"], "mode": "joint",
                    "exact_dose": dose, "generic_dose": dose,
                },
            )
        )
    return output


STARTED = time.time()
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
prior_wells = folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"].tolist()
fold3_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
if len(prior_wells) < 300 or len(fold3_wells) < 154 or len(fold2_wells) < 154:
    raise RuntimeError(
        f"Unexpected prior/Fold3/Fold2 sizes: "
        f"{len(prior_wells)}/{len(fold3_wells)}/{len(fold2_wells)}"
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

config_lookup = {config["name"]: config for config in CONFIGS}
recipe_lookup = {}
discovery_rows = []
for config in CONFIGS:
    for recipe in _menu(config):
        recipe_lookup[_recipe_name(recipe)] = recipe
        row = _evaluate(
            discovery_wells, records, packs, generic_reference, config, recipe
        )
        row["objective"] = _objective(row)
        discovery_rows.append(row)
discovery = pd.DataFrame(discovery_rows).sort_values("objective", ascending=False)
discovery.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
selected_recipe = recipe_lookup[str(discovery.iloc[0]["recipe"])]
selected_config = config_lookup[selected_recipe["config"]]
baseline_recipe = {
    "config": selected_config["name"], "mode": "baseline",
    "exact_dose": 0.0, "generic_dose": 0.0,
}

confirmation = pd.DataFrame(
    [
        {
            "stage": "fold3_confirmation",
            **_evaluate(
                confirmation_wells, records, packs, generic_reference,
                selected_config, recipe,
            ),
        }
        for recipe in (baseline_recipe, selected_recipe)
    ]
)
confirmation.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)

fold2_rows = []
for recipe in (baseline_recipe, selected_recipe):
    fold2_rows.append(
        {
            "stage": "fold2_application",
            **_evaluate(
                fold2_wells, records, packs, generic_reference,
                selected_config, recipe,
            ),
        }
    )
donor_pool = np.asarray(prior_wells)[rng.permutation(len(prior_wells))]
donor_by_well = {
    well: str(donor_pool[index % len(donor_pool)])
    for index, well in enumerate(fold2_wells)
}
for control in (
    "state_shuffle", "window_shuffle", "donor_exact",
    "shift_exact_64", "shift_generic_64",
):
    fold2_rows.append(
        {
            "stage": "fold2_control",
            **_evaluate(
                fold2_wells, records, packs, generic_reference,
                selected_config, selected_recipe, control=control,
                donor_by_well=donor_by_well,
            ),
        }
    )
fold2 = pd.DataFrame(fold2_rows)
fold2.to_csv(OUT_DIR / "fold2_application_and_controls.csv", index=False)


def _pick(frame, recipe, control="full"):
    return frame.loc[
        frame["recipe"].eq(_recipe_name(recipe)) & frame["control"].eq(control)
    ].iloc[0].to_dict()


fold3_base = _pick(confirmation, baseline_recipe)
fold3_selected = _pick(confirmation, selected_recipe)
fold2_base = _pick(fold2, baseline_recipe)
fold2_selected = _pick(fold2, selected_recipe)
controls = {
    control: _pick(fold2, selected_recipe, control)
    for control in (
        "state_shuffle", "window_shuffle", "donor_exact",
        "shift_exact_64", "shift_generic_64",
    )
}
fold3_gain = fold3_selected["top32_recall"] - fold3_base["top32_recall"]
fold2_gain = fold2_selected["top32_recall"] - fold2_base["top32_recall"]
fold2_deep_gain = (
    fold2_selected["deep_top32_recall"] - fold2_base["deep_top32_recall"]
)
control_gain = {
    key: value["top32_recall"] - fold2_base["top32_recall"]
    for key, value in controls.items()
}
exact_active = float(selected_recipe.get("exact_dose", 0.0)) > 0.0
generic_active = float(selected_recipe.get("generic_dose", 0.0)) > 0.0
gates = {
    "fold3_gain_ge_0p01": bool(fold3_gain >= 0.01),
    "fold2_gain_ge_0p01": bool(fold2_gain >= 0.01),
    "fold2_deep_gain_ge_0p01": bool(fold2_deep_gain >= 0.01),
    "state_shuffle_retains_le_30pct": bool(
        control_gain["state_shuffle"] <= max(0.30 * fold2_gain, 0.002)
    ),
    "window_shuffle_retains_le_30pct": bool(
        control_gain["window_shuffle"] <= max(0.30 * fold2_gain, 0.002)
    ),
    "donor_weakens_if_exact_used": bool(
        (not exact_active) or control_gain["donor_exact"] < fold2_gain
    ),
    "exact_shift_weakens_if_used": bool(
        (not exact_active) or control_gain["shift_exact_64"] < fold2_gain
    ),
    "generic_shift_weakens_if_used": bool(
        (not generic_active) or control_gain["shift_generic_64"] < fold2_gain
    ),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_nonlinear_emission_selection_Fold2_confirmation",
    "source": str(SOURCE_PATH),
    "raw_inputs_only": True,
    "uses_cp_neighbor_gold_strong_mainline": False,
    "dcor": {
        "paired_window_samples": 17,
        "centred_distance_matrix_features": 17 * 17,
        "fixed_random_projection_dimensions": 96,
    },
    "frozen_baseline": FROZEN_V38,
    "selected_config": selected_config,
    "selected_recipe": selected_recipe,
    "fold3": {"baseline": fold3_base, "selected": fold3_selected},
    "fold2": {"baseline": fold2_base, "selected": fold2_selected},
    "incremental_gain": {
        "fold3_top32": fold3_gain,
        "fold2_top32": fold2_gain,
        "fold2_deep_top32": fold2_deep_gain,
    },
    "controls": controls,
    "gates": gates,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> retain nonlinear emission and test it inside the trainable interval encoder"
        if strict_pass
        else "FAIL -> close hand-designed nonlinear emission; proceed to self-supervised interval embeddings"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV41 DISCOVERY TOP 25")
print(discovery.head(25).to_string(index=False))
print("\nV41 FOLD3 CONFIRMATION")
print(confirmation.to_string(index=False))
print("\nV41 FOLD2 APPLICATION AND CONTROLS")
print(fold2.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
