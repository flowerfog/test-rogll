"""Gate ordered multiscale GR-boundary evidence beyond positional-W2.

This is deliberately an observability experiment, not a deployable decoder.
It turns each local GR window into ordered rising/falling boundary events and
compares their positions with one-dimensional transport.  A positive result is
the prerequisite for a later Bayesian changepoint/thickness-prior decoder.
"""

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
    name = "phase_transport_v38_definitions_for_v39"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.__dict__, path


V38, SOURCE_PATH = _bootstrap()
V37 = V38["V"]
OOF_ROOT = V38["OOF_ROOT"]
STATE_OFFSETS = np.asarray(V38["STATE_OFFSETS"], np.float64)
REFERENCE_OFFSETS = np.asarray(V38["REFERENCE_OFFSETS"], np.float64)
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

EXPERIMENT = "ordered_boundary_transport_v39"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260727
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
SCALES = (1, 2, 4, 8)
QUANTILES = np.linspace(0.025, 0.975, 40, dtype=np.float64)
FROZEN_V38 = {"exact_dose": 0.10, "generic_dose": 0.10}
PROFILES = (
    {"name": "bnd_s1248_t025", "threshold": 0.25, "strength_weight": 0.03},
    {"name": "bnd_s1248_t050", "threshold": 0.50, "strength_weight": 0.03},
    {"name": "bnd_s1248_t100", "threshold": 1.00, "strength_weight": 0.03},
)
DOSES = (0.025, 0.05, 0.10, 0.20)


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


def _side_difference(values, valid, half_width):
    values, valid = _fill_detrend(values, valid)
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    n_rows, n_points = values.shape
    sums = np.pad(
        np.cumsum(np.where(valid, values, 0.0), axis=1), ((0, 0), (1, 0))
    )
    counts = np.pad(np.cumsum(valid, axis=1), ((0, 0), (1, 0)))
    response = np.zeros_like(values)
    usable = np.zeros_like(valid)
    minimum = max(1, int(np.ceil(0.60 * half_width)))
    for index in range(n_points):
        left0, left1 = max(0, index - half_width), index
        right0, right1 = index + 1, min(n_points, index + 1 + half_width)
        if left1 <= left0 or right1 <= right0:
            continue
        left_count = counts[:, left1] - counts[:, left0]
        right_count = counts[:, right1] - counts[:, right0]
        ok = (left_count >= minimum) & (right_count >= minimum)
        left_mean = (sums[:, left1] - sums[:, left0]) / np.maximum(left_count, 1)
        right_mean = (sums[:, right1] - sums[:, right0]) / np.maximum(right_count, 1)
        response[:, index] = right_mean - left_mean
        usable[:, index] = ok
    absolute = np.where(usable, np.abs(response), np.nan)
    scale = np.ones((n_rows, 1), np.float64)
    for row in range(n_rows):
        finite = absolute[row][np.isfinite(absolute[row])]
        if len(finite):
            scale[row, 0] = max(1.4826 * float(np.median(finite)), 1.0e-3)
    return response / scale, usable


def _inverse_cdf(mass):
    cdf = np.cumsum(mass, axis=1)
    indices = np.sum(cdf[:, :, None] < QUANTILES[None, None, :], axis=1)
    indices = np.clip(indices, 0, mass.shape[1] - 1)
    previous = np.maximum(indices - 1, 0)
    c1 = np.take_along_axis(cdf, indices, axis=1)
    c0 = np.where(indices > 0, np.take_along_axis(cdf, previous, axis=1), 0.0)
    unit = np.linspace(-1.0, 1.0, mass.shape[1], dtype=np.float64)
    x1 = unit[indices]
    x0 = np.where(indices > 0, unit[previous], unit[0])
    fraction = (QUANTILES[None] - c0) / np.maximum(c1 - c0, 1.0e-12)
    return x0 + np.clip(fraction, 0.0, 1.0) * (x1 - x0)


def _boundary_descriptors(values, valid, profile):
    descriptors = []
    for half_width in SCALES:
        response, response_valid = _side_difference(values, valid, half_width)
        for sign in (1.0, -1.0):
            mass = np.maximum(sign * response - float(profile["threshold"]), 0.0)
            mass *= response_valid
            strength = mass.sum(axis=1)
            usable = (
                (response_valid.mean(axis=1) >= 0.70)
                & np.isfinite(strength)
                & (strength >= 0.25)
            )
            density = mass / np.maximum(strength[:, None], 1.0e-12)
            descriptors.append(
                (
                    _inverse_cdf(density),
                    np.log1p(strength),
                    usable,
                )
            )
    return descriptors


def _descriptor_score(query, reference, strength_weight):
    n_reference = len(reference[0][0])
    n_query = len(query[0][0])
    output = np.zeros((n_reference, n_query), np.float64)
    count = np.zeros((n_reference, n_query), np.float64)
    for (qq, qs, qu), (rq, rs, ru) in zip(query, reference):
        for start in range(0, n_query, 256):
            stop = min(start + 256, n_query)
            delta = rq[:, None, :] - qq[None, start:stop, :]
            w2 = np.mean(delta * delta, axis=2)
            strength = np.abs(rs[:, None] - qs[None, start:stop])
            usable = ru[:, None] & qu[None, start:stop]
            value = -w2 - float(strength_weight) * strength
            output[:, start:stop] += np.where(usable, value, 0.0)
            count[:, start:stop] += usable
    minimum_channels = max(4, len(query) // 2)
    return np.where(count >= minimum_channels, output / np.maximum(count, 1.0), -40.0)


def _boundary_profile(record, pack, profile, typewell_override=None, swap_sign=False):
    query, query_valid, reference, reference_valid = _window_arrays(
        record, pack, 512.0, 16.0, typewell_override
    )
    query_descriptor = _boundary_descriptors(query, query_valid, profile)
    forward = _boundary_descriptors(reference, reference_valid, profile)
    reverse = _boundary_descriptors(reference[:, ::-1], reference_valid[:, ::-1], profile)
    if swap_sign:
        def swapped(items):
            output = []
            for index in range(0, len(items), 2):
                output.extend((items[index + 1], items[index]))
            return output
        forward = swapped(forward)
        reverse = swapped(reverse)
    weight = float(profile["strength_weight"])
    return np.maximum(
        _descriptor_score(query_descriptor, forward, weight),
        _descriptor_score(query_descriptor, reverse, weight),
    )


_BOUNDARY_CACHE = {}


def _all_components(
    record,
    pack,
    generic_reference,
    profile,
    exact_override=None,
    generic_override=None,
    swap_sign=False,
):
    # V38 is the frozen baseline. Controls below are attribution tests for the
    # new boundary term only, so they must not perturb its W2 components.
    point, exact_w2, generic_w2 = _components_w2(
        record, pack, generic_reference
    )
    exact_key = None
    generic_key = None
    if exact_override is None and not swap_sign:
        exact_key = (str(pack.well), profile["name"], "exact")
    if generic_override is None and not swap_sign:
        generic_key = (str(pack.well), profile["name"], "generic")
    if exact_key in _BOUNDARY_CACHE:
        exact_boundary = _BOUNDARY_CACHE[exact_key]
    else:
        exact_boundary, _ = _normalise_states(
            _boundary_profile(record, pack, profile, exact_override, swap_sign)
        )
        if exact_key is not None:
            _BOUNDARY_CACHE[exact_key] = exact_boundary.astype(np.float32)
    if generic_key in _BOUNDARY_CACHE:
        generic_boundary = _BOUNDARY_CACHE[generic_key]
    else:
        reference = generic_reference if generic_override is None else generic_override
        generic_boundary, _ = _normalise_states(
            _boundary_profile(record, pack, profile, reference, swap_sign)
        )
        if generic_key is not None:
            _BOUNDARY_CACHE[generic_key] = generic_boundary.astype(np.float32)
    return tuple(
        np.asarray(item, np.float64)
        for item in (point, exact_w2, generic_w2, exact_boundary, generic_boundary)
    )


def _recipe_name(recipe):
    return (
        f"{recipe['profile']}_{recipe['mode']}"
        f"_ed{recipe.get('exact_dose', 0):g}"
        f"_gd{recipe.get('generic_dose', 0):g}"
    )


def _combine(components, recipe):
    point, exact_w2, generic_w2, exact_boundary, generic_boundary = components
    score = (
        point
        + FROZEN_V38["exact_dose"] * exact_w2
        + FROZEN_V38["generic_dose"] * generic_w2
    )
    score += float(recipe.get("exact_dose", 0.0)) * exact_boundary
    score += float(recipe.get("generic_dose", 0.0)) * generic_boundary
    return score


def _evaluate(
    wells,
    records,
    packs,
    generic_reference,
    profile,
    recipe,
    control="full",
    donor_by_well=None,
):
    ranks, distances, targets, argmax_values, alias_wins = [], [], [], [], []
    path_items = []
    permutation = None
    if control == "state_shuffle":
        permutation = np.random.default_rng(SEED + 390).permutation(N_STATES)
    for number, well in enumerate(wells, start=1):
        record, pack = records[well], packs[well]
        exact_override = None
        generic_override = None
        swap_sign = control == "swap_boundary_sign"
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
        components = _all_components(
            record,
            pack,
            generic_reference,
            profile,
            exact_override,
            generic_override,
            swap_sign,
        )
        score = _combine(components, recipe)
        if permutation is not None:
            base_recipe = {
                "profile": profile["name"], "mode": "baseline",
                "exact_dose": 0.0, "generic_dose": 0.0,
            }
            base = _combine(components, base_recipe)
            score = base + (score - base)[permutation]
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
                f"V39 {control} {_recipe_name(recipe)} {number}/{len(wells)}",
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
        "profile": profile["name"],
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


def _menu(profile):
    output = [
        {
            "profile": profile["name"], "mode": "baseline",
            "exact_dose": 0.0, "generic_dose": 0.0,
        }
    ]
    for dose in DOSES:
        output.extend(
            (
                {
                    "profile": profile["name"], "mode": "exact_boundary",
                    "exact_dose": dose, "generic_dose": 0.0,
                },
                {
                    "profile": profile["name"], "mode": "generic_boundary",
                    "exact_dose": 0.0, "generic_dose": dose,
                },
                {
                    "profile": profile["name"], "mode": "joint_boundary",
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
np.save(OUT_DIR / "generic_typewell_fold45.npy", generic_reference)

discovery_rows = []
recipe_lookup = {}
profile_lookup = {profile["name"]: profile for profile in PROFILES}
for profile in PROFILES:
    for recipe in _menu(profile):
        recipe_lookup[_recipe_name(recipe)] = recipe
        row = _evaluate(
            discovery_wells, records, packs, generic_reference, profile, recipe
        )
        row["objective"] = _objective(row)
        discovery_rows.append(row)
discovery = pd.DataFrame(discovery_rows).sort_values("objective", ascending=False)
discovery.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
selected_recipe = recipe_lookup[str(discovery.iloc[0]["recipe"])]
selected_profile = profile_lookup[selected_recipe["profile"]]
baseline_recipe = {
    "profile": selected_profile["name"], "mode": "baseline",
    "exact_dose": 0.0, "generic_dose": 0.0,
}

confirmation = pd.DataFrame(
    [
        {
            "stage": "fold3_confirmation",
            **_evaluate(
                confirmation_wells,
                records,
                packs,
                generic_reference,
                selected_profile,
                recipe,
            ),
        }
        for recipe in (baseline_recipe, selected_recipe)
    ]
)
confirmation.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)

# All choices are frozen before Fold2 is evaluated.
fold2_rows = []
for recipe in (baseline_recipe, selected_recipe):
    fold2_rows.append(
        {
            "stage": "fold2_application",
            **_evaluate(
                fold2_wells,
                records,
                packs,
                generic_reference,
                selected_profile,
                recipe,
            ),
        }
    )
donor_pool = np.asarray(prior_wells)[rng.permutation(len(prior_wells))]
donor_by_well = {
    well: str(donor_pool[index % len(donor_pool)])
    for index, well in enumerate(fold2_wells)
}
for control in (
    "state_shuffle",
    "donor_exact",
    "shift_exact_64",
    "shift_generic_64",
    "swap_boundary_sign",
):
    fold2_rows.append(
        {
            "stage": "fold2_control",
            **_evaluate(
                fold2_wells,
                records,
                packs,
                generic_reference,
                selected_profile,
                selected_recipe,
                control=control,
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
        "state_shuffle", "donor_exact", "shift_exact_64",
        "shift_generic_64", "swap_boundary_sign",
    )
}
fold3_gain = fold3_selected["top32_recall"] - fold3_base["top32_recall"]
fold2_gain = fold2_selected["top32_recall"] - fold2_base["top32_recall"]
fold2_deep_gain = (
    fold2_selected["deep_top32_recall"] - fold2_base["deep_top32_recall"]
)
state_gain = controls["state_shuffle"]["top32_recall"] - fold2_base["top32_recall"]
donor_gain = controls["donor_exact"]["top32_recall"] - fold2_base["top32_recall"]
shift_exact_gain = controls["shift_exact_64"]["top32_recall"] - fold2_base["top32_recall"]
shift_generic_gain = controls["shift_generic_64"]["top32_recall"] - fold2_base["top32_recall"]
swap_sign_gain = controls["swap_boundary_sign"]["top32_recall"] - fold2_base["top32_recall"]
exact_active = float(selected_recipe.get("exact_dose", 0.0)) > 0.0
generic_active = float(selected_recipe.get("generic_dose", 0.0)) > 0.0
gates = {
    "selected_recipe_uses_exact_typewell_boundary": bool(exact_active),
    "fold3_incremental_top32_gain_ge_0p01": bool(fold3_gain >= 0.01),
    "fold2_incremental_top32_gain_ge_0p01": bool(fold2_gain >= 0.01),
    "fold2_incremental_deep_gain_ge_0p01": bool(fold2_deep_gain >= 0.01),
    "state_shuffle_retains_le_30pct": bool(
        state_gain <= max(0.30 * fold2_gain, 0.002)
    ),
    "donor_exact_retains_le_30pct": bool(
        exact_active and donor_gain <= max(0.30 * fold2_gain, 0.002)
    ),
    "shift_exact_weakens": bool(
        exact_active and shift_exact_gain < fold2_gain
    ),
    "swap_rising_falling_weakens": bool(
        exact_active and swap_sign_gain < fold2_gain
    ),
    "shift_generic_weakens_if_used": bool(
        (not generic_active) or shift_generic_gain < fold2_gain
    ),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_generic_boundary_reference_Fold3_selection_Fold2_confirmation",
    "source": str(SOURCE_PATH),
    "raw_inputs_only": True,
    "uses_cp_neighbor_gold_strong_mainline": False,
    "representation": (
        "ordered rising/falling boundary events at 1/2/4/8-sample scales; "
        "positional 1D transport plus boundary-strength mismatch"
    ),
    "protocol": {
        "generic_reference_folds": PRIOR_FOLDS,
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_wells),
        "internal_confirmation_wells": len(confirmation_wells),
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "frozen_v38_baseline": FROZEN_V38,
    "selected_profile": selected_profile,
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
        "PASS -> fit ordered layer-thickness/changepoint transition priors around this data term"
        if strict_pass
        else (
            "MIXED -> boundary locations transfer, but signed polarity or exact-reference "
            "specificity failed a control; test sign-invariant event spacing before changepoint DP"
        )
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV39 DISCOVERY TOP 25")
print(discovery.head(25).to_string(index=False))
print("\nV39 FOLD3 CONFIRMATION")
print(confirmation.to_string(index=False))
print("\nV39 FOLD2 APPLICATION AND CONTROLS")
print(fold2.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
