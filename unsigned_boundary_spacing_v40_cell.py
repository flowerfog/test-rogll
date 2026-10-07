"""Test sign-invariant boundary location and ordered event-spacing evidence."""

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
    configured = os.environ.get("ROGII_V39_SOURCE", "").strip()
    candidates = [
        Path("/root/autodl-tmp/ordered_boundary_transport_v39_cell.py"),
        Path("ordered_boundary_transport_v39_cell.py"),
    ]
    if configured:
        candidates.insert(0, Path(configured))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Upload ordered_boundary_transport_v39_cell.py to /root/autodl-tmp"
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
    name = "boundary_v39_definitions_for_v40"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.__dict__, path


V, SOURCE_PATH = _bootstrap()
OOF_ROOT = V["OOF_ROOT"]
N_STATES = V["N_STATES"]

EXPERIMENT = "unsigned_boundary_spacing_v40"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260727
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
DOSES = (0.025, 0.05, 0.10, 0.20)
PROFILES = (
    {"name": "unsigned_t025_loc", "threshold": 0.25, "strength_weight": 0.0, "spacing_weight": 0.0},
    {"name": "unsigned_t050_loc", "threshold": 0.50, "strength_weight": 0.0, "spacing_weight": 0.0},
    {"name": "unsigned_t025_space05", "threshold": 0.25, "strength_weight": 0.0, "spacing_weight": 0.5},
    {"name": "unsigned_t050_space05", "threshold": 0.50, "strength_weight": 0.0, "spacing_weight": 0.5},
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


def _unsigned_descriptors(values, valid, profile):
    descriptors = []
    for half_width in V["SCALES"]:
        response, response_valid = V["_side_difference"](values, valid, half_width)
        mass = np.maximum(np.abs(response) - float(profile["threshold"]), 0.0)
        mass *= response_valid
        strength = mass.sum(axis=1)
        usable = (
            (response_valid.mean(axis=1) >= 0.70)
            & np.isfinite(strength)
            & (strength >= 0.25)
        )
        density = mass / np.maximum(strength[:, None], 1.0e-12)
        quantile = V["_inverse_cdf"](density)
        descriptors.append((quantile, np.log1p(strength), usable))
    return descriptors


def _unsigned_score(query, reference, profile):
    n_reference = len(reference[0][0])
    n_query = len(query[0][0])
    output = np.zeros((n_reference, n_query), np.float64)
    count = np.zeros((n_reference, n_query), np.float64)
    spacing_weight = float(profile["spacing_weight"])
    strength_weight = float(profile["strength_weight"])
    for (qq, qs, qu), (rq, rs, ru) in zip(query, reference):
        q_spacing = np.diff(qq, axis=1)
        r_spacing = np.diff(rq, axis=1)
        for start in range(0, n_query, 256):
            stop = min(start + 256, n_query)
            location_delta = rq[:, None, :] - qq[None, start:stop, :]
            cost = np.mean(location_delta * location_delta, axis=2)
            if spacing_weight:
                spacing_delta = (
                    r_spacing[:, None, :] - q_spacing[None, start:stop, :]
                )
                cost += spacing_weight * np.mean(spacing_delta * spacing_delta, axis=2)
            cost += strength_weight * np.abs(rs[:, None] - qs[None, start:stop])
            usable = ru[:, None] & qu[None, start:stop]
            output[:, start:stop] += np.where(usable, -cost, 0.0)
            count[:, start:stop] += usable
    return np.where(count >= 2, output / np.maximum(count, 1.0), -40.0)


def _unsigned_profile(record, pack, profile, typewell_override=None, swap_sign=False):
    del swap_sign
    query, query_valid, reference, reference_valid = V["_window_arrays"](
        record, pack, 512.0, 16.0, typewell_override
    )
    query_descriptor = _unsigned_descriptors(query, query_valid, profile)
    forward = _unsigned_descriptors(reference, reference_valid, profile)
    reverse = _unsigned_descriptors(
        reference[:, ::-1], reference_valid[:, ::-1], profile
    )
    return np.maximum(
        _unsigned_score(query_descriptor, forward, profile),
        _unsigned_score(query_descriptor, reverse, profile),
    )


# V39's component/evaluation machinery remains unchanged; only its boundary
# data term is replaced.  Cache keys include the V40 profile name.
V["_boundary_profile"] = _unsigned_profile
V["_BOUNDARY_CACHE"].clear()


def _combine_unsigned(components, recipe):
    point, exact_w2, generic_w2, exact_boundary, generic_boundary = components
    score = (
        point
        + V["FROZEN_V38"]["exact_dose"] * exact_w2
        + V["FROZEN_V38"]["generic_dose"] * generic_w2
    )
    dose = float(recipe.get("exact_dose", 0.0))
    mode = recipe["mode"]
    if mode == "exact_unsigned":
        score += dose * exact_boundary
    elif mode == "generic_unsigned":
        score += dose * generic_boundary
    elif mode == "residual_unsigned":
        score += dose * (exact_boundary - generic_boundary)
    elif mode != "baseline":
        raise ValueError(f"Unknown V40 mode: {mode}")
    return score


V["_combine"] = _combine_unsigned


def _menu(profile):
    baseline = {
        "profile": profile["name"], "mode": "baseline",
        "exact_dose": 0.0, "generic_dose": 0.0,
    }
    output = [baseline]
    for dose in DOSES:
        output.extend(
            (
                {
                    "profile": profile["name"], "mode": "exact_unsigned",
                    "exact_dose": dose, "generic_dose": 0.0,
                },
                {
                    "profile": profile["name"], "mode": "generic_unsigned",
                    "exact_dose": dose, "generic_dose": 0.0,
                },
                {
                    "profile": profile["name"], "mode": "residual_unsigned",
                    "exact_dose": dose, "generic_dose": 0.0,
                },
            )
        )
    return output


STARTED = time.time()
folds, fold_path = V["_load_folds"]()
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
V["_ensure_records"](needed_records)
V["_ensure_packs"](needed_packs)
records = {well: V["_load_record"](well) for well in needed_records}
packs = {well: V["_load_pack"](well) for well in needed_packs}
generic_reference = V["_generic_typewell"](records, prior_wells)

discovery_rows = []
recipe_lookup = {}
profile_lookup = {profile["name"]: profile for profile in PROFILES}
for profile in PROFILES:
    for recipe in _menu(profile):
        recipe_lookup[V["_recipe_name"](recipe)] = recipe
        row = V["_evaluate"](
            discovery_wells, records, packs, generic_reference, profile, recipe
        )
        row["objective"] = V["_objective"](row)
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
            **V["_evaluate"](
                confirmation_wells, records, packs, generic_reference,
                selected_profile, recipe,
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
            **V["_evaluate"](
                fold2_wells, records, packs, generic_reference,
                selected_profile, recipe,
            ),
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
            **V["_evaluate"](
                fold2_wells, records, packs, generic_reference,
                selected_profile, selected_recipe, control=control,
                donor_by_well=donor_by_well,
            ),
        }
    )
fold2 = pd.DataFrame(fold2_rows)
fold2.to_csv(OUT_DIR / "fold2_application_and_controls.csv", index=False)


def _pick(frame, recipe, control="full"):
    return frame.loc[
        frame["recipe"].eq(V["_recipe_name"](recipe))
        & frame["control"].eq(control)
    ].iloc[0].to_dict()


fold3_base = _pick(confirmation, baseline_recipe)
fold3_selected = _pick(confirmation, selected_recipe)
fold2_base = _pick(fold2, baseline_recipe)
fold2_selected = _pick(fold2, selected_recipe)
controls = {
    control: _pick(fold2, selected_recipe, control)
    for control in ("state_shuffle", "donor_exact", "shift_exact_64", "shift_generic_64")
}
fold3_gain = fold3_selected["top32_recall"] - fold3_base["top32_recall"]
fold2_gain = fold2_selected["top32_recall"] - fold2_base["top32_recall"]
fold2_deep_gain = (
    fold2_selected["deep_top32_recall"] - fold2_base["deep_top32_recall"]
)
control_gain = {
    name: row["top32_recall"] - fold2_base["top32_recall"]
    for name, row in controls.items()
}
selected_mode = str(selected_recipe["mode"])
exact_active = selected_mode in {"exact_unsigned", "residual_unsigned"}
generic_active = selected_mode in {"generic_unsigned", "residual_unsigned"}
gates = {
    "selected_recipe_uses_exact_typewell_boundary": bool(exact_active),
    "fold3_incremental_gain_ge_0p01": bool(fold3_gain >= 0.01),
    "fold2_incremental_gain_ge_0p01": bool(fold2_gain >= 0.01),
    "fold2_deep_incremental_gain_ge_0p01": bool(fold2_deep_gain >= 0.01),
    "state_shuffle_retains_le_30pct": bool(
        control_gain["state_shuffle"] <= max(0.30 * fold2_gain, 0.002)
    ),
    "donor_exact_retains_le_30pct": bool(
        exact_active
        and control_gain["donor_exact"] <= max(0.30 * fold2_gain, 0.002)
    ),
    "shift_exact_weakens_if_used": bool(
        (not exact_active) or control_gain["shift_exact_64"] < fold2_gain
    ),
    "shift_generic_weakens_if_used": bool(
        (not generic_active) or control_gain["shift_generic_64"] < fold2_gain
    ),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_unsigned_boundary_spacing_selection_Fold2_confirmation",
    "source": str(SOURCE_PATH),
    "raw_inputs_only": True,
    "uses_cp_neighbor_gold_strong_mainline": False,
    "frozen_baseline": V["FROZEN_V38"],
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
        "PASS -> retain unsigned boundary data term and fit an explicit ordered thickness transition"
        if strict_pass
        else "FAIL -> close changepoint/thickness branch; boundary gains were polarity instability"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV40 DISCOVERY TOP 25")
print(discovery.head(25).to_string(index=False))
print("\nV40 FOLD3 CONFIRMATION")
print(confirmation.to_string(index=False))
print("\nV40 FOLD2 APPLICATION AND CONTROLS")
print(fold2.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
