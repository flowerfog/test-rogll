"""Frozen attribution audit for the V1 point-plus-window gain.

V1 selected Wasserstein(512 MD, 16 TVT) at dose 0.20 on 48 Fold-2
wells and confirmed a small gain on the remaining 107 wells.  Its shuffled
control changed both point and window references, so it could not attribute
that gain to the window evidence.  This zero-training cell keeps the point
profile intact and perturbs only the window branch.
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


def _bootstrap_v1():
    candidates = (
        Path("/root/autodl-tmp/misfit_family_point_fusion_fold2_v1_cell.py"),
        Path.cwd() / "misfit_family_point_fusion_fold2_v1_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload misfit_family_point_fusion_fold2_v1_cell.py to /root/autodl-tmp."
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
    name = "misfit_point_fusion_v1_definitions_for_attribution_v2"
    module = types.ModuleType(name)
    module.__file__ = str(source_path)
    sys.modules[name] = module
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module.__dict__, source_path


V, SOURCE_PATH = _bootstrap_v1()
OOF_ROOT = V["OOF_ROOT"]
STATE_OFFSETS = V["STATE_OFFSETS"]
_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_record = V["_load_record"]
_load_pack = V["_load_pack"]
_profile = V["_profile"]
_normalise_states = V["_normalise_states"]
_truth_rank = V["_truth_rank"]
_stable_order = V["_stable_order"]
_score_at_offset = V["_score_at_offset"]
_interpolate_grid = V["_interpolate_grid"]
_path_metrics = V["_path_metrics"]

EXPERIMENT = "misfit_family_window_attribution_fold2_v2"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260724
VALID_FOLD = 2
SELECTION_WELLS = 48
PROFILE = "wasserstein_m512_t16"
DOSE = 0.20
ALIAS_FT = 24.0
POINT_CONFIG = {
    "name": "point_typewell",
    "family": "point",
    "md_half": 0.0,
    "tvt_half": 0.0,
}
WINDOW_CONFIG = next(config for config in V["CONFIGS"] if config["name"] == PROFILE)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _evaluate(wells, records, packs, control, donor_by_well=None):
    ranks, targets, distances, argmax_values, alias_wins = [], [], [], [], []
    point_ranks, window_ranks, path_items = [], [], []
    usable_rows = total_rows = 0
    rng = np.random.default_rng(SEED + 991)

    for well in wells:
        record = records[well]
        pack = packs[well]
        point = _profile(record, pack, POINT_CONFIG)
        window_override = None
        if control == "window_typewell_shuffle":
            window_override = records[donor_by_well[well]].typewell
        window = _profile(
            record,
            pack,
            WINDOW_CONFIG,
            typewell_override=window_override,
        )
        if control == "window_state_shuffle":
            # One global state permutation preserves every column's score
            # distribution while destroying offset correspondence.
            permutation = rng.permutation(window.shape[0])
            window = window[permutation]

        point_z, _ = _normalise_states(point)
        window_z, usable = _normalise_states(window)
        if control == "point_only":
            blended = point_z
        else:
            blended = point_z.copy()
            blended[:, usable] = (
                (1.0 - DOSE) * point_z[:, usable]
                + DOSE * window_z[:, usable]
            )

        point_order = _stable_order(point)
        window_order = _stable_order(window)
        order = _stable_order(blended)
        target = pack.target.astype(np.float64)
        mask = pack.supervised.astype(bool) & (
            np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - 5.0
        )
        rank = _truth_rank(order, target)
        point_rank = _truth_rank(point_order, target)
        window_rank = _truth_rank(window_order, target)
        argmax = STATE_OFFSETS[order[0]].astype(np.float64)
        argmax[: pack.anchor_grid + 1] = 0.0

        true_score = _score_at_offset(blended, target)
        alias_plus = _score_at_offset(blended, target + ALIAS_FT)
        alias_minus = _score_at_offset(blended, target - ALIAS_FT)
        alias_mask = mask & (
            np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - ALIAS_FT
        )

        ranks.append(rank[mask])
        point_ranks.append(point_rank[mask])
        window_ranks.append(window_rank[mask])
        targets.append(target[mask])
        distances.append((pack.grid_md - pack.anchor_md)[mask])
        argmax_values.append(argmax[mask])
        alias_wins.append((true_score > np.maximum(alias_plus, alias_minus))[alias_mask])
        usable_rows += int((usable & mask).sum())
        total_rows += int(mask.sum())
        path_items.append((pack, argmax))

    rank = np.concatenate(ranks)
    point_rank = np.concatenate(point_ranks)
    window_rank = np.concatenate(window_ranks)
    target = np.concatenate(targets)
    distance = np.concatenate(distances)
    argmax = np.concatenate(argmax_values)
    alias = np.concatenate(alias_wins)
    error = argmax - target
    deep = distance > 2000.0
    path = _path_metrics(path_items)
    return {
        "control": control,
        "wells": len(wells),
        "grid_rows": len(rank),
        "window_usable_fraction": usable_rows / max(total_rows, 1),
        "median_rank": float(np.median(rank)),
        "top8_recall": float(np.mean(rank <= 8)),
        "top32_recall": float(np.mean(rank <= 32)),
        "deep_top32_recall": float(np.mean(rank[deep] <= 32)),
        "alias24_win_rate": float(np.mean(alias)),
        "argmax_grid_rmse": float(np.sqrt(np.mean(error * error))),
        "argmax_within5": float(np.mean(np.abs(error) <= 5.0)),
        "path_rmse": float(path["rmse"]),
        "path_far_gt2000_rmse": float(path["far_gt2000_rmse"]),
        "union_top32_each_recall": float(
            np.mean((point_rank <= 32) | (window_rank <= 32))
        ),
        "window_rescues_point_top32": float(
            np.mean((point_rank > 32) & (window_rank <= 32))
        ),
    }


STARTED = time.time()
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(valid_wells) != 155:
    raise RuntimeError(f"Expected 155 Fold2 wells, found {len(valid_wells)}")
rng = np.random.default_rng(SEED)
shuffled = np.asarray(valid_wells)[rng.permutation(len(valid_wells))]
confirmation_wells = shuffled[SELECTION_WELLS:].tolist()

_ensure_records(valid_wells)
_ensure_packs(valid_wells)
records = {well: _load_record(well) for well in valid_wells}
packs = {well: _load_pack(well) for well in valid_wells}

donors = confirmation_wells.copy()
rng.shuffle(donors)
donor_by_well = {
    donors[index]: donors[(index + 1) % len(donors)]
    for index in range(len(donors))
}

point_only = _evaluate(confirmation_wells, records, packs, "point_only")
full = _evaluate(confirmation_wells, records, packs, "full")
window_typewell_shuffle = _evaluate(
    confirmation_wells,
    records,
    packs,
    "window_typewell_shuffle",
    donor_by_well=donor_by_well,
)
window_state_shuffle = _evaluate(
    confirmation_wells, records, packs, "window_state_shuffle"
)


def _gain(candidate, key, lower_is_better=False):
    if lower_is_better:
        return point_only[key] - candidate[key]
    return candidate[key] - point_only[key]


gains = {}
for name, row in (
    ("full", full),
    ("window_typewell_shuffle", window_typewell_shuffle),
    ("window_state_shuffle", window_state_shuffle),
):
    gains[name] = {
        "top32": _gain(row, "top32_recall"),
        "deep_top32": _gain(row, "deep_top32_recall"),
        "argmax_rmse": _gain(row, "argmax_grid_rmse", lower_is_better=True),
        "path_rmse": _gain(row, "path_rmse", lower_is_better=True),
    }

increment = {
    "top32_vs_typewell_shuffle": (
        gains["full"]["top32"] - gains["window_typewell_shuffle"]["top32"]
    ),
    "top32_vs_state_shuffle": (
        gains["full"]["top32"] - gains["window_state_shuffle"]["top32"]
    ),
    "deep_top32_vs_typewell_shuffle": (
        gains["full"]["deep_top32"]
        - gains["window_typewell_shuffle"]["deep_top32"]
    ),
    "argmax_vs_typewell_shuffle": (
        gains["full"]["argmax_rmse"]
        - gains["window_typewell_shuffle"]["argmax_rmse"]
    ),
}
checks = {
    "full_top32_gain_ge_0p02": gains["full"]["top32"] >= 0.02,
    "full_deep_top32_gain_ge_0p02": gains["full"]["deep_top32"] >= 0.02,
    "aligned_top32_increment_ge_0p01": min(
        increment["top32_vs_typewell_shuffle"],
        increment["top32_vs_state_shuffle"],
    ) >= 0.01,
    "aligned_deep_increment_ge_0p01": (
        increment["deep_top32_vs_typewell_shuffle"] >= 0.01
    ),
    "aligned_argmax_increment_ge_1ft": (
        increment["argmax_vs_typewell_shuffle"] >= 1.0
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "frozen_fold2_window_only_matched_attribution",
    "protocol": {
        "source": str(SOURCE_PATH),
        "fold_source": str(fold_path),
        "confirmation_wells": len(confirmation_wells),
        "profile": PROFILE,
        "dose": DOSE,
        "fold1_touched": False,
        "selection_reused_without_retuning": True,
    },
    "point_only": point_only,
    "full": full,
    "controls": {
        "window_typewell_shuffle": window_typewell_shuffle,
        "window_state_shuffle": window_state_shuffle,
    },
    "gains_vs_point": gains,
    "aligned_evidence_increment": increment,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> preserve this low-dose emission as a sequence-decoder input"
        if strict_pass
        else "FAIL -> gain is menu regularisation, not aligned window evidence; close family"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)
print("WINDOW ATTRIBUTION FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
