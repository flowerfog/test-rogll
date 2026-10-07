"""Final point-plus-window emission complementarity gate.

V0 showed that every windowed misfit is worse as a standalone ranker.  This
cell tests the only remaining honest hypothesis: a weak window profile may
still retrieve true states missed by the causal pointwise typewell emission.

The split is unchanged: 48 Fold-2 wells select a profile and dose, then 107
Fold-2 wells confirm it once.  Fold 1 remains untouched.
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


def _bootstrap_v0():
    candidates = (
        Path("/root/autodl-tmp/misfit_family_observability_fold2_v0_cell.py"),
        Path.cwd() / "misfit_family_observability_fold2_v0_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload misfit_family_observability_fold2_v0_cell.py to /root/autodl-tmp."
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
    module_name = "misfit_observability_v0_definitions_for_fusion_v1"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module.__dict__, source_path


V, SOURCE_PATH = _bootstrap_v0()
OOF_ROOT = V["OOF_ROOT"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float64)
N_STATES = len(STATE_OFFSETS)
CONFIGS = tuple(config for config in V["CONFIGS"] if config["family"] != "point")
_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_record = V["_load_record"]
_load_pack = V["_load_pack"]
_profile = V["_profile"]
_truth_rank = V["_truth_rank"]
_stable_order = V["_stable_order"]
_score_at_offset = V["_score_at_offset"]
_interpolate_grid = V["_interpolate_grid"]
_metrics = V["_metrics"]

EXPERIMENT = "misfit_family_point_fusion_fold2_v1"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260724
VALID_FOLD = 2
SELECTION_WELLS = 48
ALIAS_FT = 24.0
DOSES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


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


def _normalise_states(score):
    score = np.asarray(score, np.float64)
    valid = np.isfinite(score) & (score > -39.0)
    output = np.full(score.shape, -8.0, np.float64)
    usable = valid.sum(axis=0) >= 32
    if not usable.any():
        return output, usable
    local = np.where(valid[:, usable], score[:, usable], np.nan)
    centre = np.nanmedian(local, axis=0, keepdims=True)
    q25 = np.nanquantile(local, 0.25, axis=0, keepdims=True)
    q75 = np.nanquantile(local, 0.75, axis=0, keepdims=True)
    scale = np.maximum((q75 - q25) / 1.349, 1.0e-3)
    z = np.clip((local - centre) / scale, -8.0, 8.0)
    output[:, usable] = np.where(np.isfinite(z), z, -8.0)
    return output, usable


def _scores(record, pack, config, dose, typewell_override=None, no_gr=False):
    point_config = {"name": "point_typewell", "family": "point", "md_half": 0.0, "tvt_half": 0.0}
    point = _profile(
        record,
        pack,
        point_config,
        typewell_override=typewell_override,
        no_gr=no_gr,
    )
    window = _profile(
        record,
        pack,
        config,
        typewell_override=typewell_override,
        no_gr=no_gr,
    )
    point_z, _ = _normalise_states(point)
    window_z, usable = _normalise_states(window)
    blended = point_z.copy()
    blended[:, usable] = (
        (1.0 - float(dose)) * point_z[:, usable]
        + float(dose) * window_z[:, usable]
    )
    return point, window, blended, usable


def _path_metrics(items):
    truths, predictions, distances, wells = [], [], [], []
    for pack, offset in items:
        truths.append(pack.truth.astype(np.float64))
        predictions.append(_interpolate_grid(pack, offset))
        distances.append(pack.eval_distance.astype(np.float64))
        wells.append(np.full(pack.eval_rows, pack.well, dtype="U16"))
    metrics, _, _ = _metrics(
        np.concatenate(truths),
        np.concatenate(predictions),
        np.concatenate(distances),
        np.concatenate(wells),
    )
    return metrics


def _evaluate(
    wells,
    records,
    packs,
    config,
    dose,
    control="full",
    donor_by_well=None,
):
    ranks, point_ranks, window_ranks = [], [], []
    targets, distances, argmax_values, alias_wins = [], [], [], []
    usable_rows = 0
    total_rows = 0
    path_items = []

    for well in wells:
        record = records[well]
        pack = packs[well]
        override = None
        if control == "shuffled_typewell":
            override = records[donor_by_well[well]].typewell
        point, window, blended, usable = _scores(
            record,
            pack,
            config,
            dose,
            typewell_override=override,
            no_gr=control == "no_gr",
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
        argmax = STATE_OFFSETS[order[0]]
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
        "profile": config["name"],
        "dose": float(dose),
        "control": control,
        "wells": int(len(wells)),
        "grid_rows": int(len(rank)),
        "window_usable_fraction": float(usable_rows / max(total_rows, 1)),
        "median_rank": float(np.median(rank)),
        "top8_recall": float(np.mean(rank <= 8)),
        "top32_recall": float(np.mean(rank <= 32)),
        "deep_top32_recall": float(np.mean(rank[deep] <= 32)),
        "alias24_win_rate": float(np.mean(alias)),
        "argmax_grid_rmse": float(np.sqrt(np.mean(error * error))),
        "argmax_within5": float(np.mean(np.abs(error) <= 5.0)),
        "path_rmse": float(path["rmse"]),
        "path_far_gt2000_rmse": float(path["far_gt2000_rmse"]),
        # Equal total-menu budget: 16 states from each profile versus point top32.
        "union_top16_each_recall": float(
            np.mean((point_rank <= 16) | (window_rank <= 16))
        ),
        "union_top32_each_recall": float(
            np.mean((point_rank <= 32) | (window_rank <= 32))
        ),
        "window_rescues_point_top32": float(
            np.mean((point_rank > 32) & (window_rank <= 32))
        ),
        "deep_window_rescues_point_top32": float(
            np.mean((point_rank[deep] > 32) & (window_rank[deep] <= 32))
        ),
    }


def _objective(row):
    return (
        1.5 * row["top8_recall"]
        + row["top32_recall"]
        + 0.5 * row["alias24_win_rate"]
        - 0.0025 * row["median_rank"]
        - 0.0010 * row["argmax_grid_rmse"]
    )


STARTED = time.time()
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(valid_wells) != 155:
    raise RuntimeError(f"Expected 155 Fold2 wells, found {len(valid_wells)}")
rng = np.random.default_rng(SEED)
shuffled = np.asarray(valid_wells)[rng.permutation(len(valid_wells))]
selection_wells = shuffled[:SELECTION_WELLS].tolist()
confirmation_wells = shuffled[SELECTION_WELLS:].tolist()

_ensure_records(valid_wells)
_ensure_packs(valid_wells)
records = {well: _load_record(well) for well in valid_wells}
packs = {well: _load_pack(well) for well in valid_wells}

selection_rows = []
for config in CONFIGS:
    for dose in DOSES:
        row = _evaluate(selection_wells, records, packs, config, dose)
        row["selection_objective"] = _objective(row)
        selection_rows.append(row)
        print(
            "POINT-WINDOW SELECTION",
            config["name"],
            dose,
            row["top32_recall"],
            row["window_rescues_point_top32"],
            row["argmax_grid_rmse"],
            flush=True,
        )

selection = pd.DataFrame(selection_rows).sort_values("selection_objective", ascending=False)
selection.to_csv(OUT_DIR / "selection_scan.csv", index=False)
best = selection.iloc[0]
selected_name = str(best["profile"])
selected_dose = float(best["dose"])
selected_config = next(config for config in CONFIGS if config["name"] == selected_name)

baseline = _evaluate(confirmation_wells, records, packs, selected_config, 0.0)
selected = _evaluate(
    confirmation_wells, records, packs, selected_config, selected_dose
)

donor_cycle = confirmation_wells.copy()
rng.shuffle(donor_cycle)
donor_by_well = {
    donor_cycle[index]: donor_cycle[(index + 1) % len(donor_cycle)]
    for index in range(len(donor_cycle))
}
no_gr = _evaluate(
    confirmation_wells,
    records,
    packs,
    selected_config,
    selected_dose,
    control="no_gr",
)
shuffled_typewell = _evaluate(
    confirmation_wells,
    records,
    packs,
    selected_config,
    selected_dose,
    control="shuffled_typewell",
    donor_by_well=donor_by_well,
)

gains = {
    "top32_recall_gain": selected["top32_recall"] - baseline["top32_recall"],
    "deep_top32_recall_gain": selected["deep_top32_recall"] - baseline["deep_top32_recall"],
    "argmax_grid_rmse_gain": baseline["argmax_grid_rmse"] - selected["argmax_grid_rmse"],
    "path_rmse_gain": baseline["path_rmse"] - selected["path_rmse"],
    "union_top16_each_gain_vs_point_top32": (
        selected["union_top16_each_recall"] - baseline["top32_recall"]
    ),
    "window_rescues_point_top32": selected["window_rescues_point_top32"],
    "deep_window_rescues_point_top32": selected["deep_window_rescues_point_top32"],
    "shuffled_top32_drop": selected["top32_recall"] - shuffled_typewell["top32_recall"],
}
checks = {
    "selected_dose_gt_0": selected_dose > 0.0,
    "top32_recall_gain_ge_0p02": gains["top32_recall_gain"] >= 0.02,
    "deep_top32_recall_gain_ge_0p02": gains["deep_top32_recall_gain"] >= 0.02,
    "argmax_not_worse_by_2ft": gains["argmax_grid_rmse_gain"] >= -2.0,
    "fair_union_gain_ge_0p03": gains["union_top16_each_gain_vs_point_top32"] >= 0.03,
    "shuffled_top32_drop_ge_0p05": gains["shuffled_top32_drop"] >= 0.05,
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_nested_point_plus_window_complementarity_gate",
    "protocol": {
        "selection_wells": len(selection_wells),
        "confirmation_wells": len(confirmation_wells),
        "fold1_touched": False,
        "fold_source": str(fold_path),
        "selected_profile": selected_name,
        "selected_dose": selected_dose,
    },
    "baseline": baseline,
    "selected": selected,
    "controls": {"no_gr": no_gr, "shuffled_typewell": shuffled_typewell},
    "gains": gains,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> carry the frozen low-dose emission into the existing sequence decoder"
        if strict_pass
        else "FAIL -> point/window candidate menus are not deployably complementary; close this misfit family"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nPOINT-WINDOW TOP SELECTION")
print(selection.head(20).to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
