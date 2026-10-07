"""Complete-well path-bank gate for the multiscale path emission.

The v0 audit showed that the legal typewell emission contains enough rowwise
information at top 64 (Fold-2 oracle RMSE about 3.44 ft), but top 32 is still
too narrow.  This cell asks the next necessary question before training a
selector: can those rowwise states form a small bank of coherent complete-well
trajectories whose whole-path oracle is below 5.5 ft?

Protocol:
  * fit the offset-rate transition prior on Folds 4/5 only;
  * choose fixed sparse-DP configurations on 24 Fold-3 wells;
  * confirm the frozen union on the remaining Fold-3 wells;
  * apply once to Fold 2, with Fold 1 untouched;
  * repeat Fold-2 generation with a rolled-position matched control.

Truth is used only to score path-bank oracles and to select configurations on
the declared selection split.  It is never used to generate a candidate path.
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
        Path("/root/autodl-tmp/multiscale_path_emission_topk_gateway_v0_cell.py"),
        Path.cwd() / "multiscale_path_emission_topk_gateway_v0_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload multiscale_path_emission_topk_gateway_v0_cell.py and its "
            "three cost-map prerequisites to /root/autodl-tmp."
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
    module_name = "multiscale_path_emission_v0_defs"
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
PACK_DIR = V["PACK_DIR"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float32)
_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_pack = V["_load_pack"]
_metrics = V["_metrics"]
_recipe_score = V["_recipe_score"]
_interpolate_grid = V["_interpolate_grid"]

EXPERIMENT = "multiscale_emission_pathbank_gateway_v1"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260725
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 24
TOP_STATES = 64
PATH_BUDGETS = (1, 8, 16, 32, 64, 128, 256)
CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
PATHS_PER_CHECKPOINT = 16

# Frozen from the nested v0 Fold-3 selection.
RECIPE = {
    "id": "type_ms248",
    "radii": (2, 4, 8),
    "references": (0,),
    "clip": 40.0,
    "combine": "sum",
}

# Small, preregistered structural menu.  Configurations differ in how strongly
# they consume the same emission, not in any truth-derived per-well setting.
CONFIGS = (
    {"id": "weak_wide", "transition_lambda": 0.10, "jump_cap_ft": 48.0},
    {"id": "weak_mid", "transition_lambda": 0.10, "jump_cap_ft": 24.0},
    {"id": "mid_wide", "transition_lambda": 0.35, "jump_cap_ft": 48.0},
    {"id": "mid_mid", "transition_lambda": 0.35, "jump_cap_ft": 24.0},
    {"id": "strong_mid", "transition_lambda": 1.00, "jump_cap_ft": 24.0},
    {"id": "strong_tight", "transition_lambda": 1.00, "jump_cap_ft": 12.0},
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


def _fit_transition_prior(packs):
    rates = []
    for pack in packs:
        mask = pack.supervised & np.isfinite(pack.target)
        index = np.flatnonzero(mask)
        if len(index) < 3:
            continue
        dm = np.diff(pack.grid_md[index].astype(np.float64))
        dt = np.diff(pack.target[index].astype(np.float64))
        valid = np.isfinite(dm) & np.isfinite(dt) & (dm > 0.0)
        rates.append(dt[valid] / dm[valid])
    values = np.concatenate(rates)
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    scale = max(1.4826 * mad, 0.005)
    return {
        "rate_centre": centre,
        "rate_scale": scale,
        "count": int(len(values)),
        "rate_p01": float(np.quantile(values, 0.01)),
        "rate_p99": float(np.quantile(values, 0.99)),
    }


def _huber(values):
    absolute = np.abs(values)
    return np.where(absolute <= 1.0, 0.5 * values**2, absolute - 0.5)


def _candidate_states(score, anchor_grid):
    candidates = []
    zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    for t in range(score.shape[1]):
        if t <= anchor_grid:
            candidates.append(np.asarray([zero_state], np.int16))
            continue
        column = score[:, t]
        k = min(TOP_STATES, len(column))
        index = np.argpartition(column, -k)[-k:]
        index = index[np.argsort(-column[index])]
        candidates.append(index.astype(np.int16))
    return candidates


def _transition_matrix(previous, current, delta_md, prior, config):
    previous_offset = STATE_OFFSETS[previous].astype(np.float64)
    current_offset = STATE_OFFSETS[current].astype(np.float64)
    jump = current_offset[None, :] - previous_offset[:, None]
    expected = prior["rate_centre"] * float(delta_md)
    scale = max(prior["rate_scale"] * float(delta_md), 0.75)
    penalty = _huber((jump - expected) / scale)
    penalty *= float(config["transition_lambda"])
    penalty[np.abs(jump) > float(config["jump_cap_ft"])] = np.inf
    return penalty


def _sparse_path_bank(pack, score, prior, config):
    """Generate deployable complete paths by checkpoint-conditioned sparse DP."""
    candidates = _candidate_states(score, pack.anchor_grid)
    start = int(pack.anchor_grid)
    stop = score.shape[1]
    emissions = [score[index, t].astype(np.float64) for t, index in enumerate(candidates)]

    forward = [None] * stop
    parents = [None] * stop
    forward[start] = np.asarray([0.0], np.float64)
    for t in range(start + 1, stop):
        penalty = _transition_matrix(
            candidates[t - 1], candidates[t], pack.grid_md[t] - pack.grid_md[t - 1], prior, config
        )
        value = forward[t - 1][:, None] - penalty
        parent = np.argmax(value, axis=0)
        best = value[parent, np.arange(value.shape[1])]
        forward[t] = best + emissions[t]
        parents[t] = parent.astype(np.int16)

    backward = [None] * stop
    successors = [None] * stop
    backward[stop - 1] = emissions[stop - 1].copy()
    for t in range(stop - 2, start - 1, -1):
        penalty = _transition_matrix(
            candidates[t], candidates[t + 1], pack.grid_md[t + 1] - pack.grid_md[t], prior, config
        )
        value = backward[t + 1][None, :] - penalty
        successor = np.argmax(value, axis=1)
        best = value[np.arange(value.shape[0]), successor]
        backward[t] = emissions[t] + best
        successors[t] = successor.astype(np.int16)

    span = max(stop - start - 1, 1)
    checkpoints = sorted(
        set(min(stop - 1, start + max(1, int(round(span * fraction)))) for fraction in CHECKPOINT_FRACTIONS)
    )
    bank = []
    seen = set()
    for checkpoint in checkpoints:
        joint = forward[checkpoint] + backward[checkpoint] - emissions[checkpoint]
        take = min(PATHS_PER_CHECKPOINT, len(joint))
        states = np.argpartition(joint, -take)[-take:]
        states = states[np.argsort(-joint[states])]
        for checkpoint_state in states:
            local = np.zeros(stop, np.int16)
            local[checkpoint] = int(checkpoint_state)
            for t in range(checkpoint, start, -1):
                local[t - 1] = parents[t][local[t]]
            for t in range(checkpoint, stop - 1):
                local[t + 1] = successors[t][local[t]]
            state_index = np.asarray(
                [candidates[t][local[t]] for t in range(stop)], dtype=np.int16
            )
            path = STATE_OFFSETS[state_index].astype(np.float32)
            path[: start + 1] = 0.0
            key = np.rint(path * 2.0).astype(np.int16).tobytes()
            if key in seen:
                continue
            seen.add(key)
            bank.append(
                {
                    "path": path,
                    "objective": float(joint[checkpoint_state]),
                    "checkpoint": int(checkpoint),
                    "config": config["id"],
                }
            )
    bank.sort(key=lambda row: -row["objective"])
    return bank


def _round_robin_union(per_config, config_ids):
    output = []
    seen = set()
    maximum = max((len(per_config[config_id]) for config_id in config_ids), default=0)
    for rank in range(maximum):
        for config_id in config_ids:
            rows = per_config[config_id]
            if rank >= len(rows):
                continue
            row = rows[rank]
            key = np.rint(row["path"] * 2.0).astype(np.int16).tobytes()
            if key in seen:
                continue
            seen.add(key)
            output.append(row)
    return output


def _path_sse(pack, path):
    prediction = _interpolate_grid(pack, path)
    return float(np.sum((prediction - pack.truth.astype(np.float64)) ** 2))


def _evaluate_banks(packs, banks, budgets=PATH_BUDGETS):
    curves = []
    bywell_rows = []
    for budget in budgets:
        truths, estimates, distances, wells = [], [], [], []
        for pack in packs:
            bank = banks[pack.well][: min(int(budget), len(banks[pack.well]))]
            if not bank:
                raise RuntimeError(f"{pack.well}: empty path bank")
            sse = np.asarray([_path_sse(pack, row["path"]) for row in bank])
            choice = int(np.argmin(sse))
            prediction = _interpolate_grid(pack, bank[choice]["path"])
            truths.append(pack.truth.astype(np.float64))
            estimates.append(prediction)
            distances.append(pack.eval_distance.astype(np.float64))
            wells.append(np.full(pack.eval_rows, pack.well, dtype="U16"))
            if budget == budgets[-1]:
                bywell_rows.append(
                    {
                        "well": pack.well,
                        "bank_paths": len(banks[pack.well]),
                        "oracle_choice": choice,
                        "oracle_sse": float(sse[choice]),
                        "oracle_rmse": float(np.sqrt(sse[choice] / pack.eval_rows)),
                        "winning_config": bank[choice]["config"],
                    }
                )
        metrics, _, _ = _metrics(
            np.concatenate(truths),
            np.concatenate(estimates),
            np.concatenate(distances),
            np.concatenate(wells),
        )
        curves.append({"budget": int(budget), **metrics})
    return curves, pd.DataFrame(bywell_rows)


def _generate(packs, configs, prior, control="full"):
    banks = {}
    per_config_banks = {config["id"]: {} for config in configs}
    for number, pack in enumerate(packs, 1):
        score = _recipe_score(pack, RECIPE, control=control)
        generated = {}
        for config in configs:
            generated[config["id"]] = _sparse_path_bank(pack, score, prior, config)
            per_config_banks[config["id"]][pack.well] = generated[config["id"]]
        banks[pack.well] = _round_robin_union(generated, [config["id"] for config in configs])
        if number % 25 == 0 or number == len(packs):
            print(f"pathbank {control} {number}/{len(packs)}", flush=True)
    return banks, per_config_banks


def _curve_metric(curve, budget, key="rmse"):
    return next(row[key] for row in curve if row["budget"] == budget)


STARTED = time.time()
np.random.seed(SEED)
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
fold_by_well = dict(zip(folds["well"], folds["fold"].astype(int)))

prior_wells = folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"].tolist()
fold3_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
if len(prior_wells) not in (308, 309) or len(fold3_wells) not in (154, 155) or len(fold2_wells) != 155:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3/Fold2 sizes: {len(prior_wells)}/{len(fold3_wells)}/{len(fold2_wells)}"
    )

needed = sorted(set(prior_wells + fold3_wells + fold2_wells))
_ensure_records(needed)
_ensure_packs(needed)
packs = {well: _load_pack(well) for well in needed}
prior = _fit_transition_prior([packs[well] for well in prior_wells])
print("PATHBANK TRANSITION PRIOR", prior, flush=True)

rng = np.random.default_rng(SEED)
fold3_order = np.asarray(fold3_wells)[rng.permutation(len(fold3_wells))]
discovery_wells = fold3_order[:DISCOVERY_WELLS].tolist()
confirmation_wells = fold3_order[DISCOVERY_WELLS:].tolist()
discovery_packs = [packs[well] for well in discovery_wells]
confirmation_packs = [packs[well] for well in confirmation_wells]
application_packs = [packs[well] for well in fold2_wells]

_, discovery_per_config = _generate(discovery_packs, CONFIGS, prior)
config_rows = []
for config in CONFIGS:
    config_id = config["id"]
    curve, _ = _evaluate_banks(
        discovery_packs,
        discovery_per_config[config_id],
        budgets=(16, 32, 64),
    )
    config_rows.append(
        {
            "config": config_id,
            "oracle16_rmse": _curve_metric(curve, 16),
            "oracle32_rmse": _curve_metric(curve, 32),
            "oracle64_rmse": _curve_metric(curve, 64),
        }
    )
config_table = pd.DataFrame(config_rows).sort_values(["oracle64_rmse", "oracle32_rmse"])
config_table.to_csv(OUT_DIR / "fold3_discovery_config_scan.csv", index=False)
print("PATHBANK DISCOVERY CONFIGS\n", config_table.to_string(index=False), flush=True)

selected_ids = config_table.head(3)["config"].tolist()
selected_configs = [config for config in CONFIGS if config["id"] in selected_ids]
confirmation_banks, _ = _generate(confirmation_packs, selected_configs, prior)
confirmation_curve, confirmation_bywell = _evaluate_banks(confirmation_packs, confirmation_banks)
confirmation_bywell.to_csv(OUT_DIR / "fold3_confirmation_bywell.csv", index=False)
print("PATHBANK FOLD3 CONFIRMATION\n", pd.DataFrame(confirmation_curve).to_string(index=False), flush=True)

application_banks, _ = _generate(application_packs, selected_configs, prior)
application_curve, application_bywell = _evaluate_banks(application_packs, application_banks)
application_bywell.to_csv(OUT_DIR / "fold2_pathbank_oracle_bywell.csv", index=False)
print("PATHBANK FOLD2\n", pd.DataFrame(application_curve).to_string(index=False), flush=True)

control_banks, _ = _generate(application_packs, selected_configs, prior, control="rolled_position")
control_curve, _ = _evaluate_banks(application_packs, control_banks, budgets=(64, 128, 256))
print("PATHBANK FOLD2 CONTROL\n", pd.DataFrame(control_curve).to_string(index=False), flush=True)

fold2_oracle64 = _curve_metric(application_curve, 64)
fold2_oracle128 = _curve_metric(application_curve, 128)
fold2_p95_128 = _curve_metric(application_curve, 128, "p95_well_rmse")
control_oracle128 = _curve_metric(control_curve, 128)
checks = {
    "fold3_confirmation_oracle128_rmse_le_5p5": _curve_metric(confirmation_curve, 128) <= 5.5,
    "fold2_oracle64_rmse_le_6p0": fold2_oracle64 <= 6.0,
    "fold2_oracle128_rmse_le_5p5": fold2_oracle128 <= 5.5,
    "fold2_oracle128_p95_le_10": fold2_p95_128 <= 10.0,
    "rolled_control_oracle128_worse_by_ge_1": control_oracle128 - fold2_oracle128 >= 1.0,
}
strict_pass = all(checks.values())

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_complete_well_sparse_pathbank_oracle_gate",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "truth_used_during_generation": False,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "discovery_wells": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "application_fold": APPLICATION_FOLD,
        "application_wells": len(application_packs),
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
        "fold_source": str(fold_path),
        "source": str(SOURCE_PATH),
    },
    "emission": {
        "recipe": RECIPE,
        "top_states_per_grid": TOP_STATES,
    },
    "transition_prior": prior,
    "candidate_generation": {
        "checkpoint_fractions": list(CHECKPOINT_FRACTIONS),
        "paths_per_checkpoint": PATHS_PER_CHECKPOINT,
        "configs_scanned": list(CONFIGS),
        "selected_configs": selected_configs,
    },
    "fold3_discovery": config_rows,
    "fold3_confirmation_curve": confirmation_curve,
    "fold2_curve": application_curve,
    "fold2_rolled_position_control_curve": control_curve,
    "diagnostics": {
        "fold2_oracle64_rmse": fold2_oracle64,
        "fold2_oracle128_rmse": fold2_oracle128,
        "fold2_rolled_control_oracle128_rmse": control_oracle128,
        "rolled_control_degradation": control_oracle128 - fold2_oracle128,
        "mean_fold2_bank_paths": float(np.mean([len(application_banks[well]) for well in fold2_wells])),
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze the path bank and train a complete-well listwise scorer"
        if strict_pass
        else "FAIL -> rowwise top64 support does not form a selectable complete-path bank; improve emission before training"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("PATHBANK FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
