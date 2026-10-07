"""Inject the frozen V42 interval embedding into the V32 global decoder."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
EXPERIMENT = "contrastive_emission_global_decoder_v43"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V42_DIR = OOF_ROOT / "contrastive_interval_embedding_v42"
V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V42_CHECKPOINT = V42_DIR / "models" / "best_interval_encoder.pt"
V30_VERDICT = V30_DIR / "verdict.json"


def _find_source(env, names):
    configured = os.environ.get(env, "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))
    for name in names:
        candidates.extend((ROOT / name, Path(name)))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Missing source for {env}: {names}")


V42_SOURCE = _find_source(
    "ROGII_V42_SOURCE", ("contrastive_interval_embedding_v42_cell.py",)
)
V30_SOURCE = _find_source(
    "ROGII_V30_SOURCE", ("raw_stochastic_alignment_standalone_v30_cell.py",)
)


def _bootstrap_before_assignment(path, assignment):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment
            for target in node.targets
        ):
            break
        body.append(node)
    name = f"{path.stem}_definitions_for_v43"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


V42 = _bootstrap_before_assignment(V42_SOURCE, "STARTED")
V30 = _bootstrap_before_assignment(V30_SOURCE, "started")

SEED = 20260727
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
HALF_WIDTH_FT = 160.0
STATE_STEP_FT = 4.0
DOSES = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0)
OFFSETS = np.arange(
    -HALF_WIDTH_FT, HALF_WIDTH_FT + 0.5 * STATE_STEP_FT, STATE_STEP_FT,
    dtype=np.float64,
)
ZERO_STATE = int(np.argmin(np.abs(OFFSETS)))
V30.OFFSETS_FT = OFFSETS
V30.ZERO_STATE = ZERO_STATE


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _load_v30_records(wells, label):
    output, failures = {}, []
    for number, well in enumerate(wells, 1):
        try:
            output[well] = V30._load_record(well)
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"V43 load {label} {number}/{len(wells)} failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} {label} record failures: {failures[:5]}")
    return output


def _mapped_learned_score(v42_score, v42_pack, v30_record, node_rows, state_surface):
    node_md = v30_record["md"][node_rows]
    by_state = np.empty((len(node_rows), len(V42.STATE_OFFSETS)), np.float64)
    for state in range(len(V42.STATE_OFFSETS)):
        by_state[:, state] = np.interp(
            node_md,
            v42_pack.grid_md,
            v42_score[state],
            left=float(v42_score[state, 0]),
            right=float(v42_score[state, -1]),
        )
    candidate_tvt = state_surface - v30_record["z"][node_rows, None]
    candidate_offset = candidate_tvt - float(v42_pack.anchor_tvt)
    output = np.empty(candidate_offset.shape, np.float64)
    for node in range(len(node_rows)):
        output[node] = np.interp(
            candidate_offset[node], V42.STATE_OFFSETS, by_state[node], left=0.0, right=0.0
        )
    output -= output.mean(axis=1, keepdims=True)
    scale = np.maximum(output.std(axis=1, keepdims=True), 1.0)
    output = np.clip(output / scale, -6.0, 6.0)
    output[0] = 0.0
    return output


_INPUT_CACHE = {}


@torch.no_grad()
def _decoder_inputs(
    well,
    model,
    v30_records,
    v42_records,
    v42_packs,
    prior,
    config,
    device,
    control="full",
    donor=None,
):
    key = (well, control, donor)
    if key in _INPUT_CACHE:
        return _INPUT_CACHE[key]
    record30 = v30_records[well]
    record42 = v42_records[well]
    pack42 = v42_packs[well]
    rows, distance, state_surface, costs = V30._emission(record30, prior, config)
    override = None
    profile_control = "full"
    if control == "donor_exact":
        override = v42_records[donor].typewell
    elif control == "shift_exact_64":
        override = V42._shift_typewell(record42.typewell, 64.0)
    elif control in ("window_shuffle", "no_gr"):
        profile_control = control
    score42 = V42._learned_profile(
        model, record42, pack42, device, override, profile_control
    ).astype(np.float64)
    if control == "state_shuffle":
        permutation = np.random.default_rng(SEED + 431).permutation(len(V42.STATE_OFFSETS))
        score42 = score42[permutation]
    learned = _mapped_learned_score(score42, pack42, record30, rows, state_surface)
    payload = (rows, distance, state_surface, costs, learned)
    _INPUT_CACHE[key] = payload
    return payload


def _decode_inputs(record, inputs, prior, config, dose):
    rows, distance, state_surface, costs, learned = inputs
    log_emission = -costs / float(config["temperature"]) + float(dose) * learned
    nodes, states = log_emission.shape
    transitions = [None]
    for node in range(1, nodes):
        transitions.append(V30._transition_log(
            distance[node] - distance[node - 1], prior, config["transition_multiplier"]
        ))

    alpha = np.full((nodes, states), -np.inf, np.float64)
    alpha[0, ZERO_STATE] = 0.0
    for node in range(1, nodes):
        alpha[node] = log_emission[node] + V30.logsumexp(
            alpha[node - 1, :, None] + transitions[node], axis=0
        )
        alpha[node] -= V30.logsumexp(alpha[node])

    beta = np.zeros((nodes, states), np.float64)
    for node in range(nodes - 2, -1, -1):
        beta[node] = V30.logsumexp(
            transitions[node + 1] + log_emission[node + 1][None, :] + beta[node + 1][None, :],
            axis=1,
        )
        beta[node] -= V30.logsumexp(beta[node])
    log_marginal = alpha + beta
    log_marginal -= V30.logsumexp(log_marginal, axis=1, keepdims=True)
    marginal = np.exp(log_marginal)

    viterbi = np.full((nodes, states), -np.inf, np.float64)
    backpointer = np.zeros((nodes, states), np.int16)
    viterbi[0, ZERO_STATE] = 0.0
    for node in range(1, nodes):
        candidate = viterbi[node - 1, :, None] + transitions[node]
        backpointer[node] = np.argmax(candidate, axis=0).astype(np.int16)
        viterbi[node] = log_emission[node] + np.max(candidate, axis=0)
    map_states = np.empty(nodes, np.int16)
    map_states[-1] = int(np.argmax(viterbi[-1]))
    for node in range(nodes - 2, -1, -1):
        map_states[node] = backpointer[node + 1, map_states[node + 1]]

    mean_surface = np.sum(marginal * state_surface, axis=1)
    map_surface = state_surface[np.arange(nodes), map_states]
    entropy = -np.sum(marginal * np.log(np.maximum(marginal, 1e-12)), axis=1) / math.log(states)
    offset_mean = marginal @ OFFSETS
    offset_std = np.sqrt(np.maximum(marginal @ (OFFSETS ** 2) - offset_mean ** 2, 0.0))
    predictions = {}
    for estimator, surface in (("posterior_mean", mean_surface), ("map", map_surface)):
        interpolated = np.interp(record["md"][record["evaluation"]], record["md"][rows], surface)
        predictions[estimator] = interpolated - record["z"][record["evaluation"]]
    diagnostics = {
        "mean_entropy": float(np.mean(entropy[1:])),
        "mean_offset_std": float(np.mean(offset_std[1:])),
        "terminal_offset_std": float(offset_std[-1]),
        "nodes": int(nodes),
    }
    return predictions, diagnostics


def _decode_many(
    wells,
    model,
    v30_records,
    v42_records,
    v42_packs,
    prior,
    config,
    device,
    dose,
    label,
    control="full",
    donors=None,
):
    output = {}
    for number, well in enumerate(wells, 1):
        inputs = _decoder_inputs(
            well, model, v30_records, v42_records, v42_packs, prior, config, device,
            control, None if donors is None else donors[well],
        )
        output[well] = _decode_inputs(v30_records[well], inputs, prior, config, dose)
        if number % 25 == 0 or number == len(wells):
            print(f"V43 {label} {number}/{len(wells)}", flush=True)
    records = [v30_records[well] for well in wells]
    bywell = V30._metric_rows(records, output, label)
    return bywell, V30._summary(bywell)


def _selected(summary, estimator="posterior_mean"):
    rows = summary.loc[summary["estimator"].eq(estimator)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {estimator} row, got {len(rows)}")
    return rows.iloc[0].to_dict()


STARTED = time.time()
for required in (V42_CHECKPOINT, V30_VERDICT, V30_DIR / "folds.csv"):
    if not required.exists():
        raise FileNotFoundError(required)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(V42_CHECKPOINT, map_location=device, weights_only=False)
model = V42.CrossModalIntervalEncoder().to(device)
model.load_state_dict(state["model"])
model.eval()

v30_verdict = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
prior = v30_verdict["outer_train_prior"]
config = dict(v30_verdict["selected_config"])
folds, fold_path = V42._load_folds()
folds["well"] = folds["well"].astype(str)
fold3_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
rng = np.random.default_rng(SEED + 43)
fold3_order = rng.permutation(fold3_wells).tolist()
discovery_wells = fold3_order[:DISCOVERY_WELLS]
confirmation_wells = fold3_order[DISCOVERY_WELLS:]
needed = sorted(set(fold3_wells + fold2_wells))
V42._ensure_records(needed)
V42._ensure_packs(needed)
v42_records = {well: V42._load_record(well) for well in needed}
v42_packs = {well: V42._load_pack(well) for well in needed}
v30_records = _load_v30_records(needed, "Fold23")

discovery_parts = []
for dose in DOSES:
    bywell, summary = _decode_many(
        discovery_wells, model, v30_records, v42_records, v42_packs,
        prior, config, device, dose, f"fold3_discovery_d{dose:g}",
    )
    summary["dose"] = dose
    discovery_parts.append(summary)
discovery = pd.concat(discovery_parts, ignore_index=True).sort_values("rmse")
discovery.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
selected_row = discovery.loc[discovery["estimator"].eq("posterior_mean")].sort_values("rmse").iloc[0]
selected_dose = float(selected_row["dose"])

confirmation_parts = []
for dose in (0.0, selected_dose):
    _, summary = _decode_many(
        confirmation_wells, model, v30_records, v42_records, v42_packs,
        prior, config, device, dose, f"fold3_confirmation_d{dose:g}",
    )
    summary["dose"] = dose
    confirmation_parts.append(summary)
confirmation = pd.concat(confirmation_parts, ignore_index=True)
confirmation.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)

ordered = sorted(fold2_wells)
donors = {well: ordered[(index + 1) % len(ordered)] for index, well in enumerate(ordered)}
fold2_parts = []
for dose, control in ((0.0, "full"), (selected_dose, "full"), (selected_dose, "state_shuffle"), (selected_dose, "window_shuffle"), (selected_dose, "donor_exact"), (selected_dose, "shift_exact_64"), (selected_dose, "no_gr")):
    _, summary = _decode_many(
        fold2_wells, model, v30_records, v42_records, v42_packs,
        prior, config, device, dose, f"fold2_{control}_d{dose:g}", control, donors,
    )
    summary["dose"] = dose
    summary["control"] = control
    fold2_parts.append(summary)
fold2 = pd.concat(fold2_parts, ignore_index=True)
fold2.to_csv(OUT_DIR / "fold2_application_controls.csv", index=False)

def _metric(frame, dose, control="full"):
    rows = frame.loc[
        frame["dose"].eq(dose) & frame["control"].eq(control)
        & frame["estimator"].eq("posterior_mean")
    ] if "control" in frame else frame.loc[
        frame["dose"].eq(dose) & frame["estimator"].eq("posterior_mean")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Cannot pick dose={dose}, control={control}: {len(rows)}")
    return rows.iloc[0].to_dict()

fold3_base = _metric(confirmation, 0.0)
fold3_selected = _metric(confirmation, selected_dose)
fold2_base = _metric(fold2, 0.0)
fold2_selected = _metric(fold2, selected_dose)
controls = {name: _metric(fold2, selected_dose, name) for name in ("state_shuffle", "window_shuffle", "donor_exact", "shift_exact_64", "no_gr")}
fold3_gain = fold3_base["rmse"] - fold3_selected["rmse"]
fold2_gain = fold2_base["rmse"] - fold2_selected["rmse"]
fold2_far_gain = fold2_base["far_gt2000_rmse"] - fold2_selected["far_gt2000_rmse"]
control_gain = {name: fold2_base["rmse"] - row["rmse"] for name, row in controls.items()}
gates = {
    "selected_dose_nonzero": bool(selected_dose > 0.0),
    "fold3_gain_ge_0p5": bool(fold3_gain >= 0.5),
    "fold2_gain_ge_0p5": bool(fold2_gain >= 0.5),
    "fold2_far_gain_positive": bool(fold2_far_gain > 0.0),
    "state_shuffle_retains_le_30pct": bool(control_gain["state_shuffle"] <= max(0.3 * fold2_gain, 0.1)),
    "window_shuffle_retains_le_30pct": bool(control_gain["window_shuffle"] <= max(0.3 * fold2_gain, 0.1)),
    "no_gr_retains_le_30pct": bool(control_gain["no_gr"] <= max(0.3 * fold2_gain, 0.1)),
    "donor_weakens": bool(control_gain["donor_exact"] < fold2_gain),
    "shift_weakens": bool(control_gain["shift_exact_64"] < fold2_gain),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_learned_emission_dose_selection_Fold2_global_decoder_confirmation",
    "read_only_checkpoint": str(V42_CHECKPOINT),
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "decoder": {"half_width_ft": HALF_WIDTH_FT, "state_step_ft": STATE_STEP_FT, "states": len(OFFSETS), "v30_config": config},
    "selected_dose": selected_dose,
    "fold3": {"baseline": fold3_base, "selected": fold3_selected, "gain": fold3_gain},
    "fold2": {"baseline": fold2_base, "selected": fold2_selected, "gain": fold2_gain, "far_gain": fold2_far_gain},
    "controls": controls,
    "gates": gates,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": "PASS -> cross-fit five folds and test cross-evidence value against the 6.377 endpoint" if strict_pass else "FAIL -> learned local evidence cannot survive structural decoding; close this branch",
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV43 DISCOVERY SCAN")
print(discovery.to_string(index=False))
print("\nV43 FOLD3 CONFIRMATION")
print(confirmation.to_string(index=False))
print("\nV43 FOLD2 APPLICATION AND CONTROLS")
print(fold2.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
