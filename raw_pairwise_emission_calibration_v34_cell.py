"""Calibrate learned pair evidence against the frozen handcrafted emission.

V33 improved local candidate retrieval but produced an overconfident global
path.  This cell does not retrain.  It combines log evidence as

    log p = log p_handcrafted + alpha * log p_learned

where alpha is selected on all Fold 3 wells and applied once to Fold 2.
Alpha zero exactly reproduces the repaired V32 Cauchy decoder.  No historical
prediction or non-raw feature is used.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENT = "raw_pairwise_emission_calibration_v34"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V30_VERDICT = V30_DIR / "verdict.json"
V33_DIR = OOF_ROOT / "raw_pairwise_emission_stochastic_v33"
CHECKPOINT = V33_DIR / "models" / "best_pair_emission.pt"
SOURCE_CANDIDATES = (
    ROOT / "raw_pairwise_emission_stochastic_v33_cell.py",
    Path.cwd() / "raw_pairwise_emission_stochastic_v33_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload raw_pairwise_emission_stochastic_v33_cell.py")

ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_pairwise_v33_calibration_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module


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


M = _bootstrap(SOURCE_PATH)


def _load_records(wells, label):
    output = []
    failures = []
    for number, well in enumerate(wells, 1):
        try:
            output.append(M.V._load_record(well))
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"V34 load {label} {number}/{len(wells)} failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} {label} failures: {failures[:5]}")
    return output


def _learned_maps(records, model, mean, scale, device, label):
    output = {}
    for number, record in enumerate(records, 1):
        rows, distance, state_surface, features, _, _, _ = M._pair_features(record)
        score = M._scores(model, features.astype(np.float16), mean, scale, device)
        cost = -score.astype(np.float64)
        cost -= np.min(cost, axis=1, keepdims=True)
        cost[0] = 0.0
        output[record["well"]] = (rows, distance, state_surface, cost)
        if number % 25 == 0 or number == len(records):
            print(f"V34 learned maps {label} {number}/{len(records)}", flush=True)
    return output


def _handcrafted_maps(records, prior, config, label):
    output = {}
    for number, record in enumerate(records, 1):
        rows, distance, state_surface, cost = M.V._emission(record, prior, config)
        # V30 used temperature 2.0.  Store the actual negative log evidence so
        # the combined decoder can use temperature one.
        cost = cost.astype(np.float64) / float(config["temperature"])
        output[record["well"]] = (rows, distance, state_surface, cost)
        if number % 25 == 0 or number == len(records):
            print(f"V34 handcrafted maps {label} {number}/{len(records)}", flush=True)
    return output


def _combine(handcrafted, learned, alpha):
    output = {}
    for well, (rows, distance, surface, hand_cost) in handcrafted.items():
        learned_cost = learned[well][3]
        if hand_cost.shape != learned_cost.shape:
            raise RuntimeError(f"{well}: cost-map shape mismatch")
        cost = hand_cost + float(alpha) * learned_cost
        cost -= np.min(cost, axis=1, keepdims=True)
        cost[0] = 0.0
        output[well] = (rows, distance, surface, cost)
    return output


def _decode(records, maps, prior, config, label):
    original = M.V._emission

    def emission(record, _prior, _config, donor=None, no_gr=False):
        rows, distance, surface, cost = maps[record["well"]]
        return rows, distance, surface, np.zeros_like(cost) if no_gr else cost

    M.V._emission = emission
    decoded = {}
    try:
        for number, record in enumerate(records, 1):
            decoded[record["well"]] = M.V._decode(
                record, prior, config, sample_paths=False
            )
            if number % 25 == 0 or number == len(records):
                print(f"V34 decode {label} {number}/{len(records)}", flush=True)
    finally:
        M.V._emission = original
    bywell = M.V._metric_rows(records, decoded, label)
    summary = M.V._summary(bywell)
    return bywell, summary


def _posterior(summary):
    return summary[summary["estimator"].eq("posterior_mean")].iloc[0]


started = time.time()
for required in (V30_VERDICT, CHECKPOINT):
    if not required.exists():
        raise FileNotFoundError(required)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v30 = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
M.PRIOR = v30["outer_train_prior"]
hand_config = dict(v30["selected_config"])
decode_config = dict(hand_config)
decode_config["temperature"] = 1.0
folds = pd.read_csv(V30_DIR / "folds.csv", dtype={"well": str})
fold3_wells = sorted(folds.loc[folds["fold"].eq(3), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(2), "well"].astype(str))
fold3_records = _load_records(fold3_wells, "fold3")
fold2_records = _load_records(fold2_wells, "fold2")

state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = M.PairEmission(int(state["features"])).to(device)
model.load_state_dict(state["model"])
mean = np.asarray(state["mean"], np.float32)
scale = np.asarray(state["scale"], np.float32)

learned3 = _learned_maps(fold3_records, model, mean, scale, device, "fold3")
hand3 = _handcrafted_maps(fold3_records, M.PRIOR, hand_config, "fold3")
scan_rows = []
scan_bywell = []
for alpha in ALPHAS:
    label = f"fold3_alpha{alpha:g}"
    bywell, summary = _decode(
        fold3_records, _combine(hand3, learned3, alpha), M.PRIOR, decode_config, label
    )
    bywell["alpha"] = alpha
    summary["alpha"] = alpha
    scan_bywell.append(bywell)
    scan_rows.append(summary)
scan_frame = pd.concat(scan_rows, ignore_index=True)
scan_bywell_frame = pd.concat(scan_bywell, ignore_index=True)
scan_frame.to_csv(OUT_DIR / "fold3_alpha_scan.csv", index=False)
scan_bywell_frame.to_csv(OUT_DIR / "fold3_alpha_bywell.csv", index=False)
print("\nV34 FOLD3 ALPHA SCAN")
print(scan_frame.to_string(index=False), flush=True)

posterior_scan = scan_frame[scan_frame["estimator"].eq("posterior_mean")]
selected = posterior_scan.sort_values(
    ["rmse", "far_gt2000_rmse", "alpha"], ascending=True
).iloc[0]
selected_alpha = float(selected["alpha"])

learned2 = _learned_maps(fold2_records, model, mean, scale, device, "fold2")
hand2 = _handcrafted_maps(fold2_records, M.PRIOR, hand_config, "fold2")
fold2_parts = []
fold2_summaries = []
for alpha in sorted(set((0.0, selected_alpha))):
    label = f"fold2_alpha{alpha:g}"
    bywell, summary = _decode(
        fold2_records, _combine(hand2, learned2, alpha), M.PRIOR, decode_config, label
    )
    bywell["alpha"] = alpha
    summary["alpha"] = alpha
    fold2_parts.append(bywell)
    fold2_summaries.append(summary)
fold2_frame = pd.concat(fold2_parts, ignore_index=True)
fold2_summary = pd.concat(fold2_summaries, ignore_index=True)
fold2_frame.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
fold2_summary.to_csv(OUT_DIR / "fold2_summary.csv", index=False)
print("\nV34 FOLD2 CONFIRMATION")
print(fold2_summary.to_string(index=False), flush=True)

base3 = _posterior(scan_frame[scan_frame["alpha"].eq(0.0)])
chosen3 = _posterior(scan_frame[scan_frame["alpha"].eq(selected_alpha)])
base2 = _posterior(fold2_summary[fold2_summary["alpha"].eq(0.0)])
chosen2 = _posterior(fold2_summary[fold2_summary["alpha"].eq(selected_alpha)])
gates = {
    "fold3_gain_ge_0p25": bool(base3["rmse"] - chosen3["rmse"] >= 0.25),
    "fold2_gain_positive": bool(chosen2["rmse"] < base2["rmse"]),
    "fold2_far_gain_positive": bool(
        chosen2["far_gt2000_rmse"] < base2["far_gt2000_rmse"]
    ),
    "selected_alpha_nonzero": bool(selected_alpha > 0.0),
}
if all(gates.values()):
    next_action = "PASS -> retain calibrated learned evidence and train five fold-clean endpoints"
else:
    next_action = "FAIL -> learned local rank is not consumable by this Markov decoder; move to a trainable sequence encoder"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_learned_evidence_dose_selection_Fold2_confirmation",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "formula": "handcrafted_cost/2 + alpha*learned_cost",
    "alpha_scan": list(ALPHAS),
    "selected_alpha": selected_alpha,
    "fold3": {
        "baseline_rmse": float(base3["rmse"]),
        "selected_rmse": float(chosen3["rmse"]),
        "gain": float(base3["rmse"] - chosen3["rmse"]),
    },
    "fold2": {
        "baseline_rmse": float(base2["rmse"]),
        "selected_rmse": float(chosen2["rmse"]),
        "gain": float(base2["rmse"] - chosen2["rmse"]),
        "baseline_far_rmse": float(base2["far_gt2000_rmse"]),
        "selected_far_rmse": float(chosen2["far_gt2000_rmse"]),
    },
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": next_action,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
