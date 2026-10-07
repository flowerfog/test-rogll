"""M342-K2: frozen consensus-gate attribution and deployment audit.

K1 independently selected magnitude_soft_mean1 in all five outer folds. K2
adds no profile or threshold search. It reproduces that crossfit endpoint,
compares it with a mean-preserving reversed gate, and exports the full-data
normalizer and G3 weights required for exact deployment.
"""

from pathlib import Path
import hashlib
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
K1_SOURCE = ROOT / "main342_consensus_rejection_stack_k1.py"
OUT = ROOT / "oof_exports" / "main342_consensus_rejection_attribution_k2"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_PROFILE = "magnitude_soft_mean1"
FROZEN_SHRINKAGE = 0.20
G3_EXPECTED_RMSE = 7.7079303941645545
K1_EXPECTED_RMSE = 7.689693565766868


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_k1():
    if not K1_SOURCE.is_file():
        raise FileNotFoundError(K1_SOURCE)
    source = K1_SOURCE.read_text(encoding="utf-8")
    marker = "G = _load_g1()"
    if marker not in source:
        raise RuntimeError("K1 execution marker not found")
    name = "m342_k2_k1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(K1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(K1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


K = _load_k1()
G = K["_load_g1"]()
V = G["_load_inventory"]()
base = V["base"].reset_index(drop=True)
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
base_error = base_prediction - truth
fold = base["fold"].to_numpy(np.int8)
active_names = tuple(K["ACTIVE_ENDPOINTS"])
missing = [name for name in active_names if name not in names]
if missing:
    raise RuntimeError("Missing active endpoints: " + repr(missing))

active_indices = [names.index(name) for name in active_names]
active_delta = paths[active_indices] - base_prediction[None, :]
raw_gates = K["_raw_gates"](active_delta, base)
raw_factor = raw_gates[FROZEN_PROFILE]
all_indices = [index for index in range(len(names)) if index != base_index]
candidate_names = [names[index] for index in all_indices]
all_delta = paths[all_indices] - base_prediction[None, :]

g3 = base_prediction.copy()
gated = base_prediction.copy()
reversed_gate = base_prediction.copy()
fold_reports = []
fold_weights = []
for held in range(1, 6):
    train = fold != held
    valid = fold == held
    raw_alpha, objective = G["_fit_coefficients"](all_delta, base_error, train)
    alpha = FROZEN_SHRINKAGE * raw_alpha
    move = alpha @ all_delta[:, valid]
    denominator = max(float(np.mean(raw_factor[train])), 1e-6)
    factor = np.clip(
        raw_factor[valid] / denominator,
        0.0,
        float(K["MAX_NORMALISED_FACTOR"]),
    )
    # Reflection about one preserves the intended average dose while assigning
    # larger corrections to disagreement rows instead of agreement rows.
    reverse = np.clip(2.0 - factor, 0.0, float(K["MAX_NORMALISED_FACTOR"]))
    reverse *= float(np.mean(factor)) / max(float(np.mean(reverse)), 1e-12)
    g3[valid] += move
    gated[valid] += factor * move
    reversed_gate[valid] += reverse * move

    local = base.loc[valid].reset_index(drop=True)
    base_metric = K["_metrics"](local, base_prediction[valid], "base")
    g3_metric = K["_metrics"](local, g3[valid], "g3")
    gate_metric = K["_metrics"](local, gated[valid], "gate")
    reverse_metric = K["_metrics"](local, reversed_gate[valid], "reverse")
    fold_reports.append({
        "fold": held,
        "normalizer": denominator,
        "mean_factor": float(np.mean(factor)),
        "base_rmse": base_metric["rmse"],
        "g3_rmse": g3_metric["rmse"],
        "rmse": gate_metric["rmse"],
        "increment_vs_g3": g3_metric["rmse"] - gate_metric["rmse"],
        "far_increment_vs_g3": g3_metric["far_gt2000_rmse"] - gate_metric["far_gt2000_rmse"],
        "reverse_loss": reverse_metric["rmse"] - gate_metric["rmse"],
        "p95_change_vs_g3": gate_metric["p95_well_rmse"] - g3_metric["p95_well_rmse"],
    })
    fold_weights.append({
        "fold": held,
        "objective": objective,
        **{f"weight__{name}": float(value) for name, value in zip(candidate_names, alpha)},
    })
    print("M342-K2 FOLD", _clean(fold_reports[-1]), flush=True)

full_mask = np.ones(len(base), dtype=bool)
full_raw_alpha, full_objective = G["_fit_coefficients"](
    all_delta, base_error, full_mask
)
full_alpha = FROZEN_SHRINKAGE * full_raw_alpha
full_normalizer = max(float(np.mean(raw_factor)), 1e-6)
full_factor = np.clip(
    raw_factor / full_normalizer,
    0.0,
    float(K["MAX_NORMALISED_FACTOR"]),
)
full_g3 = base_prediction + full_alpha @ all_delta
full_gated = base_prediction + full_factor * (full_alpha @ all_delta)

summary = pd.DataFrame([
    K["_metrics"](base, base_prediction, "v168"),
    K["_metrics"](base, g3, "g3_reproduction"),
    K["_metrics"](base, gated, "frozen_consensus_crossfit"),
    K["_metrics"](base, reversed_gate, "reversed_consensus_control"),
    K["_metrics"](base, full_g3, "g3_full_fit_diagnostic"),
    K["_metrics"](base, full_gated, "consensus_full_fit_diagnostic"),
]).sort_values("rmse", ignore_index=True)
fold_frame = pd.DataFrame(fold_reports)
lookup = summary.set_index("endpoint")
g3_rmse = float(lookup.loc["g3_reproduction", "rmse"])
gate_rmse = float(lookup.loc["frozen_consensus_crossfit", "rmse"])
reverse_loss = float(lookup.loc["reversed_consensus_control", "rmse"] - gate_rmse)
increment = g3_rmse - gate_rmse

deployment = {
    "base_endpoint": names[base_index],
    "profile": FROZEN_PROFILE,
    "active_endpoints": active_names,
    "active_endpoint_rms_scope": "each test well hidden suffix independently",
    "raw_factor_formula": "0.25 + 0.75 * abs(sum(delta_i/rms_i)) / sum(abs(delta_i/rms_i))",
    "normalizer": full_normalizer,
    "maximum_factor": float(K["MAX_NORMALISED_FACTOR"]),
    "factor_formula": "clip(raw_factor / normalizer, 0, maximum_factor)",
    "final_formula": "base + factor * sum(weight_i * (endpoint_i - base))",
    "weights": dict(zip(candidate_names, full_alpha)),
    "base_weight_before_gate": float(1.0 - full_alpha.sum()),
    "raw_full_objective_delta_mse": full_objective,
}
checks = {
    "g3_reproduction_abs_gap_le_1e_6": abs(g3_rmse - G3_EXPECTED_RMSE) <= 1e-6,
    "k1_reproduction_abs_gap_le_1e_9": abs(gate_rmse - K1_EXPECTED_RMSE) <= 1e-9,
    "increment_vs_g3_ge_0p015": increment >= 0.015,
    "all_folds_positive": bool((fold_frame["increment_vs_g3"] > 0.0).all()),
    "all_far_folds_positive": bool((fold_frame["far_increment_vs_g3"] > 0.0).all()),
    "reversed_gate_loses_ge_0p01": reverse_loss >= 0.01,
    "reversed_gate_loses_in_4of5": int((fold_frame["reverse_loss"] > 0.0).sum()) >= 4,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change_vs_g3"].max()) <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_consensus_rejection_attribution_k2",
    "stage": "frozen_zero_search_consensus_gate_attribution_and_deployment_audit",
    "selection_or_tuning_count": 0,
    "profile_source": "K1 selected magnitude_soft_mean1 independently in all five outer folds",
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "summary": summary.to_dict("records"),
    "folds": fold_reports,
    "deployment": deployment,
    "values": {"increment_vs_g3": increment, "reversed_gate_loss": reverse_loss},
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> generate one final Kaggle cell that gates the existing G5 move"
        if strict_pass else
        "FAIL -> K1 gain lacks directional attribution; do not deploy the gate"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
pd.DataFrame(fold_weights).to_csv(OUT / "fold_weights.csv", index=False)
(OUT / "deployment.json").write_text(
    json.dumps(_clean(deployment), indent=2), encoding="utf-8"
)
(OUT / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nM342-K2 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-K2 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-K2 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
