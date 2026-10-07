"""M342-AF2: exact full-bank K2 reproduction and candidate ablation."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_k2_fullbank_ablation_af2"
FOLDS = (1, 2, 3, 4, 5)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_af2"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _metric(frame, prediction, name):
    return B["_metrics"](frame.reset_index(drop=True), prediction, name)


started = time.time()
OUT.mkdir(parents=True, exist_ok=True)
P = _bootstrap(SOURCE)
B = P["B"]
O = P["O"]
K = P["K"]
G = K["_load_g1"]()
V = G["_load_inventory"]()
frame = V["base"].reset_index(drop=True)
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
base_error = base_prediction - truth
folds = frame["fold"].to_numpy(np.int8)
candidate_indices = [i for i in range(len(names)) if i != base_index]
candidate_names = [names[i] for i in candidate_indices]
all_delta = paths[candidate_indices] - base_prediction[None, :]
active_names = list(K["ACTIVE_ENDPOINTS"])
active_indices = [names.index(name) for name in active_names]
active_delta = paths[active_indices] - base_prediction[None, :]
raw_gate = K["_raw_gates"](active_delta, frame)["magnitude_soft_mean1"]
max_factor = float(K["MAX_NORMALISED_FACTOR"])
expected = P["_endpoint_table"]()[["id", "k2_parent"]].copy()
expected["id"] = expected["id"].astype(str)
frame["id"] = frame["id"].astype(str)
expected_parent = frame["id"].map(expected.set_index("id")["k2_parent"]).to_numpy(np.float64)
if not np.isfinite(expected_parent).all():
    raise RuntimeError("Missing expected K2 parent rows")

rows = []
gaps = []
for held_fold in FOLDS:
    held = folds == held_fold
    train = ~held
    factor = np.clip(raw_gate[held] / max(float(np.mean(raw_gate[train])), 1e-6), 0.0, max_factor)
    alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
    current_move = 0.20 * factor * (alpha @ all_delta[:, held])
    current = base_prediction[held] + current_move
    gaps.append(float(np.max(np.abs(current - expected_parent[held]))))
    local = frame.loc[held].reset_index(drop=True)
    parent_metric = _metric(local, current, "reconstructed_k2")
    for removed_name in ["none"] + active_names:
        if removed_name == "none":
            prediction = current
        else:
            removed_index = candidate_names.index(removed_name)
            keep = np.ones(len(candidate_names), dtype=bool)
            keep[removed_index] = False
            ablated_delta = all_delta[keep]
            ablated_names = [name for i, name in enumerate(candidate_names) if keep[i]]
            active_keep = [name for name in active_names if name != removed_name]
            active_delta_ablated = paths[[names.index(name) for name in active_keep]] - base_prediction[None, :]
            gate_ablated = K["_raw_gates"](active_delta_ablated, frame)["magnitude_soft_mean1"]
            factor_ablated = np.clip(
                gate_ablated[held] / max(float(np.mean(gate_ablated[train])), 1e-6),
                0.0,
                max_factor,
            )
            ablated_alpha, _ = G["_fit_coefficients"](ablated_delta, base_error, train)
            prediction = base_prediction[held] + 0.20 * factor_ablated * (ablated_alpha @ ablated_delta[:, held])
        metric = _metric(local, prediction, f"removed_{removed_name}")
        rows.append({
            "held_fold": held_fold,
            "removed": removed_name,
            "rows": len(local),
            "wells": local["well"].nunique(),
            "parent_rmse": parent_metric["rmse"],
            "rmse": metric["rmse"],
            "gain_vs_k2": parent_metric["rmse"] - metric["rmse"],
            "far_gain_vs_k2": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change_vs_k2": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        })

result = pd.DataFrame(rows)
result.to_csv(OUT / "fullbank_ablation.csv", index=False)
pooled = result.groupby("removed", as_index=False).agg(
    mean_gain=("gain_vs_k2", "mean"),
    minimum_fold_gain=("gain_vs_k2", "min"),
    mean_far_gain=("far_gain_vs_k2", "mean"),
)
checks = {
    "rows_exact": len(frame) == 3783989,
    "wells_exact": frame["well"].nunique() == 773,
    "active_names_present": all(name in names for name in active_names),
    "fullbank_size_ge_5": len(candidate_names) >= 5,
    "k2_reproduction_supported": bool(np.isfinite(result.loc[result["removed"].eq("none"), "rmse"]).all()),
    "k2_reproduction_max_gap_le_1e_9": max(gaps) <= 1e-9,
}
verdict = {
    "experiment": "main342_k2_fullbank_ablation_af2",
    "stage": "exact_v187_full_candidate_bank_k2_leave_one_active_ablation",
    "selection_or_tuning_count": 0,
    "candidate_bank_size": len(candidate_names),
    "candidate_names": candidate_names,
    "active_names": active_names,
    "coverage": {str(k): int(v) for k, v in V["base"].groupby("fold")["well"].nunique().to_dict().items()},
    "ablation": result.to_dict("records"),
    "pooled": pooled.to_dict("records"),
    "checks": checks,
    "reproduction_gaps": gaps,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> interpret active candidate contribution" if all(checks.values()) else "FAIL -> repair full-bank alignment before interpretation",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AF2 ABLATION", flush=True)
print(result.to_string(index=False), flush=True)
print("M342-AF2 POOLED", pooled.to_string(index=False), flush=True)
print("M342-AF2 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
