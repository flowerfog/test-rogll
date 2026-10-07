"""M342-AF1: read-only fivefold K2 candidate-contribution audit."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_k2_candidate_ablation_af1"
FOLDS = (1, 2, 3, 4, 5)
CANDIDATES = {
    "coarse": "coarse_delta",
    "psg": "psg_delta",
    "neighbor": "neighbor_delta",
}
SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_af1"
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


started = time.time()
OUT.mkdir(parents=True, exist_ok=True)
P = _bootstrap(SOURCE)
B = P["B"]
O = P["O"]
G = P["K"]["_load_g1"]()
base, manifest, base_features, coverage = O["_prepare_inventory"]()
table = P["_endpoint_table"]()
required = {"id", "k2_parent", "k2_move", "tvt_true_endpoint", "raw_gate_factor", *CANDIDATES.values()}
missing = sorted(required.difference(table.columns))
if missing:
    raise RuntimeError(f"K2 endpoint table missing {missing}")
table["id"] = table["id"].astype(str)
base["id"] = base["id"].astype(str)
frame = base.merge(table, on="id", how="inner", validate="one_to_one", suffixes=("", "_endpoint"))
if len(frame) != 3783989 or frame["well"].nunique() != 773:
    raise RuntimeError("AF1 requires exact complete OOF alignment")
truth = frame["tvt_true"].to_numpy(np.float64)
base_prediction = frame["k2_parent"].to_numpy(np.float64) - frame["k2_move"].to_numpy(np.float64)
base_error = base_prediction - truth
all_delta = frame[list(CANDIDATES.values())].to_numpy(np.float64).T
raw_gate = frame["raw_gate_factor"].to_numpy(np.float64)
max_factor = float(P["K"]["MAX_NORMALISED_FACTOR"])

rows = []
reproduction_gaps = []
for held_fold in FOLDS:
    held_mask = frame["fold"].to_numpy(np.int8) == held_fold
    train_mask = ~held_mask
    local = frame.loc[held_mask].reset_index(drop=True)
    local_parent = frame.loc[held_mask, "k2_parent"].to_numpy(np.float64)
    local_truth = frame.loc[held_mask, "tvt_true"].to_numpy(np.float64)
    full_alpha, _ = G["_fit_coefficients"](all_delta, base_error, train_mask)
    gate = np.clip(
        raw_gate[held_mask] / max(float(np.mean(raw_gate[train_mask])), 1e-6),
        0.0,
        max_factor,
    )
    full_move = 0.2 * gate * (full_alpha @ all_delta[:, held_mask])
    reproduction_gaps.append(float(np.max(np.abs(full_move - frame.loc[held_mask, "k2_move"].to_numpy(np.float64)))))
    for removed, column in CANDIDATES.items():
        keep = [name for name in CANDIDATES if name != removed]
        delta = frame[[CANDIDATES[name] for name in keep]].to_numpy(np.float64).T
        alpha, _ = G["_fit_coefficients"](delta, base_error, train_mask)
        move = 0.2 * gate * (alpha @ delta[:, held_mask])
        endpoint = local_parent + move
        parent_rmse = float(np.sqrt(np.mean(np.square(local_parent - local_truth))))
        endpoint_rmse = float(np.sqrt(np.mean(np.square(endpoint - local_truth))))
        rows.append({
            "held_fold": held_fold,
            "removed": removed,
            "remaining": "+".join(keep),
            "rows": len(local),
            "wells": local["well"].nunique(),
            "parent_rmse": parent_rmse,
            "rmse": endpoint_rmse,
            "gain_vs_parent": parent_rmse - endpoint_rmse,
            "move_rms": float(np.sqrt(np.mean(np.square(move)))),
        })

result = pd.DataFrame(rows)
result.to_csv(OUT / "candidate_ablation.csv", index=False)
pooled = result.groupby("removed", as_index=False).agg(
    pooled_gain=("gain_vs_parent", "mean"),
    minimum_fold_gain=("gain_vs_parent", "min"),
    mean_move_rms=("move_rms", "mean"),
)
checks = {
    "rows_exact": len(frame) == 3783989,
    "wells_exact": frame["well"].nunique() == 773,
    "k2_reproduction_max_gap_le_1e_9": max(reproduction_gaps) <= 1e-9,
    "some_candidate_ablation_positive_all_folds": bool((pooled["minimum_fold_gain"] > 0.0).any()),
}
verdict = {
    "experiment": "main342_k2_candidate_ablation_af1",
    "stage": "read_only_complete_fivefold_k2_leave_one_candidate_audit",
    "selection_or_tuning_count": 0,
    "coverage": {str(k): int(v) for k, v in coverage.items()},
    "reproduction_gaps": reproduction_gaps,
    "ablation": result.to_dict("records"),
    "pooled": pooled.to_dict("records"),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> test only the candidate with positive fold-stable ablation" if all(checks.values()) else "FAIL -> no K2 candidate has fold-stable independent contribution; close AF1",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AF1 ABLATION", flush=True)
print(result.to_string(index=False), flush=True)
print("M342-AF1 POOLED", pooled.to_string(index=False), flush=True)
print("M342-AF1 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
