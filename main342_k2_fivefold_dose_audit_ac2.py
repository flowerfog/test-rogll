"""M342-AC2: complete-fold audit of the frozen K2 endpoint and move scale."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_k2_fivefold_dose_audit_ac2"
FOLDS = (1, 2, 3, 4, 5)
SCALES = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0, 1.25], np.float64)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError(f"Cannot locate P1 runtime marker in {path}")
    name = "main342_p1_definitions_ac2"
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
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
OUT.mkdir(parents=True, exist_ok=True)
P = _bootstrap(SOURCE)
O = P["O"]
B = P["B"]
base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()[["id", "k2_parent", "k2_move", "tvt_true_endpoint"]].copy()
endpoints["id"] = endpoints["id"].astype(str)
base["id"] = base["id"].astype(str)
frame = base.merge(endpoints, on="id", how="inner", validate="one_to_one", suffixes=("", "_endpoint"))
if len(frame) != len(base) or frame["well"].nunique() != 773:
    raise RuntimeError("K2 audit alignment is not complete 773-well OOF")
truth = frame["tvt_true"].to_numpy(np.float64)
parent = frame["k2_parent"].to_numpy(np.float64)
move = frame["k2_move"].to_numpy(np.float64)
if float(np.max(np.abs(parent - (frame["k2_parent"].to_numpy(np.float64))))) > 1e-12:
    raise RuntimeError("K2 parent reconstruction failed")

def metrics(pred, name):
    return B["_metrics"](frame, np.asarray(pred, np.float64), name)

records = []
for scale in SCALES:
    pred = parent - move + float(scale) * move
    metric = metrics(pred, f"k2_move_scale_{scale:g}")
    records.append({"scale": float(scale), **metric})
summary = pd.DataFrame(records)
fold_rows = []
for fold in FOLDS:
    local = frame[frame["fold"].eq(fold)].reset_index(drop=True)
    local_truth = local["tvt_true"].to_numpy(np.float64)
    local_parent = local["k2_parent"].to_numpy(np.float64)
    local_move = local["k2_move"].to_numpy(np.float64)
    for scale in SCALES:
        pred = local_parent - local_move + float(scale) * local_move
        metric = B["_metrics"](local, pred, f"fold{fold}_scale_{scale:g}")
        fold_rows.append({"fold": fold, "scale": float(scale), **metric})
fold_summary = pd.DataFrame(fold_rows)
summary.to_csv(OUT / "pooled_scan.csv", index=False)
fold_summary.to_csv(OUT / "fold_scan.csv", index=False)
best = summary.sort_values(["rmse", "scale"]).iloc[0]
best_folds = fold_summary[fold_summary["scale"].eq(float(best["scale"]))]
checks = {
    "rows_exact": len(frame) == 3783989,
    "wells_exact": frame["well"].nunique() == 773,
    "each_fold_present": all((fold_summary["fold"] == fold).any() for fold in FOLDS),
    "best_scale_positive": float(best["scale"]) > 0.0,
    "best_gain_ge_0p03": float(summary.loc[summary["scale"].eq(1.0), "rmse"].iloc[0] - best["rmse"]) >= 0.03,
    "minimum_best_fold_gain_ge_0": bool((best_folds["rmse"] < fold_summary[fold_summary["scale"].eq(0.0)]["rmse"].to_numpy()).all()),
}
verdict = {
    "experiment": "main342_k2_fivefold_dose_audit_ac2",
    "stage": "complete_773well_frozen_k2_move_scale_audit",
    "selection_or_tuning_count": 0,
    "coverage": {str(k): int(v) for k, v in coverage.items()},
    "pooled": summary.to_dict("records"),
    "best": best.to_dict(),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> test one new K2-compatible information source" if all(checks.values()) else "FAIL -> freeze K2 exactly; do not retune dose",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AC2 POOLED", flush=True)
print(summary.to_string(index=False), flush=True)
print("M342-AC2 FOLDS", flush=True)
print(fold_summary.groupby("scale")["rmse"].agg(["min", "max", "mean"]).reset_index().to_string(index=False), flush=True)
print("M342-AC2 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
