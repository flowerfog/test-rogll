"""M342-AC3: nested transfer audit for a fixed K2 move multiplier."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_k2_dose_transfer_ac3"
DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
APPLICATION_FOLD = 2
SCALES = np.round(np.arange(0.50, 2.01, 0.125), 3)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_ac3"
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
base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()[["id", "k2_parent", "k2_move"]].copy()
base["id"] = base["id"].astype(str)
endpoints["id"] = endpoints["id"].astype(str)
frame = base.merge(endpoints, on="id", how="inner", validate="one_to_one", suffixes=("", "_endpoint"))
if len(frame) != 3783989 or frame["well"].nunique() != 773:
    raise RuntimeError("AC3 requires exact 773-well K2 alignment")

rows = []
for held_fold in (CONFIRMATION_FOLD, APPLICATION_FOLD):
    donor = frame[frame["fold"].isin(DISCOVERY_FOLDS)].reset_index(drop=True)
    held = frame[frame["fold"].eq(held_fold)].reset_index(drop=True)
    donor_scores = []
    for scale in SCALES:
        prediction = donor["k2_parent"].to_numpy(np.float64) - donor["k2_move"].to_numpy(np.float64)
        prediction += float(scale) * donor["k2_move"].to_numpy(np.float64)
        metric = B["_metrics"](donor, prediction, f"discovery_{scale:g}")
        donor_scores.append({"held_fold": held_fold, "scale": float(scale), **metric})
    donor_table = pd.DataFrame(donor_scores)
    selected = donor_table.sort_values(["rmse", "scale"]).iloc[0]
    fixed_scale = float(selected["scale"])
    parent = held["k2_parent"].to_numpy(np.float64)
    move = held["k2_move"].to_numpy(np.float64)
    fixed = parent - move + fixed_scale * move
    current = parent
    fixed_metric = B["_metrics"](held, fixed, f"fixed_{fixed_scale:g}")
    current_metric = B["_metrics"](held, current, "current_k2")
    rows.append({
        "held_fold": held_fold,
        "donor_rows": len(donor),
        "donor_wells": donor["well"].nunique(),
        "held_rows": len(held),
        "held_wells": held["well"].nunique(),
        "selected_scale": fixed_scale,
        "donor_rmse": float(selected["rmse"]),
        "current_rmse": current_metric["rmse"],
        "fixed_rmse": fixed_metric["rmse"],
        "gain_vs_current": current_metric["rmse"] - fixed_metric["rmse"],
        "far_gain_vs_current": current_metric["far_gt2000_rmse"] - fixed_metric["far_gt2000_rmse"],
        "p95_change": fixed_metric["p95_well_rmse"] - current_metric["p95_well_rmse"],
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "fold_transfer.csv", index=False)
checks = {
    "confirmation_wells_ge_145": int(result.loc[result["held_fold"].eq(3), "held_wells"].iloc[0]) >= 145,
    "application_wells_ge_145": int(result.loc[result["held_fold"].eq(2), "held_wells"].iloc[0]) >= 145,
    "confirmation_gain_ge_0p03": float(result.loc[result["held_fold"].eq(3), "gain_vs_current"].iloc[0]) >= 0.03,
    "confirmation_far_gain_ge_0": float(result.loc[result["held_fold"].eq(3), "far_gain_vs_current"].iloc[0]) >= 0.0,
    "confirmation_p95_change_le_0p20": float(result.loc[result["held_fold"].eq(3), "p95_change"].iloc[0]) <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_k2_dose_transfer_ac3",
    "stage": "fold45_discovery_fixed_multiplier_fold3_confirmation_fold2_application_support",
    "selection_or_tuning_count": 1,
    "discovery_folds": list(DISCOVERY_FOLDS),
    "coverage": {str(k): int(v) for k, v in coverage.items()},
    "folds": result.to_dict("records"),
    "scale_menu": SCALES.tolist(),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> use frozen multiplier for deployment audit" if strict_pass else "FAIL -> retain current K2 multiplier; close AC3",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AC3 FOLDS", flush=True)
print(result.to_string(index=False), flush=True)
print("M342-AC3 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
