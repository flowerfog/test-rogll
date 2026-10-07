"""M342-AD1: cross-fold K2 move datum/slope/shape component attribution."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_k2_move_component_ad1"
DISCOVERY_FOLDS = (4, 5)
EVALUATION_FOLDS = (3, 2, 1)
DOSES = np.asarray([0.25, 0.50, 0.75, 1.0], np.float64)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_ad1"
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


def _decompose(frame):
    move = frame["k2_move"].to_numpy(np.float64)
    md = frame["md_since"].to_numpy(np.float64)
    datum = np.zeros_like(move)
    slope = np.zeros_like(move)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        y = move[positions]
        x = md[positions]
        center = float(np.mean(x))
        xc = x - center
        denom = float(np.dot(xc, xc))
        local_datum = float(np.mean(y))
        local_slope = float(np.dot(xc, y - local_datum) / denom) if denom > 1e-9 else 0.0
        datum[positions] = local_datum
        slope[positions] = local_slope * xc
    shape = move - datum - slope
    return {
        "full": move,
        "datum": datum,
        "slope": slope,
        "shape": shape,
        "datum_slope": datum + slope,
        "datum_shape": datum + shape,
        "slope_shape": slope + shape,
    }


def _score(frame, parent, increment, endpoint):
    parent_metric = B["_metrics"](frame, parent, "current_k2")
    metric = B["_metrics"](frame, parent + increment, endpoint)
    return {
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "base_rmse": parent_metric["rmse"],
        "rmse": metric["rmse"],
        "gain": parent_metric["rmse"] - metric["rmse"],
        "base_far_rmse": parent_metric["far_gt2000_rmse"],
        "far_rmse": metric["far_gt2000_rmse"],
        "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
    }


started = time.time()
OUT.mkdir(parents=True, exist_ok=True)
P = _bootstrap(SOURCE)
B = P["B"]
O = P["O"]
base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()[["id", "k2_parent", "k2_move"]].copy()
base["id"] = base["id"].astype(str)
endpoints["id"] = endpoints["id"].astype(str)
frame = base.merge(endpoints, on="id", how="inner", validate="one_to_one")
if len(frame) != 3783989 or frame["well"].nunique() != 773:
    raise RuntimeError("AD1 requires exact complete K2 alignment")
components = _decompose(frame)

discovery_rows = []
for name, values in components.items():
    for dose in DOSES:
        fold_scores = []
        for fold in DISCOVERY_FOLDS:
            mask = frame["fold"].to_numpy(np.int8) == fold
            local = frame.loc[mask].reset_index(drop=True)
            score = _score(
                local,
                frame.loc[mask, "k2_parent"].to_numpy(np.float64),
                float(dose) * values[mask],
                f"{name}_{dose:g}",
            )
            fold_scores.append(score)
        pooled_mask = frame["fold"].isin(DISCOVERY_FOLDS).to_numpy()
        pooled = frame.loc[pooled_mask].reset_index(drop=True)
        pooled_score = _score(
            pooled,
            frame.loc[pooled_mask, "k2_parent"].to_numpy(np.float64),
            float(dose) * values[pooled_mask],
            f"{name}_{dose:g}",
        )
        discovery_rows.append({
            "component": name,
            "dose": float(dose),
            "pooled_rmse": pooled_score["rmse"],
            "pooled_gain": pooled_score["gain"],
            "minimum_fold_gain": min(item["gain"] for item in fold_scores),
            "minimum_fold_far_gain": min(item["far_gain"] for item in fold_scores),
            "maximum_fold_p95_change": max(item["p95_change"] for item in fold_scores),
            "fold4_gain": fold_scores[0]["gain"],
            "fold5_gain": fold_scores[1]["gain"],
        })
discovery = pd.DataFrame(discovery_rows)
safe = discovery[
    (discovery["minimum_fold_gain"] >= 0.0)
    & (discovery["minimum_fold_far_gain"] >= 0.0)
    & (discovery["maximum_fold_p95_change"] <= 0.20)
]
selected = (
    safe.sort_values(["pooled_rmse", "dose"]).iloc[0]
    if not safe.empty else discovery.sort_values(["pooled_rmse", "dose"]).iloc[0]
)
selected_component = str(selected["component"])
selected_dose = float(selected["dose"])

evaluation_rows = []
for fold in EVALUATION_FOLDS:
    mask = frame["fold"].to_numpy(np.int8) == fold
    local = frame.loc[mask].reset_index(drop=True)
    score = _score(
        local,
        frame.loc[mask, "k2_parent"].to_numpy(np.float64),
        selected_dose * components[selected_component][mask],
        f"{selected_component}_{selected_dose:g}",
    )
    evaluation_rows.append({"fold": fold, **score})
evaluation = pd.DataFrame(evaluation_rows)
discovery.to_csv(OUT / "discovery_scan.csv", index=False)
evaluation.to_csv(OUT / "evaluation_folds.csv", index=False)

checks = {
    "discovery_has_safe_component": not safe.empty,
    "fold3_gain_ge_0p01": float(evaluation.loc[evaluation["fold"].eq(3), "gain"].iloc[0]) >= 0.01,
    "fold2_gain_ge_0": float(evaluation.loc[evaluation["fold"].eq(2), "gain"].iloc[0]) >= 0.0,
    "fold1_gain_ge_0": float(evaluation.loc[evaluation["fold"].eq(1), "gain"].iloc[0]) >= 0.0,
    "all_far_gains_ge_0": bool((evaluation["far_gain"] >= 0.0).all()),
    "all_p95_changes_le_0p20": bool((evaluation["p95_change"] <= 0.20).all()),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_k2_move_component_ad1",
    "stage": "fold45_component_selection_fold3_fold2_fold1_frozen_transfer",
    "training_count": 0,
    "components": list(components),
    "coverage": {str(k): int(v) for k, v in coverage.items()},
    "selected": selected.to_dict(),
    "evaluation": evaluation.to_dict("records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> build component-only K2 deployment delta" if strict_pass else "FAIL -> K2 move components are not fold-stable; close AD1",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AD1 DISCOVERY", flush=True)
print(discovery.sort_values("pooled_rmse").head(20).to_string(index=False), flush=True)
print("M342-AD1 EVALUATION", flush=True)
print(evaluation.to_string(index=False), flush=True)
print("M342-AD1 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
