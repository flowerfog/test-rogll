"""M342-U1: multi-cut aggregated control-point endpoint after frozen K2.

Unlike P1/Q1, each physical station appears once: official/pseudo cut feature
states are averaged before fitting. Predictions are made only at those sparse
stations and linearly interpolated across the complete hidden suffix. This
tests a genuinely low-frequency endpoint rather than another full-row repair.
"""

from pathlib import Path
import gc
import json
import sys
import time
import types

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
Q1_SOURCE = ROOT / "main342_multicut_invariant_residual_q1.py"
OUT = ROOT / "oof_exports" / "main342_multicut_controlpoint_endpoint_u1"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
DOSES = (0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
MAX_P95_CHANGE = 0.20
STATION_SMOOTH = 17
CORRECTION_CAP_FT = 4.0


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
    return value


def _load_q1_definitions():
    source = Q1_SOURCE.read_text(encoding="utf-8")
    marker = 'base, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError("Q1 execution marker not found")
    name = "m342_u1_q1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(Q1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(Q1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


Q = _load_q1_definitions()
P = Q["P"]
B = Q["B"]
O = Q["O"]
P["OUT"] = OUT
P["MODEL_DIR"] = MODEL_DIR
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR


def _aggregate_stations(wells, manifest, base_features, endpoints):
    samples, state_counts = Q["_multicut_samples"](
        wells, manifest, base_features, endpoints
    )
    features = list(dict.fromkeys(base_features + list(P["ENGINEERED"])))
    identity = samples[["id", "well"]].drop_duplicates("id")
    if identity["id"].duplicated().any():
        raise RuntimeError("Ambiguous station identity")
    numeric = samples.groupby("id", sort=False)[features].mean()
    target = samples.groupby("id", sort=False).agg(
        residual_target=("residual_target", "first"),
        _global_weight=("_global_weight", "sum"),
        state_count=("_cut_tag", "nunique"),
    )
    output = identity.set_index("id").join(numeric).join(target).reset_index()
    if output["well"].nunique() != len(set(map(str, wells))):
        raise RuntimeError("Station aggregation lost wells")
    totals = output.groupby("well")["_global_weight"].transform("sum")
    output["_global_weight"] /= np.maximum(totals, 1e-12)
    output["row_idx"] = output["id"].str.rsplit("_", n=1).str[-1].astype(np.int64)
    return output.sort_values(["well", "row_idx"]).reset_index(drop=True), state_counts


def _held_full(wells, official_manifest, base, endpoints):
    frame = B["_load_full"](wells, official_manifest, base)
    frame = frame.merge(endpoints, on="id", how="inner", validate="one_to_one")
    if frame["well"].nunique() != len(wells):
        raise RuntimeError("Full held alignment lost wells")
    if "_row_idx" in frame:
        frame["row_idx_u1"] = frame["_row_idx"].astype(np.int64)
    elif "row_idx" in frame:
        frame["row_idx_u1"] = frame["row_idx"].astype(np.int64)
    else:
        frame["row_idx_u1"] = frame["id"].str.rsplit("_", n=1).str[-1].astype(np.int64)
    return frame.sort_values(["well", "row_idx_u1"]).reset_index(drop=True)


def _interpolated_correction(stations, full, model, features):
    raw = np.clip(
        model.predict(B["_matrix"](stations, features)),
        -CORRECTION_CAP_FT,
        CORRECTION_CAP_FT,
    )
    station_values = np.empty_like(raw)
    for positions in stations.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        size = min(STATION_SMOOTH, len(positions) | 1)
        station_values[positions] = uniform_filter1d(raw[positions], size=size, mode="nearest")
    correction = np.empty(len(full), np.float64)
    station_groups = stations.groupby("well", sort=False).indices
    for well, positions in full.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        station_positions = np.asarray(station_groups[str(well)], np.int64)
        x = stations.iloc[station_positions]["row_idx"].to_numpy(np.float64)
        y = station_values[station_positions]
        order = np.argsort(x)
        correction[positions] = np.interp(
            full.iloc[positions]["row_idx_u1"].to_numpy(np.float64),
            x[order], y[order], left=y[order][0], right=y[order][-1],
        )
    return np.clip(correction, -CORRECTION_CAP_FT, CORRECTION_CAP_FT)


def _scan(frame, correction):
    parent = frame["k2_parent"].to_numpy(np.float64)
    parent_metric = B["_metrics"](frame, parent, "k2_parent")
    rows = []
    for dose in DOSES:
        metric = B["_metrics"](frame, parent + float(dose) * correction, "controlpoint")
        rows.append({
            "dose": float(dose), "rmse": metric["rmse"],
            "gain": parent_metric["rmse"] - metric["rmse"],
            "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        })
    return parent_metric, pd.DataFrame(rows).sort_values("rmse", ignore_index=True)


base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()
features = list(dict.fromkeys(base_features + list(P["ENGINEERED"])))
official_manifest = manifest[manifest["tag"].eq("official")].copy()
official_manifest["cut_bin"] = 0
official_manifest["states_in_bin"] = 1

train_wells = sorted(base.loc[base["fold"].isin(TRAIN_FOLDS), "well"].unique())
selection_wells = sorted(base.loc[base["fold"].eq(SELECTION_FOLD), "well"].unique())
train, train_counts = _aggregate_stations(train_wells, manifest, base_features, endpoints)
valid, valid_counts = _aggregate_stations(selection_wells, manifest, base_features, endpoints)
selection_full = _held_full(selection_wells, official_manifest, base, endpoints)

print("M342-U1 CONFIG", {
    "coverage": coverage, "features": len(features),
    "train_stations": len(train), "selection_stations": len(valid),
    "train_three_state_wells": int((train_counts >= 3).sum()),
    "selection_three_state_wells": int((valid_counts >= 3).sum()),
}, flush=True)

model, best_iteration = P["_fit"](
    train, valid, features, label="selection_controlpoint"
)
selection_correction = _interpolated_correction(valid, selection_full, model, features)
selection_parent, scan = _scan(selection_full, selection_correction)
safe = scan.loc[
    scan["dose"].gt(0.0)
    & scan["gain"].ge(0.03)
    & scan["far_gain"].ge(0.0)
    & scan["p95_change"].le(MAX_P95_CHANGE)
].copy()
selected = (safe.iloc[0] if len(safe) else scan.iloc[0]).to_dict()
mechanism_pass = bool(len(safe) > 0)

confirmation = None
checks = {"selection_has_safe_controlpoint_dose": mechanism_pass}
if mechanism_pass:
    del train, valid, model
    gc.collect()
    train2_wells = sorted(base.loc[base["fold"].isin((3, 4, 5)), "well"].unique())
    fold2_wells = sorted(base.loc[base["fold"].eq(CONFIRMATION_FOLD), "well"].unique())
    train2, train2_counts = _aggregate_stations(train2_wells, manifest, base_features, endpoints)
    held2_stations, held2_counts = _aggregate_stations(fold2_wells, manifest, base_features, endpoints)
    held2_full = _held_full(fold2_wells, official_manifest, base, endpoints)
    model2, _ = P["_fit"](
        train2, None, features, iterations=best_iteration, label="confirmation_controlpoint"
    )
    correction2 = _interpolated_correction(held2_stations, held2_full, model2, features)
    parent2 = held2_full["k2_parent"].to_numpy(np.float64)
    dose = float(selected["dose"])
    candidate2 = parent2 + dose * correction2
    reverse2 = parent2 - dose * correction2
    base2 = B["_metrics"](held2_full, parent2, "k2_parent")
    metric2 = B["_metrics"](held2_full, candidate2, "controlpoint_endpoint")
    reverse_metric2 = B["_metrics"](held2_full, reverse2, "negative_direction_control")
    confirmation = {
        "base": base2, "candidate": metric2, "reverse": reverse_metric2,
        "gain": base2["rmse"] - metric2["rmse"],
        "far_gain": base2["far_gt2000_rmse"] - metric2["far_gt2000_rmse"],
        "reverse_loss": reverse_metric2["rmse"] - metric2["rmse"],
        "p95_change": metric2["p95_well_rmse"] - base2["p95_well_rmse"],
        "train_three_state_wells": int((train2_counts >= 3).sum()),
        "held_three_state_wells": int((held2_counts >= 3).sum()),
    }
    checks.update({
        "confirmation_gain_ge_0p02": confirmation["gain"] >= 0.02,
        "confirmation_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
        "reverse_loses_ge_0p03": confirmation["reverse_loss"] >= 0.03,
        "confirmation_p95_change_le_0p20": confirmation["p95_change"] <= MAX_P95_CHANGE,
    })

strict_pass = bool(confirmation is not None and all(checks.values()))
verdict = {
    "experiment": "main342_multicut_controlpoint_endpoint_u1",
    "stage": "fold45_multicut_station_training_fold3_selection_fold345_refit_fold2",
    "endpoint_type": "sparse multi-cut aggregated control points interpolated over full hidden suffix",
    "protocol": {
        "train_folds": TRAIN_FOLDS, "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD if mechanism_pass else None,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "coverage": coverage,
    "features": len(features),
    "best_iteration": best_iteration,
    "station_smooth": STATION_SMOOTH,
    "selected": {**selected, "safe_doses": safe["dose"].astype(float).tolist()},
    "confirmation": confirmation,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> confirm the frozen control-point endpoint once on Fold1"
        if strict_pass else
        "FAIL -> multi-cut control points do not create a transferable endpoint; close U1"
    ),
}

scan.to_csv(OUT / "selection_scan.csv", index=False)
(OUT / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-U1 SELECTION SCAN")
print(scan.to_string(index=False), flush=True)
print("\nM342-U1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
