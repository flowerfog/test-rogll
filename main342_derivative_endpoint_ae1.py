"""M342-AE1: derivative-domain trajectory endpoint after frozen K2."""

from pathlib import Path
import importlib.util
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_derivative_endpoint_ae1"
MODEL_DIR = OUT / "models"
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
DOSES = np.asarray([0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20], np.float64)
SLOPE_CLIP = 0.20
TARGET_SMOOTH_ROWS = 17
PREDICTION_SMOOTH_ROWS = 33


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    name = "main342_p1_definitions_ae1"
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


def _attach_slope_target(frame):
    output = frame.copy()
    output["slope_target"] = np.nan
    keys = ["well", "_cut_tag"] if "_cut_tag" in output else ["well"]
    for _, positions in output.groupby(keys, sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        x = output.loc[positions, "md_since"].to_numpy(np.float64)
        y = (
            output.loc[positions, "target"].to_numpy(np.float64)
            + output.loc[positions, "last_known_tvt"].to_numpy(np.float64)
        )
        order = np.argsort(x, kind="stable")
        xs = x[order]
        ys = y[order]
        if len(xs) < 5:
            continue
        # Duplicate sampled depths carry no derivative information.
        for index in range(1, len(xs)):
            if xs[index] <= xs[index - 1]:
                xs[index] = xs[index - 1] + 1e-6
        slope = np.gradient(ys, xs)
        slope = uniform_filter1d(slope, size=min(TARGET_SMOOTH_ROWS, len(slope)), mode="nearest")
        slope = np.clip(slope, -SLOPE_CLIP, SLOPE_CLIP)
        output.loc[positions[order], "slope_target"] = slope
    return output[np.isfinite(output["slope_target"])].reset_index(drop=True)


def _fit(train, valid, features):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(B["LGB_PARAMS"])
    params.update({
        "n_estimators": 2400,
        "learning_rate": 0.025,
        "num_leaves": 48,
        "min_child_samples": 100,
        "reg_lambda": 20.0,
    })
    model = LGBMRegressor(**params)
    model.fit(
        B["_matrix"](train, features),
        train["slope_target"].to_numpy(np.float64),
        sample_weight=train["_global_weight"].to_numpy(np.float64),
        eval_set=[(
            B["_matrix"](valid, features),
            valid["slope_target"].to_numpy(np.float64),
        )],
        eval_metric="rmse",
        callbacks=[early_stopping(160, verbose=False), log_evaluation(100)],
    )
    joblib.dump(model, MODEL_DIR / "derivative_lgb_fold45.joblib")
    best = getattr(model, "best_iteration_", None)
    return model, int(best if best and best > 0 else params["n_estimators"])


def _integrate(frame, slope):
    slope = np.clip(np.asarray(slope, np.float64), -SLOPE_CLIP, SLOPE_CLIP)
    output = np.empty(len(frame), np.float64)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        x = frame.loc[positions, "md_since"].to_numpy(np.float64)
        anchor = frame.loc[positions, "last_known_tvt"].to_numpy(np.float64)
        order = np.argsort(x, kind="stable")
        ps = uniform_filter1d(
            slope[positions][order], size=min(PREDICTION_SMOOTH_ROWS, len(order)), mode="nearest"
        )
        xs = x[order]
        increments = np.empty(len(order), np.float64)
        increments[0] = ps[0] * max(xs[0], 0.0)
        if len(order) > 1:
            increments[1:] = 0.5 * (ps[1:] + ps[:-1]) * np.maximum(np.diff(xs), 0.0)
        integrated = float(np.median(anchor)) + np.cumsum(increments)
        output[positions[order]] = integrated
    return output


def _metric(frame, prediction, endpoint):
    return B["_metrics"](frame.reset_index(drop=True), prediction, endpoint)


def _roll(frame, values):
    output = np.asarray(values, np.float64).copy()
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        if len(positions) > 1:
            output[positions] = np.roll(output[positions], max(1, len(positions) // 3))
    return output


started = time.time()
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
P = _bootstrap(SOURCE)
B = P["B"]
O = P["O"]
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR
base, manifest, features, coverage = O["_prepare_inventory"]()
features = list(features)
endpoints = P["_endpoint_table"]()[["id", "k2_parent"]].copy()
endpoints["id"] = endpoints["id"].astype(str)

train_wells = sorted(base.loc[base["fold"].isin(TRAIN_FOLDS), "well"].astype(str).unique())
held_wells = sorted(base.loc[base["fold"].eq(SELECTION_FOLD), "well"].astype(str).unique())
train_manifest = manifest[manifest["well"].isin(train_wells)].copy()
held_manifest = manifest[manifest["well"].isin(held_wells)].copy()
for local_manifest in (train_manifest, held_manifest):
    local_manifest["cut_bin"] = 0
    local_manifest["states_in_bin"] = 3

train = _attach_slope_target(B["_load_samples"](train_wells, train_manifest))
valid = _attach_slope_target(B["_load_samples"](held_wells, held_manifest))
official = manifest[manifest["tag"].eq("official")].copy()
held = B["_load_full"](held_wells, official, base)
held = held.merge(endpoints, on="id", how="inner", validate="one_to_one").reset_index(drop=True)
if held["well"].nunique() < 145:
    raise RuntimeError(f"Only {held['well'].nunique()} held wells")

print("M342-AE1 CONFIG", {
    "coverage": coverage,
    "features": len(features),
    "train_rows": len(train),
    "valid_rows": len(valid),
    "held_rows": len(held),
}, flush=True)
model, iteration = _fit(train, valid, features)
raw_slope = model.predict(B["_matrix"](held, features))
endpoint = _integrate(held, raw_slope)
parent = held["k2_parent"].to_numpy(np.float64)
parent_metric = _metric(held, parent, "current_k2")
endpoint_metric = _metric(held, endpoint, "derivative_endpoint")
scan_rows = []
for dose in DOSES:
    prediction = parent + float(dose) * (endpoint - parent)
    metric = _metric(held, prediction, "derivative_blend")
    scan_rows.append({
        "dose": float(dose),
        "rmse": metric["rmse"],
        "gain": parent_metric["rmse"] - metric["rmse"],
        "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
    })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "dose"], ignore_index=True)
safe = scan[(scan["far_gain"] >= 0.0) & (scan["p95_change"] <= 0.20)]
selected = safe.iloc[0] if not safe.empty else scan.iloc[0]
dose = float(selected["dose"])
move = dose * (endpoint - parent)
prediction = parent + move
reverse = parent - move
rolled = parent + _roll(held, move)
candidate_metric = _metric(held, prediction, "derivative_blend")
reverse_metric = _metric(held, reverse, "reverse_control")
roll_metric = _metric(held, rolled, "depth_roll_control")
scan.to_csv(OUT / "dose_scan.csv", index=False)

checks = {
    "fold3_wells_ge_145": held["well"].nunique() >= 145,
    "selected_positive_dose": dose > 0.0,
    "gain_ge_0p03": float(selected["gain"]) >= 0.03,
    "far_gain_ge_0": float(selected["far_gain"]) >= 0.0,
    "p95_change_le_0p20": float(selected["p95_change"]) <= 0.20,
    "reverse_loses_ge_0p03": reverse_metric["rmse"] - candidate_metric["rmse"] >= 0.03,
    "depth_roll_loses_ge_0p01": roll_metric["rmse"] - candidate_metric["rmse"] >= 0.01,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_derivative_endpoint_ae1",
    "stage": "fold45_derivative_training_fold3_k2_compatibility_gateway",
    "legal_inputs_only": 1,
    "target": "smoothed clipped dTVT/dMD",
    "protocol": {"train_folds": list(TRAIN_FOLDS), "selection_fold": 3, "fold2_touched": 0, "fold1_touched": 0},
    "coverage": coverage,
    "model": {"features": len(features), "best_iteration": iteration},
    "parent": parent_metric,
    "endpoint": endpoint_metric,
    "selected": selected.to_dict(),
    "candidate": candidate_metric,
    "controls": {
        "reverse": reverse_metric,
        "depth_roll": roll_metric,
        "reverse_loss": reverse_metric["rmse"] - candidate_metric["rmse"],
        "roll_loss": roll_metric["rmse"] - candidate_metric["rmse"],
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "PASS -> freeze derivative recipe for Fold2" if strict_pass else "FAIL -> derivative domain adds no stable information; close AE1",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AE1 DOSE SCAN", flush=True)
print(scan.to_string(index=False), flush=True)
print("M342-AE1 FINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
