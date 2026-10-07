"""M342-P1: strongly regularised residual GBDT on the frozen K2 parent.

The model predicts only truth minus K2, using legal K3 row features plus
truth-free endpoint disagreement features. Fold4+5 train, Fold3 selects a
fixed dose, and Fold2 is loaded only after a mechanism pass. Fold1 is untouched.
"""

from pathlib import Path
import gc
import json
import os
import sys
import time
import types

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
O1_SOURCE = ROOT / "main342_robust_objective_gbdt_o1.py"
OUT = ROOT / "oof_exports" / "main342_k2_residual_gbdt_p1"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
DOSES = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], np.float64)
CORRECTION_CAP_FT = 4.0
SMOOTH_ROWS = 129
SEED = 20260804
ENGINEERED = (
    "k2_move", "coarse_delta", "psg_delta", "neighbor_delta",
    "delta_median", "delta_spread", "agreement", "raw_gate_factor",
)


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


def _load_o1():
    source = O1_SOURCE.read_text(encoding="utf-8")
    marker = "base, manifest, features, coverage = _prepare_inventory()"
    if marker not in source:
        raise RuntimeError("O1 execution marker not found")
    name = "m342_p1_o1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(O1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(O1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


O = _load_o1()
B = O["B"]
K = O["K"]
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR


def _endpoint_table():
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
    active_indices = [names.index(name) for name in K["ACTIVE_ENDPOINTS"]]
    active = paths[active_indices]
    delta = active - base_prediction[None, :]
    raw_gate = K["_raw_gates"](delta, frame)["magnitude_soft_mean1"]
    candidate_indices = [index for index in range(len(names)) if index != base_index]
    all_delta = paths[candidate_indices] - base_prediction[None, :]
    k2 = base_prediction.copy()
    for held in range(1, 6):
        train = folds != held
        valid = folds == held
        alpha, _ = G["_fit_coefficients"](all_delta, base_error, train)
        move = 0.20 * (alpha @ all_delta[:, valid])
        factor = np.clip(
            raw_gate[valid] / max(float(np.mean(raw_gate[train])), 1e-6),
            0.0, float(K["MAX_NORMALISED_FACTOR"]),
        )
        k2[valid] += factor * move
    z = np.empty_like(delta)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        local = delta[:, positions]
        rms = np.maximum(np.sqrt(np.mean(np.square(local), axis=1)), 1e-6)
        z[:, positions] = local / rms[:, None]
    agreement = np.abs(np.sum(z, axis=0)) / np.maximum(np.sum(np.abs(z), axis=0), 1e-12)
    table = pd.DataFrame({
        "id": frame["id"].astype(str),
        "tvt_true_endpoint": truth,
        "k2_parent": k2,
        "k2_move": k2 - base_prediction,
        "coarse_delta": delta[0],
        "psg_delta": delta[1],
        "neighbor_delta": delta[2],
        "delta_median": np.median(delta, axis=0),
        "delta_spread": np.std(delta, axis=0),
        "agreement": agreement,
        "raw_gate_factor": raw_gate,
    })
    metric = B["_metrics"](frame, k2, "k2")
    if abs(metric["rmse"] - O["K2_EXPECTED_RMSE"]) > 1e-9:
        raise RuntimeError(f"K2 reproduction gap: {metric['rmse']}")
    return table


def _official_samples(wells, manifest, base_features):
    local = manifest[
        manifest["tag"].eq("official") & manifest["well"].isin(set(map(str, wells)))
    ].copy()
    local["cut_bin"] = 0
    local["states_in_bin"] = 1
    identity_columns = ["id", "_row_idx", "row_idx", "last_known_tvt"]
    requested = list(dict.fromkeys(list(base_features) + identity_columns))
    frame = B["_load_samples"](wells, local, requested)
    id_missing = "id" not in frame or frame["id"].isna().all()
    if id_missing:
        row_column = next(
            (
                column for column in ("_row_idx", "row_idx")
                if column in frame and frame[column].notna().all()
            ),
            None,
        )
        if row_column is None:
            raise RuntimeError(
                "Official sample cache has neither usable id nor row index; "
                f"columns={frame.columns.tolist()[:40]}"
            )
        frame["id"] = (
            frame["well"].astype(str)
            + "_"
            + frame[row_column].astype(np.int64).astype(str)
        )
    elif frame["id"].isna().any():
        raise RuntimeError("Official sample cache contains partially missing ids")
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise RuntimeError("Official sample cache contains duplicate ids")
    return frame


def _attach(frame, endpoints):
    output = frame.merge(endpoints, on="id", how="inner", validate="one_to_one")
    if output["well"].nunique() != frame["well"].nunique():
        raise RuntimeError(
            f"Endpoint merge retained {output['well'].nunique()}/{frame['well'].nunique()} wells"
        )
    output["residual_target"] = (
        output["tvt_true_endpoint"].to_numpy(np.float64)
        - output["k2_parent"].to_numpy(np.float64)
    )
    return output


def _fit(train, valid, features, iterations=None, label="model"):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = {
        "objective": "regression",
        "n_estimators": int(iterations or 2200),
        "learning_rate": 0.02,
        "num_leaves": 16,
        "max_depth": 5,
        "min_child_samples": 300,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 2.0,
        "reg_lambda": 50.0,
        "max_bin": 127,
        "random_state": SEED,
        "n_jobs": max(1, min(12, os.cpu_count() or 1)),
        "verbosity": -1,
    }
    model = LGBMRegressor(**params)
    kwargs = {"sample_weight": train["_global_weight"].to_numpy(np.float64)}
    if valid is not None:
        kwargs.update({
            "eval_set": [(B["_matrix"](valid, features), valid["residual_target"].to_numpy(np.float64))],
            "eval_metric": "rmse",
            "callbacks": [early_stopping(160, verbose=False), log_evaluation(100)],
        })
    model.fit(B["_matrix"](train, features), train["residual_target"].to_numpy(np.float64), **kwargs)
    best = getattr(model, "best_iteration_", None)
    best = int(best if best and best > 0 else params["n_estimators"])
    joblib.dump(model, MODEL_DIR / f"{label}.joblib")
    return model, best


def _correction(frame, model, features):
    raw = np.clip(model.predict(B["_matrix"](frame, features)), -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
    smooth = np.empty_like(raw)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        size = min(SMOOTH_ROWS, len(positions) | 1)
        smooth[positions] = uniform_filter1d(raw[positions], size=size, mode="nearest")
    return np.clip(smooth, -CORRECTION_CAP_FT, CORRECTION_CAP_FT)


def _evaluate(frame, correction, label):
    parent = frame["k2_parent"].to_numpy(np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    parent_metric = B["_metrics"](frame, parent, "k2_parent")
    scans = []
    for dose in DOSES:
        prediction = parent + float(dose) * correction
        metric = B["_metrics"](frame, prediction, label)
        scans.append({
            "dose": float(dose), "rmse": metric["rmse"],
            "gain": parent_metric["rmse"] - metric["rmse"],
            "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        })
    return parent_metric, pd.DataFrame(scans).sort_values(["rmse", "dose"], ignore_index=True)


base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = _endpoint_table()
features = list(dict.fromkeys(base_features + list(ENGINEERED)))
train_wells = sorted(base.loc[base["fold"].isin(TRAIN_FOLDS), "well"].unique())
selection_wells = sorted(base.loc[base["fold"].eq(SELECTION_FOLD), "well"].unique())
train = _attach(_official_samples(train_wells, manifest, base_features), endpoints)
valid_sample = _attach(_official_samples(selection_wells, manifest, base_features), endpoints)
official_manifest = manifest[manifest["tag"].eq("official")].copy()
official_manifest["cut_bin"] = 0
selection_full = B["_load_full"](selection_wells, official_manifest, base)
selection_full = selection_full.merge(endpoints, on="id", how="inner", validate="one_to_one")

print(
    "M342-P1 CONFIG",
    {"coverage": coverage, "base_features": len(base_features), "features": len(features),
     "train_rows": len(train), "selection_rows": len(selection_full)},
    flush=True,
)
model, best_iteration = _fit(train, valid_sample, features, label="selection_residual")
selection_correction = _correction(selection_full, model, features)
parent_metric, scan = _evaluate(selection_full, selection_correction, "residual_blend")
safe_scan = scan.loc[
    scan["far_gain"].ge(0.0)
    & scan["p95_change"].le(0.20)
].copy()
selected = (safe_scan.iloc[0] if len(safe_scan) else scan.iloc[0]).to_dict()
dose = float(selected["dose"])
candidate_prediction = selection_full["k2_parent"].to_numpy(np.float64) + dose * selection_correction
rolled = np.empty_like(selection_correction)
for positions in selection_full.groupby("well", sort=False).indices.values():
    positions = np.asarray(positions, np.int64)
    rolled[positions] = np.roll(selection_correction[positions], max(len(positions) // 4, 1))
roll_prediction = selection_full["k2_parent"].to_numpy(np.float64) + dose * rolled
candidate_metric = B["_metrics"](selection_full, candidate_prediction, "residual_blend")
roll_metric = B["_metrics"](selection_full, roll_prediction, "depth_roll_control")
roll_loss = roll_metric["rmse"] - candidate_metric["rmse"]
mechanism_pass = bool(
    len(safe_scan) > 0 and dose > 0.0
    and selected["gain"] >= 0.03 and selected["far_gain"] >= 0.0
    and roll_loss >= 0.015 and selected["p95_change"] <= 0.20
)
print("\nM342-P1 SELECTION SCAN")
print(scan.to_string(index=False), flush=True)
print("M342-P1 SELECTION", {**selected, "roll_loss": roll_loss, "pass": mechanism_pass}, flush=True)

confirmation = None
checks = {"selection_mechanism_pass": mechanism_pass}
if mechanism_pass:
    del train, valid_sample, model
    gc.collect()
    train2_wells = sorted(base.loc[base["fold"].isin((3, 4, 5)), "well"].unique())
    fold2_wells = sorted(base.loc[base["fold"].eq(CONFIRMATION_FOLD), "well"].unique())
    train2 = _attach(_official_samples(train2_wells, manifest, base_features), endpoints)
    held2 = B["_load_full"](fold2_wells, official_manifest, base)
    held2 = held2.merge(endpoints, on="id", how="inner", validate="one_to_one")
    model2, _ = _fit(train2, None, features, iterations=best_iteration, label="confirmation_residual")
    correction2 = _correction(held2, model2, features)
    parent2 = held2["k2_parent"].to_numpy(np.float64)
    prediction2 = parent2 + dose * correction2
    reverse2 = parent2 - dose * correction2
    parent_metric2 = B["_metrics"](held2, parent2, "k2_parent")
    metric2 = B["_metrics"](held2, prediction2, "residual_blend")
    reverse_metric2 = B["_metrics"](held2, reverse2, "negative_direction_control")
    confirmation = {
        "parent": parent_metric2, "candidate": metric2, "reverse": reverse_metric2,
        "gain": parent_metric2["rmse"] - metric2["rmse"],
        "far_gain": parent_metric2["far_gt2000_rmse"] - metric2["far_gt2000_rmse"],
        "reverse_loss": reverse_metric2["rmse"] - metric2["rmse"],
        "p95_change": metric2["p95_well_rmse"] - parent_metric2["p95_well_rmse"],
    }
    checks.update({
        "confirmation_gain_ge_0p02": confirmation["gain"] >= 0.02,
        "confirmation_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
        "reverse_loses_ge_0p03": confirmation["reverse_loss"] >= 0.03,
        "confirmation_p95_change_le_0p20": confirmation["p95_change"] <= 0.20,
    })
else:
    fold2_wells = []

strict_pass = bool(mechanism_pass and confirmation and all(checks.values()))
verdict = {
    "experiment": "main342_k2_residual_gbdt_p1",
    "stage": "fold45_k2_residual_training_fold3_selection_fold345_refit_fold2_confirmation",
    "legal_inputs_only": True,
    "sample_unit": "complete physical well; all official hidden rows",
    "protocol": {
        "train_folds": TRAIN_FOLDS, "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD if mechanism_pass else None,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "features": {"base": len(base_features), "engineered": ENGINEERED, "total": len(features)},
    "model": {"best_iteration": best_iteration, "correction_cap_ft": CORRECTION_CAP_FT,
              "smooth_rows": SMOOTH_ROWS},
    "coverage": coverage,
    "selection": {
        **selected,
        "safe_doses": safe_scan["dose"].astype(float).tolist(),
        "selection_rule": "minimum RMSE among far-nonnegative and p95<=0.20 doses",
        "roll_loss": roll_loss,
    },
    "mechanism_pass": mechanism_pass,
    "confirmation": confirmation,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze model recipe/dose and confirm once on Fold1"
        if strict_pass else
        "FAIL -> legal row features cannot predict the residual left by K2; close P1"
    ),
}

scan.to_csv(OUT / "selection_scan.csv", index=False)
(OUT / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-P1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
