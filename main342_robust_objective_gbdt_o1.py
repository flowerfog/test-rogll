"""M342-O1: robust-loss multi-cut GBDT endpoint gateway.

Uses the existing legal non-spatial K3 feature cache. Squared error is the
control; Huber and Fair losses are candidate error mechanisms. Fold4+5 train
and Fold3 select one objective, iteration count and conservative K2 blend
dose. Fold2 is evaluated only after that mechanism gate passes. Fold1 is not
loaded.
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


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
B1_SOURCE = ROOT / "main342_cut_fraction_specialist_gateway_b1.py"
K1_SOURCE = ROOT / "main342_consensus_rejection_stack_k1.py"
OUT = ROOT / "oof_exports" / "main342_robust_objective_gbdt_o1"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
DOSES = np.asarray([0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20], np.float64)
ARMS = {
    "squared_control": {"objective": "regression"},
    "huber": {"objective": "huber", "alpha": 0.90},
    "fair": {"objective": "fair", "fair_c": 1.0},
}
SEED = 20260804
FROZEN_SHRINKAGE = 0.20
K2_EXPECTED_RMSE = 7.689693565766868


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


def _load_prefix(path, marker, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Execution marker not found in {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


B = _load_prefix(B1_SOURCE, "started = time.time()", "m342_o1_b1_definitions")
K = _load_prefix(K1_SOURCE, "G = _load_g1()", "m342_o1_k1_definitions")
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR


def _build_k2():
    G = K["_load_g1"]()
    V = G["_load_inventory"]()
    frame = V["base"].reset_index(drop=True)
    paths = V["matrix"].astype(np.float64, copy=False)
    names = list(V["names"])
    base_index = int(V["base_index"])
    truth = V["truth_all"].astype(np.float64, copy=False)
    parent = paths[base_index]
    error = parent - truth
    fold = frame["fold"].to_numpy(np.int8)
    active = [names.index(name) for name in K["ACTIVE_ENDPOINTS"]]
    active_delta = paths[active] - parent[None, :]
    raw_factor = K["_raw_gates"](active_delta, frame)["magnitude_soft_mean1"]
    candidates = [index for index in range(len(names)) if index != base_index]
    delta = paths[candidates] - parent[None, :]
    output = parent.copy()
    for held in range(1, 6):
        train = fold != held
        valid = fold == held
        alpha, _ = G["_fit_coefficients"](delta, error, train)
        move = FROZEN_SHRINKAGE * (alpha @ delta[:, valid])
        factor = np.clip(
            raw_factor[valid] / max(float(np.mean(raw_factor[train])), 1e-6),
            0.0, float(K["MAX_NORMALISED_FACTOR"]),
        )
        output[valid] += factor * move
    metric = B["_metrics"](frame, output, "k2")
    if abs(metric["rmse"] - K2_EXPECTED_RMSE) > 1e-9:
        raise RuntimeError(f"K2 reproduction gap: {metric['rmse']}")
    return pd.DataFrame({"id": frame["id"].astype(str), "k2_parent": output})


def _prepare_inventory():
    for required in (B["CACHE_DIR"] / "cut_manifest.csv", B["FEATURE_DIR"], B["V168_PATH"]):
        if not required.exists():
            raise FileNotFoundError(required)
    base = pd.read_parquet(B["V168_PATH"])
    columns = ["id", "well", "fold", "tvt_true", "md_since", "metadata_analog_w0p05"]
    missing = sorted(set(columns).difference(base.columns))
    if missing:
        raise RuntimeError(f"V168 missing: {missing}")
    base = base[columns].rename(columns={"metadata_analog_w0p05": "base_v168"})
    base["id"] = base["id"].astype(str)
    base["well"] = base["well"].astype(str)
    manifest = pd.read_csv(B["CACHE_DIR"] / "cut_manifest.csv")
    manifest["well"] = manifest["well"].astype(str)
    manifest["tag"] = manifest["tag"].astype(str)
    manifest = manifest[manifest["tag"].isin(B["TAGS"])].copy()
    complete = manifest.groupby("well")["tag"].nunique()
    complete_wells = set(complete[complete.eq(len(B["TAGS"]))].index)
    base = base[base["well"].isin(complete_wells)].reset_index(drop=True)
    manifest = manifest[manifest["well"].isin(complete_wells)].copy()
    coverage = base.groupby("fold")["well"].nunique().astype(int).to_dict()
    if sum(coverage.get(fold, 0) for fold in TRAIN_FOLDS) < 295 or coverage.get(3, 0) < 145:
        raise RuntimeError(f"Insufficient O1 coverage: {coverage}")
    schema = B["CACHE_DIR"] / "matched_k3_gateway_features.csv"
    if not schema.is_file():
        raise FileNotFoundError(
            f"Missing {schema}; rerun only the schema/cache section of the K3 gateway"
        )
    features = pd.read_csv(schema)["feature"].astype(str).tolist()
    features = [name for name in features if not B["_is_spatial"](name)]
    features = list(dict.fromkeys(features + ["cut_visible_frac"]))
    return base, manifest, features, coverage


def _stage_frames(base, manifest, train_folds, held_fold, features):
    train_wells = sorted(base.loc[base["fold"].isin(train_folds), "well"].unique())
    held_wells = sorted(base.loc[base["fold"].eq(held_fold), "well"].unique())
    train_manifest = manifest[manifest["well"].isin(train_wells)].copy()
    edges = np.quantile(train_manifest["actual_frac"], [1 / 3, 2 / 3])
    train_manifest["cut_bin"] = B["_bin"](train_manifest["actual_frac"], edges)
    counts = train_manifest.groupby(["well", "cut_bin"]).size().rename("states_in_bin")
    train_manifest = train_manifest.join(counts, on=["well", "cut_bin"])
    official = manifest[manifest["tag"].eq("official")].copy()
    official["cut_bin"] = B["_bin"](official["actual_frac"], edges)
    train = B["_load_samples"](train_wells, train_manifest, features)
    held = B["_load_full"](held_wells, official, base)
    return train, held, edges


def _fit_arm(train, valid, features, arm, label, iterations=None):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(B["LGB_PARAMS"])
    params.update(ARMS[arm])
    params.update({
        "n_estimators": int(iterations or 2200),
        "random_state": SEED,
        "n_jobs": max(1, min(12, os.cpu_count() or 1)),
    })
    model = LGBMRegressor(**params)
    kwargs = {"sample_weight": train["_global_weight"].to_numpy(np.float64)}
    if valid is not None:
        kwargs.update({
            "eval_set": [(B["_matrix"](valid, features), valid["target"].to_numpy(np.float64))],
            "eval_metric": "rmse",
            "callbacks": [early_stopping(140, verbose=False), log_evaluation(100)],
        })
    model.fit(B["_matrix"](train, features), train["target"].to_numpy(np.float64), **kwargs)
    best = getattr(model, "best_iteration_", None)
    best = int(best if best and best > 0 else params["n_estimators"])
    joblib.dump(model, MODEL_DIR / f"{label}.joblib")
    return model, best


def _predict(frame, model, features):
    delta = model.predict(B["_matrix"](frame, features))
    return B["_postprocess"](frame, delta)


def _blend_scan(frame, parent, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    rows = []
    for dose in DOSES:
        prediction = parent + float(dose) * (endpoint - parent)
        rows.append({
            "dose": float(dose),
            "rmse": float(np.sqrt(np.mean(np.square(prediction - truth)))),
        })
    return pd.DataFrame(rows)


base, manifest, features, coverage = _prepare_inventory()
k2 = _build_k2()
print("M342-O1 CONFIG", {"coverage": coverage, "features": len(features)}, flush=True)

train, held, selection_edges = _stage_frames(
    base, manifest, TRAIN_FOLDS, SELECTION_FOLD, features
)
held = held.merge(k2, on="id", how="left", validate="one_to_one")
if held["k2_parent"].isna().any():
    raise RuntimeError("K2 does not cover Fold3 selection rows")
parent = held["k2_parent"].to_numpy(np.float64)
selection_rows = []
selection_predictions = {}
iterations = {}
for arm in ARMS:
    model, best = _fit_arm(train, held, features, arm, f"selection_{arm}")
    endpoint = _predict(held, model, features)
    scan = _blend_scan(held, parent, endpoint)
    best_row = scan.sort_values(["rmse", "dose"]).iloc[0]
    prediction = parent + float(best_row["dose"]) * (endpoint - parent)
    parent_metric = B["_metrics"](held, parent, "k2_parent")
    metric = B["_metrics"](held, prediction, arm)
    error_rho = float(np.corrcoef(endpoint - held["tvt_true"], parent - held["tvt_true"])[0, 1])
    selection_rows.append({
        "arm": arm, "best_iteration": best, "dose": float(best_row["dose"]),
        "endpoint_rmse": B["_metrics"](held, endpoint, arm + "_endpoint")["rmse"],
        "blend_rmse": metric["rmse"], "gain": parent_metric["rmse"] - metric["rmse"],
        "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "error_rho": error_rho, "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
    })
    selection_predictions[arm] = endpoint
    iterations[arm] = best
    scan.assign(arm=arm).to_csv(OUT / f"selection_scan_{arm}.csv", index=False)
    del model
    gc.collect()

selection = pd.DataFrame(selection_rows).sort_values("blend_rmse", ignore_index=True)
control = selection.loc[selection["arm"].eq("squared_control")].iloc[0]
robust = selection.loc[selection["arm"].isin(["huber", "fair"])].sort_values("blend_rmse").iloc[0]
selected_arm = str(robust["arm"])
mechanism_pass = bool(
    robust["dose"] > 0.0
    and robust["gain"] >= 0.03
    and control["blend_rmse"] - robust["blend_rmse"] >= 0.015
    and robust["p95_change"] <= 0.20
)
print("\nM342-O1 SELECTION")
print(selection.to_string(index=False), flush=True)

confirmation = None
checks = {"selection_mechanism_pass": mechanism_pass}
if mechanism_pass:
    del train, held
    gc.collect()
    train2, held2, confirmation_edges = _stage_frames(
        base, manifest, (3, 4, 5), CONFIRMATION_FOLD, features
    )
    held2 = held2.merge(k2, on="id", how="left", validate="one_to_one")
    parent2 = held2["k2_parent"].to_numpy(np.float64)
    model2, _ = _fit_arm(
        train2, None, features, selected_arm, f"confirmation_{selected_arm}",
        iterations=int(robust["best_iteration"]),
    )
    endpoint2 = _predict(held2, model2, features)
    dose = float(robust["dose"])
    prediction2 = parent2 + dose * (endpoint2 - parent2)
    reverse2 = parent2 - dose * (endpoint2 - parent2)
    parent_metric2 = B["_metrics"](held2, parent2, "k2_parent")
    metric2 = B["_metrics"](held2, prediction2, selected_arm)
    reverse_metric2 = B["_metrics"](held2, reverse2, "negative_direction_control")
    confirmation = {
        "arm": selected_arm, "dose": dose,
        "parent": parent_metric2, "candidate": metric2, "reverse": reverse_metric2,
        "gain": parent_metric2["rmse"] - metric2["rmse"],
        "far_gain": parent_metric2["far_gt2000_rmse"] - metric2["far_gt2000_rmse"],
        "reverse_loss": reverse_metric2["rmse"] - metric2["rmse"],
        "p95_change": metric2["p95_well_rmse"] - parent_metric2["p95_well_rmse"],
    }
    checks.update({
        "confirmation_gain_ge_0p02": confirmation["gain"] >= 0.02,
        "confirmation_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
        "negative_direction_loses_ge_0p03": confirmation["reverse_loss"] >= 0.03,
        "confirmation_p95_change_le_0p20": confirmation["p95_change"] <= 0.20,
    })
    pd.DataFrame({
        "id": held2["id"], "well": held2["well"], "fold": held2["fold"],
        "tvt_true": held2["tvt_true"], "k2_parent": parent2,
        "robust_endpoint": endpoint2, "candidate": prediction2,
    }).to_parquet(OUT / "fold2_predictions.parquet", index=False)
else:
    confirmation_edges = None

strict_pass = bool(mechanism_pass and confirmation and all(checks.values()))
verdict = {
    "experiment": "main342_robust_objective_gbdt_o1",
    "stage": "fold45_robust_loss_selection_fold345_refit_fold2_confirmation",
    "legal_inputs_only": True,
    "sample_unit": "complete physical well; all official hidden rows",
    "protocol": {
        "train_folds": TRAIN_FOLDS, "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD if mechanism_pass else None,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "feature_cache": str(B["CACHE_DIR"]),
    "feature_count": len(features),
    "coverage": coverage,
    "arms": ARMS,
    "selection_edges": selection_edges,
    "confirmation_edges": confirmation_edges,
    "selection": selection.to_dict("records"),
    "selected_arm": selected_arm,
    "mechanism_pass": mechanism_pass,
    "confirmation": confirmation,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze objective/iterations/dose and confirm once on Fold1"
        if strict_pass else
        "FAIL -> robust loss does not add a transferable GBDT error mechanism; close O1"
    ),
}

selection.to_csv(OUT / "selection.csv", index=False)
(OUT / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-O1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
