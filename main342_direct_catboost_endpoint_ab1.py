"""M342-AB1: direct-target CatBoost endpoint diversity after frozen K2.

Train direct TVT-delta models on Fold 4/5 multi-cut samples and evaluate only
on Fold 3.  This is deliberately not another K2 residual model: LightGBM and
CatBoost predict the absolute suffix trajectory through the frozen B1 decode,
then a small endpoint dose is tested against the current K2 parent.
"""

from pathlib import Path
import gc
import importlib.util
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_direct_catboost_endpoint_ab1"
MODEL_DIR = OUT / "models"
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
DOSES = np.round(np.arange(0.0, 0.3001, 0.025), 3)
SEED = 20260805


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError(f"Cannot locate P1 runtime marker in {path}")
    name = "main342_p1_definitions_ab1"
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
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _fit_cat(train, valid, features):
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise RuntimeError("CatBoost is unavailable in this environment") from exc

    train_pool = Pool(
        B["_matrix"](train, features),
        train["target"].to_numpy(np.float64),
        weight=train["_global_weight"].to_numpy(np.float64),
        feature_names=features,
    )
    valid_pool = Pool(
        B["_matrix"](valid, features),
        valid["target"].to_numpy(np.float64),
        feature_names=features,
    )
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2400,
        learning_rate=0.025,
        depth=7,
        l2_leaf_reg=30.0,
        random_strength=0.5,
        random_seed=SEED,
        thread_count=max(1, min(12, os.cpu_count() or 1)),
        od_type="Iter",
        od_wait=160,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(str(MODEL_DIR / "direct_cat_fold45.cbm"))
    return model, int(model.get_best_iteration() + 1)


def _metric(frame, prediction, endpoint):
    return B["_metrics"](frame.reset_index(drop=True), prediction, endpoint)


def _scan(frame, parent, candidates):
    parent_metric = _metric(frame, parent, "k2_parent")
    rows = []
    for name, endpoint in candidates.items():
        endpoint = np.asarray(endpoint, np.float64)
        for dose in DOSES:
            prediction = parent + float(dose) * (endpoint - parent)
            metric = _metric(frame, prediction, name)
            rows.append({
                "endpoint": name,
                "dose": float(dose),
                "rmse": metric["rmse"],
                "gain": parent_metric["rmse"] - metric["rmse"],
                "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
            })
    return parent_metric, pd.DataFrame(rows)


def _roll_by_well(frame, values):
    output = np.asarray(values, np.float64).copy()
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        if len(positions) > 1:
            output[positions] = np.roll(output[positions], max(1, len(positions) // 3))
    return output


started = time.time()
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
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
edges = np.quantile(train_manifest["actual_frac"], [1 / 3, 2 / 3])
train_manifest["cut_bin"] = B["_bin"](train_manifest["actual_frac"], edges)
state_counts = train_manifest.groupby(["well", "cut_bin"]).size().rename("states_in_bin")
train_manifest = train_manifest.join(state_counts, on=["well", "cut_bin"])
held_manifest = manifest[manifest["well"].isin(held_wells)].copy()
held_manifest["cut_bin"] = B["_bin"](held_manifest["actual_frac"], edges)
held_counts = held_manifest.groupby(["well", "cut_bin"]).size().rename("states_in_bin")
held_manifest = held_manifest.join(held_counts, on=["well", "cut_bin"])

train = B["_load_samples"](train_wells, train_manifest, features)
valid_sample = B["_load_samples"](held_wells, held_manifest, features)
official = manifest[manifest["tag"].eq("official")].copy()
held = B["_load_full"](held_wells, official, base)
held = held.merge(endpoints, on="id", how="inner", validate="one_to_one", sort=False)
held = held.reset_index(drop=True)
if held["well"].nunique() < 145:
    raise RuntimeError(f"Only {held['well'].nunique()} held wells")

print(
    "M342-AB1 CONFIG",
    {
        "coverage": coverage,
        "features": len(features),
        "train_rows": len(train),
        "valid_sample_rows": len(valid_sample),
        "held_rows": len(held),
        "train_wells": len(train_wells),
        "held_wells": held["well"].nunique(),
    },
    flush=True,
)

lgb_model, lgb_iteration = B["_fit"](
    train, valid_sample, features, "_global_weight", "direct_lgb_fold45"
)
cat_model, cat_iteration = _fit_cat(train, valid_sample, features)
matrix = B["_matrix"](held, features)
lgb_delta = np.asarray(lgb_model.predict(matrix), np.float64)
cat_delta = np.asarray(cat_model.predict(matrix), np.float64)
candidates = {
    "direct_lgb": B["_postprocess"](held, lgb_delta),
    "direct_cat": B["_postprocess"](held, cat_delta),
    "raw_delta_mean": B["_postprocess"](held, 0.5 * (lgb_delta + cat_delta)),
}
parent = held["k2_parent"].to_numpy(np.float64)
parent_metric, scan = _scan(held, parent, candidates)
safe = scan[(scan["far_gain"] >= 0.0) & (scan["p95_change"] <= 0.20)].copy()
if safe.empty:
    selected = scan.sort_values(["rmse", "dose"]).iloc[0]
else:
    selected = safe.sort_values(["rmse", "dose"]).iloc[0]
selected_name = str(selected["endpoint"])
selected_dose = float(selected["dose"])
selected_endpoint = candidates[selected_name]
selected_prediction = parent + selected_dose * (selected_endpoint - parent)

lgb_best = scan[(scan["endpoint"].eq("direct_lgb")) & (scan["far_gain"] >= 0.0) & (scan["p95_change"] <= 0.20)]
if lgb_best.empty:
    lgb_best_rmse = float("inf")
else:
    lgb_best_rmse = float(lgb_best["rmse"].min())
increment_vs_lgb = lgb_best_rmse - float(selected["rmse"])

move = selected_prediction - parent
reverse = parent - move
rolled = parent + _roll_by_well(held, move)
reverse_metric = _metric(held, reverse, "reverse_control")
roll_metric = _metric(held, rolled, "depth_roll_control")
selected_metric = _metric(held, selected_prediction, selected_name)
endpoint_metrics = [_metric(held, value, name) for name, value in candidates.items()]

scan = scan.sort_values(["rmse", "endpoint", "dose"], ignore_index=True)
scan.to_csv(OUT / "dose_scan.csv", index=False)
pd.DataFrame({
    "id": held["id"].astype(str),
    "well": held["well"].astype(str),
    "fold": held["fold"].astype(int),
    "tvt_true": held["tvt_true"].to_numpy(np.float64),
    "md_since": held["md_since"].to_numpy(np.float64),
    "k2_parent": parent,
    **candidates,
    "selected_prediction": selected_prediction,
}).to_parquet(OUT / "fold3_endpoints.parquet", index=False)
joblib.dump(lgb_model, MODEL_DIR / "direct_lgb_fold45.joblib")

checks = {
    "fold3_wells_ge_145": held["well"].nunique() >= 145,
    "selected_positive_dose": selected_dose > 0.0,
    "gain_ge_0p03": float(selected["gain"]) >= 0.03,
    "far_gain_ge_0": float(selected["far_gain"]) >= 0.0,
    "increment_vs_lgb_ge_0p01": increment_vs_lgb >= 0.01,
    "reverse_loses_ge_0p03": reverse_metric["rmse"] - selected_metric["rmse"] >= 0.03,
    "depth_roll_loses_ge_0p01": roll_metric["rmse"] - selected_metric["rmse"] >= 0.01,
    "p95_change_le_0p20": float(selected["p95_change"]) <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_direct_catboost_endpoint_ab1",
    "stage": "fold45_direct_target_model_family_fold3_k2_compatibility_gateway",
    "legal_inputs_only": 1,
    "sample_unit": "complete physical well; all official hidden rows",
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "fold2_touched": 0,
        "fold1_touched": 0,
    },
    "coverage": coverage,
    "features": len(features),
    "models": {
        "lgb_iteration": lgb_iteration,
        "cat_iteration": cat_iteration,
        "raw_delta_correlation": float(np.corrcoef(lgb_delta, cat_delta)[0, 1]),
    },
    "parent": parent_metric,
    "endpoint_metrics": endpoint_metrics,
    "selected": selected.to_dict(),
    "increment_vs_best_safe_lgb": increment_vs_lgb,
    "controls": {
        "reverse": reverse_metric,
        "depth_roll": roll_metric,
        "reverse_loss": reverse_metric["rmse"] - selected_metric["rmse"],
        "roll_loss": roll_metric["rmse"] - selected_metric["rmse"],
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> freeze model family, iterations, and dose; refit Fold3/4/5 for Fold2"
        if strict_pass else
        "FAIL -> direct CatBoost adds no transferable endpoint mechanism; close AB1"
    ),
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")

print("\nM342-AB1 ENDPOINT SUMMARY", flush=True)
print(pd.DataFrame(endpoint_metrics).to_string(index=False), flush=True)
print("\nM342-AB1 DOSE SCAN", flush=True)
print(scan.head(20).to_string(index=False), flush=True)
print("\nM342-AB1 FINAL VERDICT", flush=True)
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)

del train, valid_sample, lgb_model, cat_model, matrix
gc.collect()
