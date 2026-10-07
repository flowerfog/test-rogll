"""M342-AA1: CatBoost diversity on the exact matched P1 residual protocol.

Fold4/5 train LightGBM and CatBoost on identical official-cut K3 features and
K2 residual targets. Fold3 selects a fixed correction blend and dose, evaluated
on every hidden row. Fold2/1 are untouched. The gate requires incremental value
over the matched LightGBM control, not merely gain over K2.
"""

from pathlib import Path
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
P1_SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_catboost_residual_diversity_aa1"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
VALID_FOLD = 3
CAT_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
DOSES = (0.0, 0.25, 0.50, 0.75, 1.00)
MAX_P95_CHANGE = 0.20


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


def _load_p1():
    source = P1_SOURCE.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError("P1 execution marker not found")
    name = "m342_aa1_p1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(P1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(P1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


P = _load_p1()
O, B = P["O"], P["B"]
P["OUT"] = OUT
P["MODEL_DIR"] = MODEL_DIR
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR


def _catboost_fit(train, valid, features):
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is unavailable. Install catboost in the AutoDL environment first."
        ) from exc

    x_train = B["_matrix"](train, features)
    y_train = train["residual_target"].to_numpy(np.float64)
    w_train = train["_global_weight"].to_numpy(np.float64)
    train_pool = Pool(x_train, y_train, weight=w_train, feature_names=features)
    valid_pool = Pool(
        B["_matrix"](valid, features),
        valid["residual_target"].to_numpy(np.float64),
        feature_names=features,
    )
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1200,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=30.0,
        random_strength=0.5,
        random_seed=int(P["SEED"]),
        thread_count=max(1, min(12, os.cpu_count() or 1)),
        od_type="Iter",
        od_wait=120,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(str(MODEL_DIR / "catboost_fold45.cbm"))
    return model, int(model.get_best_iteration() + 1)


def _smooth_prediction(frame, raw):
    raw = np.clip(np.asarray(raw, np.float64), -float(P["CORRECTION_CAP_FT"]), float(P["CORRECTION_CAP_FT"]))
    output = np.empty_like(raw)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        size = min(int(P["SMOOTH_ROWS"]), len(positions) | 1)
        output[positions] = uniform_filter1d(raw[positions], size=size, mode="nearest")
    return np.clip(output, -float(P["CORRECTION_CAP_FT"]), float(P["CORRECTION_CAP_FT"]))


def _metric(frame, prediction, endpoint):
    return B["_metrics"](frame, prediction, endpoint)


base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()
features = list(dict.fromkeys(base_features + list(P["ENGINEERED"])))
train_wells = sorted(base.loc[base["fold"].isin(TRAIN_FOLDS), "well"].unique())
valid_wells = sorted(base.loc[base["fold"].eq(VALID_FOLD), "well"].unique())
train = P["_attach"](P["_official_samples"](train_wells, manifest, base_features), endpoints)
valid_sample = P["_attach"](P["_official_samples"](valid_wells, manifest, base_features), endpoints)
official_manifest = manifest.loc[manifest["tag"].eq("official")].copy()
official_manifest["cut_bin"] = 0
official_manifest["states_in_bin"] = 1
valid_full = B["_load_full"](valid_wells, official_manifest, base)
valid_full = P["_attach"](valid_full, endpoints)

print("M342-AA1 CONFIG", {
    "coverage": coverage,
    "features": len(features),
    "train_rows": len(train),
    "valid_sample_rows": len(valid_sample),
    "valid_full_rows": len(valid_full),
}, flush=True)

lgb_model, lgb_iteration = P["_fit"](
    train, valid_sample, features, label="aa1_lgb_control"
)
cat_model, cat_iteration = _catboost_fit(train, valid_sample, features)
lgb_correction = P["_correction"](valid_full, lgb_model, features)
cat_raw = cat_model.predict(B["_matrix"](valid_full, features))
cat_correction = _smooth_prediction(valid_full, cat_raw)

parent = valid_full["k2_parent"].to_numpy(np.float64)
base_metric = _metric(valid_full, parent, "k2_parent")
lgb_error = lgb_correction - valid_full["residual_target"].to_numpy(np.float64)
cat_error = cat_correction - valid_full["residual_target"].to_numpy(np.float64)
error_correlation = float(np.corrcoef(lgb_error, cat_error)[0, 1])
correction_correlation = float(np.corrcoef(lgb_correction, cat_correction)[0, 1])

scan_rows = []
corrections = {"lgb_control": lgb_correction}
for cat_weight in CAT_WEIGHTS:
    name = f"blend_cat_{cat_weight:g}"
    corrections[name] = (1.0 - cat_weight) * lgb_correction + cat_weight * cat_correction
for name, correction in corrections.items():
    for dose in DOSES:
        metric = _metric(valid_full, parent + dose * correction, name)
        scan_rows.append({
            "arm": name,
            "cat_weight": 0.0 if name == "lgb_control" else float(name.rsplit("_", 1)[1]),
            "dose": dose,
            "rmse": metric["rmse"],
            "gain": base_metric["rmse"] - metric["rmse"],
            "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
        })
scan = pd.DataFrame(scan_rows)
lgb_best = scan[
    scan["arm"].eq("lgb_control")
    & scan["far_gain"].ge(0.0)
    & scan["p95_change"].le(MAX_P95_CHANGE)
].sort_values("rmse").iloc[0]
safe = scan[
    scan["arm"].ne("lgb_control")
    & scan["dose"].gt(0.0)
    & scan["far_gain"].ge(0.0)
    & scan["p95_change"].le(MAX_P95_CHANGE)
]
selected = safe.sort_values("rmse").iloc[0] if not safe.empty else scan.sort_values("rmse").iloc[0]
selected_correction = corrections[str(selected["arm"])]
dose = float(selected["dose"])
prediction = parent + dose * selected_correction
reverse = parent - dose * selected_correction
rolled = np.empty_like(selected_correction)
for positions in valid_full.groupby("well", sort=False).indices.values():
    positions = np.asarray(positions, np.int64)
    rolled[positions] = np.roll(selected_correction[positions], max(len(positions) // 4, 1))

selected_metric = _metric(valid_full, prediction, "selected_catboost_blend")
reverse_metric = _metric(valid_full, reverse, "reverse_control")
roll_metric = _metric(valid_full, parent + dose * rolled, "depth_roll_control")
increment_vs_lgb = float(lgb_best["rmse"] - selected_metric["rmse"])
reverse_loss = float(reverse_metric["rmse"] - selected_metric["rmse"])
roll_loss = float(roll_metric["rmse"] - selected_metric["rmse"])

checks = {
    "fold3_wells_ge_145": int(selected_metric["wells"]) >= 145,
    "selected_uses_catboost": float(selected["cat_weight"]) > 0.0,
    "gain_ge_0p05": base_metric["rmse"] - selected_metric["rmse"] >= 0.05,
    "increment_vs_lgb_ge_0p03": increment_vs_lgb >= 0.03,
    "far_gain_ge_0": base_metric["far_gt2000_rmse"] - selected_metric["far_gt2000_rmse"] >= 0.0,
    "reverse_loses_ge_0p03": reverse_loss >= 0.03,
    "depth_roll_loses_ge_0p01": roll_loss >= 0.01,
    "p95_change_le_0p20": selected_metric["p95_well_rmse"] - base_metric["p95_well_rmse"] <= MAX_P95_CHANGE,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_catboost_residual_diversity_aa1",
    "stage": "fold45_matched_model_family_training_fold3_diversity_gateway",
    "legal_inputs_only": True,
    "protocol": {
        "train_folds": TRAIN_FOLDS,
        "selection_fold": VALID_FOLD,
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "coverage": coverage,
    "features": len(features),
    "models": {
        "lgb_iteration": lgb_iteration,
        "catboost_iteration": cat_iteration,
        "correction_correlation": correction_correlation,
        "residual_error_correlation": error_correlation,
    },
    "lgb_best": lgb_best.to_dict(),
    "selected": selected.to_dict(),
    "confirmation": {
        "base": base_metric,
        "candidate": selected_metric,
        "reverse": reverse_metric,
        "depth_roll": roll_metric,
        "increment_vs_lgb": increment_vs_lgb,
        "reverse_loss": reverse_loss,
        "roll_loss": roll_loss,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> refit frozen CatBoost/LGB blend on Fold3/4/5 and confirm Fold2"
        if strict_pass else
        "FAIL -> model-family diversity adds no transferable residual signal; close AA1"
    ),
}

scan.sort_values(["rmse", "arm", "dose"]).to_csv(OUT / "scan.csv", index=False)
pd.DataFrame([base_metric, selected_metric, reverse_metric, roll_metric]).to_csv(
    OUT / "summary.csv", index=False
)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-AA1 SCAN")
print(scan.sort_values(["rmse", "arm", "dose"]).head(30).to_string(index=False), flush=True)
print("\nM342-AA1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
