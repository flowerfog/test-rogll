"""M342-AA2: zero-retuning Fold2 confirmation of the marginal AA1 arm.

AA1 froze pure CatBoost, dose 0.5, and 65 iterations on Fold3. AA2 refits the
same model on Fold3/4/5 and evaluates Fold2 once, with a matched 88-iteration
LightGBM control. Fold1 remains untouched.
"""

from pathlib import Path
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
AA1_SOURCE = ROOT / "main342_catboost_residual_diversity_aa1.py"
OUT = ROOT / "oof_exports" / "main342_catboost_residual_fold2_aa2"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (3, 4, 5)
CONFIRMATION_FOLD = 2
CAT_ITERATIONS = 65
LGB_ITERATIONS = 88
DOSE = 0.50


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


def _load_aa1():
    source = AA1_SOURCE.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError("AA1 execution marker not found")
    name = "m342_aa2_aa1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(AA1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(AA1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


A = _load_aa1()
P, O, B = A["P"], A["O"], A["B"]
P["OUT"] = OUT
P["MODEL_DIR"] = MODEL_DIR
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR


def _fit_catboost(train, features):
    from catboost import CatBoostRegressor, Pool

    pool = Pool(
        B["_matrix"](train, features),
        train["residual_target"].to_numpy(np.float64),
        weight=train["_global_weight"].to_numpy(np.float64),
        feature_names=features,
    )
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=CAT_ITERATIONS,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=30.0,
        random_strength=0.5,
        random_seed=int(P["SEED"]),
        thread_count=max(1, min(12, os.cpu_count() or 1)),
        verbose=25,
        allow_writing_files=False,
    )
    model.fit(pool)
    model.save_model(str(MODEL_DIR / "catboost_fold345.cbm"))
    return model


base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()
features = list(dict.fromkeys(base_features + list(P["ENGINEERED"])))
train_wells = sorted(base.loc[base["fold"].isin(TRAIN_FOLDS), "well"].unique())
held_wells = sorted(base.loc[base["fold"].eq(CONFIRMATION_FOLD), "well"].unique())
train = P["_attach"](
    P["_official_samples"](train_wells, manifest, base_features), endpoints
)
official_manifest = manifest.loc[manifest["tag"].eq("official")].copy()
official_manifest["cut_bin"] = 0
official_manifest["states_in_bin"] = 1
held = B["_load_full"](held_wells, official_manifest, base)
held = P["_attach"](held, endpoints)

print("M342-AA2 CONFIG", {
    "coverage": coverage,
    "features": len(features),
    "train_rows": len(train),
    "held_rows": len(held),
    "cat_iterations": CAT_ITERATIONS,
    "lgb_iterations": LGB_ITERATIONS,
    "dose": DOSE,
}, flush=True)

lgb_model, _ = P["_fit"](
    train, None, features, iterations=LGB_ITERATIONS, label="aa2_lgb_control"
)
cat_model = _fit_catboost(train, features)
lgb_correction = P["_correction"](held, lgb_model, features)
cat_correction = A["_smooth_prediction"](
    held, cat_model.predict(B["_matrix"](held, features))
)

parent = held["k2_parent"].to_numpy(np.float64)
base_metric = B["_metrics"](held, parent, "k2_parent")
lgb_metric = B["_metrics"](
    held, parent + DOSE * lgb_correction, "matched_lgb"
)
cat_metric = B["_metrics"](
    held, parent + DOSE * cat_correction, "frozen_catboost"
)
reverse_metric = B["_metrics"](
    held, parent - DOSE * cat_correction, "reverse_control"
)
rolled = np.empty_like(cat_correction)
for positions in held.groupby("well", sort=False).indices.values():
    positions = np.asarray(positions, np.int64)
    rolled[positions] = np.roll(cat_correction[positions], max(len(positions) // 4, 1))
roll_metric = B["_metrics"](
    held, parent + DOSE * rolled, "depth_roll_control"
)

gain = float(base_metric["rmse"] - cat_metric["rmse"])
far_gain = float(base_metric["far_gt2000_rmse"] - cat_metric["far_gt2000_rmse"])
increment_vs_lgb = float(lgb_metric["rmse"] - cat_metric["rmse"])
reverse_loss = float(reverse_metric["rmse"] - cat_metric["rmse"])
roll_loss = float(roll_metric["rmse"] - cat_metric["rmse"])
p95_change = float(cat_metric["p95_well_rmse"] - base_metric["p95_well_rmse"])
checks = {
    "fold2_wells_ge_145": int(cat_metric["wells"]) >= 145,
    "gain_ge_0p03": gain >= 0.03,
    "far_gain_ge_0": far_gain >= 0.0,
    "increment_vs_lgb_ge_0p01": increment_vs_lgb >= 0.01,
    "reverse_loses_ge_0p03": reverse_loss >= 0.03,
    "depth_roll_loses_ge_0p01": roll_loss >= 0.01,
    "p95_change_le_0p20": p95_change <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_catboost_residual_fold2_aa2",
    "stage": "frozen_fold345_refit_fold2_confirmation",
    "selection_or_tuning_count": 0,
    "frozen_recipe": {
        "cat_weight": 1.0,
        "dose": DOSE,
        "cat_iterations": CAT_ITERATIONS,
        "lgb_control_iterations": LGB_ITERATIONS,
    },
    "protocol": {
        "train_folds": TRAIN_FOLDS,
        "confirmation_fold": CONFIRMATION_FOLD,
        "fold1_touched": False,
    },
    "coverage": coverage,
    "confirmation": {
        "base": base_metric,
        "catboost": cat_metric,
        "matched_lgb": lgb_metric,
        "reverse": reverse_metric,
        "depth_roll": roll_metric,
        "gain": gain,
        "far_gain": far_gain,
        "increment_vs_lgb": increment_vs_lgb,
        "reverse_loss": reverse_loss,
        "roll_loss": roll_loss,
        "p95_change": p95_change,
        "correction_correlation": float(np.corrcoef(lgb_correction, cat_correction)[0, 1]),
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> run fivefold CatBoost crossfit before any deployment work"
        if strict_pass else
        "FAIL -> CatBoost Fold3 diversity does not transfer; close AA1/AA2"
    ),
}

pd.DataFrame([base_metric, cat_metric, lgb_metric, reverse_metric, roll_metric]).to_csv(
    OUT / "summary.csv", index=False
)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-AA2 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
