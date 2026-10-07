"""V226: pure geometry/spatial expert plus fidelity-conditioned mixture.

The expert excludes horizontal GR, typewell, PF, likPF, beam and sequence
features.  It uses only legal trajectory geometry, visible anchor state and
outer-fold-clean spatial surfaces.  V225 fidelity features are then tested as
a well-level gate between V168 and this genuinely different expert.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
BASE = OOF / "likpf_residual_gbdt_full5_v215"
SPATIAL = OOF / "foldclean_full_fleongg_lgb_v0"
FIDELITY = OOF / "typewell_fidelity_observability_v225/well_fidelity.csv"
PARENT_PATH = OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
OUT = OOF / "pure_spatial_fidelity_moe_v226"
MODELS = OUT / "models"
for path in (OUT, MODELS):
    path.mkdir(parents=True, exist_ok=True)

SEED = 226
WARM_TAU = 100.0
WARM_FLOOR = 0.12
TARGET_CLIP = 160.0
SMOOTH_WINDOW = 81


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


required = (
    BASE / "well_metadata.csv", BASE / "features.csv",
    SPATIAL / "well_metadata.csv", SPATIAL / "all_features.csv",
    SPATIAL / "no_spatial_features.csv", FIDELITY, PARENT_PATH,
)
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

meta = pd.read_csv(BASE / "well_metadata.csv", dtype={"well": str})
spatial_meta = pd.read_csv(SPATIAL / "well_metadata.csv", dtype={"well": str})
fold_check = meta[["well", "fold"]].merge(
    spatial_meta[["well", "fold"]], on="well", how="outer",
    suffixes=("_base", "_spatial"), validate="one_to_one",
)
if len(fold_check) != 773 or not fold_check["fold_base"].eq(fold_check["fold_spatial"]).all():
    raise RuntimeError("V215 and spatial fold maps differ")

base_features = pd.read_csv(BASE / "features.csv")["feature"].astype(str).tolist()
all_spatial = pd.read_csv(SPATIAL / "all_features.csv")["feature"].astype(str).tolist()
no_spatial = set(pd.read_csv(SPATIAL / "no_spatial_features.csv")["feature"].astype(str))
spatial_features = [name for name in all_spatial if name not in no_spatial]

BANNED_GEOMETRY_TOKENS = (
    "gr", "typewell", "type_well", "tw_", "pf", "lik", "beam", "sc8",
    "sc15", "sc25", "sc_", "hyb", "sig", "ncc", "corr", "cost",
    "ancc", "astn", "egfd", "buda", "formation", "form_", "dense",
    "neighbor", "gold", "strong", "v66", "v76", "v116", "v144", "v168",
)
GEOMETRY_EXACT = {
    "md", "x", "y", "z", "last_known_tvt", "md_since", "row_norm",
    "row_fraction", "progress", "distance", "anchor_md", "anchor_z",
}
GEOMETRY_TOKENS = (
    "md_", "_md", "depth", "tvd", "incl", "inc_", "azim", "azi_",
    "curv", "dogleg", "dls", "dip", "trajectory", "progress", "distance",
    "anchor", "dx", "dy", "dz", "slope", "trend",
)


def _is_geometry(name):
    lower = name.lower()
    if any(token in lower for token in BANNED_GEOMETRY_TOKENS):
        return False
    return lower in GEOMETRY_EXACT or any(token in lower for token in GEOMETRY_TOKENS)


geometry_features = [name for name in base_features if _is_geometry(name)]
features = list(dict.fromkeys(geometry_features + spatial_features))
if len(geometry_features) < 6 or len(spatial_features) < 20:
    raise RuntimeError({"geometry": geometry_features, "spatial_count": len(spatial_features)})
print("V226 FEATURES", {"geometry": geometry_features, "spatial": len(spatial_features), "total": len(features)}, flush=True)


def _paths(root, well, full):
    suffix = "full" if full else "sample"
    return root / "features" / f"{well}__{suffix}.pkl"


def _load_one(well, full):
    base = pd.read_pickle(_paths(BASE, well, full))
    spatial = pd.read_pickle(_paths(SPATIAL, well, full))
    key = "id" if "id" in base and "id" in spatial else "_row_idx"
    aligned = spatial.set_index(key)[spatial_features].reindex(base[key].to_numpy())
    if aligned.isna().all(axis=1).any():
        raise RuntimeError(f"{well}: incomplete spatial alignment")
    base[spatial_features] = aligned.to_numpy(np.float32)
    return base


def _load(wells, full):
    return pd.concat([_load_one(str(well), full) for well in wells], ignore_index=True, sort=False)


def _prepare(frame, training):
    frame = frame.copy()
    warm = 1.0 - np.exp(-np.maximum(frame["md_since"].to_numpy(float), 0.0) / WARM_TAU)
    frame["spatial_target"] = np.clip(
        frame["target"].to_numpy(float) / np.maximum(warm, WARM_FLOOR),
        -TARGET_CLIP, TARGET_CLIP,
    ).astype(np.float32)
    if training:
        counts = frame.groupby("well").size()
        weight = frame["well"].map((1.0 / counts).to_dict()).to_numpy(float).copy()
        frame["sample_weight"] = weight / weight.mean()
    return frame


def _matrix(frame):
    return (frame.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
            .fillna(0.0).to_numpy(np.float32))


def _new_model(fold):
    params = dict(
        objective="regression", boosting_type="gbdt", n_estimators=2800,
        learning_rate=0.022, num_leaves=48, min_child_samples=120,
        subsample=0.82, subsample_freq=1, colsample_bytree=0.78,
        reg_alpha=0.25, reg_lambda=20.0, max_bin=255, n_jobs=-1,
        random_state=SEED + fold, verbosity=-1,
    )
    if os.environ.get("V226_CPU", "0") != "1":
        params["device_type"] = "gpu"
    return LGBMRegressor(**params)


def _fit(train, valid, fold):
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["spatial_target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(180, verbose=False), log_evaluation(100)],
    )
    model = _new_model(fold)
    try:
        model.fit(_matrix(train), train["spatial_target"].to_numpy(float), **kwargs)
    except Exception as exc:
        if "gpu" not in str(exc).lower() and "opencl" not in str(exc).lower():
            raise
        print("V226 GPU fallback", repr(exc), flush=True)
        os.environ["V226_CPU"] = "1"
        model = _new_model(fold)
        model.fit(_matrix(train), train["spatial_target"].to_numpy(float), **kwargs)
    return model


def _smooth(frame, prediction):
    output = np.asarray(prediction, float).copy()
    for indices in frame.groupby("well", sort=False).indices.values():
        indices = np.asarray(indices, np.int64)
        n = len(indices)
        window = min(SMOOTH_WINDOW, n if n % 2 else n - 1)
        if window >= 5:
            output[indices] = savgol_filter(output[indices], window, 3)
    return output


parts = []
model_folds = []
for held_fold in range(1, 6):
    train_wells = meta.loc[~meta["fold"].eq(held_fold), "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"].eq(held_fold), "well"].astype(str).tolist()
    train = _prepare(_load(train_wells, full=False), training=True)
    valid = _prepare(_load(valid_wells, full=True), training=False)
    valid = valid.sort_values(["well", "_row_idx"], kind="stable").reset_index(drop=True)
    print(f"V226 fold {held_fold}: train={len(train)} valid={len(valid)}", flush=True)
    model = _fit(train, valid, held_fold)
    delta = model.predict(_matrix(valid), num_iteration=model.best_iteration_)
    warm = 1.0 - np.exp(-np.maximum(valid["md_since"].to_numpy(float), 0.0) / WARM_TAU)
    last = valid["last_known_tvt"].to_numpy(float)
    prediction = _smooth(valid, last + warm * delta)
    truth = last + valid["target"].to_numpy(float)
    report = {"fold": held_fold, "rows": len(valid), "wells": len(valid_wells),
              "best_iteration": int(model.best_iteration_ or model.n_estimators),
              "spatial_rmse": _rmse(truth, prediction)}
    model_folds.append(report)
    print("V226 SPATIAL FOLD", report, flush=True)
    parts.append(pd.DataFrame({
        "id": valid["id"].astype(str), "well": valid["well"].astype(str),
        "fold": held_fold, "md_since": valid["md_since"].to_numpy(float),
        "tvt_true": truth, "pure_spatial": prediction,
    }))
    joblib.dump(model, MODELS / f"fold{held_fold}.joblib")
    del train, valid, model, delta, prediction
    gc.collect()

oof = pd.concat(parts, ignore_index=True)
if len(oof) != 3_783_989 or oof["well"].nunique() != 773 or oof["id"].duplicated().any():
    raise RuntimeError({"rows": len(oof), "wells": oof["well"].nunique()})

parent_raw = pd.read_parquet(PARENT_PATH)
if "id" not in parent_raw:
    well_col = next(name for name in ("well", "well_id") if name in parent_raw)
    row_col = next(name for name in ("row_idx", "row_index", "_row_idx") if name in parent_raw)
    parent_raw["id"] = parent_raw[well_col].astype(str) + "_" + parent_raw[row_col].astype(int).astype(str)
parent_col = "metadata_analog_w0p05"
parent = parent_raw[["id", parent_col]].rename(columns={parent_col: "v168"})
oof = oof.merge(parent, on="id", how="left", validate="one_to_one")
if oof["v168"].isna().any():
    raise RuntimeError("Incomplete V168 alignment")

fidelity = pd.read_csv(FIDELITY, dtype={"well": str})
audit_columns = [name for name in fidelity if name.startswith("audit_") or name in {"predicted_alpha", "shuffled_alpha"}]
router_features = [name for name in fidelity if name not in {"well", "fold", "rows"} and name not in audit_columns]

# Add legal disagreement summaries between the experts; no truth is used here.
well_rows = []
for well, group in oof.groupby("well", sort=False):
    move = group["pure_spatial"].to_numpy(float) - group["v168"].to_numpy(float)
    error = group["v168"].to_numpy(float) - group["tvt_true"].to_numpy(float)
    denominator = float(np.dot(move, move))
    alpha = -float(np.dot(error, move)) / denominator if denominator > 1e-9 else 0.0
    well_rows.append({
        "well": str(well), "fold": int(group["fold"].iloc[0]), "rows": len(group),
        "disagreement_mean": float(np.mean(move)),
        "disagreement_abs_mean": float(np.mean(np.abs(move))),
        "disagreement_abs_p90": float(np.quantile(np.abs(move), 0.90)),
        "disagreement_std": float(np.std(move)),
        "disagreement_far_mean": float(np.mean(move[group["md_since"].to_numpy(float) > 2000.0]))
        if np.any(group["md_since"].to_numpy(float) > 2000.0) else 0.0,
        "audit_alpha": float(np.clip(alpha, 0.0, 1.0)),
    })
well_table = pd.DataFrame(well_rows).merge(
    fidelity[["well"] + router_features], on="well", validate="one_to_one"
)
disagreement_features = [name for name in well_table if name.startswith("disagreement_")]
router_features = router_features + disagreement_features

predicted_alpha = np.zeros(len(well_table), float)
shuffled_alpha = np.zeros(len(well_table), float)
y_alpha = well_table["audit_alpha"].to_numpy(float)
x_all = well_table[router_features].replace([np.inf, -np.inf], np.nan)
rng = np.random.default_rng(SEED)
for held_fold in range(1, 6):
    train_mask = ~well_table["fold"].eq(held_fold).to_numpy()
    valid_mask = ~train_mask
    medians = x_all.loc[train_mask].median(axis=0)
    x_train = x_all.loc[train_mask].fillna(medians).fillna(0.0).to_numpy(np.float32)
    x_valid = x_all.loc[valid_mask].fillna(medians).fillna(0.0).to_numpy(np.float32)
    params = dict(n_estimators=700, min_samples_leaf=12, max_features=0.75,
                  n_jobs=-1, random_state=SEED + held_fold)
    model = ExtraTreesRegressor(**params)
    model.fit(x_train, y_alpha[train_mask], sample_weight=well_table.loc[train_mask, "rows"])
    predicted_alpha[valid_mask] = np.clip(model.predict(x_valid), 0.0, 1.0)
    control = ExtraTreesRegressor(**params)
    control.fit(x_train, rng.permutation(y_alpha[train_mask]), sample_weight=well_table.loc[train_mask, "rows"])
    shuffled_alpha[valid_mask] = np.clip(control.predict(x_valid), 0.0, 1.0)

alpha_lookup = dict(zip(well_table["well"], predicted_alpha))
shuffle_lookup = dict(zip(well_table["well"], shuffled_alpha))
oracle_lookup = dict(zip(well_table["well"], y_alpha))
alpha_row = oof["well"].map(alpha_lookup).to_numpy(float)
shuffle_row = oof["well"].map(shuffle_lookup).to_numpy(float)
oracle_row = oof["well"].map(oracle_lookup).to_numpy(float)
parent_values = oof["v168"].to_numpy(float)
spatial_values = oof["pure_spatial"].to_numpy(float)
truth = oof["tvt_true"].to_numpy(float)
move = spatial_values - parent_values
oof["fidelity_moe"] = parent_values + alpha_row * move
oof["shuffled_moe"] = parent_values + shuffle_row * move
oof["oracle_moe"] = parent_values + oracle_row * move

# Also report the best global dose ceiling without claiming deployability.
denominator = float(np.dot(move, move))
global_alpha = float(np.clip(-np.dot(parent_values - truth, move) / denominator, 0.0, 1.0))
global_prediction = parent_values + global_alpha * move
summary = pd.DataFrame([
    {"endpoint": "v168", "rmse": _rmse(truth, parent_values)},
    {"endpoint": "pure_spatial", "rmse": _rmse(truth, spatial_values)},
    {"endpoint": "global_oracle_blend", "rmse": _rmse(truth, global_prediction)},
    {"endpoint": "fidelity_moe", "rmse": _rmse(truth, oof["fidelity_moe"])},
    {"endpoint": "shuffled_moe", "rmse": _rmse(truth, oof["shuffled_moe"])},
    {"endpoint": "perwell_oracle_moe", "rmse": _rmse(truth, oof["oracle_moe"])},
])
fold_rows = []
for held_fold, group in oof.groupby("fold"):
    base_rmse = _rmse(group["tvt_true"], group["v168"])
    candidate_rmse = _rmse(group["tvt_true"], group["fidelity_moe"])
    fold_rows.append({"fold": int(held_fold), "base_rmse": base_rmse,
                      "candidate_rmse": candidate_rmse, "gain": base_rmse - candidate_rmse})

base_rmse = float(summary.loc[summary["endpoint"].eq("v168"), "rmse"].iloc[0])
moe_rmse = float(summary.loc[summary["endpoint"].eq("fidelity_moe"), "rmse"].iloc[0])
shuffle_rmse = float(summary.loc[summary["endpoint"].eq("shuffled_moe"), "rmse"].iloc[0])
oracle_rmse = float(summary.loc[summary["endpoint"].eq("perwell_oracle_moe"), "rmse"].iloc[0])
alpha_spearman = float(spearmanr(predicted_alpha, y_alpha)[0])
checks = {
    "pure_spatial_rmse_le_15": float(summary.loc[summary["endpoint"].eq("pure_spatial"), "rmse"].iloc[0]) <= 15.0,
    "perwell_oracle_rmse_le_7p3": oracle_rmse <= 7.3,
    "alpha_spearman_ge_0p15": alpha_spearman >= 0.15,
    "moe_gain_ge_0p03": base_rmse - moe_rmse >= 0.03,
    "aligned_beats_shuffle_ge_0p02": shuffle_rmse - moe_rmse >= 0.02,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in fold_rows) >= 4,
}
verdict = {
    "experiment": "pure_spatial_fidelity_moe_v226",
    "stage": "fivefold_pure_geometry_spatial_expert_and_fidelity_gate",
    "expert_features": {"geometry": geometry_features, "spatial_count": len(spatial_features),
                        "total": len(features), "sha256": hashlib.sha256("\n".join(features).encode()).hexdigest()},
    "router_features": router_features,
    "global_oracle_alpha": global_alpha,
    "alpha_spearman": alpha_spearman,
    "mean_predicted_alpha": float(np.mean(predicted_alpha)),
    "summary": summary.to_dict("records"),
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> build full-train spatial expert and Kaggle mixture" if all(checks.values())
            else "FAIL -> fidelity cannot route between genuinely independent experts; close reliability MoE",
}

well_table["predicted_alpha"] = predicted_alpha
well_table["shuffled_alpha"] = shuffled_alpha
oof.to_parquet(OUT / "oof.parquet", index=False)
well_table.to_csv(OUT / "well_router.csv", index=False)
summary.to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(model_folds).to_csv(OUT / "model_folds.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)
pd.Series(features, name="feature").to_csv(OUT / "features.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")

print("\nV226 SUMMARY")
print(summary.to_string(index=False))
print("\nV226 FOLDS")
print(pd.DataFrame(fold_rows).to_string(index=False))
print("\nV226 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
