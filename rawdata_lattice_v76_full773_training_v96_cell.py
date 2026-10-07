"""V96: extend V76 scalar-dose training from 464 to 773 honest wells.

V70 has honest coarse OOF only for folds 1/2/3 because folds 4/5 supplied
its structural prior.  This cell fits a second prior on folds 1/2/3, builds
candidate panels once under that prior, and obtains leave-fold-out coarse
predictions for folds 4 and 5.  The completed five-fold endpoint then tests
the unchanged V76 Hist+Ridge scalar-dose family.  Fold4/5 are the primary
confirmation because they were absent from V76 training.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import types
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_v76_full773_training_v96"
OUT_DIR = OOF_ROOT / EXPERIMENT
PANEL_DIR = OUT_DIR / "prior123_panels"
MODEL_DIR = OUT_DIR / "coarse_models"
EVAL_DIR = OUT_DIR / "fold45_eval"
DEPLOYMENT_DIR = OUT_DIR / "deployment"
for directory in (OUT_DIR, PANEL_DIR, MODEL_DIR, EVAL_DIR, DEPLOYMENT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V66_PATH = OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet"
V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V76_BUNDLE = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "deployment" / "adaptive_coarse_dose_bundle_v76.joblib"
V76_OOF_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH, PARENT, COARSE = "tvt_true", "incumbent_6p377_replay", "coarse_crossfit"
PRIOR_FOLDS = (1, 2, 3)
NEW_FOLDS = (4, 5)
ALL_FOLDS = (1, 2, 3, 4, 5)
ITERATIONS = 67
MIN_WEIGHT, MAX_WEIGHT = 0.0, 0.60
BASE_FEATURES = (
    "maximum_distance_ft", "move_mean", "move_std", "move_rms",
    "move_abs_mean", "move_abs_p90", "move_abs_max", "move_slope_per_ft",
    "endpoint_correlation", "parent_slope_per_ft", "coarse_slope_per_ft",
    "parent_roughness", "coarse_roughness", "near_0_500_fraction",
    "near_0_500_move_rms", "near_0_500_move_mean", "mid_500_1000_fraction",
    "mid_500_1000_move_rms", "mid_500_1000_move_mean",
    "mid_1000_2000_fraction", "mid_1000_2000_move_rms",
    "mid_1000_2000_move_mean", "far_gt2000_fraction",
    "far_gt2000_move_rms", "far_gt2000_move_mean",
)
LOG_FEATURES = (
    "move_std", "move_rms", "move_abs_mean", "move_abs_p90", "move_abs_max",
    "parent_roughness", "coarse_roughness", "near_0_500_move_rms",
    "mid_500_1000_move_rms", "mid_1000_2000_move_rms", "far_gt2000_move_rms",
)
STRATA = (
    ("near_0_500", -np.inf, 500.0), ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0), ("far_gt2000", 2000.0, np.inf),
)


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v4():
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source: raise RuntimeError("V4 bootstrap marker missing")
    module = types.ModuleType("raw_lattice_v4_v96_definitions")
    module.__file__ = str(V4_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V4_SOURCE), "exec"), module.__dict__)
    module.CACHE_DIR = PANEL_DIR
    return module


def _rank_labels(mse):
    order = np.argsort(mse)
    rank = np.empty(len(mse), np.int64); rank[order] = np.arange(len(mse))
    return np.clip(30 - np.floor(31.0 * rank / max(len(mse), 1)).astype(np.int64), 0, 30)


def _training_indices(panel, well):
    n = len(panel["mse"])
    deterministic = np.argsort(np.min(panel["arm_rank"], axis=1))[: min(384, n)]
    remaining = np.setdiff1d(np.arange(n), deterministic, assume_unique=False)
    count = min(768 - len(deterministic), len(remaining))
    seed = int.from_bytes(hashlib.sha256(well.encode("ascii")).digest()[:8], "little")
    sampled = np.random.default_rng(seed).choice(remaining, count, replace=False) if count else np.empty(0, int)
    return np.unique(np.r_[deterministic, sampled]).astype(np.int64)


def _load_panel(well):
    with np.load(PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - frame[TRUTH].to_numpy(np.float64)
    e2 = error * error
    bywell = np.sqrt(pd.DataFrame({"well": frame["well"], "e2": e2}).groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {"endpoint": endpoint, "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
              "exact_sse": float(e2.sum()), "rmse": float(np.sqrt(e2.mean())), "mae": float(np.abs(error).mean()),
              "median_well_rmse": float(np.median(bywell)), "p95_well_rmse": float(np.quantile(bywell, .95))}
    for name, lower, upper in STRATA:
        mask = (distance > lower) & (distance <= upper)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(e2[mask]))) if mask.any() else np.nan
    return result


def _slope(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); x = x - x.mean()
    den = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / den) if den > 0 else 0.0


def _optimal_weight(parent, coarse, truth):
    move = coarse - parent; den = float(np.dot(move, move))
    return float(np.clip(np.dot(move, truth - parent) / den, MIN_WEIGHT, MAX_WEIGHT)) if den > 1e-12 else 0.30


def _feature_row(local):
    parent = local[PARENT].to_numpy(np.float64); coarse = local[COARSE].to_numpy(np.float64)
    truth = local[TRUTH].to_numpy(np.float64); distance = local["md_since"].to_numpy(np.float64); move = coarse - parent
    row = {"fold": int(local["fold"].iloc[0]), "well": str(local["well"].iloc[0]), "rows": int(len(local)),
           "maximum_distance_ft": float(distance.max()), "oracle_weight": _optimal_weight(parent, coarse, truth),
           "move_mean": float(move.mean()), "move_std": float(move.std()), "move_rms": float(np.sqrt(np.mean(move**2))),
           "move_abs_mean": float(np.abs(move).mean()), "move_abs_p90": float(np.quantile(np.abs(move), .90)),
           "move_abs_max": float(np.abs(move).max()), "move_slope_per_ft": _slope(distance, move),
           "endpoint_correlation": float(np.corrcoef(parent, coarse)[0, 1]) if np.std(parent) and np.std(coarse) else 0.0,
           "parent_slope_per_ft": _slope(distance, parent), "coarse_slope_per_ft": _slope(distance, coarse),
           "parent_roughness": float(np.sqrt(np.mean(np.diff(parent)**2))),
           "coarse_roughness": float(np.sqrt(np.mean(np.diff(coarse)**2)))}
    for name, lower, upper in STRATA:
        mask = (distance > lower) & (distance <= upper)
        row[f"{name}_fraction"] = float(mask.mean())
        row[f"{name}_move_rms"] = float(np.sqrt(np.mean(move[mask]**2))) if mask.any() else 0.0
        row[f"{name}_move_mean"] = float(move[mask].mean()) if mask.any() else 0.0
    return row


def _augment(features):
    output = features.loc[:, BASE_FEATURES].astype(np.float64).copy()
    for column in LOG_FEATURES: output[f"log1p_{column}"] = np.log1p(np.maximum(output[column], 0.0))
    output["roughness_ratio"] = output["coarse_roughness"] / np.maximum(output["parent_roughness"], 1e-4)
    output["slope_difference"] = output["coarse_slope_per_ft"] - output["parent_slope_per_ft"]
    values = output.to_numpy(np.float64)
    if not np.isfinite(values).all(): raise RuntimeError("Non-finite V96 features")
    return values, list(output.columns)


def _hist():
    return HistGradientBoostingRegressor(max_iter=160, learning_rate=.035, max_leaf_nodes=7,
                                         min_samples_leaf=45, l2_regularization=12.0, random_state=20260730)


def _ridge(): return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


for path in (V4_SOURCE, V66_PATH, V70_PATH, V76_BUNDLE, V76_OOF_WEIGHTS):
    if not path.is_file(): raise FileNotFoundError(path)

Q = _bootstrap_v4(); V = Q.V; B = Q.B
incumbent = pd.read_parquet(V66_PATH).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
fold_by_well = incumbent.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
if len(incumbent) != 3_783_989 or len(fold_by_well) != 773: raise RuntimeError("Invalid V66 lineage")

# Fit a prior that has never seen Fold4/5.
prior_path = OUT_DIR / "prior123.joblib"
records123 = {}
for number, well in enumerate(sorted(w for w, f in fold_by_well.items() if f in PRIOR_FOLDS), 1):
    try: records123[well] = V["_load_record"](well)
    except Exception as exc: print("V96 record exclusion", well, repr(exc), flush=True)
    if number % 50 == 0: print(f"V96 load prior records {number}/464", flush=True)
if prior_path.is_file(): prior = joblib.load(prior_path)
else:
    prior = V["_fit_prior"](list(records123.values()), Q.SEGMENT_LENGTH_FT)
    joblib.dump(prior, prior_path, compress=3)

# Build resumable ranker panels under the Fold123-only prior.
eligible_wells = []
for number, well in enumerate(sorted(fold_by_well), 1):
    panel_path = PANEL_DIR / f"{well}.npz"
    try:
        record = records123.get(well) or V["_load_record"](well)
        Q._build_panel(record, prior)
        eligible_wells.append(well)
    except Exception as exc:
        print("V96 panel exclusion", well, repr(exc), flush=True)
    if number % 20 == 0 or number == len(fold_by_well):
        print(f"V96 panels {number}/{len(fold_by_well)} eligible={len(eligible_wells)} elapsed_min={(time.time()-STARTED)/60:.1f}", flush=True)

metadata = json.loads((PANEL_DIR / f"{eligible_wells[0]}.json").read_text(encoding="utf-8"))
feature_names = metadata["feature_names"]
new_parts, reports = [], []
for heldout_fold in NEW_FOLDS:
    model_path = MODEL_DIR / f"ranker_leave_fold{heldout_fold}_out.txt"
    if model_path.is_file(): booster = lgb.Booster(model_file=str(model_path))
    else:
        train_wells = [w for w in eligible_wells if fold_by_well[w] != heldout_fold]
        x_parts, y_parts, groups = [], [], []
        for number, well in enumerate(train_wells, 1):
            panel = _load_panel(well); indices = _training_indices(panel, well)
            x_parts.append(panel["features"][indices]); y_parts.append(_rank_labels(panel["mse"][indices])); groups.append(len(indices))
            if number % 100 == 0: print(f"V96 Fold{heldout_fold} matrix {number}/{len(train_wells)}", flush=True)
        x = np.concatenate(x_parts); y = np.concatenate(y_parts)
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=(1,8,32), n_estimators=ITERATIONS,
            learning_rate=.025, num_leaves=31, min_child_samples=80, subsample=.85, colsample_bytree=.80,
            reg_alpha=.2, reg_lambda=2.0, random_state=20260721, n_jobs=max(1,min(16,os.cpu_count() or 1)), verbosity=-1)
        model.fit(x, y, group=groups); model.booster_.save_model(str(model_path)); booster = model.booster_
        del x, y, x_parts, y_parts, model; gc.collect()
    valid_wells = sorted(w for w, f in fold_by_well.items() if f == heldout_fold)
    for number, well in enumerate(valid_wells, 1):
        cache = EVAL_DIR / f"{well}.parquet"
        if cache.is_file(): part = pd.read_parquet(cache); report = {"well": well, "fold": heldout_fold, "status": "cached"}
        else:
            local = incumbent[incumbent["well"].astype(str).eq(well)].copy()
            try:
                record = V["_load_record"](well)
                paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                    evidence, names = Q._candidate_evidence(record, prior, paths, segments, arm_rank, arm_score_z, source_mask)
                if list(names) != list(feature_names): raise RuntimeError("feature schema mismatch")
                chosen = int(np.argmax(booster.predict(evidence))); eval_rows = record["eval_rows"]
                coarse = B["_path_prediction"](record, prior, paths[chosen], segments)[eval_rows]
                if len(local) != len(coarse): raise RuntimeError(f"row mismatch {len(local)}/{len(coarse)}")
                local[COARSE] = np.asarray(coarse, np.float32); status = "built"; reason = ""
            except Exception as exc:
                local[COARSE] = local[PARENT].to_numpy(np.float32); status = "parent_fallback"; reason = repr(exc)
            part = local[["id","well","row_idx","fold",TRUTH,"md_since",PARENT,COARSE]].copy()
            part.to_parquet(cache, index=False, compression="zstd"); report = {"well":well,"fold":heldout_fold,"status":status,"reason":reason}
        new_parts.append(part); reports.append(report)
        if number % 10 == 0 or number == len(valid_wells): print(f"V96 Fold{heldout_fold} evaluate {number}/{len(valid_wells)}", flush=True)

old = pd.read_parquet(V70_PATH, columns=["id","well","row_idx","fold",TRUTH,"md_since",PARENT,COARSE])
complete = pd.concat([old, *new_parts], ignore_index=True).sort_values(["fold","well","row_idx"], kind="stable").reset_index(drop=True)
if len(complete) != 3_783_989 or complete["well"].nunique() != 773 or complete["id"].duplicated().any():
    raise RuntimeError(f"Incomplete V96 endpoint: rows={len(complete)} wells={complete['well'].nunique()}")
complete.to_parquet(OUT_DIR / "fivefold_coarse_oof.parquet", index=False, compression="zstd")
pd.DataFrame(reports).to_csv(OUT_DIR / "fold45_report.csv", index=False)

features = pd.DataFrame([_feature_row(local) for _, local in complete.groupby(["fold","well"], sort=False)])
features = features.sort_values(["fold","well"], kind="stable").reset_index(drop=True)
x, feature_names = _augment(features); y = features["oracle_weight"].to_numpy(np.float64)
sample_weight = np.sqrt(features["rows"].to_numpy(np.float64)); expanded = np.full(len(features), np.nan)
for fold in ALL_FOLDS:
    valid = features["fold"].to_numpy() == fold; train = ~valid
    hist, ridge = _hist(), _ridge(); hist.fit(x[train], y[train], sample_weight=sample_weight[train]); ridge.fit(x[train], y[train])
    expanded[valid] = np.clip(.5*hist.predict(x[valid]) + .5*ridge.predict(x[valid]), MIN_WEIGHT, MAX_WEIGHT)
    print("V96 DOSE FOLD", {"fold":fold,"train_wells":int(train.sum()),"valid_wells":int(valid.sum()),"mean":float(expanded[valid].mean())}, flush=True)

# Apply the original 464-well deployment model to all 773 as the matched baseline.
old_bundle = joblib.load(V76_BUNDLE)
baseline_weights = np.clip(old_bundle["hist_fraction"]*old_bundle["hist_model"].predict(x) + old_bundle["ridge_fraction"]*old_bundle["ridge_model"].predict(x), MIN_WEIGHT, MAX_WEIGHT)
old_oof = pd.read_parquet(V76_OOF_WEIGHTS, columns=["fold", "well", "ensemble_weight"])
old_oof_map = dict(zip(old_oof["well"].astype(str), old_oof["ensemble_weight"].astype(float)))
honest_old = features["fold"].isin(PRIOR_FOLDS).to_numpy()
baseline_weights[honest_old] = features.loc[honest_old, "well"].astype(str).map(old_oof_map).to_numpy(np.float64)
if not np.isfinite(baseline_weights).all():
    raise RuntimeError("Incomplete matched V76 baseline weights")
weight_tables = {"v76_464_training": baseline_weights, "v96_773_crossfit": expanded}
predictions = {}
for name, weights in weight_tables.items():
    mapping = dict(zip(features["well"].astype(str), weights)); row_weights = complete["well"].astype(str).map(mapping).to_numpy(np.float64)
    parent = complete[PARENT].to_numpy(np.float64); predictions[name] = parent + row_weights*(complete[COARSE].to_numpy(np.float64)-parent)

summary_rows=[]
for fold in (0,*ALL_FOLDS):
    mask = np.ones(len(complete),bool) if fold==0 else complete["fold"].to_numpy()==fold; local=complete.loc[mask]
    for name,pred in predictions.items(): summary_rows.append({"fold":fold,**_metric(local,pred[mask],name)})
summary=pd.DataFrame(summary_rows); summary.to_csv(OUT_DIR/"matched_summary.csv",index=False)

def _row(fold,name): return summary[summary["fold"].eq(fold)&summary["endpoint"].eq(name)].iloc[0]
fold_gains={str(f):float(_row(f,"v76_464_training")["rmse"]-_row(f,"v96_773_crossfit")["rmse"]) for f in ALL_FOLDS}
primary={str(f):fold_gains[str(f)] for f in NEW_FOLDS}; base=_row(0,"v76_464_training"); cand=_row(0,"v96_773_crossfit")
checks={"fold4_gain_positive":primary["4"]>0,"fold5_gain_positive":primary["5"]>0,
        "fold45_mean_gain_ge_0p02":float(np.mean(list(primary.values())))>=.02,
        "overall_gain_positive":base["rmse"]>cand["rmse"],"overall_far_gain_nonnegative":base["far_gt2000_rmse"]>=cand["far_gt2000_rmse"]}
strict=bool(all(checks.values())); bundle_path=DEPLOYMENT_DIR/"adaptive_coarse_dose_bundle_v96.joblib"
if strict:
    hist,ridge=_hist(),_ridge(); hist.fit(x,y,sample_weight=sample_weight); ridge.fit(x,y)
    joblib.dump({"experiment":EXPERIMENT,"hist_model":hist,"ridge_model":ridge,"hist_fraction":.5,"ridge_fraction":.5,
                 "min_weight":MIN_WEIGHT,"max_weight":MAX_WEIGHT,"base_features":list(BASE_FEATURES),
                 "log_features":list(LOG_FEATURES),"feature_names":feature_names,"training_wells":773,"training_folds":list(ALL_FOLDS)},bundle_path,compress=3)
verdict={"experiment":EXPERIMENT,"stage":"Fold123_prior_Fold45_primary_confirmation_complete_fivefold_scalar_dose",
         "coarse_protocol":{"fold123":"existing V70 OOF","fold45":"prior123 and leave-one-fold-out rankers"},
         "dose_family":"unchanged V76 50/50 Hist+Ridge","coverage":{"rows":len(complete),"wells":complete["well"].nunique()},
         "fold_gains":fold_gains,"primary_fold45_gains":primary,"overall_gain":float(base["rmse"]-cand["rmse"]),
         "far_gain":float(base["far_gt2000_rmse"]-cand["far_gt2000_rmse"]),"checks":checks,"strict_pass":strict,
         "bundle":str(bundle_path) if strict else None,"elapsed_minutes":(time.time()-STARTED)/60,
         "next":"PASS -> deploy V96 bundle after V69 coarse" if strict else "FAIL -> retain V76 trained on 464 wells"}
_write_json(OUT_DIR/"verdict.json",verdict)
print("\nV96 MATCHED SUMMARY"); print(summary.to_string(index=False),flush=True)
print("\nV96 FINAL VERDICT"); print(json.dumps(_clean(verdict),indent=2,allow_nan=False),flush=True)
print(f"saved: {OUT_DIR}",flush=True)
