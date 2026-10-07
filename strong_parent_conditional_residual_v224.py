"""V224: condition a fold-clean residual model directly on the strong V168 parent.

V217/V219/V223 predict TVT from the likelihood-PF representation and are only
blended with V168 afterwards.  This experiment asks a different question: can
the same legal evidence explain the *remaining V168 error* when the parent and
its disagreements with each trajectory family are explicit model inputs?
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


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
BASE = OOF / "likpf_residual_gbdt_full5_v215"
SPATIAL = OOF / "foldclean_full_fleongg_lgb_v0"
PARENT_PATH = OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
OUT = OOF / "strong_parent_conditional_residual_v224"
MODELS = OUT / "models"
for path in (OUT, MODELS):
    path.mkdir(parents=True, exist_ok=True)

SEED = 224
N_FOLDS = 5
WARM_TAU = 120.0
WARM_FLOOR = 0.12
CORRECTION_CLIP = 40.0
SMOOTH_WINDOWS = (1, 61, 181)
CAPS = (2.0, 4.0, 8.0, 16.0)
DOSES = np.round(np.arange(0.0, 1.0001, 0.10), 2)


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
    BASE / "well_metadata.csv",
    BASE / "features.csv",
    SPATIAL / "well_metadata.csv",
    SPATIAL / "all_features.csv",
    SPATIAL / "no_spatial_features.csv",
    PARENT_PATH,
)
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

base_meta = pd.read_csv(BASE / "well_metadata.csv", dtype={"well": str})
spatial_meta = pd.read_csv(SPATIAL / "well_metadata.csv", dtype={"well": str})
fold_check = base_meta[["well", "fold"]].merge(
    spatial_meta[["well", "fold"]], on="well", how="outer",
    suffixes=("_base", "_spatial"), validate="one_to_one",
)
if len(fold_check) != 773 or not fold_check["fold_base"].eq(fold_check["fold_spatial"]).all():
    raise RuntimeError("V215 and fold-clean spatial fold maps differ")

base_features = pd.read_csv(BASE / "features.csv")["feature"].astype(str).tolist()
all_spatial = pd.read_csv(SPATIAL / "all_features.csv")["feature"].astype(str).tolist()
nonspatial = set(pd.read_csv(SPATIAL / "no_spatial_features.csv")["feature"].astype(str))
spatial_features = [name for name in all_spatial if name not in nonspatial]

parent_raw = pd.read_parquet(PARENT_PATH)
if "id" not in parent_raw:
    well_column = next(name for name in ("well", "well_id") if name in parent_raw)
    row_column = next(name for name in ("row_idx", "row_index") if name in parent_raw)
    parent_raw["id"] = (
        parent_raw[well_column].astype(str) + "_" +
        parent_raw[row_column].astype(int).astype(str)
    )
parent_prediction = "metadata_analog_w0p05"
if parent_prediction not in parent_raw:
    raise KeyError(f"Missing V168 prediction column {parent_prediction!r}")
parent_columns = ["id", parent_prediction]
for candidate in ("tvt_true", "TVT", "target"):
    if candidate in parent_raw:
        parent_columns.append(candidate)
        break
parent_table = parent_raw[parent_columns].copy().rename(columns={parent_prediction: "v168"})
if parent_table["id"].duplicated().any() or len(parent_table) != 3_783_989:
    raise RuntimeError("V168 parent does not have the canonical full-row key set")
parent_lookup = parent_table.set_index("id")


def _paths(root, well, full):
    suffix = "full" if full else "sample"
    return root / "features" / f"{well}__{suffix}.pkl"


def _trajectory_columns(columns):
    confirmed = {
        "likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12",
        "likpf_mean", "pf_ancc_delta", "pf_z_delta", "beam_cons_d",
        "beam_mean_d", "beam_med_d", "sc8_d", "sc15_d", "sc25_d",
        "sc_cons_d", "sc_ens_d", "hyb_d", "sig_mean_d", "form_mean_d",
        "tvt_dense_d", "tvt_densew_d", "tvt_dense50_d", "beam_loose_d",
        "beam_vcons_d", "beam_sm5_d", "beam_vloose_d", "beam_mid_d",
        "beam_stiff_d", "spatial_ancc_d",
    }
    selected = [name for name in columns if name.lower() in confirmed]
    selected += [
        name for name in columns
        if name.lower().startswith(("tvtf_", "tvtfw_", "tvtf50_"))
    ]
    return list(dict.fromkeys(selected))


trajectory_features = _trajectory_columns(base_features + spatial_features)
derived_features = ["v168_delta", "v168_minus_likpf5", "evidence_center_minus_v168",
                    "evidence_spread", "evidence_iqr", "evidence_count"]
derived_features += [f"v168_gap__{name}" for name in trajectory_features]
features = base_features + spatial_features + derived_features


def _as_absolute(frame, name):
    value = frame[name].to_numpy(np.float32)
    lower = name.lower()
    if lower.startswith("likpf_scale_") or lower == "likpf_mean" or lower.startswith(("tvtf_", "tvtfw_", "tvtf50_")):
        return value
    return frame["last_known_tvt"].to_numpy(np.float32) + value


def _load_one(well, full):
    base = pd.read_pickle(_paths(BASE, well, full))
    spatial = pd.read_pickle(_paths(SPATIAL, well, full))
    key = "id" if "id" in base and "id" in spatial else "_row_idx"
    aligned = spatial.set_index(key)[spatial_features].reindex(base[key].to_numpy())
    if aligned.isna().all(axis=1).any():
        raise RuntimeError(f"{well}: incomplete spatial alignment")
    base[spatial_features] = aligned.to_numpy(np.float32)
    parent = parent_lookup.reindex(base["id"].astype(str).to_numpy())
    if parent["v168"].isna().any():
        raise RuntimeError(f"{well}: incomplete V168 alignment")
    base["v168"] = parent["v168"].to_numpy(np.float32)
    last = base["last_known_tvt"].to_numpy(np.float32)
    base["v168_delta"] = base["v168"].to_numpy(np.float32) - last
    if "likpf_scale_5" in base:
        base["v168_minus_likpf5"] = base["v168"] - base["likpf_scale_5"]
    else:
        base["v168_minus_likpf5"] = 0.0
    evidence = []
    for name in trajectory_features:
        absolute = _as_absolute(base, name)
        base[f"v168_gap__{name}"] = base["v168"].to_numpy(np.float32) - absolute
        evidence.append(absolute)
    if evidence:
        matrix = np.column_stack(evidence).astype(np.float32)
        finite = np.isfinite(matrix)
        matrix = np.where(finite, matrix, np.nan)
        center = np.nanmedian(matrix, axis=1)
        q25 = np.nanpercentile(matrix, 25, axis=1)
        q75 = np.nanpercentile(matrix, 75, axis=1)
        base["evidence_center_minus_v168"] = center - base["v168"].to_numpy(np.float32)
        base["evidence_spread"] = np.nanstd(matrix, axis=1)
        base["evidence_iqr"] = q75 - q25
        base["evidence_count"] = finite.sum(axis=1).astype(np.float32)
    else:
        for name in derived_features[2:6]:
            base[name] = 0.0
    return base


def _load(wells, full):
    return pd.concat([_load_one(str(well), full) for well in wells], ignore_index=True, sort=False)


def _prepare(frame, training):
    frame = frame.copy()
    truth = frame["last_known_tvt"].to_numpy(float) + frame["target"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(frame["md_since"].to_numpy(float), 0.0) / WARM_TAU)
    frame["parent_residual_target"] = np.clip(
        (truth - frame["v168"].to_numpy(float)) / np.maximum(warm, WARM_FLOOR),
        -CORRECTION_CLIP, CORRECTION_CLIP,
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
        objective="regression", boosting_type="gbdt", n_estimators=2400,
        learning_rate=0.02, num_leaves=48, min_child_samples=120,
        subsample=0.82, subsample_freq=1, colsample_bytree=0.72,
        reg_alpha=0.20, reg_lambda=18.0, max_bin=255, n_jobs=-1,
        random_state=SEED + fold, verbosity=-1,
    )
    if os.environ.get("V224_CPU", "0") != "1":
        params["device_type"] = "gpu"
    return LGBMRegressor(**params)


def _fit(train, valid, fold):
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["parent_residual_target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(160, verbose=False), log_evaluation(100)],
    )
    model = _new_model(fold)
    try:
        model.fit(_matrix(train), train["parent_residual_target"].to_numpy(float), **kwargs)
    except Exception as exc:
        if "gpu" not in str(exc).lower() and "opencl" not in str(exc).lower():
            raise
        print("V224 GPU fallback", repr(exc), flush=True)
        os.environ["V224_CPU"] = "1"
        model = _new_model(fold)
        model.fit(_matrix(train), train["parent_residual_target"].to_numpy(float), **kwargs)
    return model


def _smooth_by_well(frame, values, window):
    if window <= 1:
        return np.asarray(values, float).copy()
    output = np.asarray(values, float).copy()
    for indices in frame.groupby("well", sort=False).indices.values():
        indices = np.asarray(indices, np.int64)
        n = len(indices)
        actual = min(window, n if n % 2 else n - 1)
        if actual >= 5:
            output[indices] = savgol_filter(output[indices], actual, 3)
    return output


parts = []
fold_reports = []
for fold in range(1, N_FOLDS + 1):
    train_wells = base_meta.loc[~base_meta["fold"].eq(fold), "well"].astype(str).tolist()
    valid_wells = base_meta.loc[base_meta["fold"].eq(fold), "well"].astype(str).tolist()
    train = _prepare(_load(train_wells, full=False), training=True)
    valid = _prepare(_load(valid_wells, full=True), training=False)
    valid = valid.sort_values(["well", "_row_idx"], kind="stable").reset_index(drop=True)
    print(f"V224 fold {fold}: train={len(train)} valid={len(valid)} features={len(features)}", flush=True)
    model = _fit(train, valid, fold)
    raw = model.predict(_matrix(valid), num_iteration=model.best_iteration_)
    warm = 1.0 - np.exp(-np.maximum(valid["md_since"].to_numpy(float), 0.0) / WARM_TAU)
    truth = valid["last_known_tvt"].to_numpy(float) + valid["target"].to_numpy(float)
    part = pd.DataFrame({
        "id": valid["id"].astype(str), "well": valid["well"].astype(str),
        "fold": fold, "md_since": valid["md_since"].to_numpy(float),
        "tvt_true": truth, "v168": valid["v168"].to_numpy(float),
    })
    for window in SMOOTH_WINDOWS:
        part[f"move_s{window}"] = warm * _smooth_by_well(valid, raw, window)
    fold_reports.append({
        "fold": fold, "rows": len(valid), "wells": len(valid_wells),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "parent_rmse": _rmse(truth, part["v168"]),
        "raw_full_dose_rmse": _rmse(truth, part["v168"] + part["move_s1"]),
    })
    print("V224 FOLD", fold_reports[-1], flush=True)
    parts.append(part)
    joblib.dump(model, MODELS / f"fold{fold}.joblib")
    del train, valid, model, raw, part
    gc.collect()

oof = pd.concat(parts, ignore_index=True)
if len(oof) != 3_783_989 or oof["well"].nunique() != 773 or oof["id"].duplicated().any():
    raise RuntimeError({"rows": len(oof), "wells": oof["well"].nunique()})

truth = oof["tvt_true"].to_numpy(float)
parent = oof["v168"].to_numpy(float)
recipes = []
for window in SMOOTH_WINDOWS:
    raw_move = oof[f"move_s{window}"].to_numpy(float)
    for cap in CAPS:
        move = np.clip(raw_move, -cap, cap)
        for dose in DOSES:
            prediction = parent + dose * move
            row = {"window": window, "cap": cap, "dose": float(dose), "rmse": _rmse(truth, prediction)}
            for fold in range(1, 6):
                mask = oof["fold"].to_numpy(int) == fold
                row[f"fold{fold}_rmse"] = _rmse(truth[mask], prediction[mask])
            recipes.append(row)
scan = pd.DataFrame(recipes).sort_values("rmse").reset_index(drop=True)

# For each held fold, select one recipe using only the other four OOF folds.
nested = parent.copy()
nested_rows = []
fold_array = oof["fold"].to_numpy(int)
for held_fold in range(1, 6):
    train_mask = fold_array != held_fold
    valid_mask = ~train_mask
    best = None
    for row in recipes:
        move = np.clip(oof[f"move_s{int(row['window'])}"].to_numpy(float), -row["cap"], row["cap"])
        score = _rmse(truth[train_mask], parent[train_mask] + row["dose"] * move[train_mask])
        key = (score, row["dose"], row["cap"], row["window"])
        if best is None or key < best[0]:
            best = (key, row, move)
    _, row, move = best
    nested[valid_mask] = parent[valid_mask] + row["dose"] * move[valid_mask]
    parent_rmse = _rmse(truth[valid_mask], parent[valid_mask])
    candidate_rmse = _rmse(truth[valid_mask], nested[valid_mask])
    nested_rows.append({
        "fold": held_fold, "window": int(row["window"]), "cap": float(row["cap"]),
        "dose": float(row["dose"]), "parent_rmse": parent_rmse,
        "candidate_rmse": candidate_rmse, "gain": parent_rmse - candidate_rmse,
    })

parent_rmse = _rmse(truth, parent)
nested_rmse = _rmse(truth, nested)
checks = {
    "coverage_773": len(oof) == 3_783_989 and oof["well"].nunique() == 773,
    "nested_gain_ge_0p05": parent_rmse - nested_rmse >= 0.05,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in nested_rows) >= 4,
    "minimum_fold_gain_ge_minus0p03": min(row["gain"] for row in nested_rows) >= -0.03,
    "nonzero_dose_folds_ge_4": sum(row["dose"] > 0 for row in nested_rows) >= 4,
}
verdict = {
    "experiment": "strong_parent_conditional_residual_v224",
    "stage": "fivefold_V168_conditioned_foldclean_residual_gateway",
    "changed_information": "V168 parent state plus parent-to-trajectory disagreement features",
    "training_target": "truth minus V168, anchor-normalized",
    "features": {"base": len(base_features), "spatial": len(spatial_features),
                 "trajectory_disagreements": len(trajectory_features), "total": len(features),
                 "sha256": hashlib.sha256("\n".join(features).encode()).hexdigest()},
    "values": {"v168_rmse": parent_rmse, "nested_rmse": nested_rmse,
               "nested_gain": parent_rmse - nested_rmse},
    "folds": nested_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> build exact V168-conditioned Kaggle inference" if all(checks.values())
            else "FAIL -> current legal evidence cannot explain strong-parent residual; stop rowwise GBDT",
}

oof["nested_candidate"] = nested
oof.to_parquet(OUT / "oof.parquet", index=False)
scan.to_csv(OUT / "recipe_scan.csv", index=False)
pd.DataFrame(fold_reports).to_csv(OUT / "model_folds.csv", index=False)
pd.DataFrame(nested_rows).to_csv(OUT / "nested_folds.csv", index=False)
pd.Series(features, name="feature").to_csv(OUT / "features.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")

print("\nV224 FIXED TOP 20")
print(scan.head(20).to_string(index=False))
print("\nV224 NESTED FOLDS")
print(pd.DataFrame(nested_rows).to_string(index=False))
print("\nV224 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
