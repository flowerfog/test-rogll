"""M342-B1: official-cut-fraction specialist gateway for the 6.342 mainline.

This experiment reuses the fold-independent, non-spatial K3 feature cache. It
tests whether one global row model is hiding a protocol mismatch across early,
middle, and late prediction starts. Fold 3 freezes model iterations and blend
dose; Fold 2 is read only unless the Fold 3 mechanism gate passes. Fold 1 is
never loaded.
"""

from pathlib import Path
import gc
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
CACHE_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
FEATURE_DIR = CACHE_DIR / "features"
DONOR_FEATURE_DIR = (
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "features"
)
V168_PATH = OOF_ROOT / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
OUT_DIR = OOF_ROOT / "main342_cut_fraction_specialist_gateway_b1"
MODEL_DIR = OUT_DIR / "models"

TAGS = ("official", "pseudo1", "pseudo2")
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
N_BINS = 3
RANDOM_STATE = 20260804
DOSES = np.round(np.arange(0.0, 0.3001, 0.025), 3)

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

LGB_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "n_estimators": 2400,
    "learning_rate": 0.025,
    "num_leaves": 64,
    "min_child_samples": 80,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.10,
    "reg_lambda": 10.0,
    "max_bin": 255,
    "n_jobs": max(1, min(12, os.cpu_count() or 1)),
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

SPATIAL_PREFIXES = (
    "tvtF_", "tvtFw_", "tvtF50_", "bw_", "bww_", "bw50_",
    "bw_early_", "bw_mid_", "frm_rmse_", "dense_", "tvt_dense",
)
SPATIAL_EXACT = {
    "form_mean_d", "form_std_d", "form_rng_d", "spatial_ancc_d",
    "spatial_knn_dist", "pf_vs_spatial", "pf_vs_dense",
    "spatial_vs_dense", "beam_vs_spatial", "sig_std", "sig_mean_d",
}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _feature_paths(well, tag):
    sample = FEATURE_DIR / f"{well}__{tag}__sample.pkl"
    full = FEATURE_DIR / f"{well}__official_full.pkl" if tag == "official" else None
    if tag == "official" and not (sample.is_file() and full.is_file()):
        donor_sample = DONOR_FEATURE_DIR / f"{well}__official__sample.pkl"
        donor_full = DONOR_FEATURE_DIR / f"{well}__official_full.pkl"
        if donor_sample.is_file() and donor_full.is_file():
            return donor_sample, donor_full
    return sample, full


def _is_spatial(column):
    return column in SPATIAL_EXACT or column.startswith(SPATIAL_PREFIXES)


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    pred = np.asarray(prediction, np.float64)
    error = pred - truth
    well_rmse = pd.Series(
        {
            well: _rmse(local["tvt_true"], pred[local.index.to_numpy()])
            for well, local in frame.groupby("well", sort=False)
        }
    )
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": _rmse(truth, pred),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "far_gt2000_rmse": _rmse(truth[far], pred[far]) if far.any() else np.nan,
    }


def _matrix(frame, features):
    return frame.reindex(columns=features).to_numpy(np.float32)


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    md_since = frame["md_since"].to_numpy(np.float64)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    lik = frame.get(PP_LIKPF_COL, frame["pf_ancc"]).to_numpy(np.float64)
    lik = np.where(np.isfinite(lik), lik, last)
    prediction = last + PP_W_MODEL * PP_ALPHA * warm * model_delta + (1.0 - PP_W_MODEL) * (lik - last)
    output = prediction.copy()
    for _, positions in frame.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        n = len(positions)
        window = min(PP_SG_WIN, n)
        if window % 2 == 0:
            window -= 1
        if window >= PP_SG_POLY + 2:
            output[positions] = savgol_filter(prediction[positions], window, PP_SG_POLY)
    return output


def _fit(train, valid, features, weight_column, label, iterations=None):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(LGB_PARAMS)
    if iterations is not None:
        params["n_estimators"] = max(10, int(iterations))
    model = LGBMRegressor(**params)
    kwargs = {
        "sample_weight": train[weight_column].to_numpy(np.float64),
    }
    if valid is not None:
        kwargs.update(
            eval_set=[(_matrix(valid, features), valid["target"].to_numpy(np.float64))],
            eval_metric="rmse",
            callbacks=[early_stopping(160, verbose=False), log_evaluation(100)],
        )
    model.fit(_matrix(train, features), train["target"].to_numpy(np.float64), **kwargs)
    joblib.dump(model, MODEL_DIR / f"{label}.joblib")
    best = getattr(model, "best_iteration_", None)
    return model, int(best if best and best > 0 else params["n_estimators"])


def _bin(values, edges):
    return np.digitize(np.asarray(values, np.float64), np.asarray(edges, np.float64), right=False)


def _load_samples(wells, manifest, features=None):
    rows = []
    wanted = set(map(str, wells))
    local_manifest = manifest[manifest["well"].isin(wanted)]
    for number, item in enumerate(local_manifest.itertuples(index=False), 1):
        sample_path, _ = _feature_paths(item.well, item.tag)
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        frame = pd.read_pickle(sample_path)
        frame["well"] = str(item.well)
        frame["_cut_tag"] = str(item.tag)
        frame["cut_visible_frac"] = float(item.actual_frac)
        frame["_cut_bin"] = int(item.cut_bin)
        official_rows = float(frame["_official_n_eval"].iloc[0])
        frame["_global_weight"] = (
            {"official": 0.40, "pseudo1": 0.30, "pseudo2": 0.30}[item.tag]
            * official_rows / max(1, len(frame))
        )
        frame["_specialist_weight"] = (
            official_rows / max(1, int(item.states_in_bin)) / max(1, len(frame))
        )
        if features is not None:
            keep = list(dict.fromkeys(
                list(features) + [
                    "well", "target", "_global_weight", "_specialist_weight", "_cut_bin"
                ]
            ))
            for column in keep:
                if column not in frame:
                    frame[column] = np.nan
            frame = frame[keep]
        rows.append(frame)
        if number % 300 == 0:
            print(f"B1 load samples {number}/{len(local_manifest)}", flush=True)
    if not rows:
        raise RuntimeError("No K3 training samples loaded")
    return pd.concat(rows, ignore_index=True, sort=False)


def _load_full(wells, official_manifest, base):
    rows = []
    frac_by_well = dict(zip(official_manifest["well"], official_manifest["actual_frac"]))
    for number, well in enumerate(map(str, wells), 1):
        _, path = _feature_paths(well, "official")
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path).reset_index(drop=True)
        frame["well"] = well
        frame["cut_visible_frac"] = float(frac_by_well[well])
        rows.append(frame)
        if number % 50 == 0:
            print(f"B1 load official {number}/{len(wells)}", flush=True)
    frame = pd.concat(rows, ignore_index=True, sort=False)
    if "id" not in frame:
        frame["id"] = frame["well"] + "_" + frame["_row_idx"].astype(int).astype(str)
    frame["id"] = frame["id"].astype(str)
    aligned = frame.merge(
        base[["id", "fold", "tvt_true", "md_since", "base_v168"]],
        on="id", how="inner", validate="one_to_one", suffixes=("", "_base"), sort=False,
    )
    if aligned["well"].nunique() != len(wells):
        raise RuntimeError(
            f"Official alignment retained {aligned['well'].nunique()}/{len(wells)} wells"
        )
    feature_truth = aligned["target"].to_numpy(np.float64) + aligned["last_known_tvt"].to_numpy(np.float64)
    gap = np.max(np.abs(feature_truth - aligned["tvt_true"].to_numpy(np.float64)))
    if not np.isfinite(gap) or gap > 0.01:
        raise RuntimeError(f"Truth alignment gap {gap}")
    return aligned.reset_index(drop=True)


def _predict_models(frame, global_model, specialists, features, edges, rotated=False):
    matrix = _matrix(frame, features)
    global_delta = global_model.predict(matrix)
    specialist_deltas = np.column_stack([model.predict(matrix) for model in specialists])
    bins = _bin(frame["cut_visible_frac"], edges)
    chosen = (bins + 1) % N_BINS if rotated else bins
    specialist_delta = specialist_deltas[np.arange(len(frame)), chosen]
    return _postprocess(frame, global_delta), _postprocess(frame, specialist_delta)


def _dose_scan(frame, endpoints):
    truth = frame["tvt_true"].to_numpy(np.float64)
    base = frame["base_v168"].to_numpy(np.float64)
    records = []
    for name, endpoint in endpoints.items():
        for dose in DOSES:
            prediction = base + float(dose) * (np.asarray(endpoint) - base)
            metrics = _metrics(frame, prediction, name)
            records.append({
                "endpoint": name,
                "dose": float(dose),
                "rmse": metrics["rmse"],
                "far_gt2000_rmse": metrics["far_gt2000_rmse"],
                "gain": _rmse(truth, base) - metrics["rmse"],
            })
    return pd.DataFrame(records)


def _select_dose(scan, endpoint):
    local = scan[scan["endpoint"].eq(endpoint)].sort_values(["rmse", "dose"])
    return float(local.iloc[0]["dose"])


def _evaluate_stage(label, train_folds, held_fold, base, manifest, features, iterations=None):
    train_wells = sorted(base.loc[base["fold"].isin(train_folds), "well"].unique())
    held_wells = sorted(base.loc[base["fold"].eq(held_fold), "well"].unique())
    train_manifest = manifest[manifest["well"].isin(train_wells)].copy()
    edges = np.quantile(train_manifest["actual_frac"], [1 / 3, 2 / 3])
    train_manifest["cut_bin"] = _bin(train_manifest["actual_frac"], edges)
    state_counts = train_manifest.groupby(["well", "cut_bin"]).size().rename("states_in_bin")
    train_manifest = train_manifest.join(state_counts, on=["well", "cut_bin"])
    official = manifest[manifest["tag"].eq("official")].copy()
    official["cut_bin"] = _bin(official["actual_frac"], edges)

    train = _load_samples(train_wells, train_manifest, features)
    held = _load_full(held_wells, official, base)
    held["_cut_bin"] = _bin(held["cut_visible_frac"], edges)
    print(
        "M342-B1 STAGE",
        {"label": label, "train_wells": len(train_wells), "held_wells": len(held_wells),
         "edges": edges.tolist(), "train_rows": len(train), "held_rows": len(held)},
        flush=True,
    )

    if iterations is None:
        global_model, global_iteration = _fit(
            train, held, features, "_global_weight", f"{label}_global"
        )
    else:
        global_model, global_iteration = _fit(
            train, None, features, "_global_weight", f"{label}_global", iterations["global"]
        )
    specialists = []
    specialist_iterations = []
    for bin_id in range(N_BINS):
        local_train = train[train["_cut_bin"].eq(bin_id)]
        local_valid = held[held["_cut_bin"].eq(bin_id)]
        if local_train["well"].nunique() < 120 or local_valid["well"].nunique() < 25:
            raise RuntimeError(
                f"Bin {bin_id}: train/valid wells "
                f"{local_train['well'].nunique()}/{local_valid['well'].nunique()}"
            )
        if iterations is None:
            model, best_iteration = _fit(
                local_train, local_valid, features, "_specialist_weight",
                f"{label}_specialist{bin_id}",
            )
        else:
            model, best_iteration = _fit(
                local_train, None, features, "_specialist_weight",
                f"{label}_specialist{bin_id}", iterations[f"specialist{bin_id}"],
            )
        specialists.append(model)
        specialist_iterations.append(best_iteration)

    pred_global, pred_specialist = _predict_models(
        held, global_model, specialists, features, edges, rotated=False
    )
    _, pred_rotated = _predict_models(
        held, global_model, specialists, features, edges, rotated=True
    )
    output_iterations = {"global": global_iteration}
    output_iterations.update({f"specialist{i}": value for i, value in enumerate(specialist_iterations)})
    del train, global_model, specialists
    gc.collect()
    return held, pred_global, pred_specialist, pred_rotated, edges, output_iterations


started = time.time()
for required in (CACHE_DIR / "cut_manifest.csv", FEATURE_DIR, V168_PATH):
    if not required.exists():
        raise FileNotFoundError(required)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

base = pd.read_parquet(V168_PATH)
required_base = {"id", "well", "fold", "tvt_true", "md_since", "metadata_analog_w0p05"}
missing = required_base.difference(base.columns)
if missing:
    raise RuntimeError(f"V168 missing columns: {sorted(missing)}")
base = base[list(required_base)].copy()
base["id"] = base["id"].astype(str)
base["well"] = base["well"].astype(str)
base = base.rename(columns={"metadata_analog_w0p05": "base_v168"})
if base["well"].nunique() != 773 or base["id"].duplicated().any():
    raise RuntimeError("V168 is not the complete unique 773-well OOF")

manifest = pd.read_csv(CACHE_DIR / "cut_manifest.csv")
manifest["well"] = manifest["well"].astype(str)
manifest["tag"] = manifest["tag"].astype(str)
manifest = manifest[manifest["tag"].isin(TAGS)].copy()
manifest = manifest[manifest["well"].isin(base["well"].unique())]
complete = manifest.groupby("well")["tag"].nunique()
complete_wells = set(complete[complete.eq(3)].index)
missing_by_fold = (
    base.loc[~base["well"].isin(complete_wells), ["well", "fold"]]
    .drop_duplicates().groupby("fold")["well"].nunique().to_dict()
)
base = base[base["well"].isin(complete_wells)].reset_index(drop=True)
coverage = base.groupby("fold")["well"].nunique().astype(int).to_dict()
if sum(coverage.get(fold, 0) for fold in TRAIN_FOLDS) < 295 or coverage.get(3, 0) < 145 or coverage.get(2, 0) < 145:
    raise RuntimeError(f"Insufficient B1 coverage: {coverage}; missing={missing_by_fold}")

schema_path = CACHE_DIR / "matched_k3_gateway_features.csv"
if schema_path.is_file():
    features = pd.read_csv(schema_path)["feature"].astype(str).tolist()
else:
    probe_wells = sorted(complete_wells)[:12]
    probe_manifest = manifest[manifest["well"].isin(probe_wells)].copy()
    probe_manifest["cut_bin"] = 0
    probe_manifest["states_in_bin"] = 3
    probe = _load_samples(probe_wells, probe_manifest)
    excluded = {
        "well", "id", "target", "_row_idx", "_cut_tag", "_official_n_eval",
        "_sample_weight_k3", "_sample_weight_single", "_global_weight",
        "_specialist_weight", "_cut_bin", "cut_visible_frac",
    }
    features = [
        column for column in probe.columns
        if column not in excluded and not _is_spatial(column)
        and pd.api.types.is_numeric_dtype(probe[column])
    ]
    del probe
features = [feature for feature in features if not _is_spatial(feature)]
features = list(dict.fromkeys(features + ["cut_visible_frac"]))
(OUT_DIR / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")

print(
    "M342-B1 CONFIG",
    {"coverage": coverage, "features": len(features), "cache": str(CACHE_DIR),
     "base": str(V168_PATH), "fold1_touched": False},
    flush=True,
)

selection = _evaluate_stage(
    "selection_fold3", TRAIN_FOLDS, SELECTION_FOLD, base, manifest, features
)
sel_frame, sel_global, sel_specialist, sel_rotated, sel_edges, iterations = selection
sel_scan = _dose_scan(
    sel_frame,
    {"global": sel_global, "specialist": sel_specialist, "rotated": sel_rotated},
)
global_dose = _select_dose(sel_scan, "global")
specialist_dose = _select_dose(sel_scan, "specialist")
base_sel = sel_frame["base_v168"].to_numpy(np.float64)
sel_global_blend = base_sel + global_dose * (sel_global - base_sel)
sel_specialist_blend = base_sel + specialist_dose * (sel_specialist - base_sel)
sel_rotated_blend = base_sel + specialist_dose * (sel_rotated - base_sel)
selection_metrics = {
    name: _metrics(sel_frame, prediction, name)
    for name, prediction in {
        "base_v168": base_sel,
        "global_endpoint": sel_global,
        "specialist_endpoint": sel_specialist,
        "rotated_endpoint": sel_rotated,
        "global_blend": sel_global_blend,
        "specialist_blend": sel_specialist_blend,
        "rotated_blend": sel_rotated_blend,
    }.items()
}
selection_gain = selection_metrics["base_v168"]["rmse"] - selection_metrics["specialist_blend"]["rmse"]
selection_increment = selection_metrics["global_blend"]["rmse"] - selection_metrics["specialist_blend"]["rmse"]
selection_rotation_loss = selection_metrics["rotated_blend"]["rmse"] - selection_metrics["specialist_blend"]["rmse"]
mechanism_pass = bool(
    specialist_dose > 0
    and selection_gain >= 0.03
    and selection_increment >= 0.015
    and selection_rotation_loss >= 0.03
)

print("\nM342-B1 SELECTION DOSE SCAN")
print(sel_scan.to_string(index=False))
print("\nM342-B1 SELECTION SUMMARY")
print(pd.DataFrame(selection_metrics.values()).to_string(index=False))
print(
    "M342-B1 SELECTION GATE",
    {"global_dose": global_dose, "specialist_dose": specialist_dose,
     "gain": selection_gain, "increment_vs_global": selection_increment,
     "rotation_loss": selection_rotation_loss, "pass": mechanism_pass},
    flush=True,
)

confirmation_metrics = None
confirmation_checks = None
if mechanism_pass:
    confirmation = _evaluate_stage(
        "confirmation_fold2", (3, 4, 5), CONFIRMATION_FOLD,
        base, manifest, features, iterations=iterations,
    )
    con_frame, con_global, con_specialist, con_rotated, con_edges, _ = confirmation
    base_con = con_frame["base_v168"].to_numpy(np.float64)
    con_global_blend = base_con + global_dose * (con_global - base_con)
    con_specialist_blend = base_con + specialist_dose * (con_specialist - base_con)
    con_rotated_blend = base_con + specialist_dose * (con_rotated - base_con)
    confirmation_metrics = {
        name: _metrics(con_frame, prediction, name)
        for name, prediction in {
            "base_v168": base_con,
            "global_endpoint": con_global,
            "specialist_endpoint": con_specialist,
            "rotated_endpoint": con_rotated,
            "global_blend": con_global_blend,
            "specialist_blend": con_specialist_blend,
            "rotated_blend": con_rotated_blend,
        }.items()
    }
    con_gain = confirmation_metrics["base_v168"]["rmse"] - confirmation_metrics["specialist_blend"]["rmse"]
    con_far_gain = confirmation_metrics["base_v168"]["far_gt2000_rmse"] - confirmation_metrics["specialist_blend"]["far_gt2000_rmse"]
    con_increment = confirmation_metrics["global_blend"]["rmse"] - confirmation_metrics["specialist_blend"]["rmse"]
    con_rotation_loss = confirmation_metrics["rotated_blend"]["rmse"] - confirmation_metrics["specialist_blend"]["rmse"]
    confirmation_checks = {
        "gain_ge_0p03": con_gain >= 0.03,
        "far_gain_ge_0": con_far_gain >= 0.0,
        "increment_vs_global_ge_0p015": con_increment >= 0.015,
        "rotation_loses_ge_0p03": con_rotation_loss >= 0.03,
    }
    print("\nM342-B1 CONFIRMATION SUMMARY")
    print(pd.DataFrame(confirmation_metrics.values()).to_string(index=False))
else:
    con_edges = None

strict_pass = bool(mechanism_pass and confirmation_checks and all(confirmation_checks.values()))
verdict = {
    "experiment": "main342_cut_fraction_specialist_gateway_b1",
    "stage": "fold45_train_fold3_selection_fold345_refit_fold2_confirmation",
    "sample_unit": "complete physical well; all official hidden rows",
    "legal_inputs_only": True,
    "selection_or_tuning_fold": 3,
    "confirmation_fold": 2 if mechanism_pass else None,
    "fold1_touched": False,
    "sources": {"feature_cache": str(CACHE_DIR), "base_oof": str(V168_PATH)},
    "coverage": coverage,
    "feature_count": len(features),
    "selection_edges": sel_edges,
    "confirmation_edges": con_edges,
    "frozen_iterations": iterations,
    "frozen_doses": {"global": global_dose, "specialist": specialist_dose},
    "selection": selection_metrics,
    "selection_gate": {
        "gain": selection_gain,
        "increment_vs_global": selection_increment,
        "rotation_loss": selection_rotation_loss,
        "mechanism_pass": mechanism_pass,
    },
    "confirmation": confirmation_metrics,
    "confirmation_checks": confirmation_checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> rebuild full-data cut specialists and audit Kaggle inference cut routing"
        if strict_pass else
        "FAIL -> visible cut fraction does not justify separate row-model experts; close B1"
    ),
}
(OUT_DIR / "selection_dose_scan.csv").write_text(sel_scan.to_csv(index=False), encoding="utf-8")
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)
print("\nM342-B1 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)
