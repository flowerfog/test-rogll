"""HN27: competition-native GR nuisance augmentation for the mainline GBDT.

Raw horizontal GR is perturbed before build_well and likelihood-PF feature
generation.  Typewell GR, labels, geometry, folds, model, postprocessing and
validation features remain unchanged. Spatial features are excluded from this
mechanism gate because their reference banks would otherwise cross nested-fold
label boundaries. Folds 4/5 train a matched baseline and augmented arm; Fold3
is the mechanism gate. Fold2 is touched only after that gate passes, following
a refit on Folds3/4/5. Fold1 is never loaded.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import types
import warnings
import zlib
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import savgol_filter


warnings.filterwarnings("ignore")
STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
BASE_DIR = OOF / "foldclean_full_fleongg_lgb_v0"
BASE_FEATURE_DIR = BASE_DIR / "features"
META_PATH = BASE_DIR / "well_metadata.csv"
V66_PATH = OOF / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet"
EXPERIMENT = "competition_gr_nuisance_gbdt_hn27"
OUT_DIR = OOF / EXPERIMENT
CONTROL_DIR = OUT_DIR / "control_features"
AUG_DIR = OUT_DIR / "augmented_features"
TEMP_DIR = OUT_DIR / "masked_tmp"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, CONTROL_DIR, AUG_DIR, TEMP_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FOLD1_TOUCHED = False
FAST_REUSE_FOLDCLEAN_CONTROL = True
DOSE = 0.25
TRAIN_ROWS_PER_WELL = 1024
SEED = 20260804
MIN_SELECTION_GAIN = 0.05
MIN_SELECTION_FAR_GAIN = 0.05
MIN_CONFIRMATION_GAIN = 0.05
MIN_CONFIRMATION_FAR_GAIN = 0.05

# Frozen from the competition-only known-prefix nuisance fit in V83.
NUISANCE = {
    "log_gain_sigma": 0.15438494977609182,
    "offset_sigma_scale": 0.50,
    "noise_sigma_scale": 0.30,
    "extra_dropout_probability": 0.03205054566341184,
    "smooth_probability": 0.25,
}

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

for required in (DATA_ROOT / "train", META_PATH, BASE_FEATURE_DIR, V66_PATH):
    if not required.exists():
        raise FileNotFoundError(required)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _find_notebook():
    explicit = os.environ.get("ROGII_MAINLINE_NOTEBOOK", "").strip()
    candidates = [
        Path(explicit) if explicit else None,
        ROOT / "rogii-cp8-neighbor.ipynb",
        Path.cwd() / "rogii-cp8-neighbor.ipynb",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Upload rogii-cp8-neighbor.ipynb to /root/autodl-tmp or set "
        "ROGII_MAINLINE_NOTEBOOK"
    )


def _bootstrap_mainline():
    notebook_path = _find_notebook()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    module_name = "hn27_mainline_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    sys.modules[module_name] = module
    namespace = module.__dict__
    for cell_index in (34, 37, 38, 40, 41, 42, 43):
        source = "".join(notebook["cells"][cell_index].get("source", []))
        source = source.replace("cache=True", "cache=False")
        exec(compile(source, f"{notebook_path}:cell_{cell_index}", "exec"), namespace)
    needed = {"build_well", "lik_pf", "FormationPlaneKNN", "DenseANCCImputer"}
    missing = sorted(needed.difference(namespace))
    if missing:
        raise RuntimeError(f"Notebook bootstrap missing {missing}")
    return notebook_path, namespace


BOOTSTRAP_NOTEBOOK, MAINLINE = _bootstrap_mainline()
build_well = MAINLINE["build_well"]
lik_pf = MAINLINE["lik_pf"]
FormationPlaneKNN = MAINLINE["FormationPlaneKNN"]
DenseANCCImputer = MAINLINE["DenseANCCImputer"]


@njit(cache=False)
def _seed_numba(seed):
    np.random.seed(seed)


def _seed_all(seed):
    np.random.seed(seed)
    _seed_numba(seed)


def _even_positions(n_rows, n_take):
    if n_rows <= n_take:
        return np.arange(n_rows, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, n_rows - 1, n_take)).astype(np.int64))


def _stable_seed(well, fold):
    token = zlib.crc32(str(well).encode("ascii")) & 0x7FFFFFFF
    return int((SEED + token + 1000003 * int(fold)) % 2147483647)


def _augment_gr(values, rng):
    output = np.asarray(values, np.float64).copy()
    valid = np.isfinite(output)
    if valid.sum() < 32:
        return output, {}
    finite = output[valid]
    centre = float(np.median(finite))
    scale = float(np.median(np.abs(finite - centre)) * 1.4826)
    if not np.isfinite(scale) or scale < 1e-3:
        scale = max(float(np.std(finite)), 1e-3)
    gain = math.exp(float(rng.normal(0.0, DOSE * NUISANCE["log_gain_sigma"])))
    offset = float(rng.normal(0.0, DOSE * NUISANCE["offset_sigma_scale"])) * scale
    noise_sigma = DOSE * NUISANCE["noise_sigma_scale"] * scale
    transformed = gain * output + offset
    transformed[valid] += rng.normal(0.0, noise_sigma, valid.sum())
    smoothed = False
    if rng.random() < DOSE * NUISANCE["smooth_probability"]:
        index = np.arange(len(output))
        filled = np.interp(index, index[valid], transformed[valid])
        smooth = np.convolve(filled, np.ones(3) / 3.0, mode="same")
        transformed[valid] = 0.5 * transformed[valid] + 0.5 * smooth[valid]
        smoothed = True
    drop_probability = DOSE * NUISANCE["extra_dropout_probability"]
    drop = valid & (rng.random(len(output)) < drop_probability)
    transformed[drop] = np.nan
    transformed[~valid] = np.nan
    return transformed, {
        "gain": gain, "offset": offset, "scale": scale,
        "noise_sigma": noise_sigma, "drop_fraction": float(drop.mean()),
        "smoothed": smoothed,
    }


def _add_likpf(frame, horizontal, typewell):
    defaults = ["likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12", "likpf_mean"]
    last = frame["last_known_tvt"].to_numpy(np.float32)
    for column in defaults:
        frame[column] = last
    try:
        output, row_index, _ = lik_pf(horizontal.copy(), typewell.copy())
        generated = {}
        for key, values in output.items():
            column = "likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
            generated[column] = np.asarray(values, np.float32)
        frame = frame.drop(columns=[c for c in generated if c in frame.columns])
        table = pd.DataFrame({"_row_idx": np.asarray(row_index, dtype=int)})
        for column, values in generated.items():
            table[column] = values
        frame = frame.merge(table, on="_row_idx", how="left", validate="one_to_one")
        for column in defaults:
            if column not in frame:
                frame[column] = frame["last_known_tvt"]
            frame[column] = frame[column].fillna(frame["last_known_tvt"]).astype(np.float32)
            frame[column + "_d"] = (frame[column] - frame["last_known_tvt"]).astype(np.float32)
    except Exception as exc:
        warnings.warn(f"HN27 lik_pf fallback: {exc!r}")
        for column in defaults:
            frame[column + "_d"] = np.zeros(len(frame), np.float32)
    return frame


def _arm_path(well, arm):
    directory = CONTROL_DIR if arm == "control" else AUG_DIR
    return directory / f"{well}__sample.pkl"


def _pair_ready(well):
    control_path = _arm_path(well, "control")
    augmented_path = _arm_path(well, "augmented")
    if FAST_REUSE_FOLDCLEAN_CONTROL:
        control_path = _base_sample_path(well)
    if not control_path.is_file() or not augmented_path.is_file():
        return False
    try:
        control = pd.read_pickle(control_path)
        augmented = pd.read_pickle(augmented_path)
        return (
            len(control) == len(augmented)
            and control["id"].astype(str).tolist() == augmented["id"].astype(str).tolist()
            and np.array_equal(
                control["target"].to_numpy(np.float64),
                augmented["target"].to_numpy(np.float64),
                equal_nan=True,
            )
        )
    except Exception:
        return False


def _materialize_arm(well, fold, sanitized, typewell, arm, nuisance):
    destination = _arm_path(well, arm)
    temp_path = TEMP_DIR / f"{well}__{arm}_horizontal_well.csv"
    try:
        sanitized.to_csv(temp_path, index=False)
        seed = _stable_seed(well, fold)
        _seed_all(seed)
        frame = build_well(str(temp_path), str(DATA_ROOT / "train" / f"{well}__typewell.csv"), is_train=True)
        if frame is None or len(frame) < 64:
            raise RuntimeError(f"{arm}: build_well returned too few rows")
        frame = frame.reset_index(drop=True)
        frame["_row_idx"] = frame["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
        _seed_all(seed + 1)
        frame = _add_likpf(frame, sanitized, typewell)
        frame["well"] = str(well)
        frame["id"] = str(well) + "_" + frame["_row_idx"].astype(str)
        frame["fold"] = int(fold)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        positions = _even_positions(len(frame), TRAIN_ROWS_PER_WELL)
        sample = frame.iloc[positions].copy()
        sample["source_arm"] = arm
        sample["sample_weight"] = len(frame) / max(len(sample), 1)
        sample.to_pickle(destination)
        return {"rows": len(frame), **nuisance}
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _build_pair(well, fold):
    if _pair_ready(well):
        return {"well": well, "fold": fold, "status": "cached"}
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    try:
        source = pd.read_csv(horizontal_path)
        keep = ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]
        missing = [column for column in keep if column not in source]
        if missing:
            raise RuntimeError(f"missing columns {missing}")
        control = source[keep].copy()
        rng = np.random.default_rng(_stable_seed(well, fold))
        augmented = control.copy()
        augmented["GR"], nuisance = _augment_gr(
            pd.to_numeric(control["GR"], errors="coerce").to_numpy(np.float64), rng
        )
        typewell = pd.read_csv(typewell_path).sort_values("TVT")
        if FAST_REUSE_FOLDCLEAN_CONTROL:
            control_report = {"rows": len(control)}
        else:
            control_report = _materialize_arm(well, fold, control, typewell, "control", {})
        augmented_report = _materialize_arm(well, fold, augmented, typewell, "augmented", nuisance)
        if not _pair_ready(well):
            raise RuntimeError("control/augmented row or target alignment failed")
        return {
            "well": well, "fold": fold, "status": "built",
            "control_rows": control_report["rows"],
            "augmented_rows": augmented_report["rows"], **nuisance,
        }
    except Exception as exc:
        cleanup_arms = ("augmented",) if FAST_REUSE_FOLDCLEAN_CONTROL else ("control", "augmented")
        for arm in cleanup_arms:
            path = _arm_path(well, arm)
            if path.exists():
                path.unlink()
        return {"well": well, "fold": fold, "status": "failed", "reason": repr(exc)}


def _base_sample_path(well):
    return BASE_FEATURE_DIR / f"{well}__sample.pkl"


def _base_full_path(well):
    return BASE_FEATURE_DIR / f"{well}__full.pkl"


def _load_samples(wells, augmented):
    frames = []
    for well in wells:
        control_path = (
            _base_sample_path(well)
            if FAST_REUSE_FOLDCLEAN_CONTROL
            else _arm_path(well, "control")
        )
        control = pd.read_pickle(control_path).copy()
        control["source_arm"] = "control"
        frames.append(control)
        if augmented:
            frames.append(pd.read_pickle(_arm_path(well, "augmented")))
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if augmented:
        # Preserve each physical well's original row mass after duplicating it.
        frame["sample_weight"] = frame["sample_weight"].to_numpy(np.float64) * 0.5
    return frame


def _load_full(wells):
    return pd.concat([pd.read_pickle(_base_full_path(w)) for w in wells], ignore_index=True, sort=False)


def _feature_columns(frame):
    excluded = {"well", "id", "target", "_row_idx", "fold", "sample_weight", "source_arm"}
    all_columns = [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return [column for column in all_columns if not _is_spatial_feature(column)]


SPATIAL_PREFIXES = (
    "tvtF_", "tvtFw_", "tvtF50_", "bw_", "bww_", "bw50_",
    "bw_early_", "bw_mid_", "frm_rmse_", "dense_", "tvt_dense",
)
SPATIAL_EXACT = {
    "form_mean_d", "form_std_d", "form_rng_d", "spatial_ancc_d",
    "spatial_knn_dist", "pf_vs_spatial", "pf_vs_dense", "spatial_vs_dense",
    "beam_vs_spatial", "sig_std", "sig_mean_d",
}


def _is_spatial_feature(column):
    return column in SPATIAL_EXACT or column.startswith(SPATIAL_PREFIXES)


def _matrix(frame, columns):
    return (
        frame.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)
        .fillna(0.0).to_numpy(np.float32)
    )


def _model(seed):
    return lgb.LGBMRegressor(
        objective="regression", boosting_type="gbdt", n_estimators=2400,
        learning_rate=0.025, num_leaves=64, min_child_samples=80,
        subsample=0.80, subsample_freq=1, colsample_bytree=0.70,
        reg_alpha=0.10, reg_lambda=10.0, max_bin=255,
        n_jobs=max(1, min(16, os.cpu_count() or 1)), random_state=seed, verbosity=-1,
    )


def _fit(train, valid, columns, label):
    model = _model(SEED)
    model.fit(
        _matrix(train, columns), train["target"].to_numpy(np.float64),
        sample_weight=train["sample_weight"].to_numpy(np.float64),
        eval_set=[(_matrix(valid, columns), valid["target"].to_numpy(np.float64))],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(50)],
        feature_name=columns,
    )
    model.booster_.save_model(str(MODEL_DIR / f"{label}.txt"))
    return model


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    md_since = frame["md_since"].to_numpy(np.float64)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(np.float64) - last
    lik_delta = (
        frame[PP_LIKPF_COL].to_numpy(np.float64) - last
        if PP_LIKPF_COL in frame else pf_delta
    )
    learned = PP_ALPHA * warm * (
        model_delta * (1.0 - PP_W_PF_ANCC) + pf_delta * PP_W_PF_ANCC
    )
    prediction = last + PP_W_MODEL * learned + (1.0 - PP_W_MODEL) * lik_delta
    output = prediction.copy()
    offset = 0
    for _, group in frame.groupby("well", sort=False):
        n_rows = len(group)
        window = min(PP_SG_WIN, n_rows)
        if window % 2 == 0:
            window -= 1
        if window >= PP_SG_POLY + 2:
            output[offset:offset + n_rows] = savgol_filter(
                prediction[offset:offset + n_rows], window, PP_SG_POLY
            )
        offset += n_rows
    return output


def _metric(frame, prediction, endpoint):
    truth = frame["target"].to_numpy(np.float64) + frame["last_known_tvt"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    well_rmse = np.sqrt(
        pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
        .groupby("well", sort=False)["e2"].mean().to_numpy(np.float64)
    )
    return {
        "endpoint": endpoint, "rows": len(frame), "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def _error_correlation(frame, a, b):
    truth = frame["target"].to_numpy(np.float64) + frame["last_known_tvt"].to_numpy(np.float64)
    return float(np.corrcoef(np.asarray(a) - truth, np.asarray(b) - truth)[0, 1])


def _build_fold_features(meta, folds):
    reports = []
    original_fi = MAINLINE.get("_FI")
    original_di = MAINLINE.get("_DI")
    try:
        for fold in folds:
            wells = meta.loc[meta["fold"].eq(fold), "well"].astype(str).tolist()
            missing = [
                well for well in wells
                if not _pair_ready(well)
            ]
            if not missing:
                reports.extend({"well": w, "fold": fold, "status": "cached"} for w in wells)
                continue
            reference = meta.loc[meta["fold"].ne(fold), "well"].astype(str).tolist()
            MAINLINE["_FI"] = FormationPlaneKNN(reference, DATA_ROOT / "train")
            MAINLINE["_DI"] = DenseANCCImputer(reference, DATA_ROOT / "train")
            for number, well in enumerate(missing, 1):
                result = _build_pair(well, fold)
                reports.append(result)
                if number % 25 == 0 or number == len(missing):
                    failures = sum(r.get("status") == "failed" for r in reports)
                    print(f"HN27 augment fold={fold} {number}/{len(missing)} failures={failures}", flush=True)
    finally:
        MAINLINE["_FI"] = original_fi
        MAINLINE["_DI"] = original_di
    report = pd.DataFrame(reports)
    report.to_csv(OUT_DIR / f"augmentation_report_{'_'.join(map(str, folds))}.csv", index=False)
    failures = report[report["status"].eq("failed")] if len(report) else report
    if len(failures):
        raise RuntimeError(f"{len(failures)} augmentation failures: {failures.head().to_dict('records')}")


def _run_ab(meta, train_folds, valid_fold, label):
    train_wells = meta.loc[meta["fold"].isin(train_folds), "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"].eq(valid_fold), "well"].astype(str).tolist()
    train_base = _load_samples(train_wells, augmented=False)
    train_aug = _load_samples(train_wells, augmented=True)
    valid = _load_full(valid_wells)
    columns = _feature_columns(train_base)
    missing_aug = sorted(set(columns).difference(train_aug.columns))
    if missing_aug:
        raise RuntimeError(f"Augmented schema missing {missing_aug[:20]}")
    baseline_model = _fit(train_base, valid, columns, f"{label}_baseline")
    augmented_model = _fit(train_aug, valid, columns, f"{label}_augmented")
    baseline = _postprocess(
        valid, baseline_model.predict(_matrix(valid, columns), num_iteration=baseline_model.best_iteration_)
    )
    augmented = _postprocess(
        valid, augmented_model.predict(_matrix(valid, columns), num_iteration=augmented_model.best_iteration_)
    )
    base_metric = _metric(valid, baseline, "matched_baseline")
    aug_metric = _metric(valid, augmented, "nuisance_augmented")
    output = valid[["id", "well", "_row_idx", "md_since"]].copy()
    output["tvt_true"] = valid["target"].to_numpy(np.float64) + valid["last_known_tvt"].to_numpy(np.float64)
    output["baseline"] = baseline.astype(np.float32)
    output["augmented"] = augmented.astype(np.float32)
    output.to_parquet(OUT_DIR / f"{label}_oof.parquet", index=False, compression="zstd")
    return {
        "base": base_metric, "augmented": aug_metric,
        "gain": base_metric["rmse"] - aug_metric["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - aug_metric["far_gt2000_rmse"],
        "error_correlation": _error_correlation(valid, baseline, augmented),
        "features": len(columns), "baseline_iteration": int(baseline_model.best_iteration_),
        "augmented_iteration": int(augmented_model.best_iteration_),
    }


meta = pd.read_csv(META_PATH, dtype={"well": str})
meta = meta[meta["fold"].isin([2, 3, 4, 5])].copy()
for well in meta["well"]:
    if not _base_sample_path(well).is_file() or not _base_full_path(well).is_file():
        raise FileNotFoundError(f"Missing fold-clean cache for {well}")
coverage = meta.groupby("fold")["well"].nunique().astype(int).to_dict()
if any(coverage.get(fold, 0) < 145 for fold in (2, 3, 4, 5)):
    raise RuntimeError(f"Insufficient fold-clean coverage: {coverage}")

print("HN27 CONFIG", {
    "notebook": str(BOOTSTRAP_NOTEBOOK), "coverage": coverage,
    "dose": DOSE, "nuisance": NUISANCE, "fold1_touched": FOLD1_TOUCHED,
}, flush=True)

_build_fold_features(meta, TRAIN_FOLDS)
selection = _run_ab(meta, TRAIN_FOLDS, SELECTION_FOLD, "fold3_selection")
mechanism_pass = selection["gain"] >= MIN_SELECTION_GAIN and selection["far_gain"] >= MIN_SELECTION_FAR_GAIN

confirmation = None
if mechanism_pass:
    _build_fold_features(meta, (SELECTION_FOLD,))
    confirmation = _run_ab(meta, (3, 4, 5), CONFIRMATION_FOLD, "fold2_confirmation")

checks = {
    "selection_gain_ge_0p05": selection["gain"] >= MIN_SELECTION_GAIN,
    "selection_far_gain_ge_0p05": selection["far_gain"] >= MIN_SELECTION_FAR_GAIN,
    "confirmation_executed": confirmation is not None,
    "confirmation_gain_ge_0p05": confirmation is not None and confirmation["gain"] >= MIN_CONFIRMATION_GAIN,
    "confirmation_far_gain_ge_0p05": confirmation is not None and confirmation["far_gain"] >= MIN_CONFIRMATION_FAR_GAIN,
}
strict_pass = all(checks.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "raw_horizontal_gr_nuisance_nonspatial_gbdt_nested_gateway",
    "intervention": "horizontal GR only before build_well and lik_pf; validation remains original",
    "model_feature_scope": "deployable non-spatial mainline features; spatial stack deferred until mechanism pass",
    "control_source": (
        "existing fold-clean cache (fast screening; PF random realization is not exactly paired)"
        if FAST_REUSE_FOLDCLEAN_CONTROL else
        "same-seed raw rebuild paired with augmented arm"
    ),
    "external_data_used": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS), "selection_fold": SELECTION_FOLD,
        "outer_refit_folds": [3, 4, 5] if mechanism_pass else None,
        "confirmation_fold": CONFIRMATION_FOLD if mechanism_pass else None,
        "fold1_touched": FOLD1_TOUCHED,
    },
    "coverage": coverage,
    "dose": DOSE,
    "nuisance": NUISANCE,
    "selection": selection,
    "mechanism_pass": mechanism_pass,
    "confirmation": confirmation,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild the complete spatial stack with strict nested reference banks"
        if strict_pass else
        "FAIL -> raw GR nuisance augmentation does not improve the matched mainline GBDT; close branch"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nHN27 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
