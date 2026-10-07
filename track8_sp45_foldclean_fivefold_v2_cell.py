"""Five-fold author-like Track8 from the feature-rich standalone SP45 replay.

Track8 is deliberately kept separate from SP45:
* SP45 supplies multi-seed PF, multi-config beam and selector features.
* Track8 is a small GBDT ensemble over those engine features plus the frozen
  multiscale-correlation and formation-plane-neighbour common feature block.

A paired common-only arm uses the identical folds, rows, trainer, seeds and
post-processing.  If the raw-data STRIDE OOF exists, the cell also measures
raw and distance-detrended error correlation and a nested fixed-weight blend.
"""

from __future__ import annotations

import gc
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
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "track8_sp45_foldclean_fivefold_v2"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CACHE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = CACHE_ROOT / "features"
FEATURE_LIST_PATH = CACHE_ROOT / "all_features.csv"
FOLD_PATH = CACHE_ROOT / "well_metadata.csv"
SP45_DIR = OOF_ROOT / "sp45_foldclean_feature_export_v1"
SP45_PATH = SP45_DIR / "sp45_feature_oof.parquet"
SP45_MANIFEST_PATH = SP45_DIR / "sp45_provenance_manifest.json"
STRIDE_PATH = OOF_ROOT / "stride_backbone_773_crossfit_v2" / "stride_backbone_oof.parquet"

for required in (FEATURE_DIR, FEATURE_LIST_PATH, FOLD_PATH, SP45_PATH, SP45_MANIFEST_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

N_FOLDS = 5
MODEL_SEEDS = (20260723, 20260837)
MAX_COMMON_FEATURES = 180
MAX_TREES = 2200
PATIENCE = 160
SMOOTH_WINDOW = 41
SMOOTH_POLY = 2


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalise_folds(frame: pd.DataFrame) -> pd.DataFrame:
    well_column = next((column for column in ("well", "well_id", "Well") if column in frame), None)
    fold_column = next((column for column in ("fold", "outer_fold", "Fold") if column in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify fold columns: {frame.columns.tolist()}")
    output = frame[[well_column, fold_column]].copy()
    output.columns = ["well", "fold"]
    output["well"] = output["well"].astype(str)
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(np.int8)
    output = output.drop_duplicates("well").sort_values("well").reset_index(drop=True)
    if len(output) != 773 or not output["fold"].isin(range(1, 6)).all():
        raise RuntimeError(f"Unexpected fold metadata: {len(output)} wells")
    return output


folds = _normalise_folds(pd.read_csv(FOLD_PATH))
fold_by_well = dict(zip(folds["well"], folds["fold"]))

manifest = json.loads(SP45_MANIFEST_PATH.read_text(encoding="utf-8"))
checks = manifest.get("checks", {})
if not checks.get("full_128seed_500particle_run", False) or not checks.get("coverage_pass", False):
    raise RuntimeError(f"SP45 replay is incomplete: {checks}")
# Historical parity was an invalid comparison: the candidate-pool SP45 column
# is a GBDT/SP45 composite endpoint, not the standalone SP45 engine.


SP45_PATH_COLUMNS = [
    "pf_scale_3",
    "pf_scale_5",
    "pf_scale_8",
    "pf_scale_12",
    "pf_mean",
    "pf_p10",
    "pf_p50",
    "pf_p90",
    *[f"beam_{index:02d}" for index in range(14)],
    "beam_mean",
    "beam_median",
    "sp45_selector_raw",
    "sp45_selector_projected",
]
SP45_DIRECT_COLUMNS = [
    "pf_std",
    "beam_std",
    "pf_scale_range",
    "beam_range",
    "pf_beam_disagreement",
    "likelihood_max",
    "likelihood_mean",
    "likelihood_std",
    "likelihood_top_gap",
    "selector_code",
    "z_span",
    "gr_scale",
    "initial_rate",
]
SP45_READ_COLUMNS = ["id", "well", "row_idx", "fold"] + SP45_PATH_COLUMNS + SP45_DIRECT_COLUMNS

sp45 = pd.read_parquet(SP45_PATH, columns=SP45_READ_COLUMNS)
sp45["id"] = sp45["id"].astype(str)
if len(sp45) < 3_760_000 or sp45["well"].nunique() != 773 or sp45["id"].duplicated().any():
    raise RuntimeError(
        f"Invalid SP45 OOF coverage: rows={len(sp45)}, wells={sp45['well'].nunique()}"
    )
sp45 = sp45.set_index("id", verify_integrity=True)


def _load_feature_file(well: str, suffix: str) -> pd.DataFrame:
    path = FEATURE_DIR / f"{well}__{suffix}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_pickle(path)
    frame["well"] = str(well)
    frame["fold"] = int(fold_by_well[well])
    frame["id"] = frame["id"].astype(str)
    return frame


def _attach_sp45(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    aligned = sp45.reindex(frame["id"].to_numpy())
    if aligned.index.has_duplicates or aligned[SP45_PATH_COLUMNS[0]].isna().any():
        missing = frame.loc[aligned[SP45_PATH_COLUMNS[0]].isna().to_numpy(), "id"].head(10).tolist()
        raise RuntimeError(f"SP45 alignment failed for {len(missing)}+ rows: {missing}")
    last = frame["last_known_tvt"].to_numpy(np.float32)
    progress = np.maximum(frame["md_since"].to_numpy(np.float32), 0.0)
    well_row_count = frame.groupby("well", sort=False)["id"].transform("size").to_numpy(np.float32)
    engine_features = []
    for column in SP45_PATH_COLUMNS:
        name = f"track8_{column}_d"
        frame[name] = aligned[column].to_numpy(np.float32) - last
        engine_features.append(name)
    for column in SP45_DIRECT_COLUMNS:
        name = f"track8_{column}"
        values = aligned[column].to_numpy(np.float32)
        if column.startswith("likelihood_"):
            values = values / np.maximum(well_row_count, 1.0)
        frame[name] = values
        engine_features.append(name)

    frame["track8_selector_x_sqrt_progress"] = (
        frame["track8_sp45_selector_projected_d"].to_numpy(np.float32)
        * np.sqrt(progress + 1.0)
        / 64.0
    )
    frame["track8_pf_std_x_log_progress"] = (
        frame["track8_pf_std"].to_numpy(np.float32) * np.log1p(progress) / 8.0
    )
    frame["track8_beam_std_x_log_progress"] = (
        frame["track8_beam_std"].to_numpy(np.float32) * np.log1p(progress) / 8.0
    )
    engine_features += [
        "track8_selector_x_sqrt_progress",
        "track8_pf_std_x_log_progress",
        "track8_beam_std_x_log_progress",
    ]
    return frame, engine_features


EXCLUDED_EXACT = {
    "well",
    "id",
    "target",
    "TVT",
    "TVT_input",
    "_row_idx",
    "fold",
    "sample_weight",
}
ENGINE_TOKENS = (
    "pf_ancc",
    "pf_z",
    "likpf",
    "beam_",
    "hyb",
    "slp",
    "selector",
    "ridge",
    "sp45",
    "strong",
    "cp12",
    "gold",
)
COMMON_PREFIXES = (
    "gr",
    "ncc",
    "corr",
    "tvtf_",
    "tvtfw_",
    "tvtf50_",
    "bw_",
    "bww_",
    "bw50_",
    "frm_",
    "form_",
    "dense_",
    "spatial_",
)
COMMON_CONTAINS = (
    "corr",
    "ncc",
    "neighbor",
    "neighbour",
    "distance",
    "entropy",
    "margin",
    "spectral",
    "resid",
    "typewell",
)
CORE_GEOMETRY = {"MD", "X", "Y", "Z", "GR", "last_known_tvt", "md_since", "progress"}


def _legal_common(schema: pd.DataFrame, column: str) -> bool:
    if column in EXCLUDED_EXACT or column not in schema:
        return False
    if not pd.api.types.is_numeric_dtype(schema[column]):
        return False
    lower = column.lower()
    if any(token in lower for token in ("truth", "oracle", "label")):
        return False
    if any(token in lower for token in ENGINE_TOKENS):
        return False
    if column in CORE_GEOMETRY or lower.startswith(COMMON_PREFIXES):
        return True
    return any(token in lower for token in COMMON_CONTAINS)


def _common_priority(column: str) -> int:
    lower = column.lower()
    if column in CORE_GEOMETRY:
        return 0
    if "corr" in lower or "ncc" in lower or lower.startswith("gr"):
        return 1
    if lower.startswith(("tvtf", "bw", "frm", "form", "dense", "spatial")):
        return 2
    return 3


schema_well = folds.iloc[0]["well"]
schema = _load_feature_file(schema_well, "sample")
listed_features = pd.read_csv(FEATURE_LIST_PATH)["feature"].astype(str).tolist()
common_features = [column for column in listed_features if _legal_common(schema, column)]
common_features = sorted(common_features, key=lambda column: (_common_priority(column), listed_features.index(column)))
common_features = common_features[:MAX_COMMON_FEATURES]
schema, engine_features = _attach_sp45(schema)
track8_features = common_features + engine_features
if len(common_features) < 40 or len(engine_features) < 30:
    raise RuntimeError(
        f"Feature block too small: common={len(common_features)}, engine={len(engine_features)}"
    )


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _new_model(fold: int, arm: str, seed: int) -> LGBMRegressor:
    # The parameterization is shared exactly between both arms.
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=MAX_TREES,
        learning_rate=0.025,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=10.0,
        max_bin=255,
        n_jobs=-1,
        random_state=int(seed + fold),
        verbosity=-1,
    )


def _fit_one(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
    fold: int,
    arm: str,
    seed: int,
) -> LGBMRegressor:
    x_train = _matrix(train, columns)
    x_valid = _matrix(valid, columns)
    y_train = train["target"].to_numpy(np.float32)
    y_valid = valid["target"].to_numpy(np.float32)
    weights = train.get("sample_weight", pd.Series(np.ones(len(train)))).to_numpy(np.float32)
    model = _new_model(fold, arm, seed)
    fit_kwargs = dict(
        sample_weight=weights,
        eval_set=[(x_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(PATIENCE, verbose=False), log_evaluation(0)],
    )
    try:
        params = model.get_params()
        params["device_type"] = "gpu"
        candidate = LGBMRegressor(**params)
        candidate.fit(x_train, y_train, **fit_kwargs)
        model = candidate
    except Exception as exc:
        if "gpu" not in str(exc).lower() and "opencl" not in str(exc).lower():
            raise
        print("Track8 GPU unavailable; CPU fallback:", repr(exc), flush=True)
        model.fit(x_train, y_train, **fit_kwargs)
    del x_train, x_valid
    gc.collect()
    return model


def _smooth_prediction(frame: pd.DataFrame, model_delta: np.ndarray) -> np.ndarray:
    prediction = frame["last_known_tvt"].to_numpy(np.float64) + np.asarray(model_delta, np.float64)
    output = prediction.copy()
    for _, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        window = min(SMOOTH_WINDOW, len(indices))
        if window % 2 == 0:
            window -= 1
        if window >= SMOOTH_POLY + 2:
            output[indices] = savgol_filter(prediction[indices], window, SMOOTH_POLY)
    return output


def _metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = frame["target"].to_numpy(np.float64) + frame["last_known_tvt"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, names = pd.factorize(frame["well"].astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error * error, minlength=len(names))
    well_rmse = np.sqrt(sse / counts)
    result = {
        "rows": int(len(error)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, mask in (
        ("near_0_500_rmse", distance <= 500),
        ("mid_500_1000_rmse", (distance > 500) & (distance <= 1000)),
        ("mid_1000_2000_rmse", (distance > 1000) & (distance <= 2000)),
        ("far_gt2000_rmse", distance > 2000),
    ):
        result[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else float("nan")
    return result


print(
    "TRACK8 CONFIG",
    {
        "wells": len(folds),
        "common_features": len(common_features),
        "sp45_engine_features": len(engine_features),
        "models_per_arm_fold": len(MODEL_SEEDS),
        "sp45_rows": len(sp45),
    },
    flush=True,
)

sample_parts = []
for number, well in enumerate(folds["well"], start=1):
    sample_parts.append(_load_feature_file(well, "sample"))
    if number % 100 == 0 or number == len(folds):
        print(f"load Track8 training samples {number}/{len(folds)}", flush=True)
samples = pd.concat(sample_parts, ignore_index=True, sort=False)
del sample_parts
samples, _ = _attach_sp45(samples)

oof_parts = []
fold_reports = []
model_reports = []

for heldout_fold in range(1, N_FOLDS + 1):
    train = samples.loc[samples["fold"].ne(heldout_fold)].copy()
    valid_wells = folds.loc[folds["fold"].eq(heldout_fold), "well"].tolist()
    valid_parts = []
    for number, well in enumerate(valid_wells, start=1):
        valid_parts.append(_load_feature_file(well, "full"))
        if number % 50 == 0 or number == len(valid_wells):
            print(f"load Track8 fold{heldout_fold} validation {number}/{len(valid_wells)}", flush=True)
    valid = pd.concat(valid_parts, ignore_index=True, sort=False)
    del valid_parts
    valid, _ = _attach_sp45(valid)

    predictions = {}
    for arm, columns in (("common_only", common_features), ("track8", track8_features)):
        seed_predictions = []
        for seed in MODEL_SEEDS:
            model = _fit_one(train, valid, columns, heldout_fold, arm, seed)
            delta = model.predict(_matrix(valid, columns), num_iteration=model.best_iteration_)
            seed_predictions.append(delta.astype(np.float32))
            path = MODEL_DIR / f"{arm}_fold{heldout_fold}_seed{seed}.joblib"
            joblib.dump(model, path)
            model_reports.append(
                {
                    "fold": heldout_fold,
                    "arm": arm,
                    "seed": seed,
                    "features": len(columns),
                    "best_iteration": int(model.best_iteration_),
                    "model": str(path),
                }
            )
            del model
            gc.collect()
        mean_delta = np.mean(np.stack(seed_predictions), axis=0)
        predictions[arm] = _smooth_prediction(valid, mean_delta)

    # Matched causal controls use the trained Track8 models and alter only the
    # SP45 block at validation time.
    neutral = valid.copy()
    shuffled = valid.copy()
    for column in engine_features:
        neutral[column] = float(np.nanmedian(train[column].to_numpy(np.float64)))
    for _, indices in valid.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        shift = max(len(indices) // 3, 1)
        for column in engine_features:
            shuffled.loc[shuffled.index[indices], column] = np.roll(
                valid.iloc[indices][column].to_numpy(), shift
            )

    for control_name, control_frame in (("neutral_sp45", neutral), ("shuffled_sp45", shuffled)):
        control_seed_predictions = []
        for seed in MODEL_SEEDS:
            model = joblib.load(MODEL_DIR / f"track8_fold{heldout_fold}_seed{seed}.joblib")
            control_seed_predictions.append(
                model.predict(_matrix(control_frame, track8_features), num_iteration=model.best_iteration_)
            )
        predictions[control_name] = _smooth_prediction(
            valid, np.mean(np.stack(control_seed_predictions), axis=0)
        )

    fold_report = {"fold": heldout_fold}
    for arm, prediction in predictions.items():
        metrics = _metrics(valid, prediction)
        fold_report[f"{arm}_rmse"] = metrics["rmse"]
    fold_report["track8_gain_vs_common"] = (
        fold_report["common_only_rmse"] - fold_report["track8_rmse"]
    )
    fold_reports.append(fold_report)
    print("TRACK8 FOLD", fold_report, flush=True)

    truth = valid["target"].to_numpy(np.float64) + valid["last_known_tvt"].to_numpy(np.float64)
    oof_parts.append(
        pd.DataFrame(
            {
                "id": valid["id"].astype(str).to_numpy(),
                "well": valid["well"].astype(str).to_numpy(),
                "row_idx": valid["_row_idx"].to_numpy(np.int32),
                "fold": np.full(len(valid), heldout_fold, dtype=np.int8),
                "truth": truth.astype(np.float32),
                "md_since": valid["md_since"].to_numpy(np.float32),
                "common_only": predictions["common_only"].astype(np.float32),
                "track8": predictions["track8"].astype(np.float32),
                "neutral_sp45": predictions["neutral_sp45"].astype(np.float32),
                "shuffled_sp45": predictions["shuffled_sp45"].astype(np.float32),
            }
        )
    )
    del train, valid, neutral, shuffled, predictions
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
del oof_parts, samples, sp45
gc.collect()
if len(oof) < 3_760_000 or oof["well"].nunique() != 773 or oof["id"].duplicated().any():
    raise RuntimeError(f"Track8 OOF coverage failure: {len(oof)}/{oof['well'].nunique()}")

summary_rows = []
for arm in ("common_only", "track8", "neutral_sp45", "shuffled_sp45"):
    synthetic = pd.DataFrame(
        {
            "well": oof["well"],
            "target": oof["truth"],
            "last_known_tvt": np.zeros(len(oof)),
            "md_since": oof["md_since"],
        }
    )
    summary_rows.append({"arm": arm, **_metrics(synthetic, oof[arm].to_numpy(float))})
summary = pd.DataFrame(summary_rows).sort_values("rmse")


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _detrend(error: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x = np.log1p(np.maximum(distance, 0.0))
    x /= max(float(np.max(x)), 1e-9)
    design = np.column_stack([np.ones(len(x)), x, x * x, x * x * x])
    coefficient = np.linalg.lstsq(design, error, rcond=None)[0]
    return error - design @ coefficient


cross_evidence = None
nested_blend_metrics = None
if STRIDE_PATH.exists():
    stride = pd.read_parquet(STRIDE_PATH, columns=["id", "stride_backbone"])
    stride["id"] = stride["id"].astype(str)
    paired = oof.merge(stride, on="id", how="inner", validate="one_to_one")
    if len(paired) != len(oof):
        raise RuntimeError(f"STRIDE alignment lost {len(oof) - len(paired)} rows")
    truth = paired["truth"].to_numpy(np.float64)
    track8_error = paired["track8"].to_numpy(np.float64) - truth
    stride_error = paired["stride_backbone"].to_numpy(np.float64) - truth
    distance = paired["md_since"].to_numpy(np.float64)
    cross_evidence = {
        "rows": int(len(paired)),
        "raw_signed_error_rho": _pearson(track8_error, stride_error),
        "distance_detrended_error_rho": _pearson(
            _detrend(track8_error, distance), _detrend(stride_error, distance)
        ),
        "strata": {},
    }
    for name, mask in (
        ("0_500", distance <= 500),
        ("500_1000", (distance > 500) & (distance <= 1000)),
        ("1000_2000", (distance > 1000) & (distance <= 2000)),
        ("gt2000", distance > 2000),
    ):
        cross_evidence["strata"][name] = {
            "rows": int(mask.sum()),
            "rho": _pearson(track8_error[mask], stride_error[mask]) if mask.any() else float("nan"),
        }

    nested = np.full(len(paired), np.nan, dtype=np.float64)
    weights = []
    for heldout_fold in range(1, 6):
        train_mask = paired["fold"].to_numpy(int) != heldout_fold
        valid_mask = ~train_mask
        base = paired.loc[train_mask, "stride_backbone"].to_numpy(np.float64)
        candidate = paired.loc[train_mask, "track8"].to_numpy(np.float64)
        target = paired.loc[train_mask, "truth"].to_numpy(np.float64)
        direction = candidate - base
        denominator = float(np.dot(direction, direction))
        weight = float(np.clip(-np.dot(base - target, direction) / max(denominator, 1e-12), 0.0, 1.0))
        weights.append({"fold": heldout_fold, "weight_track8": weight})
        nested[valid_mask] = (
            (1.0 - weight) * paired.loc[valid_mask, "stride_backbone"].to_numpy(float)
            + weight * paired.loc[valid_mask, "track8"].to_numpy(float)
        )
    error = nested - truth
    nested_blend_metrics = {
        "weights": weights,
        "stride_rmse": float(np.sqrt(np.mean(stride_error * stride_error))),
        "track8_rmse": float(np.sqrt(np.mean(track8_error * track8_error))),
        "nested_blend_rmse": float(np.sqrt(np.mean(error * error))),
    }
    nested_blend_metrics["gain_vs_best_component"] = (
        min(nested_blend_metrics["stride_rmse"], nested_blend_metrics["track8_rmse"])
        - nested_blend_metrics["nested_blend_rmse"]
    )
    pd.DataFrame(weights).to_csv(OUT_DIR / "stride_track8_nested_weights.csv", index=False)

oof.to_parquet(OUT_DIR / "track8_oof.parquet", index=False, compression="zstd")
summary.to_csv(OUT_DIR / "summary.csv", index=False)
pd.DataFrame(fold_reports).to_csv(OUT_DIR / "fold_report.csv", index=False)
pd.DataFrame(model_reports).to_csv(OUT_DIR / "model_report.csv", index=False)
pd.Series(common_features, name="feature").to_csv(OUT_DIR / "common_features.csv", index=False)
pd.Series(engine_features, name="feature").to_csv(OUT_DIR / "sp45_engine_features.csv", index=False)

metric_by_arm = {row["arm"]: row for row in summary.to_dict(orient="records")}
track8_gain = metric_by_arm["common_only"]["rmse"] - metric_by_arm["track8"]["rmse"]
neutral_degradation = metric_by_arm["neutral_sp45"]["rmse"] - metric_by_arm["track8"]["rmse"]
shuffle_degradation = metric_by_arm["shuffled_sp45"]["rmse"] - metric_by_arm["track8"]["rmse"]
fold_gains = [row["track8_gain_vs_common"] for row in fold_reports]

gates = {
    "track8_rmse_le_10p5": metric_by_arm["track8"]["rmse"] <= 10.5,
    "gain_vs_common_ge_0p20": track8_gain >= 0.20,
    "minimum_fold_gain_ge_minus0p05": min(fold_gains) >= -0.05,
    "neutral_degradation_ge_0p10": neutral_degradation >= 0.10,
    "shuffle_degradation_ge_0p10": shuffle_degradation >= 0.10,
}
if cross_evidence is not None and nested_blend_metrics is not None:
    gates["stride_oof_available"] = True
    gates["distance_detrended_rho_le_0p55"] = cross_evidence["distance_detrended_error_rho"] <= 0.55
    gates["nested_blend_gain_ge_0p20"] = nested_blend_metrics["gain_vs_best_component"] >= 0.20
else:
    gates["stride_oof_available"] = False

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_author_like_sp45_to_track8_gateway",
    "definitions": {
        "sp45": "standalone 128-seed PF + 14-config beam + per-well selector",
        "track8": "small GBDT ensemble over SP45 features plus shared correlation/formation-neighbour block",
    },
    "protocol": {
        "folds": str(FOLD_PATH),
        "training_rows": "up to cached 1024 rows per train well",
        "validation_rows": "all official hidden rows",
        "models_per_arm_fold": len(MODEL_SEEDS),
        "same_trainer_and_postprocess_for_ab": True,
    },
    "features": {
        "common": len(common_features),
        "sp45_engine": len(engine_features),
        "track8_total": len(track8_features),
    },
    "summary": summary.to_dict(orient="records"),
    "folds": fold_reports,
    "track8_gain_vs_common": track8_gain,
    "neutral_sp45_degradation": neutral_degradation,
    "shuffled_sp45_degradation": shuffle_degradation,
    "stride_cross_evidence": cross_evidence,
    "stride_track8_nested_blend": nested_blend_metrics,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": (
        "PASS -> package Track8 inference and evaluate deployment blend"
        if all(gates.values())
        else "FAIL -> do not deploy; inspect Track8 fold and correlation failures"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nTRACK8 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nTRACK8 FOLD REPORT")
print(pd.DataFrame(fold_reports).to_string(index=False), flush=True)
if cross_evidence is not None:
    print("\nTRACK8 / STRIDE CROSS-EVIDENCE")
    print(json.dumps(cross_evidence, indent=2), flush=True)
    print("\nTRACK8 / STRIDE NESTED BLEND")
    print(json.dumps(nested_blend_metrics, indent=2), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
