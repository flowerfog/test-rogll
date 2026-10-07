"""Exact OOF decomposition of datum, linear drift, and nonlinear shape error.

This is a zero-training diagnostic.  It reads the fold-clean merged OOF pool,
reconstructs native MD distance and truth from each training well, and performs
an oracle residual decomposition independently for every physical well:

    residual = datum + linear_drift + nonlinear_shape

The components are orthogonal least-squares projections, so their SSE values
sum back to the original SSE up to floating-point error.  Oracle corrections
are diagnostics only and are never presented as deployable predictions.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


EXPERIMENT = "shape_datum_residual_decomposition_v0"
ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (ROOT / "rogii-wellbore-geology-prediction", Path.cwd())
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() else DATA_ROOT / "oof_exports"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PRIORITY = (
    "strong",
    "strong_bias",
    "cp12",
    "sp45",
    "fleongg",
    "ridge",
    "selector",
)
DATUM_TAIL_THRESHOLD_FT = 8.0
TIGHT_SHAPE_THRESHOLD_FT = 3.0
TARGET_RMSE_FT = 5.5
STRATA = (
    ("near_0_500", 0.0, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _load_fold_map():
    if FOLD_PATH is None:
        raise FileNotFoundError("Cannot find fold-clean well_metadata.csv")
    frame = pd.read_csv(FOLD_PATH, dtype=str)
    well_col = next((col for col in ("well", "well_id", "WELL") if col in frame), None)
    fold_col = next((col for col in ("fold", "outer_fold", "Fold") if col in frame), None)
    if well_col is None or fold_col is None:
        raise RuntimeError(f"Unexpected fold columns: {frame.columns.tolist()}")
    frame = frame[[well_col, fold_col]].rename(columns={well_col: "well", fold_col: "fold"})
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Unexpected folds: {sorted(frame['fold'].unique().tolist())}")
    return frame.set_index("well")["fold"].to_dict()


def _native_arrays(well, row_index):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["MD", "TVT", "TVT_input"])
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise RuntimeError(f"{well}: no visible TVT_input")
    anchor = int(known[-1])
    row_index = np.asarray(row_index, np.int64)
    if row_index.min() < 0 or row_index.max() >= len(frame):
        raise RuntimeError(f"{well}: row index outside native frame")
    if np.any(row_index <= anchor):
        raise RuntimeError(f"{well}: OOF pool includes visible-prefix rows")
    official_eval = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    if not np.array_equal(row_index, official_eval):
        raise RuntimeError(
            f"{well}: OOF rows do not exactly reproduce the official mask "
            f"({len(row_index)} vs {len(official_eval)})"
        )
    return truth[row_index], md[row_index] - float(md[anchor])


def _fit_components(residual, distance_ft):
    residual = np.asarray(residual, np.float64)
    x = np.asarray(distance_ft, np.float64) / 1000.0
    datum = float(np.mean(residual))
    demeaned = residual - datum
    centred_x = x - float(np.mean(x))
    denominator = float(np.dot(centred_x, centred_x))
    slope = float(np.dot(centred_x, demeaned) / denominator) if denominator > 1e-12 else 0.0
    linear_component = slope * centred_x
    nonlinear = demeaned - linear_component
    sse = float(np.dot(residual, residual))
    datum_sse = float(len(residual) * datum * datum)
    drift_sse = float(np.dot(linear_component, linear_component))
    nonlinear_sse = float(np.dot(nonlinear, nonlinear))
    closure = datum_sse + drift_sse + nonlinear_sse
    return {
        "datum_bias_ft": datum,
        "slope_ft_per_1000ft": slope,
        "sse": sse,
        "datum_sse": datum_sse,
        "drift_sse": drift_sse,
        "nonlinear_shape_sse": nonlinear_sse,
        "closure_error": closure - sse,
        "datum_corrected_error": demeaned,
        "affine_corrected_error": nonlinear,
    }


def _well_row(well, fold, model, prediction, truth, distance):
    residual = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    components = _fit_components(residual, distance)
    n = len(residual)
    sse = components["sse"]
    datum_sse = components["datum_sse"]
    drift_sse = components["drift_sse"]
    nonlinear_sse = components["nonlinear_shape_sse"]
    output = {
        "well": well,
        "fold": int(fold),
        "model": model,
        "rows": n,
        "exact_sse": sse,
        "rmse": float(math.sqrt(sse / n)),
        "mae": float(np.mean(np.abs(residual))),
        "datum_bias_ft": components["datum_bias_ft"],
        "abs_datum_bias_ft": abs(components["datum_bias_ft"]),
        "datum_sse": datum_sse,
        "datum_sse_fraction": datum_sse / max(sse, 1e-12),
        "shape_rmse_after_datum": float(math.sqrt((drift_sse + nonlinear_sse) / n)),
        "slope_ft_per_1000ft": components["slope_ft_per_1000ft"],
        "drift_sse": drift_sse,
        "drift_sse_fraction": drift_sse / max(sse, 1e-12),
        "shape_rmse_after_affine": float(math.sqrt(nonlinear_sse / n)),
        "nonlinear_shape_sse": nonlinear_sse,
        "nonlinear_shape_sse_fraction": nonlinear_sse / max(sse, 1e-12),
        "sse_closure_error": components["closure_error"],
        "sse_closure_relative_error": abs(components["closure_error"]) / max(sse, 1e-12),
    }
    datum_error = components["datum_corrected_error"]
    affine_error = components["affine_corrected_error"]
    for name, low, high in STRATA:
        mask = (distance > low if low > 0 else distance >= low) & (distance <= high)
        count = int(mask.sum())
        output[f"{name}_rows"] = count
        output[f"{name}_sse"] = float(np.sum(residual[mask] ** 2))
        output[f"{name}_datum_corrected_sse"] = float(np.sum(datum_error[mask] ** 2))
        output[f"{name}_affine_corrected_sse"] = float(np.sum(affine_error[mask] ** 2))
    output["wrong_datum"] = bool(output["abs_datum_bias_ft"] >= DATUM_TAIL_THRESHOLD_FT)
    output["tight_shape"] = bool(output["shape_rmse_after_datum"] <= TIGHT_SHAPE_THRESHOLD_FT)
    if output["wrong_datum"] and output["tight_shape"]:
        output["failure_class"] = "tight_shape_wrong_datum"
    elif output["wrong_datum"]:
        output["failure_class"] = "loose_shape_wrong_datum"
    elif output["tight_shape"]:
        output["failure_class"] = "tight_shape_small_datum"
    else:
        output["failure_class"] = "loose_shape_small_datum"
    return output


def _pooled_report(frame):
    rows = int(frame["rows"].sum())
    sse = float(frame["exact_sse"].sum())
    datum_sse = float(frame["datum_sse"].sum())
    drift_sse = float(frame["drift_sse"].sum())
    nonlinear_sse = float(frame["nonlinear_shape_sse"].sum())
    report = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "exact_sse": sse,
        "rmse": float(math.sqrt(sse / rows)),
        "datum_oracle_rmse": float(math.sqrt((drift_sse + nonlinear_sse) / rows)),
        "affine_oracle_rmse": float(math.sqrt(nonlinear_sse / rows)),
        "datum_sse_fraction": datum_sse / max(sse, 1e-12),
        "drift_sse_fraction": drift_sse / max(sse, 1e-12),
        "nonlinear_shape_sse_fraction": nonlinear_sse / max(sse, 1e-12),
        "median_well_rmse": float(frame["rmse"].median()),
        "p95_well_rmse": float(frame["rmse"].quantile(0.95)),
        "wrong_datum_wells": int(frame["wrong_datum"].sum()),
        "wrong_datum_fraction": float(frame["wrong_datum"].mean()),
        "tight_shape_wells": int(frame["tight_shape"].sum()),
        "tight_shape_fraction": float(frame["tight_shape"].mean()),
    }
    for name, _, _ in STRATA:
        count = int(frame[f"{name}_rows"].sum())
        report[f"{name}_rmse"] = (
            float(math.sqrt(frame[f"{name}_sse"].sum() / count)) if count else float("nan")
        )
        report[f"{name}_datum_oracle_rmse"] = (
            float(math.sqrt(frame[f"{name}_datum_corrected_sse"].sum() / count))
            if count
            else float("nan")
        )
        report[f"{name}_affine_oracle_rmse"] = (
            float(math.sqrt(frame[f"{name}_affine_corrected_sse"].sum() / count))
            if count
            else float("nan")
        )
    return report


def _concentration(frame):
    ordered = frame.sort_values("exact_sse", ascending=False).reset_index(drop=True)
    total_sse = float(ordered["exact_sse"].sum())
    rows = []
    for fraction in (0.10, 0.15, 0.20):
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        subset = ordered.iloc[:count]
        rows.append(
            {
                "worst_well_fraction": fraction,
                "wells": count,
                "sse_fraction": float(subset["exact_sse"].sum() / total_sse),
                "datum_sse_fraction_within_subset": float(
                    subset["datum_sse"].sum() / max(subset["exact_sse"].sum(), 1e-12)
                ),
                "tight_shape_fraction": float(subset["tight_shape"].mean()),
                "wrong_datum_fraction": float(subset["wrong_datum"].mean()),
            }
        )
    return pd.DataFrame(rows)


if POOL_PATH is None:
    raise FileNotFoundError("Cannot find merged OOF candidate pool")
fold_map = _load_fold_map()
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
models = [column for column in MODEL_PRIORITY if column in header]
if not models:
    raise RuntimeError(f"No requested model columns in pool header: {header}")
identity = [column for column in ("id", "well", "row_idx", "_row_idx") if column in header]
if "id" not in identity and not ({"well", "row_idx"}.issubset(identity) or {"well", "_row_idx"}.issubset(identity)):
    raise RuntimeError("OOF pool has no usable row identity")

started = time.time()
pool = pd.read_csv(POOL_PATH, usecols=identity + models)
if "well" not in pool:
    pool["well"] = pool["id"].astype(str).str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool:
    if "_row_idx" in pool:
        pool["row_idx"] = pd.to_numeric(pool["_row_idx"], errors="raise").astype(np.int32)
    else:
        pool["row_idx"] = pd.to_numeric(
            pool["id"].astype(str).str.rsplit("_", n=1).str[-1], errors="raise"
        ).astype(np.int32)
if pool.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("OOF pool has duplicate well/row_idx rows")
if pool["well"].nunique() != 773:
    raise RuntimeError(f"Expected 773 OOF wells; found {pool['well'].nunique()}")
invalid_models = [
    model
    for model in models
    if not np.isfinite(pd.to_numeric(pool[model], errors="coerce").to_numpy(np.float64)).all()
]
if invalid_models:
    print("SKIPPING NON-FINITE OPTIONAL MODELS", invalid_models, flush=True)
    models = [model for model in models if model not in invalid_models]
if "strong" in header and "strong" not in models:
    raise RuntimeError("The primary strong OOF column is non-finite")
if not models:
    raise RuntimeError("No finite model columns remain")

well_rows = []
for number, (well, group) in enumerate(pool.groupby("well", sort=True), start=1):
    if well not in fold_map:
        raise RuntimeError(f"Fold metadata misses {well}")
    group = group.sort_values("row_idx")
    row_index = group["row_idx"].to_numpy(np.int64)
    truth, distance = _native_arrays(well, row_index)
    if not np.isfinite(truth).all() or not np.isfinite(distance).all():
        raise RuntimeError(f"{well}: non-finite native truth/distance")
    for model in models:
        prediction = pd.to_numeric(group[model], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{well}/{model}: non-finite OOF prediction")
        well_rows.append(
            _well_row(well, fold_map[well], model, prediction, truth, distance)
        )
    if number % 50 == 0 or number == pool["well"].nunique():
        print(f"shape/datum decomposition {number}/{pool['well'].nunique()}", flush=True)

bywell = pd.DataFrame(well_rows)
bywell.to_csv(OUT_DIR / "bywell_decomposition.csv", index=False)
model_reports = {}
summary_rows = []
fold_rows = []
class_rows = []
concentration_rows = []
for model, frame in bywell.groupby("model", sort=False):
    report = _pooled_report(frame)
    model_reports[model] = report
    summary_rows.append({"model": model, **report})
    for fold, fold_frame in frame.groupby("fold", sort=True):
        fold_rows.append({"model": model, "fold": int(fold), **_pooled_report(fold_frame)})
    for failure_class, class_frame in frame.groupby("failure_class", sort=True):
        class_report = _pooled_report(class_frame)
        class_rows.append(
            {
                "model": model,
                "failure_class": failure_class,
                "overall_sse_fraction": float(
                    class_frame["exact_sse"].sum() / frame["exact_sse"].sum()
                ),
                **class_report,
            }
        )
    concentration = _concentration(frame)
    concentration.insert(0, "model", model)
    concentration_rows.append(concentration)

summary = pd.DataFrame(summary_rows).sort_values("rmse")
fold_summary = pd.DataFrame(fold_rows)
class_summary = pd.DataFrame(class_rows)
concentration_summary = pd.concat(concentration_rows, ignore_index=True)
summary.to_csv(OUT_DIR / "model_summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
class_summary.to_csv(OUT_DIR / "failure_class_summary.csv", index=False)
concentration_summary.to_csv(OUT_DIR / "tail_concentration.csv", index=False)

primary_model = next((model for model in MODEL_PRIORITY if model in model_reports), models[0])
primary_frame = bywell[bywell["model"].eq(primary_model)].copy()
primary_frame.sort_values("exact_sse", ascending=False).head(
    max(1, int(math.ceil(0.20 * len(primary_frame))))
).to_csv(OUT_DIR / "primary_worst20pct_wells.csv", index=False)
primary = model_reports[primary_model]
checks = {
    "datum_only_reaches_5p5": bool(primary["datum_oracle_rmse"] <= TARGET_RMSE_FT),
    "datum_plus_linear_reaches_5p5": bool(primary["affine_oracle_rmse"] <= TARGET_RMSE_FT),
    "sse_closure": bool(float(bywell["sse_closure_relative_error"].max()) <= 1e-10),
}
if checks["datum_only_reaches_5p5"]:
    next_action = "Datum dominates -> build a legal per-well datum estimator before another shape model"
elif checks["datum_plus_linear_reaches_5p5"]:
    next_action = "Low-dimensional shape is sufficient -> build a legal datum+slope complete-well model"
else:
    next_action = "Nonlinear shape remains above 5.5 -> prioritize a legal complete-well shape model"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "foldclean_oof_shape_datum_exact_sse_decomposition",
    "pool": str(POOL_PATH),
    "fold_source": str(FOLD_PATH),
    "models": models,
    "primary_model": primary_model,
    "thresholds": {
        "wrong_datum_abs_bias_ge_ft": DATUM_TAIL_THRESHOLD_FT,
        "tight_shape_rmse_le_ft": TIGHT_SHAPE_THRESHOLD_FT,
        "target_rmse_ft": TARGET_RMSE_FT,
    },
    "primary": primary,
    "checks": checks,
    "representation_pass": bool(checks["datum_plus_linear_reaches_5p5"]),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": next_action,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("MODEL SUMMARY")
print(summary.to_string(index=False), flush=True)
print("TAIL CONCENTRATION")
print(
    concentration_summary[concentration_summary["model"].eq(primary_model)].to_string(index=False),
    flush=True,
)
print("FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)
