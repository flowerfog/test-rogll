"""Anchored low-dimensional residual-basis oracle.

The correction basis is constrained to be exactly zero at prediction start.
Folds 2-5 choose one fixed basis family using oracle per-well coefficients;
Fold 1 is then evaluated once as a representation-capacity reproduction.
Validation truth is used to fit coefficients, so this cell is diagnostic only.
It does not train or validate a deployable coefficient predictor.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


EXPERIMENT = "anchored_residual_basis_oracle_v1"
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

MODELS = ("strong", "strong_bias", "cp12")
PRIMARY_MODEL = "strong"
SELECTION_FOLDS = (2, 3, 4, 5)
REPRODUCTION_FOLD = 1
TAUS_FT = (64.0, 128.0, 256.0, 512.0)
TARGET_RMSE = 4.5
TARGET_FAR_RMSE = 5.5
MAX_NEAR_DEGRADATION = 0.2
MAX_CONDITION_P95 = 1e8
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


def _fold_map():
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
    return frame.set_index("well")["fold"].to_dict()


def _native(well, row_index):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "TVT", "TVT_input"])
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise RuntimeError(f"{well}: no known TVT_input")
    official = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    row_index = np.asarray(row_index, np.int64)
    if not np.array_equal(row_index, official):
        raise RuntimeError(f"{well}: OOF rows do not reproduce official mask")
    distance = md[row_index] - float(md[known[-1]])
    return truth[row_index], distance


def _basis_configs():
    configs = [{"config": "linear_d1", "tau_ft": np.nan, "terms": ("linear",)}]
    for tau in TAUS_FT:
        configs.append(
            {
                "config": f"ramp{int(tau)}_linear_d2",
                "tau_ft": tau,
                "terms": ("ramp", "linear"),
            }
        )
        configs.append(
            {
                "config": f"ramp{int(tau)}_linear_quad_d3",
                "tau_ft": tau,
                "terms": ("ramp", "linear", "quadratic"),
            }
        )
    return configs


CONFIGS = _basis_configs()
CONFIG_LOOKUP = {config["config"]: config for config in CONFIGS}


def _design(distance_ft, config):
    distance = np.asarray(distance_ft, np.float64)
    d1000 = distance / 1000.0
    progress = distance / max(float(np.max(distance)), 1.0)
    columns = []
    for term in config["terms"]:
        if term == "ramp":
            columns.append(1.0 - np.exp(-distance / float(config["tau_ft"])))
        elif term == "linear":
            columns.append(d1000)
        elif term == "quadratic":
            columns.append(progress * progress)
        else:
            raise ValueError(term)
    design = np.column_stack(columns).astype(np.float64)
    if np.max(np.abs(_design_at_zero(config))) > 1e-12:
        raise RuntimeError(f"Basis {config['config']} violates the anchor")
    return design


def _design_at_zero(config):
    values = []
    for term in config["terms"]:
        if term in ("ramp", "linear", "quadratic"):
            values.append(0.0)
        else:
            raise ValueError(term)
    return np.asarray(values, np.float64)


def _fit_oracle(prediction, truth, distance, config):
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    design = _design(distance, config)
    target = truth - prediction
    try:
        coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
        correction = design @ coefficients
        condition = (
            float(singular[0] / singular[-1])
            if len(singular) and singular[-1] > 1e-12
            else float("inf")
        )
        failed = not (
            int(rank) == design.shape[1]
            and np.isfinite(coefficients).all()
            and np.isfinite(correction).all()
        )
    except Exception:
        coefficients = np.full(design.shape[1], np.nan, np.float64)
        correction = np.zeros(len(target), np.float64)
        rank = 0
        condition = float("inf")
        failed = True
    corrected = prediction + correction
    return corrected, coefficients, int(rank), condition, bool(failed)


def _metric_row(well, fold, model, config_name, prediction, corrected, truth, distance,
                coefficients, rank, condition, failed):
    error = np.asarray(prediction, np.float64) - truth
    corrected_error = np.asarray(corrected, np.float64) - truth
    output = {
        "well": well,
        "fold": int(fold),
        "model": model,
        "config": config_name,
        "dimensions": int(len(coefficients)),
        "rows": len(truth),
        "base_sse": float(np.sum(error * error)),
        "corrected_sse": float(np.sum(corrected_error * corrected_error)),
        "base_mae_sum": float(np.sum(np.abs(error))),
        "corrected_mae_sum": float(np.sum(np.abs(corrected_error))),
        "condition_number": condition,
        "solve_rank": rank,
        "solve_failed": failed,
        "anchor_correction_abs_max": 0.0,
    }
    for index, coefficient in enumerate(coefficients):
        output[f"coefficient_{index}"] = float(coefficient)
    for name, low, high in STRATA:
        mask = (distance > low if low > 0 else distance >= low) & (distance <= high)
        output[f"{name}_rows"] = int(mask.sum())
        output[f"{name}_base_sse"] = float(np.sum(error[mask] ** 2))
        output[f"{name}_corrected_sse"] = float(np.sum(corrected_error[mask] ** 2))
    return output


def _summarise(frame):
    rows = int(frame["rows"].sum())
    base_sse = float(frame["base_sse"].sum())
    corrected_sse = float(frame["corrected_sse"].sum())
    finite_condition = frame.loc[np.isfinite(frame["condition_number"]), "condition_number"]
    output = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "exact_base_sse": base_sse,
        "exact_corrected_sse": corrected_sse,
        "base_rmse": float(math.sqrt(base_sse / rows)),
        "rmse": float(math.sqrt(corrected_sse / rows)),
        "gain": float(math.sqrt(base_sse / rows) - math.sqrt(corrected_sse / rows)),
        "mae": float(frame["corrected_mae_sum"].sum() / rows),
        "solve_failures": int(frame["solve_failed"].sum()),
        "condition_p50": float(finite_condition.median()) if len(finite_condition) else float("inf"),
        "condition_p95": float(finite_condition.quantile(0.95)) if len(finite_condition) else float("inf"),
        "condition_max": float(finite_condition.max()) if len(finite_condition) else float("inf"),
        "anchor_correction_abs_max": float(frame["anchor_correction_abs_max"].max()),
    }
    for name, _, _ in STRATA:
        count = int(frame[f"{name}_rows"].sum())
        base = float(frame[f"{name}_base_sse"].sum())
        corrected = float(frame[f"{name}_corrected_sse"].sum())
        output[f"{name}_base_rmse"] = float(math.sqrt(base / count)) if count else float("nan")
        output[f"{name}_rmse"] = float(math.sqrt(corrected / count)) if count else float("nan")
        output[f"{name}_gain"] = (
            output[f"{name}_base_rmse"] - output[f"{name}_rmse"] if count else float("nan")
        )
    output["near_degradation"] = output["near_0_500_rmse"] - output["near_0_500_base_rmse"]
    return output


if POOL_PATH is None:
    raise FileNotFoundError("Cannot find merged OOF candidate pool")
fold_map = _fold_map()
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
models = [model for model in MODELS if model in header]
if PRIMARY_MODEL not in models:
    raise RuntimeError(f"Primary model {PRIMARY_MODEL} missing from OOF pool")
identity = [column for column in ("id", "well", "row_idx", "_row_idx") if column in header]
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
if pool["well"].nunique() != 773:
    raise RuntimeError(f"Expected 773 wells; found {pool['well'].nunique()}")
if pool.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("Duplicate OOF rows")

started = time.time()
rows = []
for number, (well, group) in enumerate(pool.groupby("well", sort=True), start=1):
    group = group.sort_values("row_idx")
    truth, distance = _native(well, group["row_idx"].to_numpy(np.int64))
    fold = fold_map[well]
    for model in models:
        prediction = pd.to_numeric(group[model], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{well}/{model}: non-finite prediction")
        for config in CONFIGS:
            corrected, coefficients, rank, condition, failed = _fit_oracle(
                prediction, truth, distance, config
            )
            rows.append(
                _metric_row(
                    well,
                    fold,
                    model,
                    config["config"],
                    prediction,
                    corrected,
                    truth,
                    distance,
                    coefficients,
                    rank,
                    condition,
                    failed,
                )
            )
    if number % 50 == 0 or number == pool["well"].nunique():
        print(f"anchored basis oracle {number}/{pool['well'].nunique()}", flush=True)

bywell = pd.DataFrame(rows)
bywell.to_csv(OUT_DIR / "all_config_bywell.csv", index=False)
scan_rows = []
for (model, config), frame in bywell.groupby(["model", "config"], sort=True):
    selection = frame[frame["fold"].isin(SELECTION_FOLDS)]
    reproduction = frame[frame["fold"].eq(REPRODUCTION_FOLD)]
    scan_rows.append(
        {
            "model": model,
            "config": config,
            "dimensions": int(frame["dimensions"].iloc[0]),
            **{f"selection_{key}": value for key, value in _summarise(selection).items()},
            **{f"fold1_{key}": value for key, value in _summarise(reproduction).items()},
            **{f"all_{key}": value for key, value in _summarise(frame).items()},
        }
    )
scan = pd.DataFrame(scan_rows)
scan.to_csv(OUT_DIR / "config_scan.csv", index=False)

primary_scan = scan[scan["model"].eq(PRIMARY_MODEL)].copy()
eligible = primary_scan[
    (primary_scan["selection_solve_failures"] == 0)
    & (primary_scan["selection_near_degradation"] <= MAX_NEAR_DEGRADATION)
    & (primary_scan["selection_condition_p95"] < MAX_CONDITION_P95)
]
if eligible.empty:
    raise RuntimeError("No anchored basis passes numerical and near-distance guards")
selected_row = eligible.sort_values(
    ["selection_rmse", "dimensions", "config"], ascending=[True, True, True]
).iloc[0]
selected_config = str(selected_row["config"])
selected_bywell = bywell[
    bywell["model"].eq(PRIMARY_MODEL) & bywell["config"].eq(selected_config)
].copy()
selected_bywell.to_csv(OUT_DIR / "selected_bywell.csv", index=False)

fold_reports = pd.DataFrame(
    [
        {"fold": int(fold), **_summarise(frame)}
        for fold, frame in selected_bywell.groupby("fold", sort=True)
    ]
)
fold_reports.to_csv(OUT_DIR / "selected_fold_summary.csv", index=False)
overall = _summarise(selected_bywell)
fold1 = _summarise(selected_bywell[selected_bywell["fold"].eq(REPRODUCTION_FOLD)])
selection = _summarise(selected_bywell[selected_bywell["fold"].isin(SELECTION_FOLDS)])

worst_count = max(1, int(math.ceil(0.20 * len(selected_bywell))))
worst = selected_bywell.sort_values("base_sse", ascending=False).head(worst_count)
worst.to_csv(OUT_DIR / "selected_worst20pct_wells.csv", index=False)
worst_report = _summarise(worst)

checks = {
    "overall_rmse": bool(overall["rmse"] <= TARGET_RMSE),
    "far_rmse": bool(overall["far_gt2000_rmse"] <= TARGET_FAR_RMSE),
    "near_guard": bool(overall["near_degradation"] <= MAX_NEAR_DEGRADATION),
    "fold1_reproduction": bool(fold1["rmse"] <= TARGET_RMSE),
    "all_folds_below_5p5": bool((fold_reports["rmse"] <= 5.5).all()),
    "condition": bool(overall["condition_p95"] < MAX_CONDITION_P95),
    "solve": bool(overall["solve_failures"] == 0),
    "anchor": bool(overall["anchor_correction_abs_max"] <= 1e-12),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "anchored_lowdim_residual_representation_oracle",
    "truth_usage": "per-well oracle coefficient fit; diagnostic only",
    "pool": str(POOL_PATH),
    "fold_source": str(FOLD_PATH),
    "primary_model": PRIMARY_MODEL,
    "selection_folds": list(SELECTION_FOLDS),
    "reproduction_fold": REPRODUCTION_FOLD,
    "selected_config": selected_config,
    "selected_dimensions": int(selected_row["dimensions"]),
    "selection": selection,
    "fold1": fold1,
    "overall": overall,
    "worst20pct": worst_report,
    "gates": {
        "overall_rmse_le": TARGET_RMSE,
        "far_rmse_le": TARGET_FAR_RMSE,
        "near_degradation_le": MAX_NEAR_DEGRADATION,
        "fold1_rmse_le": TARGET_RMSE,
        "each_fold_rmse_le": 5.5,
        "condition_p95_lt": MAX_CONDITION_P95,
        "solve_failures": 0,
        "anchor_abs_max": 1e-12,
    },
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["next"] = (
    "PASS -> train a complete-well head to predict the frozen anchored coefficients"
    if verdict["strict_pass"]
    else "FAIL -> anchored low-dimensional correction does not preserve the 5.5 ceiling"
)
_write_json(OUT_DIR / "verdict.json", verdict)
print("PRIMARY CONFIG SCAN")
print(
    primary_scan[
        [
            "config",
            "dimensions",
            "selection_rmse",
            "selection_near_degradation",
            "selection_far_gt2000_rmse",
            "fold1_rmse",
            "all_rmse",
            "all_near_degradation",
            "all_far_gt2000_rmse",
            "all_condition_p95",
        ]
    ].sort_values("selection_rmse").to_string(index=False),
    flush=True,
)
print("SELECTED FOLDS")
print(fold_reports.to_string(index=False), flush=True)
print("FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)

