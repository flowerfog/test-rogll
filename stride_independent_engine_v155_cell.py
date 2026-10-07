"""V155: independent STRIDE-like five-fold OOF engine.

This wrapper deliberately reuses only the legal raw-data decoder definitions
from the existing STRIDE implementation.  It does not consume any historical
prediction while fitting priors.  Every held-out fold fits segment-length,
dip-persistence, and robust GR scale priors on the other four folds.
"""
from __future__ import annotations

import ast
import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "stride_independent_engine_v155"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
STARTED = time.time()

SOURCE_CANDIDATES = (
    ROOT / "stride_joint_segment_decoder_gateway_v1_cell.py",
    Path.cwd() / "stride_joint_segment_decoder_gateway_v1_cell.py",
)
SOURCE = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE is None:
    raise FileNotFoundError("Upload stride_joint_segment_decoder_gateway_v1_cell.py")


def _bootstrap(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cutoff = None
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.If):
            text = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "FOLD_PATH is None" in text:
                cutoff = index
                break
    if cutoff is None:
        raise RuntimeError("Cannot find the STRIDE definition boundary")
    # Numba cache entries created by a notebook/dynamic module refer to the
    # synthetic module name and cannot be imported after AST bootstrapping.
    # Disable only that cache layer; the compiled kernel remains unchanged.
    class _DisableDynamicNumbaCache(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                name = decorator.func.id if isinstance(decorator.func, ast.Name) else None
                if name != "njit":
                    continue
                found = False
                for keyword in decorator.keywords:
                    if keyword.arg == "cache":
                        keyword.value = ast.Constant(False)
                        found = True
                if not found:
                    decorator.keywords.append(ast.keyword(arg="cache", value=ast.Constant(False)))
            return node

    tree = _DisableDynamicNumbaCache().visit(tree)
    ast.fix_missing_locations(tree)
    module = types.ModuleType("stride_v155_defs")
    module.__file__ = str(path)
    exec(compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


V = _bootstrap(SOURCE)
FOLD_PATH = V.FOLD_PATH
DATA_ROOT = V.DATA_ROOT
_load_record = V._load_record
_fit_outer_priors = V._fit_outer_priors
_decode = V._decode

BASE_CANDIDATES = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
    OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet",
)
BASE_PATH = next((path for path in BASE_CANDIDATES if path.exists()), None)
if BASE_PATH is None:
    raise FileNotFoundError(BASE_CANDIDATES)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _fold_table():
    table = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
    table["well"] = table["well"].astype(str)
    table["fold"] = pd.to_numeric(table["fold"], errors="raise").astype(int)
    counts = table.groupby("fold")["well"].nunique().to_dict()
    if set(counts) != {1, 2, 3, 4, 5} or min(counts.values()) < 150:
        raise RuntimeError(f"Unexpected canonical folds: {counts}")
    return table.sort_values(["fold", "well"]).reset_index(drop=True)


def _load_base():
    schema = pq.ParquetFile(BASE_PATH).schema.names
    base_column = next(
        (name for name in ("incumbent_6p377_replay", "current_dose_0p20", "full_cp8_ordered") if name in schema),
        None,
    )
    if base_column is None:
        raise RuntimeError(f"No exact V76 column in {BASE_PATH}")
    frame = pd.read_parquet(BASE_PATH, columns=["well", "row_idx", "fold", "tvt_true", "md_since", base_column])
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(int)
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if len(frame) != 3_783_989 or frame[["tvt_true", base_column]].isna().any().any():
        raise RuntimeError(f"Invalid base OOF: {len(frame)} rows")
    return frame, base_column


def _decode_well(record, priors, temperature, mode):
    prediction, objective, nodes = _decode(record, priors, mode=mode)
    rows = record["evaluation"]
    return pd.DataFrame(
        {
            "well": record["well"],
            "row_idx": rows.astype(int),
            "prediction": prediction.astype(float),
            "objective": float(objective),
            "nodes": int(nodes),
        }
    )


def _metrics(frame, prediction_column):
    error = frame[prediction_column].to_numpy(float) - frame["tvt_true"].to_numpy(float)
    distance = frame["md_since"].to_numpy(float)
    well_sse = pd.Series(error * error).groupby(frame["well"].to_numpy()).sum()
    well_rows = frame.groupby("well", sort=False).size()
    well_rmse = np.sqrt(well_sse / well_rows)
    result = {
        "rows": int(len(error)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
    }
    for name, low, high in (("near_0_500", 0.0, 500.0), ("mid_500_1000", 500.0, 1000.0), ("mid_1000_2000", 1000.0, 2000.0), ("far_gt2000", 2000.0, np.inf)):
        mask = (distance >= low) & (distance < high)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _detrend(error, distance, wells):
    out = np.empty(len(error), float)
    for well, index in pd.Series(np.arange(len(error))).groupby(wells, sort=False):
        index = index.to_numpy(int)
        x = distance[index]
        y = error[index]
        if len(index) < 3 or np.ptp(x) <= 1e-6:
            out[index] = y - np.mean(y)
            continue
        slope, intercept = np.polyfit(x, y, 1)
        out[index] = y - (slope * x + intercept)
    return out


folds = _fold_table()
base, base_column = _load_base()
all_rows = []
component_rows = []
fold_reports = []
MODES = ("full", "gaussian", "no_calibration", "no_persistence")

for heldout in (1, 2, 3, 4, 5):
    train_wells = folds.loc[folds["fold"].ne(heldout), "well"].tolist()
    valid_wells = folds.loc[folds["fold"].eq(heldout), "well"].tolist()
    train_records = []
    for number, well in enumerate(train_wells, 1):
        train_records.append(_load_record(well))
        if number % 100 == 0 or number == len(train_wells):
            print(f"V155 fit fold{heldout} priors {number}/{len(train_wells)}", flush=True)
    priors = _fit_outer_priors(train_records)
    temperature = float(priors.get("emission_temperature", 1.0))
    for mode in MODES:
        parts = []
        for number, well in enumerate(valid_wells, 1):
            record = _load_record(well)
            part = _decode_well(record, priors, temperature, mode)
            part["fold"] = heldout
            part["mode"] = mode
            parts.append(part)
            if number % 25 == 0 or number == len(valid_wells):
                print(f"V155 decode fold{heldout} {mode} {number}/{len(valid_wells)}", flush=True)
        predicted = pd.concat(parts, ignore_index=True)
        predicted["id"] = predicted["well"] + "_" + predicted["row_idx"].astype(str)
        all_rows.append(predicted)
        component_rows.append({"fold": heldout, "mode": mode, **_metrics(predicted.merge(base[["well", "row_idx", "tvt_true", "md_since"]], on=["well", "row_idx"], validate="one_to_one"), "prediction")})

all_prediction = pd.concat(all_rows, ignore_index=True)
all_prediction.loc[all_prediction["mode"].eq("full")].to_parquet(
    OUT_DIR / "stride_v155_predictions.parquet", index=False
)
component = pd.DataFrame(component_rows)
component.to_csv(OUT_DIR / "component_by_fold.csv", index=False)

full = all_prediction.loc[all_prediction["mode"].eq("full")].merge(
    base[["well", "row_idx", "tvt_true", "md_since", base_column]],
    on=["well", "row_idx"], how="inner", validate="one_to_one",
)
if len(full) != len(all_prediction.loc[all_prediction["mode"].eq("full")]):
    raise RuntimeError("V155 prediction/base alignment failed")
full["stride_error"] = full["prediction"] - full["tvt_true"]
full["base_error"] = full[base_column] - full["tvt_true"]
full["stride_error_detrended"] = _detrend(full["stride_error"].to_numpy(), full["md_since"].to_numpy(), full["well"].to_numpy())
full["base_error_detrended"] = _detrend(full["base_error"].to_numpy(), full["md_since"].to_numpy(), full["well"].to_numpy())
delta = full["prediction"].to_numpy() - full[base_column].to_numpy()
base_error = full["base_error"].to_numpy()
alpha = float(np.clip(-np.dot(base_error, delta) / max(np.dot(delta, delta), 1e-12), 0.0, 1.0))
full["blend"] = full[base_column] + alpha * delta

summary = pd.DataFrame([
    {"endpoint": "base", **_metrics(full, base_column)},
    {"endpoint": "stride_full", **_metrics(full, "prediction")},
    {"endpoint": "blend_oracle_global", **_metrics(full, "blend")},
])
summary.to_csv(OUT_DIR / "summary.csv", index=False)
base_detrended = full["base_error_detrended"].to_numpy()
stride_detrended = full["stride_error_detrended"].to_numpy()
stride_rmse = float(summary.loc[summary.endpoint.eq("stride_full"), "rmse"].iloc[0])
base_rmse = float(summary.loc[summary.endpoint.eq("base"), "rmse"].iloc[0])
oracle_rmse = float(summary.loc[summary.endpoint.eq("blend_oracle_global"), "rmse"].iloc[0])
independent_corr = float(np.corrcoef(base_detrended, stride_detrended)[0, 1])
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_canonical_fivefold_raw_STRIDE_OOF",
    "base": str(BASE_PATH),
    "base_column": base_column,
    "protocol": {"heldout_folds": [1, 2, 3, 4, 5], "training_priors": "other_four_folds", "fold1_touched": True},
    "decoder": {"state": "segment_length_x_segment_TVT_rate", "likelihood": "Cauchy", "typewell_calibration": "robust_affine_inside_likelihood", "persistence": "dynamic_drilled_trend", "backtracking": True},
    "summary": summary.to_dict(orient="records"),
    "independence": {"raw_error_corr": float(np.corrcoef(base_error, full["stride_error"].to_numpy())[0, 1]), "detrended_error_corr": independent_corr},
    "global_oracle_alpha": alpha,
    "gates": {"standalone_rmse_le_10p5": bool(stride_rmse <= 10.5), "detrended_corr_le_0p60": bool(abs(independent_corr) <= 0.60), "oracle_gain_ge_0p20": bool(base_rmse - oracle_rmse >= 0.20)},
    "strict_pass": bool(stride_rmse <= 10.5 and abs(independent_corr) <= 0.60 and base_rmse - oracle_rmse >= 0.20),
    "next": "PASS -> fit nested cross-fold blend and application bundle" if (stride_rmse <= 10.5 and abs(independent_corr) <= 0.60 and base_rmse - oracle_rmse >= 0.20) else "FAIL -> independent diversity exists but decoder accuracy is insufficient; repair STRIDE path before blending",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("V155 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("V155 FINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
