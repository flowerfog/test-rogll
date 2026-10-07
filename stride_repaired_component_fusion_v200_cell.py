"""V200: five-fold repaired STRIDE endpoints and nested V168 fusion.

Decode three truth-free STRIDE component variants on canonical held-out folds:
raw typewell reference, zero persistence, and both changes together.  Save one
wide full-row artifact, then evaluate strictly cross-fitted scalar, distance-
band, and non-negative residual-simplex consumers against canonical V168.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "stride_repaired_component_fusion_v200"
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.time()

SOURCE_CANDIDATES = (
    ROOT / "stride_joint_segment_decoder_gateway_v1_cell.py",
    Path.cwd() / "stride_joint_segment_decoder_gateway_v1_cell.py",
)
SOURCE = next((path for path in SOURCE_CANDIDATES if path.is_file()), None)
if SOURCE is None:
    raise FileNotFoundError("Upload stride_joint_segment_decoder_gateway_v1_cell.py")

BASE_PATH = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
BASE_COLUMN = "metadata_analog_w0p05"
PSG_PATH = OOF / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet"
PSG_COLUMN = "psg_tvt"
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
EXPECTED_BASE_SHA = "43f23cb1f53212e8da77b38e873b24ae53ddd092a8f2fa16fa761e7a1536e8c6"
MODES = ("no_calibration", "no_persistence", "raw_no_persistence")
DISTANCE_EDGES = np.array([0.0, 500.0, 1000.0, 2000.0, np.inf], np.float64)
DISTANCE_NAMES = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")


def _bootstrap(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cutoff = None
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.If):
            expression = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "FOLD_PATH is None" in expression:
                cutoff = index
                break
    if cutoff is None:
        raise RuntimeError("Cannot find STRIDE definition boundary")

    class DisableNumbaCache(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name == "njit":
                for keyword in node.keywords:
                    if keyword.arg == "cache":
                        keyword.value = ast.Constant(False)
                        break
                else:
                    node.keywords.append(ast.keyword(arg="cache", value=ast.Constant(False)))
            return node

    body = [DisableNumbaCache().visit(node) for node in tree.body[:cutoff]]
    module_tree = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module_tree)
    name = "stride_v200_defs"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(module_tree, str(path), "exec"), module.__dict__)
    return module


V = _bootstrap(SOURCE)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _decode_raw_no_persistence(record, priors):
    tw_tvt = record["tw_tvt"]
    tw_gr = record["tw_gr_raw"]
    expected = np.interp(record["tvt_input"], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    known = np.isfinite(record["tvt_input"]) & np.isfinite(expected)
    _, gr_scale = V._robust_location_scale(record["gr"][known] - expected[known])
    gr_scale = float(np.clip(gr_scale, 8.0, 60.0))
    node_rows = V._node_rows(record)
    path_node, objective = V._joint_decode(
        record["md"], record["gr"], node_rows,
        record["anchor_tvt"], record["initial_rate"],
        tw_tvt, tw_gr, gr_scale,
        priors["length_nodes"], priors["length_log_mu"], priors["length_log_sigma"],
        priors["rate_change_scale"], priors["rate_low"], priors["rate_high"],
        V.RATE_DELTA_MULTIPLIERS, V.BEAM_SIZE, 0.0,
        V.LENGTH_PRIOR_WEIGHT, False, V.EMISSION_SAMPLES_PER_SEGMENT,
    )
    prediction = np.interp(record["md"][record["evaluation"]], record["md"][node_rows], path_node)
    return prediction, float(objective), len(node_rows)


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, names = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error * error, minlength=len(names))
    well_rmse = np.sqrt(sse / counts)
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    for index, name in enumerate(DISTANCE_NAMES):
        mask = (distance >= DISTANCE_EDGES[index]) & (distance < DISTANCE_EDGES[index + 1])
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
    return result


def _distance_detrend(error: np.ndarray, distance: np.ndarray, bins: int = 64) -> np.ndarray:
    edges = np.unique(np.quantile(distance, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.clip(np.searchsorted(edges, distance, side="right") - 1, 0, len(edges) - 2)
    means = np.bincount(labels, weights=error, minlength=len(edges) - 1) / np.maximum(
        np.bincount(labels, minlength=len(edges) - 1), 1
    )
    return error - means[labels]


def _project_simplex_residual(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, np.float64), 0.0)
    if values.sum() <= 1.0:
        return values
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(values) + 1)
    valid = ordered - (cumulative - 1.0) / indices > 0.0
    rho = int(np.flatnonzero(valid)[-1])
    threshold = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(values - threshold, 0.0)


def _fit_simplex(base_error: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    gram = deltas @ deltas.T / deltas.shape[1]
    linear = -(deltas @ base_error) / deltas.shape[1]
    largest = float(np.max(np.linalg.eigvalsh(gram)))
    step = 1.0 / max(2.0 * largest, 1e-12)
    weights = np.zeros(len(deltas), np.float64)
    for _ in range(2000):
        updated = _project_simplex_residual(weights - step * 2.0 * (gram @ weights - linear))
        if np.max(np.abs(updated - weights)) < 1e-10:
            return updated
        weights = updated
    return weights


def _fold_report(frame: pd.DataFrame, base_prediction: np.ndarray, prediction: np.ndarray, endpoint: str) -> list[dict]:
    reports = []
    for fold in range(1, 6):
        mask = frame["fold"].to_numpy(np.int8) == fold
        local = frame.loc[mask]
        base_row = _metrics(local, base_prediction[mask], "base")
        row = _metrics(local, prediction[mask], endpoint)
        reports.append({
            "fold": fold,
            "endpoint": endpoint,
            "base_rmse": base_row["rmse"],
            "rmse": row["rmse"],
            "gain": base_row["rmse"] - row["rmse"],
            "far_gain": base_row["far_gt2000_rmse"] - row["far_gt2000_rmse"],
            "p95_gain": base_row["p95_well_rmse"] - row["p95_well_rmse"],
        })
    return reports


if not BASE_PATH.is_file():
    raise FileNotFoundError(BASE_PATH)
base = pd.read_parquet(BASE_PATH)
required = {"well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN}
missing = sorted(required.difference(base.columns))
if missing:
    raise RuntimeError(f"V168 missing columns: {missing}")
base["well"] = base["well"].astype(str)
base["row_idx"] = pd.to_numeric(base["row_idx"], errors="raise").astype(np.int64)
base["fold"] = pd.to_numeric(base["fold"], errors="raise").astype(np.int8)
base = base.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Invalid V168 coverage: {len(base)}/{base['well'].nunique()}")
base_prediction = base[BASE_COLUMN].to_numpy(np.float32)
base_sha = hashlib.sha256(base_prediction.tobytes()).hexdigest()
if base_sha != EXPECTED_BASE_SHA:
    raise RuntimeError(f"V168 SHA mismatch: {base_sha}")

folds = pd.read_csv(V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
if len(folds) != EXPECTED_WELLS or set(folds["fold"]) != {1, 2, 3, 4, 5}:
    raise RuntimeError("Invalid canonical fold table")

well_positions = {well: index.to_numpy(np.int64) for well, index in base.groupby("well", sort=False).groups.items()}
predictions = {mode: np.full(EXPECTED_ROWS, np.nan, np.float32) for mode in MODES}
objectives = []

for heldout in range(1, 6):
    train_wells = folds.loc[folds["fold"].ne(heldout), "well"].sort_values().tolist()
    valid_wells = folds.loc[folds["fold"].eq(heldout), "well"].sort_values().tolist()
    train_records = []
    for number, well in enumerate(train_wells, 1):
        train_records.append(V._load_record(well))
        if number % 100 == 0 or number == len(train_wells):
            print(f"V200 fold{heldout} priors {number}/{len(train_wells)}", flush=True)
    priors = V._fit_outer_priors(train_records)
    del train_records
    gc.collect()

    for number, well in enumerate(valid_wells, 1):
        record = V._load_record(well)
        positions = well_positions[well]
        expected_rows = base.loc[positions, "row_idx"].to_numpy(np.int64)
        if not np.array_equal(expected_rows, record["evaluation"].astype(np.int64)):
            raise RuntimeError(f"{well}: evaluation row mismatch")
        for mode in MODES:
            if mode == "raw_no_persistence":
                prediction, objective, nodes = _decode_raw_no_persistence(record, priors)
            else:
                prediction, objective, nodes = V._decode(record, priors, mode=mode)
            predictions[mode][positions] = np.asarray(prediction, np.float32)
            objectives.append({"fold": heldout, "well": well, "mode": mode, "objective": objective, "nodes": nodes})
        if number % 25 == 0 or number == len(valid_wells):
            print(
                f"V200 fold{heldout} decode {number}/{len(valid_wells)} "
                f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True
            )
    del priors
    gc.collect()

for mode, values in predictions.items():
    if not np.isfinite(values).all():
        raise RuntimeError(f"Incomplete {mode}: {np.isfinite(values).sum()}/{len(values)}")

wide = base[["well", "row_idx", "fold"]].copy()
for mode, values in predictions.items():
    wide[f"stride_{mode}"] = values
wide.to_parquet(OUT / "repaired_stride_fullrow_oof.parquet", index=False)
pd.DataFrame(objectives).to_csv(OUT / "decoder_bywell.csv", index=False)

# Add PSG only when exact canonical alignment succeeds.
candidate_arrays = {f"stride_{mode}": values for mode, values in predictions.items()}
if PSG_PATH.is_file() and PSG_COLUMN in pq.ParquetFile(PSG_PATH).schema.names:
    psg = pd.read_parquet(PSG_PATH, columns=["well", "row_idx", PSG_COLUMN])
    psg["well"] = psg["well"].astype(str)
    psg["row_idx"] = pd.to_numeric(psg["row_idx"], errors="raise").astype(np.int64)
    psg = psg.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if len(psg) == len(base) and np.array_equal(psg["well"], base["well"]) and np.array_equal(psg["row_idx"], base["row_idx"]):
        psg_values = psg[PSG_COLUMN].to_numpy(np.float32)
        if np.isfinite(psg_values).all():
            candidate_arrays["psg"] = psg_values

truth = base["tvt_true"].to_numpy(np.float64)
distance = base["md_since"].to_numpy(np.float64)
fold_values = base["fold"].to_numpy(np.int8)
base_error = base_prediction.astype(np.float64) - truth

standalone_rows = [_metrics(base, base_prediction, "v168")]
pair_rows = []
consumer_predictions = {}
weight_rows = []

for name, candidate in candidate_arrays.items():
    candidate64 = candidate.astype(np.float64)
    standalone_rows.append(_metrics(base, candidate64, name))
    candidate_error = candidate64 - truth
    pair_rows.append({
        "endpoint": name,
        "raw_error_corr": float(np.corrcoef(base_error, candidate_error)[0, 1]),
        "detrended_error_corr": float(np.corrcoef(
            _distance_detrend(base_error, distance), _distance_detrend(candidate_error, distance)
        )[0, 1]),
    })
    delta = candidate64 - base_prediction.astype(np.float64)
    nested_global = base_prediction.astype(np.float64).copy()
    nested_band = base_prediction.astype(np.float64).copy()
    for heldout in range(1, 6):
        train = fold_values != heldout
        held = ~train
        alpha = float(np.clip(-np.dot(base_error[train], delta[train]) / max(np.dot(delta[train], delta[train]), 1e-12), 0.0, 1.0))
        nested_global[held] += alpha * delta[held]
        weight_rows.append({"consumer": f"nested_global/{name}", "held_fold": heldout, "band": "all", "weight": alpha})
        for band_index, band_name in enumerate(DISTANCE_NAMES):
            train_band = train & (distance >= DISTANCE_EDGES[band_index]) & (distance < DISTANCE_EDGES[band_index + 1])
            held_band = held & (distance >= DISTANCE_EDGES[band_index]) & (distance < DISTANCE_EDGES[band_index + 1])
            alpha_band = float(np.clip(
                -np.dot(base_error[train_band], delta[train_band]) / max(np.dot(delta[train_band], delta[train_band]), 1e-12), 0.0, 1.0
            ))
            nested_band[held_band] += alpha_band * delta[held_band]
            weight_rows.append({"consumer": f"nested_band/{name}", "held_fold": heldout, "band": band_name, "weight": alpha_band})
    consumer_predictions[f"nested_global/{name}"] = nested_global
    consumer_predictions[f"nested_band/{name}"] = nested_band

# Joint non-negative residual simplex, fitted on the other four folds.
candidate_names = list(candidate_arrays)
candidate_matrix = np.stack([candidate_arrays[name] for name in candidate_names]).astype(np.float64)
deltas = candidate_matrix - base_prediction.astype(np.float64)[None, :]
nested_simplex = base_prediction.astype(np.float64).copy()
for heldout in range(1, 6):
    train = fold_values != heldout
    held = ~train
    weights = _fit_simplex(base_error[train], deltas[:, train])
    nested_simplex[held] += weights @ deltas[:, held]
    for name, weight in zip(candidate_names, weights):
        weight_rows.append({"consumer": "nested_simplex", "held_fold": heldout, "band": name, "weight": float(weight)})
consumer_predictions["nested_simplex"] = nested_simplex

summary_rows = standalone_rows.copy()
fold_reports = []
for name, prediction in consumer_predictions.items():
    summary_rows.append(_metrics(base, prediction, name))
    fold_reports.extend(_fold_report(base, base_prediction, prediction, name))

summary = pd.DataFrame(summary_rows).sort_values("rmse", ignore_index=True)
base_rmse = float(summary.loc[summary["endpoint"].eq("v168"), "rmse"].iloc[0])
summary["gain_vs_v168"] = base_rmse - summary["rmse"]
fold_report = pd.DataFrame(fold_reports)
consumer_stability = fold_report.groupby("endpoint", as_index=False).agg(
    positive_folds=("gain", lambda values: int((values > 0.0).sum())),
    minimum_fold_gain=("gain", "min"),
    mean_fold_gain=("gain", "mean"),
)
consumer_stability = consumer_stability.merge(
    summary[["endpoint", "rmse", "gain_vs_v168"]], on="endpoint", how="left", validate="one_to_one"
).sort_values("rmse", ignore_index=True)

eligible = consumer_stability.loc[
    consumer_stability["positive_folds"].ge(4)
    & consumer_stability["minimum_fold_gain"].ge(-0.03)
    & consumer_stability["gain_vs_v168"].ge(0.10)
]
selected = eligible.iloc[0].to_dict() if len(eligible) else None

summary.to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(pair_rows).to_csv(OUT / "dependence.csv", index=False)
fold_report.to_csv(OUT / "folds.csv", index=False)
consumer_stability.to_csv(OUT / "consumer_stability.csv", index=False)
pd.DataFrame(weight_rows).to_csv(OUT / "weights.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_repaired_STRIDE_component_generation_and_nested_fusion",
    "base": {"path": BASE_PATH, "column": BASE_COLUMN, "sha256": base_sha, "rmse": base_rmse},
    "modes": list(MODES),
    "candidate_endpoints": candidate_names,
    "coverage": {"rows": len(base), "wells": base["well"].nunique()},
    "selected_stable_consumer": selected,
    "checks": {
        "all_predictions_complete": all(np.isfinite(values).all() for values in predictions.values()),
        "stable_consumer_exists": selected is not None,
        "selected_gain_ge_0p10": bool(selected is not None and selected["gain_vs_v168"] >= 0.10),
        "selected_positive_folds_ge_4": bool(selected is not None and selected["positive_folds"] >= 4),
        "selected_minimum_fold_gain_ge_minus0p03": bool(selected is not None and selected["minimum_fold_gain"] >= -0.03),
    },
    "strict_pass": selected is not None,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the selected consumer and build the corresponding hidden-test inference bundle"
        if selected is not None
        else "FAIL -> repaired V155 components remain insufficient; retain V168 and stop this STRIDE implementation"
    ),
}
_write_json(OUT / "verdict.json", verdict)

print("\nV200 STANDALONE AND CONSUMERS")
print(summary.to_string(index=False), flush=True)
print("\nV200 CONSUMER STABILITY")
print(consumer_stability.to_string(index=False), flush=True)
print("\nV200 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT, flush=True)
