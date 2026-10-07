"""Z-only active-steering endpoint and 6.377 cross-evidence gateway.

V54 established that the drilled Z trajectory contains transferable signal,
while X/Y tortuosity adds essentially nothing beyond Z. This experiment keeps
only that supported mechanism. Fold 4/5 train the endpoint, 24 deterministic
Fold-3 wells select its capacity and blend dose, and the remaining Fold-3
wells confirm both the mechanism and its incremental value over the 6.377
incumbent. Fold 2 is evaluated once only after confirmation passes. Fold 1 is
never touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


EXPERIMENT = "raw_z_steering_cross_evidence_v55"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = next(
    (
        path
        for path in (
            ROOT / "raw_active_steering_tortuosity_v54_cell.py",
            Path.cwd() / "raw_active_steering_tortuosity_v54_cell.py",
        )
        if path.is_file()
    ),
    None,
)
if SOURCE is None:
    raise FileNotFoundError("Upload raw_active_steering_tortuosity_v54_cell.py")

INCUMBENT_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
INCUMBENT_COLUMN = "selected_neighbor_dose"
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
WEIGHTS = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15)
DISTANCE_EDGES = (-np.inf, 500.0, 1000.0, 2000.0, np.inf)
DISTANCE_LABELS = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")


def _bootstrap(path: Path):
    source = path.read_text(encoding="utf-8")
    # Match only the top-level execution boundary. V54's own bootstrap helper
    # contains the same text as a quoted, indented string literal.
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_active_steering_v54_definitions_for_v55"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    definitions = source.split(marker, 1)[0] + "\n"
    exec(compile(definitions, str(path), "exec"), module.__dict__)
    return module


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _read_incumbent(fold: int):
    if not INCUMBENT_PATH.is_file():
        raise FileNotFoundError(INCUMBENT_PATH)
    columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", INCUMBENT_COLUMN]
    frame = pd.read_parquet(INCUMBENT_PATH, columns=columns, filters=[("fold", "=", int(fold))])
    frame["well"] = frame["well"].astype(str)
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if frame.empty or frame["id"].duplicated().any():
        raise RuntimeError(f"Invalid incumbent Fold-{fold}: rows={len(frame)}")
    if frame[["tvt_true", "md_since", INCUMBENT_COLUMN]].isna().any().any():
        raise RuntimeError(f"Missing incumbent values in Fold-{fold}")
    return frame


def _node_predictions(model, cache, columns, smooth_nodes, shuffled=False):
    residual = N._predict_cache(model, cache, columns, shuffle_trajectory=shuffled)
    outputs = {}
    cursor = 0
    for item in cache:
        count = len(item["node_rows"])
        local = residual[cursor:cursor + count].copy()
        cursor += count
        local -= local[0]
        if smooth_nodes > 0.0:
            local = gaussian_filter1d(local, sigma=smooth_nodes, mode="nearest")
            local -= local[0]
        node_surface = item["baseline_surface"] + local
        record = item["record"]
        evaluation = np.asarray(record["evaluation"], np.int64)
        surface = np.interp(
            record["md"][evaluation], record["md"][item["node_rows"]], node_surface
        )
        outputs[str(record["well"])] = surface - record["z"][evaluation]
    if cursor != len(residual):
        raise RuntimeError(f"Prediction cursor mismatch {cursor}/{len(residual)}")
    return outputs


def _align(cache, incumbent, predictions, column):
    parts = []
    for item in cache:
        record = item["record"]
        well = str(record["well"])
        local = incumbent.loc[incumbent["well"].eq(well)].copy()
        evaluation = np.asarray(record["evaluation"], np.int64)
        prediction = np.asarray(predictions[well], np.float64)
        if len(local) != len(evaluation) or len(prediction) != len(evaluation):
            raise RuntimeError(f"{well}: incumbent/raw rows {len(local)}/{len(evaluation)}")
        truth = np.asarray(record["truth"][evaluation], np.float64)
        gap = np.abs(local["tvt_true"].to_numpy(np.float64) - truth)
        maximum_gap = float(gap.max()) if len(gap) else 0.0
        if not np.isfinite(gap).all() or maximum_gap > 0.002:
            raise RuntimeError(f"{well}: maximum truth alignment gap={maximum_gap}")
        local[column] = prediction
        parts.append(local)
    return pd.concat(parts, ignore_index=True)


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    well_rmse = (
        pd.DataFrame({"well": frame["well"], "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    result = {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
    }
    for left, right, label in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_LABELS):
        mask = (distance > left) & (distance <= right)
        result[f"{label}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _correlations(frame):
    truth = frame["tvt_true"].to_numpy(np.float64)
    incumbent_error = frame[INCUMBENT_COLUMN].to_numpy(np.float64) - truth
    endpoint_error = frame["z_trajectory"].to_numpy(np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    incumbent_demeaned = incumbent_error.copy()
    endpoint_demeaned = endpoint_error.copy()
    within = {}
    for left, right, label in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_LABELS):
        mask = (distance > left) & (distance <= right)
        if mask.sum() < 3:
            within[label] = np.nan
            continue
        within[label] = float(np.corrcoef(incumbent_error[mask], endpoint_error[mask])[0, 1])
        incumbent_demeaned[mask] -= incumbent_error[mask].mean()
        endpoint_demeaned[mask] -= endpoint_error[mask].mean()
    return {
        "signed_error_raw": float(np.corrcoef(incumbent_error, endpoint_error)[0, 1]),
        "signed_error_distance_strata_demeaned": float(
            np.corrcoef(incumbent_demeaned, endpoint_demeaned)[0, 1]
        ),
        "within_distance_strata": within,
    }


def _blend(frame, weight):
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(np.float64)
    endpoint = frame["z_trajectory"].to_numpy(np.float64)
    return incumbent + float(weight) * (endpoint - incumbent)


def _shifted_control(frame, weight):
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(np.float64)
    move = frame["z_trajectory"].to_numpy(np.float64) - incumbent
    shifted = np.empty_like(move)
    for _, indices in frame.groupby("well", sort=False).indices.items():
        local = np.asarray(indices, np.int64)
        shifted[local] = np.roll(move[local], max(len(local) // 2, 1))
    return incumbent + float(weight) * shifted


N = _bootstrap(SOURCE)
N.OUT_DIR = OUT_DIR
started = time.time()
folds = pd.read_csv(N.V30_DIR / "folds.csv", dtype={"well": str})
train_wells = sorted(folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"].astype(str))
fold3_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].astype(str))

train_records = N._load_records(train_wells, "v55_fold45")
fold3_records = N._load_records(fold3_wells, "v55_fold3")
fold3_records.sort(
    key=lambda record: hashlib.sha256(f"v54:{record['well']}".encode("ascii")).digest()
)
discovery_records = fold3_records[:DISCOVERY_WELLS]
confirmation_records = fold3_records[DISCOVERY_WELLS:]
train_cache = N._build_cache(train_records, "v55_fold45")
discovery_cache = N._build_cache(discovery_records, "v55_fold3_discovery")
confirmation_cache = N._build_cache(confirmation_records, "v55_fold3_confirmation")
base_columns = train_cache[0]["base_columns"]
z_columns = base_columns + train_cache[0]["z_columns"]

print(
    "V55 CONFIG",
    {
        "train_wells": len(train_cache),
        "discovery_wells": len(discovery_cache),
        "confirmation_wells": len(confirmation_cache),
        "distance_features": len(base_columns),
        "z_features": len(z_columns),
        "weights": WEIGHTS,
        "fold1_touched": False,
    },
    flush=True,
)

model_bank = {}
scan_rows = []
for config in N.MODEL_CONFIGS:
    model = N._fit_model(train_cache, discovery_cache, z_columns, config)
    model_bank[config["id"]] = model
    residual = N._predict_cache(model, discovery_cache, z_columns)
    for smooth in N.SMOOTH_NODES:
        report = N._summary(
            N._evaluate(discovery_cache, residual, f"{config['id']}_smooth{smooth:g}", smooth)
        )
        scan_rows.append({"config": config["id"], "smooth_nodes": smooth, **report})
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "z_model_discovery_scan.csv", index=False)
print("\nV55 Z MODEL DISCOVERY", flush=True)
print(scan.to_string(index=False), flush=True)
selected_id = str(scan.iloc[0]["config"])
selected_smooth = float(scan.iloc[0]["smooth_nodes"])
selected_config = next(config for config in N.MODEL_CONFIGS if config["id"] == selected_id)
z_model = model_bank[selected_id]
distance_model = N._fit_model(train_cache, discovery_cache, base_columns, selected_config)

incumbent3 = _read_incumbent(SELECTION_FOLD)
discovery = _align(
    discovery_cache,
    incumbent3,
    _node_predictions(z_model, discovery_cache, z_columns, selected_smooth),
    "z_trajectory",
)
confirmation = _align(
    confirmation_cache,
    incumbent3,
    _node_predictions(z_model, confirmation_cache, z_columns, selected_smooth),
    "z_trajectory",
)

mechanism_rows = []
for label, cache in (("discovery", discovery_cache), ("confirmation", confirmation_cache)):
    z_report = N._summary(
        N._prediction_reports(z_model, cache, z_columns, "z_trajectory", selected_smooth)
    )
    distance_report = N._summary(
        N._prediction_reports(distance_model, cache, base_columns, "distance_only", selected_smooth)
    )
    shuffle_report = N._summary(
        N._prediction_reports(
            z_model, cache, z_columns, "shuffled_z", selected_smooth, shuffled=True
        )
    )
    mechanism_rows.extend(
        [{"stage": label, **distance_report}, {"stage": label, **z_report}, {"stage": label, **shuffle_report}]
    )
mechanism = pd.DataFrame(mechanism_rows)
mechanism.to_csv(OUT_DIR / "z_mechanism_summary.csv", index=False)
print("\nV55 Z MECHANISM", flush=True)
print(mechanism.to_string(index=False), flush=True)

weight_rows = []
for weight in WEIGHTS:
    report = _metrics(discovery, _blend(discovery, weight))
    weight_rows.append({"weight_z": weight, **report})
weight_scan = pd.DataFrame(weight_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
weight_scan.to_csv(OUT_DIR / "blend_weight_discovery_scan.csv", index=False)
print("\nV55 BLEND DISCOVERY", flush=True)
print(weight_scan.to_string(index=False), flush=True)
selected_weight = float(weight_scan.iloc[0]["weight_z"])

incumbent_confirmation = _metrics(confirmation, confirmation[INCUMBENT_COLUMN])
endpoint_confirmation = _metrics(confirmation, confirmation["z_trajectory"])
blend_confirmation = _metrics(confirmation, _blend(confirmation, selected_weight))
shifted_confirmation = _metrics(confirmation, _shifted_control(confirmation, selected_weight))
correlation = _correlations(confirmation)

confirmation_mechanism = mechanism.loc[mechanism["stage"].eq("confirmation")].set_index("endpoint")
distance_confirmation = confirmation_mechanism.loc["distance_only"]
z_confirmation = confirmation_mechanism.loc["z_trajectory"]
shuffle_confirmation = confirmation_mechanism.loc["shuffled_z"]
mechanism_checks = {
    "z_gain_vs_distance_ge_1": bool(distance_confirmation["rmse"] - z_confirmation["rmse"] >= 1.0),
    "z_far_gain_vs_distance_ge_1": bool(
        distance_confirmation["far_gt2000_rmse"] - z_confirmation["far_gt2000_rmse"] >= 1.0
    ),
    "shuffle_loses_ge_2": bool(shuffle_confirmation["rmse"] - z_confirmation["rmse"] >= 2.0),
}
confirmation_gain = incumbent_confirmation["rmse"] - blend_confirmation["rmse"]
confirmation_far_gain = (
    incumbent_confirmation["far_gt2000_rmse"] - blend_confirmation["far_gt2000_rmse"]
)
cross_checks = {
    "selected_weight_positive": bool(selected_weight > 0.0),
    "confirmation_gain_ge_0p03": bool(confirmation_gain >= 0.03),
    "confirmation_far_gain_nonnegative": bool(confirmation_far_gain >= 0.0),
    "aligned_beats_shifted_ge_0p02": bool(
        shifted_confirmation["rmse"] - blend_confirmation["rmse"] >= 0.02
    ),
}
confirmation_pass = bool(all(mechanism_checks.values()) and all(cross_checks.values()))

confirmation.to_parquet(OUT_DIR / "fold3_confirmation_endpoints.parquet", index=False)
fold2_payload = None
application_checks = None
if confirmation_pass:
    fold2_records = N._load_records(fold2_wells, "v55_fold2")
    fold2_cache = N._build_cache(fold2_records, "v55_fold2")
    incumbent2 = _read_incumbent(APPLICATION_FOLD)
    fold2 = _align(
        fold2_cache,
        incumbent2,
        _node_predictions(z_model, fold2_cache, z_columns, selected_smooth),
        "z_trajectory",
    )
    incumbent_fold2 = _metrics(fold2, fold2[INCUMBENT_COLUMN])
    endpoint_fold2 = _metrics(fold2, fold2["z_trajectory"])
    blend_fold2 = _metrics(fold2, _blend(fold2, selected_weight))
    shifted_fold2 = _metrics(fold2, _shifted_control(fold2, selected_weight))
    fold2_gain = incumbent_fold2["rmse"] - blend_fold2["rmse"]
    fold2_far_gain = incumbent_fold2["far_gt2000_rmse"] - blend_fold2["far_gt2000_rmse"]
    application_checks = {
        "fold2_gain_ge_0p02": bool(fold2_gain >= 0.02),
        "fold2_far_gain_nonnegative": bool(fold2_far_gain >= 0.0),
        "fold2_aligned_beats_shifted": bool(blend_fold2["rmse"] < shifted_fold2["rmse"]),
    }
    fold2_payload = {
        "incumbent": incumbent_fold2,
        "z_trajectory": endpoint_fold2,
        "blend": blend_fold2,
        "shifted_control": shifted_fold2,
        "gain": fold2_gain,
        "far_gain": fold2_far_gain,
        "correlation": _correlations(fold2),
    }
    fold2["selected_blend"] = _blend(fold2, selected_weight)
    fold2.to_parquet(OUT_DIR / "fold2_application_endpoints.parquet", index=False)

strict_pass = bool(confirmation_pass and application_checks and all(application_checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_Z_training_Fold3_nested_cross_evidence_Fold2_single_application",
    "sample_unit": "one complete physical well",
    "raw_inputs_only_for_z_endpoint": True,
    "incumbent": str(INCUMBENT_PATH),
    "protocol": {
        "train_folds": PRIOR_FOLDS,
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_cache),
        "confirmation_wells": len(confirmation_cache),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "selected_z_model": {**selected_config, "smooth_nodes": selected_smooth},
    "selected_weight_z": selected_weight,
    "confirmation": {
        "incumbent": incumbent_confirmation,
        "z_trajectory": endpoint_confirmation,
        "blend": blend_confirmation,
        "shifted_control": shifted_confirmation,
        "gain": confirmation_gain,
        "far_gain": confirmation_far_gain,
        "correlation": correlation,
    },
    "mechanism_checks": mechanism_checks,
    "cross_evidence_checks": cross_checks,
    "confirmation_pass": confirmation_pass,
    "fold2": fold2_payload,
    "application_checks": application_checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain Z steering as a small independent endpoint for deployment reconstruction"
        if strict_pass
        else "FAIL -> Z trajectory is physically real but does not add stable value to the 6.377 incumbent"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV55 CONFIRMATION", flush=True)
print(
    pd.DataFrame(
        [
            {"endpoint": "incumbent_6p377", **incumbent_confirmation},
            {"endpoint": "z_trajectory", **endpoint_confirmation},
            {"endpoint": "selected_blend", **blend_confirmation},
            {"endpoint": "shifted_control", **shifted_confirmation},
        ]
    ).to_string(index=False),
    flush=True,
)
print("\nFINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
