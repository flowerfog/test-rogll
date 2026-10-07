"""Inverse-steering controller endpoint with canonical five-fold validation.

This experiment treats the drilled Z trajectory as the delayed response of a
geosteering controller.  A compact linear model predicts local formation-
surface slope from multi-scale, past/current/future Z slopes and curvatures;
the predicted slope is integrated from the exactly known anchor.  This is
deliberately different from a row-wise absolute-TVT regressor.

For every held fold, a second fold selects Ridge regularisation, controller
dose, rate smoothing, and a fixed blend weight against the complete V66
incumbent.  The remaining three folds fit the selector model, and all four
non-held folds refit the final model.  A within-well circular shift of the
trajectory channels is evaluated as a matched mechanism control.

Run on AutoDL after V30 raw-data definitions and the V66 complete OOF exist.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "inverse_steering_controller_gateway_v72"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V30_SOURCE = next(
    (
        path
        for path in (
            ROOT / "raw_stochastic_alignment_standalone_v30_cell.py",
            Path.cwd() / "raw_stochastic_alignment_standalone_v30_cell.py",
        )
        if path.is_file()
    ),
    None,
)
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
V66_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)

EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
FOLDS = (1, 2, 3, 4, 5)
SCALES_FT = (32.0, 64.0, 128.0, 256.0, 512.0)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
CONTROLLER_DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
SMOOTH_SIGMAS = (0.0, 0.75, 1.5, 3.0)
BLEND_WEIGHTS = tuple(np.round(np.arange(0.0, 0.3001, 0.025), 6))
RATE_CLIP = 0.08
SEED = 72027


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _bootstrap_v30(path: Path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "inverse_controller_v30_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module


if V30_SOURCE is None:
    raise FileNotFoundError("Upload raw_stochastic_alignment_standalone_v30_cell.py")
for required in (FOLD_PATH, V66_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
V = _bootstrap_v30(V30_SOURCE)


def _fold_map(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    well_col = next((c for c in ("well", "well_id", "Well") if c in frame), None)
    fold_col = next((c for c in ("fold", "outer_fold", "Fold") if c in frame), None)
    if well_col is None or fold_col is None:
        raise RuntimeError(f"Cannot identify fold columns: {frame.columns.tolist()}")
    frame = frame[[well_col, fold_col]].drop_duplicates()
    frame[well_col] = frame[well_col].astype(str)
    if frame[well_col].duplicated().any():
        raise RuntimeError("Duplicate wells in canonical fold metadata")
    result = dict(zip(frame[well_col], frame[fold_col].astype(int)))
    if len(result) != EXPECTED_WELLS or set(result.values()) != set(FOLDS):
        raise RuntimeError(f"Invalid canonical folds: wells={len(result)}")
    return result


def _load_records(wells: list[str]) -> tuple[dict[str, dict], list[dict]]:
    records, failures = {}, []
    for number, well in enumerate(wells, 1):
        try:
            record = V._load_record(well)
            records[well] = record
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(
                f"CONTROLLER load {number}/{len(wells)} success={len(records)} "
                f"failures={len(failures)}",
                flush=True,
            )
    pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
    if len(records) < EXPECTED_WELLS - 3:
        raise RuntimeError(f"Only {len(records)} usable records: {failures[:5]}")
    return records, failures


def _interp(axis, values, query):
    return np.interp(query, axis, values, left=values[0], right=values[-1])


def _segment_payload(record: dict) -> dict:
    md = np.asarray(record["md"], np.float64)
    z = np.asarray(record["z"], np.float64)
    truth = np.asarray(record["truth"], np.float64)
    nodes = np.asarray(V._node_rows(record), np.int64)
    if len(nodes) < 3:
        raise RuntimeError(f"{record['well']}: only {len(nodes)} controller nodes")

    node_md = md[nodes]
    delta_md = np.diff(node_md)
    mid_md = 0.5 * (node_md[:-1] + node_md[1:])
    distance = mid_md - float(record["anchor_md"])
    total = max(float(node_md[-1] - record["anchor_md"]), 1.0)
    prefix_rate = float(record["prefix_surface_rate"])
    surface = truth[nodes] + z[nodes]
    target_rate = np.diff(surface) / np.maximum(delta_md, 1e-6)

    spacing = max(float(np.median(np.diff(md))), 1e-3)
    z_rate = np.gradient(z, md)
    features = {
        "distance_kft": distance / 1000.0,
        "progress": distance / total,
        "log_distance": np.log1p(np.maximum(distance, 0.0)) / 8.0,
        "total_length_kft": np.full(len(distance), total / 1000.0),
        "prefix_surface_rate": np.full(len(distance), prefix_rate),
        "segment_z_rate": np.diff(z[nodes]) / np.maximum(delta_md, 1e-6),
    }
    sensor_columns = ["segment_z_rate"]
    for scale in SCALES_FT:
        sigma = max(scale / spacing, 0.25)
        smooth_rate = gaussian_filter1d(z_rate, sigma=sigma, mode="nearest")
        curvature = np.gradient(smooth_rate, md)
        tag = str(int(scale))
        for suffix, query in (
            ("past", mid_md - scale),
            ("now", mid_md),
            ("future", mid_md + scale),
        ):
            name = f"z_rate_{suffix}_{tag}"
            features[name] = _interp(md, smooth_rate, query)
            sensor_columns.append(name)
        name = f"z_curvature_now_{tag}"
        features[name] = 1000.0 * _interp(md, curvature, mid_md)
        sensor_columns.append(name)
        name = f"z_rate_response_{tag}"
        features[name] = _interp(md, smooth_rate, mid_md + scale) - _interp(
            md, smooth_rate, mid_md - scale
        )
        sensor_columns.append(name)

    frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "well": str(record["well"]),
        "record": record,
        "nodes": nodes,
        "node_md": node_md,
        "delta_md": delta_md,
        "features": frame,
        "sensor_columns": sensor_columns,
        "target_residual_rate": np.asarray(target_rate - prefix_rate, np.float64),
        "weights": np.maximum(delta_md, 1.0),
    }


def _matrix(payloads: list[dict]):
    x = np.concatenate([item["features"].to_numpy(np.float64) for item in payloads])
    y = np.concatenate([item["target_residual_rate"] for item in payloads])
    w = np.concatenate([item["weights"] for item in payloads])
    return x, y, w


def _fit(payloads: list[dict], alpha: float):
    x, y, weight = _matrix(payloads)
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(x, y, ridge__sample_weight=weight)
    return model


def _shift_sensor_features(item: dict, seed: int) -> np.ndarray:
    x = item["features"].to_numpy(np.float64).copy()
    columns = list(item["features"].columns)
    sensor_indices = [columns.index(name) for name in item["sensor_columns"]]
    count = len(x)
    if count > 3:
        rng = np.random.default_rng(seed)
        minimum = max(1, count // 4)
        shift = int(rng.integers(minimum, max(minimum + 1, count - minimum + 1)))
        x[:, sensor_indices] = np.roll(x[:, sensor_indices], shift, axis=0)
    return x


def _decode(
    model,
    payloads: list[dict],
    dose: float,
    smooth_sigma: float,
    shuffled: bool = False,
) -> pd.DataFrame:
    outputs = []
    for item in payloads:
        x = (
            _shift_sensor_features(item, SEED + int(item["well"], 16) % 100000)
            if shuffled
            else item["features"].to_numpy(np.float64)
        )
        residual_rate = np.asarray(model.predict(x), np.float64)
        residual_rate = np.clip(residual_rate, -RATE_CLIP, RATE_CLIP)
        if smooth_sigma > 0.0:
            residual_rate = gaussian_filter1d(
                residual_rate, sigma=float(smooth_sigma), mode="nearest"
            )
        record = item["record"]
        local_rate = float(record["prefix_surface_rate"]) + float(dose) * residual_rate
        node_surface = np.r_[
            float(record["anchor_surface"]),
            float(record["anchor_surface"]) + np.cumsum(local_rate * item["delta_md"]),
        ]
        rows = np.asarray(record["evaluation"], np.int64)
        eval_md = np.asarray(record["md"], np.float64)[rows]
        pred_surface = np.interp(eval_md, item["node_md"], node_surface)
        prediction = pred_surface - np.asarray(record["z"], np.float64)[rows]
        outputs.append(
            pd.DataFrame(
                {
                    "well": item["well"],
                    "row_idx": rows,
                    "controller": prediction.astype(np.float64),
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)


def _metric(frame: pd.DataFrame, prediction, endpoint: str) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    if len(prediction) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid endpoint {endpoint}")
    well_sq = pd.DataFrame({"well": frame["well"], "sq": error * error})
    well_rmse = np.sqrt(well_sq.groupby("well", sort=False)["sq"].mean().to_numpy())
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _load_incumbent(path: Path, fold_by_well: dict[str, int]) -> pd.DataFrame:
    header = pq.read_schema(path).names
    prediction_col = next(
        (
            c
            for c in (
                "incumbent_6p377_replay",
                "current_dose_0p20",
                "prediction",
            )
            if c in header
        ),
        None,
    )
    required = ["well", "row_idx", "tvt_true", "md_since"]
    if prediction_col is None or any(c not in header for c in required):
        raise RuntimeError(f"Cannot identify V66 columns: {header}")
    frame = pd.read_parquet(path, columns=required + [prediction_col])
    frame = frame.rename(columns={prediction_col: "incumbent"})
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = frame["well"].map(fold_by_well)
    if len(frame) != EXPECTED_ROWS or frame["fold"].isna().any():
        raise RuntimeError(f"Invalid V66 coverage: rows={len(frame)}")
    return frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)


def _align(base: pd.DataFrame, endpoint: pd.DataFrame, wells: list[str]) -> pd.DataFrame:
    left = base[base["well"].isin(wells)].copy()
    merged = left.merge(endpoint, on=["well", "row_idx"], how="left", validate="one_to_one")
    if merged["controller"].isna().any():
        missing = merged.loc[merged["controller"].isna(), ["well", "row_idx"]].head()
        raise RuntimeError(f"Controller alignment failed:\n{missing}")
    return merged


fold_by_well = _fold_map(FOLD_PATH)
wells = sorted(fold_by_well)
records, failures = _load_records(wells)
usable_wells = sorted(records)
payload_by_well = {}
payload_failures = []
for number, well in enumerate(usable_wells, 1):
    try:
        payload_by_well[well] = _segment_payload(records[well])
    except Exception as exc:
        payload_failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(usable_wells):
        print(
            f"CONTROLLER features {number}/{len(usable_wells)} "
            f"success={len(payload_by_well)} failures={len(payload_failures)}",
            flush=True,
        )
pd.DataFrame(payload_failures).to_csv(OUT_DIR / "payload_failures.csv", index=False)
if len(payload_by_well) < EXPECTED_WELLS - 3:
    raise RuntimeError(f"Only {len(payload_by_well)} controller payloads")

incumbent = _load_incumbent(V66_PATH, fold_by_well)
fold_reports = []
oof_frames = []
scan_frames = []

for held_fold in FOLDS:
    selection_fold = FOLDS[held_fold % len(FOLDS)]
    train_folds = [f for f in FOLDS if f not in (held_fold, selection_fold)]
    refit_folds = [f for f in FOLDS if f != held_fold]
    train_payloads = [
        payload_by_well[w]
        for w in payload_by_well
        if fold_by_well[w] in train_folds
    ]
    selection_wells = [
        w for w in payload_by_well if fold_by_well[w] == selection_fold
    ]
    selection_payloads = [payload_by_well[w] for w in selection_wells]

    scan = []
    selection_predictions = {}
    for alpha in RIDGE_ALPHAS:
        model = _fit(train_payloads, alpha)
        for dose in CONTROLLER_DOSES:
            for sigma in SMOOTH_SIGMAS:
                key = (alpha, dose, sigma)
                endpoint = _decode(model, selection_payloads, dose, sigma)
                aligned = _align(incumbent, endpoint, selection_wells)
                candidate = aligned["controller"].to_numpy(np.float64)
                base_prediction = aligned["incumbent"].to_numpy(np.float64)
                truth = aligned["tvt_true"].to_numpy(np.float64)
                selection_predictions[key] = (aligned, candidate)
                candidate_rmse = float(np.sqrt(np.mean((candidate - truth) ** 2)))
                for weight in BLEND_WEIGHTS:
                    blend = (1.0 - weight) * base_prediction + weight * candidate
                    rmse = float(np.sqrt(np.mean((blend - truth) ** 2)))
                    scan.append(
                        {
                            "held_fold": held_fold,
                            "selection_fold": selection_fold,
                            "alpha": alpha,
                            "dose": dose,
                            "smooth_sigma": sigma,
                            "blend_weight": weight,
                            "candidate_rmse": candidate_rmse,
                            "base_rmse": float(
                                np.sqrt(np.mean((base_prediction - truth) ** 2))
                            ),
                            "rmse": rmse,
                        }
                    )
    scan_frame = pd.DataFrame(scan).sort_values(
        ["rmse", "blend_weight", "candidate_rmse"], kind="stable"
    )
    selected = scan_frame.iloc[0].to_dict()
    scan_frames.append(scan_frame.head(50))

    refit_payloads = [
        payload_by_well[w]
        for w in payload_by_well
        if fold_by_well[w] in refit_folds
    ]
    held_wells = [w for w in payload_by_well if fold_by_well[w] == held_fold]
    held_payloads = [payload_by_well[w] for w in held_wells]
    final_model = _fit(refit_payloads, float(selected["alpha"]))
    endpoint = _decode(
        final_model,
        held_payloads,
        float(selected["dose"]),
        float(selected["smooth_sigma"]),
    )
    shuffled_endpoint = _decode(
        final_model,
        held_payloads,
        float(selected["dose"]),
        float(selected["smooth_sigma"]),
        shuffled=True,
    ).rename(columns={"controller": "controller_shuffled"})
    aligned = _align(incumbent, endpoint, held_wells).merge(
        shuffled_endpoint,
        on=["well", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    weight = float(selected["blend_weight"])
    aligned["blend"] = (
        (1.0 - weight) * aligned["incumbent"] + weight * aligned["controller"]
    )
    aligned["shuffle_blend"] = (
        (1.0 - weight) * aligned["incumbent"]
        + weight * aligned["controller_shuffled"]
    )
    base_metric = _metric(aligned, aligned["incumbent"], "incumbent")
    controller_metric = _metric(aligned, aligned["controller"], "controller")
    blend_metric = _metric(aligned, aligned["blend"], "blend")
    shuffle_metric = _metric(aligned, aligned["shuffle_blend"], "shuffle_blend")
    fold_reports.append(
        {
            "fold": held_fold,
            "selection_fold": selection_fold,
            "selected_alpha": float(selected["alpha"]),
            "selected_dose": float(selected["dose"]),
            "selected_smooth_sigma": float(selected["smooth_sigma"]),
            "selected_blend_weight": weight,
            "selection_gain": float(selected["base_rmse"] - selected["rmse"]),
            "base_rmse": base_metric["rmse"],
            "controller_rmse": controller_metric["rmse"],
            "rmse": blend_metric["rmse"],
            "gain": base_metric["rmse"] - blend_metric["rmse"],
            "far_gain": base_metric["far_gt2000_rmse"] - blend_metric["far_gt2000_rmse"],
            "shuffle_loss": shuffle_metric["rmse"] - blend_metric["rmse"],
        }
    )
    oof_frames.append(
        aligned[
            [
                "well", "row_idx", "fold", "tvt_true", "md_since", "incumbent",
                "controller", "controller_shuffled", "blend", "shuffle_blend",
            ]
        ]
    )
    print("CONTROLLER FOLD", fold_reports[-1], flush=True)

oof = pd.concat(oof_frames, ignore_index=True).sort_values(
    ["well", "row_idx"], kind="stable"
).reset_index(drop=True)
fold_frame = pd.DataFrame(fold_reports)
scan_frame = pd.concat(scan_frames, ignore_index=True)

base_metric = _metric(oof, oof["incumbent"], "incumbent")
controller_metric = _metric(oof, oof["controller"], "inverse_controller")
blend_metric = _metric(oof, oof["blend"], "nested_blend")
shuffle_metric = _metric(oof, oof["shuffle_blend"], "trajectory_shift_control")
truth = oof["tvt_true"].to_numpy(np.float64)
base_error = oof["incumbent"].to_numpy(np.float64) - truth
controller_error = oof["controller"].to_numpy(np.float64) - truth
raw_rho = float(np.corrcoef(base_error, controller_error)[0, 1])

distance = oof["md_since"].to_numpy(np.float64)
distance_bins = np.digitize(distance, (500.0, 1000.0, 2000.0))
detrended = []
for error in (base_error, controller_error):
    residual = error.copy()
    for group in np.unique(distance_bins):
        mask = distance_bins == group
        residual[mask] -= np.mean(residual[mask])
    detrended.append(residual)
detrended_rho = float(np.corrcoef(detrended[0], detrended[1])[0, 1])

checks = {
    "controller_rmse_le_15": controller_metric["rmse"] <= 15.0,
    "detrended_error_rho_le_0p60": detrended_rho <= 0.60,
    "nested_gain_ge_0p15": base_metric["rmse"] - blend_metric["rmse"] >= 0.15,
    "positive_folds_ge_4": int((fold_frame["gain"] >= 0.0).sum()) >= 4,
    "minimum_fold_gain_ge_minus0p03": float(fold_frame["gain"].min()) >= -0.03,
    "trajectory_shift_loses_ge_0p05": shuffle_metric["rmse"] - blend_metric["rmse"] >= 0.05,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "canonical_fivefold_inverse_geosteering_controller_gateway",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "uses_gr": False,
    "uses_historical_predictions_in_controller": False,
    "controller": {
        "target": "local formation-surface slope residual from visible-prefix slope",
        "decoder": "integrate predicted local slope from exact anchor",
        "scales_ft": list(SCALES_FT),
        "features": int(len(next(iter(payload_by_well.values()))["features"].columns)),
        "rate_clip": RATE_CLIP,
    },
    "protocol": {
        "outer_folds": list(FOLDS),
        "selection": "next fold; other three fit selector model; four-fold refit",
        "incumbent": str(V66_PATH),
        "fold_source": str(FOLD_PATH),
        "fold1_touched": True,
    },
    "coverage": {
        "rows": int(len(oof)),
        "wells": int(oof["well"].nunique()),
        "record_failures": len(failures),
        "payload_failures": len(payload_failures),
    },
    "base": base_metric,
    "controller_endpoint": controller_metric,
    "nested_blend": blend_metric,
    "trajectory_shift_control": shuffle_metric,
    "gain": base_metric["rmse"] - blend_metric["rmse"],
    "far_gain": base_metric["far_gt2000_rmse"] - blend_metric["far_gt2000_rmse"],
    "shuffle_loss": shuffle_metric["rmse"] - blend_metric["rmse"],
    "error_correlation": {
        "raw": raw_rho,
        "distance_strata_demeaned": detrended_rho,
    },
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> refit one full-data controller and package a frozen blend"
        if strict_pass
        else "FAIL -> close inverse-steering geometry and retain the existing endpoint bank"
    ),
}

oof.to_parquet(OUT_DIR / "inverse_controller_oof.parquet", index=False)
fold_frame.to_csv(OUT_DIR / "folds.csv", index=False)
scan_frame.to_csv(OUT_DIR / "nested_scan_top50.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nINVERSE CONTROLLER FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nINVERSE CONTROLLER FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
