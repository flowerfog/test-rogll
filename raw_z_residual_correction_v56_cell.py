"""Fold-clean Z-steering residual correction for the 6.377 incumbent.

V55 showed that drilled Z carries causal, transferable formation-trend signal,
but its absolute TVT endpoint is too noisy to blend. V56 changes where that
signal is consumed: a compact model predicts only a smooth correction to the
6.377 incumbent. Fold 4/5 train, 24 deterministic Fold-3 wells select model
capacity/dose/cap/smoothing, and the remaining Fold-3 wells confirm. Fold 2 is
touched once only after confirmation passes. Fold 1 remains untouched.
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


EXPERIMENT = "raw_z_residual_correction_v56"
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
PREFERRED_TRAIN_FOLDS = (4, 5)
FALLBACK_TRAIN_FOLDS = (1, 2)
SELECTION_FOLD = 3
PREFERRED_APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
DOSES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
CAPS_FT = (2.0, 4.0, 8.0, 12.0)
SMOOTH_NODES = (0.0, 1.0, 2.0, 4.0)
DISTANCE_EDGES = (-np.inf, 500.0, 1000.0, 2000.0, np.inf)
DISTANCE_LABELS = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
SEED = 56027


def _bootstrap(path: Path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType("raw_active_steering_v54_definitions_for_v56")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
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


def _read_incumbent(folds):
    if not INCUMBENT_PATH.is_file():
        raise FileNotFoundError(INCUMBENT_PATH)
    columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", INCUMBENT_COLUMN]
    fold_values = [int(fold) for fold in folds]
    frame = pd.read_parquet(INCUMBENT_PATH, columns=columns)
    frame = frame.loc[frame["fold"].isin(fold_values)].copy()
    frame["well"] = frame["well"].astype(str)
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if frame.empty or frame["id"].duplicated().any():
        raise RuntimeError(f"Invalid incumbent folds {fold_values}: rows={len(frame)}")
    if frame[["tvt_true", "md_since", INCUMBENT_COLUMN]].isna().any().any():
        raise RuntimeError(f"Missing incumbent values in folds {fold_values}")
    return frame


def _augment_cache(cache, incumbent, label):
    output = []
    for number, item in enumerate(cache, 1):
        record = item["record"]
        well = str(record["well"])
        local = incumbent.loc[incumbent["well"].eq(well)]
        evaluation = np.asarray(record["evaluation"], np.int64)
        if len(local) != len(evaluation):
            raise RuntimeError(f"{well}: incumbent/raw rows {len(local)}/{len(evaluation)}")
        truth = np.asarray(record["truth"][evaluation], np.float64)
        truth_gap = np.abs(local["tvt_true"].to_numpy(np.float64) - truth)
        if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.002:
            raise RuntimeError(f"{well}: maximum truth gap={float(truth_gap.max())}")

        eval_md = record["md"][evaluation]
        node_rows = item["node_rows"]
        node_md = record["md"][node_rows]
        incumbent_eval = local[INCUMBENT_COLUMN].to_numpy(np.float64)
        correction_eval = truth - incumbent_eval
        node_correction = np.interp(node_md, eval_md, correction_eval)
        node_correction[0] = 0.0

        incumbent_node = np.interp(node_md, eval_md, incumbent_eval)
        incumbent_surface = incumbent_node + record["z"][node_rows]
        incumbent_surface[0] = record["anchor_surface"]
        incumbent_residual = incumbent_surface - item["baseline_surface"]
        incumbent_residual[0] = 0.0
        spacing = max(float(np.median(np.diff(node_md))), 1.0)

        features = item["features"].copy()
        parent_columns = []

        def add(name, values):
            features[name] = np.asarray(values, np.float64)
            parent_columns.append(name)

        add("incumbent_surface_residual", incumbent_residual / 32.0)
        add("incumbent_local_rate", np.gradient(incumbent_surface, node_md) * 100.0)
        for sigma in (1.0, 2.0, 4.0):
            tag = str(int(sigma))
            smooth = gaussian_filter1d(incumbent_residual, sigma=sigma, mode="nearest")
            smooth -= smooth[0]
            add(f"incumbent_residual_s{tag}", smooth / 32.0)
            add(f"incumbent_rate_s{tag}", np.gradient(smooth, node_md) * 100.0)
            add(
                f"incumbent_curvature_s{tag}",
                np.gradient(np.gradient(smooth, node_md), node_md) * spacing * 100.0,
            )

        augmented = dict(item)
        augmented["features"] = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        augmented["target"] = node_correction
        augmented["incumbent_eval"] = incumbent_eval
        augmented["truth_eval"] = truth
        augmented["row_frame"] = local.copy()
        augmented["parent_columns"] = parent_columns
        output.append(augmented)
        if number % 50 == 0 or number == len(cache):
            print(f"V56 augment {label} {number}/{len(cache)}", flush=True)
    return output


def _predict_raw(model, cache, columns, shuffle_z=False):
    x = N._matrix(cache, columns, train=False)
    if shuffle_z:
        z_names = set(cache[0]["z_columns"])
        indices = [index for index, name in enumerate(columns) if name in z_names]
        if not indices:
            raise RuntimeError("No Z columns available for shuffled control")
        distance_index = columns.index("distance_kft")
        bins = np.digitize(x[:, distance_index], (0.5, 1.0, 2.0, 3.0, 4.0))
        rng = np.random.default_rng(SEED + 991)
        for group in np.unique(bins):
            rows = np.flatnonzero(bins == group)
            permutation = rng.permutation(rows)
            x[np.ix_(rows, indices)] = x[np.ix_(permutation, indices)]
    return model.predict(x).astype(np.float64)


def _row_output(cache, raw_correction, smooth_nodes, dose, cap_ft, column):
    parts = []
    cursor = 0
    for item in cache:
        count = len(item["node_rows"])
        correction = raw_correction[cursor:cursor + count].copy()
        cursor += count
        correction -= correction[0]
        if smooth_nodes > 0.0:
            correction = gaussian_filter1d(correction, sigma=smooth_nodes, mode="nearest")
            correction -= correction[0]
        correction = np.clip(correction, -float(cap_ft), float(cap_ft))
        record = item["record"]
        evaluation = np.asarray(record["evaluation"], np.int64)
        row_correction = np.interp(
            record["md"][evaluation], record["md"][item["node_rows"]], correction
        )
        local = item["row_frame"].copy()
        local[column] = item["incumbent_eval"] + float(dose) * row_correction
        local["raw_correction"] = row_correction
        parts.append(local)
    if cursor != len(raw_correction):
        raise RuntimeError(f"Prediction cursor mismatch {cursor}/{len(raw_correction)}")
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


def _shifted_prediction(frame, dose):
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(np.float64)
    move = frame["raw_correction"].to_numpy(np.float64) * float(dose)
    shifted = np.empty_like(move)
    for _, indices in frame.groupby("well", sort=False).indices.items():
        local = np.asarray(indices, np.int64)
        shifted[local] = np.roll(move[local], max(len(local) // 2, 1))
    return incumbent + shifted


N = _bootstrap(SOURCE)
N.OUT_DIR = OUT_DIR
started = time.time()
folds = pd.read_csv(N.V30_DIR / "folds.csv", dtype={"well": str})
incumbent_coverage = pd.read_parquet(INCUMBENT_PATH, columns=["fold", "well"])
incumbent_coverage["well"] = incumbent_coverage["well"].astype(str)
incumbent_fold_counts = {
    int(fold): int(group["well"].nunique())
    for fold, group in incumbent_coverage.groupby("fold", sort=True)
}
available_incumbent_folds = set(incumbent_fold_counts)
if set(PREFERRED_TRAIN_FOLDS).issubset(available_incumbent_folds):
    train_folds = PREFERRED_TRAIN_FOLDS
    application_fold = PREFERRED_APPLICATION_FOLD
    fallback_protocol = False
elif set(FALLBACK_TRAIN_FOLDS).issubset(available_incumbent_folds):
    # The published 6.377 replay currently covers only Fold 1/2/3. Fold 1/2
    # remain honest training data for a held-out Fold-3 confirmation, but this
    # fallback has no second untouched application fold and cannot be promoted.
    train_folds = FALLBACK_TRAIN_FOLDS
    application_fold = None
    fallback_protocol = True
else:
    raise RuntimeError(
        f"Insufficient incumbent fold coverage: {incumbent_fold_counts}"
    )
if SELECTION_FOLD not in available_incumbent_folds:
    raise RuntimeError(
        f"Selection Fold-{SELECTION_FOLD} absent from incumbent: {incumbent_fold_counts}"
    )
print(
    "V56 INCUMBENT COVERAGE",
    {
        "fold_wells": incumbent_fold_counts,
        "train_folds": train_folds,
        "selection_fold": SELECTION_FOLD,
        "application_fold": application_fold,
        "fallback_protocol": fallback_protocol,
    },
    flush=True,
)
train_wells = sorted(folds.loc[folds["fold"].isin(train_folds), "well"].astype(str))
fold3_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].astype(str))
application_wells = (
    sorted(folds.loc[folds["fold"].eq(application_fold), "well"].astype(str))
    if application_fold is not None
    else []
)

train_records = N._load_records(train_wells, "v56_fold45")
fold3_records = N._load_records(fold3_wells, "v56_fold3")
fold3_records.sort(
    key=lambda record: hashlib.sha256(f"v54:{record['well']}".encode("ascii")).digest()
)
discovery_records = fold3_records[:DISCOVERY_WELLS]
confirmation_records = fold3_records[DISCOVERY_WELLS:]

incumbent45 = _read_incumbent(train_folds)
incumbent3 = _read_incumbent((SELECTION_FOLD,))
train_cache = _augment_cache(
    N._build_cache(train_records, "v56_fold45"), incumbent45, "fold45"
)
fold3_cache = N._build_cache(fold3_records, "v56_fold3")
discovery_cache = _augment_cache(fold3_cache[:DISCOVERY_WELLS], incumbent3, "fold3_discovery")
confirmation_cache = _augment_cache(fold3_cache[DISCOVERY_WELLS:], incumbent3, "fold3_confirmation")

base_columns = train_cache[0]["base_columns"]
parent_columns = train_cache[0]["parent_columns"]
no_z_columns = base_columns + parent_columns
full_columns = no_z_columns + train_cache[0]["z_columns"]
print(
    "V56 CONFIG",
    {
        "train_wells": len(train_cache),
        "discovery_wells": len(discovery_cache),
        "confirmation_wells": len(confirmation_cache),
        "no_z_features": len(no_z_columns),
        "full_features": len(full_columns),
        "fold1_touched": bool(1 in train_folds),
        "fallback_protocol": fallback_protocol,
    },
    flush=True,
)

model_bank = {}
scan_rows = []
for config in N.MODEL_CONFIGS:
    model = N._fit_model(train_cache, discovery_cache, full_columns, config)
    model_bank[config["id"]] = model
    raw = _predict_raw(model, discovery_cache, full_columns)
    for smooth in SMOOTH_NODES:
        for cap in CAPS_FT:
            for dose in DOSES:
                frame = _row_output(
                    discovery_cache, raw, smooth, dose, cap, "candidate"
                )
                report = _metrics(frame, frame["candidate"])
                scan_rows.append(
                    {
                        "config": config["id"],
                        "smooth_nodes": smooth,
                        "cap_ft": cap,
                        "dose": dose,
                        **report,
                    }
                )
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
print("\nV56 DISCOVERY TOP 30", flush=True)
print(scan.head(30).to_string(index=False), flush=True)

selected = scan.iloc[0]
selected_id = str(selected["config"])
selected_smooth = float(selected["smooth_nodes"])
selected_cap = float(selected["cap_ft"])
selected_dose = float(selected["dose"])
selected_config = next(config for config in N.MODEL_CONFIGS if config["id"] == selected_id)
full_model = model_bank[selected_id]
no_z_model = N._fit_model(train_cache, discovery_cache, no_z_columns, selected_config)

full_raw = _predict_raw(full_model, confirmation_cache, full_columns)
no_z_raw = _predict_raw(no_z_model, confirmation_cache, no_z_columns)
shuffle_raw = _predict_raw(full_model, confirmation_cache, full_columns, shuffle_z=True)
full_confirmation = _row_output(
    confirmation_cache, full_raw, selected_smooth, selected_dose, selected_cap, "candidate"
)
no_z_confirmation = _row_output(
    confirmation_cache, no_z_raw, selected_smooth, selected_dose, selected_cap, "candidate"
)
shuffle_confirmation = _row_output(
    confirmation_cache, shuffle_raw, selected_smooth, selected_dose, selected_cap, "candidate"
)

incumbent_metrics = _metrics(full_confirmation, full_confirmation[INCUMBENT_COLUMN])
full_metrics = _metrics(full_confirmation, full_confirmation["candidate"])
no_z_metrics = _metrics(no_z_confirmation, no_z_confirmation["candidate"])
shuffle_metrics = _metrics(shuffle_confirmation, shuffle_confirmation["candidate"])
shifted_metrics = _metrics(
    full_confirmation, _shifted_prediction(full_confirmation, selected_dose)
)
confirmation_gain = incumbent_metrics["rmse"] - full_metrics["rmse"]
confirmation_far_gain = incumbent_metrics["far_gt2000_rmse"] - full_metrics["far_gt2000_rmse"]
checks = {
    "selected_dose_positive": bool(selected_dose > 0.0),
    "confirmation_gain_ge_0p03": bool(confirmation_gain >= 0.03),
    "confirmation_far_gain_nonnegative": bool(confirmation_far_gain >= 0.0),
    "z_increment_over_no_z_ge_0p02": bool(no_z_metrics["rmse"] - full_metrics["rmse"] >= 0.02),
    "shuffled_z_loses_ge_0p02": bool(shuffle_metrics["rmse"] - full_metrics["rmse"] >= 0.02),
    "shifted_move_loses_ge_0p02": bool(shifted_metrics["rmse"] - full_metrics["rmse"] >= 0.02),
}
confirmation_pass = bool(all(checks.values()))
full_confirmation.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False)

fold2_payload = None
application_checks = None
if confirmation_pass and application_fold is not None:
    incumbent2 = _read_incumbent((application_fold,))
    fold2_cache = _augment_cache(
        N._build_cache(
            N._load_records(application_wells, "v56_application"), "v56_application"
        ),
        incumbent2,
        "application",
    )
    fold2_raw = _predict_raw(full_model, fold2_cache, full_columns)
    fold2 = _row_output(
        fold2_cache, fold2_raw, selected_smooth, selected_dose, selected_cap, "candidate"
    )
    fold2_incumbent = _metrics(fold2, fold2[INCUMBENT_COLUMN])
    fold2_candidate = _metrics(fold2, fold2["candidate"])
    fold2_shifted = _metrics(fold2, _shifted_prediction(fold2, selected_dose))
    fold2_gain = fold2_incumbent["rmse"] - fold2_candidate["rmse"]
    fold2_far_gain = fold2_incumbent["far_gt2000_rmse"] - fold2_candidate["far_gt2000_rmse"]
    application_checks = {
        "fold2_gain_ge_0p02": bool(fold2_gain >= 0.02),
        "fold2_far_gain_nonnegative": bool(fold2_far_gain >= 0.0),
        "fold2_aligned_beats_shifted": bool(fold2_candidate["rmse"] < fold2_shifted["rmse"]),
    }
    fold2_payload = {
        "incumbent": fold2_incumbent,
        "candidate": fold2_candidate,
        "shifted_control": fold2_shifted,
        "gain": fold2_gain,
        "far_gain": fold2_far_gain,
    }
    fold2.to_parquet(OUT_DIR / "fold2_application_oof.parquet", index=False)

strict_pass = bool(
    not fallback_protocol
    and confirmation_pass
    and application_checks
    and all(application_checks.values())
)
summary = pd.DataFrame(
    [
        {"endpoint": "incumbent_6p377", **incumbent_metrics},
        {"endpoint": "z_residual", **full_metrics},
        {"endpoint": "no_z_residual", **no_z_metrics},
        {"endpoint": "shuffled_z", **shuffle_metrics},
        {"endpoint": "shifted_move", **shifted_metrics},
    ]
)
summary.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
print("\nV56 CONFIRMATION", flush=True)
print(summary.to_string(index=False), flush=True)

verdict = {
    "experiment": EXPERIMENT,
    "stage": (
        "Fold12_training_Fold3_nested_confirmation_coverage_limited"
        if fallback_protocol
        else "Fold45_training_Fold3_nested_confirmation_Fold2_application"
    ),
    "sample_unit": "one complete physical well",
    "incumbent": str(INCUMBENT_PATH),
    "protocol": {
        "train_folds": train_folds,
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_cache),
        "confirmation_wells": len(confirmation_cache),
        "application_fold": application_fold,
        "fold1_touched": bool(1 in train_folds),
        "fallback_protocol": fallback_protocol,
        "incumbent_fold_counts": incumbent_fold_counts,
    },
    "selected_recipe": {
        **selected_config,
        "smooth_nodes": selected_smooth,
        "cap_ft": selected_cap,
        "dose": selected_dose,
    },
    "confirmation": {
        "incumbent": incumbent_metrics,
        "candidate": full_metrics,
        "no_z": no_z_metrics,
        "shuffled_z": shuffle_metrics,
        "shifted_move": shifted_metrics,
        "gain": confirmation_gain,
        "far_gain": confirmation_far_gain,
    },
    "checks": checks,
    "confirmation_pass": confirmation_pass,
    "fold2": fold2_payload,
    "application_checks": application_checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> reconstruct this fixed residual correction in the deployment notebook"
        if strict_pass
        else (
            "CONFIRMATION PASS -> reconstruct 6.377 Fold4/5 OOF before promotion"
            if confirmation_pass and fallback_protocol
            else "FAIL -> Z is observable but its residual correction does not transfer to the 6.377 incumbent"
        )
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
