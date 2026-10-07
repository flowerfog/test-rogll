"""Apply the V76 adaptive coarse dose immediately after the V69 Kaggle cell.

Required live state comes from rawdata_lattice_coarse_all773_kaggle_v69_cell.py.
The existing fixed-0.30 submission is preserved before submission.csv is
replaced by the adaptive result.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


V77_STARTED = time.time()
V77_WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
V77_FIXED_OUTPUT = V77_WORK / "submission_v77_fixed_w030.csv"
V77_ADAPTIVE_OUTPUT = V77_WORK / "submission.csv"
V77_NAMED_OUTPUT = V77_WORK / "submission_v77_adaptive_dose.csv"
V77_WEIGHT_OUTPUT = V77_WORK / "v77_adaptive_dose_bywell.csv"
V77_SUMMARY = V77_WORK / "v77_adaptive_dose_summary.json"

required_live = (
    "sample", "sample_value_column", "parent_values", "coarse_by_model",
    "data_root", "V", "_load_test_record", "OUTPUT_REFIT",
)
missing_live = [name for name in required_live if name not in globals()]
if missing_live:
    raise RuntimeError(
        "Run rawdata_lattice_coarse_all773_kaggle_v69_cell.py immediately before "
        f"this cell. Missing live variables: {missing_live}"
    )


def _find_v76_bundle():
    configured = os.environ.get("ROGII_ADAPTIVE_DOSE_BUNDLE")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            V77_WORK / "adaptive_coarse_dose_bundle_v76.joblib",
            Path("/root/autodl-tmp/oof_exports/rawdata_lattice_coarse_adaptive_dose_bundle_v76/deployment/adaptive_coarse_dose_bundle_v76.joblib"),
        )
    )
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(input_root.rglob("adaptive_coarse_dose_bundle_v76.joblib"))
    found = [path.resolve() for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(
            "Upload adaptive_coarse_dose_bundle_v76.joblib as a Kaggle dataset or "
            "set ROGII_ADAPTIVE_DOSE_BUNDLE."
        )
    return found[0]


def _slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0.0 else 0.0


def _well_features(well, distance, parent, coarse):
    distance = np.asarray(distance, np.float64)
    parent = np.asarray(parent, np.float64)
    coarse = np.asarray(coarse, np.float64)
    move = coarse - parent
    if len(move) < 2 or not np.isfinite(np.column_stack((distance, parent, coarse))).all():
        raise RuntimeError(f"{well}: invalid endpoint arrays")
    row = {
        "well": str(well),
        "maximum_distance_ft": float(distance.max()),
        "move_mean": float(move.mean()),
        "move_std": float(move.std()),
        "move_rms": float(np.sqrt(np.mean(move * move))),
        "move_abs_mean": float(np.abs(move).mean()),
        "move_abs_p90": float(np.quantile(np.abs(move), 0.90)),
        "move_abs_max": float(np.max(np.abs(move))),
        "move_slope_per_ft": _slope(distance, move),
        "endpoint_correlation": float(np.corrcoef(parent, coarse)[0, 1])
        if np.std(parent) > 0.0 and np.std(coarse) > 0.0 else 0.0,
        "parent_slope_per_ft": _slope(distance, parent),
        "coarse_slope_per_ft": _slope(distance, coarse),
        "parent_roughness": float(np.sqrt(np.mean(np.diff(parent) ** 2))),
        "coarse_roughness": float(np.sqrt(np.mean(np.diff(coarse) ** 2))),
    }
    strata = (
        ("near_0_500", -np.inf, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    )
    for name, lower, upper in strata:
        mask = (distance > lower) & (distance <= upper)
        row[f"{name}_fraction"] = float(mask.mean())
        row[f"{name}_move_rms"] = float(np.sqrt(np.mean(move[mask] ** 2))) if mask.any() else 0.0
        row[f"{name}_move_mean"] = float(move[mask].mean()) if mask.any() else 0.0
    return row


def _augment(feature_frame, bundle):
    output = feature_frame.loc[:, bundle["base_features"]].astype(np.float64).copy()
    for column in bundle["log_features"]:
        output[f"log1p_{column}"] = np.log1p(np.maximum(output[column], 0.0))
    output["roughness_ratio"] = output["coarse_roughness"] / np.maximum(
        output["parent_roughness"], 1e-4
    )
    output["slope_difference"] = output["coarse_slope_per_ft"] - output["parent_slope_per_ft"]
    if output.columns.tolist() != bundle["feature_names"]:
        raise RuntimeError(
            f"V76 feature schema mismatch: {output.columns.tolist()} != {bundle['feature_names']}"
        )
    values = output.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite V77 feature matrix")
    return values


bundle_path = _find_v76_bundle()
bundle = joblib.load(bundle_path)
if bundle.get("experiment") != "rawdata_lattice_coarse_adaptive_dose_bundle_v76":
    raise RuntimeError(f"Unexpected adaptive bundle: {bundle.get('experiment')}")

v77_sample = sample.copy()
v77_ids = v77_sample["id"].astype(str)
v77_parent = np.asarray(parent_values, np.float64).copy()
v77_coarse = v77_ids.map(coarse_by_model["all773"]).to_numpy(np.float64)
v77_available = np.isfinite(v77_coarse)
if float(v77_available.mean()) < 0.98:
    raise RuntimeError(f"All773 coarse coverage too low: {v77_available.mean():.3%}")

# Preserve the exact fixed-dose V69 output before changing anything.
v77_fixed = pd.read_csv(OUTPUT_REFIT)
if not v77_fixed["id"].astype(str).equals(v77_ids):
    raise RuntimeError("V69 fixed output is not aligned with sample_submission")
v77_fixed.to_csv(V77_FIXED_OUTPUT, index=False)

id_to_position = pd.Series(np.arange(len(v77_sample), dtype=np.int64), index=v77_ids)
distance = np.full(len(v77_sample), np.nan, np.float64)
well_rows = {}
feature_rows = []
test_paths = sorted((Path(data_root) / "test").glob("*__horizontal_well.csv"))
for number, path in enumerate(test_paths, start=1):
    well = path.name.split("__", 1)[0]
    record = _load_test_record(Path(data_root), well, V["ROW_STRIDE"], V["_interp_nan"])
    eval_rows = record["eval_rows"]
    source_rows = record["source_rows"][eval_rows]
    ids = pd.Index([f"{well}_{int(row)}" for row in source_rows])
    positions = id_to_position.reindex(ids).to_numpy()
    valid_positions = np.isfinite(positions)
    if not valid_positions.all():
        raise RuntimeError(f"{well}: {int((~valid_positions).sum())} sample ids not found")
    positions = positions.astype(np.int64)
    local_distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    distance[positions] = local_distance
    well_rows[well] = positions
    if np.isfinite(v77_coarse[positions]).all():
        feature_rows.append(
            _well_features(
                well,
                local_distance,
                v77_parent[positions],
                v77_coarse[positions],
            )
        )
    if number % 10 == 0 or number == len(test_paths):
        print(f"V77 features {number}/{len(test_paths)}", flush=True)

if np.isnan(distance[v77_available]).any():
    raise RuntimeError("Missing md_since for available coarse rows")
feature_frame = pd.DataFrame(feature_rows).sort_values("well", kind="stable").reset_index(drop=True)
x = _augment(feature_frame, bundle)
hist_weight = np.clip(
    bundle["hist_model"].predict(x), bundle["min_weight"], bundle["max_weight"]
)
ridge_weight = np.clip(
    bundle["ridge_model"].predict(x), bundle["min_weight"], bundle["max_weight"]
)
adaptive_weight = np.clip(
    bundle["hist_fraction"] * hist_weight + bundle["ridge_fraction"] * ridge_weight,
    bundle["min_weight"],
    bundle["max_weight"],
)
feature_frame["hist_weight"] = hist_weight
feature_frame["ridge_weight"] = ridge_weight
feature_frame["adaptive_weight"] = adaptive_weight
feature_frame.to_csv(V77_WEIGHT_OUTPUT, index=False)

weight_by_well = dict(zip(feature_frame["well"], feature_frame["adaptive_weight"]))
v77_output_values = v77_parent.copy()
adapted_rows = 0
fallback_wells = []
for well, positions in well_rows.items():
    available = np.isfinite(v77_coarse[positions])
    if well in weight_by_well:
        weight = float(weight_by_well[well])
    else:
        weight = float(bundle["fixed_weight"])
        fallback_wells.append(well)
    selected = positions[available]
    v77_output_values[selected] = v77_parent[selected] + weight * (
        v77_coarse[selected] - v77_parent[selected]
    )
    adapted_rows += int(len(selected))

if not np.isfinite(v77_output_values).all():
    raise RuntimeError("Non-finite adaptive submission")
v77_output = v77_sample.copy()
v77_output[sample_value_column] = v77_output_values
v77_output.to_csv(V77_ADAPTIVE_OUTPUT, index=False)
v77_output.to_csv(V77_NAMED_OUTPUT, index=False)

fixed_values = v77_fixed[sample_value_column].to_numpy(np.float64)
summary = {
    "experiment": "rawdata_lattice_coarse_adaptive_kaggle_v77",
    "bundle": str(bundle_path),
    "input": "live V69 All773 coarse endpoint and 6.377 parent",
    "rows": int(len(v77_output)),
    "test_wells": int(len(well_rows)),
    "adaptive_wells": int(len(feature_frame)),
    "fallback_wells": fallback_wells,
    "coarse_coverage": float(v77_available.mean()),
    "weight": {
        "mean": float(adaptive_weight.mean()),
        "std": float(adaptive_weight.std()),
        "min": float(adaptive_weight.min()),
        "p10": float(np.quantile(adaptive_weight, 0.10)),
        "median": float(np.median(adaptive_weight)),
        "p90": float(np.quantile(adaptive_weight, 0.90)),
        "max": float(adaptive_weight.max()),
    },
    "difference_from_fixed_w030": {
        "changed_rows": int(np.sum(np.abs(v77_output_values - fixed_values) > 1e-12)),
        "mean_abs_ft": float(np.mean(np.abs(v77_output_values - fixed_values))),
        "p95_abs_ft": float(np.quantile(np.abs(v77_output_values - fixed_values), 0.95)),
        "max_abs_ft": float(np.max(np.abs(v77_output_values - fixed_values))),
    },
    "outputs": {
        "fixed_control": str(V77_FIXED_OUTPUT),
        "adaptive_submission": str(V77_ADAPTIVE_OUTPUT),
        "adaptive_named_copy": str(V77_NAMED_OUTPUT),
        "weights": str(V77_WEIGHT_OUTPUT),
    },
    "elapsed_minutes": (time.time() - V77_STARTED) / 60.0,
    "instruction": "Submit /kaggle/working/submission.csv as the adaptive-dose candidate.",
}
V77_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nV77 FINAL SUMMARY")
print(json.dumps(summary, indent=2), flush=True)
