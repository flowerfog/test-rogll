"""V190: strict transductive, per-well-debiased structural surface gateway.

This is a one-fold investment gate.  For the held-out fold, only TVT_input
prefix rows are admitted as observations.  Other folds contribute complete
TVT trajectories.  A local spatial plane and one additive datum per well are
estimated alternately before the surface is re-anchored at each target well's
last visible TVT.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "transductive_debiased_surface_gateway_v190"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = next(
    (
        path
        for path in (
            ROOT / "rogii-wellbore-geology-prediction",
            ROOT / "replay_data",
            ROOT / "data",
            ROOT,
        )
        if (path / "train").is_dir()
    ),
    None,
)
BASE_PATH = OOF_ROOT / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
BASE_COLUMN = "metadata_analog_w0p05"
HELD_FOLD = 3
QUERY_ROWS_PER_WELL = 192
TRAIN_SAMPLE_FT = 20.0
PREFIX_SAMPLE_FT = 10.0
NEIGHBORS = 128
QUERY_CHUNK = 4096
BIAS_ITERATIONS = 3
BIAS_DAMPING = 0.70
DISTANCE_FLOOR_FT = 40.0
SLOPE_RIDGE = 0.10
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989


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


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _sample_by_md(md, rows, spacing_ft):
    rows = np.asarray(rows, np.int64)
    if not len(rows):
        return rows
    selected = [int(rows[0])]
    last_md = float(md[rows[0]])
    for row in rows[1:]:
        value = float(md[row])
        if value - last_md >= spacing_ft:
            selected.append(int(row))
            last_md = value
    if selected[-1] != int(rows[-1]):
        selected.append(int(rows[-1]))
    return np.asarray(selected, np.int64)


def _evenly_spaced(rows, maximum):
    rows = np.asarray(rows, np.int64)
    if len(rows) <= maximum:
        return rows
    positions = np.unique(np.round(np.linspace(0, len(rows) - 1, maximum)).astype(int))
    return rows[positions]


def _metric(frame, prediction, endpoint):
    truth = frame["truth"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if not np.isfinite(error).all():
        raise RuntimeError(f"{endpoint}: non-finite prediction")
    by_well = (
        pd.DataFrame({"well": frame["well"], "e2": error * error})
        .groupby("well", sort=False)["e2"]
        .mean()
    )
    distance = frame["distance"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


class LocalPlaneField:
    def __init__(self, xy, surface, well_code, code_count):
        self.xy = np.asarray(xy, np.float64)
        self.surface = np.asarray(surface, np.float64)
        self.well_code = np.asarray(well_code, np.int32)
        self.code_count = int(code_count)
        self.tree = cKDTree(self.xy, leafsize=64)
        self.bias = np.zeros(self.code_count, np.float64)

    def predict(self, query_xy, target_code, adjusted_surface=None):
        query_xy = np.asarray(query_xy, np.float64)
        target_code = np.broadcast_to(np.asarray(target_code, np.int32), len(query_xy))
        values = self.surface - self.bias[self.well_code] if adjusted_surface is None else adjusted_surface
        output = np.empty(len(query_xy), np.float64)
        k = min(NEIGHBORS, len(self.xy))
        for start in range(0, len(query_xy), QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, len(query_xy))
            q = query_xy[start:stop]
            distance, index = self.tree.query(q, k=k, workers=-1)
            if k == 1:
                distance = distance[:, None]
                index = index[:, None]
            source_code = self.well_code[index]
            weight = 1.0 / np.square(np.maximum(distance + DISTANCE_FLOOR_FT, DISTANCE_FLOOR_FT))
            weight[source_code == target_code[start:stop, None]] = 0.0
            offset = (self.xy[index] - q[:, None, :]) / 1000.0
            x = offset[:, :, 0]
            y = offset[:, :, 1]
            z = values[index]
            sw = np.sum(weight, axis=1)
            fallback = np.sum(weight * z, axis=1) / np.maximum(sw, 1e-12)
            lhs = np.empty((len(q), 3, 3), np.float64)
            lhs[:, 0, 0] = sw + 1e-8
            lhs[:, 0, 1] = lhs[:, 1, 0] = np.sum(weight * x, axis=1)
            lhs[:, 0, 2] = lhs[:, 2, 0] = np.sum(weight * y, axis=1)
            lhs[:, 1, 1] = np.sum(weight * x * x, axis=1) + SLOPE_RIDGE
            lhs[:, 1, 2] = lhs[:, 2, 1] = np.sum(weight * x * y, axis=1)
            lhs[:, 2, 2] = np.sum(weight * y * y, axis=1) + SLOPE_RIDGE
            rhs = np.column_stack(
                (
                    np.sum(weight * z, axis=1),
                    np.sum(weight * x * z, axis=1),
                    np.sum(weight * y * z, axis=1),
                )
            )
            try:
                coefficient = np.linalg.solve(lhs, rhs[..., None])[..., 0]
                prediction = coefficient[:, 0]
            except np.linalg.LinAlgError:
                prediction = fallback
            bad = (sw <= 0.0) | ~np.isfinite(prediction)
            prediction[bad] = fallback[bad]
            output[start:stop] = prediction
        return output

    def fit_biases(self, observation_slices):
        reports = []
        for iteration in range(BIAS_ITERATIONS):
            adjusted = self.surface - self.bias[self.well_code]
            proposed = self.bias.copy()
            residual_scales = []
            for code, positions in enumerate(observation_slices):
                if len(positions) < 3:
                    continue
                field = self.predict(self.xy[positions], code, adjusted)
                residual = self.surface[positions] - field
                finite = np.isfinite(residual)
                if finite.sum() < 3:
                    continue
                proposed[code] = float(np.median(residual[finite]))
                residual_scales.append(float(np.median(np.abs(residual[finite] - proposed[code]))))
            proposed -= float(np.median(proposed[np.isfinite(proposed)]))
            self.bias = (1.0 - BIAS_DAMPING) * self.bias + BIAS_DAMPING * proposed
            report = {
                "iteration": iteration + 1,
                "bias_std": float(np.std(self.bias)),
                "bias_abs_p95": float(np.quantile(np.abs(self.bias), 0.95)),
                "median_observation_mad": float(np.median(residual_scales)),
            }
            reports.append(report)
            print("V190 BIAS", report, flush=True)
        return reports


for required in (BASE_PATH, FOLD_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
if DATA_ROOT is None:
    raise FileNotFoundError("No train directory")

schema = pq.read_schema(BASE_PATH).names
required_columns = {"well", "row_idx", "fold", BASE_COLUMN}
if not required_columns.issubset(schema):
    raise RuntimeError(f"V168 columns missing: {sorted(required_columns.difference(schema))}")
truth_column = next((name for name in ("tvt_true", "truth", "TVT") if name in schema), None)
distance_column = next((name for name in ("md_since", "distance") if name in schema), None)
if truth_column is None or distance_column is None:
    raise RuntimeError(f"Cannot identify truth/distance columns: {schema}")

base = pd.read_parquet(
    BASE_PATH,
    columns=["well", "row_idx", "fold", truth_column, distance_column, BASE_COLUMN],
)
base = base.rename(columns={truth_column: "truth", distance_column: "distance", BASE_COLUMN: "base"})
base["well"] = base["well"].astype(str)
base = base.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"V168 coverage mismatch: {len(base)}/{base['well'].nunique()}")

fold_frame = pd.read_csv(FOLD_PATH)
fold_frame["well"] = fold_frame["well"].astype(str)
fold_by_well = fold_frame.groupby("well")["fold"].first().astype(int).to_dict()
wells = sorted(base["well"].unique())
well_to_code = {well: code for code, well in enumerate(wells)}

joint_xy = []
joint_surface = []
joint_code = []
train_xy = []
train_surface = []
train_code = []
joint_slices = [[] for _ in wells]
train_slices = [[] for _ in wells]
native = {}
joint_offset = 0
train_offset = 0

for number, well in enumerate(wells, 1):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
    md = frame["MD"].to_numpy(np.float64)
    xy = frame[["X", "Y"]].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
    tvt = frame["TVT"].to_numpy(np.float64)
    tvt_input = frame["TVT_input"].to_numpy(np.float64)
    fold = int(fold_by_well[well])
    visible = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(z) & np.isfinite(xy).all(axis=1))
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(tvt) & np.isfinite(z))
    if not len(visible) or not len(hidden):
        raise RuntimeError(f"{well}: missing visible/hidden rows")
    if fold == HELD_FOLD:
        rows = _sample_by_md(md, visible, PREFIX_SAMPLE_FT)
        values = tvt_input[rows] + z[rows]
    else:
        finite = np.flatnonzero(np.isfinite(tvt) & np.isfinite(z) & np.isfinite(xy).all(axis=1))
        rows = _sample_by_md(md, finite, TRAIN_SAMPLE_FT)
        values = tvt[rows] + z[rows]
        train_xy.append(xy[rows])
        train_surface.append(values)
        train_code.append(np.full(len(rows), well_to_code[well], np.int32))
        train_slices[well_to_code[well]] = np.arange(train_offset, train_offset + len(rows), dtype=np.int64)
        train_offset += len(rows)
    joint_xy.append(xy[rows])
    joint_surface.append(values)
    joint_code.append(np.full(len(rows), well_to_code[well], np.int32))
    joint_slices[well_to_code[well]] = np.arange(joint_offset, joint_offset + len(rows), dtype=np.int64)
    joint_offset += len(rows)
    native[well] = {"xy": xy, "z": z, "tvt_input": tvt_input, "visible": visible}
    if number % 100 == 0 or number == len(wells):
        print(f"V190 LOAD {number}/{len(wells)} joint_points={joint_offset}", flush=True)

joint_field = LocalPlaneField(
    np.concatenate(joint_xy), np.concatenate(joint_surface), np.concatenate(joint_code), len(wells)
)
train_field = LocalPlaneField(
    np.concatenate(train_xy), np.concatenate(train_surface), np.concatenate(train_code), len(wells)
)
joint_reports = joint_field.fit_biases(joint_slices)
train_reports = train_field.fit_biases(train_slices)

held = base[base["fold"].eq(HELD_FOLD)].copy()
sample_parts = []
for well, group in held.groupby("well", sort=True):
    sample_parts.append(group.iloc[_evenly_spaced(np.arange(len(group)), QUERY_ROWS_PER_WELL)])
sample = pd.concat(sample_parts, ignore_index=True).reset_index(drop=True)


def _surface_prediction(field, use_bias, label):
    parts = []
    for number, (well, group) in enumerate(sample.groupby("well", sort=True), 1):
        record = native[well]
        rows = group["row_idx"].to_numpy(np.int64)
        anchor = int(record["visible"][-1])
        query_rows = np.r_[anchor, rows]
        if use_bias:
            surface = field.predict(record["xy"][query_rows], well_to_code[well])
        else:
            raw = field.surface.copy()
            saved = field.bias.copy()
            field.bias[:] = 0.0
            surface = field.predict(record["xy"][query_rows], well_to_code[well], raw)
            field.bias[:] = saved
        anchor_tvt = float(record["tvt_input"][anchor])
        prediction = anchor_tvt + (surface[1:] - surface[0]) - (
            record["z"][rows] - record["z"][anchor]
        )
        parts.append(pd.DataFrame({"index": group.index, label: prediction}))
        if number % 50 == 0 or number == sample["well"].nunique():
            print(f"V190 PREDICT {label} {number}/{sample['well'].nunique()}", flush=True)
    joined = pd.concat(parts).set_index("index").loc[sample.index]
    return joined[label].to_numpy(np.float64)


prediction = {
    "joint_debiased": _surface_prediction(joint_field, True, "joint_debiased"),
    "joint_no_bias": _surface_prediction(joint_field, False, "joint_no_bias"),
    "trainonly_debiased": _surface_prediction(train_field, True, "trainonly_debiased"),
}
base_prediction = sample["base"].to_numpy(np.float64)
truth = sample["truth"].to_numpy(np.float64)
summary = [_metric(sample, base_prediction, "v168_sample192")]
blend_rows = []

for name, candidate in prediction.items():
    summary.append(_metric(sample, candidate, name))
    move = candidate - base_prediction
    alpha = float(np.clip(np.dot(truth - base_prediction, move) / max(np.dot(move, move), 1e-12), 0.0, 1.0))
    blended = base_prediction + alpha * move
    summary.append(_metric(sample, blended, f"{name}_global_oracle_blend"))
    perwell = np.empty(len(sample), np.float64)
    alphas = []
    for well, group in sample.groupby("well", sort=False):
        idx = group.index.to_numpy(np.int64)
        local_move = move[idx]
        local_alpha = float(np.clip(
            np.dot(truth[idx] - base_prediction[idx], local_move)
            / max(np.dot(local_move, local_move), 1e-12),
            0.0,
            1.0,
        ))
        perwell[idx] = base_prediction[idx] + local_alpha * local_move
        alphas.append(local_alpha)
    summary.append(_metric(sample, perwell, f"{name}_perwell_oracle"))
    blend_rows.append({
        "candidate": name,
        "global_oracle_alpha": alpha,
        "positive_well_alpha_fraction": float(np.mean(np.asarray(alphas) > 0.0)),
        "interior_well_alpha_fraction": float(np.mean((np.asarray(alphas) > 0.0) & (np.asarray(alphas) < 1.0))),
    })

summary = pd.DataFrame(summary).sort_values("rmse")
blend_frame = pd.DataFrame(blend_rows)
lookup = summary.set_index("endpoint")
base_rmse = float(lookup.loc["v168_sample192", "rmse"])
joint_blend_rmse = float(lookup.loc["joint_debiased_global_oracle_blend", "rmse"])
train_blend_rmse = float(lookup.loc["trainonly_debiased_global_oracle_blend", "rmse"])
joint_perwell_rmse = float(lookup.loc["joint_debiased_perwell_oracle", "rmse"])
checks = {
    "joint_global_oracle_gain_ge_0p15": base_rmse - joint_blend_rmse >= 0.15,
    "joint_perwell_oracle_rmse_le_5p5": joint_perwell_rmse <= 5.5,
    "joint_beats_trainonly_blend_ge_0p05": train_blend_rmse - joint_blend_rmse >= 0.05,
    "bias_iteration_improves_standalone": float(lookup.loc["joint_no_bias", "rmse"]) > float(lookup.loc["joint_debiased", "rmse"]),
}
verdict = {
    "experiment": "transductive_debiased_surface_gateway_v190",
    "stage": "Fold3_strict_transductive_surface_investment_gate",
    "heldout_truth_usage": "metrics and oracle doses only",
    "observations": {
        "outer_train": "complete TVT+Z sampled every 20 ft",
        "heldout": "TVT_input+Z visible prefixes sampled every 10 ft",
        "joint_points": len(joint_field.xy),
        "trainonly_points": len(train_field.xy),
    },
    "field": "local weighted XY plane with alternating additive well datum",
    "bias_iterations": joint_reports,
    "trainonly_bias_iterations": train_reports,
    "blend": blend_frame.to_dict("records"),
    "values": {
        "base_rmse": base_rmse,
        "joint_global_oracle_gain": base_rmse - joint_blend_rmse,
        "joint_perwell_oracle_rmse": joint_perwell_rmse,
        "joint_increment_vs_trainonly": train_blend_rmse - joint_blend_rmse,
    },
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> run all five folds and train a legal confidence/dose gate"
        if all(checks.values())
        else "FAIL -> joint hidden-prefix spatial field cannot unlock CV<6; close this route"
    ),
}

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
blend_frame.to_csv(OUT_DIR / "blend_audit.csv", index=False)
sample_output = sample[["well", "fold", "row_idx", "truth", "distance", "base"]].copy()
for name, values in prediction.items():
    sample_output[name] = values.astype(np.float32)
sample_output.to_parquet(OUT_DIR / "sampled_predictions.parquet", index=False)
_write_json(OUT_DIR / "final_verdict.json", verdict)

print("\nV190 SUMMARY")
print(summary.to_string(index=False))
print("\nV190 BLEND")
print(blend_frame.to_string(index=False))
print("\nV190 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
print("saved:", OUT_DIR)
