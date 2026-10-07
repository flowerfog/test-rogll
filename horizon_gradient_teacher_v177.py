"""V177: legal gradient distillation of train-only structural horizons.

The mapped horizons are used only to create smooth training-rate labels.  The
student consumes columns present in test horizontal wells, predicts the local
rate of S = TVT + Z, removes a rate bias measured on the visible TVT_input
prefix, and integrates from the final visible anchor.

All model fitting, prefix-recipe selection, and blend fitting are nested by
complete physical well folds.  Run as a script or with ``%run`` in Jupyter.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


EXPERIMENT = "horizon_gradient_teacher_v177"
CWD = Path.cwd()
ROOT = CWD if (CWD / "train").is_dir() else Path("/root/autodl-tmp")
DATA_ROOT = next(
    (path for path in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (path / "train").is_dir()),
    ROOT,
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
V174_DIR = OOF_ROOT / "horizon_structural_coordinate_gateway_v174"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
LEGAL_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
VISIBLE_NODES = 96
FUTURE_NODES = 192
PREFIX_WINDOWS = (512.0, 1024.0, 2048.0)
RECIPES = ("raw", *(f"bias{int(window)}" for window in PREFIX_WINDOWS))
ROLL_FRACTION = 0.20
SEED = 177


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _metric(frame, prediction, endpoint):
    truth = frame["truth"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if not np.isfinite(error).all():
        raise RuntimeError(f"{endpoint}: non-finite predictions")
    by_well = (
        pd.DataFrame({"well": frame["well"], "e2": error * error})
        .groupby("well", sort=False)["e2"]
        .mean()
    )
    far = frame["distance"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _fill(values, md):
    values = np.asarray(values, np.float64).copy()
    valid = np.isfinite(values) & np.isfinite(md)
    if int(valid.sum()) >= 2:
        return np.interp(md, md[valid], values[valid])
    if int(valid.sum()) == 1:
        return np.full(len(md), values[valid][0], np.float64)
    return np.zeros(len(md), np.float64)


def _derivative(values, md):
    step = np.gradient(md)
    good = np.isfinite(step) & (np.abs(step) > 1e-4)
    replacement = float(np.median(np.abs(step[good]))) if np.any(good) else 1.0
    step = np.where(good, step, replacement)
    return np.gradient(values) / step


def _rolling(values, window, statistic="mean"):
    series = pd.Series(values)
    rolling = series.rolling(window, center=True, min_periods=max(3, window // 8))
    if statistic == "std":
        result = rolling.std(ddof=0).to_numpy(np.float64)
    else:
        result = rolling.mean().to_numpy(np.float64)
    return _fill(result, np.arange(len(result), dtype=np.float64))


def _nodes(length, anchor, eval_rows):
    visible = np.arange(anchor + 1, dtype=np.int64)
    visible_take = np.unique(
        np.round(np.linspace(0, len(visible) - 1, min(VISIBLE_NODES, len(visible)))).astype(np.int64)
    )
    future_take = np.unique(
        np.round(np.linspace(0, len(eval_rows) - 1, min(FUTURE_NODES, len(eval_rows)))).astype(np.int64)
    )
    return np.unique(np.r_[visible[visible_take], anchor, eval_rows[future_take]]).astype(np.int64)


def _legal_features(horizontal, nodes, anchor):
    md = horizontal["MD"].to_numpy(np.float64)
    x = _fill(horizontal["X"].to_numpy(np.float64), md)
    y = _fill(horizontal["Y"].to_numpy(np.float64), md)
    z = _fill(horizontal["Z"].to_numpy(np.float64), md)
    gr = _fill(horizontal["GR"].to_numpy(np.float64), md)
    gr_median = float(np.median(gr))
    gr_scale = float(np.quantile(gr, 0.75) - np.quantile(gr, 0.25))
    gr_scale = max(gr_scale, float(np.std(gr)) * 0.25, 1e-3)
    gr_z = np.clip((gr - gr_median) / gr_scale, -8.0, 8.0)

    dx = _derivative(x, md)
    dy = _derivative(y, md)
    dz = _derivative(z, md)
    ddx = _derivative(dx, md)
    ddy = _derivative(dy, md)
    ddz = _derivative(dz, md)
    gr_s33 = _rolling(gr_z, 33)
    gr_s129 = _rolling(gr_z, 129)
    gr_std33 = _rolling(gr_z, 33, "std")
    gr_std129 = _rolling(gr_z, 129, "std")
    dgr = np.clip(_derivative(gr_s33, md), -2.0, 2.0)

    span = max(float(md[-1] - md[0]), 1.0)
    future_span = max(float(md[-1] - md[anchor]), 1.0)
    visible = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(np.float64)))
    s_visible = horizontal["TVT_input"].to_numpy(np.float64)[visible] + z[visible]
    visible_rate = _derivative(s_visible, md[visible])
    prefix_rate = float(np.median(visible_rate[-min(128, len(visible_rate)) :]))
    prefix_rate_std = float(np.std(visible_rate[-min(128, len(visible_rate)) :]))
    total_dx = float((x[-1] - x[anchor]) / future_span)
    total_dy = float((y[-1] - y[anchor]) / future_span)
    total_dz = float((z[-1] - z[anchor]) / future_span)

    matrix = np.column_stack(
        (
            (md[nodes] - md[anchor]) / 1000.0,
            (md[nodes] - md[0]) / span,
            (x[nodes] - x[anchor]) / 1000.0,
            (y[nodes] - y[anchor]) / 1000.0,
            (z[nodes] - z[anchor]) / 1000.0,
            x[nodes] / 10000.0,
            y[nodes] / 10000.0,
            z[nodes] / 1000.0,
            dx[nodes], dy[nodes], dz[nodes],
            100.0 * ddx[nodes], 100.0 * ddy[nodes], 100.0 * ddz[nodes],
            gr_z[nodes], gr_s33[nodes], gr_s129[nodes],
            gr_z[nodes] - gr_s33[nodes], gr_z[nodes] - gr_s129[nodes],
            gr_std33[nodes], gr_std129[nodes], dgr[nodes],
            np.full(len(nodes), prefix_rate),
            np.full(len(nodes), prefix_rate_std),
            np.full(len(nodes), total_dx),
            np.full(len(nodes), total_dy),
            np.full(len(nodes), total_dz),
            (nodes >= anchor).astype(np.float64),
        )
    )
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(matrix, -100.0, 100.0).astype(np.float32)


def _teacher_basis_rate(horizontal, nodes, anchor):
    md = horizontal["MD"].to_numpy(np.float64)
    surfaces = np.column_stack(
        [_fill(horizontal[name].to_numpy(np.float64), md) for name in HORIZONS]
    )
    delta = surfaces - surfaces[anchor]
    xy = horizontal[["X", "Y"]].to_numpy(np.float64)
    basis = np.column_stack(
        (
            delta,
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            (md - md[anchor]) / 1000.0,
            (xy[:, 0] - xy[anchor, 0]) / 1000.0,
            (xy[:, 1] - xy[anchor, 1]) / 1000.0,
        )
    )
    return np.column_stack([_derivative(basis[:, index], md)[nodes] for index in range(basis.shape[1])])


def _integrate_prediction(horizontal, nodes, eval_rows, anchor, rate, bias):
    md = horizontal["MD"].to_numpy(np.float64)
    z = horizontal["Z"].to_numpy(np.float64)
    future_nodes = np.unique(np.r_[anchor, nodes[nodes >= anchor], eval_rows[-1]])
    node_rate = np.interp(future_nodes, nodes, rate) - bias
    delta_md = np.diff(md[future_nodes])
    integral = np.r_[0.0, np.cumsum(0.5 * (node_rate[1:] + node_rate[:-1]) * delta_md)]
    surface = float(horizontal["TVT_input"].iloc[anchor]) + z[anchor] + np.interp(eval_rows, future_nodes, integral)
    return surface - z[eval_rows]


started = time.time()
for required in (V174_DIR / "oof.csv.gz", V174_DIR / "coefficients.csv"):
    if not required.is_file():
        raise FileNotFoundError(f"Run V174 first: {required}")

base_frame = pd.read_csv(
    V174_DIR / "oof.csv.gz",
    usecols=["well", "fold", "row_idx", "truth", "distance", "base_current"],
    dtype={"well": str, "fold": np.int8, "row_idx": np.int32},
).rename(columns={"base_current": "base"})
base_frame = base_frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base_frame) != EXPECTED_ROWS or base_frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError("Incomplete V174 OOF")

test_files = sorted((DATA_ROOT / "test").glob("*__horizontal_well.csv"))
test_schema_legal = bool(test_files) and all(
    set(LEGAL_COLUMNS).issubset(pd.read_csv(path, nrows=0).columns) for path in test_files
)
fold_map = base_frame.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
coefficient_table = pd.read_csv(V174_DIR / "coefficients.csv").set_index("heldout_fold")
coefficient_columns = [
    *(f"coef_{name}" for name in HORIZONS),
    "coef_horizon_mean", "coef_horizon_std", "coef_md", "coef_x", "coef_y",
]

records = {}
feature_parts = []
well_parts = []
fold_parts = []
for number, well in enumerate(sorted(fold_map), 1):
    local = base_frame[base_frame["well"].eq(well)]
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=[*LEGAL_COLUMNS, "TVT", *HORIZONS],
    )
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    visible = np.flatnonzero(np.isfinite(tvt_input))
    eval_rows = local["row_idx"].to_numpy(np.int64)
    official = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(horizontal["TVT"].to_numpy(np.float64)))
    if not len(visible) or not np.array_equal(eval_rows, official):
        raise RuntimeError(f"{well}: evaluation mask mismatch")
    if np.max(np.abs(horizontal["TVT"].to_numpy(np.float64)[eval_rows] - local["truth"].to_numpy(np.float64))) > 0.01:
        raise RuntimeError(f"{well}: truth parity failed")
    anchor = int(visible[-1])
    nodes = _nodes(len(horizontal), anchor, eval_rows)
    features = _legal_features(horizontal, nodes, anchor)
    records[well] = {
        "positions": local.index.to_numpy(np.int64),
        "horizontal": horizontal,
        "nodes": nodes,
        "anchor": anchor,
        "eval_rows": eval_rows,
        "features": features,
        "teacher_basis_rate": _teacher_basis_rate(horizontal, nodes, anchor).astype(np.float32),
    }
    feature_parts.append(features)
    well_parts.extend([well] * len(nodes))
    fold_parts.extend([fold_map[well]] * len(nodes))
    if number % 100 == 0 or number == EXPECTED_WELLS:
        print(f"V177 records {number}/{EXPECTED_WELLS}", flush=True)

all_features = np.concatenate(feature_parts)
all_wells = np.asarray(well_parts, str)
all_folds = np.asarray(fold_parts, np.int8)
offset = 0
for well in sorted(records):
    count = len(records[well]["nodes"])
    records[well]["sample_slice"] = slice(offset, offset + count)
    offset += count

candidates = {name: np.full(len(base_frame), np.nan, np.float64) for name in RECIPES}
rolled_candidates = {name: np.full(len(base_frame), np.nan, np.float64) for name in RECIPES}
model_rows = []
for held_fold in range(1, 6):
    train = all_folds != held_fold
    coefficient = coefficient_table.loc[held_fold, coefficient_columns].to_numpy(np.float64)
    target_parts = []
    train_well_parts = []
    for well in sorted(records):
        if fold_map[well] == held_fold:
            continue
        record = records[well]
        target_parts.append(record["teacher_basis_rate"] @ coefficient)
        train_well_parts.extend([well] * len(record["nodes"]))
    target = np.concatenate(target_parts)
    if not np.isfinite(target).all():
        raise RuntimeError(f"Fold{held_fold}: non-finite teacher-rate labels")
    train_wells = np.asarray(train_well_parts, str)
    counts = pd.Series(train_wells).value_counts().to_dict()
    sample_weight = np.asarray([1.0 / counts[well] for well in train_wells], np.float64)
    sample_weight *= len(sample_weight) / np.sum(sample_weight)
    model = ExtraTreesRegressor(
        n_estimators=128,
        max_depth=20,
        min_samples_leaf=12,
        max_features=0.80,
        n_jobs=-1,
        random_state=SEED + held_fold,
    )
    model.fit(all_features[train], target, sample_weight=sample_weight)

    fold_wells = [well for well in sorted(records) if fold_map[well] == held_fold]
    for number, well in enumerate(fold_wells, 1):
        record = records[well]
        horizontal = record["horizontal"]
        nodes = record["nodes"]
        anchor = record["anchor"]
        eval_rows = record["eval_rows"]
        positions = record["positions"]
        features = record["features"]
        rate = model.predict(features)
        shift = max(1, int(round(len(features) * ROLL_FRACTION)))
        rolled_rate = model.predict(np.roll(features, shift, axis=0))
        md = horizontal["MD"].to_numpy(np.float64)
        visible_nodes = nodes[nodes <= anchor]
        s_input = horizontal["TVT_input"].to_numpy(np.float64) + horizontal["Z"].to_numpy(np.float64)
        actual_rate = _derivative(s_input[: anchor + 1], md[: anchor + 1])
        predicted_visible = np.interp(visible_nodes, nodes, rate)
        rolled_visible = np.interp(visible_nodes, nodes, rolled_rate)
        for recipe in RECIPES:
            if recipe == "raw":
                bias = 0.0
                rolled_bias = 0.0
            else:
                window = float(recipe.replace("bias", ""))
                keep = md[anchor] - md[visible_nodes] <= window
                residual = predicted_visible[keep] - actual_rate[visible_nodes[keep]]
                rolled_residual = rolled_visible[keep] - actual_rate[visible_nodes[keep]]
                bias = float(np.median(residual))
                rolled_bias = float(np.median(rolled_residual))
            candidates[recipe][positions] = _integrate_prediction(
                horizontal, nodes, eval_rows, anchor, rate, bias
            )
            rolled_candidates[recipe][positions] = _integrate_prediction(
                horizontal, nodes, eval_rows, anchor, rolled_rate, rolled_bias
            )
        if number % 50 == 0 or number == len(fold_wells):
            print(f"V177 fold{held_fold} {number}/{len(fold_wells)}", flush=True)
    model_rows.append({
        "heldout_fold": held_fold,
        "train_wells": int(len(set(train_wells))),
        "train_nodes": int(train.sum()),
        "target_rate_std": float(np.std(target)),
    })
    del model, target, target_parts, train_wells, sample_weight
    gc.collect()

base = base_frame["base"].to_numpy(np.float64)
truth = base_frame["truth"].to_numpy(np.float64)
fold_array = base_frame["fold"].to_numpy(np.int8)
nested_candidate = np.full(len(base_frame), np.nan, np.float64)
nested_blend = np.full(len(base_frame), np.nan, np.float64)
nested_rolled = np.full(len(base_frame), np.nan, np.float64)
selection_rows = []
for held_fold in range(1, 6):
    train = fold_array != held_fold
    valid = ~train
    train_rmse = {
        name: float(np.sqrt(np.mean((prediction[train] - truth[train]) ** 2)))
        for name, prediction in candidates.items()
    }
    selected = min(train_rmse, key=train_rmse.get)
    prediction = candidates[selected]
    move = prediction[train] - base[train]
    alpha = float(np.dot(truth[train] - base[train], move) / max(np.dot(move, move), 1e-12))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nested_candidate[valid] = prediction[valid]
    nested_blend[valid] = base[valid] + alpha * (prediction[valid] - base[valid])
    nested_rolled[valid] = base[valid] + alpha * (rolled_candidates[selected][valid] - base[valid])
    selection_rows.append({
        "heldout_fold": held_fold,
        "selected_recipe": selected,
        "selected_alpha": alpha,
        "train_candidate_rmse": train_rmse[selected],
    })

perwell_oracle = np.empty(len(base_frame), np.float64)
for well, local in base_frame.groupby("well", sort=False):
    positions = local.index.to_numpy(np.int64)
    move = nested_candidate[positions] - base[positions]
    residual = truth[positions] - base[positions]
    alpha = float(np.dot(residual, move) / max(np.dot(move, move), 1e-12))
    alpha = float(np.clip(alpha, -1.0, 1.0))
    perwell_oracle[positions] = base[positions] + alpha * move

summary_rows = [_metric(base_frame, base, "base_current")]
summary_rows += [_metric(base_frame, prediction, name) for name, prediction in candidates.items()]
summary_rows += [
    _metric(base_frame, nested_candidate, "nested_student"),
    _metric(base_frame, nested_blend, "nested_blend"),
    _metric(base_frame, nested_rolled, "rolled_feature_control"),
    _metric(base_frame, perwell_oracle, "perwell_direction_oracle"),
]
summary = pd.DataFrame(summary_rows)
fold_rows = []
for fold in range(1, 6):
    mask = fold_array == fold
    local = base_frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in (
        ("base_current", base[mask]),
        ("nested_student", nested_candidate[mask]),
        ("nested_blend", nested_blend[mask]),
        ("rolled_feature_control", nested_rolled[mask]),
    ):
        fold_rows.append({"fold": fold, **_metric(local, prediction, endpoint)})
fold_summary = pd.DataFrame(fold_rows)

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
pd.DataFrame(selection_rows).to_csv(OUT_DIR / "nested_selections.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_reports.csv", index=False)
output = base_frame[["well", "fold", "row_idx", "truth", "distance"]].copy()
output["base_current"] = base.astype(np.float32)
output["nested_student"] = nested_candidate.astype(np.float32)
output["nested_blend"] = nested_blend.astype(np.float32)
output.to_csv(OUT_DIR / "oof.csv.gz", index=False, compression="gzip")

lookup = summary.set_index("endpoint")
base_rmse = float(lookup.loc["base_current", "rmse"])
student_rmse = float(lookup.loc["nested_student", "rmse"])
blend_rmse = float(lookup.loc["nested_blend", "rmse"])
rolled_rmse = float(lookup.loc["rolled_feature_control", "rmse"])
oracle_rmse = float(lookup.loc["perwell_direction_oracle", "rmse"])
pivot = fold_summary.pivot(index="fold", columns="endpoint", values="rmse")
fold_gains = (pivot["base_current"] - pivot["nested_blend"]).to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_legal_gradient_distillation_gateway",
    "base_source": str(V174_DIR / "oof.csv.gz"),
    "legal_inference_columns": list(LEGAL_COLUMNS),
    "train_only_teacher_columns": list(HORIZONS),
    "historical_prediction_inputs": [],
    "test_schema_legal": test_schema_legal,
    "coverage": {"rows": len(base_frame), "wells": base_frame["well"].nunique()},
    "nested_selections": selection_rows,
    "values": {
        "base_rmse": base_rmse,
        "student_rmse": student_rmse,
        "nested_blend_rmse": blend_rmse,
        "nested_gain": base_rmse - blend_rmse,
        "aligned_increment_vs_rolled": rolled_rmse - blend_rmse,
        "perwell_direction_oracle_rmse": oracle_rmse,
        "fold_gains": fold_gains,
    },
    "checks": {
        "test_schema_legal": test_schema_legal,
        "student_rmse_le_8": student_rmse <= 8.0,
        "nested_gain_ge_0p20": base_rmse - blend_rmse >= 0.20,
        "positive_folds_5of5": all(value > 0.0 for value in fold_gains.values()),
        "minimum_fold_gain_nonnegative": min(fold_gains.values()) >= 0.0,
        "aligned_beats_rolled_ge_0p20": rolled_rmse - blend_rmse >= 0.20,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> package the frozen student for current-endpoint confirmation"
    if verdict["strict_pass"]
    else "FAIL -> permanently close horizon distillation and return to native GR/typewell state-space decoding"
)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, ensure_ascii=True), encoding="utf-8"
)

print("\nV177 MATCHED SUMMARY")
print(summary.sort_values("rmse").to_string(index=False), flush=True)
print("\nV177 FOLD SUMMARY")
print(fold_summary.to_string(index=False), flush=True)
print("\nV177 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
