"""V176: visible-prefix calibration of distilled formation surfaces.

V174 showed that six mapped formation surfaces almost exactly generate TVT, but
those columns are absent from test. V175 spatially distilled the surfaces but
could not recover their per-well magnitude. V176 calibrates that structural
basis using only each target well's visible TVT_input prefix.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EXPERIMENT = "horizon_teacher_prefix_calibration_v176"
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
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
REFERENCE_STRIDE = 48
QUERY_NODES = 128
QUERY_NEIGHBOURS = 512
LOCAL_WELLS = 16
DISTANCE_FLOOR = 100.0
PLANE_RIDGE = 0.10
ROLL_FRACTION = 0.20
PREFIX_WINDOWS = (0.0, 512.0, 1024.0, 2048.0)
VECTOR_STRENGTHS = (0.01, 0.10, 1.0)


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


def _fill_surfaces(frame):
    md = frame["MD"].to_numpy(np.float64)
    values = frame[list(HORIZONS)].to_numpy(np.float64)
    for column in range(values.shape[1]):
        valid = np.isfinite(md) & np.isfinite(values[:, column])
        if int(valid.sum()) >= 2:
            values[:, column] = np.interp(md, md[valid], values[valid, column])
        elif int(valid.sum()) == 1:
            values[:, column] = values[valid, column][0]
        else:
            values[:, column] = 0.0
    return values


def _local_surface(tree, ref_xy, ref_value, ref_code, query_xy):
    k = min(QUERY_NEIGHBOURS, len(ref_xy))
    distance, index = tree.query(query_xy, k=k, workers=-1)
    distance = np.atleast_2d(distance)
    index = np.atleast_2d(index)
    output = np.empty((len(query_xy), len(HORIZONS)), np.float64)
    support = np.empty(len(query_xy), np.int16)
    uncertainty = np.empty(len(query_xy), np.float64)
    for row in range(len(query_xy)):
        seen = set()
        selected = []
        for candidate in index[row]:
            code = int(ref_code[candidate])
            if code in seen:
                continue
            seen.add(code)
            selected.append(int(candidate))
            if len(selected) >= LOCAL_WELLS:
                break
        selected = np.asarray(selected, np.int64)
        if len(selected) < 1:
            raise RuntimeError("No distinct reference wells")
        offset = (ref_xy[selected] - query_xy[row][None, :]) / 1000.0
        local_distance = np.linalg.norm(ref_xy[selected] - query_xy[row][None, :], axis=1)
        weight = 1.0 / np.square(local_distance + DISTANCE_FLOOR)
        if len(selected) >= 3:
            design = np.column_stack((np.ones(len(selected)), offset))
            penalty = np.diag([1e-8, PLANE_RIDGE, PLANE_RIDGE])
            lhs = design.T @ (weight[:, None] * design) + penalty
            rhs = design.T @ (weight[:, None] * ref_value[selected])
            coefficient = np.linalg.solve(lhs, rhs)
            output[row] = coefficient[0]
            residual = ref_value[selected] - design @ coefficient
        else:
            output[row] = np.average(ref_value[selected], axis=0, weights=weight)
            residual = ref_value[selected] - output[row][None, :]
        uncertainty[row] = float(
            np.sqrt(np.mean(np.average(residual * residual, axis=0, weights=weight)))
        )
        support[row] = len(selected)
    return output, support, uncertainty


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
fold_map = base_frame.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
coefficients = pd.read_csv(V174_DIR / "coefficients.csv").set_index("heldout_fold")
coefficient_columns = [
    *(f"coef_{name}" for name in HORIZONS),
    "coef_horizon_mean", "coef_horizon_std", "coef_md", "coef_x", "coef_y",
]

# Build an equal-density spatial teacher table. Horizon columns are labels only.
reference_xy = []
reference_value = []
reference_fold = []
reference_code = []
well_names = sorted(fold_map)
well_code = {well: index for index, well in enumerate(well_names)}
raw_cache = {}
for number, well in enumerate(well_names, 1):
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input", *HORIZONS],
    )
    surface = _fill_surfaces(horizontal)
    take = np.arange(0, len(horizontal), REFERENCE_STRIDE, dtype=np.int32)
    if take[-1] != len(horizontal) - 1:
        take = np.r_[take, len(horizontal) - 1]
    reference_xy.append(horizontal[["X", "Y"]].to_numpy(np.float64)[take])
    reference_value.append(surface[take])
    reference_fold.extend([fold_map[well]] * len(take))
    reference_code.extend([well_code[well]] * len(take))
    raw_cache[well] = (horizontal, surface)
    if number % 100 == 0 or number == EXPECTED_WELLS:
        print(f"V176 references {number}/{EXPECTED_WELLS}", flush=True)

reference_xy = np.concatenate(reference_xy)
reference_value = np.concatenate(reference_value)
reference_fold = np.asarray(reference_fold, np.int8)
reference_code = np.asarray(reference_code, np.int16)

def _basis(horizontal, rows, anchor, predicted, anchor_surface):
    delta = predicted - anchor_surface[None, :]
    md = horizontal["MD"].to_numpy(np.float64)
    xy = horizontal[["X", "Y"]].to_numpy(np.float64)
    xy_delta = xy[rows] - xy[anchor]
    return np.column_stack((
        delta,
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        (md[rows] - md[anchor]) / 1000.0,
        xy_delta[:, 0] / 1000.0,
        xy_delta[:, 1] / 1000.0,
    ))


def _prefix_mask(md, anchor, rows, window):
    if window <= 0.0:
        return np.ones(len(rows), dtype=bool)
    return md[anchor] - md[rows] <= window


def _vector_calibration(x, y, prior, strength):
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    xs = x / scale
    prior_scaled = prior * scale
    gram = xs.T @ xs
    penalty = strength * max(float(np.mean(np.diag(gram))), 1.0)
    coefficient_scaled = np.linalg.solve(
        gram + penalty * np.eye(xs.shape[1]),
        xs.T @ y + penalty * prior_scaled,
    )
    return coefficient_scaled / scale


recipe_names = ["global"]
recipe_names += [f"scalar_w{int(window)}" for window in PREFIX_WINDOWS]
recipe_names += [
    f"vector_w{int(window)}_r{str(strength).replace('.', 'p')}"
    for window in PREFIX_WINDOWS
    for strength in VECTOR_STRENGTHS
]
candidates = {
    name: np.full(len(base_frame), np.nan, np.float64) for name in recipe_names
}
rolled = np.full(len(base_frame), np.nan, np.float64)
perwell_oracle = np.full(len(base_frame), np.nan, np.float64)
well_reports = []
for held_fold in range(1, 6):
    allowed = reference_fold != held_fold
    ref_xy = reference_xy[allowed]
    ref_value = reference_value[allowed]
    ref_code = reference_code[allowed]
    tree = cKDTree(ref_xy)
    prior = coefficients.loc[held_fold, coefficient_columns].to_numpy(np.float64)
    fold_wells = [well for well in well_names if fold_map[well] == held_fold]
    for number, well in enumerate(fold_wells, 1):
        local = base_frame[base_frame["well"].eq(well)]
        positions = local.index.to_numpy(np.int64)
        eval_rows = local["row_idx"].to_numpy(np.int64)
        horizontal, _ = raw_cache[well]
        tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
        visible_rows = np.flatnonzero(np.isfinite(tvt_input))
        anchor = int(visible_rows[-1])
        all_nodes = np.unique(
            np.r_[
                anchor,
                np.round(np.linspace(0, len(horizontal) - 1, 2 * QUERY_NODES)).astype(int),
            ]
        )
        xy_all = horizontal[["X", "Y"]].to_numpy(np.float64)
        predicted_nodes, support, uncertainty = _local_surface(
            tree, ref_xy, ref_value, ref_code, xy_all[all_nodes]
        )
        anchor_surface = predicted_nodes[np.flatnonzero(all_nodes == anchor)[0]]

        calibration_rows = visible_rows[
            np.unique(np.round(np.linspace(0, len(visible_rows) - 1, QUERY_NODES)).astype(int))
        ]
        predicted_visible = np.column_stack([
            np.interp(calibration_rows, all_nodes, predicted_nodes[:, column])
            for column in range(len(HORIZONS))
        ])
        predicted_eval = np.column_stack([
            np.interp(eval_rows, all_nodes, predicted_nodes[:, column])
            for column in range(len(HORIZONS))
        ])
        x_visible = _basis(
            horizontal, calibration_rows, anchor, predicted_visible, anchor_surface
        )
        x_eval = _basis(horizontal, eval_rows, anchor, predicted_eval, anchor_surface)
        md = horizontal["MD"].to_numpy(np.float64)
        z = horizontal["Z"].to_numpy(np.float64)
        anchor_tvt = float(tvt_input[anchor])
        anchor_surface_truth = anchor_tvt + z[anchor]
        y_visible = tvt_input[calibration_rows] + z[calibration_rows] - anchor_surface_truth
        anchor_projection = anchor_tvt - (z[eval_rows] - z[anchor])

        candidates["global"][positions] = anchor_projection + x_eval @ prior
        for window in PREFIX_WINDOWS:
            keep = _prefix_mask(md, anchor, calibration_rows, window)
            x_fit = x_visible[keep]
            y_fit = y_visible[keep]
            global_fit = x_fit @ prior
            scalar = float(
                np.dot(global_fit, y_fit) / max(np.dot(global_fit, global_fit), 1e-12)
            )
            scalar = float(np.clip(scalar, -3.0, 3.0))
            candidates[f"scalar_w{int(window)}"][positions] = (
                anchor_projection + scalar * (x_eval @ prior)
            )
            for strength in VECTOR_STRENGTHS:
                coefficient = _vector_calibration(x_fit, y_fit, prior, strength)
                name = f"vector_w{int(window)}_r{str(strength).replace('.', 'p')}"
                candidates[name][positions] = anchor_projection + x_eval @ coefficient

        shift = max(1, int(round(len(x_eval) * ROLL_FRACTION)))
        rolled[positions] = anchor_projection + np.roll(x_eval, shift, axis=0) @ prior
        target = local["truth"].to_numpy(np.float64)
        oracle_coefficient = np.linalg.lstsq(
            x_eval, target - anchor_projection, rcond=1e-6
        )[0]
        perwell_oracle[positions] = anchor_projection + x_eval @ oracle_coefficient
        well_reports.append({
            "well": well,
            "fold": held_fold,
            "visible_rows": len(visible_rows),
            "minimum_support_wells": int(np.min(support)),
            "mean_teacher_uncertainty": float(np.mean(uncertainty)),
        })
        if number % 25 == 0 or number == len(fold_wells):
            print(f"V176 fold{held_fold} {number}/{len(fold_wells)}", flush=True)

base = base_frame["base"].to_numpy(np.float64)
truth = base_frame["truth"].to_numpy(np.float64)
fold_array = base_frame["fold"].to_numpy(np.int8)
nested_candidate = np.full(len(base_frame), np.nan, np.float64)
nested_blend = np.full(len(base_frame), np.nan, np.float64)
selection_rows = []
for held_fold in range(1, 6):
    train = fold_array != held_fold
    valid = ~train
    train_scores = {
        name: float(np.sqrt(np.mean((prediction[train] - truth[train]) ** 2)))
        for name, prediction in candidates.items()
    }
    selected = min(train_scores, key=train_scores.get)
    selected_prediction = candidates[selected]
    nested_candidate[valid] = selected_prediction[valid]
    move = selected_prediction[train] - base[train]
    alpha = float(
        np.dot(truth[train] - base[train], move) / max(np.dot(move, move), 1e-12)
    )
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nested_blend[valid] = base[valid] + alpha * (
        selected_prediction[valid] - base[valid]
    )
    selection_rows.append({
        "heldout_fold": held_fold,
        "selected_recipe": selected,
        "train_rmse": train_scores[selected],
        "selected_alpha": alpha,
    })

summary_rows = [_metric(base_frame, base, "base_current")]
summary_rows += [
    _metric(base_frame, prediction, name) for name, prediction in candidates.items()
]
summary_rows += [
    _metric(base_frame, nested_candidate, "nested_selected_candidate"),
    _metric(base_frame, nested_blend, "nested_blend"),
    _metric(base_frame, rolled, "rolled_control"),
    _metric(base_frame, perwell_oracle, "distilled_perwell_oracle"),
]
summary = pd.DataFrame(summary_rows)
fold_rows = []
for fold in range(1, 6):
    mask = fold_array == fold
    local = base_frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in (
        ("base_current", base[mask]),
        ("nested_selected_candidate", nested_candidate[mask]),
        ("nested_blend", nested_blend[mask]),
    ):
        fold_rows.append({"fold": fold, **_metric(local, prediction, endpoint)})
fold_summary = pd.DataFrame(fold_rows)

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
pd.DataFrame(selection_rows).to_csv(OUT_DIR / "nested_selections.csv", index=False)
pd.DataFrame(well_reports).to_csv(OUT_DIR / "well_teacher_report.csv", index=False)
output = base_frame[["well", "fold", "row_idx", "truth", "distance"]].copy()
output["base_current"] = base.astype(np.float32)
output["nested_selected_candidate"] = nested_candidate.astype(np.float32)
output["nested_blend"] = nested_blend.astype(np.float32)
output.to_csv(OUT_DIR / "oof.csv.gz", index=False, compression="gzip")

lookup = summary.set_index("endpoint")
base_rmse = float(lookup.loc["base_current", "rmse"])
candidate_rmse = float(lookup.loc["nested_selected_candidate", "rmse"])
blend_rmse = float(lookup.loc["nested_blend", "rmse"])
rolled_rmse = float(lookup.loc["rolled_control", "rmse"])
oracle_rmse = float(lookup.loc["distilled_perwell_oracle", "rmse"])
pivot = fold_summary.pivot(index="fold", columns="endpoint", values="rmse")
fold_gains = (pivot["base_current"] - pivot["nested_blend"]).to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_visible_prefix_calibration_of_distilled_horizon_teacher",
    "legal_target_inputs": ["X", "Y", "Z", "MD", "TVT_input"],
    "train_only_labels": list(HORIZONS),
    "selection_search_count": len(recipe_names),
    "nested_selections": selection_rows,
    "values": {
        "base_rmse": base_rmse,
        "nested_candidate_rmse": candidate_rmse,
        "nested_blend_rmse": blend_rmse,
        "nested_gain": base_rmse - blend_rmse,
        "rolled_increment": rolled_rmse - candidate_rmse,
        "distilled_perwell_oracle_rmse": oracle_rmse,
        "fold_gains": fold_gains,
    },
    "checks": {
        "candidate_rmse_le_8": candidate_rmse <= 8.0,
        "nested_gain_ge_0p20": base_rmse - blend_rmse >= 0.20,
        "positive_folds_5of5": all(value > 0.0 for value in fold_gains.values()),
        "aligned_beats_rolled_ge_0p20": rolled_rmse - candidate_rmse >= 0.20,
        "oracle_rmse_le_6": oracle_rmse <= 6.0,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> replay the frozen calibration family on the current V144 endpoint"
    if verdict["strict_pass"]
    else "FAIL -> visible prefix cannot calibrate the spatial horizon teacher; close branch"
)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)

print("\nV176 MATCHED SUMMARY")
print(summary.sort_values("rmse").to_string(index=False), flush=True)
print("\nV176 FOLD SUMMARY")
print(fold_summary.to_string(index=False), flush=True)
print("\nV176 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
