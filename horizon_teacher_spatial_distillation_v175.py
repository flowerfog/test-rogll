"""V175: distill train-only formation surfaces from legal spatial inputs.

V174 showed that six mapped formation surfaces almost exactly generate TVT, but
those columns are absent from test. V175 uses them only as training labels. For
each held-out fold, local spatial planes are fitted from other-fold wells and
queried using only target X/Y. The predicted surfaces are consumed by V174's
fold-clean physical formula.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EXPERIMENT = "horizon_teacher_spatial_distillation_v175"
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
        print(f"V175 references {number}/{EXPECTED_WELLS}", flush=True)

reference_xy = np.concatenate(reference_xy)
reference_value = np.concatenate(reference_value)
reference_fold = np.asarray(reference_fold, np.int8)
reference_code = np.asarray(reference_code, np.int16)

distilled = np.full(len(base_frame), np.nan, np.float64)
rolled = np.full(len(base_frame), np.nan, np.float64)
perwell_oracle = np.full(len(base_frame), np.nan, np.float64)
well_reports = []
for held_fold in range(1, 6):
    allowed = reference_fold != held_fold
    ref_xy = reference_xy[allowed]
    ref_value = reference_value[allowed]
    ref_code = reference_code[allowed]
    tree = cKDTree(ref_xy)
    coefficient = coefficients.loc[held_fold, coefficient_columns].to_numpy(np.float64)
    fold_wells = [well for well in well_names if fold_map[well] == held_fold]
    for number, well in enumerate(fold_wells, 1):
        local = base_frame[base_frame["well"].eq(well)]
        positions = local.index.to_numpy(np.int64)
        rows = local["row_idx"].to_numpy(np.int64)
        horizontal, _ = raw_cache[well]
        known = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(np.float64)))
        anchor = int(known[-1])
        query_rows = np.unique(
            np.r_[anchor, rows[np.unique(np.round(np.linspace(0, len(rows) - 1, QUERY_NODES)).astype(int))]]
        )
        xy_all = horizontal[["X", "Y"]].to_numpy(np.float64)
        predicted_nodes, support, uncertainty = _local_surface(
            tree, ref_xy, ref_value, ref_code, xy_all[query_rows]
        )
        predicted = np.column_stack(
            [np.interp(rows, query_rows, predicted_nodes[:, column]) for column in range(len(HORIZONS))]
        )
        anchor_surface = predicted_nodes[np.flatnonzero(query_rows == anchor)[0]]
        delta = predicted - anchor_surface[None, :]
        md = horizontal["MD"].to_numpy(np.float64)
        xy_delta = xy_all[rows] - xy_all[anchor]
        feature = np.column_stack((
            delta,
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            (md[rows] - md[anchor]) / 1000.0,
            xy_delta[:, 0] / 1000.0,
            xy_delta[:, 1] / 1000.0,
        ))
        z = horizontal["Z"].to_numpy(np.float64)
        anchor_tvt = float(horizontal["TVT_input"].iloc[anchor])
        anchor_projection = anchor_tvt - (z[rows] - z[anchor])
        distilled[positions] = anchor_projection + feature @ coefficient
        shift = max(1, int(round(len(feature) * ROLL_FRACTION)))
        rolled[positions] = anchor_projection + np.roll(feature, shift, axis=0) @ coefficient
        target = local["truth"].to_numpy(np.float64)
        oracle_coefficient = np.linalg.lstsq(feature, target - anchor_projection, rcond=1e-6)[0]
        perwell_oracle[positions] = anchor_projection + feature @ oracle_coefficient
        well_reports.append({
            "well": well,
            "fold": held_fold,
            "minimum_support_wells": int(np.min(support)),
            "mean_teacher_uncertainty": float(np.mean(uncertainty)),
        })
        if number % 25 == 0 or number == len(fold_wells):
            print(f"V175 fold{held_fold} {number}/{len(fold_wells)}", flush=True)

base = base_frame["base"].to_numpy(np.float64)
truth = base_frame["truth"].to_numpy(np.float64)
fold_array = base_frame["fold"].to_numpy(np.int8)
nested = np.full(len(base_frame), np.nan, np.float64)
blend_rows = []
for held_fold in range(1, 6):
    train = fold_array != held_fold
    valid = ~train
    move = distilled[train] - base[train]
    alpha = float(np.dot(truth[train] - base[train], move) / max(np.dot(move, move), 1e-12))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nested[valid] = base[valid] + alpha * (distilled[valid] - base[valid])
    blend_rows.append({"heldout_fold": held_fold, "selected_alpha": alpha})

summary = pd.DataFrame([
    _metric(base_frame, base, "base_current"),
    _metric(base_frame, distilled, "distilled_horizon"),
    _metric(base_frame, nested, "nested_blend"),
    _metric(base_frame, rolled, "rolled_control"),
    _metric(base_frame, perwell_oracle, "distilled_perwell_oracle"),
])
fold_rows = []
for fold in range(1, 6):
    mask = fold_array == fold
    local = base_frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in (
        ("base_current", base[mask]),
        ("distilled_horizon", distilled[mask]),
        ("nested_blend", nested[mask]),
    ):
        fold_rows.append({"fold": fold, **_metric(local, prediction, endpoint)})
fold_summary = pd.DataFrame(fold_rows)

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
pd.DataFrame(blend_rows).to_csv(OUT_DIR / "blend_weights.csv", index=False)
pd.DataFrame(well_reports).to_csv(OUT_DIR / "well_teacher_report.csv", index=False)
output = base_frame[["well", "fold", "row_idx", "truth", "distance"]].copy()
output["base_current"] = base.astype(np.float32)
output["distilled_horizon"] = distilled.astype(np.float32)
output["nested_blend"] = nested.astype(np.float32)
output.to_csv(OUT_DIR / "oof.csv.gz", index=False, compression="gzip")

lookup = summary.set_index("endpoint")
base_rmse = float(lookup.loc["base_current", "rmse"])
candidate_rmse = float(lookup.loc["distilled_horizon", "rmse"])
blend_rmse = float(lookup.loc["nested_blend", "rmse"])
rolled_rmse = float(lookup.loc["rolled_control", "rmse"])
oracle_rmse = float(lookup.loc["distilled_perwell_oracle", "rmse"])
pivot = fold_summary.pivot(index="fold", columns="endpoint", values="rmse")
fold_gains = (pivot["base_current"] - pivot["nested_blend"]).to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_train_only_horizon_teacher_distillation_gateway",
    "legal_target_inputs": ["X", "Y", "Z", "MD", "TVT_input"],
    "train_only_labels": list(HORIZONS),
    "values": {
        "base_rmse": base_rmse,
        "distilled_rmse": candidate_rmse,
        "nested_blend_rmse": blend_rmse,
        "nested_gain": base_rmse - blend_rmse,
        "rolled_increment": rolled_rmse - candidate_rmse,
        "distilled_perwell_oracle_rmse": oracle_rmse,
        "fold_gains": fold_gains,
    },
    "checks": {
        "distilled_rmse_le_8": candidate_rmse <= 8.0,
        "nested_gain_ge_0p20": base_rmse - blend_rmse >= 0.20,
        "positive_folds_5of5": all(value > 0.0 for value in fold_gains.values()),
        "aligned_beats_rolled_ge_0p20": rolled_rmse - candidate_rmse >= 0.20,
        "oracle_rmse_le_6": oracle_rmse <= 6.0,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> replace local planes with a full-resolution spatial surface model and validate on the current V144 base"
    if verdict["strict_pass"]
    else "FAIL -> train-only horizon supervision is not spatially transferable; close structural distillation"
)
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")

print("\nV175 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV175 FOLD SUMMARY")
print(fold_summary.to_string(index=False), flush=True)
print("\nV175 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
