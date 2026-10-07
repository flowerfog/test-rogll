"""V33-centred tail-only semi-global residual alignment.

Run this as one AutoDL notebook cell after uploading this file to
``/root/autodl-tmp``.  The experiment consumes only the canonical V33 OOF
endpoint and original competition logs.  It does not use the unrelated V39/V40
branches.

Protocol
--------
* The V33 ``current_dose_0p20`` path is frozen as the centre.
* Rows at <= 1000 ft from the prediction start are bitwise unchanged.
* Tail corrections are decoded on 128-ft nodes from a robust horizontal-GR /
  typewell-GR cost map.
* The transition combines a strain penalty with a fixed-cost sparse jump, so a
  genuine structural break can beat accumulated smooth drift.
* Fold 3 selects one pre-registered recipe.  Fold 2 confirms it and Fold 1 is a
  final retrospective check.  All three folds have already been observed, so
  this is a mechanism/robustness audit, not fresh CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_tail_sgm_residual_v34"
PARENT_EXPERIMENT = "frontier_6p377_cp8_neighbor_dose_replay_v33"
PARENT_COLUMN = "current_dose_0p20"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
PARENT_PATH = OOF_ROOT / PARENT_EXPERIMENT / "cp8_neighbor_replay_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
START_FT = 1000.0
TRANSITION_FT = 250.0
NODE_SPACING_FT = 128.0
WINDOW_HALF_FT = 96.0
MAX_WINDOW_ROWS = 49
MIN_WINDOW_ROWS = 8
OFFSETS_FT = np.arange(-24.0, 24.01, 2.0, dtype=np.float64)
ZERO_STATE = int(np.argmin(np.abs(OFFSETS_FT)))
SEED = 20260727

# Small, pre-registered family.  Selection is only on Fold 3.
RECIPES = (
    {"id": "e050_s025_j2_p002", "temperature": 0.50, "smooth": 0.25, "jump": 2.0, "centre": 0.02},
    {"id": "e050_s100_j3_p002", "temperature": 0.50, "smooth": 1.00, "jump": 3.0, "centre": 0.02},
    {"id": "e100_s025_j2_p002", "temperature": 1.00, "smooth": 0.25, "jump": 2.0, "centre": 0.02},
    {"id": "e100_s100_j3_p002", "temperature": 1.00, "smooth": 1.00, "jump": 3.0, "centre": 0.02},
    {"id": "e100_s100_j6_p010", "temperature": 1.00, "smooth": 1.00, "jump": 6.0, "centre": 0.10},
    {"id": "e200_s025_j2_p002", "temperature": 2.00, "smooth": 0.25, "jump": 2.0, "centre": 0.02},
    {"id": "e200_s100_j3_p002", "temperature": 2.00, "smooth": 1.00, "jump": 3.0, "centre": 0.02},
    {"id": "e200_s400_j6_p010", "temperature": 2.00, "smooth": 4.00, "jump": 6.0, "centre": 0.10},
)
DOSES = (0.0, 0.10, 0.20, 0.35, 0.50)
started = time.time()


def _find_data_root() -> Path:
    candidates = (
        ROOT / "rogii-wellbore-geology-prediction",
        ROOT,
        Path.cwd(),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    )
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate original competition train directory")


DATA_ROOT = _find_data_root()


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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _smoothstep(distance):
    t = np.clip((np.asarray(distance, np.float64) - START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _deduplicate(axis, values):
    table = pd.DataFrame({"axis": axis, "value": values}).replace([np.inf, -np.inf], np.nan)
    table = table.dropna().groupby("axis", as_index=False)["value"].median().sort_values("axis")
    if len(table) < 32:
        raise RuntimeError("typewell has too few finite unique TVT rows")
    return table["axis"].to_numpy(np.float64), table["value"].to_numpy(np.float64)


def _robust_scale(values, floor=5.0):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = float(np.median(values))
    return float(max(1.4826 * np.median(np.abs(values - centre)), floor))


def _robust_affine(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, np.float64)[valid]
    y = np.asarray(y, np.float64)[valid]
    if len(x) < 24 or np.std(x) < 1e-6:
        shift = float(np.median(y - x)) if len(x) else 0.0
        return 1.0, shift, _robust_scale(y - x - shift)
    design = np.column_stack((x, np.ones(len(x))))
    weight = np.ones(len(x), np.float64)
    beta = np.array([1.0, np.median(y - x)], np.float64)
    for _ in range(8):
        lhs = design.T @ (weight[:, None] * design) + np.diag((1e-3, 1e-6))
        rhs = design.T @ (weight * y)
        beta = np.linalg.solve(lhs, rhs)
        beta[0] = np.clip(beta[0], 0.50, 1.50)
        residual = y - design @ beta
        scale = _robust_scale(residual, 3.0)
        ratio = np.abs(residual) / max(1.5 * scale, 1e-6)
        weight = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1e-6))
    residual = y - design @ beta
    return float(beta[0]), float(beta[1]), _robust_scale(residual, 5.0)


@dataclass
class WellVolume:
    well: str
    indices: np.ndarray
    distance: np.ndarray
    nodes: np.ndarray
    costs: np.ndarray
    usable: np.ndarray
    prefix_gain: float
    prefix_bias: float
    gamma: float


def _load_volume(well, indices, local, baseline, shifted_gr=False):
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=lambda c: c in {"MD", "GR", "TVT_input"})
    horizontal = horizontal.apply(pd.to_numeric, errors="coerce")
    typewell = pd.read_csv(typewell_path, usecols=lambda c: c in {"TVT", "GR"})
    typewell = typewell.apply(pd.to_numeric, errors="coerce")

    row_idx = local["row_idx"].to_numpy(np.int64)
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise RuntimeError(f"{well}: OOF row index outside horizontal file")
    md_all = horizontal["MD"].to_numpy(np.float64)
    gr_all = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    raw_distance = md_all[row_idx] - md_all[row_idx[0]] + float(distance[0])
    gap = np.abs(raw_distance - distance)
    if not np.isfinite(gap).all() or float(np.max(gap)) > 0.1:
        raise RuntimeError(f"{well}: OOF/raw MD mismatch {np.nanmax(gap)}")

    tw_tvt, tw_gr_raw = _deduplicate(
        typewell["TVT"].to_numpy(np.float64), typewell["GR"].to_numpy(np.float64)
    )
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(gr_all))
    expected = np.interp(tvt_input[known], tw_tvt, tw_gr_raw, left=np.nan, right=np.nan)
    gain, bias, gamma = _robust_affine(expected, gr_all[known])
    tw_gr = gain * tw_gr_raw + bias

    eval_gr = gr_all[row_idx].copy()
    if shifted_gr:
        finite_values = eval_gr[np.isfinite(eval_gr)]
        fill = float(np.median(finite_values)) if len(finite_values) else 0.0
        eval_gr = np.roll(np.where(np.isfinite(eval_gr), eval_gr, fill), max(1, len(eval_gr) // 3))

    active_rows = np.flatnonzero(distance >= START_FT)
    if not len(active_rows):
        raise RuntimeError(f"{well}: no rows beyond {START_FT:g} ft")
    targets = np.arange(START_FT, float(distance[-1]) + 0.5, NODE_SPACING_FT)
    nodes = np.searchsorted(distance, targets, side="left")
    nodes = np.clip(nodes, active_rows[0], len(distance) - 1)
    nodes = np.unique(np.concatenate(([active_rows[0]], nodes, [len(distance) - 1]))).astype(np.int64)

    costs = np.zeros((len(nodes), len(OFFSETS_FT)), np.float64)
    usable = np.zeros(len(nodes), bool)
    for node_number, node in enumerate(nodes):
        if node_number == 0:
            costs[node_number] = 1e4
            costs[node_number, ZERO_STATE] = 0.0
            usable[node_number] = True
            continue
        rows = np.flatnonzero(np.abs(distance - distance[node]) <= WINDOW_HALF_FT)
        rows = rows[np.isfinite(eval_gr[rows])]
        if len(rows) > MAX_WINDOW_ROWS:
            take = np.linspace(0, len(rows) - 1, MAX_WINDOW_ROWS).round().astype(int)
            rows = rows[take]
        if len(rows) < MIN_WINDOW_ROWS:
            costs[node_number] = 0.0
            continue
        candidate_tvt = baseline[rows, None] + OFFSETS_FT[None, :]
        expected = np.interp(
            candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
        ).reshape(candidate_tvt.shape)
        observed = eval_gr[rows, None]
        valid = np.isfinite(expected)
        coverage = valid.mean(axis=0)
        raw_residual = (observed - expected) / max(gamma, 1e-6)
        raw_cost = np.nanmean(np.where(valid, np.log1p(np.minimum(raw_residual**2, 1e4)), np.nan), axis=0)

        obs_centre = np.nanmedian(observed)
        obs_scale = _robust_scale(observed[:, 0] - obs_centre, 3.0)
        exp_centre = np.nanmedian(expected, axis=0, keepdims=True)
        exp_mad = np.nanmedian(np.abs(expected - exp_centre), axis=0)
        exp_scale = np.maximum(1.4826 * exp_mad, 3.0)
        shape_residual = (observed - obs_centre) / obs_scale - (expected - exp_centre) / exp_scale
        shape_cost = np.nanmean(
            np.where(valid, np.log1p(np.minimum(shape_residual**2, 1e4)), np.nan), axis=0
        )
        combined = 0.65 * raw_cost + 0.35 * shape_cost + 4.0 * np.maximum(0.85 - coverage, 0.0)
        finite = np.isfinite(combined)
        if not np.any(finite):
            costs[node_number] = 0.0
            continue
        worst = float(np.max(combined[finite]) + 10.0)
        combined[~finite] = worst
        costs[node_number] = combined - np.min(combined)
        usable[node_number] = True
    return WellVolume(
        well=str(well), indices=np.asarray(indices, np.int64), distance=distance,
        nodes=nodes, costs=costs, usable=usable, prefix_gain=gain,
        prefix_bias=bias, gamma=gamma,
    )


def _decode(volume, recipe, control="full"):
    costs = volume.costs.copy()
    if control == "state_shuffle":
        rng = np.random.default_rng(SEED + sum(map(ord, volume.well)))
        for node in range(1, len(costs)):
            costs[node] = costs[node, rng.permutation(len(OFFSETS_FT))]
    elif control == "transition_only":
        costs[1:] = 0.0
    elif control != "full":
        raise ValueError(control)

    temperature = float(recipe["temperature"])
    smooth = float(recipe["smooth"])
    jump = float(recipe["jump"])
    centre = float(recipe["centre"])
    states = len(OFFSETS_FT)
    differences = np.abs(OFFSETS_FT[:, None] - OFFSETS_FT[None, :])
    strain_ft = 4.0
    transition = (
        smooth * (np.minimum(differences, strain_ft) / strain_ft) ** 2
        + jump * (differences > strain_ft)
        + 0.02 * differences
    )
    dp = np.full(states, np.inf, np.float64)
    dp[ZERO_STATE] = 0.0
    back = np.full((len(volume.nodes), states), -1, np.int16)
    for node in range(1, len(volume.nodes)):
        unary = costs[node] / temperature + centre * (OFFSETS_FT / 24.0) ** 2
        candidate = dp[:, None] + transition
        predecessor = np.argmin(candidate, axis=0)
        dp = candidate[predecessor, np.arange(states)] + unary
        back[node] = predecessor.astype(np.int16)
    path = np.empty(len(volume.nodes), np.int16)
    path[-1] = int(np.argmin(dp))
    for node in range(len(volume.nodes) - 1, 0, -1):
        path[node - 1] = back[node, path[node]]
    path[0] = ZERO_STATE
    return OFFSETS_FT[path]


def _row_correction(volume, node_offsets):
    correction = np.interp(
        volume.distance,
        volume.distance[volume.nodes],
        np.asarray(node_offsets, np.float64),
        left=0.0,
        right=float(node_offsets[-1]),
    )
    correction *= _smoothstep(volume.distance)
    correction[volume.distance <= START_FT] = 0.0
    return correction


def _metrics(local, prediction, endpoint):
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    well_mse = pd.Series(error**2).groupby(local["well"].to_numpy(), sort=False).mean()
    output = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
        "max_well_rmse": float(np.max(np.sqrt(well_mse))),
    }
    bands = {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in bands.items():
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
    return output


def _evaluate(frame, baseline, correction, fold, dose, endpoint):
    mask = frame["fold"].to_numpy(np.int64) == int(fold)
    prediction = baseline[mask] + float(dose) * correction[mask]
    return _metrics(frame.loc[mask].reset_index(drop=True), prediction, endpoint)


if not PARENT_PATH.is_file():
    raise FileNotFoundError(
        f"Missing canonical V33 OOF: {PARENT_PATH}. Run V33 first; do not substitute V39/V40."
    )

frame = pd.read_parquet(PARENT_PATH)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", PARENT_COLUMN}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V33 OOF missing columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate V33 OOF ids")
if sorted(frame["fold"].unique().tolist()) != [1, 2, 3]:
    raise RuntimeError(f"Expected V33 folds [1,2,3], got {sorted(frame['fold'].unique())}")
baseline = frame[PARENT_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
if not np.isfinite(baseline).all() or not np.isfinite(truth).all():
    raise RuntimeError("Non-finite V33 baseline/truth")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}
print(
    "V34 LINEAGE GUARD",
    {
        "parent_experiment": PARENT_EXPERIMENT,
        "parent_artifact": str(PARENT_PATH),
        "parent_prediction_column": PARENT_COLUMN,
        "rows": len(frame),
        "wells": len(groups),
        "folds": sorted(frame["fold"].unique().tolist()),
        "start_ft": START_FT,
        "near_zone_policy": "exactly frozen",
        "data_root": str(DATA_ROOT),
    },
    flush=True,
)

volumes = {}
failures = []
for number, (well, indices) in enumerate(groups.items(), 1):
    local = frame.iloc[indices].reset_index(drop=True)
    try:
        volumes[well] = _load_volume(well, indices, local, baseline[indices])
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(groups):
        print(f"V34 cost maps {number}/{len(groups)} success={len(volumes)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
if len(volumes) < len(groups) - 2:
    raise RuntimeError(f"Too many V34 record failures: {failures[:10]}")

eligible = np.zeros(len(frame), bool)
for volume in volumes.values():
    eligible[volume.indices] = True
eligible_frame = frame.loc[eligible].reset_index(drop=True)
eligible_baseline = baseline[eligible]

selection_rows = []
corrections = {}
for recipe in RECIPES:
    correction = np.zeros(len(frame), np.float64)
    for volume in volumes.values():
        correction[volume.indices] = _row_correction(volume, _decode(volume, recipe))
    corrections[recipe["id"]] = correction
    for dose in DOSES:
        metric = _evaluate(frame, baseline, correction, SELECTION_FOLD, dose, recipe["id"])
        base_metric = _evaluate(frame, baseline, np.zeros(len(frame)), SELECTION_FOLD, 0.0, "base")
        selection_rows.append({
            "recipe": recipe["id"], "dose": float(dose),
            "rmse": metric["rmse"], "gain": base_metric["rmse"] - metric["rmse"],
            "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
            "near_change": metric["near_le1000_rmse"] - base_metric["near_le1000_rmse"],
        })
selection = pd.DataFrame(selection_rows).sort_values(["rmse", "p95_change"], kind="stable")
selection.to_csv(OUT_DIR / "fold3_recipe_scan.csv", index=False)

# Prefer a p95-safe recipe.  If none improves Fold 3, dose zero wins honestly.
safe = selection[selection["p95_change"].le(0.10) & selection["near_change"].abs().le(1e-10)]
selected_row = (safe if len(safe) else selection).iloc[0]
selected_recipe = next(item for item in RECIPES if item["id"] == selected_row["recipe"])
selected_dose = float(selected_row["dose"])
selected_correction = corrections[selected_recipe["id"]]

summary_rows = []
for fold in (SELECTION_FOLD, CONFIRMATION_FOLD, FINAL_FOLD):
    summary_rows.append(_evaluate(frame, baseline, np.zeros(len(frame)), fold, 0.0, "v33_base") | {"fold": fold})
    summary_rows.append(_evaluate(frame, baseline, selected_correction, fold, selected_dose, "v34_selected") | {"fold": fold})

# Matched controls are evaluated with the frozen selected recipe and dose.
control_corrections = {}
for control in ("state_shuffle", "transition_only"):
    correction = np.zeros(len(frame), np.float64)
    for volume in volumes.values():
        correction[volume.indices] = _row_correction(volume, _decode(volume, selected_recipe, control))
    control_corrections[control] = correction
    for fold in (CONFIRMATION_FOLD, FINAL_FOLD):
        summary_rows.append(_evaluate(frame, baseline, correction, fold, selected_dose, control) | {"fold": fold})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)

def _row(fold, endpoint):
    return summary[summary["fold"].eq(fold) & summary["endpoint"].eq(endpoint)].iloc[0]


fold_checks = {}
for fold in (SELECTION_FOLD, CONFIRMATION_FOLD, FINAL_FOLD):
    base_row = _row(fold, "v33_base")
    candidate_row = _row(fold, "v34_selected")
    fold_checks[str(fold)] = {
        "gain": float(base_row["rmse"] - candidate_row["rmse"]),
        "far_gain": float(base_row["far_gt2000_rmse"] - candidate_row["far_gt2000_rmse"]),
        "p95_change": float(candidate_row["p95_well_rmse"] - base_row["p95_well_rmse"]),
        "near_change": float(candidate_row["near_le1000_rmse"] - base_row["near_le1000_rmse"]),
    }

near_identity = bool(np.array_equal(
    baseline[frame["md_since"].to_numpy(np.float64) <= START_FT],
    (baseline + selected_dose * selected_correction)[frame["md_since"].to_numpy(np.float64) <= START_FT],
))
confirmation_gain = fold_checks[str(CONFIRMATION_FOLD)]["gain"]
final_gain = fold_checks[str(FINAL_FOLD)]["gain"]
control_retained = []
for fold in (CONFIRMATION_FOLD, FINAL_FOLD):
    base_rmse = float(_row(fold, "v33_base")["rmse"])
    for control in ("state_shuffle", "transition_only"):
        control_gain = base_rmse - float(_row(fold, control)["rmse"])
        denominator = max(fold_checks[str(fold)]["gain"], 1e-9)
        control_retained.append(control_gain / denominator)

checks = {
    "near_le1000_bitwise_identity": near_identity,
    "selection_gain_ge_0p02": fold_checks[str(SELECTION_FOLD)]["gain"] >= 0.02,
    "confirmation_gain_ge_0p02": confirmation_gain >= 0.02,
    "final_gain_ge_0p02": final_gain >= 0.02,
    "each_confirmation_far_gain_ge_0p03": min(
        fold_checks[str(CONFIRMATION_FOLD)]["far_gain"],
        fold_checks[str(FINAL_FOLD)]["far_gain"],
    ) >= 0.03,
    "each_fold_p95_change_le_0p10": max(item["p95_change"] for item in fold_checks.values()) <= 0.10,
    "controls_retain_le_30pct_gain": max(control_retained, default=np.inf) <= 0.30,
}
strict_pass = bool(all(checks.values()))

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["v33_current_dose_0p20"] = baseline.astype(np.float32)
output["v34_tail_correction"] = selected_correction.astype(np.float32)
output["v34_selected"] = (baseline + selected_dose * selected_correction).astype(np.float32)
output.to_parquet(OUT_DIR / "tail_sgm_oof.parquet", index=False, compression="zstd")

volume_diagnostics = pd.DataFrame([
    {
        "well": volume.well,
        "fold": int(frame.iloc[volume.indices[0]]["fold"]),
        "rows": len(volume.indices),
        "nodes": len(volume.nodes),
        "usable_fraction": float(np.mean(volume.usable)),
        "prefix_gain": volume.prefix_gain,
        "prefix_bias": volume.prefix_bias,
        "gamma": volume.gamma,
    }
    for volume in volumes.values()
])
volume_diagnostics.to_csv(OUT_DIR / "well_diagnostics.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "v33_centered_tail_only_robust_sgm_residual_gate",
    "lineage": {
        "parent_experiment": PARENT_EXPERIMENT,
        "parent_artifact": str(PARENT_PATH),
        "parent_prediction_column": PARENT_COLUMN,
        "unrelated_v39_v40_used": False,
    },
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "validation_status": "retrospective_only_all_available_folds_already_observed",
        "near_le1000_policy": "bitwise frozen",
        "raw_inputs": "same-well horizontal GR + same-well typewell + visible prefix",
    },
    "coverage": {
        "rows": len(frame), "wells": len(groups), "successful_wells": len(volumes),
        "failures": len(failures),
    },
    "graph": {
        "node_spacing_ft": NODE_SPACING_FT, "window_half_ft": WINDOW_HALF_FT,
        "offset_states_ft": OFFSETS_FT, "transition": "strain_plus_sparse_jump",
    },
    "selected_recipe": selected_recipe | {"dose": selected_dose},
    "fold_checks": fold_checks,
    "controls_retained_fraction": control_retained,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> port the frozen V34 residual decoder after V33 in Kaggle inference"
        if strict_pass else
        "FAIL -> dynamic residual SGM does not add stable aligned tail evidence to V33; close this paper-derived branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV34 FOLD3 TOP 20", flush=True)
print(selection.head(20).to_string(index=False), flush=True)
print("\nV34 FOLD SUMMARY", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
