"""M342-J1: fivefold datum/slope/shape endpoint stacking.

Each active endpoint delta is decomposed within a complete well into a fitted
datum, a linear progress trend and a residual shape.  Separate conservative
weights test whether useful trajectory shape is hidden behind a poor absolute
datum.  No truth enters the decomposition.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd
from scipy.optimize import minimize


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
G1_SOURCE = ROOT / "main342_fullbank_convex_stacking_g1.py"
OUT = ROOT / "oof_exports" / "main342_datum_slope_shape_stack_j1"
OUT.mkdir(parents=True, exist_ok=True)

ACTIVE_ENDPOINTS = (
    "v144/coarse_crossfit",
    "psg/psg_tvt",
    "neighbor/fle_neighbor",
)
COMPONENT_FAMILIES = ("datum", "slope", "shape")
MAX_RAW_DOSE_PER_FAMILY = 0.30
FROZEN_SHRINKAGE = 0.20
ROLL_FRACTION = 0.25
G3_EXPECTED_RMSE = 7.7079303941645545


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_g1():
    if not G1_SOURCE.is_file():
        raise FileNotFoundError(G1_SOURCE)
    source = G1_SOURCE.read_text(encoding="utf-8")
    marker = "V = _load_inventory()"
    if marker not in source:
        raise RuntimeError("G1 execution marker not found")
    name = "m342_j1_g1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(G1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(G1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _decompose(delta, frame):
    datum = np.empty_like(delta)
    slope = np.empty_like(delta)
    shape = np.empty_like(delta)
    reconstruction_gap = 0.0
    md_since = frame["md_since"].to_numpy(np.float64)
    for positions in frame.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        local = delta[:, positions]
        progress = md_since[positions]
        progress = progress - progress[0]
        scale = max(float(progress[-1]), 1.0)
        progress = np.clip(progress / scale, 0.0, 1.0)
        design = np.column_stack([np.ones(len(progress)), progress])
        coefficients = np.linalg.lstsq(design, local.T, rcond=None)[0]
        local_datum = np.repeat(coefficients[0, :, None], len(progress), axis=1)
        local_slope = coefficients[1, :, None] * progress[None, :]
        local_shape = local - local_datum - local_slope
        datum[:, positions] = local_datum
        slope[:, positions] = local_slope
        shape[:, positions] = local_shape
        reconstruction_gap = max(
            reconstruction_gap,
            float(np.max(np.abs(local - local_datum - local_slope - local_shape))),
        )
    components = np.concatenate([datum, slope, shape], axis=0)
    names = [
        f"{family}__{endpoint}"
        for family in COMPONENT_FAMILIES
        for endpoint in ACTIVE_ENDPOINTS
    ]
    return components, names, reconstruction_gap


def _fit(features, error, mask):
    local = features[:, mask]
    rows = max(int(mask.sum()), 1)
    gram = local @ local.T / rows
    cross = local @ error[mask] / rows

    def objective(alpha):
        return float(alpha @ gram @ alpha + 2.0 * cross @ alpha)

    def gradient(alpha):
        return 2.0 * (gram @ alpha + cross)

    constraints = []
    width = len(ACTIVE_ENDPOINTS)
    for start in range(0, features.shape[0], width):
        selector = np.zeros(features.shape[0], np.float64)
        selector[start:start + width] = 1.0
        constraints.append({
            "type": "ineq",
            "fun": lambda alpha, selector=selector: (
                MAX_RAW_DOSE_PER_FAMILY - float(selector @ alpha)
            ),
            "jac": lambda alpha, selector=selector: -selector,
        })
    result = minimize(
        objective,
        np.zeros(features.shape[0], np.float64),
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, MAX_RAW_DOSE_PER_FAMILY)] * features.shape[0],
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return FROZEN_SHRINKAGE * np.clip(np.asarray(result.x, np.float64), 0.0, None)


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, wells = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    well_sse = np.bincount(codes, weights=np.square(error), minlength=len(wells))
    well_rmse = np.sqrt(well_sse / counts)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(len(wells)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


G = _load_g1()
V = G["_load_inventory"]()
base = V["base"].reset_index(drop=True)
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
base_error = base_prediction - truth
fold = base["fold"].to_numpy(np.int8)

missing = [name for name in ACTIVE_ENDPOINTS if name not in names]
if missing:
    raise RuntimeError("Missing active endpoints: " + repr(missing))
active_indices = [names.index(name) for name in ACTIVE_ENDPOINTS]
active_delta = paths[active_indices] - base_prediction[None, :]
components, component_names, reconstruction_gap = _decompose(active_delta, base)
if not np.isfinite(components).all():
    raise RuntimeError("Non-finite datum/slope/shape components")
if reconstruction_gap > 1e-9:
    raise RuntimeError(f"Component reconstruction gap: {reconstruction_gap}")

# Exact G3 lineage control uses the complete endpoint menu, unchanged.
all_indices = [index for index in range(len(names)) if index != base_index]
all_delta = paths[all_indices] - base_prediction[None, :]
g3 = base_prediction.copy()
j1 = base_prediction.copy()
roll = base_prediction.copy()
rolled_components = components.copy()
width = len(ACTIVE_ENDPOINTS)
for positions in base.groupby("well", sort=False).indices.values():
    positions = np.asarray(positions, np.int64)
    shift = max(int(round(len(positions) * ROLL_FRACTION)), 1)
    # Datum is constant within a well. Roll only slope and residual shape.
    rolled_components[width:, positions] = np.roll(
        components[width:, positions], shift=shift, axis=1
    )

fold_reports = []
for held in range(1, 6):
    train = fold != held
    valid = fold == held
    alpha_j1 = _fit(components, base_error, train)
    raw_g3, _ = G["_fit_coefficients"](all_delta, base_error, train)
    alpha_g3 = FROZEN_SHRINKAGE * raw_g3
    j1[valid] += alpha_j1 @ components[:, valid]
    roll[valid] += alpha_j1 @ rolled_components[:, valid]
    g3[valid] += alpha_g3 @ all_delta[:, valid]

    local = base.loc[valid].reset_index(drop=True)
    base_metric = _metrics(local, base_prediction[valid], "base")
    g3_metric = _metrics(local, g3[valid], "g3")
    j1_metric = _metrics(local, j1[valid], "j1")
    roll_metric = _metrics(local, roll[valid], "roll")
    family_doses = {
        family: float(alpha_j1[index * width:(index + 1) * width].sum())
        for index, family in enumerate(COMPONENT_FAMILIES)
    }
    fold_reports.append({
        "fold": held,
        "base_rmse": base_metric["rmse"],
        "g3_rmse": g3_metric["rmse"],
        "rmse": j1_metric["rmse"],
        "gain_vs_base": base_metric["rmse"] - j1_metric["rmse"],
        "increment_vs_g3": g3_metric["rmse"] - j1_metric["rmse"],
        "far_increment_vs_g3": g3_metric["far_gt2000_rmse"] - j1_metric["far_gt2000_rmse"],
        "alignment_increment": roll_metric["rmse"] - j1_metric["rmse"],
        "p95_change_vs_g3": j1_metric["p95_well_rmse"] - g3_metric["p95_well_rmse"],
        **{f"dose_{key}": value for key, value in family_doses.items()},
        **{f"weight__{name}": float(value) for name, value in zip(component_names, alpha_j1)},
    })
    print("M342-J1 FOLD", _clean(fold_reports[-1]), flush=True)

full_alpha = _fit(components, base_error, np.ones(len(base), dtype=bool))
full_fit = base_prediction + full_alpha @ components
summary = pd.DataFrame([
    _metrics(base, base_prediction, "v168"),
    _metrics(base, g3, "g3_reproduction"),
    _metrics(base, j1, "datum_slope_shape_crossfit"),
    _metrics(base, roll, "within_well_roll_control"),
    _metrics(base, full_fit, "datum_slope_shape_full_fit_diagnostic"),
]).sort_values("rmse", ignore_index=True)
fold_frame = pd.DataFrame(fold_reports)
lookup = summary.set_index("endpoint")
g3_rmse = float(lookup.loc["g3_reproduction", "rmse"])
j1_rmse = float(lookup.loc["datum_slope_shape_crossfit", "rmse"])
increment = g3_rmse - j1_rmse
alignment_increment = float(lookup.loc["within_well_roll_control", "rmse"] - j1_rmse)
checks = {
    "component_reconstruction_gap_le_1e_9": reconstruction_gap <= 1e-9,
    "g3_reproduction_abs_gap_le_1e_6": abs(g3_rmse - G3_EXPECTED_RMSE) <= 1e-6,
    "increment_vs_g3_ge_0p015": increment >= 0.015,
    "positive_increment_folds_ge_4": int((fold_frame["increment_vs_g3"] > 0).sum()) >= 4,
    "minimum_fold_increment_ge_minus0p01": float(fold_frame["increment_vs_g3"].min()) >= -0.01,
    "minimum_far_increment_ge_minus0p01": float(fold_frame["far_increment_vs_g3"].min()) >= -0.01,
    "alignment_increment_ge_0p01": alignment_increment >= 0.01,
    "positive_alignment_folds_5of5": bool((fold_frame["alignment_increment"] > 0).all()),
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change_vs_g3"].max()) <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_datum_slope_shape_stack_j1",
    "stage": "fivefold_truth_free_endpoint_datum_slope_shape_gateway",
    "sample_unit": "complete physical well; all official hidden rows",
    "decomposition_inputs": "endpoint predictions and md_since only; no truth",
    "base": names[base_index],
    "active_endpoints": ACTIVE_ENDPOINTS,
    "component_families": COMPONENT_FAMILIES,
    "frozen_shrinkage": FROZEN_SHRINKAGE,
    "maximum_dose_per_family": MAX_RAW_DOSE_PER_FAMILY * FROZEN_SHRINKAGE,
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "component_reconstruction_max_abs_gap": reconstruction_gap,
    "summary": summary.to_dict("records"),
    "folds": fold_reports,
    "full_fit_weights": dict(zip(component_names, full_alpha)),
    "values": {"increment_vs_g3": increment, "alignment_increment": alignment_increment},
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the nine coefficients and audit exact Kaggle lineage"
        if strict_pass else
        "FAIL -> candidate datum error does not hide transferable slope/shape value; close J1"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-J1 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-J1 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-J1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
