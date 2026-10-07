"""M342-G1: fivefold conservative convex stacking of the V187 endpoint bank.

No well/block routing is used. On each outer split, non-negative coefficients
for endpoint deltas relative to V168 are fitted on the other four complete-
well folds. Their total dose is capped at 0.30. A within-well circular-depth
roll of every candidate delta is the alignment control.
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
OOF = ROOT / "oof_exports"
SOURCE = ROOT / "canonical_multiengine_block_oracle_v187.py"
OUT = OOF / "main342_fullbank_convex_stacking_g1"
OUT.mkdir(parents=True, exist_ok=True)

MAX_NONBASE_DOSE = 0.30
ROLL_ROWS = 512
OPT_TOL = 1e-10


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


def _load_inventory():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = "standalone = pd.DataFrame"
    if marker not in source:
        raise RuntimeError("V187 inventory marker not found")
    name = "m342_g1_v187_inventory"
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    required = ("base", "matrix", "names", "base_index", "truth_all")
    missing = [key for key in required if key not in module.__dict__]
    if missing:
        raise RuntimeError("Incomplete V187 inventory: " + repr(missing))
    return module.__dict__


def _fit_coefficients(delta, base_error, mask):
    local_delta = delta[:, mask].astype(np.float64, copy=False)
    local_error = base_error[mask]
    rows = float(mask.sum())
    gram = (local_delta @ local_delta.T) / rows
    cross = (local_delta @ local_error) / rows

    def objective(alpha):
        return float(alpha @ gram @ alpha + 2.0 * cross @ alpha)

    def gradient(alpha):
        return 2.0 * (gram @ alpha + cross)

    initial = np.zeros(delta.shape[0], np.float64)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, MAX_NONBASE_DOSE)] * len(initial),
        constraints=[{
            "type": "ineq",
            "fun": lambda alpha: MAX_NONBASE_DOSE - float(np.sum(alpha)),
            "jac": lambda alpha: -np.ones_like(alpha),
        }],
        options={"ftol": OPT_TOL, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise RuntimeError("Convex optimization failed: " + result.message)
    alpha = np.clip(np.asarray(result.x, np.float64), 0.0, None)
    if alpha.sum() > MAX_NONBASE_DOSE + 1e-7:
        raise RuntimeError(f"Dose constraint violated: {alpha.sum()}")
    return alpha, objective(alpha)


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, names = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    well_sse = np.bincount(codes, weights=error * error, minlength=len(names))
    well_rmse = np.sqrt(well_sse / counts)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint, "rows": int(len(frame)), "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _rolled_delta(delta, base):
    output = np.empty_like(delta)
    for _, positions in base.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        shift = min(ROLL_ROWS, max(len(positions) // 3, 1))
        output[:, positions] = np.roll(delta[:, positions], shift=shift, axis=1)
    return output


V = _load_inventory()
base = V["base"]
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
candidate_indices = [index for index in range(len(names)) if index != base_index]
candidate_names = [names[index] for index in candidate_indices]
delta = paths[candidate_indices] - base_prediction[None, :]
rolled_delta = _rolled_delta(delta, base)
base_error = base_prediction - truth
fold_array = base["fold"].to_numpy(np.int8)

crossfit = base_prediction.copy()
crossfit_rolled = base_prediction.copy()
fold_weights = []
for held in range(1, 6):
    train = fold_array != held
    valid = fold_array == held
    alpha, train_objective = _fit_coefficients(delta, base_error, train)
    crossfit[valid] += alpha @ delta[:, valid]
    crossfit_rolled[valid] += alpha @ rolled_delta[:, valid]
    fold_weights.append({
        "fold": held,
        "base_weight": float(1.0 - alpha.sum()),
        "nonbase_dose": float(alpha.sum()),
        "train_objective_delta_mse": train_objective,
        **{f"weight__{name}": float(value) for name, value in zip(candidate_names, alpha)},
    })
    print("M342-G1 WEIGHTS", fold_weights[-1], flush=True)

full_alpha, full_objective = _fit_coefficients(
    delta, base_error, np.ones(len(base), dtype=bool)
)
full_fit = base_prediction + full_alpha @ delta
predictions = {
    "v168": base_prediction,
    "fivefold_convex": crossfit,
    "fivefold_depth_roll_control": crossfit_rolled,
    "full_fit_diagnostic": full_fit,
}
summary = pd.DataFrame([
    _metrics(base, prediction, name) for name, prediction in predictions.items()
]).sort_values("rmse", ignore_index=True)

fold_rows = []
for held in range(1, 6):
    mask = fold_array == held
    local = base.loc[mask].reset_index(drop=True)
    metrics = {
        name: _metrics(local, prediction[mask], name)
        for name, prediction in predictions.items()
    }
    fold_rows.append({
        "fold": held,
        "base_rmse": metrics["v168"]["rmse"],
        "rmse": metrics["fivefold_convex"]["rmse"],
        "gain": metrics["v168"]["rmse"] - metrics["fivefold_convex"]["rmse"],
        "far_gain": metrics["v168"]["far_gt2000_rmse"] - metrics["fivefold_convex"]["far_gt2000_rmse"],
        "roll_rmse": metrics["fivefold_depth_roll_control"]["rmse"],
        "alignment_increment": metrics["fivefold_depth_roll_control"]["rmse"] - metrics["fivefold_convex"]["rmse"],
        "p95_change": metrics["fivefold_convex"]["p95_well_rmse"] - metrics["v168"]["p95_well_rmse"],
    })
fold_frame = pd.DataFrame(fold_rows)
lookup = summary.set_index("endpoint")
gain = float(lookup.loc["v168", "rmse"] - lookup.loc["fivefold_convex", "rmse"])
alignment_increment = float(
    lookup.loc["fivefold_depth_roll_control", "rmse"] - lookup.loc["fivefold_convex", "rmse"]
)
checks = {
    "gain_ge_0p05": gain >= 0.05,
    "positive_folds_ge_4": int((fold_frame["gain"] > 0).sum()) >= 4,
    "minimum_fold_gain_ge_minus0p02": float(fold_frame["gain"].min()) >= -0.02,
    "alignment_increment_ge_0p02": alignment_increment >= 0.02,
    "positive_alignment_folds_ge_4": int((fold_frame["alignment_increment"] > 0).sum()) >= 4,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change"].max()) <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_fullbank_convex_stacking_g1",
    "stage": "fivefold_complete_well_conservative_full_endpoint_bank_stacking",
    "sample_unit": "complete physical well; all official hidden rows",
    "base": names[base_index],
    "candidate_endpoints": candidate_names,
    "maximum_nonbase_dose": MAX_NONBASE_DOSE,
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "fold_weights": fold_weights,
    "full_fit": {
        "base_weight": float(1.0 - full_alpha.sum()),
        "nonbase_dose": float(full_alpha.sum()),
        "objective_delta_mse": full_objective,
        "weights": dict(zip(candidate_names, full_alpha)),
    },
    "summary": summary.to_dict("records"),
    "folds": fold_rows,
    "values": {"gain": gain, "alignment_increment": alignment_increment},
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> reduce to stable nonzero endpoints and build a Kaggle stacking bundle"
        if strict_pass else
        "FAIL -> fixed full-bank stacking is not transferable; retain V168/mainline"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
pd.DataFrame(fold_weights).to_csv(OUT / "fold_weights.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-G1 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-G1 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-G1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
