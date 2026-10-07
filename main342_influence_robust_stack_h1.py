"""M342-H1: nested influence-robust endpoint stacking gateway.

G3 estimates endpoint covariance with one vote per row, allowing very long or
catastrophic wells to dominate the direction. H1 varies the effective
per-well leverage inside each outer training split. The leverage exponent is
selected only by inner complete-well folds and then applied once to the held
outer fold. Exponent 1.0 exactly reproduces G3 and is always available.
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
OUT = ROOT / "oof_exports" / "main342_influence_robust_stack_h1"
OUT.mkdir(parents=True, exist_ok=True)

MAX_NONBASE_DOSE = 0.30
FROZEN_SHRINKAGE = 0.20
LEVERAGE_EXPONENTS = (0.0, 0.25, 0.50, 0.75, 1.0)
G3_EXPECTED_RMSE = 7.7079303941645545
ROLL_ROWS = 512


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
    name = "m342_h1_g1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(G1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(G1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _row_weights(frame, exponent):
    counts = frame.groupby("well", sort=False)["well"].transform("size").to_numpy(np.float64)
    # Total well leverage is rows**exponent. Exponent 1 is ordinary pooled SSE;
    # exponent 0 gives every physical well equal total fitting weight.
    weight = np.power(counts, float(exponent) - 1.0)
    return weight / max(float(np.mean(weight)), 1e-12)


def _solve(gram, cross):
    def objective(alpha):
        return float(alpha @ gram @ alpha + 2.0 * cross @ alpha)

    def gradient(alpha):
        return 2.0 * (gram @ alpha + cross)

    initial = np.zeros(len(cross), np.float64)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, MAX_NONBASE_DOSE)] * len(initial),
        constraints=[{
            "type": "ineq",
            "fun": lambda alpha: MAX_NONBASE_DOSE - float(alpha.sum()),
            "jac": lambda alpha: -np.ones_like(alpha),
        }],
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(result.message)
    raw = np.clip(np.asarray(result.x, np.float64), 0.0, None)
    return FROZEN_SHRINKAGE * raw


def _sufficient_statistics(delta, error, fold, row_weights):
    statistics = {}
    for exponent, weights in row_weights.items():
        statistics[exponent] = {}
        for held in range(1, 6):
            index = np.flatnonzero(fold == held)
            local_delta = delta[:, index]
            local_weight = weights[index]
            statistics[exponent][held] = {
                "weight": float(local_weight.sum()),
                "gram": (local_delta * local_weight[None, :]) @ local_delta.T,
                "cross": (local_delta * local_weight[None, :]) @ error[index],
            }
    return statistics


def _fit_folds(statistics, exponent, train_folds):
    selected = [statistics[exponent][int(value)] for value in train_folds]
    denominator = max(float(sum(item["weight"] for item in selected)), 1.0)
    gram = sum((item["gram"] for item in selected), np.zeros_like(selected[0]["gram"])) / denominator
    cross = sum((item["cross"] for item in selected), np.zeros_like(selected[0]["cross"])) / denominator
    return _solve(gram, cross)


def _rmse(prediction, truth, mask):
    error = np.asarray(prediction, np.float64)[mask] - truth[mask]
    return float(np.sqrt(np.mean(np.square(error))))


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
candidate_indices = [index for index in range(len(names)) if index != base_index]
candidate_names = [names[index] for index in candidate_indices]
delta = paths[candidate_indices] - base_prediction[None, :]
base_error = base_prediction - truth
fold = base["fold"].to_numpy(np.int8)
row_weights = {value: _row_weights(base, value) for value in LEVERAGE_EXPONENTS}
statistics = _sufficient_statistics(delta, base_error, fold, row_weights)

rolled_delta = np.empty_like(delta)
for positions in base.groupby("well", sort=False).indices.values():
    positions = np.asarray(positions, np.int64)
    shift = min(ROLL_ROWS, max(len(positions) // 3, 1))
    rolled_delta[:, positions] = np.roll(delta[:, positions], shift=shift, axis=1)

h1 = base_prediction.copy()
g3 = base_prediction.copy()
h1_roll = base_prediction.copy()
fold_reports = []
selection_rows = []

for held in range(1, 6):
    outer_train_folds = [value for value in range(1, 6) if value != held]
    inner_g3_rmse = {}
    for inner in outer_train_folds:
        train_folds = [value for value in outer_train_folds if value != inner]
        valid = fold == inner
        alpha = _fit_folds(statistics, 1.0, train_folds)
        prediction = base_prediction[valid] + alpha @ delta[:, valid]
        inner_g3_rmse[inner] = float(
            np.sqrt(np.mean(np.square(prediction - truth[valid])))
        )
    exponent_rows = []
    for exponent in LEVERAGE_EXPONENTS:
        inner_sse = 0.0
        inner_rows = 0
        inner_gains = []
        inner_increments = []
        for inner in outer_train_folds:
            valid = fold == inner
            train_folds = [
                value for value in outer_train_folds if value != inner
            ]
            alpha = _fit_folds(statistics, exponent, train_folds)
            prediction = base_prediction[valid] + alpha @ delta[:, valid]
            baseline = base_prediction[valid]
            local_truth = truth[valid]
            inner_sse += float(np.square(prediction - local_truth).sum())
            inner_rows += int(valid.sum())
            local_rmse = float(np.sqrt(np.mean(np.square(prediction - local_truth))))
            inner_gains.append(float(
                np.sqrt(np.mean(np.square(baseline - local_truth))) - local_rmse
            ))
            inner_increments.append(float(inner_g3_rmse[inner] - local_rmse))
        exponent_rows.append({
            "held_fold": held,
            "exponent": exponent,
            "inner_rmse": float(np.sqrt(inner_sse / inner_rows)),
            "minimum_inner_gain_vs_base": float(min(inner_gains)),
            "mean_inner_gain_vs_base": float(np.mean(inner_gains)),
            "minimum_inner_increment_vs_g3": float(min(inner_increments)),
            "mean_inner_increment_vs_g3": float(np.mean(inner_increments)),
        })
    local_scan = pd.DataFrame(exponent_rows).sort_values(
        ["minimum_inner_increment_vs_g3", "inner_rmse", "exponent"],
        ascending=[False, True, False],
        kind="stable",
    )
    selected = local_scan.iloc[0]
    selected_exponent = float(selected["exponent"])
    selection_rows.extend(exponent_rows)

    valid = fold == held
    alpha_h1 = _fit_folds(statistics, selected_exponent, outer_train_folds)
    alpha_g3 = _fit_folds(statistics, 1.0, outer_train_folds)
    h1[valid] += alpha_h1 @ delta[:, valid]
    g3[valid] += alpha_g3 @ delta[:, valid]
    h1_roll[valid] += alpha_h1 @ rolled_delta[:, valid]

    local = base.loc[valid].reset_index(drop=True)
    h1_metric = _metrics(local, h1[valid], "h1")
    g3_metric = _metrics(local, g3[valid], "g3")
    roll_metric = _metrics(local, h1_roll[valid], "roll")
    base_metric = _metrics(local, base_prediction[valid], "base")
    fold_reports.append({
        "fold": held,
        "selected_exponent": selected_exponent,
        "inner_rmse": float(selected["inner_rmse"]),
        "minimum_inner_gain_vs_base": float(selected["minimum_inner_gain_vs_base"]),
        "minimum_inner_increment_vs_g3": float(selected["minimum_inner_increment_vs_g3"]),
        "base_rmse": base_metric["rmse"],
        "g3_rmse": g3_metric["rmse"],
        "rmse": h1_metric["rmse"],
        "gain_vs_base": base_metric["rmse"] - h1_metric["rmse"],
        "increment_vs_g3": g3_metric["rmse"] - h1_metric["rmse"],
        "far_increment_vs_g3": g3_metric["far_gt2000_rmse"] - h1_metric["far_gt2000_rmse"],
        "alignment_increment": roll_metric["rmse"] - h1_metric["rmse"],
        "p95_change_vs_g3": h1_metric["p95_well_rmse"] - g3_metric["p95_well_rmse"],
        "weights": dict(zip(candidate_names, alpha_h1)),
    })
    print("M342-H1 FOLD", _clean(fold_reports[-1]), flush=True)

summary = pd.DataFrame([
    _metrics(base, base_prediction, "v168"),
    _metrics(base, g3, "g3_reproduction"),
    _metrics(base, h1, "nested_influence_robust"),
    _metrics(base, h1_roll, "depth_roll_control"),
]).sort_values("rmse", ignore_index=True)
fold_frame = pd.DataFrame([{key: value for key, value in row.items() if key != "weights"} for row in fold_reports])
selection_frame = pd.DataFrame(selection_rows)
lookup = summary.set_index("endpoint")
g3_rmse = float(lookup.loc["g3_reproduction", "rmse"])
h1_rmse = float(lookup.loc["nested_influence_robust", "rmse"])
increment = g3_rmse - h1_rmse
alignment_increment = float(
    lookup.loc["depth_roll_control", "rmse"] - h1_rmse
)
checks = {
    "g3_reproduction_abs_gap_le_1e_6": abs(g3_rmse - G3_EXPECTED_RMSE) <= 1e-6,
    "increment_vs_g3_ge_0p01": increment >= 0.01,
    "positive_increment_folds_ge_4": int((fold_frame["increment_vs_g3"] > 0).sum()) >= 4,
    "minimum_fold_increment_ge_minus0p01": float(fold_frame["increment_vs_g3"].min()) >= -0.01,
    "minimum_far_increment_ge_minus0p01": float(fold_frame["far_increment_vs_g3"].min()) >= -0.01,
    "alignment_increment_ge_0p01": alignment_increment >= 0.01,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change_vs_g3"].max()) <= 0.20,
    "nontrivial_exponent_selected_ge_3_folds": int((fold_frame["selected_exponent"] < 1.0).sum()) >= 3,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_influence_robust_stack_h1",
    "stage": "fivefold_nested_physical_well_leverage_stacking_gateway",
    "sample_unit": "complete physical well; all official hidden rows",
    "base": names[base_index],
    "candidate_endpoints": candidate_names,
    "leverage_definition": "total fitting weight per well equals rows**exponent",
    "leverage_exponents": LEVERAGE_EXPONENTS,
    "frozen_shrinkage": FROZEN_SHRINKAGE,
    "maximum_nonbase_dose": MAX_NONBASE_DOSE * FROZEN_SHRINKAGE,
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "summary": summary.to_dict("records"),
    "folds": fold_reports,
    "values": {
        "increment_vs_g3": increment,
        "alignment_increment": alignment_increment,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the modal exponent and run one zero-tuning confirmation"
        if strict_pass else
        "FAIL -> well-leverage robustification adds no transferable direction; close H1"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
selection_frame.to_csv(OUT / "inner_selection.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-H1 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-H1 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-H1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
