"""M342-G2: nested robust dose for the G1 convex endpoint direction.

For every outer fold, four inner held-fold predictions choose a scalar
shrinkage for the G1 non-base direction. The selected shrinkage must be safe
on every inner fold before the direction is refit on all outer-training folds.
This separates endpoint proportions from deployment dose and prevents the
poorer folds from forcing every outer model to the 30% cap.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
G1_SOURCE = ROOT / "main342_fullbank_convex_stacking_g1.py"
OUT = ROOT / "oof_exports" / "main342_fullbank_nested_robust_dose_g2"
OUT.mkdir(parents=True, exist_ok=True)

SHRINKAGES = np.asarray([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0], np.float64)
INNER_MIN_GAIN = -0.005
INNER_MAX_P95_CHANGE = 0.20


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


def _load_g1_definitions():
    if not G1_SOURCE.is_file():
        raise FileNotFoundError(G1_SOURCE)
    source = G1_SOURCE.read_text(encoding="utf-8")
    marker = "V = _load_inventory()"
    if marker not in source:
        raise RuntimeError("G1 execution marker not found")
    name = "m342_g2_g1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(G1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(G1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


G = _load_g1_definitions()
V = G["_load_inventory"]()
base = V["base"]
paths = V["matrix"].astype(np.float64, copy=False)
names = list(V["names"])
base_index = int(V["base_index"])
truth = V["truth_all"].astype(np.float64, copy=False)
base_prediction = paths[base_index]
candidate_indices = [index for index in range(len(names)) if index != base_index]
candidate_names = [names[index] for index in candidate_indices]
delta = paths[candidate_indices] - base_prediction[None, :]
rolled_delta = G["_rolled_delta"](delta, base)
base_error = base_prediction - truth
fold_array = base["fold"].to_numpy(np.int8)
fit_coefficients = G["_fit_coefficients"]
metrics = G["_metrics"]


def _p95(frame, prediction):
    error = np.asarray(prediction, np.float64) - frame["tvt_true"].to_numpy(np.float64)
    local = pd.DataFrame({
        "well": frame["well"].astype(str), "e2": error * error,
    }).groupby("well", sort=False)["e2"].mean()
    return float(np.sqrt(local).quantile(0.95))


def _select_shrinkage(outer_held):
    train_folds = [fold for fold in range(1, 6) if fold != outer_held]
    inner_payload = []
    for inner_held in train_folds:
        fit_mask = np.isin(fold_array, [fold for fold in train_folds if fold != inner_held])
        valid_mask = fold_array == inner_held
        alpha, _ = fit_coefficients(delta, base_error, fit_mask)
        move = alpha @ delta[:, valid_mask]
        local = base.loc[valid_mask].reset_index(drop=True)
        base_local = base_prediction[valid_mask]
        truth_local = truth[valid_mask]
        base_rmse = float(np.sqrt(np.mean(np.square(base_local - truth_local))))
        base_p95 = _p95(local, base_local)
        inner_payload.append({
            "fold": inner_held, "rows": int(valid_mask.sum()),
            "base": base_local, "truth": truth_local, "move": move,
            "base_rmse": base_rmse, "base_p95": base_p95,
        })

    scan = []
    for shrinkage in SHRINKAGES:
        total_sse = 0.0
        total_rows = 0
        gains, p95_changes = [], []
        for payload in inner_payload:
            prediction = payload["base"] + shrinkage * payload["move"]
            error = prediction - payload["truth"]
            rmse = float(np.sqrt(np.mean(error * error)))
            local_fold = base.loc[fold_array == payload["fold"]].reset_index(drop=True)
            gains.append(payload["base_rmse"] - rmse)
            p95_changes.append(_p95(local_fold, prediction) - payload["base_p95"])
            total_sse += float(np.dot(error, error))
            total_rows += len(error)
        pooled_rmse = float(np.sqrt(total_sse / total_rows))
        safe = bool(min(gains) >= INNER_MIN_GAIN and max(p95_changes) <= INNER_MAX_P95_CHANGE)
        scan.append({
            "shrinkage": float(shrinkage), "pooled_rmse": pooled_rmse,
            "minimum_gain": float(min(gains)), "mean_gain": float(np.mean(gains)),
            "maximum_p95_change": float(max(p95_changes)), "safe": safe,
        })
    safe_rows = [row for row in scan if row["safe"]]
    selected = min(safe_rows, key=lambda row: (row["pooled_rmse"], row["shrinkage"]))
    return selected, scan


crossfit = base_prediction.copy()
crossfit_roll = base_prediction.copy()
outer_reports = []
for held in range(1, 6):
    selected, scan = _select_shrinkage(held)
    train = fold_array != held
    valid = fold_array == held
    alpha, train_objective = fit_coefficients(delta, base_error, train)
    final_alpha = selected["shrinkage"] * alpha
    crossfit[valid] += final_alpha @ delta[:, valid]
    crossfit_roll[valid] += final_alpha @ rolled_delta[:, valid]
    outer_reports.append({
        "fold": held,
        "selected_shrinkage": selected["shrinkage"],
        "selected_inner_pooled_rmse": selected["pooled_rmse"],
        "selected_inner_minimum_gain": selected["minimum_gain"],
        "selected_inner_maximum_p95_change": selected["maximum_p95_change"],
        "outer_nonbase_dose": float(final_alpha.sum()),
        "outer_base_weight": float(1.0 - final_alpha.sum()),
        "outer_train_objective_delta_mse": train_objective,
        "inner_scan": scan,
        **{f"weight__{name}": float(value) for name, value in zip(candidate_names, final_alpha)},
    })
    print("M342-G2 OUTER", {key: value for key, value in outer_reports[-1].items() if key != "inner_scan"}, flush=True)

predictions = {
    "v168": base_prediction,
    "nested_robust_convex": crossfit,
    "nested_depth_roll_control": crossfit_roll,
}
summary = pd.DataFrame([
    metrics(base, prediction, name) for name, prediction in predictions.items()
]).sort_values("rmse", ignore_index=True)

fold_rows = []
for held in range(1, 6):
    mask = fold_array == held
    local = base.loc[mask].reset_index(drop=True)
    fold_metrics = {
        name: metrics(local, prediction[mask], name)
        for name, prediction in predictions.items()
    }
    fold_rows.append({
        "fold": held,
        "selected_shrinkage": outer_reports[held - 1]["selected_shrinkage"],
        "nonbase_dose": outer_reports[held - 1]["outer_nonbase_dose"],
        "base_rmse": fold_metrics["v168"]["rmse"],
        "rmse": fold_metrics["nested_robust_convex"]["rmse"],
        "gain": fold_metrics["v168"]["rmse"] - fold_metrics["nested_robust_convex"]["rmse"],
        "far_gain": fold_metrics["v168"]["far_gt2000_rmse"] - fold_metrics["nested_robust_convex"]["far_gt2000_rmse"],
        "alignment_increment": fold_metrics["nested_depth_roll_control"]["rmse"] - fold_metrics["nested_robust_convex"]["rmse"],
        "p95_change": fold_metrics["nested_robust_convex"]["p95_well_rmse"] - fold_metrics["v168"]["p95_well_rmse"],
    })
fold_frame = pd.DataFrame(fold_rows)
lookup = summary.set_index("endpoint")
gain = float(lookup.loc["v168", "rmse"] - lookup.loc["nested_robust_convex", "rmse"])
alignment_increment = float(
    lookup.loc["nested_depth_roll_control", "rmse"] - lookup.loc["nested_robust_convex", "rmse"]
)
checks = {
    "gain_ge_0p03": gain >= 0.03,
    "positive_folds_ge_4": int((fold_frame["gain"] > 0).sum()) >= 4,
    "minimum_fold_gain_ge_minus0p01": float(fold_frame["gain"].min()) >= -0.01,
    "alignment_increment_ge_0p02": alignment_increment >= 0.02,
    "positive_alignment_folds_ge_4": int((fold_frame["alignment_increment"] > 0).sum()) >= 4,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change"].max()) <= 0.20,
    "mean_nonbase_dose_ge_0p02": float(fold_frame["nonbase_dose"].mean()) >= 0.02,
}
strict_pass = bool(all(checks.values()))

# Deployment weights are emitted only after a successful crossfit gate. Their
# shrinkage is selected by fivefold inner predictions, then refit on all rows.
deployment = None
if strict_pass:
    full_scan = []
    for shrinkage in SHRINKAGES:
        gains, p95_changes, total_sse, total_rows = [], [], 0.0, 0
        for held in range(1, 6):
            train = fold_array != held
            valid = fold_array == held
            alpha, _ = fit_coefficients(delta, base_error, train)
            prediction = base_prediction[valid] + shrinkage * (alpha @ delta[:, valid])
            local = base.loc[valid].reset_index(drop=True)
            candidate_metric = metrics(local, prediction, "candidate")
            base_metric = metrics(local, base_prediction[valid], "base")
            gains.append(base_metric["rmse"] - candidate_metric["rmse"])
            p95_changes.append(candidate_metric["p95_well_rmse"] - base_metric["p95_well_rmse"])
            error = prediction - truth[valid]
            total_sse += float(np.dot(error, error))
            total_rows += int(valid.sum())
        full_scan.append({
            "shrinkage": float(shrinkage), "pooled_rmse": float(np.sqrt(total_sse / total_rows)),
            "minimum_gain": float(min(gains)), "maximum_p95_change": float(max(p95_changes)),
            "safe": bool(min(gains) >= INNER_MIN_GAIN and max(p95_changes) <= INNER_MAX_P95_CHANGE),
        })
    chosen = min((row for row in full_scan if row["safe"]), key=lambda row: row["pooled_rmse"])
    full_alpha, _ = fit_coefficients(delta, base_error, np.ones(len(base), bool))
    full_alpha *= chosen["shrinkage"]
    deployment = {
        "selected_shrinkage": chosen["shrinkage"],
        "base_weight": float(1.0 - full_alpha.sum()),
        "nonbase_dose": float(full_alpha.sum()),
        "weights": dict(zip(candidate_names, full_alpha)),
        "selection_scan": full_scan,
    }

verdict = {
    "experiment": "main342_fullbank_nested_robust_dose_g2",
    "stage": "nested_inner_fold_robust_shrinkage_of_conservative_endpoint_stack",
    "base": names[base_index],
    "candidate_endpoints": candidate_names,
    "shrinkages": SHRINKAGES,
    "inner_constraints": {
        "minimum_fold_gain": INNER_MIN_GAIN,
        "maximum_p95_change": INNER_MAX_P95_CHANGE,
    },
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "outer_reports": outer_reports,
    "summary": summary.to_dict("records"),
    "folds": fold_rows,
    "values": {"gain": gain, "alignment_increment": alignment_increment},
    "checks": checks,
    "strict_pass": strict_pass,
    "deployment": deployment,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> package the frozen G2 endpoint weights for Kaggle lineage testing"
        if strict_pass else
        "FAIL -> no fold-robust fixed endpoint dose; close full-bank stacking"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
pd.DataFrame([{key: value for key, value in row.items() if key != "inner_scan"} for row in outer_reports]).to_csv(
    OUT / "outer_reports.csv", index=False
)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-G2 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-G2 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-G2 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
