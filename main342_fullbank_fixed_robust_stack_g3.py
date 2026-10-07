"""M342-G3: frozen 6% full-bank stack confirmation.

The G1 convex direction permits 30% non-base mass. G3 freezes a 0.20
shrinkage of that direction, hence at most 6% non-base mass. Endpoint ratios
are fitted on four folds and applied to the fifth. No fold-specific dose or
threshold is selected. A within-well depth roll and a well bootstrap audit
alignment and sampling stability.
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
OUT = ROOT / "oof_exports" / "main342_fullbank_fixed_robust_stack_g3"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_SHRINKAGE = 0.20
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260804


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
    name = "m342_g3_g1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(G1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(G1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _well_bootstrap(base, base_prediction, candidate_prediction):
    truth = base["tvt_true"].to_numpy(np.float64)
    base_e2 = np.square(np.asarray(base_prediction, np.float64) - truth)
    candidate_e2 = np.square(np.asarray(candidate_prediction, np.float64) - truth)
    rows = []
    for well, positions in base.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        rows.append((str(well), len(positions), base_e2[positions].sum(), candidate_e2[positions].sum()))
    table = pd.DataFrame(rows, columns=["well", "rows", "base_sse", "candidate_sse"])
    count = table["rows"].to_numpy(np.float64)
    base_sse = table["base_sse"].to_numpy(np.float64)
    candidate_sse = table["candidate_sse"].to_numpy(np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gains = np.empty(BOOTSTRAP_DRAWS, np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(table), len(table))
        gains[draw] = (
            np.sqrt(base_sse[index].sum() / count[index].sum())
            - np.sqrt(candidate_sse[index].sum() / count[index].sum())
        )
    return table, {
        "draws": BOOTSTRAP_DRAWS,
        "gain_p05": float(np.quantile(gains, 0.05)),
        "gain_median": float(np.median(gains)),
        "gain_p95": float(np.quantile(gains, 0.95)),
        "probability_gain_positive": float(np.mean(gains > 0.0)),
    }


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

crossfit = base_prediction.copy()
crossfit_roll = base_prediction.copy()
fold_weights = []
for held in range(1, 6):
    train = fold_array != held
    valid = fold_array == held
    raw_alpha, objective = fit_coefficients(delta, base_error, train)
    alpha = FROZEN_SHRINKAGE * raw_alpha
    crossfit[valid] += alpha @ delta[:, valid]
    crossfit_roll[valid] += alpha @ rolled_delta[:, valid]
    fold_weights.append({
        "fold": held,
        "base_weight": float(1.0 - alpha.sum()),
        "nonbase_dose": float(alpha.sum()),
        "raw_train_objective_delta_mse": objective,
        **{f"weight__{name}": float(value) for name, value in zip(candidate_names, alpha)},
    })

full_raw_alpha, full_objective = fit_coefficients(
    delta, base_error, np.ones(len(base), dtype=bool)
)
full_alpha = FROZEN_SHRINKAGE * full_raw_alpha
predictions = {
    "v168": base_prediction,
    "fixed_stack_w0p20": crossfit,
    "fixed_depth_roll_control": crossfit_roll,
}
summary = pd.DataFrame([
    metrics(base, prediction, name) for name, prediction in predictions.items()
]).sort_values("rmse", ignore_index=True)

fold_rows = []
for held in range(1, 6):
    mask = fold_array == held
    local = base.loc[mask].reset_index(drop=True)
    local_metrics = {
        name: metrics(local, prediction[mask], name)
        for name, prediction in predictions.items()
    }
    fold_rows.append({
        "fold": held,
        "nonbase_dose": fold_weights[held - 1]["nonbase_dose"],
        "base_rmse": local_metrics["v168"]["rmse"],
        "rmse": local_metrics["fixed_stack_w0p20"]["rmse"],
        "gain": local_metrics["v168"]["rmse"] - local_metrics["fixed_stack_w0p20"]["rmse"],
        "far_gain": local_metrics["v168"]["far_gt2000_rmse"] - local_metrics["fixed_stack_w0p20"]["far_gt2000_rmse"],
        "alignment_increment": local_metrics["fixed_depth_roll_control"]["rmse"] - local_metrics["fixed_stack_w0p20"]["rmse"],
        "p95_change": local_metrics["fixed_stack_w0p20"]["p95_well_rmse"] - local_metrics["v168"]["p95_well_rmse"],
    })
fold_frame = pd.DataFrame(fold_rows)
well_table, bootstrap = _well_bootstrap(base, base_prediction, crossfit)
lookup = summary.set_index("endpoint")
gain = float(lookup.loc["v168", "rmse"] - lookup.loc["fixed_stack_w0p20", "rmse"])
alignment_increment = float(
    lookup.loc["fixed_depth_roll_control", "rmse"] - lookup.loc["fixed_stack_w0p20", "rmse"]
)
checks = {
    "gain_ge_0p03": gain >= 0.03,
    "all_folds_positive": bool((fold_frame["gain"] > 0.0).all()),
    "all_far_folds_ge_minus0p01": float(fold_frame["far_gain"].min()) >= -0.01,
    "alignment_increment_ge_0p01": alignment_increment >= 0.01,
    "positive_alignment_folds_5of5": bool((fold_frame["alignment_increment"] > 0.0).all()),
    "maximum_p95_change_le_0p25": float(fold_frame["p95_change"].max()) <= 0.25,
    "bootstrap_probability_ge_0p95": bootstrap["probability_gain_positive"] >= 0.95,
    "bootstrap_p05_nonnegative": bootstrap["gain_p05"] >= 0.0,
}
strict_pass = bool(all(checks.values()))
deployment = {
    "frozen_shrinkage": FROZEN_SHRINKAGE,
    "base_weight": float(1.0 - full_alpha.sum()),
    "nonbase_dose": float(full_alpha.sum()),
    "weights": dict(zip(candidate_names, full_alpha)),
    "raw_full_objective_delta_mse": full_objective,
} if strict_pass else None
verdict = {
    "experiment": "main342_fullbank_fixed_robust_stack_g3",
    "stage": "frozen_six_percent_fivefold_endpoint_stack_confirmation",
    "validation_status": "retrospective_same_fivefold_recipe_confirmation; no new independent CV claim",
    "recipe_source": "G2 motivated one frozen shrinkage; G3 performs no further recipe search",
    "g3_selection_or_tuning_count": 0,
    "base": names[base_index],
    "candidate_endpoints": candidate_names,
    "frozen_shrinkage": FROZEN_SHRINKAGE,
    "maximum_nonbase_dose": float(G["MAX_NONBASE_DOSE"] * FROZEN_SHRINKAGE),
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique())},
    "fold_weights": fold_weights,
    "summary": summary.to_dict("records"),
    "folds": fold_rows,
    "bootstrap": bootstrap,
    "values": {"gain": gain, "alignment_increment": alignment_increment},
    "checks": checks,
    "strict_pass": strict_pass,
    "deployment": deployment,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build exact Kaggle lineage bundle for the three nonzero endpoints"
        if strict_pass else
        "FAIL -> even the frozen 6% stack is not stable; close full-bank stacking"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
pd.DataFrame(fold_weights).to_csv(OUT / "fold_weights.csv", index=False)
well_table.to_parquet(OUT / "well_bootstrap_payload.parquet", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-G3 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-G3 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-G3 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
