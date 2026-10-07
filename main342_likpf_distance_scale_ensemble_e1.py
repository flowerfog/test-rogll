"""M342-E1: distance-stratified deterministic likPF scale ensemble.

This is the final attribution test for likelihood-scale calibration. Four
fixed md-since strata receive independently fitted convex scale weights.
Weights are fitted on Fold 5 and applied to Fold 4, then vice versa. Fold 3 is
opened only when both directions beat a donor-fitted global convex ensemble.
Fold 1 and Fold 2 are never loaded.
"""

from pathlib import Path
import itertools
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
FEATURE_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "features"
DONOR_FEATURE_DIR = (
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "features"
)
V168_PATH = OOF_ROOT / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
OUT_DIR = OOF_ROOT / "main342_likpf_distance_scale_ensemble_e1"

SCALES = (
    "likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12",
)
EDGES_FT = np.asarray([500.0, 1000.0, 2000.0], np.float64)
WEIGHT_STEP = 0.05
DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3


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


def _sample_path(well):
    primary = FEATURE_DIR / f"{well}__official__sample.pkl"
    if primary.is_file():
        return primary
    fallback = DONOR_FEATURE_DIR / f"{well}__official__sample.pkl"
    return fallback if fallback.is_file() else primary


def _weight_grid():
    units = int(round(1.0 / WEIGHT_STEP))
    rows = []
    for values in itertools.product(range(units + 1), repeat=len(SCALES) - 1):
        used = sum(values)
        if used <= units:
            rows.append((*values, units - used))
    return np.asarray(rows, np.float64) / units


def _load_fold(wells, fold):
    parts = []
    for number, well in enumerate(wells, 1):
        path = _sample_path(well)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path)
        required = {"last_known_tvt", "target", "md_since", *SCALES}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"{well}: missing {missing}")
        last = frame["last_known_tvt"].to_numpy(np.float64)
        prediction = frame[list(SCALES)].to_numpy(np.float64).copy()
        prediction = np.where(np.isfinite(prediction), prediction, last[:, None])
        parts.append(pd.DataFrame({
            "well": str(well),
            "fold": int(fold),
            "truth": last + frame["target"].to_numpy(np.float64),
            "md_since": frame["md_since"].to_numpy(np.float64),
            **{name: prediction[:, index] for index, name in enumerate(SCALES)},
        }))
        if number % 50 == 0 or number == len(wells):
            print(f"M342-E1 load fold{fold} {number}/{len(wells)}", flush=True)
    output = pd.concat(parts, ignore_index=True)
    output["stratum"] = np.searchsorted(
        EDGES_FT, output["md_since"].to_numpy(np.float64), side="right"
    ).astype(np.int8)
    return output


def _fit_one(frame, grid):
    prediction = frame[list(SCALES)].to_numpy(np.float64)
    truth = frame["truth"].to_numpy(np.float64)
    gram = prediction.T @ prediction
    cross = prediction.T @ truth
    truth_ss = float(np.dot(truth, truth))
    sse = np.einsum("bi,ij,bj->b", grid, gram, grid) - 2.0 * (grid @ cross) + truth_ss
    best = int(np.argmin(sse))
    return grid[best].copy(), float(np.sqrt(max(sse[best], 0.0) / len(frame)))


def _fit_strata(frame, grid):
    weights = np.empty((len(EDGES_FT) + 1, len(SCALES)), np.float64)
    fit_rmse = []
    for stratum in range(len(weights)):
        local = frame.loc[frame["stratum"].eq(stratum)]
        if len(local) < 1000:
            raise RuntimeError(f"Only {len(local)} donor rows in stratum {stratum}")
        weights[stratum], score = _fit_one(local, grid)
        fit_rmse.append(score)
    return weights, fit_rmse


def _predict(frame, weights):
    matrix = frame[list(SCALES)].to_numpy(np.float64)
    row_weights = weights[frame["stratum"].to_numpy(np.int64)]
    return np.einsum("ij,ij->i", matrix, row_weights)


def _metrics(frame, prediction):
    truth = frame["truth"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    local = pd.DataFrame({
        "well": frame["well"].astype(str), "e2": np.square(error),
    }).groupby("well", sort=False)["e2"].mean()
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "far_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
        "p95_well_rmse": float(np.sqrt(local).quantile(0.95)),
    }


def _evaluate(frame, stratified_weights, global_weights):
    stratified = _predict(frame, stratified_weights)
    global_prediction = frame[list(SCALES)].to_numpy(np.float64) @ global_weights
    equal_prediction = frame[list(SCALES)].to_numpy(np.float64).mean(axis=1)
    stratified_metrics = _metrics(frame, stratified)
    global_metrics = _metrics(frame, global_prediction)
    equal_metrics = _metrics(frame, equal_prediction)
    truth = frame["truth"].to_numpy(np.float64)
    local = pd.DataFrame({
        "well": frame["well"].astype(str),
        "stratified": np.square(stratified - truth),
        "global": np.square(global_prediction - truth),
    }).groupby("well", sort=False).mean()
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "global_rmse": global_metrics["rmse"],
        "equal_rmse": equal_metrics["rmse"],
        "rmse": stratified_metrics["rmse"],
        "gain_vs_global": global_metrics["rmse"] - stratified_metrics["rmse"],
        "gain_vs_equal": equal_metrics["rmse"] - stratified_metrics["rmse"],
        "far_gain_vs_global": global_metrics["far_rmse"] - stratified_metrics["far_rmse"],
        "positive_well_fraction_vs_global": float(
            np.mean(local["stratified"] < local["global"])
        ),
        "p95_change_vs_global": (
            stratified_metrics["p95_well_rmse"] - global_metrics["p95_well_rmse"]
        ),
    }


started = time.time()
for required in (FEATURE_DIR, V168_PATH):
    if not required.exists():
        raise FileNotFoundError(required)
OUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = pd.read_parquet(V168_PATH, columns=["well", "fold"])
metadata["well"] = metadata["well"].astype(str)
fold_map = metadata.groupby("well")["fold"].first().astype(int)
coverage = fold_map.value_counts().sort_index().astype(int).to_dict()
if any(coverage.get(fold, 0) < 145 for fold in (3, 4, 5)):
    raise RuntimeError(f"Insufficient E1 coverage: {coverage}")

frames = {
    fold: _load_fold(sorted(fold_map[fold_map.eq(fold)].index), fold)
    for fold in (3, 4, 5)
}
grid = _weight_grid()
discovery = []
discovery_weights = {}
for held_fold in DISCOVERY_FOLDS:
    donor_fold = 5 if held_fold == 4 else 4
    stratified_weights, stratum_fit_rmse = _fit_strata(frames[donor_fold], grid)
    global_weights, global_fit_rmse = _fit_one(frames[donor_fold], grid)
    report = _evaluate(frames[held_fold], stratified_weights, global_weights)
    discovery_weights[held_fold] = stratified_weights
    discovery.append({
        "held_fold": held_fold,
        "donor_fold": donor_fold,
        "stratum_fit_rmse": stratum_fit_rmse,
        "global_fit_rmse": global_fit_rmse,
        "stratum_weights": stratified_weights,
        "global_weights": global_weights,
        **report,
    })

weight_l1_mean = float(np.mean(np.abs(discovery_weights[4] - discovery_weights[5])))
discovery_pass = bool(
    min(row["gain_vs_global"] for row in discovery) >= 0.03
    and min(row["gain_vs_equal"] for row in discovery) >= 0.0
    and min(row["far_gain_vs_global"] for row in discovery) >= 0.03
    and max(row["p95_change_vs_global"] for row in discovery) <= 0.50
    and weight_l1_mean <= 0.25
)

confirmation = None
combined_weights = None
if discovery_pass:
    donor = pd.concat([frames[4], frames[5]], ignore_index=True)
    combined_weights, combined_fit_rmse = _fit_strata(donor, grid)
    combined_global, combined_global_fit_rmse = _fit_one(donor, grid)
    confirmation = {
        "fold": CONFIRMATION_FOLD,
        "stratum_fit_rmse": combined_fit_rmse,
        "global_fit_rmse": combined_global_fit_rmse,
        "stratum_weights": combined_weights,
        "global_weights": combined_global,
        **_evaluate(frames[CONFIRMATION_FOLD], combined_weights, combined_global),
    }

confirmation_pass = bool(
    confirmation is not None
    and confirmation["gain_vs_global"] >= 0.03
    and confirmation["gain_vs_equal"] >= 0.0
    and confirmation["far_gain_vs_global"] >= 0.03
    and confirmation["p95_change_vs_global"] <= 0.50
)
strict_pass = bool(discovery_pass and confirmation_pass)
verdict = {
    "experiment": "main342_likpf_distance_scale_ensemble_e1",
    "stage": "crossfold_sampled_distance_stratified_likpf_scale_ensemble",
    "candidate_scales": list(SCALES),
    "distance_edges_ft": EDGES_FT,
    "weight_step": WEIGHT_STEP,
    "configurations_per_stratum": int(len(grid)),
    "coverage": coverage,
    "fold1_touched": False,
    "fold2_touched": False,
    "discovery": discovery,
    "discovery_weight_mean_abs_difference": weight_l1_mean,
    "discovery_pass": discovery_pass,
    "combined_weights": combined_weights,
    "confirmation": confirmation,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> reconstruct the frozen distance-scale endpoint full-row and test V168 blend"
        if strict_pass else
        "FAIL -> close likelihood-scale calibration; scale disagreement is not transferable"
    ),
}

rows = []
for item in discovery:
    rows.append({key: value for key, value in item.items() if not isinstance(value, np.ndarray)})
pd.DataFrame(rows).to_csv(OUT_DIR / "discovery.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("M342-E1 DISCOVERY")
print(pd.DataFrame(rows).to_string(index=False))
print("\nM342-E1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
