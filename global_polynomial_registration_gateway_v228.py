"""V228: low-dimensional whole-well structural registration gateway.

Stage 1 measures the representation floor of anchored polynomial S(MD), where
S = TVT + Z.  Stage 2 builds 4096 coefficient candidates from other folds and
scores every complete candidate with the whole evaluation GR sequence.  The
truth-fitted polynomial is inserted only to audit its GR-cost rank.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF = ROOT / "oof_exports"
FOLD_PATH = OOF / "likpf_residual_gbdt_full5_v215/well_metadata.csv"
OUT = OOF / "global_polynomial_registration_gateway_v228"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 228
DEGREES = (1, 2, 3, 4, 5, 6)
SEARCH_DEGREE = 4
CANDIDATES = 4096
SEARCH_WELLS = 48
COST_POINTS = 256
BATCH = 256
TOPK = (1, 8, 32, 64)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _robust_affine(reference, observed):
    x = np.asarray(reference, float)
    y = np.asarray(observed, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 24:
        return 1.0, 0.0
    for _ in range(4):
        design = np.column_stack((x[keep], np.ones(keep.sum())))
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = np.nanmedian(residual[keep])
        scale = 1.4826 * np.nanmedian(np.abs(residual[keep] - centre)) + 1e-6
        updated = np.isfinite(x) & np.isfinite(y) & (np.abs(residual - centre) <= 3.5 * scale)
        if updated.sum() < 24 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(slope), float(intercept)


def _design(progress, degree):
    progress = np.asarray(progress, float)
    return np.column_stack([progress ** power for power in range(1, degree + 1)])


def _fit_anchored(progress, surface_delta, degree):
    matrix = _design(progress, degree)
    coefficient = np.linalg.lstsq(matrix, surface_delta, rcond=None)[0]
    for _ in range(4):
        residual = surface_delta - matrix @ coefficient
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        coefficient = np.linalg.lstsq(matrix * weight[:, None], surface_delta * weight, rcond=None)[0]
    return coefficient


def _well_hash(well):
    return int(hashlib.sha256(str(well).encode()).hexdigest()[:12], 16)


for required in (DATA_ROOT / "train", FOLD_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
if len(folds) != 773:
    raise RuntimeError(f"Expected 773 wells, found {len(folds)}")


def _load(well):
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT", "TVT_input"],
    )
    typewell = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__typewell.csv",
        usecols=["TVT", "GR"],
    ).sort_values("TVT")
    return horizontal, typewell


# Stage 1: exact representation audit and coefficient bank.
representation_rows = []
coefficient_rows = []
representation_sse = {degree: 0.0 for degree in DEGREES}
representation_count = 0
for number, row in enumerate(folds.itertuples(index=False), 1):
    well = str(row.well)
    horizontal, _ = _load(well)
    md = horizontal["MD"].to_numpy(float)
    z = horizontal["Z"].to_numpy(float)
    truth = horizontal["TVT"].to_numpy(float)
    visible = np.isfinite(horizontal["TVT_input"].to_numpy(float))
    hidden = ~visible & np.isfinite(truth)
    hidden_rows = np.flatnonzero(hidden)
    anchor = int(hidden_rows[0] - 1)
    length = max(float(md[hidden_rows[-1]] - md[anchor]), 1.0)
    progress = (md[hidden_rows] - md[anchor]) / length
    anchor_surface = float(horizontal["TVT_input"].iloc[anchor] + z[anchor])
    surface_delta = truth[hidden_rows] + z[hidden_rows] - anchor_surface
    result = {"well": well, "fold": int(row.fold), "rows": len(hidden_rows), "length_ft": length}
    for degree in DEGREES:
        coefficient = _fit_anchored(progress, surface_delta, degree)
        prediction = anchor_surface + _design(progress, degree) @ coefficient - z[hidden_rows]
        error = prediction - truth[hidden_rows]
        result[f"degree{degree}_rmse"] = float(np.sqrt(np.mean(error * error)))
        representation_sse[degree] += float(np.dot(error, error))
        if degree == SEARCH_DEGREE:
            coefficient_rows.append({
                "well": well, "fold": int(row.fold),
                **{f"c{index + 1}": float(value) for index, value in enumerate(coefficient)},
            })
    representation_count += len(hidden_rows)
    representation_rows.append(result)
    if number % 100 == 0 or number == len(folds):
        print(f"V228 representation {number}/{len(folds)}", flush=True)

representation = pd.DataFrame(representation_rows)
coefficients = pd.DataFrame(coefficient_rows)
pooled_representation = {
    str(degree): float(np.sqrt(representation_sse[degree] / representation_count))
    for degree in DEGREES
}


def _candidate_bank(prior, rng):
    columns = [f"c{index}" for index in range(1, SEARCH_DEGREE + 1)]
    values = prior[columns].to_numpy(float)
    median = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - median), axis=0) + 1e-3
    indices = rng.integers(0, len(values), size=CANDIDATES)
    bank = values[indices].copy()
    jitter_scales = np.array((0.03, 0.08, 0.18, 0.35), float)
    jitter = jitter_scales[np.arange(CANDIDATES) % len(jitter_scales), None]
    bank += rng.normal(size=bank.shape) * scale[None, :] * jitter
    # Deterministic broad anchors ensure coverage beyond the empirical donors.
    bank[: len(values)] = values[: min(len(values), CANDIDATES)]
    bank[-1] = median
    return bank


def _wholewell_cost(bank, progress, eval_gr, tw_tvt, tw_gr, slope, intercept):
    design = _design(progress, SEARCH_DEGREE)
    costs = np.empty(len(bank), float)
    for start in range(0, len(bank), BATCH):
        stop = min(start + BATCH, len(bank))
        candidate_tvt = bank[start:stop] @ design.T
        # candidate_tvt is surface delta; TVT conversion is supplied by caller
        yield start, stop, candidate_tvt


selected = (
    folds.assign(_hash=folds["well"].map(_well_hash))
    .sort_values(["fold", "_hash"])
    .groupby("fold", group_keys=False)
    .head(10)
    .sort_values(["fold", "_hash"])
    .head(SEARCH_WELLS)
)
search_rows = []
topk_sse = {topk: 0.0 for topk in TOPK}
topk_count = 0
argmin_sse = 0.0
oracle_poly_sse = 0.0
rng = np.random.default_rng(SEED)

for number, selected_row in enumerate(selected.itertuples(index=False), 1):
    well = str(selected_row.well)
    held_fold = int(selected_row.fold)
    horizontal, typewell = _load(well)
    md = horizontal["MD"].to_numpy(float)
    z = horizontal["Z"].to_numpy(float)
    gr = horizontal["GR"].to_numpy(float)
    truth = horizontal["TVT"].to_numpy(float)
    tvt_input = horizontal["TVT_input"].to_numpy(float)
    visible = np.isfinite(tvt_input) & np.isfinite(gr)
    hidden = ~np.isfinite(tvt_input) & np.isfinite(gr) & np.isfinite(truth)
    hidden_rows = np.flatnonzero(hidden)
    anchor = int(np.flatnonzero(np.isfinite(tvt_input))[-1])
    length = max(float(md[hidden_rows[-1]] - md[anchor]), 1.0)
    all_progress = (md[hidden_rows] - md[anchor]) / length
    sample_positions = np.unique(np.round(np.linspace(0, len(hidden_rows) - 1, min(COST_POINTS, len(hidden_rows)))).astype(int))
    sample_rows = hidden_rows[sample_positions]
    progress = all_progress[sample_positions]
    eval_gr = gr[sample_rows]
    anchor_surface = float(tvt_input[anchor] + z[anchor])
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = typewell["GR"].to_numpy(float)
    tw_prefix = np.interp(tvt_input[visible], tw_tvt, tw_gr)
    calibration_slope, calibration_intercept = _robust_affine(tw_prefix, gr[visible])
    gr_scale = max(1.4826 * np.median(np.abs(gr[visible] - np.median(gr[visible]))), 5.0)

    prior = coefficients.loc[~coefficients["fold"].eq(held_fold)]
    bank = _candidate_bank(prior, rng)
    oracle_row = coefficients.loc[coefficients["well"].eq(well)].iloc[0]
    oracle_coefficient = oracle_row[[f"c{index}" for index in range(1, SEARCH_DEGREE + 1)]].to_numpy(float)
    bank = np.vstack((bank, oracle_coefficient))
    oracle_index = len(bank) - 1
    design = _design(progress, SEARCH_DEGREE)
    costs = np.empty(len(bank), float)
    rolled_costs = np.empty(len(bank), float)
    rolled_tw = np.roll(tw_gr, max(31, len(tw_gr) // 5))
    for start in range(0, len(bank), BATCH):
        stop = min(start + BATCH, len(bank))
        surface = anchor_surface + bank[start:stop] @ design.T
        candidate_tvt = surface - z[sample_rows][None, :]
        inside = (candidate_tvt >= tw_tvt[0]) & (candidate_tvt <= tw_tvt[-1])
        candidate_flat = candidate_tvt.reshape(-1)
        aligned = np.interp(candidate_flat, tw_tvt, tw_gr).reshape(candidate_tvt.shape)
        aligned = calibration_slope * aligned + calibration_intercept
        residual = (eval_gr[None, :] - aligned) / gr_scale
        robust = np.mean(np.log1p((residual / 1.5) ** 2), axis=1)
        costs[start:stop] = robust + 4.0 * (1.0 - inside.mean(axis=1))
        rolled = np.interp(candidate_flat, tw_tvt, rolled_tw).reshape(candidate_tvt.shape)
        rolled = calibration_slope * rolled + calibration_intercept
        rolled_residual = (eval_gr[None, :] - rolled) / gr_scale
        rolled_costs[start:stop] = np.mean(np.log1p((rolled_residual / 1.5) ** 2), axis=1) + 4.0 * (1.0 - inside.mean(axis=1))

    order = np.argsort(costs, kind="stable")
    rolled_order = np.argsort(rolled_costs, kind="stable")
    oracle_rank = int(np.flatnonzero(order == oracle_index)[0]) + 1
    rolled_rank = int(np.flatnonzero(rolled_order == oracle_index)[0]) + 1
    full_design = _design(all_progress, SEARCH_DEGREE)
    candidate_predictions = anchor_surface + bank @ full_design.T - z[hidden_rows][None, :]
    candidate_rmse = np.sqrt(np.mean((candidate_predictions - truth[hidden_rows][None, :]) ** 2, axis=1))
    best_index = int(order[0])
    argmin_sse += float(np.sum((candidate_predictions[best_index] - truth[hidden_rows]) ** 2))
    oracle_poly_sse += float(np.sum((candidate_predictions[oracle_index] - truth[hidden_rows]) ** 2))
    topk_values = {}
    for topk in TOPK:
        chosen = order[:topk]
        oracle_choice = int(chosen[np.argmin(candidate_rmse[chosen])])
        error = candidate_predictions[oracle_choice] - truth[hidden_rows]
        topk_sse[topk] += float(np.dot(error, error))
        topk_values[f"top{topk}_oracle_rmse"] = float(candidate_rmse[oracle_choice])
    topk_count += len(hidden_rows)
    search_rows.append({
        "well": well, "fold": held_fold, "rows": len(hidden_rows),
        "oracle_rank": oracle_rank, "rolled_oracle_rank": rolled_rank,
        "argmin_rmse": float(candidate_rmse[best_index]),
        "oracle_poly_rmse": float(candidate_rmse[oracle_index]),
        "oracle_cost_increment": float(costs[oracle_index] - costs[best_index]),
        **topk_values,
    })
    print(f"V228 search {number}/{len(selected)} rank={oracle_rank} argmin_rmse={candidate_rmse[best_index]:.3f}", flush=True)

search = pd.DataFrame(search_rows)
search_summary = {
    "wells": int(len(search)),
    "median_oracle_rank": float(search["oracle_rank"].median()),
    "p75_oracle_rank": float(search["oracle_rank"].quantile(0.75)),
    "top8_recall": float((search["oracle_rank"] <= 8).mean()),
    "top32_recall": float((search["oracle_rank"] <= 32).mean()),
    "rolled_median_oracle_rank": float(search["rolled_oracle_rank"].median()),
    "argmin_pooled_rmse": float(np.sqrt(argmin_sse / topk_count)),
    "oracle_polynomial_pooled_rmse": float(np.sqrt(oracle_poly_sse / topk_count)),
    "topk_oracle_pooled_rmse": {
        str(topk): float(np.sqrt(topk_sse[topk] / topk_count)) for topk in TOPK
    },
}
checks = {
    "degree4_representation_rmse_le_4": pooled_representation[str(SEARCH_DEGREE)] <= 4.0,
    "median_oracle_rank_le_8": search_summary["median_oracle_rank"] <= 8.0,
    "top32_recall_ge_0p70": search_summary["top32_recall"] >= 0.70,
    "aligned_rank_beats_rolled": search_summary["median_oracle_rank"] < search_summary["rolled_median_oracle_rank"],
    "top32_oracle_rmse_le_6": search_summary["topk_oracle_pooled_rmse"]["32"] <= 6.0,
}
verdict = {
    "experiment": "global_polynomial_registration_gateway_v228",
    "stage": "773well_representation_and_48well_wholeGR_coefficient_rank_gateway",
    "hypothesis": "whole-well GR can identify a low-dimensional anchored structural polynomial",
    "representation_pooled_rmse": pooled_representation,
    "search": search_summary,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> build full fivefold coefficient posterior decoder" if all(checks.values())
            else "FAIL -> whole-well low-dimensional registration does not resolve aliases; close polynomial search",
}

representation.to_csv(OUT / "representation_by_well.csv", index=False)
coefficients.to_csv(OUT / "degree4_coefficients.csv", index=False)
search.to_csv(OUT / "search_wells.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
print("\nV228 REPRESENTATION")
print(pd.DataFrame([{"degree": int(degree), "pooled_rmse": value} for degree, value in pooled_representation.items()]).to_string(index=False))
print("\nV228 SEARCH")
print(search.to_string(index=False))
print("\nV228 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
