"""V230: constrain whole-well polynomial registration with visible-prefix S.

V229 showed that regularizing toward V168 is fold-unstable and pushes the true
coefficient down the rank list. V230 instead scores candidate continuation
curves against the legally visible S=TVT_input+Z immediately before the anchor.
The candidate generator and GR likelihood are frozen from V229.
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
V228 = OOF / "global_polynomial_registration_gateway_v228"
PARENT_PATH = OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
FOLD_PATH = OOF / "likpf_residual_gbdt_full5_v215/well_metadata.csv"
OUT = OOF / "prefix_constrained_polynomial_registration_v230"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 230
DEGREE = 4
CANDIDATES = 4096
EMPIRICAL_CANDIDATES = 2048
SEARCH_WELLS = 155
COST_POINTS = 256
BATCH = 256
PREFIX_WINDOWS_FT = (128.0, 256.0, 512.0, 1024.0)
LAMBDAS = (0.0, 0.01, 0.03, 0.10, 0.30, 1.0, 3.0)
MODES = ("hard", "soft8", "soft32")
DOSES = (0.25, 0.50, 1.0)
PREFIX_SCALE_FT = 4.0
SOFT_TEMPERATURE = 0.02


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(float(sse) / max(int(rows), 1)))


def _design(progress):
    progress = np.asarray(progress, float)
    return np.column_stack([progress ** power for power in range(1, DEGREE + 1)])


def _fit_coefficient(progress, surface_delta):
    matrix = _design(progress)
    coefficient = np.linalg.lstsq(matrix, surface_delta, rcond=None)[0]
    for _ in range(4):
        residual = surface_delta - matrix @ coefficient
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        coefficient = np.linalg.lstsq(matrix * weight[:, None], surface_delta * weight, rcond=None)[0]
    return coefficient


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


def _well_hash(well):
    return int(hashlib.sha256(str(well).encode()).hexdigest()[:12], 16)


for required in (DATA_ROOT / "train", V228 / "degree4_coefficients.csv", PARENT_PATH, FOLD_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
truth_coefficients = pd.read_csv(V228 / "degree4_coefficients.csv", dtype={"well": str})
coefficient_columns = [f"c{index}" for index in range(1, DEGREE + 1)]
if len(folds) != 773 or len(truth_coefficients) != 773:
    raise RuntimeError({"folds": len(folds), "coefficients": len(truth_coefficients)})

parent_raw = pd.read_parquet(PARENT_PATH)
if "id" not in parent_raw:
    well_col = next(name for name in ("well", "well_id") if name in parent_raw)
    row_col = next(name for name in ("row_idx", "row_index", "_row_idx") if name in parent_raw)
    parent_raw["id"] = parent_raw[well_col].astype(str) + "_" + parent_raw[row_col].astype(int).astype(str)
parent = parent_raw[["id", "metadata_analog_w0p05"]].rename(columns={"metadata_analog_w0p05": "v168"})
parent_lookup = parent.set_index("id")["v168"]

selected = (
    folds.assign(_hash=folds["well"].map(_well_hash))
    .sort_values(["fold", "_hash"])
    .groupby("fold", group_keys=False).head(31)
    .sort_values(["fold", "_hash"]).head(SEARCH_WELLS)
)


def _candidate_bank(prior, parent_coefficient, rng):
    values = prior[coefficient_columns].to_numpy(float)
    median = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - median), axis=0) + 1e-3
    empirical_index = rng.integers(0, len(values), size=EMPIRICAL_CANDIDATES)
    empirical = values[empirical_index].copy()
    empirical_jitter = np.array((0.02, 0.05, 0.10, 0.20))[np.arange(len(empirical)) % 4, None]
    empirical += rng.normal(size=empirical.shape) * scale[None, :] * empirical_jitter
    local_count = CANDIDATES - EMPIRICAL_CANDIDATES - 1
    local = np.repeat(parent_coefficient[None, :], local_count, axis=0)
    local_jitter = np.array((0.01, 0.03, 0.06, 0.12, 0.25))[np.arange(local_count) % 5, None]
    local += rng.normal(size=local.shape) * scale[None, :] * local_jitter
    return np.vstack((empirical, local, parent_coefficient[None, :]))


recipes = [
    {"window": float(window), "lambda": float(lam), "mode": mode, "dose": float(dose)}
    for window in PREFIX_WINDOWS_FT for lam in LAMBDAS for mode in MODES for dose in DOSES
]
recipe_keys = [(row["window"], row["lambda"], row["mode"], row["dose"]) for row in recipes]
fold_sse = {fold: {key: 0.0 for key in recipe_keys} for fold in range(1, 6)}
fold_rows = {fold: 0 for fold in range(1, 6)}
fold_parent_sse = {fold: 0.0 for fold in range(1, 6)}
rank_rows = []
rng = np.random.default_rng(SEED)

for number, selected_row in enumerate(selected.itertuples(index=False), 1):
    well = str(selected_row.well)
    held_fold = int(selected_row.fold)
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT", "TVT_input"],
    )
    typewell = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"]
    ).sort_values("TVT")
    md = horizontal["MD"].to_numpy(float)
    z = horizontal["Z"].to_numpy(float)
    gr = horizontal["GR"].to_numpy(float)
    truth = horizontal["TVT"].to_numpy(float)
    tvt_input = horizontal["TVT_input"].to_numpy(float)
    visible = np.isfinite(tvt_input) & np.isfinite(gr)
    hidden = ~np.isfinite(tvt_input) & np.isfinite(truth)
    hidden_rows = np.flatnonzero(hidden)
    anchor = int(np.flatnonzero(np.isfinite(tvt_input))[-1])
    length = max(float(md[hidden_rows[-1]] - md[anchor]), 1.0)
    progress_full = (md[hidden_rows] - md[anchor]) / length
    design_full = _design(progress_full)
    ids = [f"{well}_{index}" for index in hidden_rows]
    parent_values = parent_lookup.reindex(ids).to_numpy(float)
    if not np.isfinite(parent_values).all():
        raise RuntimeError(f"{well}: incomplete V168 alignment")
    anchor_surface = float(tvt_input[anchor] + z[anchor])
    parent_surface_delta = parent_values + z[hidden_rows] - anchor_surface
    parent_coefficient = _fit_coefficient(progress_full, parent_surface_delta)
    oracle_row = truth_coefficients.loc[truth_coefficients["well"].eq(well)].iloc[0]
    oracle_coefficient = oracle_row[coefficient_columns].to_numpy(float)
    prior = truth_coefficients.loc[~truth_coefficients["fold"].eq(held_fold)]
    bank = _candidate_bank(prior, parent_coefficient, rng)

    finite_cost_rows = np.flatnonzero(hidden & np.isfinite(gr))
    local_positions = np.searchsorted(hidden_rows, finite_cost_rows)
    sample_positions = np.unique(np.round(np.linspace(0, len(local_positions) - 1, min(COST_POINTS, len(local_positions)))).astype(int))
    local_sample = local_positions[sample_positions]
    sample_rows = hidden_rows[local_sample]
    progress = progress_full[local_sample]
    design = _design(progress)
    eval_gr = gr[sample_rows]
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = typewell["GR"].to_numpy(float)
    tw_prefix = np.interp(tvt_input[visible], tw_tvt, tw_gr)
    calibration_slope, calibration_intercept = _robust_affine(tw_prefix, gr[visible])
    gr_scale = max(1.4826 * np.median(np.abs(gr[visible] - np.median(gr[visible]))), 5.0)
    gr_cost = np.empty(len(bank), float)
    for start in range(0, len(bank), BATCH):
        stop = min(start + BATCH, len(bank))
        surface = anchor_surface + bank[start:stop] @ design.T
        candidate_tvt = surface - z[sample_rows][None, :]
        inside = (candidate_tvt >= tw_tvt[0]) & (candidate_tvt <= tw_tvt[-1])
        aligned = np.interp(candidate_tvt.reshape(-1), tw_tvt, tw_gr).reshape(candidate_tvt.shape)
        aligned = calibration_slope * aligned + calibration_intercept
        residual = (eval_gr[None, :] - aligned) / gr_scale
        gr_cost[start:stop] = np.mean(np.log1p((residual / 1.5) ** 2), axis=1) + 4.0 * (1.0 - inside.mean(axis=1))

    candidate_sample_tvt = anchor_surface + bank @ design.T - z[sample_rows][None, :]
    oracle_surface = anchor_surface + oracle_coefficient @ design.T
    oracle_tvt = oracle_surface - z[sample_rows]
    oracle_inside = (oracle_tvt >= tw_tvt[0]) & (oracle_tvt <= tw_tvt[-1])
    oracle_aligned = calibration_slope * np.interp(oracle_tvt, tw_tvt, tw_gr) + calibration_intercept
    oracle_residual = (eval_gr - oracle_aligned) / gr_scale
    oracle_gr_cost = float(np.mean(np.log1p((oracle_residual / 1.5) ** 2)) + 4.0 * (1.0 - oracle_inside.mean()))
    prefix_distance = {}
    oracle_prefix_distance = {}
    for window in PREFIX_WINDOWS_FT:
        prefix_rows = np.flatnonzero(
            visible & (md >= md[anchor] - window) & (md <= md[anchor])
        )
        if len(prefix_rows) < 16:
            prefix_rows = np.flatnonzero(visible)[-min(64, visible.sum()):]
        prefix_progress = (md[prefix_rows] - md[anchor]) / length
        prefix_design = _design(prefix_progress)
        observed_surface_delta = tvt_input[prefix_rows] + z[prefix_rows] - anchor_surface
        candidate_prefix_delta = bank @ prefix_design.T
        prefix_distance[window] = np.mean(
            (candidate_prefix_delta - observed_surface_delta[None, :]) ** 2, axis=1
        ) / (PREFIX_SCALE_FT ** 2)
        oracle_prefix_distance[window] = float(np.mean(
            (prefix_design @ oracle_coefficient - observed_surface_delta) ** 2
        ) / (PREFIX_SCALE_FT ** 2))

    coefficient_choice = {}
    bank_to_oracle_rmse = np.sqrt(np.mean(
        (candidate_sample_tvt - oracle_tvt[None, :]) ** 2, axis=1
    ))
    rank_record = {
        "well": well, "fold": held_fold, "rows": len(hidden_rows),
        "nearest_bank_oracle_rmse": float(bank_to_oracle_rmse.min()),
    }
    for window in PREFIX_WINDOWS_FT:
        for lam in LAMBDAS:
            score = gr_cost + lam * prefix_distance[window]
            order = np.argsort(score, kind="stable")
            oracle_score = oracle_gr_cost + lam * oracle_prefix_distance[window]
            rank_record[f"rank_w{int(window)}_l{lam:g}"] = int(np.sum(score < oracle_score)) + 1
            coefficient_choice[(window, lam, "hard")] = bank[order[0]]
            for topk in (8, 32):
                chosen = order[:topk]
                local_score = score[chosen]
                weight = np.exp(-(local_score - local_score.min()) / SOFT_TEMPERATURE)
                weight /= weight.sum()
                coefficient_choice[(window, lam, f"soft{topk}")] = weight @ bank[chosen]

    parent_error = parent_values - truth[hidden_rows]
    fold_parent_sse[held_fold] += float(np.dot(parent_error, parent_error))
    fold_rows[held_fold] += len(hidden_rows)
    for recipe in recipes:
        coefficient = coefficient_choice[(recipe["window"], recipe["lambda"], recipe["mode"])]
        low_frequency_move = design_full @ (coefficient - parent_coefficient)
        prediction = parent_values + recipe["dose"] * low_frequency_move
        error = prediction - truth[hidden_rows]
        key = (recipe["window"], recipe["lambda"], recipe["mode"], recipe["dose"])
        fold_sse[held_fold][key] += float(np.dot(error, error))
    rank_rows.append(rank_record)
    if number % 10 == 0 or number == len(selected):
        print(f"V230 wells {number}/{len(selected)}", flush=True)

rank_table = pd.DataFrame(rank_rows)
recipe_report = []
for key in recipe_keys:
    total_sse = sum(fold_sse[fold][key] for fold in range(1, 6))
    total_rows = sum(fold_rows.values())
    row = {"window": key[0], "lambda": key[1], "mode": key[2], "dose": key[3],
           "rmse": _rmse_from_sse(total_sse, total_rows)}
    for fold in range(1, 6):
        row[f"fold{fold}_rmse"] = _rmse_from_sse(fold_sse[fold][key], fold_rows[fold])
    recipe_report.append(row)
recipe_report = pd.DataFrame(recipe_report).sort_values("rmse").reset_index(drop=True)

# Nested recipe selection: each held fold is evaluated with a recipe selected
# solely from the other four folds.
nested_rows = []
nested_sse = 0.0
for held_fold in range(1, 6):
    best_key = min(
        recipe_keys,
        key=lambda key: (
            _rmse_from_sse(sum(fold_sse[fold][key] for fold in range(1, 6) if fold != held_fold),
                           sum(fold_rows[fold] for fold in range(1, 6) if fold != held_fold)),
            key,
        ),
    )
    candidate_sse = fold_sse[held_fold][best_key]
    nested_sse += candidate_sse
    base_rmse = _rmse_from_sse(fold_parent_sse[held_fold], fold_rows[held_fold])
    candidate_rmse = _rmse_from_sse(candidate_sse, fold_rows[held_fold])
    nested_rows.append({
        "fold": held_fold, "window": best_key[0], "lambda": best_key[1],
        "mode": best_key[2], "dose": best_key[3],
        "base_rmse": base_rmse, "candidate_rmse": candidate_rmse,
        "gain": base_rmse - candidate_rmse,
    })

total_rows = sum(fold_rows.values())
base_rmse = _rmse_from_sse(sum(fold_parent_sse.values()), total_rows)
nested_rmse = _rmse_from_sse(nested_sse, total_rows)
rank_summary = []
for window in PREFIX_WINDOWS_FT:
    for lam in LAMBDAS:
        column = f"rank_w{int(window)}_l{lam:g}"
        rank_summary.append({
            "window": window, "lambda": lam,
            "median_rank": float(rank_table[column].median()),
            "p75_rank": float(rank_table[column].quantile(0.75)),
            "top8_recall": float((rank_table[column] <= 8).mean()),
            "top32_recall": float((rank_table[column] <= 32).mean()),
        })
checks = {
    "nested_gain_ge_0p03": base_rmse - nested_rmse >= 0.03,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in nested_rows) >= 4,
    "minimum_fold_gain_ge_minus0p03": min(row["gain"] for row in nested_rows) >= -0.03,
    "nonzero_recipe_folds_ge_4": sum(row["dose"] > 0 for row in nested_rows) >= 4,
}
verdict = {
    "experiment": "prefix_constrained_polynomial_registration_v230",
    "stage": "155well_nested_visible_prefix_constrained_wholeGR_polynomial_registration",
    "candidate_generation": "other-fold coefficient bank plus V168-local jitter; oracle excluded",
    "prior": "visible-prefix S continuation only; no future V168 distance penalty",
    "base_rmse": base_rmse,
    "nested_rmse": nested_rmse,
    "nested_gain": base_rmse - nested_rmse,
    "candidate_support": {
        "median_nearest_bank_oracle_rmse": float(rank_table["nearest_bank_oracle_rmse"].median()),
        "p95_nearest_bank_oracle_rmse": float(rank_table["nearest_bank_oracle_rmse"].quantile(0.95)),
    },
    "rank_by_lambda": rank_summary,
    "folds": nested_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> run complete 773well nested confirmation" if all(checks.values())
            else "FAIL -> visible-prefix continuity cannot disambiguate whole-GR polynomial registration",
}

recipe_report.to_csv(OUT / "recipe_scan.csv", index=False)
rank_table.to_csv(OUT / "oracle_rank.csv", index=False)
pd.DataFrame(rank_summary).to_csv(OUT / "rank_summary.csv", index=False)
pd.DataFrame(nested_rows).to_csv(OUT / "folds.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
print("\nV230 RECIPE TOP 20")
print(recipe_report.head(20).to_string(index=False))
print("\nV230 RANK")
print(pd.DataFrame(rank_summary).to_string(index=False))
print("\nV230 FOLDS")
print(pd.DataFrame(nested_rows).to_string(index=False))
print("\nV230 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
