"""V201: truth-only block-transition audit before any posterior CRF training.

The audit decides whether a smooth transition prior is defensible, whether a
heavy-tailed/switching transition is required, and how wide an anchor-relative
state axis must be.  Folds 4/5 estimate distribution parameters; Fold 3 is a
single confirmation.  Folds 1/2 are not loaded.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "truth_block_transition_audit_v201"
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF = ROOT / "oof_exports"
OUT = OOF / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.time()

FOLD_CANDIDATES = (
    OOF / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.is_file()), None)
if FOLD_PATH is None:
    raise FileNotFoundError("Canonical five-fold well_metadata.csv was not found")
if not (DATA_ROOT / "train").is_dir():
    raise FileNotFoundError(DATA_ROOT / "train")

DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
UNTOUCHED_FOLDS = (1, 2)
BLOCK_SIZES = (32.0, 64.0, 128.0, 256.0)
AXIS_RADII = (32.0, 64.0, 96.0, 128.0, 192.0, 256.0)


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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 8 or np.ptp(x) < 1e-6:
        return 0.0, float(np.nanmedian(y))
    keep = np.ones(len(x), bool)
    slope, intercept = 0.0, float(np.median(y))
    for _ in range(6):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope * x + intercept)
        centre = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - centre)))
        scale = max(1.4826 * mad, 1e-4)
        updated = np.abs(residual - centre) <= 3.5 * scale
        if updated.sum() < 8 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(slope), float(intercept)


def _well_blocks(well: str, fold: int, block_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(path, usecols=["MD", "TVT", "TVT_input"])
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(visible) & np.isfinite(md))
    evaluation = np.flatnonzero(~np.isfinite(visible) & np.isfinite(truth) & np.isfinite(md))
    if not len(known) or len(evaluation) < 16:
        raise RuntimeError(f"{well}: missing visible prefix or evaluation truth")

    anchor = int(known[-1])
    anchor_md = float(md[anchor])
    anchor_tvt = float(visible[anchor])
    prefix = known[md[known] >= anchor_md - 512.0]
    slope, intercept = _robust_line(md[prefix] - anchor_md, visible[prefix] - anchor_tvt)

    eval_md = md[evaluation]
    eval_truth = truth[evaluation]
    distance = eval_md - anchor_md
    valid = distance >= -1e-6
    eval_md = eval_md[valid]
    eval_truth = eval_truth[valid]
    distance = distance[valid]
    block_id = np.floor(np.maximum(distance, 0.0) / block_size).astype(np.int64)
    raw = pd.DataFrame({"block": block_id, "md": eval_md, "truth": eval_truth})
    grouped = raw.groupby("block", sort=True).agg(
        md=("md", "median"), truth=("truth", "median"), rows=("truth", "size")
    ).reset_index()
    grouped = grouped.loc[grouped["rows"].ge(max(8, int(block_size // 4)))].reset_index(drop=True)
    if len(grouped) < 3:
        raise RuntimeError(f"{well}: fewer than three complete-ish blocks at {block_size} ft")

    grouped["well"] = well
    grouped["fold"] = fold
    grouped["block_size"] = block_size
    grouped["distance"] = grouped["md"] - anchor_md
    grouped["linear_baseline"] = anchor_tvt + slope * grouped["distance"]
    grouped["state_residual"] = grouped["truth"] - grouped["linear_baseline"]
    grouped["prefix_slope"] = slope

    delta_md = np.diff(grouped["md"].to_numpy(np.float64))
    transition = np.diff(grouped["truth"].to_numpy(np.float64))
    rate = transition / np.maximum(delta_md, 1e-6)
    edge = pd.DataFrame({
        "well": well,
        "fold": fold,
        "block_size": block_size,
        "left_block": grouped["block"].to_numpy(np.int64)[:-1],
        "delta_md": delta_md,
        "transition_ft": transition,
        "rate": rate,
    })
    edge["rate_change"] = np.nan
    if len(edge) > 1:
        edge.loc[edge.index[1:], "rate_change"] = np.diff(rate)
    return grouped, edge


def _fit_distributions(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    centre = float(np.median(values))
    residual = values - centre
    sigma = max(float(np.sqrt(np.mean(residual * residual))), 1e-8)
    laplace = max(float(np.mean(np.abs(residual))), 1e-8)
    base_scale = max(float(np.median(np.abs(residual))), 1e-8)
    scales = base_scale * np.logspace(-1.0, 1.0, 81)
    cauchy_losses = [np.mean(np.log(math.pi * scale) + np.log1p((residual / scale) ** 2)) for scale in scales]
    cauchy = float(scales[int(np.argmin(cauchy_losses))])
    return {"centre": centre, "gaussian_scale": sigma, "laplace_scale": laplace, "cauchy_scale": cauchy}


def _distribution_nll(values: np.ndarray, parameters: dict) -> dict:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    residual = values - parameters["centre"]
    sigma = parameters["gaussian_scale"]
    laplace = parameters["laplace_scale"]
    cauchy = parameters["cauchy_scale"]
    return {
        "gaussian": float(np.mean(0.5 * np.log(2.0 * math.pi * sigma * sigma) + 0.5 * (residual / sigma) ** 2)),
        "laplace": float(np.mean(np.log(2.0 * laplace) + np.abs(residual) / laplace)),
        "cauchy": float(np.mean(np.log(math.pi * cauchy) + np.log1p((residual / cauchy) ** 2))),
    }


folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
folds = folds.loc[folds["fold"].isin((*DISCOVERY_FOLDS, CONFIRMATION_FOLD))].sort_values(["fold", "well"])
expected = {3: 154, 4: 155, 5: 154}
coverage = folds.groupby("fold")["well"].nunique().to_dict()
if coverage != expected:
    raise RuntimeError(f"Unexpected Fold345 coverage: {coverage}")

block_parts = []
edge_parts = []
failures = []
for block_size in BLOCK_SIZES:
    for number, row in enumerate(folds.itertuples(index=False), 1):
        try:
            blocks, edges = _well_blocks(row.well, int(row.fold), block_size)
            block_parts.append(blocks)
            edge_parts.append(edges)
        except Exception as exc:
            failures.append({"well": row.well, "fold": int(row.fold), "block_size": block_size, "reason": repr(exc)})
        if number % 100 == 0 or number == len(folds):
            print(f"V201 block{int(block_size)} {number}/{len(folds)} failures={len(failures)}", flush=True)

blocks = pd.concat(block_parts, ignore_index=True)
edges = pd.concat(edge_parts, ignore_index=True)
pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)
blocks.to_parquet(OUT / "truth_blocks.parquet", index=False)
edges.to_parquet(OUT / "truth_transitions.parquet", index=False)

transition_rows = []
axis_rows = []
distribution_rows = []
recommendations = []

for block_size in BLOCK_SIZES:
    local_edges = edges.loc[edges["block_size"].eq(block_size)]
    local_blocks = blocks.loc[blocks["block_size"].eq(block_size)]
    fold4_change = local_edges.loc[local_edges["fold"].eq(4), "rate_change"].dropna().to_numpy(np.float64)
    fold5_change = local_edges.loc[local_edges["fold"].eq(5), "rate_change"].dropna().to_numpy(np.float64)
    discovery_change = local_edges.loc[local_edges["fold"].isin(DISCOVERY_FOLDS), "rate_change"].dropna().to_numpy(np.float64)
    confirmation_change = local_edges.loc[local_edges["fold"].eq(CONFIRMATION_FOLD), "rate_change"].dropna().to_numpy(np.float64)
    selection_parameters = _fit_distributions(fold4_change)
    selection_nll = _distribution_nll(fold5_change, selection_parameters)
    selected_family = min(selection_nll, key=selection_nll.get)
    parameters = _fit_distributions(discovery_change)
    confirmation_nll = _distribution_nll(confirmation_change, parameters)

    absolute = np.abs(confirmation_change - parameters["centre"])
    p50, p90, p95, p99, p999 = np.quantile(absolute, [0.50, 0.90, 0.95, 0.99, 0.999])
    selection_threshold = max(6.0 * np.median(np.abs(fold4_change - np.median(fold4_change))), 1e-6)
    selection_jump_fraction = float(np.mean(np.abs(fold5_change - selection_parameters["centre"]) > selection_threshold))
    robust_threshold = max(6.0 * np.median(np.abs(discovery_change - np.median(discovery_change))), 1e-6)
    jump_fraction = float(np.mean(absolute > robust_threshold))
    transition_rows.append({
        "block_size": block_size,
        "confirmation_edges": len(confirmation_change),
        "abs_rate_change_p50": p50,
        "abs_rate_change_p90": p90,
        "abs_rate_change_p95": p95,
        "abs_rate_change_p99": p99,
        "abs_rate_change_p999": p999,
        "p99_over_p95": p99 / max(p95, 1e-12),
        "robust_jump_threshold": robust_threshold,
        "selection_jump_fraction": selection_jump_fraction,
        "confirmation_jump_fraction": jump_fraction,
        "selected_distribution": selected_family,
    })
    distribution_rows.extend({
        "block_size": block_size,
        "distribution": family,
        "selection_mean_nll": selection_nll[family],
        "confirmation_mean_nll": confirmation_nll[family],
        **parameters,
    } for family in confirmation_nll)

    discovery_blocks = local_blocks.loc[local_blocks["fold"].isin(DISCOVERY_FOLDS)]
    confirmation_blocks = local_blocks.loc[local_blocks["fold"].eq(CONFIRMATION_FOLD)]
    for radius in AXIS_RADII:
        discovery_covered = np.abs(discovery_blocks["state_residual"].to_numpy(np.float64)) <= radius
        confirmation_covered = np.abs(confirmation_blocks["state_residual"].to_numpy(np.float64)) <= radius
        discovery_per_well = pd.Series(discovery_covered).groupby(discovery_blocks["well"].to_numpy()).mean()
        confirmation_per_well = pd.Series(confirmation_covered).groupby(confirmation_blocks["well"].to_numpy()).mean()
        axis_rows.append({
            "block_size": block_size,
            "axis_radius_ft": radius,
            "discovery_row_coverage": float(np.mean(discovery_covered)),
            "discovery_p05_well_coverage": float(discovery_per_well.quantile(0.05)),
            "confirmation_row_coverage": float(np.mean(confirmation_covered)),
            "confirmation_minimum_well_coverage": float(confirmation_per_well.min()),
            "confirmation_p05_well_coverage": float(confirmation_per_well.quantile(0.05)),
        })

    axis_local = pd.DataFrame([row for row in axis_rows if row["block_size"] == block_size])
    eligible_axis = axis_local.loc[
        axis_local["discovery_row_coverage"].ge(0.995)
        & axis_local["discovery_p05_well_coverage"].ge(0.95)
    ]
    radius = float(eligible_axis.iloc[0]["axis_radius_ft"]) if len(eligible_axis) else float(max(AXIS_RADII))
    transition_form = (
        "gaussian_smooth"
        if selected_family == "gaussian" and selection_jump_fraction < 0.005
        else "heavy_tail_rate_change"
        if selection_jump_fraction < 0.02
        else "switching_smooth_plus_jump"
    )
    recommendations.append({
        "block_size": block_size,
        "state_axis_radius_ft": radius,
        "transition_form": transition_form,
        "fold5_selected_distribution": selected_family,
        "selection_jump_fraction": selection_jump_fraction,
        "confirmation_jump_fraction": jump_fraction,
    })

transition_summary = pd.DataFrame(transition_rows)
axis_summary = pd.DataFrame(axis_rows)
distribution_summary = pd.DataFrame(distribution_rows)
recommendation_frame = pd.DataFrame(recommendations)
transition_summary.to_csv(OUT / "transition_summary.csv", index=False)
axis_summary.to_csv(OUT / "axis_coverage.csv", index=False)
distribution_summary.to_csv(OUT / "distribution_nll.csv", index=False)
recommendation_frame.to_csv(OUT / "recommendations.csv", index=False)

preferred = recommendation_frame.loc[recommendation_frame["block_size"].eq(128.0)].iloc[0].to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_truth_transition_discovery_Fold3_confirmation",
    "truth_usage": "transition-family and state-axis diagnostics only",
    "protocol": {
        "discovery_folds": list(DISCOVERY_FOLDS),
        "confirmation_fold": CONFIRMATION_FOLD,
        "untouched_folds": list(UNTOUCHED_FOLDS),
    },
    "coverage": coverage,
    "block_sizes_ft": list(BLOCK_SIZES),
    "failures": len(failures),
    "preferred_128ft": preferred,
    "next_model_gates": {
        "fold2_confirmation_standalone_rmse_le_9p5": True,
        "fold2_confirmation_detrended_error_corr_with_v168_le_0p50": True,
        "fold2_shuffled_typewell_degradation_ge_0p30": True,
        "fold2_no_gr_degradation_ge_0p30": True,
        "fold2_posterior_top8_oracle_rmse_le_5p5": True,
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "Use the frozen 128-ft state radius and transition form in V202; train on Fold45, select epoch/calibration on Fold3, gate once on untouched Fold2, and reserve Fold1 for the final application.",
}
_write_json(OUT / "verdict.json", verdict)

print("\nV201 TRANSITION SUMMARY")
print(transition_summary.to_string(index=False), flush=True)
print("\nV201 AXIS COVERAGE")
print(axis_summary.to_string(index=False), flush=True)
print("\nV201 RECOMMENDATIONS")
print(recommendation_frame.to_string(index=False), flush=True)
print("\nV201 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT, flush=True)
