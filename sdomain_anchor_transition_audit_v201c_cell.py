"""V201c: corrected anchor-axis and geological S-domain transition audit.

V201 used TVT transitions and a noisy prefix-tangent centre.  This audit uses
the legal final visible TVT as the state-axis centre and S=TVT+Z for geological
transition statistics.  Fold4 fits, Fold5 selects, Fold45 refits, and Fold3 is
confirmation only.  Folds1/2 are untouched.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "sdomain_anchor_transition_audit_v201c"
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
    raise FileNotFoundError("Canonical fold metadata not found")

BLOCK_SIZE = 128.0
RADII = (32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 256.0, 384.0, 512.0)


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


def _fit(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    centre = float(np.median(values))
    residual = values - centre
    gaussian = max(float(np.sqrt(np.mean(residual**2))), 1e-8)
    laplace = max(float(np.mean(np.abs(residual))), 1e-8)
    base = max(float(np.median(np.abs(residual))), 1e-8)
    grid = base * np.logspace(-1.0, 1.0, 81)
    losses = [np.mean(np.log(math.pi * scale) + np.log1p((residual / scale) ** 2)) for scale in grid]
    return {"centre": centre, "gaussian": gaussian, "laplace": laplace, "cauchy": float(grid[int(np.argmin(losses))])}


def _nll(values, parameters):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    residual = values - parameters["centre"]
    sigma, laplace, cauchy = parameters["gaussian"], parameters["laplace"], parameters["cauchy"]
    return {
        "gaussian": float(np.mean(0.5 * np.log(2 * math.pi * sigma**2) + 0.5 * (residual / sigma) ** 2)),
        "laplace": float(np.mean(np.log(2 * laplace) + np.abs(residual) / laplace)),
        "cauchy": float(np.mean(np.log(math.pi * cauchy) + np.log1p((residual / cauchy) ** 2))),
    }


folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
folds = folds.loc[folds["fold"].isin([3, 4, 5])].sort_values(["fold", "well"])
if folds.groupby("fold")["well"].nunique().to_dict() != {3: 154, 4: 155, 5: 154}:
    raise RuntimeError("Unexpected Fold345 coverage")

block_parts = []
edge_parts = []
for number, row in enumerate(folds.itertuples(index=False), 1):
    path = DATA_ROOT / "train" / f"{row.well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(np.float64)
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(np.float64)
    visible = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(visible) & np.isfinite(md) & np.isfinite(z))
    evaluation = np.flatnonzero(~np.isfinite(visible) & np.isfinite(tvt) & np.isfinite(md) & np.isfinite(z))
    if not len(known) or len(evaluation) < 16:
        raise RuntimeError(f"{row.well}: invalid official mask")
    anchor = int(known[-1])
    anchor_md = float(md[anchor])
    anchor_tvt = float(visible[anchor])
    anchor_s = anchor_tvt + float(z[anchor])
    distance = md[evaluation] - anchor_md
    keep = distance >= -1e-6
    local = pd.DataFrame({
        "block": np.floor(np.maximum(distance[keep], 0.0) / BLOCK_SIZE).astype(np.int64),
        "md": md[evaluation][keep],
        "z": z[evaluation][keep],
        "tvt": tvt[evaluation][keep],
    })
    block = local.groupby("block", sort=True).agg(
        md=("md", "median"), z=("z", "median"), tvt=("tvt", "median"), rows=("tvt", "size")
    ).reset_index()
    block = block.loc[block["rows"].ge(32)].reset_index(drop=True)
    if len(block) < 3:
        raise RuntimeError(f"{row.well}: too few 128-ft blocks")
    block["well"] = row.well
    block["fold"] = int(row.fold)
    block["anchor_tvt_offset"] = block["tvt"] - anchor_tvt
    block["anchor_s_offset"] = block["tvt"] + block["z"] - anchor_s
    block_parts.append(block)

    delta_md = np.diff(block["md"].to_numpy(np.float64))
    tvt_rate = np.diff(block["tvt"].to_numpy(np.float64)) / np.maximum(delta_md, 1e-6)
    s_values = block["tvt"].to_numpy(np.float64) + block["z"].to_numpy(np.float64)
    s_rate = np.diff(s_values) / np.maximum(delta_md, 1e-6)
    edge = pd.DataFrame({
        "well": row.well,
        "fold": int(row.fold),
        "tvt_rate": tvt_rate,
        "s_rate": s_rate,
    })
    edge["tvt_rate_change"] = np.r_[np.nan, np.diff(tvt_rate)]
    edge["s_rate_change"] = np.r_[np.nan, np.diff(s_rate)]
    edge_parts.append(edge)
    if number % 100 == 0 or number == len(folds):
        print(f"V201c wells {number}/{len(folds)}", flush=True)

blocks = pd.concat(block_parts, ignore_index=True)
edges = pd.concat(edge_parts, ignore_index=True)
blocks.to_parquet(OUT / "anchor_s_blocks.parquet", index=False)
edges.to_parquet(OUT / "anchor_s_transitions.parquet", index=False)

axis_rows = []
for coordinate in ("anchor_tvt_offset", "anchor_s_offset"):
    for radius in RADII:
        row = {"coordinate": coordinate, "radius_ft": radius}
        for label, mask in (("discovery", blocks["fold"].isin([4, 5])), ("confirmation", blocks["fold"].eq(3))):
            local = blocks.loc[mask]
            covered = np.abs(local[coordinate].to_numpy(np.float64)) <= radius
            per_well = pd.Series(covered).groupby(local["well"].to_numpy()).mean()
            row[f"{label}_row_coverage"] = float(np.mean(covered))
            row[f"{label}_p05_well_coverage"] = float(per_well.quantile(0.05))
            row[f"{label}_minimum_well_coverage"] = float(per_well.min())
        axis_rows.append(row)
axis = pd.DataFrame(axis_rows)

distribution_rows = []
recommendations = {}
for target in ("tvt_rate_change", "s_rate_change"):
    fold4 = edges.loc[edges["fold"].eq(4), target].dropna().to_numpy(np.float64)
    fold5 = edges.loc[edges["fold"].eq(5), target].dropna().to_numpy(np.float64)
    fold3 = edges.loc[edges["fold"].eq(3), target].dropna().to_numpy(np.float64)
    selection_parameters = _fit(fold4)
    selection = _nll(fold5, selection_parameters)
    selected = min(selection, key=selection.get)
    parameters = _fit(np.r_[fold4, fold5])
    confirmation = _nll(fold3, parameters)
    threshold = max(6.0 * np.median(np.abs(np.r_[fold4, fold5] - parameters["centre"])), 1e-8)
    confirmation_jump = float(np.mean(np.abs(fold3 - parameters["centre"]) > threshold))
    for family in selection:
        distribution_rows.append({
            "target": target,
            "family": family,
            "fold5_selection_nll": selection[family],
            "fold3_confirmation_nll": confirmation[family],
            "selected_on_fold5": family == selected,
        })
    recommendations[target] = {
        "selected_family": selected,
        "centre": parameters["centre"],
        "scale": parameters[selected],
        "confirmation_jump_fraction": confirmation_jump,
    }

eligible = axis.loc[
    axis["coordinate"].eq("anchor_tvt_offset")
    & axis["discovery_row_coverage"].ge(0.995)
    & axis["discovery_p05_well_coverage"].ge(0.95)
]
selected_radius = float(eligible.iloc[0]["radius_ft"]) if len(eligible) else None
confirmation_row = axis.loc[
    axis["coordinate"].eq("anchor_tvt_offset") & axis["radius_ft"].eq(selected_radius)
].iloc[0] if selected_radius is not None else None

distribution = pd.DataFrame(distribution_rows)
axis.to_csv(OUT / "axis_coverage.csv", index=False)
distribution.to_csv(OUT / "transition_distribution.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold4_fit_Fold5_select_Fold45_refit_Fold3_confirmation",
    "block_size_ft": BLOCK_SIZE,
    "selected_anchor_tvt_radius_ft": selected_radius,
    "anchor_tvt_confirmation_row_coverage": (
        float(confirmation_row["confirmation_row_coverage"]) if confirmation_row is not None else None
    ),
    "anchor_tvt_confirmation_p05_well_coverage": (
        float(confirmation_row["confirmation_p05_well_coverage"]) if confirmation_row is not None else None
    ),
    "transition_recommendations": recommendations,
    "checks": {
        "anchor_axis_found": selected_radius is not None,
        "s_domain_not_less_heavy_than_tvt": recommendations["s_rate_change"]["selected_family"] in ("laplace", "cauchy"),
    },
    "next": (
        "Freeze the anchor-centred radius and S-domain transition in V202."
        if selected_radius is not None
        else "Do not train a fixed anchor-centred CRF; use an adaptive state centre."
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")

print("\nV201c AXIS COVERAGE")
print(axis.to_string(index=False), flush=True)
print("\nV201c TRANSITION DISTRIBUTIONS")
print(distribution.to_string(index=False), flush=True)
print("\nV201c FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT, flush=True)
