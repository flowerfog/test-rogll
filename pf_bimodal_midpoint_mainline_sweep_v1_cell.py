"""Nested OOF sweep for PF midpoint hedge on the CP12/Neighbor mainline.

Run after ``pf_bimodal_midpoint_hedge_oof_v0_cell.py`` has cached the exact
128-seed PF branch statistics. This stage is cheap: it never reruns a particle
filter. It tests the actual sep3 recipe, conservative doses, and whether the
constant well move belongs before or after geometric Neighbor.

The exact deployed chain is CP12 -> Gold -> geometric Neighbor. A fold-clean
row-level Gold endpoint has not historically been reliable, so by default this
script uses CP12 -> geometric Neighbor as a labelled surrogate. Set the
environment variable ``ROGII_GOLD_OOF_PATH`` to an id,tvt CSV to test the exact
chain when a trustworthy Gold OOF endpoint becomes available. The verdict
states which endpoint was used and never calls the surrogate exact.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
STATISTICS_PATH = (
    OOF_ROOT / "pf_bimodal_midpoint_hedge_oof_v0" / "bimodal_well_statistics.csv"
)
GEOMETRY_PATH = (
    OOF_ROOT
    / "neighbor_fullcurve_projection_direction_v1"
    / "primary_candidate_oof.npz"
)
NEIGHBOR_SOURCE = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
OUT_DIR = OOF_ROOT / "pf_bimodal_midpoint_mainline_sweep_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_OOF_PATH = os.environ.get("ROGII_GOLD_OOF_PATH", "").strip()

NEIGHBOR_WEIGHT = 0.25
NEIGHBOR_RAW_CAP_FT = 24.0
MIN_ABS_DIRECTION_DOT = 0.70
SEPARATION_HIGH_FT = 40.0

ALPHAS = (0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0)
CAPS_FT = (2.0, 3.0, 4.0)
MINOR_MASSES = (0.18, 0.22, 0.25, 0.26, 0.30)
SEPARATION_LOWS_FT = (3.0, 4.0, 6.0)
PLACEMENTS = ("after_neighbor", "before_neighbor")

# Exact recipe that changed the reported source artifact from LB 6.809 to
# 6.594. It ran after every other correction layer in the source notebook.
SOURCE_6P594_RECIPE = {
    "placement": "after_neighbor",
    "alpha": 0.6,
    "cap_ft": 2.0,
    "minor_mass": 0.25,
    "separation_low_ft": 4.0,
}

PUBLIC_RECIPE = {
    "placement": "after_neighbor",
    "alpha": 1.6,
    "cap_ft": 4.0,
    "minor_mass": 0.22,
    "separation_low_ft": 3.0,
}
CONSERVATIVE_RECIPE = {
    "placement": "after_neighbor",
    "alpha": 0.8,
    "cap_ft": 2.0,
    "minor_mass": 0.22,
    "separation_low_ft": 3.0,
}

BOOTSTRAP_ITERATIONS = 4000
SHUFFLE_ITERATIONS = 1000
SEED = 20260727


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )


def _rmse_from_sse(sse, rows):
    return float(math.sqrt(float(sse) / max(int(rows), 1)))


def _load_pool():
    header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
    requested = [
        "id",
        "well",
        "row_idx",
        "fold",
        "tvt_true",
        "cp12",
        "md_since",
    ]
    usecols = [column for column in requested if column in header]
    frame = pd.read_csv(
        POOL_PATH,
        usecols=usecols,
        dtype={"id": "string", "well": "string"},
        low_memory=False,
    )
    if "id" not in frame or "tvt_true" not in frame or "cp12" not in frame:
        raise RuntimeError("candidate pool must contain id, tvt_true, and cp12")
    if "well" not in frame:
        frame["well"] = frame["id"].str.rsplit("_", n=1).str[0]
    if "row_idx" not in frame:
        frame["row_idx"] = frame["id"].str.rsplit("_", n=1).str[-1].astype(int)
    frame["well"] = frame["well"].astype(str)
    if "fold" not in frame:
        folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
        fold_map = folds.groupby("well")["fold"].first().astype(int).to_dict()
        frame["fold"] = frame["well"].map(fold_map)
    if frame["fold"].isna().any():
        raise RuntimeError("incomplete fold mapping")
    for column in ("tvt_true", "cp12"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise RuntimeError(f"non-finite {column}")
    return frame.reset_index(drop=True)


def _load_gold_endpoint(pool):
    if not GOLD_OOF_PATH:
        return None
    path = Path(GOLD_OOF_PATH)
    if not path.exists():
        raise FileNotFoundError(path)
    endpoint = pd.read_csv(path, dtype={"id": "string"})
    if not {"id", "tvt"}.issubset(endpoint.columns):
        raise RuntimeError("Gold OOF CSV must contain id,tvt")
    endpoint = endpoint[["id", "tvt"]].drop_duplicates("id", keep="last")
    joined = pool[["id"]].merge(endpoint, on="id", how="left", validate="one_to_one")
    values = pd.to_numeric(joined["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("Gold OOF endpoint does not cover every pool row")
    return values


def _eligible_neighbor_rows(pool):
    path = NEIGHBOR_SOURCE / "selected_neighbors.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    selected = pd.read_csv(path, dtype={"well": str}).set_index("well")
    direction_column = (
        "abs_direction_dot" if "abs_direction_dot" in selected else "direction_dot"
    )
    required = {"median_distance", "p90_distance", "anchor_distance_full", direction_column}
    if not required.issubset(selected.columns):
        raise RuntimeError(f"selected_neighbors missing {sorted(required - set(selected.columns))}")
    eligible = (
        (selected["median_distance"] <= 600.0)
        & (selected["p90_distance"] <= 1200.0)
        & (selected["anchor_distance_full"] <= 1200.0)
        & (selected[direction_column].abs() >= MIN_ABS_DIRECTION_DOT)
    )
    return pool["well"].map(eligible.to_dict()).fillna(False).to_numpy(bool)


def _load_geometry_candidate(rows):
    if not GEOMETRY_PATH.exists():
        raise FileNotFoundError(GEOMETRY_PATH)
    with np.load(GEOMETRY_PATH, allow_pickle=False) as data:
        if "candidate" not in data:
            raise RuntimeError(f"candidate key missing from {GEOMETRY_PATH}")
        candidate = data["candidate"].astype(np.float64)
    if len(candidate) != rows:
        raise RuntimeError(f"geometry candidate length {len(candidate)} != {rows}")
    return candidate


def _ensure_md_since(pool):
    if "md_since" in pool:
        values = pd.to_numeric(pool["md_since"], errors="coerce").to_numpy(float)
        if np.isfinite(values).mean() > 0.99:
            return values
    output = np.full(len(pool), np.nan, np.float64)
    for number, (well, indices) in enumerate(pool.groupby("well", sort=False).indices.items(), 1):
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        horizontal = pd.read_csv(path, usecols=["MD", "TVT_input"])
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        known = horizontal["TVT_input"].notna().to_numpy(bool)
        if not known.any():
            raise RuntimeError(f"{well} has no visible TVT_input")
        anchor = int(np.flatnonzero(known)[-1])
        row_indices = pool.loc[indices, "row_idx"].to_numpy(int)
        output[indices] = md[row_indices] - md[anchor]
        if number % 100 == 0 or number == pool["well"].nunique():
            print(f"md_since {number}/{pool['well'].nunique()}", flush=True)
    if not np.isfinite(output).all():
        raise RuntimeError("failed to reconstruct md_since")
    return output


def _neighbor_prediction(upstream, candidate, eligible):
    move = np.zeros(len(upstream), np.float64)
    valid = eligible & np.isfinite(candidate)
    move[valid] = NEIGHBOR_WEIGHT * np.clip(
        candidate[valid] - upstream[valid],
        -NEIGHBOR_RAW_CAP_FT,
        NEIGHBOR_RAW_CAP_FT,
    )
    return upstream + move


def _recipe_key(recipe):
    return (
        recipe["placement"],
        float(recipe["alpha"]),
        float(recipe["cap_ft"]),
        float(recipe["minor_mass"]),
        float(recipe["separation_low_ft"]),
    )


def _recipe_name(recipe):
    return (
        f"{recipe['placement']}|a{recipe['alpha']:g}|c{recipe['cap_ft']:g}|"
        f"m{recipe['minor_mass']:g}|s{recipe['separation_low_ft']:g}"
    )


def _gate_and_shift(statistics, recipe):
    qualified = (
        (statistics["minor_mass"].to_numpy(float) >= recipe["minor_mass"])
        & (statistics["separation_ft"].to_numpy(float) >= recipe["separation_low_ft"])
        & (statistics["separation_ft"].to_numpy(float) <= SEPARATION_HIGH_FT)
    )
    shift = np.zeros(len(statistics), np.float64)
    proposed = recipe["alpha"] * statistics["midpoint_delta_ft"].to_numpy(float)
    shift[qualified] = np.clip(
        proposed[qualified], -recipe["cap_ft"], recipe["cap_ft"]
    )
    shift[np.abs(shift) <= 0.01] = 0.0
    return qualified, shift


def _row_shift(pool, statistics, recipe):
    qualified, shift = _gate_and_shift(statistics, recipe)
    shift_map = dict(zip(statistics["well"], shift))
    values = pool["well"].map(shift_map).fillna(0.0).to_numpy(float)
    return values, qualified, shift


def _prediction_for_recipe(
    pool, statistics, recipe, upstream, neighbor_base, candidate, eligible
):
    row_shift, qualified, well_shift = _row_shift(pool, statistics, recipe)
    if recipe["placement"] == "after_neighbor":
        prediction = neighbor_base + row_shift
    elif recipe["placement"] == "before_neighbor":
        shifted_upstream = upstream + row_shift
        prediction = _neighbor_prediction(shifted_upstream, candidate, eligible)
    else:
        raise ValueError(recipe["placement"])
    return prediction, qualified, well_shift


def _per_well_sse(pool, truth, prediction, statistics):
    records = []
    stat_index = statistics.set_index("well")
    for well, indices in pool.groupby("well", sort=False).indices.items():
        error = prediction[indices] - truth[indices]
        row = {
            "well": well,
            "fold": int(pool.loc[indices[0], "fold"]),
            "rows": int(len(indices)),
            "sse": float(np.sum(error * error)),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "error_sum": float(np.sum(error)),
        }
        if well in stat_index.index:
            for column in ("minor_mass", "separation_ft", "midpoint_delta_ft"):
                row[column] = float(stat_index.loc[well, column])
        records.append(row)
    return pd.DataFrame(records)


def _summarize_prediction(pool, truth, prediction, base, md_since):
    error = prediction - truth
    base_error = base - truth
    rows = len(error)
    report = {
        "rows": rows,
        "exact_sse": float(np.sum(error * error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "base_rmse": float(np.sqrt(np.mean(base_error * base_error))),
        "gain": float(np.sqrt(np.mean(base_error * base_error)) - np.sqrt(np.mean(error * error))),
        "mean_abs_move": float(np.mean(np.abs(prediction - base))),
        "p95_abs_move": float(np.quantile(np.abs(prediction - base), 0.95)),
        "max_abs_move": float(np.max(np.abs(prediction - base))),
    }
    masks = {
        "near_0_500": md_since <= 500.0,
        "mid_500_1000": (md_since > 500.0) & (md_since <= 1000.0),
        "mid_1000_2000": (md_since > 1000.0) & (md_since <= 2000.0),
        "far_gt2000": md_since > 2000.0,
    }
    for name, mask in masks.items():
        report[f"{name}_rows"] = int(mask.sum())
        report[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
        report[f"{name}_base_rmse"] = (
            float(np.sqrt(np.mean(base_error[mask] ** 2))) if mask.any() else np.nan
        )
        report[f"{name}_gain"] = (
            report[f"{name}_base_rmse"] - report[f"{name}_rmse"]
            if mask.any()
            else np.nan
        )
    return report


for path in (POOL_PATH, FOLD_PATH, STATISTICS_PATH, GEOMETRY_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

started = time.time()
pool = _load_pool()
truth = pool["tvt_true"].to_numpy(np.float64)
cp12 = pool["cp12"].to_numpy(np.float64)
gold = _load_gold_endpoint(pool)
if gold is None:
    upstream = cp12
    upstream_name = "cp12_surrogate_no_gold"
    exact_mainline = False
else:
    upstream = gold
    upstream_name = "cp12_gold_oof"
    exact_mainline = True

statistics = pd.read_csv(STATISTICS_PATH, dtype={"well": str})
statistics = statistics.drop_duplicates("well", keep="last")
needed_columns = {"well", "minor_mass", "separation_ft", "midpoint_delta_ft"}
if not needed_columns.issubset(statistics.columns):
    raise RuntimeError(
        f"statistics missing {sorted(needed_columns - set(statistics.columns))}"
    )
wells = sorted(pool["well"].unique())
statistics = (
    pd.DataFrame({"well": wells})
    .merge(statistics, on="well", how="left", validate="one_to_one")
)
if statistics[list(needed_columns - {"well"})].isna().any().any():
    raise RuntimeError("PF statistics do not cover all OOF wells")

candidate = _load_geometry_candidate(len(pool))
eligible = _eligible_neighbor_rows(pool)
neighbor_base = _neighbor_prediction(upstream, candidate, eligible)
md_since = _ensure_md_since(pool)
base_wells = _per_well_sse(pool, truth, neighbor_base, statistics).rename(
    columns={"sse": "base_sse", "rmse": "base_well_rmse", "error_sum": "base_error_sum"}
)
base_sse_by_well = base_wells.set_index("well")["base_sse"]
rows_by_well = base_wells.set_index("well")["rows"]

recipes = []
for placement in PLACEMENTS:
    for alpha in ALPHAS:
        for cap_ft in CAPS_FT:
            for minor_mass in MINOR_MASSES:
                for separation_low_ft in SEPARATION_LOWS_FT:
                    recipes.append(
                        {
                            "placement": placement,
                            "alpha": alpha,
                            "cap_ft": cap_ft,
                            "minor_mass": minor_mass,
                            "separation_low_ft": separation_low_ft,
                        }
                    )

scan_rows = []
recipe_well_sse = {}
recipe_predictions = {}
folds = sorted(pool["fold"].unique())
for number, recipe in enumerate(recipes, 1):
    prediction, qualified, well_shift = _prediction_for_recipe(
        pool, statistics, recipe, upstream, neighbor_base, candidate, eligible
    )
    bywell = _per_well_sse(pool, truth, prediction, statistics)
    sse = bywell.set_index("well")["sse"].reindex(wells).to_numpy(float)
    recipe_well_sse[_recipe_key(recipe)] = sse
    total_sse = float(sse.sum())
    total_rows = int(bywell["rows"].sum())
    base_total_sse = float(base_sse_by_well.sum())
    row = {
        **recipe,
        "recipe": _recipe_name(recipe),
        "qualified_wells": int((np.abs(well_shift) > 0.0).sum()),
        "mean_abs_well_shift": float(np.mean(np.abs(well_shift))),
        "max_abs_well_shift": float(np.max(np.abs(well_shift))),
        "base_rmse": _rmse_from_sse(base_total_sse, total_rows),
        "rmse": _rmse_from_sse(total_sse, total_rows),
        "gain": _rmse_from_sse(base_total_sse, total_rows) - _rmse_from_sse(total_sse, total_rows),
    }
    fold_gains = []
    for fold in folds:
        mask = base_wells["fold"].eq(fold).to_numpy(bool)
        fold_rows = int(base_wells.loc[mask, "rows"].sum())
        fold_base = _rmse_from_sse(base_wells.loc[mask, "base_sse"].sum(), fold_rows)
        fold_prediction = _rmse_from_sse(sse[mask].sum(), fold_rows)
        fold_gains.append(fold_base - fold_prediction)
        row[f"fold{int(fold)}_gain"] = fold_base - fold_prediction
    row["min_fold_gain"] = float(np.min(fold_gains))
    row["positive_folds"] = int(np.sum(np.asarray(fold_gains) > 0.0))
    scan_rows.append(row)
    if number % 50 == 0 or number == len(recipes):
        print(f"recipe scan {number}/{len(recipes)}", flush=True)

scan = pd.DataFrame(scan_rows).sort_values(
    ["gain", "mean_abs_well_shift"], ascending=[False, True]
)
scan.to_csv(OUT_DIR / "recipe_scan.csv", index=False)


def _lookup_recipe(target):
    key = _recipe_key(target)
    matches = scan[
        (scan["placement"] == key[0])
        & np.isclose(scan["alpha"], key[1])
        & np.isclose(scan["cap_ft"], key[2])
        & np.isclose(scan["minor_mass"], key[3])
        & np.isclose(scan["separation_low_ft"], key[4])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"recipe lookup failed: {target}")
    return matches.iloc[0].to_dict()


source_6p594_row = _lookup_recipe(SOURCE_6P594_RECIPE)
public_row = _lookup_recipe(PUBLIC_RECIPE)
conservative_row = _lookup_recipe(CONSERVATIVE_RECIPE)

# Nested selection includes an explicit no-op. A fold can abstain when every
# learned recipe is harmful on the other four folds.
nested_rows = []
nested_prediction = neighbor_base.copy()
well_order = base_wells["well"].tolist()
well_to_position = {well: index for index, well in enumerate(well_order)}
for outer_fold in folds:
    train_well_mask = base_wells["fold"].ne(outer_fold).to_numpy(bool)
    valid_well_mask = ~train_well_mask
    train_rows = int(base_wells.loc[train_well_mask, "rows"].sum())
    train_base_rmse = _rmse_from_sse(
        base_wells.loc[train_well_mask, "base_sse"].sum(), train_rows
    )
    candidates_for_fold = [
        {
            "recipe": "no_op",
            "placement": "after_neighbor",
            "alpha": 0.0,
            "cap_ft": 0.0,
            "minor_mass": 1.0,
            "separation_low_ft": SEPARATION_HIGH_FT,
            "train_gain": 0.0,
            "mean_abs_well_shift": 0.0,
        }
    ]
    for recipe in recipes:
        sse = recipe_well_sse[_recipe_key(recipe)]
        train_rmse = _rmse_from_sse(sse[train_well_mask].sum(), train_rows)
        scan_match = _lookup_recipe(recipe)
        candidates_for_fold.append(
            {
                **recipe,
                "recipe": _recipe_name(recipe),
                "train_gain": train_base_rmse - train_rmse,
                "mean_abs_well_shift": scan_match["mean_abs_well_shift"],
            }
        )
    selected_recipe = max(
        candidates_for_fold,
        key=lambda row: (
            row["train_gain"],
            -row["mean_abs_well_shift"],
            -row["alpha"],
            row["minor_mass"],
            row["separation_low_ft"],
        ),
    )
    valid_rows_mask = pool["fold"].eq(outer_fold).to_numpy(bool)
    if selected_recipe["recipe"] != "no_op":
        prediction, _, _ = _prediction_for_recipe(
            pool,
            statistics,
            selected_recipe,
            upstream,
            neighbor_base,
            candidate,
            eligible,
        )
        nested_prediction[valid_rows_mask] = prediction[valid_rows_mask]
    valid_base_rmse = float(
        np.sqrt(np.mean((neighbor_base[valid_rows_mask] - truth[valid_rows_mask]) ** 2))
    )
    valid_prediction_rmse = float(
        np.sqrt(np.mean((nested_prediction[valid_rows_mask] - truth[valid_rows_mask]) ** 2))
    )
    nested_rows.append(
        {
            "fold": int(outer_fold),
            **selected_recipe,
            "valid_base_rmse": valid_base_rmse,
            "valid_prediction_rmse": valid_prediction_rmse,
            "valid_gain": valid_base_rmse - valid_prediction_rmse,
        }
    )

nested = pd.DataFrame(nested_rows)
nested.to_csv(OUT_DIR / "nested_recipe_folds.csv", index=False)
nested_report = _summarize_prediction(
    pool, truth, nested_prediction, neighbor_base, md_since
)

source_6p594_prediction, _, source_6p594_shift = _prediction_for_recipe(
    pool, statistics, SOURCE_6P594_RECIPE, upstream, neighbor_base, candidate, eligible
)
public_prediction, _, public_shift = _prediction_for_recipe(
    pool, statistics, PUBLIC_RECIPE, upstream, neighbor_base, candidate, eligible
)
conservative_prediction, _, conservative_shift = _prediction_for_recipe(
    pool, statistics, CONSERVATIVE_RECIPE, upstream, neighbor_base, candidate, eligible
)
source_6p594_report = _summarize_prediction(
    pool, truth, source_6p594_prediction, neighbor_base, md_since
)
public_report = _summarize_prediction(
    pool, truth, public_prediction, neighbor_base, md_since
)
conservative_report = _summarize_prediction(
    pool, truth, conservative_prediction, neighbor_base, md_since
)

base_bywell = _per_well_sse(pool, truth, neighbor_base, statistics)
nested_bywell = _per_well_sse(pool, truth, nested_prediction, statistics)
bywell = base_bywell[["well", "fold", "rows", "sse", "rmse"]].rename(
    columns={"sse": "base_sse", "rmse": "base_rmse"}
).merge(
    nested_bywell[["well", "sse", "rmse"]].rename(
        columns={"sse": "nested_sse", "rmse": "nested_rmse"}
    ),
    on="well",
    validate="one_to_one",
)
bywell["gain"] = bywell["base_rmse"] - bywell["nested_rmse"]
bywell = bywell.merge(
    statistics[["well", "minor_mass", "separation_ft", "midpoint_delta_ft"]],
    on="well",
    validate="one_to_one",
)
bywell.sort_values("nested_rmse", ascending=False).to_csv(
    OUT_DIR / "nested_bywell.csv", index=False
)

rng = np.random.default_rng(SEED)
bootstrap = np.empty(BOOTSTRAP_ITERATIONS, np.float64)
base_well_sse = bywell["base_sse"].to_numpy(float)
nested_well_sse = bywell["nested_sse"].to_numpy(float)
well_rows = bywell["rows"].to_numpy(int)
for iteration in range(BOOTSTRAP_ITERATIONS):
    sample = rng.integers(0, len(bywell), len(bywell))
    sampled_rows = int(well_rows[sample].sum())
    bootstrap[iteration] = _rmse_from_sse(
        base_well_sse[sample].sum(), sampled_rows
    ) - _rmse_from_sse(nested_well_sse[sample].sum(), sampled_rows)

# Matched shuffle control for the conservative after-Neighbor recipe. It keeps
# the same shift magnitudes and number of moved wells but assigns them to other
# wells, then recomputes exact row-level SSE.
shuffle_gains = np.empty(SHUFFLE_ITERATIONS, np.float64)
base_rmse = float(np.sqrt(np.mean((neighbor_base - truth) ** 2)))
conservative_shift_map = dict(zip(statistics["well"], conservative_shift))
well_names = statistics["well"].to_numpy(str)
shift_values = np.asarray([conservative_shift_map[well] for well in well_names])
for iteration in range(SHUFFLE_ITERATIONS):
    shuffled = rng.permutation(shift_values)
    shuffled_map = dict(zip(well_names, shuffled))
    row_shift = pool["well"].map(shuffled_map).fillna(0.0).to_numpy(float)
    prediction = neighbor_base + row_shift
    shuffle_gains[iteration] = base_rmse - float(
        np.sqrt(np.mean((prediction - truth) ** 2))
    )

worst_count = max(1, int(math.ceil(0.20 * len(bywell))))
worst = bywell.nlargest(worst_count, "base_rmse")
worst_rows = int(worst["rows"].sum())
worst_base_rmse = _rmse_from_sse(worst["base_sse"].sum(), worst_rows)
worst_nested_rmse = _rmse_from_sse(worst["nested_sse"].sum(), worst_rows)

gates = {
    "nested_gain_ge": 0.03,
    "nested_min_fold_gain_ge": -0.02,
    "nested_positive_folds_ge": 4,
    "bootstrap_p05_gain_gt": 0.0,
    "far_gain_ge": 0.0,
    "conservative_gain_gt_shuffle_p95": True,
}
checks = {
    "nested_gain": bool(nested_report["gain"] >= gates["nested_gain_ge"]),
    "nested_min_fold_gain": bool(
        nested["valid_gain"].min() >= gates["nested_min_fold_gain_ge"]
    ),
    "nested_positive_folds": bool(
        (nested["valid_gain"] > 0.0).sum() >= gates["nested_positive_folds_ge"]
    ),
    "bootstrap_p05": bool(
        np.quantile(bootstrap, 0.05) > gates["bootstrap_p05_gain_gt"]
    ),
    "far_gain": bool(nested_report["far_gt2000_gain"] >= gates["far_gain_ge"]),
    "shuffle": bool(
        conservative_report["gain"] > np.quantile(shuffle_gains, 0.95)
    ),
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "pf_bimodal_midpoint_mainline_sweep_v1",
    "source_statistics": str(STATISTICS_PATH),
    "upstream_endpoint": upstream_name,
    "exact_cp12_gold_neighbor_chain": exact_mainline,
    "gold_oof_path": GOLD_OOF_PATH or None,
    "rows": len(pool),
    "wells": len(wells),
    "neighbor": {
        "candidate": str(GEOMETRY_PATH),
        "weight": NEIGHBOR_WEIGHT,
        "raw_cap_ft": NEIGHBOR_RAW_CAP_FT,
        "eligible_row_fraction": float(eligible.mean()),
    },
    "source_6p594_recipe": {
        "recipe": SOURCE_6P594_RECIPE,
        "scan": source_6p594_row,
        "result": source_6p594_report,
        "note": (
            "Exact branch parameters on the stated OOF endpoint; this does not "
            "claim that the OOF endpoint reproduces the LB artifact"
        ),
    },
    "public_recipe": {"recipe": PUBLIC_RECIPE, "scan": public_row, "result": public_report},
    "conservative_recipe": {
        "recipe": CONSERVATIVE_RECIPE,
        "scan": conservative_row,
        "result": conservative_report,
    },
    "best_same_data_diagnostic": scan.iloc[0].to_dict(),
    "nested": {
        "result": nested_report,
        "folds": nested_rows,
        "positive_folds": int((nested["valid_gain"] > 0.0).sum()),
        "min_fold_gain": float(nested["valid_gain"].min()),
    },
    "worst_20_percent_base_wells": {
        "wells": worst_count,
        "rows": worst_rows,
        "base_rmse": worst_base_rmse,
        "nested_rmse": worst_nested_rmse,
        "gain": worst_base_rmse - worst_nested_rmse,
    },
    "bootstrap": {
        "iterations": BOOTSTRAP_ITERATIONS,
        "p01_gain": float(np.quantile(bootstrap, 0.01)),
        "p05_gain": float(np.quantile(bootstrap, 0.05)),
        "p50_gain": float(np.quantile(bootstrap, 0.50)),
        "p95_gain": float(np.quantile(bootstrap, 0.95)),
        "positive_probability": float(np.mean(bootstrap > 0.0)),
    },
    "shuffle_control": {
        "iterations": SHUFFLE_ITERATIONS,
        "p50_gain": float(np.quantile(shuffle_gains, 0.50)),
        "p95_gain": float(np.quantile(shuffle_gains, 0.95)),
        "conservative_gain_gt_shuffle_p95": bool(
            conservative_report["gain"] > np.quantile(shuffle_gains, 0.95)
        ),
    },
    "gates": gates,
    "checks": checks,
    "strict_pass": strict_pass,
    "interpretation": (
        "Exact mainline OOF" if exact_mainline else "CP12+Neighbor surrogate; Gold is absent"
    ),
    "next": (
        "PASS -> freeze the nested consensus recipe and build one isolated Kaggle endpoint"
        if strict_pass
        else "FAIL -> do not add midpoint hedge to the 6.781 mainline"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

_write_json(OUT_DIR / "verdict.json", verdict)
pd.DataFrame(
    [
        {"variant": "source_6p594_exact_recipe", **source_6p594_report},
        {"variant": "public", **public_report},
        {"variant": "conservative", **conservative_report},
        {"variant": "nested", **nested_report},
    ]
).to_csv(OUT_DIR / "primary_results.csv", index=False)
print("\nTOP RECIPE SCAN")
print(scan.head(30).to_string(index=False))
print("\nNESTED FOLDS")
print(nested.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT_DIR)
