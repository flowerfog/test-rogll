"""Robustness audit for the frozen safe neighbor-transfer configuration.

Consumes neighbor_fullcurve_transfer_v0 outputs; it does not rebuild neighbors.
The frozen rule is selected before this audit:

    strong + 0.25 * clip(structure_progress - strong, -24, 24)

applied only to coherent600 wells. The applied movement is therefore capped at
6 ft. This cell measures well-level concentration, bootstrap uncertainty, fold
stability, and nested grid-selection behavior.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / "neighbor_fullcurve_frozen_robustness_v0"

BASE_COL = "strong"
METHOD = "structure_progress"
GATE = "coherent600"
WEIGHT = 0.25
RAW_MOVE_CAP = 24.0
APPLIED_MOVE_CAP = WEIGHT * RAW_MOVE_CAP
MIN_ABS_DIRECTION_DOT = 0.70

N_FOLDS = 5
RANDOM_STATE = 20260715
N_BOOTSTRAP = 4000

for path in (
    SOURCE_DIR / "neighbor_transfer_oof.npz",
    SOURCE_DIR / "selected_neighbors.csv",
    SOURCE_DIR / "neighbor_transfer_scan.csv",
    SOURCE_DIR / "neighbor_transfer_folds.csv",
):
    if not path.exists():
        raise FileNotFoundError(path)
if POOL_PATH is None:
    raise FileNotFoundError(f"candidate pool not found under {OOF_ROOT}")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


print(
    {
        "experiment": "neighbor_fullcurve_frozen_robustness_v0",
        "source": str(SOURCE_DIR),
        "pool": str(POOL_PATH),
        "frozen": {
            "base": BASE_COL,
            "method": METHOD,
            "gate": GATE,
            "weight": WEIGHT,
            "raw_move_cap": RAW_MOVE_CAP,
            "applied_move_cap": APPLIED_MOVE_CAP,
        },
    }
)


# -----------------------------------------------------------------------------
# Reconstruct the exact sorted row order used by the source experiment
# -----------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", BASE_COL) if column in header]
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)

sidecar = np.load(SOURCE_DIR / "neighbor_transfer_oof.npz", allow_pickle=False)
candidate = sidecar[METHOD].astype(np.float64)
if len(candidate) != len(pool):
    raise RuntimeError(f"sidecar/pool row mismatch: {len(candidate)} != {len(pool)}")

neighbor = pd.read_csv(SOURCE_DIR / "selected_neighbors.csv")
neighbor["well"] = neighbor["well"].astype(str)
neighbor["neighbor"] = neighbor["neighbor"].astype(str)
meta = neighbor.set_index("well")

median_distance = pool["well"].map(meta["median_distance"]).to_numpy(np.float64)
p90_distance = pool["well"].map(meta["p90_distance"]).to_numpy(np.float64)
anchor_distance = pool["well"].map(meta["anchor_distance_full"]).to_numpy(np.float64)
direction_dot = pool["well"].map(meta["direction_dot"]).to_numpy(np.float64)

eligible = (
    (median_distance <= 600.0)
    & (p90_distance <= 1200.0)
    & (anchor_distance <= 1200.0)
    & (np.abs(direction_dot) >= MIN_ABS_DIRECTION_DOT)
    & np.isfinite(candidate)
)

truth = pool["tvt_true"].to_numpy(np.float64)
base = pool[BASE_COL].to_numpy(np.float64)
raw_delta = candidate - base
move = np.zeros(len(pool), np.float64)
move[eligible] = WEIGHT * np.clip(raw_delta[eligible], -RAW_MOVE_CAP, RAW_MOVE_CAP)
prediction = base + move

if float(np.max(np.abs(move))) > APPLIED_MOVE_CAP + 1e-6:
    raise RuntimeError("applied move exceeds frozen cap")


# -----------------------------------------------------------------------------
# Reuse the established fold map when available
# -----------------------------------------------------------------------------
wells = sorted(pool["well"].unique())
fold_source_candidates = [
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "well_metadata.csv",
]
fold_map = None
fold_source = None
for path in fold_source_candidates:
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    if {"well", "fold"}.issubset(frame.columns):
        frame["well"] = frame["well"].astype(str)
        current = frame.groupby("well")["fold"].first().astype(int).to_dict()
        if set(wells).issubset(current):
            fold_map = current
            fold_source = str(path)
            break

if fold_map is None:
    counts = pool.groupby("well", as_index=False).size().rename(columns={"size": "rows"})
    counts["stratum"] = pd.qcut(
        counts["rows"].rank(method="first"), q=10, labels=False
    ).astype(int)
    counts["fold"] = -1
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, valid) in enumerate(splitter.split(counts, counts["stratum"]), 1):
        counts.loc[valid, "fold"] = fold
    fold_map = dict(zip(counts["well"].astype(str), counts["fold"].astype(int)))
    fold_source = "deterministic row-count StratifiedKFold"

fold_values = pool["well"].map(fold_map).to_numpy(np.int8)
print("fold source:", fold_source)


# -----------------------------------------------------------------------------
# Per-well and grouped diagnostics
# -----------------------------------------------------------------------------
pool["base_se"] = (base - truth) ** 2
pool["prediction_se"] = (prediction - truth) ** 2
pool["abs_move"] = np.abs(move)
pool["eligible"] = eligible

per_well = (
    pool.groupby("well", as_index=False)
    .agg(
        rows=("id", "size"),
        base_sse=("base_se", "sum"),
        prediction_sse=("prediction_se", "sum"),
        eligible_rows=("eligible", "sum"),
        mean_abs_move=("abs_move", "mean"),
        max_abs_move=("abs_move", "max"),
    )
)
per_well["base_rmse"] = np.sqrt(per_well["base_sse"] / per_well["rows"])
per_well["prediction_rmse"] = np.sqrt(per_well["prediction_sse"] / per_well["rows"])
per_well["gain"] = per_well["base_rmse"] - per_well["prediction_rmse"]
per_well["sse_gain"] = per_well["base_sse"] - per_well["prediction_sse"]
per_well["fold"] = per_well["well"].map(fold_map).astype(int)
per_well = per_well.merge(
    neighbor[
        [
            "well",
            "neighbor",
            "min_distance",
            "median_distance",
            "p90_distance",
            "anchor_distance_full",
            "direction_dot",
            "abs_direction_dot",
            "progress_coverage",
        ]
    ],
    on="well",
    how="left",
    validate="one_to_one",
)
per_well["direction_relation"] = np.where(
    per_well["direction_dot"] < 0, "opposite", "same"
)
per_well["distance_bin"] = pd.cut(
    per_well["median_distance"],
    bins=[-np.inf, 150.0, 300.0, 600.0, np.inf],
    labels=["lt150", "150to300", "300to600", "gt600"],
).astype(str)
per_well.to_csv(OUT_DIR / "frozen_per_well.csv", index=False)

overall = {
    "rows": int(len(pool)),
    "wells": int(len(per_well)),
    "eligible_wells": int((per_well["eligible_rows"] > 0).sum()),
    "eligible_rows": int(eligible.sum()),
    "base_rmse": _rmse(truth, base),
    "prediction_rmse": _rmse(truth, prediction),
    "gain": _rmse(truth, base) - _rmse(truth, prediction),
    "positive_wells": int((per_well["gain"] > 0).sum()),
    "negative_wells": int((per_well["gain"] < 0).sum()),
    "positive_fraction_eligible": float(
        (per_well.loc[per_well["eligible_rows"] > 0, "gain"] > 0).mean()
    ),
    "mean_abs_move_all": float(np.mean(np.abs(move))),
    "p95_abs_move_all": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
}

fold_rows = []
for fold in range(1, N_FOLDS + 1):
    mask = fold_values == fold
    fold_rows.append(
        {
            "fold": fold,
            "rows": int(mask.sum()),
            "wells": int(np.unique(pool.loc[mask, "well"]).size),
            "base_rmse": _rmse(truth[mask], base[mask]),
            "prediction_rmse": _rmse(truth[mask], prediction[mask]),
            "gain": _rmse(truth[mask], base[mask]) - _rmse(truth[mask], prediction[mask]),
        }
    )
fold_report = pd.DataFrame(fold_rows)
fold_report.to_csv(OUT_DIR / "frozen_folds.csv", index=False)


def _group_score(group_columns):
    rows = []
    grouped = per_well.groupby(group_columns, dropna=False, sort=False)
    for key, frame in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row.update(
            {
                "wells": int(len(frame)),
                "rows": int(frame["rows"].sum()),
                "base_rmse": float(np.sqrt(frame["base_sse"].sum() / frame["rows"].sum())),
                "prediction_rmse": float(
                    np.sqrt(frame["prediction_sse"].sum() / frame["rows"].sum())
                ),
                "gain": float(
                    np.sqrt(frame["base_sse"].sum() / frame["rows"].sum())
                    - np.sqrt(frame["prediction_sse"].sum() / frame["rows"].sum())
                ),
                "positive_well_fraction": float((frame["gain"] > 0).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


subgroup_report = pd.concat(
    [
        _group_score(["distance_bin"]).assign(grouping="distance_bin"),
        _group_score(["direction_relation"]).assign(grouping="direction_relation"),
        _group_score(["fold"]).assign(grouping="fold"),
    ],
    ignore_index=True,
    sort=False,
)
subgroup_report.to_csv(OUT_DIR / "frozen_subgroups.csv", index=False)


# -----------------------------------------------------------------------------
# Well bootstrap and concentration stress tests
# -----------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
n_wells = len(per_well)
well_rows = per_well["rows"].to_numpy(np.int64)
well_base_sse = per_well["base_sse"].to_numpy(np.float64)
well_prediction_sse = per_well["prediction_sse"].to_numpy(np.float64)
bootstrap_gain = np.empty(N_BOOTSTRAP, np.float64)

for iteration in range(N_BOOTSTRAP):
    sample = rng.integers(0, n_wells, size=n_wells)
    rows = int(well_rows[sample].sum())
    bootstrap_gain[iteration] = (
        np.sqrt(well_base_sse[sample].sum() / rows)
        - np.sqrt(well_prediction_sse[sample].sum() / rows)
    )

bootstrap_summary = {
    "iterations": N_BOOTSTRAP,
    "mean_gain": float(np.mean(bootstrap_gain)),
    "p01_gain": float(np.quantile(bootstrap_gain, 0.01)),
    "p05_gain": float(np.quantile(bootstrap_gain, 0.05)),
    "p50_gain": float(np.quantile(bootstrap_gain, 0.50)),
    "p95_gain": float(np.quantile(bootstrap_gain, 0.95)),
    "p99_gain": float(np.quantile(bootstrap_gain, 0.99)),
    "positive_probability": float(np.mean(bootstrap_gain > 0)),
}

ranked = per_well.sort_values("sse_gain", ascending=False).reset_index(drop=True)
positive_sse = ranked.loc[ranked["sse_gain"] > 0, "sse_gain"].sum()
concentration_rows = []
for top_n in (1, 5, 10, 20, 40):
    removed = set(ranked.head(top_n)["well"])
    kept = per_well[~per_well["well"].isin(removed)]
    rows = int(kept["rows"].sum())
    gain = float(
        np.sqrt(kept["base_sse"].sum() / rows)
        - np.sqrt(kept["prediction_sse"].sum() / rows)
    )
    concentration_rows.append(
        {
            "removed_top_positive_wells": top_n,
            "remaining_wells": int(len(kept)),
            "remaining_gain": gain,
            "top_share_of_positive_sse_gain": float(
                ranked.head(top_n)["sse_gain"].clip(lower=0).sum() / max(positive_sse, 1e-12)
            ),
        }
    )
concentration_report = pd.DataFrame(concentration_rows)
concentration_report.to_csv(OUT_DIR / "frozen_gain_concentration.csv", index=False)


# -----------------------------------------------------------------------------
# Nested selection audit over the original fixed scan
# -----------------------------------------------------------------------------
scan = pd.read_csv(SOURCE_DIR / "neighbor_transfer_scan.csv")
grid_folds = pd.read_csv(SOURCE_DIR / "neighbor_transfer_folds.csv")
grid = grid_folds[
    grid_folds["base_col"].eq(BASE_COL)
    & grid_folds["method"].eq(METHOD)
    & grid_folds["gate"].eq(GATE)
].copy()
key_columns = ["base_col", "method", "gate", "weight", "move_cap_ft"]
nested_rows = []

for held_fold in range(1, N_FOLDS + 1):
    train = grid[grid["fold"].ne(held_fold)]
    train_scores = (
        train.groupby(key_columns, as_index=False)
        .agg(train_mean_gain=("gain", "mean"), train_min_gain=("gain", "min"))
        .sort_values(["train_min_gain", "train_mean_gain"], ascending=False)
    )
    selected = train_scores.iloc[0]
    match = grid[grid["fold"].eq(held_fold)].copy()
    for column in key_columns:
        match = match[match[column].astype(str).eq(str(selected[column]))]
    if len(match) != 1:
        raise RuntimeError(f"nested config lookup failed for fold {held_fold}: {len(match)}")
    valid = match.iloc[0]
    nested_rows.append(
        {
            "fold": held_fold,
            "selected_weight": selected["weight"],
            "selected_move_cap_ft": selected["move_cap_ft"],
            "train_mean_gain": selected["train_mean_gain"],
            "train_min_gain": selected["train_min_gain"],
            "valid_gain": valid["gain"],
            "valid_base_rmse": valid["base_rmse"],
            "valid_rmse": valid["rmse"],
        }
    )
nested_report = pd.DataFrame(nested_rows)
nested_report.to_csv(OUT_DIR / "nested_grid_selection.csv", index=False)
nested_summary = {
    "mean_valid_gain": float(nested_report["valid_gain"].mean()),
    "min_valid_gain": float(nested_report["valid_gain"].min()),
    "positive_valid_folds": int((nested_report["valid_gain"] > 0).sum()),
}


strict_pass = bool(
    overall["gain"] >= 0.25
    and fold_report["gain"].min() > 0.0
    and bootstrap_summary["p05_gain"] >= 0.10
    and concentration_report.loc[
        concentration_report["removed_top_positive_wells"].eq(20), "remaining_gain"
    ].iloc[0]
    >= 0.10
    and nested_summary["positive_valid_folds"] == N_FOLDS
    and nested_summary["min_valid_gain"] > 0.0
)

verdict = {
    "frozen": {
        "base": BASE_COL,
        "method": METHOD,
        "gate": GATE,
        "weight": WEIGHT,
        "raw_move_cap": RAW_MOVE_CAP,
        "applied_move_cap": APPLIED_MOVE_CAP,
    },
    "overall": overall,
    "min_fold_gain": float(fold_report["gain"].min()),
    "positive_folds": int((fold_report["gain"] > 0).sum()),
    "bootstrap": bootstrap_summary,
    "nested": nested_summary,
    "remaining_gain_after_top20_removed": float(
        concentration_report.loc[
            concentration_report["removed_top_positive_wells"].eq(20), "remaining_gain"
        ].iloc[0]
    ),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build deterministic Kaggle neighbor-transfer inference and hidden coverage guard"
        if strict_pass
        else "FAIL -> do not promote neighbor transfer"
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, ensure_ascii=True)

print("\nFROZEN OVERALL")
print(overall)
print("\nFOLDS")
display(fold_report)
print("\nSUBGROUPS")
display(subgroup_report)
print("\nBOOTSTRAP")
print(bootstrap_summary)
print("\nGAIN CONCENTRATION")
display(concentration_report)
print("\nNESTED GRID SELECTION")
display(nested_report)
print(nested_summary)
print("\nVERDICT", verdict)
print("saved:", OUT_DIR)
