"""Deployable cross-well visible-prefix spatial move, evaluated on honest OOF.

The 773 train wells are split into three pseudo-hidden cohorts (~258 wells each),
so each target only sees prefixes from a test-sized peer set rather than all 772
other wells. Only test-visible columns X/Y/Z/TVT_input are used to build moves.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
POOL_CANDIDATES = [
    OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_DIR / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((p for p in POOL_CANDIDATES if p.exists()), None)
if POOL_PATH is None:
    raise RuntimeError(f"Cannot find candidate pool in {OOF_DIR}")
if not (DATA_ROOT / "train").exists():
    raise RuntimeError(f"Missing train data: {DATA_ROOT / 'train'}")

PREFIX_EVERY = 20
N_COHORTS = 3       # 773 / 3 ~= 258, close to 285 hidden wells
IDW_K = 64
QUERY_K = 192       # enough room to remove the target well's own prefix points
RANDOM_STATE = 42

# Fixed, deliberately small scan. cap is applied before alpha.
CONFIGS = [
    {"name": "tiny_all",       "alpha": 0.10, "cap": 4.0, "ramp": 500,  "max_dist": np.inf},
    {"name": "safe_all",       "alpha": 0.25, "cap": 4.0, "ramp": 500,  "max_dist": np.inf},
    {"name": "balanced_all",   "alpha": 0.50, "cap": 4.0, "ramp": 500,  "max_dist": np.inf},
    {"name": "safe_d2500",     "alpha": 0.25, "cap": 4.0, "ramp": 500,  "max_dist": 2500.0},
    {"name": "balanced_d2500", "alpha": 0.50, "cap": 4.0, "ramp": 500,  "max_dist": 2500.0},
    {"name": "balanced_slow",  "alpha": 0.50, "cap": 4.0, "ramp": 1000, "max_dist": 2500.0},
    {"name": "wide_d4000",     "alpha": 0.50, "cap": 6.0, "ramp": 500,  "max_dist": 4000.0},
]

OOF_DIR.mkdir(parents=True, exist_ok=True)
print("POOL:", POOL_PATH)
print(
    {
        "prefix_every": PREFIX_EVERY,
        "cohorts": N_COHORTS,
        "idw_k": IDW_K,
        "query_k": QUERY_K,
        "configs": len(CONFIGS),
    }
)


def rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


# -----------------------------------------------------------------------------
# Load only the OOF columns needed for this test
# -----------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
base_cols = [c for c in ("cp12", "strong", "strong_bias") if c in header]
if "cp12" not in base_cols:
    raise RuntimeError(f"cp12 missing from pool columns: {header}")
usecols = ["id", "well", "tvt_true", *base_cols]
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = (
    pool["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int32)
)
print("pool:", pool.shape, "wells:", pool["well"].nunique(), "bases:", base_cols)

wells = sorted(pool["well"].unique())
well_code = {wid: i for i, wid in enumerate(wells)}

# Deterministic pseudo-hidden cohorts, independent of target values.
rng = np.random.default_rng(RANDOM_STATE)
shuffled = np.asarray(wells, dtype=object).copy()
rng.shuffle(shuffled)
cohort_map = {wid: int(i % N_COHORTS) for i, wid in enumerate(shuffled)}

cohort_df = pd.DataFrame(
    {"well": wells, "cohort": [cohort_map[w] for w in wells]}
)
cohort_df.to_csv(OOF_DIR / "spatial_prefix_pseudo_hidden_cohorts.csv", index=False)
print("cohort sizes:", cohort_df["cohort"].value_counts().sort_index().to_dict())


# -----------------------------------------------------------------------------
# Build per-cohort visible-prefix residual fields
# residual = (TVT_input + Z) - its own well's final visible datum
# -----------------------------------------------------------------------------
source_parts = {c: {"xy": [], "resid": [], "well_code": []} for c in range(N_COHORTS)}
well_anchor_row = {}

for i, wid in enumerate(tqdm(wells, desc="build prefix fields"), 1):
    path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    d = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT_input"])
    known = np.flatnonzero(d["TVT_input"].notna().to_numpy())
    if len(known) == 0:
        continue
    well_anchor_row[wid] = int(known[-1])
    take = known[::PREFIX_EVERY]
    if take[-1] != known[-1]:
        take = np.append(take, known[-1])
    xy = d[["X", "Y"]].to_numpy(dtype=float)[take]
    datum = d["TVT_input"].to_numpy(dtype=float)[take] + d["Z"].to_numpy(dtype=float)[take]
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(datum)
    xy = xy[ok]
    datum = datum[ok]
    if len(datum) == 0:
        continue
    cohort = cohort_map[wid]
    source_parts[cohort]["xy"].append(xy)
    source_parts[cohort]["resid"].append(datum - datum[-1])
    source_parts[cohort]["well_code"].append(
        np.full(len(datum), well_code[wid], dtype=np.int32)
    )

cohort_sources = {}
for cohort, parts in source_parts.items():
    xy = np.concatenate(parts["xy"])
    resid = np.concatenate(parts["resid"])
    codes = np.concatenate(parts["well_code"])
    cohort_sources[cohort] = {
        "xy": xy,
        "resid": resid,
        "well_code": codes,
        "tree": cKDTree(xy, leafsize=64),
    }
    print(f"cohort {cohort}: prefix points={len(xy)}")


# -----------------------------------------------------------------------------
# Query a target-free spatial movement for every OOF row
# -----------------------------------------------------------------------------
raw_move = np.zeros(len(pool), dtype=np.float32)
nearest_other = np.full(len(pool), np.inf, dtype=np.float32)
anchor_nearest = np.full(len(pool), np.inf, dtype=np.float32)
distance_from_anchor = np.zeros(len(pool), dtype=np.float32)
failures = []

pool_indices = pool.groupby("well", sort=False).indices
t0 = time.time()

for n_done, wid in enumerate(tqdm(wells, desc="query spatial moves"), 1):
    try:
        out_idx = np.asarray(pool_indices[wid], dtype=np.int64)
        row_idx = pool["row_idx"].to_numpy()[out_idx]
        path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
        d = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT_input"])
        anchor = well_anchor_row[wid]
        query_rows = np.concatenate(([anchor], row_idx))
        query_xy = d[["X", "Y"]].to_numpy(dtype=float)[query_rows]

        source = cohort_sources[cohort_map[wid]]
        query_k = min(QUERY_K, len(source["xy"]))
        dist, idx = source["tree"].query(query_xy, k=query_k, workers=-1)
        if query_k == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        # Remove all prefix points from the target itself.
        masked_dist = np.where(
            source["well_code"][idx] != well_code[wid], dist, np.inf
        )
        k = min(IDW_K, masked_dist.shape[1])
        take = np.argpartition(masked_dist, kth=k - 1, axis=1)[:, :k]
        chosen_dist = np.take_along_axis(masked_dist, take, axis=1)
        chosen_idx = np.take_along_axis(idx, take, axis=1)
        valid = np.isfinite(chosen_dist)
        weights = np.where(valid, 1.0 / np.maximum(chosen_dist, 10.0) ** 2, 0.0)
        denom = weights.sum(axis=1)
        field = np.zeros(len(query_rows), dtype=np.float64)
        good = denom > 0
        field[good] = (
            weights[good] * source["resid"][chosen_idx[good]]
        ).sum(axis=1) / denom[good]

        move = field[1:] - field[0]
        move[~np.isfinite(move)] = 0.0
        nearest = np.min(masked_dist, axis=1)

        raw_move[out_idx] = move.astype(np.float32)
        nearest_other[out_idx] = nearest[1:].astype(np.float32)
        anchor_nearest[out_idx] = np.float32(nearest[0])
        distance_from_anchor[out_idx] = np.maximum(row_idx - anchor, 0).astype(np.float32)
    except Exception as exc:
        failures.append({"well": wid, "reason": repr(exc)})

print("move build elapsed_sec:", time.time() - t0)
print("failures:", len(failures))
if failures:
    display(pd.DataFrame(failures).head(30))
if len(failures) > 0.05 * len(wells):
    raise RuntimeError("Too many spatial-move failures")

print(
    "raw move stats:",
    {
        "mean_abs": float(np.mean(np.abs(raw_move))),
        "p95_abs": float(np.quantile(np.abs(raw_move), 0.95)),
        "max_abs": float(np.max(np.abs(raw_move))),
        "nearest_median": float(np.median(nearest_other[np.isfinite(nearest_other)])),
        "nearest_p90": float(np.quantile(nearest_other[np.isfinite(nearest_other)], 0.90)),
    },
)


# -----------------------------------------------------------------------------
# Fixed five-fold stability labels and top-40 definition
# -----------------------------------------------------------------------------
well_meta = (
    pool.groupby("well", as_index=False)
    .size()
    .rename(columns={"size": "rows"})
)
well_meta["stratum"] = pd.qcut(
    well_meta["rows"].rank(method="first"), q=10, labels=False
).astype(int)
well_meta["fold"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (_, valid) in enumerate(skf.split(well_meta, well_meta["stratum"]), 1):
    well_meta.loc[valid, "fold"] = fold
fold_map = dict(zip(well_meta["well"], well_meta["fold"]))
fold_values = pool["well"].map(fold_map).to_numpy(dtype=np.int8)

y = pool["tvt_true"].to_numpy(dtype=np.float64)
cp12 = pool["cp12"].to_numpy(dtype=np.float64)
pool["_cp12_sse"] = (y - cp12) ** 2
well_sse = pool.groupby("well", as_index=False)["_cp12_sse"].sum()
top40_wells = set(well_sse.nlargest(40, "_cp12_sse")["well"])
top40_mask = pool["well"].isin(top40_wells).to_numpy()
pool.drop(columns=["_cp12_sse"], inplace=True)


# -----------------------------------------------------------------------------
# Apply the fixed small scan to CP12 (primary) and other available bases
# -----------------------------------------------------------------------------
records = []
fold_records = []

for base_col in base_cols:
    base = pool[base_col].to_numpy(dtype=np.float64)
    base_score = rmse(y, base)
    top_base_score = rmse(y[top40_mask], base[top40_mask])

    for config in CONFIGS:
        ramp = 1.0 - np.exp(
            -distance_from_anchor.astype(np.float64) / float(config["ramp"])
        )
        eligible = np.ones(len(pool), dtype=bool)
        if np.isfinite(config["max_dist"]):
            eligible &= nearest_other <= config["max_dist"]
            eligible &= anchor_nearest <= config["max_dist"]

        move = (
            float(config["alpha"])
            * np.clip(raw_move.astype(np.float64), -config["cap"], config["cap"])
            * ramp
            * eligible
        )
        pred = base + move
        score = rmse(y, pred)
        top_score = rmse(y[top40_mask], pred[top40_mask])

        gains = []
        for fold in range(1, 6):
            mask = fold_values == fold
            fold_base = rmse(y[mask], base[mask])
            fold_score = rmse(y[mask], pred[mask])
            gain = fold_base - fold_score
            gains.append(gain)
            fold_records.append(
                {
                    "base_col": base_col,
                    "config": config["name"],
                    "fold": fold,
                    "base_rmse": fold_base,
                    "rmse": fold_score,
                    "gain": gain,
                    "rows": int(mask.sum()),
                }
            )

        records.append(
            {
                "base_col": base_col,
                "config": config["name"],
                "alpha": config["alpha"],
                "cap": config["cap"],
                "ramp": config["ramp"],
                "max_dist": config["max_dist"],
                "base_rmse": base_score,
                "rmse": score,
                "gain": base_score - score,
                "mean_fold_gain": float(np.mean(gains)),
                "min_fold_gain": float(np.min(gains)),
                "positive_folds": int(np.sum(np.asarray(gains) > 0)),
                "top40_base_rmse": top_base_score,
                "top40_rmse": top_score,
                "top40_gain": top_base_score - top_score,
                "selected_rows": int(eligible.sum()),
                "mean_abs_move": float(np.mean(np.abs(move))),
                "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
            }
        )

summary = pd.DataFrame(records).sort_values(["base_col", "rmse"]).reset_index(drop=True)
summary["strict_gate"] = (
    (summary["gain"] >= 0.10)
    & (summary["min_fold_gain"] > 0)
    & (summary["positive_folds"] == 5)
    & (summary["top40_gain"] > 0)
)
fold_df = pd.DataFrame(fold_records)

print("\nSPATIAL PREFIX MOVE SCAN")
display(summary.head(40))
print("\nCP12 RESULTS")
display(summary[summary["base_col"] == "cp12"])

summary.to_csv(OOF_DIR / "spatial_prefix_move_cp12_scan.csv", index=False)
fold_df.to_csv(OOF_DIR / "spatial_prefix_move_cp12_folds.csv", index=False)

passing = summary[(summary["base_col"] == "cp12") & summary["strict_gate"]]
print("STRICT CP12 PASS:", passing[["config", "gain", "min_fold_gain", "top40_gain"]].to_dict("records"))

if len(passing):
    best = passing.sort_values("rmse").iloc[0]
    best_config = next(c for c in CONFIGS if c["name"] == best["config"])
    best_ramp = 1.0 - np.exp(
        -distance_from_anchor.astype(np.float64) / float(best_config["ramp"])
    )
    best_eligible = np.ones(len(pool), dtype=bool)
    if np.isfinite(best_config["max_dist"]):
        best_eligible &= nearest_other <= best_config["max_dist"]
        best_eligible &= anchor_nearest <= best_config["max_dist"]
    best_move = (
        float(best_config["alpha"])
        * np.clip(raw_move.astype(np.float64), -best_config["cap"], best_config["cap"])
        * best_ramp
        * best_eligible
    )
    out = pool[["id", "well", "tvt_true", "cp12"]].copy()
    out["pred_spatial_prefix_move"] = (cp12 + best_move).astype(np.float32)
    out.to_pickle(OOF_DIR / "spatial_prefix_move_cp12_best_oof.pkl")
    print("saved passing OOF:", OOF_DIR / "spatial_prefix_move_cp12_best_oof.pkl")
else:
    print("No qualifying spatial signal. Do not deploy or add it to Gold.")
