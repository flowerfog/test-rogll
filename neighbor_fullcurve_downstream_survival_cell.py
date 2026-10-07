"""Frozen neighbor-move survival audit across historical downstream stages.

No parameter is selected here. The already frozen strong-stage movement is
carried unchanged onto strong_bias and CP12, then compared with re-centering the
same neighbor candidate around each stage. This determines the eventual
deployment insertion point if the standalone LB probe confirms transfer.
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
OUT_DIR = OOF_ROOT / "neighbor_fullcurve_downstream_survival_v0"

WEIGHT = 0.25
RAW_MOVE_CAP = 24.0
MIN_ABS_DIRECTION_DOT = 0.70
N_FOLDS = 5
RANDOM_STATE = 20260715

if POOL_PATH is None:
    raise FileNotFoundError(f"candidate pool not found under {OOF_ROOT}")
for path in (
    SOURCE_DIR / "neighbor_transfer_oof.npz",
    SOURCE_DIR / "selected_neighbors.csv",
):
    if not path.exists():
        raise FileNotFoundError(path)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
stage_cols = [column for column in ("strong", "strong_bias", "cp12") if column in header]
if "strong" not in stage_cols:
    raise RuntimeError(f"strong missing from pool: {header}")

usecols = [column for column in ("id", "well", "row_idx", "tvt_true", *stage_cols) if column in header]
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
neighbor_candidate = sidecar["structure_progress"].astype(np.float64)
if len(neighbor_candidate) != len(pool):
    raise RuntimeError("neighbor sidecar row mismatch")

neighbor = pd.read_csv(SOURCE_DIR / "selected_neighbors.csv")
neighbor["well"] = neighbor["well"].astype(str)
meta = neighbor.set_index("well")
median_distance = pool["well"].map(meta["median_distance"]).to_numpy(np.float64)
p90_distance = pool["well"].map(meta["p90_distance"]).to_numpy(np.float64)
anchor_distance = pool["well"].map(meta["anchor_distance_full"]).to_numpy(np.float64)
abs_direction_dot = pool["well"].map(meta["abs_direction_dot"]).to_numpy(np.float64)
eligible = (
    (median_distance <= 600.0)
    & (p90_distance <= 1200.0)
    & (anchor_distance <= 1200.0)
    & (abs_direction_dot >= MIN_ABS_DIRECTION_DOT)
    & np.isfinite(neighbor_candidate)
)

truth = pool["tvt_true"].to_numpy(np.float64)
strong = pool["strong"].to_numpy(np.float64)
frozen_move = np.zeros(len(pool), np.float64)
frozen_move[eligible] = WEIGHT * np.clip(
    neighbor_candidate[eligible] - strong[eligible],
    -RAW_MOVE_CAP,
    RAW_MOVE_CAP,
)

predictions = {"strong_plus_frozen_move": strong + frozen_move}
variant_meta = {
    "strong_plus_frozen_move": {
        "base_col": "strong",
        "move_mode": "same frozen strong-stage move",
    }
}

for stage in ("strong_bias", "cp12"):
    if stage not in stage_cols:
        continue
    base = pool[stage].to_numpy(np.float64)
    predictions[f"{stage}_plus_frozen_move"] = base + frozen_move
    variant_meta[f"{stage}_plus_frozen_move"] = {
        "base_col": stage,
        "move_mode": "same frozen strong-stage move",
    }

    recentered = np.zeros(len(pool), np.float64)
    recentered[eligible] = WEIGHT * np.clip(
        neighbor_candidate[eligible] - base[eligible],
        -RAW_MOVE_CAP,
        RAW_MOVE_CAP,
    )
    predictions[f"{stage}_recentered_move"] = base + recentered
    variant_meta[f"{stage}_recentered_move"] = {
        "base_col": stage,
        "move_mode": f"same candidate re-centered around {stage}",
    }


# Established folds when available
wells = sorted(pool["well"].unique())
fold_candidates = [
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
]
fold_map = None
fold_source = None
for path in fold_candidates:
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
    counts["stratum"] = pd.qcut(counts["rows"].rank(method="first"), 10, labels=False).astype(int)
    counts["fold"] = -1
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, valid) in enumerate(splitter.split(counts, counts["stratum"]), 1):
        counts.loc[valid, "fold"] = fold
    fold_map = dict(zip(counts["well"].astype(str), counts["fold"].astype(int)))
    fold_source = "deterministic row-count StratifiedKFold"
fold_values = pool["well"].map(fold_map).to_numpy(np.int8)


score_rows = []
fold_rows = []
for variant, prediction in predictions.items():
    base_col = variant_meta[variant]["base_col"]
    base = pool[base_col].to_numpy(np.float64)
    base_error = base - truth
    well_sse = (
        pd.DataFrame({"well": pool["well"], "se": base_error * base_error})
        .groupby("well", as_index=False)["se"]
        .sum()
    )
    top40_wells = set(well_sse.nlargest(40, "se")["well"].astype(str))
    top40 = pool["well"].isin(top40_wells).to_numpy()

    gains = []
    for fold in range(1, N_FOLDS + 1):
        mask = fold_values == fold
        base_rmse = _rmse(truth[mask], base[mask])
        prediction_rmse = _rmse(truth[mask], prediction[mask])
        gain = base_rmse - prediction_rmse
        gains.append(gain)
        fold_rows.append(
            {
                "variant": variant,
                "base_col": base_col,
                "fold": fold,
                "rows": int(mask.sum()),
                "base_rmse": base_rmse,
                "prediction_rmse": prediction_rmse,
                "gain": gain,
            }
        )

    base_rmse = _rmse(truth, base)
    prediction_rmse = _rmse(truth, prediction)
    top40_base = _rmse(truth[top40], base[top40])
    top40_prediction = _rmse(truth[top40], prediction[top40])
    applied_move = prediction - base
    score_rows.append(
        {
            "variant": variant,
            "base_col": base_col,
            "move_mode": variant_meta[variant]["move_mode"],
            "base_rmse": base_rmse,
            "prediction_rmse": prediction_rmse,
            "gain": base_rmse - prediction_rmse,
            "mean_fold_gain": float(np.mean(gains)),
            "min_fold_gain": float(np.min(gains)),
            "positive_folds": int(np.sum(np.asarray(gains) > 0)),
            "top40_gain": top40_base - top40_prediction,
            "mean_abs_move": float(np.mean(np.abs(applied_move))),
            "p95_abs_move": float(np.quantile(np.abs(applied_move), 0.95)),
            "max_abs_move": float(np.max(np.abs(applied_move))),
        }
    )

score_report = pd.DataFrame(score_rows).sort_values("gain", ascending=False)
fold_report = pd.DataFrame(fold_rows)
score_report["survival_gate"] = (
    (score_report["gain"] >= 0.15)
    & (score_report["positive_folds"] == N_FOLDS)
    & (score_report["min_fold_gain"] > 0.0)
    & (score_report["top40_gain"] >= 0.30)
)

score_report.to_csv(OUT_DIR / "downstream_survival_scores.csv", index=False)
fold_report.to_csv(OUT_DIR / "downstream_survival_folds.csv", index=False)

post_stage = score_report[score_report["base_col"].isin(["strong_bias", "cp12"])]
strict_pass = bool(len(post_stage) and post_stage["survival_gate"].any())
verdict = {
    "pool": str(POOL_PATH),
    "fold_source": fold_source,
    "eligible_wells": int(pool.loc[eligible, "well"].nunique()),
    "eligible_rows": int(eligible.sum()),
    "frozen_move": {
        "weight": WEIGHT,
        "raw_cap": RAW_MOVE_CAP,
        "applied_cap": WEIGHT * RAW_MOVE_CAP,
    },
    "passing_post_stages": post_stage.loc[post_stage["survival_gate"], "variant"].tolist(),
    "strict_pass": strict_pass,
    "next": (
        "LB confirmation permitting, preserve the move through the best passing downstream stage"
        if strict_pass
        else "Neighbor signal is absorbed downstream; standalone only"
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, ensure_ascii=True)

print(
    {
        "experiment": "neighbor_fullcurve_downstream_survival_v0",
        "pool": str(POOL_PATH),
        "fold_source": fold_source,
    }
)
print("\nSCORES")
display(score_report)
print("\nFOLDS")
display(fold_report)
print("\nVERDICT", verdict)
print("saved:", OUT_DIR)
