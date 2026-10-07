"""Free diagnostic for anchor-respecting TVT gain-family corrections."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
POOL_CANDIDATES = [
    OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_DIR / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((p for p in POOL_CANDIDATES if p.exists()), None)
if POOL_PATH is None:
    raise RuntimeError("candidate pool not found")

K_MIN, K_MAX = 0.94, 1.06
WARP_GRID = np.round(np.arange(0.94, 1.0601, 0.005), 6)
DELTA_GRID = np.arange(-24.0, 24.01, 2.0)
PREFIX_EVERY = 5
RANDOM_STATE = 42


def rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return -np.inf
    x = x[ok] - np.mean(x[ok])
    y = y[ok] - np.mean(y[ok])
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 1e-12 else -np.inf


pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "well", "tvt_true", "cp12"],
    dtype={"id": "string", "well": "string"},
)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = (
    pool["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int32)
)
print("POOL:", POOL_PATH)
print("pool:", pool.shape, "wells:", pool["well"].nunique())

y = pool["tvt_true"].to_numpy(dtype=np.float64)
base = pool["cp12"].to_numpy(dtype=np.float64)
k_hat_inverse = np.ones(len(pool), dtype=np.float32)
k_hat_direct = np.ones(len(pool), dtype=np.float32)
oracle_pred = base.copy()
well_records = []
pool_indices = pool.groupby("well", sort=False).indices
t0 = time.time()

for n_done, (wid, indices) in enumerate(
    tqdm(pool_indices.items(), total=len(pool_indices), desc="gain-family wells"), 1
):
    indices = np.asarray(indices, dtype=np.int64)
    hw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__typewell.csv").sort_values("TVT")
    known = hw[hw["TVT_input"].notna() & hw["GR"].notna()].iloc[::PREFIX_EVERY]
    if len(known) < 50:
        continue
    anchor = float(hw.loc[hw["TVT_input"].notna(), "TVT_input"].iloc[-1])

    pred_w = base[indices]
    truth_w = y[indices]
    x = pred_w - anchor
    target = truth_w - anchor
    denom = float(np.dot(x, x))
    oracle_raw = float(np.dot(x, target) / denom) if denom > 1e-12 else 1.0
    oracle_k = float(np.clip(oracle_raw, K_MIN, K_MAX))
    oracle_w = anchor + oracle_k * x
    oracle_pred[indices] = oracle_w

    prefix_tvt = known["TVT_input"].to_numpy(dtype=np.float64)
    prefix_gr = known["GR"].to_numpy(dtype=np.float64)
    tw_tvt = tw["TVT"].to_numpy(dtype=np.float64)
    tw_gr = tw["GR"].interpolate(limit_direction="both").to_numpy(dtype=np.float64)
    center = anchor

    scores = []
    params = []
    for warp in WARP_GRID:
        warped_base = center + warp * (prefix_tvt - center)
        for delta in DELTA_GRID:
            query = warped_base + delta
            expected = np.interp(query, tw_tvt, tw_gr)
            score = safe_corr(prefix_gr, expected)
            scores.append(score)
            params.append((float(warp), float(delta)))

    scores = np.asarray(scores, dtype=np.float64)
    best_pos = int(np.argmax(scores))
    best_warp, best_delta = params[best_pos]
    finite_scores = scores[np.isfinite(scores)]
    second_score = (
        float(np.partition(finite_scores, len(finite_scores) - 2)[len(finite_scores) - 2])
        if len(finite_scores) >= 2
        else np.nan
    )
    inverse_k = float(np.clip(1.0 / best_warp, K_MIN, K_MAX))
    direct_k = float(np.clip(best_warp, K_MIN, K_MAX))
    k_hat_inverse[indices] = inverse_k
    k_hat_direct[indices] = direct_k

    base_rmse = rmse(truth_w, pred_w)
    oracle_rmse = rmse(truth_w, oracle_w)
    well_records.append(
        {
            "well": wid,
            "rows": len(indices),
            "anchor": anchor,
            "base_rmse": base_rmse,
            "base_sse": float(np.sum((truth_w - pred_w) ** 2)),
            "oracle_k_raw": oracle_raw,
            "oracle_k": oracle_k,
            "oracle_rmse": oracle_rmse,
            "oracle_gain": base_rmse - oracle_rmse,
            "best_warp": best_warp,
            "best_delta": best_delta,
            "prefix_score": float(scores[best_pos]),
            "prefix_margin": float(scores[best_pos] - second_score),
            "k_hat_inverse": inverse_k,
            "k_hat_direct": direct_k,
            "prefix_tvt_span": float(np.ptp(prefix_tvt)),
        }
    )

print("elapsed_sec:", time.time() - t0)
well_df = pd.DataFrame(well_records)
if len(well_df) != pool["well"].nunique():
    print("warning: diagnostic wells:", len(well_df))


# Fixed folds for stability.
well_df["stratum"] = pd.qcut(
    well_df["rows"].rank(method="first"), q=10, labels=False
).astype(int)
well_df["fold"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (_, valid) in enumerate(skf.split(well_df, well_df["stratum"]), 1):
    well_df.loc[valid, "fold"] = fold
fold_map = dict(zip(well_df["well"], well_df["fold"]))
fold_values = pool["well"].map(fold_map).to_numpy()

top40_wells = set(well_df.nlargest(40, "base_sse")["well"])
top40_well_mask = well_df["well"].isin(top40_wells)
top40_row_mask = pool["well"].isin(top40_wells).to_numpy()


def corr_report(mask, estimate_col):
    d = well_df.loc[mask, ["oracle_k", estimate_col]].dropna()
    if len(d) < 10:
        return np.nan, np.nan
    return (
        float(pearsonr(d["oracle_k"], d[estimate_col])[0]),
        float(spearmanr(d["oracle_k"], d[estimate_col])[0]),
    )


all_corr = corr_report(np.ones(len(well_df), dtype=bool), "k_hat_inverse")
top_corr = corr_report(top40_well_mask, "k_hat_inverse")
print("\nK DISTRIBUTION")
print(
    well_df[["oracle_k_raw", "oracle_k", "k_hat_inverse", "best_warp", "oracle_gain"]]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .to_string()
)
print("inverse k corr (Pearson, Spearman):", all_corr)
print("top40 inverse k corr:", top_corr)
print("oracle boundary frac:", float(((well_df["oracle_k"] == K_MIN) | (well_df["oracle_k"] == K_MAX)).mean()))


# Direct deployable candidates from the prefix estimate; inverse is primary.
candidate_specs = [
    ("inverse_half", k_hat_inverse, 0.5),
    ("inverse_full", k_hat_inverse, 1.0),
    ("direct_half_sanity", k_hat_direct, 0.5),
]
base_score = rmse(y, base)
top_base = rmse(y[top40_row_mask], base[top40_row_mask])
records = []
fold_records = []

for name, row_k, shrink in candidate_specs:
    effective_k = 1.0 + shrink * (row_k.astype(np.float64) - 1.0)
    # Each row uses its well's true visible anchor, already stored in well_df.
    anchor_map = dict(zip(well_df["well"], well_df["anchor"]))
    anchors = pool["well"].map(anchor_map).to_numpy(dtype=np.float64)
    pred = anchors + effective_k * (base - anchors)
    score = rmse(y, pred)
    gains = []
    for fold in range(1, 6):
        mask = fold_values == fold
        fold_base = rmse(y[mask], base[mask])
        fold_score = rmse(y[mask], pred[mask])
        gain = fold_base - fold_score
        gains.append(gain)
        fold_records.append(
            {"candidate": name, "fold": fold, "base_rmse": fold_base,
             "rmse": fold_score, "gain": gain, "rows": int(mask.sum())}
        )
    top_score = rmse(y[top40_row_mask], pred[top40_row_mask])
    records.append(
        {
            "candidate": name,
            "base_rmse": base_score,
            "rmse": score,
            "gain": base_score - score,
            "mean_fold_gain": float(np.mean(gains)),
            "min_fold_gain": float(np.min(gains)),
            "positive_folds": int(np.sum(np.asarray(gains) > 0)),
            "top40_base_rmse": top_base,
            "top40_rmse": top_score,
            "top40_gain": top_base - top_score,
            "mean_abs_move": float(np.mean(np.abs(pred - base))),
            "p95_abs_move": float(np.quantile(np.abs(pred - base), 0.95)),
        }
    )

summary = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
summary["mechanism_gate"] = (
    (all_corr[0] > 0.30)
    & (summary["gain"] > 0)
    & (summary["min_fold_gain"] > 0)
    & (summary["positive_folds"] == 5)
    & (summary["top40_gain"] > 0)
)
print("\nDEPLOYABLE K-HAT CANDIDATES")
display(summary)
display(pd.DataFrame(fold_records))

well_df.to_csv(OOF_DIR / "gain_family_prefix_scale_bywell.csv", index=False)
summary.to_csv(OOF_DIR / "gain_family_prefix_scale_summary.csv", index=False)
pd.DataFrame(fold_records).to_csv(
    OOF_DIR / "gain_family_prefix_scale_folds.csv", index=False
)
print("MECHANISM GATE PASS:", summary.loc[summary["mechanism_gate"], "candidate"].tolist())
