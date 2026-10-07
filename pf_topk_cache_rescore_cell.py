"""Rescore top-2/4/8 PF likelihood subsets from the quick-run cache."""

from pathlib import Path
import numpy as np
import pandas as pd


OOF_DIR = Path("/root/autodl-tmp/oof_exports")
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
CACHE_DIR = OOF_DIR / "pf_seed_cache_s16_p200"
ROWS_PATH = OOF_DIR / "pf_posterior_quick_s16_p200_rows.pkl"
TOP_KS = (2, 4, 8)

required = ["selector_pf_params", "likelihood_weights", "rmse"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        f"Run the original PF posterior quick cell first; missing helpers: {missing}"
    )
if not ROWS_PATH.exists():
    raise RuntimeError(f"Missing quick rows: {ROWS_PATH}")

rows = pd.read_pickle(ROWS_PATH)
extra_parts = []

for n_done, (wid, g) in enumerate(rows.groupby("well", sort=False), 1):
    cache_path = CACHE_DIR / f"{wid}.npz"
    hw_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    cached = np.load(cache_path)
    pred_arr = cached["pred_arr"].astype(np.float64)
    log_lik = cached["log_lik"].astype(np.float64)
    eval_pos = cached["eval_pos"].astype(int)

    hw = pd.read_csv(hw_path, usecols=["TVT_input", "Z"])
    _, scale, hold = selector_pf_params(hw)
    last_tvt = float(hw.loc[hw["TVT_input"].notna(), "TVT_input"].iloc[-1])

    out = pd.DataFrame({"well": wid, "row_idx": eval_pos})
    order = np.argsort(log_lik)
    for k in TOP_KS:
        idx = order[-min(k, len(order)) :]
        weights = likelihood_weights(log_lik[idx], scale)
        pred = np.sum(weights[:, None] * pred_arr[idx], axis=0)
        out[f"top{k}_mean"] = (1.0 - hold) * pred + hold * last_tvt
    extra_parts.append(out)

    if n_done % 20 == 0:
        print(f"rescore {n_done}/{rows['well'].nunique()}", flush=True)

extra = pd.concat(extra_parts, ignore_index=True)
rows = rows.merge(extra, on=["well", "row_idx"], how="left", validate="one_to_one")

base_col = "weighted_mean"
pred_cols = [f"top{k}_mean" for k in TOP_KS]
base_rmse = rmse(rows["tvt_true"], rows[base_col])

well_sse = (
    rows.assign(base_sq=(rows["tvt_true"] - rows[base_col]) ** 2)
    .groupby("well", as_index=False)["base_sq"]
    .sum()
    .sort_values("base_sq", ascending=False)
)
top_wells = set(well_sse.head(min(40, len(well_sse)))["well"])
top_rows = rows[rows["well"].isin(top_wells)]
top_base = rmse(top_rows["tvt_true"], top_rows[base_col])

fold_records = []
summary_records = []
for candidate in pred_cols:
    candidate_folds = []
    for fold, fg in rows.groupby("fold", sort=True):
        ref = rmse(fg["tvt_true"], fg[base_col])
        score = rmse(fg["tvt_true"], fg[candidate])
        gain = ref - score
        candidate_folds.append(gain)
        fold_records.append(
            {
                "candidate": candidate,
                "fold": int(fold),
                "base_rmse": ref,
                "rmse": score,
                "gain": gain,
                "rows": len(fg),
            }
        )

    score = rmse(rows["tvt_true"], rows[candidate])
    top_score = rmse(top_rows["tvt_true"], top_rows[candidate])
    summary_records.append(
        {
            "candidate": candidate,
            "base_rmse": base_rmse,
            "rmse": score,
            "gain": base_rmse - score,
            "mean_fold_gain": float(np.mean(candidate_folds)),
            "min_fold_gain": float(np.min(candidate_folds)),
            "positive_folds": int(np.sum(np.asarray(candidate_folds) > 0)),
            "top40_base_rmse": top_base,
            "top40_rmse": top_score,
            "top40_gain": top_base - top_score,
        }
    )

summary = pd.DataFrame(summary_records).sort_values("rmse").reset_index(drop=True)
summary["strict_gate"] = (
    (summary["gain"] >= 0.15)
    & (summary["min_fold_gain"] >= 0.05)
    & (summary["positive_folds"] == 5)
    & (summary["top40_gain"] > 0)
)

display(summary)
display(pd.DataFrame(fold_records))
summary.to_csv(OOF_DIR / "pf_posterior_quick_topk_rescore_summary.csv", index=False)
pd.DataFrame(fold_records).to_csv(
    OOF_DIR / "pf_posterior_quick_topk_rescore_folds.csv", index=False
)
print("STRICT GATE PASS:", summary.loc[summary["strict_gate"], "candidate"].tolist())
