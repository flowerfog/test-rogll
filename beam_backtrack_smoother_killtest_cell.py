"""Compare the current online beam filter with true backtracked beam smoothing."""

from contextlib import contextmanager
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
MODE = "quick"  # quick=150 wells; confirm=all 773 wells
N_JOBS = 4
RANDOM_STATE = 42

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),  (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),   (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),  (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),   (10, 50.0, 400.0, 0),
]

if MODE not in {"quick", "confirm"}:
    raise ValueError(MODE)
if not (DATA_ROOT / "train").exists():
    raise RuntimeError(f"Missing train data: {DATA_ROOT / 'train'}")
OOF_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OOF_DIR / "beam_backtrack_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def tqdm_joblib(progress):
    old_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_callback):
        def __call__(self, *args, **kwargs):
            progress.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        progress.close()


def rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def beam_filter_and_smoother(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r):
    """Return current online-best path and final-best backtracked path."""
    n = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    moves = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    move_cost = mc * np.array([2.0, 1.0, 0.0, 1.0, 2.0])

    prev_states = np.array([si], dtype=np.int64)
    prev_costs = np.array([0.0], dtype=np.float64)
    state_history = []
    parent_history = []
    online = np.empty(n, dtype=np.float64)

    for step in range(n):
        bn = len(prev_states)
        next_state = prev_states[:, None] + moves[None, :]
        clipped = np.clip(next_state, 0, nt - 1)
        valid = (next_state >= 0) & (next_state < nt)

        emission = (sgr[step] - tw_gr[clipped]) ** 2 / es
        total = prev_costs[:, None] + emission + move_cost[None, :]
        total = np.where(valid, total, np.inf)

        state_flat = next_state.ravel()
        cost_flat = total.ravel()
        parent_flat = np.repeat(np.arange(bn, dtype=np.int64), len(moves))
        valid_flat = valid.ravel()
        state_flat = state_flat[valid_flat]
        cost_flat = cost_flat[valid_flat]
        parent_flat = parent_flat[valid_flat]

        # Keep the cheapest parent for each unique next state.
        order = np.argsort(cost_flat)
        state_sorted = state_flat[order]
        cost_sorted = cost_flat[order]
        parent_sorted = parent_flat[order]
        _, first = np.unique(state_sorted, return_index=True)
        unique_state = state_sorted[first]
        unique_cost = cost_sorted[first]
        unique_parent = parent_sorted[first]

        kept = min(bs, len(unique_state))
        top = np.argpartition(unique_cost, kept - 1)[:kept]
        top = top[np.argsort(unique_cost[top])]
        current_states = unique_state[top].astype(np.int64)
        current_costs = unique_cost[top].astype(np.float64)
        current_parents = unique_parent[top].astype(np.int64)

        state_history.append(current_states)
        parent_history.append(current_parents)
        online[step] = tw_tvt[current_states[0]]
        prev_states = current_states
        prev_costs = current_costs

    # Slot zero is the globally cheapest final beam state.
    slot = 0
    smooth_index = np.empty(n, dtype=np.int64)
    for step in range(n - 1, -1, -1):
        smooth_index[step] = state_history[step][slot]
        if step > 0:
            slot = int(parent_history[step][slot])
    smooth = tw_tvt[smooth_index]
    return online, smooth


def process_well(wid, fold):
    started = time.time()
    try:
        cache_path = CACHE_DIR / f"{wid}.npz"
        if cache_path.exists():
            c = np.load(cache_path)
            return {
                "ok": True,
                "well": wid,
                "fold": fold,
                "row_idx": c["row_idx"].astype(np.int32),
                "truth": c["truth"].astype(np.float64),
                "filter_mean": c["filter_mean"].astype(np.float64),
                "smooth_mean": c["smooth_mean"].astype(np.float64),
                "smooth_median": c["smooth_median"].astype(np.float64),
                "elapsed_sec": 0.0,
            }

        hw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__typewell.csv")
        eval_mask = hw["TVT_input"].isna().to_numpy()
        eval_pos_all = np.flatnonzero(eval_mask)
        truth_all = hw["TVT"].to_numpy(dtype=float)[eval_pos_all]
        truth_ok = np.isfinite(truth_all)
        eval_pos = eval_pos_all[truth_ok]
        truth = truth_all[truth_ok]
        if len(eval_pos) < 30:
            raise RuntimeError("too few scored eval rows")

        known = hw[hw["TVT_input"].notna()]
        if len(known) == 0:
            raise RuntimeError("missing TVT_input anchor")
        last_tvt = float(known.iloc[-1]["TVT_input"])

        tw_sorted = tw.sort_values("TVT")
        tw_tvt = tw_sorted["TVT"].to_numpy(dtype=float)
        tw_gr = (
            tw_sorted["GR"]
            .fillna(tw_sorted["GR"].mean())
            .to_numpy(dtype=float)
        )
        gr_all = (
            hw["GR"]
            .interpolate(limit_direction="both")
            .fillna(np.nanmean(tw_gr))
            .to_numpy(dtype=float)
        )
        hgr = gr_all[eval_pos_all]

        filter_paths = []
        smooth_paths = []
        for beam_config in BEAM_CONFIGS:
            filtered, smoothed = beam_filter_and_smoother(
                hgr, tw_tvt, tw_gr, last_tvt, *beam_config
            )
            filter_paths.append(filtered[truth_ok])
            smooth_paths.append(smoothed[truth_ok])

        filter_paths = np.stack(filter_paths)
        smooth_paths = np.stack(smooth_paths)
        filter_mean = np.mean(filter_paths, axis=0)
        smooth_mean = np.mean(smooth_paths, axis=0)
        smooth_median = np.median(smooth_paths, axis=0)

        np.savez_compressed(
            cache_path,
            row_idx=eval_pos.astype(np.int32),
            truth=truth.astype(np.float32),
            filter_mean=filter_mean.astype(np.float32),
            smooth_mean=smooth_mean.astype(np.float32),
            smooth_median=smooth_median.astype(np.float32),
        )
        return {
            "ok": True,
            "well": wid,
            "fold": fold,
            "row_idx": eval_pos,
            "truth": truth,
            "filter_mean": filter_mean,
            "smooth_mean": smooth_mean,
            "smooth_median": smooth_median,
            "elapsed_sec": time.time() - started,
        }
    except Exception as exc:
        return {"ok": False, "well": wid, "reason": repr(exc)}


# Target-free sample and fixed folds.
meta_rows = []
files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
for path in tqdm(files, desc="scan metadata"):
    wid = path.name.replace("__horizontal_well.csv", "")
    h = pd.read_csv(path, usecols=["TVT", "TVT_input"])
    n_eval = int((h["TVT_input"].isna() & h["TVT"].notna()).sum())
    if n_eval >= 30:
        meta_rows.append({"well": wid, "n_eval": n_eval})
meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
meta["stratum"] = pd.qcut(
    meta["n_eval"].rank(method="first"), q=10, labels=False
).astype(int)

if MODE == "quick":
    split = StratifiedShuffleSplit(
        n_splits=1, train_size=150, random_state=RANDOM_STATE
    )
    sample_idx, _ = next(split.split(meta, meta["stratum"]))
    sample = meta.iloc[sample_idx].copy().reset_index(drop=True)
else:
    sample = meta.copy()

sample["fold"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (_, valid) in enumerate(skf.split(sample, sample["stratum"]), 1):
    sample.loc[valid, "fold"] = fold
print("sample:", sample.shape, "mode:", MODE)

tasks = [(r.well, int(r.fold)) for r in sample.itertuples(index=False)]
with tqdm_joblib(tqdm(total=len(tasks), desc="beam smoother wells")):
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS, batch_size=1)(
            delayed(process_well)(wid, fold) for wid, fold in tasks
        )

failures = pd.DataFrame([r for r in results if not r["ok"]])
successes = [r for r in results if r["ok"]]
if len(successes) < 0.9 * len(tasks):
    display(failures.head(30))
    raise RuntimeError(f"Too many failures: {len(failures)} / {len(tasks)}")

parts = []
diag = []
for r in successes:
    parts.append(
        pd.DataFrame(
            {
                "well": r["well"],
                "fold": r["fold"],
                "row_idx": r["row_idx"],
                "tvt_true": r["truth"],
                "filter_mean": r["filter_mean"],
                "smooth_mean": r["smooth_mean"],
                "smooth_median": r["smooth_median"],
            }
        )
    )
    diag.append({"well": r["well"], "fold": r["fold"], "elapsed_sec": r["elapsed_sec"]})
rows = pd.concat(parts, ignore_index=True)
diag = pd.DataFrame(diag)

base_col = "filter_mean"
candidate_cols = ["filter_mean", "smooth_mean", "smooth_median"]
base_score = rmse(rows["tvt_true"], rows[base_col])

well_sse = (
    rows.assign(sq=(rows["tvt_true"] - rows[base_col]) ** 2)
    .groupby("well", as_index=False)["sq"]
    .sum()
)
top_wells = set(well_sse.nlargest(min(40, len(well_sse)), "sq")["well"])
top_rows = rows[rows["well"].isin(top_wells)]
top_base = rmse(top_rows["tvt_true"], top_rows[base_col])

fold_records = []
summary_records = []
for candidate in candidate_cols:
    gains = []
    for fold, g in rows.groupby("fold", sort=True):
        ref = rmse(g["tvt_true"], g[base_col])
        score = rmse(g["tvt_true"], g[candidate])
        gain = ref - score
        gains.append(gain)
        fold_records.append(
            {"candidate": candidate, "fold": fold, "base_rmse": ref,
             "rmse": score, "gain": gain, "rows": len(g)}
        )
    score = rmse(rows["tvt_true"], rows[candidate])
    top_score = rmse(top_rows["tvt_true"], top_rows[candidate])
    summary_records.append(
        {
            "candidate": candidate,
            "base_rmse": base_score,
            "rmse": score,
            "gain": base_score - score,
            "mean_fold_gain": float(np.mean(gains)),
            "min_fold_gain": float(np.min(gains)),
            "positive_folds": int(np.sum(np.asarray(gains) > 0)),
            "top40_base_rmse": top_base,
            "top40_rmse": top_score,
            "top40_gain": top_base - top_score,
            "mean_abs_change": float(np.mean(np.abs(rows[candidate] - rows[base_col]))),
        }
    )

summary = pd.DataFrame(summary_records).sort_values("rmse").reset_index(drop=True)
summary["strict_gate"] = (
    (summary["gain"] >= 0.15)
    & (summary["min_fold_gain"] >= 0.05)
    & (summary["positive_folds"] == 5)
    & (summary["top40_gain"] > 0)
)
fold_df = pd.DataFrame(fold_records)

display(summary)
display(fold_df)
display(diag["elapsed_sec"].describe())

prefix = OOF_DIR / f"beam_backtrack_{MODE}"
summary.to_csv(str(prefix) + "_summary.csv", index=False)
fold_df.to_csv(str(prefix) + "_folds.csv", index=False)
rows.to_pickle(str(prefix) + "_rows.pkl")
failures.to_csv(str(prefix) + "_failures.csv", index=False)

passing = summary.loc[summary["strict_gate"], "candidate"].tolist()
print("STRICT GATE PASS:", passing)
if MODE == "quick" and passing:
    print("Set MODE='confirm' and rerun all 773 wells before integration.")
elif not passing:
    print("No qualifying smoother signal. Stop this beam-smoothing direction.")
