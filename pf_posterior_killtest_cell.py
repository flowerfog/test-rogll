"""Notebook cell: honest PF seed-posterior aggregation kill-test.

Run the original 6.991 notebook through the cell that defines
``run_particle_filter`` first, then execute this file as a notebook cell.
"""

from contextlib import contextmanager
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")

# quick: kill only; confirm: rerun a promising signal at the declared budget.
MODE = "quick"  # "quick" or "confirm"
CONFIGS = {
    "quick": {"n_wells": 100, "n_seeds": 16, "n_particles": 200},
    "confirm": {"n_wells": 150, "n_seeds": 32, "n_particles": 300},
}
N_JOBS = 4
TOP_K = 16
RANDOM_STATE = 42

if "run_particle_filter" not in globals():
    raise RuntimeError(
        "Run the original notebook function-definition cells first; "
        "run_particle_filter is missing."
    )
if MODE not in CONFIGS:
    raise ValueError(f"Unknown MODE={MODE!r}")
if not (DATA_ROOT / "train").exists():
    raise RuntimeError(f"Missing train directory: {DATA_ROOT / 'train'}")

cfg = CONFIGS[MODE]
N_WELLS = cfg["n_wells"]
N_SEEDS = cfg["n_seeds"]
N_PARTICLES = cfg["n_particles"]
OOF_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OOF_DIR / f"pf_seed_cache_s{N_SEEDS}_p{N_PARTICLES}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(
    {
        "mode": MODE,
        "n_wells": N_WELLS,
        "n_seeds": N_SEEDS,
        "n_particles": N_PARTICLES,
        "n_jobs": N_JOBS,
        "cache": str(CACHE_DIR),
    }
)


# -----------------------------------------------------------------------------
# Progress support for joblib
# -----------------------------------------------------------------------------
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


def selector_pf_params(hw):
    """Return the PF scale and hold weight used by the existing selector."""
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0

    n_bin = int(n_eval > 4840.0)
    z_bin = int(np.searchsorted((136.73000000000016, 185.5133333333342), z_span, side="right"))
    code = n_bin + 2 * z_bin
    variants = {
        0: "pf_scale_5_hold_0.2",
        1: "pf_scale_3_hold_0.15",
        2: "pf_scale_12_beam_0.2_hold_0.15",
        3: "pf_scale_5_hold_0.15",
        4: "pf_scale_5_beam_0.05_hold_0.05",
        5: "pf_scale_12_beam_0.2_hold_0.05",
    }
    name = variants.get(code, "pf_scale_8_hold_0.2")
    parts = name.split("_")
    scale = float(parts[2])
    hold = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    return name, scale, hold


def likelihood_weights(log_lik, scale):
    z = (np.asarray(log_lik, dtype=np.float64) - np.max(log_lik)) / float(scale)
    w = np.exp(np.clip(z, -700.0, 0.0))
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(len(w), 1.0 / len(w), dtype=np.float64)
    return w / total


def weighted_median_rows(pred_arr, weights):
    order = np.argsort(pred_arr, axis=0)
    sorted_pred = np.take_along_axis(pred_arr, order, axis=0)
    weight_matrix = np.broadcast_to(weights[:, None], pred_arr.shape)
    sorted_weight = np.take_along_axis(weight_matrix, order, axis=0)
    cumulative = np.cumsum(sorted_weight, axis=0)
    median_pos = np.argmax(cumulative >= 0.5, axis=0)
    cols = np.arange(pred_arr.shape[1])
    return sorted_pred[median_pos, cols]


def process_well(wid, fold):
    started = time.time()
    try:
        hw_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
        tw_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path)

        all_eval_pos = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
        truth_all = hw["TVT"].to_numpy(dtype=float)[all_eval_pos]
        truth_ok = np.isfinite(truth_all)
        eval_pos = all_eval_pos[truth_ok]
        truth = truth_all[truth_ok]
        if len(eval_pos) < 30:
            raise RuntimeError(f"only {len(eval_pos)} scored eval rows")

        cache_path = CACHE_DIR / f"{wid}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            pred_arr = cached["pred_arr"].astype(np.float64)
            log_lik = cached["log_lik"].astype(np.float64)
            cached_pos = cached["eval_pos"].astype(int)
            if not np.array_equal(cached_pos, eval_pos):
                raise RuntimeError("cache eval positions do not match source file")
        else:
            seed_predictions = []
            log_lik = []
            for seed in range(N_SEEDS):
                full_pred, ll = run_particle_filter(
                    hw, tw, n_particles=N_PARTICLES, seed=seed
                )
                seed_predictions.append(np.asarray(full_pred, dtype=float)[eval_pos])
                log_lik.append(float(ll))
            pred_arr = np.stack(seed_predictions, axis=0)
            log_lik = np.asarray(log_lik, dtype=np.float64)
            np.savez_compressed(
                cache_path,
                pred_arr=pred_arr.astype(np.float32),
                log_lik=log_lik,
                eval_pos=eval_pos.astype(np.int32),
            )

        variant, scale, hold = selector_pf_params(hw)
        weights = likelihood_weights(log_lik, scale)
        weighted_mean = np.sum(weights[:, None] * pred_arr, axis=0)

        map_idx = int(np.argmax(log_lik))
        map_seed = pred_arr[map_idx]

        top_idx = np.argsort(log_lik)[-min(TOP_K, len(log_lik)) :]
        top_weights = likelihood_weights(log_lik[top_idx], scale)
        top16_mean = np.sum(top_weights[:, None] * pred_arr[top_idx], axis=0)

        # This is the exact weighted medoid under row-mean squared path distance.
        medoid_idx = int(np.argmin(np.mean((pred_arr - weighted_mean[None, :]) ** 2, axis=1)))
        weighted_medoid = pred_arr[medoid_idx]
        weighted_median = weighted_median_rows(pred_arr, weights)

        known = hw.loc[hw["TVT_input"].notna(), "TVT_input"]
        if len(known) == 0:
            raise RuntimeError("no known TVT_input anchor")
        last_tvt = float(known.iloc[-1])

        def apply_hold(values):
            return (1.0 - hold) * np.asarray(values) + hold * last_tvt

        row_df = pd.DataFrame(
            {
                "well": wid,
                "fold": int(fold),
                "row_idx": eval_pos,
                "tvt_true": truth,
                "weighted_mean": apply_hold(weighted_mean),
                "map_seed": apply_hold(map_seed),
                "top16_mean": apply_hold(top16_mean),
                "weighted_medoid": apply_hold(weighted_medoid),
                "weighted_median": apply_hold(weighted_median),
            }
        )
        diag = {
            "well": wid,
            "fold": int(fold),
            "rows": len(row_df),
            "selector_variant": variant,
            "lik_scale": scale,
            "hold_weight": hold,
            "effective_seeds": float(1.0 / np.sum(weights**2)),
            "top_seed_weight": float(np.max(weights)),
            "map_seed_idx": map_idx,
            "medoid_seed_idx": medoid_idx,
            "elapsed_sec": time.time() - started,
        }
        return {"ok": True, "rows": row_df, "diag": diag}
    except Exception as exc:
        return {"ok": False, "well": wid, "reason": repr(exc)}


# -----------------------------------------------------------------------------
# Build a target-free stratified well sample and fixed folds
# -----------------------------------------------------------------------------
meta_rows = []
hw_files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
for path in tqdm(hw_files, desc="scan well metadata"):
    wid = path.name.replace("__horizontal_well.csv", "")
    h = pd.read_csv(path, usecols=["TVT_input", "TVT", "Z"])
    mask = h["TVT_input"].isna() & h["TVT"].notna()
    if mask.sum() >= 30:
        meta_rows.append({"well": wid, "n_eval": int(mask.sum())})

meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
meta["stratum"] = pd.qcut(
    meta["n_eval"].rank(method="first"), q=min(10, len(meta)), labels=False
).astype(int)

if N_WELLS < len(meta):
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=N_WELLS, random_state=RANDOM_STATE
    )
    sample_idx, _ = next(splitter.split(meta, meta["stratum"]))
    sample = meta.iloc[sample_idx].copy().reset_index(drop=True)
else:
    sample = meta.copy()

sample["fold"] = -1
fold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (_, valid_idx) in enumerate(fold_splitter.split(sample, sample["stratum"]), 1):
    sample.loc[valid_idx, "fold"] = fold

print("sample:", sample.shape, "all wells:", len(meta))
print(sample.groupby(["fold", "stratum"]).size().unstack(fill_value=0))


# -----------------------------------------------------------------------------
# Run wells in parallel. Seed paths are cached, so interrupted runs can resume.
# -----------------------------------------------------------------------------
tasks = [(row.well, int(row.fold)) for row in sample.itertuples(index=False)]
started = time.time()
with tqdm_joblib(tqdm(total=len(tasks), desc="PF posterior wells")):
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS, batch_size=1)(
            delayed(process_well)(wid, fold) for wid, fold in tasks
        )

failures = pd.DataFrame([r for r in results if not r["ok"]])
successes = [r for r in results if r["ok"]]
if len(successes) < 0.9 * len(tasks):
    print(failures.head(30))
    raise RuntimeError(f"Too many failed wells: {len(failures)} / {len(tasks)}")

rows = pd.concat([r["rows"] for r in successes], ignore_index=True)
well_diag = pd.DataFrame([r["diag"] for r in successes])
print("rows:", rows.shape, "wells:", rows["well"].nunique())
print("elapsed_sec:", time.time() - started)
if len(failures):
    print("failures:", len(failures))
    print(failures[["well", "reason"]].head(20))


# -----------------------------------------------------------------------------
# Honest pooled-row, fold, and high-SSE-well evaluation
# -----------------------------------------------------------------------------
base_col = "weighted_mean"
pred_cols = [
    "weighted_mean",
    "map_seed",
    "top16_mean",
    "weighted_medoid",
    "weighted_median",
]
base_rmse = rmse(rows["tvt_true"], rows[base_col])

fold_records = []
for candidate in pred_cols:
    for fold, g in rows.groupby("fold", sort=True):
        score = rmse(g["tvt_true"], g[candidate])
        ref_score = rmse(g["tvt_true"], g[base_col])
        fold_records.append(
            {
                "candidate": candidate,
                "fold": int(fold),
                "base_rmse": ref_score,
                "rmse": score,
                "gain": ref_score - score,
                "rows": len(g),
            }
        )
fold_df = pd.DataFrame(fold_records)

well_records = []
for wid, g in rows.groupby("well", sort=False):
    record = {
        "well": wid,
        "fold": int(g["fold"].iloc[0]),
        "rows": len(g),
        "base_sse": float(np.sum((g["tvt_true"] - g[base_col]) ** 2)),
    }
    for candidate in pred_cols:
        record[f"rmse_{candidate}"] = rmse(g["tvt_true"], g[candidate])
    record["best_candidate"] = min(
        pred_cols, key=lambda c: record[f"rmse_{c}"]
    )
    well_records.append(record)
well_df = pd.DataFrame(well_records)

top_n = min(40, len(well_df))
top_wells = set(well_df.nlargest(top_n, "base_sse")["well"])
top_rows = rows[rows["well"].isin(top_wells)]
top_base_rmse = rmse(top_rows["tvt_true"], top_rows[base_col])

summary_records = []
for candidate in pred_cols:
    score = rmse(rows["tvt_true"], rows[candidate])
    candidate_folds = fold_df[fold_df["candidate"] == candidate]
    top_score = rmse(top_rows["tvt_true"], top_rows[candidate])
    summary_records.append(
        {
            "candidate": candidate,
            "base_rmse": base_rmse,
            "rmse": score,
            "gain": base_rmse - score,
            "mean_fold_gain": float(candidate_folds["gain"].mean()),
            "min_fold_gain": float(candidate_folds["gain"].min()),
            "positive_folds": int((candidate_folds["gain"] > 0).sum()),
            "top40_base_rmse": top_base_rmse,
            "top40_rmse": top_score,
            "top40_gain": top_base_rmse - top_score,
        }
    )
summary = pd.DataFrame(summary_records).sort_values("rmse").reset_index(drop=True)
summary["strict_gate"] = (
    (summary["gain"] >= 0.15)
    & (summary["min_fold_gain"] >= 0.05)
    & (summary["positive_folds"] == 5)
    & (summary["top40_gain"] > 0)
)

print("\nOVERALL / FOLD / TOP40")
display(summary)
print("\nBEST CANDIDATE BY WELL")
display(well_df["best_candidate"].value_counts().rename_axis("candidate").reset_index(name="wells"))
print("\nLIKELIHOOD CONCENTRATION")
display(well_diag[["effective_seeds", "top_seed_weight", "elapsed_sec"]].describe())

prefix = OOF_DIR / f"pf_posterior_{MODE}_s{N_SEEDS}_p{N_PARTICLES}"
summary.to_csv(str(prefix) + "_summary.csv", index=False)
fold_df.to_csv(str(prefix) + "_folds.csv", index=False)
well_df.to_csv(str(prefix) + "_by_well.csv", index=False)
well_diag.to_csv(str(prefix) + "_diagnostics.csv", index=False)
failures.to_csv(str(prefix) + "_failures.csv", index=False)
rows.to_pickle(str(prefix) + "_rows.pkl")

passing = summary.loc[summary["strict_gate"], "candidate"].tolist()
print("\nSTRICT GATE PASS:", passing)
if MODE == "quick" and passing:
    print("Quick signal only. Set MODE='confirm' and rerun before any full-pipeline work.")
elif not passing:
    print("No qualifying signal. Stop this PF aggregation direction.")
print("saved prefix:", prefix)
