"""Disjoint confirmation for the frozen stable GR-calibration PF rule.

Run after the original 6.991 notebook cells that define ``run_particle_filter``.
The well sample used by the quick experiment is excluded completely.
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


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
QUICK_ROWS = OOF_DIR / "gr_guarded_calibration_pf_rows.pkl"

# Frozen before looking at the disjoint confirmation result.
N_WELLS = 150
N_SEEDS = 16
N_PARTICLES = 200
N_JOBS = 4
PREFIX_EVERY = 3
SHRINK = 0.5
ROW_MOVE_CAP = 2.0
RANDOM_STATE = 20260711

BASE_CACHE = OOF_DIR / f"pf_seed_cache_s{N_SEEDS}_p{N_PARTICLES}"
CAL_CACHE = OOF_DIR / f"gr_stable2_confirm_cache_s{N_SEEDS}_p{N_PARTICLES}"
BASE_CACHE.mkdir(parents=True, exist_ok=True)
CAL_CACHE.mkdir(parents=True, exist_ok=True)

if "run_particle_filter" not in globals():
    raise RuntimeError(
        "run_particle_filter is missing. Run the original 6.991 function-definition cells first."
    )
if not QUICK_ROWS.exists():
    raise RuntimeError(f"Missing quick-result file: {QUICK_ROWS}")

print(
    {
        "stage": "disjoint_confirm",
        "n_wells": N_WELLS,
        "n_seeds": N_SEEDS,
        "n_particles": N_PARTICLES,
        "shrink": SHRINK,
        "row_move_cap": ROW_MOVE_CAP,
    }
)


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


def robust_affine(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    keep = np.ones(len(x), dtype=bool)
    slope, intercept = 1.0, 0.0
    for _ in range(6):
        if keep.sum() < 20:
            break
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        center = np.median(residual[keep])
        scale = 1.4826 * np.median(np.abs(residual[keep] - center)) + 1e-6
        new_keep = np.abs(residual - center) <= 3.0 * scale
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    slope = float(np.clip(slope, 0.25, 4.0))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def heldout_gain(x, y, train_end, valid_start, valid_end):
    n = len(x)
    train_end = int(train_end * n)
    valid_start = int(valid_start * n)
    valid_end = int(valid_end * n)
    if train_end < 50 or valid_end - valid_start < 30:
        return np.nan
    slope, intercept = robust_affine(x[:train_end], y[:train_end])
    base_score = rmse(y[valid_start:valid_end], x[valid_start:valid_end])
    mapped_score = rmse(
        y[valid_start:valid_end],
        slope * x[valid_start:valid_end] + intercept,
    )
    return base_score - mapped_score


def selector_pf_params(hw):
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    code = int(n_eval > 4840.0) + 2 * int(
        np.searchsorted((136.73000000000016, 185.5133333333342), z_span, side="right")
    )
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
    return scale, hold


def likelihood_weights(log_lik, scale):
    log_lik = np.asarray(log_lik, dtype=np.float64)
    z = (log_lik - np.max(log_lik)) / float(scale)
    weights = np.exp(np.clip(z, -700.0, 0.0))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def run_pf_ensemble(hw, tw, eval_pos, scale, cache_path):
    if cache_path.exists():
        cached = np.load(cache_path)
        cached_pos = cached["eval_pos"].astype(int)
        if not np.array_equal(cached_pos, eval_pos):
            raise RuntimeError(f"cache position mismatch: {cache_path}")
        pred_arr = cached["pred_arr"].astype(np.float64)
        log_lik = cached["log_lik"].astype(np.float64)
    else:
        seed_predictions = []
        log_lik = []
        for seed in range(N_SEEDS):
            full_pred, ll = run_particle_filter(
                hw, tw, n_particles=N_PARTICLES, seed=seed
            )
            seed_predictions.append(np.asarray(full_pred, dtype=float)[eval_pos])
            log_lik.append(float(ll))
        pred_arr = np.stack(seed_predictions)
        log_lik = np.asarray(log_lik, dtype=np.float64)
        np.savez_compressed(
            cache_path,
            pred_arr=pred_arr.astype(np.float32),
            log_lik=log_lik,
            eval_pos=eval_pos.astype(np.int32),
        )
    weights = likelihood_weights(log_lik, scale)
    return np.sum(weights[:, None] * pred_arr, axis=0)


# Build a disjoint, target-free stratified sample.
used_wells = set(pd.read_pickle(QUICK_ROWS)["well"].astype(str).unique())
meta_records = []
for path in tqdm(
    sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv")),
    desc="scan confirm metadata",
):
    wid = path.name.replace("__horizontal_well.csv", "")
    h = pd.read_csv(path, usecols=["TVT_input", "TVT"])
    n_eval = int((h["TVT_input"].isna() & h["TVT"].notna()).sum())
    if n_eval >= 30 and wid not in used_wells:
        meta_records.append({"well": wid, "n_eval": n_eval})

meta = pd.DataFrame(meta_records).sort_values("well").reset_index(drop=True)
meta["stratum"] = pd.qcut(
    meta["n_eval"].rank(method="first"), q=10, labels=False
).astype(int)
splitter = StratifiedShuffleSplit(
    n_splits=1, train_size=N_WELLS, random_state=RANDOM_STATE
)
sample_idx, _ = next(splitter.split(meta, meta["stratum"]))
sample = meta.iloc[sample_idx].copy().reset_index(drop=True)
sample["fold"] = -1
fold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (_, valid_idx) in enumerate(
    fold_splitter.split(sample, sample["stratum"]), 1
):
    sample.loc[valid_idx, "fold"] = fold

assert not (set(sample["well"]) & used_wells)
print("quick wells excluded:", len(used_wells))
print("confirm sample:", sample.shape)
display(sample.groupby(["fold", "stratum"]).size().unstack(fill_value=0))


def process_well(wid, fold):
    started = time.time()
    try:
        hw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__typewell.csv"
        ).sort_values("TVT")

        eval_pos_all = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
        truth_all = hw["TVT"].to_numpy(dtype=float)[eval_pos_all]
        truth_ok = np.isfinite(truth_all)
        eval_pos = eval_pos_all[truth_ok]
        truth = truth_all[truth_ok]
        if len(eval_pos) < 30:
            raise RuntimeError("too few scored eval rows")

        known = hw[hw["TVT_input"].notna() & hw["GR"].notna()].iloc[::PREFIX_EVERY]
        if len(known) < 100:
            raise RuntimeError("too few calibration prefix rows")
        tw_tvt = tw["TVT"].to_numpy(dtype=float)
        tw_gr = tw["GR"].interpolate(limit_direction="both").to_numpy(dtype=float)
        hz = known["GR"].to_numpy(dtype=float)
        target = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)

        prefix_gain = heldout_gain(hz, target, 0.70, 0.70, 1.00)
        window_gains = np.asarray(
            [
                heldout_gain(hz, target, 0.40, 0.40, 0.60),
                heldout_gain(hz, target, 0.60, 0.60, 0.80),
                heldout_gain(hz, target, 0.80, 0.80, 1.00),
            ],
            dtype=float,
        )
        finite_window_gains = window_gains[np.isfinite(window_gains)]
        window_median = (
            float(np.median(finite_window_gains))
            if len(finite_window_gains) else np.nan
        )
        positive_windows = int(np.sum(finite_window_gains > 0))
        selected = bool(
            prefix_gain >= 0.5
            and positive_windows >= 2
            and window_median >= 0.5
        )

        scale, hold = selector_pf_params(hw)
        last_tvt = float(hw.loc[hw["TVT_input"].notna(), "TVT_input"].iloc[-1])
        base_pred = run_pf_ensemble(
            hw,
            tw,
            eval_pos,
            scale,
            BASE_CACHE / f"{wid}.npz",
        )
        base_pred = (1.0 - hold) * base_pred + hold * last_tvt

        if selected:
            slope, intercept = robust_affine(hz, target)
            calibrated = hw.copy()
            original_gr = calibrated["GR"].to_numpy(dtype=float)
            mapped_gr = slope * original_gr + intercept
            calibrated["GR"] = (
                (1.0 - SHRINK) * original_gr + SHRINK * mapped_gr
            )
            cal_pred = run_pf_ensemble(
                calibrated,
                tw,
                eval_pos,
                scale,
                CAL_CACHE / f"{wid}_shrink{SHRINK:.2f}.npz",
            )
            cal_pred = (1.0 - hold) * cal_pred + hold * last_tvt
            pred = base_pred + np.clip(
                cal_pred - base_pred, -ROW_MOVE_CAP, ROW_MOVE_CAP
            )
        else:
            slope, intercept = robust_affine(hz, target)
            pred = base_pred.copy()

        rows = pd.DataFrame(
            {
                "well": wid,
                "fold": int(fold),
                "row_idx": eval_pos,
                "tvt_true": truth,
                "base": base_pred,
                "pred": pred,
            }
        )
        diag = {
            "well": wid,
            "fold": int(fold),
            "rows": len(rows),
            "selected": selected,
            "prefix_gain": prefix_gain,
            "window_gain_1": window_gains[0],
            "window_gain_2": window_gains[1],
            "window_gain_3": window_gains[2],
            "window_gain_median": window_median,
            "positive_windows": positive_windows,
            "slope": slope,
            "intercept": intercept,
            "elapsed_sec": time.time() - started,
        }
        return {"ok": True, "rows": rows, "diag": diag}
    except Exception as exc:
        return {"ok": False, "well": wid, "reason": repr(exc)}


tasks = [(row.well, int(row.fold)) for row in sample.itertuples(index=False)]
started = time.time()
with tqdm_joblib(tqdm(total=len(tasks), desc="stable2 disjoint confirm")):
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS, batch_size=1)(
            delayed(process_well)(wid, fold) for wid, fold in tasks
        )

failures = pd.DataFrame([r for r in results if not r["ok"]])
successes = [r for r in results if r["ok"]]
if len(successes) < 0.9 * len(tasks):
    display(failures.head(30))
    raise RuntimeError(f"Too many failures: {len(failures)} / {len(tasks)}")

rows = pd.concat([r["rows"] for r in successes], ignore_index=True)
diag = pd.DataFrame([r["diag"] for r in successes])
print("confirm rows/wells:", rows.shape, rows["well"].nunique())
print("elapsed_sec:", time.time() - started)

fold_records = []
for fold, g in rows.groupby("fold", sort=True):
    base_score = rmse(g["tvt_true"], g["base"])
    score = rmse(g["tvt_true"], g["pred"])
    fold_records.append(
        {
            "fold": int(fold),
            "rows": len(g),
            "base_rmse": base_score,
            "rmse": score,
            "gain": base_score - score,
            "selected_wells": int(diag.loc[diag["fold"] == fold, "selected"].sum()),
        }
    )
fold_df = pd.DataFrame(fold_records)

well_sse = (
    rows.assign(sq=(rows["tvt_true"] - rows["base"]) ** 2)
    .groupby("well", as_index=False)["sq"]
    .sum()
)
top_wells = set(well_sse.nlargest(min(40, len(well_sse)), "sq")["well"])
top_rows = rows[rows["well"].isin(top_wells)]
selected_wells = set(diag.loc[diag["selected"], "well"])
selected_rows = rows[rows["well"].isin(selected_wells)]

base_score = rmse(rows["tvt_true"], rows["base"])
score = rmse(rows["tvt_true"], rows["pred"])
top_base = rmse(top_rows["tvt_true"], top_rows["base"])
top_score = rmse(top_rows["tvt_true"], top_rows["pred"])
selected_gain = (
    rmse(selected_rows["tvt_true"], selected_rows["base"])
    - rmse(selected_rows["tvt_true"], selected_rows["pred"])
    if len(selected_rows) else 0.0
)
move = rows["pred"].to_numpy() - rows["base"].to_numpy()

summary = pd.DataFrame(
    [
        {
            "candidate": "stable2_cap2_frozen",
            "base_rmse": base_score,
            "rmse": score,
            "gain": base_score - score,
            "mean_fold_gain": float(fold_df["gain"].mean()),
            "min_fold_gain": float(fold_df["gain"].min()),
            "positive_folds": int((fold_df["gain"] > 0).sum()),
            "top40_base_rmse": top_base,
            "top40_rmse": top_score,
            "top40_gain": top_base - top_score,
            "selected_wells": len(selected_wells),
            "selected_rows": len(selected_rows),
            "selected_eval_gain": selected_gain,
            "mean_abs_move": float(np.mean(np.abs(move))),
            "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
        }
    ]
)
summary["strict_gate"] = (
    (summary["gain"] >= 0.15)
    & (summary["min_fold_gain"] >= 0.05)
    & (summary["positive_folds"] == 5)
    & (summary["top40_gain"] > 0)
    & (summary["selected_eval_gain"] > 0.10)
)

display(summary)
display(fold_df)
display(
    diag[
        [
            "selected", "prefix_gain", "window_gain_median",
            "positive_windows", "slope", "intercept", "elapsed_sec",
        ]
    ].describe(include="all")
)

prefix = OOF_DIR / "gr_calibration_stable2_disjoint_confirm"
rows.to_pickle(str(prefix) + "_rows.pkl")
diag.to_csv(str(prefix) + "_bywell.csv", index=False)
fold_df.to_csv(str(prefix) + "_folds.csv", index=False)
summary.to_csv(str(prefix) + "_summary.csv", index=False)
failures.to_csv(str(prefix) + "_failures.csv", index=False)

print("STRICT GATE PASS:", bool(summary.loc[0, "strict_gate"]))
print("saved prefix:", prefix)
