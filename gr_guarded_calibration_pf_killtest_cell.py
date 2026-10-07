"""Guarded per-well GR tool calibration followed by the unchanged PF tracker."""

from contextlib import contextmanager
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
BASE_ROWS = OOF_DIR / "pf_posterior_quick_s16_p200_rows.pkl"
N_SEEDS = 16
N_PARTICLES = 200
N_JOBS = 4
PREFIX_EVERY = 3
TRAIN_FRAC = 0.70

# Selection is based only on held-out visible-prefix calibration gain.
CONFIGS = [
    {"name": "guard0p5_half", "gain_thr": 0.5, "shrink": 0.5},
    {"name": "guard1p0_half", "gain_thr": 1.0, "shrink": 0.5},
    {"name": "guard2p0_half", "gain_thr": 2.0, "shrink": 0.5},
    {"name": "guard0p5_full", "gain_thr": 0.5, "shrink": 1.0},
]

if "run_particle_filter" not in globals():
    raise RuntimeError(
        "run_particle_filter is missing. Run the original 6.991 function-definition cells first."
    )
if not BASE_ROWS.exists():
    raise RuntimeError(f"Run PF posterior quick first; missing {BASE_ROWS}")

CACHE_DIR = OOF_DIR / "gr_guarded_calibration_pf_cache_s16_p200"
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


def robust_affine(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    keep = np.ones(len(x), dtype=bool)
    slope = 1.0
    intercept = 0.0
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


def selector_pf_params(hw):
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    code = int(n_eval > 4840.0) + 2 * int(
        np.searchsorted((136.73000000000016, 185.5133333333342), z_span, side="right")
    )
    variants = {
        0: "pf_scale_5_hold_0.2", 1: "pf_scale_3_hold_0.15",
        2: "pf_scale_12_beam_0.2_hold_0.15", 3: "pf_scale_5_hold_0.15",
        4: "pf_scale_5_beam_0.05_hold_0.05", 5: "pf_scale_12_beam_0.2_hold_0.05",
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
    return weights / weights.sum()


base_rows = pd.read_pickle(BASE_ROWS)
sample_wells = sorted(base_rows["well"].astype(str).unique())
base_lookup = {
    wid: g.sort_values("row_idx").reset_index(drop=True)
    for wid, g in base_rows.groupby("well", sort=False)
}
print("sample wells:", len(sample_wells), "rows:", len(base_rows))


def process_well(wid):
    started = time.time()
    try:
        base_g = base_lookup[wid]
        row_idx = base_g["row_idx"].to_numpy(dtype=int)
        base_pred = base_g["weighted_mean"].to_numpy(dtype=np.float64)
        hw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / "train" / f"{wid}__typewell.csv").sort_values("TVT")

        known = hw[hw["TVT_input"].notna() & hw["GR"].notna()].iloc[::PREFIX_EVERY]
        if len(known) < 100:
            raise RuntimeError("too few calibration prefix rows")
        tw_tvt = tw["TVT"].to_numpy(dtype=float)
        tw_gr = tw["GR"].interpolate(limit_direction="both").to_numpy(dtype=float)
        hz = known["GR"].to_numpy(dtype=float)
        target = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)

        cut = max(50, min(len(hz) - 30, int(TRAIN_FRAC * len(hz))))
        slope_train, intercept_train = robust_affine(hz[:cut], target[:cut])
        identity_rmse = rmse(target[cut:], hz[cut:])
        mapped_valid = slope_train * hz[cut:] + intercept_train
        mapped_rmse = rmse(target[cut:], mapped_valid)
        prefix_gain = identity_rmse - mapped_rmse

        slope_full, intercept_full = robust_affine(hz, target)
        scale, hold = selector_pf_params(hw)
        last_tvt = float(hw.loc[hw["TVT_input"].notna(), "TVT_input"].iloc[-1])

        shrink_predictions = {}
        if prefix_gain >= min(c["gain_thr"] for c in CONFIGS):
            for shrink in sorted(set(c["shrink"] for c in CONFIGS)):
                cache_path = CACHE_DIR / f"{wid}_shrink{shrink:.2f}.npz"
                if cache_path.exists():
                    shrink_predictions[shrink] = np.load(cache_path)["pred"].astype(np.float64)
                    continue

                calibrated = hw.copy()
                original_gr = calibrated["GR"].to_numpy(dtype=float)
                mapped_gr = slope_full * original_gr + intercept_full
                calibrated["GR"] = (1.0 - shrink) * original_gr + shrink * mapped_gr

                seed_pred = []
                log_lik = []
                for seed in range(N_SEEDS):
                    full_pred, ll = run_particle_filter(
                        calibrated, tw, n_particles=N_PARTICLES, seed=seed
                    )
                    seed_pred.append(np.asarray(full_pred, dtype=float)[row_idx])
                    log_lik.append(float(ll))
                seed_pred = np.stack(seed_pred)
                weights = likelihood_weights(log_lik, scale)
                pred = np.sum(weights[:, None] * seed_pred, axis=0)
                pred = (1.0 - hold) * pred + hold * last_tvt
                shrink_predictions[shrink] = pred
                np.savez_compressed(cache_path, pred=pred.astype(np.float32))

        output = {
            "ok": True,
            "well": wid,
            "row_idx": row_idx,
            "prefix_gain": prefix_gain,
            "identity_prefix_rmse": identity_rmse,
            "mapped_prefix_rmse": mapped_rmse,
            "slope": slope_full,
            "intercept": intercept_full,
            "elapsed_sec": time.time() - started,
        }
        for config in CONFIGS:
            if prefix_gain >= config["gain_thr"]:
                output[config["name"]] = shrink_predictions[config["shrink"]]
            else:
                output[config["name"]] = base_pred
        return output
    except Exception as exc:
        return {"ok": False, "well": wid, "reason": repr(exc)}


with tqdm_joblib(tqdm(total=len(sample_wells), desc="guarded calibrated PF wells")):
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS, batch_size=1)(
            delayed(process_well)(wid) for wid in sample_wells
        )

failures = pd.DataFrame([r for r in results if not r["ok"]])
successes = [r for r in results if r["ok"]]
if len(successes) < 0.9 * len(sample_wells):
    display(failures.head(30))
    raise RuntimeError(f"Too many failures: {len(failures)}")

parts = []
diag = []
for result in successes:
    g = base_lookup[result["well"]][
        ["well", "fold", "row_idx", "tvt_true", "weighted_mean"]
    ].copy()
    for config in CONFIGS:
        g[config["name"]] = result[config["name"]]
    parts.append(g)
    diag.append({k: v for k, v in result.items() if k not in {c["name"] for c in CONFIGS} and k not in {"ok", "row_idx"}})
rows = pd.concat(parts, ignore_index=True)
diag = pd.DataFrame(diag)

base_col = "weighted_mean"
base_score = rmse(rows["tvt_true"], rows[base_col])
well_sse = (
    rows.assign(sq=(rows["tvt_true"] - rows[base_col]) ** 2)
    .groupby("well", as_index=False)["sq"].sum()
)
top_wells = set(well_sse.nlargest(40, "sq")["well"])
top_rows = rows[rows["well"].isin(top_wells)]
top_base = rmse(top_rows["tvt_true"], top_rows[base_col])

records = []
fold_records = []
for config in CONFIGS:
    candidate = config["name"]
    gains = []
    for fold, fg in rows.groupby("fold", sort=True):
        ref = rmse(fg["tvt_true"], fg[base_col])
        score = rmse(fg["tvt_true"], fg[candidate])
        gain = ref - score
        gains.append(gain)
        fold_records.append(
            {"candidate": candidate, "fold": int(fold), "base_rmse": ref,
             "rmse": score, "gain": gain, "rows": len(fg)}
        )
    score = rmse(rows["tvt_true"], rows[candidate])
    top_score = rmse(top_rows["tvt_true"], top_rows[candidate])
    selected_wells = set(diag.loc[diag["prefix_gain"] >= config["gain_thr"], "well"])
    selected = rows["well"].isin(selected_wells)
    selected_gain = (
        rmse(rows.loc[selected, "tvt_true"], rows.loc[selected, base_col])
        - rmse(rows.loc[selected, "tvt_true"], rows.loc[selected, candidate])
        if selected.any() else 0.0
    )
    records.append(
        {
            "candidate": candidate,
            "gain_thr": config["gain_thr"],
            "shrink": config["shrink"],
            "base_rmse": base_score,
            "rmse": score,
            "gain": base_score - score,
            "mean_fold_gain": float(np.mean(gains)),
            "min_fold_gain": float(np.min(gains)),
            "positive_folds": int(np.sum(np.asarray(gains) > 0)),
            "top40_base_rmse": top_base,
            "top40_rmse": top_score,
            "top40_gain": top_base - top_score,
            "selected_wells": len(selected_wells),
            "selected_rows": int(selected.sum()),
            "selected_eval_gain": selected_gain,
            "mean_abs_change": float(np.mean(np.abs(rows[candidate] - rows[base_col]))),
        }
    )

summary = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
summary["strict_gate"] = (
    (summary["gain"] >= 0.15)
    & (summary["min_fold_gain"] >= 0.05)
    & (summary["positive_folds"] == 5)
    & (summary["top40_gain"] > 0)
    & (summary["selected_eval_gain"] > 0.10)
)
fold_df = pd.DataFrame(fold_records)

display(summary)
display(fold_df)
display(
    diag[["prefix_gain", "slope", "intercept", "elapsed_sec"]]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
)

summary.to_csv(OOF_DIR / "gr_guarded_calibration_pf_summary.csv", index=False)
fold_df.to_csv(OOF_DIR / "gr_guarded_calibration_pf_folds.csv", index=False)
diag.to_csv(OOF_DIR / "gr_guarded_calibration_pf_bywell.csv", index=False)
rows.to_pickle(OOF_DIR / "gr_guarded_calibration_pf_rows.pkl")
failures.to_csv(OOF_DIR / "gr_guarded_calibration_pf_failures.csv", index=False)
print("STRICT GATE PASS:", summary.loc[summary["strict_gate"], "candidate"].tolist())
