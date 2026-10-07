"""Target-free multi-window guard and move-cap rescore for calibrated PF cache."""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
ROWS_PATH = OOF_DIR / "gr_guarded_calibration_pf_rows.pkl"
DIAG_PATH = OOF_DIR / "gr_guarded_calibration_pf_bywell.csv"
PREFIX_EVERY = 3

CONFIGS = [
    {"name": "stable2_cap2", "rule": "stable2", "cap": 2.0},
    {"name": "stable2_cap4", "rule": "stable2", "cap": 4.0},
    {"name": "stable3_cap4", "rule": "stable3", "cap": 4.0},
    {"name": "strong2_cap4", "rule": "strong2", "cap": 4.0},
]

if not ROWS_PATH.exists() or not DIAG_PATH.exists():
    raise RuntimeError("Run gr_guarded_calibration_pf_killtest_cell first")


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


def window_gain(x, y, train_end, valid_start, valid_end):
    n = len(x)
    tr_end = int(train_end * n)
    va_start = int(valid_start * n)
    va_end = int(valid_end * n)
    if tr_end < 50 or va_end - va_start < 30:
        return np.nan
    slope, intercept = robust_affine(x[:tr_end], y[:tr_end])
    identity = rmse(y[va_start:va_end], x[va_start:va_end])
    mapped = rmse(
        y[va_start:va_end],
        slope * x[va_start:va_end] + intercept,
    )
    return identity - mapped


rows = pd.read_pickle(ROWS_PATH)
diag = pd.read_csv(DIAG_PATH, dtype={"well": str})
diag["well"] = diag["well"].astype(str)
wells = sorted(rows["well"].astype(str).unique())

stability_records = []
for wid in tqdm(wells, desc="prefix stability"):
    hw = pd.read_csv(
        DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
        usecols=["GR", "TVT_input"],
    )
    tw = pd.read_csv(
        DATA_ROOT / "train" / f"{wid}__typewell.csv",
        usecols=["TVT", "GR"],
    ).sort_values("TVT")
    known = hw[hw["TVT_input"].notna() & hw["GR"].notna()].iloc[::PREFIX_EVERY]
    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_gr = tw["GR"].interpolate(limit_direction="both").to_numpy(dtype=float)
    x = known["GR"].to_numpy(dtype=float)
    y = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)

    gains = np.array(
        [
            window_gain(x, y, 0.40, 0.40, 0.60),
            window_gain(x, y, 0.60, 0.60, 0.80),
            window_gain(x, y, 0.80, 0.80, 1.00),
        ],
        dtype=float,
    )
    finite = gains[np.isfinite(gains)]
    stability_records.append(
        {
            "well": wid,
            "window_gain_1": gains[0],
            "window_gain_2": gains[1],
            "window_gain_3": gains[2],
            "window_gain_min": float(np.min(finite)) if len(finite) else np.nan,
            "window_gain_median": float(np.median(finite)) if len(finite) else np.nan,
            "positive_windows": int(np.sum(finite > 0)),
        }
    )

stability = pd.DataFrame(stability_records)
diag = diag.merge(stability, on="well", how="left", validate="one_to_one")
diag["stable2"] = (
    (diag["prefix_gain"] >= 0.5)
    & (diag["positive_windows"] >= 2)
    & (diag["window_gain_median"] >= 0.5)
)
diag["stable3"] = (
    (diag["prefix_gain"] >= 0.5)
    & (diag["positive_windows"] == 3)
    & (diag["window_gain_min"] > 0)
    & (diag["window_gain_median"] >= 0.5)
)
diag["strong2"] = (
    (diag["prefix_gain"] >= 0.5)
    & (diag["positive_windows"] >= 2)
    & (diag["window_gain_median"] >= 1.0)
)

print("guard counts:", {rule: int(diag[rule].sum()) for rule in ["stable2", "stable3", "strong2"]})
display(
    diag[["prefix_gain", "window_gain_min", "window_gain_median", "positive_windows"]]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
)

guard_map = {
    rule: dict(zip(diag["well"], diag[rule]))
    for rule in ["stable2", "stable3", "strong2"]
}
base_col = "weighted_mean"
source_col = "guard0p5_half"
base = rows[base_col].to_numpy(dtype=np.float64)
source = rows[source_col].to_numpy(dtype=np.float64)
y_true = rows["tvt_true"].to_numpy(dtype=np.float64)

well_sse = (
    rows.assign(sq=(rows["tvt_true"] - rows[base_col]) ** 2)
    .groupby("well", as_index=False)["sq"]
    .sum()
)
top_wells = set(well_sse.nlargest(40, "sq")["well"])
top_mask = rows["well"].isin(top_wells).to_numpy()
base_score = rmse(y_true, base)
top_base = rmse(y_true[top_mask], base[top_mask])

records = []
fold_records = []
candidate_data = {}
for config in CONFIGS:
    selected = rows["well"].map(guard_map[config["rule"]]).fillna(False).to_numpy(bool)
    move = np.clip(source - base, -config["cap"], config["cap"]) * selected
    pred = base + move
    candidate_data[config["name"]] = pred
    gains = []
    for fold, fg_idx in rows.groupby("fold", sort=True).groups.items():
        idx = np.asarray(fg_idx, dtype=int)
        ref = rmse(y_true[idx], base[idx])
        score = rmse(y_true[idx], pred[idx])
        gain = ref - score
        gains.append(gain)
        fold_records.append(
            {"candidate": config["name"], "fold": int(fold), "base_rmse": ref,
             "rmse": score, "gain": gain, "rows": len(idx)}
        )
    score = rmse(y_true, pred)
    top_score = rmse(y_true[top_mask], pred[top_mask])
    selected_gain = (
        rmse(y_true[selected], base[selected]) - rmse(y_true[selected], pred[selected])
        if selected.any() else 0.0
    )
    records.append(
        {
            "candidate": config["name"],
            "rule": config["rule"],
            "cap": config["cap"],
            "base_rmse": base_score,
            "rmse": score,
            "gain": base_score - score,
            "mean_fold_gain": float(np.mean(gains)),
            "min_fold_gain": float(np.min(gains)),
            "positive_folds": int(np.sum(np.asarray(gains) > 0)),
            "top40_base_rmse": top_base,
            "top40_rmse": top_score,
            "top40_gain": top_base - top_score,
            "selected_wells": int(diag[config["rule"]].sum()),
            "selected_rows": int(selected.sum()),
            "selected_eval_gain": selected_gain,
            "mean_abs_move": float(np.mean(np.abs(move))),
            "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
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

display(summary)
display(pd.DataFrame(fold_records))

diag.to_csv(OOF_DIR / "gr_calibration_prefix_stability_bywell.csv", index=False)
summary.to_csv(OOF_DIR / "gr_calibration_stability_rescore_summary.csv", index=False)
pd.DataFrame(fold_records).to_csv(
    OOF_DIR / "gr_calibration_stability_rescore_folds.csv", index=False
)
print("STRICT GATE PASS:", summary.loc[summary["strict_gate"], "candidate"].tolist())
