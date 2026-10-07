from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_FRAC = 0.70
SUBSAMPLE = 3


def robust_affine(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    keep = np.ones(len(x), dtype=bool)
    coef = np.array([1.0, 0.0])
    for _ in range(6):
        if keep.sum() < 20:
            break
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        coef = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        resid = y - (coef[0] * x + coef[1])
        center = np.median(resid[keep])
        scale = 1.4826 * np.median(np.abs(resid[keep] - center)) + 1e-6
        new_keep = np.abs(resid - center) <= 3.0 * scale
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(coef[0]), float(coef[1])


rows = []
totals = {"identity": [0.0, 0], "affine": [0.0, 0], "quantile": [0.0, 0]}

files = sorted((ROOT / "train").glob("*__horizontal_well.csv"))
for i, hw_path in enumerate(files, 1):
    wid = hw_path.name.split("__")[0]
    tw_path = hw_path.parent / f"{wid}__typewell.csv"
    hw = pd.read_csv(hw_path, usecols=["GR", "TVT_input"])
    tw = pd.read_csv(tw_path, usecols=["TVT", "GR"]).sort_values("TVT")

    known = hw[hw["TVT_input"].notna() & hw["GR"].notna()].copy()
    known = known.iloc[::SUBSAMPLE]
    if len(known) < 100:
        continue
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].interpolate(limit_direction="both").to_numpy(float)
    hz = known["GR"].to_numpy(float)
    target = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)

    cut = max(50, min(len(hz) - 30, int(TRAIN_FRAC * len(hz))))
    x_train, y_train = hz[:cut], target[:cut]
    x_valid, y_valid = hz[cut:], target[cut:]
    if len(x_valid) < 30:
        continue

    slope, intercept = robust_affine(x_train, y_train)
    # Avoid unstable extrapolation while retaining substantial tool mismatch.
    slope_clip = float(np.clip(slope, 0.25, 4.0))
    affine = slope_clip * x_valid + intercept

    xq25, xq50, xq75 = np.quantile(x_train, [0.25, 0.5, 0.75])
    yq25, yq50, yq75 = np.quantile(y_train, [0.25, 0.5, 0.75])
    q_slope = float(np.clip((yq75 - yq25) / max(xq75 - xq25, 1e-6), 0.25, 4.0))
    q_intercept = float(yq50 - q_slope * xq50)
    quantile = q_slope * x_valid + q_intercept

    pred = {
        "identity": x_valid,
        "affine": affine,
        "quantile": quantile,
    }
    record = {
        "well": wid,
        "n_train": len(x_train),
        "n_valid": len(x_valid),
        "slope": slope,
        "slope_clipped": slope_clip,
        "intercept": intercept,
        "q_slope": q_slope,
        "q_intercept": q_intercept,
    }
    for name, values in pred.items():
        err = values - y_valid
        sse = float(np.sum(err**2))
        totals[name][0] += sse
        totals[name][1] += len(err)
        record[f"rmse_{name}"] = float(np.sqrt(np.mean(err**2)))
    record["gain_affine"] = record["rmse_identity"] - record["rmse_affine"]
    record["gain_quantile"] = record["rmse_identity"] - record["rmse_quantile"]
    rows.append(record)

    if i % 100 == 0:
        print(f"{i}/{len(files)}", flush=True)

by_well = pd.DataFrame(rows)
summary = pd.DataFrame(
    [
        {"candidate": name, "rmse": float(np.sqrt(sse / n)), "rows": n}
        for name, (sse, n) in totals.items()
    ]
).sort_values("rmse")

print("\nPOOLED HELD-OUT PREFIX")
print(summary.to_string(index=False))
print("\nBY-WELL GAINS")
print(
    by_well[["gain_affine", "gain_quantile", "slope", "intercept"]]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .to_string()
)
print("affine positive wells", float((by_well["gain_affine"] > 0).mean()))
print("quantile positive wells", float((by_well["gain_quantile"] > 0).mean()))

by_well.to_csv(ROOT / "gr_sensor_calibration_prefix_bywell.csv", index=False)
summary.to_csv(ROOT / "gr_sensor_calibration_prefix_summary.csv", index=False)
