"""Cross-fitted local structural-event correction around the current OOF path.

This is a cheap mechanism gateway, not a deployment model.  It asks whether
local trajectory changes plus calibrated GR/typewell mismatch contain enough
legal evidence to correct an already strong path.  Every row prediction and
every correction dose is selected without using the held well.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
PREDICTION_PATH = ROOT / "full_oof_cp_crossfit_viewer.csv"
OUT_DIR = ROOT / "output" / "geology_event_residual_gateway_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_POINTS = 64
WINDOWS_FT = (64.0, 128.0, 256.0, 512.0)
DOSES = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
RAMPS = (0.0, 0.5, 1.0)
CORRECTION_CAP_FT = 48.0
SEED = 20260725


def stable_fold(well: str) -> int:
    digest = hashlib.sha256(well.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 + 1


def finite_interp(x, y, query):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full(np.asarray(query).shape, np.nan)
    order = np.argsort(x[valid], kind="mergesort")
    xv = x[valid][order]
    yv = y[valid][order]
    xv, index = np.unique(xv, return_index=True)
    return np.interp(query, xv, yv[index], left=np.nan, right=np.nan)


def robust_location_scale(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return 0.0, 1.0
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values), 1.0))
    return center, scale


def local_stats(md, values, centers, radius):
    output = []
    for center in centers:
        mask = (md >= center - radius) & (md <= center + radius) & np.isfinite(values)
        before = mask & (md <= center)
        after = mask & (md > center)
        selected = values[mask]
        if len(selected) < 4:
            output.append([0.0] * 8)
            continue
        mean = float(np.mean(selected))
        std = float(np.std(selected))
        bmean = float(np.mean(values[before])) if before.sum() >= 2 else mean
        amean = float(np.mean(values[after])) if after.sum() >= 2 else mean
        q10, q50, q90 = np.quantile(selected, [0.1, 0.5, 0.9])
        output.append(
            [mean, std, bmean, amean, amean - bmean, float(q10), float(q50), float(q90)]
        )
    return np.asarray(output, np.float32)


def slopes_at_centers(md, values, centers, window_ft):
    result = np.full(len(centers), np.nan)
    half = window_ft / 2.0
    for index, center in enumerate(centers):
        mask = (md >= center - half) & (md <= center + half) & np.isfinite(values)
        if mask.sum() >= 4 and np.ptp(md[mask]) > 0:
            result[index] = np.polyfit(md[mask], values[mask], 1)[0]
    return result


def calibrate_typewell(horizontal, typewell, known):
    tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].to_numpy(float)
    h_tvt = horizontal["TVT_input"].to_numpy(float)[known]
    h_gr = horizontal["GR"].to_numpy(float)[known]
    reference = finite_interp(tw_tvt, tw_gr, h_tvt)
    valid = np.isfinite(reference) & np.isfinite(h_gr)
    if valid.sum() >= 30 and np.std(reference[valid]) > 1e-3:
        design = np.column_stack([reference[valid], np.ones(valid.sum())])
        slope, intercept = np.linalg.lstsq(design, h_gr[valid], rcond=None)[0]
        slope = float(np.clip(slope, 0.25, 2.0))
        intercept = float(np.clip(intercept, -100.0, 100.0))
    else:
        slope, intercept = 1.0, 0.0
    return tw_tvt, slope * tw_gr + intercept, slope, intercept


def load_predictions():
    frame = pd.read_csv(PREDICTION_PATH, dtype={"id": str})
    split = frame["id"].str.rsplit("_", n=1, expand=True)
    frame["well"] = split[0].str.lower()
    frame["row_idx"] = pd.to_numeric(split[1], errors="raise").astype(np.int32)
    return {
        well: group.sort_values("row_idx")[["row_idx", "tvt"]].to_numpy(float)
        for well, group in frame.groupby("well", sort=False)
    }


def build_well(path, prediction):
    well = path.name.split("__", 1)[0].lower()
    horizontal = pd.read_csv(path)
    typewell = pd.read_csv(TRAIN_DIR / f"{well}__typewell.csv")
    for column in ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"):
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    md = horizontal["MD"].to_numpy(float)
    z = horizontal["Z"].to_numpy(float)
    gr = horizontal["GR"].to_numpy(float)
    truth = horizontal["TVT"].to_numpy(float)
    tvt_input = horizontal["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    anchor = int(known[-1])
    rows = prediction[:, 0].astype(np.int64)
    base = prediction[:, 1]
    valid = (rows >= 0) & (rows < len(md)) & np.isfinite(base) & np.isfinite(truth[rows])
    rows = rows[valid]
    base = base[valid]
    order = np.argsort(rows)
    rows = rows[order]
    base = base[order]
    eval_md = md[rows]
    target = truth[rows]
    distance = eval_md - md[anchor]
    span = max(float(distance[-1]), 1.0)

    node_md = np.linspace(eval_md[0], eval_md[-1], CONTROL_POINTS)
    node_rows = np.searchsorted(md, node_md)
    node_rows = np.clip(node_rows, rows[0], rows[-1])
    node_md = md[node_rows]
    node_base = np.interp(node_md, eval_md, base)
    node_truth = truth[node_rows]
    node_z = z[node_rows]
    base_surface = base + z[rows]
    node_base_surface = np.interp(node_md, eval_md, base_surface)

    gr_center, gr_scale = robust_location_scale(gr)
    gr_norm = (gr - gr_center) / gr_scale
    tw_tvt, tw_gr, cal_slope, cal_intercept = calibrate_typewell(horizontal, typewell, known)
    tw_at_base = finite_interp(tw_tvt, tw_gr, base)
    mismatch = gr[rows] - tw_at_base
    mismatch_norm = mismatch / gr_scale

    dx = np.gradient(horizontal["X"].to_numpy(float), md, edge_order=1)
    dy = np.gradient(horizontal["Y"].to_numpy(float), md, edge_order=1)
    dz = np.gradient(z, md, edge_order=1)
    ddz = np.gradient(dz, md, edge_order=1) * 1000.0
    dogleg = np.sqrt(
        np.gradient(dx, md, edge_order=1) ** 2
        + np.gradient(dy, md, edge_order=1) ** 2
        + np.gradient(dz, md, edge_order=1) ** 2
    ) * 1000.0

    parts = [
        np.column_stack(
            [
                (node_md - md[anchor]) / 1000.0,
                (node_md - md[anchor]) / span,
                node_z / 10000.0,
                dx[node_rows], dy[node_rows], dz[node_rows],
                ddz[node_rows], dogleg[node_rows],
                node_base / 10000.0,
                node_base_surface / 10000.0,
                np.full(len(node_md), np.mean(~np.isfinite(gr[rows]))),
                np.full(len(node_md), gr_scale / 100.0),
                np.full(len(node_md), cal_slope),
                np.full(len(node_md), cal_intercept / 100.0),
            ]
        )
    ]
    names = [
        "md_since_kft", "progress", "z_10k", "dx", "dy", "dz", "ddz_1k",
        "dogleg_1k", "base_tvt_10k", "base_surface_10k", "eval_gr_missing",
        "gr_scale_100", "tw_cal_slope", "tw_cal_intercept_100",
    ]
    for window in WINDOWS_FT:
        rate_at_node = slopes_at_centers(eval_md, base_surface, node_md, window)
        zrate_at_node = slopes_at_centers(md, z, node_md, window)
        parts.append(np.column_stack([rate_at_node, zrate_at_node]))
        names.extend([f"base_surface_rate_{int(window)}", f"z_rate_{int(window)}"])
        for prefix, values in (("gr", gr_norm[rows]), ("misfit", mismatch_norm)):
            stats = local_stats(eval_md, values, node_md, window / 2.0)
            parts.append(stats)
            names.extend(
                [f"{prefix}_{suffix}_{int(window)}" for suffix in
                 ("mean", "std", "before", "after", "change", "q10", "q50", "q90")]
            )
    features = np.column_stack(parts).astype(np.float32)
    features[~np.isfinite(features)] = np.nan
    target_correction = (node_truth - node_base).astype(np.float32)
    true_surface = node_truth + node_z
    true_rate = np.gradient(true_surface, node_md, edge_order=1)
    true_event = np.abs(np.gradient(true_rate, node_md, edge_order=1) * 256.0) > 0.015
    return {
        "well": well,
        "fold": stable_fold(well),
        "features": features,
        "feature_names": names,
        "target_correction": target_correction,
        "true_event": true_event.astype(np.int8),
        "node_md": node_md,
        "eval_md": eval_md,
        "eval_rows": rows,
        "base": base,
        "truth": target,
        "distance": distance,
    }


def fit_predict(records, feature_indices):
    node_predictions = {record["well"]: np.full(CONTROL_POINTS, np.nan) for record in records}
    histories = []
    for fold in range(1, 6):
        train_records = [record for record in records if record["fold"] != fold]
        valid_records = [record for record in records if record["fold"] == fold]
        x_train = np.concatenate([record["features"][:, feature_indices] for record in train_records])
        y_train = np.concatenate([record["target_correction"] for record in train_records])
        x_valid = np.concatenate([record["features"][:, feature_indices] for record in valid_records])
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=10.0,
            random_state=SEED + fold,
        )
        model.fit(x_train, y_train)
        prediction = np.clip(model.predict(x_valid), -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
        offset = 0
        for record in valid_records:
            values = prediction[offset: offset + CONTROL_POINTS].copy()
            offset += CONTROL_POINTS
            values[0] = 0.0
            node_predictions[record["well"]] = values
        histories.append({"fold": fold, "train_nodes": len(y_train), "valid_nodes": len(prediction)})
    return node_predictions, histories


def expanded_rows(records, node_predictions):
    frames = []
    for record in records:
        correction = np.interp(record["eval_md"], record["node_md"], node_predictions[record["well"]])
        taper = np.clip(record["distance"] / 128.0, 0.0, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "well": record["well"], "fold": record["fold"],
                    "row_idx": record["eval_rows"], "truth": record["truth"],
                    "base": record["base"], "correction": correction * taper,
                    "distance": record["distance"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def metrics(frame, prediction):
    error = np.asarray(prediction, float) - frame["truth"].to_numpy(float)
    result = {
        "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.sum(error**2)), "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }
    per_well = pd.Series(error**2).groupby(frame["well"].to_numpy()).mean().pow(0.5)
    result["median_well_rmse"] = float(per_well.median())
    result["p95_well_rmse"] = float(per_well.quantile(0.95))
    for name, mask in (
        ("near_0_500_rmse", frame["distance"] <= 500),
        ("mid_500_1000_rmse", (frame["distance"] > 500) & (frame["distance"] <= 1000)),
        ("mid_1000_2000_rmse", (frame["distance"] > 1000) & (frame["distance"] <= 2000)),
        ("far_gt2000_rmse", frame["distance"] > 2000),
    ):
        result[name] = float(np.sqrt(np.mean(error[np.asarray(mask)] ** 2)))
    return result


def nested_dose(frame):
    output = np.empty(len(frame), float)
    choices = []
    for fold in range(1, 6):
        held = frame["fold"].eq(fold).to_numpy()
        selection = ~held
        best = None
        for dose in DOSES:
            for ramp in RAMPS:
                progress = np.clip(frame.loc[selection, "distance"].to_numpy() / 2000.0, 0.0, 1.0)
                weight = dose * (progress ** ramp if ramp else np.ones_like(progress))
                pred = frame.loc[selection, "base"].to_numpy() + weight * frame.loc[selection, "correction"].to_numpy()
                score = metrics(frame.loc[selection].reset_index(drop=True), pred)["rmse"]
                candidate = (score, dose, ramp)
                if best is None or candidate < best:
                    best = candidate
        _, dose, ramp = best
        progress = np.clip(frame.loc[held, "distance"].to_numpy() / 2000.0, 0.0, 1.0)
        weight = dose * (progress ** ramp if ramp else np.ones_like(progress))
        output[held] = frame.loc[held, "base"].to_numpy() + weight * frame.loc[held, "correction"].to_numpy()
        choices.append({"fold": fold, "dose": dose, "ramp": ramp, "selection_rmse": best[0]})
    return output, choices


predictions = load_predictions()
records = []
failures = []
paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
for number, path in enumerate(paths, 1):
    well = path.name.split("__", 1)[0].lower()
    try:
        records.append(build_well(path, predictions[well]))
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == len(paths):
        print(f"event features {number}/{len(paths)} failures={len(failures)}", flush=True)
if len(records) < 760:
    raise RuntimeError(f"Only {len(records)} records: {failures[:10]}")

feature_names = records[0]["feature_names"]
all_indices = np.arange(len(feature_names))
no_gr_indices = np.asarray(
    [index for index, name in enumerate(feature_names) if not (name.startswith("gr_") or name.startswith("misfit_"))],
    dtype=int,
)
reports = []
predictions_by_arm = {}
for arm, indices in (("full", all_indices), ("no_gr", no_gr_indices)):
    print(f"train {arm}: features={len(indices)}", flush=True)
    node_prediction, history = fit_predict(records, indices)
    rows = expanded_rows(records, node_prediction)
    candidate, choices = nested_dose(rows)
    predictions_by_arm[arm] = (rows, candidate)
    arm_metrics = metrics(rows, candidate)
    base_metrics = metrics(rows, rows["base"].to_numpy())
    reports.append(
        {
            "arm": arm, "features": len(indices), "base_rmse": base_metrics["rmse"],
            **arm_metrics, "gain": base_metrics["rmse"] - arm_metrics["rmse"],
            "choices": choices, "history": history,
        }
    )

full_rows, full_prediction = predictions_by_arm["full"]
no_gr_rows, no_gr_prediction = predictions_by_arm["no_gr"]
base_metrics = metrics(full_rows, full_rows["base"].to_numpy())
full_metrics = metrics(full_rows, full_prediction)
no_gr_metrics = metrics(no_gr_rows, no_gr_prediction)
fold_report = []
for fold in range(1, 6):
    mask = full_rows["fold"].eq(fold).to_numpy()
    subset = full_rows.loc[mask].reset_index(drop=True)
    baseline = metrics(subset, subset["base"].to_numpy())
    candidate = metrics(subset, full_prediction[mask])
    fold_report.append(
        {"fold": fold, "base_rmse": baseline["rmse"], "candidate_rmse": candidate["rmse"],
         "gain": baseline["rmse"] - candidate["rmse"]}
    )

checks = {
    "overall_gain_ge_0p10": base_metrics["rmse"] - full_metrics["rmse"] >= 0.10,
    "minimum_fold_gain_ge_minus0p05": min(row["gain"] for row in fold_report) >= -0.05,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in fold_report) >= 4,
    "gr_increment_ge_0p03": no_gr_metrics["rmse"] - full_metrics["rmse"] >= 0.03,
    "far_gain_ge_0p20": base_metrics["far_gt2000_rmse"] - full_metrics["far_gt2000_rmse"] >= 0.20,
}
verdict = {
    "experiment": "geology_event_residual_gateway_v1",
    "stage": "fivefold_local_event_correction_mechanism_gateway",
    "sample_unit": "control points nested within complete physical wells",
    "features": feature_names,
    "base": base_metrics,
    "full": full_metrics,
    "no_gr": no_gr_metrics,
    "gain": base_metrics["rmse"] - full_metrics["rmse"],
    "far_gain": base_metrics["far_gt2000_rmse"] - full_metrics["far_gt2000_rmse"],
    "folds": fold_report,
    "arms": reports,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "next": (
        "PASS -> replace coarse interpolation with an explicit smooth/dip-change/jump beam decoder."
        if all(checks.values())
        else "FAIL -> local legal evidence cannot repair structural events; do not build the expensive state decoder."
    ),
}

output = full_rows[["well", "fold", "row_idx", "truth", "base", "distance"]].copy()
output["prediction"] = full_prediction
output["correction"] = full_prediction - output["base"]
pd.DataFrame(fold_report).to_csv(OUT_DIR / "folds.csv", index=False)
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
output.to_pickle(OUT_DIR / "event_correction_oof.pkl", compression="gzip")

print("\nFOLD REPORT")
print(pd.DataFrame(fold_report).to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
