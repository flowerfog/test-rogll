"""V190b: read-only robust consumption audit of the V190 surface direction."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE_DIR = ROOT / "oof_exports" / "transductive_debiased_surface_gateway_v190"
SOURCE = SOURCE_DIR / "sampled_predictions.parquet"
OUT_DIR = ROOT / "oof_exports" / "transductive_surface_attractor_audit_v190b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOVES = ("joint_no_bias", "joint_debiased")
WINDOWS = (1, 5, 11, 21)
CAPS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
DOSES = (0.25, 0.50, 0.75, 1.00)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _metric(frame, prediction, endpoint):
    truth = frame["truth"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    if not np.isfinite(error).all():
        raise RuntimeError(f"{endpoint}: non-finite prediction")
    by_well = (
        pd.DataFrame({"well": frame["well"], "e2": error * error})
        .groupby("well", sort=False)["e2"]
        .mean()
    )
    far = frame["distance"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
required = {"well", "row_idx", "truth", "distance", "base", *MOVES}
if not required.issubset(frame.columns):
    raise RuntimeError(f"Missing V190 columns: {sorted(required.difference(frame.columns))}")

base = frame["base"].to_numpy(np.float64)
truth = frame["truth"].to_numpy(np.float64)
base_metric = _metric(frame, base, "v168_sample192")
well_codes, wells = pd.factorize(frame["well"], sort=True)
well_indices = [np.flatnonzero(well_codes == code) for code in range(len(wells))]


def _smooth_by_well(move, window):
    if window == 1:
        return move.copy()
    output = np.empty_like(move)
    for indices in well_indices:
        output[indices] = (
            pd.Series(move[indices])
            .rolling(window, center=True, min_periods=1)
            .median()
            .to_numpy(np.float64)
        )
    return output


scan = []
prediction_cache = {}
for source in MOVES:
    raw_move = frame[source].to_numpy(np.float64) - base
    for window in WINDOWS:
        smooth = _smooth_by_well(raw_move, window)
        for cap in CAPS:
            bounded = np.clip(smooth, -cap, cap)
            for dose in DOSES:
                prediction = base + dose * bounded
                metric = _metric(frame, prediction, "surface_attractor")
                recipe = (source, window, cap, dose)
                prediction_cache[recipe] = prediction
                scan.append({
                    **metric,
                    "source": source,
                    "window": window,
                    "cap": cap,
                    "dose": dose,
                    "gain": base_metric["rmse"] - metric["rmse"],
                    "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                    "p95_gain": base_metric["p95_well_rmse"] - metric["p95_well_rmse"],
                })

scan = pd.DataFrame(scan).sort_values(["rmse", "p95_well_rmse"], kind="stable").reset_index(drop=True)
eligible = scan[
    (scan["far_gain"] >= 0.0)
    & (scan["p95_gain"] >= -0.10)
]
selected = (eligible.iloc[0] if len(eligible) else scan.iloc[0]).to_dict()
recipe = (selected["source"], int(selected["window"]), float(selected["cap"]), float(selected["dose"]))
selected_prediction = prediction_cache[recipe]

# Preserve each well's complete move while breaking its spatial assignment.
rolled_prediction = np.empty(len(frame), np.float64)
for position, indices in enumerate(well_indices):
    source_indices = well_indices[(position + max(1, len(well_indices) // 7)) % len(well_indices)]
    source_move = selected_prediction[source_indices] - base[source_indices]
    target_progress = np.linspace(0.0, 1.0, len(indices))
    source_progress = np.linspace(0.0, 1.0, len(source_indices))
    rolled_prediction[indices] = base[indices] + np.interp(target_progress, source_progress, source_move)

selected_metric = _metric(frame, selected_prediction, "selected_attractor")
rolled_metric = _metric(frame, rolled_prediction, "rolled_well_control")
summary = pd.DataFrame([base_metric, selected_metric, rolled_metric])
values = {
    "gain": base_metric["rmse"] - selected_metric["rmse"],
    "far_gain": base_metric["far_gt2000_rmse"] - selected_metric["far_gt2000_rmse"],
    "p95_gain": base_metric["p95_well_rmse"] - selected_metric["p95_well_rmse"],
    "aligned_increment_vs_rolled": rolled_metric["rmse"] - selected_metric["rmse"],
}
checks = {
    "gain_ge_0p15": values["gain"] >= 0.15,
    "far_gain_ge_0p10": values["far_gain"] >= 0.10,
    "p95_not_worse_by_gt_0p10": values["p95_gain"] >= -0.10,
    "aligned_beats_rolled_ge_0p05": values["aligned_increment_vs_rolled"] >= 0.05,
}
verdict = {
    "experiment": "transductive_surface_attractor_audit_v190b",
    "stage": "read_only_Fold3_robust_move_consumption_gate",
    "selection_search_count": len(scan),
    "source": SOURCE,
    "selected_recipe": {
        "source": recipe[0], "median_window": recipe[1], "cap_ft": recipe[2], "dose": recipe[3]
    },
    "values": values,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze recipe and rerun V190 on untouched Fold2"
        if all(checks.values())
        else "FAIL -> close transductive structural-surface route"
    ),
}

scan.head(30).to_csv(OUT_DIR / "top30.csv", index=False)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
(OUT_DIR / "final_verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("V190b TOP 30")
print(scan.head(30).to_string(index=False))
print("\nV190b SUMMARY")
print(summary.to_string(index=False))
print("\nV190b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
print("saved:", OUT_DIR)
