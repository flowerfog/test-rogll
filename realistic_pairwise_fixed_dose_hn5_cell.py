from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE = ROOT / "realistic_pairwise_mainline_fusion_hn4" / "aligned_oof.parquet"
OUT_DIR = ROOT / "realistic_pairwise_fixed_dose_hn5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = np.round(np.arange(0.0, 0.601, 0.025), 3)
MAXIMUM_P95_CHANGE = 0.20
MINIMUM_FOLD_GAIN = 0.02
MINIMUM_FAR_GAIN = 0.02


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    well_rmse = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    return {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


started = time.time()
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE)
required = {
    "well", "hn_fold", "tvt_true", "md_since", "base_6p377", "hn3_pairwise"
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Missing HN4 columns: {missing}")

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["base_6p377"].to_numpy(np.float64)
candidate = frame["hn3_pairwise"].to_numpy(np.float64)
delta = candidate - base
base_error = base - truth
analytic_weight = float(np.clip(-np.dot(base_error, delta) / np.dot(delta, delta), 0.0, 1.0))

base_all = _metrics(frame, base)
fold_base = {
    int(fold): _metrics(frame.loc[frame["hn_fold"].eq(fold)].reset_index(drop=True), base[frame["hn_fold"].eq(fold)])
    for fold in sorted(frame["hn_fold"].unique())
}

rows = []
for weight in WEIGHTS:
    prediction = base + float(weight) * delta
    overall = _metrics(frame, prediction)
    local = []
    for fold in sorted(frame["hn_fold"].unique()):
        mask = frame["hn_fold"].to_numpy(int) == int(fold)
        held = frame.loc[mask].reset_index(drop=True)
        result = _metrics(held, prediction[mask])
        reference = fold_base[int(fold)]
        local.append(
            {
                "fold": int(fold),
                "gain": reference["rmse"] - result["rmse"],
                "far_gain": reference["far_gt2000_rmse"] - result["far_gt2000_rmse"],
                "p95_change": result["p95_well_rmse"] - reference["p95_well_rmse"],
            }
        )
    gains = np.asarray([item["gain"] for item in local])
    far_gains = np.asarray([item["far_gain"] for item in local])
    p95_changes = np.asarray([item["p95_change"] for item in local])
    robust = bool(
        gains.min() >= MINIMUM_FOLD_GAIN
        and far_gains.min() >= MINIMUM_FAR_GAIN
        and p95_changes.max() <= MAXIMUM_P95_CHANGE
    )
    rows.append(
        {
            "weight": float(weight),
            "rmse": overall["rmse"],
            "gain": base_all["rmse"] - overall["rmse"],
            "far_rmse": overall["far_gt2000_rmse"],
            "far_gain": base_all["far_gt2000_rmse"] - overall["far_gt2000_rmse"],
            "minimum_fold_gain": float(gains.min()),
            "minimum_far_gain": float(far_gains.min()),
            "maximum_p95_change": float(p95_changes.max()),
            "positive_folds": int((gains > 0).sum()),
            "robust": robust,
        }
    )

scan = pd.DataFrame(rows)
scan.to_csv(OUT_DIR / "dose_scan.csv", index=False)
eligible = scan.loc[scan["robust"]].copy()
if len(eligible):
    selected = eligible.sort_values(
        ["minimum_fold_gain", "gain", "weight"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
else:
    selected = scan.sort_values(
        ["minimum_fold_gain", "gain", "weight"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]

selected_weight = float(selected["weight"])
selected_prediction = base + selected_weight * delta
selected_metrics = _metrics(frame, selected_prediction)
fold_rows = []
for fold in sorted(frame["hn_fold"].unique()):
    mask = frame["hn_fold"].to_numpy(int) == int(fold)
    held = frame.loc[mask].reset_index(drop=True)
    result = _metrics(held, selected_prediction[mask])
    reference = fold_base[int(fold)]
    fold_rows.append(
        {
            "fold": int(fold),
            "weight": selected_weight,
            "rows": int(mask.sum()),
            "wells": int(held["well"].nunique()),
            "base_rmse": reference["rmse"],
            "rmse": result["rmse"],
            "gain": reference["rmse"] - result["rmse"],
            "base_far_rmse": reference["far_gt2000_rmse"],
            "far_rmse": result["far_gt2000_rmse"],
            "far_gain": reference["far_gt2000_rmse"] - result["far_gt2000_rmse"],
            "p95_change": result["p95_well_rmse"] - reference["p95_well_rmse"],
        }
    )
fold_frame = pd.DataFrame(fold_rows)
fold_frame.to_csv(OUT_DIR / "selected_folds.csv", index=False)

checks = {
    "robust_fixed_weight_exists": bool(len(eligible)),
    "gain_ge_0p30": base_all["rmse"] - selected_metrics["rmse"] >= 0.30,
    "far_gain_ge_0p30": base_all["far_gt2000_rmse"] - selected_metrics["far_gt2000_rmse"] >= 0.30,
    "minimum_fold_gain_ge_0p02": float(fold_frame["gain"].min()) >= 0.02,
    "minimum_far_gain_ge_0p02": float(fold_frame["far_gain"].min()) >= 0.02,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change"].max()) <= 0.20,
}
verdict = {
    "experiment": "realistic_pairwise_fixed_dose_hn5",
    "stage": "retrospective_fourfold_fixed_weight_robustness",
    "source": SOURCE,
    "validation_status": "retrospective only; exact mainline overlap",
    "weights": WEIGHTS.tolist(),
    "analytic_unconstrained_weight": analytic_weight,
    "selected_weight": selected_weight,
    "selection_rule": "maximize minimum fold gain among far- and p95-safe doses",
    "base": base_all,
    "selected": selected_metrics,
    "gain": base_all["rmse"] - selected_metrics["rmse"],
    "far_gain": base_all["far_gt2000_rmse"] - selected_metrics["far_gt2000_rmse"],
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> train one all-fold evidence-only tournament model and package hidden inference at the frozen weight"
        if all(checks.values())
        else "FAIL -> retain HN3 as an audit endpoint only"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("HN5 DOSE SCAN")
print(scan.to_string(index=False), flush=True)
print("\nHN5 SELECTED FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nHN5 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
