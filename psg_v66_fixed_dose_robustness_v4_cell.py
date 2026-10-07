"""PSG-V4: retrospective robustness check for a conservative fixed PSG dose.

PSG-V3 established directionality at dose 0.40 but failed its fold-p95 guard.
This audit freezes 0.20 as the only candidate dose. Other doses are reported as
diagnostics only and never selected. No untouched-fold or fresh-CV claim is made.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE = ROOT / "psg_v66_exact_delta_gateway_v3" / "psg_v66_delta_oof.parquet"
OUT_DIR = ROOT / "psg_v66_fixed_dose_robustness_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_DOSE = 0.20
DIAGNOSTIC_DOSES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
started = time.time()


def _rmse(error):
    error = np.asarray(error, np.float64)
    return float(np.sqrt(np.mean(error * error)))


def _well_p95(frame, prediction):
    error = np.asarray(prediction, np.float64) - frame["tvt_true"].to_numpy(np.float64)
    codes, names = pd.factorize(frame["well"].astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error * error, minlength=len(names))
    return float(np.quantile(np.sqrt(sse / counts), 0.95))


def _row(frame, dose, fold=None):
    selected = np.ones(len(frame), dtype=bool) if fold is None else frame["fold"].to_numpy() == fold
    local = frame.loc[selected].reset_index(drop=True)
    truth = local["tvt_true"].to_numpy(np.float64)
    base = local["base"].to_numpy(np.float64)
    correction = local["psg_feature_correction"].to_numpy(np.float64)
    candidate = base + dose * correction
    distance = local["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    base_error = base - truth
    candidate_error = candidate - truth
    return {
        "dose": float(dose),
        "fold": 0 if fold is None else int(fold),
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "base_rmse": _rmse(base_error),
        "rmse": _rmse(candidate_error),
        "gain": _rmse(base_error) - _rmse(candidate_error),
        "base_far_rmse": _rmse(base_error[far]),
        "far_rmse": _rmse(candidate_error[far]),
        "far_gain": _rmse(base_error[far]) - _rmse(candidate_error[far]),
        "p95_change": _well_p95(local, candidate) - _well_p95(local, base),
        "mean_abs_move_ft": float(np.mean(np.abs(dose * correction))),
        "p95_abs_move_ft": float(np.quantile(np.abs(dose * correction), 0.95)),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
data = pd.read_parquet(SOURCE)
required = {
    "well", "row_idx", "fold", "tvt_true", "md_since", "base",
    "psg_feature_correction",
}
if missing := sorted(required.difference(data.columns)):
    raise RuntimeError(f"PSG-V3 artifact missing columns: {missing}")
if data["well"].nunique() != 773 or len(data) != 3783989:
    raise RuntimeError(f"Unexpected V3 coverage: rows={len(data)}, wells={data['well'].nunique()}")

rows = []
for dose in DIAGNOSTIC_DOSES:
    rows.append(_row(data, dose))
    for fold in sorted(data["fold"].unique()):
        rows.append(_row(data, dose, int(fold)))
report = pd.DataFrame(rows)
report.to_csv(OUT_DIR / "dose_fold_report.csv", index=False)

primary = report[np.isclose(report["dose"], FROZEN_DOSE)].copy()
overall = primary[primary["fold"] == 0].iloc[0]
folds = primary[primary["fold"] != 0].sort_values("fold")
checks = {
    "overall_gain_ge_0p05": float(overall["gain"]) >= 0.05,
    "overall_far_gain_ge_0p05": float(overall["far_gain"]) >= 0.05,
    "all_fold_gains_nonnegative": float(folds["gain"].min()) >= 0.0,
    "minimum_fold_far_gain_ge_minus0p01": float(folds["far_gain"].min()) >= -0.01,
    "maximum_fold_p95_change_le_0p25": float(folds["p95_change"].max()) <= 0.25,
    "positive_fold_far_gains_ge_3": int((folds["far_gain"] > 0.0).sum()) >= 3,
}
robust = bool(all(checks.values()))

diagnostic_summary = []
for dose, local in report.groupby("dose", sort=True):
    total = local[local["fold"] == 0].iloc[0]
    held = local[local["fold"] != 0]
    diagnostic_summary.append(
        {
            "dose": float(dose),
            "gain": float(total["gain"]),
            "far_gain": float(total["far_gain"]),
            "minimum_fold_gain": float(held["gain"].min()),
            "minimum_fold_far_gain": float(held["far_gain"].min()),
            "maximum_fold_p95_change": float(held["p95_change"].max()),
        }
    )

verdict = {
    "experiment": "psg_v66_fixed_dose_robustness_v4",
    "stage": "retrospective_fixed_0p20_psg_delta_robustness",
    "validation_status": "retrospective_only; all five folds previously observed",
    "selection_or_tuning_in_this_cell": 0,
    "source": str(SOURCE),
    "frozen_dose": FROZEN_DOSE,
    "primary_overall": overall.to_dict(),
    "primary_folds": folds.to_dict("records"),
    "diagnostic_dose_summary": diagnostic_summary,
    "checks": checks,
    "robust": robust,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "ROBUST -> package as one LB-validation candidate, without a new CV claim"
        if robust
        else "FAIL -> close PSG and retain untouched 6.377"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x),
    encoding="utf-8",
)

print("PSG-V4 DOSE SUMMARY")
print(pd.DataFrame(diagnostic_summary).to_string(index=False), flush=True)
print("\nPSG-V4 PRIMARY FOLDS")
print(folds.to_string(index=False), flush=True)
print("\nPSG-V4 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x), flush=True)
print("saved:", OUT_DIR, flush=True)
