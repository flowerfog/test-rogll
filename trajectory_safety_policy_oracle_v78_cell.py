"""Route 7 Phase 0: conservative trajectory-endpoint activation oracle.

This is an observability audit, not a deployable selector.  It asks whether the
V77b geometry-only endpoints contain enough complementary wells to justify
training an asymmetric safety policy.  Rows through 1000 ft are bitwise frozen;
the candidate dose ramps to full strength at 2000 ft.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
SOURCE = OOF_ROOT / "drilling_trajectory_sensor_gateway_v77b" / "trajectory_sensor_sample_oof.parquet"
EXPERIMENT = "trajectory_safety_policy_oracle_v78"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARMS = ("full_ridge_a10", "full_ridge_a100", "full_extra_leaf4", "full_extra_leaf8")
DOSES = np.round(np.arange(0.0, 0.251, 0.025), 3)
FREEZE_FT = 1000.0
FULL_DOSE_FT = 2000.0

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)


def metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    bywell = (
        pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
        .groupby("well", sort=False)["e2"].mean().to_numpy(np.float64)
    )
    near = frame["md_since"].to_numpy(np.float64) <= FREEZE_FT
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(bywell))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell), 0.95)),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


frame = pd.read_parquet(SOURCE)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", "base", *ARMS}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Missing V77b columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["well"].nunique() < 760 or frame["fold"].nunique() != 5:
    raise RuntimeError(f"Unexpected coverage: {len(frame)}/{frame['well'].nunique()}")

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["base"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
ramp = np.clip((distance - FREEZE_FT) / (FULL_DOSE_FT - FREEZE_FT), 0.0, 1.0)
base_metric = metric(frame, base, "base_v66")

fixed_rows = []
arm_oracle_rows = []
well_rows = []
arm_oracle_predictions = {}
for arm in ARMS:
    candidate = frame[arm].to_numpy(np.float64)
    delta = ramp * (candidate - base)
    for dose in DOSES:
        prediction = base + float(dose) * delta
        result = metric(frame, prediction, f"{arm}_w{dose:g}")
        fixed_rows.append({"arm": arm, "dose": float(dose), **result})

    oracle = base.copy()
    for well, positions in frame.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(positions), np.int64)
        local_truth = truth[positions]
        local_base = base[positions]
        local_delta = delta[positions]
        losses = np.asarray([
            np.mean((local_base + float(dose) * local_delta - local_truth) ** 2)
            for dose in DOSES
        ])
        selected_index = int(np.argmin(losses))
        selected_dose = float(DOSES[selected_index])
        oracle[positions] = local_base + selected_dose * local_delta
        well_rows.append({
            "well": str(well), "fold": int(frame.loc[positions[0], "fold"]),
            "arm": arm, "selected_dose": selected_dose,
            "base_mse": float(np.mean((local_base - local_truth) ** 2)),
            "oracle_mse": float(losses[selected_index]),
            "gain_mse": float(np.mean((local_base - local_truth) ** 2) - losses[selected_index]),
            "delta_rms": float(np.sqrt(np.mean(local_delta * local_delta))),
        })
    arm_oracle_predictions[arm] = oracle
    result = metric(frame, oracle, f"{arm}_per_well_dose_oracle")
    selected = pd.DataFrame([row for row in well_rows if row["arm"] == arm])
    arm_oracle_rows.append({
        "arm": arm,
        "active_well_fraction": float(np.mean(selected["selected_dose"] > 0)),
        "mean_selected_dose": float(selected["selected_dose"].mean()),
        **result,
    })

# Strongest diagnostic: choose one arm and dose per complete well.
union = base.copy()
union_choices = []
for well, positions in frame.groupby("well", sort=False).groups.items():
    positions = np.asarray(list(positions), np.int64)
    local_truth = truth[positions]
    choices = [(float(np.mean((base[positions] - local_truth) ** 2)), "base", 0.0, base[positions])]
    for arm in ARMS:
        candidate = frame.loc[positions, arm].to_numpy(np.float64)
        local_delta = ramp[positions] * (candidate - base[positions])
        for dose in DOSES[1:]:
            prediction = base[positions] + float(dose) * local_delta
            choices.append((float(np.mean((prediction - local_truth) ** 2)), arm, float(dose), prediction))
    loss, arm, dose, prediction = min(choices, key=lambda item: item[0])
    union[positions] = prediction
    union_choices.append({
        "well": str(well), "fold": int(frame.loc[positions[0], "fold"]),
        "arm": arm, "dose": dose, "oracle_mse": loss,
    })

fixed = pd.DataFrame(fixed_rows)
arm_oracles = pd.DataFrame(arm_oracle_rows)
well_choices = pd.DataFrame(well_rows)
union_choices = pd.DataFrame(union_choices)
union_metric = metric(frame, union, "union_arm_and_dose_oracle")
best_fixed = fixed.loc[fixed["rmse"].idxmin()].to_dict()

fold_rows = []
for fold in range(1, 6):
    mask = frame["fold"].to_numpy(np.int16) == fold
    base_fold = metric(frame.loc[mask], base[mask], "base")
    union_fold = metric(frame.loc[mask], union[mask], "union_oracle")
    fold_rows.append({
        "fold": fold,
        "base_rmse": base_fold["rmse"], "rmse": union_fold["rmse"],
        "gain": base_fold["rmse"] - union_fold["rmse"],
        "base_far_rmse": base_fold["far_gt2000_rmse"],
        "far_rmse": union_fold["far_gt2000_rmse"],
        "far_gain": base_fold["far_gt2000_rmse"] - union_fold["far_gt2000_rmse"],
        "p95_change": union_fold["p95_well_rmse"] - base_fold["p95_well_rmse"],
        "active_well_fraction": float(np.mean(union_choices.loc[union_choices["fold"].eq(fold), "dose"] > 0)),
    })
folds = pd.DataFrame(fold_rows)

fixed.to_csv(OUT_DIR / "fixed_dose_scan.csv", index=False)
arm_oracles.to_csv(OUT_DIR / "arm_dose_oracles.csv", index=False)
well_choices.to_csv(OUT_DIR / "arm_well_oracle_choices.csv", index=False)
union_choices.to_csv(OUT_DIR / "union_well_oracle_choices.csv", index=False)
folds.to_csv(OUT_DIR / "union_oracle_folds.csv", index=False)

checks = {
    "union_oracle_gain_ge_0p30": bool(base_metric["rmse"] - union_metric["rmse"] >= 0.30),
    "union_oracle_far_gain_ge_0p40": bool(base_metric["far_gt2000_rmse"] - union_metric["far_gt2000_rmse"] >= 0.40),
    "minimum_fold_oracle_gain_ge_0p15": bool(folds["gain"].min() >= 0.15),
    "active_well_fraction_ge_0p15": bool(np.mean(union_choices["dose"] > 0) >= 0.15),
    "best_fixed_dose_is_zero_or_small": bool(float(best_fixed["dose"]) <= 0.05),
    "near_rows_bitwise_unchanged": bool(np.array_equal(union[distance <= FREEZE_FT], base[distance <= FREEZE_FT])),
}
strict_pass = bool(
    checks["union_oracle_gain_ge_0p30"]
    and checks["union_oracle_far_gain_ge_0p40"]
    and checks["minimum_fold_oracle_gain_ge_0p15"]
    and checks["active_well_fraction_ge_0p15"]
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "route7_phase0_far_only_per_well_arm_and_dose_oracle",
    "truth_usage": "oracle investment decision only; not deployable",
    "source": str(SOURCE),
    "policy_space": {
        "arms": list(ARMS), "doses": DOSES.tolist(),
        "frozen_through_ft": FREEZE_FT, "full_dose_at_ft": FULL_DOSE_FT,
    },
    "coverage": {"rows": len(frame), "wells": frame["well"].nunique()},
    "base": base_metric,
    "best_fixed": best_fixed,
    "arm_oracles": arm_oracles.to_dict("records"),
    "union_oracle": union_metric,
    "union_gain": base_metric["rmse"] - union_metric["rmse"],
    "union_far_gain": base_metric["far_gt2000_rmse"] - union_metric["far_gt2000_rmse"],
    "active_well_fraction": float(np.mean(union_choices["dose"] > 0)),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train a nested asymmetric lower-confidence activation policy"
        if strict_pass else
        "FAIL -> selective geometry dosing lacks enough oracle headroom; close Route 7 for this endpoint"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("V78 ARM ORACLES")
print(arm_oracles.sort_values("rmse").to_string(index=False))
print("\nV78 UNION ORACLE FOLDS")
print(folds.to_string(index=False))
print("\nV78 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
