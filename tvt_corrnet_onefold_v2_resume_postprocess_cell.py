"""Resume TVT-CorrNet v2 after the row_idx-only postprocess failure.

All three models and all four required prediction artifacts already exist.
This cell performs no training and does not need a GPU.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "tvt_corrnet_onefold_v2"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
ARMS = ("full", "no_gr", "shuffled_typewell")
INTERVENTIONS = ("full", "no_gr", "shuffled_typewell")
BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _rmse(truth, prediction):
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _cosine(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else np.nan


def _score_rows(frame):
    frame = frame.copy()
    frame["error"] = frame["prediction"] - frame["truth"]
    by_well = frame.groupby("well", sort=False)["error"].agg(
        rows="size",
        rmse=lambda values: float(np.sqrt(np.mean(values**2))),
        mae=lambda values: float(np.mean(np.abs(values))),
    )
    return {
        "rmse": float(np.sqrt(np.mean(frame["error"] ** 2))),
        "mae": float(np.mean(np.abs(frame["error"]))),
        "median_well_rmse": float(by_well["rmse"].median()),
        "p95_well_rmse": float(by_well["rmse"].quantile(0.95)),
    }


arm_frames = {}
arm_rows = []
for arm in ARMS:
    path = OUT_DIR / f"{arm}_fold1_predictions.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_pickle(path)
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = frame["row_idx"].astype(np.int32)
    arm_frames[arm] = frame
    arm_rows.append({"arm": arm, **_score_rows(frame)})
arm_report = pd.DataFrame(arm_rows).sort_values("rmse").reset_index(drop=True)

intervention_frames = {}
intervention_rows = []
for mode in INTERVENTIONS:
    path = OUT_DIR / f"full_intervention_{mode}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_pickle(path)
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = frame["row_idx"].astype(np.int32)
    intervention_frames[mode] = frame
    intervention_rows.append({"mode": mode, **_score_rows(frame)})
intervention_report = pd.DataFrame(intervention_rows)
full_intervention_rmse = float(
    intervention_report.loc[intervention_report["mode"] == "full", "rmse"].iloc[0]
)
intervention_report["degradation_vs_full"] = (
    intervention_report["rmse"] - full_intervention_rmse
)

pool_path = _first_existing(POOL_CANDIDATES)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
reference_columns = [
    name for name in ("strong", "strong_bias", "cp12") if name in header
]
usecols = [
    name
    for name in ("id", "well", "row_idx", "tvt_true", *reference_columns)
    if name in header
]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": str, "well": str})
if "id" not in pool.columns:
    raise RuntimeError("candidate pool must contain id")
pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool.columns:
    pool["row_idx"] = (
        pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    )
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)

full_rows = arm_frames["full"]
valid_wells = set(full_rows["well"].unique())
pool = pool[pool["well"].isin(valid_wells)]
merged = full_rows.merge(
    pool[["well", "row_idx", "tvt_true", *reference_columns]],
    on=["well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
if len(merged) != len(full_rows):
    raise RuntimeError(
        f"reference merge retained {len(merged)} / {len(full_rows)} rows"
    )

truth = merged["truth"].to_numpy(float)
corrnet_prediction = merged["prediction"].to_numpy(float)
corrnet_error = corrnet_prediction - truth
reference_rows = []
blend_rows = []
for base in reference_columns:
    base_prediction = merged[base].to_numpy(float)
    base_error = base_prediction - truth
    base_rmse = _rmse(truth, base_prediction)
    reference_rows.append(
        {
            "model": base,
            "rmse": base_rmse,
            "error_cosine_vs_corrnet": _cosine(base_error, corrnet_error),
        }
    )
    for weight in BLEND_WEIGHTS:
        prediction = (1.0 - weight) * base_prediction + weight * corrnet_prediction
        score = _rmse(truth, prediction)
        blend_rows.append(
            {
                "base": base,
                "weight_corrnet": weight,
                "base_rmse": base_rmse,
                "rmse": score,
                "gain_vs_base": base_rmse - score,
            }
        )
reference_report = pd.DataFrame(reference_rows).sort_values("rmse")
blend_report = pd.DataFrame(blend_rows).sort_values("rmse")

full_rmse = float(arm_report.loc[arm_report["arm"] == "full", "rmse"].iloc[0])
no_gr_rmse = float(arm_report.loc[arm_report["arm"] == "no_gr", "rmse"].iloc[0])
shuffled_rmse = float(
    arm_report.loc[arm_report["arm"] == "shuffled_typewell", "rmse"].iloc[0]
)
intervention_no_gr = float(
    intervention_report.loc[intervention_report["mode"] == "no_gr", "rmse"].iloc[0]
)
intervention_shuffled = float(
    intervention_report.loc[
        intervention_report["mode"] == "shuffled_typewell", "rmse"
    ].iloc[0]
)
best_reference_rmse = float(reference_report["rmse"].min())
best_blend_gain = float(blend_report["gain_vs_base"].max())
predictive_pass = bool(
    full_rmse < best_reference_rmse or best_blend_gain >= 0.10
)
causal_pass = bool(
    no_gr_rmse - full_rmse >= 0.10
    and shuffled_rmse - full_rmse >= 0.10
    and intervention_no_gr - full_rmse >= 0.25
    and intervention_shuffled - full_rmse >= 0.15
)
strict_pass = bool(predictive_pass and causal_pass)

verdict = {
    "experiment": "tvt_corrnet_onefold_v2_resume_postprocess",
    "pool": str(pool_path),
    "rows": len(full_rows),
    "wells": len(valid_wells),
    "full_rmse": full_rmse,
    "best_reference_rmse": best_reference_rmse,
    "best_blend_gain": best_blend_gain,
    "trained_no_gr_degradation": no_gr_rmse - full_rmse,
    "trained_shuffled_typewell_degradation": shuffled_rmse - full_rmse,
    "intervention_no_gr_degradation": intervention_no_gr - full_rmse,
    "intervention_shuffled_typewell_degradation": intervention_shuffled - full_rmse,
    "predictive_pass": predictive_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train five fold CorrNet and test OOF fusion"
        if strict_pass
        else "FAIL -> stop CorrNet OOF route; keep capacity result as diagnostic only"
    ),
}

arm_report.to_csv(OUT_DIR / "arm_scores.csv", index=False)
intervention_report.to_csv(OUT_DIR / "intervention_scores.csv", index=False)
reference_report.to_csv(OUT_DIR / "reference_scores.csv", index=False)
blend_report.to_csv(OUT_DIR / "blend_scan.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)

print("ARM SCORES")
print(arm_report.to_string(index=False))
print("\nFULL-MODEL INTERVENTIONS")
print(intervention_report.to_string(index=False))
print("\nHISTORICAL REFERENCES")
print(reference_report.to_string(index=False))
print("\nBLEND SCAN TOP 20")
print(blend_report.head(20).to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
