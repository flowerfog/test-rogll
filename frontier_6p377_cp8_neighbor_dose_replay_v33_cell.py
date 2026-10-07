"""Retrospective CP8 -> Neighbor dose replay for the current 6.377 lineage.

The full CP endpoint parquet belongs to the approximately 6.39 pre-CP8,
pre-Neighbor pipeline.  This audit reconstructs the missing order explicitly:

    full_cp8_ordered -> frozen Neighbor absolute path -> tail dose

Only the Neighbor dose changes.  The selected neighbor, availability mask,
1000-ft prefix freeze, 250-ft transition and 24-ft cap are inherited from V26.
All available folds have already been observed, so this is a retrospective
robustness audit rather than a fresh CV claim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
NEIGHBOR_PATH = (
    OOF_ROOT / "frontier_6p39_tail_only_neighbor_v26" / "tail_neighbor_oof.parquet"
)
CP_PATH = (
    OOF_ROOT
    / "frontier_6p39_full_oof_cp_increment_v18"
    / "full_oof_cp_endpoints.parquet"
)
OUT_DIR = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CP_COLUMN = "full_cp8_ordered"
CP_FALLBACK_COLUMN = "full_parent_matched_vp_replay"
CURRENT_DOSE = 0.20
START_FT = 1000.0
TRANSITION_FT = 250.0
CAP_FT = 24.0
DOSES = np.round(np.arange(0.0, 0.401, 0.025), 3)
FOLDS = (3, 2, 1)

MIN_EACH_FOLD_GAIN = 0.02
MIN_EACH_FAR_GAIN = 0.02
P95_TOLERANCE_FT = 0.10


def smoothstep(distance: np.ndarray) -> np.ndarray:
    t = np.clip((distance - START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    per_well = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    near = distance <= START_FT
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


for path in (NEIGHBOR_PATH, CP_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)

neighbor_frame = pd.read_parquet(
    NEIGHBOR_PATH,
    columns=[
        "id",
        "well",
        "row_idx",
        "fold",
        "tvt_true",
        "md_since",
        "neighbor_absolute",
        "old_active",
    ],
)
cp_frame = pd.read_parquet(
    CP_PATH, columns=["id", CP_COLUMN, CP_FALLBACK_COLUMN]
)
neighbor_frame["id"] = neighbor_frame["id"].astype(str)
cp_frame["id"] = cp_frame["id"].astype(str)
frame = neighbor_frame.merge(cp_frame, on="id", how="left", validate="one_to_one")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids after alignment")
if frame[["tvt_true", "md_since"]].isna().any().any():
    raise RuntimeError("Truth or distance is missing after row alignment")

active = frame["old_active"].fillna(False).to_numpy(bool)
cp_raw = frame[CP_COLUMN].to_numpy(np.float64)
cp_fallback = frame[CP_FALLBACK_COLUMN].to_numpy(np.float64)
cp_missing = ~np.isfinite(cp_raw)
fallback_missing = ~np.isfinite(cp_fallback)
if np.any(cp_missing & fallback_missing):
    examples = frame.loc[
        cp_missing & fallback_missing, ["id", "well", "row_idx"]
    ].head(20)
    raise RuntimeError(
        "CP8 and its pre-CP8 fallback are both missing:\n" + examples.to_string(index=False)
    )
cp8 = np.where(cp_missing, cp_fallback, cp_raw)

neighbor_raw = frame["neighbor_absolute"].to_numpy(np.float64)
neighbor_missing = ~np.isfinite(neighbor_raw)
if np.any(neighbor_missing & active):
    examples = frame.loc[
        neighbor_missing & active, ["id", "well", "row_idx"]
    ].head(20)
    raise RuntimeError(
        "Active Neighbor rows have no absolute prediction:\n"
        + examples.to_string(index=False)
    )
# Inactive rows must remain exactly on the CP8 upstream path.
neighbor = np.where(neighbor_missing, cp8, neighbor_raw)

alignment_diagnostics = {
    "rows": int(len(frame)),
    "cp8_missing_fallback_rows": int(cp_missing.sum()),
    "cp8_missing_fallback_wells": int(frame.loc[cp_missing, "well"].nunique()),
    "neighbor_missing_inactive_rows": int((neighbor_missing & ~active).sum()),
    "neighbor_missing_active_rows": int((neighbor_missing & active).sum()),
    "active_rows": int(active.sum()),
}
print("V33 ALIGNMENT", alignment_diagnostics, flush=True)

truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)

# Rebase the frozen absolute Neighbor path on the CP8-updated upstream path.
# V26's old_active retains the original coverage/eligibility decisions.
raw_move = np.where(active, neighbor - cp8, 0.0)
capped_move = np.clip(raw_move, -CAP_FT, CAP_FT)
tail_profile = smoothstep(distance)


def prediction(dose: float) -> np.ndarray:
    return cp8 + float(dose) * tail_profile * capped_move


current = prediction(CURRENT_DOSE)
prefix = distance <= START_FT
if not np.array_equal(current[prefix], cp8[prefix]):
    raise RuntimeError("The first 1000 ft changed during Neighbor replay")

print(
    "V33 CP8-NEIGHBOR CONFIG",
    {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "folds": sorted(np.unique(fold).tolist()),
        "pre_neighbor": CP_COLUMN,
        "cp8_fallback": CP_FALLBACK_COLUMN,
        "current_neighbor_dose": CURRENT_DOSE,
        "active_fraction": float(active.mean()),
        "start_ft": START_FT,
        "transition_ft": TRANSITION_FT,
        "cap_ft": CAP_FT,
    },
    flush=True,
)

current_metrics = {
    held: metrics(frame.loc[fold == held].reset_index(drop=True), current[fold == held])
    for held in FOLDS
}
rows = []
for dose in DOSES:
    candidate = prediction(float(dose))
    local_rows = []
    for held in FOLDS:
        mask = fold == held
        result = metrics(frame.loc[mask].reset_index(drop=True), candidate[mask])
        reference = current_metrics[held]
        row = {
            "dose": float(dose),
            "fold": int(held),
            **result,
            "gain_vs_current_0p20": reference["rmse"] - result["rmse"],
            "far_gain_vs_current_0p20": (
                reference["far_gt2000_rmse"] - result["far_gt2000_rmse"]
            ),
            "p95_change_vs_current_0p20": (
                result["p95_well_rmse"] - reference["p95_well_rmse"]
            ),
            "near_change_vs_current_0p20": (
                result["near_le1000_rmse"] - reference["near_le1000_rmse"]
            ),
        }
        local_rows.append(row)

    gains = np.array([row["gain_vs_current_0p20"] for row in local_rows])
    far_gains = np.array([row["far_gain_vs_current_0p20"] for row in local_rows])
    p95_changes = np.array([row["p95_change_vs_current_0p20"] for row in local_rows])
    near_changes = np.array([row["near_change_vs_current_0p20"] for row in local_rows])
    robust = bool(
        gains.min() >= MIN_EACH_FOLD_GAIN
        and far_gains.min() >= MIN_EACH_FAR_GAIN
        and p95_changes.max() <= P95_TOLERANCE_FT
        and np.max(np.abs(near_changes)) <= 1e-8
    )
    for row in local_rows:
        row["robust"] = robust
        rows.append(row)

scan = pd.DataFrame(rows)
summary = (
    scan.groupby("dose", as_index=False)
    .agg(
        mean_gain=("gain_vs_current_0p20", "mean"),
        minimum_gain=("gain_vs_current_0p20", "min"),
        mean_far_gain=("far_gain_vs_current_0p20", "mean"),
        minimum_far_gain=("far_gain_vs_current_0p20", "min"),
        maximum_p95_change=("p95_change_vs_current_0p20", "max"),
        maximum_abs_near_change=(
            "near_change_vs_current_0p20", lambda x: float(np.max(np.abs(x)))
        ),
        robust=("robust", "all"),
    )
)
eligible = summary[summary["robust"]]
if len(eligible):
    selected = eligible.sort_values(
        ["minimum_gain", "mean_gain", "dose"], ascending=[False, False, True]
    ).iloc[0]
else:
    # No deployment claim: retain the current notebook dose.
    selected = summary.loc[np.isclose(summary["dose"], CURRENT_DOSE)].iloc[0]

selected_dose = float(selected["dose"])
selected_prediction = prediction(selected_dose)
selected_by_fold = scan[np.isclose(scan["dose"], selected_dose)].copy()
strict_pass = bool(selected["robust"])

scan.to_csv(OUT_DIR / "dose_by_fold.csv", index=False)
summary.to_csv(OUT_DIR / "dose_summary.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    pre_neighbor_cp8=cp8.astype(np.float32),
    neighbor_absolute=neighbor.astype(np.float32),
    current_dose_0p20=current.astype(np.float32),
    selected_neighbor_dose=selected_prediction.astype(np.float32),
    neighbor_active=active,
).to_parquet(OUT_DIR / "cp8_neighbor_replay_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "frontier_6p377_cp8_neighbor_dose_replay_v33",
    "stage": "retrospective_cp8_then_frozen_neighbor_dose_replay",
    "lineage": ["approximately_6p39_parent", CP_COLUMN, "frozen_neighbor_absolute"],
    "sources": {"cp8": str(CP_PATH), "neighbor": str(NEIGHBOR_PATH)},
    "alignment": alignment_diagnostics,
    "current_dose": CURRENT_DOSE,
    "selected_dose": selected_dose,
    "selected_summary": {
        key: (bool(value) if key == "robust" else float(value))
        for key, value in selected.to_dict().items()
    },
    "selected_by_fold": selected_by_fold.to_dict(orient="records"),
    "strict_pass": strict_pass,
    "validation_status": "retrospective_only_folds_1_2_3_already_observed",
    "limitations": (
        "This reuses V26's frozen absolute Neighbor path and eligibility. "
        "It tests final-dose ordering without rerunning neighbor selection."
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> translate selected/current dose ratio into the final notebook"
        if strict_pass
        else "FAIL -> retain the notebook's current Neighbor dose 0.20"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nV33 CP8-NEIGHBOR DOSE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV33 SELECTED BY FOLD")
print(selected_by_fold.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
