"""Matched OOF replay of the fixed PF midpoint part of the 6.39 tail.

Run after the v9 CP8-noCP12 audit. This cell performs no fitting or search.
It applies the exact fixed PF branch-midpoint hedge from the 6.39 notebook to
the parent and CP8 endpoints, then checks whether CP8's gain survives.

The selected 6.39 profile (``vp_balanced_modelpkg_005``) also enables balanced
visible-prefix calibration and a tiny saved-model correction. Those require a
separate pseudo-mask OOF reconstruction and the complete model package; this
cell does not claim to replay them. Hidden-ID edits and same-well hidden
overrides have no honest OOF analogue and are deliberately excluded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p39_cp8_nocp12_postprocess_v11"
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
V9_PATH = (
    OOF_ROOT
    / "frontier_6p39_cp8_nocp12_matched_v9"
    / "cp8_nocp12_oof.parquet"
)
V10_PATH = (
    OOF_ROOT
    / "frontier_6p39_cp8_nocp12_neighbor_v10"
    / "cp8_nocp12_neighbor_oof.parquet"
)
PF_STATS_CANDIDATES = (
    OOF_ROOT
    / "pf_bimodal_midpoint_hedge_oof_v0"
    / "bimodal_well_statistics.csv",
    OOF_ROOT
    / "pf_bimodal_midpoint_hedge_oof_v1"
    / "bimodal_well_statistics.csv",
)
OUT_DIR = OOF_ROOT / EXPERIMENT

# Exact deployed-notebook recipe. These values are frozen, not selected here.
HEDGE_STRENGTH = 0.60
MIN_MINOR_MASS = 0.25
SEPARATION_LOW_FT = 4.0
SEPARATION_HIGH_FT = 40.0
MOVE_CAP_FT = 2.0
ZERO_MOVE_FT = 0.01
SHUFFLE_SEED = 20260724


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")

    per_well_mse = (
        pd.DataFrame({"well": frame["well"].to_numpy(), "se": error * error})
        .groupby("well", sort=False)["se"]
        .mean()
    )
    per_well_rmse = np.sqrt(per_well_mse.to_numpy(np.float64))
    row = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(per_well_rmse)),
        "p75_well_rmse": float(np.quantile(per_well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(per_well_rmse, 0.95)),
        "max_well_rmse": float(np.max(per_well_rmse)),
    }
    bands = (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    )
    distance = frame["md_since"].to_numpy(np.float64)
    for name, low, high in bands:
        mask = (distance > low) & (distance <= high)
        row[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return row


def _worst_twenty(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    error = np.asarray(prediction, np.float64) - frame["tvt_true"].to_numpy(np.float64)
    work = pd.DataFrame(
        {"well": frame["well"].to_numpy(), "se": error * error, "ae": np.abs(error)}
    )
    per_well = work.groupby("well", sort=False).agg(rows=("se", "size"), mse=("se", "mean"))
    per_well["rmse"] = np.sqrt(per_well["mse"])
    count = max(1, int(np.ceil(0.20 * len(per_well))))
    worst_wells = per_well.nlargest(count, "rmse").index
    mask = frame["well"].isin(worst_wells).to_numpy()
    return {
        "endpoint": endpoint,
        "wells": int(count),
        "rows": int(mask.sum()),
        "exact_sse": float(np.dot(error[mask], error[mask])),
        "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
        "mae": float(np.mean(np.abs(error[mask]))),
    }


started = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)
if not V9_PATH.exists():
    raise FileNotFoundError(
        f"Missing {V9_PATH}. Run frontier_6p39_cp8_nocp12_matched_v9 first."
    )

stats_path = next((path for path in PF_STATS_CANDIDATES if path.exists()), None)
if stats_path is None:
    matches = sorted(OOF_ROOT.glob("**/bimodal_well_statistics.csv"))
    stats_path = matches[0] if matches else None
if stats_path is None:
    raise FileNotFoundError(
        "Missing bimodal_well_statistics.csv. Run "
        "pf_bimodal_midpoint_hedge_oof_v0_cell.py once; its per-well PF cache "
        "is reusable and this cell will not rerun the 128-seed PF."
    )

frame = pd.read_parquet(V9_PATH)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "frontier_6p39_core",
    "cp8_nocp12",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V9 OOF is missing columns: {missing}")
frame["well"] = frame["well"].astype(str)

# Prefer the exact distance coordinate persisted by v10. Otherwise OOF rows are
# the contiguous scored suffix, so rank within well is its 1-ft distance proxy.
neighbor_available = False
if V10_PATH.exists():
    v10 = pd.read_parquet(V10_PATH)
    wanted = [column for column in ("id", "cp8_neighbor", "md_since") if column in v10]
    if "id" in wanted and "md_since" in wanted:
        v10 = v10[wanted].drop_duplicates("id")
        frame = frame.merge(v10, on="id", how="left", validate="one_to_one")
        neighbor_available = "cp8_neighbor" in frame and frame["cp8_neighbor"].notna().all()
if "md_since" not in frame or frame["md_since"].isna().any():
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    frame["md_since"] = frame.groupby("well", sort=False).cumcount().astype(np.float32) + 1.0

stats = pd.read_csv(stats_path, dtype={"well": str})
stats_required = {"well", "minor_mass", "separation_ft", "midpoint_delta_ft"}
missing_stats = sorted(stats_required.difference(stats.columns))
if missing_stats:
    raise RuntimeError(f"PF statistics are missing columns: {missing_stats}")
if stats["well"].duplicated().any():
    raise RuntimeError("Duplicate wells in PF statistics")

stats["qualified"] = (
    (stats["minor_mass"] >= MIN_MINOR_MASS)
    & stats["separation_ft"].between(SEPARATION_LOW_FT, SEPARATION_HIGH_FT)
    & np.isfinite(stats["midpoint_delta_ft"])
)
stats["deployed_shift_ft"] = 0.0
stats.loc[stats["qualified"], "deployed_shift_ft"] = np.clip(
    HEDGE_STRENGTH * stats.loc[stats["qualified"], "midpoint_delta_ft"],
    -MOVE_CAP_FT,
    MOVE_CAP_FT,
)
stats.loc[stats["deployed_shift_ft"].abs() < ZERO_MOVE_FT, "deployed_shift_ft"] = 0.0
shift_map = stats.set_index("well")["deployed_shift_ft"]
frame["midpoint_shift_ft"] = frame["well"].map(shift_map).fillna(0.0).astype(np.float32)

parent = frame["frontier_6p39_core"].to_numpy(np.float64)
cp8 = frame["cp8_nocp12"].to_numpy(np.float64)
shift = frame["midpoint_shift_ft"].to_numpy(np.float64)
endpoints = {
    "parent_before": parent,
    "parent_after_midpoint": parent + shift,
    "cp8_before": cp8,
    "cp8_after_midpoint": cp8 + shift,
}
if neighbor_available:
    neighbor = frame["cp8_neighbor"].to_numpy(np.float64)
    endpoints["cp8_neighbor_before"] = neighbor
    endpoints["cp8_neighbor_after_midpoint"] = neighbor + shift

# Matched controls test sign and well assignment without selecting alternatives.
rng = np.random.default_rng(SHUFFLE_SEED)
well_order = frame["well"].drop_duplicates().to_numpy()
well_shift = pd.Series(shift_map.reindex(well_order).fillna(0.0).to_numpy(), index=well_order)
shuffled = pd.Series(rng.permutation(well_shift.to_numpy()), index=well_order)
shuffle_shift = frame["well"].map(shuffled).to_numpy(np.float64)
controls = {
    "cp8_reverse_midpoint": cp8 - shift,
    "cp8_shuffled_midpoint": cp8 + shuffle_shift,
}

summary_rows = []
for fold in sorted(frame["fold"].astype(int).unique()):
    mask = frame["fold"].astype(int).eq(int(fold)).to_numpy()
    fold_frame = frame.loc[mask].reset_index(drop=True)
    for name, prediction in endpoints.items():
        summary_rows.append({"fold": int(fold), **_metric(fold_frame, prediction[mask], name)})
for name, prediction in endpoints.items():
    summary_rows.append({"fold": 0, **_metric(frame, prediction, name)})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_postprocess_summary.csv", index=False)

control_rows = [_metric(frame, prediction, name) for name, prediction in controls.items()]
controls_frame = pd.DataFrame(control_rows)
controls_frame.to_csv(OUT_DIR / "matched_controls.csv", index=False)

worst = pd.DataFrame(
    [_worst_twenty(frame, prediction, name) for name, prediction in endpoints.items()]
)
worst.to_csv(OUT_DIR / "worst20_percent_wells.csv", index=False)

overall = summary[summary["fold"].eq(0)].set_index("endpoint")
fold_table = summary[summary["fold"].ne(0)].pivot(
    index="fold", columns="endpoint", values="rmse"
)
cp_gain_before = float(overall.loc["parent_before", "rmse"] - overall.loc["cp8_before", "rmse"])
cp_gain_after = float(
    overall.loc["parent_after_midpoint", "rmse"]
    - overall.loc["cp8_after_midpoint", "rmse"]
)
midpoint_gain_on_cp8 = float(
    overall.loc["cp8_before", "rmse"] - overall.loc["cp8_after_midpoint", "rmse"]
)
fold_cp_gains_after = (
    fold_table["parent_after_midpoint"] - fold_table["cp8_after_midpoint"]
)

gates = {
    "cp8_gain_after_tail_ge_0p10": bool(cp_gain_after >= 0.10),
    "cp8_gain_positive_5of5_after_tail": bool((fold_cp_gains_after > 0.0).all()),
    "cp8_gain_loss_after_tail_le_0p10": bool(cp_gain_before - cp_gain_after <= 0.10),
    "midpoint_not_required_to_help": True,
}

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_matched_replay_of_fixed_pf_midpoint_tail_only",
    "inputs": {
        "v9_oof": str(V9_PATH),
        "v10_neighbor_oof": str(V10_PATH) if neighbor_available else None,
        "pf_statistics": str(stats_path),
    },
    "frozen_midpoint_recipe": {
        "strength": HEDGE_STRENGTH,
        "minimum_minor_mass": MIN_MINOR_MASS,
        "separation_low_ft": SEPARATION_LOW_FT,
        "separation_high_ft": SEPARATION_HIGH_FT,
        "move_cap_ft": MOVE_CAP_FT,
        "optimizer_or_search_count": 0,
    },
    "postprocess_inventory": {
        "selected_submission_profile": "vp_balanced_modelpkg_005",
        "visible_prefix_calibration": "active in notebook; not replayed by this midpoint-only audit",
        "saved_model_correction": "active in notebook; not replayed without complete model package",
        "pf_seed_branch_midpoint": "active and replayed exactly",
        "hidden_id_correction": "excluded: hidden-specific, no honest OOF analogue",
        "guarded_samewell_override": "excluded: hidden/test-specific, no honest OOF analogue",
    },
    "coverage": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "statistics_wells": int(stats["well"].nunique()),
        "qualified_wells": int(stats["qualified"].sum()),
        "moved_oof_wells": int(frame.loc[frame["midpoint_shift_ft"].ne(0), "well"].nunique()),
        "mean_abs_move_ft": float(frame["midpoint_shift_ft"].abs().mean()),
        "maximum_abs_move_ft": float(frame["midpoint_shift_ft"].abs().max()),
    },
    "overall": overall.reset_index().to_dict(orient="records"),
    "cp8_gain_before_tail": cp_gain_before,
    "cp8_gain_after_tail": cp_gain_after,
    "cp8_gain_loss_after_tail": float(cp_gain_before - cp_gain_after),
    "midpoint_gain_on_cp8": midpoint_gain_on_cp8,
    "fold_cp8_gain_after_tail": {str(int(k)): float(v) for k, v in fold_cp_gains_after.items()},
    "controls": controls_frame.to_dict(orient="records"),
    "worst20_percent_wells": worst.to_dict(orient="records"),
    "gates": gates,
    "midpoint_scope_pass": bool(all(gates.values())),
    "full_tail_replay_complete": False,
    "strict_pass": False,
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "next": (
        "MIDPOINT PASS -> next reconstruct balanced visible-prefix OOF and model-package guard before full-stack promotion"
        if all(gates.values())
        else "MIDPOINT FAIL -> keep CP8 decision on its pre-tail evidence; full-stack replay remains incomplete"
    ),
}

row_output_columns = [
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "frontier_6p39_core", "cp8_nocp12", "midpoint_shift_ft",
]
if neighbor_available:
    row_output_columns.append("cp8_neighbor")
frame[row_output_columns].to_parquet(
    OUT_DIR / "postprocess_oof.parquet", index=False, compression="zstd"
)
stats.to_csv(OUT_DIR / "frozen_midpoint_well_moves.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)

print("\nMATCHED POSTPROCESS SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nMATCHED CONTROLS")
print(controls_frame.to_string(index=False), flush=True)
print("\nWORST 20% WELLS")
print(worst.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
