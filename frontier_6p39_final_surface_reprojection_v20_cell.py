"""Nested audit of a zero-forward final-stack surface reprojection.

The published notebook projects SP45 before blending the learned endpoint.  This
experiment instead projects the already-complete deployable OOF trajectory.  It
adds no PF/model inference at deployment: one robust polynomial solve per well
and a vector update are the entire postprocess.
"""

from __future__ import annotations

from pathlib import Path
import itertools
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
SOURCE = (
    OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18"
    / "full_oof_cp_endpoints.parquet"
)
EXPERIMENT = "frontier_6p39_final_surface_reprojection_v20"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
DEGREES = (2, 3, 4, 5)
DOSES = (0.10, 0.25, 0.40, 0.60)
CAPS_FT = (2.0, 4.0, 8.0)
WARMUP_TAU_FT = (250.0, 500.0, 1000.0)
HUBER_K = 2.5
ROBUST_ITERATIONS = 6
RIDGE = 1e-6
SEED = 20260725
started = time.time()


def _write_json(path: Path, payload: dict) -> None:
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"invalid prediction: {endpoint}")
    well_sse = pd.Series(error * error).groupby(frame["well"], sort=False).sum()
    well_n = frame.groupby("well", sort=False).size().reindex(well_sse.index)
    well_rmse = np.sqrt(well_sse.to_numpy() / well_n.to_numpy())
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    strata = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in strata.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _robust_projection(x: np.ndarray, y: np.ndarray, degree: int):
    design = np.column_stack([x ** power for power in range(1, degree + 1)])
    weights = np.ones(len(x), np.float64)
    beta = np.zeros(degree, np.float64)
    condition = np.nan
    for _ in range(ROBUST_ITERATIONS):
        weighted = design * np.sqrt(weights)[:, None]
        target = y * np.sqrt(weights)
        gram = weighted.T @ weighted + RIDGE * np.eye(degree)
        condition = float(np.linalg.cond(gram))
        beta = np.linalg.solve(gram, weighted.T @ target)
        residual = y - design @ beta
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        ratio = np.abs(residual) / (HUBER_K * scale)
        weights = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1e-12))
    return design @ beta, condition


if not SOURCE.exists():
    raise FileNotFoundError(
        f"Missing {SOURCE}. Run the completed v18 full-tail OOF audit first."
    )
if not (DATA_ROOT / "train").exists():
    raise FileNotFoundError(DATA_ROOT / "train")

header = pq.ParquetFile(SOURCE).schema.names
parent_column = (
    "full_parent_matched_vp_replay"
    if "full_parent_matched_vp_replay" in header
    else "core"
)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", parent_column}
missing = sorted(required.difference(header))
if missing:
    raise RuntimeError(f"source missing columns: {missing}")
frame = pd.read_parquet(SOURCE, columns=sorted(required)).reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
frame["row_idx"] = frame["row_idx"].astype(int)
frame["fold"] = frame["fold"].astype(int)
parent = frame[parent_column].to_numpy(np.float64)
if frame["id"].duplicated().any() or not np.isfinite(parent).all():
    raise RuntimeError("invalid source ids or parent prediction")

z = np.full(len(frame), np.nan, np.float64)
anchor_surface = np.full(len(frame), np.nan, np.float64)
anchor_error = []
groups = frame.groupby("well", sort=False).indices
for number, (well, indices) in enumerate(groups.items(), 1):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
    rows = frame.loc[indices, "row_idx"].to_numpy(int)
    if rows.min() < 0 or rows.max() >= len(horizontal):
        raise RuntimeError(f"{well}: row index outside raw horizontal file")
    known = horizontal["TVT_input"].notna().to_numpy(bool)
    if not known.any():
        raise RuntimeError(f"{well}: no visible prefix")
    anchor = int(np.flatnonzero(known)[-1])
    z_values = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    z[indices] = z_values[rows]
    anchor_surface[indices] = tvt_input[anchor] + z_values[anchor]
    first_local = indices[np.argmin(frame.loc[indices, "md_since"].to_numpy(float))]
    anchor_error.append(float(parent[first_local] + z[first_local] - anchor_surface[first_local]))
    if number % 100 == 0 or number == len(groups):
        print(f"load raw geometry {number}/{len(groups)}", flush=True)
if not np.isfinite(z).all() or not np.isfinite(anchor_surface).all():
    raise RuntimeError("raw geometry enrichment is incomplete")

# Fit each polynomial degree once per well. Configuration scans only combine
# frozen corrections and therefore cannot alter or refit held-out wells.
raw_correction = {degree: np.zeros(len(frame), np.float64) for degree in DEGREES}
condition_rows = []
solve_failures = []
for number, (well, indices) in enumerate(groups.items(), 1):
    distance = frame.loc[indices, "md_since"].to_numpy(np.float64)
    scale = max(float(np.max(distance)), 1.0)
    x = np.clip(distance / scale, 0.0, 1.0)
    surface_delta = parent[indices] + z[indices] - anchor_surface[indices]
    for degree in DEGREES:
        try:
            fitted, condition = _robust_projection(x, surface_delta, degree)
            raw_correction[degree][indices] = fitted - surface_delta
            condition_rows.append({
                "well": well, "degree": degree, "condition_number": condition,
                "rows": int(len(indices)), "status": "ok",
            })
        except Exception as exc:
            solve_failures.append({"well": well, "degree": degree, "reason": repr(exc)})
            condition_rows.append({
                "well": well, "degree": degree, "condition_number": np.nan,
                "rows": int(len(indices)), "status": "failed",
            })
    if number % 100 == 0 or number == len(groups):
        print(f"fit final surfaces {number}/{len(groups)} failures={len(solve_failures)}", flush=True)

configs = [
    {"degree": degree, "dose": dose, "cap_ft": cap, "warmup_tau_ft": tau}
    for degree, dose, cap, tau in itertools.product(DEGREES, DOSES, CAPS_FT, WARMUP_TAU_FT)
]


def _candidate(config: dict) -> np.ndarray:
    correction = np.clip(
        raw_correction[int(config["degree"])],
        -float(config["cap_ft"]),
        float(config["cap_ft"]),
    )
    warmup = 1.0 - np.exp(
        -np.maximum(frame["md_since"].to_numpy(np.float64), 0.0)
        / float(config["warmup_tau_ft"])
    )
    return parent + float(config["dose"]) * warmup * correction


selection_mask = frame["fold"].eq(SELECTION_FOLD).to_numpy()
selection = frame.loc[selection_mask].reset_index(drop=True)
scan_rows = []
for config in configs:
    prediction = _candidate(config)[selection_mask]
    scan_rows.append({**config, **_metric(selection, prediction, "candidate")})
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"]).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_scan.csv", index=False)
selected = {key: scan.iloc[0][key].item() for key in ("degree", "dose", "cap_ft", "warmup_tau_ft")}
selected["degree"] = int(selected["degree"])
prediction = _candidate(selected)

summary_rows = []
for fold in sorted(frame["fold"].unique()):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    summary_rows.append({"fold": int(fold), **_metric(subset, parent[mask], "parent")})
    summary_rows.append({"fold": int(fold), **_metric(subset, prediction[mask], "reprojected")})
summary_rows.append({"fold": 0, **_metric(frame, parent, "parent")})
summary_rows.append({"fold": 0, **_metric(frame, prediction, "reprojected")})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "nested_summary.csv", index=False)

# Matched control: preserve fold and distance-band correction distributions but
# break row-wise correspondence with each well's path.
rng = np.random.default_rng(SEED)
move = prediction - parent
shuffled_move = move.copy()
bands = pd.cut(
    frame["md_since"], [-np.inf, 500, 1000, 2000, np.inf],
    labels=False, include_lowest=True,
).astype(int)
for _, indices in pd.DataFrame({"fold": frame["fold"], "band": bands}).groupby(
    ["fold", "band"], sort=False
).groups.items():
    idx = np.asarray(list(indices), np.int64)
    shuffled_move[idx] = rng.permutation(shuffled_move[idx])
shuffled = parent + shuffled_move

control_rows = []
for fold in (CONFIRMATION_FOLD, FINAL_FOLD):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    for name, values in (
        ("parent", parent),
        ("reprojected", prediction),
        ("shuffled_move", shuffled),
        ("opposite_move", parent - move),
    ):
        control_rows.append({"fold": fold, **_metric(subset, values[mask], name)})
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)

overall_parent = _metric(frame, parent, "parent")
well_parent = frame.assign(error2=(parent - frame["tvt_true"].to_numpy()) ** 2).groupby(
    "well", sort=False
)["error2"].mean().sort_values(ascending=False)
worst_wells = set(well_parent.head(max(1, math.ceil(0.20 * len(well_parent)))).index)
worst_mask = frame["well"].isin(worst_wells).to_numpy()
worst_frame = frame.loc[worst_mask].reset_index(drop=True)
worst = [
    _metric(worst_frame, values[worst_mask], name)
    for name, values in (("parent", parent), ("reprojected", prediction))
]
pd.DataFrame(worst).to_csv(OUT_DIR / "worst20_wells.csv", index=False)

wide = summary.pivot(index="fold", columns="endpoint", values="rmse")
fold_gain = wide["parent"] - wide["reprojected"]
control_wide = controls.pivot(index="fold", columns="endpoint", values="rmse")
control_gain = control_wide["parent"] - control_wide["shuffled_move"]
near = summary.pivot(index="fold", columns="endpoint", values="near_0_500_rmse")
p95 = summary.pivot(index="fold", columns="endpoint", values="p95_well_rmse")
gates = {
    "fold2_gain_ge_0p05": bool(fold_gain.loc[CONFIRMATION_FOLD] >= 0.05),
    "fold1_gain_ge_0p05": bool(fold_gain.loc[FINAL_FOLD] >= 0.05),
    "fold2_near_degradation_le_0p03": bool(
        near.loc[CONFIRMATION_FOLD, "reprojected"]
        <= near.loc[CONFIRMATION_FOLD, "parent"] + 0.03
    ),
    "fold1_near_degradation_le_0p03": bool(
        near.loc[FINAL_FOLD, "reprojected"]
        <= near.loc[FINAL_FOLD, "parent"] + 0.03
    ),
    "fold2_p95_degradation_le_0p10": bool(
        p95.loc[CONFIRMATION_FOLD, "reprojected"]
        <= p95.loc[CONFIRMATION_FOLD, "parent"] + 0.10
    ),
    "fold1_p95_degradation_le_0p10": bool(
        p95.loc[FINAL_FOLD, "reprojected"]
        <= p95.loc[FINAL_FOLD, "parent"] + 0.10
    ),
    "shuffle_retains_le_30pct_fold2_gain": bool(
        control_gain.loc[CONFIRMATION_FOLD]
        <= max(0.0, 0.30 * fold_gain.loc[CONFIRMATION_FOLD])
    ),
}

condition_frame = pd.DataFrame(condition_rows)
condition_frame.to_csv(OUT_DIR / "solve_diagnostics.csv", index=False)
row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["parent"] = parent.astype(np.float32)
row_output["reprojected"] = prediction.astype(np.float32)
row_output["move"] = move.astype(np.float32)
row_output.to_parquet(OUT_DIR / "reprojection_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_selection_Fold2_confirmation_Fold1_final_zero_forward_postprocess",
    "source": str(SOURCE),
    "parent_column": parent_column,
    "selected_config": selected,
    "selection_search_count": int(len(configs)),
    "test_well_optimizer_or_search_count": 0,
    "anchor_error": {
        "rows": int(len(anchor_error)),
        "rmse": float(np.sqrt(np.mean(np.asarray(anchor_error) ** 2))),
        "mae": float(np.mean(np.abs(anchor_error))),
    },
    "condition_number": {
        "median": float(condition_frame["condition_number"].median()),
        "p95": float(condition_frame["condition_number"].quantile(0.95)),
        "max": float(condition_frame["condition_number"].max()),
    },
    "solve_failures": solve_failures,
    "fold_gain": {str(int(k)): float(v) for k, v in fold_gain.items()},
    "summary": summary.to_dict(orient="records"),
    "matched_controls": controls.to_dict(orient="records"),
    "worst_twenty_percent_wells": worst,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "deployment_cost": "one robust polynomial solve per well; zero PF/model forward calls",
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> insert after the generalizable model-package/VP tail, before hidden-only branch/Q edits and final audit"
        if all(gates.values())
        else "FAIL -> keep the published 6.39 writer unchanged; close final-stack reprojection"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("FINAL-STACK REPROJECTION FOLD SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nMATCHED CONTROLS")
print(controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
