"""Nested distance-taper audit for the validated v20 reprojection direction."""

from pathlib import Path
import itertools
import json
import math
import shutil
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE = (
    OOF_ROOT / "frontier_6p39_final_surface_reprojection_v20"
    / "reprojection_oof.parquet"
)
EXPERIMENT = "frontier_6p39_reprojection_distance_taper_v21"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
END_STARTS_FT = (750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
FADE_WIDTHS_FT = (250.0, 500.0, 1000.0, 1500.0)
MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)
SEED = 20260725
started = time.time()


def metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    well_mse = pd.Series(error * error).groupby(frame["well"], sort=False).mean()
    well_rmse = np.sqrt(well_mse.to_numpy())
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, mask in {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
    return result


if not SOURCE.exists():
    raise FileNotFoundError(f"Missing {SOURCE}; run v20 first")
frame = pd.read_parquet(SOURCE).reset_index(drop=True)
required = {"id", "well", "fold", "tvt_true", "md_since", "parent", "move"}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"v20 output missing {missing}")

parent = frame["parent"].to_numpy(np.float64)
source_move = frame["move"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)


def taper(start, width):
    return np.clip((float(start) + float(width) - distance) / float(width), 0.0, 1.0)


configs = [
    {"end_start_ft": start, "fade_width_ft": width, "multiplier": multiplier}
    for start, width, multiplier in itertools.product(
        END_STARTS_FT, FADE_WIDTHS_FT, MULTIPLIERS
    )
]
selection_mask = frame["fold"].eq(SELECTION_FOLD).to_numpy()
selection = frame.loc[selection_mask].reset_index(drop=True)
scan_rows = []
for config in configs:
    move = (
        float(config["multiplier"])
        * source_move
        * taper(config["end_start_ft"], config["fade_width_ft"])
    )
    scan_rows.append({
        **config,
        **metric(selection, (parent + move)[selection_mask], "candidate"),
    })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"]).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_scan.csv", index=False)
selected = {
    key: float(scan.iloc[0][key])
    for key in ("end_start_ft", "fade_width_ft", "multiplier")
}
selected_taper = taper(selected["end_start_ft"], selected["fade_width_ft"])
move = selected["multiplier"] * source_move * selected_taper
prediction = parent + move

rows = []
for fold in sorted(frame["fold"].unique()):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    rows.append({"fold": int(fold), **metric(subset, parent[mask], "parent")})
    rows.append({"fold": int(fold), **metric(subset, prediction[mask], "tapered")})
rows.append({"fold": 0, **metric(frame, parent, "parent")})
rows.append({"fold": 0, **metric(frame, prediction, "tapered")})
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

rng = np.random.default_rng(SEED)
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

controls = []
for fold in (CONFIRMATION_FOLD, FINAL_FOLD):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    for name, values in (
        ("parent", parent),
        ("tapered", prediction),
        ("shuffled_move", parent + shuffled_move),
        ("opposite_move", parent - move),
    ):
        controls.append({"fold": fold, **metric(subset, values[mask], name)})
controls = pd.DataFrame(controls)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)

wide = summary.pivot(index="fold", columns="endpoint", values="rmse")
gains = wide["parent"] - wide["tapered"]
near = summary.pivot(index="fold", columns="endpoint", values="near_0_500_rmse")
far = summary.pivot(index="fold", columns="endpoint", values="far_gt2000_rmse")
control_wide = controls.pivot(index="fold", columns="endpoint", values="rmse")
shuffle_gain = control_wide["parent"] - control_wide["shuffled_move"]
gates = {
    "fold2_gain_ge_0p04": bool(gains.loc[CONFIRMATION_FOLD] >= 0.04),
    "fold1_gain_ge_0p04": bool(gains.loc[FINAL_FOLD] >= 0.04),
    "fold2_near_improves": bool(
        near.loc[CONFIRMATION_FOLD, "tapered"] < near.loc[CONFIRMATION_FOLD, "parent"]
    ),
    "fold1_near_improves": bool(
        near.loc[FINAL_FOLD, "tapered"] < near.loc[FINAL_FOLD, "parent"]
    ),
    "fold2_far_degradation_le_0p01": bool(
        far.loc[CONFIRMATION_FOLD, "tapered"]
        <= far.loc[CONFIRMATION_FOLD, "parent"] + 0.01
    ),
    "fold1_far_degradation_le_0p01": bool(
        far.loc[FINAL_FOLD, "tapered"] <= far.loc[FINAL_FOLD, "parent"] + 0.01
    ),
    "shuffle_retains_le_30pct_fold2_gain": bool(
        shuffle_gain.loc[CONFIRMATION_FOLD]
        <= max(0.0, 0.30 * gains.loc[CONFIRMATION_FOLD])
    ),
}

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["parent"] = parent.astype(np.float32)
output["tapered"] = prediction.astype(np.float32)
output["move"] = move.astype(np.float32)
output.to_parquet(OUT_DIR / "tapered_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_taper_selection_Fold2_confirmation_Fold1_final",
    "source": str(SOURCE),
    "selected_config": selected,
    "selection_search_count": int(len(configs)),
    "test_well_optimizer_or_search_count": 0,
    "fold_gain": {str(int(k)): float(v) for k, v in gains.items()},
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> freeze v20 projection plus this distance taper for a zero-forward Kaggle cell"
        if all(gates.values())
        else "FAIL -> close final-stack surface postprocessing"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("DISTANCE-TAPER SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nMATCHED CONTROLS")
print(controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
