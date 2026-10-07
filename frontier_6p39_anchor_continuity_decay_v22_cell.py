"""Nested zero-forward anchor-continuity correction for the full 6.39 tail."""

from pathlib import Path
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
SOURCE = (
    OOF_ROOT / "frontier_6p39_final_surface_reprojection_v20"
    / "reprojection_oof.parquet"
)
EXPERIMENT = "frontier_6p39_anchor_continuity_decay_v22"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
WINDOWS_FT = (32.0, 64.0, 128.0)
TAUS_FT = (100.0, 250.0, 500.0, 1000.0)
DOSES = (0.25, 0.50, 0.75, 1.00)
CAPS_FT = (1.0, 2.0, 4.0)
SEED = 20260725
started = time.time()


def metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    well_rmse = np.sqrt(
        pd.Series(error * error).groupby(frame["well"], sort=False).mean().to_numpy()
    )
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
        "near_0_500": distance <= 500,
        "mid_500_1000": (distance > 500) & (distance <= 1000),
        "mid_1000_2000": (distance > 1000) & (distance <= 2000),
        "far_gt2000": distance > 2000,
    }.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
    return result


if not SOURCE.exists():
    raise FileNotFoundError(f"Missing {SOURCE}; run v20 first")
frame = pd.read_parquet(SOURCE).reset_index(drop=True)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", "parent"}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"v20 output missing {missing}")
frame["well"] = frame["well"].astype(str)
parent = frame["parent"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)

z = np.full(len(frame), np.nan, np.float64)
anchor_surface = np.full(len(frame), np.nan, np.float64)
groups = frame.groupby("well", sort=False).indices
for number, (well, indices) in enumerate(groups.items(), 1):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(path, usecols=["Z", "TVT_input"])
    rows = frame.loc[indices, "row_idx"].to_numpy(int)
    known = horizontal["TVT_input"].notna().to_numpy(bool)
    if not known.any() or rows.min() < 0 or rows.max() >= len(horizontal):
        raise RuntimeError(f"{well}: invalid anchor geometry")
    anchor = int(np.flatnonzero(known)[-1])
    zv = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    z[indices] = zv[rows]
    anchor_surface[indices] = tvt[anchor] + zv[anchor]
    if number % 100 == 0 or number == len(groups):
        print(f"anchor geometry {number}/{len(groups)}", flush=True)
if not np.isfinite(z).all() or not np.isfinite(anchor_surface).all():
    raise RuntimeError("incomplete anchor geometry")

# Estimate each well's predicted surface intercept at distance zero.  The fit is
# label-free and uses only the first suffix window plus the visible anchor.
offsets = {window: np.zeros(len(frame), np.float64) for window in WINDOWS_FT}
well_offsets = []
fit_failures = []
for well, indices in groups.items():
    d = distance[indices]
    surface = parent[indices] + z[indices]
    for window in WINDOWS_FT:
        selected = np.flatnonzero(d <= window)
        try:
            if len(selected) < 4:
                selected = np.arange(min(len(indices), 16))
            x = d[selected]
            y = surface[selected]
            design = np.column_stack([np.ones(len(x)), x])
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            offset = float(coef[0] - anchor_surface[indices[0]])
            offsets[window][indices] = offset
            well_offsets.append({"well": well, "window_ft": window, "offset_ft": offset})
        except Exception as exc:
            fit_failures.append({"well": well, "window_ft": window, "reason": repr(exc)})
pd.DataFrame(well_offsets).to_csv(OUT_DIR / "anchor_offsets.csv", index=False)


def candidate(config):
    offset = np.clip(
        offsets[float(config["window_ft"])],
        -float(config["cap_ft"]),
        float(config["cap_ft"]),
    )
    decay = np.exp(-np.maximum(distance, 0.0) / float(config["tau_ft"]))
    return parent - float(config["dose"]) * offset * decay


configs = [
    {"window_ft": window, "tau_ft": tau, "dose": dose, "cap_ft": cap}
    for window, tau, dose, cap in itertools.product(
        WINDOWS_FT, TAUS_FT, DOSES, CAPS_FT
    )
]
selection_mask = frame["fold"].eq(SELECTION_FOLD).to_numpy()
selection = frame.loc[selection_mask].reset_index(drop=True)
scan = pd.DataFrame([
    {**config, **metric(selection, candidate(config)[selection_mask], "candidate")}
    for config in configs
]).sort_values(["rmse", "p95_well_rmse"]).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_scan.csv", index=False)
selected = {
    key: float(scan.iloc[0][key])
    for key in ("window_ft", "tau_ft", "dose", "cap_ft")
}
prediction = candidate(selected)
move = prediction - parent

summary_rows = []
for fold in sorted(frame["fold"].unique()):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    summary_rows.append({"fold": int(fold), **metric(subset, parent[mask], "parent")})
    summary_rows.append({"fold": int(fold), **metric(subset, prediction[mask], "anchor_decay")})
summary_rows.append({"fold": 0, **metric(frame, parent, "parent")})
summary_rows.append({"fold": 0, **metric(frame, prediction, "anchor_decay")})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

# Matched control shuffles complete-well anchor offsets within each fold, while
# preserving every held-out well's distance-decay shape.
rng = np.random.default_rng(SEED)
shuffled_move = np.zeros(len(frame), np.float64)
well_table = frame[["well", "fold"]].drop_duplicates("well").reset_index(drop=True)
selected_offset = {
    well: float(offsets[selected["window_ft"]][indices[0]])
    for well, indices in groups.items()
}
for fold, wells in well_table.groupby("fold", sort=False)["well"]:
    wells = wells.astype(str).tolist()
    shuffled_values = rng.permutation([selected_offset[well] for well in wells])
    for well, shuffled_offset in zip(wells, shuffled_values):
        indices = groups[well]
        clipped = float(np.clip(shuffled_offset, -selected["cap_ft"], selected["cap_ft"]))
        shuffled_move[indices] = (
            -selected["dose"] * clipped
            * np.exp(-np.maximum(distance[indices], 0.0) / selected["tau_ft"])
        )

control_rows = []
for fold in (CONFIRMATION_FOLD, FINAL_FOLD):
    mask = frame["fold"].eq(fold).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    for name, values in (
        ("parent", parent),
        ("anchor_decay", prediction),
        ("shuffled_offset", parent + shuffled_move),
        ("opposite_move", parent - move),
    ):
        control_rows.append({"fold": fold, **metric(subset, values[mask], name)})
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)

wide = summary.pivot(index="fold", columns="endpoint", values="rmse")
gain = wide["parent"] - wide["anchor_decay"]
far = summary.pivot(index="fold", columns="endpoint", values="far_gt2000_rmse")
control_wide = controls.pivot(index="fold", columns="endpoint", values="rmse")
shuffle_gain = control_wide["parent"] - control_wide["shuffled_offset"]
gates = {
    "fold2_gain_ge_0p04": bool(gain.loc[CONFIRMATION_FOLD] >= 0.04),
    "fold1_gain_ge_0p04": bool(gain.loc[FINAL_FOLD] >= 0.04),
    "fold2_far_degradation_le_0p005": bool(
        far.loc[CONFIRMATION_FOLD, "anchor_decay"]
        <= far.loc[CONFIRMATION_FOLD, "parent"] + 0.005
    ),
    "fold1_far_degradation_le_0p005": bool(
        far.loc[FINAL_FOLD, "anchor_decay"]
        <= far.loc[FINAL_FOLD, "parent"] + 0.005
    ),
    "shuffle_retains_le_30pct_fold2_gain": bool(
        shuffle_gain.loc[CONFIRMATION_FOLD]
        <= max(0.0, 0.30 * gain.loc[CONFIRMATION_FOLD])
    ),
}

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["parent"] = parent.astype(np.float32)
output["anchor_decay"] = prediction.astype(np.float32)
output["move"] = move.astype(np.float32)
output.to_parquet(OUT_DIR / "anchor_decay_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_selection_Fold2_confirmation_Fold1_final_anchor_continuity",
    "selected_config": selected,
    "selection_search_count": len(configs),
    "test_well_optimizer_or_search_count": 0,
    "fit_failures": fit_failures,
    "fold_gain": {str(int(k)): float(v) for k, v in gain.items()},
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "deployment_cost": "one 2-parameter least-squares fit per well; zero model/PF forwards",
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> emit final Kaggle anchor-decay cell"
        if all(gates.values())
        else "FAIL -> close anchor/surface postprocessing and keep 6.39 unchanged"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("ANCHOR-DECAY SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nMATCHED CONTROLS")
print(controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
