"""V119: dominant trajectory-mode center on complete Fold3 development data.

Top-K uniform averaging can mix parallel alias modes. This experiment embeds
each candidate trajectory at 48 tail locations, finds the candidate with the
largest radius-neighborhood, and averages only that dominant local mode.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_consensus_dominant_mode_v119"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
FOLD = 3
POOL_SIZES = (33, 65, 129)
RADII_FT = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
EMBED_POINTS = 48
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"


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


def _bootstrap_v110():
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("raw_lattice_v119_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _pairwise_rms(values):
    values = np.asarray(values, np.float64)
    square = np.mean(values * values, axis=1)
    distance2 = square[:, None] + square[None, :] - 2.0 * (values @ values.T) / values.shape[1]
    return np.sqrt(np.maximum(distance2, 0.0))


def _dominant_center(trajectories, tail_mask, radius):
    tail_indices = np.flatnonzero(tail_mask)
    if len(tail_indices) < 8:
        tail_indices = np.arange(trajectories.shape[1])
    sample = tail_indices[np.linspace(0, len(tail_indices) - 1, min(EMBED_POINTS, len(tail_indices))).round().astype(int)]
    distance = _pairwise_rms(trajectories[:, sample])
    neighbors = distance <= float(radius)
    counts = neighbors.sum(axis=1)
    mean_distance = np.divide(
        (distance * neighbors).sum(axis=1),
        counts,
        out=np.full(len(counts), np.inf),
        where=counts > 0,
    )
    # Stable rank order is the final tie breaker.
    center_index = int(np.lexsort((np.arange(len(counts)), mean_distance, -counts))[0])
    members = neighbors[center_index]
    return np.mean(trajectories[members], axis=0), int(members.sum())


if not V110_SOURCE.is_file():
    raise FileNotFoundError(V110_SOURCE)
R = _bootstrap_v110()
for required in (R.V81_SOURCE, R.PANEL_DIR, R.MODEL_PATH, R.V70_PATH, R.V76_WEIGHTS):
    if not required.exists():
        raise FileNotFoundError(required)
N = R._bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
model = lgb.Booster(model_file=str(R.MODEL_PATH))

frame = pd.read_parquet(
    R.V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
    filters=[("fold", "=", FOLD)],
).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
weights = pd.read_parquet(R.V76_WEIGHTS)
weights = weights[weights["fold"].eq(FOLD)][["well", "ensemble_weight"]].copy()
weights["well"] = weights["well"].astype(str)
frame["dose"] = frame["well"].map(dict(zip(weights["well"], weights["ensemble_weight"])))
if frame.empty or frame["dose"].isna().any() or frame["id"].duplicated().any():
    raise RuntimeError("Incomplete exact Fold3 V76 source")
frame["exact_v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
    frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
)

wells = sorted(well for well in frame["well"].unique() if (R.PANEL_DIR / f"{well}.npz").is_file())
if len(wells) < 153:
    raise RuntimeError(f"Only {len(wells)} Fold3 panels")
recipes = [(pool, radius) for pool in POOL_SIZES for radius in RADII_FT]
prediction_parts = {recipe: [] for recipe in recipes}
member_counts = {recipe: [] for recipe in recipes}
top33_parts = []
base_parts = []

for number, well in enumerate(wells, 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    with np.load(R.PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
        paths = data["paths"].astype(np.int16, copy=False)
    record = V["_load_record"](well)
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: Fold3 alignment")
    order = R._deduplicated_order(paths, model.predict(features))
    if len(order) < max(POOL_SIZES):
        raise RuntimeError(f"{well}: only {len(order)} unique candidates")
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    trajectories = np.vstack([
        B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
        for index in order[:max(POOL_SIZES)]
    ]).astype(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    dose = float(local["dose"].iloc[0])
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    tail_mask = distance > FREEZE_FT
    base_parts.append(local)
    top33_center = np.mean(trajectories[:33], axis=0)
    top33_parts.append(exact + taper * np.clip(dose * (top33_center - coarse), -CAP_FT, CAP_FT))
    for recipe in recipes:
        pool, radius = recipe
        center, members = _dominant_center(trajectories[:pool], tail_mask, radius)
        move = taper * np.clip(dose * (center - coarse), -CAP_FT, CAP_FT)
        prediction_parts[recipe].append(exact + move)
        member_counts[recipe].append(members)
    if number % 10 == 0 or number == len(wells):
        print(f"V119 dominant mode {number}/{len(wells)} elapsed_min={(time.time()-STARTED)/60:.1f}", flush=True)

base_frame = pd.concat(base_parts, ignore_index=True)
base = R._metric(base_frame, base_frame["exact_v76"].to_numpy(np.float64), "exact_v76")
top33_prediction = np.concatenate(top33_parts)
top33 = R._metric(base_frame, top33_prediction, "consensus_top33")
scan_rows = []
predictions = {}
for recipe in recipes:
    pool, radius = recipe
    prediction = np.concatenate(prediction_parts[recipe])
    metric = R._metric(base_frame, prediction, "dominant_mode")
    metric.update({
        "pool": pool,
        "radius_ft": radius,
        "median_members": float(np.median(member_counts[recipe])),
        "mean_members": float(np.mean(member_counts[recipe])),
        "gain_vs_v76": float(base["rmse"] - metric["rmse"]),
        "gain_vs_top33": float(top33["rmse"] - metric["rmse"]),
        "far_gain_vs_top33": float(top33["far_gt2000_rmse"] - metric["far_gt2000_rmse"]),
        "p95_gain_vs_top33": float(top33["p95_well_rmse"] - metric["p95_well_rmse"]),
        "near_degradation": float(metric["near_0_500_rmse"] - base["near_0_500_rmse"]),
    })
    scan_rows.append(metric)
    predictions[recipe] = prediction

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "dominant_mode_scan.csv", index=False)
eligible = scan[
    (scan["gain_vs_top33"] >= 0.03)
    & (scan["far_gain_vs_top33"] >= 0.0)
    & (scan["p95_gain_vs_top33"] >= -0.10)
    & (scan["near_degradation"] <= 1e-10)
    & (scan["median_members"] >= 2.0)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (int(selected_row["pool"]), float(selected_row["radius_ft"]))

output = base_frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, "exact_v76"]].copy()
output["consensus_top33"] = top33_prediction.astype(np.float32)
output["dominant_mode"] = predictions[selected].astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_dominant_mode_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_dominant_trajectory_mode_development",
    "search_count": len(recipes),
    "fixed": {"embed_points": EMBED_POINTS, "freeze_ft": FREEZE_FT, "ramp_ft": RAMP_FT, "cap_ft": CAP_FT},
    "selected_recipe": {"pool": selected[0], "radius_ft": selected[1]},
    "selected": selected_row.to_dict(),
    "top33_reference": top33,
    "checks": {
        "eligible_recipe_exists": bool(len(eligible)),
        "gain_vs_top33_ge_0p03": float(selected_row["gain_vs_top33"]) >= 0.03,
        "far_gain_vs_top33_nonnegative": float(selected_row["far_gain_vs_top33"]) >= 0.0,
        "p95_not_worse_by_gt_0p10": float(selected_row["p95_gain_vs_top33"]) >= -0.10,
        "nontrivial_cluster": float(selected_row["median_members"]) >= 2.0,
    },
    "strict_pass": bool(len(eligible)),
    "fold2_touched": False,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> freeze dominant-mode recipe and replay Fold2/Fold1; FAIL -> retain top33 and close consensus geometry",
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("\nV119 DOMINANT-MODE TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV119 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
