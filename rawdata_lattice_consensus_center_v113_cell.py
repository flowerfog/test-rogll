"""V113: develop a rank-insensitive candidate consensus endpoint on Fold3.

V112 confirmed that both top3 and next3 barycenters improve exact V76.  The
useful mechanism is therefore candidate-bank centralization, not precise top1
ranking.  This cell scans robust centers on complete Fold3 development data
under the frozen V111 tail guard.  A selected recipe must still pass untouched
Fold1 before deployment.
"""

from __future__ import annotations

import hashlib
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
EXPERIMENT = "rawdata_lattice_consensus_center_v113"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
FOLD = 3
TOPK_VALUES = (3, 6, 9, 17, 33)
AGGREGATORS = ("median", "trimmed_mean", "uniform_mean")
ALPHAS = (0.25, 0.50, 0.75, 1.00)
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


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v110():
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("raw_lattice_v110_consensus_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _center(values, method):
    values = np.asarray(values, np.float64)
    if method == "median":
        return np.median(values, axis=0)
    if method == "uniform_mean":
        return np.mean(values, axis=0)
    if method == "trimmed_mean":
        ordered = np.sort(values, axis=0)
        trim = max(1, int(np.floor(len(values) * 0.20))) if len(values) >= 5 else 0
        return np.mean(ordered[trim: len(values) - trim], axis=0) if trim else np.mean(ordered, axis=0)
    raise KeyError(method)


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
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))
frame["dose"] = frame["well"].map(weight_map)
if frame.empty or frame["dose"].isna().any() or frame["id"].duplicated().any():
    raise RuntimeError("Incomplete exact Fold3 V76 source")
frame["exact_v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
    frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
)

wells = sorted(well for well in frame["well"].unique() if (R.PANEL_DIR / f"{well}.npz").is_file())
if len(wells) < 153:
    raise RuntimeError(f"Only {len(wells)} Fold3 panels")

recipes = [(k, method, alpha) for k in TOPK_VALUES for method in AGGREGATORS for alpha in ALPHAS]
aligned_parts = {recipe: [] for recipe in recipes}
shuffled_parts = {recipe: [] for recipe in recipes}
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
    scores = model.predict(features)
    order = R._deduplicated_order(paths, scores)
    maximum = 2 * max(TOPK_VALUES)
    if len(order) < maximum:
        raise RuntimeError(f"{well}: only {len(order)} unique candidates")
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    trajectories = np.vstack([
        B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
        for index in order[:maximum]
    ]).astype(np.float64)
    seed = int.from_bytes(hashlib.sha256(f"v113:{well}".encode()).digest()[:8], "little")
    shuffled = trajectories[np.random.default_rng(seed).permutation(len(trajectories))]
    exact = local["exact_v76"].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    dose = float(local["dose"].iloc[0])
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    base_parts.append(local)
    for recipe in recipes:
        k, method, alpha = recipe
        aligned_center = _center(trajectories[:k], method)
        shuffled_center = _center(shuffled[:k], method)
        aligned_move = np.clip(alpha * dose * (aligned_center - coarse), -CAP_FT, CAP_FT)
        shuffled_move = np.clip(alpha * dose * (shuffled_center - coarse), -CAP_FT, CAP_FT)
        aligned_parts[recipe].append(exact + taper * aligned_move)
        shuffled_parts[recipe].append(exact + taper * shuffled_move)
    if number % 10 == 0 or number == len(wells):
        print(
            f"V113 consensus {number}/{len(wells)} elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

base_frame = pd.concat(base_parts, ignore_index=True)
base_prediction = base_frame["exact_v76"].to_numpy(np.float64)
base = R._metric(base_frame, base_prediction, "exact_v76")
scan_rows = []
for recipe in recipes:
    k, method, alpha = recipe
    prediction = np.concatenate(aligned_parts[recipe])
    control = np.concatenate(shuffled_parts[recipe])
    result = R._metric(base_frame, prediction, "consensus_center")
    control_metric = R._metric(base_frame, control, "score_shuffled_control")
    result.update({
        "topk": k,
        "aggregator": method,
        "alpha": alpha,
        "gain": float(base["rmse"] - result["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
        "near_degradation": float(result["near_0_500_rmse"] - base["near_0_500_rmse"]),
        "p95_gain": float(base["p95_well_rmse"] - result["p95_well_rmse"]),
        "aligned_increment_vs_shuffle": float(control_metric["rmse"] - result["rmse"]),
        "shuffled_rmse": float(control_metric["rmse"]),
    })
    scan_rows.append(result)

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "consensus_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.05)
    & (scan["far_gain"] >= 0.05)
    & (scan["near_degradation"] <= 1e-10)
    & (scan["p95_gain"] >= 0.0)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (int(selected_row["topk"]), str(selected_row["aggregator"]), float(selected_row["alpha"]))
selected_prediction = np.concatenate(aligned_parts[selected])
selected_control = np.concatenate(shuffled_parts[selected])
candidate = R._metric(base_frame, selected_prediction, "consensus_center")
control = R._metric(base_frame, selected_control, "score_shuffled_control")
summary = pd.DataFrame([base, candidate, control])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

values = {
    "gain": float(base["rmse"] - candidate["rmse"]),
    "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
    "near_degradation": float(candidate["near_0_500_rmse"] - base["near_0_500_rmse"]),
    "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
    "aligned_increment_vs_shuffle": float(control["rmse"] - candidate["rmse"]),
}
checks = {
    "eligible_recipe_exists": bool(len(eligible)),
    "gain_ge_0p05": values["gain"] >= 0.05,
    "far_gain_ge_0p05": values["far_gain"] >= 0.05,
    "near_frozen_exact": abs(values["near_degradation"]) <= 1e-10,
    "p95_nonworse": values["p95_gain"] >= 0.0,
}
strict_pass = bool(all(checks.values()))

output = base_frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, "exact_v76"]].copy()
output["consensus_center"] = selected_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_consensus_center_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_Fold3_consensus_development_before_untouched_Fold1",
    "candidate_generation_uses_truth": False,
    "truth_usage": "recipe selection and metrics only",
    "search_count": len(recipes),
    "fixed_guard": {"freeze_ft": FREEZE_FT, "ramp_ft": RAMP_FT, "cap_ft": CAP_FT},
    "selected_recipe": {
        "topk": selected[0],
        "aggregator": selected[1],
        "alpha": selected[2],
        "passed_constraints": bool(len(eligible)),
    },
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "fold2_used_only_for_prior_mechanism_evidence": True,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> reconstruct the frozen consensus once on untouched Fold1"
        if strict_pass else
        "FAIL -> retain the simpler frozen V110 recipe and audit Fold1 only if desired"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV113 CONSENSUS TOP 25")
print(scan.head(25).to_string(index=False), flush=True)
print("\nV113 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV113 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
