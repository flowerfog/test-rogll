"""V117: test whether V113 top-33 consensus stopped at an artificial K boundary.

This is a Fold3 development-only support-size scan. It keeps the validated
uniform path center and tail guard fixed, changing only the number of unique
ranked candidates consumed by the center. Fold2 and Fold1 are not read here.
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
EXPERIMENT = "rawdata_lattice_consensus_support_scan_v117"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
FOLD = 3
TOPK_VALUES = (33, 49, 65, 97, 129)
ALPHAS = (0.50, 0.75, 1.00)
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
    module = types.ModuleType("raw_lattice_v117_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


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

recipes = [(k, alpha) for k in TOPK_VALUES for alpha in ALPHAS]
aligned_parts = {recipe: [] for recipe in recipes}
shuffled_parts = {recipe: [] for recipe in recipes}
base_parts = []
minimum_unique = 10**9

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
    minimum_unique = min(minimum_unique, len(order))
    maximum = 2 * max(TOPK_VALUES)
    if len(order) < maximum:
        raise RuntimeError(f"{well}: only {len(order)} unique candidates; need {maximum}")
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    trajectories = np.vstack([
        B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
        for index in order[:maximum]
    ]).astype(np.float64)
    seed = int.from_bytes(hashlib.sha256(f"v117:{well}".encode()).digest()[:8], "little")
    shuffled = trajectories[np.random.default_rng(seed).permutation(len(trajectories))]
    exact = local["exact_v76"].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    dose = float(local["dose"].iloc[0])
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    base_parts.append(local)
    for recipe in recipes:
        k, alpha = recipe
        aligned_center = np.mean(trajectories[:k], axis=0)
        shuffled_center = np.mean(shuffled[:k], axis=0)
        aligned_move = np.clip(alpha * dose * (aligned_center - coarse), -CAP_FT, CAP_FT)
        shuffled_move = np.clip(alpha * dose * (shuffled_center - coarse), -CAP_FT, CAP_FT)
        aligned_parts[recipe].append(exact + taper * aligned_move)
        shuffled_parts[recipe].append(exact + taper * shuffled_move)
    if number % 10 == 0 or number == len(wells):
        print(f"V117 support {number}/{len(wells)} elapsed_min={(time.time()-STARTED)/60:.1f}", flush=True)

base_frame = pd.concat(base_parts, ignore_index=True)
base_prediction = base_frame["exact_v76"].to_numpy(np.float64)
base = R._metric(base_frame, base_prediction, "exact_v76")
rows = []
predictions = {}
for recipe in recipes:
    k, alpha = recipe
    prediction = np.concatenate(aligned_parts[recipe])
    control = np.concatenate(shuffled_parts[recipe])
    result = R._metric(base_frame, prediction, "consensus_center")
    control_metric = R._metric(base_frame, control, "score_shuffled_control")
    result.update({
        "topk": k,
        "alpha": alpha,
        "gain": float(base["rmse"] - result["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
        "near_degradation": float(result["near_0_500_rmse"] - base["near_0_500_rmse"]),
        "p95_gain": float(base["p95_well_rmse"] - result["p95_well_rmse"]),
        "aligned_increment_vs_shuffle": float(control_metric["rmse"] - result["rmse"]),
    })
    rows.append(result)
    predictions[recipe] = prediction

scan = pd.DataFrame(rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "support_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.05)
    & (scan["far_gain"] >= 0.05)
    & (scan["near_degradation"] <= 1e-10)
    & (scan["p95_gain"] >= 0.0)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (int(selected_row["topk"]), float(selected_row["alpha"]))
candidate = R._metric(base_frame, predictions[selected], "consensus_center")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_extended_consensus_support_development",
    "base": "exact V76",
    "fixed": {"aggregator": "uniform_mean", "freeze_ft": FREEZE_FT, "ramp_ft": RAMP_FT, "cap_ft": CAP_FT},
    "search": {"topk": TOPK_VALUES, "alpha": ALPHAS, "count": len(recipes)},
    "minimum_unique_candidates": minimum_unique,
    "selected_recipe": {"topk": selected[0], "alpha": selected[1]},
    "selected": selected_row.to_dict(),
    "top33_reference": scan[(scan["topk"].eq(33)) & (scan["alpha"].eq(1.0))].iloc[0].to_dict(),
    "checks": {
        "eligible_recipe_exists": bool(len(eligible)),
        "selected_beats_top33": float(selected_row["rmse"]) < float(
            scan[(scan["topk"].eq(33)) & (scan["alpha"].eq(1.0))]["rmse"].iloc[0]
        ),
        "selected_p95_nonworse": float(selected_row["p95_gain"]) >= 0.0,
        "selected_far_gain_ge_0p05": float(selected_row["far_gain"]) >= 0.05,
    },
    "strict_pass": bool(
        len(eligible)
        and float(selected_row["rmse"]) + 0.01
        <= float(scan[(scan["topk"].eq(33)) & (scan["alpha"].eq(1.0))]["rmse"].iloc[0])
    ),
    "fold2_touched": False,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> freeze support and replay Fold2/Fold1; FAIL -> retain V113 top33 and test dominant-mode clustering",
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("\nV117 SUPPORT TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV117 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
