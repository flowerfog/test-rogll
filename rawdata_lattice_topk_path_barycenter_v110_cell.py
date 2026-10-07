"""V110: label-free top-K path barycenter on the exact V76 endpoint.

The existing ranker is much better at retrieving a useful shortlist than at
identifying its single best path.  This experiment replaces only V76's coarse
component with a robust center of the ranker's top-K trajectories.  Fold3's
first 24 wells select one frozen aggregation recipe; the remaining wells are
confirmation.  Truth is used only for recipe selection and metrics.
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
EXPERIMENT = "rawdata_lattice_topk_path_barycenter_v110"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V81_SOURCE = ROOT / "rawdata_lattice_structural_risk_ranker_v81b_cell.py"
PANEL_DIR = OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103" / "matched_panels"
MODEL_PATH = OOF_ROOT / "rawdata_lattice_parentrelative_ranker_v104" / "models" / "fold45_base.txt"
V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

FOLD = 3
DISCOVERY_WELLS = 24
TOPK_VALUES = (3, 5, 9, 17, 33)
AGGREGATORS = ("median", "trimmed_mean", "rank_weighted")
ALPHAS = (0.25, 0.50, 0.75, 1.00)
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


def _bootstrap_v81():
    source = V81_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V81B EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V81b marker missing from {V81_SOURCE}")
    module = types.ModuleType("raw_lattice_v81_topk_barycenter_definitions")
    module.__file__ = str(V81_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V81_SOURCE), "exec"), module.__dict__)
    return module


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid endpoint {endpoint}")
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    output = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    masks = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in masks.items():
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


def _aggregate(values, method):
    values = np.asarray(values, np.float64)
    if method == "median":
        return np.median(values, axis=0)
    if method == "trimmed_mean":
        ordered = np.sort(values, axis=0)
        trim = max(1, int(np.floor(len(values) * 0.20))) if len(values) >= 5 else 0
        return np.mean(ordered[trim: len(values) - trim], axis=0) if trim else np.mean(ordered, axis=0)
    if method == "rank_weighted":
        scale = max(len(values) / 3.0, 1.0)
        weight = np.exp(-np.arange(len(values), dtype=np.float64) / scale)
        weight /= weight.sum()
        return np.sum(weight[:, None] * values, axis=0)
    raise KeyError(method)


def _deduplicated_order(paths, scores):
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    retained, seen = [], set()
    for index in order:
        key = np.ascontiguousarray(paths[index]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        retained.append(int(index))
    return np.asarray(retained, np.int32)


# --- V110 EXECUTION ---

for required in (V81_SOURCE, PANEL_DIR, MODEL_PATH, V70_PATH, V76_WEIGHTS):
    if not required.exists():
        raise FileNotFoundError(required)

N = _bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
model = lgb.Booster(model_file=str(MODEL_PATH))

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
    filters=[("fold", "=", FOLD)],
).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
weights = pd.read_parquet(V76_WEIGHTS)
weights = weights[weights["fold"].eq(FOLD)][["well", "ensemble_weight"]].copy()
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))
frame["dose"] = frame["well"].map(weight_map)
if frame.empty or frame["dose"].isna().any() or frame["id"].duplicated().any():
    raise RuntimeError("Incomplete exact Fold3 V70/V76 alignment")
frame["exact_v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
    frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
)

wells = sorted(
    well for well in frame["well"].unique()
    if (PANEL_DIR / f"{well}.npz").is_file()
)
if len(wells) < 153:
    raise RuntimeError(f"Only {len(wells)} Fold3 panels")
discovery_wells = set(wells[:DISCOVERY_WELLS])

recipes = [
    (k, method, alpha)
    for k in TOPK_VALUES
    for method in AGGREGATORS
    for alpha in ALPHAS
]
prediction_parts = {recipe: [] for recipe in recipes}
control_parts = {recipe: [] for recipe in recipes}
base_parts, report_rows = [], []

for number, well in enumerate(wells, 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    with np.load(PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
        paths = data["paths"].astype(np.int16, copy=False)
    record = V["_load_record"](well)
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: exact V76/record alignment")
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    scores = model.predict(features)
    order = _deduplicated_order(paths, scores)
    maximum = 2 * max(TOPK_VALUES)
    if len(order) < maximum:
        raise RuntimeError(f"{well}: only {len(order)} unique candidates")
    retained = order[:maximum]
    trajectories = np.vstack([
        B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
        for index in retained
    ]).astype(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    dose = float(local["dose"].iloc[0])
    base_parts.append(local)
    for recipe in recipes:
        k, method, alpha = recipe
        top_center = _aggregate(trajectories[:k], method)
        next_center = _aggregate(trajectories[k: 2 * k], method)
        prediction_parts[recipe].append(exact + alpha * dose * (top_center - coarse))
        control_parts[recipe].append(exact + alpha * dose * (next_center - coarse))
    report_rows.append({
        "well": well,
        "split": "discovery" if well in discovery_wells else "confirmation",
        "rows": len(local),
        "dose": dose,
        "candidates": len(paths),
        "unique_candidates": len(order),
        "top1_score": float(scores[order[0]]),
        "top2_margin": float(scores[order[0]] - scores[order[1]]),
    })
    if number % 10 == 0 or number == len(wells):
        print(
            f"V110 trajectories {number}/{len(wells)} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

base_frame = pd.concat(base_parts, ignore_index=True)
pd.DataFrame(report_rows).to_csv(OUT_DIR / "well_report.csv", index=False)
discovery_mask = base_frame["well"].isin(discovery_wells).to_numpy()
confirmation_mask = ~discovery_mask
base_prediction = base_frame["exact_v76"].to_numpy(np.float64)

scan_rows = []
for recipe in recipes:
    k, method, alpha = recipe
    prediction = np.concatenate(prediction_parts[recipe])
    result = _metric(base_frame.loc[discovery_mask], prediction[discovery_mask], "topk_barycenter")
    base = _metric(base_frame.loc[discovery_mask], base_prediction[discovery_mask], "exact_v76")
    result.update({
        "topk": k,
        "aggregator": method,
        "alpha": alpha,
        "gain": float(base["rmse"] - result["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
        "near_degradation": float(result["near_0_500_rmse"] - base["near_0_500_rmse"]),
        "p95_gain": float(base["p95_well_rmse"] - result["p95_well_rmse"]),
    })
    scan_rows.append(result)

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "topk", "alpha"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.03)
    & (scan["far_gain"] >= 0.0)
    & (scan["near_degradation"] <= 0.05)
    & (scan["p95_gain"] >= -0.20)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (int(selected_row["topk"]), str(selected_row["aggregator"]), float(selected_row["alpha"]))

selected_prediction = np.concatenate(prediction_parts[selected])
control_prediction = np.concatenate(control_parts[selected])
summary_rows = []
stage_metrics = {}
for stage, mask in (("fold3_discovery", discovery_mask), ("fold3_confirmation", confirmation_mask)):
    local = base_frame.loc[mask]
    metrics = {
        "exact_v76": _metric(local, base_prediction[mask], "exact_v76"),
        "topk_barycenter": _metric(local, selected_prediction[mask], "topk_barycenter"),
        "nextk_control": _metric(local, control_prediction[mask], "nextk_control"),
    }
    stage_metrics[stage] = metrics
    for metric in metrics.values():
        summary_rows.append({"stage": stage, **metric})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
confirmation = stage_metrics["fold3_confirmation"]
base = confirmation["exact_v76"]
candidate = confirmation["topk_barycenter"]
control = confirmation["nextk_control"]
values = {
    "discovery_gain": float(selected_row["gain"]),
    "confirmation_gain": float(base["rmse"] - candidate["rmse"]),
    "confirmation_far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
    "confirmation_near_degradation": float(candidate["near_0_500_rmse"] - base["near_0_500_rmse"]),
    "confirmation_p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
    "topk_increment_vs_nextk": float(control["rmse"] - candidate["rmse"]),
}
checks = {
    "discovery_gain_ge_0p03": values["discovery_gain"] >= 0.03,
    "confirmation_gain_ge_0p03": values["confirmation_gain"] >= 0.03,
    "confirmation_far_gain_nonnegative": values["confirmation_far_gain"] >= 0.0,
    "confirmation_near_degradation_le_0p05": values["confirmation_near_degradation"] <= 0.05,
    "confirmation_p95_not_worse_by_gt_0p20": values["confirmation_p95_gain"] >= -0.20,
    "topk_beats_nextk_ge_0p03": values["topk_increment_vs_nextk"] >= 0.03,
}
strict_pass = bool(all(checks.values()))

output = base_frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, "exact_v76"]].copy()
output["topk_barycenter"] = selected_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_topk_barycenter_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_24well_recipe_selection_130well_confirmation",
    "sample_unit": "one complete physical well",
    "candidate_generation_uses_truth": False,
    "truth_usage": "discovery recipe selection and heldout metrics only",
    "base": "exact V76",
    "formula": "exact_V76 + alpha*V76_dose*(topK_path_center-persisted_coarse)",
    "search": {
        "topk": list(TOPK_VALUES),
        "aggregators": list(AGGREGATORS),
        "alphas": list(ALPHAS),
        "count": len(recipes),
    },
    "selected_recipe": {
        "topk": selected[0],
        "aggregator": selected[1],
        "alpha": selected[2],
        "passed_discovery_constraints": bool(len(eligible)),
    },
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "fold2_touched": False,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild the same barycenter endpoint on Fold2 once"
        if strict_pass else
        "FAIL -> shortlist averaging is not a useful candidate generator; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV110 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV110 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV110 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
