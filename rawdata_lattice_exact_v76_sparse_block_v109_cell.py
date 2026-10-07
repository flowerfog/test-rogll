"""V109: exact-V76 sparse block test.

V108 accidentally used the pre-coarse V66 lineage as its stay state.  This
experiment repairs that attribution.  It trains the unchanged V107 block-gain
regressor on exact Fold1/2 V76 OOF baselines and applies the already frozen
V108 policy (freeze 500 ft, retain 40%, continuity 0.3) once to all of Fold3.
There is no Fold3 selection in this cell; Fold1 and Fold2 are training data.
"""

from __future__ import annotations

import gc
import json
import os
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
EXPERIMENT = "rawdata_lattice_exact_v76_sparse_block_v109"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V107_SOURCE = ROOT / "rawdata_lattice_parent_block_residual_v107_cell.py"
V70_PATH = (
    OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70"
    / "all773_proxy_crossfit_oof.parquet"
)
TRAIN_FOLDS = (1, 2)
APPLICATION_FOLD = 3
FREEZE_FT = 500.0
RETAIN_FRACTION = 0.40
CONTINUITY_WEIGHT = 0.30
RANDOM_SEED = 20260801
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


def _bootstrap_v107():
    source = V107_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V107 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V107 bootstrap marker missing from {V107_SOURCE}")
    module = types.ModuleType("raw_lattice_v107_exact_v76_definitions")
    module.__file__ = str(V107_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V107_SOURCE), "exec"), module.__dict__)
    return module


for required in (V107_SOURCE, V70_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
R = _bootstrap_v107()

v70 = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", R.TRUTH, "md_since", R.PARENT, COARSE],
).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
v70["well"] = v70["well"].astype(str)
if v70["id"].duplicated().any():
    raise RuntimeError("Duplicate V70 ids")


def _exact_baseline(well):
    local = v70[v70["well"].eq(well)].sort_values("row_idx", kind="stable")
    if local.empty or well not in R.weight_map:
        raise RuntimeError(f"{well}: missing exact V70/V76 data")
    dose = float(R.weight_map[well])
    parent = local[R.PARENT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    return local, parent + dose * (coarse - parent)


def _context(well, labelled):
    exact, baseline = _exact_baseline(well)
    context = R._well_context(
        well, labelled=labelled, baseline_override=baseline
    )
    local = context["local"]
    if len(local) != len(exact) or not local["id"].astype(str).reset_index(drop=True).equals(
        exact["id"].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError(f"{well}: exact baseline id alignment")
    truth_gap = float(np.max(np.abs(
        local[R.TRUTH].to_numpy(np.float64) - exact[R.TRUTH].to_numpy(np.float64)
    )))
    if truth_gap > 0.01:
        raise RuntimeError(f"{well}: truth gap {truth_gap}")
    return context


train_wells = sorted(v70.loc[v70["fold"].isin(TRAIN_FOLDS), "well"].unique())
fold3_wells = sorted(v70.loc[v70["fold"].eq(APPLICATION_FOLD), "well"].unique())
if len(train_wells) != 310 or len(fold3_wells) != 154:
    raise RuntimeError(f"Unexpected exact coverage train={len(train_wells)} fold3={len(fold3_wells)}")

print(
    "V109 CONFIG",
    {
        "train_folds": TRAIN_FOLDS, "train_wells": len(train_wells),
        "application_fold": APPLICATION_FOLD, "application_wells": len(fold3_wells),
        "freeze_ft": FREEZE_FT, "retain_fraction": RETAIN_FRACTION,
        "continuity": CONTINUITY_WEIGHT,
    },
    flush=True,
)

x_parts, y_parts, weights_parts, reports = [], [], [], []
for number, well in enumerate(train_wells, 1):
    try:
        context = _context(well, labelled=True)
        for matrix, target, rows in zip(context["x"], context["y"], context["row_counts"]):
            x_parts.append(matrix)
            y_parts.append(target)
            weights_parts.append(np.full(len(target), rows, np.float32))
        reports.append({"well": well, "status": "built", "blocks": len(context["x"])})
    except Exception as exc:
        reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 20 == 0 or number == len(train_wells):
        failures = sum(row["status"] == "failed" for row in reports)
        print(
            f"V109 train {number}/{len(train_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )
pd.DataFrame(reports).to_csv(OUT_DIR / "train_report.csv", index=False)
if sum(row["status"] == "built" for row in reports) < 300:
    raise RuntimeError(f"Insufficient training coverage: {pd.DataFrame(reports)['status'].value_counts().to_dict()}")

x_train = np.concatenate(x_parts).astype(np.float32, copy=False)
y_train = np.concatenate(y_parts).astype(np.float32, copy=False)
sample_weight = np.concatenate(weights_parts).astype(np.float32, copy=False)
low, high = np.quantile(y_train, [0.005, 0.995])
y_train = np.clip(y_train, low, high)
model = lgb.LGBMRegressor(
    objective="huber", alpha=0.85, n_estimators=450, learning_rate=0.035,
    num_leaves=31, max_depth=-1, min_child_samples=160,
    subsample=0.85, colsample_bytree=0.80,
    reg_alpha=0.5, reg_lambda=8.0, random_state=RANDOM_SEED,
    n_jobs=max(1, min(16, os.cpu_count() or 1)), verbosity=-1,
)
model.fit(x_train, y_train, sample_weight=sample_weight)
model.booster_.save_model(str(MODEL_DIR / "fold12_exact_v76_block_gain.txt"))
del x_train, y_train, sample_weight, x_parts, y_parts, weights_parts
gc.collect()

prepared, eval_reports = [], []
for number, well in enumerate(fold3_wells, 1):
    try:
        context = _context(well, labelled=False)
        context["predicted_gain"] = [model.predict(matrix) for matrix in context["x"]]
        del context["x"], context["y"]
        prepared.append(context)
        eval_reports.append({"well": well, "status": "built"})
    except Exception as exc:
        eval_reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 10 == 0 or number == len(fold3_wells):
        failures = sum(row["status"] == "failed" for row in eval_reports)
        print(
            f"V109 apply Fold3 {number}/{len(fold3_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )
pd.DataFrame(eval_reports).to_csv(OUT_DIR / "fold3_report.csv", index=False)
if len(prepared) < 150:
    raise RuntimeError(f"Only {len(prepared)} Fold3 contexts")


def _eligible(context):
    return np.asarray([
        bool(context["distance"][rows[0]] >= FREEZE_FT)
        for rows in context["block_rows"]
    ])


def _decode(context, roll=False):
    predicted = [np.asarray(values, np.float64) for values in context["predicted_gain"]]
    if roll:
        predicted = predicted[1:] + predicted[:1]
    eligible = _eligible(context)
    maxima = np.asarray([
        np.max(predicted[i]) if eligible[i] else -np.inf for i in range(len(predicted))
    ])
    threshold = float(np.quantile(maxima[eligible], 1.0 - RETAIN_FRACTION))
    n_blocks, n_candidates = len(predicted), len(predicted[0])
    stay = n_candidates
    back = np.full((n_blocks, n_candidates + 1), -1, np.int16)
    score = np.zeros(n_candidates + 1, np.float64)
    for block in range(n_blocks):
        if eligible[block] and maxima[block] >= threshold:
            emission = np.r_[
                (predicted[block] - threshold) * context["row_counts"][block], 0.0
            ]
        else:
            emission = np.r_[np.full(n_candidates, -1e15), 0.0]
        if block == 0:
            score = emission
            continue
        previous_end = np.r_[context["move_end"][block - 1], 0.0]
        current_start = np.r_[context["move_start"][block], 0.0]
        transition = CONTINUITY_WEIGHT * 32.0 * (
            previous_end[:, None] - current_start[None, :]
        ) ** 2
        values = score[:, None] - transition
        previous = np.argmax(values, axis=0)
        score = values[previous, np.arange(n_candidates + 1)] + emission
        back[block] = previous.astype(np.int16)
    state = int(np.argmax(score))
    states = np.empty(n_blocks, np.int16)
    states[-1] = state
    for block in range(n_blocks - 1, 0, -1):
        state = int(back[block, state])
        states[block - 1] = state
    prediction = context["parent"].copy()
    changed = 0
    for rows, state in zip(context["block_rows"], states):
        if int(state) != stay:
            prediction[rows] = context["candidate"][int(state), rows]
            changed += len(rows)
    return prediction, states, changed


def _endpoint(roll=False):
    parts, changed_wells, changed_rows, switches = [], 0, 0, []
    for context in prepared:
        prediction, states, changed = _decode(context, roll=roll)
        local = context["local"].copy()
        local["exact_v76"] = context["parent"].astype(np.float32)
        local["prediction"] = prediction.astype(np.float32)
        parts.append(local)
        changed_wells += int(changed > 0)
        changed_rows += changed
        switches.append(int(np.sum(states[1:] != states[:-1])))
    frame = pd.concat(parts, ignore_index=True)
    metric = R.N._metric(frame, frame["prediction"], "rolled_control" if roll else "exact_v76_sparse")
    metric.update({
        "changed_wells": changed_wells,
        "changed_rows_fraction": changed_rows / len(frame),
        "mean_switches": float(np.mean(switches)),
    })
    return metric, frame


base_frame = pd.concat([item["local"] for item in prepared], ignore_index=True)
base = R.N._metric(base_frame, np.concatenate([item["parent"] for item in prepared]), "exact_v76")
candidate, candidate_frame = _endpoint(False)
rolled, _ = _endpoint(True)
for row in (candidate, rolled):
    row["gain"] = float(base["rmse"] - row["rmse"])
    row["far_gain"] = float(base["far_gt2000_rmse"] - row["far_gt2000_rmse"])
    row["near_degradation"] = float(row["near_0_500_rmse"] - base["near_0_500_rmse"])
    row["p95_gain"] = float(base["p95_well_rmse"] - row["p95_well_rmse"])

summary = pd.DataFrame([base, candidate, rolled])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
candidate_frame[[
    "id", "well", "row_idx", "fold", R.TRUTH, "md_since",
    R.PARENT, "exact_v76", "prediction",
]].to_parquet(
    OUT_DIR / "fold3_exact_v76_sparse_oof.parquet", index=False
)
checks = {
    "gain_ge_0p05": bool(candidate["gain"] >= 0.05),
    "far_gain_nonnegative": bool(candidate["far_gain"] >= 0.0),
    "prefix_frozen_exact": bool(abs(candidate["near_degradation"]) <= 1e-10),
    "p95_not_worse_by_gt_0p20": bool(candidate["p95_gain"] >= -0.20),
    "aligned_beats_rolled_ge_0p03": bool(candidate["rmse"] <= rolled["rmse"] - 0.03),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_exact_V76_training_Fold3_matched_rebaseline_attribution",
    "correction_of": "V108 used V66 rather than exact V76 as stay state",
    "selection_or_tuning_count": 0,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "attribution_fold": APPLICATION_FOLD,
        "attribution_wells": len(prepared),
        "fold1_used_for_training": True,
        "fold2_used_for_training": True,
        "fold3_selection_or_search_count_in_v109": 0,
        "fold3_was_used_to_select_the_frozen_v108_recipe": True,
    },
    "frozen_recipe": {
        "freeze_ft": FREEZE_FT,
        "retain_fraction": RETAIN_FRACTION,
        "continuity_weight": CONTINUITY_WEIGHT,
        "topk": R.TOPK,
    },
    "base": base,
    "candidate": candidate,
    "rolled_control": rolled,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build exact Fold4/5 V76 baselines for an independent confirmation"
        if strict_pass else
        "FAIL -> V108 gain was entirely pre-coarse attribution; close block branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV109 MATCHED SUMMARY")
print(summary.to_string(index=False))
print("\nV109 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
print("saved:", OUT_DIR)
