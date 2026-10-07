"""V108: sparse, prefix-frozen consumption of the trained V107 block model.

V107 proved that its top64 block oracle reaches 5.60 ft, but its uncalibrated
gain model moved essentially every row and failed.  This read-only gateway
keeps the model and candidate bank fixed, freezes the first 500 ft exactly,
and lets only the highest-confidence fraction of blocks in each well leave
V76.  Fold3 discovery selects one sparse policy; the other Fold3 wells confirm
it.  Fold2 and Fold1 remain untouched.
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
EXPERIMENT = "rawdata_lattice_sparse_block_gate_v108"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V107_SOURCE = ROOT / "rawdata_lattice_parent_block_residual_v107_cell.py"
V107_MODEL = (
    OOF_ROOT / "rawdata_lattice_parent_block_residual_v107"
    / "models" / "fold45_block_gain.txt"
)
FREEZE_FT = 500.0
RETAIN_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
CONTINUITY_WEIGHTS = (0.0, 0.10, 0.30)


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
    if not V107_SOURCE.is_file():
        raise FileNotFoundError(V107_SOURCE)
    source = V107_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V107 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V107 bootstrap marker missing from {V107_SOURCE}")
    module = types.ModuleType("raw_lattice_v107_sparse_definitions")
    module.__file__ = str(V107_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V107_SOURCE), "exec"), module.__dict__)
    return module


if not V107_MODEL.is_file():
    raise FileNotFoundError(V107_MODEL)
R = _bootstrap_v107()
model = lgb.Booster(model_file=str(V107_MODEL))


def _prepare(well):
    context = R._well_context(well, labelled=False)
    context["predicted_gain"] = [model.predict(matrix) for matrix in context["x"]]
    del context["x"], context["y"]
    return context


fold3_wells = sorted(
    well for well, fold in R.fold_by_well.items()
    if fold == R.SELECTION_FOLD and well in R.available
)
if len(fold3_wells) < 153:
    raise RuntimeError(f"Only {len(fold3_wells)} Fold3 wells")

prepared, reports = [], []
for number, well in enumerate(fold3_wells, 1):
    try:
        prepared.append(_prepare(well))
        reports.append({"well": well, "status": "built"})
    except Exception as exc:
        reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 10 == 0 or number == len(fold3_wells):
        failures = sum(row["status"] == "failed" for row in reports)
        print(
            f"V108 prepare Fold3 {number}/{len(fold3_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )
pd.DataFrame(reports).to_csv(OUT_DIR / "build_report.csv", index=False)
if len(prepared) < 150:
    raise RuntimeError(f"Only {len(prepared)} prepared Fold3 wells")


def _eligible_blocks(context):
    return np.asarray([
        bool(context["distance"][rows[0]] >= FREEZE_FT)
        for rows in context["block_rows"]
    ])


def _diagnostic(contexts, stage):
    predicted_parts, actual_parts = [], []
    selected_actual = []
    selected_positive = []
    oracle_positive = []
    for context in contexts:
        eligible = _eligible_blocks(context)
        for block in np.flatnonzero(eligible):
            predicted = np.asarray(context["predicted_gain"][block], np.float64)
            actual = np.asarray(context["actual_gain"][block], np.float64)
            predicted_parts.append(predicted)
            actual_parts.append(actual)
            chosen = int(np.argmax(predicted))
            selected_actual.append(float(actual[chosen]))
            selected_positive.append(bool(actual[chosen] > 0.0))
            oracle_positive.append(bool(np.max(actual) > 0.0))
    predicted = np.concatenate(predicted_parts)
    actual = np.concatenate(actual_parts)
    correlation = pd.Series(predicted).corr(pd.Series(actual), method="spearman")
    return {
        "stage": stage,
        "candidate_rows": int(len(predicted)),
        "spearman_predicted_actual_gain": float(correlation),
        "selected_actual_gain_mean": float(np.mean(selected_actual)),
        "selected_positive_fraction": float(np.mean(selected_positive)),
        "oracle_positive_fraction": float(np.mean(oracle_positive)),
    }


def _decode(context, retain_fraction, continuity, roll_scores=False):
    predicted = [np.asarray(values, np.float64) for values in context["predicted_gain"]]
    if roll_scores and len(predicted) > 1:
        predicted = predicted[1:] + predicted[:1]
    eligible = _eligible_blocks(context)
    maxima = np.asarray([
        float(np.max(predicted[block])) if eligible[block] else -np.inf
        for block in range(len(predicted))
    ])
    eligible_maxima = maxima[eligible]
    if not len(eligible_maxima):
        return context["parent"].copy(), np.full(len(predicted), len(predicted[0]), np.int16), 0
    threshold = float(np.quantile(eligible_maxima, 1.0 - float(retain_fraction)))

    n_blocks = len(predicted)
    n_candidates = len(predicted[0])
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
        transition = (
            float(continuity) * 32.0
            * (previous_end[:, None] - current_start[None, :]) ** 2
        )
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

    output = context["parent"].copy()
    changed = 0
    for rows, state in zip(context["block_rows"], states):
        if int(state) != stay:
            output[rows] = context["candidate"][int(state), rows]
            changed += len(rows)
    return output, states, changed


def _evaluate(contexts, retain_fraction, continuity, stage, roll_scores=False):
    parts, changed_wells, changed_rows, switches = [], 0, 0, []
    for context in contexts:
        prediction, states, changed = _decode(
            context, retain_fraction, continuity, roll_scores=roll_scores
        )
        local = context["local"].copy()
        local["prediction"] = prediction.astype(np.float32)
        parts.append(local)
        changed_wells += int(changed > 0)
        changed_rows += changed
        switches.append(int(np.sum(states[1:] != states[:-1])))
    frame = pd.concat(parts, ignore_index=True)
    result = R.N._metric(frame, frame["prediction"], "rolled_control" if roll_scores else "sparse_dp")
    result.update({
        "stage": stage,
        "retain_fraction": float(retain_fraction),
        "continuity": float(continuity),
        "changed_wells": int(changed_wells),
        "changed_rows_fraction": float(changed_rows / len(frame)),
        "mean_switches": float(np.mean(switches)),
    })
    return result, frame


discovery_names = set(fold3_wells[:R.DISCOVERY_WELLS])
discovery = [item for item in prepared if item["well"] in discovery_names]
confirmation = [item for item in prepared if item["well"] not in discovery_names]


def _parent_metric(contexts, stage):
    frame = pd.concat([item["local"] for item in contexts], ignore_index=True)
    return {"stage": stage, **R.N._metric(frame, frame[R.PARENT], "v76")}


parent_discovery = _parent_metric(discovery, "fold3_discovery")
parent_confirmation = _parent_metric(confirmation, "fold3_confirmation")
diagnostics = pd.DataFrame([
    _diagnostic(discovery, "fold3_discovery"),
    _diagnostic(confirmation, "fold3_confirmation"),
])
diagnostics.to_csv(OUT_DIR / "gain_observability.csv", index=False)

scan_rows = []
for retain in RETAIN_FRACTIONS:
    for continuity in CONTINUITY_WEIGHTS:
        row, _ = _evaluate(discovery, retain, continuity, "fold3_discovery")
        row["gain"] = float(parent_discovery["rmse"] - row["rmse"])
        row["far_gain"] = float(parent_discovery["far_gt2000_rmse"] - row["far_gt2000_rmse"])
        row["near_degradation"] = float(row["near_0_500_rmse"] - parent_discovery["near_0_500_rmse"])
        row["p95_gain"] = float(parent_discovery["p95_well_rmse"] - row["p95_well_rmse"])
        scan_rows.append(row)
scan = pd.DataFrame(scan_rows).sort_values("rmse")
eligible = scan[
    (scan["near_degradation"].abs() <= 1e-10)
    & (scan["far_gain"] >= 0.0)
    & (scan["p95_gain"] >= -0.20)
]
selected_pool = eligible if len(eligible) else scan
selected = selected_pool.sort_values(["rmse", "far_gt2000_rmse"]).iloc[0]
retain = float(selected["retain_fraction"])
continuity = float(selected["continuity"])

confirmed, confirmation_frame = _evaluate(
    confirmation, retain, continuity, "fold3_confirmation"
)
rolled, _ = _evaluate(
    confirmation, retain, continuity, "fold3_confirmation", roll_scores=True
)
for row in (confirmed, rolled):
    row["gain"] = float(parent_confirmation["rmse"] - row["rmse"])
    row["far_gain"] = float(parent_confirmation["far_gt2000_rmse"] - row["far_gt2000_rmse"])
    row["near_degradation"] = float(row["near_0_500_rmse"] - parent_confirmation["near_0_500_rmse"])
    row["p95_gain"] = float(parent_confirmation["p95_well_rmse"] - row["p95_well_rmse"])

summary = pd.concat([
    scan,
    pd.DataFrame([parent_confirmation, confirmed, rolled]),
], ignore_index=True)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
confirmation_frame[["id", "well", "row_idx", "fold", R.TRUTH, "md_since", R.PARENT, "prediction"]].to_parquet(
    OUT_DIR / "fold3_confirmation_oof.parquet", index=False
)

checks = {
    "discovery_gain_ge_0p05": bool(float(selected["gain"]) >= 0.05),
    "confirmation_gain_ge_0p05": bool(confirmed["gain"] >= 0.05),
    "confirmation_far_gain_nonnegative": bool(confirmed["far_gain"] >= 0.0),
    "prefix_frozen_exact": bool(abs(confirmed["near_degradation"]) <= 1e-10),
    "confirmation_p95_not_worse_by_gt_0p20": bool(confirmed["p95_gain"] >= -0.20),
    "aligned_beats_rolled_ge_0p03": bool(confirmed["rmse"] <= rolled["rmse"] - 0.03),
    "confirmation_selected_positive_fraction_ge_0p55": bool(
        diagnostics.loc[diagnostics["stage"].eq("fold3_confirmation"), "selected_positive_fraction"].iloc[0] >= 0.55
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "V107_fixed_model_sparse_prefix_frozen_Fold3_gateway",
    "selection_or_tuning_count": len(RETAIN_FRACTIONS) * len(CONTINUITY_WEIGHTS),
    "changed_information": "none; only sparse consumption of frozen V107 scores",
    "protocol": {
        "train_folds": list(R.TRAIN_FOLDS),
        "selection_fold": R.SELECTION_FOLD,
        "selection_wells": len(discovery),
        "confirmation_wells": len(confirmation),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "freeze_ft": FREEZE_FT,
    "selected_config": {
        "retain_fraction": retain,
        "continuity_weight": continuity,
    },
    "observability": diagnostics.to_dict("records"),
    "discovery": selected.to_dict(),
    "confirmation_parent": parent_confirmation,
    "confirmation": confirmed,
    "rolled_control": rolled,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> refit on Fold345 and touch Fold2 once"
        if strict_pass else
        "FAIL -> V107 gain ordering is not transferable; close block-selector branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV108 GAIN OBSERVABILITY")
print(diagnostics.to_string(index=False))
print("\nV108 DISCOVERY SCAN")
print(scan.to_string(index=False))
print("\nV108 CONFIRMATION")
print(pd.DataFrame([parent_confirmation, confirmed, rolled]).to_string(index=False))
print("\nV108 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
print("saved:", OUT_DIR)
