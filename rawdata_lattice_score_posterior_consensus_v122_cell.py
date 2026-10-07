"""V122: consume frozen ranker score magnitude as a soft path posterior.

V113 established a useful rank-insensitive top-33 uniform center. This cell
tests the remaining label-free signal in the original LightGBM score magnitude:
scores are standardized within each complete well, converted to softmax
weights, and consumed under the exact V113 tail guard. Fold3's first 24 wells
select K/temperature; the remaining 130 wells are a single confirmation.
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
EXPERIMENT = "rawdata_lattice_score_posterior_consensus_v122"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
BASE_PATH = (
    OOF_ROOT
    / "rawdata_lattice_consensus_center_fold2_replay_v115"
    / "fold123_consensus_center_oof.parquet"
)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
TOPK_VALUES = (17, 33, 65, 129)
TEMPERATURES = (0.25, 0.50, 1.00, 2.00, 4.00)
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
EXACT = "exact_v76"
BASE = "consensus_center"


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
    module = types.ModuleType("raw_lattice_v122_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _posterior_weights(scores, temperature):
    scores = np.asarray(scores, np.float64)
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale < 1e-8:
        return np.full(len(scores), 1.0 / len(scores), np.float64)
    logits = (scores - np.max(scores)) / (scale * float(temperature))
    weights = np.exp(np.clip(logits, -50.0, 0.0))
    return weights / weights.sum()


def _stage_metric(frame, prediction, endpoint, stage, metric_fn):
    result = metric_fn(frame, prediction, endpoint)
    result["stage"] = stage
    return result


for required in (V110_SOURCE, BASE_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

R = _bootstrap_v110()
for required in (
    R.V81_SOURCE, R.PANEL_DIR, R.MODEL_PATH, R.V70_PATH, R.V76_WEIGHTS
):
    if not required.exists():
        raise FileNotFoundError(required)

N = R._bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
model = lgb.Booster(model_file=str(R.MODEL_PATH))
weight_frame = pd.read_parquet(R.V76_WEIGHTS)
weight_frame = weight_frame[weight_frame["fold"].eq(EVAL_FOLD)][
    ["well", "ensemble_weight"]
].copy()
weight_frame["well"] = weight_frame["well"].astype(str)
weight_map = dict(zip(weight_frame["well"], weight_frame["ensemble_weight"].astype(float)))

frame = pd.read_parquet(BASE_PATH, filters=[("fold", "=", EVAL_FOLD)])
lineage = pd.read_parquet(
    R.V70_PATH,
    columns=["id", TRUTH, PARENT, COARSE],
    filters=[("fold", "=", EVAL_FOLD)],
).rename(columns={TRUTH: "lineage_truth"})
if lineage["id"].duplicated().any():
    raise RuntimeError("Duplicate Fold3 ids in V70 lineage")
frame = frame.merge(lineage, on="id", how="left", validate="one_to_one")
if frame[["lineage_truth", PARENT, COARSE]].isna().any().any():
    missing_rows = int(frame[["lineage_truth", PARENT, COARSE]].isna().any(axis=1).sum())
    raise RuntimeError(f"Incomplete V70 lineage: missing_rows={missing_rows}")
lineage_truth_gap = float(
    np.max(np.abs(frame[TRUTH].to_numpy(np.float64) - frame["lineage_truth"].to_numpy(np.float64)))
)
if lineage_truth_gap > 0.01:
    raise RuntimeError(f"V70 lineage truth gap {lineage_truth_gap}")
frame = frame.drop(columns="lineage_truth")
required_columns = {
    "id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, EXACT, BASE
}
missing = sorted(required_columns.difference(frame.columns))
if missing:
    raise RuntimeError(f"Base OOF missing columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
if frame["id"].duplicated().any() or frame["well"].nunique() < 153:
    raise RuntimeError(f"Invalid Fold3 base: rows={len(frame)} wells={frame['well'].nunique()}")

wells = sorted(frame["well"].unique())
discovery_wells = set(wells[:DISCOVERY_WELLS])
confirmation_wells = set(wells[DISCOVERY_WELLS:])
recipes = [(int(k), float(temperature)) for k in TOPK_VALUES for temperature in TEMPERATURES]
predictions = {recipe: {} for recipe in recipes}
controls = {recipe: {} for recipe in recipes}
base_by_well = {}
parity_gaps = []

maximum = max(max(TOPK_VALUES), 33)
for number, well in enumerate(wells, 1):
    panel_path = R.PANEL_DIR / f"{well}.npz"
    if not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    with np.load(panel_path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
        paths = data["paths"].astype(np.int16, copy=False)
    record = V["_load_record"](well)
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: row count {len(local)}/{len(eval_rows)}")
    truth_gap = float(np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)))
    if truth_gap > 0.01:
        raise RuntimeError(f"{well}: truth gap {truth_gap}")

    scores = np.asarray(model.predict(features), np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{well}: non-finite ranker scores")
    order = np.asarray(R._deduplicated_order(paths, scores), np.int64)
    if len(order) < maximum:
        raise RuntimeError(f"{well}: only {len(order)} unique candidates")
    chosen = order[:maximum]
    chosen_scores = scores[chosen]
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    trajectories = np.vstack([
        B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
        for index in chosen
    ]).astype(np.float64)

    exact = local[EXACT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    loaded_base = local[BASE].to_numpy(np.float64)
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    dose = float(weight_map.get(well, np.nan))
    if not np.isfinite(dose) or dose < -0.01 or dose > 1.01:
        raise RuntimeError(f"{well}: invalid reconstructed V76 dose {dose}")

    uniform_center = np.mean(trajectories[:33], axis=0)
    rebuilt_base = exact + taper * np.clip(dose * (uniform_center - coarse), -CAP_FT, CAP_FT)
    parity_gaps.append(float(np.max(np.abs(rebuilt_base - loaded_base))))
    base_by_well[well] = local

    for recipe in recipes:
        k, temperature = recipe
        weights = _posterior_weights(chosen_scores[:k], temperature)
        center = np.sum(weights[:, None] * trajectories[:k], axis=0)
        seed = int.from_bytes(
            hashlib.sha256(f"v122:{well}:{k}:{temperature}".encode()).digest()[:8],
            "little",
        )
        shuffled_weights = weights[np.random.default_rng(seed).permutation(k)]
        shuffled_center = np.sum(shuffled_weights[:, None] * trajectories[:k], axis=0)
        predictions[recipe][well] = exact + taper * np.clip(
            dose * (center - coarse), -CAP_FT, CAP_FT
        )
        controls[recipe][well] = exact + taper * np.clip(
            dose * (shuffled_center - coarse), -CAP_FT, CAP_FT
        )

    if number % 10 == 0 or number == len(wells):
        print(
            f"V122 posterior {number}/{len(wells)} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

maximum_parity_gap = float(max(parity_gaps, default=np.inf))
if maximum_parity_gap > 0.01:
    raise RuntimeError(f"V113 top33 parity failed: max_gap={maximum_parity_gap}")


def _materialize(selected_wells, recipe, source):
    ordered = sorted(selected_wells)
    local_frame = pd.concat([base_by_well[well] for well in ordered], ignore_index=True)
    prediction = np.concatenate([source[recipe][well] for well in ordered])
    return local_frame, prediction


discovery_frame = pd.concat(
    [base_by_well[well] for well in sorted(discovery_wells)], ignore_index=True
)
discovery_base_prediction = discovery_frame[BASE].to_numpy(np.float64)
discovery_base = _stage_metric(
    discovery_frame, discovery_base_prediction, BASE, "fold3_discovery", R._metric
)

scan_rows = []
for recipe in recipes:
    k, temperature = recipe
    local_frame, prediction = _materialize(discovery_wells, recipe, predictions)
    _, control_prediction = _materialize(discovery_wells, recipe, controls)
    result = _stage_metric(
        local_frame, prediction, "score_posterior", "fold3_discovery", R._metric
    )
    control = R._metric(local_frame, control_prediction, "weight_shuffled_control")
    result.update({
        "topk": k,
        "temperature": temperature,
        "gain": float(discovery_base["rmse"] - result["rmse"]),
        "far_gain": float(
            discovery_base["far_gt2000_rmse"] - result["far_gt2000_rmse"]
        ),
        "p95_gain": float(discovery_base["p95_well_rmse"] - result["p95_well_rmse"]),
        "near_degradation": float(
            result["near_0_500_rmse"] - discovery_base["near_0_500_rmse"]
        ),
        "aligned_increment_vs_shuffle": float(control["rmse"] - result["rmse"]),
        "shuffled_rmse": float(control["rmse"]),
    })
    scan_rows.append(result)

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.02)
    & (scan["far_gain"] >= 0.0)
    & (scan["near_degradation"] <= 1e-10)
    & (scan["p95_gain"] >= -0.10)
    & (scan["aligned_increment_vs_shuffle"] >= 0.01)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (int(selected_row["topk"]), float(selected_row["temperature"]))

summary_rows = [discovery_base]
discovery_selected_frame, discovery_selected_prediction = _materialize(
    discovery_wells, selected, predictions
)
summary_rows.append(_stage_metric(
    discovery_selected_frame,
    discovery_selected_prediction,
    "score_posterior",
    "fold3_discovery",
    R._metric,
))

confirmation_values = None
confirmation_checks = None
if len(eligible):
    confirmation_frame, confirmation_prediction = _materialize(
        confirmation_wells, selected, predictions
    )
    _, confirmation_control_prediction = _materialize(
        confirmation_wells, selected, controls
    )
    confirmation_base_prediction = confirmation_frame[BASE].to_numpy(np.float64)
    confirmation_base = _stage_metric(
        confirmation_frame,
        confirmation_base_prediction,
        BASE,
        "fold3_confirmation",
        R._metric,
    )
    confirmation_candidate = _stage_metric(
        confirmation_frame,
        confirmation_prediction,
        "score_posterior",
        "fold3_confirmation",
        R._metric,
    )
    confirmation_control = R._metric(
        confirmation_frame,
        confirmation_control_prediction,
        "weight_shuffled_control",
    )
    summary_rows.extend([confirmation_base, confirmation_candidate])
    confirmation_values = {
        "gain": float(confirmation_base["rmse"] - confirmation_candidate["rmse"]),
        "far_gain": float(
            confirmation_base["far_gt2000_rmse"]
            - confirmation_candidate["far_gt2000_rmse"]
        ),
        "p95_gain": float(
            confirmation_base["p95_well_rmse"] - confirmation_candidate["p95_well_rmse"]
        ),
        "near_degradation": float(
            confirmation_candidate["near_0_500_rmse"]
            - confirmation_base["near_0_500_rmse"]
        ),
        "aligned_increment_vs_shuffle": float(
            confirmation_control["rmse"] - confirmation_candidate["rmse"]
        ),
    }
    confirmation_checks = {
        "gain_ge_0p02": confirmation_values["gain"] >= 0.02,
        "far_gain_nonnegative": confirmation_values["far_gain"] >= 0.0,
        "near_frozen_exact": abs(confirmation_values["near_degradation"]) <= 1e-10,
        "p95_not_worse_by_gt_0p10": confirmation_values["p95_gain"] >= -0.10,
        "aligned_beats_shuffle_ge_0p01": (
            confirmation_values["aligned_increment_vs_shuffle"] >= 0.01
        ),
    }

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
discovery_values = {
    "gain": float(selected_row["gain"]),
    "far_gain": float(selected_row["far_gain"]),
    "p95_gain": float(selected_row["p95_gain"]),
    "near_degradation": float(selected_row["near_degradation"]),
    "aligned_increment_vs_shuffle": float(selected_row["aligned_increment_vs_shuffle"]),
}
discovery_pass = bool(len(eligible))
strict_pass = bool(
    discovery_pass
    and confirmation_checks is not None
    and all(confirmation_checks.values())
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_score_posterior_discovery_confirmation",
    "candidate_generation_uses_truth": False,
    "changed_variable": "within-well standardized original ranker score magnitude",
    "baseline": "frozen V113 top33 uniform consensus on exact V76",
    "protocol": {
        "selection_fold": 3,
        "selection_wells": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "search": {
        "topk": list(TOPK_VALUES),
        "temperature": list(TEMPERATURES),
        "count": len(recipes),
    },
    "selected_recipe": {
        "topk": selected[0],
        "temperature": selected[1],
        "passed_discovery_constraints": discovery_pass,
    },
    "v70_lineage_truth_max_abs_gap": lineage_truth_gap,
    "top33_reconstruction_max_abs_gap": maximum_parity_gap,
    "discovery": discovery_values,
    "confirmation": confirmation_values,
    "confirmation_checks": confirmation_checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> replay the frozen score posterior on Fold2 and Fold1"
        if strict_pass
        else "FAIL -> score magnitude adds no transferable information; close candidate-bank weighting"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV122 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV122 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV122 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
