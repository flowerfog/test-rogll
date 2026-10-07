"""V122b: read-only confirmation of V122's best subthreshold recipe.

The frozen recipe is K=65, temperature=4.0. The remaining 130 Fold3 wells are
evaluated once. Uniform top-65 and repeated weight-shuffle controls attribute
gain to candidate support versus original ranker score magnitude.
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
EXPERIMENT = "rawdata_lattice_score_posterior_confirmation_v122b"
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
TOPK = 65
TEMPERATURE = 4.0
SHUFFLE_DRAWS = 16
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
TRUTH = "tvt_true"
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
    module = types.ModuleType("raw_lattice_v122b_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _posterior_weights(scores):
    scores = np.asarray(scores, np.float64)
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale < 1e-8:
        return np.full(len(scores), 1.0 / len(scores), np.float64)
    logits = (scores - np.max(scores)) / (scale * TEMPERATURE)
    weights = np.exp(np.clip(logits, -50.0, 0.0))
    return weights / weights.sum()


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

weights_frame = pd.read_parquet(R.V76_WEIGHTS)
weights_frame = weights_frame[weights_frame["fold"].eq(EVAL_FOLD)][
    ["well", "ensemble_weight"]
].copy()
weights_frame["well"] = weights_frame["well"].astype(str)
weight_map = dict(
    zip(weights_frame["well"], weights_frame["ensemble_weight"].astype(float))
)

frame = pd.read_parquet(BASE_PATH, filters=[("fold", "=", EVAL_FOLD)])
lineage = pd.read_parquet(
    R.V70_PATH,
    columns=["id", TRUTH, COARSE],
    filters=[("fold", "=", EVAL_FOLD)],
).rename(columns={TRUTH: "lineage_truth"})
if lineage["id"].duplicated().any():
    raise RuntimeError("Duplicate Fold3 ids in V70 lineage")
frame = frame.merge(lineage, on="id", how="left", validate="one_to_one")
required_columns = {
    "id", "well", "row_idx", "fold", TRUTH, "lineage_truth", "md_since",
    COARSE, EXACT, BASE,
}
missing = sorted(required_columns.difference(frame.columns))
if missing:
    raise RuntimeError(f"Confirmation OOF missing columns: {missing}")
if frame[["lineage_truth", COARSE]].isna().any().any():
    raise RuntimeError("Incomplete V70 Fold3 lineage")
truth_gap = float(np.max(np.abs(
    frame[TRUTH].to_numpy(np.float64) - frame["lineage_truth"].to_numpy(np.float64)
)))
if truth_gap > 0.01:
    raise RuntimeError(f"V70 truth gap {truth_gap}")
frame = frame.drop(columns="lineage_truth")
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
wells = sorted(frame["well"].unique())
confirmation_wells = wells[DISCOVERY_WELLS:]
if len(confirmation_wells) < 129:
    raise RuntimeError(f"Only {len(confirmation_wells)} confirmation wells")

frame_parts = []
posterior_parts = []
uniform_parts = []
shuffle_parts = [[] for _ in range(SHUFFLE_DRAWS)]
parity_gaps = []

for number, well in enumerate(confirmation_wells, 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    panel_path = R.PANEL_DIR / f"{well}.npz"
    if not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    with np.load(panel_path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
        paths = data["paths"].astype(np.int16, copy=False)
    record = V["_load_record"](well)
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: row count {len(local)}/{len(eval_rows)}")
    local_truth_gap = float(np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)))
    if local_truth_gap > 0.01:
        raise RuntimeError(f"{well}: truth gap {local_truth_gap}")

    scores = np.asarray(model.predict(features), np.float64)
    order = np.asarray(R._deduplicated_order(paths, scores), np.int64)
    if len(order) < TOPK:
        raise RuntimeError(f"{well}: only {len(order)} unique candidates")
    chosen = order[:TOPK]
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    trajectories = np.vstack([
        B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
        for index in chosen
    ]).astype(np.float64)

    exact = local[EXACT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    dose = float(weight_map.get(well, np.nan))
    if not np.isfinite(dose):
        raise RuntimeError(f"{well}: missing V76 dose")
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    score_weights = _posterior_weights(scores[chosen])
    posterior_center = np.sum(score_weights[:, None] * trajectories, axis=0)
    uniform_center = np.mean(trajectories, axis=0)
    posterior = exact + taper * np.clip(
        dose * (posterior_center - coarse), -CAP_FT, CAP_FT
    )
    uniform = exact + taper * np.clip(
        dose * (uniform_center - coarse), -CAP_FT, CAP_FT
    )

    # Rebuild the loaded V113 baseline from the same candidate order.
    top33_center = np.mean(trajectories[:33], axis=0)
    rebuilt_top33 = exact + taper * np.clip(
        dose * (top33_center - coarse), -CAP_FT, CAP_FT
    )
    parity_gaps.append(float(np.max(np.abs(
        rebuilt_top33 - local[BASE].to_numpy(np.float64)
    ))))

    for draw in range(SHUFFLE_DRAWS):
        seed = int.from_bytes(
            hashlib.sha256(f"v122b:{draw}:{well}".encode()).digest()[:8], "little"
        )
        shuffled = score_weights[np.random.default_rng(seed).permutation(TOPK)]
        center = np.sum(shuffled[:, None] * trajectories, axis=0)
        shuffle_parts[draw].append(
            exact + taper * np.clip(dose * (center - coarse), -CAP_FT, CAP_FT)
        )

    frame_parts.append(local)
    posterior_parts.append(posterior)
    uniform_parts.append(uniform)
    if number % 10 == 0 or number == len(confirmation_wells):
        print(
            f"V122b confirmation {number}/{len(confirmation_wells)} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

maximum_parity_gap = float(max(parity_gaps, default=np.inf))
if maximum_parity_gap > 0.01:
    raise RuntimeError(f"V113 top33 parity failed: max_gap={maximum_parity_gap}")

matched = pd.concat(frame_parts, ignore_index=True)
base_prediction = matched[BASE].to_numpy(np.float64)
posterior_prediction = np.concatenate(posterior_parts)
uniform_prediction = np.concatenate(uniform_parts)
base = R._metric(matched, base_prediction, "consensus_top33")
posterior = R._metric(matched, posterior_prediction, "score_posterior_top65")
uniform = R._metric(matched, uniform_prediction, "uniform_top65")
shuffle_metrics = []
for draw in range(SHUFFLE_DRAWS):
    metric = R._metric(
        matched, np.concatenate(shuffle_parts[draw]), f"weight_shuffle_{draw:02d}"
    )
    metric["draw"] = draw
    shuffle_metrics.append(metric)
shuffle_frame = pd.DataFrame(shuffle_metrics)
shuffle_frame.to_csv(OUT_DIR / "shuffle_metrics.csv", index=False)

summary = pd.DataFrame([base, posterior, uniform])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
shuffle_rmse = shuffle_frame["rmse"].to_numpy(np.float64)
values = {
    "posterior_gain_vs_top33": float(base["rmse"] - posterior["rmse"]),
    "uniform65_gain_vs_top33": float(base["rmse"] - uniform["rmse"]),
    "score_increment_over_uniform65": float(uniform["rmse"] - posterior["rmse"]),
    "posterior_far_gain": float(
        base["far_gt2000_rmse"] - posterior["far_gt2000_rmse"]
    ),
    "posterior_p95_gain": float(base["p95_well_rmse"] - posterior["p95_well_rmse"]),
    "posterior_near_degradation": float(
        posterior["near_0_500_rmse"] - base["near_0_500_rmse"]
    ),
    "shuffle_rmse_p05": float(np.quantile(shuffle_rmse, 0.05)),
    "shuffle_rmse_median": float(np.median(shuffle_rmse)),
    "shuffle_rmse_p95": float(np.quantile(shuffle_rmse, 0.95)),
    "aligned_increment_vs_median_shuffle": float(
        np.median(shuffle_rmse) - posterior["rmse"]
    ),
}
posterior_checks = {
    "gain_ge_0p03": values["posterior_gain_vs_top33"] >= 0.03,
    "far_gain_nonnegative": values["posterior_far_gain"] >= 0.0,
    "p95_not_worse_by_gt_0p10": values["posterior_p95_gain"] >= -0.10,
    "near_frozen_with_float_tolerance": abs(values["posterior_near_degradation"]) <= 1e-5,
    "score_beats_uniform65_ge_0p01": values["score_increment_over_uniform65"] >= 0.01,
    "aligned_beats_median_shuffle_ge_0p01": (
        values["aligned_increment_vs_median_shuffle"] >= 0.01
    ),
}
uniform_checks = {
    "gain_ge_0p03": values["uniform65_gain_vs_top33"] >= 0.03,
    "far_gain_nonnegative": (
        base["far_gt2000_rmse"] - uniform["far_gt2000_rmse"]
    ) >= 0.0,
    "p95_not_worse_by_gt_0p10": (
        base["p95_well_rmse"] - uniform["p95_well_rmse"]
    ) >= -0.10,
    "near_frozen_with_float_tolerance": abs(
        uniform["near_0_500_rmse"] - base["near_0_500_rmse"]
    ) <= 1e-5,
}

output = matched[["id", "well", "row_idx", "fold", TRUTH, "md_since", EXACT, BASE]].copy()
output["score_posterior_top65"] = posterior_prediction.astype(np.float32)
output["uniform_top65"] = uniform_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")

posterior_pass = bool(all(posterior_checks.values()))
uniform_pass = bool(all(uniform_checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "read_only_confirmation_of_V122_best_subthreshold_recipe",
    "selection_or_tuning_count": 0,
    "frozen_recipe": {"topk": TOPK, "temperature": TEMPERATURE},
    "protocol": {
        "selection_fold": 3,
        "selection_wells_excluded": DISCOVERY_WELLS,
        "confirmation_wells": len(confirmation_wells),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "controls": {
        "uniform_same_support": True,
        "weight_shuffle_draws": SHUFFLE_DRAWS,
    },
    "lineage_truth_max_abs_gap": truth_gap,
    "top33_reconstruction_max_abs_gap": maximum_parity_gap,
    "values": values,
    "posterior_checks": posterior_checks,
    "uniform65_checks": uniform_checks,
    "posterior_pass": posterior_pass,
    "uniform65_pass": uniform_pass,
    "strict_pass": bool(posterior_pass or uniform_pass),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS posterior -> replay frozen score posterior on Fold2 and Fold1"
        if posterior_pass else
        "PASS support -> replay uniform top65 on Fold2 and Fold1"
        if uniform_pass else
        "FAIL -> close candidate-bank score/support weighting"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV122b MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV122b SHUFFLE SUMMARY")
print(shuffle_frame[["draw", "rmse", "far_gt2000_rmse", "p95_well_rmse"]].to_string(index=False), flush=True)
print("\nV122b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
