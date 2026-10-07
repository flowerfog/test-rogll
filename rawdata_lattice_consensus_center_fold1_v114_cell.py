"""V114: final Fold1 confirmation of the frozen V113 consensus center."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import types
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_consensus_center_fold1_v114"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "fold1_panels"
for directory in (OUT_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V112_SOURCE = ROOT / "rawdata_lattice_topk_barycenter_fold2_v112_cell.py"
FOLD = 1
TOPK = 33
ALPHA = 1.0
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


def _bootstrap_v112():
    source = V112_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V112 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V112 marker missing from {V112_SOURCE}")
    module = types.ModuleType("raw_lattice_v112_consensus_fold1_definitions")
    module.__file__ = str(V112_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V112_SOURCE), "exec"), module.__dict__)
    return module


if not V112_SOURCE.is_file():
    raise FileNotFoundError(V112_SOURCE)
R = _bootstrap_v112()
for required in (R.V81_SOURCE, R.MODEL_PATH, R.V70_PATH, R.V76_WEIGHTS):
    if not required.is_file():
        raise FileNotFoundError(required)

N = R._bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
feature_names = list(N.feature_names)
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
    raise RuntimeError("Incomplete exact Fold1 V76 alignment")
if frame["well"].nunique() != 155:
    raise RuntimeError(f"Only {frame['well'].nunique()} Fold1 wells")
frame["exact_v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
    frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
)


def _panel(well, record):
    path = CACHE_DIR / f"{well}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            features = data["features"]
            paths = data["paths"]
        if features.shape[1] == len(feature_names) and len(features) == len(paths):
            return features, paths, "reused"
    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        features, names = Q._candidate_evidence(
            record, prior, paths, segments, arm_rank, arm_score_z, source_mask
        )
    if list(names) != feature_names:
        raise RuntimeError(f"{well}: feature schema mismatch")
    np.savez_compressed(path, features=features.astype(np.float32), paths=paths.astype(np.int16))
    return features, paths, "built"


candidate_parts, control_parts, report_rows = [], [], []
for number, well in enumerate(sorted(frame["well"].unique()), 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable")
    exact = local["exact_v76"].to_numpy(np.float64)
    candidate = exact.copy()
    control = exact.copy()
    report = {"well": well, "status": "fallback", "reason": ""}
    try:
        record = V["_load_record"](well)
        eval_rows = record["eval_rows"]
        truth = record["truth"][eval_rows].astype(np.float64)
        distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
        if len(local) != len(eval_rows):
            raise RuntimeError(f"row count {len(local)}/{len(eval_rows)}")
        truth_gap = float(np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)))
        distance_gap = float(np.max(np.abs(local["md_since"].to_numpy(np.float64) - distance)))
        if truth_gap > 0.01 or distance_gap > 1.01:
            raise RuntimeError(f"alignment truth={truth_gap} distance={distance_gap}")
        features, paths, panel_status = _panel(well, record)
        scores = model.predict(features)
        order = R._deduplicated_order(paths, scores)
        if len(order) < 2 * TOPK:
            raise RuntimeError(f"only {len(order)} unique paths")
        segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
        trajectories = np.vstack([
            B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
            for index in order[: 2 * TOPK]
        ]).astype(np.float64)
        seed = int.from_bytes(hashlib.sha256(f"v113:{well}".encode()).digest()[:8], "little")
        shuffled = trajectories[np.random.default_rng(seed).permutation(len(trajectories))]
        center = np.mean(trajectories[:TOPK], axis=0)
        shuffled_center = np.mean(shuffled[:TOPK], axis=0)
        coarse = local[COARSE].to_numpy(np.float64)
        dose = float(local["dose"].iloc[0])
        taper = R._smoothstep((distance - FREEZE_FT) / RAMP_FT)
        move = np.clip(ALPHA * dose * (center - coarse), -CAP_FT, CAP_FT)
        shuffled_move = np.clip(ALPHA * dose * (shuffled_center - coarse), -CAP_FT, CAP_FT)
        candidate = exact + taper * move
        control = exact + taper * shuffled_move
        report.update({
            "status": panel_status,
            "candidates": int(len(paths)),
            "unique_candidates": int(len(order)),
            "mean_abs_move": float(np.mean(np.abs(candidate - exact))),
            "p95_abs_move": float(np.quantile(np.abs(candidate - exact), 0.95)),
        })
    except Exception as exc:
        report["reason"] = repr(exc)
    candidate_parts.append(candidate)
    control_parts.append(control)
    report_rows.append(report)
    if number % 10 == 0 or number == frame["well"].nunique():
        fallbacks = sum(row["status"] == "fallback" for row in report_rows)
        print(
            f"V114 Fold1 {number}/{frame['well'].nunique()} fallbacks={fallbacks} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

report = pd.DataFrame(report_rows)
report.to_csv(OUT_DIR / "build_report.csv", index=False)
fallback_wells = int(report["status"].eq("fallback").sum())
if fallback_wells > 5:
    raise RuntimeError(f"Too many Fold1 fallbacks: {fallback_wells}")

base_prediction = frame["exact_v76"].to_numpy(np.float64)
candidate_prediction = np.concatenate(candidate_parts)
control_prediction = np.concatenate(control_parts)
base_metric, base_bywell = R._metric(frame, base_prediction, "exact_v76")
candidate_metric, candidate_bywell = R._metric(frame, candidate_prediction, "consensus_center")
control_metric, _ = R._metric(frame, control_prediction, "score_shuffled_control")
summary = pd.DataFrame([base_metric, candidate_metric, control_metric])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

bywell = pd.DataFrame({
    "well": base_bywell.index,
    "base_rmse": base_bywell.to_numpy(),
    "candidate_rmse": candidate_bywell.reindex(base_bywell.index).to_numpy(),
})
bywell["gain"] = bywell["base_rmse"] - bywell["candidate_rmse"]
bywell.sort_values("gain").to_csv(OUT_DIR / "bywell_gain.csv", index=False)

values = {
    "gain": float(base_metric["rmse"] - candidate_metric["rmse"]),
    "far_gain": float(base_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"]),
    "near_degradation": float(candidate_metric["near_0_500_rmse"] - base_metric["near_0_500_rmse"]),
    "p90_gain": float(base_metric["p90_well_rmse"] - candidate_metric["p90_well_rmse"]),
    "p95_gain": float(base_metric["p95_well_rmse"] - candidate_metric["p95_well_rmse"]),
    "p97p5_gain": float(base_metric["p97p5_well_rmse"] - candidate_metric["p97p5_well_rmse"]),
    "max_gain": float(base_metric["max_well_rmse"] - candidate_metric["max_well_rmse"]),
    "aligned_increment_vs_shuffle": float(control_metric["rmse"] - candidate_metric["rmse"]),
    "positive_well_fraction": float(np.mean(bywell["gain"] > 0.0)),
}
checks = {
    "fold1_gain_ge_0p05": values["gain"] >= 0.05,
    "fold1_far_gain_nonnegative": values["far_gain"] >= 0.0,
    "near_frozen_exact": abs(values["near_degradation"]) <= 1e-10,
    "p95_not_worse_by_gt_0p20": values["p95_gain"] >= -0.20,
    "aligned_beats_shuffle_ge_0p03": values["aligned_increment_vs_shuffle"] >= 0.03,
    "positive_well_fraction_ge_0p50": values["positive_well_fraction"] >= 0.50,
    "fallback_wells_le_5": fallback_wells <= 5,
}
strict_pass = bool(all(checks.values()))

output = frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, "exact_v76"]].copy()
output["consensus_center"] = candidate_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold1_consensus_center_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "untouched_Fold1_final_confirmation_of_frozen_V113_consensus",
    "selection_or_tuning_count": 0,
    "candidate_generation_uses_truth": False,
    "recipe": {
        "topk": TOPK,
        "aggregator": "uniform_mean",
        "alpha": ALPHA,
        "freeze_ft": FREEZE_FT,
        "ramp_ft": RAMP_FT,
        "cap_ft": CAP_FT,
    },
    "coverage": {"wells": int(frame["well"].nunique()), "fallback_wells": fallback_wells},
    "base": base_metric,
    "candidate": candidate_metric,
    "score_shuffled_control": control_metric,
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> retain consensus center and build fivefold OOF plus Kaggle deployment adapter"
        if strict_pass else
        "FAIL -> consensus does not survive untouched Fold1; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV114 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV114 WORST 20 WELL DELTAS")
print(bywell.sort_values("gain").head(20).to_string(index=False), flush=True)
print("\nV114 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
