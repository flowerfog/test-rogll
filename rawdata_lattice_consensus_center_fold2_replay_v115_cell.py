"""V115: frozen top33 consensus replay on Fold2 and threefold aggregation."""

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
EXPERIMENT = "rawdata_lattice_consensus_center_fold2_replay_v115"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V112_SOURCE = ROOT / "rawdata_lattice_topk_barycenter_fold2_v112_cell.py"
PANEL_DIR = OOF_ROOT / "rawdata_lattice_topk_barycenter_fold2_v112" / "fold2_panels"
FOLD1_PATH = OOF_ROOT / "rawdata_lattice_consensus_center_fold1_v114" / "fold1_consensus_center_oof.parquet"
FOLD3_PATH = OOF_ROOT / "rawdata_lattice_consensus_center_v113" / "fold3_consensus_center_oof.parquet"
FOLD = 2
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
    module = types.ModuleType("raw_lattice_v112_top33_fold2_definitions")
    module.__file__ = str(V112_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V112_SOURCE), "exec"), module.__dict__)
    return module


for required in (V112_SOURCE, PANEL_DIR, FOLD1_PATH, FOLD3_PATH):
    if not required.exists():
        raise FileNotFoundError(required)
R = _bootstrap_v112()
for required in (R.V81_SOURCE, R.MODEL_PATH, R.V70_PATH, R.V76_WEIGHTS):
    if not required.is_file():
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
    raise RuntimeError("Incomplete exact Fold2 V76 source")
if frame["well"].nunique() != 155:
    raise RuntimeError(f"Only {frame['well'].nunique()} Fold2 wells")
frame["exact_v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
    frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
)

candidate_parts, control_parts, report_rows = [], [], []
for number, well in enumerate(sorted(frame["well"].unique()), 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable")
    exact = local["exact_v76"].to_numpy(np.float64)
    candidate = exact.copy()
    control = exact.copy()
    report = {"well": well, "status": "fallback", "reason": ""}
    try:
        panel_path = PANEL_DIR / f"{well}.npz"
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
            raise RuntimeError(f"row count {len(local)}/{len(eval_rows)}")
        if np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
            raise RuntimeError("truth alignment")
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
        candidate = exact + taper * np.clip(ALPHA * dose * (center - coarse), -CAP_FT, CAP_FT)
        control = exact + taper * np.clip(ALPHA * dose * (shuffled_center - coarse), -CAP_FT, CAP_FT)
        report.update({
            "status": "built",
            "candidates": int(len(paths)),
            "unique_candidates": int(len(order)),
            "mean_abs_move": float(np.mean(np.abs(candidate - exact))),
        })
    except Exception as exc:
        report["reason"] = repr(exc)
    candidate_parts.append(candidate)
    control_parts.append(control)
    report_rows.append(report)
    if number % 10 == 0 or number == frame["well"].nunique():
        fallback = sum(row["status"] == "fallback" for row in report_rows)
        print(
            f"V115 Fold2 {number}/{frame['well'].nunique()} fallbacks={fallback} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

report = pd.DataFrame(report_rows)
report.to_csv(OUT_DIR / "build_report.csv", index=False)
fallback_wells = int(report["status"].eq("fallback").sum())
if fallback_wells > 5:
    raise RuntimeError(f"Too many Fold2 fallbacks: {fallback_wells}")

base_prediction = frame["exact_v76"].to_numpy(np.float64)
candidate_prediction = np.concatenate(candidate_parts)
control_prediction = np.concatenate(control_parts)
base2, base_bywell = R._metric(frame, base_prediction, "exact_v76")
candidate2, candidate_bywell = R._metric(frame, candidate_prediction, "consensus_center")
control2, _ = R._metric(frame, control_prediction, "score_shuffled_control")

fold2_output = frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, "exact_v76"]].copy()
fold2_output["consensus_center"] = candidate_prediction.astype(np.float32)
fold2_output.to_parquet(OUT_DIR / "fold2_consensus_center_oof.parquet", index=False, compression="zstd")

fold1 = pd.read_parquet(FOLD1_PATH)
fold3 = pd.read_parquet(FOLD3_PATH)
required_columns = {"id", "well", "row_idx", "fold", TRUTH, "md_since", "exact_v76", "consensus_center"}
for label, local in (("fold1", fold1), ("fold2", fold2_output), ("fold3", fold3)):
    missing = sorted(required_columns.difference(local.columns))
    if missing:
        raise RuntimeError(f"{label} missing {missing}")
combined = pd.concat(
    [fold1[list(required_columns)], fold2_output[list(required_columns)], fold3[list(required_columns)]],
    ignore_index=True,
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
if combined["id"].duplicated().any() or combined["well"].nunique() != 464:
    raise RuntimeError("Invalid combined Fold123 consensus OOF")

summary_rows = []
fold_gains = {}
for fold in (1, 2, 3, 0):
    local = combined if fold == 0 else combined[combined["fold"].eq(fold)]
    base_metric, _ = R._metric(local, local["exact_v76"], "exact_v76")
    candidate_metric, _ = R._metric(local, local["consensus_center"], "consensus_center")
    summary_rows.extend([{"fold": fold, **base_metric}, {"fold": fold, **candidate_metric}])
    fold_gains[str(fold)] = float(base_metric["rmse"] - candidate_metric["rmse"])
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "threefold_summary.csv", index=False)
combined.to_parquet(OUT_DIR / "fold123_consensus_center_oof.parquet", index=False, compression="zstd")

values = {
    "fold2_gain": float(base2["rmse"] - candidate2["rmse"]),
    "fold2_far_gain": float(base2["far_gt2000_rmse"] - candidate2["far_gt2000_rmse"]),
    "fold2_p95_gain": float(base2["p95_well_rmse"] - candidate2["p95_well_rmse"]),
    "fold2_aligned_increment_vs_shuffle": float(control2["rmse"] - candidate2["rmse"]),
    "fold2_positive_well_fraction": float(np.mean(
        base_bywell.to_numpy() > candidate_bywell.reindex(base_bywell.index).to_numpy()
    )),
    "threefold_gain": fold_gains["0"],
}
checks = {
    "fold1_gain_positive": fold_gains["1"] > 0.0,
    "fold2_gain_positive": fold_gains["2"] > 0.0,
    "fold3_gain_positive": fold_gains["3"] > 0.0,
    "threefold_gain_ge_0p05": values["threefold_gain"] >= 0.05,
    "fold2_far_gain_nonnegative": values["fold2_far_gain"] >= 0.0,
    "fold2_p95_not_worse_by_gt_0p20": values["fold2_p95_gain"] >= -0.20,
    "fallback_wells_le_5": fallback_wells <= 5,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "frozen_Fold2_replay_and_Fold123_matched_aggregation",
    "selection_or_tuning_count": 0,
    "recipe": {
        "topk": TOPK,
        "aggregator": "uniform_mean",
        "alpha": ALPHA,
        "freeze_ft": FREEZE_FT,
        "ramp_ft": RAMP_FT,
        "cap_ft": CAP_FT,
    },
    "fold2": {"base": base2, "candidate": candidate2, "control": control2},
    "fold_gains": fold_gains,
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build the all-fold OOF lineage and Kaggle deployment adapter"
        if strict_pass else
        "FAIL -> retain V76 despite positive Fold1/3 evidence"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV115 FOLD2 MATCHED SUMMARY")
print(pd.DataFrame([base2, candidate2, control2]).to_string(index=False), flush=True)
print("\nV115 THREEFOLD SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV115 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
