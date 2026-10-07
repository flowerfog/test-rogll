"""V88: oracle investment gate for local deformations of the V70 coarse path.

This cell does not propose a deployable selector. It asks whether the frozen
candidate support is missing useful paths near the selected V70 trajectory.
All deformations are exactly zero through 1000 ft from the anchor. Truth is
used only to select the per-well oracle candidate and compute metrics.
Fold 3 is discovery, Fold 2 is confirmation, and Fold 1 is untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_local_deformation_oracle_v88"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
FOLDS = (3, 2)
FREEZE_FT = 1000.0
RAMP_FT = 250.0
MID_KNOT_FT = 2000.0
MID_OFFSETS = (-16.0, -8.0, 0.0, 8.0, 16.0)
END_OFFSETS = (-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0)
STEP_ONSETS = (1250.0, 1750.0, 2250.0)
STEP_OFFSETS = (-24.0, -12.0, 12.0, 24.0)


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


def _smoothstep(values):
    values = np.clip(np.asarray(values, np.float64), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _tail_taper(distance):
    return _smoothstep((distance - FREEZE_FT) / RAMP_FT)


def _trend_deformation(distance, mid_offset, end_offset):
    distance = np.asarray(distance, np.float64)
    maximum = max(float(np.max(distance)), MID_KNOT_FT + 1.0)
    raw = np.interp(
        distance,
        [FREEZE_FT, MID_KNOT_FT, maximum],
        [0.0, float(mid_offset), float(end_offset)],
    )
    return _tail_taper(distance) * raw


def _step_deformation(distance, onset, offset):
    distance = np.asarray(distance, np.float64)
    return float(offset) * _smoothstep((distance - float(onset)) / RAMP_FT)


def _candidate_deformations(distance):
    names = ["identity"]
    values = [np.zeros(len(distance), dtype=np.float64)]
    for middle in MID_OFFSETS:
        for endpoint in END_OFFSETS:
            if middle == 0.0 and endpoint == 0.0:
                continue
            names.append(f"trend_m{middle:+g}_e{endpoint:+g}")
            values.append(_trend_deformation(distance, middle, endpoint))
    for onset in STEP_ONSETS:
        for offset in STEP_OFFSETS:
            names.append(f"step_s{onset:g}_o{offset:+g}")
            values.append(_step_deformation(distance, onset, offset))
    matrix = np.vstack(values)
    if np.max(np.abs(matrix[:, distance <= FREEZE_FT])) > 1e-12:
        raise RuntimeError("Prefix deformation is not exactly frozen")
    return names, matrix


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    error = prediction - truth
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    output = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.sum(error2)),
        "rmse": float(np.sqrt(np.mean(error2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    strata = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in strata.items():
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


def _fold_frame(fold):
    frame = pd.read_parquet(
        V70_PATH,
        columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
        filters=[("fold", "=", int(fold))],
    ).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    weights = pd.read_parquet(V76_WEIGHTS)
    weights = weights[weights["fold"].eq(fold)][["well", "ensemble_weight"]]
    weight_map = dict(zip(weights["well"].astype(str), weights["ensemble_weight"]))
    frame["dose"] = frame["well"].astype(str).map(weight_map)
    if frame["dose"].isna().any() or frame.empty:
        raise RuntimeError(f"Fold{fold}: incomplete V76 alignment")
    frame["v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
        frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
    )
    return frame


def _oracle_fold(fold):
    frame = _fold_frame(fold)
    oracle = np.empty(len(frame), dtype=np.float64)
    rowwise = np.empty(len(frame), dtype=np.float64)
    selected_rows = []
    for number, (well, local) in enumerate(frame.groupby("well", sort=False), 1):
        indices = local.index.to_numpy(np.int64)
        distance = local["md_since"].to_numpy(np.float64)
        parent = local[PARENT].to_numpy(np.float64)
        coarse = local[COARSE].to_numpy(np.float64)
        truth = local[TRUTH].to_numpy(np.float64)
        dose = float(local["dose"].iloc[0])
        names, deformation = _candidate_deformations(distance)
        candidates = parent[None, :] + dose * (
            coarse[None, :] + deformation - parent[None, :]
        )
        sse = np.sum((candidates - truth[None, :]) ** 2, axis=1)
        best = int(np.argmin(sse))
        oracle[indices] = candidates[best]
        rowwise[indices] = candidates[np.argmin((candidates - truth[None, :]) ** 2, axis=0), np.arange(len(local))]
        selected_rows.append({
            "fold": int(fold),
            "well": str(well),
            "rows": int(len(local)),
            "dose": dose,
            "selected_candidate": names[best],
            "selected_index": best,
            "baseline_rmse": float(np.sqrt(sse[0] / len(local))),
            "oracle_rmse": float(np.sqrt(sse[best] / len(local))),
            "oracle_gain": float(np.sqrt(sse[0] / len(local)) - np.sqrt(sse[best] / len(local))),
        })
        if number % 50 == 0 or number == frame["well"].nunique():
            print(f"V88 Fold{fold} oracle {number}/{frame['well'].nunique()}", flush=True)
    selected = pd.DataFrame(selected_rows)
    metrics = {
        "baseline": _metric(frame, frame["v76"], "v76"),
        "perwell_oracle": _metric(frame, oracle, "local_deformation_perwell_oracle"),
        "rowwise_oracle": _metric(frame, rowwise, "local_deformation_rowwise_oracle"),
    }
    output = frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", "v76"]].copy()
    output["local_deformation_oracle"] = oracle.astype(np.float32)
    return metrics, selected, output


for required in (V70_PATH, V76_WEIGHTS):
    if not required.is_file():
        raise FileNotFoundError(required)

all_metrics = {}
summary_rows = []
selected_parts = []
oof_parts = []
for fold in FOLDS:
    metrics, selected, output = _oracle_fold(fold)
    all_metrics[str(fold)] = metrics
    selected_parts.append(selected)
    oof_parts.append(output)
    for endpoint, metric in metrics.items():
        summary_rows.append({"fold": int(fold), **metric})

summary = pd.DataFrame(summary_rows)
selected = pd.concat(selected_parts, ignore_index=True)
summary.to_csv(OUT_DIR / "oracle_summary.csv", index=False)
selected.to_csv(OUT_DIR / "selected_candidates_bywell.csv", index=False)
pd.concat(oof_parts, ignore_index=True).to_parquet(
    OUT_DIR / "local_deformation_oracle_oof.parquet", index=False, compression="zstd"
)

gains = {
    str(fold): float(
        all_metrics[str(fold)]["baseline"]["rmse"]
        - all_metrics[str(fold)]["perwell_oracle"]["rmse"]
    )
    for fold in FOLDS
}
far_gains = {
    str(fold): float(
        all_metrics[str(fold)]["baseline"]["far_gt2000_rmse"]
        - all_metrics[str(fold)]["perwell_oracle"]["far_gt2000_rmse"]
    )
    for fold in FOLDS
}
checks = {
    "fold3_oracle_gain_ge_0p30": gains["3"] >= 0.30,
    "fold2_oracle_gain_ge_0p30": gains["2"] >= 0.30,
    "fold3_far_gain_ge_0p30": far_gains["3"] >= 0.30,
    "fold2_far_gain_ge_0p30": far_gains["2"] >= 0.30,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "anchored_local_candidate_support_oracle_Fold3_discovery_Fold2_confirmation",
    "deployable": False,
    "truth_usage": "per-well and rowwise oracle selection plus metrics only",
    "candidate_family": {
        "prefix_frozen_ft": FREEZE_FT,
        "ramp_ft": RAMP_FT,
        "mid_knot_ft": MID_KNOT_FT,
        "mid_offsets": list(MID_OFFSETS),
        "end_offsets": list(END_OFFSETS),
        "step_onsets": list(STEP_ONSETS),
        "step_offsets": list(STEP_OFFSETS),
        "candidate_count": 1 + len(MID_OFFSETS) * len(END_OFFSETS) - 1 + len(STEP_ONSETS) * len(STEP_OFFSETS),
    },
    "metrics": all_metrics,
    "oracle_gain": gains,
    "far_oracle_gain": far_gains,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> build V89 label-free GR/physics scorer for the frozen deformation family"
        if all(checks.values())
        else "FAIL -> local support is not the bottleneck; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV88 ORACLE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV88 SELECTED FAMILY COUNTS")
print(selected["selected_candidate"].value_counts().head(20).to_string(), flush=True)
print("\nV88 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
