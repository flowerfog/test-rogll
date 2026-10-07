"""M342-AI1: staged five-fold audit of the frozen V116 tail magnitude.

The complete V144 OOF contains the exact cross-fit V116 consensus_center and
its V77/exact_v76 parent.  This read-only audit scales only that frozen move:

    candidate = exact_v76 + scale * (consensus_center - exact_v76)

Fold4/5 select once, Fold3 confirms, and Fold2/Fold1 are zero-retuning support.
No model is trained and submission.csv is never touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
EXPERIMENT = "main342_v116_tail_scale_audit_ai1"
OUT = OOF / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = OOF / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet"
SCALES = (0.50, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.50)
DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
SUPPORT_FOLDS = (2, 1)
BASE = "consensus_center"
PARENT = "exact_v76"
TRUTH = "tvt_true"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame[TRUTH].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    bywell = pd.DataFrame({"well": frame["well"].astype(str), "sq": error * error}).groupby("well")["sq"].mean().pow(0.5)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] * error[far]))) if far.any() else None,
    }


def _prediction(frame: pd.DataFrame, scale: float) -> np.ndarray:
    parent = frame[PARENT].to_numpy(np.float64)
    move = frame[BASE].to_numpy(np.float64) - parent
    return parent + float(scale) * move


def _evaluate(frame: pd.DataFrame, scale: float, fold: int) -> dict:
    local = frame[frame["fold"].eq(fold)].copy()
    base = _metric(local, local[BASE].to_numpy(np.float64), "v116_scale1")
    candidate = _metric(local, _prediction(local, scale), f"v116_scale_{scale:g}")
    return {
        "fold": int(fold),
        "scale": float(scale),
        "rows": candidate["rows"],
        "wells": candidate["wells"],
        "base_rmse": base["rmse"],
        "rmse": candidate["rmse"],
        "gain": float(base["rmse"] - candidate["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        "p95_change": float(candidate["p95_well_rmse"] - base["p95_well_rmse"]),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE, columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, BASE])
frame["well"] = frame["well"].astype(str)
coverage = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if len(frame) != 3_783_989 or sum(coverage.values()) != 773 or min(coverage.values()) < 154:
    raise RuntimeError(f"Incomplete V116 OOF coverage: rows={len(frame)} folds={coverage}")
if not np.isfinite(frame[[TRUTH, "md_since", PARENT, BASE]].to_numpy(np.float64)).all():
    raise RuntimeError("Non-finite V116 OOF values")

parity = _prediction(frame, 1.0)
reproduction_gap = float(np.max(np.abs(parity - frame[BASE].to_numpy(np.float64))))
if reproduction_gap > 1e-10:
    raise RuntimeError(f"V116 scale1 reproduction failed: {reproduction_gap}")

discovery_rows = []
for scale in SCALES:
    folds = [_evaluate(frame, scale, fold) for fold in DISCOVERY_FOLDS]
    local = frame[frame["fold"].isin(DISCOVERY_FOLDS)]
    base = _metric(local, local[BASE].to_numpy(np.float64), "v116_scale1")
    candidate = _metric(local, _prediction(local, scale), f"v116_scale_{scale:g}")
    discovery_rows.append({
        "scale": scale,
        "pooled_rmse": candidate["rmse"],
        "pooled_gain": base["rmse"] - candidate["rmse"],
        "minimum_fold_gain": min(item["gain"] for item in folds),
        "minimum_fold_far_gain": min(item["far_gain"] for item in folds),
        "maximum_fold_p95_change": max(item["p95_change"] for item in folds),
        "fold4_gain": folds[0]["gain"],
        "fold5_gain": folds[1]["gain"],
    })
discovery = pd.DataFrame(discovery_rows).sort_values(["pooled_rmse", "scale"], kind="stable")
discovery.to_csv(OUT / "discovery_scan.csv", index=False)
safe = discovery[
    (discovery["minimum_fold_gain"] >= 0.0)
    & (discovery["minimum_fold_far_gain"] >= 0.0)
    & (discovery["maximum_fold_p95_change"] <= 0.20)
]
selected = (safe if len(safe) else discovery).iloc[0]
selected_scale = float(selected["scale"])

confirmation = _evaluate(frame, selected_scale, CONFIRMATION_FOLD)
mechanism_pass = bool(
    selected_scale != 1.0
    and selected["pooled_gain"] >= 0.01
    and selected["minimum_fold_gain"] >= 0.0
    and confirmation["gain"] >= 0.01
    and confirmation["far_gain"] >= 0.0
    and confirmation["p95_change"] <= 0.20
)
support = [_evaluate(frame, selected_scale, fold) for fold in SUPPORT_FOLDS] if mechanism_pass else []

checks = {
    "scale1_reproduces_v116": reproduction_gap <= 1e-10,
    "discovery_has_safe_scale": bool(len(safe)),
    "selected_scale_differs_from_1": selected_scale != 1.0,
    "discovery_pooled_gain_ge_0p01": float(selected["pooled_gain"]) >= 0.01,
    "confirmation_gain_ge_0p01": confirmation["gain"] >= 0.01,
    "confirmation_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
    "confirmation_p95_change_le_0p20": confirmation["p95_change"] <= 0.20,
    "support_folds_nonnegative": bool(support) and all(item["gain"] >= 0.0 and item["far_gain"] >= 0.0 for item in support),
}
strict_pass = bool(mechanism_pass and checks["support_folds_nonnegative"])
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold45_scale_selection_fold3_confirmation_fold2_fold1_support",
    "selection_or_tuning_count": 1,
    "parent": "exact fivefold V116 consensus_center",
    "protocol": {"discovery_folds": DISCOVERY_FOLDS, "confirmation_fold": CONFIRMATION_FOLD, "support_folds": SUPPORT_FOLDS},
    "coverage": coverage,
    "scale_menu": SCALES,
    "selected": selected.to_dict(),
    "confirmation": confirmation,
    "support": support,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> build one V116-scaled Kaggle candidate" if strict_pass else "FAIL -> retain V116 exactly at scale 1.0",
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-AI1 DISCOVERY")
print(discovery.to_string(index=False), flush=True)
print("\nM342-AI1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
