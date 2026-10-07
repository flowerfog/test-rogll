"""V91b: read-only confirmation of V91's best sub-threshold recipe.

V91 found a small positive discovery signal (alpha=100, dose=0.25) that did
not meet its pre-registered 0.03-ft gate. Freeze that recipe and evaluate it
once on the remaining Fold3 wells. This is diagnostic only: it cannot turn
V91 into a strict pass without a fresh Fold4/5 outer confirmation.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_local_deformation_conditional_mean_v91b"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "rawdata_lattice_local_deformation_conditional_mean_v91_cell.py"

FROZEN_ALPHA = 100.0
FROZEN_DOSE = 0.25


def _bootstrap(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    marker = "train = pd.concat"
    if marker not in source:
        raise RuntimeError(f"Cannot identify execution boundary in {path}")
    module = types.ModuleType("v91b_defs")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
    return module


N = _bootstrap(SOURCE)
T = N.T

train = pd.concat([T.N.O._fold_frame(1), T.N.O._fold_frame(2)], ignore_index=True)
train, train_caches = T._eligible_frame_and_caches(train, 12)
x_train, y_train, train_wells, candidate_names = N._dataset(train, train_caches, True)

fold3 = T.N.O._fold_frame(3)
fold3, fold3_caches = T._eligible_frame_and_caches(fold3, 3)
x3, _, wells3, names3 = N._dataset(fold3, fold3_caches, False)
if names3 != candidate_names:
    raise RuntimeError("Train/Fold3 candidate mismatch")

ordered_wells = sorted(wells3)
discovery_wells = ordered_wells[:24]
confirmation_wells = ordered_wells[24:]

model = N.make_pipeline(N.StandardScaler(), N.Ridge(alpha=FROZEN_ALPHA))
model.fit(x_train, y_train)
knots = model.predict(x3)
knots[:, 0] = 0.0

prediction = N._predict_frame(fold3, fold3_caches, wells3, knots, FROZEN_DOSE)

# Roll only within the confirmation partition so its prediction distribution
# is matched without borrowing a curve from one of the discovery wells.
confirmation_set = set(confirmation_wells)
confirmation_indices = np.array(
    [index for index, well in enumerate(wells3) if well in confirmation_set], dtype=np.int64
)
rolled_knots = knots.copy()
rolled_knots[confirmation_indices] = np.roll(
    knots[confirmation_indices], max(1, len(confirmation_indices) // 5), axis=0
)
rolled_prediction = N._predict_frame(
    fold3, fold3_caches, wells3, rolled_knots, FROZEN_DOSE
)


def _evaluate(wells, stage):
    base = N._subset_metric(fold3, fold3["v76"], wells, "v76")
    candidate = N._subset_metric(fold3, prediction, wells, "conditional_mean")
    control = N._subset_metric(fold3, rolled_prediction, wells, "rolled_well_control")
    delta = N._delta(base, candidate)
    delta["aligned_increment_vs_rolled"] = control["rmse"] - candidate["rmse"]
    rows = [
        {"stage": stage, **base},
        {"stage": stage, **candidate},
        {"stage": stage, **control},
    ]
    return base, candidate, control, delta, rows


discovery = _evaluate(discovery_wells, "fold3_discovery_replay")
confirmation = _evaluate(confirmation_wells, "fold3_confirmation")
summary = pd.DataFrame(discovery[4] + confirmation[4])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

fold3_export = fold3[["id", "well", "row_idx", T.N.TRUTH, "md_since", "v76"]].copy()
fold3_export["conditional_mean"] = prediction.astype(np.float32)
fold3_export["rolled_well_control"] = rolled_prediction.astype(np.float32)
fold3_export.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False)

confirmation_delta = confirmation[3]
mechanism_pass = bool(
    confirmation_delta["gain"] > 0.0
    and confirmation_delta["far_gain"] > 0.0
    and confirmation_delta["near_degradation"] <= 1e-8
    and confirmation_delta["p95_gain"] >= -0.05
    and confirmation_delta["aligned_increment_vs_rolled"] >= 0.02
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "read_only_confirmation_of_V91_best_subthreshold_recipe",
    "selection_or_tuning_count": 0,
    "frozen_recipe": {"alpha": FROZEN_ALPHA, "dose": FROZEN_DOSE},
    "protocol": {
        "train_folds": [1, 2],
        "selection_fold": 3,
        "selection_wells": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "fold1_touched": True,
        "coverage_limited": True,
    },
    "discovery_replay": discovery[3],
    "confirmation": confirmation_delta,
    "checks": {
        "confirmation_gain_positive": confirmation_delta["gain"] > 0.0,
        "confirmation_far_gain_positive": confirmation_delta["far_gain"] > 0.0,
        "near_frozen_exact": confirmation_delta["near_degradation"] <= 1e-8,
        "p95_not_worse_by_gt_0p05": confirmation_delta["p95_gain"] >= -0.05,
        "aligned_beats_rolled_ge_0p02": confirmation_delta["aligned_increment_vs_rolled"] >= 0.02,
    },
    "mechanism_pass": mechanism_pass,
    "strict_pass": False,
    "strict_pass_reason": "V91 discovery missed its pre-registered 0.03-ft gain gate",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "DIAGNOSTIC POSITIVE -> rebuild Folds 4/5 and run a fresh outer protocol"
        if mechanism_pass
        else "FAIL -> close the complete V88 local-deformation consumption branch"
    ),
}
N._write_json(OUT_DIR / "verdict.json", verdict)
print("\nV91b MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV91b FINAL VERDICT")
print(json.dumps(N._clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
