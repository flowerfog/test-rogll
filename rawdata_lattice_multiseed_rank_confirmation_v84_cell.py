"""Independent confirmation of V83b's borderline rank-mean-3 result.

V83b missed its Fold-3 near-distance guard by 0.000295 ft while improving all
primary, far, tail, and member-variance metrics.  This cell does not relabel
that original gate as passed.  It freezes rank_mean3 and uses untouched Fold-2
for one confirmation.  Fold-1 is evaluated only if Fold-2 passes.
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
EXPERIMENT = "rawdata_lattice_multiseed_rank_confirmation_v84"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V83_SOURCE = ROOT / "rawdata_lattice_multiseed_rank_ensemble_v83_cell.py"
V83_DIR = OOF_ROOT / "rawdata_lattice_multiseed_rank_ensemble_v83b"
V83_VERDICT = V83_DIR / "verdict.json"
V83_SUMMARY = V83_DIR / "matched_summary.csv"
FROZEN_AGGREGATION = "rank_mean3"


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


def _bootstrap_v83():
    if not V83_SOURCE.is_file():
        raise FileNotFoundError(V83_SOURCE)
    source = V83_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V83 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"Bootstrap marker missing from {V83_SOURCE}; upload the updated V83 file")
    module = types.ModuleType("raw_lattice_v83_confirmation_definitions")
    module.__file__ = str(V83_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V83_SOURCE), "exec"), module.__dict__)
    return module


if not V83_VERDICT.is_file() or not V83_SUMMARY.is_file():
    raise FileNotFoundError("Run corrected V83b before V84")

v83_verdict = json.loads(V83_VERDICT.read_text(encoding="utf-8"))
v83_summary = pd.read_csv(V83_SUMMARY)
if v83_verdict.get("selected_aggregation") != FROZEN_AGGREGATION:
    raise RuntimeError(f"Unexpected V83 aggregation: {v83_verdict.get('selected_aggregation')}")
parity = v83_verdict.get("v82_seed1_parity") or {}
if float(parity.get("absolute_difference", np.inf)) > 1e-6:
    raise RuntimeError(f"V83 seed-1 parity is not valid: {parity}")
fold3 = v83_verdict["fold3_selection"]
if not (
    float(fold3["gain"]) >= 0.03
    and float(fold3["far_gain"]) >= 0.0
    and float(fold3["gain_vs_median_member"]) >= 0.02
    and 0.03 < float(fold3["near_degradation"]) <= 0.031
):
    raise RuntimeError(f"V83 is not the registered borderline case: {fold3}")

T = _bootstrap_v83()
N = T.N
fold_by_well = T.fold_by_well
available = set(T.available)
fold2_wells = sorted(well for well, fold in fold_by_well.items() if fold == 2)
fold1_wells = sorted(well for well, fold in fold_by_well.items() if fold == 1)

print(
    "V84 CONFIG",
    {
        "frozen_aggregation": FROZEN_AGGREGATION,
        "fold3_original_gate_pass": False,
        "fold3_near_excess_ft": float(fold3["near_degradation"]) - 0.03,
        "fold2_wells": len(fold2_wells),
        "fold1_wells": len(fold1_wells),
    },
    flush=True,
)

# No aggregation search is performed on Fold-2.
train345 = sorted(
    well for well, fold in fold_by_well.items() if fold in (3, 4, 5) and well in available
)
models2, _ = T._fit_models(train345, "v84_fold345")
frame2, reports2 = T._evaluate(fold2_wells, models2, "v84_fold2_confirmation")
table2 = pd.DataFrame(T._summary(frame2, "fold2_confirmation"))
result2 = T._gains(table2, FROZEN_AGGREGATION)
confirmation_pass = bool(
    result2["gain"] >= 0.03
    and result2["far_gain"] >= 0.0
    and result2["near_degradation"] <= 0.03
    and result2["gain_vs_median_member"] >= 0.01
)
frame2.to_parquet(OUT_DIR / "fold2_matched_oof.parquet", index=False)

rows1, reports1, result1 = [], [], None
supplement2 = []
final_pass = False
if confirmation_pass:
    missing2 = [well for well in fold2_wells if well not in available]
    for number, well in enumerate(missing2, 1):
        report = N._build_supplement_panel(well)
        supplement2.append(report)
        if report["status"] != "failed":
            available.add(well)
        if number % 10 == 0 or number == len(missing2):
            print(f"V84 supplement Fold2 {number}/{len(missing2)}", flush=True)
    train2345 = sorted(
        well for well, fold in fold_by_well.items() if fold in (2, 3, 4, 5) and well in available
    )
    models1, _ = T._fit_models(train2345, "v84_fold2345")
    frame1, reports1 = T._evaluate(fold1_wells, models1, "v84_fold1_final")
    rows1 = T._summary(frame1, "fold1_final")
    table1 = pd.DataFrame(rows1)
    result1 = T._gains(table1, FROZEN_AGGREGATION)
    final_pass = bool(
        result1["gain"] >= 0.03
        and result1["far_gain"] >= 0.0
        and result1["near_degradation"] <= 0.03
        and result1["gain_vs_median_member"] >= 0.01
    )
    frame1.to_parquet(OUT_DIR / "fold1_matched_oof.parquet", index=False)

summary = pd.concat([v83_summary, table2, pd.DataFrame(rows1)], ignore_index=True)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(reports2 + reports1).to_json(
    OUT_DIR / "evaluation_report.jsonl", orient="records", lines=True
)
pd.DataFrame(supplement2).to_csv(OUT_DIR / "fold2_supplement_report.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "V83_borderline_Fold2_independent_confirmation_Fold1_final",
    "frozen_aggregation": FROZEN_AGGREGATION,
    "protocol_amendment": {
        "original_fold3_gate_pass": False,
        "reason_for_confirmation": "near guard missed by less than 0.001 ft while parent-relative near RMSE improved",
        "aggregation_search_on_fold2": 0,
        "fold1_touched_only_after_fold2_pass": True,
    },
    "fold3_selection": fold3,
    "fold2_confirmation": {**result2, "pass": confirmation_pass},
    "fold1_final": None if result1 is None else {**result1, "pass": final_pass},
    "fold1_touched": bool(confirmation_pass),
    "evidence_pass": bool(confirmation_pass and final_pass),
    "strict_pass_under_original_v83_preregistration": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "INDEPENDENT PASS -> rebuild the adaptive-dose endpoint with frozen rank_mean3"
        if confirmation_pass and final_pass
        else "FAIL -> retain V76; close multi-seed selector averaging"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV84 MATCHED SUMMARY")
print(pd.concat([table2, pd.DataFrame(rows1)], ignore_index=True).to_string(index=False), flush=True)
print("\nV84 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
