"""Confirm the frozen V59 rolled-GR bank reranker on Fold 2.

V59's aligned recurrence improved Fold-3 confirmation, but its predeclared
rolled control improved more.  The geologic attribution therefore failed.
The rolled transform is nevertheless legal and deterministic.  This cell
treats it as a newly discovered endpoint, freezes V59's 20% roll and dose 0.1,
and performs one read-only Fold-2 confirmation.  It performs no search and
does not touch Fold 1.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "future_gr_rolled_bank_confirmation_v59b"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = next(
    (
        path
        for path in (
            ROOT / "future_gr_recurrence_bank_reranker_v59_cell.py",
            Path.cwd() / "future_gr_recurrence_bank_reranker_v59_cell.py",
        )
        if path.is_file()
    ),
    None,
)
if SOURCE is None:
    raise FileNotFoundError("Upload future_gr_recurrence_bank_reranker_v59_cell.py")

FROZEN_DOSE = 0.1
APPLICATION_FOLD = 2


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType("future_gr_v59_definitions_for_v59b")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
    return module


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


started = time.time()
N = _bootstrap(SOURCE)
V = N.V
wells = [
    path.name.split("__", 1)[0]
    for path in sorted((N.DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = V["_stable_folds"](wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
needed = [well for well in wells if fold_by_well[well] in (*N.PRIOR_FOLDS, APPLICATION_FOLD)]
records, failures = {}, []
for number, well in enumerate(needed, 1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed):
        print(f"V59b load {number}/{len(needed)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

prior_records = [
    records[well]
    for well in needed
    if well in records and fold_by_well[well] in N.PRIOR_FOLDS
]
application_records = [
    records[well]
    for well in needed
    if well in records and fold_by_well[well] == APPLICATION_FOLD
]
prior = V["_fit_prior"](prior_records, N.N.SEGMENT_LENGTH_FT)
print(
    "V59b CONFIG",
    {
        "prior_wells": len(prior_records),
        "fold2_wells": len(application_records),
        "dose": FROZEN_DOSE,
        "roll": "max(n_eval_stride2//5, 1)",
        "search_count": 0,
    },
    flush=True,
)

cache = N._build(application_records, prior, "fold2_frozen_confirmation")
baseline = N._evaluate(cache, 0.0, "aligned", "baseline")
aligned = N._evaluate(cache, FROZEN_DOSE, "aligned", "aligned")
rolled = N._evaluate(cache, FROZEN_DOSE, "rolled", "rolled")
detail = pd.concat([baseline, aligned, rolled], ignore_index=True)
detail.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
summary = pd.DataFrame(
    [
        {"variant": "baseline", **N._summary(baseline)},
        {"variant": "aligned", **N._summary(aligned)},
        {"variant": "rolled", **N._summary(rolled)},
    ]
)
summary.to_csv(OUT_DIR / "fold2_summary.csv", index=False)
print("\nV59b FOLD2 CONFIRMATION")
print(summary.to_string(index=False), flush=True)
by_variant = {row["variant"]: row for row in summary.to_dict("records")}
base = by_variant["baseline"]
candidate = by_variant["rolled"]
aligned_result = by_variant["aligned"]
gain = float(base["rmse"] - candidate["rmse"])
far_gain = float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"])
checks = {
    "rolled_gain_ge_0p20": bool(gain >= 0.20),
    "rolled_far_gain_positive": bool(far_gain > 0.0),
    "path_change_fraction_ge_0p10": bool(candidate["selected_path_change_fraction"] >= 0.10),
    "coverage_ge_0p95": bool(candidate["mean_available_fraction"] >= 0.95),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_single_confirmation_of_Fold3_discovered_rolled_GR_endpoint",
    "selection_or_tuning_count": 0,
    "candidate_generation_uses_truth": False,
    "truth_used_for_metrics_only": True,
    "legal_inputs_only": True,
    "fold1_touched": False,
    "frozen_recipe": {
        "arm": N.ARM,
        "bin_width_ft": N.BIN_WIDTH_FT,
        "min_separation_ft": N.MIN_SEPARATION_FT,
        "dose": FROZEN_DOSE,
        "gr_transform": "circular_roll_by_20_percent_of_stride2_eval_rows",
    },
    "fold2": {
        "baseline": base,
        "aligned_reference": aligned_result,
        "rolled_candidate": candidate,
        "gain": gain,
        "far_gain": far_gain,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> confirm once on untouched Fold1, then assess cross-evidence with 6.377"
        if strict_pass
        else "FAIL -> rolled gain was Fold3-specific; close the recurrence bank branch"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nV59b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
