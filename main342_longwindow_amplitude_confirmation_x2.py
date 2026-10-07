"""M342-X2: frozen 256-row versus 128-row raw-GR observability confirmation.

X1 selected on the first 24 Fold3 wells and showed on Fold2 that the apparent
multiscale gain was largely explained by the 256-row score itself. This audit
uses only the remaining Fold3 wells, fixes both scales in advance, and repeats
the matched-range typewell shuffle attribution. No model is trained.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
X1_SOURCE = ROOT / "main342_multiscale_rank_consistency_x1.py"
OUT = ROOT / "oof_exports" / "main342_longwindow_amplitude_confirmation_x2"
OUT.mkdir(parents=True, exist_ok=True)

FOLD = 3
X1_DISCOVERY_WELLS = 24
BASELINE = "single_128"
CANDIDATE = "single_256"


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


def _load_x1_definitions():
    source = X1_SOURCE.read_text(encoding="utf-8")
    marker = "fold3_wells = sorted("
    if marker not in source:
        raise RuntimeError("X1 execution marker not found")
    name = "m342_x2_x1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(X1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(X1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


X = _load_x1_definitions()
v66 = X["v66"]
DATA_ROOT = X["DATA_ROOT"]
fold_by_well = X["fold_by_well"]
all_fold_wells = sorted(well for well, fold in fold_by_well.items() if fold == FOLD)
evaluation_wells = all_fold_wells[X1_DISCOVERY_WELLS:]
if len(evaluation_wells) < 120:
    raise RuntimeError(f"Only {len(evaluation_wells)} clean Fold3 wells remain")

aligned_records, failures_a = X["_evaluate"](evaluation_wells, "aligned")

centres = {}
for well in evaluation_wells:
    table = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT"])
    centres[well] = float(pd.to_numeric(table["TVT"], errors="coerce").median())
ordered = sorted(evaluation_wells, key=lambda well: (centres[well], well))
shuffle_map = dict(zip(ordered, ordered[1:] + ordered[:1]))
shuffle_records, failures_b = X["_evaluate"](
    evaluation_wells, "typewell_shuffle", shuffle_map
)

summary = X["_summary"](
    pd.concat([aligned_records, shuffle_records], ignore_index=True)
)


def _row(control, config):
    local = summary[
        summary["control"].eq(control) & summary["config"].eq(config)
    ]
    if len(local) != 1:
        raise RuntimeError(f"Missing X2 result: {control}/{config}")
    return local.iloc[0]


aligned_base = _row("aligned", BASELINE)
aligned_candidate = _row("aligned", CANDIDATE)
shuffle_base = _row("typewell_shuffle", BASELINE)
shuffle_candidate = _row("typewell_shuffle", CANDIDATE)

top32_gain = float(aligned_candidate["top32_recall"] - aligned_base["top32_recall"])
top8_gain = float(aligned_candidate["top8_recall"] - aligned_base["top8_recall"])
argmax_gain = float(aligned_base["argmax_rmse"] - aligned_candidate["argmax_rmse"])
shuffle_top32_gain = float(
    shuffle_candidate["top32_recall"] - shuffle_base["top32_recall"]
)
aligned_increment = top32_gain - shuffle_top32_gain

checks = {
    "aligned_wells_ge_120": int(aligned_candidate["wells"]) >= 120,
    "top32_gain_ge_0p02": top32_gain >= 0.02,
    "top8_gain_ge_0p015": top8_gain >= 0.015,
    "aligned_increment_vs_shuffle_ge_0p01": aligned_increment >= 0.01,
    "argmax_rmse_not_worse": argmax_gain >= 0.0,
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "main342_longwindow_amplitude_confirmation_x2",
    "stage": "frozen_256_vs_128_remaining_fold3_confirmation",
    "model_training": False,
    "selection_or_tuning_count": 0,
    "protocol": {
        "confirmation_fold": FOLD,
        "excluded_x1_discovery_wells": X1_DISCOVERY_WELLS,
        "evaluation_wells_requested": len(evaluation_wells),
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "untouched_fold": 1,
    },
    "confirmation": {
        "baseline": aligned_base.to_dict(),
        "candidate": aligned_candidate.to_dict(),
        "shuffle_baseline": shuffle_base.to_dict(),
        "shuffle_candidate": shuffle_candidate.to_dict(),
        "top32_gain": top32_gain,
        "top8_gain": top8_gain,
        "argmax_rmse_gain": argmax_gain,
        "shuffle_top32_gain": shuffle_top32_gain,
        "aligned_increment_vs_shuffle": aligned_increment,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "failures": {"aligned": len(failures_a), "shuffle": len(failures_b)},
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> add frozen 256-row amplitude evidence to the path decoder"
        if strict_pass else
        "FAIL -> long context is not transferable; close raw-GR scale expansion"
    ),
}

summary.to_csv(OUT / "confirmation_summary.csv", index=False)
(OUT / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nM342-X2 CONFIRMATION")
print(summary.to_string(index=False), flush=True)
print("\nM342-X2 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
