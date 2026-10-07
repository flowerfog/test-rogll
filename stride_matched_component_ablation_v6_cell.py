"""Matched Fold-2 component ablations for the repaired STRIDE objective.

V5 materially repaired sequence scoring (22.41 -> 13.14 ft) but did not
identify which disclosed components now have the expected direction.  Keep
the V5 recipe frozen and run only three matched controls on the same untouched
Fold-2 wells.  The already-saved V5 full result is the reference, so it is not
decoded again and no row-level control files are written.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_matched_component_ablation_v6"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "stride_fullsequence_beam_v4_cell.py",
    Path.cwd() / "stride_fullsequence_beam_v4_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
V3_VERDICT = OOF_ROOT / "stride_figure1_posterior_parity_v3" / "verdict.json"
V5_VERDICT = OOF_ROOT / "stride_sequence_scale_repair_v5" / "verdict.json"
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload the updated stride_fullsequence_beam_v4_cell.py")
for required in (V3_VERDICT, V5_VERDICT):
    if not required.exists():
        raise FileNotFoundError(required)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "v3" in names:
                break
        body.append(node)
    name = "stride_v4_repaired_definitions_for_v6"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


V = _bootstrap(SOURCE_PATH)
v3 = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
v5 = json.loads(V5_VERDICT.read_text(encoding="utf-8"))
prior = v3["selected"]["prior"]
temperature = float(v3["selected"]["emission_temperature"])
recipe = v5["selected_recipe"]
full = v5["fold2"]

folds = pd.read_csv(V.V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
application_wells = sorted(folds.loc[folds["fold"].eq(V.V.APPLICATION_FOLD), "well"])
if len(application_wells) != 155:
    raise RuntimeError(f"Unexpected Fold2 size: {len(application_wells)}")

records = []
for number, well in enumerate(application_wells, start=1):
    records.append(V.load_record(well))
    if number % 50 == 0 or number == len(application_wells):
        print(f"STRIDE v6 load Fold2 {number}/{len(application_wells)}", flush=True)

V.BEAM_SIZE = 48
controls = (
    ("raw_reference", "raw_reference"),
    ("gaussian", "gaussian"),
    ("no_persistence", "no_persistence"),
)
started = time.time()
summaries = []

for label, mode in controls:
    bywell = []
    for number, record in enumerate(records, start=1):
        prediction, objective, nodes = V._decode(
            record,
            prior,
            temperature,
            mode,
            effective_spacing_ft=float(recipe["effective_spacing_ft"]),
            persistence_weight=float(recipe["persistence_weight"]),
            lambda_override=float(recipe["lambda_previous_dip"]),
        )
        bywell.append(V._metric_row(record, prediction, label, objective, nodes))
        if number % 25 == 0 or number == len(records):
            print(f"STRIDE v6 {label} {number}/{len(records)}", flush=True)
    frame = pd.DataFrame(bywell)
    frame.to_csv(OUT_DIR / f"fold2_{label}_bywell.csv", index=False)
    summary = V._summarise(frame)
    summary["mode"] = label
    summary["gain_vs_full"] = float(summary["rmse"] - full["rmse"])
    summaries.append(summary)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "component_summary_partial.csv", index=False)

summary_frame = pd.DataFrame(summaries).sort_values("rmse")
summary_frame.to_csv(OUT_DIR / "component_summary.csv", index=False)
by_mode = {row["mode"]: row for row in summaries}

effects = {
    "affine_reference_ft": float(by_mode["raw_reference"]["rmse"] - full["rmse"]),
    "cauchy_vs_gaussian_ft": float(by_mode["gaussian"]["rmse"] - full["rmse"]),
    "persistence_ft": float(by_mode["no_persistence"]["rmse"] - full["rmse"]),
}
directions = {key: bool(value > 0.0) for key, value in effects.items()}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_matched_component_attribution_after_sequence_scale_repair",
    "raw_inputs_only": True,
    "uses_sp45_track8_mainline_strong_cp12_gold_neighbor": False,
    "protocol": {
        "prior_folds": list(V.V.PRIOR_FOLDS),
        "selection_fold": V.V.SELECTION_FOLD,
        "application_fold": V.V.APPLICATION_FOLD,
        "untouched_fold": V.V.UNTOUCHED_FOLD,
        "fold2_used_for_tuning": False,
    },
    "frozen_recipe": recipe,
    "full_reference": full,
    "controls": by_mode,
    "component_effects_ft": effects,
    "expected_direction": directions,
    "all_component_directions_pass": bool(all(directions.values())),
    "author_reference_not_a_gate": {
        "backbone_pooled_all_773": 9.99,
        "persistence_ablation_cost": 2.30,
        "typewell_reference_ablation_cost": 1.26,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS directions -> rebuild the segment-length posterior and test posterior mean/consensus decoding"
        if all(directions.values())
        else "FAIL directions -> retain only positive components and audit the segment boundary definition"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSTRIDE V6 MATCHED COMPONENT SUMMARY")
print(summary_frame.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
