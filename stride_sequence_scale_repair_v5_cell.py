"""Repair STRIDE full-sequence evidence scaling before another full ablation.

V4 proved that its per-segment mean Cauchy cost was not invariant to path
partitioning: a 600-ft segment and a 50-ft segment each contributed one unit
of evidence.  This gateway restores an approximate product over independent
GR observations by scaling segment cost with length / effective_spacing.

Fold 4/5 priors and the Figure-1 state definition remain frozen.  Eight Fold-3
wells screen evidence spacing and persistence dose with a small beam; 32
disjoint Fold-3 wells confirm the best three with the production beam.  The
winning recipe is then applied once to untouched Fold 2.  Fold 1 is untouched.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_sequence_scale_repair_v5"
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
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload the updated stride_fullsequence_beam_v4_cell.py")
if not V3_VERDICT.exists():
    raise FileNotFoundError(V3_VERDICT)

DISCOVERY_WELLS = 8
CONFIRMATION_WELLS = 32
SCAN_BEAM = 24
PRODUCTION_BEAM = 48
SPACING_GRID_FT = (4.0, 8.0, 16.0, 32.0)
PERSISTENCE_GRID = (0.03, 0.10, 0.30, 1.00)
LAMBDA_GRID = (0.0, 0.06, 0.15, 0.30)


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
    module_name = "stride_v4_repaired_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


V = _bootstrap(SOURCE_PATH)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def _decode_group(records, prior, temperature, recipe, label, save_rows=False):
    metrics = []
    row_frames = []
    for number, record in enumerate(records, start=1):
        prediction, objective, nodes = V._decode(
            record,
            prior,
            temperature,
            "full",
            effective_spacing_ft=float(recipe["effective_spacing_ft"]),
            persistence_weight=float(recipe["persistence_weight"]),
            lambda_override=float(recipe["lambda_previous_dip"]),
        )
        metrics.append(V._metric_row(record, prediction, label, objective, nodes))
        if save_rows:
            rows = record["evaluation"]
            row_frames.append(
                pd.DataFrame(
                    {
                        "id": [f"{record['well']}_{int(row)}" for row in rows],
                        "well": record["well"],
                        "row_idx": rows,
                        "truth": record["truth"][rows],
                        "stride_v5": prediction,
                        "distance_ft": record["md"][rows] - record["anchor_md"],
                    }
                )
            )
        if number % 25 == 0 or number == len(records):
            print(f"{label} {number}/{len(records)}", flush=True)
    frame = pd.DataFrame(metrics)
    summary = V._summarise(frame)
    return frame, summary, row_frames


v3 = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
prior = v3["selected"]["prior"]
temperature = float(v3["selected"]["emission_temperature"])
folds = pd.read_csv(V.V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
selection_wells = sorted(folds.loc[folds["fold"].eq(V.V.SELECTION_FOLD), "well"])
application_wells = sorted(folds.loc[folds["fold"].eq(V.V.APPLICATION_FOLD), "well"])
if (len(selection_wells), len(application_wells)) != (154, 155):
    raise RuntimeError(f"Unexpected Fold3/Fold2 sizes: {len(selection_wells)}/{len(application_wells)}")

ordered = sorted(
    selection_wells,
    key=lambda well: hashlib.sha256(f"stride-v5:{well}".encode("ascii")).digest(),
)
discovery_wells = ordered[:DISCOVERY_WELLS]
confirmation_wells = ordered[DISCOVERY_WELLS:DISCOVERY_WELLS + CONFIRMATION_WELLS]
needed = discovery_wells + confirmation_wells + application_wells
records = {}
for number, well in enumerate(needed, start=1):
    records[well] = V.load_record(well)
    if number % 50 == 0 or number == len(needed):
        print(f"STRIDE v5 load {number}/{len(needed)}", flush=True)
discovery = [records[well] for well in discovery_wells]
confirmation = [records[well] for well in confirmation_wells]
application = [records[well] for well in application_wells]

started = time.time()
V.BEAM_SIZE = SCAN_BEAM
V._decode(discovery[0], prior, temperature, "full", 16.0, 0.3, prior["lambda_previous_dip"])

stage1 = []
for spacing in SPACING_GRID_FT:
    for persistence in PERSISTENCE_GRID:
        recipe = {
            "effective_spacing_ft": spacing,
            "persistence_weight": persistence,
            "lambda_previous_dip": float(prior["lambda_previous_dip"]),
        }
        _, metrics, _ = _decode_group(discovery, prior, temperature, recipe, "stage1")
        stage1.append({**recipe, **metrics})
        print("V5 STAGE1", spacing, persistence, metrics["rmse"], flush=True)
stage1_frame = pd.DataFrame(stage1).sort_values("rmse")

best_spacing = float(stage1_frame.iloc[0]["effective_spacing_ft"])
best_persistence = float(stage1_frame.iloc[0]["persistence_weight"])
stage2 = []
for lam in LAMBDA_GRID:
    recipe = {
        "effective_spacing_ft": best_spacing,
        "persistence_weight": best_persistence,
        "lambda_previous_dip": lam,
    }
    _, metrics, _ = _decode_group(discovery, prior, temperature, recipe, "stage2")
    stage2.append({**recipe, **metrics})
    print("V5 STAGE2", lam, metrics["rmse"], flush=True)
stage2_frame = pd.DataFrame(stage2).sort_values("rmse")

candidates = pd.concat([stage1_frame.head(3), stage2_frame.head(3)], ignore_index=True)
candidates = candidates.drop_duplicates(
    ["effective_spacing_ft", "persistence_weight", "lambda_previous_dip"]
).sort_values("rmse", kind="stable").head(3)

V.BEAM_SIZE = PRODUCTION_BEAM
confirmation_rows = []
for config_id, (_, candidate) in enumerate(candidates.iterrows(), start=1):
    recipe = {
        "effective_spacing_ft": float(candidate["effective_spacing_ft"]),
        "persistence_weight": float(candidate["persistence_weight"]),
        "lambda_previous_dip": float(candidate["lambda_previous_dip"]),
    }
    _, metrics, _ = _decode_group(
        confirmation, prior, temperature, recipe, f"confirmation_{config_id}"
    )
    confirmation_rows.append({"config_id": config_id, **recipe, **metrics})
    print("V5 CONFIRMATION", config_id, metrics["rmse"], flush=True)
confirmation_frame = pd.DataFrame(confirmation_rows).sort_values("rmse")
selected = confirmation_frame.iloc[0].to_dict()
selected_recipe = {
    "effective_spacing_ft": float(selected["effective_spacing_ft"]),
    "persistence_weight": float(selected["persistence_weight"]),
    "lambda_previous_dip": float(selected["lambda_previous_dip"]),
}

fold2_bywell, fold2_metrics, fold2_rows = _decode_group(
    application,
    prior,
    temperature,
    selected_recipe,
    "fold2_application",
    save_rows=True,
)
fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
pd.concat(fold2_rows, ignore_index=True).to_csv(
    OUT_DIR / "fold2_rows.csv.gz", index=False, compression="gzip"
)
stage1_frame.to_csv(OUT_DIR / "fold3_stage1_scan.csv", index=False)
stage2_frame.to_csv(OUT_DIR / "fold3_stage2_lambda_scan.csv", index=False)
confirmation_frame.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "full_sequence_evidence_scale_and_persistence_repair",
    "sample_unit": "one complete physical well",
    "raw_inputs_only": True,
    "uses_sp45_track8_mainline_strong_cp12_gold_neighbor": False,
    "protocol": {
        "prior_folds": list(V.V.PRIOR_FOLDS),
        "selection_fold": V.V.SELECTION_FOLD,
        "discovery_wells": DISCOVERY_WELLS,
        "confirmation_wells": CONFIRMATION_WELLS,
        "application_fold": V.V.APPLICATION_FOLD,
        "untouched_fold": V.V.UNTOUCHED_FOLD,
    },
    "repair": {
        "old_segment_evidence": "mean Cauchy cost",
        "new_segment_evidence": "Cauchy sum divided by effective independent-row spacing",
        "spacing_grid_ft": list(SPACING_GRID_FT),
        "persistence_grid": list(PERSISTENCE_GRID),
        "lambda_grid": list(LAMBDA_GRID),
    },
    "selected_recipe": selected_recipe,
    "fold3_confirmation": selected,
    "fold2": fold2_metrics,
    "reference": {
        "v4_full_rmse": 22.411423540468387,
        "v4_gaussian_rmse": 15.454286307195758,
        "author_backbone_pooled_773": 9.99,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "If Fold2 improves materially, run matched Cauchy/calibration/persistence ablations only for this recipe; "
        "otherwise revisit the segment-length and emission definitions."
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV5 STAGE1 SCAN")
print(stage1_frame.to_string(index=False), flush=True)
print("\nV5 STAGE2 SCAN")
print(stage2_frame.to_string(index=False), flush=True)
print("\nV5 CONFIRMATION")
print(confirmation_frame.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
