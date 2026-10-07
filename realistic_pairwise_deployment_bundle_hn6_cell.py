"""Package the validated HN3 crossfit tournament as a Kaggle deployment bundle.

Run on AutoDL after HN3, HN4 and HN5.  Candidate generation remains on the
Fold-4/5 structural prior used by the validation lineage.  Hidden inference
uses the mean pair probability from the four HN3 crossfit models; no model is
refit and no validation-selected component is changed here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "realistic_pairwise_deployment_bundle_hn6"
OUT_DIR = OOF_ROOT / EXPERIMENT
BUNDLE_DIR = OUT_DIR / "deployment"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
HN3_DIR = OOF_ROOT / "realistic_pairwise_crossfit_hn3"
HN4_VERDICT = OOF_ROOT / "realistic_pairwise_mainline_fusion_hn4" / "verdict.json"
HN5_VERDICT = OOF_ROOT / "realistic_pairwise_fixed_dose_hn5" / "verdict.json"
MODEL_FOLDS = (2, 3, 4, 5)
PRIOR_FOLDS = (4, 5)
BLEND_WEIGHT = 0.275
MENU_SIZE = 128


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_v4():
    if not V4_SOURCE.is_file():
        raise FileNotFoundError(V4_SOURCE)
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 bootstrap marker missing from {V4_SOURCE}")
    namespace = {"__name__": "hn6_v4_definitions", "__file__": str(V4_SOURCE)}
    exec(compile(source.split(marker, 1)[0], str(V4_SOURCE), "exec"), namespace)
    return namespace


for verdict_path in (HN4_VERDICT, HN5_VERDICT):
    if not verdict_path.is_file():
        raise FileNotFoundError(verdict_path)
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if not verdict.get("strict_pass"):
        raise RuntimeError(f"Required passed verdict is not strict-pass: {verdict_path}")

hn5 = json.loads(HN5_VERDICT.read_text(encoding="utf-8"))
selected_weight = float(hn5["selected_weight"])
if abs(selected_weight - BLEND_WEIGHT) > 1e-12:
    raise RuntimeError(f"HN5 weight changed: expected {BLEND_WEIGHT}, got {selected_weight}")

selected_features_path = HN3_DIR / "selected_features.json"
if not selected_features_path.is_file():
    raise FileNotFoundError(selected_features_path)
selected_features = json.loads(selected_features_path.read_text(encoding="utf-8"))
if len(selected_features) != 122:
    raise RuntimeError(f"Expected 122 evidence-only features, got {len(selected_features)}")

model_names = []
for fold in MODEL_FOLDS:
    source = HN3_DIR / f"pairwise_hold{fold}.txt"
    if not source.is_file():
        raise FileNotFoundError(source)
    model = lgb.Booster(model_file=str(source))
    if model.num_feature() != len(selected_features):
        raise RuntimeError(
            f"Fold {fold} model/schema mismatch: {model.num_feature()}/{len(selected_features)}"
        )
    name = f"pairwise_hold{fold}.txt"
    shutil.copy2(source, BUNDLE_DIR / name)
    model_names.append(name)

Q = _bootstrap_v4()
V = Q["V"]
DATA_ROOT = Path(Q["DATA_ROOT"])
all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
if len(all_wells) != 773:
    raise RuntimeError(f"Expected 773 training wells, found {len(all_wells)}")
folds = V["_stable_folds"](all_wells)
fold_by_well = dict(zip(folds["well"].astype(str), folds["fold"].astype(int)))
prior_wells = [well for well in all_wells if fold_by_well[well] in PRIOR_FOLDS]

records = []
failures = []
for number, well in enumerate(prior_wells, 1):
    try:
        records.append(V["_load_record"](well))
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(prior_wells):
        print(
            f"HN6 prior records {number}/{len(prior_wells)} failures={len(failures)}",
            flush=True,
        )
if len(records) < 307:
    raise RuntimeError(f"Only {len(records)} prior records: {failures[:10]}")

prior = V["_fit_prior"](records, float(Q["SEGMENT_LENGTH_FT"]))
_write_json(BUNDLE_DIR / "structural_prior.json", prior)
_write_json(BUNDLE_DIR / "selected_features.json", selected_features)
_write_json(BUNDLE_DIR / "prior_record_failures.json", failures)

source_names = (
    "rawdata_segment_lattice_nested_v1_cell.py",
    "rawdata_lattice_beam_survival_v2_cell.py",
    "rawdata_lattice_mode_union_v3_cell.py",
    "rawdata_lattice_coarse_ranker_v4_cell.py",
)
for name in source_names:
    source = ROOT / name
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, BUNDLE_DIR / name)

manifest = {
    "experiment": EXPERIMENT,
    "stage": "validated_hn3_crossfit_model_ensemble_deployment_bundle",
    "candidate_lineage": "rawdata_lattice_mode_union_v3",
    "model_lineage": "realistic_pairwise_crossfit_hn3",
    "validation_lineage": [
        "realistic_pairwise_mainline_fusion_hn4",
        "realistic_pairwise_fixed_dose_hn5",
    ],
    "models": model_names,
    "ensemble": "mean_pair_probability_then_round_robin_argmax",
    "selected_features": "selected_features.json",
    "feature_count": len(selected_features),
    "menu_size": MENU_SIZE,
    "prior_folds": list(PRIOR_FOLDS),
    "prior_wells": len(records),
    "blend_weight_hn": BLEND_WEIGHT,
    "blend_weight_parent": 1.0 - BLEND_WEIGHT,
    "parent_requirement": "exact 6.377 CP8 plus Neighbor submission.csv",
    "source_files": list(source_names),
    "hashes": {},
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
for path in sorted(BUNDLE_DIR.iterdir()):
    if path.is_file():
        manifest["hashes"][path.name] = _sha256(path)
_write_json(BUNDLE_DIR / "bundle.json", manifest)

archive_base = ROOT / EXPERIMENT
archive_path = Path(shutil.make_archive(str(archive_base), "zip", BUNDLE_DIR))
summary = {
    **manifest,
    "bundle_dir": BUNDLE_DIR,
    "archive": archive_path,
    "archive_mb": archive_path.stat().st_size / (1024 ** 2),
}
_write_json(OUT_DIR / "verdict.json", summary)

print("\nHN6 DEPLOYMENT BUNDLE COMPLETE")
print(json.dumps(_clean(summary), indent=2), flush=True)
print(f"Upload this archive as a Kaggle dataset: {archive_path}", flush=True)
