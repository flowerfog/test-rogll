"""Fit the V67-confirmed coarse ranker on all eligible training wells.

This AutoDL cell freezes the exact V67 candidate distribution: the canonical
Fold-4/5 structural prior, 67 ranker iterations, and a 0.30 deployment dose.
Only the rank-label training population expands to all eligible training wells.
The 463 cached V67 Fold-3/4/5 samples are reused and only Fold-1/2 samples are
built, so interrupted runs can resume without regenerating completed banks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("Install lightgbm before running V68: pip install lightgbm") from exc


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_coarse_canonical_final_bundle_v68"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "sampled_training_rows"
BUNDLE_DIR = OUT_DIR / "deployment"
for directory in (OUT_DIR, CACHE_DIR, BUNDLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V67_DIR = OOF_ROOT / "rawdata_lattice_coarse_canonical_gateway_v67"
V67_VERDICT = V67_DIR / "verdict.json"
V67_PRIOR = V67_DIR / "canonical_fold45_prior.json"
V67_CACHE_DIR = V67_DIR / "train_samples"
RANDOM_SEED = 20260721
FROZEN_ITERATIONS = 67
FROZEN_BLEND_WEIGHT = 0.30
MIN_ELIGIBLE_WELLS = 770


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_v4():
    if not V4_SOURCE.is_file():
        raise FileNotFoundError(
            f"Upload rawdata_lattice_coarse_ranker_v4_cell.py to {ROOT}"
        )
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 bootstrap marker missing from {V4_SOURCE}")
    module = types.ModuleType("raw_lattice_v4_canonical_final_definitions")
    module.__file__ = str(V4_SOURCE)
    sys.modules[module.__name__] = module
    exec(
        compile(source.split(marker, 1)[0] + "\n", str(V4_SOURCE), "exec"),
        module.__dict__,
    )
    return module


Q = _bootstrap_v4()
V = Q.V
DATA_ROOT = Path(Q.DATA_ROOT)
SEGMENT_LENGTH_FT = float(Q.SEGMENT_LENGTH_FT)
SOURCE_NAMES = (
    "rawdata_segment_lattice_nested_v1_cell.py",
    "rawdata_lattice_beam_survival_v2_cell.py",
    "rawdata_lattice_mode_union_v3_cell.py",
    "rawdata_lattice_coarse_ranker_v4_cell.py",
)


def _prior_payload(prior):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in prior.items()
    }


def _cache_path(well):
    return CACHE_DIR / f"{well}.npz"


def _build_sample(record, prior, feature_names_expected=None):
    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(
        record, prior
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Mean of empty slice", category=RuntimeWarning
        )
        features, feature_names = Q._candidate_evidence(
            record, prior, paths, segments, arm_rank, arm_score_z, source_mask
        )
    if feature_names_expected is not None and feature_names != feature_names_expected:
        raise RuntimeError(f"{record['well']}: feature schema changed")
    targets = Q._exact_targets(record, prior, paths, segments)
    panel = {
        "features": features,
        "mse": targets["mse"].astype(np.float32),
        "arm_rank": arm_rank,
    }
    indices = Q._training_indices(panel, record["well"])
    x = features[indices].astype(np.float32, copy=False)
    y = Q._rank_labels(targets["mse"][indices]).astype(np.int8)
    return x, y, feature_names, int(len(paths))


for required in (V4_SOURCE, V67_VERDICT, V67_PRIOR):
    if not required.is_file():
        raise FileNotFoundError(required)
v67 = json.loads(V67_VERDICT.read_text(encoding="utf-8"))
if not v67.get("strict_pass"):
    raise RuntimeError("V67 did not strictly pass; refusing to create deployment bundle")
v67_iterations = int(v67["frozen_v4_recipe"]["iterations"])
v67_weight = float(v67["weight"]["frozen"])
if v67_iterations != FROZEN_ITERATIONS or abs(v67_weight - FROZEN_BLEND_WEIGHT) > 1e-12:
    raise RuntimeError(
        f"Frozen V67 recipe changed: iterations={v67_iterations}, weight={v67_weight}"
    )
v67_model_source = Path(v67["model"])
if not v67_model_source.is_file():
    raise FileNotFoundError(v67_model_source)
v67_model_path = BUNDLE_DIR / "coarse_ranker_canonical345_validated.txt"
shutil.copy2(v67_model_source, v67_model_path)

prior_payload = json.loads(V67_PRIOR.read_text(encoding="utf-8"))
prior = {
    key: np.asarray(value, np.float64) if key == "rate_grid" else value
    for key, value in prior_payload.items()
}
shutil.copy2(V67_PRIOR, BUNDLE_DIR / "structural_prior.json")

well_paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
wells = [path.name.split("__", 1)[0] for path in well_paths]
if len(wells) < 773:
    raise RuntimeError(f"Expected 773 training wells, found {len(wells)} in {DATA_ROOT}")

failures = []
feature_names = None
build_rows = []
sample_paths = {}
for number, well in enumerate(wells, 1):
    path = _cache_path(well)
    v67_path = V67_CACHE_DIR / f"{well}.npz"
    source_path = path if path.is_file() else v67_path if v67_path.is_file() else None
    try:
        if source_path is not None:
            with np.load(source_path, allow_pickle=False) as data:
                x = data["x"]
                y = data["y"]
                names = json.loads(str(data["feature_names_json"].item()))
                candidate_key = "candidate_count" if "candidate_count" in data else "bank_candidates"
                candidate_count = int(data[candidate_key][0])
            status = "reused_v68" if source_path == path else "reused_v67"
        else:
            record = V["_load_record"](well)
            x, y, names, candidate_count = _build_sample(record, prior, feature_names)
            np.savez_compressed(
                path,
                x=x,
                y=y,
                feature_names_json=np.asarray(json.dumps(names)),
                candidate_count=np.asarray([candidate_count], np.int32),
            )
            source_path = path
            status = "built"
        if feature_names is None:
            feature_names = list(names)
        elif list(names) != feature_names:
            raise RuntimeError(f"{well}: cached feature schema mismatch")
        sample_paths[well] = source_path
        build_rows.append(
            {
                "well": well,
                "sampled_candidates": int(len(y)),
                "bank_candidates": candidate_count,
                "status": status,
            }
        )
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 20 == 0 or number == len(wells):
        print(
            f"V68 sampled panels {number}/{len(wells)} usable={len(sample_paths)} "
            f"failures={len(failures)} elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

eligible_wells = [well for well in wells if well in sample_paths]
if len(eligible_wells) < MIN_ELIGIBLE_WELLS:
    raise RuntimeError(f"Only {len(eligible_wells)} eligible wells; failures={failures[:10]}")
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

build_frame = pd.DataFrame(build_rows)
build_frame.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
_write_json(BUNDLE_DIR / "feature_names.json", feature_names)

x_parts = []
y_parts = []
groups = []
for well in eligible_wells:
    with np.load(sample_paths[well], allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int32, copy=False)
    x_parts.append(x)
    y_parts.append(y)
    groups.append(int(len(y)))
x_train = np.concatenate(x_parts, axis=0)
y_train = np.concatenate(y_parts, axis=0)
del x_parts, y_parts

iterations = FROZEN_ITERATIONS

print(
    "V68 FINAL MATRIX",
    {
        "rows": len(x_train),
        "features": x_train.shape[1],
        "groups": len(groups),
        "iterations": iterations,
    },
    flush=True,
)

model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=(1, 8, 32),
    n_estimators=iterations,
    learning_rate=0.025,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=RANDOM_SEED,
    n_jobs=max(1, min(16, os.cpu_count() or 1)),
    verbosity=-1,
)
model.fit(x_train, y_train, group=groups)
model_path = BUNDLE_DIR / "coarse_ranker_canonical_all773.txt"
model.booster_.save_model(str(model_path))

reloaded = lgb.Booster(model_file=str(model_path))
probe = x_train[: min(1024, len(x_train))]
prediction_gap = float(
    np.max(np.abs(model.predict(probe) - reloaded.predict(probe)))
)
if prediction_gap > 1e-10:
    raise RuntimeError(f"Model reload parity failed: {prediction_gap}")

source_manifest = []
for name in SOURCE_NAMES:
    source = ROOT / name
    if not source.is_file():
        raise FileNotFoundError(source)
    target = BUNDLE_DIR / name
    shutil.copy2(source, target)
    source_manifest.append(
        {"name": name, "sha256": _sha256(target), "bytes": target.stat().st_size}
    )

bundle = {
    "experiment": EXPERIMENT,
    "stage": "v67_frozen_prior_all_eligible_train_wells_final_ranker_bundle",
    "legal_inputs_only": True,
    "training_wells_requested": len(wells),
    "training_wells_eligible": len(eligible_wells),
    "excluded_records": failures,
    "candidate_union": list(Q.SELECTED_ARM_IDS),
    "segment_length_ft": SEGMENT_LENGTH_FT,
    "feature_count": len(feature_names),
    "sampled_candidate_rows": int(len(x_train)),
    "iterations": iterations,
    "structural_prior_source": str(V67_PRIOR),
    "structural_prior_sha256": _sha256(BUNDLE_DIR / "structural_prior.json"),
    "blend_weight_coarse": FROZEN_BLEND_WEIGHT,
    "blend_weight_source": "V67 canonical Fold2 analytic 0.27944 rounded to 0.30",
    "model": model_path.name,
    "model_sha256": _sha256(model_path),
    "validated_model": v67_model_path.name,
    "validated_model_sha256": _sha256(v67_model_path),
    "validated_model_source": str(v67_model_source),
    "reload_prediction_max_abs": prediction_gap,
    "source_manifest": source_manifest,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": "Upload deployment directory and run V69 as the final Kaggle cell",
}
_write_json(BUNDLE_DIR / "bundle.json", bundle)
_write_json(OUT_DIR / "verdict.json", {**bundle, "strict_pass": True})

print("\nV68 FINAL VERDICT")
print(json.dumps(_clean({**bundle, "strict_pass": True}), indent=2), flush=True)
print(f"saved bundle: {BUNDLE_DIR}", flush=True)
