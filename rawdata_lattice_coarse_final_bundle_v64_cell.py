"""Fit the final all-well V4 coarse ranker and export a deployment bundle.

This is an AutoDL training cell.  All training records use one structural
prior fitted on every eligible training well.  Per-well sampled matrices are
cached so an interrupted run can resume without regenerating beam banks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("Install lightgbm before running V64: pip install lightgbm") from exc


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_coarse_final_bundle_v64"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "sampled_training_rows"
BUNDLE_DIR = OUT_DIR / "deployment"
for directory in (OUT_DIR, CACHE_DIR, BUNDLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V4_VERDICT = OOF_ROOT / "rawdata_lattice_coarse_ranker_v4" / "verdict.json"
RANDOM_SEED = 20260721
DEFAULT_ITERATIONS = 400
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
    namespace = {"__name__": "raw_lattice_v4_final_definitions"}
    exec(compile(source.split(marker, 1)[0], str(V4_SOURCE), "exec"), namespace)
    return namespace


Q = _bootstrap_v4()
V = Q["V"]
DATA_ROOT = Path(Q["DATA_ROOT"])
SEGMENT_LENGTH_FT = float(Q["SEGMENT_LENGTH_FT"])
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
    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q["_candidate_union"](
        record, prior
    )
    features, feature_names = Q["_candidate_evidence"](
        record, prior, paths, segments, arm_rank, arm_score_z, source_mask
    )
    if feature_names_expected is not None and feature_names != feature_names_expected:
        raise RuntimeError(f"{record['well']}: feature schema changed")
    targets = Q["_exact_targets"](record, prior, paths, segments)
    panel = {
        "features": features,
        "mse": targets["mse"].astype(np.float32),
        "arm_rank": arm_rank,
    }
    indices = Q["_training_indices"](panel, record["well"])
    x = features[indices].astype(np.float32, copy=False)
    y = Q["_rank_labels"](targets["mse"][indices]).astype(np.int8)
    return x, y, feature_names, int(len(paths))


well_paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
wells = [path.name.split("__", 1)[0] for path in well_paths]
if len(wells) < 773:
    raise RuntimeError(f"Expected 773 training wells, found {len(wells)} in {DATA_ROOT}")

records = {}
failures = []
for number, well in enumerate(wells, 1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(wells):
        print(f"V64 load records {number}/{len(wells)} failures={len(failures)}", flush=True)

eligible_wells = [well for well in wells if well in records]
if len(eligible_wells) < MIN_ELIGIBLE_WELLS:
    raise RuntimeError(
        f"Only {len(eligible_wells)} eligible wells; failures={failures[:10]}"
    )
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

print(f"V64 fitting structural prior on {len(eligible_wells)} wells", flush=True)
prior = V["_fit_prior"](
    [records[well] for well in eligible_wells], SEGMENT_LENGTH_FT
)
_write_json(BUNDLE_DIR / "structural_prior.json", _prior_payload(prior))

feature_names = None
build_rows = []
for number, well in enumerate(eligible_wells, 1):
    path = _cache_path(well)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            x = data["x"]
            y = data["y"]
            names = json.loads(str(data["feature_names_json"].item()))
            candidate_count = int(data["candidate_count"][0])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError(f"{well}: cached feature schema mismatch")
        status = "reused"
    else:
        x, y, names, candidate_count = _build_sample(
            records[well], prior, feature_names
        )
        if feature_names is None:
            feature_names = names
        np.savez_compressed(
            path,
            x=x,
            y=y,
            feature_names_json=np.asarray(json.dumps(feature_names)),
            candidate_count=np.asarray([candidate_count], np.int32),
        )
        status = "built"
    build_rows.append(
        {
            "well": well,
            "sampled_candidates": int(len(y)),
            "bank_candidates": candidate_count,
            "status": status,
        }
    )
    if number % 20 == 0 or number == len(eligible_wells):
        print(
            f"V64 sampled panels {number}/{len(eligible_wells)} "
            f"last={status} elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

build_frame = pd.DataFrame(build_rows)
build_frame.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
_write_json(BUNDLE_DIR / "feature_names.json", feature_names)

x_parts = []
y_parts = []
groups = []
for well in eligible_wells:
    with np.load(_cache_path(well), allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int32, copy=False)
    x_parts.append(x)
    y_parts.append(y)
    groups.append(int(len(y)))
x_train = np.concatenate(x_parts, axis=0)
y_train = np.concatenate(y_parts, axis=0)
del x_parts, y_parts

if V4_VERDICT.is_file():
    v4_verdict = json.loads(V4_VERDICT.read_text(encoding="utf-8"))
    iterations = int(v4_verdict.get("best_iteration") or DEFAULT_ITERATIONS)
else:
    v4_verdict = {}
    iterations = DEFAULT_ITERATIONS
iterations = max(1, iterations)

print(
    "V64 FINAL MATRIX",
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
model_path = BUNDLE_DIR / "coarse_ranker_all773.txt"
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
    "stage": "all_eligible_train_wells_final_coarse_ranker_bundle",
    "legal_inputs_only": True,
    "training_wells_requested": len(wells),
    "training_wells_eligible": len(eligible_wells),
    "excluded_records": failures,
    "candidate_union": list(Q["SELECTED_ARM_IDS"]),
    "segment_length_ft": SEGMENT_LENGTH_FT,
    "feature_count": len(feature_names),
    "sampled_candidate_rows": int(len(x_train)),
    "iterations": iterations,
    "blend_weight_coarse": 0.10,
    "blend_weight_source": "V63 Fold2 analytic 0.09835 rounded to 0.10",
    "model": model_path.name,
    "model_sha256": _sha256(model_path),
    "reload_prediction_max_abs": prediction_gap,
    "source_manifest": source_manifest,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": "Upload deployment directory and run the V65 Kaggle test-inference/fusion cell",
}
_write_json(BUNDLE_DIR / "bundle.json", bundle)
_write_json(OUT_DIR / "verdict.json", {**bundle, "strict_pass": True})

print("\nV64 FINAL VERDICT")
print(json.dumps(_clean({**bundle, "strict_pass": True}), indent=2), flush=True)
print(f"saved bundle: {BUNDLE_DIR}", flush=True)
