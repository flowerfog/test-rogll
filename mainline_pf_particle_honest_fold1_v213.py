"""V213: honest Fold1 gateway for the 600 -> 1200 PF particle change.

The deployment boosters cannot produce OOF.  This script rebuilds spatial
imputers on Folds2-5 only, trains three LightGBM models on those wells using
600-particle features, and predicts Fold1 twice with the same fitted models.
Only the PF particle count changes between the two held-out predictions.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / "mainline_pf_particle_honest_fold1_v213"
MODEL_OUT = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "mainline_pf_particle_count_gateway_v209.py"
HELD_FOLD = 1
REFERENCE_SEED = 1
WORKERS = max(1, int(os.environ.get("V213_WORKERS", min(8, os.cpu_count() or 8))))


def _bootstrap() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = 'print(\n    "V209 CONFIG",'
    if marker not in source:
        raise RuntimeError(f"Cannot locate V209 boundary in {SOURCE}")
    exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), globals())


_bootstrap()

ORIGINAL_BUILD_WELL_V213 = build_well
ORIGINAL_RUN_PF_ANCC_V213 = ORIGINAL_RUN_PF_ANCC
ORIGINAL_RUN_PF_Z_V213 = ORIGINAL_RUN_PF_Z
DEFAULT_ANCC_V213 = int(DEFAULT_ANCC_N)
DEFAULT_Z_V213 = int(DEFAULT_PF_N)
if (DEFAULT_ANCC_V213, DEFAULT_Z_V213) != (600, 600):
    raise RuntimeError(f"Expected 600/600 defaults, got {DEFAULT_ANCC_V213}/{DEFAULT_Z_V213}")


def _find_fold_path() -> Path:
    candidates = (
        OOF / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        OOF / "frontier_6p377_full_fivefold_lineage_v66" / "well_metadata.csv",
        ROOT / "well_metadata.csv",
    )
    for path in candidates:
        if path.is_file():
            frame = pd.read_csv(path, nrows=5)
            if {"well", "fold"}.issubset(frame.columns):
                return path
    for path in OOF.rglob("well_metadata.csv"):
        frame = pd.read_csv(path, nrows=5)
        if {"well", "fold"}.issubset(frame.columns):
            return path
    raise FileNotFoundError("Cannot locate canonical well_metadata.csv")


fold_path = _find_fold_path()
folds = pd.read_csv(fold_path, dtype={"well": str})[["well", "fold"]].drop_duplicates()
if folds["well"].duplicated().any() or folds["well"].nunique() != 773:
    raise RuntimeError(f"Invalid fold metadata: {folds['well'].nunique()} wells")
train_wells = sorted(folds.loc[~folds["fold"].eq(HELD_FOLD), "well"].astype(str))
valid_wells = sorted(folds.loc[folds["fold"].eq(HELD_FOLD), "well"].astype(str))
if len(train_wells) != 618 or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected Fold1 split: {len(train_wells)}/{len(valid_wells)}")

# Critical leakage guard: hidden Fold1 wells contribute neither surfaces nor
# dense-ANCC points to the Fold1 feature builder.
_FI = None
_DI = None
init_imputers(train_wells)

thread_state = threading.local()
likpf_cache: dict[str, pd.DataFrame] = {}
likpf_lock = threading.Lock()


def _stable_seed(well: str) -> int:
    token = zlib.crc32(str(well).encode("ascii")) & 0x7FFFFFFF
    return int((token + 1000003 * REFERENCE_SEED) % 2147483647)


def run_pf_ancc(hw, tw_tvt, tw_gr, N=DEFAULT_ANCC_V213):
    well = getattr(thread_state, "well", None)
    multiplier = getattr(thread_state, "multiplier", None)
    if well is None or multiplier not in (1, 2):
        raise RuntimeError("V213 PF called outside an active well")
    _seed_numba_p5(_stable_seed(well))
    ancc = ORIGINAL_RUN_PF_ANCC_V213(hw, tw_tvt, tw_gr, N=DEFAULT_ANCC_V213 * multiplier)
    thread_state.cached_z = ORIGINAL_RUN_PF_Z_V213(hw, tw_tvt, tw_gr, N=DEFAULT_Z_V213 * multiplier)
    return ancc


def run_pf_z(hw, tw_tvt, tw_gr, N=DEFAULT_Z_V213):
    cached = getattr(thread_state, "cached_z", None)
    if cached is None:
        raise RuntimeError("V213 paired Z state missing")
    thread_state.cached_z = None
    return cached


def _likpf(well: str) -> pd.DataFrame:
    with likpf_lock:
        cached = likpf_cache.get(well)
    if cached is not None:
        return cached
    frame = _likpf_rows(well, "train")
    if frame is None or frame.empty:
        raise RuntimeError(f"{well}: no likPF rows")
    with likpf_lock:
        likpf_cache[well] = frame
    return frame


def _build_one(well: str, multiplier: int) -> pd.DataFrame:
    train_dir = Path(CFG.DATA) / "train"
    thread_state.well = well
    thread_state.multiplier = multiplier
    thread_state.cached_z = None
    try:
        frame = ORIGINAL_BUILD_WELL_V213(
            str(train_dir / f"{well}__horizontal_well.csv"),
            str(train_dir / f"{well}__typewell.csv"),
            True,
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"{well}: build_well returned no rows")
        if thread_state.cached_z is not None:
            raise RuntimeError(f"{well}: paired Z state was not consumed")
        return add_likpf_features(frame, _likpf(well))
    finally:
        thread_state.well = None
        thread_state.multiplier = None
        thread_state.cached_z = None


def _build_many(wells: list[str], multiplier: int, stage: str) -> pd.DataFrame:
    parts = []
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_build_one, well, multiplier): well for well in wells}
        for number, future in enumerate(as_completed(futures), 1):
            well = futures[future]
            try:
                parts.append(future.result())
            except Exception as exc:
                failures.append({"well": well, "reason": repr(exc)})
            if number % 20 == 0 or number == len(futures):
                print(
                    f"V213 {stage} {number}/{len(futures)} failures={len(failures)} "
                    f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
                    flush=True,
                )
    if failures:
        pd.DataFrame(failures).to_csv(OUT / f"{stage}_failures.csv", index=False)
        raise RuntimeError(f"V213 {stage} failures: {failures[:5]}")
    output = pd.concat(parts, ignore_index=True)
    output["well"] = output["well"].astype(str)
    return output.sort_values(["well", "id"], kind="stable").reset_index(drop=True)


print(
    "V213 CONFIG",
    json.dumps(
        {
            "held_fold": HELD_FOLD,
            "train_wells": len(train_wells),
            "valid_wells": len(valid_wells),
            "fold_path": str(fold_path),
            "imputer_training_wells": len(train_wells),
            "workers": WORKERS,
            "train_particles": 600,
            "valid_particles": [600, 1200],
        },
        indent=2,
    ),
    flush=True,
)

train = _build_many(train_wells, 1, "train_x1")
valid_x1 = _build_many(valid_wells, 1, "valid_x1")
valid_x2 = _build_many(valid_wells, 2, "valid_x2")

keys = ["well", "id"]
if not valid_x1[keys].equals(valid_x2[keys]):
    raise RuntimeError("Fold1 x1/x2 row alignment differs")
if train["well"].isin(valid_wells).any():
    raise RuntimeError("Fold1 wells leaked into training rows")

features = list(FEATURE_NAMES)
for frame in (train, valid_x1, valid_x2):
    for column in features:
        if column not in frame:
            frame[column] = 0.0
    values = frame[features].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite V213 feature matrix")

X_train = train[features].to_numpy(np.float32)
y_train = train["target"].to_numpy(np.float32)
X_valid_x1 = valid_x1[features].to_numpy(np.float32)
X_valid_x2 = valid_x2[features].to_numpy(np.float32)

deployment_paths = sorted(Path(MODELS_DIR).glob("lgb*.pkl"))
if len(deployment_paths) != 3:
    raise RuntimeError(f"Expected three deployment LGB models, found {len(deployment_paths)}")
pred_x1 = np.zeros(len(valid_x1), np.float64)
pred_x2 = np.zeros(len(valid_x2), np.float64)
model_reports = []
for number, path in enumerate(deployment_paths):
    deployment = joblib.load(path)
    params = deployment.get_params()
    trees = int(deployment.booster_.num_trees())
    params.update(n_estimators=trees, n_jobs=WORKERS)
    # A 3080Ti can train these histogram boosters quickly.  Fall back to CPU
    # if this LightGBM build has no GPU/OpenCL support.
    gpu_params = dict(params, device_type="gpu", gpu_use_dp=False)
    try:
        model = LGBMRegressor(**gpu_params)
        model.fit(X_train, y_train)
        device = "gpu"
    except Exception as exc:
        print(f"V213 lgb{number} GPU fallback: {exc!r}", flush=True)
        params.pop("device_type", None)
        params.pop("gpu_use_dp", None)
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        device = "cpu"
    pred_x1 += model.predict(X_valid_x1) / len(deployment_paths)
    pred_x2 += model.predict(X_valid_x2) / len(deployment_paths)
    joblib.dump(model, MODEL_OUT / f"lgb{number}_heldout_fold1.pkl", compress=3)
    model_reports.append({"model": f"lgb{number}", "trees": trees, "device": device})
    print(f"V213 trained lgb{number}: trees={trees} device={device}", flush=True)

absolute_x1 = make_prediction(valid_x1, pred_x1, None).astype(np.float64)
absolute_x2 = make_prediction(valid_x2, pred_x2, None).astype(np.float64)
truth = valid_x1["last_known_tvt"].to_numpy(np.float64) + valid_x1["target"].to_numpy(np.float64)
distance = valid_x1["md_since"].to_numpy(np.float64)


def _metric(prediction: np.ndarray, name: str) -> dict:
    error = prediction - truth
    e2 = np.square(error)
    by_well = pd.DataFrame({"well": valid_x1["well"], "e2": e2}).groupby("well")["e2"].mean().pow(0.5)
    far = distance > 2000.0
    return {
        "endpoint": name,
        "rows": int(len(prediction)),
        "wells": int(valid_x1["well"].nunique()),
        "rmse": float(np.sqrt(e2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(by_well.median()),
        "p95_well_rmse": float(by_well.quantile(0.95)),
        "far_gt2000_rmse": float(np.sqrt(e2[far].mean())),
    }


reference = _metric(absolute_x1, "particles_x1")
candidate = _metric(absolute_x2, "particles_x2")
summary = pd.DataFrame([reference, candidate])
gain = reference["rmse"] - candidate["rmse"]
far_gain = reference["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]
p95_gain = reference["p95_well_rmse"] - candidate["p95_well_rmse"]

output = valid_x1[["well", "id", "md_since"]].copy()
output["fold"] = HELD_FOLD
output["truth"] = truth.astype(np.float32)
output["particles_x1"] = absolute_x1.astype(np.float32)
output["particles_x2"] = absolute_x2.astype(np.float32)
output.to_parquet(OUT / "fold1_predictions.parquet", index=False, compression="zstd")

checks = {
    "coverage_155": int(valid_x1["well"].nunique()) == 155,
    "heldout_wells_absent_from_training": not train["well"].isin(valid_wells).any(),
    "gain_positive": gain > 0.0,
    "far_gain_positive": far_gain > 0.0,
    "p95_not_worse_by_gt_0p20": p95_gain >= -0.20,
}
verdict = {
    "experiment": "mainline_pf_particle_honest_fold1_v213",
    "stage": "honest_outer_Fold1_600_vs_1200_particle_gateway",
    "model_training": "Folds2-5 only",
    "imputer_training": "Folds2-5 only",
    "heldout_fold": HELD_FOLD,
    "models": model_reports,
    "reference": reference,
    "candidate": candidate,
    "values": {"gain": gain, "far_gain": far_gain, "p95_gain": p95_gain},
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> run remaining four outer folds; FAIL -> reject particle scaling as an in-sample artifact",
}
summary.to_csv(OUT / "summary.csv", index=False)
(OUT / "result.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nV213 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV213 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT, flush=True)

del train, valid_x1, valid_x2, X_train, X_valid_x1, X_valid_x2
gc.collect()
