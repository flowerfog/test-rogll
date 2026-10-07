"""V211: full-773 paired OOF audit of the confirmed 1200-particle PF core.

This is a scale confirmation, not a tuning run.  It compares the exact
seed-1 600-particle reference with the frozen 1200-particle candidate on every
available training well.  Per-well files make the run resumable.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


WORKERS = max(1, int(os.environ.get("V211_WORKERS", min(8, os.cpu_count() or 8))))
for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "mainline_pf_particle_count_gateway_v209.py"
OUT_DIR = ROOT / "oof_exports" / "mainline_pf_particle_count_full773_v211"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MULTIPLIERS = (1, 2)


def _bootstrap() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = 'print(\n    "V209 CONFIG",'
    if marker not in source:
        raise RuntimeError(f"Cannot locate V209 execution boundary in {SOURCE}")
    exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), globals())


_bootstrap()

# Redirect V209's resumable cache into an independent all-well directory.
OUT = OUT_DIR
CACHE = OUT_DIR / "well_cache"
CACHE.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = Path(CFG.DATA) / "train"
WELLS = sorted(
    path.name.replace("__horizontal_well.csv", "")
    for path in TRAIN_DIR.glob("*__horizontal_well.csv")
    if (TRAIN_DIR / path.name.replace("__horizontal_well.csv", "__typewell.csv")).is_file()
)
if len(WELLS) != 773:
    raise RuntimeError(f"Expected 773 complete training wells, found {len(WELLS)}")

print(
    "V211 CONFIG",
    json.dumps(
        {
            "wells": len(WELLS),
            "reference_particles": {"ancc": DEFAULT_ANCC_N, "z": DEFAULT_PF_N},
            "candidate_particles": {"ancc": 2 * DEFAULT_ANCC_N, "z": 2 * DEFAULT_PF_N},
            "seed": REFERENCE_SEED,
            "workers": WORKERS,
            "selection_or_tuning_count": 0,
        },
        indent=2,
    ),
    flush=True,
)


def _load_or_predict_v211(well: str, multiplier: int) -> tuple[pd.DataFrame, bool]:
    destination = _cache_path("full773", well, multiplier)
    expected_variant = f"particles_x{multiplier}"
    if destination.is_file():
        frame = pd.read_parquet(destination)
        if set(frame["variant"].astype(str).unique()) != {expected_variant}:
            raise RuntimeError(f"{well}: invalid cached variant in {destination}")
        return frame, True

    prior_paths = (
        ROOT / "oof_exports" / "mainline_pf_particle_count_gateway_v209"
        / "well_cache" / "discovery" / f"{well}__x{multiplier}.parquet",
        ROOT / "oof_exports" / "mainline_pf_particle_count_gateway_v209"
        / "well_cache" / "confirmation" / f"{well}__x{multiplier}.parquet",
        ROOT / "oof_exports" / "mainline_pf_particle_count_confirmation_v209b"
        / "well_cache" / "confirmation192" / f"{well}__x{multiplier}.parquet",
    )
    source = next((path for path in prior_paths if path.is_file()), None)
    if source is not None:
        frame = pd.read_parquet(source)
        if set(frame["variant"].astype(str).unique()) != {expected_variant}:
            raise RuntimeError(f"{well}: invalid prior cached variant in {source}")
        temporary = destination.with_suffix(".tmp.parquet")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        return frame, True
    return _load_or_predict("full773", well, multiplier)


def _worker(well: str) -> tuple[str, int, str | None]:
    try:
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        for model in MODELS:
            try:
                model.set_params(n_jobs=1, num_threads=1)
            except (AttributeError, TypeError, ValueError):
                pass
        resumed = 0
        frames = []
        for multiplier in MULTIPLIERS:
            frame, was_resumed = _load_or_predict_v211(well, multiplier)
            frames.append(frame)
            resumed += int(was_resumed)
        keys = ["well", "id", "md_since", "truth"]
        if not frames[0][keys].equals(frames[1][keys]):
            raise RuntimeError(f"{well}: multiplier row alignment differs")
        return well, resumed, None
    except Exception as exc:
        return well, 0, repr(exc)


def _run() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    completed = 0
    resumed = 0
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        futures = {executor.submit(_worker, well): well for well in WELLS}
        for future in as_completed(futures):
            well, local_resumed, error = future.result()
            completed += 1
            resumed += local_resumed
            if error is not None:
                failures.append({"well": well, "reason": error})
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"V211 full773 {completed}/{len(futures)} resumed={resumed} "
                    f"failures={len(failures)} workers={WORKERS} "
                    f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
                    flush=True,
                )
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
    parts = []
    for well in WELLS:
        paths = [_cache_path("full773", well, multiplier) for multiplier in MULTIPLIERS]
        if all(path.is_file() for path in paths):
            parts.extend(pd.read_parquet(path) for path in paths)
    if not parts:
        raise RuntimeError(f"No complete V211 wells; failures={failures[:5]}")
    return pd.concat(parts, ignore_index=True), failures


def _fold_table(long: pd.DataFrame) -> pd.DataFrame | None:
    candidates = (
        ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        ROOT / "well_metadata.csv",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    folds = pd.read_csv(path, dtype={"well": str})
    if not {"well", "fold"}.issubset(folds.columns):
        return None
    fold_map = folds.drop_duplicates("well").set_index("well")["fold"]
    local = long.copy()
    local["fold"] = local["well"].astype(str).map(fold_map)
    if local["fold"].isna().any():
        return None
    rows = []
    for fold, frame in local.groupby("fold", sort=True):
        summary, comparisons = _evaluate(frame, MULTIPLIERS, f"fold{int(fold)}")
        comparison = comparisons[0]
        rows.append(
            {
                "fold": int(fold),
                "wells": int(frame["well"].nunique()),
                "rows": int(summary.iloc[0]["rows"]),
                "reference_rmse": comparison["reference_rmse"],
                "candidate_rmse": comparison["candidate_rmse"],
                "gain": comparison["gain"],
                "far_gain": comparison["far_gain"],
                "p95_gain": comparison["p95_gain"],
            }
        )
    return pd.DataFrame(rows)


long, failures = _run()
complete_wells = int(long.groupby("well")["variant"].nunique().eq(2).sum())
summary, comparisons = _evaluate(long, MULTIPLIERS, "full773")
comparison = comparisons[0]
folds = _fold_table(long)

checks = {
    "coverage_773": complete_wells == 773,
    "pooled_gain_positive": comparison["gain"] > 0.0,
    "far_gain_positive": comparison["far_gain"] > 0.0,
    "bootstrap_probability_gain_ge_0p90": comparison["bootstrap"]["probability_gain_positive"] >= 0.90,
    "positive_folds_ge_4": bool(folds is not None and (folds["gain"] > 0.0).sum() >= 4),
}
verdict = {
    "experiment": "mainline_pf_particle_count_full773_v211",
    "stage": "complete_773well_same_seed_600_vs_1200_particle_core_oof",
    "changed_variable": "ANCC and Z particle count 600 -> 1200 only",
    "selection_or_tuning_count": 0,
    "coverage": {"wells": complete_wells, "rows": int(summary.iloc[0]["rows"]), "failures": len(failures)},
    "comparison": comparison,
    "folds": [] if folds is None else folds.to_dict(orient="records"),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "Propagate the paired core delta through the exact final OOF chain",
}

summary.to_csv(OUT_DIR / "summary.csv", index=False)
long.to_parquet(OUT_DIR / "paired_predictions.parquet", index=False, compression="zstd")
if folds is not None:
    folds.to_csv(OUT_DIR / "folds.csv", index=False)
(OUT_DIR / "result.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nV211 SUMMARY")
print(summary.to_string(index=False), flush=True)
if folds is not None:
    print("\nV211 FOLDS")
    print(folds.to_string(index=False), flush=True)
print("\nV211 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
