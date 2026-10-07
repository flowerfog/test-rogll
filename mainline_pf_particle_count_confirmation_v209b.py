"""V209b: 192-well frozen confirmation of the V209 2x particle count.

V209 selected 2x on 32 discovery wells with +0.134 pooled RMSE gain and
+0.202 far-tail gain.  Its old fail-fast gate rejected the result solely on a
32-well p95 estimate.  This script freezes 2x, excludes every discovery well,
and evaluates it on a difficulty-stratified 192-well confirmation set.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


WORKERS_V209B = max(
    1, int(os.environ.get("V209_WORKERS", min(8, os.cpu_count() or 8)))
)

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd


STARTED_V209B = time.time()
ROOT_V209B = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE_V209B = ROOT_V209B / "mainline_pf_particle_count_gateway_v209.py"
OOF_V209B = ROOT_V209B / "oof_exports"
OUT_V209B = OOF_V209B / "mainline_pf_particle_count_confirmation_v209b"
OUT_V209B.mkdir(parents=True, exist_ok=True)

FROZEN_MULTIPLIER_V209B = 2
N_WELLS_V209B = 192
MIN_COMPLETE_V209B = 184


def _bootstrap_v209b():
    if not SOURCE_V209B.is_file():
        raise FileNotFoundError(SOURCE_V209B)
    source = SOURCE_V209B.read_text(encoding="utf-8")
    marker = 'print(\n    "V209 CONFIG",'
    if marker not in source:
        raise RuntimeError(f"Cannot locate V209 execution boundary in {SOURCE_V209B}")
    exec(compile(source.split(marker, 1)[0], str(SOURCE_V209B), "exec"), globals())


_bootstrap_v209b()


def _select_confirmation_v209b() -> tuple[list[str], int]:
    excluded = set(EXCLUDED)
    excluded.update(DISCOVERY_SET)

    ranking_path = Path(RANKING_SOURCE)
    truth_column, prediction_column = _prediction_columns_v197(ranking_path)
    frame = pd.read_parquet(
        ranking_path, columns=["well", truth_column, prediction_column]
    )
    frame["well"] = frame["well"].astype(str)
    frame = frame.loc[~frame["well"].isin(excluded)].copy()
    frame["e2"] = np.square(
        frame[prediction_column].to_numpy(np.float64)
        - frame[truth_column].to_numpy(np.float64)
    )
    difficulty = frame.groupby("well")["e2"].mean().pow(0.5).sort_values()
    available = {
        path.stem.replace("__horizontal_well", "")
        for path in (Path(CFG.DATA) / "train").glob("*__horizontal_well.csv")
    }
    difficulty = difficulty[difficulty.index.isin(available)]
    strata = np.array_split(difficulty.index.to_numpy(), 3)
    selected: list[str] = []
    for stratum in strata:
        if len(stratum) < 66:
            raise RuntimeError(f"Only {len(stratum)} wells in a V209b stratum")
        positions = np.linspace(0, len(stratum) - 1, 66)[1:-1]
        selected.extend(str(stratum[int(round(position))]) for position in positions)
    if len(selected) != N_WELLS_V209B or len(set(selected)) != N_WELLS_V209B:
        raise RuntimeError(f"Invalid V209b selection: {len(selected)}/{len(set(selected))}")
    if set(selected).intersection(excluded):
        raise RuntimeError("V209b selected a discovery or prior-gateway well")
    return sorted(selected), len(excluded)


CONFIRMATION_WELLS_V209B, EXCLUDED_COUNT_V209B = _select_confirmation_v209b()

# V209's helper writes beneath its own CACHE global.  Redirect it before any
# inference so V209b remains independently resumable.
OUT = OUT_V209B
CACHE = OUT_V209B / "well_cache"
CACHE.mkdir(parents=True, exist_ok=True)

print(
    "V209b CONFIG",
    json.dumps(
        {
            "frozen_multiplier": FROZEN_MULTIPLIER_V209B,
            "default_ancc_particles": DEFAULT_ANCC_N,
            "default_z_particles": DEFAULT_PF_N,
            "candidate_ancc_particles": DEFAULT_ANCC_N * FROZEN_MULTIPLIER_V209B,
            "candidate_z_particles": DEFAULT_PF_N * FROZEN_MULTIPLIER_V209B,
            "confirmation_wells": len(CONFIRMATION_WELLS_V209B),
            "excluded_prior_and_discovery_wells": EXCLUDED_COUNT_V209B,
            "parallel_well_workers": WORKERS_V209B,
            "threads_per_worker": 1,
            "selection_or_tuning_count": 0,
        },
        indent=2,
    ),
    flush=True,
)

def _worker_v209b(well: str) -> tuple[str, int, str | None]:
    """Materialize both frozen variants for one well in an isolated process."""
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
        local = []
        for multiplier in (1, FROZEN_MULTIPLIER_V209B):
            frame, was_resumed = _load_or_predict("confirmation192", well, multiplier)
            local.append(frame)
            resumed += int(was_resumed)
        keys = ["well", "id", "md_since", "truth"]
        if not local[0][keys].equals(local[1][keys]):
            raise RuntimeError(f"{well}: multiplier row alignment differs")
        return well, resumed, None
    except Exception as exc:
        return well, 0, repr(exc)


def _run_parallel_v209b() -> pd.DataFrame:
    failures = []
    resumed = 0
    completed = 0
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS_V209B, mp_context=context) as executor:
        futures = {
            executor.submit(_worker_v209b, well): well
            for well in CONFIRMATION_WELLS_V209B
        }
        for future in as_completed(futures):
            well, local_resumed, error = future.result()
            resumed += local_resumed
            completed += 1
            if error is not None:
                failures.append({"stage": "confirmation192", "well": well, "reason": error})
            if completed % 6 == 0 or completed == len(futures):
                print(
                    f"V209b confirmation192 {completed}/{len(futures)} "
                    f"resumed={resumed} failures={len(failures)} "
                    f"workers={WORKERS_V209B} "
                    f"elapsed_min={(time.time() - STARTED_V209B) / 60.0:.1f}",
                    flush=True,
                )
    if failures:
        pd.DataFrame(failures).to_csv(
            OUT_V209B / "confirmation192_failures.csv", index=False
        )

    parts = []
    for well in CONFIRMATION_WELLS_V209B:
        local = []
        for multiplier in (1, FROZEN_MULTIPLIER_V209B):
            path = _cache_path("confirmation192", well, multiplier)
            if path.is_file():
                local.append(pd.read_parquet(path))
        if len(local) == 2:
            parts.extend(local)
    if not parts:
        raise RuntimeError(f"No complete V209b predictions; failures={failures[:5]}")
    return pd.concat(parts, ignore_index=True)


long_v209b = _run_parallel_v209b()
complete_v209b = long_v209b.groupby("well")["variant"].nunique().eq(2).sum()
if complete_v209b < MIN_COMPLETE_V209B:
    raise RuntimeError(f"Only {complete_v209b} complete V209b wells")

summary_v209b, comparisons_v209b = _evaluate(
    long_v209b, (1, FROZEN_MULTIPLIER_V209B), "confirmation192"
)
comparison_v209b = comparisons_v209b[0]

# Pooled RMSE is the competition objective.  P95 is reported as a safety
# diagnostic, but it cannot veto a positive pooled result by itself.
checks_v209b = {
    "complete_wells_ge_184": bool(complete_v209b >= MIN_COMPLETE_V209B),
    "pooled_gain_ge_0p05": bool(comparison_v209b["gain"] >= 0.05),
    "far_gain_positive": bool(comparison_v209b["far_gain"] > 0.0),
    "bootstrap_probability_gain_ge_0p90": bool(
        comparison_v209b["bootstrap"]["probability_gain_positive"] >= 0.90
    ),
    "bootstrap_p05_not_worse_than_minus0p02": bool(
        comparison_v209b["bootstrap"]["gain_p05"] >= -0.02
    ),
}
strict_pass_v209b = bool(all(checks_v209b.values()))
verdict_v209b = {
    "experiment": "mainline_pf_particle_count_confirmation_v209b",
    "stage": "frozen_192well_same_seed_2x_particle_confirmation",
    "changed_variable": "ANCC and Z particle count 600 -> 1200 only",
    "selection_or_tuning_count": 0,
    "discovery_wells_reused": 0,
    "coverage": {
        "requested_wells": N_WELLS_V209B,
        "complete_wells": int(complete_v209b),
        "rows": int(summary_v209b.iloc[0]["rows"]),
    },
    "comparison": comparison_v209b,
    "checks": checks_v209b,
    "strict_pass": strict_pass_v209b,
    "elapsed_minutes": (time.time() - STARTED_V209B) / 60.0,
    "next": (
        "PASS -> propagate 1200-particle ANCC/Z through the exact final V69/V77 chain"
        if strict_pass_v209b else
        "FAIL -> 2x particle scaling does not generalize at 192-well scale; retain 600 particles"
    ),
}

summary_v209b.to_csv(OUT_V209B / "summary.csv", index=False)
long_v209b.to_parquet(
    OUT_V209B / "paired_predictions.parquet", index=False, compression="zstd"
)
(OUT_V209B / "confirmation_wells.json").write_text(
    json.dumps(CONFIRMATION_WELLS_V209B, indent=2), encoding="utf-8"
)
(OUT_V209B / "result.json").write_text(
    json.dumps(verdict_v209b, indent=2), encoding="utf-8"
)
print("\nV209b SUMMARY")
print(summary_v209b.to_string(index=False), flush=True)
print("\nV209b FINAL VERDICT")
print(json.dumps(verdict_v209b, indent=2), flush=True)
print("saved:", OUT_V209B, flush=True)
