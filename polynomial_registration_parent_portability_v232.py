"""V232: frozen parent-portability audit for the V231 polynomial move."""

from __future__ import annotations

import gc
import json
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
WORKER = next(
    path for path in (
        ROOT / "polynomial_registration_parent_worker_v232.py",
        Path.cwd() / "polynomial_registration_parent_worker_v232.py",
    ) if path.is_file()
)
OUT = OOF / "polynomial_registration_parent_portability_v232"
OUT.mkdir(parents=True, exist_ok=True)

PARENTS = (
    (
        "v144",
        OOF / "final_consensus_sequence_coefficient_full5_v144/crossfit_oof.parquet",
        "nested_sequence_coefficient",
    ),
    (
        "v169",
        OOF / "historical_analog_ranked_consensus_v169/fullrow_oof.parquet",
        "nested_consensus",
    ),
    (
        "v66",
        OOF / "frontier_6p377_full_fivefold_lineage_v66/incumbent_6p377_full_fivefold_oof.parquet",
        "incumbent_6p377_replay",
    ),
)
CAP = 12.0
DOSE = 0.25


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


summary_rows = []
fold_rows = []
bootstrap_rows = []
for name, path, column in PARENTS:
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"\nV232 START {name}", flush=True)
    namespace = runpy.run_path(
        str(WORKER),
        init_globals={
            "V232_PARENT_NAME": name,
            "V232_PARENT_PATH": path,
            "V232_PARENT_COLUMN": column,
        },
    )
    del namespace
    gc.collect()
    source = OOF / f"polynomial_registration_parent_replay_v232_{name}/fullrow_oof.parquet"
    frame = pd.read_parquet(source)
    if len(frame) != 3_783_989 or frame["well"].nunique() != 773:
        raise RuntimeError({"parent": name, "rows": len(frame), "wells": frame["well"].nunique()})
    truth = frame["truth"].to_numpy(np.float64)
    parent = frame["v168"].to_numpy(np.float64)
    move = frame["polynomial_move"].to_numpy(np.float64)
    candidate = parent + DOSE * CAP * np.tanh(move / CAP)
    base_error = parent - truth
    candidate_error = candidate - truth
    base_rmse = float(np.sqrt(np.mean(base_error ** 2)))
    candidate_rmse = float(np.sqrt(np.mean(candidate_error ** 2)))
    local_folds = []
    for held in range(1, 6):
        mask = frame["fold"].to_numpy(np.int8) == held
        left = float(np.sqrt(np.mean(base_error[mask] ** 2)))
        right = float(np.sqrt(np.mean(candidate_error[mask] ** 2)))
        row = {
            "parent": name, "fold": held, "base_rmse": left,
            "candidate_rmse": right, "gain": left - right,
        }
        fold_rows.append(row)
        local_folds.append(row)

    perwell = pd.DataFrame({
        "well": frame["well"].astype(str), "base_sq": base_error ** 2,
        "candidate_sq": candidate_error ** 2,
    }).groupby("well", as_index=False).agg(
        rows=("base_sq", "size"), base_sse=("base_sq", "sum"),
        candidate_sse=("candidate_sq", "sum"),
    )
    rng = np.random.default_rng(232000 + sum(map(ord, name)))
    gain = np.empty(3000, np.float64)
    rows = perwell["rows"].to_numpy(np.int64)
    left = perwell["base_sse"].to_numpy(np.float64)
    right = perwell["candidate_sse"].to_numpy(np.float64)
    for draw in range(len(gain)):
        index = rng.integers(0, len(perwell), len(perwell))
        denominator = rows[index].sum()
        gain[draw] = np.sqrt(left[index].sum() / denominator) - np.sqrt(right[index].sum() / denominator)
    bootstrap = {
        "parent": name, "gain_p05": float(np.quantile(gain, 0.05)),
        "gain_median": float(np.median(gain)), "gain_p95": float(np.quantile(gain, 0.95)),
        "probability_gain_positive": float(np.mean(gain > 0)),
    }
    bootstrap_rows.append(bootstrap)
    summary_rows.append({
        "parent": name, "base_rmse": base_rmse, "candidate_rmse": candidate_rmse,
        "gain": base_rmse - candidate_rmse,
        "positive_folds": sum(row["gain"] > 0 for row in local_folds),
        "minimum_fold_gain": min(row["gain"] for row in local_folds),
        "bootstrap_p05": bootstrap["gain_p05"],
        "bootstrap_probability": bootstrap["probability_gain_positive"],
    })
    del frame, truth, parent, move, candidate, base_error, candidate_error, perwell
    source.unlink()
    gc.collect()

summary = pd.DataFrame(summary_rows)
fold_report = pd.DataFrame(fold_rows)
bootstrap_report = pd.DataFrame(bootstrap_rows)
strong = summary.set_index("parent").loc[["v144", "v169"]]
checks = {
    "v144_and_v169_gain_ge_0p03": bool((strong["gain"] >= 0.03).all()),
    "v144_and_v169_positive_folds_ge_4": bool((strong["positive_folds"] >= 4).all()),
    "v144_and_v169_minimum_fold_gain_ge_minus0p10": bool((strong["minimum_fold_gain"] >= -0.10).all()),
    "at_least_two_bootstrap_probabilities_ge_0p90": int((summary["bootstrap_probability"] >= 0.90).sum()) >= 2,
}
verdict = {
    "experiment": "polynomial_registration_parent_portability_v232",
    "stage": "frozen_full773_parent_portability_audit",
    "recipe": {"window": 128.0, "lambda": 0.3, "mode": "soft32", "cap": CAP, "dose": DOSE},
    "parents": summary_rows,
    "folds": fold_rows,
    "bootstrap": bootstrap_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> permit conservative transfer test on the deployable mainline" if all(checks.values())
            else "FAIL -> keep V231 attached only to V168/V170",
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_report.to_csv(OUT / "folds.csv", index=False)
bootstrap_report.to_csv(OUT / "bootstrap.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
print("\nV232 SUMMARY")
print(summary.to_string(index=False))
print("\nV232 FOLDS")
print(fold_report.to_string(index=False))
print("\nV232 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
