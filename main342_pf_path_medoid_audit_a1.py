"""M342-A1: truth-free whole-well PF seed-path medoid audit.

This consumes the already generated V97 paired seed predictions.  It does not
rerun PF and does not average PF state.  For each complete physical well, the
candidate is one of the six original seed paths: the path with minimum mean
RMS distance to the other seed paths.  Truth is used only for final metrics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "main342_pf_path_medoid_audit_a1"
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
SOURCE = OOF / "mainline_pf_repeatability_v97" / "paired_seed_predictions.parquet"
OUT = OOF / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.time()

REFERENCE_SEED = 1
EXPECTED_SEEDS = tuple(range(6))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260804
MIN_GAIN = 0.05
MIN_FAR_GAIN = 0.05
MIN_POSITIVE_WELL_FRACTION = 0.55
MIN_BOOTSTRAP_PROBABILITY = 0.90


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


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["truth"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    local = pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
    well_rmse = np.sqrt(local.groupby("well", sort=False)["e2"].mean().to_numpy(np.float64))
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def _bootstrap(by_well: pd.DataFrame) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = by_well["rows"].to_numpy(np.float64)
    reference_sse = by_well["reference_sse"].to_numpy(np.float64)
    candidate_sse = by_well["candidate_sse"].to_numpy(np.float64)
    gains = np.empty(BOOTSTRAP_DRAWS, np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(by_well), len(by_well))
        gains[draw] = (
            np.sqrt(reference_sse[index].sum() / rows[index].sum())
            - np.sqrt(candidate_sse[index].sum() / rows[index].sum())
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "gain_p05": float(np.quantile(gains, 0.05)),
        "gain_median": float(np.median(gains)),
        "gain_p95": float(np.quantile(gains, 0.95)),
        "probability_gain_positive": float(np.mean(gains > 0.0)),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(
        f"Missing {SOURCE}. Run mainline_pf_repeatability_v97_cell.py first; "
        "M342-A1 itself does not rerun PF."
    )

long = pd.read_parquet(SOURCE)
required = {"well", "id", "md_since", "truth", "seed", "prediction"}
missing = sorted(required.difference(long.columns))
if missing:
    raise RuntimeError(f"V97 artifact missing columns: {missing}")
long["well"] = long["well"].astype(str)
long["id"] = long["id"].astype(str)
long["seed"] = pd.to_numeric(long["seed"], errors="raise").astype(int)

coverage = long.groupby("well")["seed"].agg(lambda values: tuple(sorted(set(values))))
complete_wells = coverage[coverage.map(lambda values: values == EXPECTED_SEEDS)].index
long = long.loc[long["well"].isin(complete_wells)].copy()
if len(complete_wells) < 20:
    raise RuntimeError(f"Only {len(complete_wells)} complete six-seed wells")
if long.duplicated(["well", "id", "seed"]).any():
    raise RuntimeError("Duplicate well/id/seed rows")

keys = ["well", "id", "md_since", "truth"]
wide = long.pivot(index=keys, columns="seed", values="prediction").sort_index()
if wide[list(EXPECTED_SEEDS)].isna().any().any():
    raise RuntimeError("Incomplete seed matrix after pivot")
base = wide.reset_index()[keys]
matrix = wide[list(EXPECTED_SEEDS)].to_numpy(np.float64)

predictions = {
    f"seed_{seed}": matrix[:, seed] for seed in EXPECTED_SEEDS
}
predictions["mean_6"] = np.mean(matrix, axis=1)
predictions["median_6"] = np.median(matrix, axis=1)
ordered = np.sort(matrix, axis=1)
predictions["trimmed_mean_4"] = np.mean(ordered[:, 1:-1], axis=1)

medoid = np.empty(len(base), np.float64)
medoid_seed_by_well = []
offset = 0
for well, local in base.groupby("well", sort=False):
    n_rows = len(local)
    local_matrix = matrix[offset:offset + n_rows].T
    distances = np.zeros((len(EXPECTED_SEEDS), len(EXPECTED_SEEDS)), np.float64)
    for left in range(len(EXPECTED_SEEDS)):
        for right in range(left + 1, len(EXPECTED_SEEDS)):
            distance = float(np.sqrt(np.mean(np.square(local_matrix[left] - local_matrix[right]))))
            distances[left, right] = distance
            distances[right, left] = distance
    centrality = distances.mean(axis=1)
    selected_position = int(np.argmin(centrality))
    selected_seed = EXPECTED_SEEDS[selected_position]
    medoid[offset:offset + n_rows] = local_matrix[selected_position]
    medoid_seed_by_well.append({
        "well": well,
        "rows": n_rows,
        "selected_seed": selected_seed,
        "centrality": float(centrality[selected_position]),
        "pairwise_rms_mean": float(distances[np.triu_indices(len(EXPECTED_SEEDS), 1)].mean()),
        "pairwise_rms_max": float(distances.max()),
    })
    offset += n_rows
predictions["wholewell_medoid_6"] = medoid

summary = pd.DataFrame([
    _metric(base, prediction, endpoint) for endpoint, prediction in predictions.items()
]).sort_values("rmse", ignore_index=True)
reference = predictions[f"seed_{REFERENCE_SEED}"]
candidate = predictions["wholewell_medoid_6"]
reference_metric = _metric(base, reference, f"seed_{REFERENCE_SEED}")
candidate_metric = _metric(base, candidate, "wholewell_medoid_6")

truth = base["truth"].to_numpy(np.float64)
row_report = pd.DataFrame({
    "well": base["well"].astype(str),
    "reference_e2": np.square(reference - truth),
    "candidate_e2": np.square(candidate - truth),
})
by_well = row_report.groupby("well", sort=False).agg(
    rows=("well", "size"),
    reference_sse=("reference_e2", "sum"),
    candidate_sse=("candidate_e2", "sum"),
).reset_index()
by_well["reference_rmse"] = np.sqrt(by_well["reference_sse"] / by_well["rows"])
by_well["candidate_rmse"] = np.sqrt(by_well["candidate_sse"] / by_well["rows"])
by_well["gain"] = by_well["reference_rmse"] - by_well["candidate_rmse"]
by_well = by_well.merge(pd.DataFrame(medoid_seed_by_well), on=["well", "rows"], validate="one_to_one")

gain = reference_metric["rmse"] - candidate_metric["rmse"]
far_gain = reference_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"]
positive_fraction = float(np.mean(by_well["gain"] > 0.0))
bootstrap = _bootstrap(by_well)
changed_fraction = float(np.mean(np.abs(candidate - reference) > 1e-12))
error_correlation = float(np.corrcoef(reference - truth, candidate - truth)[0, 1])

checks = {
    "complete_wells_ge_20": len(complete_wells) >= 20,
    "medoid_changes_rows": changed_fraction > 0.01,
    "gain_vs_seed1_ge_0p05": gain >= MIN_GAIN,
    "far_gain_vs_seed1_ge_0p05": far_gain >= MIN_FAR_GAIN,
    "positive_well_fraction_ge_0p55": positive_fraction >= MIN_POSITIVE_WELL_FRACTION,
    "bootstrap_probability_gain_ge_0p90": bootstrap["probability_gain_positive"] >= MIN_BOOTSTRAP_PROBABILITY,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "cached_six_seed_truth_free_wholewell_path_medoid_discovery",
    "scope": "learned trajectory core before CP8/Neighbor/V69/V76; no PF rerun",
    "source": SOURCE,
    "selection_rule": "one original complete seed path minimizing mean RMS distance to all seed paths",
    "truth_usage": "metrics only; medoid selection is truth-free",
    "reference_seed": REFERENCE_SEED,
    "seeds": list(EXPECTED_SEEDS),
    "coverage": {"rows": len(base), "wells": len(complete_wells)},
    "summary": summary.to_dict("records"),
    "reference": reference_metric,
    "candidate": candidate_metric,
    "gain": gain,
    "far_gain": far_gain,
    "positive_well_fraction": positive_fraction,
    "changed_row_fraction": changed_fraction,
    "error_correlation": error_correlation,
    "well_bootstrap": bootstrap,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> generate six-seed paths on a disjoint 48-well confirmation panel"
        if strict_pass else
        "FAIL -> seed-path mode is not useful; close PF aggregation and retain deterministic seed1"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
by_well.sort_values("gain").to_csv(OUT / "by_well.csv", index=False)
pd.DataFrame(medoid_seed_by_well).to_csv(OUT / "selected_seed_by_well.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("M342-A1 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-A1 MEDOID SEEDS")
print(pd.DataFrame(medoid_seed_by_well)["selected_seed"].value_counts().sort_index().to_string(), flush=True)
print("\nM342-A1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT, flush=True)
