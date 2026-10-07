"""V209: same-seed particle-count scaling in the frozen mainline PF core.

V197 averaged states from different seeds and therefore could average across
parallel geological modes.  V209 keeps seed 1 fixed and changes only the
number of ANCC/Z particles.  The frozen feature builder, LightGBM models, and
trajectory decoder are unchanged.

The experiment uses a 32-well discovery set and only opens a disjoint 64-well
confirmation set if a multiplier passes the pre-registered discovery gate.
Every well/multiplier prediction is checkpointed independently.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_variable, "6")

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / "mainline_pf_particle_count_gateway_v209"
CACHE = OUT / "well_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

SOURCE = ROOT / "mainline_pf_state_ensemble_gateway_v197_cell.py"
REFERENCE_SEED = 1
DISCOVERY_MULTIPLIERS = (1, 2, 4)
DISCOVERY_WELLS = 32
CONFIRMATION_WELLS = 64
MIN_COMPLETE_DISCOVERY = 30
MIN_COMPLETE_CONFIRMATION = 60


def _bootstrap():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = "_bootstrap_v197()\nselected_wells_v197"
    if marker not in source:
        raise RuntimeError(f"Cannot locate V197 bootstrap boundary in {SOURCE}")
    prefix = source.split(marker, 1)[0] + "_bootstrap_v197()\n"
    exec(compile(prefix, str(SOURCE), "exec"), globals())
    needed = {
        "CFG", "ANCC_N", "PF_N", "build_well", "run_pf_ancc", "run_pf_z",
        "_seed_numba_p5", "_stable_seed_p5", "_load_models_p5", "_likpf_rows",
        "add_likpf_features", "make_prediction", "_metrics_p5",
        "OOF_CANDIDATES_P5", "_prediction_columns_v197",
    }
    missing = sorted(needed.difference(globals()))
    if missing:
        raise RuntimeError(f"Incomplete V197 bootstrap: {missing}")


_bootstrap()

try:
    import numba

    numba.set_num_threads(6)
except Exception:
    pass


MODELS_DIR, FEATURE_NAMES, MODELS = _load_models_p5()
for _model in MODELS:
    try:
        _model.set_params(n_jobs=6, num_threads=6)
    except (AttributeError, TypeError, ValueError):
        pass

ORIGINAL_RUN_PF_ANCC = run_pf_ancc
ORIGINAL_RUN_PF_Z = run_pf_z
DEFAULT_ANCC_N = int(ANCC_N)
DEFAULT_PF_N = int(PF_N)
LIKPF_CACHE: dict[str, pd.DataFrame] = {}
ACTIVE_WELL: str | None = None
ACTIVE_MULTIPLIER = 1
CACHED_Z: tuple[np.ndarray, np.ndarray] | None = None


def _exclusion_paths() -> tuple[Path, ...]:
    return (
        OOF / "mainline_pf_repeatability_v97" / "paired_seed_predictions.parquet",
        OOF / "mainline_pf_seed1_confirmation_v98" / "paired_predictions.parquet",
        OOF / "mainline_pf_state_ensemble_gateway_v197" / "paired_predictions.parquet",
        OOF / "mainline_pf_state_ensemble_confirmation_v197b" / "paired_predictions.parquet",
    )


def _select_wells() -> tuple[list[str], list[str], str, list[str]]:
    excluded: set[str] = set()
    missing_paths = []
    for path in _exclusion_paths():
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        excluded.update(
            pd.read_parquet(path, columns=["well"])["well"].astype(str).unique()
        )
    if missing_paths:
        raise FileNotFoundError(
            "V209 requires frozen V97/V98/V197/V197b exclusion artifacts: "
            + repr(missing_paths)
        )
    if len(excluded) < 200:
        raise RuntimeError(f"Only {len(excluded)} unique prior-gateway wells were excluded")

    ranking_path = next((path for path in OOF_CANDIDATES_P5 if path.is_file()), None)
    if ranking_path is None:
        raise FileNotFoundError("No frozen difficulty-ranking OOF was found")
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
    discovery_counts = (11, 10, 11)
    confirmation_counts = (21, 22, 21)
    discovery: list[str] = []
    confirmation: list[str] = []
    rng = np.random.default_rng(209)
    for stratum, n_discovery, n_confirmation in zip(
        strata, discovery_counts, confirmation_counts
    ):
        required = n_discovery + n_confirmation
        if len(stratum) < required:
            raise RuntimeError(f"Only {len(stratum)} available wells in a V209 stratum")
        positions = np.linspace(0, len(stratum) - 1, required + 2)[1:-1]
        selected = np.asarray(
            [str(stratum[int(round(position))]) for position in positions], dtype=object
        )
        rng.shuffle(selected)
        discovery.extend(selected[:n_discovery].tolist())
        confirmation.extend(selected[n_discovery:].tolist())

    if len(discovery) != DISCOVERY_WELLS or len(set(discovery)) != DISCOVERY_WELLS:
        raise RuntimeError(f"Invalid discovery selection: {len(discovery)}/{len(set(discovery))}")
    if len(confirmation) != CONFIRMATION_WELLS or len(set(confirmation)) != CONFIRMATION_WELLS:
        raise RuntimeError(
            f"Invalid confirmation selection: {len(confirmation)}/{len(set(confirmation))}"
        )
    if set(discovery).intersection(confirmation) or set(discovery + confirmation).intersection(excluded):
        raise RuntimeError("V209 well partitions overlap")
    return sorted(discovery), sorted(confirmation), str(ranking_path), sorted(excluded)


DISCOVERY_SET, CONFIRMATION_SET, RANKING_SOURCE, EXCLUDED = _select_wells()


def _scaled_run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    """Run paired ANCC/Z states at one fixed seed and particle multiplier."""
    global CACHED_Z
    if ACTIVE_WELL is None:
        raise RuntimeError("V209 scaled PF called without an active well")
    _seed_numba_p5(_stable_seed_p5(ACTIVE_WELL, REFERENCE_SEED))
    ancc = ORIGINAL_RUN_PF_ANCC(
        hw, tw_tvt, tw_gr, N=DEFAULT_ANCC_N * ACTIVE_MULTIPLIER
    )
    CACHED_Z = ORIGINAL_RUN_PF_Z(
        hw, tw_tvt, tw_gr, N=DEFAULT_PF_N * ACTIVE_MULTIPLIER
    )
    return ancc


def _scaled_run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    global CACHED_Z
    if CACHED_Z is None:
        raise RuntimeError("V209 Z state was requested before the paired ANCC call")
    result = CACHED_Z
    CACHED_Z = None
    return result


def _predict(well: str, multiplier: int, original_reference: bool = False) -> pd.DataFrame:
    global ACTIVE_WELL, ACTIVE_MULTIPLIER, CACHED_Z, run_pf_ancc, run_pf_z
    train_dir = Path(CFG.DATA) / "train"
    horizontal = train_dir / f"{well}__horizontal_well.csv"
    typewell = train_dir / f"{well}__typewell.csv"
    ACTIVE_WELL = well
    ACTIVE_MULTIPLIER = int(multiplier)
    CACHED_Z = None
    if original_reference:
        run_pf_ancc = ORIGINAL_RUN_PF_ANCC
        run_pf_z = ORIGINAL_RUN_PF_Z
        _seed_numba_p5(_stable_seed_p5(well, REFERENCE_SEED))
    else:
        run_pf_ancc = _scaled_run_pf_ancc
        run_pf_z = _scaled_run_pf_z
    try:
        features = build_well(str(horizontal), str(typewell), True)
    finally:
        run_pf_ancc = ORIGINAL_RUN_PF_ANCC
        run_pf_z = ORIGINAL_RUN_PF_Z
        ACTIVE_WELL = None
        ACTIVE_MULTIPLIER = 1
        CACHED_Z = None
    if features is None or features.empty:
        raise RuntimeError(f"{well}: build_well returned no rows")
    if well not in LIKPF_CACHE:
        LIKPF_CACHE[well] = _likpf_rows(well, "train")
    features = add_likpf_features(features, LIKPF_CACHE[well]).reset_index(drop=True)
    missing = [column for column in FEATURE_NAMES if column not in features]
    if missing:
        features = pd.concat(
            [
                features,
                pd.DataFrame(0.0, index=features.index, columns=missing),
            ],
            axis=1,
        )
    matrix = features[FEATURE_NAMES].to_numpy(np.float32)
    model_delta = np.mean([model.predict(matrix) for model in MODELS], axis=0)
    prediction = make_prediction(features, model_delta, None).astype(np.float64)
    output = features[["well", "id", "md_since", "pf_ancc", "pf_ancc_std", "pf_z"]].copy()
    output["truth"] = (
        features["last_known_tvt"].to_numpy(np.float64)
        + features["target"].to_numpy(np.float64)
    )
    output["prediction"] = prediction
    output["variant"] = "original_seed1" if original_reference else f"particles_x{multiplier}"
    return output


def _cache_path(stage: str, well: str, multiplier: int) -> Path:
    directory = CACHE / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{well}__x{multiplier}.parquet"


def _load_or_predict(stage: str, well: str, multiplier: int) -> tuple[pd.DataFrame, bool]:
    path = _cache_path(stage, well, multiplier)
    expected_variant = f"particles_x{multiplier}"
    if path.is_file():
        frame = pd.read_parquet(path)
        if set(frame["variant"].astype(str).unique()) != {expected_variant}:
            raise RuntimeError(f"{well}: invalid cached variant in {path}")
        return frame, True
    frame = _predict(well, multiplier)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return frame, False


def _run_stage(stage: str, wells: list[str], multipliers: tuple[int, ...]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    resumed = 0
    calls = 0
    for number, well in enumerate(wells, 1):
        local: list[pd.DataFrame] = []
        try:
            for multiplier in multipliers:
                prediction, was_resumed = _load_or_predict(stage, well, multiplier)
                local.append(prediction)
                resumed += int(was_resumed)
                calls += 1
            keys = ["well", "id", "md_since", "truth"]
            reference_keys = local[0][keys]
            if any(not reference_keys.equals(candidate[keys]) for candidate in local[1:]):
                raise RuntimeError(f"{well}: multiplier row alignment differs")
            parts.extend(local)
        except Exception as exc:
            failures.append({"stage": stage, "well": well, "reason": repr(exc)})
        if number % 2 == 0 or number == len(wells):
            print(
                f"V209 {stage} {number}/{len(wells)} calls={calls} "
                f"resumed={resumed} failures={len(failures)} "
                f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
                flush=True,
            )
    if failures:
        pd.DataFrame(failures).to_csv(OUT / f"{stage}_failures.csv", index=False)
    if not parts:
        raise RuntimeError(f"No V209 {stage} predictions: {failures[:5]}")
    result = pd.concat(parts, ignore_index=True)
    coverage = result.groupby("well")["variant"].nunique()
    complete = coverage[coverage.eq(len(multipliers))].index
    result = result[result["well"].isin(complete)].copy()
    return result


def _well_bootstrap(by_well: pd.DataFrame, draws: int = 5000) -> dict[str, float | int]:
    rng = np.random.default_rng(20260804)
    rows = by_well["rows"].to_numpy(np.float64)
    reference_sse = by_well["reference_sse"].to_numpy(np.float64)
    candidate_sse = by_well["candidate_sse"].to_numpy(np.float64)
    gains = np.empty(draws, np.float64)
    for draw in range(draws):
        index = rng.integers(0, len(by_well), len(by_well))
        gains[draw] = (
            np.sqrt(reference_sse[index].sum() / rows[index].sum())
            - np.sqrt(candidate_sse[index].sum() / rows[index].sum())
        )
    return {
        "draws": draws,
        "gain_p05": float(np.quantile(gains, 0.05)),
        "gain_median": float(np.median(gains)),
        "gain_p95": float(np.quantile(gains, 0.95)),
        "probability_gain_positive": float(np.mean(gains > 0.0)),
    }


def _evaluate(long: pd.DataFrame, multipliers: tuple[int, ...], stage: str):
    keys = ["well", "id", "md_since", "truth"]
    prediction = long.pivot(index=keys, columns="variant", values="prediction").sort_index()
    base = prediction.reset_index()[keys]
    reference = prediction["particles_x1"].to_numpy(np.float64)
    reference_metric = _metrics_p5(base, reference, "particles_x1")
    metrics = [reference_metric]
    comparisons = []
    truth = base["truth"].to_numpy(np.float64)
    for multiplier in multipliers:
        if multiplier == 1:
            continue
        name = f"particles_x{multiplier}"
        candidate = prediction[name].to_numpy(np.float64)
        candidate_metric = _metrics_p5(base, candidate, name)
        metrics.append(candidate_metric)
        rows = pd.DataFrame({
            "well": base["well"].astype(str).to_numpy(),
            "reference_e2": np.square(reference - truth),
            "candidate_e2": np.square(candidate - truth),
        })
        by_well = rows.groupby("well").agg(
            rows=("well", "size"),
            reference_sse=("reference_e2", "sum"),
            candidate_sse=("candidate_e2", "sum"),
        )
        by_well["reference_rmse"] = np.sqrt(by_well["reference_sse"] / by_well["rows"])
        by_well["candidate_rmse"] = np.sqrt(by_well["candidate_sse"] / by_well["rows"])
        by_well["gain"] = by_well["reference_rmse"] - by_well["candidate_rmse"]
        by_well = by_well.reset_index()
        bootstrap = _well_bootstrap(by_well)
        comparison = {
            "stage": stage,
            "multiplier": multiplier,
            "reference_rmse": float(reference_metric["rmse"]),
            "candidate_rmse": float(candidate_metric["rmse"]),
            "gain": float(reference_metric["rmse"] - candidate_metric["rmse"]),
            "far_gain": float(
                reference_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"]
            ),
            "p95_gain": float(
                reference_metric["p95_well_rmse"] - candidate_metric["p95_well_rmse"]
            ),
            "positive_well_fraction": float(np.mean(by_well["gain"] > 0.0)),
            "bootstrap": bootstrap,
        }
        comparisons.append(comparison)
        by_well.sort_values("gain").to_csv(
            OUT / f"{stage}_x{multiplier}_by_well.csv", index=False
        )
    summary = pd.DataFrame(metrics)
    summary.to_csv(OUT / f"{stage}_summary.csv", index=False)
    pd.DataFrame([
        {**item, **{f"bootstrap_{key}": value for key, value in item["bootstrap"].items()}}
        for item in comparisons
    ]).drop(columns="bootstrap").to_csv(OUT / f"{stage}_comparisons.csv", index=False)
    return summary, comparisons


print(
    "V209 CONFIG",
    json.dumps(
        {
            "default_ancc_particles": DEFAULT_ANCC_N,
            "default_z_particles": DEFAULT_PF_N,
            "multipliers": list(DISCOVERY_MULTIPLIERS),
            "discovery_wells": len(DISCOVERY_SET),
            "confirmation_wells_reserved": len(CONFIRMATION_SET),
            "excluded_prior_gateway_wells": len(EXCLUDED),
            "ranking_source": RANKING_SOURCE,
            "models_dir": str(MODELS_DIR),
        },
        indent=2,
    ),
    flush=True,
)

# Verify that the x1 paired wrapper is exactly the original seed-1 execution.
parity_well = DISCOVERY_SET[0]
parity_original = _predict(parity_well, 1, original_reference=True)
parity_scaled, _ = _load_or_predict("discovery", parity_well, 1)
parity_keys = ["well", "id", "md_since", "truth"]
if not parity_original[parity_keys].equals(parity_scaled[parity_keys]):
    raise RuntimeError("V209 x1 parity rows differ")
parity_prediction_gap = float(np.max(np.abs(
    parity_original["prediction"].to_numpy(np.float64)
    - parity_scaled["prediction"].to_numpy(np.float64)
)))
parity_state_gap = float(np.max(np.abs(
    parity_original["pf_ancc"].to_numpy(np.float64)
    - parity_scaled["pf_ancc"].to_numpy(np.float64)
)))
if parity_prediction_gap > 1e-5 or parity_state_gap > 1e-5:
    raise RuntimeError(
        f"V209 x1 parity failed: prediction={parity_prediction_gap}, state={parity_state_gap}"
    )
print(
    "V209 X1 PARITY",
    {"well": parity_well, "prediction_max_gap": parity_prediction_gap, "state_max_gap": parity_state_gap},
    flush=True,
)

discovery_long = _run_stage("discovery", DISCOVERY_SET, DISCOVERY_MULTIPLIERS)
discovery_complete = discovery_long.groupby("well")["variant"].nunique().eq(3).sum()
if discovery_complete < MIN_COMPLETE_DISCOVERY:
    raise RuntimeError(f"Only {discovery_complete} complete discovery wells")
discovery_summary, discovery_comparisons = _evaluate(
    discovery_long, DISCOVERY_MULTIPLIERS, "discovery"
)

for item in discovery_comparisons:
    item["passes_discovery"] = bool(
        item["gain"] >= 0.05
        and item["far_gain"] > 0.0
        and item["p95_gain"] >= -0.05
        and item["bootstrap"]["probability_gain_positive"] >= 0.80
    )
eligible = [item for item in discovery_comparisons if item["passes_discovery"]]
selected = max(eligible, key=lambda item: item["gain"]) if eligible else None

confirmation_summary = None
confirmation_comparison = None
if selected is not None:
    selected_multiplier = int(selected["multiplier"])
    confirmation_long = _run_stage(
        "confirmation", CONFIRMATION_SET, (1, selected_multiplier)
    )
    confirmation_complete = (
        confirmation_long.groupby("well")["variant"].nunique().eq(2).sum()
    )
    if confirmation_complete < MIN_COMPLETE_CONFIRMATION:
        raise RuntimeError(f"Only {confirmation_complete} complete confirmation wells")
    confirmation_summary, confirmation_comparisons = _evaluate(
        confirmation_long, (1, selected_multiplier), "confirmation"
    )
    confirmation_comparison = confirmation_comparisons[0]
    confirmation_comparison["passes_confirmation"] = bool(
        confirmation_comparison["gain"] >= 0.05
        and confirmation_comparison["far_gain"] > 0.0
        and confirmation_comparison["p95_gain"] >= 0.0
        and confirmation_comparison["bootstrap"]["probability_gain_positive"] >= 0.90
    )

strict_pass = bool(
    selected is not None
    and confirmation_comparison is not None
    and confirmation_comparison["passes_confirmation"]
)
verdict = {
    "experiment": "mainline_pf_particle_count_gateway_v209",
    "stage": (
        "same_seed_particle_scaling_discovery_and_confirmation"
        if selected is not None else "same_seed_particle_scaling_discovery_fail_fast"
    ),
    "changed_variable": "ANCC and Z particle counts only",
    "unchanged": [
        "seed 1", "paired ANCC/Z random stream", "PF parameters",
        "feature generation", "frozen learned-core models", "trajectory decoder",
    ],
    "scope": "learned trajectory core only; not CP8, Neighbor, V69, or V77",
    "default_particles": {"ancc": DEFAULT_ANCC_N, "z": DEFAULT_PF_N},
    "discovery_multipliers": list(DISCOVERY_MULTIPLIERS),
    "protocol": {
        "discovery_wells": DISCOVERY_WELLS,
        "confirmation_wells_reserved": CONFIRMATION_WELLS,
        "prior_gateway_wells_excluded": len(EXCLUDED),
        "confirmation_opened": selected is not None,
    },
    "x1_parity": {
        "well": parity_well,
        "prediction_max_gap": parity_prediction_gap,
        "pf_ancc_max_gap": parity_state_gap,
    },
    "discovery": discovery_comparisons,
    "selected_multiplier": None if selected is None else int(selected["multiplier"]),
    "confirmation": confirmation_comparison,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> propagate the confirmed particle multiplier through the final V69/V77 chain"
        if strict_pass else
        "FAIL -> particle count is not the missing mainline mechanism; retain seed-1 PF"
    ),
}

discovery_long.to_parquet(OUT / "discovery_predictions.parquet", index=False, compression="zstd")
if confirmation_summary is not None:
    confirmation_long.to_parquet(
        OUT / "confirmation_predictions.parquet", index=False, compression="zstd"
    )
(OUT / "well_partitions.json").write_text(
    json.dumps({"discovery": DISCOVERY_SET, "confirmation": CONFIRMATION_SET}, indent=2),
    encoding="utf-8",
)
(OUT / "result.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nV209 DISCOVERY SUMMARY")
print(discovery_summary.to_string(index=False), flush=True)
if confirmation_summary is not None:
    print("\nV209 CONFIRMATION SUMMARY")
    print(confirmation_summary.to_string(index=False), flush=True)
print("\nV209 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT, flush=True)
