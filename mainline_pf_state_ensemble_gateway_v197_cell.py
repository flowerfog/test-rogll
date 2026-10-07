"""V197: disjoint gateway for PF-state aggregation before feature generation.

V97 averaged final learned-core predictions from independently built seeds.
V197 instead averages paired ANCC/Z particle-filter states first, then lets the
unchanged mainline ``build_well`` regenerate every PF-derived feature before
the frozen LightGBM ensemble.  The 48 gateway wells exclude all V97 and V98
wells.  This experiment stops before CP8, Neighbor, V69, and V77.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED_V197 = time.time()
ROOT_V197 = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_V197 = ROOT_V197 / "oof_exports"
OUT_V197 = OOF_V197 / "mainline_pf_state_ensemble_gateway_v197"
OUT_V197.mkdir(parents=True, exist_ok=True)
V97_SOURCE_V197 = ROOT_V197 / "mainline_pf_repeatability_v97_cell.py"
V97_DIR_V197 = OOF_V197 / "mainline_pf_repeatability_v97"
V98_DIR_V197 = OOF_V197 / "mainline_pf_seed1_confirmation_v98"

REFERENCE_SEED_V197 = 1
STATE_SEEDS_V197 = (0, 1, 2, 3)
N_WELLS_V197 = 48


def _bootstrap_v197():
    needed = {
        "CFG", "build_well", "run_pf_ancc", "run_pf_z", "_seed_numba_p5",
        "_stable_seed_p5", "_load_models_p5", "_likpf_rows",
        "add_likpf_features", "make_prediction", "_metrics_p5",
        "OOF_CANDIDATES_P5",
    }
    if not needed.difference(globals()):
        return
    if not V97_SOURCE_V197.is_file():
        raise FileNotFoundError(V97_SOURCE_V197)
    source = V97_SOURCE_V197.read_text(encoding="utf-8")
    marker = "started_p5 = time.time()"
    if marker not in source:
        raise RuntimeError(f"Cannot locate V97 execution boundary in {V97_SOURCE_V197}")
    exec(compile(source.split(marker, 1)[0] + "\n", str(V97_SOURCE_V197), "exec"), globals())


def _prediction_columns_v197(path: Path) -> tuple[str, str]:
    columns = pd.read_parquet(path).columns.tolist()
    truth = next((name for name in ("tvt_true", "truth", "TVT") if name in columns), None)
    prediction = next(
        (
            name for name in (
                "incumbent_6p377_replay", "full_parent_plus_raw_cp_crossfit",
                "full_cp8_ordered", "full_parent_matched_vp_replay", "core",
            ) if name in columns
        ),
        None,
    )
    if truth is None or prediction is None or "well" not in columns:
        raise RuntimeError(f"Cannot rank wells from {path}: {columns}")
    return truth, prediction


def _select_wells_v197() -> tuple[list[str], str, list[str]]:
    exclusion_paths = (
        V97_DIR_V197 / "paired_seed_predictions.parquet",
        V98_DIR_V197 / "paired_predictions.parquet",
    )
    for path in exclusion_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"V197 requires the frozen V97/V98 exclusion artifact: {path}"
            )
    excluded: set[str] = set()
    for path in exclusion_paths:
        excluded.update(pd.read_parquet(path, columns=["well"])["well"].astype(str).unique())
    if len(excluded) < 68:
        raise RuntimeError(f"Only {len(excluded)} unique V97/V98 exclusion wells")

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
    groups = np.array_split(difficulty.index.to_numpy(), 3)
    per_group = N_WELLS_V197 // 3
    selected: list[str] = []
    for group in groups:
        if len(group) < per_group + 2:
            raise RuntimeError(f"Only {len(group)} wells in a V197 difficulty stratum")
        positions = np.linspace(0, len(group) - 1, per_group + 2)[1:-1]
        selected.extend(str(group[int(round(position))]) for position in positions)
    if len(selected) != N_WELLS_V197 or len(set(selected)) != N_WELLS_V197:
        raise RuntimeError(f"Invalid V197 selection: {len(selected)}/{len(set(selected))}")
    if set(selected).intersection(excluded):
        raise RuntimeError("V197 selected a V97/V98 well")
    return sorted(selected), str(ranking_path), sorted(excluded)


def _total_spread_v197(predictions: list[np.ndarray], spreads: list[np.ndarray]) -> np.ndarray:
    prediction_matrix = np.stack(predictions).astype(np.float64)
    spread_matrix = np.stack(spreads).astype(np.float64)
    centre = prediction_matrix.mean(axis=0)
    variance = np.mean(np.square(spread_matrix) + np.square(prediction_matrix - centre), axis=0)
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def _well_bootstrap_v197(by_well: pd.DataFrame, draws: int = 5000) -> dict[str, float | int]:
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


_bootstrap_v197()
selected_wells_v197, ranking_source_v197, excluded_wells_v197 = _select_wells_v197()
models_dir_v197, feature_names_v197, models_v197 = _load_models_p5()
likpf_cache_v197: dict[str, pd.DataFrame] = {}

original_run_pf_ancc_v197 = run_pf_ancc
original_run_pf_z_v197 = run_pf_z
active_well_v197: str | None = None
cached_z_v197: tuple[np.ndarray, np.ndarray] | None = None


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    """Aggregate paired ANCC/Z states before build_well derives features."""
    global cached_z_v197
    if active_well_v197 is None:
        raise RuntimeError("V197 aggregate PF called without an active well")
    ancc_predictions, ancc_spreads = [], []
    z_predictions, z_spreads = [], []
    for seed in STATE_SEEDS_V197:
        _seed_numba_p5(_stable_seed_p5(active_well_v197, seed))
        ancc_prediction, ancc_spread = original_run_pf_ancc_v197(hw, tw_tvt, tw_gr, N=N)
        z_prediction, z_spread = original_run_pf_z_v197(hw, tw_tvt, tw_gr)
        if not (
            len(ancc_prediction) == len(ancc_spread) == len(z_prediction) == len(z_spread)
        ):
            raise RuntimeError(f"{active_well_v197}: inconsistent PF state lengths")
        ancc_predictions.append(np.asarray(ancc_prediction, np.float32))
        ancc_spreads.append(np.asarray(ancc_spread, np.float32))
        z_predictions.append(np.asarray(z_prediction, np.float32))
        z_spreads.append(np.asarray(z_spread, np.float32))
    ancc_centre = np.mean(np.stack(ancc_predictions), axis=0).astype(np.float32)
    z_centre = np.mean(np.stack(z_predictions), axis=0).astype(np.float32)
    cached_z_v197 = (z_centre, _total_spread_v197(z_predictions, z_spreads))
    return ancc_centre, _total_spread_v197(ancc_predictions, ancc_spreads)


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    global cached_z_v197
    if cached_z_v197 is None:
        raise RuntimeError("V197 Z aggregate was requested before ANCC")
    result = cached_z_v197
    cached_z_v197 = None
    return result


aggregate_run_pf_ancc_v197 = run_pf_ancc
aggregate_run_pf_z_v197 = run_pf_z


def _predict_v197(well: str, aggregate: bool) -> pd.DataFrame:
    global active_well_v197, cached_z_v197, run_pf_ancc, run_pf_z
    train_dir = Path(CFG.DATA) / "train"
    horizontal = train_dir / f"{well}__horizontal_well.csv"
    typewell = train_dir / f"{well}__typewell.csv"
    cached_z_v197 = None
    active_well_v197 = well
    if aggregate:
        run_pf_ancc = aggregate_run_pf_ancc_v197
        run_pf_z = aggregate_run_pf_z_v197
    else:
        run_pf_ancc = original_run_pf_ancc_v197
        run_pf_z = original_run_pf_z_v197
        _seed_numba_p5(_stable_seed_p5(well, REFERENCE_SEED_V197))
    try:
        features = build_well(str(horizontal), str(typewell), True)
    finally:
        run_pf_ancc = original_run_pf_ancc_v197
        run_pf_z = original_run_pf_z_v197
        active_well_v197 = None
        cached_z_v197 = None
    if features is None or features.empty:
        raise RuntimeError(f"{well}: build_well returned no rows")
    if well not in likpf_cache_v197:
        likpf_cache_v197[well] = _likpf_rows(well, "train")
    features = add_likpf_features(features, likpf_cache_v197[well]).reset_index(drop=True)
    for column in feature_names_v197:
        if column not in features:
            features[column] = 0.0
    matrix = features[feature_names_v197].to_numpy(np.float32)
    model_delta = np.mean([model.predict(matrix) for model in models_v197], axis=0)
    prediction = make_prediction(features, model_delta, None).astype(np.float64)
    output = features[["well", "id", "md_since", "pf_ancc", "pf_ancc_std", "pf_z"]].copy()
    output["truth"] = (
        features["last_known_tvt"].to_numpy(np.float64)
        + features["target"].to_numpy(np.float64)
    )
    output["prediction"] = prediction
    output["variant"] = "state_mean4" if aggregate else "seed1_reference"
    return output


parts_v197: list[pd.DataFrame] = []
failures_v197: list[dict[str, str]] = []
for number_v197, well_v197 in enumerate(selected_wells_v197, 1):
    try:
        reference_v197 = _predict_v197(well_v197, aggregate=False)
        candidate_v197 = _predict_v197(well_v197, aggregate=True)
        keys_v197 = ["well", "id", "md_since", "truth"]
        if not reference_v197[keys_v197].equals(candidate_v197[keys_v197]):
            raise RuntimeError(f"{well_v197}: reference/candidate row alignment differs")
        parts_v197.extend((reference_v197, candidate_v197))
    except Exception as exc_v197:
        failures_v197.append({"well": well_v197, "reason": repr(exc_v197)})
    if number_v197 % 4 == 0 or number_v197 == len(selected_wells_v197):
        print(
            f"V197 gateway {number_v197}/{len(selected_wells_v197)} "
            f"failures={len(failures_v197)} "
            f"elapsed_min={(time.time() - STARTED_V197) / 60.0:.1f}",
            flush=True,
        )

if failures_v197:
    pd.DataFrame(failures_v197).to_csv(OUT_V197 / "failures.csv", index=False)
if not parts_v197:
    raise RuntimeError(f"No V197 predictions: {failures_v197[:5]}")

long_v197 = pd.concat(parts_v197, ignore_index=True)
coverage_v197 = long_v197.groupby("well")["variant"].nunique()
complete_wells_v197 = coverage_v197[coverage_v197.eq(2)].index
long_v197 = long_v197[long_v197["well"].isin(complete_wells_v197)].copy()
if len(complete_wells_v197) < 44:
    raise RuntimeError(f"Only {len(complete_wells_v197)} complete V197 wells")

keys_v197 = ["well", "id", "md_since", "truth"]
wide_prediction_v197 = long_v197.pivot(
    index=keys_v197, columns="variant", values="prediction"
).sort_index()
base_v197 = wide_prediction_v197.reset_index()[keys_v197]
reference_prediction_v197 = wide_prediction_v197["seed1_reference"].to_numpy(np.float64)
candidate_prediction_v197 = wide_prediction_v197["state_mean4"].to_numpy(np.float64)
reference_metric_v197 = _metrics_p5(base_v197, reference_prediction_v197, "seed1_reference")
candidate_metric_v197 = _metrics_p5(base_v197, candidate_prediction_v197, "state_mean4")
summary_v197 = pd.DataFrame([reference_metric_v197, candidate_metric_v197])

truth_v197 = base_v197["truth"].to_numpy(np.float64)
row_report_v197 = pd.DataFrame({
    "well": base_v197["well"].astype(str).to_numpy(),
    "reference_e2": np.square(reference_prediction_v197 - truth_v197),
    "candidate_e2": np.square(candidate_prediction_v197 - truth_v197),
})
by_well_v197 = row_report_v197.groupby("well").agg(
    rows=("well", "size"),
    reference_sse=("reference_e2", "sum"),
    candidate_sse=("candidate_e2", "sum"),
)
by_well_v197["reference_rmse"] = np.sqrt(by_well_v197["reference_sse"] / by_well_v197["rows"])
by_well_v197["candidate_rmse"] = np.sqrt(by_well_v197["candidate_sse"] / by_well_v197["rows"])
by_well_v197["gain"] = by_well_v197["reference_rmse"] - by_well_v197["candidate_rmse"]
by_well_v197 = by_well_v197.reset_index()

gain_v197 = float(reference_metric_v197["rmse"] - candidate_metric_v197["rmse"])
far_gain_v197 = float(
    reference_metric_v197["far_gt2000_rmse"] - candidate_metric_v197["far_gt2000_rmse"]
)
positive_fraction_v197 = float(np.mean(by_well_v197["gain"] > 0.0))
bootstrap_v197 = _well_bootstrap_v197(by_well_v197)

state_wide_v197 = long_v197.pivot(
    index=keys_v197, columns="variant", values="pf_ancc"
).sort_index()
state_rms_change_v197 = float(np.sqrt(np.mean(np.square(
    state_wide_v197["state_mean4"].to_numpy(np.float64)
    - state_wide_v197["seed1_reference"].to_numpy(np.float64)
))))
checks_v197 = {
    "complete_wells_ge_44": len(complete_wells_v197) >= 44,
    "state_rms_change_ge_0p10": state_rms_change_v197 >= 0.10,
    "gain_ge_0p08": gain_v197 >= 0.08,
    "far_gain_positive": far_gain_v197 > 0.0,
    "positive_well_fraction_ge_0p55": positive_fraction_v197 >= 0.55,
    "bootstrap_probability_gain_ge_0p90": bootstrap_v197["probability_gain_positive"] >= 0.90,
}
strict_pass_v197 = bool(all(checks_v197.values()))
verdict_v197 = {
    "experiment": "mainline_pf_state_ensemble_gateway_v197",
    "stage": "disjoint_48well_pre_feature_PF_state_aggregation_gateway",
    "changed_variable": (
        "mean four paired ANCC/Z PF states and total uncertainty before all "
        "PF-derived features and frozen learned-core inference"
    ),
    "scope": "learned trajectory core only; not CP8, Neighbor, V69, or V77",
    "reference_seed": REFERENCE_SEED_V197,
    "state_seeds": list(STATE_SEEDS_V197),
    "selection_or_tuning_count": 0,
    "gateway_wells_requested": N_WELLS_V197,
    "gateway_wells_complete": int(len(complete_wells_v197)),
    "excluded_v97_v98_wells": int(len(excluded_wells_v197)),
    "ranking_source": ranking_source_v197,
    "models_dir": str(models_dir_v197),
    "reference": reference_metric_v197,
    "candidate": candidate_metric_v197,
    "gain": gain_v197,
    "far_gain": far_gain_v197,
    "positive_well_fraction": positive_fraction_v197,
    "pf_ancc_state_rms_change": state_rms_change_v197,
    "well_bootstrap": bootstrap_v197,
    "checks": checks_v197,
    "strict_pass": strict_pass_v197,
    "elapsed_minutes": (time.time() - STARTED_V197) / 60.0,
    "next": (
        "PASS -> test exact state aggregation through the final V69/V77 lineage"
        if strict_pass_v197 else
        "FAIL -> state aggregation is not strong enough; do not build a Kaggle submission"
    ),
}

summary_v197.to_csv(OUT_V197 / "summary.csv", index=False)
by_well_v197.sort_values("gain").to_csv(OUT_V197 / "by_well.csv", index=False)
long_v197.to_parquet(OUT_V197 / "paired_predictions.parquet", index=False, compression="zstd")
(OUT_V197 / "result.json").write_text(json.dumps(verdict_v197, indent=2), encoding="utf-8")
print("\nV197 SUMMARY")
print(summary_v197.to_string(index=False), flush=True)
print("\nV197 FINAL VERDICT")
print(json.dumps(verdict_v197, indent=2), flush=True)
print("saved:", OUT_V197, flush=True)
