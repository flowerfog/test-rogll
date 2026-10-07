"""V98: disjoint confirmation of the seed-1 result discovered by V97.

V97 selected seed 1 against seed 0 on 24 wells. This cell freezes that
comparison and evaluates 48 different difficulty-stratified wells. No other
seed, averaging rule, or downstream prediction layer is searched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED_V98 = time.time()
ROOT_V98 = Path("/root/autodl-tmp")
P5_DIR_V98 = ROOT_V98 / "oof_exports" / "mainline_pf_repeatability_v97"
P5_SOURCE_V98 = ROOT_V98 / "mainline_pf_repeatability_v97_cell.py"
OUT_DIR_V98 = ROOT_V98 / "oof_exports" / "mainline_pf_seed1_confirmation_v98"
OUT_DIR_V98.mkdir(parents=True, exist_ok=True)
N_WELLS_V98 = 48
REFERENCE_SEED_V98 = 0
FROZEN_SEED_V98 = 1


def _bootstrap_v98():
    needed = {
        "_one_prediction_p5", "_load_models_p5", "_metrics_p5",
        "CFG", "OOF_CANDIDATES_P5",
    }
    if not needed.difference(globals()):
        return
    if not P5_SOURCE_V98.is_file():
        raise FileNotFoundError(P5_SOURCE_V98)
    source = P5_SOURCE_V98.read_text(encoding="utf-8")
    marker = "started_p5 = time.time()"
    if marker not in source:
        raise RuntimeError(f"Cannot identify V97 execution boundary in {P5_SOURCE_V98}")
    exec(compile(source.split(marker, 1)[0] + "\n", str(P5_SOURCE_V98), "exec"), globals())


def _prediction_columns(path):
    columns = pd.read_parquet(path).columns.tolist()
    truth = next((column for column in ("tvt_true", "truth", "TVT") if column in columns), None)
    prediction = next(
        (
            column
            for column in (
                "incumbent_6p377_replay",
                "full_parent_plus_raw_cp_crossfit",
                "full_cp8_ordered",
                "full_parent_matched_vp_replay",
                "core",
            )
            if column in columns
        ),
        None,
    )
    if truth is None or prediction is None or "well" not in columns:
        raise RuntimeError(f"Cannot identify ranking columns in {path}: {columns}")
    return truth, prediction


def _confirmation_wells(excluded):
    result_path = P5_DIR_V98 / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ranking_source = result.get("ranking_source")
    oof_path = Path(ranking_source) if ranking_source else next(
        (path for path in OOF_CANDIDATES_P5 if path.exists()), None
    )
    if oof_path is None or not oof_path.is_file():
        raise FileNotFoundError(f"Cannot find V98 ranking OOF: {ranking_source}")
    truth_column, prediction_column = _prediction_columns(oof_path)
    frame = pd.read_parquet(
        oof_path, columns=["well", truth_column, prediction_column]
    )
    frame["well"] = frame["well"].astype(str)
    frame = frame.loc[~frame["well"].isin(excluded)].copy()
    frame["e2"] = (
        frame[prediction_column].to_numpy(np.float64)
        - frame[truth_column].to_numpy(np.float64)
    ) ** 2
    difficulty = frame.groupby("well")["e2"].mean().pow(0.5).sort_values()
    train_wells = {
        path.stem.replace("__horizontal_well", "")
        for path in (Path(CFG.DATA) / "train").glob("*__horizontal_well.csv")
    }
    difficulty = difficulty[difficulty.index.isin(train_wells)]
    groups = np.array_split(difficulty.index.to_numpy(), 3)
    per_group = N_WELLS_V98 // 3
    selected = []
    for group in groups:
        if len(group) < per_group:
            raise RuntimeError(f"Only {len(group)} wells in a V98 difficulty stratum")
        positions = np.linspace(0, len(group) - 1, per_group + 2)[1:-1]
        selected.extend(str(group[int(round(position))]) for position in positions)
    if len(selected) != N_WELLS_V98 or len(set(selected)) != N_WELLS_V98:
        raise RuntimeError(f"Invalid V98 confirmation selection: {len(selected)}/{len(set(selected))}")
    return sorted(selected), str(oof_path)


def _well_metrics(base, reference, candidate):
    local = pd.DataFrame(
        {
            "well": base["well"].astype(str).to_numpy(),
            "reference_e2": np.square(reference - base["truth"].to_numpy(np.float64)),
            "candidate_e2": np.square(candidate - base["truth"].to_numpy(np.float64)),
        }
    )
    grouped = local.groupby("well").agg(
        rows=("well", "size"),
        reference_sse=("reference_e2", "sum"),
        candidate_sse=("candidate_e2", "sum"),
    )
    grouped["reference_rmse"] = np.sqrt(grouped["reference_sse"] / grouped["rows"])
    grouped["candidate_rmse"] = np.sqrt(grouped["candidate_sse"] / grouped["rows"])
    grouped["gain"] = grouped["reference_rmse"] - grouped["candidate_rmse"]
    return grouped.reset_index()


def _well_bootstrap_gain(by_well, draws=5000):
    rng = np.random.default_rng(20260731)
    rows = by_well["rows"].to_numpy(np.float64)
    ref = by_well["reference_sse"].to_numpy(np.float64)
    cand = by_well["candidate_sse"].to_numpy(np.float64)
    gains = np.empty(draws, np.float64)
    for draw in range(draws):
        index = rng.integers(0, len(by_well), len(by_well))
        gains[draw] = (
            np.sqrt(ref[index].sum() / rows[index].sum())
            - np.sqrt(cand[index].sum() / rows[index].sum())
        )
    return {
        "draws": int(draws),
        "gain_p05": float(np.quantile(gains, 0.05)),
        "gain_median": float(np.median(gains)),
        "gain_p95": float(np.quantile(gains, 0.95)),
        "probability_gain_positive": float(np.mean(gains > 0.0)),
    }


_bootstrap_v98()
paired_path_v98 = P5_DIR_V98 / "paired_seed_predictions.parquet"
if not paired_path_v98.is_file():
    raise FileNotFoundError(paired_path_v98)
excluded_wells_v98 = set(
    pd.read_parquet(paired_path_v98, columns=["well"])["well"].astype(str).unique()
)
confirmation_wells_v98, ranking_source_v98 = _confirmation_wells(excluded_wells_v98)
models_dir_v98, feature_names_v98, models_v98 = _load_models_p5()
likpf_cache_v98 = {}

parts_v98 = []
failures_v98 = []
for number_v98, well_v98 in enumerate(confirmation_wells_v98, 1):
    for seed_v98 in (REFERENCE_SEED_V98, FROZEN_SEED_V98):
        try:
            parts_v98.append(
                _one_prediction_p5(
                    well_v98,
                    seed_v98,
                    feature_names_v98,
                    models_v98,
                    likpf_cache_v98,
                )
            )
        except Exception as exc_v98:
            failures_v98.append(
                {"well": well_v98, "seed": seed_v98, "reason": repr(exc_v98)}
            )
    if number_v98 % 8 == 0 or number_v98 == len(confirmation_wells_v98):
        print(
            f"V98 confirmation {number_v98}/{len(confirmation_wells_v98)} "
            f"failures={len(failures_v98)}",
            flush=True,
        )

if failures_v98:
    pd.DataFrame(failures_v98).to_csv(OUT_DIR_V98 / "failures.csv", index=False)
if not parts_v98:
    raise RuntimeError(f"No V98 predictions: {failures_v98[:5]}")

long_v98 = pd.concat(parts_v98, ignore_index=True)
coverage_v98 = long_v98.groupby("well")["seed"].nunique()
complete_wells_v98 = coverage_v98[coverage_v98.eq(2)].index
long_v98 = long_v98[long_v98["well"].isin(complete_wells_v98)].copy()
if len(complete_wells_v98) < 44:
    raise RuntimeError(f"Only {len(complete_wells_v98)} complete V98 wells")

keys_v98 = ["well", "id", "md_since", "truth"]
wide_v98 = long_v98.pivot(
    index=keys_v98, columns="seed", values="prediction"
).sort_index()
base_v98 = wide_v98.reset_index()[keys_v98]
reference_v98 = wide_v98[REFERENCE_SEED_V98].to_numpy(np.float64)
candidate_v98 = wide_v98[FROZEN_SEED_V98].to_numpy(np.float64)

reference_metrics_v98 = _metrics_p5(base_v98, reference_v98, "seed_0_reference")
candidate_metrics_v98 = _metrics_p5(base_v98, candidate_v98, "seed_1_frozen")
summary_v98 = pd.DataFrame([reference_metrics_v98, candidate_metrics_v98])
by_well_v98 = _well_metrics(base_v98, reference_v98, candidate_v98)
bootstrap_v98 = _well_bootstrap_gain(by_well_v98)

gain_v98 = float(reference_metrics_v98["rmse"] - candidate_metrics_v98["rmse"])
far_gain_v98 = float(
    reference_metrics_v98["far_gt2000_rmse"]
    - candidate_metrics_v98["far_gt2000_rmse"]
)
positive_well_fraction_v98 = float(np.mean(by_well_v98["gain"] > 0.0))
checks_v98 = {
    "complete_wells_ge_44": int(len(complete_wells_v98)) >= 44,
    "seed1_gain_ge_0p05": gain_v98 >= 0.05,
    "far_gain_nonnegative": far_gain_v98 >= 0.0,
    "positive_well_fraction_ge_0p55": positive_well_fraction_v98 >= 0.55,
    "bootstrap_probability_gain_ge_0p90": (
        bootstrap_v98["probability_gain_positive"] >= 0.90
    ),
}
strict_pass_v98 = bool(all(checks_v98.values()))

verdict_v98 = {
    "experiment": "mainline_pf_seed1_confirmation_v98",
    "stage": "disjoint_48well_confirmation_of_V97_seed1",
    "scope": "upstream ANCC/Z PF learned core only",
    "selection_or_tuning_count": 0,
    "discovery_wells_excluded": len(excluded_wells_v98),
    "confirmation_wells_requested": len(confirmation_wells_v98),
    "confirmation_wells_complete": len(complete_wells_v98),
    "ranking_source": ranking_source_v98,
    "models_dir": str(models_dir_v98),
    "reference_seed": REFERENCE_SEED_V98,
    "frozen_candidate_seed": FROZEN_SEED_V98,
    "reference": reference_metrics_v98,
    "candidate": candidate_metrics_v98,
    "gain": gain_v98,
    "far_gain": far_gain_v98,
    "positive_well_fraction": positive_well_fraction_v98,
    "well_bootstrap": bootstrap_v98,
    "checks": checks_v98,
    "strict_pass": strict_pass_v98,
    "elapsed_minutes": (time.time() - STARTED_V98) / 60.0,
    "next": (
        "PASS -> implement exact seed1 deterministic patch, then test downstream survival"
        if strict_pass_v98
        else "FAIL -> seed1 was discovery noise; close PF seed optimization"
    ),
}

summary_v98.to_csv(OUT_DIR_V98 / "summary.csv", index=False)
by_well_v98.to_csv(OUT_DIR_V98 / "by_well.csv", index=False)
long_v98.to_parquet(OUT_DIR_V98 / "paired_predictions.parquet", index=False)
(OUT_DIR_V98 / "result.json").write_text(
    json.dumps(verdict_v98, indent=2), encoding="utf-8"
)

print("\nV98 SUMMARY")
print(summary_v98.to_string(index=False))
print("\nV98 FINAL VERDICT")
print(json.dumps(verdict_v98, indent=2), flush=True)
print("saved:", OUT_DIR_V98, flush=True)
