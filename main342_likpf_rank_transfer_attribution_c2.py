"""M342-C2: robust likPF-family same-well rank-transfer attribution.

This consumes the corrected C1 rank table, which must contain only folds
3/4/5 after a failed Fold-3 gate. Fold 4 and Fold 5 select one frozen blend of
population rank and same-well pseudo-cut rank. Fold 3 then confirms that rule.
Only sampled official rows are used; no full-row endpoint is built here.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
C1_DIR = OOF_ROOT / "main342_samewell_endpoint_rank_transfer_c1"
DETAIL_PATH = C1_DIR / "rank_transfer_detail.parquet"
OUT_DIR = OOF_ROOT / "main342_likpf_rank_transfer_attribution_c2"

FAMILY = (
    "likpf_scale_3", "likpf_scale_5", "likpf_scale_8",
    "likpf_scale_12", "likpf_mean",
)
DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
PRIOR_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)


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


def _population(donor):
    rows = []
    for candidate in FAMILY:
        sse = donor[f"official_sse__{candidate}"].sum()
        count = donor[f"official_rows__{candidate}"].sum()
        rows.append({
            "candidate": candidate,
            "rmse": float(np.sqrt(sse / count)),
        })
    output = pd.DataFrame(rows).sort_values(["rmse", "candidate"]).reset_index(drop=True)
    output["population_rank"] = output["rmse"].rank(method="average").astype(float)
    return output


def _select(local, population, prior_weight):
    population_rank = population.set_index("candidate")["population_rank"].to_dict()
    selections = {}
    for row in local.itertuples(index=False):
        pseudo = pd.Series(
            {candidate: getattr(row, f"pseudo_rmse__{candidate}") for candidate in FAMILY},
            dtype=float,
        )
        if pseudo.notna().sum() != len(FAMILY):
            continue
        pseudo_rank = pseudo.rank(method="average")
        score = pd.Series({
            candidate: float(pseudo_rank[candidate])
            + float(prior_weight) * float(population_rank[candidate])
            for candidate in FAMILY
        })
        selections[str(row.well)] = str(score.sort_values(kind="stable").index[0])
    return selections


def _evaluate(local, selections, population_fixed, shuffled=False):
    local = local.sort_values("well").reset_index(drop=True)
    names = np.asarray(
        [selections.get(str(well), population_fixed) for well in local["well"]],
        dtype=object,
    )
    if shuffled and len(names) > 1:
        names = np.roll(names.copy(), 1)
    selected_sse = np.asarray([
        local.iloc[index][f"official_sse__{name}"] for index, name in enumerate(names)
    ], np.float64)
    selected_rows = np.asarray([
        local.iloc[index][f"official_rows__{name}"] for index, name in enumerate(names)
    ], np.float64)
    fixed_sse = local[f"official_sse__{population_fixed}"].to_numpy(np.float64)
    fixed_rows = local[f"official_rows__{population_fixed}"].to_numpy(np.float64)
    oracle_sse = np.column_stack([
        local[f"official_sse__{name}"].to_numpy(np.float64) for name in FAMILY
    ]).min(axis=1)
    oracle_rows = local[f"official_rows__{FAMILY[0]}"].to_numpy(np.float64)
    selected_rmse_by_well = np.sqrt(selected_sse / selected_rows)
    fixed_rmse_by_well = np.sqrt(fixed_sse / fixed_rows)
    return {
        "rows": int(selected_rows.sum()),
        "wells": int(len(local)),
        "selected_rmse": float(np.sqrt(selected_sse.sum() / selected_rows.sum())),
        "fixed_rmse": float(np.sqrt(fixed_sse.sum() / fixed_rows.sum())),
        "oracle_rmse": float(np.sqrt(oracle_sse.sum() / oracle_rows.sum())),
        "gain_vs_fixed": float(
            np.sqrt(fixed_sse.sum() / fixed_rows.sum())
            - np.sqrt(selected_sse.sum() / selected_rows.sum())
        ),
        "positive_well_fraction": float(np.mean(selected_rmse_by_well < fixed_rmse_by_well)),
        "changed_well_fraction": float(np.mean(names != population_fixed)),
    }


started = time.time()
if not DETAIL_PATH.is_file():
    raise FileNotFoundError(DETAIL_PATH)
OUT_DIR.mkdir(parents=True, exist_ok=True)

detail = pd.read_parquet(DETAIL_PATH)
required = {"well", "fold"}
for candidate in FAMILY:
    required.update({
        f"pseudo_rmse__{candidate}", f"official_sse__{candidate}",
        f"official_rows__{candidate}",
    })
missing = sorted(required.difference(detail.columns))
if missing:
    raise RuntimeError(f"C1 detail missing columns: {missing}")
detail["well"] = detail["well"].astype(str)
detail["fold"] = detail["fold"].astype(int)

# Older C1 artifacts materialized all folds before the Fold-3 gate. Quarantine
# those rows before any aggregation so they cannot affect C2 selection.
unexpected_folds = sorted(set(detail["fold"]) - {3, 4, 5})
discarded_by_fold = (
    detail.loc[~detail["fold"].isin({3, 4, 5})]
    .groupby("fold")["well"].nunique().astype(int).to_dict()
)
detail = detail.loc[detail["fold"].isin({3, 4, 5})].copy().reset_index(drop=True)
if unexpected_folds:
    print(
        "M342-C2 SOURCE QUARANTINE",
        {"discarded_folds": unexpected_folds, "discarded_wells": discarded_by_fold},
        flush=True,
    )
coverage = detail.groupby("fold")["well"].nunique().astype(int).to_dict()
if coverage.get(3, 0) < 145 or coverage.get(4, 0) < 145 or coverage.get(5, 0) < 145:
    raise RuntimeError(f"Insufficient C2 coverage: {coverage}")

discovery_reports = []
for held_fold in DISCOVERY_FOLDS:
    donor_fold = 5 if held_fold == 4 else 4
    donor = detail[detail["fold"].eq(donor_fold)]
    held = detail[detail["fold"].eq(held_fold)]
    population = _population(donor)
    fixed = str(population.iloc[0]["candidate"])
    for weight in PRIOR_WEIGHTS:
        selections = _select(held, population, weight)
        aligned = _evaluate(held, selections, fixed, shuffled=False)
        shuffled = _evaluate(held, selections, fixed, shuffled=True)
        discovery_reports.append({
            "fold": held_fold,
            "donor_fold": donor_fold,
            "prior_weight": float(weight),
            "population_fixed": fixed,
            **aligned,
            "shuffle_rmse": shuffled["selected_rmse"],
            "shuffle_loss": shuffled["selected_rmse"] - aligned["selected_rmse"],
        })

discovery = pd.DataFrame(discovery_reports)
summary = discovery.groupby("prior_weight", as_index=False).agg(
    mean_gain=("gain_vs_fixed", "mean"),
    minimum_gain=("gain_vs_fixed", "min"),
    mean_shuffle_loss=("shuffle_loss", "mean"),
    minimum_shuffle_loss=("shuffle_loss", "min"),
    mean_changed_fraction=("changed_well_fraction", "mean"),
    minimum_positive_well_fraction=("positive_well_fraction", "min"),
)
eligible = summary[
    summary["minimum_gain"].ge(0.0)
    & summary["minimum_shuffle_loss"].ge(0.0)
].copy()
if eligible.empty:
    selected_weight = 0.0
else:
    selected_weight = float(
        eligible.sort_values(
            ["minimum_gain", "mean_gain", "prior_weight"],
            ascending=[False, False, True],
        ).iloc[0]["prior_weight"]
    )

discovery_selected = discovery[discovery["prior_weight"].eq(selected_weight)]
discovery_pass = bool(
    discovery_selected["gain_vs_fixed"].min() >= 0.03
    and discovery_selected["shuffle_loss"].min() >= 0.02
    and discovery_selected["positive_well_fraction"].min() >= 0.50
)

confirmation = None
confirmation_pass = False
if discovery_pass:
    donor = detail[detail["fold"].isin(DISCOVERY_FOLDS)]
    held = detail[detail["fold"].eq(CONFIRMATION_FOLD)]
    population = _population(donor)
    fixed = str(population.iloc[0]["candidate"])
    selections = _select(held, population, selected_weight)
    aligned = _evaluate(held, selections, fixed, shuffled=False)
    shuffled = _evaluate(held, selections, fixed, shuffled=True)
    confirmation = {
        "fold": CONFIRMATION_FOLD,
        "prior_weight": selected_weight,
        "population_fixed": fixed,
        **aligned,
        "shuffle_rmse": shuffled["selected_rmse"],
        "shuffle_loss": shuffled["selected_rmse"] - aligned["selected_rmse"],
    }
    confirmation_pass = bool(
        confirmation["gain_vs_fixed"] >= 0.03
        and confirmation["shuffle_loss"] >= 0.02
        and confirmation["positive_well_fraction"] >= 0.50
    )

strict_pass = bool(discovery_pass and confirmation_pass)
verdict = {
    "experiment": "main342_likpf_rank_transfer_attribution_c2",
    "stage": "robust_likpf_family_sampled_rank_transfer_attribution",
    "sample_unit": "complete physical well; sampled official rows",
    "candidate_family": list(FAMILY),
    "coverage": coverage,
    "discovery_folds": list(DISCOVERY_FOLDS),
    "confirmation_fold": CONFIRMATION_FOLD if discovery_pass else None,
    "source_quarantine": {
        "historical_c1_contained_held_folds": unexpected_folds,
        "discarded_wells_by_fold": discarded_by_fold,
        "fold1_used_in_any_c2_statistic": False,
        "fold2_used_in_any_c2_statistic": False,
    },
    "selected_prior_weight": selected_weight,
    "discovery": discovery.to_dict("records"),
    "discovery_summary": summary.to_dict("records"),
    "discovery_pass": discovery_pass,
    "confirmation": confirmation,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> reconstruct full-row Fold3/Fold2 likPF-family selections and test V168 blend"
        if strict_pass else
        "FAIL -> even the robust likPF family has no same-well rank transfer; close replay selection"
    ),
}
discovery.to_csv(OUT_DIR / "discovery_folds.csv", index=False)
summary.to_csv(OUT_DIR / "discovery_summary.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-C2 DISCOVERY FOLDS")
print(discovery.to_string(index=False))
print("\nM342-C2 SUMMARY")
print(summary.to_string(index=False))
print("\nM342-C2 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
