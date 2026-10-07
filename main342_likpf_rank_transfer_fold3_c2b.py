"""M342-C2b: frozen Fold-3 confirmation of C2 likPF rank transfer.

C2 discovered prior_weight=0 independently on Folds 4 and 5. Its original
gate rejected that arm only because fewer than half of all wells improved,
which is not aligned with the competition's pooled-row RMSE. C2b does not
retune anything: it confirms the frozen rule on Fold 3 and adds bootstrap and
tail-risk diagnostics before any full-row reconstruction.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
DETAIL_PATH = (
    OOF_ROOT / "main342_samewell_endpoint_rank_transfer_c1"
    / "rank_transfer_detail.parquet"
)
C2_VERDICT = (
    OOF_ROOT / "main342_likpf_rank_transfer_attribution_c2" / "verdict.json"
)
OUT_DIR = OOF_ROOT / "main342_likpf_rank_transfer_fold3_c2b"

FAMILY = (
    "likpf_scale_3", "likpf_scale_5", "likpf_scale_8",
    "likpf_scale_12", "likpf_mean",
)
DONOR_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
FROZEN_PRIOR_WEIGHT = 0.0
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260804


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
        sse = float(donor[f"official_sse__{candidate}"].sum())
        count = float(donor[f"official_rows__{candidate}"].sum())
        rows.append({"candidate": candidate, "rmse": np.sqrt(sse / count)})
    output = pd.DataFrame(rows).sort_values(["rmse", "candidate"]).reset_index(drop=True)
    output["population_rank"] = output["rmse"].rank(method="average")
    return output


def _selections(held, population):
    population_rank = population.set_index("candidate")["population_rank"]
    output = {}
    for _, row in held.iterrows():
        pseudo = pd.Series(
            {name: row[f"pseudo_rmse__{name}"] for name in FAMILY}, dtype=float
        )
        if pseudo.notna().sum() != len(FAMILY):
            continue
        pseudo_rank = pseudo.rank(method="average")
        score = pseudo_rank + FROZEN_PRIOR_WEIGHT * population_rank
        output[str(row["well"])] = str(score.sort_values(kind="stable").index[0])
    return output


def _well_table(held, selections, fixed):
    held = held.sort_values("well").reset_index(drop=True)
    selected_names = np.asarray(
        [selections.get(str(well), fixed) for well in held["well"]], dtype=object
    )
    shuffled_names = np.roll(selected_names.copy(), 1)
    records = []
    for index, row in held.iterrows():
        selected = str(selected_names[index])
        shuffled = str(shuffled_names[index])
        rows = float(row[f"official_rows__{fixed}"])
        family_sse = np.asarray(
            [row[f"official_sse__{name}"] for name in FAMILY], np.float64
        )
        records.append({
            "well": str(row["well"]),
            "rows": int(rows),
            "fixed_candidate": fixed,
            "selected_candidate": selected,
            "shuffled_candidate": shuffled,
            "fixed_sse": float(row[f"official_sse__{fixed}"]),
            "selected_sse": float(row[f"official_sse__{selected}"]),
            "shuffled_sse": float(row[f"official_sse__{shuffled}"]),
            "oracle_sse": float(np.min(family_sse)),
        })
    output = pd.DataFrame(records)
    for name in ("fixed", "selected", "shuffled", "oracle"):
        output[f"{name}_rmse"] = np.sqrt(output[f"{name}_sse"] / output["rows"])
    output["gain"] = output["fixed_rmse"] - output["selected_rmse"]
    return output


def _pooled(table, prefix):
    return float(np.sqrt(table[f"{prefix}_sse"].sum() / table["rows"].sum()))


def _bootstrap(table):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = table["rows"].to_numpy(np.float64)
    fixed = table["fixed_sse"].to_numpy(np.float64)
    selected = table["selected_sse"].to_numpy(np.float64)
    gains = np.empty(BOOTSTRAP_DRAWS, np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(table), len(table))
        gains[draw] = (
            np.sqrt(fixed[index].sum() / rows[index].sum())
            - np.sqrt(selected[index].sum() / rows[index].sum())
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "gain_p05": float(np.quantile(gains, 0.05)),
        "gain_median": float(np.median(gains)),
        "gain_p95": float(np.quantile(gains, 0.95)),
        "probability_gain_positive": float(np.mean(gains > 0.0)),
    }


started = time.time()
for path in (DETAIL_PATH, C2_VERDICT):
    if not path.is_file():
        raise FileNotFoundError(path)
OUT_DIR.mkdir(parents=True, exist_ok=True)

c2 = json.loads(C2_VERDICT.read_text(encoding="utf-8"))
if float(c2.get("selected_prior_weight", np.nan)) != FROZEN_PRIOR_WEIGHT:
    raise RuntimeError(f"C2 did not freeze prior weight {FROZEN_PRIOR_WEIGHT}: {c2}")

detail = pd.read_parquet(DETAIL_PATH)
detail["fold"] = detail["fold"].astype(int)
detail["well"] = detail["well"].astype(str)
source_folds = sorted(detail["fold"].unique().tolist())
detail = detail[detail["fold"].isin(DONOR_FOLDS + (CONFIRMATION_FOLD,))].copy()
donor = detail[detail["fold"].isin(DONOR_FOLDS)]
held = detail[detail["fold"].eq(CONFIRMATION_FOLD)]
if donor["well"].nunique() < 295 or held["well"].nunique() < 145:
    raise RuntimeError(
        f"Insufficient C2b coverage donor={donor['well'].nunique()} held={held['well'].nunique()}"
    )

population = _population(donor)
fixed = str(population.iloc[0]["candidate"])
selected = _selections(held, population)
table = _well_table(held, selected, fixed)

fixed_rmse = _pooled(table, "fixed")
selected_rmse = _pooled(table, "selected")
shuffled_rmse = _pooled(table, "shuffled")
oracle_rmse = _pooled(table, "oracle")
gain = fixed_rmse - selected_rmse
shuffle_loss = shuffled_rmse - selected_rmse
changed = table["selected_candidate"].ne(fixed)
positive_all = float(np.mean(table["gain"] > 0.0))
positive_changed = float(np.mean(table.loc[changed, "gain"] > 0.0)) if changed.any() else 0.0
p95_change = float(table["selected_rmse"].quantile(0.95) - table["fixed_rmse"].quantile(0.95))
bootstrap = _bootstrap(table)

checks = {
    "pooled_gain_ge_0p10": gain >= 0.10,
    "shuffle_loses_ge_0p10": shuffle_loss >= 0.10,
    "bootstrap_probability_ge_0p90": bootstrap["probability_gain_positive"] >= 0.90,
    "changed_well_positive_fraction_ge_0p55": positive_changed >= 0.55,
    "p95_not_worse_by_gt_0p50": p95_change <= 0.50,
}
strict_pass = bool(all(checks.values()))
summary = {
    "rows": int(table["rows"].sum()),
    "wells": int(len(table)),
    "population_fixed": fixed,
    "fixed_rmse": fixed_rmse,
    "selected_rmse": selected_rmse,
    "oracle_rmse": oracle_rmse,
    "shuffled_rmse": shuffled_rmse,
    "gain_vs_fixed": gain,
    "shuffle_loss": shuffle_loss,
    "changed_well_fraction": float(changed.mean()),
    "positive_all_well_fraction": positive_all,
    "positive_changed_well_fraction": positive_changed,
    "p95_change": p95_change,
}
verdict = {
    "experiment": "main342_likpf_rank_transfer_fold3_c2b",
    "stage": "frozen_fold3_sampled_confirmation_of_c2_discovery",
    "frozen_prior_weight": FROZEN_PRIOR_WEIGHT,
    "candidate_family": list(FAMILY),
    "donor_folds": list(DONOR_FOLDS),
    "confirmation_fold": CONFIRMATION_FOLD,
    "source_folds_present": source_folds,
    "fold1_used_in_any_statistic": False,
    "fold2_used_in_any_statistic": False,
    "summary": summary,
    "bootstrap": bootstrap,
    "selection_counts": table["selected_candidate"].value_counts().to_dict(),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> reconstruct full-row Fold3 first, then open Fold2 only after a full-row gate"
        if strict_pass else
        "FAIL -> sampled Fold4/5 gain does not confirm; close same-well replay selection"
    ),
}
population.to_csv(OUT_DIR / "population_candidates.csv", index=False)
table.to_csv(OUT_DIR / "fold3_by_well.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-C2b POPULATION")
print(population.to_string(index=False))
print("\nM342-C2b SUMMARY")
print(json.dumps(_clean(summary), indent=2))
print("\nM342-C2b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)

