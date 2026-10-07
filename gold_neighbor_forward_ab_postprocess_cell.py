"""A/B post-process for Gold forward chaining with neighbor candidates.

Run immediately after:
1. gold_neighbor_candidate_setup_cell.py
2. patched block_paired_gold_forward_chaining_cell.py

The same cached W1/W2/W3 metrics are evaluated twice: once with neighbor_*
candidates removed and once with the complete candidate menu. No rescoring or
parameter selection occurs in this cell.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


_required = [
    "successes",
    "_choose_original",
    "_choose_block_paired",
    "_evaluate_choice",
    "rmse_from_sse",
    "OOF_DIR",
]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(
        "Run setup and block-paired Gold cells first; missing: " + repr(_missing)
    )

EXTRA_PREFIXES = tuple(globals().get("_GOLD_EXTRA_CANDIDATE_PREFIXES", ("neighbor_",)))
OUT_DIR = Path(OOF_DIR) / "gold_neighbor_forward_ab_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _is_extra(name):
    return any(str(name).startswith(prefix) for prefix in EXTRA_PREFIXES)


def _without_extra(record):
    metrics = {
        name: value
        for name, value in record["metrics"].items()
        if not _is_extra(name)
    }
    if record["default_name"] not in metrics or len(metrics) < 2:
        raise RuntimeError(f"invalid baseline candidate menu for {record['well']}")
    return {**record, "metrics": metrics, "candidate_count": len(metrics)}


rows = []
for record in successes:
    baseline = _without_extra(record)
    choices = {
        "baseline_original": _choose_original(baseline),
        "extended_original": _choose_original(record),
        "baseline_paired": _choose_block_paired(baseline),
        "extended_paired": _choose_block_paired(record),
    }
    evaluations = {
        "baseline_original": _evaluate_choice(baseline, choices["baseline_original"]),
        "extended_original": _evaluate_choice(record, choices["extended_original"]),
        "baseline_paired": _evaluate_choice(baseline, choices["baseline_paired"]),
        "extended_paired": _evaluate_choice(record, choices["extended_paired"]),
    }
    row = {
        "well": str(record["well"]),
        "fold": int(record["fold"]),
        "rows": int(evaluations["baseline_original"]["rows"]),
        "base_sse": float(evaluations["baseline_original"]["base_sse"]),
        "baseline_candidate_count": int(len(baseline["metrics"])),
        "extended_candidate_count": int(len(record["metrics"])),
    }
    for method in choices:
        choice = choices[method]
        evaluation = evaluations[method]
        row[f"{method}_candidate"] = str(choice["candidate"])
        row[f"{method}_selected"] = bool(choice["selected"])
        row[f"{method}_neighbor"] = bool(_is_extra(choice["candidate"]))
        row[f"{method}_sse"] = float(evaluation["pred_sse"])
        row[f"{method}_well_rmse"] = float(evaluation["pred_rmse_well"])
        row[f"{method}_well_gain"] = float(evaluation["gain_well"])
    rows.append(row)

detail = pd.DataFrame(rows)
METHODS = (
    "baseline_original",
    "extended_original",
    "baseline_paired",
    "extended_paired",
)


def _summary(frame, method):
    n = int(frame["rows"].sum())
    base_rmse = rmse_from_sse(frame["base_sse"].sum(), n)
    score = rmse_from_sse(frame[f"{method}_sse"].sum(), n)
    neighbor = frame[f"{method}_neighbor"].to_numpy(bool)
    return {
        "method": method,
        "rows": n,
        "wells": len(frame),
        "base_rmse": base_rmse,
        "rmse": score,
        "gain_vs_default": base_rmse - score,
        "selected_wells": int(frame[f"{method}_selected"].sum()),
        "neighbor_selected_wells": int(neighbor.sum()),
        "neighbor_positive_frac": (
            float((frame.loc[neighbor, f"{method}_well_gain"] > 0).mean())
            if neighbor.any()
            else 0.0
        ),
        "neighbor_mean_well_gain": (
            float(frame.loc[neighbor, f"{method}_well_gain"].mean())
            if neighbor.any()
            else 0.0
        ),
    }


summary = pd.DataFrame([_summary(detail, method) for method in METHODS])
fold_rows = []
for fold, group in detail.groupby("fold", sort=True):
    for method in METHODS:
        record = _summary(group, method)
        record["fold"] = int(fold)
        fold_rows.append(record)
folds = pd.DataFrame(fold_rows)


def _comparison(judge):
    baseline_name = f"baseline_{judge}"
    extended_name = f"extended_{judge}"
    baseline = summary[summary["method"] == baseline_name].iloc[0]
    extended = summary[summary["method"] == extended_name].iloc[0]
    pivot = folds[folds["method"].isin([baseline_name, extended_name])].pivot(
        index="fold", columns="method", values="rmse"
    )
    fold_gain = pivot[baseline_name] - pivot[extended_name]
    pass_gate = bool(
        float(baseline["rmse"] - extended["rmse"]) >= 0.05
        and (fold_gain >= 0).all()
        and int((fold_gain > 0).sum()) >= 4
        and int(extended["neighbor_selected_wells"]) > 0
        and float(extended["neighbor_positive_frac"]) >= 0.60
    )
    return {
        "judge": judge,
        "baseline_rmse": float(baseline["rmse"]),
        "extended_rmse": float(extended["rmse"]),
        "gain_from_neighbor_menu": float(baseline["rmse"] - extended["rmse"]),
        "min_fold_gain": float(fold_gain.min()),
        "positive_folds": int((fold_gain > 0).sum()),
        "neighbor_selected_wells": int(extended["neighbor_selected_wells"]),
        "neighbor_positive_frac": float(extended["neighbor_positive_frac"]),
        "pass": pass_gate,
    }


comparisons = pd.DataFrame([_comparison("original"), _comparison("paired")])
strict_pass = bool(comparisons["pass"].any())
verdict = {
    "extra_prefixes": list(EXTRA_PREFIXES),
    "successful_wells": len(detail),
    "strict_pass": strict_pass,
    "passing_judges": comparisons.loc[comparisons["pass"], "judge"].tolist(),
    "next": (
        "PASS -> add neighbor factory to deployment Gold and test chain order"
        if strict_pass
        else "STOP neighbor-as-Gold-ammunition; retain direct recentered move only"
    ),
}

detail.to_csv(OUT_DIR / "bywell.csv", index=False)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
folds.to_csv(OUT_DIR / "folds.csv", index=False)
comparisons.to_csv(OUT_DIR / "comparisons.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))

print("\nA/B SUMMARY")
display(summary)
print("\nA/B FOLDS")
display(folds)
print("\nNEIGHBOR CANDIDATE COUNTS")
for method in ("extended_original", "extended_paired"):
    display(
        detail.loc[detail[f"{method}_neighbor"], f"{method}_candidate"]
        .value_counts()
        .rename_axis("candidate")
        .reset_index(name="wells")
    )
print("\nCOMPARISONS")
display(comparisons)
print("\nVERDICT", verdict)
print("saved:", OUT_DIR)
