"""A/B sanity report for the restricted path-bank Gold menu.

Run immediately after ``block_paired_gold_forward_chaining_cell.py``. Only the
original Gold judge is compared because that is the judge confirmed by LB.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


required = [
    "successes",
    "_choose_original",
    "_evaluate_choice",
    "rmse_from_sse",
    "OOF_DIR",
]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError("Run setup and block replay first; missing: " + repr(missing))

PREFIXES = tuple(globals().get("_GOLD_EXTRA_CANDIDATE_PREFIXES", ("pathbank_",)))
OUT_DIR = Path(OOF_DIR) / "gold_pathbank_forward_ab_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _is_pathbank(name):
    return any(str(name).startswith(prefix) for prefix in PREFIXES)


def _without_pathbank(record):
    metrics = {
        name: value
        for name, value in record["metrics"].items()
        if not _is_pathbank(name)
    }
    if record["default_name"] not in metrics or len(metrics) < 2:
        raise RuntimeError(f"invalid baseline menu for {record['well']}")
    return {**record, "metrics": metrics, "candidate_count": len(metrics)}


rows = []
for record in successes:
    baseline_record = _without_pathbank(record)
    baseline_choice = _choose_original(baseline_record)
    extended_choice = _choose_original(record)
    baseline_eval = _evaluate_choice(baseline_record, baseline_choice)
    extended_eval = _evaluate_choice(record, extended_choice)
    rows.append(
        {
            "well": str(record["well"]),
            "fold": int(record["fold"]),
            "rows": int(baseline_eval["rows"]),
            "base_sse": float(baseline_eval["base_sse"]),
            "baseline_sse": float(baseline_eval["pred_sse"]),
            "extended_sse": float(extended_eval["pred_sse"]),
            "baseline_candidate": str(baseline_choice["candidate"]),
            "extended_candidate": str(extended_choice["candidate"]),
            "baseline_selected": bool(baseline_choice["selected"]),
            "extended_selected": bool(extended_choice["selected"]),
            "pathbank_selected": bool(_is_pathbank(extended_choice["candidate"])),
            "baseline_well_gain": float(baseline_eval["gain_well"]),
            "extended_well_gain": float(extended_eval["gain_well"]),
            "menu_gain_well": float(
                baseline_eval["pred_rmse_well"] - extended_eval["pred_rmse_well"]
            ),
            "baseline_candidate_count": int(len(baseline_record["metrics"])),
            "extended_candidate_count": int(len(record["metrics"])),
        }
    )

detail = pd.DataFrame(rows)


def _summary(frame):
    n = int(frame["rows"].sum())
    base_rmse = rmse_from_sse(frame["base_sse"].sum(), n)
    baseline_rmse = rmse_from_sse(frame["baseline_sse"].sum(), n)
    extended_rmse = rmse_from_sse(frame["extended_sse"].sum(), n)
    selected = frame["pathbank_selected"].to_numpy(bool)
    return {
        "rows": n,
        "wells": len(frame),
        "base_rmse": base_rmse,
        "baseline_gold_rmse": baseline_rmse,
        "extended_gold_rmse": extended_rmse,
        "gain_from_pathbank_menu": baseline_rmse - extended_rmse,
        "pathbank_selected_wells": int(selected.sum()),
        "pathbank_selected_fraction": float(selected.mean()),
        "pathbank_positive_well_fraction": (
            float((frame.loc[selected, "extended_well_gain"] > 0).mean())
            if selected.any()
            else 0.0
        ),
        "pathbank_mean_well_gain": (
            float(frame.loc[selected, "extended_well_gain"].mean())
            if selected.any()
            else 0.0
        ),
        "pathbank_mean_menu_gain": (
            float(frame.loc[selected, "menu_gain_well"].mean())
            if selected.any()
            else 0.0
        ),
    }


overall = _summary(detail)
fold_rows = []
for fold, group in detail.groupby("fold", sort=True):
    row = _summary(group)
    row["fold"] = int(fold)
    fold_rows.append(row)
folds = pd.DataFrame(fold_rows)

# This is a sanity gate, not a claim that local Gold predicts LB. The expanded
# menu must participate without creating a new catastrophic local tail.
fold_gain = folds["gain_from_pathbank_menu"].to_numpy(float)
sanity_pass = bool(
    0.01 <= overall["pathbank_selected_fraction"] <= 0.20
    and overall["gain_from_pathbank_menu"] >= -0.05
    and float(np.min(fold_gain)) >= -0.10
    and overall["pathbank_positive_well_fraction"] >= 0.50
)
verdict = {
    "experiment": "gold_pathbank_forward_ab_v1",
    "candidate_prefixes": list(PREFIXES),
    "successful_wells": len(detail),
    "overall": overall,
    "min_fold_menu_gain": float(np.min(fold_gain)),
    "positive_fold_menu_gains": int(np.sum(fold_gain > 0)),
    "sanity_pass": sanity_pass,
    "next": (
        "SANITY PASS -> build all-well deployment bank and one LB candidate"
        if sanity_pass
        else "FAIL -> stop path-bank Gold expansion"
    ),
}

detail.to_csv(OUT_DIR / "bywell.csv", index=False)
folds.to_csv(OUT_DIR / "folds.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, ensure_ascii=True), encoding="utf-8"
)

print("\nPATHBANK GOLD A/B OVERALL")
print(pd.DataFrame([overall]).to_string(index=False))
print("\nPATHBANK GOLD A/B FOLDS")
print(folds.to_string(index=False))
print("\nSELECTED PATH CANDIDATES")
print(
    detail.loc[detail["pathbank_selected"], "extended_candidate"]
    .value_counts()
    .rename_axis("candidate")
    .reset_index(name="wells")
    .to_string(index=False)
)
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
