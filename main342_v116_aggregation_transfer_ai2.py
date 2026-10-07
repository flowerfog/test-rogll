"""M342-AI2: staged transfer audit for V116 candidate aggregation.

Runs the existing V117 support scan on folds 4, 5, 3, 2, and 1 without
changing its candidate generator.  Fold4/5 select one (top-k, alpha) recipe;
Fold3 confirms it; Fold2/Fold1 are zero-retuning support.  No submission is
written.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "rawdata_lattice_consensus_support_scan_v117_cell.py"
OOF = ROOT / "oof_exports"
EXPERIMENT = "main342_v116_aggregation_transfer_ai2"
OUT = OOF / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)
DISCOVERY_FOLDS = (4, 5)
EVALUATION_FOLDS = (3, 2, 1)


def _run_fold(fold: int) -> pd.DataFrame:
    source = SOURCE.read_text(encoding="utf-8")
    source = re.sub(r"^EXPERIMENT = .*?$", f'EXPERIMENT = "{EXPERIMENT}_fold{fold}"', source, count=1, flags=re.MULTILINE)
    source, count = re.subn(r"^FOLD = 3\s*$", f"FOLD = {fold}", source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("Could not parameterize V117 FOLD")
    namespace = {"__name__": f"{EXPERIMENT}_fold{fold}", "__file__": str(SOURCE)}
    exec(compile(source, str(SOURCE), "exec"), namespace)
    scan = OUT / f"{EXPERIMENT}_fold{fold}" / "support_scan.csv"
    if not scan.is_file():
        raise RuntimeError(f"Missing fold scan: {scan}")
    result = pd.read_csv(scan)
    result["fold"] = fold
    return result


def _key(frame: pd.DataFrame) -> pd.Series:
    return frame["topk"].astype(int).astype(str) + "_" + frame["alpha"].round(8).astype(str)


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

scans = {}
for fold in (*DISCOVERY_FOLDS, *EVALUATION_FOLDS):
    print(f"M342-AI2 START FOLD {fold}", flush=True)
    scans[fold] = _run_fold(fold)
    scans[fold].to_csv(OUT / f"fold{fold}_scan.csv", index=False)

discovery = pd.concat([scans[fold] for fold in DISCOVERY_FOLDS], ignore_index=True)
key = _key(discovery)
discovery["recipe_key"] = key
grouped = discovery.groupby("recipe_key", as_index=False).agg(
    topk=("topk", "first"), alpha=("alpha", "first"),
    pooled_rmse=("rmse", "mean"), mean_gain=("gain", "mean"),
    minimum_fold_gain=("gain", "min"), minimum_fold_far_gain=("far_gain", "min"),
    maximum_fold_p95_change=("p95_change", "max"),
    minimum_shuffle_loss=("aligned_increment_vs_shuffle", "min"),
)
safe = grouped[
    (grouped["minimum_fold_gain"] >= 0.0)
    & (grouped["minimum_fold_far_gain"] >= 0.0)
    & (grouped["maximum_fold_p95_change"] <= 0.20)
    & (grouped["minimum_shuffle_loss"] >= 0.0)
]
selected = (safe if len(safe) else grouped).sort_values(["pooled_rmse", "topk", "alpha"], kind="stable").iloc[0]
selected_key = str(selected["recipe_key"])

all_rows = []
for fold, scan in scans.items():
    local = scan[_key(scan).eq(selected_key)]
    if len(local) != 1:
        raise RuntimeError(f"Selected recipe missing on fold {fold}: {selected_key}")
    row = local.iloc[0].to_dict()
    row["role"] = "discovery" if fold in DISCOVERY_FOLDS else ("confirmation" if fold == 3 else "support")
    all_rows.append(row)
fold_report = pd.DataFrame(all_rows)
fold_report.to_csv(OUT / "selected_recipe_by_fold.csv", index=False)

discovery_gain = float(selected["mean_gain"])
confirmation = fold_report[fold_report["fold"].eq(3)].iloc[0]
support = fold_report[fold_report["fold"].isin(SUPPORT_FOLDS)]
checks = {
    "discovery_safe_recipe": bool(len(safe)),
    "selected_beats_v116_on_discovery": discovery_gain >= 0.01,
    "confirmation_gain_ge_0p01": float(confirmation["gain"]) >= 0.01,
    "confirmation_far_gain_ge_0": float(confirmation["far_gain"]) >= 0.0,
    "confirmation_p95_change_le_0p20": float(confirmation["p95_change"]) <= 0.20,
    "support_folds_positive": bool(len(support)) and bool((support["gain"] > 0.0).all()),
    "support_far_nonnegative": bool(len(support)) and bool((support["far_gain"] >= 0.0).all()),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold45_aggregation_selection_fold3_confirmation_fold2_fold1_support",
    "selection_or_tuning_count": 1,
    "source_generator": str(SOURCE),
    "protocol": {"discovery_folds": DISCOVERY_FOLDS, "confirmation_fold": 3, "support_folds": SUPPORT_FOLDS},
    "selected": selected.to_dict(),
    "fold_report": fold_report.to_dict(orient="records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> build one frozen V116 aggregation deployment candidate" if strict_pass else "FAIL -> retain V116 top33 uniform mean exactly",
}
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, allow_nan=False), encoding="utf-8")
print("M342-AI2 SELECTED", selected.to_dict(), flush=True)
print("M342-AI2 FOLDS")
print(fold_report[["fold", "role", "topk", "alpha", "rmse", "gain", "far_gain", "p95_change", "aligned_increment_vs_shuffle"]].to_string(index=False), flush=True)
print("M342-AI2 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT, flush=True)
