"""V170: read-only parent-lineage audit for the frozen V168 analog move.

V168 was confirmed on V144's nested sequence endpoint, while the directly
deployable Kaggle chain currently ends at V116 consensus_center. This audit
recovers V168's selected absolute analog trajectory and applies the same
pre-registered 0.05 dose to every available parent endpoint. No labels are
used for selection, fitting, or dose tuning.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "historical_analog_parent_lineage_v170"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V144_PATH = (
    OOF_ROOT / "final_consensus_sequence_coefficient_full5_v144"
    / "crossfit_oof.parquet"
)
V168_PATH = (
    OOF_ROOT / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
)
WEIGHT = 0.05
BASE_COLUMNS = (
    "incumbent_6p377_replay",
    "exact_v76",
    "consensus_center",
    "nested_sequence_coefficient",
)


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


def _metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    wells = frame["well"].astype(str).to_numpy()
    distance = frame["md_since"].to_numpy(np.float64)
    well_sse = pd.Series(error * error).groupby(wells, sort=False).sum()
    well_rows = pd.Series(np.ones(len(frame), np.int64)).groupby(wells, sort=False).sum()
    well_rmse = np.sqrt(well_sse / well_rows)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


for path in (V144_PATH, V168_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)

v144 = pd.read_parquet(V144_PATH)
v168 = pd.read_parquet(V168_PATH)
required_v168 = {"id", "base_current", "metadata_analog_w0p05"}
missing_v168 = sorted(required_v168.difference(v168.columns))
if missing_v168:
    raise RuntimeError(f"V168 missing columns: {missing_v168}")

available_bases = [column for column in BASE_COLUMNS if column in v144.columns]
if "consensus_center" not in available_bases or "nested_sequence_coefficient" not in available_bases:
    raise RuntimeError(f"V144 parent lineage incomplete: {available_bases}")

analog = v168[["id", "base_current", "metadata_analog_w0p05"]].copy()
analog["analog_absolute"] = (
    analog["base_current"].to_numpy(np.float64)
    + (analog["metadata_analog_w0p05"].to_numpy(np.float64)
       - analog["base_current"].to_numpy(np.float64)) / WEIGHT
)
frame = v144.merge(
    analog[["id", "analog_absolute"]], on="id", how="left", validate="one_to_one"
)
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
if len(frame) != 3_783_989 or frame["well"].nunique() != 773:
    raise RuntimeError(f"Coverage mismatch rows={len(frame)} wells={frame['well'].nunique()}")
if frame[["analog_absolute", *available_bases]].isna().any().any():
    raise RuntimeError("Non-finite parent-lineage alignment")

analog_absolute = frame["analog_absolute"].to_numpy(np.float64)
summary_rows = []
fold_rows = []
for base_column in available_bases:
    base = frame[base_column].to_numpy(np.float64)
    candidate = base + WEIGHT * (analog_absolute - base)
    for fold in (0, 1, 2, 3, 4, 5):
        mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].to_numpy(int) == fold
        local = frame.loc[mask].reset_index(drop=True)
        base_metric = _metric(local, base[mask], f"{base_column}_base")
        candidate_metric = _metric(local, candidate[mask], f"{base_column}_analog_w0p05")
        summary_rows.extend([
            {"parent": base_column, "fold": fold, **base_metric},
            {"parent": base_column, "fold": fold, **candidate_metric},
        ])
        if fold:
            fold_rows.append({
                "parent": base_column,
                "fold": fold,
                "base_rmse": base_metric["rmse"],
                "rmse": candidate_metric["rmse"],
                "gain": base_metric["rmse"] - candidate_metric["rmse"],
                "far_gain": (
                    base_metric["far_gt2000_rmse"]
                    - candidate_metric["far_gt2000_rmse"]
                ),
                "p95_gain": (
                    base_metric["p95_well_rmse"]
                    - candidate_metric["p95_well_rmse"]
                ),
            })

summary = pd.DataFrame(summary_rows)
folds = pd.DataFrame(fold_rows)
parent_rows = []
for parent, local in folds.groupby("parent", sort=False):
    pooled = summary[(summary["parent"].eq(parent)) & summary["fold"].eq(0)].set_index("endpoint")
    base = pooled.loc[f"{parent}_base"]
    candidate = pooled.loc[f"{parent}_analog_w0p05"]
    parent_rows.append({
        "parent": parent,
        "base_rmse": base["rmse"],
        "rmse": candidate["rmse"],
        "gain": base["rmse"] - candidate["rmse"],
        "far_gain": base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"],
        "p95_gain": base["p95_well_rmse"] - candidate["p95_well_rmse"],
        "positive_folds": int((local["gain"] > 0).sum()),
        "minimum_fold_gain": float(local["gain"].min()),
        "deployable_gate": bool(
            base["rmse"] - candidate["rmse"] >= 0.05
            and (local["gain"] > 0).all()
            and base["far_gt2000_rmse"] > candidate["far_gt2000_rmse"]
            and base["p95_well_rmse"] - candidate["p95_well_rmse"] >= -0.10
        ),
    })

parents = pd.DataFrame(parent_rows).sort_values("gain", ascending=False, kind="stable")
deployable = parents[parents["deployable_gate"]]
verdict = {
    "experiment": EXPERIMENT,
    "stage": "read_only_frozen_V168_move_parent_lineage_audit",
    "selection_or_tuning_count": 0,
    "frozen_weight": WEIGHT,
    "coverage": {"rows": len(frame), "wells": frame["well"].nunique()},
    "parents": parent_rows,
    "deployable_parents": deployable["parent"].tolist(),
    "strict_pass": bool(len(deployable)),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> deploy only after the highest-ranked deployable parent cell"
        if len(deployable)
        else "FAIL -> do not deploy V168; retain the existing Kaggle endpoint"
    ),
}

summary.to_csv(OUT_DIR / "summary.csv", index=False)
folds.to_csv(OUT_DIR / "folds.csv", index=False)
parents.to_csv(OUT_DIR / "parents.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nV170 PARENT SUMMARY")
print(parents.to_string(index=False), flush=True)
print("\nV170 FOLD SUMMARY")
print(folds.to_string(index=False), flush=True)
print("\nV170 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
