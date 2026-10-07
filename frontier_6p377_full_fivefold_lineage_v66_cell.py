"""Build the complete five-fold OOF lineage for the 6.377 incumbent.

This cell fixes the coverage error in V61-V63.  It never intersects unrelated
fold subsets.  Instead it starts from the complete 773-well V18 CP8 OOF,
replays the frozen tail-only Neighbor recipe on every canonical fold, and
requires row-level parity with the overlapping V33 replay.

Run on AutoDL after the V18, Neighbor-bank, and fold-metadata artifacts exist.
No model is trained and no hyperparameter is selected here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "frontier_6p377_full_fivefold_lineage_v66"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_CP_PATH = (
    OOF_ROOT
    / "frontier_6p39_full_oof_cp_increment_v18"
    / "full_oof_cp_endpoints.parquet"
)
NEIGHBOR_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
NEIGHBOR_BANK = NEIGHBOR_DIR / "neighbor_transfer_oof.npz"
NEIGHBOR_META = NEIGHBOR_DIR / "selected_neighbors.csv"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "frontier_6p39_existing_models_oof_v0" / "well_metadata.csv",
)
OLD_V33_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)

EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
CP8_COLUMN = "full_cp8_ordered"
CP8_FALLBACK_COLUMN = "full_parent_matched_vp_replay"
CORE_COLUMN = "frontier_6p39_core"
NEIGHBOR_METHOD = "structure_progress"
START_FT = 1000.0
TRANSITION_FT = 250.0
CAP_FT = 24.0
DOSE = 0.20
MIN_NEIGHBOR_COVERAGE = 0.80
PARITY_TOLERANCE_FT = 2e-3


def _first_file(candidates: tuple[Path, ...], label: str) -> Path:
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {label}: {[str(item) for item in candidates]}")
    return path


def _smoothstep(distance: np.ndarray) -> np.ndarray:
    t = np.clip((distance - START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    error = prediction - truth
    well_error = pd.DataFrame(
        {"well": frame["well"].astype(str), "error2": error * error}
    )
    well_rmse = np.sqrt(
        well_error.groupby("well", sort=False)["error2"].mean().to_numpy(np.float64)
    )
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    for name, mask in {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _load_canonical_folds(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    well_column = next(
        (column for column in ("well", "well_id", "Well") if column in frame), None
    )
    fold_column = next(
        (column for column in ("fold", "Fold", "outer_fold") if column in frame), None
    )
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify fold columns in {path}: {frame.columns.tolist()}")
    frame = frame[[well_column, fold_column]].drop_duplicates()
    frame[well_column] = frame[well_column].astype(str)
    if frame[well_column].duplicated().any():
        raise RuntimeError(f"Canonical fold source has duplicate wells: {path}")
    mapping = dict(zip(frame[well_column], frame[fold_column].astype(int)))
    if len(mapping) != EXPECTED_WELLS or set(mapping.values()) != set(EXPECTED_FOLDS):
        raise RuntimeError(
            f"Invalid canonical fold map: wells={len(mapping)} folds={sorted(set(mapping.values()))}"
        )
    return mapping


for required in (FULL_CP_PATH, NEIGHBOR_BANK, NEIGHBOR_META):
    if not required.is_file():
        raise FileNotFoundError(required)
pool_path = _first_file(POOL_CANDIDATES, "Neighbor-aligned candidate pool")
fold_path = _first_file(FOLD_CANDIDATES, "canonical five-fold metadata")
canonical_fold = _load_canonical_folds(fold_path)

full_header = pq.read_schema(FULL_CP_PATH).names
required_full = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    CP8_COLUMN, CP8_FALLBACK_COLUMN,
}
missing_full = sorted(required_full.difference(full_header))
if missing_full:
    raise RuntimeError(f"Complete CP8 OOF is missing columns: {missing_full}")
full_columns = sorted(required_full)
if CORE_COLUMN in full_header:
    full_columns.append(CORE_COLUMN)
frame = pd.read_parquet(FULL_CP_PATH, columns=full_columns)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

if frame["id"].duplicated().any():
    raise RuntimeError("Complete CP8 OOF contains duplicate ids")
if len(frame) != EXPECTED_ROWS or frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(
        f"Complete CP8 OOF coverage is not exact: rows={len(frame)} "
        f"wells={frame['well'].nunique()}"
    )
if frame[["tvt_true", "md_since"]].isna().any().any():
    raise RuntimeError("Complete CP8 OOF contains missing truth or distance")

mapped_fold = frame["well"].map(canonical_fold)
if mapped_fold.isna().any():
    raise RuntimeError(f"Canonical folds miss {int(mapped_fold.isna().sum())} OOF rows")
declared_fold = frame["fold"].astype(int).to_numpy()
canonical_values = mapped_fold.astype(int).to_numpy()
fold_mismatch = declared_fold != canonical_values
if fold_mismatch.any():
    examples = frame.loc[fold_mismatch, ["well", "fold"]].drop_duplicates().head(20)
    raise RuntimeError(
        "V18 and canonical fold identities disagree; refusing relabeling:\n"
        + examples.to_string(index=False)
    )
frame["fold"] = canonical_values.astype(np.int8)
fold_wells = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if set(fold_wells) != set(EXPECTED_FOLDS) or min(fold_wells.values()) < 154:
    raise RuntimeError(f"Incomplete canonical fold coverage: {fold_wells}")

cp8_raw = frame[CP8_COLUMN].to_numpy(np.float64)
cp8_fallback = frame[CP8_FALLBACK_COLUMN].to_numpy(np.float64)
cp8_missing = ~np.isfinite(cp8_raw)
if np.any(cp8_missing & ~np.isfinite(cp8_fallback)):
    raise RuntimeError("Rows exist with neither CP8 nor its registered fallback")
cp8 = np.where(cp8_missing, cp8_fallback, cp8_raw)

pool_header = pd.read_csv(pool_path, nrows=0).columns.tolist()
pool_columns = [column for column in ("id", "well", "row_idx") if column in pool_header]
if "id" not in pool_columns:
    raise RuntimeError(f"Candidate pool has no id: {pool_path}")
pool = pd.read_csv(
    pool_path, usecols=pool_columns, dtype={"id": "string", "well": "string"}
)
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if pool["id"].duplicated().any():
    raise RuntimeError("Candidate pool contains duplicate ids")

with np.load(NEIGHBOR_BANK, allow_pickle=False) as handle:
    if NEIGHBOR_METHOD not in handle.files:
        raise RuntimeError(f"Neighbor bank lacks {NEIGHBOR_METHOD}")
    bank_neighbor = handle[NEIGHBOR_METHOD].astype(np.float64)
if len(bank_neighbor) != len(pool):
    raise RuntimeError(
        f"Neighbor bank/pool length mismatch: {len(bank_neighbor)}/{len(pool)}"
    )
if len(frame) == len(pool) and np.array_equal(frame["id"].to_numpy(), pool["id"].to_numpy()):
    neighbor = bank_neighbor.copy()
else:
    pool_position = pd.Series(np.arange(len(pool), dtype=np.int64), index=pool["id"])
    position = frame["id"].map(pool_position)
    if position.isna().any():
        examples = frame.loc[position.isna(), ["id", "well", "row_idx"]].head(20)
        raise RuntimeError(
            f"Neighbor bank misses {int(position.isna().sum())} complete-OOF rows:\n"
            + examples.to_string(index=False)
        )
    neighbor = bank_neighbor[position.to_numpy(np.int64)]
del pool, bank_neighbor

# V26's all-geometry recipe activates a well only when at least 80% of its
# scored rows have a finite Neighbor path.  The per-row active mask additionally
# freezes the first 1000 ft and excludes missing Neighbor values.
distance = frame["md_since"].to_numpy(np.float64)
finite_neighbor = np.isfinite(neighbor)
active = np.zeros(len(frame), dtype=bool)
coverage_rows = []
for well, indices in frame.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    coverage = float(finite_neighbor[indices].mean())
    eligible = coverage >= MIN_NEIGHBOR_COVERAGE
    if eligible:
        active[indices] = finite_neighbor[indices] & (distance[indices] > START_FT)
    coverage_rows.append(
        {"well": str(well), "fold": int(frame.loc[indices[0], "fold"]),
         "neighbor_coverage": coverage, "eligible": bool(eligible)}
    )
coverage_frame = pd.DataFrame(coverage_rows)
coverage_frame.to_csv(OUT_DIR / "neighbor_coverage_bywell.csv", index=False)

raw_move = np.zeros(len(frame), np.float64)
raw_move[active] = neighbor[active] - cp8[active]
capped_move = np.clip(raw_move, -CAP_FT, CAP_FT)
tail_profile = _smoothstep(distance)
incumbent = cp8 + DOSE * tail_profile * capped_move
if not np.array_equal(incumbent[distance <= START_FT], cp8[distance <= START_FT]):
    raise RuntimeError("Tail-only replay changed the frozen first 1000 ft")
if not np.isfinite(incumbent).all():
    raise RuntimeError("Five-fold incumbent contains non-finite predictions")

parity = {
    "available": OLD_V33_PATH.is_file(),
    "rows": 0,
    "wells": 0,
    "prediction_column": None,
    "maximum_abs_difference": None,
}
if OLD_V33_PATH.is_file():
    old_header = pq.read_schema(OLD_V33_PATH).names
    old_prediction_column = next(
        (
            column
            for column in ("current_dose_0p20", "selected_prediction", "prediction")
            if column in old_header
        ),
        None,
    )
    if old_prediction_column is None:
        raise RuntimeError(f"Cannot identify V33 prediction column: {old_header}")
    old = pd.read_parquet(OLD_V33_PATH, columns=["id", old_prediction_column])
    old["id"] = old["id"].astype(str)
    if old["id"].duplicated().any():
        raise RuntimeError("Old V33 replay contains duplicate ids")
    comparison = frame[["id", "well"]].copy()
    comparison["v66"] = incumbent
    comparison = comparison.merge(old, on="id", how="inner", validate="one_to_one")
    difference = np.abs(
        comparison["v66"].to_numpy(np.float64)
        - comparison[old_prediction_column].to_numpy(np.float64)
    )
    maximum = float(difference.max()) if len(difference) else np.inf
    parity = {
        "available": True,
        "rows": int(len(comparison)),
        "wells": int(comparison["well"].nunique()),
        "prediction_column": old_prediction_column,
        "maximum_abs_difference": maximum,
    }
    if maximum > PARITY_TOLERANCE_FT:
        raise RuntimeError(
            f"Full replay does not match old V33 on shared rows: max_abs={maximum}"
        )

metric_rows = []
for heldout_fold in (0, *EXPECTED_FOLDS):
    mask = (
        np.ones(len(frame), dtype=bool)
        if heldout_fold == 0
        else frame["fold"].to_numpy(np.int8) == heldout_fold
    )
    local = frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in {
        "full_cp8_ordered": cp8,
        "incumbent_6p377_replay": incumbent,
    }.items():
        metric_rows.append(
            {"fold": int(heldout_fold), **_metric(local, prediction[mask], endpoint)}
        )
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUT_DIR / "fivefold_metrics.csv", index=False)

output_columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since"]
output = frame[output_columns].copy()
if CORE_COLUMN in frame:
    output[CORE_COLUMN] = frame[CORE_COLUMN].to_numpy(np.float32)
output["full_cp8_ordered"] = cp8.astype(np.float32)
output["neighbor_absolute"] = neighbor.astype(np.float32)
output["neighbor_active"] = active
output["tail_profile"] = tail_profile.astype(np.float32)
output["incumbent_6p377_replay"] = incumbent.astype(np.float32)
output_path = OUT_DIR / "incumbent_6p377_full_fivefold_oof.parquet"
output.to_parquet(output_path, index=False, compression="zstd")

checks = {
    "rows_exact": bool(len(output) == EXPECTED_ROWS),
    "wells_exact": bool(output["well"].nunique() == EXPECTED_WELLS),
    "canonical_folds_exact": bool(not fold_mismatch.any()),
    "each_fold_ge_154_wells": bool(min(fold_wells.values()) >= 154),
    "all_predictions_finite": bool(np.isfinite(incumbent).all()),
    "prefix_frozen_exact": bool(
        np.array_equal(incumbent[distance <= START_FT], cp8[distance <= START_FT])
    ),
    "old_v33_parity": bool(
        parity["available"]
        and parity["rows"] > 0
        and parity["maximum_abs_difference"] <= PARITY_TOLERANCE_FT
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_773well_canonical_fivefold_incumbent_reconstruction",
    "selection_or_tuning_count": 0,
    "sources": {
        "full_cp8": str(FULL_CP_PATH),
        "neighbor_bank": str(NEIGHBOR_BANK),
        "neighbor_pool": str(pool_path),
        "canonical_folds": str(fold_path),
        "old_partial_v33": str(OLD_V33_PATH),
    },
    "recipe": {
        "cp8_column": CP8_COLUMN,
        "cp8_fallback_column": CP8_FALLBACK_COLUMN,
        "neighbor_method": NEIGHBOR_METHOD,
        "geometry_gate": "all",
        "minimum_neighbor_coverage": MIN_NEIGHBOR_COVERAGE,
        "start_ft": START_FT,
        "transition_ft": TRANSITION_FT,
        "reanchor": "none",
        "cap_ft": CAP_FT,
        "dose": DOSE,
    },
    "coverage": {
        "rows": int(len(output)),
        "wells": int(output["well"].nunique()),
        "fold_wells": {str(key): int(value) for key, value in fold_wells.items()},
        "eligible_neighbor_wells": int(coverage_frame["eligible"].sum()),
        "active_rows": int(active.sum()),
        "cp8_fallback_rows": int(cp8_missing.sum()),
    },
    "old_v33_parity": parity,
    "checks": checks,
    "strict_pass": strict_pass,
    "output": str(output_path),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> generate the coarse endpoint independently on all five canonical folds"
        if strict_pass
        else "FAIL -> do not run another coarse fusion experiment"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, allow_nan=False), encoding="utf-8"
)

print("\nV66 FIVEFOLD METRICS")
print(metrics.to_string(index=False), flush=True)
print("\nV66 FINAL VERDICT")
print(json.dumps(verdict, indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
if not strict_pass:
    raise RuntimeError("V66 strict lineage reconstruction failed; inspect verdict.json")
