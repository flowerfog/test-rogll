"""M342 validation similarity/leakage audit.

Read-only audit for the complete V66 OOF.  It does not retrain a model and
does not touch submission.csv.  The audit asks whether canonical folds split
highly similar wells across train/validation, using only train-well metadata,
typewell GR fingerprints, trajectory geometry, and XY.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EXPERIMENT = "main342_validation_similarity_leakage_audit_v1"
ROOT = Path(os.environ.get("ROGII_ROOT", "/root/autodl-tmp"))
OOF = ROOT / "oof_exports"
DATA = ROOT / "rogii-wellbore-geology-prediction"
OUT = OOF / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)

V66 = OOF / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet"
FOLD_FILES = (
    OOF / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF / "frontier_6p39_existing_models_oof_v0" / "well_metadata.csv",
)
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
GR_POINTS = 128
GEOM_POINTS = 32
NEIGHBOURS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find(path: Path, label: str) -> Path:
    if path.is_file():
        return path
    for candidate in FOLD_FILES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {label}: {path}")


def _resample(axis: np.ndarray, values: np.ndarray, points: int) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    values = np.asarray(values, np.float64)
    mask = np.isfinite(axis) & np.isfinite(values)
    axis, values = axis[mask], values[mask]
    if len(axis) < 2:
        return np.zeros(points, np.float64)
    order = np.argsort(axis, kind="stable")
    axis, values = axis[order], values[order]
    axis, unique = np.unique(axis, return_index=True)
    values = values[unique]
    if len(axis) < 2 or axis[-1] <= axis[0]:
        return np.zeros(points, np.float64)
    return np.interp(np.linspace(axis[0], axis[-1], points), axis, values)


def _standard(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    centre = np.nanmedian(values)
    scale = 1.4826 * np.nanmedian(np.abs(values - centre))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-8:
        return np.zeros_like(values)
    return np.nan_to_num((values - centre) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def _fingerprint(well: str) -> tuple[np.ndarray, float, float, float, float]:
    horizontal = DATA / "train" / f"{well}__horizontal_well.csv"
    typewell = DATA / "train" / f"{well}__typewell.csv"
    hw = pd.read_csv(horizontal, usecols=lambda c: c in {"MD", "X", "Y", "Z"})
    tw = pd.read_csv(typewell, usecols=lambda c: c in {"TVT", "GR"})
    md = pd.to_numeric(hw.get("MD", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    z = pd.to_numeric(hw.get("Z", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    x = pd.to_numeric(hw.get("X", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    y = pd.to_numeric(hw.get("Y", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(tw.get("TVT", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    gr = pd.to_numeric(tw.get("GR", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    if len(tvt) < 2 or len(gr) < 2:
        raise RuntimeError("insufficient typewell samples")
    fp = _standard(_resample(tvt, gr, GR_POINTS))
    if len(md) < 2:
        raise RuntimeError("insufficient trajectory samples")
    valid = np.isfinite(md) & np.isfinite(z)
    geom = _resample(md[valid], np.gradient(z[valid], md[valid]), GEOM_POINTS)
    valid_xy = np.isfinite(x) & np.isfinite(y)
    if not valid_xy.any():
        raise RuntimeError("missing XY")
    return np.r_[fp, _standard(geom)], float(np.nanmedian(x[valid_xy])), float(np.nanmedian(y[valid_xy])), float(np.ptp(x[valid_xy])), float(np.ptp(y[valid_xy]))


started = time.time()
v66_path = _find(V66, "V66 OOF")
fold_path = _find(FOLD_FILES[0], "canonical folds")
oof = pd.read_parquet(v66_path, columns=["id", "well", "row_idx", "fold", "tvt_true", "incumbent_6p377_replay"])
folds = pd.read_csv(fold_path)
well_col = next(c for c in ("well", "well_id") if c in folds)
fold_col = next(c for c in ("fold", "outer_fold") if c in folds)
fold_lookup = dict(zip(folds[well_col].astype(str), folds[fold_col].astype(int)))
oof["fold"] = oof["well"].astype(str).map(fold_lookup).astype(int)
wells = sorted(oof["well"].astype(str).unique())
if len(oof) != EXPECTED_ROWS or len(wells) != EXPECTED_WELLS:
    raise RuntimeError(f"Unexpected V66 coverage: rows={len(oof)} wells={len(wells)}")

records, failures = [], []
for number, well in enumerate(wells, 1):
    try:
        fp, x, y, x_span, y_span = _fingerprint(well)
        records.append({"well": well, "fold": fold_lookup[well], "x": x, "y": y, "x_span": x_span, "y_span": y_span, "fingerprint": fp})
    except Exception as error:
        failures.append({"well": well, "reason": repr(error)})
    if number % 100 == 0 or number == len(wells):
        print(f"M342 SIMILARITY {number}/{len(wells)} failures={len(failures)}", flush=True)
if failures:
    pd.DataFrame(failures).to_csv(OUT / "fingerprint_failures.csv", index=False)
if len(records) < 700:
    raise RuntimeError(f"Insufficient fingerprint coverage: {len(records)}/{len(wells)}")

meta = pd.DataFrame(records)
fp = np.vstack(meta["fingerprint"].to_numpy())
fp /= np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
xy = meta[["x", "y"]].to_numpy(float)
xy = (xy - np.nanmedian(xy, axis=0)) / np.maximum(np.nanstd(xy, axis=0), 1e-8)
descriptor = np.c_[fp, xy * 0.25]
tree = cKDTree(descriptor)
distances, indices = tree.query(descriptor, k=min(NEIGHBOURS + 1, len(meta)))

rows = []
for i, well in enumerate(meta["well"]):
    neighbour_ids = indices[i, 1:]
    neighbour_dist = distances[i, 1:]
    neighbour_folds = meta.iloc[neighbour_ids]["fold"].to_numpy(int)
    same = neighbour_folds == int(meta.iloc[i]["fold"])
    cross = ~same
    rows.append({
        "well": well,
        "fold": int(meta.iloc[i]["fold"]),
        "nearest_distance": float(neighbour_dist[0]),
        "nearest_cross_fold": bool(cross[0]),
        "cross_fold_fraction_top8": float(cross.mean()),
        "same_fold_fraction_top8": float(same.mean()),
        "cross_fold_neighbours_top8": int(cross.sum()),
    })
neighbour_frame = pd.DataFrame(rows)
neighbour_frame.to_csv(OUT / "well_similarity_neighbours.csv", index=False)

oof["abs_error"] = np.abs(oof["tvt_true"] - oof["incumbent_6p377_replay"])
oof["row_sq_error"] = np.square(oof["tvt_true"] - oof["incumbent_6p377_replay"])
well_error = oof.groupby("well").agg(rows=("id", "size"), sq_error=("row_sq_error", "sum"), abs_error=("abs_error", "mean")).reset_index()
well_error["well_rmse"] = np.sqrt(well_error["sq_error"] / well_error["rows"])
joined = well_error.merge(neighbour_frame, on="well", validate="one_to_one")
joined.to_csv(OUT / "well_similarity_error_join.csv", index=False)

similarity_bins = pd.qcut(joined["cross_fold_fraction_top8"], q=4, duplicates="drop")
strata = joined.assign(stratum=similarity_bins).groupby("stratum", observed=True).agg(
    wells=("well", "size"), mean_well_rmse=("well_rmse", "mean"), median_well_rmse=("well_rmse", "median"), mean_nearest_distance=("nearest_distance", "mean"), mean_cross_fold_fraction=("cross_fold_fraction_top8", "mean")
).reset_index()
strata.to_csv(OUT / "similarity_error_strata.csv", index=False)

fold_mix = neighbour_frame.groupby("fold").agg(wells=("well", "size"), mean_cross_fold_fraction=("cross_fold_fraction_top8", "mean"), nearest_cross_fold_fraction=("nearest_cross_fold", "mean"), median_nearest_distance=("nearest_distance", "median")).reset_index()
fold_mix.to_csv(OUT / "fold_similarity_mix.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "read_only_typewell_geometry_xy_neighbour_graph_audit",
    "source": {"v66": str(v66_path), "v66_sha256": _sha256(v66_path), "folds": str(fold_path)},
    "coverage": {"rows": int(len(oof)), "wells": int(len(wells)), "fingerprint_wells": int(len(records)), "failures": int(len(failures))},
    "descriptor": {"typewell_gr_points": GR_POINTS, "geometry_points": GEOM_POINTS, "top_k": NEIGHBOURS, "xy_weight": 0.25},
    "nearest": {"nearest_cross_fold_fraction": float(neighbour_frame["nearest_cross_fold"].mean()), "mean_cross_fold_fraction_top8": float(neighbour_frame["cross_fold_fraction_top8"].mean()), "median_nearest_distance": float(neighbour_frame["nearest_distance"].median())},
    "interpretation": "Descriptive leakage audit only; no fold reassignment and no deployment recommendation.",
    "elapsed_minutes": (time.time() - started) / 60.0,
    "output": str(OUT),
}
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("M342 VALIDATION SIMILARITY AUDIT", json.dumps(verdict, indent=2), flush=True)
