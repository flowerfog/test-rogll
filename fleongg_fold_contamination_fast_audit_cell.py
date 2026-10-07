"""Fast audit of spatial-feature fold contamination in the fleongg pipeline.

This is a notebook cell, not a model experiment. Run it after the original
notebook defines ``FormationPlaneKNN`` and ``DenseANCCImputer``. It compares:

1. current LOO: FI/DI built from all wells, excluding only the target well;
2. fold clean: FI/DI built only from the four training folds.

Only sparse target-well rows are queried, so this is much cheaper than rerunning
PF, beam, boosters, or the full Gold candidate pool.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")

N_FOLDS = 5
KNOWN_QUERY_POINTS = 64
EVAL_QUERY_POINTS = 128
DENSE_NFETCH = 512
SEED = 20260711

required = ["FormationPlaneKNN", "DenseANCCImputer"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the original fleongg class-definition cells first; missing: "
        + repr(missing)
    )

OOF_DIR.mkdir(parents=True, exist_ok=True)


def rmse_from_sse(sse, n):
    return float(np.sqrt(float(sse) / max(1, int(n))))


def evenly_spaced(index, n_points):
    index = np.asarray(index, dtype=int)
    if len(index) <= n_points:
        return index
    position = np.unique(
        np.round(np.linspace(0, len(index) - 1, int(n_points))).astype(int)
    )
    return index[position]


def assign_balanced_group_folds(meta, n_folds=5):
    """Match non-shuffled GroupKFold's row-count balancing principle."""
    ordered = meta.sort_values(["n_eval", "well"], ascending=[False, True])
    loads = np.zeros(n_folds, dtype=np.int64)
    assignment = {}
    for row in ordered.itertuples(index=False):
        fold = int(np.argmin(loads))
        assignment[str(row.well)] = fold + 1
        loads[fold] += int(row.n_eval)
    return assignment, loads


def anchored_surface_predictions(fi, xy_known, xy_eval, z_known, z_eval, tvt_known, wid):
    surface_known, _ = fi.impute(xy_known, self_wid=wid)
    surface_eval, distance = fi.impute(xy_eval, self_wid=wid)
    surface_known = np.asarray(surface_known, dtype=float)
    surface_eval = np.asarray(surface_eval, dtype=float)

    candidates = []
    for column in range(surface_known.shape[1]):
        known_surface = surface_known[:, column]
        eval_surface = surface_eval[:, column]
        valid_known = (
            np.isfinite(known_surface)
            & np.isfinite(z_known)
            & np.isfinite(tvt_known)
        )
        if int(valid_known.sum()) < 20:
            continue
        offset = float(
            np.nanmedian(tvt_known[valid_known] + z_known[valid_known] - known_surface[valid_known])
        )
        candidates.append(-z_eval + eval_surface + offset)
    if not candidates:
        raise RuntimeError("no finite formation-plane candidates")
    return np.nanmean(np.column_stack(candidates), axis=1), np.asarray(distance, float)


def anchored_dense_prediction(di, xy_known, xy_eval, z_known, z_eval, tvt_known, wid):
    dense_known, _, _ = di.impute(
        xy_known, self_wid=wid, nfetch=DENSE_NFETCH
    )
    dense_eval, _, distance = di.impute(
        xy_eval, self_wid=wid, nfetch=DENSE_NFETCH
    )
    dense_known = np.asarray(dense_known, dtype=float)
    dense_eval = np.asarray(dense_eval, dtype=float)
    valid_known = (
        np.isfinite(dense_known)
        & np.isfinite(z_known)
        & np.isfinite(tvt_known)
    )
    if int(valid_known.sum()) < 20:
        raise RuntimeError("too few finite dense known rows")
    offset = float(
        np.nanmedian(tvt_known[valid_known] + z_known[valid_known] - dense_known[valid_known])
    )
    return -z_eval + dense_eval + offset, np.asarray(distance, float)


# ---------------------------------------------------------------------------
# Metadata and fold assignment. No target values are used here.
# ---------------------------------------------------------------------------
meta_rows = []
paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
for path in tqdm(paths, desc="scan spatial audit metadata"):
    wid = path.name.replace("__horizontal_well.csv", "")
    tvt_input = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"].to_numpy(float)
    n_eval = int(np.sum(~np.isfinite(tvt_input)))
    n_known = int(np.sum(np.isfinite(tvt_input)))
    if n_known >= 64 and n_eval >= 64:
        meta_rows.append({"well": wid, "n_known": n_known, "n_eval": n_eval})

meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
fold_map, fold_loads = assign_balanced_group_folds(meta, N_FOLDS)
meta["fold"] = meta["well"].map(fold_map).astype(int)
print("eligible wells:", len(meta), "fold row loads:", fold_loads.tolist())


# ---------------------------------------------------------------------------
# Build the current all-well objects and five honest fold-clean objects.
# ---------------------------------------------------------------------------
all_wells = meta["well"].astype(str).tolist()
print("building current all-well FI/DI", flush=True)
current_fi = FormationPlaneKNN(all_wells, DATA_ROOT / "train")
current_di = DenseANCCImputer(all_wells, DATA_ROOT / "train")

fold_objects = {}
for fold in range(1, N_FOLDS + 1):
    reference_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    print(f"building fold={fold} FI/DI references={len(reference_wells)}", flush=True)
    fold_objects[fold] = (
        FormationPlaneKNN(reference_wells, DATA_ROOT / "train"),
        DenseANCCImputer(reference_wells, DATA_ROOT / "train"),
    )


# ---------------------------------------------------------------------------
# Sparse all-well scoring. TVT is isolated and used only after predictions exist.
# ---------------------------------------------------------------------------
records = []
failures = []
started = time.time()

for number, row in enumerate(tqdm(meta.itertuples(index=False), total=len(meta), desc="fold contamination audit"), 1):
    wid = str(row.well)
    fold = int(row.fold)
    try:
        source = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
        test_columns = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
        missing_columns = [c for c in test_columns + ["TVT"] if c not in source.columns]
        if missing_columns:
            raise RuntimeError(f"missing columns: {missing_columns}")

        # Prediction functions receive only hidden-test schema columns.
        target = source[test_columns].copy()
        truth_all = source["TVT"].to_numpy(float)
        tvt_input = target["TVT_input"].to_numpy(float)
        xy = target[["X", "Y"]].to_numpy(float)
        z = target["Z"].to_numpy(float)

        known = np.flatnonzero(
            np.isfinite(tvt_input) & np.isfinite(z) & np.isfinite(xy).all(axis=1)
        )
        eval_index = np.flatnonzero(
            ~np.isfinite(tvt_input)
            & np.isfinite(truth_all)
            & np.isfinite(z)
            & np.isfinite(xy).all(axis=1)
        )
        known = evenly_spaced(known, KNOWN_QUERY_POINTS)
        eval_index = evenly_spaced(eval_index, EVAL_QUERY_POINTS)
        if len(known) < 20 or len(eval_index) < 30:
            raise RuntimeError("too few valid sampled rows")

        xy_known = xy[known]
        xy_eval = xy[eval_index]
        z_known = z[known]
        z_eval = z[eval_index]
        tvt_known = tvt_input[known]
        truth = truth_all[eval_index]

        fold_fi, fold_di = fold_objects[fold]

        current_surface, current_surface_distance = anchored_surface_predictions(
            current_fi, xy_known, xy_eval, z_known, z_eval, tvt_known, wid
        )
        clean_surface, clean_surface_distance = anchored_surface_predictions(
            fold_fi, xy_known, xy_eval, z_known, z_eval, tvt_known, wid
        )
        current_dense, current_dense_distance = anchored_dense_prediction(
            current_di, xy_known, xy_eval, z_known, z_eval, tvt_known, wid
        )
        clean_dense, clean_dense_distance = anchored_dense_prediction(
            fold_di, xy_known, xy_eval, z_known, z_eval, tvt_known, wid
        )

        item = {
            "well": wid,
            "fold": fold,
            "n": len(eval_index),
            "current_surface_sse": float(np.sum((current_surface - truth) ** 2)),
            "clean_surface_sse": float(np.sum((clean_surface - truth) ** 2)),
            "current_dense_sse": float(np.sum((current_dense - truth) ** 2)),
            "clean_dense_sse": float(np.sum((clean_dense - truth) ** 2)),
            "surface_sum_abs_diff": float(np.sum(np.abs(clean_surface - current_surface))),
            "surface_p95_abs_diff": float(np.quantile(np.abs(clean_surface - current_surface), 0.95)),
            "dense_sum_abs_diff": float(np.sum(np.abs(clean_dense - current_dense))),
            "dense_p95_abs_diff": float(np.quantile(np.abs(clean_dense - current_dense), 0.95)),
            "current_surface_distance_mean": float(np.nanmean(current_surface_distance)),
            "clean_surface_distance_mean": float(np.nanmean(clean_surface_distance)),
            "current_dense_distance_mean": float(np.nanmean(current_dense_distance)),
            "clean_dense_distance_mean": float(np.nanmean(clean_dense_distance)),
        }
        records.append(item)
    except Exception as exc:
        failures.append({"well": wid, "fold": fold, "reason": repr(exc)})

    if number % 100 == 0:
        print(
            f"{number}/{len(meta)} ok={len(records)} elapsed={(time.time()-started)/60:.1f} min",
            flush=True,
        )

bywell = pd.DataFrame(records)
failure_df = pd.DataFrame(failures)
if len(bywell) < 0.9 * len(meta):
    display(failure_df.head(50))
    raise RuntimeError(f"too many failures: {len(failure_df)} / {len(meta)}")


def summarize(group, candidate):
    n = int(group["n"].sum())
    current_rmse = rmse_from_sse(group[f"current_{candidate}_sse"].sum(), n)
    clean_rmse = rmse_from_sse(group[f"clean_{candidate}_sse"].sum(), n)
    return {
        "candidate": candidate,
        "rows": n,
        "current_loo_rmse": current_rmse,
        "fold_clean_rmse": clean_rmse,
        "fold_clean_minus_current_rmse": clean_rmse - current_rmse,
        "mean_abs_curve_diff": float(group[f"{candidate}_sum_abs_diff"].sum() / n),
        "mean_well_p95_curve_diff": float(group[f"{candidate}_p95_abs_diff"].mean()),
        "current_neighbor_distance_mean": float(group[f"current_{candidate}_distance_mean"].mean()),
        "clean_neighbor_distance_mean": float(group[f"clean_{candidate}_distance_mean"].mean()),
    }


summary = pd.DataFrame(
    [summarize(bywell, "surface"), summarize(bywell, "dense")]
)
fold_rows = []
for fold, group in bywell.groupby("fold", sort=True):
    for candidate in ("surface", "dense"):
        row = summarize(group, candidate)
        row["fold"] = int(fold)
        fold_rows.append(row)
fold_summary = pd.DataFrame(fold_rows)

material = bool(
    (
        summary["fold_clean_minus_current_rmse"] > 0.50
    ).any()
    or (summary["mean_abs_curve_diff"] > 1.00).any()
)
negligible = bool(
    (
        summary["fold_clean_minus_current_rmse"].abs() < 0.10
    ).all()
    and (summary["mean_abs_curve_diff"] < 0.25).all()
)

print("\nOVERALL")
display(summary)
print("\nFOLDS")
display(fold_summary)
print(
    "AUDIT VERDICT:",
    "MATERIAL" if material else ("NEGLIGIBLE" if negligible else "INTERMEDIATE"),
)

prefix = OOF_DIR / "fleongg_fold_contamination_fast_audit"
bywell.to_csv(str(prefix) + "_bywell.csv", index=False)
summary.to_csv(str(prefix) + "_summary.csv", index=False)
fold_summary.to_csv(str(prefix) + "_folds.csv", index=False)
failure_df.to_csv(str(prefix) + "_failures.csv", index=False)
print("saved prefix:", prefix)
