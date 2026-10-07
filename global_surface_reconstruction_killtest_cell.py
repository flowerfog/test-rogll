"""Fold-clean global structural-surface reconstruction kill-test.

This notebook cell learns the mean of the six train-only structural surfaces as
a function of (X, Y). Validation target wells are stripped to hidden-test schema,
and their whole fold is excluded from the surface training set. A per-well datum
is estimated only from visible TVT_input before scoring the eval suffix.

The four fixed models are intentionally not tuned on this run:
  * local weighted plane over well medians;
  * dense inverse-distance interpolation;
  * local thin-plate RBF;
  * ExtraTrees regression.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
CACHE_VERSION = "v2_last_anchor_all6_p80_q256"
CACHE_DIR = OOF_DIR / f"global_surface_killtest_{CACHE_VERSION}"

SURFACES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
REFERENCE_POINTS_PER_WELL = 80
EVAL_POINTS_PER_WELL = 256
N_FOLDS = 5
RANDOM_STATE = 20260711

LOCAL_PLANE_K = 10
DENSE_IDW_K = 24
RBF_NEIGHBORS = 32
RBF_SMOOTHING = 0.10
PREDICT_CHUNK = 4096

OOF_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(
    {
        "experiment": "global_surface_reconstruction",
        "surface_target": "row_mean_of_six_surfaces",
        "reference_points_per_well": REFERENCE_POINTS_PER_WELL,
        "eval_points_per_well": EVAL_POINTS_PER_WELL,
        "rbf_neighbors": RBF_NEIGHBORS,
        "cache": str(CACHE_DIR),
    }
)


def rmse_from_sse(sse, n):
    return float(np.sqrt(float(sse) / max(1, int(n))))


def evenly_spaced(index, n_points):
    index = np.asarray(index, dtype=int)
    if len(index) <= n_points:
        return index
    take = np.unique(
        np.round(np.linspace(0, len(index) - 1, int(n_points))).astype(int)
    )
    return index[take]


def balanced_group_folds(meta):
    ordered = meta.sort_values(["n_eval", "well"], ascending=[False, True])
    loads = np.zeros(N_FOLDS, dtype=np.int64)
    assignment = {}
    for row in ordered.itertuples(index=False):
        fold = int(np.argmin(loads))
        assignment[str(row.well)] = fold
        loads[fold] += int(row.n_eval)
    return assignment, loads


class LocalPlaneSurface:
    def fit(self, well_meta):
        self.xy = well_meta[["x", "y"]].to_numpy(float)
        self.value = well_meta["surface"].to_numpy(float)
        self.scale = np.where(self.xy.std(axis=0) < 1e-6, 1.0, self.xy.std(axis=0))
        self.tree = cKDTree(self.xy / self.scale)
        return self

    def predict(self, xy):
        xy = np.asarray(xy, dtype=float)
        k = min(LOCAL_PLANE_K, len(self.xy))
        distance, index = self.tree.query(xy / self.scale, k=k, workers=-1)
        distance = np.atleast_2d(distance)
        index = np.atleast_2d(index)
        if distance.shape[0] != len(xy):
            distance = distance.T
            index = index.T

        output = np.empty(len(xy), dtype=float)
        for row in range(len(xy)):
            neighbor_xy = self.xy[index[row]]
            neighbor_value = self.value[index[row]]
            weight = 1.0 / (distance[row] + 1e-3)
            design = np.column_stack(
                [neighbor_xy[:, 0], neighbor_xy[:, 1], np.ones(len(neighbor_xy))]
            )
            sw = np.sqrt(weight)
            try:
                coefficient = np.linalg.lstsq(
                    design * sw[:, None], neighbor_value * sw, rcond=None
                )[0]
                output[row] = (
                    xy[row, 0] * coefficient[0]
                    + xy[row, 1] * coefficient[1]
                    + coefficient[2]
                )
            except Exception:
                output[row] = float(np.average(neighbor_value, weights=weight))
        return output


class DenseIDWSurface:
    def fit(self, xy, value):
        self.xy = np.asarray(xy, dtype=float)
        self.value = np.asarray(value, dtype=float)
        self.scale = np.where(self.xy.std(axis=0) < 1e-6, 1.0, self.xy.std(axis=0))
        self.tree = cKDTree(self.xy / self.scale)
        return self

    def predict(self, xy):
        xy = np.asarray(xy, dtype=float)
        k = min(DENSE_IDW_K, len(self.xy))
        distance, index = self.tree.query(xy / self.scale, k=k, workers=-1)
        distance = np.atleast_2d(distance)
        index = np.atleast_2d(index)
        if distance.shape[0] != len(xy):
            distance = distance.T
            index = index.T
        weight = 1.0 / (distance + 1e-3)
        return np.sum(weight * self.value[index], axis=1) / np.sum(weight, axis=1)


class ThinPlateSurface:
    def fit(self, xy, value):
        self.xy_mean = np.mean(xy, axis=0)
        self.xy_scale = np.where(np.std(xy, axis=0) < 1e-6, 1.0, np.std(xy, axis=0))
        self.value_center = float(np.mean(value))
        x_scaled = (np.asarray(xy, float) - self.xy_mean) / self.xy_scale
        y_centered = np.asarray(value, float) - self.value_center
        self.model = RBFInterpolator(
            x_scaled,
            y_centered,
            neighbors=min(RBF_NEIGHBORS, len(x_scaled)),
            smoothing=RBF_SMOOTHING,
            kernel="thin_plate_spline",
            degree=1,
        )
        return self

    def predict(self, xy):
        xy = np.asarray(xy, dtype=float)
        output = np.empty(len(xy), dtype=float)
        for start in range(0, len(xy), PREDICT_CHUNK):
            stop = min(len(xy), start + PREDICT_CHUNK)
            scaled = (xy[start:stop] - self.xy_mean) / self.xy_scale
            output[start:stop] = self.model(scaled).reshape(-1) + self.value_center
        return output


class ExtraTreesSurface:
    def fit(self, xy, value):
        self.model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        self.model.fit(np.asarray(xy, float), np.asarray(value, float))
        return self

    def predict(self, xy):
        return self.model.predict(np.asarray(xy, float))


MODEL_BUILDERS = {
    "local_plane": lambda train: LocalPlaneSurface().fit(train["well_meta"]),
    "dense_idw": lambda train: DenseIDWSurface().fit(train["xy"], train["surface"]),
    "thin_plate_rbf": lambda train: ThinPlateSurface().fit(train["xy"], train["surface"]),
    "extra_trees": lambda train: ExtraTreesSurface().fit(train["xy"], train["surface"]),
}


# ---------------------------------------------------------------------------
# Read each well once and cache sparse reference/query rows in memory.
# ---------------------------------------------------------------------------
well_data = {}
meta_rows = []
reference_parts = []
files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))

for path in tqdm(files, desc="load global-surface data"):
    wid = path.name.replace("__horizontal_well.csv", "")
    source = pd.read_csv(path)
    required = ["X", "Y", "Z", "TVT", "TVT_input", *SURFACES]
    if any(column not in source.columns for column in required):
        continue

    xy_all = source[["X", "Y"]].to_numpy(float)
    z_all = source["Z"].to_numpy(float)
    tvt_input = source["TVT_input"].to_numpy(float)
    truth_all = source["TVT"].to_numpy(float)
    surface_matrix = source[SURFACES].to_numpy(float)
    surface_mean = np.nanmean(surface_matrix, axis=1)

    reference_valid = np.flatnonzero(
        np.isfinite(xy_all).all(axis=1)
        & np.isfinite(surface_matrix).all(axis=1)
    )
    known_valid = np.flatnonzero(
        np.isfinite(tvt_input)
        & np.isfinite(z_all)
        & np.isfinite(xy_all).all(axis=1)
    )
    eval_valid = np.flatnonzero(
        ~np.isfinite(tvt_input)
        & np.isfinite(truth_all)
        & np.isfinite(z_all)
        & np.isfinite(xy_all).all(axis=1)
    )
    if len(reference_valid) < 20 or len(known_valid) < 30 or len(eval_valid) < 30:
        continue

    reference_index = evenly_spaced(reference_valid, REFERENCE_POINTS_PER_WELL)
    anchor_index = int(known_valid[-1])
    eval_index = evenly_spaced(eval_valid, EVAL_POINTS_PER_WELL)

    reference_parts.append(
        pd.DataFrame(
            {
                "well": wid,
                "x": xy_all[reference_index, 0],
                "y": xy_all[reference_index, 1],
                "surface": surface_mean[reference_index],
            }
        )
    )
    well_data[wid] = {
        "anchor_xy": xy_all[anchor_index : anchor_index + 1],
        "anchor_z": float(z_all[anchor_index]),
        "anchor_tvt": float(tvt_input[anchor_index]),
        "eval_xy": xy_all[eval_index],
        "eval_z": z_all[eval_index],
        "eval_truth": truth_all[eval_index],
        "eval_row_idx": eval_index.astype(np.int32),
    }
    meta_rows.append(
        {
            "well": wid,
            "x": float(np.mean(xy_all[reference_index, 0])),
            "y": float(np.mean(xy_all[reference_index, 1])),
            "surface": float(np.mean(surface_mean[reference_index])),
            "n_eval": int(len(eval_valid)),
        }
    )

reference = pd.concat(reference_parts, ignore_index=True)
meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
print("reference points:", reference.shape, "wells:", len(meta))


# ---------------------------------------------------------------------------
# Two pre-registered split systems.
# ---------------------------------------------------------------------------
group_assignment, group_loads = balanced_group_folds(meta)
meta["group_fold"] = meta["well"].map(group_assignment).astype(int)

spatial_model = KMeans(n_clusters=N_FOLDS, random_state=RANDOM_STATE, n_init=20)
meta["spatial_fold"] = spatial_model.fit_predict(meta[["x", "y"]].to_numpy(float))

print("group fold loads:", group_loads.tolist())
print("spatial fold wells:", meta["spatial_fold"].value_counts().sort_index().to_dict())


def build_training_bundle(train_wells):
    train_wells = set(map(str, train_wells))
    points = reference[reference["well"].isin(train_wells)]
    well_meta = meta[meta["well"].isin(train_wells)][["well", "x", "y", "surface"]]
    return {
        "xy": points[["x", "y"]].to_numpy(float),
        "surface": points["surface"].to_numpy(float),
        "well_meta": well_meta.reset_index(drop=True),
    }


def predict_anchored(model, data):
    anchor_surface = float(
        np.asarray(model.predict(data["anchor_xy"]), dtype=float).reshape(-1)[0]
    )
    eval_surface = np.asarray(model.predict(data["eval_xy"]), dtype=float)
    if not np.isfinite(anchor_surface):
        raise RuntimeError("non-finite anchor surface")
    datum = data["anchor_tvt"] + data["anchor_z"] - anchor_surface
    return -data["eval_z"] + eval_surface + datum


split_records = []
failure_records = []
started = time.time()

for split_name, fold_column in (
    ("group", "group_fold"),
    ("spatial", "spatial_fold"),
):
    for fold in range(N_FOLDS):
        cache_path = CACHE_DIR / f"{split_name}_fold{fold}.pkl"
        if cache_path.exists():
            fold_rows = pd.read_pickle(cache_path)
            split_records.append(fold_rows)
            print("loaded cache:", cache_path, fold_rows.shape)
            continue

        valid_wells = meta.loc[meta[fold_column] == fold, "well"].astype(str).tolist()
        train_wells = meta.loc[meta[fold_column] != fold, "well"].astype(str).tolist()
        train_bundle = build_training_bundle(train_wells)

        print(
            f"\n{split_name} fold={fold} train_wells={len(train_wells)} "
            f"valid_wells={len(valid_wells)} train_points={len(train_bundle['xy'])}",
            flush=True,
        )

        raw_xy = train_bundle["xy"]
        raw_tree = cKDTree(raw_xy)
        models = {}
        for model_name, builder in MODEL_BUILDERS.items():
            model_started = time.time()
            print(" fitting", model_name, flush=True)
            models[model_name] = builder(train_bundle)
            print(
                " fitted", model_name, f"elapsed={time.time()-model_started:.1f}s",
                flush=True,
            )

        rows = []
        for wid in tqdm(valid_wells, desc=f"{split_name} fold {fold} wells"):
            data = well_data[wid]
            truth = data["eval_truth"]
            distance, _ = raw_tree.query(data["eval_xy"], k=1, workers=-1)
            frame = pd.DataFrame(
                {
                    "well": wid,
                    "split": split_name,
                    "fold": int(fold),
                    "row_idx": data["eval_row_idx"],
                    "truth": truth,
                    "nearest_train_distance": np.asarray(distance, dtype=float),
                }
            )
            for model_name, model in models.items():
                try:
                    frame[model_name] = predict_anchored(model, data)
                except Exception as exc:
                    frame[model_name] = np.nan
                    failure_records.append(
                        {
                            "well": wid,
                            "split": split_name,
                            "fold": int(fold),
                            "model": model_name,
                            "reason": repr(exc),
                        }
                    )
            rows.append(frame)

        fold_rows = pd.concat(rows, ignore_index=True)
        fold_rows.to_pickle(cache_path)
        split_records.append(fold_rows)
        print(
            f"saved {cache_path} rows={len(fold_rows)} "
            f"total_elapsed={(time.time()-started)/60:.1f} min",
            flush=True,
        )

predictions = pd.concat(split_records, ignore_index=True)
model_names = list(MODEL_BUILDERS)


# ---------------------------------------------------------------------------
# Honest summaries: pooled rows, folds, and nearest-distance quintiles.
# ---------------------------------------------------------------------------
summary_records = []
fold_records = []
for split_name, split_rows in predictions.groupby("split", sort=False):
    for model_name in model_names:
        valid = np.isfinite(split_rows[model_name]) & np.isfinite(split_rows["truth"])
        error = split_rows.loc[valid, model_name] - split_rows.loc[valid, "truth"]
        score = float(np.sqrt(np.mean(error**2))) if len(error) else np.nan
        summary_records.append(
            {
                "split": split_name,
                "model": model_name,
                "rmse": score,
                "rows": int(len(error)),
                "wells": int(split_rows.loc[valid, "well"].nunique()),
            }
        )
        for fold, fold_rows in split_rows.groupby("fold", sort=True):
            finite = np.isfinite(fold_rows[model_name]) & np.isfinite(fold_rows["truth"])
            fold_error = fold_rows.loc[finite, model_name] - fold_rows.loc[finite, "truth"]
            fold_records.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "fold": int(fold),
                    "rmse": float(np.sqrt(np.mean(fold_error**2))) if len(fold_error) else np.nan,
                    "rows": int(len(fold_error)),
                }
            )

summary = pd.DataFrame(summary_records).sort_values(["split", "rmse"])
fold_summary = pd.DataFrame(fold_records)

distance_records = []
for split_name, split_rows in predictions.groupby("split", sort=False):
    split_rows = split_rows.copy()
    split_rows["distance_bin"] = pd.qcut(
        split_rows["nearest_train_distance"].rank(method="first"),
        q=5,
        labels=["q1_near", "q2", "q3", "q4", "q5_far"],
    )
    for distance_bin, group in split_rows.groupby("distance_bin", observed=True):
        for model_name in model_names:
            valid = np.isfinite(group[model_name]) & np.isfinite(group["truth"])
            error = group.loc[valid, model_name] - group.loc[valid, "truth"]
            distance_records.append(
                {
                    "split": split_name,
                    "distance_bin": str(distance_bin),
                    "model": model_name,
                    "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                    "rows": int(len(error)),
                    "distance_median": float(group["nearest_train_distance"].median()),
                    "distance_p90": float(group["nearest_train_distance"].quantile(0.90)),
                }
            )
distance_summary = pd.DataFrame(distance_records)


verdict_records = []
for model_name in model_names:
    group_row = summary[(summary["split"] == "group") & (summary["model"] == model_name)]
    spatial_row = summary[(summary["split"] == "spatial") & (summary["model"] == model_name)]
    group_rmse = float(group_row["rmse"].iloc[0]) if len(group_row) else np.inf
    spatial_rmse = float(spatial_row["rmse"].iloc[0]) if len(spatial_row) else np.inf
    far_row = distance_summary[
        (distance_summary["split"] == "group")
        & (distance_summary["model"] == model_name)
        & (distance_summary["distance_bin"] == "q5_far")
    ]
    far_rmse = float(far_row["rmse"].iloc[0]) if len(far_row) else np.inf
    max_group_fold = float(
        fold_summary[
            (fold_summary["split"] == "group")
            & (fold_summary["model"] == model_name)
        ]["rmse"].max()
    )

    if group_rmse < 8.0 and spatial_rmse < 15.0 and far_rmse < 20.0 and max_group_fold < 12.0:
        verdict = "GOLD_LEVEL"
    elif group_rmse < 15.0 and spatial_rmse < 20.0:
        verdict = "GOLD_AMMO_ONLY"
    else:
        verdict = "STOP"
    verdict_records.append(
        {
            "model": model_name,
            "group_rmse": group_rmse,
            "spatial_rmse": spatial_rmse,
            "group_far_rmse": far_rmse,
            "max_group_fold_rmse": max_group_fold,
            "verdict": verdict,
        }
    )

verdict = pd.DataFrame(verdict_records).sort_values("group_rmse")
failure_df = pd.DataFrame(failure_records)

print("\nOVERALL")
display(summary)
print("\nFOLDS")
display(fold_summary)
print("\nDISTANCE BINS")
display(distance_summary)
print("\nVERDICT")
display(verdict)

prefix = OOF_DIR / "global_surface_reconstruction_killtest"
summary.to_csv(str(prefix) + "_summary.csv", index=False)
fold_summary.to_csv(str(prefix) + "_folds.csv", index=False)
distance_summary.to_csv(str(prefix) + "_distance.csv", index=False)
verdict.to_csv(str(prefix) + "_verdict.csv", index=False)
failure_df.to_csv(str(prefix) + "_failures.csv", index=False)
predictions.to_pickle(str(prefix) + "_sampled_predictions.pkl")
print("saved prefix:", prefix)
