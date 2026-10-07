"""P0-A one-fold gateway: future trajectory plus geology-state distillation.

Formation surfaces are privileged labels only. They are audited and used to
define low-frequency S-state supervision during training, but never enter a
deployable feature matrix. The held-out fold compares geometry-only,
GR/typewell-only, combined, and cross-fitted predicted-state variants.
"""

from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "trajectory_geology_state_gateway_v0"
CACHE_DIR = OUT_DIR / "well_features"
MODEL_DIR = OUT_DIR / "models"

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "gr_path_bank_oracle_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)

SURFACES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
VALID_FOLD = 1
N_CONTROL = 24
TRAJECTORY_POINTS = 32
FULL_GR_POINTS = 64
PREFIX_POINTS = 16
LOCAL_GR_OFFSETS = np.asarray(
    [-512, -256, -128, -64, -32, -16, 0, 16, 32, 64, 128, 256, 512],
    np.float32,
)
TYPEWELL_OFFSETS = np.arange(-128.0, 128.01, 8.0, dtype=np.float32)
STATE_FLAT_SLOPE = 0.005
RANDOM_STATE = 20260717


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _robust_location_scale(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return 0.0, 1.0
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values), 1.0))
    return center, scale


def _interp_finite(x, y, query, left=np.nan, right=np.nan):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    query = np.asarray(query, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, float)
    order = np.argsort(x[valid], kind="mergesort")
    xv = x[valid][order]
    yv = y[valid][order]
    unique, index = np.unique(xv, return_index=True)
    return np.interp(query, unique, yv[index], left=left, right=right)


def _nearest_distance(x, query):
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    query = np.asarray(query, float)
    if not len(x):
        return np.full(query.shape, np.inf)
    position = np.searchsorted(x, query)
    left = np.abs(query - x[np.clip(position - 1, 0, len(x) - 1)])
    right = np.abs(query - x[np.clip(position, 0, len(x) - 1)])
    return np.minimum(left, right)


def _add_matrix(parts, names, prefix, values):
    values = np.asarray(values, np.float32)
    if values.ndim == 1:
        values = values[:, None]
    parts.append(values)
    names.extend([f"{prefix}_{index:03d}" for index in range(values.shape[1])])


def _build_well(well):
    output = CACHE_DIR / f"{well}.npz"
    audit_output = CACHE_DIR / f"{well}_surface.csv"
    if output.exists() and audit_output.exists():
        audit_columns = pd.read_csv(audit_output, nrows=0).columns
        if "available" in audit_columns:
            with np.load(output, allow_pickle=False) as data:
                return {
                    "well": well,
                    "status": "cached",
                    "nodes": int(len(data["node_md"])),
                    "features": int(data["features"].shape[1]),
                }
    try:
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        )
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__typewell.csv"
        )
        required = {
            "MD", "X", "Y", "Z", "GR", "TVT", "TVT_input", *SURFACES
        }
        if not required.issubset(horizontal.columns):
            raise RuntimeError(
                f"missing columns {sorted(required-set(horizontal.columns))}"
            )
        if not {"TVT", "GR"}.issubset(typewell.columns):
            raise RuntimeError("typewell misses TVT/GR")

        numeric = horizontal[list(required)].apply(pd.to_numeric, errors="coerce")
        md = numeric["MD"].to_numpy(float)
        x = numeric["X"].to_numpy(float)
        y = numeric["Y"].to_numpy(float)
        z = numeric["Z"].to_numpy(float)
        gr = numeric["GR"].to_numpy(float)
        truth = numeric["TVT"].to_numpy(float)
        tvt_input = numeric["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        eval_rows = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        if len(known) < 30 or len(eval_rows) < 128:
            raise RuntimeError("too few prefix/eval rows")
        anchor = int(known[-1])
        anchor_tvt = float(tvt_input[anchor])
        anchor_s = anchor_tvt + float(z[anchor])
        s_true = truth + z

        # The audit is independent of the deployable features.
        surface_rows = []
        for surface in SURFACES:
            values = numeric[surface].to_numpy(float)
            valid = np.isfinite(s_true) & np.isfinite(values)
            if int(valid.sum()) < 2 or not np.isfinite(values[anchor]):
                surface_rows.append(
                    {
                        "well": well,
                        "surface": surface,
                        "available": False,
                        "rows": int(valid.sum()),
                        "residual_mean": np.nan,
                        "residual_std": np.nan,
                        "residual_p95_span": np.nan,
                        "anchored_delta_rmse": np.nan,
                    }
                )
                continue
            residual = s_true[valid] - values[valid]
            delta_error = (
                (s_true[valid] - s_true[anchor])
                - (values[valid] - values[anchor])
            )
            surface_rows.append(
                {
                    "well": well,
                    "surface": surface,
                    "available": True,
                    "rows": int(valid.sum()),
                    "residual_mean": float(np.mean(residual)),
                    "residual_std": float(np.std(residual)),
                    "residual_p95_span": float(
                        np.quantile(residual, 0.975) - np.quantile(residual, 0.025)
                    ),
                    "anchored_delta_rmse": float(
                        np.sqrt(np.mean(delta_error**2))
                    ),
                }
            )
        pd.DataFrame(surface_rows).to_csv(audit_output, index=False)

        node_md = np.linspace(md[anchor], md[eval_rows[-1]], N_CONTROL)
        node_rows = np.searchsorted(md, node_md)
        node_rows = np.clip(node_rows, anchor, eval_rows[-1]).astype(np.int32)
        node_md = md[node_rows]
        n = len(node_rows)
        node_z = z[node_rows]
        target_s_delta = s_true[node_rows] - anchor_s
        target_tvt_delta = truth[node_rows] - anchor_tvt
        target_s_delta[0] = 0.0
        target_tvt_delta[0] = 0.0
        target_s_slope = np.gradient(target_s_delta, node_md, edge_order=1)
        target_state = np.where(
            target_s_slope < -STATE_FLAT_SLOPE,
            0,
            np.where(target_s_slope > STATE_FLAT_SLOPE, 2, 1),
        ).astype(np.int8)

        dx = np.gradient(x, md, edge_order=1)
        dy = np.gradient(y, md, edge_order=1)
        dz = np.gradient(z, md, edge_order=1)
        ddx = np.gradient(dx, md, edge_order=1)
        ddy = np.gradient(dy, md, edge_order=1)
        ddz = np.gradient(dz, md, edge_order=1)
        azimuth = np.arctan2(dy, dx)
        dogleg = np.sqrt(ddx**2 + ddy**2 + ddz**2)

        span = max(md[eval_rows[-1]] - md[anchor], 1.0)
        geometry_local = np.column_stack(
            [
                (node_md - md[anchor]) / 1000.0,
                (node_md - md[anchor]) / span,
                (x[node_rows] - x[anchor]) / 1000.0,
                (y[node_rows] - y[anchor]) / 1000.0,
                (z[node_rows] - z[anchor]) / 1000.0,
                dx[node_rows],
                dy[node_rows],
                dz[node_rows],
                ddx[node_rows] * 1000.0,
                ddy[node_rows] * 1000.0,
                ddz[node_rows] * 1000.0,
                dogleg[node_rows] * 1000.0,
                np.sin(azimuth[node_rows]),
                np.cos(azimuth[node_rows]),
                np.full(n, x[anchor] / 100000.0),
                np.full(n, y[anchor] / 100000.0),
                np.full(n, z[anchor] / 10000.0),
                np.full(n, anchor_tvt / 10000.0),
                np.full(n, anchor_s / 10000.0),
                np.full(n, span / 1000.0),
            ]
        ).astype(np.float32)
        geometry_names = [
            "md_since_kft", "progress", "x_since_kft", "y_since_kft",
            "z_since_kft", "dxdmd", "dydmd", "dzdmd", "ddx_1k",
            "ddy_1k", "ddz_1k", "dogleg_1k", "azimuth_sin",
            "azimuth_cos", "anchor_x_100k", "anchor_y_100k",
            "anchor_z_10k", "anchor_tvt_10k", "anchor_s_10k",
            "eval_span_kft",
        ]
        geometry_parts = [geometry_local]

        trajectory_md = np.linspace(md[anchor], md[eval_rows[-1]], TRAJECTORY_POINTS)
        trajectory_values = {
            "traj_x_kft": (_interp_finite(md, x, trajectory_md) - x[anchor]) / 1000.0,
            "traj_y_kft": (_interp_finite(md, y, trajectory_md) - y[anchor]) / 1000.0,
            "traj_z_kft": (_interp_finite(md, z, trajectory_md) - z[anchor]) / 1000.0,
            "traj_dx": _interp_finite(md, dx, trajectory_md),
            "traj_dy": _interp_finite(md, dy, trajectory_md),
            "traj_dz": _interp_finite(md, dz, trajectory_md),
            "traj_dogleg": _interp_finite(md, dogleg, trajectory_md) * 1000.0,
        }
        for prefix, values in trajectory_values.items():
            _add_matrix(
                geometry_parts,
                geometry_names,
                prefix,
                np.broadcast_to(values, (n, len(values))),
            )

        prefix_md = np.linspace(md[0], md[anchor], PREFIX_POINTS)
        prefix_tvt = _interp_finite(md[: anchor + 1], tvt_input[: anchor + 1], prefix_md)
        prefix_z = _interp_finite(md[: anchor + 1], z[: anchor + 1], prefix_md)
        for prefix, values in (
            ("prefix_tvt_delta", prefix_tvt - anchor_tvt),
            ("prefix_z_delta", prefix_z - z[anchor]),
        ):
            _add_matrix(
                geometry_parts,
                geometry_names,
                prefix,
                np.broadcast_to(values, (n, len(values))),
            )
        geometry_features = np.column_stack(geometry_parts).astype(np.float32)

        # GR/typewell input is isolated so its value can be measured directly.
        gr_parts = []
        gr_names = []
        gr_center, gr_scale = _robust_location_scale(gr)
        gr_normalised = np.zeros(len(gr), np.float32)
        finite_gr = np.isfinite(gr)
        gr_normalised[finite_gr] = np.clip(
            (gr[finite_gr] - gr_center) / gr_scale, -8.0, 8.0
        )
        md_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
        support_radius = max(6.0, 2.5 * md_step)

        full_gr_md = np.linspace(md[anchor], md[eval_rows[-1]], FULL_GR_POINTS)
        full_gr = _interp_finite(
            md[finite_gr], gr_normalised[finite_gr], full_gr_md
        )
        full_gr_mask = (
            _nearest_distance(md[finite_gr], full_gr_md) <= support_radius
        ).astype(np.float32)
        for prefix, values in (
            ("full_gr", full_gr),
            ("full_gr_mask", full_gr_mask),
        ):
            _add_matrix(
                gr_parts,
                gr_names,
                prefix,
                np.broadcast_to(values, (n, len(values))),
            )

        local_query = node_md[:, None] + LOCAL_GR_OFFSETS[None, :]
        local_gr = _interp_finite(
            md[finite_gr], gr_normalised[finite_gr], local_query
        )
        local_mask = (
            _nearest_distance(md[finite_gr], local_query) <= support_radius
        ).astype(np.float32)
        _add_matrix(gr_parts, gr_names, "local_gr", local_gr)
        _add_matrix(gr_parts, gr_names, "local_gr_mask", local_mask)

        tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
        tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(float)
        tw_gr = tw["GR"].to_numpy(float)
        tw_center, tw_scale = _robust_location_scale(tw_gr)
        tw_query = anchor_tvt + TYPEWELL_OFFSETS
        tw_values = _interp_finite(tw_tvt, tw_gr, tw_query)
        tw_values = np.clip((tw_values - tw_center) / tw_scale, -8.0, 8.0)
        tw_mask = ((tw_query >= tw_tvt.min()) & (tw_query <= tw_tvt.max())).astype(np.float32)
        for prefix, values in (("typewell_gr", tw_values), ("typewell_mask", tw_mask)):
            _add_matrix(
                gr_parts,
                gr_names,
                prefix,
                np.broadcast_to(values, (n, len(values))),
            )

        domain = np.column_stack(
            [
                np.full(n, float(np.mean(~finite_gr))),
                np.full(n, float(np.nanmedian(np.diff(tw_tvt)))),
                np.full(n, float(tw_tvt.min()) / 10000.0),
                np.full(n, float(tw_tvt.max()) / 10000.0),
                np.full(n, float(np.mean(tw_mask))),
                np.full(n, gr_center / 100.0),
                np.full(n, gr_scale / 100.0),
            ]
        ).astype(np.float32)
        gr_parts.append(domain)
        gr_names.extend(
            [
                "gr_missing_fraction", "typewell_step", "typewell_min_10k",
                "typewell_max_10k", "typewell_valid_fraction",
                "gr_center_100", "gr_scale_100",
            ]
        )
        gr_features = np.column_stack(gr_parts).astype(np.float32)
        features = np.column_stack([geometry_features, gr_features]).astype(np.float32)
        feature_names = np.asarray([*geometry_names, *gr_names], str)

        np.savez_compressed(
            output,
            features=features,
            feature_names=feature_names,
            geometry_width=np.asarray([geometry_features.shape[1]], np.int32),
            node_rows=node_rows,
            node_md=node_md.astype(np.float32),
            node_z=node_z.astype(np.float32),
            target_s_delta=target_s_delta.astype(np.float32),
            target_tvt_delta=target_tvt_delta.astype(np.float32),
            target_s_slope=target_s_slope.astype(np.float32),
            target_state=target_state,
            eval_rows=eval_rows.astype(np.int32),
            eval_md=md[eval_rows].astype(np.float32),
            eval_z=z[eval_rows].astype(np.float32),
            eval_truth=truth[eval_rows].astype(np.float32),
            anchor_s=np.asarray([anchor_s], np.float32),
            anchor_tvt=np.asarray([anchor_tvt], np.float32),
        )
        return {
            "well": well,
            "status": "built",
            "nodes": n,
            "features": features.shape[1],
        }
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}


def _load_rows(wells):
    x_parts, y_parts, state_parts, owner_parts = [], [], [], []
    lengths = {}
    for well in wells:
        with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
            x = data["features"].astype(np.float32)
            x_parts.append(x)
            y_parts.append(data["target_s_delta"].astype(np.float32))
            state_parts.append(data["target_state"].astype(np.int32))
            owner_parts.append(np.full(len(x), well, object))
            lengths[well] = len(x)
    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(state_parts),
        np.concatenate(owner_parts),
        lengths,
    )


def _balanced_accuracy(truth, prediction):
    scores = []
    for label in (0, 1, 2):
        mask = np.asarray(truth) == label
        if mask.any():
            scores.append(float(np.mean(np.asarray(prediction)[mask] == label)))
    return float(np.mean(scores))


def _fit_classifier(train_x, train_y, valid_x, valid_y, label):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1600,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=10.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    joblib.dump(model, MODEL_DIR / f"{label}.joblib", compress=3)
    return model


def _fit_regressor(train_x, train_y, valid_x, valid_y, label):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=3200,
        learning_rate=0.025,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=15.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        callbacks=[early_stopping(150, verbose=False), log_evaluation(0)],
    )
    joblib.dump(model, MODEL_DIR / f"{label}.joblib", compress=3)
    return model


def _reconstruct(valid_wells, node_predictions):
    frames = []
    offset = 0
    for well in valid_wells:
        with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
            n = len(data["node_md"])
            prediction = np.asarray(node_predictions[offset : offset + n], float).copy()
            offset += n
            prediction[0] = 0.0
            eval_s_delta = np.interp(
                data["eval_md"].astype(float),
                data["node_md"].astype(float),
                prediction,
            )
            eval_tvt = (
                float(data["anchor_s"][0])
                + eval_s_delta
                - data["eval_z"].astype(float)
            )
            frames.append(
                pd.DataFrame(
                    {
                        "well": well,
                        "row_idx": data["eval_rows"].astype(np.int32),
                        "truth": data["eval_truth"].astype(float),
                        "prediction": eval_tvt,
                    }
                )
            )
    if offset != len(node_predictions):
        raise RuntimeError("node prediction length mismatch")
    return pd.concat(frames, ignore_index=True)


def _rmse(frame):
    return float(np.sqrt(np.mean((frame["truth"] - frame["prediction"]) ** 2)))


for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

fold_path = _first_existing(FOLD_CANDIDATES)
if fold_path is None:
    raise FileNotFoundError("fold metadata not found")
fold = pd.read_csv(fold_path, dtype={"well": str})
fold = fold.groupby("well", as_index=False)["fold"].first()
fold["well"] = fold["well"].astype(str)
fold["fold"] = fold["fold"].astype(int)
wells = fold["well"].tolist()
train_wells = fold.loc[fold["fold"] != VALID_FOLD, "well"].tolist()
valid_wells = fold.loc[fold["fold"] == VALID_FOLD, "well"].tolist()

print(
    {
        "experiment": "trajectory_geology_state_gateway_v0",
        "fold_source": str(fold_path),
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "surfaces": SURFACES,
        "state_flat_slope": STATE_FLAT_SLOPE,
    }
)
started = time.time()
build_report = pd.DataFrame(
    [_build_well(well) for well in tqdm(wells, desc="trajectory/state features")]
)
build_report.to_csv(OUT_DIR / "feature_build_report.csv", index=False)
failed = build_report[build_report["status"] == "failed"]
if len(failed):
    raise RuntimeError(f"failed to build {len(failed)} wells")

surface_frames = []
for well in wells:
    frame = pd.read_csv(CACHE_DIR / f"{well}_surface.csv")
    frame["well"] = str(well)
    if "available" not in frame.columns:
        frame["available"] = frame["rows"].fillna(0).astype(int) >= 2
    surface_frames.append(frame)
surface_audit = pd.concat(surface_frames, ignore_index=True)
surface_audit.to_csv(OUT_DIR / "surface_relation_bywell.csv", index=False)
surface_summary = (
    surface_audit.groupby("surface", as_index=False)
    .agg(
        wells=("well", "nunique"),
        available_wells=("available", "sum"),
        median_residual_std=("residual_std", "median"),
        p95_residual_std=("residual_std", lambda x: float(np.nanquantile(x, 0.95))),
        median_anchored_rmse=("anchored_delta_rmse", "median"),
        p95_anchored_rmse=("anchored_delta_rmse", lambda x: float(np.nanquantile(x, 0.95))),
    )
    .sort_values("median_anchored_rmse")
)
surface_summary.to_csv(OUT_DIR / "surface_relation_summary.csv", index=False)

train_x, train_y, train_state, train_owner, _ = _load_rows(train_wells)
valid_x, valid_y, valid_state, valid_owner, _ = _load_rows(valid_wells)
with np.load(CACHE_DIR / f"{wells[0]}.npz", allow_pickle=False) as first:
    feature_names = first["feature_names"].astype(str)
    geometry_width = int(first["geometry_width"][0])
geometry_indices = np.arange(geometry_width)
gr_indices = np.arange(geometry_width, len(feature_names))
all_indices = np.arange(len(feature_names))

# Cross-fitted state probabilities for the outer-train wells.
train_state_probability = np.zeros((len(train_x), 3), np.float32)
state_rows = []
for held_fold in sorted(fold.loc[fold["fold"] != VALID_FOLD, "fold"].unique()):
    held_wells = set(fold.loc[fold["fold"] == held_fold, "well"])
    held_mask = np.asarray([well in held_wells for well in train_owner])
    fit_mask = ~held_mask
    classifier = _fit_classifier(
        train_x[fit_mask][:, geometry_indices],
        train_state[fit_mask],
        train_x[held_mask][:, geometry_indices],
        train_state[held_mask],
        f"state_inner_hold{held_fold}",
    )
    probability = classifier.predict_proba(
        train_x[held_mask][:, geometry_indices]
    ).astype(np.float32)
    train_state_probability[held_mask] = probability
    prediction = probability.argmax(axis=1)
    state_rows.append(
        {
            "split": f"inner_fold_{held_fold}",
            "rows": int(held_mask.sum()),
            "accuracy": float(np.mean(prediction == train_state[held_mask])),
            "balanced_accuracy": _balanced_accuracy(
                train_state[held_mask], prediction
            ),
        }
    )
    del classifier
    gc.collect()

state_model = _fit_classifier(
    train_x[:, geometry_indices],
    train_state,
    valid_x[:, geometry_indices],
    valid_state,
    "state_outer_full",
)
valid_state_probability = state_model.predict_proba(
    valid_x[:, geometry_indices]
).astype(np.float32)
valid_state_prediction = valid_state_probability.argmax(axis=1)
state_rows.append(
    {
        "split": "outer_fold_1",
        "rows": len(valid_state),
        "accuracy": float(np.mean(valid_state_prediction == valid_state)),
        "balanced_accuracy": _balanced_accuracy(
            valid_state, valid_state_prediction
        ),
    }
)
state_report = pd.DataFrame(state_rows)
state_report.to_csv(OUT_DIR / "state_classification.csv", index=False)

variant_specs = {
    "geometry": (train_x[:, geometry_indices], valid_x[:, geometry_indices]),
    "gr_typewell": (train_x[:, gr_indices], valid_x[:, gr_indices]),
    "combined": (train_x[:, all_indices], valid_x[:, all_indices]),
    "geometry_state": (
        np.column_stack([train_x[:, geometry_indices], train_state_probability]),
        np.column_stack([valid_x[:, geometry_indices], valid_state_probability]),
    ),
    "combined_state": (
        np.column_stack([train_x, train_state_probability]),
        np.column_stack([valid_x, valid_state_probability]),
    ),
}

model_rows = []
prediction_frames = {}
for variant, (variant_train_x, variant_valid_x) in variant_specs.items():
    print("train", variant, variant_train_x.shape, flush=True)
    model = _fit_regressor(
        variant_train_x,
        train_y,
        variant_valid_x,
        valid_y,
        variant,
    )
    node_prediction = model.predict(variant_valid_x)
    frame = _reconstruct(valid_wells, node_prediction)
    frame.to_pickle(OUT_DIR / f"{variant}_fold1_predictions.pkl")
    prediction_frames[variant] = frame
    model_rows.append(
        {
            "variant": variant,
            "features": variant_train_x.shape[1],
            "rmse": _rmse(frame),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        }
    )
    del model
    gc.collect()
model_report = pd.DataFrame(model_rows).sort_values("rmse").reset_index(drop=True)

reference_rows = []
pool_path = _first_existing(POOL_CANDIDATES)
if pool_path is not None:
    header = pd.read_csv(pool_path, nrows=0).columns.tolist()
    references = [name for name in ("strong", "strong_bias", "cp12") if name in header]
    usecols = [
        name for name in ("id", "well", "row_idx", "tvt_true", *references)
        if name in header
    ]
    pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": str, "well": str})
    pool["id"] = pool["id"].astype(str)
    if "well" not in pool.columns:
        pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
    if "row_idx" not in pool.columns:
        pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    pool["well"] = pool["well"].astype(str)
    pool = pool[pool["well"].isin(valid_wells)]
    for reference in references:
        reference_rows.append(
            {
                "model": reference,
                "rmse": float(
                    np.sqrt(np.mean((pool["tvt_true"] - pool[reference]) ** 2))
                ),
            }
        )
reference_report = pd.DataFrame(reference_rows).sort_values("rmse")

outer_state_balanced_accuracy = float(
    state_report.loc[
        state_report["split"] == "outer_fold_1", "balanced_accuracy"
    ].iloc[0]
)
score_map = model_report.set_index("variant")["rmse"].to_dict()
best_variant = str(model_report.iloc[0]["variant"])
best_rmse = float(model_report.iloc[0]["rmse"])
geometry_state_gain = score_map["geometry"] - score_map["geometry_state"]
combined_state_gain = score_map["combined"] - score_map["combined_state"]
state_signal_pass = bool(outer_state_balanced_accuracy >= 0.55)
best_trajectory_rmse = min(
    score_map[name]
    for name in ("geometry", "geometry_state", "combined", "combined_state")
)
trajectory_pass = bool(best_trajectory_rmse < 8.0)
distillation_pass = bool(
    state_signal_pass and max(geometry_state_gain, combined_state_gain) >= 0.05
)
strict_pass = bool(trajectory_pass)
verdict = {
    "experiment": "trajectory_geology_state_gateway_v0",
    "fold_source": str(fold_path),
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "features": len(feature_names),
    "geometry_features": len(geometry_indices),
    "gr_typewell_features": len(gr_indices),
    "best_variant": best_variant,
    "best_rmse": best_rmse,
    "best_trajectory_rmse": best_trajectory_rmse,
    "outer_state_balanced_accuracy": outer_state_balanced_accuracy,
    "geometry_state_gain": geometry_state_gain,
    "combined_state_gain": combined_state_gain,
    "state_signal_pass": state_signal_pass,
    "trajectory_pass": trajectory_pass,
    "distillation_pass": distillation_pass,
    "runtime_minutes": (time.time() - started) / 60.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> add multi-seed/five-fold trajectory-state model"
        if strict_pass
        else "FAIL -> stop P0-A trajectory/state route"
    ),
}

pd.DataFrame({"feature": feature_names}).to_csv(
    OUT_DIR / "feature_names.csv", index=False
)
model_report.to_csv(OUT_DIR / "model_scores.csv", index=False)
reference_report.to_csv(OUT_DIR / "historical_references.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)

print("\nSURFACE RELATIONSHIP")
print(surface_summary.to_string(index=False))
print("\nSTATE CLASSIFICATION")
print(state_report.to_string(index=False))
print("\nMODEL SCORES")
print(model_report.to_string(index=False))
print("\nHISTORICAL REFERENCES")
print(reference_report.to_string(index=False) if len(reference_report) else "<none>")
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
