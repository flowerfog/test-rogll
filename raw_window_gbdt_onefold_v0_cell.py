"""One-fold MVP for physical-grid raw-window GBDT.

This model uses only test-time legal single-well inputs. It predicts nodes at
approximately 16-ft MD spacing and interpolates them back to every official
eval row. Three target representations are compared, followed by no-future-GR
and no-typewell ablations for the winning target.
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
OUT_DIR = OOF_ROOT / "raw_window_gbdt_onefold_v0"
CACHE_DIR = OUT_DIR / "well_features"
MODEL_DIR = OUT_DIR / "models"
VALID_FOLD = 1
RANDOM_STATE = 20260717

NODE_SPACING_FT = 16.0
HORIZONTAL_OFFSETS_FT = np.asarray(
    [
        -2048, -1024, -512, -256, -128, -64, -32, -16, -8,
        0,
        8, 16, 32, 64, 128, 256, 512, 1024, 2048,
    ],
    np.float32,
)
TYPEWELL_OFFSETS_FT = np.arange(-128.0, 128.01, 4.0, dtype=np.float32)
TRAJECTORY_POINTS = 32
PREFIX_TVT_POINTS = 16

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "gr_path_bank_oracle_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)

LGB_PARAMS = dict(
    objective="regression",
    boosting_type="gbdt",
    n_estimators=3200,
    learning_rate=0.025,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.20,
    reg_lambda=15.0,
    max_bin=255,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=-1,
)


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _robust_location_scale(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return 0.0, 1.0
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values), 1.0))
    return center, scale


def _interp_finite(x, y, query, left=np.nan, right=np.nan):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    order = np.argsort(x[valid], kind="mergesort")
    xv = x[valid][order]
    yv = y[valid][order]
    unique, unique_index = np.unique(xv, return_index=True)
    yv = yv[unique_index]
    return np.interp(query, unique, yv, left=left, right=right)


def _nearest_distance(x, query):
    x = np.sort(np.asarray(x, np.float64)[np.isfinite(x)])
    query = np.asarray(query, np.float64)
    if not len(x):
        return np.full(query.shape, np.inf)
    position = np.searchsorted(x, query)
    left = np.abs(query - x[np.clip(position - 1, 0, len(x) - 1)])
    right = np.abs(query - x[np.clip(position, 0, len(x) - 1)])
    return np.minimum(left, right)


def _physical_nodes(md, eval_rows, spacing):
    eval_rows = np.asarray(eval_rows, np.int32)
    target_md = np.arange(md[eval_rows[0]], md[eval_rows[-1]] + spacing, spacing)
    positions = np.searchsorted(md[eval_rows], target_md)
    positions = np.clip(positions, 0, len(eval_rows) - 1)
    nodes = np.unique(eval_rows[positions])
    nodes = np.unique(np.r_[eval_rows[0], nodes, eval_rows[-1]]).astype(np.int32)
    return nodes


def _add_matrix(parts, names, prefix, matrix):
    matrix = np.asarray(matrix, np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    parts.append(matrix)
    names.extend([f"{prefix}_{index:03d}" for index in range(matrix.shape[1])])


def _build_well(well):
    output = CACHE_DIR / f"{well}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as data:
            return {
                "well": well,
                "status": "cached",
                "nodes": int(len(data["node_rows"])),
                "features": int(data["features"].shape[1]),
            }
    try:
        horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        required = {"MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"}
        if not required.issubset(horizontal.columns):
            raise RuntimeError(f"missing horizontal columns: {required-set(horizontal.columns)}")
        if not {"TVT", "GR"}.issubset(typewell.columns):
            raise RuntimeError("typewell misses TVT/GR")

        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
        x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(np.float64)
        z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
        gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
        truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
        known = np.flatnonzero(np.isfinite(tvt_input))
        eval_rows = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        if len(known) < 30 or len(eval_rows) < 64:
            raise RuntimeError("too few prefix/eval rows")
        anchor = int(known[-1])
        nodes = _physical_nodes(md, eval_rows, NODE_SPACING_FT)
        node_md = md[nodes]
        n = len(nodes)
        anchor_tvt = float(tvt_input[anchor])
        anchor_s = anchor_tvt + float(z[anchor])

        parts = []
        names = []

        # Node-centred horizontal GR windows in physical MD units.
        queries = node_md[:, None] + HORIZONTAL_OFFSETS_FT[None, :]
        h_raw = _interp_finite(md, gr, queries, left=np.nan, right=np.nan)
        finite_gr_md = md[np.isfinite(gr)]
        md_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
        support_radius = max(8.0, 2.5 * md_step)
        h_mask = (_nearest_distance(finite_gr_md, queries) <= support_radius).astype(np.float32)
        gr_center, gr_scale = _robust_location_scale(gr)
        h_norm = np.clip((h_raw - gr_center) / gr_scale, -8.0, 8.0)
        h_slope = (
            _interp_finite(md, gr, queries + 4.0, left=np.nan, right=np.nan)
            - _interp_finite(md, gr, queries - 4.0, left=np.nan, right=np.nan)
        ) / 8.0
        _add_matrix(parts, names, "hgr_raw", h_raw)
        _add_matrix(parts, names, "hgr_norm", h_norm)
        _add_matrix(parts, names, "hgr_slope", h_slope)
        _add_matrix(parts, names, "hgr_mask", h_mask)

        # Heel-centred typewell TVT grid. It is identical across nodes by design.
        tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
        tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(np.float64)
        tw_gr = tw["GR"].to_numpy(np.float64)
        tw_query = anchor_tvt + TYPEWELL_OFFSETS_FT
        tw_raw_one = _interp_finite(tw_tvt, tw_gr, tw_query, left=np.nan, right=np.nan)
        tw_mask_one = ((tw_query >= tw_tvt.min()) & (tw_query <= tw_tvt.max())).astype(np.float32)
        tw_center, tw_scale = _robust_location_scale(tw_gr)
        tw_norm_one = np.clip((tw_raw_one - tw_center) / tw_scale, -8.0, 8.0)
        tw_slope_one = (
            _interp_finite(tw_tvt, tw_gr, tw_query + 2.0, left=np.nan, right=np.nan)
            - _interp_finite(tw_tvt, tw_gr, tw_query - 2.0, left=np.nan, right=np.nan)
        ) / 4.0
        for prefix, values in (
            ("tw_raw", tw_raw_one),
            ("tw_norm", tw_norm_one),
            ("tw_slope", tw_slope_one),
            ("tw_mask", tw_mask_one),
        ):
            _add_matrix(parts, names, prefix, np.broadcast_to(values, (n, len(values))))

        # Fixed representation of the complete future trajectory.
        trajectory_md = np.linspace(md[anchor], md[-1], TRAJECTORY_POINTS)
        trajectory_x = _interp_finite(md, x, trajectory_md) - x[anchor]
        trajectory_y = _interp_finite(md, y, trajectory_md) - y[anchor]
        trajectory_z = _interp_finite(md, z, trajectory_md) - z[anchor]
        trajectory_dz = np.gradient(trajectory_z, trajectory_md, edge_order=1)
        trajectory_gr = _interp_finite(md, gr, trajectory_md, left=np.nan, right=np.nan)
        trajectory_gr = np.clip((trajectory_gr - gr_center) / gr_scale, -8.0, 8.0)
        trajectory_gr_mask = (
            _nearest_distance(finite_gr_md, trajectory_md) <= support_radius
        ).astype(np.float32)
        for prefix, values in (
            ("trajectory_x_kft", trajectory_x / 1000.0),
            ("trajectory_y_kft", trajectory_y / 1000.0),
            ("trajectory_z_kft", trajectory_z / 1000.0),
            ("trajectory_dz", trajectory_dz),
            ("trajectory_gr", trajectory_gr),
            ("trajectory_gr_mask", trajectory_gr_mask),
        ):
            _add_matrix(parts, names, prefix, np.broadcast_to(values, (n, len(values))))

        # Full known-TV T prefix compressed only by physical progress resampling.
        prefix_rows = np.arange(0, anchor + 1)
        prefix_md_grid = np.linspace(md[prefix_rows[0]], md[anchor], PREFIX_TVT_POINTS)
        prefix_tvt = _interp_finite(md[prefix_rows], tvt_input[prefix_rows], prefix_md_grid)
        prefix_tvt = prefix_tvt - anchor_tvt
        _add_matrix(
            parts,
            names,
            "prefix_tvt_delta",
            np.broadcast_to(prefix_tvt, (n, len(prefix_tvt))),
        )

        # Node-local geometry and acquisition/domain descriptors.
        dx = np.gradient(x, md, edge_order=1)
        dy = np.gradient(y, md, edge_order=1)
        dz = np.gradient(z, md, edge_order=1)
        local = np.column_stack(
            [
                (node_md - md[anchor]) / 1000.0,
                (node_md - md[anchor]) / max(md[-1] - md[anchor], 1.0),
                (x[nodes] - x[anchor]) / 1000.0,
                (y[nodes] - y[anchor]) / 1000.0,
                (z[nodes] - z[anchor]) / 1000.0,
                dx[nodes],
                dy[nodes],
                dz[nodes],
                np.full(n, anchor_tvt / 10000.0),
                np.full(n, anchor_s / 10000.0),
                np.full(n, (md[-1] - md[anchor]) / 1000.0),
                np.full(n, float(np.mean(~np.isfinite(gr)))),
                np.full(n, float(np.nanmedian(np.diff(tw_tvt)))),
                np.full(n, float(tw_tvt.min()) / 10000.0),
                np.full(n, float(tw_tvt.max()) / 10000.0),
                np.full(n, float(np.mean(np.isfinite(tw_raw_one)))),
            ]
        )
        local_names = [
            "md_since_kft", "eval_progress", "x_since_kft", "y_since_kft",
            "z_since_kft", "dxdmd", "dydmd", "dzdmd", "anchor_tvt_10k",
            "anchor_s_10k", "eval_span_kft", "gr_missing_fraction",
            "typewell_step", "typewell_min_10k", "typewell_max_10k",
            "typewell_heel_grid_valid_fraction",
        ]
        parts.append(local.astype(np.float32))
        names.extend(local_names)

        features = np.column_stack(parts).astype(np.float32)
        if len(names) != features.shape[1]:
            raise RuntimeError("feature name/width mismatch")
        np.savez_compressed(
            output,
            features=features,
            feature_names=np.asarray(names, str),
            node_rows=nodes.astype(np.int32),
            node_md=node_md.astype(np.float32),
            truth_tvt=truth[nodes].astype(np.float32),
            truth_delta=(truth[nodes] - anchor_tvt).astype(np.float32),
            truth_s_delta=(truth[nodes] + z[nodes] - anchor_s).astype(np.float32),
            eval_rows=eval_rows.astype(np.int32),
            eval_md=md[eval_rows].astype(np.float32),
            eval_truth=truth[eval_rows].astype(np.float32),
            eval_z=z[eval_rows].astype(np.float32),
            anchor_row=np.asarray([anchor], np.int32),
            anchor_tvt=np.asarray([anchor_tvt], np.float32),
            anchor_s=np.asarray([anchor_s], np.float32),
        )
        return {"well": well, "status": "built", "nodes": n, "features": len(names)}
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}


def _load_matrix(wells, feature_indices=None, target="truth_delta"):
    x_parts, y_parts, owner_parts = [], [], []
    for well in wells:
        with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
            x = data["features"]
            if feature_indices is not None:
                x = x[:, feature_indices]
            x_parts.append(x.astype(np.float32))
            y_parts.append(data[target].astype(np.float32))
            owner_parts.append(np.full(len(x), well, object))
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(owner_parts)


def _fit(train_wells, valid_wells, target, feature_indices, label):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    train_x, train_y, _ = _load_matrix(train_wells, feature_indices, target)
    valid_x, valid_y, _ = _load_matrix(valid_wells, feature_indices, target)
    model = LGBMRegressor(**LGB_PARAMS)
    try:
        model.set_params(device_type="gpu")
        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            callbacks=[early_stopping(150, verbose=False), log_evaluation(200)],
        )
    except Exception as exc:
        print("GPU LightGBM fallback:", repr(exc), flush=True)
        model = LGBMRegressor(**LGB_PARAMS)
        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            callbacks=[early_stopping(150, verbose=False), log_evaluation(200)],
        )
    joblib.dump(model, MODEL_DIR / f"{label}.joblib", compress=3)
    del train_x, train_y, valid_x, valid_y
    gc.collect()
    return model


def _reconstruct(model, valid_wells, feature_indices, target):
    rows = []
    for well in valid_wells:
        with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
            x = data["features"][:, feature_indices]
            node_prediction = model.predict(x).astype(np.float64)
            anchor_tvt = float(data["anchor_tvt"][0])
            anchor_s = float(data["anchor_s"][0])
            node_rows = data["node_rows"].astype(int)
            node_md = data["node_md"].astype(float)
            eval_rows = data["eval_rows"].astype(int)
            eval_md = data["eval_md"].astype(float)
            eval_z = data["eval_z"].astype(float)
            truth = data["eval_truth"].astype(float)
            if target == "truth_tvt":
                node_tvt = node_prediction
            elif target == "truth_delta":
                node_tvt = anchor_tvt + node_prediction
            elif target == "truth_s_delta":
                node_z = np.interp(node_md, eval_md, eval_z)
                node_tvt = anchor_s + node_prediction - node_z
            else:
                raise ValueError(target)
            prediction = np.interp(eval_md, node_md, node_tvt)
            rows.append(
                pd.DataFrame(
                    {
                        "well": well,
                        "row_idx": eval_rows,
                        "truth": truth,
                        "prediction": prediction,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _score(frame):
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

print(
    {
        "experiment": "raw_window_gbdt_onefold_v0",
        "fold_source": str(fold_path),
        "valid_fold": VALID_FOLD,
        "wells": len(wells),
        "node_spacing_ft": NODE_SPACING_FT,
        "cache": str(CACHE_DIR),
    }
)
started = time.time()
build_rows = [_build_well(well) for well in tqdm(wells, desc="raw-window features")]
build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "feature_build_report.csv", index=False)
failed = build_report[build_report["status"] == "failed"]
if len(failed):
    raise RuntimeError(f"failed to build {len(failed)} wells")

train_wells = fold.loc[fold["fold"] != VALID_FOLD, "well"].tolist()
valid_wells = fold.loc[fold["fold"] == VALID_FOLD, "well"].tolist()
with np.load(CACHE_DIR / f"{wells[0]}.npz", allow_pickle=False) as first:
    feature_names = first["feature_names"].astype(str)
all_indices = np.arange(len(feature_names))
future_indices = np.asarray(
    [
        index
        for index, name in enumerate(feature_names)
        if name.startswith(("hgr_", "trajectory_gr"))
        and not any(
            name.startswith(f"hgr_{kind}_{slot:03d}")
            for kind in ("raw", "norm", "slope", "mask")
            for slot, offset in enumerate(HORIZONTAL_OFFSETS_FT)
            if offset <= 0
        )
    ],
    int,
)
typewell_indices = np.asarray(
    [index for index, name in enumerate(feature_names) if name.startswith("tw_")], int
)
no_future_indices = np.setdiff1d(all_indices, future_indices)
no_typewell_indices = np.setdiff1d(all_indices, typewell_indices)

target_rows = []
target_models = {}
for target in ("truth_tvt", "truth_delta", "truth_s_delta"):
    label = target.removeprefix("truth_")
    model = _fit(train_wells, valid_wells, target, all_indices, f"full_{label}")
    prediction = _reconstruct(model, valid_wells, all_indices, target)
    score = _score(prediction)
    prediction.to_pickle(OUT_DIR / f"oof_fold1_full_{label}.pkl")
    target_rows.append(
        {
            "variant": f"full_{label}",
            "target": target,
            "features": len(all_indices),
            "rmse": score,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        }
    )
    target_models[target] = model

target_report = pd.DataFrame(target_rows).sort_values("rmse").reset_index(drop=True)
best_target = str(target_report.iloc[0]["target"])
best_full_rmse = float(target_report.iloc[0]["rmse"])

ablation_rows = []
for label, indices in (
    ("no_future_gr", no_future_indices),
    ("no_typewell", no_typewell_indices),
):
    model = _fit(train_wells, valid_wells, best_target, indices, label)
    prediction = _reconstruct(model, valid_wells, indices, best_target)
    score = _score(prediction)
    prediction.to_pickle(OUT_DIR / f"oof_fold1_{label}.pkl")
    ablation_rows.append(
        {
            "variant": label,
            "target": best_target,
            "features": len(indices),
            "rmse": score,
            "degradation_vs_full": score - best_full_rmse,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        }
    )
ablation_report = pd.DataFrame(ablation_rows)

# Optional historical references on exactly the same held-out rows.
reference_rows = []
pool_path = _first_existing(POOL_CANDIDATES)
if pool_path is not None:
    header = pd.read_csv(pool_path, nrows=0).columns.tolist()
    references = [name for name in ("strong", "strong_bias", "cp12") if name in header]
    if references:
        usecols = [name for name in ("id", "well", "row_idx", "tvt_true", *references) if name in header]
        pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": str, "well": str})
        if "well" not in pool:
            split = pool["id"].str.rsplit("_", n=1, expand=True)
            pool["well"] = split[0]
        pool = pool[pool["well"].isin(valid_wells)]
        for name in references:
            reference_rows.append(
                {
                    "model": name,
                    "rmse": float(np.sqrt(np.mean((pool["tvt_true"] - pool[name]) ** 2))),
                }
            )
reference_report = pd.DataFrame(reference_rows)

future_degradation = float(
    ablation_report.loc[ablation_report["variant"] == "no_future_gr", "degradation_vs_full"].iloc[0]
)
typewell_degradation = float(
    ablation_report.loc[ablation_report["variant"] == "no_typewell", "degradation_vs_full"].iloc[0]
)
strict_pass = bool(
    best_full_rmse < 8.0
    and future_degradation >= 0.05
    and typewell_degradation >= 0.05
)
verdict = {
    "experiment": "raw_window_gbdt_onefold_v0",
    "fold_source": str(fold_path),
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "features": len(feature_names),
    "nodes": int(build_report["nodes"].sum()),
    "best_target": best_target,
    "best_full_rmse": best_full_rmse,
    "future_gr_degradation": future_degradation,
    "typewell_degradation": typewell_degradation,
    "runtime_hours": (time.time() - started) / 3600.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> run remaining folds and multi-seed models"
        if strict_pass
        else "FAIL -> stop raw-window GBDT or inspect the failed causal ablation"
    ),
}

pd.DataFrame({"feature": feature_names}).to_csv(OUT_DIR / "feature_names.csv", index=False)
target_report.to_csv(OUT_DIR / "target_report.csv", index=False)
ablation_report.to_csv(OUT_DIR / "ablation_report.csv", index=False)
reference_report.to_csv(OUT_DIR / "historical_references.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nTARGETS")
print(target_report.to_string(index=False))
print("\nABLATIONS")
print(ablation_report.to_string(index=False))
print("\nHISTORICAL REFERENCES")
print(reference_report.to_string(index=False) if len(reference_report) else "<none>")
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
