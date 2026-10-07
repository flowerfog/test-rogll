"""Strict five-fold shape-basket and S-interpolation ceiling audit.

This is G0-G2 of the shape-first, offset-second plan:

1. Build five outer-fold-pure learned shape candidates from the existing
   trajectory_geology_state_gateway_v0 control-node cache.
2. Reconstruct each learned shape in both S=TVT+Z space and TVT space, using
   the S reconstruction in the candidate basket.
3. Add carry-S, the exact frozen coherent600 Neighbor candidate, and strong.
4. Measure standalone quality, fold stability, error correlation, and a
   whole-well oracle.  The oracle selects one complete path per well; it never
   selects rows independently.

An optional CP12 column is exported as a ninth reference candidate, but it is
not part of the pre-registered primary B8 basket.

Expected runtime: minutes, not hours.  Expected output size is roughly
120-180 MB, dominated by the compact float32 candidate matrix.
"""

from pathlib import Path
import gc
import json
import shutil
import time
import warnings

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
TRAJECTORY_DIR = OOF_ROOT / "trajectory_geology_state_gateway_v0"
CACHE_DIR = TRAJECTORY_DIR / "well_features"
NEIGHBOR_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / "shape_basket_ceiling_audit_v0"

RANDOM_STATE = 20260717
N_ESTIMATORS = 1800
LEARNING_RATE = 0.025

NEIGHBOR_WEIGHT = 0.25
NEIGHBOR_RAW_CAP = 24.0
MIN_ABS_DIRECTION_DOT = 0.70
EXPECTED_NEIGHBOR_RMSE = 8.818104
MAX_NEIGHBOR_LINEAGE_DELTA = 0.05

PRIMARY_CANDIDATES = [
    "geom_q15",
    "geom_q50",
    "geom_q85",
    "full_q50",
    "full_l1",
    "carry_s",
    "neighbor_frozen",
    "strong",
]
ALL_CANDIDATES = [*PRIMARY_CANDIDATES, "cp12"]
LEARNED_CANDIDATES = PRIMARY_CANDIDATES[:5]

MODEL_SPECS = {
    "geom_q15": {"arm": "geometry", "objective": "quantile", "alpha": 0.15},
    "geom_q50": {"arm": "geometry", "objective": "quantile", "alpha": 0.50},
    "geom_q85": {"arm": "geometry", "objective": "quantile", "alpha": 0.85},
    "full_q50": {"arm": "full", "objective": "quantile", "alpha": 0.50},
    "full_l1": {"arm": "full", "objective": "regression_l1", "alpha": None},
}

BASKETS = {
    "shape7_no_safe": LEARNED_CANDIDATES + ["carry_s", "neighbor_frozen"],
    "primary_b8": PRIMARY_CANDIDATES,
    "reference_b9_cp12": ALL_CANDIDATES,
}


def _rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _node_quadrature_weights(node_md):
    """Approximate row-pooled loss while training on sparse control nodes."""
    node_md = np.asarray(node_md, np.float64)
    if len(node_md) <= 1:
        return np.ones(len(node_md), np.float32)
    mid = 0.5 * (node_md[:-1] + node_md[1:])
    left = np.r_[node_md[0], mid]
    right = np.r_[mid, node_md[-1]]
    weight = np.maximum(right - left, 1.0)
    return weight.astype(np.float32)


def _interp_unique(x_new, x, y):
    """np.interp with a deterministic last-value rule for duplicate nodes."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    keep = np.r_[x[1:] != x[:-1], True]
    return np.interp(np.asarray(x_new, np.float64), x[keep], y[keep])


def _model(spec, seed):
    from lightgbm import LGBMRegressor

    parameters = dict(
        objective=spec["objective"],
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=15.0,
        max_bin=127,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )
    if spec["objective"] == "quantile":
        parameters["alpha"] = spec["alpha"]
    return LGBMRegressor(**parameters)


def _candidate_scores(truth, predictions, row_folds, top40_mask, names):
    score_rows = []
    fold_rows = []
    strong_index = names.index("strong")
    strong = predictions[:, strong_index]
    strong_fold_rmse = {
        fold: _rmse(truth[row_folds == fold], strong[row_folds == fold])
        for fold in sorted(np.unique(row_folds))
    }
    for index, name in enumerate(names):
        prediction = predictions[:, index]
        folds = []
        for fold in sorted(np.unique(row_folds)):
            mask = row_folds == fold
            score = _rmse(truth[mask], prediction[mask])
            gain = strong_fold_rmse[fold] - score
            folds.append(gain)
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": int(fold),
                    "rows": int(mask.sum()),
                    "strong_rmse": strong_fold_rmse[fold],
                    "rmse": score,
                    "gain_vs_strong": gain,
                }
            )
        score_rows.append(
            {
                "candidate": name,
                "rmse": _rmse(truth, prediction),
                "gain_vs_strong": _rmse(truth, strong) - _rmse(truth, prediction),
                "mean_fold_gain_vs_strong": float(np.mean(folds)),
                "min_fold_gain_vs_strong": float(np.min(folds)),
                "positive_folds_vs_strong": int(np.sum(np.asarray(folds) > 0)),
                "top40_rmse": _rmse(truth[top40_mask], prediction[top40_mask]),
                "top40_gain_vs_strong": (
                    _rmse(truth[top40_mask], strong[top40_mask])
                    - _rmse(truth[top40_mask], prediction[top40_mask])
                ),
            }
        )
    return pd.DataFrame(score_rows).sort_values("rmse"), pd.DataFrame(fold_rows)


for path in (
    CACHE_DIR,
    FOLD_PATH,
    NEIGHBOR_DIR / "neighbor_transfer_oof.npz",
    NEIGHBOR_DIR / "selected_neighbors.csv",
):
    if not path.exists():
        raise FileNotFoundError(path)
if POOL_PATH is None:
    raise FileNotFoundError(f"candidate pool not found under {OOF_ROOT}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

started = time.time()
print(
    {
        "experiment": "shape_basket_ceiling_audit_v0",
        "pool": str(POOL_PATH),
        "feature_cache": str(CACHE_DIR),
        "fold_source": str(FOLD_PATH),
        "model_specs": MODEL_SPECS,
        "primary_basket": PRIMARY_CANDIDATES,
    },
    flush=True,
)


# -----------------------------------------------------------------------------
# Load the fold-clean control-node training table once.
# -----------------------------------------------------------------------------
fold_frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
fold_frame = fold_frame.groupby("well", as_index=False)["fold"].first()
fold_frame["well"] = fold_frame["well"].astype(str)
fold_frame["fold"] = fold_frame["fold"].astype(np.int8)
fold_map = dict(zip(fold_frame["well"], fold_frame["fold"]))
wells = sorted(fold_map)

x_parts = []
y_parts = []
weight_parts = []
node_fold_parts = []
node_slices = {}
feature_names = None
geometry_width = None
offset = 0

for position, well in enumerate(wells, 1):
    path = CACHE_DIR / f"{well}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        current_names = data["feature_names"].astype(str)
        current_geometry_width = int(data["geometry_width"][0])
        if feature_names is None:
            feature_names = current_names
            geometry_width = current_geometry_width
        elif not np.array_equal(feature_names, current_names):
            raise RuntimeError(f"feature schema mismatch for {well}")
        elif geometry_width != current_geometry_width:
            raise RuntimeError(f"geometry width mismatch for {well}")

        x = data["features"].astype(np.float32)
        y = data["target_s_delta"].astype(np.float32)
        if len(x) != len(y):
            raise RuntimeError(f"node feature/target mismatch for {well}")
        x_parts.append(x)
        y_parts.append(y)
        weight_parts.append(_node_quadrature_weights(data["node_md"]))
        node_fold_parts.append(np.full(len(x), fold_map[well], np.int8))
        node_slices[well] = (offset, offset + len(x))
        offset += len(x)
    if position % 100 == 0:
        print(f"load node cache {position}/{len(wells)}", flush=True)

node_x = np.concatenate(x_parts)
node_y = np.concatenate(y_parts)
node_weight = np.concatenate(weight_parts)
node_folds = np.concatenate(node_fold_parts)
node_weight *= len(node_weight) / node_weight.sum()
node_x[~np.isfinite(node_x)] = np.nan
if not np.isfinite(node_y).all():
    raise RuntimeError("non-finite control-node target")
del x_parts, y_parts, weight_parts, node_fold_parts
gc.collect()

geometry_indices = np.arange(geometry_width)
full_indices = np.arange(len(feature_names))
arm_indices = {"geometry": geometry_indices, "full": full_indices}

print(
    "node table:",
    {
        "rows": len(node_y),
        "wells": len(wells),
        "features": len(feature_names),
        "geometry_features": geometry_width,
    },
    flush=True,
)


# -----------------------------------------------------------------------------
# Strict outer-fold training.  Validation wells are not used for early stopping.
# -----------------------------------------------------------------------------
node_predictions = {
    name: np.full(len(node_y), np.nan, np.float32) for name in LEARNED_CANDIDATES
}
model_rows = []

for fold in sorted(np.unique(node_folds)):
    train_mask = node_folds != fold
    valid_mask = node_folds == fold
    print(
        f"outer fold {fold}/5 train_nodes={train_mask.sum()} valid_nodes={valid_mask.sum()}",
        flush=True,
    )
    for model_number, (name, spec) in enumerate(MODEL_SPECS.items()):
        indices = arm_indices[spec["arm"]]
        model = _model(spec, RANDOM_STATE + 100 * int(fold) + model_number)
        fit_started = time.time()
        model.fit(
            node_x[train_mask][:, indices],
            node_y[train_mask],
            sample_weight=node_weight[train_mask],
        )
        prediction = model.predict(node_x[valid_mask][:, indices]).astype(np.float32)
        node_predictions[name][valid_mask] = prediction
        model_rows.append(
            {
                "fold": int(fold),
                "candidate": name,
                "arm": spec["arm"],
                "objective": spec["objective"],
                "alpha": spec["alpha"],
                "features": len(indices),
                "train_nodes": int(train_mask.sum()),
                "valid_nodes": int(valid_mask.sum()),
                "node_target_rmse": _rmse(node_y[valid_mask], prediction),
                "fit_seconds": time.time() - fit_started,
            }
        )
        print("  model", model_rows[-1], flush=True)
        del model, prediction
        gc.collect()

for name, prediction in node_predictions.items():
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"incomplete OOF node predictions for {name}")

model_report = pd.DataFrame(model_rows)
model_report.to_csv(OUT_DIR / "model_folds.csv", index=False)


# -----------------------------------------------------------------------------
# Load and establish the exact row order shared by the pool and Neighbor sidecar.
# -----------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", "strong", "cp12")
    if column in header
]
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
if "id" in pool.columns:
    pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
required_pool = {"well", "row_idx", "tvt_true", "strong", "cp12"}
if not required_pool.issubset(pool.columns):
    raise RuntimeError(f"pool missing columns: {sorted(required_pool - set(pool.columns))}")
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool_row_idx = pool["row_idx"].to_numpy(np.int32)

row_wells = pool["well"].to_numpy(str)
boundaries = np.r_[0, np.flatnonzero(row_wells[1:] != row_wells[:-1]) + 1, len(pool)]
row_slices = {
    str(row_wells[start]): (int(start), int(end))
    for start, end in zip(boundaries[:-1], boundaries[1:])
}
if set(row_slices) != set(wells):
    raise RuntimeError(
        f"pool/cache well mismatch: pool_only={len(set(row_slices)-set(wells))}, "
        f"cache_only={len(set(wells)-set(row_slices))}"
    )

truth = pool["tvt_true"].to_numpy(np.float64)
row_folds = pool["well"].map(fold_map).to_numpy(np.int8)
candidate_index = {name: index for index, name in enumerate(ALL_CANDIDATES)}
candidate_matrix = np.full((len(pool), len(ALL_CANDIDATES)), np.nan, np.float32)

interp_sse = {
    name: {int(fold): 0.0 for fold in sorted(np.unique(node_folds))}
    for name in LEARNED_CANDIDATES
}
interp_tvt_sse = {
    name: {int(fold): 0.0 for fold in sorted(np.unique(node_folds))}
    for name in LEARNED_CANDIDATES
}
interp_rows = {int(fold): 0 for fold in sorted(np.unique(node_folds))}

for position, well in enumerate(wells, 1):
    row_start, row_end = row_slices[well]
    node_start, node_end = node_slices[well]
    fold = int(fold_map[well])
    with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
        eval_rows = data["eval_rows"].astype(np.int32)
        pool_rows = pool_row_idx[row_start:row_end]
        if not np.array_equal(eval_rows, pool_rows):
            raise RuntimeError(f"row alignment mismatch for {well}")
        eval_md = data["eval_md"].astype(np.float64)
        eval_z = data["eval_z"].astype(np.float64)
        eval_truth = data["eval_truth"].astype(np.float64)
        if np.max(np.abs(eval_truth - truth[row_start:row_end])) > 1e-3:
            raise RuntimeError(f"truth alignment mismatch for {well}")
        node_md = data["node_md"].astype(np.float64)
        node_z = data["node_z"].astype(np.float64)
        anchor_s = float(data["anchor_s"][0])
        anchor_tvt = float(data["anchor_tvt"][0])

        for name in LEARNED_CANDIDATES:
            node_s_delta = node_predictions[name][node_start:node_end].astype(np.float64).copy()
            node_s_delta[0] = 0.0

            eval_s_delta = _interp_unique(eval_md, node_md, node_s_delta)
            prediction_s = anchor_s + eval_s_delta - eval_z

            node_tvt = anchor_s + node_s_delta - node_z
            node_tvt[0] = anchor_tvt
            prediction_tvt = _interp_unique(eval_md, node_md, node_tvt)

            candidate_matrix[row_start:row_end, candidate_index[name]] = prediction_s.astype(
                np.float32
            )
            interp_sse[name][fold] += float(np.sum((prediction_s - eval_truth) ** 2))
            interp_tvt_sse[name][fold] += float(np.sum((prediction_tvt - eval_truth) ** 2))

        candidate_matrix[row_start:row_end, candidate_index["carry_s"]] = (
            anchor_s - eval_z
        ).astype(np.float32)
        interp_rows[fold] += len(eval_rows)

    if position % 100 == 0:
        print(f"reconstruct learned shapes {position}/{len(wells)}", flush=True)


# -----------------------------------------------------------------------------
# Reconstruct the exact hidden-confirmed Neighbor mechanism in OOF row order.
# -----------------------------------------------------------------------------
with np.load(NEIGHBOR_DIR / "neighbor_transfer_oof.npz", allow_pickle=False) as sidecar:
    neighbor_raw = sidecar["structure_progress"].astype(np.float64)
if len(neighbor_raw) != len(pool):
    raise RuntimeError(f"Neighbor sidecar row mismatch: {len(neighbor_raw)} != {len(pool)}")

neighbor_meta = pd.read_csv(NEIGHBOR_DIR / "selected_neighbors.csv")
neighbor_meta["well"] = neighbor_meta["well"].astype(str)
neighbor_meta = neighbor_meta.set_index("well")
median_distance = pool["well"].map(neighbor_meta["median_distance"]).to_numpy(np.float64)
p90_distance = pool["well"].map(neighbor_meta["p90_distance"]).to_numpy(np.float64)
anchor_distance = pool["well"].map(neighbor_meta["anchor_distance_full"]).to_numpy(np.float64)
direction_dot = pool["well"].map(neighbor_meta["direction_dot"]).to_numpy(np.float64)
neighbor_eligible = (
    (median_distance <= 600.0)
    & (p90_distance <= 1200.0)
    & (anchor_distance <= 1200.0)
    & (np.abs(direction_dot) >= MIN_ABS_DIRECTION_DOT)
    & np.isfinite(neighbor_raw)
)

strong = pool["strong"].to_numpy(np.float64)
neighbor_move = np.zeros(len(pool), np.float64)
neighbor_move[neighbor_eligible] = NEIGHBOR_WEIGHT * np.clip(
    neighbor_raw[neighbor_eligible] - strong[neighbor_eligible],
    -NEIGHBOR_RAW_CAP,
    NEIGHBOR_RAW_CAP,
)
neighbor_prediction = strong + neighbor_move
neighbor_rmse = _rmse(truth, neighbor_prediction)
neighbor_lineage_match = abs(neighbor_rmse - EXPECTED_NEIGHBOR_RMSE) <= MAX_NEIGHBOR_LINEAGE_DELTA
if not neighbor_lineage_match:
    raise RuntimeError(
        f"frozen Neighbor lineage mismatch: got {neighbor_rmse:.6f}, "
        f"expected about {EXPECTED_NEIGHBOR_RMSE:.6f}"
    )

candidate_matrix[:, candidate_index["neighbor_frozen"]] = neighbor_prediction.astype(np.float32)
candidate_matrix[:, candidate_index["strong"]] = strong.astype(np.float32)
candidate_matrix[:, candidate_index["cp12"]] = pool["cp12"].to_numpy(np.float32)

if not np.isfinite(candidate_matrix).all():
    bad = np.argwhere(~np.isfinite(candidate_matrix))[:10]
    raise RuntimeError(f"non-finite candidate predictions; first positions: {bad.tolist()}")


# -----------------------------------------------------------------------------
# G1: paired S-space versus TVT-space interpolation report.
# -----------------------------------------------------------------------------
interp_fold_rows = []
for name in LEARNED_CANDIDATES:
    for fold in sorted(interp_rows):
        rows = interp_rows[fold]
        s_rmse = float(np.sqrt(interp_sse[name][fold] / rows))
        tvt_rmse = float(np.sqrt(interp_tvt_sse[name][fold] / rows))
        interp_fold_rows.append(
            {
                "candidate": name,
                "fold": fold,
                "rows": rows,
                "s_interp_rmse": s_rmse,
                "tvt_interp_rmse": tvt_rmse,
                "s_gain_vs_tvt": tvt_rmse - s_rmse,
            }
        )
interp_folds = pd.DataFrame(interp_fold_rows)
interp_overall_rows = []
for name in LEARNED_CANDIDATES:
    rows = sum(interp_rows.values())
    s_rmse = float(np.sqrt(sum(interp_sse[name].values()) / rows))
    tvt_rmse = float(np.sqrt(sum(interp_tvt_sse[name].values()) / rows))
    gains = interp_folds.loc[interp_folds["candidate"] == name, "s_gain_vs_tvt"]
    interp_overall_rows.append(
        {
            "candidate": name,
            "rows": rows,
            "s_interp_rmse": s_rmse,
            "tvt_interp_rmse": tvt_rmse,
            "s_gain_vs_tvt": tvt_rmse - s_rmse,
            "mean_fold_gain": float(gains.mean()),
            "min_fold_gain": float(gains.min()),
            "positive_folds": int((gains > 0).sum()),
        }
    )
interp_overall = pd.DataFrame(interp_overall_rows).sort_values("s_interp_rmse")
interp_folds.to_csv(OUT_DIR / "interpolation_folds.csv", index=False)
interp_overall.to_csv(OUT_DIR / "interpolation_overall.csv", index=False)


# -----------------------------------------------------------------------------
# Candidate quality, top-40 behavior, correlation, and whole-well oracle.
# -----------------------------------------------------------------------------
well_rows = []
well_sse = np.zeros((len(wells), len(ALL_CANDIDATES)), np.float64)
strong_i = candidate_index["strong"]
for well_i, well in enumerate(wells):
    start, end = row_slices[well]
    error = candidate_matrix[start:end].astype(np.float64) - truth[start:end, None]
    well_sse[well_i] = np.sum(error**2, axis=0)
    well_rows.append(end - start)
well_rows = np.asarray(well_rows, np.int64)

top40_indices = np.argsort(well_sse[:, strong_i])[-40:]
top40_wells = {wells[index] for index in top40_indices}
top40_mask = pool["well"].isin(top40_wells).to_numpy(bool)

candidate_scores, candidate_folds = _candidate_scores(
    truth, candidate_matrix, row_folds, top40_mask, ALL_CANDIDATES
)
candidate_scores.to_csv(OUT_DIR / "candidate_scores.csv", index=False)
candidate_folds.to_csv(OUT_DIR / "candidate_folds.csv", index=False)

# Chunked float64 moments avoid a large float64 error matrix.
n_candidates = len(ALL_CANDIDATES)
gram = np.zeros((n_candidates, n_candidates), np.float64)
error_sum = np.zeros(n_candidates, np.float64)
chunk_size = 250_000
for start in range(0, len(pool), chunk_size):
    end = min(len(pool), start + chunk_size)
    error = candidate_matrix[start:end].astype(np.float64) - truth[start:end, None]
    gram += error.T @ error
    error_sum += error.sum(axis=0)
norm = np.sqrt(np.maximum(np.diag(gram), 1e-12))
cosine = gram / np.outer(norm, norm)
centered_gram = gram - np.outer(error_sum, error_sum) / len(pool)
centered_norm = np.sqrt(np.maximum(np.diag(centered_gram), 1e-12))
pearson = centered_gram / np.outer(centered_norm, centered_norm)
pd.DataFrame(cosine, index=ALL_CANDIDATES, columns=ALL_CANDIDATES).to_csv(
    OUT_DIR / "error_cosine.csv"
)
pd.DataFrame(pearson, index=ALL_CANDIDATES, columns=ALL_CANDIDATES).to_csv(
    OUT_DIR / "error_pearson.csv"
)

well_report = pd.DataFrame({"well": wells, "rows": well_rows})
well_report["fold"] = well_report["well"].map(fold_map).astype(int)
for index, name in enumerate(ALL_CANDIDATES):
    well_report[f"{name}_rmse"] = np.sqrt(well_sse[:, index] / well_rows)
well_report["top40_strong"] = well_report["well"].isin(top40_wells)

oracle_rows = []
oracle_selection_rows = []
for basket, basket_names in BASKETS.items():
    indices = np.asarray([candidate_index[name] for name in basket_names], int)
    local_sse = well_sse[:, indices]
    selected_local = np.argmin(local_sse, axis=1)
    selected_indices = indices[selected_local]
    selected_sse = well_sse[np.arange(len(wells)), selected_indices]
    selected_names = np.asarray(ALL_CANDIDATES, object)[selected_indices]

    single_scores = {
        name: float(np.sqrt(well_sse[:, candidate_index[name]].sum() / well_rows.sum()))
        for name in basket_names
    }
    best_single_name = min(single_scores, key=single_scores.get)
    best_single_rmse = single_scores[best_single_name]
    oracle_rmse = float(np.sqrt(selected_sse.sum() / well_rows.sum()))
    top_selected_sse = selected_sse[top40_indices]
    top_rows = well_rows[top40_indices]
    top40_oracle_rmse = float(np.sqrt(top_selected_sse.sum() / top_rows.sum()))
    top40_best_single_rmse = float(
        np.sqrt(well_sse[top40_indices, candidate_index[best_single_name]].sum() / top_rows.sum())
    )

    oracle_rows.append(
        {
            "basket": basket,
            "candidates": len(basket_names),
            "best_single": best_single_name,
            "best_single_rmse": best_single_rmse,
            "whole_well_oracle_rmse": oracle_rmse,
            "oracle_gain_vs_best_single": best_single_rmse - oracle_rmse,
            "top40_best_single_rmse": top40_best_single_rmse,
            "top40_oracle_rmse": top40_oracle_rmse,
            "top40_oracle_gain": top40_best_single_rmse - top40_oracle_rmse,
            "learned_shape_selection_fraction": float(
                np.mean(np.isin(selected_names, LEARNED_CANDIDATES))
            ),
        }
    )
    counts = pd.Series(selected_names).value_counts()
    for name in basket_names:
        oracle_selection_rows.append(
            {
                "basket": basket,
                "candidate": name,
                "selected_wells": int(counts.get(name, 0)),
                "selected_fraction": float(counts.get(name, 0) / len(wells)),
            }
        )
    well_report[f"{basket}_oracle_candidate"] = selected_names

oracle_report = pd.DataFrame(oracle_rows)
oracle_selection = pd.DataFrame(oracle_selection_rows)
oracle_report.to_csv(OUT_DIR / "whole_well_oracle.csv", index=False)
oracle_selection.to_csv(OUT_DIR / "oracle_selection_counts.csv", index=False)
well_report.to_csv(OUT_DIR / "well_scores_and_oracle.csv", index=False)


# -----------------------------------------------------------------------------
# Compact handoff for the dSSE-weighted selector experiment.
# -----------------------------------------------------------------------------
free_bytes = shutil.disk_usage(OOF_ROOT).free
estimated_output_bytes = int(candidate_matrix.nbytes * 1.25 + row_folds.nbytes)
if free_bytes < estimated_output_bytes + 512 * 1024**2:
    raise RuntimeError(
        f"insufficient disk for candidate handoff: free={free_bytes / 1024**3:.2f} GB, "
        f"estimated={estimated_output_bytes / 1024**3:.2f} GB"
    )

well_starts = np.asarray([row_slices[well][0] for well in wells], np.int64)
well_ends = np.asarray([row_slices[well][1] for well in wells], np.int64)
np.savez_compressed(
    OUT_DIR / "shape_basket_oof.npz",
    predictions=candidate_matrix,
    candidate_names=np.asarray(ALL_CANDIDATES, str),
    primary_candidate_names=np.asarray(PRIMARY_CANDIDATES, str),
    row_folds=row_folds,
    well_names=np.asarray(wells, str),
    well_starts=well_starts,
    well_ends=well_ends,
)

pd.DataFrame(
    {
        "candidate": ALL_CANDIDATES,
        "source_family": [
            "learned_geometry_quantile",
            "learned_geometry_quantile",
            "learned_geometry_quantile",
            "learned_full_quantile",
            "learned_full_l1",
            "deterministic_anchor_S",
            "fold_clean_neighbor_transfer",
            "historical_oof_safe",
            "historical_oof_reference",
        ],
        "primary_b8": [name in PRIMARY_CANDIDATES for name in ALL_CANDIDATES],
    }
).to_csv(OUT_DIR / "candidate_manifest.csv", index=False)

primary = oracle_report.loc[oracle_report["basket"] == "primary_b8"].iloc[0]
s_interp_pass = bool(
    (interp_overall["s_gain_vs_tvt"].median() > 0)
    and (interp_overall["positive_folds"] >= 4).sum() >= 3
)
ceiling_pass = bool(
    primary["whole_well_oracle_rmse"] <= 7.80
    and primary["oracle_gain_vs_best_single"] >= 0.75
    and primary["learned_shape_selection_fraction"] >= 0.05
)
verdict = {
    "experiment": "shape_basket_ceiling_audit_v0",
    "pool": str(POOL_PATH),
    "fold_source": str(FOLD_PATH),
    "rows": len(pool),
    "wells": len(wells),
    "control_nodes": len(node_y),
    "features": len(feature_names),
    "geometry_features": geometry_width,
    "neighbor_rmse": neighbor_rmse,
    "neighbor_lineage_match": neighbor_lineage_match,
    "primary_basket": PRIMARY_CANDIDATES,
    "primary_best_single_rmse": float(primary["best_single_rmse"]),
    "primary_whole_well_oracle_rmse": float(primary["whole_well_oracle_rmse"]),
    "primary_oracle_gain": float(primary["oracle_gain_vs_best_single"]),
    "primary_top40_oracle_gain": float(primary["top40_oracle_gain"]),
    "primary_learned_selection_fraction": float(primary["learned_shape_selection_fraction"]),
    "s_interpolation_pass": s_interp_pass,
    "candidate_ceiling_pass": ceiling_pass,
    "strict_pass": ceiling_pass,
    "gates": {
        "whole_well_oracle_rmse_max": 7.80,
        "oracle_gain_vs_best_single_min": 0.75,
        "learned_shape_selection_fraction_min": 0.05,
    },
    "next": (
        "PASS -> train dSSE-weighted pairwise/listwise selector on this frozen basket"
        if ceiling_pass
        else "FAIL -> stop shape-selector route; basket ceiling is insufficient"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "candidate_handoff": str(OUT_DIR / "shape_basket_oof.npz"),
    "free_gb_after": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, default=_json_default)

print("\nMODEL FOLDS")
print(model_report.to_string(index=False))
print("\nS VS TVT INTERPOLATION")
print(interp_overall.to_string(index=False))
print("\nCANDIDATE SCORES")
print(candidate_scores.to_string(index=False))
print("\nWHOLE-WELL ORACLE")
print(oracle_report.to_string(index=False))
print("\nORACLE SELECTION")
print(oracle_selection.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT_DIR)
