"""Nested fixed convex blend of V40's far-tail endpoints.

V43 proved that the useful candidate identity is coherent per well. V44/V45
showed that legal evidence cannot reliably classify that identity. This test
removes classification entirely: learn one non-negative endpoint mixture on
training folds and apply it unchanged to held wells. All rows through 2000 ft
remain exactly equal to the V33 / LB 6.377 lineage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize


EXPERIMENT = "frontier_6p377_far_simplex_blend_v46"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
LAMBDAS = (0.0, 0.001, 0.01, 0.05, 0.10, 0.50, 1.0, 4.0, 16.0)
SEED = 20260727
started = time.time()
FIT_STATS = {}


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _smoothstep(values):
    t = np.clip((np.asarray(values, np.float64) - FAR_START_FT) / FADE_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _centred_mean(x, values, width):
    x = np.asarray(x, np.float64)
    values = np.asarray(values, np.float64)
    left = np.searchsorted(x, x - width / 2.0, side="left")
    right = np.searchsorted(x, x + width / 2.0, side="right")
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[right] - prefix[left]) / np.maximum(right - left, 1)


def _metrics(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local["tvt_true"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    wells = local["well"].astype(str).to_numpy()
    per_well = pd.Series(error**2).groupby(wells, sort=False).mean().pow(0.5)
    result = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
    }
    for name, mask in {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
    return result


def _fit_weights(row_mask, regularization):
    fit = np.asarray(row_mask, bool) & far
    key = tuple(sorted(np.unique(fold_values[fit]).astype(int).tolist()))
    if key not in FIT_STATS:
        design = deltas[fit, 1:]
        target = truth[fit] - base[fit]
        if len(target) < 1000:
            raise RuntimeError("Too few far rows for simplex fit")
        FIT_STATS[key] = {
            "gram": (design.T @ design) / len(target),
            "cross": (design.T @ target) / len(target),
            "target2": float(np.mean(target**2)),
            "rows": int(len(target)),
        }
    stats = FIT_STATS[key]
    gram = stats["gram"]
    cross = stats["cross"]

    def objective(q):
        return float(
            q @ gram @ q - 2.0 * np.dot(cross, q) + stats["target2"]
            + float(regularization) * np.dot(q, q)
        )

    def gradient(q):
        return 2.0 * (gram @ q - cross) + 2.0 * float(regularization) * q

    initial = np.full(gram.shape[0], 0.05, np.float64)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * gram.shape[0],
        constraints=[{"type": "ineq", "fun": lambda q: 1.0 - np.sum(q), "jac": lambda q: -np.ones_like(q)}],
        options={"maxiter": 300, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Simplex optimization failed: {result.message}")
    nonbase = np.clip(result.x, 0.0, 1.0)
    if nonbase.sum() > 1.0:
        nonbase /= nonbase.sum()
    return np.concatenate([[1.0 - nonbase.sum()], nonbase])


def _predict(weights, row_mask=None):
    prediction = candidates @ np.asarray(weights, np.float64)
    if row_mask is not None:
        output = base.copy()
        output[np.asarray(row_mask, bool)] = prediction[np.asarray(row_mask, bool)]
        prediction = output
    prediction[~far] = base[~far]
    return prediction


for required in (V33_PATH, V28_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", "pre_neighbor_cp8", BASE_COLUMN]
frame = pd.read_parquet(V33_PATH, columns=columns)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame[frame["fold"].isin(FOLDS)].sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

tail_columns = sorted(column for column in pq.ParquetFile(V28_PATH).schema.names if column.startswith("tail_"))
tail = pd.read_parquet(V28_PATH, columns=["id"] + tail_columns)
tail["id"] = tail["id"].astype(str)
frame = frame.merge(tail, on="id", how="left", validate="one_to_one")

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold_values = frame["fold"].to_numpy(np.int8)
fade = _smoothstep(distance)
far = distance > FAR_START_FT

raw_paths = [base]
candidate_names = ["base_6p377"]
for name in ["pre_neighbor_cp8"] + tail_columns:
    absolute = frame[name].to_numpy(np.float64)
    finite = np.isfinite(absolute)
    if finite.mean() < 0.95:
        continue
    correction = np.where(finite, np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[~far] = base[~far]
    raw_paths.append(candidate)
    candidate_names.append(name)
candidates = np.column_stack(raw_paths)

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}
for indices in groups.values():
    local_distance = distance[indices]
    for state in range(1, candidates.shape[1]):
        correction = _centred_mean(local_distance, candidates[indices, state] - base[indices], SMOOTH_FT)
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        selected = base[indices] + fade[indices] * correction
        selected[local_distance <= FAR_START_FT] = base[indices][local_distance <= FAR_START_FT]
        candidates[indices, state] = selected
if not np.isfinite(candidates).all():
    raise RuntimeError("Non-finite candidate matrix")
deltas = candidates - base[:, None]

base_metrics = _metrics(frame, base, "base_6p377")
outer_prediction = base.copy()
fold_rows = []
weight_rows = []
inner_rows = []

for held_fold in FOLDS:
    train_folds = [fold for fold in FOLDS if fold != held_fold]
    lambda_scores = []
    for regularization in LAMBDAS:
        validation_sse = 0.0
        validation_rows = 0
        minimum_gain = np.inf
        for inner_held in train_folds:
            fit_mask = np.isin(fold_values, [fold for fold in train_folds if fold != inner_held])
            validation_mask = fold_values == inner_held
            weights = _fit_weights(fit_mask, regularization)
            prediction = _predict(weights)
            error = prediction[validation_mask] - truth[validation_mask]
            base_error = base[validation_mask] - truth[validation_mask]
            validation_sse += float(np.dot(error, error))
            validation_rows += int(validation_mask.sum())
            gain = float(np.sqrt(np.mean(base_error**2)) - np.sqrt(np.mean(error**2)))
            minimum_gain = min(minimum_gain, gain)
        score = float(np.sqrt(validation_sse / validation_rows))
        lambda_scores.append((regularization, score, minimum_gain))
        inner_rows.append({
            "outer_fold": held_fold,
            "regularization": regularization,
            "inner_rmse": score,
            "minimum_inner_gain": minimum_gain,
        })
    robust = [row for row in lambda_scores if row[2] >= -0.005]
    selected_lambda, inner_rmse, minimum_inner_gain = min(robust or lambda_scores, key=lambda row: row[1])

    train_mask = fold_values != held_fold
    held_mask = fold_values == held_fold
    weights = _fit_weights(train_mask, selected_lambda)
    prediction = _predict(weights)
    outer_prediction[held_mask] = prediction[held_mask]

    local = frame.loc[held_mask].reset_index(drop=True)
    base_local = _metrics(local, base[held_mask], "base")
    blend_local = _metrics(local, prediction[held_mask], "blend")
    fold_rows.append({
        "fold": held_fold,
        "selected_lambda": selected_lambda,
        "inner_rmse": inner_rmse,
        "minimum_inner_gain": minimum_inner_gain,
        "base_rmse": base_local["rmse"],
        "rmse": blend_local["rmse"],
        "gain": base_local["rmse"] - blend_local["rmse"],
        "base_far_rmse": base_local["far_gt2000_rmse"],
        "far_rmse": blend_local["far_gt2000_rmse"],
        "far_gain": base_local["far_gt2000_rmse"] - blend_local["far_gt2000_rmse"],
        "p95_change": blend_local["p95_well_rmse"] - base_local["p95_well_rmse"],
    })
    for candidate, weight in zip(candidate_names, weights):
        weight_rows.append({"fold": held_fold, "candidate": candidate, "weight": float(weight)})

outer_metrics = _metrics(frame, outer_prediction, "nested_simplex_blend")

# A semantic-alignment control: retain the learned weight vector but permute
# non-base endpoint identities independently within each held well.
rng = np.random.default_rng(SEED)
shuffle_prediction = base.copy()
weight_table = pd.DataFrame(weight_rows)
for held_fold in FOLDS:
    weights = weight_table[weight_table["fold"].eq(held_fold)].set_index("candidate").loc[candidate_names, "weight"].to_numpy()
    held_wells = frame.loc[fold_values == held_fold, "well"].drop_duplicates().tolist()
    for well in held_wells:
        indices = groups[str(well)]
        permutation = np.concatenate([[0], rng.permutation(np.arange(1, candidates.shape[1]))])
        shuffle_prediction[indices] = candidates[indices][:, permutation] @ weights
shuffle_prediction[~far] = base[~far]
shuffle_metrics = _metrics(frame, shuffle_prediction, "candidate_identity_shuffle")

# Select one deployment regularization by threefold cross-prediction, then fit
# all observed folds. This artifact is only packaged if the nested gate passes.
deployment_scan = []
for regularization in LAMBDAS:
    prediction = base.copy()
    minimum_gain = np.inf
    for held_fold in FOLDS:
        weights = _fit_weights(fold_values != held_fold, regularization)
        local_prediction = _predict(weights)
        mask = fold_values == held_fold
        prediction[mask] = local_prediction[mask]
        gain = _metrics(frame.loc[mask].reset_index(drop=True), base[mask], "base")["rmse"] - _metrics(
            frame.loc[mask].reset_index(drop=True), local_prediction[mask], "blend"
        )["rmse"]
        minimum_gain = min(minimum_gain, gain)
    metrics = _metrics(frame, prediction, "crossfit")
    deployment_scan.append({"regularization": regularization, "rmse": metrics["rmse"], "minimum_fold_gain": minimum_gain})
deployment_scan = pd.DataFrame(deployment_scan)
eligible = deployment_scan[deployment_scan["minimum_fold_gain"] >= -0.005]
deployment_row = (eligible if len(eligible) else deployment_scan).sort_values("rmse").iloc[0]
deployment_lambda = float(deployment_row["regularization"])
deployment_weights = _fit_weights(np.ones(len(frame), bool), deployment_lambda)

fold_report = pd.DataFrame(fold_rows)
weight_report = pd.DataFrame(weight_rows)
weight_summary = weight_report.groupby("candidate", as_index=False).agg(
    mean_weight=("weight", "mean"), min_weight=("weight", "min"), max_weight=("weight", "max"), std_weight=("weight", "std")
)
deployment = pd.DataFrame({"candidate": candidate_names, "weight": deployment_weights})

fold_report.to_csv(OUT_DIR / "folds.csv", index=False)
weight_report.to_csv(OUT_DIR / "outer_weights.csv", index=False)
weight_summary.to_csv(OUT_DIR / "weight_stability.csv", index=False)
pd.DataFrame(inner_rows).to_csv(OUT_DIR / "inner_scan.csv", index=False)
deployment_scan.to_csv(OUT_DIR / "deployment_lambda_scan.csv", index=False)
deployment.to_csv(OUT_DIR / "deployment_weights.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    base_6p377=base.astype(np.float32), nested_simplex_blend=outer_prediction.astype(np.float32)
).to_parquet(OUT_DIR / "simplex_blend_oof.parquet", index=False, compression="zstd")

checks = {
    "overall_gain_ge_0p03": bool(base_metrics["rmse"] - outer_metrics["rmse"] >= 0.03),
    "minimum_fold_gain_ge_0": bool(fold_report["gain"].min() >= 0.0),
    "minimum_far_gain_ge_0p03": bool(fold_report["far_gain"].min() >= 0.03),
    "maximum_p95_change_le_0p10": bool(fold_report["p95_change"].max() <= 0.10),
    "shuffle_worse_by_ge_0p02": bool(shuffle_metrics["rmse"] - outer_metrics["rmse"] >= 0.02),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_fixed_far_endpoint_simplex_blend",
    "sample_unit": "complete physical well",
    "lineage": "V33 current_dose_0p20 / LB 6.377",
    "candidate_names": candidate_names,
    "base": base_metrics,
    "nested_blend": outer_metrics,
    "gain": base_metrics["rmse"] - outer_metrics["rmse"],
    "far_gain": base_metrics["far_gt2000_rmse"] - outer_metrics["far_gt2000_rmse"],
    "candidate_identity_shuffle": shuffle_metrics,
    "shuffle_loss": shuffle_metrics["rmse"] - outer_metrics["rmse"],
    "folds": fold_rows,
    "deployment_lambda": deployment_lambda,
    "deployment_weights": dict(zip(candidate_names, deployment_weights)),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> translate the fixed far-tail weights into a Kaggle addon and submit as a conservative candidate"
        if all(checks.values())
        else "FAIL -> convex averaging cannot bypass endpoint identity; close this bank"
    ),
}
_write_json(OUT_DIR / "result.json", verdict)

print("\nV46 FOLDS")
print(fold_report.to_string(index=False))
print("\nV46 WEIGHT STABILITY")
print(weight_summary.to_string(index=False))
print("\nV46 DEPLOYMENT WEIGHTS")
print(deployment.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT_DIR)
