"""Zero-training audit of spatial evidence for V43's coherent tail endpoint.

V43 showed that a single endpoint per well captures most of the far-tail oracle,
while V44 showed that GR/path features cannot identify that endpoint.  This
audit asks the next narrower question: does the already-deployable Neighbor
branch contain candidate-specific evidence about the correct endpoint?

No predictor is fitted.  Candidate-score orientation is estimated on the other
two observed folds and applied to the held fold.  This is a diagnostic gateway,
not a deployable model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


EXPERIMENT = "frontier_6p377_neighbor_identity_audit_v45"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
V44_FEATURE_PATH = OOF_ROOT / "frontier_6p377_far_wholewell_selector_v44" / "well_candidate_features.parquet"
NEIGHBOR_META_PATH = OOF_ROOT / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
CURRENT_NEIGHBOR_DOSE = 0.20
SEED = 20260727
started = time.time()


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


def _safe_corr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) < 1e-9 or np.std(b[valid]) < 1e-9:
        return 0.0
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _safe_cosine(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-9 else 0.0


def _safe_spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.unique(a[valid]).size < 2 or np.unique(b[valid]).size < 2:
        return 0.0
    result = spearmanr(a[valid], b[valid]).statistic
    return float(result) if np.isfinite(result) else 0.0


def _metrics(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local["tvt_true"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    wells = local["well"].astype(str).to_numpy()
    per_well = pd.Series(error**2).groupby(wells, sort=False).mean().pow(0.5)
    far = distance > FAR_START_FT
    return {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


for required in (V33_PATH, V28_PATH, V44_FEATURE_PATH, NEIGHBOR_META_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v33_columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", "pre_neighbor_cp8", BASE_COLUMN]
frame = pd.read_parquet(V33_PATH, columns=v33_columns)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame[frame["fold"].isin(FOLDS)].sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

tail_columns = sorted(column for column in pq.ParquetFile(V28_PATH).schema.names if column.startswith("tail_"))
tail = pd.read_parquet(V28_PATH, columns=["id"] + tail_columns)
tail["id"] = tail["id"].astype(str)
frame = frame.merge(tail, on="id", how="left", validate="one_to_one")

base = frame[BASE_COLUMN].to_numpy(np.float64)
pre_neighbor = frame["pre_neighbor_cp8"].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fade = _smoothstep(distance)
prefix = distance <= FAR_START_FT

candidate_paths = [base]
candidate_names = ["base_6p377"]
for name in ["pre_neighbor_cp8"] + tail_columns:
    absolute = frame[name].to_numpy(np.float64)
    finite = np.isfinite(absolute)
    if finite.mean() < 0.95:
        continue
    correction = np.where(finite, np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[prefix] = base[prefix]
    candidate_paths.append(candidate)
    candidate_names.append(name)
candidates = np.column_stack(candidate_paths)
if candidates.shape[1] < 3 or not np.isfinite(candidates).all():
    raise RuntimeError(f"Invalid candidate bank: {candidate_names}")

features = pd.read_parquet(V44_FEATURE_PATH)
features["well"] = features["well"].astype(str)
features = features[features["fold"].isin(FOLDS)].copy()
if features["state"].max() >= candidates.shape[1]:
    raise RuntimeError("V44 state count does not match reconstructed candidate bank")

neighbor_meta = pd.read_csv(NEIGHBOR_META_PATH)
if "well" not in neighbor_meta.columns:
    raise RuntimeError(f"selected_neighbors.csv lacks well: {neighbor_meta.columns.tolist()}")
neighbor_meta["well"] = neighbor_meta["well"].astype(str)
neighbor_meta = neighbor_meta.drop_duplicates("well", keep="last")
meta_numeric = [
    column for column in neighbor_meta.columns
    if column != "well" and pd.to_numeric(neighbor_meta[column], errors="coerce").notna().mean() >= 0.80
]
for column in meta_numeric:
    neighbor_meta[column] = pd.to_numeric(neighbor_meta[column], errors="coerce")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}

# V44 scores the smoothed, capped version of every endpoint.  Reproduce that
# exact candidate lineage before deriving spatial features or checking labels.
feasible_candidates = candidates.copy()
for indices in groups.values():
    local_distance = distance[indices]
    for state in range(1, candidates.shape[1]):
        correction = _centred_mean(
            local_distance,
            candidates[indices, state] - base[indices],
            SMOOTH_FT,
        )
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        selected = base[indices] + fade[indices] * correction
        selected[local_distance <= FAR_START_FT] = base[indices][local_distance <= FAR_START_FT]
        feasible_candidates[indices, state] = selected
candidates = feasible_candidates

meta_index = neighbor_meta.set_index("well")
records = []
for well, indices in groups.items():
    local_distance = distance[indices]
    far = local_distance > FAR_START_FT
    if not np.any(far):
        continue

    # V33 current = pre-neighbor CP8 + 0.20 * full Neighbor correction.
    applied = base[indices] - pre_neighbor[indices]
    full_neighbor = applied / CURRENT_NEIGHBOR_DOSE
    residual_neighbor = full_neighbor - applied
    applied_far = applied[far]
    residual_far = residual_neighbor[far]

    geometry = {}
    if well in meta_index.index:
        row = meta_index.loc[well]
        geometry = {column: float(row[column]) for column in meta_numeric if np.isfinite(row[column])}
    median_distance = geometry.get("median_distance", np.nan)
    p90_distance = geometry.get("p90_distance", np.nan)
    abs_dot = geometry.get("abs_direction_dot", np.nan)
    geometry_confidence = (
        np.exp(-max(median_distance, 0.0) / 300.0)
        * np.exp(-max(p90_distance, 0.0) / 600.0)
        * np.clip(abs_dot, 0.0, 1.0)
        if np.isfinite(median_distance) and np.isfinite(p90_distance) and np.isfinite(abs_dot)
        else 0.0
    )

    for state, name in enumerate(candidate_names):
        candidate_delta = candidates[indices, state] - base[indices]
        candidate_far = candidate_delta[far]
        target_gain = float(np.mean((base[indices] - truth[indices]) ** 2) - np.mean((candidates[indices, state] - truth[indices]) ** 2))
        record = {
            "well": well,
            "fold": int(frame.loc[indices[0], "fold"]),
            "state": int(state),
            "candidate": name,
            "target_gain_mse": target_gain,
            "is_base": float(state == 0),
            "geometry_available": float(bool(geometry)),
            "geometry_confidence": float(geometry_confidence),
            "neighbor_applied_rms": float(np.sqrt(np.mean(applied_far**2))),
            "neighbor_applied_end": float(applied_far[-1]),
            "neighbor_residual_rms": float(np.sqrt(np.mean(residual_far**2))),
            "candidate_rms": float(np.sqrt(np.mean(candidate_far**2))),
            "candidate_end": float(candidate_far[-1]),
            "candidate_vs_residual_rmse": float(np.sqrt(np.mean((candidate_far - residual_far) ** 2))),
            "candidate_vs_undo_rmse": float(np.sqrt(np.mean((candidate_far + applied_far) ** 2))),
            "candidate_neighbor_corr": _safe_corr(candidate_far, residual_far),
            "candidate_neighbor_cosine": _safe_cosine(candidate_far, residual_far),
            "candidate_neighbor_dot": float(np.mean(candidate_far * residual_far)),
            "candidate_neighbor_endpoint_product": float(candidate_far[-1] * residual_far[-1]),
        }
        for key in (
            "candidate_vs_residual_rmse", "candidate_vs_undo_rmse",
            "candidate_neighbor_corr", "candidate_neighbor_cosine",
            "candidate_neighbor_dot", "candidate_neighbor_endpoint_product",
        ):
            record[f"conf_x_{key}"] = float(geometry_confidence * record[key])
        for column, value in geometry.items():
            record[f"geometry_{column}"] = value
        records.append(record)

audit = pd.DataFrame(records)
expected = features[["well", "fold", "state", "raw_gain_mse"]].copy()
audit = audit.merge(expected, on=["well", "fold", "state"], how="inner", validate="one_to_one")
gain_gap = np.max(np.abs(audit["target_gain_mse"] - audit["raw_gain_mse"]))
if gain_gap > 1e-3:
    raise RuntimeError(f"Candidate gain mismatch versus V44: {gain_gap}")

candidate_feature_names = [
    column for column in audit.columns
    if column.startswith("candidate_") or column.startswith("conf_x_")
]
association_rows = []
for feature in candidate_feature_names:
    row = {"feature": feature, "rho_all": _safe_spearman(audit[feature], audit["raw_gain_mse"])}
    fold_rhos = []
    within_well = []
    for fold in FOLDS:
        local = audit[audit["fold"].eq(fold)]
        rho = _safe_spearman(local[feature], local["raw_gain_mse"])
        row[f"rho_fold{fold}"] = rho
        fold_rhos.append(rho)
        for _, group in local.groupby("well", sort=False):
            within_well.append(_safe_spearman(group[feature], group["raw_gain_mse"]))
    nonzero = [np.sign(value) for value in fold_rhos if abs(value) >= 0.02]
    row["same_sign_folds"] = int(len(nonzero) == len(FOLDS) and len(set(nonzero)) == 1)
    row["minimum_abs_fold_rho"] = float(min(abs(value) for value in fold_rhos))
    row["mean_within_well_rho"] = float(np.mean(within_well))
    association_rows.append(row)
associations = pd.DataFrame(association_rows).sort_values(
    ["same_sign_folds", "minimum_abs_fold_rho", "mean_within_well_rho"],
    ascending=False,
).reset_index(drop=True)


def _crossfit_feature(feature, shuffled=False):
    prediction = base.copy()
    choices = []
    rng = np.random.default_rng(SEED + int(shuffled))
    for held_fold in FOLDS:
        train = audit[~audit["fold"].eq(held_fold)]
        held = audit[audit["fold"].eq(held_fold)].copy()
        orientation = float(np.sign(_safe_spearman(train[feature], train["raw_gain_mse"])))
        if orientation == 0.0:
            orientation = 1.0
        held["score"] = orientation * held[feature].to_numpy(np.float64)
        if shuffled:
            values = held["score"].to_numpy(np.float64).copy()
            for _, positions in held.groupby("well", sort=False).indices.items():
                positions = np.asarray(positions, np.int64)
                values[positions] = values[rng.permutation(positions)]
            held["score"] = values
        selected = held.loc[held.groupby("well", sort=False)["score"].idxmax()]
        for row in selected.itertuples(index=False):
            indices = groups[str(row.well)]
            prediction[indices] = candidates[indices, int(row.state)]
            choices.append({
                "well": str(row.well), "fold": int(held_fold), "state": int(row.state),
                "candidate": str(row.candidate), "feature": feature,
                "orientation": orientation, "shuffled": bool(shuffled),
            })
    return prediction, pd.DataFrame(choices)


base_metrics = _metrics(frame, base, "base_6p377")
screen_rows = []
choice_parts = []
for feature in associations.head(min(12, len(associations)))["feature"]:
    prediction, choices = _crossfit_feature(feature, shuffled=False)
    shuffled_prediction, shuffled_choices = _crossfit_feature(feature, shuffled=True)
    metrics = _metrics(frame, prediction, feature)
    shuffled_metrics = _metrics(frame, shuffled_prediction, f"{feature}__shuffle")
    screen_rows.append({
        "feature": feature,
        "rmse": metrics["rmse"],
        "gain": base_metrics["rmse"] - metrics["rmse"],
        "far_rmse": metrics["far_gt2000_rmse"],
        "far_gain": base_metrics["far_gt2000_rmse"] - metrics["far_gt2000_rmse"],
        "shuffle_rmse": shuffled_metrics["rmse"],
        "shuffle_loss": shuffled_metrics["rmse"] - metrics["rmse"],
    })
    choice_parts.extend([choices, shuffled_choices])
screen = pd.DataFrame(screen_rows).sort_values("gain", ascending=False).reset_index(drop=True)
best = screen.iloc[0].to_dict()

audit.to_parquet(OUT_DIR / "neighbor_candidate_features.parquet", index=False, compression="zstd")
associations.to_csv(OUT_DIR / "feature_associations.csv", index=False)
screen.to_csv(OUT_DIR / "single_feature_crossfit.csv", index=False)
pd.concat(choice_parts, ignore_index=True).to_csv(OUT_DIR / "single_feature_choices.csv", index=False)

checks = {
    "best_feature_same_sign_all_folds": bool(
        associations.set_index("feature").loc[best["feature"], "same_sign_folds"] == 1
    ),
    "best_crossfit_gain_ge_0p05": bool(best["gain"] >= 0.05),
    "best_far_gain_ge_0p08": bool(best["far_gain"] >= 0.08),
    "shuffle_worse_by_ge_0p03": bool(best["shuffle_loss"] >= 0.03),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "zero_training_neighbor_candidate_identity_observability",
    "sample_unit": "complete physical well candidate",
    "lineage": "V33 current_dose_0p20 / LB 6.377",
    "wells": int(frame["well"].nunique()),
    "states": int(candidates.shape[1]),
    "candidate_names": candidate_names,
    "neighbor_meta_coverage": float(audit.groupby("well")["geometry_available"].first().mean()),
    "candidate_gain_max_gap_vs_v44": float(gain_gap),
    "base": base_metrics,
    "best_single_feature": best,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> add only confirmed spatial interactions to a nested V46 whole-well selector"
        if all(checks.values())
        else "FAIL -> spatial metadata cannot identify the coherent endpoint; close this candidate bank"
    ),
}
_write_json(OUT_DIR / "result.json", verdict)

print("\nV45 TOP ASSOCIATIONS")
print(associations.head(20).to_string(index=False))
print("\nV45 SINGLE-FEATURE CROSSFIT")
print(screen.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT_DIR)
