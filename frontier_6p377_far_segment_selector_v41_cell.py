"""Nested legal selector for V40's 256-ft far-segment endpoint bank.

The model scores (well, segment, candidate) records.  Candidate-specific legal
features include endpoint disagreement and prefix-calibrated assigned-typewell
GR likelihood evaluated along that candidate path.  The first 2000 ft remains
bitwise equal to the exact V33/6.377 lineage.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_far_segment_selector_v41"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
SEGMENT_FT = 256.0
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
FOLDS = (1, 2, 3)
THRESHOLDS = (-0.05, 0.0, 0.025, 0.05, 0.10, 0.20, 0.35)
DOSES = (0.25, 0.50, 0.75, 1.00)
SEED = 20260727
started = time.time()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _find_data_root():
    for candidate in (
        ROOT / "rogii-wellbore-geology-prediction",
        ROOT,
        Path.cwd(),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ):
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate original competition train directory")


DATA_ROOT = _find_data_root()


def _smoothstep(distance):
    t = np.clip((np.asarray(distance, np.float64) - FAR_START_FT) / FADE_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _robust_scale(values, floor=5.0):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = float(np.median(values))
    return float(max(1.4826 * np.median(np.abs(values - centre)), floor))


def _robust_affine(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, np.float64)[valid]
    y = np.asarray(y, np.float64)[valid]
    if len(x) < 24 or np.std(x) < 1e-6:
        shift = float(np.median(y - x)) if len(x) else 0.0
        return 1.0, shift, _robust_scale(y - x - shift)
    design = np.column_stack((x, np.ones(len(x))))
    weight = np.ones(len(x), np.float64)
    beta = np.array([1.0, np.median(y - x)], np.float64)
    for _ in range(8):
        lhs = design.T @ (weight[:, None] * design) + np.diag((1e-3, 1e-6))
        rhs = design.T @ (weight * y)
        beta = np.linalg.solve(lhs, rhs)
        beta[0] = np.clip(beta[0], 0.50, 1.50)
        residual = y - design @ beta
        scale = _robust_scale(residual, 3.0)
        ratio = np.abs(residual) / max(1.5 * scale, 1e-6)
        weight = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1e-6))
    residual = y - design @ beta
    return float(beta[0]), float(beta[1]), _robust_scale(residual, 5.0)


def _deduplicate(axis, values):
    table = pd.DataFrame({"axis": axis, "value": values}).replace([np.inf, -np.inf], np.nan).dropna()
    table = table.groupby("axis", as_index=False)["value"].median().sort_values("axis")
    if len(table) < 32:
        raise RuntimeError("too few finite typewell TVT rows")
    return table["axis"].to_numpy(np.float64), table["value"].to_numpy(np.float64)


def _centred_mean(md, values, width_ft):
    md = np.asarray(md, np.float64)
    values = np.asarray(values, np.float64)
    half = 0.5 * float(width_ft)
    left = np.searchsorted(md, md - half, side="left")
    right = np.searchsorted(md, md + half, side="right")
    csum = np.concatenate(([0.0], np.cumsum(values)))
    return (csum[right] - csum[left]) / np.maximum(right - left, 1)


def _slope(values):
    values = np.asarray(values, np.float64)
    if len(values) < 2:
        return 0.0
    x = np.linspace(-1.0, 1.0, len(values))
    return float(np.dot(x, values - np.mean(values)) / max(np.dot(x, x), 1e-9))


def _metrics(local, prediction, endpoint):
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    per_well = pd.Series(error**2).groupby(local["well"].astype(str).to_numpy(), sort=False).mean().pow(0.5)
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


for required in (V33_PATH, V28_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v33_columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", "pre_neighbor_cp8", BASE_COLUMN]
frame = pd.read_parquet(V33_PATH, columns=v33_columns)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
tail_columns = sorted(column for column in pq.ParquetFile(V28_PATH).schema.names if column.startswith("tail_"))
tail = pd.read_parquet(V28_PATH, columns=["id"] + tail_columns)
tail["id"] = tail["id"].astype(str)
frame = frame.merge(tail, on="id", how="left", validate="one_to_one")

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fade = _smoothstep(distance)
prefix = distance <= FAR_START_FT
candidate_names = ["base_6p377", "pre_neighbor_cp8"] + tail_columns
candidate_paths = [base]
usable_names = [candidate_names[0]]
for name in candidate_names[1:]:
    absolute = frame[name].to_numpy(np.float64)
    if np.isfinite(absolute).mean() < 0.95:
        continue
    correction = np.where(np.isfinite(absolute), np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[prefix] = base[prefix]
    candidate_paths.append(candidate)
    usable_names.append(name)
candidates = np.column_stack(candidate_paths)
state_count = candidates.shape[1]
if state_count < 3 or not np.isfinite(candidates).all():
    raise RuntimeError(f"Invalid candidate bank: {usable_names}")

groups = {str(well): np.asarray(indices, np.int64) for well, indices in frame.groupby("well", sort=False).indices.items()}
records = []
segment_rows = {}
failures = []
for number, (well, indices) in enumerate(groups.items(), 1):
    try:
        local = frame.loc[indices]
        row_idx = local["row_idx"].to_numpy(np.int64)
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=lambda c: c in {"GR", "TVT_input"},
        ).apply(pd.to_numeric, errors="coerce")
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__typewell.csv",
            usecols=lambda c: c in {"TVT", "GR"},
        ).apply(pd.to_numeric, errors="coerce")
        if row_idx.min() < 0 or row_idx.max() >= len(horizontal):
            raise RuntimeError("row_idx outside horizontal file")
        observed_all = horizontal["GR"].to_numpy(np.float64)
        tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
        tw_tvt, tw_gr_raw = _deduplicate(typewell["TVT"], typewell["GR"])
        known = np.flatnonzero(np.isfinite(observed_all) & np.isfinite(tvt_input))
        expected_known = np.interp(tvt_input[known], tw_tvt, tw_gr_raw, left=np.nan, right=np.nan)
        gain, bias, gamma = _robust_affine(expected_known, observed_all[known])
        tw_gr = gain * tw_gr_raw + bias
        observed = observed_all[row_idx]
        local_distance = distance[indices]
        far = local_distance > FAR_START_FT
        block = np.floor(np.maximum(local_distance - FAR_START_FT, 0.0) / SEGMENT_FT).astype(np.int32)
        median_path = np.median(candidates[indices], axis=1)
        max_distance = float(local_distance[-1])
        for block_id in np.unique(block[far]):
            local_rows = np.flatnonzero(far & (block == block_id))
            global_rows = indices[local_rows]
            segment_key = (well, int(block_id))
            segment_rows[segment_key] = global_rows
            base_error = base[global_rows] - truth[global_rows]
            base_mse = float(np.mean(base_error**2))
            path_spread = np.std(candidates[global_rows] - base[global_rows, None], axis=1)
            common = {
                "well": well,
                "fold": int(local["fold"].iloc[0]),
                "block": int(block_id),
                "segment_rows": int(len(global_rows)),
                "segment_start": float(distance[global_rows[0]]),
                "segment_end": float(distance[global_rows[-1]]),
                "segment_fraction": float(distance[global_rows].mean() / max(max_distance, 1.0)),
                "max_md_since": max_distance,
                "eval_gr_coverage": float(np.isfinite(observed[local_rows]).mean()),
                "path_spread_mean": float(np.mean(path_spread)),
                "path_spread_max": float(np.max(path_spread)),
                "base_path_slope": _slope(base[global_rows]),
                "base_path_std": float(np.std(base[global_rows])),
                "prefix_gain": gain,
                "prefix_bias": bias,
                "prefix_gamma": gamma,
            }
            for state, name in enumerate(usable_names):
                path = candidates[global_rows, state]
                correction = path - base[global_rows]
                consensus_delta = path - median_path[local_rows]
                expected = np.interp(path, tw_tvt, tw_gr, left=np.nan, right=np.nan)
                valid = np.isfinite(expected) & np.isfinite(observed[local_rows])
                emission_coverage = float(valid.mean())
                if valid.sum() >= 8:
                    residual = (observed[local_rows][valid] - expected[valid]) / max(gamma, 1e-6)
                    centred_observed = observed[local_rows][valid] - np.median(observed[local_rows][valid])
                    centred_expected = expected[valid] - np.median(expected[valid])
                    denominator = np.sqrt(np.dot(centred_observed, centred_observed) * np.dot(centred_expected, centred_expected))
                    emission_ncc = float(np.dot(centred_observed, centred_expected) / denominator) if denominator > 1e-9 else 0.0
                    emission_cauchy = float(np.mean(np.log1p(np.minimum(residual**2, 1e4))))
                    emission_bias = float(np.mean(residual))
                    emission_std = float(np.std(residual))
                else:
                    emission_ncc, emission_cauchy, emission_bias, emission_std = 0.0, 20.0, 0.0, 10.0
                candidate_error = path - truth[global_rows]
                candidate_mse = float(np.mean(candidate_error**2))
                records.append({
                    **common,
                    "state": state,
                    "is_base": float(state == 0),
                    **{f"is_state_{slot}": float(state == slot) for slot in range(state_count)},
                    "correction_mean": float(np.mean(correction)),
                    "correction_rms": float(np.sqrt(np.mean(correction**2))),
                    "correction_std": float(np.std(correction)),
                    "correction_p95abs": float(np.quantile(np.abs(correction), 0.95)),
                    "correction_start": float(correction[0]),
                    "correction_end": float(correction[-1]),
                    "correction_slope": _slope(correction),
                    "correction_tv": float(np.sum(np.abs(np.diff(correction)))) if len(correction) > 1 else 0.0,
                    "consensus_mean": float(np.mean(consensus_delta)),
                    "consensus_rms": float(np.sqrt(np.mean(consensus_delta**2))),
                    "consensus_std": float(np.std(consensus_delta)),
                    "emission_coverage": emission_coverage,
                    "emission_ncc": emission_ncc,
                    "emission_cauchy": emission_cauchy,
                    "emission_bias": emission_bias,
                    "emission_std": emission_std,
                    "target_gain": float(np.arcsinh((base_mse - candidate_mse) / 25.0)),
                    "raw_gain_mse": base_mse - candidate_mse,
                })
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(groups):
        print(f"V41 features {number}/{len(groups)} failures={len(failures)}", flush=True)

features = pd.DataFrame(records)
pd.DataFrame(failures).to_csv(OUT_DIR / "feature_failures.csv", index=False)
if features["well"].nunique() < 455:
    raise RuntimeError(f"Only {features['well'].nunique()} feature wells: {failures[:5]}")
excluded = {"well", "fold", "block", "state", "target_gain", "raw_gain_mse"}
feature_names = [column for column in features.columns if column not in excluded]
if features[feature_names].isna().any().any():
    raise RuntimeError("NaN in V41 feature matrix")
features.to_parquet(OUT_DIR / "segment_candidate_features.parquet", index=False, compression="zstd")


def _fit_model(train):
    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=500, learning_rate=0.025,
        num_leaves=15, max_depth=5, min_child_samples=48,
        subsample=0.85, colsample_bytree=0.75,
        reg_alpha=1.0, reg_lambda=10.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    weight = np.sqrt(np.maximum(train["segment_rows"].to_numpy(np.float64), 1.0))
    model.fit(train[feature_names], train["target_gain"], sample_weight=weight)
    return model


def _predict(model, local, shuffled=False, seed=SEED):
    matrix = np.array(local[feature_names].to_numpy(np.float64), copy=True)
    if shuffled:
        rng = np.random.default_rng(seed)
        keys = local[["well", "block"]].astype(str).agg("|".join, axis=1).to_numpy()
        for key in np.unique(keys):
            positions = np.flatnonzero(keys == key)
            if len(positions) > 1:
                matrix[positions] = matrix[rng.permutation(positions)]
    return model.predict(matrix)


def _assemble(score_frame, threshold, dose, wells):
    prediction = base.copy()
    choices = []
    score_lookup = {(str(row.well), int(row.block), int(row.state)): float(row.score) for row in score_frame.itertuples()}
    for well in wells:
        indices = groups[well]
        local_distance = distance[indices]
        far = local_distance > FAR_START_FT
        block = np.floor(np.maximum(local_distance - FAR_START_FT, 0.0) / SEGMENT_FT).astype(np.int32)
        selected = base[indices].copy()
        for block_id in np.unique(block[far]):
            rows = np.flatnonzero(far & (block == block_id))
            scores = np.array([score_lookup[(well, int(block_id), state)] for state in range(state_count)])
            best = int(np.argmax(scores[1:]) + 1)
            margin = float(scores[best] - scores[0])
            state = best if margin >= float(threshold) else 0
            selected[rows] = candidates[indices[rows], state]
            choices.append({"well": well, "block": int(block_id), "state": state, "candidate": usable_names[state], "margin": margin})
        correction = _centred_mean(local_distance, selected - base[indices], SMOOTH_FT)
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        local_prediction = base[indices] + float(dose) * fade[indices] * correction
        local_prediction[local_distance <= FAR_START_FT] = base[indices][local_distance <= FAR_START_FT]
        prediction[indices] = local_prediction
    return prediction, pd.DataFrame(choices)


def _select_config(score_frame, folds):
    wells = sorted(features.loc[features["fold"].isin(folds), "well"].unique())
    mask = frame["fold"].isin(folds).to_numpy()
    local = frame.loc[mask].reset_index(drop=True)
    base_metric = _metrics(local, base[mask], "base")
    rows = []
    for threshold in THRESHOLDS:
        for dose in DOSES:
            prediction, choices = _assemble(score_frame, threshold, dose, wells)
            result = _metrics(local, prediction[mask], "candidate")
            rows.append({
                "threshold": threshold, "dose": dose,
                "rmse": result["rmse"], "gain": base_metric["rmse"] - result["rmse"],
                "far_gain": base_metric["far_gt2000_rmse"] - result["far_gt2000_rmse"],
                "p95_change": result["p95_well_rmse"] - base_metric["p95_well_rmse"],
                "switch_rate": float((choices["state"] != 0).mean()),
            })
    scan = pd.DataFrame(rows)
    safe = scan[scan["p95_change"] <= 0.15]
    selected = (safe if len(safe) else scan).sort_values(["rmse", "far_gain"], ascending=[True, False]).iloc[0]
    return selected, scan


selector_prediction = base.copy()
shuffle_prediction = base.copy()
fold_reports = []
choice_tables = []
scan_tables = []
for held_fold in FOLDS:
    train_folds = [fold for fold in FOLDS if fold != held_fold]
    inner_scores = []
    for inner_held in train_folds:
        inner_train = features[features["fold"].eq(next(f for f in train_folds if f != inner_held))]
        inner_valid = features[features["fold"].eq(inner_held)].copy()
        inner_model = _fit_model(inner_train)
        inner_valid["score"] = _predict(inner_model, inner_valid)
        inner_scores.append(inner_valid[["well", "block", "state", "score"]])
    inner_scores = pd.concat(inner_scores, ignore_index=True)
    selected, scan = _select_config(inner_scores, train_folds)
    scan["held_fold"] = held_fold
    scan_tables.append(scan)

    model = _fit_model(features[features["fold"].isin(train_folds)])
    model.booster_.save_model(str(MODEL_DIR / f"fold{held_fold}.txt"))
    held = features[features["fold"].eq(held_fold)].copy()
    held["score"] = _predict(model, held)
    shuffled = held[["well", "block", "state"]].copy()
    shuffled["score"] = _predict(model, held, shuffled=True, seed=SEED + held_fold)
    held_wells = sorted(held["well"].unique())
    held_prediction, held_choices = _assemble(held, selected["threshold"], selected["dose"], held_wells)
    shuffled_path, _ = _assemble(shuffled, selected["threshold"], selected["dose"], held_wells)
    row_mask = frame["fold"].to_numpy(np.int8) == held_fold
    selector_prediction[row_mask] = held_prediction[row_mask]
    shuffle_prediction[row_mask] = shuffled_path[row_mask]
    local = frame.loc[row_mask].reset_index(drop=True)
    base_metric = _metrics(local, base[row_mask], "base")
    result = _metrics(local, selector_prediction[row_mask], "selector")
    shuffle_result = _metrics(local, shuffle_prediction[row_mask], "shuffle")
    held_choices["fold"] = held_fold
    choice_tables.append(held_choices)
    fold_reports.append({
        "fold": held_fold,
        "threshold": float(selected["threshold"]), "dose": float(selected["dose"]),
        "inner_rmse": float(selected["rmse"]), "switch_rate": float((held_choices["state"] != 0).mean()),
        "base_rmse": base_metric["rmse"], "rmse": result["rmse"],
        "gain": base_metric["rmse"] - result["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - result["far_gt2000_rmse"],
        "p95_change": result["p95_well_rmse"] - base_metric["p95_well_rmse"],
        "shuffle_rmse": shuffle_result["rmse"],
        "shuffle_loss": shuffle_result["rmse"] - result["rmse"],
    })
    print("V41 FOLD", fold_reports[-1], flush=True)

base_result = _metrics(frame, base, "base_6p377")
selector_result = _metrics(frame, selector_prediction, "nested_selector")
shuffle_result = _metrics(frame, shuffle_prediction, "candidate_feature_shuffle")
gain = base_result["rmse"] - selector_result["rmse"]
shuffle_loss = shuffle_result["rmse"] - selector_result["rmse"]
fold_table = pd.DataFrame(fold_reports)
checks = {
    "selector_gain_ge_0p10": bool(gain >= 0.10),
    "minimum_fold_gain_ge_minus0p02": bool(fold_table["gain"].min() >= -0.02),
    "positive_folds_ge_2": bool((fold_table["gain"] > 0).sum() >= 2),
    "far_gain_ge_0p15": bool(base_result["far_gt2000_rmse"] - selector_result["far_gt2000_rmse"] >= 0.15),
    "shuffle_loses_70pct_gain": bool(shuffle_loss >= 0.70 * max(gain, 0.0)),
    "maximum_fold_p95_change_le_0p15": bool(fold_table["p95_change"].max() <= 0.15),
}
strict_pass = bool(all(checks.values()))

pd.concat(scan_tables, ignore_index=True).to_csv(OUT_DIR / "inner_scans.csv", index=False)
pd.concat(choice_tables, ignore_index=True).to_csv(OUT_DIR / "segment_choices.csv", index=False)
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    base_6p377=base.astype(np.float32),
    nested_selector=selector_prediction.astype(np.float32),
    candidate_feature_shuffle=shuffle_prediction.astype(np.float32),
).to_parquet(OUT_DIR / "segment_selector_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_legal_256ft_far_segment_selector",
    "sample_unit": "complete-well 256-ft segment candidate",
    "deployable_inputs_only": True,
    "lineage": {"base": str(V33_PATH), "base_column": BASE_COLUMN, "candidate_source": str(V28_PATH)},
    "data_root": str(DATA_ROOT),
    "candidate_names": usable_names,
    "feature_count": len(feature_names),
    "feature_wells": int(features["well"].nunique()),
    "feature_rows": int(len(features)),
    "base": base_result,
    "nested_selector": selector_result,
    "candidate_feature_shuffle": shuffle_result,
    "gain": gain,
    "far_gain": base_result["far_gt2000_rmse"] - selector_result["far_gt2000_rmse"],
    "shuffle_loss": shuffle_loss,
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": "PASS -> refit/package the far segment router" if strict_pass else "FAIL -> legal evidence cannot select the V40 oracle; close existing-endpoint routing",
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV41 SUMMARY")
print(pd.DataFrame([base_result, selector_result, shuffle_result]).to_string(index=False), flush=True)
print("\nV41 FOLDS")
print(fold_table.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
