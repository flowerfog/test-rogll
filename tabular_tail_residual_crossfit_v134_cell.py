"""V134: well-cross-fit tabular replacement for the V127 far-tail trajectory."""

from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "tabular_tail_residual_crossfit_v134"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V127_SOURCE = ROOT / "final_consensus_sequence_coefficient_crossfit_v127_cell.py"
V127_OOF = OOF_ROOT / "final_consensus_sequence_coefficient_crossfit_v127" / "crossfit_oof.parquet"
FOLDS = (1, 2, 3)
SEED = 20260801
MAX_TRAIN_ROWS_PER_WELL = 2048
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 24.0
SMOOTH_WINDOWS_FT = (0.0, 64.0, 128.0, 256.0)
DOSES = (0.25, 0.50, 0.75, 1.00)
OFFSETS_FT = (-48.0, -24.0, -12.0, 0.0, 12.0, 24.0, 48.0)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v127():
    if not V127_SOURCE.is_file():
        raise FileNotFoundError(V127_SOURCE)
    source = V127_SOURCE.read_text(encoding="utf-8")
    marker = "crossfit_prediction = np.full_like(target_raw, np.nan, dtype=np.float32)"
    if marker not in source:
        raise RuntimeError(f"V127 marker missing from {V127_SOURCE}")
    module = types.ModuleType("v134_v127_definitions")
    module.__file__ = str(V127_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V127_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _rolling(values, rows, statistic):
    series = pd.Series(np.asarray(values, np.float64))
    rolling = series.rolling(max(int(rows), 1), center=True, min_periods=1)
    if statistic == "mean":
        return rolling.mean().to_numpy(np.float64)
    if statistic == "std":
        return rolling.std(ddof=0).fillna(0.0).to_numpy(np.float64)
    raise KeyError(statistic)


N = _bootstrap_v127()
Q, R, V = N.Q, N.R, N.V
frame = N.frame.copy()
frame["well"] = frame["well"].astype(str)
if not V127_OOF.is_file():
    raise FileNotFoundError(V127_OOF)
v127 = pd.read_parquet(V127_OOF, columns=["id", "nested_sequence_coefficient"])
frame = frame.merge(v127, on="id", how="left", validate="one_to_one")
if frame["nested_sequence_coefficient"].isna().any():
    raise RuntimeError("Incomplete V127 alignment")
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
well_positions = {
    str(well): np.asarray(positions, np.int64)
    for well, positions in frame.groupby("well", sort=False).indices.items()
}
well_order = sorted(well_positions)
if len(well_order) != 464:
    raise RuntimeError(f"Unexpected well count {len(well_order)}")


def _well_features(well):
    positions = well_positions[well]
    local = frame.iloc[positions]
    record = Q._load_record_relaxed(V, well)
    eval_rows = record["eval_rows"]
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: row mismatch {len(local)}/{len(eval_rows)}")
    truth = record["truth"][eval_rows].astype(np.float64)
    if np.max(np.abs(local[Q.TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: truth mismatch")
    md = record["md"][eval_rows].astype(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    spacing = max(float(np.median(np.diff(md))), 1e-3)
    progress = distance / max(float(distance[-1]), 1.0)
    prediction = local["nested_sequence_coefficient"].to_numpy(np.float64)
    consensus = local[Q.BASE].to_numpy(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    parent = local[Q.PARENT].to_numpy(np.float64)
    coarse = local[Q.COARSE].to_numpy(np.float64)
    full_md = record["md"].astype(np.float64)
    gr = Q._fill(record["gr"], full_md)[eval_rows]
    z = Q._fill(record["z"], full_md)[eval_rows]
    gr_robust = Q._robust(gr)
    type_tvt = np.asarray(record["type_tvt"], np.float64)
    type_gr = np.asarray(record["type_gr"], np.float64)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_frame = pd.DataFrame({"tvt": type_tvt[finite], "gr": type_gr[finite]})
    type_frame = type_frame.groupby("tvt", as_index=False, sort=True)["gr"].mean()
    if len(type_frame) < 16:
        raise RuntimeError(f"{well}: only {len(type_frame)} typewell samples")
    type_tvt = type_frame["tvt"].to_numpy(np.float64)
    type_gr = type_frame["gr"].to_numpy(np.float64)
    type_gradient = np.gradient(type_gr, type_tvt)

    columns = [
        distance / 5000.0,
        progress,
        np.log1p(distance) / math.log1p(max(float(distance[-1]), 1.0)),
        (prediction - prediction[0]) / 100.0,
        np.gradient(prediction, md),
        np.gradient(np.gradient(prediction, md), md) * 1000.0,
        (consensus - prediction) / 8.0,
        (exact - prediction) / 8.0,
        (parent - prediction) / 16.0,
        (coarse - prediction) / 16.0,
        (coarse - parent) / 16.0,
        gr_robust,
        np.gradient(gr_robust, md),
        (z - z[0]) / 100.0,
        np.gradient(z, md),
    ]
    names = [
        "distance", "progress", "log_distance", "prediction_relative",
        "prediction_slope", "prediction_curvature", "consensus_gap", "exact_gap",
        "parent_gap", "coarse_gap", "coarse_parent_gap", "gr_robust", "gr_gradient",
        "z_relative", "z_gradient",
    ]
    for window_ft in (32.0, 128.0, 512.0):
        rows = max(1, int(round(window_ft / spacing)))
        columns.extend([
            _rolling(gr_robust, rows, "mean"),
            _rolling(gr_robust, rows, "std"),
            _rolling(np.gradient(prediction, md), rows, "mean"),
        ])
        label = int(window_ft)
        names.extend([f"gr_mean_{label}", f"gr_std_{label}", f"dip_mean_{label}"])
    noise = max(float(record["noise_scale"]), 5.0)
    for offset in OFFSETS_FT:
        sampled = np.interp(prediction + offset, type_tvt, type_gr)
        derivative = np.interp(prediction + offset, type_tvt, type_gradient)
        columns.extend([(gr - sampled) / noise, sampled / 100.0, derivative])
        label = str(int(offset)).replace("-", "m")
        names.extend([f"mismatch_{label}", f"typegr_{label}", f"typegrad_{label}"])
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{well}: non-finite feature matrix")
    target = np.clip(truth - prediction, -48.0, 48.0).astype(np.float32)
    return matrix, target, names


features_by_well, target_by_well = {}, {}
feature_names = None
for number, well in enumerate(well_order, 1):
    features, target, names = _well_features(well)
    if feature_names is None:
        feature_names = names
    elif names != feature_names:
        raise RuntimeError(f"{well}: feature schema mismatch")
    features_by_well[well] = features
    target_by_well[well] = target
    if number % 50 == 0 or number == len(well_order):
        print(f"V134 features {number}/{len(well_order)} elapsed_min={(time.time()-STARTED)/60:.1f}", flush=True)

fold_map = {well: int(frame.iloc[well_positions[well]]["fold"].iloc[0]) for well in well_order}
raw_residual = np.full(len(frame), np.nan, np.float32)
model_reports = []
for heldout_fold in FOLDS:
    x_parts, y_parts, weight_parts = [], [], []
    for well in well_order:
        if fold_map[well] == heldout_fold:
            continue
        local = frame.iloc[well_positions[well]]
        eligible = np.flatnonzero(local["md_since"].to_numpy(np.float64) > 750.0)
        if len(eligible) > MAX_TRAIN_ROWS_PER_WELL:
            rng = np.random.default_rng(SEED + heldout_fold * 1009 + int(well[:6], 16))
            chosen = np.sort(rng.choice(eligible, MAX_TRAIN_ROWS_PER_WELL, replace=False))
        else:
            chosen = eligible
        x_parts.append(features_by_well[well][chosen])
        y_parts.append(target_by_well[well][chosen])
        weight_parts.append(np.full(len(chosen), len(eligible) / max(len(chosen), 1), np.float32))
    x_train = np.concatenate(x_parts)
    y_train = np.concatenate(y_parts)
    sample_weight = np.concatenate(weight_parts)
    sample_weight /= np.mean(sample_weight)
    model = lgb.LGBMRegressor(
        objective="regression_l2", n_estimators=500, learning_rate=0.03,
        num_leaves=31, max_depth=-1, min_child_samples=240,
        subsample=0.80, colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=12.0,
        random_state=SEED + heldout_fold, n_jobs=8, verbosity=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    model.booster_.save_model(str(MODEL_DIR / f"heldout_fold{heldout_fold}.txt"))
    heldout_wells = [well for well in well_order if fold_map[well] == heldout_fold]
    for well in heldout_wells:
        raw_residual[well_positions[well]] = model.predict(features_by_well[well]).astype(np.float32)
    report = {
        "heldout_fold": heldout_fold, "train_rows": len(x_train),
        "train_wells": len(well_order) - len(heldout_wells), "valid_wells": len(heldout_wells),
        "features": len(feature_names),
    }
    model_reports.append(report)
    print("V134 MODEL", report, flush=True)
    del x_train, y_train, sample_weight, x_parts, y_parts, weight_parts
if not np.isfinite(raw_residual).all():
    raise RuntimeError("Incomplete V134 residual OOF")

smoothed = {}
for window_ft in SMOOTH_WINDOWS_FT:
    values = np.empty(len(frame), np.float64)
    for well in well_order:
        positions = well_positions[well]
        local = frame.iloc[positions]
        spacing = max(float(np.median(np.diff(local["md_since"].to_numpy(np.float64)))), 1e-3)
        rows = 1 if window_ft <= 0 else max(1, int(round(window_ft / spacing)))
        values[positions] = _rolling(raw_residual[positions], rows, "mean")
    smoothed[window_ft] = values

base_prediction = frame["nested_sequence_coefficient"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
recipes = [(window, dose) for window in SMOOTH_WINDOWS_FT for dose in DOSES]
predictions = {
    recipe: base_prediction + taper * np.clip(float(recipe[1]) * smoothed[recipe[0]], -CAP_FT, CAP_FT)
    for recipe in recipes
}

selected_by_fold = {}
scan_rows = []
for heldout_fold in FOLDS:
    selection_folds = [fold for fold in FOLDS if fold != heldout_fold]
    selection_mask = frame["fold"].isin(selection_folds).to_numpy()
    selection_frame = frame.loc[selection_mask]
    base = R._metric(selection_frame, base_prediction[selection_mask], "v127")
    candidates = []
    for recipe in recipes:
        result = R._metric(selection_frame, predictions[recipe][selection_mask], "tabular_tail")
        fold_gains = []
        for fold in selection_folds:
            mask = frame["fold"].eq(fold).to_numpy()
            fold_base = R._metric(frame.loc[mask], base_prediction[mask], "v127")
            fold_candidate = R._metric(frame.loc[mask], predictions[recipe][mask], "tabular_tail")
            fold_gains.append(float(fold_base["rmse"] - fold_candidate["rmse"]))
        row = {
            "heldout_fold": heldout_fold, "window_ft": recipe[0], "dose": recipe[1],
            "rmse": result["rmse"], "gain": float(base["rmse"] - result["rmse"]),
            "far_gain": float(base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
            "p95_gain": float(base["p95_well_rmse"] - result["p95_well_rmse"]),
            "min_fold_gain": min(fold_gains),
        }
        candidates.append(row)
        scan_rows.append(row)
    candidate_frame = pd.DataFrame(candidates).sort_values("rmse", kind="stable")
    eligible = candidate_frame[
        (candidate_frame["gain"] >= 0.03) & (candidate_frame["far_gain"] >= 0.0)
        & (candidate_frame["p95_gain"] >= -0.10) & (candidate_frame["min_fold_gain"] >= 0.0)
    ]
    selected = (eligible if len(eligible) else candidate_frame).iloc[0]
    selected_by_fold[heldout_fold] = {
        "recipe": (float(selected["window_ft"]), float(selected["dose"])),
        "selection_pass": bool(len(eligible)), "selection_gain": float(selected["gain"]),
    }
pd.DataFrame(scan_rows).to_csv(OUT_DIR / "nested_recipe_scan.csv", index=False)

nested_prediction = np.empty(len(frame), np.float64)
opposite_prediction = np.empty(len(frame), np.float64)
for fold in FOLDS:
    mask = frame["fold"].eq(fold).to_numpy()
    recipe = selected_by_fold[fold]["recipe"]
    move = predictions[recipe][mask] - base_prediction[mask]
    nested_prediction[mask] = base_prediction[mask] + move
    opposite_prediction[mask] = base_prediction[mask] - move

summary_rows = []
fold_values = {}
for fold in (*FOLDS, 0):
    mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].eq(fold).to_numpy()
    local = frame.loc[mask]
    base = R._metric(local, base_prediction[mask], "v127")
    candidate = R._metric(local, nested_prediction[mask], "tabular_tail")
    opposite = R._metric(local, opposite_prediction[mask], "opposite_move_control")
    summary_rows.extend([
        {"fold": fold, **base}, {"fold": fold, **candidate}, {"fold": fold, **opposite},
    ])
    if fold:
        fold_values[str(fold)] = {
            "recipe": selected_by_fold[fold]["recipe"],
            "selection_pass": selected_by_fold[fold]["selection_pass"],
            "gain": float(base["rmse"] - candidate["rmse"]),
            "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        }
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
overall = summary[summary["fold"].eq(0)].set_index("endpoint")
base, candidate, opposite = overall.loc["v127"], overall.loc["tabular_tail"], overall.loc["opposite_move_control"]
values = {
    "gain": float(base["rmse"] - candidate["rmse"]),
    "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
    "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
    "aligned_increment_vs_opposite": float(opposite["rmse"] - candidate["rmse"]),
}
checks = {
    "gain_ge_0p10": values["gain"] >= 0.10,
    "positive_folds_3of3": all(value["gain"] > 0.0 for value in fold_values.values()),
    "far_gain_positive": values["far_gain"] > 0.0,
    "p95_not_worse_by_gt_0p10": values["p95_gain"] >= -0.10,
    "aligned_beats_opposite_ge_0p05": values["aligned_increment_vs_opposite"] >= 0.05,
    "all_nested_selections_pass": all(value["selection_pass"] for value in fold_values.values()),
}
strict_pass = bool(all(checks.values()))
output = frame[["id", "well", "row_idx", "fold", Q.TRUTH, "md_since", "nested_sequence_coefficient"]].copy()
output["tabular_tail"] = nested_prediction.astype(np.float32)
output["raw_tabular_residual"] = raw_residual
output.to_parquet(OUT_DIR / "crossfit_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold123_well_crossfit_tabular_tail_replacement",
    "base": "exact V127",
    "model": "LightGBM row-level residual from raw GR/Z, endpoint lineage, and typewell mismatch",
    "prefix_freeze_ft": FREEZE_FT,
    "coverage": {"rows": len(frame), "wells": int(frame["well"].nunique())},
    "model_reports": model_reports,
    "nested_recipes": selected_by_fold,
    "fold_values": fold_values,
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train on all 464 wells and build a test inference bundle"
        if strict_pass else
        "FAIL -> local tabular evidence cannot repair catastrophic tails; standalone trajectory replacement is required"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV134 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV134 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
