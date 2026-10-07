"""Route 5 Phase 0: treat the complete drilling trajectory as a sensor.

This is a complete-well, five-fold gateway.  It predicts hidden TVT change
directly from the visible surface prefix and complete MD/X/Y/Z trajectory.
Predicting S = TVT + Z would create a tautological path from future Z into the
target followed by subtracting Z at decode time; this residualised target is
the corrected V77b attribution.  Evaluation GR and historical predictions are
not model inputs.

The decisive attribution is matched: a prefix-only model is compared with the
same estimator using future trajectory features, and those future features are
then shuffled inside suffix-length bins at validation time.  A useful endpoint
must improve over prefix-only and lose that improvement under the shuffle.

Run as one AutoDL notebook cell or as a script.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "drilling_trajectory_sensor_gateway_v77b"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

V66_PATH = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
DATA_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    ROOT / "replay_data",
    ROOT / "data",
)
DATA_ROOT = next((path for path in DATA_CANDIDATES if (path / "train").is_dir()), None)

EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
QUERY_ROWS_PER_WELL = 192
PREFIX_NODES = 32
FUTURE_NODES = 48
TARGET_NODES = 33
SEED = 20260729
BLEND_WEIGHTS = np.round(np.linspace(0.0, 0.40, 17), 3)

if DATA_ROOT is None:
    raise FileNotFoundError(f"No train directory under {[str(path) for path in DATA_CANDIDATES]}")
for required in (V66_PATH, FOLD_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"{endpoint}: invalid prediction")
    bywell = (
        pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
        .groupby("well", sort=False)["e2"].mean().to_numpy(np.float64)
    )
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(bywell))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def _interp(md: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    valid = np.isfinite(md) & np.isfinite(values)
    if valid.sum() < 4:
        raise RuntimeError("too few finite interpolation points")
    x, y = md[valid], values[valid]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    return np.interp(query, x, y, left=y[0], right=y[-1]).astype(np.float64)


def _slope(md: np.ndarray, values: np.ndarray, edge: float, window: float) -> float:
    valid = np.isfinite(values) & (md >= edge - window) & (md <= edge + 1e-6)
    if valid.sum() < 8:
        return 0.0
    x = md[valid] - edge
    return float(np.polyfit(x, values[valid], 1)[0])


def _normalise_shape(values: np.ndarray, scale: float) -> np.ndarray:
    values = np.asarray(values, np.float64)
    values = values - values[0]
    return values / max(float(scale), 1.0)


def _record(well: str, fold: int) -> dict:
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    md = frame["MD"].to_numpy(np.float64)
    x = frame["X"].to_numpy(np.float64)
    y = frame["Y"].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
    tvt = frame["TVT"].to_numpy(np.float64)
    tvt_input = frame["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) < 64:
        raise RuntimeError(f"only {len(known)} visible rows")
    anchor = int(known[-1])
    if len(md) - anchor < 100:
        raise RuntimeError("short suffix")
    anchor_md = float(md[anchor])
    suffix_length = float(md[-1] - anchor_md)
    if not np.isfinite(suffix_length) or suffix_length < 96.0:
        raise RuntimeError("invalid suffix length")

    surface = tvt + z
    surface_input = tvt_input + z
    prefix_grid = np.linspace(anchor_md - min(768.0, anchor_md - md[0]), anchor_md, PREFIX_NODES)
    future_grid = anchor_md + np.linspace(0.0, suffix_length, FUTURE_NODES)
    target_grid = anchor_md + np.linspace(0.0, suffix_length, TARGET_NODES)

    prefix_surface = _interp(md, surface_input, prefix_grid)
    prefix_z = _interp(md, z, prefix_grid)
    future_x = _interp(md, x, future_grid)
    future_y = _interp(md, y, future_grid)
    future_z = _interp(md, z, future_grid)
    target_tvt = _interp(md, tvt, target_grid)

    progress_scale = max(suffix_length, 256.0)
    prefix_features = np.concatenate([
        _normalise_shape(prefix_surface, 24.0),
        _normalise_shape(prefix_z, 256.0),
        np.asarray([
            _slope(md, surface_input, anchor_md, 128.0),
            _slope(md, surface_input, anchor_md, 256.0),
            _slope(md, surface_input, anchor_md, 512.0),
            suffix_length / 5000.0,
            float(x[anchor]) / 100000.0,
            float(y[anchor]) / 100000.0,
            float(z[anchor]) / 10000.0,
        ], np.float64),
    ])

    dx = future_x - future_x[0]
    dy = future_y - future_y[0]
    dz = future_z - future_z[0]
    d1x = np.gradient(dx, future_grid)
    d1y = np.gradient(dy, future_grid)
    d1z = np.gradient(dz, future_grid)
    d2z = np.gradient(d1z, future_grid)
    direction_change = np.gradient(np.unwrap(np.arctan2(d1y, d1x)), future_grid)
    future_features = np.concatenate([
        dx / progress_scale,
        dy / progress_scale,
        dz / 256.0,
        d1x,
        d1y,
        d1z,
        256.0 * d2z,
        256.0 * direction_change,
        np.asarray([
            np.hypot(dx[-1], dy[-1]) / progress_scale,
            np.sum(np.hypot(np.diff(future_x), np.diff(future_y))) / progress_scale,
            np.std(d1z),
            np.max(np.abs(d2z)) * 256.0,
            np.max(np.abs(direction_change)) * 256.0,
        ], np.float64),
    ])

    # Predict TVT movement itself.  Future Z remains a legal sensor input, but
    # no longer appears algebraically in both the target and decoder.
    target_delta = target_tvt - float(tvt[anchor])
    if not np.isfinite(prefix_features).all() or not np.isfinite(future_features).all():
        raise RuntimeError("non-finite features")
    if not np.isfinite(target_delta).all():
        raise RuntimeError("non-finite target")
    return {
        "well": well,
        "fold": int(fold),
        "md": md,
        "z": z,
        "anchor": anchor,
        "anchor_md": anchor_md,
        "anchor_tvt": float(tvt_input[anchor]),
        "anchor_z": float(z[anchor]),
        "suffix_length": suffix_length,
        "prefix": prefix_features.astype(np.float32),
        "future": future_features.astype(np.float32),
        "target": target_delta.astype(np.float32),
    }


def _model(name: str):
    if name == "ridge_a10":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "ridge_a100":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if name == "extra_leaf4":
        return ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=4, max_features=0.75,
            n_jobs=-1, random_state=SEED,
        )
    if name == "extra_leaf8":
        return ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=8, max_features=0.75,
            n_jobs=-1, random_state=SEED + 1,
        )
    raise KeyError(name)


def _decode(record: dict, row_idx: np.ndarray, node_delta: np.ndarray) -> np.ndarray:
    progress = record["md"][row_idx] - record["anchor_md"]
    node_progress = np.linspace(0.0, record["suffix_length"], TARGET_NODES)
    tvt_delta = np.interp(progress, node_progress, np.asarray(node_delta, np.float64))
    return record["anchor_tvt"] + tvt_delta


def _shuffle_future(matrix: np.ndarray, lengths: np.ndarray, seed: int) -> np.ndarray:
    output = np.asarray(matrix, np.float32).copy()
    bins = pd.qcut(lengths, q=min(5, len(np.unique(lengths))), labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    for value in np.unique(bins):
        positions = np.flatnonzero(np.asarray(bins) == value)
        if len(positions) > 1:
            output[positions] = output[rng.permutation(positions)]
    return output


# Canonical metadata and V66 sampled evaluation panel.
fold_frame = pd.read_csv(FOLD_PATH)
fold_well = next(column for column in ("well", "well_id", "pair_id") if column in fold_frame)
fold_column = next(column for column in ("fold", "canonical_fold") if column in fold_frame)
fold_by_well = dict(zip(fold_frame[fold_well].astype(str), fold_frame[fold_column].astype(int)))

schema = pq.read_schema(V66_PATH).names
prediction_column = next((column for column in (
    "incumbent_6p377_replay", "prediction", "current_dose_0p20", "tvt_pred"
) if column in schema), None)
if prediction_column is None:
    raise RuntimeError(f"Cannot identify V66 prediction from {schema}")
v66 = pd.read_parquet(V66_PATH, columns=[
    "id", "well", "row_idx", "fold", "tvt_true", "md_since", prediction_column,
])
v66["well"] = v66["well"].astype(str)
if len(v66) != EXPECTED_ROWS or v66["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Unexpected V66 coverage: {len(v66)}/{v66['well'].nunique()}")
sample = pd.concat([
    group.iloc[np.unique(np.round(np.linspace(0, len(group) - 1, min(QUERY_ROWS_PER_WELL, len(group)))).astype(int))]
    for _, group in v66.groupby("well", sort=False)
], ignore_index=True).rename(columns={prediction_column: "base"})

records = {}
failures = []
for number, well in enumerate(sorted(sample["well"].unique()), 1):
    try:
        records[well] = _record(well, fold_by_well[well])
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == EXPECTED_WELLS:
        print(f"V77 load {number}/{EXPECTED_WELLS} success={len(records)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
if len(records) < 760:
    raise RuntimeError(f"Only {len(records)} records: {failures[:5]}")

wells = sorted(records)
prefix_x = np.stack([records[well]["prefix"] for well in wells])
future_x = np.stack([records[well]["future"] for well in wells])
full_x = np.column_stack([prefix_x, future_x]).astype(np.float32)
target_y = np.stack([records[well]["target"] for well in wells]).astype(np.float32)
folds = np.asarray([records[well]["fold"] for well in wells], np.int16)
lengths = np.asarray([records[well]["suffix_length"] for well in wells], np.float64)
index_by_well = {well: index for index, well in enumerate(wells)}

MODEL_NAMES = ("ridge_a10", "ridge_a100", "extra_leaf4", "extra_leaf8")
ARMS = {
    "prefix_extra_leaf8": ("extra_leaf8", "prefix"),
    **{f"full_{name}": (name, "full") for name in MODEL_NAMES},
}
node_oof = {arm: np.full((len(wells), TARGET_NODES), np.nan, np.float32) for arm in ARMS}
node_shuffle = {
    arm: np.full((len(wells), TARGET_NODES), np.nan, np.float32)
    for arm, (_, feature_arm) in ARMS.items() if feature_arm == "full"
}
fold_model_rows = []

print("V77 CONFIG", {
    "records": len(records), "sample_rows": len(sample),
    "prefix_features": prefix_x.shape[1], "future_features": future_x.shape[1],
    "target_nodes": TARGET_NODES, "arms": list(ARMS),
}, flush=True)

for held_fold in range(1, 6):
    train = folds != held_fold
    valid = folds == held_fold
    valid_indices = np.flatnonzero(valid)
    shuffled_future = _shuffle_future(future_x[valid], lengths[valid], SEED + held_fold)
    for arm, (model_name, feature_arm) in ARMS.items():
        train_matrix = prefix_x[train] if feature_arm == "prefix" else full_x[train]
        valid_matrix = prefix_x[valid] if feature_arm == "prefix" else full_x[valid]
        model = _model(model_name)
        model.fit(train_matrix, target_y[train])
        node_oof[arm][valid] = model.predict(valid_matrix).astype(np.float32)
        if feature_arm == "full":
            shuffled_matrix = np.column_stack([prefix_x[valid], shuffled_future]).astype(np.float32)
            node_shuffle[arm][valid] = model.predict(shuffled_matrix).astype(np.float32)
        joblib.dump(model, MODEL_DIR / f"{arm}_hold{held_fold}.joblib", compress=3)
        fold_model_rows.append({
            "fold": held_fold, "arm": arm, "train_wells": int(train.sum()),
            "valid_wells": int(valid.sum()),
        })
        print(f"V77 fold={held_fold} arm={arm}", flush=True)

aligned = sample[sample["well"].isin(records)].copy().reset_index(drop=True)
predictions = {arm: np.empty(len(aligned), np.float64) for arm in ARMS}
shuffle_predictions = {arm: np.empty(len(aligned), np.float64) for arm in node_shuffle}
for well, positions in aligned.groupby("well", sort=False).groups.items():
    positions = np.asarray(list(positions), np.int64)
    record = records[well]
    row_idx = aligned.loc[positions, "row_idx"].to_numpy(np.int64)
    wi = index_by_well[well]
    for arm in ARMS:
        predictions[arm][positions] = _decode(record, row_idx, node_oof[arm][wi])
    for arm in node_shuffle:
        shuffle_predictions[arm][positions] = _decode(record, row_idx, node_shuffle[arm][wi])

summary_rows = [{"control": "aligned", **_metric(aligned, aligned["base"], "base_v66")}]
fold_rows = []
correlation_rows = []
blend_rows = []
truth = aligned["tvt_true"].to_numpy(np.float64)
base = aligned["base"].to_numpy(np.float64)
base_error = base - truth
for arm in ARMS:
    summary_rows.append({"control": "aligned", **_metric(aligned, predictions[arm], arm)})
    if arm in shuffle_predictions:
        summary_rows.append({"control": "future_geometry_shuffle", **_metric(aligned, shuffle_predictions[arm], arm)})
    error = predictions[arm] - truth
    correlation_rows.append({
        "arm": arm,
        "raw_error_rho": float(pearsonr(base_error, error).statistic),
        "candidate_rmse": float(np.sqrt(np.mean(error * error))),
    })
    for weight in BLEND_WEIGHTS:
        blend = base + float(weight) * (predictions[arm] - base)
        blend_rows.append({"arm": arm, "weight": float(weight), **_metric(aligned, blend, "blend")})
    for held_fold in range(1, 6):
        mask = aligned["well"].map(fold_by_well).to_numpy(np.int16) == held_fold
        base_metric = _metric(aligned.loc[mask], base[mask], "base")
        prefix_metric = _metric(
            aligned.loc[mask], predictions["prefix_extra_leaf8"][mask], "prefix"
        )
        arm_metric = _metric(aligned.loc[mask], predictions[arm][mask], arm)
        shuffled_metric = (
            _metric(aligned.loc[mask], shuffle_predictions[arm][mask], arm)
            if arm in shuffle_predictions else None
        )
        fold_rows.append({
            "fold": held_fold, "arm": arm,
            "base_rmse": base_metric["rmse"], "rmse": arm_metric["rmse"],
            "gain_vs_base": base_metric["rmse"] - arm_metric["rmse"],
            "prefix_rmse": prefix_metric["rmse"],
            "gain_vs_prefix": prefix_metric["rmse"] - arm_metric["rmse"],
            "shuffle_rmse": shuffled_metric["rmse"] if shuffled_metric else np.nan,
            "shuffle_loss": shuffled_metric["rmse"] - arm_metric["rmse"] if shuffled_metric else np.nan,
        })

summary = pd.DataFrame(summary_rows)
fold_report = pd.DataFrame(fold_rows)
correlations = pd.DataFrame(correlation_rows)
blends = pd.DataFrame(blend_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)
correlations.to_csv(OUT_DIR / "error_correlations.csv", index=False)
blends.to_csv(OUT_DIR / "blend_scan.csv", index=False)
pd.DataFrame(fold_model_rows).to_csv(OUT_DIR / "model_manifest.csv", index=False)

# A model-menu oracle is diagnostic only: one complete endpoint is selected per well.
model_arms = [arm for arm in ARMS if arm.startswith("full_")]
oracle = np.empty(len(aligned), np.float64)
oracle_choice = []
for well, positions in aligned.groupby("well", sort=False).groups.items():
    positions = np.asarray(list(positions), np.int64)
    local_truth = truth[positions]
    losses = [float(np.mean((predictions[arm][positions] - local_truth) ** 2)) for arm in model_arms]
    selected = int(np.argmin(losses))
    oracle[positions] = predictions[model_arms[selected]][positions]
    oracle_choice.append({"well": well, "arm": model_arms[selected], "rmse": float(np.sqrt(losses[selected]))})
pd.DataFrame(oracle_choice).to_csv(OUT_DIR / "model_oracle_choices.csv", index=False)
oracle_metric = _metric(aligned, oracle, "per_well_model_oracle")

aligned_output = aligned[["id", "well", "row_idx", "fold", "tvt_true", "md_since", "base"]].copy()
for arm in ARMS:
    aligned_output[arm] = predictions[arm].astype(np.float32)
aligned_output.to_parquet(OUT_DIR / "trajectory_sensor_sample_oof.parquet", index=False)

aligned_summary = summary[summary["control"].eq("aligned")].set_index("endpoint")
prefix_rmse = float(aligned_summary.loc["prefix_extra_leaf8", "rmse"])
best_arm = min(model_arms, key=lambda arm: float(aligned_summary.loc[arm, "rmse"]))
best_rmse = float(aligned_summary.loc[best_arm, "rmse"])
shuffle_rmse = float(summary[
    summary["control"].eq("future_geometry_shuffle") & summary["endpoint"].eq(best_arm)
]["rmse"].iloc[0])
best_blend = blends.loc[blends["rmse"].idxmin()].to_dict()
best_folds = fold_report[fold_report["arm"].eq(best_arm)]

checks = {
    "best_geometry_rmse_le_12": bool(best_rmse <= 12.0),
    "future_geometry_beats_prefix_by_ge_0p30": bool(prefix_rmse - best_rmse >= 0.30),
    "future_shuffle_loses_ge_0p30": bool(shuffle_rmse - best_rmse >= 0.30),
    "future_geometry_beats_prefix_in_ge_4_folds": bool(
        (best_folds["gain_vs_prefix"] > 0).sum() >= 4
    ),
    "best_fixed_blend_gain_ge_0p05": bool(float(aligned_summary.loc["base_v66", "rmse"]) - float(best_blend["rmse"]) >= 0.05),
    "model_menu_oracle_rmse_le_8": bool(oracle_metric["rmse"] <= 8.0),
}
mechanism_pass = bool(
    checks["best_geometry_rmse_le_12"]
    and checks["future_geometry_beats_prefix_by_ge_0p30"]
    and checks["future_shuffle_loses_ge_0p30"]
    and checks["future_geometry_beats_prefix_in_ge_4_folds"]
)
strict_pass = bool(mechanism_pass and checks["best_fixed_blend_gain_ge_0p05"])
verdict = {
    "experiment": EXPERIMENT,
    "stage": "route5_phase0b_residualised_future_trajectory_inverse_control_gateway",
    "sample_unit": "192 fixed hidden-suffix rows per complete physical well",
    "model_inputs": "visible surface prefix plus complete MD/X/Y/Z trajectory; no eval GR",
    "target": "TVT_delta directly; future Z is not added to the target or subtracted at decode",
    "protocol": "canonical fivefold complete-well OOF",
    "coverage": {"rows": len(aligned), "wells": aligned["well"].nunique(), "records": len(records)},
    "features": {"prefix": prefix_x.shape[1], "future_trajectory": future_x.shape[1]},
    "base": aligned_summary.loc["base_v66"].to_dict(),
    "prefix_control": aligned_summary.loc["prefix_extra_leaf8"].to_dict(),
    "best_arm": best_arm,
    "best_arm_metrics": aligned_summary.loc[best_arm].to_dict(),
    "future_geometry_gain_vs_prefix": prefix_rmse - best_rmse,
    "future_geometry_shuffle_rmse": shuffle_rmse,
    "future_geometry_shuffle_loss": shuffle_rmse - best_rmse,
    "best_fixed_blend": best_blend,
    "model_menu_oracle": oracle_metric,
    "checks": checks,
    "mechanism_pass": mechanism_pass,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train a higher-capacity geometry-only endpoint and export full-row OOF"
        if strict_pass else
        "FAIL -> complete future drilling trajectory is not a stable geological sensor; close Route 5 and move to Route 7"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nV77 SUMMARY")
print(summary.sort_values(["control", "rmse"]).to_string(index=False))
print("\nV77 ERROR CORRELATIONS")
print(correlations.sort_values("candidate_rmse").to_string(index=False))
print("\nV77 FOLDS")
print(fold_report[fold_report["arm"].eq(best_arm)].to_string(index=False))
print("\nV77 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
