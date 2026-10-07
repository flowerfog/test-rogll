"""V120: endpoint-conditioned GR residual head after top-33 consensus.

Train on complete Fold1/2 wells, select a conservative correction recipe on
the first 24 Fold3 wells, and confirm once on the remaining Fold3 wells.
All features are legal at inference: deployed endpoints, horizontal GR/Z/MD,
assigned typewell GR, and deterministic transforms thereof.
"""

from __future__ import annotations

import json
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
EXPERIMENT = "final_endpoint_gr_residual_head_v120"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
BASE_PATH = OOF_ROOT / "rawdata_lattice_consensus_center_fold2_replay_v115" / "fold123_consensus_center_oof.parquet"
TRAIN_FOLDS = (1, 2)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
MAX_TRAIN_ROWS_PER_WELL = 2048
FREEZE_FT = 500.0
RAMP_FT = 250.0
SMOOTH_FT = (32.0, 64.0, 128.0, 256.0)
DOSES = (0.25, 0.50, 0.75, 1.00)
CAPS_FT = (1.0, 2.0, 3.0, 4.0)
TRUTH = "tvt_true"
BASE = "consensus_center"
OFFSETS = (-24.0, -12.0, -6.0, 0.0, 6.0, 12.0, 24.0)


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


def _bootstrap_v110():
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("final_endpoint_v120_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _gradient(values, coordinate):
    values = np.asarray(values, np.float64)
    coordinate = np.asarray(coordinate, np.float64)
    if len(values) < 3:
        return np.zeros_like(values)
    return np.gradient(values, coordinate, edge_order=1)


def _fill_log(values, coordinate, label):
    values = np.asarray(values, np.float64)
    coordinate = np.asarray(coordinate, np.float64)
    finite = np.isfinite(values) & np.isfinite(coordinate)
    if finite.sum() < 2:
        raise RuntimeError(f"Only {int(finite.sum())} finite values in {label}")
    output = values.copy()
    output[~finite] = np.interp(coordinate[~finite], coordinate[finite], values[finite])
    return output


def _rolling(values, rows, statistic):
    series = pd.Series(np.asarray(values, np.float64))
    window = max(3, int(rows) | 1)
    rolled = series.rolling(window, center=True, min_periods=1)
    if statistic == "mean":
        return rolled.mean().to_numpy(np.float64)
    if statistic == "std":
        return rolled.std(ddof=0).fillna(0.0).to_numpy(np.float64)
    raise KeyError(statistic)


def _features(local, record):
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{local['well'].iloc[0]}: row/truth alignment")
    full_md = record["md"].astype(np.float64)
    if not np.isfinite(full_md).all():
        raise RuntimeError(f"{local['well'].iloc[0]}: non-finite MD")
    md = full_md[eval_rows]
    z = _fill_log(record["z"], full_md, "horizontal Z")[eval_rows]
    gr = _fill_log(record["gr"], full_md, "horizontal GR")[eval_rows]
    distance = md - float(record["anchor_md"])
    base = local[BASE].to_numpy(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    parent = local["incumbent_6p377_replay"].to_numpy(np.float64)
    coarse = local["coarse_crossfit"].to_numpy(np.float64)
    type_tvt = record["type_tvt"].astype(np.float64)
    type_gr = record["type_gr"].astype(np.float64)
    finite_type = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_frame = pd.DataFrame({"tvt": type_tvt[finite_type], "gr": type_gr[finite_type]})
    type_frame = type_frame.groupby("tvt", as_index=False, sort=True)["gr"].mean()
    if len(type_frame) < 16:
        raise RuntimeError(f"{local['well'].iloc[0]}: only {len(type_frame)} finite typewell levels")
    type_tvt = type_frame["tvt"].to_numpy(np.float64)
    type_gr = type_frame["gr"].to_numpy(np.float64)
    spacing = float(np.median(np.diff(md))) if len(md) > 1 else 1.0
    rows32 = max(3, int(round(32.0 / max(spacing, 1e-3))))
    rows128 = max(3, int(round(128.0 / max(spacing, 1e-3))))

    data = {
        "distance": distance,
        "distance_log": np.log1p(np.maximum(distance, 0.0)),
        "base": base,
        "parent": parent,
        "coarse": coarse,
        "exact_v76": exact,
        "consensus_move": base - exact,
        "coarse_parent_move": coarse - parent,
        "gr": gr,
        "gr_grad": _gradient(gr, md),
        "gr_mean32": _rolling(gr, rows32, "mean"),
        "gr_std32": _rolling(gr, rows32, "std"),
        "gr_mean128": _rolling(gr, rows128, "mean"),
        "gr_std128": _rolling(gr, rows128, "std"),
        "z": z,
        "z_grad": _gradient(z, md),
        "base_grad": _gradient(base, md),
        "base_curvature": _gradient(_gradient(base, md), md),
    }
    for offset in OFFSETS:
        label = f"m{abs(int(offset))}" if offset < 0 else f"p{int(offset)}"
        sampled = np.interp(base + offset, type_tvt, type_gr)
        data[f"type_gr_{label}"] = sampled
        data[f"mismatch_{label}"] = gr - sampled
    type_minus = np.interp(base - 4.0, type_tvt, type_gr)
    type_zero = np.interp(base, type_tvt, type_gr)
    type_plus = np.interp(base + 4.0, type_tvt, type_gr)
    data["type_slope"] = (type_plus - type_minus) / 8.0
    data["type_curvature"] = (type_plus - 2.0 * type_zero + type_minus) / 16.0
    values = pd.DataFrame(data)
    if not np.isfinite(values.to_numpy(np.float64)).all():
        raise RuntimeError(f"{local['well'].iloc[0]}: non-finite feature matrix")
    return values, truth - base, distance


def _smooth_by_well(frame, values, smooth_ft):
    output = np.empty(len(frame), np.float64)
    values = np.asarray(values, np.float64)
    for well, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        distance = frame.iloc[indices]["md_since"].to_numpy(np.float64)
        spacing = float(np.median(np.diff(distance))) if len(distance) > 1 else 1.0
        rows = max(3, int(round(float(smooth_ft) / max(spacing, 1e-3))))
        output[indices] = _rolling(values[indices], rows, "mean")
    return output


def _load_record_relaxed(well):
    original_stride = V["ROW_STRIDE"]
    try:
        # V120 consumes full-resolution logs only; the V4 sampled-node minimum
        # is irrelevant and excludes a few otherwise valid short wells.
        V["ROW_STRIDE"] = 1
        return V["_load_record"](well)
    finally:
        V["ROW_STRIDE"] = original_stride


for required in (V110_SOURCE, BASE_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
R = _bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

frame = pd.read_parquet(BASE_PATH).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
lineage_columns = ("incumbent_6p377_replay", "coarse_crossfit")
missing_lineage = [column for column in lineage_columns if column not in frame]
if missing_lineage:
    lineage = pd.read_parquet(
        R.V70_PATH,
        columns=["id", "fold", TRUTH, *missing_lineage],
        filters=[("fold", "in", [1, 2, 3])],
    )
    if lineage["id"].duplicated().any() or frame["id"].duplicated().any():
        raise RuntimeError("Duplicate ids while enriching the V120 lineage")
    frame = frame.merge(
        lineage,
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_lineage"),
    )
    if frame[missing_lineage].isna().any().any():
        raise RuntimeError("V70 lineage enrichment left missing endpoint values")
    fold_gap = int(np.max(np.abs(frame["fold"].to_numpy(np.int16) - frame["fold_lineage"].to_numpy(np.int16))))
    truth_gap = float(np.max(np.abs(frame[TRUTH].to_numpy(np.float64) - frame[f"{TRUTH}_lineage"].to_numpy(np.float64))))
    if fold_gap != 0 or truth_gap > 0.01:
        raise RuntimeError(f"V70 lineage alignment failed: fold={fold_gap}, truth={truth_gap}")
    frame = frame.drop(columns=["fold_lineage", f"{TRUTH}_lineage"])
    print(
        "V120 LINEAGE ENRICHMENT",
        {"rows": len(frame), "wells": frame["well"].nunique(), "columns": missing_lineage, "truth_gap": truth_gap},
        flush=True,
    )
required_columns = {"id", "well", "row_idx", "fold", TRUTH, "md_since", "incumbent_6p377_replay", "coarse_crossfit", "exact_v76", BASE}
missing = sorted(required_columns.difference(frame.columns))
if missing:
    raise RuntimeError(f"Base OOF missing columns: {missing}")

feature_parts = []
target_parts = []
meta_parts = []
for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    local = local.sort_values("row_idx", kind="stable").copy()
    features, target, distance = _features(local, _load_record_relaxed(well))
    feature_parts.append(features)
    target_parts.append(target)
    meta_parts.append(local.reset_index(drop=True))
    if number % 100 == 0 or number == frame["well"].nunique():
        print(f"V120 features {number}/{frame['well'].nunique()}", flush=True)

features = pd.concat(feature_parts, ignore_index=True)
target = np.concatenate(target_parts)
meta = pd.concat(meta_parts, ignore_index=True)
if not meta["id"].equals(frame["id"]):
    # groupby sorting may differ from the original frame; all downstream arrays use meta order.
    frame = meta.copy()
else:
    frame = meta

train_mask = frame["fold"].isin(TRAIN_FOLDS).to_numpy()
fold3_wells = sorted(frame.loc[frame["fold"].eq(EVAL_FOLD), "well"].unique())
discovery_wells = set(fold3_wells[:DISCOVERY_WELLS])
discovery_mask = frame["well"].isin(discovery_wells).to_numpy()
confirmation_mask = frame["fold"].eq(EVAL_FOLD).to_numpy() & ~discovery_mask

sampled_train = np.zeros(len(frame), bool)
for well, local_train in frame.loc[train_mask].groupby("well", sort=False):
    indices = local_train.index.to_numpy(np.int64)
    step = max(1, int(np.ceil(len(indices) / MAX_TRAIN_ROWS_PER_WELL)))
    sampled_train[indices[::step]] = True
sampled_discovery = discovery_mask & (np.arange(len(frame)) % 4 == 0)

model = lgb.LGBMRegressor(
    objective="huber",
    n_estimators=1200,
    learning_rate=0.025,
    num_leaves=31,
    min_child_samples=160,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=2.0,
    reg_lambda=12.0,
    random_state=20260801,
    verbosity=-1,
)
model.fit(
    features.loc[sampled_train],
    np.clip(target[sampled_train], -24.0, 24.0),
    sample_weight=np.ones(int(sampled_train.sum()), np.float64),
    eval_set=[(features.loc[sampled_discovery], target[sampled_discovery])],
    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)],
)
raw = model.predict(features)
np.save(OUT_DIR / "raw_residual_prediction.npy", raw.astype(np.float32))
model.booster_.save_model(str(OUT_DIR / "residual_head.txt"))

base_prediction = frame[BASE].to_numpy(np.float64)
smoothed = {smooth: _smooth_by_well(frame, raw, smooth) for smooth in SMOOTH_FT}
recipes = [(smooth, dose, cap) for smooth in SMOOTH_FT for dose in DOSES for cap in CAPS_FT]
predictions = {}
scan_rows = []
base_discovery = R._metric(frame.loc[discovery_mask], base_prediction[discovery_mask], "consensus_top33")
for recipe in recipes:
    smooth, dose, cap = recipe
    taper = _smoothstep((frame["md_since"].to_numpy(np.float64) - FREEZE_FT) / RAMP_FT)
    move = taper * np.clip(float(dose) * smoothed[smooth], -float(cap), float(cap))
    prediction = base_prediction + move
    predictions[recipe] = prediction
    metric = R._metric(frame.loc[discovery_mask], prediction[discovery_mask], "residual_head")
    metric.update({
        "smooth_ft": smooth,
        "dose": dose,
        "cap_ft": cap,
        "gain": float(base_discovery["rmse"] - metric["rmse"]),
        "far_gain": float(base_discovery["far_gt2000_rmse"] - metric["far_gt2000_rmse"]),
        "near_degradation": float(metric["near_0_500_rmse"] - base_discovery["near_0_500_rmse"]),
        "p95_gain": float(base_discovery["p95_well_rmse"] - metric["p95_well_rmse"]),
    })
    scan_rows.append(metric)

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.05)
    & (scan["far_gain"] >= 0.0)
    & (scan["near_degradation"] <= 1e-10)
    & (scan["p95_gain"] >= -0.10)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (float(selected_row["smooth_ft"]), float(selected_row["dose"]), float(selected_row["cap_ft"]))
candidate_prediction = predictions[selected]

base_confirmation = R._metric(frame.loc[confirmation_mask], base_prediction[confirmation_mask], "consensus_top33")
candidate_confirmation = R._metric(frame.loc[confirmation_mask], candidate_prediction[confirmation_mask], "residual_head")
rolled = raw.copy()
for _, indices in frame.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    rolled[indices] = np.roll(raw[indices], max(1, len(indices) // 5))
rolled_smooth = _smooth_by_well(frame, rolled, selected[0])
taper = _smoothstep((frame["md_since"].to_numpy(np.float64) - FREEZE_FT) / RAMP_FT)
rolled_prediction = base_prediction + taper * np.clip(selected[1] * rolled_smooth, -selected[2], selected[2])
rolled_confirmation = R._metric(frame.loc[confirmation_mask], rolled_prediction[confirmation_mask], "rolled_control")

values = {
    "discovery_gain": float(base_discovery["rmse"] - R._metric(frame.loc[discovery_mask], candidate_prediction[discovery_mask], "candidate")["rmse"]),
    "confirmation_gain": float(base_confirmation["rmse"] - candidate_confirmation["rmse"]),
    "confirmation_far_gain": float(base_confirmation["far_gt2000_rmse"] - candidate_confirmation["far_gt2000_rmse"]),
    "confirmation_p95_gain": float(base_confirmation["p95_well_rmse"] - candidate_confirmation["p95_well_rmse"]),
    "aligned_increment_vs_rolled": float(rolled_confirmation["rmse"] - candidate_confirmation["rmse"]),
}
checks = {
    "eligible_recipe_exists": bool(len(eligible)),
    "confirmation_gain_ge_0p05": values["confirmation_gain"] >= 0.05,
    "confirmation_far_gain_nonnegative": values["confirmation_far_gain"] >= 0.0,
    "confirmation_p95_not_worse_by_gt_0p10": values["confirmation_p95_gain"] >= -0.10,
    "aligned_beats_rolled_ge_0p02": values["aligned_increment_vs_rolled"] >= 0.02,
}
strict_pass = bool(all(checks.values()))

summary = pd.DataFrame([base_discovery, R._metric(frame.loc[discovery_mask], candidate_prediction[discovery_mask], "residual_head"), base_confirmation, candidate_confirmation, rolled_confirmation])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
output = frame.loc[frame["fold"].eq(EVAL_FOLD), ["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
output["residual_head"] = candidate_prediction[frame["fold"].eq(EVAL_FOLD).to_numpy()].astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_residual_head_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_training_Fold3_discovery_confirmation",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "train_folds": TRAIN_FOLDS,
    "selection_fold": EVAL_FOLD,
    "discovery_wells": len(discovery_wells),
    "confirmation_wells": int(frame.loc[confirmation_mask, "well"].nunique()),
    "features": list(features.columns),
    "selected_recipe": {"smooth_ft": selected[0], "dose": selected[1], "cap_ft": selected[2], "freeze_ft": FREEZE_FT, "ramp_ft": RAMP_FT},
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> train cross-fitted residual heads and build complete OOF; FAIL -> residual is not observable from endpoint-conditioned GR features",
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("\nV120 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV120 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV120 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
