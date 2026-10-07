"""Raw candidate cost-profile ablation for the proven no-GR CP8 model.

The exact prior CP table, folds, target, LightGBM parameters, interpolation,
shrink, and cap are reused. Added inputs are fully deployable:

* reliability_only: GR observation/missingness and prefix calibration quality;
* cost_profile: raw MAE/NCC vectors over 13 TVT offsets for six candidates;
* cost_plus_reliability: both families.

The cost_profile arm is the pre-registered primary test. Calibration uses only
known-prefix TVT_input and GR. Eval truth is never used to build a feature.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CP_PATH = OOF_ROOT / "aux_aware_cp8_ablation_v0" / "control_point_table.pkl"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "cp8_raw_cost_profile_ablation_v0"

BASE_COL = "cp12"
CANDIDATES = ["ridge", "selector", "fleongg", "sp45", "strong_bias", "strong"]
OFFSETS = np.array(
    [-24, -16, -12, -8, -4, -2, 0, 2, 4, 8, 12, 16, 24],
    dtype=np.float32,
)
RELIABILITY_SCALES = (48, 192, 768)
WIN_ROWS = 192
N_CP = 8
N_FOLDS = 5
SEED = 20260713
CP_ALPHA = 0.50
CP_CAP = 6.0
EXPECTED_REFERENCE_GAIN = 0.10307155203761553
REFERENCE_TOLERANCE = 0.005
PRIMARY_INCREMENT_GATE = 0.02

OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (DATA_ROOT / "train", CP_PATH, POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _robust_center_scale(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    center = float(np.median(values))
    q10, q90 = np.quantile(values, [0.10, 0.90])
    scale = float((q90 - q10) / 2.563103)
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    return center, scale


def _mean_cost(horizontal, typewell):
    valid = np.isfinite(horizontal) & np.isfinite(typewell)
    if valid.sum() < 4:
        return 0.0
    difference = np.clip(horizontal[valid] - typewell[valid], -6.0, 6.0)
    return float(np.mean(np.abs(difference)))


def _ncc(horizontal, typewell):
    valid = np.isfinite(horizontal) & np.isfinite(typewell)
    if valid.sum() < 8:
        return 0.0
    x = horizontal[valid].astype(float)
    y = typewell[valid].astype(float)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator < 1e-8:
        return 0.0
    return float(np.clip(np.sum(x * y) / denominator, -1.0, 1.0))


def _offset_name(offset):
    value = int(offset)
    if value < 0:
        return f"m{abs(value)}"
    if value > 0:
        return f"p{value}"
    return "z0"


def _nearest_valid_distance(valid):
    valid = np.asarray(valid, bool)
    n = len(valid)
    distance = np.full(n, n + 1, dtype=np.int32)
    last = -(n + 1)
    for index in range(n):
        if valid[index]:
            last = index
        distance[index] = index - last
    last = 2 * n + 1
    for index in range(n - 1, -1, -1):
        if valid[index]:
            last = index
        distance[index] = min(distance[index], last - index)
    return distance.astype(np.float32)


# ---------------------------------------------------------------------------
# Exact no-GR reference table and official OOF candidate trajectories.
# ---------------------------------------------------------------------------
cp = pd.read_pickle(CP_PATH).copy().reset_index(drop=True)
cp["well"] = cp["well"].astype(str)
metadata_columns = {"well", "fold", "center_local", "target_residual"}
base_feature_columns = [
    column
    for column in cp.columns
    if column not in metadata_columns
    and not column.startswith("AUX_")
    and pd.api.types.is_numeric_dtype(cp[column])
]
reference_features = [
    column for column in base_feature_columns if "gr" not in column.lower()
]
if len(reference_features) != 100:
    raise RuntimeError(f"expected 100 no-GR reference features, got {len(reference_features)}")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
missing_candidates = [column for column in CANDIDATES if column not in header]
if missing_candidates:
    raise RuntimeError(f"pool missing candidate columns: {missing_candidates}")
usecols = [
    column
    for column in [
        "id",
        "well",
        "row_idx",
        "tvt_true",
        BASE_COL,
        *CANDIDATES,
    ]
    if column in header
]
df = pd.read_csv(POOL_PATH, usecols=usecols)
df["id"] = df["id"].astype(str)
if "well" not in df.columns:
    df["well"] = df["id"].str.rsplit("_", n=1).str[0]
df["well"] = df["well"].astype(str)
if "row_idx" not in df.columns:
    df["row_idx"] = df["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
df = df.sort_values(["well", "row_idx"]).reset_index(drop=True)

fold_map = cp.groupby("well")["fold"].first().astype(int).to_dict()
if set(df["well"].unique()) != set(fold_map):
    raise RuntimeError("pool and control-point well sets differ")

for column in ["tvt_true", BASE_COL, *CANDIDATES]:
    df[column] = df[column].astype(np.float32)
truth = df["tvt_true"].to_numpy(np.float32)
base = df[BASE_COL].to_numpy(np.float32)
candidate_arrays = {
    column: df[column].to_numpy(np.float32) for column in CANDIDATES
}
groups = df.groupby("well", sort=False).indices


# ---------------------------------------------------------------------------
# Build target-free reliability and raw cost-profile columns at the same CPs.
# ---------------------------------------------------------------------------
records = []
failures = []
started = time.time()
for wid, indices in tqdm(groups.items(), desc="raw cost profiles"):
    indices = np.asarray(indices, dtype=np.int64)
    controls = cp[cp["well"].eq(wid)].sort_values("center_local")
    try:
        hw = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
            usecols=["GR", "TVT_input"],
        )
        tw = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__typewell.csv",
            usecols=["TVT", "GR"],
        ).sort_values("TVT")

        tvt_input = hw["TVT_input"].to_numpy(float)
        horizontal_gr = hw["GR"].to_numpy(float)
        evaluation_rows = np.flatnonzero(~np.isfinite(tvt_input))
        if not np.array_equal(
            df.iloc[indices]["row_idx"].to_numpy(int), evaluation_rows
        ):
            raise RuntimeError("pool/eval row mismatch")

        tw_tvt = tw["TVT"].to_numpy(float)
        tw_gr = (
            pd.Series(tw["GR"].to_numpy(float))
            .interpolate(limit_direction="both")
            .to_numpy(float)
        )
        known = np.isfinite(tvt_input) & np.isfinite(horizontal_gr)
        if known.sum() < 16:
            raise RuntimeError("too few finite prefix GR/TVT rows")
        tw_at_known = np.interp(tvt_input[known], tw_tvt, tw_gr)
        horizontal_center, horizontal_scale = _robust_center_scale(
            horizontal_gr[known]
        )
        typewell_center, typewell_scale = _robust_center_scale(tw_at_known)

        eval_gr = horizontal_gr[evaluation_rows]
        eval_normalized = (
            (eval_gr - horizontal_center) / horizontal_scale
        ).astype(np.float32)
        valid_eval = np.isfinite(eval_normalized)
        nearest_valid = _nearest_valid_distance(valid_eval)
        tw_normalized = ((tw_gr - typewell_center) / typewell_scale).astype(
            np.float32
        )
        known_normalized = (
            horizontal_gr[known] - horizontal_center
        ) / horizontal_scale
        tw_known_normalized = (
            tw_at_known - typewell_center
        ) / typewell_scale
        known_calibration_rmse = _rmse(
            known_normalized, tw_known_normalized
        )
        n_rows = len(indices)

        # Precompute every candidate/offset typewell trace once per well.
        candidate_profiles = {}
        for candidate in CANDIDATES:
            curve = candidate_arrays[candidate][indices].astype(float)
            for offset in OFFSETS:
                candidate_profiles[(candidate, int(offset))] = np.interp(
                    curve + float(offset),
                    tw_tvt,
                    tw_normalized,
                    left=float(tw_normalized[0]),
                    right=float(tw_normalized[-1]),
                ).astype(np.float32)

        for cp_index, control in controls.iterrows():
            center = int(control["center_local"])
            left = max(0, center - WIN_ROWS)
            right = min(n_rows, center + WIN_ROWS + 1)
            local = np.arange(left, right)
            record = {"cp_index": int(cp_index)}

            record["REL_eval_valid_fraction"] = float(valid_eval.mean())
            record["REL_known_valid_fraction"] = float(
                np.mean(np.isfinite(horizontal_gr[np.isfinite(tvt_input)]))
            )
            record["REL_center_is_valid"] = float(valid_eval[center])
            record["REL_center_nearest_valid"] = float(nearest_valid[center])
            record["REL_prefix_calibration_rmse"] = float(
                known_calibration_rmse
            )
            record["REL_scale_ratio"] = float(
                horizontal_scale / max(typewell_scale, 1e-6)
            )
            record["REL_center_difference"] = float(
                horizontal_center - typewell_center
            )
            for scale in RELIABILITY_SCALES:
                rel_left = max(0, center - scale)
                rel_right = min(n_rows, center + scale + 1)
                local_valid = valid_eval[rel_left:rel_right]
                record[f"REL_valid_fraction_s{scale}"] = float(
                    local_valid.mean()
                )
                record[f"REL_valid_rows_s{scale}"] = int(local_valid.sum())

            horizontal_local = eval_normalized[local]
            for candidate in CANDIDATES:
                for offset in OFFSETS:
                    suffix = _offset_name(offset)
                    typewell_local = candidate_profiles[
                        (candidate, int(offset))
                    ][local]
                    prefix = f"COST_{candidate}_{suffix}"
                    record[f"{prefix}_mae"] = _mean_cost(
                        horizontal_local, typewell_local
                    )
                    record[f"{prefix}_ncc"] = _ncc(
                        horizontal_local, typewell_local
                    )
            records.append(record)
    except Exception as exc:
        failures.append({"well": wid, "reason": repr(exc)})

pd.DataFrame(failures).to_csv(OUT_DIR / "feature_failures.csv", index=False)
if failures:
    display(pd.DataFrame(failures).head(30))
    raise RuntimeError(f"failed wells: {len(failures)}")

added = pd.DataFrame(records).set_index("cp_index").sort_index()
if not np.array_equal(added.index.to_numpy(), np.arange(len(cp))):
    raise RuntimeError("cost-profile rows do not align with CP table")
for column in added.columns:
    cp[column] = added[column].to_numpy(np.float32)

reliability_features = [
    column for column in cp.columns if column.startswith("REL_")
]
cost_features = [
    column for column in cp.columns if column.startswith("COST_")
]
variants = {
    "reference": reference_features,
    "reliability_only": sorted(set(reference_features + reliability_features)),
    "cost_profile": sorted(set(reference_features + cost_features)),
    "cost_plus_reliability": sorted(
        set(reference_features + cost_features + reliability_features)
    ),
}
print(
    {
        "experiment": "cp8_raw_cost_profile_ablation_v0",
        "base_rmse": _rmse(truth, base),
        "reference_features": len(reference_features),
        "reliability_features": len(reliability_features),
        "cost_features": len(cost_features),
        "variant_sizes": {name: len(columns) for name, columns in variants.items()},
        "feature_elapsed_sec": time.time() - started,
    }
)


def _train_control_oof(name, feature_columns):
    matrix = (
        cp[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    target = cp["target_residual"].to_numpy(np.float32)
    folds = cp["fold"].to_numpy(int)
    output = np.full(len(cp), np.nan, dtype=np.float32)
    rows = []
    for fold in range(1, N_FOLDS + 1):
        train_indices = np.flatnonzero(folds != fold)
        valid_indices = np.flatnonzero(folds == fold)
        model = LGBMRegressor(
            objective="huber",
            alpha=0.90,
            n_estimators=2000,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.90,
            reg_alpha=0.20,
            reg_lambda=4.0,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            matrix.iloc[train_indices],
            target[train_indices],
            eval_set=[(matrix.iloc[valid_indices], target[valid_indices])],
            eval_metric="rmse",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        prediction = model.predict(matrix.iloc[valid_indices]).astype(np.float32)
        output[valid_indices] = prediction
        rows.append(
            {
                "variant": name,
                "fold": fold,
                "features": len(feature_columns),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "target_rmse": _rmse(target[valid_indices], prediction),
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError(f"non-finite controls: {name}")
    return output, rows


def _expand(control_prediction):
    move = np.zeros(len(df), dtype=np.float32)
    for wid, indices in groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        controls = cp[cp["well"].eq(wid)].sort_values("center_local")
        positions = controls["center_local"].to_numpy(float)
        values = control_prediction[controls.index.to_numpy(int)].astype(float)
        keep = np.r_[True, np.diff(positions) > 0]
        positions = positions[keep]
        values = values[keep]
        if len(positions) == 1:
            well_move = np.full(len(indices), values[0])
        else:
            well_move = np.interp(
                np.arange(len(indices), dtype=float), positions, values
            )
        sigma = max(4.0, len(indices) / N_CP / 8.0)
        well_move = gaussian_filter1d(well_move, sigma=sigma, mode="nearest")
        move[indices] = well_move.astype(np.float32)
    return np.clip(CP_ALPHA * move, -CP_CAP, CP_CAP)


model_rows = []
moves = {}
predictions = {}
for name, columns in tqdm(variants.items(), desc="cost-profile variants"):
    controls, report = _train_control_oof(name, columns)
    model_rows.extend(report)
    moves[name] = _expand(controls)
    predictions[name] = base + moves[name]

base_sse_bywell = []
for wid, indices in groups.items():
    indices = np.asarray(indices, dtype=np.int64)
    error = base[indices] - truth[indices]
    base_sse_bywell.append({"well": wid, "sse": float(np.sum(error * error))})
top40 = set(pd.DataFrame(base_sse_bywell).nlargest(40, "sse")["well"].astype(str))
top40_mask = df["well"].isin(top40).to_numpy()
base_rmse = _rmse(truth, base)
top40_base_rmse = _rmse(truth[top40_mask], base[top40_mask])

score_rows = []
fold_rows = []
for name, prediction in predictions.items():
    fold_gains = []
    for fold in range(1, N_FOLDS + 1):
        valid_wells = {wid for wid, value in fold_map.items() if value == fold}
        mask = df["well"].isin(valid_wells).to_numpy()
        fold_base = _rmse(truth[mask], base[mask])
        fold_score = _rmse(truth[mask], prediction[mask])
        gain = fold_base - fold_score
        fold_gains.append(gain)
        fold_rows.append(
            {
                "variant": name,
                "fold": fold,
                "rows": int(mask.sum()),
                "base_rmse": fold_base,
                "rmse": fold_score,
                "gain": gain,
            }
        )
    score = _rmse(truth, prediction)
    score_rows.append(
        {
            "variant": name,
            "features": len(variants[name]),
            "base_rmse": base_rmse,
            "rmse": score,
            "gain": base_rmse - score,
            "mean_fold_gain": float(np.mean(fold_gains)),
            "min_fold_gain": float(np.min(fold_gains)),
            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
            "top40_gain": (
                top40_base_rmse
                - _rmse(truth[top40_mask], prediction[top40_mask])
            ),
            "mean_abs_move": float(np.mean(np.abs(moves[name]))),
        }
    )

score_report = pd.DataFrame(score_rows).sort_values("gain", ascending=False)
fold_report = pd.DataFrame(fold_rows)
reference = score_report[score_report["variant"].eq("reference")].iloc[0]
primary = score_report[score_report["variant"].eq("cost_profile")].iloc[0]
primary_folds = fold_report[fold_report["variant"].eq("cost_profile")]
reference_reproduced = bool(
    abs(float(reference["gain"]) - EXPECTED_REFERENCE_GAIN)
    <= REFERENCE_TOLERANCE
)
strict_pass = bool(
    reference_reproduced
    and float(primary["gain"]) - float(reference["gain"])
    >= PRIMARY_INCREMENT_GATE
    and (primary_folds["gain"] > 0).all()
)
both = score_report[
    score_report["variant"].eq("cost_plus_reliability")
].iloc[0]
both_folds = fold_report[
    fold_report["variant"].eq("cost_plus_reliability")
]
recommended_variant = "cost_profile"
if (
    strict_pass
    and float(both["gain"]) - float(primary["gain"]) >= 0.01
    and (both_folds["gain"] > 0).all()
):
    recommended_variant = "cost_plus_reliability"

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nSCORES")
display(score_report)
print("\nFOLDS")
display(fold_report)
verdict = {
    "reference_gain": float(reference["gain"]),
    "reference_reproduced": reference_reproduced,
    "primary_gain": float(primary["gain"]),
    "primary_incremental_gain": float(primary["gain"])
    - float(reference["gain"]),
    "primary_positive_folds": int(primary["positive_folds"]),
    "strict_pass": strict_pass,
    "recommended_variant": recommended_variant if strict_pass else "reference",
    "next": (
        "train LGBM/CatBoost/XGBoost ensemble on the accepted input set"
        if strict_pass
        else "stop raw cost-profile input route"
    ),
}
print("\nVERDICT", verdict)

pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_folds.csv", index=False)
score_report.to_csv(OUT_DIR / "score_report.csv", index=False)
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
cp.to_pickle(OUT_DIR / "cost_profile_control_point_table.pkl")
if strict_pass:
    output = df[["id", "well", "row_idx", "tvt_true"]].copy()
    output[BASE_COL] = base
    for name, prediction in predictions.items():
        output[name] = prediction
    output.to_pickle(OUT_DIR / "cost_profile_oof.pkl")
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
