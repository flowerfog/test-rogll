"""Matched candidate-input expansion for the no-GR CP8 residual model.

This cell reuses the exact control-point table from the feature-family
ablation. The reference variant is therefore expected to reproduce the
previous no_gr gain (0.10307155). Only candidate-trajectory features are
added:

* no_gr_reference: the proven 100-feature model, unchanged;
* core6_multiscale: core six candidates at 48/192/768-row contexts;
* expanded_basic: additional pool candidates at the original 192-row context;
* expanded_multiscale: both changes together.

No GR-derived feature is introduced.
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

OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "aux_aware_cp8_ablation_v0"
CP_PATH = SOURCE_DIR / "control_point_table.pkl"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "cp8_candidate_input_expansion_v1_matched"

BASE_COL = "cp12"
CORE_WANTED = ["ridge", "selector", "fleongg", "sp45", "strong_bias", "strong"]
EXPANDED_WANTED = [
    *CORE_WANTED,
    "ridge_clip",
    "cpfam_sb_k8",
    "risk_gr_combo",
    "cp_pack",
    "cp_single12",
    "strong_raw",
    "sp45_raw",
    "gr_offset",
]
SCALES = (48, 192, 768)
N_CP = 8
N_FOLDS = 5
SEED = 20260713
CP_ALPHA = 0.50
CP_CAP = 6.0
EXPECTED_REFERENCE_GAIN = 0.10307155203761553
REFERENCE_TOLERANCE = 0.005

OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (CP_PATH, POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _stats(values, prefix):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        finite = np.array([0.0])
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(np.quantile(finite, 0.10)),
        f"{prefix}_p50": float(np.quantile(finite, 0.50)),
        f"{prefix}_p90": float(np.quantile(finite, 0.90)),
        f"{prefix}_span": float(np.ptp(finite)),
    }


def _is_candidate(column):
    roots = ("ridge", "selector", "fleongg", "sp45", "strong_bias", "strong")
    return column.startswith("candidate_") or column.startswith(roots)


def _is_gr(column):
    return "gr" in column.lower()


# ---------------------------------------------------------------------------
# Load the exact prior CP table and construct the unchanged no-GR reference.
# ---------------------------------------------------------------------------
cp = pd.read_pickle(CP_PATH).copy().reset_index(drop=True)
cp["well"] = cp["well"].astype(str)
metadata = {"well", "fold", "center_local", "target_residual"}
base_feature_columns = [
    column
    for column in cp.columns
    if column not in metadata
    and not column.startswith("AUX_")
    and pd.api.types.is_numeric_dtype(cp[column])
]
no_gr_reference = [column for column in base_feature_columns if not _is_gr(column)]

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
core = [column for column in CORE_WANTED if column in header]
expanded = [
    column
    for column in EXPANDED_WANTED
    if column in header and column != BASE_COL
]
extras = [column for column in expanded if column not in core]
if core != [column for column in CORE_WANTED if column in header]:
    raise RuntimeError(f"core candidate order mismatch: {core}")
if len(core) < 4:
    raise RuntimeError(f"too few core candidates: {core}")
if not extras:
    raise RuntimeError("no expanded candidates are present in the pool")

usecols = [
    column
    for column in ["id", "well", "row_idx", "tvt_true", BASE_COL, *expanded]
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

for column in ["tvt_true", BASE_COL, *expanded]:
    df[column] = df[column].astype(np.float32)

truth = df["tvt_true"].to_numpy(np.float32)
base = df[BASE_COL].to_numpy(np.float32)
candidate_arrays = {
    column: df[column].to_numpy(np.float32) for column in expanded
}

print(
    {
        "experiment": "cp8_candidate_input_expansion_v1_matched",
        "base_rmse": _rmse(truth, base),
        "reference_features": len(no_gr_reference),
        "core": core,
        "extras": extras,
        "scales": SCALES,
    }
)


# ---------------------------------------------------------------------------
# Add only matched candidate features. The original 192-row core features
# already exist in cp, so they are not rebuilt.
# ---------------------------------------------------------------------------
added_records = []
started = time.time()
groups = df.groupby("well", sort=False).indices
for wid, indices in tqdm(groups.items(), desc="candidate expansion features"):
    indices = np.asarray(indices, dtype=np.int64)
    n_rows = len(indices)
    base_well = base[indices]
    controls = cp[cp["well"].eq(wid)].sort_values("center_local")
    if len(controls) != N_CP:
        raise RuntimeError(f"{wid}: expected {N_CP} controls, got {len(controls)}")

    for cp_index, control in controls.iterrows():
        center = int(control["center_local"])
        record = {"cp_index": int(cp_index)}
        for scale in SCALES:
            left = max(0, center - scale)
            right = min(n_rows, center + scale + 1)
            local = np.arange(left, right)
            global_rows = indices[local]

            core_stack = np.column_stack(
                [candidate_arrays[column][global_rows] for column in core]
            )
            expanded_stack = np.column_stack(
                [candidate_arrays[column][global_rows] for column in expanded]
            )

            # Existing candidate_std_mean/range_mean are the core s192 values.
            if scale != 192:
                record[f"MS_CORE_disp_s{scale}_std_mean"] = float(
                    np.mean(np.std(core_stack, axis=1))
                )
                record[f"MS_CORE_disp_s{scale}_range_mean"] = float(
                    np.mean(np.max(core_stack, axis=1) - np.min(core_stack, axis=1))
                )

            record[f"MS_EXP_disp_s{scale}_std_mean"] = float(
                np.mean(np.std(expanded_stack, axis=1))
            )
            record[f"MS_EXP_disp_s{scale}_range_mean"] = float(
                np.mean(
                    np.max(expanded_stack, axis=1)
                    - np.min(expanded_stack, axis=1)
                )
            )

            for column in expanded:
                if scale == 192 and column in core:
                    continue
                difference = (
                    candidate_arrays[column][global_rows] - base_well[local]
                )
                family = "CORE" if column in core else "EXTRA"
                prefix = f"MS_{family}_{column}_s{scale}"
                record.update(_stats(difference, prefix))
                record[f"{prefix}_center"] = float(difference[center - left])
        added_records.append(record)

added = pd.DataFrame(added_records).set_index("cp_index").sort_index()
if not np.array_equal(added.index.to_numpy(), np.arange(len(cp))):
    raise RuntimeError("candidate feature rows do not align with CP table")
for column in added.columns:
    cp[column] = added[column].to_numpy(np.float32)

core_multiscale_add = [
    column for column in cp.columns if column.startswith("MS_CORE_")
]
extra_192_add = [
    column
    for column in cp.columns
    if column.startswith("MS_EXTRA_") and "_s192_" in column
]
expanded_dispersion_192 = [
    column
    for column in cp.columns
    if column.startswith("MS_EXP_disp_s192_")
]
all_expansion_add = [
    column
    for column in cp.columns
    if column.startswith(("MS_CORE_", "MS_EXTRA_", "MS_EXP_"))
]

variants = {
    "no_gr_reference": no_gr_reference,
    "core6_multiscale": sorted(set(no_gr_reference + core_multiscale_add)),
    "expanded_basic": sorted(
        set(no_gr_reference + extra_192_add + expanded_dispersion_192)
    ),
    "expanded_multiscale": sorted(
        set(no_gr_reference + all_expansion_add)
    ),
}
print("CP table:", cp.shape, "elapsed_sec:", time.time() - started)
print("variant sizes:", {name: len(columns) for name, columns in variants.items()})


def _train_control_oof(name, feature_columns):
    matrix = (
        cp[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    target = cp["target_residual"].to_numpy(np.float32)
    output = np.full(len(cp), np.nan, dtype=np.float32)
    rows = []
    folds = cp["fold"].to_numpy(int)

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
        raise RuntimeError(f"non-finite control predictions: {name}")
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
predictions = {}
moves = {}
for name, columns in tqdm(variants.items(), desc="candidate variants"):
    control, report = _train_control_oof(name, columns)
    model_rows.extend(report)
    moves[name] = _expand(control)
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
reference_gain = float(
    score_report.loc[
        score_report["variant"].eq("no_gr_reference"), "gain"
    ].iloc[0]
)
reference_reproduced = abs(reference_gain - EXPECTED_REFERENCE_GAIN) <= REFERENCE_TOLERANCE
best = score_report.iloc[0]
best_fold_rows = fold_report[fold_report["variant"].eq(best["variant"])]
strict_pass = bool(
    reference_reproduced
    and best["variant"] != "no_gr_reference"
    and float(best["gain"]) >= 0.12
    and float(best["gain"]) - reference_gain >= 0.02
    and (best_fold_rows["gain"] > 0).all()
)

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nSCORES")
display(score_report)
print("\nFOLDS")
display(fold_report)
verdict = {
    "reference_gain": reference_gain,
    "reference_reproduced": reference_reproduced,
    "best_variant": str(best["variant"]),
    "best_gain": float(best["gain"]),
    "incremental_gain": float(best["gain"]) - reference_gain,
    "strict_pass": strict_pass,
    "next": (
        "deployability audit, then model-family ensemble"
        if strict_pass
        else "keep no_gr reference; stop candidate expansion"
    ),
}
print("\nVERDICT", verdict)

pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_folds.csv", index=False)
score_report.to_csv(OUT_DIR / "score_report.csv", index=False)
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
cp.to_pickle(OUT_DIR / "expanded_control_point_table.pkl")
if strict_pass:
    output = df[["id", "well", "row_idx", "tvt_true"]].copy()
    output[BASE_COL] = base
    for name in variants:
        output[name] = predictions[name]
    output.to_pickle(OUT_DIR / "candidate_expansion_oof.pkl")
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
