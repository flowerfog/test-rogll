"""CP8 residual ablation with calibrated auxiliary-future features.

Two otherwise identical fold-honest LightGBM control-point models are trained:

* base_features: trajectory, GR, CP12, and deployable candidate-pool context;
* aux_features: base_features plus four calibrated ridge2 future-delta-S OOF
  predictions and their interpolated per-row structural trajectory.

The target is local mean ``TVT_true - CP12`` at eight control points per well.
Both models use fixed 0.5 shrink and a 6 ft movement cap. Auxiliary targets are
accepted only if they improve the no-aux ablation consistently under the same
folds, in addition to improving CP12 itself.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


warnings.filterwarnings("ignore")

DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
AUX_PATH = (
    OOF_ROOT
    / "aux_future_calibration_rescore_v0"
    / "calibrated_auxiliary_well_oof.pkl"
)
OUT_DIR = OOF_ROOT / "aux_aware_cp8_ablation_v0"

BASE_COL = "cp12"
N_CP = 8
WIN_ROWS = 192
N_FOLDS = 5
SEED = 20260713
CP_ALPHA = 0.50
CP_CAP = 6.0
HORIZONS = np.array([0.25, 0.50, 0.75, 1.00], dtype=float)

PASS_GAIN_VS_CP12 = 0.08
PASS_GAIN_VS_ABLATION = 0.03

OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (POOL_PATH, AUX_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

print(
    {
        "experiment": "aux_aware_cp8_ablation_v0",
        "base": BASE_COL,
        "n_cp": N_CP,
        "window_rows": WIN_ROWS,
        "alpha": CP_ALPHA,
        "cap": CP_CAP,
    }
)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _safe_stats(values, prefix):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_span": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(np.quantile(finite, 0.10)),
        f"{prefix}_p50": float(np.quantile(finite, 0.50)),
        f"{prefix}_p90": float(np.quantile(finite, 0.90)),
        f"{prefix}_span": float(np.ptp(finite)),
    }


header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
candidate_wanted = [
    "ridge",
    "selector",
    "fleongg",
    "sp45",
    "strong_bias",
    "strong",
]
candidate_cols = [column for column in candidate_wanted if column in header]
if not candidate_cols:
    raise RuntimeError("No deployable candidate columns found in pool")
required = ["id", "tvt_true", BASE_COL]
missing = [column for column in required if column not in header]
if missing:
    raise RuntimeError(f"pool missing: {missing}")
usecols = [
    column
    for column in [
        "id",
        "well",
        "row_idx",
        "tvt_true",
        BASE_COL,
        *candidate_cols,
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

aux = pd.read_pickle(AUX_PATH).copy()
aux["well"] = aux["well"].astype(str)
aux_by_well = aux.set_index("well")
fold_map = aux.groupby("well")["fold"].first().astype(int).to_dict()
df["fold"] = df["well"].map(fold_map)
if df["fold"].isna().any():
    raise RuntimeError("missing auxiliary/fold rows")
df["fold"] = df["fold"].astype(int)

for column in ["tvt_true", BASE_COL, *candidate_cols]:
    df[column] = df[column].astype(np.float32)

print(
    "pool:",
    df.shape,
    "wells:",
    df["well"].nunique(),
    "base_rmse:",
    _rmse(df["tvt_true"], df[BASE_COL]),
    "candidates:",
    candidate_cols,
)


# ---------------------------------------------------------------------------
# Build per-row calibrated future features and raw test-schema features.
# ---------------------------------------------------------------------------
well_arrays = {}
failures = []
started = time.time()

for number, (wid, indices) in enumerate(df.groupby("well", sort=False).indices.items(), 1):
    indices = np.asarray(indices, dtype=np.int64)
    group = df.iloc[indices].sort_values("row_idx")
    try:
        hw = pd.read_csv(
            DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
            usecols=["MD", "X", "Y", "Z", "GR", "TVT_input"],
        )
        row_idx = group["row_idx"].to_numpy(int)
        tvt_input = hw["TVT_input"].to_numpy(float)
        evaluation = np.flatnonzero(~np.isfinite(tvt_input))
        known = np.flatnonzero(np.isfinite(tvt_input))
        if not np.array_equal(row_idx, evaluation):
            raise RuntimeError("pool/eval row mismatch")
        anchor = int(known[-1])
        z = hw["Z"].to_numpy(float)
        md = hw["MD"].to_numpy(float)
        x = hw["X"].to_numpy(float)
        y = hw["Y"].to_numpy(float)
        gr = hw["GR"].to_numpy(float)
        gr_fill = pd.Series(gr).interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
        dmd = np.gradient(md)
        dmd = np.where(np.abs(dmd) < 1e-6, np.nan, dmd)
        build = np.gradient(z) / dmd
        dx = np.gradient(x) / dmd
        dy = np.gradient(y) / dmd
        dogleg = np.sqrt(
            np.nan_to_num(np.gradient(dx), nan=0.0) ** 2
            + np.nan_to_num(np.gradient(dy), nan=0.0) ** 2
            + np.nan_to_num(np.gradient(build), nan=0.0) ** 2
        )

        fraction = (row_idx - anchor) / max(float(row_idx[-1] - anchor), 1.0)
        aux_row = aux_by_well.loc[wid]
        predicted_ds = np.asarray(
            [aux_row[f"cal_ridge2_h{int(round(100*h)):03d}"] for h in HORIZONS],
            dtype=float,
        )
        baseline_ds = np.asarray(
            [aux_row[f"baseline_h{int(round(100*h)):03d}"] for h in HORIZONS],
            dtype=float,
        )
        interpolated_ds = np.interp(
            fraction,
            np.concatenate(([0.0], HORIZONS)),
            np.concatenate(([0.0], predicted_ds)),
        )
        interpolated_baseline_ds = np.interp(
            fraction,
            np.concatenate(([0.0], HORIZONS)),
            np.concatenate(([0.0], baseline_ds)),
        )
        anchor_s = float(tvt_input[anchor] + z[anchor])
        aux_tvt = anchor_s + interpolated_ds - z[row_idx]
        slope_tvt = anchor_s + interpolated_baseline_ds - z[row_idx]

        well_arrays[wid] = {
            "indices": indices,
            "row_idx": row_idx,
            "fraction": fraction,
            "MD": md[row_idx],
            "X": x[row_idx],
            "Y": y[row_idx],
            "Z": z[row_idx],
            "GR": gr[row_idx],
            "build": build[row_idx],
            "dogleg": dogleg[row_idx],
            "gr_d1": np.gradient(gr_fill)[row_idx],
            "gr_d2": np.gradient(np.gradient(gr_fill))[row_idx],
            "aux_tvt": aux_tvt,
            "slope_tvt": slope_tvt,
            "aux_ds": interpolated_ds,
            "slope_ds": interpolated_baseline_ds,
            "predicted_horizons": predicted_ds,
            "baseline_horizons": baseline_ds,
            "prefix_n": len(known),
        }
    except Exception as exc:
        failures.append({"well": wid, "reason": repr(exc)})
    if number % 100 == 0:
        print(f"well features {number}/773 elapsed={time.time()-started:.1f}s")

pd.DataFrame(failures).to_csv(OUT_DIR / "feature_failures.csv", index=False)
if failures:
    raise RuntimeError(f"failed wells: {len(failures)}")


# ---------------------------------------------------------------------------
# One supervised row per control point.
# ---------------------------------------------------------------------------
records = []
truth = df["tvt_true"].to_numpy(np.float32)
base = df[BASE_COL].to_numpy(np.float32)
residual = truth - base
candidate_arrays = {column: df[column].to_numpy(np.float32) for column in candidate_cols}

for wid, arrays in tqdm(well_arrays.items(), desc="build CP8 rows"):
    indices = arrays["indices"]
    n_rows = len(indices)
    centers = np.unique(np.round(np.linspace(0, n_rows - 1, N_CP)).astype(int))
    base_well = base[indices]
    aux_minus_base = arrays["aux_tvt"] - base_well
    slope_minus_base = arrays["slope_tvt"] - base_well

    for order, center in enumerate(centers):
        left = max(0, center - WIN_ROWS)
        right = min(n_rows, center + WIN_ROWS + 1)
        local = np.arange(left, right)
        global_rows = indices[local]
        feature = {
            "well": wid,
            "fold": int(fold_map[wid]),
            "center_local": int(center),
            "n_well": n_rows,
            "n_window": len(local),
            "cp_order": order,
            "cp_order_norm": order / max(len(centers) - 1, 1),
            "row_norm": center / max(n_rows - 1, 1),
            "target_residual": float(np.mean(residual[global_rows])),
            "prefix_n": arrays["prefix_n"],
            "base_center": float(base_well[center]),
            "base_slope_window": float(
                (base_well[right - 1] - base_well[left]) / max(right - left - 1, 1)
            ),
        }

        for name in (
            "MD",
            "X",
            "Y",
            "Z",
            "GR",
            "build",
            "dogleg",
            "gr_d1",
            "gr_d2",
            "fraction",
        ):
            feature.update(_safe_stats(arrays[name][local], name))
        feature.update(_safe_stats(base_well[local], "base"))

        stack = np.column_stack([candidate_arrays[column][global_rows] for column in candidate_cols])
        feature["candidate_std_mean"] = float(np.mean(np.std(stack, axis=1)))
        feature["candidate_range_mean"] = float(
            np.mean(np.max(stack, axis=1) - np.min(stack, axis=1))
        )
        for column in candidate_cols:
            difference = candidate_arrays[column][global_rows] - base_well[local]
            feature.update(_safe_stats(difference, f"{column}_minus_base"))
            feature[f"{column}_minus_base_center"] = float(difference[center - left])

        # Auxiliary family: excluded from the base-feature ablation.
        feature.update(_safe_stats(aux_minus_base[local], "AUX_aux_minus_base"))
        feature.update(_safe_stats(slope_minus_base[local], "AUX_slope_minus_base"))
        feature.update(_safe_stats(arrays["aux_ds"][local], "AUX_predicted_ds"))
        feature.update(_safe_stats(arrays["slope_ds"][local], "AUX_slope_ds"))
        feature["AUX_aux_minus_base_center"] = float(aux_minus_base[center])
        feature["AUX_slope_minus_base_center"] = float(slope_minus_base[center])
        feature["AUX_future_disagreement_center"] = float(
            arrays["aux_ds"][center] - arrays["slope_ds"][center]
        )
        for horizon, value, baseline_value in zip(
            HORIZONS,
            arrays["predicted_horizons"],
            arrays["baseline_horizons"],
        ):
            suffix = f"h{int(round(100*horizon)):03d}"
            feature[f"AUX_pred_{suffix}"] = float(value)
            feature[f"AUX_baseline_{suffix}"] = float(baseline_value)
            feature[f"AUX_difference_{suffix}"] = float(value - baseline_value)
        records.append(feature)

cp = pd.DataFrame(records)
print("CP table:", cp.shape)

excluded = {"well", "fold", "center_local", "target_residual"}
all_features = [
    column
    for column in cp.columns
    if column not in excluded and pd.api.types.is_numeric_dtype(cp[column])
]
base_features = [column for column in all_features if not column.startswith("AUX_")]
aux_features = all_features
print("feature counts:", {"base": len(base_features), "aux": len(aux_features)})


def _train_variant(name, features):
    matrix = (
        cp[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    target = cp["target_residual"].to_numpy(np.float32)
    oof = np.full(len(cp), np.nan, dtype=np.float32)
    model_rows = []
    for fold in range(1, N_FOLDS + 1):
        train_indices = np.flatnonzero(cp["fold"].to_numpy(int) != fold)
        valid_indices = np.flatnonzero(cp["fold"].to_numpy(int) == fold)
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
        oof[valid_indices] = prediction
        model_rows.append(
            {
                "variant": name,
                "fold": fold,
                "train_rows": len(train_indices),
                "valid_rows": len(valid_indices),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "target_rmse": _rmse(target[valid_indices], prediction),
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"non-finite OOF for {name}")
    return oof, model_rows


cp_predictions = {}
model_reports = []
for variant, features in (("base_features", base_features), ("aux_features", aux_features)):
    cp_predictions[variant], rows = _train_variant(variant, features)
    model_reports.extend(rows)


def _expand_control_predictions(control_prediction):
    raw_move = np.zeros(len(df), dtype=np.float32)
    for wid, arrays in well_arrays.items():
        indices = arrays["indices"]
        sub = cp[cp["well"].eq(wid)].sort_values("center_local")
        positions = sub["center_local"].to_numpy(float)
        values = control_prediction[sub.index.to_numpy(int)].astype(float)
        keep = np.r_[True, np.diff(positions) > 0]
        positions = positions[keep]
        values = values[keep]
        if len(positions) == 1:
            move = np.full(len(indices), values[0])
        else:
            move = np.interp(np.arange(len(indices), dtype=float), positions, values)
        sigma = max(4.0, len(indices) / N_CP / 8.0)
        move = gaussian_filter1d(move, sigma=sigma, mode="nearest")
        raw_move[indices] = move.astype(np.float32)
    return np.clip(CP_ALPHA * raw_move, -CP_CAP, CP_CAP)


moves = {variant: _expand_control_predictions(values) for variant, values in cp_predictions.items()}
predictions = {variant: base + move for variant, move in moves.items()}


def _summarize_rows(frame):
    rows = int(frame["rows"].sum())
    output = {"rows": rows}
    for name in ("cp12", "base_features", "aux_features"):
        output[f"{name}_rmse"] = float(np.sqrt(frame[f"{name}_sse"].sum() / rows))
    output["base_gain_vs_cp12"] = output["cp12_rmse"] - output["base_features_rmse"]
    output["aux_gain_vs_cp12"] = output["cp12_rmse"] - output["aux_features_rmse"]
    output["aux_gain_vs_ablation"] = (
        output["base_features_rmse"] - output["aux_features_rmse"]
    )
    return output


well_rows = []
for wid, indices in df.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, dtype=np.int64)
    row = {
        "well": wid,
        "fold": int(fold_map[wid]),
        "rows": len(indices),
        "cp12_sse": float(np.sum((base[indices] - truth[indices]) ** 2)),
    }
    for variant in ("base_features", "aux_features"):
        row[f"{variant}_sse"] = float(
            np.sum((predictions[variant][indices] - truth[indices]) ** 2)
        )
        row[f"{variant}_mean_abs_move"] = float(np.mean(np.abs(moves[variant][indices])))
    well_rows.append(row)
well_report = pd.DataFrame(well_rows)

overall = _summarize_rows(well_report)
fold_report = pd.DataFrame(
    [
        {"fold": int(fold), **_summarize_rows(group)}
        for fold, group in well_report.groupby("fold", sort=True)
    ]
)
top40 = well_report.nlargest(40, "cp12_sse")
top40_report = _summarize_rows(top40)

strict_pass = bool(
    overall["aux_gain_vs_cp12"] >= PASS_GAIN_VS_CP12
    and overall["aux_gain_vs_ablation"] >= PASS_GAIN_VS_ABLATION
    and (fold_report["aux_gain_vs_cp12"] > 0).all()
    and (fold_report["aux_gain_vs_ablation"] > 0).all()
    and top40_report["aux_gain_vs_cp12"] > 0
    and top40_report["aux_gain_vs_ablation"] > 0
)

print("\nMODEL FOLDS")
display(pd.DataFrame(model_reports))
print("\nOVERALL")
print(overall)
print("\nFOLDS")
display(fold_report)
print("\nTOP40")
print(top40_report)
print(
    "\nVERDICT",
    {
        "strict_pass": strict_pass,
        "next": (
            "promote calibrated auxiliary features into the main tabular stack"
            if strict_pass
            else "stop auxiliary feature route"
        ),
    },
)

pd.DataFrame(model_reports).to_csv(OUT_DIR / "model_folds.csv", index=False)
fold_report.to_csv(OUT_DIR / "score_folds.csv", index=False)
well_report.to_csv(OUT_DIR / "score_bywell.csv", index=False)
cp.to_pickle(OUT_DIR / "control_point_table.pkl")
if strict_pass:
    output = df[["id", "well", "row_idx", "tvt_true"]].copy()
    output["cp12"] = base
    output["base_features"] = predictions["base_features"]
    output["aux_features"] = predictions["aux_features"]
    output.to_pickle(OUT_DIR / "aux_aware_cp8_oof.pkl")
(OUT_DIR / "verdict.json").write_text(
    json.dumps(
        {
            "strict_pass": strict_pass,
            "overall": overall,
            "top40": top40_report,
        },
        indent=2,
    )
)
print("saved:", OUT_DIR)
