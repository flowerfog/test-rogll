"""Leave-one-fold-out calibration of existing auxiliary-future OOF predictions.

No LightGBM model is retrained. For each horizon and held-out fold, two fixed
calibrators are fitted on the other four folds:

* affine: truth ~ auxiliary prediction;
* ridge2: truth ~ auxiliary prediction + visible-prefix slope extrapolation.

The held fold is never used to estimate its calibration. The calibrated
horizon predictions are then converted to complete trajectories and evaluated
against CP12 with the same pooled-row/fold gates as v0.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "aux_future_deltas_v0_k5_h025_050_075_100"
AUX_PATH = SOURCE_DIR / "auxiliary_well_oof.pkl"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "aux_future_calibration_rescore_v0"

BASE_COLUMN = "cp12"
HORIZONS = np.array([0.25, 0.50, 0.75, 1.00], dtype=float)
CALIBRATORS = ("affine", "ridge2")
BLEND_WEIGHTS = (0.02, 0.05, 0.10, 0.20, 0.30)
PASS_BLEND_GAIN = 0.10
PASS_MIN_FOLD_GAIN = 0.05

OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (AUX_PATH, POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

aux = pd.read_pickle(AUX_PATH).copy()
aux["well"] = aux["well"].astype(str)
if aux["well"].nunique() < 760:
    raise RuntimeError("Auxiliary OOF is incomplete")


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _corr(y, prediction, kind):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    valid = np.isfinite(y) & np.isfinite(prediction)
    if valid.sum() < 5:
        return 0.0
    if kind == "pearson":
        return float(pearsonr(y[valid], prediction[valid])[0])
    return float(spearmanr(y[valid], prediction[valid])[0])


calibration_rows = []
for horizon in HORIZONS:
    suffix = f"h{int(round(100*horizon)):03d}"
    pred_column = f"pred_{suffix}"
    truth_column = f"true_{suffix}"
    baseline_column = f"baseline_{suffix}"
    aux[f"cal_affine_{suffix}"] = np.nan
    aux[f"cal_ridge2_{suffix}"] = np.nan

    for fold in sorted(aux["fold"].unique()):
        train_mask = aux["fold"].ne(fold)
        valid_mask = aux["fold"].eq(fold)
        y_train = aux.loc[train_mask, truth_column].to_numpy(float)

        affine = LinearRegression()
        affine.fit(aux.loc[train_mask, [pred_column]], y_train)
        affine_prediction = affine.predict(aux.loc[valid_mask, [pred_column]])
        aux.loc[valid_mask, f"cal_affine_{suffix}"] = affine_prediction

        ridge2 = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge_features = [pred_column, baseline_column]
        ridge2.fit(aux.loc[train_mask, ridge_features], y_train)
        ridge_prediction = ridge2.predict(aux.loc[valid_mask, ridge_features])
        aux.loc[valid_mask, f"cal_ridge2_{suffix}"] = ridge_prediction

        valid_truth = aux.loc[valid_mask, truth_column].to_numpy(float)
        calibration_rows.extend(
            [
                {
                    "fold": int(fold),
                    "horizon": horizon,
                    "calibrator": "affine",
                    "rows": int(valid_mask.sum()),
                    "rmse": _rmse(valid_truth, affine_prediction),
                    "slope_baseline_rmse": _rmse(
                        valid_truth,
                        aux.loc[valid_mask, baseline_column],
                    ),
                },
                {
                    "fold": int(fold),
                    "horizon": horizon,
                    "calibrator": "ridge2",
                    "rows": int(valid_mask.sum()),
                    "rmse": _rmse(valid_truth, ridge_prediction),
                    "slope_baseline_rmse": _rmse(
                        valid_truth,
                        aux.loc[valid_mask, baseline_column],
                    ),
                },
            ]
        )

calibration_folds = pd.DataFrame(calibration_rows)
calibration_folds["gain_vs_slope"] = (
    calibration_folds["slope_baseline_rmse"] - calibration_folds["rmse"]
)

horizon_rows = []
for horizon in HORIZONS:
    suffix = f"h{int(round(100*horizon)):03d}"
    truth = aux[f"true_{suffix}"].to_numpy(float)
    baseline = aux[f"baseline_{suffix}"].to_numpy(float)
    for calibrator in CALIBRATORS:
        prediction = aux[f"cal_{calibrator}_{suffix}"].to_numpy(float)
        horizon_rows.append(
            {
                "horizon": horizon,
                "calibrator": calibrator,
                "rmse": _rmse(truth, prediction),
                "slope_baseline_rmse": _rmse(truth, baseline),
                "gain_vs_slope": _rmse(truth, baseline) - _rmse(truth, prediction),
                "pearson": _corr(truth, prediction, "pearson"),
                "spearman": _corr(truth, prediction, "spearman"),
                "min_fold_gain_vs_slope": float(
                    calibration_folds.loc[
                        calibration_folds["horizon"].eq(horizon)
                        & calibration_folds["calibrator"].eq(calibrator),
                        "gain_vs_slope",
                    ].min()
                ),
            }
        )
horizon_report = pd.DataFrame(horizon_rows)


pool_header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols = [
    column
    for column in ["id", "well", "row_idx", "tvt_true", BASE_COLUMN]
    if column in pool_header
]
pool = pd.read_csv(POOL_PATH, usecols=usecols)
pool["id"] = pool["id"].astype(str)
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(int)
fold_map = aux.groupby("well")["fold"].first().astype(int).to_dict()
pool["fold"] = pool["well"].map(fold_map).astype(int)
aux_by_well = aux.set_index("well")

curve_parts = []
for wid, group in tqdm(pool.groupby("well", sort=False), desc="calibrated curves"):
    group = group.sort_values("row_idx").copy()
    hw = pd.read_csv(
        DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
        usecols=["Z", "TVT_input"],
    )
    tvt_input = hw["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    anchor = int(known[-1])
    evaluation = group["row_idx"].to_numpy(int)
    expected = np.flatnonzero(~np.isfinite(tvt_input))
    if not np.array_equal(evaluation, expected):
        raise RuntimeError(f"row mismatch for {wid}")
    z = hw["Z"].to_numpy(float)
    anchor_s = float(tvt_input[anchor] + z[anchor])
    fraction = (evaluation - anchor) / max(float(evaluation[-1] - anchor), 1.0)
    aux_row = aux_by_well.loc[wid]
    for calibrator in CALIBRATORS:
        predicted_ds = np.asarray(
            [
                aux_row[f"cal_{calibrator}_h{int(round(100*h)):03d}"]
                for h in HORIZONS
            ],
            dtype=float,
        )
        ds = np.interp(
            fraction,
            np.concatenate(([0.0], HORIZONS)),
            np.concatenate(([0.0], predicted_ds)),
        )
        group[f"aux_{calibrator}"] = anchor_s + ds - z[evaluation]
    curve_parts.append(group)

curves = pd.concat(curve_parts, ignore_index=True).reset_index(drop=True)
y = curves["tvt_true"].to_numpy(float)
base_prediction = curves[BASE_COLUMN].to_numpy(float)
base_rmse = _rmse(y, base_prediction)

bywell = []
for wid, group in curves.groupby("well", sort=False):
    error = group[BASE_COLUMN].to_numpy(float) - group["tvt_true"].to_numpy(float)
    bywell.append({"well": wid, "base_sse": float(np.sum(error * error))})
top40 = set(pd.DataFrame(bywell).nlargest(40, "base_sse")["well"].astype(str))
top40_mask = curves["well"].isin(top40).to_numpy()
top40_base = _rmse(y[top40_mask], base_prediction[top40_mask])

blend_rows = []
fold_rows = []
for calibrator in CALIBRATORS:
    aux_prediction = curves[f"aux_{calibrator}"].to_numpy(float)
    aux_rmse = _rmse(y, aux_prediction)
    base_error = base_prediction - y
    aux_error = aux_prediction - y
    error_cosine = float(
        np.dot(base_error, aux_error)
        / max(np.linalg.norm(base_error) * np.linalg.norm(aux_error), 1e-12)
    )
    for weight in BLEND_WEIGHTS:
        prediction = (1.0 - weight) * base_prediction + weight * aux_prediction
        fold_gains = []
        for fold, index in curves.groupby("fold", sort=True).groups.items():
            positions = curves.index.get_indexer(index)
            fold_base = _rmse(y[positions], base_prediction[positions])
            fold_score = _rmse(y[positions], prediction[positions])
            gain = fold_base - fold_score
            fold_gains.append(gain)
            fold_rows.append(
                {
                    "calibrator": calibrator,
                    "weight_aux": weight,
                    "fold": int(fold),
                    "base_rmse": fold_base,
                    "rmse": fold_score,
                    "gain": gain,
                }
            )
        score = _rmse(y, prediction)
        top40_score = _rmse(y[top40_mask], prediction[top40_mask])
        movement = prediction - base_prediction
        blend_rows.append(
            {
                "calibrator": calibrator,
                "weight_aux": weight,
                "base_rmse": base_rmse,
                "aux_rmse": aux_rmse,
                "error_cosine": error_cosine,
                "rmse": score,
                "gain": base_rmse - score,
                "mean_fold_gain": float(np.mean(fold_gains)),
                "min_fold_gain": float(np.min(fold_gains)),
                "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
                "top40_gain": top40_base - top40_score,
                "mean_abs_move": float(np.mean(np.abs(movement))),
                "p95_abs_move": float(np.quantile(np.abs(movement), 0.95)),
            }
        )

blend_report = pd.DataFrame(blend_rows)
blend_report["strict_gate"] = (
    (blend_report["gain"] >= PASS_BLEND_GAIN)
    & (blend_report["positive_folds"] == 5)
    & (blend_report["min_fold_gain"] >= PASS_MIN_FOLD_GAIN)
    & (blend_report["top40_gain"] > 0.0)
)
blend_report = blend_report.sort_values(
    ["strict_gate", "gain"], ascending=[False, False]
)
fold_report = pd.DataFrame(fold_rows)

calibrator_pass = {}
for calibrator in CALIBRATORS:
    report = horizon_report[horizon_report["calibrator"].eq(calibrator)]
    calibrator_pass[calibrator] = bool(
        len(report) == len(HORIZONS)
        and (report["gain_vs_slope"] > 0).all()
        and (report["min_fold_gain_vs_slope"] > 0).all()
        and (report["spearman"] > 0.20).all()
    )
target_pass = bool(any(calibrator_pass.values()))
blend_pass = bool(blend_report["strict_gate"].any())

print("\nCALIBRATION FOLDS")
display(calibration_folds)
print("\nCALIBRATED HORIZONS")
display(horizon_report)
print("\nCALIBRATED BLENDS")
display(blend_report)
print("\nBLEND FOLDS")
display(fold_report)
print(
    "\nVERDICT",
    {
        "base_rmse": base_rmse,
        "target_pass": target_pass,
        "calibrator_pass": calibrator_pass,
        "blend_pass": blend_pass,
        "next": (
            "use calibrated future OOF features in main tabular model"
            if target_pass
            else "stop auxiliary-future route"
        ),
    },
)

aux.to_pickle(OUT_DIR / "calibrated_auxiliary_well_oof.pkl")
curves.to_pickle(OUT_DIR / "calibrated_auxiliary_curve_oof.pkl")
calibration_folds.to_csv(OUT_DIR / "calibration_fold_report.csv", index=False)
horizon_report.to_csv(OUT_DIR / "calibrated_horizon_report.csv", index=False)
blend_report.to_csv(OUT_DIR / "calibrated_blend_report.csv", index=False)
fold_report.to_csv(OUT_DIR / "calibrated_blend_folds.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(
        {
            "base_rmse": base_rmse,
            "target_pass": target_pass,
            "calibrator_pass": calibrator_pass,
            "blend_pass": blend_pass,
        },
        indent=2,
    )
)
print("saved:", OUT_DIR)
