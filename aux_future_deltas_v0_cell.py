"""Multi-cut auxiliary-future target experiment v0.

The model predicts anchor-relative future structural movement
``delta_S = (TVT + Z)_future - (TVT + Z)_anchor`` at 25/50/75/100% of the
remaining trajectory.  Training wells contribute five synthetic anchors;
validation wells are evaluated only at their official TVT_input anchor.

All features are deployable under the test schema: MD/X/Y/Z/GR for the full
well, TVT on the visible prefix, and the assigned typewell TVT/GR.  Formation
surface columns and Geology are deliberately excluded.

This cell answers two separate questions:
1. Are future structural targets predictable out of well/fold?
2. Does the resulting independent trajectory safely complement CP12?
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError("This experiment requires lightgbm") from exc


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
FOLD_OOF = (
    OOF_ROOT
    / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
    / "multicut_v0_oof.pkl"
)

CACHE_VERSION = "v0_k5_h025_050_075_100"
CACHE_DIR = OOF_ROOT / f"aux_future_deltas_{CACHE_VERSION}"
MODEL_DIR = CACHE_DIR / "models"
STATE_CACHE = CACHE_DIR / "multicut_state_table.pkl"

BASE_COLUMN = "cp12"
HORIZONS = np.array([0.25, 0.50, 0.75, 1.00], dtype=float)
CUT_MULTIPLIERS = (0.70, 0.85, 1.00, 1.15, 1.30)
N_FOLDS = 5
RANDOM_STATE = 20260713
BLEND_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.50)

PASS_AUX_CORR = 0.20
PASS_BLEND_GAIN = 0.10
PASS_MIN_FOLD_GAIN = 0.05

for directory in (CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for path in (POOL_PATH, FOLD_OOF):
    if not path.exists():
        raise FileNotFoundError(path)

print(
    {
        "experiment": "aux_future_deltas_v0",
        "base": BASE_COLUMN,
        "horizons": HORIZONS.tolist(),
        "cut_multipliers": CUT_MULTIPLIERS,
        "cache": str(CACHE_DIR),
    }
)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _safe_stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "mean": np.nan,
            "std": np.nan,
            "q10": np.nan,
            "q50": np.nan,
            "q90": np.nan,
            "range": np.nan,
        }
    q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q10": float(q10),
        "q50": float(q50),
        "q90": float(q90),
        "range": float(np.ptp(values)),
    }


def _safe_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.ptp(x[valid]) < 1e-6:
        return 0.0
    x0 = x[valid] - np.mean(x[valid])
    y0 = y[valid] - np.mean(y[valid])
    return float(np.dot(x0, y0) / max(np.dot(x0, x0), 1e-12))


def _add_stats(features, prefix, values):
    for name, value in _safe_stats(values).items():
        features[f"{prefix}_{name}"] = value


def _segment_bounds(start, stop, parts=4):
    edges = np.round(np.linspace(start, stop, parts + 1)).astype(int)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(parts)]


def _build_state(wid, fold, hw, tw, cut, official_anchor):
    n_rows = len(hw)
    cut = int(cut)
    future_start = cut + 1
    if cut < 99 or n_rows - future_start < 256:
        raise RuntimeError("cut lacks prefix/future support")

    md = hw["MD"].to_numpy(float)
    x = hw["X"].to_numpy(float)
    y = hw["Y"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    gr = hw["GR"].to_numpy(float)
    tvt = hw["TVT"].to_numpy(float)
    structural = tvt + z

    anchor_md = float(md[cut])
    anchor_tvt = float(tvt[cut])
    anchor_z = float(z[cut])
    anchor_s = float(structural[cut])
    future_rows = np.arange(future_start, n_rows)
    target_rows = np.clip(
        np.round(cut + HORIZONS * (n_rows - 1 - cut)).astype(int),
        future_start,
        n_rows - 1,
    )

    features = {
        "well": wid,
        "fold": int(fold),
        "cut": cut,
        "official_anchor": int(official_anchor),
        "is_official": bool(cut == official_anchor),
        "n_rows": n_rows,
        "n_prefix": cut + 1,
        "n_future": n_rows - future_start,
        "prefix_frac": (cut + 1) / n_rows,
        "anchor_md": anchor_md,
        "anchor_x": float(x[cut]),
        "anchor_y": float(y[cut]),
        "anchor_z": anchor_z,
        "anchor_tvt": anchor_tvt,
        "anchor_s": anchor_s,
        "future_md_span": float(md[-1] - anchor_md),
        "future_x_delta": float(x[-1] - x[cut]),
        "future_y_delta": float(y[-1] - y[cut]),
        "future_z_delta": float(z[-1] - anchor_z),
        "future_xy_distance": float(np.hypot(x[-1] - x[cut], y[-1] - y[cut])),
        "future_xyz_distance": float(
            np.sqrt(
                (x[-1] - x[cut]) ** 2
                + (y[-1] - y[cut]) ** 2
                + (z[-1] - anchor_z) ** 2
            )
        ),
        "future_z_slope": _safe_slope(md[future_rows], z[future_rows]),
        "future_x_slope": _safe_slope(md[future_rows], x[future_rows]),
        "future_y_slope": _safe_slope(md[future_rows], y[future_rows]),
    }

    # Visible-prefix structural trend: this is measured TVT_input at deploy.
    for window in (50, 200, 500, 1000):
        left = max(0, cut - window + 1)
        rows = np.arange(left, cut + 1)
        features[f"prefix_s_slope_{window}"] = _safe_slope(md[rows], structural[rows])
        features[f"prefix_tvt_slope_{window}"] = _safe_slope(md[rows], tvt[rows])
        features[f"prefix_z_slope_{window}"] = _safe_slope(md[rows], z[rows])
        features[f"prefix_s_delta_{window}"] = float(structural[cut] - structural[left])
        features[f"prefix_tvt_delta_{window}"] = float(tvt[cut] - tvt[left])
        features[f"prefix_z_delta_{window}"] = float(z[cut] - z[left])
        _add_stats(features, f"prefix_gr_{window}", gr[rows])
        features[f"prefix_gr_missing_{window}"] = float(np.mean(~np.isfinite(gr[rows])))

    # The complete future geometry and GR are available in test horizontal files.
    _add_stats(features, "future_gr_all", gr[future_rows])
    features["future_gr_missing_all"] = float(np.mean(~np.isfinite(gr[future_rows])))
    finite_gr = pd.Series(gr).interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    features["future_gr_d1_std"] = float(np.std(np.diff(finite_gr[future_rows])))
    features["future_gr_d1_abs_mean"] = float(np.mean(np.abs(np.diff(finite_gr[future_rows]))))

    for part, (left, right) in enumerate(
        _segment_bounds(future_start, n_rows, parts=4), 1
    ):
        rows = np.arange(left, max(left + 1, right))
        _add_stats(features, f"future_gr_q{part}", gr[rows])
        features[f"future_gr_missing_q{part}"] = float(np.mean(~np.isfinite(gr[rows])))
        features[f"future_z_delta_q{part}"] = float(z[rows[-1]] - z[rows[0]])
        features[f"future_x_delta_q{part}"] = float(x[rows[-1]] - x[rows[0]])
        features[f"future_y_delta_q{part}"] = float(y[rows[-1]] - y[rows[0]])
        features[f"future_z_slope_q{part}"] = _safe_slope(md[rows], z[rows])

    # Assigned typewell context, with no Geology usage.
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].to_numpy(float)
    valid_tw = np.isfinite(tw_tvt)
    tw_tvt = tw_tvt[valid_tw]
    tw_gr = tw_gr[valid_tw]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    features["tw_tvt_min"] = float(np.min(tw_tvt))
    features["tw_tvt_max"] = float(np.max(tw_tvt))
    features["tw_tvt_range"] = float(np.ptp(tw_tvt))
    features["tw_anchor_position"] = float(
        (anchor_tvt - np.min(tw_tvt)) / max(np.ptp(tw_tvt), 1e-6)
    )
    _add_stats(features, "tw_gr_all", tw_gr)
    for radius in (25.0, 50.0, 100.0):
        local = np.abs(tw_tvt - anchor_tvt) <= radius
        _add_stats(features, f"tw_gr_local_{int(radius)}", tw_gr[local])

    # Four supervised future targets and a visible-prefix slope extrapolator.
    recent_left = max(0, cut - 499)
    recent_rows = np.arange(recent_left, cut + 1)
    recent_s_slope = _safe_slope(md[recent_rows], structural[recent_rows])
    for horizon, target_row in zip(HORIZONS, target_rows):
        suffix = f"h{int(round(100 * horizon)):03d}"
        features[f"target_ds_{suffix}"] = float(structural[target_row] - anchor_s)
        features[f"baseline_ds_{suffix}"] = float(
            recent_s_slope * (md[target_row] - anchor_md)
        )
        features[f"target_row_{suffix}"] = int(target_row)
        features[f"target_md_delta_{suffix}"] = float(md[target_row] - anchor_md)
        features[f"target_z_delta_{suffix}"] = float(z[target_row] - anchor_z)

    return features


def _build_state_table():
    fold_source = pd.read_pickle(FOLD_OOF)[["well", "fold"]].drop_duplicates("well")
    fold_source["well"] = fold_source["well"].astype(str)
    fold_map = dict(zip(fold_source["well"], fold_source["fold"].astype(int)))
    files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
    rows = []
    failures = []
    for path in tqdm(files, desc="build multi-cut auxiliary states"):
        wid = path.name.replace("__horizontal_well.csv", "")
        try:
            hw = pd.read_csv(
                path,
                usecols=["MD", "X", "Y", "Z", "TVT", "GR", "TVT_input"],
            )
            tw = pd.read_csv(
                DATA_ROOT / "train" / f"{wid}__typewell.csv",
                usecols=["TVT", "GR"],
            )
            tvt_input = hw["TVT_input"].to_numpy(float)
            known = np.flatnonzero(np.isfinite(tvt_input))
            if len(known) < 100:
                raise RuntimeError("too little visible prefix")
            official_anchor = int(known[-1])
            cut_candidates = []
            for multiplier in CUT_MULTIPLIERS:
                cut = int(round(official_anchor * multiplier))
                cut = int(np.clip(cut, 99, len(hw) - 257))
                cut_candidates.append(cut)
            cut_candidates.append(official_anchor)
            for cut in sorted(set(cut_candidates)):
                rows.append(
                    _build_state(
                        wid,
                        int(fold_map[wid]),
                        hw,
                        tw,
                        cut,
                        official_anchor,
                    )
                )
        except Exception as exc:
            failures.append({"well": wid, "reason": repr(exc)})
    state = pd.DataFrame(rows)
    pd.DataFrame(failures).to_csv(CACHE_DIR / "state_build_failures.csv", index=False)
    if failures:
        print("state failures:", len(failures))
        display(pd.DataFrame(failures).head(20))
    state.to_pickle(STATE_CACHE)
    return state


if STATE_CACHE.exists():
    states = pd.read_pickle(STATE_CACHE)
else:
    states = _build_state_table()

print(
    "states:",
    states.shape,
    "wells:",
    states["well"].nunique(),
    "official:",
    int(states["is_official"].sum()),
)
if states["well"].nunique() < 760 or states["is_official"].sum() < 760:
    raise RuntimeError("Too many wells missing from auxiliary state table")


target_columns = [f"target_ds_h{int(round(100*h)):03d}" for h in HORIZONS]
baseline_columns = [f"baseline_ds_h{int(round(100*h)):03d}" for h in HORIZONS]
metadata_columns = {
    "well",
    "fold",
    "cut",
    "official_anchor",
    "is_official",
}
excluded = metadata_columns | set(target_columns) | set(baseline_columns)
excluded |= {column for column in states.columns if column.startswith("target_row_")}
feature_columns = [column for column in states.columns if column not in excluded]

print("features:", len(feature_columns))


official_predictions = []
model_rows = []
training_started = time.time()

for fold in range(1, N_FOLDS + 1):
    train = states[states["fold"].ne(fold)].copy()
    valid = states[states["fold"].eq(fold) & states["is_official"]].copy()
    x_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    x_valid = valid[feature_columns].replace([np.inf, -np.inf], np.nan)
    sample_weight = np.sqrt(
        train["n_future"].to_numpy(float)
        / max(float(np.median(train["n_future"])), 1.0)
    )
    prediction_frame = valid[["well", "fold", "cut"]].copy()

    for target_column, baseline_column in zip(target_columns, baseline_columns):
        suffix = target_column.replace("target_ds_", "")
        model_path = MODEL_DIR / f"fold{fold}_{suffix}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            model = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=2500,
                learning_rate=0.025,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=35,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.2,
                reg_lambda=2.0,
                random_state=RANDOM_STATE + fold,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(
                x_train,
                train[target_column],
                sample_weight=sample_weight,
                eval_set=[(x_valid, valid[target_column])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            joblib.dump(model, model_path)
        prediction = model.predict(x_valid)
        prediction_frame[f"pred_{suffix}"] = prediction
        prediction_frame[f"true_{suffix}"] = valid[target_column].to_numpy(float)
        prediction_frame[f"baseline_{suffix}"] = valid[baseline_column].to_numpy(float)
        model_rows.append(
            {
                "fold": fold,
                "target": suffix,
                "train_states": len(train),
                "valid_wells": len(valid),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "model_rmse": _rmse(valid[target_column], prediction),
                "slope_baseline_rmse": _rmse(
                    valid[target_column], valid[baseline_column]
                ),
            }
        )
    official_predictions.append(prediction_frame)

aux_oof = pd.concat(official_predictions, ignore_index=True)
model_report = pd.DataFrame(model_rows)
model_report["gain_vs_slope"] = (
    model_report["slope_baseline_rmse"] - model_report["model_rmse"]
)


def _safe_corr(x, y, method):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or np.std(x[valid]) < 1e-9 or np.std(y[valid]) < 1e-9:
        return 0.0
    if method == "pearson":
        return float(pearsonr(x[valid], y[valid])[0])
    return float(spearmanr(x[valid], y[valid])[0])


horizon_rows = []
for horizon in HORIZONS:
    suffix = f"h{int(round(100*horizon)):03d}"
    truth = aux_oof[f"true_{suffix}"].to_numpy(float)
    prediction = aux_oof[f"pred_{suffix}"].to_numpy(float)
    baseline = aux_oof[f"baseline_{suffix}"].to_numpy(float)
    horizon_rows.append(
        {
            "horizon": horizon,
            "model_rmse": _rmse(truth, prediction),
            "slope_baseline_rmse": _rmse(truth, baseline),
            "gain_vs_slope": _rmse(truth, baseline) - _rmse(truth, prediction),
            "pearson": _safe_corr(truth, prediction, "pearson"),
            "spearman": _safe_corr(truth, prediction, "spearman"),
        }
    )
horizon_report = pd.DataFrame(horizon_rows)


# ---------------------------------------------------------------------------
# Convert the four predicted future structural points into complete curves.
# ---------------------------------------------------------------------------
pool_header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
pool_usecols = [
    column
    for column in ["id", "well", "row_idx", "tvt_true", BASE_COLUMN]
    if column in pool_header
]
pool = pd.read_csv(POOL_PATH, usecols=pool_usecols)
pool["well"] = pool["well"].astype(str)
pool["id"] = pool["id"].astype(str)
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(int)
fold_map = states.groupby("well")["fold"].first().astype(int).to_dict()
pool["fold"] = pool["well"].map(fold_map).astype(int)
aux_by_well = aux_oof.set_index("well")

curve_parts = []
for wid, group in tqdm(pool.groupby("well", sort=False), desc="build auxiliary OOF curves"):
    group = group.sort_values("row_idx").copy()
    hw = pd.read_csv(
        DATA_ROOT / "train" / f"{wid}__horizontal_well.csv",
        usecols=["Z", "TVT_input"],
    )
    tvt_input = hw["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    anchor = int(known[-1])
    evaluation = group["row_idx"].to_numpy(int)
    if not np.array_equal(evaluation, np.flatnonzero(~np.isfinite(tvt_input))):
        raise RuntimeError(f"row mismatch for {wid}")
    z = hw["Z"].to_numpy(float)
    anchor_s = float(tvt_input[anchor] + z[anchor])
    fraction = (evaluation - anchor) / max(float(evaluation[-1] - anchor), 1.0)
    prediction_row = aux_by_well.loc[wid]
    predicted_ds = np.array(
        [prediction_row[f"pred_h{int(round(100*h)):03d}"] for h in HORIZONS],
        dtype=float,
    )
    ds = np.interp(
        fraction,
        np.concatenate(([0.0], HORIZONS)),
        np.concatenate(([0.0], predicted_ds)),
    )
    group["aux_tvt"] = anchor_s + ds - z[evaluation]
    curve_parts.append(group)

curves = pd.concat(curve_parts, ignore_index=True).reset_index(drop=True)
y = curves["tvt_true"].to_numpy(float)
base_prediction = curves[BASE_COLUMN].to_numpy(float)
aux_prediction = curves["aux_tvt"].to_numpy(float)
base_rmse = _rmse(y, base_prediction)
aux_rmse = _rmse(y, aux_prediction)
base_error = base_prediction - y
aux_error = aux_prediction - y
error_cosine = float(
    np.dot(base_error, aux_error)
    / max(np.linalg.norm(base_error) * np.linalg.norm(aux_error), 1e-12)
)

bywell = []
for wid, group in curves.groupby("well", sort=False):
    error = group[BASE_COLUMN].to_numpy(float) - group["tvt_true"].to_numpy(float)
    bywell.append({"well": wid, "base_sse": float(np.sum(error * error))})
top40 = set(pd.DataFrame(bywell).nlargest(40, "base_sse")["well"].astype(str))
top40_mask = curves["well"].isin(top40).to_numpy()
top40_base = _rmse(y[top40_mask], base_prediction[top40_mask])

blend_rows = []
fold_rows = []
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
                "weight_aux": weight,
                "fold": int(fold),
                "rows": len(positions),
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

aux_target_pass = bool(
    (horizon_report["gain_vs_slope"] > 0).all()
    and (horizon_report["spearman"] >= PASS_AUX_CORR).all()
)
blend_pass = bool(blend_report["strict_gate"].any())

print("\nMODEL FOLDS / TARGETS")
display(model_report)
print("\nHORIZON OOF")
display(horizon_report)
print("\nTRAJECTORY BLEND")
display(blend_report)
print("\nTRAJECTORY FOLDS")
display(fold_report)
print(
    "\nVERDICT",
    {
        "states": len(states),
        "features": len(feature_columns),
        "base_rmse": base_rmse,
        "aux_rmse": aux_rmse,
        "error_cosine": error_cosine,
        "aux_target_pass": aux_target_pass,
        "blend_pass": blend_pass,
        "training_hours": (time.time() - training_started) / 3600.0,
        "next": (
            "feed fold-honest auxiliary predictions into the main row model"
            if aux_target_pass
            else "stop auxiliary-future target route"
        ),
    },
)

states.to_pickle(STATE_CACHE)
aux_oof.to_pickle(CACHE_DIR / "auxiliary_well_oof.pkl")
model_report.to_csv(CACHE_DIR / "model_fold_report.csv", index=False)
horizon_report.to_csv(CACHE_DIR / "horizon_oof_report.csv", index=False)
blend_report.to_csv(CACHE_DIR / "trajectory_blend_report.csv", index=False)
fold_report.to_csv(CACHE_DIR / "trajectory_fold_report.csv", index=False)
curves.to_pickle(CACHE_DIR / "auxiliary_curve_oof.pkl")
(CACHE_DIR / "verdict.json").write_text(
    json.dumps(
        {
            "base_rmse": base_rmse,
            "aux_rmse": aux_rmse,
            "error_cosine": error_cosine,
            "aux_target_pass": aux_target_pass,
            "blend_pass": blend_pass,
        },
        indent=2,
    )
)
print("saved:", CACHE_DIR)
