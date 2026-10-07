"""Rebuild the well-level Huber bias bundle used by strong_bias.

This runs on AutoDL. It trains only the bias layer; it does not rebuild the
SP45/fleongg strong curve. The existing honest OOF pool is used as the source
of strong OOF predictions and true labels, and as a lineage check against the
historical strong_bias column when that column is present.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_DIR = ROOT / "oof_exports"
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
RUN_DIR = OOF_DIR / "strong_bias_huber_rebuild_v1"
BUNDLE_PATH = RUN_DIR / "strong_bias_huber_bundle_v1.joblib"

# Historical pool.strong_bias lineage. The a0.5/c6 CSV found in the old
# directory inventory was a separate probe; the source cell and OOF movement
# scale both identify alpha=0.70 for the authoritative pool column.
BIAS_ALPHA = 0.70
BIAS_CAP = 6.0
MODEL_ALPHA = 0.001
MODEL_EPSILON = 1.50
MODEL_MAX_ITER = 500


def _rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _safe_stats(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.10)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
    }


def _robust_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 8 or np.max(x) <= np.min(x):
        return 0.0
    x = x - x[-1]
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def _build_well_features(wells, split):
    rows = []
    for index, wid in enumerate(wells, 1):
        try:
            hw = pd.read_csv(DATA_ROOT / split / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(DATA_ROOT / split / f"{wid}__typewell.csv")

            tvt_input = hw["TVT_input"].to_numpy(float)
            known = np.flatnonzero(np.isfinite(tvt_input))
            evaluation = np.flatnonzero(~np.isfinite(tvt_input))
            row = {"well": str(wid)}

            if len(known) == 0 or len(evaluation) == 0:
                rows.append(row)
                continue

            before_eval = known[known < evaluation[0]]
            last_index = int(before_eval[-1]) if len(before_eval) else int(known[-1])
            last_tvt = float(tvt_input[last_index])

            md = hw["MD"].to_numpy(float)
            z = hw["Z"].to_numpy(float)
            gr = (
                hw["GR"].to_numpy(float)
                if "GR" in hw.columns
                else np.full(len(hw), np.nan)
            )

            row["known_len"] = float(len(known))
            row["eval_len"] = float(len(evaluation))
            row["last_tvt"] = last_tvt
            row["last_z"] = float(z[last_index])
            row["last_u"] = float(last_tvt + z[last_index])
            row["eval_md_len"] = float(md[evaluation[-1]] - md[evaluation[0]])
            row["eval_md_since"] = float(md[evaluation[-1]] - md[last_index])
            row["eval_z_delta"] = float(z[evaluation[-1]] - z[last_index])
            row["eval_z_span"] = float(
                np.nanmax(z[evaluation]) - np.nanmin(z[evaluation])
            )
            row["eval_gr_cov"] = float(np.isfinite(gr[evaluation]).mean())
            row["known_gr_cov"] = float(np.isfinite(gr[known]).mean())

            tail50 = known[-min(50, len(known)) :]
            tail200 = known[-min(200, len(known)) :]
            row["slope_tvt_tail50"] = _robust_slope(md[tail50], tvt_input[tail50])
            row["slope_tvt_tail200"] = _robust_slope(md[tail200], tvt_input[tail200])
            row["slope_u_tail50"] = _robust_slope(
                md[tail50], tvt_input[tail50] + z[tail50]
            )
            row["slope_u_tail200"] = _robust_slope(
                md[tail200], tvt_input[tail200] + z[tail200]
            )
            row["slope_z_eval"] = _robust_slope(md[evaluation], z[evaluation])

            for values, prefix in (
                (gr[known], "known_gr"),
                (gr[tail200], "tail200_gr"),
                (gr[evaluation], "eval_gr"),
                (z[evaluation] - z[last_index], "eval_dz"),
            ):
                row.update(_safe_stats(values, prefix))

            if {"TVT", "GR"}.issubset(tw.columns):
                tw_clean = tw[["TVT", "GR"]].dropna().sort_values("TVT")
                if len(tw_clean):
                    tw_tvt = tw_clean["TVT"].to_numpy(float)
                    tw_gr = tw_clean["GR"].to_numpy(float)
                    tw_range = max(float(np.ptp(tw_tvt)), 1e-6)
                    row["tw_tvt_min"] = float(np.min(tw_tvt))
                    row["tw_tvt_max"] = float(np.max(tw_tvt))
                    row["tw_tvt_range"] = float(np.ptp(tw_tvt))
                    row["last_tvt_rel_tw"] = float(
                        (last_tvt - np.min(tw_tvt)) / tw_range
                    )
                    for offset in (-120, -60, -30, 0, 30, 60, 120):
                        row[f"tw_gr_at_{offset}"] = float(
                            np.interp(last_tvt + offset, tw_tvt, tw_gr)
                        )

            rows.append(row)
        except Exception as exc:
            print("bias feature fallback", wid, repr(exc))
            rows.append({"well": str(wid)})

        if index % 100 == 0:
            print(f"bias features {index}/{len(wells)}")

    return (
        pd.DataFrame(rows)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


for required in (DATA_ROOT / "train", POOL_PATH):
    if not required.exists():
        raise FileNotFoundError(required)
RUN_DIR.mkdir(parents=True, exist_ok=True)

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
required_columns = {"id", "tvt_true", "strong"}
missing = required_columns - set(header)
if missing:
    raise RuntimeError(f"Pool is missing required columns: {sorted(missing)}")

usecols = ["id", "tvt_true", "strong"]
for optional in ("well", "fold", "strong_bias"):
    if optional in header:
        usecols.append(optional)

print("loading:", POOL_PATH)
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": str, "tvt_true": np.float32, "strong": np.float32},
)
if "well" not in pool:
    pool["well"] = pool["id"].astype(str).str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
pool["residual"] = pool["tvt_true"].astype(float) - pool["strong"].astype(float)
pool["abs_residual"] = pool["residual"].abs()

aggregate_spec = {
    "n": ("id", "size"),
    "bias": ("residual", "mean"),
    "rmse": ("residual", lambda x: _rmse(np.zeros(len(x)), x)),
    "p95_abs": ("abs_residual", lambda x: float(np.quantile(x, 0.95))),
}
if "fold" in pool:
    aggregate_spec["fold"] = ("fold", "first")
target = pool.groupby("well", sort=False).agg(**aggregate_spec).reset_index()

wells = target["well"].astype(str).tolist()
features = _build_well_features(wells, "train")
train = target.merge(features, on="well", how="left", validate="one_to_one")
train = train.replace([np.inf, -np.inf], np.nan).fillna(0)

if "fold" not in train or train["fold"].nunique() < 5:
    # Deterministic fallback used only when the pool has no fold assignment.
    train = train.sort_values("well").reset_index(drop=True)
    train["fold"] = np.arange(len(train)) % 5 + 1
train["fold"] = train["fold"].astype(int)
well_fold_map = dict(zip(train["well"].astype(str), train["fold"].astype(int)))
pool["_eval_fold"] = pool["well"].map(well_fold_map)
if pool["_eval_fold"].isna().any():
    raise RuntimeError("Failed to map fallback folds back to OOF rows")
pool["_eval_fold"] = pool["_eval_fold"].astype(int)

excluded = {"well", "bias", "rmse", "p95_abs", "fold"}
feature_columns = [
    column
    for column in train.columns
    if column not in excluded and pd.api.types.is_numeric_dtype(train[column])
]
print({"train_wells": len(train), "features": len(feature_columns)})


def _new_model():
    return make_pipeline(
        RobustScaler(),
        HuberRegressor(
            alpha=MODEL_ALPHA,
            epsilon=MODEL_EPSILON,
            max_iter=MODEL_MAX_ITER,
        ),
    )


oof_raw = np.full(len(train), np.nan, dtype=np.float64)
fold_rows = []
for fold in sorted(train["fold"].unique()):
    train_mask = train["fold"].ne(fold).to_numpy()
    valid_mask = train["fold"].eq(fold).to_numpy()
    model = _new_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            train.loc[train_mask, feature_columns].to_numpy(float),
            train.loc[train_mask, "bias"].to_numpy(float),
        )
    oof_raw[valid_mask] = model.predict(
        train.loc[valid_mask, feature_columns].to_numpy(float)
    )
    fold_rows.append(
        {
            "fold": int(fold),
            "wells": int(valid_mask.sum()),
            "bias_target_rmse": _rmse(
                train.loc[valid_mask, "bias"], oof_raw[valid_mask]
            ),
        }
    )

if not np.isfinite(oof_raw).all():
    raise RuntimeError("OOF bias predictions are incomplete")
train["bias_pred_raw"] = oof_raw
train["bias_move"] = np.clip(BIAS_ALPHA * oof_raw, -BIAS_CAP, BIAS_CAP)
move_map = dict(zip(train["well"], train["bias_move"]))
row_move = pool["well"].map(move_map).to_numpy(float)
reconstructed = pool["strong"].to_numpy(float) + row_move

base_rmse = _rmse(pool["tvt_true"], pool["strong"])
candidate_rmse = _rmse(pool["tvt_true"], reconstructed)
for row in fold_rows:
    mask = pool["_eval_fold"].eq(row["fold"]).to_numpy()
    row["base_rmse"] = _rmse(
        pool.loc[mask, "tvt_true"], pool.loc[mask, "strong"]
    )
    row["candidate_rmse"] = _rmse(
        pool.loc[mask, "tvt_true"], reconstructed[mask]
    )
    row["gain"] = row["base_rmse"] - row["candidate_rmse"]

reference_check = None
lineage_match = False
if "strong_bias" in pool:
    reference = pool["strong_bias"].to_numpy(float)
    difference = reconstructed - reference
    reference_check = {
        "reference_rmse": _rmse(pool["tvt_true"], reference),
        "reconstructed_rmse": candidate_rmse,
        "curve_rmse_diff": float(np.sqrt(np.mean(difference**2))),
        "mean_abs_diff": float(np.mean(np.abs(difference))),
        "p95_abs_diff": float(np.quantile(np.abs(difference), 0.95)),
        "max_abs_diff": float(np.max(np.abs(difference))),
    }
    lineage_match = bool(
        reference_check["curve_rmse_diff"] <= 0.50
        and abs(
            reference_check["reconstructed_rmse"]
            - reference_check["reference_rmse"]
        )
        <= 0.03
    )

fold_frame = pd.DataFrame(fold_rows)
positive_folds = int((fold_frame.get("gain", pd.Series(dtype=float)) > 0).sum())
min_fold_gain = float(fold_frame["gain"].min()) if "gain" in fold_frame else float("nan")
performance_pass = bool(
    base_rmse - candidate_rmse >= 0.10
    and positive_folds == 5
    and min_fold_gain > 0
)
strict_pass = bool(performance_pass and lineage_match)

print("\nFOLDS")
display(fold_frame)
print("\nOVERALL")
summary = {
    "base_rmse": base_rmse,
    "candidate_rmse": candidate_rmse,
    "gain": base_rmse - candidate_rmse,
    "positive_folds": positive_folds,
    "min_fold_gain": min_fold_gain,
    "mean_abs_move": float(np.mean(np.abs(row_move))),
    "p95_abs_move": float(np.quantile(np.abs(row_move), 0.95)),
    "max_abs_move": float(np.max(np.abs(row_move))),
    "reference_check": reference_check,
    "lineage_match": lineage_match,
    "performance_pass": performance_pass,
    "strict_pass": strict_pass,
}
print(summary)

full_model = _new_model()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    full_model.fit(
        train[feature_columns].to_numpy(float),
        train["bias"].to_numpy(float),
    )

bundle = {
    "version": "strong_bias_huber_bundle_v1",
    "model": full_model,
    "feature_columns": feature_columns,
    "bias_alpha": BIAS_ALPHA,
    "bias_cap": BIAS_CAP,
    "model_alpha": MODEL_ALPHA,
    "model_epsilon": MODEL_EPSILON,
    "model_max_iter": MODEL_MAX_ITER,
    "training_wells": len(train),
    "oof_summary": summary,
    "feature_builder": "visible_well_features_v1",
}
joblib.dump(bundle, BUNDLE_PATH, compress=3)

train.to_csv(RUN_DIR / "strong_bias_train_well_features.csv", index=False)
fold_frame.to_csv(RUN_DIR / "strong_bias_rebuild_folds.csv", index=False)
pd.DataFrame(
    {
        "id": pool["id"].astype(str),
        "well": pool["well"].astype(str),
        "tvt_true": pool["tvt_true"].to_numpy(float),
        "strong": pool["strong"].to_numpy(float),
        "strong_bias_rebuilt": reconstructed,
        "bias_move": row_move,
    }
).to_csv(RUN_DIR / "strong_bias_rebuilt_oof.csv", index=False)
(RUN_DIR / "strong_bias_rebuild_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\nsaved bundle:", BUNDLE_PATH)
print("bundle_mb:", BUNDLE_PATH.stat().st_size / 2**20)
if not strict_pass:
    print("WARNING: reconstruction did not pass the OOF gate; do not blend on Kaggle yet.")
