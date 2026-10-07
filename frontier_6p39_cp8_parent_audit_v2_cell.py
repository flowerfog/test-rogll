"""Exact CP8/12ft mechanism rebased onto the reconstructed 6.39 core.

Run after ``frontier_6p39_existing_models_oof_v0_cell.py``. This preserves the
original effective CP recipe and changes only its parent and residual target.
"""

from pathlib import Path
import json
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Fixed, pre-registered experiment. Do not tune these on this run.
# ------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
CORE_OOF_PATH = (
    OOF_DIR
    / "frontier_6p39_existing_models_oof_v0"
    / "frontier_6p39_core_oof.parquet"
)
FOLD_PATH_CANDIDATES = [
    OOF_DIR / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    Path("/root/autodl-tmp/well_metadata.csv"),
]
OUT_DIR = OOF_DIR / "frontier_6p39_cp8_parent_audit_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COL = "frontier_6p39_core"
REFERENCE_COL = "cp12"
N_CP = 8
WIN_ROWS = 192
N_SPLITS = 5
SEED = 20260711
CP_ALPHA = 1.0
CP_CAP = 12.0

SAVE_FULL_OOF_IF_PASS = True


def rmse_from_sse(sse, n):
    return float(np.sqrt(float(sse) / max(int(n), 1)))


def safe_stats(v, prefix):
    v = np.asarray(v, float)
    x = v[np.isfinite(v)]
    if len(x) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_span": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p50": float(np.quantile(x, 0.50)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
        f"{prefix}_span": float(np.max(x) - np.min(x)),
    }


print("DATA_ROOT:", DATA_ROOT)
print("POOL:", POOL_PATH)
print({
    "base": BASE_COL,
    "reference": REFERENCE_COL,
    "N_CP": N_CP,
    "WIN_ROWS": WIN_ROWS,
    "CP_ALPHA": CP_ALPHA,
    "CP_CAP": CP_CAP,
})

if not POOL_PATH.exists():
    raise RuntimeError(f"missing pool: {POOL_PATH}")
if not CORE_OOF_PATH.exists():
    raise RuntimeError(f"missing reconstructed 6.39 OOF: {CORE_OOF_PATH}")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()

# Keep only auxiliary curves that also have a known Kaggle deployment path.
aux_wanted = [
    "cp12", "ridge", "selector", "fleongg", "sp45",
]
aux_cols = [c for c in aux_wanted if c in header]

required = ["id", "tvt_true", REFERENCE_COL]
missing = [c for c in required if c not in header]
if missing:
    raise RuntimeError(f"missing required pool columns: {missing}")

usecols = [c for c in [
    "id", "well", "row_idx", "tvt_true", *aux_cols
] if c in header]

df = pd.read_csv(POOL_PATH, usecols=usecols)
df["id"] = df["id"].astype(str)

if "well" not in df.columns:
    df["well"] = df["id"].str.rsplit("_", n=1).str[0]
df["well"] = df["well"].astype(str)

if "row_idx" not in df.columns:
    df["row_idx"] = df["id"].str.rsplit("_", n=1).str[1].astype(np.int32)

df = df.sort_values(["well", "row_idx"]).reset_index(drop=True)

core = pd.read_parquet(CORE_OOF_PATH)
required_core = {"id", "truth", "fle", BASE_COL, "fold"}
if not required_core.issubset(core.columns):
    raise RuntimeError(f"Malformed 6.39 OOF columns: {list(core.columns)}")
core = core[["id", "truth", "fle", BASE_COL, "fold"]].copy()
core["id"] = core["id"].astype(str)
if core["id"].duplicated().any():
    raise RuntimeError("Duplicate 6.39 OOF ids")
df = df.merge(core, on="id", how="inner", validate="one_to_one")
if len(df) != len(core):
    raise RuntimeError(f"6.39/pool alignment incomplete: {len(df)}/{len(core)}")
truth_gap = np.abs(df["tvt_true"].to_numpy(float) - df["truth"].to_numpy(float))
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
    raise RuntimeError(f"Truth mismatch after merge: max={float(np.nanmax(truth_gap))}")
df.drop(columns="truth", inplace=True)
df["fle_gs13"] = df["fle"].astype(np.float32)
df.drop(columns="fle", inplace=True)
df = df.sort_values(["well", "row_idx"]).reset_index(drop=True)

for c in ["tvt_true", BASE_COL, "fle_gs13", *aux_cols]:
    df[c] = df[c].astype(np.float32)

print("df:", df.shape, "wells:", df["well"].nunique())
print("aux cols:", aux_cols)

y_all = df["tvt_true"].to_numpy(np.float32)
base_all = df[BASE_COL].to_numpy(np.float32)
cp12_all = df[REFERENCE_COL].to_numpy(np.float32)

print("frontier 6.39 core parent RMSE:", float(np.sqrt(np.mean((y_all - base_all) ** 2))))
print("cp12 RMSE  :", float(np.sqrt(np.mean((y_all - cp12_all) ** 2))))

# ------------------------------------------------------------------
# Raw horizontal-well features, cached once per well.
# ------------------------------------------------------------------
well_cache = {}


def load_hw_features(wid):
    if wid in well_cache:
        return well_cache[wid]

    path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    hw = pd.read_csv(path)
    for c in ["MD", "X", "Y", "Z", "GR", "TVT_input"]:
        if c not in hw.columns:
            hw[c] = np.nan

    md = hw["MD"].to_numpy(float)
    x = hw["X"].to_numpy(float)
    y = hw["Y"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    gr = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)

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

    gr_fill = pd.Series(gr).interpolate(limit_direction="both").fillna(0).to_numpy(float)
    gr_d1 = np.gradient(gr_fill)
    gr_d2 = np.gradient(gr_d1)

    known = np.flatnonzero(np.isfinite(tvt_input))
    last_known_idx = int(known[-1]) if len(known) else 0
    n = len(hw)

    out = {
        "n_total": n,
        "MD": md,
        "X": x,
        "Y": y,
        "Z": z,
        "GR": gr,
        "build": build,
        "dogleg": dogleg,
        "gr_d1": gr_d1,
        "gr_d2": gr_d2,
        "pos": np.arange(n, dtype=float) / max(n - 1, 1),
        "since_anchor_norm": (
            np.arange(n, dtype=float) - last_known_idx
        ) / max(n - last_known_idx - 1, 1),
        "prefix_n": int(len(known)),
        "last_known_idx": last_known_idx,
    }
    well_cache[wid] = out
    return out


# ------------------------------------------------------------------
# One row per control point. Target is local mean residual to strong.
# ------------------------------------------------------------------
records = []
caches = {}
t0 = time.time()

base_vec = df[BASE_COL].to_numpy(np.float32)
truth_vec = df["tvt_true"].to_numpy(np.float32)
resid_vec = truth_vec - base_vec
row_idx_vec = df["row_idx"].to_numpy(np.int32)
candidate_arrays = {c: df[c].to_numpy(np.float32) for c in aux_cols}

group_items = list(df.groupby("well", sort=False).indices.items())

for wi, (wid, idx) in enumerate(group_items, 1):
    idx = np.asarray(idx, dtype=np.int64)
    n = len(idx)
    if n < 32:
        continue

    hwf = load_hw_features(wid)
    ridx = row_idx_vec[idx]
    if ridx.min() < 0 or ridx.max() >= hwf["n_total"]:
        continue

    centers = np.unique(np.round(np.linspace(0, n - 1, N_CP)).astype(int))
    row_norm = np.arange(n, dtype=float) / max(n - 1, 1)
    base_well = base_vec[idx]

    caches[wid] = {
        "idx": idx,
        "n": n,
    }

    for order, center_local in enumerate(centers):
        lo = max(0, center_local - WIN_ROWS)
        hi = min(n, center_local + WIN_ROWS + 1)
        local_positions = np.arange(lo, hi)
        loc = idx[local_positions]
        ridx_win = ridx[local_positions]
        center_global = int(idx[center_local])
        center_ridx = int(ridx[center_local])
        base_window = base_vec[loc]

        feat = {
            "well": wid,
            "center_global": center_global,
            "center_local": int(center_local),
            "n_well": int(n),
            "n_window": int(len(loc)),
            "cp_order": int(order),
            "cp_order_norm": float(order / max(len(centers) - 1, 1)),
            "row_norm": float(row_norm[center_local]),
            "target_resid": float(np.nanmean(resid_vec[loc])),
            "prefix_n": hwf["prefix_n"],
            "center_GR": float(np.nan_to_num(hwf["GR"][center_ridx], nan=0.0)),
            "center_Z": float(np.nan_to_num(hwf["Z"][center_ridx], nan=0.0)),
            "center_build": float(np.nan_to_num(hwf["build"][center_ridx], nan=0.0)),
            "center_since_anchor_norm": float(
                np.nan_to_num(hwf["since_anchor_norm"][center_ridx], nan=0.0)
            ),
            "base_center": float(base_vec[center_global]),
            "base_slope_win": float(
                (base_window[-1] - base_window[0]) / max(len(base_window) - 1, 1)
            ),
        }

        for col in [
            "MD", "X", "Y", "Z", "GR", "build", "dogleg",
            "gr_d1", "gr_d2", "pos", "since_anchor_norm",
        ]:
            feat.update(safe_stats(hwf[col][ridx_win], col))

        feat.update(safe_stats(base_window, "base"))

        pred_stack = np.column_stack([candidate_arrays[c][loc] for c in aux_cols])
        feat["pred_std_mean"] = float(np.mean(np.std(pred_stack, axis=1)))
        feat["pred_range_mean"] = float(
            np.mean(np.max(pred_stack, axis=1) - np.min(pred_stack, axis=1))
        )

        for c in aux_cols:
            diff = candidate_arrays[c][loc] - base_window
            feat.update(safe_stats(diff, f"{c}_minus_base"))
            feat[f"{c}_minus_base_center"] = float(
                candidate_arrays[c][center_global] - base_vec[center_global]
            )
            feat[f"{c}_minus_base_slope"] = float(
                (diff[-1] - diff[0]) / max(len(diff) - 1, 1)
            )

        records.append(feat)

    if wi % 50 == 0:
        print(f"features {wi}/{len(group_items)} elapsed={time.time()-t0:.1f}s", flush=True)

cp = pd.DataFrame(records)
print("CP table:", cp.shape)

# Persist the compact deployment-training table. Notebook experiments reuse
# short global names such as ``cp`` and ``X``; this artifact makes the final
# refit independent of later live-kernel state.
cp_table_path = OUT_DIR / "cp_training_table.parquet"
cp.to_parquet(cp_table_path, index=False)
print("saved CP training table:", cp_table_path, flush=True)

feature_cols = [
    c for c in cp.columns
    if c not in {"well", "center_global", "center_local", "target_resid"}
    and pd.api.types.is_numeric_dtype(cp[c])
]

X = (
    cp[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)
y_cp = cp["target_resid"].to_numpy(np.float32)
groups = cp["well"].astype(str).to_numpy()

fold_path = next((path for path in FOLD_PATH_CANDIDATES if path.exists()), None)
if fold_path is None:
    raise FileNotFoundError(f"Missing frozen fold metadata: {FOLD_PATH_CANDIDATES}")
folds = pd.read_csv(fold_path, usecols=["well", "fold"])
folds["well"] = folds["well"].astype(str)
if folds["well"].duplicated().any() or set(folds["fold"].astype(int)) != set(range(1, 6)):
    raise RuntimeError("Malformed frozen fold metadata")
frozen_fold_map = dict(zip(folds["well"], folds["fold"].astype(int)))
cp_folds = cp["well"].map(frozen_fold_map)
if cp_folds.isna().any():
    raise RuntimeError("Frozen folds do not cover every CP well")
cp_folds = cp_folds.to_numpy(np.int8)

print("features:", len(feature_cols))

# ------------------------------------------------------------------
# Honest well-grouped OOF model.
# ------------------------------------------------------------------
oof_cp = np.full(len(cp), np.nan, dtype=np.float32)
well_to_fold = {}
fold_model_rows = []

for fold in range(1, N_SPLITS + 1):
    tr = np.flatnonzero(cp_folds != fold)
    va = np.flatnonzero(cp_folds == fold)
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(f"Empty frozen CP split {fold}: {len(tr)}/{len(va)}")
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
        X.iloc[tr],
        y_cp[tr],
        eval_set=[(X.iloc[va], y_cp[va])],
        eval_metric="rmse",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    oof_cp[va] = model.predict(X.iloc[va]).astype(np.float32)
    valid_wells = np.unique(groups[va])
    for wid in valid_wells:
        well_to_fold[str(wid)] = fold

    cp_rmse = float(np.sqrt(np.mean((oof_cp[va] - y_cp[va]) ** 2)))
    fold_model_rows.append({
        "fold": fold,
        "train_cp_rows": len(tr),
        "valid_cp_rows": len(va),
        "valid_wells": len(valid_wells),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "cp_target_rmse": cp_rmse,
    })
    print("fold model:", fold_model_rows[-1], flush=True)

if not np.isfinite(oof_cp).all():
    raise RuntimeError("non-finite OOF control predictions")

cp["oof_pred_resid"] = oof_cp

# ------------------------------------------------------------------
# Interpolate control predictions back to all OOF rows.
# ------------------------------------------------------------------
raw_move = np.zeros(len(df), dtype=np.float32)

for wid, cache in caches.items():
    idx = cache["idx"]
    n = cache["n"]
    sub = cp[cp["well"].eq(wid)].sort_values("center_local")

    xp = sub["center_local"].to_numpy(float)
    fp = sub["oof_pred_resid"].to_numpy(float)
    keep = np.r_[True, np.diff(xp) > 0]
    xp = xp[keep]
    fp = fp[keep]

    if len(xp) == 1:
        move = np.full(n, fp[0], dtype=float)
    else:
        move = np.interp(np.arange(n, dtype=float), xp, fp)

    sigma = max(4.0, n / N_CP / 8.0)
    move = gaussian_filter1d(move, sigma=sigma, mode="nearest")
    raw_move[idx] = move.astype(np.float32)

move = np.clip(CP_ALPHA * raw_move, -CP_CAP, CP_CAP)
candidate = base_all + move

# ------------------------------------------------------------------
# Fixed-config pooled, fold and top40 diagnostics.
# ------------------------------------------------------------------
well_rows = []
for wid, idx in df.groupby("well", sort=False).indices.items():
    idx = np.asarray(idx, dtype=np.int64)
    y = y_all[idx].astype(float)
    strong = base_all[idx].astype(float)
    cp12 = cp12_all[idx].astype(float)
    pred = candidate[idx].astype(float)

    well_rows.append({
        "well": str(wid),
        "fold": int(well_to_fold[str(wid)]),
        "n": len(idx),
        "strong_sse": float(np.sum((strong - y) ** 2)),
        "cp12_sse": float(np.sum((cp12 - y) ** 2)),
        "candidate_sse": float(np.sum((pred - y) ** 2)),
        "sum_abs_move": float(np.sum(np.abs(move[idx]))),
        "p95_abs_move": float(np.quantile(np.abs(move[idx]), 0.95)),
    })

well_df = pd.DataFrame(well_rows)


def summarize(g):
    n = int(g["n"].sum())
    strong_rmse = rmse_from_sse(g["strong_sse"].sum(), n)
    cp12_rmse = rmse_from_sse(g["cp12_sse"].sum(), n)
    candidate_rmse = rmse_from_sse(g["candidate_sse"].sum(), n)
    return {
        "n": n,
        "strong_rmse": strong_rmse,
        "cp12_rmse": cp12_rmse,
        "candidate_rmse": candidate_rmse,
        "gain_vs_parent": strong_rmse - candidate_rmse,
        "gain_vs_cp12": cp12_rmse - candidate_rmse,
        "mean_abs_move": float(g["sum_abs_move"].sum() / n),
    }


overall = summarize(well_df)
fold_df = pd.DataFrame([
    {"fold": fold, **summarize(g)}
    for fold, g in well_df.groupby("fold", sort=True)
])

top40 = well_df.sort_values("strong_sse", ascending=False).head(40)
top40_summary = summarize(top40)

print("\nFIXED CONFIG OVERALL")
print(overall)
print("\nFIXED CONFIG FOLDS")
print(fold_df.to_string(index=False))
print("\nTOP40")
print(top40_summary)

passed = (
    overall["gain_vs_parent"] >= 0.08
    and int((fold_df["gain_vs_parent"] > 0).sum()) == 5
    and float(fold_df["gain_vs_parent"].min()) > 0
    and top40_summary["gain_vs_parent"] > 0
)

print("\n" + "=" * 72)
print("EXACT-BASE CP-FAMILY:", "PASS" if passed else "FAIL / STOP")
print({
    "gain_vs_parent": overall["gain_vs_parent"],
    "positive_folds": int((fold_df["gain_vs_parent"] > 0).sum()),
    "min_fold_gain_vs_parent": float(fold_df["gain_vs_parent"].min()),
    "top40_gain_vs_parent": top40_summary["gain_vs_parent"],
})
print("=" * 72)

# ------------------------------------------------------------------
# Save compact diagnostics. Save the 3.8M-row file only if it passed.
# ------------------------------------------------------------------
pd.DataFrame(fold_model_rows).to_csv(
    OUT_DIR / "model_folds.csv", index=False
)
fold_df.to_csv(
    OUT_DIR / "score_folds.csv", index=False
)
well_df.to_csv(
    OUT_DIR / "bywell.csv", index=False
)

out = df[["id", "well", "row_idx", "tvt_true", "sp45", "fle_gs13", BASE_COL]].copy()
out["fold"] = out["well"].map(frozen_fold_map).astype(np.int8)
out["cp8_rebased"] = candidate.astype(np.float32)
out["raw_move"] = raw_move.astype(np.float32)
out["capped_move"] = move.astype(np.float32)
out.to_pickle(OUT_DIR / "cp8_rebased_oof.pkl")

verdict = {
    "experiment": "frontier_6p39_cp8_parent_audit_v2",
    "stage": "fivefold_exact_original_CP8_mechanism_on_reconstructed_6p39_core",
    "parent": "frontier_6p39_core reconstructed from fold-clean SP45/FLE OOF",
    "core_oof": str(CORE_OOF_PATH),
    "pool": str(POOL_PATH),
    "fold_source": str(fold_path),
    "fixed_recipe": {
        "control_points": N_CP,
        "window_rows": WIN_ROWS,
        "objective": "huber",
        "alpha": CP_ALPHA,
        "cap_ft": CP_CAP,
    },
    "q0522_included": False,
    "q0522_reason": "hidden-well-specific LB edit; not a fold-clean transform",
    "overall": overall,
    "folds": fold_df.to_dict(orient="records"),
    "top40_parent_error_wells": top40_summary,
    "gates": {
        "gain_vs_parent_ge_0p08": bool(overall["gain_vs_parent"] >= 0.08),
        "positive_folds_5of5": bool((fold_df["gain_vs_parent"] > 0).all()),
        "top40_gain_positive": bool(top40_summary["gain_vs_parent"] > 0),
    },
    "strict_pass": bool(passed),
    "free_gb": shutil.disk_usage(Path("/root/autodl-tmp")).free / (1024**3),
    "next": (
        "PASS -> freeze CP8 move, then test CP->Neighbor and Neighbor->CP on a held-out fold"
        if passed
        else "FAIL -> original CP8/12ft gain does not transfer to the reconstructed 6.39 parent"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved diagnostics to:", OUT_DIR)
