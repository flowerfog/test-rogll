from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


ROOT = Path.cwd()
REMOTE_DATA = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
DATA_ROOT = REMOTE_DATA if (REMOTE_DATA / "train").exists() else ROOT
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
N_FOLDS = 5
N_QUERY_TRAIN = 20
N_QUERY_VALID = 28
SEED = 20260711


def slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def prefix_features(md, x, y, z, gr, datum, anchor):
    out = {
        "anchor_md": md[anchor] / 10000.0,
        "anchor_x": x[anchor] / 1_000_000.0,
        "anchor_y": y[anchor] / 1_000_000.0,
        "anchor_z": z[anchor] / 10000.0,
        "anchor_datum": datum[anchor] / 1000.0,
        "prefix_frac": anchor / max(len(md) - 1, 1),
    }
    for w in (50, 200, 500, 1000):
        lo = max(0, anchor - w + 1)
        sl = slice(lo, anchor + 1)
        out[f"datum_slope_{w}"] = slope(md[sl], datum[sl])
        out[f"z_slope_{w}"] = slope(md[sl], z[sl])
        out[f"x_slope_{w}"] = slope(md[sl], x[sl])
        out[f"y_slope_{w}"] = slope(md[sl], y[sl])
        dd = np.diff(datum[sl])
        out[f"datum_d_std_{w}"] = float(np.nanstd(dd)) if len(dd) else 0.0
        gg = gr[sl]
        out[f"gr_mean_{w}"] = float(np.nanmean(gg)) if np.isfinite(gg).any() else 0.0
        out[f"gr_std_{w}"] = float(np.nanstd(gg)) if np.isfinite(gg).any() else 0.0
    return out


def query_feature(prefix, md, x, y, z, gr, anchor, q):
    dx = x[q] - x[anchor]
    dy = y[q] - y[anchor]
    delta_md = md[q] - md[anchor]
    remaining = max(md[-1] - md[anchor], 1e-6)
    row = dict(prefix)
    row.update({
        "delta_md": delta_md / 1000.0,
        "delta_x": dx / 1000.0,
        "delta_y": dy / 1000.0,
        "delta_z": (z[q] - z[anchor]) / 1000.0,
        "horizontal_delta": np.hypot(dx, dy) / 1000.0,
        "future_frac": delta_md / remaining,
        "current_x": x[q] / 1_000_000.0,
        "current_y": y[q] / 1_000_000.0,
        "current_z": z[q] / 10000.0,
        "current_gr": float(gr[q] / 100.0) if np.isfinite(gr[q]) else 0.0,
    })
    return row


files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
train_rows = []
valid_rows = []
well_cache = {}
t0 = time.time()

for i, path in enumerate(files, 1):
    wid = path.name.split("__")[0]
    d = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"])
    md = d["MD"].to_numpy(float)
    x = d["X"].to_numpy(float)
    y = d["Y"].to_numpy(float)
    z = d["Z"].to_numpy(float)
    gr = d["GR"].to_numpy(float)
    tvt = d["TVT"].to_numpy(float)
    datum = tvt + z
    known = np.flatnonzero(d["TVT_input"].notna().to_numpy())
    hidden = np.flatnonzero(d["TVT_input"].isna().to_numpy())
    if len(known) < 100 or len(hidden) < 50:
        continue
    actual_anchor = int(known[-1])
    cuts = np.unique([
        max(50, actual_anchor - 1000),
        max(50, actual_anchor - 500),
        actual_anchor,
    ])

    for cut in cuts:
        pref = prefix_features(md, x, y, z, gr, datum, int(cut))
        qs = np.unique(np.round(np.linspace(cut, len(d) - 1, N_QUERY_TRAIN)).astype(int))
        for q in qs:
            row = query_feature(pref, md, x, y, z, gr, int(cut), int(q))
            row["well"] = wid
            row["target"] = float(datum[q] - datum[cut])
            train_rows.append(row)

    pref = prefix_features(md, x, y, z, gr, datum, actual_anchor)
    qs = np.unique(np.r_[
        actual_anchor,
        np.round(np.linspace(hidden[0], hidden[-1], N_QUERY_VALID)).astype(int),
    ])
    for q in qs:
        row = query_feature(pref, md, x, y, z, gr, actual_anchor, int(q))
        row.update({"well": wid, "q": int(q), "anchor": actual_anchor})
        valid_rows.append(row)

    well_cache[wid] = {
        "hidden": hidden,
        "z": z,
        "tvt": tvt,
        "datum_anchor": float(datum[actual_anchor]),
    }
    if i % 100 == 0:
        print(i, "/", len(files), "elapsed", time.time() - t0)

train = pd.DataFrame(train_rows)
valid = pd.DataFrame(valid_rows)
feature_cols = [c for c in train.columns if c not in {"well", "target"}]
X = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
target = train["target"].to_numpy(np.float32)
print("train", train.shape, "valid controls", valid.shape, "features", len(feature_cols))

wells = np.array(sorted(well_cache))
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_rows = []
pred_rows = []
all_sse = 0.0
all_const_sse = 0.0
all_n = 0

for fold, (tr_wi, va_wi) in enumerate(kf.split(wells), 1):
    tr_wells = set(wells[tr_wi])
    va_wells = set(wells[va_wi])
    tr_mask = train["well"].isin(tr_wells).to_numpy()
    va_control = valid[valid["well"].isin(va_wells)].copy()

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=1.0,
        random_state=SEED + fold,
    )
    model.fit(X.loc[tr_mask], target[tr_mask])
    va_control["pred_drift"] = model.predict(
        va_control[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    fold_sse = 0.0
    fold_const_sse = 0.0
    fold_n = 0
    for wid, cg in va_control.groupby("well", sort=False):
        cache = well_cache[wid]
        cg = cg.sort_values("q")
        q = cg["q"].to_numpy(int)
        drift = cg["pred_drift"].to_numpy(float)
        # Exact continuity at the real prediction start.
        drift = drift - drift[0]
        hidden = cache["hidden"]
        drift_full = np.interp(hidden, q, drift)
        pred = cache["datum_anchor"] + drift_full - cache["z"][hidden]
        const = cache["datum_anchor"] - cache["z"][hidden]
        truth = cache["tvt"][hidden]
        fold_sse += float(np.sum((pred - truth) ** 2))
        fold_const_sse += float(np.sum((const - truth) ** 2))
        fold_n += len(hidden)
        pred_rows.append(pd.DataFrame({
            "well": wid,
            "row_idx": hidden.astype(np.int32),
            "fold": fold,
            "datum_model": pred.astype(np.float32),
        }))
    fold_rows.append({
        "fold": fold,
        "constant_anchor_rmse": float(np.sqrt(fold_const_sse / fold_n)),
        "model_rmse": float(np.sqrt(fold_sse / fold_n)),
        "gain": float(np.sqrt(fold_const_sse / fold_n) - np.sqrt(fold_sse / fold_n)),
        "rows": fold_n,
    })
    all_sse += fold_sse
    all_const_sse += fold_const_sse
    all_n += fold_n
    print("fold", fold, fold_rows[-1], flush=True)

print(pd.DataFrame(fold_rows).to_string(index=False))
print({
    "constant_anchor_rmse": float(np.sqrt(all_const_sse / all_n)),
    "model_rmse": float(np.sqrt(all_sse / all_n)),
    "gain": float(np.sqrt(all_const_sse / all_n) - np.sqrt(all_sse / all_n)),
    "rows": all_n,
})

# Optional exact comparison with CP12 when the merged OOF pool is available.
pool_candidates = [
    OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_DIR / "candidate_pool_full_merged_oof.csv",
]
pool_path = next((p for p in pool_candidates if p.exists()), None)

if pool_path is not None:
    pred_df = pd.concat(pred_rows, ignore_index=True)
    header = pd.read_csv(pool_path, nrows=0).columns.tolist()
    usecols = [c for c in ["id", "well", "row_idx", "tvt_true", "cp12"] if c in header]
    pool = pd.read_csv(pool_path, usecols=usecols)
    pool["id"] = pool["id"].astype(str)
    if "well" not in pool:
        pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
    pool["well"] = pool["well"].astype(str)
    if "row_idx" not in pool:
        pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[1].astype(np.int32)

    merged = pool.merge(pred_df, on=["well", "row_idx"], how="inner")
    if len(merged) != len(pool):
        raise RuntimeError(f"prediction merge mismatch: pool={len(pool)} merged={len(merged)}")

    # Fixed, pre-registered diversification weight. Do not tune on this run.
    merged["blend_cp12_datum10"] = (
        0.90 * merged["cp12"].astype(float)
        + 0.10 * merged["datum_model"].astype(float)
    )

    score_rows = []
    for (fold, wid), g in merged.groupby(["fold", "well"], sort=False):
        yv = g["tvt_true"].to_numpy(float)
        cpv = g["cp12"].to_numpy(float)
        bv = g["blend_cp12_datum10"].to_numpy(float)
        score_rows.append({
            "fold": int(fold),
            "well": str(wid),
            "n": len(g),
            "cp12_sse": float(np.sum((cpv - yv) ** 2)),
            "blend_sse": float(np.sum((bv - yv) ** 2)),
            "sum_abs_move": float(np.sum(np.abs(bv - cpv))),
        })

    bywell = pd.DataFrame(score_rows)

    def summarize(g):
        n = int(g["n"].sum())
        cp_rmse = float(np.sqrt(g["cp12_sse"].sum() / n))
        blend_rmse = float(np.sqrt(g["blend_sse"].sum() / n))
        return {
            "n": n,
            "cp12_rmse": cp_rmse,
            "blend_rmse": blend_rmse,
            "gain_vs_cp12": cp_rmse - blend_rmse,
            "mean_abs_move": float(g["sum_abs_move"].sum() / n),
        }

    blend_folds = pd.DataFrame([
        {"fold": fold, **summarize(g)}
        for fold, g in bywell.groupby("fold", sort=True)
    ])
    blend_overall = summarize(bywell)
    top40 = bywell.sort_values("cp12_sse", ascending=False).head(40)
    top40_summary = summarize(top40)

    print("\nFIXED CP12 90% + DATUM 10% OVERALL")
    print(blend_overall)
    print("\nFOLDS")
    print(blend_folds.to_string(index=False))
    print("\nTOP40")
    print(top40_summary)

    passed = (
        blend_overall["gain_vs_cp12"] >= 0.03
        and int((blend_folds["gain_vs_cp12"] > 0).sum()) == 5
        and float(blend_folds["gain_vs_cp12"].min()) > 0
        and top40_summary["gain_vs_cp12"] > 0
    )
    print("\nANCHOR-RESET DATUM BLEND:", "PASS" if passed else "FAIL / STOP")

    bywell.to_csv(OOF_DIR / "anchor_reset_datum10_bywell.csv", index=False)
    blend_folds.to_csv(OOF_DIR / "anchor_reset_datum10_folds.csv", index=False)
    if passed:
        merged[["id", "well", "tvt_true", "blend_cp12_datum10"]].rename(
            columns={"blend_cp12_datum10": "tvt_base"}
        ).to_csv(OOF_DIR / "anchor_reset_datum10_oof.csv", index=False)
        print("saved full datum-blend OOF")
else:
    print("candidate pool not found; skipped CP12 blend evaluation")
