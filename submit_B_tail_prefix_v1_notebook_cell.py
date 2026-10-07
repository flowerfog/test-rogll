# ============================================================
# Submit B local validation, using the previous effective method:
# tail_prefix_full_apply_scan_v1 style.
#
# What it does:
#   1) split each well tail into prefix + hidden
#   2) choose candidate by prefix RMSE gain
#   3) apply hard-select + shrink blend on hidden
#   4) score pooled hidden RMSE + GroupKFold min-fold
#
# This is a LOCAL NOTEBOOK CELL. It is not Kaggle submit code.
# ============================================================
from pathlib import Path
import time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# Start with the focused grid that matches our previous useful validation.
# Set FAST_GRID=False after the first pass if the result is promising.
FAST_GRID = True

OOF_DIR_CANDIDATES = [
    Path("/root/autodl-tmp/oof_exports"),
    Path("oof_exports"),
    Path("."),
]
OOF_DIR = next((p for p in OOF_DIR_CANDIDATES if (p / "candidate_pool_full_merged_oof_honest_fle.csv").exists()), None)
if OOF_DIR is None:
    raise RuntimeError("Cannot find candidate_pool_full_merged_oof_honest_fle.csv")

POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
print("OOF_DIR:", OOF_DIR)
print("POOL:", POOL_PATH)

def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _read_optional_oof(name, filenames):
    for fn in filenames:
        p = OOF_DIR / fn
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "id" not in d.columns:
            continue
        pred_col = None
        for c in ["tvt_base", "pred", "tvt", name]:
            if c in d.columns:
                pred_col = c
                break
        if pred_col is None:
            num_cols = [
                c for c in d.columns
                if c not in ["id", "well", "row_idx", "tvt_true"]
                and np.issubdtype(d[c].dtype, np.number)
            ]
            if num_cols:
                pred_col = num_cols[-1]
        if pred_col is None:
            continue
        return d[["id", pred_col]].rename(columns={pred_col: name})
    return None

def add_optional_candidate(pool, name, filenames):
    if name in pool.columns:
        return pool
    d = _read_optional_oof(name, filenames)
    if d is None:
        return pool
    n0 = len(pool)
    out = pool.merge(d, on="id", how="left")
    if len(out) != n0 or out[name].isna().any() or not np.isfinite(out[name].to_numpy(float)).all():
        print("skip optional candidate with mismatch:", name)
        return pool
    print("added", name, "finite_rate", np.isfinite(out[name].to_numpy(float)).mean())
    return out

t0 = time.time()
pool = pd.read_csv(POOL_PATH)
pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str[:8]
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[1].astype(int)

pool = add_optional_candidate(pool, "cpfam_sb_k8", [
    "cpfam_strong_bias_k8_a1p00_c999_oof.csv",
    "cpfam_strong_bias_k8_a1p00_c999_oofpcsv",
])
pool = add_optional_candidate(pool, "ridge_clip", [
    "ridge_clip_a0p25_c8_oof.csv",
])
pool = add_optional_candidate(pool, "risk_gr_combo", [
    "risk_gr_event_combo_best_oof.csv",
])

need = ["id", "well", "row_idx", "tvt_true", "ridge", "selector"]
missing = [c for c in need if c not in pool.columns]
if missing:
    raise RuntimeError(f"missing required columns: {missing}")

pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool["sp45_like"] = 0.3 * pool["ridge"].to_numpy(float) + 0.7 * pool["selector"].to_numpy(float)

print("pool:", pool.shape, "wells:", pool["well"].nunique(), "elapsed:", time.time() - t0)

base_choices = [
    c for c in ["cpfam_sb_k8", "sp45_like", "cp12", "strong_bias", "risk_gr_combo"]
    if c in pool.columns
]
if "sp45_like" not in base_choices:
    base_choices.append("sp45_like")

full_pool_raw = [
    c for c in [
        "risk_gr_combo", "cpfam_sb_k8", "ridge", "selector", "strong_bias",
        "ridge_clip", "cp12", "fleongg", "sp45", "cp_pack", "cp_single12",
        "strong", "strong_raw", "sp45_raw", "gr_offset", "sp45_like"
    ]
    if c in pool.columns
]

shift_deltas = np.array([-16, -12, -8, -4, 0, 4, 8, 12, 16], dtype=float)

candidate_sets = {
    # This reproduces the previous strong local validation style.
    "full_pool_raw": full_pool_raw,
    # This is the first deployable B candidate set: honestly replayable routes + shifts.
    "deployable_strict_shift": ["ridge", "selector", "sp45_like"],
}

if FAST_GRID:
    tail_settings = [
        ("n_tail_1000", "n", 1000),
        ("n_tail_2000", "n", 2000),
        ("frac_0.25", "frac", 0.25),
        ("frac_0.40", "frac", 0.40),
        ("frac_0.55", "frac", 0.55),
    ]
    prefix_fracs = [0.40, 0.50]
    gain_thrs = [0.0, 0.25, 0.50, 1.0, 2.0]
    blend_refs = [0.50, 0.60]
    caps = [8.0, 999.0]
else:
    tail_settings = [
        ("frac_0.25", "frac", 0.25),
        ("frac_0.40", "frac", 0.40),
        ("frac_0.55", "frac", 0.55),
        ("frac_0.70", "frac", 0.70),
        ("frac_0.85", "frac", 0.85),
        ("n_tail_1000", "n", 1000),
        ("n_tail_2000", "n", 2000),
        ("n_tail_3500", "n", 3500),
        ("n_tail_5000", "n", 5000),
    ]
    prefix_fracs = [0.20, 0.30, 0.40, 0.50]
    gain_thrs = [0.0, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0]
    blend_refs = [0.50, 0.60, 0.70]
    caps = [4.0, 8.0, 999.0]

MIN_PREFIX_ROWS = 80
MIN_HIDDEN_ROWS = 80

def tail_positions(n, kind, value):
    if kind == "frac":
        m = int(round(n * float(value)))
    else:
        m = int(value)
    m = max(0, min(n, m))
    if m <= MIN_PREFIX_ROWS + MIN_HIDDEN_ROWS:
        return None
    return np.arange(n - m, n, dtype=int)

def make_candidate_values(df, cand_cols, include_shifts):
    vals = {}
    for c in cand_cols:
        if c in df.columns:
            v = df[c].to_numpy(float)
            if np.isfinite(v).all():
                vals[c] = v
    if include_shifts:
        for src in ["ridge", "selector", "sp45_like"]:
            if src not in vals:
                continue
            v = vals[src]
            for dlt in shift_deltas:
                vals[f"{src}_shift_{dlt:+.0f}"] = v + dlt
    return vals

def build_records(base_col, cand_set_name, cand_cols, tail_name, kind, value, prefix_frac):
    include_shifts = cand_set_name == "deployable_strict_shift"
    records = []
    for wid, g in pool.groupby("well", sort=False):
        idx_all = g.index.to_numpy()
        n = len(g)
        pos_tail = tail_positions(n, kind, value)
        if pos_tail is None:
            continue

        n_pref = int(round(len(pos_tail) * prefix_frac))
        n_pref = max(MIN_PREFIX_ROWS, min(n_pref, len(pos_tail) - MIN_HIDDEN_ROWS))
        if n_pref < MIN_PREFIX_ROWS or len(pos_tail) - n_pref < MIN_HIDDEN_ROWS:
            continue

        pref_idx = idx_all[pos_tail[:n_pref]]
        hidden_idx = idx_all[pos_tail[n_pref:]]

        pref = pool.loc[pref_idx]
        hidden = pool.loc[hidden_idx]

        if base_col not in pref.columns:
            continue

        y_pref = pref["tvt_true"].to_numpy(float)
        pref_vals = make_candidate_values(pref, cand_cols, include_shifts=include_shifts)
        hidden_vals = make_candidate_values(hidden, cand_cols, include_shifts=include_shifts)

        if base_col not in pref_vals:
            continue

        rmses = {}
        for nm, v in pref_vals.items():
            if len(v) == len(y_pref):
                rmses[nm] = rmse(v, y_pref)
        if base_col not in rmses or not rmses:
            continue

        best_cand = min(rmses, key=rmses.get)
        base_pref_rmse = rmses[base_col]
        best_pref_rmse = rmses[best_cand]

        base_h = hidden[base_col].to_numpy(float)
        y_h = hidden["tvt_true"].to_numpy(float)
        best_h = hidden_vals.get(best_cand, None)
        if best_h is None:
            continue

        records.append({
            "base_col": base_col,
            "candidate_set": cand_set_name,
            "tail_setting": tail_name,
            "prefix_frac": prefix_frac,
            "well": wid,
            "prefix_n": len(pref),
            "hidden_n": len(hidden),
            "hidden_idx": hidden_idx,
            "best_cand": best_cand,
            "base_pref_rmse": base_pref_rmse,
            "best_pref_rmse": best_pref_rmse,
            "pref_gain": base_pref_rmse - best_pref_rmse,
            "base_hidden_rmse": rmse(base_h, y_h),
            "best_hidden_rmse": rmse(best_h, y_h),
            "pref_best_hidden_gain": rmse(base_h, y_h) - rmse(best_h, y_h),
        })
    return records

print("building records...")
records = []
for base_col in base_choices:
    for cand_set_name, cand_cols in candidate_sets.items():
        if base_col not in cand_cols:
            # Keep base in the candidate space; otherwise gain has a drifting meaning.
            cand_cols2 = list(dict.fromkeys([base_col] + cand_cols))
        else:
            cand_cols2 = cand_cols
        print("base/cands:", base_col, cand_set_name, "n_cands", len(cand_cols2), flush=True)
        for tail_name, kind, value in tail_settings:
            for pf in prefix_fracs:
                records.extend(build_records(base_col, cand_set_name, cand_cols2, tail_name, kind, value, pf))

diag = pd.DataFrame(records)
print("records:", diag.shape)
if len(diag) == 0:
    raise RuntimeError("No records built")

summary_diag = (
    diag.groupby(["base_col", "candidate_set", "tail_setting", "prefix_frac"])
    .agg(
        wells=("well", "nunique"),
        mean_pref_gain=("pref_gain", "mean"),
        mean_hidden_gain=("pref_best_hidden_gain", "mean"),
        positive_hidden_frac=("pref_best_hidden_gain", lambda x: float((x > 0).mean())),
        exact_base=("best_cand", lambda x: float((x == x.name if False else 0))),
        mean_base_hidden_rmse=("base_hidden_rmse", "mean"),
        mean_best_hidden_rmse=("best_hidden_rmse", "mean"),
    )
    .reset_index()
)
display(summary_diag.head(80))

def candidate_array(g, cand_name):
    if "_shift_" in cand_name:
        src, dlt_s = cand_name.split("_shift_")
        return g[src].to_numpy(float) + float(dlt_s)
    return g[cand_name].to_numpy(float)

def apply_records(rows_df, gain_thr, blend_ref, cap):
    pred_parts = []
    base_parts = []
    y_parts = []
    move_parts = []
    selected_wells = 0
    selected_rows = 0
    chosen_counts = {}

    for r in rows_df.itertuples(index=False):
        idx = np.asarray(r.hidden_idx, dtype=int)
        g = pool.loc[idx]
        base_v = g[r.base_col].to_numpy(float)
        y = g["tvt_true"].to_numpy(float)

        do = bool(r.pref_gain > gain_thr and r.best_cand != r.base_col)
        if do:
            cand_v = candidate_array(g, r.best_cand)
            out = blend_ref * base_v + (1.0 - blend_ref) * cand_v
            if cap < 900:
                out = base_v + np.clip(out - base_v, -cap, cap)
            selected_wells += 1
            selected_rows += len(g)
            chosen_counts[r.best_cand] = chosen_counts.get(r.best_cand, 0) + 1
        else:
            out = base_v

        pred_parts.append(out)
        base_parts.append(base_v)
        y_parts.append(y)
        move_parts.append(out - base_v)

    pred = np.concatenate(pred_parts)
    base = np.concatenate(base_parts)
    y = np.concatenate(y_parts)
    move = np.concatenate(move_parts)

    return {
        "base_rmse": rmse(base, y),
        "pred_rmse": rmse(pred, y),
        "gain": rmse(base, y) - rmse(pred, y),
        "selected_wells": int(selected_wells),
        "selected_rows": int(selected_rows),
        "mean_abs_move": float(np.mean(np.abs(move))),
        "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
        "top_chosen": str(sorted(chosen_counts.items(), key=lambda x: -x[1])[:8]),
    }

def fold_scan(rows_df, gain_thr, blend_ref, cap):
    wells = np.array(sorted(rows_df["well"].astype(str).unique()))
    gkf = GroupKFold(n_splits=5)
    fold_rows = []
    dummy = np.zeros(len(wells))
    for fold, (_, va) in enumerate(gkf.split(dummy, dummy, groups=wells), start=1):
        vw = set(wells[va])
        sub = rows_df[rows_df["well"].astype(str).isin(vw)]
        m = apply_records(sub, gain_thr, blend_ref, cap)
        fold_rows.append({"fold": fold, "valid_wells": len(vw), **m})
    return pd.DataFrame(fold_rows)

print("apply scan...")
scan_rows = []
group_cols = ["base_col", "candidate_set", "tail_setting", "prefix_frac"]
for key, rows_df in diag.groupby(group_cols, sort=False):
    base_col, cand_set, tail_setting, prefix_frac = key
    for gain_thr in gain_thrs:
        for blend_ref in blend_refs:
            for cap in caps:
                m = apply_records(rows_df, gain_thr, blend_ref, cap)
                fg = fold_scan(rows_df, gain_thr, blend_ref, cap)
                scan_rows.append({
                    "base_col": base_col,
                    "candidate_set": cand_set,
                    "tail_setting": tail_setting,
                    "prefix_frac": prefix_frac,
                    "gain_thr": gain_thr,
                    "blend_ref": blend_ref,
                    "cap": cap,
                    **m,
                    "mean_fold_gain": float(fg["gain"].mean()),
                    "min_fold_gain": float(fg["gain"].min()),
                    "positive_folds": int((fg["gain"] > 0).sum()),
                })

scan = pd.DataFrame(scan_rows)
scan["robust_score"] = scan["mean_fold_gain"] + np.minimum(scan["min_fold_gain"], 0.0)

display(
    scan.sort_values("robust_score", ascending=False)
    .head(100)
    .reset_index(drop=True)
)

print("best by base/candidate_set")
display(
    scan.sort_values("robust_score", ascending=False)
    .groupby(["base_col", "candidate_set"])
    .head(5)
    .reset_index(drop=True)
)

best = scan.sort_values("robust_score", ascending=False).iloc[0].to_dict()
print("BEST:", best)

best_rows = diag[
    (diag["base_col"] == best["base_col"]) &
    (diag["candidate_set"] == best["candidate_set"]) &
    (diag["tail_setting"] == best["tail_setting"]) &
    (diag["prefix_frac"] == best["prefix_frac"])
].copy()
best_folds = fold_scan(
    best_rows,
    gain_thr=float(best["gain_thr"]),
    blend_ref=float(best["blend_ref"]),
    cap=float(best["cap"]),
)
display(best_folds)

scan_path = OOF_DIR / "submit_B_tail_prefix_v1_scan.csv"
diag_path = OOF_DIR / "submit_B_tail_prefix_v1_diag.csv"
fold_path = OOF_DIR / "submit_B_tail_prefix_v1_best_folds.csv"

scan.to_csv(scan_path, index=False)
diag.drop(columns=["hidden_idx"], errors="ignore").to_csv(diag_path, index=False)
best_folds.to_csv(fold_path, index=False)

print("saved:")
print(scan_path)
print(diag_path)
print(fold_path)
