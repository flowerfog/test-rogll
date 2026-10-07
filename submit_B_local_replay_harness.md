# Submit B Local Replay Harness

这份代码用于本地验证提交 B，不是 Kaggle 提交代码。

它只验证已经被本地证据支持的核心规则：

- visible-prefix replay
- hard select
- `gain_thr`
- `blend_ref`
- `ridge / selector / sp45_like + shifts`

它不会加入 softmax、GR evidence、pool disagreement、risk model。那些必须单独验证后才能进入 B2/B3。

## 运行位置

放在本地 notebook 里运行。需要已有：

- `/root/autodl-tmp/oof_exports/candidate_pool_full_merged_oof_honest_fle.csv`
- 其中至少包含：`id, well, tvt_true, ridge, selector`

如果有 `cp12 / strong_bias / cpfam_sb_k8 / risk_gr_combo`，代码会自动做 proxy 对照；没有也不影响严格 B 验证。

## Code

```python
# ============================================================
# Submit B local validation: visible-prefix hard-select replay
#
# Main deployable B rule:
#   base = sp45_like = 0.3 * ridge + 0.7 * selector
#   candidates = ridge / selector / sp45_like + shift family
#   choose by prefix RMSE gain
#   apply to hidden tail with shrink blend and cap
#
# This is a local validation harness. It is NOT Kaggle submit code.
# ============================================================
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

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

def add_optional_candidate(pool, name, filenames):
    if name in pool.columns:
        return pool
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
            num_cols = [c for c in d.columns if c not in ["id", "well", "row_idx", "tvt_true"] and np.issubdtype(d[c].dtype, np.number)]
            if len(num_cols):
                pred_col = num_cols[-1]
        if pred_col is None:
            continue
        d = d[["id", pred_col]].rename(columns={pred_col: name}).copy()
        before = len(pool)
        pool = pool.merge(d, on="id", how="left")
        if len(pool) != before or pool[name].isna().any():
            print("skip optional with mismatch:", name, p)
            pool = pool.drop(columns=[name], errors="ignore")
            continue
        print("added optional", name, "from", p.name, "finite_rate", np.isfinite(pool[name].to_numpy(float)).mean())
        return pool
    return pool

t0 = time.time()
pool = pd.read_csv(POOL_PATH)
pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str[:8]
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[1].astype(int)

required = ["id", "well", "row_idx", "tvt_true", "ridge", "selector"]
missing = [c for c in required if c not in pool.columns]
if missing:
    raise RuntimeError(f"pool missing required columns: {missing}")

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

pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool["sp45_like"] = 0.3 * pool["ridge"].to_numpy(float) + 0.7 * pool["selector"].to_numpy(float)

print("pool:", pool.shape, "wells:", pool["well"].nunique(), "elapsed:", time.time() - t0)

base_col = "sp45_like"
strict_sources = ["ridge", "selector", "sp45_like"]
shift_deltas = np.array([-16, -12, -8, -4, 0, 4, 8, 12, 16], dtype=float)

tail_settings = [
    ("frac_0.25", "frac", 0.25),
    ("frac_0.40", "frac", 0.40),
    ("frac_0.55", "frac", 0.55),
    ("frac_0.70", "frac", 0.70),
    ("n_tail_1000", "n", 1000),
    ("n_tail_2000", "n", 2000),
]
prefix_fracs = [0.30, 0.40, 0.50]
gain_thrs = [0.0, 0.25, 0.50, 1.0, 2.0]
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

def build_records_for_setting(tail_name, kind, value, prefix_frac):
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
        hid_idx = idx_all[pos_tail[n_pref:]]

        pref = pool.loc[pref_idx]
        hid = pool.loc[hid_idx]
        y_pref = pref["tvt_true"].to_numpy(float)

        cand_pref = {}
        cand_hid = {}
        for src in strict_sources:
            pv = pref[src].to_numpy(float)
            hv = hid[src].to_numpy(float)
            cand_pref[src] = pv
            cand_hid[src] = hv
            for dlt in shift_deltas:
                nm = f"{src}_shift_{dlt:+.0f}"
                cand_pref[nm] = pv + dlt
                cand_hid[nm] = hv + dlt

        rmses = {}
        for nm, v in cand_pref.items():
            if np.isfinite(v).all():
                rmses[nm] = rmse(v, y_pref)
        if base_col not in rmses:
            continue
        best = min(rmses, key=rmses.get)
        base_pref_rmse = rmses[base_col]
        best_pref_rmse = rmses[best]
        pref_gain = base_pref_rmse - best_pref_rmse

        records.append({
            "tail_setting": tail_name,
            "prefix_frac": prefix_frac,
            "well": wid,
            "prefix_n": len(pref),
            "hidden_n": len(hid),
            "prefix_idx": pref_idx,
            "hidden_idx": hid_idx,
            "base_pref_rmse": base_pref_rmse,
            "best_cand": best,
            "best_pref_rmse": best_pref_rmse,
            "pref_gain": pref_gain,
            "base_hidden_rmse": rmse(hid[base_col], hid["tvt_true"]),
            "best_hidden_rmse": rmse(cand_hid[best], hid["tvt_true"]),
            "pref_best_hidden_gain": rmse(hid[base_col], hid["tvt_true"]) - rmse(cand_hid[best], hid["tvt_true"]),
        })
    return records

print("building prefix records...")
all_records = []
for tail_name, kind, value in tail_settings:
    print("tail:", tail_name, flush=True)
    for pf in prefix_fracs:
        all_records.extend(build_records_for_setting(tail_name, kind, value, pf))

diag = pd.DataFrame(all_records)
print("records:", diag.shape)
if len(diag) == 0:
    raise RuntimeError("No prefix records built")

display(
    diag.groupby(["tail_setting", "prefix_frac"])
    .agg(
        wells=("well", "nunique"),
        mean_pref_gain=("pref_gain", "mean"),
        mean_hidden_gain=("pref_best_hidden_gain", "mean"),
        positive_hidden_frac=("pref_best_hidden_gain", lambda x: float((x > 0).mean())),
        exact_base_wins=("best_cand", lambda x: float((x == base_col).mean())),
        mean_base_hidden_rmse=("base_hidden_rmse", "mean"),
        mean_best_hidden_rmse=("best_hidden_rmse", "mean"),
    )
    .reset_index()
)

def apply_setting(rows_df, gain_thr, blend_ref, cap, proxy_base=None):
    if proxy_base is None:
        proxy_base = base_col
    pred = {}
    base_vals = {}
    y_vals = {}
    move_vals = []
    selected_wells = 0
    selected_rows = 0
    chosen_counts = {}

    for r in rows_df.itertuples(index=False):
        hid_idx = np.asarray(r.hidden_idx, dtype=int)
        g = pool.loc[hid_idx]
        y = g["tvt_true"].to_numpy(float)
        base_v = g[proxy_base].to_numpy(float)

        do = bool(r.pref_gain > gain_thr and r.best_cand != base_col)
        if do:
            selected_wells += 1
            selected_rows += len(g)
            chosen_counts[r.best_cand] = chosen_counts.get(r.best_cand, 0) + 1

            # Best candidate is defined in strict source space. Apply the same candidate shape.
            nm = r.best_cand
            if "_shift_" in nm:
                src, dlt_s = nm.split("_shift_")
                cand_v = g[src].to_numpy(float) + float(dlt_s)
            else:
                cand_v = g[nm].to_numpy(float)

            # For proxy bases, transfer selected candidate by shrink from proxy base.
            out = blend_ref * base_v + (1.0 - blend_ref) * cand_v
            if cap < 900:
                out = base_v + np.clip(out - base_v, -cap, cap)
        else:
            out = base_v

        pred[len(pred)] = out
        base_vals[len(base_vals)] = base_v
        y_vals[len(y_vals)] = y
        move_vals.append(out - base_v)

    pred_all = np.concatenate(list(pred.values()))
    base_all = np.concatenate(list(base_vals.values()))
    y_all = np.concatenate(list(y_vals.values()))
    move_all = np.concatenate(move_vals)

    return {
        "base_rmse": rmse(base_all, y_all),
        "pred_rmse": rmse(pred_all, y_all),
        "gain": rmse(base_all, y_all) - rmse(pred_all, y_all),
        "selected_wells": selected_wells,
        "selected_rows": int(selected_rows),
        "mean_abs_move": float(np.mean(np.abs(move_all))),
        "p95_abs_move": float(np.quantile(np.abs(move_all), 0.95)),
        "top_chosen": sorted(chosen_counts.items(), key=lambda x: -x[1])[:8],
    }

def fold_gains(rows_df, gain_thr, blend_ref, cap, proxy_base=None):
    wells = rows_df["well"].astype(str).to_numpy()
    unique_wells = np.array(sorted(rows_df["well"].astype(str).unique()))
    gkf = GroupKFold(n_splits=5)
    fold_rows = []
    dummy = np.zeros(len(unique_wells))
    for fold, (_, valid_well_pos) in enumerate(gkf.split(dummy, dummy, groups=unique_wells), start=1):
        valid_wells = set(unique_wells[valid_well_pos])
        sub = rows_df[rows_df["well"].astype(str).isin(valid_wells)].copy()
        m = apply_setting(sub, gain_thr, blend_ref, cap, proxy_base=proxy_base)
        fold_rows.append({
            "fold": fold,
            "valid_wells": len(valid_wells),
            **m,
        })
    return pd.DataFrame(fold_rows)

scan_rows = []
proxy_bases = [base_col]
for c in ["cp12", "strong_bias", "cpfam_sb_k8", "risk_gr_combo"]:
    if c in pool.columns:
        proxy_bases.append(c)

print("proxy bases:", proxy_bases)
for (tail_setting, prefix_frac), rows_df in diag.groupby(["tail_setting", "prefix_frac"], sort=False):
    for gain_thr in gain_thrs:
        for blend_ref in blend_refs:
            for cap in caps:
                for proxy_base in proxy_bases:
                    m = apply_setting(rows_df, gain_thr, blend_ref, cap, proxy_base=proxy_base)
                    fg = fold_gains(rows_df, gain_thr, blend_ref, cap, proxy_base=proxy_base)
                    scan_rows.append({
                        "tail_setting": tail_setting,
                        "prefix_frac": prefix_frac,
                        "proxy_base": proxy_base,
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
    scan.sort_values(["proxy_base", "robust_score"], ascending=[True, False])
    .groupby("proxy_base")
    .head(15)
    .reset_index(drop=True)
)

strict = scan[scan["proxy_base"] == base_col].sort_values("robust_score", ascending=False).reset_index(drop=True)
print("STRICT B best:")
display(strict.head(30))

best = strict.iloc[0].to_dict()
print("BEST STRICT:", best)

best_rows = diag[
    (diag["tail_setting"] == best["tail_setting"]) &
    (diag["prefix_frac"] == best["prefix_frac"])
].copy()
best_folds = fold_gains(
    best_rows,
    gain_thr=float(best["gain_thr"]),
    blend_ref=float(best["blend_ref"]),
    cap=float(best["cap"]),
    proxy_base=base_col,
)
display(best_folds)

out_scan = OOF_DIR / "submit_B_local_replay_strict_scan.csv"
out_diag = OOF_DIR / "submit_B_local_replay_prefix_diag.csv"
out_fold = OOF_DIR / "submit_B_local_replay_strict_best_folds.csv"
scan.to_csv(out_scan, index=False)
diag_drop = diag.drop(columns=["prefix_idx", "hidden_idx"], errors="ignore")
diag_drop.to_csv(out_diag, index=False)
best_folds.to_csv(out_fold, index=False)

print("saved:")
print(out_scan)
print(out_diag)
print(out_fold)
```

## 如何读结果

优先看 `STRICT B best`，也就是 `proxy_base == sp45_like` 的结果。

可以进入 Kaggle 提交代码的条件：

- `gain > 0`
- `mean_fold_gain > 0`
- `min_fold_gain > 0`，至少不能明显负
- `positive_folds == 5`
- `p95_abs_move` 不离谱

`proxy_base != sp45_like` 只用于判断旧链路/后处理是否可能更稳，不是第一版 B 的提交依据。
