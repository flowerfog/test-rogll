# Kaggle Strict Goldline Submit Patch

这是严格按“本地有效结论”写的提交版，不再把有效模块阉割成单一 postprocess。

核心使用的本地结论：

- `tail_prefix_full_apply_scan_v1` 是目前最强稳定信号：visible-prefix replay 选择候选，5 折全正。
- raw fallback oracle 空间大，top bad wells 主要由 `ridge / selector / fleongg / sp45 / cp12` 等候选覆盖。
- risk/GR combo 本地 OOF 有用，但 LB 直接上崩过，所以这版不把 risk 模型当硬覆盖，而是把 GR/event 与 prefix replay 变成 soft evidence。
- Gold 不再作为最后独立覆盖层；Gold 的思想被吸收到 prefix replay evidence。

这份 patch 的输出是最终 `submission.csv`，不要再接原 Gold cell。

---

## 必须修改的两个小地方

### 1. Cell 7 末尾加

```python
base_level_trainers = {}
```

### 2. Cell 8 / Cell 9 保存 first-level trainer

Cell 8 LightGBM 循环里，在写 `oof_preds/test_preds` 前加：

```python
base_level_trainers[f"lightgbm-{i+1}"] = trainer
```

Cell 9 CatBoost 循环里，在写 `oof_preds/test_preds` 前加：

```python
base_level_trainers[f"catboost-{i+1}"] = trainer
```

---

## 推荐运行顺序

1. 正常跑到 Cell 7。
2. 加上 `base_level_trainers = {}`。
3. 修改 Cell 8 / Cell 9 保存 trainer，然后正常跑 Cell 8、Cell 9。
4. 正常跑 Cell 10 到 Cell 19，得到：
   - `sub_1` ridge_pp
   - `sub_2` selector/PF
   - `sp45_projection_submission.csv`
5. 正常跑 fleongg stack 到 Cell 29，得到：
   - `fleongg_pretrained_submission.csv`
   - `submission_before_gold.csv` 或当前 `submission.csv` 的 sp45/fle blend
6. 跑 Cell 32 保存 `submission_before_gold.csv`。
7. **不要跑 Cell 35 CP12，不要跑 Cell 36 Gold。**
8. 跑下面这个 Strict Goldline cell，直接输出最终 `submission.csv`。

---

## Strict Goldline Submit Cell

```python
# ============================================================
# Strict Goldline Candidate Pool + Visible-Prefix Replay Submit
# Put this AFTER submission_before_gold.csv is saved.
# Do NOT run old CP12 / Gold after this cell.
# ============================================================
from pathlib import Path
import os, glob, tempfile, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
DATA = Path(getattr(CFG, "dataset_path", getattr(CFG, "DATA", "/kaggle/input/competitions/rogii-wellbore-geology-prediction")))

SAMPLE_PATH = DATA / "sample_submission.csv"
sample = pd.read_csv(SAMPLE_PATH)
sample["id"] = sample["id"].astype(str)
sample["well"] = sample["id"].str[:8]
sample["row_idx"] = sample["id"].str[9:].astype(int)

PREFIX_TAIL_N = 1000
PREFIX_SCORE_FRAC = 0.50
PREFIX_MIN_SCORE_ROWS = 80

# Local effective config. This was the best robust prefix setting locally.
PREFIX_GAIN_THR = 0.25
SOFTMAX_TEMP = 1.25
BASE_KEEP = 0.50
ROW_MOVE_CAP = 8.0

# Shift family from the constant-bias diagnosis.
SHIFT_DELTAS = np.array([-24, -16, -12, -8, -4, 0, 4, 8, 12, 16, 24], dtype=float)

print("STRICT DATA:", DATA)
print("STRICT WORK:", WORK)

def _read_sub(path, name):
    p = Path(path)
    if not p.exists():
        print(f"skip missing {name}: {p}")
        return None
    d = pd.read_csv(p)
    if set(["id", "tvt"]) - set(d.columns):
        print(f"skip bad {name}: columns={list(d.columns)}")
        return None
    d = d[["id", "tvt"]].copy()
    d["id"] = d["id"].astype(str)
    d["tvt"] = d["tvt"].astype(float)
    d = sample[["id"]].merge(d, on="id", how="left")
    if d["tvt"].isna().any() or not np.isfinite(d["tvt"].to_numpy(float)).all():
        print(f"skip nonfinite {name}")
        return None
    return d.rename(columns={"tvt": name})

def _sub_from_global(df, name, pred_col="tvt"):
    if df is None:
        return None
    d = df[["id", pred_col]].rename(columns={pred_col: name}).copy()
    d["id"] = d["id"].astype(str)
    d[name] = d[name].astype(float)
    d = sample[["id"]].merge(d, on="id", how="left")
    if d[name].isna().any() or not np.isfinite(d[name].to_numpy(float)).all():
        print(f"skip global nonfinite {name}")
        return None
    return d

def _robfit(s, y, deg=4):
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - np.polyval(c, s)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)

def _project_submission(sub, name, raw_weight=0.25, fit_weight=0.75):
    d = sub[["id", name]].rename(columns={name: "tvt"}).copy()
    d["well"] = d["id"].str[:8]
    d["row_idx"] = d["id"].str[9:].astype(int)
    out = dict(zip(d["id"].values, d["tvt"].astype(float).values))
    n_ok = 0
    for wid, g in d.groupby("well", sort=False):
        try:
            hw = pd.read_csv(DATA / "test" / f"{wid}__horizontal_well.csv")
            kn = hw[hw["TVT_input"].notna()]
            if len(kn) < 5:
                continue
            last = kn.iloc[-1]
            anchor = float(last["TVT_input"]) + float(last["Z"])
            ps = float(last["MD"])
            end = float(hw["MD"].iloc[-1])
            gg = g.sort_values("row_idx")
            ri = gg["row_idx"].to_numpy(int)
            z = hw["Z"].to_numpy(float)[ri]
            md = hw["MD"].to_numpy(float)[ri]
            s = (md - ps) / max(end - ps, 1e-6)
            tvt = gg["tvt"].to_numpy(float)
            fit = _robfit(s, (tvt + z) - anchor, deg=4)
            tvt_fit = raw_weight * tvt + fit_weight * ((anchor + fit) - z)
            if not np.isfinite(tvt_fit).all():
                continue
            for rid, val in zip(gg["id"].values, tvt_fit):
                out[rid] = float(val)
            n_ok += 1
        except Exception as e:
            print("project fallback", wid, e)
    res = sub[["id"]].copy()
    res[name] = res["id"].map(out).astype(float)
    print(f"projected {name}: wells={n_ok}")
    return res

def _clip_move(base, cand, cap=ROW_MOVE_CAP):
    return base + np.clip(cand - base, -cap, cap)

# ------------------------------------------------------------
# 1) Build eval candidate pool
# ------------------------------------------------------------
cands = []

ridge_eval = _sub_from_global(globals().get("sub_1", None), "ridge", "tvt")
selector_eval = _sub_from_global(globals().get("sub_2", None), "selector", "tvt")
if ridge_eval is not None:
    cands.append(ridge_eval)
if selector_eval is not None:
    cands.append(selector_eval)

for name, path in [
    ("sp45", WORK / "sp45_projection_submission.csv"),
    ("fleongg", WORK / "fleongg_pretrained_submission.csv"),
    ("blend055", WORK / "submission_before_gold.csv"),
    ("cp12", WORK / "submission_cp12_before_gold.csv"),
    ("risk_gr_combo", WORK / "submission_risk_gr_combo.csv"),
]:
    d = _read_sub(path, name)
    if d is not None:
        cands.append(d)

pool = sample[["id", "well", "row_idx"]].copy()
for d in cands:
    pool = pool.merge(d, on="id", how="left")

pred_cols = [c for c in pool.columns if c not in ["id", "well", "row_idx"]]
if not pred_cols:
    raise RuntimeError("No candidate predictions loaded.")

for c in pred_cols:
    if pool[c].isna().any():
        raise RuntimeError(f"candidate {c} has missing values")

if "blend055" in pred_cols:
    BASE_COL = "blend055"
elif "sp45" in pred_cols:
    BASE_COL = "sp45"
else:
    BASE_COL = pred_cols[0]

print("loaded eval candidates:", pred_cols)
print("base candidate:", BASE_COL)

# Add local effective derived candidates.
if "ridge" in pred_cols:
    pool["ridge_clip"] = _clip_move(pool[BASE_COL].to_numpy(float), pool["ridge"].to_numpy(float), ROW_MOVE_CAP)
if "ridge" in pred_cols and "selector" in pred_cols:
    pool["sp45_like"] = 0.3 * pool["ridge"].to_numpy(float) + 0.7 * pool["selector"].to_numpy(float)
    pool["sp45_like_proj"] = _project_submission(pool[["id", "sp45_like"]], "sp45_like")["sp45_like"].to_numpy(float)

pred_cols = [c for c in pool.columns if c not in ["id", "well", "row_idx"]]

# Shift family: only for strong candidates, not every weak route.
shift_sources = [c for c in ["ridge", "selector", "sp45", "fleongg", "blend055", "cp12", "ridge_clip"] if c in pred_cols]
for c in shift_sources:
    for dlt in SHIFT_DELTAS:
        if dlt == 0:
            continue
        pool[f"{c}_shift_{dlt:+.0f}"] = pool[c].to_numpy(float) + dlt

pred_cols = [c for c in pool.columns if c not in ["id", "well", "row_idx"]]
print("candidate count with shifts:", len(pred_cols))

# ------------------------------------------------------------
# 2) Replay candidates for visible-prefix evidence
# ------------------------------------------------------------
train_hw_files = sorted(glob.glob(str(DATA / "train" / "*__horizontal_well.csv")))
train_wells = {os.path.basename(f).split("__")[0] for f in train_hw_files}

def _build_score_idx(hw):
    known_idx = np.where(hw["TVT_input"].notna().to_numpy())[0]
    if len(known_idx) < PREFIX_MIN_SCORE_ROWS + 30:
        return None
    tail = known_idx[-min(PREFIX_TAIL_N, len(known_idx)):]
    n_score = int(round(len(tail) * PREFIX_SCORE_FRAC))
    n_score = max(PREFIX_MIN_SCORE_ROWS, min(n_score, len(tail) - 30))
    if n_score < PREFIX_MIN_SCORE_ROWS:
        return None
    return tail[-n_score:]

def _predict_stack_features(feat_df):
    if "base_level_trainers" not in globals():
        raise RuntimeError("base_level_trainers missing. Add trainer storage in Cell 8/9.")
    X_new = feat_df[features]
    cols = []
    for name in list(oof_preds.columns):
        if name not in base_level_trainers:
            raise RuntimeError(f"missing first-level trainer for {name}")
        cols.append(pd.Series(base_level_trainers[name].predict(X_new), name=name))
    return pd.concat(cols, axis=1)

def _ridge_pp_from_masked_hw(wid, hw_mask, tw_path):
    with tempfile.TemporaryDirectory(dir=str(WORK)) as td:
        tmp_hw = Path(td) / f"{wid}__horizontal_well.csv"
        hw_mask.to_csv(tmp_hw, index=False)
        feat = build_well(str(tmp_hw), str(tw_path), is_train=False)
    if feat is None or len(feat) == 0:
        return None
    stack = _predict_stack_features(feat)
    raw = ridge_trainer.predict(stack)
    pf_delta = feat["pf_ancc"].to_numpy(float) - feat["last_known_tvt"].to_numpy(float)
    pred = feat["last_known_tvt"].to_numpy(float) + apply_pp(feat, raw, pf_delta, **pp_params)
    d = feat[["id", "well"]].copy()
    d["pred"] = pred
    d = sg_smooth(d, "pred")
    return d[["id", "pred"]].rename(columns={"pred": "ridge"})

def _selector_full_array(wid, hw_te, tw_te):
    tvt_phys = None
    hw_tr = None
    tw_tr = None
    if wid in train_wells:
        try:
            hw_tr, tw_tr = load_well(wid, "train")
            if len(hw_tr) == len(hw_te):
                hw_te = hw_te.copy()
                hw_te["TVT_input"] = hw_tr["TVT_input"].values
                tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
        except Exception:
            tvt_phys = None

    selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(hw_te)
    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref, n_particles=500, n_seeds=128)
        tvt_pf = pf_by_scale["pf_scale_8"]
    except Exception:
        last_known = hw_te["TVT_input"].dropna()
        last_val = float(last_known.iloc[-1]) if len(last_known) else 0.0
        tvt_pf = hw_te["TVT_input"].fillna(last_val).to_numpy(float)
        pf_by_scale = {f"pf_scale_{scale:g}": tvt_pf.copy() for scale in SELECTOR_SCALES}
    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        tvt_beam = run_beam_ensemble(hw_te, tw_ref)
    except Exception:
        tvt_beam = tvt_pf.copy()

    last_known = hw_te["TVT_input"].dropna()
    last_known_tvt = float(last_known.iloc[-1]) if len(last_known) else float(np.nanmean(tvt_pf))
    tvt_selector = apply_selector_variant(selector_variant, pf_by_scale, tvt_beam, last_known_tvt)

    if tvt_phys is not None:
        arr = np.asarray(tvt_phys, float)
        if len(arr) == len(hw_te) and np.isfinite(arr).all():
            return arr
    return np.asarray(tvt_selector, float)

def _gr_match_score(hw, tw, row_idx, pred):
    # Robust GR/typewell evidence; higher is better.
    gr = hw["GR"].to_numpy(float)[row_idx]
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].to_numpy(float)
    exp_gr = np.interp(pred, tw_tvt, tw_gr)
    m = np.isfinite(gr) & np.isfinite(exp_gr)
    if m.sum() < 20:
        return -999.0
    a = gr[m]
    b = exp_gr[m]
    a = pd.Series(a).rank(pct=True).to_numpy(float)
    b = pd.Series(b).rank(pct=True).to_numpy(float)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    corr = float(np.mean(a * b))
    mae = float(np.mean(np.abs(a - b)))
    return corr - 0.15 * mae

def _softmax(x, temp=1.0):
    x = np.asarray(x, float)
    x = x / max(temp, 1e-6)
    x = x - np.nanmax(x)
    e = np.exp(np.clip(x, -50, 50))
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(e) / len(e)
    return e / s

well_decisions = []
final_pred = pool[BASE_COL].to_numpy(float).copy()

t0 = time.time()
for iw, (wid, g) in enumerate(pool.groupby("well", sort=False), start=1):
    if iw % 25 == 0:
        print(f"strict well {iw}/{pool['well'].nunique()} elapsed={time.time()-t0:.1f}s", flush=True)

    hw_path = DATA / "test" / f"{wid}__horizontal_well.csv"
    tw_path = DATA / "test" / f"{wid}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        continue
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values("TVT")

    eval_idx = g["row_idx"].to_numpy(int)
    score_idx = _build_score_idx(hw)

    evidence = {c: 0.0 for c in pred_cols}
    prefix_used = False

    # 2a) Prefix replay evidence for candidates that can be honestly replayed.
    if score_idx is not None:
        hw_mask = hw.copy()
        hw_mask.loc[score_idx, "TVT_input"] = np.nan
        try:
            rpref = _ridge_pp_from_masked_hw(wid, hw_mask, tw_path) if "ridge" in pred_cols else None
            sel_arr = _selector_full_array(wid, hw_mask, tw) if "selector" in pred_cols else None
            replay = None
            if rpref is not None:
                replay = rpref.copy()
            if sel_arr is not None:
                sref = pd.DataFrame({"id": [f"{wid}_{i}" for i in score_idx], "selector": sel_arr[score_idx].astype(float)})
                replay = sref if replay is None else replay.merge(sref, on="id", how="outer")
            if replay is not None and len(replay):
                replay["row_idx"] = replay["id"].str[9:].astype(int)
                truth = hw.loc[replay["row_idx"].to_numpy(int), "TVT_input"].to_numpy(float)
                if "ridge" in replay and "selector" in replay:
                    replay["sp45_like"] = 0.3 * replay["ridge"].astype(float) + 0.7 * replay["selector"].astype(float)
                if "blend055" in pred_cols and "sp45_like" in replay:
                    # replay approximation: blend055 not honestly replayable here, so use sp45_like as its replay anchor.
                    replay["blend055"] = replay["sp45_like"]
                replay_cols = [c for c in replay.columns if c not in ["id", "row_idx"]]
                rmses = {}
                for c in replay_cols:
                    v = replay[c].to_numpy(float)
                    if len(v) == len(truth) and np.isfinite(v).all():
                        rmses[c] = float(np.sqrt(np.mean((v - truth) ** 2)))
                if rmses:
                    base_replay_col = "blend055" if "blend055" in rmses else ("sp45_like" if "sp45_like" in rmses else list(rmses)[0])
                    base_rmse = rmses[base_replay_col]
                    for c, r in rmses.items():
                        gain = base_rmse - r
                        if c in evidence:
                            evidence[c] += 1.0 * gain
                        # shifted replay family gets prefix evidence too.
                        for dlt in SHIFT_DELTAS:
                            sc = f"{c}_shift_{dlt:+.0f}"
                            if sc in evidence:
                                shifted = replay[c].to_numpy(float) + dlt
                                sr = float(np.sqrt(np.mean((shifted - truth) ** 2)))
                                evidence[sc] += 1.0 * (base_rmse - sr)
                    prefix_used = True
        except Exception as e:
            print("prefix replay fallback", wid, e)

    # 2b) GR/event evidence for all final eval candidates.
    gr_scores = {}
    for c in pred_cols:
        pred_c = g[c].to_numpy(float)
        gr_scores[c] = _gr_match_score(hw, tw, eval_idx, pred_c)
    med_gr = np.nanmedian([v for v in gr_scores.values() if np.isfinite(v)])
    if not np.isfinite(med_gr):
        med_gr = 0.0
    for c, s in gr_scores.items():
        if np.isfinite(s):
            evidence[c] += 0.8 * (s - med_gr)

    # 2c) Candidate geometry risk evidence: prefer alternatives when pool disagreement is large.
    vals = np.column_stack([g[c].to_numpy(float) for c in pred_cols])
    pool_std = np.nanmedian(np.nanstd(vals, axis=1))
    if np.isfinite(pool_std) and pool_std > 1.5:
        for c in pred_cols:
            # Do not overly favor base in high-disagreement wells.
            if c != BASE_COL:
                evidence[c] += 0.05 * min(pool_std, 20.0)

    scores = np.array([evidence[c] for c in pred_cols], dtype=float)

    # Participation gate from local prefix conclusion: if prefix replay exists and no candidate beats base,
    # keep the base mostly intact. If no prefix replay, GR evidence still can move softly.
    if prefix_used and np.nanmax(scores) < PREFIX_GAIN_THR:
        weights = np.zeros(len(pred_cols), dtype=float)
        weights[pred_cols.index(BASE_COL)] = 1.0
    else:
        weights = _softmax(scores, SOFTMAX_TEMP)

    cand_mat = vals
    soft_pred = cand_mat @ weights
    base_v = g[BASE_COL].to_numpy(float)
    pred_well = BASE_KEEP * base_v + (1.0 - BASE_KEEP) * soft_pred
    pred_well = _clip_move(base_v, pred_well, ROW_MOVE_CAP)

    final_pred[g.index.to_numpy()] = pred_well

    top_order = np.argsort(-weights)[:5]
    well_decisions.append({
        "well": wid,
        "rows": int(len(g)),
        "prefix_used": bool(prefix_used),
        "pool_std": float(pool_std) if np.isfinite(pool_std) else np.nan,
        "top1": pred_cols[int(top_order[0])],
        "top1_weight": float(weights[int(top_order[0])]),
        "top5": ";".join([f"{pred_cols[int(i)]}:{weights[int(i)]:.3f}" for i in top_order]),
        "move_mean_abs": float(np.mean(np.abs(pred_well - base_v))),
        "move_p95_abs": float(np.quantile(np.abs(pred_well - base_v), 0.95)),
    })

out = pool[["id"]].copy()
out["tvt"] = final_pred
out = sample[["id"]].merge(out, on="id", how="left")

if out["tvt"].isna().any():
    raise RuntimeError("strict submit produced missing predictions")
if not np.isfinite(out["tvt"].to_numpy(float)).all():
    raise RuntimeError("strict submit produced non-finite predictions")

dec = pd.DataFrame(well_decisions)
dec.to_csv(WORK / "strict_goldline_decisions.csv", index=False)
out.to_csv(WORK / "submission_strict_goldline_candidate_pool.csv", index=False)
out.to_csv(WORK / "submission.csv", index=False)

base_move_ref = sample[["id"]].merge(pool[["id", BASE_COL]], on="id", how="left")[BASE_COL].to_numpy(float)
move = out["tvt"].to_numpy(float) - base_move_ref

audit = {
    "base_col": BASE_COL,
    "candidate_count": int(len(pred_cols)),
    "rows": int(len(out)),
    "wells": int(sample["well"].nunique()),
    "prefix_used_wells": int(dec["prefix_used"].sum()) if len(dec) and "prefix_used" in dec else 0,
    "mean_abs_move_vs_base": float(np.mean(np.abs(move))),
    "p95_abs_move_vs_base": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move_vs_base": float(np.max(np.abs(move))),
    "output": str(WORK / "submission_strict_goldline_candidate_pool.csv"),
}
with open(WORK / "strict_goldline_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, sort_keys=True)

print("STRICT GOLDLINE DONE")
print(audit)
if len(dec):
    print(dec.sort_values("move_p95_abs", ascending=False).head(30).to_string(index=False))
```

---

## 说明

这版不是“Gold 后加东西”，而是：

- 把 `submission_before_gold.csv` 当 base。
- 把多个路线放进 candidate pool。
- 对可见 prefix 做 replay 自监督。
- 用 GR/event 与候选分歧补充 evidence。
- softmax 混合候选。
- 直接输出最终 `submission.csv`。

如果想更激进：

```python
BASE_KEEP = 0.35
ROW_MOVE_CAP = 12.0
```

如果想更保守：

```python
PREFIX_GAIN_THR = 1.0
BASE_KEEP = 0.70
ROW_MOVE_CAP = 4.0
```
