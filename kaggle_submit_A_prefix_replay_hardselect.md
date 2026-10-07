# Kaggle Submit A: Visible-Prefix Replay Hard Select

这份是按你贴的最新分析拆出来的 **提交 A**。

它只部署已经本地验证过的核心信号：

- visible-prefix replay
- 可诚实 replay 的候选：`ridge_pp / selector / sp45_like`
- 平移族：`candidate + delta`
- hard select：prefix 上谁 RMSE 最低选谁
- 参与门槛：`gain > 0.25`
- 最终收缩：`0.5 * base + 0.5 * selected`
- row cap：保留，避免大幅漂移

它明确删除：

- GR 证据
- pool disagreement 奖励
- softmax
- risk_gr_combo 直接候选
- 不可 replay 候选的“0 分中性混入”

---

## 运行顺序

1. 正常跑原 notebook 到 Cell 7。
2. 在 Cell 7 末尾加：

```python
base_level_trainers = {}
```

3. 修改 Cell 8 / Cell 9，保存 first-level trainer：

Cell 8 LightGBM 循环中，在写 `oof_preds/test_preds` 前加：

```python
base_level_trainers[f"lightgbm-{i+1}"] = trainer
```

Cell 9 CatBoost 循环中，在写 `oof_preds/test_preds` 前加：

```python
base_level_trainers[f"catboost-{i+1}"] = trainer
```

4. 正常跑 Cell 8 到 Cell 16。
5. 用下面的 Cell A 替换原 Cell 17。
6. 后面继续跑原 notebook 的 Cell 18 projection、Cell 19 保存 `sp45_projection_submission.csv`、fleongg blend、CP12、Gold。

这份提交 A 是“替换 sp45 分支”的忠实复刻版；不是 Gold 后乱加东西。

---

## Cell A：替换原 Cell 17

```python
# ============================================================
# Submit A: Visible-prefix replay hard select
# Replaces fixed:
#     0.3 * ridge_pp + 0.7 * selector
#
# Only uses locally validated signal:
#   prefix replay RMSE gain, hard candidate select, shrink blend.
#
# Continue original notebook after this cell:
#   projection -> save sp45_projection_submission.csv
#   -> fleongg blend -> CP12 -> Gold
# ============================================================
from pathlib import Path
import os, glob, tempfile, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
DATA = Path(getattr(CFG, "dataset_path", getattr(CFG, "DATA", "/kaggle/input/competitions/rogii-wellbore-geology-prediction")))

PREFIX_TAIL_N = 1000
PREFIX_SCORE_FRAC = 0.50
PREFIX_MIN_SCORE_ROWS = 80

GAIN_THR = 0.25
BLEND_REF = 0.50
ROW_MOVE_CAP = 8.0

# Keep this as the validated phase set. Wider phase families need separate CV.
SHIFT_DELTAS = np.array([-16, -12, -8, -4, 0, 4, 8, 12, 16], dtype=float)

sample = pd.read_csv(DATA / "sample_submission.csv")
sample["id"] = sample["id"].astype(str)
sample["well"] = sample["id"].str[:8]
sample["row_idx"] = sample["id"].str[9:].astype(int)

def _with_meta(df, pred_col="tvt", name="tvt"):
    d = df[["id", pred_col]].rename(columns={pred_col: name}).copy()
    d["id"] = d["id"].astype(str)
    d[name] = d[name].astype(float)
    d = sample[["id", "well", "row_idx"]].merge(d, on="id", how="left")
    if d[name].isna().any():
        raise RuntimeError(f"{name} has missing values")
    if not np.isfinite(d[name].to_numpy(float)).all():
        raise RuntimeError(f"{name} has non-finite values")
    return d

ridge_final = _with_meta(sub_1, "tvt", "ridge")
selector_final = _with_meta(sub_2, "tvt", "selector")

base = ridge_final[["id", "well", "row_idx", "ridge"]].merge(
    selector_final[["id", "selector"]],
    on="id",
    how="inner",
)
if len(base) != len(sample):
    raise RuntimeError(f"base merge mismatch: base={len(base)} sample={len(sample)}")

base["sp45_like"] = 0.3 * base["ridge"].to_numpy(float) + 0.7 * base["selector"].to_numpy(float)

def _clip_to_base(base_v, pred_v, cap=ROW_MOVE_CAP):
    return base_v + np.clip(pred_v - base_v, -cap, cap)

def _build_score_idx(hw):
    if not hw.index.equals(pd.RangeIndex(len(hw))):
        hw = hw.reset_index(drop=True)
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
        raise RuntimeError("base_level_trainers missing. Add it after Cell 7 and store trainers in Cell 8/9.")
    if not isinstance(oof_preds, pd.DataFrame):
        raise RuntimeError("oof_preds must be DataFrame. Run Cell 10 before this cell.")

    X_new = feat_df[features]
    cols = []
    for name in list(oof_preds.columns):
        if name not in base_level_trainers:
            raise RuntimeError(f"missing first-level trainer for {name}")
        cols.append(pd.Series(base_level_trainers[name].predict(X_new), name=name))
    return pd.concat(cols, axis=1)

def _ridge_pp_from_masked_hw(wid, hw_mask, tw_path):
    # build_well expects a path, so write a temporary masked horizontal-well csv.
    with tempfile.TemporaryDirectory(dir=str(WORK)) as td:
        tmp_hw = Path(td) / f"{wid}__horizontal_well.csv"
        hw_mask.to_csv(tmp_hw, index=False)
        feat = build_well(str(tmp_hw), str(tw_path), is_train=False)

    if feat is None or len(feat) == 0:
        return None

    stack = _predict_stack_features(feat)
    ridge_raw = ridge_trainer.predict(stack)
    pf_delta = feat["pf_ancc"].to_numpy(float) - feat["last_known_tvt"].to_numpy(float)
    pred = feat["last_known_tvt"].to_numpy(float) + apply_pp(feat, ridge_raw, pf_delta, **pp_params)

    d = feat[["id", "well"]].copy()
    d["pred"] = pred
    d = sg_smooth(d, "pred")
    return d[["id", "pred"]].rename(columns={"pred": "ridge"})

def _selector_full_array_no_leak(wid, hw_te, tw_te):
    # Important: no tvt_from_contacts branch here.
    # That branch can leak in local/train-style validation and does not exist for hidden test.
    selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(hw_te)

    try:
        pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_te, n_particles=500, n_seeds=128)
        tvt_pf = pf_by_scale["pf_scale_8"]
    except Exception:
        last_known = hw_te["TVT_input"].dropna()
        last_val = float(last_known.iloc[-1]) if len(last_known) else 0.0
        tvt_pf = hw_te["TVT_input"].fillna(last_val).to_numpy(float)
        pf_by_scale = {f"pf_scale_{scale:g}": tvt_pf.copy() for scale in SELECTOR_SCALES}

    try:
        tvt_beam = run_beam_ensemble(hw_te, tw_te)
    except Exception:
        tvt_beam = tvt_pf.copy()

    last_known = hw_te["TVT_input"].dropna()
    last_known_tvt = float(last_known.iloc[-1]) if len(last_known) else float(np.nanmean(tvt_pf))
    return np.asarray(apply_selector_variant(selector_variant, pf_by_scale, tvt_beam, last_known_tvt), float)

def _replay_selector_for_score_idx(wid, hw_mask, tw, score_idx):
    arr = _selector_full_array_no_leak(wid, hw_mask, tw)
    return pd.DataFrame({
        "id": [f"{wid}_{i}" for i in score_idx],
        "selector": arr[score_idx].astype(float),
    })

def _candidate_table_from_replay(pref):
    cand = pref[["id", "ridge", "selector"]].copy()
    cand["sp45_like"] = 0.3 * cand["ridge"].to_numpy(float) + 0.7 * cand["selector"].to_numpy(float)

    sources = ["ridge", "selector", "sp45_like"]
    for src in sources:
        base_v = cand[src].to_numpy(float)
        for dlt in SHIFT_DELTAS:
            name = f"{src}_shift_{dlt:+.0f}"
            cand[name] = base_v + dlt
    return cand

def _candidate_values_eval(g, cand_name):
    if cand_name in ["ridge", "selector", "sp45_like"]:
        return g[cand_name].to_numpy(float)
    if "_shift_" in cand_name:
        src, dlt_s = cand_name.split("_shift_")
        dlt = float(dlt_s)
        return g[src].to_numpy(float) + dlt
    raise KeyError(cand_name)

records = []
pred_final = base["sp45_like"].to_numpy(float).copy()

t0 = time.time()
for iw, (wid, g) in enumerate(base.groupby("well", sort=False), start=1):
    if iw % 25 == 0:
        print(f"submit-A prefix replay {iw}/{base['well'].nunique()} elapsed={time.time()-t0:.1f}s", flush=True)

    hw_path = DATA / "test" / f"{wid}__horizontal_well.csv"
    tw_path = DATA / "test" / f"{wid}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        continue

    hw = pd.read_csv(hw_path).reset_index(drop=True)
    tw = pd.read_csv(tw_path).sort_values("TVT")
    score_idx = _build_score_idx(hw)
    if score_idx is None:
        continue

    hw_mask = hw.copy()
    # Use iloc-equivalent assignment to avoid label/position ambiguity.
    hw_mask.loc[score_idx, "TVT_input"] = np.nan

    ridge_pref = _ridge_pp_from_masked_hw(wid, hw_mask, tw_path)
    selector_pref = _replay_selector_for_score_idx(wid, hw_mask, tw, score_idx)
    if ridge_pref is None or selector_pref is None:
        continue

    pref = ridge_pref.merge(selector_pref, on="id", how="inner")
    pref["row_idx"] = pref["id"].astype(str).str[9:].astype(int)
    pref = pref[pref["row_idx"].isin(score_idx)].copy()
    if len(pref) < PREFIX_MIN_SCORE_ROWS:
        continue

    truth = hw.loc[pref["row_idx"].to_numpy(int), "TVT_input"].to_numpy(float)
    if not np.isfinite(truth).all():
        continue

    cand_pref = _candidate_table_from_replay(pref)
    cand_cols = [c for c in cand_pref.columns if c != "id"]

    rmses = {}
    for c in cand_cols:
        v = cand_pref[c].to_numpy(float)
        if np.isfinite(v).all() and len(v) == len(truth):
            rmses[c] = float(np.sqrt(np.mean((v - truth) ** 2)))

    if "sp45_like" not in rmses:
        continue

    base_rmse = rmses["sp45_like"]
    best_cand = min(rmses, key=rmses.get)
    best_rmse = rmses[best_cand]
    gain = base_rmse - best_rmse
    apply = bool(gain > GAIN_THR and best_cand != "sp45_like")

    idx = g.index.to_numpy()
    base_eval = g["sp45_like"].to_numpy(float)
    if apply:
        chosen_eval = _candidate_values_eval(g, best_cand)
        pred = BLEND_REF * base_eval + (1.0 - BLEND_REF) * chosen_eval
        pred = _clip_to_base(base_eval, pred, ROW_MOVE_CAP)
        pred_final[idx] = pred

    move = pred_final[idx] - base_eval
    records.append({
        "well": wid,
        "rows": int(len(g)),
        "score_rows": int(len(pref)),
        "base_pref_rmse": float(base_rmse),
        "best_cand": best_cand,
        "best_pref_rmse": float(best_rmse),
        "prefix_gain": float(gain),
        "apply": apply,
        "mean_abs_move": float(np.mean(np.abs(move))),
        "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    })

diag = pd.DataFrame(records)
diag.to_csv(WORK / "submit_A_prefix_replay_hardselect_diag.csv", index=False)

sub = base[["id"]].copy()
sub["tvt"] = pred_final
sub = sample[["id"]].merge(sub, on="id", how="left")

if sub["tvt"].isna().any():
    raise RuntimeError("Submit A produced missing predictions")
if not np.isfinite(sub["tvt"].to_numpy(float)).all():
    raise RuntimeError("Submit A produced non-finite predictions")

sub.to_csv(WORK / "submission_submit_A_prefix_replay_sp45_branch.csv", index=False)
sub.to_csv(WORK / "submission.csv", index=False)

move_all = sub["tvt"].to_numpy(float) - base["sp45_like"].to_numpy(float)
audit = {
    "rows": int(len(sub)),
    "wells": int(sample["well"].nunique()),
    "scored_wells": int(len(diag)),
    "applied_wells": int(diag["apply"].sum()) if len(diag) else 0,
    "gain_thr": GAIN_THR,
    "blend_ref": BLEND_REF,
    "row_move_cap": ROW_MOVE_CAP,
    "mean_abs_move_vs_sp45_like": float(np.mean(np.abs(move_all))),
    "p95_abs_move_vs_sp45_like": float(np.quantile(np.abs(move_all), 0.95)),
    "max_abs_move_vs_sp45_like": float(np.max(np.abs(move_all))),
    "output": str(WORK / "submission_submit_A_prefix_replay_sp45_branch.csv"),
}
with open(WORK / "submit_A_prefix_replay_hardselect_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, sort_keys=True)

print("SUBMIT A PREFIX REPLAY DONE")
print(audit)
if len(diag):
    print(diag.sort_values("prefix_gain", ascending=False).head(30).to_string(index=False))

sub
```

---

## Kaggle 跑完后看这几个数

看输出的 `submit_A_prefix_replay_hardselect_audit.json`：

- `applied_wells`：如果 hidden 285 口里改动远超本地比例，先别交。
- `mean_abs_move_vs_sp45_like`：建议不要比本地 E2E 区间大太多。
- `p95_abs_move_vs_sp45_like`：如果明显超过 4-6 ft，要谨慎。

如果想更保守：

```python
GAIN_THR = 1.0
ROW_MOVE_CAP = 4.0
```

如果想更激进：

```python
GAIN_THR = 0.0
ROW_MOVE_CAP = 12.0
```

但第一发按默认。
