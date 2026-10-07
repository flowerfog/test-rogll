# Complete Submit Patch: Full-Row Prefix Selector Before Existing CP12 + Gold

这份是完整提交链路，不是上一版的裸 probe。

核心原则：

- 保留原来的 `0.3 * ridge_pp + 0.7 * selector` 作为兜底。
- 用可见 prefix replay 只决定“这口井是否应该更偏 ridge 或 selector”。
- 决策不直接替代全链路，而是产出一个新的 sp45 分支输入。
- 后面继续跑原 notebook 的 projection、fleongg blend、CP12、Gold。

这版仍然先只纳入两个当前 notebook 能完整 replay 的候选：

- `ridge_pp`
- `selector/PF`

后续如果这条链路在 LB 有正反馈，再把 `fleongg/sp45/CP12/ridge_clip` 做 full-row 化纳入候选池。

---

## 运行顺序总览

1. 正常运行原 notebook 到 Cell 7。
2. 在 Cell 7 末尾加 **Cell A**。
3. 按 **Cell B** 修改 Cell 8 / Cell 9。
4. 正常运行 Cell 8 到 Cell 16。
5. 用 **Cell C** 替换原 Cell 17。
6. 正常运行 Cell 18 projection。
7. 正常运行 Cell 19，保存新的 `sp45_projection_submission.csv`。
8. 正常继续跑 fleongg stack / Cell 29 blend+projection。
9. 正常运行 Cell 32 保存 `submission_before_gold.csv`。
10. 正常运行 Cell 35 CP12。
11. 正常运行 Cell 36 Gold。
12. 最终提交 `submission.csv`。

不要在 Cell C 后直接提交。Cell C 只是替换 sp45 分支，不是最终提交。

---

## Cell A：加在 Cell 7 末尾

位置：`oof_preds = {}`、`test_preds = {}`、`overall_scores = {}`、`fold_scores = {}` 后面。

```python
# Store fitted first-level models so replay-prefix rows can be predicted later.
base_level_trainers = {}
```

---

## Cell B：修改 Cell 8 / Cell 9

### Cell 8 LightGBM 循环

在：

```python
oof_preds[f"lightgbm-{i+1}"] = trainer.oof_preds
test_preds[f"lightgbm-{i+1}"] = trainer.predict(X_test)
```

前面插入：

```python
base_level_trainers[f"lightgbm-{i+1}"] = trainer
```

### Cell 9 CatBoost 循环

在：

```python
oof_preds[f"catboost-{i+1}"] = trainer.oof_preds
test_preds[f"catboost-{i+1}"] = trainer.predict(X_test)
```

前面插入：

```python
base_level_trainers[f"catboost-{i+1}"] = trainer
```

---

## Cell C：完整替换原 Cell 17

原 Cell 17 是：

```python
sub = (
    sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
       .assign(tvt=lambda x: 0.3 * x['tvt_1'] + 0.7 * x['tvt_2'])
       [['id', 'tvt']]
)
sub.to_csv("submission.csv", index=False)
sub
```

用下面这一整段替换。

```python
# ============================================================
# Complete full-row visible-prefix selector for the sp45 branch
#
# Replaces fixed:
#     0.3 * ridge_pp + 0.7 * selector
#
# Fallback remains exactly the fixed blend.
# If prefix replay is not confident, the well is unchanged.
# After this cell, continue the original notebook:
#     Cell 18 projection -> Cell 19 save sp45_projection_submission.csv
#     -> fleongg blend -> CP12 -> Gold
# ============================================================
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

# Conservative first-submit config.
# Local tail-prefix scan liked gain_thr=0.25, blend_ref=0.5.
PREFIX_TAIL_N = 1000
PREFIX_SCORE_FRAC = 0.50
PREFIX_MIN_SCORE_ROWS = 80
PREFIX_GAIN_THR = 0.25
FINAL_BLEND_REF = 0.50

sample = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sample["id"] = sample["id"].astype(str)
sample["well"] = sample["id"].str[:8]
sample["row_idx"] = sample["id"].str[9:].astype(int)

train_hw_files = sorted(glob.glob(str(CFG.dataset_path / "train" / "*__horizontal_well.csv")))
train_wells = {os.path.basename(f).split("__")[0] for f in train_hw_files}

def _submission_with_meta(df, pred_col="tvt"):
    out = df[["id", pred_col]].rename(columns={pred_col: "tvt"}).copy()
    out["id"] = out["id"].astype(str)
    out["well"] = out["id"].str[:8]
    out["row_idx"] = out["id"].str[9:].astype(int)
    out["tvt"] = out["tvt"].astype(float)
    return out

ridge_final = _submission_with_meta(sub_1, "tvt")
selector_final = _submission_with_meta(sub_2, "tvt")

fixed = ridge_final[["id", "well", "row_idx", "tvt"]].rename(columns={"tvt": "ridge"})
fixed = fixed.merge(
    selector_final[["id", "tvt"]].rename(columns={"tvt": "selector"}),
    on="id",
    how="inner",
)
if len(fixed) != len(sample):
    raise RuntimeError(f"fixed candidate merge mismatch: fixed={len(fixed)} sample={len(sample)}")

fixed["fixed_blend"] = 0.3 * fixed["ridge"].astype(float) + 0.7 * fixed["selector"].astype(float)

def _predict_stack_features(feat_df):
    if not isinstance(oof_preds, pd.DataFrame):
        raise RuntimeError("oof_preds must already be converted to DataFrame in Cell 10.")
    X_new = feat_df[features]
    cols = []
    for name in list(oof_preds.columns):
        if name not in base_level_trainers:
            raise RuntimeError(
                f"missing first-level trainer for {name}. "
                "Add base_level_trainers[...] = trainer in Cell 8/9."
            )
        cols.append(pd.Series(base_level_trainers[name].predict(X_new), name=name))
    return pd.concat(cols, axis=1)

def _ridge_pp_predict_from_features(feat_df):
    stack_new = _predict_stack_features(feat_df)
    ridge_raw = ridge_trainer.predict(stack_new)
    pf_delta = feat_df["pf_ancc"].to_numpy(float) - feat_df["last_known_tvt"].to_numpy(float)
    pred = feat_df["last_known_tvt"].to_numpy(float) + apply_pp(
        feat_df,
        ridge_raw,
        pf_delta,
        **pp_params,
    )
    d = feat_df[["id", "well"]].copy()
    d["pred"] = pred
    return sg_smooth(d, "pred")

def _selector_predict_full_array(wid, hw_te, tw_te):
    tvt_phys = None
    hw_tr = None
    tw_tr = None

    # Keep original public-visible branch. Hidden test wells normally are not in train.
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

def _build_replay_score_idx(hw):
    known_idx = np.where(hw["TVT_input"].notna().to_numpy())[0]
    if len(known_idx) < PREFIX_MIN_SCORE_ROWS + 30:
        return None

    tail = known_idx[-min(PREFIX_TAIL_N, len(known_idx)):]
    n_score = int(round(len(tail) * PREFIX_SCORE_FRAC))
    n_score = max(PREFIX_MIN_SCORE_ROWS, min(n_score, len(tail) - 30))
    if n_score < PREFIX_MIN_SCORE_ROWS:
        return None
    return tail[-n_score:]

def _replay_ridge_for_well(wid, hw, tw_path, score_idx):
    hw_mask = hw.copy()
    hw_mask.loc[score_idx, "TVT_input"] = np.nan

    with tempfile.TemporaryDirectory(dir=str(WORK)) as td:
        tmp_hw = Path(td) / f"{wid}__horizontal_well.csv"
        hw_mask.to_csv(tmp_hw, index=False)
        feat = build_well(str(tmp_hw), str(tw_path), is_train=False)

    if feat is None or len(feat) == 0:
        return None

    pred_df = _ridge_pp_predict_from_features(feat)
    return pred_df[["id", "pred"]].rename(columns={"pred": "ridge"})

def _replay_selector_for_well(wid, hw, tw, score_idx):
    hw_mask = hw.copy()
    hw_mask.loc[score_idx, "TVT_input"] = np.nan
    arr = _selector_predict_full_array(wid, hw_mask, tw)
    return pd.DataFrame({
        "id": [f"{wid}_{i}" for i in score_idx],
        "selector": arr[score_idx].astype(float),
    })

decision_rows = []
test_wells = sorted(sample["well"].unique())

for iw, wid in enumerate(test_wells, start=1):
    if iw % 25 == 0:
        print(f"visible-prefix replay {iw}/{len(test_wells)}", flush=True)

    hw_path = CFG.dataset_path / "test" / f"{wid}__horizontal_well.csv"
    tw_path = CFG.dataset_path / "test" / f"{wid}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        continue

    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values("TVT")
    score_idx = _build_replay_score_idx(hw)
    if score_idx is None:
        continue

    ridge_pref = _replay_ridge_for_well(wid, hw, tw_path, score_idx)
    selector_pref = _replay_selector_for_well(wid, hw, tw, score_idx)
    if ridge_pref is None or selector_pref is None:
        continue

    pref = ridge_pref.merge(selector_pref, on="id", how="inner")
    pref["row_idx"] = pref["id"].astype(str).str[9:].astype(int)
    if len(pref) < PREFIX_MIN_SCORE_ROWS:
        continue

    truth = hw.loc[pref["row_idx"].to_numpy(int), "TVT_input"].to_numpy(float)
    if not np.isfinite(truth).all():
        continue

    ridge_v = pref["ridge"].to_numpy(float)
    selector_v = pref["selector"].to_numpy(float)
    fixed_v = 0.3 * ridge_v + 0.7 * selector_v

    base_rmse = float(np.sqrt(np.mean((fixed_v - truth) ** 2)))
    ridge_rmse = float(np.sqrt(np.mean((ridge_v - truth) ** 2)))
    selector_rmse = float(np.sqrt(np.mean((selector_v - truth) ** 2)))

    if ridge_rmse <= selector_rmse:
        best_cand, best_rmse = "ridge", ridge_rmse
    else:
        best_cand, best_rmse = "selector", selector_rmse

    best_gain = base_rmse - best_rmse
    decision_rows.append({
        "well": wid,
        "score_rows": int(len(pref)),
        "base_pref_rmse": base_rmse,
        "ridge_pref_rmse": ridge_rmse,
        "selector_pref_rmse": selector_rmse,
        "best_cand": best_cand,
        "best_gain": best_gain,
        "apply": bool(best_gain >= PREFIX_GAIN_THR),
    })

dec = pd.DataFrame(decision_rows)
dec.to_csv(WORK / "visible_prefix_ridge_selector_decisions.csv", index=False)

choice = dict(zip(dec.loc[dec["apply"], "well"], dec.loc[dec["apply"], "best_cand"])) if len(dec) else {}

pred = fixed["fixed_blend"].to_numpy(float).copy()
for cand in ["ridge", "selector"]:
    m = fixed["well"].map(choice).eq(cand).to_numpy()
    base_vals = fixed.loc[m, "fixed_blend"].to_numpy(float)
    cand_vals = fixed.loc[m, cand].to_numpy(float)
    pred[m] = FINAL_BLEND_REF * base_vals + (1.0 - FINAL_BLEND_REF) * cand_vals

sub = fixed[["id"]].copy()
sub["tvt"] = pred
sub = sample[["id"]].merge(sub, on="id", how="left")

if sub["tvt"].isna().any():
    raise RuntimeError("visible-prefix selector produced missing predictions")
if not np.isfinite(sub["tvt"].to_numpy(float)).all():
    raise RuntimeError("visible-prefix selector produced non-finite predictions")

sub.to_csv(WORK / "submission_sp45_visible_prefix_selector_raw.csv", index=False)
sub.to_csv(WORK / "submission.csv", index=False)

base_order = sample[["id"]].merge(fixed[["id", "fixed_blend"]], on="id", how="left")
move = sub["tvt"].to_numpy(float) - base_order["fixed_blend"].to_numpy(float)

summary = {
    "rows": int(len(sub)),
    "wells_total": int(sample["well"].nunique()),
    "wells_scored": int(len(dec)),
    "wells_changed": int(len(choice)),
    "gain_thr": PREFIX_GAIN_THR,
    "blend_ref": FINAL_BLEND_REF,
    "mean_abs_move_vs_fixed_sp45": float(np.abs(move).mean()),
    "p95_abs_move_vs_fixed_sp45": float(np.quantile(np.abs(move), 0.95)),
    "output": str(WORK / "submission_sp45_visible_prefix_selector_raw.csv"),
}

print("visible-prefix sp45 branch done")
print(summary)
if len(dec):
    print(dec.sort_values("best_gain", ascending=False).head(30).to_string(index=False))

sub
```

---

## 关键解释

这份完整提交链路和上一版的区别：

- 上一版是裸 probe：Cell C 后可以停，所以可能低于 6.991。
- 这版是完整链路：Cell C 只替换 sp45 分支，然后继续原 notebook 的后半段。
- 没有 prefix 证据的井不动，还是原固定 blend。
- prefix replay 明显胜出的井才偏向 `ridge` 或 `selector`。

所以这版才适合拿去和 6.991 比。

---

## 提交建议

第一次提交用默认：

```python
PREFIX_GAIN_THR = 0.25
FINAL_BLEND_REF = 0.50
```

如果这版 LB 还是差，第二版调保守：

```python
PREFIX_GAIN_THR = 1.00
FINAL_BLEND_REF = 0.70
```

含义：更少井会改，且改动更靠近原 fixed blend。
