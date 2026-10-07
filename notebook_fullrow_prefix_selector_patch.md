# Full-Row Visible-Prefix Candidate Selector Patch

目标：替换原来固定 `0.3 * ridge_pp + 0.7 * selector` 的 Cell 17，改成“可见尾段 replay 自监督选择候选，再应用到隐藏 eval 段”。

这版先只纳入两个可部署候选：

- `ridge_pp`
- `selector/PF`

原因：本地 tail-prefix 结论里吃肉最多的是 `ridge/selector`，而这两个候选在当前 notebook 里能用同一套函数生成 replay 和 final 两套预测。`fleongg/sp45/CP12` 需要更深的 full-row 化，下一版再加。

---

## Cell A: 在 Cell 7 末尾加

位置：`oof_preds = {}`、`test_preds = {}` 后面。

```python
# Keep fitted base learners so replay rows can be predicted too.
base_level_trainers = {}
```

---

## Cell B: 修改 Cell 8 / Cell 9

Cell 8 的 LightGBM 循环，在这三行前：

```python
oof_preds[f"lightgbm-{i+1}"] = trainer.oof_preds
test_preds[f"lightgbm-{i+1}"] = trainer.predict(X_test)
```

插入：

```python
base_level_trainers[f"lightgbm-{i+1}"] = trainer
```

Cell 9 的 CatBoost 循环，在这三行前：

```python
oof_preds[f"catboost-{i+1}"] = trainer.oof_preds
test_preds[f"catboost-{i+1}"] = trainer.predict(X_test)
```

插入：

```python
base_level_trainers[f"catboost-{i+1}"] = trainer
```

---

## Cell C: 用这个替换 Cell 17

位置：Cell 16 `sub_2 = pd.DataFrame(rows)` 之后、Cell 18 projection 之前。

```python
# ============================================================
# Visible-prefix candidate selector
# Replaces fixed 0.3 ridge_pp + 0.7 selector blend.
#
# Needs already-built:
#   test_df2 with ridge_pp final predictions in column pred
#   sub_1  = ridge_pp final submission rows
#   sub_2  = selector/PF final submission rows
#   base_level_trainers, ridge_trainer, features, pp_params
#   build_well, apply_pp, sg_smooth
#   run_pf_lik_ensemble_scales, run_beam_ensemble, apply_selector_variant
# ============================================================
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

PREFIX_TAIL_N = 1000
PREFIX_SCORE_FRAC = 0.50
PREFIX_MIN_SCORE_ROWS = 80

PREFIX_GAIN_THR = 0.25
FINAL_BLEND_REF = 0.50

def _submission_with_meta(df, pred_col="tvt"):
    out = df[["id", pred_col]].rename(columns={pred_col: "tvt"}).copy()
    out["id"] = out["id"].astype(str)
    out["well"] = out["id"].str[:8]
    out["row_idx"] = out["id"].str[9:].astype(int)
    out["tvt"] = out["tvt"].astype(float)
    return out

ridge_final = _submission_with_meta(sub_1, "tvt")
selector_final = _submission_with_meta(sub_2, "tvt")

sample = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sample["well"] = sample["id"].astype(str).str[:8]
sample["row_idx"] = sample["id"].astype(str).str[9:].astype(int)

train_hw_files = sorted(glob.glob(str(CFG.dataset_path / "train" / "*__horizontal_well.csv")))
train_wells = {os.path.basename(f).split("__")[0] for f in train_hw_files}

def _predict_stack_features(feat_df):
    X_new = feat_df[features]
    cols = []
    for name in list(oof_preds.columns):
        if name not in base_level_trainers:
            raise RuntimeError(f"missing base trainer for {name}; store it in Cell 8/9")
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

    # Keep same visible-well branch as original selector cell.
    if wid in train_wells:
        try:
            hw_tr, tw_tr = load_well(wid, "train")
            hw_te = hw_te.copy()
            if len(hw_tr) == len(hw_te):
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

def _build_replay_mask(wid, hw):
    known_idx = np.where(hw["TVT_input"].notna().to_numpy())[0]
    if len(known_idx) < PREFIX_MIN_SCORE_ROWS + 30:
        return None

    tail = known_idx[-min(PREFIX_TAIL_N, len(known_idx)):]
    n_score = int(round(len(tail) * PREFIX_SCORE_FRAC))
    n_score = max(PREFIX_MIN_SCORE_ROWS, min(n_score, len(tail) - 30))
    if n_score < PREFIX_MIN_SCORE_ROWS:
        return None

    # Hide the latest part of the visible prefix, then use its true TVT_input for candidate scoring.
    score_idx = tail[-n_score:]
    if len(score_idx) < PREFIX_MIN_SCORE_ROWS:
        return None
    return score_idx

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
    ids = [f"{wid}_{i}" for i in score_idx]
    return pd.DataFrame({"id": ids, "selector": arr[score_idx].astype(float)})

decision_rows = []
for iw, wid in enumerate(sorted(sample["well"].unique()), start=1):
    if iw % 25 == 0:
        print(f"visible-prefix selector well {iw}/{sample['well'].nunique()}", flush=True)

    hw_path = CFG.dataset_path / "test" / f"{wid}__horizontal_well.csv"
    tw_path = CFG.dataset_path / "test" / f"{wid}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        continue

    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values("TVT")
    score_idx = _build_replay_mask(wid, hw)
    if score_idx is None:
        continue

    ridge_pref = _replay_ridge_for_well(wid, hw, tw_path, score_idx)
    selector_pref = _replay_selector_for_well(wid, hw, tw, score_idx)
    if ridge_pref is None or selector_pref is None:
        continue

    pref = ridge_pref.merge(selector_pref, on="id", how="inner")
    pref["row_idx"] = pref["id"].astype(str).str[9:].astype(int)
    truth = hw.loc[pref["row_idx"].to_numpy(int), "TVT_input"].to_numpy(float)

    cand_rmse = {}
    for c in ["ridge", "selector"]:
        v = pref[c].to_numpy(float)
        cand_rmse[c] = float(np.sqrt(np.mean((v - truth) ** 2)))

    # Original branch is fixed 0.3 ridge + 0.7 selector.
    base_pref = 0.3 * pref["ridge"].to_numpy(float) + 0.7 * pref["selector"].to_numpy(float)
    base_rmse = float(np.sqrt(np.mean((base_pref - truth) ** 2)))
    best_cand = min(cand_rmse, key=cand_rmse.get)
    best_gain = base_rmse - cand_rmse[best_cand]

    decision_rows.append({
        "well": wid,
        "score_rows": int(len(pref)),
        "base_pref_rmse": base_rmse,
        "ridge_pref_rmse": cand_rmse["ridge"],
        "selector_pref_rmse": cand_rmse["selector"],
        "best_cand": best_cand,
        "best_gain": best_gain,
        "apply": bool(best_gain >= PREFIX_GAIN_THR),
    })

dec = pd.DataFrame(decision_rows)
dec.to_csv(WORK / "visible_prefix_candidate_decisions.csv", index=False)

print("visible-prefix decisions")
print({
    "wells_scored": int(len(dec)),
    "wells_apply": int(dec["apply"].sum()) if len(dec) else 0,
    "gain_thr": PREFIX_GAIN_THR,
    "blend_ref": FINAL_BLEND_REF,
})
if len(dec):
    print(dec.sort_values("best_gain", ascending=False).head(20).to_string(index=False))

# Build final eval submission.
base_fixed = ridge_final[["id", "well", "row_idx", "tvt"]].rename(columns={"tvt": "base_fixed"})
base_fixed = base_fixed.merge(
    selector_final[["id", "tvt"]].rename(columns={"tvt": "selector"}),
    on="id",
    how="inner",
)
base_fixed["base_fixed"] = 0.3 * base_fixed["base_fixed"].astype(float) + 0.7 * base_fixed["selector"].astype(float)

ridge_eval = ridge_final[["id", "tvt"]].rename(columns={"tvt": "ridge"})
selector_eval = selector_final[["id", "tvt"]].rename(columns={"tvt": "selector"})

final = base_fixed[["id", "well", "base_fixed"]].merge(ridge_eval, on="id", how="inner")
final = final.merge(selector_eval, on="id", how="inner")
if len(final) != len(sample):
    raise RuntimeError(f"final merge row mismatch: final={len(final)} sample={len(sample)}")

choice = dict(zip(dec.loc[dec["apply"], "well"], dec.loc[dec["apply"], "best_cand"])) if len(dec) else {}

pred = final["base_fixed"].to_numpy(float).copy()
for cand in ["ridge", "selector"]:
    m = final["well"].map(choice).eq(cand).to_numpy()
    cand_vals = final.loc[m, cand].to_numpy(float)
    base_vals = final.loc[m, "base_fixed"].to_numpy(float)
    pred[m] = FINAL_BLEND_REF * base_vals + (1.0 - FINAL_BLEND_REF) * cand_vals

out = final[["id"]].copy()
out["tvt"] = pred
out = sample[["id"]].merge(out, on="id", how="left")
if out["tvt"].isna().any():
    raise RuntimeError("visible-prefix selector produced missing predictions")
if not np.isfinite(out["tvt"].to_numpy(float)).all():
    raise RuntimeError("visible-prefix selector produced non-finite predictions")

out.to_csv(WORK / "submission_visible_prefix_ridge_selector.csv", index=False)
out.to_csv(WORK / "submission.csv", index=False)

move = out["tvt"].to_numpy(float) - base_fixed["base_fixed"].to_numpy(float)
print("wrote visible-prefix selected submission.csv")
print({
    "output": str(WORK / "submission_visible_prefix_ridge_selector.csv"),
    "rows": int(len(out)),
    "changed_wells": int(len(choice)),
    "mean_abs_move_vs_fixed_blend": float(np.abs(move).mean()),
    "p95_abs_move_vs_fixed_blend": float(np.quantile(np.abs(move), 0.95)),
})
```

---

## Run order

1. Run notebook normally through Cell 7.
2. Add Cell A.
3. Run modified Cell 8 and Cell 9 with Cell B inserted.
4. Run Cell 10 through Cell 16 normally.
5. Replace Cell 17 with Cell C and run it.
6. Run Cell 18 projection.
7. Save `sp45_projection_submission.csv` if you still want the old blend/fle path, but for this branch do **not** run CP12 or Gold after it.

For a clean A/B submission, stop after Cell 18 and the final audit cell. Keep this branch isolated from CP12/Gold.
