"""V219: CatBoost architecture confirmation on the frozen V217 representation.

The feature cache, folds, 1024-row-per-well sampling, residual target, and
trajectory readout are inherited verbatim from V217.  Only the tree learner is
changed.  A read-only 2D blend audit follows training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "likpf_residual_foldclean_spatial_v217.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
start = source.find("def _new_model(fold):")
end = source.find("def _smooth(frame, prediction):")
if start < 0 or end <= start:
    raise RuntimeError("V219 could not locate the audited V217 model block")

cat_block = '''def _new_model(fold):
    from catboost import CatBoostRegressor
    task_type = "CPU" if os.environ.get("V219_CPU", "0") == "1" else "GPU"
    iterations = 1800 if task_type == "CPU" else 3000
    params = dict(
        iterations=iterations,
        depth=8,
        learning_rate=0.035,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=8.0,
        random_strength=0.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        random_seed=SEED + fold,
        task_type=task_type,
        od_type="Iter",
        od_wait=180,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )
    if task_type == "GPU":
        params["devices"] = "0"
    return CatBoostRegressor(**params)


def _fit(train, valid, fold):
    model = _new_model(fold)
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=(_matrix(valid), valid["residual_target"].to_numpy(float)),
        use_best_model=True,
        verbose=100,
    )
    try:
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    except Exception as exc:
        if os.environ.get("V219_CPU", "0") == "1":
            raise
        print("V219 GPU fallback", repr(exc), flush=True)
        os.environ["V219_CPU"] = "1"
        model = _new_model(fold)
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    return model


'''
source = source[:start] + cat_block + source[end:]
old_predict = 'residual = model.predict(_matrix(valid), num_iteration=model.best_iteration_)'
if source.count(old_predict) != 1:
    raise RuntimeError("V219 could not locate the V217 prediction call")
source = source.replace(old_predict, 'residual = model.predict(_matrix(valid))', 1)
old_iteration = 'int(model.best_iteration_ or model.n_estimators)'
if source.count(old_iteration) != 1:
    raise RuntimeError("V219 could not locate the V217 iteration report")
source = source.replace(old_iteration, 'int(model.get_best_iteration() + 1)', 1)
source = source.replace('SEED = 217', 'SEED = 219', 1)
source = source.replace(
    'likpf_residual_foldclean_spatial_v217',
    'likpf_residual_foldclean_spatial_catboost_v219',
)
source = source.replace('V217', 'V219').replace('v217', 'v219')

exec(compile(source, str(SOURCE), "exec"), globals())

# Frozen OOF architecture/blend audit.
oof_root = ROOT / "oof_exports"
paths = {
    "base": oof_root / "likpf_residual_gbdt_full5_v215" / "oof.parquet",
    "lgb": oof_root / "likpf_residual_foldclean_spatial_v217" / "oof.parquet",
    "cat": oof_root / "likpf_residual_foldclean_spatial_catboost_v219" / "oof.parquet",
}
for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)

base = pd.read_parquet(paths["base"])[["id", "candidate"]].rename(columns={"candidate": "base"})
lgb = pd.read_parquet(paths["lgb"])[["id", "well", "fold", "md_since", "tvt_true", "candidate"]].rename(
    columns={"candidate": "lgb"}
)
cat = pd.read_parquet(paths["cat"])[["id", "candidate"]].rename(columns={"candidate": "cat"})
frame = lgb.merge(base, on="id", how="left", validate="one_to_one").merge(
    cat, on="id", how="left", validate="one_to_one"
)
if len(frame) != 3_783_989 or frame[["base", "cat"]].isna().any().any():
    raise RuntimeError("Incomplete V215/V217/V219 alignment")

truth = frame["tvt_true"].to_numpy(float)
base_pred = frame["base"].to_numpy(float)
lgb_pred = frame["lgb"].to_numpy(float)
cat_pred = frame["cat"].to_numpy(float)
folds = frame["fold"].to_numpy(int)


def _rmse(mask, prediction):
    error = prediction[mask] - truth[mask]
    return float(np.sqrt(np.mean(error * error)))


all_mask = np.ones(len(frame), dtype=bool)
base_rmse = _rmse(all_mask, base_pred)
lgb_rmse = _rmse(all_mask, lgb_pred)
cat_rmse = _rmse(all_mask, cat_pred)
lgb_move = lgb_pred - base_pred
cat_move = cat_pred - base_pred

scan_rows = []
grid = np.round(np.arange(0.0, 1.2501, 0.05), 6)
for lgb_weight in grid:
    for cat_weight in grid:
        if lgb_weight + cat_weight > 1.25 + 1e-9:
            continue
        prediction = base_pred + lgb_weight * lgb_move + cat_weight * cat_move
        row = {
            "lgb_weight": float(lgb_weight),
            "cat_weight": float(cat_weight),
            "rmse": _rmse(all_mask, prediction),
        }
        row["gain_vs_v215"] = base_rmse - row["rmse"]
        row["gain_vs_v217"] = lgb_rmse - row["rmse"]
        fold_gains = []
        for held in range(1, 6):
            mask = folds == held
            gain = _rmse(mask, lgb_pred) - _rmse(mask, prediction)
            row[f"fold{held}_gain_vs_v217"] = gain
            fold_gains.append(gain)
        row["positive_folds_vs_v217"] = int(sum(value > 0 for value in fold_gains))
        row["minimum_fold_gain_vs_v217"] = float(min(fold_gains))
        scan_rows.append(row)
scan = pd.DataFrame(scan_rows).sort_values("rmse", kind="stable").reset_index(drop=True)

# Select the 2D blend on the other four folds and apply once to each held fold.
nested_prediction = base_pred.copy()
nested_rows = []
for held in range(1, 6):
    train_mask = folds != held
    valid_mask = folds == held
    choices = []
    for row in scan_rows:
        prediction = base_pred + row["lgb_weight"] * lgb_move + row["cat_weight"] * cat_move
        choices.append((
            _rmse(train_mask, prediction),
            row["lgb_weight"] + row["cat_weight"],
            row["lgb_weight"],
            row["cat_weight"],
        ))
    inner_rmse, _, lgb_weight, cat_weight = min(choices)
    nested_prediction[valid_mask] = (
        base_pred[valid_mask]
        + lgb_weight * lgb_move[valid_mask]
        + cat_weight * cat_move[valid_mask]
    )
    nested_rows.append({
        "fold": held,
        "lgb_weight": lgb_weight,
        "cat_weight": cat_weight,
        "inner_rmse": inner_rmse,
        "v217_rmse": _rmse(valid_mask, lgb_pred),
        "rmse": _rmse(valid_mask, nested_prediction),
    })
for row in nested_rows:
    row["gain_vs_v217"] = row["v217_rmse"] - row["rmse"]

best = scan.iloc[0].to_dict()
stable = scan.loc[
    (scan["positive_folds_vs_v217"] >= 4)
    & (scan["minimum_fold_gain_vs_v217"] >= -0.05)
]
best_stable = stable.iloc[0].to_dict() if len(stable) else None
nested_rmse = _rmse(all_mask, nested_prediction)
audit = {
    "experiment": "likpf_residual_foldclean_spatial_catboost_v219_architecture_audit",
    "changed_variable": "LightGBM -> CatBoost only",
    "rmse": {"v215": base_rmse, "v217_lgb": lgb_rmse, "v219_cat": cat_rmse},
    "error_correlation_lgb_cat": float(np.corrcoef(lgb_pred - truth, cat_pred - truth)[0, 1]),
    "move_correlation_lgb_cat": float(np.corrcoef(lgb_move, cat_move)[0, 1]),
    "best_fixed": best,
    "best_stable_fixed": best_stable,
    "nested": {
        "rmse": nested_rmse,
        "gain_vs_v217": lgb_rmse - nested_rmse,
        "folds": nested_rows,
    },
    "checks": {
        "cat_rmse_le_9p8": cat_rmse <= 9.8,
        "error_correlation_le_0p98": float(np.corrcoef(lgb_pred - truth, cat_pred - truth)[0, 1]) <= 0.98,
        "stable_blend_gain_vs_v217_ge_0p05": bool(best_stable is not None and best_stable["gain_vs_v217"] >= 0.05),
        "nested_gain_vs_v217_ge_0p05": lgb_rmse - nested_rmse >= 0.05,
        "nested_positive_folds_ge_4": sum(row["gain_vs_v217"] > 0 for row in nested_rows) >= 4,
    },
}
audit["strict_pass"] = bool(all(audit["checks"].values()))
out = oof_root / "likpf_residual_foldclean_spatial_catboost_v219"
scan.to_csv(out / "architecture_blend_scan.csv", index=False)
pd.DataFrame(nested_rows).to_csv(out / "architecture_nested_folds.csv", index=False)
(out / "architecture_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\nV219 ARCHITECTURE BLEND TOP 20")
print(scan.head(20).to_string(index=False))
print("\nV219 ARCHITECTURE NESTED")
print(pd.DataFrame(nested_rows).to_string(index=False))
print("\nV219 ARCHITECTURE VERDICT")
print(json.dumps(audit, indent=2))
