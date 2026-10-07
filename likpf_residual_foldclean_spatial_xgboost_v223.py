"""V223: final tree-architecture gateway on the frozen V217 205 inputs.

Only LightGBM is replaced by XGBoost.  The script then forms a non-negative
three-tree residual ensemble and tests it directly against V168 with nested
cross-fold blend doses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "likpf_residual_foldclean_spatial_v217.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
start = source.find("def _new_model(fold):")
end = source.find("def _smooth(frame, prediction):")
if start < 0 or end <= start:
    raise RuntimeError("V223 could not locate the audited V217 model block")

xgb_block = '''def _new_model(fold):
    from xgboost import XGBRegressor
    use_cpu = os.environ.get("V223_CPU", "0") == "1"
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000,
        learning_rate=0.025,
        max_depth=8,
        min_child_weight=40.0,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=10.0,
        max_bin=256,
        tree_method="hist",
        device="cpu" if use_cpu else "cuda",
        eval_metric="rmse",
        early_stopping_rounds=200,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=1,
    )


def _fit(train, valid, fold):
    model = _new_model(fold)
    kwargs = dict(
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["residual_target"].to_numpy(float))],
        verbose=100,
    )
    try:
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    except Exception as exc:
        if os.environ.get("V223_CPU", "0") == "1":
            raise
        print("V223 GPU fallback", repr(exc), flush=True)
        os.environ["V223_CPU"] = "1"
        model = _new_model(fold)
        model.fit(_matrix(train), train["residual_target"].to_numpy(float), **kwargs)
    return model


'''
source = source[:start] + xgb_block + source[end:]
old_predict = 'residual = model.predict(_matrix(valid), num_iteration=model.best_iteration_)'
old_iteration = 'int(model.best_iteration_ or model.n_estimators)'
if source.count(old_predict) != 1 or source.count(old_iteration) != 1:
    raise RuntimeError("V223 could not locate V217 prediction/report calls")
source = source.replace(old_predict, 'residual = model.predict(_matrix(valid))', 1)
source = source.replace(old_iteration, 'int(model.best_iteration + 1)', 1)
# Keep V217's seed: architecture is the only intended change.
source = source.replace('SEED = 217', 'SEED = 217', 1)
source = source.replace(
    'likpf_residual_foldclean_spatial_v217',
    'likpf_residual_foldclean_spatial_xgboost_v223',
)
source = source.replace('V217', 'V223').replace('v217', 'v223')
exec(compile(source, str(SOURCE), "exec"), globals())

OOF = ROOT / "oof_exports"
paths = {
    "base": OOF / "likpf_residual_gbdt_full5_v215/oof.parquet",
    "lgb": OOF / "likpf_residual_foldclean_spatial_v217/oof.parquet",
    "cat": OOF / "likpf_residual_foldclean_spatial_catboost_v219/oof.parquet",
    "xgb": OOF / "likpf_residual_foldclean_spatial_xgboost_v223/oof.parquet",
    "v168": OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet",
}
for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)

base = pd.read_parquet(paths["base"])[["id", "candidate"]].rename(columns={"candidate": "base"})
lgb = pd.read_parquet(paths["lgb"])[["id", "well", "fold", "md_since", "tvt_true", "candidate"]].rename(columns={"candidate": "lgb"})
cat = pd.read_parquet(paths["cat"])[["id", "candidate"]].rename(columns={"candidate": "cat"})
xgb = pd.read_parquet(paths["xgb"])[["id", "candidate"]].rename(columns={"candidate": "xgb"})
parent_raw = pd.read_parquet(paths["v168"])
if "id" not in parent_raw:
    well_col = next(c for c in ("well", "well_id") if c in parent_raw)
    row_col = next(c for c in ("row_idx", "row_index") if c in parent_raw)
    parent_raw["id"] = parent_raw[well_col].astype(str) + "_" + parent_raw[row_col].astype(int).astype(str)
parent = parent_raw[["id", "metadata_analog_w0p05"]].rename(columns={"metadata_analog_w0p05": "v168"})

frame = lgb.merge(base, on="id", validate="one_to_one").merge(cat, on="id", validate="one_to_one").merge(xgb, on="id", validate="one_to_one").merge(parent, on="id", validate="one_to_one")
if len(frame) != 3_783_989 or frame.isna().any().any():
    raise RuntimeError("Incomplete V223 architecture alignment")

truth = frame["tvt_true"].to_numpy(float)
base_pred = frame["base"].to_numpy(float)
fold = frame["fold"].to_numpy(int)
moves = np.column_stack([
    frame["lgb"].to_numpy(float) - base_pred,
    frame["cat"].to_numpy(float) - base_pred,
    frame["xgb"].to_numpy(float) - base_pred,
])


def _rmse(mask, prediction):
    error = prediction[mask] - truth[mask]
    return float(np.sqrt(np.mean(error * error)))


def _weights(mask):
    weights, _ = nnls(moves[mask], truth[mask] - base_pred[mask])
    if weights.sum() > 1.25:
        weights *= 1.25 / weights.sum()
    return weights


all_mask = np.ones(len(frame), dtype=bool)
global_weights = _weights(all_mask)
tree3 = base_pred + moves @ global_weights
standalone = {
    "v215": _rmse(all_mask, base_pred),
    "lgb": _rmse(all_mask, frame["lgb"].to_numpy(float)),
    "cat": _rmse(all_mask, frame["cat"].to_numpy(float)),
    "xgb": _rmse(all_mask, frame["xgb"].to_numpy(float)),
    "tree3_nnls": _rmse(all_mask, tree3),
}

# Nested architecture weights, followed by a nested V168 blend dose.
nested_tree = base_pred.copy()
nested_rows = []
for held in range(1, 6):
    train_mask = fold != held
    valid_mask = fold == held
    weights = _weights(train_mask)
    nested_tree[valid_mask] = base_pred[valid_mask] + moves[valid_mask] @ weights
    nested_rows.append({
        "fold": held,
        "lgb_weight": float(weights[0]),
        "cat_weight": float(weights[1]),
        "xgb_weight": float(weights[2]),
        "tree_rmse": _rmse(valid_mask, nested_tree),
    })

v168 = frame["v168"].to_numpy(float)
nested_final = v168.copy()
for row in nested_rows:
    held = row["fold"]
    train_mask = fold != held
    valid_mask = fold == held
    move = nested_tree - v168
    error = v168 - truth
    denominator = float(np.dot(move[train_mask], move[train_mask]))
    alpha = 0.0 if denominator <= 0 else float(np.clip(-np.dot(error[train_mask], move[train_mask]) / denominator, 0.0, 0.5))
    nested_final[valid_mask] = v168[valid_mask] + alpha * move[valid_mask]
    row["selected_v168_alpha"] = alpha
    row["v168_rmse"] = _rmse(valid_mask, v168)
    row["final_rmse"] = _rmse(valid_mask, nested_final)
    row["final_gain"] = row["v168_rmse"] - row["final_rmse"]

v168_rmse = _rmse(all_mask, v168)
final_rmse = _rmse(all_mask, nested_final)
audit = {
    "experiment": "likpf_residual_foldclean_spatial_xgboost_v223_architecture_gateway",
    "changed_variable": "XGBoost architecture only",
    "standalone_rmse": standalone,
    "global_nnls_weights": {
        "lgb": float(global_weights[0]),
        "cat": float(global_weights[1]),
        "xgb": float(global_weights[2]),
    },
    "pairwise_error_correlation": pd.DataFrame({
        name: frame[name].to_numpy(float) - truth for name in ("lgb", "cat", "xgb")
    }).corr().to_dict(),
    "nested": {
        "tree_rmse": _rmse(all_mask, nested_tree),
        "v168_rmse": v168_rmse,
        "final_rmse": final_rmse,
        "gain_vs_v168": v168_rmse - final_rmse,
        "folds": nested_rows,
    },
    "checks": {
        "xgb_rmse_le_9p8": standalone["xgb"] <= 9.8,
        "tree3_gain_vs_best_tree_ge_0p05": min(standalone["lgb"], standalone["cat"], standalone["xgb"]) - standalone["tree3_nnls"] >= 0.05,
        "nested_v168_gain_positive": v168_rmse - final_rmse > 0.0,
        "nested_v168_positive_folds_ge_4": sum(row["final_gain"] > 0 for row in nested_rows) >= 4,
    },
}
audit["strict_pass"] = bool(all(audit["checks"].values()))
out = OOF / "likpf_residual_foldclean_spatial_xgboost_v223"
pd.DataFrame(nested_rows).to_csv(out / "nested_architecture_fusion.csv", index=False)
(out / "architecture_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\nV223 NESTED ARCHITECTURE/FUSION")
print(pd.DataFrame(nested_rows).to_string(index=False))
print("\nV223 FINAL ARCHITECTURE VERDICT")
print(json.dumps(audit, indent=2))
