"""V220: read-only fusion of the V217/V219 spatial tree engine with mainline OOF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / "spatial_tree_mainline_fusion_v220"
OUT.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

TREE_SPECS = {
    "base_v215": (OOF / "likpf_residual_gbdt_full5_v215/oof.parquet", "candidate"),
    "lgb_v217": (OOF / "likpf_residual_foldclean_spatial_v217/oof.parquet", "candidate"),
    "cat_v219": (OOF / "likpf_residual_foldclean_spatial_catboost_v219/oof.parquet", "candidate"),
}
PARENT_SPECS = {
    "v66": (OOF / "frontier_6p377_full_fivefold_lineage_v66/incumbent_6p377_full_fivefold_oof.parquet", "incumbent_6p377_replay"),
    "v144": (OOF / "final_consensus_sequence_coefficient_full5_v144/crossfit_oof.parquet", "nested_sequence_coefficient"),
    "v168": (OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet", "metadata_analog_w0p05"),
    "v169": (OOF / "historical_analog_ranked_consensus_v169/fullrow_oof.parquet", "nested_consensus"),
}


def _read(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _normalise(frame, prediction, name):
    local = frame.copy()
    if "id" not in local:
        well_col = next((c for c in ("well", "well_id") if c in local), None)
        row_col = next((c for c in ("row_idx", "row_index") if c in local), None)
        if well_col is None or row_col is None:
            raise RuntimeError(f"{name}: cannot construct id")
        local["id"] = local[well_col].astype(str) + "_" + local[row_col].astype(int).astype(str)
    if prediction not in local:
        raise RuntimeError(f"{name}: missing prediction {prediction}")
    columns = ["id", prediction]
    optional = [c for c in ("well", "well_id", "fold", "outer_fold", "md_since", "distance", "tvt_true", "truth", "target_tvt") if c in local]
    local = local[columns + optional].copy()
    local = local.rename(columns={prediction: name})
    if local["id"].duplicated().any():
        raise RuntimeError(f"{name}: duplicate ids")
    return local


tree_frames = {
    name: _normalise(_read(path), column, name)
    for name, (path, column) in TREE_SPECS.items()
}
base = tree_frames["lgb_v217"]
well_col = next(c for c in ("well", "well_id") if c in base)
fold_col = next(c for c in ("fold", "outer_fold") if c in base)
distance_col = next(c for c in ("md_since", "distance") if c in base)
truth_col = next(c for c in ("tvt_true", "truth", "target_tvt") if c in base)
frame = base[["id", well_col, fold_col, distance_col, truth_col, "lgb_v217"]].rename(
    columns={well_col: "well", fold_col: "fold", distance_col: "md_since", truth_col: "tvt_true"}
)
for name in ("base_v215", "cat_v219"):
    frame = frame.merge(tree_frames[name][["id", name]], on="id", how="left", validate="one_to_one")
if len(frame) != 3_783_989 or frame[["base_v215", "lgb_v217", "cat_v219"]].isna().any().any():
    raise RuntimeError("Incomplete spatial-tree alignment")

base_tree = frame["base_v215"].to_numpy(float)
lgb = frame["lgb_v217"].to_numpy(float)
cat = frame["cat_v219"].to_numpy(float)
tree_predictions = {
    "lgb": lgb,
    "cat": cat,
    "equal_50_50": base_tree + 0.50 * (lgb - base_tree) + 0.50 * (cat - base_tree),
    "stable_50_55": base_tree + 0.50 * (lgb - base_tree) + 0.55 * (cat - base_tree),
}

truth = frame["tvt_true"].to_numpy(float)
fold = frame["fold"].to_numpy(int)
far = frame["md_since"].to_numpy(float) > 2000.0
all_mask = np.ones(len(frame), dtype=bool)


def _rmse(mask, prediction):
    error = prediction[mask] - truth[mask]
    return float(np.sqrt(np.mean(error * error)))


def _p95(mask, prediction):
    local = pd.DataFrame({
        "well": frame.loc[mask, "well"].to_numpy(),
        "se": (prediction[mask] - truth[mask]) ** 2,
    })
    return float(np.sqrt(local.groupby("well")["se"].mean()).quantile(0.95))


results = []
nested_results = []
best_outputs = []
for parent_name, (path, column) in PARENT_SPECS.items():
    if not path.is_file():
        print("V220 skip", parent_name, path, flush=True)
        continue
    parent_local = _normalise(_read(path), column, parent_name)[["id", parent_name]]
    aligned = frame[["id"]].merge(parent_local, on="id", how="left", validate="one_to_one")
    if aligned[parent_name].isna().any():
        raise RuntimeError(f"{parent_name}: incomplete alignment")
    parent = aligned[parent_name].to_numpy(float)
    parent_rmse = _rmse(all_mask, parent)
    parent_far = _rmse(far, parent)
    parent_p95 = _p95(all_mask, parent)

    for tree_name, tree in tree_predictions.items():
        tree_error = tree - truth
        parent_error = parent - truth
        raw_corr = float(np.corrcoef(parent_error, tree_error)[0, 1])
        move = tree - parent
        scan = []
        for alpha in np.round(np.arange(0.0, 0.5001, 0.025), 6):
            prediction = parent + alpha * move
            row = {
                "parent": parent_name,
                "tree": tree_name,
                "alpha": float(alpha),
                "rmse": _rmse(all_mask, prediction),
                "far_gt2000_rmse": _rmse(far, prediction),
                "p95_well_rmse": _p95(all_mask, prediction),
                "raw_error_corr": raw_corr,
            }
            row["gain"] = parent_rmse - row["rmse"]
            row["far_gain"] = parent_far - row["far_gt2000_rmse"]
            row["p95_gain"] = parent_p95 - row["p95_well_rmse"]
            fold_gains = []
            for held in range(1, 6):
                mask = fold == held
                gain = _rmse(mask, parent) - _rmse(mask, prediction)
                row[f"fold{held}_gain"] = gain
                fold_gains.append(gain)
            row["positive_folds"] = int(sum(value > 0 for value in fold_gains))
            row["minimum_fold_gain"] = float(min(fold_gains))
            scan.append(row)
        best = min(scan, key=lambda row: (row["rmse"], row["alpha"]))
        stable = [row for row in scan if row["positive_folds"] >= 4 and row["minimum_fold_gain"] >= -0.03]
        best_stable = min(stable, key=lambda row: (row["rmse"], row["alpha"])) if stable else None
        results.extend(scan)

        nested_prediction = parent.copy()
        nested_folds = []
        for held in range(1, 6):
            train_mask = fold != held
            valid_mask = fold == held
            choices = []
            for alpha in np.round(np.arange(0.0, 0.5001, 0.025), 6):
                prediction = parent + alpha * move
                choices.append((_rmse(train_mask, prediction), float(alpha)))
            inner_rmse, selected_alpha = min(choices)
            nested_prediction[valid_mask] = parent[valid_mask] + selected_alpha * move[valid_mask]
            base_fold = _rmse(valid_mask, parent)
            candidate_fold = _rmse(valid_mask, nested_prediction)
            nested_folds.append({
                "fold": held,
                "selected_alpha": selected_alpha,
                "inner_rmse": inner_rmse,
                "parent_rmse": base_fold,
                "rmse": candidate_fold,
                "gain": base_fold - candidate_fold,
            })
        nested_rmse = _rmse(all_mask, nested_prediction)
        nested = {
            "parent": parent_name,
            "tree": tree_name,
            "parent_rmse": parent_rmse,
            "rmse": nested_rmse,
            "gain": parent_rmse - nested_rmse,
            "positive_folds": sum(row["gain"] > 0 for row in nested_folds),
            "minimum_fold_gain": min(row["gain"] for row in nested_folds),
            "folds": nested_folds,
        }
        nested_results.append(nested)
        best_outputs.append({
            "parent": parent_name,
            "tree": tree_name,
            "parent_rmse": parent_rmse,
            "tree_rmse": _rmse(all_mask, tree),
            "raw_error_corr": raw_corr,
            "best_fixed": best,
            "best_stable_fixed": best_stable,
            "nested": nested,
        })

scan_frame = pd.DataFrame(results).sort_values("rmse", kind="stable").reset_index(drop=True)
nested_frame = pd.DataFrame([{k: v for k, v in row.items() if k != "folds"} for row in nested_results]).sort_values("rmse")
qualified = [
    row for row in best_outputs
    if row["best_stable_fixed"] is not None
    and row["best_stable_fixed"]["gain"] >= 0.05
    and row["nested"]["gain"] >= 0.03
    and row["nested"]["positive_folds"] >= 4
    and row["nested"]["minimum_fold_gain"] >= -0.03
]
selected = min(qualified, key=lambda row: row["nested"]["rmse"]) if qualified else None
verdict = {
    "experiment": "spatial_tree_mainline_fusion_v220",
    "stage": "read_only_complete773_spatial_tree_to_mainline_nested_fusion",
    "training_count": 0,
    "tree_recipes": list(tree_predictions),
    "parents_available": sorted({row["parent"] for row in best_outputs}),
    "top_fixed": scan_frame.head(20).to_dict("records"),
    "nested_summary": nested_frame.to_dict("records"),
    "selected": selected,
    "strict_pass": selected is not None,
    "next": "PASS -> build one deployment candidate from the selected fixed recipe" if selected is not None else "FAIL -> independent spatial trees do not improve the deployable mainline",
}

scan_frame.to_csv(OUT / "fixed_scan.csv", index=False)
nested_frame.to_csv(OUT / "nested_summary.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")

print("V220 FIXED TOP 20")
print(scan_frame.head(20).to_string(index=False))
print("\nV220 NESTED SUMMARY")
print(nested_frame.to_string(index=False))
print("\nV220 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
