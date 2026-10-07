"""Nested fusion audit for strict-spatial K5 and the established strong chain.

No model training. Every blend weight is selected on four folds and scored on
the held-out fold. The primary test is strict_spatial + strong_bias; CP12,
SP45, and fleongg are included as controls and optional simplex components.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


OOF_DIR = Path("/root/autodl-tmp/oof_exports")
SPATIAL_PATH = (
    OOF_DIR
    / "multicut_k5_strict_outer_spatial_v1"
    / "strict_outer_spatial_oof.pkl"
)
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
RUN_DIR = OOF_DIR / "strict_spatial_strong_blend_audit_v1"

PAIR_STEP = 0.025
MULTI_STEP = 0.05
RUN_DIR.mkdir(parents=True, exist_ok=True)
for path in (SPATIAL_PATH, POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)


def _rmse(y_true, prediction):
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - y_true) ** 2)))


def _error_cosine(y_true, prediction_a, prediction_b):
    y_true = np.asarray(y_true, dtype=np.float64)
    error_a = np.asarray(prediction_a, dtype=np.float64) - y_true
    error_b = np.asarray(prediction_b, dtype=np.float64) - y_true
    denominator = np.sqrt(np.dot(error_a, error_a) * np.dot(error_b, error_b))
    return float(np.dot(error_a, error_b) / denominator) if denominator else np.nan


def _composition_weights(n_components, step):
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1 exactly")
    rows = []

    def visit(prefix, remaining, slots):
        if slots == 1:
            rows.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value, slots - 1)

    visit([], units, n_components)
    return np.asarray(rows, dtype=np.float64) / units


def _fit_grid_weights(y_true, prediction_matrix, weight_grid):
    errors = np.asarray(prediction_matrix, dtype=np.float64) - np.asarray(
        y_true, dtype=np.float64
    )[:, None]
    gram = errors.T @ errors / len(errors)
    mse = np.einsum("ij,jk,ik->i", weight_grid, gram, weight_grid)
    index = int(np.argmin(mse))
    return weight_grid[index], float(mse[index])


def _first_present(columns, aliases):
    for alias in aliases:
        if alias in columns:
            return alias
    raise RuntimeError(f"Missing aliases: {aliases}")


spatial = pd.read_pickle(SPATIAL_PATH)
spatial["id"] = spatial["id"].astype(str)
spatial["well"] = spatial["well"].astype(str)
required_spatial = {"id", "well", "fold", "tvt_true", "pred_spatial"}
missing_spatial = required_spatial - set(spatial.columns)
if missing_spatial:
    raise RuntimeError(f"Spatial OOF missing: {sorted(missing_spatial)}")

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
sp45_source = _first_present(header, ["sp45_like", "sp45"])
fle_source = _first_present(header, ["fleongg", "fle"])
required_pool = ["id", "strong_bias", "cp12", sp45_source, fle_source]
missing_pool = [column for column in required_pool if column not in header]
if missing_pool:
    raise RuntimeError(f"Pool missing: {missing_pool}")
pool = pd.read_csv(
    POOL_PATH,
    usecols=required_pool,
    dtype={
        "id": str,
        "strong_bias": np.float32,
        "cp12": np.float32,
        sp45_source: np.float32,
        fle_source: np.float32,
    },
).rename(columns={sp45_source: "sp45", fle_source: "fleongg"})
pool["id"] = pool["id"].astype(str)

data = spatial[["id", "well", "fold", "tvt_true", "pred_spatial"]].merge(
    pool, on="id", how="inner", validate="one_to_one"
)
if len(data) != len(spatial):
    raise RuntimeError(f"Merge mismatch: {len(spatial)} -> {len(data)}")
data = data.rename(columns={"pred_spatial": "strict_spatial"})

model_columns = {
    "strict_spatial": "strict_spatial",
    "strong_bias": "strong_bias",
    "cp12": "cp12",
    "sp45": "sp45",
    "fleongg": "fleongg",
}
truth = data["tvt_true"].to_numpy(float)
print(
    {
        "experiment": "strict_spatial_strong_blend_audit_v1",
        "rows": len(data),
        "wells": int(data["well"].nunique()),
        "pair_step": PAIR_STEP,
        "multi_step": MULTI_STEP,
    }
)


# Standalone and residual geometry.
overall = pd.DataFrame(
    [
        {"model": name, "rmse": _rmse(truth, data[column])}
        for name, column in model_columns.items()
    ]
).sort_values("rmse").reset_index(drop=True)
names = list(model_columns)
cosines = pd.DataFrame(index=names, columns=names, dtype=float)
for name_a in names:
    for name_b in names:
        cosines.loc[name_a, name_b] = _error_cosine(
            truth, data[model_columns[name_a]], data[model_columns[name_b]]
        )


# Primary pair plus broader controls. Four-way excludes CP12 because CP12 is a
# downstream correction, while strong_bias/SP45/fleongg are base comparators.
blend_specs = {
    "spatial_strong": (["strict_spatial", "strong_bias"], PAIR_STEP),
    "spatial_cp12": (["strict_spatial", "cp12"], PAIR_STEP),
    "spatial_strong_fle": (
        ["strict_spatial", "strong_bias", "fleongg"],
        MULTI_STEP,
    ),
    "spatial_strong_sp45_fle": (
        ["strict_spatial", "strong_bias", "sp45", "fleongg"],
        MULTI_STEP,
    ),
}

fold_values = data["fold"].to_numpy(int)
nested_predictions = {}
blend_fold_rows = []
for blend_name, (component_names, step) in blend_specs.items():
    columns = [model_columns[name] for name in component_names]
    weight_grid = _composition_weights(len(columns), step)
    nested = np.full(len(data), np.nan, dtype=np.float64)
    for fold in sorted(data["fold"].unique()):
        train_mask = fold_values != fold
        valid_mask = ~train_mask
        train_truth = data.loc[train_mask, "tvt_true"].to_numpy(float)
        train_matrix = data.loc[train_mask, columns].to_numpy(float)
        weights, train_mse = _fit_grid_weights(
            train_truth, train_matrix, weight_grid
        )
        valid_truth = data.loc[valid_mask, "tvt_true"].to_numpy(float)
        valid_matrix = data.loc[valid_mask, columns].to_numpy(float)
        valid_prediction = valid_matrix @ weights
        nested[valid_mask] = valid_prediction
        component_scores = [
            _rmse(valid_truth, valid_matrix[:, index])
            for index in range(len(columns))
        ]
        blend_score = _rmse(valid_truth, valid_prediction)
        row = {
            "blend": blend_name,
            "fold": int(fold),
            "components": "+".join(component_names),
            "train_rmse": float(np.sqrt(train_mse)),
            "blend_rmse": blend_score,
            "best_component_rmse": min(component_scores),
            "gain_vs_best_component": min(component_scores) - blend_score,
        }
        for index, name in enumerate(component_names):
            row[f"weight_{name}"] = float(weights[index])
            row[f"rmse_{name}"] = component_scores[index]
        blend_fold_rows.append(row)
    nested_predictions[blend_name] = nested

blend_folds = pd.DataFrame(blend_fold_rows)
blend_summary_rows = []
for blend_name, prediction in nested_predictions.items():
    component_names = blend_specs[blend_name][0]
    best_component_rmse = min(
        _rmse(truth, data[model_columns[name]]) for name in component_names
    )
    folds = blend_folds[blend_folds["blend"] == blend_name]
    blend_summary_rows.append(
        {
            "blend": blend_name,
            "components": "+".join(component_names),
            "rmse": _rmse(truth, prediction),
            "best_component_rmse": best_component_rmse,
            "gain_vs_best_component": best_component_rmse
            - _rmse(truth, prediction),
            "mean_fold_gain_vs_best": float(
                folds["gain_vs_best_component"].mean()
            ),
            "min_fold_gain_vs_best": float(
                folds["gain_vs_best_component"].min()
            ),
            "positive_folds_vs_best": int(
                (folds["gain_vs_best_component"] > 0).sum()
            ),
        }
    )
blend_summary = pd.DataFrame(blend_summary_rows).sort_values("rmse").reset_index(drop=True)


# Top40 is defined once by strong_bias SSE, then all candidates use that set.
bywell_rows = []
for wid, group in data.groupby("well", sort=False):
    row = {"well": wid, "rows": len(group)}
    group_truth = group["tvt_true"].to_numpy(float)
    for name, column in model_columns.items():
        error = group[column].to_numpy(float) - group_truth
        row[f"{name}_sse"] = float(np.dot(error, error))
    bywell_rows.append(row)
bywell = pd.DataFrame(bywell_rows)
top40 = bywell.nlargest(40, "strong_bias_sse")
top40_rows = []
for name in model_columns:
    top40_rows.append(
        {
            "model": name,
            "rmse": float(
                np.sqrt(top40[f"{name}_sse"].sum() / top40["rows"].sum())
            ),
        }
    )
for blend_name, prediction in nested_predictions.items():
    blend_sse = {}
    for wid, group_index in data.groupby("well", sort=False).groups.items():
        index = np.asarray(list(group_index), dtype=int)
        error = prediction[index] - truth[index]
        blend_sse[str(wid)] = float(np.dot(error, error))
    top40_rows.append(
        {
            "model": f"nested_{blend_name}",
            "rmse": float(
                np.sqrt(
                    sum(blend_sse[str(wid)] for wid in top40["well"])
                    / top40["rows"].sum()
                )
            ),
        }
    )
top40_scores = pd.DataFrame(top40_rows).sort_values("rmse").reset_index(drop=True)

primary = blend_summary[blend_summary["blend"] == "spatial_strong"].iloc[0]
primary_folds = blend_folds[blend_folds["blend"] == "spatial_strong"]
primary_pass = bool(
    primary["gain_vs_best_component"] >= 0.05
    and int((primary_folds["gain_vs_best_component"] > 0).sum()) >= 4
)
best_multi = blend_summary.iloc[0]
multi_folds = blend_folds[blend_folds["blend"] == best_multi["blend"]]
multi_pass = bool(
    best_multi["gain_vs_best_component"] >= 0.05
    and int((multi_folds["gain_vs_best_component"] > 0).sum()) >= 4
)
strict_pass = bool(primary_pass or multi_pass)
result = {
    "strict_spatial_rmse": _rmse(truth, data["strict_spatial"]),
    "strong_bias_rmse": _rmse(truth, data["strong_bias"]),
    "spatial_strong_error_cosine": float(
        cosines.loc["strict_spatial", "strong_bias"]
    ),
    "nested_spatial_strong_rmse": float(primary["rmse"]),
    "nested_spatial_strong_gain": float(primary["gain_vs_best_component"]),
    "nested_spatial_strong_min_fold_gain": float(
        primary["min_fold_gain_vs_best"]
    ),
    "nested_spatial_strong_positive_folds": int(
        primary["positive_folds_vs_best"]
    ),
    "best_nested_blend": str(best_multi["blend"]),
    "best_nested_rmse": float(best_multi["rmse"]),
    "best_nested_gain": float(best_multi["gain_vs_best_component"]),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train strict-spatial CatBoost/XGBoost, then deploy blend"
        if strict_pass
        else "strong-chain fusion is not stable; keep strict-spatial as research base only"
    ),
}

print("\nOVERALL")
display(overall)
print("\nERROR COSINES")
display(cosines)
print("\nNESTED FOLDS")
display(blend_folds)
print("\nNESTED SUMMARY")
display(blend_summary)
print("\nSTRONG-BIAS TOP40")
display(top40_scores)
print("\nRESULT")
print(result)
print("\nVERDICT:", result["next"])

overall.to_csv(RUN_DIR / "overall.csv", index=False)
cosines.to_csv(RUN_DIR / "error_cosines.csv")
blend_folds.to_csv(RUN_DIR / "nested_folds.csv", index=False)
blend_summary.to_csv(RUN_DIR / "nested_summary.csv", index=False)
bywell.to_csv(RUN_DIR / "bywell.csv", index=False)
top40_scores.to_csv(RUN_DIR / "top40_scores.csv", index=False)
(RUN_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", RUN_DIR)
