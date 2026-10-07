"""Compare matched-K5 OOF against SP45/fleongg and test honest fusion.

This is a read-only OOF audit. It does not train a model and does not use
CP12/Gold as a gate. Blend weights are selected on four folds and evaluated
on the held-out fifth fold, so the primary blend result is not a full-OOF
weight fit.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


OOF_DIR = Path("/root/autodl-tmp/oof_exports")
K5_PATH = (
    OOF_DIR
    / "multicut_fleongg_matched_k5_three_model_v1"
    / "three_model_oof.pkl"
)
POOL_PATH = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
RUN_DIR = OOF_DIR / "multicut_k5_external_blend_audit_v1"

PAIR_STEP = 0.025
SIMPLEX_STEP = 0.025
PRIMARY_K5 = "k5_equal3"

RUN_DIR.mkdir(parents=True, exist_ok=True)
for path in (K5_PATH, POOL_PATH):
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


def _first_present(columns, aliases, required=True):
    for alias in aliases:
        if alias in columns:
            return alias
    if required:
        raise RuntimeError(f"None of these columns exist: {aliases}")
    return None


def _gram(y_true, predictions):
    y_true = np.asarray(y_true, dtype=np.float64)
    matrix = np.column_stack(
        [np.asarray(prediction, dtype=np.float64) - y_true for prediction in predictions]
    )
    return matrix.T @ matrix / len(matrix)


def _best_weight_from_gram(gram, weight_matrix):
    mse = np.einsum("ij,jk,ik->i", weight_matrix, gram, weight_matrix)
    index = int(np.argmin(mse))
    return weight_matrix[index], float(mse[index])


def _pair_weight_matrix(step):
    weights_a = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float64)
    return np.column_stack([weights_a, 1.0 - weights_a])


def _simplex_weight_matrix(step):
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("SIMPLEX_STEP must divide 1 exactly")
    rows = []
    for a in range(units + 1):
        for b in range(units - a + 1):
            c = units - a - b
            rows.append((a / units, b / units, c / units))
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Load a common row set.
# ---------------------------------------------------------------------------
k5 = pd.read_pickle(K5_PATH)
k5["id"] = k5["id"].astype(str)
k5["well"] = k5["well"].astype(str)
required_k5 = {
    "id",
    "well",
    "fold",
    "tvt_true",
    "pred_lgb",
    "pred_cat",
    "pred_xgb",
    "pred_equal_3",
}
missing_k5 = required_k5 - set(k5.columns)
if missing_k5:
    raise RuntimeError(f"K5 OOF missing columns: {sorted(missing_k5)}")
if k5["id"].duplicated().any():
    raise RuntimeError("K5 OOF contains duplicate ids")

pool_columns = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
sp45_source = _first_present(pool_columns, ["sp45_like", "sp45"])
fle_source = _first_present(pool_columns, ["fleongg", "fle"])
pool_truth_source = _first_present(
    pool_columns, ["tvt_true", "TVT", "target"], required=False
)
context_aliases = {
    "strong": ["strong"],
    "strong_bias": ["strong_bias"],
    "cp12": ["cp12"],
}
context_sources = {
    name: _first_present(pool_columns, aliases, required=False)
    for name, aliases in context_aliases.items()
}
context_sources = {name: source for name, source in context_sources.items() if source}

usecols = ["id", sp45_source, fle_source]
if pool_truth_source:
    usecols.append(pool_truth_source)
usecols.extend(context_sources.values())
usecols = list(dict.fromkeys(usecols))
dtypes = {"id": str}
for column in usecols:
    if column != "id":
        dtypes[column] = np.float32

print(
    {
        "experiment": "multicut_k5_external_blend_audit_v1",
        "k5": str(K5_PATH),
        "pool": str(POOL_PATH),
        "sp45_source": sp45_source,
        "fleongg_source": fle_source,
        "context": context_sources,
    }
)
pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype=dtypes)
pool["id"] = pool["id"].astype(str)
if pool["id"].duplicated().any():
    raise RuntimeError("Candidate pool contains duplicate ids")

rename = {sp45_source: "pred_sp45", fle_source: "pred_fleongg"}
if pool_truth_source:
    rename[pool_truth_source] = "pool_truth"
rename.update({source: f"pred_{name}" for name, source in context_sources.items()})
pool = pool.rename(columns=rename)

keep_k5 = [
    "id",
    "well",
    "fold",
    "tvt_true",
    "pred_lgb",
    "pred_cat",
    "pred_xgb",
    "pred_equal_3",
]
data = k5[keep_k5].merge(pool, on="id", how="inner", validate="one_to_one")
if len(data) != len(k5):
    raise RuntimeError(f"Common-row mismatch: K5={len(k5)} merged={len(data)}")
if pool_truth_source:
    truth_difference = float(
        np.nanmax(np.abs(data["tvt_true"].to_numpy(float) - data["pool_truth"].to_numpy(float)))
    )
    if truth_difference > 1e-3:
        raise RuntimeError(f"Truth mismatch against pool: max={truth_difference}")
    data = data.drop(columns="pool_truth")

data = data.rename(
    columns={
        "pred_lgb": "pred_k5_lgb",
        "pred_cat": "pred_k5_cat",
        "pred_xgb": "pred_k5_xgb",
        "pred_equal_3": "pred_k5_equal3",
    }
)

model_columns = {
    "k5_lgb": "pred_k5_lgb",
    "k5_cat": "pred_k5_cat",
    "k5_xgb": "pred_k5_xgb",
    "k5_equal3": "pred_k5_equal3",
    "sp45": "pred_sp45",
    "fleongg": "pred_fleongg",
}
for name in context_sources:
    model_columns[name] = f"pred_{name}"

for name, column in model_columns.items():
    missing = int(data[column].isna().sum())
    if missing:
        raise RuntimeError(f"{name} has {missing} missing predictions")

print("common rows/wells:", data.shape, data["well"].nunique())


# ---------------------------------------------------------------------------
# Standalone scores, fold scores, disagreements, and residual geometry.
# ---------------------------------------------------------------------------
overall_rows = []
fold_rows = []
for name, column in model_columns.items():
    overall_rows.append(
        {"model": name, "rmse": _rmse(data["tvt_true"], data[column])}
    )
    for fold in sorted(data["fold"].unique()):
        part = data[data["fold"] == fold]
        fold_rows.append(
            {
                "model": name,
                "fold": int(fold),
                "rows": len(part),
                "rmse": _rmse(part["tvt_true"], part[column]),
            }
        )
overall = pd.DataFrame(overall_rows).sort_values("rmse").reset_index(drop=True)
fold_scores = pd.DataFrame(fold_rows)

geometry_names = list(model_columns)
cosine = pd.DataFrame(index=geometry_names, columns=geometry_names, dtype=float)
disagreement_rows = []
for name_a in geometry_names:
    for name_b in geometry_names:
        cosine.loc[name_a, name_b] = _error_cosine(
            data["tvt_true"],
            data[model_columns[name_a]],
            data[model_columns[name_b]],
        )
    for name_b in geometry_names[geometry_names.index(name_a) + 1 :]:
        difference = (
            data[model_columns[name_a]].to_numpy(float)
            - data[model_columns[name_b]].to_numpy(float)
        )
        disagreement_rows.append(
            {
                "pair": f"{name_a}__{name_b}",
                "mean_abs_difference": float(np.mean(np.abs(difference))),
                "p95_abs_difference": float(np.quantile(np.abs(difference), 0.95)),
                "rms_difference": float(np.sqrt(np.mean(difference**2))),
            }
        )
disagreements = pd.DataFrame(disagreement_rows)


# ---------------------------------------------------------------------------
# Per-well complementarity and a K5-defined top40 audit.
# ---------------------------------------------------------------------------
bywell_rows = []
for wid, part in data.groupby("well", sort=False):
    row = {"well": wid, "rows": len(part), "fold": int(part["fold"].iloc[0])}
    truth = part["tvt_true"].to_numpy(float)
    for name, column in model_columns.items():
        error = part[column].to_numpy(float) - truth
        row[f"{name}_sse"] = float(np.dot(error, error))
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
    bywell_rows.append(row)
bywell = pd.DataFrame(bywell_rows)
winner_columns = [f"{name}_rmse" for name in (PRIMARY_K5, "sp45", "fleongg")]
bywell["winner"] = (
    bywell[winner_columns].idxmin(axis=1).str.replace("_rmse", "", regex=False)
)
winner_counts = bywell["winner"].value_counts().rename_axis("winner").reset_index(name="wells")

top40 = bywell.nlargest(40, f"{PRIMARY_K5}_sse")
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
top40_scores = pd.DataFrame(top40_rows).sort_values("rmse").reset_index(drop=True)

truth = data["tvt_true"].to_numpy(float)
oracle_columns = [model_columns[name] for name in (PRIMARY_K5, "sp45", "fleongg")]
oracle_matrix = data[oracle_columns].to_numpy(float)
oracle_index = np.argmin(np.abs(oracle_matrix - truth[:, None]), axis=1)
row_oracle = oracle_matrix[np.arange(len(data)), oracle_index]
row_oracle_rmse = _rmse(truth, row_oracle)


# ---------------------------------------------------------------------------
# Four-fold-select / one-fold-evaluate pairwise and three-way blends.
# ---------------------------------------------------------------------------
pair_weights = _pair_weight_matrix(PAIR_STEP)
simplex_weights = _simplex_weight_matrix(SIMPLEX_STEP)
primary_blends = {
    "k5_sp45": (PRIMARY_K5, "sp45", pair_weights),
    "k5_fleongg": (PRIMARY_K5, "fleongg", pair_weights),
    "sp45_fleongg": ("sp45", "fleongg", pair_weights),
}

nested_predictions = {}
nested_components = {}
nested_fold_rows = []
for blend_name, (name_a, name_b, weight_matrix) in primary_blends.items():
    prediction = np.full(len(data), np.nan, dtype=np.float64)
    columns = [model_columns[name_a], model_columns[name_b]]
    for fold in sorted(data["fold"].unique()):
        train_mask = data["fold"].to_numpy(int) != fold
        valid_mask = ~train_mask
        gram = _gram(
            data.loc[train_mask, "tvt_true"],
            [data.loc[train_mask, column] for column in columns],
        )
        weights, train_mse = _best_weight_from_gram(gram, weight_matrix)
        valid_prediction = (
            data.loc[valid_mask, columns].to_numpy(float) @ weights
        )
        prediction[valid_mask] = valid_prediction
        valid_truth = data.loc[valid_mask, "tvt_true"].to_numpy(float)
        score_a = _rmse(valid_truth, data.loc[valid_mask, columns[0]])
        score_b = _rmse(valid_truth, data.loc[valid_mask, columns[1]])
        score_blend = _rmse(valid_truth, valid_prediction)
        nested_fold_rows.append(
            {
                "blend": blend_name,
                "fold": int(fold),
                "component_a": name_a,
                "component_b": name_b,
                "weight_a": float(weights[0]),
                "weight_b": float(weights[1]),
                "train_rmse": float(np.sqrt(train_mse)),
                "valid_a_rmse": score_a,
                "valid_b_rmse": score_b,
                "valid_blend_rmse": score_blend,
                "gain_vs_a": score_a - score_blend,
                "gain_vs_best_component": min(score_a, score_b) - score_blend,
            }
        )
    nested_predictions[blend_name] = prediction
    nested_components[blend_name] = (name_a, name_b)

three_names = [PRIMARY_K5, "sp45", "fleongg"]
three_columns = [model_columns[name] for name in three_names]
three_prediction = np.full(len(data), np.nan, dtype=np.float64)
for fold in sorted(data["fold"].unique()):
    train_mask = data["fold"].to_numpy(int) != fold
    valid_mask = ~train_mask
    gram = _gram(
        data.loc[train_mask, "tvt_true"],
        [data.loc[train_mask, column] for column in three_columns],
    )
    weights, train_mse = _best_weight_from_gram(gram, simplex_weights)
    valid_prediction = data.loc[valid_mask, three_columns].to_numpy(float) @ weights
    three_prediction[valid_mask] = valid_prediction
    valid_truth = data.loc[valid_mask, "tvt_true"].to_numpy(float)
    component_scores = {
        name: _rmse(valid_truth, data.loc[valid_mask, column])
        for name, column in zip(three_names, three_columns)
    }
    blend_score = _rmse(valid_truth, valid_prediction)
    nested_fold_rows.append(
        {
            "blend": "k5_sp45_fleongg",
            "fold": int(fold),
            "component_a": PRIMARY_K5,
            "component_b": "sp45+fleongg",
            "weight_a": float(weights[0]),
            "weight_sp45": float(weights[1]),
            "weight_fleongg": float(weights[2]),
            "train_rmse": float(np.sqrt(train_mse)),
            "valid_a_rmse": component_scores[PRIMARY_K5],
            "valid_b_rmse": min(component_scores["sp45"], component_scores["fleongg"]),
            "valid_blend_rmse": blend_score,
            "gain_vs_a": component_scores[PRIMARY_K5] - blend_score,
            "gain_vs_best_component": min(component_scores.values()) - blend_score,
        }
    )
nested_predictions["k5_sp45_fleongg"] = three_prediction
nested_components["k5_sp45_fleongg"] = tuple(three_names)
nested_folds = pd.DataFrame(nested_fold_rows)

nested_summary_rows = []
for blend_name, prediction in nested_predictions.items():
    fold_part = nested_folds[nested_folds["blend"] == blend_name]
    component_names = nested_components[blend_name]
    best_component_rmse = min(
        _rmse(truth, data[model_columns[name]]) for name in component_names
    )
    nested_summary_rows.append(
        {
            "blend": blend_name,
            "components": "+".join(component_names),
            "rmse": _rmse(truth, prediction),
            "gain_vs_k5_equal3": _rmse(truth, data[model_columns[PRIMARY_K5]])
            - _rmse(truth, prediction),
            "best_component_rmse": best_component_rmse,
            "gain_vs_best_component": best_component_rmse
            - _rmse(truth, prediction),
            "mean_fold_gain_vs_best_component": float(
                fold_part["gain_vs_best_component"].mean()
            ),
            "min_fold_gain_vs_best_component": float(
                fold_part["gain_vs_best_component"].min()
            ),
            "positive_folds_vs_best_component": int(
                (fold_part["gain_vs_best_component"] > 0).sum()
            ),
        }
    )
nested_summary = pd.DataFrame(nested_summary_rows).sort_values("rmse").reset_index(drop=True)

three_summary = nested_summary[nested_summary["blend"] == "k5_sp45_fleongg"].iloc[0]
three_folds = nested_folds[nested_folds["blend"] == "k5_sp45_fleongg"]
fusion_pass = bool(
    three_summary["gain_vs_best_component"] >= 0.05
    and int((three_folds["gain_vs_best_component"] > 0).sum()) >= 4
)

k5_sp45_cosine = float(cosine.loc[PRIMARY_K5, "sp45"])
k5_fle_cosine = float(cosine.loc[PRIMARY_K5, "fleongg"])
result = {
    "k5_equal3_rmse": _rmse(truth, data[model_columns[PRIMARY_K5]]),
    "sp45_rmse": _rmse(truth, data[model_columns["sp45"]]),
    "fleongg_rmse": _rmse(truth, data[model_columns["fleongg"]]),
    "k5_sp45_error_cosine": k5_sp45_cosine,
    "k5_fleongg_error_cosine": k5_fle_cosine,
    "row_oracle_rmse_k5_sp45_fleongg": row_oracle_rmse,
    "nested_three_way_rmse": float(three_summary["rmse"]),
    "nested_three_way_gain_vs_k5": float(three_summary["gain_vs_k5_equal3"]),
    "nested_three_way_gain_vs_best_component": float(
        three_summary["gain_vs_best_component"]
    ),
    "nested_three_way_min_fold_gain_vs_best": float(
        three_summary["min_fold_gain_vs_best_component"]
    ),
    "nested_three_way_positive_folds_vs_best": int(
        three_summary["positive_folds_vs_best_component"]
    ),
    "fusion_pass": fusion_pass,
    "next": (
        "PASS -> K5 is useful as an external ensemble member; build deployable blend"
        if fusion_pass
        else "K5 does not add enough independent error correction to SP45/fleongg"
    ),
}

print("\nOVERALL MODELS")
display(overall)
print("\nFOLD SCORES")
display(fold_scores)
print("\nERROR COSINES")
display(cosine)
print("\nPREDICTION DISAGREEMENTS")
display(disagreements)
print("\nPER-WELL WINNERS: K5 / SP45 / FLEONGG")
display(winner_counts)
print("\nK5 TOP40")
display(top40_scores)
print("\nNESTED BLEND FOLDS")
display(nested_folds)
print("\nNESTED BLEND SUMMARY")
display(nested_summary)
print("\nRESULT")
print(result)
print("\nVERDICT:", result["next"])

overall.to_csv(RUN_DIR / "overall_models.csv", index=False)
fold_scores.to_csv(RUN_DIR / "fold_scores.csv", index=False)
cosine.to_csv(RUN_DIR / "error_cosines.csv")
disagreements.to_csv(RUN_DIR / "prediction_disagreements.csv", index=False)
bywell.to_csv(RUN_DIR / "bywell.csv", index=False)
winner_counts.to_csv(RUN_DIR / "winner_counts.csv", index=False)
top40_scores.to_csv(RUN_DIR / "top40_scores.csv", index=False)
nested_folds.to_csv(RUN_DIR / "nested_blend_folds.csv", index=False)
nested_summary.to_csv(RUN_DIR / "nested_blend_summary.csv", index=False)
(RUN_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", RUN_DIR)
