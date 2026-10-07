"""V101: nested legal binary router between V76 and V96.

V100 showed no fixed-blend head but a 0.124-ft per-well oracle head concentrated
at the two endpoints. Fold 4/5 train legal well-level classifiers, Fold 3
selects one model/consumption mode, Fold 2 confirms, and Fold 1 is touched only
after confirmation passes.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_v76_v96_binary_router_v101"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V96_SOURCE = ROOT / "rawdata_lattice_v76_full773_training_v96_cell.py"
V96_COARSE = (
    OOF_ROOT / "rawdata_lattice_v76_full773_training_v96"
    / "fivefold_coarse_oof.parquet"
)
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
MODEL_NAMES = ("logistic", "hist", "ensemble")
MODES = ("hard", "soft")


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _bootstrap_v96():
    if not V96_SOURCE.is_file():
        raise FileNotFoundError(V96_SOURCE)
    source = V96_SOURCE.read_text(encoding="utf-8")
    marker = "for path in (V4_SOURCE, V66_PATH, V70_PATH, V76_BUNDLE, V76_OOF_WEIGHTS):"
    if marker not in source:
        raise RuntimeError(f"Cannot identify V96 definition boundary in {V96_SOURCE}")
    module = types.ModuleType("v101_v96_definitions")
    module.__file__ = str(V96_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V96_SOURCE), "exec"), module.__dict__)
    return module


def _metrics(frame, prediction, endpoint):
    truth = frame[N.TRUTH].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    by_well = pd.DataFrame(
        {"well": frame["well"].astype(str), "e2": error ** 2}
    ).groupby("well")["e2"].mean()
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well.to_numpy()))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well.to_numpy()), 0.95)),
        "far_gt2000_rmse": (
            float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan
        ),
    }


def _models():
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, max_iter=2000, random_state=20260731),
    )
    hist = HistGradientBoostingClassifier(
        max_iter=140,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=35,
        l2_regularization=12.0,
        random_state=20260731,
    )
    return logistic, hist


def _positive_probability(model, matrix):
    probability = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(matrix), np.float64)
    return probability[:, classes.index(1)].astype(np.float64)


N = _bootstrap_v96()
for required in (V96_COARSE, N.V76_BUNDLE, N.V76_OOF_WEIGHTS):
    if not Path(required).is_file():
        raise FileNotFoundError(required)

complete = pd.read_parquet(V96_COARSE).sort_values(
    ["fold", "well", "row_idx"], kind="stable"
).reset_index(drop=True)
if len(complete) != 3_783_989 or complete["well"].nunique() != 773:
    raise RuntimeError("Invalid V96 complete coarse OOF")

features = pd.DataFrame(
    [
        N._feature_row(local)
        for _, local in complete.groupby(["fold", "well"], sort=False)
    ]
).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
x, feature_names = N._augment(features)
y_oracle = features["oracle_weight"].to_numpy(np.float64)
row_count = features["rows"].to_numpy(np.float64)

# Reconstruct the exact V96 crossfit dose.
v96_weight = np.full(len(features), np.nan, np.float64)
for heldout_fold in N.ALL_FOLDS:
    valid = features["fold"].to_numpy(np.int64) == int(heldout_fold)
    train = ~valid
    hist = N._hist()
    ridge = N._ridge()
    hist.fit(x[train], y_oracle[train], sample_weight=np.sqrt(row_count[train]))
    ridge.fit(x[train], y_oracle[train])
    v96_weight[valid] = np.clip(
        0.5 * hist.predict(x[valid]) + 0.5 * ridge.predict(x[valid]),
        N.MIN_WEIGHT,
        N.MAX_WEIGHT,
    )

# Reconstruct the exact matched V76 dose.
bundle = joblib.load(N.V76_BUNDLE)
v76_weight = np.clip(
    bundle["hist_fraction"] * bundle["hist_model"].predict(x)
    + bundle["ridge_fraction"] * bundle["ridge_model"].predict(x),
    N.MIN_WEIGHT,
    N.MAX_WEIGHT,
)
old_oof = pd.read_parquet(
    N.V76_OOF_WEIGHTS, columns=["fold", "well", "ensemble_weight"]
)
old_map = dict(
    zip(old_oof["well"].astype(str), old_oof["ensemble_weight"].astype(float))
)
honest = features["fold"].isin(N.PRIOR_FOLDS).to_numpy()
v76_weight[honest] = (
    features.loc[honest, "well"].astype(str).map(old_map).to_numpy(np.float64)
)
if not np.isfinite(v76_weight).all() or not np.isfinite(v96_weight).all():
    raise RuntimeError("Non-finite reconstructed endpoint weights")

v76_map = dict(zip(features["well"].astype(str), v76_weight))
v96_map = dict(zip(features["well"].astype(str), v96_weight))
row_v76_weight = complete["well"].astype(str).map(v76_map).to_numpy(np.float64)
row_v96_weight = complete["well"].astype(str).map(v96_map).to_numpy(np.float64)
parent = complete[N.PARENT].to_numpy(np.float64)
move = complete[N.COARSE].to_numpy(np.float64) - parent
truth = complete[N.TRUTH].to_numpy(np.float64)
v76_prediction = parent + row_v76_weight * move
v96_prediction = parent + row_v96_weight * move

# Binary labels and importance are based only on each training well's endpoint
# SSE difference. Router inputs remain the legal features returned by _augment.
v76_sse = np.empty(len(features), np.float64)
v96_sse = np.empty(len(features), np.float64)
row_indices_by_well = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in complete.groupby("well", sort=False).indices.items()
}
for index, well in enumerate(features["well"].astype(str)):
    rows = row_indices_by_well[well]
    v76_error = v76_prediction[rows] - truth[rows]
    v96_error = v96_prediction[rows] - truth[rows]
    v76_sse[index] = np.dot(v76_error, v76_error)
    v96_sse[index] = np.dot(v96_error, v96_error)
label = (v96_sse < v76_sse).astype(np.int8)
raw_importance = np.abs(v76_sse - v96_sse)
importance_cap = float(np.quantile(raw_importance, 0.95))
importance = np.minimum(raw_importance, importance_cap)
importance = np.maximum(importance / max(float(np.mean(importance)), 1e-8), 0.05)

well_to_feature_index = {
    str(well): index for index, well in enumerate(features["well"].astype(str))
}


def _fit_probabilities(train_folds, eval_fold):
    train = features["fold"].isin(train_folds).to_numpy()
    valid = features["fold"].eq(eval_fold).to_numpy()
    if train.sum() < 250 or valid.sum() < 150:
        raise RuntimeError(
            f"Invalid router split train={train.sum()} valid={valid.sum()}"
        )
    if np.unique(label[train]).size < 2:
        prevalence = float(np.average(label[train], weights=importance[train]))
        base = np.full(valid.sum(), prevalence, np.float64)
        return {
            "logistic": base,
            "hist": base,
            "ensemble": base,
            "constant": base,
        }
    logistic, hist = _models()
    logistic.fit(x[train], label[train], logisticregression__sample_weight=importance[train])
    hist.fit(x[train], label[train], sample_weight=importance[train])
    logistic_probability = _positive_probability(logistic, x[valid])
    hist_probability = _positive_probability(hist, x[valid])
    prevalence = float(np.average(label[train], weights=importance[train]))
    return {
        "logistic": logistic_probability,
        "hist": hist_probability,
        "ensemble": 0.5 * logistic_probability + 0.5 * hist_probability,
        "constant": np.full(valid.sum(), prevalence, np.float64),
    }


def _evaluate_phase(train_folds, eval_fold, stage, selected=None):
    probabilities = _fit_probabilities(tuple(train_folds), int(eval_fold))
    feature_mask = features["fold"].eq(eval_fold).to_numpy()
    valid_wells = features.loc[feature_mask, "well"].astype(str).tolist()
    frame_mask = complete["fold"].eq(eval_fold).to_numpy()
    frame = complete.loc[frame_mask]
    base = v76_prediction[frame_mask]
    candidate = v96_prediction[frame_mask]
    summary = [{**_metrics(frame, base, "v76"), "stage": stage, "model": "base", "mode": "base"}]
    probability_rows = []
    recipes = (
        [(model, mode) for model in MODEL_NAMES for mode in MODES]
        if selected is None else [selected]
    )
    for model, mode in recipes:
        probability_map = dict(zip(valid_wells, probabilities[model]))
        row_probability = (
            frame["well"].astype(str).map(probability_map).to_numpy(np.float64)
        )
        alpha = (row_probability >= 0.5).astype(np.float64) if mode == "hard" else row_probability
        prediction = base + alpha * (candidate - base)
        summary.append(
            {
                **_metrics(frame, prediction, f"{model}_{mode}"),
                "stage": stage,
                "model": model,
                "mode": mode,
            }
        )
        for well, probability in zip(valid_wells, probabilities[model]):
            probability_rows.append(
                {
                    "stage": stage,
                    "fold": int(eval_fold),
                    "well": well,
                    "model": model,
                    "mode": mode,
                    "probability_v96": float(probability),
                    "label_v96_better": int(label[well_to_feature_index[well]]),
                }
            )
    # Constant soft routing is a matched no-feature control.
    constant_map = dict(zip(valid_wells, probabilities["constant"]))
    constant_probability = (
        frame["well"].astype(str).map(constant_map).to_numpy(np.float64)
    )
    constant_prediction = base + constant_probability * (candidate - base)
    summary.append(
        {
            **_metrics(frame, constant_prediction, "constant_soft_control"),
            "stage": stage,
            "model": "constant",
            "mode": "soft",
        }
    )
    return pd.DataFrame(summary), pd.DataFrame(probability_rows)


selection_summary, selection_probabilities = _evaluate_phase(
    TRAIN_FOLDS, SELECTION_FOLD, "fold3_selection"
)
selection_base = selection_summary.loc[selection_summary["endpoint"].eq("v76")].iloc[0]
candidate_selection = selection_summary.loc[
    selection_summary["model"].isin(MODEL_NAMES)
].copy()
candidate_selection["gain"] = (
    float(selection_base["rmse"]) - candidate_selection["rmse"]
)
candidate_selection["far_gain"] = (
    float(selection_base["far_gt2000_rmse"])
    - candidate_selection["far_gt2000_rmse"]
)
candidate_selection = candidate_selection.sort_values(
    ["rmse", "p95_well_rmse"], ascending=True
)
best = candidate_selection.iloc[0]
selected_recipe = (str(best["model"]), str(best["mode"]))
selection_pass = bool(
    float(best["gain"]) >= 0.02
    and float(best["far_gain"]) >= 0.0
    and float(selection_base["p95_well_rmse"]) - float(best["p95_well_rmse"]) >= -0.05
)

confirmation_summary = None
confirmation_probabilities = None
confirmation_pass = False
final_summary = None
final_probabilities = None
final_pass = False
if selection_pass:
    confirmation_summary, confirmation_probabilities = _evaluate_phase(
        (3, 4, 5), CONFIRMATION_FOLD, "fold2_confirmation", selected_recipe
    )
    confirmation_base = confirmation_summary.loc[
        confirmation_summary["endpoint"].eq("v76")
    ].iloc[0]
    confirmation_candidate = confirmation_summary.loc[
        confirmation_summary["model"].eq(selected_recipe[0])
        & confirmation_summary["mode"].eq(selected_recipe[1])
    ].iloc[0]
    confirmation_gain = float(
        confirmation_base["rmse"] - confirmation_candidate["rmse"]
    )
    confirmation_far_gain = float(
        confirmation_base["far_gt2000_rmse"]
        - confirmation_candidate["far_gt2000_rmse"]
    )
    confirmation_p95_gain = float(
        confirmation_base["p95_well_rmse"]
        - confirmation_candidate["p95_well_rmse"]
    )
    confirmation_pass = bool(
        confirmation_gain >= 0.02
        and confirmation_far_gain >= 0.0
        and confirmation_p95_gain >= -0.05
    )
    if confirmation_pass:
        final_summary, final_probabilities = _evaluate_phase(
            (2, 3, 4, 5), FINAL_FOLD, "fold1_final", selected_recipe
        )
        final_base = final_summary.loc[final_summary["endpoint"].eq("v76")].iloc[0]
        final_candidate = final_summary.loc[
            final_summary["model"].eq(selected_recipe[0])
            & final_summary["mode"].eq(selected_recipe[1])
        ].iloc[0]
        final_pass = bool(
            float(final_base["rmse"] - final_candidate["rmse"]) >= 0.0
            and float(
                final_base["far_gt2000_rmse"]
                - final_candidate["far_gt2000_rmse"]
            ) >= 0.0
            and float(final_base["p95_well_rmse"] - final_candidate["p95_well_rmse"])
            >= -0.05
        )

all_summary = pd.concat(
    [
        frame
        for frame in (selection_summary, confirmation_summary, final_summary)
        if frame is not None
    ],
    ignore_index=True,
)
all_probabilities = pd.concat(
    [
        frame
        for frame in (
            selection_probabilities,
            confirmation_probabilities,
            final_probabilities,
        )
        if frame is not None
    ],
    ignore_index=True,
)
strict_pass = bool(selection_pass and confirmation_pass and final_pass)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_binary_router_training_Fold3_selection_Fold2_confirmation_Fold1_final",
    "legal_inputs_only": True,
    "target": "which endpoint has lower complete-well SSE",
    "training_weight": "absolute endpoint SSE difference clipped at p95",
    "candidate_models": list(MODEL_NAMES),
    "candidate_modes": list(MODES),
    "selection_search_count": len(MODEL_NAMES) * len(MODES),
    "selected_recipe": {
        "model": selected_recipe[0],
        "mode": selected_recipe[1],
    },
    "selection": {
        "gain": float(best["gain"]),
        "far_gain": float(best["far_gain"]),
        "pass": selection_pass,
    },
    "confirmation_pass": confirmation_pass,
    "fold1_touched": bool(confirmation_pass),
    "final_pass": final_pass,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> fit the frozen router on all legal training wells and export bundle"
        if strict_pass
        else "FAIL -> V76/V96 fold reversal is not legally routable; retain V76"
    ),
}

all_summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
all_probabilities.to_csv(OUT_DIR / "well_probabilities.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)

print("V101 MATCHED SUMMARY")
print(all_summary.to_string(index=False))
print("\nV101 SELECTION TOP")
print(candidate_selection.to_string(index=False))
print("\nV101 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
