"""Nested fixed and per-well confidence blending for STRIDE plus SP45.

The STRIDE recipe touched 40 deterministic Fold-3 wells in V5.  Exclude those
wells, decode the remaining Fold-3 wells, fit the global blend dose there, and
cross-fit a deliberately small per-well Ridge confidence model using only
label-free disagreement/path features.  Freeze the selected rule and apply it
once to the already-saved Fold-2 endpoints from V7.  Fold 1 remains untouched.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT = "stride_sp45_nested_confidence_v8"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "stride_fullsequence_beam_v4_cell.py",
    Path.cwd() / "stride_fullsequence_beam_v4_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
V3_VERDICT = OOF_ROOT / "stride_figure1_posterior_parity_v3" / "verdict.json"
V5_VERDICT = OOF_ROOT / "stride_sequence_scale_repair_v5" / "verdict.json"
V7_DIR = OOF_ROOT / "stride_sp45_cross_evidence_v7"
V7_VERDICT = V7_DIR / "verdict.json"
V7_FOLD2 = V7_DIR / "fold2_stride_sp45_oof.csv.gz"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload the updated stride_fullsequence_beam_v4_cell.py")
for required in (V3_VERDICT, V5_VERDICT, V7_VERDICT, V7_FOLD2):
    if not required.exists():
        raise FileNotFoundError(required)
if POOL_PATH is None:
    raise FileNotFoundError("Missing candidate_pool_full_merged_oof_honest_fle.csv")

ALPHAS = (0.1, 1.0, 10.0, 100.0)
CONFIDENCE_FOLDS = 5
MIN_ADAPTIVE_GAIN = 0.05
MIN_POSITIVE_FOLDS = 4


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "v3" in names:
                break
        body.append(node)
    name = "stride_v4_repaired_definitions_for_v8"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def _metrics(frame, prediction_column):
    error = frame[prediction_column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    well_rmse = np.sqrt(
        pd.Series(error * error).groupby(frame["well"].to_numpy()).mean()
    )
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
    }
    distance = frame["distance_ft"].to_numpy(np.float64)
    for name, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (distance >= low) & (distance < high)
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


def _oracle_weights(frame):
    output = {}
    for well, group in frame.groupby("well", sort=False):
        truth = group["truth"].to_numpy(np.float64)
        sp45 = group["sp45"].to_numpy(np.float64)
        direction = group["stride_raw"].to_numpy(np.float64) - sp45
        denominator = float(np.dot(direction, direction))
        output[well] = 0.0 if denominator <= 1e-12 else float(
            np.clip(-np.dot(sp45 - truth, direction) / denominator, 0.0, 1.0)
        )
    return pd.Series(output, name="oracle_weight")


def _features(frame):
    rows = []
    for well, group in frame.groupby("well", sort=False):
        group = group.sort_values("distance_ft")
        distance = group["distance_ft"].to_numpy(np.float64)
        sp45 = group["sp45"].to_numpy(np.float64)
        stride = group["stride_raw"].to_numpy(np.float64)
        disagreement = stride - sp45
        absolute = np.abs(disagreement)
        centred_distance = distance - np.mean(distance)
        denominator = float(np.dot(centred_distance, centred_distance))
        slope = 0.0 if denominator <= 1e-12 else float(
            np.dot(centred_distance, disagreement - np.mean(disagreement)) / denominator
        )
        item = {
            "well": well,
            "log_rows": float(np.log1p(len(group))),
            "max_distance": float(np.max(distance)),
            "mean_signed_disagreement": float(np.mean(disagreement)),
            "std_disagreement": float(np.std(disagreement)),
            "rms_disagreement": float(np.sqrt(np.mean(disagreement ** 2))),
            "mean_abs_disagreement": float(np.mean(absolute)),
            "p50_abs_disagreement": float(np.quantile(absolute, 0.50)),
            "p90_abs_disagreement": float(np.quantile(absolute, 0.90)),
            "p99_abs_disagreement": float(np.quantile(absolute, 0.99)),
            "endpoint_disagreement": float(disagreement[-1]),
            "disagreement_slope": slope,
            "sp45_span": float(sp45[-1] - sp45[0]),
            "stride_span": float(stride[-1] - stride[0]),
            "sp45_d1_std": float(np.std(np.diff(sp45))) if len(sp45) > 1 else 0.0,
            "stride_d1_std": float(np.std(np.diff(stride))) if len(stride) > 1 else 0.0,
            "d1_disagreement_rms": float(np.sqrt(np.mean(np.diff(disagreement) ** 2))) if len(disagreement) > 1 else 0.0,
        }
        for name, low, high in (
            ("near", 0.0, 500.0),
            ("mid1", 500.0, 1000.0),
            ("mid2", 1000.0, 2000.0),
            ("far", 2000.0, np.inf),
        ):
            mask = (distance >= low) & (distance < high)
            item[f"{name}_rms_disagreement"] = (
                float(np.sqrt(np.mean(disagreement[mask] ** 2))) if mask.any() else 0.0
            )
            item[f"{name}_mean_abs_disagreement"] = (
                float(np.mean(absolute[mask])) if mask.any() else 0.0
            )
        rows.append(item)
    output = pd.DataFrame(rows).set_index("well").sort_index()
    if not np.isfinite(output.to_numpy()).all():
        raise RuntimeError("Non-finite confidence features")
    return output


def _apply_weights(frame, weights, column):
    local_weight = frame["well"].map(weights).to_numpy(np.float64)
    if not np.isfinite(local_weight).all():
        raise RuntimeError(f"Missing weights for {column}")
    frame[column] = frame["sp45"].to_numpy(np.float64) + local_weight * (
        frame["stride_raw"].to_numpy(np.float64) - frame["sp45"].to_numpy(np.float64)
    )


V = _bootstrap(SOURCE_PATH)
v3 = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
v5 = json.loads(V5_VERDICT.read_text(encoding="utf-8"))
v7 = json.loads(V7_VERDICT.read_text(encoding="utf-8"))
prior = v3["selected"]["prior"]
temperature = float(v3["selected"]["emission_temperature"])
recipe = v5["selected_recipe"]

folds = pd.read_csv(V.V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
fold3_wells = sorted(folds.loc[folds["fold"].eq(V.V.SELECTION_FOLD), "well"])
ordered = sorted(
    fold3_wells,
    key=lambda well: hashlib.sha256(f"stride-v5:{well}".encode("ascii")).digest(),
)
touched_wells = set(ordered[:40])
nested_wells = sorted(set(fold3_wells).difference(touched_wells))
if len(fold3_wells) != 154 or len(nested_wells) != 114:
    raise RuntimeError(f"Unexpected Fold3 coverage: {len(fold3_wells)}/{len(nested_wells)}")

records = []
for number, well in enumerate(nested_wells, start=1):
    records.append(V.load_record(well))
    if number % 50 == 0 or number == len(nested_wells):
        print(f"STRIDE v8 load nested Fold3 {number}/{len(nested_wells)}", flush=True)

started = time.time()
V.BEAM_SIZE = 48
stride_rows = []
for number, record in enumerate(records, start=1):
    prediction, _, _ = V._decode(
        record,
        prior,
        temperature,
        "raw_reference",
        effective_spacing_ft=float(recipe["effective_spacing_ft"]),
        persistence_weight=float(recipe["persistence_weight"]),
        lambda_override=float(recipe["lambda_previous_dip"]),
    )
    evaluation = record["evaluation"]
    stride_rows.append(
        pd.DataFrame(
            {
                "id": [f"{record['well']}_{int(row)}" for row in evaluation],
                "well": record["well"],
                "row_idx": evaluation.astype(np.int64),
                "truth": record["truth"][evaluation].astype(np.float32),
                "stride_raw": prediction.astype(np.float32),
                "distance_ft": (record["md"][evaluation] - record["anchor_md"]).astype(np.float32),
            }
        )
    )
    if number % 25 == 0 or number == len(records):
        print(f"STRIDE v8 decode nested Fold3 {number}/{len(records)}", flush=True)
selection = pd.concat(stride_rows, ignore_index=True)

pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "well", "tvt_true", "sp45"],
    dtype={"id": str, "well": str},
)
pool = pool.loc[pool["well"].isin(nested_wells)]
selection = selection.merge(pool, on=["id", "well"], how="inner", validate="one_to_one")
if selection["well"].nunique() != 114:
    raise RuntimeError(f"Nested Fold3 alignment lost wells: {selection['well'].nunique()}")
truth_gap = np.abs(selection["truth"].to_numpy(np.float64) - selection["tvt_true"].to_numpy(np.float64))
if len(selection) == 0 or float(np.max(truth_gap)) > 0.01:
    raise RuntimeError(f"Nested Fold3 truth mismatch: {float(np.max(truth_gap))}")
selection.drop(columns="tvt_true", inplace=True)

truth = selection["truth"].to_numpy(np.float64)
sp45 = selection["sp45"].to_numpy(np.float64)
stride_prediction = selection["stride_raw"].to_numpy(np.float64)
direction = stride_prediction - sp45
fixed_weight = float(np.clip(-np.dot(sp45 - truth, direction) / np.dot(direction, direction), 0.0, 1.0))
selection["fixed_blend"] = sp45 + fixed_weight * direction

features = _features(selection)
oracle = _oracle_weights(selection).reindex(features.index)
feature_columns = features.columns.tolist()
x = features.to_numpy(np.float64)
y = oracle.to_numpy(np.float64)
kfold = KFold(n_splits=CONFIDENCE_FOLDS, shuffle=True, random_state=20260722)
alpha_rows = []
oof_by_alpha = {}
fold_by_well = {}
for fold_number, (_, valid_index) in enumerate(kfold.split(x), start=1):
    for index in valid_index:
        fold_by_well[features.index[index]] = fold_number

for alpha in ALPHAS:
    oof = np.zeros(len(features), np.float64)
    for train_index, valid_index in kfold.split(x):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train_index], y[train_index])
        oof[valid_index] = np.clip(model.predict(x[valid_index]), 0.0, 1.0)
    weights = pd.Series(oof, index=features.index)
    column = f"adaptive_alpha_{str(alpha).replace('.', 'p')}"
    _apply_weights(selection, weights, column)
    metrics = _metrics(selection, column)
    alpha_rows.append({"alpha": alpha, **metrics})
    oof_by_alpha[alpha] = weights
alpha_frame = pd.DataFrame(alpha_rows).sort_values("rmse")
selected_alpha = float(alpha_frame.iloc[0]["alpha"])
selection_weights = oof_by_alpha[selected_alpha]
selection["adaptive_oof"] = selection["well"].map(selection_weights).to_numpy() * direction + sp45

fold_rows = []
for fold_number in range(1, CONFIDENCE_FOLDS + 1):
    wells = [well for well, local_fold in fold_by_well.items() if local_fold == fold_number]
    local = selection.loc[selection["well"].isin(wells)]
    fixed_rmse = _metrics(local, "fixed_blend")["rmse"]
    adaptive_rmse = _metrics(local, "adaptive_oof")["rmse"]
    fold_rows.append(
        {
            "confidence_fold": fold_number,
            "wells": len(wells),
            "fixed_rmse": fixed_rmse,
            "adaptive_rmse": adaptive_rmse,
            "gain": fixed_rmse - adaptive_rmse,
        }
    )
fold_frame = pd.DataFrame(fold_rows)
selection_metrics = {
    "sp45": _metrics(selection, "sp45"),
    "stride_raw": _metrics(selection, "stride_raw"),
    "fixed_blend": _metrics(selection, "fixed_blend"),
    "adaptive_oof": _metrics(selection, "adaptive_oof"),
}
adaptive_gain = selection_metrics["fixed_blend"]["rmse"] - selection_metrics["adaptive_oof"]["rmse"]
positive_folds = int((fold_frame["gain"] > 0.0).sum())
use_adaptive = bool(adaptive_gain >= MIN_ADAPTIVE_GAIN and positive_folds >= MIN_POSITIVE_FOLDS)

final_model = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha))
final_model.fit(x, y)
fold2 = pd.read_csv(V7_FOLD2, dtype={"id": str, "well": str})
fold2_features = _features(fold2).reindex(columns=feature_columns)
fold2_adaptive_weights = pd.Series(
    np.clip(final_model.predict(fold2_features.to_numpy(np.float64)), 0.0, 1.0),
    index=fold2_features.index,
)
fold2_fixed_weights = pd.Series(fixed_weight, index=fold2_features.index)
_apply_weights(fold2, fold2_fixed_weights, "fixed_nested")
_apply_weights(fold2, fold2_adaptive_weights, "adaptive_nested")
selected_column = "adaptive_nested" if use_adaptive else "fixed_nested"
fold2["selected_nested"] = fold2[selected_column]
fold2_metrics = {
    "sp45": _metrics(fold2, "sp45"),
    "stride_raw": _metrics(fold2, "stride_raw"),
    "fixed_nested": _metrics(fold2, "fixed_nested"),
    "adaptive_nested": _metrics(fold2, "adaptive_nested"),
    "selected_nested": _metrics(fold2, "selected_nested"),
}

alpha_frame.to_csv(OUT_DIR / "confidence_alpha_scan.csv", index=False)
fold_frame.to_csv(OUT_DIR / "confidence_crossfit_folds.csv", index=False)
pd.DataFrame(
    {
        "well": features.index,
        "oracle_weight": oracle,
        "oof_predicted_weight": selection_weights,
        "confidence_fold": [fold_by_well[well] for well in features.index],
    }
).to_csv(OUT_DIR / "fold3_nested_confidence_weights.csv", index=False)
pd.DataFrame(
    {
        "well": fold2_features.index,
        "fixed_weight": fixed_weight,
        "adaptive_weight": fold2_adaptive_weights,
    }
).to_csv(OUT_DIR / "fold2_applied_weights.csv", index=False)
selection[["id", "well", "row_idx", "truth", "stride_raw", "sp45", "distance_ft"]].to_csv(
    OUT_DIR / "fold3_nested_stride_sp45_oof.csv.gz", index=False, compression="gzip"
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_nested_weight_selection_to_Fold2_single_application",
    "protocol": {
        "prior_folds": list(V.V.PRIOR_FOLDS),
        "recipe_touched_fold3_wells_excluded": 40,
        "nested_selection_fold": V.V.SELECTION_FOLD,
        "nested_selection_wells": 114,
        "application_fold": V.V.APPLICATION_FOLD,
        "untouched_fold": V.V.UNTOUCHED_FOLD,
        "fold2_used_for_selection": False,
    },
    "frozen_stride_recipe": {**recipe, "reference": "raw", "likelihood": "cauchy"},
    "fixed_weight_stride": fixed_weight,
    "confidence_model": {
        "model": "StandardScaler + Ridge",
        "features": feature_columns,
        "alphas": list(ALPHAS),
        "selected_alpha": selected_alpha,
        "crossfit_adaptive_gain": adaptive_gain,
        "positive_confidence_folds": positive_folds,
        "use_adaptive": use_adaptive,
    },
    "selection_metrics": selection_metrics,
    "application_metrics": fold2_metrics,
    "application_gain_vs_sp45": float(
        fold2_metrics["sp45"]["rmse"] - fold2_metrics["selected_nested"]["rmse"]
    ),
    "reference_only": {
        "fold2_oracle_fixed_weight_stride": v7["oracle_audit"]["fixed_weight_stride"],
        "fold2_oracle_fixed_rmse": v7["metrics"]["fixed_blend"]["rmse"],
        "fold2_perwell_oracle_rmse": v7["metrics"]["perwell_oracle"]["rmse"],
    },
    "gates": {
        "fixed_weight_nonzero": bool(fixed_weight >= 0.10),
        "fold2_gain_ge_0p30": bool(
            fold2_metrics["sp45"]["rmse"] - fold2_metrics["selected_nested"]["rmse"] >= 0.30
        ),
        "adaptive_requires_crossfit_support": bool(not use_adaptive or (adaptive_gain >= MIN_ADAPTIVE_GAIN and positive_folds >= MIN_POSITIVE_FOLDS)),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> generate five-fold cross-fit STRIDE endpoint and combine with the stronger deployment chain"
    if verdict["strict_pass"]
    else "FAIL -> close this blend rule; Fold2 did not confirm the nested Fold3 dose"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV8 CONFIDENCE ALPHA SCAN")
print(alpha_frame.to_string(index=False), flush=True)
print("\nV8 CONFIDENCE FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nV8 FOLD2 APPLICATION")
print(pd.DataFrame(fold2_metrics).T.to_string(), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
