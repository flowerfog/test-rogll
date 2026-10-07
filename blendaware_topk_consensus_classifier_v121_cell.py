"""V121: blend-aware top-K classifiers consumed as top-33 path consensus.

This combines two mechanisms that were only tested separately before:
blend-aware candidate labels and binary top-K classification. Classifier
probabilities define a deduplicated top-33 set; the deployed correction remains
the frozen V113 tail-guarded consensus move on exact V76.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "blendaware_topk_consensus_classifier_v121"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
BASE_PATH = OOF_ROOT / "rawdata_lattice_consensus_center_fold2_replay_v115" / "fold123_consensus_center_oof.parquet"
PANEL_DIR = OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103" / "matched_panels"
TRAIN_FOLDS = (4, 5)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
POSITIVE_K = (4, 8, 16, 32, 64)
CONSENSUS_K = 33
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
TRUTH = "tvt_true"
BASE = "consensus_center"
COARSE = "coarse_crossfit"


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


def _bootstrap_v110():
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("blendaware_v121_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


for required in (V110_SOURCE, BASE_PATH, PANEL_DIR):
    if not required.exists():
        raise FileNotFoundError(required)
R = _bootstrap_v110()
N = R._bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
fold_by_well = {str(well): int(fold) for well, fold in N.fold_by_well.items()}

panel_files = {path.stem: path for path in PANEL_DIR.glob("*.npz")}
train_wells = sorted(well for well, fold in fold_by_well.items() if fold in TRAIN_FOLDS and well in panel_files)
fold3_wells = sorted(well for well, fold in fold_by_well.items() if fold == EVAL_FOLD and well in panel_files)
if len(train_wells) < 300 or len(fold3_wells) < 153:
    raise RuntimeError(f"Incomplete panels train/fold3={len(train_wells)}/{len(fold3_wells)}")


def _load_panel(well):
    with np.load(panel_files[well], allow_pickle=False) as data:
        panel = {key: data[key] for key in data.files}
    required = {"features", "paths", "mse", "blend_mse"}
    missing = sorted(required.difference(panel))
    if missing:
        raise RuntimeError(f"{well}: panel missing {missing}")
    return panel


def _training_matrix(positive_k):
    x_parts, y_parts, weight_parts = [], [], []
    for number, well in enumerate(train_wells, 1):
        panel = _load_panel(well)
        indices = Q._training_indices(panel, well)
        risk = panel["blend_mse"].astype(np.float64)[indices]
        order = np.argsort(risk, kind="stable")
        k = min(int(positive_k), len(indices) - 1)
        labels = np.zeros(len(indices), np.int8)
        labels[order[:k]] = 1
        positive = int(labels.sum())
        negative = int(len(labels) - positive)
        weights = np.where(labels == 1, 0.5 / positive, 0.5 / negative).astype(np.float32)
        x_parts.append(panel["features"][indices].astype(np.float32, copy=False))
        y_parts.append(labels)
        weight_parts.append(weights)
        if number % 100 == 0 or number == len(train_wells):
            print(f"V121 matrix top{positive_k} {number}/{len(train_wells)}", flush=True)
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(weight_parts)


models = {}
for positive_k in POSITIVE_K:
    x_train, y_train, sample_weight = _training_matrix(positive_k)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=4.0,
        random_state=20260801 + positive_k,
        verbosity=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    models[positive_k] = model.booster_
    model.booster_.save_model(str(MODEL_DIR / f"blendaware_top{positive_k}.txt"))
    print("V121 MODEL", {"topk": positive_k, "rows": len(y_train), "positive": int(y_train.sum())}, flush=True)

base_frame = pd.read_parquet(BASE_PATH, filters=[("fold", "=", EVAL_FOLD)])
base_frame = base_frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
base_frame["well"] = base_frame["well"].astype(str)
missing_lineage = [column for column in ("incumbent_6p377_replay", COARSE) if column not in base_frame]
if missing_lineage:
    lineage = pd.read_parquet(
        R.V70_PATH,
        columns=["id", *missing_lineage],
        filters=[("fold", "=", EVAL_FOLD)],
    )
    if lineage["id"].duplicated().any() or base_frame["id"].duplicated().any():
        raise RuntimeError("Duplicate ids during V121 lineage enrichment")
    base_frame = base_frame.merge(lineage, on="id", how="left", validate="one_to_one")
    if base_frame[missing_lineage].isna().any().any():
        raise RuntimeError("Incomplete V121 endpoint lineage")
weights = pd.read_parquet(R.V76_WEIGHTS)
weights = weights[weights["fold"].eq(EVAL_FOLD)][["well", "ensemble_weight"]].copy()
weights["well"] = weights["well"].astype(str)
dose_by_well = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))
fold3 = base_frame.copy()
if fold3["well"].nunique() < 153:
    raise RuntimeError("Incomplete Fold3 base endpoint")

prediction_parts = {positive_k: [] for positive_k in POSITIVE_K}
base_parts = []
for number, well in enumerate(fold3_wells, 1):
    local = fold3[fold3["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    panel = _load_panel(well)
    record = V["_load_record"](well)
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: Fold3 alignment")
    paths = panel["paths"].astype(np.int16, copy=False)
    features = panel["features"].astype(np.float32, copy=False)
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    exact = local["exact_v76"].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    dose = float(dose_by_well.get(well, np.nan))
    if not np.isfinite(dose):
        raise RuntimeError(f"{well}: missing V76 dose")
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    base_parts.append(local)
    for positive_k, model in models.items():
        probability = model.predict(features)
        order = R._deduplicated_order(paths, probability)
        if len(order) < CONSENSUS_K:
            raise RuntimeError(f"{well}: only {len(order)} unique classified candidates")
        center = np.mean(
            np.vstack([
                B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
                for index in order[:CONSENSUS_K]
            ]),
            axis=0,
        )
        move = taper * np.clip(dose * (center - coarse), -CAP_FT, CAP_FT)
        prediction_parts[positive_k].append(exact + move)
    if number % 10 == 0 or number == len(fold3_wells):
        print(f"V121 evaluate Fold3 {number}/{len(fold3_wells)}", flush=True)

evaluation = pd.concat(base_parts, ignore_index=True)
base_prediction = evaluation[BASE].to_numpy(np.float64)
ordered_wells = sorted(evaluation["well"].unique())
discovery_set = set(ordered_wells[:DISCOVERY_WELLS])
discovery = evaluation["well"].isin(discovery_set).to_numpy()
confirmation = ~discovery
base_discovery = R._metric(evaluation.loc[discovery], base_prediction[discovery], "consensus_top33")
base_confirmation = R._metric(evaluation.loc[confirmation], base_prediction[confirmation], "consensus_top33")

scan_rows = []
predictions = {}
for positive_k in POSITIVE_K:
    prediction = np.concatenate(prediction_parts[positive_k])
    predictions[positive_k] = prediction
    metric = R._metric(evaluation.loc[discovery], prediction[discovery], f"blendaware_top{positive_k}")
    metric.update({
        "positive_k": positive_k,
        "gain": float(base_discovery["rmse"] - metric["rmse"]),
        "far_gain": float(base_discovery["far_gt2000_rmse"] - metric["far_gt2000_rmse"]),
        "p95_gain": float(base_discovery["p95_well_rmse"] - metric["p95_well_rmse"]),
    })
    scan_rows.append(metric)
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
selected_k = int(scan.iloc[0]["positive_k"])
selected_prediction = predictions[selected_k]
candidate_discovery = R._metric(evaluation.loc[discovery], selected_prediction[discovery], "blendaware_classifier_consensus")
candidate_confirmation = R._metric(evaluation.loc[confirmation], selected_prediction[confirmation], "blendaware_classifier_consensus")

values = {
    "discovery_gain": float(base_discovery["rmse"] - candidate_discovery["rmse"]),
    "confirmation_gain": float(base_confirmation["rmse"] - candidate_confirmation["rmse"]),
    "confirmation_far_gain": float(base_confirmation["far_gt2000_rmse"] - candidate_confirmation["far_gt2000_rmse"]),
    "confirmation_p95_gain": float(base_confirmation["p95_well_rmse"] - candidate_confirmation["p95_well_rmse"]),
}
checks = {
    "discovery_gain_ge_0p05": values["discovery_gain"] >= 0.05,
    "confirmation_gain_ge_0p05": values["confirmation_gain"] >= 0.05,
    "confirmation_far_gain_nonnegative": values["confirmation_far_gain"] >= 0.0,
    "confirmation_p95_not_worse_by_gt_0p10": values["confirmation_p95_gain"] >= -0.10,
}
strict_pass = bool(all(checks.values()))

summary = pd.DataFrame([base_discovery, candidate_discovery, base_confirmation, candidate_confirmation])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
output = evaluation[["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
output["blendaware_classifier_consensus"] = selected_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_blendaware_classifier_consensus_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_blendaware_binary_training_Fold3_discovery_confirmation",
    "sample_unit": "one complete physical well",
    "candidate_bank": "V4 frozen three-arm union",
    "changed_objective": "blend-aware top-K binary classification",
    "consumption": "deduplicated top33 uniform trajectory center on exact V76",
    "train_wells": len(train_wells),
    "selection_wells": len(discovery_set),
    "confirmation_wells": int(evaluation.loc[confirmation, "well"].nunique()),
    "selected_positive_k": selected_k,
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "fold2_touched": False,
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> cross-fit classifier consensus and build Kaggle bundle; FAIL -> candidate evidence cannot learn the blend-aware oracle",
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("\nV121 DISCOVERY SCAN")
print(scan.to_string(index=False), flush=True)
print("\nV121 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV121 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
