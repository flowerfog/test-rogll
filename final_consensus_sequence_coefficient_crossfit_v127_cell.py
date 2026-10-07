"""V127: three-fold cross-fit audit of the V126 sequence coefficient head.

Each Fold1/2/3 endpoint is produced by a model trained on the other two folds.
The V126 architecture, optimizer, and three-epoch budget are frozen. For every
held-out fold, the consumption recipe is selected only from the other two OOF
folds. No held-out truth participates in model or recipe selection.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "final_consensus_sequence_coefficient_crossfit_v127"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V126_SOURCE = ROOT / "final_consensus_sequence_coefficient_v126_cell.py"
FOLDS = (1, 2, 3)
FIXED_EPOCHS = 3
SEED = 20260801
FROZEN_V126_RECIPE = ("tail_affine", 1.0, 4.0)


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


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v126():
    if not V126_SOURCE.is_file():
        raise FileNotFoundError(V126_SOURCE)
    source = V126_SOURCE.read_text(encoding="utf-8")
    marker = "for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):"
    if marker not in source:
        raise RuntimeError(f"V126 definition marker missing from {V126_SOURCE}")
    module = types.ModuleType("final_consensus_v127_v126_definitions")
    module.__file__ = str(V126_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V126_SOURCE), "exec"), module.__dict__)
    return module


Q = _bootstrap_v126()
Q._seed_everything(SEED)
R = Q._bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

for required in (Q.BASE_PATH, Q.TARGET_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(Q.BASE_PATH)
frame["well"] = frame["well"].astype(str)
required_base = {
    "id", "well", "row_idx", "fold", Q.TRUTH, "md_since", Q.BASE, "exact_v76"
}
missing_base = sorted(required_base.difference(frame.columns))
if missing_base:
    raise RuntimeError(f"Base OOF missing columns: {missing_base}")
if frame["id"].duplicated().any():
    raise RuntimeError("Base OOF contains duplicate ids")

missing_lineage = [column for column in (Q.PARENT, Q.COARSE) if column not in frame]
if missing_lineage:
    lineage = pd.read_parquet(
        R.V70_PATH,
        columns=["id", "fold", Q.TRUTH, *missing_lineage],
        filters=[("fold", "in", list(FOLDS))],
    )
    frame = frame.merge(
        lineage, on="id", how="left", validate="one_to_one", suffixes=("", "_lineage")
    )
    if frame[missing_lineage].isna().any().any():
        raise RuntimeError("Incomplete V70 endpoint lineage")
    truth_gap = float(np.max(np.abs(frame[Q.TRUTH] - frame[f"{Q.TRUTH}_lineage"])))
    fold_gap = int(np.max(np.abs(frame["fold"] - frame["fold_lineage"])))
    if truth_gap > 0.01 or fold_gap != 0:
        raise RuntimeError(f"Lineage mismatch truth={truth_gap} fold={fold_gap}")
    frame = frame.drop(columns=[f"{Q.TRUTH}_lineage", "fold_lineage"])
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
well_positions = {
    str(well): np.asarray(positions, np.int64)
    for well, positions in frame.groupby("well", sort=False).indices.items()
}


def _well_frame(well):
    return frame.iloc[well_positions[str(well)]]

targets = pd.read_csv(Q.TARGET_PATH, dtype={"well": str})
targets["well"] = targets["well"].astype(str)
targets = targets[["well", "fold", *Q.TARGET_COLUMNS]].copy()
if targets["well"].duplicated().any():
    raise RuntimeError("V123 targets contain duplicate wells")
target_map = targets.set_index("well")

well_order = sorted(frame["well"].unique())
missing_targets = sorted(set(well_order).difference(target_map.index))
if missing_targets:
    raise RuntimeError(f"V123 targets missing {len(missing_targets)} wells: {missing_targets[:10]}")

sequence_rows = []
metadata_rows = []
target_rows = []
fold_rows = []
for number, well in enumerate(well_order, 1):
    local = _well_frame(well)
    local_fold = int(local["fold"].iloc[0])
    target_fold = int(target_map.loc[well, "fold"])
    if local_fold != target_fold:
        raise RuntimeError(f"{well}: base/target fold mismatch {local_fold}/{target_fold}")
    sequence, metadata = Q._sequence_features(local, Q._load_record_relaxed(V, well))
    sequence_rows.append(sequence)
    metadata_rows.append(metadata)
    target_rows.append(target_map.loc[well, list(Q.TARGET_COLUMNS)].to_numpy(np.float32))
    fold_rows.append(local_fold)
    if number % 100 == 0 or number == len(well_order):
        print(f"V127 sequences {number}/{len(well_order)}", flush=True)

sequence_raw = np.stack(sequence_rows).astype(np.float32)
metadata_raw = np.stack(metadata_rows).astype(np.float32)
target_raw = np.stack(target_rows).astype(np.float32)
fold_by_index = np.asarray(fold_rows, np.int8)
if len(well_order) != 464 or set(fold_by_index.tolist()) != set(FOLDS):
    raise RuntimeError(f"Unexpected V127 coverage wells={len(well_order)} folds={set(fold_by_index)}")


def _fit_predict(heldout_fold):
    train_index = np.flatnonzero(fold_by_index != heldout_fold)
    valid_index = np.flatnonzero(fold_by_index == heldout_fold)
    sequence_mean = np.mean(sequence_raw[train_index], axis=(0, 2), keepdims=True)
    sequence_std = np.maximum(
        np.std(sequence_raw[train_index], axis=(0, 2), keepdims=True), 1e-3
    )
    metadata_mean = np.mean(metadata_raw[train_index], axis=0, keepdims=True)
    metadata_std = np.maximum(np.std(metadata_raw[train_index], axis=0, keepdims=True), 1e-3)
    target_mean = np.mean(target_raw[train_index], axis=0, keepdims=True)
    target_std = np.maximum(np.std(target_raw[train_index], axis=0, keepdims=True), 0.25)
    target_low = np.quantile(target_raw[train_index], 0.01, axis=0)
    target_high = np.quantile(target_raw[train_index], 0.99, axis=0)

    sequence = ((sequence_raw - sequence_mean) / sequence_std).astype(np.float32)
    metadata = ((metadata_raw - metadata_mean) / metadata_std).astype(np.float32)
    target_normalized = np.clip(
        (target_raw - target_mean) / target_std, -6.0, 6.0
    ).astype(np.float32)

    Q._seed_everything(SEED + int(heldout_fold))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Q.SequenceCoefficientEncoder(sequence.shape[1], metadata.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Q.LEARNING_RATE, weight_decay=Q.WEIGHT_DECAY
    )
    criterion = nn.SmoothL1Loss(beta=0.75)
    dataset = TensorDataset(
        torch.from_numpy(sequence[train_index]),
        torch.from_numpy(metadata[train_index]),
        torch.from_numpy(target_normalized[train_index]),
    )
    generator = torch.Generator().manual_seed(SEED + int(heldout_fold))
    loader = DataLoader(
        dataset, batch_size=Q.BATCH_SIZE, shuffle=True, generator=generator
    )
    history = []
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        losses = []
        for batch_sequence, batch_metadata, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_sequence.to(device), batch_metadata.to(device))
            loss = criterion(prediction, batch_target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({
            "heldout_fold": heldout_fold,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
        })
    prediction = Q._predict(model, sequence[valid_index], metadata[valid_index], device)
    prediction = prediction * target_std + target_mean
    prediction = np.clip(prediction, target_low, target_high).astype(np.float32)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "sequence_mean": sequence_mean,
            "sequence_std": sequence_std,
            "metadata_mean": metadata_mean,
            "metadata_std": metadata_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_low": target_low,
            "target_high": target_high,
            "target_columns": Q.TARGET_COLUMNS,
            "heldout_fold": heldout_fold,
            "epochs": FIXED_EPOCHS,
        },
        MODEL_DIR / f"heldout_fold{heldout_fold}.pt",
    )
    report = {
        "heldout_fold": heldout_fold,
        "train_wells": len(train_index),
        "valid_wells": len(valid_index),
        "coefficient_mse": float(np.mean(np.square(
            (prediction - target_raw[valid_index]) / target_std
        ))),
    }
    print("V127 MODEL", report, flush=True)
    return valid_index, prediction, history, report


crossfit_prediction = np.full_like(target_raw, np.nan, dtype=np.float32)
history_rows = []
model_reports = []
for heldout_fold in FOLDS:
    valid_index, prediction, history, report = _fit_predict(heldout_fold)
    crossfit_prediction[valid_index] = prediction
    history_rows.extend(history)
    model_reports.append(report)
if not np.isfinite(crossfit_prediction).all():
    raise RuntimeError("Incomplete V127 cross-fit coefficient predictions")
pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False)

coefficient_map = dict(zip(well_order, crossfit_prediction))
recipes = [
    (profile, dose, cap)
    for profile in Q.PROFILE_SLICES
    for dose in Q.DOSES
    for cap in Q.CAPS_FT
]


def _prediction_for_wells(wells, recipe, coefficient_override=None):
    profile, dose, cap = recipe
    source = coefficient_map if coefficient_override is None else coefficient_override
    coefficient_slice = Q.PROFILE_SLICES[profile]
    local_parts = []
    prediction_parts = []
    for well in sorted(wells):
        local = _well_frame(well)
        distance = local["md_since"].to_numpy(np.float64)
        coefficient = np.asarray(source[well], np.float64)[coefficient_slice]
        move = Q._basis(profile, distance) @ coefficient
        move = np.clip(float(dose) * move, -float(cap), float(cap))
        local_parts.append(local)
        prediction_parts.append(local[Q.BASE].to_numpy(np.float64) + move)
    return pd.concat(local_parts, ignore_index=True), np.concatenate(prediction_parts)


def _evaluate_folds(folds, recipe, endpoint, coefficient_override=None):
    wells = set(
        well for well, fold in zip(well_order, fold_by_index)
        if int(fold) in set(int(value) for value in folds)
    )
    matched, prediction = _prediction_for_wells(wells, recipe, coefficient_override)
    return R._metric(matched, prediction, endpoint), matched, prediction


def _base_metric(folds):
    local = frame[frame["fold"].isin(list(folds))].copy()
    return R._metric(local, local[Q.BASE].to_numpy(np.float64), Q.BASE)


scan_rows = []
selected_by_fold = {}
for heldout_fold in FOLDS:
    selection_folds = tuple(fold for fold in FOLDS if fold != heldout_fold)
    selection_base = _base_metric(selection_folds)
    per_fold_bases = {fold: _base_metric((fold,)) for fold in selection_folds}
    candidates = []
    for recipe in recipes:
        result, _, _ = _evaluate_folds(selection_folds, recipe, "sequence_coefficient")
        fold_gains = []
        for fold in selection_folds:
            fold_result, _, _ = _evaluate_folds((fold,), recipe, "sequence_coefficient")
            fold_gains.append(per_fold_bases[fold]["rmse"] - fold_result["rmse"])
        row = dict(result)
        row.update({
            "heldout_fold": heldout_fold,
            "selection_folds": "".join(str(value) for value in selection_folds),
            "profile": recipe[0],
            "dose": recipe[1],
            "cap_ft": recipe[2],
            "gain": float(selection_base["rmse"] - result["rmse"]),
            "far_gain": float(
                selection_base["far_gt2000_rmse"] - result["far_gt2000_rmse"]
            ),
            "p95_gain": float(selection_base["p95_well_rmse"] - result["p95_well_rmse"]),
            "min_fold_gain": float(min(fold_gains)),
        })
        candidates.append(row)
        scan_rows.append(row)
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["rmse", "p95_well_rmse"], kind="stable"
    )
    eligible = candidate_frame[
        (candidate_frame["gain"] >= 0.02)
        & (candidate_frame["far_gain"] >= 0.0)
        & (candidate_frame["p95_gain"] >= -0.10)
        & (candidate_frame["min_fold_gain"] >= 0.0)
    ]
    selected_row = (eligible if len(eligible) else candidate_frame).iloc[0]
    selected_by_fold[heldout_fold] = {
        "recipe": (
            str(selected_row["profile"]),
            float(selected_row["dose"]),
            float(selected_row["cap_ft"]),
        ),
        "selection_pass": bool(len(eligible)),
        "selection_gain": float(selected_row["gain"]),
        "selection_far_gain": float(selected_row["far_gain"]),
        "selection_p95_gain": float(selected_row["p95_gain"]),
        "selection_min_fold_gain": float(selected_row["min_fold_gain"]),
    }

scan = pd.DataFrame(scan_rows)
scan.to_csv(OUT_DIR / "nested_recipe_scan.csv", index=False)

summary_rows = []
nested_parts = []
nested_prediction_parts = []
rolled_prediction_parts = []
frozen_prediction_parts = []
fold_values = {}
for heldout_fold in FOLDS:
    wells = set(well for well, fold in zip(well_order, fold_by_index) if fold == heldout_fold)
    base = _base_metric((heldout_fold,))
    recipe = selected_by_fold[heldout_fold]["recipe"]
    candidate, matched, prediction = _evaluate_folds(
        (heldout_fold,), recipe, "nested_sequence_coefficient"
    )
    frozen, _, frozen_prediction = _evaluate_folds(
        (heldout_fold,), FROZEN_V126_RECIPE, "frozen_v126_recipe"
    )
    ordered = sorted(wells)
    rolled_values = np.roll(
        np.vstack([coefficient_map[well] for well in ordered]),
        max(1, len(ordered) // 7),
        axis=0,
    )
    rolled_map = dict(coefficient_map)
    for well, value in zip(ordered, rolled_values):
        rolled_map[well] = value
    rolled, _, rolled_prediction = _evaluate_folds(
        (heldout_fold,), recipe, "rolled_sequence_control", rolled_map
    )
    summary_rows.extend([
        dict(base, fold=heldout_fold),
        dict(frozen, fold=heldout_fold),
        dict(candidate, fold=heldout_fold),
        dict(rolled, fold=heldout_fold),
    ])
    nested_parts.append(matched)
    nested_prediction_parts.append(prediction)
    frozen_prediction_parts.append(frozen_prediction)
    rolled_prediction_parts.append(rolled_prediction)
    fold_values[str(heldout_fold)] = {
        "recipe": recipe,
        "selection_pass": selected_by_fold[heldout_fold]["selection_pass"],
        "gain": float(base["rmse"] - candidate["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
        "aligned_increment_vs_rolled": float(rolled["rmse"] - candidate["rmse"]),
    }

matched = pd.concat(nested_parts, ignore_index=True)
nested_prediction = np.concatenate(nested_prediction_parts)
frozen_prediction = np.concatenate(frozen_prediction_parts)
rolled_prediction = np.concatenate(rolled_prediction_parts)
pooled_base = R._metric(matched, matched[Q.BASE].to_numpy(np.float64), Q.BASE)
pooled_nested = R._metric(matched, nested_prediction, "nested_sequence_coefficient")
pooled_frozen = R._metric(matched, frozen_prediction, "frozen_v126_recipe")
pooled_rolled = R._metric(matched, rolled_prediction, "rolled_sequence_control")
summary_rows.extend([
    dict(pooled_base, fold=0),
    dict(pooled_frozen, fold=0),
    dict(pooled_nested, fold=0),
    dict(pooled_rolled, fold=0),
])
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

output = matched[["id", "well", "row_idx", "fold", Q.TRUTH, "md_since", Q.BASE]].copy()
output["nested_sequence_coefficient"] = nested_prediction.astype(np.float32)
output["frozen_v126_recipe"] = frozen_prediction.astype(np.float32)
output["rolled_sequence_control"] = rolled_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "crossfit_oof.parquet", index=False, compression="zstd")

coefficient_panel = pd.DataFrame(
    crossfit_prediction, columns=[f"pred_{name}" for name in Q.TARGET_COLUMNS]
)
coefficient_panel.insert(0, "well", well_order)
coefficient_panel.insert(1, "fold", fold_by_index)
for number, name in enumerate(Q.TARGET_COLUMNS):
    coefficient_panel[f"true_{name}"] = target_raw[:, number]
coefficient_panel.to_csv(OUT_DIR / "crossfit_coefficients.csv", index=False)

pooled_values = {
    "gain": float(pooled_base["rmse"] - pooled_nested["rmse"]),
    "far_gain": float(
        pooled_base["far_gt2000_rmse"] - pooled_nested["far_gt2000_rmse"]
    ),
    "p95_gain": float(pooled_base["p95_well_rmse"] - pooled_nested["p95_well_rmse"]),
    "aligned_increment_vs_rolled": float(pooled_rolled["rmse"] - pooled_nested["rmse"]),
    "frozen_v126_gain": float(pooled_base["rmse"] - pooled_frozen["rmse"]),
}
checks = {
    "nested_gain_ge_0p03": pooled_values["gain"] >= 0.03,
    "positive_folds_3of3": all(value["gain"] > 0.0 for value in fold_values.values()),
    "far_gain_nonnegative": pooled_values["far_gain"] >= 0.0,
    "p95_not_worse_by_gt_0p10": pooled_values["p95_gain"] >= -0.10,
    "aligned_beats_rolled_ge_0p02": pooled_values["aligned_increment_vs_rolled"] >= 0.02,
    "all_nested_selections_pass": all(
        value["selection_pass"] for value in fold_values.values()
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_Fold123_crossfit_sequence_coefficient_audit",
    "sample_unit": "one complete physical well",
    "model_training": "each heldout fold predicted by the other two folds",
    "recipe_selection": "each heldout fold recipe selected from the other two OOF folds",
    "fixed_architecture_source": str(V126_SOURCE),
    "fixed_epochs": FIXED_EPOCHS,
    "coverage": {"rows": len(matched), "wells": int(matched["well"].nunique())},
    "model_reports": model_reports,
    "nested_recipes": selected_by_fold,
    "fold_values": fold_values,
    "pooled": pooled_values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train the sequence coefficient head on all 464 wells and build deployment"
        if strict_pass else
        "FAIL -> V126 was selection overfit; close sequence coefficient postprocessing"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV127 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV127 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
