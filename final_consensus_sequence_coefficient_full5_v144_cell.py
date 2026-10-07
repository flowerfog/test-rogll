"""V144: complete five-fold cross-fit of the V127 sequence coefficient head.

Folds1-3 reuse the frozen V115 consensus base. Folds4-5 are reconstructed from
the honest V96 prior123 panels and leave-one-fold-out rankers. Residual labels
are then generated for all 773 wells, while each held-out coefficient endpoint
is trained and calibrated using only the other four folds.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "final_consensus_sequence_coefficient_full5_v144"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V126_SOURCE = ROOT / "final_consensus_sequence_coefficient_v126_cell.py"
V96_SOURCE = ROOT / "rawdata_lattice_v76_full773_training_v96_cell.py"
V96_DIR = OOF_ROOT / "rawdata_lattice_v76_full773_training_v96"
V96_COARSE = V96_DIR / "fivefold_coarse_oof.parquet"
V96_PRIOR = V96_DIR / "prior123.joblib"
V96_PANEL_DIR = V96_DIR / "prior123_panels"
V96_MODEL_DIR = V96_DIR / "coarse_models"
FOLDS = (1, 2, 3, 4, 5)
FIXED_EPOCHS = 3
SEED = 20260801
FROZEN_V126_RECIPE = ("tail_affine", 1.0, 4.0)
TOPK = 33
CONSENSUS_FREEZE_FT = 1000.0
CONSENSUS_RAMP_FT = 250.0
CONSENSUS_CAP_FT = 6.0


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
    module = types.ModuleType("final_consensus_v144_v126_definitions")
    module.__file__ = str(V126_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V126_SOURCE), "exec"), module.__dict__)
    return module


def _bootstrap_v96():
    if not V96_SOURCE.is_file():
        raise FileNotFoundError(V96_SOURCE)
    source = V96_SOURCE.read_text(encoding="utf-8")
    marker = "for path in (V4_SOURCE, V66_PATH, V70_PATH, V76_BUNDLE, V76_OOF_WEIGHTS):"
    if marker not in source:
        raise RuntimeError(f"V96 definition marker missing from {V96_SOURCE}")
    module = types.ModuleType("final_consensus_v144_v96_definitions")
    module.__file__ = str(V96_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V96_SOURCE), "exec"), module.__dict__)
    return module


Q = _bootstrap_v126()
Q._seed_everything(SEED)
R = Q._bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

for required in (Q.BASE_PATH, V96_COARSE, V96_PRIOR):
    if not required.is_file():
        raise FileNotFoundError(required)
for required in (V96_PANEL_DIR, V96_MODEL_DIR):
    if not required.is_dir():
        raise FileNotFoundError(required)

W = _bootstrap_v96()
Q4 = W._bootstrap_v4()
V4, B4 = Q4.V, Q4.B
prior123 = joblib.load(V96_PRIOR)


def _deduplicated_order(paths, scores):
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    retained, seen = [], set()
    for index in order:
        key = np.ascontiguousarray(paths[index]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        retained.append(int(index))
    return np.asarray(retained, np.int32)


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


old_base = pd.read_parquet(Q.BASE_PATH)
old_base["well"] = old_base["well"].astype(str)
coarse_all = pd.read_parquet(V96_COARSE).sort_values(
    ["fold", "well", "row_idx"], kind="stable"
).reset_index(drop=True)
coarse_all["well"] = coarse_all["well"].astype(str)
if len(coarse_all) != 3_783_989 or coarse_all["well"].nunique() != 773:
    raise RuntimeError("V96 fivefold coarse OOF is incomplete")

feature_table = pd.DataFrame([
    W._feature_row(local)
    for _, local in coarse_all.groupby(["fold", "well"], sort=False)
]).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
x_weight, weight_feature_names = W._augment(feature_table)
bundle = joblib.load(W.V76_BUNDLE)
if list(bundle["feature_names"]) != list(weight_feature_names):
    raise RuntimeError("V76 deployment feature schema mismatch")
baseline_weights = np.clip(
    float(bundle["hist_fraction"]) * bundle["hist_model"].predict(x_weight)
    + float(bundle["ridge_fraction"]) * bundle["ridge_model"].predict(x_weight),
    float(bundle["min_weight"]), float(bundle["max_weight"]),
)
weight_map = dict(zip(feature_table["well"].astype(str), baseline_weights))

fold45_parts, build_reports = [], []
for heldout_fold in (4, 5):
    model_path = V96_MODEL_DIR / f"ranker_leave_fold{heldout_fold}_out.txt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    ranker = lgb.Booster(model_file=str(model_path))
    fold_frame = coarse_all[coarse_all["fold"].eq(heldout_fold)].copy()
    fold_frame["dose"] = fold_frame["well"].map(weight_map)
    if fold_frame["dose"].isna().any():
        raise RuntimeError(f"Fold{heldout_fold}: incomplete V76 dose")
    fold_frame["exact_v76"] = (
        fold_frame[Q.PARENT].to_numpy(np.float64)
        + fold_frame["dose"].to_numpy(np.float64)
        * (fold_frame[Q.COARSE].to_numpy(np.float64) - fold_frame[Q.PARENT].to_numpy(np.float64))
    )
    candidate_parts = []
    for number, (well, local) in enumerate(fold_frame.groupby("well", sort=True), 1):
        local = local.sort_values("row_idx", kind="stable")
        candidate = local["exact_v76"].to_numpy(np.float64).copy()
        status, reason = "fallback", ""
        try:
            panel_path = V96_PANEL_DIR / f"{well}.npz"
            if not panel_path.is_file():
                raise FileNotFoundError(panel_path)
            with np.load(panel_path, allow_pickle=False) as data:
                evidence = data["features"].astype(np.float32, copy=False)
            record = V4["_load_record"](str(well))
            paths, source_mask, arm_rank, arm_score_z, segments, _ = Q4._candidate_union(
                record, prior123
            )
            if len(paths) != len(evidence):
                raise RuntimeError(f"candidate count mismatch {len(paths)}/{len(evidence)}")
            order = _deduplicated_order(paths, ranker.predict(evidence))
            if len(order) < TOPK:
                raise RuntimeError(f"only {len(order)} unique candidates")
            eval_rows = record["eval_rows"]
            if len(local) != len(eval_rows):
                raise RuntimeError(f"row mismatch {len(local)}/{len(eval_rows)}")
            trajectories = np.vstack([
                B4["_path_prediction"](record, prior123, paths[index], segments)[eval_rows]
                for index in order[:TOPK]
            ]).astype(np.float64)
            center = np.mean(trajectories, axis=0)
            distance = local["md_since"].to_numpy(np.float64)
            taper = _smoothstep((distance - CONSENSUS_FREEZE_FT) / CONSENSUS_RAMP_FT)
            move = np.clip(
                float(local["dose"].iloc[0])
                * (center - local[Q.COARSE].to_numpy(np.float64)),
                -CONSENSUS_CAP_FT, CONSENSUS_CAP_FT,
            )
            candidate += taper * move
            status = "built"
        except Exception as exc:
            reason = repr(exc)
        candidate_parts.append(candidate)
        build_reports.append({
            "fold": heldout_fold, "well": str(well), "status": status, "reason": reason,
        })
        if number % 20 == 0 or number == fold_frame["well"].nunique():
            failures = sum(
                row["status"] == "fallback" and row["fold"] == heldout_fold
                for row in build_reports
            )
            print(
                f"V144 consensus Fold{heldout_fold} {number}/{fold_frame['well'].nunique()} "
                f"fallbacks={failures}", flush=True,
            )
    fold_frame[Q.BASE] = np.concatenate(candidate_parts).astype(np.float32)
    fold45_parts.append(fold_frame)

build_report = pd.DataFrame(build_reports)
build_report.to_csv(OUT_DIR / "fold45_consensus_build_report.csv", index=False)
fallback_counts = build_report.groupby("fold")["status"].apply(
    lambda value: int((value == "fallback").sum())
)
if any(fallback_counts.get(fold, 0) > 5 for fold in (4, 5)):
    raise RuntimeError(f"Too many Fold4/5 consensus fallbacks: {fallback_counts.to_dict()}")

required_base = {
    "id", "well", "row_idx", "fold", Q.TRUTH, "md_since", Q.BASE, "exact_v76",
    Q.PARENT, Q.COARSE,
}
old_missing = sorted({Q.PARENT, Q.COARSE}.difference(old_base.columns))
if old_missing:
    lineage123 = coarse_all[coarse_all["fold"].isin((1, 2, 3))][
        ["id", Q.PARENT, Q.COARSE]
    ]
    old_base = old_base.merge(lineage123, on="id", how="left", validate="one_to_one")
frame = pd.concat([
    old_base[list(required_base)],
    *[local[list(required_base)] for local in fold45_parts],
], ignore_index=True)
frame["well"] = frame["well"].astype(str)
if frame["id"].duplicated().any() or len(frame) != 3_783_989 or frame["well"].nunique() != 773:
    raise RuntimeError(
        f"Invalid fivefold base rows={len(frame)} wells={frame['well'].nunique()}"
    )
if frame[list(required_base)].isna().any().any():
    raise RuntimeError("Fivefold consensus base contains missing values")
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
well_positions = {
    str(well): np.asarray(positions, np.int64)
    for well, positions in frame.groupby("well", sort=False).indices.items()
}


def _well_frame(well):
    return frame.iloc[well_positions[str(well)]]

well_order = sorted(frame["well"].unique())


def _fit_projection(target, design):
    design = np.asarray(design, np.float64)
    if design.ndim == 1:
        design = design[:, None]
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    keep = np.isfinite(scale) & (scale > 1e-10)
    coefficient = np.zeros(design.shape[1], np.float64)
    if keep.any():
        normalized = design[:, keep] / scale[keep]
        scaled, *_ = np.linalg.lstsq(normalized, target, rcond=1e-10)
        coefficient[keep] = scaled / scale[keep]
    return coefficient


def _target_coefficients(local):
    target = local[Q.TRUTH].to_numpy(np.float64) - local[Q.BASE].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    x = (distance - float(np.mean(distance))) / 1000.0
    taper = _smoothstep((distance - Q.FREEZE_FT) / Q.RAMP_FT)
    tail_datum = _fit_projection(target, taper)
    tail_affine = _fit_projection(target, np.column_stack([taper, taper * x]))
    hinge1 = taper * np.clip((distance - 1250.0) / 1000.0, 0.0, None)
    hinge2 = taper * np.clip((distance - 2250.0) / 1000.0, 0.0, None)
    tail_piecewise = _fit_projection(
        target, np.column_stack([taper, hinge1, hinge2])
    )
    return np.concatenate([tail_datum, tail_affine, tail_piecewise]).astype(np.float32)

sequence_rows = []
metadata_rows = []
target_rows = []
fold_rows = []
for number, well in enumerate(well_order, 1):
    local = _well_frame(well)
    local_fold = int(local["fold"].iloc[0])
    sequence, metadata = Q._sequence_features(local, Q._load_record_relaxed(V, well))
    sequence_rows.append(sequence)
    metadata_rows.append(metadata)
    target_rows.append(_target_coefficients(local))
    fold_rows.append(local_fold)
    if number % 100 == 0 or number == len(well_order):
        print(f"V144 sequences {number}/{len(well_order)}", flush=True)

sequence_raw = np.stack(sequence_rows).astype(np.float32)
metadata_raw = np.stack(metadata_rows).astype(np.float32)
target_raw = np.stack(target_rows).astype(np.float32)
fold_by_index = np.asarray(fold_rows, np.int8)
if len(well_order) != 773 or set(fold_by_index.tolist()) != set(FOLDS):
    raise RuntimeError(f"Unexpected V144 coverage wells={len(well_order)} folds={set(fold_by_index)}")

target_parity_max_abs = np.nan
if Q.TARGET_PATH.is_file():
    historical_targets = pd.read_csv(Q.TARGET_PATH, dtype={"well": str}).set_index("well")
    parity_wells = [well for well in well_order if well in historical_targets.index]
    parity_index = np.asarray([well_order.index(well) for well in parity_wells], np.int64)
    historical_values = historical_targets.loc[parity_wells, list(Q.TARGET_COLUMNS)].to_numpy(np.float32)
    target_parity_max_abs = float(np.max(np.abs(target_raw[parity_index] - historical_values)))
    if len(parity_wells) != 464 or target_parity_max_abs > 1e-3:
        raise RuntimeError(
            f"V123 target parity failed wells={len(parity_wells)} gap={target_parity_max_abs}"
        )
    print(
        "V144 TARGET PARITY",
        {"wells": len(parity_wells), "maximum_abs_difference": target_parity_max_abs},
        flush=True,
    )


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
    print("V144 MODEL", report, flush=True)
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
    raise RuntimeError("Incomplete V144 cross-fit coefficient predictions")
pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False)

coefficient_map = dict(zip(well_order, crossfit_prediction))


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


selected_by_fold = {
    heldout_fold: {
        "recipe": FROZEN_V126_RECIPE,
        "selection_pass": True,
        "selection_source": "frozen_before_V144",
    }
    for heldout_fold in FOLDS
}

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

output = matched[[
    "id", "well", "row_idx", "fold", Q.TRUTH, "md_since",
    Q.PARENT, Q.COARSE, "exact_v76", Q.BASE,
]].copy()
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
    "positive_folds_5of5": all(value["gain"] > 0.0 for value in fold_values.values()),
    "far_gain_nonnegative": pooled_values["far_gain"] >= 0.0,
    "p95_not_worse_by_gt_0p10": pooled_values["p95_gain"] >= -0.10,
    "aligned_beats_rolled_ge_0p02": pooled_values["aligned_increment_vs_rolled"] >= 0.02,
    "frozen_recipe_used_5of5": all(
        tuple(value["recipe"]) == tuple(FROZEN_V126_RECIPE)
        for value in fold_values.values()
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_773well_fivefold_consensus_and_sequence_coefficient_crossfit",
    "sample_unit": "one complete physical well",
    "selection_or_tuning_count": 0,
    "model_training": "each heldout fold predicted by the other four folds",
    "recipe_selection": "none; V126 tail_affine dose1 cap4 frozen for all folds",
    "consensus_protocol": {
        "fold123": "existing V115 prior45 and Fold45 ranker",
        "fold45": "V96 prior123 and leave-one-fold-out rankers",
        "topk": TOPK,
        "fallback_counts": fallback_counts.to_dict(),
    },
    "fixed_architecture_source": str(V126_SOURCE),
    "fixed_epochs": FIXED_EPOCHS,
    "v123_target_parity_max_abs": target_parity_max_abs,
    "coverage": {"rows": len(matched), "wells": int(matched["well"].nunique())},
    "model_reports": model_reports,
    "nested_recipes": selected_by_fold,
    "fold_values": fold_values,
    "pooled": pooled_values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> retain V144 as the complete local-CV reference"
        if strict_pass else
        "FAIL -> sequence coefficient does not transfer across all five canonical folds"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV144 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV144 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
