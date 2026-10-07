"""M342-F2: zero-training nested consumption of frozen F1 block choices.

Reconstruct the V187 endpoint inventory without rebuilding GR block features,
apply F1 crossfit choices, and crossfit a confidence threshold plus conservative
dose. The same recipe is applied to F1's in-block feature-shuffle choices.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
SOURCE = ROOT / "canonical_multiengine_block_oracle_v187.py"
F1_DIR = OOF / "main342_segment_pairwise_observability_f1"
CHOICES_PATH = F1_DIR / "block_choices.parquet"
OUT = OOF / "main342_segment_pairwise_consumption_f2"
OUT.mkdir(parents=True, exist_ok=True)

FREEZE_FT = 1000.0
BLOCK_FT = 128.0
DOSES = np.asarray([0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30], np.float64)
QUANTILES = np.asarray([0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95], np.float64)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_inventory():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = SOURCE.read_text(encoding="utf-8")
    marker = "standalone = pd.DataFrame"
    if marker not in source:
        raise RuntimeError("V187 inventory marker not found")
    name = "m342_f2_v187_inventory"
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    required = ("base", "matrix", "names", "base_index", "truth_all", "_metrics")
    missing = [key for key in required if key not in module.__dict__]
    if missing:
        raise RuntimeError("Incomplete V187 inventory: " + repr(missing))
    return module.__dict__


def _row_groups(base):
    row_group = np.full(len(base), -1, np.int32)
    group_fold, group_positions = [], []
    group = 0
    for _, positions in base.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        distance = base.iloc[positions]["md_since"].to_numpy(np.float64)
        active = distance > FREEZE_FT
        labels = np.floor((distance[active] - FREEZE_FT) / BLOCK_FT).astype(np.int32)
        active_positions = np.flatnonzero(active)
        for label in np.unique(labels):
            local = active_positions[labels == label]
            global_positions = positions[local]
            row_group[global_positions] = group
            group_positions.append(global_positions)
            group_fold.append(int(base.iloc[global_positions[0]]["fold"]))
            group += 1
    return row_group, np.asarray(group_fold, np.int8), group_positions


def _assemble(base_prediction, paths, choices, group_positions):
    output = base_prediction.copy()
    for group, positions in enumerate(group_positions):
        output[positions] = paths[int(choices[group]), positions]
    return output


def _metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, names = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    well_sse = np.bincount(codes, weights=error * error, minlength=len(names))
    well_rmse = np.sqrt(well_sse / counts)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)), "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


V = _load_inventory()
base = V["base"]
paths = V["matrix"]
names = V["names"]
base_index = int(V["base_index"])
truth = V["truth_all"]
if not CHOICES_PATH.is_file():
    raise FileNotFoundError(CHOICES_PATH)
choices = pd.read_parquet(CHOICES_PATH).sort_values("group", kind="stable").reset_index(drop=True)
required = {"group", "fold", "choice", "shuffle_choice", "margin"}
missing = sorted(required.difference(choices.columns))
if missing:
    raise RuntimeError("F1 choices missing columns: " + repr(missing))

row_group, group_fold, group_positions = _row_groups(base)
groups = len(group_positions)
if len(choices) != groups or not np.array_equal(choices["group"].to_numpy(), np.arange(groups)):
    raise RuntimeError(f"F1/V187 group mismatch: choices={len(choices)} rebuilt={groups}")
if not np.array_equal(choices["fold"].to_numpy(np.int8), group_fold):
    raise RuntimeError("F1/V187 group fold mismatch")

base_prediction = paths[base_index].astype(np.float64)
selected_prediction = _assemble(
    base_prediction, paths, choices["choice"].to_numpy(np.int16), group_positions
)
shuffle_prediction = _assemble(
    base_prediction, paths, choices["shuffle_choice"].to_numpy(np.int16), group_positions
)
margin = choices["margin"].to_numpy(np.float64)
fold_array = base["fold"].to_numpy(np.int8)
active = row_group >= 0


def _quadratics(candidate):
    move = np.asarray(candidate, np.float64) - base_prediction
    base_error = base_prediction - truth
    b = np.bincount(
        row_group[active], weights=(base_error * move)[active], minlength=groups
    ).astype(np.float64)
    c = np.bincount(
        row_group[active], weights=(move * move)[active], minlength=groups
    ).astype(np.float64)
    return move, b, c


selected_move, selected_b, selected_c = _quadratics(selected_prediction)
shuffle_move, _, _ = _quadratics(shuffle_prediction)
base_error = base_prediction - truth
nested = base_prediction.copy()
nested_shuffle = base_prediction.copy()
selection_rows = []
for held in range(1, 6):
    train_rows = fold_array != held
    valid_rows = fold_array == held
    train_groups = group_fold != held
    base_train_sse = float(np.dot(base_error[train_rows], base_error[train_rows]))
    scan = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(margin[train_groups], quantile))
        eligible_groups = train_groups & (margin >= threshold)
        for dose in DOSES:
            delta_sse = np.sum(
                2.0 * dose * selected_b[eligible_groups]
                + dose * dose * selected_c[eligible_groups]
            )
            scan.append((base_train_sse + float(delta_sse), quantile, threshold, dose))
    _, quantile, threshold, dose = min(scan, key=lambda row: row[0])
    eligible_rows = active & (margin[row_group.clip(min=0)] >= threshold)
    apply_rows = valid_rows & eligible_rows
    nested[apply_rows] += dose * selected_move[apply_rows]
    nested_shuffle[apply_rows] += dose * shuffle_move[apply_rows]
    selection_rows.append({
        "fold": held, "quantile": quantile, "threshold": threshold, "dose": dose,
        "activation_fraction": float(np.mean(eligible_rows[valid_rows])),
    })

# Truth-only attribution: whether any non-zero global dose exists at all.
denominator = float(np.dot(selected_move, selected_move))
analytic_dose = float(np.clip(-np.dot(base_error, selected_move) / max(denominator, 1e-12), 0.0, 1.0))
analytic_prediction = base_prediction + analytic_dose * selected_move
reverse_denominator = denominator
reverse_dose = float(np.clip(np.dot(base_error, selected_move) / max(reverse_denominator, 1e-12), 0.0, 1.0))
reverse_prediction = base_prediction - reverse_dose * selected_move

predictions = {
    "v168": base_prediction,
    "f1_full": selected_prediction,
    "nested_confidence_dose": nested,
    "nested_feature_shuffle": nested_shuffle,
    "analytic_aligned_oracle": analytic_prediction,
    "analytic_reversed_control": reverse_prediction,
}
summary = pd.DataFrame([
    _metric(base, prediction, name) for name, prediction in predictions.items()
]).sort_values("rmse", ignore_index=True)

fold_rows = []
for held in range(1, 6):
    mask = fold_array == held
    local = base.loc[mask].reset_index(drop=True)
    metrics = {name: _metric(local, prediction[mask], name) for name, prediction in predictions.items()}
    fold_rows.append({
        "fold": held,
        "dose": selection_rows[held - 1]["dose"],
        "activation_fraction": selection_rows[held - 1]["activation_fraction"],
        "base_rmse": metrics["v168"]["rmse"],
        "rmse": metrics["nested_confidence_dose"]["rmse"],
        "gain": metrics["v168"]["rmse"] - metrics["nested_confidence_dose"]["rmse"],
        "far_gain": metrics["v168"]["far_gt2000_rmse"] - metrics["nested_confidence_dose"]["far_gt2000_rmse"],
        "shuffle_rmse": metrics["nested_feature_shuffle"]["rmse"],
        "shuffle_loss": metrics["nested_feature_shuffle"]["rmse"] - metrics["nested_confidence_dose"]["rmse"],
        "p95_change": metrics["nested_confidence_dose"]["p95_well_rmse"] - metrics["v168"]["p95_well_rmse"],
    })
fold_frame = pd.DataFrame(fold_rows)
lookup = summary.set_index("endpoint")
gain = float(lookup.loc["v168", "rmse"] - lookup.loc["nested_confidence_dose", "rmse"])
shuffle_loss = float(lookup.loc["nested_feature_shuffle", "rmse"] - lookup.loc["nested_confidence_dose", "rmse"])
checks = {
    "nested_gain_ge_0p03": gain >= 0.03,
    "positive_folds_ge_4": int((fold_frame["gain"] > 0).sum()) >= 4,
    "minimum_fold_gain_ge_minus0p02": float(fold_frame["gain"].min()) >= -0.02,
    "shuffle_loses_ge_0p02": shuffle_loss >= 0.02,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change"].max()) <= 0.20,
    "analytic_aligned_dose_ge_0p01": analytic_dose >= 0.01,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_segment_pairwise_consumption_f2",
    "stage": "zero_training_nested_confidence_dose_of_f1_block_choices",
    "source": str(CHOICES_PATH),
    "candidate_endpoints": names,
    "coverage": {"rows": int(len(base)), "wells": int(base["well"].nunique()), "blocks": groups},
    "recipe": {"doses": DOSES, "quantiles": QUANTILES},
    "nested_selections": selection_rows,
    "analytic_aligned_dose_oracle": analytic_dose,
    "analytic_reversed_dose_control": reverse_dose,
    "summary": summary.to_dict("records"),
    "folds": fold_rows,
    "values": {"nested_gain": gain, "shuffle_loss": shuffle_loss},
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze F2 recipe and test continuity-DP consumption"
        if strict_pass else
        "FAIL -> F1 choices have no safely consumable residual signal; close blockwise selection"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
fold_frame.to_csv(OUT / "folds.csv", index=False)
pd.DataFrame(selection_rows).to_csv(OUT / "nested_selections.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-F2 SELECTIONS")
print(pd.DataFrame(selection_rows).to_string(index=False), flush=True)
print("\nM342-F2 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-F2 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nM342-F2 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
