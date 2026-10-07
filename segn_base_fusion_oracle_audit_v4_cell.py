"""Final zero-training oracle audit for the SEGN/base fusion branch.

Rebuild only the selected v3 endpoint, then measure:
* a tune-fitted continuous global weight on the 107-well confirmation subset;
* hard and continuous per-well oracle ceilings;
* label-free disagreement-feature association with required per-well benefit.

This is an investment gate, not a deployable model.  Fold 1 is not loaded.
"""

from pathlib import Path
import json
import math
import sys
import time
import types

import numpy as np
import pandas as pd
import torch


EXPERIMENT = "segn_base_fusion_oracle_audit_v4"
ROOT = Path("/root/autodl-tmp")
V3_SOURCE_CANDIDATES = (
    ROOT / "segn_foldclean_base_reanchor_audit_v3_cell.py",
    Path.cwd() / "segn_foldclean_base_reanchor_audit_v3_cell.py",
)
V3_SOURCE = next((path for path in V3_SOURCE_CANDIDATES if path.exists()), None)
if V3_SOURCE is None:
    raise FileNotFoundError(
        "Upload segn_foldclean_base_reanchor_audit_v3_cell.py to /root/autodl-tmp"
    )

source = V3_SOURCE.read_text(encoding="utf-8")
marker = "started = time.time()"
if marker not in source:
    raise RuntimeError(f"Cannot locate execution marker in {V3_SOURCE}")
module_name = "_segn_base_reanchor_runtime_v4"
module = types.ModuleType(module_name)
module.__file__ = str(V3_SOURCE)
sys.modules[module_name] = module
R = module.__dict__
exec(compile(source.split(marker, 1)[0], str(V3_SOURCE), "exec"), R)

V = R["V"]
OOF_ROOT = R["OOF_ROOT"]
V3_ROOT = OOF_ROOT / "segn_foldclean_base_reanchor_audit_v3"
V3_VERDICT = V3_ROOT / "verdict.json"
CHECKPOINT_PATH = R["CHECKPOINT_PATH"]
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
TUNE_WELLS = R["TUNE_WELLS"]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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


def _optimal_weight(truth, base, model):
    truth = np.asarray(truth, np.float64)
    base = np.asarray(base, np.float64)
    model = np.asarray(model, np.float64)
    direction = model - base
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0
    weight = -float(np.dot(base - truth, direction)) / denominator
    return float(np.clip(weight, 0.0, 1.0))


def _with_prediction(rows, weights):
    output = []
    scalar = np.ndim(weights) == 0
    for index, row in enumerate(rows):
        weight = float(weights) if scalar else float(weights[index])
        output.append(
            {
                **row,
                "prediction": row["base"] + weight * (row["model"] - row["base"]),
            }
        )
    return output


def _flatten(rows, key):
    return np.concatenate([row[key] for row in rows]).astype(np.float64)


def _feature_row(row, oracle_weight, benefit_mse):
    base = row["base"].astype(np.float64)
    model = row["model"].astype(np.float64)
    move = model - base
    distance = row["distance"].astype(np.float64)
    return {
        "well": row["well"],
        "rows": len(base),
        "oracle_weight": oracle_weight,
        "benefit_mse": benefit_mse,
        "move_mean": float(np.mean(move)),
        "move_abs_mean": float(np.mean(np.abs(move))),
        "move_rms": float(np.sqrt(np.mean(move * move))),
        "move_std": float(np.std(move)),
        "move_p95_abs": float(np.quantile(np.abs(move), 0.95)),
        "move_end": float(move[-1]),
        "move_range": float(np.ptp(move)),
        "base_range": float(np.ptp(base)),
        "model_range": float(np.ptp(model)),
        "base_roughness": float(np.mean(np.abs(np.diff(base)))) if len(base) > 1 else 0.0,
        "model_roughness": float(np.mean(np.abs(np.diff(model)))) if len(model) > 1 else 0.0,
        "maximum_distance": float(np.max(distance)),
    }


started = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
if not V3_VERDICT.exists():
    raise FileNotFoundError(V3_VERDICT)
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

v3_verdict = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
selected = v3_verdict["selected"]
base_name = str(selected["base"])
anchor_dose = float(selected["anchor_dose"])
offset_gain = float(selected["offset_gain"])
fixed_weight = float(selected["model_weight"])

device = torch.device("cuda")
folds, fold_path = V["_load_folds"]()
valid_wells = folds.loc[folds["fold"].eq(V["VALID_FOLD"]), "well"].tolist()
records = V["_load_records"](valid_wells, "fold2_fusion_oracle")
if len(records) != 155:
    raise RuntimeError(f"Expected 155 records, got {len(records)}")
paths, base_names, alignment = R["_load_base_paths"](records)
if base_name not in base_names:
    raise RuntimeError(f"Selected base {base_name} unavailable: {base_names}")

model = V["ChunkedCostMapNet"]().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()
rows = R["_build_rows"](
    model,
    records,
    paths,
    base_name,
    device,
    anchor_dose,
    offset_gain,
)
tune_rows = rows[:TUNE_WELLS]
confirmation_rows = rows[TUNE_WELLS:]

tune_weight = _optimal_weight(
    _flatten(tune_rows, "truth"),
    _flatten(tune_rows, "base"),
    _flatten(tune_rows, "model"),
)
base_confirmation = R["_metrics"](_with_prediction(confirmation_rows, 0.0))
fixed_confirmation = R["_metrics"](_with_prediction(confirmation_rows, fixed_weight))
tuned_confirmation = R["_metrics"](_with_prediction(confirmation_rows, tune_weight))

full_global_oracle_weight = _optimal_weight(
    _flatten(rows, "truth"), _flatten(rows, "base"), _flatten(rows, "model")
)
full_global_oracle = R["_metrics"](_with_prediction(rows, full_global_oracle_weight))

oracle_weights = []
benefits = []
hard_weights = []
for row in rows:
    weight = _optimal_weight(row["truth"], row["base"], row["model"])
    base_mse = float(np.mean((row["base"] - row["truth"]) ** 2))
    model_mse = float(np.mean((row["model"] - row["truth"]) ** 2))
    oracle_weights.append(weight)
    benefits.append(base_mse - model_mse)
    hard_weights.append(float(model_mse < base_mse))

perwell_oracle = R["_metrics"](_with_prediction(rows, oracle_weights))
hard_oracle = R["_metrics"](_with_prediction(rows, hard_weights))
perwell_oracle_confirmation = R["_metrics"](
    _with_prediction(confirmation_rows, oracle_weights[TUNE_WELLS:])
)
hard_oracle_confirmation = R["_metrics"](
    _with_prediction(confirmation_rows, hard_weights[TUNE_WELLS:])
)
feature_frame = pd.DataFrame(
    [
        _feature_row(row, oracle_weights[index], benefits[index])
        for index, row in enumerate(rows)
    ]
)
feature_frame.to_csv(OUT_DIR / "perwell_oracle_and_features.csv", index=False)

feature_columns = [
    column
    for column in feature_frame.columns
    if column not in {"well", "oracle_weight", "benefit_mse"}
]
associations = []
for feature in feature_columns:
    associations.append(
        {
            "feature": feature,
            "spearman_oracle_weight": float(
                feature_frame[[feature, "oracle_weight"]].corr(method="spearman").iloc[0, 1]
            ),
            "spearman_benefit_mse": float(
                feature_frame[[feature, "benefit_mse"]].corr(method="spearman").iloc[0, 1]
            ),
        }
    )
association_frame = pd.DataFrame(associations)
association_frame["max_abs_association"] = association_frame[
    ["spearman_oracle_weight", "spearman_benefit_mse"]
].abs().max(axis=1)
association_frame = association_frame.sort_values("max_abs_association", ascending=False)
association_frame.to_csv(OUT_DIR / "feature_associations.csv", index=False)
print("SEGN FUSION FEATURE ASSOCIATIONS")
print(association_frame.to_string(index=False), flush=True)

oracle_gain = base_confirmation["rmse"] - perwell_oracle_confirmation["rmse"]
best_association = float(association_frame["max_abs_association"].max())
worth_training_gate = bool(oracle_gain >= 0.50 and best_association >= 0.20)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_selected_endpoint_fusion_oracle_and_observability",
    "source": str(V3_SOURCE),
    "checkpoint": str(CHECKPOINT_PATH),
    "fold_source": str(fold_path),
    "fold1_touched": False,
    "alignment": alignment,
    "selected_endpoint": {
        "base": base_name,
        "anchor_dose": anchor_dose,
        "offset_gain": offset_gain,
        "v3_fixed_weight": fixed_weight,
    },
    "tune_fitted_global_weight": tune_weight,
    "base_confirmation": base_confirmation,
    "fixed_confirmation": fixed_confirmation,
    "tuned_confirmation": tuned_confirmation,
    "full_global_oracle_weight": full_global_oracle_weight,
    "full_global_oracle": full_global_oracle,
    "perwell_oracle": perwell_oracle,
    "hard_perwell_oracle": hard_oracle,
    "perwell_oracle_confirmation": perwell_oracle_confirmation,
    "hard_perwell_oracle_confirmation": hard_oracle_confirmation,
    "perwell_oracle_weight_summary": {
        "mean": float(np.mean(oracle_weights)),
        "median": float(np.median(oracle_weights)),
        "p25": float(np.quantile(oracle_weights, 0.25)),
        "p75": float(np.quantile(oracle_weights, 0.75)),
        "zero_fraction": float(np.mean(np.asarray(oracle_weights) <= 1e-8)),
        "one_fraction": float(np.mean(np.asarray(oracle_weights) >= 1.0 - 1e-8)),
    },
    "oracle_gain_vs_confirmation_base": oracle_gain,
    "best_label_free_feature_association": best_association,
    "gates": {
        "perwell_oracle_gain_ge": 0.50,
        "label_free_feature_abs_spearman_ge": 0.20,
    },
    "worth_training_gate": worth_training_gate,
    "strict_pass": worth_training_gate,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> build nested continuous trust gate"
        if worth_training_gate
        else "FAIL -> close SEGN/base fusion; retain only the diagnostic encoder"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("SEGN FUSION ORACLE FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
