"""Local cost-profile sufficiency audit on raw, same-well evidence.

This is a decision experiment, not a deployable model proposal.  It separates:

1. state-axis representation coverage;
2. legal GR/typewell profile candidate coverage;
3. the ability of a GBDT to decode a perfectly centred profile;
4. transfer of the real legal profile, with flat and rolled controls.

Protocol: train folds 3/4/5, evaluate fold 2, leave fold 1 untouched.
"""

from __future__ import annotations

from pathlib import Path
import ast
import gc
import json
import shutil
import sys
import time
import types

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise RuntimeError("lightgbm is required for this audit") from exc


def _bootstrap_costmap():
    candidates = (
        Path("/root/autodl-tmp/inverted_costmap_transformer_fold2_v1_cell.py"),
        Path.cwd() / "inverted_costmap_transformer_fold2_v1_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload inverted_costmap_transformer_fold2_v1_cell.py, "
            "inverted_transformer_fullwell_fold2_v0_cell.py, and "
            "native_resolution_dualref_capacity_v0_cell.py to /root/autodl-tmp."
        )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    module_name = "costprofile_oracle_source_v0"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module.__dict__, source_path


V, SOURCE_PATH = _bootstrap_costmap()
OOF_ROOT = V["OOF_ROOT"]
PACK_DIR = V["PACK_DIR"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float32)
N_STATES = len(STATE_OFFSETS)
N_CONDITION = V["N_CONDITION"]
BASE_VARIABLE_NAMES = list(V["BASE_VARIABLE_NAMES"])
_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_pack = V["_load_pack"]
_metrics = V["_metrics"]

EXPERIMENT = "local_costprofile_feature_oracle_fold2_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260724
TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
UNTOUCHED_FOLD = 1
PROFILE_POINTS = 65
PROFILE_INDICES = np.unique(
    np.rint(np.linspace(0, N_STATES - 1, PROFILE_POINTS)).astype(np.int64)
)
TOPK_BUDGETS = (1, 4, 8, 16, 32, 64, 128, N_STATES)
POSITIVE_SIGMA_FT = 4.0


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _softmax(logits, temperature):
    scaled = logits.astype(np.float64) / float(temperature)
    scaled -= np.max(scaled, axis=0, keepdims=True)
    probability = np.exp(np.clip(scaled, -60.0, 0.0))
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-12)
    return probability


def _profile_statistics(profile, prefix):
    columns = []
    names = []
    order = np.argsort(profile, axis=0)
    best = order[-1]
    second = order[-2]
    time_index = np.arange(profile.shape[1])
    columns.extend(
        [
            STATE_OFFSETS[best],
            profile[best, time_index],
            profile[second, time_index],
            profile[best, time_index] - profile[second, time_index],
        ]
    )
    names.extend(
        [
            f"{prefix}_argmax_offset",
            f"{prefix}_best_logit",
            f"{prefix}_second_logit",
            f"{prefix}_top_gap",
        ]
    )
    for temperature in (0.5, 1.0, 2.0, 4.0):
        probability = _softmax(profile, temperature)
        mean = np.sum(probability * STATE_OFFSETS[:, None], axis=0)
        variance = np.sum(
            probability * (STATE_OFFSETS[:, None] - mean[None]) ** 2, axis=0
        )
        entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=0)
        columns.extend([mean, np.sqrt(np.maximum(variance, 0.0)), entropy])
        tag = str(temperature).replace(".", "p")
        names.extend(
            [
                f"{prefix}_mean_t{tag}",
                f"{prefix}_std_t{tag}",
                f"{prefix}_entropy_t{tag}",
            ]
        )
    return np.column_stack(columns).astype(np.float32), names


def _profiles(pack, mode):
    if mode == "legal":
        typewell = pack.emissions[0].astype(np.float32)
        heel = pack.emissions[1].astype(np.float32)
    elif mode == "truth_positive":
        difference = STATE_OFFSETS[:, None] - pack.target[None]
        profile = np.clip(-0.5 * (difference / POSITIVE_SIGMA_FT) ** 2, -40.0, 0.0)
        typewell = profile.astype(np.float32)
        heel = profile.astype(np.float32)
    else:
        raise ValueError(mode)
    combined = np.logaddexp(typewell, heel).astype(np.float32) - np.float32(np.log(2.0))
    return typewell, heel, combined


def _feature_matrix(pack, mode, profile_control="full"):
    typewell, heel, combined = _profiles(pack, mode)
    blocks = [pack.condition.T.astype(np.float32)]
    names = list(BASE_VARIABLE_NAMES)
    for profile, prefix in (
        (typewell, "typewell"),
        (heel, "heel"),
        (combined, "combined"),
    ):
        sampled = profile[PROFILE_INDICES].T.astype(np.float32) / 10.0
        sampled_names = [
            f"{prefix}_profile_{int(round(STATE_OFFSETS[index])):+d}ft"
            for index in PROFILE_INDICES
        ]
        statistics, statistic_names = _profile_statistics(profile, prefix)
        blocks.extend([sampled, statistics])
        names.extend(sampled_names + statistic_names)
    matrix = np.concatenate(blocks, axis=1)
    if profile_control == "flat":
        matrix[:, N_CONDITION:] = 0.0
    elif profile_control == "rolled":
        matrix[:, N_CONDITION:] = np.roll(
            matrix[:, N_CONDITION:], max(1, len(matrix) // 3), axis=0
        )
    elif profile_control != "full":
        raise ValueError(profile_control)
    return matrix, names


def _grid_row_weights(pack):
    grid = pack.grid_md.astype(np.float64)
    md = pack.eval_md.astype(np.float64)
    right = np.searchsorted(grid, md, side="left")
    right = np.clip(right, 1, len(grid) - 1)
    left = right - 1
    choose_right = np.abs(grid[right] - md) < np.abs(md - grid[left])
    nearest = np.where(choose_right, right, left)
    return np.bincount(nearest, minlength=len(grid)).astype(np.float32)


def _training_matrix(packs, mode, include_profile):
    x_values, y_values, weights, groups = [], [], [], []
    feature_names = None
    for number, pack in enumerate(packs, 1):
        matrix, names = _feature_matrix(pack, mode)
        if not include_profile:
            matrix = matrix[:, :N_CONDITION]
            names = names[:N_CONDITION]
        row_weight = _grid_row_weights(pack)
        mask = pack.supervised & (row_weight > 0)
        x_values.append(matrix[mask])
        y_values.append(pack.target[mask].astype(np.float32))
        weights.append(row_weight[mask])
        groups.append(np.full(mask.sum(), pack.well, dtype="U16"))
        feature_names = names
        if number % 100 == 0 or number == len(packs):
            print(f"build {mode} matrix {number}/{len(packs)}", flush=True)
    return (
        np.concatenate(x_values),
        np.concatenate(y_values),
        np.concatenate(weights),
        np.concatenate(groups),
        feature_names,
    )


def _model():
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=1600,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=160,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.15,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def _validation_grid(packs, mode, include_profile, profile_control="full"):
    x_values, y_values, weights = [], [], []
    for pack in packs:
        matrix, _ = _feature_matrix(pack, mode, profile_control=profile_control)
        if not include_profile:
            matrix = matrix[:, :N_CONDITION]
        row_weight = _grid_row_weights(pack)
        mask = pack.supervised & (row_weight > 0)
        x_values.append(matrix[mask])
        y_values.append(pack.target[mask])
        weights.append(row_weight[mask])
    return np.concatenate(x_values), np.concatenate(y_values), np.concatenate(weights)


def _fit_arm(name, train_packs, valid_packs, mode, include_profile):
    x_train, y_train, weight_train, _, feature_names = _training_matrix(
        train_packs, mode, include_profile
    )
    x_valid, y_valid, weight_valid = _validation_grid(
        valid_packs, mode, include_profile
    )
    print(
        f"FIT {name}: train={x_train.shape}, valid={x_valid.shape}, "
        f"features={len(feature_names)}",
        flush=True,
    )
    model = _model()
    model.fit(
        x_train,
        y_train,
        sample_weight=weight_train,
        eval_set=[(x_valid, y_valid)],
        eval_sample_weight=[weight_valid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )
    model.booster_.save_model(str(MODEL_DIR / f"{name}.txt"))
    del x_train, y_train, weight_train, x_valid, y_valid, weight_valid
    gc.collect()
    return model, feature_names


def _interpolate_grid(pack, grid_offset):
    support = pack.grid_md > pack.anchor_md
    support_md = np.r_[pack.anchor_md, pack.grid_md[support]]
    support_offset = np.r_[0.0, np.asarray(grid_offset)[support]]
    return pack.anchor_tvt + np.interp(pack.eval_md, support_md, support_offset)


def _score_predictions(packs, grid_predictions):
    truths, predictions, distances, wells = [], [], [], []
    for pack, grid_prediction in zip(packs, grid_predictions):
        truths.append(pack.truth.astype(np.float64))
        predictions.append(_interpolate_grid(pack, grid_prediction))
        distances.append(pack.eval_distance.astype(np.float64))
        wells.append(np.full(pack.eval_rows, pack.well, dtype="U16"))
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    distance = np.concatenate(distances)
    well = np.concatenate(wells)
    metrics, bywell, _ = _metrics(truth, prediction, distance, well)
    return metrics, bywell


def _predict_arm(model, packs, mode, include_profile, profile_control="full"):
    output = []
    for number, pack in enumerate(packs, 1):
        matrix, _ = _feature_matrix(pack, mode, profile_control=profile_control)
        if not include_profile:
            matrix = matrix[:, :N_CONDITION]
        prediction = model.predict(
            matrix, num_iteration=getattr(model, "best_iteration_", None)
        )
        prediction[: pack.anchor_grid + 1] = 0.0
        output.append(prediction.astype(np.float32))
        if number % 50 == 0 or number == len(packs):
            print(f"predict {mode}/{profile_control} {number}/{len(packs)}", flush=True)
    return output


def _topk_oracle(pack, budget):
    _, _, combined = _profiles(pack, "legal")
    k = min(int(budget), N_STATES)
    if k == N_STATES:
        candidate_indices = np.broadcast_to(
            np.arange(N_STATES)[:, None], (N_STATES, combined.shape[1])
        )
    else:
        candidate_indices = np.argpartition(combined, -k, axis=0)[-k:]
    candidate_offsets = STATE_OFFSETS[candidate_indices]
    choice = np.argmin(np.abs(candidate_offsets - pack.target[None]), axis=0)
    selected = candidate_offsets[choice, np.arange(combined.shape[1])]
    selected[: pack.anchor_grid + 1] = 0.0
    return selected.astype(np.float32)


STARTED = time.time()
np.random.seed(SEED)
folds, fold_path = _load_folds()
wells = folds["well"].astype(str).tolist()
if len(wells) != 773:
    raise RuntimeError(f"Expected 773 wells, found {len(wells)}")
_ensure_records(wells)
_ensure_packs(wells)
packs = {well: _load_pack(well) for well in wells}

train_wells = folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].astype(str).tolist()
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].astype(str).tolist()
if len(train_wells) not in (463, 464) or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected Fold345/Fold2 sizes: {len(train_wells)}/{len(valid_wells)}")
train_packs = [packs[well] for well in train_wells]
valid_packs = [packs[well] for well in valid_wells]

# Fold 2 is a mechanism gateway, so reserve most wells for confirmation rather
# than reporting the same wells used for early stopping.
split_rng = np.random.default_rng(SEED)
shuffled_valid_wells = np.asarray(valid_wells)[split_rng.permutation(len(valid_wells))]
selection_wells = shuffled_valid_wells[:48].tolist()
confirmation_wells = shuffled_valid_wells[48:].tolist()
selection_packs = [packs[well] for well in selection_wells]
confirmation_packs = [packs[well] for well in confirmation_wells]

oracle_rows = []
oracle_bywell = {}
for budget in TOPK_BUDGETS:
    grid_predictions = [_topk_oracle(pack, budget) for pack in confirmation_packs]
    metrics, bywell = _score_predictions(confirmation_packs, grid_predictions)
    row = {"budget": int(budget), **metrics}
    oracle_rows.append(row)
    oracle_bywell[int(budget)] = bywell
    print("LEGAL PROFILE TOPK ORACLE", row, flush=True)
oracle_frame = pd.DataFrame(oracle_rows)
oracle_frame.to_csv(OUT_DIR / "legal_profile_topk_oracle.csv", index=False)
oracle_bywell[N_STATES].to_csv(OUT_DIR / "state_grid_oracle_bywell.csv", index=False)

baseline_model, baseline_features = _fit_arm(
    "geometry_baseline", train_packs, selection_packs, "legal", False
)
legal_model, legal_features = _fit_arm(
    "legal_profile", train_packs, selection_packs, "legal", True
)
positive_model, positive_features = _fit_arm(
    "truth_profile_positive_control",
    train_packs,
    selection_packs,
    "truth_positive",
    True,
)

arms = {}
arm_specs = {
    "geometry_baseline": (baseline_model, "legal", False, "full"),
    "legal_profile": (legal_model, "legal", True, "full"),
    "legal_profile_flat": (legal_model, "legal", True, "flat"),
    "legal_profile_rolled": (legal_model, "legal", True, "rolled"),
    "truth_profile_positive": (positive_model, "truth_positive", True, "full"),
}
for name, (model, mode, include_profile, control) in arm_specs.items():
    prediction = _predict_arm(
        model, confirmation_packs, mode, include_profile, profile_control=control
    )
    metrics, bywell = _score_predictions(confirmation_packs, prediction)
    arms[name] = metrics
    bywell.to_csv(OUT_DIR / f"{name}_bywell.csv", index=False)
    print("PROFILE FEATURE ARM", name, metrics, flush=True)

top32 = next(row for row in oracle_rows if row["budget"] == 32)
state_grid = next(row for row in oracle_rows if row["budget"] == N_STATES)
legal_gain = arms["geometry_baseline"]["rmse"] - arms["legal_profile"]["rmse"]
flat_degradation = arms["legal_profile_flat"]["rmse"] - arms["legal_profile"]["rmse"]
rolled_degradation = arms["legal_profile_rolled"]["rmse"] - arms["legal_profile"]["rmse"]

if state_grid["rmse"] > 1.5:
    conclusion = "STATE GRID FAIL -> representation/anchoring is broken"
elif arms["truth_profile_positive"]["rmse"] > 2.0:
    conclusion = "POSITIVE CONTROL FAIL -> GBDT/profile encoding cannot decode a correct peak"
elif top32["rmse"] > 5.5:
    conclusion = "CANDIDATE COVERAGE FAIL -> local legal profile cannot support 5.5"
elif arms["legal_profile"]["rmse"] > 5.5:
    conclusion = "SELECTION FAIL -> candidates contain truth; build a sequence engine, not more scalar features"
else:
    conclusion = "PROFILE FEATURE PASS -> promote the legal profile model to five-fold OOF"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_local_costprofile_feature_vs_engine_decision",
    "sample_unit": "complete physical well; grid rows nested within well",
    "legal_inputs_only_for_deployable_arm": True,
    "truth_positive_control_is_deployable": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "selection_wells": len(selection_wells),
        "confirmation_wells": len(confirmation_wells),
        "reported_metrics": "Fold2 confirmation wells only",
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
        "fold_source": str(fold_path),
        "costmap_source": str(PACK_DIR),
    },
    "features": {
        "geometry": baseline_features,
        "legal_profile_count": len(legal_features) - len(baseline_features),
        "legal_total_count": len(legal_features),
        "profile_sample_points_per_channel": int(len(PROFILE_INDICES)),
    },
    "legal_profile_topk_oracle": oracle_rows,
    "arms": arms,
    "diagnostics": {
        "legal_gain_vs_geometry": legal_gain,
        "flat_degradation": flat_degradation,
        "rolled_degradation": rolled_degradation,
    },
    "checks": {
        "state_grid_oracle_rmse_le_1p5": state_grid["rmse"] <= 1.5,
        "truth_profile_positive_rmse_le_2": arms["truth_profile_positive"]["rmse"] <= 2.0,
        "legal_top32_oracle_rmse_le_5p5": top32["rmse"] <= 5.5,
        "legal_profile_gain_ge_0p2": legal_gain >= 0.2,
        "each_control_degradation_ge_0p2": min(flat_degradation, rolled_degradation) >= 0.2,
        "legal_profile_rmse_le_5p5": arms["legal_profile"]["rmse"] <= 5.5,
    },
    "conclusion": conclusion,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
