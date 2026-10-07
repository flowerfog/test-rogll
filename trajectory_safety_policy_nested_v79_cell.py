"""Route 7 Phase 1: nested asymmetric safety policy for V77b endpoints.

For every candidate arm, learn a conservative lower quantile of the first-order
MSE benefit.  The known candidate energy supplies the quadratic cost, so a dose
is activated only when its predicted lower-bound utility is positive by a
selected safety margin.  Selection is nested by complete physical well.

Rows through 1000 ft are frozen and the dose ramps to full strength at 2000 ft.
This is an OOF gateway, not a Kaggle deployment bundle.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
SOURCE = OOF_ROOT / "drilling_trajectory_sensor_gateway_v77b" / "trajectory_sensor_sample_oof.parquet"
EXPERIMENT = "trajectory_safety_policy_nested_v79"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ARMS = ("full_ridge_a10", "full_ridge_a100", "full_extra_leaf4", "full_extra_leaf8")
DOSES = np.round(np.arange(0.0, 0.251, 0.025), 3)
MAX_DOSES = (0.05, 0.10, 0.15, 0.20, 0.25)
MARGINS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0)
FREEZE_FT = 1000.0
FULL_DOSE_FT = 2000.0
SEED = 20260729

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)


def metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    bywell = (
        pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
        .groupby("well", sort=False)["e2"].mean().to_numpy(np.float64)
    )
    near = frame["md_since"].to_numpy(np.float64) <= FREEZE_FT
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(bywell))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell), 0.95)),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def stats(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(values))),
        f"{prefix}_rms": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p95_abs": float(np.quantile(np.abs(values), 0.95)),
        f"{prefix}_endpoint": float(values[-1]),
        f"{prefix}_range": float(np.ptp(values)),
    }


def build_well_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, np.ndarray]:
    distance = frame["md_since"].to_numpy(np.float64)
    ramp = np.clip((distance - FREEZE_FT) / (FULL_DOSE_FT - FREEZE_FT), 0.0, 1.0)
    truth = frame["tvt_true"].to_numpy(np.float64)
    base = frame["base"].to_numpy(np.float64)
    rows, row_positions, deltas = [], {}, {arm: {} for arm in ARMS}
    for well, positions in frame.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(positions), np.int64)
        row_positions[str(well)] = positions
        local_distance = distance[positions]
        local_truth = truth[positions]
        local_base = base[positions]
        base_error = local_base - local_truth
        base_move = local_base - local_base[0]
        row = {
            "well": str(well), "fold": int(frame.loc[positions[0], "fold"]),
            "rows": int(len(positions)), "max_md_since": float(local_distance.max()),
            **stats(base_move, "base_move"),
        }
        arm_delta_vectors = {}
        for arm in ARMS:
            candidate = frame.loc[positions, arm].to_numpy(np.float64)
            delta = ramp[positions] * (candidate - local_base)
            deltas[arm][str(well)] = delta
            arm_delta_vectors[arm] = delta
            row.update(stats(delta, f"{arm}_delta"))
            row[f"{arm}_energy"] = float(np.mean(delta * delta))
            row[f"{arm}_cross_target"] = float(-2.0 * np.mean(base_error * delta))
            for low, high, label in (
                (0.0, 500.0, "0_500"), (500.0, 1000.0, "500_1000"),
                (1000.0, 2000.0, "1000_2000"), (2000.0, np.inf, "gt2000"),
            ):
                mask = (local_distance > low) & (local_distance <= high)
                row[f"{arm}_delta_rms_{label}"] = (
                    float(np.sqrt(np.mean(delta[mask] ** 2))) if mask.any() else 0.0
                )
        stack = np.stack([arm_delta_vectors[arm] for arm in ARMS])
        row["arm_dispersion_mean"] = float(np.mean(np.std(stack, axis=0)))
        row["arm_dispersion_p95"] = float(np.quantile(np.std(stack, axis=0), 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("well").reset_index(drop=True), row_positions, deltas, ramp


def make_model(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile", alpha=0.20, n_estimators=220,
        learning_rate=0.025, max_depth=2, min_samples_leaf=24,
        subsample=0.85, random_state=seed,
    )


def fit_predict(train_idx, valid_idx, arm, features, table, seed):
    target = table[f"{arm}_cross_target"].to_numpy(np.float64)
    # Winsorisation is fitted on outer-train labels only.
    lo, hi = np.quantile(target[train_idx], [0.01, 0.99])
    train_target = np.clip(target[train_idx], lo, hi)
    model = make_model(seed)
    model.fit(
        features[train_idx], train_target,
        sample_weight=table.loc[train_idx, "rows"].to_numpy(np.float64),
    )
    return model, model.predict(features[valid_idx]).astype(np.float64)


def choose(scores: dict[str, np.ndarray], energies: dict[str, np.ndarray], max_dose: float, margin: float):
    n = len(next(iter(scores.values())))
    best_utility = np.zeros(n, np.float64)
    best_arm = np.full(n, "base", object)
    best_dose = np.zeros(n, np.float64)
    doses = DOSES[DOSES <= max_dose + 1e-9]
    for arm in ARMS:
        for dose in doses[1:]:
            utility = float(dose) * scores[arm] - float(dose * dose) * energies[arm]
            update = utility > best_utility
            best_utility[update] = utility[update]
            best_arm[update] = arm
            best_dose[update] = float(dose)
    reject = best_utility <= margin
    best_arm[reject] = "base"
    best_dose[reject] = 0.0
    return best_arm, best_dose, best_utility


frame = pd.read_parquet(SOURCE)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", "base", *ARMS}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Missing source columns: {missing}")
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
fold_coverage = frame.groupby("fold")["well"].nunique().astype(int).to_dict()
if frame["well"].nunique() != 773 or set(fold_coverage) != {1, 2, 3, 4, 5}:
    raise RuntimeError(
        f"Unexpected V77b coverage: rows={len(frame)} "
        f"wells={frame['well'].nunique()} folds={fold_coverage}"
    )
if not np.isfinite(frame[["tvt_true", "md_since", "base", *ARMS]].to_numpy(np.float64)).all():
    raise RuntimeError("Non-finite V77b endpoint values")
table, row_positions, delta_by_arm, ramp = build_well_table(frame)
feature_names = [
    column for column in table.columns
    if column not in {"well", "fold"} and not column.endswith("_cross_target")
]
features = table[feature_names].to_numpy(np.float64)
if not np.isfinite(features).all():
    raise RuntimeError("Non-finite legal features")

well_to_table = {well: index for index, well in enumerate(table["well"])}
truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["base"].to_numpy(np.float64)
fold_array = table["fold"].to_numpy(np.int16)
energies = {arm: table[f"{arm}_energy"].to_numpy(np.float64) for arm in ARMS}


def assemble(well_indices, selected_arm, selected_dose):
    positions_all, prediction_all = [], []
    choices = []
    for local, table_index in enumerate(well_indices):
        well = str(table.loc[table_index, "well"])
        positions = row_positions[well]
        arm = str(selected_arm[local])
        dose = float(selected_dose[local])
        prediction = base[positions].copy()
        if arm != "base" and dose > 0:
            prediction += dose * delta_by_arm[arm][well]
        positions_all.append(positions)
        prediction_all.append(prediction)
        choices.append({"well": well, "fold": int(table.loc[table_index, "fold"]), "arm": arm, "dose": dose})
    positions = np.concatenate(positions_all)
    prediction = np.concatenate(prediction_all)
    order = np.argsort(positions, kind="stable")
    return positions[order], prediction[order], pd.DataFrame(choices)


def evaluate_indices(well_indices, selected_arm, selected_dose, endpoint):
    positions, prediction, choices = assemble(well_indices, selected_arm, selected_dose)
    return metric(frame.loc[positions], prediction, endpoint), positions, prediction, choices


oof_prediction = np.full(len(frame), np.nan, np.float64)
shuffle_prediction = np.full(len(frame), np.nan, np.float64)
fold_reports, selection_reports, choice_parts = [], [], []

print("V79 CONFIG", {
    "rows": len(frame), "wells": len(table), "features": len(feature_names),
    "arms": ARMS, "doses": DOSES.tolist(),
}, flush=True)

for held_fold in range(1, 6):
    outer_train = np.flatnonzero(fold_array != held_fold)
    outer_valid = np.flatnonzero(fold_array == held_fold)

    # Cross-fitted lower bounds inside outer-train drive configuration choice.
    inner_scores = {arm: np.full(len(outer_train), np.nan, np.float64) for arm in ARMS}
    outer_position = {index: position for position, index in enumerate(outer_train)}
    for inner_fold in sorted(np.unique(fold_array[outer_train])):
        inner_valid = outer_train[fold_array[outer_train] == inner_fold]
        inner_train = outer_train[fold_array[outer_train] != inner_fold]
        destinations = np.asarray([outer_position[index] for index in inner_valid], np.int64)
        for arm_index, arm in enumerate(ARMS):
            _, prediction = fit_predict(
                inner_train, inner_valid, arm, features, table,
                SEED + held_fold * 100 + int(inner_fold) * 10 + arm_index,
            )
            inner_scores[arm][destinations] = prediction

    inner_energies = {arm: energies[arm][outer_train] for arm in ARMS}
    config_rows = []
    for max_dose in MAX_DOSES:
        for margin in MARGINS:
            arms, doses, _ = choose(inner_scores, inner_energies, max_dose, margin)
            result, _, _, choices = evaluate_indices(outer_train, arms, doses, "inner_policy")
            base_result, _, _, _ = evaluate_indices(
                outer_train, np.full(len(outer_train), "base", object), np.zeros(len(outer_train)), "inner_base"
            )
            per_fold_gains, per_fold_p95 = [], []
            for fold in sorted(np.unique(fold_array[outer_train])):
                local = np.flatnonzero(fold_array[outer_train] == fold)
                local_indices = outer_train[local]
                candidate_result, _, _, _ = evaluate_indices(local_indices, arms[local], doses[local], "inner_fold")
                local_base, _, _, _ = evaluate_indices(
                    local_indices, np.full(len(local), "base", object), np.zeros(len(local)), "inner_fold_base"
                )
                per_fold_gains.append(local_base["rmse"] - candidate_result["rmse"])
                per_fold_p95.append(candidate_result["p95_well_rmse"] - local_base["p95_well_rmse"])
            config_rows.append({
                "held_fold": held_fold, "max_dose": max_dose, "margin": margin,
                "inner_gain": base_result["rmse"] - result["rmse"],
                "minimum_inner_fold_gain": float(np.min(per_fold_gains)),
                "maximum_inner_p95_change": float(np.max(per_fold_p95)),
                "activation_fraction": float(np.mean(doses > 0)),
            })
    configs = pd.DataFrame(config_rows)
    safe = configs[
        (configs["minimum_inner_fold_gain"] >= 0.0)
        & (configs["maximum_inner_p95_change"] <= 0.20)
        & (configs["activation_fraction"] <= 0.50)
    ]
    if safe.empty or float(safe["inner_gain"].max()) <= 0.0:
        selected = {"max_dose": 0.0, "margin": 1.0e30, "inner_gain": 0.0,
                    "minimum_inner_fold_gain": 0.0, "maximum_inner_p95_change": 0.0,
                    "activation_fraction": 0.0}
    else:
        selected = safe.sort_values(
            ["inner_gain", "minimum_inner_fold_gain", "margin"],
            ascending=[False, False, False], kind="stable",
        ).iloc[0].to_dict()
    selection_reports.extend(config_rows)

    held_scores = {}
    for arm_index, arm in enumerate(ARMS):
        model, prediction = fit_predict(
            outer_train, outer_valid, arm, features, table,
            SEED + held_fold * 1000 + arm_index,
        )
        held_scores[arm] = prediction
        joblib.dump(model, MODEL_DIR / f"{arm}_hold{held_fold}.joblib", compress=3)

    if selected["max_dose"] == 0.0:
        held_arms = np.full(len(outer_valid), "base", object)
        held_doses = np.zeros(len(outer_valid), np.float64)
    else:
        held_arms, held_doses, _ = choose(
            held_scores, {arm: energies[arm][outer_valid] for arm in ARMS},
            float(selected["max_dose"]), float(selected["margin"]),
        )
    held_result, held_positions, held_prediction, held_choices = evaluate_indices(
        outer_valid, held_arms, held_doses, "nested_policy"
    )
    oof_prediction[held_positions] = held_prediction
    choice_parts.append(held_choices)

    # Matched negative control: preserve score distributions and suffix length,
    # but break the score-to-well alignment within length quintiles.
    shuffled_scores = {arm: values.copy() for arm, values in held_scores.items()}
    held_lengths = table.loc[outer_valid, "max_md_since"].to_numpy(np.float64)
    bins = pd.qcut(held_lengths, q=5, labels=False, duplicates="drop")
    rng = np.random.default_rng(SEED + held_fold)
    for value in np.unique(bins):
        local = np.flatnonzero(np.asarray(bins) == value)
        if len(local) > 1:
            permutation = rng.permutation(local)
            for arm in ARMS:
                shuffled_scores[arm][local] = held_scores[arm][permutation]
    if selected["max_dose"] == 0.0:
        shuffled_arms = np.full(len(outer_valid), "base", object)
        shuffled_doses = np.zeros(len(outer_valid), np.float64)
    else:
        shuffled_arms, shuffled_doses, _ = choose(
            shuffled_scores, {arm: energies[arm][outer_valid] for arm in ARMS},
            float(selected["max_dose"]), float(selected["margin"]),
        )
    shuffle_result, shuffle_positions, shuffle_values, _ = evaluate_indices(
        outer_valid, shuffled_arms, shuffled_doses, "score_alignment_shuffle"
    )
    shuffle_prediction[shuffle_positions] = shuffle_values
    base_result, _, _, _ = evaluate_indices(
        outer_valid, np.full(len(outer_valid), "base", object), np.zeros(len(outer_valid)), "base"
    )
    fold_reports.append({
        "fold": held_fold,
        "selected_max_dose": float(selected["max_dose"]),
        "selected_margin": float(selected["margin"]),
        "inner_gain": float(selected["inner_gain"]),
        "minimum_inner_fold_gain": float(selected["minimum_inner_fold_gain"]),
        "activation_fraction": float(np.mean(held_doses > 0)),
        "base_rmse": base_result["rmse"], "rmse": held_result["rmse"],
        "gain": base_result["rmse"] - held_result["rmse"],
        "base_far_rmse": base_result["far_gt2000_rmse"],
        "far_rmse": held_result["far_gt2000_rmse"],
        "far_gain": base_result["far_gt2000_rmse"] - held_result["far_gt2000_rmse"],
        "p95_change": held_result["p95_well_rmse"] - base_result["p95_well_rmse"],
        "shuffle_rmse": shuffle_result["rmse"],
        "shuffle_loss": shuffle_result["rmse"] - held_result["rmse"],
    })
    print("V79 FOLD", fold_reports[-1], flush=True)

if not np.isfinite(oof_prediction).all() or not np.isfinite(shuffle_prediction).all():
    raise RuntimeError("Incomplete nested OOF")

base_result = metric(frame, base, "base_v66")
policy_result = metric(frame, oof_prediction, "nested_safety_policy")
shuffle_result = metric(frame, shuffle_prediction, "score_alignment_shuffle")
fold_report = pd.DataFrame(fold_reports)
choices = pd.concat(choice_parts, ignore_index=True)
pd.DataFrame(selection_reports).to_csv(OUT_DIR / "inner_scan.csv", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)
choices.to_csv(OUT_DIR / "choices.csv", index=False)
table[["well", "fold", *feature_names]].to_parquet(OUT_DIR / "legal_features.parquet", index=False)
output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since", "base"]].copy()
output["nested_policy"] = oof_prediction.astype(np.float32)
output["score_alignment_shuffle"] = shuffle_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "nested_policy_sample_oof.parquet", index=False)

gain = base_result["rmse"] - policy_result["rmse"]
far_gain = base_result["far_gt2000_rmse"] - policy_result["far_gt2000_rmse"]
shuffle_loss = shuffle_result["rmse"] - policy_result["rmse"]
checks = {
    "overall_gain_ge_0p05": bool(gain >= 0.05),
    "far_gain_ge_0p075": bool(far_gain >= 0.075),
    "positive_folds_ge_4": bool((fold_report["gain"] > 0).sum() >= 4),
    "minimum_fold_gain_ge_minus0p02": bool(fold_report["gain"].min() >= -0.02),
    "maximum_p95_change_le_0p20": bool(fold_report["p95_change"].max() <= 0.20),
    "shuffle_loses_ge_half_gain": bool(shuffle_loss >= max(0.02, 0.50 * gain)),
    "near_rows_bitwise_unchanged": bool(np.array_equal(
        oof_prediction[frame["md_since"].to_numpy(np.float64) <= FREEZE_FT],
        base[frame["md_since"].to_numpy(np.float64) <= FREEZE_FT],
    )),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "route7_phase1_fivefold_nested_asymmetric_lower_confidence_policy",
    "sample_unit": "complete physical well",
    "legal_features_only": True,
    "target": "lower 20% quantile of first-order MSE benefit; candidate energy is exact and label-free",
    "source": str(SOURCE),
    "coverage": {"rows": int(len(frame)), "wells": int(table["well"].nunique()), "fold_wells": fold_coverage},
    "features": len(feature_names),
    "base": base_result, "policy": policy_result, "shuffle": shuffle_result,
    "gain": gain, "far_gain": far_gain, "shuffle_loss": shuffle_loss,
    "activation_fraction": float(np.mean(choices["dose"] > 0)),
    "folds": fold_report.to_dict("records"),
    "checks": checks, "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild full-row V77b OOF and package this frozen policy"
        if strict_pass else
        "FAIL -> oracle headroom is not reachable from legal safety features; close Route 7"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nV79 FOLDS")
print(fold_report.to_string(index=False))
print("\nV79 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
