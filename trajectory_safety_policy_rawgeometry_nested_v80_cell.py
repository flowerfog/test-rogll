"""Route 7 final gate: raw-geometry nested asymmetric safety policy.

Requires the legal feature export added to V77b.  The policy predicts the
first-order MSE benefit of each geometry endpoint from the exact 71 prefix and
389 future-trajectory features. Candidate energy is known without labels.
Rows <=1000 ft remain bitwise equal to the V66 base.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
V77_DIR = ROOT / "oof_exports" / "drilling_trajectory_sensor_gateway_v77b"
ENDPOINT_PATH = V77_DIR / "trajectory_sensor_sample_oof.parquet"
FEATURE_PATH = V77_DIR / "trajectory_sensor_legal_features.npz"
OUT_DIR = ROOT / "oof_exports" / "trajectory_safety_policy_rawgeometry_nested_v80"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ARMS = ("full_ridge_a10", "full_ridge_a100", "full_extra_leaf4", "full_extra_leaf8")
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
DOSES = np.round(np.arange(0.0, 0.251, 0.025), 3)
MAX_DOSES = (0.05, 0.10, 0.15, 0.20, 0.25)
MARGINS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0)
FREEZE_FT = 1000.0
FULL_DOSE_FT = 2000.0
SEED = 20260729

for path in (ENDPOINT_PATH, FEATURE_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)


def metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    bywell = pd.DataFrame({"well": frame["well"].astype(str), "e2": error**2}).groupby(
        "well", sort=False
    )["e2"].mean().to_numpy(np.float64)
    near = frame["md_since"].to_numpy(np.float64) <= FREEZE_FT
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint, "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(bywell))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell), 0.95)),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near]**2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far]**2))),
    }


frame = pd.read_parquet(ENDPOINT_PATH).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", "base", *ARMS}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Missing endpoint columns: {missing}")
if frame["well"].nunique() != 773 or frame["fold"].nunique() != 5:
    raise RuntimeError(f"Unexpected endpoint coverage: {len(frame)}/{frame['well'].nunique()}")

payload = np.load(FEATURE_PATH, allow_pickle=False)
feature_wells = payload["wells"].astype(str)
feature_folds = payload["folds"].astype(np.int16)
prefix = payload["prefix_features"].astype(np.float64)
future = payload["future_trajectory_features"].astype(np.float64)
raw_features = np.column_stack([prefix, future])
if (
    raw_features.shape != (773, 460)
    or len(np.unique(feature_wells)) != 773
    or not np.isfinite(raw_features).all()
):
    raise RuntimeError(f"Unexpected legal feature matrix: {raw_features.shape}")
raw_by_well = {well: raw_features[i] for i, well in enumerate(feature_wells)}
feature_fold_by_well = {well: int(feature_folds[i]) for i, well in enumerate(feature_wells)}
endpoint_fold_by_well = frame.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
if feature_fold_by_well != endpoint_fold_by_well:
    raise RuntimeError("V77b legal-feature folds do not match endpoint folds")

truth = frame["tvt_true"].to_numpy(np.float64)
base = frame["base"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
ramp = np.clip((distance - FREEZE_FT) / (FULL_DOSE_FT - FREEZE_FT), 0.0, 1.0)
row_positions, delta_by_arm, rows = {}, {arm: {} for arm in ARMS}, []

for well, positions in frame.groupby("well", sort=False).groups.items():
    well = str(well)
    if well not in raw_by_well:
        raise RuntimeError(f"No raw legal features for {well}")
    positions = np.asarray(list(positions), np.int64)
    row_positions[well] = positions
    local_base = base[positions]
    local_error = local_base - truth[positions]
    row = {
        "well": well, "fold": int(frame.loc[positions[0], "fold"]), "rows": len(positions),
        "max_md_since": float(distance[positions].max()),
    }
    local_deltas = []
    for arm in ARMS:
        candidate = frame.loc[positions, arm].to_numpy(np.float64)
        delta = ramp[positions] * (candidate - local_base)
        delta_by_arm[arm][well] = delta
        local_deltas.append(delta)
        row.update({
            f"{arm}_mean": float(np.mean(delta)),
            f"{arm}_abs_mean": float(np.mean(np.abs(delta))),
            f"{arm}_rms": float(np.sqrt(np.mean(delta**2))),
            f"{arm}_std": float(np.std(delta)),
            f"{arm}_p95_abs": float(np.quantile(np.abs(delta), 0.95)),
            f"{arm}_endpoint": float(delta[-1]),
            f"{arm}_range": float(np.ptp(delta)),
            f"{arm}_energy": float(np.mean(delta**2)),
            f"{arm}_target": float(-2.0 * np.mean(local_error * delta)),
        })
    stack = np.stack(local_deltas)
    row["arm_dispersion_mean"] = float(np.mean(np.std(stack, axis=0)))
    row["arm_dispersion_p95"] = float(np.quantile(np.std(stack, axis=0), 0.95))
    rows.append(row)

table = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
well_order = table["well"].tolist()
raw_matrix = np.stack([raw_by_well[well] for well in well_order])
derived_names = [c for c in table.columns if c not in {"well", "fold"} and not c.endswith("_target")]
features = np.column_stack([raw_matrix, table[derived_names].to_numpy(np.float64)])
if not np.isfinite(features).all():
    raise RuntimeError("Non-finite policy features")

folds = table["fold"].to_numpy(np.int16)
energies = {arm: table[f"{arm}_energy"].to_numpy(np.float64) for arm in ARMS}
targets = {arm: table[f"{arm}_target"].to_numpy(np.float64) for arm in ARMS}


def fit_ridge(train, valid, arm, alpha, seed):
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_valid = scaler.transform(features[valid])
    y = targets[arm]
    lo, hi = np.quantile(y[train], [0.01, 0.99])
    model = Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1e-6)
    model.fit(x_train, np.clip(y[train], lo, hi), sample_weight=table.loc[train, "rows"])
    return (scaler, model), model.predict(x_valid)


def choose(scores, local_energy, max_dose, margin):
    n = len(next(iter(scores.values())))
    utility = np.zeros(n)
    selected_arm = np.full(n, "base", object)
    selected_dose = np.zeros(n)
    for arm in ARMS:
        for dose in DOSES[(DOSES > 0) & (DOSES <= max_dose + 1e-9)]:
            value = dose * scores[arm] - dose * dose * local_energy[arm]
            update = value > utility
            utility[update] = value[update]
            selected_arm[update] = arm
            selected_dose[update] = dose
    reject = utility <= margin
    selected_arm[reject], selected_dose[reject] = "base", 0.0
    return selected_arm, selected_dose


def assemble(indices, selected_arm, selected_dose):
    all_positions, all_predictions, choices = [], [], []
    for j, index in enumerate(indices):
        well = table.loc[index, "well"]
        positions = row_positions[well]
        arm, dose = str(selected_arm[j]), float(selected_dose[j])
        prediction = base[positions].copy()
        if arm != "base" and dose > 0:
            prediction += dose * delta_by_arm[arm][well]
        all_positions.append(positions); all_predictions.append(prediction)
        choices.append({"well": well, "fold": int(table.loc[index, "fold"]), "arm": arm, "dose": dose})
    positions = np.concatenate(all_positions); prediction = np.concatenate(all_predictions)
    order = np.argsort(positions, kind="stable")
    return positions[order], prediction[order], pd.DataFrame(choices)


def evaluate(indices, selected_arm, selected_dose, endpoint):
    positions, prediction, choices = assemble(indices, selected_arm, selected_dose)
    return metric(frame.loc[positions], prediction, endpoint), positions, prediction, choices


oof = np.full(len(frame), np.nan); shuffled_oof = np.full(len(frame), np.nan)
fold_reports, scan_parts, choice_parts = [], [], []
print("V80 CONFIG", {"rows": len(frame), "wells": len(table), "features": features.shape[1]}, flush=True)

for held_fold in range(1, 6):
    outer_train = np.flatnonzero(folds != held_fold)
    outer_valid = np.flatnonzero(folds == held_fold)
    position = {index: j for j, index in enumerate(outer_train)}
    predictions = {(alpha, arm): np.full(len(outer_train), np.nan) for alpha in ALPHAS for arm in ARMS}

    for inner_fold in sorted(np.unique(folds[outer_train])):
        inner_valid = outer_train[folds[outer_train] == inner_fold]
        inner_train = outer_train[folds[outer_train] != inner_fold]
        destinations = np.asarray([position[index] for index in inner_valid])
        for alpha in ALPHAS:
            for arm in ARMS:
                _, pred = fit_ridge(inner_train, inner_valid, arm, alpha, SEED)
                predictions[(alpha, arm)][destinations] = pred

    configurations = []
    for alpha in ALPHAS:
        lower_scores, shifts = {}, {}
        for arm in ARMS:
            residual = targets[arm][outer_train] - predictions[(alpha, arm)]
            shifts[arm] = float(np.quantile(residual, 0.20))
            lower_scores[arm] = predictions[(alpha, arm)] + shifts[arm]
        local_energy = {arm: energies[arm][outer_train] for arm in ARMS}
        for max_dose in MAX_DOSES:
            for margin in MARGINS:
                arms, doses = choose(lower_scores, local_energy, max_dose, margin)
                result, _, _, _ = evaluate(outer_train, arms, doses, "inner")
                base_result, _, _, _ = evaluate(
                    outer_train, np.full(len(outer_train), "base", object), np.zeros(len(outer_train)), "base"
                )
                gains, p95_changes = [], []
                for fold in sorted(np.unique(folds[outer_train])):
                    local = np.flatnonzero(folds[outer_train] == fold)
                    indices = outer_train[local]
                    candidate, _, _, _ = evaluate(indices, arms[local], doses[local], "candidate")
                    control, _, _, _ = evaluate(indices, np.full(len(local), "base", object), np.zeros(len(local)), "base")
                    gains.append(control["rmse"] - candidate["rmse"])
                    p95_changes.append(candidate["p95_well_rmse"] - control["p95_well_rmse"])
                configurations.append({
                    "held_fold": held_fold, "alpha": alpha, "max_dose": max_dose, "margin": margin,
                    "inner_gain": base_result["rmse"] - result["rmse"],
                    "minimum_inner_fold_gain": float(np.min(gains)),
                    "maximum_inner_p95_change": float(np.max(p95_changes)),
                    "activation_fraction": float(np.mean(doses > 0)),
                    "shifts": shifts,
                })
    scan_parts.extend([{k: v for k, v in row.items() if k != "shifts"} for row in configurations])
    safe = [row for row in configurations if row["minimum_inner_fold_gain"] >= 0 and row["maximum_inner_p95_change"] <= 0.20 and row["activation_fraction"] <= 0.50 and row["inner_gain"] > 0]
    selected = max(safe, key=lambda row: (row["inner_gain"], row["minimum_inner_fold_gain"], row["margin"]), default=None)

    if selected is None:
        held_arms = np.full(len(outer_valid), "base", object); held_doses = np.zeros(len(outer_valid))
        held_scores = {arm: np.zeros(len(outer_valid)) for arm in ARMS}
    else:
        held_scores = {}
        for arm in ARMS:
            bundle, pred = fit_ridge(outer_train, outer_valid, arm, selected["alpha"], SEED + held_fold)
            held_scores[arm] = pred + selected["shifts"][arm]
            joblib.dump(bundle, MODEL_DIR / f"{arm}_hold{held_fold}.joblib", compress=3)
        held_arms, held_doses = choose(
            held_scores, {arm: energies[arm][outer_valid] for arm in ARMS},
            selected["max_dose"], selected["margin"],
        )
    result, positions, prediction, choices = evaluate(outer_valid, held_arms, held_doses, "policy")
    oof[positions] = prediction; choice_parts.append(choices)

    rng = np.random.default_rng(SEED + held_fold)
    shuffled_scores = {arm: values.copy() for arm, values in held_scores.items()}
    bins = np.asarray(pd.qcut(table.loc[outer_valid, "max_md_since"], 5, labels=False, duplicates="drop"))
    for value in np.unique(bins):
        local = np.flatnonzero(bins == value)
        permutation = rng.permutation(local)
        for arm in ARMS:
            shuffled_scores[arm][local] = held_scores[arm][permutation]
    if selected is None:
        shuffle_arms, shuffle_doses = held_arms.copy(), held_doses.copy()
    else:
        shuffle_arms, shuffle_doses = choose(
            shuffled_scores, {arm: energies[arm][outer_valid] for arm in ARMS},
            selected["max_dose"], selected["margin"],
        )
    shuffle_result, shuffle_positions, shuffle_prediction, _ = evaluate(outer_valid, shuffle_arms, shuffle_doses, "shuffle")
    shuffled_oof[shuffle_positions] = shuffle_prediction
    base_result, _, _, _ = evaluate(outer_valid, np.full(len(outer_valid), "base", object), np.zeros(len(outer_valid)), "base")
    report = {
        "fold": held_fold, "selected_alpha": 0.0 if selected is None else selected["alpha"],
        "selected_max_dose": 0.0 if selected is None else selected["max_dose"],
        "selected_margin": 1e30 if selected is None else selected["margin"],
        "activation_fraction": float(np.mean(held_doses > 0)),
        "base_rmse": base_result["rmse"], "rmse": result["rmse"], "gain": base_result["rmse"] - result["rmse"],
        "base_far_rmse": base_result["far_gt2000_rmse"], "far_rmse": result["far_gt2000_rmse"],
        "far_gain": base_result["far_gt2000_rmse"] - result["far_gt2000_rmse"],
        "p95_change": result["p95_well_rmse"] - base_result["p95_well_rmse"],
        "shuffle_loss": shuffle_result["rmse"] - result["rmse"],
    }
    fold_reports.append(report); print("V80 FOLD", report, flush=True)

if not np.isfinite(oof).all() or not np.isfinite(shuffled_oof).all():
    raise RuntimeError("Incomplete V80 OOF")
fold_report = pd.DataFrame(fold_reports)
choices = pd.concat(choice_parts, ignore_index=True)
base_result = metric(frame, base, "base_v66")
policy_result = metric(frame, oof, "raw_geometry_safety_policy")
shuffle_result = metric(frame, shuffled_oof, "score_alignment_shuffle")
gain = base_result["rmse"] - policy_result["rmse"]
far_gain = base_result["far_gt2000_rmse"] - policy_result["far_gt2000_rmse"]
shuffle_loss = shuffle_result["rmse"] - policy_result["rmse"]
checks = {
    "gain_ge_0p05": bool(gain >= 0.05), "far_gain_ge_0p075": bool(far_gain >= 0.075),
    "positive_folds_ge_4": bool((fold_report["gain"] > 0).sum() >= 4),
    "minimum_fold_gain_ge_minus0p02": bool(fold_report["gain"].min() >= -0.02),
    "maximum_p95_change_le_0p20": bool(fold_report["p95_change"].max() <= 0.20),
    "shuffle_loses_ge_half_gain": bool(shuffle_loss >= max(0.02, 0.5 * gain)),
    "near_rows_bitwise_unchanged": bool(np.array_equal(oof[distance <= FREEZE_FT], base[distance <= FREEZE_FT])),
}
verdict = {
    "experiment": "trajectory_safety_policy_rawgeometry_nested_v80",
    "stage": "route7_final_raw460_feature_nested_safety_policy",
    "features": int(features.shape[1]), "base": base_result, "policy": policy_result,
    "shuffle": shuffle_result, "gain": gain, "far_gain": far_gain, "shuffle_loss": shuffle_loss,
    "activation_fraction": float(np.mean(choices["dose"] > 0)), "folds": fold_reports,
    "checks": checks, "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
verdict["next"] = "PASS -> rebuild full-row endpoint and freeze policy" if verdict["strict_pass"] else "FAIL -> close Route 7 completely"
pd.DataFrame(scan_parts).to_csv(OUT_DIR / "inner_scan.csv", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False); choices.to_csv(OUT_DIR / "choices.csv", index=False)
output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since", "base"]].copy()
output["policy"] = oof.astype(np.float32); output["shuffle"] = shuffled_oof.astype(np.float32)
output.to_parquet(OUT_DIR / "policy_sample_oof.parquet", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nV80 FOLDS"); print(fold_report.to_string(index=False))
print("\nV80 FINAL VERDICT"); print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
