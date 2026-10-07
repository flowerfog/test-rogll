"""V124: legal complete-well prediction of V123 tail correction coefficients.

One physical well is one sample. Fold1+2 train legal well-level regressors;
the first 24 Fold3 wells select profile/model/dose/cap, and the remaining 130
Fold3 wells confirm once. Oracle coefficients are training targets only.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "final_consensus_coefficient_regressor_v124"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
BASE_PATH = OOF_ROOT / "rawdata_lattice_consensus_center_fold2_replay_v115" / "fold123_consensus_center_oof.parquet"
TARGET_PATH = OOF_ROOT / "final_consensus_residual_oracle_v123" / "oracle_bywell.csv"
TRAIN_FOLDS = (1, 2)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
FREEZE_FT = 1000.0
RAMP_FT = 250.0
DOSES = (0.25, 0.50, 1.00)
CAPS_FT = (4.0, 8.0)
TRUTH = "tvt_true"
BASE = "consensus_center"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"

PROFILES = {
    "tail_datum": ["tail_datum_ft"],
    "tail_affine": ["tail_affine_intercept_ft", "tail_affine_slope_ft_per_1000ft"],
    "tail_piecewise": [
        "tail_piecewise_intercept_ft",
        "tail_piecewise_hinge1250_ft",
        "tail_piecewise_hinge2250_ft",
    ],
}


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


def _bootstrap_v110():
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("final_consensus_v124_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _fill(values, coordinate):
    values = np.asarray(values, np.float64)
    coordinate = np.asarray(coordinate, np.float64)
    finite = np.isfinite(values) & np.isfinite(coordinate)
    if finite.sum() < 2:
        return np.full(len(values), float(np.nanmedian(values)), np.float64)
    output = values.copy()
    output[~finite] = np.interp(coordinate[~finite], coordinate[finite], values[finite])
    return output


def _slope(values, distance):
    values = np.asarray(values, np.float64)
    x = (np.asarray(distance, np.float64) - np.mean(distance)) / 1000.0
    denominator = float(np.dot(x, x))
    return float(np.dot(x, values - np.mean(values)) / denominator) if denominator > 1e-12 else 0.0


def _summarize(output, name, values, distance):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        values = np.zeros(len(distance), np.float64)
    else:
        values = np.interp(distance, distance[finite], values[finite])
    output[f"{name}_mean"] = float(np.mean(values))
    output[f"{name}_std"] = float(np.std(values))
    output[f"{name}_q10"] = float(np.quantile(values, 0.10))
    output[f"{name}_q50"] = float(np.quantile(values, 0.50))
    output[f"{name}_q90"] = float(np.quantile(values, 0.90))
    output[f"{name}_first"] = float(values[0])
    output[f"{name}_last"] = float(values[-1])
    output[f"{name}_slope"] = _slope(values, distance)
    bands = {
        "near": distance <= 500.0,
        "mid1": (distance > 500.0) & (distance <= 1000.0),
        "mid2": (distance > 1000.0) & (distance <= 2000.0),
        "far": distance > 2000.0,
    }
    for label, mask in bands.items():
        output[f"{name}_{label}_mean"] = float(np.mean(values[mask])) if mask.any() else 0.0
        output[f"{name}_{label}_std"] = float(np.std(values[mask])) if mask.any() else 0.0


def _load_record_relaxed(V, well):
    original = V["ROW_STRIDE"]
    try:
        V["ROW_STRIDE"] = 1
        return V["_load_record"](well)
    finally:
        V["ROW_STRIDE"] = original


def _well_features(local, record):
    eval_rows = record["eval_rows"]
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    md = record["md"][eval_rows].astype(np.float64)
    gr = _fill(record["gr"], record["md"])[eval_rows]
    z = _fill(record["z"], record["md"])[eval_rows]
    base = local[BASE].to_numpy(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    parent = local[PARENT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{local['well'].iloc[0]}: row mismatch")
    output = {
        "well": str(local["well"].iloc[0]),
        "fold": int(local["fold"].iloc[0]),
        "rows": len(local),
        "length_ft": float(distance[-1]),
        "spacing_ft": float(np.median(np.diff(md))) if len(md) > 1 else 0.0,
        "noise_scale": float(record["noise_scale"]),
        "anchor_surface": float(record["anchor_surface"]),
        "anchor_tvt": float(record["tvt_input"][record["anchor_index"]]),
        "anchor_z": float(record["z"][record["anchor_index"]]),
    }
    signals = {
        "base": base,
        "exact": exact,
        "parent": parent,
        "coarse": coarse,
        "consensus_move": base - exact,
        "coarse_move": coarse - parent,
        "gr": gr,
        "gr_grad": np.gradient(gr, md),
        "z": z,
        "z_grad": np.gradient(z, md),
        "base_grad": np.gradient(base, md),
        "base_curvature": np.gradient(np.gradient(base, md), md),
    }
    type_tvt = np.asarray(record["type_tvt"], np.float64)
    type_gr = np.asarray(record["type_gr"], np.float64)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_frame = pd.DataFrame({"tvt": type_tvt[finite], "gr": type_gr[finite]})
    type_frame = type_frame.groupby("tvt", as_index=False, sort=True)["gr"].mean()
    for offset in (-24.0, -12.0, 0.0, 12.0, 24.0):
        reference = np.interp(
            base + offset,
            type_frame["tvt"].to_numpy(np.float64),
            type_frame["gr"].to_numpy(np.float64),
        )
        label = f"m{abs(int(offset))}" if offset < 0 else f"p{int(offset)}"
        signals[f"mismatch_{label}"] = gr - reference
    known = np.flatnonzero(np.isfinite(record["tvt_input"]) & np.isfinite(record["gr"]))
    prefix = known[record["md"][known] >= float(record["anchor_md"]) - 500.0]
    prefix_gr = record["gr"][prefix].astype(np.float64)
    output["prefix_gr_mean"] = float(np.nanmean(prefix_gr))
    output["prefix_gr_std"] = float(np.nanstd(prefix_gr))
    output["prefix_gr_last"] = float(prefix_gr[-1])
    for name, values in signals.items():
        _summarize(output, name, values, distance)
    return output


def _basis(profile, distance):
    distance = np.asarray(distance, np.float64)
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    x = (distance - np.mean(distance)) / 1000.0
    if profile == "tail_datum":
        return taper[:, None]
    if profile == "tail_affine":
        return np.column_stack([taper, taper * x])
    if profile == "tail_piecewise":
        hinge1 = taper * np.clip((distance - 1250.0) / 1000.0, 0.0, None)
        hinge2 = taper * np.clip((distance - 2250.0) / 1000.0, 0.0, None)
        return np.column_stack([taper, hinge1, hinge2])
    raise KeyError(profile)


def _fit_models(x, target_by_profile, family):
    fitted = {}
    for profile, target in target_by_profile.items():
        members = []
        for column in range(target.shape[1]):
            if family == "ridge10":
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
            elif family == "ridge100":
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
            elif family == "hist":
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    HistGradientBoostingRegressor(
                        max_iter=150, learning_rate=0.04, max_leaf_nodes=7,
                        min_samples_leaf=25, l2_regularization=10.0, random_state=20260801 + column,
                    ),
                )
            else:
                raise KeyError(family)
            model.fit(x, target[:, column])
            members.append(model)
        fitted[profile] = members
    return fitted


def _predict(models, x):
    return np.column_stack([model.predict(x) for model in models])


for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
R = _bootstrap_v110()
N = R._bootstrap_v81()
V = N.V
frame = pd.read_parquet(BASE_PATH)
frame["well"] = frame["well"].astype(str)
missing_lineage = [column for column in (PARENT, COARSE) if column not in frame]
if missing_lineage:
    lineage = pd.read_parquet(
        R.V70_PATH, columns=["id", "fold", TRUTH, *missing_lineage],
        filters=[("fold", "in", [1, 2, 3])],
    )
    frame = frame.merge(lineage, on="id", how="left", validate="one_to_one", suffixes=("", "_lineage"))
    if frame[missing_lineage].isna().any().any():
        raise RuntimeError("Incomplete V70 lineage")
    truth_gap = float(np.max(np.abs(frame[TRUTH] - frame[f"{TRUTH}_lineage"])))
    fold_gap = int(np.max(np.abs(frame["fold"] - frame["fold_lineage"])))
    if truth_gap > 0.01 or fold_gap != 0:
        raise RuntimeError(f"Lineage mismatch truth={truth_gap} fold={fold_gap}")
    frame = frame.drop(columns=[f"{TRUTH}_lineage", "fold_lineage"])
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)

feature_rows = []
for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    feature_rows.append(_well_features(local, _load_record_relaxed(V, well)))
    if number % 100 == 0 or number == frame["well"].nunique():
        print(f"V124 features {number}/{frame['well'].nunique()}", flush=True)
features = pd.DataFrame(feature_rows).sort_values("well", kind="stable").reset_index(drop=True)
targets = pd.read_csv(TARGET_PATH, dtype={"well": str})
targets["well"] = targets["well"].astype(str)
target_columns = [column for columns in PROFILES.values() for column in columns]
targets = targets[["well", "fold", *target_columns]].copy()
panel = features.merge(targets, on=["well", "fold"], how="left", validate="one_to_one")
if panel[target_columns].isna().any().any():
    raise RuntimeError("Incomplete V123 coefficient targets")
feature_columns = [column for column in features if column not in {"well", "fold"}]
if not np.isfinite(panel[feature_columns].to_numpy(np.float64)).all():
    raise RuntimeError("Non-finite V124 well features")

train_mask = panel["fold"].isin(TRAIN_FOLDS).to_numpy()
fold3_wells = sorted(panel.loc[panel["fold"].eq(EVAL_FOLD), "well"])
discovery_wells = set(fold3_wells[:DISCOVERY_WELLS])
confirmation_wells = set(fold3_wells[DISCOVERY_WELLS:])
x_train = panel.loc[train_mask, feature_columns].to_numpy(np.float64)
target_by_profile = {
    profile: panel.loc[train_mask, columns].to_numpy(np.float64)
    for profile, columns in PROFILES.items()
}
families = ("ridge10", "ridge100", "hist")
models = {family: _fit_models(x_train, target_by_profile, family) for family in families}

coefficient_predictions = {}
for family in families:
    for profile in PROFILES:
        coefficient_predictions[(family, profile)] = dict(zip(
            panel["well"], _predict(models[family][profile], panel[feature_columns].to_numpy(np.float64))
        ))

recipes = [
    (family, profile, dose, cap)
    for family in families for profile in PROFILES for dose in DOSES for cap in CAPS_FT
]


def _evaluate(selected_wells, recipe, endpoint):
    family, profile, dose, cap = recipe
    local_parts, prediction_parts = [], []
    for well in sorted(selected_wells):
        local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        base = local[BASE].to_numpy(np.float64)
        distance = local["md_since"].to_numpy(np.float64)
        coefficient = coefficient_predictions[(family, profile)][well]
        move = _basis(profile, distance) @ coefficient
        move = np.clip(float(dose) * move, -float(cap), float(cap))
        prediction_parts.append(base + move)
        local_parts.append(local)
    matched = pd.concat(local_parts, ignore_index=True)
    prediction = np.concatenate(prediction_parts)
    return R._metric(matched, prediction, endpoint), matched, prediction


discovery_frame = frame[frame["well"].isin(discovery_wells)].copy()
discovery_base = R._metric(discovery_frame, discovery_frame[BASE].to_numpy(np.float64), BASE)
scan_rows = []
for recipe in recipes:
    result, _, _ = _evaluate(discovery_wells, recipe, "coefficient_regressor")
    family, profile, dose, cap = recipe
    result.update({
        "family": family, "profile": profile, "dose": dose, "cap_ft": cap,
        "gain": float(discovery_base["rmse"] - result["rmse"]),
        "far_gain": float(discovery_base["far_gt2000_rmse"] - result["far_gt2000_rmse"]),
        "near_degradation": float(result["near_0_500_rmse"] - discovery_base["near_0_500_rmse"]),
        "p95_gain": float(discovery_base["p95_well_rmse"] - result["p95_well_rmse"]),
    })
    scan_rows.append(result)
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.05) & (scan["far_gain"] >= 0.0)
    & (scan["near_degradation"].abs() <= 1e-5) & (scan["p95_gain"] >= -0.10)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected = (
    str(selected_row["family"]), str(selected_row["profile"]),
    float(selected_row["dose"]), float(selected_row["cap_ft"]),
)

summary_rows = [dict(discovery_base, stage="fold3_discovery")]
discovery_candidate, _, _ = _evaluate(discovery_wells, selected, "coefficient_regressor")
summary_rows.append(dict(discovery_candidate, stage="fold3_discovery"))
confirmation_values = None
confirmation_checks = None
output = None
if len(eligible):
    confirmation_frame = frame[frame["well"].isin(confirmation_wells)].copy()
    confirmation_base = R._metric(
        confirmation_frame, confirmation_frame[BASE].to_numpy(np.float64), BASE
    )
    confirmation_candidate, matched, prediction = _evaluate(
        confirmation_wells, selected, "coefficient_regressor"
    )
    summary_rows.extend([
        dict(confirmation_base, stage="fold3_confirmation"),
        dict(confirmation_candidate, stage="fold3_confirmation"),
    ])
    confirmation_values = {
        "gain": float(confirmation_base["rmse"] - confirmation_candidate["rmse"]),
        "far_gain": float(confirmation_base["far_gt2000_rmse"] - confirmation_candidate["far_gt2000_rmse"]),
        "near_degradation": float(confirmation_candidate["near_0_500_rmse"] - confirmation_base["near_0_500_rmse"]),
        "p95_gain": float(confirmation_base["p95_well_rmse"] - confirmation_candidate["p95_well_rmse"]),
    }
    confirmation_checks = {
        "gain_ge_0p05": confirmation_values["gain"] >= 0.05,
        "far_gain_nonnegative": confirmation_values["far_gain"] >= 0.0,
        "near_frozen_exact": abs(confirmation_values["near_degradation"]) <= 1e-5,
        "p95_not_worse_by_gt_0p10": confirmation_values["p95_gain"] >= -0.10,
    }
    output = matched[["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
    output["coefficient_regressor"] = prediction.astype(np.float32)
    output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
panel.to_csv(OUT_DIR / "well_feature_target_panel.csv", index=False)
strict_pass = bool(len(eligible) and confirmation_checks and all(confirmation_checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_well_coefficient_training_Fold3_discovery_confirmation",
    "sample_unit": "one complete physical well",
    "legal_features_only": True,
    "target": "V123 tail correction coefficients",
    "feature_count": len(feature_columns),
    "train_wells": int(train_mask.sum()),
    "selection_wells": len(discovery_wells),
    "confirmation_wells": len(confirmation_wells),
    "search_count": len(recipes),
    "selected_recipe": {
        "family": selected[0], "profile": selected[1],
        "dose": selected[2], "cap_ft": selected[3],
        "passed_discovery_constraints": bool(len(eligible)),
    },
    "discovery": {
        "gain": float(selected_row["gain"]), "far_gain": float(selected_row["far_gain"]),
        "near_degradation": float(selected_row["near_degradation"]),
        "p95_gain": float(selected_row["p95_gain"]),
    },
    "confirmation": confirmation_values,
    "confirmation_checks": confirmation_checks,
    "strict_pass": strict_pass,
    "fold2_used_for_training": True,
    "fold1_used_for_training": True,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> cross-fit all available folds and freeze a deployment coefficient model"
        if strict_pass else
        "FAIL -> low-dimensional oracle coefficients are not legally observable; replace trajectory generation"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV124 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV124 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV124 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
