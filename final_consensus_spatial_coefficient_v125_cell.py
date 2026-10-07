"""V125: spatial interpolation of frozen top-33 tail coefficients.

Unlike historical neighbour curve transfer or early-base offset kriging, this
uses only low-dimensional V123 residual coefficients from complete reference
wells. Fold1+2 are references, 24 Fold3 wells select a frozen recipe, and the
remaining 130 Fold3 wells confirm once.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "final_consensus_spatial_coefficient_v125"
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
THETA = 2.2166
TRUTH = "tvt_true"
BASE = "consensus_center"
PROFILES = {
    "tail_datum": ["tail_datum_ft"],
    "tail_affine": ["tail_affine_intercept_ft", "tail_affine_slope_ft_per_1000ft"],
}
GEOMETRIES = ("euclidean", "anisotropic")
POINTS = ("anchor", "midpoint")
KS = (4, 8, 16)
POWERS = (1.0, 2.0)
DOSES = (0.25, 0.50)
CAP_FT = 4.0


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
    module = types.ModuleType("final_consensus_v125_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _basis(profile, distance):
    distance = np.asarray(distance, np.float64)
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    if profile == "tail_datum":
        return taper[:, None]
    x = (distance - np.mean(distance)) / 1000.0
    if profile == "tail_affine":
        return np.column_stack([taper, taper * x])
    raise KeyError(profile)


def _distance(query, reference, geometry):
    delta = np.asarray(reference, np.float64) - np.asarray(query, np.float64)[None, :]
    if geometry == "euclidean":
        return np.sqrt(np.sum(delta * delta, axis=1))
    if geometry == "anisotropic":
        cosine, sine = np.cos(THETA), np.sin(THETA)
        along = cosine * delta[:, 0] + sine * delta[:, 1]
        across = -sine * delta[:, 0] + cosine * delta[:, 1]
        return np.sqrt(along * along + (2.0 * across) ** 2)
    raise KeyError(geometry)


def _idw(query, reference_xy, reference_target, geometry, k, power):
    distance = _distance(query, reference_xy, geometry)
    order = np.argsort(distance, kind="stable")[: int(k)]
    chosen_distance = np.maximum(distance[order], 40.0)
    weight = 1.0 / np.power(chosen_distance, float(power))
    weight /= weight.sum()
    return np.sum(weight[:, None] * reference_target[order], axis=0)


for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
R = _bootstrap_v110()
N = R._bootstrap_v81()
V = N.V
data_root = Path(V["DATA_ROOT"])

frame = pd.read_parquet(BASE_PATH)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
targets = pd.read_csv(TARGET_PATH, dtype={"well": str})
targets["well"] = targets["well"].astype(str)
target_columns = [column for columns in PROFILES.values() for column in columns]
targets = targets[["well", "fold", *target_columns]].copy()
if targets[target_columns].isna().any().any():
    raise RuntimeError("Missing V123 targets")

coordinate_rows = []
for number, well in enumerate(sorted(targets["well"]), 1):
    path = data_root / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(path, usecols=["MD", "X", "Y", "TVT_input"])
    horizontal = horizontal.sort_values("MD", kind="stable").reset_index(drop=True)
    known = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(np.float64)))
    if not len(known):
        raise RuntimeError(f"{well}: no visible anchor")
    anchor = int(known[-1])
    xy = horizontal[["X", "Y"]].to_numpy(np.float64)
    if not np.isfinite(xy).all():
        raise RuntimeError(f"{well}: non-finite coordinates")
    anchor_xy = xy[anchor]
    end_xy = xy[-1]
    coordinate_rows.append({
        "well": well,
        "anchor_x": float(anchor_xy[0]), "anchor_y": float(anchor_xy[1]),
        "midpoint_x": float((anchor_xy[0] + end_xy[0]) * 0.5),
        "midpoint_y": float((anchor_xy[1] + end_xy[1]) * 0.5),
    })
    if number % 100 == 0 or number == len(targets):
        print(f"V125 coordinates {number}/{len(targets)}", flush=True)
coordinates = pd.DataFrame(coordinate_rows)
panel = targets.merge(coordinates, on="well", how="left", validate="one_to_one")

train = panel[panel["fold"].isin(TRAIN_FOLDS)].copy()
fold3_wells = sorted(panel.loc[panel["fold"].eq(EVAL_FOLD), "well"])
discovery_wells = set(fold3_wells[:DISCOVERY_WELLS])
confirmation_wells = set(fold3_wells[DISCOVERY_WELLS:])
if len(train) != 310 or len(discovery_wells) != 24 or len(confirmation_wells) < 129:
    raise RuntimeError(
        f"Coverage train/discovery/confirmation={len(train)}/{len(discovery_wells)}/{len(confirmation_wells)}"
    )

coefficient_maps = {}
for geometry in GEOMETRIES:
    for point in POINTS:
        reference_xy = train[[f"{point}_x", f"{point}_y"]].to_numpy(np.float64)
        query_xy = panel[[f"{point}_x", f"{point}_y"]].to_numpy(np.float64)
        for profile, columns in PROFILES.items():
            reference_target = train[columns].to_numpy(np.float64)
            for k in KS:
                for power in POWERS:
                    values = np.vstack([
                        _idw(query, reference_xy, reference_target, geometry, k, power)
                        for query in query_xy
                    ])
                    coefficient_maps[(geometry, point, profile, k, power)] = dict(
                        zip(panel["well"], values)
                    )

recipes = [
    (geometry, point, profile, k, power, dose)
    for geometry in GEOMETRIES for point in POINTS for profile in PROFILES
    for k in KS for power in POWERS for dose in DOSES
]


def _evaluate(selected_wells, recipe, endpoint, coefficient_override=None):
    geometry, point, profile, k, power, dose = recipe
    key = (geometry, point, profile, k, power)
    coefficient_map = coefficient_maps[key] if coefficient_override is None else coefficient_override
    local_parts, prediction_parts = [], []
    for well in sorted(selected_wells):
        local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        distance = local["md_since"].to_numpy(np.float64)
        move = _basis(profile, distance) @ coefficient_map[well]
        move = np.clip(float(dose) * move, -CAP_FT, CAP_FT)
        prediction_parts.append(local[BASE].to_numpy(np.float64) + move)
        local_parts.append(local)
    matched = pd.concat(local_parts, ignore_index=True)
    prediction = np.concatenate(prediction_parts)
    return R._metric(matched, prediction, endpoint), matched, prediction


discovery_frame = frame[frame["well"].isin(discovery_wells)].copy()
discovery_base = R._metric(discovery_frame, discovery_frame[BASE].to_numpy(np.float64), BASE)
scan_rows = []
for recipe in recipes:
    result, _, _ = _evaluate(discovery_wells, recipe, "spatial_coefficient")
    geometry, point, profile, k, power, dose = recipe
    result.update({
        "geometry": geometry, "point": point, "profile": profile,
        "k": k, "power": power, "dose": dose,
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
    str(selected_row["geometry"]), str(selected_row["point"]), str(selected_row["profile"]),
    int(selected_row["k"]), float(selected_row["power"]), float(selected_row["dose"]),
)

summary_rows = [dict(discovery_base, stage="fold3_discovery")]
discovery_candidate, _, _ = _evaluate(discovery_wells, selected, "spatial_coefficient")
summary_rows.append(dict(discovery_candidate, stage="fold3_discovery"))
confirmation_values = None
checks = None
if len(eligible):
    confirmation_frame = frame[frame["well"].isin(confirmation_wells)].copy()
    confirmation_base = R._metric(
        confirmation_frame, confirmation_frame[BASE].to_numpy(np.float64), BASE
    )
    candidate, matched, prediction = _evaluate(
        confirmation_wells, selected, "spatial_coefficient"
    )
    key = selected[:5]
    original_map = coefficient_maps[key]
    ordered = sorted(confirmation_wells)
    rolled_values = np.roll(
        np.vstack([original_map[well] for well in ordered]), max(1, len(ordered) // 7), axis=0
    )
    rolled_map = dict(original_map)
    for well, value in zip(ordered, rolled_values):
        rolled_map[well] = value
    rolled, _, rolled_prediction = _evaluate(
        confirmation_wells, selected, "rolled_spatial_control", rolled_map
    )
    summary_rows.extend([
        dict(confirmation_base, stage="fold3_confirmation"),
        dict(candidate, stage="fold3_confirmation"),
        dict(rolled, stage="fold3_confirmation"),
    ])
    confirmation_values = {
        "gain": float(confirmation_base["rmse"] - candidate["rmse"]),
        "far_gain": float(confirmation_base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        "near_degradation": float(candidate["near_0_500_rmse"] - confirmation_base["near_0_500_rmse"]),
        "p95_gain": float(confirmation_base["p95_well_rmse"] - candidate["p95_well_rmse"]),
        "aligned_increment_vs_rolled": float(rolled["rmse"] - candidate["rmse"]),
    }
    checks = {
        "gain_ge_0p05": confirmation_values["gain"] >= 0.05,
        "far_gain_nonnegative": confirmation_values["far_gain"] >= 0.0,
        "near_frozen_exact": abs(confirmation_values["near_degradation"]) <= 1e-5,
        "p95_not_worse_by_gt_0p10": confirmation_values["p95_gain"] >= -0.10,
        "aligned_beats_rolled_ge_0p02": confirmation_values["aligned_increment_vs_rolled"] >= 0.02,
    }
    output = matched[["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
    output["spatial_coefficient"] = prediction.astype(np.float32)
    output["rolled_spatial_control"] = rolled_prediction.astype(np.float32)
    output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
strict_pass = bool(len(eligible) and checks and all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_spatial_coefficient_reference_Fold3_discovery_confirmation",
    "sample_unit": "one complete physical well",
    "changed_information": "X/Y spatial continuity of V123 tail coefficients",
    "curve_transfer": False,
    "train_reference_wells": len(train),
    "selection_wells": len(discovery_wells),
    "confirmation_wells": len(confirmation_wells),
    "search_count": len(recipes),
    "selected_recipe": {
        "geometry": selected[0], "point": selected[1], "profile": selected[2],
        "k": selected[3], "power": selected[4], "dose": selected[5], "cap_ft": CAP_FT,
        "passed_discovery_constraints": bool(len(eligible)),
    },
    "discovery": {
        "gain": float(selected_row["gain"]), "far_gain": float(selected_row["far_gain"]),
        "near_degradation": float(selected_row["near_degradation"]),
        "p95_gain": float(selected_row["p95_gain"]),
    },
    "confirmation": confirmation_values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> cross-fit spatial coefficients and build deployment lookup"
        if strict_pass else
        "FAIL -> remaining low-dimensional residual is not spatially observable; replace trajectory model"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV125 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV125 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV125 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
