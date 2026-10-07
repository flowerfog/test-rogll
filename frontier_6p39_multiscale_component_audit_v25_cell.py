"""Decompose the v23 multiscale correction before deciding whether to deploy it.

The v23 correction improves Fold 3, Fold 2, and Fold 1, but a within-well roll
retains part of the gain because that control preserves each well's constant
correction.  This cell separates the deployable correction into constant,
affine-trend, low-frequency, and local-shape components.  Fold 3 selects one
frozen component/dose/risk recipe; Fold 2 confirms it and Fold 1 is final.

No model or PF is rerun, no hidden prediction is inspected, and no Fold 1/2
label participates in recipe selection.
"""

from pathlib import Path
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE_DIR = OOF_ROOT / "frontier_6p39_multiscale_endpoint_fusion_v23"
SOURCE_PATH = SOURCE_DIR / "fold123_multiscale_oof.parquet"
OUT_DIR = OOF_ROOT / "frontier_6p39_multiscale_component_audit_v25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
COMPONENTS = (
    "full",
    "well_mean",
    "well_median",
    "well_affine",
    "smooth128",
    "smooth256",
    "smooth512",
    "smooth1024",
    "demeaned_full",
)
DOSES = (0.25, 0.50, 0.75, 1.00, 1.25)
CAPS_FT = (2.0, 4.0, 8.0, 12.0, np.inf)
WELL_MOVE_RMS_LIMITS = (2.0, 4.0, 6.0, 8.0, np.inf)
started = time.time()


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    well_rmse = np.sqrt(
        pd.Series(error * error).groupby(frame["well"], sort=False).mean().to_numpy()
    )
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    bands = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in bands.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _component(raw, distance, name):
    raw = np.asarray(raw, np.float64)
    distance = np.asarray(distance, np.float64)
    if name == "full":
        return raw.copy()
    if name == "well_mean":
        return np.full(len(raw), float(np.mean(raw)), np.float64)
    if name == "well_median":
        return np.full(len(raw), float(np.median(raw)), np.float64)
    if name == "well_affine":
        centre = float(np.mean(distance))
        scale = float(max(np.std(distance), 1.0))
        x = np.column_stack((np.ones(len(raw)), (distance - centre) / scale))
        coefficients, *_ = np.linalg.lstsq(x, raw, rcond=None)
        return x @ coefficients
    if name == "demeaned_full":
        return raw - float(np.mean(raw))
    if name.startswith("smooth"):
        scale_ft = float(name.removeprefix("smooth"))
        steps = np.diff(distance)
        positive = steps[np.isfinite(steps) & (steps > 0.0)]
        spacing = float(np.median(positive)) if len(positive) else 1.0
        spacing = float(np.clip(spacing, 0.25, 4.0))
        window = max(3, int(round(scale_ft / spacing)))
        return uniform_filter1d(raw, size=window, mode="nearest")
    raise KeyError(name)


if not SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Missing {SOURCE_PATH}. Run frontier_6p39_multiscale_endpoint_fusion_v23 first."
    )

columns = [
    "id", "well", "row_idx", "fold", "tvt_true", "md_since", "parent",
    "raw_multiscale_correction", "multiscale_correction", "multiscale_prediction",
]
frame = pd.read_parquet(SOURCE_PATH, columns=columns)
frame["well"] = frame["well"].astype(str)
frame = frame[frame["fold"].isin((1, 2, 3))].copy()
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
required = ["tvt_true", "md_since", "parent", "multiscale_correction"]
if frame[required].isna().any().any():
    raise RuntimeError("v23 Fold1/2/3 OOF contains non-finite required values")
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in v23 OOF")

parent = frame["parent"].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
correction = frame["multiscale_correction"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
distance = frame["md_since"].to_numpy(np.float64)
groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}

print(
    "COMPONENT AUDIT CONFIG",
    {
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "components": COMPONENTS,
        "selection_fold": SELECTION_FOLD,
    },
    flush=True,
)

# Per-well sufficient statistics make the Fold3 scan inexpensive and preserve
# exact pooled SSE and per-well tail metrics.
stat_rows = []
component_cache = {}
for number, (well, indices) in enumerate(groups.items(), 1):
    local_error = parent[indices] - truth[indices]
    local_distance = distance[indices]
    local_raw = correction[indices]
    held_fold = int(fold[indices[0]])
    base = {
        "well": well,
        "fold": held_fold,
        "rows": int(len(indices)),
        "base_sse": float(np.dot(local_error, local_error)),
    }
    for component_name in COMPONENTS:
        move = _component(local_raw, local_distance, component_name)
        component_cache[(well, component_name)] = move.astype(np.float32)
        for cap_ft in CAPS_FT:
            clipped = move if not np.isfinite(cap_ft) else np.clip(move, -cap_ft, cap_ft)
            row = {
                **base,
                "component": component_name,
                "cap_ft": float(cap_ft),
                "cross": float(np.dot(local_error, clipped)),
                "delta2": float(np.dot(clipped, clipped)),
                "move_rms": float(np.sqrt(np.mean(clipped * clipped))),
            }
            for band_name, mask in {
                "near": local_distance <= 500.0,
                "far": local_distance > 2000.0,
            }.items():
                row[f"{band_name}_rows"] = int(mask.sum())
                row[f"{band_name}_base_sse"] = float(
                    np.dot(local_error[mask], local_error[mask])
                )
                row[f"{band_name}_cross"] = float(
                    np.dot(local_error[mask], clipped[mask])
                )
                row[f"{band_name}_delta2"] = float(
                    np.dot(clipped[mask], clipped[mask])
                )
            stat_rows.append(row)
    if number % 100 == 0 or number == len(groups):
        print(f"COMPONENT stats {number}/{len(groups)}", flush=True)
stats = pd.DataFrame(stat_rows)


def _evaluate(recipe, held_folds):
    held_folds = tuple(int(value) for value in held_folds)
    table = stats[
        stats["fold"].isin(held_folds)
        & stats["component"].eq(str(recipe["component"]))
        & stats["cap_ft"].eq(float(recipe["cap_ft"]))
    ].copy()
    if table["well"].nunique() == 0:
        raise RuntimeError(f"No stats for {recipe} / {held_folds}")
    dose = float(recipe["dose"])
    selected = table["move_rms"].to_numpy() <= float(recipe["well_move_rms_limit"])
    cross = table["cross"].to_numpy(np.float64)
    delta2 = table["delta2"].to_numpy(np.float64)
    base_sse = table["base_sse"].to_numpy(np.float64)
    well_sse = base_sse.copy()
    well_sse[selected] += (
        2.0 * dose * cross[selected] + dose * dose * delta2[selected]
    )
    rows_by_well = table["rows"].to_numpy(np.float64)
    well_rmse = np.sqrt(np.maximum(well_sse, 0.0) / rows_by_well)
    total_rows = int(rows_by_well.sum())
    output = {
        "rows": total_rows,
        "wells": int(len(table)),
        "selected_wells": int(selected.sum()),
        "exact_sse": float(well_sse.sum()),
        "rmse": float(np.sqrt(max(float(well_sse.sum()), 0.0) / total_rows)),
        "parent_rmse": float(np.sqrt(float(base_sse.sum()) / total_rows)),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    parent_well_rmse = np.sqrt(base_sse / rows_by_well)
    output["parent_p95_well_rmse"] = float(np.quantile(parent_well_rmse, 0.95))
    for band_name in ("near", "far"):
        band_rows_by_well = table[f"{band_name}_rows"].to_numpy(np.float64)
        band_base = table[f"{band_name}_base_sse"].to_numpy(np.float64)
        band_cross = table[f"{band_name}_cross"].to_numpy(np.float64)
        band_delta2 = table[f"{band_name}_delta2"].to_numpy(np.float64)
        band_sse = band_base.copy()
        band_sse[selected] += (
            2.0 * dose * band_cross[selected]
            + dose * dose * band_delta2[selected]
        )
        valid_rows = float(band_rows_by_well.sum())
        output[f"{band_name}_rmse"] = float(
            np.sqrt(max(float(band_sse.sum()), 0.0) / valid_rows)
        )
        output[f"{band_name}_parent_rmse"] = float(
            np.sqrt(float(band_base.sum()) / valid_rows)
        )
    output["gain"] = output["parent_rmse"] - output["rmse"]
    output["p95_degradation"] = (
        output["p95_well_rmse"] - output["parent_p95_well_rmse"]
    )
    output["near_degradation"] = output["near_rmse"] - output["near_parent_rmse"]
    output["far_degradation"] = output["far_rmse"] - output["far_parent_rmse"]
    output["risk_pass"] = bool(
        output["selected_wells"] >= 20
        and output["p95_degradation"] <= 0.10
        and output["near_degradation"] <= 0.02
        and output["far_degradation"] <= 0.02
    )
    output["selection_score"] = (
        output["rmse"]
        + 0.50 * max(output["p95_degradation"], 0.0)
        + 0.25 * max(output["near_degradation"], 0.0)
        + 0.25 * max(output["far_degradation"], 0.0)
    )
    return output


scan_rows = []
for component_name, dose, cap_ft, rms_limit in itertools.product(
    COMPONENTS, DOSES, CAPS_FT, WELL_MOVE_RMS_LIMITS
):
    recipe = {
        "component": component_name,
        "dose": float(dose),
        "cap_ft": float(cap_ft),
        "well_move_rms_limit": float(rms_limit),
    }
    scan_rows.append({**recipe, **_evaluate(recipe, (SELECTION_FOLD,))})
scan = pd.DataFrame(scan_rows).sort_values(
    ["risk_pass", "selection_score", "rmse", "p95_well_rmse"],
    ascending=[False, True, True, True],
    kind="stable",
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_component_scan.csv", index=False)
selected = {
    key: scan.iloc[0][key].item() if hasattr(scan.iloc[0][key], "item") else scan.iloc[0][key]
    for key in ("component", "dose", "cap_ft", "well_move_rms_limit")
}


def _prediction(recipe):
    output = parent.copy()
    active = np.zeros(len(frame), bool)
    for well, indices in groups.items():
        move = component_cache[(well, str(recipe["component"]))].astype(np.float64)
        cap_ft = float(recipe["cap_ft"])
        if np.isfinite(cap_ft):
            move = np.clip(move, -cap_ft, cap_ft)
        if float(np.sqrt(np.mean(move * move))) > float(recipe["well_move_rms_limit"]):
            continue
        output[indices] += float(recipe["dose"]) * move
        active[indices] = True
    return output, active


candidate, active = _prediction(selected)
summary_rows = []
for held_fold, role in (
    (SELECTION_FOLD, "selection"),
    (CONFIRMATION_FOLD, "confirmation"),
    (FINAL_FOLD, "final"),
):
    mask = fold == held_fold
    local = frame.loc[mask].reset_index(drop=True)
    summary_rows.extend([
        {"fold": held_fold, "role": role, **_metric(local, parent[mask], "parent")},
        {"fold": held_fold, "role": role, **_metric(local, candidate[mask], "component")},
    ])
primary_mask = np.isin(fold, (CONFIRMATION_FOLD, FINAL_FOLD))
primary_frame = frame.loc[primary_mask].reset_index(drop=True)
summary_rows.extend([
    {"fold": 0, "role": "fold1_plus_fold2_primary", **_metric(
        primary_frame, parent[primary_mask], "parent"
    )},
    {"fold": 0, "role": "fold1_plus_fold2_primary", **_metric(
        primary_frame, candidate[primary_mask], "component"
    )},
])
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

# Attribute the selected correction.  The preserved-mean roll is the correct
# placement control: unlike v23's raw roll, it reports separately how much of
# the gain comes from a legal well-level offset and from along-well placement.
selected_move = candidate - parent
mean_move = np.zeros(len(frame), np.float64)
shape_move = np.zeros(len(frame), np.float64)
rolled_shape_move = np.zeros(len(frame), np.float64)
for well, indices in groups.items():
    local = selected_move[indices]
    centre = float(np.mean(local))
    shape = local - centre
    mean_move[indices] = centre
    shape_move[indices] = shape
    rolled_shape_move[indices] = centre + np.roll(shape, max(1, len(indices) // 2))

control_rows = []
for held_fold, role in ((CONFIRMATION_FOLD, "confirmation"), (FINAL_FOLD, "final")):
    mask = fold == held_fold
    local = frame.loc[mask].reset_index(drop=True)
    endpoints = {
        "parent": parent[mask],
        "component": candidate[mask],
        "mean_only": parent[mask] + mean_move[mask],
        "shape_only": parent[mask] + shape_move[mask],
        "roll_shape_preserve_mean": parent[mask] + rolled_shape_move[mask],
        "opposite_move": parent[mask] - selected_move[mask],
    }
    for endpoint, prediction in endpoints.items():
        control_rows.append({
            "fold": held_fold,
            "role": role,
            **_metric(local, prediction, endpoint),
        })
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "matched_controls.csv", index=False)


def _value(table, held_fold, endpoint, column):
    row = table[table["fold"].eq(held_fold) & table["endpoint"].eq(endpoint)]
    return float(row.iloc[0][column])


fold_gain = {
    held_fold: _value(summary, held_fold, "parent", "rmse")
    - _value(summary, held_fold, "component", "rmse")
    for held_fold in (SELECTION_FOLD, CONFIRMATION_FOLD, FINAL_FOLD, 0)
}
far_change = {
    held_fold: _value(summary, held_fold, "component", "far_gt2000_rmse")
    - _value(summary, held_fold, "parent", "far_gt2000_rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
p95_change = {
    held_fold: _value(summary, held_fold, "component", "p95_well_rmse")
    - _value(summary, held_fold, "parent", "p95_well_rmse")
    for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD)
}
attribution = {}
for held_fold in (CONFIRMATION_FOLD, FINAL_FOLD):
    parent_rmse = _value(controls, held_fold, "parent", "rmse")
    attribution[str(held_fold)] = {
        endpoint: float(parent_rmse - _value(controls, held_fold, endpoint, "rmse"))
        for endpoint in (
            "component", "mean_only", "shape_only", "roll_shape_preserve_mean"
        )
    }

gates = {
    "fold2_gain_ge_0p08": bool(fold_gain[CONFIRMATION_FOLD] >= 0.08),
    "fold1_gain_ge_0p08": bool(fold_gain[FINAL_FOLD] >= 0.08),
    "fold12_gain_ge_0p08": bool(fold_gain[0] >= 0.08),
    "fold2_far_degradation_le_0p02": bool(far_change[CONFIRMATION_FOLD] <= 0.02),
    "fold1_far_degradation_le_0p02": bool(far_change[FINAL_FOLD] <= 0.02),
    "fold2_p95_degradation_le_0p10": bool(p95_change[CONFIRMATION_FOLD] <= 0.10),
    "fold1_p95_degradation_le_0p10": bool(p95_change[FINAL_FOLD] <= 0.10),
}

row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["parent"] = parent.astype(np.float32)
row_output["v23_correction"] = correction.astype(np.float32)
row_output["selected_component_move"] = selected_move.astype(np.float32)
row_output["component_prediction"] = candidate.astype(np.float32)
row_output["active"] = active
row_output.to_parquet(
    OUT_DIR / "component_oof.parquet", index=False, compression="zstd"
)


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


verdict = {
    "experiment": "frontier_6p39_multiscale_component_audit_v25",
    "stage": "Fold3_component_selection_Fold2_confirmation_Fold1_final",
    "source": str(SOURCE_PATH),
    "zero_refit": True,
    "hidden_predictions_observed": False,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "Fold1_tuning_or_search_count": 0,
        "Fold2_tuning_or_search_count": 0,
    },
    "selected": {key: _json_scalar(value) for key, value in selected.items()},
    "selection_search_count": int(len(scan)),
    "fold_gain": {str(key): float(value) for key, value in fold_gain.items()},
    "far_change": {str(key): float(value) for key, value in far_change.items()},
    "p95_change": {str(key): float(value) for key, value in p95_change.items()},
    "gain_attribution": attribution,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> integrate v23 bundle plus the frozen component transform"
        if all(gates.values())
        else "FAIL -> retain v23 only as a research endpoint; do not submit"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

display_columns = [
    "fold", "role", "endpoint", "rows", "wells", "rmse", "mae",
    "median_well_rmse", "p95_well_rmse", "near_0_500_rmse",
    "mid_500_1000_rmse", "mid_1000_2000_rmse", "far_gt2000_rmse",
]
print("COMPONENT FOLD3 TOP 30")
print(scan.head(30).to_string(index=False), flush=True)
print("\nCOMPONENT SUMMARY")
print(summary[display_columns].to_string(index=False), flush=True)
print("\nCOMPONENT ATTRIBUTION CONTROLS")
print(controls[display_columns].to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
