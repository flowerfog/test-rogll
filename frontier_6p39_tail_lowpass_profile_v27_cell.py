"""Low-frequency and distance-profile extensions of the frozen tail Neighbor.

Run on AutoDL after frontier_6p39_tail_only_neighbor_v26_cell.py.  This cell
does not rebuild Neighbor paths.  It consumes the aligned v26 OOF, selects on
Fold 3, confirms once on Fold 2, and uses Fold 1 only as the final check.
"""

from pathlib import Path
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE = OOF_ROOT / "frontier_6p39_tail_only_neighbor_v26" / "tail_neighbor_oof.parquet"
OUT_DIR = OOF_ROOT / "frontier_6p39_tail_lowpass_profile_v27"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
START_FT = 1000.0
TRANSITION_FT = 250.0
LOWPASS_WINDOWS_FT = (0.0, 256.0, 512.0, 768.0)
CAPS_FT = (12.0, 24.0)

# Values are the dose at 1250, 2000, and 3000 ft.  Between knots the dose is
# linearly interpolated; the 1000-1250 ft smoothstep still guarantees an exact
# frozen prefix and a C1 activation.
DOSE_PROFILES = {
    "flat_010": (0.10, 0.10, 0.10),
    "flat_015": (0.15, 0.15, 0.15),
    "flat_020": (0.20, 0.20, 0.20),
    "flat_025": (0.25, 0.25, 0.25),
    "flat_030": (0.30, 0.30, 0.30),
    "ramp_010_020_020": (0.10, 0.20, 0.20),
    "ramp_010_020_030": (0.10, 0.20, 0.30),
    "ramp_015_020_025": (0.15, 0.20, 0.25),
    "ramp_020_025_030": (0.20, 0.25, 0.30),
}
FROZEN_RECIPE = {
    "lowpass_window_ft": 0.0,
    "cap_ft": 24.0,
    "dose_profile": "flat_020",
}
PROPOSED_RECIPE = {
    "lowpass_window_ft": 512.0,
    "cap_ft": 24.0,
    "dose_profile": "ramp_010_020_030",
}
started = time.time()


def _smoothstep(distance):
    t = np.clip((distance - START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dose(distance, profile_name):
    values = np.asarray(DOSE_PROFILES[str(profile_name)], np.float64)
    return np.interp(
        np.asarray(distance, np.float64),
        np.asarray((START_FT + TRANSITION_FT, 2000.0, 3000.0)),
        values,
        left=values[0],
        right=values[-1],
    )


def _centred_lowpass(md, values, valid, window_ft):
    """NaN-aware centred box filter in measured-distance coordinates."""
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    if float(window_ft) <= 0.0:
        return np.where(valid, values, 0.0), valid.copy()
    md = np.asarray(md, np.float64)
    if len(md) == 0 or np.any(np.diff(md) < 0.0):
        raise RuntimeError("Low-pass input MD must be sorted")
    half = 0.5 * float(window_ft)
    left = np.searchsorted(md, md - half, side="left")
    right = np.searchsorted(md, md + half, side="right")
    weighted = np.where(valid, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(weighted, dtype=np.float64)))
    count = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    sums = csum[right] - csum[left]
    counts = count[right] - count[left]
    output_valid = counts > 0
    output = np.zeros(len(values), np.float64)
    output[output_valid] = sums[output_valid] / counts[output_valid]
    return output, output_valid


def _metric(local, prediction, endpoint):
    truth = local["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if len(error) != len(local) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    well_mse = pd.Series(error * error).groupby(local["well"], sort=False).mean()
    result = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
    }
    distance = local["md_since"].to_numpy(np.float64)
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


if not SOURCE.is_file():
    raise FileNotFoundError(
        f"Missing {SOURCE}. Run frontier_6p39_tail_only_neighbor_v26_cell.py first."
    )

frame = pd.read_parquet(SOURCE)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since", "parent",
    "neighbor_absolute", "old_frozen_tail",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"v26 OOF missing columns: {missing}")
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate OOF ids")

parent = frame["parent"].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
neighbor = frame["neighbor_absolute"].to_numpy(np.float64)
old_prediction = frame["old_frozen_tail"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
raw_delta = neighbor - parent
raw_valid = np.isfinite(raw_delta)
groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}

print(
    "TAIL-LOWPASS CONFIG",
    {
        "rows": len(frame),
        "wells": len(groups),
        "windows_ft": LOWPASS_WINDOWS_FT,
        "caps_ft": CAPS_FT,
        "profiles": tuple(DOSE_PROFILES),
        "selection_fold": SELECTION_FOLD,
    },
    flush=True,
)

# Cache each low-pass delta once.  This is also the exact object used for the
# final prediction, so the scan and confirmation cannot drift apart.
filtered = {
    window: np.zeros(len(frame), np.float32) for window in LOWPASS_WINDOWS_FT
}
filtered_valid = {
    window: np.zeros(len(frame), bool) for window in LOWPASS_WINDOWS_FT
}
for number, (well, indices) in enumerate(groups.items(), 1):
    local_md = distance[indices]
    for window in LOWPASS_WINDOWS_FT:
        values, valid = _centred_lowpass(
            local_md, raw_delta[indices], raw_valid[indices], window
        )
        filtered[window][indices] = values.astype(np.float32)
        filtered_valid[window][indices] = valid
    if number % 100 == 0 or number == len(groups):
        print(f"TAIL-LOWPASS filter {number}/{len(groups)}", flush=True)


def _prediction(recipe, shuffled=False):
    output = parent.copy()
    active = np.zeros(len(frame), bool)
    window = float(recipe["lowpass_window_ft"])
    cap = float(recipe["cap_ft"])
    profile = str(recipe["dose_profile"])
    for well, indices in groups.items():
        valid = filtered_valid[window][indices].copy()
        if valid.mean() < 0.80:
            continue
        delta = filtered[window][indices].astype(np.float64)
        if shuffled:
            shift = max(1, len(indices) // 3)
            delta = np.roll(delta, shift)
            valid = np.roll(valid, shift)
        delta = np.where(valid, np.clip(delta, -cap, cap), 0.0)
        local_distance = distance[indices]
        move = _smoothstep(local_distance) * _dose(local_distance, profile) * delta
        output[indices] += move
        active[indices] = valid & (local_distance > START_FT)
    return output, active


def _fold_metric(prediction, held_fold, endpoint):
    mask = fold == int(held_fold)
    return _metric(frame.loc[mask].reset_index(drop=True), prediction[mask], endpoint)


old_fold3 = _fold_metric(old_prediction, SELECTION_FOLD, "old_frozen_tail")
scan_rows = []
for window, cap, profile in itertools.product(
    LOWPASS_WINDOWS_FT, CAPS_FT, DOSE_PROFILES
):
    recipe = {
        "lowpass_window_ft": float(window),
        "cap_ft": float(cap),
        "dose_profile": str(profile),
    }
    prediction, active = _prediction(recipe)
    metric = _fold_metric(prediction, SELECTION_FOLD, "candidate")
    metric.update(recipe)
    metric["gain_vs_old"] = float(old_fold3["rmse"] - metric["rmse"])
    metric["p95_change_vs_old"] = float(
        metric["p95_well_rmse"] - old_fold3["p95_well_rmse"]
    )
    metric["active_rows"] = int(active[fold == SELECTION_FOLD].sum())
    metric["risk_pass"] = bool(
        metric["active_rows"] > 0
        and metric["p95_change_vs_old"] <= 0.10
    )
    metric["selection_score"] = float(
        metric["rmse"] + 0.5 * max(metric["p95_change_vs_old"], 0.0)
    )
    scan_rows.append(metric)

scan = pd.DataFrame(scan_rows).sort_values(
    ["risk_pass", "selection_score", "rmse"],
    ascending=[False, True, True],
    kind="stable",
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "fold3_scan.csv", index=False)
selected = {
    "lowpass_window_ft": float(scan.iloc[0]["lowpass_window_ft"]),
    "cap_ft": float(scan.iloc[0]["cap_ft"]),
    "dose_profile": str(scan.iloc[0]["dose_profile"]),
}

selected_prediction, selected_active = _prediction(selected)
proposed_prediction, proposed_active = _prediction(PROPOSED_RECIPE)
shuffle_prediction, _ = _prediction(selected, shuffled=True)

summary_rows = []
for held_fold, role in (
    (SELECTION_FOLD, "selection"),
    (CONFIRMATION_FOLD, "confirmation"),
    (FINAL_FOLD, "final"),
):
    mask = fold == held_fold
    local = frame.loc[mask].reset_index(drop=True)
    endpoints = {
        "parent": parent[mask],
        "old_frozen_tail": old_prediction[mask],
        "proposed_512_ramp": proposed_prediction[mask],
        "selected": selected_prediction[mask],
        "shuffled_selected": shuffle_prediction[mask],
        "opposite_selected": (2.0 * parent - selected_prediction)[mask],
    }
    for endpoint, prediction in endpoints.items():
        summary_rows.append({
            "fold": held_fold,
            "role": role,
            **_metric(local, prediction, endpoint),
        })

mask12 = np.isin(fold, (CONFIRMATION_FOLD, FINAL_FOLD))
local12 = frame.loc[mask12].reset_index(drop=True)
for endpoint, prediction in {
    "parent": parent[mask12],
    "old_frozen_tail": old_prediction[mask12],
    "proposed_512_ramp": proposed_prediction[mask12],
    "selected": selected_prediction[mask12],
}.items():
    summary_rows.append({
        "fold": 0,
        "role": "fold1_and_fold2",
        **_metric(local12, prediction, endpoint),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)


def _value(held_fold, endpoint, column="rmse"):
    row = summary[summary["fold"].eq(held_fold) & summary["endpoint"].eq(endpoint)]
    return float(row.iloc[0][column])


selected_gain_vs_parent = {
    held: _value(held, "parent") - _value(held, "selected")
    for held in (3, 2, 1, 0)
}
selected_gain_vs_old = {
    held: _value(held, "old_frozen_tail") - _value(held, "selected")
    for held in (3, 2, 1, 0)
}
proposed_gain_vs_old = {
    held: _value(held, "old_frozen_tail") - _value(held, "proposed_512_ramp")
    for held in (3, 2, 1, 0)
}
far_gain_vs_old = {
    held: _value(held, "old_frozen_tail", "far_gt2000_rmse")
    - _value(held, "selected", "far_gt2000_rmse")
    for held in (2, 1)
}
p95_change_vs_old = {
    held: _value(held, "selected", "p95_well_rmse")
    - _value(held, "old_frozen_tail", "p95_well_rmse")
    for held in (2, 1)
}
frozen_change = float(np.max(np.abs(
    selected_prediction[distance <= START_FT] - parent[distance <= START_FT]
)))
shuffle_gain = {
    held: _value(held, "parent") - _value(held, "shuffled_selected")
    for held in (2, 1)
}

gates = {
    "first_1000ft_exactly_frozen": bool(frozen_change <= 1e-12),
    "fold2_positive_vs_parent": bool(selected_gain_vs_parent[2] > 0.0),
    "fold1_positive_vs_parent": bool(selected_gain_vs_parent[1] > 0.0),
    "fold2_nonworse_than_old_by_0p02": bool(selected_gain_vs_old[2] >= -0.02),
    "fold1_nonworse_than_old_by_0p02": bool(selected_gain_vs_old[1] >= -0.02),
    "fold12_gain_over_old_ge_0p02": bool(selected_gain_vs_old[0] >= 0.02),
    "fold2_far_nonworse_than_old": bool(far_gain_vs_old[2] >= -0.01),
    "fold1_far_nonworse_than_old": bool(far_gain_vs_old[1] >= -0.01),
    "fold2_p95_change_le_0p10": bool(p95_change_vs_old[2] <= 0.10),
    "fold1_p95_change_le_0p10": bool(p95_change_vs_old[1] <= 0.10),
    "shuffle_does_not_match_fold2_gain": bool(
        shuffle_gain[2] <= 0.50 * max(selected_gain_vs_parent[2], 1e-9)
    ),
}

row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["parent"] = parent.astype(np.float32)
row_output["old_frozen_tail"] = old_prediction.astype(np.float32)
row_output["proposed_512_ramp"] = proposed_prediction.astype(np.float32)
row_output["selected"] = selected_prediction.astype(np.float32)
row_output["selected_active"] = selected_active
row_output.to_parquet(OUT_DIR / "tail_lowpass_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "frontier_6p39_tail_lowpass_profile_v27",
    "stage": "Fold3_lowpass_profile_selection_Fold2_confirmation_Fold1_final",
    "source": str(SOURCE),
    "hidden_predictions_observed": False,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "fold2_search_count": 0,
        "fold1_search_count": 0,
    },
    "frozen_recipe": FROZEN_RECIPE,
    "proposed_recipe": PROPOSED_RECIPE,
    "selected_recipe": selected,
    "selection_search_count": int(len(scan)),
    "selected_gain_vs_parent": {str(k): float(v) for k, v in selected_gain_vs_parent.items()},
    "selected_gain_vs_old": {str(k): float(v) for k, v in selected_gain_vs_old.items()},
    "proposed_gain_vs_old": {str(k): float(v) for k, v in proposed_gain_vs_old.items()},
    "far_gain_vs_old": {str(k): float(v) for k, v in far_gain_vs_old.items()},
    "p95_change_vs_old": {str(k): float(v) for k, v in p95_change_vs_old.items()},
    "maximum_frozen_change": frozen_change,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> replace the raw frozen tail delta with the selected low-pass/profile recipe"
        if all(gates.values())
        else "FAIL -> retain the deployed raw frozen tail recipe"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nFOLD3 LOWPASS/PROFILE TOP 20", flush=True)
print(scan.head(20).to_string(index=False), flush=True)
print("\nTAIL LOWPASS SUMMARY", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
