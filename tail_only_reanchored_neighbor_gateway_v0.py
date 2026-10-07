"""Nested tail-only consumption of the fold-clean full-curve neighbor path.

The neighbor candidate is rebuilt by the validated event-aware gateway.  This
experiment freezes the first 1000 ft exactly, optionally reanchors the neighbor
delta at the start of a smooth transition, and only spends correction dose in
the distal part of the lateral.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "event_aware_neighbor_transfer_gateway_v0.py"
OUT_DIR = ROOT / "output" / "tail_only_reanchored_neighbor_gateway_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STARTS_FT = (1000.0, 1250.0, 1500.0, 1750.0, 2000.0)
TRANSITION_WIDTHS_FT = (250.0, 500.0)
REANCHOR_MODES = ("none", "value")
CAPS_FT = (24.0, 40.0)
DOSES = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75)
MAX_MID_DEGRADATION_FT = 0.02
SEED = 20260725


def smoothstep(distance: np.ndarray, start: float, width: float) -> np.ndarray:
    t = np.clip((distance - start) / width, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def strata(distance: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(distance), dtype=bool),
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }


def report(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    truth = frame["truth"].to_numpy(np.float64)
    error = prediction - truth
    distance = frame["distance"].to_numpy(np.float64)
    per_well = pd.DataFrame({"well": frame["well"], "e2": error**2}).groupby("well")["e2"].mean()
    output: dict[str, float | int] = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(per_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(per_well), 0.95)),
    }
    for name, mask in strata(distance).items():
        if name != "all":
            output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
    return output


def reanchor_by_well(
    raw: np.ndarray,
    coverage: np.ndarray,
    distance: np.ndarray,
    groups: list[np.ndarray],
    start: float,
    mode: str,
) -> np.ndarray:
    output = np.zeros(len(raw), dtype=np.float64)
    for index in groups:
        valid = coverage[index]
        if not valid.any():
            continue
        local_distance = distance[index]
        local_raw = raw[index]
        if mode == "value":
            usable = valid & np.isfinite(local_raw)
            if not usable.any():
                continue
            anchor = float(np.interp(start, local_distance[usable], local_raw[usable]))
        else:
            anchor = 0.0
        output[index[valid]] = local_raw[valid] - anchor
    return output


def shuffled_within_well(raw: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    output = raw.copy()
    for index in groups:
        if len(index) < 8:
            continue
        shift = max(len(index) // 3, 1)
        output[index] = np.roll(raw[index], shift)
    return output


started = time.time()
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)
source = SOURCE.read_text(encoding="utf-8")
marker = "def shuffled(values: np.ndarray, wells_array: np.ndarray) -> np.ndarray:"
if marker not in source:
    raise RuntimeError("Validated neighbor gateway marker not found")

# Reuse the exact fold-clean neighbor construction, stopping before its old dose scan.
namespace = {"__file__": str(SOURCE), "__name__": "tail_neighbor_builder_v0"}
exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), namespace)
data = namespace["data"].copy()

required = {"well", "fold", "truth", "base", "distance", "fullcurve_delta", "fullcurve_coverage"}
missing = sorted(required.difference(data.columns))
if missing:
    raise RuntimeError(f"Missing rebuilt columns: {missing}")
data = data.sort_values(["well", "row_idx"]).reset_index(drop=True)

truth = data["truth"].to_numpy(np.float64)
base = data["base"].to_numpy(np.float64)
error = base - truth
distance = data["distance"].to_numpy(np.float64)
folds = data["fold"].to_numpy(np.int8)
raw = data["fullcurve_delta"].to_numpy(np.float64)
coverage = data["fullcurve_coverage"].to_numpy(bool) & np.isfinite(raw)
groups = [group.index.to_numpy(np.int64) for _, group in data.groupby("well", sort=False)]
raw_shuffle = shuffled_within_well(raw, groups)
band_masks = strata(distance)
base_report = report(data, base)

scan_rows = []
correction_cache: dict[tuple[float, float, str, float], np.ndarray] = {}
for start in STARTS_FT:
    weight_by_width = {
        width: smoothstep(distance, start, width) for width in TRANSITION_WIDTHS_FT
    }
    for mode in REANCHOR_MODES:
        anchored = reanchor_by_well(raw, coverage, distance, groups, start, mode)
        for width, weight in weight_by_width.items():
            for cap in CAPS_FT:
                key = (start, width, mode, cap)
                correction = weight * np.clip(anchored, -cap, cap)
                correction_cache[key] = correction.astype(np.float32)
                for held_fold in range(1, 6):
                    train = folds != held_fold
                    valid = ~train
                    for dose in DOSES:
                        train_error = error[train] + dose * correction[train]
                        train_mid = train & band_masks["mid_1000_2000"]
                        base_mid = float(np.sqrt(np.mean(error[train_mid] ** 2)))
                        candidate_mid = float(np.sqrt(np.mean((error[train_mid] + dose * correction[train_mid]) ** 2)))
                        scan_rows.append({
                            "held_fold": held_fold,
                            "start_ft": start,
                            "transition_width_ft": width,
                            "reanchor": mode,
                            "cap_ft": cap,
                            "dose": dose,
                            "train_rmse": float(np.sqrt(np.mean(train_error**2))),
                            "train_mid_degradation": candidate_mid - base_mid,
                            "valid_base_rmse": float(np.sqrt(np.mean(error[valid] ** 2))),
                        })

scan = pd.DataFrame(scan_rows)
scan.to_csv(OUT_DIR / "nested_scan.csv", index=False)

prediction = base.copy()
shuffle_prediction = base.copy()
selected_rows = []
for held_fold in range(1, 6):
    candidates = scan[scan["held_fold"].eq(held_fold)].copy()
    feasible = candidates[candidates["train_mid_degradation"] <= MAX_MID_DEGRADATION_FT]
    if feasible.empty:
        feasible = candidates[candidates["dose"].eq(0.0)]
    chosen = feasible.sort_values(
        ["train_rmse", "dose", "start_ft", "transition_width_ft", "cap_ft"]
    ).iloc[0]
    key = (
        float(chosen["start_ft"]),
        float(chosen["transition_width_ft"]),
        str(chosen["reanchor"]),
        float(chosen["cap_ft"]),
    )
    valid = folds == held_fold
    correction = correction_cache[key].astype(np.float64)
    prediction[valid] = base[valid] + float(chosen["dose"]) * correction[valid]

    shuffled_anchored = reanchor_by_well(
        raw_shuffle, coverage, distance, groups, float(chosen["start_ft"]), str(chosen["reanchor"])
    )
    shuffled_correction = smoothstep(
        distance, float(chosen["start_ft"]), float(chosen["transition_width_ft"])
    ) * np.clip(shuffled_anchored, -float(chosen["cap_ft"]), float(chosen["cap_ft"]))
    shuffle_prediction[valid] = base[valid] + float(chosen["dose"]) * shuffled_correction[valid]

    valid_report = report(data.loc[valid].reset_index(drop=True), prediction[valid])
    selected_rows.append({
        **chosen.to_dict(),
        "valid_rmse": valid_report["rmse"],
        "valid_gain": float(chosen["valid_base_rmse"] - valid_report["rmse"]),
        "valid_far_rmse": valid_report["far_gt2000_rmse"],
    })

selected = pd.DataFrame(selected_rows)
selected.to_csv(OUT_DIR / "nested_selected.csv", index=False)
candidate_report = report(data, prediction)
shuffle_report = report(data, shuffle_prediction)

# A stricter diagnostic arm changes no row at or before 2000 ft.  It is not
# allowed to influence the primary selection above.
far_only_prediction = base.copy()
far_only_selected_rows = []
for held_fold in range(1, 6):
    candidates = scan[
        scan["held_fold"].eq(held_fold) & scan["start_ft"].eq(2000.0)
    ].copy()
    chosen = candidates.sort_values(
        ["train_rmse", "dose", "transition_width_ft", "cap_ft"]
    ).iloc[0]
    key = (
        float(chosen["start_ft"]),
        float(chosen["transition_width_ft"]),
        str(chosen["reanchor"]),
        float(chosen["cap_ft"]),
    )
    valid = folds == held_fold
    correction = correction_cache[key].astype(np.float64)
    far_only_prediction[valid] = base[valid] + float(chosen["dose"]) * correction[valid]
    valid_report = report(data.loc[valid].reset_index(drop=True), far_only_prediction[valid])
    far_only_selected_rows.append({
        **chosen.to_dict(),
        "valid_rmse": valid_report["rmse"],
        "valid_gain": float(chosen["valid_base_rmse"] - valid_report["rmse"]),
        "valid_far_rmse": valid_report["far_gt2000_rmse"],
    })
far_only_selected = pd.DataFrame(far_only_selected_rows)
far_only_selected.to_csv(OUT_DIR / "far_only_nested_selected.csv", index=False)
far_only_report = report(data, far_only_prediction)
maximum_first_2000_change = float(
    np.max(np.abs(far_only_prediction[distance <= 2000.0] - base[distance <= 2000.0]))
)

frozen = distance <= 1000.0
maximum_frozen_change = float(np.max(np.abs(prediction[frozen] - base[frozen])))
mid_degradation = candidate_report["mid_1000_2000_rmse"] - base_report["mid_1000_2000_rmse"]
far_gain = base_report["far_gt2000_rmse"] - candidate_report["far_gt2000_rmse"]
gain = base_report["rmse"] - candidate_report["rmse"]
positive_folds = int((selected["valid_gain"] > 0).sum())
nonzero_folds = int((selected["dose"] > 0).sum())
shuffle_loss = shuffle_report["rmse"] - candidate_report["rmse"]

verdict = {
    "experiment": "tail_only_reanchored_neighbor_gateway_v0",
    "stage": "fivefold_nested_distal_only_foldclean_neighbor_consumption",
    "sample_unit": "complete physical well",
    "source": str(SOURCE),
    "base": str(namespace["PREDICTION_PATH"]),
    "protocol": {
        "selection": "other four folds choose one global tail configuration",
        "frozen_distance_le_ft": 1000.0,
        "maximum_train_mid_degradation_ft": MAX_MID_DEGRADATION_FT,
        "matched_control": "within-well circular shift before transition reanchoring",
    },
    "base_metrics": base_report,
    "candidate_metrics": candidate_report,
    "strict_far_only_metrics": far_only_report,
    "shuffle_metrics": shuffle_report,
    "selected_folds": selected.to_dict(orient="records"),
    "strict_far_only_selected_folds": far_only_selected.to_dict(orient="records"),
    "effects": {
        "gain": gain,
        "far_gain": far_gain,
        "mid_1000_2000_degradation": mid_degradation,
        "maximum_frozen_change": maximum_frozen_change,
        "shuffle_loss": shuffle_loss,
        "positive_folds": positive_folds,
        "nonzero_folds": nonzero_folds,
        "strict_far_only_gain": base_report["rmse"] - far_only_report["rmse"],
        "strict_far_only_far_gain": base_report["far_gt2000_rmse"] - far_only_report["far_gt2000_rmse"],
        "strict_far_only_maximum_first_2000_change": maximum_first_2000_change,
    },
    "checks": {
        "first_1000ft_exactly_frozen": maximum_frozen_change == 0.0,
        "mid_degradation_le_0p02": mid_degradation <= 0.02,
        "far_gain_ge_0p20": far_gain >= 0.20,
        "pooled_gain_ge_0p10": gain >= 0.10,
        "positive_folds_ge_4": positive_folds >= 4,
        "nonzero_folds_ge_4": nonzero_folds >= 4,
        "shuffle_loss_ge_0p05": shuffle_loss >= 0.05,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> package the frozen tail-only transform for the submission pipeline"
    if verdict["strict_pass"]
    else "MARGINAL -> mechanism is positive and causal, but misses the preregistered far-gain gate"
)

output = data[["well", "fold", "row_idx", "truth", "base", "distance"]].assign(
    prediction=prediction,
    strict_far_only_prediction=far_only_prediction,
    shuffled_prediction=shuffle_prediction,
)
try:
    output.to_parquet(OUT_DIR / "tail_only_oof.parquet", index=False)
except ImportError:
    output.to_csv(OUT_DIR / "tail_only_oof.csv.gz", index=False, compression="gzip")
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nTAIL-ONLY SELECTED FOLDS")
print(selected.to_string(index=False), flush=True)
print("\nTAIL-ONLY COMPARISON")
print(pd.DataFrame([
    {"variant": "base", **base_report},
    {"variant": "tail_only", **candidate_report},
    {"variant": "strict_far_only", **far_only_report},
    {"variant": "shuffle", **shuffle_report},
]).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
