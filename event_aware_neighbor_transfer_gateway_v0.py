"""Fold-clean sparse structural-event transfer from neighboring wells.

The established neighbor method transfers a complete labeled trajectory.  This
gateway asks a narrower geological question: do nearby historical laterals tell
us where persistent changes in structural dip occur?  A reference trajectory is
decomposed into its anchor tangent plus a smoothed residual.  Only that residual
is added to the target's current OOF path.

Reference truth is restricted to wells outside the target fold.  Configuration
and dose are selected on the other four folds and then applied to the held fold.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
PREDICTION_PATH = ROOT / "full_oof_cp_crossfit_viewer.csv"
OUT_DIR = ROOT / "output" / "event_aware_neighbor_transfer_gateway_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRY_STRIDE = 64
SHORTLIST_QUERY_K = 192
SHORTLIST_WELLS = 16
MIN_ABS_DIRECTION_DOT = 0.55
MAX_MEDIAN_DISTANCE_FT = 1600.0
SMOOTHING_FT = (128.0, 256.0, 512.0)
NEIGHBOR_COUNTS = (1, 3)
DOSES = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75)
CAPS_FT = (12.0, 24.0, 40.0)
SEED = 20260725


def stable_fold(well: str) -> int:
    digest = hashlib.sha256(well.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 + 1


def cumulative_xy(xy: np.ndarray) -> np.ndarray:
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    step[~np.isfinite(step)] = 0.0
    return np.r_[0.0, np.cumsum(step)]


def unique_axis(axis: np.ndarray, *arrays: np.ndarray):
    keep = np.r_[np.diff(axis) > 1e-7, True]
    return (axis[keep], *(np.asarray(values)[keep] for values in arrays))


def rmse(truth, prediction) -> float:
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - frame["truth"].to_numpy(np.float64)
    per_well = pd.DataFrame({"well": frame["well"], "e2": error**2}).groupby("well")["e2"].mean()
    distance = frame["distance"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(per_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(per_well), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(
        PREDICTION_PATH,
        usecols=["id", "tvt"],
        dtype={"id": "string", "tvt": np.float32},
    )
    if not {"id", "tvt"}.issubset(frame.columns):
        raise RuntimeError(f"Unexpected prediction columns: {frame.columns.tolist()}")
    # Competition IDs are fixed-width: eight hex characters, underscore, row.
    # Avoid expand=True here; it multiplies memory and runtime on 3.8M rows.
    frame["well"] = frame["id"].str.slice(0, 8).str.lower().astype("category")
    frame["row_idx"] = pd.to_numeric(
        frame["id"].str.slice(9), errors="raise", downcast="integer"
    ).astype(np.int32)
    if frame.duplicated(["well", "row_idx"]).any():
        raise RuntimeError("Duplicate OOF keys")
    return frame[["well", "row_idx", "tvt"]].rename(columns={"tvt": "base"})


def load_well(path: Path) -> dict:
    well = path.name.split("__", 1)[0].lower()
    frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[["MD", "X", "Y", "Z", "TVT"]].to_numpy()).all():
        raise RuntimeError(f"{well}: non-finite geometry/truth")
    known = np.flatnonzero(np.isfinite(frame["TVT_input"].to_numpy(float)))
    if not len(known) or known[-1] >= len(frame) - 32:
        raise RuntimeError(f"{well}: invalid visible-prefix mask")
    anchor = int(known[-1])
    xy = frame[["X", "Y"]].to_numpy(np.float64)
    direction = xy[-1] - xy[anchor]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0:
        raise RuntimeError(f"{well}: degenerate lateral direction")
    direction /= norm
    progress = cumulative_xy(xy)
    structure = frame["TVT"].to_numpy(np.float64) + frame["Z"].to_numpy(np.float64)
    up, us = unique_axis(progress, structure)
    sampled = np.arange(anchor, len(frame), GEOMETRY_STRIDE, dtype=np.int32)
    if sampled[-1] != len(frame) - 1:
        sampled = np.r_[sampled, len(frame) - 1]
    return {
        "well": well,
        "fold": stable_fold(well),
        "md": frame["MD"].to_numpy(np.float64),
        "xy": xy,
        "z": frame["Z"].to_numpy(np.float64),
        "truth": frame["TVT"].to_numpy(np.float64),
        "tvt_input": frame["TVT_input"].to_numpy(np.float64),
        "anchor": anchor,
        "direction": direction,
        "progress": progress,
        "unique_progress": up,
        "unique_structure": us,
        "sampled": sampled,
        "sample_xy": xy[sampled],
        "tree": cKDTree(xy[sampled], leafsize=32),
    }


started = time.time()
oof = load_predictions()
wells = sorted(oof["well"].unique())
records: dict[str, dict] = {}
failures = []
for number, well in enumerate(wells, 1):
    try:
        records[well] = load_well(TRAIN_DIR / f"{well}__horizontal_well.csv")
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == len(wells):
        print(f"load geology wells {number}/{len(wells)} failures={len(failures)}", flush=True)
if len(records) < len(wells) - 3:
    raise RuntimeError(f"Loaded only {len(records)}/{len(wells)} wells: {failures[:5]}")
pd.DataFrame(failures).to_csv(OUT_DIR / "load_failures.csv", index=False)

# Build one global geometry index. Fold restrictions are applied when candidates
# are collected, before any reference truth is read.
point_xy = np.concatenate([records[w]["sample_xy"] for w in records])
point_well = np.concatenate([
    np.full(len(records[w]["sample_xy"]), w, dtype=object) for w in records
])
global_tree = cKDTree(point_xy, leafsize=64)


def shortlist(target: dict) -> pd.DataFrame:
    distances, indices = global_tree.query(
        target["sample_xy"], k=min(SHORTLIST_QUERY_K, len(point_xy)), workers=-1
    )
    indices = np.atleast_2d(indices)
    rough: dict[str, float] = {}
    for index in indices.ravel():
        candidate = str(point_well[int(index)])
        if candidate == target["well"] or records[candidate]["fold"] == target["fold"]:
            continue
        rough[candidate] = min(rough.get(candidate, np.inf), 0.0)
    rows = []
    for candidate in list(rough)[:80]:
        reference = records[candidate]
        distance, _ = reference["tree"].query(target["sample_xy"], k=1)
        dot = float(np.dot(target["direction"], reference["direction"]))
        median = float(np.median(distance))
        p90 = float(np.quantile(distance, 0.90))
        anchor_distance = float(reference["tree"].query(target["xy"][target["anchor"]], k=1)[0])
        score = median + 0.25 * p90 + 0.10 * anchor_distance + 500.0 * max(0.0, MIN_ABS_DIRECTION_DOT - abs(dot))
        rows.append({
            "neighbor": candidate, "score": score, "median_distance": median,
            "p90_distance": p90, "anchor_distance": anchor_distance,
            "direction_dot": dot,
        })
    report = pd.DataFrame(rows).sort_values("score")
    eligible = report[
        (report["median_distance"] <= MAX_MEDIAN_DISTANCE_FT)
        & (report["direction_dot"].abs() >= MIN_ABS_DIRECTION_DOT)
    ]
    return (eligible if len(eligible) else report).head(SHORTLIST_WELLS).reset_index(drop=True)


def mapped_reference(target: dict, reference: dict, target_rows: np.ndarray):
    target_progress = target["progress"][target_rows] - target["progress"][target["anchor"]]
    anchor_xy = target["xy"][target["anchor"]]
    _, local_index = reference["tree"].query(anchor_xy, k=1)
    ref_anchor_row = int(reference["sampled"][int(local_index)])
    sign = 1.0 if np.dot(target["direction"], reference["direction"]) >= 0.0 else -1.0
    query = reference["progress"][ref_anchor_row] + sign * target_progress
    valid = (query >= reference["unique_progress"][0]) & (query <= reference["unique_progress"][-1])
    mapped = np.full(len(query), np.nan, np.float64)
    mapped[valid] = np.interp(
        query[valid], reference["unique_progress"], reference["unique_structure"]
    )
    anchor_structure = float(reference["structure_at_anchor"])
    return target_progress, mapped, valid, anchor_structure, sign, ref_anchor_row


def event_residual(reference: dict, sigma_ft: float) -> tuple[np.ndarray, np.ndarray]:
    axis = reference["unique_progress"]
    structure = reference["unique_structure"]
    step = max(float(np.median(np.diff(axis))), 1.0)
    smooth = gaussian_filter1d(structure, sigma=max(sigma_ft / step, 1.0), mode="nearest")
    return axis, smooth


# Precompute smoothed reference structures. The anchor tangent is evaluated at
# the target/reference crossing, so the transferred residual is exactly zero at
# the target anchor and cannot alter the visible prefix datum.
for reference in records.values():
    reference["smooth"] = {}
    for smoothing in SMOOTHING_FT:
        reference["smooth"][smoothing] = event_residual(reference, smoothing)[1]


def contribution(target: dict, reference: dict, target_rows: np.ndarray, smoothing: float):
    target_progress = target["progress"][target_rows] - target["progress"][target["anchor"]]
    _, local_index = reference["tree"].query(target["xy"][target["anchor"]], k=1)
    anchor_row = int(reference["sampled"][int(local_index)])
    sign = 1.0 if np.dot(target["direction"], reference["direction"]) >= 0.0 else -1.0
    anchor_progress = float(reference["progress"][anchor_row])
    query = anchor_progress + sign * target_progress
    axis = reference["unique_progress"]
    smooth = reference["smooth"][smoothing]
    valid = (query >= axis[0]) & (query <= axis[-1])
    values = np.full(len(query), np.nan)
    values[valid] = np.interp(query[valid], axis, smooth)
    anchor_value = float(np.interp(anchor_progress, axis, smooth))
    derivative = np.gradient(smooth, axis, edge_order=1)
    anchor_rate = float(np.interp(anchor_progress, axis, derivative)) * sign
    residual = values - anchor_value - anchor_rate * target_progress
    residual[~valid] = np.nan
    return residual, valid


row_parts = []
neighbor_parts = []
for number, well in enumerate(sorted(records), 1):
    target = records[well]
    selected = shortlist(target)
    selected.insert(0, "well", well)
    selected.insert(1, "fold", target["fold"])
    selected.insert(2, "rank", np.arange(1, len(selected) + 1))
    neighbor_parts.append(selected)

    group = oof[oof["well"].eq(well)].sort_values("row_idx")
    rows = group["row_idx"].to_numpy(np.int64)
    valid_rows = (rows >= target["anchor"]) & (rows < len(target["md"]))
    rows = rows[valid_rows]
    base = group["base"].to_numpy(np.float64)[valid_rows]
    truth = target["truth"][rows]
    finite = np.isfinite(base) & np.isfinite(truth)
    rows, base, truth = rows[finite], base[finite], truth[finite]
    output = pd.DataFrame({
        "well": well, "fold": target["fold"], "row_idx": rows,
        "truth": truth, "base": base,
        "distance": target["md"][rows] - target["md"][target["anchor"]],
    })
    refs = [records[name] for name in selected["neighbor"].tolist()]
    weights = 1.0 / np.maximum(selected["score"].to_numpy(np.float64), 25.0)
    for smoothing in SMOOTHING_FT:
        residuals = []
        for reference in refs:
            residual, _ = contribution(target, reference, rows, smoothing)
            residuals.append(residual)
        matrix = np.asarray(residuals, np.float64)
        for count in NEIGHBOR_COUNTS:
            available = min(count, len(matrix))
            values = matrix[:available]
            local_weights = weights[:available, None] * np.isfinite(values)
            numerator = np.nansum(values * local_weights, axis=0)
            denominator = np.sum(local_weights, axis=0)
            aggregate = np.divide(numerator, denominator, out=np.zeros(len(rows)), where=denominator > 0)
            coverage = denominator > 0
            aggregate[~coverage] = 0.0
            output[f"event_s{int(smoothing)}_k{count}"] = aggregate.astype(np.float32)
            output[f"coverage_s{int(smoothing)}_k{count}"] = coverage.astype(np.int8)

    # Complete-curve transfer is retained only as a matched benchmark.
    reference = refs[0]
    target_progress = target["progress"][rows] - target["progress"][target["anchor"]]
    _, local_index = reference["tree"].query(target["xy"][target["anchor"]], k=1)
    anchor_row = int(reference["sampled"][int(local_index)])
    sign = 1.0 if np.dot(target["direction"], reference["direction"]) >= 0 else -1.0
    query = reference["progress"][anchor_row] + sign * target_progress
    valid = (query >= reference["unique_progress"][0]) & (query <= reference["unique_progress"][-1])
    ref_structure = np.full(len(rows), np.nan)
    ref_structure[valid] = np.interp(query[valid], reference["unique_progress"], reference["unique_structure"])
    ref_anchor_structure = float(np.interp(reference["progress"][anchor_row], reference["unique_progress"], reference["unique_structure"]))
    target_anchor_structure = float(target["tvt_input"][target["anchor"]] + target["z"][target["anchor"]])
    full_tvt = ref_structure + (target_anchor_structure - ref_anchor_structure) - target["z"][rows]
    output["fullcurve_delta"] = np.where(valid, full_tvt - base, 0.0).astype(np.float32)
    output["fullcurve_coverage"] = valid.astype(np.int8)

    row_parts.append(output)
    if number % 50 == 0 or number == len(records):
        print(f"event transfer {number}/{len(records)}", flush=True)

data = pd.concat(row_parts, ignore_index=True)
neighbors = pd.concat(neighbor_parts, ignore_index=True)
neighbors.to_csv(OUT_DIR / "selected_foldclean_neighbors.csv", index=False)


def shuffled(values: np.ndarray, wells_array: np.ndarray) -> np.ndarray:
    result = values.copy()
    for well in np.unique(wells_array):
        index = np.flatnonzero(wells_array == well)
        if len(index) < 8:
            continue
        shift = max(len(index) // 3, 1)
        result[index] = np.roll(values[index], shift)
        result[index[0]] = 0.0
    return result


base_values = data["base"].to_numpy(np.float64)
truth_values = data["truth"].to_numpy(np.float64)
fold_values = data["fold"].to_numpy(np.int8)
well_values = data["well"].to_numpy(str)
candidate_columns = [column for column in data if column.startswith("event_")]
candidate_columns.append("fullcurve_delta")

selection_rows = []
oof_predictions = {}
oof_shuffled = {}
for candidate in candidate_columns:
    raw = data[candidate].to_numpy(np.float64)
    raw_shuffle = shuffled(raw, well_values)
    prediction = base_values.copy()
    prediction_shuffle = base_values.copy()
    for fold in range(1, 6):
        train = fold_values != fold
        valid = ~train
        best = None
        for cap in CAPS_FT:
            clipped = np.clip(raw, -cap, cap)
            for dose in DOSES:
                score = rmse(truth_values[train], base_values[train] + dose * clipped[train])
                key = (score, dose, cap)
                if best is None or key < best[0]:
                    best = (key, dose, cap)
        _, dose, cap = best
        prediction[valid] = base_values[valid] + dose * np.clip(raw[valid], -cap, cap)
        prediction_shuffle[valid] = base_values[valid] + dose * np.clip(raw_shuffle[valid], -cap, cap)
        selection_rows.append({
            "candidate": candidate, "held_fold": fold, "dose": dose, "cap_ft": cap,
            "train_rmse": best[0][0], "valid_base_rmse": rmse(truth_values[valid], base_values[valid]),
            "valid_rmse": rmse(truth_values[valid], prediction[valid]),
        })
    oof_predictions[candidate] = prediction
    oof_shuffled[candidate] = prediction_shuffle

selection = pd.DataFrame(selection_rows)
selection.to_csv(OUT_DIR / "nested_selection.csv", index=False)
summary_rows = []
base_report = metrics(data, base_values)
for candidate in candidate_columns:
    report = metrics(data, oof_predictions[candidate])
    control = metrics(data, oof_shuffled[candidate])
    summary_rows.append({
        "candidate": candidate,
        "base_rmse": base_report["rmse"],
        "rmse": report["rmse"],
        "gain": base_report["rmse"] - report["rmse"],
        "far_rmse": report["far_gt2000_rmse"],
        "far_gain": base_report["far_gt2000_rmse"] - report["far_gt2000_rmse"],
        "median_well_rmse": report["median_well_rmse"],
        "p95_well_rmse": report["p95_well_rmse"],
        "shuffle_rmse": control["rmse"],
        "shuffle_loss": control["rmse"] - report["rmse"],
        "positive_folds": int(sum(
            row.valid_rmse < row.valid_base_rmse
            for row in selection[selection["candidate"].eq(candidate)].itertuples()
        )),
        "nonzero_folds": int(sum(
            row.dose > 0 for row in selection[selection["candidate"].eq(candidate)].itertuples()
        )),
    })
summary = pd.DataFrame(summary_rows).sort_values(["rmse", "candidate"]).reset_index(drop=True)
summary.to_csv(OUT_DIR / "candidate_summary.csv", index=False)
print("\nEVENT-AWARE NEIGHBOR SUMMARY")
print(summary.to_string(index=False), flush=True)

event_summary = summary[summary["candidate"].str.startswith("event_")]
best = event_summary.iloc[0].to_dict()
best_name = str(best["candidate"])
fullcurve_benchmark = summary[summary["candidate"].eq("fullcurve_delta")].iloc[0].to_dict()

verdict = {
    "experiment": "event_aware_neighbor_transfer_gateway_v0",
    "stage": "fivefold_sparse_structural_event_transfer_gateway",
    "sample_unit": "complete target well; historical reference truth excludes target fold",
    "base": str(PREDICTION_PATH),
    "wells": int(data["well"].nunique()),
    "rows": int(len(data)),
    "base_metrics": base_report,
    "best": best,
    "fullcurve_benchmark": fullcurve_benchmark,
    "fold_doses": selection[selection["candidate"].eq(best_name)].to_dict(orient="records"),
    "checks": {
        "gain_ge_0p10": bool(best["gain"] >= 0.10),
        "far_gain_ge_0p20": bool(best["far_gain"] >= 0.20),
        "positive_folds_ge_4": bool(best["positive_folds"] >= 4),
        "nonzero_folds_ge_4": bool(best["nonzero_folds"] >= 4),
        "shuffle_loss_ge_0p05": bool(best["shuffle_loss"] >= 0.05),
    },
    "strict_pass": bool(
        best["gain"] >= 0.10 and best["far_gain"] >= 0.20
        and best["positive_folds"] >= 4 and best["nonzero_folds"] >= 4
        and best["shuffle_loss"] >= 0.05
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["next"] = (
    "PASS -> integrate sparse neighbor events as a separately gated mainline feature"
    if verdict["strict_pass"]
    else "FAIL -> nearby labeled events do not transfer; retain full-curve neighbor only"
)
if verdict["strict_pass"]:
    best_output = data[["well", "fold", "row_idx", "truth", "base", "distance"]].assign(
        prediction=oof_predictions[best_name],
        shuffled_prediction=oof_shuffled[best_name],
    )
    try:
        best_output.to_parquet(OUT_DIR / "best_oof.parquet", index=False)
    except ImportError:
        best_output.to_csv(OUT_DIR / "best_oof.csv.gz", index=False, compression="gzip")
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
