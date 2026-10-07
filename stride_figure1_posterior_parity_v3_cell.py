"""STRIDE Figure-1 parity gateway from original competition data.

This cell implements only the first extension out of the visible prefix.  It
does not decode a complete well and therefore cannot be promoted as a model.
Its purpose is to verify that the documented state and probability terms form
a useful joint posterior over (segment length, delta TVT) before another full
beam run is attempted.

Author-disclosed structure
--------------------------
* state: explicit (h_len, delta_TVT), with delta-TVT spacing <= 1 ft;
* segment-length log-normal prior fit from real trajectory behaviour;
* persistence on dip = delta_TVT / h_len;
* drilled-trend centering from the visible tail and known Z geometry;
* visible-prefix robust affine typewell calibration;
* segment-aggregated Cauchy GR likelihood.

Implementation assumptions, selected without touching Fold 2
--------------------------------------------------------------
The writeup does not disclose the truth segmentation tolerance or the
effective temperature of autocorrelated GR rows.  Fold 4/5 fit priors, a fixed
32-well subset of Fold 3 selects these two assumptions, the rest of Fold 3
confirms them, and Fold 2 is then evaluated once.  Fold 1 remains untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_figure1_posterior_parity_v3"
SEED = 20260722
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1

SEGMENTATION_TOLERANCES_FT = (0.5, 1.0, 2.0)
EMISSION_TEMPERATURES = (0.03, 0.10, 0.30, 1.00)
H_GRID_SIZE = 10
H_MIN_FT = 50.0
H_MAX_FT = 600.0
D_STEP_FT = 1.0
DISCOVERY_WELLS = 32
MAX_EMISSION_ROWS = 160
MIN_PREFIX_ROWS = 64
MIN_EVAL_ROWS = 64


def _find_data_root() -> Path:
    candidates = (
        Path.cwd(),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
        Path("/root/autodl-tmp"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    )
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate original competition train directory")


DATA_ROOT = _find_data_root()
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
else:
    OOF_ROOT = Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _stable_folds(wells):
    scored = []
    for well in wells:
        digest = hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    output = []
    for fold, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        output.extend((ordered[int(index)], fold) for index in indices)
    return pd.DataFrame(output, columns=["well", "fold"])


def _robust_scale(values, floor=1e-4):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return float(max(1.4826 * mad, 0.20 * np.std(values), floor))


def _deduplicate(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).dropna()
    frame = frame.groupby("axis", as_index=False)["value"].mean().sort_values("axis")
    return frame["axis"].to_numpy(np.float64), frame["value"].to_numpy(np.float64)


def _robust_affine(expected, observed):
    valid = np.isfinite(expected) & np.isfinite(observed)
    x = expected[valid]
    y = observed[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        intercept = float(np.median(y - x)) if len(x) else 0.0
        residual = y - (x + intercept)
        return 1.0, intercept, _robust_scale(residual, 3.0), int(len(x))
    keep = np.ones(len(x), dtype=bool)
    gain, intercept = 1.0, 0.0
    for _ in range(8):
        design = np.column_stack([x[keep], np.ones(int(keep.sum()))])
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (gain * x + intercept)
        centre = float(np.median(residual[keep]))
        scale = _robust_scale(residual[keep], 1.0)
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if next_keep.sum() < 24 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    # The note describes a calibration, not an unconstrained remapping.  This
    # broad guard catches degenerate prefix fits without imposing a tuned dose.
    gain = float(np.clip(gain, 0.5, 1.5))
    residual = y[keep] - (gain * x[keep] + intercept)
    return gain, float(intercept), _robust_scale(residual, 3.0), int(keep.sum())


def _load_record(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    horizontal = horizontal.sort_values("MD").reset_index(drop=True)
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(
        limit_direction="both"
    ).to_numpy(np.float64)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    evaluation = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth) & np.isfinite(md))
    if len(known) < MIN_PREFIX_ROWS or len(evaluation) < MIN_EVAL_ROWS:
        raise RuntimeError(f"{well}: prefix/eval rows={len(known)}/{len(evaluation)}")
    anchor = int(known[-1])
    tw_tvt, tw_gr = _deduplicate(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    expected = np.interp(tvt_input[known], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    gain, intercept, gamma, calibration_rows = _robust_affine(expected, gr[known])
    tail = known[md[known] >= md[anchor] - 500.0]
    if len(tail) < 32:
        tail = known[-min(256, len(known)):]
    previous_dip = float(np.polyfit(md[tail], tvt_input[tail], 1)[0])
    previous_z_rate = float(np.polyfit(md[tail], z[tail], 1)[0])
    return {
        "well": well,
        "md": md,
        "z": z,
        "gr": gr,
        "truth": truth,
        "tvt_input": tvt_input,
        "known": known,
        "evaluation": evaluation,
        "anchor": anchor,
        "anchor_md": float(md[anchor]),
        "anchor_tvt": float(tvt_input[anchor]),
        "tw_tvt": tw_tvt,
        "tw_gr_raw": tw_gr,
        "tw_gr_affine": gain * tw_gr + intercept,
        "calibration_gain": gain,
        "calibration_intercept": intercept,
        "calibration_rows": calibration_rows,
        "gamma": gamma,
        "previous_dip": previous_dip,
        "previous_z_rate": previous_z_rate,
    }


def _rdp_breaks(md, tvt, tolerance):
    """Piecewise-linear truth segmentation used only to fit outer priors."""
    keep = {0, len(md) - 1}
    stack = [(0, len(md) - 1)]
    while stack:
        left, right = stack.pop()
        if right - left < 4:
            continue
        fraction = (md[left + 1:right] - md[left]) / max(md[right] - md[left], 1e-9)
        linear = tvt[left] + fraction * (tvt[right] - tvt[left])
        residual = np.abs(tvt[left + 1:right] - linear)
        if not len(residual):
            continue
        local = int(np.argmax(residual))
        if residual[local] > tolerance:
            split = left + 1 + local
            keep.add(split)
            stack.append((left, split))
            stack.append((split, right))
    return np.asarray(sorted(keep), np.int64)


def _truth_segments(record, tolerance):
    rows = np.r_[record["anchor"], record["evaluation"]]
    md_raw = record["md"][rows]
    tvt_raw = np.r_[record["anchor_tvt"], record["truth"][record["evaluation"]]]
    z_raw = record["z"][rows]
    grid = np.arange(md_raw[0], md_raw[-1] + 8.0, 16.0)
    if grid[-1] > md_raw[-1]:
        grid[-1] = md_raw[-1]
    grid = np.unique(grid)
    tvt = np.interp(grid, md_raw, tvt_raw)
    z = np.interp(grid, md_raw, z_raw)
    breaks = _rdp_breaks(grid, tvt, tolerance)
    segments = []
    previous_dip = record["previous_dip"]
    previous_structure_rate = record["previous_dip"] + record["previous_z_rate"]
    for left, right in zip(breaks[:-1], breaks[1:]):
        length = float(grid[right] - grid[left])
        if length < 32.0:
            continue
        dip = float((tvt[right] - tvt[left]) / length)
        z_rate = float((z[right] - z[left]) / length)
        drilled_trend = previous_structure_rate - z_rate
        segments.append((length, dip, previous_dip, drilled_trend))
        previous_dip = dip
        previous_structure_rate = dip + z_rate
    return segments


def _fit_priors(records, tolerance):
    segments = []
    for record in records:
        segments.extend(_truth_segments(record, tolerance))
    if len(segments) < 500:
        raise RuntimeError(f"too few fitted segments at tolerance={tolerance}: {len(segments)}")
    array = np.asarray(segments, np.float64)
    length = np.clip(array[:, 0], H_MIN_FT, H_MAX_FT)
    current = array[:, 1]
    previous = array[:, 2]
    trend = array[:, 3]
    denominator = float(np.sum((previous - trend) ** 2))
    if denominator > 1e-12:
        lam = float(np.sum((current - trend) * (previous - trend)) / denominator)
    else:
        lam = 1.0
    lam = float(np.clip(lam, 0.0, 1.0))
    centre = lam * previous + (1.0 - lam) * trend
    residual = current - centre
    dip_low, dip_high = np.quantile(current, [0.005, 0.995])
    return {
        "segmentation_tolerance_ft": float(tolerance),
        "length_log_mu": float(np.mean(np.log(length))),
        "length_log_sigma": float(max(np.std(np.log(length)), 0.15)),
        "lambda_previous_dip": lam,
        "sigma_persist": _robust_scale(residual, 0.0025),
        "dip_low": float(dip_low),
        "dip_high": float(dip_high),
        "segments": int(len(array)),
        "wells": int(len(records)),
    }


def _state_grids(prior):
    z = np.linspace(-1.8, 1.8, H_GRID_SIZE)
    h = np.exp(prior["length_log_mu"] + prior["length_log_sigma"] * z)
    h = np.unique(np.clip(np.rint(h), H_MIN_FT, H_MAX_FT)).astype(np.float64)
    h = np.unique(np.r_[H_MIN_FT, h, H_MAX_FT])
    dip_abs = max(abs(prior["dip_low"]), abs(prior["dip_high"]))
    d_half = float(np.clip(math.ceil((dip_abs * H_MAX_FT + 8.0) / D_STEP_FT) * D_STEP_FT, 48.0, 160.0))
    d = np.arange(-d_half, d_half + 0.5 * D_STEP_FT, D_STEP_FT, dtype=np.float64)
    return h, d


def _length_nll(length, prior):
    log_length = math.log(max(length, 1.0))
    z = (log_length - prior["length_log_mu"]) / prior["length_log_sigma"]
    # H_grid is approximately uniform in log(h).  Its discrete bin width is
    # proportional to h, which cancels the 1/h Jacobian in the continuous
    # log-normal density.  Scoring the density without that bin mass would
    # spuriously favour the shortest segment.
    return 0.5 * z * z


def _evaluate_well(record, prior, temperature, reference_mode="affine", likelihood="cauchy"):
    h_grid, d_grid = _state_grids(prior)
    tw_gr = record["tw_gr_affine"] if reference_mode == "affine" else record["tw_gr_raw"]
    map_rows = []
    global_best = None
    for requested_h in h_grid:
        target_md = record["anchor_md"] + requested_h
        candidates = record["evaluation"][record["md"][record["evaluation"]] <= target_md]
        if len(candidates) < 16:
            continue
        endpoint = int(candidates[-1])
        actual_h = float(record["md"][endpoint] - record["anchor_md"])
        if actual_h < 0.8 * H_MIN_FT:
            continue
        rows = np.arange(record["anchor"] + 1, endpoint + 1, dtype=np.int64)
        if len(rows) > MAX_EMISSION_ROWS:
            rows = rows[np.linspace(0, len(rows) - 1, MAX_EMISSION_ROWS).astype(np.int64)]
        fraction = (record["md"][rows] - record["anchor_md"]) / actual_h
        candidate_tvt = record["anchor_tvt"] + d_grid[:, None] * fraction[None, :]
        flat = candidate_tvt.ravel()
        expected = np.interp(flat, record["tw_tvt"], tw_gr).reshape(candidate_tvt.shape)
        inside = (candidate_tvt >= record["tw_tvt"][0]) & (candidate_tvt <= record["tw_tvt"][-1])
        residual = (record["gr"][rows][None, :] - expected) / max(record["gamma"], 1e-6)
        if likelihood == "cauchy":
            point_cost = np.log1p(np.minimum(residual * residual, 1e4))
        else:
            point_cost = 0.5 * np.minimum(residual * residual, 400.0)
        point_cost[~inside] = 50.0
        emission = point_cost.mean(axis=1)

        z_rate = float((record["z"][endpoint] - record["z"][record["anchor"]]) / actual_h)
        previous_structure = record["previous_dip"] + record["previous_z_rate"]
        drilled_trend = previous_structure - z_rate
        centre = (
            prior["lambda_previous_dip"] * record["previous_dip"]
            + (1.0 - prior["lambda_previous_dip"]) * drilled_trend
        )
        dip = d_grid / actual_h
        persistence = 0.5 * ((dip - centre) / prior["sigma_persist"]) ** 2
        total = _length_nll(actual_h, prior) + persistence + temperature * emission
        selected = int(np.argmin(total))
        truth_d = float(record["truth"][endpoint] - record["anchor_tvt"])
        truth_index = int(np.argmin(np.abs(d_grid - truth_d)))
        rank = int(np.sum(total < total[truth_index]) + 1)
        item = {
            "well": record["well"],
            "requested_h_ft": float(requested_h),
            "h_len_ft": actual_h,
            "truth_delta_tvt": truth_d,
            "map_delta_tvt": float(d_grid[selected]),
            "map_error_ft": float(d_grid[selected] - truth_d),
            "truth_grid_error_ft": float(d_grid[truth_index] - truth_d),
            "truth_rank": rank,
            "states": int(len(d_grid)),
            "map_cost": float(total[selected]),
            "truth_cost": float(total[truth_index]),
            "posterior_margin": float(total[truth_index] - total[selected]),
            "previous_dip": record["previous_dip"],
            "drilled_trend_dip": drilled_trend,
            "persistence_centre": centre,
            "calibration_gain": record["calibration_gain"],
            "calibration_intercept": record["calibration_intercept"],
            "gamma": record["gamma"],
        }
        map_rows.append(item)
        if global_best is None or item["map_cost"] < global_best["map_cost"]:
            global_best = item
    if global_best is None:
        raise RuntimeError(f"{record['well']}: no valid Figure-1 states")
    return map_rows, global_best


def _summarise(frame, label):
    error = frame["map_error_ft"].to_numpy(np.float64)
    oracle = frame["truth_grid_error_ft"].to_numpy(np.float64)
    return {
        "variant": label,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "map_endpoint_rmse": float(np.sqrt(np.mean(error**2))),
        "map_endpoint_mae": float(np.mean(np.abs(error))),
        "grid_oracle_rmse": float(np.sqrt(np.mean(oracle**2))),
        "within_1ft": float(np.mean(np.abs(error) <= 1.0)),
        "within_5ft": float(np.mean(np.abs(error) <= 5.0)),
        "median_truth_rank": float(np.median(frame["truth_rank"])),
        "p90_truth_rank": float(np.quantile(frame["truth_rank"], 0.90)),
        "top5_contains_truth": float(np.mean(frame["truth_rank"] <= 5)),
        "median_posterior_margin": float(np.median(frame["posterior_margin"])),
        "median_selected_h_ft": float(np.median(frame["h_len_ft"])),
        "p10_selected_h_ft": float(np.quantile(frame["h_len_ft"], 0.10)),
        "p90_selected_h_ft": float(np.quantile(frame["h_len_ft"], 0.90)),
        "short_h_le_64_fraction": float(np.mean(frame["h_len_ft"] <= 64.0)),
    }


if FOLD_PATH is not None:
    folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]]
    folds = folds.drop_duplicates("well")
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
else:
    wells = [
        path.name.split("__", 1)[0]
        for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
    ]
    folds = _stable_folds(wells)
folds["well"] = folds["well"].astype(str)
folds = folds[folds["fold"].between(1, 5)].reset_index(drop=True)
if len(folds) != 773:
    raise RuntimeError(f"Expected 773 fold wells, got {len(folds)}")

needed = folds.loc[folds["fold"].isin((*PRIOR_FOLDS, SELECTION_FOLD, APPLICATION_FOLD)), "well"].tolist()
fold_map = dict(zip(folds["well"], folds["fold"]))
records = {}
failures = []
started = time.time()
for number, well in enumerate(needed, start=1):
    try:
        records[well] = _load_record(well)
    except Exception as exc:
        failures.append({"well": well, "fold": int(fold_map[well]), "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed):
        print(f"Figure1 load records {number}/{len(needed)} failures={len(failures)}", flush=True)
pd.DataFrame(failures, columns=["well", "fold", "reason"]).to_csv(
    OUT_DIR / "record_failures.csv", index=False
)
if failures:
    raise RuntimeError(f"{len(failures)} record failures: {failures[:5]}")

prior_records = [records[w] for w in needed if fold_map[w] in PRIOR_FOLDS]
selection_records = [records[w] for w in needed if fold_map[w] == SELECTION_FOLD]
application_records = [records[w] for w in needed if fold_map[w] == APPLICATION_FOLD]
ordered_selection = sorted(
    selection_records,
    key=lambda record: hashlib.sha256(f"figure1:{record['well']}".encode("ascii")).digest(),
)
discovery_records = ordered_selection[:DISCOVERY_WELLS]
confirmation_records = ordered_selection[DISCOVERY_WELLS:]

priors = {
    float(tolerance): _fit_priors(prior_records, float(tolerance))
    for tolerance in SEGMENTATION_TOLERANCES_FT
}
_write_json(OUT_DIR / "outer_train_priors.json", priors)
print("FIGURE1 OUTER-TRAIN PRIORS", json.dumps(priors, indent=2), flush=True)

scan_rows = []
for tolerance, prior in priors.items():
    for temperature in EMISSION_TEMPERATURES:
        ridge_rows = []
        global_rows = []
        for record in discovery_records:
            ridge, global_best = _evaluate_well(record, prior, temperature)
            ridge_rows.extend(ridge)
            global_rows.append(global_best)
        ridge_metrics = _summarise(pd.DataFrame(ridge_rows), "ridge")
        global_metrics = _summarise(pd.DataFrame(global_rows), "global_map")
        scan_rows.append(
            {
                "segmentation_tolerance_ft": tolerance,
                "emission_temperature": temperature,
                **{f"ridge_{key}": value for key, value in ridge_metrics.items() if key not in ("variant",)},
                **{f"global_{key}": value for key, value in global_metrics.items() if key not in ("variant",)},
            }
        )
        print(
            "FIGURE1 DISCOVERY",
            tolerance,
            temperature,
            ridge_metrics["map_endpoint_rmse"],
            global_metrics["map_endpoint_rmse"],
            flush=True,
        )

scan = pd.DataFrame(scan_rows).sort_values(
    ["ridge_map_endpoint_rmse", "global_map_endpoint_rmse"]
)
scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
selected = scan.iloc[0].to_dict()
selected_tolerance = float(selected["segmentation_tolerance_ft"])
selected_temperature = float(selected["emission_temperature"])
selected_prior = priors[selected_tolerance]


def _run_group(group, group_name, reference_mode="affine", likelihood="cauchy"):
    ridge_rows = []
    global_rows = []
    for number, record in enumerate(group, start=1):
        ridge, global_best = _evaluate_well(
            record,
            selected_prior,
            selected_temperature,
            reference_mode=reference_mode,
            likelihood=likelihood,
        )
        ridge_rows.extend(ridge)
        global_rows.append(global_best)
        if number % 50 == 0 or number == len(group):
            print(f"FIGURE1 {group_name} {number}/{len(group)}", flush=True)
    ridge_frame = pd.DataFrame(ridge_rows)
    global_frame = pd.DataFrame(global_rows)
    return ridge_frame, global_frame, {
        "ridge": _summarise(ridge_frame, f"{group_name}_ridge"),
        "global_map": _summarise(global_frame, f"{group_name}_global"),
    }


confirm_ridge, confirm_global, confirmation = _run_group(
    confirmation_records, "fold3_confirmation"
)
fold2_ridge, fold2_global, fold2 = _run_group(application_records, "fold2_application")
raw_ridge, raw_global, raw_control = _run_group(
    application_records, "fold2_raw_reference", reference_mode="raw"
)
gauss_ridge, gauss_global, gaussian_control = _run_group(
    application_records, "fold2_gaussian", likelihood="gaussian"
)

confirm_ridge.to_csv(OUT_DIR / "fold3_confirmation_ridge.csv", index=False)
confirm_global.to_csv(OUT_DIR / "fold3_confirmation_global.csv", index=False)
fold2_ridge.to_csv(OUT_DIR / "fold2_ridge.csv", index=False)
fold2_global.to_csv(OUT_DIR / "fold2_global.csv", index=False)

comparison = pd.DataFrame(
    [
        {"mode": "affine_cauchy", **fold2["ridge"]},
        {"mode": "raw_cauchy", **raw_control["ridge"]},
        {"mode": "affine_gaussian", **gaussian_control["ridge"]},
    ]
)
comparison.to_csv(OUT_DIR / "fold2_component_comparison.csv", index=False)

affine_rmse = float(fold2["ridge"]["map_endpoint_rmse"])
raw_rmse = float(raw_control["ridge"]["map_endpoint_rmse"])
gaussian_rmse = float(gaussian_control["ridge"]["map_endpoint_rmse"])
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Figure1_first_extension_joint_posterior_parity",
    "sample_unit": "first hidden segment nested within complete physical well",
    "raw_inputs_only": True,
    "uses_sp45_track8_mainline_strong_cp12_gold_neighbor": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "selection_discovery_wells": len(discovery_records),
        "selection_confirmation_wells": len(confirmation_records),
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "author_disclosed": {
        "state": "explicit h_len x delta_TVT",
        "segment_length_prior": "log-normal",
        "persistence_target": "delta_TVT / h_len",
        "typewell_reference": "visible-prefix robust affine",
        "likelihood": "Cauchy",
    },
    "implementation_assumptions": {
        "truth_segmentation": "Ramer-Douglas-Peucker piecewise-linear TVT(MD)",
        "segmentation_tolerances_scanned_ft": list(SEGMENTATION_TOLERANCES_FT),
        "emission_temperatures_scanned": list(EMISSION_TEMPERATURES),
        "h_grid_size_target": H_GRID_SIZE,
        "h_range_ft": [H_MIN_FT, H_MAX_FT],
        "delta_tvt_step_ft": D_STEP_FT,
        "maximum_emission_rows": MAX_EMISSION_ROWS,
    },
    "selected": {
        "segmentation_tolerance_ft": selected_tolerance,
        "emission_temperature": selected_temperature,
        "prior": selected_prior,
    },
    "fold3_confirmation": confirmation,
    "fold2": fold2,
    "fold2_controls": {
        "raw_reference": raw_control,
        "gaussian": gaussian_control,
        "affine_gain_vs_raw_endpoint_rmse": raw_rmse - affine_rmse,
        "cauchy_gain_vs_gaussian_endpoint_rmse": gaussian_rmse - affine_rmse,
    },
    "diagnostic_checks": {
        "grid_oracle_rmse_le_0p5": fold2["ridge"]["grid_oracle_rmse"] <= 0.5,
        "affine_improves_endpoint_rmse": affine_rmse < raw_rmse,
        "cauchy_improves_endpoint_rmse": affine_rmse < gaussian_rmse,
        "median_truth_rank_le_5": fold2["ridge"]["median_truth_rank"] <= 5.0,
        "top5_contains_truth_ge_0p7": fold2["ridge"]["top5_contains_truth"] >= 0.70,
    },
    "posterior_geometry_pass": bool(
        fold2["ridge"]["grid_oracle_rmse"] <= 0.5
        and fold2["ridge"]["median_truth_rank"] <= 5.0
        and fold2["ridge"]["top5_contains_truth"] >= 0.70
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> connect this exact state/prior/emission definition to a full-sequence beam"
        if (
            fold2["ridge"]["grid_oracle_rmse"] <= 0.5
            and fold2["ridge"]["median_truth_rank"] <= 5.0
            and fold2["ridge"]["top5_contains_truth"] >= 0.70
        )
        else "HOLD -> repair posterior coverage/ranking before full-sequence decoding"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nFIGURE1 DISCOVERY SCAN")
print(scan.to_string(index=False), flush=True)
print("\nFIGURE1 FOLD2 COMPONENT COMPARISON")
print(comparison.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
