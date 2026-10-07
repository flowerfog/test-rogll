"""Matched audit of two STRIDE likelihood components on the public beam engine.

This is deliberately not the full STRIDE joint (segment length, delta-TVT)
decoder.  It first tests whether the two inexpensive differences documented in
the STRIDE working note transfer to our data and OOF protocol:

* visible-prefix robust affine calibration of typewell GR;
* heavy-tailed Cauchy emission instead of squared-error/Gaussian emission.

Folds 3/4/5 select one frozen recipe. Fold 2 is evaluated once; Fold 1 is never
read. Every prediction uses only the physical well's legal test-time inputs.
"""

from pathlib import Path
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from numba import njit

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


EXPERIMENT = "stride_component_parity_audit_v0"
ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLDS = (3, 4, 5)
VALID_FOLD = 2
PRIOR_FACTORS = (0.10, 0.25, 0.50, 1.00)
BEAMS = (
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
)
STRATA = (
    ("near_0_500", 0.0, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _robust_scale(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    scale = max(1.4826 * mad, 0.25 * float(np.std(values)), 1e-3)
    return centre, scale


def _deduplicate(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).dropna()
    frame = frame.groupby("axis", as_index=False)["value"].mean().sort_values("axis")
    return frame["axis"].to_numpy(np.float64), frame["value"].to_numpy(np.float64)


def _prefix_calibration(horizontal, tw_tvt, tw_gr):
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    observed = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    expected = np.interp(tvt_input, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    valid = np.isfinite(tvt_input) & np.isfinite(observed) & np.isfinite(expected)
    x = expected[valid]
    y = observed[valid]
    if len(x) < 32:
        shift = float(np.median(y) - np.median(x)) if len(x) else 0.0
        return 1.0, shift, int(len(x)), np.nan
    keep = np.ones(len(x), bool)
    gain, intercept = 1.0, 0.0
    for _ in range(6):
        design = np.column_stack([x[keep], np.ones(int(keep.sum()))])
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (gain * x + intercept)
        centre, scale = _robust_scale(residual[keep])
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if next_keep.sum() < 24 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    gain = float(np.clip(gain, 0.20, 5.0))
    residual = y[keep] - (gain * x[keep] + intercept)
    rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan
    return gain, float(intercept), int(keep.sum()), rmse


def _smooth(values, radius):
    values = np.asarray(values, np.float64)
    if radius <= 0 or len(values) < 5:
        return values.copy()
    window = min(2 * int(radius) + 1, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values.copy()
    if savgol_filter is not None:
        return savgol_filter(values, window, min(2, window - 1)).astype(np.float64)
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(np.float64)


@njit(cache=True)
def _beam_decode(observed, tw_gr, start_index, beam_size, move_cost, emission_scale,
                 prior_factor, likelihood_mode):
    n = len(observed)
    nt = len(tw_gr)
    maximum = beam_size * 5
    beam_index = np.zeros(beam_size, np.int64)
    beam_index[0] = start_index
    beam_cost = np.full(beam_size, 1e30)
    beam_cost[0] = 0.0
    beam_count = 1
    history_index = np.zeros((n, beam_size), np.int64)
    history_parent = np.zeros((n, beam_size), np.int64)
    candidate_index = np.zeros(maximum, np.int64)
    candidate_cost = np.full(maximum, 1e30)
    candidate_parent = np.zeros(maximum, np.int64)

    for step in range(n):
        count = 0
        value = observed[step]
        for parent in range(beam_count):
            old_index = beam_index[parent]
            old_cost = beam_cost[parent]
            for move in range(-2, 3):
                new_index = old_index + move
                if new_index < 0 or new_index >= nt:
                    continue
                residual = value - tw_gr[new_index]
                ratio2 = residual * residual / max(emission_scale, 1e-9)
                if likelihood_mode == 0:
                    emission = ratio2
                else:
                    emission = np.log1p(ratio2)
                total = old_cost + emission + prior_factor * move_cost * abs(move)
                found = -1
                for item in range(count):
                    if candidate_index[item] == new_index:
                        found = item
                        break
                if found >= 0:
                    if total < candidate_cost[found]:
                        candidate_cost[found] = total
                        candidate_parent[found] = parent
                elif count < maximum:
                    candidate_index[count] = new_index
                    candidate_cost[count] = total
                    candidate_parent[count] = parent
                    count += 1

        kept = min(beam_size, count)
        for left in range(kept):
            minimum = left
            for right in range(left + 1, count):
                if candidate_cost[right] < candidate_cost[minimum]:
                    minimum = right
            if minimum != left:
                candidate_index[left], candidate_index[minimum] = candidate_index[minimum], candidate_index[left]
                candidate_cost[left], candidate_cost[minimum] = candidate_cost[minimum], candidate_cost[left]
                candidate_parent[left], candidate_parent[minimum] = candidate_parent[minimum], candidate_parent[left]
        for item in range(kept):
            history_index[step, item] = candidate_index[item]
            history_parent[step, item] = candidate_parent[item]
            beam_index[item] = candidate_index[item]
            beam_cost[item] = candidate_cost[item]
        beam_count = kept

    best = 0
    for item in range(1, beam_count):
        if beam_cost[item] < beam_cost[best]:
            best = item
    path = np.zeros(n, np.int64)
    parent = best
    for step in range(n - 1, -1, -1):
        path[step] = history_index[step, parent]
        parent = history_parent[step, parent]
    return path


def _load_well(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(
        limit_direction="both"
    ).to_numpy(np.float64)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    evaluation = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    if not len(known) or not len(evaluation):
        raise RuntimeError(f"{well}: missing prefix/evaluation rows")
    tw_tvt, tw_gr = _deduplicate(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    if len(tw_tvt) < 32:
        raise RuntimeError(f"{well}: short typewell")
    gain, intercept, calibration_rows, calibration_rmse = _prefix_calibration(
        horizontal, tw_tvt, tw_gr
    )
    return {
        "well": well,
        "horizontal": horizontal,
        "md": md,
        "gr": gr,
        "truth": truth,
        "tvt_input": tvt_input,
        "known": known,
        "evaluation": evaluation,
        "anchor_md": float(md[known[-1]]),
        "last_tvt": float(tvt_input[known[-1]]),
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr,
        "gain": gain,
        "intercept": intercept,
        "calibration_rows": calibration_rows,
        "calibration_rmse": calibration_rmse,
    }


def _predict(record, calibrated, likelihood, prior_factor, calibration_override=None):
    tw_gr = record["tw_gr"].copy()
    if calibrated:
        gain, intercept = (
            calibration_override
            if calibration_override is not None
            else (record["gain"], record["intercept"])
        )
        tw_gr = gain * tw_gr + intercept
    evaluation = record["evaluation"]
    observed = record["gr"][evaluation]
    start = int(np.argmin(np.abs(record["tw_tvt"] - record["last_tvt"])))
    paths = []
    mode = 1 if likelihood == "cauchy" else 0
    for beam_size, move_cost, emission_scale, radius, _ in BEAMS:
        smoothed = _smooth(observed, radius)
        indices = _beam_decode(
            smoothed,
            tw_gr,
            start,
            int(beam_size),
            float(move_cost),
            float(emission_scale),
            float(prior_factor),
            mode,
        )
        paths.append(record["tw_tvt"][indices])
    return np.mean(np.stack(paths, axis=0), axis=0).astype(np.float64)


def _row_metrics(record, prediction, fold, config):
    index = record["evaluation"]
    truth = record["truth"][index]
    distance = record["md"][index] - record["anchor_md"]
    error = prediction - truth
    row = {
        "well": record["well"],
        "fold": int(fold),
        "config": config,
        "rows": len(error),
        "exact_sse": float(np.sum(error**2)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
        "calibration_gain": record["gain"],
        "calibration_intercept": record["intercept"],
        "calibration_rows": record["calibration_rows"],
        "calibration_rmse": record["calibration_rmse"],
    }
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        row[f"{name}_rows"] = int(mask.sum())
        row[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return row


def _summarise(frame):
    rows = int(frame["rows"].sum())
    sse = float(frame["exact_sse"].sum())
    output = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "exact_sse": sse,
        "rmse": float(np.sqrt(sse / rows)),
        "mae": float(frame["absolute_error_sum"].sum() / rows),
    }
    well_rmse = np.sqrt(frame["exact_sse"] / frame["rows"])
    output.update(
        median_well_rmse=float(well_rmse.median()),
        p75_well_rmse=float(well_rmse.quantile(0.75)),
        p95_well_rmse=float(well_rmse.quantile(0.95)),
        max_well_rmse=float(well_rmse.max()),
    )
    for name, _, _ in STRATA:
        count = int(frame[f"{name}_rows"].sum())
        local_sse = float(frame[f"{name}_sse"].sum())
        output[f"{name}_rmse"] = float(np.sqrt(local_sse / count)) if count else np.nan
    return output


if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
if not (DATA_ROOT / "train").exists():
    raise FileNotFoundError(DATA_ROOT / "train")

fold_frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
fold_frame["well"] = fold_frame["well"].astype(str)
fold_frame["fold"] = pd.to_numeric(fold_frame["fold"], errors="raise").astype(int)
fold_map = fold_frame.drop_duplicates("well").set_index("well")["fold"].to_dict()
wells = sorted(
    well for well, fold in fold_map.items()
    if fold in (*SELECTION_FOLDS, VALID_FOLD)
)
expected = sum(int((fold_frame.drop_duplicates("well")["fold"] == fold).sum()) for fold in (*SELECTION_FOLDS, VALID_FOLD))
if len(wells) != expected or len(wells) != 618:
    raise RuntimeError(f"Unexpected wells: {len(wells)} expected={expected}")

CONFIGS = []
for calibrated in (False, True):
    for likelihood in ("gaussian", "cauchy"):
        for prior_factor in PRIOR_FACTORS:
            CONFIGS.append(
                {
                    "config": f"{'cal' if calibrated else 'raw'}_{likelihood}_p{prior_factor:g}",
                    "calibrated": calibrated,
                    "likelihood": likelihood,
                    "prior_factor": prior_factor,
                }
            )

started = time.time()
rows = []
calibrations = {}
records = {}
for number, well in enumerate(wells, 1):
    record = _load_well(well)
    if fold_map[well] == VALID_FOLD:
        records[well] = record
    calibrations[well] = (record["gain"], record["intercept"])
    fold = fold_map[well]
    for config in CONFIGS:
        prediction = _predict(
            record,
            config["calibrated"],
            config["likelihood"],
            config["prior_factor"],
        )
        rows.append(_row_metrics(record, prediction, fold, config["config"]))
    if number % 25 == 0 or number == len(wells):
        print(f"STRIDE parity wells {number}/{len(wells)}", flush=True)

bywell = pd.DataFrame(rows)
bywell.to_csv(OUT_DIR / "all_config_bywell.csv", index=False)
selection_rows = []
for config_name, frame in bywell.groupby("config", sort=True):
    selection = frame[frame["fold"].isin(SELECTION_FOLDS)]
    validation = frame[frame["fold"].eq(VALID_FOLD)]
    selection_rows.append(
        {
            "config": config_name,
            **{f"selection_{key}": value for key, value in _summarise(selection).items()},
            **{f"fold2_{key}": value for key, value in _summarise(validation).items()},
        }
    )
scan = pd.DataFrame(selection_rows).sort_values("selection_rmse").reset_index(drop=True)
scan.to_csv(OUT_DIR / "config_scan.csv", index=False)
selected_name = str(scan.iloc[0]["config"])
selected_config = next(config for config in CONFIGS if config["config"] == selected_name)

selected_fold2 = bywell[
    bywell["config"].eq(selected_name) & bywell["fold"].eq(VALID_FOLD)
].copy()
selected_metrics = _summarise(selected_fold2)

# A shuffled calibration is a matched control for the prefix reference term.
fold2_wells = sorted(well for well in wells if fold_map[well] == VALID_FOLD)
control_rows = []
if selected_config["calibrated"]:
    for index, well in enumerate(fold2_wells):
        donor = fold2_wells[(index + 37) % len(fold2_wells)]
        prediction = _predict(
            records[well],
            True,
            selected_config["likelihood"],
            selected_config["prior_factor"],
            calibration_override=calibrations[donor],
        )
        control_rows.append(
            _row_metrics(records[well], prediction, VALID_FOLD, "shuffled_calibration")
        )
    shuffled_metrics = _summarise(pd.DataFrame(control_rows))
else:
    shuffled_metrics = None

raw_gaussian_candidates = scan[scan["config"].str.startswith("raw_gaussian_")]
raw_gaussian_name = str(
    raw_gaussian_candidates.sort_values("selection_rmse").iloc[0]["config"]
)
raw_fold2 = bywell[
    bywell["config"].eq(raw_gaussian_name) & bywell["fold"].eq(VALID_FOLD)
]
raw_metrics = _summarise(raw_fold2)

gain_vs_raw = float(raw_metrics["rmse"] - selected_metrics["rmse"])
far_gain_vs_raw = float(
    raw_metrics["far_gt2000_rmse"] - selected_metrics["far_gt2000_rmse"]
)
control_pass = bool(
    shuffled_metrics is None
    or shuffled_metrics["rmse"] >= selected_metrics["rmse"] + 0.05
)
strict_pass = bool(gain_vs_raw >= 0.15 and far_gain_vs_raw >= 0.15 and control_pass)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "matched_STRIDE_likelihood_component_parity",
    "protocol": {
        "selection_folds": list(SELECTION_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "sample_unit": "complete physical well",
    },
    "legal_inputs_only": True,
    "uses_neighbor_cp12_gold": False,
    "components_tested": {
        "prefix_typewell_affine_calibration": True,
        "cauchy_emission": True,
        "joint_segment_decoder": False,
    },
    "selected_config": selected_config,
    "raw_gaussian_reference_config": raw_gaussian_name,
    "raw_gaussian_fold2": raw_metrics,
    "selected_fold2": selected_metrics,
    "shuffled_calibration_fold2": shuffled_metrics,
    "gain_vs_raw_gaussian": gain_vs_raw,
    "far_gain_vs_raw_gaussian": far_gain_vs_raw,
    "gates": {
        "gain_vs_raw_gaussian_ge": 0.15,
        "far_gain_vs_raw_gaussian_ge": 0.15,
        "shuffled_calibration_degradation_ge": 0.05,
    },
    "control_pass": control_pass,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> implement outer-train priors and the joint (h_len, delta_TVT) decoder"
        if strict_pass
        else "FAIL -> do not transplant these components blindly; build the joint decoder only as a separate representation test"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nCONFIG SCAN")
print(scan.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
