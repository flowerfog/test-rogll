"""Raw-data-first investment audit for a STRIDE-like complete-well decoder.

This cell deliberately has no dependency on oof_exports, PF, strong, CP12,
Gold, Neighbor, or any previous experiment. It uses only the competition's
original train horizontal/typewell CSVs.

The audit answers four questions before any model is built:

1. Is the hidden-zone structural surface low dimensional (surface oracle)?
2. Does local GR matching exhibit persistent cycle skips?
3. Can complete-well evidence reverse local mistakes on the same candidate set?
4. Do affine reference calibration and a Cauchy likelihood help inside the same
   complete-well scoring skeleton?

Truth is used to define translated diagnostic candidates and fit surface
oracles. Consequently this experiment is non-deployable by design. Its output
is an investment decision, not a validation score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "rawdata_reasoning_phase0_v0"
SEED = 20260721
OFFSET_GRID_FT = np.arange(-64.0, 64.0001, 2.0, dtype=np.float64)
ZERO_OFFSET_INDEX = int(np.argmin(np.abs(OFFSET_GRID_FT)))
EVAL_STRIDE = 4
SEGMENT_MD_FT = 256.0
MIN_SEGMENT_SAMPLES = 20
PREFIX_SURFACE_WINDOW_FT = 1500.0


def _find_data_root() -> Path:
    candidates = [
        Path.cwd(),
        Path("/root/autodl-tmp"),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate original competition train directory")


DATA_ROOT = _find_data_root()
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OUT_DIR = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT
else:
    OUT_DIR = Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        return value

    path.write_text(
        json.dumps(clean(payload), indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _finite(values) -> np.ndarray:
    return np.isfinite(np.asarray(values, dtype=np.float64))


def _interp_nan(x, xp, fp):
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    valid = _finite(xp) & _finite(fp)
    if valid.sum() < 2:
        return np.full(x.shape, np.nan, dtype=np.float64)
    xp = xp[valid]
    fp = fp[valid]
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    unique, unique_index = np.unique(xp, return_index=True)
    fp = fp[unique_index]
    result = np.interp(x, unique, fp)
    result[(x < unique[0]) | (x > unique[-1])] = np.nan
    return result


def _fit_line(x, y, degree=1):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = _finite(x) & _finite(y)
    if valid.sum() < degree + 2:
        return None
    centre = float(np.mean(x[valid]))
    scale = max(float(np.std(x[valid])), 100.0)
    design = np.stack(
        [((x[valid] - centre) / scale) ** power for power in range(degree + 1)],
        axis=1,
    )
    coefficient = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return centre, scale, coefficient


def _predict_line(fit, x):
    if fit is None:
        return np.full(np.asarray(x).shape, np.nan, dtype=np.float64)
    centre, scale, coefficient = fit
    u = (np.asarray(x, dtype=np.float64) - centre) / scale
    result = np.zeros(u.shape, dtype=np.float64)
    for power, value in enumerate(coefficient):
        result += value * u**power
    return result


def _robust_affine(reference, observed):
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    valid = _finite(reference) & _finite(observed)
    if valid.sum() < 64:
        return 1.0, 0.0, np.nan, int(valid.sum())
    x = reference[valid]
    y = observed[valid]
    keep = np.ones(len(x), dtype=bool)
    coefficient = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(5):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        coefficient = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        coefficient[0] = np.clip(coefficient[0], 0.5, 1.5)
        residual = y - (coefficient[0] * x + coefficient[1])
        centre = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - centre)) + 1e-6
        keep = np.abs(residual - centre) <= 3.5 * scale
        if keep.sum() < 64:
            keep[:] = True
            break
    residual = y - (coefficient[0] * x + coefficient[1])
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    return float(coefficient[0]), float(coefficient[1]), float(noise), int(valid.sum())


def _stable_fold_table(wells):
    scored = []
    for well in wells:
        digest = hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    folds = np.empty(len(ordered), dtype=np.int64)
    for fold, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        folds[indices] = fold
    return pd.DataFrame({"well": ordered, "fold": folds})


def _surface_audit(md, z, truth, tvt_input, known, evaluation):
    anchor_index = int(np.flatnonzero(known)[-1])
    eval_md = md[evaluation]
    eval_z = z[evaluation]
    eval_truth = truth[evaluation]
    true_surface = eval_truth + eval_z
    anchor_tvt = float(tvt_input[anchor_index])

    distance = eval_md - md[anchor_index]
    flat = np.full(eval_truth.shape, anchor_tvt)

    prefix_window = known & (md >= md[anchor_index] - PREFIX_SURFACE_WINDOW_FT)
    prefix_surface = tvt_input[prefix_window] + z[prefix_window]
    prefix_md = md[prefix_window]
    prefix_linear = _predict_line(_fit_line(prefix_md, prefix_surface, 1), eval_md) - eval_z
    prefix_quadratic = _predict_line(_fit_line(prefix_md, prefix_surface, 2), eval_md) - eval_z
    oracle_linear = _predict_line(_fit_line(eval_md, true_surface, 1), eval_md) - eval_z
    oracle_quadratic = _predict_line(_fit_line(eval_md, true_surface, 2), eval_md) - eval_z

    known_surface = tvt_input[prefix_window] + z[prefix_window]
    known_fit = _fit_line(prefix_md, known_surface, 1)
    eval_fit = _fit_line(eval_md, true_surface, 1)
    known_slope = np.nan if known_fit is None else float(known_fit[2][1] / known_fit[1])
    eval_slope = np.nan if eval_fit is None else float(eval_fit[2][1] / eval_fit[1])

    predictions = {
        "flat": flat,
        "prefix_linear_surface": prefix_linear,
        "prefix_quadratic_surface": prefix_quadratic,
        "oracle_linear_surface": oracle_linear,
        "oracle_quadratic_surface": oracle_quadratic,
    }
    rows = []
    for method, prediction in predictions.items():
        valid = _finite(prediction) & _finite(eval_truth)
        error = prediction[valid] - eval_truth[valid]
        rows.append(
            {
                "method": method,
                "rows": int(valid.sum()),
                "sse": float(np.sum(error**2)),
                "sae": float(np.sum(np.abs(error))),
            }
        )
    return rows, {
        "anchor_md": float(md[anchor_index]),
        "eval_length_ft": float(np.nanmax(distance)),
        "known_surface_slope": known_slope,
        "eval_surface_slope": eval_slope,
        "surface_slope_change": eval_slope - known_slope,
    }, distance, eval_truth, predictions


def _loss_matrix(observed, reference_matrix, scale, likelihood):
    difference = reference_matrix - observed[:, None]
    valid = _finite(reference_matrix) & _finite(observed)[:, None]
    normalised = difference / max(float(scale), 1.0)
    if likelihood == "gaussian":
        loss = normalised**2
    elif likelihood == "cauchy":
        loss = np.log1p(normalised**2)
    else:
        raise ValueError(likelihood)
    loss[~valid] = np.nan
    return loss


def _persistent_alias(segment_offsets):
    best = 0
    current = 0
    previous = np.nan
    for offset in segment_offsets:
        alias = abs(offset) >= 12.0
        coherent = alias and np.isfinite(previous) and np.sign(offset) == np.sign(previous) and abs(offset - previous) <= 6.0
        current = current + 1 if coherent else (1 if alias else 0)
        best = max(best, current)
        previous = offset
    return int(best)


def _gr_audit(md, truth, horizontal_gr, tvt_input, known, evaluation, type_frame):
    type_tvt = type_frame["TVT"].to_numpy(np.float64)
    type_gr = type_frame["GR"].to_numpy(np.float64)

    known_reference = _interp_nan(tvt_input[known], type_tvt, type_gr)
    gain, bias, calibration_noise, calibration_rows = _robust_affine(
        known_reference, horizontal_gr[known]
    )
    if not np.isfinite(calibration_noise):
        calibration_noise = 12.0
    scale = float(np.clip(calibration_noise, 3.0, 30.0))

    eval_indices = np.flatnonzero(evaluation & _finite(horizontal_gr))[::EVAL_STRIDE]
    if len(eval_indices) < 100:
        raise RuntimeError(f"only {len(eval_indices)} sampled GR evaluation rows")
    eval_md = md[eval_indices]
    eval_truth = truth[eval_indices]
    observed = horizontal_gr[eval_indices]
    candidate_tvt = eval_truth[:, None] + OFFSET_GRID_FT[None, :]
    raw_reference = _interp_nan(candidate_tvt.ravel(), type_tvt, type_gr).reshape(candidate_tvt.shape)
    affine_reference = gain * raw_reference + bias

    segment_id = np.floor((eval_md - eval_md[0]) / SEGMENT_MD_FT).astype(np.int64)
    arms = []
    for calibration, reference in (("raw", raw_reference), ("affine", affine_reference)):
        for likelihood in ("gaussian", "cauchy"):
            loss = _loss_matrix(observed, reference, scale, likelihood)
            total = np.nansum(loss, axis=0)
            coverage = np.sum(_finite(loss), axis=0)
            total[coverage < max(50, int(0.5 * len(loss)))] = np.inf
            whole_index = int(np.argmin(total))
            whole_offset = float(OFFSET_GRID_FT[whole_index])
            zero_rank = int(np.sum(total < total[ZERO_OFFSET_INDEX]) + 1)

            local_offsets = []
            local_counts = []
            local_ranks = []
            for segment in np.unique(segment_id):
                rows = segment_id == segment
                if rows.sum() < MIN_SEGMENT_SAMPLES:
                    continue
                segment_loss = np.nansum(loss[rows], axis=0)
                segment_coverage = np.sum(_finite(loss[rows]), axis=0)
                segment_loss[segment_coverage < int(0.5 * rows.sum())] = np.inf
                selected = int(np.argmin(segment_loss))
                local_offsets.append(float(OFFSET_GRID_FT[selected]))
                local_counts.append(int(rows.sum()))
                local_ranks.append(int(np.sum(segment_loss < segment_loss[ZERO_OFFSET_INDEX]) + 1))
            if not local_offsets:
                raise RuntimeError("no eligible GR segments")
            local_offsets = np.asarray(local_offsets, dtype=np.float64)
            local_counts = np.asarray(local_counts, dtype=np.float64)
            local_sse = float(np.sum(local_counts * local_offsets**2))
            whole_sse = float(np.sum(local_counts) * whole_offset**2)
            arms.append(
                {
                    "calibration": calibration,
                    "likelihood": likelihood,
                    "sampled_rows": int(np.sum(local_counts)),
                    "segments": int(len(local_offsets)),
                    "local_sse": local_sse,
                    "whole_sse": whole_sse,
                    "local_abs_error_sum": float(np.sum(local_counts * np.abs(local_offsets))),
                    "whole_abs_error_sum": float(np.sum(local_counts) * abs(whole_offset)),
                    "whole_offset_ft": whole_offset,
                    "whole_zero_rank": zero_rank,
                    "local_median_zero_rank": float(np.median(local_ranks)),
                    "local_alias_row_fraction": float(
                        np.sum(local_counts[np.abs(local_offsets) >= 12.0]) / np.sum(local_counts)
                    ),
                    "local_cycle24_row_fraction": float(
                        np.sum(local_counts[(np.abs(local_offsets) >= 18.0) & (np.abs(local_offsets) <= 30.0)])
                        / np.sum(local_counts)
                    ),
                    "longest_persistent_alias_segments": _persistent_alias(local_offsets),
                }
            )
    calibration = {
        "affine_gain": gain,
        "affine_bias": bias,
        "calibration_noise": scale,
        "calibration_rows": calibration_rows,
    }
    return arms, calibration


def _summarise_surface(frame):
    rows = []
    for method, group in frame.groupby("method", sort=False):
        count = int(group["rows"].sum())
        sse = float(group["sse"].sum())
        rows.append(
            {
                "method": method,
                "rows": count,
                "wells": int(group["well"].nunique()),
                "exact_sse": sse,
                "pooled_rmse": math.sqrt(sse / count),
                "pooled_mae": float(group["sae"].sum() / count),
                "median_well_rmse": float(np.median(np.sqrt(group["sse"] / group["rows"]))),
                "p95_well_rmse": float(np.quantile(np.sqrt(group["sse"] / group["rows"]), 0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_rmse")


def _summarise_gr(frame):
    rows = []
    for (calibration, likelihood), group in frame.groupby(["calibration", "likelihood"]):
        count = int(group["sampled_rows"].sum())
        local_sse = float(group["local_sse"].sum())
        whole_sse = float(group["whole_sse"].sum())
        rows.append(
            {
                "calibration": calibration,
                "likelihood": likelihood,
                "rows": count,
                "wells": int(group["well"].nunique()),
                "segments": int(group["segments"].sum()),
                "local_rmse": math.sqrt(local_sse / count),
                "whole_rmse": math.sqrt(whole_sse / count),
                "whole_gain_vs_local": math.sqrt(local_sse / count) - math.sqrt(whole_sse / count),
                "whole_relative_mse_gain": 1.0 - whole_sse / local_sse,
                "whole_within5_well_fraction": float(np.mean(np.abs(group["whole_offset_ft"]) <= 5.0)),
                "whole_median_zero_rank": float(np.median(group["whole_zero_rank"])),
                "local_alias_row_fraction": float(
                    np.average(group["local_alias_row_fraction"], weights=group["sampled_rows"])
                ),
                "local_cycle24_row_fraction": float(
                    np.average(group["local_cycle24_row_fraction"], weights=group["sampled_rows"])
                ),
                "persistent_alias_well_fraction": float(
                    np.mean(group["longest_persistent_alias_segments"] >= 3)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("whole_rmse")


def _distance_summary(well_predictions):
    rows = []
    bands = (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    )
    for method in ("flat", "prefix_linear_surface", "prefix_quadratic_surface"):
        for band, low, high in bands:
            sse = 0.0
            count = 0
            for payload in well_predictions:
                mask = (payload["distance"] >= low) & (payload["distance"] < high)
                prediction = payload[method]
                truth = payload["truth"]
                valid = mask & _finite(prediction) & _finite(truth)
                error = prediction[valid] - truth[valid]
                sse += float(np.sum(error**2))
                count += int(valid.sum())
            rows.append(
                {
                    "method": method,
                    "distance_band": band,
                    "rows": count,
                    "exact_sse": sse,
                    "rmse": math.sqrt(sse / count) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)


started = time.time()
horizontal_paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
wells = [path.name.split("__", 1)[0] for path in horizontal_paths]
folds = _stable_fold_table(wells)
folds.to_csv(OUT_DIR / "raw_grouped_fivefold.csv", index=False)
max_wells = int(os.environ.get("PHASE0_MAX_WELLS", "0"))
if max_wells > 0:
    wells = wells[:max_wells]

surface_rows = []
well_rows = []
gr_rows = []
calibration_rows = []
failures = []
well_predictions = []

for number, well in enumerate(wells, start=1):
    try:
        horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        horizontal = horizontal.sort_values("MD").reset_index(drop=True)
        md = horizontal["MD"].to_numpy(np.float64)
        z = horizontal["Z"].to_numpy(np.float64)
        truth = horizontal["TVT"].to_numpy(np.float64)
        horizontal_gr = horizontal["GR"].to_numpy(np.float64)
        tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
        known = _finite(tvt_input) & _finite(truth)
        evaluation = (~_finite(tvt_input)) & _finite(truth)
        if known.sum() < 100 or evaluation.sum() < 100:
            raise RuntimeError(f"known/evaluation rows={known.sum()}/{evaluation.sum()}")

        methods, well_info, distance, eval_truth, predictions = _surface_audit(
            md, z, truth, tvt_input, known, evaluation
        )
        for row in methods:
            row.update({"well": well})
            surface_rows.append(row)
        well_info.update(
            {
                "well": well,
                "known_rows": int(known.sum()),
                "evaluation_rows": int(evaluation.sum()),
            }
        )
        well_rows.append(well_info)
        well_predictions.append(
            {"well": well, "distance": distance, "truth": eval_truth, **predictions}
        )

        arms, calibration = _gr_audit(
            md, truth, horizontal_gr, tvt_input, known, evaluation, typewell
        )
        for row in arms:
            row["well"] = well
            gr_rows.append(row)
        calibration["well"] = well
        calibration_rows.append(calibration)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(wells):
        print(f"raw phase0 {number}/{len(wells)} failures={len(failures)}", flush=True)

surface_frame = pd.DataFrame(surface_rows)
well_frame = pd.DataFrame(well_rows)
gr_frame = pd.DataFrame(gr_rows)
calibration_frame = pd.DataFrame(calibration_rows)
failure_frame = pd.DataFrame(failures, columns=["well", "reason"])

surface_summary = _summarise_surface(surface_frame)
gr_summary = _summarise_gr(gr_frame)
distance_summary = _distance_summary(well_predictions)

surface_frame.to_csv(OUT_DIR / "surface_by_well.csv", index=False)
well_frame.to_csv(OUT_DIR / "well_structure_diagnostics.csv", index=False)
gr_frame.to_csv(OUT_DIR / "gr_sequence_by_well.csv", index=False)
calibration_frame.to_csv(OUT_DIR / "reference_calibration_by_well.csv", index=False)
failure_frame.to_csv(OUT_DIR / "failures.csv", index=False)
surface_summary.to_csv(OUT_DIR / "surface_summary.csv", index=False)
gr_summary.to_csv(OUT_DIR / "gr_sequence_summary.csv", index=False)
distance_summary.to_csv(OUT_DIR / "distance_strata.csv", index=False)

flat_wells = surface_frame[surface_frame["method"].eq("flat")].copy()
flat_wells["rmse"] = np.sqrt(flat_wells["sse"] / flat_wells["rows"])
flat_wells.sort_values("rmse", ascending=False).head(max(1, int(math.ceil(0.2 * len(flat_wells))))).to_csv(
    OUT_DIR / "worst_20_percent_flat_wells.csv", index=False
)

surface_lookup = surface_summary.set_index("method")["pooled_rmse"].to_dict()
best_gr = gr_summary.iloc[0].to_dict()
gates = {
    "raw_record_success_fraction_ge_0p98": min(
        well_frame["well"].nunique(), gr_frame["well"].nunique()
    ) / len(wells) >= 0.98,
    "oracle_quadratic_surface_rmse_le_4p5": surface_lookup.get("oracle_quadratic_surface", np.inf) <= 4.5,
    "persistent_cycle_skip_well_fraction_ge_0p10": best_gr["persistent_alias_well_fraction"] >= 0.10,
    "complete_well_relative_mse_gain_ge_0p20": best_gr["whole_relative_mse_gain"] >= 0.20,
}
if not gates["oracle_quadratic_surface_rmse_le_4p5"]:
    decision = "STOP -> structural surface is not representable enough; do not build STRIDE"
elif not gates["complete_well_relative_mse_gain_ge_0p20"]:
    decision = "HOLD -> translated candidate audit does not justify a beam; expand the nonrigid candidate family first"
elif not gates["persistent_cycle_skip_well_fraction_ge_0p10"]:
    decision = "HOLD -> errors are not dominated by persistent aliases; sequence decoding is not the first investment"
else:
    decision = "GO -> build a raw-data-only segment lattice; fit every prior inside outer-train folds"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "raw_data_first_investment_audit",
    "data_root": str(DATA_ROOT),
    "upstream_dependencies": [],
    "deployable": False,
    "truth_usage": "surface oracles and translated diagnostic candidate shapes only",
    "wells_discovered": len(wells),
    "surface_wells_completed": int(well_frame["well"].nunique()),
    "gr_wells_completed": int(gr_frame["well"].nunique()),
    "failures": len(failures),
    "official_mask": "finite TVT_input is visible; missing TVT_input with finite TVT is scored",
    "fold_artifact": str(OUT_DIR / "raw_grouped_fivefold.csv"),
    "surface_summary": surface_summary.to_dict(orient="records"),
    "gr_sequence_summary": gr_summary.to_dict(orient="records"),
    "distance_strata": distance_summary.to_dict(orient="records"),
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "decision": decision,
    "elapsed_minutes": (time.time() - started) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSURFACE SUMMARY")
print(surface_summary.to_string(index=False))
print("\nGR SEQUENCE SUMMARY")
print(gr_summary.to_string(index=False))
print("\nDISTANCE STRATA")
print(distance_summary.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT_DIR)
