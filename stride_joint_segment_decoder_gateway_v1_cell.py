"""Fold-2 gateway for a STRIDE-style joint segment decoder.

The decoder searches a joint lattice of segment length and segment TVT rate.
Each complete path starts at the final visible TVT point. Segment emissions
accumulate calibrated typewell/lateral GR mismatch with a Cauchy likelihood;
outer-train priors regularise segment length and rate persistence.

Folds 3/4/5 fit every prior. Fold 2 is decoded once and Fold 1 is never read.
This is a faithful structural reproduction, not an exact reproduction of the
unpublished STRIDE hyperparameters.
"""

from pathlib import Path
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from numba import njit


EXPERIMENT = "stride_joint_segment_decoder_gateway_v1"
ROOT = Path("/root/autodl-tmp")
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "oof_exports"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
NODE_STEP_FT = 32.0
BEAM_SIZE = 32
EMISSION_SAMPLES_PER_SEGMENT = 16
RATE_DELTA_MULTIPLIERS = np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0), np.float64)
LENGTH_PRIOR_WEIGHT = 0.25
PERSISTENCE_WEIGHT = 1.0
MAX_LENGTH_NODES = 16
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


def _robust_location_scale(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return centre, max(1.4826 * mad, 0.25 * float(np.std(values)), 1e-4)


def _deduplicate(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).dropna()
    frame = frame.groupby("axis", as_index=False)["value"].mean().sort_values("axis")
    return frame["axis"].to_numpy(np.float64), frame["value"].to_numpy(np.float64)


def _fill(values):
    return pd.Series(values).interpolate(limit_direction="both").to_numpy(np.float64)


def _calibrate_reference(tvt_input, observed_gr, tw_tvt, tw_gr):
    expected = np.interp(tvt_input, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    valid = np.isfinite(tvt_input) & np.isfinite(observed_gr) & np.isfinite(expected)
    x = expected[valid]
    y = observed_gr[valid]
    if len(x) < 32:
        shift = float(np.median(y) - np.median(x)) if len(x) else 0.0
        calibrated = tw_gr + shift
        residual = y - (x + shift) if len(x) else np.asarray([], np.float64)
        scale = max(_robust_location_scale(residual)[1], 10.0)
        return calibrated, 1.0, shift, scale, int(len(x))
    keep = np.ones(len(x), bool)
    gain, intercept = 1.0, 0.0
    for _ in range(6):
        design = np.column_stack([x[keep], np.ones(int(keep.sum()))])
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (gain * x + intercept)
        centre, scale = _robust_location_scale(residual[keep])
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if next_keep.sum() < 24 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    gain = float(np.clip(gain, 0.20, 5.0))
    calibrated = gain * tw_gr + float(intercept)
    residual = y[keep] - (gain * x[keep] + intercept)
    _, scale = _robust_location_scale(residual)
    return calibrated, gain, float(intercept), float(np.clip(scale, 8.0, 60.0)), int(keep.sum())


def _load_record(well, include_truth=True):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    gr = _fill(pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64))
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    truth = (
        pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
        if include_truth else np.full(len(horizontal), np.nan)
    )
    known = np.flatnonzero(np.isfinite(tvt_input))
    evaluation = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    if not len(known) or (include_truth and not len(evaluation)):
        raise RuntimeError(f"{well}: missing official prefix/evaluation mask")
    tw_tvt, tw_gr = _deduplicate(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    if len(tw_tvt) < 32:
        raise RuntimeError(f"{well}: typewell too short")
    calibrated, gain, intercept, gr_scale, calibration_rows = _calibrate_reference(
        tvt_input, gr, tw_tvt, tw_gr
    )
    anchor = int(known[-1])
    tail = known[md[known] >= md[anchor] - 500.0]
    if len(tail) >= 8 and np.ptp(md[tail]) >= 50.0:
        initial_rate = float(np.polyfit(md[tail], tvt_input[tail], 1)[0])
    else:
        initial_rate = 0.0
    return {
        "well": well,
        "md": md,
        "gr": gr,
        "z": z,
        "truth": truth,
        "tvt_input": tvt_input,
        "known": known,
        "evaluation": evaluation,
        "anchor": anchor,
        "anchor_md": float(md[anchor]),
        "anchor_tvt": float(tvt_input[anchor]),
        "initial_rate": initial_rate,
        "tw_tvt": tw_tvt,
        "tw_gr_raw": tw_gr,
        "tw_gr_calibrated": calibrated,
        "calibration_gain": gain,
        "calibration_intercept": intercept,
        "calibration_rows": calibration_rows,
        "gr_scale": gr_scale,
    }


def _resample_truth(record, step=NODE_STEP_FT):
    index = np.r_[record["anchor"], record["evaluation"]]
    md = record["md"][index]
    tvt = np.r_[record["anchor_tvt"], record["truth"][record["evaluation"]]]
    targets = np.arange(md[0], md[-1] + 0.5 * step, step)
    if targets[-1] < md[-1] - 1e-6:
        targets = np.r_[targets, md[-1]]
    return targets, np.interp(targets, md, tvt)


def _fit_outer_priors(records):
    slope_sequences = []
    decorrelation_lengths = []
    for record in records:
        md, tvt = _resample_truth(record)
        delta_md = np.diff(md)
        slope = np.diff(tvt) / np.maximum(delta_md, 1e-6)
        slope = pd.Series(slope).rolling(3, center=True, min_periods=1).median().to_numpy()
        slope_sequences.append(slope)
        chosen = 4
        for lag in range(2, min(17, max(len(slope) // 3, 3))):
            if len(slope) <= lag + 8:
                break
            correlation = np.corrcoef(slope[:-lag], slope[lag:])[0, 1]
            if not np.isfinite(correlation) or correlation < 0.5:
                chosen = lag
                break
            chosen = lag
        decorrelation_lengths.append(float(chosen * NODE_STEP_FT))
    all_slopes = np.concatenate(slope_sequences)
    length_log = np.log(np.clip(decorrelation_lengths, NODE_STEP_FT, MAX_LENGTH_NODES * NODE_STEP_FT))
    length_mu = float(np.mean(length_log))
    length_sigma = float(max(np.std(length_log), 0.30))
    median_length = float(np.exp(length_mu))
    lag_nodes = max(int(round(median_length / NODE_STEP_FT)), 1)
    changes = []
    for slope in slope_sequences:
        if len(slope) > lag_nodes:
            changes.append(slope[lag_nodes:] - slope[:-lag_nodes])
    all_changes = np.concatenate(changes) if changes else np.diff(all_slopes)
    _, change_scale = _robust_location_scale(all_changes)
    rate_low, rate_high = np.quantile(all_slopes, [0.005, 0.995])
    margin = max(0.25 * (rate_high - rate_low), 0.01)
    quantile_z = np.asarray((-1.25, -0.50, 0.0, 0.50, 1.25), np.float64)
    lengths = np.exp(length_mu + length_sigma * quantile_z)
    length_nodes = np.unique(
        np.clip(np.rint(lengths / NODE_STEP_FT), 1, MAX_LENGTH_NODES).astype(np.int64)
    )
    length_nodes = np.unique(np.r_[1, length_nodes]).astype(np.int64)
    return {
        "length_log_mu": length_mu,
        "length_log_sigma": length_sigma,
        "median_length_ft": median_length,
        "length_nodes": length_nodes,
        "rate_change_scale": float(max(change_scale, 0.0025)),
        "rate_low": float(rate_low - margin),
        "rate_high": float(rate_high + margin),
        "slope_count": int(len(all_slopes)),
        "well_count": int(len(records)),
    }


def _node_rows(record):
    anchor = record["anchor"]
    evaluation = record["evaluation"]
    last = int(evaluation[-1])
    target_md = np.arange(record["md"][anchor], record["md"][last] + 0.5 * NODE_STEP_FT, NODE_STEP_FT)
    rows = [anchor]
    for value in target_md[1:]:
        position = int(np.searchsorted(record["md"], value, side="left"))
        position = min(max(position, anchor + 1), last)
        if position > rows[-1]:
            rows.append(position)
    if rows[-1] != last:
        rows.append(last)
    return np.asarray(rows, np.int64)


@njit(cache=True)
def _interp_sorted(axis, values, query):
    if query <= axis[0]:
        return values[0]
    if query >= axis[-1]:
        return values[-1]
    low = 0
    high = len(axis) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if axis[middle] <= query:
            low = middle
        else:
            high = middle
    fraction = (query - axis[low]) / max(axis[high] - axis[low], 1e-9)
    return values[low] * (1.0 - fraction) + values[high] * fraction


@njit(cache=True)
def _segment_emission(md, gr, row0, row1, tvt0, tvt1, tw_tvt, tw_gr,
                      gr_scale, gaussian_mode, maximum_samples):
    count = row1 - row0
    if count <= 0:
        return 0.0
    samples = min(count, maximum_samples)
    total = 0.0
    for sample in range(samples):
        row = row0 + 1 + ((sample + 1) * count - 1) // samples
        fraction = (md[row] - md[row0]) / max(md[row1] - md[row0], 1e-9)
        candidate_tvt = tvt0 + fraction * (tvt1 - tvt0)
        expected = _interp_sorted(tw_tvt, tw_gr, candidate_tvt)
        standardised = (gr[row] - expected) / max(gr_scale, 1e-6)
        square = min(standardised * standardised, 400.0)
        if gaussian_mode:
            total += 0.5 * square
        else:
            total += np.log1p(square)
    return total * count / samples


@njit(cache=True)
def _joint_decode(md, gr, node_rows, anchor_tvt, initial_rate, tw_tvt, tw_gr,
                  gr_scale, length_nodes, length_mu, length_sigma,
                  rate_change_scale, rate_low, rate_high, delta_multipliers,
                  beam_size, persistence_weight, length_weight, gaussian_mode,
                  maximum_samples):
    nodes = len(node_rows)
    cost = np.full((nodes, beam_size), 1e30)
    state_tvt = np.zeros((nodes, beam_size))
    state_rate = np.zeros((nodes, beam_size))
    parent_node = np.full((nodes, beam_size), -1, np.int64)
    parent_slot = np.full((nodes, beam_size), -1, np.int64)
    cost[0, 0] = 0.0
    state_tvt[0, 0] = anchor_tvt
    state_rate[0, 0] = min(max(initial_rate, rate_low), rate_high)

    for target_node in range(1, nodes):
        for length_position in range(len(length_nodes)):
            source_node = target_node - length_nodes[length_position]
            if source_node < 0:
                continue
            length_ft = md[node_rows[target_node]] - md[node_rows[source_node]]
            if length_ft <= 0:
                continue
            length_z = (np.log(max(length_ft, 1.0)) - length_mu) / max(length_sigma, 1e-6)
            length_cost = length_weight * (0.5 * length_z * length_z + np.log(max(length_ft, 1.0)))
            for source_slot in range(beam_size):
                if cost[source_node, source_slot] >= 1e29:
                    continue
                old_tvt = state_tvt[source_node, source_slot]
                old_rate = state_rate[source_node, source_slot]
                for delta_position in range(len(delta_multipliers)):
                    multiplier = delta_multipliers[delta_position]
                    new_rate = old_rate + multiplier * rate_change_scale
                    new_rate = min(max(new_rate, rate_low), rate_high)
                    new_tvt = old_tvt + new_rate * length_ft
                    emission = _segment_emission(
                        md, gr, node_rows[source_node], node_rows[target_node],
                        old_tvt, new_tvt, tw_tvt, tw_gr, gr_scale,
                        gaussian_mode, maximum_samples,
                    )
                    transition = persistence_weight * 0.5 * multiplier * multiplier
                    new_cost = cost[source_node, source_slot] + emission + transition + length_cost
                    worst_slot = 0
                    for slot in range(1, beam_size):
                        if cost[target_node, slot] > cost[target_node, worst_slot]:
                            worst_slot = slot
                    if new_cost < cost[target_node, worst_slot]:
                        cost[target_node, worst_slot] = new_cost
                        state_tvt[target_node, worst_slot] = new_tvt
                        state_rate[target_node, worst_slot] = new_rate
                        parent_node[target_node, worst_slot] = source_node
                        parent_slot[target_node, worst_slot] = source_slot

    final_slot = 0
    for slot in range(1, beam_size):
        if cost[nodes - 1, slot] < cost[nodes - 1, final_slot]:
            final_slot = slot
    path_node = np.full(nodes, np.nan)
    node = nodes - 1
    slot = final_slot
    while node >= 0 and slot >= 0:
        path_node[node] = state_tvt[node, slot]
        previous_node = parent_node[node, slot]
        previous_slot = parent_slot[node, slot]
        if previous_node < 0:
            break
        # Fill skipped graph nodes by linear interpolation for stable output.
        left_tvt = state_tvt[previous_node, previous_slot]
        right_tvt = state_tvt[node, slot]
        left_md = md[node_rows[previous_node]]
        right_md = md[node_rows[node]]
        for fill_node in range(previous_node + 1, node):
            fraction = (md[node_rows[fill_node]] - left_md) / max(right_md - left_md, 1e-9)
            path_node[fill_node] = left_tvt + fraction * (right_tvt - left_tvt)
        node = previous_node
        slot = previous_slot
    path_node[0] = anchor_tvt
    return path_node, cost[nodes - 1, final_slot]


def _decode(record, priors, mode="full", donor=None):
    if mode == "shuffled_typewell":
        if donor is None:
            raise ValueError("donor is required")
        tw_tvt = donor["tw_tvt"]
        tw_gr_raw = donor["tw_gr_raw"]
        tw_gr, _, _, gr_scale, _ = _calibrate_reference(
            record["tvt_input"], record["gr"], tw_tvt, tw_gr_raw
        )
    else:
        tw_tvt = record["tw_tvt"]
        tw_gr = record["tw_gr_raw"] if mode == "no_calibration" else record["tw_gr_calibrated"]
        if mode == "no_calibration":
            expected = np.interp(
                record["tvt_input"], tw_tvt, tw_gr, left=np.nan, right=np.nan
            )
            known = np.isfinite(record["tvt_input"]) & np.isfinite(expected)
            _, gr_scale = _robust_location_scale(record["gr"][known] - expected[known])
            gr_scale = float(np.clip(gr_scale, 8.0, 60.0))
        else:
            gr_scale = record["gr_scale"]
    node_rows = _node_rows(record)
    persistence_weight = 0.0 if mode == "no_persistence" else PERSISTENCE_WEIGHT
    gaussian_mode = mode == "gaussian"
    path_node, objective = _joint_decode(
        record["md"], record["gr"], node_rows,
        record["anchor_tvt"], record["initial_rate"],
        tw_tvt, tw_gr, gr_scale,
        priors["length_nodes"], priors["length_log_mu"], priors["length_log_sigma"],
        priors["rate_change_scale"], priors["rate_low"], priors["rate_high"],
        RATE_DELTA_MULTIPLIERS, BEAM_SIZE, persistence_weight,
        LENGTH_PRIOR_WEIGHT, gaussian_mode, EMISSION_SAMPLES_PER_SEGMENT,
    )
    node_md = record["md"][node_rows]
    evaluation = record["evaluation"]
    prediction = np.interp(record["md"][evaluation], node_md, path_node)
    return prediction, float(objective), len(node_rows)


def _metric_row(record, prediction, mode, objective, nodes):
    index = record["evaluation"]
    truth = record["truth"][index]
    distance = record["md"][index] - record["anchor_md"]
    error = prediction - truth
    output = {
        "well": record["well"],
        "mode": mode,
        "rows": len(error),
        "exact_sse": float(np.sum(error**2)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
        "objective": objective,
        "nodes": int(nodes),
        "calibration_gain": record["calibration_gain"],
        "calibration_intercept": record["calibration_intercept"],
        "calibration_rows": record["calibration_rows"],
    }
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        output[f"{name}_rows"] = int(mask.sum())
        output[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return output


def _summarise(frame):
    rows = int(frame["rows"].sum())
    sse = float(frame["exact_sse"].sum())
    well_rmse = np.sqrt(frame["exact_sse"] / frame["rows"])
    output = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "exact_sse": sse,
        "rmse": float(np.sqrt(sse / rows)),
        "mae": float(frame["absolute_error_sum"].sum() / rows),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    }
    for name, _, _ in STRATA:
        count = int(frame[f"{name}_rows"].sum())
        sse_local = float(frame[f"{name}_sse"].sum())
        output[f"{name}_rmse"] = float(np.sqrt(sse_local / count)) if count else np.nan
    return output


if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
if not (DATA_ROOT / "train").exists():
    raise FileNotFoundError(DATA_ROOT / "train")

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
folds = folds.drop_duplicates("well")
train_wells = sorted(folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"])
valid_wells = sorted(folds.loc[folds["fold"].eq(VALID_FOLD), "well"])
if len(train_wells) != 463 or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected folds: {len(train_wells)}/{len(valid_wells)}")

started = time.time()
train_records = []
for number, well in enumerate(train_wells, 1):
    train_records.append(_load_record(well))
    if number % 50 == 0 or number == len(train_wells):
        print(f"fit STRIDE priors {number}/{len(train_wells)}", flush=True)
priors = _fit_outer_priors(train_records)
_write_json(OUT_DIR / "outer_train_priors.json", priors)
del train_records

valid_records = []
for number, well in enumerate(valid_wells, 1):
    valid_records.append(_load_record(well))
    if number % 50 == 0 or number == len(valid_wells):
        print(f"prepare Fold2 records {number}/{len(valid_wells)}", flush=True)

# Compile before timing/reporting the actual decodes.
_ = _decode(valid_records[0], priors, mode="full")

modes = ("full", "gaussian", "no_calibration", "no_persistence", "shuffled_typewell")
rows = []
row_predictions = []
for mode in modes:
    for number, record in enumerate(valid_records, 1):
        donor = valid_records[(number - 1 + 37) % len(valid_records)] if mode == "shuffled_typewell" else None
        prediction, objective, nodes = _decode(record, priors, mode=mode, donor=donor)
        rows.append(_metric_row(record, prediction, mode, objective, nodes))
        if mode == "full":
            index = record["evaluation"]
            row_predictions.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": index,
                        "truth": record["truth"][index],
                        "prediction": prediction,
                        "distance_ft": record["md"][index] - record["anchor_md"],
                    }
                )
            )
        if number % 25 == 0 or number == len(valid_records):
            print(f"STRIDE Fold2 {mode} {number}/{len(valid_records)}", flush=True)

bywell = pd.DataFrame(rows)
bywell.to_csv(OUT_DIR / "fold2_bywell_ablation.csv", index=False)
predictions = pd.concat(row_predictions, ignore_index=True)
predictions.to_csv(OUT_DIR / "fold2_rows.csv.gz", index=False, compression="gzip")

summary_rows = []
for mode, frame in bywell.groupby("mode", sort=False):
    summary_rows.append({"mode": mode, **_summarise(frame)})
summary = pd.DataFrame(summary_rows)
full = summary.loc[summary["mode"].eq("full")].iloc[0].to_dict()
for column in ("rmse", "far_gt2000_rmse"):
    summary[f"degradation_{column}_vs_full"] = summary[column] - float(full[column])
summary.to_csv(OUT_DIR / "fold2_ablation_summary.csv", index=False)

strong_context = None
row_error_correlation = np.nan
well_error_correlation = np.nan
blend_scan = None
if POOL_PATH is not None:
    header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
    if "strong" in header:
        usecols = [column for column in ("id", "well", "row_idx", "_row_idx", "strong") if column in header]
        pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": str, "well": str})
        if "well" not in pool:
            pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
        if "row_idx" not in pool:
            if "_row_idx" in pool:
                pool["row_idx"] = pd.to_numeric(pool["_row_idx"], errors="raise").astype(int)
            else:
                pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(int)
        pool = pool[["well", "row_idx", "strong"]]
        merged = predictions.merge(pool, on=["well", "row_idx"], how="left", validate="one_to_one")
        if merged["strong"].notna().all():
            stride_error = merged["prediction"].to_numpy(float) - merged["truth"].to_numpy(float)
            strong_error = merged["strong"].to_numpy(float) - merged["truth"].to_numpy(float)
            row_error_correlation = float(np.corrcoef(stride_error, strong_error)[0, 1])
            well_errors = merged.assign(stride_error=stride_error, strong_error=strong_error).groupby("well")[["stride_error", "strong_error"]].mean()
            well_error_correlation = float(well_errors.corr().iloc[0, 1])
            strong_rmse = float(np.sqrt(np.mean(strong_error**2)))
            strong_context = {"rows": len(merged), "wells": merged["well"].nunique(), "rmse": strong_rmse}
            blend_rows = []
            for weight in np.linspace(0.0, 1.0, 21):
                blended = (1.0 - weight) * merged["strong"].to_numpy(float) + weight * merged["prediction"].to_numpy(float)
                blend_rows.append(
                    {"stride_weight": float(weight), "rmse": float(np.sqrt(np.mean((blended - merged["truth"])**2)))}
                )
            blend_scan = pd.DataFrame(blend_rows)
            blend_scan.to_csv(OUT_DIR / "fold2_strong_blend_diagnostic.csv", index=False)

def _mode_metric(mode):
    return summary.loc[summary["mode"].eq(mode)].iloc[0].to_dict()

gaussian = _mode_metric("gaussian")
no_calibration = _mode_metric("no_calibration")
no_persistence = _mode_metric("no_persistence")
shuffled = _mode_metric("shuffled_typewell")
path_pass = bool(float(full["rmse"]) <= 11.0 and float(full["far_gt2000_rmse"]) <= 13.0)
causal_pass = bool(
    float(shuffled["rmse"]) >= float(full["rmse"]) + 1.0
    and float(no_persistence["rmse"]) >= float(full["rmse"]) + 0.30
)
independence_pass = bool(np.isfinite(row_error_correlation) and row_error_correlation <= 0.50)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_joint_segment_STRIDE_gateway",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_cp12_gold": False,
    "protocol": {
        "prior_train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
    },
    "decoder": {
        "state": "joint segment length and segment TVT rate",
        "node_step_ft": NODE_STEP_FT,
        "beam_size": BEAM_SIZE,
        "emission": "segment-aggregated Cauchy",
        "prefix_typewell_calibration": "robust affine",
        "full_sequence_backtracking": True,
    },
    "outer_train_priors": priors,
    "fold2": full,
    "ablations": {
        "gaussian": gaussian,
        "no_calibration": no_calibration,
        "no_persistence": no_persistence,
        "shuffled_typewell": shuffled,
    },
    "strong_context": strong_context,
    "error_correlation": {
        "row_level": row_error_correlation,
        "well_mean_error": well_error_correlation,
    },
    "best_same_fold_blend": (
        blend_scan.sort_values("rmse").iloc[0].to_dict() if blend_scan is not None else None
    ),
    "gates": {
        "rmse_le": 11.0,
        "far_gt2000_rmse_le": 13.0,
        "shuffled_typewell_degradation_ge": 1.0,
        "no_persistence_degradation_ge": 0.30,
        "row_error_correlation_with_strong_le": 0.50,
    },
    "path_pass": path_pass,
    "causal_pass": causal_pass,
    "independence_pass": independence_pass,
    "strict_pass": bool(path_pass and causal_pass and independence_pass),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze decoder, generate all-fold OOF, then cross-fit STRIDE/Strong blending"
        if path_pass and causal_pass and independence_pass
        else "FAIL -> inspect the ablation table before changing the lattice or priors"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nOUTER-TRAIN PRIORS")
print(json.dumps(priors, indent=2, default=_json_default), flush=True)
print("\nFOLD2 ABLATIONS")
print(summary.to_string(index=False), flush=True)
if blend_scan is not None:
    print("\nSTRONG BLEND DIAGNOSTIC")
    print(blend_scan.sort_values("rmse").head(10).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
