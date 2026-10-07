"""Nested observability and calibration audit for the STRIDE segment lattice.

Protocol
--------
* Folds 4/5 fit trajectory priors.
* Fold 3 selects lattice reachability, calibration, likelihood, and prior dose.
* Fold 2 is decoded once with the frozen recipe.
* Fold 1 is never read.

The cell imports definitions (not execution) from
``stride_joint_segment_decoder_gateway_v1_cell.py``. Keep that source file in
``/root/autodl-tmp`` beside this cell.
"""

from pathlib import Path
import ast
import json
import shutil
import time

import numpy as np
import pandas as pd
from numba import njit


ROOT = Path("/root/autodl-tmp")
SOURCE_CANDIDATES = (
    ROOT / "stride_joint_segment_decoder_gateway_v1_cell.py",
    Path.cwd() / "stride_joint_segment_decoder_gateway_v1_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(
        "Upload stride_joint_segment_decoder_gateway_v1_cell.py to /root/autodl-tmp"
    )

source = SOURCE_PATH.read_text(encoding="utf-8")
marker = "if FOLD_PATH is None:"
if marker not in source:
    raise RuntimeError("Cannot locate the definition boundary in the v1 source")
V = {"__name__": "stride_joint_segment_v1_definitions", "__file__": str(SOURCE_PATH)}
exec(compile(source.split(marker, 1)[0], str(SOURCE_PATH), "exec"), V)

EXPERIMENT = "stride_lattice_observability_nested_v2"
DATA_ROOT = V["DATA_ROOT"]
OOF_ROOT = V["OOF_ROOT"]
FOLD_PATH = V["FOLD_PATH"]
POOL_PATH = V["POOL_PATH"]
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
VALID_FOLD = 2
NODE_STEP_FT = float(V["NODE_STEP_FT"])
RATE_MULTIPLIERS = np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0), np.float64)
EMISSION_SAMPLES = 24
LATTICE_CONFIGS = (
    ("short_b32_r1", np.asarray((1, 3, 4, 5, 6, 7), np.int64), 32, 1.0),
    ("medium_b32_r1", np.asarray((1, 2, 3, 4, 6, 8, 12, 16), np.int64), 32, 1.0),
    ("long_b32_r1", np.asarray((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), np.int64), 32, 1.0),
    ("long_b64_r1", np.asarray((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), np.int64), 64, 1.0),
    ("long_b128_r1", np.asarray((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), np.int64), 128, 1.0),
    ("long_b64_r2", np.asarray((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), np.int64), 64, 2.0),
    ("long_b128_r2", np.asarray((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), np.int64), 128, 2.0),
    ("long_b128_r4", np.asarray((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), np.int64), 128, 4.0),
)
CALIBRATION_MODES = ("raw", "shift", "shrink_affine")
LIKELIHOODS = ("gaussian", "cauchy")
PERSISTENCE_WEIGHTS = (0.0, 0.05, 0.15, 0.50)
LENGTH_WEIGHTS = (0.0, 0.01, 0.05, 0.20)
STRATA = V["STRATA"]

_load_record = V["_load_record"]
_fit_outer_priors = V["_fit_outer_priors"]
_node_rows = V["_node_rows"]
_joint_decode = V["_joint_decode"]
_robust_location_scale = V["_robust_location_scale"]
_calibrate_reference = V["_calibrate_reference"]


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


@njit(cache=True)
def _truth_segment_cost(md, truth, row0, row1, tvt0, tvt1, maximum_samples):
    count = row1 - row0
    if count <= 0:
        return 0.0
    samples = min(count, maximum_samples)
    total = 0.0
    for sample in range(samples):
        row = row0 + 1 + ((sample + 1) * count - 1) // samples
        fraction = (md[row] - md[row0]) / max(md[row1] - md[row0], 1e-9)
        prediction = tvt0 + fraction * (tvt1 - tvt0)
        residual = prediction - truth[row]
        total += residual * residual
    return total * count / samples


@njit(cache=True)
def _oracle_lattice_decode(md, truth, node_rows, anchor_tvt, initial_rate,
                           length_nodes, rate_change_scale, rate_low, rate_high,
                           delta_multipliers, beam_size, maximum_samples):
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
            for source_slot in range(beam_size):
                if cost[source_node, source_slot] >= 1e29:
                    continue
                old_tvt = state_tvt[source_node, source_slot]
                old_rate = state_rate[source_node, source_slot]
                for delta_position in range(len(delta_multipliers)):
                    new_rate = old_rate + delta_multipliers[delta_position] * rate_change_scale
                    new_rate = min(max(new_rate, rate_low), rate_high)
                    new_tvt = old_tvt + new_rate * length_ft
                    new_cost = cost[source_node, source_slot] + _truth_segment_cost(
                        md, truth, node_rows[source_node], node_rows[target_node],
                        old_tvt, new_tvt, maximum_samples,
                    )
                    worst = 0
                    for slot in range(1, beam_size):
                        if cost[target_node, slot] > cost[target_node, worst]:
                            worst = slot
                    if new_cost < cost[target_node, worst]:
                        cost[target_node, worst] = new_cost
                        state_tvt[target_node, worst] = new_tvt
                        state_rate[target_node, worst] = new_rate
                        parent_node[target_node, worst] = source_node
                        parent_slot[target_node, worst] = source_slot

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
    return path_node


def _reference(record, calibration_mode, donor=None):
    if donor is not None:
        tw_tvt = donor["tw_tvt"]
        raw = donor["tw_gr_raw"]
    else:
        tw_tvt = record["tw_tvt"]
        raw = record["tw_gr_raw"]
    expected = np.interp(record["tvt_input"], tw_tvt, raw, left=np.nan, right=np.nan)
    valid = np.isfinite(record["tvt_input"]) & np.isfinite(expected)
    observed = record["gr"][valid]
    expected_valid = expected[valid]
    if calibration_mode == "raw":
        reference = raw.copy()
    elif calibration_mode == "shift":
        shift = float(np.median(observed - expected_valid)) if len(observed) else 0.0
        reference = raw + shift
    elif calibration_mode == "shrink_affine":
        _, gain, _, _, _ = _calibrate_reference(
            record["tvt_input"], record["gr"], tw_tvt, raw
        )
        gain = 1.0 + 0.25 * (gain - 1.0)
        intercept = (
            float(np.median(observed) - gain * np.median(expected_valid))
            if len(observed) else 0.0
        )
        reference = gain * raw + intercept
    else:
        raise ValueError(calibration_mode)
    expected_after = np.interp(
        record["tvt_input"], tw_tvt, reference, left=np.nan, right=np.nan
    )
    valid_after = np.isfinite(record["tvt_input"]) & np.isfinite(expected_after)
    _, scale = _robust_location_scale(record["gr"][valid_after] - expected_after[valid_after])
    return tw_tvt, reference, float(np.clip(scale, 8.0, 60.0))


def _continuous_oracle(record):
    node_rows = _node_rows(record)
    node_truth = record["truth"][node_rows].copy()
    node_truth[0] = record["anchor_tvt"]
    prediction = np.interp(
        record["md"][record["evaluation"]], record["md"][node_rows], node_truth
    )
    return prediction


def _quantized_oracle(record, priors, length_nodes, beam_size, rate_scale):
    node_rows = _node_rows(record)
    truth_full = record["truth"].copy()
    truth_full[record["anchor"]] = record["anchor_tvt"]
    path_node = _oracle_lattice_decode(
        record["md"], truth_full, node_rows, record["anchor_tvt"],
        record["initial_rate"], length_nodes,
        priors["rate_change_scale"] * rate_scale,
        priors["rate_low"], priors["rate_high"], RATE_MULTIPLIERS,
        int(beam_size), EMISSION_SAMPLES,
    )
    return np.interp(
        record["md"][record["evaluation"]], record["md"][node_rows], path_node
    )


def _decode(record, priors, lattice, calibration_mode, likelihood,
            persistence_weight, length_weight, donor=None):
    _, length_nodes, beam_size, rate_scale = lattice
    tw_tvt, tw_gr, gr_scale = _reference(record, calibration_mode, donor=donor)
    node_rows = _node_rows(record)
    path_node, objective = _joint_decode(
        record["md"], record["gr"], node_rows, record["anchor_tvt"],
        record["initial_rate"], tw_tvt, tw_gr, gr_scale,
        length_nodes, priors["length_log_mu"], priors["length_log_sigma"],
        priors["rate_change_scale"] * rate_scale,
        priors["rate_low"], priors["rate_high"], RATE_MULTIPLIERS,
        int(beam_size), float(persistence_weight), float(length_weight),
        likelihood == "gaussian", EMISSION_SAMPLES,
    )
    prediction = np.interp(
        record["md"][record["evaluation"]], record["md"][node_rows], path_node
    )
    return prediction, float(objective)


def _error_payload(record, prediction):
    index = record["evaluation"]
    truth = record["truth"][index]
    distance = record["md"][index] - record["anchor_md"]
    error = prediction - truth
    payload = {
        "well": record["well"],
        "rows": len(error),
        "sse": float(np.sum(error**2)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
    }
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        payload[f"{name}_rows"] = int(mask.sum())
        payload[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return payload


def _summary(rows):
    frame = pd.DataFrame(rows)
    count = int(frame["rows"].sum())
    sse = float(frame["sse"].sum())
    well_rmse = np.sqrt(frame["sse"] / frame["rows"])
    result = {
        "rows": count,
        "wells": len(frame),
        "exact_sse": sse,
        "rmse": float(np.sqrt(sse / count)),
        "mae": float(frame["absolute_error_sum"].sum() / count),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    }
    for name, _, _ in STRATA:
        local_count = int(frame[f"{name}_rows"].sum())
        local_sse = float(frame[f"{name}_sse"].sum())
        result[f"{name}_rmse"] = float(np.sqrt(local_sse / local_count)) if local_count else np.nan
    return result


def _evaluate(records, predictor):
    rows = []
    predictions = {}
    for record in records:
        prediction = predictor(record)
        rows.append(_error_payload(record, prediction))
        predictions[record["well"]] = prediction
    return _summary(rows), pd.DataFrame(rows), predictions


if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
folds = pd.read_csv(FOLD_PATH, dtype={"well": str}).drop_duplicates("well")
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
prior_wells = sorted(folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"])
selection_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"])
valid_wells = sorted(folds.loc[folds["fold"].eq(VALID_FOLD), "well"])
if (len(prior_wells), len(selection_wells), len(valid_wells)) != (309, 154, 155):
    raise RuntimeError(
        f"Unexpected nested folds: {len(prior_wells)}/{len(selection_wells)}/{len(valid_wells)}"
    )

started = time.time()
prior_records = []
for number, well in enumerate(prior_wells, 1):
    prior_records.append(_load_record(well))
    if number % 50 == 0 or number == len(prior_wells):
        print(f"V2 fit priors {number}/{len(prior_wells)}", flush=True)
priors = _fit_outer_priors(prior_records)
del prior_records
selection_records = [_load_record(well) for well in selection_wells]
valid_records = [_load_record(well) for well in valid_wells]

# Representation audit: continuous interpolation versus quantised reachable paths.
continuous_metrics, _, _ = _evaluate(selection_records, _continuous_oracle)
lattice_rows = []
for lattice in LATTICE_CONFIGS:
    name, length_nodes, beam_size, rate_scale = lattice
    metrics, _, _ = _evaluate(
        selection_records,
        lambda record, lattice=lattice: _quantized_oracle(
            record, priors, lattice[1], lattice[2], lattice[3]
        ),
    )
    lattice_rows.append(
        {
            "config": name,
            "beam_size": beam_size,
            "rate_scale": rate_scale,
            "length_nodes": ",".join(map(str, length_nodes)),
            **metrics,
        }
    )
    print("LATTICE ORACLE", name, metrics["rmse"], flush=True)
lattice_scan = pd.DataFrame(lattice_rows).sort_values("rmse").reset_index(drop=True)
lattice_scan.to_csv(OUT_DIR / "fold3_lattice_oracle_scan.csv", index=False)
selected_lattice_name = str(lattice_scan.iloc[0]["config"])
selected_lattice = next(item for item in LATTICE_CONFIGS if item[0] == selected_lattice_name)

# Stage A: select the observation model with priors disabled.
observation_rows = []
for calibration_mode in CALIBRATION_MODES:
    for likelihood in LIKELIHOODS:
        metrics, _, _ = _evaluate(
            selection_records,
            lambda record, cm=calibration_mode, lk=likelihood: _decode(
                record, priors, selected_lattice, cm, lk, 0.0, 0.0
            )[0],
        )
        observation_rows.append(
            {"calibration": calibration_mode, "likelihood": likelihood, **metrics}
        )
        print("OBSERVATION", calibration_mode, likelihood, metrics["rmse"], flush=True)
observation_scan = pd.DataFrame(observation_rows).sort_values("rmse").reset_index(drop=True)
observation_scan.to_csv(OUT_DIR / "fold3_observation_scan.csv", index=False)
selected_calibration = str(observation_scan.iloc[0]["calibration"])
selected_likelihood = str(observation_scan.iloc[0]["likelihood"])

# Stage B: select persistence and length-prior doses under the frozen observation model.
prior_rows = []
for persistence_weight in PERSISTENCE_WEIGHTS:
    for length_weight in LENGTH_WEIGHTS:
        metrics, _, _ = _evaluate(
            selection_records,
            lambda record, pw=persistence_weight, lw=length_weight: _decode(
                record, priors, selected_lattice, selected_calibration,
                selected_likelihood, pw, lw
            )[0],
        )
        prior_rows.append(
            {"persistence_weight": persistence_weight, "length_weight": length_weight, **metrics}
        )
        print("PRIOR DOSE", persistence_weight, length_weight, metrics["rmse"], flush=True)
prior_scan = pd.DataFrame(prior_rows).sort_values("rmse").reset_index(drop=True)
prior_scan.to_csv(OUT_DIR / "fold3_prior_dose_scan.csv", index=False)
selected_persistence = float(prior_scan.iloc[0]["persistence_weight"])
selected_length = float(prior_scan.iloc[0]["length_weight"])

recipe = {
    "lattice": selected_lattice_name,
    "length_nodes": selected_lattice[1],
    "beam_size": selected_lattice[2],
    "rate_scale": selected_lattice[3],
    "calibration": selected_calibration,
    "likelihood": selected_likelihood,
    "persistence_weight": selected_persistence,
    "length_weight": selected_length,
}

# One frozen Fold-2 application plus controls.
fold2_metrics, fold2_bywell, fold2_predictions = _evaluate(
    valid_records,
    lambda record: _decode(
        record, priors, selected_lattice, selected_calibration,
        selected_likelihood, selected_persistence, selected_length
    )[0],
)
fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

quantized_fold2, _, _ = _evaluate(
    valid_records,
    lambda record: _quantized_oracle(
        record, priors, selected_lattice[1], selected_lattice[2], selected_lattice[3]
    ),
)

donor_map = {
    record["well"]: valid_records[(index + 37) % len(valid_records)]
    for index, record in enumerate(valid_records)
}
shuffled_metrics, _, _ = _evaluate(
    valid_records,
    lambda record: _decode(
        record, priors, selected_lattice, selected_calibration,
        selected_likelihood, selected_persistence, selected_length,
        donor=donor_map[record["well"]],
    )[0],
)

row_frames = []
for record in valid_records:
    index = record["evaluation"]
    row_frames.append(
        pd.DataFrame(
            {
                "well": record["well"],
                "row_idx": index,
                "truth": record["truth"][index],
                "prediction": fold2_predictions[record["well"]],
                "distance_ft": record["md"][index] - record["anchor_md"],
            }
        )
    )
fold2_rows = pd.concat(row_frames, ignore_index=True)
fold2_rows.to_csv(OUT_DIR / "fold2_rows.csv.gz", index=False, compression="gzip")

strong_context = None
row_correlation = np.nan
blend_best = None
if POOL_PATH is not None:
    header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
    if "strong" in header:
        usecols = [column for column in ("id", "well", "row_idx", "_row_idx", "strong") if column in header]
        pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": str, "well": str})
        if "well" not in pool:
            pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
        if "row_idx" not in pool:
            pool["row_idx"] = (
                pd.to_numeric(pool["_row_idx"], errors="raise").astype(int)
                if "_row_idx" in pool
                else pool["id"].str.rsplit("_", n=1).str[-1].astype(int)
            )
        merged = fold2_rows.merge(
            pool[["well", "row_idx", "strong"]],
            on=["well", "row_idx"], how="left", validate="one_to_one",
        )
        if merged["strong"].notna().all():
            error = merged["prediction"].to_numpy(float) - merged["truth"].to_numpy(float)
            strong_error = merged["strong"].to_numpy(float) - merged["truth"].to_numpy(float)
            row_correlation = float(np.corrcoef(error, strong_error)[0, 1])
            strong_rmse = float(np.sqrt(np.mean(strong_error**2)))
            strong_context = {"rmse": strong_rmse, "rows": len(merged), "wells": merged["well"].nunique()}
            blend_rows = []
            for weight in np.linspace(0.0, 0.50, 21):
                prediction = (1.0 - weight) * merged["strong"].to_numpy(float) + weight * merged["prediction"].to_numpy(float)
                blend_rows.append(
                    {"stride_weight": float(weight), "rmse": float(np.sqrt(np.mean((prediction - merged["truth"])**2)))}
                )
            blend_scan = pd.DataFrame(blend_rows)
            blend_scan.to_csv(OUT_DIR / "fold2_strong_blend_diagnostic.csv", index=False)
            blend_best = blend_scan.sort_values("rmse").iloc[0].to_dict()

representation_pass = bool(
    continuous_metrics["rmse"] <= 0.50
    and lattice_scan.iloc[0]["rmse"] <= 2.0
    and quantized_fold2["rmse"] <= 2.5
)
path_pass = bool(fold2_metrics["rmse"] <= 12.0 and fold2_metrics["far_gt2000_rmse"] <= 14.0)
causal_pass = bool(shuffled_metrics["rmse"] >= fold2_metrics["rmse"] + 1.0)
independence_pass = bool(np.isfinite(row_correlation) and row_correlation <= 0.50)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_STRIDE_lattice_observability_and_recipe_selection",
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
    },
    "outer_train_priors": priors,
    "fold3_representation": {
        "continuous_oracle": continuous_metrics,
        "best_quantized_lattice": lattice_scan.iloc[0].to_dict(),
    },
    "selected_recipe": recipe,
    "fold2_quantized_oracle": quantized_fold2,
    "fold2": fold2_metrics,
    "fold2_shuffled_typewell": shuffled_metrics,
    "strong_context": strong_context,
    "row_error_correlation_with_strong": row_correlation,
    "best_same_fold_blend": blend_best,
    "gates": {
        "continuous_oracle_rmse_le": 0.50,
        "fold3_quantized_oracle_rmse_le": 2.0,
        "fold2_quantized_oracle_rmse_le": 2.5,
        "path_rmse_le": 12.0,
        "far_gt2000_rmse_le": 14.0,
        "shuffled_typewell_degradation_ge": 1.0,
        "row_error_correlation_le": 0.50,
    },
    "representation_pass": representation_pass,
    "path_pass": path_pass,
    "causal_pass": causal_pass,
    "independence_pass": independence_pass,
    "strict_pass": bool(representation_pass and path_pass and causal_pass and independence_pass),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> generate all-fold OOF with the frozen nested recipe"
        if representation_pass and path_pass and causal_pass and independence_pass
        else "FAIL -> use oracle/recipe tables to identify lattice, likelihood, or prior failure"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nLATTICE ORACLE SCAN")
print(lattice_scan.to_string(index=False), flush=True)
print("\nOBSERVATION SCAN")
print(observation_scan.to_string(index=False), flush=True)
print("\nPRIOR DOSE SCAN")
print(prior_scan.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
