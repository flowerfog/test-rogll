"""STRIDE v4: dynamic drilled-trajectory centering in structure space.

For a segment, structure S = TVT + Z, hence

    dS/dMD = dTVT/dMD + dZ/dMD.

The v3 decoder persisted raw TVT slope and therefore penalised bends caused
only by the known drilled trajectory. This version persists structure slope,
uses the visible-prefix S trend for the first segment, and keeps the absolute
TVT-slope proposal lattice from v3.
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
V3_CANDIDATES = (
    ROOT / "stride_absolute_slope_nested_v3_cell.py",
    Path.cwd() / "stride_absolute_slope_nested_v3_cell.py",
)
V3_SOURCE = next((path for path in V3_CANDIDATES if path.exists()), None)
if V3_SOURCE is None:
    raise FileNotFoundError(
        "Upload STRIDE v1, v2, and v3 source files beside this v4 cell"
    )
source = V3_SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(V3_SOURCE))
cutoff = next(
    (
        index for index, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and source.splitlines()[node.lineno - 1].strip().startswith("if FOLD_PATH is None:")
    ),
    None,
)
if cutoff is None:
    raise RuntimeError("Cannot locate the v3 definition boundary")
V = {"__name__": "stride_v3_definitions", "__file__": str(V3_SOURCE)}
exec(compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(V3_SOURCE), "exec"), V)

EXPERIMENT = "stride_dynamic_structure_trend_nested_v4"
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
SEGMENT_LENGTH_NODES = V["SEGMENT_LENGTH_NODES"].astype(np.int64)
ABSOLUTE_SLOPES = np.linspace(-0.16, 0.16, 17).astype(np.float64)
BEAM_SIZE = 32
EMISSION_SAMPLES = 24
PERSISTENCE_WEIGHTS = (0.0, 0.10, 0.30, 1.0, 3.0, 10.0)
TREND_WEIGHTS = (0.0, 0.10, 0.30, 1.0, 3.0)
LENGTH_WEIGHTS = (0.0, 0.01, 0.05, 0.20)
LIKELIHOOD = "cauchy"
REFERENCE = "shift"
STRATA = V["STRATA"]

_load_record = V["_load_record"]
_node_rows = V["_node_rows"]
_reference = V["_reference"]
_segment_emission = V["_segment_emission"]
_absolute_oracle_decode = V["_absolute_oracle_decode"]
_oracle = V["_oracle"]


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
        return 0.01
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return float(max(1.4826 * mad, 0.25 * np.std(values), 0.001))


def _prefix_structure_rate(record):
    known = record["known"]
    anchor = record["anchor"]
    tail = known[record["md"][known] >= record["md"][anchor] - 500.0]
    structure = record["tvt_input"] + record["z"]
    if len(tail) >= 8 and np.ptp(record["md"][tail]) >= 50.0:
        return float(np.polyfit(record["md"][tail], structure[tail], 1)[0])
    return 0.0


def _fit_structure_priors(records):
    rates = []
    changes = []
    prefix_rates = []
    for record in records:
        index = np.r_[record["anchor"], record["evaluation"]]
        md_native = record["md"][index]
        structure_native = np.r_[
            record["anchor_tvt"] + record["z"][record["anchor"]],
            record["truth"][record["evaluation"]] + record["z"][record["evaluation"]],
        ]
        grid = np.arange(md_native[0], md_native[-1] + 0.5 * NODE_STEP_FT, NODE_STEP_FT)
        if grid[-1] < md_native[-1] - 1e-6:
            grid = np.r_[grid, md_native[-1]]
        structure = np.interp(grid, md_native, structure_native)
        slope = np.diff(structure) / np.maximum(np.diff(grid), 1e-6)
        slope = pd.Series(slope).rolling(3, center=True, min_periods=1).median().to_numpy()
        rates.append(slope)
        lag = max(int(round(150.0 / NODE_STEP_FT)), 1)
        if len(slope) > lag:
            changes.append(slope[lag:] - slope[:-lag])
        prefix_rates.append(_prefix_structure_rate(record))
    all_rates = np.concatenate(rates)
    all_changes = np.concatenate(changes)
    rate_low, rate_high = np.quantile(all_rates, [0.005, 0.995])
    return {
        "structure_rate_change_scale": _robust_scale(all_changes),
        "structure_rate_low": float(rate_low),
        "structure_rate_high": float(rate_high),
        "structure_rate_median": float(np.median(all_rates)),
        "prefix_structure_rate_median": float(np.median(prefix_rates)),
        "rate_samples": int(len(all_rates)),
        "change_samples": int(len(all_changes)),
        "wells": int(len(records)),
    }


@njit(cache=True)
def _dynamic_structure_decode(md, z, gr, node_rows, anchor_tvt,
                              prefix_structure_rate, tw_tvt, tw_gr, gr_scale,
                              absolute_tvt_slopes, length_nodes,
                              structure_change_scale, length_mu, length_sigma,
                              beam_size, persistence_weight, trend_weight,
                              length_weight, gaussian_mode, maximum_samples,
                              dynamic_geometry):
    nodes = len(node_rows)
    cost = np.full((nodes, beam_size), 1e30)
    state_tvt = np.zeros((nodes, beam_size))
    state_structure_rate = np.zeros((nodes, beam_size))
    parent_node = np.full((nodes, beam_size), -1, np.int64)
    parent_slot = np.full((nodes, beam_size), -1, np.int64)
    cost[0, 0] = 0.0
    state_tvt[0, 0] = anchor_tvt
    state_structure_rate[0, 0] = prefix_structure_rate

    for target_node in range(1, nodes):
        for length_position in range(len(length_nodes)):
            source_node = target_node - length_nodes[length_position]
            if source_node < 0:
                continue
            source_row = node_rows[source_node]
            target_row = node_rows[target_node]
            length_ft = md[target_row] - md[source_row]
            if length_ft <= 0:
                continue
            z_rate = (z[target_row] - z[source_row]) / length_ft if dynamic_geometry else 0.0
            length_z = (np.log(max(length_ft, 1.0)) - length_mu) / max(length_sigma, 1e-6)
            length_cost = length_weight * (0.5 * length_z * length_z + np.log(max(length_ft, 1.0)))
            for source_slot in range(beam_size):
                if cost[source_node, source_slot] >= 1e29:
                    continue
                old_tvt = state_tvt[source_node, source_slot]
                previous_structure_rate = state_structure_rate[source_node, source_slot]
                for slope_position in range(len(absolute_tvt_slopes)):
                    tvt_slope = absolute_tvt_slopes[slope_position]
                    structure_rate = tvt_slope + z_rate
                    new_tvt = old_tvt + tvt_slope * length_ft
                    emission = _segment_emission(
                        md, gr, source_row, target_row, old_tvt, new_tvt,
                        tw_tvt, tw_gr, gr_scale, gaussian_mode, maximum_samples,
                    )
                    difference = (structure_rate - previous_structure_rate) / max(structure_change_scale, 1e-6)
                    weight = trend_weight if source_node == 0 else persistence_weight
                    transition = weight * 0.5 * difference * difference
                    new_cost = cost[source_node, source_slot] + emission + transition + length_cost
                    worst = 0
                    for slot in range(1, beam_size):
                        if cost[target_node, slot] > cost[target_node, worst]:
                            worst = slot
                    if new_cost < cost[target_node, worst]:
                        cost[target_node, worst] = new_cost
                        state_tvt[target_node, worst] = new_tvt
                        state_structure_rate[target_node, worst] = structure_rate
                        parent_node[target_node, worst] = source_node
                        parent_slot[target_node, worst] = source_slot

    final_slot = 0
    for slot in range(1, beam_size):
        if cost[nodes - 1, slot] < cost[nodes - 1, final_slot]:
            final_slot = slot
    path = np.full(nodes, np.nan)
    node = nodes - 1
    slot = final_slot
    while node >= 0 and slot >= 0:
        path[node] = state_tvt[node, slot]
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
            path[fill_node] = left_tvt + fraction * (right_tvt - left_tvt)
        node = previous_node
        slot = previous_slot
    path[0] = anchor_tvt
    return path


def _decode(record, priors, persistence_weight, trend_weight, length_weight,
            reference=REFERENCE, donor=None, dynamic_geometry=True):
    tw_tvt, tw_gr, gr_scale = _reference(record, reference, donor=donor)
    node_rows = _node_rows(record)
    prefix_rate = (
        _prefix_structure_rate(record)
        if dynamic_geometry else float(record["initial_rate"])
    )
    path = _dynamic_structure_decode(
        record["md"], record["z"], record["gr"], node_rows,
        record["anchor_tvt"], prefix_rate, tw_tvt, tw_gr, gr_scale,
        ABSOLUTE_SLOPES, SEGMENT_LENGTH_NODES,
        priors["structure_rate_change_scale"],
        priors["length_log_mu"], priors["length_log_sigma"],
        BEAM_SIZE, float(persistence_weight), float(trend_weight),
        float(length_weight), LIKELIHOOD == "gaussian", EMISSION_SAMPLES,
        bool(dynamic_geometry),
    )
    return np.interp(
        record["md"][record["evaluation"]], record["md"][node_rows], path
    )


def _payload(record, prediction):
    index = record["evaluation"]
    truth = record["truth"][index]
    distance = record["md"][index] - record["anchor_md"]
    error = prediction - truth
    result = {
        "well": record["well"],
        "rows": len(error),
        "sse": float(np.sum(error**2)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
    }
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        result[f"{name}_rows"] = int(mask.sum())
        result[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return result


def _summarise(rows):
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
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    }
    for name, _, _ in STRATA:
        local_count = int(frame[f"{name}_rows"].sum())
        local_sse = float(frame[f"{name}_sse"].sum())
        result[f"{name}_rmse"] = float(np.sqrt(local_sse / local_count)) if local_count else np.nan
    return result, frame


def _evaluate(records, predictor, keep_predictions=False):
    rows = []
    predictions = {}
    for record in records:
        prediction = predictor(record)
        rows.append(_payload(record, prediction))
        if keep_predictions:
            predictions[record["well"]] = prediction
    metrics, frame = _summarise(rows)
    return metrics, frame, predictions


if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
folds = pd.read_csv(FOLD_PATH, dtype={"well": str}).drop_duplicates("well")
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
prior_wells = sorted(folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"])
selection_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"])
valid_wells = sorted(folds.loc[folds["fold"].eq(VALID_FOLD), "well"])
if (len(prior_wells), len(selection_wells), len(valid_wells)) != (309, 154, 155):
    raise RuntimeError("Unexpected nested fold sizes")

started = time.time()
prior_records = []
for number, well in enumerate(prior_wells, 1):
    prior_records.append(_load_record(well))
    if number % 50 == 0 or number == len(prior_wells):
        print(f"DYNAMIC-STRUCTURE fit priors {number}/{len(prior_wells)}", flush=True)
structure_priors = _fit_structure_priors(prior_records)
# Reuse only the length distribution from the older outer-train fit.
old_priors = V["_fit_outer_priors"](prior_records)
structure_priors.update(
    length_log_mu=old_priors["length_log_mu"],
    length_log_sigma=old_priors["length_log_sigma"],
)
del prior_records
selection_records = [_load_record(well) for well in selection_wells]
valid_records = [_load_record(well) for well in valid_wells]

transition_rows = []
for persistence_weight in PERSISTENCE_WEIGHTS:
    for trend_weight in TREND_WEIGHTS:
        metrics, _, _ = _evaluate(
            selection_records,
            lambda record, pw=persistence_weight, tw=trend_weight: _decode(
                record, structure_priors, pw, tw, 0.0
            ),
        )
        transition_rows.append(
            {"persistence_weight": persistence_weight, "trend_weight": trend_weight, **metrics}
        )
        print("DYNAMIC-STRUCTURE TRANSITION", persistence_weight, trend_weight, metrics["rmse"], flush=True)
transition_scan = pd.DataFrame(transition_rows).sort_values("rmse").reset_index(drop=True)
transition_scan.to_csv(OUT_DIR / "fold3_transition_scan.csv", index=False)
selected_persistence = float(transition_scan.iloc[0]["persistence_weight"])
selected_trend = float(transition_scan.iloc[0]["trend_weight"])

length_rows = []
for length_weight in LENGTH_WEIGHTS:
    metrics, _, _ = _evaluate(
        selection_records,
        lambda record, lw=length_weight: _decode(
            record, structure_priors, selected_persistence, selected_trend, lw
        ),
    )
    length_rows.append({"length_weight": length_weight, **metrics})
    print("DYNAMIC-STRUCTURE LENGTH", length_weight, metrics["rmse"], flush=True)
length_scan = pd.DataFrame(length_rows).sort_values("rmse").reset_index(drop=True)
length_scan.to_csv(OUT_DIR / "fold3_length_scan.csv", index=False)
selected_length = float(length_scan.iloc[0]["length_weight"])

recipe = {
    "absolute_tvt_slopes": ABSOLUTE_SLOPES,
    "segment_lengths_ft": SEGMENT_LENGTH_NODES * NODE_STEP_FT,
    "beam_size": BEAM_SIZE,
    "likelihood": LIKELIHOOD,
    "reference": REFERENCE,
    "persistence_space": "structure_rate = TVT_rate + Z_rate",
    "persistence_weight": selected_persistence,
    "trend_weight": selected_trend,
    "length_weight": selected_length,
}

fold2_metrics, fold2_bywell, fold2_predictions = _evaluate(
    valid_records,
    lambda record: _decode(
        record, structure_priors, selected_persistence, selected_trend,
        selected_length
    ),
    keep_predictions=True,
)
fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

donor_map = {
    record["well"]: valid_records[(index + 37) % len(valid_records)]
    for index, record in enumerate(valid_records)
}
ablation_specs = {
    "full": dict(pw=selected_persistence, tw=selected_trend, lw=selected_length, ref="shift", dynamic=True),
    "no_persistence": dict(pw=0.0, tw=selected_trend, lw=selected_length, ref="shift", dynamic=True),
    "no_drilled_trend": dict(pw=selected_persistence, tw=0.0, lw=selected_length, ref="shift", dynamic=True),
    "raw_tvt_persistence": dict(pw=selected_persistence, tw=selected_trend, lw=selected_length, ref="shift", dynamic=False),
    "no_reference": dict(pw=selected_persistence, tw=selected_trend, lw=selected_length, ref="raw", dynamic=True),
}
ablation_rows = []
for name, config in ablation_specs.items():
    metrics, _, _ = _evaluate(
        valid_records,
        lambda record, cfg=config: _decode(
            record, structure_priors, cfg["pw"], cfg["tw"], cfg["lw"],
            reference=cfg["ref"], dynamic_geometry=cfg["dynamic"]
        ),
    )
    ablation_rows.append({"mode": name, **metrics})
shuffled_metrics, _, _ = _evaluate(
    valid_records,
    lambda record: _decode(
        record, structure_priors, selected_persistence, selected_trend,
        selected_length, reference="shift", donor=donor_map[record["well"]]
    ),
)
ablation_rows.append({"mode": "shuffled_typewell", **shuffled_metrics})
ablations = pd.DataFrame(ablation_rows)
for column in ("rmse", "far_gt2000_rmse"):
    ablations[f"degradation_{column}_vs_full"] = ablations[column] - fold2_metrics[column]
ablations.to_csv(OUT_DIR / "fold2_ablations.csv", index=False)

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
correlation = np.nan
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
                if "_row_idx" in pool else pool["id"].str.rsplit("_", n=1).str[-1].astype(int)
            )
        merged = fold2_rows.merge(
            pool[["well", "row_idx", "strong"]], on=["well", "row_idx"],
            how="left", validate="one_to_one",
        )
        if merged["strong"].notna().all():
            error = merged["prediction"].to_numpy(float) - merged["truth"].to_numpy(float)
            strong_error = merged["strong"].to_numpy(float) - merged["truth"].to_numpy(float)
            correlation = float(np.corrcoef(error, strong_error)[0, 1])
            strong_context = {"rmse": float(np.sqrt(np.mean(strong_error**2))), "rows": len(merged)}
            blend_rows = []
            for weight in np.linspace(0.0, 0.50, 21):
                prediction = (1.0 - weight) * merged["strong"].to_numpy(float) + weight * merged["prediction"].to_numpy(float)
                blend_rows.append({"stride_weight": float(weight), "rmse": float(np.sqrt(np.mean((prediction - merged["truth"])**2)))})
            blend_scan = pd.DataFrame(blend_rows)
            blend_scan.to_csv(OUT_DIR / "fold2_strong_blend_diagnostic.csv", index=False)
            blend_best = blend_scan.sort_values("rmse").iloc[0].to_dict()

full_row = ablations.loc[ablations["mode"].eq("full")].iloc[0]
no_persistence = ablations.loc[ablations["mode"].eq("no_persistence")].iloc[0]
no_trend = ablations.loc[ablations["mode"].eq("no_drilled_trend")].iloc[0]
raw_tvt = ablations.loc[ablations["mode"].eq("raw_tvt_persistence")].iloc[0]
path_pass = bool(fold2_metrics["rmse"] <= 11.0 and fold2_metrics["far_gt2000_rmse"] <= 13.0)
geometry_pass = bool(raw_tvt["rmse"] >= full_row["rmse"] + 0.30)
component_pass = bool(
    no_persistence["rmse"] >= full_row["rmse"] + 0.50
    and no_trend["rmse"] >= full_row["rmse"] + 0.20
)
independence_pass = bool(np.isfinite(correlation) and correlation <= 0.50)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_dynamic_structure_trend_STRIDE_gateway",
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
    },
    "structure_priors": structure_priors,
    "selected_recipe": recipe,
    "fold2": fold2_metrics,
    "fold2_ablations": ablations.to_dict(orient="records"),
    "strong_context": strong_context,
    "row_error_correlation_with_strong": correlation,
    "best_same_fold_blend": blend_best,
    "gates": {
        "path_rmse_le": 11.0,
        "far_gt2000_rmse_le": 13.0,
        "dynamic_geometry_gain_ge": 0.30,
        "no_persistence_degradation_ge": 0.50,
        "no_trend_degradation_ge": 0.20,
        "error_correlation_le": 0.50,
    },
    "path_pass": path_pass,
    "geometry_pass": geometry_pass,
    "component_pass": component_pass,
    "independence_pass": independence_pass,
    "strict_pass": bool(path_pass and geometry_pass and component_pass and independence_pass),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> build consensus and all-fold OOF"
        if path_pass and geometry_pass and component_pass and independence_pass
        else "FAIL -> dynamic drilled geometry does not recover the missing backbone gain"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSTRUCTURE PRIORS")
print(json.dumps(structure_priors, indent=2, default=_json_default), flush=True)
print("\nDYNAMIC TRANSITION SCAN")
print(transition_scan.to_string(index=False), flush=True)
print("\nLENGTH SCAN")
print(length_scan.to_string(index=False), flush=True)
print("\nFOLD2 ABLATIONS")
print(ablations.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
