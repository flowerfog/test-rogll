"""STRIDE v5: diversity-preserving complete-well state-lattice gateway.

The earlier STRIDE approximations retained the globally cheapest paths at each
MD node. That can collapse every survivor into one alias basin even when later
GR evidence would have rescued a temporarily worse path. This version keeps a
small, depth-diverse set of paths for every absolute TVT-slope state.

Protocol:
  * Folds 4/5 fit all priors and the global Cauchy scale.
  * Fold 3 selects state diversity and transition weights.
  * Fold 2 is evaluated once with the frozen recipe.
  * Fold 1 is never read.

Upload ``stride_joint_segment_decoder_gateway_v1_cell.py`` beside this file.
The v1 definitions are loaded without executing its experiment tail.
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
V1_CANDIDATES = (
    ROOT / "stride_joint_segment_decoder_gateway_v1_cell.py",
    Path.cwd() / "stride_joint_segment_decoder_gateway_v1_cell.py",
)
V1_SOURCE = next((path for path in V1_CANDIDATES if path.exists()), None)
if V1_SOURCE is None:
    raise FileNotFoundError(
        "Upload stride_joint_segment_decoder_gateway_v1_cell.py beside this cell"
    )

source = V1_SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(V1_SOURCE))
cutoff = next(
    (
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and source.splitlines()[node.lineno - 1].strip().startswith(
            "if FOLD_PATH is None:"
        )
    ),
    None,
)
if cutoff is None:
    raise RuntimeError("Cannot locate the v1 definition boundary")
V = {"__name__": "stride_v1_definitions", "__file__": str(V1_SOURCE)}
exec(
    compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(V1_SOURCE), "exec"),
    V,
)


EXPERIMENT = "stride_diverse_state_lattice_nested_v5"
DATA_ROOT = V["DATA_ROOT"]
OOF_ROOT = V["OOF_ROOT"]
FOLD_PATH = V["FOLD_PATH"]
POOL_PATH = V["POOL_PATH"]
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
VALID_FOLD = 2
NODE_STEP_FT = 32.0

# The six non-myopic lengths visible in the note are approximately
# 70/110/160/230/330/500 ft. The nearest 32-ft graph lengths are used here.
SEGMENT_LENGTH_NODES = np.asarray((2, 3, 5, 7, 10, 16), np.int64)
ABSOLUTE_SLOPES = np.linspace(-0.20, 0.20, 17).astype(np.float64)
EMISSION_SAMPLES = 24
DEPTH_SEPARATION_FT = 4.0
FAST_SELECTION_WELLS = 24
DIVERSITY_SLOTS = (1, 2, 4)

# A compact screen is intentional. The best two recipes are re-evaluated on
# every Fold-3 well before Fold 2 is touched.
TRANSITION_RECIPES = (
    (0.10, 0.10, 0.00),
    (0.30, 0.10, 0.00),
    (0.30, 0.30, 0.00),
    (1.00, 0.30, 0.00),
    (1.00, 1.00, 0.00),
    (3.00, 1.00, 0.00),
)

STRATA = V["STRATA"]
_load_record = V["_load_record"]
_fit_outer_priors = V["_fit_outer_priors"]
_node_rows = V["_node_rows"]
_robust_location_scale = V["_robust_location_scale"]


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


def _prefix_reference_residual(record, tw_tvt=None, tw_gr=None, apply_shift=True):
    if tw_tvt is None:
        tw_tvt = record["tw_tvt"]
        tw_gr = record["tw_gr_raw"]
    expected = np.interp(
        record["tvt_input"], tw_tvt, tw_gr, left=np.nan, right=np.nan
    )
    valid = (
        np.isfinite(record["tvt_input"])
        & np.isfinite(record["gr"])
        & np.isfinite(expected)
    )
    observed = record["gr"][valid]
    expected = expected[valid]
    shift = float(np.median(observed - expected)) if apply_shift and len(observed) else 0.0
    return observed - (expected + shift), shift, int(valid.sum())


def _fit_global_cauchy_scale(records):
    residuals = []
    for record in records:
        residual, _, _ = _prefix_reference_residual(record, apply_shift=True)
        if len(residual) > 4096:
            positions = np.linspace(0, len(residual) - 1, 4096).astype(int)
            residual = residual[positions]
        residuals.append(residual)
    pooled = np.concatenate(residuals)
    centre, scale = _robust_location_scale(pooled)
    return {
        "centre": float(centre),
        "scale": float(np.clip(scale, 6.0, 60.0)),
        "rows": int(len(pooled)),
        "wells": int(len(records)),
    }


def _reference(record, global_scale, mode="shift", donor=None):
    if donor is None:
        tw_tvt = record["tw_tvt"]
        tw_gr = record["tw_gr_raw"]
    else:
        tw_tvt = donor["tw_tvt"]
        tw_gr = donor["tw_gr_raw"]
    apply_shift = mode != "raw"
    residual, shift, rows = _prefix_reference_residual(
        record, tw_tvt=tw_tvt, tw_gr=tw_gr, apply_shift=apply_shift
    )
    reference = tw_gr + shift
    if mode == "perwell_scale":
        _, scale = _robust_location_scale(residual)
        scale = float(np.clip(scale, 6.0, 60.0))
    else:
        scale = float(global_scale)
    return tw_tvt.astype(np.float64), reference.astype(np.float64), scale, rows


@njit(cache=False)
def _interp_sorted_local(axis, values, query):
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


@njit(cache=False)
def _segment_emission_local(
    md,
    gr,
    row0,
    row1,
    tvt0,
    tvt1,
    tw_tvt,
    tw_gr,
    gr_scale,
    maximum_samples,
):
    count = row1 - row0
    if count <= 0:
        return 0.0
    samples = min(count, maximum_samples)
    total = 0.0
    for sample in range(samples):
        row = row0 + 1 + ((sample + 1) * count - 1) // samples
        fraction = (md[row] - md[row0]) / max(md[row1] - md[row0], 1e-9)
        candidate_tvt = tvt0 + fraction * (tvt1 - tvt0)
        expected = _interp_sorted_local(tw_tvt, tw_gr, candidate_tvt)
        standardised = (gr[row] - expected) / max(gr_scale, 1e-6)
        total += np.log1p(min(standardised * standardised, 400.0))
    return total * count / samples


@njit(cache=False)
def _insert_diverse_state(
    target_node,
    target_slope,
    new_cost,
    new_tvt,
    source_node,
    source_slope,
    source_slot,
    depth_separation,
    cost,
    state_tvt,
    parent_node,
    parent_slope,
    parent_slot,
):
    slots = cost.shape[2]
    for slot in range(slots):
        if cost[target_node, target_slope, slot] >= 1e29:
            continue
        if abs(state_tvt[target_node, target_slope, slot] - new_tvt) <= depth_separation:
            if new_cost < cost[target_node, target_slope, slot]:
                cost[target_node, target_slope, slot] = new_cost
                state_tvt[target_node, target_slope, slot] = new_tvt
                parent_node[target_node, target_slope, slot] = source_node
                parent_slope[target_node, target_slope, slot] = source_slope
                parent_slot[target_node, target_slope, slot] = source_slot
            return

    worst = 0
    for slot in range(1, slots):
        if cost[target_node, target_slope, slot] > cost[target_node, target_slope, worst]:
            worst = slot
    if new_cost < cost[target_node, target_slope, worst]:
        cost[target_node, target_slope, worst] = new_cost
        state_tvt[target_node, target_slope, worst] = new_tvt
        parent_node[target_node, target_slope, worst] = source_node
        parent_slope[target_node, target_slope, worst] = source_slope
        parent_slot[target_node, target_slope, worst] = source_slot


@njit(cache=False)
def _diverse_joint_decode(
    md,
    gr,
    node_rows,
    anchor_tvt,
    drilled_trend,
    tw_tvt,
    tw_gr,
    gr_scale,
    absolute_slopes,
    length_nodes,
    length_mu,
    length_sigma,
    change_scale,
    diversity_slots,
    persistence_weight,
    trend_weight,
    length_weight,
    maximum_samples,
    depth_separation,
):
    nodes = len(node_rows)
    slopes = len(absolute_slopes)
    cost = np.full((nodes, slopes, diversity_slots), 1e30)
    state_tvt = np.zeros((nodes, slopes, diversity_slots))
    parent_node = np.full((nodes, slopes, diversity_slots), -1, np.int32)
    parent_slope = np.full((nodes, slopes, diversity_slots), -1, np.int16)
    parent_slot = np.full((nodes, slopes, diversity_slots), -1, np.int8)

    initial_slope = 0
    initial_distance = abs(absolute_slopes[0] - drilled_trend)
    for slope_position in range(1, slopes):
        distance = abs(absolute_slopes[slope_position] - drilled_trend)
        if distance < initial_distance:
            initial_slope = slope_position
            initial_distance = distance
    cost[0, initial_slope, 0] = 0.0
    state_tvt[0, initial_slope, 0] = anchor_tvt

    for target_node in range(1, nodes):
        for length_position in range(len(length_nodes)):
            source_node = target_node - length_nodes[length_position]
            if source_node < 0:
                continue
            row0 = node_rows[source_node]
            row1 = node_rows[target_node]
            length_ft = md[row1] - md[row0]
            if length_ft <= 0.0:
                continue
            length_z = (np.log(max(length_ft, 1.0)) - length_mu) / max(length_sigma, 1e-6)
            length_cost = length_weight * (
                0.5 * length_z * length_z + np.log(max(length_ft, 1.0))
            )
            for old_slope_position in range(slopes):
                old_slope = absolute_slopes[old_slope_position]
                for old_slot in range(diversity_slots):
                    old_cost = cost[source_node, old_slope_position, old_slot]
                    if old_cost >= 1e29:
                        continue
                    old_tvt = state_tvt[source_node, old_slope_position, old_slot]
                    for new_slope_position in range(slopes):
                        new_slope = absolute_slopes[new_slope_position]
                        new_tvt = old_tvt + new_slope * length_ft
                        if new_tvt < tw_tvt[0] or new_tvt > tw_tvt[-1]:
                            continue
                        emission = _segment_emission_local(
                            md,
                            gr,
                            row0,
                            row1,
                            old_tvt,
                            new_tvt,
                            tw_tvt,
                            tw_gr,
                            gr_scale,
                            maximum_samples,
                        )
                        if source_node == 0:
                            difference = (new_slope - drilled_trend) / max(change_scale, 1e-6)
                            transition = trend_weight * 0.5 * difference * difference
                        else:
                            difference = (new_slope - old_slope) / max(change_scale, 1e-6)
                            transition = persistence_weight * 0.5 * difference * difference
                        new_cost = old_cost + emission + transition + length_cost
                        _insert_diverse_state(
                            target_node,
                            new_slope_position,
                            new_cost,
                            new_tvt,
                            source_node,
                            old_slope_position,
                            old_slot,
                            depth_separation,
                            cost,
                            state_tvt,
                            parent_node,
                            parent_slope,
                            parent_slot,
                        )

    final_slope = 0
    final_slot = 0
    final_cost = cost[nodes - 1, 0, 0]
    for slope_position in range(slopes):
        for slot in range(diversity_slots):
            if cost[nodes - 1, slope_position, slot] < final_cost:
                final_cost = cost[nodes - 1, slope_position, slot]
                final_slope = slope_position
                final_slot = slot

    path = np.full(nodes, np.nan)
    node = nodes - 1
    slope_position = final_slope
    slot = final_slot
    while node >= 0 and slope_position >= 0 and slot >= 0:
        path[node] = state_tvt[node, slope_position, slot]
        previous_node = parent_node[node, slope_position, slot]
        previous_slope = parent_slope[node, slope_position, slot]
        previous_slot = parent_slot[node, slope_position, slot]
        if previous_node < 0:
            break
        left_tvt = state_tvt[previous_node, previous_slope, previous_slot]
        right_tvt = state_tvt[node, slope_position, slot]
        left_md = md[node_rows[previous_node]]
        right_md = md[node_rows[node]]
        for fill_node in range(previous_node + 1, node):
            fraction = (md[node_rows[fill_node]] - left_md) / max(right_md - left_md, 1e-9)
            path[fill_node] = left_tvt + fraction * (right_tvt - left_tvt)
        node = previous_node
        slope_position = previous_slope
        slot = previous_slot
    path[0] = anchor_tvt
    return path, final_cost


def _decode(
    record,
    priors,
    global_scale,
    diversity_slots,
    persistence_weight,
    trend_weight,
    length_weight,
    reference_mode="shift",
    donor=None,
):
    tw_tvt, tw_gr, gr_scale, _ = _reference(
        record, global_scale, mode=reference_mode, donor=donor
    )
    node_rows = _node_rows(record)
    path, objective = _diverse_joint_decode(
        record["md"],
        record["gr"],
        node_rows,
        record["anchor_tvt"],
        record["initial_rate"],
        tw_tvt,
        tw_gr,
        gr_scale,
        ABSOLUTE_SLOPES,
        SEGMENT_LENGTH_NODES,
        priors["length_log_mu"],
        priors["length_log_sigma"],
        priors["rate_change_scale"],
        int(diversity_slots),
        float(persistence_weight),
        float(trend_weight),
        float(length_weight),
        EMISSION_SAMPLES,
        DEPTH_SEPARATION_FT,
    )
    prediction = np.interp(
        record["md"][record["evaluation"]], record["md"][node_rows], path
    )
    return prediction, float(objective)


def _payload(record, prediction):
    index = record["evaluation"]
    truth = record["truth"][index]
    distance = record["md"][index] - record["anchor_md"]
    error = prediction - truth
    row = {
        "well": record["well"],
        "rows": int(len(error)),
        "exact_sse": float(np.sum(error**2)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
    }
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        row[f"{name}_rows"] = int(mask.sum())
        row[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return row


def _summarise(rows):
    frame = pd.DataFrame(rows)
    count = int(frame["rows"].sum())
    sse = float(frame["exact_sse"].sum())
    well_rmse = np.sqrt(frame["exact_sse"] / frame["rows"])
    metrics = {
        "rows": count,
        "wells": int(len(frame)),
        "exact_sse": sse,
        "rmse": float(np.sqrt(sse / count)),
        "mae": float(frame["absolute_error_sum"].sum() / count),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    }
    for name, _, _ in STRATA:
        local_rows = int(frame[f"{name}_rows"].sum())
        local_sse = float(frame[f"{name}_sse"].sum())
        metrics[f"{name}_rmse"] = (
            float(np.sqrt(local_sse / local_rows)) if local_rows else np.nan
        )
    return metrics, frame


def _evaluate(records, predictor, keep_predictions=False):
    rows = []
    predictions = {}
    for number, record in enumerate(records, 1):
        prediction = predictor(record)
        rows.append(_payload(record, prediction))
        if keep_predictions:
            predictions[record["well"]] = prediction
        if number % 50 == 0 or number == len(records):
            print(f"decode {number}/{len(records)}", flush=True)
    metrics, frame = _summarise(rows)
    return metrics, frame, predictions


def _even_subset(records, count):
    if len(records) <= count:
        return records
    positions = np.linspace(0, len(records) - 1, count).round().astype(int)
    return [records[position] for position in np.unique(positions)]


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
        f"Unexpected nested fold sizes: {len(prior_wells)}/{len(selection_wells)}/{len(valid_wells)}"
    )

started = time.time()
prior_records = []
for number, well in enumerate(prior_wells, 1):
    prior_records.append(_load_record(well))
    if number % 50 == 0 or number == len(prior_wells):
        print(f"V5 fit priors {number}/{len(prior_wells)}", flush=True)
priors = _fit_outer_priors(prior_records)
scale_fit = _fit_global_cauchy_scale(prior_records)
global_scale = float(scale_fit["scale"])
del prior_records

selection_records = [_load_record(well) for well in selection_wells]
fast_records = _even_subset(selection_records, FAST_SELECTION_WELLS)

diversity_rows = []
for slots in DIVERSITY_SLOTS:
    metrics, _, _ = _evaluate(
        fast_records,
        lambda record, slots=slots: _decode(
            record, priors, global_scale, slots, 0.30, 0.30, 0.0
        )[0],
    )
    diversity_rows.append({"diversity_slots_per_slope": slots, **metrics})
    print("V5 DIVERSITY", slots, metrics["rmse"], flush=True)
diversity_scan = pd.DataFrame(diversity_rows).sort_values("rmse").reset_index(drop=True)
diversity_scan.to_csv(OUT_DIR / "fold3_fast_diversity_scan.csv", index=False)
selected_slots = int(diversity_scan.iloc[0]["diversity_slots_per_slope"])

fast_recipe_rows = []
for persistence_weight, trend_weight, length_weight in TRANSITION_RECIPES:
    metrics, _, _ = _evaluate(
        fast_records,
        lambda record, pw=persistence_weight, tw=trend_weight, lw=length_weight: _decode(
            record, priors, global_scale, selected_slots, pw, tw, lw
        )[0],
    )
    fast_recipe_rows.append(
        {
            "persistence_weight": persistence_weight,
            "trend_weight": trend_weight,
            "length_weight": length_weight,
            **metrics,
        }
    )
    print(
        "V5 FAST RECIPE",
        persistence_weight,
        trend_weight,
        length_weight,
        metrics["rmse"],
        flush=True,
    )
fast_recipe_scan = pd.DataFrame(fast_recipe_rows).sort_values("rmse").reset_index(drop=True)
fast_recipe_scan.to_csv(OUT_DIR / "fold3_fast_recipe_scan.csv", index=False)

full_recipe_rows = []
for item in fast_recipe_scan.head(2).itertuples(index=False):
    pw = float(item.persistence_weight)
    tw = float(item.trend_weight)
    lw = float(item.length_weight)
    metrics, _, _ = _evaluate(
        selection_records,
        lambda record, pw=pw, tw=tw, lw=lw: _decode(
            record, priors, global_scale, selected_slots, pw, tw, lw
        )[0],
    )
    full_recipe_rows.append(
        {
            "persistence_weight": pw,
            "trend_weight": tw,
            "length_weight": lw,
            **metrics,
        }
    )
    print("V5 FULL RECIPE", pw, tw, lw, metrics["rmse"], flush=True)
full_recipe_scan = pd.DataFrame(full_recipe_rows).sort_values("rmse").reset_index(drop=True)
full_recipe_scan.to_csv(OUT_DIR / "fold3_full_recipe_scan.csv", index=False)
winner = full_recipe_scan.iloc[0]
selected_persistence = float(winner["persistence_weight"])
selected_trend = float(winner["trend_weight"])
selected_length = float(winner["length_weight"])

# Reference parity is measured on Fold 3 before the untouched fold is loaded.
selection_reference_rows = []
for mode in ("shift", "raw", "perwell_scale"):
    metrics, _, _ = _evaluate(
        selection_records,
        lambda record, mode=mode: _decode(
            record,
            priors,
            global_scale,
            selected_slots,
            selected_persistence,
            selected_trend,
            selected_length,
            reference_mode=mode,
        )[0],
    )
    selection_reference_rows.append({"reference_mode": mode, **metrics})
    print("V5 REFERENCE", mode, metrics["rmse"], flush=True)
selection_reference_scan = pd.DataFrame(selection_reference_rows).sort_values("rmse")
selection_reference_scan.to_csv(OUT_DIR / "fold3_reference_scan.csv", index=False)

valid_records = [_load_record(well) for well in valid_wells]
fold2_metrics, fold2_bywell, fold2_predictions = _evaluate(
    valid_records,
    lambda record: _decode(
        record,
        priors,
        global_scale,
        selected_slots,
        selected_persistence,
        selected_trend,
        selected_length,
        reference_mode="shift",
    )[0],
    keep_predictions=True,
)
fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

donor_map = {
    record["well"]: valid_records[(index + 37) % len(valid_records)]
    for index, record in enumerate(valid_records)
}
ablation_specs = {
    "full": dict(pw=selected_persistence, tw=selected_trend, ref="shift"),
    "no_persistence": dict(pw=0.0, tw=selected_trend, ref="shift"),
    "no_drilled_trend": dict(pw=selected_persistence, tw=0.0, ref="shift"),
    "no_reference": dict(pw=selected_persistence, tw=selected_trend, ref="raw"),
    "perwell_scale": dict(pw=selected_persistence, tw=selected_trend, ref="perwell_scale"),
}
ablation_rows = []
for name, spec in ablation_specs.items():
    if name == "full":
        metrics = fold2_metrics
    else:
        metrics, _, _ = _evaluate(
            valid_records,
            lambda record, spec=spec: _decode(
                record,
                priors,
                global_scale,
                selected_slots,
                spec["pw"],
                spec["tw"],
                selected_length,
                reference_mode=spec["ref"],
            )[0],
        )
    ablation_rows.append({"mode": name, **metrics})

shuffled_metrics, _, _ = _evaluate(
    valid_records,
    lambda record: _decode(
        record,
        priors,
        global_scale,
        selected_slots,
        selected_persistence,
        selected_trend,
        selected_length,
        reference_mode="shift",
        donor=donor_map[record["well"]],
    )[0],
)
ablation_rows.append({"mode": "shuffled_typewell", **shuffled_metrics})
ablations = pd.DataFrame(ablation_rows)
for column in ("rmse", "far_gt2000_rmse"):
    ablations[f"degradation_{column}_vs_full"] = (
        ablations[column] - fold2_metrics[column]
    )
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
row_error_correlation = np.nan
well_error_correlation = np.nan
blend_scan = None
if POOL_PATH is not None:
    header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
    strong_column = next(
        (column for column in ("strong", "strong_bias", "cp12") if column in header),
        None,
    )
    if strong_column is not None:
        usecols = [
            column
            for column in ("id", "well", "row_idx", "_row_idx", strong_column)
            if column in header
        ]
        pool = pd.read_csv(POOL_PATH, usecols=usecols, dtype={"id": str, "well": str})
        if "well" not in pool:
            pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
        if "row_idx" not in pool:
            if "_row_idx" in pool:
                pool["row_idx"] = pd.to_numeric(pool["_row_idx"], errors="raise").astype(int)
            else:
                pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(int)
        merged = fold2_rows.merge(
            pool[["well", "row_idx", strong_column]],
            on=["well", "row_idx"],
            how="left",
            validate="one_to_one",
        )
        if merged[strong_column].notna().all():
            stride_error = merged["prediction"].to_numpy(float) - merged["truth"].to_numpy(float)
            strong_error = merged[strong_column].to_numpy(float) - merged["truth"].to_numpy(float)
            row_error_correlation = float(np.corrcoef(stride_error, strong_error)[0, 1])
            signed = pd.DataFrame(
                {
                    "well": merged["well"],
                    "stride": stride_error,
                    "strong": strong_error,
                }
            ).groupby("well")[["stride", "strong"]].mean()
            well_error_correlation = float(signed.corr().iloc[0, 1])
            strong_rmse = float(np.sqrt(np.mean(strong_error**2)))
            strong_context = {
                "column": strong_column,
                "rows": int(len(merged)),
                "wells": int(merged["well"].nunique()),
                "rmse": strong_rmse,
            }
            blend_rows = []
            for stride_weight in np.linspace(0.0, 0.60, 13):
                prediction = (
                    (1.0 - stride_weight) * merged[strong_column].to_numpy(float)
                    + stride_weight * merged["prediction"].to_numpy(float)
                )
                blend_rows.append(
                    {
                        "stride_weight": float(stride_weight),
                        "rmse": float(
                            np.sqrt(np.mean((prediction - merged["truth"].to_numpy(float)) ** 2))
                        ),
                    }
                )
            blend_scan = pd.DataFrame(blend_rows)
            blend_scan.to_csv(OUT_DIR / "fold2_strong_blend_diagnostic.csv", index=False)


def _ablation(name):
    return ablations.loc[ablations["mode"].eq(name)].iloc[0]


no_persistence = _ablation("no_persistence")
no_reference = _ablation("no_reference")
shuffled = _ablation("shuffled_typewell")
fold3_shift = selection_reference_scan.loc[
    selection_reference_scan["reference_mode"].eq("shift")
].iloc[0]
fold3_raw = selection_reference_scan.loc[
    selection_reference_scan["reference_mode"].eq("raw")
].iloc[0]
diversity_baseline = diversity_scan.loc[
    diversity_scan["diversity_slots_per_slope"].eq(1)
].iloc[0]
diversity_best = diversity_scan.iloc[0]

path_pass = bool(
    fold2_metrics["rmse"] <= 11.0 and fold2_metrics["far_gt2000_rmse"] <= 13.0
)
component_pass = bool(
    no_persistence["rmse"] >= fold2_metrics["rmse"] + 0.80
    and no_reference["rmse"] >= fold2_metrics["rmse"] + 0.30
)
reference_selection_pass = bool(fold3_shift["rmse"] <= fold3_raw["rmse"])
diversity_gain = float(diversity_baseline["rmse"] - diversity_best["rmse"])
diversity_pass = bool(selected_slots > 1 and diversity_gain >= 0.20)
causal_pass = bool(shuffled["rmse"] >= fold2_metrics["rmse"] + 1.0)
independence_pass = bool(
    np.isfinite(row_error_correlation) and row_error_correlation <= 0.45
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_fold2_diversity_preserving_STRIDE_backbone_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_cp12_gold_spatial": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
    },
    "decoder": {
        "node_step_ft": NODE_STEP_FT,
        "segment_lengths_ft": SEGMENT_LENGTH_NODES * NODE_STEP_FT,
        "absolute_slopes": ABSOLUTE_SLOPES,
        "diversity_slots_per_slope": selected_slots,
        "depth_separation_ft": DEPTH_SEPARATION_FT,
        "emission": "segment Cauchy with outer-train fixed scale",
        "emission_samples_per_segment": EMISSION_SAMPLES,
        "reference": "known-prefix median GR-level shift",
    },
    "outer_train_priors": priors,
    "outer_train_cauchy_scale": scale_fit,
    "selected_recipe": {
        "persistence_weight": selected_persistence,
        "trend_weight": selected_trend,
        "length_weight": selected_length,
    },
    "fast_diversity_scan": diversity_scan.to_dict(orient="records"),
    "fold3_reference_scan": selection_reference_scan.to_dict(orient="records"),
    "fold2": fold2_metrics,
    "fold2_ablations": ablations.to_dict(orient="records"),
    "strong_context": strong_context,
    "error_correlation": {
        "row_level": row_error_correlation,
        "well_mean_signed_error": well_error_correlation,
    },
    "best_same_fold_blend_diagnostic": (
        blend_scan.sort_values("rmse").iloc[0].to_dict()
        if blend_scan is not None
        else None
    ),
    "gates": {
        "rmse_le": 11.0,
        "far_gt2000_rmse_le": 13.0,
        "no_persistence_degradation_ge": 0.80,
        "no_reference_degradation_ge": 0.30,
        "shuffled_typewell_degradation_ge": 1.0,
        "diversity_gain_ge": 0.20,
        "row_error_correlation_le": 0.45,
    },
    "diagnostics": {
        "diversity_gain": diversity_gain,
        "reference_selection_pass": reference_selection_pass,
    },
    "path_pass": path_pass,
    "component_pass": component_pass,
    "diversity_pass": diversity_pass,
    "causal_pass": causal_pass,
    "independence_pass": independence_pass,
    "strict_pass": bool(
        path_pass
        and component_pass
        and reference_selection_pass
        and diversity_pass
        and causal_pass
        and independence_pass
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> generate all-fold STRIDE OOF, then cross-fit track8/sp45 disagreement weights"
        if (
            path_pass
            and component_pass
            and reference_selection_pass
            and diversity_pass
            and causal_pass
            and independence_pass
        )
        else "FAIL -> this implementation still does not reproduce the published STRIDE backbone"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV5 DIVERSITY SCAN")
print(diversity_scan.to_string(index=False), flush=True)
print("\nV5 FOLD3 FULL RECIPES")
print(full_recipe_scan.to_string(index=False), flush=True)
print("\nV5 FOLD3 REFERENCES")
print(selection_reference_scan.to_string(index=False), flush=True)
print("\nV5 FOLD2 ABLATIONS")
print(ablations.to_string(index=False), flush=True)
if blend_scan is not None:
    print("\nV5 STRONG BLEND DIAGNOSTIC")
    print(blend_scan.sort_values("rmse").head(10).to_string(index=False), flush=True)
print("\nV5 FINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
