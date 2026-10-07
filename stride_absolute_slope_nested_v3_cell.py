"""STRIDE v3: nested joint segment search with absolute slope proposals.

Figure 1 of the STRIDE note shows a wedge in (h_len, delta-TVT): every
segment length evaluates the same global set of absolute slopes. Persistence
is a transition score, not a restriction on which slopes may be proposed.

Protocol: Folds 4/5 fit priors, Fold 3 selects a recipe, Fold 2 confirms it
once, and Fold 1 is never read.
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
V2_CANDIDATES = (
    ROOT / "stride_lattice_observability_nested_v2_cell.py",
    Path.cwd() / "stride_lattice_observability_nested_v2_cell.py",
)
V2_SOURCE = next((path for path in V2_CANDIDATES if path.exists()), None)
if V2_SOURCE is None:
    raise FileNotFoundError(
        "Upload stride_lattice_observability_nested_v2_cell.py and its v1 dependency"
    )
source = V2_SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(V2_SOURCE))
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
    raise RuntimeError("Cannot locate the v2 definition boundary")
V = {"__name__": "stride_v2_definitions", "__file__": str(V2_SOURCE)}
definition_tree = ast.Module(body=tree.body[:cutoff], type_ignores=[])
exec(compile(definition_tree, str(V2_SOURCE), "exec"), V)

EXPERIMENT = "stride_absolute_slope_nested_v3"
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
# Approximately the six columns visible in Figure 1, plus a 32-ft boundary fallback.
SEGMENT_LENGTH_NODES = np.asarray((1, 2, 3, 5, 7, 10, 16), np.int64)
EMISSION_SAMPLES = 24
LATTICE_CONFIGS = (
    ("s17_w016_b32", np.linspace(-0.16, 0.16, 17), 32),
    ("s17_w020_b32", np.linspace(-0.20, 0.20, 17), 32),
    ("s17_w020_b64", np.linspace(-0.20, 0.20, 17), 64),
    ("s21_w020_b32", np.linspace(-0.20, 0.20, 21), 32),
    ("s21_w020_b64", np.linspace(-0.20, 0.20, 21), 64),
    ("s25_w020_b64", np.linspace(-0.20, 0.20, 25), 64),
    ("s21_w024_b64", np.linspace(-0.24, 0.24, 21), 64),
)
LIKELIHOODS = ("gaussian", "cauchy")
PERSISTENCE_WEIGHTS = (0.0, 1.0, 4.0, 16.0, 64.0)
TREND_WEIGHTS = (0.0, 1.0, 4.0, 16.0)
LENGTH_WEIGHTS = (0.0, 0.01, 0.05, 0.20)
STRATA = V["STRATA"]

_load_record = V["_load_record"]
_fit_outer_priors = V["_fit_outer_priors"]
_node_rows = V["_node_rows"]
_reference = V["_reference"]
_continuous_oracle = V["_continuous_oracle"]
_truth_segment_cost = V["_truth_segment_cost"]
_interp_sorted = V["V"]["_interp_sorted"] if "V" in V else None
_segment_emission = V["V"]["_segment_emission"] if "V" in V else None
if _interp_sorted is None or _segment_emission is None:
    raise RuntimeError("v1 STRIDE kernels were not exposed through the v2 definitions")


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
def _absolute_oracle_decode(md, truth, node_rows, anchor_tvt, absolute_slopes,
                            length_nodes, beam_size, maximum_samples):
    nodes = len(node_rows)
    cost = np.full((nodes, beam_size), 1e30)
    state_tvt = np.zeros((nodes, beam_size))
    state_slope = np.zeros((nodes, beam_size))
    parent_node = np.full((nodes, beam_size), -1, np.int64)
    parent_slot = np.full((nodes, beam_size), -1, np.int64)
    cost[0, 0] = 0.0
    state_tvt[0, 0] = anchor_tvt

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
                for slope_position in range(len(absolute_slopes)):
                    slope = absolute_slopes[slope_position]
                    new_tvt = old_tvt + slope * length_ft
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
                        state_slope[target_node, worst] = slope
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


@njit(cache=True)
def _absolute_joint_decode(md, gr, node_rows, anchor_tvt, drilled_trend,
                           tw_tvt, tw_gr, gr_scale, absolute_slopes,
                           length_nodes, length_mu, length_sigma,
                           change_scale, beam_size, persistence_weight,
                           trend_weight, length_weight, gaussian_mode,
                           maximum_samples):
    nodes = len(node_rows)
    cost = np.full((nodes, beam_size), 1e30)
    state_tvt = np.zeros((nodes, beam_size))
    state_slope = np.zeros((nodes, beam_size))
    parent_node = np.full((nodes, beam_size), -1, np.int64)
    parent_slot = np.full((nodes, beam_size), -1, np.int64)
    cost[0, 0] = 0.0
    state_tvt[0, 0] = anchor_tvt
    state_slope[0, 0] = drilled_trend

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
                previous_slope = state_slope[source_node, source_slot]
                for slope_position in range(len(absolute_slopes)):
                    slope = absolute_slopes[slope_position]
                    new_tvt = old_tvt + slope * length_ft
                    emission = _segment_emission(
                        md, gr, node_rows[source_node], node_rows[target_node],
                        old_tvt, new_tvt, tw_tvt, tw_gr, gr_scale,
                        gaussian_mode, maximum_samples,
                    )
                    difference = (slope - previous_slope) / max(change_scale, 1e-6)
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
                        state_slope[target_node, worst] = slope
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
    return path, cost[nodes - 1, final_slot]


def _oracle(record, lattice):
    _, slopes, beam_size = lattice
    node_rows = _node_rows(record)
    truth = record["truth"].copy()
    truth[record["anchor"]] = record["anchor_tvt"]
    path = _absolute_oracle_decode(
        record["md"], truth, node_rows, record["anchor_tvt"],
        slopes.astype(np.float64), SEGMENT_LENGTH_NODES,
        int(beam_size), EMISSION_SAMPLES,
    )
    return np.interp(
        record["md"][record["evaluation"]], record["md"][node_rows], path
    )


def _decode(record, priors, lattice, likelihood, persistence_weight,
            trend_weight, length_weight, reference="shift", donor=None):
    _, slopes, beam_size = lattice
    tw_tvt, tw_gr, gr_scale = _reference(record, reference, donor=donor)
    node_rows = _node_rows(record)
    path, objective = _absolute_joint_decode(
        record["md"], record["gr"], node_rows,
        record["anchor_tvt"], record["initial_rate"],
        tw_tvt, tw_gr, gr_scale, slopes.astype(np.float64),
        SEGMENT_LENGTH_NODES, priors["length_log_mu"],
        priors["length_log_sigma"], priors["rate_change_scale"],
        int(beam_size), float(persistence_weight), float(trend_weight),
        float(length_weight), likelihood == "gaussian", EMISSION_SAMPLES,
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
        print(f"ABS-SLOPE fit priors {number}/{len(prior_wells)}", flush=True)
priors = _fit_outer_priors(prior_records)
del prior_records
selection_records = [_load_record(well) for well in selection_wells]
valid_records = [_load_record(well) for well in valid_wells]

# Select the smallest lattice within 0.03 ft of the best oracle.
oracle_rows = []
for lattice in LATTICE_CONFIGS:
    metrics, _, _ = _evaluate(selection_records, lambda record, item=lattice: _oracle(record, item))
    oracle_rows.append(
        {
            "config": lattice[0],
            "slope_count": len(lattice[1]),
            "slope_min": float(lattice[1].min()),
            "slope_max": float(lattice[1].max()),
            "beam_size": lattice[2],
            "complexity": int(len(lattice[1]) * lattice[2]),
            **metrics,
        }
    )
    print("ABS-SLOPE ORACLE", lattice[0], metrics["rmse"], flush=True)
oracle_scan = pd.DataFrame(oracle_rows).sort_values("rmse").reset_index(drop=True)
oracle_scan.to_csv(OUT_DIR / "fold3_absolute_lattice_oracle.csv", index=False)
oracle_limit = float(oracle_scan.iloc[0]["rmse"] + 0.03)
parsimonious = oracle_scan[oracle_scan["rmse"] <= oracle_limit].sort_values(
    ["complexity", "rmse"]
).iloc[0]
selected_lattice_name = str(parsimonious["config"])
selected_lattice = next(item for item in LATTICE_CONFIGS if item[0] == selected_lattice_name)

# Observation selection with every prior genuinely removed.
observation_rows = []
for likelihood in LIKELIHOODS:
    metrics, _, _ = _evaluate(
        selection_records,
        lambda record, lk=likelihood: _decode(
            record, priors, selected_lattice, lk, 0.0, 0.0, 0.0
        )[0],
    )
    observation_rows.append({"likelihood": likelihood, **metrics})
    print("ABS-SLOPE OBSERVATION", likelihood, metrics["rmse"], flush=True)
observation_scan = pd.DataFrame(observation_rows).sort_values("rmse").reset_index(drop=True)
observation_scan.to_csv(OUT_DIR / "fold3_likelihood_scan.csv", index=False)
selected_likelihood = str(observation_scan.iloc[0]["likelihood"])

# Persistence and drilled-trend are independent penalties in this implementation.
transition_rows = []
for persistence_weight in PERSISTENCE_WEIGHTS:
    for trend_weight in TREND_WEIGHTS:
        metrics, _, _ = _evaluate(
            selection_records,
            lambda record, pw=persistence_weight, tw=trend_weight: _decode(
                record, priors, selected_lattice, selected_likelihood,
                pw, tw, 0.0
            )[0],
        )
        transition_rows.append(
            {"persistence_weight": persistence_weight, "trend_weight": trend_weight, **metrics}
        )
        print("ABS-SLOPE TRANSITION", persistence_weight, trend_weight, metrics["rmse"], flush=True)
transition_scan = pd.DataFrame(transition_rows).sort_values("rmse").reset_index(drop=True)
transition_scan.to_csv(OUT_DIR / "fold3_transition_scan.csv", index=False)
selected_persistence = float(transition_scan.iloc[0]["persistence_weight"])
selected_trend = float(transition_scan.iloc[0]["trend_weight"])

length_rows = []
for length_weight in LENGTH_WEIGHTS:
    metrics, _, _ = _evaluate(
        selection_records,
        lambda record, lw=length_weight: _decode(
            record, priors, selected_lattice, selected_likelihood,
            selected_persistence, selected_trend, lw
        )[0],
    )
    length_rows.append({"length_weight": length_weight, **metrics})
    print("ABS-SLOPE LENGTH", length_weight, metrics["rmse"], flush=True)
length_scan = pd.DataFrame(length_rows).sort_values("rmse").reset_index(drop=True)
length_scan.to_csv(OUT_DIR / "fold3_length_scan.csv", index=False)
selected_length = float(length_scan.iloc[0]["length_weight"])

recipe = {
    "lattice": selected_lattice_name,
    "segment_lengths_ft": SEGMENT_LENGTH_NODES * NODE_STEP_FT,
    "absolute_slopes": selected_lattice[1],
    "beam_size": selected_lattice[2],
    "reference": "shift",
    "likelihood": selected_likelihood,
    "persistence_weight": selected_persistence,
    "trend_weight": selected_trend,
    "length_weight": selected_length,
}

fold2_metrics, fold2_bywell, fold2_predictions = _evaluate(
    valid_records,
    lambda record: _decode(
        record, priors, selected_lattice, selected_likelihood,
        selected_persistence, selected_trend, selected_length
    )[0],
    keep_predictions=True,
)
fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
fold2_oracle, _, _ = _evaluate(valid_records, lambda record: _oracle(record, selected_lattice))

ablation_rows = []
ablation_specs = {
    "full": (selected_persistence, selected_trend, selected_length, "shift", None),
    "no_persistence": (0.0, selected_trend, selected_length, "shift", None),
    "no_drilled_trend": (selected_persistence, 0.0, selected_length, "shift", None),
    "no_reference": (selected_persistence, selected_trend, selected_length, "raw", None),
}
donor_map = {
    record["well"]: valid_records[(index + 37) % len(valid_records)]
    for index, record in enumerate(valid_records)
}
for name, (pw, tw, lw, reference, _) in ablation_specs.items():
    metrics, _, _ = _evaluate(
        valid_records,
        lambda record, pw=pw, tw=tw, lw=lw, ref=reference: _decode(
            record, priors, selected_lattice, selected_likelihood,
            pw, tw, lw, reference=ref
        )[0],
    )
    ablation_rows.append({"mode": name, **metrics})
shuffled_metrics, _, _ = _evaluate(
    valid_records,
    lambda record: _decode(
        record, priors, selected_lattice, selected_likelihood,
        selected_persistence, selected_trend, selected_length,
        reference="shift", donor=donor_map[record["well"]]
    )[0],
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

full_ablation = ablations.loc[ablations["mode"].eq("full")].iloc[0]
no_persistence = ablations.loc[ablations["mode"].eq("no_persistence")].iloc[0]
no_trend = ablations.loc[ablations["mode"].eq("no_drilled_trend")].iloc[0]
no_reference = ablations.loc[ablations["mode"].eq("no_reference")].iloc[0]
representation_pass = bool(oracle_scan.iloc[0]["rmse"] <= 2.0 and fold2_oracle["rmse"] <= 2.5)
path_pass = bool(fold2_metrics["rmse"] <= 11.0 and fold2_metrics["far_gt2000_rmse"] <= 13.0)
component_pass = bool(
    no_persistence["rmse"] >= full_ablation["rmse"] + 0.50
    and no_trend["rmse"] >= full_ablation["rmse"] + 0.20
    and no_reference["rmse"] >= full_ablation["rmse"] + 0.20
)
independence_pass = bool(np.isfinite(correlation) and correlation <= 0.50)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_absolute_slope_STRIDE_gateway",
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
    },
    "proposal": "all absolute slopes at every segment length",
    "outer_train_priors": priors,
    "fold3_best_oracle": oracle_scan.iloc[0].to_dict(),
    "selected_recipe": recipe,
    "fold2_oracle": fold2_oracle,
    "fold2": fold2_metrics,
    "fold2_ablations": ablations.to_dict(orient="records"),
    "strong_context": strong_context,
    "row_error_correlation_with_strong": correlation,
    "best_same_fold_blend": blend_best,
    "gates": {
        "fold3_oracle_rmse_le": 2.0,
        "fold2_oracle_rmse_le": 2.5,
        "path_rmse_le": 11.0,
        "far_gt2000_rmse_le": 13.0,
        "no_persistence_degradation_ge": 0.50,
        "no_trend_degradation_ge": 0.20,
        "no_reference_degradation_ge": 0.20,
        "error_correlation_le": 0.50,
    },
    "representation_pass": representation_pass,
    "path_pass": path_pass,
    "component_pass": component_pass,
    "independence_pass": independence_pass,
    "strict_pass": bool(representation_pass and path_pass and component_pass and independence_pass),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> generate all-fold OOF and cross-fit independent blending"
        if representation_pass and path_pass and component_pass and independence_pass
        else "FAIL -> compare true component ablations with Figure 10 before another decoder change"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nABSOLUTE-SLOPE ORACLE SCAN")
print(oracle_scan.to_string(index=False), flush=True)
print("\nLIKELIHOOD SCAN")
print(observation_scan.to_string(index=False), flush=True)
print("\nTRANSITION SCAN")
print(transition_scan.to_string(index=False), flush=True)
print("\nLENGTH SCAN")
print(length_scan.to_string(index=False), flush=True)
print("\nFOLD2 ABLATIONS")
print(ablations.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
