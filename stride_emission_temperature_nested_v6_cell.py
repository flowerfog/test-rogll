"""STRIDE v6: nested Cauchy-temperature and reference-dose audit.

V5 established three useful facts: raw-reference decoding was stable across
folds, persistence had the expected large contribution, and a full per-well GR
level shift was unstable. This cell freezes the one-slot-per-slope search and
tests the remaining coupled variable: Cauchy emission temperature versus a
shrunken known-prefix GR-level nudge.

Folds 3/4/5 fit priors, Fold 2 selects the frozen recipe, and Fold 1 is the
confirmation fold. Upload the v1 and v5 source files beside this file; neither
experiment tail is executed.
"""

from pathlib import Path
import ast
import json
import shutil
import sys
import time
import types

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
V5_CANDIDATES = (
    ROOT / "stride_diverse_state_lattice_nested_v5_cell.py",
    Path.cwd() / "stride_diverse_state_lattice_nested_v5_cell.py",
)
V5_SOURCE = next((path for path in V5_CANDIDATES if path.exists()), None)
if V5_SOURCE is None:
    raise FileNotFoundError(
        "Upload stride_joint_segment_decoder_gateway_v1_cell.py and "
        "stride_diverse_state_lattice_nested_v5_cell.py beside this cell"
    )

source = V5_SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(V5_SOURCE))
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
    raise RuntimeError("Cannot locate the v5 definition boundary")
module_name = "stride_v5_definitions_for_v6"
module = types.ModuleType(module_name)
module.__file__ = str(V5_SOURCE)
sys.modules[module_name] = module
exec(
    compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(V5_SOURCE), "exec"),
    module.__dict__,
)
V = module.__dict__


EXPERIMENT = "stride_emission_temperature_nested_v6"
OOF_ROOT = V["OOF_ROOT"]
FOLD_PATH = V["FOLD_PATH"]
POOL_PATH = V["POOL_PATH"]
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_FOLDS = (3, 4, 5)
SELECTION_FOLD = 2
VALID_FOLD = 1
DIVERSITY_SLOTS = 1
TREND_WEIGHT = 0.30
LENGTH_WEIGHT = 0.0

# Scale multiplier, persistence weight. The base outer-train robust scale is
# about 10 GR units, so this covers effective Cauchy scales from ~1.5 to ~16.
TEMPERATURE_RECIPES = (
    (0.15, 0.30),
    (0.15, 1.00),
    (0.25, 0.30),
    (0.25, 1.00),
    (0.40, 0.30),
    (0.40, 1.00),
    (0.40, 3.00),
    (0.65, 1.00),
    (0.65, 3.00),
    (1.00, 1.00),
)
REFERENCE_DOSES = (0.0, 0.125, 0.25, 0.50, 1.0)

_load_record = V["_load_record"]
_fit_outer_priors = V["_fit_outer_priors"]
_fit_global_cauchy_scale = V["_fit_global_cauchy_scale"]
_prefix_reference_residual = V["_prefix_reference_residual"]
_robust_location_scale = V["_robust_location_scale"]
_node_rows = V["_node_rows"]
_diverse_joint_decode = V["_diverse_joint_decode"]
_evaluate = V["_evaluate"]
SEGMENT_LENGTH_NODES = V["SEGMENT_LENGTH_NODES"]
ABSOLUTE_SLOPES = V["ABSOLUTE_SLOPES"]
EMISSION_SAMPLES = V["EMISSION_SAMPLES"]
DEPTH_SEPARATION_FT = V["DEPTH_SEPARATION_FT"]


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


def _reference_dose(record, scale, dose, donor=None):
    if donor is None:
        tw_tvt = record["tw_tvt"]
        tw_gr = record["tw_gr_raw"]
    else:
        tw_tvt = donor["tw_tvt"]
        tw_gr = donor["tw_gr_raw"]
    _, full_shift, rows = _prefix_reference_residual(
        record, tw_tvt=tw_tvt, tw_gr=tw_gr, apply_shift=True
    )
    reference = tw_gr + float(dose) * full_shift
    return tw_tvt.astype(np.float64), reference.astype(np.float64), float(scale), rows


def _decode(
    record,
    priors,
    base_scale,
    scale_multiplier,
    persistence_weight,
    reference_dose,
    donor=None,
):
    scale = float(base_scale * scale_multiplier)
    tw_tvt, tw_gr, gr_scale, _ = _reference_dose(
        record, scale, reference_dose, donor=donor
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
        DIVERSITY_SLOTS,
        float(persistence_weight),
        TREND_WEIGHT,
        LENGTH_WEIGHT,
        EMISSION_SAMPLES,
        DEPTH_SEPARATION_FT,
    )
    prediction = np.interp(
        record["md"][record["evaluation"]], record["md"][node_rows], path
    )
    return prediction, float(objective)


if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")
folds = pd.read_csv(FOLD_PATH, dtype={"well": str}).drop_duplicates("well")
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
prior_wells = sorted(folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"])
selection_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"])
valid_wells = sorted(folds.loc[folds["fold"].eq(VALID_FOLD), "well"])
if (len(prior_wells), len(selection_wells), len(valid_wells)) != (463, 155, 155):
    raise RuntimeError(
        f"Unexpected folds: {len(prior_wells)}/{len(selection_wells)}/{len(valid_wells)}"
    )

started = time.time()
prior_records = []
for number, well in enumerate(prior_wells, 1):
    prior_records.append(_load_record(well))
    if number % 50 == 0 or number == len(prior_wells):
        print(f"V6 fit priors {number}/{len(prior_wells)}", flush=True)
priors = _fit_outer_priors(prior_records)
scale_fit = _fit_global_cauchy_scale(prior_records)
base_scale = float(scale_fit["scale"])
del prior_records

selection_records = [_load_record(well) for well in selection_wells]
temperature_rows = []
for scale_multiplier, persistence_weight in TEMPERATURE_RECIPES:
    metrics, _, _ = _evaluate(
        selection_records,
        lambda record, sm=scale_multiplier, pw=persistence_weight: _decode(
            record, priors, base_scale, sm, pw, 0.0
        )[0],
    )
    temperature_rows.append(
        {
            "scale_multiplier": scale_multiplier,
            "effective_scale": base_scale * scale_multiplier,
            "persistence_weight": persistence_weight,
            **metrics,
        }
    )
    print(
        "V6 FULL TEMPERATURE",
        scale_multiplier,
        persistence_weight,
        metrics["rmse"],
        flush=True,
    )
full_scan = pd.DataFrame(temperature_rows).sort_values("rmse").reset_index(drop=True)
full_scan.to_csv(OUT_DIR / "fold2_full_temperature_scan.csv", index=False)
temperature_winner = full_scan.iloc[0]
selected_scale_multiplier = float(temperature_winner["scale_multiplier"])
selected_persistence = float(temperature_winner["persistence_weight"])

dose_rows = []
selection_predictions = {}
for dose in REFERENCE_DOSES:
    metrics, _, predictions = _evaluate(
        selection_records,
        lambda record, dose=dose: _decode(
            record,
            priors,
            base_scale,
            selected_scale_multiplier,
            selected_persistence,
            dose,
        )[0],
        keep_predictions=True,
    )
    dose_rows.append({"reference_dose": dose, **metrics})
    selection_predictions[dose] = predictions
    print("V6 REFERENCE DOSE", dose, metrics["rmse"], flush=True)
dose_scan = pd.DataFrame(dose_rows).sort_values("rmse").reset_index(drop=True)
dose_scan.to_csv(OUT_DIR / "fold2_reference_dose_scan.csv", index=False)
selected_dose = float(dose_scan.iloc[0]["reference_dose"])

valid_records = [_load_record(well) for well in valid_wells]
fold1_metrics, fold1_bywell, fold1_predictions = _evaluate(
    valid_records,
    lambda record: _decode(
        record,
        priors,
        base_scale,
        selected_scale_multiplier,
        selected_persistence,
        selected_dose,
    )[0],
    keep_predictions=True,
)
fold1_bywell.to_csv(OUT_DIR / "fold1_bywell.csv", index=False)

donor_map = {
    record["well"]: valid_records[(index + 37) % len(valid_records)]
    for index, record in enumerate(valid_records)
}
ablation_rows = [{"mode": "full", **fold1_metrics}]
ablation_specs = {
    "no_persistence": (0.0, selected_dose, None),
    "no_reference": (selected_persistence, 0.0, None),
    "full_reference_dose": (selected_persistence, 1.0, None),
}
for name, (pw, dose, _) in ablation_specs.items():
    metrics, _, _ = _evaluate(
        valid_records,
        lambda record, pw=pw, dose=dose: _decode(
            record,
            priors,
            base_scale,
            selected_scale_multiplier,
            pw,
            dose,
        )[0],
    )
    ablation_rows.append({"mode": name, **metrics})
shuffled_metrics, _, _ = _evaluate(
    valid_records,
    lambda record: _decode(
        record,
        priors,
        base_scale,
        selected_scale_multiplier,
        selected_persistence,
        selected_dose,
        donor=donor_map[record["well"]],
    )[0],
)
ablation_rows.append({"mode": "shuffled_typewell", **shuffled_metrics})
ablations = pd.DataFrame(ablation_rows)
for column in ("rmse", "far_gt2000_rmse"):
    ablations[f"degradation_{column}_vs_full"] = (
        ablations[column] - fold1_metrics[column]
    )
ablations.to_csv(OUT_DIR / "fold1_ablations.csv", index=False)

row_frames = []
for record in valid_records:
    index = record["evaluation"]
    row_frames.append(
        pd.DataFrame(
            {
                "well": record["well"],
                "row_idx": index,
                "truth": record["truth"][index],
                "prediction": fold1_predictions[record["well"]],
                "distance_ft": record["md"][index] - record["anchor_md"],
            }
        )
    )
fold1_rows = pd.concat(row_frames, ignore_index=True)
fold1_rows.to_csv(OUT_DIR / "fold1_rows.csv.gz", index=False, compression="gzip")

strong_context = None
row_error_correlation = np.nan
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
        merged = fold1_rows.merge(
            pool[["well", "row_idx", strong_column]],
            on=["well", "row_idx"],
            how="left",
            validate="one_to_one",
        )
        if merged[strong_column].notna().all():
            stride_error = merged["prediction"].to_numpy(float) - merged["truth"].to_numpy(float)
            strong_error = merged[strong_column].to_numpy(float) - merged["truth"].to_numpy(float)
            row_error_correlation = float(np.corrcoef(stride_error, strong_error)[0, 1])
            strong_rmse = float(np.sqrt(np.mean(strong_error**2)))
            strong_context = {
                "column": strong_column,
                "rmse": strong_rmse,
                "rows": int(len(merged)),
                "wells": int(merged["well"].nunique()),
            }
            blend_rows = []
            truth = merged["truth"].to_numpy(float)
            strong_prediction = merged[strong_column].to_numpy(float)
            stride_prediction = merged["prediction"].to_numpy(float)
            for stride_weight in np.linspace(0.0, 0.60, 13):
                prediction = (
                    (1.0 - stride_weight) * strong_prediction
                    + stride_weight * stride_prediction
                )
                blend_rows.append(
                    {
                        "stride_weight": float(stride_weight),
                        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                    }
                )
            blend_scan = pd.DataFrame(blend_rows)
            blend_scan.to_csv(OUT_DIR / "fold1_strong_blend_diagnostic.csv", index=False)


def _ablation(name):
    return ablations.loc[ablations["mode"].eq(name)].iloc[0]


no_persistence = _ablation("no_persistence")
no_reference = _ablation("no_reference")
shuffled = _ablation("shuffled_typewell")
path_pass = bool(
    fold1_metrics["rmse"] <= 11.0 and fold1_metrics["far_gt2000_rmse"] <= 13.0
)
component_pass = bool(
    no_persistence["rmse"] >= fold1_metrics["rmse"] + 0.80
    and selected_dose > 0.0
    and no_reference["rmse"] >= fold1_metrics["rmse"] + 0.30
)
causal_pass = bool(shuffled["rmse"] >= fold1_metrics["rmse"] + 1.0)
independence_pass = bool(
    np.isfinite(row_error_correlation) and row_error_correlation <= 0.45
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold1_STRIDE_emission_temperature_confirmation",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_cp12_gold_spatial": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "validation_fold": VALID_FOLD,
    },
    "outer_train_priors": priors,
    "outer_train_cauchy_scale": scale_fit,
    "selected_recipe": {
        "scale_multiplier": selected_scale_multiplier,
        "effective_scale": base_scale * selected_scale_multiplier,
        "persistence_weight": selected_persistence,
        "trend_weight": TREND_WEIGHT,
        "reference_dose": selected_dose,
        "diversity_slots_per_slope": DIVERSITY_SLOTS,
    },
    "fold2_temperature_scan": full_scan.to_dict(orient="records"),
    "fold2_reference_dose_scan": dose_scan.to_dict(orient="records"),
    "fold1": fold1_metrics,
    "fold1_ablations": ablations.to_dict(orient="records"),
    "strong_context": strong_context,
    "row_error_correlation_with_strong": row_error_correlation,
    "best_same_fold_blend_diagnostic": (
        blend_scan.sort_values("rmse").iloc[0].to_dict()
        if blend_scan is not None
        else None
    ),
    "gates": {
        "rmse_le": 11.0,
        "far_gt2000_rmse_le": 13.0,
        "no_persistence_degradation_ge": 0.80,
        "selected_reference_dose_gt": 0.0,
        "no_reference_degradation_ge": 0.30,
        "shuffled_typewell_degradation_ge": 1.0,
        "row_error_correlation_le": 0.45,
    },
    "path_pass": path_pass,
    "component_pass": component_pass,
    "causal_pass": causal_pass,
    "independence_pass": independence_pass,
    "strict_pass": bool(path_pass and component_pass and causal_pass and independence_pass),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> generate all-fold OOF and train the track8 disagreement-weight blend"
        if path_pass and component_pass and causal_pass and independence_pass
        else "FAIL -> close the published STRIDE-backbone reproduction; retain only measured insights"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV6 FULL TEMPERATURE SCAN")
print(full_scan.to_string(index=False), flush=True)
print("\nV6 REFERENCE DOSE SCAN")
print(dose_scan.to_string(index=False), flush=True)
print("\nV6 FOLD1 ABLATIONS")
print(ablations.to_string(index=False), flush=True)
if blend_scan is not None:
    print("\nV6 STRONG BLEND DIAGNOSTIC")
    print(blend_scan.sort_values("rmse").head(10).to_string(index=False), flush=True)
print("\nV6 FINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
