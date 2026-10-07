"""Five-fold, raw-data-only STRIDE backbone reproduction benchmark.

Purpose
-------
The STRIDE writeup reports 9.99 ft as pooled RMSE over all 773 training
wells decoded leak-free as hidden laterals.  It separately reports 7.28 as a
mean-per-well LOWO number.  This cell establishes the first, all-well pooled
measurement for our structural reproduction and also reports mean-per-well.

This is not presented as the author's exact implementation.  Unpublished
choices (lattice resolution, beam size, candidate rate increments, and prior
weights) remain implementation assumptions inherited from
``stride_joint_segment_decoder_gateway_v1_cell.py``.  The cell does not tune
those choices on any validation fold and does not use SP45, Track8, Strong,
CP12, Gold, Neighbor, or leaderboard feedback.

Run requirements
----------------
Upload these two files to /root/autodl-tmp and run this file:

* stride_joint_segment_decoder_gateway_v1_cell.py
* stride_backbone_773_crossfit_v2_cell.py

The original competition data must be under
/root/autodl-tmp/rogii-wellbore-geology-prediction/train.  Existing fold
metadata is used only for well IDs and fold assignment; no prediction column
is read.
"""

from __future__ import annotations

import ast
import gc
import json
import math
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_backbone_773_crossfit_v2"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "stride_joint_segment_decoder_gateway_v1_cell.py",
    Path.cwd() / "stride_joint_segment_decoder_gateway_v1_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(
        "Upload stride_joint_segment_decoder_gateway_v1_cell.py to "
        "/root/autodl-tmp before running this cross-fit cell."
    )

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
if FOLD_PATH is None:
    raise FileNotFoundError("well_metadata.csv with five-fold assignments was not found")


def _bootstrap_definitions(path: Path):
    """Load definitions without executing the source cell's experiment tail."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "FOLD_PATH is None" in test:
                break
        body.append(node)

    # Numba's on-disk cache serialises the defining module name.  Definitions
    # loaded from an AST bootstrap do not have a normal importable module path,
    # so restoring a cache produced by an earlier notebook execution can fail
    # with ``ModuleNotFoundError: <dynamic>``.  Compile fresh dispatchers in
    # this process; the one-time cost is negligible relative to five-fold
    # decoding.
    class _DisableNumbaDiskCache(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            function_name = None
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name == "njit":
                found = False
                for keyword in node.keywords:
                    if keyword.arg == "cache":
                        keyword.value = ast.Constant(value=False)
                        found = True
                if not found:
                    node.keywords.append(
                        ast.keyword(arg="cache", value=ast.Constant(value=False))
                    )
            return node

    body = [_DisableNumbaDiskCache().visit(node) for node in body]
    ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))

    module_name = "stride_joint_segment_decoder_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    code = compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")
    exec(code, module.__dict__)
    return module


V = _bootstrap_definitions(SOURCE_PATH)
load_record = V._load_record
fit_outer_priors = V._fit_outer_priors
decode = V._decode
metric_row = V._metric_row
summarise = V._summarise


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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


def _safe_parquet(frame: pd.DataFrame, path: Path) -> str:
    try:
        frame.to_parquet(path, index=False)
        return str(path)
    except Exception as exc:
        fallback = path.with_suffix(".csv.gz")
        frame.to_csv(fallback, index=False, compression="gzip")
        print(f"parquet unavailable ({exc!r}); wrote {fallback}", flush=True)
        return str(fallback)


folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
if not {"well", "fold"}.issubset(folds.columns):
    raise RuntimeError(f"Unexpected fold columns: {folds.columns.tolist()}")
folds = folds[["well", "fold"]].drop_duplicates("well").copy()
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
folds = folds[folds["fold"].between(1, 5)].sort_values("well").reset_index(drop=True)
if len(folds) != 773 or set(folds["fold"]) != {1, 2, 3, 4, 5}:
    raise RuntimeError(
        f"Expected 773 wells in five folds; got {len(folds)} and {sorted(folds['fold'].unique())}"
    )

fold_counts = folds.groupby("fold")["well"].nunique().to_dict()
print(
    "STRIDE 773 CROSSFIT CONFIG",
    {
        "source": str(SOURCE_PATH),
        "fold_source": str(FOLD_PATH),
        "fold_counts": fold_counts,
        "metrics": ["pooled_rmse", "mean_well_rmse"],
        "prediction_dependencies": [],
    },
    flush=True,
)

started = time.time()
records = {}
failures = []
for number, well in enumerate(folds["well"], start=1):
    try:
        records[well] = load_record(well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(folds):
        print(f"load raw STRIDE records {number}/{len(folds)} failures={len(failures)}", flush=True)

pd.DataFrame(failures, columns=["well", "reason"]).to_csv(
    OUT_DIR / "record_failures.csv", index=False
)
if failures:
    raise RuntimeError(f"{len(failures)} raw record builds failed: {failures[:5]}")

# The full mode is the candidate reproduction.  The other modes are matched
# component checks, not separately tuned models.
MODES = ("full", "gaussian", "no_calibration", "no_persistence")
all_bywell = []
all_rows = []
prior_reports = {}

for held_fold in range(1, 6):
    train_wells = folds.loc[folds["fold"].ne(held_fold), "well"].tolist()
    valid_wells = folds.loc[folds["fold"].eq(held_fold), "well"].tolist()
    train_records = [records[well] for well in train_wells]
    valid_records = [records[well] for well in valid_wells]
    priors = fit_outer_priors(train_records)
    prior_reports[str(held_fold)] = priors
    print(
        f"FOLD {held_fold} priors fitted from {len(train_records)} wells; "
        f"decode {len(valid_records)} wells",
        flush=True,
    )

    # Trigger numba compilation outside the progress loop.
    decode(valid_records[0], priors, mode="full")

    for mode in MODES:
        for number, record in enumerate(valid_records, start=1):
            prediction, objective, nodes = decode(record, priors, mode=mode)
            item = metric_row(record, prediction, mode, objective, nodes)
            item["fold"] = held_fold
            all_bywell.append(item)

            if mode == "full":
                index = record["evaluation"]
                all_rows.append(
                    pd.DataFrame(
                        {
                            "id": [f"{record['well']}_{int(row)}" for row in index],
                            "well": record["well"],
                            "row_idx": index.astype(np.int64),
                            "fold": held_fold,
                            "truth": record["truth"][index],
                            "stride_backbone": prediction,
                            "distance_ft": record["md"][index] - record["anchor_md"],
                        }
                    )
                )
            if number % 25 == 0 or number == len(valid_records):
                print(
                    f"STRIDE crossfit fold={held_fold} mode={mode} "
                    f"{number}/{len(valid_records)}",
                    flush=True,
                )

    del train_records, valid_records
    gc.collect()

bywell = pd.DataFrame(all_bywell)
bywell.to_csv(OUT_DIR / "fivefold_bywell_components.csv", index=False)
row_oof = pd.concat(all_rows, ignore_index=True)

if row_oof["well"].nunique() != 773:
    raise RuntimeError(f"OOF covers only {row_oof['well'].nunique()} wells")
if row_oof.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("Duplicate OOF well/row_idx keys")
if not np.isfinite(row_oof[["truth", "stride_backbone"]].to_numpy()).all():
    raise RuntimeError("Non-finite values in STRIDE OOF")

oof_path = _safe_parquet(row_oof, OUT_DIR / "stride_backbone_oof.parquet")
_write_json(OUT_DIR / "fivefold_outer_train_priors.json", prior_reports)

summary_rows = []
for mode, frame in bywell.groupby("mode", sort=False):
    metrics = summarise(frame)
    well_rmse = np.sqrt(frame["exact_sse"].to_numpy(float) / frame["rows"].to_numpy(float))
    metrics["mean_well_rmse"] = float(np.mean(well_rmse))
    metrics["mode"] = mode
    summary_rows.append(metrics)
summary = pd.DataFrame(summary_rows)
full = summary.loc[summary["mode"].eq("full")].iloc[0]
summary["pooled_degradation_vs_full"] = summary["rmse"] - float(full["rmse"])
summary["mean_well_degradation_vs_full"] = (
    summary["mean_well_rmse"] - float(full["mean_well_rmse"])
)
summary.to_csv(OUT_DIR / "fivefold_component_summary.csv", index=False)

fold_rows = []
for (mode, fold), frame in bywell.groupby(["mode", "fold"], sort=True):
    metrics = summarise(frame)
    well_rmse = np.sqrt(frame["exact_sse"].to_numpy(float) / frame["rows"].to_numpy(float))
    fold_rows.append(
        {
            "mode": mode,
            "fold": int(fold),
            **metrics,
            "mean_well_rmse": float(np.mean(well_rmse)),
        }
    )
fold_summary = pd.DataFrame(fold_rows)
fold_summary.to_csv(OUT_DIR / "fivefold_fold_summary.csv", index=False)


def _row(mode):
    return summary.loc[summary["mode"].eq(mode)].iloc[0].to_dict()


gaussian = _row("gaussian")
no_calibration = _row("no_calibration")
no_persistence = _row("no_persistence")
full_dict = _row("full")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "all_773_well_leak_free_hidden_lateral_crossfit",
    "claim_scope": (
        "Structural reproduction benchmark only; unpublished STRIDE parameters "
        "are not claimed to be recovered."
    ),
    "sample_unit": "one complete physical well",
    "raw_inputs_only": True,
    "uses_sp45_track8_mainline_strong_cp12_gold_neighbor": False,
    "protocol": {
        "folds": 5,
        "outer_train_wells_per_fold": {str(fold): 773 - count for fold, count in fold_counts.items()},
        "held_wells_per_fold": {str(fold): count for fold, count in fold_counts.items()},
        "official_tvt_input_mask": True,
        "hyperparameter_selection": "none in this cell",
        "test_time_optimisation": "beam search only; zero label fitting",
    },
    "implementation_assumptions": {
        "source": str(SOURCE_PATH),
        "state": "joint segment length and segment TVT rate",
        "node_step_ft": float(V.NODE_STEP_FT),
        "beam_size": int(V.BEAM_SIZE),
        "prefix_typewell_calibration": "robust affine",
        "likelihood": "segment-aggregated Cauchy",
        "note": "These values are not disclosed by the author.",
    },
    "author_reference_not_a_gate": {
        "backbone_pooled_all_773": 9.99,
        "backbone_mean_per_well_lowo": 7.28,
        "persistence_ablation_cost_pooled": 2.30,
        "typewell_reference_ablation_cost_pooled": 1.26,
    },
    "our_crossfit": full_dict,
    "matched_components": {
        "gaussian": gaussian,
        "no_calibration": no_calibration,
        "no_persistence": no_persistence,
    },
    "parity_diagnostics": {
        "persistence_ablation_cost_pooled": float(no_persistence["rmse"] - full_dict["rmse"]),
        "calibration_ablation_cost_pooled": float(no_calibration["rmse"] - full_dict["rmse"]),
        "gaussian_minus_cauchy_pooled": float(gaussian["rmse"] - full_dict["rmse"]),
    },
    "outputs": {
        "row_oof": oof_path,
        "bywell": str(OUT_DIR / "fivefold_bywell_components.csv"),
        "fold_summary": str(OUT_DIR / "fivefold_fold_summary.csv"),
        "component_summary": str(OUT_DIR / "fivefold_component_summary.csv"),
        "priors": str(OUT_DIR / "fivefold_outer_train_priors.json"),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "Use component parity and both metric definitions to revise the backbone; "
        "do not compare to Mainline.  Once frozen, build fold-clean Track8 from SP45."
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSTRIDE 773 COMPONENT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nSTRIDE FIVE-FOLD SUMMARY")
print(fold_summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
