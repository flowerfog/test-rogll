"""Fold-2 investment gate for the reconstructed STRIDE endpoint.

Decode the frozen V5 recipe with the V6-winning raw typewell reference, align
it to the fold-clean SP45 endpoint, and measure whether it is useful because
it fails differently.  This is an audit, not a deployable weight fit: Fold 2
is used only to estimate the fixed/per-well oracle ceilings before spending
the compute required for nested or five-fold OOF generation.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_sp45_cross_evidence_v7"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "stride_fullsequence_beam_v4_cell.py",
    Path.cwd() / "stride_fullsequence_beam_v4_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
V3_VERDICT = OOF_ROOT / "stride_figure1_posterior_parity_v3" / "verdict.json"
V5_VERDICT = OOF_ROOT / "stride_sequence_scale_repair_v5" / "verdict.json"
V6_VERDICT = OOF_ROOT / "stride_matched_component_ablation_v6" / "verdict.json"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload the updated stride_fullsequence_beam_v4_cell.py")
for required in (V3_VERDICT, V5_VERDICT, V6_VERDICT):
    if not required.exists():
        raise FileNotFoundError(required)
if POOL_PATH is None:
    raise FileNotFoundError("Missing candidate_pool_full_merged_oof_honest_fle.csv")


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "v3" in names:
                break
        body.append(node)
    name = "stride_v4_repaired_definitions_for_v7"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
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


def _metrics(frame, prediction_column):
    error = frame[prediction_column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
    }
    well_mse = pd.Series(error * error).groupby(frame["well"].to_numpy()).mean()
    well_rmse = np.sqrt(well_mse)
    output.update(
        {
            "mean_well_rmse": float(well_rmse.mean()),
            "median_well_rmse": float(well_rmse.median()),
            "p95_well_rmse": float(well_rmse.quantile(0.95)),
        }
    )
    strata = (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    )
    distance = frame["distance_ft"].to_numpy(np.float64)
    for name, low, high in strata:
        mask = (distance >= low) & (distance < high)
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


V = _bootstrap(SOURCE_PATH)
v3 = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
v5 = json.loads(V5_VERDICT.read_text(encoding="utf-8"))
v6 = json.loads(V6_VERDICT.read_text(encoding="utf-8"))
if v6["controls"]["raw_reference"]["rmse"] >= v5["fold2"]["rmse"]:
    raise RuntimeError("V6 did not select raw_reference over the V5 full endpoint")

prior = v3["selected"]["prior"]
temperature = float(v3["selected"]["emission_temperature"])
recipe = v5["selected_recipe"]
folds = pd.read_csv(V.V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
application_wells = sorted(folds.loc[folds["fold"].eq(V.V.APPLICATION_FOLD), "well"])
if len(application_wells) != 155:
    raise RuntimeError(f"Unexpected Fold2 size: {len(application_wells)}")

records = []
for number, well in enumerate(application_wells, start=1):
    records.append(V.load_record(well))
    if number % 50 == 0 or number == len(application_wells):
        print(f"STRIDE v7 load Fold2 {number}/{len(application_wells)}", flush=True)

started = time.time()
V.BEAM_SIZE = 48
rows = []
for number, record in enumerate(records, start=1):
    prediction, _, _ = V._decode(
        record,
        prior,
        temperature,
        "raw_reference",
        effective_spacing_ft=float(recipe["effective_spacing_ft"]),
        persistence_weight=float(recipe["persistence_weight"]),
        lambda_override=float(recipe["lambda_previous_dip"]),
    )
    evaluation = record["evaluation"]
    rows.append(
        pd.DataFrame(
            {
                "id": [f"{record['well']}_{int(row)}" for row in evaluation],
                "well": record["well"],
                "row_idx": evaluation.astype(np.int64),
                "truth": record["truth"][evaluation].astype(np.float32),
                "stride_raw": prediction.astype(np.float32),
                "distance_ft": (record["md"][evaluation] - record["anchor_md"]).astype(np.float32),
            }
        )
    )
    if number % 25 == 0 or number == len(records):
        print(f"STRIDE v7 decode {number}/{len(records)}", flush=True)
stride = pd.concat(rows, ignore_index=True)

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
required_columns = {"id", "well", "tvt_true", "sp45"}
if not required_columns.issubset(header):
    raise RuntimeError(f"Pool lacks {sorted(required_columns.difference(header))}: {header}")
pool = pd.read_csv(POOL_PATH, usecols=["id", "well", "tvt_true", "sp45"], dtype={"id": str, "well": str})
pool = pool.loc[pool["well"].isin(application_wells)].copy()
if pool["id"].duplicated().any():
    raise RuntimeError("Duplicate Fold2 ids in candidate pool")
frame = stride.merge(pool, on=["id", "well"], how="inner", validate="one_to_one")
if len(frame) != len(stride):
    raise RuntimeError(f"Alignment failure: {len(frame)}/{len(stride)}")
truth_gap = np.abs(frame["truth"].to_numpy(np.float64) - frame["tvt_true"].to_numpy(np.float64))
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
    raise RuntimeError(f"Truth mismatch: max={float(truth_gap.max())}")
frame.drop(columns="tvt_true", inplace=True)

truth = frame["truth"].to_numpy(np.float64)
sp45 = frame["sp45"].to_numpy(np.float64)
stride_prediction = frame["stride_raw"].to_numpy(np.float64)
sp45_error = sp45 - truth
stride_error = stride_prediction - truth
raw_correlation = float(np.corrcoef(sp45_error, stride_error)[0, 1])

distance_bin = pd.cut(
    frame["distance_ft"],
    bins=[-np.inf, 500.0, 1000.0, 2000.0, np.inf],
    labels=False,
).to_numpy()
sp45_detrended = sp45_error.copy()
stride_detrended = stride_error.copy()
stratum_correlations = {}
for index, name in enumerate(("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")):
    mask = distance_bin == index
    sp45_detrended[mask] -= np.mean(sp45_detrended[mask])
    stride_detrended[mask] -= np.mean(stride_detrended[mask])
    stratum_correlations[name] = float(np.corrcoef(sp45_error[mask], stride_error[mask])[0, 1])
detrended_correlation = float(np.corrcoef(sp45_detrended, stride_detrended)[0, 1])

direction = stride_prediction - sp45
denominator = float(np.dot(direction, direction))
fixed_weight = float(np.clip(-np.dot(sp45_error, direction) / denominator, 0.0, 1.0))
frame["fixed_blend"] = sp45 + fixed_weight * direction

oracle_weight = np.zeros(len(frame), np.float64)
for _, indices in frame.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    local_direction = direction[indices]
    local_denominator = float(np.dot(local_direction, local_direction))
    weight = 0.0 if local_denominator <= 1e-12 else float(
        np.clip(-np.dot(sp45_error[indices], local_direction) / local_denominator, 0.0, 1.0)
    )
    oracle_weight[indices] = weight
frame["perwell_oracle"] = sp45 + oracle_weight * direction
frame["perwell_oracle_weight_stride"] = oracle_weight.astype(np.float32)

metrics = {
    "sp45": _metrics(frame, "sp45"),
    "stride_raw": _metrics(frame, "stride_raw"),
    "fixed_blend": _metrics(frame, "fixed_blend"),
    "perwell_oracle": _metrics(frame, "perwell_oracle"),
}
frame[["id", "well", "row_idx", "truth", "stride_raw", "sp45", "distance_ft"]].to_csv(
    OUT_DIR / "fold2_stride_sp45_oof.csv.gz", index=False, compression="gzip"
)
pd.DataFrame(
    {"well": frame["well"], "weight": oracle_weight}
).groupby("well", as_index=False)["weight"].first().to_csv(
    OUT_DIR / "fold2_perwell_oracle_weights.csv", index=False
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_STRIDE_vs_foldclean_SP45_cross_evidence_investment_gate",
    "deployable_weight_fit": False,
    "raw_inputs_only_for_stride": True,
    "alignment": {
        "pool": str(POOL_PATH),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "maximum_truth_gap": float(truth_gap.max()),
    },
    "frozen_stride_recipe": {**recipe, "reference": "raw", "likelihood": "cauchy"},
    "metrics": metrics,
    "correlation": {
        "signed_error_raw": raw_correlation,
        "signed_error_distance_strata_demeaned": detrended_correlation,
        "within_distance_strata": stratum_correlations,
    },
    "oracle_audit": {
        "fixed_weight_stride": fixed_weight,
        "fixed_gain_vs_sp45": float(metrics["sp45"]["rmse"] - metrics["fixed_blend"]["rmse"]),
        "perwell_oracle_gain_vs_fixed": float(metrics["fixed_blend"]["rmse"] - metrics["perwell_oracle"]["rmse"]),
        "mean_perwell_oracle_weight_stride": float(np.mean(oracle_weight)),
    },
    "gates": {
        "stride_rmse_le_11": bool(metrics["stride_raw"]["rmse"] <= 11.0),
        "detrended_error_correlation_le_0p60": bool(detrended_correlation <= 0.60),
        "fixed_blend_gain_ge_0p30": bool(metrics["sp45"]["rmse"] - metrics["fixed_blend"]["rmse"] >= 0.30),
        "perwell_oracle_gain_over_fixed_ge_0p30": bool(metrics["fixed_blend"]["rmse"] - metrics["perwell_oracle"]["rmse"] >= 0.30),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> generate nested Fold3 STRIDE/SP45 endpoints, freeze the blend rule, then apply once to Fold2"
    if verdict["strict_pass"]
    else "FAIL -> do not spend five-fold compute; the endpoint is not sufficiently complementary to SP45"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSTRIDE-SP45 METRICS")
print(pd.DataFrame(metrics).T.to_string(), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
