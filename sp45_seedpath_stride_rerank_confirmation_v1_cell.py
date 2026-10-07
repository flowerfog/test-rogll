"""Frozen 48-well confirmation for complete SP45 seed-path reranking.

This cell reuses the v0 seed-bank builder and its cached 24 discovery banks,
but evaluates only the next 48 Fold3 wells.  The winning discovery recipe is
frozen: robust-z Cauchy cost + 0.3 * robust-z persistence cost, temperature
0.5.  No confirmation labels are used for recipe selection.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "sp45_seedpath_stride_rerank_confirmation_v1"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V0_CANDIDATES = (
    ROOT / "sp45_seedpath_stride_rerank_gateway_v0_cell.py",
    Path.cwd() / "sp45_seedpath_stride_rerank_gateway_v0_cell.py",
)
V0_PATH = next((path for path in V0_CANDIDATES if path.exists()), None)
if V0_PATH is None:
    raise FileNotFoundError("Upload sp45_seedpath_stride_rerank_gateway_v0_cell.py")

V0_VERDICT = OOF_ROOT / "sp45_seedpath_stride_rerank_gateway_v0" / "verdict.json"
if not V0_VERDICT.exists():
    raise FileNotFoundError(V0_VERDICT)

CONFIRMATION_START = 24
CONFIRMATION_WELLS = 48
FROZEN_PERSISTENCE_WEIGHT = 0.3
FROZEN_TEMPERATURE = 0.5
N_JOBS = int(os.environ.get("SEEDPATH_JOBS", str(min(8, os.cpu_count() or 1))))


def _load_v0(path):
    source = path.read_text(encoding="utf-8")
    marker = "reports = Parallel(n_jobs=N_JOBS, backend=\"threading\")"
    if marker not in source:
        raise RuntimeError("The uploaded v0 source does not contain the threading repair")
    prefix = source.split(marker, 1)[0]
    name = "sp45_seedpath_confirmation_v1_bootstrap"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(prefix, str(path), "exec"), module.__dict__)

    tree = ast.parse(source, filename=str(path))
    keep = {"_robust_z", "_soft_path", "_candidate_costs"}
    body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in keep
    ]
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


B = _load_v0(V0_PATH)
v0_verdict = json.loads(V0_VERDICT.read_text(encoding="utf-8"))
selected_name = str(v0_verdict["selected"]["variant"])
if selected_name != "cauchy_p0.3_t0.5":
    raise RuntimeError(f"Unexpected frozen discovery winner: {selected_name}")

discovery_wells = list(B.discovery_wells)
confirmation_wells = list(
    B.fold3_wells[CONFIRMATION_START:CONFIRMATION_START + CONFIRMATION_WELLS]
)
if len(confirmation_wells) != CONFIRMATION_WELLS:
    raise RuntimeError(f"Only {len(confirmation_wells)} confirmation wells")
overlap = sorted(set(discovery_wells).intersection(confirmation_wells))
if overlap:
    raise RuntimeError(f"Discovery/confirmation overlap: {overlap}")

print(
    "SEEDPATH CONFIRMATION CONFIG",
    {
        "fold": B.DISCOVERY_FOLD,
        "discovery_wells_excluded": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "seeds": B.PF_SEEDS,
        "particles": B.PF_PARTICLES,
        "jobs": N_JOBS,
        "frozen_recipe": selected_name,
        "candidate_unit": "one complete PF seed path",
    },
    flush=True,
)

reports = Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(B._build_bank)(well) for well in confirmation_wells
)
build_report = pd.DataFrame(reports)
build_report.to_csv(OUT_DIR / "build_report.csv", index=False)
failures = build_report.loc[build_report["status"].eq("failed")]
if len(failures):
    raise RuntimeError(f"{len(failures)} banks failed: {failures.to_dict(orient='records')}")

truth_parts = []
well_parts = []
distance_parts = []
prediction_parts = {
    "pf_mean": [],
    "original_scale_12": [],
    "cauchy_t1": [],
    selected_name: [],
    "whole_seed_oracle": [],
}
oracle_rows = []

for number, well in enumerate(confirmation_wells, start=1):
    with np.load(B._bank_path(well), allow_pickle=False) as data:
        members = data["members"].astype(np.float64)
        truth = data["truth"].astype(np.float64)
        log_likelihood = data["log_likelihood"].astype(np.float64)
        cauchy, persistence = B._candidate_costs(data)
        cauchy_z = B._robust_z(cauchy)
        persistence_z = B._robust_z(persistence)

        prediction_parts["pf_mean"].append(members.mean(axis=0).astype(np.float32))
        original_cost = -(log_likelihood - np.max(log_likelihood))
        prediction_parts["original_scale_12"].append(
            B._soft_path(members, original_cost, 12.0).astype(np.float32)
        )
        prediction_parts["cauchy_t1"].append(
            B._soft_path(members, cauchy_z, 1.0).astype(np.float32)
        )
        joint = cauchy_z + FROZEN_PERSISTENCE_WEIGHT * persistence_z
        prediction_parts[selected_name].append(
            B._soft_path(members, joint, FROZEN_TEMPERATURE).astype(np.float32)
        )

        seed_mse = np.mean((members - truth[None, :]) ** 2, axis=1)
        oracle_index = int(np.argmin(seed_mse))
        prediction_parts["whole_seed_oracle"].append(
            members[oracle_index].astype(np.float32)
        )
        oracle_rows.append(
            {
                "well": well,
                "best_seed": oracle_index,
                "best_seed_rmse": float(np.sqrt(seed_mse[oracle_index])),
                "median_seed_rmse": float(np.sqrt(np.median(seed_mse))),
                "cauchy_rank_of_oracle": int(np.sum(cauchy < cauchy[oracle_index]) + 1),
                "joint_rank_of_oracle": int(np.sum(joint < joint[oracle_index]) + 1),
            }
        )
        truth_parts.append(truth.astype(np.float32))
        well_parts.append(np.full(len(truth), well))
        distance_parts.append(
            (data["md"] - float(data["last_md"][0])).astype(np.float32)
        )
    print(f"confirm seed bank {number}/{len(confirmation_wells)}", flush=True)

truth = np.concatenate(truth_parts).astype(np.float64)
wells = np.concatenate(well_parts)
distance = np.concatenate(distance_parts).astype(np.float64)
predictions = {
    name: np.concatenate(parts).astype(np.float64)
    for name, parts in prediction_parts.items()
}


def _metrics(prediction):
    error = np.asarray(prediction, np.float64) - truth
    bywell = np.sqrt(
        pd.DataFrame({"well": wells, "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
    )
    near = (distance >= 0.0) & (distance < 500.0)
    far = distance >= 2000.0
    return {
        "rows": int(len(error)),
        "wells": int(len(bywell)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(bywell.mean()),
        "median_well_rmse": float(bywell.median()),
        "p75_well_rmse": float(bywell.quantile(0.75)),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
        "near_0_500_rmse": (
            float(np.sqrt(np.mean(error[near] ** 2))) if near.any() else np.nan
        ),
        "far_gt2000_rmse": (
            float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan
        ),
    }


summary = pd.DataFrame(
    [{"variant": name, **_metrics(prediction)} for name, prediction in predictions.items()]
).sort_values("rmse", ignore_index=True)
summary.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
pd.DataFrame(oracle_rows).to_csv(OUT_DIR / "oracle_bywell.csv", index=False)

base_name = "pf_mean"
candidate_name = selected_name
base = summary.loc[summary["variant"].eq(base_name)].iloc[0].to_dict()
candidate = summary.loc[summary["variant"].eq(candidate_name)].iloc[0].to_dict()
oracle = summary.loc[summary["variant"].eq("whole_seed_oracle")].iloc[0].to_dict()

bywell = pd.DataFrame(
    {
        "well": wells,
        "base2": (predictions[base_name] - truth) ** 2,
        "candidate2": (predictions[candidate_name] - truth) ** 2,
        "oracle2": (predictions["whole_seed_oracle"] - truth) ** 2,
    }
).groupby("well", sort=False).mean()
bywell["base_rmse"] = np.sqrt(bywell.pop("base2"))
bywell["candidate_rmse"] = np.sqrt(bywell.pop("candidate2"))
bywell["oracle_rmse"] = np.sqrt(bywell.pop("oracle2"))
bywell["gain"] = bywell["base_rmse"] - bywell["candidate_rmse"]
bywell = bywell.sort_values("gain", ascending=False)
bywell.to_csv(OUT_DIR / "confirmation_bywell.csv")

gain = float(base["rmse"] - candidate["rmse"])
far_gain = float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"])
positive_fraction = float((bywell["gain"] > 0.0).mean())
p95_change = float(candidate["p95_well_rmse"] - base["p95_well_rmse"])

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_48well_frozen_complete_seed_path_confirmation",
    "sample_unit": "one complete PF seed path",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "discovery_fold": int(B.DISCOVERY_FOLD),
        "discovery_wells_excluded": discovery_wells,
        "confirmation_wells": confirmation_wells,
        "recipe_selected_from": str(V0_VERDICT),
        "truth_usage": "confirmation metrics and diagnostic whole-seed oracle only",
        "hyperparameter_search_on_confirmation": False,
    },
    "pf": {
        "seeds": int(B.PF_SEEDS),
        "particles": int(B.PF_PARTICLES),
        "reference": "raw",
    },
    "frozen_recipe": {
        "variant": selected_name,
        "persistence_weight": FROZEN_PERSISTENCE_WEIGHT,
        "temperature": FROZEN_TEMPERATURE,
        "block_ft": float(B.BLOCK_FT),
        "sigma_persist_outer_train": float(B.sigma_persist),
    },
    "summary": {row["variant"]: row for row in summary.to_dict(orient="records")},
    "baseline": base,
    "candidate": candidate,
    "whole_seed_oracle": oracle,
    "comparison": {
        "gain_vs_pf_mean": gain,
        "far_gain_vs_pf_mean": far_gain,
        "positive_well_fraction": positive_fraction,
        "p95_change_candidate_minus_base": p95_change,
    },
    "gates": {
        "whole_seed_oracle_rmse_le_8": bool(oracle["rmse"] <= 8.0),
        "confirmation_gain_ge_0p5": bool(gain >= 0.5),
        "confirmation_far_gain_ge_0p5": bool(far_gain >= 0.5),
        "positive_well_fraction_ge_0p45": bool(positive_fraction >= 0.45),
        "p95_nonworse": bool(p95_change <= 0.0),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "artifacts": {
        "shared_banks": str(B.BANK_DIR),
        "summary": str(OUT_DIR / "confirmation_summary.csv"),
        "bywell": str(OUT_DIR / "confirmation_bywell.csv"),
    },
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> run the remaining Fold3 wells with the frozen recipe, then audit against fold-clean SP45"
    if verdict["strict_pass"]
    else "FAIL -> the 24-well reranking gain did not transfer; close complete-seed reranking"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSEEDPATH CONFIRMATION SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
