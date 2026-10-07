"""24-well gateway: rerank complete SP45 seed paths with STRIDE costs.

No blockwise recombination is allowed.  Each candidate is one complete path
produced by an original raw-reference SP45 PF seed.  Label-free whole-path
Cauchy and structure-rate persistence costs rerank those candidates.  Truth is
read only after every candidate path and score have been frozen.
"""

from __future__ import annotations

import ast
import hashlib
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
EXPERIMENT = "sp45_seedpath_stride_rerank_gateway_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
BANK_DIR = OUT_DIR / "seed_banks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BANK_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "sp45_localref_fold2_gateway_v2_cell.py",
    Path.cwd() / "sp45_localref_fold2_gateway_v2_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload sp45_localref_fold2_gateway_v2_cell.py")
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "folds.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
if FOLD_PATH is None:
    raise FileNotFoundError("Missing foldclean fold metadata")
PRIOR_PATH = OOF_ROOT / "stride_sp45_fivefold_oof_v9" / "folds" / "fold3_prior.json"
if not PRIOR_PATH.exists():
    raise FileNotFoundError(PRIOR_PATH)

DISCOVERY_FOLD = 3
DISCOVERY_WELLS = 24
PF_SEEDS = 128
PF_PARTICLES = 500
N_JOBS = int(os.environ.get("SEEDPATH_JOBS", str(min(8, os.cpu_count() or 1))))
FORCE = bool(int(os.environ.get("SEEDPATH_FORCE", "0")))
BLOCK_FT = 128.0
CAUCHY_TEMPERATURES = (0.25, 0.5, 1.0, 2.0)
PERSISTENCE_WEIGHTS = (0.1, 0.3, 1.0, 3.0)
JOINT_TEMPERATURES = (0.5, 1.0)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    name = "sp45_embedded_definitions_for_seedpath_v0"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    keep_names = {
        "_find_data_root",
        "_embedded_prefix_setup",
        "_embedded_run_pf_seed",
    }
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep_names:
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    module.ROOT = ROOT
    module.PF_SEEDS = PF_SEEDS
    module.PF_PARTICLES = PF_PARTICLES
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


def _stable_order(well):
    return hashlib.sha256(f"seedpath-stride-v0:{well}".encode("ascii")).digest()


def _bank_path(well):
    return BANK_DIR / f"{well}.npz"


def _bank_valid(path):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["members"].shape[0] == PF_SEEDS
                and int(data["pf_particles"][0]) == PF_PARTICLES
                and len(data["row_idx"]) >= 50
            )
    except Exception:
        return False


V = _bootstrap(SOURCE_PATH)
DATA_ROOT = V._find_data_root()
prior = json.loads(PRIOR_PATH.read_text(encoding="utf-8"))
sigma_persist = float(max(prior["sigma_persist"], 1e-4))

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
well_column = next(column for column in ("well", "well_id", "Well") if column in folds)
fold_column = next(column for column in ("fold", "outer_fold", "Fold") if column in folds)
folds = folds[[well_column, fold_column]].copy()
folds.columns = ["well", "fold"]
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
fold3_wells = sorted(
    folds.loc[folds["fold"].eq(DISCOVERY_FOLD), "well"].drop_duplicates(),
    key=_stable_order,
)
if len(fold3_wells) < 154:
    raise RuntimeError(f"Unexpected Fold3 size: {len(fold3_wells)}")
discovery_wells = fold3_wells[:DISCOVERY_WELLS]


def _build_bank(well):
    destination = _bank_path(well)
    if not FORCE and _bank_valid(destination):
        return {"well": well, "status": "cached"}
    try:
        hw = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        payload = V._embedded_prefix_setup(hw, tw)
        members = []
        log_likelihood = np.empty(PF_SEEDS, np.float64)
        for seed in range(PF_SEEDS):
            path, score = V._embedded_run_pf_seed(payload, seed)
            members.append(path)
            log_likelihood[seed] = score
        members = np.stack(members).astype(np.float32)
        np.savez_compressed(
            destination,
            well=np.asarray([well]),
            row_idx=payload["eval_index"].astype(np.int32),
            md=payload["md_eval"].astype(np.float32),
            z=payload["z_eval"].astype(np.float32),
            gr=payload["gr_eval"].astype(np.float32),
            truth=payload["truth_eval"].astype(np.float32),
            tw_tvt=payload["tw_tvt"].astype(np.float32),
            tw_gr=payload["tw_gr"].astype(np.float32),
            gr_scale=np.asarray([payload["gr_scale"]], np.float32),
            initial_rate=np.asarray([payload["initial_rate"]], np.float32),
            last_md=np.asarray([payload["last_md"]], np.float32),
            last_tvt=np.asarray([payload["last_tvt"]], np.float32),
            last_z=np.asarray([payload["last_z"]], np.float32),
            members=members,
            log_likelihood=log_likelihood,
            pf_particles=np.asarray([PF_PARTICLES], np.int16),
        )
        return {"well": well, "status": "built", "rows": members.shape[1]}
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}


print(
    "SEEDPATH GATEWAY CONFIG",
    {
        "fold": DISCOVERY_FOLD,
        "discovery_wells": len(discovery_wells),
        "seeds": PF_SEEDS,
        "particles": PF_PARTICLES,
        "jobs": N_JOBS,
        "block_ft": BLOCK_FT,
        "sigma_persist": sigma_persist,
        "candidate_unit": "one complete PF seed path",
    },
    flush=True,
)

# The SP45 helpers are bootstrapped into a dynamic in-memory module.  A loky
# worker cannot import that module while unpickling _build_bank, whereas
# threads share the already initialised namespace and the NumPy/Numba-heavy PF
# work still runs concurrently.
reports = Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(_build_bank)(well) for well in discovery_wells
)
build_report = pd.DataFrame(reports)
build_report.to_csv(OUT_DIR / "build_report.csv", index=False)
failures = build_report.loc[build_report["status"].eq("failed")]
if len(failures):
    raise RuntimeError(f"{len(failures)} banks failed: {failures.to_dict(orient='records')}")


def _robust_z(cost):
    cost = np.asarray(cost, np.float64)
    centre = float(np.median(cost))
    scale = float(1.4826 * np.median(np.abs(cost - centre)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(cost))
    return (cost - centre) / max(scale, 1e-8)


def _soft_path(members, cost, temperature):
    logits = -np.asarray(cost, np.float64) / max(float(temperature), 1e-8)
    logits -= np.max(logits)
    weights = np.exp(np.clip(logits, -80.0, 0.0))
    weights /= np.sum(weights)
    return np.sum(members * weights[:, None], axis=0, dtype=np.float64)


def _candidate_costs(data):
    members = data["members"].astype(np.float64)
    md = data["md"].astype(np.float64)
    z = data["z"].astype(np.float64)
    gr = data["gr"].astype(np.float64)
    tw_tvt = data["tw_tvt"].astype(np.float64)
    tw_gr = data["tw_gr"].astype(np.float64)
    gamma = max(float(data["gr_scale"][0]), 1e-6)
    expected = np.interp(members.ravel(), tw_tvt, tw_gr).reshape(members.shape)
    residual = (gr[None, :] - expected) / gamma
    cauchy = np.mean(np.log1p(np.minimum(residual * residual, 1e4)), axis=1)

    targets = np.arange(md[0], md[-1] + BLOCK_FT, BLOCK_FT)
    if targets[-1] > md[-1]:
        targets[-1] = md[-1]
    targets = np.unique(targets)
    if len(targets) < 3:
        targets = np.linspace(md[0], md[-1], 3)
    z_nodes = np.interp(targets, md, z)
    surface = np.empty((len(members), len(targets)), np.float64)
    for seed in range(len(members)):
        surface[seed] = np.interp(targets, md, members[seed]) + z_nodes
    rates = np.diff(surface, axis=1) / np.diff(targets)[None, :]
    initial = np.full((len(members), 1), float(data["initial_rate"][0]))
    rate_change = np.diff(np.concatenate([initial, rates], axis=1), axis=1)
    persistence = np.mean(0.5 * (rate_change / sigma_persist) ** 2, axis=1)
    return cauchy, persistence


predictions = {}
truth_parts = []
well_parts = []
distance_parts = []
oracle_parts = []
oracle_well_rows = []


def _append(name, values):
    predictions.setdefault(name, []).append(np.asarray(values, np.float32))


for number, well in enumerate(discovery_wells, start=1):
    with np.load(_bank_path(well), allow_pickle=False) as data:
        members = data["members"].astype(np.float64)
        truth = data["truth"].astype(np.float64)
        log_likelihood = data["log_likelihood"].astype(np.float64)
        cauchy, persistence = _candidate_costs(data)
        cauchy_z = _robust_z(cauchy)
        persistence_z = _robust_z(persistence)

        _append("pf_mean", members.mean(axis=0))
        original_cost = -(log_likelihood - np.max(log_likelihood))
        for scale in (3.0, 5.0, 8.0, 12.0):
            _append(f"original_scale_{int(scale)}", _soft_path(members, original_cost, scale))
        for temperature in CAUCHY_TEMPERATURES:
            _append(f"cauchy_t{temperature:g}", _soft_path(members, cauchy_z, temperature))
        _append("cauchy_top1", members[int(np.argmin(cauchy))])
        for weight in PERSISTENCE_WEIGHTS:
            joint = cauchy_z + float(weight) * persistence_z
            for temperature in JOINT_TEMPERATURES:
                _append(
                    f"cauchy_p{weight:g}_t{temperature:g}",
                    _soft_path(members, joint, temperature),
                )
        seed_mse = np.mean((members - truth[None, :]) ** 2, axis=1)
        oracle_index = int(np.argmin(seed_mse))
        oracle_path = members[oracle_index]
        oracle_parts.append(oracle_path.astype(np.float32))
        oracle_well_rows.append(
            {
                "well": well,
                "best_seed": oracle_index,
                "best_seed_rmse": float(np.sqrt(seed_mse[oracle_index])),
                "median_seed_rmse": float(np.sqrt(np.median(seed_mse))),
                "cauchy_rank_of_oracle": int(np.sum(cauchy < cauchy[oracle_index]) + 1),
                "joint_rank_of_oracle": int(
                    np.sum((cauchy_z + persistence_z) < (cauchy_z[oracle_index] + persistence_z[oracle_index])) + 1
                ),
            }
        )
        truth_parts.append(truth.astype(np.float32))
        well_parts.append(np.full(len(truth), well))
        distance_parts.append((data["md"] - float(data["last_md"][0])).astype(np.float32))
    print(f"score seed bank {number}/{len(discovery_wells)}", flush=True)

truth = np.concatenate(truth_parts).astype(np.float64)
wells = np.concatenate(well_parts)
distance = np.concatenate(distance_parts).astype(np.float64)
oracle = np.concatenate(oracle_parts).astype(np.float64)


def _metrics(prediction):
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    bywell = np.sqrt(
        pd.DataFrame({"well": wells, "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
    )
    output = {
        "rows": int(len(error)),
        "wells": int(len(np.unique(wells))),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(bywell.mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }
    near = (distance >= 0.0) & (distance < 500.0)
    far = distance >= 2000.0
    output["near_0_500_rmse"] = float(np.sqrt(np.mean(error[near] ** 2))) if near.any() else np.nan
    output["far_gt2000_rmse"] = float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan
    return output


summary_rows = []
for name, parts in predictions.items():
    summary_rows.append({"variant": name, **_metrics(np.concatenate(parts))})
summary_rows.append({"variant": "whole_seed_oracle", **_metrics(oracle)})
summary = pd.DataFrame(summary_rows).sort_values("rmse", ignore_index=True)
summary.to_csv(OUT_DIR / "discovery_summary.csv", index=False)
pd.DataFrame(oracle_well_rows).to_csv(OUT_DIR / "oracle_bywell.csv", index=False)

baseline_names = [name for name in summary["variant"] if name == "pf_mean" or name.startswith("original_")]
candidate_names = [name for name in summary["variant"] if name.startswith("cauchy")]
baseline = summary.loc[summary["variant"].isin(baseline_names)].sort_values("rmse").iloc[0].to_dict()
selected = summary.loc[summary["variant"].isin(candidate_names)].sort_values("rmse").iloc[0].to_dict()

baseline_prediction = np.concatenate(predictions[baseline["variant"]]).astype(np.float64)
selected_prediction = np.concatenate(predictions[selected["variant"]]).astype(np.float64)
work = pd.DataFrame(
    {
        "well": wells,
        "base2": (baseline_prediction - truth) ** 2,
        "candidate2": (selected_prediction - truth) ** 2,
    }
).groupby("well", sort=False)[["base2", "candidate2"]].mean()
positive_well_fraction = float((work["candidate2"] < work["base2"]).mean())
gain = float(baseline["rmse"] - selected["rmse"])
far_gain = float(baseline["far_gt2000_rmse"] - selected["far_gt2000_rmse"])
oracle_metrics = summary.loc[summary["variant"].eq("whole_seed_oracle")].iloc[0].to_dict()

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_24well_complete_seed_path_reranking_gateway",
    "sample_unit": "one complete PF seed path",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "discovery_fold": DISCOVERY_FOLD,
        "discovery_wells": discovery_wells,
        "truth_usage": "metrics and whole-seed oracle only after all paths and label-free scores were frozen",
        "next_confirmation": "remaining Fold3 wells only if strict_pass",
    },
    "pf": {"seeds": PF_SEEDS, "particles": PF_PARTICLES, "reference": "raw"},
    "reranker": {
        "cauchy_temperatures": list(CAUCHY_TEMPERATURES),
        "persistence_weights": list(PERSISTENCE_WEIGHTS),
        "joint_temperatures": list(JOINT_TEMPERATURES),
        "block_ft": BLOCK_FT,
        "sigma_persist_outer_train": sigma_persist,
        "cross_seed_component_scaling": "per-well robust z-score",
    },
    "baseline": baseline,
    "selected": selected,
    "whole_seed_oracle": oracle_metrics,
    "comparison": {
        "gain_vs_best_original_aggregation": gain,
        "far_gain": far_gain,
        "positive_well_fraction": positive_well_fraction,
    },
    "gates": {
        "whole_seed_oracle_rmse_le_8": bool(oracle_metrics["rmse"] <= 8.0),
        "rerank_gain_ge_0p30": bool(gain >= 0.30),
        "far_gain_ge_0p30": bool(far_gain >= 0.30),
        "positive_well_fraction_ge_0p55": bool(positive_well_fraction >= 0.55),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "artifacts": {
        "banks": str(BANK_DIR),
        "summary": str(OUT_DIR / "discovery_summary.csv"),
        "oracle_bywell": str(OUT_DIR / "oracle_bywell.csv"),
    },
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> freeze the selected score and run the remaining Fold3 wells as confirmation"
    if verdict["strict_pass"]
    else "FAIL -> complete-seed reranking lacks candidate headroom or label-free ordering power"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSEEDPATH RERANK SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
