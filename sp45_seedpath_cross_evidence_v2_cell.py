"""Cross-evidence audit for cached complete-seed SP45 endpoints.

The first 24 Fold3 wells select one endpoint and one fixed blend weight.
The next 48 Fold3 wells are an untouched application set.  The comparison
target is the fold-clean SP45 column in the honest candidate pool.
"""

from __future__ import annotations

import ast
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "sp45_seedpath_cross_evidence_v2"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V0_CANDIDATES = (
    ROOT / "sp45_seedpath_stride_rerank_gateway_v0_cell.py",
    Path.cwd() / "sp45_seedpath_stride_rerank_gateway_v0_cell.py",
)
V0_PATH = next((path for path in V0_CANDIDATES if path.exists()), None)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
if V0_PATH is None:
    raise FileNotFoundError("Upload sp45_seedpath_stride_rerank_gateway_v0_cell.py")
if POOL_PATH is None:
    raise FileNotFoundError("Missing candidate_pool_full_merged_oof_honest_fle.csv")

DISCOVERY_COUNT = 24
APPLICATION_COUNT = 48
ENDPOINTS = ("pf_mean", "original_scale_12", "cauchy_p0.3_t0.5")


def _load_v0(path):
    source = path.read_text(encoding="utf-8")
    marker = "reports = Parallel(n_jobs=N_JOBS, backend=\"threading\")"
    if marker not in source:
        raise RuntimeError("Uploaded v0 source lacks the threading repair")
    prefix = source.split(marker, 1)[0]
    name = "sp45_seedpath_cross_evidence_v2_bootstrap"
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


def _metric(frame, column):
    error = frame[column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    bywell = np.sqrt(
        pd.DataFrame({"well": frame["well"], "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
    )
    distance = frame["distance_ft"].to_numpy(np.float64)
    near = (distance >= 0.0) & (distance < 500.0)
    far = distance >= 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(len(bywell)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(bywell.mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "near_0_500_rmse": float(np.sqrt(np.mean(error[near] ** 2))) if near.any() else np.nan,
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def _correlations(frame, endpoint):
    truth = frame["truth"].to_numpy(np.float64)
    first = frame["sp45"].to_numpy(np.float64) - truth
    second = frame[endpoint].to_numpy(np.float64) - truth
    distance = frame["distance_ft"].to_numpy(np.float64)
    bins = np.digitize(distance, [500.0, 1000.0, 2000.0])
    first_d = first.copy()
    second_d = second.copy()
    within = {}
    names = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
    for index, name in enumerate(names):
        mask = bins == index
        if int(mask.sum()) < 2:
            within[name] = np.nan
            continue
        first_d[mask] -= np.mean(first_d[mask])
        second_d[mask] -= np.mean(second_d[mask])
        within[name] = float(np.corrcoef(first[mask], second[mask])[0, 1])
    return {
        "raw": float(np.corrcoef(first, second)[0, 1]),
        "distance_strata_demeaned": float(np.corrcoef(first_d, second_d)[0, 1]),
        "within_distance_strata": within,
    }


def _optimal_weight(frame, endpoint):
    truth = frame["truth"].to_numpy(np.float64)
    base = frame["sp45"].to_numpy(np.float64)
    other = frame[endpoint].to_numpy(np.float64)
    direction = other - base
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(base - truth, direction) / denominator, 0.0, 1.0))


B = _load_v0(V0_PATH)
discovery_wells = list(B.fold3_wells[:DISCOVERY_COUNT])
application_wells = list(B.fold3_wells[DISCOVERY_COUNT:DISCOVERY_COUNT + APPLICATION_COUNT])
all_wells = discovery_wells + application_wells
if len(set(all_wells)) != DISCOVERY_COUNT + APPLICATION_COUNT:
    raise RuntimeError("Discovery/application overlap")

rows = []
for number, well in enumerate(all_wells, start=1):
    bank_path = B._bank_path(well)
    if not B._bank_valid(bank_path):
        raise FileNotFoundError(f"Missing valid cached seed bank: {bank_path}")
    with np.load(bank_path, allow_pickle=False) as data:
        members = data["members"].astype(np.float64)
        log_likelihood = data["log_likelihood"].astype(np.float64)
        cauchy, persistence = B._candidate_costs(data)
        cauchy_z = B._robust_z(cauchy)
        persistence_z = B._robust_z(persistence)
        original_cost = -(log_likelihood - np.max(log_likelihood))
        endpoint = {
            "pf_mean": members.mean(axis=0),
            "original_scale_12": B._soft_path(members, original_cost, 12.0),
            "cauchy_p0.3_t0.5": B._soft_path(
                members, cauchy_z + 0.3 * persistence_z, 0.5
            ),
        }
        row_idx = data["row_idx"].astype(np.int64)
        payload = {
            "id": [f"{well}_{int(row)}" for row in row_idx],
            "well": well,
            "row_idx": row_idx,
            "truth": data["truth"].astype(np.float32),
            "distance_ft": (
                data["md"].astype(np.float64) - float(data["last_md"][0])
            ).astype(np.float32),
        }
        for name, prediction in endpoint.items():
            payload[name] = np.asarray(prediction, np.float32)
        rows.append(pd.DataFrame(payload))
    if number % 12 == 0 or number == len(all_wells):
        print(f"reconstruct cached endpoint {number}/{len(all_wells)}", flush=True)

endpoint_frame = pd.concat(rows, ignore_index=True)
pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "well", "tvt_true", "sp45"],
    dtype={"id": str, "well": str},
)
pool = pool.loc[pool["well"].isin(all_wells)].copy()
frame = endpoint_frame.merge(pool, on=["id", "well"], how="inner", validate="one_to_one")
if len(frame) != len(endpoint_frame):
    raise RuntimeError(f"Candidate-pool alignment failed: {len(frame)}/{len(endpoint_frame)}")
truth_gap = np.abs(frame["truth"].to_numpy(np.float64) - frame["tvt_true"].to_numpy(np.float64))
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
    raise RuntimeError(f"Truth mismatch: {float(np.nanmax(truth_gap))}")

discovery = frame.loc[frame["well"].isin(discovery_wells)].copy()
application = frame.loc[frame["well"].isin(application_wells)].copy()

scan_rows = []
for endpoint in ENDPOINTS:
    weight = _optimal_weight(discovery, endpoint)
    blend_column = f"blend__{endpoint}"
    discovery[blend_column] = discovery["sp45"] + weight * (
        discovery[endpoint] - discovery["sp45"]
    )
    endpoint_metric = _metric(discovery, endpoint)
    blend_metric = _metric(discovery, blend_column)
    scan_rows.append(
        {
            "endpoint": endpoint,
            "weight_endpoint": weight,
            "sp45_rmse": _metric(discovery, "sp45")["rmse"],
            "endpoint_rmse": endpoint_metric["rmse"],
            "blend_rmse": blend_metric["rmse"],
            "blend_gain_vs_sp45": _metric(discovery, "sp45")["rmse"] - blend_metric["rmse"],
            "detrended_error_correlation": _correlations(discovery, endpoint)["distance_strata_demeaned"],
        }
    )

scan = pd.DataFrame(scan_rows).sort_values("blend_rmse", ignore_index=True)
scan.to_csv(OUT_DIR / "discovery_endpoint_scan.csv", index=False)
selected_endpoint = str(scan.iloc[0]["endpoint"])
selected_weight = float(scan.iloc[0]["weight_endpoint"])

application["fixed_blend"] = application["sp45"] + selected_weight * (
    application[selected_endpoint] - application["sp45"]
)
direction = application[selected_endpoint].to_numpy(np.float64) - application["sp45"].to_numpy(np.float64)
oracle_weight = np.zeros(len(application), np.float64)
for _, indices in application.groupby("well", sort=False).indices.items():
    indices = np.asarray(indices, np.int64)
    error = application["sp45"].to_numpy(np.float64)[indices] - application["truth"].to_numpy(np.float64)[indices]
    delta = direction[indices]
    denominator = float(np.dot(delta, delta))
    oracle_weight[indices] = float(np.clip(-np.dot(error, delta) / denominator, 0.0, 1.0)) if denominator > 1e-12 else 0.0
application["perwell_oracle"] = application["sp45"] + oracle_weight * direction

metrics = {
    "sp45": _metric(application, "sp45"),
    "selected_endpoint": _metric(application, selected_endpoint),
    "fixed_blend": _metric(application, "fixed_blend"),
    "perwell_oracle": _metric(application, "perwell_oracle"),
}
correlation = _correlations(application, selected_endpoint)

well_compare = application.assign(
    sp45_e2=(application["sp45"] - application["truth"]) ** 2,
    blend_e2=(application["fixed_blend"] - application["truth"]) ** 2,
).groupby("well", sort=False)[["sp45_e2", "blend_e2"]].mean()
positive_well_fraction = float((well_compare["blend_e2"] < well_compare["sp45_e2"]).mean())
gain = float(metrics["sp45"]["rmse"] - metrics["fixed_blend"]["rmse"])
far_gain = float(
    metrics["sp45"]["far_gt2000_rmse"] - metrics["fixed_blend"]["far_gt2000_rmse"]
)

application[[
    "id", "well", "row_idx", "truth", "distance_ft", "sp45",
    selected_endpoint, "fixed_blend", "perwell_oracle",
]].to_csv(OUT_DIR / "application_oof.csv.gz", index=False, compression="gzip")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_24_to_48_seedpath_vs_foldclean_SP45_cross_evidence",
    "protocol": {
        "selection_wells": discovery_wells,
        "application_wells": application_wells,
        "endpoint_and_weight_selected_on_first_24_only": True,
        "hyperparameter_search_on_application": False,
    },
    "alignment": {
        "pool": str(POOL_PATH),
        "rows": int(len(frame)),
        "maximum_truth_gap": float(truth_gap.max()),
    },
    "discovery_scan": scan.to_dict(orient="records"),
    "selected_endpoint": selected_endpoint,
    "fixed_weight_endpoint": selected_weight,
    "application_metrics": metrics,
    "application_correlation": correlation,
    "application_comparison": {
        "fixed_blend_gain_vs_sp45": gain,
        "far_gain_vs_sp45": far_gain,
        "positive_well_fraction": positive_well_fraction,
        "perwell_oracle_gain_over_fixed": float(
            metrics["fixed_blend"]["rmse"] - metrics["perwell_oracle"]["rmse"]
        ),
    },
    "gates": {
        "fixed_weight_ge_0p05": bool(selected_weight >= 0.05),
        "detrended_correlation_le_0p70": bool(correlation["distance_strata_demeaned"] <= 0.70),
        "application_gain_ge_0p15": bool(gain >= 0.15),
        "application_far_gain_nonnegative": bool(far_gain >= 0.0),
        "positive_well_fraction_ge_0p45": bool(positive_well_fraction >= 0.45),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> complete the remaining Fold3 endpoints and build a nested confidence-weight model"
    if verdict["strict_pass"]
    else "FAIL -> cached seed-path endpoint does not add to formal fold-clean SP45"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nDISCOVERY ENDPOINT SCAN")
print(scan.to_string(index=False), flush=True)
print("\nAPPLICATION METRICS")
print(pd.DataFrame(metrics).T.to_string(), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
