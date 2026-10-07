"""M342-G4: read-only deployment-lineage audit for the G3 endpoint stack.

G3 is valid only if its four non-zero absolute endpoints can be reproduced on
Kaggle without substituting an already blended submission. This cell verifies
the OOF lineage, records the frozen coefficients, and emits the exact recipe
needed by a later Kaggle inference cell. It performs no fitting or tuning.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
OUT = OOF / "main342_fullbank_kaggle_lineage_g4"
OUT.mkdir(parents=True, exist_ok=True)

G3_DIR = OOF / "main342_fullbank_fixed_robust_stack_g3"
V168_PATH = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
V144_PATH = OOF / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet"
PSG_PATH = OOF / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet"
NEIGHBOR_PATH = OOF / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet"
NEIGHBOR_VERDICT = OOF / "frontier_6p39_neighbor_parent_audit_v2" / "verdict.json"
V170_LINEAGE = OOF / "historical_analog_parent_lineage_v170" / "verdict.json"

EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
EXPECTED_WEIGHTS = {
    "base_v168": 0.9399999999999695,
    "coarse_crossfit": 0.022155949463498307,
    "psg_tvt": 0.027592981219235015,
    "fle_neighbor": 0.010251069317290395,
}
PARITY_TOLERANCE_FT = 1e-3


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aligned(path, columns, canonical):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    schema = set(pq.read_schema(path).names)
    missing = sorted({"well", "row_idx", *columns}.difference(schema))
    if missing:
        raise RuntimeError(f"{path} missing columns {missing}")
    local = pd.read_parquet(path, columns=["well", "row_idx", *columns])
    local["well"] = local["well"].astype(str)
    local["row_idx"] = local["row_idx"].astype(np.int64)
    local = local.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if len(local) != len(canonical):
        raise RuntimeError(f"{path}: row mismatch {len(local)}/{len(canonical)}")
    if not np.array_equal(local["well"].to_numpy(), canonical["well"].to_numpy()):
        raise RuntimeError(f"{path}: well alignment mismatch")
    if not np.array_equal(local["row_idx"].to_numpy(), canonical["row_idx"].to_numpy()):
        raise RuntimeError(f"{path}: row_idx alignment mismatch")
    values = local[list(columns)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path}: non-finite endpoint values")
    return {column: values[:, index] for index, column in enumerate(columns)}


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    codes, wells = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error * error, minlength=len(wells))
    well_rmse = np.sqrt(sse / counts)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(len(wells)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


def _find_experiment(experiment):
    hits = []
    candidates = list(OOF.glob("**/*.json"))
    direct = ROOT / experiment
    if direct.is_dir():
        candidates.extend(direct.rglob("*.json"))
    for manifest in dict.fromkeys(candidates):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("experiment") == experiment:
            hits.append(manifest)
    return sorted(hits, key=lambda path: path.stat().st_mtime, reverse=True)


for required in (G3_DIR / "verdict.json", V168_PATH, V144_PATH, PSG_PATH, NEIGHBOR_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

g3 = _json(G3_DIR / "verdict.json")
if not g3.get("strict_pass") or not g3.get("deployment"):
    raise RuntimeError("G3 did not pass its frozen confirmation gate")

observed_weights = {
    "base_v168": float(g3["deployment"]["base_weight"]),
    "coarse_crossfit": float(g3["deployment"]["weights"]["v144/coarse_crossfit"]),
    "psg_tvt": float(g3["deployment"]["weights"]["psg/psg_tvt"]),
    "fle_neighbor": float(g3["deployment"]["weights"]["neighbor/fle_neighbor"]),
}
weight_gap = max(abs(observed_weights[key] - value) for key, value in EXPECTED_WEIGHTS.items())
if weight_gap > 1e-12 or abs(sum(observed_weights.values()) - 1.0) > 1e-10:
    raise RuntimeError(f"G3 deployment weights changed: {observed_weights}")

base = pd.read_parquet(
    V168_PATH,
    columns=["id", "well", "row_idx", "fold", "tvt_true", "md_since", "base_current", "metadata_analog_w0p05"],
)
base["well"] = base["well"].astype(str)
base["row_idx"] = base["row_idx"].astype(np.int64)
base = base.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"V168 coverage mismatch {len(base)}/{base['well'].nunique()}")

v144 = _aligned(V144_PATH, ("nested_sequence_coefficient", "coarse_crossfit"), base)
psg = _aligned(PSG_PATH, ("psg_tvt",), base)
neighbor = _aligned(NEIGHBOR_PATH, ("fle_neighbor", "fle_parent"), base)
base_current = base["base_current"].to_numpy(np.float64)
v168 = base["metadata_analog_w0p05"].to_numpy(np.float64)
base_parent_gap = float(np.max(np.abs(base_current - v144["nested_sequence_coefficient"])))

stack = (
    observed_weights["base_v168"] * v168
    + observed_weights["coarse_crossfit"] * v144["coarse_crossfit"]
    + observed_weights["psg_tvt"] * psg["psg_tvt"]
    + observed_weights["fle_neighbor"] * neighbor["fle_neighbor"]
)
summary = pd.DataFrame([
    _metrics(base, v168, "v168_exact"),
    _metrics(base, stack, "g3_full_fit_diagnostic"),
])

neighbor_recipe = None
neighbor_strict = False
if NEIGHBOR_VERDICT.is_file():
    neighbor_verdict = _json(NEIGHBOR_VERDICT)
    fle_payload = neighbor_verdict.get("parents", {}).get("fle")
    if fle_payload is None:
        fle_payload = neighbor_verdict.get("parent_verdicts", {}).get("fle")
    if fle_payload:
        neighbor_recipe = fle_payload.get("selected_recipe")
        neighbor_strict = bool(fle_payload.get("strict_pass"))

v170_deployable = []
if V170_LINEAGE.is_file():
    v170_deployable = _json(V170_LINEAGE).get("deployable_parents", [])

# The current Neighbor Kaggle implementation can reproduce structure-progress;
# other historical bank methods require a separate mapper before deployment.
neighbor_method_supported = bool(
    neighbor_recipe and neighbor_recipe.get("method") == "structure_progress"
)
deployment_assets = {
    "v170_bundle_manifests": [str(path) for path in _find_experiment("historical_analog_deployment_bundle_v170")],
    "v68_bundle_manifests": [str(path) for path in _find_experiment("rawdata_lattice_coarse_canonical_final_bundle_v68")],
    "psg_bundle_manifests": [str(path) for path in _find_experiment("psg_v66_delta_deployment_v5")],
    "neighbor_selected_neighbors": str(OOF / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"),
    "fle_model_package": str(ROOT / "rogii-claude-models-pub"),
}
asset_checks = {
    "v170_bundle_present": bool(deployment_assets["v170_bundle_manifests"]),
    "v68_bundle_present": bool(deployment_assets["v68_bundle_manifests"]),
    "psg_bundle_present": bool(deployment_assets["psg_bundle_manifests"]),
    "neighbor_metadata_present": Path(deployment_assets["neighbor_selected_neighbors"]).is_file(),
    "fle_model_package_present": Path(deployment_assets["fle_model_package"]).is_dir(),
}

checks = {
    "g3_strict_pass": bool(g3.get("strict_pass")),
    "weights_sum_to_one": abs(sum(observed_weights.values()) - 1.0) <= 1e-10,
    "v168_parent_is_v144_nested_sequence": base_parent_gap <= PARITY_TOLERANCE_FT,
    "v170_nested_sequence_parent_is_deployable": "nested_sequence_coefficient" in v170_deployable,
    "neighbor_recipe_available": neighbor_recipe is not None,
    "neighbor_method_supported_by_kaggle_mapper": neighbor_method_supported,
    "all_endpoint_values_finite": bool(np.isfinite(stack).all()),
    **asset_checks,
}
strict_pass = bool(all(checks.values()))
config = {
    "experiment": "main342_fullbank_kaggle_lineage_g4",
    "parent": {
        "endpoint": "V168 exact",
        "formula": "V144 nested_sequence_coefficient + 0.05 * (analog_absolute - V144 nested_sequence_coefficient)",
        "required_kaggle_order": ["V69", "V77", "V116", "V144", "V170-on-V144"],
    },
    "weights": observed_weights,
    "endpoints": {
        "coarse_crossfit": "raw v77_coarse absolute path, before adaptive dose",
        "psg_tvt": "raw PSG_PRED absolute path, not PSG feature-model correction",
        "fle_neighbor": "frozen V2 Neighbor recipe applied to the raw FLE endpoint",
    },
    "neighbor_recipe": neighbor_recipe,
    "neighbor_original_standalone_gate_passed": neighbor_strict,
    "source_hashes": {
        "g3_verdict": _sha256(G3_DIR / "verdict.json"),
        "v168": _sha256(V168_PATH),
        "v144": _sha256(V144_PATH),
        "psg": _sha256(PSG_PATH),
        "neighbor": _sha256(NEIGHBOR_PATH),
    },
}
verdict = {
    "experiment": "main342_fullbank_kaggle_lineage_g4",
    "stage": "zero_training_exact_absolute_endpoint_deployment_lineage_audit",
    "selection_or_tuning_count": 0,
    "coverage": {"rows": len(base), "wells": base["well"].nunique()},
    "weights": observed_weights,
    "v168_parent_max_abs_gap_ft": base_parent_gap,
    "neighbor_recipe": neighbor_recipe,
    "v170_deployable_parents": v170_deployable,
    "deployment_assets": deployment_assets,
    "summary": summary.to_dict("records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> generate the exact G3 Kaggle endpoint cell"
        if strict_pass else
        "FAIL -> repair the reported lineage or missing deployment asset; do not submit a substituted stack"
    ),
}

summary.to_csv(OUT / "summary.csv", index=False)
(OUT / "deployment_config.json").write_text(json.dumps(_clean(config), indent=2), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-G4 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nM342-G4 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
