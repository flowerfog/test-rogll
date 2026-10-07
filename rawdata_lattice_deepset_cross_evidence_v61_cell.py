"""Read-only cross-evidence investment audit for the V60b DeepSets endpoint.

V60b is not competitive as a standalone endpoint, but it may still be useful
if it fails differently from the current 6.377 incumbent.  This cell exactly
reconstructs V60b's Fold-2 paths, verifies parity with the frozen V4 panels and
reported V60b metrics, then measures correlation and blend ceilings.  It fits
no deployable weights and leaves Fold 1 untouched.
"""

from __future__ import annotations

import json
import math
import sys
import time
import types
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch


EXPERIMENT = "rawdata_lattice_deepset_cross_evidence_v61"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V60_SOURCE = next(
    (
        path
        for path in (
            ROOT / "rawdata_lattice_deepset_hardnegative_v60b_cell.py",
            Path.cwd() / "rawdata_lattice_deepset_hardnegative_v60b_cell.py",
        )
        if path.is_file()
    ),
    None,
)
V4_SOURCE = next(
    (
        path
        for path in (
            ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py",
            Path.cwd() / "rawdata_lattice_coarse_ranker_v4_cell.py",
        )
        if path.is_file()
    ),
    None,
)
if V60_SOURCE is None or V4_SOURCE is None:
    raise FileNotFoundError(
        "Upload rawdata_lattice_deepset_hardnegative_v60b_cell.py and "
        "rawdata_lattice_coarse_ranker_v4_cell.py to /root/autodl-tmp"
    )

V60_DIR = OOF_ROOT / "rawdata_lattice_deepset_hardnegative_v60b"
CHECKPOINT = V60_DIR / "models" / "best_outer345_to_2.pt"
NORMALIZER = V60_DIR / "outer_normalizer.npz"
INCUMBENT_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
INCUMBENT_PREFERENCE = (
    "current_dose_0p20",
    "cp8_neighbor",
    "prediction",
    "pred",
)
EXPECTED_V60_MAP_RMSE = 11.107115870
EXPECTED_V60_COARSE_RMSE = 11.193110066
PARITY_TOLERANCE_RMSE = 0.02
FEATURE_TOLERANCE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bootstrap(path, marker, name):
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Execution marker {marker!r} missing from {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
    return module


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _prediction_metrics(frame, column):
    error = frame[column].to_numpy(np.float64) - frame["tvt_true"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": frame["well"], "sq": error * error}).groupby("well")["sq"].mean()
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(bywell.to_numpy()))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell.to_numpy()), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


def _blend(parent, endpoint, truth, groups=None):
    delta = endpoint - parent
    error = parent - truth
    if groups is None:
        denominator = float(np.dot(delta, delta))
        weight = float(np.clip(-np.dot(error, delta) / max(denominator, 1e-12), 0.0, 1.0))
        return parent + weight * delta, weight
    output = parent.copy()
    weights = {}
    for well in pd.unique(groups):
        mask = np.asarray(groups == well)
        denominator = float(np.dot(delta[mask], delta[mask]))
        weight = float(
            np.clip(-np.dot(error[mask], delta[mask]) / max(denominator, 1e-12), 0.0, 1.0)
        )
        output[mask] += weight * delta[mask]
        weights[str(well)] = weight
    return output, weights


def _correlations(frame):
    truth = frame["tvt_true"].to_numpy(np.float64)
    first = frame["incumbent_6p377"].to_numpy(np.float64) - truth
    second = frame["v60b_endpoint"].to_numpy(np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    edges = np.asarray([0.0, 500.0, 1000.0, 2000.0, np.inf])
    labels = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
    strata = np.clip(np.digitize(distance, edges[1:-1], right=True), 0, 3)
    within = {}
    first_demeaned = first.copy()
    second_demeaned = second.copy()
    for index, label in enumerate(labels):
        mask = strata == index
        within[label] = float(np.corrcoef(first[mask], second[mask])[0, 1])
        first_demeaned[mask] -= float(np.mean(first[mask]))
        second_demeaned[mask] -= float(np.mean(second[mask]))
    well_error = pd.DataFrame(
        {"well": frame["well"], "first": first, "second": second}
    ).groupby("well")[["first", "second"]].mean()
    return {
        "signed_error_raw": float(np.corrcoef(first, second)[0, 1]),
        "signed_error_distance_strata_demeaned": float(
            np.corrcoef(first_demeaned, second_demeaned)[0, 1]
        ),
        "within_distance_strata": within,
        "well_mean_signed_error": float(
            np.corrcoef(well_error["first"], well_error["second"])[0, 1]
        ),
    }


started = time.time()
for required in (CHECKPOINT, NORMALIZER, INCUMBENT_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

M = _bootstrap(V60_SOURCE, "\nstarted = time.time()\n", "v60b_definitions_for_v61")
Q = _bootstrap(V4_SOURCE, "\nwells = [\n", "v4_definitions_for_v61")
N = M.N

application_wells = list(M.APPLICATION_WELLS)
if len(application_wells) < 154:
    raise RuntimeError(f"Incomplete canonical Fold2 panel coverage: {len(application_wells)}")

print("V61 fitting exact V4 Fold45 prior", flush=True)
prior_records = []
for number, well in enumerate(M.TRAIN_WELLS, 1):
    prior_records.append(Q.V["_load_record"](well))
    if number % 50 == 0 or number == len(M.TRAIN_WELLS):
        print(f"V61 load prior records {number}/{len(M.TRAIN_WELLS)}", flush=True)
prior = Q.V["_fit_prior"](prior_records, Q.SEGMENT_LENGTH_FT)
del prior_records

outer_booster = lgb.Booster(model_file=str(N.OUTER_COARSE_PATH))
with np.load(NORMALIZER, allow_pickle=False) as data:
    normalizer_mean = data["mean"].astype(np.float32)
    normalizer_std = data["std"].astype(np.float32)
model = M.CandidateDeepSet(len(normalizer_mean)).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
model.eval()

incumbent = pd.read_parquet(INCUMBENT_PATH)
well_column = next((column for column in ("well", "well_id", "Well") if column in incumbent), None)
truth_column = next((column for column in ("tvt_true", "truth", "target", "TVT") if column in incumbent), None)
if well_column is None or truth_column is None or "row_idx" not in incumbent:
    raise RuntimeError(f"Cannot identify incumbent keys/truth: {incumbent.columns.tolist()}")
incumbent_column = next((column for column in INCUMBENT_PREFERENCE if column in incumbent), None)
if incumbent_column is None:
    raise RuntimeError(
        f"Cannot identify incumbent prediction. Available: {incumbent.columns.tolist()}"
    )
incumbent = incumbent.rename(columns={well_column: "well", truth_column: "tvt_true"})
incumbent["well"] = incumbent["well"].astype(str)
incumbent = incumbent[incumbent["well"].isin(application_wells)].copy()
duplicate_rows = (
    incumbent["id"].duplicated().any()
    if "id" in incumbent
    else incumbent[["well", "row_idx"]].duplicated().any()
)
if duplicate_rows:
    raise RuntimeError("Duplicate incumbent rows")

# The incumbent and V4 artifacts were built with different stable-fold
# lineages.  Their numeric fold labels are therefore not comparable; both are
# honest OOF endpoints and are joined only by physical well and row identity.
matched_application_wells = [
    well for well in application_wells if incumbent["well"].eq(well).any()
]
missing_incumbent_wells = [
    well for well in application_wells if well not in set(matched_application_wells)
]
if len(matched_application_wells) < 75:
    raise RuntimeError(
        f"Only {len(matched_application_wells)} cross-lineage matched wells; expected at least 75"
    )
incumbent_fold_labels = (
    incumbent.groupby("fold")["well"].nunique().astype(int).to_dict()
    if "fold" in incumbent
    else {}
)
print(
    "V61 CROSS-LINEAGE COVERAGE",
    {
        "v60b_fold2_wells": len(application_wells),
        "matched_wells": len(matched_application_wells),
        "incumbent_declared_fold_labels": incumbent_fold_labels,
    },
    flush=True,
)

# This reproduces the all-155 V60b metrics directly from frozen panels.  Path
# reconstruction below is needed only on wells that also have incumbent OOF.
panel_parity_metrics, _ = M._evaluate(
    model,
    application_wells,
    outer_booster,
    normalizer_mean,
    normalizer_std,
    "full",
)

parts = []
parity_rows = []
feature_max_abs = 0.0
source_mismatch_count = 0
path_target_max_abs = 0.0
for number, well in enumerate(matched_application_wells, 1):
    record = Q.V["_load_record"](well)
    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
    # Empty reference bins are expected and are converted to the panel's
    # sentinel value immediately after pooling.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        regenerated_features, _ = Q._candidate_evidence(
            record, prior, paths, segments, arm_rank, arm_score_z, source_mask
        )
    panel = N._load_panel(well)
    if regenerated_features.shape != panel["features"].shape:
        raise RuntimeError(
            f"{well}: regenerated panel shape {regenerated_features.shape} != {panel['features'].shape}"
        )
    feature_difference = float(
        np.max(np.abs(regenerated_features.astype(np.float64) - panel["features"].astype(np.float64)))
    )
    feature_max_abs = max(feature_max_abs, feature_difference)
    source_mismatch_count += int(np.count_nonzero(source_mask != panel["source_mask"]))
    if feature_difference > FEATURE_TOLERANCE or source_mismatch_count:
        raise RuntimeError(
            f"{well}: V4 panel lineage mismatch, feature_max_abs={feature_difference}, "
            f"source_mismatches={source_mismatch_count}"
        )

    indices, coarse_score = N._coarse_menu(panel, outer_booster)
    values = N._augment(panel, indices, coarse_score, "full", M._seed(f"full:{well}"))
    values = M._normalise(values, normalizer_mean, normalizer_std)
    with torch.no_grad():
        score = model(torch.from_numpy(values).to(DEVICE)).float().cpu().numpy()
    chosen = int(indices[int(np.argmax(score))])
    coarse = int(indices[0])
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    endpoint = Q.B["_path_prediction"](record, prior, paths[chosen], segments)[eval_rows]
    coarse_prediction = Q.B["_path_prediction"](record, prior, paths[coarse], segments)[eval_rows]
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    endpoint_sse = float(np.sum(np.square(endpoint - truth)))
    coarse_sse = float(np.sum(np.square(coarse_prediction - truth)))
    path_target_max_abs = max(
        path_target_max_abs,
        abs(endpoint_sse - float(panel["sse"][chosen])),
        abs(coarse_sse - float(panel["sse"][coarse])),
    )
    parity_rows.append(
        {
            "well": well,
            "rows": len(eval_rows),
            "endpoint_sse": endpoint_sse,
            "coarse_sse": coarse_sse,
            "feature_max_abs": feature_difference,
        }
    )

    parent_well = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    if len(parent_well) != len(eval_rows):
        raise RuntimeError(f"{well}: incumbent/evaluation rows {len(parent_well)}/{len(eval_rows)}")
    truth_gap = float(np.max(np.abs(parent_well["tvt_true"].to_numpy(np.float64) - truth)))
    if truth_gap > 0.01:
        raise RuntimeError(f"{well}: truth alignment gap {truth_gap}")
    if "md_since" in parent_well:
        distance_gap = float(
            np.max(np.abs(parent_well["md_since"].to_numpy(np.float64) - distance))
        )
        if distance_gap > 1.01:
            raise RuntimeError(f"{well}: md_since alignment gap {distance_gap}")
    else:
        parent_well["md_since"] = distance
    parent_well["tvt_true"] = truth
    parent_well["incumbent_6p377"] = parent_well[incumbent_column].to_numpy(np.float64)
    parent_well["v60b_endpoint"] = endpoint
    parent_well["v60b_coarse"] = coarse_prediction
    keep = [
        column
        for column in ("id", "well", "row_idx", "fold", "tvt_true", "md_since")
        if column in parent_well
    ] + ["incumbent_6p377", "v60b_endpoint", "v60b_coarse"]
    parts.append(parent_well[keep])
    if number % 25 == 0 or number == len(matched_application_wells):
        print(
            f"V61 reconstruct matched wells {number}/{len(matched_application_wells)}",
            flush=True,
        )

frame = pd.concat(parts, ignore_index=True)
parity_frame = pd.DataFrame(parity_rows)
if frame["well"].nunique() < 75:
    raise RuntimeError(
        f"Only {frame['well'].nunique()} matched incumbent wells; expected at least 75"
    )
frame.to_parquet(OUT_DIR / "v60b_cross_evidence_fold2_oof.parquet", index=False)
parity_frame.to_csv(OUT_DIR / "reconstruction_bywell.csv", index=False)

metrics = {
    column: _prediction_metrics(frame, column)
    for column in ("incumbent_6p377", "v60b_endpoint", "v60b_coarse")
}
parent = frame["incumbent_6p377"].to_numpy(np.float64)
endpoint = frame["v60b_endpoint"].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
fixed_prediction, fixed_weight = _blend(parent, endpoint, truth)
perwell_prediction, perwell_weights = _blend(
    parent, endpoint, truth, frame["well"].to_numpy(str)
)
frame["fixed_oracle_blend"] = fixed_prediction
frame["perwell_oracle_blend"] = perwell_prediction
metrics["fixed_oracle_blend"] = _prediction_metrics(frame, "fixed_oracle_blend")
metrics["perwell_oracle_blend"] = _prediction_metrics(frame, "perwell_oracle_blend")

correlation = _correlations(frame)
incumbent_rmse = metrics["incumbent_6p377"]["rmse"]
fixed_gain = incumbent_rmse - metrics["fixed_oracle_blend"]["rmse"]
perwell_increment = metrics["fixed_oracle_blend"]["rmse"] - metrics["perwell_oracle_blend"]["rmse"]
parity_rows_total = int(panel_parity_metrics["map"]["rows"])
parity_map_rmse = float(panel_parity_metrics["map"]["rmse"])
parity_coarse_rmse = float(panel_parity_metrics["coarse"]["rmse"])
parity = {
    "feature_max_abs": feature_max_abs,
    "source_mismatch_count": source_mismatch_count,
    "path_target_sse_max_abs": path_target_max_abs,
    "rows": parity_rows_total,
    "wells": int(panel_parity_metrics["map"]["wells"]),
    "path_reconstruction_matched_wells": int(parity_frame["well"].nunique()),
    "map_reported_rmse": EXPECTED_V60_MAP_RMSE,
    "map_reconstructed_rmse": parity_map_rmse,
    "coarse_reported_rmse": EXPECTED_V60_COARSE_RMSE,
    "coarse_reconstructed_rmse": parity_coarse_rmse,
}
parity_pass = bool(
    abs(parity_map_rmse - EXPECTED_V60_MAP_RMSE) <= PARITY_TOLERANCE_RMSE
    and abs(parity_coarse_rmse - EXPECTED_V60_COARSE_RMSE) <= PARITY_TOLERANCE_RMSE
    and feature_max_abs <= FEATURE_TOLERANCE
    and source_mismatch_count == 0
)
gates = {
    "reconstruction_parity": parity_pass,
    "detrended_error_correlation_le_0p60": bool(
        correlation["signed_error_distance_strata_demeaned"] <= 0.60
    ),
    "fixed_oracle_gain_ge_0p05": bool(fixed_gain >= 0.05),
    "perwell_oracle_increment_ge_0p15": bool(perwell_increment >= 0.15),
}
investment_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_read_only_V60b_vs_6p377_cross_evidence_investment_audit",
    "deployable_weight_fit": False,
    "selection_or_tuning_count": 0,
    "fold1_touched": False,
    "sources": {
        "v60b": V60_DIR,
        "incumbent": INCUMBENT_PATH,
        "incumbent_column": incumbent_column,
    },
    "coverage": {
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "v60b_fold2_wells": len(application_wells),
        "incumbent_matched_wells": frame["well"].nunique(),
        "incumbent_missing_wells": missing_incumbent_wells,
        "incumbent_declared_fold_labels_on_intersection": incumbent_fold_labels,
        "fold_label_comparison_valid": False,
        "application_fold": int(N.APPLICATION_FOLD),
    },
    "parity": parity,
    "metrics": metrics,
    "correlation": correlation,
    "oracle_audit": {
        "fixed_weight_v60b": fixed_weight,
        "fixed_gain_vs_incumbent": fixed_gain,
        "perwell_oracle_increment_over_fixed": perwell_increment,
        "mean_perwell_oracle_weight_v60b": float(np.mean(list(perwell_weights.values()))),
        "median_perwell_oracle_weight_v60b": float(np.median(list(perwell_weights.values()))),
    },
    "gates": gates,
    "investment_pass": investment_pass,
    "strict_pass": investment_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> build Fold1 V4 panels; fit a deployable blend rule on Fold3 and confirm once on Fold1"
        if investment_pass
        else "FAIL -> V60b has no useful fusion ceiling against 6.377; close the V3-bank selector"
    ),
}

print("\nV61 CROSS-EVIDENCE METRICS")
print(pd.DataFrame(metrics).T.to_string(), flush=True)
print("\nV61 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
_write_json(OUT_DIR / "verdict.json", verdict)
print(f"saved: {OUT_DIR}", flush=True)
