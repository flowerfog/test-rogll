"""Untouched Fold-1 confirmation of the frozen V60b cross-evidence dose.

V61 selected a rounded 0.20 V60b dose on the cross-lineage Fold-2
intersection.  V62 performs no search: it builds the previously absent V4
Fold-1 panels, reconstructs the frozen outer V60b endpoint, and applies that
dose once against the honest 6.377 incumbent OOF.
"""

from __future__ import annotations

import json
import sys
import time
import types
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch


EXPERIMENT = "rawdata_lattice_deepset_fold1_confirmation_v62"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_WEIGHT_V60B = 0.20

V61_SOURCE = next(
    (
        path
        for path in (
            ROOT / "rawdata_lattice_deepset_cross_evidence_v61_cell.py",
            Path.cwd() / "rawdata_lattice_deepset_cross_evidence_v61_cell.py",
        )
        if path.is_file()
    ),
    None,
)
if V61_SOURCE is None:
    raise FileNotFoundError(
        "Upload rawdata_lattice_deepset_cross_evidence_v61_cell.py to /root/autodl-tmp"
    )


def _bootstrap_prefix(path, marker, name):
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


started = time.time()
A = _bootstrap_prefix(V61_SOURCE, "\nstarted = time.time()\n", "v61_definitions_for_v62")
for required in (A.CHECKPOINT, A.NORMALIZER, A.INCUMBENT_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

M = A._bootstrap(A.V60_SOURCE, "\nstarted = time.time()\n", "v60b_definitions_for_v62")
Q = A._bootstrap(A.V4_SOURCE, "\nwells = [\n", "v4_definitions_for_v62")
N = M.N

all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((Q.DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
fold_table = Q.V["_stable_folds"](all_wells)
fold_by_well = dict(
    zip(fold_table["well"].astype(str), fold_table["fold"].astype(int))
)
fold1_requested = [
    well for well in all_wells if fold_by_well.get(well) == int(N.UNTOUCHED_FOLD)
]
if len(fold1_requested) < 154:
    raise RuntimeError(f"Unexpected canonical Fold1 size: {len(fold1_requested)}")

print("V62 fitting exact V4 Fold45 prior", flush=True)
prior_records = []
for number, well in enumerate(M.TRAIN_WELLS, 1):
    prior_records.append(Q.V["_load_record"](well))
    if number % 50 == 0 or number == len(M.TRAIN_WELLS):
        print(f"V62 load prior records {number}/{len(M.TRAIN_WELLS)}", flush=True)
prior = Q.V["_fit_prior"](prior_records, Q.SEGMENT_LENGTH_FT)
del prior_records

fold1_records = {}
record_failures = []
for number, well in enumerate(fold1_requested, 1):
    try:
        fold1_records[well] = Q.V["_load_record"](well)
    except Exception as exc:
        record_failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(fold1_requested):
        print(
            f"V62 load Fold1 {number}/{len(fold1_requested)} failures={len(record_failures)}",
            flush=True,
        )
fold1_wells = [well for well in fold1_requested if well in fold1_records]
if len(fold1_wells) < 154:
    raise RuntimeError(
        f"Insufficient eligible canonical Fold1 wells: {len(fold1_wells)}; "
        f"failures={record_failures[:5]}"
    )

outer_booster = lgb.Booster(model_file=str(N.OUTER_COARSE_PATH))
with np.load(A.NORMALIZER, allow_pickle=False) as data:
    normalizer_mean = data["mean"].astype(np.float32)
    normalizer_std = data["std"].astype(np.float32)
model = M.CandidateDeepSet(len(normalizer_mean)).to(A.DEVICE)
model.load_state_dict(
    torch.load(A.CHECKPOINT, map_location=A.DEVICE, weights_only=True)
)
model.eval()

incumbent = pd.read_parquet(A.INCUMBENT_PATH)
well_column = next(
    (column for column in ("well", "well_id", "Well") if column in incumbent), None
)
truth_column = next(
    (column for column in ("tvt_true", "truth", "target", "TVT") if column in incumbent),
    None,
)
if well_column is None or truth_column is None or "row_idx" not in incumbent:
    raise RuntimeError(f"Cannot identify incumbent keys/truth: {incumbent.columns.tolist()}")
incumbent_column = next(
    (column for column in A.INCUMBENT_PREFERENCE if column in incumbent), None
)
if incumbent_column is None:
    raise RuntimeError(f"Cannot identify incumbent prediction: {incumbent.columns.tolist()}")
incumbent = incumbent.rename(columns={well_column: "well", truth_column: "tvt_true"})
incumbent["well"] = incumbent["well"].astype(str)
incumbent = incumbent[incumbent["well"].isin(fold1_wells)].copy()
duplicate_rows = (
    incumbent["id"].duplicated().any()
    if "id" in incumbent
    else incumbent[["well", "row_idx"]].duplicated().any()
)
if duplicate_rows:
    raise RuntimeError("Duplicate incumbent rows")
matched_wells = [well for well in fold1_wells if incumbent["well"].eq(well).any()]
missing_wells = [well for well in fold1_wells if well not in set(matched_wells)]
if len(matched_wells) < 75:
    raise RuntimeError(f"Only {len(matched_wells)} matched Fold1 wells; expected at least 75")
print(
    "V62 FOLD1 MATCHED COVERAGE",
    {
        "canonical_fold1_wells": len(fold1_wells),
        "matched_wells": len(matched_wells),
        "frozen_weight_v60b": FROZEN_WEIGHT_V60B,
    },
    flush=True,
)

parts = []
feature_max_abs = 0.0
source_mismatch_count = 0
path_target_max_abs = 0.0
for number, well in enumerate(matched_wells, 1):
    record = fold1_records[well]
    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(
        record, prior
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Mean of empty slice", category=RuntimeWarning
        )
        regenerated_features, feature_names = Q._candidate_evidence(
            record, prior, paths, segments, arm_rank, arm_score_z, source_mask
        )
    panel_path = Q._panel_path(well)
    if panel_path.is_file():
        panel = N._load_panel(well)
    else:
        targets = Q._exact_targets(record, prior, paths, segments)
        panel = {
            "features": regenerated_features,
            "source_mask": source_mask,
            "arm_rank": arm_rank,
            "mse": targets["mse"].astype(np.float32),
            "sse": targets["sse"],
            "sae": targets["sae"],
            "far_sse": targets["far_sse"],
            "rows": np.asarray([targets["rows"]], np.int64),
            "far_rows": np.asarray([targets["far_rows"]], np.int64),
        }
        np.savez_compressed(panel_path, **panel)
        _write_json(
            panel_path.with_suffix(".json"),
            {
                "well": well,
                "candidates_after_dedup": len(paths),
                "feature_count": regenerated_features.shape[1],
                "feature_names": feature_names,
                "built_by": EXPERIMENT,
            },
        )
    if regenerated_features.shape != panel["features"].shape:
        raise RuntimeError(f"{well}: regenerated panel shape mismatch")
    feature_difference = float(
        np.max(
            np.abs(
                regenerated_features.astype(np.float64)
                - panel["features"].astype(np.float64)
            )
        )
    )
    feature_max_abs = max(feature_max_abs, feature_difference)
    source_mismatch_count += int(
        np.count_nonzero(source_mask != panel["source_mask"])
    )
    if feature_difference > A.FEATURE_TOLERANCE or source_mismatch_count:
        raise RuntimeError(
            f"{well}: V4 panel lineage mismatch, feature_max_abs={feature_difference}, "
            f"source_mismatches={source_mismatch_count}"
        )

    indices, coarse_score = N._coarse_menu(panel, outer_booster)
    values = N._augment(panel, indices, coarse_score, "full", M._seed(f"full:{well}"))
    values = M._normalise(values, normalizer_mean, normalizer_std)
    with torch.no_grad():
        scores = model(torch.from_numpy(values).to(A.DEVICE)).float().cpu().numpy()
    chosen = int(indices[int(np.argmax(scores))])
    coarse = int(indices[0])
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    endpoint = Q.B["_path_prediction"](record, prior, paths[chosen], segments)[eval_rows]
    coarse_endpoint = Q.B["_path_prediction"](
        record, prior, paths[coarse], segments
    )[eval_rows]
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    endpoint_sse = float(np.sum(np.square(endpoint - truth)))
    coarse_sse = float(np.sum(np.square(coarse_endpoint - truth)))
    path_target_max_abs = max(
        path_target_max_abs,
        abs(endpoint_sse - float(panel["sse"][chosen])),
        abs(coarse_sse - float(panel["sse"][coarse])),
    )

    parent_well = incumbent[incumbent["well"].eq(well)].sort_values(
        "row_idx", kind="stable"
    ).copy()
    if len(parent_well) != len(eval_rows):
        raise RuntimeError(
            f"{well}: incumbent/evaluation rows {len(parent_well)}/{len(eval_rows)}"
        )
    truth_gap = float(
        np.max(np.abs(parent_well["tvt_true"].to_numpy(np.float64) - truth))
    )
    if truth_gap > 0.01:
        raise RuntimeError(f"{well}: truth alignment gap {truth_gap}")
    if "md_since" in parent_well:
        distance_gap = float(
            np.max(
                np.abs(parent_well["md_since"].to_numpy(np.float64) - distance)
            )
        )
        if distance_gap > 1.01:
            raise RuntimeError(f"{well}: md_since alignment gap {distance_gap}")
    else:
        parent_well["md_since"] = distance
    parent = parent_well[incumbent_column].to_numpy(np.float64)
    parent_well["tvt_true"] = truth
    parent_well["incumbent_6p377"] = parent
    parent_well["v60b_endpoint"] = endpoint
    parent_well["v60b_coarse"] = coarse_endpoint
    parent_well["frozen_blend_w0p20"] = (
        parent + FROZEN_WEIGHT_V60B * (endpoint - parent)
    )
    parent_well["coarse_blend_w0p20"] = (
        parent + FROZEN_WEIGHT_V60B * (coarse_endpoint - parent)
    )
    keep = [
        column
        for column in ("id", "well", "row_idx", "fold", "tvt_true", "md_since")
        if column in parent_well
    ] + [
        "incumbent_6p377",
        "v60b_endpoint",
        "v60b_coarse",
        "frozen_blend_w0p20",
        "coarse_blend_w0p20",
    ]
    parts.append(parent_well[keep])
    if number % 25 == 0 or number == len(matched_wells):
        print(f"V62 reconstruct Fold1 {number}/{len(matched_wells)}", flush=True)

frame = pd.concat(parts, ignore_index=True)
frame.to_parquet(OUT_DIR / "fold1_confirmation_oof.parquet", index=False)
metric_columns = (
    "incumbent_6p377",
    "v60b_endpoint",
    "v60b_coarse",
    "frozen_blend_w0p20",
    "coarse_blend_w0p20",
)
metrics = {column: A._prediction_metrics(frame, column) for column in metric_columns}
correlation = A._correlations(frame)
parent_metrics = metrics["incumbent_6p377"]
blend_metrics = metrics["frozen_blend_w0p20"]
gain = parent_metrics["rmse"] - blend_metrics["rmse"]
far_gain = parent_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"]
p95_gain = parent_metrics["p95_well_rmse"] - blend_metrics["p95_well_rmse"]
deepset_increment = (
    metrics["coarse_blend_w0p20"]["rmse"] - blend_metrics["rmse"]
)
checks = {
    "matched_wells_ge_75": bool(frame["well"].nunique() >= 75),
    "fold1_gain_ge_0p05": bool(gain >= 0.05),
    "fold1_far_gain_nonnegative": bool(far_gain >= 0.0),
    "fold1_p95_gain_nonnegative": bool(p95_gain >= 0.0),
    "deepset_increment_over_coarse_positive": bool(deepset_increment > 0.0),
    "panel_reconstruction_exact": bool(
        feature_max_abs <= A.FEATURE_TOLERANCE
        and source_mismatch_count == 0
        and path_target_max_abs <= 1e-5
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "untouched_canonical_Fold1_single_confirmation_of_Fold2_frozen_weight",
    "selection_or_tuning_count_on_fold1": 0,
    "fold1_touched_before_this_cell": False,
    "frozen_recipe": {
        "endpoint": "V60b outer345 DeepSets MAP path",
        "weight_v60b": FROZEN_WEIGHT_V60B,
        "weight_source": "V61 Fold2 analytic optimum 0.201827 rounded to 0.20",
    },
    "coverage": {
        "canonical_fold1_requested": len(fold1_requested),
        "canonical_fold1_eligible": len(fold1_wells),
        "matched_incumbent_wells": frame["well"].nunique(),
        "rows": len(frame),
        "missing_incumbent_wells": missing_wells,
        "record_failures": record_failures,
        "cross_lineage_fold_labels_compared": False,
    },
    "parity": {
        "feature_max_abs": feature_max_abs,
        "source_mismatch_count": source_mismatch_count,
        "path_target_sse_max_abs": path_target_max_abs,
    },
    "metrics": metrics,
    "correlation": correlation,
    "confirmation": {
        "gain": gain,
        "far_gain": far_gain,
        "p95_gain": p95_gain,
        "deepset_increment_over_coarse_blend": deepset_increment,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> package the frozen V60b endpoint and 0.20 blend for a timed Kaggle inference rehearsal"
        if strict_pass
        else "FAIL -> V61 fusion gain does not transfer to untouched Fold1; close V60b deployment"
    ),
}

print("\nV62 FOLD1 CONFIRMATION")
print(pd.DataFrame(metrics).T.to_string(), flush=True)
print("\nV62 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
_write_json(OUT_DIR / "verdict.json", verdict)
print(f"saved: {OUT_DIR}", flush=True)
