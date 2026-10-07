"""V102: investment gate for blend-aware coarse-candidate labels.

The frozen V4/V70 ranker learns standalone candidate-path MSE, although the
deployed endpoint consumes only a V76 dose of that path on top of the 6.377
parent.  This read-only Fold-3 audit asks whether those two ranking objectives
select materially different paths.  Truth is used only for oracle selection
and metrics; no model is fit and Fold-2/Fold-1 remain untouched.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_blendaware_label_oracle_v102"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "well_cache"
for directory in (OUT_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
V67_PRIOR = OOF_ROOT / "rawdata_lattice_coarse_canonical_gateway_v67" / "canonical_fold45_prior.json"
V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

FOLD = 3
DISCOVERY_WELLS = 24
FIXED_DOSE = 0.30
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"


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


def _bootstrap_v4():
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 bootstrap marker missing from {V4_SOURCE}")
    module = types.ModuleType("raw_lattice_v4_blendaware_oracle_definitions")
    module.__file__ = str(V4_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V4_SOURCE), "exec"), module.__dict__)
    return module


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid endpoint {endpoint}")
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def _rank(values):
    order = np.argsort(np.asarray(values, np.float64), kind="stable")
    rank = np.empty(len(order), np.float64)
    rank[order] = np.arange(len(order), dtype=np.float64)
    return rank


def _spearman(left, right):
    left_rank = _rank(left)
    right_rank = _rank(right)
    if len(left_rank) < 2 or left_rank.std() <= 0.0 or right_rank.std() <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


for required in (V4_SOURCE, V67_PRIOR, V70_PATH, V76_WEIGHTS):
    if not required.is_file():
        raise FileNotFoundError(required)

Q = _bootstrap_v4()
V = Q.V
B = Q.B
prior_payload = json.loads(V67_PRIOR.read_text(encoding="utf-8"))
prior = {
    key: np.asarray(value, np.float64) if key == "rate_grid" else value
    for key, value in prior_payload.items()
}

frame = pd.read_parquet(
    V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
    filters=[("fold", "=", FOLD)],
).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
weights = pd.read_parquet(V76_WEIGHTS)
weights = weights[weights["fold"].eq(FOLD)][["well", "ensemble_weight"]].copy()
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))
frame["well"] = frame["well"].astype(str)
frame["dose"] = frame["well"].map(weight_map)
if frame.empty or frame["dose"].isna().any() or frame["id"].duplicated().any():
    raise RuntimeError("Incomplete or duplicate Fold3 V70/V76 alignment")

wells = sorted(frame["well"].unique())
if len(wells) < 153:
    raise RuntimeError(f"Only {len(wells)} Fold3 wells")
discovery = set(wells[:DISCOVERY_WELLS])


def _cache_path(well):
    return CACHE_DIR / f"{well}.npz"


def _build_well(well):
    output = _cache_path(well)
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable")
    signature = hashlib.sha256(local["id"].astype(str).str.cat(sep="|").encode()).hexdigest()
    if output.is_file():
        with np.load(output, allow_pickle=False) as data:
            if str(data["signature"].item()) == signature:
                return "reused"

    record = V["_load_record"](well)
    paths, _, _, _, segments, _ = Q._candidate_union(record, prior)
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
    parent = local[PARENT].to_numpy(np.float64)
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: row count {len(local)}/{len(eval_rows)}")
    truth_gap = float(np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)))
    distance_gap = float(np.max(np.abs(local["md_since"].to_numpy(np.float64) - distance)))
    if truth_gap > 0.01 or distance_gap > 1.01:
        raise RuntimeError(f"{well}: alignment truth={truth_gap} distance={distance_gap}")

    dose = float(local["dose"].iloc[0])
    parent_error = parent - truth
    standalone_sse = np.empty(len(paths), np.float64)
    fixed_sse = np.empty(len(paths), np.float64)
    adaptive_sse = np.empty(len(paths), np.float64)
    best_predictions = {}
    best_values = {"standalone": np.inf, "fixed": np.inf, "adaptive": np.inf}
    best_indices = {"standalone": -1, "fixed": -1, "adaptive": -1}

    for index, path in enumerate(paths):
        prediction = B["_path_prediction"](record, prior, path, segments)[eval_rows].astype(np.float64)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{well}: non-finite candidate {index}")
        move = prediction - parent
        raw_error = prediction - truth
        fixed_error = parent_error + FIXED_DOSE * move
        adaptive_error = parent_error + dose * move
        values = {
            "standalone": float(np.dot(raw_error, raw_error)),
            "fixed": float(np.dot(fixed_error, fixed_error)),
            "adaptive": float(np.dot(adaptive_error, adaptive_error)),
        }
        standalone_sse[index] = values["standalone"]
        fixed_sse[index] = values["fixed"]
        adaptive_sse[index] = values["adaptive"]
        for name, value in values.items():
            if value < best_values[name]:
                best_values[name] = value
                best_indices[name] = index
                best_predictions[name] = prediction.copy()

    standalone_prediction = best_predictions["standalone"]
    np.savez_compressed(
        output,
        signature=np.asarray(signature),
        dose=np.asarray([dose], np.float64),
        candidate_count=np.asarray([len(paths)], np.int32),
        standalone_index=np.asarray([best_indices["standalone"]], np.int32),
        fixed_index=np.asarray([best_indices["fixed"]], np.int32),
        adaptive_index=np.asarray([best_indices["adaptive"]], np.int32),
        standalone_rank_correlation=np.asarray([_spearman(standalone_sse, adaptive_sse)], np.float64),
        standalone_raw=standalone_prediction.astype(np.float32),
        standalone_blend=(parent + dose * (standalone_prediction - parent)).astype(np.float32),
        fixed_oracle_blend=(parent + FIXED_DOSE * (best_predictions["fixed"] - parent)).astype(np.float32),
        adaptive_oracle_blend=(parent + dose * (best_predictions["adaptive"] - parent)).astype(np.float32),
    )
    return "built"


reports = []
for number, well in enumerate(wells, 1):
    try:
        status = _build_well(well)
        reports.append({"well": well, "status": status, "reason": ""})
    except Exception as exc:
        reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 10 == 0 or number == len(wells):
        failed = sum(row["status"] == "failed" for row in reports)
        print(
            f"V102 Fold3 oracle {number}/{len(wells)} failures={failed} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

report_frame = pd.DataFrame(reports)
report_frame.to_csv(OUT_DIR / "build_report.csv", index=False)
failed = report_frame[report_frame["status"].eq("failed")]
if len(failed):
    raise RuntimeError(f"{len(failed)} V102 well failures: {failed.head().to_dict('records')}")

parts = []
audit_rows = []
for well in wells:
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    with np.load(_cache_path(well), allow_pickle=False) as data:
        for column in (
            "standalone_raw", "standalone_blend", "fixed_oracle_blend", "adaptive_oracle_blend"
        ):
            local[column] = data[column].astype(np.float32)
        audit_rows.append({
            "well": well,
            "split": "discovery" if well in discovery else "confirmation",
            "dose": float(data["dose"][0]),
            "candidate_count": int(data["candidate_count"][0]),
            "standalone_index": int(data["standalone_index"][0]),
            "fixed_index": int(data["fixed_index"][0]),
            "adaptive_index": int(data["adaptive_index"][0]),
            "standalone_equals_adaptive": int(data["standalone_index"][0] == data["adaptive_index"][0]),
            "standalone_rank_correlation": float(data["standalone_rank_correlation"][0]),
        })
    parent = local[PARENT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    dose = local["dose"].to_numpy(np.float64)
    local["v76"] = (parent + dose * (coarse - parent)).astype(np.float32)
    parts.append(local)

result = pd.concat(parts, ignore_index=True)
audit = pd.DataFrame(audit_rows)
audit.to_csv(OUT_DIR / "perwell_label_alignment.csv", index=False)

summary_rows = []
metrics = {}
for split, split_wells in {
    "fold3_discovery": discovery,
    "fold3_confirmation": set(wells).difference(discovery),
    "fold3_all": set(wells),
}.items():
    local = result[result["well"].isin(split_wells)].copy()
    metrics[split] = {}
    for endpoint, column in {
        "parent": PARENT,
        "v76": "v76",
        "standalone_oracle_raw": "standalone_raw",
        "standalone_label_oracle_blend": "standalone_blend",
        "fixed_blendaware_oracle": "fixed_oracle_blend",
        "adaptive_blendaware_oracle": "adaptive_oracle_blend",
    }.items():
        row = _metric(local, local[column], endpoint)
        row["stage"] = split
        summary_rows.append(row)
        metrics[split][endpoint] = row

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
print("\nV102 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)

alignment = audit.groupby("split", sort=False).agg(
    wells=("well", "count"),
    same_candidate_fraction=("standalone_equals_adaptive", "mean"),
    median_rank_correlation=("standalone_rank_correlation", "median"),
    mean_rank_correlation=("standalone_rank_correlation", "mean"),
    mean_dose=("dose", "mean"),
).reset_index()
alignment.to_csv(OUT_DIR / "label_alignment_summary.csv", index=False)
print("\nV102 LABEL ALIGNMENT")
print(alignment.to_string(index=False), flush=True)

def _gain(stage):
    local = metrics[stage]
    return float(
        local["standalone_label_oracle_blend"]["rmse"]
        - local["adaptive_blendaware_oracle"]["rmse"]
    )


discovery_gain = _gain("fold3_discovery")
confirmation_gain = _gain("fold3_confirmation")
confirmation_far_gain = float(
    metrics["fold3_confirmation"]["standalone_label_oracle_blend"]["far_gt2000_rmse"]
    - metrics["fold3_confirmation"]["adaptive_blendaware_oracle"]["far_gt2000_rmse"]
)
confirmation_alignment = alignment[alignment["split"].eq("confirmation")].iloc[0]
checks = {
    "discovery_label_oracle_gain_ge_0p10": discovery_gain >= 0.10,
    "confirmation_label_oracle_gain_ge_0p10": confirmation_gain >= 0.10,
    "confirmation_far_gain_nonnegative": confirmation_far_gain >= 0.0,
    "confirmation_candidate_change_fraction_ge_0p20": (
        1.0 - float(confirmation_alignment["same_candidate_fraction"]) >= 0.20
    ),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_discovery_confirmation_blendaware_label_oracle_gate",
    "selection_or_tuning_count": 0,
    "truth_usage": "oracle candidate selection and metrics only",
    "candidate_bank": "frozen V4 three-arm union",
    "current_training_target": "standalone complete-well path MSE",
    "proposed_training_target": "complete-well SSE after parent plus V76-dose candidate move",
    "protocol": {
        "fold": FOLD,
        "discovery_wells": len(discovery),
        "confirmation_wells": len(wells) - len(discovery),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "discovery_label_oracle_gain": discovery_gain,
    "confirmation_label_oracle_gain": confirmation_gain,
    "confirmation_far_gain": confirmation_far_gain,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> add blend-aware labels to Fold45 panels and train one matched V4 ranker"
        if strict_pass
        else "FAIL -> label mismatch ceiling is too small; retain V76 and close candidate-label branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV102 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
