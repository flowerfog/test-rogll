"""V145: complete five-fold blend-aware candidate-support oracle audit.

This is a non-deployable investment gate.  For each physical well, the frozen
honest ranker retrieves up to 64 unique candidate trajectories.  Every
candidate is consumed through the exact V144 final formula, and truth is used
only to select the lowest-SSE member inside each cumulative top-K support.
"""

from __future__ import annotations

import json
import sys
import time
import types
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "full5_blendaware_candidate_support_oracle_v145"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "rebuilt_panels"
for directory in (OUT_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V112_SOURCE = ROOT / "rawdata_lattice_topk_barycenter_fold2_v112_cell.py"
V96_SOURCE = ROOT / "rawdata_lattice_v76_full773_training_v96_cell.py"
V144_PATH = (
    OOF_ROOT / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet"
)
V96_DIR = OOF_ROOT / "rawdata_lattice_v76_full773_training_v96"
V96_PRIOR = V96_DIR / "prior123.joblib"
V96_PANEL_DIR = V96_DIR / "prior123_panels"
V96_MODEL_DIR = V96_DIR / "coarse_models"
MODEL123_PATH = (
    OOF_ROOT / "rawdata_lattice_parentrelative_ranker_v104" / "models" / "fold45_base.txt"
)
PANEL_DIRS123 = (
    OOF_ROOT / "rawdata_lattice_consensus_center_fold1_v114" / "fold1_panels",
    OOF_ROOT / "rawdata_lattice_topk_barycenter_fold2_v112" / "fold2_panels",
    OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103" / "matched_panels",
)

FOLDS = (1, 2, 3, 4, 5)
TOPK_VALUES = (1, 4, 8, 16, 32, 64)
MAX_TOPK = max(TOPK_VALUES)
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
BASE = "consensus_center"
V144_ENDPOINT = "nested_sequence_coefficient"


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


def _bootstrap_v112():
    source = V112_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V112 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V112 definition marker missing from {V112_SOURCE}")
    module = types.ModuleType("candidate_support_v145_v112_definitions")
    module.__file__ = str(V112_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V112_SOURCE), "exec"), module.__dict__)
    return module


def _bootstrap_v96():
    source = V96_SOURCE.read_text(encoding="utf-8")
    marker = "for path in (V4_SOURCE, V66_PATH, V70_PATH, V76_BUNDLE, V76_OOF_WEIGHTS):"
    if marker not in source:
        raise RuntimeError(f"V96 definition marker missing from {V96_SOURCE}")
    module = types.ModuleType("candidate_support_v145_v96_definitions")
    module.__file__ = str(V96_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V96_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _deduplicated_order(paths, scores):
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    retained, seen = [], set()
    for index in order:
        key = np.ascontiguousarray(paths[index]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        retained.append(int(index))
    return np.asarray(retained, np.int32)


def _recover_dose(local):
    source = local[COARSE].to_numpy(np.float64) - local[PARENT].to_numpy(np.float64)
    target = local["exact_v76"].to_numpy(np.float64) - local[PARENT].to_numpy(np.float64)
    denominator = float(np.dot(source, source))
    dose = 0.0 if denominator <= 1e-12 else float(np.dot(source, target) / denominator)
    parity = float(np.max(np.abs(local[PARENT].to_numpy(np.float64) + dose * source - local["exact_v76"].to_numpy(np.float64))))
    if parity > 0.002 or not np.isfinite(dose):
        raise RuntimeError(f"V76 dose recovery parity={parity} dose={dose}")
    return dose, parity


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid endpoint {endpoint}")
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean())
    distance = frame["md_since"].to_numpy(np.float64)
    output = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }
    masks = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in masks.items():
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


for required in (V112_SOURCE, V96_SOURCE, V144_PATH, V96_PRIOR, MODEL123_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
for required in (V96_PANEL_DIR, V96_MODEL_DIR):
    if not required.is_dir():
        raise FileNotFoundError(required)

R = _bootstrap_v112()
N = R._bootstrap_v81()
Q123, V123, B123, prior45 = N.Q, N.V, N.B, N.prior
feature_names123 = list(N.feature_names)
model123 = lgb.Booster(model_file=str(MODEL123_PATH))

W = _bootstrap_v96()
Q4 = W._bootstrap_v4()
V45, B45 = Q4.V, Q4.B
prior123 = joblib.load(V96_PRIOR)
models45 = {
    fold: lgb.Booster(model_file=str(V96_MODEL_DIR / f"ranker_leave_fold{fold}_out.txt"))
    for fold in (4, 5)
}

frame = pd.read_parquet(V144_PATH).sort_values(
    ["fold", "well", "row_idx"], kind="stable"
).reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
required_columns = {
    "id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE,
    "exact_v76", BASE, V144_ENDPOINT,
}
missing = sorted(required_columns.difference(frame.columns))
if missing:
    raise RuntimeError(f"V144 OOF missing columns: {missing}")
if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame["id"].duplicated().any():
    raise RuntimeError(f"Invalid V144 coverage rows={len(frame)} wells={frame['well'].nunique()}")
fold_counts = frame.groupby("fold")["well"].nunique().to_dict()
if fold_counts != {1: 155, 2: 155, 3: 154, 4: 155, 5: 154}:
    raise RuntimeError(f"Unexpected canonical folds: {fold_counts}")


def _load_or_build123(well, record):
    for directory in PANEL_DIRS123:
        path = directory / f"{well}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            if "features" not in data.files or "paths" not in data.files:
                continue
            features = data["features"].astype(np.float32, copy=False)
            paths = data["paths"].astype(np.int16, copy=False)
        if len(features) == len(paths) and features.shape[1] == len(feature_names123):
            return features, paths, "reused"

    destination = CACHE_DIR / f"{well}.npz"
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
            paths = data["paths"].astype(np.int16, copy=False)
        if len(features) == len(paths) and features.shape[1] == len(feature_names123):
            return features, paths, "cached"

    paths, source_mask, arm_rank, arm_score_z, segments, _ = Q123._candidate_union(record, prior45)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        features, names = Q123._candidate_evidence(
            record, prior45, paths, segments, arm_rank, arm_score_z, source_mask
        )
    if list(names) != feature_names123:
        raise RuntimeError(f"{well}: Fold123 feature schema mismatch")
    np.savez_compressed(
        destination, features=features.astype(np.float32), paths=paths.astype(np.int16)
    )
    return features, paths, "built"


prediction_parts = {topk: [] for topk in TOPK_VALUES}
base_parts = []
v144_parts = []
matched_parts = []
report_rows = []
dose_parities = []

for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    local = local.sort_values("row_idx", kind="stable").copy()
    fold = int(local["fold"].iloc[0])
    truth = local[TRUTH].to_numpy(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    base = local[BASE].to_numpy(np.float64)
    dose, dose_parity = _recover_dose(local)
    dose_parities.append(dose_parity)
    predictions = {topk: base.copy() for topk in TOPK_VALUES}
    report = {
        "well": str(well), "fold": fold, "status": "fallback", "reason": "",
        "candidate_count": 0, "unique_candidates": 0, "dose": dose,
        "best_rank_top64": np.nan, "best_top64_rmse": np.nan,
        "base_rmse": float(np.sqrt(np.mean((base - truth) ** 2))),
    }
    try:
        if fold <= 3:
            record = V123["_load_record"](str(well))
            eval_rows = record["eval_rows"]
            features, paths, panel_status = _load_or_build123(str(well), record)
            scores = model123.predict(features)
            segments = V123["_segments"](record, float(Q123.SEGMENT_LENGTH_FT))
            path_prediction = B123["_path_prediction"]
            prior = prior45
        else:
            record = V45["_load_record"](str(well))
            eval_rows = record["eval_rows"]
            paths, source_mask, arm_rank, arm_score_z, segments, _ = Q4._candidate_union(
                record, prior123
            )
            panel_path = V96_PANEL_DIR / f"{well}.npz"
            if not panel_path.is_file():
                raise FileNotFoundError(panel_path)
            with np.load(panel_path, allow_pickle=False) as data:
                features = data["features"].astype(np.float32, copy=False)
            if len(features) != len(paths):
                raise RuntimeError(f"candidate count mismatch {len(features)}/{len(paths)}")
            scores = models45[fold].predict(features)
            panel_status = "reused"
            path_prediction = B45["_path_prediction"]
            prior = prior123

        if len(local) != len(eval_rows):
            raise RuntimeError(f"row mismatch {len(local)}/{len(eval_rows)}")
        order = _deduplicated_order(paths, scores)
        if len(order) < MAX_TOPK:
            raise RuntimeError(f"only {len(order)} unique candidates")
        taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
        candidate_sse = np.empty(MAX_TOPK, np.float64)
        candidate_predictions = []
        for rank, index in enumerate(order[:MAX_TOPK]):
            trajectory = path_prediction(record, prior, paths[index], segments)[eval_rows]
            trajectory = np.asarray(trajectory, np.float64)
            if len(trajectory) != len(local) or not np.isfinite(trajectory).all():
                raise RuntimeError(f"invalid candidate rank {rank + 1}")
            move = np.clip(dose * (trajectory - coarse), -CAP_FT, CAP_FT)
            candidate = exact + taper * move
            candidate_predictions.append(candidate)
            candidate_sse[rank] = float(np.sum(np.square(candidate - truth)))

        candidate_predictions = np.asarray(candidate_predictions, np.float64)
        cumulative_best = np.minimum.accumulate(candidate_sse)
        for topk in TOPK_VALUES:
            best_index = int(np.argmin(candidate_sse[:topk]))
            predictions[topk] = candidate_predictions[best_index]
            report[f"best_rank_top{topk}"] = best_index + 1
            report[f"oracle_rmse_top{topk}"] = float(
                np.sqrt(candidate_sse[best_index] / len(local))
            )
        best64 = int(np.argmin(candidate_sse))
        report.update({
            "status": panel_status,
            "candidate_count": int(len(paths)),
            "unique_candidates": int(len(order)),
            "best_rank_top64": best64 + 1,
            "best_top64_rmse": float(np.sqrt(cumulative_best[-1] / len(local))),
        })
    except Exception as exc:
        report["reason"] = repr(exc)

    matched_parts.append(local)
    base_parts.append(base)
    v144_parts.append(local[V144_ENDPOINT].to_numpy(np.float64))
    for topk in TOPK_VALUES:
        prediction_parts[topk].append(predictions[topk])
    report_rows.append(report)
    if number % 20 == 0 or number == frame["well"].nunique():
        fallbacks = sum(row["status"] == "fallback" for row in report_rows)
        print(
            f"V145 candidate oracle {number}/{frame['well'].nunique()} "
            f"fallbacks={fallbacks} elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

matched = pd.concat(matched_parts, ignore_index=True)
base_prediction = np.concatenate(base_parts)
v144_prediction = np.concatenate(v144_parts)
oracle_predictions = {topk: np.concatenate(parts) for topk, parts in prediction_parts.items()}
report = pd.DataFrame(report_rows)
report.to_csv(OUT_DIR / "bywell_oracle_support.csv", index=False)
fallbacks = int(report["status"].eq("fallback").sum())
if fallbacks > 5:
    failed = report.loc[report["status"].eq("fallback"), ["well", "fold", "reason"]]
    raise RuntimeError(f"Too many V145 fallbacks ({fallbacks}): {failed.head().to_dict('records')}")

summary_rows = []
for fold in (0, *FOLDS):
    mask = np.ones(len(matched), bool) if fold == 0 else matched["fold"].to_numpy() == fold
    local = matched.loc[mask]
    summary_rows.append({"fold": fold, **_metric(local, base_prediction[mask], BASE)})
    summary_rows.append({"fold": fold, **_metric(local, v144_prediction[mask], V144_ENDPOINT)})
    for topk in TOPK_VALUES:
        summary_rows.append({
            "fold": fold,
            **_metric(local, oracle_predictions[topk][mask], f"blendaware_oracle_top{topk}"),
        })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

pooled = summary[summary["fold"].eq(0)].set_index("endpoint")
base_rmse = float(pooled.loc[BASE, "rmse"])
v144_rmse = float(pooled.loc[V144_ENDPOINT, "rmse"])
oracle_values = {
    str(topk): {
        "rmse": float(pooled.loc[f"blendaware_oracle_top{topk}", "rmse"]),
        "gain_vs_consensus": float(
            base_rmse - pooled.loc[f"blendaware_oracle_top{topk}", "rmse"]
        ),
        "gain_vs_v144": float(
            v144_rmse - pooled.loc[f"blendaware_oracle_top{topk}", "rmse"]
        ),
        "far_rmse": float(pooled.loc[f"blendaware_oracle_top{topk}", "far_gt2000_rmse"]),
        "median_best_rank": float(report[f"best_rank_top{topk}"].median()),
    }
    for topk in TOPK_VALUES
}
top64_fold_rmse = {
    str(fold): float(
        summary.loc[
            summary["fold"].eq(fold)
            & summary["endpoint"].eq("blendaware_oracle_top64"),
            "rmse",
        ].iloc[0]
    )
    for fold in FOLDS
}
checks = {
    "coverage_exact": len(matched) == 3_783_989 and matched["well"].nunique() == 773,
    "fallbacks_le_5": fallbacks <= 5,
    "dose_reconstruction_max_gap_le_0p002": max(dose_parities) <= 0.002,
    "top64_oracle_rmse_lt_6": oracle_values["64"]["rmse"] < 6.0,
    "top64_oracle_each_fold_lt_6p5": all(value < 6.5 for value in top64_fold_rmse.values()),
    "top32_oracle_gain_vs_v144_ge_0p50": oracle_values["32"]["gain_vs_v144"] >= 0.50,
}
investment_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_773well_fivefold_blendaware_candidate_support_oracle",
    "deployable": False,
    "selection_or_training_count": 0,
    "truth_usage": "selects the lowest final-SSE candidate inside each cumulative top-K support",
    "candidate_protocol": {
        "fold123": "prior45 plus frozen Fold45 ranker",
        "fold45": "prior123 plus leave-one-fold-out rankers",
        "deduplication": "exact path bytes",
        "topk": list(TOPK_VALUES),
    },
    "final_consumption_formula": (
        "exact_v76 + smoothstep(md_since,1000,1250) * "
        "clip(v76_dose*(candidate-coarse),-6,6)"
    ),
    "coverage": {
        "rows": len(matched), "wells": int(matched["well"].nunique()),
        "fallbacks": fallbacks, "fold_wells": fold_counts,
    },
    "references": {"consensus_rmse": base_rmse, "v144_rmse": v144_rmse},
    "oracle_support": oracle_values,
    "top64_fold_rmse": top64_fold_rmse,
    "checks": checks,
    "investment_pass": investment_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> current bank has sub-6 capacity; build a new self-supervised support selector"
        if investment_pass else
        "FAIL -> current ranked support cannot reach sub-6 consistently; replace candidate generation"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV145 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV145 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
