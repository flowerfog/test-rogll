"""Cross-evidence audit: heavy-tail transition endpoint vs the 6.377 lineage.

V51 established a causal Student-t transition effect, but the standalone
lattice remained weak.  This cell asks the relevant ensemble question: does
that raw-only endpoint add independent error information to the deployed
CP8 -> tail-only Neighbor lineage?

Fold 4/5 fit only the raw transition prior.  Twenty-four deterministic Fold-3
wells select one fixed blend weight; the remaining Fold-3 wells confirm it.
Fold 2 is loaded only after all confirmation gates pass.  Fold 1 is untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
EXPERIMENT = "student_transition_6p377_cross_evidence_v52"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V51_SOURCE_CANDIDATES = (
    ROOT / "dip_fault_beam_expansion_attribution_v51_cell.py",
    Path.cwd() / "dip_fault_beam_expansion_attribution_v51_cell.py",
)
V51_SOURCE = next((path for path in V51_SOURCE_CANDIDATES if path.is_file()), None)
if V51_SOURCE is None:
    raise FileNotFoundError("Upload dip_fault_beam_expansion_attribution_v51_cell.py")

INCUMBENT_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
V51_VERDICT = OOF_ROOT / "dip_fault_beam_expansion_attribution_v51" / "verdict.json"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
INCUMBENT_COLUMN = "selected_neighbor_dose"
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
SEGMENT_LENGTH_FT = 128.0
WEIGHTS = np.round(np.arange(0.0, 0.301, 0.025), 3)
DISTANCE_EDGES = (-np.inf, 500.0, 1000.0, 2000.0, np.inf)
DISTANCE_LABELS = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")


def _bootstrap(path: Path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType("dip_fault_v51_cross_evidence_definitions")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _read_incumbent(fold: int) -> pd.DataFrame:
    if not INCUMBENT_PATH.is_file():
        raise FileNotFoundError(INCUMBENT_PATH)
    columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", INCUMBENT_COLUMN]
    frame = pd.read_parquet(INCUMBENT_PATH, columns=columns, filters=[("fold", "=", int(fold))])
    frame["well"] = frame["well"].astype(str)
    frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if frame.empty or frame["id"].duplicated().any():
        raise RuntimeError(f"Invalid incumbent Fold-{fold}: rows={len(frame)}")
    if frame[["tvt_true", "md_since", INCUMBENT_COLUMN]].isna().any().any():
        raise RuntimeError(f"Missing incumbent values in Fold-{fold}")
    return frame


def _fold_map(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    well_column = next(
        (name for name in ("well", "well_id", "Well") if name in frame), None
    )
    fold_column = next(
        (name for name in ("fold", "Fold", "outer_fold") if name in frame), None
    )
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify fold columns in {path}: {list(frame)}")
    result = dict(
        zip(frame[well_column].astype(str), frame[fold_column].astype(int))
    )
    if len(result) != 773 or set(result.values()) != {1, 2, 3, 4, 5}:
        raise RuntimeError(
            f"Unexpected fold metadata: wells={len(result)} folds={set(result.values())}"
        )
    return result


def _decode(record, prior, config, shuffled=False):
    scores, paths, segments = N._beam_bank(record, prior, config, shuffled)
    selected = int(np.argmin(scores))
    full = np.asarray(
        N.V2["_path_prediction"](record, prior, paths[selected], segments), np.float64
    )
    evaluation = np.asarray(record["eval_rows"], np.int64)
    return full[evaluation]


def _align_record(record, incumbent: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    local = incumbent.loc[incumbent["well"].eq(str(record["well"]))].copy()
    evaluation = np.asarray(record["eval_rows"], np.int64)
    if len(local) != len(evaluation) or len(prediction) != len(evaluation):
        raise RuntimeError(
            f"{record['well']}: incumbent/raw rows {len(local)}/{len(evaluation)}"
        )
    truth = np.asarray(record["truth"][evaluation], np.float64)
    gap = np.abs(local["tvt_true"].to_numpy(np.float64) - truth)
    maximum_gap = float(gap.max()) if len(gap) else 0.0
    if not np.isfinite(gap).all() or maximum_gap > 0.002:
        raise RuntimeError(
            f"{record['well']}: maximum truth alignment gap={maximum_gap}"
        )
    local["student_transition"] = np.asarray(prediction, np.float64)
    return local


def _decode_records(records, prior, config, incumbent, label):
    parts, failures = [], []
    for number, record in enumerate(records, 1):
        try:
            prediction = _decode(record, prior, config)
            parts.append(_align_record(record, incumbent, prediction))
        except Exception as exc:
            failures.append({"well": str(record["well"]), "reason": repr(exc)})
        if number % 12 == 0 or number == len(records):
            print(f"V52 decode {label} {number}/{len(records)} failures={len(failures)}", flush=True)
    pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
    if len(parts) < max(len(records) - 2, 1):
        raise RuntimeError(f"Only {len(parts)}/{len(records)} successful {label} records")
    return pd.concat(parts, ignore_index=True)


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    per_well = (
        pd.DataFrame({"well": frame["well"], "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    output = {
        "rows": int(len(frame)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p95_well_rmse": float(per_well.quantile(0.95)),
    }
    for left, right, label in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_LABELS):
        mask = (distance > left) & (distance <= right)
        output[f"{label}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return output


def _correlations(frame):
    truth = frame["tvt_true"].to_numpy(np.float64)
    incumbent_error = frame[INCUMBENT_COLUMN].to_numpy(np.float64) - truth
    student_error = frame["student_transition"].to_numpy(np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    raw = float(np.corrcoef(incumbent_error, student_error)[0, 1])
    incumbent_demeaned = incumbent_error.copy()
    student_demeaned = student_error.copy()
    within = {}
    for left, right, label in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_LABELS):
        mask = (distance > left) & (distance <= right)
        if mask.sum() < 3:
            within[label] = np.nan
            continue
        within[label] = float(np.corrcoef(incumbent_error[mask], student_error[mask])[0, 1])
        incumbent_demeaned[mask] -= incumbent_error[mask].mean()
        student_demeaned[mask] -= student_error[mask].mean()
    return {
        "signed_error_raw": raw,
        "signed_error_distance_strata_demeaned": float(
            np.corrcoef(incumbent_demeaned, student_demeaned)[0, 1]
        ),
        "within_distance_strata": within,
    }


def _blend(frame, weight):
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(np.float64)
    student = frame["student_transition"].to_numpy(np.float64)
    return incumbent + float(weight) * (student - incumbent)


def _shifted_control(frame, weight):
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(np.float64)
    student = frame["student_transition"].to_numpy(np.float64)
    move = student - incumbent
    shifted = np.empty_like(move)
    for _, indices in frame.groupby("well", sort=False).indices.items():
        local = np.asarray(indices, np.int64)
        shift = max(len(local) // 2, 1)
        shifted[local] = np.roll(move[local], shift)
    return incumbent + float(weight) * shifted


N = _bootstrap(V51_SOURCE)
started = time.time()
if not V51_VERDICT.is_file():
    raise FileNotFoundError(V51_VERDICT)
v51 = json.loads(V51_VERDICT.read_text(encoding="utf-8"))
selected_config = dict(v51["selected_config"])
if selected_config.get("family") != "student":
    raise RuntimeError(f"V51 did not freeze a Student endpoint: {selected_config}")

wells = [
    path.name.split("__", 1)[0]
    for path in sorted((N.DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
fold_by_well = _fold_map(FOLD_PATH)
missing_fold_wells = sorted(set(wells).difference(fold_by_well))
if missing_fold_wells:
    raise RuntimeError(f"Raw wells missing from fold metadata: {missing_fold_wells[:10]}")
prior_wells = [well for well in wells if fold_by_well.get(well) in PRIOR_FOLDS]
fold3_wells = [well for well in wells if fold_by_well.get(well) == SELECTION_FOLD]

prior_records, failures = [], []
for number, well in enumerate(prior_wells, 1):
    try:
        prior_records.append(N.V["_load_record"](well))
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(prior_wells):
        print(f"V52 prior records {number}/{len(prior_wells)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "prior_record_failures.csv", index=False)
prior = N.V["_fit_prior"](prior_records, SEGMENT_LENGTH_FT)

fold3_records = []
for well in fold3_wells:
    try:
        fold3_records.append(N.V["_load_record"](well))
    except Exception:
        pass
fold3_records = sorted(
    fold3_records,
    key=lambda record: hashlib.sha256(
        f"dip-fault-v51:{record['well']}".encode("ascii")
    ).digest(),
)
discovery_records = fold3_records[:DISCOVERY_WELLS]
confirmation_records = fold3_records[DISCOVERY_WELLS:]

incumbent3 = _read_incumbent(SELECTION_FOLD)
available3 = set(incumbent3["well"])
incumbent_fold_mismatch = sorted(
    well for well in available3 if fold_by_well.get(well) != SELECTION_FOLD
)
if incumbent_fold_mismatch:
    raise RuntimeError(
        "Incumbent Fold3 disagrees with fold-clean metadata: "
        f"{incumbent_fold_mismatch[:10]}"
    )
discovery_records = [record for record in discovery_records if record["well"] in available3]
confirmation_records = [record for record in confirmation_records if record["well"] in available3]
if len(discovery_records) < 22 or len(confirmation_records) < 120:
    raise RuntimeError(
        f"Insufficient matched Fold3 wells: {len(discovery_records)}/{len(confirmation_records)}"
    )

discovery = _decode_records(
    discovery_records, prior, selected_config, incumbent3, "fold3_discovery"
)
confirmation = _decode_records(
    confirmation_records, prior, selected_config, incumbent3, "fold3_confirmation"
)

scan_rows = []
for weight in WEIGHTS:
    result = _metrics(discovery, _blend(discovery, weight))
    scan_rows.append({"weight_student": float(weight), **result})
scan = pd.DataFrame(scan_rows).sort_values("rmse").reset_index(drop=True)
selected_weight = float(scan.iloc[0]["weight_student"])
scan.to_csv(OUT_DIR / "fold3_discovery_weight_scan.csv", index=False)

incumbent_metrics = _metrics(confirmation, confirmation[INCUMBENT_COLUMN])
student_metrics = _metrics(confirmation, confirmation["student_transition"])
blend_metrics = _metrics(confirmation, _blend(confirmation, selected_weight))
shuffle_metrics = _metrics(confirmation, _shifted_control(confirmation, selected_weight))
correlation = _correlations(confirmation)
confirmation_gain = incumbent_metrics["rmse"] - blend_metrics["rmse"]
far_gain = incumbent_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"]
shuffle_increment = shuffle_metrics["rmse"] - blend_metrics["rmse"]
confirmation_gates = {
    "selected_weight_nonzero": selected_weight > 0.0,
    "confirmation_gain_ge_0p05": confirmation_gain >= 0.05,
    "confirmation_far_gain_nonnegative": far_gain >= 0.0,
    "aligned_increment_vs_shift_ge_0p03": shuffle_increment >= 0.03,
    "detrended_error_correlation_le_0p80": (
        correlation["signed_error_distance_strata_demeaned"] <= 0.80
    ),
}
mechanism_pass = bool(all(confirmation_gates.values()))

summary_rows = []
for endpoint, metrics in (
    ("incumbent_6p377", incumbent_metrics),
    ("student_transition", student_metrics),
    ("fixed_blend", blend_metrics),
    ("shifted_move_control", shuffle_metrics),
):
    summary_rows.append({"stage": "fold3_confirmation", "endpoint": endpoint, **metrics})

application = None
application_frame = None
if mechanism_pass:
    incumbent2 = _read_incumbent(APPLICATION_FOLD)
    available2 = set(incumbent2["well"])
    fold2_records = []
    for well in wells:
        if fold_by_well.get(well) != APPLICATION_FOLD or well not in available2:
            continue
        try:
            fold2_records.append(N.V["_load_record"](well))
        except Exception:
            pass
    application_frame = _decode_records(
        fold2_records, prior, selected_config, incumbent2, "fold2_application"
    )
    base2 = _metrics(application_frame, application_frame[INCUMBENT_COLUMN])
    student2 = _metrics(application_frame, application_frame["student_transition"])
    blend2 = _metrics(application_frame, _blend(application_frame, selected_weight))
    shift2 = _metrics(application_frame, _shifted_control(application_frame, selected_weight))
    for endpoint, metrics in (
        ("incumbent_6p377", base2),
        ("student_transition", student2),
        ("fixed_blend", blend2),
        ("shifted_move_control", shift2),
    ):
        summary_rows.append({"stage": "fold2_application", "endpoint": endpoint, **metrics})
    application = {
        "incumbent": base2,
        "student": student2,
        "blend": blend2,
        "shifted_control": shift2,
        "correlation": _correlations(application_frame),
        "gain": base2["rmse"] - blend2["rmse"],
        "far_gain": base2["far_gt2000_rmse"] - blend2["far_gt2000_rmse"],
        "aligned_increment_vs_shift": shift2["rmse"] - blend2["rmse"],
    }

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
discovery.to_parquet(OUT_DIR / "fold3_discovery_endpoints.parquet", index=False)
confirmation.to_parquet(OUT_DIR / "fold3_confirmation_endpoints.parquet", index=False)
if application_frame is not None:
    application_frame.to_parquet(OUT_DIR / "fold2_application_endpoints.parquet", index=False)

strict_pass = bool(
    mechanism_pass
    and application is not None
    and application["gain"] >= 0.05
    and application["far_gain"] >= 0.0
    and application["aligned_increment_vs_shift"] >= 0.03
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_weight_selection_confirmation_Fold2_conditional_cross_evidence",
    "incumbent": str(INCUMBENT_PATH),
    "incumbent_column": INCUMBENT_COLUMN,
    "student_config": selected_config,
    "selected_weight_student": selected_weight,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "discovery_wells": len(discovery_records),
        "confirmation_wells": len(confirmation_records),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "confirmation": {
        "incumbent": incumbent_metrics,
        "student": student_metrics,
        "blend": blend_metrics,
        "shifted_control": shuffle_metrics,
        "correlation": correlation,
        "gain": confirmation_gain,
        "far_gain": far_gain,
        "aligned_increment_vs_shift": shuffle_increment,
    },
    "confirmation_gates": confirmation_gates,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain the Student transition as a fixed low-dose endpoint"
        if strict_pass
        else "FAIL -> causal transition signal is too correlated or too weak for the 6.377 lineage; close branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV52 DISCOVERY WEIGHT SCAN")
print(scan.to_string(index=False), flush=True)
print("\nV52 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV52 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
