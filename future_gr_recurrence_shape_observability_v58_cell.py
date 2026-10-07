"""Future lateral-GR recurrence observability gate for path shape.

This diagnostic does not try to resolve a constant TVT translation.  Such a
translation preserves all same-TVT relations.  Instead, truth-shaped paths are
anchored at prediction start and given a range of accumulating toe drifts.  We
ask whether non-local visits to the same candidate TVT have similar lateral GR
and whether that evidence identifies the zero-drift path beyond typewell GR.

Truth is used only to construct and score the candidate family, so this is an
investment gate rather than a deployable decoder.
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


EXPERIMENT = "future_gr_recurrence_shape_observability_v58"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = next(
    (
        path
        for path in (
            ROOT / "raw_stochastic_alignment_standalone_v30_cell.py",
            Path.cwd() / "raw_stochastic_alignment_standalone_v30_cell.py",
        )
        if path.is_file()
    ),
    None,
)
if SOURCE is None:
    raise FileNotFoundError("Upload raw_stochastic_alignment_standalone_v30_cell.py")

SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
ENDPOINT_OFFSETS = np.arange(-64.0, 64.01, 4.0)
ZERO_INDEX = int(np.argmin(np.abs(ENDPOINT_OFFSETS)))
BIN_WIDTHS_FT = (0.5, 1.0, 2.0)
MIN_SEPARATIONS_FT = (128.0, 256.0)
DOSES = (0.0, 0.1, 0.3, 1.0, 3.0)
ROW_STRIDE = 2
MIN_VISIT_ROWS = 2
MIN_RECURRENCE_PAIRS = 8
VISIT_BREAK_FT = 12.0


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType("raw_stochastic_v30_definitions_for_v58")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _robust_standardise(values):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, 10.0)
    centre = float(np.median(values[finite]))
    scale = float(1.4826 * np.median(np.abs(values[finite] - centre)))
    scale = max(scale, 1e-4)
    ceiling = float(np.max((values[finite] - centre) / scale) + 3.0)
    output = np.full_like(values, ceiling)
    output[finite] = (values[finite] - centre) / scale
    return output


def _recurrence_cost(md, candidate_tvt, gr, gamma, bin_width, min_separation):
    valid = np.isfinite(md) & np.isfinite(candidate_tvt) & np.isfinite(gr)
    md = md[valid]
    tvt = candidate_tvt[valid]
    gr = gr[valid]
    if len(md) < 64:
        return np.nan, 0
    labels = np.floor(tvt / float(bin_width) + 0.5).astype(np.int64)
    differences = []
    for label in np.unique(labels):
        local = np.flatnonzero(labels == label)
        if len(local) < 2 * MIN_VISIT_ROWS:
            continue
        split = np.flatnonzero(np.diff(md[local]) > VISIT_BREAK_FT) + 1
        visits = np.split(local, split)
        visits = [visit for visit in visits if len(visit) >= MIN_VISIT_ROWS]
        if len(visits) < 2:
            continue
        visit_md = np.asarray([np.median(md[visit]) for visit in visits])
        visit_gr = np.asarray([np.median(gr[visit]) for visit in visits])
        for right in range(1, len(visits)):
            left_candidates = np.flatnonzero(
                visit_md[right] - visit_md[:right] >= float(min_separation)
            )
            if not len(left_candidates):
                continue
            left = int(left_candidates[-1])
            differences.append(float(visit_gr[right] - visit_gr[left]))
    if len(differences) < MIN_RECURRENCE_PAIRS:
        return np.nan, len(differences)
    residual = np.asarray(differences, np.float64) / max(float(gamma), 5.0)
    return float(np.mean(np.log1p(residual * residual))), len(differences)


def _typewell_cost(record, rows, candidate_tvt):
    observed = record["horizontal_channels"][0, rows]
    inside = (
        np.isfinite(observed)
        & (candidate_tvt >= record["tw_tvt"][0])
        & (candidate_tvt <= record["tw_tvt"][-1])
    )
    if inside.sum() < 64:
        return np.nan
    reference = np.interp(
        candidate_tvt[inside], record["tw_tvt"], record["tw_channels"][0]
    ) + record["prefix_shift"]
    residual = (observed[inside] - reference) / max(float(record["gamma"]), 5.0)
    return float(np.mean(np.log1p(residual * residual)))


def _well_cache(record, bin_width, min_separation, rolled=False):
    rows = record["evaluation"][::ROW_STRIDE]
    md = record["md"][rows]
    truth = record["truth"][rows]
    gr = record["horizontal_channels"][0, rows].copy()
    if rolled:
        gr = np.roll(gr, max(len(gr) // 5, 1))
    distance = md - record["anchor_md"]
    maximum_distance = float(max(distance[-1], 1.0))
    progress = np.clip(distance / maximum_distance, 0.0, 1.0)
    type_cost = np.full(len(ENDPOINT_OFFSETS), np.nan, np.float64)
    recurrence_cost = np.full(len(ENDPOINT_OFFSETS), np.nan, np.float64)
    recurrence_pairs = np.zeros(len(ENDPOINT_OFFSETS), np.int32)
    for index, endpoint_offset in enumerate(ENDPOINT_OFFSETS):
        candidate = truth + float(endpoint_offset) * progress
        type_cost[index] = _typewell_cost(record, rows, candidate)
        recurrence_cost[index], recurrence_pairs[index] = _recurrence_cost(
            md,
            candidate,
            gr,
            record["gamma"],
            bin_width,
            min_separation,
        )
    return {
        "well": record["well"],
        "rows": int(len(rows)),
        "far_rows": int(np.sum(distance > 2000.0)),
        "negative_direction": bool(truth[-1] < record["anchor_tvt"]),
        "progress_square_sum": float(np.sum(progress**2)),
        "far_progress_square_sum": float(np.sum(progress[distance > 2000.0] ** 2)),
        "type_cost": type_cost,
        "recurrence_cost": recurrence_cost,
        "truth_pairs": int(recurrence_pairs[ZERO_INDEX]),
    }


def _build(records, bin_width, min_separation, label, rolled=False):
    output, failures = [], []
    for number, record in enumerate(records, 1):
        try:
            output.append(_well_cache(record, bin_width, min_separation, rolled=rolled))
        except Exception as exc:
            failures.append({"well": record["well"], "reason": repr(exc)})
        if number % 25 == 0 or number == len(records):
            usable = sum(np.isfinite(item["recurrence_cost"][ZERO_INDEX]) for item in output)
            print(
                f"V58 {label} {number}/{len(records)} truth_recurrence={usable} failures={len(failures)}",
                flush=True,
            )
    pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
    return output


def _score(cache, dose, variant):
    reports = []
    for item in cache:
        if not np.isfinite(item["recurrence_cost"][ZERO_INDEX]):
            continue
        type_score = _robust_standardise(item["type_cost"])
        recurrence_score = _robust_standardise(item["recurrence_cost"])
        total = type_score + float(dose) * recurrence_score
        order = np.argsort(total, kind="stable")
        selected = int(order[0])
        rank = int(np.flatnonzero(order == ZERO_INDEX)[0]) + 1
        offset = float(ENDPOINT_OFFSETS[selected])
        reports.append(
            {
                "well": item["well"],
                "variant": variant,
                "dose": float(dose),
                "rows": item["rows"],
                "far_rows": item["far_rows"],
                "negative_direction": item["negative_direction"],
                "truth_pairs": item["truth_pairs"],
                "truth_rank": rank,
                "top1_within8": bool(abs(offset) <= 8.0),
                "top8_contains_truth": bool(rank <= 8),
                "selected_endpoint_offset": offset,
                "path_sse": offset**2 * item["progress_square_sum"],
                "far_path_sse": offset**2 * item["far_progress_square_sum"],
            }
        )
    return pd.DataFrame(reports)


def _summary(frame, requested_wells):
    if frame.empty:
        raise RuntimeError("No wells have observable future-GR recurrence")
    rows = int(frame["rows"].sum())
    far_rows = int(frame["far_rows"].sum())
    negative = frame["negative_direction"]
    endpoint_square = frame["selected_endpoint_offset"] ** 2
    return {
        "wells": int(frame["well"].nunique()),
        "requested_wells": int(requested_wells),
        "well_coverage": float(frame["well"].nunique() / max(requested_wells, 1)),
        "negative_direction_fraction": float(negative.mean()),
        "median_truth_pairs": float(frame["truth_pairs"].median()),
        "median_truth_rank": float(frame["truth_rank"].median()),
        "top1_within8": float(frame["top1_within8"].mean()),
        "top8_contains_truth": float(frame["top8_contains_truth"].mean()),
        "endpoint_offset_rmse": float(np.sqrt(endpoint_square.mean())),
        "path_rmse": float(np.sqrt(frame["path_sse"].sum() / max(rows, 1))),
        "far_path_rmse": float(
            np.sqrt(frame["far_path_sse"].sum() / max(far_rows, 1))
        ),
        "negative_endpoint_rmse": (
            float(np.sqrt(np.mean(endpoint_square[negative]))) if negative.any() else np.nan
        ),
    }


V = _bootstrap(SOURCE)
started = time.time()
fold_candidates = [
    getattr(V, "FOLD_PATH", None),
    OOF_ROOT / "raw_stochastic_alignment_standalone_v30" / "folds.csv",
]
fold_path = next(
    (Path(path) for path in fold_candidates if path is not None and Path(path).is_file()),
    None,
)
if fold_path is None:
    raise FileNotFoundError("Missing V30 fold metadata")
folds = pd.read_csv(fold_path, dtype={"well": str})[["well", "fold"]]
if folds["well"].duplicated().any() or not {2, 3}.issubset(set(folds["fold"])):
    raise RuntimeError(f"Invalid fold metadata: {fold_path}")

fold3_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].astype(str))
fold3_records = [V._load_record(well) for well in fold3_wells]
fold3_records.sort(
    key=lambda record: hashlib.sha256(f"v58:{record['well']}".encode("ascii")).digest()
)
discovery_records = fold3_records[:DISCOVERY_WELLS]
confirmation_records = fold3_records[DISCOVERY_WELLS:]

scan_rows = []
for bin_width in BIN_WIDTHS_FT:
    for min_separation in MIN_SEPARATIONS_FT:
        cache = _build(
            discovery_records,
            bin_width,
            min_separation,
            f"discovery_b{bin_width:g}_g{min_separation:g}",
        )
        for dose in DOSES:
            frame = _score(cache, dose, f"dose{dose:g}")
            scan_rows.append(
                {
                    "bin_width_ft": bin_width,
                    "min_separation_ft": min_separation,
                    "dose": dose,
                    **_summary(frame, len(discovery_records)),
                }
            )
scan = pd.DataFrame(scan_rows).sort_values(
    ["path_rmse", "endpoint_offset_rmse", "median_truth_rank"], kind="stable"
)
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
print("\nV58 DISCOVERY SCAN", flush=True)
print(scan.head(30).to_string(index=False), flush=True)
selected = scan.iloc[0].to_dict()
bin_width = float(selected["bin_width_ft"])
min_separation = float(selected["min_separation_ft"])
dose = float(selected["dose"])

confirmation_cache = _build(
    confirmation_records, bin_width, min_separation, "confirmation"
)
control_cache = _build(
    confirmation_records, bin_width, min_separation, "confirmation_rolled", rolled=True
)
baseline_frame = _score(confirmation_cache, 0.0, "typewell_only")
aligned_frame = _score(confirmation_cache, dose, "aligned_recurrence")
control_frame = _score(control_cache, dose, "rolled_recurrence")
baseline = _summary(baseline_frame, len(confirmation_records))
aligned = _summary(aligned_frame, len(confirmation_records))
control = _summary(control_frame, len(confirmation_records))
confirmation = pd.DataFrame(
    [
        {"variant": "typewell_only", **baseline},
        {"variant": "aligned_recurrence", **aligned},
        {"variant": "rolled_recurrence", **control},
    ]
)
confirmation.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
aligned_frame.to_csv(OUT_DIR / "confirmation_bywell.csv", index=False)
print("\nV58 CONFIRMATION", flush=True)
print(confirmation.to_string(index=False), flush=True)

checks = {
    "well_coverage_ge_0p15": bool(aligned["well_coverage"] >= 0.15),
    "path_rmse_gain_ge_0p50": bool(baseline["path_rmse"] - aligned["path_rmse"] >= 0.50),
    "top1_gain_ge_0p03": bool(aligned["top1_within8"] - baseline["top1_within8"] >= 0.03),
    "aligned_path_gain_vs_rolled_ge_0p30": bool(
        control["path_rmse"] - aligned["path_rmse"] >= 0.30
    ),
    "far_path_gain_positive": bool(aligned["far_path_rmse"] < baseline["far_path_rmse"]),
    "negative_endpoint_gain_positive": bool(
        aligned["negative_endpoint_rmse"] < baseline["negative_endpoint_rmse"]
    ),
}
mechanism_pass = bool(all(checks.values()))
application = None
if mechanism_pass:
    fold2_records = [V._load_record(well) for well in fold2_wells]
    fold2_cache = _build(fold2_records, bin_width, min_separation, "fold2")
    fold2_control_cache = _build(
        fold2_records, bin_width, min_separation, "fold2_rolled", rolled=True
    )
    fold2_baseline = _summary(_score(fold2_cache, 0.0, "typewell_only"), len(fold2_records))
    fold2_aligned = _summary(
        _score(fold2_cache, dose, "aligned_recurrence"), len(fold2_records)
    )
    fold2_control = _summary(
        _score(fold2_control_cache, dose, "rolled_recurrence"), len(fold2_records)
    )
    application = {
        "baseline": fold2_baseline,
        "aligned": fold2_aligned,
        "rolled_control": fold2_control,
        "checks": {
            "path_gain_positive": bool(
                fold2_aligned["path_rmse"] < fold2_baseline["path_rmse"]
            ),
            "far_gain_positive": bool(
                fold2_aligned["far_path_rmse"] < fold2_baseline["far_path_rmse"]
            ),
            "aligned_beats_rolled": bool(
                fold2_aligned["path_rmse"] < fold2_control["path_rmse"]
            ),
        },
    }

strict_pass = bool(
    mechanism_pass and application is not None and all(application["checks"].values())
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_shape_discovery_confirmation_Fold2_single_application",
    "deployable": False,
    "truth_usage": "truth defines anchored toe-drift candidate shapes and held-out metrics",
    "legal_evidence": "future lateral GR self-recurrence plus assigned typewell GR",
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_records),
        "confirmation_wells": len(confirmation_records),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "candidate_endpoint_offsets_ft": [
        float(ENDPOINT_OFFSETS[0]),
        float(ENDPOINT_OFFSETS[-1]),
        4.0,
    ],
    "selected_recipe": {
        "bin_width_ft": bin_width,
        "min_separation_ft": min_separation,
        "recurrence_dose": dose,
    },
    "confirmation": {
        "typewell_only": baseline,
        "aligned_recurrence": aligned,
        "rolled_recurrence": control,
    },
    "checks": checks,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> add lateral-GR recurrence energy to the deployable path bank reranker"
        if strict_pass
        else "FAIL -> future lateral-GR recurrence does not identify useful path-shape corrections"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
