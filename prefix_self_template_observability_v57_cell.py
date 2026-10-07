"""Direction-aware lateral-prefix GR self-template observability gateway.

The competition host notes that lateral GR has higher vertical resolution than
the assigned typewell and can correlate with itself when a candidate path
returns to TVT already traversed before prediction start. This truth-shape
audit asks only whether that evidence distinguishes the correct absolute
register from parallel aliases. It is diagnostic and not deployable: truth is
used solely to freeze translated candidate shapes.
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


EXPERIMENT = "prefix_self_template_observability_v57"
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
OFFSETS = np.arange(-64.0, 64.01, 2.0)
ZERO_INDEX = int(np.argmin(np.abs(OFFSETS)))
SEGMENT_FT = 128.0
MIN_POINTS = 32
MIN_COVERAGE = 0.70
MAX_PREFIX_NEAREST_FT = 2.0
DOSES = (0.0, 0.1, 0.3, 1.0, 3.0)
SEED = 57027


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType("raw_stochastic_v30_definitions_for_v57")
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


def _prefix_template(record, rolled=False):
    known = record["known"]
    frame = pd.DataFrame(
        {
            "tvt": record["tvt_input"][known],
            "gr": record["horizontal_channels"][0, known],
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    # Half-foot bins retain much more vertical detail than the assigned
    # typewell while making repeated visits to the same TVT robust.
    frame["bin"] = np.round(frame["tvt"] * 2.0) / 2.0
    frame = frame.groupby("bin", as_index=False)["gr"].median().sort_values("bin")
    if len(frame) < 64:
        raise RuntimeError(f"{record['well']}: only {len(frame)} prefix TVT bins")
    axis = frame["bin"].to_numpy(np.float64)
    values = frame["gr"].to_numpy(np.float64)
    if rolled:
        values = np.roll(values, max(len(values) // 5, 1))
    return axis, values


def _nearest_distance(axis, query):
    right = np.searchsorted(axis, query, side="left")
    left = np.clip(right - 1, 0, len(axis) - 1)
    right = np.clip(right, 0, len(axis) - 1)
    return np.minimum(np.abs(query - axis[left]), np.abs(query - axis[right]))


def _cauchy_cost(observed, reference, scale):
    residual = (observed - reference) / max(float(scale), 1.0)
    return float(np.mean(np.log1p(residual * residual)))


def _record_rows(record, rolled=False):
    prefix_axis, prefix_gr = _prefix_template(record, rolled=rolled)
    evaluation = record["evaluation"]
    eval_md = record["md"][evaluation]
    eval_tvt = record["truth"][evaluation]
    eval_gr = record["horizontal_channels"][0, evaluation]
    centres = np.arange(eval_md[0] + SEGMENT_FT / 2, eval_md[-1] + 1e-6, SEGMENT_FT)
    output = []
    alias_indices = [int(np.argmin(np.abs(OFFSETS - value))) for value in (-24.0, 24.0)]
    for centre in centres:
        rows = np.flatnonzero(np.abs(eval_md - centre) <= SEGMENT_FT / 2)
        if len(rows) < MIN_POINTS:
            continue
        if len(rows) > 256:
            rows = rows[np.linspace(0, len(rows) - 1, 256).round().astype(int)]
        observed = eval_gr[rows]
        truth_tvt = eval_tvt[rows]
        type_cost = np.full(len(OFFSETS), np.nan, np.float64)
        self_cost = np.full(len(OFFSETS), np.nan, np.float64)
        coverage = np.zeros(len(OFFSETS), np.float64)
        for index, offset in enumerate(OFFSETS):
            candidate = truth_tvt + offset
            tw_valid = (candidate >= record["tw_tvt"][0]) & (candidate <= record["tw_tvt"][-1])
            if tw_valid.sum() >= MIN_POINTS:
                reference = np.interp(candidate[tw_valid], record["tw_tvt"], record["tw_channels"][0])
                type_cost[index] = _cauchy_cost(
                    observed[tw_valid], reference + record["prefix_shift"], record["gamma"]
                )
            prefix_valid = (
                (candidate >= prefix_axis[0])
                & (candidate <= prefix_axis[-1])
                & (_nearest_distance(prefix_axis, candidate) <= MAX_PREFIX_NEAREST_FT)
            )
            coverage[index] = float(prefix_valid.mean())
            if prefix_valid.sum() >= MIN_POINTS and coverage[index] >= MIN_COVERAGE:
                reference = np.interp(candidate[prefix_valid], prefix_axis, prefix_gr)
                self_cost[index] = _cauchy_cost(observed[prefix_valid], reference, record["gamma"])
        required = [ZERO_INDEX, *alias_indices]
        if not np.isfinite(self_cost[required]).all() or not np.isfinite(type_cost[required]).all():
            continue
        output.append(
            {
                "well": record["well"],
                "centre_md": float(centre),
                "md_since": float(centre - record["anchor_md"]),
                "negative_tvt_direction": bool(np.median(truth_tvt) < record["anchor_tvt"]),
                "truth_self_coverage": float(coverage[ZERO_INDEX]),
                "type_cost": type_cost,
                "self_cost": self_cost,
            }
        )
    return output, int(len(centres))


def _build(records, label, rolled=False):
    rows, requested, failures = [], 0, []
    for number, record in enumerate(records, 1):
        try:
            local, count = _record_rows(record, rolled=rolled)
            rows.extend(local)
            requested += count
        except Exception as exc:
            failures.append({"well": record["well"], "reason": repr(exc)})
        if number % 25 == 0 or number == len(records):
            print(
                f"V57 {label} {number}/{len(records)} segments={len(rows)} failures={len(failures)}",
                flush=True,
            )
    pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
    return rows, requested


def _score(rows, dose, variant):
    reports = []
    alias_indices = [int(np.argmin(np.abs(OFFSETS - value))) for value in (-24.0, 24.0)]
    for row in rows:
        cost = row["type_cost"] + float(dose) * row["self_cost"]
        available = np.flatnonzero(np.isfinite(cost))
        if ZERO_INDEX not in available:
            continue
        order = available[np.argsort(cost[available], kind="stable")]
        rank = int(np.flatnonzero(order == ZERO_INDEX)[0]) + 1
        selected = int(order[0])
        reports.append(
            {
                "well": row["well"],
                "variant": variant,
                "dose": float(dose),
                "md_since": row["md_since"],
                "negative_tvt_direction": row["negative_tvt_direction"],
                "truth_self_coverage": row["truth_self_coverage"],
                "truth_rank": rank,
                "top1_within5": bool(abs(OFFSETS[selected]) <= 5.0),
                "top8_contains_truth": bool(rank <= 8),
                "alias24_win": bool(cost[ZERO_INDEX] < min(cost[index] for index in alias_indices)),
                "selected_offset": float(OFFSETS[selected]),
            }
        )
    return pd.DataFrame(reports)


def _summary(frame, requested):
    if frame.empty:
        raise RuntimeError("No usable prefix-self-template segments")
    deep = frame["md_since"].gt(2000.0)
    negative = frame["negative_tvt_direction"]
    return {
        "segments": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "requested_segments": int(requested),
        "usable_fraction": float(len(frame) / max(requested, 1)),
        "negative_direction_fraction": float(negative.mean()),
        "mean_truth_coverage": float(frame["truth_self_coverage"].mean()),
        "median_truth_rank": float(frame["truth_rank"].median()),
        "top1_within5": float(frame["top1_within5"].mean()),
        "top8_contains_truth": float(frame["top8_contains_truth"].mean()),
        "alias24_win_rate": float(frame["alias24_win"].mean()),
        "selected_offset_rmse": float(np.sqrt(np.mean(frame["selected_offset"] ** 2))),
        "deep_alias24_win_rate": float(frame.loc[deep, "alias24_win"].mean()) if deep.any() else np.nan,
        "negative_alias24_win_rate": (
            float(frame.loc[negative, "alias24_win"].mean()) if negative.any() else np.nan
        ),
    }


V = _bootstrap(SOURCE)
V.OUT_DIR = OUT_DIR
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
    raise FileNotFoundError(
        "Missing fold metadata. Expected V30's resolved FOLD_PATH or "
        f"{OOF_ROOT / 'raw_stochastic_alignment_standalone_v30' / 'folds.csv'}"
    )
folds = pd.read_csv(fold_path, dtype={"well": str})[["well", "fold"]]
if folds["well"].duplicated().any() or not {2, 3}.issubset(set(folds["fold"])):
    raise RuntimeError(f"Invalid fold metadata: {fold_path}")
fold3_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].astype(str))
fold3_records = [V._load_record(well) for well in fold3_wells]
fold3_records.sort(
    key=lambda record: hashlib.sha256(f"v57:{record['well']}".encode("ascii")).digest()
)
discovery_records = fold3_records[:DISCOVERY_WELLS]
confirmation_records = fold3_records[DISCOVERY_WELLS:]

discovery_rows, discovery_requested = _build(discovery_records, "discovery")
scan_rows = []
for dose in DOSES:
    frame = _score(discovery_rows, dose, f"aligned_dose{dose:g}")
    scan_rows.append({"dose": dose, **_summary(frame, discovery_requested)})
scan = pd.DataFrame(scan_rows).sort_values(
    ["alias24_win_rate", "top1_within5", "median_truth_rank"],
    ascending=[False, False, True],
    kind="stable",
)
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
print("\nV57 DISCOVERY SCAN", flush=True)
print(scan.to_string(index=False), flush=True)
selected_dose = float(scan.iloc[0]["dose"])

confirmation_rows, confirmation_requested = _build(confirmation_records, "confirmation")
control_rows, control_requested = _build(confirmation_records, "confirmation_rolled", rolled=True)
baseline_frame = _score(confirmation_rows, 0.0, "typewell_only")
aligned_frame = _score(confirmation_rows, selected_dose, "aligned_prefix")
control_frame = _score(control_rows, selected_dose, "rolled_prefix")
baseline = _summary(baseline_frame, confirmation_requested)
aligned = _summary(aligned_frame, confirmation_requested)
control = _summary(control_frame, control_requested)
confirmation_summary = pd.DataFrame(
    [
        {"variant": "typewell_only", **baseline},
        {"variant": "aligned_prefix", **aligned},
        {"variant": "rolled_prefix", **control},
    ]
)
confirmation_summary.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
aligned_frame.to_csv(OUT_DIR / "confirmation_segments.csv", index=False)
print("\nV57 CONFIRMATION", flush=True)
print(confirmation_summary.to_string(index=False), flush=True)

checks = {
    "usable_fraction_ge_0p05": bool(aligned["usable_fraction"] >= 0.05),
    "alias24_gain_ge_0p05": bool(aligned["alias24_win_rate"] - baseline["alias24_win_rate"] >= 0.05),
    "top1_gain_ge_0p03": bool(aligned["top1_within5"] - baseline["top1_within5"] >= 0.03),
    "aligned_alias_gain_vs_rolled_ge_0p03": bool(
        aligned["alias24_win_rate"] - control["alias24_win_rate"] >= 0.03
    ),
    "negative_alias_gain_positive": bool(
        aligned["negative_alias24_win_rate"] > baseline["negative_alias24_win_rate"]
    ),
}
mechanism_pass = bool(all(checks.values()))
application = None
if mechanism_pass:
    fold2_records = [V._load_record(well) for well in fold2_wells]
    fold2_rows, fold2_requested = _build(fold2_records, "fold2")
    fold2_control_rows, fold2_control_requested = _build(fold2_records, "fold2_rolled", rolled=True)
    fold2_baseline = _summary(_score(fold2_rows, 0.0, "typewell_only"), fold2_requested)
    fold2_aligned = _summary(
        _score(fold2_rows, selected_dose, "aligned_prefix"), fold2_requested
    )
    fold2_control = _summary(
        _score(fold2_control_rows, selected_dose, "rolled_prefix"), fold2_control_requested
    )
    application = {
        "baseline": fold2_baseline,
        "aligned": fold2_aligned,
        "rolled_control": fold2_control,
        "checks": {
            "alias_gain_positive": bool(
                fold2_aligned["alias24_win_rate"] > fold2_baseline["alias24_win_rate"]
            ),
            "aligned_beats_rolled": bool(
                fold2_aligned["alias24_win_rate"] > fold2_control["alias24_win_rate"]
            ),
        },
    }

strict_pass = bool(
    mechanism_pass and application is not None and all(application["checks"].values())
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_discovery_confirmation_Fold2_single_application",
    "deployable": False,
    "truth_usage": "truth defines translated candidate shape only",
    "legal_evidence": "same well visible-prefix GR plus assigned typewell and future horizontal GR",
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_records),
        "confirmation_wells": len(confirmation_records),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "candidate_offsets_ft": [float(OFFSETS[0]), float(OFFSETS[-1]), 2.0],
    "segment_ft": SEGMENT_FT,
    "selected_self_template_dose": selected_dose,
    "confirmation": {
        "typewell_only": baseline,
        "aligned_prefix": aligned,
        "rolled_prefix": control,
    },
    "checks": checks,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> add direction-aware prefix-self likelihood to a deployable candidate decoder"
        if strict_pass
        else "FAIL -> visible-prefix self correlation does not resolve the parallel alias at useful coverage"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT", json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
