"""Repair the absolute-state support of the raw stochastic endpoint.

V31 showed a low median state oracle but a heavy pooled tail caused by paths
leaving the +/-128 ft offset lattice.  This experiment changes only lattice
support.  Fold 3 selects the narrowest support satisfying pre-registered
coverage and oracle gates; Fold 2 is then a single confirmation application.

No historical prediction, neighbour, CP, Gold, or spatial feature is read.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "raw_stochastic_alignment_support_repair_v32"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V30_VERDICT = V30_DIR / "verdict.json"
SOURCE_CANDIDATES = (
    ROOT / "raw_stochastic_alignment_standalone_v30_cell.py",
    Path.cwd() / "raw_stochastic_alignment_standalone_v30_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload raw_stochastic_alignment_standalone_v30_cell.py")
if not V30_VERDICT.exists():
    raise FileNotFoundError(V30_VERDICT)

HALF_WIDTHS_FT = (128.0, 160.0, 192.0, 224.0, 256.0, 320.0)
STATE_STEP_FT = 4.0
MIN_COVERAGE = 0.995
MAX_ORACLE_RMSE = 3.0


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_stochastic_v30_support_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
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


V = _bootstrap(SOURCE_PATH)


def _records(wells, label):
    output = []
    failures = []
    for number, well in enumerate(wells, 1):
        try:
            output.append(V._load_record(well))
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"V32 load {label} {number}/{len(wells)} failures={len(failures)}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} {label} record failures: {failures[:5]}")
    return output


def _centre(record, prior, rows):
    distance = record["md"][rows] - record["anchor_md"]
    rate = float(np.clip(
        record["prefix_surface_rate"], prior["rate_low"], prior["rate_high"]
    ))
    return distance, record["anchor_surface"] + rate * distance


def _oracle_row(record, prior, half_width):
    node_rows = V._node_rows(record)
    distance, centre = _centre(record, prior, node_rows)
    truth_surface = record["truth"][node_rows] + record["z"][node_rows]
    offset = truth_surface - centre
    clipped = np.clip(offset, -half_width, half_width)
    quantised = np.round(clipped / STATE_STEP_FT) * STATE_STEP_FT
    node_surface = centre + quantised
    evaluation = record["evaluation"]
    prediction_surface = np.interp(
        record["md"][evaluation], record["md"][node_rows], node_surface
    )
    prediction = prediction_surface - record["z"][evaluation]
    error = prediction - record["truth"][evaluation]
    far = record["md"][evaluation] - record["anchor_md"] > 2000.0
    return {
        "well": record["well"],
        "half_width_ft": float(half_width),
        "rows": int(len(error)),
        "covered_nodes": int(np.sum(np.abs(offset) <= half_width + 1e-9)),
        "nodes": int(len(offset)),
        "max_abs_true_offset": float(np.max(np.abs(offset))),
        "sse": float(np.dot(error, error)),
        "far_rows": int(far.sum()),
        "far_sse": float(np.dot(error[far], error[far])) if far.any() else 0.0,
    }


def _oracle_summary(frame):
    output = []
    for width, local in frame.groupby("half_width_ft", sort=True):
        rows = int(local["rows"].sum())
        far_rows = int(local["far_rows"].sum())
        well_rmse = np.sqrt(local["sse"] / local["rows"])
        output.append({
            "half_width_ft": float(width),
            "states": int(round(2.0 * width / STATE_STEP_FT)) + 1,
            "rows": rows,
            "wells": int(len(local)),
            "node_coverage": float(local["covered_nodes"].sum() / local["nodes"].sum()),
            "oracle_rmse": float(np.sqrt(local["sse"].sum() / rows)),
            "oracle_median_well_rmse": float(np.median(well_rmse)),
            "oracle_p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "oracle_far_gt2000_rmse": (
                float(np.sqrt(local["far_sse"].sum() / far_rows)) if far_rows else np.nan
            ),
        })
    return pd.DataFrame(output)


def _set_lattice(half_width):
    offsets = np.arange(
        -float(half_width), float(half_width) + 0.5 * STATE_STEP_FT, STATE_STEP_FT,
        dtype=np.float64,
    )
    V.OFFSETS_FT = offsets
    V.ZERO_STATE = int(np.argmin(np.abs(offsets)))


def _decode_records(records, prior, config, half_width, label):
    _set_lattice(half_width)
    decoded = {}
    for number, record in enumerate(records, 1):
        decoded[record["well"]] = V._decode(
            record, prior, config, no_gr=False, sample_paths=False
        )
        if number % 25 == 0 or number == len(records):
            print(f"V32 decode {label} {number}/{len(records)}", flush=True)
    rows = V._metric_rows(records, decoded, label)
    return rows, V._summary(rows)


started = time.time()
v30 = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
prior = v30["outer_train_prior"]
config = dict(v30["selected_config"])
folds = pd.read_csv(V30_DIR / "folds.csv", dtype={"well": str})
fold3_wells = sorted(folds.loc[folds["fold"].eq(3), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(2), "well"].astype(str))
fold3_records = _records(fold3_wells, "fold3")
fold2_records = _records(fold2_wells, "fold2")

oracle_rows = []
for width in HALF_WIDTHS_FT:
    oracle_rows.extend(_oracle_row(record, prior, width) for record in fold3_records)
oracle_bywell = pd.DataFrame(oracle_rows)
oracle_summary = _oracle_summary(oracle_bywell)
oracle_bywell.to_csv(OUT_DIR / "fold3_oracle_bywell.csv", index=False)
oracle_summary.to_csv(OUT_DIR / "fold3_support_scan.csv", index=False)
print("\nV32 FOLD3 SUPPORT SCAN")
print(oracle_summary.to_string(index=False), flush=True)

eligible = oracle_summary[
    oracle_summary["node_coverage"].ge(MIN_COVERAGE)
    & oracle_summary["oracle_rmse"].le(MAX_ORACLE_RMSE)
]
if len(eligible):
    selected_width = float(eligible.sort_values("half_width_ft").iloc[0]["half_width_ft"])
    selection_reason = "narrowest_support_passing_coverage_and_oracle_gates"
else:
    selected_width = float(oracle_summary.sort_values(
        ["oracle_rmse", "half_width_ft"]
    ).iloc[0]["half_width_ft"])
    selection_reason = "no_support_passed; diagnostic_best_oracle_only"

widths_to_decode = [128.0]
if selected_width != 128.0:
    widths_to_decode.append(selected_width)
metric_parts = []
summary_parts = []
for width in widths_to_decode:
    for fold, records in ((3, fold3_records), (2, fold2_records)):
        label = f"fold{fold}_w{int(width)}"
        bywell, summary = _decode_records(records, prior, config, width, label)
        bywell["fold"] = fold
        bywell["half_width_ft"] = width
        summary["fold"] = fold
        summary["half_width_ft"] = width
        metric_parts.append(bywell)
        summary_parts.append(summary)

metric_frame = pd.concat(metric_parts, ignore_index=True)
summary_frame = pd.concat(summary_parts, ignore_index=True)
metric_frame.to_csv(OUT_DIR / "decoder_bywell.csv", index=False)
summary_frame.to_csv(OUT_DIR / "decoder_summary.csv", index=False)
print("\nV32 DECODER SUMMARY")
print(summary_frame.to_string(index=False), flush=True)

posterior = summary_frame[summary_frame["estimator"].eq("posterior_mean")]
base2 = posterior[(posterior["fold"].eq(2)) & (posterior["half_width_ft"].eq(128.0))].iloc[0]
selected2 = posterior[
    (posterior["fold"].eq(2)) & (posterior["half_width_ft"].eq(selected_width))
].iloc[0]
base3 = posterior[(posterior["fold"].eq(3)) & (posterior["half_width_ft"].eq(128.0))].iloc[0]
selected3 = posterior[
    (posterior["fold"].eq(3)) & (posterior["half_width_ft"].eq(selected_width))
].iloc[0]
selected_oracle = oracle_summary[
    oracle_summary["half_width_ft"].eq(selected_width)
].iloc[0].to_dict()

gates = {
    "fold3_state_coverage_ge_0p995": bool(selected_oracle["node_coverage"] >= MIN_COVERAGE),
    "fold3_state_oracle_rmse_le_3": bool(selected_oracle["oracle_rmse"] <= MAX_ORACLE_RMSE),
    "fold3_posterior_improves": bool(selected3["rmse"] < base3["rmse"]),
    "fold2_posterior_improves": bool(selected2["rmse"] < base2["rmse"]),
    "fold2_far_improves": bool(
        selected2["far_gt2000_rmse"] < base2["far_gt2000_rmse"]
    ),
}
if not gates["fold3_state_coverage_ge_0p995"] or not gates["fold3_state_oracle_rmse_le_3"]:
    next_action = "Replace offset-only state with joint slope/offset state; width alone cannot represent the tail"
elif gates["fold2_posterior_improves"]:
    next_action = "Retain repaired support and build v33 learned pairwise emission on the frozen stochastic decoder"
else:
    next_action = "State tail is repaired but aliases expand too; build v33 learned pairwise emission before any wider decoder"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_support_selection_Fold2_single_confirmation",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "frozen_v30_config": config,
    "support_scan_ft": list(HALF_WIDTHS_FT),
    "selection_gates": {"minimum_coverage": MIN_COVERAGE, "maximum_oracle_rmse": MAX_ORACLE_RMSE},
    "selected_half_width_ft": selected_width,
    "selection_reason": selection_reason,
    "selected_fold3_oracle": selected_oracle,
    "fold3_posterior_rmse": {"width128": float(base3["rmse"]), "selected": float(selected3["rmse"])},
    "fold2_posterior_rmse": {"width128": float(base2["rmse"]), "selected": float(selected2["rmse"])},
    "fold2_far_rmse": {
        "width128": float(base2["far_gt2000_rmse"]),
        "selected": float(selected2["far_gt2000_rmse"]),
    },
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": next_action,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
