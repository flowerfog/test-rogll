"""SEGN-min v4: frozen Fold-3 anchor-recovery observability audit.

V3 obtained low error when every chunk received its true anchor, but failed
under recursive rollout.  This cell isolates the missing mechanism without
training or touching Fold 2: can the frozen V3 cost map remove a known error
from a chunk anchor using the horizontal GR/typewell pair?

For each sampled Fold-3 chunk, the true anchor is shifted by
{-48, -24, -12, 0, 12, 24, 48} ft.  A successful absolute matcher must shift
its relative prediction by the opposite amount.  The primary recovery slope
is therefore the regression slope of predicted correction on -anchor error:

    1.0 = complete recovery, 0.0 = the model copies the wrong anchor.

Matched no-GR and shuffled-typewell controls are evaluated on the exact same
chunks and offsets.  Truth is used only to inject and score this diagnostic.
No optimizer is created and Fold 1/Fold 2 are never loaded.

Prerequisites in /root/autodl-tmp (or the current directory):
* segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py
* segn_min_scheduled_gate_fold2_v2_cell.py
* segn_min_refined_channels_fold2_v3_cell.py
* the V3 scheduled checkpoint under oof_exports, unless CHECKPOINT_V4 is set
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Bootstrap V3 definitions without executing its training/gateway tail.
# ---------------------------------------------------------------------------

V3_SOURCE_CANDIDATES = (
    Path("/root/autodl-tmp/segn_min_refined_channels_fold2_v3_cell.py"),
    Path.cwd() / "segn_min_refined_channels_fold2_v3_cell.py",
)
V3_SOURCE = next((path for path in V3_SOURCE_CANDIDATES if path.exists()), None)
if V3_SOURCE is None:
    raise FileNotFoundError(
        "Upload segn_min_refined_channels_fold2_v3_cell.py to /root/autodl-tmp."
    )

v3_source = V3_SOURCE.read_text(encoding="utf-8")
v3_marker = "\n# Execute V2's nested protocol"
if v3_marker not in v3_source:
    raise RuntimeError(f"Cannot find V3 execution marker in {V3_SOURCE}")
exec(compile(v3_source.split(v3_marker, 1)[0], str(V3_SOURCE), "exec"), globals())


EXPERIMENT_V4 = "segn_min_anchor_recovery_fold3_v4"
OUT_DIR = OOF_ROOT / EXPERIMENT_V4
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Override this in the notebook before running the cell if needed.
CHECKPOINT_V4 = globals().get("CHECKPOINT_V4", None)
CHECKPOINT_CANDIDATES_V4 = tuple(
    path
    for path in (
        Path(CHECKPOINT_V4) if CHECKPOINT_V4 else None,
        OOF_ROOT
        / "segn_min_refined_channels_fold2_v3"
        / "models"
        / "ab_scheduled45_to_3_best.pt",
        Path("/root/autodl-tmp/ab_scheduled45_to_3_best.pt"),
        Path.cwd() / "ab_scheduled45_to_3_best.pt",
    )
    if path is not None
)
CHECKPOINT_PATH_V4 = next(
    (path for path in CHECKPOINT_CANDIDATES_V4 if path.exists()), None
)
if CHECKPOINT_PATH_V4 is None:
    raise FileNotFoundError(
        "Cannot find the V3 scheduled checkpoint. Checked: "
        + repr([str(path) for path in CHECKPOINT_CANDIDATES_V4])
    )

INJECTED_OFFSETS_FT_V4 = np.asarray(
    [-48.0, -24.0, -12.0, 0.0, 12.0, 24.0, 48.0], np.float64
)
CONTROLS_V4 = ("full", "no_gr", "shuffled_typewell")
CHUNKS_PER_WELL_V4 = 4

# These are diagnostic gates, not deployment claims.
ZERO_OFFSET_RMSE_GATE_V4 = 4.0
RECOVERY_SLOPE_GATE_V4 = 0.50
RESIDUAL_RESPONSE_ABS_GATE_V4 = 0.50
CONTROL_SLOPE_DROP_GATE_V4 = 0.10
MIN_PAYLOAD_COVERAGE_V4 = 0.80


def _json_v4(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _select_chunk_starts_v4(record):
    """Select deployment-relevant starts after the known initial anchor."""

    starts = np.asarray(record.chunk_starts, np.int64)
    starts = starts[starts > 0]
    if not len(starts):
        return np.empty(0, np.int64)
    count = min(CHUNKS_PER_WELL_V4, len(starts))
    indices = np.unique(
        np.rint(np.linspace(0, len(starts) - 1, count)).astype(np.int64)
    )
    return starts[indices]


def _previous_start_v4(record, start):
    starts = np.asarray(record.chunk_starts, np.int64)
    prior = starts[starts < int(start)]
    return int(prior[-1]) if len(prior) else 0


def _distance_band_v4(distance):
    if distance <= 500.0:
        return "near_0_500"
    if distance <= 1000.0:
        return "mid_500_1000"
    if distance <= 2000.0:
        return "mid_1000_2000"
    return "far_gt2000"


def _ols_slope_v4(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or float(np.var(x)) <= 1e-12:
        return math.nan, math.nan
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _metric_row_v4(frame):
    residual = frame["residual"].to_numpy(np.float64)
    return {
        "rows": int(len(frame)),
        "chunks": int(frame[["well", "start", "offset_ft"]].drop_duplicates().shape[0]),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(residual, residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def _position_rmse_v4(frame, position_name):
    values = frame.loc[frame["position_name"].eq(position_name), "residual"].to_numpy(
        np.float64
    )
    return float(np.sqrt(np.mean(values**2))) if len(values) else math.nan


def _control_summary_v4(frame):
    nonzero = frame.loc[frame["offset_ft"].ne(0.0)]
    recovery_slope, recovery_intercept = _ols_slope_v4(
        -nonzero["offset_ft"], nonzero["predicted_correction"]
    )
    residual_slope, residual_intercept = _ols_slope_v4(
        nonzero["offset_ft"], nonzero["residual"]
    )
    zero = frame.loc[frame["offset_ft"].eq(0.0)]
    corrupted = frame.loc[frame["offset_ft"].ne(0.0)]
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "recovery_slope": recovery_slope,
        "recovery_intercept": recovery_intercept,
        "residual_response_slope": residual_slope,
        "residual_response_intercept": residual_intercept,
        "zero_offset_rmse": _metric_row_v4(zero)["rmse"],
        "corrupted_rmse": _metric_row_v4(corrupted)["rmse"],
        "corrupted_first_rmse": _position_rmse_v4(corrupted, "first"),
        "corrupted_rebase_rmse": _position_rmse_v4(corrupted, "rebase"),
        "corrupted_endpoint_rmse": _position_rmse_v4(corrupted, "endpoint"),
    }


@torch.no_grad()
def _run_anchor_recovery_v4(model, records, device):
    model.eval()
    node_output = {
        "well": [],
        "control": [],
        "offset_ft": [],
        "start": [],
        "distance": [],
        "distance_band": [],
        "position": [],
        "position_name": [],
        "residual": [],
        "predicted_correction": [],
    }
    chunk_rows = []
    requested_payloads = 0
    successful_payloads = 0
    requested_by_key = {}
    successful_by_key = {}

    shuffled_by_well = {
        record.well: records[(index + 37) % len(records)]
        for index, record in enumerate(records)
    }

    for record_number, record in enumerate(records, 1):
        for start_value in _select_chunk_starts_v4(record):
            start = int(start_value)
            start_row = int(record.node_rows[start])
            previous_start = _previous_start_v4(record, start)
            previous_row = int(record.node_rows[previous_start])
            md_delta = max(float(record.md[start_row] - record.md[previous_row]), 1.0)
            true_previous_delta = float(
                record.truth[start_row] - record.truth[previous_row]
            )
            previous_rate = true_previous_delta / md_delta
            distance = float(record.md[start_row] - record.md[record.anchor_index])
            distance_band = _distance_band_v4(distance)

            for control in CONTROLS_V4:
                shuffled = (
                    shuffled_by_well[record.well]
                    if control == "shuffled_typewell"
                    else None
                )
                for offset_ft in INJECTED_OFFSETS_FT_V4:
                    requested_payloads += 1
                    key = (control, float(offset_ft))
                    requested_by_key[key] = requested_by_key.get(key, 0) + 1
                    deployed_anchor = float(record.truth[start_row]) + float(offset_ft)
                    payload = _chunk_payload(
                        record,
                        start,
                        deployed_anchor=deployed_anchor,
                        previous_delta=true_previous_delta,
                        previous_rate=previous_rate,
                        control=control,
                        shuffled_typewell=shuffled,
                    )
                    if float(np.max(np.abs(payload["target_state"]))) > STATE_HALF_WIDTH_FT - 0.5:
                        continue
                    expected, _, _, _ = _predict_chunk_min(model, payload, device)
                    target_rows = payload["target_rows"]
                    true_delta = (
                        record.truth[target_rows].astype(np.float64)
                        - float(record.truth[start_row])
                    )
                    predicted_absolute = deployed_anchor + expected
                    truth = record.truth[target_rows].astype(np.float64)
                    residual = predicted_absolute - truth
                    predicted_correction = expected - true_delta
                    if not np.isfinite(residual).all():
                        continue
                    successful_payloads += 1
                    successful_by_key[key] = successful_by_key.get(key, 0) + 1

                    rebase_position = min(CHUNK_STRIDE_NODES, len(residual))
                    positions = np.arange(1, len(residual) + 1, dtype=np.int16)
                    names = np.full(len(residual), "interior", dtype=object)
                    names[0] = "first"
                    names[rebase_position - 1] = "rebase"
                    names[-1] = "endpoint"
                    count = len(residual)
                    node_output["well"].extend([record.well] * count)
                    node_output["control"].extend([control] * count)
                    node_output["offset_ft"].extend([float(offset_ft)] * count)
                    node_output["start"].extend([start] * count)
                    node_output["distance"].extend([distance] * count)
                    node_output["distance_band"].extend([distance_band] * count)
                    node_output["position"].extend(positions.tolist())
                    node_output["position_name"].extend(names.tolist())
                    node_output["residual"].extend(residual.tolist())
                    node_output["predicted_correction"].extend(
                        predicted_correction.tolist()
                    )
                    chunk_rows.append(
                        {
                            "well": record.well,
                            "control": control,
                            "offset_ft": float(offset_ft),
                            "start": start,
                            "distance": distance,
                            "distance_band": distance_band,
                            "nodes": count,
                            "exact_sse": float(np.dot(residual, residual)),
                            "rmse": float(np.sqrt(np.mean(residual**2))),
                            "mae": float(np.mean(np.abs(residual))),
                            "bias": float(np.mean(residual)),
                            "first_error": float(residual[0]),
                            "rebase_error": float(residual[rebase_position - 1]),
                            "endpoint_error": float(residual[-1]),
                        }
                    )

        if record_number % 25 == 0 or record_number == len(records):
            print(
                f"ANCHOR-RECOVERY {record_number}/{len(records)} "
                f"successful={successful_payloads}/{requested_payloads}",
                flush=True,
            )

    frame = pd.DataFrame(node_output)
    chunks = pd.DataFrame(chunk_rows)
    return (
        frame,
        chunks,
        requested_payloads,
        successful_payloads,
        requested_by_key,
        successful_by_key,
    )


# ---------------------------------------------------------------------------
# Fold-3-only execution.
# ---------------------------------------------------------------------------

started_v4 = time.time()
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the V4 anchor-recovery audit")
device_v4 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_v4, fold_path_v4 = _load_folds()
fold3_wells_v4 = sorted(folds_v4.loc[folds_v4["fold"].eq(AB_VALID_FOLD), "well"])
fold3_records_v4 = _load_records(fold3_wells_v4, "fold3_anchor_recovery")
if len(fold3_records_v4) < 150:
    raise RuntimeError(f"Unexpected Fold-3 coverage: {len(fold3_records_v4)} wells")

checkpoint_v4 = torch.load(CHECKPOINT_PATH_V4, map_location="cpu", weights_only=False)
if "state_offsets" in checkpoint_v4 and not np.array_equal(
    np.asarray(checkpoint_v4["state_offsets"]), STATE_OFFSETS
):
    raise RuntimeError("Checkpoint state axis does not match the V4 source axis")
model_v4 = SEGNMinCostMapNet().to(device_v4)
model_v4.load_state_dict(checkpoint_v4["model"], strict=True)

guards_v4 = {
    "source": str(V3_SOURCE),
    "checkpoint": str(CHECKPOINT_PATH_V4),
    "checkpoint_epoch": int(checkpoint_v4.get("epoch", -1)),
    "fold_source": str(fold_path_v4),
    "fold3_wells": len(fold3_records_v4),
    "chunks_per_well": CHUNKS_PER_WELL_V4,
    "offsets_ft": INJECTED_OFFSETS_FT_V4.tolist(),
    "controls": list(CONTROLS_V4),
    "fold1_touched": False,
    "fold2_touched": False,
    "optimizer_steps": 0,
}
print("ANCHOR-RECOVERY GUARDS", json.dumps(guards_v4, indent=2), flush=True)

(
    nodes_v4,
    chunks_v4,
    requested_v4,
    successful_v4,
    requested_by_key_v4,
    successful_by_key_v4,
) = _run_anchor_recovery_v4(model_v4, fold3_records_v4, device_v4)
coverage_v4 = successful_v4 / max(requested_v4, 1)
if nodes_v4.empty:
    raise RuntimeError("Anchor-recovery audit produced no valid predictions")

offset_rows_v4 = []
for (control, offset_ft), frame in nodes_v4.groupby(
    ["control", "offset_ft"], sort=True
):
    offset_rows_v4.append(
        {
            "control": control,
            "offset_ft": float(offset_ft),
            "payloads_requested": int(
                requested_by_key_v4.get((control, float(offset_ft)), 0)
            ),
            "payloads_successful": int(
                successful_by_key_v4.get((control, float(offset_ft)), 0)
            ),
            "payload_coverage": successful_by_key_v4.get(
                (control, float(offset_ft)), 0
            )
            / max(requested_by_key_v4.get((control, float(offset_ft)), 0), 1),
            **_metric_row_v4(frame),
            "first_rmse": _position_rmse_v4(frame, "first"),
            "rebase_rmse": _position_rmse_v4(frame, "rebase"),
            "endpoint_rmse": _position_rmse_v4(frame, "endpoint"),
        }
    )
offset_summary_v4 = pd.DataFrame(offset_rows_v4)
minimum_offset_coverage_v4 = float(offset_summary_v4["payload_coverage"].min())

control_summary_v4 = {
    control: _control_summary_v4(nodes_v4.loc[nodes_v4["control"].eq(control)])
    for control in CONTROLS_V4
}

distance_rows_v4 = []
for (control, distance_band), frame in nodes_v4.groupby(
    ["control", "distance_band"], sort=False
):
    nonzero = frame.loc[frame["offset_ft"].ne(0.0)]
    recovery_slope, recovery_intercept = _ols_slope_v4(
        -nonzero["offset_ft"], nonzero["predicted_correction"]
    )
    residual_slope, residual_intercept = _ols_slope_v4(
        nonzero["offset_ft"], nonzero["residual"]
    )
    distance_rows_v4.append(
        {
            "control": control,
            "distance_band": distance_band,
            "rows": int(len(frame)),
            "wells": int(frame["well"].nunique()),
            "recovery_slope": recovery_slope,
            "recovery_intercept": recovery_intercept,
            "residual_response_slope": residual_slope,
            "residual_response_intercept": residual_intercept,
        }
    )
distance_summary_v4 = pd.DataFrame(distance_rows_v4)

chunks_v4.to_csv(OUT_DIR / "anchor_recovery_chunks.csv", index=False)
offset_summary_v4.to_csv(OUT_DIR / "anchor_recovery_by_offset.csv", index=False)
distance_summary_v4.to_csv(OUT_DIR / "anchor_recovery_by_distance.csv", index=False)
np.savez_compressed(
    OUT_DIR / "anchor_recovery_nodes.npz",
    well=nodes_v4["well"].to_numpy(),
    control=nodes_v4["control"].to_numpy(),
    offset_ft=nodes_v4["offset_ft"].to_numpy(np.float32),
    start=nodes_v4["start"].to_numpy(np.int16),
    distance=nodes_v4["distance"].to_numpy(np.float32),
    position=nodes_v4["position"].to_numpy(np.int16),
    residual=nodes_v4["residual"].to_numpy(np.float32),
    predicted_correction=nodes_v4["predicted_correction"].to_numpy(np.float32),
)

full_v4 = control_summary_v4["full"]
no_gr_v4 = control_summary_v4["no_gr"]
shuffled_v4 = control_summary_v4["shuffled_typewell"]
checks_v4 = {
    "each_offset_payload_coverage_ge_0p80": minimum_offset_coverage_v4
    >= MIN_PAYLOAD_COVERAGE_V4,
    "zero_offset_rmse_le_4": full_v4["zero_offset_rmse"]
    <= ZERO_OFFSET_RMSE_GATE_V4,
    "full_recovery_slope_ge_0p50": full_v4["recovery_slope"]
    >= RECOVERY_SLOPE_GATE_V4,
    "full_residual_response_abs_le_0p50": abs(full_v4["residual_response_slope"])
    <= RESIDUAL_RESPONSE_ABS_GATE_V4,
    "recovery_slope_drop_no_gr_ge_0p10": full_v4["recovery_slope"]
    - no_gr_v4["recovery_slope"]
    >= CONTROL_SLOPE_DROP_GATE_V4,
    "recovery_slope_drop_shuffled_ge_0p10": full_v4["recovery_slope"]
    - shuffled_v4["recovery_slope"]
    >= CONTROL_SLOPE_DROP_GATE_V4,
}
strict_pass_v4 = all(checks_v4.values())
verdict_v4 = {
    "experiment": EXPERIMENT_V4,
    "stage": "frozen_fold3_anchor_recovery_observability",
    "sample_unit": "sampled chunks nested within complete Fold-3 wells",
    "deployable": False,
    "truth_usage": "truth injects and scores controlled chunk-anchor errors",
    "legal_model_inputs_only": True,
    "training": False,
    "guards": guards_v4,
    "payload_coverage": coverage_v4,
    "minimum_offset_payload_coverage": minimum_offset_coverage_v4,
    "control_summary": control_summary_v4,
    "checks": checks_v4,
    "strict_pass": strict_pass_v4,
    "fold1_touched": False,
    "fold2_touched": False,
    "elapsed_minutes": (time.time() - started_v4) / 60.0,
    "next": (
        "PASS -> repair warm-started scheduled rollout; frozen maps can remove anchor error"
        if strict_pass_v4
        else "FAIL -> train paired anchor-jitter/equivariance before another rollout experiment"
    ),
}
_json_v4(OUT_DIR / "verdict.json", verdict_v4)

print("\nANCHOR-RECOVERY BY OFFSET")
print(offset_summary_v4.to_string(index=False))
print("\nANCHOR-RECOVERY BY DISTANCE")
print(distance_summary_v4.to_string(index=False))
print("\nANCHOR-RECOVERY FINAL VERDICT")
print(json.dumps(verdict_v4, indent=2), flush=True)
print("saved:", OUT_DIR)
