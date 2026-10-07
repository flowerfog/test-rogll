"""Zero-training stability audit for the overlap-rebased SEGN v1 checkpoint.

This cell answers two questions before another training run:

1. Do horizontal GR and the two claimed matching channels carry causal local
   evidence when recursive anchor error is removed (teacher-anchor controls)?
2. Can a deployment-legal geometry baseline and a fixed offset gain stabilise
   rollout without changing the trained evidence model?

The first 48 Fold-2 wells select one fixed (base dose, gain) pair.  The other
107 Fold-2 wells are an untouched decoder confirmation subset.  Fold 1 is not
loaded.  Truth is used only for scoring and the registered response audit.
"""

from pathlib import Path
import json
import math
import sys
import time
import types

import numpy as np
import pandas as pd
import torch


EXPERIMENT = "segn_recurrence_stability_audit_v2"
ROOT = Path("/root/autodl-tmp")
SOURCE_CANDIDATES = (
    ROOT / "segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py",
    Path.cwd() / "segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(
        "Upload segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py to /root/autodl-tmp"
    )

source = SOURCE_PATH.read_text(encoding="utf-8")
marker = "# Execute"
if marker not in source:
    raise RuntimeError(f"Cannot locate execution marker in {SOURCE_PATH}")
module_name = "_segn_rebased_runtime_v2"
module = types.ModuleType(module_name)
module.__file__ = str(SOURCE_PATH)
sys.modules[module_name] = module
V = module.__dict__
exec(compile(source.split(marker, 1)[0], str(SOURCE_PATH), "exec"), V)

OOF_ROOT = V["OOF_ROOT"]
SEGN_ROOT = OOF_ROOT / "segn_chunked_costmap_rebased_fold2_gateway_v1"
CHECKPOINT_PATH = SEGN_ROOT / "models" / "final345_rollout_best.pt"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260721
TUNE_WELLS = 48
BASE_DOSES = (0.0, 0.03, 0.07, 0.12, 0.20, 0.35)
OFFSET_GAINS = (0.90, 1.00, 1.15, 1.30, 1.45)
INJECTED_ERRORS_FT = (-48.0, -24.0, -12.0, 0.0, 12.0, 24.0, 48.0)
RESPONSE_WELLS = 24


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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


def _rmse(truth, prediction):
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(error * error)))


def _metrics(rows):
    truth = np.concatenate([row["truth"] for row in rows])
    prediction = np.concatenate([row["prediction"] for row in rows])
    distance = np.concatenate([row["distance"] for row in rows])
    bywell = pd.DataFrame(
        {
            "well": [row["well"] for row in rows],
            "rows": [len(row["truth"]) for row in rows],
            "rmse": [_rmse(row["truth"], row["prediction"]) for row in rows],
        }
    )
    result = {
        "rows": int(len(truth)),
        "wells": int(len(rows)),
        "exact_sse": float(np.sum((prediction - truth) ** 2)),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
    }
    strata = (
        ("near_0_500_rmse", distance <= 500.0),
        ("mid_500_1000_rmse", (distance > 500.0) & (distance <= 1000.0)),
        ("mid_1000_2000_rmse", (distance > 1000.0) & (distance <= 2000.0)),
        ("far_gt2000_rmse", distance > 2000.0),
    )
    for name, mask in strata:
        result[name] = _rmse(truth[mask], prediction[mask]) if int(mask.sum()) else math.nan
    return result, bywell


@torch.no_grad()
def _decode_record(model, record, device, base_dose, offset_gain):
    model.eval()
    node_prediction = np.full(len(record.node_rows), np.nan, np.float64)
    node_prediction[0] = float(record.anchor_tvt)
    geometry_base = (
        float(record.anchor_tvt)
        - 512.0 * record.condition[4, record.node_rows].astype(np.float64)
    )
    prior_deployed_anchor = float(record.anchor_tvt)
    prior_start = 0
    for start_value in record.chunk_starts:
        start = int(start_value)
        start_row = int(record.node_rows[start])
        if not np.isfinite(node_prediction[start]):
            prior = np.flatnonzero(np.isfinite(node_prediction[: start + 1]))
            if not len(prior):
                raise RuntimeError(f"{record.well}: rollout lost anchor at {start}")
            raw_anchor = float(node_prediction[prior[-1]])
        else:
            raw_anchor = float(node_prediction[start])
        deployed_anchor = (
            raw_anchor
            if start == 0
            else (1.0 - base_dose) * raw_anchor + base_dose * float(geometry_base[start])
        )
        prior_row = int(record.node_rows[prior_start])
        md_delta = max(float(record.md[start_row] - record.md[prior_row]), 1.0)
        previous_delta = deployed_anchor - prior_deployed_anchor if start > 0 else 0.0
        previous_rate = previous_delta / md_delta if start > 0 else 0.0
        payload = V["_chunk_payload"](
            record,
            start,
            deployed_anchor=deployed_anchor,
            previous_delta=previous_delta,
            previous_rate=previous_rate,
        )
        expected, _, _ = V["_predict_chunk"](model, payload, device)
        target_count = len(payload["target_rows"])
        end = start + target_count
        node_prediction[start + 1 : end + 1] = deployed_anchor + offset_gain * expected
        prior_deployed_anchor = deployed_anchor
        prior_start = start

    valid = np.flatnonzero(np.isfinite(node_prediction))
    prediction = np.interp(
        record.md[record.eval_index],
        record.md[record.node_rows[valid]],
        node_prediction[valid],
    )
    truth = record.truth[record.eval_index].astype(np.float64)
    distance = record.md[record.eval_index] - float(record.md[record.anchor_index])
    return {
        "well": record.well,
        "truth": truth,
        "prediction": prediction.astype(np.float64),
        "distance": distance.astype(np.float64),
    }


def _evaluate_decode(model, records, device, base_dose, offset_gain, label):
    rows = []
    for number, record in enumerate(records, 1):
        rows.append(_decode_record(model, record, device, base_dose, offset_gain))
        if number % 24 == 0 or number == len(records):
            print(f"SEGN stability {label} {number}/{len(records)}", flush=True)
    return _metrics(rows)


@torch.no_grad()
def _response_audit(model, records, device):
    rows = []
    for record in records:
        for start_value in record.chunk_starts:
            start = int(start_value)
            start_row = int(record.node_rows[start])
            prior_start = max(0, start - V["CHUNK_STRIDE_NODES"])
            prior_row = int(record.node_rows[prior_start])
            md_delta = max(float(record.md[start_row] - record.md[prior_row]), 1.0)
            true_delta = float(record.truth[start_row] - record.truth[prior_row])
            for injected in INJECTED_ERRORS_FT:
                deployed_anchor = float(record.truth[start_row]) + injected
                payload = V["_chunk_payload"](
                    record,
                    start,
                    deployed_anchor=deployed_anchor,
                    previous_delta=true_delta,
                    previous_rate=true_delta / md_delta,
                )
                expected, _, _ = V["_predict_chunk"](model, payload, device)
                endpoint_truth = float(record.truth[payload["target_rows"][-1]])
                endpoint_error = deployed_anchor + float(expected[-1]) - endpoint_truth
                rows.append(
                    {
                        "well": record.well,
                        "distance": float(payload["distance"]),
                        "injected_anchor_error": float(injected),
                        "endpoint_error": endpoint_error,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "injected_anchor_response.csv", index=False)
    summary = []
    bins = pd.cut(
        frame["distance"],
        [-np.inf, 500.0, 1000.0, 2000.0, np.inf],
        labels=["0_500", "500_1000", "1000_2000", "gt2000"],
    )
    for stratum, group in frame.groupby(bins, observed=True):
        x = group["injected_anchor_error"].to_numpy(np.float64)
        y = group["endpoint_error"].to_numpy(np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        summary.append(
            {
                "stratum": str(stratum),
                "samples": int(len(group)),
                "response_slope": float(slope),
                "response_intercept": float(intercept),
                "zero_injection_rmse": _rmse(
                    np.zeros(int((group["injected_anchor_error"] == 0).sum())),
                    group.loc[group["injected_anchor_error"] == 0, "endpoint_error"],
                ),
            }
        )
    return pd.DataFrame(summary)


started = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)
device = torch.device("cuda")

folds, fold_path = V["_load_folds"]()
valid_wells = folds.loc[folds["fold"].eq(V["VALID_FOLD"]), "well"].tolist()
records = V["_load_records"](valid_wells, "fold2_stability")
if len(records) != 155:
    raise RuntimeError(f"Expected 155 Fold-2 records, got {len(records)}")

model = V["ChunkedCostMapNet"]().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

teacher_controls = {}
for control in ("full", "no_gr", "shuffled_typewell", "no_bilinear", "no_cosine"):
    metrics, _ = V["_evaluate"](
        model,
        records,
        device,
        mode="teacher",
        control=control,
    )
    teacher_controls[control] = metrics
teacher_table = pd.DataFrame(
    [{"control": key, **value} for key, value in teacher_controls.items()]
)
teacher_table.to_csv(OUT_DIR / "teacher_control_audit.csv", index=False)
print("SEGN TEACHER CONTROLS")
print(teacher_table.to_string(index=False), flush=True)

response = _response_audit(model, records[:RESPONSE_WELLS], device)
response.to_csv(OUT_DIR / "injected_anchor_response_summary.csv", index=False)
print("SEGN INJECTED-ANCHOR RESPONSE")
print(response.to_string(index=False), flush=True)

tune_records = records[:TUNE_WELLS]
confirmation_records = records[TUNE_WELLS:]
scan = []
for dose in BASE_DOSES:
    for gain in OFFSET_GAINS:
        metrics, _ = _evaluate_decode(
            model,
            tune_records,
            device,
            dose,
            gain,
            f"tune_d{dose:.2f}_g{gain:.2f}",
        )
        scan.append({"base_dose": dose, "offset_gain": gain, **metrics})
scan_frame = pd.DataFrame(scan).sort_values(["rmse", "base_dose", "offset_gain"])
scan_frame.to_csv(OUT_DIR / "decoder_selection_scan.csv", index=False)
print("SEGN DECODER SELECTION")
print(scan_frame.head(20).to_string(index=False), flush=True)

winner = scan_frame.iloc[0].to_dict()
confirmation_metrics, confirmation_bywell = _evaluate_decode(
    model,
    confirmation_records,
    device,
    float(winner["base_dose"]),
    float(winner["offset_gain"]),
    "confirmation",
)
confirmation_bywell.to_csv(OUT_DIR / "confirmation_bywell.csv", index=False)
full_metrics, full_bywell = _evaluate_decode(
    model,
    records,
    device,
    float(winner["base_dose"]),
    float(winner["offset_gain"]),
    "full",
)
full_bywell.to_csv(OUT_DIR / "full_bywell.csv", index=False)

baseline_confirmation, _ = _evaluate_decode(
    model, confirmation_records, device, 0.0, 1.0, "baseline_confirmation"
)
confirmation_gain = baseline_confirmation["rmse"] - confirmation_metrics["rmse"]
teacher_no_gr_drop = (
    teacher_controls["no_gr"]["rmse"] - teacher_controls["full"]["rmse"]
)
response_stable = bool((response["response_slope"].abs() < 0.85).all())
decoder_pass = bool(
    confirmation_gain >= 1.0
    and confirmation_metrics["rmse"] <= 10.0
    and confirmation_metrics.get("far_gt2000_rmse", math.inf) <= 12.0
)
causal_teacher_pass = teacher_no_gr_drop >= 1.0

if not causal_teacher_pass:
    next_step = "FAIL LOCAL CAUSALITY -> rebuild query evidence; do not add a spatial/base branch"
elif not response_stable:
    next_step = "UNSTABLE RESPONSE -> train explicit absolute-anchor correction before another rollout"
elif not decoder_pass:
    next_step = "STABLE BUT WEAK -> add cross-fitted continuous base-prior fusion"
else:
    next_step = "PASS -> freeze decoder and run untouched Fold1"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_zero_training_teacher_causality_and_recurrence_stability",
    "source": str(SOURCE_PATH),
    "checkpoint": str(CHECKPOINT_PATH),
    "fold_source": str(fold_path),
    "fold1_touched": False,
    "teacher_controls": teacher_controls,
    "teacher_no_gr_degradation": teacher_no_gr_drop,
    "injected_anchor_response": response.to_dict(orient="records"),
    "decoder_selection": winner,
    "baseline_confirmation": baseline_confirmation,
    "confirmation": confirmation_metrics,
    "confirmation_gain": confirmation_gain,
    "full": full_metrics,
    "gates": {
        "teacher_no_gr_degradation_ge": 1.0,
        "each_abs_response_slope_lt": 0.85,
        "confirmation_gain_ge": 1.0,
        "confirmation_rmse_le": 10.0,
        "confirmation_far_gt2000_rmse_le": 12.0,
    },
    "causal_teacher_pass": causal_teacher_pass,
    "response_stable": response_stable,
    "decoder_pass": decoder_pass,
    "strict_pass": bool(causal_teacher_pass and response_stable and decoder_pass),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": next_step,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("SEGN STABILITY FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
