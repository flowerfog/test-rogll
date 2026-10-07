"""Zero-training audit of a fold-clean base as SEGN's absolute-level channel.

The SEGN v1 encoder has strong teacher-anchor performance but accumulates
large recursive level error.  This audit gives the frozen encoder a stable,
fold-clean OOF base at each chunk boundary and asks whether its local cost map
adds value.  No base is selected on the confirmation subset.

The first 48 Fold-2 wells select one fixed base/anchor-dose/output-blend/gain
configuration.  The remaining 107 wells confirm it.  Fold 1 is not loaded.
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


EXPERIMENT = "segn_foldclean_base_reanchor_audit_v3"
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
if "# Execute" not in source:
    raise RuntimeError(f"Cannot locate execution marker in {SOURCE_PATH}")
module_name = "_segn_rebased_runtime_v3"
module = types.ModuleType(module_name)
module.__file__ = str(SOURCE_PATH)
sys.modules[module_name] = module
V = module.__dict__
exec(compile(source.split("# Execute", 1)[0], str(SOURCE_PATH), "exec"), V)

OOF_ROOT = V["OOF_ROOT"]
SEGN_ROOT = OOF_ROOT / "segn_chunked_costmap_rebased_fold2_gateway_v1"
CHECKPOINT_PATH = SEGN_ROOT / "models" / "final345_rollout_best.pt"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

TUNE_WELLS = 48
BASE_PREFERENCE = ("strong_bias", "cp12", "strong")
ANCHOR_DOSES = (0.50, 0.75, 1.00)
OFFSET_GAINS = (0.85, 1.00, 1.15)
OUTPUT_MODEL_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
TRUTH_TOLERANCE_FT = 0.02


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


def _metrics(rows, prediction_key="prediction"):
    truth = np.concatenate([row["truth"] for row in rows])
    prediction = np.concatenate([row[prediction_key] for row in rows])
    distance = np.concatenate([row["distance"] for row in rows])
    well_rmse = np.asarray(
        [_rmse(row["truth"], row[prediction_key]) for row in rows], np.float64
    )
    result = {
        "rows": int(len(truth)),
        "wells": int(len(rows)),
        "exact_sse": float(np.sum((prediction - truth) ** 2)),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    for name, mask in (
        ("near_0_500_rmse", distance <= 500.0),
        ("mid_500_1000_rmse", (distance > 500.0) & (distance <= 1000.0)),
        ("mid_1000_2000_rmse", (distance > 1000.0) & (distance <= 2000.0)),
        ("far_gt2000_rmse", distance > 2000.0),
    ):
        result[name] = _rmse(truth[mask], prediction[mask]) if int(mask.sum()) else math.nan
    return result


def _load_base_paths(records):
    if POOL_PATH is None:
        raise FileNotFoundError(f"No candidate pool under {OOF_ROOT}")
    header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
    truth_column = next(
        (column for column in ("tvt_true", "TVT", "target") if column in header),
        None,
    )
    bases = [column for column in BASE_PREFERENCE if column in header]
    if truth_column is None or "well" not in header or not bases:
        raise RuntimeError(f"Unexpected pool columns: {header}")
    wanted = {record.well for record in records}
    chunks = []
    for chunk in pd.read_csv(
        POOL_PATH,
        usecols=["well", truth_column, *bases],
        dtype={"well": str},
        chunksize=500_000,
    ):
        chunks.append(chunk[chunk["well"].isin(wanted)].copy())
    pool = pd.concat(chunks, ignore_index=True)
    grouped = {well: frame for well, frame in pool.groupby("well", sort=False)}
    output = {}
    alignment = []
    for record in records:
        frame = grouped.get(record.well)
        if frame is None or len(frame) != len(record.eval_index):
            raise RuntimeError(
                f"{record.well}: pool rows {0 if frame is None else len(frame)} "
                f"!= eval rows {len(record.eval_index)}"
            )
        pool_truth = pd.to_numeric(frame[truth_column], errors="raise").to_numpy(np.float64)
        record_truth = record.truth[record.eval_index].astype(np.float64)
        gap = np.abs(pool_truth - record_truth)
        if not np.isfinite(gap).all() or float(gap.max()) > TRUTH_TOLERANCE_FT:
            raise RuntimeError(f"{record.well}: truth alignment max gap={gap.max():.6f}")
        output[record.well] = {
            base: pd.to_numeric(frame[base], errors="raise").to_numpy(np.float64)
            for base in bases
        }
        alignment.append(float(gap.max()))
    return output, bases, {
        "pool": str(POOL_PATH),
        "rows": int(len(pool)),
        "wells": int(len(grouped)),
        "maximum_truth_gap": float(max(alignment)),
    }


@torch.no_grad()
def _decode_record(model, record, base_eval, device, anchor_dose, offset_gain):
    model.eval()
    eval_md = record.md[record.eval_index].astype(np.float64)
    base_nodes = np.interp(
        record.md[record.node_rows].astype(np.float64),
        np.r_[float(record.md[record.anchor_index]), eval_md],
        np.r_[float(record.anchor_tvt), base_eval],
    )
    node_prediction = np.full(len(record.node_rows), np.nan, np.float64)
    node_prediction[0] = float(record.anchor_tvt)
    prior_deployed_anchor = float(record.anchor_tvt)
    prior_start = 0
    for start_value in record.chunk_starts:
        start = int(start_value)
        start_row = int(record.node_rows[start])
        raw_anchor = (
            float(node_prediction[start])
            if np.isfinite(node_prediction[start])
            else float(base_nodes[start])
        )
        deployed_anchor = (
            raw_anchor
            if start == 0
            else (1.0 - anchor_dose) * raw_anchor + anchor_dose * float(base_nodes[start])
        )
        prior_row = int(record.node_rows[prior_start])
        md_delta = max(float(record.md[start_row] - record.md[prior_row]), 1.0)
        previous_delta = deployed_anchor - prior_deployed_anchor if start > 0 else 0.0
        payload = V["_chunk_payload"](
            record,
            start,
            deployed_anchor=deployed_anchor,
            previous_delta=previous_delta,
            previous_rate=previous_delta / md_delta if start > 0 else 0.0,
        )
        expected, _, _ = V["_predict_chunk"](model, payload, device)
        count = len(payload["target_rows"])
        node_prediction[start + 1 : start + count + 1] = (
            deployed_anchor + offset_gain * expected
        )
        prior_deployed_anchor = deployed_anchor
        prior_start = start
    valid = np.flatnonzero(np.isfinite(node_prediction))
    model_eval = np.interp(
        eval_md,
        record.md[record.node_rows[valid]],
        node_prediction[valid],
    )
    return model_eval.astype(np.float64)


def _build_rows(model, records, paths, base_name, device, anchor_dose, offset_gain):
    rows = []
    for number, record in enumerate(records, 1):
        base = paths[record.well][base_name]
        model_prediction = _decode_record(
            model, record, base, device, anchor_dose, offset_gain
        )
        rows.append(
            {
                "well": record.well,
                "truth": record.truth[record.eval_index].astype(np.float64),
                "base": base,
                "model": model_prediction,
                "distance": (
                    record.md[record.eval_index] - float(record.md[record.anchor_index])
                ).astype(np.float64),
            }
        )
        if number % 24 == 0 or number == len(records):
            print(
                f"SEGN base-reanchor {base_name} d={anchor_dose:.2f} "
                f"g={offset_gain:.2f} {number}/{len(records)}",
                flush=True,
            )
    return rows


def _blend_rows(rows, model_weight):
    return [
        {
            **row,
            "prediction": (1.0 - model_weight) * row["base"] + model_weight * row["model"],
        }
        for row in rows
    ]


started = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)
device = torch.device("cuda")

folds, fold_path = V["_load_folds"]()
valid_wells = folds.loc[folds["fold"].eq(V["VALID_FOLD"]), "well"].tolist()
records = V["_load_records"](valid_wells, "fold2_base_reanchor")
if len(records) != 155:
    raise RuntimeError(f"Expected 155 Fold-2 records, got {len(records)}")
paths, base_names, alignment = _load_base_paths(records)

model = V["ChunkedCostMapNet"]().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

tune_records = records[:TUNE_WELLS]
confirmation_records = records[TUNE_WELLS:]
scan = []
for base_name in base_names:
    base_tune_rows = [
        {
            "well": record.well,
            "truth": record.truth[record.eval_index].astype(np.float64),
            "prediction": paths[record.well][base_name],
            "distance": (
                record.md[record.eval_index] - float(record.md[record.anchor_index])
            ).astype(np.float64),
        }
        for record in tune_records
    ]
    base_metrics = _metrics(base_tune_rows)
    for anchor_dose in ANCHOR_DOSES:
        for offset_gain in OFFSET_GAINS:
            decoded = _build_rows(
                model,
                tune_records,
                paths,
                base_name,
                device,
                anchor_dose,
                offset_gain,
            )
            for model_weight in OUTPUT_MODEL_WEIGHTS:
                metrics = _metrics(_blend_rows(decoded, model_weight))
                scan.append(
                    {
                        "base": base_name,
                        "anchor_dose": anchor_dose,
                        "offset_gain": offset_gain,
                        "model_weight": model_weight,
                        "base_rmse": base_metrics["rmse"],
                        "gain_vs_base": base_metrics["rmse"] - metrics["rmse"],
                        **metrics,
                    }
                )

scan_frame = pd.DataFrame(scan).sort_values(
    ["rmse", "base", "anchor_dose", "offset_gain", "model_weight"]
)
scan_frame.to_csv(OUT_DIR / "selection_scan.csv", index=False)
print("SEGN BASE-REANCHOR SELECTION")
print(scan_frame.head(30).to_string(index=False), flush=True)
winner = scan_frame.iloc[0].to_dict()

confirmation_decoded = _build_rows(
    model,
    confirmation_records,
    paths,
    str(winner["base"]),
    device,
    float(winner["anchor_dose"]),
    float(winner["offset_gain"]),
)
confirmation_rows = _blend_rows(confirmation_decoded, float(winner["model_weight"]))
confirmation = _metrics(confirmation_rows)
confirmation_base = _metrics(
    [{**row, "prediction": row["base"]} for row in confirmation_decoded]
)
confirmation_gain = confirmation_base["rmse"] - confirmation["rmse"]

full_decoded = _build_rows(
    model,
    records,
    paths,
    str(winner["base"]),
    device,
    float(winner["anchor_dose"]),
    float(winner["offset_gain"]),
)
full_rows = _blend_rows(full_decoded, float(winner["model_weight"]))
full = _metrics(full_rows)
full_base = _metrics([{**row, "prediction": row["base"]} for row in full_decoded])

bywell = pd.DataFrame(
    [
        {
            "well": row["well"],
            "rows": len(row["truth"]),
            "base_rmse": _rmse(row["truth"], row["base"]),
            "model_rmse": _rmse(row["truth"], row["model"]),
            "blend_rmse": _rmse(row["truth"], row["prediction"]),
        }
        for row in full_rows
    ]
)
bywell.to_csv(OUT_DIR / "full_bywell.csv", index=False)

strict_pass = bool(
    confirmation_gain >= 0.20
    and confirmation["rmse"] <= confirmation_base["rmse"] - 0.20
    and confirmation.get("far_gt2000_rmse", math.inf)
    <= confirmation_base.get("far_gt2000_rmse", -math.inf) - 0.20
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_zero_training_foldclean_base_reanchor_confirmation",
    "source": str(SOURCE_PATH),
    "checkpoint": str(CHECKPOINT_PATH),
    "fold_source": str(fold_path),
    "fold1_touched": False,
    "alignment": alignment,
    "selection_wells": TUNE_WELLS,
    "confirmation_wells": len(confirmation_records),
    "selected": winner,
    "confirmation_base": confirmation_base,
    "confirmation": confirmation,
    "confirmation_gain": confirmation_gain,
    "full_base": full_base,
    "full": full,
    "gates": {
        "confirmation_gain_ge": 0.20,
        "confirmation_rmse_improvement_ge": 0.20,
        "confirmation_far_improvement_ge": 0.20,
    },
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> train continuous evidence/base trust gate on outer folds"
        if strict_pass
        else "FAIL -> SEGN local evidence does not add to the fold-clean base"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("SEGN BASE-REANCHOR FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
