"""Zero-training K-anchor beam audit for the trained Fold-2 SEGN encoder.

The single-anchor rollout permanently commits the posterior mean at every
512-ft chunk boundary.  This audit carries several separated endpoint modes
and lets later cost-map evidence rerank them.  No truth enters beam expansion,
scoring, pruning, or decoding.

The 48 Fold-2 wells already used for epoch selection choose one fixed beam
configuration.  The remaining 107 Fold-2 wells are the decoder confirmation
set.  Fold 1 is never loaded.
"""

from pathlib import Path
import ast
import copy
import gc
import json
import math
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


EXPERIMENT = "segn_kanchor_beam_fold2_audit_v1"
ROOT = Path("/root/autodl-tmp")
SOURCE_CANDIDATES = (
    ROOT / "segn_chunked_costmap_fold2_gateway_v0_cell.py",
    Path.cwd() / "segn_chunked_costmap_fold2_gateway_v0_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(
        "Upload segn_chunked_costmap_fold2_gateway_v0_cell.py to /root/autodl-tmp"
    )

source = SOURCE_PATH.read_text(encoding="utf-8")
marker = "# Execute"
if marker not in source:
    raise RuntimeError(f"Cannot locate execution marker in {SOURCE_PATH}")
module_name = "_segn_chunked_costmap_runtime_v1"
module = types.ModuleType(module_name)
module.__file__ = str(SOURCE_PATH)
sys.modules[module_name] = module
V = module.__dict__
exec(compile(source.split(marker, 1)[0], str(SOURCE_PATH), "exec"), V)

OOF_ROOT = V["OOF_ROOT"]
SEGN_ROOT = OOF_ROOT / "segn_chunked_costmap_fold2_gateway_v0"
CHECKPOINT_PATH = SEGN_ROOT / "models" / "final345_rollout_best.pt"
BASELINE_PATH = SEGN_ROOT / "fold2_rollout_rows.npz"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260721
TUNE_WELLS = 48
NMS_SEPARATION_FT = 6.0
RATE_SCALE = 0.02
BEAM_CONFIGS = (
    {"name": "b4_k3_free", "beam_size": 4, "branches": 3, "transition_lambda": 0.0},
    {"name": "b8_k4_free", "beam_size": 8, "branches": 4, "transition_lambda": 0.0},
    {"name": "b8_k4_p005", "beam_size": 8, "branches": 4, "transition_lambda": 0.05},
    {"name": "b12_k5_p005", "beam_size": 12, "branches": 5, "transition_lambda": 0.05},
    {"name": "b8_k4_p020", "beam_size": 8, "branches": 4, "transition_lambda": 0.20},
)
DECODES = ("map", "posterior_mean")
POSTERIOR_TEMPERATURE = 1.0

CONFIRMATION_RMSE_GATE = 10.5
CONFIRMATION_GAIN_GATE = 1.0
FAR_RMSE_GATE = 13.0


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


def _select_modes(probability, expected, count):
    probability = np.asarray(probability, np.float64)
    order = np.argsort(-probability)
    selected = []
    expected_index = int(np.argmin(np.abs(V["STATE_OFFSETS"] - float(expected))))
    for index in np.r_[expected_index, order]:
        index = int(index)
        state = float(V["STATE_OFFSETS"][index])
        if all(abs(state - float(V["STATE_OFFSETS"][other])) >= NMS_SEPARATION_FT for other in selected):
            selected.append(index)
        if len(selected) >= count:
            break
    return selected


@torch.no_grad()
def _costmap(model, payload, device):
    tensors = V["_tensor_payload"](payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(*tensors)
    log_probability = F.log_softmax(logits.float(), dim=0)
    probability = torch.exp(log_probability)
    states = torch.from_numpy(V["STATE_OFFSETS"]).to(device)[:, None]
    expected = (probability * states).sum(dim=0)
    return (
        log_probability.cpu().numpy().astype(np.float64),
        probability.cpu().numpy().astype(np.float64),
        expected.cpu().numpy().astype(np.float64),
    )


def _deduplicate_beams(beams, beam_size):
    beams = sorted(beams, key=lambda item: item["score"], reverse=True)
    kept = []
    for beam in beams:
        if any(abs(beam["anchor"] - prior["anchor"]) < 1.0 for prior in kept):
            continue
        kept.append(beam)
        if len(kept) >= beam_size:
            break
    if not kept:
        raise RuntimeError("Beam pruning removed every hypothesis")
    return kept


@torch.no_grad()
def _decode_record(model, record, device, config, decode):
    model.eval()
    node_count = len(record.node_rows)
    initial_path = np.full(node_count, np.nan, np.float64)
    initial_path[0] = float(record.anchor_tvt)
    beams = [
        {
            "score": 0.0,
            "anchor": float(record.anchor_tvt),
            "previous_rate": None,
            "path": initial_path,
        }
    ]

    for start in record.chunk_starts:
        start = int(start)
        target_count = min(V["CHUNK_TARGET_NODES"], node_count - 1 - start)
        if target_count <= 0:
            continue
        expanded = []
        for parent in beams:
            payload = V["_chunk_payload"](
                record,
                start,
                deployed_anchor=float(parent["anchor"]),
            )
            log_probability, probability, expected = _costmap(model, payload, device)
            endpoint_modes = _select_modes(
                probability[:, -1], expected[-1], int(config["branches"])
            )
            chunk_md = max(
                float(record.md[payload["target_rows"][-1]] - record.md[record.node_rows[start]]),
                1.0,
            )
            ramp = np.linspace(1.0 / target_count, 1.0, target_count)
            for endpoint_index in endpoint_modes:
                endpoint_state = float(V["STATE_OFFSETS"][endpoint_index])
                path_state = expected + ramp * (endpoint_state - expected[-1])
                path_indices = np.rint(path_state + V["STATE_HALF_WIDTH_FT"]).astype(np.int64)
                path_indices = np.clip(path_indices, 0, len(V["STATE_OFFSETS"]) - 1)
                columns = np.arange(target_count)
                evidence_score = float(np.mean(log_probability[path_indices, columns]))
                rate = endpoint_state / chunk_md
                if parent["previous_rate"] is None:
                    transition_cost = 0.0
                else:
                    rate_change = (rate - float(parent["previous_rate"])) / RATE_SCALE
                    transition_cost = float(config["transition_lambda"]) * rate_change * rate_change
                child_path = parent["path"].copy()
                child_path[start + 1 : start + target_count + 1] = float(parent["anchor"]) + path_state
                expanded.append(
                    {
                        "score": float(parent["score"]) + evidence_score - transition_cost,
                        "anchor": float(parent["anchor"]) + endpoint_state,
                        "previous_rate": rate,
                        "path": child_path,
                    }
                )
        beams = _deduplicate_beams(expanded, int(config["beam_size"]))

    scores = np.asarray([beam["score"] for beam in beams], np.float64)
    paths = np.stack([beam["path"] for beam in beams], axis=0)
    if decode == "map":
        node_prediction = paths[int(np.argmax(scores))]
    elif decode == "posterior_mean":
        weights = np.exp((scores - np.max(scores)) / POSTERIOR_TEMPERATURE)
        weights = weights / np.sum(weights)
        node_prediction = np.sum(weights[:, None] * paths, axis=0)
    else:
        raise ValueError(decode)
    valid = np.flatnonzero(np.isfinite(node_prediction))
    support_md = record.md[record.node_rows[valid]]
    support_tvt = node_prediction[valid]
    prediction = np.interp(record.md[record.eval_index], support_md, support_tvt)
    truth = record.truth[record.eval_index].astype(np.float64)
    distance = record.md[record.eval_index] - float(record.md[record.anchor_index])
    return truth, prediction.astype(np.float64), distance.astype(np.float64), len(beams)


def _metrics(rows):
    truth = np.concatenate([row["truth"] for row in rows])
    prediction = np.concatenate([row["prediction"] for row in rows])
    distance = np.concatenate([row["distance"] for row in rows])
    bywell = pd.DataFrame(
        [
            {
                "well": row["well"],
                "rows": len(row["truth"]),
                "rmse": _rmse(row["truth"], row["prediction"]),
                "final_beams": row["final_beams"],
            }
            for row in rows
        ]
    )
    result = {
        "rows": len(truth),
        "wells": len(bywell),
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
    return result, bywell, truth, prediction


@torch.no_grad()
def _evaluate(model, records, device, config, decode, label):
    rows = []
    started = time.time()
    for number, record in enumerate(records, 1):
        truth, prediction, distance, final_beams = _decode_record(
            model, record, device, config, decode
        )
        rows.append(
            {
                "well": record.well,
                "truth": truth,
                "prediction": prediction,
                "distance": distance,
                "final_beams": final_beams,
            }
        )
        if number % 12 == 0 or number == len(records):
            print(
                f"K-ANCHOR {label} {number}/{len(records)} "
                f"elapsed={(time.time() - started) / 60.0:.1f}m",
                flush=True,
            )
    return _metrics(rows)


def _baseline_metrics(path, selected_wells):
    with np.load(path, allow_pickle=False) as data:
        truth = data["truth"].astype(np.float64)
        prediction = data["prediction"].astype(np.float64)
        well = data["well"].astype(str)
    mask = np.isin(well, np.asarray(selected_wells).astype(str))
    return {
        "rows": int(mask.sum()),
        "wells": int(len(np.unique(well[mask]))),
        "rmse": _rmse(truth[mask], prediction[mask]),
    }


started = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)
if not BASELINE_PATH.exists():
    raise FileNotFoundError(BASELINE_PATH)
device = torch.device("cuda")

folds, fold_path = V["_load_folds"]()
valid_wells = folds.loc[folds["fold"].eq(V["VALID_FOLD"]), "well"].tolist()
valid_records = V["_load_records"](valid_wells, "fold2_kanchor")
valid_order = np.argsort([record.eval_rows for record in valid_records])
tune_positions = np.rint(np.linspace(0, len(valid_order) - 1, TUNE_WELLS)).astype(int)
tune_indices = set(int(valid_order[position]) for position in np.unique(tune_positions))
tune_records = [record for index, record in enumerate(valid_records) if index in tune_indices]
confirmation_records = [record for index, record in enumerate(valid_records) if index not in tune_indices]
if (len(tune_records), len(confirmation_records)) != (48, 107):
    raise RuntimeError(f"Unexpected tune/confirmation sizes: {len(tune_records)}/{len(confirmation_records)}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model = V["ChunkedCostMapNet"]().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

print(
    "K-ANCHOR CONFIG",
    {
        "source": str(SOURCE_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("best_epoch"),
        "tune_wells": len(tune_records),
        "confirmation_wells": len(confirmation_records),
        "configs": [config["name"] for config in BEAM_CONFIGS],
    },
    flush=True,
)

selection_rows = []
for config in BEAM_CONFIGS:
    for decode in DECODES:
        label = f"{config['name']}__{decode}"
        metrics, _, _, _ = _evaluate(model, tune_records, device, config, decode, label)
        selection_rows.append({"config": config["name"], "decode": decode, **metrics})
        print("K-ANCHOR SELECTION", selection_rows[-1], flush=True)
selection = pd.DataFrame(selection_rows).sort_values("rmse", ignore_index=True)
selection.to_csv(OUT_DIR / "selection_scan.csv", index=False)
winner = selection.iloc[0]
selected_config = next(config for config in BEAM_CONFIGS if config["name"] == winner["config"])
selected_decode = str(winner["decode"])

confirmation_metrics, confirmation_bywell, confirmation_truth, confirmation_prediction = _evaluate(
    model,
    confirmation_records,
    device,
    selected_config,
    selected_decode,
    "confirmation",
)
full_metrics, full_bywell, full_truth, full_prediction = _evaluate(
    model,
    valid_records,
    device,
    selected_config,
    selected_decode,
    "full",
)
confirmation_bywell.to_csv(OUT_DIR / "confirmation_bywell.csv", index=False)
full_bywell.to_csv(OUT_DIR / "full_bywell.csv", index=False)
np.savez_compressed(
    OUT_DIR / "fold2_kanchor_rows.npz",
    truth=full_truth.astype(np.float32),
    prediction=full_prediction.astype(np.float32),
    well=np.concatenate([np.repeat(record.well, len(record.eval_index)) for record in valid_records]),
)

tune_baseline = _baseline_metrics(BASELINE_PATH, [record.well for record in tune_records])
confirmation_baseline = _baseline_metrics(
    BASELINE_PATH, [record.well for record in confirmation_records]
)
full_baseline = _baseline_metrics(BASELINE_PATH, valid_wells)

comparison = pd.DataFrame(
    [
        {"subset": "tune", "variant": "single_anchor", **tune_baseline},
        {"subset": "tune", "variant": "k_anchor", **{key: winner[key] for key in ("rows", "wells", "rmse")}},
        {"subset": "confirmation", "variant": "single_anchor", **confirmation_baseline},
        {"subset": "confirmation", "variant": "k_anchor", **confirmation_metrics},
        {"subset": "full", "variant": "single_anchor", **full_baseline},
        {"subset": "full", "variant": "k_anchor", **full_metrics},
    ]
)
comparison.to_csv(OUT_DIR / "comparison.csv", index=False)
print("K-ANCHOR SELECTION SCAN")
print(selection.to_string(index=False), flush=True)
print("K-ANCHOR COMPARISON")
print(comparison.to_string(index=False), flush=True)

confirmation_gain = confirmation_baseline["rmse"] - confirmation_metrics["rmse"]
strict_pass = bool(
    confirmation_metrics["rmse"] <= CONFIRMATION_RMSE_GATE
    and confirmation_gain >= CONFIRMATION_GAIN_GATE
    and confirmation_metrics["far_gt2000_rmse"] <= FAR_RMSE_GATE
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_zero_training_k_anchor_decoder_confirmation",
    "source_checkpoint": str(CHECKPOINT_PATH),
    "fold_source": str(fold_path),
    "fold1_touched": False,
    "selection_wells": len(tune_records),
    "confirmation_wells": len(confirmation_records),
    "selected_config": selected_config,
    "selected_decode": selected_decode,
    "selection_metrics": winner.to_dict(),
    "confirmation_single_anchor": confirmation_baseline,
    "confirmation_k_anchor": confirmation_metrics,
    "confirmation_gain": confirmation_gain,
    "full_single_anchor": full_baseline,
    "full_k_anchor": full_metrics,
    "gates": {
        "confirmation_rmse_le": CONFIRMATION_RMSE_GATE,
        "confirmation_gain_ge": CONFIRMATION_GAIN_GATE,
        "confirmation_far_gt2000_rmse_le": FAR_RMSE_GATE,
    },
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> rerun cross-evidence audit on fold2_kanchor_rows.npz"
        if strict_pass
        else "FAIL -> close SEGN rollout repair and build the complete STRIDE interaction skeleton"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("K-ANCHOR FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)

del model
gc.collect()
torch.cuda.empty_cache()
