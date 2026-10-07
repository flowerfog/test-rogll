"""Decode audit for a stopped full-well global TVT U-Net run.

Run in the same kernel after interrupting
``fullwell_global_tvt_unet_onefold_v0_cell.py``.  No training is performed.
The best saved checkpoint is decoded by posterior means at fixed temperatures,
argmax, and anchored first-order Viterbi paths.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


REQUIRED_GLOBALS = (
    "OUT_DIR",
    "MODEL_DIR",
    "SEED",
    "STATE_OFFSETS",
    "FullWellGlobalUNet",
    "valid_store",
    "input_channels",
    "device",
    "_compose_panel",
)
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the full-well cell through at least one saved epoch first; missing: "
        + repr(missing)
    )

AUDIT_DIR = Path(OUT_DIR) / "decode_audit_v0"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
MEAN_TEMPERATURES = (0.50, 0.75, 1.00, 1.50, 2.00)
DP_CONFIGS = (
    (2, 0.00),
    (2, 0.05),
    (4, 0.00),
    (4, 0.02),
    (4, 0.10),
    (8, 0.02),
    (8, 0.10),
)


def _viterbi(emission, valid, start_column, start_state, max_step, penalty):
    emission = np.asarray(emission, np.float64)
    valid = np.asarray(valid, bool)
    states, columns = emission.shape
    length = columns - int(start_column)
    if length <= 0:
        raise RuntimeError("empty eval suffix")
    back = np.full((length, states), -1, np.int16)
    previous = np.full(states, -np.inf, np.float64)
    for state in range(states):
        delta = state - int(start_state)
        if abs(delta) <= max_step and valid[state, start_column]:
            previous[state] = emission[state, start_column] - penalty * delta * delta
            back[0, state] = int(start_state)
    for local_column in range(1, length):
        column = int(start_column) + local_column
        current = np.full(states, -np.inf, np.float64)
        for state in range(states):
            if not valid[state, column]:
                continue
            left = max(0, state - max_step)
            right = min(states, state + max_step + 1)
            candidates = previous[left:right]
            if not np.isfinite(candidates).any():
                continue
            previous_states = np.arange(left, right)
            values = candidates - penalty * (previous_states - state) ** 2
            local_best = int(np.argmax(values))
            best_previous = int(previous_states[local_best])
            current[state] = values[local_best] + emission[state, column]
            back[local_column, state] = best_previous
        previous = current
    final_state = int(np.argmax(previous))
    path = np.full(columns, int(start_state), np.int32)
    state = final_state
    for local_column in range(length - 1, -1, -1):
        path[int(start_column) + local_column] = state
        predecessor = int(back[local_column, state])
        if local_column > 0 and predecessor < 0:
            predecessor = state
        state = predecessor
    return path


def _column_prediction(record, panel, offset):
    offset = np.asarray(offset, np.float64).copy()
    known = panel["known_mask"] > 0.5
    offset[known] = panel["target_offset"][known]
    column_tvt = float(panel["anchor_tvt"]) + offset
    column_s = column_tvt + record["column_z"]
    return (
        np.interp(record["eval_md"], record["column_md"], column_s)
        - record["eval_z"]
    )


checkpoint_path = Path(MODEL_DIR) / f"seed_{SEED}.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
audit_model = FullWellGlobalUNet(input_channels).to(device)
audit_model.load_state_dict(checkpoint["model"])
audit_model.eval()
offsets_tensor = torch.from_numpy(STATE_OFFSETS).to(device)

variant_errors = {}
variant_well_rows = {}
row_frames = []
started = time.time()
for index, record in enumerate(tqdm(valid_store.records, desc="decode full-well logits")):
    panel = _compose_panel(record, tag="official", mode="full")
    values = torch.from_numpy(panel["input"])[None].to(device)
    state_valid_tensor = torch.from_numpy(panel["state_valid"])[None].to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        logits_tensor = audit_model(values, offsets_tensor)
    logits = logits_tensor[0].float().cpu().numpy()
    state_valid = panel["state_valid"]
    logits[~state_valid] = -30.0
    truth = record["eval_truth"].astype(np.float64)
    local_predictions = {}

    for temperature in MEAN_TEMPERATURES:
        scaled = logits / float(temperature)
        scaled -= np.max(scaled, axis=0, keepdims=True)
        probability = np.exp(scaled)
        probability *= state_valid
        probability /= np.maximum(np.sum(probability, axis=0, keepdims=True), 1e-12)
        offset = np.sum(probability * STATE_OFFSETS[:, None], axis=0)
        name = f"mean_t{temperature:.2f}"
        local_predictions[name] = _column_prediction(record, panel, offset)

    argmax_state = np.argmax(logits, axis=0)
    local_predictions["argmax"] = _column_prediction(
        record, panel, STATE_OFFSETS[argmax_state]
    )

    start_column = int(np.argmax(panel["eval_mask"] > 0.5))
    start_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    for max_step, penalty in DP_CONFIGS:
        state_path = _viterbi(
            logits,
            state_valid,
            start_column,
            start_state,
            int(max_step),
            float(penalty),
        )
        name = f"dp_s{max_step}_p{penalty:.2f}"
        local_predictions[name] = _column_prediction(
            record, panel, STATE_OFFSETS[state_path]
        )

    if index == 0:
        for name in local_predictions:
            variant_errors[name] = []
            variant_well_rows[name] = []
    output = {
        "well": record["well"],
        "row_idx": record["eval_rows"],
        "truth": truth,
    }
    for name, prediction in local_predictions.items():
        error = prediction - truth
        variant_errors[name].append(error)
        variant_well_rows[name].append(
            {
                "well": record["well"],
                "rows": len(error),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
        output[name] = prediction
    row_frames.append(pd.DataFrame(output))

score_rows = []
for name, chunks in variant_errors.items():
    errors = np.concatenate(chunks)
    by_well = pd.DataFrame(variant_well_rows[name])
    score_rows.append(
        {
            "decoder": name,
            "rows": len(errors),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "median_well_rmse": float(by_well["rmse"].median()),
            "p95_well_rmse": float(by_well["rmse"].quantile(0.95)),
        }
    )
scores = pd.DataFrame(score_rows).sort_values("rmse")
scores.to_csv(AUDIT_DIR / "decoder_scores.csv", index=False)
pd.concat(row_frames, ignore_index=True).to_pickle(AUDIT_DIR / "decoder_predictions.pkl")
best = scores.iloc[0]
strict_pass = bool(float(best["rmse"]) < 10.0)
verdict = {
    "experiment": "fullwell_global_tvt_decode_audit_v0",
    "checkpoint": str(checkpoint_path),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "checkpoint_metrics": checkpoint["metrics"],
    "best_decoder": str(best["decoder"]),
    "best_rmse": float(best["rmse"]),
    "best_median_well_rmse": float(best["median_well_rmse"]),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> decoder was the bottleneck; rebuild training around structured decoding"
        if strict_pass
        else "FAIL -> posterior contains no usable full-well path; stop this representation"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(AUDIT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nDECODER SCORES")
print(scores.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", AUDIT_DIR)

