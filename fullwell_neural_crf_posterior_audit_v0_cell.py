"""Forward-backward posterior audit for the full-well neural CRF.

Run in the same kernel after interrupting neural-CRF training.  The best saved
checkpoint is loaded; no training is performed.  The audit computes exact
linear-chain marginals for a small frozen temperature/transition/centre grid,
then reports both the full posterior mean and conservative anchor doses.
"""

from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


REQUIRED_GLOBALS = (
    "OUT_DIR",
    "MODEL_DIR",
    "SEED",
    "STATE_OFFSETS",
    "MAX_TRANSITION_STATES",
    "NEGATIVE_SCORE",
    "FullWellNeuralCRF",
    "valid_store",
    "_compose_alignment",
    "device",
)
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run fullwell_neural_crf_onefold_v0 through checkpoint creation first; "
        "missing: " + repr(missing)
    )

AUDIT_DIR = Path(OUT_DIR) / "posterior_audit_v0"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE_AUDIT = 8
CONFIGS = (
    {"name": "t1_m1_c0", "temperature": 1.0, "transition_multiplier": 1.0, "centre_precision": 0.00},
    {"name": "t1p5_m1_c0", "temperature": 1.5, "transition_multiplier": 1.0, "centre_precision": 0.00},
    {"name": "t2_m1_c0", "temperature": 2.0, "transition_multiplier": 1.0, "centre_precision": 0.00},
    {"name": "t1p5_m2_c0", "temperature": 1.5, "transition_multiplier": 2.0, "centre_precision": 0.00},
    {"name": "t2_m2_c0", "temperature": 2.0, "transition_multiplier": 2.0, "centre_precision": 0.00},
    {"name": "t1p5_m4_c0", "temperature": 1.5, "transition_multiplier": 4.0, "centre_precision": 0.00},
    {"name": "t2_m4_c0", "temperature": 2.0, "transition_multiplier": 4.0, "centre_precision": 0.00},
    {"name": "t1p5_m4_c002", "temperature": 1.5, "transition_multiplier": 4.0, "centre_precision": 0.02},
    {"name": "t2_m4_c005", "temperature": 2.0, "transition_multiplier": 4.0, "centre_precision": 0.05},
    {"name": "t2_m8_c005", "temperature": 2.0, "transition_multiplier": 8.0, "centre_precision": 0.05},
)
DOSES = (0.0, 0.25, 0.50, 0.75, 1.00)


def _shift_forward(alpha, delta):
    if delta == 0:
        return alpha
    if delta > 0:
        return F.pad(alpha[:, :-delta], (delta, 0), value=NEGATIVE_SCORE)
    return F.pad(alpha[:, -delta:], (0, -delta), value=NEGATIVE_SCORE)


def _shift_backward(values, delta):
    """Map a score at current state = previous state + delta to previous state."""
    if delta == 0:
        return values
    if delta > 0:
        return F.pad(values[:, delta:], (0, delta), value=NEGATIVE_SCORE)
    return F.pad(values[:, :delta], (-delta, 0), value=NEGATIVE_SCORE)


@torch.no_grad()
def _posterior_offsets(
    emission,
    eval_mask,
    learned_transition_penalty,
    temperature,
    transition_multiplier,
    centre_precision,
):
    emission = emission.float()
    eval_mask = eval_mask.bool()
    batch, states, columns = emission.shape
    offsets = torch.as_tensor(STATE_OFFSETS, dtype=torch.float32, device=emission.device)
    normalised_offset = offsets / float(np.max(np.abs(STATE_OFFSETS)))
    score = emission / float(temperature)
    if centre_precision > 0:
        score = score - float(centre_precision) * normalised_offset[None, :, None].square()
    penalty = (
        float(learned_transition_penalty)
        * float(transition_multiplier)
        / float(temperature)
    )
    deltas = list(range(-MAX_TRANSITION_STATES, MAX_TRANSITION_STATES + 1))
    transition = score.new_tensor(
        [-penalty * float(delta * delta) for delta in deltas]
    )

    zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    alpha = torch.full(
        (batch, states), NEGATIVE_SCORE, dtype=score.dtype, device=score.device
    )
    alpha[:, zero_state] = 0.0
    alpha_columns = []
    for column in range(columns):
        candidates = torch.stack(
            [
                _shift_forward(alpha, delta) + transition[index]
                for index, delta in enumerate(deltas)
            ],
            dim=0,
        )
        advanced = score[:, :, column] + torch.logsumexp(candidates, dim=0)
        alpha = torch.where(eval_mask[:, column, None], advanced, alpha)
        alpha_columns.append(alpha)
    alpha_volume = torch.stack(alpha_columns, dim=2)
    log_partition = torch.logsumexp(alpha, dim=1)

    beta = torch.zeros((batch, states), dtype=score.dtype, device=score.device)
    posterior_offset = torch.zeros(
        (batch, columns), dtype=score.dtype, device=score.device
    )
    for column in range(columns - 1, -1, -1):
        log_marginal = alpha_volume[:, :, column] + beta - log_partition[:, None]
        probability = torch.softmax(log_marginal, dim=1)
        posterior_offset[:, column] = (probability * offsets[None]).sum(dim=1)
        current = score[:, :, column] + beta
        candidates = torch.stack(
            [
                _shift_backward(current, delta) + transition[index]
                for index, delta in enumerate(deltas)
            ],
            dim=0,
        )
        previous_beta = torch.logsumexp(candidates, dim=0)
        beta = torch.where(eval_mask[:, column, None], previous_beta, beta)
    return posterior_offset


checkpoint_path = Path(MODEL_DIR) / f"seed_{SEED}.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
posterior_model = FullWellNeuralCRF().to(device)
posterior_model.load_state_dict(checkpoint["model"])
posterior_model.eval()
learned_transition = float(posterior_model.transition_penalty().detach().cpu())

panels = []
emission_chunks = []
started = time.time()
for start in tqdm(
    range(0, len(valid_store.records), BATCH_SIZE_AUDIT),
    desc="cache neural-CRF emissions",
):
    records = valid_store.records[start : start + BATCH_SIZE_AUDIT]
    batch_panels = [_compose_alignment(record, tag="official") for record in records]
    horizontal = torch.from_numpy(
        np.stack([panel["horizontal"] for panel in batch_panels])
    ).to(device)
    reference = torch.from_numpy(
        np.stack([panel["reference"] for panel in batch_panels])
    ).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        emission = posterior_model(horizontal, reference)
    emission_chunks.append(emission.half().cpu())
    panels.extend(batch_panels)
emission_cache = torch.cat(emission_chunks, dim=0)
del emission_chunks
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

truth_chunks = [record["eval_truth"].astype(np.float64) for record in valid_store.records]
anchor_chunks = [
    np.full(len(truth), panel["anchor_tvt"], np.float64)
    for truth, panel in zip(truth_chunks, panels)
]
truth_all = np.concatenate(truth_chunks)
anchor_all = np.concatenate(anchor_chunks)


def _row_predictions(offsets_batch, start_index):
    predictions = []
    for local_index in range(len(offsets_batch)):
        record = valid_store.records[start_index + local_index]
        panel = panels[start_index + local_index]
        offset = offsets_batch[local_index].astype(np.float64)
        offset[panel["known_mask"]] = panel["target_offset"][panel["known_mask"]]
        column_tvt = float(panel["anchor_tvt"]) + offset
        column_s = column_tvt + panel["column_z"]
        prediction = (
            np.interp(record["eval_md"], panel["column_md"], column_s)
            - record["eval_z"]
        )
        predictions.append(prediction)
    return predictions


def _score(prediction_chunks):
    prediction = np.concatenate(prediction_chunks)
    error = prediction - truth_all
    well_rmse = np.asarray(
        [
            np.sqrt(np.mean((candidate - truth) ** 2))
            for candidate, truth in zip(prediction_chunks, truth_chunks)
        ]
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }, prediction


score_rows = []
best_rmse = np.inf
best_prediction = None
best_description = None
for config in CONFIGS:
    posterior_chunks = []
    for start in range(0, len(valid_store.records), BATCH_SIZE_AUDIT):
        stop = min(start + BATCH_SIZE_AUDIT, len(valid_store.records))
        emission = emission_cache[start:stop].to(device)
        eval_mask = torch.from_numpy(
            np.stack([panel["eval_mask"] for panel in panels[start:stop]])
        ).to(device)
        offsets = _posterior_offsets(
            emission,
            eval_mask,
            learned_transition,
            config["temperature"],
            config["transition_multiplier"],
            config["centre_precision"],
        ).cpu().numpy()
        posterior_chunks.extend(_row_predictions(offsets, start))
    posterior_all = np.concatenate(posterior_chunks)
    for dose in DOSES:
        prediction_all = anchor_all + float(dose) * (posterior_all - anchor_all)
        prediction_chunks = []
        cursor = 0
        for truth in truth_chunks:
            prediction_chunks.append(prediction_all[cursor : cursor + len(truth)])
            cursor += len(truth)
        metrics, flat_prediction = _score(prediction_chunks)
        row = {
            **config,
            "dose": float(dose),
            **metrics,
            "gain_vs_anchor": float(
                np.sqrt(np.mean((anchor_all - truth_all) ** 2)) - metrics["rmse"]
            ),
        }
        score_rows.append(row)
        if dose > 0 and metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_prediction = flat_prediction.copy()
            best_description = {
                "config": config["name"],
                "dose": float(dose),
                **metrics,
            }
    print(
        "POSTERIOR CONFIG",
        config["name"],
        min(
            row["rmse"]
            for row in score_rows
            if row["name"] == config["name"] and row["dose"] > 0
        ),
        flush=True,
    )

scores = pd.DataFrame(score_rows).sort_values("rmse")
scores.to_csv(AUDIT_DIR / "posterior_scores.csv", index=False)
row_metadata = pd.concat(
    [
        pd.DataFrame(
            {
                "well": record["well"],
                "row_idx": record["eval_rows"],
                "truth": truth,
            }
        )
        for record, truth in zip(valid_store.records, truth_chunks)
    ],
    ignore_index=True,
)
row_metadata["prediction"] = best_prediction
row_metadata.to_pickle(AUDIT_DIR / "best_posterior_predictions.pkl")

anchor_rmse = float(np.sqrt(np.mean((anchor_all - truth_all) ** 2)))
decoder_pass = bool(best_rmse < 10.0)
architecture_pass = bool(best_rmse <= 7.0)
target_pass = bool(best_rmse <= 5.5)
verdict = {
    "experiment": "fullwell_neural_crf_posterior_audit_v0",
    "checkpoint": str(checkpoint_path),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "checkpoint_viterbi_metrics": checkpoint["metrics"],
    "learned_transition_penalty": learned_transition,
    "anchor_rmse": anchor_rmse,
    "best_nonzero_posterior": best_description,
    "optimistic_gain_vs_anchor": float(anchor_rmse - best_rmse),
    "decoder_pass": decoder_pass,
    "architecture_pass": architecture_pass,
    "target_5p5_pass": target_pass,
    "strict_pass": architecture_pass,
    "next": (
        "TARGET PASS -> verify decoder settings nested and run second seed"
        if target_pass
        else (
            "DECODER PASS -> rebuild training/checkpoint selection around posterior mean"
            if decoder_pass
            else "FAIL -> close S1 and proceed to S2 meta-adapted CorrNet"
        )
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(AUDIT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nPOSTERIOR SCORES TOP 25")
print(scores.head(25).to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", AUDIT_DIR)
