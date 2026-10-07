"""Train a complete-well 2D cost-map sequence encoder from raw evidence.

V33 learned useful local pair rankings, but the first-order Markov decoder
accumulated confident local cycle skips.  This model consumes the complete
[node, candidate-state, pair-feature] tensor and refines it with dilated 2D
convolutions across both measured-distance and candidate-depth axes.

Only original horizontal/typewell data, Z, MD, and the visible prefix are used.
Folds 4/5 train, Fold 3 selects the epoch, Fold 2 confirms once, and Fold 1 is
untouched.  No historical prediction, neighbour, CP, Gold, or spatial input is
read.  The output is direct complete-well MAP/posterior-mean decoding; V30's
Markov transition is not used.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT = "raw_costmap_sequence_encoder_v36"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V30_VERDICT = V30_DIR / "verdict.json"
SOURCE_CANDIDATES = (
    ROOT / "raw_pairwise_emission_stochastic_v33_cell.py",
    Path.cwd() / "raw_pairwise_emission_stochastic_v33_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload raw_pairwise_emission_stochastic_v33_cell.py")

SEED = 20260726
HALF_WIDTH_FT = 160.0
STATE_STEP_FT = 4.0
OFFSETS = np.arange(-HALF_WIDTH_FT, HALF_WIDTH_FT + 0.1, STATE_STEP_FT).astype(np.float32)
ZERO_STATE = int(np.argmin(np.abs(OFFSETS)))
EVIDENCE_FEATURES = 57  # 56 V33 pair features + handcrafted Cauchy cost.
COORD_FEATURES = 3
MAX_EPOCHS = 24
PATIENCE = 6
BATCH_WELLS = 4
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-4
SOFT_TARGET_SIGMA_FT = 6.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_pairwise_v33_sequence_definitions"
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


M = _bootstrap(SOURCE_PATH)


def _load_records(wells, label):
    records = []
    failures = []
    for number, well in enumerate(wells, 1):
        try:
            records.append(M.V._load_record(well))
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"V36 load {label} {number}/{len(wells)} failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} {label} record failures: {failures[:5]}")
    return records


def _sequence_payload(record, donor=None):
    rows, distance, state_surface, pair, target, covered, true_offset = M._pair_features(
        record, donor=donor
    )
    _, _, _, handcrafted = M.V._emission(
        record, PRIOR, HAND_CONFIG, donor=donor
    )
    handcrafted = handcrafted.astype(np.float32) / float(HAND_CONFIG["temperature"])
    if pair.shape[:2] != handcrafted.shape:
        raise RuntimeError(
            f"{record['well']}: pair/handcrafted node-state mismatch "
            f"{pair.shape[:2]} != {handcrafted.shape}"
        )
    if pair.shape[2] + 1 != EVIDENCE_FEATURES:
        raise RuntimeError(
            f"{record['well']}: expected {EVIDENCE_FEATURES - 1} V33 features, "
            f"received {pair.shape[2]}"
        )
    evidence = np.concatenate((pair.astype(np.float32), handcrafted[:, :, None]), axis=2)
    nodes, states, _ = evidence.shape
    offset_coord = np.broadcast_to(
        (OFFSETS / HALF_WIDTH_FT)[None, :, None], (nodes, states, 1)
    )
    progress = distance / max(float(distance[-1]), 1.0)
    progress_coord = np.broadcast_to(progress[:, None, None], (nodes, states, 1))
    anchor = np.zeros((nodes, states, 1), np.float32)
    anchor[0, ZERO_STATE, 0] = 1.0
    features = np.concatenate((evidence, offset_coord, progress_coord, anchor), axis=2)
    return {
        "well": record["well"],
        "rows": rows,
        "distance": distance.astype(np.float32),
        "state_surface": state_surface.astype(np.float32),
        "features": features.astype(np.float16),
        "target": target.astype(np.int64),
        "covered": covered.astype(bool),
        "true_offset": true_offset.astype(np.float32),
    }


def _build_payload(records, label, donors=None):
    output = []
    for number, record in enumerate(records, 1):
        donor = None if donors is None else donors[record["well"]]
        output.append(_sequence_payload(record, donor=donor))
        if number % 25 == 0 or number == len(records):
            print(f"V36 features {label} {number}/{len(records)}", flush=True)
    return output


def _normalisation(payload):
    total = np.zeros(EVIDENCE_FEATURES, np.float64)
    square = np.zeros(EVIDENCE_FEATURES, np.float64)
    count = 0
    for item in payload:
        values = item["features"][:, :, :EVIDENCE_FEATURES].astype(np.float32)
        flat = values.reshape(-1, EVIDENCE_FEATURES)
        total += np.sum(flat, axis=0, dtype=np.float64)
        square += np.sum(flat * flat, axis=0, dtype=np.float64)
        count += len(flat)
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _normalise_features(features, mean, scale, no_gr=False):
    output = features.astype(np.float32, copy=True)
    if no_gr:
        output[:, :, :EVIDENCE_FEATURES] = mean[None, None, :]
    output[:, :, :EVIDENCE_FEATURES] = (
        output[:, :, :EVIDENCE_FEATURES] - mean[None, None, :]
    ) / scale[None, None, :]
    return output


def _batch(items, mean, scale, device, no_gr=False):
    batch = len(items)
    max_nodes = max(len(item["rows"]) for item in items)
    states = len(OFFSETS)
    features = np.zeros((batch, max_nodes, states, EVIDENCE_FEATURES + COORD_FEATURES), np.float32)
    target = np.zeros((batch, max_nodes), np.int64)
    true_offset = np.zeros((batch, max_nodes), np.float32)
    mask = np.zeros((batch, max_nodes), bool)
    for index, item in enumerate(items):
        nodes = len(item["rows"])
        features[index, :nodes] = _normalise_features(item["features"], mean, scale, no_gr=no_gr)
        target[index, :nodes] = item["target"]
        true_offset[index, :nodes] = item["true_offset"]
        mask[index, :nodes] = item["covered"]
        mask[index, 0] = False
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(true_offset).to(device),
        torch.from_numpy(mask).to(device),
    )


class ResidualCostBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=(3, 5),
                padding=(dilation, 2), dilation=(dilation, 1),
            ),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Dropout2d(0.08),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.network(x))


class CostMapSequenceEncoder(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        channels = 24
        self.candidate = nn.Sequential(
            nn.Linear(input_features, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualCostBlock(channels, 1),
            ResidualCostBlock(channels, 2),
            ResidualCostBlock(channels, 4),
            ResidualCostBlock(channels, 8),
        )
        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features):
        # [B,T,S,F] -> [B,T,S] logits.
        x = self.candidate(features).permute(0, 3, 1, 2)
        x = self.blocks(x)
        return self.output(x).squeeze(1)


def _soft_target(true_offset, device):
    candidates = torch.as_tensor(OFFSETS, dtype=torch.float32, device=device)
    probability = torch.exp(
        -0.5 * ((candidates[None, None, :] - true_offset[:, :, None]) / SOFT_TARGET_SIGMA_FT) ** 2
    )
    return probability / probability.sum(dim=2, keepdim=True)


def _loss(model, batch):
    features, target, true_offset, mask = batch
    logits = model(features)
    probability = _soft_target(true_offset, features.device)
    log_probability = F.log_softmax(logits, dim=2)
    ce = -(probability * log_probability).sum(dim=2)
    ce = ce[mask].mean()
    posterior = F.softmax(logits, dim=2)
    offsets = torch.as_tensor(OFFSETS, dtype=torch.float32, device=features.device)
    expected = torch.sum(posterior * offsets[None, None, :], dim=2)
    regression = F.smooth_l1_loss(expected[mask] / 16.0, true_offset[mask] / 16.0)
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if pair_mask.any():
        predicted_change = expected[:, 1:] - expected[:, :-1]
        true_change = true_offset[:, 1:] - true_offset[:, :-1]
        change = F.smooth_l1_loss(
            predicted_change[pair_mask] / 8.0,
            true_change[pair_mask] / 8.0,
        )
    else:
        change = regression.new_zeros(())
    return ce + 0.20 * regression + 0.10 * change, {
        "ce": float(ce.detach().cpu()),
        "regression": float(regression.detach().cpu()),
        "change": float(change.detach().cpu()),
    }


def _train_epoch(model, optimizer, scaler, payload, mean, scale, device, rng):
    model.train()
    order = np.arange(len(payload))
    rng.shuffle(order)
    reports = []
    for start in range(0, len(order), BATCH_WELLS):
        items = [payload[int(index)] for index in order[start:start + BATCH_WELLS]]
        batch = _batch(items, mean, scale, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, parts = _loss(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        reports.append({"loss": float(loss.detach().cpu()), **parts})
    return {key: float(np.mean([row[key] for row in reports])) for key in reports[0]}


@torch.no_grad()
def _predict_item(model, item, mean, scale, device, no_gr=False):
    model.eval()
    features, _, _, _ = _batch([item], mean, scale, device, no_gr=no_gr)
    logits = model(features)[0, :len(item["rows"])]
    logits[0] = -1e4
    logits[0, ZERO_STATE] = 0.0
    posterior = F.softmax(logits, dim=1)
    map_state = torch.argmax(logits, dim=1).cpu().numpy()
    probability = posterior.cpu().numpy()
    mean_surface = np.sum(probability * item["state_surface"], axis=1)
    map_surface = item["state_surface"][np.arange(len(map_state)), map_state]
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1) / math.log(len(OFFSETS))
    offset_mean = probability @ OFFSETS
    offset_std = np.sqrt(np.maximum(probability @ (OFFSETS ** 2) - offset_mean ** 2, 0.0))
    return {"posterior_mean": mean_surface, "map": map_surface}, {
        "mean_entropy": float(np.mean(entropy[1:])),
        "mean_offset_std": float(np.mean(offset_std[1:])),
    }


def _evaluate(model, records, payload, mean, scale, device, variant, no_gr=False):
    rows = []
    for record, item in zip(records, payload):
        surfaces, diagnostics = _predict_item(model, item, mean, scale, device, no_gr=no_gr)
        evaluation = record["evaluation"]
        distance = record["md"][evaluation] - record["anchor_md"]
        truth = record["truth"][evaluation]
        for estimator, surface in surfaces.items():
            interpolated = np.interp(
                record["md"][evaluation], record["md"][item["rows"]], surface
            )
            prediction = interpolated - record["z"][evaluation]
            error = prediction - truth
            far = distance > 2000.0
            rows.append({
                "well": record["well"], "variant": variant, "estimator": estimator,
                "rows": int(len(error)), "sse": float(np.dot(error, error)),
                "sae": float(np.sum(np.abs(error))), "far_rows": int(far.sum()),
                "far_sse": float(np.dot(error[far], error[far])) if far.any() else 0.0,
                **diagnostics,
            })
    frame = pd.DataFrame(rows)
    return frame, M.V._summary(frame)


started = time.time()
for required in (V30_VERDICT,):
    if not required.exists():
        raise FileNotFoundError(required)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v30 = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
PRIOR = v30["outer_train_prior"]
HAND_CONFIG = dict(v30["selected_config"])
M.PRIOR = PRIOR
folds = pd.read_csv(V30_DIR / "folds.csv", dtype={"well": str})
train_wells = sorted(folds.loc[folds["fold"].isin((4, 5)), "well"].astype(str))
valid_wells = sorted(folds.loc[folds["fold"].eq(3), "well"].astype(str))
application_wells = sorted(folds.loc[folds["fold"].eq(2), "well"].astype(str))
train_records = _load_records(train_wells, "train45")
valid_records = _load_records(valid_wells, "fold3")
application_records = _load_records(application_wells, "fold2")
train_payload = _build_payload(train_records, "train45")
valid_payload = _build_payload(valid_records, "fold3")
application_payload = _build_payload(application_records, "fold2")
feature_mean, feature_scale = _normalisation(train_payload)
np.savez_compressed(OUT_DIR / "normalisation.npz", mean=feature_mean, scale=feature_scale)

model = CostMapSequenceEncoder(EVIDENCE_FEATURES + COORD_FEATURES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
rng = np.random.default_rng(SEED)
checkpoint = MODEL_DIR / "best_sequence_encoder.pt"
history = []
best_rmse = np.inf
best_epoch = 0
bad_epochs = 0
for epoch in range(1, MAX_EPOCHS + 1):
    training = _train_epoch(
        model, optimizer, scaler, train_payload, feature_mean, feature_scale, device, rng
    )
    valid_rows, valid_summary = _evaluate(
        model, valid_records, valid_payload, feature_mean, feature_scale,
        device, f"fold3_epoch{epoch}",
    )
    valid_posterior = valid_summary[valid_summary["estimator"].eq("posterior_mean")].iloc[0]
    report = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        "valid_rmse": float(valid_posterior["rmse"]),
        "valid_far_rmse": float(valid_posterior["far_gt2000_rmse"]),
        "valid_median_well_rmse": float(valid_posterior["median_well_rmse"]),
    }
    history.append(report)
    print("V36 EPOCH", report, flush=True)
    if report["valid_rmse"] < best_rmse - 1e-4:
        best_rmse = report["valid_rmse"]
        best_epoch = epoch
        bad_epochs = 0
        torch.save({
            "model": model.state_dict(), "epoch": epoch,
            "mean": feature_mean, "scale": feature_scale,
            "features": EVIDENCE_FEATURES + COORD_FEATURES,
        }, checkpoint)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
state = torch.load(checkpoint, map_location=device, weights_only=False)
model.load_state_dict(state["model"])

metric_parts = []
summary_parts = []
for fold, records, payload, variant in (
    (3, valid_records, valid_payload, "fold3_full"),
    (2, application_records, application_payload, "fold2_full"),
):
    bywell, summary = _evaluate(
        model, records, payload, feature_mean, feature_scale, device, variant
    )
    bywell["fold"] = fold
    summary["fold"] = fold
    metric_parts.append(bywell)
    summary_parts.append(summary)

no_gr_rows, no_gr_summary = _evaluate(
    model, application_records, application_payload, feature_mean, feature_scale,
    device, "fold2_no_gr", no_gr=True,
)
no_gr_rows["fold"] = 2
no_gr_summary["fold"] = 2
metric_parts.append(no_gr_rows)
summary_parts.append(no_gr_summary)

donor_order = application_records[1:] + application_records[:1]
donors = {record["well"]: donor for record, donor in zip(application_records, donor_order)}
shuffled_payload = _build_payload(application_records, "fold2_shuffled", donors=donors)
shuffle_rows, shuffle_summary = _evaluate(
    model, application_records, shuffled_payload, feature_mean, feature_scale,
    device, "fold2_shuffled_typewell",
)
shuffle_rows["fold"] = 2
shuffle_summary["fold"] = 2
metric_parts.append(shuffle_rows)
summary_parts.append(shuffle_summary)

bywell_frame = pd.concat(metric_parts, ignore_index=True)
summary_frame = pd.concat(summary_parts, ignore_index=True)
bywell_frame.to_csv(OUT_DIR / "bywell.csv", index=False)
summary_frame.to_csv(OUT_DIR / "summary.csv", index=False)
print("\nV36 SUMMARY")
print(summary_frame.to_string(index=False), flush=True)

def _row(variant, estimator="posterior_mean"):
    return summary_frame[
        summary_frame["variant"].eq(variant)
        & summary_frame["estimator"].eq(estimator)
    ].iloc[0]

fold3 = _row("fold3_full")
fold2 = _row("fold2_full")
no_gr = _row("fold2_no_gr")
shuffled = _row("fold2_shuffled_typewell")
gates = {
    "fold3_rmse_le_15": bool(fold3["rmse"] <= 15.0),
    "fold2_rmse_le_18": bool(fold2["rmse"] <= 18.0),
    "fold2_far_rmse_le_22": bool(fold2["far_gt2000_rmse"] <= 22.0),
    "no_gr_degradation_ge_5": bool(no_gr["rmse"] - fold2["rmse"] >= 5.0),
    "shuffled_typewell_degradation_ge_5": bool(shuffled["rmse"] - fold2["rmse"] >= 5.0),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_complete_well_training_Fold3_epoch_selection_Fold2_confirmation",
    "sample_unit": "one complete physical well",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "architecture": "candidate MLP + four dilated 2D cost-map residual blocks",
    "states": int(len(OFFSETS)),
    "input_features": EVIDENCE_FEATURES + COORD_FEATURES,
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "selected_epoch": int(best_epoch),
    "fold3": fold3.to_dict(),
    "fold2": fold2.to_dict(),
    "controls": {
        "no_gr": no_gr.to_dict(),
        "shuffled_typewell": shuffled.to_dict(),
        "no_gr_degradation": float(no_gr["rmse"] - fold2["rmse"]),
        "shuffled_typewell_degradation": float(shuffled["rmse"] - fold2["rmse"]),
    },
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "checkpoint": str(checkpoint),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain complete-well encoder and test five-fold variance"
        if all(gates.values())
        else "FAIL -> raw pair cost-map remains insufficient; stop this standalone branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
