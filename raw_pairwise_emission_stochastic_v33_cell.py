"""Learn a raw candidate-conditioned emission for the repaired stochastic decoder.

The model scores one (horizontal node, candidate typewell depth) pair.  It does
not receive candidate offset/index, historical predictions, neighbours, CP,
Gold, or spatial features.  Local horizontal/typewell agreement is summarized
over fixed 32--256 ft windows.  The learned score replaces only V30's unary
emission; the V32 +/-160 ft state lattice and transition are frozen.

Protocol: Folds 4/5 train, Fold 3 early-stops, Fold 2 confirms once.  Fold 1 is
untouched.  Candidate order carries no information because the same scalar MLP
is independently applied to every candidate.
"""

from __future__ import annotations

import json
import math
import os
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


EXPERIMENT = "raw_pairwise_emission_stochastic_v33"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V30_VERDICT = V30_DIR / "verdict.json"
SOURCE_CANDIDATES = (
    ROOT / "raw_stochastic_alignment_standalone_v30_cell.py",
    Path.cwd() / "raw_stochastic_alignment_standalone_v30_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload raw_stochastic_alignment_standalone_v30_cell.py")

SEED = 20260726
HALF_WIDTH_FT = 160.0
STATE_STEP_FT = 4.0
OFFSETS = np.arange(-HALF_WIDTH_FT, HALF_WIDTH_FT + 0.1, STATE_STEP_FT)
RADII_FT = (32.0, 64.0, 128.0, 256.0)
SAMPLES_PER_WINDOW = 17
PAIR_CHANNELS = (0, 2)
SOFT_TARGET_SIGMA_FT = 6.0
MAX_EPOCHS = 20
PATIENCE = 5
BATCH_NODES = 192
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_stochastic_v30_pairwise_definitions"
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
V.OFFSETS_FT = OFFSETS.astype(np.float64)
V.ZERO_STATE = int(np.argmin(np.abs(V.OFFSETS_FT)))


def _load_records(wells, label):
    records = []
    failures = []
    for number, well in enumerate(wells, 1):
        try:
            records.append(V._load_record(well))
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"V33 load {label} {number}/{len(wells)} failures={len(failures)}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} {label} failures: {failures[:5]}")
    return records


def _safe_corr(observed, expected):
    # observed: [samples], expected: [states, samples]
    x = observed - np.mean(observed)
    y = expected - np.mean(expected, axis=1, keepdims=True)
    numerator = np.sum(y * x[None, :], axis=1)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)


def _pair_features(record, donor=None):
    reference_record = record if donor is None else donor
    rows = V._node_rows(record)
    md_nodes = record["md"][rows]
    distance = md_nodes - record["anchor_md"]
    rate = float(np.clip(
        record["prefix_surface_rate"], PRIOR["rate_low"], PRIOR["rate_high"]
    ))
    centre_surface = record["anchor_surface"] + rate * distance
    state_surface = centre_surface[:, None] + OFFSETS[None, :]
    axis = reference_record["tw_tvt"]
    reference = reference_record["tw_channels"]
    states = len(OFFSETS)
    feature_rows = []

    for node, (row, md_node) in enumerate(zip(rows, md_nodes)):
        pieces = []
        # Signed and robust point residuals at all native multiscale channels.
        for channel in range(record["horizontal_channels"].shape[0]):
            candidate_tvt = state_surface[node] - record["z"][row]
            expected = np.interp(candidate_tvt, axis, reference[channel], left=np.nan, right=np.nan)
            residual = (record["horizontal_channels"][channel, row] - expected) / max(record["gamma"], 1e-6)
            residual = np.nan_to_num(residual, nan=8.0, posinf=8.0, neginf=-8.0)
            pieces.extend((residual[:, None], np.log1p(np.minimum(residual * residual, 1e3))[:, None]))

        for radius in RADII_FT:
            sample_md = md_node + np.linspace(-radius, radius, SAMPLES_PER_WINDOW)
            sample_md = np.clip(sample_md, record["md"][0], record["md"][-1])
            sample_z = np.interp(sample_md, record["md"], record["z"])
            delta_md = sample_md - md_node
            candidate_surface = state_surface[node, :, None] + rate * delta_md[None, :]
            candidate_tvt = candidate_surface - sample_z[None, :]
            for channel in PAIR_CHANNELS:
                observed = np.interp(
                    sample_md, record["md"], record["horizontal_channels"][channel]
                )
                expected = np.interp(
                    candidate_tvt.ravel(), axis, reference[channel], left=np.nan, right=np.nan
                ).reshape(states, SAMPLES_PER_WINDOW)
                valid = np.isfinite(expected)
                valid_count = np.sum(valid, axis=1)
                fill = np.divide(
                    np.nansum(expected, axis=1),
                    valid_count,
                    out=np.zeros(states, dtype=np.float64),
                    where=valid_count > 0,
                )
                expected = np.where(valid, expected, fill[:, None])
                residual = (observed[None, :] - expected) / max(record["gamma"], 1e-6)
                abs_residual = np.abs(residual)
                pieces.extend((
                    np.mean(np.log1p(np.minimum(residual * residual, 1e3)), axis=1)[:, None],
                    np.mean(abs_residual, axis=1)[:, None],
                    np.median(residual, axis=1)[:, None],
                    np.std(residual, axis=1)[:, None],
                    _safe_corr(observed, expected)[:, None],
                    np.mean(valid, axis=1)[:, None],
                ))
        feature_rows.append(np.concatenate(pieces, axis=1).astype(np.float32))

    truth_surface = record["truth"][rows] + record["z"][rows]
    true_offset = truth_surface - centre_surface
    target = np.argmin(np.abs(OFFSETS[None, :] - true_offset[:, None]), axis=1).astype(np.int64)
    covered = np.abs(true_offset) <= HALF_WIDTH_FT + 1e-9
    return rows, distance, state_surface, np.stack(feature_rows), target, covered, true_offset


def _build_arrays(records, label):
    features = []
    targets = []
    covered = []
    owners = []
    payload = {}
    for number, record in enumerate(records, 1):
        rows, distance, state_surface, x, y, keep, true_offset = _pair_features(record)
        # Anchor state is fixed by the decoder and is not a training example.
        features.append(x[1:].astype(np.float16))
        targets.append(y[1:])
        covered.append(keep[1:])
        owners.extend([record["well"]] * (len(rows) - 1))
        payload[record["well"]] = {
            "rows": rows,
            "distance": distance,
            "state_surface": state_surface,
            "features": x.astype(np.float16),
            "target": y,
            "covered": keep,
            "true_offset": true_offset,
        }
        if number % 25 == 0 or number == len(records):
            print(f"V33 features {label} {number}/{len(records)}", flush=True)
    return (
        np.concatenate(features),
        np.concatenate(targets),
        np.concatenate(covered),
        np.asarray(owners),
        payload,
    )


class PairEmission(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def _normalise(x, mean, scale):
    return (x.astype(np.float32) - mean[None, None, :]) / scale[None, None, :]


@torch.no_grad()
def _scores(model, x, mean, scale, device, batch_nodes=256):
    model.eval()
    output = []
    for start in range(0, len(x), batch_nodes):
        local = _normalise(x[start:start + batch_nodes], mean, scale)
        tensor = torch.from_numpy(local).to(device)
        output.append(model(tensor).cpu().numpy())
    return np.concatenate(output)


def _rank_metrics(scores, target, covered):
    indices = np.arange(len(scores))
    truth_score = scores[indices, target]
    rank = 1 + np.sum(scores > truth_score[:, None], axis=1)
    order = np.argsort(-scores, axis=1)
    top1 = OFFSETS[order[:, 0]]
    top8 = OFFSETS[order[:, :8]]
    truth = OFFSETS[target]
    valid = covered.astype(bool)
    return {
        "groups": int(valid.sum()),
        "top1_within5": float(np.mean(np.abs(top1[valid] - truth[valid]) <= 5.0)),
        "top8_within5": float(np.mean(np.min(np.abs(top8[valid] - truth[valid, None]), axis=1) <= 5.0)),
        "median_truth_rank": float(np.median(rank[valid])),
        "p90_truth_rank": float(np.quantile(rank[valid], 0.90)),
    }


def _soft_targets(target, device):
    target_offset = torch.as_tensor(OFFSETS[target], dtype=torch.float32, device=device)
    candidates = torch.as_tensor(OFFSETS, dtype=torch.float32, device=device)
    probability = torch.exp(-0.5 * ((candidates[None, :] - target_offset[:, None]) / SOFT_TARGET_SIGMA_FT) ** 2)
    return probability / probability.sum(dim=1, keepdim=True)


def _train_epoch(model, optimizer, scaler, x, target, covered, mean, scale, device, rng):
    model.train()
    indices = np.flatnonzero(covered)
    rng.shuffle(indices)
    losses = []
    for start in range(0, len(indices), BATCH_NODES):
        batch = indices[start:start + BATCH_NODES]
        local = _normalise(x[batch], mean, scale)
        tensor = torch.from_numpy(local).to(device)
        probability = _soft_targets(target[batch], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(tensor)
            loss = -(probability * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _cost_payload(model, payload, mean, scale, device, donor_records=None):
    output = {}
    for number, (well, local) in enumerate(payload.items(), 1):
        x = local["features"]
        if donor_records is not None:
            record = RECORD_BY_WELL[well]
            donor = donor_records[well]
            rows, distance, state_surface, x, _, _, _ = _pair_features(record, donor=donor)
        else:
            rows = local["rows"]
            distance = local["distance"]
            state_surface = local["state_surface"]
        score = _scores(model, x, mean, scale, device)
        cost = -score.astype(np.float64)
        cost -= np.min(cost, axis=1, keepdims=True)
        cost[0] = 0.0
        output[well] = (rows, distance, state_surface, cost)
        if number % 25 == 0 or number == len(payload):
            print(f"V33 score maps {number}/{len(payload)}", flush=True)
    return output


def _decode_costs(records, costs, prior, config, label):
    original = V._emission

    def learned(record, _prior, _config, donor=None, no_gr=False):
        rows, distance, state_surface, cost = costs[record["well"]]
        if no_gr:
            cost = np.zeros_like(cost)
        return rows, distance, state_surface, cost

    V._emission = learned
    decoded = {}
    try:
        for number, record in enumerate(records, 1):
            decoded[record["well"]] = V._decode(
                record, prior, config, no_gr=False, sample_paths=False
            )
            if number % 25 == 0 or number == len(records):
                print(f"V33 decode {label} {number}/{len(records)}", flush=True)
    finally:
        V._emission = original
    rows = V._metric_rows(records, decoded, label)
    return rows, V._summary(rows)


started = time.time()
if not V30_VERDICT.exists():
    raise FileNotFoundError(V30_VERDICT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v30 = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
PRIOR = v30["outer_train_prior"]
DECODER_CONFIG = dict(v30["selected_config"])
# Learned logits are trained as calibrated log evidence; do not inherit the
# hand-crafted Cauchy temperature.
DECODER_CONFIG["temperature"] = 1.0
folds = pd.read_csv(V30_DIR / "folds.csv", dtype={"well": str})
train_wells = sorted(folds.loc[folds["fold"].isin((4, 5)), "well"].astype(str))
valid_wells = sorted(folds.loc[folds["fold"].eq(3), "well"].astype(str))
application_wells = sorted(folds.loc[folds["fold"].eq(2), "well"].astype(str))
train_records = _load_records(train_wells, "train45")
valid_records = _load_records(valid_wells, "fold3")
application_records = _load_records(application_wells, "fold2")
RECORD_BY_WELL = {record["well"]: record for record in train_records + valid_records + application_records}

x_train, y_train, keep_train, owner_train, train_payload = _build_arrays(train_records, "train45")
x_valid, y_valid, keep_valid, owner_valid, valid_payload = _build_arrays(valid_records, "fold3")
x_application, y_application, keep_application, owner_application, application_payload = _build_arrays(application_records, "fold2")

flat_train = x_train[keep_train].astype(np.float32).reshape(-1, x_train.shape[-1])
feature_mean = np.mean(flat_train, axis=0).astype(np.float32)
feature_scale = np.std(flat_train, axis=0).astype(np.float32)
feature_scale = np.maximum(feature_scale, 1e-3)
del flat_train
np.savez_compressed(OUT_DIR / "feature_normalisation.npz", mean=feature_mean, scale=feature_scale)

model = PairEmission(x_train.shape[-1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
rng = np.random.default_rng(SEED)
best_key = None
best_epoch = 0
bad_epochs = 0
history = []
checkpoint = MODEL_DIR / "best_pair_emission.pt"
for epoch in range(1, MAX_EPOCHS + 1):
    loss = _train_epoch(
        model, optimizer, scaler, x_train, y_train, keep_train,
        feature_mean, feature_scale, device, rng,
    )
    valid_scores = _scores(model, x_valid, feature_mean, feature_scale, device)
    metrics = _rank_metrics(valid_scores, y_valid, keep_valid)
    report = {"epoch": epoch, "train_loss": loss, **metrics}
    history.append(report)
    print("V33 EPOCH", report, flush=True)
    key = (metrics["top8_within5"], metrics["top1_within5"], -metrics["median_truth_rank"])
    if best_key is None or key > best_key:
        best_key = key
        best_epoch = epoch
        bad_epochs = 0
        torch.save({
            "model": model.state_dict(),
            "features": int(x_train.shape[-1]),
            "mean": feature_mean,
            "scale": feature_scale,
            "epoch": epoch,
            "metrics": metrics,
        }, checkpoint)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
state = torch.load(checkpoint, map_location=device, weights_only=False)
model.load_state_dict(state["model"])

valid_scores = _scores(model, x_valid, feature_mean, feature_scale, device)
application_scores = _scores(model, x_application, feature_mean, feature_scale, device)
local_metrics = {
    "fold3": _rank_metrics(valid_scores, y_valid, keep_valid),
    "fold2": _rank_metrics(application_scores, y_application, keep_application),
}

valid_costs = _cost_payload(model, valid_payload, feature_mean, feature_scale, device)
application_costs = _cost_payload(model, application_payload, feature_mean, feature_scale, device)
metric_parts = []
summary_parts = []
for fold, records, costs, label in (
    (3, valid_records, valid_costs, "fold3_learned"),
    (2, application_records, application_costs, "fold2_learned"),
):
    bywell, summary = _decode_costs(records, costs, PRIOR, DECODER_CONFIG, label)
    bywell["fold"] = fold
    summary["fold"] = fold
    metric_parts.append(bywell)
    summary_parts.append(summary)

# Matched causal controls on Fold 2.
zero_costs = {
    well: (rows, distance, surface, np.zeros_like(cost))
    for well, (rows, distance, surface, cost) in application_costs.items()
}
no_gr_rows, no_gr_summary = _decode_costs(
    application_records, zero_costs, PRIOR, DECODER_CONFIG, "fold2_no_gr"
)
no_gr_rows["fold"] = 2
no_gr_summary["fold"] = 2
metric_parts.append(no_gr_rows)
summary_parts.append(no_gr_summary)

donor_order = application_records[1:] + application_records[:1]
donors = {record["well"]: donor for record, donor in zip(application_records, donor_order)}
shuffled_costs = _cost_payload(
    model, application_payload, feature_mean, feature_scale, device, donor_records=donors
)
shuffle_rows, shuffle_summary = _decode_costs(
    application_records, shuffled_costs, PRIOR, DECODER_CONFIG, "fold2_shuffled_typewell"
)
shuffle_rows["fold"] = 2
shuffle_summary["fold"] = 2
metric_parts.append(shuffle_rows)
summary_parts.append(shuffle_summary)

bywell_frame = pd.concat(metric_parts, ignore_index=True)
summary_frame = pd.concat(summary_parts, ignore_index=True)
bywell_frame.to_csv(OUT_DIR / "decoder_bywell.csv", index=False)
summary_frame.to_csv(OUT_DIR / "decoder_summary.csv", index=False)
print("\nV33 DECODER SUMMARY")
print(summary_frame.to_string(index=False), flush=True)

def _row(variant):
    return summary_frame[
        summary_frame["variant"].eq(variant)
        & summary_frame["estimator"].eq("posterior_mean")
    ].iloc[0]

fold3 = _row("fold3_learned")
fold2 = _row("fold2_learned")
no_gr = _row("fold2_no_gr")
shuffled = _row("fold2_shuffled_typewell")
gates = {
    "fold3_local_top8_ge_0p75": bool(local_metrics["fold3"]["top8_within5"] >= 0.75),
    "fold2_local_top8_ge_0p70": bool(local_metrics["fold2"]["top8_within5"] >= 0.70),
    "fold3_path_rmse_le_15": bool(fold3["rmse"] <= 15.0),
    "fold2_path_rmse_le_18": bool(fold2["rmse"] <= 18.0),
    "no_gr_degradation_ge_5": bool(no_gr["rmse"] - fold2["rmse"] >= 5.0),
    "shuffled_typewell_degradation_ge_5": bool(shuffled["rmse"] - fold2["rmse"] >= 5.0),
}
if gates["fold2_path_rmse_le_18"]:
    next_action = "PASS -> train all five fold-clean pair emissions, then test heteroscedastic segment aggregation"
elif gates["fold2_local_top8_ge_0p70"]:
    next_action = "Local pair ranking works but path calibration fails; calibrate emission temperature on Fold3 only"
else:
    next_action = "Pair statistics remain alias-limited; move to a trainable sequence encoder, not another decoder scan"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_pair_training_Fold3_earlystop_Fold2_confirmation",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "state_half_width_ft": HALF_WIDTH_FT,
    "state_step_ft": STATE_STEP_FT,
    "states": int(len(OFFSETS)),
    "radii_ft": list(RADII_FT),
    "pair_channels": list(PAIR_CHANNELS),
    "features": int(x_train.shape[-1]),
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "selected_epoch": int(best_epoch),
    "local_metrics": local_metrics,
    "fold3_path": fold3.to_dict(),
    "fold2_path": fold2.to_dict(),
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
    "next": next_action,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
