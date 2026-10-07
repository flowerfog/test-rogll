"""Complete-well multiple-trajectory prediction from raw legal evidence.

The frozen raw pair/cost-map representation is reused only as input.  Unlike
the earlier state classifier and path rerankers, this model directly emits K
continuous complete-well trajectories and one probability per trajectory.
The MTP winner-takes-best loss updates only the closest trajectory head for
each well, while a probability head learns which trajectory won.

Fold 4/5 train.  A deterministic 24-well subset of Fold 3 selects the epoch;
the remaining Fold-3 wells confirm the mechanism.  Fold 2 is touched only if
the confirmation gates pass.  Fold 1 remains untouched.
"""

from __future__ import annotations

import hashlib
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


EXPERIMENT = "raw_multitrajectory_mtp_gateway_v50"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "raw_costmap_sequence_encoder_v36_cell.py",
    Path.cwd() / "raw_costmap_sequence_encoder_v36_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.is_file()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload raw_costmap_sequence_encoder_v36_cell.py")

V30_DIR = OOF_ROOT / "raw_stochastic_alignment_standalone_v30"
V30_VERDICT = V30_DIR / "verdict.json"
FOLDS_PATH = V30_DIR / "folds.csv"
SEED = 20260727
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
K_TRAJECTORIES = 8
MAX_OFFSET_FT = 160.0
MAX_EPOCHS = 36
PATIENCE = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-4
GRAD_ACCUM_WELLS = 8
MTP_SCALE_FT = 12.0
CLASSIFICATION_WEIGHT = 0.20
DIVERSITY_WEIGHT = 0.04
DIVERSITY_MARGIN_FT = 10.0
CURVATURE_WEIGHT = 0.01

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _bootstrap(path: Path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    name = "raw_costmap_v36_mtp_definitions"
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


S = _bootstrap(SOURCE_PATH)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return F.silu(x + self.network(x))


class MultiTrajectoryPredictor(nn.Module):
    def __init__(self, input_features, trajectories):
        super().__init__()
        candidate_channels = 32
        temporal_channels = 64
        self.trajectories = trajectories
        self.candidate = nn.Sequential(
            nn.Linear(input_features, candidate_channels),
            nn.LayerNorm(candidate_channels),
            nn.SiLU(),
            nn.Linear(candidate_channels, candidate_channels),
            nn.SiLU(),
        )
        self.attention = nn.Linear(candidate_channels, 1)
        self.temporal_in = nn.Linear(2 * candidate_channels, temporal_channels)
        self.temporal = nn.Sequential(
            TemporalResidualBlock(temporal_channels, 1),
            TemporalResidualBlock(temporal_channels, 2),
            TemporalResidualBlock(temporal_channels, 4),
            TemporalResidualBlock(temporal_channels, 8),
            TemporalResidualBlock(temporal_channels, 16),
        )
        self.offset_head = nn.Conv1d(
            temporal_channels, trajectories, kernel_size=1
        )
        self.probability_head = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(temporal_channels, trajectories),
        )

    def forward(self, features, valid_nodes):
        # features [B,T,S,F]. Candidate pooling deliberately preserves
        # competing state valleys rather than collapsing with an argmax.
        candidate = self.candidate(features)
        weights = F.softmax(self.attention(candidate).squeeze(-1), dim=2)
        weighted = torch.sum(candidate * weights[:, :, :, None], dim=2)
        maximum = torch.max(candidate, dim=2).values
        temporal = self.temporal_in(torch.cat((weighted, maximum), dim=2))
        temporal = self.temporal(temporal.transpose(1, 2))
        offsets = torch.tanh(self.offset_head(temporal)).transpose(1, 2)
        offsets = offsets * MAX_OFFSET_FT

        # Every hypothesis is exactly anchored at the last visible TVT.
        anchor_mask = torch.ones_like(offsets)
        anchor_mask[:, 0, :] = 0.0
        offsets = offsets * anchor_mask
        mask = valid_nodes[:, None, :].to(temporal.dtype)
        pooled = torch.sum(temporal * mask, dim=2) / mask.sum(dim=2).clamp_min(1.0)
        probability_logits = self.probability_head(pooled)
        return offsets, probability_logits


def _normalise_item(item, mean, scale, no_gr=False):
    return S._normalise_features(item["features"], mean, scale, no_gr=no_gr)


def _batch(items, mean, scale, device, no_gr=False):
    batch = len(items)
    max_nodes = max(len(item["rows"]) for item in items)
    states = len(S.OFFSETS)
    feature_count = S.EVIDENCE_FEATURES + S.COORD_FEATURES
    features = np.zeros((batch, max_nodes, states, feature_count), np.float32)
    truth = np.zeros((batch, max_nodes), np.float32)
    mask = np.zeros((batch, max_nodes), bool)
    for index, item in enumerate(items):
        nodes = len(item["rows"])
        features[index, :nodes] = _normalise_item(item, mean, scale, no_gr=no_gr)
        truth[index, :nodes] = item["true_offset"]
        mask[index, :nodes] = item["covered"]
        mask[index, 0] = True
        truth[index, 0] = 0.0
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(truth).to(device),
        torch.from_numpy(mask).to(device),
    )


def _mtp_loss(model, batch):
    features, truth, mask = batch
    hypotheses, probability_logits = model(features, mask)
    valid = mask[:, :, None].to(hypotheses.dtype)
    point = F.smooth_l1_loss(
        hypotheses / MTP_SCALE_FT,
        truth[:, :, None].expand_as(hypotheses) / MTP_SCALE_FT,
        reduction="none",
    )
    trajectory = torch.sum(point * valid, dim=1) / valid.sum(dim=1).clamp_min(1.0)
    winner = torch.argmin(trajectory.detach(), dim=1)
    regression = trajectory.gather(1, winner[:, None]).mean()
    classification = F.cross_entropy(probability_logits, winner)

    pair_distances = []
    for left in range(K_TRAJECTORIES):
        for right in range(left + 1, K_TRAJECTORIES):
            delta2 = (hypotheses[:, :, left] - hypotheses[:, :, right]) ** 2
            rms = torch.sqrt(
                torch.sum(delta2 * mask, dim=1)
                / mask.sum(dim=1).clamp_min(1)
                + 1e-6
            )
            pair_distances.append(rms)
    pair_distance = torch.stack(pair_distances, dim=1)
    diversity = F.relu(DIVERSITY_MARGIN_FT - pair_distance).mean() / DIVERSITY_MARGIN_FT

    if hypotheses.shape[1] >= 3:
        curvature = hypotheses[:, 2:, :] - 2.0 * hypotheses[:, 1:-1, :] + hypotheses[:, :-2, :]
        curvature_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        curvature_loss = torch.sum(
            F.smooth_l1_loss(curvature / 8.0, torch.zeros_like(curvature), reduction="none")
            * curvature_mask[:, :, None]
        ) / (curvature_mask.sum().clamp_min(1) * K_TRAJECTORIES)
    else:
        curvature_loss = regression.new_zeros(())

    loss = (
        regression
        + CLASSIFICATION_WEIGHT * classification
        + DIVERSITY_WEIGHT * diversity
        + CURVATURE_WEIGHT * curvature_loss
    )
    return loss, {
        "mtp": float(regression.detach().cpu()),
        "classification": float(classification.detach().cpu()),
        "diversity": float(diversity.detach().cpu()),
        "curvature": float(curvature_loss.detach().cpu()),
        "winner_entropy": float(
            (-(F.softmax(probability_logits, 1) * F.log_softmax(probability_logits, 1)).sum(1).mean())
            .detach()
            .cpu()
        ),
    }


def _train_epoch(model, optimizer, scaler, payload, mean, scale, device, rng):
    model.train()
    order = np.arange(len(payload))
    rng.shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    reports = []
    pending = 0
    for number, index in enumerate(order, 1):
        batch = _batch([payload[int(index)]], mean, scale, device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _mtp_loss(model, batch)
            scaled_loss = loss / GRAD_ACCUM_WELLS
        scaler.scale(scaled_loss).backward()
        pending += 1
        if pending == GRAD_ACCUM_WELLS or number == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        reports.append({"loss": float(loss.detach().cpu()), **parts})
    return pd.DataFrame(reports).mean(numeric_only=True).to_dict()


@torch.no_grad()
def _predict(model, item, mean, scale, device, no_gr=False):
    model.eval()
    features, _, mask = _batch([item], mean, scale, device, no_gr=no_gr)
    offsets, logits = model(features, mask)
    offsets = offsets[0, : len(item["rows"])].float().cpu().numpy()
    probabilities = F.softmax(logits[0], dim=0).float().cpu().numpy()
    base_surface = item["state_surface"][:, S.ZERO_STATE]
    surfaces = base_surface[:, None] + offsets
    return surfaces, probabilities


def _evaluate(model, records, payload, mean, scale, device, variant, no_gr=False):
    rows = []
    by_well_predictions = []
    for record, item in zip(records, payload):
        surfaces, probabilities = _predict(
            model, item, mean, scale, device, no_gr=no_gr
        )
        evaluation = record["evaluation"]
        truth = record["truth"][evaluation]
        distance = record["md"][evaluation] - record["anchor_md"]
        predictions = np.stack(
            [
                np.interp(record["md"][evaluation], record["md"][item["rows"]], surfaces[:, k])
                - record["z"][evaluation]
                for k in range(K_TRAJECTORIES)
            ],
            axis=1,
        )
        errors = predictions - truth[:, None]
        path_sse = np.sum(errors**2, axis=0)
        best = int(np.argmin(path_sse))
        top = int(np.argmax(probabilities))
        probability_mean = predictions @ probabilities
        estimators = {
            "top_probability": predictions[:, top],
            "best_of_k": predictions[:, best],
            "probability_mean": probability_mean,
            "uniform_mean": np.mean(predictions, axis=1),
        }
        node_offsets = surfaces - surfaces[:, [0]]
        pairwise = []
        for left in range(K_TRAJECTORIES):
            for right in range(left + 1, K_TRAJECTORIES):
                pairwise.append(
                    float(np.sqrt(np.mean((node_offsets[:, left] - node_offsets[:, right]) ** 2)))
                )
        far = distance > 2000.0
        for estimator, prediction in estimators.items():
            error = prediction - truth
            rows.append(
                {
                    "well": record["well"],
                    "variant": variant,
                    "estimator": estimator,
                    "rows": int(len(error)),
                    "sse": float(np.dot(error, error)),
                    "sae": float(np.sum(np.abs(error))),
                    "far_rows": int(far.sum()),
                    "far_sse": float(np.dot(error[far], error[far])) if far.any() else 0.0,
                    "top_is_best": bool(top == best),
                    "winning_probability": float(probabilities[best]),
                    "top_probability": float(probabilities[top]),
                    "mean_pairwise_rms": float(np.mean(pairwise)),
                    "minimum_pairwise_rms": float(np.min(pairwise)),
                }
            )
        by_well_predictions.append(
            {
                "well": record["well"],
                "best_head": best,
                "top_head": top,
                "top_is_best": bool(top == best),
                "winning_probability": float(probabilities[best]),
                "top_probability": float(probabilities[top]),
                "mean_pairwise_rms": float(np.mean(pairwise)),
                "minimum_pairwise_rms": float(np.min(pairwise)),
            }
        )
    frame = pd.DataFrame(rows)
    summaries = []
    for estimator, group in frame.groupby("estimator", sort=False):
        per_well = np.sqrt(group["sse"] / group["rows"])
        summaries.append(
            {
                "variant": variant,
                "estimator": estimator,
                "rows": int(group["rows"].sum()),
                "wells": int(len(group)),
                "exact_sse": float(group["sse"].sum()),
                "rmse": float(math.sqrt(group["sse"].sum() / group["rows"].sum())),
                "mae": float(group["sae"].sum() / group["rows"].sum()),
                "median_well_rmse": float(np.median(per_well)),
                "p95_well_rmse": float(np.quantile(per_well, 0.95)),
                "far_gt2000_rmse": float(
                    math.sqrt(group["far_sse"].sum() / group["far_rows"].sum())
                    if group["far_rows"].sum()
                    else np.nan
                ),
                "top_is_best_rate": float(group["top_is_best"].mean()),
                "mean_winning_probability": float(group["winning_probability"].mean()),
                "mean_pairwise_rms": float(group["mean_pairwise_rms"].mean()),
                "mean_minimum_pairwise_rms": float(group["minimum_pairwise_rms"].mean()),
            }
        )
    return frame, pd.DataFrame(summaries), pd.DataFrame(by_well_predictions)


def _summary_row(frame, estimator):
    return frame.loc[frame["estimator"].eq(estimator)].iloc[0]


started = time.time()
for required in (V30_VERDICT, FOLDS_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v30 = json.loads(V30_VERDICT.read_text(encoding="utf-8"))
prior = v30["outer_train_prior"]
hand_config = dict(v30["selected_config"])
S.PRIOR = prior
S.HAND_CONFIG = hand_config
S.M.PRIOR = prior

folds = pd.read_csv(FOLDS_PATH, dtype={"well": str})
train_wells = sorted(folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].astype(str))
fold3_wells = sorted(folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].astype(str))
fold2_wells = sorted(folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].astype(str))

train_records = S._load_records(train_wells, "train45")
fold3_records = S._load_records(fold3_wells, "fold3")
train_payload = S._build_payload(train_records, "train45")
fold3_payload = S._build_payload(fold3_records, "fold3")
feature_mean, feature_scale = S._normalisation(train_payload)

order = sorted(
    range(len(fold3_records)),
    key=lambda index: hashlib.sha256(
        f"mtp-v50:{fold3_records[index]['well']}".encode("ascii")
    ).digest(),
)
discovery_indices = order[:DISCOVERY_WELLS]
confirmation_indices = order[DISCOVERY_WELLS:]
discovery_records = [fold3_records[index] for index in discovery_indices]
discovery_payload = [fold3_payload[index] for index in discovery_indices]
confirmation_records = [fold3_records[index] for index in confirmation_indices]
confirmation_payload = [fold3_payload[index] for index in confirmation_indices]

model = MultiTrajectoryPredictor(
    S.EVIDENCE_FEATURES + S.COORD_FEATURES, K_TRAJECTORIES
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
rng = np.random.default_rng(SEED)
checkpoint = MODEL_DIR / "best_mtp.pt"
history = []
best_objective = np.inf
best_epoch = 0
bad_epochs = 0

print(
    "V50 CONFIG",
    {
        "device": str(device),
        "train_wells": len(train_records),
        "discovery_wells": len(discovery_records),
        "confirmation_wells": len(confirmation_records),
        "trajectories": K_TRAJECTORIES,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    },
    flush=True,
)

for epoch in range(1, MAX_EPOCHS + 1):
    training = _train_epoch(
        model, optimizer, scaler, train_payload, feature_mean, feature_scale, device, rng
    )
    _, discovery_summary, _ = _evaluate(
        model,
        discovery_records,
        discovery_payload,
        feature_mean,
        feature_scale,
        device,
        f"fold3_discovery_epoch{epoch}",
    )
    top = _summary_row(discovery_summary, "top_probability")
    oracle = _summary_row(discovery_summary, "best_of_k")
    objective = float(top["rmse"] + 0.15 * max(float(oracle["rmse"]) - 8.0, 0.0))
    report = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        "discovery_top_rmse": float(top["rmse"]),
        "discovery_oracle_rmse": float(oracle["rmse"]),
        "discovery_far_rmse": float(top["far_gt2000_rmse"]),
        "discovery_top_is_best_rate": float(top["top_is_best_rate"]),
        "discovery_pairwise_rms": float(top["mean_pairwise_rms"]),
        "selection_objective": objective,
    }
    history.append(report)
    print("V50 EPOCH", report, flush=True)
    if objective < best_objective - 1e-4:
        best_objective = objective
        best_epoch = epoch
        bad_epochs = 0
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "mean": feature_mean,
                "scale": feature_scale,
                "trajectories": K_TRAJECTORIES,
            },
            checkpoint,
        )
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
state = torch.load(checkpoint, map_location=device, weights_only=False)
model.load_state_dict(state["model"])

discovery_bywell, discovery_summary, discovery_modes = _evaluate(
    model,
    discovery_records,
    discovery_payload,
    feature_mean,
    feature_scale,
    device,
    "fold3_discovery",
)
confirmation_bywell, confirmation_summary, confirmation_modes = _evaluate(
    model,
    confirmation_records,
    confirmation_payload,
    feature_mean,
    feature_scale,
    device,
    "fold3_confirmation",
)

confirmation_top = _summary_row(confirmation_summary, "top_probability")
confirmation_oracle = _summary_row(confirmation_summary, "best_of_k")
confirmation_mean = _summary_row(confirmation_summary, "probability_mean")
confirmation_gates = {
    "best_of_k_rmse_le_8": bool(confirmation_oracle["rmse"] <= 8.0),
    "top_probability_rmse_le_16": bool(confirmation_top["rmse"] <= 16.0),
    "probability_mean_rmse_le_16": bool(confirmation_mean["rmse"] <= 16.0),
    "mean_pairwise_rms_ge_8": bool(confirmation_top["mean_pairwise_rms"] >= 8.0),
    "minimum_pairwise_rms_ge_2": bool(
        confirmation_top["mean_minimum_pairwise_rms"] >= 2.0
    ),
    "top_is_best_rate_ge_0p20": bool(confirmation_top["top_is_best_rate"] >= 0.20),
}
mechanism_pass = bool(all(confirmation_gates.values()))

metric_parts = [discovery_bywell, confirmation_bywell]
summary_parts = [discovery_summary, confirmation_summary]
mode_parts = [
    discovery_modes.assign(variant="fold3_discovery"),
    confirmation_modes.assign(variant="fold3_confirmation"),
]
application_payload = None
application = None

if mechanism_pass:
    fold2_records = S._load_records(fold2_wells, "fold2")
    application_payload = S._build_payload(fold2_records, "fold2")
    fold2_bywell, fold2_summary, fold2_modes = _evaluate(
        model,
        fold2_records,
        application_payload,
        feature_mean,
        feature_scale,
        device,
        "fold2_application",
    )
    metric_parts.append(fold2_bywell)
    summary_parts.append(fold2_summary)
    mode_parts.append(fold2_modes.assign(variant="fold2_application"))
    fold2_top = _summary_row(fold2_summary, "top_probability")
    fold2_oracle = _summary_row(fold2_summary, "best_of_k")
    fold2_mean = _summary_row(fold2_summary, "probability_mean")
    application = {
        "top_probability": fold2_top.to_dict(),
        "best_of_k": fold2_oracle.to_dict(),
        "probability_mean": fold2_mean.to_dict(),
        "gates": {
            "best_of_k_rmse_le_9": bool(fold2_oracle["rmse"] <= 9.0),
            "top_probability_rmse_le_17": bool(fold2_top["rmse"] <= 17.0),
            "probability_mean_rmse_le_17": bool(fold2_mean["rmse"] <= 17.0),
            "far_rmse_le_21": bool(fold2_top["far_gt2000_rmse"] <= 21.0),
        },
    }

bywell_frame = pd.concat(metric_parts, ignore_index=True)
summary_frame = pd.concat(summary_parts, ignore_index=True)
modes_frame = pd.concat(mode_parts, ignore_index=True)
bywell_frame.to_csv(OUT_DIR / "bywell.csv", index=False)
summary_frame.to_csv(OUT_DIR / "summary.csv", index=False)
modes_frame.to_csv(OUT_DIR / "mode_diagnostics.csv", index=False)
np.savez_compressed(OUT_DIR / "normalisation.npz", mean=feature_mean, scale=feature_scale)

strict_pass = bool(
    mechanism_pass
    and application is not None
    and all(application["gates"].values())
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_MTP_training_Fold3_discovery_confirmation_Fold2_conditional",
    "sample_unit": "one complete physical well",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "architecture": "candidate-attention pooling + dilated temporal encoder + K continuous trajectory heads",
    "loss": "winner-takes-best MTP + winning-head probability + diversity + curvature",
    "trajectories": K_TRAJECTORIES,
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "selected_epoch": int(best_epoch),
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "discovery_wells": len(discovery_records),
        "confirmation_wells": len(confirmation_records),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "confirmation": {
        "top_probability": confirmation_top.to_dict(),
        "best_of_k": confirmation_oracle.to_dict(),
        "probability_mean": confirmation_mean.to_dict(),
    },
    "confirmation_gates": confirmation_gates,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "checkpoint": str(checkpoint),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain MTP as a standalone multimodal endpoint and run five-fold variance"
        if strict_pass
        else "FAIL -> MTP does not jointly achieve mode coverage and legal mode selection; close raw standalone branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV50 SUMMARY")
print(summary_frame.to_string(index=False), flush=True)
print("\nV50 FINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
