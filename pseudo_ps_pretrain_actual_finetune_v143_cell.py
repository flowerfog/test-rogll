"""V143: pseudo-PS pretraining followed by actual-PS fine-tuning.

The model does not consume V127, candidate-bank trajectories, CP, Neighbor, or
any residual endpoint as an input. It starts from the visible anchor and a
constant-formation-surface reference and directly predicts the complete TVT
trajectory at multiple resolutions. This removes the ill-conditioned cumulative
sum used by V137/V138 while retaining an exactly fixed visible anchor.

The V139 architecture and V142 samples are unchanged. V142 mixed five pseudo
starts with every actual start throughout optimization. V143 first learns from
the augmented set, then fine-tunes only on the 310 competition-shaped actual
starts at a lower learning rate. Fold3 remains actual-PS only.
"""

from __future__ import annotations

import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "pseudo_ps_pretrain_actual_finetune_v143"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V126_SOURCE = ROOT / "final_consensus_sequence_coefficient_v126_cell.py"
V127_PATH = OOF_ROOT / "final_consensus_sequence_coefficient_crossfit_v127" / "crossfit_oof.parquet"
BASE = "nested_sequence_coefficient"
TRUTH = "tvt_true"
TRAIN_FOLDS = (1, 2)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
NODES = 256
PREFIX_NODES = 96
PREFIX_FT = 1500.0
PSEUDO_PS_FRACTIONS = (0.12, 0.24, 0.36, 0.48, 0.60)
MIN_PSEUDO_FUTURE_ROWS = 256
TYPE_OFFSETS = np.asarray([-96, -64, -32, -16, 0, 16, 32, 64, 96], np.float32)
SEED = 20260802
CAPACITY_WELLS = 24
CAPACITY_EPOCHS = 400
CAPACITY_GATE_FT = 3.0
MAIN_EPOCHS = 120
FINETUNE_EPOCHS = 100
FINETUNE_LEARNING_RATE = 1e-4
PATIENCE = 16
BATCH_SIZE = 16
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 2e-3
TARGET_SCALE_FT = 16.0


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v126():
    if not V126_SOURCE.is_file():
        raise FileNotFoundError(V126_SOURCE)
    source = V126_SOURCE.read_text(encoding="utf-8")
    marker = "for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):"
    if marker not in source:
        raise RuntimeError(f"V126 marker missing from {V126_SOURCE}")
    module = types.ModuleType("absolute_v143_v126_definitions")
    module.__file__ = str(V126_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V126_SOURCE), "exec"), module.__dict__)
    return module


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _robust(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(np.std(values))
    return np.clip((values - centre) / max(scale, 1e-3), -8.0, 8.0)


Q = _bootstrap_v126()
_seed_everything(SEED)
R = Q._bootstrap_v110()
N = R._bootstrap_v81()
V = N.V

if not V127_PATH.is_file():
    raise FileNotFoundError(V127_PATH)
frame = pd.read_parquet(V127_PATH)
frame["well"] = frame["well"].astype(str)
required = {"id", "well", "row_idx", "fold", TRUTH, "md_since", BASE}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V127 OOF missing columns: {missing}")
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate V127 ids")
frame = frame[frame["fold"].isin((1, 2, 3))].copy()
frame = frame.sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
well_positions = {
    str(well): np.asarray(position, np.int64)
    for well, position in frame.groupby("well", sort=False).indices.items()
}


def _well_frame(well):
    return frame.iloc[well_positions[str(well)]]


grid = np.linspace(0.0, 1.0, NODES).astype(np.float64)
well_order = sorted(well_positions)
feature_rows = []
prefix_rows = []
target_rows = []
reference_rows = []
distance_rows = []
fold_rows = []
sample_wells = []
sample_kinds = []
actual_sample_index_by_well = {}


def _append_sample(well, fold, record, future_rows, anchor_index, anchor_surface,
                   visible_tvt, kind):
    full_md = record["md"].astype(np.float64)
    full_gr = Q._fill(record["gr"], full_md)
    full_z = Q._fill(record["z"], full_md)
    future_rows = np.asarray(future_rows, np.int64)
    truth = record["truth"][future_rows].astype(np.float64)
    md = full_md[future_rows]
    anchor_md = float(full_md[anchor_index])
    distance = md - anchor_md
    invalid_start = distance[0] < -1e-6 or (kind == "pseudo" and distance[0] <= 0.0)
    if len(distance) < 32 or invalid_start or np.any(np.diff(distance) < 0.0):
        raise RuntimeError(f"{well}/{kind}: invalid future trajectory")
    maximum_distance = max(float(distance[-1]), 1.0)
    progress = distance / maximum_distance
    z = full_z[future_rows]
    gr = full_gr[future_rows]
    reference = float(anchor_surface) - z
    target_delta = truth - reference

    anchor_gr = float(full_gr[anchor_index])
    anchor_z = float(full_z[anchor_index])
    anchor_tvt = float(anchor_surface) - anchor_z
    progress_extended = np.concatenate(([0.0], progress))
    gr_extended = np.concatenate(([anchor_gr], gr))
    z_extended = np.concatenate(([anchor_z], z))
    reference_extended = np.concatenate(([anchor_tvt], reference))
    target_extended = np.concatenate(([0.0], target_delta))
    md_extended = np.concatenate(([anchor_md], md))

    visible_tvt = np.asarray(visible_tvt, np.float64)
    known = np.flatnonzero(
        np.isfinite(visible_tvt) & np.isfinite(full_gr) & (full_md <= anchor_md)
        & (full_md >= anchor_md - PREFIX_FT)
    )
    if len(known) < 32:
        known = np.flatnonzero(
            np.isfinite(visible_tvt) & np.isfinite(full_gr) & (full_md <= anchor_md)
        )[-max(32, min(256, int(np.isfinite(visible_tvt).sum()))):]
    if len(known) < 16:
        raise RuntimeError(f"{well}/{kind}: only {len(known)} visible-prefix rows")
    prefix_md = full_md[known]
    prefix_coordinate = (prefix_md - anchor_md) / max(anchor_md - float(prefix_md[0]), 1.0)
    prefix_grid = np.linspace(-1.0, 0.0, PREFIX_NODES)
    prefix_gr_raw = np.interp(prefix_grid, prefix_coordinate, full_gr[known])
    prefix_tvt = np.interp(prefix_grid, prefix_coordinate, visible_tvt[known])
    prefix_z = np.interp(prefix_grid, prefix_coordinate, full_z[known])
    prefix_surface = prefix_tvt + prefix_z
    prefix_md_grid = np.interp(prefix_grid, prefix_coordinate, prefix_md)
    prefix_features = np.asarray([
        prefix_grid,
        _robust(prefix_gr_raw),
        np.gradient(_robust(prefix_gr_raw)),
        (prefix_tvt - anchor_tvt) / 50.0,
        (prefix_z - anchor_z) / 50.0,
        (prefix_surface - float(anchor_surface)) / 50.0,
        np.gradient(prefix_tvt, prefix_md_grid),
        np.gradient(prefix_surface, prefix_md_grid),
    ], np.float64)

    gr_grid_raw = np.interp(grid, progress_extended, gr_extended)
    gr_grid = _robust(gr_grid_raw)
    z_grid = np.interp(grid, progress_extended, z_extended)
    reference_grid = np.interp(grid, progress_extended, reference_extended)
    target_grid = np.interp(grid, progress_extended, target_extended)
    md_grid = np.interp(grid, progress_extended, md_extended)
    spacing = np.maximum(np.gradient(md_grid), 1e-3)

    type_tvt = np.asarray(record["type_tvt"], np.float64)
    type_gr = np.asarray(record["type_gr"], np.float64)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_frame = pd.DataFrame({"tvt": type_tvt[finite], "gr": type_gr[finite]})
    type_frame = type_frame.groupby("tvt", as_index=False, sort=True)["gr"].mean()
    if len(type_frame) < 32:
        raise RuntimeError(f"{well}: only {len(type_frame)} typewell rows")
    type_tvt = type_frame["tvt"].to_numpy(np.float64)
    type_gr = type_frame["gr"].to_numpy(np.float64)
    candidates = np.stack([
        np.interp(reference_grid + float(offset), type_tvt, type_gr)
        for offset in TYPE_OFFSETS
    ])
    mismatch = (gr_grid_raw[None, :] - candidates) / max(float(record["noise_scale"]), 5.0)

    channels = [
        grid,
        np.log1p(grid * maximum_distance) / np.log1p(maximum_distance),
        gr_grid,
        np.gradient(gr_grid) / np.maximum(spacing / np.median(spacing), 0.25),
        (z_grid - z_grid[0]) / 100.0,
        np.gradient(z_grid, md_grid),
        (reference_grid - reference_grid[0]) / 100.0,
        np.gradient(reference_grid, md_grid),
    ]
    channels.extend(mismatch)
    channels.extend(np.abs(mismatch))
    features = np.asarray(channels, np.float64)
    if not np.isfinite(features).all() or not np.isfinite(target_grid).all():
        raise RuntimeError(f"{well}/{kind}: non-finite V143 tensors")
    feature_rows.append(features.astype(np.float32))
    prefix_rows.append(prefix_features.astype(np.float32))
    target_rows.append(target_grid.astype(np.float32))
    reference_rows.append(reference_grid.astype(np.float32))
    distance_rows.append(maximum_distance)
    fold_rows.append(int(fold))
    sample_wells.append(str(well))
    sample_kinds.append(str(kind))

for number, well in enumerate(well_order, 1):
    local = _well_frame(well)
    record = Q._load_record_relaxed(V, well)
    eval_rows = record["eval_rows"]
    if len(local) != len(eval_rows):
        raise RuntimeError(f"{well}: OOF/raw row mismatch {len(local)}/{len(eval_rows)}")
    truth = local[TRUTH].to_numpy(np.float64)
    truth_gap = float(np.max(np.abs(truth - record["truth"][eval_rows].astype(np.float64))))
    if truth_gap > 0.01:
        raise RuntimeError(f"{well}: truth mismatch {truth_gap}")

    fold = int(local["fold"].iloc[0])
    tvt_input = np.asarray(record["tvt_input"], np.float64)
    actual_sample_index_by_well[well] = len(feature_rows)
    _append_sample(
        well, fold, record, eval_rows, int(record["anchor_index"]),
        float(record["anchor_surface"]), tvt_input, "actual",
    )

    if fold in TRAIN_FOLDS:
        maximum_anchor = len(eval_rows) - MIN_PSEUDO_FUTURE_ROWS - 1
        pseudo_positions = sorted({
            int(round(fraction * maximum_anchor))
            for fraction in PSEUDO_PS_FRACTIONS
            if maximum_anchor >= 32
        })
        for position in pseudo_positions:
            if position < 16 or position >= len(eval_rows) - MIN_PSEUDO_FUTURE_ROWS:
                continue
            anchor_index = int(eval_rows[position])
            visible_pseudo = tvt_input.copy()
            visible_rows = eval_rows[:position + 1]
            visible_pseudo[visible_rows] = record["truth"][visible_rows]
            full_z = Q._fill(record["z"], record["md"].astype(np.float64))
            anchor_surface = float(record["truth"][anchor_index] + full_z[anchor_index])
            _append_sample(
                well, fold, record, eval_rows[position + 1:], anchor_index,
                anchor_surface, visible_pseudo, "pseudo",
            )
    if number % 50 == 0 or number == len(well_order):
        print(
            f"V143 sequences {number}/{len(well_order)} samples={len(feature_rows)}",
            flush=True,
        )

features_raw = np.stack(feature_rows)
prefix_raw = np.stack(prefix_rows)
targets = np.stack(target_rows)
references = np.stack(reference_rows)
maximum_distance = np.asarray(distance_rows, np.float32)
fold_by_index = np.asarray(fold_rows, np.int8)
sample_kinds_array = np.asarray(sample_kinds, object)
train_index = np.flatnonzero(np.isin(fold_by_index, TRAIN_FOLDS))
fold3_index = np.flatnonzero((fold_by_index == EVAL_FOLD) & (sample_kinds_array == "actual"))
discovery_index = fold3_index[:DISCOVERY_WELLS]
confirmation_index = fold3_index[DISCOVERY_WELLS:]
training_actual = int(np.sum(sample_kinds_array[train_index] == "actual"))
training_pseudo = int(np.sum(sample_kinds_array[train_index] == "pseudo"))
if training_actual != 310 or training_pseudo < 1200 or len(discovery_index) != 24 or len(confirmation_index) < 129:
    raise RuntimeError(
        "Coverage actual/pseudo/discovery/confirmation="
        f"{training_actual}/{training_pseudo}/{len(discovery_index)}/{len(confirmation_index)}"
    )

feature_mean = np.mean(features_raw[train_index], axis=(0, 2), keepdims=True)
feature_scale = np.maximum(np.std(features_raw[train_index], axis=(0, 2), keepdims=True), 1e-3)
features = ((features_raw - feature_mean) / feature_scale).astype(np.float32)
prefix_mean = np.mean(prefix_raw[train_index], axis=(0, 2), keepdims=True)
prefix_scale = np.maximum(np.std(prefix_raw[train_index], axis=(0, 2), keepdims=True), 1e-3)
prefixes = ((prefix_raw - prefix_mean) / prefix_scale).astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class DirectAbsoluteTrajectory(nn.Module):
    def __init__(self, channels, prefix_channels, width=96, dropout=0.08):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels, width, 7, padding=3),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.position = nn.Parameter(torch.zeros(1, width, NODES))
        prefix_width = width // 2
        self.prefix_encoder = nn.Sequential(
            nn.Conv1d(prefix_channels, prefix_width, 7, padding=3),
            nn.GroupNorm(8, prefix_width),
            nn.SiLU(),
            nn.Conv1d(prefix_width, width, 5, padding=2),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=4, dilation=2),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.context = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.SiLU(),
            nn.Linear(width, 2 * width),
        )
        self.blocks = nn.Sequential(*(
            ResidualBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16, 32, 1, 4)
        ))
        self.coarse_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(32),
            nn.Conv1d(width, width // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width // 2, 1, 1),
        )
        self.medium_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),
            nn.Conv1d(width, width // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width // 2, 1, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Conv1d(width, width // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width // 2, 1, 1),
        )
        for head in (self.coarse_head[-1], self.medium_head[-1], self.residual_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, value, prefix_value):
        encoded = self.stem(value) + self.position
        prefix_context = self.prefix_encoder(prefix_value)
        pooled = torch.cat([
            encoded.mean(dim=2), encoded.std(dim=2), prefix_context
        ], dim=1)
        scale, bias = self.context(pooled).chunk(2, dim=1)
        encoded = encoded * (1.0 + 0.20 * torch.tanh(scale[:, :, None])) + 0.20 * bias[:, :, None]
        encoded = self.blocks(encoded)
        coarse32 = 128.0 * torch.tanh(self.coarse_head(encoded).squeeze(1) / 8.0)
        medium64 = 32.0 * torch.tanh(self.medium_head(encoded).squeeze(1) / 8.0)
        residual256 = 12.0 * torch.tanh(self.residual_head(encoded).squeeze(1) / 4.0)
        coarse256 = F.interpolate(
            coarse32[:, None, :], size=NODES, mode="linear", align_corners=True
        ).squeeze(1)
        medium256 = F.interpolate(
            medium64[:, None, :], size=NODES, mode="linear", align_corners=True
        ).squeeze(1)
        path = coarse256 + medium256 + residual256
        path = path - path[:, :1]
        return path, coarse32, medium64, residual256


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loss(output, target, distance_max):
    path, coarse32, medium64, residual256 = output
    row_weight = distance_max[:, None] / torch.clamp(distance_max.sum(), min=1.0)
    row_weight = row_weight.expand_as(path)
    error = (path - target) / TARGET_SCALE_FT
    point = torch.sum(row_weight * F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.5, reduction="none"))
    point = point / path.shape[1]
    mse = torch.sum(row_weight * error.square()) / path.shape[1]
    target32 = F.interpolate(target[:, None, :], size=32, mode="linear", align_corners=True).squeeze(1)
    target64 = F.interpolate(target[:, None, :], size=64, mode="linear", align_corners=True).squeeze(1)
    coarse_loss = F.smooth_l1_loss(coarse32 / TARGET_SCALE_FT, target32 / TARGET_SCALE_FT, beta=0.5)
    medium_loss = F.smooth_l1_loss(
        (F.interpolate(coarse32[:, None, :], size=64, mode="linear", align_corners=True).squeeze(1) + medium64)
        / TARGET_SCALE_FT,
        target64 / TARGET_SCALE_FT,
        beta=0.5,
    )
    change_error = torch.diff(path, dim=1) - torch.diff(target, dim=1)
    derivative = F.smooth_l1_loss(change_error / 4.0, torch.zeros_like(change_error), beta=0.5)
    endpoint = F.smooth_l1_loss((path[:, -1] - target[:, -1]) / 16.0, torch.zeros_like(path[:, -1]))
    residual_penalty = torch.mean(torch.abs(residual256) / 12.0)
    total = (
        point + 0.50 * mse + 0.35 * coarse_loss + 0.25 * medium_loss
        + 0.08 * derivative + 0.08 * endpoint + 0.001 * residual_penalty
    )
    return total


def _fit(indices, validation_indices, width, dropout, epochs, patience, label,
         initial_state=None, learning_rate=LEARNING_RATE):
    _seed_everything(SEED + len(indices))
    model = DirectAbsoluteTrajectory(
        features.shape[1], prefixes.shape[1], width=width, dropout=dropout
    ).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=WEIGHT_DECAY
    )
    dataset = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(prefixes[indices]),
        torch.from_numpy(targets[indices]),
        torch.from_numpy(maximum_distance[indices]),
    )
    loader = DataLoader(
        dataset, batch_size=min(BATCH_SIZE, len(indices)), shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_features, batch_prefix, batch_target, batch_distance in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_features.to(device), batch_prefix.to(device))
            loss = _loss(output, batch_target.to(device), batch_distance.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_output = model(
                torch.from_numpy(features[validation_indices]).to(device),
                torch.from_numpy(prefixes[validation_indices]).to(device),
            )
            valid_loss = float(_loss(
                valid_output,
                torch.from_numpy(targets[validation_indices]).to(device),
                torch.from_numpy(maximum_distance[validation_indices]).to(device),
            ).cpu())
        row = {"phase": label, "epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": valid_loss}
        history.append(row)
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print("V143 EPOCH", row, flush=True)
        if patience is not None and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"V143 {label} selected no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


@torch.no_grad()
def _predict_grid(model, indices, input_override=None, prefix_override=None):
    model.eval()
    source = features if input_override is None else input_override
    prefix_source = prefixes if prefix_override is None else prefix_override
    output = []
    residual_energy = []
    for start in range(0, len(indices), BATCH_SIZE):
        index = indices[start:start + BATCH_SIZE]
        path, _, _, residual = model(
            torch.from_numpy(source[index]).to(device),
            torch.from_numpy(prefix_source[index]).to(device),
        )
        output.append(path.cpu().numpy())
        residual_energy.extend(torch.mean(torch.abs(residual), dim=1).cpu().numpy().tolist())
    return np.concatenate(output), float(np.mean(residual_energy))


def _evaluate(indices, grid_prediction, endpoint):
    parts = []
    predictions = []
    for index, delta_grid in zip(indices, grid_prediction):
        if sample_kinds[int(index)] != "actual":
            raise RuntimeError("Metrics may only be computed on actual prediction-start samples")
        well = sample_wells[int(index)]
        local = _well_frame(well)
        distance = local["md_since"].to_numpy(np.float64)
        progress = distance / max(float(distance[-1]), 1.0)
        record = Q._load_record_relaxed(V, well)
        z = Q._fill(record["z"], record["md"])[record["eval_rows"]]
        reference = float(record["anchor_surface"]) - z
        prediction = reference + np.interp(progress, grid, np.asarray(delta_grid, np.float64))
        parts.append(local)
        predictions.append(prediction)
    matched = pd.concat(parts, ignore_index=True)
    prediction = np.concatenate(predictions)
    return R._metric(matched, prediction, endpoint), matched, prediction


# Capacity gate: train and evaluate on the same 24 physical wells.
training_wells = [
    well for well in well_order
    if int(_well_frame(well)["fold"].iloc[0]) in TRAIN_FOLDS
]
capacity_index = np.asarray([
    actual_sample_index_by_well[well] for well in training_wells[:CAPACITY_WELLS]
], np.int64)
capacity_model, capacity_history, capacity_epoch, _ = _fit(
    capacity_index, capacity_index, width=160, dropout=0.0,
    epochs=CAPACITY_EPOCHS, patience=50, label="capacity24",
)
capacity_grid, capacity_residual = _predict_grid(capacity_model, capacity_index)
capacity_metric, _, _ = _evaluate(capacity_index, capacity_grid, "capacity24")
print("V143 CAPACITY", capacity_metric, flush=True)
pd.DataFrame(capacity_history).to_csv(OUT_DIR / "capacity_history.csv", index=False)
capacity_pass = bool(capacity_metric["rmse"] <= CAPACITY_GATE_FT)
if not capacity_pass:
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "24well_absolute_trajectory_capacity_gate",
        "architecture": "V139 direct trajectory TCN with pseudo-PS training augmentation",
        "capacity": capacity_metric,
        "capacity_best_epoch": capacity_epoch,
        "capacity_mean_abs_fine_residual": capacity_residual,
        "capacity_gate_ft": CAPACITY_GATE_FT,
        "strict_pass": False,
        "elapsed_minutes": (time.time() - STARTED) / 60.0,
        "next": "FAIL CAPACITY -> even direct absolute supervision is not optimizable enough; stop this architecture family",
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    print("V143 FINAL VERDICT")
    print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
    raise RuntimeError("V143 capacity gate failed; Fold3 was not trained or evaluated")

del capacity_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

pretrain_model, pretrain_history, pretrain_epoch, _ = _fit(
    train_index, discovery_index, width=128, dropout=0.08,
    epochs=MAIN_EPOCHS, patience=PATIENCE, label="augmented_pretrain",
)
pretrain_state = {
    key: value.detach().cpu().clone()
    for key, value in pretrain_model.state_dict().items()
}
actual_train_index = train_index[sample_kinds_array[train_index] == "actual"]
model, finetune_history, finetune_epoch, best_loss = _fit(
    actual_train_index, discovery_index, width=128, dropout=0.04,
    epochs=FINETUNE_EPOCHS, patience=PATIENCE, label="actual_only_finetune",
    initial_state=pretrain_state, learning_rate=FINETUNE_LEARNING_RATE,
)
history = pretrain_history + finetune_history
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
del pretrain_model, pretrain_state
if torch.cuda.is_available():
    torch.cuda.empty_cache()

discovery_grid, discovery_residual = _predict_grid(model, discovery_index)
confirmation_grid, confirmation_residual = _predict_grid(model, confirmation_index)
discovery_metric, _, _ = _evaluate(discovery_index, discovery_grid, "direct_absolute_trajectory")
confirmation_metric, matched, confirmation_prediction = _evaluate(
    confirmation_index, confirmation_grid, "direct_absolute_trajectory"
)
discovery_frame = frame[frame["well"].isin([sample_wells[index] for index in discovery_index])]
confirmation_frame = frame[frame["well"].isin([sample_wells[index] for index in confirmation_index])]
discovery_base = R._metric(discovery_frame, discovery_frame[BASE].to_numpy(np.float64), BASE)
confirmation_base = R._metric(confirmation_frame, confirmation_frame[BASE].to_numpy(np.float64), BASE)

# Matched causal controls on the fixed confirmation model.
control_features = features.copy()
control_features[:, 2:4, :] = 0.0
control_features[:, 8:, :] = 0.0
control_prefix = prefixes.copy()
control_prefix[:, 1:3, :] = 0.0
no_gr_grid, _ = _predict_grid(
    model, confirmation_index, control_features, control_prefix
)
no_gr_metric, _, _ = _evaluate(confirmation_index, no_gr_grid, "no_gr_control")
shuffled_features = features.copy()
rolled = np.roll(confirmation_index, max(1, len(confirmation_index) // 7))
shuffled_features[confirmation_index, 8:, :] = features[rolled, 8:, :]
shuffled_grid, _ = _predict_grid(model, confirmation_index, shuffled_features)
shuffled_metric, _, _ = _evaluate(confirmation_index, shuffled_grid, "shuffled_typewell_control")

summary = pd.DataFrame([
    dict(discovery_base, stage="fold3_discovery"),
    dict(discovery_metric, stage="fold3_discovery"),
    dict(confirmation_base, stage="fold3_confirmation"),
    dict(confirmation_metric, stage="fold3_confirmation"),
    dict(no_gr_metric, stage="fold3_confirmation"),
    dict(shuffled_metric, stage="fold3_confirmation"),
])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
output = matched[["id", "well", "row_idx", "fold", TRUTH, "md_since", BASE]].copy()
output["direct_absolute_trajectory"] = confirmation_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_confirmation_oof.parquet", index=False, compression="zstd")

values = {
    "discovery_gain_vs_v127": discovery_base["rmse"] - discovery_metric["rmse"],
    "confirmation_gain_vs_v127": confirmation_base["rmse"] - confirmation_metric["rmse"],
    "confirmation_far_gain": confirmation_base["far_gt2000_rmse"] - confirmation_metric["far_gt2000_rmse"],
    "no_gr_degradation": no_gr_metric["rmse"] - confirmation_metric["rmse"],
    "shuffled_typewell_degradation": shuffled_metric["rmse"] - confirmation_metric["rmse"],
}
checks = {
    "capacity_rmse_le_3": capacity_pass,
    "discovery_gain_ge_0p30": values["discovery_gain_vs_v127"] >= 0.30,
    "confirmation_gain_ge_0p30": values["confirmation_gain_vs_v127"] >= 0.30,
    "confirmation_far_gain_positive": values["confirmation_far_gain"] > 0.0,
    "no_gr_degradation_ge_0p20": values["no_gr_degradation"] >= 0.20,
    "shuffled_typewell_degradation_ge_0p20": values["shuffled_typewell_degradation"] >= 0.20,
}
strict_pass = bool(all(checks.values()))
torch.save({
    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    "feature_mean": feature_mean,
    "feature_scale": feature_scale,
    "prefix_mean": prefix_mean,
    "prefix_scale": prefix_scale,
    "nodes": NODES,
    "type_offsets": TYPE_OFFSETS,
    "pretrain_best_epoch": pretrain_epoch,
    "finetune_best_epoch": finetune_epoch,
}, MODEL_DIR / "best_direct_absolute_trajectory.pt")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "pseudoPS_pretrain_actualPS_finetune_Fold3_discovery_confirmation",
    "sample_unit": "actual or legal rolling-origin pseudo prediction start",
    "historical_prediction_inputs": [],
    "architecture": "unchanged V139 visible-prefix direct absolute trajectory TCN",
    "changed_variable": "V142 joint training replaced by pseudo-assisted pretrain then actual-only fine-tune",
    "pseudo_ps": {
        "fractions": PSEUDO_PS_FRACTIONS,
        "minimum_future_rows": MIN_PSEUDO_FUTURE_ROWS,
        "fold3_augmented": False,
    },
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "capacity": capacity_metric,
    "capacity_best_epoch": capacity_epoch,
    "pretrain_best_epoch": pretrain_epoch,
    "finetune_best_epoch": finetune_epoch,
    "finetune_learning_rate": FINETUNE_LEARNING_RATE,
    "coverage": {
        "train_wells": training_actual,
        "train_samples": len(train_index),
        "pseudo_train_samples": training_pseudo,
        "discovery_wells": len(discovery_index),
        "confirmation_wells": len(confirmation_index),
    },
    "values": values,
    "mean_abs_fine_residual": {
        "discovery": discovery_residual,
        "confirmation": confirmation_residual,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> three-fold cross-fit the absolute trajectory replacement"
        if strict_pass else
        "FAIL -> actual-only fine-tuning cannot repair absolute-trajectory transfer; close this model family"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV143 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV143 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
