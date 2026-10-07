"""State-token iTransformer over the complete typewell/heel cost map.

This replaces v0's lossy six-number emission summaries.  Each of the 257 TVT
offset states is one token whose input is its complete 512-position likelihood
trajectory.  Thirteen same-well geometry/prefix trajectories are additional
conditioning tokens.  The model outputs a full [state, position] posterior;
evaluation reports both posterior mean and globally decoded paths.

Protocol: 24-well capacity -> folds 3/4/5 train -> fold 2 validation.  Fold 1
is untouched.  No neighbour, spatial transfer, PF, CP12, Gold, or historical
prediction is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import gc
import json
import math
import random
import shutil
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bootstrap_v0():
    candidates = (
        Path("/root/autodl-tmp/inverted_transformer_fullwell_fold2_v0_cell.py"),
        Path.cwd() / "inverted_transformer_fullwell_fold2_v0_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload inverted_transformer_fullwell_fold2_v0_cell.py and "
            "native_resolution_dualref_capacity_v0_cell.py to /root/autodl-tmp."
        )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    module_name = "itransformer_v0_definitions_for_costmap_v1"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module.__dict__, source_path


V, V0_SOURCE = _bootstrap_v0()
Record = V["Record"]
DATA_ROOT = V["DATA_ROOT"]
OOF_ROOT = V["OOF_ROOT"]
RECORD_DIR = V["RECORD_DIR"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float32)
N_STATES = len(STATE_OFFSETS)
GRID_POINTS = V["GRID_POINTS"]
BASE_VARIABLE_NAMES = V["BASE_VARIABLE_NAMES"]
N_CONDITION = len(BASE_VARIABLE_NAMES)
_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_load_record = V["_load_record"]
_interp_channels = V["_interp_channels"]
_metrics = V["_metrics"]
_rmse = V["_rmse"]
_seed_everything = V["_seed_everything"]


EXPERIMENT = "inverted_costmap_transformer_fold2_v1"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
PACK_DIR = OUT_DIR / "cost_packs"
for directory in (OUT_DIR, MODEL_DIR, PACK_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260724
REQUIRE_CUDA = True
TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
UNTOUCHED_FOLD = 1

MODEL_DIM = 128
HEADS = 8
LAYERS = 4
FF_DIM = 384
DROPOUT = 0.08
EMISSION_TEMPERATURE = 0.35

CAPACITY_WELLS = 24
CAPACITY_EPOCHS = 320
CAPACITY_REPORT_EVERY = 20
CAPACITY_GATE_RMSE = 1.50

FOLD_EPOCHS = 50
FOLD_REPORT_EVERY = 2
MIN_EPOCHS = 12
EARLY_STOPPING_REPORTS = 7
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 2e-5
GRADIENT_CLIP = 3.0
GRAD_ACCUM_WELLS = 4

LOSS_CE = 1.0
LOSS_PATH = 0.30
LOSS_SLOPE = 0.06
LOSS_CURVATURE = 0.01
DP_MAX_JUMP = 10
DP_TRANSITION_LAMBDA = 0.10

ARCHITECTURE_GATE_RMSE = 8.0
TARGET_GATE_RMSE = 5.5
CONTROL_DEGRADATION_GATE = 0.50
PACK_CACHE_VERSION = 1


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


@dataclass
class CostPack:
    well: str
    emissions: np.ndarray
    condition: np.ndarray
    target: np.ndarray
    target_state: np.ndarray
    supervised: np.ndarray
    grid_md: np.ndarray
    anchor_grid: int
    anchor_md: float
    anchor_tvt: float
    truth: np.ndarray
    eval_md: np.ndarray
    eval_distance: np.ndarray
    eval_rows: int


def _pack_path(well):
    return PACK_DIR / f"{well}.npz"


def _reference_emission(query, reference):
    start = int((reference.shape[1] - N_STATES) // 2)
    reference = reference[:, start : start + N_STATES].astype(np.float32)
    ref_valid = reference[1] > 0.5
    query_valid = query[1] > 0.5
    distance = (
        (query[0, :, None] - reference[0, None, :]) ** 2
        + 0.20 * (query[2, :, None] - reference[2, None, :]) ** 2
        + 0.08 * (query[3, :, None] - reference[3, None, :]) ** 2
    )
    logits = -distance / EMISSION_TEMPERATURE
    valid = query_valid[:, None] & ref_valid[None, :]
    logits = np.where(valid, logits, -40.0)
    logits -= np.max(logits, axis=1, keepdims=True)
    # Return [state, position], clipped so float16 cache remains well behaved.
    return np.clip(logits.T, -40.0, 0.0).astype(np.float32)


def _build_pack(record, typewell_override=None):
    grid_md = np.linspace(record.md[0], record.md[-1], GRID_POINTS, dtype=np.float32)
    horizontal = _interp_channels(record.md, record.horizontal, grid_md)
    native_condition = _interp_channels(record.md, record.condition, grid_md)
    condition = np.concatenate([horizontal[:4], native_condition[:9]], axis=0).astype(np.float32)
    typewell = record.typewell if typewell_override is None else typewell_override
    emissions = np.stack(
        [
            _reference_emission(horizontal, typewell),
            _reference_emission(horizontal, record.heel),
        ],
        axis=0,
    ).astype(np.float32)
    target = np.interp(
        grid_md,
        record.md,
        record.truth.astype(np.float64) - record.anchor_tvt,
    ).astype(np.float32)
    anchor_grid = int(np.argmin(np.abs(grid_md - record.md[record.anchor_index])))
    target -= target[anchor_grid]
    target_state = np.rint(target - STATE_OFFSETS[0]).astype(np.int64)
    target_state = np.clip(target_state, 0, N_STATES - 1)
    supervised = grid_md >= record.md[record.anchor_index]
    eval_md = record.md[record.eval_index].astype(np.float32)
    return CostPack(
        well=record.well,
        emissions=emissions,
        condition=condition,
        target=target,
        target_state=target_state,
        supervised=supervised,
        grid_md=grid_md,
        anchor_grid=anchor_grid,
        anchor_md=float(record.md[record.anchor_index]),
        anchor_tvt=float(record.anchor_tvt),
        truth=record.truth[record.eval_index].astype(np.float32),
        eval_md=eval_md,
        eval_distance=(eval_md - record.md[record.anchor_index]).astype(np.float32),
        eval_rows=record.eval_rows,
    )


def _save_pack(pack):
    np.savez_compressed(
        _pack_path(pack.well),
        well=np.asarray(pack.well),
        emissions=pack.emissions.astype(np.float16),
        condition=pack.condition.astype(np.float16),
        target=pack.target.astype(np.float32),
        target_state=pack.target_state.astype(np.int16),
        supervised=pack.supervised.astype(np.uint8),
        grid_md=pack.grid_md.astype(np.float32),
        anchor_grid=np.asarray([pack.anchor_grid], np.int16),
        anchor_md=np.asarray([pack.anchor_md], np.float32),
        anchor_tvt=np.asarray([pack.anchor_tvt], np.float32),
        truth=pack.truth.astype(np.float32),
        eval_md=pack.eval_md.astype(np.float32),
        eval_distance=pack.eval_distance.astype(np.float32),
        eval_rows=np.asarray([pack.eval_rows], np.int32),
        cache_version=np.asarray([PACK_CACHE_VERSION], np.int16),
    )


def _load_pack(well):
    with np.load(_pack_path(well), allow_pickle=False) as data:
        if int(data["cache_version"][0]) != PACK_CACHE_VERSION:
            raise RuntimeError("stale cost pack")
        return CostPack(
            well=str(data["well"].item()),
            emissions=data["emissions"].astype(np.float32),
            condition=data["condition"].astype(np.float32),
            target=data["target"].astype(np.float32),
            target_state=data["target_state"].astype(np.int64),
            supervised=data["supervised"].astype(bool),
            grid_md=data["grid_md"].astype(np.float32),
            anchor_grid=int(data["anchor_grid"][0]),
            anchor_md=float(data["anchor_md"][0]),
            anchor_tvt=float(data["anchor_tvt"][0]),
            truth=data["truth"].astype(np.float32),
            eval_md=data["eval_md"].astype(np.float32),
            eval_distance=data["eval_distance"].astype(np.float32),
            eval_rows=int(data["eval_rows"][0]),
        )


def _ensure_packs(wells):
    rows = []
    for number, well in enumerate(wells, 1):
        try:
            if _pack_path(well).exists():
                try:
                    pack = _load_pack(well)
                    source = "cache"
                except Exception:
                    pack = _build_pack(_load_record(well))
                    _save_pack(pack)
                    source = "rebuilt"
            else:
                pack = _build_pack(_load_record(well))
                _save_pack(pack)
                source = "built"
            rows.append({"well": well, "status": "ok", "source": source, "rows": pack.eval_rows})
        except Exception as exc:
            rows.append({"well": well, "status": "failed", "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"cost-map packs {number}/{len(wells)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "pack_report.csv", index=False)
    failures = frame[frame["status"] != "ok"]
    if len(failures):
        raise RuntimeError(f"{len(failures)} cost-pack failures: {failures.head().to_dict('records')}")
    return frame


class StateTokenCostMapTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_input = nn.Sequential(
            nn.Linear(2 * GRID_POINTS, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.GELU(),
        )
        self.condition_input = nn.Sequential(
            nn.Linear(GRID_POINTS, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.GELU(),
        )
        self.state_embedding = nn.Parameter(torch.zeros(1, N_STATES, MODEL_DIM))
        self.condition_embedding = nn.Parameter(torch.zeros(1, N_CONDITION, MODEL_DIM))
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.output = nn.Sequential(nn.LayerNorm(MODEL_DIM), nn.Linear(MODEL_DIM, GRID_POINTS))
        self.typewell_residual = nn.Parameter(torch.tensor(0.0))
        self.heel_residual = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.state_embedding, std=0.02)
        nn.init.normal_(self.condition_embedding, std=0.02)

    def forward(self, emissions, condition):
        # emissions [B, 2, S, T] -> S tokens carrying complete trajectories.
        state_values = emissions.permute(0, 2, 1, 3).flatten(2)
        state_tokens = self.state_input(state_values) + self.state_embedding
        condition_tokens = self.condition_input(condition) + self.condition_embedding
        encoded = self.encoder(torch.cat([state_tokens, condition_tokens], dim=1))
        logits = self.output(encoded[:, :N_STATES])
        logits = (
            logits
            + F.softplus(self.typewell_residual) * emissions[:, 0]
            + F.softplus(self.heel_residual) * emissions[:, 1]
        )
        return logits


def _tensor_pack(pack, device, mode="full", donor=None):
    emissions = pack.emissions.copy()
    condition = pack.condition.copy()
    if mode == "no_gr":
        emissions[:] = 0.0
        condition[:4] = 0.0
    elif mode == "shuffled_typewell":
        if donor is None:
            raise ValueError("shuffled_typewell requires donor")
        record = _load_record(pack.well)
        donor_record = _load_record(donor.well)
        emissions[0] = _build_pack(record, typewell_override=donor_record.typewell).emissions[0]
    elif mode != "full":
        raise ValueError(mode)
    return (
        torch.from_numpy(emissions)[None].to(device),
        torch.from_numpy(condition)[None].to(device),
    )


def _loss(model, pack, device, row_weight):
    emissions, condition = _tensor_pack(pack, device)
    logits = model(emissions, condition)[0]
    target_state = torch.from_numpy(pack.target_state).to(device)
    target = torch.from_numpy(pack.target).to(device)
    mask = torch.from_numpy(pack.supervised).to(device)
    ce = F.cross_entropy(logits[:, mask].transpose(0, 1), target_state[mask])
    probability = F.softmax(logits.float(), dim=0)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    expected = torch.sum(probability * states, dim=0)
    path = torch.mean((expected[mask] - target[mask]).square()) / 64.0
    slope = torch.mean(
        (expected[1:] - expected[:-1] - target[1:] + target[:-1]).square()
    ) / 4.0
    curvature = torch.mean(
        (expected[2:] - 2 * expected[1:-1] + expected[:-2]
         - target[2:] + 2 * target[1:-1] - target[:-2]).square()
    ) / 4.0
    total = float(row_weight) * (
        LOSS_CE * ce + LOSS_PATH * path + LOSS_SLOPE * slope + LOSS_CURVATURE * curvature
    )
    return total, ce.detach(), path.detach(), slope.detach()


def _decode_dp(logits, anchor_grid):
    log_probability = logits - np.max(logits, axis=0, keepdims=True)
    start = int(anchor_grid)
    zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    length = logits.shape[1] - start
    score = np.full(N_STATES, -np.inf, np.float64)
    score[zero_state] = 0.0
    back = np.zeros((length, N_STATES), np.int16)
    for local_t, t in enumerate(range(start + 1, logits.shape[1]), 1):
        best = np.full(N_STATES, -np.inf, np.float64)
        source_best = np.zeros(N_STATES, np.int16)
        for delta in range(-DP_MAX_JUMP, DP_MAX_JUMP + 1):
            if delta >= 0:
                source = np.arange(0, N_STATES - delta)
                destination = source + delta
            else:
                destination = np.arange(0, N_STATES + delta)
                source = destination - delta
            candidate = score[source] - DP_TRANSITION_LAMBDA * float(delta * delta)
            improve = candidate > best[destination]
            if np.any(improve):
                selected_destination = destination[improve]
                best[selected_destination] = candidate[improve]
                source_best[selected_destination] = source[improve]
        score = best + log_probability[:, t]
        back[local_t] = source_best
    states = np.zeros(logits.shape[1], np.int16)
    states[-1] = int(np.argmax(score))
    for local_t in range(length - 1, 0, -1):
        states[start + local_t - 1] = back[local_t, states[start + local_t]]
    states[: start + 1] = zero_state
    return STATE_OFFSETS[states]


@torch.no_grad()
def _predict(model, pack, device, mode="full", donor=None):
    model.eval()
    emissions, condition = _tensor_pack(pack, device, mode=mode, donor=donor)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(emissions, condition)[0].float()
    probability = F.softmax(logits, dim=0)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    mean_offset = torch.sum(probability * states, dim=0).cpu().numpy()
    logits_np = logits.cpu().numpy()
    dp_offset = _decode_dp(logits_np, pack.anchor_grid)

    support = pack.grid_md > pack.anchor_md
    support_md = np.r_[pack.anchor_md, pack.grid_md[support]]
    mean_support = np.r_[0.0, mean_offset[support]]
    dp_support = np.r_[0.0, dp_offset[support]]
    mean_prediction = pack.anchor_tvt + np.interp(pack.eval_md, support_md, mean_support)
    dp_prediction = pack.anchor_tvt + np.interp(pack.eval_md, support_md, dp_support)
    return (
        pack.truth.astype(np.float64),
        mean_prediction.astype(np.float64),
        dp_prediction.astype(np.float64),
        pack.eval_distance.astype(np.float64),
    )


@torch.no_grad()
def _evaluate(model, packs, device, mode="full", save_rows=False):
    truths, means, dps, distances, wells = [], [], [], [], []
    for index, pack in enumerate(packs):
        donor = None
        if mode == "shuffled_typewell":
            donor = packs[(index + max(1, len(packs) // 2)) % len(packs)]
        truth, mean, dp, distance = _predict(model, pack, device, mode=mode, donor=donor)
        truths.append(truth)
        means.append(mean)
        dps.append(dp)
        distances.append(distance)
        wells.append(np.full(len(truth), pack.well, dtype="U16"))
    truth = np.concatenate(truths)
    mean = np.concatenate(means)
    dp = np.concatenate(dps)
    distance = np.concatenate(distances)
    well = np.concatenate(wells)
    mean_metrics, mean_bywell, mean_frame = _metrics(truth, mean, distance, well)
    dp_metrics, dp_bywell, dp_frame = _metrics(truth, dp, distance, well)
    payload = None
    if save_rows:
        payload = {
            "well": well,
            "truth": truth.astype(np.float32),
            "posterior_mean": mean.astype(np.float32),
            "global_dp": dp.astype(np.float32),
            "distance": distance.astype(np.float32),
        }
    return {"posterior_mean": mean_metrics, "global_dp": dp_metrics}, mean_bywell, dp_bywell, payload


def _fit(train_packs, valid_packs, device, epochs, phase):
    _seed_everything(SEED + len(train_packs))
    model = StateTokenCostMapTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=512.0)
    mean_rows = float(np.mean([pack.eval_rows for pack in train_packs]))
    rng = np.random.default_rng(SEED + epochs)
    history, best, stale = [], None, 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = np.zeros(4, np.float64)
        for step, index in enumerate(rng.permutation(len(train_packs)), 1):
            pack = train_packs[int(index)]
            weight = pack.eval_rows / max(mean_rows, 1.0)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, ce, path, slope = _loss(model, pack, device, weight)
                scaled = loss / GRAD_ACCUM_WELLS
            scaler.scale(scaled).backward()
            if step % GRAD_ACCUM_WELLS == 0 or step == len(train_packs):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            sums += [float(loss.detach().cpu()), float(ce.cpu()), float(path.cpu()), float(slope.cpu())]
        scheduler.step()
        interval = CAPACITY_REPORT_EVERY if phase == "capacity" else FOLD_REPORT_EVERY
        if not (epoch <= 2 or epoch % interval == 0 or epoch == epochs):
            continue
        metrics, _, _, _ = _evaluate(model, valid_packs, device)
        selection_rmse = min(metrics["posterior_mean"]["rmse"], metrics["global_dp"]["rmse"])
        row = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": sums[0] / len(train_packs),
            "train_ce": sums[1] / len(train_packs),
            "train_path": sums[2] / len(train_packs),
            "train_slope": sums[3] / len(train_packs),
            "posterior_mean_rmse": metrics["posterior_mean"]["rmse"],
            "global_dp_rmse": metrics["global_dp"]["rmse"],
            "selection_rmse": selection_rmse,
            "typewell_residual": float(F.softplus(model.typewell_residual).detach().cpu()),
            "heel_residual": float(F.softplus(model.heel_residual).detach().cpu()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("COSTMAP ITRANSFORMER EPOCH", row, flush=True)
        if best is None or selection_rmse < best["selection_rmse"] - 1e-5:
            best = {
                "epoch": epoch,
                "selection_rmse": selection_rmse,
                "metrics": metrics,
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            stale = 0
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "phase": phase,
                    "epoch": epoch,
                    "metrics": metrics,
                    "model": best["state"],
                },
                MODEL_DIR / f"best_{phase}.pt",
            )
        else:
            stale += 1
        pd.DataFrame(history).to_csv(OUT_DIR / f"history_{phase}.csv", index=False)
        if phase == "capacity" and selection_rmse <= CAPACITY_GATE_RMSE:
            print(f"capacity gate reached at epoch {epoch}", flush=True)
            break
        if phase != "capacity" and epoch >= MIN_EPOCHS and stale >= EARLY_STOPPING_REPORTS:
            print(f"fold2 early stopping at epoch {epoch}", flush=True)
            break
    if best is None:
        raise RuntimeError(f"No checkpoint for {phase}")
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history)


STARTED = time.time()
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a CUDA GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_everything(SEED)

folds, fold_path = _load_folds()
wells = folds["well"].tolist()
if len(wells) != 773:
    raise RuntimeError(f"Expected 773 wells, found {len(wells)}")
_ensure_records(wells)
_ensure_packs(wells)
packs = {well: _load_pack(well) for well in wells}

train_wells = folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].tolist()
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(train_wells) not in (463, 464) or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected Fold345/Fold2 sizes: {len(train_wells)}/{len(valid_wells)}")

capacity_wells = sorted(train_wells, key=lambda well: (-packs[well].eval_rows, well))[:CAPACITY_WELLS]
capacity_packs = [packs[well] for well in capacity_wells]
capacity_model, capacity_best, _ = _fit(
    capacity_packs, capacity_packs, device, CAPACITY_EPOCHS, "capacity"
)
capacity_pass = capacity_best["selection_rmse"] <= CAPACITY_GATE_RMSE
capacity = {
    "wells": len(capacity_packs),
    "parameters": _parameter_count(capacity_model),
    "best_epoch": capacity_best["epoch"],
    "selection_rmse": capacity_best["selection_rmse"],
    "metrics": capacity_best["metrics"],
    "gate_rmse_le": CAPACITY_GATE_RMSE,
    "pass": capacity_pass,
}
print("COSTMAP ITRANSFORMER CAPACITY", json.dumps(capacity, indent=2), flush=True)
if not capacity_pass:
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "24_well_state_token_capacity",
        "capacity": capacity,
        "strict_pass": False,
        "next": "FAIL -> state-token cost-map representation cannot fit 24 wells",
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    raise RuntimeError(json.dumps(verdict, indent=2))

del capacity_model
gc.collect()
torch.cuda.empty_cache()

train_packs = [packs[well] for well in train_wells]
valid_packs = [packs[well] for well in valid_wells]
model, best, history = _fit(train_packs, valid_packs, device, FOLD_EPOCHS, "fold2")
full, mean_bywell, dp_bywell, payload = _evaluate(model, valid_packs, device, save_rows=True)
mean_bywell.to_csv(OUT_DIR / "fold2_posterior_mean_bywell.csv", index=False)
dp_bywell.to_csv(OUT_DIR / "fold2_global_dp_bywell.csv", index=False)
np.savez_compressed(OUT_DIR / "fold2_oof.npz", **payload)

selected_mode = min(full, key=lambda mode: full[mode]["rmse"])
controls = {}
for mode in ("no_gr", "shuffled_typewell"):
    metrics, _, _, _ = _evaluate(model, valid_packs, device, mode=mode)
    controls[mode] = metrics

selected = full[selected_mode]
control_degradations = {
    mode: controls[mode][selected_mode]["rmse"] - selected["rmse"]
    for mode in controls
}
architecture_pass = selected["rmse"] <= ARCHITECTURE_GATE_RMSE
causal_pass = all(value >= CONTROL_DEGRADATION_GATE for value in control_degradations.values())
strict_pass = architecture_pass and causal_pass

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_complete_costmap_state_token_itransformer_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold_pf": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
        "fold_source": str(fold_path),
    },
    "representation": {
        "state_tokens": N_STATES,
        "grid_points": GRID_POINTS,
        "emission_channels": ["typewell", "visible_heel"],
        "condition_tokens": BASE_VARIABLE_NAMES,
        "parameters": _parameter_count(model),
    },
    "capacity": capacity,
    "selected_epoch": best["epoch"],
    "selected_mode": selected_mode,
    "fold2": full,
    "controls": controls,
    "control_degradations": control_degradations,
    "gates": {
        "capacity_rmse_le": CAPACITY_GATE_RMSE,
        "architecture_rmse_le": ARCHITECTURE_GATE_RMSE,
        "target_rmse_le": TARGET_GATE_RMSE,
        "each_control_degradation_ge": CONTROL_DEGRADATION_GATE,
    },
    "capacity_pass": capacity_pass,
    "architecture_pass": architecture_pass,
    "target_pass": selected["rmse"] <= TARGET_GATE_RMSE,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze architecture and run untouched Fold1"
        if strict_pass
        else "FAIL -> full cost-map iTransformer still does not transfer; close this architecture"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
