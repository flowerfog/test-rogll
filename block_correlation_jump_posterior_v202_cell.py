"""V202: block correlation posterior with heavy-tail S-domain decoding.

Train on Folds 4/5, select epoch and decoder on Fold 3, and open Fold 2 once
for the registered performance, independence, support, and causal gates.
Fold 1 is never loaded.  Inputs are same-well horizontal GR/geometry, visible
heel TVT, and the assigned typewell; no historical prediction is consumed.
"""
from __future__ import annotations

import ast
import gc
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT = "block_correlation_jump_posterior_v202"
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
OUT = OOF / EXPERIMENT
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STARTED = time.time()

SOURCE_CANDIDATES = (
    ROOT / "native_resolution_dualref_capacity_v0_cell.py",
    Path.cwd() / "native_resolution_dualref_capacity_v0_cell.py",
)
SOURCE = next((path for path in SOURCE_CANDIDATES if path.is_file()), None)
if SOURCE is None:
    raise FileNotFoundError("Upload native_resolution_dualref_capacity_v0_cell.py")


def _bootstrap(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED" for target in node.targets
        ):
            break
        body.append(node)
    module_tree = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module_tree)
    name = "native_v202_records"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(module_tree, str(path), "exec"), module.__dict__)
    return module.__dict__


V = _bootstrap(SOURCE)
STATE_OFFSETS = np.asarray(V["STATE_OFFSETS"], np.float32)
N_STATES = len(STATE_OFFSETS)
STATE_STEP = float(np.median(np.diff(STATE_OFFSETS)))
REFERENCE_STATE_START = int(V["REFERENCE_STATE_START"])
CACHE_DIR = Path(V["CACHE_DIR"])
DATA_ROOT = Path(V["DATA_ROOT"])
_load_record = V["_load_record"]
_build_record = V["_build_record"]
_save_record = V["_save_record"]

if N_STATES != 257 or abs(STATE_STEP - 1.0) > 1e-6 or not np.isclose(STATE_OFFSETS[[0, -1]], [-128.0, 128.0]).all():
    raise RuntimeError(f"Unexpected validated native state axis: {STATE_OFFSETS[[0, -1]]}/{N_STATES}")

FOLD_CANDIDATES = (
    OOF / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.is_file()), None)
if FOLD_PATH is None:
    raise FileNotFoundError("Canonical fold metadata not found")

BASE_PATH = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
BASE_COLUMN = "metadata_analog_w0p05"
SEED = 20260804
TRAIN_FOLDS = (4, 5)
SELECT_FOLD = 3
GATE_FOLD = 2
UNTOUCHED_FOLD = 1
BLOCK_FT = 128.0
MAX_EPOCHS = 16
MIN_EPOCHS = 5
PATIENCE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-5
TARGET_SIGMA_FT = 2.0
EMBEDDING = 24
MAP_CHANNELS = 24
TOPK_DECODE = 32
S_RATE_CHANGE_SCALE = 0.0069121911222675865
JUMP_SCALE = 0.05
TEMPERATURES = (0.75, 1.0, 1.5)
TRANSITION_WEIGHTS = (0.10, 0.30, 1.00)
JUMP_COSTS = (2.0, 4.0)


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


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_record(well: str):
    path = CACHE_DIR / f"{well}.npz"
    if not path.is_file():
        record = _build_record(well)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _save_record(record)
    return _load_record(well)


def _block_nodes(record):
    eval_rows = record.eval_index.astype(np.int64)
    distance = record.md[eval_rows].astype(np.float64) - float(record.md[record.anchor_index])
    block_id = np.floor(np.maximum(distance, 0.0) / BLOCK_FT).astype(np.int64)
    rows = []
    for _, positions in pd.Series(np.arange(len(eval_rows))).groupby(block_id, sort=True):
        local = eval_rows[positions.to_numpy(np.int64)]
        rows.append(int(local[len(local) // 2]))
    rows = np.asarray(rows, np.int64)
    label = record.truth[rows].astype(np.float32) - float(record.anchor_tvt)
    # Node placement is an inference-time operation and must not depend on
    # hidden truth or on whether the label happens to lie inside the state axis.
    if len(rows) < 2:
        raise RuntimeError(f"{record.well}: only {len(rows)} MD-defined block nodes")
    md = record.md[rows].astype(np.float32) - float(record.md[record.anchor_index])
    z = record.condition[4, rows].astype(np.float32) * 512.0
    return rows, label, md, z


class ResidualMap(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, (5, 3), padding=(2, 1), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.network(values))


class BlockCorrelationPosterior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, EMBEDDING, 9, padding=4),
            nn.GroupNorm(4, EMBEDDING),
            nn.GELU(),
            nn.Conv1d(EMBEDDING, EMBEDDING, 9, padding=16, dilation=4),
            nn.GroupNorm(4, EMBEDDING),
            nn.GELU(),
            nn.Conv1d(EMBEDDING, EMBEDDING, 9, padding=64, dilation=16),
            nn.GroupNorm(4, EMBEDDING),
            nn.GELU(),
        )
        self.condition = nn.Sequential(nn.Conv1d(10, 8, 1), nn.GELU(), nn.Conv1d(8, 8, 1))
        # cosine, dot, two raw differences, validity, state, distance, 8 conditions
        self.input = nn.Conv2d(15, MAP_CHANNELS, 1)
        self.context = nn.Sequential(
            ResidualMap(MAP_CHANNELS),
            ResidualMap(MAP_CHANNELS),
            ResidualMap(MAP_CHANNELS),
            ResidualMap(MAP_CHANNELS),
        )
        self.output = nn.Conv2d(MAP_CHANNELS, 1, 1)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, horizontal, condition, typewell, block_rows):
        horizontal_embedding = self.encoder(horizontal)
        type_embedding = self.encoder(typewell)
        query = horizontal_embedding[:, :, block_rows]
        key = type_embedding[:, :, REFERENCE_STATE_START:REFERENCE_STATE_START + N_STATES]
        query_norm = F.normalize(query, dim=1)
        key_norm = F.normalize(key, dim=1)
        cosine = torch.einsum("bcs,bct->bst", key_norm, query_norm)
        dot = torch.einsum("bcs,bct->bst", key, query) / math.sqrt(EMBEDDING)

        horizontal_gr = horizontal[:, 0, block_rows]
        horizontal_d1 = horizontal[:, 2, block_rows]
        type_gr = typewell[:, 0, REFERENCE_STATE_START:REFERENCE_STATE_START + N_STATES]
        type_d1 = typewell[:, 2, REFERENCE_STATE_START:REFERENCE_STATE_START + N_STATES]
        type_valid = typewell[:, 1, REFERENCE_STATE_START:REFERENCE_STATE_START + N_STATES]
        gr_difference = torch.abs(type_gr[:, :, None] - horizontal_gr[:, None, :])
        d1_difference = torch.abs(type_d1[:, :, None] - horizontal_d1[:, None, :])
        validity = type_valid[:, :, None].expand_as(cosine)
        state = torch.linspace(-1.0, 1.0, N_STATES, device=horizontal.device)[None, :, None].expand_as(cosine)
        progress = torch.linspace(0.0, 1.0, len(block_rows), device=horizontal.device)[None, None, :].expand_as(cosine)
        cond = self.condition(condition)[:, :, block_rows]
        cond = cond[:, :, None, :].expand(-1, -1, N_STATES, -1)
        scalar = torch.stack([cosine, dot, gr_difference, d1_difference, validity, state, progress], dim=1)
        values = torch.cat([scalar, cond], dim=1)
        logits = self.output(self.context(self.input(values)))[:, 0]
        return logits / torch.exp(self.log_temperature).clamp(0.25, 4.0)


def _inputs(record, rows, device, mode="full", donor=None):
    horizontal = record.horizontal.copy()
    condition = record.condition.copy()
    typewell = record.typewell.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        condition[9] = 0.0
    elif mode == "shuffled_typewell":
        if donor is None:
            raise ValueError("donor required")
        typewell = donor.typewell.copy()
    return (
        torch.from_numpy(horizontal[None]).to(device),
        torch.from_numpy(condition[None]).to(device),
        torch.from_numpy(typewell[None]).to(device),
        torch.from_numpy(rows.astype(np.int64)).to(device),
    )


def _loss(logits, label, md, z):
    states = torch.from_numpy(STATE_OFFSETS).to(logits.device)[:, None]
    target = torch.from_numpy(label).to(logits.device)[None]
    supervised = torch.isfinite(target[0]) & (target[0] >= states[0, 0]) & (target[0] <= states[-1, 0])
    if not torch.any(supervised):
        raise RuntimeError("record has no supervised block nodes inside the state axis")
    supervised_target = target[:, supervised]
    distribution = torch.exp(-0.5 * ((states - supervised_target) / TARGET_SIGMA_FT) ** 2)
    distribution /= distribution.sum(dim=0, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(logits[0], dim=0)
    ce = -(distribution * log_probability[:, supervised]).sum(dim=0).mean()
    probability = F.softmax(logits[0], dim=0)
    expected = (probability * states).sum(dim=0)
    path = F.smooth_l1_loss(expected[supervised] / 8.0, target[0, supervised] / 8.0)
    if len(label) >= 3:
        md_tensor = torch.from_numpy(md).to(logits.device)
        z_tensor = torch.from_numpy(z).to(logits.device)
        s = expected + z_tensor
        rate = (s[1:] - s[:-1]) / (md_tensor[1:] - md_tensor[:-1]).clamp_min(1.0)
        acceleration = rate[1:] - rate[:-1]
        transition = torch.mean(torch.sqrt(acceleration**2 + (0.25 * S_RATE_CHANGE_SCALE) ** 2))
        transition = transition / S_RATE_CHANGE_SCALE
    else:
        transition = ce * 0.0
    return ce + 0.20 * path + 0.01 * transition, {
        "ce": float(ce.detach()), "path": float(path.detach()), "transition": float(transition.detach())
    }


def _posterior_offsets(logits: np.ndarray, temperature: float):
    scaled = logits.astype(np.float64) / temperature
    scaled -= np.max(scaled, axis=0, keepdims=True)
    probability = np.exp(scaled)
    probability /= probability.sum(axis=0, keepdims=True)
    return STATE_OFFSETS.astype(np.float64) @ probability


def _jump_penalty(change, weight, jump_cost):
    regular = np.abs(change) / S_RATE_CHANGE_SCALE
    jump = jump_cost + np.abs(change) / JUMP_SCALE
    return weight * np.minimum(regular, jump)


def _second_order_decode(logits, md, z, temperature, weight, jump_cost):
    states, steps = logits.shape
    if steps < 2:
        return _posterior_offsets(logits, temperature)
    logp = logits.astype(np.float64) / temperature
    logp -= np.logaddexp.reduce(logp, axis=0, keepdims=True)
    candidates = []
    for step in range(steps):
        chosen = np.argpartition(logp[:, step], -TOPK_DECODE)[-TOPK_DECODE:]
        candidates.append(chosen[np.argsort(logp[chosen, step])[::-1]])
    zero = int(np.argmin(np.abs(STATE_OFFSETS)))
    if zero not in candidates[0]:
        candidates[0][-1] = zero

    c0, c1 = candidates[0], candidates[1]
    score = np.full((len(c0), len(c1)), -np.inf)
    anchor_md = 0.0
    anchor_z = 0.0
    for i, left in enumerate(c0):
        rate0 = (float(STATE_OFFSETS[left]) + float(z[0]) - anchor_z) / max(float(md[0] - anchor_md), 1.0)
        for j, right in enumerate(c1):
            rate1 = (
                float(STATE_OFFSETS[right] + z[1] - STATE_OFFSETS[left] - z[0])
                / max(float(md[1] - md[0]), 1.0)
            )
            score[i, j] = logp[left, 0] + logp[right, 1] - _jump_penalty(rate1 - rate0, weight, jump_cost)
    backs = {}
    for step in range(2, steps):
        previous_left, previous_right = candidates[step - 2], candidates[step - 1]
        current = candidates[step]
        updated = np.full((len(previous_right), len(current)), -np.inf)
        back = np.full((len(previous_right), len(current)), -1, np.int16)
        for j, middle in enumerate(previous_right):
            rate_previous = (
                STATE_OFFSETS[middle] + z[step - 1] - STATE_OFFSETS[previous_left] - z[step - 2]
            ) / max(float(md[step - 1] - md[step - 2]), 1.0)
            for k, right in enumerate(current):
                rate_new = float(
                    STATE_OFFSETS[right] + z[step] - STATE_OFFSETS[middle] - z[step - 1]
                ) / max(float(md[step] - md[step - 1]), 1.0)
                values = score[:, j] - _jump_penalty(rate_new - rate_previous, weight, jump_cost)
                best = int(np.argmax(values))
                updated[j, k] = values[best] + logp[right, step]
                back[j, k] = best
        score = updated
        backs[step] = back

    j, k = np.unravel_index(np.argmax(score), score.shape)
    path = np.empty(steps, np.int16)
    path[-2], path[-1] = j, k
    for step in range(steps - 1, 1, -1):
        i = int(backs[step][j, k])
        path[step - 2] = i
        k, j = j, i
    state_indices = np.asarray([candidates[step][path[step]] for step in range(steps)], np.int64)
    offsets = STATE_OFFSETS[state_indices].astype(np.float64)
    # Truth-free sub-state parabolic refinement.
    for step, state in enumerate(state_indices):
        if 0 < state < states - 1:
            left, centre, right = logp[state - 1:state + 2, step]
            denominator = left - 2.0 * centre + right
            if denominator < -1e-8:
                offsets[step] += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)) * STATE_STEP
    return offsets


def _row_prediction(record, rows, offsets):
    support_md = np.r_[float(record.md[record.anchor_index]), record.md[rows].astype(np.float64)]
    support_tvt = np.r_[float(record.anchor_tvt), float(record.anchor_tvt) + offsets]
    return np.interp(record.md[record.eval_index].astype(np.float64), support_md, support_tvt)


def _metrics(parts, endpoint):
    truth = np.concatenate([part["truth"] for part in parts])
    prediction = np.concatenate([part["prediction"] for part in parts])
    distance = np.concatenate([part["distance"] for part in parts])
    wells = np.concatenate([np.full(len(part["truth"]), part["well"], dtype="U16") for part in parts])
    error = prediction - truth
    codes, names = pd.factorize(wells, sort=False)
    counts = np.bincount(codes)
    well_rmse = np.sqrt(np.bincount(codes, weights=error**2) / counts)
    return {
        "endpoint": endpoint, "rows": len(error), "wells": len(names),
        "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[distance >= 2000.0] ** 2))),
    }


@torch.no_grad()
def _logits_for_records(model, records, device, mode="full"):
    model.eval()
    output = []
    for index, record in enumerate(records):
        rows, label, md, z = _block_nodes(record)
        donor = records[(index + max(1, len(records) // 2)) % len(records)] if mode == "shuffled_typewell" else None
        logits = model(*_inputs(record, rows, device, mode=mode, donor=donor))[0].float().cpu().numpy()
        output.append({"record": record, "rows": rows, "label": label, "md": md, "z": z, "logits": logits})
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"V202 inference {mode} {index + 1}/{len(records)}", flush=True)
    return output


def _evaluate_logits(items, decoder):
    parts = []
    for item in items:
        record = item["record"]
        if decoder["kind"] == "mean":
            offsets = _posterior_offsets(item["logits"], decoder["temperature"])
        else:
            offsets = _second_order_decode(
                item["logits"], item["md"], item["z"], decoder["temperature"],
                decoder["transition_weight"], decoder["jump_cost"],
            )
        prediction = _row_prediction(record, item["rows"], offsets)
        eval_rows = record.eval_index
        parts.append({
            "well": record.well,
            "row_idx": eval_rows.astype(np.int64),
            "truth": record.truth[eval_rows].astype(np.float64),
            "prediction": prediction,
            "distance": record.md[eval_rows].astype(np.float64) - float(record.md[record.anchor_index]),
        })
    return parts


def _top8_oracle(items):
    parts = []
    for item in items:
        record = item["record"]
        top = np.argpartition(item["logits"], -8, axis=0)[-8:]
        candidates = STATE_OFFSETS[top]
        nearest = np.argmin(np.abs(candidates - item["label"][None, :]), axis=0)
        offsets = candidates[nearest, np.arange(candidates.shape[1])]
        prediction = _row_prediction(record, item["rows"], offsets)
        eval_rows = record.eval_index
        parts.append({
            "well": record.well, "row_idx": eval_rows.astype(np.int64),
            "truth": record.truth[eval_rows].astype(np.float64), "prediction": prediction,
            "distance": record.md[eval_rows].astype(np.float64) - float(record.md[record.anchor_index]),
        })
    return parts


folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
counts = folds.groupby("fold")["well"].nunique().to_dict()
if counts != {1: 155, 2: 155, 3: 154, 4: 155, 5: 154}:
    raise RuntimeError(f"Unexpected folds: {counts}")

train_wells = folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].sort_values().tolist()
select_wells = folds.loc[folds["fold"].eq(SELECT_FOLD), "well"].sort_values().tolist()
for number, well in enumerate([*train_wells, *select_wells], 1):
    _ensure_record(well)
    if number % 100 == 0 or number == len(train_wells) + len(select_wells):
        print(f"V202 records {number}/{len(train_wells) + len(select_wells)}", flush=True)
train_records = [_load_record(well) for well in train_wells]
select_records = [_load_record(well) for well in select_wells]

if not torch.cuda.is_available():
    raise RuntimeError("V202 requires CUDA")
device = torch.device("cuda")
_seed(SEED)
model = BlockCorrelationPosterior().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
best = None
history = []
stale = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    rng = np.random.default_rng(SEED + epoch)
    sums = {"loss": 0.0, "ce": 0.0, "path": 0.0, "transition": 0.0}
    for sequence, index in enumerate(rng.permutation(len(train_records)), 1):
        record = train_records[int(index)]
        rows, label, md, z = _block_nodes(record)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*_inputs(record, rows, device))
        loss, diagnostics = _loss(logits, label, md, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        sums["loss"] += float(loss.detach())
        for key, value in diagnostics.items():
            sums[key] += value
        if sequence % 100 == 0:
            print(f"V202 epoch{epoch} train {sequence}/{len(train_records)}", flush=True)
    select_items = _logits_for_records(model, select_records, device)
    select_parts = _evaluate_logits(select_items, {"kind": "mean", "temperature": 1.0})
    select_metrics = _metrics(select_parts, "fold3_posterior_mean")
    row = {"epoch": epoch, **{f"train_{key}": value / len(train_records) for key, value in sums.items()}, **select_metrics}
    history.append(row)
    print("V202 EPOCH", row, flush=True)
    if best is None or select_metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
        best = {"epoch": epoch, "metrics": select_metrics, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        torch.save(best, MODEL_DIR / "fold45_best.pt")
        stale = 0
    else:
        stale += 1
    if epoch >= MIN_EPOCHS and stale >= PATIENCE:
        break

pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
model.load_state_dict(best["model"])
select_items = _logits_for_records(model, select_records, device)
decoder_rows = []
decoders = [{"kind": "mean", "temperature": temperature} for temperature in TEMPERATURES]
decoders += [
    {"kind": "jump_viterbi", "temperature": temperature, "transition_weight": weight, "jump_cost": jump}
    for temperature in TEMPERATURES for weight in TRANSITION_WEIGHTS for jump in JUMP_COSTS
]
for decoder in decoders:
    metrics = _metrics(_evaluate_logits(select_items, decoder), "fold3_decoder")
    decoder_rows.append({**decoder, **metrics})
decoder_scan = pd.DataFrame(decoder_rows).sort_values("rmse", ignore_index=True)
decoder_scan.to_csv(OUT / "fold3_decoder_scan.csv", index=False)
selected_decoder = {key: decoder_scan.iloc[0][key] for key in decoder_scan.columns if key in {"kind", "temperature", "transition_weight", "jump_cost"} and pd.notna(decoder_scan.iloc[0][key])}
selected_decoder["temperature"] = float(selected_decoder["temperature"])
if selected_decoder["kind"] != "mean":
    selected_decoder["transition_weight"] = float(selected_decoder["transition_weight"])
    selected_decoder["jump_cost"] = float(selected_decoder["jump_cost"])

# Fold2 is opened once after all model and decoder choices are frozen.
gate_wells = folds.loc[folds["fold"].eq(GATE_FOLD), "well"].sort_values().tolist()
for number, well in enumerate(gate_wells, 1):
    _ensure_record(well)
    if number % 50 == 0 or number == len(gate_wells):
        print(f"V202 Fold2 records {number}/{len(gate_wells)}", flush=True)
gate_records = [_load_record(well) for well in gate_wells]
full_items = _logits_for_records(model, gate_records, device, mode="full")
no_gr_items = _logits_for_records(model, gate_records, device, mode="no_gr")
shuffled_items = _logits_for_records(model, gate_records, device, mode="shuffled_typewell")
full_parts = _evaluate_logits(full_items, selected_decoder)
no_gr_parts = _evaluate_logits(no_gr_items, selected_decoder)
shuffled_parts = _evaluate_logits(shuffled_items, selected_decoder)
oracle_parts = _top8_oracle(full_items)
fold2_metrics = {
    "full": _metrics(full_parts, "full"),
    "no_gr": _metrics(no_gr_parts, "no_gr"),
    "shuffled_typewell": _metrics(shuffled_parts, "shuffled_typewell"),
    "posterior_top8_oracle": _metrics(oracle_parts, "posterior_top8_oracle"),
}

full_rows = pd.concat([
    pd.DataFrame({"well": part["well"], "row_idx": part["row_idx"], "tvt_true": part["truth"], "prediction": part["prediction"], "md_since": part["distance"]})
    for part in full_parts
], ignore_index=True)
full_rows.to_parquet(OUT / "fold2_oof.parquet", index=False)

detrended_corr = None
if BASE_PATH.is_file():
    base = pd.read_parquet(BASE_PATH, columns=["well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN])
    base["well"] = base["well"].astype(str)
    base = base.loc[base["fold"].eq(GATE_FOLD)].merge(full_rows[["well", "row_idx", "prediction"]], on=["well", "row_idx"], validate="one_to_one")
    distance = base["md_since"].to_numpy(np.float64)
    edges = np.unique(np.quantile(distance, np.linspace(0, 1, 65)))
    labels = np.clip(np.searchsorted(edges, distance, side="right") - 1, 0, len(edges) - 2)
    errors = []
    for column in (BASE_COLUMN, "prediction"):
        error = base[column].to_numpy(np.float64) - base["tvt_true"].to_numpy(np.float64)
        means = np.bincount(labels, weights=error, minlength=len(edges) - 1) / np.maximum(np.bincount(labels, minlength=len(edges) - 1), 1)
        errors.append(error - means[labels])
    detrended_corr = float(np.corrcoef(errors[0], errors[1])[0, 1])

no_gr_degradation = fold2_metrics["no_gr"]["rmse"] - fold2_metrics["full"]["rmse"]
shuffle_degradation = fold2_metrics["shuffled_typewell"]["rmse"] - fold2_metrics["full"]["rmse"]
checks = {
    "fold2_rmse_le_9p5": fold2_metrics["full"]["rmse"] <= 9.5,
    "fold2_detrended_corr_le_0p50": detrended_corr is not None and detrended_corr <= 0.50,
    "fold2_no_gr_degradation_ge_0p30": no_gr_degradation >= 0.30,
    "fold2_shuffled_typewell_degradation_ge_0p30": shuffle_degradation >= 0.30,
    "fold2_top8_oracle_rmse_le_5p5": fold2_metrics["posterior_top8_oracle"]["rmse"] <= 5.5,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_training_Fold3_selection_Fold2_once_only_gateway",
    "historical_prediction_inputs": [],
    "architecture": "128ft block shared-GR correlation posterior plus S-domain Laplace/jump decoder",
    "state_axis": {"centre": "last visible TVT", "minimum": -128.0, "maximum": 128.0, "step": 1.0, "v201c_required_radius": 96.0},
    "transition": {"coordinate": "S=TVT+Z", "rate_change_scale": S_RATE_CHANGE_SCALE, "jump_scale": JUMP_SCALE},
    "protocol": {"train_folds": list(TRAIN_FOLDS), "selection_fold": SELECT_FOLD, "gateway_fold": GATE_FOLD, "untouched_fold": UNTOUCHED_FOLD},
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "best_epoch": best["epoch"],
    "selected_decoder": selected_decoder,
    "fold3_best_mean": best["metrics"],
    "fold2": fold2_metrics,
    "fold2_detrended_error_corr_with_v168": detrended_corr,
    "no_gr_degradation": no_gr_degradation,
    "shuffled_typewell_degradation": shuffle_degradation,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> retrain on Fold2345 with frozen epochs/decoder and apply Fold1 once" if strict_pass else "FAIL -> close block posterior CRF before Fold1",
}
_write_json(OUT / "verdict.json", verdict)
print("\nV202 FOLD3 DECODER TOP")
print(decoder_scan.head(12).to_string(index=False), flush=True)
print("\nV202 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT, flush=True)
