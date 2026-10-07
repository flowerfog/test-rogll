"""Learn cross-modal GR interval embeddings and gate them as an emission.

The model never predicts TVT directly.  Folds 4/5 train a horizontal-window
encoder and a typewell-window encoder with InfoNCE plus explicit +/-24 ft hard
negatives.  Fold 3 selects the epoch and emission dose; Fold 2 confirms once;
Fold 1 remains untouched.  Only raw logs and training truth are consumed.
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _source_path() -> Path:
    configured = os.environ.get("ROGII_V38_SOURCE", "").strip()
    candidates = [
        Path("/root/autodl-tmp/phase_transport_specificity_v38_cell.py"),
        Path("phase_transport_specificity_v38_cell.py"),
    ]
    if configured:
        candidates.insert(0, Path(configured))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Upload phase_transport_specificity_v38_cell.py to /root/autodl-tmp"
    )


def _bootstrap() -> tuple[dict, Path]:
    path = _source_path()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    name = "phase_transport_v38_definitions_for_v42"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.__dict__, path


V38, SOURCE_PATH = _bootstrap()
V37 = V38["V"]
OOF_ROOT = V38["OOF_ROOT"]
STATE_OFFSETS = np.asarray(V38["STATE_OFFSETS"], np.float64)
N_STATES = int(V38["N_STATES"])
ALIAS_FT = float(V38["ALIAS_FT"])

_load_folds = V38["_load_folds"]
_ensure_records = V38["_ensure_records"]
_ensure_packs = V38["_ensure_packs"]
_load_record = V38["_load_record"]
_load_pack = V38["_load_pack"]
_generic_typewell = V38["_generic_typewell"]
_shift_typewell = V38["_shift_typewell"]
_components_w2 = V38["_components"]
_normalise_states = V38["_normalise_states"]
_stable_order = V38["_stable_order"]
_truth_rank = V38["_truth_rank"]
_score_at_offset = V38["_score_at_offset"]
_path_metrics = V38["_path_metrics"]
_window_arrays = V37["_window_arrays"]

EXPERIMENT = "contrastive_interval_embedding_v42"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260727
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
MD_HALF_FT = 512.0
TVT_HALF_FT = 16.0
MAX_ROWS_PER_TRAIN_WELL = 192
MAX_ROWS_PER_VALID_WELL = 128
BATCH_SIZE = 256
MAX_EPOCHS = 14
PATIENCE = 4
LEARNING_RATE = 4.0e-4
WEIGHT_DECAY = 1.0e-4
TEMPERATURE = 0.08
EMBEDDING_DIM = 32
FROZEN_V38 = {"exact_dose": 0.10, "generic_dose": 0.10}
DOSES = (0.025, 0.05, 0.10, 0.20, 0.35)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


def _select_even(indices, limit):
    indices = np.asarray(indices, np.int64)
    if len(indices) <= limit:
        return indices
    take = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return indices[np.unique(take)]


def _prepare_windows(values, valid):
    values = np.asarray(values, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    output = np.zeros_like(values)
    x = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
    for row in range(len(values)):
        mask = valid[row]
        if int(mask.sum()) < 8:
            continue
        filled = np.interp(x, x[mask], values[row, mask]).astype(np.float32)
        design = np.column_stack((np.ones(len(x), np.float32), x))
        coef = np.linalg.lstsq(design[mask], filled[mask], rcond=None)[0]
        residual = filled - design @ coef
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        scale = max(1.4826 * float(mad), 1.0)
        output[row] = np.clip((residual - median) / scale, -8.0, 8.0)
    coverage = valid.mean(axis=1).astype(np.float32)
    channels = np.stack((output, valid.astype(np.float32)), axis=1)
    return channels.astype(np.float32), coverage >= 0.75


def _well_windows(record, pack, reference_override=None):
    query, q_valid, reference, r_valid = _window_arrays(
        record, pack, MD_HALF_FT, TVT_HALF_FT, reference_override
    )
    query, q_usable = _prepare_windows(query, q_valid)
    reference, r_usable = _prepare_windows(reference, r_valid)
    reverse, reverse_usable = _prepare_windows(
        reference[:, 0, ::-1], reference[:, 1, ::-1] > 0.5
    )
    return query, q_usable, reference, r_usable, reverse, reverse_usable


def _nearest_state(offset):
    return np.argmin(np.abs(STATE_OFFSETS[None] - np.asarray(offset)[:, None]), axis=1)


def _training_cache(wells, records, packs, rows_per_well, label):
    cache, samples = {}, []
    margin = ALIAS_FT + 4.0
    for number, well in enumerate(wells, 1):
        record, pack = records[well], packs[well]
        query, qu, reference, ru, _, _ = _well_windows(record, pack)
        target = pack.target.astype(np.float64)
        eligible = np.flatnonzero(
            pack.supervised.astype(bool)
            & qu
            & (np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - margin)
        )
        eligible = _select_even(eligible, rows_per_well)
        state = _nearest_state(target[eligible])
        keep = ru[state]
        eligible, state = eligible[keep], state[keep]
        cache[well] = (query, reference)
        samples.extend((well, int(row), int(candidate)) for row, candidate in zip(eligible, state))
        if number % 50 == 0 or number == len(wells):
            print(f"V42 cache {label} {number}/{len(wells)} samples={len(samples)}", flush=True)
    minimum_samples = max(1000, 64 * len(wells))
    if len(samples) < minimum_samples:
        raise RuntimeError(
            f"Only {len(samples)} usable {label} interval pairs; "
            f"minimum={minimum_samples} for {len(wells)} wells"
        )
    return cache, samples


class IntervalTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 24, 7, padding=3), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv1d(24, 32, 5, padding=2, dilation=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv1d(32, 40, 5, padding=4, dilation=2), nn.GroupNorm(5, 40), nn.SiLU(),
            nn.Conv1d(40, 48, 5, padding=8, dilation=4), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(48, 64), nn.SiLU(), nn.Linear(64, EMBEDDING_DIM)
        )

    def forward(self, values):
        return F.normalize(self.projection(self.network(values)), dim=1)


class CrossModalIntervalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = IntervalTower()
        self.reference = IntervalTower()

    def encode_query(self, values):
        return self.query(values)

    def encode_reference(self, values):
        return self.reference(values)


def _augment(values, generator):
    output = values.clone()
    signal = output[:, :1]
    scale = torch.empty((len(output), 1, 1), device=output.device).uniform_(0.80, 1.20, generator=generator)
    bias = torch.empty((len(output), 1, 1), device=output.device).uniform_(-0.20, 0.20, generator=generator)
    signal.mul_(scale).add_(bias)
    signal.add_(0.06 * torch.randn(signal.shape, device=signal.device, generator=generator))
    dropout = torch.rand(signal.shape, device=signal.device, generator=generator) < 0.05
    signal.masked_fill_(dropout, 0.0)
    output[:, 1:2].masked_fill_(dropout, 0.0)
    return output


def _sample_batch(samples, cache, indices, device, augment, generator):
    query = np.stack([cache[samples[i][0]][0][samples[i][1]] for i in indices])
    state = np.asarray([samples[i][2] for i in indices], np.int64)
    reference = np.stack([cache[samples[i][0]][1][samples[i][2]] for i in indices])
    alias_steps = max(int(round(ALIAS_FT / np.median(np.diff(STATE_OFFSETS)))), 1)
    sign = np.where(np.arange(len(indices)) % 2 == 0, 1, -1)
    hard_state = np.clip(state + sign * alias_steps, 0, N_STATES - 1)
    hard = np.stack([cache[samples[i][0]][1][candidate] for i, candidate in zip(indices, hard_state)])
    query = torch.from_numpy(query).to(device)
    reference = torch.from_numpy(reference).to(device)
    hard = torch.from_numpy(hard).to(device)
    if augment:
        query = _augment(query, generator)
        reference = _augment(reference, generator)
        hard = _augment(hard, generator)
    return query, reference, hard


def _contrastive_loss(model, query, reference, hard):
    q = model.encode_query(query)
    r = model.encode_reference(reference)
    h = model.encode_reference(hard)
    logits = q @ r.T / TEMPERATURE
    labels = torch.arange(len(q), device=q.device)
    forward = F.cross_entropy(logits, labels)
    reverse = F.cross_entropy(logits.T, labels)
    hard_logits = torch.stack(((q * r).sum(1), (q * h).sum(1)), dim=1) / TEMPERATURE
    hard_loss = F.cross_entropy(hard_logits, torch.zeros(len(q), dtype=torch.long, device=q.device))
    return 0.5 * (forward + reverse) + 0.5 * hard_loss, {
        "inbatch": float((0.5 * (forward + reverse)).detach().cpu()),
        "hard": float(hard_loss.detach().cpu()),
    }


def _train_epoch(model, optimizer, scaler, samples, cache, device, epoch):
    model.train()
    rng = np.random.default_rng(SEED + epoch)
    order = rng.permutation(len(samples))
    reports = []
    generator = torch.Generator(device=device).manual_seed(SEED + 1000 + epoch)
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start + BATCH_SIZE]
        if len(indices) < 32:
            continue
        batch = _sample_batch(samples, cache, indices, device, True, generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, parts = _contrastive_loss(model, *batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        reports.append({"loss": float(loss.detach().cpu()), **parts})
    return {key: float(np.mean([row[key] for row in reports])) for key in reports[0]}


@torch.no_grad()
def _validation_retrieval(model, samples, cache, device):
    model.eval()
    ranks = []
    for start in range(0, len(samples), BATCH_SIZE):
        subset = samples[start:start + BATCH_SIZE]
        by_well = {}
        for local, sample in enumerate(subset):
            by_well.setdefault(sample[0], []).append((local, sample))
        batch_ranks = np.zeros(len(subset), np.int64)
        for well, items in by_well.items():
            query = np.stack([cache[well][0][item[1][1]] for item in items])
            reference = cache[well][1]
            q = model.encode_query(torch.from_numpy(query).to(device))
            r = model.encode_reference(torch.from_numpy(reference).to(device))
            score = (q @ r.T).cpu().numpy()
            for row, (local, sample) in enumerate(items):
                target_score = score[row, sample[2]]
                batch_ranks[local] = 1 + int(np.sum(score[row] > target_score))
        ranks.append(batch_ranks)
    ranks = np.concatenate(ranks)
    return {
        "median_rank": float(np.median(ranks)),
        "top8_recall": float(np.mean(ranks <= 8)),
        "top32_recall": float(np.mean(ranks <= 32)),
    }


@torch.no_grad()
def _learned_profile(model, record, pack, device, reference_override=None, control="full"):
    query, qu, reference, ru, reverse, rru = _well_windows(record, pack, reference_override)
    if control == "window_shuffle":
        permutation = np.random.default_rng(SEED + 422).permutation(query.shape[2])
        query = query[:, :, permutation]
    elif control == "no_gr":
        query = query.copy()
        query[:, 0] = 0.0
    model.eval()
    q_parts = []
    for start in range(0, len(query), 512):
        q_parts.append(model.encode_query(torch.from_numpy(query[start:start + 512]).to(device)).cpu().numpy())
    q = np.concatenate(q_parts)
    r = model.encode_reference(torch.from_numpy(reference).to(device)).cpu().numpy()
    rr = model.encode_reference(torch.from_numpy(reverse).to(device)).cpu().numpy()
    score = np.maximum(r @ q.T, rr @ q.T)
    usable = (ru | rru)[:, None] & qu[None]
    score = np.where(usable, score, -40.0)
    score, _ = _normalise_states(score)
    return score.astype(np.float32)


_PROFILE_CACHE = {}


def _profiles(model, well, records, packs, generic_reference, device, control="full", donor=None):
    record, pack = records[well], packs[well]
    point, exact_w2, generic_w2 = _components_w2(record, pack, generic_reference)
    exact_override = None
    generic_override = None
    if control == "donor_exact":
        exact_override = records[donor].typewell
    elif control == "shift_exact_64":
        exact_override = _shift_typewell(record.typewell, 64.0)
    elif control == "shift_generic_64":
        generic_override = _shift_typewell(generic_reference, 64.0)
    key_exact = (well, "exact") if control == "full" else None
    key_generic = (well, "generic") if control == "full" else None
    if key_exact in _PROFILE_CACHE:
        exact = _PROFILE_CACHE[key_exact]
    else:
        exact = _learned_profile(model, record, pack, device, exact_override, control)
        if key_exact is not None:
            _PROFILE_CACHE[key_exact] = exact
    if key_generic in _PROFILE_CACHE:
        generic = _PROFILE_CACHE[key_generic]
    else:
        reference = generic_reference if generic_override is None else generic_override
        generic = _learned_profile(model, record, pack, device, reference, control)
        if key_generic is not None:
            _PROFILE_CACHE[key_generic] = generic
    return tuple(np.asarray(x, np.float64) for x in (point, exact_w2, generic_w2, exact, generic))


def _combine(values, recipe):
    point, exact_w2, generic_w2, exact, generic = values
    return (
        point + FROZEN_V38["exact_dose"] * exact_w2
        + FROZEN_V38["generic_dose"] * generic_w2
        + float(recipe.get("exact_dose", 0.0)) * exact
        + float(recipe.get("generic_dose", 0.0)) * generic
    )


def _recipe_name(recipe):
    return f"{recipe['mode']}_ed{recipe.get('exact_dose', 0):g}_gd{recipe.get('generic_dose', 0):g}"


def _menu():
    output = [{"mode": "baseline", "exact_dose": 0.0, "generic_dose": 0.0}]
    for dose in DOSES:
        output.extend((
            {"mode": "exact", "exact_dose": dose, "generic_dose": 0.0},
            {"mode": "generic", "exact_dose": 0.0, "generic_dose": dose},
            {"mode": "joint", "exact_dose": dose, "generic_dose": dose},
        ))
    return output


def _evaluate(model, wells, records, packs, generic_reference, device, recipe, stage, control="full", donors=None):
    ranks, distances, targets, argmax_values, alias_wins, path_items = [], [], [], [], [], []
    state_permutation = None
    if control == "state_shuffle":
        state_permutation = np.random.default_rng(SEED + 421).permutation(N_STATES)
    for number, well in enumerate(wells, 1):
        values = _profiles(
            model, well, records, packs, generic_reference, device, control,
            None if donors is None else donors[well],
        )
        score = _combine(values, recipe)
        if state_permutation is not None:
            baseline = _combine(values, {"mode": "baseline"})
            score = baseline + (score - baseline)[state_permutation]
        pack = packs[well]
        order = _stable_order(score)
        target = pack.target.astype(np.float64)
        mask = pack.supervised.astype(bool) & (
            np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - 5.0
        )
        rank = _truth_rank(order, target)
        argmax = STATE_OFFSETS[order[0]]
        argmax[:pack.anchor_grid + 1] = 0.0
        true_score = _score_at_offset(score, target)
        plus = _score_at_offset(score, target + ALIAS_FT)
        minus = _score_at_offset(score, target - ALIAS_FT)
        alias_mask = mask & (np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - ALIAS_FT)
        ranks.append(rank[mask])
        distances.append((pack.grid_md - pack.anchor_md)[mask])
        targets.append(target[mask])
        argmax_values.append(argmax[mask])
        alias_wins.append((true_score > np.maximum(plus, minus))[alias_mask])
        path_items.append((pack, argmax))
        if number % 25 == 0 or number == len(wells):
            print(f"V42 {stage} {control} {_recipe_name(recipe)} {number}/{len(wells)}", flush=True)
    rank = np.concatenate(ranks)
    distance = np.concatenate(distances)
    target = np.concatenate(targets)
    argmax = np.concatenate(argmax_values)
    alias = np.concatenate(alias_wins)
    deep = distance > 2000.0
    error = argmax - target
    path = _path_metrics(path_items)
    return {
        "stage": stage, "recipe": _recipe_name(recipe), "mode": recipe["mode"],
        "control": control, "wells": int(len(wells)), "grid_rows": int(len(rank)),
        "median_rank": float(np.median(rank)), "top8_recall": float(np.mean(rank <= 8)),
        "top32_recall": float(np.mean(rank <= 32)),
        "deep_top32_recall": float(np.mean(rank[deep] <= 32)),
        "alias24_win_rate": float(np.mean(alias)),
        "argmax_grid_rmse": float(np.sqrt(np.mean(error * error))),
        "path_rmse": float(path["rmse"]),
        "path_far_gt2000_rmse": float(path["far_gt2000_rmse"]),
    }


def _objective(row):
    return 1.25 * row["top32_recall"] + row["deep_top32_recall"] + 0.5 * row["alias24_win_rate"] - 0.001 * row["argmax_grid_rmse"]


STARTED = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
train_wells = folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"].tolist()
fold3_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
if len(train_wells) < 300 or len(fold3_wells) < 154 or len(fold2_wells) < 154:
    raise RuntimeError(f"Unexpected Fold45/Fold3/Fold2 sizes: {len(train_wells)}/{len(fold3_wells)}/{len(fold2_wells)}")

rng = np.random.default_rng(SEED + 42)
fold3_order = rng.permutation(fold3_wells).tolist()
discovery_wells = fold3_order[:DISCOVERY_WELLS]
confirmation_wells = fold3_order[DISCOVERY_WELLS:]
needed = sorted(set(train_wells + fold3_wells + fold2_wells))
_ensure_records(needed)
_ensure_packs(needed)
records = {well: _load_record(well) for well in needed}
packs = {well: _load_pack(well) for well in needed}
generic_reference = _generic_typewell(records, train_wells)

train_cache, train_samples = _training_cache(
    train_wells, records, packs, MAX_ROWS_PER_TRAIN_WELL, "Fold45"
)
valid_cache, valid_samples = _training_cache(
    discovery_wells, records, packs, MAX_ROWS_PER_VALID_WELL, "Fold3-discovery"
)

model = CrossModalIntervalEncoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
checkpoint = MODEL_DIR / "best_interval_encoder.pt"
history, best_score, best_epoch, bad_epochs = [], -np.inf, 0, 0
for epoch in range(1, MAX_EPOCHS + 1):
    training = _train_epoch(model, optimizer, scaler, train_samples, train_cache, device, epoch)
    retrieval = _validation_retrieval(model, valid_samples, valid_cache, device)
    report = {"epoch": epoch, **{f"train_{k}": v for k, v in training.items()}, **{f"valid_{k}": v for k, v in retrieval.items()}}
    history.append(report)
    print("V42 EPOCH", report, flush=True)
    score = retrieval["top32_recall"] + 0.25 * retrieval["top8_recall"]
    if score > best_score + 1e-5:
        best_score, best_epoch, bad_epochs = score, epoch, 0
        torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
state = torch.load(checkpoint, map_location=device, weights_only=False)
model.load_state_dict(state["model"])

discovery_rows = []
for recipe in _menu():
    row = _evaluate(model, discovery_wells, records, packs, generic_reference, device, recipe, "fold3_discovery")
    row["objective"] = _objective(row)
    discovery_rows.append(row)
discovery = pd.DataFrame(discovery_rows).sort_values("objective", ascending=False)
discovery.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
nonbaseline = discovery.loc[~discovery["mode"].eq("baseline")]
selected_name = str(nonbaseline.iloc[0]["recipe"])
selected = next(recipe for recipe in _menu() if _recipe_name(recipe) == selected_name)
baseline = {"mode": "baseline", "exact_dose": 0.0, "generic_dose": 0.0}

confirmation = pd.DataFrame([
    _evaluate(model, confirmation_wells, records, packs, generic_reference, device, baseline, "fold3_confirmation"),
    _evaluate(model, confirmation_wells, records, packs, generic_reference, device, selected, "fold3_confirmation"),
])
confirmation.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)

donor_order = np.random.default_rng(SEED + 423).permutation(fold2_wells)
donors = dict(zip(fold2_wells, donor_order))
fold2_rows = [
    _evaluate(model, fold2_wells, records, packs, generic_reference, device, baseline, "fold2_application"),
    _evaluate(model, fold2_wells, records, packs, generic_reference, device, selected, "fold2_application"),
]
for control in ("state_shuffle", "window_shuffle", "donor_exact", "shift_exact_64", "shift_generic_64", "no_gr"):
    fold2_rows.append(_evaluate(
        model, fold2_wells, records, packs, generic_reference, device, selected,
        "fold2_control", control, donors,
    ))
fold2 = pd.DataFrame(fold2_rows)
fold2.to_csv(OUT_DIR / "fold2_application_controls.csv", index=False)

def _pick(frame, mode, control="full"):
    rows = frame.loc[frame["mode"].eq(mode) & frame["control"].eq(control)]
    if len(rows) != 1:
        raise RuntimeError(f"Cannot pick {mode}/{control}: {len(rows)} rows")
    return rows.iloc[0].to_dict()

fold3_base = _pick(confirmation, "baseline")
fold3_selected = _pick(confirmation, selected["mode"])
fold2_base = _pick(fold2, "baseline")
fold2_selected = _pick(fold2, selected["mode"])
controls = {name: _pick(fold2, selected["mode"], name) for name in ("state_shuffle", "window_shuffle", "donor_exact", "shift_exact_64", "shift_generic_64", "no_gr")}
fold3_gain = fold3_selected["top32_recall"] - fold3_base["top32_recall"]
fold2_gain = fold2_selected["top32_recall"] - fold2_base["top32_recall"]
deep_gain = fold2_selected["deep_top32_recall"] - fold2_base["deep_top32_recall"]
control_gain = {name: row["top32_recall"] - fold2_base["top32_recall"] for name, row in controls.items()}
exact_active = selected.get("exact_dose", 0.0) > 0.0
generic_active = selected.get("generic_dose", 0.0) > 0.0
gates = {
    "fold3_gain_ge_0p01": bool(fold3_gain >= 0.01),
    "fold2_gain_ge_0p01": bool(fold2_gain >= 0.01),
    "fold2_deep_gain_ge_0p01": bool(deep_gain >= 0.01),
    "state_shuffle_retains_le_30pct": bool(control_gain["state_shuffle"] <= max(0.30 * fold2_gain, 0.002)),
    "window_shuffle_retains_le_30pct": bool(control_gain["window_shuffle"] <= max(0.30 * fold2_gain, 0.002)),
    "no_gr_retains_le_30pct": bool(control_gain["no_gr"] <= max(0.30 * fold2_gain, 0.002)),
    "donor_weakens_if_exact_used": bool((not exact_active) or control_gain["donor_exact"] < fold2_gain),
    "exact_shift_weakens_if_used": bool((not exact_active) or control_gain["shift_exact_64"] < fold2_gain),
    "generic_shift_weakens_if_used": bool((not generic_active) or control_gain["shift_generic_64"] < fold2_gain),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_contrastive_training_Fold3_selection_Fold2_confirmation",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "sample_unit": "paired horizontal/typewell interval",
    "architecture": "two 1D-CNN towers with InfoNCE and +/-24ft hard negatives",
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "protocol": {"train_folds": list(PRIOR_FOLDS), "selection_fold": 3, "application_fold": 2, "untouched_fold": 1, "fold1_touched": False},
    "training": {"pairs": len(train_samples), "selected_epoch": best_epoch, "checkpoint": str(checkpoint)},
    "frozen_baseline": FROZEN_V38,
    "selected_recipe": selected,
    "fold3": {"baseline": fold3_base, "selected": fold3_selected},
    "fold2": {"baseline": fold2_base, "selected": fold2_selected},
    "incremental_gain": {"fold3_top32": fold3_gain, "fold2_top32": fold2_gain, "fold2_deep_top32": deep_gain},
    "controls": controls,
    "gates": gates,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": "PASS -> expose learned emission to a sequence decoder" if strict_pass else "FAIL -> interval identity is not transferable; close this raw-GR localization family",
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV42 DISCOVERY SCAN")
print(discovery.to_string(index=False))
print("\nV42 FOLD3 CONFIRMATION")
print(confirmation.to_string(index=False))
print("\nV42 FOLD2 APPLICATION AND CONTROLS")
print(fold2.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
