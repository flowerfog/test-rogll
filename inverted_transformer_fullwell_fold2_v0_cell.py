"""Matched temporal-Transformer vs iTransformer complete-well gateway.

Upload this file together with ``native_resolution_dualref_capacity_v0_cell.py``
to ``/root/autodl-tmp`` and run it as one Jupyter cell (or ``%run`` it).

The experiment uses only deployment-legal, same-well evidence: native horizontal
GR/geometry, visible TVT_input prefix, and the assigned typewell.  It first runs
a 24-well iTransformer capacity audit, then a matched Fold-2 A/B:

    train folds 3/4/5 -> validation fold 2; fold 1 remains untouched.

The iTransformer is genuinely inverted: each evidence trajectory is one token
and its complete 512-point well history is embedded before attention.  The
matched baseline instead treats the 512 positions as tokens.
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


# ---------------------------------------------------------------------------
# Bootstrap the validated native record builder without executing its ladder.
# ---------------------------------------------------------------------------


def _bootstrap_native():
    candidates = (
        Path("/root/autodl-tmp/native_resolution_dualref_capacity_v0_cell.py"),
        Path.cwd() / "native_resolution_dualref_capacity_v0_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload native_resolution_dualref_capacity_v0_cell.py to "
            "/root/autodl-tmp before running this cell."
        )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        definitions.append(node)
    module_name = "native_dualref_itransformer_definitions"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module.__dict__, source_path


V, NATIVE_SOURCE = _bootstrap_native()
Record = V["Record"]
DATA_ROOT = V["DATA_ROOT"]
OOF_ROOT = V["OOF_ROOT"]
RECORD_DIR = V["CACHE_DIR"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float32)
_build_record = V["_build_record"]
_load_record = V["_load_record"]
_save_record = V["_save_record"]
_record_path = V["_record_path"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


EXPERIMENT = "inverted_transformer_fullwell_fold2_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
PACK_DIR = OUT_DIR / "packs"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)

SEED = 20260724
REQUIRE_CUDA = True
TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
UNTOUCHED_FOLD = 1

GRID_POINTS = 512
PACK_CACHE_VERSION = 2
STATE_SCALE_FT = 128.0
EMISSION_TEMPERATURE = 0.35
MODEL_DIM = 96
HEADS = 4
LAYERS = 4
FF_DIM = 256
DROPOUT = 0.08

CAPACITY_WELLS = 24
CAPACITY_EPOCHS = 240
CAPACITY_GATE_RMSE = 1.25
CAPACITY_REPORT_EVERY = 20

FOLD_EPOCHS = 44
FOLD_REPORT_EVERY = 2
MIN_EPOCHS = 10
EARLY_STOPPING_REPORTS = 6
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 3e-5
GRADIENT_CLIP = 3.0
GRAD_ACCUM_WELLS = 4

LOSS_PATH = 1.0
LOSS_SLOPE = 0.08
LOSS_CURVATURE = 0.015
ARCHITECTURE_GATE_RMSE = 10.0
TARGET_GATE_RMSE = 7.5
CONTROL_DEGRADATION_GATE = 0.30

for directory in (OUT_DIR, MODEL_DIR, PACK_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def _rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _load_folds():
    fold_path = next((path for path in FOLD_CANDIDATES if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError("Cannot find fold-clean well_metadata.csv")
    frame = pd.read_csv(fold_path, dtype={"well": str})
    well_column = next((c for c in ("well", "well_id", "WELL") if c in frame), None)
    fold_column = next((c for c in ("fold", "outer_fold", "Fold") if c in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify fold columns: {frame.columns.tolist()}")
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last").sort_values(["fold", "well"])
    if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected folds 1-5")
    return frame.reset_index(drop=True), fold_path


def _ensure_records(wells):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, well in enumerate(wells, 1):
        try:
            if _record_path(well).exists():
                record = _load_record(well)
                source = "cache"
            else:
                record = _build_record(well)
                _save_record(record)
                source = "built"
            rows.append(
                {
                    "well": well,
                    "status": "ok",
                    "source": source,
                    "rows": record.eval_rows,
                    "axis_coverage": record.axis_coverage,
                }
            )
        except Exception as exc:
            rows.append({"well": well, "status": "failed", "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"native records {number}/{len(wells)}", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "record_report.csv", index=False)
    failures = report[report["status"] != "ok"]
    if len(failures):
        raise RuntimeError(f"{len(failures)} native records failed: {failures.head().to_dict('records')}")
    return report


# ---------------------------------------------------------------------------
# Fixed-grid whole-well evidence packs
# ---------------------------------------------------------------------------


@dataclass
class WellPack:
    well: str
    values: np.ndarray
    target: np.ndarray
    supervised: np.ndarray
    grid_md: np.ndarray
    anchor_grid: int
    anchor_md: float
    anchor_tvt: float
    truth: np.ndarray
    eval_md: np.ndarray
    eval_distance: np.ndarray
    eval_rows: int


BASE_VARIABLE_NAMES = [
    "horizontal_gr",
    "horizontal_gr_valid",
    "horizontal_gr_d1",
    "horizontal_gr_d2",
    "known_mask",
    "known_tvt_offset",
    "whole_progress",
    "post_anchor_progress",
    "z_relative",
    "dx",
    "dy",
    "dz",
    "d2z",
]
EMISSION_NAMES = [
    "mean",
    "std",
    "entropy",
    "peak",
    "margin",
    "map",
]
VARIABLE_NAMES = BASE_VARIABLE_NAMES + [
    f"{reference}_{name}"
    for reference in ("typewell", "heel")
    for name in EMISSION_NAMES
]
N_VARIABLES = len(VARIABLE_NAMES)
GR_VARIABLES = np.asarray(
    [
        index
        for index, name in enumerate(VARIABLE_NAMES)
        if "gr" in name or name.startswith("typewell_") or name.startswith("heel_")
    ],
    np.int64,
)


def _interp_channels(md, channels, grid_md):
    output = np.empty((channels.shape[0], len(grid_md)), np.float32)
    for channel in range(channels.shape[0]):
        output[channel] = np.interp(grid_md, md, channels[channel]).astype(np.float32)
    return output


def _emission_summary(query, reference):
    # The native reference axis contains 64 ft context on both sides.  The
    # central 257 states correspond exactly to [-128, 128] ft.
    start = int((reference.shape[1] - len(STATE_OFFSETS)) // 2)
    reference = reference[:, start : start + len(STATE_OFFSETS)].astype(np.float32)
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
    probability = np.exp(logits)
    probability *= valid
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-8)
    states = STATE_OFFSETS[None, :]
    mean = np.sum(probability * states, axis=1)
    variance = np.sum(probability * (states - mean[:, None]) ** 2, axis=1)
    entropy = -np.sum(probability * np.log(probability + 1e-9), axis=1) / math.log(len(STATE_OFFSETS))
    ordered = np.partition(probability, -2, axis=1)[:, -2:]
    peak = ordered[:, 1]
    margin = ordered[:, 1] - ordered[:, 0]
    map_offset = STATE_OFFSETS[np.argmax(probability, axis=1)]
    invalid = ~query_valid
    output = np.stack(
        [
            mean / STATE_SCALE_FT,
            np.sqrt(np.maximum(variance, 0.0)) / STATE_SCALE_FT,
            entropy,
            peak,
            margin,
            map_offset / STATE_SCALE_FT,
        ],
        axis=0,
    ).astype(np.float32)
    output[:, invalid] = 0.0
    return output


def _pack_path(well):
    return PACK_DIR / f"{well}.npz"


def _build_pack(record, typewell_override=None):
    grid_md = np.linspace(record.md[0], record.md[-1], GRID_POINTS, dtype=np.float32)
    horizontal = _interp_channels(record.md, record.horizontal, grid_md)
    condition = _interp_channels(record.md, record.condition, grid_md)
    base = np.concatenate([horizontal[:4], condition[:9]], axis=0)
    typewell = record.typewell if typewell_override is None else typewell_override
    tw_summary = _emission_summary(horizontal, typewell)
    heel_summary = _emission_summary(horizontal, record.heel)
    values = np.concatenate([base, tw_summary, heel_summary], axis=0).astype(np.float32)
    if values.shape != (N_VARIABLES, GRID_POINTS):
        raise RuntimeError(f"{record.well}: bad pack shape {values.shape}")
    target = np.interp(
        grid_md,
        record.md,
        record.truth.astype(np.float64) - record.anchor_tvt,
    ).astype(np.float32)
    anchor_grid = int(np.argmin(np.abs(grid_md - record.md[record.anchor_index])))
    supervised = grid_md > record.md[record.anchor_index]
    target -= target[anchor_grid]
    eval_md = record.md[record.eval_index].astype(np.float32)
    return WellPack(
        well=record.well,
        values=values,
        target=target,
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
        values=pack.values.astype(np.float16),
        target=pack.target.astype(np.float32),
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
            raise RuntimeError("stale pack cache")
        return WellPack(
            well=str(data["well"].item()),
            values=data["values"].astype(np.float32),
            target=data["target"].astype(np.float32),
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
    report = []
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
            report.append({"well": well, "status": "ok", "source": source, "rows": pack.eval_rows})
        except Exception as exc:
            report.append({"well": well, "status": "failed", "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"iTransformer packs {number}/{len(wells)}", flush=True)
    frame = pd.DataFrame(report)
    frame.to_csv(OUT_DIR / "pack_report.csv", index=False)
    failures = frame[frame["status"] != "ok"]
    if len(failures):
        raise RuntimeError(f"{len(failures)} pack failures: {failures.head().to_dict('records')}")
    return frame


# ---------------------------------------------------------------------------
# Matched architectures
# ---------------------------------------------------------------------------


def _encoder_layer():
    return nn.TransformerEncoderLayer(
        d_model=MODEL_DIM,
        nhead=HEADS,
        dim_feedforward=FF_DIM,
        dropout=DROPOUT,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class InvertedFullWellTransformer(nn.Module):
    """Variables are tokens; each token embeds one complete well trajectory."""

    def __init__(self):
        super().__init__()
        self.trajectory_embedding = nn.Sequential(
            nn.Linear(GRID_POINTS, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.GELU(),
        )
        self.variable_embedding = nn.Parameter(torch.zeros(1, N_VARIABLES + 1, MODEL_DIM))
        self.target_token = nn.Parameter(torch.zeros(1, 1, MODEL_DIM))
        self.encoder = nn.TransformerEncoder(_encoder_layer(), num_layers=LAYERS)
        self.output = nn.Sequential(
            nn.LayerNorm(MODEL_DIM),
            nn.Linear(MODEL_DIM, GRID_POINTS),
        )
        nn.init.normal_(self.variable_embedding, std=0.02)
        nn.init.normal_(self.target_token, std=0.02)

    def forward(self, values, anchor_grid):
        tokens = self.trajectory_embedding(values)
        target = self.target_token.expand(values.shape[0], -1, -1)
        tokens = torch.cat([target, tokens], dim=1) + self.variable_embedding
        encoded = self.encoder(tokens)
        path = self.output(encoded[:, 0])
        anchor = path.gather(1, anchor_grid[:, None])
        return path - anchor


class TemporalFullWellTransformer(nn.Module):
    """Matched baseline: positions are tokens and variables are channels."""

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(N_VARIABLES, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, GRID_POINTS, MODEL_DIM))
        self.encoder = nn.TransformerEncoder(_encoder_layer(), num_layers=LAYERS)
        self.output = nn.Sequential(nn.LayerNorm(MODEL_DIM), nn.Linear(MODEL_DIM, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values, anchor_grid):
        tokens = self.input(values.transpose(1, 2)) + self.position
        path = self.output(self.encoder(tokens)).squeeze(-1)
        anchor = path.gather(1, anchor_grid[:, None])
        return path - anchor


def _make_model(architecture):
    if architecture == "inverted":
        return InvertedFullWellTransformer()
    if architecture == "temporal":
        return TemporalFullWellTransformer()
    raise ValueError(architecture)


def _tensor_pack(pack, device, mode="full", donor=None):
    values = pack.values.copy()
    if mode == "no_gr":
        values[GR_VARIABLES] = 0.0
    elif mode == "shuffled_typewell":
        if donor is None:
            raise ValueError("shuffled_typewell requires donor")
        record = _load_record(pack.well)
        donor_record = _load_record(donor.well)
        values = _build_pack(record, typewell_override=donor_record.typewell).values
    elif mode != "full":
        raise ValueError(mode)
    return (
        torch.from_numpy(values)[None].to(device),
        torch.tensor([pack.anchor_grid], dtype=torch.long, device=device),
    )


def _loss(model, pack, device, row_weight=1.0):
    values, anchor = _tensor_pack(pack, device)
    prediction = model(values, anchor)[0]
    target = torch.from_numpy(pack.target).to(device)
    mask = torch.from_numpy(pack.supervised).to(device)
    error = prediction[mask] - target[mask]
    path = torch.mean(error.square()) / 64.0
    slope = torch.mean((prediction[1:] - prediction[:-1] - (target[1:] - target[:-1])).square()) / 4.0
    curvature = torch.mean(
        (prediction[2:] - 2 * prediction[1:-1] + prediction[:-2]
         - target[2:] + 2 * target[1:-1] - target[:-2]).square()
    ) / 4.0
    total = float(row_weight) * (
        LOSS_PATH * path + LOSS_SLOPE * slope + LOSS_CURVATURE * curvature
    )
    return total, path.detach(), slope.detach()


@torch.no_grad()
def _predict(model, pack, device, mode="full", donor=None):
    model.eval()
    values, anchor = _tensor_pack(pack, device, mode=mode, donor=donor)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        grid_offset = model(values, anchor)[0].float().cpu().numpy()
    # The first hidden row may be one foot after the anchor; interpolate from
    # an explicit zero-offset anchor for correct near-heel behaviour.
    support_md = np.r_[pack.anchor_md, pack.grid_md[pack.grid_md > pack.anchor_md]]
    support_offset = np.r_[0.0, grid_offset[pack.grid_md > pack.anchor_md]]
    prediction = pack.anchor_tvt + np.interp(pack.eval_md, support_md, support_offset)
    return pack.truth.astype(np.float64), prediction.astype(np.float64), pack.eval_distance.astype(np.float64)


def _metrics(truth, prediction, distance, wells):
    frame = pd.DataFrame(
        {"well": wells, "truth": truth, "prediction": prediction, "distance": distance}
    )
    frame["error"] = frame["prediction"] - frame["truth"]
    bywell = frame.groupby("well", sort=False).agg(
        rows=("error", "size"),
        mse=("error", lambda value: float(np.mean(np.square(value)))),
        mae=("error", lambda value: float(np.mean(np.abs(value)))),
    ).reset_index()
    bywell["rmse"] = np.sqrt(bywell["mse"])
    result = {
        "rows": int(len(frame)),
        "wells": int(bywell["well"].nunique()),
        "exact_sse": float(np.sum(frame["error"].to_numpy(np.float64) ** 2)),
        "rmse": float(np.sqrt(np.mean(frame["error"] ** 2))),
        "mae": float(np.mean(np.abs(frame["error"]))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
    }
    strata = (
        ("near_0_500_rmse", 0, 500),
        ("mid_500_1000_rmse", 500, 1000),
        ("mid_1000_2000_rmse", 1000, 2000),
        ("far_gt2000_rmse", 2000, np.inf),
    )
    for name, low, high in strata:
        selected = (frame["distance"] >= low) & (frame["distance"] < high)
        result[name] = (
            float(np.sqrt(np.mean(frame.loc[selected, "error"] ** 2)))
            if selected.any() else float("nan")
        )
    return result, bywell, frame


@torch.no_grad()
def _evaluate(model, packs, device, mode="full", save_rows=False):
    truths, predictions, distances, well_labels = [], [], [], []
    for index, pack in enumerate(packs):
        donor = None
        if mode == "shuffled_typewell":
            donor = packs[(index + max(1, len(packs) // 2)) % len(packs)]
        truth, prediction, distance = _predict(model, pack, device, mode=mode, donor=donor)
        truths.append(truth)
        predictions.append(prediction)
        distances.append(distance)
        well_labels.append(np.full(len(truth), pack.well, dtype="U16"))
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    distance = np.concatenate(distances)
    wells = np.concatenate(well_labels)
    metrics, bywell, frame = _metrics(truth, prediction, distance, wells)
    payload = None
    if save_rows:
        payload = {
            "well": wells,
            "truth": truth.astype(np.float32),
            "prediction": prediction.astype(np.float32),
            "distance": distance.astype(np.float32),
        }
    return metrics, bywell, frame, payload


def _fit_model(architecture, train_packs, valid_packs, device, epochs, phase):
    _seed_everything(SEED + (0 if architecture == "inverted" else 1000))
    model = _make_model(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=512.0)
    mean_rows = float(np.mean([pack.eval_rows for pack in train_packs]))
    rng = np.random.default_rng(SEED + len(train_packs))
    history, best = [], None
    stale = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_packs))
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_path = total_slope = 0.0
        successful = 0
        for step, index in enumerate(order, 1):
            pack = train_packs[int(index)]
            weight = pack.eval_rows / max(mean_rows, 1.0)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, path, slope = _loss(model, pack, device, weight)
                scaled = loss / GRAD_ACCUM_WELLS
            scaler.scale(scaled).backward()
            if step % GRAD_ACCUM_WELLS == 0 or step == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                successful += 1
            total_loss += float(loss.detach().cpu())
            total_path += float(path.cpu())
            total_slope += float(slope.cpu())
        scheduler.step()

        report = (
            epoch <= 2
            or epoch % (CAPACITY_REPORT_EVERY if phase == "capacity" else FOLD_REPORT_EVERY) == 0
            or epoch == epochs
        )
        if not report:
            continue
        valid_metrics, _, _, _ = _evaluate(model, valid_packs, device)
        row = {
            "architecture": architecture,
            "phase": phase,
            "epoch": epoch,
            "train_loss": total_loss / len(train_packs),
            "train_path": total_path / len(train_packs),
            "train_slope": total_slope / len(train_packs),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_steps": successful,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("ITRANSFORMER EPOCH", row, flush=True)
        if best is None or valid_metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
            best = {
                "epoch": epoch,
                "metrics": valid_metrics,
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            stale = 0
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "architecture": architecture,
                    "phase": phase,
                    "epoch": epoch,
                    "metrics": valid_metrics,
                    "variable_names": VARIABLE_NAMES,
                    "model": best["state"],
                },
                MODEL_DIR / f"best_{phase}_{architecture}.pt",
            )
        else:
            stale += 1
        pd.DataFrame(history).to_csv(
            OUT_DIR / f"history_{phase}_{architecture}.csv", index=False
        )
        if phase != "capacity" and epoch >= MIN_EPOCHS and stale >= EARLY_STOPPING_REPORTS:
            print(f"{architecture} early stopping at {epoch}", flush=True)
            break
        if phase == "capacity" and valid_metrics["rmse"] <= CAPACITY_GATE_RMSE:
            print(f"capacity gate reached at epoch {epoch}", flush=True)
            break
    if best is None:
        raise RuntimeError(f"No checkpoint for {architecture}/{phase}")
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history)


def _error_correlation(frame_a, frame_b):
    keys = ["well", "truth", "distance"]
    left = frame_a[keys + ["error"]].rename(columns={"error": "error_a"})
    right = frame_b[["well", "distance", "error"]].rename(columns={"error": "error_b"})
    merged = left.merge(right, on=["well", "distance"], how="inner", validate="one_to_one")
    raw = float(np.corrcoef(merged["error_a"], merged["error_b"])[0, 1])
    # Remove each model's own smooth distance-error trend before correlation.
    bins = np.minimum((merged["distance"] // 250).astype(int), 16)
    for column in ("error_a", "error_b"):
        trend = merged.groupby(bins)[column].transform("mean")
        merged[f"{column}_detrended"] = merged[column] - trend
    detrended = float(
        np.corrcoef(merged["error_a_detrended"], merged["error_b_detrended"])[0, 1]
    )
    strata = {}
    for name, low, high in (
        ("0_500", 0, 500),
        ("500_1000", 500, 1000),
        ("1000_2000", 1000, 2000),
        ("gt2000", 2000, np.inf),
    ):
        selected = (merged["distance"] >= low) & (merged["distance"] < high)
        strata[name] = (
            float(np.corrcoef(merged.loc[selected, "error_a"], merged.loc[selected, "error_b"])[0, 1])
            if int(selected.sum()) >= 100 else float("nan")
        )
    return {"raw_signed_error_rho": raw, "distance_detrended_rho": detrended, "strata": strata}


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


STARTED = time.time()
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a CUDA GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_everything(SEED)

folds, fold_path = _load_folds()
wells = folds["well"].tolist()
if len(wells) != 773:
    raise RuntimeError(f"Expected 773 wells, found {len(wells)}")
record_report = _ensure_records(wells)
pack_report = _ensure_packs(wells)

packs = {well: _load_pack(well) for well in wells}
train_wells = folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].tolist()
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(train_wells) not in (463, 464) or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected Fold345/Fold2 sizes: {len(train_wells)}/{len(valid_wells)}")
train_packs = [packs[well] for well in train_wells]
valid_packs = [packs[well] for well in valid_wells]

capacity_wells = sorted(train_wells, key=lambda well: (-packs[well].eval_rows, well))[:CAPACITY_WELLS]
capacity_packs = [packs[well] for well in capacity_wells]
capacity_model, capacity_best, capacity_history = _fit_model(
    "inverted", capacity_packs, capacity_packs, device, CAPACITY_EPOCHS, "capacity"
)
capacity_pass = capacity_best["metrics"]["rmse"] <= CAPACITY_GATE_RMSE
capacity_verdict = {
    "wells": len(capacity_packs),
    "parameters": _parameter_count(capacity_model),
    "best_epoch": capacity_best["epoch"],
    "metrics": capacity_best["metrics"],
    "gate_rmse_le": CAPACITY_GATE_RMSE,
    "pass": capacity_pass,
}
print("ITRANSFORMER CAPACITY", json.dumps(capacity_verdict, indent=2), flush=True)
if not capacity_pass:
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "24_well_capacity",
        "capacity": capacity_verdict,
        "strict_pass": False,
        "next": "FAIL -> inverted representation cannot fit 24 wells; stop before Fold2",
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    raise RuntimeError(json.dumps(verdict, indent=2))

del capacity_model
gc.collect()
torch.cuda.empty_cache()

models, results, frames = {}, {}, {}
for architecture in ("temporal", "inverted"):
    model, best, history = _fit_model(
        architecture, train_packs, valid_packs, device, FOLD_EPOCHS, "fold2"
    )
    full, bywell, frame, payload = _evaluate(model, valid_packs, device, save_rows=True)
    controls = {}
    for mode in ("no_gr", "shuffled_typewell"):
        metric, _, _, _ = _evaluate(model, valid_packs, device, mode=mode)
        metric["degradation_vs_full"] = metric["rmse"] - full["rmse"]
        controls[mode] = metric
    bywell.to_csv(OUT_DIR / f"fold2_{architecture}_bywell.csv", index=False)
    np.savez_compressed(OUT_DIR / f"fold2_{architecture}_oof.npz", **payload)
    models[architecture] = model
    results[architecture] = {
        "parameters": _parameter_count(model),
        "best_epoch": best["epoch"],
        "full": full,
        "controls": controls,
    }
    frames[architecture] = frame
    print(f"{architecture.upper()} RESULT", json.dumps(results[architecture], indent=2), flush=True)

correlation = _error_correlation(frames["temporal"], frames["inverted"])
inverted = results["inverted"]
capacity_and_path = capacity_pass and inverted["full"]["rmse"] <= ARCHITECTURE_GATE_RMSE
causal_pass = all(
    inverted["controls"][mode]["degradation_vs_full"] >= CONTROL_DEGRADATION_GATE
    for mode in ("no_gr", "shuffled_typewell")
)
strict_pass = capacity_and_path and causal_pass

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_matched_temporal_vs_inverted_transformer_gateway",
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
        "grid_points": GRID_POINTS,
        "variables": VARIABLE_NAMES,
        "inverted_token_definition": "one complete evidence trajectory per token",
        "temporal_token_definition": "one resampled MD position per token",
    },
    "capacity": capacity_verdict,
    "temporal": results["temporal"],
    "inverted": results["inverted"],
    "matched_error_correlation": correlation,
    "gates": {
        "capacity_rmse_le": CAPACITY_GATE_RMSE,
        "inverted_architecture_rmse_le": ARCHITECTURE_GATE_RMSE,
        "target_rmse_le": TARGET_GATE_RMSE,
        "each_control_degradation_ge": CONTROL_DEGRADATION_GATE,
    },
    "capacity_pass": capacity_pass,
    "architecture_pass": capacity_and_path,
    "target_pass": inverted["full"]["rmse"] <= TARGET_GATE_RMSE,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> run untouched Fold1 with the frozen winning architecture"
        if strict_pass
        else "FAIL -> compare temporal/inverted residuals; do not scale this representation"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
