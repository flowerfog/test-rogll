"""V150: prefix-state residual refiner on the deployed 6.377 incumbent.

This is a new direction. It does not rank candidates, rerank Neighbor, or
select between endpoint families. It learns a small tail-basis correction on
top of the deployed incumbent using only legal raw well geometry, GR/typewell,
and deployed baseline trajectories. A geology-state histogram acts as an
auxiliary supervision signal so the encoder has to explain the well's regime
before it predicts the correction coefficients.
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
EXPERIMENT = "prefix_state_residual_refiner_v150"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "well_features"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

STATE_SOURCE = ROOT / "trajectory_geology_state_gateway_v0_cell.py"
BASE_PATH_CANDIDATES = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
    ROOT / "full_oof_cp_endpoints.parquet",
)
FOLD_PATH_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "gr_path_bank_oracle_v0" / "well_metadata.csv",
)

BASE_COLUMN = "incumbent_6p377_replay"
AUX_COLUMN = "full_cp8_ordered"
TRUTH = "tvt_true"
FOLDS = (1, 2, 3, 4, 5)
SEED = 20260802
NODES = 24
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 8.0
COEFF_DIM = 5
STATE_DIM = 3
EPOCHS = 140
PATIENCE = 22
BATCH_SIZE = 32
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-3


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


def _first_existing(paths):
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError([str(path) for path in paths])


def _smoothstep(values):
    values = np.clip(np.asarray(values, np.float64), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _basis(distance):
    distance = np.asarray(distance, np.float64)
    taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
    x = (distance - 1500.0) / 1000.0
    hinge1 = np.clip((distance - 1250.0) / 1000.0, 0.0, None)
    hinge2 = np.clip((distance - 1750.0) / 1000.0, 0.0, None)
    hinge3 = np.clip((distance - 2250.0) / 1000.0, 0.0, None)
    return np.column_stack([taper, taper * x, taper * hinge1, taper * hinge2, taper * hinge3])


def _oracle_coefficients(distance, residual):
    design = _basis(distance)
    regulariser = 1e-3 * np.eye(design.shape[1], dtype=np.float64)
    return np.linalg.solve(design.T @ design + regulariser, design.T @ residual)


def _state_histogram(state):
    state = np.asarray(state, np.int64)
    counts = np.bincount(state, minlength=STATE_DIM).astype(np.float64)
    total = float(max(counts.sum(), 1.0))
    return counts / total


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    if len(prediction) != len(frame):
        raise RuntimeError(f"{endpoint}: prediction length mismatch")
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{endpoint}: non-finite prediction")
    error = prediction - truth
    well_error = pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
    well_rmse = np.sqrt(well_error.groupby("well", sort=False)["e2"].mean().to_numpy(np.float64))
    distance = frame["md_since"].to_numpy(np.float64)
    output = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    for name, mask in {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


def _bootstrap_state_defs():
    if not STATE_SOURCE.is_file():
        raise FileNotFoundError(STATE_SOURCE)
    source = STATE_SOURCE.read_text(encoding="utf-8")
    marker = "fold_path = _first_existing(FOLD_CANDIDATES)"
    if marker not in source:
        raise RuntimeError(f"bootstrap marker missing from {STATE_SOURCE}")
    module = types.ModuleType("v150_state_gateway_defs")
    module.__file__ = str(STATE_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(STATE_SOURCE), "exec"), module.__dict__)
    return module


STATE = _bootstrap_state_defs()
STATE.OUT_DIR = OUT_DIR
STATE.CACHE_DIR = CACHE_DIR
STATE.MODEL_DIR = MODEL_DIR
STATE.DATA_ROOT = DATA_ROOT

BASE_PATH = _first_existing(BASE_PATH_CANDIDATES)
FOLD_PATH = _first_existing(FOLD_PATH_CANDIDATES)

for required in (BASE_PATH, FOLD_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(BASE_PATH)
if BASE_COLUMN not in frame.columns and AUX_COLUMN not in frame.columns:
    raise RuntimeError(f"Neither {BASE_COLUMN} nor {AUX_COLUMN} exists in {BASE_PATH}")
base_runtime_column = BASE_COLUMN if BASE_COLUMN in frame.columns else AUX_COLUMN
aux_runtime_column = AUX_COLUMN if AUX_COLUMN in frame.columns and AUX_COLUMN != base_runtime_column else None
if aux_runtime_column is None:
    frame[AUX_COLUMN] = frame[base_runtime_column].to_numpy(np.float64)
    aux_runtime_column = AUX_COLUMN
selected_columns = [
    "id",
    "well",
    "row_idx",
    "fold",
    TRUTH,
    "md_since",
    base_runtime_column,
]
if aux_runtime_column != base_runtime_column:
    selected_columns.append(aux_runtime_column)
frame = frame[selected_columns].copy()
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Base OOF contains duplicate ids")
if frame[["well", "row_idx", TRUTH, "md_since", BASE_COLUMN, AUX_COLUMN]].isna().any().any():
    raise RuntimeError("Base OOF contains missing required values")

fold_map = pd.read_csv(FOLD_PATH, dtype={"well": str})
well_column = next((column for column in ("well", "well_id", "Well") if column in fold_map), None)
fold_column = next((column for column in ("fold", "Fold", "outer_fold") if column in fold_map), None)
if well_column is None or fold_column is None:
    raise RuntimeError(f"Cannot identify fold columns in {FOLD_PATH}")
fold_map = fold_map[[well_column, fold_column]].drop_duplicates()
fold_map[well_column] = fold_map[well_column].astype(str)
if fold_map[well_column].duplicated().any():
    raise RuntimeError("Duplicate wells in fold metadata")
fold_map = dict(zip(fold_map[well_column], fold_map[fold_column].astype(int)))

well_order = sorted(frame["well"].unique())
if len(well_order) != 773:
    raise RuntimeError(f"Unexpected well coverage: {len(well_order)}")
fold_by_well = {well: int(fold_map[well]) for well in well_order}
if set(fold_by_well.values()) != set(FOLDS):
    raise RuntimeError(f"Unexpected fold values: {sorted(set(fold_by_well.values()))}")


def _augment_well(well):
    path = CACHE_DIR / f"{well}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if {"target_coeffs", "metadata", "state_hist", "baseline_eval", "eval_distance"}.issubset(data.files):
                return {"well": well, "status": "cached", "nodes": int(len(data["node_md"]))}
    build = STATE._build_well(well)
    if build["status"] != "built":
        return build
    with np.load(path, allow_pickle=False) as data:
        cache = {key: data[key] for key in data.files}
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").reset_index(drop=True)
    eval_rows = cache["eval_rows"].astype(np.int64)
    if len(local) != len(eval_rows) or not np.array_equal(local["row_idx"].to_numpy(np.int64), eval_rows):
        raise RuntimeError(f"{well}: row alignment mismatch")
    eval_distance = local["md_since"].to_numpy(np.float64)
    eval_truth = local[TRUTH].to_numpy(np.float64)
    baseline_eval = local[base_runtime_column].to_numpy(np.float64)
    aux_eval = local[aux_runtime_column].to_numpy(np.float64)
    residual = eval_truth - baseline_eval
    coeff_target = _oracle_coefficients(eval_distance, residual).astype(np.float32)
    state_hist = _state_histogram(cache["target_state"]).astype(np.float32)
    node_md = cache["node_md"].astype(np.float64)
    node_distance = node_md - float(node_md[0])
    node_baseline = np.interp(node_distance, eval_distance, baseline_eval)
    node_aux = np.interp(node_distance, eval_distance, aux_eval)
    node_gap = node_aux - node_baseline
    node_gradient = np.gradient(node_baseline, node_distance, edge_order=1) if len(node_distance) > 2 else np.zeros_like(node_baseline)
    node_curvature = np.gradient(node_gradient, node_distance, edge_order=1) if len(node_distance) > 2 else np.zeros_like(node_baseline)
    extra_features = np.column_stack(
        [
            node_baseline,
            node_baseline - node_baseline[0],
            node_gradient,
            node_curvature,
            node_aux,
            node_gap,
        ]
    ).astype(np.float32)
    features = np.column_stack([cache["features"].astype(np.float32), extra_features]).astype(np.float32)
    feature_names = np.concatenate(
        [
            cache["feature_names"].astype(str),
            np.asarray(
                [
                    "incumbent_6p377_replay",
                    "incumbent_6p377_delta",
                    "incumbent_6p377_grad",
                    "incumbent_6p377_curvature",
                    "full_cp8_ordered",
                    "cp8_minus_incumbent",
                ],
                dtype=str,
            ),
        ]
    )
    node_count = len(node_md)
    step_ft = float(np.median(np.diff(node_distance))) if node_count > 1 else 1.0
    tail_start = max(3, int(round(0.75 * len(eval_distance))))
    tail_slice = slice(tail_start, len(eval_distance))
    tail_x = eval_distance[tail_slice]
    tail_y = baseline_eval[tail_slice]
    tail_slope = float(np.polyfit(tail_x, tail_y, 1)[0]) if len(tail_x) >= 2 else 0.0
    metadata = np.asarray(
        [
            float(node_count) / 24.0,
            float(node_distance[-1]) / 5000.0,
            step_ft / 100.0,
            float(cache["anchor_tvt"][0]) / 10000.0,
            float(cache["anchor_s"][0]) / 10000.0,
            float(node_baseline[0]) / 10000.0,
            float(node_baseline[-1]) / 10000.0,
            float((node_baseline[-1] - node_baseline[0])) / 1000.0,
            float(np.mean(node_baseline)) / 10000.0,
            float(np.std(node_baseline)) / 1000.0,
            tail_slope / 10.0,
            float(np.mean(node_gap)) / 1000.0,
            float(np.std(node_gap)) / 1000.0,
        ],
        np.float32,
    )
    # Save oracle audit metrics for the new residual family.
    basis = _basis(eval_distance)
    oracle_prediction = baseline_eval + np.clip(basis @ coeff_target.astype(np.float64), -CAP_FT, CAP_FT)
    base_rmse = float(np.sqrt(np.mean((baseline_eval - eval_truth) ** 2)))
    oracle_rmse = float(np.sqrt(np.mean((oracle_prediction - eval_truth) ** 2)))
    payload = dict(cache)
    payload.update(
        {
            "features": features,
            "feature_names": feature_names,
            "metadata": metadata,
            "node_distance": node_distance.astype(np.float32),
            "eval_distance": eval_distance.astype(np.float32),
            "eval_base": baseline_eval.astype(np.float32),
            "eval_aux": aux_eval.astype(np.float32),
            "target_coeffs": coeff_target,
            "state_hist": state_hist,
            "oracle_rmse": np.asarray([oracle_rmse], np.float32),
            "base_rmse": np.asarray([base_rmse], np.float32),
            "oracle_gain": np.asarray([base_rmse - oracle_rmse], np.float32),
        }
    )
    np.savez_compressed(path, **payload)
    return {
        "well": well,
        "status": "built",
        "nodes": node_count,
        "features": int(features.shape[1]),
        "oracle_gain": base_rmse - oracle_rmse,
    }


def _load_well(well):
    with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
        return {
            "sequence": data["features"].astype(np.float32).T,
            "metadata": data["metadata"].astype(np.float32),
            "coeff": data["target_coeffs"].astype(np.float32),
            "state_hist": data["state_hist"].astype(np.float32),
            "eval_distance": data["eval_distance"].astype(np.float32),
            "eval_base": data["eval_base"].astype(np.float32),
            "eval_truth": data["eval_truth"].astype(np.float32),
            "eval_rows": data["eval_rows"].astype(np.int32),
            "oracle_gain": float(data["oracle_gain"][0]),
            "base_rmse": float(data["base_rmse"][0]),
            "oracle_rmse": float(data["oracle_rmse"][0]),
        }


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, width),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class ResidualStateRefiner(nn.Module):
    def __init__(self, channels, metadata_features, coeff_dim=COEFF_DIM, state_dim=STATE_DIM, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8)))
        self.attention = nn.Conv1d(width, 1, 1)
        self.metadata = nn.Sequential(
            nn.Linear(metadata_features, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.trunk = nn.Sequential(
            nn.Linear(width * 3 + 32, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.coeff_head = nn.Linear(128, coeff_dim)
        self.state_head = nn.Linear(128, state_dim)

    def forward(self, sequence, metadata):
        encoded = self.blocks(self.stem(sequence))
        attention = torch.softmax(self.attention(encoded), dim=-1)
        pooled = torch.sum(attention * encoded, dim=-1)
        mean = torch.mean(encoded, dim=-1)
        maximum = torch.amax(encoded, dim=-1)
        meta = self.metadata(metadata)
        trunk = self.trunk(torch.cat([pooled, mean, maximum, meta], dim=1))
        return self.coeff_head(trunk), self.state_head(trunk)


def _predict(model, sequence, metadata, device):
    model.eval()
    with torch.no_grad():
        coeff, state = model(
            torch.from_numpy(sequence).to(device),
            torch.from_numpy(metadata).to(device),
        )
    return coeff.cpu().numpy(), state.cpu().numpy()


def _reconstruct_frame(wells, coefficient_map):
    frames = []
    for well in wells:
        with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
            coeff = np.asarray(coefficient_map[well], np.float64)
            base = data["eval_base"].astype(np.float64)
            truth = data["eval_truth"].astype(np.float64)
            distance = data["eval_distance"].astype(np.float64)
            correction = np.clip(_basis(distance) @ coeff, -CAP_FT, CAP_FT)
            prediction = base + correction
            frames.append(
                pd.DataFrame(
                    {
                        "well": well,
                        "row_idx": data["eval_rows"].astype(np.int32),
                        "truth": truth,
                        "baseline": base,
                        "prediction": prediction,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _stack_wells(wells):
    sequence_parts = []
    metadata_parts = []
    coeff_parts = []
    state_parts = []
    for number, well in enumerate(wells, 1):
        payload = _load_well(well)
        sequence_parts.append(payload["sequence"])
        metadata_parts.append(payload["metadata"])
        coeff_parts.append(payload["coeff"])
        state_parts.append(payload["state_hist"])
        if number % 100 == 0 or number == len(wells):
            print(f"V150 stack {number}/{len(wells)}", flush=True)
    return (
        np.stack(sequence_parts).astype(np.float32),
        np.stack(metadata_parts).astype(np.float32),
        np.stack(coeff_parts).astype(np.float32),
        np.stack(state_parts).astype(np.float32),
    )


def _train_fold(heldout_fold, wells, fold_by_well, sequence_raw, metadata_raw, coeff_raw, state_raw):
    heldout_mask = np.asarray([fold_by_well[well] == heldout_fold for well in wells], dtype=bool)
    train_mask = ~heldout_mask
    train_index = np.flatnonzero(train_mask)
    valid_index = np.flatnonzero(heldout_mask)
    sequence_mean = np.mean(sequence_raw[train_index], axis=(0, 2), keepdims=True)
    sequence_std = np.maximum(np.std(sequence_raw[train_index], axis=(0, 2), keepdims=True), 1e-3)
    metadata_mean = np.mean(metadata_raw[train_index], axis=0, keepdims=True)
    metadata_std = np.maximum(np.std(metadata_raw[train_index], axis=0, keepdims=True), 1e-3)
    coeff_mean = np.mean(coeff_raw[train_index], axis=0, keepdims=True)
    coeff_std = np.maximum(np.std(coeff_raw[train_index], axis=0, keepdims=True), 0.10)

    sequence = ((sequence_raw - sequence_mean) / sequence_std).astype(np.float32)
    metadata = ((metadata_raw - metadata_mean) / metadata_std).astype(np.float32)
    coeff_target = np.clip((coeff_raw - coeff_mean) / coeff_std, -6.0, 6.0).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualStateRefiner(sequence.shape[1], metadata.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    coeff_loss_fn = nn.SmoothL1Loss(beta=0.25)
    state_loss_fn = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(sequence[train_index]),
        torch.from_numpy(metadata[train_index]),
        torch.from_numpy(coeff_target[train_index]),
        torch.from_numpy(state_raw[train_index]),
    )
    generator = torch.Generator().manual_seed(SEED + int(heldout_fold))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_sequence, batch_metadata, batch_coeff, batch_state in loader:
            batch_sequence = batch_sequence.to(device)
            batch_metadata = batch_metadata.to(device)
            batch_coeff = batch_coeff.to(device)
            batch_state = batch_state.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_coeff, pred_state = model(batch_sequence, batch_metadata)
            coeff_loss = coeff_loss_fn(pred_coeff, batch_coeff)
            state_prob = torch.softmax(pred_state, dim=1)
            state_loss = state_loss_fn(state_prob, batch_state)
            loss = coeff_loss + 0.20 * state_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        fold_prediction = _predict(model, sequence[valid_index], metadata[valid_index], device)[0]
        fold_prediction = fold_prediction * coeff_std + coeff_mean
        fold_map = dict(zip([wells[index] for index in valid_index], fold_prediction))
        valid_frame = _reconstruct_frame([wells[index] for index in valid_index], fold_map)
        valid_rmse = _metric(valid_frame, valid_frame["prediction"], f"fold{heldout_fold}_refined")["rmse"]
        history.append(
            {
                "heldout_fold": int(heldout_fold),
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)),
                "valid_rmse": float(valid_rmse),
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            print("V150 EPOCH", history[-1], flush=True)
        if valid_rmse < best_rmse - 1e-5:
            best_rmse = float(valid_rmse)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"Fold{heldout_fold}: no checkpoint selected")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "sequence_mean": sequence_mean,
            "sequence_std": sequence_std,
            "metadata_mean": metadata_mean,
            "metadata_std": metadata_std,
            "coeff_mean": coeff_mean,
            "coeff_std": coeff_std,
            "best_epoch": best_epoch,
            "heldout_fold": heldout_fold,
            "coeff_dim": COEFF_DIM,
            "state_dim": STATE_DIM,
        },
        MODEL_DIR / f"heldout_fold{heldout_fold}.pt",
    )

    coeff_pred, _ = _predict(model, sequence[valid_index], metadata[valid_index], device)
    coeff_pred = coeff_pred * coeff_std + coeff_mean
    coeff_map = {wells[index]: coeff_pred[number] for number, index in enumerate(valid_index)}
    refined_frame = _reconstruct_frame([wells[index] for index in valid_index], coeff_map)
    oracle_coeff_map = {wells[index]: coeff_raw[index] for index in valid_index}
    oracle_frame = _reconstruct_frame([wells[index] for index in valid_index], oracle_coeff_map)
    shuffled_index = np.random.default_rng(SEED + int(heldout_fold)).permutation(len(valid_index))
    shuffled_map = {
        wells[index]: coeff_pred[position]
        for position, index in enumerate(valid_index[shuffled_index])
    }
    shuffled_frame = _reconstruct_frame([wells[index] for index in valid_index], shuffled_map)
    baseline_frame = _reconstruct_frame(
        [wells[index] for index in valid_index],
        {wells[index]: np.zeros(COEFF_DIM, np.float64) for index in valid_index},
    )
    return {
        "history": history,
        "model": model,
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "sequence_mean": sequence_mean,
        "sequence_std": sequence_std,
        "metadata_mean": metadata_mean,
        "metadata_std": metadata_std,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "valid_wells": [wells[index] for index in valid_index],
        "baseline_frame": baseline_frame,
        "refined_frame": refined_frame,
        "oracle_frame": oracle_frame,
        "shuffled_frame": shuffled_frame,
    }


train_report = []
augment_report = []
for number, well in enumerate(well_order, 1):
    augment_report.append(_augment_well(well))
    if number % 50 == 0 or number == len(well_order):
        print(f"V150 augment {number}/{len(well_order)}", flush=True)
augment_df = pd.DataFrame(augment_report)
failed = augment_df[augment_df["status"].eq("failed")]
if len(failed):
    raise RuntimeError(f"Failed to build {len(failed)} wells: {failed.head().to_dict('records')}")

sequence_raw, metadata_raw, coeff_raw, state_raw = _stack_wells(well_order)
sequence_channels = int(sequence_raw.shape[1])
metadata_features = int(metadata_raw.shape[1])

fold_results = []
history_rows = []
refined_parts = []
oracle_parts = []
shuffled_parts = []
baseline_parts = []
oracle_gain_by_fold = {}
learned_gain_by_fold = {}
shuffled_gain_by_fold = {}

for heldout_fold in FOLDS:
    result = _train_fold(
        heldout_fold,
        well_order,
        fold_by_well,
        sequence_raw,
        metadata_raw,
        coeff_raw,
        state_raw,
    )
    history_rows.extend(result["history"])
    baseline_parts.append(result["baseline_frame"])
    refined_parts.append(result["refined_frame"])
    oracle_parts.append(result["oracle_frame"])
    shuffled_parts.append(result["shuffled_frame"])
    base_metric = _metric(result["baseline_frame"], result["baseline_frame"]["prediction"], f"fold{heldout_fold}_baseline")
    refined_metric = _metric(result["refined_frame"], result["refined_frame"]["prediction"], f"fold{heldout_fold}_refined")
    oracle_metric = _metric(result["oracle_frame"], result["oracle_frame"]["prediction"], f"fold{heldout_fold}_oracle")
    shuffled_metric = _metric(result["shuffled_frame"], result["shuffled_frame"]["prediction"], f"fold{heldout_fold}_shuffled")
    fold_results.append(
        {
            "fold": int(heldout_fold),
            "baseline_rmse": base_metric["rmse"],
            "refined_rmse": refined_metric["rmse"],
            "oracle_rmse": oracle_metric["rmse"],
            "shuffled_rmse": shuffled_metric["rmse"],
            "gain": base_metric["rmse"] - refined_metric["rmse"],
            "oracle_gain": base_metric["rmse"] - oracle_metric["rmse"],
            "shuffled_gain": base_metric["rmse"] - shuffled_metric["rmse"],
            "far_gain": base_metric["far_gt2000_rmse"] - refined_metric["far_gt2000_rmse"],
            "p95_gain": base_metric["p95_well_rmse"] - refined_metric["p95_well_rmse"],
            "best_epoch": result["best_epoch"],
        }
    )
    oracle_gain_by_fold[str(heldout_fold)] = base_metric["rmse"] - oracle_metric["rmse"]
    learned_gain_by_fold[str(heldout_fold)] = base_metric["rmse"] - refined_metric["rmse"]
    shuffled_gain_by_fold[str(heldout_fold)] = base_metric["rmse"] - shuffled_metric["rmse"]

summary = pd.DataFrame(fold_results).sort_values("fold").reset_index(drop=True)
history = pd.DataFrame(history_rows)
baseline_frame = pd.concat(baseline_parts, ignore_index=True)
refined_frame = pd.concat(refined_parts, ignore_index=True)
oracle_frame = pd.concat(oracle_parts, ignore_index=True)
shuffled_frame = pd.concat(shuffled_parts, ignore_index=True)

for name, frame_out in (
    ("baseline", baseline_frame),
    ("refined", refined_frame),
    ("oracle", oracle_frame),
    ("shuffled", shuffled_frame),
):
    frame_out.to_parquet(OUT_DIR / f"{name}_oof.parquet", index=False, compression="zstd")

history.to_csv(OUT_DIR / "training_history.csv", index=False)
summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
augment_df.to_csv(OUT_DIR / "augment_report.csv", index=False)

overall_rows = []
for name, frame_out in (
    ("baseline", baseline_frame),
    ("refined", refined_frame),
    ("oracle", oracle_frame),
    ("shuffled", shuffled_frame),
):
    overall_rows.append(_metric(frame_out, frame_out["prediction"], name))
overall = pd.DataFrame(overall_rows)

base_overall = overall[overall["endpoint"].eq("baseline")].iloc[0]
refined_overall = overall[overall["endpoint"].eq("refined")].iloc[0]
oracle_overall = overall[overall["endpoint"].eq("oracle")].iloc[0]
shuffled_overall = overall[overall["endpoint"].eq("shuffled")].iloc[0]

checks = {
    "oracle_gain_ge_0p20": float(oracle_overall["rmse"]) <= float(base_overall["rmse"]) - 0.20,
    "learned_gain_ge_0p05": float(refined_overall["rmse"]) <= float(base_overall["rmse"]) - 0.05,
    "shuffled_loses_ge_0p02": float(shuffled_overall["rmse"]) >= float(refined_overall["rmse"]) + 0.02,
    "far_gain_nonnegative": float(refined_overall["far_gt2000_rmse"]) <= float(base_overall["far_gt2000_rmse"]),
}

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_prefix_state_residual_refiner",
    "base_column": BASE_COLUMN,
    "aux_column": AUX_COLUMN,
    "folds": {str(fold): int(frame.loc[frame["fold"].eq(fold), "well"].nunique()) for fold in FOLDS},
    "sequence_channels": sequence_channels,
    "metadata_features": metadata_features,
    "coeff_dim": COEFF_DIM,
    "state_dim": STATE_DIM,
    "overall": _clean(overall.to_dict(orient="records")),
    "fold_summary": _clean(summary.to_dict(orient="records")),
    "oracle_gain_by_fold": oracle_gain_by_fold,
    "learned_gain_by_fold": learned_gain_by_fold,
    "shuffled_gain_by_fold": shuffled_gain_by_fold,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> candidate for a wider fold-clean refinement sweep"
        if all(checks.values())
        else "FAIL -> this residual direction is not yet strong enough"
    ),
}

_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV150 AUGMENT REPORT")
print(augment_df.to_string(index=False))
print("\nV150 FOLD SUMMARY")
print(summary.to_string(index=False))
print("\nV150 OVERALL")
print(overall.to_string(index=False))
print("\nV150 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False))
print("saved:", OUT_DIR)
