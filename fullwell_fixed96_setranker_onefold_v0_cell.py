"""One-fold gateway for a standalone complete-well fixed-menu SetRanker.

Run this cell in the same AutoDL kernel after
``fullwell_realwarp_energy_onefold_v0_cell.py``. The prior cell's helper
definitions and fold-clean v4 panels are reused, but its trained energy model
is not used by this experiment.

The experiment closes the train/inference menu mismatch:

* every train/validation panel receives exactly 96 deployable candidates;
* candidate generation is deterministic and truth-independent;
* a shared temporal encoder reads full candidate-aligned GR evidence;
* a permutation-equivariant Set Transformer compares candidates jointly;
* the primary loss is exact row-level MSE of the posterior trajectory.

No strong, strong_bias, CP12, Gold, Neighbor, validation truth at inference,
or hidden/LB feedback is consumed. The sample unit is a complete well/pseudo
prediction-start panel.
"""

from pathlib import Path
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Prerequisites and frozen configuration
# ---------------------------------------------------------------------------
REQUIRED_GLOBALS_SET96 = (
    "OOF_ROOT",
    "OUT_DIR",
    "train_manifest",
    "valid_manifest",
    "ReferenceStore",
    "_record_from_panel",
    "_prepare_record",
    "_spline_basis",
    "_initial_coefficients",
    "_candidate_features",
    "_candidate_row_predictions",
    "_candidate_exact_rmse",
    "_smooth_noise",
    "_deduplicate_coefficients",
    "SPLINE_CONTROLS",
    "EVAL_COLUMNS",
    "COEFFICIENT_CAP_FT",
)
missing_set96 = [name for name in REQUIRED_GLOBALS_SET96 if name not in globals()]
if missing_set96:
    raise RuntimeError(
        "Run fullwell_realwarp_energy_onefold_v0_cell.py first in this kernel; "
        "missing globals: " + repr(missing_set96)
    )

BASE_OUT_DIR_SET96 = Path(OOF_ROOT) / "fullwell_realwarp_energy_onefold_v0"
OUT_DIR_SET96 = Path(OOF_ROOT) / "fullwell_fixed96_setranker_onefold_v0"
MODEL_DIR_SET96 = OUT_DIR_SET96 / "models"
MENU_DIR_SET96 = OUT_DIR_SET96 / "fixed96_menus"
for directory in (OUT_DIR_SET96, MODEL_DIR_SET96, MENU_DIR_SET96):
    directory.mkdir(parents=True, exist_ok=True)

VALID_FOLD_SET96 = 1
SEED_SET96 = 20260727
MENU_SIZE_SET96 = 96
HEURISTIC_KEEP_SET96 = 48
MENU_RANDOM_COUNT_SET96 = 48
MENU_VERSION_SET96 = 1

EPOCHS_SET96 = 12
PATIENCE_SET96 = 3
BATCH_SIZE_SET96 = 1
LEARNING_RATE_SET96 = 2.0e-4
MIN_LEARNING_RATE_SET96 = 2.0e-5
WEIGHT_DECAY_SET96 = 2.0e-4
TARGET_TEMPERATURE_FT_SET96 = 1.50

LOSS_POSTERIOR_SET96 = 1.00
LOSS_LISTWISE_SET96 = 0.25
LOSS_EXPECTED_SET96 = 0.05
LOSS_PAIRWISE_SET96 = 0.10

ORACLE_GATE_SET96 = 5.20
ARCHITECTURE_GATE_SET96 = 6.00
TARGET_GATE_SET96 = 5.50
MEDIAN_GATE_SET96 = 4.50
P95_GATE_SET96 = 12.00
CAUSAL_GATE_SET96 = 0.30


def _seed_set96(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rmse_set96(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _tail_metrics_set96(bywell):
    if bywell.empty:
        return {}
    frame = bywell.copy().sort_values("sse", ascending=False).reset_index(drop=True)
    total_sse = max(float(frame["sse"].sum()), 1e-12)
    output = {}
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, int(math.ceil(fraction * len(frame))))
        key = int(round(100 * fraction))
        output[f"top{key}pct_well_sse_share"] = float(
            frame.iloc[:count]["sse"].sum() / total_sse
        )
        remainder = frame.iloc[count:]
        output[f"rmse_without_top{key}pct_wells"] = (
            float(np.sqrt(remainder["sse"].sum() / remainder["rows"].sum()))
            if len(remainder)
            else 0.0
        )
    return output


def _menu_path_set96(record):
    return MENU_DIR_SET96 / f"{record['well']}__{record['tag']}.npz"


def _load_records_set96():
    all_wells = sorted(
        set(train_manifest["well"].astype(str))
        | set(valid_manifest["well"].astype(str))
    )
    references = ReferenceStore(all_wells)
    basis = _spline_basis(EVAL_COLUMNS, SPLINE_CONTROLS)
    train_records = [
        _prepare_record(_record_from_panel(row.path, row.fold), references, basis)
        for row in tqdm(
            train_manifest.itertuples(index=False),
            total=len(train_manifest),
            desc="reload SetRanker train panels",
        )
    ]
    valid_records = [
        _prepare_record(_record_from_panel(row.path, row.fold), references, basis)
        for row in tqdm(
            valid_manifest.itertuples(index=False),
            total=len(valid_manifest),
            desc="reload SetRanker valid panels",
        )
    ]
    return references, basis, train_records, valid_records


# ---------------------------------------------------------------------------
# Fixed, truth-independent 96-path menu
# ---------------------------------------------------------------------------
def _heuristic_score_set96(summary):
    summary = np.asarray(summary, np.float32)
    # _candidate_features summary columns:
    # 0 NCC, 1 MAE/4, 2 mismatch-RMSE/4, 3 coverage,
    # 4 self-NCC, 5 self-MAE/4, 6 self coverage, 13 TV, 14 curvature.
    return (
        1.00 * summary[:, 0]
        - 0.35 * summary[:, 1]
        - 0.20 * summary[:, 2]
        + 0.15 * summary[:, 3]
        + 0.20 * summary[:, 4]
        - 0.10 * summary[:, 5]
        + 0.05 * summary[:, 6]
        - 0.03 * summary[:, 13]
        - 0.03 * summary[:, 14]
    ).astype(np.float32)


def _fixed_menu_set96(record, templates, basis, seed):
    rng = np.random.default_rng(int(seed))
    coefficients = _initial_coefficients(
        templates,
        basis.shape[1],
        seed,
        random_count=MENU_RANDOM_COUNT_SET96,
    ).astype(np.float32)
    while len(coefficients) < MENU_SIZE_SET96:
        extra = np.stack(
            [
                _smooth_noise(
                    rng,
                    basis.shape[1],
                    float(rng.choice([8.0, 16.0, 24.0, 32.0, 48.0])),
                )
                for _ in range(MENU_SIZE_SET96)
            ]
        )
        coefficients = _deduplicate_coefficients(
            np.vstack(
                [
                    coefficients,
                    np.clip(extra, -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT),
                ]
            )
        )

    _, summary, correction = _candidate_features(
        record, coefficients, basis, mode="full"
    )
    heuristic = _heuristic_score_set96(summary)
    selected = []

    def add(index):
        index = int(index)
        if index not in selected:
            selected.append(index)

    # Always preserve the zero/centre path.
    zero_index = int(np.argmin(np.mean(coefficients * coefficients, axis=1)))
    add(zero_index)
    for index in np.argsort(-heuristic, kind="mergesort"):
        add(index)
        if len(selected) >= HEURISTIC_KEEP_SET96:
            break

    # Fill by farthest-point trajectory diversity. This is invariant to truth
    # and prevents the heuristic shortlist from collapsing into one GR mode.
    sampled = correction[:, ::4].astype(np.float32)
    norm = np.sum(sampled * sampled, axis=1)
    min_distance_squared = np.full(len(coefficients), np.inf, np.float64)
    for index in selected:
        distance_squared = (
            norm
            + norm[index]
            - 2.0 * (sampled @ sampled[index])
        ) / max(sampled.shape[1], 1)
        min_distance_squared = np.minimum(
            min_distance_squared, np.maximum(distance_squared, 0.0)
        )
    min_distance_squared[np.asarray(selected, np.int64)] = -np.inf

    while len(selected) < MENU_SIZE_SET96:
        index = int(np.argmax(min_distance_squared))
        if not np.isfinite(min_distance_squared[index]):
            break
        add(index)
        distance_squared = (
            norm
            + norm[index]
            - 2.0 * (sampled @ sampled[index])
        ) / max(sampled.shape[1], 1)
        min_distance_squared = np.minimum(
            min_distance_squared, np.maximum(distance_squared, 0.0)
        )
        min_distance_squared[np.asarray(selected, np.int64)] = -np.inf

    if len(selected) < MENU_SIZE_SET96:
        for index in np.argsort(-heuristic, kind="mergesort"):
            add(index)
            if len(selected) >= MENU_SIZE_SET96:
                break
    if len(selected) != MENU_SIZE_SET96:
        raise RuntimeError(
            f"fixed menu has {len(selected)} candidates, expected {MENU_SIZE_SET96}"
        )
    selected = np.asarray(selected, np.int64)
    return {
        "coefficients": coefficients[selected].astype(np.float32),
        "heuristic": heuristic[selected].astype(np.float32),
        "pool_candidates": int(len(coefficients)),
    }


def _build_menus_set96(records, templates, basis, split_name):
    rows = []
    started = time.time()
    for index, record in enumerate(
        tqdm(records, desc=f"build fixed96 {split_name} menus")
    ):
        path = _menu_path_set96(record)
        valid_cache = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    valid_cache = bool(
                        "version" in data.files
                        and int(data["version"][0]) == MENU_VERSION_SET96
                        and data["coefficients"].shape[0] == MENU_SIZE_SET96
                    )
            except Exception:
                valid_cache = False
        if not valid_cache:
            menu = _fixed_menu_set96(
                record,
                templates,
                basis,
                seed=SEED_SET96 + 10007 * index + (0 if split_name == "train" else 7000001),
            )
            np.savez_compressed(
                path,
                version=np.asarray([MENU_VERSION_SET96], np.int16),
                well=np.asarray([record["well"]], str),
                tag=np.asarray([record["tag"]], str),
                coefficients=menu["coefficients"].astype(np.float16),
                heuristic=menu["heuristic"].astype(np.float32),
                pool_candidates=np.asarray([menu["pool_candidates"]], np.int16),
            )
            status = "built"
        else:
            status = "cached"
        with np.load(path, allow_pickle=False) as data:
            coefficients = data["coefficients"].astype(np.float32)
            pool_candidates = int(data["pool_candidates"][0])
        oracle = _candidate_exact_rmse(record, coefficients, basis)
        rows.append(
            {
                "split": split_name,
                "well": record["well"],
                "tag": record["tag"],
                "status": status,
                "candidates": int(len(coefficients)),
                "pool_candidates": pool_candidates,
                "oracle_rmse": float(np.min(oracle)),
            }
        )
        if (index + 1) % 250 == 0:
            print(
                "FIXED96 MENU",
                {
                    "split": split_name,
                    "panels": index + 1,
                    "total": len(records),
                    "elapsed_minutes": (time.time() - started) / 60.0,
                },
                flush=True,
            )
    return pd.DataFrame(rows)


def _oracle_audit_set96(records, basis):
    rows = []
    total_sse = 0.0
    total_rows = 0
    truth_target_sse = 0.0
    for record in records:
        with np.load(_menu_path_set96(record), allow_pickle=False) as data:
            coefficients = data["coefficients"].astype(np.float32)
        candidates = _candidate_row_predictions(record, coefficients, basis)
        truth = np.asarray(record["eval_truth"], np.float32)
        mse = np.mean((candidates - truth[None]) ** 2, axis=1)
        rmse = np.sqrt(np.maximum(mse, 1e-12))
        probability = np.exp(
            np.clip(
                -(rmse - rmse.min()) / TARGET_TEMPERATURE_FT_SET96,
                -80.0,
                0.0,
            )
        )
        probability /= max(float(probability.sum()), 1e-12)
        truth_target = probability @ candidates
        total_sse += float(np.min(mse) * len(truth))
        truth_target_sse += float(np.sum((truth_target - truth) ** 2))
        total_rows += len(truth)
        rows.append(
            {
                "well": record["well"],
                "rows": len(truth),
                "oracle_rmse": float(np.sqrt(np.min(mse))),
                "truth_target_rmse": _rmse_set96(truth, truth_target),
            }
        )
    bywell = pd.DataFrame(rows)
    report = {
        "rows": int(total_rows),
        "wells": int(len(bywell)),
        "candidates": MENU_SIZE_SET96,
        "oracle_rmse": float(np.sqrt(total_sse / max(total_rows, 1))),
        "truth_target_rmse": float(
            np.sqrt(truth_target_sse / max(total_rows, 1))
        ),
        "median_oracle_rmse": float(bywell["oracle_rmse"].median()),
        "p95_oracle_rmse": float(bywell["oracle_rmse"].quantile(0.95)),
    }
    return report, bywell


# ---------------------------------------------------------------------------
# Dataset and model
# ---------------------------------------------------------------------------
class Fixed96Dataset(Dataset):
    def __init__(self, records, basis, mode="full"):
        self.records = records
        self.basis = basis
        self.mode = str(mode)
        weights = np.asarray([record["panel_weight"] for record in records], np.float64)
        positive = weights[weights > 0]
        self.weight_scale = float(np.median(positive)) if len(positive) else 1.0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(_menu_path_set96(record), allow_pickle=False) as data:
            coefficients = data["coefficients"].astype(np.float32)
        sequence, summary, _ = _candidate_features(
            record, coefficients, self.basis, mode=self.mode
        )
        candidate_rows = _candidate_row_predictions(
            record, coefficients, self.basis
        ).astype(np.float32)
        truth = record["eval_truth"].astype(np.float32)
        candidate_mse = np.mean(
            (candidate_rows - truth[None]) ** 2, axis=1
        ).astype(np.float32)
        panel_weight = np.float32(
            np.clip(
                record["panel_weight"] / max(self.weight_scale, 1e-6),
                0.20,
                5.0,
            )
        )
        return {
            "sequence": torch.from_numpy(sequence),
            "summary": torch.from_numpy(summary),
            "candidate_rows": torch.from_numpy(candidate_rows),
            "candidate_mse": torch.from_numpy(candidate_mse),
            "truth": torch.from_numpy(truth),
            "panel_weight": torch.tensor(panel_weight, dtype=torch.float32),
        }


class Set96ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * int(dilation),
                dilation=int(dilation),
                groups=channels,
            ),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class FullWellFixed96SetRanker(nn.Module):
    def __init__(self, sequence_channels, summary_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(sequence_channels, 48, 7, stride=2, padding=3),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            Set96ResidualBlock(48, 1),
            Set96ResidualBlock(48, 2),
            Set96ResidualBlock(48, 4),
            nn.Conv1d(48, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            Set96ResidualBlock(64, 2),
            Set96ResidualBlock(64, 4),
        )
        self.summary = nn.Sequential(
            nn.Linear(summary_channels, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, 48),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 * 3 + 48, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=384,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set = nn.TransformerEncoder(layer, num_layers=3)
        self.score = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, sequence, summary):
        batch, candidates, channels, length = sequence.shape
        values = sequence.reshape(batch * candidates, channels, length)
        hidden = self.sequence(values)
        pooled = torch.cat(
            [hidden.mean(dim=-1), hidden.amax(dim=-1), hidden.std(dim=-1)],
            dim=1,
        )
        summary_hidden = self.summary(
            summary.reshape(batch * candidates, summary.shape[-1])
        )
        token = self.fusion(torch.cat([pooled, summary_hidden], dim=1))
        token = token.reshape(batch, candidates, 192)
        token = self.candidate_set(token)
        logits = self.score(token).squeeze(-1)
        temperature = torch.exp(self.log_temperature).clamp(0.25, 4.0)
        return logits / temperature, temperature


def _loss_set96(
    logits,
    candidate_rows,
    candidate_mse,
    truth,
    panel_weight,
    epoch,
):
    logits = logits.float()
    candidate_rows = candidate_rows.float()
    candidate_mse = candidate_mse.float()
    truth = truth.float()
    probability = torch.softmax(logits, dim=1)
    posterior = torch.sum(probability[:, :, None] * candidate_rows, dim=1)
    posterior_mse = torch.mean((posterior - truth) ** 2, dim=1)

    candidate_rmse = torch.sqrt(candidate_mse.clamp_min(1e-8))
    relative = candidate_rmse - candidate_rmse.amin(dim=1, keepdim=True)
    target = torch.softmax(
        -relative / TARGET_TEMPERATURE_FT_SET96,
        dim=1,
    )
    listwise = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    expected_mse = torch.sum(probability * candidate_mse, dim=1)

    best = torch.argmin(candidate_mse, dim=1)
    best_logit = logits.gather(1, best[:, None])
    margin = torch.clamp(relative / 4.0, 0.0, 2.0)
    pairwise = F.relu(margin - (best_logit - logits))
    pair_mask = torch.ones_like(pairwise).scatter(1, best[:, None], 0.0)
    pairwise = (pairwise * pair_mask).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)

    # Two warm-up epochs establish a sensible posterior before exact MSE
    # becomes the dominant objective.
    posterior_weight = 0.25 if int(epoch) <= 2 else LOSS_POSTERIOR_SET96
    per_panel = (
        posterior_weight * posterior_mse / 25.0
        + LOSS_LISTWISE_SET96 * listwise
        + LOSS_EXPECTED_SET96 * expected_mse / 25.0
        + LOSS_PAIRWISE_SET96 * pairwise
    )
    weight = panel_weight.float().clamp(0.20, 5.0)
    total = (per_panel * weight).mean()
    return total, {
        "posterior_mse": posterior_mse.mean(),
        "listwise": listwise.mean(),
        "expected_mse": expected_mse.mean(),
        "pairwise": pairwise.mean(),
        "entropy": (
            -probability * torch.log(probability.clamp_min(1e-12))
        ).sum(dim=1).mean(),
    }


def _make_scaler_set96(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def _loader_set96(dataset, shuffle, seed, device):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE_SET96,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )


@torch.no_grad()
def _evaluate_set96(
    model,
    records,
    basis,
    device,
    mode="full",
    shuffled=False,
    return_rows=False,
):
    model.eval()
    errors = []
    map_errors = []
    oracle_sse = 0.0
    truth_target_sse = 0.0
    total_rows = 0
    bywell = []
    row_frames = []
    correlations = []

    for index, record in enumerate(records):
        with np.load(_menu_path_set96(record), allow_pickle=False) as data:
            coefficients = data["coefficients"].astype(np.float32)
        reference_override = None
        feature_mode = mode
        if shuffled:
            donor = records[(index + max(1, len(records) // 3)) % len(records)]
            reference_override = {
                "tvt": donor["reference_tvt"],
                "gr": donor["reference_gr"],
            }
            feature_mode = "full"
        sequence, summary, _ = _candidate_features(
            record,
            coefficients,
            basis,
            mode=feature_mode,
            reference_override=reference_override,
        )
        candidate_rows = _candidate_row_predictions(
            record, coefficients, basis
        ).astype(np.float32)
        truth = record["eval_truth"].astype(np.float64)
        candidate_mse = np.mean(
            (candidate_rows.astype(np.float64) - truth[None]) ** 2,
            axis=1,
        )

        sequence_tensor = torch.from_numpy(sequence)[None].to(device)
        summary_tensor = torch.from_numpy(summary)[None].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits, temperature = model(sequence_tensor, summary_tensor)
        logits = logits[0].float().cpu().numpy()
        logits -= float(np.max(logits))
        probability = np.exp(np.clip(logits, -80.0, 0.0))
        probability /= max(float(probability.sum()), 1e-12)
        posterior = probability @ candidate_rows
        map_prediction = candidate_rows[int(np.argmax(probability))]

        candidate_rmse = np.sqrt(np.maximum(candidate_mse, 1e-12))
        truth_probability = np.exp(
            np.clip(
                -(candidate_rmse - candidate_rmse.min())
                / TARGET_TEMPERATURE_FT_SET96,
                -80.0,
                0.0,
            )
        )
        truth_probability /= max(float(truth_probability.sum()), 1e-12)
        truth_target = truth_probability @ candidate_rows

        error = posterior.astype(np.float64) - truth
        map_error = map_prediction.astype(np.float64) - truth
        errors.append(error)
        map_errors.append(map_error)
        oracle_sse += float(np.min(candidate_mse) * len(truth))
        truth_target_sse += float(np.sum((truth_target - truth) ** 2))
        total_rows += len(truth)
        correlation = spearmanr(-logits, candidate_rmse).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        bywell.append(
            {
                "well": record["well"],
                "rows": len(truth),
                "sse": float(np.sum(error**2)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "map_rmse": float(np.sqrt(np.mean(map_error**2))),
                "oracle_rmse": float(np.sqrt(np.min(candidate_mse))),
                "truth_target_rmse": _rmse_set96(truth, truth_target),
                "entropy": float(
                    -np.sum(probability * np.log(probability + 1e-12))
                ),
            }
        )
        if return_rows:
            row_frames.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": record["eval_rows"],
                        "truth": truth,
                        "prediction": posterior,
                        "map_prediction": map_prediction,
                    }
                )
            )

    error = np.concatenate(errors)
    map_error = np.concatenate(map_errors)
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": int(len(error)),
        "wells": int(len(bywell)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "map_rmse": float(np.sqrt(np.mean(map_error**2))),
        "oracle_rmse": float(np.sqrt(oracle_sse / max(total_rows, 1))),
        "truth_target_rmse": float(
            np.sqrt(truth_target_sse / max(total_rows, 1))
        ),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "mean_entropy": float(bywell["entropy"].mean()),
        "mean_rank_spearman": float(np.mean(correlations)) if correlations else np.nan,
        "temperature": float(temperature.detach().cpu()),
    }
    metrics.update(_tail_metrics_set96(bywell))
    rows = pd.concat(row_frames, ignore_index=True) if row_frames else None
    return metrics, bywell, rows


def _train_set96(train_records, valid_records, basis, device):
    probe_dataset = Fixed96Dataset(train_records[:1], basis, mode="full")
    probe = probe_dataset[0]
    sequence_channels = int(probe["sequence"].shape[1])
    summary_channels = int(probe["summary"].shape[1])
    del probe, probe_dataset

    _seed_set96(SEED_SET96)
    model = FullWellFixed96SetRanker(
        sequence_channels, summary_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_SET96,
        weight_decay=WEIGHT_DECAY_SET96,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_SET96,
        eta_min=MIN_LEARNING_RATE_SET96,
    )
    scaler = _make_scaler_set96(device)
    dataset = Fixed96Dataset(train_records, basis, mode="full")
    loader = _loader_set96(
        dataset, shuffle=True, seed=SEED_SET96 + 31, device=device
    )

    best_rmse = np.inf
    best_epoch = 0
    stale = 0
    history = []
    checkpoint_path = MODEL_DIR_SET96 / "best_fold1.pt"
    started = time.time()

    for epoch in range(1, EPOCHS_SET96 + 1):
        model.train()
        loss_rows = []
        epoch_started = time.time()
        for batch_index, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            sequence = batch["sequence"].to(device, non_blocking=True)
            summary = batch["summary"].to(device, non_blocking=True)
            candidate_rows = batch["candidate_rows"].to(device, non_blocking=True)
            candidate_mse = batch["candidate_mse"].to(device, non_blocking=True)
            truth = batch["truth"].to(device, non_blocking=True)
            panel_weight = batch["panel_weight"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits, temperature = model(sequence, summary)
                loss, components = _loss_set96(
                    logits,
                    candidate_rows,
                    candidate_mse,
                    truth,
                    panel_weight,
                    epoch,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in components.items()
                    },
                }
            )
            if batch_index % 500 == 0 or batch_index == len(loader):
                print(
                    "SET96 TRAIN",
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(loader),
                        "loss": float(np.mean([row["loss"] for row in loss_rows[-100:]])),
                        "temperature": float(temperature.detach().cpu()),
                        "epoch_minutes": (time.time() - epoch_started) / 60.0,
                    },
                    flush=True,
                )
        scheduler.step()
        metrics, bywell, _ = _evaluate_set96(
            model, valid_records, basis, device, mode="full", return_rows=False
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{name}": float(np.mean([item[name] for item in loss_rows]))
                for name in loss_rows[0]
            },
            **{f"valid_{name}": value for name, value in metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("SET96 EPOCH", row, flush=True)

        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = float(metrics["rmse"])
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                {
                    "model": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                    "epoch": best_epoch,
                    "metrics": metrics,
                    "sequence_channels": sequence_channels,
                    "summary_channels": summary_channels,
                    "menu_size": MENU_SIZE_SET96,
                    "seed": SEED_SET96,
                },
                checkpoint_path,
            )
            bywell.to_csv(OUT_DIR_SET96 / "best_bywell.csv", index=False)
        else:
            stale += 1
        if epoch >= 5 and stale >= PATIENCE_SET96:
            print(f"SetRanker early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(
        OUT_DIR_SET96 / "training_history.csv", index=False
    )
    if not checkpoint_path.exists():
        raise RuntimeError("SetRanker produced no checkpoint")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, checkpoint_path, (time.time() - started) / 60.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
started_set96 = time.time()
device_set96 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device_set96.type == "cuda":
    torch.set_float32_matmul_precision("high")

templates_path_set96 = BASE_OUT_DIR_SET96 / "foldclean_spline_templates.npy"
if not templates_path_set96.exists():
    raise FileNotFoundError(templates_path_set96)
templates_set96 = np.load(templates_path_set96).astype(np.float32)

references_set96, basis_set96, train_records_set96, valid_records_set96 = (
    _load_records_set96()
)

print(
    {
        "experiment": "fullwell_fixed96_setranker_onefold_v0",
        "sample_unit": "complete well/pseudo-PS panel",
        "standalone": True,
        "device": str(device_set96),
        "train_panels": len(train_records_set96),
        "train_wells": len({record["well"] for record in train_records_set96}),
        "valid_wells": len(valid_records_set96),
        "menu_size": MENU_SIZE_SET96,
        "templates": len(templates_set96),
    },
    flush=True,
)

train_menu_report_set96 = _build_menus_set96(
    train_records_set96, templates_set96, basis_set96, "train"
)
valid_menu_report_set96 = _build_menus_set96(
    valid_records_set96, templates_set96, basis_set96, "valid"
)
menu_report_set96 = pd.concat(
    [train_menu_report_set96, valid_menu_report_set96], ignore_index=True
)
menu_report_set96.to_csv(OUT_DIR_SET96 / "menu_build_report.csv", index=False)

oracle_report_set96, oracle_bywell_set96 = _oracle_audit_set96(
    valid_records_set96, basis_set96
)
oracle_bywell_set96.to_csv(OUT_DIR_SET96 / "oracle_bywell.csv", index=False)
(OUT_DIR_SET96 / "oracle_report.json").write_text(
    json.dumps(oracle_report_set96, indent=2), encoding="utf-8"
)
print("\nFIXED96 ORACLE GATE")
print(json.dumps(oracle_report_set96, indent=2))

if oracle_report_set96["oracle_rmse"] > ORACLE_GATE_SET96:
    verdict_set96 = {
        "experiment": "fullwell_fixed96_setranker_onefold_v0",
        "stage": "fixed_menu_oracle_gate",
        "standalone": True,
        "oracle": oracle_report_set96,
        "oracle_gate_ft": ORACLE_GATE_SET96,
        "strict_pass": False,
        "next": (
            "FAIL -> fixed K=96 pruning lost the measured 4.x ceiling; "
            "do not train or tune the SetRanker"
        ),
        "elapsed_minutes": (time.time() - started_set96) / 60.0,
    }
    (OUT_DIR_SET96 / "verdict.json").write_text(
        json.dumps(verdict_set96, indent=2), encoding="utf-8"
    )
    print("\nFINAL VERDICT")
    print(json.dumps(verdict_set96, indent=2))
    raise RuntimeError(json.dumps(verdict_set96, indent=2))

model_set96, checkpoint_set96, checkpoint_path_set96, training_minutes_set96 = (
    _train_set96(
        train_records_set96,
        valid_records_set96,
        basis_set96,
        device_set96,
    )
)

full_metrics_set96, full_bywell_set96, full_rows_set96 = _evaluate_set96(
    model_set96,
    valid_records_set96,
    basis_set96,
    device_set96,
    mode="full",
    return_rows=True,
)
full_bywell_set96.to_csv(OUT_DIR_SET96 / "full_bywell.csv", index=False)
full_rows_set96.to_pickle(OUT_DIR_SET96 / "fold1_predictions.pkl")

controls_set96 = [{"mode": "full", **full_metrics_set96}]
if full_metrics_set96["rmse"] <= 7.0:
    no_gr_metrics_set96, no_gr_bywell_set96, _ = _evaluate_set96(
        model_set96,
        valid_records_set96,
        basis_set96,
        device_set96,
        mode="no_gr",
        return_rows=False,
    )
    shuffled_metrics_set96, shuffled_bywell_set96, _ = _evaluate_set96(
        model_set96,
        valid_records_set96,
        basis_set96,
        device_set96,
        mode="full",
        shuffled=True,
        return_rows=False,
    )
    no_gr_bywell_set96.to_csv(OUT_DIR_SET96 / "no_gr_bywell.csv", index=False)
    shuffled_bywell_set96.to_csv(
        OUT_DIR_SET96 / "shuffled_typewell_bywell.csv", index=False
    )
    controls_set96.extend(
        [
            {"mode": "no_gr", **no_gr_metrics_set96},
            {"mode": "shuffled_typewell", **shuffled_metrics_set96},
        ]
    )
control_report_set96 = pd.DataFrame(controls_set96)
control_report_set96["degradation_vs_full"] = (
    control_report_set96["rmse"] - full_metrics_set96["rmse"]
)
control_report_set96.to_csv(
    OUT_DIR_SET96 / "interventions.csv", index=False
)

no_gr_degradation_set96 = (
    float(
        control_report_set96.loc[
            control_report_set96["mode"].eq("no_gr"),
            "degradation_vs_full",
        ].iloc[0]
    )
    if "no_gr" in set(control_report_set96["mode"])
    else None
)
shuffled_degradation_set96 = (
    float(
        control_report_set96.loc[
            control_report_set96["mode"].eq("shuffled_typewell"),
            "degradation_vs_full",
        ].iloc[0]
    )
    if "shuffled_typewell" in set(control_report_set96["mode"])
    else None
)

architecture_pass_set96 = bool(
    full_metrics_set96["rmse"] <= ARCHITECTURE_GATE_SET96
)
target_pass_set96 = bool(full_metrics_set96["rmse"] <= TARGET_GATE_SET96)
median_pass_set96 = bool(
    full_metrics_set96["median_well_rmse"] <= MEDIAN_GATE_SET96
)
tail_pass_set96 = bool(
    full_metrics_set96["p95_well_rmse"] <= P95_GATE_SET96
)
causal_pass_set96 = bool(
    no_gr_degradation_set96 is not None
    and shuffled_degradation_set96 is not None
    and no_gr_degradation_set96 >= CAUSAL_GATE_SET96
    and shuffled_degradation_set96 >= CAUSAL_GATE_SET96
)
strict_pass_set96 = bool(
    architecture_pass_set96 and median_pass_set96 and tail_pass_set96 and causal_pass_set96
)

verdict_set96 = {
    "experiment": "fullwell_fixed96_setranker_onefold_v0",
    "sample_unit": "complete well/pseudo-PS panel",
    "standalone": True,
    "fold": VALID_FOLD_SET96,
    "train_panels": len(train_records_set96),
    "train_wells": len({record["well"] for record in train_records_set96}),
    "valid_wells": len(valid_records_set96),
    "menu": {
        "size": MENU_SIZE_SET96,
        "heuristic_keep": HEURISTIC_KEEP_SET96,
        "oracle": oracle_report_set96,
        "same_train_validation_algorithm": True,
    },
    "checkpoint": str(checkpoint_path_set96),
    "checkpoint_epoch": int(checkpoint_set96["epoch"]),
    "parameters": int(
        sum(parameter.numel() for parameter in model_set96.parameters())
    ),
    "full": full_metrics_set96,
    "no_gr_degradation": no_gr_degradation_set96,
    "shuffled_typewell_degradation": shuffled_degradation_set96,
    "gates": {
        "architecture_rmse_le": ARCHITECTURE_GATE_SET96,
        "target_rmse_le": TARGET_GATE_SET96,
        "median_well_rmse_le": MEDIAN_GATE_SET96,
        "p95_well_rmse_le": P95_GATE_SET96,
        "causal_degradation_ge": CAUSAL_GATE_SET96,
    },
    "architecture_pass": architecture_pass_set96,
    "target_5p5_pass": target_pass_set96,
    "median_pass": median_pass_set96,
    "tail_pass": tail_pass_set96,
    "causal_pass": causal_pass_set96,
    "strict_pass": strict_pass_set96,
    "training_minutes": training_minutes_set96,
    "elapsed_minutes": (time.time() - started_set96) / 60.0,
    "next": (
        "PASS <=5.5 -> freeze recipe, repeat seed, then five outer folds"
        if target_pass_set96 and causal_pass_set96
        else (
            "PASS <=6.0 -> run one prespecified structural correction only"
            if strict_pass_set96
            else (
                "HOLD 6-7 -> diagnose posterior calibration once; no broad tuning"
                if full_metrics_set96["rmse"] <= 7.0
                else "FAIL >7 -> stop fixed96 SetRanker architecture"
            )
        )
    ),
}
(OUT_DIR_SET96 / "verdict.json").write_text(
    json.dumps(verdict_set96, indent=2), encoding="utf-8"
)

print("\nINTERVENTIONS")
print(control_report_set96.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict_set96, indent=2))
print("saved:", OUT_DIR_SET96)

del train_records_set96, valid_records_set96, references_set96
gc.collect()
if device_set96.type == "cuda":
    torch.cuda.empty_cache()

