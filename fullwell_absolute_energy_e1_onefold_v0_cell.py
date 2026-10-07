"""PDF-aligned E1 gateway: menu-size-independent absolute path energy.

Run in the same AutoDL kernel after
``fullwell_realwarp_energy_onefold_v0_cell.py``. This experiment reuses only
the fold-clean panel/reference helpers and the outer-train spline templates.
It does not reuse the old trained energy checkpoint.

Every candidate is scored independently as an absolute predicted log-RMSE.
There is no candidate-set Transformer and no candidate softmax decoder. Train,
validation, and deployment menus use one frozen truth-independent generator.
The primary output is the minimum-energy complete path.

No strong, CP12, Gold, Neighbor, neighbour-well data, validation truth at
inference, or LB feedback is used.
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


REQUIRED_E1 = (
    "OOF_ROOT", "train_manifest", "valid_manifest", "ReferenceStore",
    "_record_from_panel", "_prepare_record", "_spline_basis",
    "_initial_coefficients", "_candidate_features",
    "_candidate_row_predictions", "_candidate_exact_rmse", "_smooth_noise",
    "_deduplicate_coefficients", "SPLINE_CONTROLS", "EVAL_COLUMNS",
    "COEFFICIENT_CAP_FT",
)
missing_e1 = [name for name in REQUIRED_E1 if name not in globals()]
if missing_e1:
    raise RuntimeError(
        "Run fullwell_realwarp_energy_onefold_v0_cell.py first in this kernel; "
        "missing globals: " + repr(missing_e1)
    )

BASE_DIR_E1 = Path(OOF_ROOT) / "fullwell_realwarp_energy_onefold_v0"
OUT_DIR_E1 = Path(OOF_ROOT) / "fullwell_absolute_energy_e1_onefold_v0"
MODEL_DIR_E1 = OUT_DIR_E1 / "models"
MENU_DIR_E1 = OUT_DIR_E1 / "fixed_menus"
for directory in (OUT_DIR_E1, MODEL_DIR_E1, MENU_DIR_E1):
    directory.mkdir(parents=True, exist_ok=True)

SEED_E1 = 20260728
FIXED_K_E1 = 96
EXPANDED_K_E1 = 192
HEURISTIC_KEEP_E1 = 48
RANDOM_COUNT_E1 = 160
MENU_VERSION_E1 = 1

EPOCHS_E1 = 10
PATIENCE_E1 = 3
LEARNING_RATE_E1 = 2.5e-4
MIN_LEARNING_RATE_E1 = 2.5e-5
WEIGHT_DECAY_E1 = 2.0e-4
BATCH_SIZE_E1 = 1

REGRESSION_WEIGHT_E1 = 1.0
PAIRWISE_WEIGHT_E1 = 0.40
NEAR_ORACLE_WEIGHT_E1 = 2.0
PAIR_MARGIN_SCALE_E1 = 4.0

FIXED_ORACLE_SANITY_GATE_E1 = 6.00
EXPANDED_ORACLE_GATE_E1 = 5.20
RECOVERY_GATE_E1 = 0.30
EXPANSION_PENALTY_GATE_E1 = 0.10
CAUSAL_GATE_E1 = 0.30
ARCHITECTURE_GATE_E1 = 7.0
TARGET_GATE_E1 = 5.5
EPSILON_FT_E1 = 1.0
BREADTH_GATE_E1 = 0.55
MEDIAN_DELTA_GATE_E1 = -0.10


def _seed_e1(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rmse_e1(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _menu_path_e1(record):
    return MENU_DIR_E1 / f"{record['well']}__{record['tag']}.npz"


def _reload_records_e1():
    wells = sorted(
        set(train_manifest["well"].astype(str))
        | set(valid_manifest["well"].astype(str))
    )
    references = ReferenceStore(wells)
    basis = _spline_basis(EVAL_COLUMNS, SPLINE_CONTROLS)
    train_records = [
        _prepare_record(_record_from_panel(row.path, row.fold), references, basis)
        for row in tqdm(
            train_manifest.itertuples(index=False), total=len(train_manifest),
            desc="reload E1 train panels"
        )
    ]
    valid_records = [
        _prepare_record(_record_from_panel(row.path, row.fold), references, basis)
        for row in tqdm(
            valid_manifest.itertuples(index=False), total=len(valid_manifest),
            desc="reload E1 valid panels"
        )
    ]
    return references, basis, train_records, valid_records


def _heuristic_e1(summary):
    summary = np.asarray(summary, np.float32)
    return (
        summary[:, 0]
        - 0.35 * summary[:, 1]
        - 0.20 * summary[:, 2]
        + 0.15 * summary[:, 3]
        + 0.20 * summary[:, 4]
        - 0.10 * summary[:, 5]
        + 0.05 * summary[:, 6]
        - 0.03 * summary[:, 13]
        - 0.03 * summary[:, 14]
    ).astype(np.float32)


def _make_menu_e1(record, templates, basis, seed):
    rng = np.random.default_rng(int(seed))
    coefficients = _initial_coefficients(
        templates, basis.shape[1], seed, random_count=RANDOM_COUNT_E1
    ).astype(np.float32)
    while len(coefficients) < EXPANDED_K_E1:
        extra = np.stack([
            _smooth_noise(
                rng, basis.shape[1],
                float(rng.choice([8.0, 16.0, 24.0, 32.0, 48.0, 64.0]))
            )
            for _ in range(EXPANDED_K_E1)
        ])
        coefficients = _deduplicate_coefficients(np.vstack([
            coefficients,
            np.clip(extra, -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT),
        ]))

    _, summary, correction = _candidate_features(
        record, coefficients, basis, mode="full"
    )
    heuristic = _heuristic_e1(summary)
    selected = []

    def add(index):
        index = int(index)
        if index not in selected:
            selected.append(index)

    add(int(np.argmin(np.mean(coefficients * coefficients, axis=1))))
    for index in np.argsort(-heuristic, kind="mergesort"):
        add(index)
        if len(selected) >= HEURISTIC_KEEP_E1:
            break

    sampled = correction[:, ::4].astype(np.float32)
    norm = np.sum(sampled * sampled, axis=1)
    minimum = np.full(len(coefficients), np.inf, np.float64)
    for index in selected:
        distance = (
            norm + norm[index] - 2.0 * sampled @ sampled[index]
        ) / max(sampled.shape[1], 1)
        minimum = np.minimum(minimum, np.maximum(distance, 0.0))
    minimum[np.asarray(selected)] = -np.inf
    while len(selected) < EXPANDED_K_E1:
        index = int(np.argmax(minimum))
        if not np.isfinite(minimum[index]):
            break
        add(index)
        distance = (
            norm + norm[index] - 2.0 * sampled @ sampled[index]
        ) / max(sampled.shape[1], 1)
        minimum = np.minimum(minimum, np.maximum(distance, 0.0))
        minimum[np.asarray(selected)] = -np.inf
    if len(selected) != EXPANDED_K_E1:
        raise RuntimeError(f"menu size {len(selected)} != {EXPANDED_K_E1}")
    selected = np.asarray(selected, np.int64)
    return coefficients[selected], heuristic[selected], len(coefficients)


def _build_menus_e1(records, templates, basis, split):
    rows = []
    for index, record in enumerate(tqdm(records, desc=f"build E1 {split} menus")):
        path = _menu_path_e1(record)
        cached = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    cached = bool(
                        int(data["version"][0]) == MENU_VERSION_E1
                        and data["coefficients"].shape[0] == EXPANDED_K_E1
                    )
            except Exception:
                cached = False
        if not cached:
            coefficients, heuristic, pool_size = _make_menu_e1(
                record, templates, basis,
                SEED_E1 + 10007 * index + (0 if split == "train" else 7000001)
            )
            np.savez_compressed(
                path,
                version=np.asarray([MENU_VERSION_E1], np.int16),
                coefficients=coefficients.astype(np.float16),
                heuristic=heuristic.astype(np.float32),
                pool_size=np.asarray([pool_size], np.int16),
            )
            status = "built"
        else:
            status = "cached"
        with np.load(path, allow_pickle=False) as data:
            coefficients = data["coefficients"].astype(np.float32)
            pool_size = int(data["pool_size"][0])
        exact = _candidate_exact_rmse(record, coefficients, basis)
        rows.append({
            "split": split,
            "well": record["well"],
            "tag": record["tag"],
            "status": status,
            "fixed_candidates": FIXED_K_E1,
            "expanded_candidates": EXPANDED_K_E1,
            "pool_candidates": pool_size,
            "fixed_oracle_rmse": float(exact[:FIXED_K_E1].min()),
            "expanded_oracle_rmse": float(exact.min()),
        })
        if (index + 1) % 250 == 0:
            print("E1 MENU", {"split": split, "done": index + 1, "total": len(records)}, flush=True)
    return pd.DataFrame(rows)


def _menu_oracle_e1(records, basis, count):
    total_sse = 0.0
    total_rows = 0
    rows = []
    for record in records:
        with np.load(_menu_path_e1(record), allow_pickle=False) as data:
            coefficients = data["coefficients"][:count].astype(np.float32)
        exact = _candidate_exact_rmse(record, coefficients, basis)
        best = float(exact.min())
        total_sse += best * best * len(record["eval_truth"])
        total_rows += len(record["eval_truth"])
        rows.append({"well": record["well"], "rows": len(record["eval_truth"]), "oracle_rmse": best})
    frame = pd.DataFrame(rows)
    return {
        "candidates": int(count),
        "rows": int(total_rows),
        "wells": int(len(frame)),
        "pooled_oracle_rmse": float(np.sqrt(total_sse / max(total_rows, 1))),
        "median_oracle_rmse": float(frame["oracle_rmse"].median()),
        "p95_oracle_rmse": float(frame["oracle_rmse"].quantile(0.95)),
    }, frame


class E1Dataset(Dataset):
    def __init__(self, records, basis):
        self.records = records
        self.basis = basis
        weights = np.asarray([record["panel_weight"] for record in records], np.float64)
        positive = weights[weights > 0]
        self.weight_scale = float(np.median(positive)) if len(positive) else 1.0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(_menu_path_e1(record), allow_pickle=False) as data:
            coefficients = data["coefficients"][:FIXED_K_E1].astype(np.float32)
        sequence, summary, _ = _candidate_features(record, coefficients, self.basis, mode="full")
        exact = _candidate_exact_rmse(record, coefficients, self.basis)
        weight = np.float32(np.clip(
            record["panel_weight"] / max(self.weight_scale, 1e-6), 0.20, 5.0
        ))
        return {
            "sequence": torch.from_numpy(sequence),
            "summary": torch.from_numpy(summary),
            "exact_rmse": torch.from_numpy(exact.astype(np.float32)),
            "panel_weight": torch.tensor(weight, dtype=torch.float32),
        }


class E1ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class FullWellAbsoluteEnergyE1(nn.Module):
    def __init__(self, sequence_channels, summary_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(sequence_channels, 48, 7, stride=2, padding=3),
            nn.GroupNorm(8, 48), nn.GELU(),
            E1ResidualBlock(48, 1), E1ResidualBlock(48, 2), E1ResidualBlock(48, 4),
            nn.Conv1d(48, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.GELU(),
            E1ResidualBlock(64, 2), E1ResidualBlock(64, 4),
        )
        self.summary = nn.Sequential(
            nn.Linear(summary_channels, 48), nn.LayerNorm(48), nn.GELU(),
            nn.Linear(48, 48), nn.GELU(),
        )
        self.energy = nn.Sequential(
            nn.Linear(64 * 3 + 48, 160), nn.LayerNorm(160), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, sequence, summary):
        if sequence.ndim == 4:
            batch, candidates, channels, length = sequence.shape
            sequence = sequence.reshape(batch * candidates, channels, length)
            summary = summary.reshape(batch * candidates, summary.shape[-1])
            shape = (batch, candidates)
        else:
            shape = None
        hidden = self.sequence(sequence)
        pooled = torch.cat([hidden.mean(-1), hidden.amax(-1), hidden.std(-1)], dim=1)
        raw = self.energy(torch.cat([pooled, self.summary(summary)], dim=1)).squeeze(-1)
        predicted_log_rmse = F.softplus(raw)
        return predicted_log_rmse if shape is None else predicted_log_rmse.reshape(*shape)


def _loss_e1(predicted_log_rmse, exact_rmse, panel_weight):
    predicted_log_rmse = predicted_log_rmse.float()
    exact_rmse = exact_rmse.float()
    target = torch.log1p(exact_rmse)
    relative = exact_rmse - exact_rmse.amin(dim=1, keepdim=True)
    candidate_weight = 1.0 + NEAR_ORACLE_WEIGHT_E1 * torch.exp(-relative / 4.0)
    regression = F.smooth_l1_loss(
        predicted_log_rmse, target, beta=0.20, reduction="none"
    )
    regression = (regression * candidate_weight).sum(dim=1) / candidate_weight.sum(dim=1)

    best = torch.argmin(exact_rmse, dim=1)
    best_energy = predicted_log_rmse.gather(1, best[:, None])
    margin = torch.clamp(relative / PAIR_MARGIN_SCALE_E1, 0.0, 2.0)
    pairwise = F.relu(margin - (predicted_log_rmse - best_energy))
    mask = torch.ones_like(pairwise).scatter(1, best[:, None], 0.0)
    pairwise = (pairwise * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    per_panel = REGRESSION_WEIGHT_E1 * regression + PAIRWISE_WEIGHT_E1 * pairwise
    weight = panel_weight.float().clamp(0.20, 5.0)
    return (per_panel * weight).mean(), {
        "regression": regression.mean(),
        "pairwise": pairwise.mean(),
        "calibration_mae": torch.mean(torch.abs(torch.expm1(predicted_log_rmse) - exact_rmse)),
    }


def _score_e1(model, record, coefficients, basis, device, mode="full", reference_override=None, chunk=96):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coefficients), chunk):
            local = coefficients[start:start + chunk]
            sequence, summary, _ = _candidate_features(
                record, local, basis, mode=mode, reference_override=reference_override
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                energy = model(
                    torch.from_numpy(sequence).to(device),
                    torch.from_numpy(summary).to(device),
                )
            outputs.append(energy.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _evaluate_e1(model, records, basis, device, count, mode="full", shuffled=False, return_rows=False):
    totals = {"rows": 0, "safe_sse": 0.0, "selected_sse": 0.0, "oracle_sse": 0.0}
    top_sse = {1: 0.0, 5: 0.0, 10: 0.0}
    recalls = {1: [], 5: [], 10: []}
    correlations = []
    calibration = []
    bywell = []
    row_frames = []
    for index, record in enumerate(tqdm(records, desc=f"E1 evaluate K={count} {mode}")):
        with np.load(_menu_path_e1(record), allow_pickle=False) as data:
            coefficients = data["coefficients"][:count].astype(np.float32)
        override = None
        feature_mode = mode
        if shuffled:
            donor = records[(index + max(1, len(records) // 3)) % len(records)]
            override = {"tvt": donor["reference_tvt"], "gr": donor["reference_gr"]}
            feature_mode = "full"
        energy = _score_e1(model, record, coefficients, basis, device, mode=feature_mode, reference_override=override)
        candidate_rows = _candidate_row_predictions(record, coefficients, basis).astype(np.float64)
        truth = record["eval_truth"].astype(np.float64)
        candidate_sse = np.sum((candidate_rows - truth[None]) ** 2, axis=1)
        candidate_rmse = np.sqrt(candidate_sse / len(truth))
        order = np.argsort(energy, kind="mergesort")
        selected = int(order[0])
        safe = int(np.argmin(np.mean(coefficients * coefficients, axis=1)))
        oracle = int(np.argmin(candidate_sse))
        totals["rows"] += len(truth)
        totals["safe_sse"] += float(candidate_sse[safe])
        totals["selected_sse"] += float(candidate_sse[selected])
        totals["oracle_sse"] += float(candidate_sse[oracle])
        for k in (1, 5, 10):
            top = order[:min(k, len(order))]
            top_best = int(top[np.argmin(candidate_sse[top])])
            top_sse[k] += float(candidate_sse[top_best])
            recalls[k].append(bool(candidate_rmse[top].min() <= candidate_rmse[oracle] + EPSILON_FT_E1))
        correlation = spearmanr(energy, candidate_rmse).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        calibration.append(float(np.mean(np.abs(np.expm1(energy) - candidate_rmse))))
        selected_error = candidate_rows[selected] - truth
        bywell.append({
            "well": record["well"], "rows": len(truth),
            "safe_rmse": float(candidate_rmse[safe]),
            "selected_rmse": float(candidate_rmse[selected]),
            "oracle_rmse": float(candidate_rmse[oracle]),
            "regret": float(candidate_rmse[selected] - candidate_rmse[oracle]),
            "safe_sse": float(candidate_sse[safe]),
            "sse": float(candidate_sse[selected]),
        })
        if return_rows:
            row_frames.append(pd.DataFrame({
                "well": record["well"], "row_idx": record["eval_rows"],
                "truth": truth, "prediction": candidate_rows[selected],
            }))
    n = max(int(totals["rows"]), 1)
    available = max(totals["safe_sse"] - totals["oracle_sse"], 1e-12)
    recovered = totals["safe_sse"] - totals["selected_sse"]
    frame = pd.DataFrame(bywell)
    metrics = {
        "candidates": int(count), "rows": int(totals["rows"]), "wells": len(frame),
        "safe_rmse": float(np.sqrt(totals["safe_sse"] / n)),
        "selected_rmse": float(np.sqrt(totals["selected_sse"] / n)),
        "oracle_rmse": float(np.sqrt(totals["oracle_sse"] / n)),
        "oracle_sse_recovery_fraction": float(recovered / available),
        "mean_well_regret": float(frame["regret"].mean()),
        "median_well_rmse": float(frame["selected_rmse"].median()),
        "p95_well_rmse": float(frame["selected_rmse"].quantile(0.95)),
        "improved_well_fraction": float(
            (frame["selected_rmse"] < frame["safe_rmse"]).mean()
        ),
        "median_well_delta": float(
            (frame["selected_rmse"] - frame["safe_rmse"]).median()
        ),
        "rank_spearman": float(np.mean(correlations)) if correlations else np.nan,
        "absolute_rmse_calibration_mae": float(np.mean(calibration)),
    }
    ordered = frame.sort_values("sse", ascending=False).reset_index(drop=True)
    total_selected_sse = max(float(ordered["sse"].sum()), 1e-12)
    for fraction in (0.05, 0.10, 0.20):
        key = int(round(100 * fraction))
        take = max(1, int(math.ceil(fraction * len(ordered))))
        metrics[f"top{key}pct_well_sse_share"] = float(
            ordered.iloc[:take]["sse"].sum() / total_selected_sse
        )
    safe_worst = frame.sort_values("safe_sse", ascending=False).head(
        max(1, int(math.ceil(0.20 * len(frame))))
    )
    metrics["safe_worst20_rmse"] = float(
        np.sqrt(safe_worst["safe_sse"].sum() / safe_worst["rows"].sum())
    )
    metrics["selected_on_safe_worst20_rmse"] = float(
        np.sqrt(safe_worst["sse"].sum() / safe_worst["rows"].sum())
    )
    metrics["safe_worst20_gain"] = float(
        metrics["safe_worst20_rmse"]
        - metrics["selected_on_safe_worst20_rmse"]
    )
    for k in (1, 5, 10):
        metrics[f"top{k}_oracle_rmse"] = float(np.sqrt(top_sse[k] / n))
        metrics[f"top{k}_regret_rmse"] = float(
            metrics[f"top{k}_oracle_rmse"] - metrics["oracle_rmse"]
        )
        metrics[f"top{k}_epsilon_recall"] = float(np.mean(recalls[k]))
    rows = pd.concat(row_frames, ignore_index=True) if row_frames else None
    return metrics, frame, rows


def _scaler_e1(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def _train_e1(train_records, valid_records, basis, device):
    dataset = E1Dataset(train_records, basis)
    probe = dataset[0]
    _seed_e1(SEED_E1)
    model = FullWellAbsoluteEnergyE1(probe["sequence"].shape[1], probe["summary"].shape[1]).to(device)
    del probe
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_E1, weight_decay=WEIGHT_DECAY_E1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_E1, eta_min=MIN_LEARNING_RATE_E1)
    scaler = _scaler_e1(device)
    generator = torch.Generator().manual_seed(SEED_E1 + 31)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_E1, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), generator=generator)
    best = np.inf
    stale = 0
    history = []
    checkpoint_path = MODEL_DIR_E1 / "best_fold1.pt"
    started = time.time()
    for epoch in range(1, EPOCHS_E1 + 1):
        model.train()
        losses = []
        for batch_index, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            sequence = batch["sequence"].to(device, non_blocking=True)
            summary = batch["summary"].to(device, non_blocking=True)
            exact = batch["exact_rmse"].to(device, non_blocking=True)
            weight = batch["panel_weight"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                prediction = model(sequence, summary)
                loss, components = _loss_e1(prediction, exact, weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append({"loss": float(loss.detach().cpu()), **{k: float(v.detach().cpu()) for k, v in components.items()}})
            if batch_index % 500 == 0 or batch_index == len(loader):
                print("E1 TRAIN", {"epoch": epoch, "batch": batch_index, "batches": len(loader), "loss": float(np.mean([x["loss"] for x in losses[-100:]]))}, flush=True)
        scheduler.step()
        fixed_metrics, fixed_bywell, _ = _evaluate_e1(
            model, valid_records, basis, device, FIXED_K_E1
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{k}": float(np.mean([x[k] for x in losses]))
                for k in losses[0]
            },
            **{f"valid_{k}": v for k, v in fixed_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("E1 EPOCH", row, flush=True)
        # Keep K192 label-blind until the K96 checkpoint is frozen. This makes
        # the final expansion penalty a genuine candidate-count stress test.
        if fixed_metrics["selected_rmse"] < best - 1e-4:
            best = fixed_metrics["selected_rmse"]
            stale = 0
            torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch, "metrics": fixed_metrics, "sequence_channels": int(dataset[0]["sequence"].shape[1]), "summary_channels": int(dataset[0]["summary"].shape[1]), "fixed_k": FIXED_K_E1, "expanded_k": EXPANDED_K_E1}, checkpoint_path)
            fixed_bywell.to_csv(OUT_DIR_E1 / "best_bywell.csv", index=False)
        else:
            stale += 1
        if epoch >= 5 and stale >= PATIENCE_E1:
            print(f"E1 early stopping at epoch {epoch}", flush=True)
            break
    pd.DataFrame(history).to_csv(OUT_DIR_E1 / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, checkpoint_path, (time.time() - started) / 60.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
started_e1 = time.time()
device_e1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device_e1.type == "cuda":
    torch.set_float32_matmul_precision("high")
templates_path_e1 = BASE_DIR_E1 / "foldclean_spline_templates.npy"
if not templates_path_e1.exists():
    raise FileNotFoundError(templates_path_e1)
templates_e1 = np.load(templates_path_e1).astype(np.float32)
references_e1, basis_e1, train_records_e1, valid_records_e1 = _reload_records_e1()

print({
    "experiment": "fullwell_absolute_energy_e1_onefold_v0",
    "device": str(device_e1), "train_panels": len(train_records_e1),
    "valid_wells": len(valid_records_e1), "fixed_k": FIXED_K_E1,
    "expanded_k": EXPANDED_K_E1, "standalone": True,
}, flush=True)

train_menu_e1 = _build_menus_e1(train_records_e1, templates_e1, basis_e1, "train")
valid_menu_e1 = _build_menus_e1(valid_records_e1, templates_e1, basis_e1, "valid")
pd.concat([train_menu_e1, valid_menu_e1], ignore_index=True).to_csv(OUT_DIR_E1 / "menu_report.csv", index=False)
fixed_oracle_e1, fixed_oracle_bywell_e1 = _menu_oracle_e1(valid_records_e1, basis_e1, FIXED_K_E1)
expanded_oracle_e1, expanded_oracle_bywell_e1 = _menu_oracle_e1(valid_records_e1, basis_e1, EXPANDED_K_E1)
fixed_oracle_bywell_e1.to_csv(OUT_DIR_E1 / "oracle_bywell.csv", index=False)
expanded_oracle_bywell_e1.to_csv(OUT_DIR_E1 / "expanded_oracle_bywell.csv", index=False)
print("\nE1 ORACLE GATE")
print(json.dumps({"fixed": fixed_oracle_e1, "expanded": expanded_oracle_e1}, indent=2))

fixed_oracle_ok_e1 = (
    fixed_oracle_e1["pooled_oracle_rmse"] <= FIXED_ORACLE_SANITY_GATE_E1
)
expanded_oracle_ok_e1 = (
    expanded_oracle_e1["pooled_oracle_rmse"] <= EXPANDED_ORACLE_GATE_E1
)
if not (fixed_oracle_ok_e1 and expanded_oracle_ok_e1):
    verdict_e1 = {
        "experiment": "fullwell_absolute_energy_e1_onefold_v0",
        "stage": "oracle_gate",
        "fixed_oracle": fixed_oracle_e1,
        "expanded_oracle": expanded_oracle_e1,
        "gates": {
            "fixed_training_menu_oracle_le": FIXED_ORACLE_SANITY_GATE_E1,
            "expanded_deployment_menu_oracle_le": EXPANDED_ORACLE_GATE_E1,
        },
        "strict_pass": False,
        "next": "FAIL -> E1 training or deployment menu lacks the required ceiling",
    }
    (OUT_DIR_E1 / "verdict.json").write_text(json.dumps(verdict_e1, indent=2), encoding="utf-8")
    print(json.dumps(verdict_e1, indent=2))
    raise RuntimeError(json.dumps(verdict_e1, indent=2))

model_e1, checkpoint_e1, checkpoint_path_e1, training_minutes_e1 = _train_e1(train_records_e1, valid_records_e1, basis_e1, device_e1)
fixed_metrics_e1, fixed_bywell_e1, fixed_rows_e1 = _evaluate_e1(model_e1, valid_records_e1, basis_e1, device_e1, FIXED_K_E1, return_rows=True)
expanded_metrics_e1, expanded_bywell_e1, expanded_rows_e1 = _evaluate_e1(model_e1, valid_records_e1, basis_e1, device_e1, EXPANDED_K_E1, return_rows=True)
fixed_bywell_e1.to_csv(OUT_DIR_E1 / "fixed_bywell.csv", index=False)
expanded_bywell_e1.to_csv(OUT_DIR_E1 / "expanded_bywell.csv", index=False)
fixed_rows_e1.to_pickle(OUT_DIR_E1 / "fold1_fixed_predictions.pkl")
expanded_rows_e1.to_pickle(OUT_DIR_E1 / "fold1_predictions.pkl")

search_penalty_e1 = expanded_metrics_e1["selected_rmse"] - fixed_metrics_e1["selected_rmse"]
controls_e1 = [{"mode": "full", **expanded_metrics_e1}]
if expanded_metrics_e1["selected_rmse"] <= ARCHITECTURE_GATE_E1:
    no_gr_e1, _, _ = _evaluate_e1(model_e1, valid_records_e1, basis_e1, device_e1, EXPANDED_K_E1, mode="no_gr")
    shuffled_e1, _, _ = _evaluate_e1(model_e1, valid_records_e1, basis_e1, device_e1, EXPANDED_K_E1, shuffled=True)
    controls_e1.extend([{"mode": "no_gr", **no_gr_e1}, {"mode": "shuffled_typewell", **shuffled_e1}])
controls_frame_e1 = pd.DataFrame(controls_e1)
controls_frame_e1["degradation_vs_full"] = controls_frame_e1["selected_rmse"] - expanded_metrics_e1["selected_rmse"]
controls_frame_e1.to_csv(OUT_DIR_E1 / "interventions.csv", index=False)

def _control_value_e1(name):
    row = controls_frame_e1[controls_frame_e1["mode"].eq(name)]
    return float(row["degradation_vs_full"].iloc[0]) if len(row) else None

no_gr_degradation_e1 = _control_value_e1("no_gr")
shuffle_degradation_e1 = _control_value_e1("shuffled_typewell")
recovery_pass_e1 = expanded_metrics_e1["oracle_sse_recovery_fraction"] >= RECOVERY_GATE_E1
expansion_pass_e1 = search_penalty_e1 <= EXPANSION_PENALTY_GATE_E1
breadth_pass_e1 = bool(
    expanded_metrics_e1["improved_well_fraction"] >= BREADTH_GATE_E1
    and expanded_metrics_e1["median_well_delta"] <= MEDIAN_DELTA_GATE_E1
    and expanded_metrics_e1["safe_worst20_gain"] >= 0.0
)
causal_pass_e1 = bool(
    no_gr_degradation_e1 is not None and shuffle_degradation_e1 is not None
    and no_gr_degradation_e1 >= CAUSAL_GATE_E1
    and shuffle_degradation_e1 >= CAUSAL_GATE_E1
)
architecture_pass_e1 = expanded_metrics_e1["selected_rmse"] <= ARCHITECTURE_GATE_E1
target_pass_e1 = expanded_metrics_e1["selected_rmse"] <= TARGET_GATE_E1
strict_pass_e1 = bool(
    recovery_pass_e1 and expansion_pass_e1 and breadth_pass_e1
    and causal_pass_e1 and architecture_pass_e1
)

verdict_e1 = {
    "experiment": "fullwell_absolute_energy_e1_onefold_v0",
    "standalone": True, "fold": 1,
    "train_panels": len(train_records_e1), "valid_wells": len(valid_records_e1),
    "checkpoint": str(checkpoint_path_e1), "checkpoint_epoch": int(checkpoint_e1["epoch"]),
    "fixed": fixed_metrics_e1, "expanded": expanded_metrics_e1,
    "search_penalty": float(search_penalty_e1),
    "no_gr_degradation": no_gr_degradation_e1,
    "shuffled_typewell_degradation": shuffle_degradation_e1,
    "gates": {"fixed_training_menu_oracle_le": FIXED_ORACLE_SANITY_GATE_E1, "expanded_deployment_menu_oracle_le": EXPANDED_ORACLE_GATE_E1, "oracle_sse_recovery_ge": RECOVERY_GATE_E1, "expansion_penalty_le": EXPANSION_PENALTY_GATE_E1, "improved_well_fraction_ge": BREADTH_GATE_E1, "median_well_delta_le": MEDIAN_DELTA_GATE_E1, "safe_worst20_gain_ge": 0.0, "causal_degradation_ge": CAUSAL_GATE_E1, "architecture_rmse_le": ARCHITECTURE_GATE_E1, "target_rmse_le": TARGET_GATE_E1},
    "recovery_pass": bool(recovery_pass_e1), "expansion_pass": bool(expansion_pass_e1),
    "breadth_pass": bool(breadth_pass_e1),
    "causal_pass": bool(causal_pass_e1), "architecture_pass": bool(architecture_pass_e1),
    "target_5p5_pass": bool(target_pass_e1), "strict_pass": bool(strict_pass_e1),
    "training_minutes": training_minutes_e1,
    "elapsed_minutes": (time.time() - started_e1) / 60.0,
    "next": (
        "PASS E1 <=5.5 -> freeze and run a new confirmation split"
        if target_pass_e1 and strict_pass_e1 else
        "PASS E1 mechanism -> confirm on a new split before any five-fold expansion"
        if strict_pass_e1 else
        "FAIL -> absolute energy did not recover 30% oracle SSE or failed expansion/causal gate"
    ),
}
(OUT_DIR_E1 / "verdict.json").write_text(json.dumps(verdict_e1, indent=2), encoding="utf-8")
print("\nE1 FIXED")
print(json.dumps(fixed_metrics_e1, indent=2))
print("\nE1 EXPANDED")
print(json.dumps(expanded_metrics_e1, indent=2))
print("\nE1 INTERVENTIONS")
print(controls_frame_e1.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict_e1, indent=2))
print("saved:", OUT_DIR_E1)

del train_records_e1, valid_records_e1, references_e1
gc.collect()
if device_e1.type == "cuda":
    torch.cuda.empty_cache()
