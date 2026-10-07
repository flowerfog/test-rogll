"""Mode-level NLL and hard-negative fine-tuning for the real-warp scorer.

Run in the same AutoDL kernel after ``fullwell_realwarp_energy_onefold_v0``.
The base model, fold-clean spline templates, and helper definitions are reused.

This cell follows the surviving branch of the S5 repair plan:

* training menus are generated exactly like inference menus;
* false low-energy paths found by the frozen v0 scorer are mined;
* labels use exact row-level whole-well RMSE;
* candidates within 2 ft are treated as one mode;
* mode-level NLL and multiplicity-corrected posterior replace broad ranking.

No neighbour, strong, CP12, Gold, or hidden feedback is used.
"""

from pathlib import Path
import gc
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


REQUIRED_GLOBALS_MODE = (
    "OOF_ROOT",
    "OUT_DIR",
    "MODEL_DIR",
    "train_manifest",
    "valid_manifest",
    "ReferenceStore",
    "_record_from_panel",
    "_prepare_record",
    "_spline_basis",
    "_initial_coefficients",
    "_smooth_noise",
    "_deduplicate_coefficients",
    "_candidate_features",
    "_candidate_exact_rmse",
    "_candidate_row_predictions",
    "_coefficients_to_column_correction",
    "_score_coefficients",
    "_search_record",
    "_well_tail_metrics",
    "WholeWellEnergyModel",
    "_make_scaler",
    "SPLINE_CONTROLS",
    "COEFFICIENT_CAP_FT",
    "SEED",
)
missing_mode_globals = [
    name for name in REQUIRED_GLOBALS_MODE if name not in globals()
]
if missing_mode_globals:
    raise RuntimeError(
        "Run fullwell_realwarp_energy_onefold_v0 first in this kernel; "
        "missing globals: " + repr(missing_mode_globals)
    )


BASE_OUT_DIR_MODE = Path(OOF_ROOT) / "fullwell_realwarp_energy_onefold_v0"
OUT_DIR_MODE = Path(OOF_ROOT) / "fullwell_mode_nll_hardneg_v1"
MODEL_DIR_MODE = OUT_DIR_MODE / "models"
MENU_DIR_MODE = OUT_DIR_MODE / "mined_menus"
for directory in (OUT_DIR_MODE, MODEL_DIR_MODE, MENU_DIR_MODE):
    directory.mkdir(parents=True, exist_ok=True)

MODE_SEED = 20260726
MODE_RADIUS_FT = 2.0
TARGET_MODE_TEMPERATURE_FT = 2.0
MENU_CANDIDATES = 48
MINE_TOP_ENERGY = 16
MINE_TOP_ORACLE = 4
MINE_HARD_FALSE = 12
MINE_PARENTS = 8
MINE_CHILDREN = 4
MINE_SCALES_FT = (8.0, 4.0)

MODE_EPOCHS = 10
MODE_PATIENCE = 4
MODE_LEARNING_RATE = 1e-4
MODE_MIN_LEARNING_RATE = 1e-5
MODE_WEIGHT_DECAY = 1e-4
MODE_BATCH_SIZE = 1
MODE_SCORE_CHUNK = 96

LOSS_MODE_NLL = 1.0
LOSS_CANDIDATE_NLL = 0.30
LOSS_POSTERIOR_PATH = 0.45
LOSS_PAIRWISE = 0.45
LOSS_REGRESSION = 0.15


def _mode_softmax(values):
    values = np.asarray(values, np.float64)
    values -= np.max(values)
    probability = np.exp(np.clip(values, -80.0, 0.0))
    return probability / max(float(probability.sum()), 1e-12)


def _mode_labels_from_correction(correction, threshold_ft=MODE_RADIUS_FT):
    correction = np.asarray(correction, np.float64)
    sampled = correction[:, :: max(1, correction.shape[1] // 128)]
    norm = np.sum(sampled * sampled, axis=1)
    distance_squared = (
        norm[:, None] + norm[None, :] - 2.0 * sampled @ sampled.T
    ) / max(sampled.shape[1], 1)
    distance = np.sqrt(np.maximum(distance_squared, 0.0))
    parent = np.arange(len(correction), dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in np.argwhere(np.triu(distance <= threshold_ft, 1)):
        union(int(left), int(right))
    roots = np.asarray([find(index) for index in range(len(parent))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def _multiplicity_corrected_probability(energy, labels):
    energy = np.asarray(energy, np.float64)
    labels = np.asarray(labels, np.int64)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    logits = -energy - np.log(np.maximum(counts[labels], 1))
    return _mode_softmax(logits)


def _mode_menu_path(record):
    return MENU_DIR_MODE / f"{record['well']}__{record['tag']}.npz"


def _load_base_model_mode(device):
    checkpoint_path = BASE_OUT_DIR_MODE / "models" / "best_fold1.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = WholeWellEnergyModel(
        int(checkpoint["sequence_channels"]),
        int(checkpoint["summary_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, checkpoint_path


def _mine_candidate_pool(base_model, record, templates_mode, basis_mode, device, seed):
    rng = np.random.default_rng(int(seed))
    initial = _initial_coefficients(
        templates_mode,
        basis_mode.shape[1],
        seed,
        random_count=24,
    )
    initial_energy = _score_coefficients(
        base_model, record, initial, basis_mode, device, mode="full"
    )
    current = initial[np.argsort(initial_energy)[:MINE_PARENTS]]
    mined = [initial]
    for scale in MINE_SCALES_FT:
        proposals = [current]
        for candidate in current:
            for _ in range(MINE_CHILDREN):
                proposals.append(
                    candidate[None]
                    + _smooth_noise(rng, basis_mode.shape[1], scale)[None]
                )
        proposals = np.clip(
            np.vstack(proposals), -COEFFICIENT_CAP_FT, COEFFICIENT_CAP_FT
        )
        proposals = _deduplicate_coefficients(proposals)
        proposal_energy = _score_coefficients(
            base_model, record, proposals, basis_mode, device, mode="full"
        )
        current = proposals[np.argsort(proposal_energy)[:MINE_PARENTS]]
        mined.append(proposals)

    # A fitted-truth teacher is present on only 20% of panels.  Making it
    # universal recreates the v0 train/inference menu mismatch.
    include_teacher = bool(int(seed) % 5 == 0)
    if include_teacher:
        teacher = np.asarray(record["truth_coefficients"], np.float32)[None]
        pool = _deduplicate_coefficients(np.vstack(mined + [teacher]))
    else:
        pool = _deduplicate_coefficients(np.vstack(mined))
    energy = _score_coefficients(
        base_model, record, pool, basis_mode, device, mode="full"
    )
    exact_rmse = _candidate_exact_rmse(record, pool, basis_mode)
    correction = _coefficients_to_column_correction(pool, basis_mode)

    selected = []

    def add(indices, limit):
        added = 0
        for candidate_index in indices:
            candidate_index = int(candidate_index)
            if candidate_index not in selected:
                selected.append(candidate_index)
                added += 1
            if added >= int(limit) or len(selected) >= MENU_CANDIDATES:
                break

    add(np.argsort(energy), MINE_TOP_ENERGY)
    hard_false = np.argsort(energy)
    hard_false = [
        index
        for index in hard_false
        if exact_rmse[index] >= float(np.min(exact_rmse)) + 4.0
    ]
    add(hard_false, MINE_HARD_FALSE)
    add(np.argsort(exact_rmse), MINE_TOP_ORACLE)

    # Fill the menu by farthest-point diversity in trajectory space.
    remaining = [index for index in range(len(pool)) if index not in selected]
    while remaining and len(selected) < MENU_CANDIDATES:
        if not selected:
            chosen = remaining[0]
        else:
            selected_paths = correction[np.asarray(selected)]
            candidate_paths = correction[np.asarray(remaining)]
            distance = np.sqrt(
                np.mean(
                    (candidate_paths[:, None, :] - selected_paths[None, :, :]) ** 2,
                    axis=2,
                )
            )
            chosen = remaining[int(np.argmax(distance.min(axis=1)))]
        selected.append(int(chosen))
        remaining.remove(int(chosen))

    selected = np.asarray(selected[:MENU_CANDIDATES], np.int64)
    coefficients = pool[selected].astype(np.float32)
    energy = energy[selected].astype(np.float32)
    exact_rmse = exact_rmse[selected].astype(np.float32)
    correction = correction[selected].astype(np.float32)
    labels = _mode_labels_from_correction(correction)
    return {
        "coefficients": coefficients,
        "base_energy": energy,
        "exact_rmse": exact_rmse,
        "mode_labels": labels,
        "pool_candidates": int(len(pool)),
        "hard_false_candidates": int(
            np.sum(exact_rmse >= float(np.min(exact_rmse)) + 4.0)
        ),
    }


def _build_mined_menus(base_model, records, templates_mode, basis_mode, device):
    rows = []
    started = time.time()
    for index, record in enumerate(
        tqdm(records, desc="mine inference-matched hard-negative menus")
    ):
        output = _mode_menu_path(record)
        if output.exists():
            with np.load(output, allow_pickle=False) as cached:
                if (
                    "version" in cached.files
                    and int(cached["version"][0]) == 1
                    and int(cached["coefficients"].shape[0]) == MENU_CANDIDATES
                ):
                    rows.append(
                        {
                            "well": record["well"],
                            "tag": record["tag"],
                            "status": "cached",
                            "candidates": int(cached["coefficients"].shape[0]),
                            "modes": int(len(np.unique(cached["mode_labels"]))),
                            "oracle_rmse": float(cached["exact_rmse"].min()),
                        }
                    )
                    continue
        menu = _mine_candidate_pool(
            base_model,
            record,
            templates_mode,
            basis_mode,
            device,
            seed=MODE_SEED + 10007 * index,
        )
        np.savez_compressed(
            output,
            version=np.asarray([1], np.int16),
            well=np.asarray([record["well"]], str),
            tag=np.asarray([record["tag"]], str),
            coefficients=menu["coefficients"].astype(np.float16),
            base_energy=menu["base_energy"].astype(np.float32),
            exact_rmse=menu["exact_rmse"].astype(np.float32),
            mode_labels=menu["mode_labels"].astype(np.int16),
        )
        rows.append(
            {
                "well": record["well"],
                "tag": record["tag"],
                "status": "built",
                "candidates": len(menu["coefficients"]),
                "modes": int(len(np.unique(menu["mode_labels"]))),
                "oracle_rmse": float(menu["exact_rmse"].min()),
                "pool_candidates": menu["pool_candidates"],
                "hard_false_candidates": menu["hard_false_candidates"],
            }
        )
        if (index + 1) % 250 == 0:
            print(
                "MENU MINING",
                {
                    "panels": index + 1,
                    "total": len(records),
                    "elapsed_minutes": (time.time() - started) / 60.0,
                },
                flush=True,
            )
    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR_MODE / "menu_build_report.csv", index=False)
    return report


class ModeMenuDataset(Dataset):
    def __init__(self, records, basis_mode):
        self.records = records
        self.basis = basis_mode
        weights = np.asarray(
            [record["panel_weight"] for record in records], np.float64
        )
        self.weight_scale = float(
            np.median(weights[weights > 0]) if np.any(weights > 0) else 1.0
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(_mode_menu_path(record), allow_pickle=False) as data:
            coefficients = data["coefficients"].astype(np.float32)
            exact_rmse = data["exact_rmse"].astype(np.float32)
            labels = data["mode_labels"].astype(np.int64)
        sequence, summary, correction = _candidate_features(
            record, coefficients, self.basis, mode="full"
        )
        weight = np.float32(
            np.clip(
                record["panel_weight"] / max(self.weight_scale, 1e-6),
                0.20,
                5.0,
            )
        )
        return {
            "sequence": torch.from_numpy(sequence),
            "summary": torch.from_numpy(summary),
            "correction": torch.from_numpy(correction.astype(np.float32)),
            "truth_correction": torch.from_numpy(
                record["eval_truth_residual"].astype(np.float32)
            ),
            "exact_rmse": torch.from_numpy(exact_rmse),
            "mode_labels": torch.from_numpy(labels),
            "panel_weight": torch.tensor(weight, dtype=torch.float32),
        }


def _single_panel_mode_loss(
    energy,
    exact_rmse,
    correction,
    truth_correction,
    mode_labels,
):
    energy = energy.float()
    exact_rmse = exact_rmse.float()
    correction = correction.float()
    truth_correction = truth_correction.float()
    labels = mode_labels.long()
    unique = torch.unique(labels, sorted=True)
    candidate_logits = -energy
    mode_logits = []
    mode_quality = []
    mode_members = []
    candidate_target = torch.zeros_like(energy)

    for mode in unique:
        member = torch.nonzero(labels == mode, as_tuple=False).squeeze(1)
        member_logits = candidate_logits[member]
        # log-mean-exp is invariant to duplicate candidates in a mode.
        mode_logits.append(
            torch.logsumexp(member_logits, dim=0)
            - math.log(max(int(member.numel()), 1))
        )
        quality = exact_rmse[member].amin()
        mode_quality.append(quality)
        mode_members.append(member)

    mode_logits = torch.stack(mode_logits)
    mode_quality = torch.stack(mode_quality)
    relative_mode_quality = mode_quality - mode_quality.amin()
    mode_target = torch.softmax(
        -relative_mode_quality / TARGET_MODE_TEMPERATURE_FT, dim=0
    )
    mode_nll = -(mode_target * F.log_softmax(mode_logits, dim=0)).sum()

    for mode_index, member in enumerate(mode_members):
        within_quality = exact_rmse[member] - exact_rmse[member].amin()
        within_target = torch.softmax(
            -within_quality / TARGET_MODE_TEMPERATURE_FT, dim=0
        )
        candidate_target[member] = mode_target[mode_index] * within_target

    count = torch.bincount(labels, minlength=int(unique.max()) + 1).float()
    adjusted_logits = candidate_logits - torch.log(count[labels].clamp_min(1.0))
    candidate_nll = -(
        candidate_target * F.log_softmax(adjusted_logits, dim=0)
    ).sum()
    probability = torch.softmax(adjusted_logits, dim=0)
    posterior = torch.sum(probability[:, None] * correction, dim=0)
    path_mse = torch.mean((posterior - truth_correction) ** 2) / 64.0

    best = torch.argmin(exact_rmse)
    relative = exact_rmse - exact_rmse[best]
    margin = torch.clamp(relative / 4.0, 0.0, 2.0)
    pairwise = F.relu(margin - (energy - energy[best]))
    pair_mask = torch.ones_like(pairwise)
    pair_mask[best] = 0.0
    pairwise = (pairwise * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    regression = F.smooth_l1_loss(
        energy, torch.log1p(exact_rmse), beta=0.25
    )
    total = (
        LOSS_MODE_NLL * mode_nll
        + LOSS_CANDIDATE_NLL * candidate_nll
        + LOSS_POSTERIOR_PATH * path_mse
        + LOSS_PAIRWISE * pairwise
        + LOSS_REGRESSION * regression
    )
    return total, {
        "mode_nll": mode_nll,
        "candidate_nll": candidate_nll,
        "path_mse": path_mse * 64.0,
        "pairwise": pairwise,
        "regression": regression,
        "modes": torch.tensor(float(len(unique)), device=energy.device),
    }


def _mode_batch_loss(
    energy,
    exact_rmse,
    correction,
    truth_correction,
    mode_labels,
    panel_weight,
):
    losses = []
    components = []
    for batch_index in range(len(energy)):
        loss, row = _single_panel_mode_loss(
            energy[batch_index],
            exact_rmse[batch_index],
            correction[batch_index],
            truth_correction[batch_index],
            mode_labels[batch_index],
        )
        losses.append(loss * panel_weight[batch_index].clamp(0.20, 5.0))
        components.append(row)
    total = torch.stack(losses).mean()
    summary = {
        name: torch.stack([row[name] for row in components]).mean()
        for name in components[0]
    }
    return total, summary


@torch.no_grad()
def _evaluate_mode_fixed(model, records, templates_mode, basis_mode, device):
    model.eval()
    rows = []
    posterior_errors = []
    map_errors = []
    oracle_sse = 0.0
    truth_target_sse = 0.0
    total_rows = 0
    correlations = []
    for index, record in enumerate(records):
        coefficients = _initial_coefficients(
            templates_mode,
            basis_mode.shape[1],
            MODE_SEED + 1709 * index,
            random_count=24,
        )
        energy = _score_coefficients(
            model, record, coefficients, basis_mode, device, mode="full"
        )
        correction = _coefficients_to_column_correction(coefficients, basis_mode)
        labels = _mode_labels_from_correction(correction)
        probability = _multiplicity_corrected_probability(energy, labels)
        candidate_rows = _candidate_row_predictions(record, coefficients, basis_mode)
        exact_rmse = np.sqrt(
            np.mean(
                (candidate_rows - record["eval_truth"][None]) ** 2, axis=1
            )
        )
        posterior = probability @ candidate_rows
        map_prediction = candidate_rows[int(np.argmin(energy))]
        truth_probability = _mode_softmax(
            -(exact_rmse - exact_rmse.min()) / TARGET_MODE_TEMPERATURE_FT
        )
        truth_posterior = truth_probability @ candidate_rows
        truth = np.asarray(record["eval_truth"], np.float64)
        posterior_error = posterior - truth
        map_error = map_prediction - truth
        posterior_errors.append(posterior_error)
        map_errors.append(map_error)
        oracle_sse += float(np.min(exact_rmse) ** 2 * len(truth))
        truth_target_sse += float(np.sum((truth_posterior - truth) ** 2))
        total_rows += len(truth)
        correlation = spearmanr(energy, exact_rmse).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        rows.append(
            {
                "well": record["well"],
                "rows": len(truth),
                "posterior_rmse": float(np.sqrt(np.mean(posterior_error**2))),
                "map_rmse": float(np.sqrt(np.mean(map_error**2))),
                "oracle_rmse": float(np.min(exact_rmse)),
                "truth_target_rmse": float(
                    np.sqrt(np.mean((truth_posterior - truth) ** 2))
                ),
                "modes": int(len(np.unique(labels))),
                "rank_spearman": float(correlation) if np.isfinite(correlation) else np.nan,
            }
        )
    bywell = pd.DataFrame(rows)
    posterior_error = np.concatenate(posterior_errors)
    map_error = np.concatenate(map_errors)
    metrics = {
        "posterior_rmse": float(np.sqrt(np.mean(posterior_error**2))),
        "map_rmse": float(np.sqrt(np.mean(map_error**2))),
        "oracle_rmse": float(np.sqrt(oracle_sse / max(total_rows, 1))),
        "truth_target_rmse": float(
            np.sqrt(truth_target_sse / max(total_rows, 1))
        ),
        "median_well_rmse": float(bywell["posterior_rmse"].median()),
        "p75_well_rmse": float(bywell["posterior_rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["posterior_rmse"].quantile(0.95)),
        "mean_modes": float(bywell["modes"].mean()),
        "mean_rank_spearman": float(np.mean(correlations)) if correlations else np.nan,
    }
    metrics.update(_well_tail_metrics(bywell, "posterior_rmse"))
    return metrics, bywell


def _train_mode_model(
    base_model,
    train_records_mode,
    valid_records_mode,
    templates_mode,
    basis_mode,
    device,
):
    model = WholeWellEnergyModel(
        base_model.sequence[0].in_channels,
        base_model.summary[0].in_features,
    ).to(device)
    model.load_state_dict(base_model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODE_LEARNING_RATE,
        weight_decay=MODE_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MODE_EPOCHS, eta_min=MODE_MIN_LEARNING_RATE
    )
    scaler = _make_scaler(device)
    dataset = ModeMenuDataset(train_records_mode, basis_mode)
    loader = DataLoader(
        dataset,
        batch_size=MODE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    checkpoint_path = MODEL_DIR_MODE / "best_fold1.pt"
    best_rmse = np.inf
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, MODE_EPOCHS + 1):
        model.train()
        loss_rows = []
        epoch_started = time.time()
        for batch_index, batch in enumerate(loader, start=1):
            sequence = batch["sequence"].to(device, non_blocking=True)
            summary = batch["summary"].to(device, non_blocking=True)
            correction = batch["correction"].to(device, non_blocking=True)
            truth_correction = batch["truth_correction"].to(
                device, non_blocking=True
            )
            exact_rmse = batch["exact_rmse"].to(device, non_blocking=True)
            mode_labels = batch["mode_labels"].to(device, non_blocking=True)
            panel_weight = batch["panel_weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                energy = model(sequence, summary)
            loss, parts = _mode_batch_loss(
                energy.float(),
                exact_rmse,
                correction,
                truth_correction,
                mode_labels,
                panel_weight,
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
                        for name, value in parts.items()
                    },
                }
            )
            if batch_index % 500 == 0 or batch_index == len(loader):
                print(
                    "MODE TRAIN",
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(loader),
                        "loss": float(
                            np.mean([row["loss"] for row in loss_rows[-500:]])
                        ),
                        "epoch_minutes": (time.time() - epoch_started) / 60.0,
                    },
                    flush=True,
                )
        scheduler.step()
        metrics, bywell = _evaluate_mode_fixed(
            model, valid_records_mode, templates_mode, basis_mode, device
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
        print("MODE EPOCH", row, flush=True)
        if metrics["posterior_rmse"] < best_rmse - 1e-4:
            best_rmse = float(metrics["posterior_rmse"])
            stale = 0
            torch.save(
                {
                    "model": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "metrics": metrics,
                    "sequence_channels": base_model.sequence[0].in_channels,
                    "summary_channels": base_model.summary[0].in_features,
                    "mode_radius_ft": MODE_RADIUS_FT,
                    "target_temperature_ft": TARGET_MODE_TEMPERATURE_FT,
                },
                checkpoint_path,
            )
            bywell.to_csv(OUT_DIR_MODE / "best_fixed_bywell.csv", index=False)
        else:
            stale += 1
        if epoch >= 4 and stale >= MODE_PATIENCE:
            print(f"mode-NLL early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(
        OUT_DIR_MODE / "training_history.csv", index=False
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, bywell = _evaluate_mode_fixed(
        model, valid_records_mode, templates_mode, basis_mode, device
    )
    bywell.to_csv(OUT_DIR_MODE / "fixed_bywell.csv", index=False)
    return model, checkpoint, metrics, (time.time() - started) / 60.0


@torch.no_grad()
def _evaluate_mode_search(
    model, records, templates_mode, basis_mode, device
):
    rows = []
    posterior_errors = []
    map_errors = []
    oracle_sse = 0.0
    truth_target_sse = 0.0
    total_rows = 0
    for index, record in enumerate(
        tqdm(records, desc="mode-NLL deployment search")
    ):
        raw = _search_record(
            model,
            record,
            templates_mode,
            basis_mode,
            device,
            seed=MODE_SEED + 10007 * index,
            mode="full",
        )
        coefficients = raw["coefficients"]
        energy = raw["energies"]
        correction = _coefficients_to_column_correction(coefficients, basis_mode)
        labels = _mode_labels_from_correction(correction)
        probability = _multiplicity_corrected_probability(energy, labels)
        candidate_rows = _candidate_row_predictions(record, coefficients, basis_mode)
        truth = np.asarray(record["eval_truth"], np.float64)
        exact_rmse = np.sqrt(
            np.mean((candidate_rows - truth[None]) ** 2, axis=1)
        )
        posterior = probability @ candidate_rows
        map_prediction = candidate_rows[int(np.argmin(energy))]
        truth_probability = _mode_softmax(
            -(exact_rmse - exact_rmse.min()) / TARGET_MODE_TEMPERATURE_FT
        )
        truth_posterior = truth_probability @ candidate_rows
        posterior_error = posterior - truth
        map_error = map_prediction - truth
        posterior_errors.append(posterior_error)
        map_errors.append(map_error)
        oracle_sse += float(np.min(exact_rmse) ** 2 * len(truth))
        truth_target_sse += float(np.sum((truth_posterior - truth) ** 2))
        total_rows += len(truth)
        rows.append(
            {
                "well": record["well"],
                "rows": len(truth),
                "posterior_rmse": float(np.sqrt(np.mean(posterior_error**2))),
                "map_rmse": float(np.sqrt(np.mean(map_error**2))),
                "oracle_rmse": float(np.min(exact_rmse)),
                "truth_target_rmse": float(
                    np.sqrt(np.mean((truth_posterior - truth) ** 2))
                ),
                "modes": int(len(np.unique(labels))),
                "candidates": int(len(coefficients)),
            }
        )
    bywell = pd.DataFrame(rows)
    posterior_error = np.concatenate(posterior_errors)
    map_error = np.concatenate(map_errors)
    metrics = {
        "posterior_rmse": float(np.sqrt(np.mean(posterior_error**2))),
        "map_rmse": float(np.sqrt(np.mean(map_error**2))),
        "oracle_rmse": float(np.sqrt(oracle_sse / max(total_rows, 1))),
        "truth_target_rmse": float(
            np.sqrt(truth_target_sse / max(total_rows, 1))
        ),
        "median_well_rmse": float(bywell["posterior_rmse"].median()),
        "p75_well_rmse": float(bywell["posterior_rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["posterior_rmse"].quantile(0.95)),
        "mean_modes": float(bywell["modes"].mean()),
        "mean_candidates": float(bywell["candidates"].mean()),
    }
    metrics.update(_well_tail_metrics(bywell, "posterior_rmse"))
    return metrics, bywell


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


device_mode = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device_mode.type == "cuda":
    torch.set_float32_matmul_precision("high")

basis_mode = _spline_basis(256, SPLINE_CONTROLS)
templates_mode = np.load(
    BASE_OUT_DIR_MODE / "foldclean_spline_templates.npy"
).astype(np.float32)
all_wells_mode = sorted(
    set(train_manifest["well"].astype(str))
    | set(valid_manifest["well"].astype(str))
)
references_mode = ReferenceStore(all_wells_mode)
train_records_mode = [
    _prepare_record(
        _record_from_panel(row.path, row.fold), references_mode, basis_mode
    )
    for row in tqdm(
        train_manifest.itertuples(index=False),
        total=len(train_manifest),
        desc="reload mode-NLL train panels",
    )
]
valid_records_mode = [
    _prepare_record(
        _record_from_panel(row.path, row.fold), references_mode, basis_mode
    )
    for row in tqdm(
        valid_manifest.itertuples(index=False),
        total=len(valid_manifest),
        desc="reload mode-NLL valid panels",
    )
]
base_model_mode, base_checkpoint_mode, base_checkpoint_path_mode = (
    _load_base_model_mode(device_mode)
)

print(
    {
        "experiment": "fullwell_mode_nll_hardneg_v1",
        "base_checkpoint": str(base_checkpoint_path_mode),
        "base_epoch": int(base_checkpoint_mode["epoch"]),
        "train_panels": len(train_records_mode),
        "valid_wells": len(valid_records_mode),
        "menu_candidates": MENU_CANDIDATES,
        "mode_radius_ft": MODE_RADIUS_FT,
        "target_temperature_ft": TARGET_MODE_TEMPERATURE_FT,
        "device": str(device_mode),
    },
    flush=True,
)

menu_report_mode = _build_mined_menus(
    base_model_mode,
    train_records_mode,
    templates_mode,
    basis_mode,
    device_mode,
)
print("\nMINED MENU SUMMARY")
print(
    menu_report_mode[["candidates", "modes", "oracle_rmse"]]
    .describe(percentiles=[0.1, 0.5, 0.9])
    .to_string()
)

base_fixed_metrics_mode, _ = _evaluate_mode_fixed(
    base_model_mode,
    valid_records_mode,
    templates_mode,
    basis_mode,
    device_mode,
)
model_mode, checkpoint_mode, fixed_metrics_mode, training_minutes_mode = (
    _train_mode_model(
        base_model_mode,
        train_records_mode,
        valid_records_mode,
        templates_mode,
        basis_mode,
        device_mode,
    )
)
search_metrics_mode, search_bywell_mode = _evaluate_mode_search(
    model_mode,
    valid_records_mode,
    templates_mode,
    basis_mode,
    device_mode,
)
search_bywell_mode.to_csv(OUT_DIR_MODE / "search_bywell.csv", index=False)

architecture_pass_mode = bool(
    search_metrics_mode["posterior_rmse"] <= 7.0
    and search_metrics_mode["oracle_rmse"] <= 5.5
)
target_pass_mode = bool(search_metrics_mode["posterior_rmse"] <= 5.5)
improvement_mode = float(
    base_fixed_metrics_mode["posterior_rmse"]
    - fixed_metrics_mode["posterior_rmse"]
)
strict_pass_mode = bool(architecture_pass_mode and improvement_mode >= 0.30)
verdict_mode = {
    "experiment": "fullwell_mode_nll_hardneg_v1",
    "standalone": True,
    "base_checkpoint": str(base_checkpoint_path_mode),
    "base_epoch": int(base_checkpoint_mode["epoch"]),
    "mode_checkpoint_epoch": int(checkpoint_mode["epoch"]),
    "train_panels": len(train_records_mode),
    "valid_wells": len(valid_records_mode),
    "menu": {
        "candidates": MENU_CANDIDATES,
        "mode_radius_ft": MODE_RADIUS_FT,
        "target_temperature_ft": TARGET_MODE_TEMPERATURE_FT,
        "mean_train_modes": float(menu_report_mode["modes"].mean()),
        "median_train_oracle_rmse": float(menu_report_mode["oracle_rmse"].median()),
    },
    "base_fixed": base_fixed_metrics_mode,
    "mode_fixed": fixed_metrics_mode,
    "fixed_improvement": improvement_mode,
    "mode_search": search_metrics_mode,
    "training_minutes": training_minutes_mode,
    "architecture_pass": architecture_pass_mode,
    "target_5p5_pass": target_pass_mode,
    "strict_pass": strict_pass_mode,
    "next": (
        "TARGET PASS -> freeze recipe and confirm on a new outer split"
        if target_pass_mode
        else (
            "PASS -> freeze recipe and run new split/nested OOF confirmation"
            if strict_pass_mode
            else "FAIL -> mode weighting does not generalise; deliver final PDF/STOP"
        )
    ),
}
with open(OUT_DIR_MODE / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict_mode, handle, indent=2)

print("\nBASE FIXED MENU")
print(json.dumps(base_fixed_metrics_mode, indent=2))
print("\nMODE-NLL FIXED MENU")
print(json.dumps(fixed_metrics_mode, indent=2))
print("\nMODE-NLL DEPLOYMENT SEARCH")
print(json.dumps(search_metrics_mode, indent=2))
print("\nFINAL VERDICT")
print(json.dumps(verdict_mode, indent=2))
print("saved:", OUT_DIR_MODE)

del train_records_mode, valid_records_mode, references_mode
gc.collect()
if device_mode.type == "cuda":
    torch.cuda.empty_cache()
