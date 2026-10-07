"""Pretrain a GR candidate matcher on 50k deterministic synthetic samples.

Run in the same kernel after ``synthetic_gr_generator_calibration_v1_cell.py``.
The selected orthogonal residual-transfer generator is frozen.  Folds 3-5
provide synthetic source segments, real fold 2 selects the checkpoint, and
untouched real fold 1 is evaluated exactly once at the end.

The 50k bank is virtual: index + seed deterministically identify target,
donor, mask, and centre bias.  Only the compact matcher checkpoint is saved.
"""

from pathlib import Path
import copy
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REQUIRED_GLOBALS = (
    "CALIBRATION_DIR",
    "calibration_source_segments",
    "calibration_real_segments",
    "valid_segments",
    "_transfer_residual",
    "POINTS",
    "RANDOM_STATE",
)
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run synthetic_gr_generator_calibration_v1_cell.py first; missing: "
        + repr(missing)
    )

PRETRAIN_DIR = Path(CALIBRATION_DIR).parent / "synthetic_gr_matcher_pretrain_v0"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_SAMPLES = 50_000
CANDIDATE_OFFSETS = np.arange(-64.0, 64.01, 4.0, dtype=np.float32)
CENTRE_BIASES = np.arange(-32.0, 32.01, 4.0, dtype=np.float32)
SOFT_LABEL_SIGMA = 4.0
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
REPORT_EVERY_STEPS = 250
EARLY_STOPPING_REPORTS = 5
PRETRAIN_SEED = 20260718


def _candidate_curves(sample, centre_bias):
    candidate_tvt = (
        np.asarray(sample["truth_tvt"], np.float64)[None, :]
        + float(centre_bias)
        + CANDIDATE_OFFSETS[:, None]
    )
    tw_tvt = np.asarray(sample["tw_tvt"], np.float64)
    tw_gr = np.asarray(sample["tw_gr"], np.float64)
    flat = np.interp(
        candidate_tvt.ravel(),
        tw_tvt,
        tw_gr,
        left=np.nan,
        right=np.nan,
    )
    curves = flat.reshape(len(CANDIDATE_OFFSETS), POINTS)
    valid = np.isfinite(curves)
    curves = np.nan_to_num(curves, nan=0.0).astype(np.float32)
    label_value = -float(centre_bias)
    label_index = int(np.argmin(np.abs(CANDIDATE_OFFSETS - label_value)))
    return curves, valid.astype(np.float32), label_index, label_value


class VirtualSyntheticMatcherDataset(Dataset):
    def __init__(self, source_segments, length=SYNTHETIC_SAMPLES):
        self.source_segments = source_segments
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(PRETRAIN_SEED + int(index) * 104729)
        target = self.source_segments[int(rng.integers(0, len(self.source_segments)))]
        donor = self.source_segments[int(rng.integers(0, len(self.source_segments)))]
        synthetic = _transfer_residual(target, donor, 1.0, "orthogonal")
        bias = float(CENTRE_BIASES[int(rng.integers(0, len(CENTRE_BIASES)))])
        candidates, candidate_valid, label_index, label_value = _candidate_curves(
            synthetic, bias
        )
        horizontal = np.stack(
            [
                np.asarray(synthetic["horizontal"], np.float32),
                np.asarray(synthetic["horizontal_valid"], np.float32),
            ],
            axis=0,
        )
        candidate = np.stack([candidates, candidate_valid], axis=1)
        return {
            "horizontal": torch.from_numpy(horizontal),
            "candidate": torch.from_numpy(candidate),
            "label_index": torch.tensor(label_index, dtype=torch.long),
            "label_value": torch.tensor(label_value, dtype=torch.float32),
        }


class RealMatcherDataset(Dataset):
    def __init__(self, segments, bias_seed):
        self.segments = segments
        self.bias_seed = int(bias_seed)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        sample = self.segments[index]
        bias_index = (int(index) * 7 + self.bias_seed) % len(CENTRE_BIASES)
        bias = float(CENTRE_BIASES[bias_index])
        candidates, candidate_valid, label_index, label_value = _candidate_curves(
            sample, bias
        )
        horizontal = np.stack(
            [
                np.asarray(sample["horizontal"], np.float32),
                np.asarray(sample["horizontal_valid"], np.float32),
            ],
            axis=0,
        )
        candidate = np.stack([candidates, candidate_valid], axis=1)
        return {
            "horizontal": torch.from_numpy(horizontal),
            "candidate": torch.from_numpy(candidate),
            "label_index": torch.tensor(label_index, dtype=torch.long),
            "label_value": torch.tensor(label_value, dtype=torch.float32),
        }


def _raw_ncc(horizontal, candidate):
    # horizontal [B,2,L], candidate [B,K,2,L]
    horizontal_value = horizontal[:, 0, :][:, None, :]
    horizontal_valid = horizontal[:, 1, :][:, None, :]
    candidate_value = candidate[:, :, 0, :]
    candidate_valid = candidate[:, :, 1, :]
    valid = horizontal_valid * candidate_valid
    count = valid.sum(dim=-1).clamp_min(1.0)
    horizontal_mean = (horizontal_value * valid).sum(dim=-1) / count
    candidate_mean = (candidate_value * valid).sum(dim=-1) / count
    left = (horizontal_value - horizontal_mean[:, :, None]) * valid
    right = (candidate_value - candidate_mean[:, :, None]) * valid
    numerator = (left * right).sum(dim=-1)
    denominator = torch.sqrt(
        (left.square().sum(dim=-1) * right.square().sum(dim=-1)).clamp_min(1e-8)
    )
    score = numerator / denominator
    score = score.masked_fill(count < 12.0, -1.0)
    return score


class PairRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 32, 7, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv1d(48, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(65, 48),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(48, 1),
        )

    def forward(self, horizontal, candidate):
        batch, candidates, _, length = candidate.shape
        horizontal_value = horizontal[:, 0, :][:, None, :].expand(
            batch, candidates, length
        )
        horizontal_valid = horizontal[:, 1, :][:, None, :].expand(
            batch, candidates, length
        )
        candidate_value = candidate[:, :, 0, :]
        candidate_valid = candidate[:, :, 1, :]
        values = torch.stack(
            [
                horizontal_value,
                horizontal_valid,
                candidate_value,
                candidate_valid,
                horizontal_value * candidate_value,
                -torch.abs(horizontal_value - candidate_value),
            ],
            dim=2,
        ).reshape(batch * candidates, 6, length)
        encoded = self.encoder(values).squeeze(-1)
        ncc = _raw_ncc(horizontal, candidate).reshape(batch * candidates, 1)
        score = self.head(torch.cat([encoded, ncc], dim=1))
        return score.reshape(batch, candidates)


def _soft_cross_entropy(logits, label_value, offsets):
    distance = offsets[None, :] - label_value[:, None]
    target = torch.softmax(-0.5 * (distance / SOFT_LABEL_SIGMA) ** 2, dim=1)
    return -(target * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


@torch.no_grad()
def _evaluate_matcher(model, dataset, device, shuffled=False):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    learned_errors = []
    ncc_errors = []
    losses = []
    offsets = torch.from_numpy(CANDIDATE_OFFSETS).to(device)
    for batch in loader:
        horizontal = batch["horizontal"].to(device)
        candidate = batch["candidate"].to(device)
        label_value = batch["label_value"].to(device)
        if shuffled and len(candidate) > 1:
            candidate = torch.roll(candidate, shifts=1, dims=0)
        logits = model(horizontal, candidate)
        losses.append(float(_soft_cross_entropy(logits, label_value, offsets).cpu()))
        learned_value = offsets[torch.argmax(logits, dim=1)]
        ncc_value = offsets[torch.argmax(_raw_ncc(horizontal, candidate), dim=1)]
        learned_errors.append(torch.abs(learned_value - label_value).cpu().numpy())
        ncc_errors.append(torch.abs(ncc_value - label_value).cpu().numpy())
    learned = np.concatenate(learned_errors)
    ncc = np.concatenate(ncc_errors)
    return {
        "samples": len(learned),
        "loss": float(np.mean(losses)),
        "learned_within4": float(np.mean(learned <= 4.0)),
        "learned_within8": float(np.mean(learned <= 8.0)),
        "learned_median_abs": float(np.median(learned)),
        "learned_p90_abs": float(np.quantile(learned, 0.90)),
        "ncc_within4": float(np.mean(ncc <= 4.0)),
        "ncc_within8": float(np.mean(ncc <= 8.0)),
        "ncc_median_abs": float(np.median(ncc)),
        "within4_gain": float(np.mean(learned <= 4.0) - np.mean(ncc <= 4.0)),
        "within8_gain": float(np.mean(learned <= 8.0) - np.mean(ncc <= 8.0)),
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
torch.manual_seed(PRETRAIN_SEED)
np.random.seed(PRETRAIN_SEED)

source_segments = list(calibration_source_segments)
selection_segments = list(calibration_real_segments)
confirmation_segments = list(valid_segments)
train_dataset = VirtualSyntheticMatcherDataset(source_segments)
selection_dataset = RealMatcherDataset(selection_segments, bias_seed=17)
confirmation_dataset = RealMatcherDataset(confirmation_segments, bias_seed=31)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

model = PairRanker().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
total_steps = len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=MIN_LEARNING_RATE
)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
offsets_tensor = torch.from_numpy(CANDIDATE_OFFSETS).to(device)

print(
    {
        "experiment": "synthetic_gr_matcher_pretrain_v0",
        "device": str(device),
        "synthetic_samples": SYNTHETIC_SAMPLES,
        "source_folds": [3, 4, 5],
        "selection_fold": 2,
        "confirmation_fold": 1,
        "source_segments": len(source_segments),
        "selection_segments": len(selection_segments),
        "confirmation_segments": len(confirmation_segments),
        "candidates": len(CANDIDATE_OFFSETS),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    },
    flush=True,
)

baseline_selection = _evaluate_matcher(model, selection_dataset, device)
print("UNTRAINED SELECTION", baseline_selection, flush=True)
best_score = -np.inf
best_step = 0
best_state = copy.deepcopy(model.state_dict())
stale_reports = 0
history = []
started = time.time()
model.train()
for step, batch in enumerate(train_loader, 1):
    horizontal = batch["horizontal"].to(device, non_blocking=True)
    candidate = batch["candidate"].to(device, non_blocking=True)
    label_value = batch["label_value"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        logits = model(horizontal, candidate)
        loss = _soft_cross_entropy(logits, label_value, offsets_tensor)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    should_report = (
        step == 1
        or step % REPORT_EVERY_STEPS == 0
        or step == total_steps
    )
    if not should_report:
        continue
    metrics = _evaluate_matcher(model, selection_dataset, device)
    score = (
        metrics["learned_within8"]
        + 0.5 * metrics["learned_within4"]
        - 0.005 * metrics["learned_median_abs"]
    )
    row = {
        "step": step,
        "train_loss": float(loss.detach().cpu()),
        **metrics,
        "selection_score": float(score),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    history.append(row)
    print("PRETRAIN", row, flush=True)
    if score > best_score + 1e-5:
        best_score = float(score)
        best_step = step
        best_state = copy.deepcopy(model.state_dict())
        stale_reports = 0
    else:
        stale_reports += 1
    if step >= 750 and stale_reports >= EARLY_STOPPING_REPORTS:
        print(f"synthetic pretraining early stopping at step {step}", flush=True)
        break
    model.train()

pd.DataFrame(history).to_csv(PRETRAIN_DIR / "selection_history.csv", index=False)
model.load_state_dict(best_state)
model.eval()
selection_metrics = _evaluate_matcher(model, selection_dataset, device)
confirmation_metrics = _evaluate_matcher(model, confirmation_dataset, device)
shuffled_metrics = _evaluate_matcher(
    model, confirmation_dataset, device, shuffled=True
)
torch.save(
    {
        "model": best_state,
        "best_step": best_step,
        "candidate_offsets": CANDIDATE_OFFSETS,
        "selection_metrics": selection_metrics,
        "confirmation_metrics": confirmation_metrics,
        "generator": {"mode": "orthogonal", "residual_scale": 1.0},
        "synthetic_samples": SYNTHETIC_SAMPLES,
    },
    PRETRAIN_DIR / "matcher.pt",
)

shuffled_degradation = float(
    confirmation_metrics["learned_within8"] - shuffled_metrics["learned_within8"]
)
strict_pass = bool(
    confirmation_metrics["learned_within4"] >= 0.35
    and confirmation_metrics["learned_within8"] >= 0.60
    and confirmation_metrics["within8_gain"] >= 0.15
    and confirmation_metrics["learned_median_abs"] <= 8.0
    and shuffled_degradation >= 0.15
)
verdict = {
    "experiment": "synthetic_gr_matcher_pretrain_v0",
    "synthetic_samples": SYNTHETIC_SAMPLES,
    "best_step": best_step,
    "selection_fold_metrics": selection_metrics,
    "confirmation_fold_metrics": confirmation_metrics,
    "shuffled_typewell_metrics": shuffled_metrics,
    "shuffled_within8_degradation": shuffled_degradation,
    "checkpoint": str(PRETRAIN_DIR / "matcher.pt"),
    "runtime_minutes": (time.time() - started) / 60.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> use the frozen matcher to build a PF-centred full-path cost volume"
        if strict_pass
        else "FAIL -> synthetic pretraining does not transfer; stop the synthetic matcher route"
    ),
}
with open(PRETRAIN_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nSELECTION", selection_metrics)
print("\nCONFIRMATION", confirmation_metrics)
print("\nSHUFFLED TYPEWELL", shuffled_metrics)
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", PRETRAIN_DIR)

del optimizer, scheduler, scaler, train_loader, train_dataset
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

